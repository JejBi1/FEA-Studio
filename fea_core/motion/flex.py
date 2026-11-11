"""Flexible bodies of the mechanism – Craig–Bampton (specification F8, 5a.3 D6, variant 1; stage F8.8).

Main process (this module, may use OCP and Gmsh):
  prepare(study, parts, body, jd) – mesh of the body (Gmsh, tet10) with the faces of the joints of the body as the interface, check of the
    material (only isotropic – printed and orthotropic parts are NOT SUPPORTED, 5a.7 c), the files for the worker in SI:
    flex<i>/mesh.npz (nodes in m), materials.json (E Pa, ν, ρ kg/m³ by the parts of the mesh), interfaces.json (nodes and weights
    HRZ of the faces of the interface by joints).
  finish(res, infos) – after the calculation: shapes Φ (nodes × 3 × shapes, mm per unit of the modal coordinate), stress shapes
    (fea_core.recovery from Φ – as ccx C3D10) and the modal coordinates q(t) into MotionResult (fields flex<i>_*).
  displacement / positions / stress / stress_history – the deformation and the stress of the flexible body in time (axes of the body).
Working process (exudyn_adapter.Model._flex_body): matrices M, K from NGSolve on OUR mesh tet10 in the nodal basis
(fea_core.ngs.worker: build, transform, node_to_ngs – as the NGSolve backend), Exudyn FEMinterface →
ComputeHurtyCraigBamptonModes (RBE2 / RBE3 on the interfaces) → CreateFFRFReducedOrderObject (ObjectFFRFreducedOrder,
markers MarkerSuperElementRigid at the joints). The first interface determines the floating frame (excludeRigidBodyMotion).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np


class FlexError(ValueError):
    """The flexible body cannot be prepared (material, a missing face of a joint, an overlap of the interfaces…)."""


def prepare(study, parts, body, bi: int, jd: Path, log=None) -> dict:
    """Mesh and files of the flexible body for the worker; returns a record into the job (+ "_mesh" = mesh in mm for the recomputation)."""
    from ..mesh import generate_mesh
    from ..springs import node_weights
    mech = study.mechanism
    by_name = {p.name: p for p in parts}
    plist = [by_name[n] for n in body.parts if n in by_name]
    if not plist:
        raise FlexError(f"Flexible body '{body.name}' has no parts in the geometry")
    for n in body.parts:
        m = study.material_of(n)
        if m.is_print or m.is_ortho:
            raise FlexError(f"Flexible body '{body.name}': part '{n}' has a {('printed' if m.is_print else 'orthotropic')} material – flexible bodies support only isotropic materials (E, ν, ρ); use a rigid body or an isotropic substitute")
        if not (m.E and m.rho):
            raise FlexError(f"Flexible body '{body.name}': material '{m.name}' needs E and density ρ")
    ifaces = []
    for k, j in enumerate(mech.joints):
        for side, bn in (("b", j.body_b), ("a", j.body_a)):
            if bn != body.name:
                continue
            if j.kind in ("gear", "rack_pinion", "distance"):
                raise FlexError(f"Joint '{j.name}' ({j.kind}) on flexible body '{body.name}' is not supported yet – only joints with a face (revolute, prismatic, spherical, fixed…)")
            sel = j.selection_b if side == "b" else j.selection_a
            if not sel or sel not in study.selections or not study.selections[sel].faces:
                raise FlexError(f"Flexible body '{body.name}': joint '{j.name}' has no face on this body (selection_{side}) – the Craig–Bampton interfaces are the joint faces")
            ifaces.append({"joint": k, "side": side, "selection": sel})
    if not ifaces:
        raise FlexError(f"Flexible body '{body.name}' has no joint (interface)")
    for f in mech.forces:
        if body.name in (f.body_a, f.body_b) and f.kind != "torsion":
            raise FlexError(f"Force element '{f.name}' on flexible body '{body.name}' is not supported yet")
    for ld in mech.loads:
        if ld.body == body.name:
            raise FlexError(f"External load '{ld.name}' on flexible body '{body.name}' is not supported yet")
    settings = asdict(study.mesh)
    if body.mesh_size:
        settings["size"] = float(body.mesh_size)
    sels = {}
    for it in ifaces:
        sels[it["selection"]] = [f.to_dict() for f in study.selections[it["selection"]].faces]
    fd = Path(jd) / f"flex{bi}"
    fd.mkdir(parents=True, exist_ok=True)
    if log:
        log(f"Mesh of flexible body '{body.name}' (Gmsh)")
    mesh = generate_mesh(plist, sels, settings, fd / "gmsh")
    if mesh.tets.shape[1] != 10:
        raise FlexError("A flexible body needs quadratic tet10 elements (MeshSettings.order = 2)")
    used = set()
    for it in ifaces:
        nodes, w = node_weights(mesh, it["selection"])
        if used & set(nodes.tolist()):
            raise FlexError(f"Flexible body '{body.name}': the joint interface faces overlap (shared nodes) – Craig–Bampton needs separate interfaces")
        used |= set(nodes.tolist())
        it["nodes"] = nodes.tolist()
        it["weights"] = (w / w.sum()).tolist()
    from ..mesh import Mesh
    Mesh(mesh.nodes * 1e-3, mesh.tets, mesh.tet_part, mesh.part_names).save(fd / "mesh.npz")
    mats = []
    for pn in mesh.part_names:
        m = study.material_of(pn)
        mats.append({"E": m.E * 1e6, "nu": m.nu, "rho": m.rho * 1e12})
    (fd / "materials.json").write_text(json.dumps(mats), encoding="utf-8")
    (fd / "interfaces.json").write_text(json.dumps(ifaces), encoding="utf-8")
    if body.modes < 0 or body.modes > 200:
        raise FlexError("Number of flexible-body modes 0–200")
    return {"flex_dir": str(fd), "modes": int(body.modes), "interface": body.interface, "zeta": float(body.zeta),
            "interfaces": [{"joint": it["joint"], "side": it["side"]} for it in ifaces], "_mesh": mesh, "mats": mats}


def finish(arrays: dict, meta: dict, raw: dict, infos: dict) -> None:
    """Shapes, stress shapes and q(t) of the flexible bodies into the fields of the result (units of the project)."""
    from ..recovery import recover
    from ..solve_large import iso_stiffness
    meta["flex"] = []
    for bi, info in infos.items():
        fd = Path(info["flex_dir"])
        with np.load(fd / "modes.npz") as z:
            phi = z["phi"]                                   # (N, 3, nm) m per unit q, axes of the body
            freq = z["freq"]
            ref = z["ref"]                                   # origin of the floating frame [m] (position of the CAD)
        mesh = info["_mesh"]
        C = np.array([iso_stiffness(*_Enu(info["mats"][p])) for p in range(len(mesh.part_names))])[mesh.tet_part]
        nm = phi.shape[2]
        smodes = np.zeros((len(mesh.nodes), 6, nm), np.float32)
        for k in range(nm):
            S, _ = recover(None, mesh, phi[:, :, k] * 1e3, C=C)          # MPa na jednotku q (posuvy v mm)
            smodes[:, :, k] = S
        key = f"flex{bi}"
        arrays[key + "_nodes"] = mesh.nodes
        arrays[key + "_tets"] = mesh.tets.astype(np.int32)
        arrays[key + "_phi"] = (phi * 1e3).astype(np.float32)
        arrays[key + "_smodes"] = smodes
        arrays[key + "_q"] = raw.get(f"flexq{bi}", np.zeros((len(raw["t"]), nm)))
        meta["bodies"][bi]["com"] = [float(x) * 1e3 for x in ref]       # MotionResult.pose: origin of the frame
        meta["flex"].append({"body": meta["bodies"][bi]["name"], "index": int(bi), "modes": int(nm),
                             "freq_hz": [float(x) for x in freq], "nodes": int(len(mesh.nodes)),
                             "elements": int(len(mesh.tets)), "ref_mm": [float(x) * 1e3 for x in ref]})


def _Enu(m):
    return m["E"] * 1e-6, m["nu"]


def _key(res, body) -> tuple[str, int]:
    for f in res.meta.get("flex", []):
        if f["body"] == body:
            return f"flex{f['index']}", f["index"]
    raise KeyError(f"Body '{body}' is not flexible")


def displacement(res, body: str, k: int) -> np.ndarray:
    """Deformation (N, 3) [mm] in the axes of the body (without the rigid motion) at the time k."""
    key, _ = _key(res, body)
    return res.arrays[key + "_phi"] @ res.arrays[key + "_q"][k]


def positions(res, body: str, k: int) -> np.ndarray:
    """Positions of the nodes of the mesh [mm] in global axes at the time k (rigid motion of the floating frame + deformation)."""
    key, bi = _key(res, body)
    R, T = res.pose(bi, k)
    X = res.arrays[key + "_nodes"]
    return (X + displacement(res, body, k)) @ R.T + T


def stress(res, body: str, k: int) -> np.ndarray:
    """Stress (N, 6) [MPa] in the axes of the body (order of ccx xx yy zz xy yz zx) at the time k."""
    key, _ = _key(res, body)
    return res.arrays[key + "_smodes"] @ res.arrays[key + "_q"][k]


def von_mises(S: np.ndarray) -> np.ndarray:
    sx, sy, sz, txy, tyz, tzx = (S[..., i] for i in range(6))
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))


def stress_history(res, body: str) -> dict:
    """Max. von Mises over the nodes at each time, max |u|, the node and the time of the maximum."""
    key, _ = _key(res, body)
    Sm = res.arrays[key + "_smodes"].astype(np.float64)
    Q = res.arrays[key + "_q"]
    phi = res.arrays[key + "_phi"].astype(np.float64)
    vm = np.zeros(len(Q))
    um = np.zeros(len(Q))
    arg = np.zeros(len(Q), int)
    for k in range(len(Q)):
        v = von_mises(Sm @ Q[k])
        arg[k] = int(np.argmax(v))
        vm[k] = v[arg[k]]
        um[k] = float(np.linalg.norm(phi @ Q[k], axis=1).max())
    k = int(np.argmax(vm))
    return {"t": res.arrays["t"], "von_mises_max": vm, "u_max": um, "k_max": k, "node_max": int(arg[k]),
            "point_mm": res.arrays[key + "_nodes"][arg[k]].tolist()}
