"""Bolts and welds in the FE model: inputs for the mesh, equivalent beams, evaluation of the forces after the calculation.

Bolt = a chain of U1 elements (Timoshenko, ccx) between points on the axis; the ends connected by a rigid constraint
(*COUPLING/*KINEMATIC to an auxiliary node R, R ↔ node of the beam through *EQUATION in DOF 1–6).
CAUTION (verified 12. 10. 2026, scratch/f3/kin_test.py): REF NODE = directly the node of U1 gives in ccx 2.23
a mechanism (the rotation about x is not transmitted); *COUPLING/*DISTRIBUTING in ccx is not a kinematic
constraint (it only distributes the load) – not usable for connecting the beam.
Preload: the beam is split in the middle by a pair of nodes M1/M2 and by the "gap node" PT:
u_M2 = u_M1 + g·e (g = DOF 1 of the node PT), rotation M2 = M1. In normal states g = 0 (*BOUNDARY),
in the state "Bolt preload" the force −F_V on PT = tension F_V in the bolt (as *PRE-TENSION SECTION).
Forces in the bolt are evaluated after the calculation (see joints.evaluate).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import tables
from .model import BoltSpec, PRELOAD_CASE
from .vdi2230 import basics

MIN_BOLT_ELEMENTS = 64                   # U1 (see above): 64 elements → error of the bending of the beam ~−0.02 % (16: −0.37 %)


@dataclass
class BoltFE:
    """Data of the equivalent beam of one bolt for writing the .inp."""
    name: str
    index: int
    points: np.ndarray                 # (K, 3) nodes of the beam (without the preload pair)
    split: int | None                  # index of the point where the preload cut is (M1 = M2 = points[split])
    couplings: list[tuple[str, int]]   # (selection in the mesh, index of the point of the beam)
    A: float
    I: float
    E: float
    nu: float = 0.3
    e: np.ndarray = field(default_factory=lambda: np.zeros(3))
    preload: float = 0.0               # F_V for the state "Bolt preload" [N]
    alpha: float = 11.5e-6             # thermal expansion of the bolt [1/K] (steel; stainless A2/A4 16e-6)
    length: float = 0.0                # length of the beam between the end constraints [mm]


def bolt_materials(study, b: BoltSpec) -> tuple[float, dict, float, float, float | None]:
    """(E of the bolt, E of the clamped parts, E of the nut/internal thread, Rp0.2 of the bolt, Rm min. of the clamped parts)."""
    sc = tables.strength_class(b.strength, b.size)
    plate_E, Rm = {}, []
    for pl in b.plates:
        try:
            m = study.material_of(pl["part"])
            plate_E[pl["part"]] = m.E
            if m.Rm:
                Rm.append(m.Rm)
        except KeyError:
            plate_E[pl["part"]] = 210000.0
    E_M = sc.E
    if b.connection == "tapped":
        try:
            E_M = study.material_of(b.nut_part).E
        except KeyError:
            pass
    return sc.E, plate_E, E_M, sc.Rp02, (min(Rm) if Rm else None)


def bearing_limit(study, b: BoltSpec) -> tuple[float | None, str]:
    """Limiting pressure p_G under the head and the nut (the weaker of the parts under them) – limits the automatic preload."""
    from .vdi2230 import limiting_pressure
    parts = [b.head_part] + ([b.nut_part] if b.connection == "nut" else [])
    best = None
    for pn in parts:
        try:
            m = study.material_of(pn)
        except KeyError:
            continue
        pg, src = limiting_pressure(m.Rm, m.E)
        if best is None or pg < best[0]:
            best = (pg, f"{pn}: {src}")
    return best if best else (None, "")


def bolt_basics(study, b: BoltSpec) -> dict:
    E_S, plate_E, E_M, Rp, _ = bolt_materials(study, b)
    pG, _ = bearing_limit(study, b)
    return basics(b, E_S, plate_E, E_M, Rp, getattr(b, "DA", None), pG)


def package_plates(b: BoltSpec) -> list[int]:
    """Bundle with clearance in CAD: indices of the intermediate parts that are connected to the bolt (the clearance closed after tightening)."""
    if not has_gap(b) or b.connection == "fitted":
        return []
    return list(range(1, len(b.plates) - 1)) if b.connection == "nut" else list(range(1, len(b.plates)))


def mesh_inputs(study, parts) -> tuple[list, dict, list[dict], list[dict], list[str]]:
    """Parts to mesh, extra selections (cylinders of holes), imprints of the annuli, sizes in the welds, warnings."""
    from ..geometry import face_props
    from ..selectors import FaceSelector
    from .detect import _coaxial_cylinders, weld_geometry
    excl = study.excluded_parts()
    mparts = [p for p in parts if p.name not in excl]
    pmap = {p.name: p for p in parts}
    extra: dict[str, list[dict]] = {}
    imprints: list[dict] = []
    warnings: list[str] = []
    js = study.joints
    from .detect import coaxial_cones
    for i, b in enumerate(study.active_bolts(), 1):
        e = b.e
        h = js.bolt_mesh * b.size
        tol = max(0.05, 0.02 * b.size)
        if b.connection in ("nut", "tapped"):
            if b.head_cone > 0:                      # countersunk head: constraint on the conical countersink
                q = pmap.get(b.head_part)
                cones = [c for c in (coaxial_cones(q, b.point(0.0), e, tol) if q else [])
                         if c["s0"] < b.head_cone + tol and c["s1"] > -tol]
                if not cones:
                    warnings.append(f"{b.name}: conical countersink not found in part '{b.head_part}'")
                extra[f"__b{i}_h"] = [FaceSelector.from_props(b.head_part, face_props(c["face"])).to_dict()
                                      for c in cones]
            else:
                imprints.append({"name": f"__b{i}_h", "part": b.head_part, "center": b.point(0.0).tolist(),
                                 "normal": (-e).tolist(), "radius": 0.5 * b.head_dw, "size": h})
        if b.connection == "nut":
            imprints.append({"name": f"__b{i}_n", "part": b.nut_part, "center": b.point(b.nut_s).tolist(),
                             "normal": e.tolist(), "radius": 0.5 * b.nut_dw, "size": h})
        for k in package_plates(b):                  # intermediate parts of a bundle with clearance: face to the head
            pl = b.plates[k]
            imprints.append({"name": f"__b{i}_m{k}", "part": pl["part"], "center": b.point(pl["s0"]).tolist(),
                             "normal": (-e).tolist(), "radius": 0.5 * b.head_dw, "size": h})
        if b.connection == "tapped":
            q = pmap.get(b.nut_part)
            cyls = _coaxial_cylinders(q, b.point(0.0), e, tol) if q else []
            cyls = [c for c in cyls if c["s1"] > b.nut_s + tol and c["s0"] < b.thread_end - tol]
            if not cyls:
                warnings.append(f"{b.name}: cylindrical thread face not found in part '{b.nut_part}'")
            extra[f"__b{i}_t"] = [FaceSelector.from_props(b.nut_part, face_props(c["face"])).to_dict() for c in cyls]
        if b.connection == "fitted":
            for k, pl in enumerate(b.plates):
                q = pmap.get(pl["part"])
                cyls = _coaxial_cylinders(q, b.point(0.0), e, tol) if q else []
                cyls = [c for c in cyls if c["s1"] > pl["s0"] + tol and c["s0"] < pl["s1"] - tol]
                extra[f"__b{i}_p{k}"] = [FaceSelector.from_props(pl["part"], face_props(c["face"])).to_dict()
                                         for c in cyls]
    vsizes = []
    for w in study.active_welds():
        if w.part not in pmap:
            warnings.append(f"{w.name}: part '{w.part}' is not in the model")
            continue
        a = w.a
        if a is None:
            try:
                a = weld_geometry(parts, w.part).get("a")
            except Exception as e:  # noqa: BLE001 – mesh refinement only; the weld is still assessed
                a = None
                warnings.append(f"{w.name}: the throat thickness could not be measured from the CAD ({e}) – "
                                f"the weld body is meshed with the global element size (no local refinement)")
        if a:
            vsizes.append({"part": w.part, "size": js.weld_mesh * a})
    return mparts, extra, imprints, vsizes, warnings


def open_weld_roots(study, mesh) -> list[str]:
    """Splits the direct contact of parts joined by a weld (doubling of the nodes of the joint including the root line for the
    second part). The weld stays connected to both parts by its contact faces, the root line only to
    the first part (root gap) → the whole load goes through the weld. The pressure in the contact is not transferred
    (conservative for the weld). Mesh.node_alias remembers the original numbers of the doubled nodes.
    """
    from .forces import mesh_interfaces
    welds = [w for w in study.active_welds() if w.open_root and w.part in mesh.part_names]
    if not welds:
        return []
    ifaces = mesh_interfaces(mesh)
    pidx = {n: i for i, n in enumerate(mesh.part_names)}
    weld_parts = {pidx[w.part] for w in study.active_welds() if w.part in pidx}
    keep = np.zeros(len(mesh.nodes), bool)               # nodes of parts other than the neighbouring ones are not split
    pairs = set()
    for w in welds:
        pw = pidx[w.part]
        touch = sorted({b if a == pw else a for (a, b) in ifaces if pw in (a, b)} - weld_parts)
        for i in range(len(touch)):
            for j in range(i + 1, len(touch)):
                if (touch[i], touch[j]) in ifaces:
                    pairs.add((touch[i], touch[j]))
    info = []
    for (pa, pb) in sorted(pairs):
        nodes = np.unique(ifaces[(pa, pb)]["tri"])
        nodes = nodes[~keep[nodes]]
        if not len(nodes):
            continue
        n0 = len(mesh.nodes)
        alias = mesh.canonical()
        remap = np.full(n0, -1, dtype=np.int64)
        remap[nodes] = n0 + np.arange(len(nodes))
        mesh.nodes = np.vstack([mesh.nodes, mesh.nodes[nodes]])
        mesh.node_alias = np.concatenate([alias, alias[nodes]])
        el = mesh.tet_part == pb
        T = mesh.tets[el]
        hit = remap[T] >= 0
        T[hit] = remap[T][hit]
        mesh.tets[el] = T
        # selections on part pb that reach the split nodes → new node number
        for key, tri in mesh.surfaces.items():
            if not len(tri):
                continue
            ok_old = _faces_exist(mesh.tets, tri)
            if ok_old.all():
                continue
            rm = remap[tri]
            t2 = np.where(rm >= 0, rm, tri)
            ok_new = _faces_exist(mesh.tets, t2)
            mesh.surfaces[key] = np.where(ok_old[:, None], tri, t2)
            if not np.all(ok_old | ok_new):
                info.append(f"Selection '{key}': part of the faces not found after separating the interface")
        info.append(f"Interface '{mesh.part_names[pa]}' – '{mesh.part_names[pb]}' separated ({len(nodes)} nodes), load is carried through the weld")
    mesh.report["nodes"] = int(len(mesh.nodes))
    return info


def _faces_exist(tets, tri) -> np.ndarray:
    from ..mesh import TET_FACES, _row_view
    faces = tets[:, :4][:, TET_FACES].reshape(-1, 3)
    faces_s = np.sort(faces, axis=1)
    tri_s = np.sort(tri[:, :3], axis=1)
    max_node = max(int(faces_s.max(initial=0)), int(tri_s.max(initial=0)))
    if max_node < (1 << 21):
        f64 = (faces_s[:, 0].astype(np.uint64) << 42) | (faces_s[:, 1].astype(np.uint64) << 21) | faces_s[:, 2].astype(np.uint64)
        q64 = (tri_s[:, 0].astype(np.uint64) << 42) | (tri_s[:, 1].astype(np.uint64) << 21) | tri_s[:, 2].astype(np.uint64)
        keys = np.sort(f64)
        pos = np.clip(np.searchsorted(keys, q64), 0, len(keys) - 1)
        return keys[pos] == q64
    keys = np.sort(_row_view(faces_s))
    q = _row_view(tri_s)
    pos = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
    return keys[pos] == q


def finish_mesh(study, mesh) -> tuple[list[str], list[str]]:
    """After meshing: limit the thread to the screwed-in length, check of the constraint faces of the bolts.
    Returns (errors, warnings); a missing face of an intermediate part of the bundle is only a warning.
    """
    errors, warnings = [], []
    for i, b in enumerate(study.active_bolts(), 1):
        if b.connection == "tapped":
            key = f"__b{i}_t"
            tri = mesh.surfaces.get(key, np.zeros((0, 6), int))
            if len(tri):
                s = (mesh.nodes[tri] - np.asarray(b.origin)) @ b.e
                tol = 0.02 * b.size
                ok = (s >= b.nut_s - tol).all(axis=1) & (s <= b.thread_end + tol).all(axis=1)
                mesh.surfaces[key] = tri[ok]
        for key, _ in coupling_keys(b, i):
            if len(mesh.surfaces.get(key, [])) == 0:
                if key.rsplit("_", 1)[-1].startswith("m"):
                    warnings.append(f"{b.name}: intermediate part of the stack not attached (face '{key}' not found)")
                else:
                    errors.append(f"{b.name}: support face '{key}' has no elements in the mesh")
    return errors, warnings


def coupling_keys(b: BoltSpec, i: int, mesh=None) -> list[tuple[str, float]]:
    """(selection in the mesh, position s on the axis) for the constraints of the bolt; with mesh only the faces that exist in the mesh."""
    sh = 0.5 * b.head_cone                           # countersunk head: centre of the conical countersink
    if b.connection == "nut":
        keys = [(f"__b{i}_h", sh), (f"__b{i}_n", b.nut_s)]
    elif b.connection == "tapped":
        keys = [(f"__b{i}_h", sh), (f"__b{i}_t", b.beam_end())]
    else:
        keys = [(f"__b{i}_p{k}", 0.5 * (pl["s0"] + pl["s1"])) for k, pl in enumerate(b.plates)]
    keys += [(f"__b{i}_m{k}", b.plates[k]["s0"]) for k in package_plates(b)]
    if mesh is not None:
        keys = [(k, s) for k, s in keys if len(mesh.surfaces.get(k, [])) > 0]
    return keys


def has_gap(b: BoltSpec) -> bool:
    """Gap between the clamped parts (or to a part with a thread) – the parts do not touch."""
    tol = max(0.05, 0.02 * b.size)
    stack = [(pl["s0"], pl["s1"]) for pl in b.plates]
    if b.connection == "tapped":
        stack.append((b.nut_s, b.thread_end))
    return any(stack[k + 1][0] - stack[k][1] > tol for k in range(len(stack) - 1))


def bolt_models(study, mesh=None) -> list[BoltFE]:
    out = []
    js = study.joints
    for i, b in enumerate(study.active_bolts(), 1):
        bas = bolt_basics(study, b)
        keys = coupling_keys(b, i, mesh)
        s_end = max(s for _, s in keys)
        s_beg = min(s for _, s in keys)
        L = s_end - s_beg
        n = max(MIN_BOLT_ELEMENTS, int(js.bolt_elements))
        n += n % 2                                       # even number → preload cut in the middle
        ss = sorted(set(np.round(np.linspace(s_beg, s_end, n + 1), 9).tolist() + [round(s, 9) for _, s in keys]))
        pts = np.array([b.point(s) for s in ss])
        split = None
        # the preload cut also serves the thermal elongation of the bolt (prescribed gap α·ΔT·L)
        if (js.preload_case or study.has_thermal_loads()) and not has_gap(b):   # without contact of the parts = mechanism
            mid = 0.5 * (s_beg + s_end)
            split = int(np.argmin([abs(s - mid) for s in ss]))
            if split in (0, len(ss) - 1):
                split = None
        couplings = [(k, int(np.argmin([abs(s - sk) for s in ss]))) for k, sk in keys]
        # stiffness of the beam = bolt (VDI δ_S) + washers in series between the surfaces of the clamped parts
        dW = sum(w["h"] / (bas["E_S"] * np.pi / 4 * (w["d2"] ** 2 - w["d1"] ** 2)) for w in b.washers)
        comp = bas["delta_S"] + dW
        E = bas["E_S"]
        A = L / (E * comp) if L > 0 else bas["As"]
        d_eq = np.sqrt(4 * A / np.pi)
        I = np.pi * d_eq ** 4 / 64
        alpha = 16.0e-6 if str(b.strength).upper().startswith("A") else 11.5e-6
        out.append(BoltFE(b.name, i, pts, split, couplings, A, I, E, 0.3, b.e,
                          bas["F_Vnom"] if js.preload_case else 0.0, alpha, L))
    return out


def preload_case_needed(study) -> bool:
    return bool(study.active_bolts()) and study.joints.preload_case


__all__ = ["BoltFE", "mesh_inputs", "finish_mesh", "bolt_models", "PRELOAD_CASE",
           "coupling_keys", "bolt_basics", "preload_case_needed"]
