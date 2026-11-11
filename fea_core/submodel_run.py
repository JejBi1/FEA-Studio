"""Submodel of a solved study (C1 for GUI, MCP and the report): a finer local model of a region (box / sphere) cut out of the meshed
parts, driven by the displacements of the global solution on the cut faces.

Procedure (`run_submodel`):
  1. checks: linear static load case of a 3D result, isotropic materials, no thermal load (the thermal strain is not transferred);
  2. every meshed part is intersected with the region (OCP BRepAlgoAPI_Common) and the pieces are meshed together (conforming, tet10,
     separate Gmsh process, element size of the submodel);
  3. boundary triangles of the submodel lying on the region surface AND inside the global body (a point a little beyond the face is
     inside the global mesh) = CUT faces: the global displacement is interpolated to their nodes (tet10 shape functions) and prescribed;
     boundary triangles on the region surface outside the global body = original free surface (no condition);
  4. other boundary triangles are matched to the faces of the global selections (distance to the global selection triangles + the
     same outward normal): Fixed / Displacement → prescribed components; Force on faces → the same traction F / A_global; Pressure →
     the same p(x); body loads (gravity, rotation, inertia) → consistent nodal forces on the submodel; anything else inside the region
     (bolt couplings, bearing, remote or elastic supports, edge/point forces, weld roots…) = SubmodelError (better an error than a
     wrong result);
  5. ccx linear static analysis; equilibrium check = residual of the free nodes Σ(RF − F) against the applied forces (the loads were
     applied as intended) and the resultant of the cut reactions;
  6. cut check: von Mises of the submodel on the cut against the global field interpolated to the same nodes – a large difference
     means the cut lies too close to the stress concentration (enlarge the region).

The result (`SubmodelResult`) is stored in `Results.submodels` and saved with the results (`results_sub.npz`).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mesh import Mesh
from .results import Results, StepResult, tet10_shape, von_mises

FILE_FORMAT = "fea-studio-submodels"
MAX_DOF = 2_500_000                 # largest submodel (ccx PARDISO ~ 10 GB)
CUT_MEAN_TOL = 0.10                 # cut check: mean |Δσ_vM| / mean σ_vM,global
CUT_MAX_TOL = 0.25                  # cut check: max |Δσ_vM| / max σ_vM,global
_OPP = np.array([3, 2, 0, 1])


class SubmodelError(ValueError):
    """The submodel cannot be computed (setting, unsupported condition inside the region)."""


# ===================================================================== specification

@dataclass
class SubmodelRegion:
    """Region of the submodel: shape "box" (centre, size = [dx, dy, dz] in the global axes) | "sphere" (centre, radius)."""
    center: list[float]
    shape: str = "box"
    size: list[float] = field(default_factory=lambda: [20.0, 20.0, 20.0])
    radius: float = 10.0
    mesh_size: float = 1.0
    step: int = 0
    name: str = ""

    def check(self) -> None:
        if self.shape not in ("box", "sphere"):
            raise SubmodelError("Submodel region shape must be box or sphere")
        c = [float(v) for v in self.center]
        if len(c) != 3 or not all(math.isfinite(v) for v in c):
            raise SubmodelError("Submodel centre must be three numbers [x, y, z] mm")
        if self.shape == "box":
            s = [float(v) for v in self.size]
            if len(s) != 3 or not all(math.isfinite(v) and v > 0 for v in s):
                raise SubmodelError("Submodel box size must be three positive numbers [mm]")
            if self.mesh_size * 2 > min(s):
                raise SubmodelError("Submodel element size must be at most half of the smallest box size")
        else:
            if not (math.isfinite(self.radius) and self.radius > 0):
                raise SubmodelError("Submodel sphere radius must be positive")
            if self.mesh_size * 2 > self.radius:
                raise SubmodelError("Submodel element size must be at most half of the sphere radius")
        if not (math.isfinite(self.mesh_size) and self.mesh_size > 0):
            raise SubmodelError("Submodel element size must be positive")

    @property
    def extent(self) -> float:
        """Characteristic size of the region (box diagonal / sphere diameter)."""
        return float(np.linalg.norm(self.size)) if self.shape == "box" else 2.0 * float(self.radius)

    def signed_distance(self, X: np.ndarray) -> np.ndarray:
        """< 0 inside the region, 0 on its surface (for a box the maximum-norm distance, exact on the surface)."""
        X = np.asarray(X, float)
        c = np.asarray(self.center, float)
        if self.shape == "sphere":
            return np.linalg.norm(X - c, axis=-1) - float(self.radius)
        h = 0.5 * np.asarray(self.size, float)
        return np.max(np.abs(X - c) - h, axis=-1)

    def to_dict(self) -> dict:
        d = {"shape": self.shape, "center": [float(v) for v in self.center], "mesh_size": float(self.mesh_size),
             "step": int(self.step), "name": self.name}
        if self.shape == "box":
            d["size"] = [float(v) for v in self.size]
        else:
            d["radius"] = float(self.radius)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SubmodelRegion":
        return cls(center=list(d["center"]), shape=d.get("shape", "box"), size=list(d.get("size") or [20.0] * 3),
                   radius=float(d.get("radius", 10.0)), mesh_size=float(d.get("mesh_size", 1.0)),
                   step=int(d.get("step", 0)), name=str(d.get("name", "")))

    def tool_shape(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
        from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
        c = [float(v) for v in self.center]
        if self.shape == "sphere":
            return BRepPrimAPI_MakeSphere(gp_Pnt(*c), float(self.radius)).Shape()
        dx, dy, dz = (float(v) for v in self.size)
        return BRepPrimAPI_MakeBox(gp_Ax2(gp_Pnt(c[0] - dx / 2, c[1] - dy / 2, c[2] - dz / 2), gp_Dir(0, 0, 1)),
                                   dx, dy, dz).Shape()

    def describe(self) -> str:
        c = ", ".join(f"{v:.6g}" for v in self.center)
        if self.shape == "box":
            return f"box {' × '.join(f'{v:.6g}' for v in self.size)} mm at ({c}) mm"
        return f"sphere R {self.radius:.6g} mm at ({c}) mm"


# ===================================================================== result

@dataclass
class SubmodelResult:
    region: SubmodelRegion
    step_name: str
    mesh: Mesh
    U: np.ndarray                      # (N, 3) mm
    S: np.ndarray                      # (N, 6) MPa
    cut_nodes: np.ndarray              # submodel nodes with the interpolated global displacement
    glob_cut_mises: np.ndarray         # global von Mises interpolated to cut_nodes
    summary: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    _step: StepResult | None = None
    _results: Results | None = None

    @property
    def name(self) -> str:
        return self.region.name or f"Submodel – {self.step_name}"

    def step(self) -> StepResult:
        if self._step is None:
            self._step = StepResult(self.step_name, self.U, self.S)
        return self._step

    def as_results(self) -> Results:
        """The submodel as a one-state Results (render, probes, diagnostics of the 3D results path)."""
        if self._results is None:
            self._results = Results(self.mesh, [self.step()], list(self.warnings), {"submodel": self.region.to_dict()})
        return self._results

    @property
    def check(self) -> dict:
        return self.summary.get("cut_check", {})


FIELD_KEYS = ("mises", "s1", "s2", "s3", "tresca", "sxx", "syy", "szz", "sxy", "syz", "szx",
              "displacement", "ux", "uy", "uz", "safety")


def sub_field(sr: SubmodelResult, key: str, study=None) -> np.ndarray:
    """Nodal field of the submodel (keys as in fea_core.mcp.diagnostics.FIELDS, subset FIELD_KEYS); safety = Re/σ_vM (max 10)."""
    st = sr.step()
    if key == "mises":
        return st.mises
    if key in ("s1", "s2", "s3", "tresca"):
        p = st.principal
        return p[:, 0] - p[:, 2] if key == "tresca" else p[:, int(key[1]) - 1]
    comp = {"sxx": 0, "syy": 1, "szz": 2, "sxy": 3, "syz": 4, "szx": 5}
    if key in comp:
        return st.S[:, comp[key]]
    if key == "displacement":
        return st.umag
    if key in ("ux", "uy", "uz"):
        return st.U[:, "xyz".index(key[1])]
    if key == "safety":
        m = sr.mesh
        re_node = np.full(len(m.nodes), np.inf)
        for pi, pn in enumerate(m.part_names):
            try:
                re = study.material_of(pn).Re if study is not None else None
            except KeyError:
                re = None
            if re:
                np.minimum.at(re_node, m.tets[m.tet_part == pi].ravel(), float(re))
        if not np.isfinite(re_node).any():
            raise SubmodelError("No material of the submodel parts has Re – safety factor cannot be computed")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.minimum(np.where(np.isfinite(re_node), re_node / np.maximum(st.mises, 1e-12), 10.0), 10.0)
    raise SubmodelError(f"Unknown submodel quantity '{key}' (available: {', '.join(FIELD_KEYS)})")


def _arr(d: dict, key: str, kind: str, ndim: int, width: int | None = None) -> np.ndarray:
    if key not in d:
        raise ValueError(f"Invalid submodel file: missing {key}")
    a = d[key]
    if not isinstance(a, np.ndarray) or a.dtype.kind not in kind or a.ndim != ndim or \
            (width is not None and a.shape[1] != width):
        raise ValueError(f"Invalid submodel file: {key} has a wrong type or shape")
    return a


def save_submodels(subs: list[SubmodelResult], path: str | Path) -> None:
    arrays, meta = {}, []
    for k, s in enumerate(subs):
        arrays[f"k{k}_nodes"] = s.mesh.nodes
        arrays[f"k{k}_tets"] = np.asarray(s.mesh.tets, np.int32)
        arrays[f"k{k}_tet_part"] = np.asarray(s.mesh.tet_part, np.int32)
        arrays[f"k{k}_U"] = s.U
        arrays[f"k{k}_S"] = s.S
        arrays[f"k{k}_cut"] = np.asarray(s.cut_nodes, np.int32)
        arrays[f"k{k}_gcut"] = np.asarray(s.glob_cut_mises, float)
        meta.append({"region": s.region.to_dict(), "step_name": s.step_name, "part_names": list(s.mesh.part_names),
                     "summary": s.summary, "warnings": list(s.warnings), "report": s.mesh.report})
    head = {"format": FILE_FORMAT, "version": 1, "submodels": meta}
    arrays["meta"] = np.frombuffer(json.dumps(head, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **arrays)


def load_submodels(path: str | Path) -> list[SubmodelResult]:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    head = json.loads(bytes(_arr(d, "meta", "u", 1)).decode("utf-8"))
    if head.get("format") != FILE_FORMAT:
        raise ValueError("Invalid submodel file: unknown format")
    out = []
    for k, m in enumerate(head.get("submodels", [])):
        nodes = _arr(d, f"k{k}_nodes", "f", 2, 3).astype(float)
        tets = _arr(d, f"k{k}_tets", "iu", 2).astype(np.int64)
        tp = _arr(d, f"k{k}_tet_part", "iu", 1).astype(np.int64)
        U = _arr(d, f"k{k}_U", "f", 2, 3)
        S = _arr(d, f"k{k}_S", "f", 2, 6)
        cut = _arr(d, f"k{k}_cut", "iu", 1).astype(np.int64)
        g = _arr(d, f"k{k}_gcut", "f", 1)
        names = [str(x) for x in m.get("part_names", [])]
        n = len(nodes)
        if (tets.shape[1] not in (4, 10) or len(tp) != len(tets) or len(U) != n or len(S) != n or len(g) != len(cut)
                or (tets.size and (tets.min() < 0 or tets.max() >= n)) or (cut.size and (cut.min() < 0 or cut.max() >= n))
                or (tp.size and (tp.min() < 0 or tp.max() >= max(len(names), 1)))):
            raise ValueError("Invalid submodel file: inconsistent arrays")
        mesh = Mesh(nodes, tets, tp, names, report=m.get("report") or {})
        out.append(SubmodelResult(SubmodelRegion.from_dict(m["region"]), str(m.get("step_name", "")), mesh, U, S, cut, g,
                                  m.get("summary") or {}, [str(w) for w in m.get("warnings", [])]))
    return out


# ===================================================================== geometry helpers

def _locate_err(global_results: Results, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(element, natural coordinates, barycentric excess – 0 inside) of the points in the global mesh."""
    from .submodel import locate_batch
    el, xi = locate_batch(global_results, pts)
    err = np.maximum(np.maximum(-xi.min(axis=1), xi.sum(axis=1) - 1.0), 0.0) if len(xi) else np.zeros(0)
    return el, xi, err


def interpolate(global_results: Results, field_vals: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Nodal field of the global mesh (N, …) interpolated to the points (tet10 shape functions)."""
    el, xi, _ = _locate_err(global_results, pts)
    tets = global_results.mesh.tets
    N = tet10_shape(xi) if tets.shape[1] == 10 else np.column_stack([1 - xi.sum(axis=1), xi])
    return np.einsum("ki,ki...->k...", N, np.asarray(field_vals)[tets[el]])


def _outward(mesh: Mesh, tri: np.ndarray, elem: np.ndarray) -> np.ndarray:
    """Unit outward normals of the corner triangles (orientation by the opposite corner of the element)."""
    X = mesh.nodes[tri[:, :3]]
    n = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
    corners = mesh.tets[elem, :4]
    other = ~(corners[:, :, None] == tri[:, None, :3]).any(axis=2)
    opp = mesh.nodes[corners[np.arange(len(elem)), np.argmax(other, axis=1)]]
    sign = np.where(np.einsum("ij,ij->i", n, X.mean(axis=1) - opp) < 0, -1.0, 1.0)
    n = n * sign[:, None]
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]


def _linear_pieces(tri: np.ndarray) -> np.ndarray:
    """tri6 → 4 linear triangles (better on curved faces), tri3 unchanged."""
    if tri.shape[1] == 6:
        return np.concatenate([tri[:, [0, 3, 5]], tri[:, [3, 1, 4]], tri[:, [5, 4, 2]], tri[:, [3, 4, 5]]])
    return tri[:, :3]


def match_selection(sub_mesh: Mesh, sub_tri: np.ndarray, sub_n: np.ndarray, gmesh: Mesh, name: str) -> np.ndarray:
    """Mask of the submodel boundary triangles lying on the global selection face `name` (centroid within ~15 % of the local
    global element size from the global selection triangles and the same outward normal)."""
    from scipy.spatial import cKDTree
    from .printing.region import point_triangle_distance
    gt = gmesh.surfaces.get(name)
    if gt is None or not len(gt) or not len(sub_tri):
        return np.zeros(len(sub_tri), bool)
    elems, _ = gmesh.element_faces(name)
    gn = _outward(gmesh, gt, elems)
    pieces = _linear_pieces(gt)
    reps = len(pieces) // len(gt)
    pn = np.tile(gn, (reps, 1))
    P = gmesh.nodes[pieces]
    h = np.linalg.norm(P - np.roll(P, 1, axis=1), axis=2).max(axis=1) * (2.0 if reps == 4 else 1.0)
    tree = cKDTree(P.mean(axis=1))
    c = sub_mesh.nodes[sub_tri[:, :3]].mean(axis=1)
    k = min(12, len(P))
    _, cand = tree.query(c, k=k)
    cand = np.asarray(cand).reshape(len(c), k)
    best = np.full(len(c), np.inf)
    ok = np.zeros(len(c), bool)
    for j in range(k):
        ci = cand[:, j]
        d = point_triangle_distance(c, P[ci, 0], P[ci, 1], P[ci, 2])
        good = (np.einsum("ij,ij->i", sub_n, pn[ci]) > 0.7) & (d < 0.15 * h[ci] + 1e-6)
        better = good & (d < best)
        best[better] = d[better]
        ok |= good
    return ok


# ===================================================================== analysis

def _find_case(study, name: str):
    for lc in study.load_cases:
        if lc.name == name:
            return lc
    return None


def _check_inputs(study, results: Results, region: SubmodelRegion):
    region.check()
    if results is None or not results.steps:
        raise SubmodelError("The submodel needs the results of a load case (run the analysis first)")
    if len(results.mesh.nodes) == 0 or results.mesh.tets.shape[1] != 10:
        raise SubmodelError("The submodel needs a 3D solid (tet10) result – not available for frames, 2D sections or shells")
    if not 0 <= region.step < len(results.steps):
        raise SubmodelError(f"Load case index {region.step} is out of range (the results have {len(results.steps)} cases)")
    st = results.steps[region.step]
    if study.is_nonlinear() or (results.info or {}).get("nonlinear") or st.partial:
        raise SubmodelError("The submodel is for linear static results – nonlinear analyses (contact, NLGEOM, plasticity, sequences) are not supported")
    if "T" in st.fields:
        raise SubmodelError(f"Load case “{st.name}” has a thermal load – the submodel does not transfer thermal strain")
    return st


def _part_shapes(parts, names: list[str]) -> dict:
    by = {p.name: p for p in (parts or [])}
    missing = [n for n in names if n not in by]
    if missing:
        raise SubmodelError(f"Geometry of the meshed parts not found: {', '.join(missing)}")
    return by


def _volume(shape) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    return abs(float(g.Mass()))


def trimmed_parts(study, parts, results: Results, region: SubmodelRegion) -> list:
    """Pieces of the meshed parts inside the region (Part with the original names, only non-empty ones)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from .geometry import Part
    names = list(results.mesh.part_names)
    by = _part_shapes(parts, names)
    tool = region.tool_shape()
    tiny = 1e-6 * region.extent ** 3
    out = []
    for n in names:
        op = BRepAlgoAPI_Common(by[n].shape, tool)
        op.Build()
        if not op.IsDone():
            raise SubmodelError(f"Cutting part “{n}” by the submodel region failed")
        sh = op.Shape()
        if _volume(sh) > tiny:
            out.append(Part(n, sh))
    if not out:
        raise SubmodelError("The submodel region does not intersect any meshed part")
    return out


def estimate(study, parts, results: Results, region: SubmodelRegion) -> dict:
    """Size estimate before the run: volume in the region, elements, nodes, equations (tet10 ≈ h³/8.5 per element)."""
    _check_inputs(study, results, region)
    pieces = trimmed_parts(study, parts, results, region)
    vol = sum(_volume(p.shape) for p in pieces)
    elems = vol / (region.mesh_size ** 3 / 8.5)
    nodes = 1.45 * elems + 2000
    return {"parts": [p.name for p in pieces], "volume_mm3": vol, "elements": int(elems), "nodes": int(nodes),
            "dof": int(3 * nodes), "ok": 3 * nodes <= MAX_DOF,
            "message": "" if 3 * nodes <= MAX_DOF else f"Too large: ~{int(3 * nodes):,} equations (limit {MAX_DOF:,}) – increase the element size or reduce the region".replace(",", " ")}


def _selection_entity(study, name: str) -> str:
    sel = study.selections.get(name)
    return sel.entity if sel is not None else "faces"


def run_submodel(study, parts, results: Results, region: SubmodelRegion, workdir: str | Path, on_log=None,
                 cancel=None) -> SubmodelResult:
    """Computes the submodel of `results.steps[region.step]` (see the module description). parts = OCP parts of the study."""
    from .ccx.frd import read_frd
    from .ccx.runner import run_ccx
    from .config import EQUILIBRIUM_TOL
    from .loads import force_nodal, global_vector, gravity_nodal_forces, inertia_forces, pressure_nodal, rotation_forces
    from .mesh import generate_mesh
    from .study import BearingLoad, Displacement, Fixed, Force, Gravity, Pressure, RemoteLoad, Rotation, ThermalLoad
    log = on_log or (lambda s: None)
    t0 = time.time()
    st = _check_inputs(study, results, region)
    gm = results.mesh
    lc = _find_case(study, st.name)
    if lc is None:
        from .joints.model import is_preload
        if not is_preload(st.name):
            raise SubmodelError(f"State “{st.name}” is not a load case of the study (creep or derived states cannot be submodelled)")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    # ---- materials (isotropic only)
    mats = {}
    for n in gm.part_names:
        m = study.material_of(n)
        if m.is_print or m.ortho:
            raise SubmodelError(f"Part “{n}” has a printed or orthotropic material – the submodel supports isotropic materials only")
        mats[n] = m

    # ---- weld roots (disconnected contact) inside the region
    reg_tol = 1e-6 * max(region.extent, 1.0)
    if gm.node_alias is not None:
        dup = np.nonzero(gm.node_alias != np.arange(len(gm.node_alias)))[0]
        if len(dup) and np.any(region.signed_distance(gm.nodes[dup]) < -reg_tol):
            raise SubmodelError("A weld root (disconnected contact of welded parts) lies inside the submodel region – not supported, move the region")

    # ---- geometry and mesh
    log(f"Submodel: cutting the parts by the region ({region.describe()})")
    pieces = trimmed_parts(study, parts, results, region)
    if cancel is not None and cancel.is_set():
        raise SubmodelError("Submodel cancelled by user")
    h = float(region.mesh_size)
    dof_est = 3 * (1.45 * sum(_volume(p.shape) for p in pieces) / (h ** 3 / 8.5) + 2000)
    if dof_est > MAX_DOF:
        raise SubmodelError(f"The submodel would be too large (~{int(dof_est):,} equations, limit {MAX_DOF:,}) – increase the element size or reduce the region".replace(",", " "))
    settings = {"size": h, "min_size": h / 10.0, "order": 2, "optimize_high_order": True, "threads": 1,
                "algorithm3d": int(getattr(study.mesh, "algorithm3d", 10) or 10)}
    log(f"Submodel: meshing {len(pieces)} part(s), element size {h:g} mm")
    sm = generate_mesh(pieces, {}, settings, workdir / "mesh", cancel=cancel)
    sm.surfaces = {}
    n_sub = len(sm.nodes)
    if 3 * n_sub > MAX_DOF:
        raise SubmodelError(f"The submodel mesh is too large ({3 * n_sub:,} equations, limit {MAX_DOF:,})".replace(",", " "))
    log(f"Submodel mesh: {n_sub} nodes, {len(sm.tets)} C3D10 elements")

    # ---- cut faces
    btri, belem = sm.boundary_faces()
    bn = _outward(sm, btri, belem)
    tol_s = 1e-4 * region.extent + 1e-6
    on_surf = np.all(np.abs(region.signed_distance(sm.nodes[btri])) < tol_s, axis=1)
    edge = np.linalg.norm(sm.nodes[btri[:, 1]] - sm.nodes[btri[:, 0]], axis=1)
    probe = sm.nodes[btri[:, :3]].mean(axis=1) + (0.3 * edge)[:, None] * bn
    is_cut = np.zeros(len(btri), bool)
    if on_surf.any():
        _, _, err = _locate_err(results, probe[on_surf])
        is_cut[np.nonzero(on_surf)[0]] = err <= 1e-6
    if not is_cut.any():
        raise SubmodelError("The submodel region has no cut face inside the body – the region contains whole parts; analyse them directly")
    cut_nodes = np.unique(btri[is_cut])

    # ---- conditions and loads inside the region
    free_tri = ~is_cut
    ft, fn = btri[free_tri], bn[free_tri]
    bc_val = {}                                            # node -> [ux, uy, uz] (None = free)
    F = np.zeros((n_sub, 3))
    mapped = []
    unsupported = []
    glob_U = st.U

    def matched(name):
        if name not in gm.surfaces:
            return np.zeros(len(ft), bool)
        return match_selection(sm, ft, fn, gm, name)

    for c in study.constraints:
        if c.selection not in gm.surfaces:
            continue
        mk = matched(c.selection)
        if not mk.any():
            continue
        nodes = np.unique(ft[mk])
        if isinstance(c, Fixed):
            comp = [0.0, 0.0, 0.0]
        elif isinstance(c, Displacement):
            comp = [c.ux, c.uy, c.uz]
        else:
            unsupported.append(f"support “{c.selection}” ({c.kind})")
            continue
        for nd in nodes.tolist():
            cur = bc_val.setdefault(nd, [None, None, None])
            for k in range(3):
                if comp[k] is not None:
                    cur[k] = float(comp[k])
        mapped.append({"selection": c.selection, "kind": c.kind, "nodes": int(len(nodes))})

    for ld in (lc.loads if lc is not None else []):
        if isinstance(ld, ThermalLoad):
            raise SubmodelError(f"Load case “{st.name}” has a thermal load – the submodel does not transfer thermal strain")
        if isinstance(ld, Gravity):
            F += gravity_nodal_forces(study, sm, global_vector(study, ld))
            mapped.append({"selection": "", "kind": "gravity"})
            continue
        if isinstance(ld, Rotation):
            F += rotation_forces(study, sm, ld)
            mapped.append({"selection": "", "kind": "rotation"})
            continue
        if getattr(ld, "kind", None) == "inertia":
            F += inertia_forces(study, sm, ld)
            mapped.append({"selection": "", "kind": "inertia"})
            continue
        sel = getattr(ld, "selection", None)
        if not sel:
            continue
        if isinstance(ld, (Force, Pressure)) and _selection_entity(study, sel) == "faces" and sel in gm.surfaces:
            mk = matched(sel)
            if not mk.any():
                continue
            sm.surfaces[sel] = ft[mk]
            if isinstance(ld, Pressure):
                f = pressure_nodal(sm, ld)
            else:
                from .mesh import tri_areas
                a_g = float(tri_areas(gm.nodes, gm.surfaces[sel]).sum())
                a_s = float(tri_areas(sm.nodes, ft[mk]).sum())
                f = force_nodal(study, sm, Force(sel, [float(v) * a_s / a_g for v in ld.vector], ld.csys))
            F += f
            mapped.append({"selection": sel, "kind": ld.kind, "force_N": f.sum(axis=0).tolist()})
            continue
        # edge / point forces, bearing and remote loads: unsupported only when they act inside the region
        if isinstance(ld, (Force, BearingLoad, RemoteLoad)):
            inside = False
            if sel in gm.surfaces:
                inside = bool(matched(sel).any())
            for arr in (gm.edges.get(sel), gm.points.get(sel)):
                if arr is not None and len(arr):
                    inside |= bool(np.any(region.signed_distance(gm.nodes[np.asarray(arr).ravel()]) < tol_s))
            if inside:
                unsupported.append(f"load “{sel}” ({ld.kind})")

    others = [n for n in gm.surfaces if n.startswith("__") and not n.startswith("__tie")]
    if study.frame.active:
        others += [c.selection for c in study.frame.connections]
    for ms in getattr(study, "masses", None) or []:
        if getattr(ms, "selection", None):
            others.append(ms.selection)
    for n in sorted(set(others)):
        if n in gm.surfaces and matched(n).any():
            unsupported.append(f"bolt coupling “{n}”" if n.startswith("__b") else f"face “{n}”")
    if unsupported:
        raise SubmodelError("Conditions inside the submodel region are not supported: " + ", ".join(unsupported)
                            + " – move or shrink the region so that they lie outside it")

    # cut nodes: global displacement (also over a support face – the global solution already satisfies it)
    cut_U = interpolate(results, glob_U, sm.nodes[cut_nodes])
    for nd, u in zip(cut_nodes.tolist(), cut_U):
        bc_val[nd] = [float(u[0]), float(u[1]), float(u[2])]
    if cancel is not None and cancel.is_set():
        raise SubmodelError("Submodel cancelled by user")

    # ---- ccx
    job = "submodel"
    inp = workdir / f"{job}.inp"
    _write_inp(inp, sm, mats, bc_val, F, f"Submodel {st.name}")
    log("Submodel: CalculiX linear static analysis")
    run = run_ccx(workdir, job, cancel=cancel, on_line=None)
    if cancel is not None and cancel.is_set():
        raise SubmodelError("Submodel cancelled by user")
    if not run.finished or run.errors:
        raise SubmodelError("CalculiX failed in the submodel: " + "; ".join(run.errors or ["no 'Job finished'"]))
    frd = read_frd(workdir / f"{job}.frd")
    U = np.zeros((n_sub, 3))
    S = np.zeros((n_sub, 6))
    RF = np.zeros((n_sub, 3))
    for b in frd.blocks:
        if b.name in ("DISP", "U"):
            U[b.node_ids - 1] = b.values[:, :3]
        elif b.name in ("STRESS", "S"):
            S[b.node_ids - 1] = b.values[:, :6]
        elif b.name in ("FORC", "RF"):
            RF[b.node_ids - 1] = b.values[:, :3]

    # ---- equilibrium: the free nodes carry exactly the applied loads
    pres = np.zeros(n_sub, bool)
    pres[np.fromiter(bc_val.keys(), np.int64, len(bc_val))] = True
    R = RF[pres] - F[pres]
    resid = (RF[~pres] - F[~pres]).sum(axis=0)
    ctr = sm.nodes.mean(axis=0)
    mres = np.cross(sm.nodes[~pres] - ctr, RF[~pres] - F[~pres]).sum(axis=0)
    scale = float(np.abs(F).sum() + np.linalg.norm(R, axis=1).sum()) or 1e-300
    imb = max(float(np.linalg.norm(resid)) / scale, float(np.linalg.norm(mres)) / (scale * max(region.extent, 1e-9)))
    eq = {"imbalance": imb, "residual_force_N": resid.tolist(), "applied_force_N": F.sum(axis=0).tolist(),
          "cut_reaction_N": (RF[cut_nodes] - F[cut_nodes]).sum(axis=0).tolist()}
    if imb > EQUILIBRIUM_TOL:
        raise SubmodelError(f"Submodel equilibrium check failed: imbalance {imb:.2e} > {EQUILIBRIUM_TOL:g}")

    # ---- cut check and summary
    vm = von_mises(S)
    g_cut = interpolate(results, st.mises, sm.nodes[cut_nodes])
    s_cut = vm[cut_nodes]
    diff = np.abs(s_cut - g_cut)
    mean_g = float(np.mean(np.abs(g_cut))) or 1e-300
    max_g = float(np.max(np.abs(g_cut))) or 1e-300
    check = {"nodes": int(len(cut_nodes)), "mean_rel_diff": float(diff.mean() / mean_g),
             "max_rel_diff": float(diff.max() / max_g), "mean_global_MPa": mean_g, "mean_sub_MPa": float(s_cut.mean()),
             "tol_mean": CUT_MEAN_TOL, "tol_max": CUT_MAX_TOL}
    check["passed"] = check["mean_rel_diff"] <= CUT_MEAN_TOL and check["max_rel_diff"] <= CUT_MAX_TOL
    if not check["passed"]:
        warnings.append(f"Cut check: the submodel stress at the cut differs from the global model by {100 * check['mean_rel_diff']:.1f} % on average (max {100 * check['max_rel_diff']:.1f} %) – the cut lies too close to the stress concentration or the global mesh is too coarse there; enlarge the region")
    inside_g = np.nonzero(region.signed_distance(gm.nodes) <= tol_s)[0]
    used = np.unique(gm.tets)
    inside_g = np.intersect1d(inside_g, used)
    gvm = st.mises
    pr = StepResult(st.name, U, S).principal
    i_s = int(np.argmax(vm))
    summ = {"step": st.name, "region": region.to_dict(), "region_text": region.describe(),
            "parts": [p.name for p in pieces], "nodes": int(n_sub), "elements": int(len(sm.tets)),
            "min_scaled_jacobian": sm.report.get("min_scaled_jacobian"),
            "max_mises_MPa": float(vm[i_s]), "max_mises_at": sm.nodes[i_s].tolist(),
            "max_s1_MPa": float(pr[:, 0].max()), "min_s3_MPa": float(pr[:, 2].min()),
            "max_displacement_mm": float(np.linalg.norm(U, axis=1).max()),
            "cut_check": check, "equilibrium": eq, "mapped": mapped, "time_s": time.time() - t0}
    if len(inside_g):
        j = int(inside_g[np.argmax(gvm[inside_g])])
        summ.update(global_max_mises_MPa=float(gvm[j]), global_max_mises_at=gm.nodes[j].tolist(),
                    global_mesh_size_mm=float(study.mesh.size))
        summ["ratio_to_global"] = float(vm[i_s] / gvm[j]) if gvm[j] > 0 else None
    if i_s in set(cut_nodes.tolist()):
        warnings.append("The submodel maximum lies on the cut boundary – the region does not contain the stress peak; move or enlarge it")
    elif pres[i_s]:
        warnings.append("The submodel maximum lies on a support face – probably a support singularity")
    log(f"Submodel finished: max. von Mises {vm[i_s]:.4g} MPa, cut check {100 * check['mean_rel_diff']:.1f} % "
        f"(max {100 * check['max_rel_diff']:.1f} %), equilibrium {imb:.1e}")
    sm.report = {k: v for k, v in sm.report.items() if k in ("nodes", "elements", "min_scaled_jacobian")}
    return SubmodelResult(region, st.name, sm, U, S, cut_nodes, g_cut, summ, warnings)


def _write_inp(path: Path, mesh: Mesh, mats: dict, bc_val: dict, F: np.ndarray, title: str) -> None:
    from .ccx import deck_text
    from .config import solver_name
    from .submodel import _write_table
    n, ne = len(mesh.nodes), len(mesh.tets)
    with open(path, "w", encoding="ascii", errors="replace") as f:
        f.write(f"*HEADING\n{deck_text(title)}\n*NODE, NSET=NALL\n")
        X = mesh.nodes
        _write_table(f, "%d, %.12g, %.12g, %.12g\n", [list(range(1, n + 1)), X[:, 0].tolist(), X[:, 1].tolist(),
                                                     X[:, 2].tolist()], n)
        for pi, pname in enumerate(mesh.part_names):
            el = np.nonzero(mesh.tet_part == pi)[0]
            if not len(el):
                continue
            f.write(f"*ELEMENT, TYPE=C3D10, ELSET=P{pi}\n")
            conn = (mesh.tets[el] + 1).T.tolist()
            _write_table(f, "%d" + ", %d" * 10 + "\n", [(el + 1).tolist(), *conn], len(el))
        for pi, pname in enumerate(mesh.part_names):
            if not np.any(mesh.tet_part == pi):
                continue
            m = mats[pname]
            f.write(f"** part {deck_text(pname)}, material {deck_text(m.name)}\n")
            f.write(f"*MATERIAL, NAME=M{pi}\n*ELASTIC\n{float(m.E):.9e}, {float(m.nu):.9e}\n")
            f.write(f"*SOLID SECTION, ELSET=P{pi}, MATERIAL=M{pi}\n")
        f.write(f"*STEP\n*STATIC, SOLVER={solver_name()}\n*BOUNDARY\n")
        rows = []
        for nd in sorted(bc_val):
            for k, v in enumerate(bc_val[nd]):
                if v is not None:
                    rows.append(f"{nd + 1}, {k + 1}, {k + 1}, {v:.10e}\n")
        f.write("".join(rows))
        loaded = np.nonzero(np.abs(F).max(axis=1) > 0)[0] if ne else []
        if len(loaded):
            f.write("*CLOAD\n")
            f.write("".join(f"{nd + 1}, {k + 1}, {F[nd, k]:.10e}\n" for nd in loaded for k in range(3) if F[nd, k] != 0.0))
        f.write("*NODE FILE\nU, RF\n*EL FILE\nS\n*END STEP\n")
