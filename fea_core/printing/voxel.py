"""Local voxel detail of the 3D printing (F5 B): a submodel of the beads around the critical place.

Procedure:
  1. region = a box in the print axes (centre + size + margin) around the given point or the place of the lowest printing safety
     of the homogenised model (StepResult.fields["print_safety"]);
  2. paths from the print settings in voxels (voxel_paths.generate): perimeters, upper/lower layers, thin places, infill;
  3. hex8 elements C3D8I (incompatible modes – bending of thin beads without locking) in ccx, material = full extrusion
     in the axes of the line (printing.part_materials: perimeter `ml`, axes 1 = line, 2 = across, 3 = build – the weak joint of the layers
     is in E_z, Rm_z), *ELASTIC TYPE=ORTHO by classes of the angle of the line with *ORIENTATION (as F4);
  4. boundary conditions = displacements of the global (homogenised) solution interpolated by the shape functions of tet10 in the nodes of the
     cut faces of the box (NOT on the free surface of the part) + in the nodes at the contact with other parts and at the constraint faces → *BOUNDARY,
     one *STEP per load state;
  5. stresses at the integration points (*EL PRINT, GLOBAL=YES) → axes of the bead → Tsai-Wu / Tsai-Hill (failure.safety),
     safety in the region of the evaluation (without the margin), comparison with the homogenised model in the same region.

Verified conventions of ccx 2.23 (scratch/voxel/exp1.py, exp2.py, 14. 10. 2026):
  * C3D8I has 3 internal nodes of incompatible modes as GLOBAL unknowns (ccx does not write them to the .frd): equations
    ≈ 3·nodes + 9·elements (75 k voxels → 914 k equations), C3D8 only 3·nodes;
  * *ORIENTATION RECTANGULAR a = axis 1, b = axis 2 of the material; S in the .frd and *EL PRINT GLOBAL=YES in the global axes,
    *EL PRINT without GLOBAL (default NO) in the axes of the material; .dat order xx yy zz xy xz yz (read_el_print re-sorts);
  * the nodal S in the .frd are extrapolations (for C3D8I a deviation ~4e-5 from the exact homogeneous state) – the safety is therefore
    computed from the integration points (.dat, precision 1e-7).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import solver_name
from . import voxel_paths as vp
from ..ccx import deck_text

MAX_DOF = 1_500_000          # limit of the ccx PARDISO equations for the detail (C3D8I ≈ 12 equations per voxel)
RES = {"w/2": 2, "w/4": 4}


class VoxelError(RuntimeError):
    pass


@dataclass
class VoxelSpec:
    center: list[float] | None = None      # global point [mm]; None = the place of the lowest printing safety
    size: list[float] = field(default_factory=lambda: [6.0, 6.0, 4.0])   # region of the evaluation in the print axes [mm]
    resolution: str = "w/2"                # "w/2" | "w/4" (dx = dy = width of the line / 2 or / 4, dz = height of the layer)
    part: str | None = None                # printed part (None = by the place)
    step: int | str | None = None          # load state (None = all)
    margin: float = 1.0                    # margin around the region (boundary conditions, not evaluated) [mm]
    element: str = "C3D8I"                 # C3D8I | C3D8 (cheaper; in bending of thin beads it locks)
    angle_bins: int | None = None          # classes of the angle of the line (None = PrintSettings.line_bins)
    threads: int | None = None
    max_dof: int = MAX_DOF

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VoxelSpec":
        from dataclasses import fields
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


# ---------------------------------------------------------------- preparation of the region
def _print_parts(study, res) -> list[str]:
    from .region import print_parts
    return [p for p in print_parts(study, list(res.mesh.part_names)) if study.printing.spec(p).enabled]


def _steps(res, step) -> list[int]:
    if step is None:
        return list(range(len(res.steps)))
    if isinstance(step, int):
        return [step]
    return [[s.name for s in res.steps].index(step)]


def auto_center(study, res, part: str | None = None, step=None) -> tuple[np.ndarray, str, str, float]:
    """Place of the lowest printing safety of the homogenised model: (point, part, state, safety)."""
    mesh = res.mesh
    parts = [part] if part else _print_parts(study, res)
    pis = [mesh.part_names.index(p) for p in parts]
    best = (np.inf, None, None, None)
    for si in _steps(res, step):
        st = res.steps[si]
        lam = st.fields.get("print_safety")
        if lam is None:
            continue
        for pi in pis:
            nodes = np.unique(mesh.tets[mesh.tet_part == pi])
            v = lam[nodes]
            ok = np.isfinite(v)
            if not ok.any():
                continue
            j = int(nodes[ok][np.argmin(v[ok])])
            if lam[j] < best[0]:
                best = (float(lam[j]), j, mesh.part_names[pi], st.name)
    if best[1] is None:
        raise VoxelError("The result has no print safety factor (print_safety) – specify the detail region center.")
    return mesh.nodes[best[1]].copy(), best[2], best[3], best[0]


def _part_of_point(study, res, x: np.ndarray) -> str:
    mesh = res.mesh
    best, name = np.inf, None
    for p in _print_parts(study, res):
        pi = mesh.part_names.index(p)
        el = np.nonzero(mesh.tet_part == pi)[0]
        loc = vp.tet_locator(mesh.nodes, mesh.tets[el])
        _, _, err = vp.locate_points(loc, x[None, :])
        if err[0] < best:
            best, name = float(err[0]), p
    if name is None:
        raise VoxelError("The result contains no printed part with a print model.")
    return name


@dataclass
class _Setup:
    part: str
    pi: int
    pspec: object
    R: np.ndarray
    plate: np.ndarray
    lo_in: np.ndarray        # region of the evaluation (coordinates of the print)
    hi_in: np.ndarray
    elems: np.ndarray        # elements of the part in the global mesh
    loc: dict
    grid: vp.VoxelGrid
    center: np.ndarray
    auto: dict


def _setup(study, res, spec: VoxelSpec) -> _Setup:
    from .infill import rotation
    mesh = res.mesh
    auto = {}
    if spec.center is None:
        c, part, sname, lam = auto_center(study, res, spec.part, spec.step)
        auto = {"step": sname, "safety": lam, "point": c.tolist()}
        part = spec.part or part
    else:
        c = np.asarray(spec.center, float)
        part = spec.part or _part_of_point(study, res, c)
    if part not in mesh.part_names:
        raise VoxelError(f"Part '{part}' is not in the result mesh.")
    if part not in _print_parts(study, res):
        raise VoxelError(f"Part '{part}' is not computed with the print model (printed material and print settings enabled).")
    pi = mesh.part_names.index(part)
    pspec = study.printing.spec(part)
    if spec.resolution not in RES:
        raise VoxelError(f"Resolution '{spec.resolution}' – allowed: {', '.join(RES)}.")
    R = rotation(pspec.build_dir)
    elems = np.nonzero(mesh.tet_part == pi)[0]
    pn = mesh.nodes[np.unique(mesh.tets[elems])] @ R.T
    plo, phi = pn.min(axis=0), pn.max(axis=0)
    loc = vp.tet_locator(mesh.nodes, mesh.tets[elems])
    q0 = R @ c
    size = np.asarray(spec.size, float)
    if size.shape != (3,) or np.any(size <= 0):
        raise VoxelError("Detail region size = 3 positive numbers [mm].")
    lo_in, hi_in = q0 - size / 2, q0 + size / 2
    m = max(float(spec.margin), 0.0)
    dx = float(pspec.line_width) / RES[spec.resolution]

    def inside(q):
        _, _, err = vp.locate_points(loc, q @ R)
        return err <= 1e-9
    try:
        grid = vp.generate(pspec, R, plo.copy(), lo_in - m, hi_in + m, dx, inside, plo, phi)
    except ValueError as e:
        raise VoxelError(str(e)) from None
    return _Setup(part, pi, pspec, R, plo, lo_in, hi_in, elems, loc, grid, c, auto)


# ---------------------------------------------------------------- model from voxels
def build_model(grid: vp.VoxelGrid) -> dict:
    """Nodes (global), hex8 (order of ccx: lower face anticlockwise when viewed from +3, then the upper one), indices of the voxels of the elements and indices of the nodes in the grid."""
    nx, ny, nz = grid.shape
    ijk = np.argwhere(grid.kind > 0)
    if len(ijk) == 0:
        raise VoxelError("There is no material in the detail region.")
    off = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    cor = ijk[:, None, :] + off[None]                                  # (E, 8, 3)
    lin = (cor[..., 0] * (ny + 1) + cor[..., 1]) * (nz + 1) + cor[..., 2]
    used, hexes = np.unique(lin, return_inverse=True)
    hexes = hexes.reshape(-1, 8)
    nijk = np.stack([used // ((ny + 1) * (nz + 1)), (used // (nz + 1)) % (ny + 1), used % (nz + 1)], 1)
    q = grid.origin + nijk * grid.h
    return {"ijk": ijk, "hexes": hexes, "node_ijk": nijk, "nodes": grid.to_global(q)}


def _special_triangles(study, res, pi: int, elems: np.ndarray) -> np.ndarray:
    """Surface triangles of the part that are NOT free: contact with another part (shared mesh) and the faces of constraints.
    Returns the corners (K, 3, 3).
    """
    from .region import part_surface
    mesh = res.mesh
    tri, _ = part_surface(mesh, elems)
    canon = mesh.canonical()
    key = lambda t: [tuple(r) for r in np.sort(canon[t], axis=1)]
    out = []
    other = np.nonzero(mesh.tet_part != pi)[0]
    if len(other):
        tri_o, _ = part_surface(mesh, other)
        so = set(key(tri_o))
        shared = np.array([k in so for k in key(tri)], bool)
        out.append(tri[shared])
    pnodes = set(np.unique(mesh.tets[elems]).tolist())
    for c in study.constraints:
        sel = getattr(c, "selection", None)
        if sel and sel in mesh.surfaces:
            t = mesh.surfaces[sel][:, :3]
            ok = np.array([all(int(n) in pnodes for n in r) for r in t], bool)
            out.append(t[ok])
    t = np.concatenate(out) if out else np.zeros((0, 3), np.int64)
    return mesh.nodes[t]


def _near_triangles(pts: np.ndarray, X: np.ndarray, delta: float) -> np.ndarray:
    """Mask of the points within the distance delta of some of the triangles X (K, 3, 3)."""
    if len(X) == 0 or len(pts) == 0:
        return np.zeros(len(pts), bool)
    from scipy.spatial import cKDTree
    from .region import point_triangle_distance
    cen = X.mean(axis=1)
    r = float(np.linalg.norm(X - cen[:, None], axis=2).max())
    lists = cKDTree(cen).query_ball_point(pts, delta + r)
    out = np.zeros(len(pts), bool)
    pi = np.fromiter((i for i, l in enumerate(lists) for _ in l), np.int64)
    ti = np.fromiter((t for l in lists for t in l), np.int64)
    if len(pi):
        d = point_triangle_distance(pts[pi], X[ti, 0], X[ti, 1], X[ti, 2])
        out[np.unique(pi[d <= delta])] = True
    return out


def _bc_nodes(study, res, s: _Setup, model: dict) -> tuple[np.ndarray, dict]:
    """Nodes with a prescribed displacement: the cut faces of the box (without the faces on the envelope of the part) + the contact with other parts and the constraints."""
    g = s.grid
    nijk = model["node_ijk"]
    n = np.array(g.shape)
    mask = np.zeros(len(nijk), bool)
    faces = {}
    for a, nm in enumerate("xyz"):
        for side, val in (("-", 0), ("+", n[a])):
            if nm + side in g.free_faces:
                continue
            m = nijk[:, a] == val
            faces[nm + side] = int(m.sum())
            mask |= m
    X = _special_triangles(study, res, s.pi, s.elems)
    near = _near_triangles(model["nodes"], X, 0.75 * float(g.h.max()))
    info = {"faces": faces, "interface_or_support": int((near & ~mask).sum())}
    return np.nonzero(mask | near)[0], info


def _drop_floating(grid: vp.VoxelGrid, model: dict, bc: np.ndarray) -> tuple[dict, int]:
    """Removes islands of voxels (connected by a face) with fewer than 4 nodes with a prescribed displacement (free bodies)."""
    from scipy.ndimage import label
    lab, nlab = label(grid.kind > 0)
    el_lab = lab[tuple(model["ijk"].T)]
    nn = len(model["nodes"])
    pairs = np.unique(np.repeat(el_lab, 8).astype(np.int64) * nn + model["hexes"].ravel())
    isbc = np.zeros(nn, bool)
    isbc[bc] = True
    cnt = np.bincount((pairs // nn)[isbc[pairs % nn]], minlength=nlab + 1)
    keep = (cnt >= 4)[el_lab]
    dropped = int((~keep).sum())
    if dropped:
        grid.kind[tuple(model["ijk"][~keep].T)] = vp.EMPTY
        model = build_model(grid)
    return model, dropped


def _angle_bins(angle: np.ndarray, nb: int) -> np.ndarray:
    return np.mod(np.rint(np.mod(angle, np.pi) / (np.pi / nb)).astype(np.int64), nb)


def bin_axes(R: np.ndarray, nb: int) -> np.ndarray:
    """Axes of the material of the classes of the angle (nb, 3, 3): rows 1 = line at the angle k·π/nb from the print axis 1, 2 = across, 3 = build."""
    from . import line_axes
    return np.stack([line_axes(R, k * np.pi / nb) for k in range(nb)])


# ---------------------------------------------------------------- estimate
# measured (notebook 8 cores, in parallel with another load): full block C3D8I 75 k voxels 914 k equations 106 s / 4.0 GB
# (scratch/voxel/exp2.py); example 1 (3 states): 81 k equations 9.9 s / 0.18 GB, 216 k 27 s / 0.6 GB, 305 k 40 s /
# 0.86 GB; bar X1 (1 state) 559 k equations 44 s. C3D8 75 k voxels 239 k equations 83 s / 3.3 GB (assembly by elements).
def _cost(dof: int, nel: int, steps: int, element: str) -> tuple[float, float]:
    x = max(dof, 1) / 1e6
    if element == "C3D8":
        x = max(nel * 12, 1) / 1e6                 # time and memory of ccx for C3D8 go mainly into the assembly by elements
    mem = 0.1 + 2.2 * x + 2.5 * x * x
    t = 2.0 + 80.0 * x * (x / 0.5) ** 0.2 * (1 + 0.25 * (steps - 1)) + 2e-5 * nel * steps
    return mem, t


def estimate(study, results, spec: VoxelSpec) -> dict:
    """Size of the voxel detail in advance: voxels by kinds, nodes, equations, estimate of memory and time, ok + message."""
    t0 = time.time()
    s = _setup(study, results, spec)
    model = build_model(s.grid)
    return _estimate_of(s, model, spec, len(_steps(results, spec.step)), time.time() - t0)


def _estimate_of(s: _Setup, model: dict, spec: VoxelSpec, nsteps: int, t_gen: float) -> dict:
    nel, nn = len(model["hexes"]), len(model["nodes"])
    dof = 3 * nn + (9 * nel if spec.element == "C3D8I" else 0)
    mem, t = _cost(dof, nel, nsteps, spec.element)
    try:
        import psutil
        free = psutil.virtual_memory().available / 1e9
    except Exception:  # noqa: BLE001
        free = float("inf")
    ok, msg = True, ""
    if dof > spec.max_dof:
        ok, msg = False, (f"The voxel detail has {dof:,} equations (limit {spec.max_dof:,}) – reduce the region or margin, or choose resolution w/2.").replace(",", " ")
    elif mem > 0.8 * free:
        ok, msg = False, f"The ccx memory estimate of {mem:.1f} GB exceeds 80 % of free RAM ({free:.1f} GB) – reduce the region."
    g = s.grid
    return {"voxels": int(nel), "nodes": int(nn), "dof": int(dof), "element": spec.element,
            "memory_gb": round(mem, 2), "time_s": round(t + t_gen, 1), "steps": nsteps, "ok": ok, "message": msg,
            "part": s.part, "center": s.center.tolist(), "shape": list(g.shape), "voxel_mm": g.h.tolist(),
            "kinds": g.counts(), "auto": s.auto}


# ---------------------------------------------------------------- writing ccx
def _fmt(v: float) -> str:
    return f"{v:.12g}"


def write_inp(path: Path, nodes: np.ndarray, hexes: np.ndarray, bins: np.ndarray, axes: np.ndarray, C: np.ndarray,
              bc_nodes: np.ndarray, bc_values: list[np.ndarray], element: str = "C3D8I",
              bc_dofs: np.ndarray | None = None, step_names: list[str] | None = None) -> None:
    """.inp of the voxel model: one material (C in the axes of the line), classes of the angle = ELSET + *ORIENTATION + *SOLID SECTION,
    each state = *STEP with *BOUNDARY (prescribed displacements bc_values[k] (B, 3) in the nodes bc_nodes; bc_dofs (B, 3) bool =
    which components, None = all).
    """
    from ..ccx.inp import _write_ids
    d = [C[0, 0], C[0, 1], C[1, 1], C[0, 2], C[1, 2], C[2, 2], C[3, 3], C[4, 4], C[5, 5]]
    dofs = np.ones((len(bc_nodes), 3), bool) if bc_dofs is None else np.asarray(bc_dofs, bool)
    bi, bd = np.nonzero(dofs)
    with open(path, "w", encoding="ascii") as f:
        f.write("** FEA Studio - voxelovy detail 3D tisku\n*NODE, NSET=NALL\n")
        np.savetxt(f, np.column_stack([np.arange(1, len(nodes) + 1), nodes]), fmt="%d, %.12g, %.12g, %.12g")
        f.write(f"*ELEMENT, TYPE={element}, ELSET=EALL\n")
        np.savetxt(f, np.column_stack([np.arange(1, len(hexes) + 1), hexes + 1]), fmt="%d", delimiter=", ")
        f.write("*MATERIAL, NAME=ML\n*ELASTIC, TYPE=ORTHO\n" + ", ".join(_fmt(v) for v in d[:8])
                + f",\n{_fmt(d[8])}, 0.\n")
        for k in np.unique(bins):
            A = axes[k]
            f.write(f"*ELSET, ELSET=B{k}\n")
            _write_ids(f, np.nonzero(bins == k)[0] + 1)
            f.write(f"*ORIENTATION, NAME=OB{k}, SYSTEM=RECTANGULAR\n" + ", ".join(_fmt(v) for v in (*A[0], *A[1]))
                    + f"\n*SOLID SECTION, ELSET=B{k}, MATERIAL=ML, ORIENTATION=OB{k}\n")
        for si, val in enumerate(bc_values):
            nm = deck_text(step_names[si] if step_names else f"state {si + 1}")
            f.write(f"** {nm}\n*STEP\n*STATIC, SOLVER={solver_name()}\n*BOUNDARY{', OP=NEW' if si else ''}\n")
            v = np.asarray(val)[bi, bd]
            np.savetxt(f, np.column_stack([bc_nodes[bi] + 1, bd + 1, bd + 1, v]), fmt="%d, %d, %d, %.12g")
            f.write("*NODE FILE\nU\n*EL FILE\nS\n*EL PRINT, ELSET=EALL, GLOBAL=YES\nS\n*END STEP\n")


def interpolate_U(res, loc_elems: np.ndarray, loc: dict, pts: np.ndarray, step: int) -> np.ndarray:
    """Displacements of the global solution at the points pts by the shape functions of tet10 of the element (part) with the point (slightly outside = extrapolation)."""
    from ..results import tet10_shape
    el, xi, _ = vp.locate_points(loc, pts)
    conn = res.mesh.tets[loc_elems[el]]
    U = res.steps[step].U
    if conn.shape[1] == 10:
        N = tet10_shape(xi)
    else:
        N = np.column_stack([1 - xi.sum(axis=1), xi])
    return np.einsum("pa,paj->pj", N, U[conn])


def to_local(S: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Stresses (…, 6) globally in the order of ccx xx yy zz xy yz zx → axes A (…, 3, 3) (rows), order of failure:
    11 22 33 12 13 23.
    """
    T = np.empty(S.shape[:-1] + (3, 3))
    T[..., 0, 0], T[..., 1, 1], T[..., 2, 2] = S[..., 0], S[..., 1], S[..., 2]
    T[..., 0, 1] = T[..., 1, 0] = S[..., 3]
    T[..., 1, 2] = T[..., 2, 1] = S[..., 4]
    T[..., 0, 2] = T[..., 2, 0] = S[..., 5]
    L = np.einsum("...ai,...ij,...bj->...ab", A, T, A)
    return np.stack([L[..., 0, 0], L[..., 1, 1], L[..., 2, 2], L[..., 0, 1], L[..., 0, 2], L[..., 1, 2]], axis=-1)


def solve_voxels(workdir: Path, nodes, hexes, bins, axes, C, bc_nodes, bc_values, element="C3D8I", bc_dofs=None,
                 step_names=None, on_log=None, cancel=None, threads=None, jobname="voxel") -> list[dict]:
    """Writing, ccx calculation and reading: by states {"U" (N, 3), "S" (N, 6) nodal globally, "S_ip" (E, 8, 6) globally}."""
    from ..ccx.frd import read_frd
    from ..ccx.runner import run_ccx, CcxError
    from ..joints.forces import read_el_print
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    for ext in (".frd", ".dat", ".sta", ".cvg"):
        (workdir / f"{jobname}{ext}").unlink(missing_ok=True)
    log = on_log or (lambda m: None)
    write_inp(workdir / f"{jobname}.inp", nodes, hexes, bins, axes, C, bc_nodes, bc_values, element, bc_dofs,
              step_names)
    t0 = time.time()
    peak = _PeakMemory()
    try:
        run = run_ccx(workdir, jobname, cancel=cancel, threads=threads)
    except CcxError as e:
        raise VoxelError(str(e)) from None
    finally:
        peak.stop()
    t_ccx = time.time() - t0
    if not run.finished or run.errors:
        raise VoxelError("ccx (voxel detail) ended with an error: " + "; ".join(run.errors[:3] or [f"code {run.returncode}"]))
    log(f"Voxel detail: ccx {t_ccx:.1f} s, peak memory {peak.gb:.2f} GB")
    t1 = time.time()
    frd = read_frd(workdir / f"{jobname}.frd")
    out = []
    ep = read_el_print(workdir / f"{jobname}.dat", "EALL")
    E = len(hexes)
    for si in range(len(bc_values)):
        d = frd.get("DISP", si + 1)
        U = np.zeros((len(nodes), 3))
        U[d.node_ids - 1] = d.values[:, :3]
        sb = frd.get("STRESS", si + 1)
        S = np.zeros((len(nodes), 6))
        S[sb.node_ids - 1] = sb.values[:, :6]
        e = ep[si]
        S_ip = np.stack([e[i + 1] for i in range(E)])
        out.append({"U": U, "S": S, "S_ip": S_ip})
    stats = {"ccx_s": round(t_ccx, 2), "read_s": round(time.time() - t1, 2), "ccx_peak_gb": round(peak.gb, 2)}
    for r in out:
        r["stats"] = stats
    return out


class _PeakMemory:
    """Peak of memory of the children (ccx) in GB – a thread, a query every 0.25 s (psutil; without psutil 0)."""
    def __init__(self):
        import threading
        self.gb = 0.0
        self._stop = threading.Event()
        try:
            import psutil
            self._me = psutil.Process()
        except Exception:  # noqa: BLE001
            return
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop.wait(0.25):
            try:
                rss = sum(c.memory_info().rss for c in self._me.children(recursive=True))
            except Exception:  # noqa: BLE001
                continue
            self.gb = max(self.gb, rss / 1e9)

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------- results
@dataclass
class VoxelStep:
    name: str
    U: np.ndarray            # (N, 3) mm
    S: np.ndarray            # (N, 6) MPa nodal (ccx), globally xx yy zz xy yz zx
    S_el: np.ndarray         # (E, 6) average of the integration points, globally
    S_loc: np.ndarray        # (E, 6) in the axes of the bead (11 22 33 12 13 23) at the deciding integration point
    safety_el: np.ndarray    # (E,) safety of the element (minimum over the integration points)
    safety: np.ndarray       # (N,) nodal = minimum of the elements around the node
    safety_avg: np.ndarray | None = None   # (E,) safety from the average stress of the element (less sensitive to the steps of the voxels)
    creep: dict | None = None              # long-term state: stiffness and strength factors (printing.creep)


@dataclass
class VoxelResults:
    nodes: np.ndarray        # (N, 3) globally
    hexes: np.ndarray        # (E, 8) 0-based
    ijk: np.ndarray          # (E, 3) index of the voxel in the grid
    kind: np.ndarray         # (E,) kind of the voxel (voxel_paths: 1 perimeter, 2 layer, 3 infill, 4 thin place)
    angle: np.ndarray        # (E,) angle of the line [rad] from the print axis 1
    wall: np.ndarray         # (E,) number of the perimeter (−1 elsewhere)
    bins: np.ndarray         # (E,) class of the angle
    axes: np.ndarray         # (B, 3, 3) axes of the material of the classes (rows: line, across, build)
    inner: np.ndarray        # (E,) element in the region of the evaluation (without the margin)
    R: np.ndarray
    origin: np.ndarray
    h: np.ndarray
    shape: tuple
    bc_nodes: np.ndarray
    steps: list[VoxelStep] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def step(self, key: int | str = 0) -> VoxelStep:
        if isinstance(key, int):
            return self.steps[key]
        return next(s for s in self.steps if s.name == key)

    def element_centers(self) -> np.ndarray:
        return self.nodes[self.hexes].mean(axis=1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        arr = {k: getattr(self, k) for k in ("nodes", "hexes", "ijk", "kind", "angle", "wall", "bins", "axes", "inner",
                                              "R", "origin", "h", "bc_nodes")}
        arr["shape"] = np.array(self.shape)
        for i, s in enumerate(self.steps):
            for k in _STEP_ARRAYS:
                if getattr(s, k) is not None:
                    arr[f"s{i}_{k}"] = getattr(s, k)
        np.savez(path.with_suffix(".npz"), **arr)
        meta = {"steps": [s.name for s in self.steps], "creep": [s.creep for s in self.steps],
                "summary": self.summary, "warnings": self.warnings}
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float),
                                             encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VoxelResults":
        path = Path(path)
        z = np.load(path.with_suffix(".npz"), allow_pickle=False)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        vr = cls(*(z[k] for k in ("nodes", "hexes", "ijk", "kind", "angle", "wall", "bins", "axes", "inner", "R",
                                  "origin", "h")), shape=tuple(int(v) for v in z["shape"]), bc_nodes=z["bc_nodes"],
                 summary=meta["summary"], warnings=meta["warnings"])
        creep = meta.get("creep") or [None] * len(meta["steps"])
        for i, nm in enumerate(meta["steps"]):
            vr.steps.append(VoxelStep(nm, *(z[f"s{i}_{k}"] if f"s{i}_{k}" in z else None for k in _STEP_ARRAYS),
                                      creep=creep[i]))
        return vr


_STEP_ARRAYS = ("U", "S", "S_el", "S_loc", "safety_el", "safety", "safety_avg")


def creep_scaled(ml, fac: dict | None):
    """Material of the long-term state: strengths × k_R (creep rupture). The stiffness × k_s is projected into the stress by multiplication
    (the boundary conditions are only displacements → the displacements of the voxel model do not depend on k_s, stress = k_s·σ).
    """
    if not fac:
        return ml, 1.0
    from dataclasses import replace
    kr = float(fac["strength"])
    upd = {k: getattr(ml, k) * kr for k in ("Xt", "Xc", "Zt", "Zc", "Sxy", "Sz", "Yt", "Yc")
           if getattr(ml, k) is not None}
    return replace(ml, **upd), float(fac["stiffness"])


def evaluate_steps(raw: list[dict], names: list[str], bins: np.ndarray, axes: np.ndarray, ml, criterion: str,
                   hexes: np.ndarray, n_nodes: int, creep: list | None = None) -> list[VoxelStep]:
    """Stresses in the axes of the bead and the safety per element (minimum over the integration points; safety_avg from the average stress of the
    element) and in the nodes. creep[k] = factors of the long-term state k (stiffness, strength) or None.
    """
    from .failure import safety
    A = axes[bins]                                                  # (E, 3, 3)
    out = []
    for k, (nm, r) in enumerate(zip(names, raw)):
        fac = creep[k] if creep else None
        m, ks = creep_scaled(ml, fac)
        Sip = r["S_ip"] * ks
        Sl = to_local(Sip, A[:, None])                              # (E, 8, 6)
        lam_ip = safety(Sl, m, criterion)
        j = np.argmin(lam_ip, axis=1)
        e = np.arange(len(Sip))
        lam = lam_ip[e, j]
        lam_avg = safety(Sl.mean(axis=1), m, criterion)
        node = np.full(n_nodes, np.inf)
        np.minimum.at(node, hexes.ravel(), np.repeat(lam, 8))
        out.append(VoxelStep(nm, r["U"], r["S"] * ks, Sip.mean(axis=1), Sl[e, j], lam, node, lam_avg,
                             dict(fac) if fac else None))
    return out


def run_voxel(study, results, spec: VoxelSpec, workdir: str | Path, on_log: Callable[[str], None] | None = None,
              cancel=None) -> VoxelResults:
    """Voxel detail of the printing: generation of the paths, boundary conditions from the global result, ccx C3D8I, Tsai-Wu/Hill."""
    from . import part_materials
    log = on_log or (lambda m: None)
    t0 = time.time()
    timing = {}
    s = _setup(study, results, spec)
    timing["drahy_s"] = round(time.time() - t0, 2)
    g = s.grid
    model = build_model(g)
    bc, bc_info = _bc_nodes(study, results, s, model)
    model, dropped = _drop_floating(g, model, bc)
    if dropped:
        bc, bc_info = _bc_nodes(study, results, s, model)
    steps = _steps(results, spec.step)
    est = _estimate_of(s, model, spec, len(steps), 0.0)
    log(f"Voxel detail '{s.part}': {est['voxels']} voxels, {est['dof']} equations, estimate {est['memory_gb']} GB, {est['time_s']:.0f} s (preparation {time.time() - t0:.1f} s)")
    if not est["ok"]:
        raise VoxelError(est["message"])
    if len(bc) == 0:
        raise VoxelError("The detail region has no cut faces or contact with other parts – the whole part lies inside the region; reduce the region (the detail needs global-solution displacements on its boundary).")
    warnings = list(g.warnings)
    if dropped:
        warnings.append(f"Omitted {dropped} voxels of islands not connected to the region boundary.")
    warnings += _load_warnings(study, results, s)
    _, ml, _ = part_materials(study, s.part)
    nb = int(spec.angle_bins or study.printing.line_bins or 36)
    ijk = model["ijk"]
    angle = g.angle[tuple(ijk.T)]
    bins = _angle_bins(angle, nb)
    axes = bin_axes(s.R, nb)
    bcx = model["nodes"][bc]
    values = [interpolate_U(results, s.elems, s.loc, bcx, si) for si in steps]
    names = [results.steps[si].name for si in steps]
    timing["priprava_s"] = round(time.time() - t0, 2)
    t1 = time.time()
    raw = solve_voxels(Path(workdir), model["nodes"], model["hexes"], bins, axes, ml.stiffness(), bc, values,
                       spec.element, step_names=names, on_log=log, cancel=cancel, threads=spec.threads)
    timing["ccx_a_cteni_s"] = round(time.time() - t1, 2)
    timing.update(raw[0]["stats"])
    mname = study.material_of(s.part).name
    creep = [((results.steps[si].printing or {}).get("creep") or {}).get("factors", {}).get(mname) for si in steps]
    vsteps = evaluate_steps(raw, names, bins, axes, ml, study.printing.criterion, model["hexes"], len(model["nodes"]),
                            creep)
    q_el = g.origin + (ijk + 0.5) * g.h
    inner = np.all((q_el >= s.lo_in - 1e-9) & (q_el <= s.hi_in + 1e-9), axis=1)
    vr = VoxelResults(model["nodes"], model["hexes"], ijk, g.kind[tuple(ijk.T)].astype(np.int8), angle,
                      g.wall[tuple(ijk.T)].astype(np.int8), bins, axes, inner, s.R, g.origin, g.h, g.shape, bc,
                      vsteps, warnings=warnings)
    vr.summary = _summary(study, results, s, vr, steps, est, bc_info, time.time() - t0)
    vr.summary["timing"] = timing
    vr.summary["estimate"] = {k: est[k] for k in ("memory_gb", "time_s")}
    log(f"Voxel detail finished in {time.time() - t0:.1f} s")
    return vr


def _load_warnings(study, res, s: _Setup) -> list[str]:
    """Loads that are not transferred to the detail (forces on the faces in the region, body and thermal loads)."""
    from ..study import Gravity, Rotation
    out = []
    mesh = res.mesh
    lo, hi = s.grid.origin, s.grid.origin + np.array(s.grid.shape) * s.grid.h
    names = set()
    for lc in study.load_cases:
        for l in lc.loads:
            if isinstance(l, (Gravity, Rotation)):
                names.add("body loads (gravity, rotation)")
                continue
            if getattr(l, "kind", "") == "thermal" or type(l).__name__ == "ThermalLoad":
                names.add("thermal load")
                continue
            sel = getattr(l, "selection", None)
            if sel and sel in mesh.surfaces:
                c = mesh.nodes[mesh.surfaces[sel][:, :3]].mean(axis=1) @ s.R.T
                if np.any(np.all((c >= lo) & (c <= hi), axis=1)):
                    names.add(f"load on face '{sel}' in the detail region")
    if names:
        out.append("The voxel detail takes only the displacements at the boundary of the region – it does not include: " + ", ".join(sorted(names))
                   + ". The result there is only approximate (move or enlarge the region).")
    return out


def _summary(study, res, s: _Setup, vr: VoxelResults, steps: list[int], est: dict, bc_info: dict, dt: float) -> dict:
    mesh = res.mesh
    pn = np.unique(mesh.tets[s.elems])
    q = mesh.nodes[pn] @ s.R.T
    gin = pn[np.all((q >= s.lo_in) & (q <= s.hi_in), axis=1)]
    vol = float(np.prod(vr.h))
    g = s.grid
    # centres of all voxels of the part in the region of the evaluation (also the empty ones – the average of the stress over the volume of the region)
    allin = np.argwhere(g.inside)
    qa = g.origin + (allin + 0.5) * g.h
    allin = allin[np.all((qa >= s.lo_in) & (qa <= s.hi_in), axis=1)]
    out = {"part": s.part, "center": s.center.tolist(), "size": (s.hi_in - s.lo_in).tolist(), "auto": s.auto,
           "voxel_mm": vr.h.tolist(), "shape": list(vr.shape), "voxels": int(len(vr.hexes)),
           "nodes": int(len(vr.nodes)), "dof": est["dof"], "element": est["element"], "bc_nodes": int(len(vr.bc_nodes)),
           "bc": bc_info, "free_faces": g.free_faces, "kinds": {vp.KIND_LABELS[k]: int((vr.kind == k).sum())
                                                                for k in (1, 2, 3, 4)},
           "infill_fraction": vp.infill_fraction(g), "time_s": round(dt, 1), "criterion": study.printing.criterion,
           "steps": []}
    xin = g.to_global(g.origin + (allin + 0.5) * g.h)
    for si, st in zip(steps, vr.steps):
        lam = np.where(vr.inner, st.safety_el, np.inf)
        k = int(np.argmin(lam))
        d = {"name": st.name, "min_safety": float(lam[k]), "at": vr.element_centers()[k].tolist(),
             "kind": vp.KIND_LABELS[int(vr.kind[k])], "wall": int(vr.wall[k]),
             "angle_deg": float(np.degrees(vr.angle[k])), "stress_local": st.S_loc[k].tolist(),
             "max_displacement_mm": float(np.linalg.norm(st.U, axis=1).max())}
        # more robust measures: the steps of the voxels and the corners of the beads give local peaks at the integration points
        la = np.where(vr.inner, st.safety_avg, np.inf)
        d["min_safety_element_avg"] = float(la.min())
        v_in = st.safety_el[vr.inner]
        d["safety_p1"] = float(np.percentile(v_in, 1)) if len(v_in) else float("nan")
        if st.creep:
            d["creep"] = st.creep
        ps = res.steps[si].fields.get("print_safety")
        if ps is not None:
            nd = gin if len(gin) else pn[[np.argmin(np.linalg.norm(mesh.nodes[pn] - s.center, axis=1))]]
            nd = nd[np.isfinite(ps[nd])]
            if len(nd):
                j = int(nd[np.argmin(ps[nd])])
                d["homogenized_min_safety"] = float(ps[j])
                d["homogenized_at"] = mesh.nodes[j].tolist()
                d["ratio"] = float(d["min_safety"] / ps[j])
        # average of the stress over the volume of the region of the evaluation: voxels (empty = 0) × the homogenised model at the same points
        if len(allin):
            Vbox = len(allin) * vol
            sv = (st.S_el[vr.inner] * vol).sum(axis=0) / Vbox
            sg = _probe_S(res, s, xin, si).mean(axis=0)
            d["mean_stress_voxel"] = sv.tolist()
            d["mean_stress_global"] = sg.tolist()
            nrm = max(float(np.linalg.norm(sg)), 1e-12)
            d["mean_stress_difference"] = float(np.linalg.norm(sv - sg) / nrm)
        out["steps"].append(d)
    return out


def _probe_S(res, s: _Setup, pts: np.ndarray, step: int) -> np.ndarray:
    from ..results import tet10_shape
    el, xi, _ = vp.locate_points(s.loc, pts)
    conn = res.mesh.tets[s.elems[el]]
    N = tet10_shape(xi) if conn.shape[1] == 10 else np.column_stack([1 - xi.sum(axis=1), xi])
    return np.einsum("pa,paj->pj", N, res.steps[step].S[conn])


# ---------------------------------------------------------------- picture
KIND_COLORS = {0: "#ffffff", 1: "#d9534f", 2: "#f0ad4e", 3: "#5b9bd5", 4: "#8e6bbf"}


def _field(vr: VoxelResults, field_name: str, step: int):
    """Value per element, description, (vmin, vmax) and the colour map (None = kinds of voxels)."""
    if field_name == "kind":
        return vr.kind.astype(float), "Voxel type", None
    st = vr.steps[step]
    if field_name == "safety":
        v = st.safety_el.astype(float).copy()
        fin = np.isfinite(v)
        lo = float(v[fin].min()) if fin.any() else 1.0
        v = np.minimum(np.where(fin, v, np.inf), max(3.0 * lo, lo + 1.0))     # rozsah min … 3·min (kontrast)
        return v, f"Safety factor ({('Tsai-Hill' if vr.summary.get('criterion') == 'tsai_hill' else 'Tsai-Wu')})", "RdYlGn"
    if field_name == "angle":
        return np.degrees(vr.angle), "Line angle [°]", "twilight"
    if field_name == "mises":
        from ..results import von_mises
        return von_mises(st.S_el), "von Mises [MPa]", "jet"
    if field_name == "umag":
        return np.linalg.norm(st.U[vr.hexes].mean(axis=1), axis=1), "Displacement [mm]", "jet"
    if field_name in ("s11", "s22", "s33", "t12", "t13", "t23"):
        i = ["s11", "s22", "s33", "t12", "t13", "t23"].index(field_name)
        return st.S_loc[:, i], f"σ in bead axes {field_name} [MPa]", "coolwarm"
    raise KeyError(f"Unknown field '{field_name}'")


def render_png(vr: VoxelResults, path: str | Path, field: str = "kind", slice_z: int | None = None, step: int = 0,
               dpi: int = 110, lines: bool = False) -> Path:
    """PNG: a section by the layer slice_z (index of the layer in the region) or an isometric view of the surface of the voxels.
    field = kind | safety | angle | mises | umag | s11 … t23 (stresses in the axes of the bead); lines = short dashes in the section
    in the direction of the beads (every 2nd voxel).
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import colors as mcolors, cm
    path = Path(path)
    val, label, cmap = _field(vr, field, step)
    fig = Figure(figsize=(7.5, 6.0), dpi=dpi)
    FigureCanvasAgg(fig)
    if cmap is None:
        ck = np.array([mcolors.to_rgba(KIND_COLORS[int(k)]) for k in vr.kind])
        norm = None
    else:
        lo, hi = float(np.nanmin(val)), float(np.nanmax(val))
        if hi <= lo:
            hi = lo + 1e-9
        norm = mcolors.Normalize(lo, hi)
        import matplotlib
        ck = matplotlib.colormaps[cmap](norm(val))
    nx, ny, nz = vr.shape
    if slice_z is not None:
        ax = fig.add_subplot(111)
        k = int(slice_z)
        img = np.ones((ny, nx, 4))
        sel = vr.ijk[:, 2] == k
        img[vr.ijk[sel, 1], vr.ijk[sel, 0]] = ck[sel]
        x0, y0 = vr.origin[0], vr.origin[1]
        ax.imshow(img, origin="lower", extent=[x0, x0 + nx * vr.h[0], y0, y0 + ny * vr.h[1]], interpolation="nearest")
        ax.set_xlabel("print axis 1 [mm]")
        ax.set_ylabel("print axis 2 [mm]")
        z = vr.origin[2] + (k + 0.5) * vr.h[2]
        ax.set_title(f"{label} – layer {k} (print z {z:.2f} mm)")
        if lines:
            ss = sel & (vr.ijk[:, 0] % 2 == 0) & (vr.ijk[:, 1] % 2 == 0)
            c = vr.origin[:2] + (vr.ijk[ss, :2] + 0.5) * vr.h[:2]
            a = vr.angle[ss]
            ax.quiver(c[:, 0], c[:, 1], np.cos(a), np.sin(a), angles="xy", scale_units="xy", scale=1 / vr.h[0],
                      headwidth=0, headlength=0, headaxislength=0, pivot="middle", width=0.002, color="k")
    else:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax = fig.add_subplot(111, projection="3d")
        occ = np.zeros((nx + 2, ny + 2, nz + 2), bool)
        occ[tuple((vr.ijk + 1).T)] = True
        quads, cols = [], []
        q0 = vr.origin
        base = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        for a in range(3):
            b, c = [d for d in range(3) if d != a]
            for sgn in (-1, 1):
                nb = vr.ijk + 1
                nb = nb.copy()
                nb[:, a] += sgn
                exposed = ~occ[tuple(nb.T)]
                ids = np.nonzero(exposed)[0]
                if not len(ids):
                    continue
                corner = np.zeros((len(ids), 4, 3))
                corner[:, :, a] = vr.ijk[ids, a][:, None] + (1 if sgn > 0 else 0)
                corner[:, :, b] = vr.ijk[ids, b][:, None] + base[None, :, 0]
                corner[:, :, c] = vr.ijk[ids, c][:, None] + base[None, :, 1]
                quads.append(q0 + corner * vr.h)
                shade = 0.75 + 0.25 * (a == 2) * (sgn > 0) + 0.1 * (a == 0)
                cc = ck[ids].copy()
                cc[:, :3] *= min(shade, 1.0)
                cols.append(cc)
        if quads:
            Q = np.concatenate(quads)
            pc = Poly3DCollection(Q, facecolors=np.concatenate(cols), edgecolors="none", linewidths=0)
            ax.add_collection3d(pc)
            lo, hi = Q.reshape(-1, 3).min(axis=0), Q.reshape(-1, 3).max(axis=0)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect(hi - lo)
        ax.view_init(elev=28, azim=-55)
        ax.set_xlabel("1 [mm]")
        ax.set_ylabel("2 [mm]")
        ax.set_zlabel("build [mm]")
        ax.set_title(f"{label} – voxel surface (print axes)")
    if cmap is None:
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=KIND_COLORS[k], label=vp.KIND_LABELS[k]) for k in (1, 2, 3, 4)
                            if (vr.kind == k).any()], loc="lower center", ncol=4, fontsize=8, frameon=False)
    else:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, shrink=0.75, label=label)
    fig.savefig(path)
    return path
