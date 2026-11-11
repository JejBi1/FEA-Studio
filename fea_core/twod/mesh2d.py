"""2D tri6 mesh (Gmsh in a separate process – fea_core.twod.worker) and integration over the elements and the edges.

The node order of tri6 in Gmsh = ccx (corners 1–3, midpoints 1-2, 2-3, 3-1). ccx wants the elements anticlockwise (positive
jacobian, otherwise "*ERROR in e_c3d: nonpositive jacobian" – verified scratch/twod/exp1.py) → the orientation is corrected here.
Faces of the element (ccx *DLOAD Pk): 1 = nodes 1-2, 2 = 2-3, 3 = 3-1. The outward normal of the face CCW of the element = the tangent rotated by −90°.

Weight of the integration w(x): axisym 2π·r (whole circumference), planar problems thickness t.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import config

# face k (0-based) of the tri6 element: (corner a, corner b, midpoint)
FACE_NODES = np.array([[0, 1, 3], [1, 2, 4], [2, 0, 5]])
FLIP = [0, 2, 1, 5, 4, 3]
SUBTRI = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])

# Dunavant 6 points (degree 4) – (L2, L3), weights on the reference triangle (sum 1/2)
_A1, _B1, _W1 = 0.445948490915965, 0.108103018168070, 0.223381589678011
_A2, _B2, _W2 = 0.091576213509771, 0.816847572980459, 0.109951743655322
TRI_GP = np.array([[_A1, _A1], [_B1, _A1], [_A1, _B1], [_A2, _A2], [_B2, _A2], [_A2, _B2]])
TRI_GW = np.array([_W1] * 3 + [_W2] * 3) / 2.0
LINE_GP, LINE_GW = np.polynomial.legendre.leggauss(4)


class Mesh2DError(RuntimeError):
    pass


def tri6_shape(l2, l3):
    """Shape functions of tri6 and their derivatives with respect to (L2, L3): N (…, 6), dN (…, 2, 6)."""
    l2, l3 = np.asarray(l2, float), np.asarray(l3, float)
    l1 = 1.0 - l2 - l3
    N = np.stack([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1), 4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1], -1)
    z = np.zeros_like(l1)
    d2 = np.stack([-(4 * l1 - 1), 4 * l2 - 1, z, 4 * (l1 - l2), 4 * l3, -4 * l3], -1)
    d3 = np.stack([-(4 * l1 - 1), z, 4 * l3 - 1, -4 * l2, 4 * l2, 4 * (l1 - l3)], -1)
    return N, np.stack([d2, d3], -2)


def line3_shape(s):
    """Shape functions of line3 (ends a, b, midpoint) in s ∈ ⟨−1, 1⟩: N (…, 3), dN/ds (…, 3)."""
    s = np.asarray(s, float)
    N = np.stack([0.5 * s * (s - 1), 0.5 * s * (s + 1), 1 - s * s], -1)
    dN = np.stack([s - 0.5, s + 0.5, -2 * s], -1)
    return N, dN


@dataclass
class Mesh2D:
    nodes: np.ndarray                 # (N, 2) mm
    tri: np.ndarray                   # (M, 6) ccx order, CCW
    tri_part: np.ndarray              # (M,)
    part_names: list[str]
    seg: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))      # (K, 3) boundary line3 (a, b, midpoint) oriented according to the element
    seg_elem: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))      # (K,) element of the face
    seg_face: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))      # (K,) face 1..3
    seg_curve: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))     # (K,) boundary curve
    curve_pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 7, 2)))       # (C, 7, 2) sampling points of the curves
    report: dict = field(default_factory=dict)

    def mean_edge(self) -> float:
        X = self.nodes[self.tri[:, :3]]
        return float(np.mean(np.linalg.norm(X - X[:, [1, 2, 0]], axis=2)))

    def size(self) -> float:
        return float(np.linalg.norm(self.nodes.max(axis=0) - self.nodes.min(axis=0))) if len(self.nodes) else 1.0

    def arrays(self, prefix: str = "mesh_") -> dict:
        return {prefix + k: getattr(self, k) for k in ("nodes", "tri", "tri_part", "seg", "seg_elem", "seg_face",
                                                         "seg_curve", "curve_pts")}

    @classmethod
    def from_arrays(cls, d: dict, part_names: list[str], report: dict, prefix: str = "mesh_") -> "Mesh2D":
        kw = {k: d[prefix + k] for k in ("nodes", "tri", "tri_part", "seg", "seg_elem", "seg_face", "seg_curve",
                                         "curve_pts")}
        return cls(part_names=list(part_names), report=dict(report), **kw)


def orient(nodes: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, int]:
    """Elements anticlockwise (positive area of the corner triangle)."""
    x = nodes[tri[:, :3]]
    a = (x[:, 1, 0] - x[:, 0, 0]) * (x[:, 2, 1] - x[:, 0, 1]) - (x[:, 2, 0] - x[:, 0, 0]) * (x[:, 1, 1] - x[:, 0, 1])
    bad = a < 0
    out = tri.copy()
    out[bad] = tri[bad][:, FLIP]
    return out, int(bad.sum())


def boundary_lookup(tri: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For the boundary segments (a, b, midpoint) finds (element, face 1..3) and returns the segments oriented according to the element."""
    if len(seg) == 0:
        return seg.reshape(0, 3), np.zeros(0, np.int64), np.zeros(0, np.int64)
    n = int(max(tri.max(), seg.max())) + 1
    a = tri[:, FACE_NODES[:, 0]].T.ravel()                 # face k of element e → index k·M + e
    b = tri[:, FACE_NODES[:, 1]].T.ravel()
    keys = np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    q = np.minimum(seg[:, 0], seg[:, 1]).astype(np.int64) * n + np.maximum(seg[:, 0], seg[:, 1])
    pos = np.clip(np.searchsorted(sk, q), 0, len(sk) - 1)
    if np.any(sk[pos] != q):
        raise Mesh2DError("A mesh boundary segment does not belong to any element")
    fid = order[pos]
    M = len(tri)
    elem, k = fid % M, fid // M
    return tri[elem[:, None], FACE_NODES[k]], elem.astype(np.int64), (k + 1).astype(np.int64)


def generate_mesh2d(sections, settings: dict, workdir: str | Path, cancel=None, timeout: float | None = None,
                    axis_clamp: bool = False) -> Mesh2D:
    """Mesh of the sections (SectionPart with brep) in a separate Gmsh process. axis_clamp: nodes at the axis (|x| < 1e-9·size) → x = 0."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    plist = [{"name": s.name, "brep": s.brep} for s in sections if s.brep]
    if not plist:
        raise Mesh2DError("The section plane does not intersect any part")
    job = {"parts": plist, "settings": settings, "output": str(workdir / "mesh2d_raw.npz"),
           "report": str(workdir / "mesh2d_report.json")}
    jp = workdir / "mesh2d_job.json"
    jp.write_text(json.dumps(job, indent=1, ensure_ascii=False), encoding="utf-8")
    for old in (job["output"], job["report"]):
        Path(old).unlink(missing_ok=True)
    log = workdir / "mesh2d_worker.log"
    rc = config.run_logged([sys.executable, "-m", "fea_core.twod.worker", str(jp)], log, cancel, timeout=timeout)
    if rc is None:
        raise Mesh2DError("Meshing cancelled by user")
    rp = Path(job["report"])
    if not rp.exists():
        raise Mesh2DError(f"The 2D meshing process crashed (code {rc}):\n"
                          + log.read_text(encoding="utf-8", errors="replace")[-3000:])
    rep = json.loads(rp.read_text(encoding="utf-8"))
    if not rep.get("ok"):
        raise Mesh2DError(f"2D meshing failed: {rep.get('error')}\n{rep.get('traceback', '')}")
    with np.load(job["output"]) as z:
        d = {k: z[k] for k in z.files}
    nodes = d["nodes"].astype(float)
    if axis_clamp:
        tol = 1e-9 * max(float(np.linalg.norm(nodes.max(0) - nodes.min(0))), 1.0)
        nodes[np.abs(nodes[:, 0]) < tol, 0] = 0.0
    tri, flipped = orient(nodes, d["tri"].astype(np.int64))
    rep["flipped"] = flipped
    seg, elem, face = boundary_lookup(tri, d["seg"].astype(np.int64))
    return Mesh2D(nodes, tri, d["tri_part"].astype(np.int64), rep["part_names"], seg, elem, face,
                  d["seg_curve"].astype(np.int64), d["curve_pts"], rep)


# ------------------------------------------------------------------ integrace

def element_points(mesh: Mesh2D):
    """Gauss points of all the elements: x (M, G, 2), N (G, 6), |J|·w (M, G)."""
    N, dN = tri6_shape(TRI_GP[:, 0], TRI_GP[:, 1])
    X = mesh.nodes[mesh.tri]                                    # (M, 6, 2)
    x = np.einsum("gi,mik->mgk", N, X)
    J = np.einsum("gai,mik->mgak", dN, X)                       # (M, G, 2, 2)
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    return x, N, np.abs(det) * TRI_GW[None, :]


def body_forces(mesh: Mesh2D, density: np.ndarray, accel, weight) -> np.ndarray:
    """Consistent nodal forces of the volume load f = ρ_e·a(x) [N]: accel(x (M, G, 2)) → (M, G, 2) [mm/s²],
    weight(x) → (M, G) (2π r | t). density (M,) [t/mm³].
    """
    x, N, jw = element_points(mesh)
    b = accel(x) * (density[:, None, None] * (jw * weight(x))[..., None])      # (M, G, 2)
    fe = np.einsum("gi,mgk->mik", N, b)                                       # (M, 6, 2)
    F = np.zeros((len(mesh.nodes), 2))
    np.add.at(F, mesh.tri.ravel(), fe.reshape(-1, 2))
    return F


def edge_points(mesh: Mesh2D, rows: np.ndarray):
    """Gauss points of the boundary segments rows: x (K, G, 2), N (G, 3), ds·w (K, G), outward normal (K, G, 2)."""
    N, dN = line3_shape(LINE_GP)
    X = mesh.nodes[mesh.seg[rows]]                               # (K, 3, 2)
    x = np.einsum("gi,kij->kgj", N, X)
    t = np.einsum("gi,kij->kgj", dN, X)
    ln = np.linalg.norm(t, axis=2)
    n = np.stack([t[..., 1], -t[..., 0]], -1) / np.maximum(ln, 1e-300)[..., None]
    return x, N, ln * LINE_GW[None, :], n


def edge_forces(mesh: Mesh2D, rows: np.ndarray, traction, weight) -> np.ndarray:
    """Consistent nodal forces of the edge load: traction(x (K, G, 2), n (K, G, 2)) → (K, G, 2) [MPa]."""
    F = np.zeros((len(mesh.nodes), 2))
    if len(rows) == 0:
        return F
    x, N, dw, n = edge_points(mesh, rows)
    q = traction(x, n) * (dw * weight(x))[..., None]
    fe = np.einsum("gi,kgj->kij", N, q)
    np.add.at(F, mesh.seg[rows].ravel(), fe.reshape(-1, 2))
    return F


def edge_measure(mesh: Mesh2D, rows: np.ndarray, weight) -> float:
    """∫ w ds over the segments (axisym: the area of the surface by the rotation over the whole circumference, planar: length × t)."""
    if len(rows) == 0:
        return 0.0
    x, _, dw, _ = edge_points(mesh, rows)
    return float((dw * weight(x)).sum())
