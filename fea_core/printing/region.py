"""Assignment of the shell and the core to the elements of a printed part according to the distance from the surface (F4).

Robust path as specified (not OCC offset): the surface of the part from the mesh (the faces of the elements of the part that do not share
another element in the part – also the contact with another part is a surface of the printing). Each surface triangle has a shell thickness
according to the slope to the build axis b (c = n·b):
    t = max(perimeters·width of the line·√(1 − c²), layers·height of the layer·|c|),   layers = upper (c > 0) / lower.
A point is in the shell when a triangle exists with the distance d < t. The shell fraction of the element f = the fraction of
uniformly distributed sampling points in the shell → rounded to 1/bins (materials by levels).
The direction of the line of the perimeters in the element = b × n of the triangle that decides at the centroid of the element (argmin d − t), if
the shell is formed there by the perimeters (otherwise the upper/lower layers – without direction) → cell_data["print_line"].
Thin places (F5, PrintSpec.solid_thin): where the interior of the part in the plane of the layer after subtracting the perimeters is narrower than
PrintSpec.thin_limit (default 2 × width of the line), the slicer prints a full infill (a sparse one does not fit there) → the element
is full (f = 1), cell_data["print_thin"] = 1. Width = the shortest of the chords in the directions of the normals of the nearby walls
in the plane of the layer (rays as for the thickness of the sheet of the welds), INDICATIVE (the slicers decide by the area of the region, e.g. 15 mm²).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..mesh import Mesh, TET_FACES

if TYPE_CHECKING:
    from ..study import Study

_WORKERS = -1                     # threads of cKDTree.query for big batches (config.NUM_THREADS measured the same or slower)
_BATCH = 1_000_000                # queried points × neighbours per batch of shell_fraction (300 k: example 2 3.5 s → 1 M: 2.6 s, bit-identical)
_OPP = np.array([3, 2, 0, 1])     # opposite corner to the face TET_FACES[k]


def sample_points(n: int) -> np.ndarray:
    """Barycentric coordinates (n, 4) of uniformly distributed points in a tetrahedron (Sobol → simplex
    by sorting three coordinates). Uniform distribution = an unbiased estimate of the volume fraction (a grid
    with points on the boundary of the element systematically distorted the shell fraction).
    """
    from ._sobol import SOBOL_3D_SEED7            # = qmc.Sobol(3, scramble=True, seed=7).random(512); scipy.stats import ~2 s
    if n <= len(SOBOL_3D_SEED7):
        u = np.sort(SOBOL_3D_SEED7[:n], axis=1)
    else:
        from scipy.stats import qmc
        u = np.sort(qmc.Sobol(3, scramble=True, seed=7).random(n), axis=1)
    return np.diff(np.hstack([np.zeros((n, 1)), u, np.ones((n, 1))]), axis=1)


def point_triangle_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Exact distance of the points p (K, 3) from the triangles (a, b, c) (K, 3) – pairwise."""
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = np.einsum("ij,ij->i", n, n)
    nn = np.where(nn > 0, nn, 1e-300)
    ap = p - a
    dist_pl = np.einsum("ij,ij->i", ap, n) / np.sqrt(nn)
    q = p - (dist_pl / np.sqrt(nn))[:, None] * n          # projection onto the plane
    # barycentric coordinates of the projection
    v0, v1, v2 = ab, ac, q - a
    d00, d01, d11 = (np.einsum("ij,ij->i", v0, v0), np.einsum("ij,ij->i", v0, v1), np.einsum("ij,ij->i", v1, v1))
    d20, d21 = np.einsum("ij,ij->i", v2, v0), np.einsum("ij,ij->i", v2, v1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) > 0, d00 * d11 - d01 * d01, 1e-300)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)

    def seg(p, s0, s1):
        d = s1 - s0
        dd = np.einsum("ij,ij->i", d, d)
        t = np.clip(np.einsum("ij,ij->i", p - s0, d) / np.where(dd > 0, dd, 1e-300), 0.0, 1.0)
        return np.linalg.norm(p - (s0 + t[:, None] * d), axis=1)
    edge = np.minimum(np.minimum(seg(p, a, b), seg(p, b, c)), seg(p, c, a))
    return np.where(inside, np.abs(dist_pl), edge)


def part_surface(mesh: Mesh, elems: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Surface of the part: corners of the triangles (K, 3) and outward normals (K, 3)."""
    canon = mesh.canonical()
    corners = mesh.tets[elems, :4]
    faces = corners[:, TET_FACES]                           # (E, 4, 3)
    flat = faces.reshape(-1, 3)
    keys = np.sort(canon[flat], axis=1)
    if len(keys) and keys.max() < (1 << 21):            # one uint64 key – the same order as unique by rows
        packed = ((keys[:, 0].astype(np.uint64) << 42) | (keys[:, 1].astype(np.uint64) << 21)
                  | keys[:, 2].astype(np.uint64))
        _, inv, cnt = np.unique(packed, return_inverse=True, return_counts=True)
    else:
        _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    once = np.nonzero(cnt[inv.ravel()] == 1)[0]
    tri = flat[once]
    opp = corners[once // 4, _OPP[once % 4]]
    x = mesh.nodes[tri]
    n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    flip = np.einsum("ij,ij->i", n, x.mean(axis=1) - mesh.nodes[opp]) < 0
    n[flip] *= -1
    return tri, n


def shell_thickness(normals: np.ndarray, spec) -> np.ndarray:
    b = np.asarray(spec.build_dir, float)
    b = b / np.linalg.norm(b)
    c = normals @ b
    layers = np.where(c > 0, spec.top_layers, spec.bottom_layers) * spec.layer_height
    t = np.maximum(spec.wall_thickness * np.sqrt(np.clip(1 - c * c, 0, 1)), layers * np.abs(c))
    return np.maximum(t, 1e-9)


def wall_lines(normals: np.ndarray, spec) -> np.ndarray:
    """Direction of the line of the perimeters (K, 3) at the surface triangles: b × n (along the contour of the layer), where the perimeters
    form the shell (the wall term of the thickness ≥ the layer term). Zeros = upper/lower full layers (alternating raster ±45°,
    on average isotropic in the plane of the layer).
    """
    b = np.asarray(spec.build_dir, float)
    b = b / np.linalg.norm(b)
    c = normals @ b
    s = np.sqrt(np.clip(1 - c * c, 0, 1))
    layers = np.where(c > 0, spec.top_layers, spec.bottom_layers) * spec.layer_height
    wall = (spec.wall_thickness * s >= layers * np.abs(c)) & (s > 1e-3) & (spec.walls > 0)
    L = np.cross(b[None, :], normals)
    L /= np.maximum(np.linalg.norm(L, axis=1), 1e-300)[:, None]
    return np.where(wall[:, None], L, 0.0)


def shell_fraction(mesh: Mesh, elems: np.ndarray, spec, samples: int = 32, k: int = 8,
                   lines: bool = False):
    """Volume fraction of the elements (elems) in the shell, 0–1 (not rounded); with lines=True also the direction of the line (E, 3)
    of the triangle deciding at the centroid of the element (argmin d − t; zeros = core or full layers).

    Shell = {p: g(p) < 0}, g(p) = min_f (d_f(p) − t_f). The function g is 1-Lipschitz, so an element
    with |g(centre)| > radius of the element lies entirely inside / outside and only the element that crosses the boundary is sampled.
    The deciding quantity is d − t, not the distance alone (a nearer thin lower face × a farther thick wall), therefore
    the triangles are divided into classes by t and the k nearest (by centroid) are searched for in each class separately.
    """
    from scipy.spatial import cKDTree               # lazily: import of scipy.spatial ~0.5 s (check_printing does not need it)
    tri, nrm = part_surface(mesh, elems)
    t = shell_thickness(nrm, spec)
    X = mesh.nodes[tri]                                    # (K, 3, 3)
    cen = X.mean(axis=1)
    r_tri = np.linalg.norm(X - cen[:, None], axis=2).max(axis=1)
    tmax, rmax = float(t.max()), float(r_tri.max())
    cls = np.round(t / tmax * 20).astype(int)              # thickness classes every 5 % of t_max
    classes = []
    for c in np.unique(cls):
        ids = np.nonzero(cls == c)[0]
        classes.append((ids, cKDTree(cen[ids]), min(k, len(ids))))
    corners = mesh.nodes[mesh.tets[elems, :4]]             # (E, 4, 3)
    ec = corners.mean(axis=1)
    er = np.linalg.norm(corners - ec[:, None], axis=2).max(axis=1)
    d0, _ = cKDTree(cen).query(ec)
    out = np.zeros(len(elems))
    line = np.zeros((len(elems), 3))
    near = np.nonzero(d0 - er - rmax <= tmax)[0]           # other elements lie entirely in the core
    if len(near) == 0:
        return (out, line) if lines else out

    from ._dist import g_min                               # numba (the same computation as point_triangle_distance)
    Xc = np.ascontiguousarray(X, dtype=np.float64)
    tc = np.ascontiguousarray(t, dtype=np.float64)

    def g_of(pts, arg=False):                              # pts (Q, 3) → g (Q,) [, deciding triangle]
        pts = np.ascontiguousarray(pts, dtype=np.float64)
        g = np.full(len(pts), np.inf)
        best = np.zeros(len(pts), dtype=np.int64)
        workers = _WORKERS if len(pts) >= 20_000 else 1    # start of the threads of cKDTree costs ~1 ms – small batches in 1 thread
        for ids, tree, kk in classes:
            _, cand = tree.query(pts, k=kk, workers=workers)
            cf = np.ascontiguousarray(ids[np.asarray(cand).reshape(len(pts), kk)], dtype=np.int64)
            g_min(pts, cf, Xc, tc, g, best)
        return (g, best) if arg else g
    gc, bc = g_of(ec[near], arg=True)
    out[near[gc < -er[near]]] = 1.0
    if lines:
        line[near] = wall_lines(nrm[bc], spec)
    cut = near[np.abs(gc) <= er[near]]                     # elements crossing the boundary of the shell
    if len(cut) == 0:
        line[out <= 0] = 0.0
        return (out, line) if lines else out
    L = sample_points(samples)                             # (P, 4)
    P = len(L)
    # batch ~1 M queried points × k neighbours per class (_BATCH); do NOT divide by the number of thickness classes (up to 21) – batches of ~50
    # elements meant ~20 k calls of cKDTree.query with thread starts (arm of example 2: 16.6 → 3.0 s, bitwise equal)
    chunk = max(1, _BATCH // (P * k))
    for s in range(0, len(cut), chunk):
        idx = cut[s:s + chunk]
        pts = np.einsum("pa,eaj->epj", L, corners[idx]).reshape(-1, 3)
        out[idx] = (g_of(pts) < 0).reshape(len(idx), P).mean(axis=1)
    if lines:
        line[out <= 0] = 0.0
        return out, line
    return out


def thin_core(mesh: Mesh, elems: np.ndarray, spec, f: np.ndarray | None = None,
              chunk: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """Thin places of the part: (mask of elements elems with an interior narrower than spec.thin_limit, width of the interior [mm] (inf = wide
    or undetermined)). Width of the interior = width of the part in the plane of the layer (chord through the centroid of the element in the direction of the projection of the normal
    of the walls nearer than half the largest possible thin width, the shortest – at the end of a thin wall the nearest face is the front) −
    the shell at both intersected walls (t/|u·n|). f = shell fraction (elements with f ≥ 1 are skipped).
    """
    from itertools import chain
    from scipy.spatial import cKDTree
    from ._dist import line_width
    width = np.full(len(elems), np.inf)
    thr = float(spec.thin_limit)
    if len(elems) == 0 or thr <= 0:
        return np.zeros(len(elems), bool), width
    tri, nrm = part_surface(mesh, elems)
    t = shell_thickness(nrm, spec)
    b = np.asarray(spec.build_dir, float)
    b = b / np.linalg.norm(b)
    inp = nrm - (nrm @ b)[:, None] * b[None, :]
    sl = np.linalg.norm(inp, axis=1)
    wall = sl >= 0.5                                       # wall (slope > 30° from the horizontal) – not the upper/lower face
    if not wall.any():
        return np.zeros(len(elems), bool), width
    inplane = inp / np.maximum(sl, 1e-300)[:, None]
    X = np.ascontiguousarray(mesh.nodes[tri], dtype=np.float64)
    cen = X.mean(axis=1)
    rmax = float(np.linalg.norm(X - cen[:, None], axis=2).max())
    w_max = thr + 2.0 * float(t[wall].max())               # a wider part cannot be thin
    ec = mesh.nodes[mesh.tets[elems, :4]].mean(axis=1)
    cand = np.arange(len(elems)) if f is None else np.nonzero(np.asarray(f) < 1.0)[0]
    tree = cKDTree(cen)
    if len(cand):
        d0, _ = tree.query(ec[cand])
        cand = cand[d0 - rmax <= 0.5 * w_max]
    R = w_max + rmax
    for s0 in range(0, len(cand), chunk):
        c = cand[s0:s0 + chunk]
        pts = np.ascontiguousarray(ec[c], dtype=np.float64)
        lists = tree.query_ball_point(pts, R, workers=-1 if len(c) >= 5000 else 1)
        lens = np.fromiter((len(x) for x in lists), np.int64, len(lists))
        ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        idx = np.fromiter(chain.from_iterable(lists), np.int64, int(ptr[-1]))
        W = np.empty(len(c))
        line_width(pts, ptr, idx, X, np.ascontiguousarray(inplane), wall, np.ascontiguousarray(t, dtype=np.float64),
                   np.ascontiguousarray(nrm, dtype=np.float64), 0.5 * w_max, W)
        width[c] = W
    return width < thr, width


def print_parts(study: Study, part_names: list[str]) -> list[str]:
    """Parts with printed material (they are computed by the printing model)."""
    out = []
    for p in part_names:
        try:
            if study.material_of(p).is_print:
                out.append(p)
        except KeyError:
            pass
    return out


def assign(study: Study, mesh: Mesh) -> list[str]:
    """Shell fraction per element → mesh.cell_data["print_shell"] (−1 = unprinted part), returns warnings."""
    warnings: list[str] = []
    shell = np.full(len(mesh.tets), -1.0)
    lines = np.zeros((len(mesh.tets), 3))
    thin = np.zeros(len(mesh.tets))
    ps = study.printing
    bins = max(1, int(ps.bins))
    for pi, pname in enumerate(mesh.part_names):
        if pname not in print_parts(study, [pname]):
            continue
        elems = np.nonzero(mesh.tet_part == pi)[0]
        spec = ps.spec(pname)
        if not spec.enabled:
            shell[elems] = 1.0                              # whole part as a full extrusion
            continue
        f, ln = shell_fraction(mesh, elems, spec, ps.samples, lines=True)
        fr = np.round(f * bins) / bins
        if getattr(spec, "solid_thin", True):              # thin places → full infill (as the slicer)
            th, _ = thin_core(mesh, elems, spec, fr)
            if th.any():
                fr[th] = 1.0
                thin[elems[th]] = 1.0
        shell[elems] = fr
        lines[elems] = ln
        # coarse mesh against the shell → warning (the stiffness is averaged per element)
        t_min = min(spec.wall_thickness, min(spec.top_layers, spec.bottom_layers) * spec.layer_height)
        partial = (f > 0) & (f < 1)
        if partial.any():
            x = mesh.nodes[mesh.tets[elems[partial], :4]]
            h = float(np.median(np.linalg.norm(x[:, 1:] - x[:, :1], axis=2).mean(axis=1)))
            if h > 3 * t_min:
                warnings.append(f"3D printing '{pname}': elements at the surface (~{h:.2g} mm) are much larger than the shell ({t_min:.2g} mm) – shell stiffness is only averaged and stress in the shell depends on the mesh; refine the mesh at critical locations (Mesh step – local refinement).")
    mesh.cell_data["print_shell"] = shell
    mesh.cell_data["print_line"] = lines                   # direction of the line of the perimeters (0 = no line)
    mesh.cell_data["print_thin"] = thin                    # 1 = thin place with full infill
    return warnings


# fields of PrintSpec on which assign depends (not the infill and the pattern – they change only the material of the core; the limit of a thin place on the infill
# does not depend on them, precisely so that a change of the infill does not recompute the assignment)
ASSIGN_FIELDS = ("enabled", "build_dir", "layer_height", "line_width", "walls", "top_layers", "bottom_layers",
                 "solid_thin", "thin_width", "tolerance")


def print_signature(study: Study, part_names: list[str]) -> str:
    """Signature of the inputs of assign: printed parts, their shell geometry, bins, samples. A match = cell_data is valid."""
    import hashlib
    import json
    ps = study.printing
    d = {"bins": int(ps.bins), "samples": int(ps.samples),
         "parts": {p: {k: getattr(ps.spec(p), k) for k in ASSIGN_FIELDS} for p in print_parts(study, part_names)}}
    if any(getattr(ps.spec(p), "tolerance", 0) for p in print_parts(study, part_names)):
        # the tolerance moves the free surface, the faces of constraints and loads stand → their selections belong to the signature
        d["bc"] = sorted({c.selection for c in study.constraints}
                         | {str(getattr(l, "selection", "")) for lc in study.load_cases for l in lc.loads})
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def ensure(study: Study, mesh: Mesh) -> list[str]:
    """Assignment of the printing to a finished mesh (the mesh does not depend on the print settings – a change of the perimeters or infill does not re-mesh).
    Recomputes assign only when the signature changes (mesh.report["print_sig"]); the warnings of the assignment are kept in
    mesh.report["print_warnings"] and always returned (also without recomputation).
    """
    sig = print_signature(study, mesh.part_names)
    if mesh.report.get("print_sig") != sig:
        from .tolerance import apply as apply_tolerance
        wt = apply_tolerance(study, mesh)              # tolerance of the printing: displacement of the free surface (before the assignment)
        if print_parts(study, mesh.part_names):
            w = wt + assign(study, mesh)
        else:
            mesh.cell_data.pop("print_shell", None)
            mesh.cell_data.pop("print_line", None)
            mesh.cell_data.pop("print_thin", None)
            w = []
        mesh.report["print_sig"] = sig
        mesh.report["print_warnings"] = w
    return list(mesh.report.get("print_warnings", []))
