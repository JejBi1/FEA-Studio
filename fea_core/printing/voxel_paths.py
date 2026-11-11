"""Paths of the print in voxels (F5 B): perimeters, upper/lower layers, thin places and infill generated from PrintSpec.

Voxels lie in the print axes (infill.rotation: 1, 2 = plane of the layer, 3 = build), dz = height of the layer, dx = dy = w/2
(or w/4). The layers start at the bed = the lowest point of the part in the build direction, the grid in the plane of the layer is aligned
to the smallest coordinates of the part (the pattern of the infill is thus the same for the whole part regardless of the region of the detail).

Procedure (the same logic as the homogenised model region.py, only resolved by beads):
  * mask "inside the part" at the centres of the voxels – a point-in-tetrahedron test in the mesh of the global result (corner nodes);
  * layer: distance from the contour d = EDT(mask) − dx/2 (scipy.ndimage.distance_transform_edt) → perimeter k
    = band [k·w, (k + 1)·w), k < walls; direction of the line = tangent of the contour (gradient of the smoothed distance rotated by 90°);
  * upper/lower layers = number of layers of material above / below the voxel in the column < top/bottom_layers, raster ±45°
    alternating by layers (odd / even layer of the bed);
  * thin places (solid_thin): the interior without perimeters that disappears by opening with a circle of diameter thin_limit → full
    infill (±45° as the full layers);
  * infill: the pattern in the coordinates of the bed – grid (two families ±45°), rectilinear (one family, alternating ±45°),
    triangles (0/60/120°), cubic (0/60/120° shifted with the height), gyroid (a bead of the width w along the section
    by the surface sin x cos y + sin y cos z + sin z cos x = 0 at the height of the layer), concentric (further bands of the contour). The pitch is chosen so that the volume fraction = the density of the infill
    (for several families the overlap of the crossings is subtracted: 1 − (1 − w/S)^n = ρ).
The result is a VoxelGrid: kind of voxel (0 empty, 1 perimeter, 2 upper/lower layer, 3 infill, 4 full infill of a thin
place), angle of the line in the plane of the layer (from the print axis 1, mod π), number of the perimeter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EMPTY, PERIMETER, SKIN, INFILL, THIN = 0, 1, 2, 3, 4
KIND_LABELS = {EMPTY: "empty", PERIMETER: "perimetr", SKIN: "top/bottom layer", INFILL: "infill",
               THIN: "solid infill (thin region)"}
# patterns which are only approximated in voxels (warning)
APPROX = {"honeycomb": "grid", "3dhoneycomb": "cubic", "crosshatch": "grid", "tri-hexagon": "triangles",
          "adaptivecubic": "cubic"}


@dataclass
class VoxelGrid:
    """Voxels of the region of the detail in the print axes. The node (i, j, k) has the print coordinates origin + (i, j, k)·h,
    global x = Rᵀ·q. The arrays (nx, ny, nz) are per voxel.
    """
    R: np.ndarray                  # (3, 3) rows = print axes in global coordinates
    origin: np.ndarray             # (3,) print coordinates of the node (0, 0, 0)
    h: np.ndarray                  # (3,) size of the voxel
    kind: np.ndarray               # (nx, ny, nz) int8
    angle: np.ndarray              # (nx, ny, nz) angle of the line [rad] from the print axis 1 (mod π)
    wall: np.ndarray               # (nx, ny, nz) int8: number of the perimeter (0 = outer), −1 elsewhere
    inside: np.ndarray             # (nx, ny, nz) bool: centre of the voxel in the part
    layer0: int = 0                # number of the layer of the bed of the first layer of the region
    free_faces: list = field(default_factory=list)   # walls of the box of the region on the boundary of the part: "x-", "x+", … (free)
    info: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.kind.shape)

    def centers(self, idx: np.ndarray | None = None) -> np.ndarray:
        """Centres of the voxels in the print coordinates: all (nx·ny·nz, 3) or the indices idx (K, 3)."""
        if idx is None:
            idx = np.stack(np.meshgrid(*[np.arange(n) for n in self.shape], indexing="ij"), -1).reshape(-1, 3)
        return self.origin + (np.asarray(idx) + 0.5) * self.h

    def to_global(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q) @ self.R

    def to_print(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x) @ self.R.T

    def counts(self) -> dict:
        return {KIND_LABELS[k]: int((self.kind == k).sum()) for k in (PERIMETER, SKIN, INFILL, THIN)}


# ---------------------------------------------------------------- point in the mesh
def tet_locator(nodes: np.ndarray, tets: np.ndarray):
    """Preprocessing for finding the points in the tetrahedra (corner nodes): KD tree of the centroids, inversion of the Jacobians."""
    from scipy.spatial import cKDTree
    from ..loads import inv3
    P = nodes[tets[:, :4]]                                   # (E, 4, 3)
    a = P[:, 0]
    J = (P[:, 1:] - a[:, None, :]).transpose(0, 2, 1)        # columns = edges
    cen = P.mean(axis=1)
    rad = np.linalg.norm(P - cen[:, None], axis=2).max(axis=1)
    return {"tree": cKDTree(cen), "a": a, "Jinv": inv3(J), "rmax": float(rad.max()), "n": len(tets)}


def locate_points(loc: dict, pts: np.ndarray, k: int = 12, tol: float = 1e-9, chunk: int = 50_000):
    """Element with the point (index into the locator mesh), natural coordinates xi (P, 3) and the deviation err (0 = inside).
    Candidates = the k nearest centroids; the points without a found element that may lie inside (the centroid nearer than
    the largest radius of the element) are searched again among 64 candidates. Always returns the element with the smallest deviation.
    """
    pts = np.asarray(pts, float)
    P = len(pts)
    el = np.zeros(P, np.int64)
    xi = np.zeros((P, 3))
    err = np.full(P, np.inf)
    if P == 0:
        return el, xi, err

    def run(ids: np.ndarray, kk: int):
        kk = min(kk, loc["n"])
        for s in range(0, len(ids), chunk):
            ii = ids[s:s + chunk]
            d0, cand = loc["tree"].query(pts[ii], k=kk, workers=-1 if len(ii) > 5000 else 1)
            cand = np.asarray(cand).reshape(len(ii), kk)
            x = np.einsum("pkij,pkj->pki", loc["Jinv"][cand], pts[ii][:, None, :] - loc["a"][cand])
            e = np.maximum(np.maximum(-x.min(axis=2), x.sum(axis=2) - 1.0), 0.0)
            j = np.argmin(e, axis=1)
            r = np.arange(len(ii))
            better = e[r, j] < err[ii]
            sel = ii[better]
            el[sel] = cand[r, j][better]
            xi[sel] = x[r, j][better]
            err[sel] = e[r, j][better]
            yield ii, np.atleast_2d(d0)[:, 0] if kk > 1 else np.atleast_1d(d0)
    near = []
    for ii, d0 in run(np.arange(P), k):
        miss = (err[ii] > tol) & (d0 <= loc["rmax"])
        near.append(ii[miss])
    near = np.concatenate(near) if near else np.zeros(0, np.int64)
    if len(near) and k < 64:
        for _ in run(near, 64):
            pass
    return el, xi, err


# ---------------------------------------------------------------- infill patterns
_GYROID: dict = {}


def _gyroid(X, Y, Z):
    """Gyroid f and its derivatives in the plane of the layer (∂f/∂X, ∂f/∂Y) in dimensionless coordinates."""
    f = np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)
    gx = np.cos(X) * np.cos(Y) - np.sin(Z) * np.sin(X)
    gy = -np.sin(X) * np.sin(Y) + np.cos(Y) * np.cos(Z)
    return f, gx, gy


def gyroid_scale(rho: float) -> float:
    """q(ρ): the voxel is infill when |f| < q·|∇_xy f| (the distance from the curve of the section ≈ |f|/|∇_xy f| < w·k/2), and the fraction
    of such points of the periodic cell (over the whole period in Z) = ρ → k = 2q/w. Quantile of the ratio |f|/|∇_xy f| on 96³.
    """
    if "r" not in _GYROID:
        n = 96
        t = (np.arange(n) + 0.5) * 2 * np.pi / n
        X, Y, Z = np.meshgrid(t, t, t, indexing="ij")
        f, gx, gy = _gyroid(X, Y, Z)
        _GYROID["r"] = np.sort((np.abs(f) / np.maximum(np.sqrt(gx * gx + gy * gy), 1e-12)).ravel())
    r = _GYROID["r"]
    return float(r[min(len(r) - 1, int(rho * len(r)))])


def _strip(s: np.ndarray, S: float, w: float, phase: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Strips of the width w with the pitch S perpendicular to the coordinate s: (mask, |distance from the axis of the strip|)."""
    d = np.abs(np.mod(s - phase + 0.5 * S, S) - 0.5 * S)
    return d < 0.5 * w, d


def infill_layer(pattern: str, rho: float, w: float, u: np.ndarray, v: np.ndarray, z: float,
                 layer: int, d_edge: np.ndarray | None = None, walls: int = 0):
    """Infill of one layer at the points (u, v) of the coordinates of the bed [mm]: (mask, angle of the line [rad])."""
    shape = u.shape
    if rho >= 0.99:                                             # full infill: straight ±45° alternating
        return np.ones(shape, bool), np.full(shape, np.pi / 4 if layer % 2 == 0 else 3 * np.pi / 4)
    if rho <= 0.0 or pattern == "lightning":
        return np.zeros(shape, bool), np.zeros(shape)
    pattern = APPROX.get(pattern, pattern)

    def families(angles, n_fam, shift=None, spacing=None):
        S = spacing or w / (1.0 - (1.0 - rho) ** (1.0 / n_fam))      # independent families: 1 − (1 − w/S)^n = ρ
        mask = np.zeros(shape, bool)
        best = np.full(shape, np.inf)
        ang = np.zeros(shape)
        for i, a in enumerate(angles):
            s = -math.sin(a) * u + math.cos(a) * v              # coordinate perpendicular to the line at the angle a
            m, d = _strip(s, S, w, 0.0 if shift is None else shift[i])
            take = m & (d < best)
            ang[take] = a % math.pi
            best = np.where(take, d, best)
            mask |= m
        return mask, ang
    if pattern == "grid":
        return families([math.pi / 4, 3 * math.pi / 4], 2)
    if pattern in ("rectilinear", "alignedrectilinear", "zigzag"):
        a = math.pi / 4 if layer % 2 == 0 else 3 * math.pi / 4
        return families([a], 1)
    if pattern == "triangles":                                # 3 families at one point: uncovered triangle
        S = 1.5 * w / (1.0 - math.sqrt(1.0 - rho))               # heights S − 1.5w → ρ = 1 − (1 − 1.5w/S)²
        return families([0.0, math.pi / 3, 2 * math.pi / 3], 3, spacing=S)
    if pattern == "cubic":                                     # cube at the corner: families 0/60/120° shifted with the height
        sh = z / math.sqrt(2.0)
        return families([0.0, math.pi / 3, 2 * math.pi / 3], 3, shift=[sh, -sh, sh])
    if pattern == "gyroid":                                  # bead of the width w along the section of the gyroid f = 0 in the layer
        k = 2.0 * gyroid_scale(rho) / w
        f, gx, gy = _gyroid(k * u, k * v, k * z)
        gn = np.sqrt(gx * gx + gy * gy)                          # distance from the curve ≈ |f| / (k·|∇_xy f|)
        return np.abs(f) < 0.5 * w * k * gn, np.mod(np.arctan2(gy, gx) + math.pi / 2, math.pi)
    if pattern == "concentric":
        if d_edge is None:
            raise ValueError("concentric infill needs the distance from the outline")
        step = max(1, int(round(1.0 / rho)))
        p = np.floor(d_edge / w).astype(np.int64) - walls
        return (p >= 0) & (np.mod(p, step) == 0), None
    raise KeyError(f"Unknown infill pattern '{pattern}'")


# ---------------------------------------------------------------- generator
def _edt(mask: np.ndarray, sampling) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    if mask.all():
        return np.full(mask.shape, np.inf)
    return distance_transform_edt(mask, sampling=sampling)


def layer_paths(inside: np.ndarray, spec, dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perimeters and thin places of one layer: (number of the perimeter −1/0.., angle of the line, distance from the contour d,
    mask of the thin place). inside (nx, ny) – the edge of the array is NOT taken as a contour (the region lies inside a larger part).
    """
    from scipy.ndimage import gaussian_filter
    w = float(spec.line_width)
    d = _edt(inside, (dx, dx)) - 0.5 * dx
    d = np.where(inside, d, -1.0)
    wall = np.where(inside & (d < spec.walls * w), np.floor(np.maximum(d, 0) / w), -1).astype(np.int8)
    dd = np.where(np.isfinite(d), d, 1e6)
    sm = gaussian_filter(np.where(inside, dd, -0.5 * dx), 1.0, mode="nearest")
    gx, gy = np.gradient(sm, dx)
    ang = np.mod(np.arctan2(gy, gx) + np.pi / 2, np.pi)       # tangent of the contour = gradient rotated by 90°
    thin = np.zeros(inside.shape, bool)
    if getattr(spec, "solid_thin", True):
        core = inside & (wall < 0)
        r = 0.5 * float(spec.thin_limit)
        if core.any():
            dc = _edt(core, (dx, dx)) - 0.5 * dx
            er = core & (dc >= r)
            if er.any():
                # an opening by a circle of radius r would also mark the corners of a wide interior (up to r·√2 from the narrowed core)
                # → dilation by r·√2: only what lies further from the wide core remains thin (narrow strips, protrusions)
                op = _edt(~er, (dx, dx)) <= math.sqrt(2.0) * r + 0.5 * dx
                thin = core & ~op
            else:
                thin = core
    return wall, ang, d, thin


def generate(spec, R: np.ndarray, plate: np.ndarray, lo: np.ndarray, hi: np.ndarray, dx: float,
             inside_fn, part_lo: np.ndarray, part_hi: np.ndarray) -> VoxelGrid:
    """Voxels of the box [lo, hi] (print coordinates) of the part with the settings spec.

    plate = origin of the bed in the print coordinates (the smallest coordinates of the part: the layers and the pattern of the infill are computed from
    it), inside_fn(q (P, 3)) → bool (centre of the voxel in the part), part_lo/hi = envelope of the part in the axes of the print.
    """
    h = np.array([dx, dx, float(spec.layer_height)])
    w = float(spec.line_width)
    # grid aligned to the bed, the region rounded outwards to whole voxels
    i0 = np.floor((lo - plate) / h + 1e-9).astype(int)
    i1 = np.ceil((hi - plate) / h - 1e-9).astype(int)
    p0 = np.floor((part_lo - plate) / h + 1e-9).astype(int)
    p1 = np.ceil((part_hi - plate) / h - 1e-9).astype(int)
    free = []
    for a, nm in enumerate("xyz"):
        if i0[a] <= p0[a]:
            i0[a] = p0[a]
            free.append(nm + "-")
        if i1[a] >= p1[a]:
            i1[a] = p1[a]
            free.append(nm + "+")
    if np.any(i1 <= i0):
        raise ValueError("Detail region does not lie in the part")
    # extended window: perimeters and thin places need the contour up to the distance walls·w + thin_limit, layers a column
    pxy = int(math.ceil((spec.walls * w + float(spec.thin_limit) + 2 * dx) / dx)) + 2
    pz = max(int(spec.top_layers), int(spec.bottom_layers)) + 1
    pad = np.array([pxy, pxy, pz])
    w0 = np.maximum(i0 - pad, p0 - 1)
    w1 = np.minimum(i1 + pad, p1 + 1)
    shp = tuple(int(v) for v in (w1 - w0))
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shp], indexing="ij"), -1).reshape(-1, 3)
    q = plate + (w0 + ijk + 0.5) * h
    inside = np.asarray(inside_fn(q), bool).reshape(shp)
    nx, ny, nz = shp
    kind = np.zeros(shp, np.int8)
    angle = np.zeros(shp)
    wall = np.full(shp, -1, np.int8)
    # columns: number of layers of material above / below the voxel (outside the window = material, the edge of the part is empty in the window)
    big = 10 ** 6
    up = np.zeros(shp, np.int64)
    run = np.full((nx, ny), big, np.int64)
    for k in range(nz - 1, -1, -1):
        up[:, :, k] = run
        run = np.where(inside[:, :, k], np.minimum(run + 1, big), 0)
    down = np.zeros(shp, np.int64)
    run = np.full((nx, ny), big, np.int64)
    for k in range(nz):
        down[:, :, k] = run
        run = np.where(inside[:, :, k], np.minimum(run + 1, big), 0)
    skin = inside & ((up < int(spec.top_layers)) | (down < int(spec.bottom_layers)))
    u = plate[0] + (w0[0] + np.arange(nx) + 0.5) * h[0] - plate[0]
    v = plate[1] + (w0[1] + np.arange(ny) + 0.5) * h[1] - plate[1]
    U, V = np.meshgrid(u, v, indexing="ij")
    rho = float(spec.infill)
    pattern = spec.pattern
    for k in range(nz):
        ins = inside[:, :, k]
        if not ins.any():
            continue
        layer = int(w0[2] + k)
        zc = (layer + 0.5) * h[2]
        wl, ang, d, thin = layer_paths(ins, spec, dx)
        alt = np.pi / 4 if layer % 2 == 0 else 3 * np.pi / 4
        per = wl >= 0
        sk = skin[:, :, k] & ~per
        th = thin & ~sk
        rest = ins & ~per & ~sk & ~th
        fm, fa = infill_layer(pattern, rho, w, U, V, zc, layer, d_edge=d, walls=int(spec.walls))
        if fa is None:                                          # concentric: the direction = tangent of the contour
            fa = ang
        kl = np.zeros(ins.shape, np.int8)
        al = np.zeros(ins.shape)
        kl[per], al[per] = PERIMETER, ang[per]
        kl[sk], al[sk] = SKIN, alt
        kl[th], al[th] = THIN, alt
        fill = rest & fm
        kl[fill], al[fill] = INFILL, fa[fill]
        kind[:, :, k], angle[:, :, k], wall[:, :, k] = kl, al, np.where(per, wl, -1)
    s = tuple(slice(int(a), int(b)) for a, b in zip(i0 - w0, i1 - w0))
    g = VoxelGrid(R=np.asarray(R, float), origin=plate + i0 * h, h=h, kind=kind[s].copy(), angle=angle[s].copy(),
                  wall=wall[s].copy(), inside=inside[s].copy(), layer0=int(i0[2]), free_faces=free)
    if spec.pattern in APPROX:
        g.warnings.append(f"Infill pattern '{spec.pattern}' is approximated by pattern '{APPROX[spec.pattern]}' in the voxels.")
    if spec.pattern == "lightning":
        g.warnings.append("Lightning infill is non-structural – omitted in the voxel detail.")
    g.info = {"window": [int(v) for v in shp], "pad": [int(v) for v in pad]}
    return g


def infill_fraction(g: VoxelGrid) -> float:
    """Infill fraction: voxels of the infill / voxels of the interior without perimeters, full layers and thin places."""
    core = g.inside & ~np.isin(g.kind, (PERIMETER, SKIN, THIN))
    n = int(core.sum())
    return float((g.kind[core] == INFILL).sum() / n) if n else 0.0
