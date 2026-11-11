"""Numba core for the assignment of the print shell: min_f (d_f(p) − t_f) over the candidate triangles of the point.

The same computation as region.point_triangle_distance (projection onto the plane, barycentric coordinates, otherwise the nearest
edge), but without temporary numpy arrays and in parallel over the points (~10× faster, the holder of the print 9 s → <1 s).
"""
from __future__ import annotations

import math

import numpy as np

from ..modules import need

_numba = need("numba", "accel")      # if the Numba module is missing, a clear error with the installation procedure
njit = _numba.njit
prange = _numba.prange


@njit(cache=True, inline="always")
def _seg(px, py, pz, ax, ay, az, bx, by, bz):
    dx, dy, dz = bx - ax, by - ay, bz - az
    dd = dx * dx + dy * dy + dz * dz
    t = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / (dd if dd > 0 else 1e-300)
    t = min(max(t, 0.0), 1.0)
    qx, qy, qz = px - (ax + t * dx), py - (ay + t * dy), pz - (az + t * dz)
    return math.sqrt(qx * qx + qy * qy + qz * qz)


@njit(cache=True)
def _dist(px, py, pz, X, f):
    ax, ay, az = X[f, 0, 0], X[f, 0, 1], X[f, 0, 2]
    bx, by, bz = X[f, 1, 0], X[f, 1, 1], X[f, 1, 2]
    cx, cy, cz = X[f, 2, 0], X[f, 2, 1], X[f, 2, 2]
    abx, aby, abz = bx - ax, by - ay, bz - az
    acx, acy, acz = cx - ax, cy - ay, cz - az
    nx, ny, nz = aby * acz - abz * acy, abz * acx - abx * acz, abx * acy - aby * acx
    nn = nx * nx + ny * ny + nz * nz
    if nn <= 0:
        nn = 1e-300
    sn = math.sqrt(nn)
    dpl = ((px - ax) * nx + (py - ay) * ny + (pz - az) * nz) / sn
    qx, qy, qz = px - dpl / sn * nx, py - dpl / sn * ny, pz - dpl / sn * nz
    v2x, v2y, v2z = qx - ax, qy - ay, qz - az
    d00 = abx * abx + aby * aby + abz * abz
    d01 = abx * acx + aby * acy + abz * acz
    d11 = acx * acx + acy * acy + acz * acz
    d20 = v2x * abx + v2y * aby + v2z * abz
    d21 = v2x * acx + v2y * acy + v2z * acz
    den = d00 * d11 - d01 * d01
    if abs(den) <= 0:
        den = 1e-300
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    if v >= 0 and w >= 0 and v + w <= 1:
        return abs(dpl)
    e = _seg(px, py, pz, ax, ay, az, bx, by, bz)
    e = min(e, _seg(px, py, pz, bx, by, bz, cx, cy, cz))
    return min(e, _seg(px, py, pz, cx, cy, cz, ax, ay, az))


@njit(cache=True, parallel=True)
def g_min(pts, cand, X, t, g, best):
    """For each point p: g[p] = min(g[p], min_j d(p, cand[p, j]) − t[cand[p, j]]), best = the deciding triangle."""
    for i in prange(pts.shape[0]):
        px, py, pz = pts[i, 0], pts[i, 1], pts[i, 2]
        for j in range(cand.shape[1]):
            f = cand[i, j]
            v = _dist(px, py, pz, X, f) - t[f]
            if v < g[i]:
                g[i] = v
                best[i] = f



@njit(cache=True)
def _chord(px, py, pz, ux, uy, uz, lo, hi, idx, X):
    """Chord of the line p + s·u: the nearest intersections with the triangles idx[lo:hi] before (s ≥ 0) and after the point
    (Möller–Trumbore) → (s₊ + s₋, triangle before, after); inf / −1 = not found.
    """
    sp, sm = np.inf, np.inf
    fa, fb = -1, -1
    for j in range(lo, hi):
        f = idx[j]
        ax, ay, az = X[f, 0, 0], X[f, 0, 1], X[f, 0, 2]
        e1x, e1y, e1z = X[f, 1, 0] - ax, X[f, 1, 1] - ay, X[f, 1, 2] - az
        e2x, e2y, e2z = X[f, 2, 0] - ax, X[f, 2, 1] - ay, X[f, 2, 2] - az
        qx, qy, qz = uy * e2z - uz * e2y, uz * e2x - ux * e2z, ux * e2y - uy * e2x
        det = e1x * qx + e1y * qy + e1z * qz
        if abs(det) < 1e-14:
            continue
        inv = 1.0 / det
        sx, sy, sz = px - ax, py - ay, pz - az
        u = (sx * qx + sy * qy + sz * qz) * inv
        if u < -1e-9 or u > 1 + 1e-9:
            continue
        rx, ry, rz = sy * e1z - sz * e1y, sz * e1x - sx * e1z, sx * e1y - sy * e1x
        v = (ux * rx + uy * ry + uz * rz) * inv
        if v < -1e-9 or u + v > 1 + 1e-9:
            continue
        s = (e2x * rx + e2y * ry + e2z * rz) * inv
        if s >= 0:
            if s < sp:
                sp = s
                fa = f
        elif -s < sm:
            sm = -s
            fb = f
    if fa < 0 or fb < 0:
        return np.inf, -1, -1
    return sp + sm, fa, fb



@njit(cache=True, parallel=True)
def line_width(pts, ptr, idx, X, inplane, wall, t, nrm, reach, out_w):
    """Width of the interior of the part in the plane of the layer (thin places, F5). For the point p and each "wall" triangle
    (wall) from the candidates idx[ptr[i]:ptr[i+1]] nearer than reach: chord by a line through p in the direction of the projection of its normal
    into the plane of the layer (inplane; directions within ~10° of those already tried are skipped, at most 32 directions) → width of the interior =
    chord − t/|u·n| at both intersected triangles (shell); out_w[i] = the smallest over the directions (inf = not found).
    More directions (not only the nearest wall): at the end of a thin wall the nearest face is the front, but the thin one is the side wall.
    """
    for i in prange(pts.shape[0]):
        px, py, pz = pts[i, 0], pts[i, 1], pts[i, 2]
        out_w[i] = np.inf
        tried = np.zeros((32, 3))
        nt = 0
        for j in range(ptr[i], ptr[i + 1]):
            f = idx[j]
            if not wall[f]:
                continue
            if _dist(px, py, pz, X, f) > reach:
                continue
            ux, uy, uz = inplane[f, 0], inplane[f, 1], inplane[f, 2]
            dup = False
            for q in range(nt):
                if abs(ux * tried[q, 0] + uy * tried[q, 1] + uz * tried[q, 2]) > 0.985:
                    dup = True
                    break
            if dup or nt >= 32:
                continue
            tried[nt, 0], tried[nt, 1], tried[nt, 2] = ux, uy, uz
            nt += 1
            w, fa, fb = _chord(px, py, pz, ux, uy, uz, ptr[i], ptr[i + 1], idx, X)
            if fa < 0:
                continue
            ca = max(abs(ux * nrm[fa, 0] + uy * nrm[fa, 1] + uz * nrm[fa, 2]), 0.3)
            cb = max(abs(ux * nrm[fb, 0] + uy * nrm[fb, 1] + uz * nrm[fb, 2]), 0.3)
            core = max(w - t[fa] / ca - t[fb] / cb, 0.0)
            if core < out_w[i]:
                out_w[i] = core
