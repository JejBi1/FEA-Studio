"""Coloured contours of results in bands – pure numpy, without GUI (shared by the GUI and the headless MCP renderer).

Surface tri6 are split into 4 linear triangles, each is trimmed exactly along
the band boundaries (a linear field in a triangle → the band boundary is a segment).
"""
from __future__ import annotations

import numpy as np

# 12 bands: blue → green → red (habit of FE programs)
BAND_COLORS = [
    (0, 0, 255), (0, 93, 255), (0, 185, 255), (0, 255, 232), (0, 255, 139), (0, 255, 46),
    (46, 255, 0), (139, 255, 0), (232, 255, 0), (255, 185, 0), (255, 93, 0), (255, 0, 0),
]

SUBTRI = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])


def linear_triangles(tri: np.ndarray) -> np.ndarray:
    """tri6 → 4 linear triangles (node indices), tri3 unchanged."""
    if tri.shape[1] == 3:
        return tri
    return tri[:, SUBTRI].reshape(-1, 3)


def band_edges(vmin: float, vmax: float, n: int = 12) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmax - vmin < 1e-12 * max(1.0, abs(vmax)):
        vmax = vmin + max(1e-9, abs(vmin) * 1e-6)
    return np.linspace(vmin, vmax, n + 1)


def band_colors(n: int) -> list[tuple[int, int, int]]:
    """n band colours (12 = the GUI table, otherwise interpolation of the same scale)."""
    if n == len(BAND_COLORS):
        return list(BAND_COLORS)
    c = np.asarray(BAND_COLORS, float)
    t = np.linspace(0, len(c) - 1, n)
    return [tuple(int(round(np.interp(v, np.arange(len(c)), c[:, k]))) for k in range(3)) for v in t]


def _clip(P, V, cnt, c, sign):
    """Trims a batch of convex polygons by the half-plane sign*(v - c) >= 0.

    P (B, M, 3), V (B, M), cnt (B,) → new array with max. M+1 vertices.
    """
    B, M = V.shape
    f = sign * (V - c[:, None])
    outP = np.zeros((B, M + 1, 3))
    outV = np.zeros((B, M + 1))
    ocnt = np.zeros(B, dtype=np.int64)
    rows = np.arange(B)
    for i in range(M):
        valid = i < cnt
        j = np.where(i + 1 < cnt, i + 1, 0)
        fi, fj = f[rows, i], f[rows, j]
        ins_i = valid & (fi >= 0)
        # current vertex inside → write
        k = ins_i
        outP[rows[k], ocnt[k]] = P[rows[k], i]
        outV[rows[k], ocnt[k]] = V[rows[k], i]
        ocnt[k] += 1
        # edge crosses the boundary → intersection
        cross = valid & ((fi >= 0) != (fj >= 0)) & (fi != fj)
        if np.any(cross):
            r, jc = rows[cross], j[cross]
            t = (fi[cross] / (fi[cross] - fj[cross]))[:, None]
            outP[r, ocnt[cross]] = P[r, i] + t * (P[r, jc] - P[r, i])
            outV[r, ocnt[cross]] = V[r, i] + t[:, 0] * (V[r, jc] - V[r, i])
            ocnt[cross] += 1
    return outP, outV, ocnt


def contour_bands(X: np.ndarray, tri: np.ndarray, values: np.ndarray, edges: np.ndarray) -> list[np.ndarray]:
    """Triangles (n, 3, 3) for each band. Values outside the range fall into the extreme bands."""
    lt = linear_triangles(tri)
    return triangle_bands(X[lt], values[lt], edges)


def triangle_bands(P: np.ndarray, V: np.ndarray, edges: np.ndarray) -> list[np.ndarray]:
    """As contour_bands, but from ready linear triangles P (T, 3, 3) with values V (T, 3)
    (mesh section, trimmed surface – fea_core.section).
    """
    nb = len(edges) - 1
    if len(P) == 0:
        return [np.zeros((0, 3, 3)) for _ in range(nb)]
    b = np.clip(np.searchsorted(edges, V, side="right") - 1, 0, nb - 1)
    bmin, bmax = b.min(axis=1), b.max(axis=1)
    out: list[list[np.ndarray]] = [[] for _ in range(nb)]
    same = bmin == bmax
    for k in range(nb):
        sel = same & (bmin == k)
        if np.any(sel):
            out[k].append(P[sel])
    multi = np.nonzero(~same)[0]
    if len(multi):
        span = bmax[multi] - bmin[multi] + 1
        tid = np.repeat(multi, span)
        band = np.concatenate([np.arange(lo, hi + 1) for lo, hi in zip(bmin[multi], bmax[multi])])
        Pp, Vp = P[tid], V[tid]
        cnt = np.full(len(tid), 3)
        lo = np.where(band > 0, edges[band], -np.inf)
        hi = np.where(band < nb - 1, edges[band + 1], np.inf)
        lo_f = np.where(np.isfinite(lo), lo, Vp.min(axis=1) - 1.0)
        hi_f = np.where(np.isfinite(hi), hi, Vp.max(axis=1) + 1.0)
        Pp, Vp, cnt = _clip(Pp, Vp, cnt, lo_f, 1.0)
        Pp, Vp, cnt = _clip(Pp, Vp, cnt, hi_f, -1.0)
        # fan triangulation of polygons
        for k_tri in range(Pp.shape[1] - 2):
            sel = cnt >= k_tri + 3
            if not np.any(sel):
                break
            t3 = np.stack([Pp[sel, 0], Pp[sel, k_tri + 1], Pp[sel, k_tri + 2]], axis=1)
            bs = band[sel]
            for k in np.unique(bs):
                out[k].append(t3[bs == k])
    return [np.concatenate(o) if o else np.zeros((0, 3, 3)) for o in out]
