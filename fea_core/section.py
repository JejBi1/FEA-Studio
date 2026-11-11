"""Section of the mesh by a plane: coloured section through the interior (tet10/tet4) and trimming of the surface (without GUI, numpy).

Tet10 is split into 8 linear tetrahedra (4 corner + an octahedron split by the diagonal 6–8, ccx node order),
the field is linear in them → the intersection with the plane is a triangle or a quadrilateral with exactly interpolated nodal values. The plane is defined in the UNDEFORMED position (material section);
display positions X (e.g. deformed) are interpolated with the same parameter on the edge.
Surface tri6 → 4 linear triangles, the side (x − o)·n ≤ 0 is kept.

Use (GUI and headless rendering):
    P, V = plane_section(nodes, tets, values, origin, normal, X)     # (K, 3, 3), (K, 3)
    P, V = clip_surface(nodes, tri, values, origin, normal, X)
"""
from __future__ import annotations

import numpy as np

# 8 linear tetrahedra in tet10 (ccx order: 4=(0,1), 5=(1,2), 6=(0,2), 7=(0,3), 8=(1,3), 9=(2,3))
SUBTETS = np.array([[0, 4, 6, 7], [4, 1, 5, 8], [6, 5, 2, 9], [7, 8, 9, 3],
                    [6, 8, 4, 5], [6, 8, 5, 9], [6, 8, 9, 7], [6, 8, 7, 4]])
# tri6 (corners, midpoints 01, 12, 20) → 4 linear triangles (the same as fea_gui.contour.SUBTRI)
SUBTRI = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])
_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
_EI = np.full((4, 4), -1)
for _k, (_a, _b) in enumerate(_EDGES):
    _EI[_a, _b] = _EI[_b, _a] = _k


def _unit(n) -> np.ndarray:
    n = np.asarray(n, float)
    return n / np.linalg.norm(n)


def _distance(nodes: np.ndarray, origin, n: np.ndarray) -> np.ndarray:
    """Distance from the plane; the plane is shifted by 1e-9 of the size so that no node lies exactly on it
    (a plane through a layer of nodes would otherwise give an empty section – the elements would only touch it).
    """
    size = float(np.ptp(nodes, axis=0).max()) if len(nodes) else 1.0
    return (nodes - np.asarray(origin, float)) @ n - 1e-9 * max(size, 1e-30)


def linear_tets(tets: np.ndarray) -> np.ndarray:
    """tet10 → (8M, 4) linear tetrahedra, tet4 unchanged."""
    if tets.shape[1] == 4:
        return tets
    return tets[:, SUBTETS].reshape(-1, 4)


def plane_section(nodes: np.ndarray, tets: np.ndarray, values: np.ndarray, origin, normal,
                  X: np.ndarray | None = None, elem_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Triangles of the section by the plane (o, n) and the values at their vertices.

    nodes = undeformed positions (they define the section), X = displayed positions (default nodes),
    elem_mask = only the selected elements (e.g. parts). The normals of the triangles point in the direction n.
    """
    n = _unit(normal)
    X = nodes if X is None else X
    d = _distance(nodes, origin, n)
    T = tets if elem_mask is None else tets[elem_mask]
    dt = d[T]
    cut = (dt.min(axis=1) < 0) & (dt.max(axis=1) > 0)       # pre-selection (including the middle nodes)
    E = linear_tets(T[cut])
    de = d[E]
    pos = de > 0
    npos = pos.sum(axis=1)
    keep = (npos > 0) & (npos < 4)
    E, de, pos, npos = E[keep], de[keep], pos[keep], npos[keep]
    if len(E) == 0:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    # intersections on all 6 edges (invalid ones are not used)
    a, b = E[:, _EDGES[:, 0]], E[:, _EDGES[:, 1]]           # (S, 6)
    da, db = de[:, _EDGES[:, 0]], de[:, _EDGES[:, 1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(da != db, da / (da - db), 0.5)
    IP = X[a] + t[..., None] * (X[b] - X[a])                # (S, 6, 3)
    IV = values[a] + t * (values[b] - values[a])            # (S, 6)
    rows = np.arange(len(E))
    tris_P, tris_V = [], []
    # one vertex on the opposite side → triangle from its three edges
    one = np.nonzero((npos == 1) | (npos == 3))[0]
    if len(one):
        lone = np.where(npos[one] == 1, np.argmax(pos[one], axis=1), np.argmin(pos[one], axis=1))
        others = (lone[:, None] + np.array([1, 2, 3])) % 4
        ek = _EI[lone[:, None], others]                     # (k, 3)
        r = rows[one][:, None]
        tris_P.append(IP[r, ek])
        tris_V.append(IV[r, ek])
    # two and two → quadrilateral (a,c), (a,d), (b,d), (b,c)
    two = np.nonzero(npos == 2)[0]
    if len(two):
        order = np.argsort(pos[two], axis=1, kind="stable")  # [c, d, a, b] (negative, positive)
        c_, d_, a_, b_ = order.T
        ek = np.stack([_EI[a_, c_], _EI[a_, d_], _EI[b_, d_], _EI[b_, c_]], axis=1)
        r = rows[two][:, None]
        Q, QV = IP[r, ek], IV[r, ek]
        tris_P += [Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]]
        tris_V += [QV[:, [0, 1, 2]], QV[:, [0, 2, 3]]]
    P = np.concatenate(tris_P)
    V = np.concatenate(tris_V)
    # orientation: normal in the direction n (looking from the cut-off side)
    nn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    flip = nn @ n < 0
    P[flip] = P[flip][:, [0, 2, 1]]
    V[flip] = V[flip][:, [0, 2, 1]]
    return P, V


def clip_surface(nodes: np.ndarray, tri: np.ndarray, values: np.ndarray, origin, normal,
                 X: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Surface triangles (tri6/tri3) trimmed by the plane – the side (x − o)·n ≤ 0 is kept.

    Returns (P (K, 3, 3), V (K, 3)) linear triangles with the orientation preserved.
    """
    n = _unit(normal)
    X = nodes if X is None else X
    lt = tri[:, SUBTRI].reshape(-1, 3) if tri.shape[1] == 6 else tri
    d = _distance(nodes, origin, n)
    dt = d[lt]
    inside = dt <= 0
    nin = inside.sum(axis=1)
    out_P = [X[lt[nin == 3]]]
    out_V = [values[lt[nin == 3]]]
    part = np.nonzero((nin == 1) | (nin == 2))[0]
    if len(part):
        ins = inside[part]
        # "odd" vertex: the only one inside (nin = 1) or the only one outside (nin = 2); rotate to the start
        odd = np.where(nin[part] == 1, np.argmax(ins, axis=1), np.argmin(ins, axis=1))
        roll = (odd[:, None] + np.arange(3)) % 3
        L = np.take_along_axis(lt[part], roll, axis=1)       # (k, 3): o, a, b (cyklicky)
        D = d[L]
        P3, V3 = X[L], values[L]

        def cross(i, j):
            t = (D[:, i] / (D[:, i] - D[:, j]))[:, None]
            return P3[:, i] + t * (P3[:, j] - P3[:, i]), V3[:, i] + t[:, 0] * (V3[:, j] - V3[:, i])
        Ioa, Voa = cross(0, 1)
        Iob, Vob = cross(0, 2)
        m1 = nin[part] == 1                                  # o inside → triangle o, I(o,a), I(o,b)
        if np.any(m1):
            out_P.append(np.stack([P3[m1, 0], Ioa[m1], Iob[m1]], axis=1))
            out_V.append(np.stack([V3[m1, 0], Voa[m1], Vob[m1]], axis=1))
        m2 = ~m1                                             # o outside → quadrilateral a, b, I(o,b), I(o,a)
        if np.any(m2):
            q = [P3[m2, 1], P3[m2, 2], Iob[m2], Ioa[m2]]
            qv = [V3[m2, 1], V3[m2, 2], Vob[m2], Voa[m2]]
            out_P += [np.stack([q[0], q[1], q[2]], axis=1), np.stack([q[0], q[2], q[3]], axis=1)]
            out_V += [np.stack([qv[0], qv[1], qv[2]], axis=1), np.stack([qv[0], qv[2], qv[3]], axis=1)]
    return np.concatenate(out_P), np.concatenate(out_V)


def section_outline(P: np.ndarray, V: np.ndarray | None = None) -> np.ndarray:
    """Edges on the perimeter of the section (segments (k, 2, 3)) – edges of triangles not shared with another triangle."""
    if len(P) == 0:
        return np.zeros((0, 2, 3))
    E = np.concatenate([P[:, [0, 1]], P[:, [1, 2]], P[:, [2, 0]]])
    q = np.round(E / max(np.abs(P).max(), 1e-30) * 1e9).astype(np.int64).reshape(len(E), 6)
    key = np.where(_lex_le(q[:, :3], q[:, 3:])[:, None], q, np.hstack([q[:, 3:], q[:, :3]]))
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return E[cnt[inv.ravel()] == 1]


def _lex_le(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Lexicographic a ≤ b row by row."""
    out = np.ones(len(a), bool)
    decided = np.zeros(len(a), bool)
    for k in range(a.shape[1]):
        lt, gt = a[:, k] < b[:, k], a[:, k] > b[:, k]
        out = np.where(~decided & gt, False, out)
        decided |= lt | gt
    return out
