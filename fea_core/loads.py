"""Resultant of the loads (for the equilibrium check after each calculation).

The quadratic geometry of tet10 / tri6 is integrated exactly (Gauss), so the resultant
corresponds to what ccx integrates, and the imbalance shows the real error of the solution.
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .study import Study, LoadCase, Force, Pressure, Gravity, RemoteLoad, BearingLoad, Rotation

# opposite corner to the faces S1..S4 (ccx)
_OPPOSITE = np.array([3, 2, 0, 1])

# 6-point Dunavant rule on a triangle (exact for degree 4: N_i·|J| of curved tri6)
_A, _B = 0.445948490915965, 0.091576213509771
_TRI_PTS = np.array([[_A, _A], [1 - 2 * _A, _A], [_A, 1 - 2 * _A], [_B, _B], [1 - 2 * _B, _B], [_B, 1 - 2 * _B]])
_TRI_W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3) / 2

# 5-point rule on a tetrahedron (exact for degree 3)
_TET_PTS = np.array([[0.25, 0.25, 0.25], [1 / 6, 1 / 6, 1 / 6], [0.5, 1 / 6, 1 / 6],
                     [1 / 6, 0.5, 1 / 6], [1 / 6, 1 / 6, 0.5]])
_TET_W = np.array([-4 / 5, 9 / 20, 9 / 20, 9 / 20, 9 / 20]) / 6

# 4-point rule on a tetrahedron = integration points of C3D10 in ccx (gauss3d4), point j lies near corner j, weight 1/24
GAUSS4_A, GAUSS4_B = 0.5854101966249685, 0.1381966011250105
GAUSS4 = np.array([[GAUSS4_B] * 3, [GAUSS4_A, GAUSS4_B, GAUSS4_B], [GAUSS4_B, GAUSS4_A, GAUSS4_B],
                   [GAUSS4_B, GAUSS4_B, GAUSS4_A]])
W_GAUSS4 = 1.0 / 24.0


def _tri6_dN(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives of the shape functions of tri6 (Gmsh order: corners, midpoints 01, 12, 20)."""
    l0 = 1 - xi - eta
    dxi = np.array([-(4 * l0 - 1), 4 * xi - 1, 0, 4 * (l0 - xi), 4 * eta, -4 * eta])
    deta = np.array([-(4 * l0 - 1), 0, 4 * eta - 1, -4 * xi, 4 * xi, 4 * (l0 - eta)])
    return dxi, deta


def _tet10_dN(r: float, s: float, t: float) -> np.ndarray:
    """Derivatives of the shape functions of C3D10 (ccx order), shape (3, 10)."""
    u = 1 - r - s - t
    a = -(4 * u - 1)
    return np.array([
        [a, 4 * r - 1, 0, 0, 4 * (u - r), 4 * s, -4 * s, -4 * t, 4 * t, 0],
        [a, 0, 4 * s - 1, 0, -4 * r, 4 * r, 4 * (u - s), -4 * t, 0, 4 * t],
        [a, 0, 0, 4 * t - 1, -4 * r, 0, -4 * s, 4 * (u - t), 4 * r, 4 * s],
    ])


def inv3(J: np.ndarray) -> np.ndarray:
    """Inversion of 3×3 matrices (…, 3, 3) through the adjugate (np.linalg.inv is slow for small matrices)."""
    a, b, c = J[..., 0, 0], J[..., 0, 1], J[..., 0, 2]
    d, e, f = J[..., 1, 0], J[..., 1, 1], J[..., 1, 2]
    g, h, i = J[..., 2, 0], J[..., 2, 1], J[..., 2, 2]
    A, B, C = e * i - f * h, f * g - d * i, d * h - e * g
    det = a * A + b * B + c * C
    out = np.empty_like(J)
    out[..., 0, 0], out[..., 0, 1], out[..., 0, 2] = A, c * h - b * i, b * f - c * e
    out[..., 1, 0], out[..., 1, 1], out[..., 1, 2] = B, a * i - c * g, c * d - a * f
    out[..., 2, 0], out[..., 2, 1], out[..., 2, 2] = C, b * g - a * h, a * e - b * d
    return out / det[..., None, None]


def scatter_add(F: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    """F[idx[:, j]] += vals[:, j] step by step for the columns j (as a loop over j with np.add.at), but through np.bincount.

    idx (K, C) nodes, vals (K, C, 3) contributions; F (N, 3) is changed in place. The order of summation in each node is
    the same as with np.add.at per column (the original value of F, then column 0 row by row, column 1 …) →
    the result is bitwise equal, ~7× faster (np.add.at is slow in numpy 1.26).
    """
    n = len(F)
    if idx.size == 0:
        return
    order = np.concatenate([np.arange(n), idx.T.ravel()])
    v = vals.transpose(1, 0, 2).reshape(-1, 3)
    for k in range(3):
        F[:, k] = np.bincount(order, weights=np.concatenate([F[:, k], v[:, k]]), minlength=n)


def _tet10_dets(mesh: Mesh) -> np.ndarray:
    """det J at the 5 Gauss points _TET_PTS per element of tet10, (M, 5) – mesh cache (weight and volumes)."""
    def compute():
        M = len(mesh.tets)
        # = np.tensordot(dN, X, axes=(1, 1)) (the same np.dot over the same matrix → bitwise equal), but
        # the transposition of X (M, 10, 3) → (10, 3M) only once instead of at every point
        Xt = np.ascontiguousarray(mesh.nodes[mesh.tets].transpose(1, 0, 2)).reshape(10, -1)
        out = np.empty((M, len(_TET_PTS)))
        for i, p in enumerate(_TET_PTS):
            dN = _tet10_dN(*p)
            J = np.dot(dN, Xt).reshape(3, M, 3).swapaxes(0, 1)
            out[:, i] = (J[:, 0, 0] * (J[:, 1, 1] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 1])
                         - J[:, 0, 1] * (J[:, 1, 0] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 0])
                         + J[:, 0, 2] * (J[:, 1, 0] * J[:, 2, 1] - J[:, 1, 1] * J[:, 2, 0]))
        return out
    return mesh.cached("tet10_dets", ("nodes", "tets"), compute)


def vector_area(mesh: Mesh, name: str) -> np.ndarray:
    """∫ n dA over the selection with the OUTWARD normal (relative to the body) [mm²]."""
    tri = mesh.surfaces[name]
    X = mesh.nodes[tri]                                  # (K, 6|3, 3)
    if tri.shape[1] == 6:
        va = np.zeros((len(tri), 3))
        for (xi, eta), w in zip(_TRI_PTS, _TRI_W):
            dxi, deta = _tri6_dN(xi, eta)
            a = np.einsum("i,kij->kj", dxi, X)
            b = np.einsum("i,kij->kj", deta, X)
            va += w * np.cross(a, b)
    else:
        va = 0.5 * np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
    # orientation outward from the element: against the opposite corner
    elems, fno = mesh.element_faces(name)
    opp = mesh.nodes[mesh.tets[elems, _OPPOSITE[fno - 1]]]
    out = np.einsum("kj,kj->k", va, X[:, :3].mean(axis=1) - opp) < 0
    va[out] *= -1
    return va.sum(axis=0)


def part_volumes(mesh: Mesh) -> np.ndarray:
    return np.bincount(mesh.tet_part, weights=element_volumes(mesh), minlength=len(mesh.part_names))


def element_volumes(mesh: Mesh) -> np.ndarray:
    if mesh.tets.shape[1] == 10:
        dets = _tet10_dets(mesh)
        vol = np.zeros(len(dets))
        for i, w in enumerate(_TET_W):
            vol += w * dets[:, i]
    else:
        X = mesh.nodes[mesh.tets]                        # (M, 4, 3)
        vol = np.abs(np.einsum("mi,mi->m", X[:, 1] - X[:, 0],
                               np.cross(X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]))) / 6
    return vol


def _tri6_N(xi: float, eta: float) -> np.ndarray:
    l0 = 1 - xi - eta
    return np.array([l0 * (2 * l0 - 1), xi * (2 * xi - 1), eta * (2 * eta - 1),
                     4 * l0 * xi, 4 * xi * eta, 4 * eta * l0])


# ---------------------------------------------------------------- integration over faces, edges, volume

def _tri_shape(xi: float, eta: float, nn: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N, dN/dξ, dN/dη of the tri6 triangle (Gmsh order) or tri3."""
    if nn == 6:
        return (_tri6_N(xi, eta), *_tri6_dN(xi, eta))
    return (np.array([1 - xi - eta, xi, eta]), np.array([-1.0, 1.0, 0.0]), np.array([-1.0, 0.0, 1.0]))


def _sub_rule(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Dunavant 6 points on each of the level² sub-triangles (for an integrand with a kink – the level of hydrostatics)."""
    if level <= 1:
        return _TRI_PTS, _TRI_W
    pts, ws = [], []
    h = 1.0 / level
    for i in range(level):
        for j in range(level - i):
            tris = [((i, j), (i + 1, j), (i, j + 1))]
            if i + j < level - 1:
                tris.append(((i + 1, j), (i + 1, j + 1), (i, j + 1)))
            for a, b, c in tris:
                A, B, C = (np.array(v, float) * h for v in (a, b, c))
                for (u, v), w in zip(_TRI_PTS, _TRI_W):
                    pts.append(A + u * (B - A) + v * (C - A))
                    ws.append(w * h * h)
    return np.array(pts), np.array(ws)


def surface_quadrature(mesh: Mesh, tri: np.ndarray, level: int = 1, outward: bool = True):
    """Gauss points of the triangles of a selection: N (G, nn), points x (K, G, 3), vectors n·dA·w (K, G, 3) – with the outward normal
    relative to the body (outward, the triangles must be element faces), otherwise in the orientation of the triangle. Exact for curved tri6.
    """
    X = mesh.nodes[tri]
    nn = tri.shape[1]
    pts, ws = _sub_rule(level)
    Ns, xs, nAs = [], [], []
    for (xi, eta), w in zip(pts, ws):
        N, dxi, deta = _tri_shape(xi, eta, nn)
        a = np.einsum("i,kij->kj", dxi, X)
        b = np.einsum("i,kij->kj", deta, X)
        Ns.append(N)
        xs.append(np.einsum("i,kij->kj", N, X))
        nAs.append(w * np.cross(a, b))
    N = np.array(Ns)
    x = np.stack(xs, axis=1)
    nA = np.stack(nAs, axis=1)
    if outward and len(tri):
        elems, fno = surface_element_faces(mesh, tri)
        opp = mesh.nodes[mesh.tets[elems, _OPPOSITE[fno - 1]]]
        sign = np.where(np.einsum("kj,kj->k", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]),
                                  X[:, :3].mean(axis=1) - opp) < 0, -1.0, 1.0)
        nA *= sign[:, None, None]
    return N, x, nA


def surface_element_faces(mesh: Mesh, tri: np.ndarray):
    from .mesh import surface_element_faces as sef
    return sef(mesh.tets, tri, mesh._face_index())


_LINE_PTS = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
_LINE_W = np.array([5 / 9, 8 / 9, 5 / 9])


def line_quadrature(nodes: np.ndarray, seg: np.ndarray):
    """Gauss points of the segments of an edge (Gmsh line3: ends, midpoint; or line2): N (G, nn), points x (K, G, 3),
    w·|dx/du| (K, G). 3 points – exact for a straight edge; for an arc the error ~ (h/R)⁴.
    """
    X = nodes[seg]
    nn = seg.shape[1]
    Ns, xs, js = [], [], []
    for u, w in zip(_LINE_PTS, _LINE_W):
        if nn == 3:
            N = np.array([u * (u - 1) / 2, u * (u + 1) / 2, 1 - u * u])
            dN = np.array([u - 0.5, u + 0.5, -2 * u])
        else:
            N = np.array([(1 - u) / 2, (1 + u) / 2])
            dN = np.array([-0.5, 0.5])
        Ns.append(N)
        xs.append(np.einsum("i,kij->kj", N, X))
        js.append(w * np.linalg.norm(np.einsum("i,kij->kj", dN, X), axis=1))
    return np.array(Ns), np.stack(xs, axis=1), np.stack(js, axis=1)


def _scatter_points(n: int, idx: np.ndarray, N: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Σ_g N_i(g)·vals(k, g) into the nodes idx (K, nn); vals (K, G, 3) → (n, 3)."""
    fe = np.einsum("gi,kgj->kij", N, vals)
    F = np.zeros((n, 3))
    scatter_add(F, idx, fe)
    return F


def force_nodal(study: Study, mesh: Mesh, l: Force) -> np.ndarray:
    """Consistent nodal forces (N, 3) of a Force according to the kind of selection (faces / edges / points) and the coordinate system.

    Faces in the global / Cartesian system = the same nodal forces as before (tri6: corners 0, midpoints 1/3 of the element force according to
    the area of the corner triangle → the resultant exactly F). Cylindrical system: density F/A in the axes (r, θ, z) of the Gauss point,
    A = ∫dA. Edges: density F/L along the length (3 Gauss points, L by the same integration → the resultant in the Cartesian system
    exactly F). Points: F/n to the node of each point (for a cylindrical one in the axes of that point).
    """
    from . import csys
    n = len(mesh.nodes)
    sel = study.selections.get(l.selection)
    kind = sel.entity if sel is not None else "faces"
    what = f"Force '{l.selection}'"
    cs = csys.get(study, l.csys)
    cyl = cs is not None and cs.kind == "cylindrical"
    if kind == "points":
        idx = np.asarray(mesh.points[l.selection], dtype=np.int64)
        vals = csys.field_global(study, l.csys, l.vector, mesh.nodes[idx], what) / len(idx)
        F = np.zeros((n, 3))
        scatter_add(F, idx[:, None], vals[:, None, :])
        return F
    if kind == "edges":
        seg = mesh.edges[l.selection]
        N, x, wj = line_quadrature(mesh.nodes, seg)
        L = float(wj.sum())
        if L <= 0:
            raise ValueError(f"{what}: edge has zero length in the mesh")
        q = csys.field_global(study, l.csys, np.asarray(l.vector, float) / L, x.reshape(-1, 3), what).reshape(x.shape)
        return _scatter_points(n, seg, N, q * wj[:, :, None])
    tri = mesh.surfaces[l.selection]
    if not cyl:
        from .mesh import tri_areas
        from .ccx.inp import consistent_nodal_forces
        v = csys.vector_global(study, l.csys, l.vector, what)
        t = v / float(tri_areas(mesh.nodes, tri).sum())
        cnf = consistent_nodal_forces(mesh.nodes, tri, t)
        F = np.zeros((n, 3))
        if cnf:
            F[np.fromiter(cnf.keys(), np.int64, len(cnf))] = np.array(list(cnf.values()))
        return F
    # triangles at the axis (the direction e_r, e_θ turns sharply in them; a face across the axis – the end face of a shaft) finer, otherwise
    # Gauss integration leaves a false resultant (end face Ø20, tangential force: 4e-4·F_θ → after refinement ~1e-6·F_θ)
    X = mesh.nodes[tri[:, :3]]
    R = csys.matrix(cs)
    d = X - np.asarray(cs.origin, float)
    r = np.linalg.norm(d - np.einsum("kij,j->ki", d, R[2])[:, :, None] * R[2], axis=2).min(axis=1)
    h = np.linalg.norm(X - np.roll(X, 1, axis=1), axis=2).max(axis=1)
    near = r < 3 * h
    parts = []
    for sel_t, level in ((~near, 1), (near, 16)):
        if sel_t.any():
            N, x, nA = surface_quadrature(mesh, tri[sel_t], level, outward=False)
            parts.append((tri[sel_t], N, x, np.linalg.norm(nA, axis=2)))
    A = sum(float(dA.sum()) for _, _, _, dA in parts)
    F = np.zeros((n, 3))
    for t_idx, N, x, dA in parts:
        t = csys.field_global(study, l.csys, np.asarray(l.vector, float) / A, x.reshape(-1, 3), what).reshape(x.shape)
        F += _scatter_points(n, t_idx, N, t * dA[:, :, None])
    return F


def pressure_value(l: Pressure, x: np.ndarray) -> np.ndarray:
    """Pressure [MPa] at points x (..., 3)."""
    if not l.variable:
        return np.full(x.shape[:-1], float(l.value))
    g = np.asarray(l.gradient, float)
    o = np.asarray(l.origin if l.origin is not None else [0.0, 0.0, 0.0], float)
    lin = (x - o) @ g
    if l.clip:
        lin = np.maximum(lin, 0.0)
    return float(l.value) + lin


def pressure_nodal(mesh: Mesh, l: Pressure) -> np.ndarray:
    """Consistent nodal forces of pressure (N, 3): f_i = −∫ p(x) N_i n dA (n outward). Variable pressure with a kink (level,
    clip) – the triangles that the level cuts are integrated more finely (8² sub-triangles × 6 points).
    """
    n = len(mesh.nodes)
    tri = mesh.surfaces[l.selection]
    F = np.zeros((n, 3))
    if len(tri) == 0:
        return F
    if not l.variable:
        N, x, nA = surface_quadrature(mesh, tri)
        return _scatter_points(n, tri, N, -float(l.value) * nA)
    cut = np.zeros(len(tri), bool)
    if l.clip:
        g = np.asarray(l.gradient, float)
        o = np.asarray(l.origin if l.origin is not None else [0.0, 0.0, 0.0], float)
        d = (mesh.nodes[tri] - o) @ g
        cut = (d.min(axis=1) < 0) & (d.max(axis=1) > 0)
    for part, level in ((~cut, 1), (cut, 8)):
        if not part.any():
            continue
        t = tri[part]
        N, x, nA = surface_quadrature(mesh, t, level)
        F += _scatter_points(n, t, N, -pressure_value(l, x)[:, :, None] * nA)
    return F


def rotation_forces(study: Study, mesh: Mesh, l: Rotation) -> np.ndarray:
    """Consistent nodal forces from rotation (N, 3): ∫ ρ N_i (ω²·r⊥ − α·(a × r)) dV, r = x − point of the axis, a = unit axis.
    Tet10: 5-point rule (exact for straight elements – integrand of degree 3), ρ per element (also 3D printing).
    """
    from .printing import element_density
    from .results import tet10_shape
    n = len(mesh.nodes)
    F = np.zeros((n, 3))
    a = np.asarray(l.axis, float)
    a = a / np.linalg.norm(a)
    p0 = np.asarray(l.point, float)
    w2, al = float(l.omega) ** 2, float(l.alpha)
    sel = np.ones(len(mesh.tets), bool)
    if l.parts:
        idx = [i for i, pn in enumerate(mesh.part_names) if pn in set(l.parts)]
        sel = np.isin(mesh.tet_part, idx)
    el = np.nonzero(sel)[0]
    if len(el) == 0 or (w2 == 0 and al == 0):
        return F
    rho = element_density(study, mesh)[el]
    tets = mesh.tets[el]
    X = mesh.nodes[tets]                               # (M, nn, 3)
    nn = tets.shape[1]
    if nn == 10:
        dets = _tet10_dets(mesh)[el]
    else:
        vol6 = np.abs(np.einsum("mi,mi->m", X[:, 1] - X[:, 0],
                                np.cross(X[:, 2] - X[:, 0], X[:, 3] - X[:, 0])))
        dets = np.repeat(vol6[:, None], len(_TET_PTS), axis=1)
    fe = np.zeros((len(el), nn, 3))
    for i, (pt, w) in enumerate(zip(_TET_PTS, _TET_W)):
        if nn == 10:
            N = tet10_shape(np.asarray(pt))
        else:
            N = np.array([1 - pt.sum(), pt[0], pt[1], pt[2]])
        x = np.einsum("i,mij->mj", N, X)
        r = x - p0
        rperp = r - np.outer(r @ a, a)
        b = w2 * rperp - al * np.cross(a, r)
        dv = w * dets[:, i] * rho
        fe += (dv[:, None] * b)[:, None, :] * N[None, :, None]
    scatter_add(F, tets, fe)
    return F


def inertia_forces(study: Study, mesh: Mesh, l) -> np.ndarray:
    """Consistent nodal forces of the inertia of the rigid motion (study.Inertia, d'Alembert): ∫ ρ N_i b dV,
    b = −(a₀ + α × r + ω × (ω × r)), r = x − point. The same integration as rotation_forces (5 points of tet10).
    """
    from .printing import element_density
    from .results import tet10_shape
    n = len(mesh.nodes)
    F = np.zeros((n, 3))
    a0 = np.asarray(l.accel, float)
    om = np.asarray(l.omega, float)
    al = np.asarray(l.alpha, float)
    p0 = np.asarray(l.point, float)
    W = np.asarray(l.wmat, float).reshape(3, 3) if getattr(l, "wmat", None) is not None else         np.outer(om, om) - (om @ om) * np.eye(3)
    if not (a0.any() or W.any() or al.any()):
        return F
    sel = np.ones(len(mesh.tets), bool)
    if l.parts:
        idx = [i for i, pn in enumerate(mesh.part_names) if pn in set(l.parts)]
        sel = np.isin(mesh.tet_part, idx)
    el = np.nonzero(sel)[0]
    if len(el) == 0:
        return F
    rho = element_density(study, mesh)[el]
    tets = mesh.tets[el]
    X = mesh.nodes[tets]
    nn = tets.shape[1]
    if nn == 10:
        dets = _tet10_dets(mesh)[el]
    else:
        vol6 = np.abs(np.einsum("mi,mi->m", X[:, 1] - X[:, 0], np.cross(X[:, 2] - X[:, 0], X[:, 3] - X[:, 0])))
        dets = np.repeat(vol6[:, None], len(_TET_PTS), axis=1)
    fe = np.zeros((len(el), nn, 3))
    for i, (pt, w) in enumerate(zip(_TET_PTS, _TET_W)):
        N = tet10_shape(np.asarray(pt)) if nn == 10 else np.array([1 - pt.sum(), pt[0], pt[1], pt[2]])
        r = np.einsum("i,mij->mj", N, X) - p0
        b = -(a0[None, :] + np.cross(al, r) + r @ W.T)
        dv = w * dets[:, i] * rho
        fe += (dv[:, None] * b)[:, None, :] * N[None, :, None]
    scatter_add(F, tets, fe)
    return F


def global_vector(study: Study, l, vec=None) -> np.ndarray:
    """Load vector (force / acceleration) in the global axes – Cartesian local system `l.csys`."""
    from . import csys
    v = l.vector if vec is None else vec
    return csys.vector_global(study, getattr(l, "csys", None), v, f"Load '{getattr(l, 'selection', l.kind)}'")


def remote_vectors(study: Study, l: RemoteLoad) -> tuple[np.ndarray, np.ndarray]:
    """(force, moment) of a remote load in the global axes."""
    return global_vector(study, l, l.force), global_vector(study, l, l.moment)


def nodal_loads(study: Study, mesh: Mesh, lc: LoadCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Consistent nodal forces of the state (N, 3) + force and moment at remote points.

    Used only for the equilibrium check (splitting the load into constrained / free nodes);
    the resultant is computed exactly in load_resultant.
    Returns (F_nodes, F_remote, Mr_remote) – moment of the remote loads about the origin.
    """
    F = np.zeros((len(mesh.nodes), 3))
    Fr, Mr = np.zeros(3), np.zeros(3)
    for l in lc.loads:
        if isinstance(l, Force):
            F += force_nodal(study, mesh, l)
        elif isinstance(l, Pressure):                # constant and variable – the same as *CLOAD in ccx/inp.py
            F += pressure_nodal(mesh, l)
        elif isinstance(l, Rotation):
            F += rotation_forces(study, mesh, l)
        elif getattr(l, "kind", None) == "inertia":
            F += inertia_forces(study, mesh, l)
        elif isinstance(l, Gravity):
            F += gravity_nodal_forces(study, mesh, global_vector(study, l))
        elif isinstance(l, BearingLoad):
            from .cylinder import bearing_forces
            F += bearing_forces(mesh, l.selection, global_vector(study, l))
        elif isinstance(l, RemoteLoad):
            f, m = remote_vectors(study, l)
            Fr += f
            Mr += m + np.cross(np.asarray(l.point, float), f)
    if getattr(study, "masses", None):                 # added mass: surface mass to the nodes of the mesh, point mass outside the mesh
        from .dynamics.mass import static_resultants
        Fm, fr, mr = static_resultants(study, mesh, lc)
        F += Fm
        Fr += fr
        Mr += mr
    return F, Fr, Mr


def gravity_nodal_forces(study: Study, mesh: Mesh, vector) -> np.ndarray:
    """Consistent nodal forces of the weight ρ·g (exact integration of tet10). For ccx instead of *DLOAD GRAV
    when the model contains U1 elements (bolts) – they cannot take body forces (verified 12. 10. 2026).
    """
    from .results import tet10_shape
    F = np.zeros((len(mesh.nodes), 3))
    g = np.asarray(vector, float)
    from .printing import element_density
    rho = element_density(study, mesh)
    if mesh.tets.shape[1] == 10:
        dets = _tet10_dets(mesh)
        fe_nodes = np.zeros((len(dets), 10))
        for i, (p, w) in enumerate(zip(_TET_PTS, _TET_W)):
            dv = w * dets[:, i] * rho
            N = tet10_shape(np.asarray(p))
            fe_nodes += dv[:, None] * N[None, :]
        # = np.add.at(F, tets[:, i], fe_nodes[:, i, None] * g) per column i; F is zero → no pre-sorting of F
        order = mesh.tets.T.ravel()
        fT = fe_nodes.T.ravel()
        for k in range(3):
            F[:, k] = np.bincount(order, weights=fT * g[k], minlength=len(F))
    else:
        X = mesh.nodes[mesh.tets]
        vol = np.abs(np.einsum("mi,mi->m", X[:, 1] - X[:, 0],
                               np.cross(X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]))) / 6
        fe = (vol * rho / 4)[:, None] * g[None, :]
        scatter_add(F, mesh.tets, np.repeat(fe[:, None, :], 4, axis=1))
    return F


def distributing_forces(mesh: Mesh, selection: str, point, force, moment) -> np.ndarray:
    """Nodal forces of a distributing constraint (RBE3 / ccx *DISTRIBUTING) without added stiffness.

    Weights w_i = ∫N_i dA. Force and moment transferred to the weighted centroid c:
        f_i = w_i/W · F + w_i · (Γ⁻¹ M_c) × r_i,  r_i = x_i − c,
        Γ = Σ w_i (|r_i|² I − r_i r_iᵀ),  M_c = M + (p − c) × F.
    The resultant f_i gives exactly F and M (about the point p). Returns a field (N, 3).
    """
    from .ccx.inp import consistent_nodal_forces
    w = consistent_nodal_forces(mesh.nodes, mesh.surfaces[selection], np.array([1.0, 0.0, 0.0]))
    idx = np.array([n for n, f in w.items() if f[0] > 0])
    wi = np.array([w[n][0] for n in idx])
    X = mesh.nodes[idx]
    W = wi.sum()
    c = (wi[:, None] * X).sum(axis=0) / W
    r = X - c
    F = np.asarray(force, float)
    Mc = np.asarray(moment, float) + np.cross(np.asarray(point, float) - c, F)
    G = (wi[:, None, None] * ((r * r).sum(axis=1)[:, None, None] * np.eye(3) - r[:, :, None] * r[:, None, :])).sum(axis=0)
    om = np.linalg.solve(G, Mc) if np.linalg.norm(Mc) > 0 else np.zeros(3)
    out = np.zeros((len(mesh.nodes), 3))
    out[idx] = wi[:, None] * (F / W) + wi[:, None] * np.cross(om, r)
    return out


def load_resultant(study: Study, mesh: Mesh, lc: LoadCase) -> np.ndarray:
    """Resulting external force of the load state [N]."""
    from . import csys
    F = np.zeros(3)
    for l in lc.loads:
        if isinstance(l, Force):
            cs = csys.get(study, l.csys)
            if cs is not None and cs.kind == "cylindrical":
                F += force_nodal(study, mesh, l).sum(axis=0)
            else:
                F += global_vector(study, l)
        elif isinstance(l, Pressure) and l.variable:
            F += pressure_nodal(mesh, l).sum(axis=0)
        elif isinstance(l, Pressure):
            F += -l.value * vector_area(mesh, l.selection)
        elif isinstance(l, Gravity):
            from .printing import element_density
            rho = element_density(study, mesh)
            F += (element_volumes(mesh) * rho).sum() * global_vector(study, l)
        elif isinstance(l, Rotation):
            F += rotation_forces(study, mesh, l).sum(axis=0)
        elif getattr(l, "kind", None) == "inertia":
            F += inertia_forces(study, mesh, l).sum(axis=0)
        elif isinstance(l, BearingLoad):
            F += global_vector(study, l)
        elif isinstance(l, RemoteLoad):
            F += remote_vectors(study, l)[0]
    if getattr(study, "masses", None):
        from .dynamics.mass import static_resultants
        Fm, fr, _ = static_resultants(study, mesh, lc)
        F += Fm.sum(axis=0) + fr
    return F
