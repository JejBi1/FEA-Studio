"""Cylindrical faces in the mesh: axis from the nodes, cylindrical support (*EQUATION) and bearing load.

The axis is fitted through the CORNER nodes of the surface triangles of the selection (they always lie exactly on the CAD
face; the middle nodes do not, in the fallback with straight edges). Works also for several coaxial faces
of the same radius (a hole split into half-cylinders). The cylindrical support also accepts a selection of several non-coaxial
holes (selection_axes: connected parts of the selection → the axis of each part, coaxial parts of the same radius are
merged); the reaction stays one per selection (sum). The bearing load needs one axis (selection_axis).

Cylindrical support in ccx: verified 12. 10. 2026 (scratch/cyl/exp_transform.py) – *TRANSFORM, TYPE=C
switches ALL *BOUNDARY (also *CLOAD) at the node to local axes, so a symmetry ux = 0 on an edge
shared with the cylinder would silently become a radial component. Therefore *EQUATION in global DOF:
u·e = 0 for every constrained direction e (radial / tangential / axial), dependent DOF chosen by elimination
(RREF) outside the DOF held by *BOUNDARY. The output U and RF in the .frd are global, RF = K·u includes
the force of the constraint equation too → reaction = RF − F (projection onto the constrained directions).
"""
from __future__ import annotations

import zlib

import numpy as np

from .mesh import Mesh


class CylinderError(ValueError):
    pass


# ------------------------------------------------------------------ fitting a cylinder

def fit_cylinder(points: np.ndarray, normals: np.ndarray | None = None,
                 weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fits a cylinder to the points. Returns (point on the axis nearest to the centroid of the points, unit direction of the axis, radius,
    max. deviation of the points from the cylinder). normals (n, 3) = normals of the facets for the initial estimate of the axis.
    """
    from scipy.optimize import least_squares
    X = np.asarray(points, float)
    if len(X) < 6:
        raise CylinderError("Too few nodes to fit a cylinder")
    x0 = X.mean(axis=0)
    size = float(np.linalg.norm(X.max(axis=0) - X.min(axis=0)))
    # direction of the axis: the normals of the cylinder are perpendicular to the axis → eigenvector of the smallest eigenvalue of Σ w n nᵀ
    if normals is not None and len(normals):
        n = np.asarray(normals, float)
        w = np.ones(len(n)) if weights is None else np.asarray(weights, float)
        M = (n * w[:, None]).T @ n
        lam, vec = np.linalg.eigh(M)
        if lam[1] <= 1e-8 * lam[2]:
            raise CylinderError("The face is not curved (plane) – the cylinder axis cannot be determined")
        d = vec[:, 0]
    else:
        lam, vec = np.linalg.eigh(np.cov((X - x0).T))
        d = vec[:, 2]
    e1 = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    # circle in the plane perpendicular to the axis (Kåsa, linear least squares)
    q = np.column_stack([(X - x0) @ e1, (X - x0) @ e2])
    A = np.column_stack([2 * q, np.ones(len(q))])
    sol, *_ = np.linalg.lstsq(A, (q ** 2).sum(axis=1), rcond=None)
    a, b, c = sol
    r0 = float(np.sqrt(max(c + a * a + b * b, 1e-30)))

    def frame(p):
        dd = d + p[0] * e1 + p[1] * e2
        dd /= np.linalg.norm(dd)
        return x0 + p[2] * e1 + p[3] * e2, dd

    def resid(p):
        cc, dd = frame(p)
        v = X - cc
        v = v - ((v @ dd)[:, None]) * dd
        return np.linalg.norm(v, axis=1) - p[4]

    sc = max(r0, 1e-9)
    p0 = np.array([0.0, 0.0, a, b, r0])
    sol = least_squares(resid, p0, x_scale=[1.0, 1.0, sc, sc, sc], xtol=1e-14, ftol=1e-14, gtol=1e-14,
                        max_nfev=200)
    cc, dd = frame(sol.x)
    r = float(abs(sol.x[4]))
    err = float(np.abs(resid(sol.x)).max())
    if not np.isfinite(r) or r > 1e3 * max(size, 1e-9):
        raise CylinderError("The face is not cylindrical (radius comes out infinite)")
    # point on the axis nearest to the centroid of the points, direction with the positive largest component (stable sign)
    cc = cc + ((x0 - cc) @ dd) * dd
    k = int(np.argmax(np.abs(dd)))
    if dd[k] < 0:
        dd = -dd
    return cc, dd, r, err


def _fit_tris(mesh: Mesh, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit of a cylinder through the corner nodes of the triangles (normals weighted by area for the estimate of the axis)."""
    corners = np.unique(tri[:, :3])
    X = mesh.nodes[tri[:, :3]]
    n = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
    a = np.linalg.norm(n, axis=1)
    ok = a > 0
    return fit_cylinder(mesh.nodes[corners], n[ok] / a[ok, None], a[ok])


def selection_axis(mesh: Mesh, name: str, tol: float = 1e-3) -> tuple[np.ndarray, np.ndarray, float]:
    """(point on the axis, axis direction, radius) of a cylindrical selection from the mesh; error if the selection is not one cylinder."""
    tri = mesh.surfaces[name]
    if len(tri) == 0:
        raise CylinderError(f"Selection '{name}' has no faces in the mesh")
    try:
        c, d, r, err = _fit_tris(mesh, tri)
    except CylinderError as e:
        raise CylinderError(f"Selection '{name}': {e}") from None
    if err > tol * r + 1e-9:
        raise CylinderError(f"Selection '{name}' is not a single cylindrical face (node deviation from the cylinder {err:.3g} mm at radius {r:.4g} mm) – select coaxial cylinders of the same radius")
    return c, d, r


def _tri_components(tri: np.ndarray) -> np.ndarray:
    """Connected parts of the triangles (shared corner node) → number of the part per triangle."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    corners = tri[:, :3]
    uniq, inv = np.unique(corners, return_inverse=True)
    inv = inv.reshape(-1, 3)
    i = np.concatenate([inv[:, 0], inv[:, 1]])
    j = np.concatenate([inv[:, 1], inv[:, 2]])
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(uniq), len(uniq)))
    _, lab = connected_components(g, directed=False)
    return lab[inv[:, 0]]


def selection_axes(mesh: Mesh, name: str, tol: float = 1e-3) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Groups of coaxial cylinders of a selection: [(nodes of the group, point on the axis, axis direction, radius)].

    A selection = one cylinder (also one split into several coaxial faces of the same radius) → one group, the same as
    selection_axis. Otherwise (e.g. all holes of a flange in one selection) the triangles are split into connected parts,
    each is fitted by a cylinder separately, and the parts with the same axis and radius are merged. A part that is not
    a cylinder = error (the selection is neither a cylindrical face nor a group of cylindrical holes).
    """
    tri = mesh.surfaces[name]
    if len(tri) == 0:
        raise CylinderError(f"Selection '{name}' has no faces in the mesh")
    try:
        c, d, r = selection_axis(mesh, name, tol)
        return [(np.unique(tri), c, d, r)]
    except CylinderError as e:
        first = e
    lab = _tri_components(tri)
    ncomp = int(lab.max()) + 1
    if ncomp < 2:
        raise first
    fits = []
    for k in range(ncomp):
        t = tri[lab == k]
        where = ", ".join(f"{v:.4g}" for v in mesh.nodes[np.unique(t[:, :3])].mean(axis=0))
        try:
            c, d, r, err = _fit_tris(mesh, t)
        except CylinderError as e:
            raise CylinderError(f"Selection '{name}': part {k + 1} of {ncomp} (near point ({where}) mm) is not a cylindrical face – {e}") from None
        if err > tol * r + 1e-9:
            raise CylinderError(f"Selection '{name}' is neither a cylindrical face nor a group of cylindrical holes: part {k + 1} of {ncomp} (near point ({where}) mm) deviates from a cylinder by {err:.3g} mm at radius {r:.4g} mm – select cylindrical faces only (holes / pins)")
        fits.append((c, d, r))
    # merging of coaxial parts of the same radius (a hole split into parts that do not touch)
    group = list(range(ncomp))
    for a in range(ncomp):
        for b in range(a):
            ca, da, ra = fits[a]
            cb, db, rb = fits[b]
            v = cb - ca
            off = np.linalg.norm(v - (v @ da) * da)
            if abs(ra - rb) <= tol * ra and abs(abs(da @ db) - 1.0) <= tol ** 2 and off <= tol * ra:
                old, new = group[a], group[b]
                group = [new if g == old else g for g in group]
                break
    out = []
    for g in sorted(set(group)):
        ks = [k for k in range(ncomp) if group[k] == g]
        t = tri[np.isin(lab, ks)]
        if len(ks) == 1:
            c, d, r = fits[ks[0]]
        else:
            c, d, r, err = _fit_tris(mesh, t)
            if err > tol * r + 1e-9:                  # merging does not fit → parts separately
                for k in ks:
                    out.append((np.unique(tri[lab == k]), *fits[k]))
                continue
        out.append((np.unique(t), c, d, r))
    return out


def radial_frames(points: np.ndarray, c: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit radial and tangential directions at the points (e_t = d × e_r)."""
    v = np.asarray(points, float) - c
    v = v - ((v @ d)[:, None]) * d
    rho = np.linalg.norm(v, axis=1)
    if np.any(rho < 1e-9):
        raise CylinderError("A node of the cylindrical face lies on the cylinder axis")
    er = v / rho[:, None]
    return er, np.cross(d, er)


# ------------------------------------------------------------------ cylindrical support

def _crc(a) -> tuple | None:
    if a is None:
        return None
    a = np.ascontiguousarray(a)
    return (a.shape, a.dtype.str, zlib.crc32(a))


def _support_key(study, mesh: Mesh, kinds: tuple) -> tuple:
    """Fingerprint of everything the cylindrical-support fields depend on: the constraints of `kinds` (dataclass repr = all fields),
    the mesh nodes and the triangles of their selections (CRC32 – also a change in place). Used for the cache on the mesh."""
    cons = tuple(repr(c) for c in study.constraints if isinstance(c, kinds))
    sels = sorted({c.selection for c in study.constraints if isinstance(c, kinds)})
    return (cons, _crc(mesh.nodes), tuple((n, _crc(mesh.surfaces.get(n))) for n in sels))


def _mesh_memo(mesh: Mesh, key: str, fp: tuple, compute):
    """Value cached on the mesh object under `key` while the fingerprint `fp` is the same (repeated .inp writing on one mesh –
    topology optimization, variants, convergence). The caller copies what it hands out."""
    store = getattr(mesh, "_cache", None)
    if not isinstance(store, dict):
        return compute()
    hit = store.get(key)
    if hit is not None and hit[0] == fp:
        return hit[1]
    val = compute()
    store[key] = (fp, val)
    return val


def support_directions(study, mesh: Mesh) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Selection of a cylindrical support -> (nodes, directions (n, k, 3)) of the constrained components (orthonormal per node).
    Cached on the mesh (axis fitting of every selection); the arrays returned are copies."""
    from .study import CylindricalSupport
    if not any(isinstance(c, CylindricalSupport) for c in study.constraints):
        return {}
    val = _mesh_memo(mesh, "cyl_support_directions", _support_key(study, mesh, (CylindricalSupport,)),
                     lambda: _support_directions(study, mesh))
    return {k: (idx.copy(), D.copy()) for k, (idx, D) in val.items()}


def _support_directions(study, mesh: Mesh) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from .study import CylindricalSupport
    out = {}
    for c in study.constraints:
        if not isinstance(c, CylindricalSupport):
            continue
        comps = [k for k, on in (("r", c.radial), ("t", c.tangential), ("a", c.axial)) if on]
        if not comps:
            continue
        groups = selection_axes(mesh, c.selection)          # several non-coaxial holes → each with its own axis
        if len(groups) == 1:
            idx, c0, d, _ = groups[0]
            er, et = radial_frames(mesh.nodes[idx], c0, d)
            dirs = {"r": er, "t": et, "a": np.broadcast_to(d, er.shape)}
            new = np.stack([dirs[k] for k in comps], axis=1)
        else:
            idx = np.concatenate([g[0] for g in groups])
            parts = []
            for gi, c0, d, _ in groups:
                er, et = radial_frames(mesh.nodes[gi], c0, d)
                dirs = {"r": er, "t": et, "a": np.broadcast_to(d, er.shape)}
                parts.append(np.stack([dirs[k] for k in comps], axis=1))
            order = np.argsort(idx, kind="stable")          # nodes in ascending order as surface_nodes
            idx, new = idx[order], np.concatenate(parts)[order]
        if c.selection in out:                              # two supports on the same selection → unite
            idx0, old = out[c.selection]
            new = _orthonormal_union(np.concatenate([old, new], axis=1))
        out[c.selection] = (idx, new)
    return out


def _orthonormal_union(D: np.ndarray) -> np.ndarray:
    """(n, k, 3) directions → (n, m, 3) orthonormal basis of their span (m = rank, the same for all)."""
    out = []
    for row in D:
        u, s, vt = np.linalg.svd(row)
        out.append(vt[: int((s > 1e-9 * s[0]).sum())])
    m = min(len(o) for o in out)
    return np.stack([o[:m] for o in out])


def projectors(study, mesh: Mesh) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Selection of a cylindrical support -> (nodes, P (n, 3, 3)) of the projection onto the constrained directions (for reactions)."""
    return {k: (idx, D.swapaxes(1, 2) @ D) for k, (idx, D) in support_directions(study, mesh).items()}


def boundary_dofs(study, mesh: Mesh) -> np.ndarray:
    """(N, 3) bool: DOF held by *BOUNDARY (clamping, prescribed displacement)."""
    from .study import Fixed, Displacement
    B = np.zeros((len(mesh.nodes), 3), bool)
    for c in study.constraints:
        idx = mesh.surface_nodes(c.selection)
        if isinstance(c, Fixed):
            B[idx] = True
        elif isinstance(c, Displacement):
            for k, v in enumerate((c.ux, c.uy, c.uz)):
                if v is not None:
                    B[idx, k] = True
    return B


def support_equations(study, mesh: Mesh, extra_bc: np.ndarray | None = None) -> list[tuple[int, list[tuple[int, float]]]]:
    """Constraint equations of the cylindrical supports: [(node 0-based, [(dof 0..2, coef.), ...])], the first term is dependent.

    At each node the directions of all cylindrical supports are merged and eliminated (RREF with pivots only in DOF
    that are not held by *BOUNDARY) → every dependent DOF is in a single equation and is not in *BOUNDARY.
    Equations fully covered by *BOUNDARY are left out (homogeneous, satisfied).
    Cached on the mesh per (constraints, mesh nodes / selections, extra_bc) – the result is a fresh list.
    """
    from .study import CylindricalSupport, Fixed, Displacement
    if not any(isinstance(c, CylindricalSupport) for c in study.constraints):
        return []
    fp = (_support_key(study, mesh, (CylindricalSupport, Fixed, Displacement)), _crc(extra_bc))
    val = _mesh_memo(mesh, "cyl_support_equations", fp, lambda: _support_equations(study, mesh, extra_bc))
    return [(n, list(terms)) for n, terms in val]


def _support_equations(study, mesh: Mesh, extra_bc: np.ndarray | None) -> list[tuple[int, list[tuple[int, float]]]]:
    dirs = support_directions(study, mesh)
    if not dirs:
        return []
    B = boundary_dofs(study, mesh)
    if extra_bc is not None:
        B |= extra_bc
    rows: dict[int, list[np.ndarray]] = {}
    for idx, D in dirs.values():
        for i, n in enumerate(idx):
            rows.setdefault(int(n), []).extend(D[i])
    out = []
    for n in sorted(rows):
        C = np.array(rows[n], float)
        free = [k for k in range(3) if not B[n, k]]
        used = []
        r = 0
        for col in free:
            if r >= len(C):
                break
            piv = r + int(np.argmax(np.abs(C[r:, col])))
            if abs(C[piv, col]) < 1e-9:
                continue
            C[[r, piv]] = C[[piv, r]]
            C[r] /= C[r, col]
            for j in range(len(C)):
                if j != r:
                    C[j] -= C[j, col] * C[r]
            used.append((r, col))
            r += 1
        for row, col in used:
            terms = [(col, 1.0)] + [(k, float(C[row, k])) for k in range(3)
                                    if k != col and abs(C[row, k]) > 1e-12]
            out.append((n, terms))
    return out


# ------------------------------------------------------------------ bearing load

_A, _B = 0.445948490915965, 0.091576213509771
_TRI_PTS = np.array([[_A, _A], [1 - 2 * _A, _A], [_A, 1 - 2 * _A], [_B, _B], [1 - 2 * _B, _B], [_B, 1 - 2 * _B]])
_TRI_W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3) / 2


def _tri_shape(xi: float, eta: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N, dN/dξ, dN/dη for tri6 (Gmsh order: corners, midpoints 01, 12, 20) or tri3."""
    l0 = 1 - xi - eta
    if n == 3:
        return np.array([l0, xi, eta]), np.array([-1.0, 1.0, 0.0]), np.array([-1.0, 0.0, 1.0])
    N = np.array([l0 * (2 * l0 - 1), xi * (2 * xi - 1), eta * (2 * eta - 1), 4 * l0 * xi, 4 * xi * eta, 4 * eta * l0])
    dxi = np.array([-(4 * l0 - 1), 4 * xi - 1, 0, 4 * (l0 - xi), 4 * eta, -4 * eta])
    deta = np.array([-(4 * l0 - 1), 0, 4 * eta - 1, -4 * xi, 4 * xi, 4 * (l0 - eta)])
    return N, dxi, deta


def bearing_forces(mesh: Mesh, selection: str, vector) -> np.ndarray:
    return bearing_load(mesh, selection, vector)[0]


def bearing_load(mesh: Mesh, selection: str, vector) -> tuple[np.ndarray, dict]:
    """Consistent nodal forces of the bearing load (N, 3) + info (axis, a, pressure at the Gauss points).

    Radial component F_r (perpendicular to the axis): normal pressure p = u·a on the part of the cylinder where u·F_r > 0
    (u = unit radial direction INTO the material: u of a hole points from the axis, u of a pin points to the axis) – cosine
    distribution. The vector a (in the plane perpendicular to the axis) is computed so that the resultant (with the same Gauss integration
    as the nodal forces) is exactly F_r: G·a = F_r, G = ∫ u uᵀ dA.
    The force at each Gauss point points perpendicular to the axis through the axis → the moment about the axis is exactly zero.
    Axial component: uniform shear over the whole face of the selection (as in Ansys), resultant exact.
    """
    from .ccx.inp import consistent_nodal_forces
    from .mesh import tri_areas
    F = np.asarray(vector, float)
    out = np.zeros((len(mesh.nodes), 3))
    tri = mesh.surfaces[selection]
    c0, d, _ = selection_axis(mesh, selection)
    Fa = (F @ d) * d
    Fr = F - Fa
    if np.linalg.norm(Fa) > 1e-12 * max(np.linalg.norm(F), 1e-300):
        t = Fa / tri_areas(mesh.nodes, tri).sum()
        for nd, fv in consistent_nodal_forces(mesh.nodes, tri, t).items():
            out[nd] += fv
    fr = float(np.linalg.norm(Fr))
    info = {"axis_point": c0, "axis_dir": d, "a": np.zeros(3), "x": np.zeros((0, 3)), "p": np.zeros(0),
            "dA": np.zeros(0)}
    if fr == 0.0:
        return out, info
    X = mesh.nodes[tri]                                   # (K, 6|3, 3)
    k = tri.shape[1]
    # orientation: outer normal of the body against the radial direction → hole (s = −1) / pin (s = +1)
    elems, fno = mesh.element_faces(selection)
    opp = mesh.nodes[mesh.tets[elems, np.array([3, 2, 0, 1])[fno - 1]]]
    xc = X[:, :3].mean(axis=1)
    er_c, _ = radial_frames(xc, c0, d)
    s = np.sign(np.einsum("kj,kj->k", er_c, xc - opp))   # + : the radial direction points out of the body (pin)
    s[s == 0] = 1.0
    gp = []
    for (xi, eta), w in zip(_TRI_PTS, _TRI_W):
        N, dxi, deta = _tri_shape(xi, eta, k)
        xg = np.einsum("i,kij->kj", N, X)
        J = np.linalg.norm(np.cross(np.einsum("i,kij->kj", dxi, X), np.einsum("i,kij->kj", deta, X)), axis=1)
        er, _ = radial_frames(xg, c0, d)
        u = -s[:, None] * er                              # into the material
        gp.append((N, w * J, u))
    fh = Fr / fr
    G = np.zeros((3, 3))
    for _, wj, u in gp:
        on = (u @ fh) > 0
        G += ((wj * on)[:, None] * u).T @ u
    e1, e2 = fh, np.cross(d, fh)
    G2 = np.array([[e1 @ G @ e1, e1 @ G @ e2], [e2 @ G @ e1, e2 @ G @ e2]])
    if abs(np.linalg.det(G2)) < 1e-12 * max(np.abs(G2).max(), 1e-300) ** 2:
        raise CylinderError(f"Selection '{selection}': the bearing load cannot be distributed – the face does not cover the side of the cylinder facing the force")
    a2 = np.linalg.solve(G2, [fr, 0.0])
    a = a2[0] * e1 + a2[1] * e2
    xs, ps, das = [], [], []
    f_nodal = np.zeros((len(tri), k, 3))
    for N, wj, u in gp:
        on = (u @ fh) > 0
        p = on * (u @ a)                                  # normal pressure at the Gauss points
        f = (wj * p)[:, None] * u                         # force at the Gauss point (K, 3)
        f_nodal += N[None, :, None] * f[:, None, :]
        xs.append(np.einsum("i,kij->kj", N, X))
        ps.append(p)
        das.append(wj)
    for i in range(k):
        np.add.at(out, tri[:, i], f_nodal[:, i])
    info.update(a=a, x=np.concatenate(xs), p=np.concatenate(ps), dA=np.concatenate(das))
    return out, info
