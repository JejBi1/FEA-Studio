"""Local coordinate systems of loads (Study.coordinate_systems).

Cartesian: rows of the matrix R = unit axes x, y, z of the local system in global coordinates; components v → global
Rᵀ·v (same direction everywhere). Cylindrical: components (r, θ, z) at point x → e_r = radial from the axis, e_θ = e_z × e_r, e_z = axis;
the direction depends on the point, so it is converted point by point (Gauss points of the face / edge, nodes of a point load).
θ = 0 lies in the direction of x_axis (description only; it has no effect on the conversion of the components).
"""
from __future__ import annotations

import numpy as np

from .study import Study, CoordinateSystem


class CsysError(ValueError):
    pass


def check(cs: CoordinateSystem) -> list[str]:
    """Errors in the definition of the system (empty list = all right)."""
    out = []
    if cs.kind not in ("cartesian", "cylindrical"):
        out.append(f"Coordinate system '{cs.name}': unknown kind '{cs.kind}' (cartesian | cylindrical)")
    for key in ("origin", "z_axis", "x_axis"):
        v = getattr(cs, key)
        try:
            ok = len(v) == 3 and all(np.isfinite(float(c)) for c in v)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            out.append(f"Coordinate system '{cs.name}': {key} must have 3 finite coordinates")
    if not out:
        z = np.asarray(cs.z_axis, float)
        x = np.asarray(cs.x_axis, float)
        if np.linalg.norm(z) == 0:
            out.append(f"Coordinate system '{cs.name}': the z axis is a zero vector")
        elif np.linalg.norm(np.cross(z, x)) <= 1e-9 * np.linalg.norm(z) * max(np.linalg.norm(x), 1e-300):
            out.append(f"Coordinate system '{cs.name}': the x axis is parallel to the z axis (or zero)")
    return out


def matrix(cs: CoordinateSystem) -> np.ndarray:
    """R (3×3): rows = axes x, y, z of the local system in global coordinates (right-handed, orthonormal)."""
    errs = check(cs)
    if errs:
        raise CsysError(errs[0])
    z = np.asarray(cs.z_axis, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(cs.x_axis, float)
    x = x - (x @ z) * z
    x = x / np.linalg.norm(x)
    return np.array([x, np.cross(z, x), z])


def get(study: Study, name: str | None) -> CoordinateSystem | None:
    if name is None or name == "":
        return None
    cs = study.coordinate_systems.get(name)
    if cs is None:
        raise CsysError(f"Coordinate system '{name}' does not exist")
    return cs


def cylindrical_axes(cs: CoordinateSystem, X: np.ndarray, what: str = "") -> np.ndarray:
    """Basis (e_r, e_θ, e_z) at points X (P, 3) → (P, 3, 3), rows = axes. A point on the axis → CsysError."""
    R = matrix(cs)
    ez = R[2]
    d = np.asarray(X, float).reshape(-1, 3) - np.asarray(cs.origin, float)
    rv = d - np.outer(d @ ez, ez)
    r = np.linalg.norm(rv, axis=1)
    size = max(float(np.abs(d).max()) if len(d) else 0.0, 1e-9)
    if len(r) and r.min() <= 1e-9 * size:
        raise CsysError(f"{what or 'Load'}: the point lies on the axis of cylindrical system '{cs.name}' – the radial and tangential directions are undefined there")
    er = rv / r[:, None]
    et = np.cross(ez, er)
    return np.stack([er, et, np.broadcast_to(ez, er.shape)], axis=1)


def vector_global(study: Study, name: str | None, v, what: str = "") -> np.ndarray:
    """Vector of components v → global (Cartesian system only; the cylindrical one depends on the point → error)."""
    v = np.asarray(v, float)
    cs = get(study, name)
    if cs is None:
        return v
    if cs.kind != "cartesian":
        raise CsysError(f"{what or 'Load'}: cylindrical system '{cs.name}' cannot be used here (the direction would depend on the point) – use a Cartesian system")
    return matrix(cs).T @ v


def field_global(study: Study, name: str | None, v, X: np.ndarray, what: str = "") -> np.ndarray:
    """Components v (3,) at points X (P, 3) → global vectors (P, 3); Cartesian / global system = the same everywhere."""
    v = np.asarray(v, float)
    X = np.asarray(X, float).reshape(-1, 3)
    cs = get(study, name)
    if cs is None:
        return np.broadcast_to(v, X.shape).copy()
    if cs.kind == "cartesian":
        return np.broadcast_to(matrix(cs).T @ v, X.shape).copy()
    if not np.any(v[:2]):                 # only the axial component – independent of the point (also works on the axis)
        return np.broadcast_to(matrix(cs)[2] * v[2], X.shape).copy()
    B = cylindrical_axes(cs, X, what)
    return np.einsum("pij,i->pj", B, v)


def describe(cs: CoordinateSystem) -> str:
    def f(v):
        return "(" + "; ".join(f"{float(c):.4g}" for c in v) + ")"
    kind = "cylindrical (r, θ, z)" if cs.kind == "cylindrical" else "Cartesian"
    return f"{cs.name}: {kind}, origin {f(cs.origin)}, z axis {f(cs.z_axis)}, x axis {f(cs.x_axis)}"


def labels(study: Study, name: str | None) -> tuple[str, str, str]:
    """Labels of the components for the GUI / report."""
    cs = study.coordinate_systems.get(name) if name else None
    if cs is not None and cs.kind == "cylindrical":
        return "r", "θ", "z"
    return "x", "y", "z"


def transform_displacements(study: Study, name: str | None, U: np.ndarray,
                            X: np.ndarray, what: str = "") -> np.ndarray:
    """Transforms the displacement field U (N, 3) at points X (N, 3) from global to local system.
    Cartesian: U_loc = U @ R.T  (R rows = axes x, y, z of the local system)
    Cylindrical: U_loc[:, i] = U_glob . e_i(X), components (ur, uθ, uz).
    """
    cs = get(study, name)
    if cs is None:
        return U.copy()
    if cs.kind == "cartesian":
        R = matrix(cs)
        return U @ R.T
    B = cylindrical_axes(cs, X, what)  # (N, 3, 3), rows are e_r, e_θ, e_z
    return np.einsum("nij,nj->ni", B, U)


def transform_stress(study: Study, name: str | None, S: np.ndarray,
                     X: np.ndarray, what: str = "") -> np.ndarray:
    """Transforms the stress tensor S (N, 6 in ccx order: xx, yy, zz, xy, yz, zx)
    at points X (N, 3) from global to local system:
    T_loc = R @ T_glob @ R.T.
    Returns the field (N, 6) in the same order of components (11, 22, 33, 12, 23, 31):
    for a cylindrical system: (σr, σθ, σz, τrθ, τθz, τzr).
    """
    cs = get(study, name)
    if cs is None:
        return S.copy()
    from .results import tensor
    T = tensor(S)  # (N, 3, 3)
    if cs.kind == "cartesian":
        R = matrix(cs)  # (3, 3)
        T_loc = np.einsum("ij,njk,lk->nil", R, T, R)
    else:
        B = cylindrical_axes(cs, X, what)  # (N, 3, 3), rows e_r, e_θ, e_z
        T_loc = np.einsum("nij,njk,nlk->nil", B, T, B)

    out = np.empty((len(S), 6), dtype=float)
    out[:, 0] = T_loc[:, 0, 0]  # 11
    out[:, 1] = T_loc[:, 1, 1]  # 22
    out[:, 2] = T_loc[:, 2, 2]  # 33
    out[:, 3] = T_loc[:, 0, 1]  # 12
    out[:, 4] = T_loc[:, 1, 2]  # 23
    out[:, 5] = T_loc[:, 2, 0]  # 31
    return out


def stress_labels(study: Study, name: str | None) -> list[str]:
    cs = study.coordinate_systems.get(name) if name else None
    if cs is not None and cs.kind == "cylindrical":
        return ["σr", "σθ", "σz", "τrθ", "τθz", "τzr"]
    return ["σx", "σy", "σz", "τxy", "τyz", "τzx"]


def displacement_labels(study: Study, name: str | None) -> list[str]:
    cs = study.coordinate_systems.get(name) if name else None
    if cs is not None and cs.kind == "cylindrical":
        return ["ur", "uθ", "uz"]
    return ["ux", "uy", "uz"]
