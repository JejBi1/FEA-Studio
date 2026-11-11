"""Results along a line and stress linearisation (membrane + bending) on a line SCL across a wall.

`line_values`: field values at n points of the line p0 → p1 (interpolation with the shape functions of the element as the probes; points outside
the body NaN). `linearize`: ASME BPVC VIII-2, Annex 5-A (5-A.4.1.2, stress integration method): t = length of the SCL,
x from p0,
  membrane  σ_m,ij = 1/t ∫ σ_ij dx              (all components)
  bending   σ_b,ij = 6/t² ∫ σ_ij (t/2 − x) dx   (only the normal components perpendicular to the SCL – "circumferential" and "longitudinal";
            the component in the direction of the SCL and the shears have no bending)
  peak      σ_F = σ − (σ_m + σ_b) at the end points (peak stress),
von Mises and Tresca equivalents for P_m, P_m + P_b (both ends) and the total stress at the ends. Simpson integration
(odd number of points). The stresses are nodal (averaged) – a warning for a coarse mesh through the thickness (< 3 elements).
"""
from __future__ import annotations

import numpy as np

from .results import tensor, von_mises


def _tresca(T: np.ndarray) -> float:
    w = np.linalg.eigvalsh(T)
    return float(w.max() - w.min())


def _voigt(T: np.ndarray) -> np.ndarray:
    return np.array([T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[1, 2], T[2, 0]])


def inside(res, p, tol: float = 1e-6) -> bool:
    e, xi = res.locate(p)
    return bool(min(xi.min(), 1.0 - xi.sum()) >= -tol)


def line_values(res, p0, p1, field: str = "mises", step: int | str = 0, n: int = 51) -> dict:
    """Field values along the line: {"s" [mm], "points", "values" (n,) or (n, k), "field"}.
    field: mises | s1 | s2 | s3 | sx … szx | ux uy uz | umag | T (temperature of the thermal state step) | another StepResult.fields.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    s = np.linspace(0.0, 1.0, n)
    pts = p0[None] + s[:, None] * (p1 - p0)[None]
    comp = {"sx": 0, "sy": 1, "sz": 2, "sxy": 3, "syz": 4, "szx": 5}
    out = []
    for p in pts:
        if not inside(res, p, 0.02):           # curved surface: a point slightly beyond the chord of a linear element
            out.append(np.nan)
            continue
        if field == "T":
            out.append(float(res.probe(p, "T", step)[0]))
        elif field in ("ux", "uy", "uz", "umag", "U"):
            u = res.probe(p, "U", step)
            out.append(float(np.linalg.norm(u)) if field == "umag" else u if field == "U" else float(u["xyz".index(field[1])]))
        elif field in comp or field in ("mises", "s1", "s2", "s3", "S"):
            S = res.probe(p, "S", step)
            if field == "S":
                out.append(S)
            elif field == "mises":
                out.append(float(von_mises(S)))
            elif field in comp:
                out.append(float(S[comp[field]]))
            else:
                out.append(float(np.linalg.eigvalsh(tensor(S))[::-1][int(field[1]) - 1]))
        else:
            out.append(float(res.probe(p, field, step)))
    vals = np.array([np.full(6 if field == "S" else 3, np.nan) if (isinstance(v, float) and np.isnan(v)
                     and field in ("S", "U")) else v for v in out], dtype=float)
    return {"s": (s * np.linalg.norm(p1 - p0)).tolist(), "points": pts.tolist(), "values": vals, "field": field}


def local_frame(p0, p1, hint=None) -> np.ndarray:
    """Rows: axis a (along the SCL), b, c (perpendicular). hint = direction for b (e.g. circumferential); otherwise any direction perpendicular."""
    a = np.asarray(p1, float) - np.asarray(p0, float)
    a /= np.linalg.norm(a)
    h = np.asarray(hint, float) if hint is not None else (np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0]))
    b = h - (h @ a) * a
    if np.linalg.norm(b) < 1e-9:
        b = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
    b /= np.linalg.norm(b)
    return np.array([a, b, np.cross(a, b)])


def linearize(res, p0, p1, step: int | str = 0, n: int = 41, hint=None) -> dict:
    """Stress linearisation on the SCL p0 → p1 (see the module description). Tensors in global axes (Voigt xx yy zz xy yz zx)
    and in the local axes of the SCL (a = along the SCL, b, c perpendicular; hint = direction of axis b).
    """
    if n % 2 == 0:
        n += 1
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = float(np.linalg.norm(p1 - p0))
    lv = line_values(res, p0, p1, "S", step, n)
    S = lv["values"]
    if np.isnan(S).any():
        raise ValueError("The linearization line runs outside the body – both ends and the whole line must lie within the material")
    R = local_frame(p0, p1, hint)
    T = np.array([R @ tensor(s) @ R.T for s in S])                   # local axes (a, b, c)
    x = np.linspace(0.0, t, n)
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    w *= (t / (n - 1)) / 3.0                                         # Simpson
    Tm = np.einsum("i,ijk->jk", w, T) / t
    Tb = np.einsum("i,ijk->jk", w * (t / 2 - x), T) * 6.0 / t ** 2
    mask = np.zeros((3, 3))
    mask[1, 1] = mask[2, 2] = 1.0                                    # bending of only the normal components perpendicular to the SCL
    Tb *= mask
    ends = {"start": (Tm + Tb, T[0]), "end": (Tm - Tb, T[-1])}
    glob = lambda M: _voigt(R.T @ M @ R)

    def eq(M):
        return {"mises": float(von_mises(_voigt(M))), "tresca": _tresca(M)}
    out = {"length_mm": t, "points": [p0.tolist(), p1.tolist()], "axes": R.tolist(),
           "membrane": {"local": _voigt(Tm).tolist(), "global": glob(Tm).tolist(), **eq(Tm)},
           "bending": {"local": _voigt(Tb).tolist(), "global": glob(Tb).tolist()},
           "ends": {}}
    for k, (mb, tot) in ends.items():
        out["ends"][k] = {"membrane_bending": {**eq(mb), "local": _voigt(mb).tolist()},
                          "total": {**eq(tot), "local": _voigt(tot).tolist()},
                          "peak": {**eq(tot - mb), "local": _voigt(tot - mb).tolist()}}
    out["Pm_mises"] = out["membrane"]["mises"]
    out["PmPb_mises"] = max(out["ends"][k]["membrane_bending"]["mises"] for k in ends)
    out["total_mises"] = max(out["ends"][k]["total"]["mises"] for k in ends)
    # coarseness of the mesh through the thickness
    h = mean_edge(res.mesh)
    if t / h < 3:
        out["warning"] = (f"Across the {t:.3g} mm thickness there are only ~{t / h:.1f} elements (mean edge {h:.3g} mm) – refine the mesh for linearization (≥ 3–4 elements)")
    return out


def mean_edge(mesh) -> float:
    """Mean length of the element edge (sample of the first 5000 elements) [mm]."""
    t = mesh.tets[: min(len(mesh.tets), 5000), :4]
    X = mesh.nodes
    return float(np.mean([np.linalg.norm(X[t[:, i]] - X[t[:, j]], axis=1)
                          for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]))
