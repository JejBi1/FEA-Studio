"""Evaluation of the load–displacement curve of the nonlinear buckling analysis (no solver, pure numpy).

ccx 2.23 has no arc-length (Riks) method → load control: the increments shrink at a limit point until the minimum increment is
reached and the job stops. The ultimate factor is then the last converged λ; the tangent stiffness dλ/du at the end of the
curve relative to the initial one tells whether a limit point was really reached (≈ 0) or the job stopped for another reason.
"""
from __future__ import annotations

import numpy as np

LIMIT_STIFFNESS = 0.10          # k_t / k_0 below this at the end of a stopped curve = limit point


def tangent_ratio(lam: np.ndarray, u: np.ndarray) -> float | None:
    """k_t/k_0 = (dλ/du at the end) / (dλ/du at the start) over |u| (None = too few points / no displacement)."""
    lam, u = np.asarray(lam, float), np.abs(np.asarray(u, float))
    if len(lam) < 3:
        return None
    du0 = u[1] - u[0]
    du1 = u[-1] - u[-2]
    if not du0 > 0:
        # first increments without a measurable displacement change: take the first pair with du > 0
        nz = np.nonzero(np.diff(u) > 0)[0]
        if not len(nz):
            return None
        i = int(nz[0])
        du0 = u[i + 1] - u[i]
        k0 = (lam[i + 1] - lam[i]) / du0
    else:
        k0 = (lam[1] - lam[0]) / du0
    if not k0 > 0:
        return None
    if du1 <= 0:
        return 1.0 if lam[-1] >= lam[-2] else 0.0
    return float((lam[-1] - lam[-2]) / du1 / k0)


def first_crossing(lam: np.ndarray, vals: np.ndarray, limit: float) -> float | None:
    """λ at which vals(λ) first reaches the limit (linear interpolation), None = never."""
    lam, vals = np.asarray(lam, float), np.asarray(vals, float)
    idx = np.nonzero(vals >= limit)[0]
    if not len(idx):
        return None
    i = int(idx[0])
    if i == 0:
        return float(lam[0])
    v0, v1 = vals[i - 1], vals[i]
    t = (limit - v0) / (v1 - v0) if v1 != v0 else 1.0
    return float(lam[i - 1] + t * (lam[i] - lam[i - 1]))


def evaluate(lam, u, finished: bool, max_factor: float, lam_cr: float | None = None,
             mises=None, Re: float | None = None) -> dict:
    """Ultimate factor and status of the curve.

    status: limit_point (stopped, tangent ≈ 0 or descending) | stopped (stopped with a tangent still significant – numerical
    difficulty or a very sharp limit; check the curve) | max_factor (ramp completed to λ_max without a limit point – increase
    λ_max or judge by deflection / first yield) | no_increment.
    """
    lam = np.asarray(lam, float)
    u = np.asarray(u, float)
    out: dict = {"max_factor": float(max_factor)}
    if not len(lam):
        out.update(status="no_increment", ultimate_factor=None)
        return out
    i_max = int(np.argmax(lam))
    lam_u = float(lam[i_max])
    ratio = tangent_ratio(lam[:i_max + 1], u[:i_max + 1]) if i_max >= 2 else None
    descending = i_max < len(lam) - 1 and lam[-1] < lam_u
    if descending:
        status = "limit_point"
    elif finished and lam[-1] >= max_factor * (1 - 1e-6):
        status = "max_factor"
    elif ratio is not None and ratio <= LIMIT_STIFFNESS:
        status = "limit_point"
    else:
        status = "stopped"
    out.update(status=status, ultimate_factor=lam_u, ultimate_index=i_max, tangent_ratio_end=ratio,
               increments=int(len(lam)))
    if lam_cr:
        out["linear_factor"] = float(lam_cr)
        out["knockdown"] = lam_u / float(lam_cr)
    if mises is not None and Re:
        ly = first_crossing(lam, mises, float(Re))
        out["first_yield_factor"] = ly
        out["first_yield_Re_MPa"] = float(Re)
    return out


STATUS_TEXT = {
    "limit_point": "limit point reached (load-carrying capacity – tangent stiffness ≈ 0)",
    "stopped": "the solution stopped before the tangent stiffness vanished – check the curve (numerical difficulty or a sharp limit point)",
    "max_factor": "no limit point up to the maximum load factor – increase λ max or judge by deflection / first yield",
    "no_increment": "no converged increment",
}
