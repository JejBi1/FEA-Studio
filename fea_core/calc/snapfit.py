"""Snap-fit hook – a bending beam of rectangular cross-section according to Bayer (Snap-Fit Joints for Plastics,
Bayer MaterialScience, pp. 8–16), also tapered along the length (thickness h → h·r_h, width b → b·r_b linearly).

  ε = h₀·y / (2·l²·C_y),  C_y = ∫₀¹ (1 − u)² / f(u) du,  f(u) = (1 − (1 − r_b)u)·(1 − (1 − r_h)u)³
      (constant section C_y = 1/3 → ε = 1.5·h·y/l², Bayer y = 0.67·ε·l²/h; h → h/2: Bayer y = 1.09·ε·l²/h)
  P = (b·h²/6)·E_s·ε / l                         bending (deflecting) force at the end
  W = P·(μ + tan α)/(1 − μ·tan α)                assembly force (lead-in face α)
  W' = P·(μ + tan α')/(1 − μ·tan α')             release force (retaining face α'), not releasable for μ·tan α' ≥ 1

Correction for the rotation of the end (own, in the literature NOT VERIFIED): on deflection the end of the hook rotates by θ = P·l²/(E·I₀)·C_θ
(constant section θ = 1.5·y/l) → the lead-in face is steeper (α + θ), the retaining one more inclined (α' − θ).
"""
from __future__ import annotations

import math

import numpy as np

from .common import CalcError, positive, result

SOURCE = "Bayer MaterialScience: Snap-Fit Joints for Plastics – A Design Guide, pp. 8–16 (hook and force equations)"
_GL = np.polynomial.legendre.leggauss(64)


def _taper_integrals(r_h: float, r_b: float) -> tuple[float, float, float, float]:
    """(C_y, C_θ, max. ratio of the stress σ/σ_root, position of the maximum u = x/l) for a linear taper."""
    xg, wg = _GL
    u = 0.5 * (xg + 1.0)
    w = 0.5 * wg
    f = (1.0 - (1.0 - r_b) * u) * (1.0 - (1.0 - r_h) * u) ** 3
    cy = float(np.sum(w * (1.0 - u) ** 2 / f))
    ct = float(np.sum(w * (1.0 - u) / f))
    uu = np.linspace(0.0, 1.0, 2001)
    s = (1.0 - uu) / ((1.0 - (1.0 - r_b) * uu) * (1.0 - (1.0 - r_h) * uu) ** 2)
    k = int(np.argmax(s))
    return cy, ct, float(s[k]), float(uu[k])


def _wedge_factor(mu: float, ang: float) -> float:
    den = 1.0 - mu * math.tan(ang)
    return math.inf if den <= 0 else (mu + math.tan(ang)) / den


def snap_fit(l: float, h: float, b: float, E: float, y: float, alpha_deg: float = 30.0, mu: float = 0.3,
             alpha_release_deg: float | None = None, h_ratio: float = 1.0, b_ratio: float = 1.0,
             tip_rotation: bool = True, eps_allow_pct: float | None = None) -> dict:
    """Snap-fit hook of rectangular cross-section (Bayer).

    l length of the arm [mm], h thickness at the root [mm], b width at the root [mm], E secant modulus [MPa], y deflection
    (the interference of the hook) [mm], alpha_deg lead-in angle, mu coefficient of friction, alpha_release_deg retaining angle (None = without
    the calculation of the release), h_ratio / b_ratio = thickness / width at the end relative to the root (1 = constant section; Bayer type 2:
    h_ratio 0.5), tip_rotation = correction for the rotation of the end, eps_allow_pct = allowable strain [%] (Bayer tables
    by the material; None = no check).
    """
    l, h, b, E, y = (positive(n, v) for n, v in (("l", l), ("h", h), ("b", b), ("E", E), ("y", y)))
    mu = positive("mu", mu, zero=True)
    r_h = positive("h_ratio", h_ratio)
    r_b = positive("b_ratio", b_ratio)
    if r_h > 1 or r_b > 1:
        raise CalcError("h_ratio and b_ratio must be in (0; 1] – taper toward the tip")
    if not 0 <= alpha_deg < 90:
        raise CalcError("alpha_deg must be in [0; 90)°")
    cy, ct, smax, umax = _taper_integrals(r_h, r_b)
    I0 = b * h ** 3 / 12.0
    P = y * E * I0 / (l ** 3 * cy)
    eps_root = P * l * h / (2.0 * E * I0)
    eps = eps_root * smax
    theta = P * l ** 2 / (E * I0) * ct
    a = math.radians(alpha_deg)
    W = P * _wedge_factor(mu, a)
    W_rot = P * _wedge_factor(mu, a + theta)
    warn = []
    vals = {"eps": eps, "eps_pct": 100.0 * eps, "eps_root_pct": 100.0 * eps_root, "eps_max_at_x_mm": umax * l,
            "P_N": P, "W_N": W, "theta_tip_deg": math.degrees(theta), "W_tip_rotation_N": W_rot,
            "W_design_N": W_rot if tip_rotation else W, "C_y": cy, "I_root_mm4": I0}
    if alpha_deg + math.degrees(theta) >= 90 or not math.isfinite(W_rot):
        warn.append("Lead-in face becomes self-locking after tip rotation – the hook cannot be pushed on")
    if alpha_release_deg is not None:
        if not 0 <= alpha_release_deg <= 90:
            raise CalcError("alpha_release_deg must be in [0; 90]°")
        ar = math.radians(alpha_release_deg)
        ar_eff = max(ar - theta, 0.0) if tip_rotation else ar
        Wr = P * _wedge_factor(mu, ar) if alpha_release_deg < 90 else math.inf
        Wr_rot = P * _wedge_factor(mu, ar_eff)
        vals.update({"W_release_N": Wr, "W_release_tip_rotation_N": Wr_rot,
                     "permanent": bool(alpha_release_deg >= 90 or mu * math.tan(ar) >= 1.0),
                     "permanent_tip_rotation": bool(mu * math.tan(ar_eff) >= 1.0),
                     "alpha_release_eff_deg": math.degrees(ar_eff)})
        if vals["permanent"] and not vals["permanent_tip_rotation"] and tip_rotation:
            warn.append("Non-separable per Bayer (μ·tan α' ≥ 1), but the retention face yields after tip rotation – verify by FEA (NLGEOM) or testing")
    if eps_allow_pct is not None:
        ea = positive("eps_allow_pct", eps_allow_pct)
        vals["eps_allow_pct"] = ea
        vals["utilization"] = 100.0 * eps / ea
        if eps * 100.0 > ea:
            warn.append(f"Strain {100 * eps:.3g} % exceeds the allowable {ea:.3g} %")
    if umax > 0:
        warn.append(f"Maximum stress is not at the root but at a distance of {umax * l:.3g} mm (taper too large)")
    if y / l > 0.2:
        warn.append("Deflection > 0.2·l – linear beam theory is only approximate (large rotation)")
    return result(vals, "ε = h·y/(2l²C_y) (constant section 1.5·h·y/l²); P = (bh²/6)·E·ε/l; W = P(μ + tan α)/(1 − μ tan α); W' with α'; non-separable μ·tan α' ≥ 1; tip rotation θ = P·l²·C_θ/(E·I) (α + θ when mating, α' − θ when separating – UNVERIFIED)",
                  SOURCE, warn)


KINDS = {"snap_fit": (snap_fit, "snap-fit hook of rectangular cross-section (Bayer), also tapered")}
