"""A fall as an equivalent pulse of acceleration (specification F8, part D4): impact velocity v = √(2gh), duration of the half-sine shock
τ ≈ π√(m/k) (half a period of the elastic impact on an elastic support of stiffness k), peak A = π(1 + e)·v/(2τ)
(change of velocity (1 + e)·v = the area of the half-sine 2Aτ/π). Units mm, N, t, s.
"""
from __future__ import annotations

import math

from .common import G, CalcError, positive, result


def drop(h: float, m: float | None = None, k: float | None = None, tau: float | None = None, e: float = 0.0,
         g: float = G) -> dict:
    """A fall from the height h [mm]: the duration of the pulse τ [s] directly, or from the mass m [t] and the stiffness of the impact k [N/mm]; e coefficient
    of restitution (0 = no rebound, 1 = elastic rebound). Returns v [mm/s], τ, the peak A [mm/s² and g], Δv.
    """
    h = positive("h", h)
    e = positive("e", e, zero=True)
    if e > 1:
        raise CalcError("Coefficient of restitution e must be in [0; 1]")
    if tau is None:
        if m is None or k is None:
            raise CalcError("Enter the pulse duration tau, or the mass m and the impact stiffness k")
        tau = math.pi * math.sqrt(positive("m", m) / positive("k", k))
    tau = positive("tau", tau)
    v = math.sqrt(2.0 * g * h)
    A = math.pi * (1.0 + e) * v / (2.0 * tau)
    return result({"v_mm_s": v, "v_m_s": v / 1e3, "tau_s": tau, "tau_ms": 1e3 * tau, "A_mm_s2": A, "A_g": A / g,
                   "dv_mm_s": (1.0 + e) * v},
                  "v = √(2gh); τ ≈ π√(m/k); A = π(1 + e)·v/(2τ) (half-sine with the same velocity change)",
                  "equivalent half-sine pulse (Harris: Shock and Vibration Handbook, ch. 8 and 31 – approximate)",
                  ["Actual pulse shape depends on the floor and packaging – verify τ and A by measurement or dynamic analysis"])


KINDS = {"drop": (drop, "drop as an equivalent half-sine pulse")}
