"""S-N curve (Basquin with a knee), Marin factors, estimate from Rm.

S-N in the stress amplitudes:  N = ND · (SD / Sa)^k        for Sa ≥ SD
                           N = ND · (SD / Sa)^k2       for Sa < SD (k2 = 0 → fatigue limit, N = ∞)
Haibach: k2 = 2k − 1.  Sources of the estimates: Shigley, Mechanical Engineering Design (10th ed.),
ch. 6 (Se' = 0.5 Rm, f·Rm at 10³ cycles, Marin factors).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

# Marin ka = a · Rm^b (Rm v MPa), Shigley tab. 6-2
SURFACE = {
    "lesteny": (1.0, 0.0),            # polished specimen
    "brouseny": (1.58, -0.085),
    "obrobeny": (4.51, -0.265),       # machined / cold-drawn
    "valcovany": (57.7, -0.718),      # hot-rolled
    "kovany": (272.0, -0.995),        # forging
}
SURFACE_LABELS = {"lesteny": "polished", "brouseny": "ground", "obrobeny": "machined / cold-drawn",
                  "valcovany": "hot-rolled", "kovany": "forged"}
# spolehlivost [%] → ke (Shigley tab. 6-5)
RELIABILITY = {50.0: 1.0, 90.0: 0.897, 95.0: 0.868, 99.0: 0.814, 99.9: 0.753, 99.99: 0.702}


@dataclass
class SNCurve:
    SD: float                 # amplitude of the fatigue limit / of the knee point [MPa]
    ND: float = 1e6           # number of cycles at the knee point
    k: float = 10.0           # sklon nad zlomem
    k2: float = 0.0           # slope below the knee; 0 = fatigue limit (without damage)
    source: str = ""

    def N(self, Sa: float) -> float:
        if Sa <= 0:
            return math.inf
        if Sa >= self.SD:
            return self.ND * (self.SD / Sa) ** self.k
        if self.k2 <= 0:
            return math.inf
        return self.ND * (self.SD / Sa) ** self.k2

    def to_dict(self) -> dict:
        return asdict(self)


def material_class(E: float) -> str:
    if E >= 150000:
        return "ocel"
    if E >= 40000:
        return "hlinik"
    return "plast"


def estimate_sn(E: float, Rm: float | None) -> SNCurve:
    """Estimate of the S-N of the smooth specimen from Rm (without the Marin factors)."""
    cls = material_class(E)
    if not Rm:
        raise ValueError("The material's Rm is required to estimate the S-N curve")
    if cls == "ocel":
        se = 0.5 * Rm if Rm <= 1400 else 700.0
        f = 0.9
        k = 3.0 / math.log10(f * Rm / se)          # line between 10³ and 10⁶ cycles
        return SNCurve(se, 1e6, k, 0.0, "estimate from Rm: Se' = 0.5 Rm at 10⁶, 0.9 Rm at 10³ (Shigley)")
    if cls == "hlinik":
        se = min(0.4 * Rm, 130.0)                    # at 5·10⁸ cycles, without a fatigue limit
        k = math.log10(5e8 / 1e3) / math.log10(0.9 * Rm / se)
        return SNCurve(se, 5e8, k, 2 * k - 1, "estimate from Rm: 0.4 Rm at 5·10⁸, Haibach below the knee (Shigley)")
    return SNCurve(0.25 * Rm, 1e6, 10.0, 2 * 10.0 - 1,
                   "approximate estimate for plastic – CALIBRATION FROM OWN TESTS REQUIRED")


def marin(Rm: float | None, surface: str = "obrobeny", d_mm: float | None = None,
          reliability: float = 50.0, k_load: float = 1.0, k_temp: float = 1.0, k_misc: float = 1.0) -> dict:
    """Marin factors, returns {'ka', 'kb', 'kc', 'kd', 'ke', 'kf', 'k'}."""
    a, b = SURFACE.get(surface, (1.0, 0.0))
    ka = min(1.0, a * Rm ** b) if Rm and b != 0 else 1.0
    if d_mm is None or d_mm <= 2.79:
        kb = 1.0
    elif d_mm <= 51:
        kb = 1.24 * d_mm ** -0.107
    else:
        kb = 1.51 * min(d_mm, 254) ** -0.157
    ke = RELIABILITY.get(float(reliability))
    if ke is None:                                   # interpolation in the normal distribution (8 % scatter)
        from statistics import NormalDist
        z = NormalDist().inv_cdf(reliability / 100.0)
        ke = max(0.0, 1 - 0.08 * z)
    out = {"ka": ka, "kb": kb, "kc": k_load, "kd": k_temp, "ke": ke, "kf": k_misc}
    out["k"] = ka * kb * k_load * k_temp * ke * k_misc
    return out


def apply_marin(sn: SNCurve, k_total: float, Rm: float | None, estimated: bool) -> SNCurve:
    """Reduces the fatigue limit by the factor. For the estimate from Rm the point 0.9 Rm at 10³ stays (Shigley)."""
    SD = sn.SD * k_total
    k = sn.k
    if estimated and Rm and sn.ND > 1e3 and 0.9 * Rm > SD:
        k = math.log10(sn.ND / 1e3) / math.log10(0.9 * Rm / SD)
    k2 = sn.k2 if sn.k2 <= 0 else (2 * k - 1 if estimated else sn.k2)
    return SNCurve(SD, sn.ND, k, k2, sn.source + f"; Marin k = {k_total:.3f}")


def mean_corrected(Sa: float, Sm: float, Rm: float, method: str) -> float:
    """Equivalent amplitude at zero mean stress (a compressive mean stress gives no benefit)."""
    if method == "none" or Sm <= 0 or not Rm:
        return Sa
    r = Sm / Rm
    if r >= 1:
        return math.inf
    if method == "goodman":
        return Sa / (1 - r)
    if method == "gerber":
        return Sa / (1 - r * r)
    raise ValueError(method)
