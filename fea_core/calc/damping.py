"""Damping from measurement and conversions (specification F8, section 5a.4): half-power bandwidth −3 dB, logarithmic decrement, Rayleigh
damping from two frequencies, conversions ζ ↔ Q ↔ η, indicative values of ζ with sources, a populated PCB Q = √f_n.

Caution: ccx `*DAMPING, STRUCTURAL` takes the loss factor η = 2ζ (not ζ) – see specification 5a.4.
"""
from __future__ import annotations

import math

from .common import CalcError, positive, result

# Indicative damping ratio ζ [%] (specification 5a.4) – (construction, range [%], source / state)
TABLE = [
    ("Steel building (bolted)", (0.8, 0.8), "EN 1991-1-4 Table F.2: δ_s = 0.05 → ζ ≈ 0.8 % (via SBE example)"),
    ("Welded steel", (0.3, 0.3), "EN 1991-1-4 Table F.2: δ_s ≈ 0.02 → ζ ≈ 0.3 %; HS bolts 0.03, ordinary 0.05 – UNVERIFIED"),
    ("Aluminum – material, monolithic", (0.1, 0.3), "Harris – UNVERIFIED"),
    ("Aluminum – assembly", (1.0, 3.0), "Harris – UNVERIFIED"),
    ("PLA FDM (small beams)", (1.0, 2.0), "J. Vib. Control 2025 (Cranfield), max 1.63 % (triangular infill 60°)"),
    ("Populated PCB", (2.5, 5.0), "Steinberg 2000: Q ≈ √f_n → 5 % @ 100 Hz, 2.5 % @ 400 Hz – UNVERIFIED against original"),
    ("Fixtures (bolts, guides, pneumatics)", (2.0, 5.0), "estimate UNVERIFIED; ±50 %, impact hammer measurement recommended"),
    ("SRS / Miles convention", (5.0, 5.0), "standards convention: Q = 10"),
]


def convert(zeta: float | None = None, Q: float | None = None, eta: float | None = None,
            delta: float | None = None) -> dict:
    """Conversion of one given quantity: ζ (damping ratio, not in %), Q = 1/(2ζ), η = 2ζ (loss factor,
    ccx STRUCTURAL), δ logarithmic decrement = 2πζ/√(1 − ζ²).
    """
    given = [k for k, v in (("zeta", zeta), ("Q", Q), ("eta", eta), ("delta", delta)) if v is not None]
    if len(given) != 1:
        raise CalcError("Enter exactly one quantity: zeta | Q | eta | delta")
    if zeta is not None:
        z = positive("zeta", zeta)
    elif Q is not None:
        z = 1.0 / (2.0 * positive("Q", Q))
    elif eta is not None:
        z = positive("eta", eta) / 2.0
    else:
        d = positive("delta", delta)
        z = d / math.sqrt(4.0 * math.pi ** 2 + d * d)
    if z >= 1:
        raise CalcError("ζ ≥ 1 – overdamped system (does not oscillate)")
    return result({"zeta": z, "zeta_pct": 100.0 * z, "Q": 1.0 / (2.0 * z), "eta": 2.0 * z,
                   "delta": 2.0 * math.pi * z / math.sqrt(1.0 - z * z)},
                  "Q = 1/(2ζ); η = 2ζ (ccx *DAMPING, STRUCTURAL = η); δ = 2πζ/√(1 − ζ²)",
                  "Chopra: Dynamics of Structures, ch. 2–3; ccx manual (*DAMPING)")


def half_power(f_n: float, f1: float, f2: float) -> dict:
    """Half-power bandwidth: frequencies f1 < f_n < f2 where the amplitude falls to 1/√2 of the peak → ζ ≈ (f2 − f1)/(2f_n)."""
    f, f1, f2 = positive("f_n", f_n), positive("f1", f1), positive("f2", f2)
    if not f1 < f2:
        raise CalcError("f1 < f2 is required")
    z = (f2 - f1) / (2.0 * f)
    warn = [] if f1 < f < f2 else ["f_n does not lie between f1 and f2 – check the readings"]
    if z > 0.1:
        warn.append("ζ > 10 % – the bandwidth method is only approximate (valid for light damping)")
    return result({"zeta": z, "zeta_pct": 100.0 * z, "Q": 1.0 / (2.0 * z), "eta": 2.0 * z},
                  "ζ ≈ Δf₋₃dB/(2f_n)", "half-power bandwidth method, Chopra ch. 3", warn)


def log_decrement(x0: float, xn: float, n: int = 1) -> dict:
    """Logarithmic decrement from the free decay: amplitude x0 and xn n periods later."""
    x0, xn = positive("x0", x0), positive("xn", xn)
    n = int(n)
    if n < 1:
        raise CalcError("n must be ≥ 1")
    if xn >= x0:
        raise CalcError("Amplitude must decrease (xn < x0)")
    d = math.log(x0 / xn) / n
    z = d / math.sqrt(4.0 * math.pi ** 2 + d * d)
    return result({"delta": d, "zeta": z, "zeta_pct": 100.0 * z, "Q": 1.0 / (2.0 * z), "eta": 2.0 * z},
                  "δ = (1/n)·ln(x₀/x_n); ζ = δ/√(4π² + δ²)", "Chopra ch. 2.2 (free damped vibration)")


def rayleigh(f1: float, f2: float, zeta1: float, zeta2: float | None = None, f_check: list | None = None) -> dict:
    """Rayleigh damping C = αM + βK from the damping ratios ζ₁, ζ₂ (not in %) at the frequencies f1, f2 [Hz];
    ζ(f) = α/(2ω) + βω/2. f_check = frequencies at which ζ is printed.
    """
    f1, f2 = positive("f1", f1), positive("f2", f2)
    if f1 == f2:
        raise CalcError("f1 and f2 must be different")
    z1 = positive("zeta1", zeta1, zero=True)
    z2 = z1 if zeta2 is None else positive("zeta2", zeta2, zero=True)
    w1, w2 = 2.0 * math.pi * f1, 2.0 * math.pi * f2
    alpha = 2.0 * w1 * w2 * (z1 * w2 - z2 * w1) / (w2 * w2 - w1 * w1)
    beta = 2.0 * (z2 * w2 - z1 * w1) / (w2 * w2 - w1 * w1)
    vals = {"alpha_1_s": alpha, "beta_s": beta}
    fs = list(f_check) if f_check else [f1 / 2.0, math.sqrt(f1 * f2), 2.0 * f2]
    vals["zeta_at"] = {f"{float(f):g} Hz": alpha / (4.0 * math.pi * float(f)) + beta * math.pi * float(f) for f in fs}
    warn = []
    if alpha < 0 or beta < 0:
        warn.append("Negative coefficient α or β – damping at some frequencies will be negative")
    return result(vals, "α = 2ω₁ω₂(ζ₁ω₂ − ζ₂ω₁)/(ω₂² − ω₁²), β = 2(ζ₂ω₂ − ζ₁ω₁)/(ω₂² − ω₁²) "
                        "(ζ₁ = ζ₂: α = 2ζω₁ω₂/(ω₁+ω₂), β = 2ζ/(ω₁+ω₂)); ζ(ω) = α/(2ω) + βω/2",
                  "Chopra: Dynamics of Structures, ch. 11; ccx manual (*DAMPING ALPHA, BETA)", warn)


def pcb(f_n: float) -> dict:
    """Populated PCB (Steinberg): Q ≈ √f_n [Hz] → ζ = 1/(2√f_n). NOT verified against the original."""
    f = positive("f_n", f_n)
    Q = math.sqrt(f)
    return result({"Q": Q, "zeta": 1.0 / (2.0 * Q), "zeta_pct": 50.0 / Q, "eta": 1.0 / Q},
                  "Q ≈ √f_n; ζ = 1/(2√f_n)", "Steinberg: Vibration Analysis for Electronic Equipment (2000) – UNVERIFIED",
                  ["Approximate relation – PCB damping varies with mounting and components; measurement recommended"])


def table() -> dict:
    """Indicative values of ζ with sources."""
    return result({"rows": [{"structure": n, "zeta_pct_min": lo, "zeta_pct_max": hi, "source": s}
                            for n, (lo, hi), s in TABLE]},
                  "approximate damping ratio ζ [%]", "F8 specification, 5a.4 (sources in rows)",
                  ["Damping is an uncertain input (sensitivity ±50 %) – where it governs, measure it (impact hammer, free decay)"])


KINDS = {"convert": (convert, "conversion ζ ↔ Q ↔ η ↔ δ"),
         "half_power": (half_power, "−3 dB bandwidth"),
         "log_decrement": (log_decrement, "logarithmic decrement from free decay"),
         "rayleigh": (rayleigh, "Rayleigh α, β from two frequencies"),
         "pcb": (pcb, "populated PCB Q = √f_n"),
         "table": (table, "approximate ζ values with sources")}
