"""Spectral fatigue (frequency domain) per node – random vibration (D3) and sinusoidal sweep (D1).

CONVENTIONS:
  - Input = spectral moments of the ONE-SIDED PSD of the stress G(f) in Hz: m_i = ∫₀^∞ f^i·G(f) df [MPa²·Hz^i],
    array m (n, 4) with the columns m0, m1, m2, m4 (from `dynamics.psd.equivalent_moments` or `quad_forms`).
    ν₀⁺ = √(m2/m0) [1/s] = mean rate of the upward zero crossings, ν_p = √(m4/m2) = rate of the peaks,
    α₁ = m1/√(m0·m2), α₂ = γ = m2/√(m0·m4) (irregularity factor; 1 = narrow-band process).
    (FLife computes the moments in (rad/s)^i – dimensionless parameters and the damage are the same.)
  - S-N curve `SNCurve` in AMPLITUDES s = range/2: N = ND·(SD/s)^k for s ≥ SD, ND·(SD/s)^k2 below the knee,
    k2 = 0 → fatigue limit (without damage). The probability densities below are densities of the AMPLITUDES of the cycles.
  - Methods (damage over the time T [s], Miner): D = T·Σ_c ν_c·∫ p_c(s)/N(s) ds
      nb      narrow band (Rayleigh): ν₀⁺, p = Rayleigh(σ² = m0) → without a knee D = ν₀⁺T(√(2m0))^kΓ(1 + k/2)/(ND·SD^k)
      dirlik  Dirlik 1985: ν_p, p = D1·Exp(Q√m0) + D2·Rayleigh(σ = R√m0) + D3·Rayleigh(σ = √m0)
              (Z = s/√m0; the same coefficients as FLife `Dirlik`)
      tb      Tovo–Benasciutti 2005 ("method 2" of FLife): D = b·D_NB + (1 − b)·D_RC, D_RC = α₂^(k−1)·D_NB, i.e. cycles
              ν₀⁺·b·Rayleigh(m0) + ν_p·(1 − b)·Rayleigh(α₂²m0); b = (α₁ − α₂)[1.112(1 + α₁α₂ − (α₁ + α₂))e^(2.11α₂)
              + (α₁ − α₂)]/(α₂ − 1)², clipped to [0; 1]
    The integrals over the S-N with a knee are EXACT (incomplete gamma function: ∫ s^k p ds above / below SD), vectorised over the nodes.
  - Mean stress (static states, 5a.6): Goodman / Gerber enlarge the amplitudes by the factor 1/(1 − σm/Rm) resp.
    1/(1 − (σm/Rm)²) (only σm > 0) – equivalently SD/λ.
  - Sinusoidal sweep (logarithmic, R octaves/min): f(t) = f₀·2^(R·t/60) → dn/df = 60/(R·ln 2) cycles per Hz per one pass
    (R = 1 → 86.56), D = passes·∫(dn/df)/N(σ_a(f)) df; σ_a(f) = amplitude of the steady harmonic
    response (quasi-static sweep – slow with respect to the width of the resonance).
"""
from __future__ import annotations

import math

import numpy as np

from .sn import SNCurve

METHODS = ("nb", "dirlik", "tb")
MEAN_METHODS = ("none", "goodman", "gerber")


def _moments(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
    if m.ndim != 2 or m.shape[1] != 4:
        raise ValueError("Moments must have shape (n, 4): m0, m1, m2, m4")
    return m


def params(m) -> dict:
    """Parameters of the process from the moments (n, 4): nu0 (ν₀⁺), nup (ν_p), alpha1, alpha2, xm (Dirlik) – arrays (n,).
    A zero process (m0 = 0) has rates 0 and α = 1.
    """
    m = _moments(m)
    m0, m1, m2, m4 = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
    ok = (m0 > 0) & (m2 > 0) & (m4 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu0 = np.where(ok, np.sqrt(m2 / m0), 0.0)
        nup = np.where(ok, np.sqrt(m4 / m2), 0.0)
        a1 = np.where(ok, m1 / np.sqrt(m0 * m2), 1.0)
        a2 = np.where(ok, m2 / np.sqrt(m0 * m4), 1.0)
        xm = np.where(ok, m1 / m0 * np.sqrt(m2 / m4), 1.0)
    return {"nu0": nu0, "nup": nup, "alpha1": np.minimum(a1, 1.0), "alpha2": np.minimum(a2, 1.0), "xm": xm,
            "ok": ok}


def dirlik_coefficients(m) -> dict:
    """Coefficients of the Dirlik density (arrays (n,)): D1, D2, D3, Q, R (Dirlik 1985, notation of Zorman et al. 2023)."""
    p = params(m)
    g = p["alpha2"]
    xm = p["xm"]
    D1 = 2.0 * (xm - g * g) / (1.0 + g * g)
    den = 1.0 - g - D1 + D1 * D1
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(np.abs(den) > 1e-14, (g - xm - D1 * D1) / den, 0.0)
        D2 = np.where(np.abs(1.0 - R) > 1e-14, den / (1.0 - R), 0.0)
        D3 = 1.0 - D1 - D2
        Q = np.where(D1 > 1e-12, 1.25 * (g - D3 - D2 * R) / D1, 1.0)
    narrow = (D1 <= 1e-12) | (g >= 1.0 - 1e-9) | ~p["ok"]      # narrow-band limit → Rayleigh
    D1 = np.where(narrow, 0.0, D1)
    D2 = np.where(narrow, 0.0, D2)
    D3 = np.where(narrow, 1.0, D3)
    return {"D1": D1, "D2": D2, "D3": D3, "Q": np.abs(Q), "R": np.abs(R)}


def tb_weight(m) -> np.ndarray:
    """Weight b of Tovo–Benasciutti 2005 (array (n,)), clipped to [0; 1]; a narrow-band process → 1."""
    p = params(m)
    a1, a2 = p["alpha1"], p["alpha2"]
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (a1 - a2) * (1.112 * (1.0 + a1 * a2 - (a1 + a2)) * np.exp(2.11 * a2) + (a1 - a2)) / (a2 - 1.0) ** 2
    b = np.where(a2 >= 1.0 - 1e-9, 1.0, b)
    return np.clip(np.nan_to_num(b, nan=1.0), 0.0, 1.0)


# ------------------------------------------------------------------ exact integrals of the S-N

def _sn_terms(sn: SNCurve):
    SD, ND, k, k2 = float(sn.SD), float(sn.ND), float(sn.k), float(sn.k2)
    if not (SD > 0 and ND > 0 and k > 0):
        raise ValueError("S-N curve: SD, ND and k must be positive")
    return SD, ND, k, k2


def _rayleigh_inv_n(sigma, SD, ND, k, k2):
    """E[1/N(s)] for the Rayleigh density of the amplitudes with the parameter sigma (array) and an S-N with a knee at SD (array/scalar)."""
    from scipy.special import gammaincc, gammainc, gammaln

    sigma = np.asarray(sigma, dtype=np.float64)
    out = np.zeros(np.broadcast(sigma, SD).shape)
    ok = sigma > 0
    if not np.any(ok):
        return out
    s = np.where(ok, sigma, 1.0)
    x = (SD / (math.sqrt(2.0) * s)) ** 2                       # SD²/(2σ²)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        lr = np.log(math.sqrt(2.0) * s / SD)
        hi = np.exp(k * lr + gammaln(1.0 + k / 2.0)) * gammaincc(1.0 + k / 2.0, x)
        if k2 > 0:
            lo = np.exp(k2 * lr + gammaln(1.0 + k2 / 2.0)) * gammainc(1.0 + k2 / 2.0, x)
        else:
            lo = 0.0
    out = np.where(ok, np.nan_to_num(hi, nan=0.0, posinf=np.inf) + np.nan_to_num(lo, nan=0.0), 0.0)
    return out / ND


def _exp_inv_n(lam, SD, ND, k, k2):
    """E[1/N(s)] for the exponential density of the amplitudes with the mean lam."""
    from scipy.special import gammaincc, gammainc, gammaln

    lam = np.asarray(lam, dtype=np.float64)
    ok = lam > 0
    l_ = np.where(ok, lam, 1.0)
    x = SD / l_
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        lr = np.log(l_ / SD)
        hi = np.exp(k * lr + gammaln(1.0 + k)) * gammaincc(1.0 + k, x)
        lo = np.exp(k2 * lr + gammaln(1.0 + k2)) * gammainc(1.0 + k2, x) if k2 > 0 else 0.0
    return np.where(ok, np.nan_to_num(hi, nan=0.0) + np.nan_to_num(lo, nan=0.0), 0.0) / ND


def _mean_factor(n, mean, Rm, mean_method):
    if mean is None or mean_method == "none":
        return np.ones(n)
    if mean_method not in MEAN_METHODS:
        raise ValueError(f"Unknown mean stress correction '{mean_method}'")
    if not Rm or Rm <= 0:
        raise ValueError("Mean stress correction needs Rm > 0")
    r = np.clip(np.broadcast_to(np.asarray(mean, dtype=np.float64), (n,)) / float(Rm), 0.0, None)
    with np.errstate(divide="ignore", over="ignore"):
        lam = 1.0 / (1.0 - r) if mean_method == "goodman" else 1.0 / (1.0 - r * r)
    return np.where(r >= 1.0, np.inf, lam)


def damage_rate(m, sn: SNCurve, method: str = "dirlik", scale=1.0) -> np.ndarray:
    """Damage per second (n,) for the moments m (n, 4). scale (n,) or a scalar = multiple of the stress amplitudes."""
    if method not in METHODS:
        raise ValueError(f"Unknown spectral fatigue method '{method}' (options: {', '.join(METHODS)})")
    m = _moments(m)
    SD, ND, k, k2 = _sn_terms(sn)
    n = m.shape[0]
    lam = np.broadcast_to(np.asarray(scale, dtype=np.float64), (n,))
    with np.errstate(divide="ignore"):
        SDe = np.where(lam > 0, SD / np.where(lam > 0, lam, 1.0), np.inf)   # increase of the amplitudes ≡ smaller SD
    p = params(m)
    s0 = np.sqrt(np.maximum(m[:, 0], 0.0))
    if method == "nb":
        d = p["nu0"] * _rayleigh_inv_n(s0, SDe, ND, k, k2)
    elif method == "tb":
        b = tb_weight(m)
        d = (p["nu0"] * b * _rayleigh_inv_n(s0, SDe, ND, k, k2)
             + p["nup"] * (1.0 - b) * _rayleigh_inv_n(p["alpha2"] * s0, SDe, ND, k, k2))
    else:
        c = dirlik_coefficients(m)
        d = p["nup"] * (c["D1"] * _exp_inv_n(c["Q"] * s0, SDe, ND, k, k2)
                        + c["D2"] * _rayleigh_inv_n(c["R"] * s0, SDe, ND, k, k2)
                        + c["D3"] * _rayleigh_inv_n(s0, SDe, ND, k, k2))
    d = np.where(np.isinf(lam), np.inf, d)
    return np.where(p["ok"] | np.isinf(lam), d, 0.0)


def damage(m, T: float, sn: SNCurve, method: str = "dirlik", mean=None, Rm: float | None = None,
           mean_method: str = "goodman") -> np.ndarray:
    """Damage D (n,) over the time T [s] for the moments m (n, 4) (m0, m1, m2, m4 one-sided PSD in Hz).
    mean (n,) = mean (static) stress [MPa] for the Goodman / Gerber correction (None = no correction).
    """
    m = _moments(m)
    lam = _mean_factor(m.shape[0], mean, Rm, mean_method)
    return float(T) * damage_rate(m, sn, method, lam)


def life_hours(D, T: float) -> np.ndarray:
    """Life [h] at the damage D over the time T [s] (D = 0 → ∞)."""
    D = np.asarray(D, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.where(D > 0, float(T) / np.where(D > 0, D, 1.0) / 3600.0, np.inf)


def safety(m, T: float, sn: SNCurve, method: str = "dirlik", target: float = 1.0, mean=None,
           Rm: float | None = None, mean_method: str = "goodman") -> np.ndarray:
    """Safety (n,) = multiple of the stress λ at which the damage over the time T equals `target`.

    D(λ) grows monotonically with the local exponent d ln D/d ln λ ≥ e_min = min(k, k2) (k2 = 0 → k) → the root lies
    between 1 and (target/D(1))^(1/e_min); searched by regula falsi (Illinois) in ln λ – ln D is almost linear,
    ~5–10 evaluations suffice. Zero stress → ∞; λ clipped to [1e-6; 1e6].
    """
    m = _moments(m)
    n = m.shape[0]
    T = float(T)
    base = _mean_factor(n, mean, Rm, mean_method)
    fin = np.isfinite(base) & (m[:, 0] > 0)
    lam = np.where(m[:, 0] > 0, np.nan, np.inf)
    lam[~np.isfinite(base)] = 0.0
    idx = np.nonzero(fin)[0]
    if idx.size == 0:
        return lam
    mm, bb = m[idx], base[idx]
    k, k2 = float(sn.k), float(sn.k2)
    e_min = min(k, k2) if k2 > 0 else k
    LO, HI = math.log(1e-6), math.log(1e6)
    lt = math.log(target)

    def g(x, sel):
        d = T * damage_rate(mm[sel], sn, method, bb[sel] * np.exp(x))
        return np.log(np.maximum(d, 1e-300)) - lt

    alln = np.arange(idx.size)
    g1 = g(np.zeros(idx.size), alln)
    if k2 == k:                                             # S-N without a knee: D ∝ λ^k exactly
        lam[idx] = np.exp(np.clip(-g1 / k, LO, HI))
        return lam
    xb = np.clip(-g1 / e_min, LO, HI)                       # ln λ from the smallest exponent
    a = np.minimum(0.0, xb)
    b = np.maximum(0.0, xb)
    ga = np.where(a == 0.0, g1, g(a, alln))
    gb = np.where(b == 0.0, g1, g(b, alln))
    for _ in range(40):                                     # safeguard (underflow of D at the fatigue limit)
        bad_a = np.nonzero((ga > 0) & (a > LO))[0]
        bad_b = np.nonzero((gb < 0) & (b < HI))[0]
        if bad_a.size == 0 and bad_b.size == 0:
            break
        a[bad_a] = np.maximum(LO, a[bad_a] - 2.0)
        b[bad_b] = np.minimum(HI, b[bad_b] + 2.0)
        if bad_a.size:
            ga[bad_a] = g(a[bad_a], bad_a)
        if bad_b.size:
            gb[bad_b] = g(b[bad_b], bad_b)
    x = np.where(ga > 0, a, np.where(gb < 0, b, 0.5 * (a + b)))   # root outside the limits → the limit
    act = np.nonzero((ga <= 0) & (gb >= 0) & (b > a))[0]
    last = np.zeros(idx.size, dtype=np.int8)               # 1 = the right end was replaced last time, −1 = the left
    for _ in range(100):                                    # regula falsi (Illinois) v ln λ
        if act.size == 0:
            break
        A, B, GA, GB = a[act], b[act], ga[act], gb[act]
        den = GB - GA
        c = np.where(den > 0, B - GB * (B - A) / np.where(den > 0, den, 1.0), 0.5 * (A + B))
        c = np.where((c <= A) | (c >= B), 0.5 * (A + B), c)
        gc = g(c, act)
        x[act] = c
        right = gc > 0                                      # root to the left of c → replace the right end
        L = last[act]
        a[act] = np.where(right, A, c)
        b[act] = np.where(right, c, B)
        ga[act] = np.where(right, np.where(L == 1, 0.5 * GA, GA), gc)
        gb[act] = np.where(right, gc, np.where(L == -1, 0.5 * GB, GB))
        last[act] = np.where(right, 1, -1)
        conv = (np.abs(gc) < 1e-9) | (b[act] - a[act] < 1e-10)
        act = act[~conv]
    lam[idx] = np.exp(x)
    return lam


def pdf(s, m, method: str = "dirlik") -> tuple[np.ndarray, float]:
    """Density of the cycle amplitudes p(s) for ONE process (m = (4,)) and the rate of cycles ν [1/s] (for graphs and tests).
    For tb it is the weighted mixture (ν = b·ν₀⁺ + (1 − b)·ν_p, p normalised to 1).
    """
    m = _moments(m)[:1]
    s = np.asarray(s, dtype=np.float64)
    p = params(m)
    m0 = float(m[0, 0])

    def ray(sig):
        return s / sig ** 2 * np.exp(-s * s / (2.0 * sig * sig))

    s0 = math.sqrt(m0)
    if method == "nb":
        return ray(s0), float(p["nu0"][0])
    if method == "tb":
        b = float(tb_weight(m)[0])
        a2 = float(p["alpha2"][0])
        nu0, nup = float(p["nu0"][0]), float(p["nup"][0])
        nu = b * nu0 + (1 - b) * nup
        return (b * nu0 * ray(s0) + (1 - b) * nup * ray(a2 * s0)) / nu, nu
    c = {k: float(v[0]) for k, v in dirlik_coefficients(m).items()}
    Z = s / s0
    pz = (c["D1"] / c["Q"] * np.exp(-Z / c["Q"]) if c["D1"] > 0 else 0.0) \
        + (c["D2"] * Z / c["R"] ** 2 * np.exp(-Z * Z / (2 * c["R"] ** 2)) if c["D2"] > 0 else 0.0) \
        + c["D3"] * Z * np.exp(-Z * Z / 2.0)
    return pz / s0, float(p["nup"][0])


# ------------------------------------------------------------------ sinusoidal sweep

def sweep_cycles_per_hz(rate_oct_min: float) -> float:
    """Number of cycles per 1 Hz per one pass of the logarithmic sweep of R [octaves/min]: 60/(R·ln 2)."""
    if not rate_oct_min > 0:
        raise ValueError("Sweep rate must be positive [oct/min]")
    return 60.0 / (float(rate_oct_min) * math.log(2.0))


def sweep_damage(f, sigma_a, rate_oct_min: float, sweeps: float, sn: SNCurve) -> np.ndarray:
    """Damage from the logarithmic sinusoidal sweep: D = passes·∫(dn/df)/N(σ_a(f)) df (trapezium in f).

    f (n_f,) [Hz] increasing – must be dense at the resonances (`dynamics.psd.freq_grid`), sigma_a (n_f,) or (n_f, n)
    = AMPLITUDE of the stress of the steady harmonic response [MPa] (e.g. max over the phase); sweeps = number of passes (one pass =
    one path f_min → f_max; a "sweep cycle" up and down = 2 passes). Returns a scalar or (n,).
    """
    from ..dynamics.psd import trapz_weights

    f = np.asarray(f, dtype=np.float64).ravel()
    if np.any(np.diff(f) <= 0):
        raise ValueError("Sweep frequencies must be increasing")
    S = np.asarray(sigma_a, dtype=np.float64)
    one = S.ndim == 1
    if one:
        S = S[:, None]
    if S.shape[0] != f.size:
        raise ValueError("sigma_a must have the same first dimension as f")
    D = float(sweeps) * sweep_cycles_per_hz(rate_oct_min) * (trapz_weights(f) @ inverse_life(S, sn))
    return float(D[0]) if one else D


def inverse_life(sigma_a, sn: SNCurve) -> np.ndarray:
    """1/N(σ_a) per element (S-N with the fatigue limit k2 = 0 → 0 below SD); for the summation of the damage over batches."""
    SD, ND, k, k2 = _sn_terms(sn)
    S = np.abs(np.asarray(sigma_a, dtype=np.float64))
    with np.errstate(divide="ignore", over="ignore"):
        r = S / SD
        return np.where(r >= 1.0, r ** k, (r ** k2 if k2 > 0 else 0.0)) / ND
