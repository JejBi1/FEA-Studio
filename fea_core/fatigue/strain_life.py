"""Low-cycle fatigue ε-N (strain-life) + Neuber / Glinka notch correction.

Theory and relations:
1. Ramberg-Osgood for the cyclic curve:
   ε = σ/E + (σ/K')^(1/n')
   Hysteresis loop (Masing): Δε/2 = Δσ/(2E) + (Δσ/(2K'))^(1/n')

2. Neuber rule for an elastic-plastic notch:
   σ · ε = σ_el · ε_el = (σ_el)^2 / E
   for cyclic ranges: Δσ · Δε = (Δσ_el)^2 / E  <=>  σ_a · ε_a = (σ_el,a)^2 / E

3. Glinka method (Equivalent Strain Energy Density - ESED):
   W_el = (σ_el)^2 / (2E) = σ^2 / (2E) + (σ / (n'+1)) · (σ/K')^(1/n')
   for cyclic ranges: W_el,a = (σ_el,a)^2 / (2E) = W(σ_a)

4. Coffin-Manson-Basquin equation for the life:
   Δε/2 = ε_a = (σ'_f / E) · (2N_f)^b + ε'_f · (2N_f)^c

5. Mean-stress correction:
   - Morrow: ε_a = ((σ'_f - σ_m) / E) · (2N_f)^b + ε'_f · (2N_f)^c
   - Smith-Watson-Topper (SWT): σ_max · ε_a = ((σ'_f)^2 / E) · (2N_f)^(2b) + σ'_f · ε'_f · (2N_f)^(b+c)

6. Bäumel-Seeger Uniform Material Law (UML):
   Estimate of the parameters of the cyclic curve and of the fatigue from the strength Rm and the modulus E for steels and Al/Ti alloys.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from ..modules import need
from . import kernel

_numba = need("numba", "accel")
njit = _numba.njit
prange = _numba.prange

# codes of the notch-correction rules
RULE_NEUBER = 0
RULE_GLINKA = 1

# codes of the mean-stress correction for ε-N
MEAN_EN_NONE = 0
MEAN_EN_MORROW = 1
MEAN_EN_SWT = 2

# labels taken from settings.py (removing duplication)
from .settings import NOTCH_RULE_LABELS, MEAN_LABELS as MEAN_EN_LABELS


@dataclass
class StrainLifeParams:
    """Parameters of the cyclic Ramberg-Osgood curve and of the Coffin-Manson-Basquin fatigue curve."""
    E: float              # tensile modulus of elasticity [MPa]
    sigma_f_prime: float  # fatigue strength coefficient σ'_f [MPa]
    b: float              # fatigue strength exponent b [-] (usually −0.05 to −0.15)
    eps_f_prime: float    # fatigue ductility coefficient ε'_f [-] (usually 0.1 to 1.0)
    c: float              # fatigue ductility exponent c [-] (usually −0.5 to −0.7)
    K_prime: float        # cyclic strength coefficient K' [MPa]
    n_prime: float        # cyclic strain-hardening exponent n' [-] (usually 0.1 to 0.25)
    source: str = "custom"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StrainLifeParams":
        d = dict(d)
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})


def estimate_strain_life(Rm: float, E: float, material_type: str = "steel") -> StrainLifeParams:
    """Estimate of the parameters according to the Bäumel-Seeger Uniform Material Law (UML, 1990).

    According to: Bäumel, A. Jr., Seeger, T. (1990): Materials Data for Cyclic Loading, Supplement 1.
    - Structural and low-alloy steels (for Rm/E <= 0.003 psi = 1.0, for > 0.003 psi = 1.375 - 125*(Rm/E))
    - Aluminium and titanium alloys
    """
    if Rm <= 0.0 or E <= 0.0:
        raise ValueError(f"Rm and E must be positive (entered Rm={Rm}, E={E})")

    mat_type = material_type.lower().strip()
    is_al_ti = False

    # 1. We try the converter of material designations
    try:
        from .. import grades
        g = grades.grade_of(material_type)
        if g is not None:
            if g.base in ("al", "ti", "mg"):
                is_al_ti = True
            elif g.base in ("ocel", "litina", "fe"):
                is_al_ti = False
    except Exception:
        pass

    # 2. Word tokens (exact words, not plain substrings such as 'al' in 'galvanized')
    if not is_al_ti:
        import re
        tokens = set(re.findall(r"[a-zA-Z0-9_á-žÁ-Ž]+", mat_type))
        steel_tokens = {
            "steel", "steels", "ocel", "oceli", "ferritic", "austenitic",
            "carbon", "stainless", "litina", "iron", "nerez"
        }
        al_tokens = {
            "al", "ti", "aluminum", "aluminium", "hlinik", "hliník",
            "dural", "titan", "titanium", "ti6al4v", "alsi"
        }
        if tokens & steel_tokens:
            is_al_ti = False
        elif tokens & al_tokens:
            is_al_ti = True
        elif mat_type.startswith("al") or mat_type.startswith("ti") or mat_type.startswith("en aw"):
            is_al_ti = True
        elif mat_type == "auto" or not mat_type:
            is_al_ti = E < 150000.0

    if is_al_ti:
        sigma_f_prime = 1.67 * Rm
        b = -0.095
        eps_f_prime = 0.35
        c = -0.69
        K_prime = 1.61 * Rm
        n_prime = 0.11
        source = f"Bäumel-Seeger UML (Al/Ti, Rm={Rm:g} MPa, E={E:g} MPa)"
    else:
        sigma_f_prime = 1.5 * Rm
        b = -0.087
        ratio = Rm / E
        psi = 1.0 if ratio <= 3e-3 else (1.375 - 125.0 * ratio)
        eps_f_prime = 0.59 * psi
        c = -0.58
        K_prime = 1.65 * Rm
        n_prime = 0.15
        source = f"Bäumel-Seeger UML (steel, Rm={Rm:g} MPa, E={E:g} MPa)"

    return StrainLifeParams(
        E=float(E),
        sigma_f_prime=float(sigma_f_prime),
        b=float(b),
        eps_f_prime=float(eps_f_prime),
        c=float(c),
        K_prime=float(K_prime),
        n_prime=float(n_prime),
        source=source,
    )


# ------------------------------------------------------------------ Ramberg-Osgood
def ramberg_osgood_strain(sigma: float, E: float, K_prime: float, n_prime: float) -> float:
    """Cyclic strain ε from the stress σ: ε = σ/E + sgn(σ)·(|σ|/K')^(1/n')."""
    if sigma == 0.0:
        return 0.0
    sgn = 1.0 if sigma > 0.0 else -1.0
    s_abs = abs(sigma)
    return sigma / E + sgn * (s_abs / K_prime) ** (1.0 / n_prime)


def ramberg_osgood_stress(eps: float, E: float, K_prime: float, n_prime: float) -> float:
    """Cyclic stress σ from the strain ε (inversion of Ramberg-Osgood by Newton's method)."""
    if abs(eps) <= 1e-15:
        return 0.0
    sgn = 1.0 if eps > 0.0 else -1.0
    e_abs = abs(eps)
    sig = _ro_stress_fast(e_abs, E, K_prime, n_prime)
    return sgn * sig


@njit(cache=True)
def _ro_stress_fast(eps_abs: float, E: float, K_prime: float, n_prime: float) -> float:
    if eps_abs <= 1e-15:
        return 0.0
    sig = min(eps_abs * E, K_prime * (eps_abs ** n_prime))
    for _ in range(50):
        sp = (sig / K_prime) ** (1.0 / n_prime)
        f = sig / E + sp - eps_abs
        fp = 1.0 / E + (1.0 / (n_prime * K_prime)) * ((sig / K_prime) ** (1.0 / n_prime - 1.0))
        ds = f / fp
        sig -= ds
        if abs(ds) < 1e-9 * sig:
            break
    return sig


# ------------------------------------------------------------------ notch corrections (Neuber / Glinka)
@njit(cache=True)
def _neuber_solve_fast(sigma_el: float, E: float, K_prime: float, n_prime: float) -> tuple[float, float]:
    """Solution of the Neuber rule: σ · ε = σ_el² / E, where ε = σ/E + (σ/K')^(1/n')."""
    if sigma_el <= 0.0:
        return 0.0, 0.0
    lo = 0.0
    hi = sigma_el
    sig = min(sigma_el, K_prime * (sigma_el * sigma_el / (E * K_prime)) ** (n_prime / (n_prime + 1.0)))
    if sig < 1e-6:
        sig = 1e-6
    coeff_inv_n = 1.0 / n_prime
    s_el_sq = sigma_el * sigma_el

    for _ in range(60):
        sp = (sig / K_prime) ** coeff_inv_n
        g = sig * sig + E * sig * sp - s_el_sq
        if abs(g) < 1e-11 * max(s_el_sq, 1.0):
            break
        if g < 0.0:
            lo = max(lo, sig)
        else:
            hi = min(hi, sig)
        gp = 2.0 * sig + E * (1.0 + coeff_inv_n) * sp
        if gp <= 0.0:
            sig = 0.5 * (lo + hi)
            continue
        ds = g / gp
        next_sig = sig - ds
        if next_sig <= lo or next_sig >= hi:
            sig = 0.5 * (lo + hi)
            if (hi - lo) < 1e-9 * max(sig, 1e-6):
                break
        else:
            sig = next_sig
            if abs(ds) < 1e-9 * max(sig, 1e-6):
                break

    eps = sig / E + (sig / K_prime) ** coeff_inv_n
    return sig, eps


@njit(cache=True)
def _glinka_solve_fast(sigma_el: float, E: float, K_prime: float, n_prime: float) -> tuple[float, float]:
    """Solution of the Glinka method (ESED): W_el = σ_el²/(2E) = σ²/(2E) + (σ/(n'+1))·(σ/K')^(1/n')."""
    if sigma_el <= 0.0:
        return 0.0, 0.0
    lo = 0.0
    hi = sigma_el
    sig = min(sigma_el, K_prime * (sigma_el * sigma_el / (E * K_prime)) ** (n_prime / (n_prime + 1.0)))
    if sig < 1e-6:
        sig = 1e-6
    coeff = 2.0 * E / (n_prime + 1.0)
    coeff_inv_n = 1.0 / n_prime
    hp_coeff = 2.0 * E / n_prime
    s_el_sq = sigma_el * sigma_el

    for _ in range(60):
        sp = (sig / K_prime) ** coeff_inv_n
        h = sig * sig + coeff * sig * sp - s_el_sq
        if abs(h) < 1e-11 * max(s_el_sq, 1.0):
            break
        if h < 0.0:
            lo = max(lo, sig)
        else:
            hi = min(hi, sig)
        hp = 2.0 * sig + hp_coeff * sp
        if hp <= 0.0:
            sig = 0.5 * (lo + hi)
            continue
        ds = h / hp
        next_sig = sig - ds
        if next_sig <= lo or next_sig >= hi:
            sig = 0.5 * (lo + hi)
            if (hi - lo) < 1e-9 * max(sig, 1e-6):
                break
        else:
            sig = next_sig
            if abs(ds) < 1e-9 * max(sig, 1e-6):
                break

    eps = sig / E + (sig / K_prime) ** coeff_inv_n
    return sig, eps


def neuber_solve(sigma_el: float, E: float, K_prime: float, n_prime: float) -> tuple[float, float]:
    """Neuber rule for the given elastic stress: returns (sigma, epsilon)."""
    if sigma_el == 0.0:
        return 0.0, 0.0
    sgn = 1.0 if sigma_el > 0.0 else -1.0
    sig, eps = _neuber_solve_fast(abs(sigma_el), float(E), float(K_prime), float(n_prime))
    return sgn * sig, sgn * eps


def glinka_solve(sigma_el: float, E: float, K_prime: float, n_prime: float) -> tuple[float, float]:
    """Glinka method (ESED) for the given elastic stress: returns (sigma, epsilon)."""
    if sigma_el == 0.0:
        return 0.0, 0.0
    sgn = 1.0 if sigma_el > 0.0 else -1.0
    sig, eps = _glinka_solve_fast(abs(sigma_el), float(E), float(K_prime), float(n_prime))
    return sgn * sig, sgn * eps


def notch_correction(
    sigma_el: float,
    E: float,
    K_prime: float,
    n_prime: float,
    rule: str = "neuber",
) -> tuple[float, float]:
    """Notch correction (rule='neuber' | 'glinka') for the elastic stress: returns (sigma, epsilon)."""
    r = rule.lower()
    if r == "neuber":
        return neuber_solve(sigma_el, E, K_prime, n_prime)
    elif r == "glinka":
        return glinka_solve(sigma_el, E, K_prime, n_prime)
    raise ValueError(f"Unknown notch correction rule '{rule}' (allowed: 'neuber' or 'glinka')")


# ------------------------------------------------------------------ Coffin-Manson-Basquin & mean stress
@njit(cache=True)
def _coffin_manson_life_fast(
    eps_a: float,
    E: float,
    sigma_f_prime: float,
    b: float,
    eps_f_prime: float,
    c: float,
) -> float:
    """Computation of the life N_f from the Coffin-Manson-Basquin equation:
    eps_a = (sigma_f_prime / E) * (2N_f)^b + eps_f_prime * (2N_f)^c
    """
    if eps_a <= 1e-12:
        return math.inf
    u_lo = 0.0
    u_hi = 70.0  # 2N_f ~ 2.5e30

    f_lo = (sigma_f_prime / E) * math.exp(b * u_lo) + eps_f_prime * math.exp(c * u_lo) - eps_a
    if f_lo <= 0.0:
        return 0.5
    f_hi = (sigma_f_prime / E) * math.exp(b * u_hi) + eps_f_prime * math.exp(c * u_hi) - eps_a
    if f_hi >= 0.0:
        return math.inf

    u = 10.0
    for _ in range(60):
        e_bu = math.exp(b * u)
        e_cu = math.exp(c * u)
        f = (sigma_f_prime / E) * e_bu + eps_f_prime * e_cu - eps_a
        if abs(f) < 1e-12 * eps_a:
            break
        if f > 0.0:
            u_lo = max(u_lo, u)
        else:
            u_hi = min(u_hi, u)
        fp = b * (sigma_f_prime / E) * e_bu + c * eps_f_prime * e_cu
        if fp >= 0.0:
            u = 0.5 * (u_lo + u_hi)
            continue
        du = f / fp
        next_u = u - du
        if next_u <= u_lo or next_u >= u_hi:
            u = 0.5 * (u_lo + u_hi)
            if (u_hi - u_lo) < 1e-9:
                break
        else:
            u = next_u
            if abs(du) < 1e-9:
                break
    return 0.5 * math.exp(u)


@njit(cache=True)
def _morrow_life_fast(
    eps_a: float,
    sigma_m: float,
    E: float,
    sigma_f_prime: float,
    b: float,
    eps_f_prime: float,
    c: float,
) -> float:
    """Morrow mean-stress correction:
    eps_a = ((sigma_f_prime - sigma_m) / E) * (2N_f)^b + eps_f_prime * (2N_f)^c
    """
    if eps_a <= 1e-12:
        return math.inf
    sf_eff = sigma_f_prime - sigma_m
    if sf_eff <= 0.0:
        return 0.5

    u_lo = 0.0
    u_hi = 70.0
    f_lo = (sf_eff / E) * math.exp(b * u_lo) + eps_f_prime * math.exp(c * u_lo) - eps_a
    if f_lo <= 0.0:
        return 0.5
    f_hi = (sf_eff / E) * math.exp(b * u_hi) + eps_f_prime * math.exp(c * u_hi) - eps_a
    if f_hi >= 0.0:
        return math.inf

    u = 10.0
    for _ in range(60):
        e_bu = math.exp(b * u)
        e_cu = math.exp(c * u)
        f = (sf_eff / E) * e_bu + eps_f_prime * e_cu - eps_a
        if abs(f) < 1e-12 * eps_a:
            break
        if f > 0.0:
            u_lo = max(u_lo, u)
        else:
            u_hi = min(u_hi, u)
        fp = b * (sf_eff / E) * e_bu + c * eps_f_prime * e_cu
        if fp >= 0.0:
            u = 0.5 * (u_lo + u_hi)
            continue
        du = f / fp
        next_u = u - du
        if next_u <= u_lo or next_u >= u_hi:
            u = 0.5 * (u_lo + u_hi)
            if (u_hi - u_lo) < 1e-9:
                break
        else:
            u = next_u
            if abs(du) < 1e-9:
                break
    return 0.5 * math.exp(u)


@njit(cache=True)
def _swt_life_fast(
    eps_a: float,
    sigma_max: float,
    E: float,
    sigma_f_prime: float,
    b: float,
    eps_f_prime: float,
    c: float,
) -> float:
    """Smith-Watson-Topper (SWT) parameter for the mean stress:
    sigma_max · eps_a = ((sigma_f_prime)^2 / E) * (2N_f)^(2b) + sigma_f_prime * eps_f_prime * (2N_f)^(b+c)
    """
    if eps_a <= 1e-12 or sigma_max <= 0.0:
        return math.inf
    p_swt = sigma_max * eps_a

    u_lo = 0.0
    u_hi = 70.0
    coeff1 = (sigma_f_prime * sigma_f_prime) / E
    coeff2 = sigma_f_prime * eps_f_prime
    exp1 = 2.0 * b
    exp2 = b + c

    f_lo = coeff1 * math.exp(exp1 * u_lo) + coeff2 * math.exp(exp2 * u_lo) - p_swt
    if f_lo <= 0.0:
        return 0.5
    f_hi = coeff1 * math.exp(exp1 * u_hi) + coeff2 * math.exp(exp2 * u_hi) - p_swt
    if f_hi >= 0.0:
        return math.inf

    u = 10.0
    for _ in range(60):
        e1 = math.exp(exp1 * u)
        e2 = math.exp(exp2 * u)
        f = coeff1 * e1 + coeff2 * e2 - p_swt
        if abs(f) < 1e-12 * p_swt:
            break
        if f > 0.0:
            u_lo = max(u_lo, u)
        else:
            u_hi = min(u_hi, u)
        fp = exp1 * coeff1 * e1 + exp2 * coeff2 * e2
        if fp >= 0.0:
            u = 0.5 * (u_lo + u_hi)
            continue
        du = f / fp
        next_u = u - du
        if next_u <= u_lo or next_u >= u_hi:
            u = 0.5 * (u_lo + u_hi)
            if (u_hi - u_lo) < 1e-9:
                break
        else:
            u = next_u
            if abs(du) < 1e-9:
                break
    return 0.5 * math.exp(u)


def coffin_manson_life(
    eps_a: float,
    E: float,
    sigma_f_prime: float,
    b: float,
    eps_f_prime: float,
    c: float,
) -> float:
    """Computation of the life N_f from the Coffin-Manson-Basquin equation (number of cycles to failure)."""
    return _coffin_manson_life_fast(float(eps_a), float(E), float(sigma_f_prime), float(b),
                                   float(eps_f_prime), float(c))


def morrow_life(
    eps_a: float,
    sigma_m: float,
    E: float,
    sigma_f_prime: float,
    b: float,
    eps_f_prime: float,
    c: float,
) -> float:
    """Computation of the life N_f according to Morrow with the mean stress sigma_m."""
    return _morrow_life_fast(float(eps_a), float(sigma_m), float(E), float(sigma_f_prime),
                             float(b), float(eps_f_prime), float(c))


def swt_life(
    eps_a: float,
    sigma_max: float,
    E: float,
    sigma_f_prime: float,
    b: float,
    eps_f_prime: float,
    c: float,
) -> float:
    """Computation of the life N_f according to Smith-Watson-Topper (SWT) with the maximum stress sigma_max."""
    return _swt_life_fast(float(eps_a), float(sigma_max), float(E), float(sigma_f_prime),
                          float(b), float(eps_f_prime), float(c))


def strain_life_cycles(
    eps_a: float,
    sigma_m: float = 0.0,
    sigma_max: float | None = None,
    params: StrainLifeParams | None = None,
    mean_correction: str = "none",
) -> float:
    """Computation of the life N_f [cycles] for the given strain amplitude and mean stress."""
    if params is None:
        raise ValueError("StrainLifeParams must be provided")
    mc = mean_correction.lower()
    if mc in ("none", "cm"):
        return coffin_manson_life(eps_a, params.E, params.sigma_f_prime, params.b,
                                  params.eps_f_prime, params.c)
    elif mc == "morrow":
        return morrow_life(eps_a, sigma_m, params.E, params.sigma_f_prime, params.b,
                           params.eps_f_prime, params.c)
    elif mc == "swt":
        smax = sigma_max if sigma_max is not None else (sigma_m + params.E * eps_a)
        return swt_life(eps_a, smax, params.E, params.sigma_f_prime, params.b,
                        params.eps_f_prime, params.c)
    raise ValueError(f"Unknown mean stress correction '{mean_correction}' (allowed: 'none', 'morrow', 'swt')")


# ------------------------------------------------------------------ damage and rainflow integration
@njit(cache=True)
def _cycle_damage_en(
    s_el_a: float,
    s_el_m: float,
    cnt: float,
    E: float,
    K_prime: float,
    n_prime: float,
    sigma_f_prime: float,
    b: float,
    eps_f_prime: float,
    c: float,
    rule_code: int,
    mean_code: int,
) -> float:
    """Damage of one cycle of the elastic amplitude s_el_a and the mean value s_el_m."""
    if s_el_a <= 0.0 or cnt <= 0.0:
        return 0.0

    # 1. Notch correction for the amplitude (cyclic loop)
    if rule_code == RULE_GLINKA:
        sig_a, eps_a = _glinka_solve_fast(s_el_a, E, K_prime, n_prime)
    else:
        sig_a, eps_a = _neuber_solve_fast(s_el_a, E, K_prime, n_prime)

    # 2. Determination of the mean or the maximum stress
    if mean_code == MEAN_EN_NONE:
        nf = _coffin_manson_life_fast(eps_a, E, sigma_f_prime, b, eps_f_prime, c)
    elif mean_code == MEAN_EN_MORROW:
        if s_el_m == 0.0:
            sm = 0.0
        else:
            s_el_max = s_el_m + s_el_a
            if s_el_max > 0.0:
                if rule_code == RULE_GLINKA:
                    sig_max, _ = _glinka_solve_fast(s_el_max, E, K_prime, n_prime)
                else:
                    sig_max, _ = _neuber_solve_fast(s_el_max, E, K_prime, n_prime)
                sm = sig_max - sig_a
            else:
                sm = s_el_m
        nf = _morrow_life_fast(eps_a, sm, E, sigma_f_prime, b, eps_f_prime, c)
    elif mean_code == MEAN_EN_SWT:
        s_el_max = s_el_m + s_el_a
        if s_el_max <= 0.0:
            return 0.0
        if rule_code == RULE_GLINKA:
            sig_max, _ = _glinka_solve_fast(s_el_max, E, K_prime, n_prime)
        else:
            sig_max, _ = _neuber_solve_fast(s_el_max, E, K_prime, n_prime)
        nf = _swt_life_fast(eps_a, sig_max, E, sigma_f_prime, b, eps_f_prime, c)
    else:
        nf = _coffin_manson_life_fast(eps_a, E, sigma_f_prime, b, eps_f_prime, c)

    if nf <= 0.0:
        return math.inf
    return cnt / nf


@njit(cache=True)
def damage_en(
    rng: np.ndarray,
    mean: np.ndarray,
    cnt: np.ndarray,
    nc: int,
    scale: float,
    E: float,
    K_prime: float,
    n_prime: float,
    sigma_f_prime: float,
    b: float,
    eps_f_prime: float,
    c: float,
    rule_code: int,
    mean_code: int,
    stress_scale: float = 1.0,
    mean_scale: float = 1.0,
) -> float:
    """Total cumulative damage (Palmgren-Miner) of the rainflow cycles at the given stress scale."""
    d_total = 0.0
    scale = abs(scale)
    s_scale = scale * stress_scale
    m_scale = scale * mean_scale
    for i in range(nc):
        s_el_a = 0.5 * rng[i] * s_scale
        s_el_m = mean[i] * m_scale
        c_val = cnt[i]
        d = _cycle_damage_en(
            s_el_a, s_el_m, c_val, E, K_prime, n_prime,
            sigma_f_prime, b, eps_f_prime, c, rule_code, mean_code
        )
        d_total += d
    return d_total


@njit(cache=True)
def node_result_en(
    rng: np.ndarray,
    mean: np.ndarray,
    cnt: np.ndarray,
    nc: int,
    E: float,
    K_prime: float,
    n_prime: float,
    sigma_f_prime: float,
    b: float,
    eps_f_prime: float,
    c: float,
    rule_code: int,
    mean_code: int,
    repeats: float,
    stress_scale: float = 1.0,
    mean_scale: float = 1.0,
) -> tuple[float, float, float]:
    """Result of a node: (damage per 1 block, safety, max. elastic amplitude)."""
    amax = 0.0
    for i in range(nc):
        a = 0.5 * rng[i] * stress_scale
        if a > amax:
            amax = a

    d1 = damage_en(
        rng, mean, cnt, nc, 1.0, E, K_prime, n_prime,
        sigma_f_prime, b, eps_f_prime, c, rule_code, mean_code,
        stress_scale=stress_scale, mean_scale=mean_scale,
    )

    if nc == 0 or amax <= 0.0:
        return d1, math.inf, amax

    val1 = repeats * d1
    if val1 <= 0.0:
        return d1, math.inf, amax
    if abs(val1 - 1.0) < 1e-6:
        return d1, 1.0, amax

    # 1. Bracketing of the root
    if val1 < 1.0:
        lo = 1.0
        d_lo = val1
        hi = 2.0
        d_hi = repeats * damage_en(
            rng, mean, cnt, nc, hi, E, K_prime, n_prime,
            sigma_f_prime, b, eps_f_prime, c, rule_code, mean_code,
            stress_scale=stress_scale, mean_scale=mean_scale,
        )
        while d_hi < 1.0:
            lo = hi
            d_lo = d_hi
            hi *= 2.0
            if hi > 1e4:
                return d1, math.inf, amax
            d_hi = repeats * damage_en(
                rng, mean, cnt, nc, hi, E, K_prime, n_prime,
                sigma_f_prime, b, eps_f_prime, c, rule_code, mean_code,
                stress_scale=stress_scale, mean_scale=mean_scale,
            )
    else:
        hi = 1.0
        d_hi = val1
        lo = 0.5
        d_lo = repeats * damage_en(
            rng, mean, cnt, nc, lo, E, K_prime, n_prime,
            sigma_f_prime, b, eps_f_prime, c, rule_code, mean_code,
            stress_scale=stress_scale, mean_scale=mean_scale,
        )
        while d_lo > 1.0:
            hi = lo
            d_hi = d_lo
            lo *= 0.5
            if lo < 1e-4:
                return d1, 0.0, amax
            d_lo = repeats * damage_en(
                rng, mean, cnt, nc, lo, E, K_prime, n_prime,
                sigma_f_prime, b, eps_f_prime, c, rule_code, mean_code,
                stress_scale=stress_scale, mean_scale=mean_scale,
            )

    # 2. Root finding: regula falsi (Illinois) in log-log space (converges in about 4–8 evaluations)
    u_lo = math.log(max(1e-8, lo))
    v_lo = math.log(max(1e-30, d_lo))
    u_hi = math.log(max(1e-8, hi))
    v_hi = math.log(max(1e-30, d_hi))

    side = 0
    sf = 0.5 * (lo + hi)
    for _ in range(16):
        denom = v_hi - v_lo
        if abs(denom) < 1e-12:
            break
        u_next = u_lo - v_lo * (u_hi - u_lo) / denom
        margin = 0.001 * (u_hi - u_lo)
        u_next = max(u_lo + margin, min(u_hi - margin, u_next))
        s_next = math.exp(u_next)
        d_val = repeats * damage_en(
            rng, mean, cnt, nc, s_next, E, K_prime, n_prime,
            sigma_f_prime, b, eps_f_prime, c, rule_code, mean_code,
            stress_scale=stress_scale, mean_scale=mean_scale,
        )
        sf = s_next
        if abs(d_val - 1.0) < 1e-5 or (u_hi - u_lo) < 1e-5:
            break
        v_next = math.log(max(1e-30, d_val))
        if v_next < 0.0:
            u_lo, v_lo = u_next, v_next
            if side == -1:
                v_hi *= 0.5
            side = -1
        else:
            u_hi, v_hi = u_next, v_next
            if side == 1:
                v_lo *= 0.5
            side = 1

    return d1, sf, amax


@njit(parallel=True, cache=True)
def fatigue_nodes_en(
    F: np.ndarray,
    S: np.ndarray,
    eq_method: int,
    rule_code: int,
    mean_code: int,
    E_arr: np.ndarray,
    Kp_arr: np.ndarray,
    np_arr: np.ndarray,
    sfp_arr: np.ndarray,
    b_arr: np.ndarray,
    efp_arr: np.ndarray,
    c_arr: np.ndarray,
    repeats: float,
    closed: bool,
    out_D: np.ndarray,
    out_sf: np.ndarray,
    out_amp: np.ndarray,
):
    """Parallel evaluation of the ε-N fatigue for a general history F (T, K) and the stresses S (K, n, 6)."""
    T, K = F.shape
    n = S.shape[1]
    for j in prange(n):
        x = np.empty(T)
        for t in range(T):
            s0 = s1 = s2 = s3 = s4 = s5 = 0.0
            for kk in range(K):
                f = F[t, kk]
                s0 += f * S[kk, j, 0]
                s1 += f * S[kk, j, 1]
                s2 += f * S[kk, j, 2]
                s3 += f * S[kk, j, 3]
                s4 += f * S[kk, j, 4]
                s5 += f * S[kk, j, 5]
            x[t] = kernel.equivalent(s0, s1, s2, s3, s4, s5, eq_method)
        if closed:
            x = kernel.close_block(x)
        L = x.shape[0] + 1
        rng = np.empty(L)
        mean = np.empty(L)
        cnt = np.empty(L)
        nc = kernel.rainflow_cycles(x, rng, mean, cnt)
        nc = kernel.merge_cycles(rng, mean, cnt, nc)
        out_D[j], out_sf[j], out_amp[j] = node_result_en(
            rng, mean, cnt, nc,
            E_arr[j], Kp_arr[j], np_arr[j],
            sfp_arr[j], b_arr[j], efp_arr[j], c_arr[j],
            rule_code, mean_code, repeats
        )


@njit(parallel=True, cache=True)
def fatigue_nodes_en_proportional(
    f: np.ndarray,
    Seff: np.ndarray,
    eq_method: int,
    rule_code: int,
    mean_code: int,
    E_arr: np.ndarray,
    Kp_arr: np.ndarray,
    np_arr: np.ndarray,
    sfp_arr: np.ndarray,
    b_arr: np.ndarray,
    efp_arr: np.ndarray,
    c_arr: np.ndarray,
    repeats: float,
    closed: bool,
    out_D: np.ndarray,
    out_sf: np.ndarray,
    out_amp: np.ndarray,
):
    """Fast proportional path of the ε-N fatigue: rainflow computed only once over the signal f(t)."""
    x = kernel.close_block(f) if closed else f.copy()
    L = x.shape[0] + 1
    rng_f = np.empty(L)
    mean_f = np.empty(L)
    cnt_f = np.empty(L)
    nc = kernel.rainflow_cycles(x, rng_f, mean_f, cnt_f)
    nc = kernel.merge_cycles(rng_f, mean_f, cnt_f, nc)

    n = Seff.shape[0]
    rng_f_use = rng_f[:nc]
    mean_f_use = mean_f[:nc]
    cnt_f_use = cnt_f[:nc]

    for j in prange(n):
        s0 = Seff[j, 0]
        s1 = Seff[j, 1]
        s2 = Seff[j, 2]
        s3 = Seff[j, 3]
        s4 = Seff[j, 4]
        s5 = Seff[j, 5]
        s_eq = kernel.equivalent(s0, s1, s2, s3, s4, s5, eq_method)
        abs_s = abs(s_eq)

        out_D[j], out_sf[j], out_amp[j] = node_result_en(
            rng_f_use, mean_f_use, cnt_f_use, nc,
            E_arr[j], Kp_arr[j], np_arr[j],
            sfp_arr[j], b_arr[j], efp_arr[j], c_arr[j],
            rule_code, mean_code, repeats,
            stress_scale=abs_s, mean_scale=s_eq
        )
