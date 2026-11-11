"""Verification cases of fatigue (stage F9.4):
- U4: low-cycle fatigue e-N – a bar with a notch Kt (plate with a hole from V3).
  Comparison of Neuber vs. Glinka vs. the analytic estimate and the life by Coffin-Manson / Morrow / SWT.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from fea_core.study import Study
from fea_core.results import Results
from fea_core.geometry import Part
from fea_core.fatigue.settings import FatigueSettings
from fea_core.fatigue.history import LoadHistory, Channel
from fea_core.fatigue.analysis import run_fatigue
from fea_core.fatigue.strain_life import (
    estimate_strain_life,
    neuber_solve,
    glinka_solve,
    coffin_manson_life,
    morrow_life,
    swt_life,
)

# ===================================================================== U4
U4_CYCLES = 1000
U4_REPEATS = 100


def _zivotnost_puleni(rhs, eps_a, lhs_extra=0.0):
    """Independent inversion of the Coffin-Manson-Basquin equation by bisection of the interval in log N.
    rhs(x) = the right-hand side as a function of x = ln(2N) (decreasing), we look for rhs(x) = eps_a.
    Returns N_f (bisection independent of the numba solver of the program).
    """
    lo, hi = 0.0, 70.0
    if rhs(lo) <= eps_a:
        return 0.5
    if rhs(hi) >= eps_a:
        return math.inf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rhs(mid) > eps_a:
            lo = mid
        else:
            hi = mid
    return 0.5 * math.exp(0.5 * (lo + hi))


def _cm_nezavisle(eps_a, E, sf, b, ef, c, sigma_m=0.0):
    """Coffin-Manson-Basquin (Morrow: sf -> sf - sigma_m) by bisection, without the program's code."""
    return _zivotnost_puleni(
        lambda x: (sf - sigma_m) / E * math.exp(b * x) + ef * math.exp(c * x), eps_a)


def _swt_nezavisle(eps_a, sigma_max, E, sf, b, ef, c):
    """SWT: sigma_max·eps_a = sf²/E·(2N)^(2b) + sf·ef·(2N)^(b+c), by bisection."""
    return _zivotnost_puleni(
        lambda x: (sf * sf / E * math.exp(2.0 * b * x) + sf * ef * math.exp((b + c) * x)),
        sigma_max * eps_a)


def _glinka_nezavisle(sig_el, E, K, n):
    """Glinka: ESED σ²/(2E) + σ/(n'+1)·(σ/K')^(1/n') = σ_el²/(2E) by bisection on σ ∈ (0, σ_el]."""
    wel = sig_el * sig_el / (2.0 * E)
    lo, hi = 0.0, sig_el
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        w = mid * mid / (2.0 * E) + mid / (n + 1.0) * (mid / K) ** (1.0 / n)
        if w < wel:
            lo = mid
        else:
            hi = mid
    sig = 0.5 * (lo + hi)
    return sig, sig / E + (sig / K) ** (1.0 / n)


def u4_build(wd: Path) -> tuple[Study, list[Part]]:
    """U4: Bar with a notch (plate with a hole 150×50×2.5 mm, d=10 mm, Peterson Kt=2.729).
    Loading: pulsating tension 100 MPa at the end (amplitude 100 MPa, 1000 cycles per block).
    Fatigue: low-cycle e-N (strain-life), Neuber rule, Coffin-Manson.
    """
    from .cases import v3_build
    st, parts = v3_build(wd)
    st.name = "U4 tyc s vrubem Kt, nizkocyklova unava e-N"
    channel_name = st.load_cases[0].name
    st.fatigue = FatigueSettings(
        history=LoadHistory([Channel(channel_name, "sine", amplitude=1.0, cycles=U4_CYCLES)]),
        method="en",
        en_notch_rule="neuber",
        mean_correction="none",
        repeats=float(U4_REPEATS),
    )
    return st, parts


def u4_eval(res: Results, st: Study):
    """Evaluation of U4:
    1. Analytic check of the Neuber rule for the notch Kt:
       σ_el = Kt_net * σ_net = 2.729 * 111.11 = 303.2 MPa
       σ · ε = σ_el² / E
    2. Analytic check of Glinka's method (ESED):
       W = W_el = σ_el² / (2E)
    3. Analytic calculation of the life by Coffin-Manson-Basquin, Morrow and SWT.
    4. Comparison with the FEA results at the root of the notch (hole).
    """
    from .cases import Check

    # 1. Peterson's analytic stress concentration factor
    V3_W, V3_R = 50.0, 5.0  # half width, radius of the hole
    d_over_W = (2.0 * V3_R) / (2.0 * V3_W)  # 10 / 100 = 0.1
    kt_net = 3.0 - 3.14 * d_over_W + 3.667 * (d_over_W ** 2) - 1.527 * (d_over_W ** 3)  # ≈ 2.181
    kt_gross = kt_net * (2.0 * V3_W) / (2.0 * V3_W - 2.0 * V3_R)                       # ≈ 2.729
    sigma_nom = 100.0  # MPa
    ref_el = kt_net * sigma_nom * 100.0 / 90.0  # ≈ 303.2 MPa (the same as v3_eval)

    # Material properties
    mat = list(st.materials.values())[0]
    p = estimate_strain_life(mat.Rm, mat.E, "steel")

    # 2. Analytic solution of the Neuber rule for the theoretical elastic stress ref_el
    sig_n_theor, eps_n_theor = neuber_solve(ref_el, p.E, p.K_prime, p.n_prime)
    neuber_prod = sig_n_theor * eps_n_theor
    neuber_ref_prod = (ref_el ** 2) / p.E

    # 3. Analytic solution of Glinka's method
    sig_g_theor, eps_g_theor = glinka_solve(ref_el, p.E, p.K_prime, p.n_prime)
    Wel = (ref_el ** 2) / (2.0 * p.E)
    Wg = (sig_g_theor ** 2) / (2.0 * p.E) + (sig_g_theor / (p.n_prime + 1.0)) * ((sig_g_theor / p.K_prime) ** (1.0 / p.n_prime))

    # 4. Theoretical life by Coffin-Manson from the Neuber strain
    Nf_cm_theor = coffin_manson_life(eps_n_theor, p.E, p.sigma_f_prime, p.b, p.eps_f_prime, p.c)

    # Theoretical life by Morrow with a mean stress of 50 MPa
    Nf_mor_theor = morrow_life(eps_n_theor, 50.0, p.E, p.sigma_f_prime, p.b, p.eps_f_prime, p.c)

    # Theoretical life by SWT with a maximum stress of 300 MPa
    Nf_swt_theor = swt_life(eps_n_theor, 300.0, p.E, p.sigma_f_prime, p.b, p.eps_f_prime, p.c)

    # 5. FEA fatigue calculation
    fr = run_fatigue(st, res)

    # Finding the node at the root of the hole (x = 0, y ≈ R)
    mesh = res.mesh
    n = mesh.nodes
    edge = np.nonzero((np.abs(n[:, 0]) < 1e-6) & (np.abs(n[:, 1] - V3_R) < 1e-3))[0]
    j = edge[np.argmax(res.step(0).S[edge, 0])]

    s_el_fea = float(res.step(0).S[j, 0])
    amp_fea = float(fr.amplitude[j])
    d_fea = float(fr.damage[j])
    sf_fea = float(fr.safety[j])

    # Analytic damage from the exact elastic stress found in the FEM node
    sig_n_fea, eps_n_fea = neuber_solve(amp_fea, p.E, p.K_prime, p.n_prime)
    Nf_fea_nez = _cm_nezavisle(eps_n_fea, p.E, p.sigma_f_prime, p.b, p.eps_f_prime, p.c)
    d_expected = float(U4_CYCLES) / Nf_fea_nez

    # Safety = multiple of the stress s at which D·blocks = 1 (bisection on s, Nf by an independent solution)
    def _d_bloky(sc):
        _, ea = neuber_solve(sc * amp_fea, p.E, p.K_prime, p.n_prime)
        return float(U4_REPEATS) * float(U4_CYCLES) / _cm_nezavisle(
            ea, p.E, p.sigma_f_prime, p.b, p.eps_f_prime, p.c)
    s_lo, s_hi = 1e-3, 100.0
    for _ in range(100):
        s_mid = math.sqrt(s_lo * s_hi)
        if _d_bloky(s_mid) < 1.0:
            s_lo = s_mid
        else:
            s_hi = s_mid
    sf_expected = math.sqrt(s_lo * s_hi)

    # Independent references (bisection in the test, not the program's code)
    Nf_cm_nez = _cm_nezavisle(eps_n_theor, p.E, p.sigma_f_prime, p.b, p.eps_f_prime, p.c)
    Nf_mor_nez = _cm_nezavisle(eps_n_theor, p.E, p.sigma_f_prime, p.b, p.eps_f_prime, p.c, sigma_m=50.0)
    Nf_swt_nez = _swt_nezavisle(eps_n_theor, 300.0, p.E, p.sigma_f_prime, p.b, p.eps_f_prime, p.c)
    sig_g_nez, eps_g_nez = _glinka_nezavisle(ref_el, p.E, p.K_prime, p.n_prime)
    # Ramberg-Osgood: ε = σ/E + (σ/K')^(1/n') in the Neuber solution
    eps_ro = sig_n_theor / p.E + (sig_n_theor / p.K_prime) ** (1.0 / p.n_prime)

    checks = [\
        Check("U4 Neuber soucin sig*eps = sig_el^2/E", neuber_prod, neuber_ref_prod, 1e-5, "MPa"),\
        Check("U4 Glinka energie W = Wel", Wg, Wel, 1e-5, "MPa"),\
        Check("U4 Neuber: eps splnuje Ramberg-Osgood", eps_n_theor, eps_ro, 1e-6, "-"),
        Check("U4 Glinka deformace eps (nezavisle pulenim)", eps_g_theor, eps_g_nez, 1e-5, "-"),
        Check("U4 Coffin-Manson zivotnost Nf (nezavisle pulenim)", Nf_cm_theor, Nf_cm_nez, 1e-5, "cyklu"),
        Check("U4 Morrow zivotnost Nf, sigma_m = 50 MPa (nezavisle)", Nf_mor_theor, Nf_mor_nez, 1e-5, "cyklu"),
        Check("U4 SWT zivotnost Nf, sigma_max = 300 MPa (nezavisle)", Nf_swt_theor, Nf_swt_nez, 1e-5, "cyklu"),
        Check("U4 MKP elasticka spicka sig_el na otvoru vs Peterson (Kt=2.729)", s_el_fea, ref_el, 0.05, "MPa"),\
        Check("U4 MKP poskozeni uzlu D odpovida Coffin-Manson(eps_Neuber)", d_fea, d_expected, 1e-4, "-"),\
        Check("U4 MKP soucinitel bezpecnosti (D*bloky = 1, nezavisle pulenim)", sf_fea, sf_expected, 1e-3, "-"),\
    ]
    return checks


def verify_u4(tmp_path: Path | None = None):
    """Runs the complete verification case U4 and returns the list of the resulting Check checks."""
    import tempfile
    from fea_core.solve import run_study

    if tmp_path is not None:
        wd = Path(tmp_path)
        st, _ = u4_build(wd)
        res = run_study(st, wd / "run")
        return u4_eval(res, st)

    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        st, _ = u4_build(wd)
        res = run_study(st, wd / "run")
        return u4_eval(res, st)


def _get_case():
    from .cases import Case
    return Case(
        "U4",
        "Unava e-N: tyc s vrubem Kt - Neuber / Coffin-Manson",
        u4_build,
        u4_eval,
    )


class _CaseDict(dict):
    def __getitem__(self, key):
        if key == "U4":
            return _get_case()
        return super().__getitem__(key)

    def __contains__(self, key):
        return key == "U4" or super().__contains__(key)

    def __iter__(self):
        return iter(["U4"])

    def keys(self):
        return ["U4"]

    def items(self):
        return [("U4", _get_case())]


FATIGUE_CASES = _CaseDict()
