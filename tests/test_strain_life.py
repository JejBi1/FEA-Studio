"""Unit tests for the module of low-cycle fatigue ε-N (Strain-Life):
- Ramberg-Osgood cyclic curve
- Neuber rule and Glinka's ESED method (convergence and identity)
- Coffin-Manson-Basquin and the mean-stress correction Morrow / SWT
- Bäumel-Seeger Uniform Material Law (UML) for steels and alloys
- Integration with rainflow and the calculation of the damage / safety factors
- Verification case U4: bar with a notch Kt (plate with a hole)
"""
import math
import numpy as np
import pytest

pytest.importorskip("numba")                     # module accel (fatigue kernels)

from fea_core.fatigue.strain_life import (  # noqa: E402
    estimate_strain_life,
    ramberg_osgood_strain,
    ramberg_osgood_stress,
    neuber_solve,
    glinka_solve,
    coffin_manson_life,
    morrow_life,
    swt_life,
    strain_life_cycles,
    damage_en,
    node_result_en,
    RULE_NEUBER,
    RULE_GLINKA,
    MEAN_EN_NONE,
    MEAN_EN_MORROW,
)
from fea_core.fatigue.settings import FatigueSettings
from fea_core.fatigue.history import Channel, LoadHistory
from fea_core.study import Material, Study
from fea_core.fatigue.analysis import en_for_material, run_fatigue


def test_ramberg_osgood_inversion():
    E = 210000.0
    K_prime = 1000.0
    n_prime = 0.15

    for sig in [0.0, 50.0, 200.0, 450.0, 700.0, -300.0]:
        eps = ramberg_osgood_strain(sig, E, K_prime, n_prime)
        sig_rec = ramberg_osgood_stress(eps, E, K_prime, n_prime)
        assert sig_rec == pytest.approx(sig, rel=1e-5, abs=1e-9)


def test_neuber_convergence_and_identity():
    E = 206000.0
    K_prime = 1100.0
    n_prime = 0.14

    for s_el in [10.0, 100.0, 300.0, 500.0, 800.0, 1200.0]:
        sig, eps = neuber_solve(s_el, E, K_prime, n_prime)
        # Neuberova identita: sigma * eps == s_el^2 / E
        neuber_product = sig * eps
        neuber_ref = (s_el ** 2) / E
        assert neuber_product == pytest.approx(neuber_ref, rel=1e-6)

        # Check of the constitutive relation Ramberg-Osgood
        eps_ro = ramberg_osgood_strain(sig, E, K_prime, n_prime)
        assert eps == pytest.approx(eps_ro, rel=1e-6)

        # On yielding the real stress is lower than the elastic one and the strain higher
        if s_el > 400.0:
            assert sig < s_el
            assert eps > s_el / E

    # Negative stresses (symmetry)
    sig_neg, eps_neg = neuber_solve(-500.0, E, K_prime, n_prime)
    sig_pos, eps_pos = neuber_solve(500.0, E, K_prime, n_prime)
    assert sig_neg == pytest.approx(-sig_pos, rel=1e-9)
    assert eps_neg == pytest.approx(-eps_pos, rel=1e-9)


def test_glinka_convergence_and_identity():
    E = 210000.0
    K_prime = 950.0
    n_prime = 0.16

    for s_el in [20.0, 150.0, 350.0, 600.0, 900.0]:
        sig, eps = glinka_solve(s_el, E, K_prime, n_prime)

        # Glinkova identita: Wel == W
        W_el = (s_el ** 2) / (2.0 * E)
        W = (sig ** 2) / (2.0 * E) + (sig / (n_prime + 1.0)) * ((sig / K_prime) ** (1.0 / n_prime))
        assert W == pytest.approx(W_el, rel=1e-6)

        # Check of Ramberg-Osgood
        eps_ro = ramberg_osgood_strain(sig, E, K_prime, n_prime)
        assert eps == pytest.approx(eps_ro, rel=1e-6)


def test_neuber_vs_glinka_relationship():
    """Glinka's method (ESED) counts the strain energy of the plasticity,
    and therefore gives a slightly lower notch stress and strain than the more conservative Neuber rule.
    """
    E = 210000.0
    K_prime = 900.0
    n_prime = 0.15
    s_el = 600.0  # yielded notch

    sig_n, eps_n = neuber_solve(s_el, E, K_prime, n_prime)
    sig_g, eps_g = glinka_solve(s_el, E, K_prime, n_prime)

    assert sig_g < sig_n
    assert eps_g < eps_n


def test_coffin_manson_life():
    E = 210000.0
    sf = 950.0
    b = -0.09
    ef = 0.60
    c = -0.60

    for eps_a in [0.03, 0.01, 0.003, 0.001]:
        Nf = coffin_manson_life(eps_a, E, sf, b, ef, c)
        assert Nf > 0.5
        # Back substitution into the Coffin-Manson equation
        eps_calc = (sf / E) * ((2.0 * Nf) ** b) + ef * ((2.0 * Nf) ** c)
        assert eps_calc == pytest.approx(eps_a, rel=1e-7)

    # Zero or negative amplitude -> infinite life
    assert coffin_manson_life(0.0, E, sf, b, ef, c) == math.inf


def test_morrow_mean_stress():
    E = 210000.0
    sf = 900.0
    b = -0.087
    ef = 0.59
    c = -0.58
    eps_a = 0.003

    Nf_zero = morrow_life(eps_a, 0.0, E, sf, b, ef, c)
    Nf_tens = morrow_life(eps_a, 150.0, E, sf, b, ef, c)
    Nf_comp = morrow_life(eps_a, -150.0, E, sf, b, ef, c)

    # A tensile mean stress reduces the life, a compressive one increases it
    assert Nf_tens < Nf_zero
    assert Nf_comp > Nf_zero

    # Back check of the Morrow relation
    eps_rec = ((sf - 150.0) / E) * ((2.0 * Nf_tens) ** b) + ef * ((2.0 * Nf_tens) ** c)
    assert eps_rec == pytest.approx(eps_a, rel=1e-7)


def test_swt_mean_stress():
    E = 210000.0
    sf = 900.0
    b = -0.087
    ef = 0.59
    c = -0.58
    eps_a = 0.003

    sig_max1 = 400.0
    sig_max2 = 600.0

    Nf1 = swt_life(eps_a, sig_max1, E, sf, b, ef, c)
    Nf2 = swt_life(eps_a, sig_max2, E, sf, b, ef, c)

    # A higher maximum stress reduces the life
    assert Nf2 < Nf1

    # Back substitution into the SWT relation
    p_swt = sig_max1 * eps_a
    p_calc = ((sf ** 2) / E) * ((2.0 * Nf1) ** (2.0 * b)) + sf * ef * ((2.0 * Nf1) ** (b + c))
    assert p_calc == pytest.approx(p_swt, rel=1e-7)

    # A compressive cycle (sigma_max <= 0) does not cause fatigue damage in SWT
    assert swt_life(eps_a, -50.0, E, sf, b, ef, c) == math.inf


def test_baumel_seeger_material_law():
    # 1. Ocel s Rm = 600 MPa, E = 210000 MPa (Rm/E = 0.002857 <= 0.003 -> psi = 1.0)
    p_steel1 = estimate_strain_life(600.0, 210000.0, "steel")
    assert p_steel1.sigma_f_prime == pytest.approx(1.5 * 600.0)
    assert p_steel1.b == pytest.approx(-0.087)
    assert p_steel1.eps_f_prime == pytest.approx(0.59 * 1.0)
    assert p_steel1.c == pytest.approx(-0.58)
    assert p_steel1.K_prime == pytest.approx(1.65 * 600.0)
    assert p_steel1.n_prime == pytest.approx(0.15)

    # 2. High-strength steel Rm = 1000 MPa, E = 200000 MPa (Rm/E = 0.005 > 0.003 -> psi = 1.375 - 125*0.005 = 0.75)
    p_steel2 = estimate_strain_life(1000.0, 200000.0, "steel")
    assert p_steel2.sigma_f_prime == pytest.approx(1500.0)
    assert p_steel2.eps_f_prime == pytest.approx(0.59 * 0.75)

    # 3. Aluminium alloy Rm = 400 MPa, E = 70000 MPa
    p_al = estimate_strain_life(400.0, 70000.0, "aluminum")
    assert p_al.sigma_f_prime == pytest.approx(1.67 * 400.0)
    assert p_al.b == pytest.approx(-0.095)
    assert p_al.eps_f_prime == pytest.approx(0.35)
    assert p_al.c == pytest.approx(-0.69)
    assert p_al.K_prime == pytest.approx(1.61 * 400.0)
    assert p_al.n_prime == pytest.approx(0.11)


def test_galvanized_steel_family_detection():
    # 'galvanized steel' must not be detected as aluminium because of the 'al' in 'galvanized'
    p_galv = estimate_strain_life(500.0, 210000.0, "galvanized steel")
    assert "steel" in p_galv.source.lower()
    assert p_galv.sigma_f_prime == pytest.approx(1.5 * 500.0)

    # Other steel names
    p_struct = estimate_strain_life(450.0, 210000.0, "structural steel")
    assert "steel" in p_struct.source.lower()

    # Aluminium and titanium
    p_dural = estimate_strain_life(400.0, 70000.0, "dural")
    assert "al/ti" in p_dural.source.lower()
    p_ti = estimate_strain_life(900.0, 110000.0, "Ti-6Al-4V")
    assert "al/ti" in p_ti.source.lower()


def test_en_for_material_validations():
    settings = FatigueSettings()

    # 1. A missing Rm must raise a clear error
    mat_no_rm = Material(name="SteelNoRm", E=210000.0, nu=0.3, Rm=None)
    with pytest.raises(ValueError, match="tensile strength Rm is missing"):
        en_for_material(mat_no_rm, settings)

    # 2. A material for 3D printing (E_z is not None) must raise an error that ε-N is not supported for printing
    mat_print = Material(name="PLA_Print", E=3000.0, nu=0.35, Rm=50.0, E_z=2000.0)
    with pytest.raises(ValueError, match="is not supported for 3D printing"):
        en_for_material(mat_print, settings)


def test_sn_mean_correction_validation():
    # Morrow or SWT for the S-N method must raise an error listing the valid choices
    mat = Material(name="Steel", E=210000.0, nu=0.3, Rm=600.0)
    st = Study(name="Test", geometry="dummy.step")
    st.add_material(mat, default=True)

    fs_bad = FatigueSettings(
        history=LoadHistory([Channel("LC1", "sine", 1.0, 100)]),
        method="sn",
        mean_correction="morrow",
    )
    st.fatigue = fs_bad
    with pytest.raises(ValueError, match="Mean stress correction 'morrow' is intended for the ε-N"):
        run_fatigue(st, None)


def test_node_result_en_secant_precision():
    # Check of the accuracy of the log-log secant root finder
    params = estimate_strain_life(600.0, 210000.0, "steel")
    rng = np.array([400.0, 300.0])
    mean = np.array([50.0, 20.0])
    cnt = np.array([1000.0, 5000.0])
    nc = 2
    repeats = 1.0

    d1, sf, amax = node_result_en(
        rng, mean, cnt, nc,
        params.E, params.K_prime, params.n_prime,
        params.sigma_f_prime, params.b, params.eps_f_prime, params.c,
        RULE_NEUBER, MEAN_EN_MORROW, repeats
    )

    # Safety factor sf: repeats * damage_en(sf) == 1.0
    d_at_sf = repeats * damage_en(
        rng, mean, cnt, nc, sf,
        params.E, params.K_prime, params.n_prime,
        params.sigma_f_prime, params.b, params.eps_f_prime, params.c,
        RULE_NEUBER, MEAN_EN_MORROW
    )
    # The result must satisfy D(sf) == 1.0 to a relative tolerance < 1e-4
    assert pytest.approx(d_at_sf, rel=1e-4) == 1.0
    # Expected value of sf approx. 3.766
    assert pytest.approx(sf, rel=1e-4) == 3.7662


def test_strain_life_cycles_wrapper():
    params = estimate_strain_life(600.0, 210000.0, "steel")
    eps_a = 0.004

    n_cm = strain_life_cycles(eps_a, params=params, mean_correction="none")
    n_mor = strain_life_cycles(eps_a, sigma_m=100.0, params=params, mean_correction="morrow")
    n_swt = strain_life_cycles(eps_a, sigma_max=400.0, params=params, mean_correction="swt")

    assert n_cm > 100
    assert n_mor < n_cm
    assert n_swt > 100


def test_damage_en_rainflow():
    params = estimate_strain_life(600.0, 210000.0, "steel")

    # 1000 cycles with a range of 800 MPa (amplitude s_el_a = 400 MPa)
    rng = np.array([800.0])
    mean = np.array([0.0])
    cnt = np.array([1000.0])

    d_neuber = damage_en(
        rng, mean, cnt, 1, 1.0,
        params.E, params.K_prime, params.n_prime,
        params.sigma_f_prime, params.b, params.eps_f_prime, params.c,
        RULE_NEUBER, MEAN_EN_NONE
    )

    d_glinka = damage_en(
        rng, mean, cnt, 1, 1.0,
        params.E, params.K_prime, params.n_prime,
        params.sigma_f_prime, params.b, params.eps_f_prime, params.c,
        RULE_GLINKA, MEAN_EN_NONE
    )

    # Neuber is more conservative (higher damage) than Glinka
    assert d_neuber > 0.0
    assert d_glinka > 0.0
    assert d_neuber > d_glinka


def test_fatigue_settings_en():
    hist = LoadHistory([Channel("LC", "sine", amplitude=1.0, cycles=500)])
    fs = FatigueSettings(
        history=hist,
        method="en",
        en_notch_rule="glinka",
        en_mean_correction="swt",
    )
    d = fs.to_dict()
    assert d["method"] == "en"
    assert d["en_notch_rule"] == "glinka"
    assert d["en_mean_correction"] == "swt"

    fs2 = FatigueSettings.from_dict(d)
    assert fs2.method == "en"
    assert fs2.en_notch_rule == "glinka"
    assert fs2.en_mean_correction == "swt"

    desc = fs.describe()
    assert any("strain-life" in line for line in desc)
    assert any("Glinka" in line for line in desc)
    assert any("SWT" in line for line in desc)


def test_u4_verification_case(tmp_path):
    from verification.fatigue_cases import verify_u4
    checks = verify_u4(tmp_path)
    failed = [f"{c.name}: {c.value} vs {c.ref} (err: {c.error:.3%})" for c in checks if not c.ok]
    assert not failed, "\n".join(failed)
