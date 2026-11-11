import numpy as np
import pytest

from fea_core.combinations import (
    Combination,
    generate_en1990_combinations,
    combine_results,
    compute_envelope,
    load_standards_data,
)
from fea_core.results import Results, StepResult
from fea_core.mesh import Mesh


def test_load_standards_data():
    data = load_standards_data()
    assert "standards" in data
    assert "CSN_EN_1990" in data["standards"]
    std = data["standards"]["CSN_EN_1990"]
    assert "psi" in std
    assert std["psi"]["cat_A"]["psi_0"] == 0.7
    assert std["psi"]["snow_cz_low"]["psi_0"] == 0.5
    assert std["psi"]["wind"]["psi_0"] == 0.6


def test_generate_en1990_str_geo_6_10():
    cases = [
        {"name": "G1", "kind": "G", "favorable": False},
        {"name": "Q1_uzytne", "kind": "Q", "category": "cat_A", "favorable": False},
        {"name": "Q2_snih", "kind": "Q", "category": "snow_cz_low", "favorable": False},
        {"name": "Q3_vitr", "kind": "Q", "category": "wind", "favorable": False},
    ]
    combs = generate_en1990_combinations(cases, situation="STR_GEO_6_10")
    assert len(combs) == 3  # 3 variable loads → 3 combinations (each once as the leading one)

    # Combination 1: Q1 leading (1.5), the others with psi_0 (1.5 * psi_0)
    c1 = combs[0]
    assert c1.factors["G1"] == 1.35
    assert c1.factors["Q1_uzytne"] == 1.50
    assert c1.factors["Q2_snih"] == pytest.approx(1.50 * 0.5)   # 0.75
    assert c1.factors["Q3_vitr"] == pytest.approx(1.50 * 0.6)   # 0.90

    # Combination 2: Q2 leading
    c2 = combs[1]
    assert c2.factors["G1"] == 1.35
    assert c2.factors["Q1_uzytne"] == pytest.approx(1.50 * 0.7) # 1.05
    assert c2.factors["Q2_snih"] == 1.50
    assert c2.factors["Q3_vitr"] == pytest.approx(1.50 * 0.6)   # 0.90


def test_generate_en1990_6_10a_6_10b():
    cases = [
        {"name": "G1", "kind": "G", "favorable": False},
        {"name": "Q1", "kind": "Q", "category": "cat_A", "favorable": False},
    ]
    combs = generate_en1990_combinations(cases, situation="STR_GEO_6_10a_6_10b")
    # 6.10a (all with psi_0) + 6.10b (xi * gamma_G + Q1 leading)
    assert len(combs) == 2
    c_a, c_b = combs[0], combs[1]
    assert c_a.factors["G1"] == 1.35
    assert c_a.factors["Q1"] == pytest.approx(1.50 * 0.7)  # 1.05

    assert c_b.factors["G1"] == pytest.approx(0.85 * 1.35) # 1.1475
    assert c_b.factors["Q1"] == 1.50


def test_generate_en1990_sls():
    cases = [
        {"name": "G1", "kind": "G"},
        {"name": "Q1", "kind": "Q", "category": "cat_A"},
        {"name": "Q2", "kind": "Q", "category": "wind"},
    ]
    # SLS characteristic: G + Q1 + psi0*Q2
    c_char = generate_en1990_combinations(cases, situation="SLS_CHAR")
    assert len(c_char) == 2
    assert c_char[0].factors["G1"] == 1.0
    assert c_char[0].factors["Q1"] == 1.0
    assert c_char[0].factors["Q2"] == pytest.approx(0.6)

    # SLS quasi-permanent: G + sum(psi2 * Qi)
    c_quasi = generate_en1990_combinations(cases, situation="SLS_QUASI")
    assert len(c_quasi) == 1
    assert c_quasi[0].factors["G1"] == 1.0
    assert c_quasi[0].factors["Q1"] == pytest.approx(0.3)  # cat_A psi_2 = 0.3
    assert c_quasi[0].factors["Q2"] == pytest.approx(0.0)  # wind psi_2 = 0.0


def test_linear_superposition_and_von_mises():
    """Check that the stresses are added as tensors and the von Mises is computed from the sum of the tensors."""
    class DummyStudy:
        def is_nonlinear(self):
            return False

    mesh = Mesh(
        nodes=np.zeros((2, 3)),
        tets=np.zeros((0, 10), dtype=int),
        tet_part=np.zeros((0,), dtype=int),
        part_names=["p"],
    )

    # State 1: tension in X: SXX = 100 MPa, the rest 0
    # von Mises = 100 MPa
    s1 = np.zeros((2, 6))
    s1[:, 0] = 100.0
    u1 = np.ones((2, 3)) * 1.0
    step1 = StepResult(name="LC1", U=u1, S=s1, reactions={"fixed": np.array([10.0, 0.0, 0.0])})

    # State 2: tension in Y: SYY = 100 MPa, the rest 0
    # von Mises = 100 MPa
    s2 = np.zeros((2, 6))
    s2[:, 1] = 100.0
    u2 = np.ones((2, 3)) * 2.0
    step2 = StepResult(name="LC2", U=u2, S=s2, reactions={"fixed": np.array([0.0, 20.0, 0.0])})

    results = Results(mesh=mesh, steps=[step1, step2])

    comb = Combination(name="Komb1", factors={"LC1": 1.0, "LC2": 1.0})
    comb_res = combine_results(DummyStudy(), results, comb)

    # Posuvy: 1.0 + 2.0 = 3.0
    assert np.allclose(comb_res.U, 3.0)

    # Reakce: [10, 20, 0]
    assert np.allclose(comb_res.reactions["fixed"], [10.0, 20.0, 0.0])

    # Stress tensor: SXX = 100, SYY = 100
    # Equal biaxial tension: von Mises = sqrt(100^2 - 100*100 + 100^2) = 100 MPa!
    # If the von Mises were wrongly added: 100 + 100 = 200 MPa (wrong!).
    assert np.allclose(comb_res.S[:, 0], 100.0)
    assert np.allclose(comb_res.S[:, 1], 100.0)
    assert np.allclose(comb_res.mises, 100.0)


def test_envelope_computation():
    class DummyStudy:
        combinations = []
        def is_nonlinear(self):
            return False

    mesh = Mesh(
        nodes=np.zeros((3, 3)),
        tets=np.zeros((0, 10), dtype=int),
        tet_part=np.zeros((0,), dtype=int),
        part_names=["p"],
    )

    # Three nodes
    # State 1: von Mises [50, 150, 100]
    s1 = np.zeros((3, 6))
    s1[:, 0] = [50.0, 150.0, 100.0]
    step1 = StepResult(name="C1", U=np.zeros((3, 3)), S=s1)

    # Stav 2: von Mises [120, 80, 110]
    s2 = np.zeros((3, 6))
    s2[:, 0] = [120.0, 80.0, 110.0]
    step2 = StepResult(name="C2", U=np.zeros((3, 3)), S=s2)

    results = Results(mesh=mesh, steps=[step1, step2])

    env_comb = Combination(
        name="Obalka_max",
        envelope=True,
        envelope_type="max",
        envelope_combinations=["C1", "C2"],
        envelope_field="von_mises",
    )

    env_res = compute_envelope(DummyStudy(), results, env_comb)

    # Node 0: max(50, 120) = 120 (from C2)
    # Node 1: max(150, 80) = 150 (from C1)
    # Node 2: max(100, 110) = 110 (from C2)
    assert np.allclose(env_res.values, [120.0, 150.0, 110.0])
    assert env_res.governing_combinations == ["C2", "C1", "C2"]
    assert env_res.governing_summary == {"C1": 1, "C2": 2}
    assert env_res.max_val == 150.0
    assert env_res.min_val == 110.0
