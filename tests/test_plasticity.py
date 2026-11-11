import numpy as np
import pytest

from fea_core.plasticity import (
    PlasticitySpec,
    eng_to_true_stress_strain,
    make_bilinear_table,
    make_ramberg_osgood_table,
    make_hollomon_table,
    format_plastic_ccx,
    assess_plastic_strain_en1993,
    assess_local_failure_asme,
)
from fea_core.study import Study, Material, BoltSpec, FEATURE_LABELS
from fea_core.combinations import Combination
from fea_core.imprint import Imprint


def test_eng_to_true():
    s_eng = 300.0
    e_eng = 0.05
    E = 200000.0
    s_true, e_pl = eng_to_true_stress_strain(s_eng, e_eng, E)
    assert pytest.approx(s_true, 1e-3) == 300.0 * 1.05
    assert pytest.approx(e_pl, 1e-4) == np.log(1.05) - (315.0 / 200000.0)


def test_bilinear_table():
    # Manual calculation of the hardening modulus H:
    # H = E * Et / (E - Et)
    # For E = 200 000 MPa and Et = 20 000 MPa (Et = E/10):
    # H = 200000 * 20000 / (200000 - 20000) = 4e9 / 180000 = 22222.222 MPa
    # H is exactly by Et/E (11.11 %) higher than the tangent modulus Et!
    E = 200000.0
    Et = 20000.0
    Re = 250.0
    max_eps_pl = 0.01
    H_expected = (E * Et) / (E - Et)
    assert pytest.approx(H_expected / Et - 1.0, 1e-4) == Et / (E - Et)  # 11.11 %

    tbl = make_bilinear_table(Re=Re, Et=Et, max_eps_pl=max_eps_pl, E=E)
    assert len(tbl) == 2
    assert tbl[0] == (250.0, 0.0)
    sigma_max_expected = Re + H_expected * max_eps_pl
    assert pytest.approx(tbl[1][0], 1e-5) == sigma_max_expected
    assert tbl[1][1] == max_eps_pl

    # Et >= E must raise an error
    with pytest.raises(ValueError, match="less than Young's modulus"):
        make_bilinear_table(Re=250.0, Et=210000.0, E=200000.0)


def test_ramberg_osgood_table():
    Re = 235.0
    K_prime = 1000.0
    n_prime = 0.15
    tbl = make_ramberg_osgood_table(Re=Re, K_prime=K_prime, n_prime=n_prime, max_eps_pl=0.10)
    assert len(tbl) >= 10
    # The first point must be exactly (Re, 0.0)
    assert tbl[0][0] == Re
    assert tbl[0][1] == 0.0
    # Monotonic increase of the stress and of the plastic strain
    stresses = [r[0] for r in tbl]
    strains = [r[1] for r in tbl]
    assert all(stresses[i] < stresses[i + 1] for i in range(len(stresses) - 1))
    assert all(strains[i] < strains[i + 1] for i in range(len(strains) - 1))


def test_hollomon_table():
    Re = 235.0
    tbl = make_hollomon_table(Re=Re, Rm=360.0, A=25.0, E=210000.0)
    assert len(tbl) >= 5
    # The first point must be anchored exactly at (Re, 0.0)
    assert tbl[0][0] == Re
    assert tbl[0][1] == 0.0
    # No jump: the stress grows smoothly from Re
    assert tbl[1][0] >= Re
    stresses = [row[0] for row in tbl]
    assert all(stresses[i] <= stresses[i + 1] for i in range(len(stresses) - 1))


def test_format_plastic_ccx():
    # 1. Bilinear model
    spec_bi = PlasticitySpec(model="bilinear", hardening="kinematic", Et=1500.0)
    lines_bi = format_plastic_ccx(spec_bi, Re=300.0, E=200000.0)
    assert lines_bi[0] == "*PLASTIC, HARDENING=KINEMATIC"
    assert "300.0000, 0.000000" in lines_bi[1]

    # 2. Ramberg-Osgood model
    spec_ro = PlasticitySpec(model="ramberg_osgood", hardening="isotropic", K_prime=1100.0, n_prime=0.14)
    lines_ro = format_plastic_ccx(spec_ro, Re=350.0, E=210000.0)
    assert lines_ro[0] == "*PLASTIC, HARDENING=ISOTROPIC"
    assert "350.0000, 0.000000" in lines_ro[1]
    assert len(lines_ro) >= 15

    # 3. Ramberg-Osgood without parameters must raise an error – no silent fallback
    spec_ro_bad = PlasticitySpec(model="ramberg_osgood")
    with pytest.raises(ValueError, match="requires the parameters K_prime and n_prime"):
        format_plastic_ccx(spec_ro_bad, Re=300.0, E=200000.0)

    # 4. Hollomon model
    spec_hol = PlasticitySpec(model="hollomon", hardening="isotropic", Rm=450.0, A=0.20)
    lines_hol = format_plastic_ccx(spec_hol, Re=280.0, E=210000.0)
    assert "280.0000, 0.000000" in lines_hol[1]


def test_assess_en1993():
    peeq_ok = np.array([0.01, 0.03, 0.045])
    res_ok = assess_plastic_strain_en1993(peeq_ok, limit=0.05)
    assert res_ok["ok"] is True
    assert pytest.approx(res_ok["utilization"]) == 0.045 / 0.05

    peeq_fail = np.array([0.01, 0.06])
    res_fail = assess_plastic_strain_en1993(peeq_fail, limit=0.05)
    assert res_fail["ok"] is False
    assert res_fail["utilization"] > 1.0


def test_assess_asme():
    # Uniaxial tension: s1 = 300, s2 = 0, s3 = 0 -> s_hyd = 100, s_vM = 300 -> triaxiality = 1/3
    # Exponent according to ASME VIII-2 §5.3.3: -(m2/(1+m2)) * (1/3 - 1/3) = 0 -> eps_L = eps_Lu
    s = np.zeros((1, 6))
    s[0, 0] = 300.0
    peeq = np.array([0.10])
    res = assess_local_failure_asme(s, peeq, eps_Lu=0.30, material_type="ferritic")
    assert res["ok"] is True
    assert pytest.approx(res["max_utilization"]) == 0.10 / 0.30

    # Triaxial tension (high triaxiality): s_hyd / s_vM > 1/3 -> the limit eps_L decreases below eps_Lu
    s_triax = np.array([[400.0, 300.0, 300.0, 0.0, 0.0, 0.0]])
    res_triax = assess_local_failure_asme(s_triax, peeq, eps_Lu=0.30, material_type="ferritic")
    # The utilisation must be higher than in uniaxial tension
    assert res_triax["max_utilization"] > res["max_utilization"]


def test_study_plastic_is_nonlinear():
    st = Study(name="Test", geometry="dummy.step")
    m_el = Material("Steel", E=210000.0, nu=0.3)
    m_pl = Material("Steel_Plastic", E=210000.0, nu=0.3, plastic=PlasticitySpec().to_dict())
    st.add_material(m_el, default=True)
    st.add_material(m_pl, default=False)
    st.part_materials["Solid1"] = "Steel"

    # 1. An unused plastic material must not switch on is_nonlinear()
    assert st.is_nonlinear() is False

    # 2. Assigned to an active part it switches on is_nonlinear()
    st.part_materials["Solid1"] = "Steel_Plastic"
    assert st.is_nonlinear() is True

    # 3. Assigned only to an excluded part (a bolt) must not switch on is_nonlinear()
    st.part_materials["Solid1"] = "Steel"
    st.part_materials["Bolt1"] = "Steel_Plastic"
    st.bolts.append(BoltSpec(name="B1", cad_parts=["Bolt1"], enabled=True))
    assert st.is_nonlinear() is False


def test_study_features():
    st = Study(name="TestFeatures", geometry="dummy.step")
    m_pl = Material("Steel_Plastic", E=210000.0, nu=0.3, plastic=PlasticitySpec().to_dict())
    st.add_material(m_pl, default=True)
    st.assessment = {"stress_tool": "tresca"}
    st.combinations = [Combination(name="Comb1", factors={"LC1": 1.35})]
    st.imprints = [Imprint(name="Imp1", part="P1", center=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], shape="circle", radius=5.0)]

    d = st.to_dict()
    feats = set(d["features"])
    assert "plastic" in feats
    assert "assessment" in feats
    assert "combinations" in feats
    assert "imprints" in feats

    # Check in FEATURE_LABELS
    for f in ["plastic", "assessment", "combinations", "imprints"]:
        assert f in FEATURE_LABELS

    # Check of load_warnings for an unknown extension
    d["features"].append("future_feature_x")
    st_loaded = Study.from_dict(d)
    assert any("future_feature_x" in w for w in st_loaded.load_warnings)


def test_kinematic_hardening_requires_nlgeom():
    """Kinematic hardening in ccx: NLGEOM is switched on automatically with a warning, or an error with an explicit prohibition."""
    from fea_core.plasticity import PlasticitySpec, check
    from fea_core.study import Study, Material

    # 1. Kinematic without NLGEOM -> NLGEOM is switched on automatically and a warning is added
    st1 = Study(name="t1", geometry="x.step")
    m1 = Material("M", 200000.0, 0.3, 7.85e-9, Re=250.0, Rm=400.0,
                  plastic=PlasticitySpec(model="bilinear", hardening="kinematic", Et=20000.0).to_dict())
    st1.add_material(m1, default=True)
    assert st1.contact_settings.nlgeom is False
    err1, warn1 = check(st1, ["Dil"])
    assert len(err1) == 0
    assert len(warn1) == 1
    assert "NLGEOM option was enabled automatically" in warn1[0]
    assert st1.contact_settings.nlgeom is True

    # 2. Kinematic with NLGEOM already on -> no error, no warning
    st2 = Study(name="t2", geometry="x.step")
    st2.add_material(m1, default=True)
    st2.contact_settings.nlgeom = True
    err2, warn2 = check(st2, ["Dil"])
    assert len(err2) == 0
    assert len(warn2) == 0

    # 3. Isotropic without NLGEOM -> fine, no error or warning
    st3 = Study(name="t3", geometry="x.step")
    m_iso = Material("M_iso", 200000.0, 0.3, 7.85e-9, Re=250.0, Rm=400.0,
                     plastic=PlasticitySpec(model="bilinear", hardening="isotropic", Et=20000.0).to_dict())
    st3.add_material(m_iso, default=True)
    err3, warn3 = check(st3, ["Dil"])
    assert len(err3) == 0
    assert len(warn3) == 0

    # 4. Kinematic with NLGEOM explicitly forbidden -> error in the check
    st4 = Study(name="t4", geometry="x.step")
    st4.add_material(m1, default=True)
    st4.contact_settings.allow_nlgeom = False
    err4, warn4 = check(st4, ["Dil"])
    assert len(err4) == 1
    assert "kinematic hardening is calculated correctly by CalculiX only with NLGEOM" in err4[0]
