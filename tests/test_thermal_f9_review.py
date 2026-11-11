"""Review of F9.6 (heat): convection against the textbooks, one implementation of the calculator, cavity radiation and the contact
conductance in the model and in the .inp, the balance with a cavity, σ units for ccx. Fast tests (without Gmsh and ccx).
"""
from __future__ import annotations

import numpy as np
import pytest

from fea_core.calc.thermal_convection import (
    fluid_properties,
    natural_convection_vertical_plate,
    natural_convection_horizontal_cylinder,
    forced_convection_flat_plate,
    forced_convection_internal_pipe,
    gnielinski_correlation,
    dittus_boelter_correlation,
)
from fea_core.thermal.contact_conductance import suggest_contact_conductance
from fea_core.thermal.model import (
    ContactConductance, CavityRadiation, Radiation, ThermalCase, FixedTemperature, Convection,
    STEFAN_BOLTZMANN, STEFAN_BOLTZMANN_CCX,
)


# ------------------------------------------------------------------ vlastnosti tekutin a korelace
def test_air_properties_incropera_table():
    """Air 1 atm against Incropera table A.4 (300 K and 350 K) and at 20 °C (Pr 0.709, k 0.0257)."""
    p20 = fluid_properties("air", 20.0)
    assert p20["k"] == pytest.approx(0.0257, rel=0.01)          # previously 0.0242 → Pr 0.754
    assert p20["Pr"] == pytest.approx(0.709, rel=0.01)
    p300 = fluid_properties("air", 26.85)
    assert p300["k"] == pytest.approx(0.0263, rel=0.01)
    assert p300["nu"] == pytest.approx(15.89e-6, rel=0.02)     # the table has ρ = 1.161 (ideal gas 1.177)
    assert p300["Pr"] == pytest.approx(0.707, rel=0.01)
    p350 = fluid_properties("air", 76.85)
    assert p350["k"] == pytest.approx(0.030, rel=0.01)
    assert p350["nu"] == pytest.approx(20.92e-6, rel=0.02)
    assert p350["Pr"] == pytest.approx(0.700, rel=0.01)


def test_water_properties_incropera_table():
    """Water against table A.6: k, Pr, ν at 40 °C and the coefficient of volumetric expansion β (20 °C 2.07e-4, 50 °C 4.58e-4)."""
    p = fluid_properties("water", 40.0)
    assert p["k"] == pytest.approx(0.630, rel=0.01)
    assert p["Pr"] == pytest.approx(4.3, rel=0.02)
    assert p["nu"] == pytest.approx(0.657e-6, rel=0.02)
    assert fluid_properties("water", 20.0)["beta"] == pytest.approx(2.07e-4, rel=0.03)
    assert fluid_properties("water", 50.0)["beta"] == pytest.approx(4.58e-4, rel=0.03)
    assert fluid_properties("water", 100.0)["k"] == pytest.approx(0.679, rel=0.01)


def test_vertical_plate_hand_calculation():
    """Vertical plate 300 mm, 60/20 °C, air: h = 5.2–5.3 W/(m²·K); hand Churchill–Chu with the properties from table
    A.4 (T_film 313 K: ν 17.2e-6, α 24.4e-6, k 0.0273, Pr 0.705).
    """
    r = natural_convection_vertical_plate(0.3, 60.0, 20.0)
    assert 5.2 <= r["h"] <= 5.3
    nu_, al_, k_, pr_ = 17.21e-6, 24.4e-6, 0.02727, 0.705
    ra = 9.81 / 313.15 * 40.0 * 0.3 ** 3 / (nu_ * al_)
    nu_ref = (0.825 + 0.387 * ra ** (1 / 6) / (1 + (0.492 / pr_) ** (9 / 16)) ** (8 / 27)) ** 2
    assert r["Ra"] == pytest.approx(ra, rel=0.05)
    assert r["h"] == pytest.approx(nu_ref * k_ / 0.3, rel=0.02)
    f = (1 + (0.492 / r["Pr"]) ** (9 / 16)) ** (8 / 27)          # and the formula from the own Ra, Pr of the result
    assert r["Nu"] == pytest.approx((0.825 + 0.387 * r["Ra"] ** (1 / 6) / f) ** 2, rel=1e-12)


def test_laminar_flat_plate_and_pipe_correlations():
    """Nu = 0.664·Re^½·Pr^⅓ (laminar plate), Dittus–Boelter and Gnielinski against the formulas of the textbook."""
    r = forced_convection_flat_plate(0.5, 2.0, 50.0, 20.0)
    assert r["Nu"] == pytest.approx(0.664 * r["Re"] ** 0.5 * r["Pr"] ** (1 / 3), rel=1e-12)
    assert r["h"] == pytest.approx(7.78, rel=0.02)               # hand calculation: T_film 308 K (ν 16.7e-6, k 0.0269, Pr 0.706)
    assert gnielinski_correlation(1e4, 0.7) == pytest.approx(29.8, rel=0.01)      # Re = 1e4, Pr = 0,7
    assert dittus_boelter_correlation(1e4, 0.7, heating=True) == pytest.approx(0.023 * 1e4 ** 0.8 * 0.7 ** 0.4, rel=1e-12)
    w = forced_convection_internal_pipe(0.02, 1.0, 45.0, 40.0, "water", correlation="dittus_boelter")
    assert w["h"] == pytest.approx(5100.0, rel=0.03)             # Re 3,2e4, Pr 4,1: Nu 162


def test_convection_registry_uses_single_implementation():
    """KINDS / MCP / GUI go through a thin adapter over thermal_convection (no second table of properties)."""
    from fea_core.calc import call
    from fea_core.calc import convection as adapter
    assert adapter.fluid_properties is fluid_properties
    r = call("convection", "vertical_plate", {"L_mm": 300, "Ts": 60, "Tinf": 20})
    assert r["h"] == pytest.approx(natural_convection_vertical_plate(0.3, 60.0, 20.0)["h"], abs=1e-3)
    assert r["k_air"] == pytest.approx(0.0273, abs=2e-4)
    cyl = call("convection", "horizontal_cylinder", {"D_mm": 50, "Ts": 80, "Tinf": 20})
    assert cyl["h"] == pytest.approx(natural_convection_horizontal_cylinder(0.05, 80.0, 20.0)["h"], abs=1e-3)


def test_horizontal_plate_adapter_keys_and_orientation():
    from fea_core.calc import call
    base = {"L_mm": 200, "W_mm": 100, "Ts": 60, "Tinf": 20}
    up = call("convection", "horizontal_plate", {**base, "orientation": "upper_hot"})
    dn = call("convection", "horizontal_plate", {**base, "orientation": "lower_hot"})
    assert up["L_char_mm"] == pytest.approx(33.3, abs=0.1) and up["orientation"] == "upper_hot"
    assert up["h"] > dn["h"] > 0
    assert call("convection", "horizontal_plate", {**base, "Ts": 20})["h"] == 0.0


# ------------------------------------------------------------------ contact conductance
def test_suggest_contact_conductance_czech_words():
    """A material is recognised by whole words (also with diacritics), not by substrings."""
    assert suggest_contact_conductance("Hliník", "Hliník", roughness_um=2.0) == 4000.0
    assert suggest_contact_conductance("Hliník", "Ocel", roughness_um=2.0) == 2500.0
    assert suggest_contact_conductance("Měď", "Ocel", roughness_um=2.0) == 10000.0
    assert suggest_contact_conductance("medium alloy steel", "galvanized", roughness_um=2.0) == 2000.0
    assert suggest_contact_conductance("Ocel", "Ocel", roughness_um=0.0) > 0       # roughness 0 does not divide by zero


def test_contact_conductance_between_selections(tmp_path):
    """ContactConductance with two selections (without a record in contacts_report) is written as *GAP CONDUCTANCE."""
    from tests.test_thermal import bar_mesh, study
    from fea_core.thermal.inp import write_model, thermal_contact_conductances
    m, st = bar_mesh(parts=2), study(parts=2)
    st.thermal.cases = [ThermalCase("A", [ContactConductance("x0", master="x100", conductance=2500.0)])]
    assert thermal_contact_conductances(st, m) == [("x0", "x100", 2500.0)]
    imap = write_model(st, m, tmp_path / "model.inc")
    txt = (tmp_path / "model.inc").read_text(encoding="utf-8")
    assert "*GAP CONDUCTANCE\n2.5\n" in txt                       # 2500 W/(m²·K) = 2,5 mW/(mm²·K)
    sl, ms = imap.surfaces["x0"], imap.surfaces["x100"]            # generated names (S_SEL<i>), not the names of the selections
    assert sl.startswith("S_SEL") and ms.startswith("S_SEL")
    assert f"*CONTACT PAIR, INTERACTION=INT_TC0, TYPE=SURFACE TO SURFACE\n{sl}, {ms}\n" in txt


def test_check_contact_conductance_selections():
    from fea_core.thermal.check import check
    from tests.test_thermal import study
    st = study(parts=2)
    st.thermal.cases = [ThermalCase("A", [FixedTemperature("x0", 100.0), Convection("x100", 10.0, 20.0),
                                          ContactConductance("y0", master="x100", conductance=100.0)])]
    errors, _, used = check(st, ["Dil1", "Dil2"])
    assert {"x0", "x100", "y0"} <= used and not errors
    st.thermal.cases[0].loads[2].conductance = 0.0
    errors, _, _ = check(st, ["Dil1", "Dil2"])
    assert any("contact conductance" in e for e in errors)


def test_gap_conductance_warning_is_ignored_only_with_conductances():
    from fea_core.thermal import run as thermal_run
    assert "no conductivity" not in thermal_run._IGNORED
    assert "no conductivity" in thermal_run._IGNORED_GAP


# ------------------------------------------------------------------ radiation, cavities
def test_stefan_boltzmann_ccx_units(tmp_path):
    """σ for ccx = σ·1e-3 (mW/(mm²·K⁴)) and exactly that is written into model.inc."""
    from tests.test_thermal import bar_mesh, study
    from fea_core.thermal.inp import write_model
    assert STEFAN_BOLTZMANN_CCX == pytest.approx(5.670374419e-11, rel=1e-12)
    assert STEFAN_BOLTZMANN_CCX * 1e3 == pytest.approx(STEFAN_BOLTZMANN, rel=1e-12)
    write_model(study(), bar_mesh(), tmp_path / "model.inc")
    txt = (tmp_path / "model.inc").read_text(encoding="utf-8")
    assert "*PHYSICAL CONSTANTS, ABSOLUTE ZERO=-273.15, STEFAN BOLTZMANN=5.670374419e-11" in txt


def test_cavity_with_selection_list_and_json_roundtrip(tmp_path):
    """A cavity from several selections: the faces of all selections under one code; model ↔ JSON without loss."""
    from tests.test_thermal import bar_mesh, study
    from fea_core.thermal.inp import write_model, write_case
    m, st = bar_mesh(), study()
    cav = CavityRadiation(selection=["x0", "x100"], emissivity=0.7, cavity_name="Dutina A", closed=True)
    st.thermal.cases = [ThermalCase("A", [FixedTemperature("y0", 50.0), cav])]
    mp = write_model(st, m, tmp_path / "model.inc")
    write_case(st, m, mp, st.thermal.cases[0], tmp_path / "c.inp", "c")
    txt = (tmp_path / "c.inp").read_text(encoding="utf-8")
    assert txt.count("*RADIATE, CAVITY=C01") == 1
    n_faces = sum(len(m.element_faces(s)[0]) for s in ("x0", "x100"))
    assert txt.count("CR, -300, 0.7") == n_faces
    d = st.thermal.cases[0].to_dict()
    case2 = ThermalCase.from_dict(d)
    assert case2.to_dict() == d
    assert isinstance(case2.loads[1], CavityRadiation) and case2.loads[1].selections == ["x0", "x100"]
    cc = ContactConductance("a", master="b", conductance=300.0)
    rad = Radiation("r", 0.9, 20.0, cavity=True, cavity_name="X", closed=True)
    d = ThermalCase("B", [cc, rad]).to_dict()
    assert ThermalCase.from_dict(d).to_dict() == d


# ------------------------------------------------------------------ bilance s dutinou
def _cavity_case(closed: bool):
    from tests.test_thermal import bar_mesh
    m = bar_mesh()
    case = ThermalCase("A", [FixedTemperature("x0", 100.0), Convection("y0", 10.0, 20.0),
                             Radiation("x100", 0.8, 20.0, cavity=True, cavity_name="OTV", closed=closed)])
    return m, case, 100.0 - 0.3 * m.nodes[:, 0]


def _synthetic_rfl(m, case, T, cav):
    """RFL at the free nodes = external heat (convection + cavity), at the nodes x0 the reaction so that Σ RFL = 0."""
    from fea_core.thermal.loads import bc_nodal
    ext = bc_nodal(m, case.loads[1], T) + cav
    rfl = ext.copy()
    fix = np.unique(m.surfaces["x0"])
    rfl[fix] -= ext.sum() / len(fix)
    return rfl


def test_heat_balance_with_open_cavity_not_skipped():
    """The heat of an open cavity (exchange with the surroundings) is determined from the RFL and the balance of the other nodes is still checked."""
    from fea_core.thermal.run import heat_flows
    from tests.test_thermal import study
    m, case, T = _cavity_case(closed=False)
    idx = np.unique(m.surfaces["x100"])
    cav = np.zeros(len(m.nodes))
    cav[idx] = -0.5 / len(idx)                                    # the cavity removes 0.5 W through the opening into the surroundings
    rfl = _synthetic_rfl(m, case, T, cav)
    heat, bal = heat_flows(study(), m, case, T, rfl)
    q = {h["kind"]: h["Q_W"] for h in heat}
    assert q["cavity_radiation"] == pytest.approx(-0.5, rel=1e-9)
    assert bal["imbalance"] < 1e-12 and "warnings" not in bal
    fix = np.unique(m.surfaces["x0"])
    node = np.setdiff1d(np.unique(m.surfaces["y0"]), np.concatenate([idx, fix]))[0]
    bad = rfl.copy()                                              # error in a convection node outside the cavity
    bad[node] += 0.1
    bad[fix] -= 0.1 / len(fix)
    assert heat_flows(study(), m, case, T, bad)[1]["imbalance"] > 1e-3


def test_closed_cavity_net_heat_warning():
    """A closed cavity only redistributes the heat between the faces: the net flux ≈ 0; otherwise a warning (view factors)."""
    from fea_core.thermal.run import heat_flows
    from tests.test_thermal import study
    m, case, T = _cavity_case(closed=True)
    idx = np.unique(m.surfaces["x100"])
    h = len(idx) // 2
    cav = np.zeros(len(m.nodes))
    cav[idx[:h]] = 0.5 / h
    cav[idx[h:]] = -0.5 / (len(idx) - h)
    _, bal = heat_flows(study(), m, case, T, _synthetic_rfl(m, case, T, cav))
    assert "warnings" not in bal and bal["imbalance"] < 1e-12
    cav[idx] = -0.5 / len(idx)                                    # the cavity "loses" heat – it is not closed
    _, bal = heat_flows(study(), m, case, T, _synthetic_rfl(m, case, T, cav))
    assert any("Closed cavity 'OTV'" in w for w in bal["warnings"])
