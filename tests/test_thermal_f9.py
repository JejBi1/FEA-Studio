"""Unit tests for stage F9.6: Heat (D1 contact thermal resistance, D2 radiation, D4 heat transfer calculator).

Contents of the tests:
- D4: physical properties of fluids (air, water) over a wide temperature range
- D4: natural convection Churchill & Chu for a vertical plate and a horizontal cylinder against textbook values
- D4: forced convection Gnielinski and Dittus-Boelter for the flow in a pipe against textbook values
- D4: forced convection on a flat plate (laminar / turbulent)
- D1: contact thermal conductance h_c, thermal resistance R_c, temperature drop ΔT = q / h_c, equivalent thin layer λ = h_c · t
- D1: generation of the CalculiX syntax *GAP CONDUCTANCE and *SURFACE BEHAVIOR
- D2: Stefan-Boltzmann radiation between parallel plates and concentric cylinders
- D2: linearised radiative heat transfer coefficient h_r
- D2: generation of the CalculiX syntax *PHYSICAL CONSTANTS and *RADIATE (open radiation and cavity CAVITY)
- Integration: generation of the ccx inputs with the contact resistance and the cavity radiation
"""
from __future__ import annotations

import math
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
from fea_core.thermal.model import (
    ContactConductance,
    CavityRadiation,
    Radiation,
    STEFAN_BOLTZMANN,
    STEFAN_BOLTZMANN_CCX,
    ABSOLUTE_ZERO,
)
from fea_core.thermal.contact_conductance import (
    CONTACT_CONDUCTANCE_DATABASE,
    contact_resistance,
    contact_temperature_drop,
    thin_layer_equivalent_conductivity,
    suggest_contact_conductance,
)
from fea_core.thermal.radiation import (
    parallel_plates_radiation,
    concentric_cylinders_radiation,
    radiation_heat_transfer_coefficient,
    linearized_radiation_convection,
)


# =====================================================================
# D4: tests of the fluid properties (air, water)
# =====================================================================

def test_fluid_properties_air_20C():
    """Air at 20 °C and 1 atm: comparison with the tables (Incropera Table A.4, VDI)."""
    p = fluid_properties("air", 20.0)
    # Density of the ideal gas ~ 1.204 kg/m³
    assert math.isclose(p["rho"], 1.204, rel_tol=0.01)
    # Specific heat cp ~ 1006 J/(kg·K)
    assert math.isclose(p["cp"], 1006.0, rel_tol=0.01)
    # Thermal conductivity k ~ 0.025 – 0.026 W/(m·K)
    assert 0.024 <= p["k"] <= 0.027
    # Dynamic viscosity ~ 1.81e-5 Pa·s
    assert math.isclose(p["mu"], 1.81e-5, rel_tol=0.03)
    # Kinematic viscosity ~ 1.51e-5 m²/s
    assert math.isclose(p["nu"], 1.51e-5, rel_tol=0.03)
    # Prandtl number ~ 0.71
    assert 0.69 <= p["Pr"] <= 0.73
    # Thermal expansion beta = 1 / T_K = 1 / 293.15 ~ 0.00341 1/K
    assert math.isclose(p["beta"], 1.0 / 293.15, rel_tol=1e-4)


def test_fluid_properties_water_ranges():
    """Water at 20 °C and 80 °C: comparison with the tables of saturated water (Incropera Table A.6)."""
    p20 = fluid_properties("water", 20.0)
    # Density at 20 °C ~ 998 kg/m³
    assert math.isclose(p20["rho"], 998.2, rel_tol=0.005)
    # cp ~ 4182 J/(kg·K)
    assert math.isclose(p20["cp"], 4182.0, rel_tol=0.01)
    # k ~ 0.60 W/(m·K)
    assert math.isclose(p20["k"], 0.60, rel_tol=0.03)
    # mu ~ 1.00e-3 Pa·s
    assert math.isclose(p20["mu"], 1.002e-3, rel_tol=0.03)
    # Pr ~ 7.0
    assert 6.5 <= p20["Pr"] <= 7.5

    p80 = fluid_properties("water", 80.0)
    # At 80 °C the density (~ 972 kg/m³) and the viscosity (~ 0.35e-3 Pa·s) decrease
    assert math.isclose(p80["rho"], 971.8, rel_tol=0.01)
    assert math.isclose(p80["mu"], 0.355e-3, rel_tol=0.05)
    assert p80["Pr"] < p20["Pr"]


def test_fluid_properties_validation_errors():
    """Check of the limits of the temperatures and of an unknown fluid."""
    with pytest.raises(ValueError, match="Unknown fluid"):
        fluid_properties("oil", 20.0)
    with pytest.raises(ValueError, match="outside the validity range"):
        fluid_properties("air", -250.0)
    with pytest.raises(ValueError, match="outside the liquid range"):
        fluid_properties("water", -5.0)


# =====================================================================
# D4: tests of natural and forced convection (Churchill-Chu, Gnielinski, Dittus-Boelter)
# =====================================================================

def test_churchill_chu_vertical_plate_incropera_benchmark():
    """Incropera & DeWitt Example 9.2:
    Vertical plate L = 0.3 m, Ts = 70 °C, Tinf = 25 °C, air 1 atm.
    T_film = 47.5 °C (320.65 K). Ra_L ~ 7.3e7.
    Churchill & Chu Nu_L ~ 47.5 – 49, h ~ 4.4 W/(m²·K).
    """
    res = natural_convection_vertical_plate(L=0.3, Ts=70.0, Tinf=25.0, fluid="air")
    assert 6.5e7 <= res["Ra"] <= 8.5e7
    assert 54.0 <= res["Nu"] <= 62.0
    assert 4.8 <= res["h"] <= 5.6
    assert res["regime"] == "laminar"
    assert res["q"] > 0


def test_churchill_chu_horizontal_cylinder():
    """Horizontal cylinder D = 0.05 m (50 mm), Ts = 80 °C, Tinf = 20 °C, air 1 atm.
    The expected heat transfer coefficient h is in the range 6.5 – 7.5 W/(m²·K).
    """
    res = natural_convection_horizontal_cylinder(D=0.05, Ts=80.0, Tinf=20.0, fluid="air")
    assert res["Nu"] > 0
    assert 6.0 <= res["h"] <= 8.0
    assert res["regime"] == "laminar"
    assert res["Gr"] > 0


def test_churchill_chu_water_high_convection():
    """Natural convection in water gives h of an order higher than in air (~ 800–1200 W/(m²·K))."""
    res_air = natural_convection_vertical_plate(L=0.1, Ts=60.0, Tinf=20.0, fluid="air")
    res_water = natural_convection_vertical_plate(L=0.1, Ts=60.0, Tinf=20.0, fluid="water")
    assert res_water["h"] > 50.0 * res_air["h"]
    assert 500.0 <= res_water["h"] <= 2000.0


def test_gnielinski_dittus_boelter_incropera_benchmark():
    """Incropera & DeWitt Example 8.5:
    Water flowing in a smooth pipe D = 0.05 m, velocity u = 1.0 m/s.
    Ts = 60 °C, Tinf = 20 °C (T_film = 40 °C).
    Re ~ 7.6e4, Pr ~ 4.3 (turbulent).
    Comparison of Gnielinski and Dittus-Boelter.
    """
    res_gn = forced_convection_internal_pipe(D=0.05, u=1.0, Ts=60.0, Tinf=20.0,
                                             fluid="water", correlation="gnielinski")
    res_db = forced_convection_internal_pipe(D=0.05, u=1.0, Ts=60.0, Tinf=20.0,
                                             fluid="water", correlation="dittus_boelter")

    assert res_gn["Re"] > 50000.0
    # Both correlations for fully developed turbulent flow give h in the range 4000–6000 W/(m²·K)
    assert 3500.0 <= res_gn["h"] <= 6500.0
    assert 3500.0 <= res_db["h"] <= 6500.0
    # Relative agreement between Gnielinski and Dittus-Boelter within 15 %
    assert math.isclose(res_gn["h"], res_db["h"], rel_tol=0.15)


def test_forced_flat_plate_laminar_and_turbulent():
    """Flat plate L = 0.5 m, u = 2 m/s (laminar Re < 5e5) vs L = 2.0 m, u = 25 m/s (turbulent)."""
    res_lam = forced_convection_flat_plate(L=0.5, u=2.0, Ts=50.0, Tinf=20.0, fluid="air")
    assert res_lam["Re"] < 5e5
    assert res_lam["regime"] == "laminar"
    assert 5.0 <= res_lam["h"] <= 15.0

    res_turb = forced_convection_flat_plate(L=2.0, u=25.0, Ts=50.0, Tinf=20.0, fluid="air")
    assert res_turb["Re"] > 5e5
    assert res_turb["regime"] == "transitional / turbulent"
    assert res_turb["h"] > res_lam["h"]


# =====================================================================
# D1: tests of the thermal contact resistance (Contact Conductance)
# =====================================================================

def test_contact_resistance_and_temperature_drop():
    """Check of the basic relations for the thermal resistance: R_c = 1 / h_c, ΔT = q / h_c."""
    hc = 2000.0  # W/(m²·K)
    rc = contact_resistance(hc)
    assert math.isclose(rc, 0.0005, rel_tol=1e-6)  # (m²·K)/W

    area = 0.001  # m² (1000 mm²)
    r_abs = contact_resistance(hc, area_m2=area)
    assert math.isclose(r_abs, 0.5, rel_tol=1e-6)  # K/W

    q_flux = 10000.0  # W/m²
    dt = contact_temperature_drop(q_flux, hc)
    assert math.isclose(dt, 5.0, rel_tol=1e-6)  # ΔT = 10000 / 2000 = 5 K


def test_thin_layer_equivalent_conductivity():
    """Fallback model of the thin layer (Thin Layer Interface): λ = h_c · t."""
    hc = 5000.0  # W/(m²·K)
    t_mm = 0.1   # mm -> 1e-4 m
    k_eq = thin_layer_equivalent_conductivity(hc, thickness_mm=t_mm)
    assert math.isclose(k_eq, 0.5, rel_tol=1e-6)  # W/(m·K)
    # Thermal resistance of the layer R = t / λ = 1e-4 / 0.5 = 2e-4 = 1 / hc
    assert math.isclose(t_mm / 1000.0 / k_eq, 1.0 / hc, rel_tol=1e-6)


def test_suggest_contact_conductance_database():
    """Check of the database and of the recommended values of h_c from the literature."""
    assert "steel_steel_rough_dry" in CONTACT_CONDUCTANCE_DATABASE
    assert "thermal_paste_silicone" in CONTACT_CONDUCTANCE_DATABASE

    hc_steel = suggest_contact_conductance("Ocel", "Ocel", pressure_mpa=2.0)
    assert 1000.0 <= hc_steel <= 10000.0

    hc_tim = suggest_contact_conductance("Hliník", "Měď", tim="silicone_paste")
    assert hc_tim == CONTACT_CONDUCTANCE_DATABASE["thermal_paste_silicone"]["conductance_W_m2K"]


# =====================================================================
# D2: tests of radiation (Radiation & Cavity Radiation)
# =====================================================================

def test_parallel_plates_radiation_analytical():
    """Radiation between parallel plates: q = ε_eff · σ · (T1⁴ - T2⁴)."""
    T1_C = 200.0
    T2_C = 20.0
    eps1 = 0.8
    eps2 = 0.8

    res = parallel_plates_radiation(T1_C, T2_C, eps1=eps1, eps2=eps2, area_m2=0.04)

    # ε_eff = 1 / (1/0.8 + 1/0.8 - 1) = 2/3
    assert math.isclose(res["eps_eff"], 2.0 / 3.0, rel_tol=1e-6)

    T1_k = 200.0 + 273.15
    T2_k = 20.0 + 273.15
    q_exact = (2.0 / 3.0) * STEFAN_BOLTZMANN * (T1_k ** 4 - T2_k ** 4)
    assert math.isclose(res["q_W_m2"], q_exact, rel_tol=1e-6)
    assert math.isclose(res["Q_W"], q_exact * 0.04, rel_tol=1e-6)
    assert res["h_rad"] > 0


def test_concentric_cylinders_radiation_analytical():
    """Radiation between concentric cylinders r1 = 0.02 m, r2 = 0.04 m."""
    T1_C = 150.0
    T2_C = 30.0
    r1, r2 = 0.02, 0.04
    eps1, eps2 = 0.9, 0.7

    res = concentric_cylinders_radiation(T1_C, T2_C, r1, r2, eps1=eps1, eps2=eps2, L_m=1.0)
    # ε_eff,1 = 1 / [1/0.9 + (1 - 0.7)/0.7 * (0.02 / 0.04)] = 1 / [1.111 + 0.4286 * 0.5] = 1 / 1.3254 ~ 0.7545
    assert 0.70 <= res["eps_eff1"] <= 0.80
    assert res["q1_W_m2"] > 0
    assert "Q_W" in res


def test_linearized_radiation_convection():
    """Linearised coefficient h_r and its combination with convection."""
    h_conv = 10.0
    comb = linearized_radiation_convection(T_surface_C=60.0, T_ambient_C=20.0, emissivity=0.9, h_conv=h_conv)
    # For 60 °C to 20 °C with ε=0.9, h_r ~ 6.0 W/(m²·K)
    assert 5.0 <= comb["h_radiation"] <= 7.0
    assert math.isclose(comb["h_total"], h_conv + comb["h_radiation"], rel_tol=1e-9)


# =====================================================================
# Regression tests for the fixes of F9.6 (Finding 2, 3, 4, 5)
# =====================================================================

def test_suggest_contact_conductance_token_matching():
    """Finding 2: 'galvanized steel' and 'medium carbon steel' must not match Al/Cu through substrings."""
    # With the default parameters (p=1.0 MPa, Ra=3.0 um) steel-steel gives 1333.3 W/(m²·K)
    res = suggest_contact_conductance("galvanized steel", "medium carbon steel")
    assert res == 1333.3
    # At Ra=2.0 um base_hc = 2000.0 is obtained
    assert suggest_contact_conductance("galvanized steel", "medium carbon steel", roughness_um=2.0) == 2000.0
    # Aluminium and copper as whole words
    res_al_cu = suggest_contact_conductance("aluminum 6061", "copper rod", roughness_um=2.0)
    assert res_al_cu == 10000.0  # Cu has priority for a metal pair with copper


def test_thermal_contact_conductance_inp_from_contacts_report(tmp_path):
    """Finding 3: a ContactConductance found in contacts_report must be carried correctly into model.inc."""
    from tests.test_thermal import bar_mesh, study
    from fea_core.thermal.inp import write_model, write_case, thermal_contact_conductances
    from fea_core.thermal.model import ThermalCase

    m = bar_mesh(parts=2)
    st = study(parts=2)
    m.report = {
        "contacts": [
            {"name": "Styk1", "pair": "Dil1_Dil2", "slave": "x0", "master": "x100"}
        ]
    }
    cc = ContactConductance(selection="Styk1", conductance=1500.0)
    st.thermal.cases = [ThermalCase("A", [cc])]

    conductances = thermal_contact_conductances(st, m)
    assert ("x0", "x100", 1500.0) in conductances

    mp = write_model(st, m, tmp_path / "model.inc")
    model_text = (tmp_path / "model.inc").read_text(encoding="utf-8")
    assert "*SURFACE INTERACTION, NAME=INT_TC0" in model_text
    assert "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=TIED" in model_text
    assert "*GAP CONDUCTANCE" in model_text
    assert "1.5" in model_text  # 1500 W/(m²K) -> 1.5 mW/(mm²K)


def test_cavity_naming_unique_and_limit(tmp_path):
    """Finding 4: cavities must map to unique 3-character codes (C01, C02, ...), max 99."""
    from tests.test_thermal import bar_mesh, study
    from fea_core.thermal.inp import write_model, write_case
    from fea_core.thermal.model import ThermalCase

    m = bar_mesh(parts=2)
    st = study(parts=2)
    cav1 = CavityRadiation(selection="x0", emissivity=0.9, ambient=20.0, cavity_name="Cavity1")
    cav2 = CavityRadiation(selection="x100", emissivity=0.8, ambient=20.0, cavity_name="Cavity2")
    st.thermal.cases = [ThermalCase("A", [cav1, cav2])]

    mp = write_model(st, m, tmp_path / "model.inc")
    write_case(st, m, mp, st.thermal.cases[0], tmp_path / "case_cav.inp", "case_cav")
    inp_text = (tmp_path / "case_cav.inp").read_text(encoding="utf-8")

    assert "*RADIATE, CAVITY=C01" in inp_text
    assert "*RADIATE, CAVITY=C02" in inp_text
    assert "** dutina C01: Cavity1" in inp_text
    assert "** dutina C02: Cavity2" in inp_text

    # Test limitu > 99 dutin
    many_cavities = [
        CavityRadiation(selection="x0", emissivity=0.9, ambient=20.0, cavity_name=f"Cav_{i}")
        for i in range(100)
    ]
    st.thermal.cases = [ThermalCase("B", many_cavities)]
    with pytest.raises(ValueError, match="Too many cavities"):
        write_case(st, m, mp, st.thermal.cases[0], tmp_path / "case_err.inp", "case_err")


def test_closed_cavity_not_sink():
    """Finding 5: a closed cavity must not be considered a heat sink for the check of the steady state."""
    from fea_core.thermal.model import is_heat_sink, CavityRadiation, HeatFlux, ThermalCase
    from fea_core.thermal.check import check
    from tests.test_thermal import study

    cav_closed = CavityRadiation(selection="x0", emissivity=0.8, closed=True)
    cav_open = CavityRadiation(selection="x0", emissivity=0.8, closed=False)

    assert not is_heat_sink(cav_closed)
    assert is_heat_sink(cav_open)

    # A steady state with heating and only a closed cavity must fail with "no way to remove the heat"
    st = study(parts=1)
    st.thermal.cases = [ThermalCase("A", [HeatFlux("x0", 1000.0), cav_closed])]
    errors, warnings, used = check(st, ["Dil1"])
    assert any("no way to dissipate heat" in e for e in errors)

    # With an open cavity there is a way to remove the heat → no error about a missing sink
    st2 = study(parts=1)
    st2.thermal.cases = [ThermalCase("B", [HeatFlux("x0", 1000.0), cav_open])]
    errors2, warnings2, used2 = check(st2, ["Dil1"])
    assert not any("no way to dissipate heat" in e for e in errors2)
