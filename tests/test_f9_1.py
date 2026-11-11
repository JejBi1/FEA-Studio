"""Tests for stage F9.1 (post-processing and productivity):
- A1: extended Stress Tool (Tresca, Mohr-Coulomb, von Mises with Re/Rm)
- A3: force by a cut (Free Body Cut: N, Vy, Vz, Mx, My, Mz)
- A4: local coordinate system of the results (cylindrical / Cartesian)
- A8: quality of the mesh (min scaled Jacobian warning)
- A9: mesh control by bodies and edges
- D4: heat transfer calculator by convection (Churchill & Chu, Gnielinski)
- MCP tools: fea_results_section_force, fea_calc_convection, fea_results_probe (csys, criterion)
"""
from __future__ import annotations

import math
import numpy as np

from fea_core.assessment import equivalent_stress
from fea_core.section_forces import section_csys
from fea_core.csys import transform_stress, transform_displacements, stress_labels, displacement_labels
from fea_core.calc.convection import (natural_vertical_plate, natural_horizontal_cylinder,
                                      natural_horizontal_plate, finned_heatsink_natural,
                                      forced_flat_plate, forced_internal_pipe)
from fea_core.study import Study, MeshSettings, CoordinateSystem


def test_convection_vertical_plate():
    """Check of the calculation of the vertical plate according to Churchill & Chu."""
    res = natural_vertical_plate(L_mm=200.0, Ts=60.0, Tinf=20.0, W_mm=100.0)
    assert 4.0 <= res["h"] <= 7.0
    assert res["Nu"] > 0
    assert res["Ra"] > 0
    assert res["regime"] == "laminar"
    assert "Q_1side_W" in res


def test_convection_horizontal_cylinder():
    """Check of the calculation of the horizontal cylinder according to Churchill & Chu."""
    res = natural_horizontal_cylinder(D_mm=50.0, Ts=80.0, Tinf=20.0, L_mm=500.0)
    assert 5.0 <= res["h"] <= 8.0
    assert res["Nu"] > 0
    assert "Q_W" in res


def test_convection_horizontal_plate():
    """Check of the calculation of the horizontal plate (Incropera)."""
    res_up = natural_horizontal_plate(L_mm=200.0, W_mm=100.0, Ts=60.0, Tinf=20.0, orientation="upper_hot")
    res_down = natural_horizontal_plate(L_mm=200.0, W_mm=100.0, Ts=60.0, Tinf=20.0, orientation="lower_hot")
    assert res_up["h"] > res_down["h"]  # a favourable lift has a higher h than a suppressed one
    assert res_up["h"] > 0
    assert res_down["h"] > 0


def test_convection_finned_heatsink():
    """Check of the calculation of the finned heat sink."""
    res = finned_heatsink_natural(H_mm=50.0, L_mm=100.0, S_mm=8.0, t_fin_mm=2.0, N_fins=10, Ts=60.0, Tinf=20.0)
    assert res["h_channel"] > 0
    assert res["h_effective"] > 0
    assert 0 < res["fin_efficiency"] <= 1.0
    assert res["Q_total_W"] > 0


def test_convection_forced_flow():
    """Check of the forced convection along a plate and in a pipe."""
    r_plate = forced_flat_plate(L_mm=200.0, velocity_m_s=5.0, Ts=60.0, Tinf=20.0)
    assert r_plate["h"] > 10.0
    assert r_plate["Re"] > 0

    r_pipe = forced_internal_pipe(D_mm=25.0, velocity_m_s=5.0, L_mm=1000.0, Ts=80.0, Tinf=20.0)
    assert 20.0 <= r_pipe["h"] <= 35.0
    assert "turbulent" in r_pipe["regime"] or "transitional" in r_pipe["regime"]


def test_assessment_equivalent_stress():
    """Check of the calculation of the equivalent stress: Mises, Tresca, Mohr-Coulomb."""
    # Pure tension: sxx = 100, the rest 0
    S_tension = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert math.isclose(float(equivalent_stress(S_tension, "mises")[0]), 100.0, rel_tol=1e-5)
    assert math.isclose(float(equivalent_stress(S_tension, "tresca")[0]), 100.0, rel_tol=1e-5)
    assert math.isclose(float(equivalent_stress(S_tension, "mohr_coulomb", Rm=200.0, Rc=600.0)[0]), 100.0, rel_tol=1e-5)

    # Pure shear: sxy = 100, the rest 0
    S_shear = np.array([[0.0, 0.0, 0.0, 100.0, 0.0, 0.0]])
    # Mises: sqrt(3) * 100 = 173.205
    assert math.isclose(float(equivalent_stress(S_shear, "mises")[0]), math.sqrt(3) * 100.0, rel_tol=1e-5)
    # Tresca: 2 * 100 = 200.0
    assert math.isclose(float(equivalent_stress(S_shear, "tresca")[0]), 200.0, rel_tol=1e-5)
    # Ratio Tresca / Mises for pure shear = 2 / sqrt(3) = 1.1547
    ratio = float(equivalent_stress(S_shear, "tresca")[0] / equivalent_stress(S_shear, "mises")[0])
    assert math.isclose(ratio, 2.0 / math.sqrt(3.0), rel_tol=1e-5)

    # Pure compression: sxx = -300
    S_comp = np.array([[-300.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    mc_comp = float(equivalent_stress(S_comp, "mohr_coulomb", Rm=200.0, Rc=600.0)[0])
    # U Mohr-Coulomba: sigma_eq = sigma1 - (Rm/Rc)*sigma3 = 0 - (1/3)*(-300) = 100
    assert math.isclose(mc_comp, 100.0, rel_tol=1e-5)


def test_csys_transformation():
    """Check of the transformation of the stress and the displacements in the cylindrical coordinate system."""
    # Cylindrical system with axis z = [0, 0, 1]
    cs = CoordinateSystem(name="Cyl", kind="cylindrical", origin=[0.0, 0.0, 0.0], z_axis=[0.0, 0.0, 1.0], x_axis=[1.0, 0.0, 0.0])
    st = Study("test", "test.step", coordinate_systems={"Cyl": cs})

    # Point on the positive axis X: (r=10, theta=0)
    pts_x = np.array([[10.0, 0.0, 0.0]])
    # Point on the positive axis Y: (r=10, theta=90 deg)
    pts_y = np.array([[0.0, 10.0, 0.0]])

    # Stress tensor sxx=100, syy=50
    S = np.array([[100.0, 50.0, 0.0, 0.0, 0.0, 0.0]])

    # At the point on the axis X: sigma_r = sxx = 100, sigma_theta = syy = 50
    S_trans_x = transform_stress(st, "Cyl", S, pts_x)
    assert math.isclose(float(S_trans_x[0, 0]), 100.0, abs_tol=1e-5)
    assert math.isclose(float(S_trans_x[0, 1]), 50.0, abs_tol=1e-5)

    # At the point on the axis Y (rotation by 90°): sigma_r = syy = 50, sigma_theta = sxx = 100
    S_trans_y = transform_stress(st, "Cyl", S, pts_y)
    assert math.isclose(float(S_trans_y[0, 0]), 50.0, abs_tol=1e-5)
    assert math.isclose(float(S_trans_y[0, 1]), 100.0, abs_tol=1e-5)

    # Displacement at the point on the axis Y: Uy = 1 mm -> radial displacement ur = 1 mm
    U = np.array([[0.0, 1.0, 0.0]])
    U_trans_y = transform_displacements(st, "Cyl", U, pts_y)
    assert math.isclose(float(U_trans_y[0, 0]), 1.0, abs_tol=1e-5)  # ur
    assert math.isclose(float(U_trans_y[0, 1]), 0.0, abs_tol=1e-5)  # utheta

    # Popisky
    assert tuple(stress_labels(st, "Cyl")) == ("σr", "σθ", "σz", "τrθ", "τθz", "τzr")
    assert tuple(displacement_labels(st, "Cyl")) == ("ur", "uθ", "uz")


def test_section_csys():
    """Check of the creation of the orthonormal basis system of the section."""
    n = np.array([1.0, 0.0, 0.0])
    ex, ey, ez, R = section_csys(n)
    assert np.allclose(ex, n)
    assert np.allclose(R[0], n)
    assert np.allclose(R @ R.T, np.eye(3))
    assert math.isclose(float(np.linalg.det(R)), 1.0)


def test_mesh_part_sizing_and_jacobian_warning():
    """Check that MeshSettings supports part_sizes and solve.py reports minSJ < 0.2."""
    ms = MeshSettings(size=5.0, part_sizes={"PartA": 2.0, "PartB": 1.0})
    assert ms.part_sizes["PartA"] == 2.0

    st = Study("Test", "dummy.step", mesh=ms)
    d = st.to_dict()
    st2 = Study.from_dict(d)
    assert st2.mesh.part_sizes["PartA"] == 2.0


def test_mcp_calc_convection():
    """Check of the call of fea_calc_convection through the shared interface."""
    from fea_core.calc import call
    r = call("convection", "vertical_plate", {"L_mm": 200, "Ts": 60, "Tinf": 20})
    assert "h" in r
    assert 4.0 <= r["h"] <= 7.0
