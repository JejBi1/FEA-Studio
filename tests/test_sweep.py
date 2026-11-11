import math
import copy
from pathlib import Path
import numpy as np
import pytest

from fea_core.sweep import (
    DesignPoint, ParametricStudySpec, ParametricStudyResult,
    identify_pareto, scale_study_loads, run_parametric_study
)
from fea_core.study import (
    Study, Material, LoadCase, Force, Pressure, Gravity,
    RemoteLoad, BearingLoad, Rotation, ThermalLoad, MeshSettings
)
from fea_core.mesh import Mesh
from fea_core.results import Results, StepResult


def test_spec_combinations():
    spec = ParametricStudySpec(
        load_factors=[0.5, 1.0, 1.5],
        mesh_sizes=[5.0, 10.0],
        max_points=20,
    )
    combos = spec.combinations()
    assert len(combos) == 6
    assert combos[0] == {"load_factor": 0.5, "mesh_size": 5.0}
    assert combos[-1] == {"load_factor": 1.5, "mesh_size": 10.0}
    assert len(spec.warnings) == 0


def test_spec_one_element_mesh_sizes_and_max_points():
    # a 1-element list of mesh_sizes must not be ignored
    spec1 = ParametricStudySpec(
        mesh_sizes=[3.5],
    )
    combos1 = spec1.combinations()
    assert len(combos1) == 1
    assert combos1[0] == {"mesh_size": 3.5}

    # max_points limits the number and generates a warning
    spec2 = ParametricStudySpec(
        load_factors=[0.5, 1.0, 1.5, 2.0],
        mesh_sizes=[2.0, 4.0],
        max_points=3,
    )
    combos2 = spec2.combinations()
    assert len(combos2) == 3
    assert len(spec2.warnings) == 1
    assert "max_points=3" in spec2.warnings[0]


def test_pareto_front():
    # DP0: light and low stress -> pareto
    dp0 = DesignPoint(index=0, params={}, mass=1.0, max_stress=100.0, converged=True)
    # DP1: heavier and higher stress -> dominated by DP0
    dp1 = DesignPoint(index=1, params={}, mass=2.0, max_stress=150.0, converged=True)
    # DP2: heavier but lower stress -> trade-off -> pareto
    dp2 = DesignPoint(index=2, params={}, mass=1.5, max_stress=80.0, converged=True)
    # DP3: error
    dp3 = DesignPoint(index=3, params={}, mass=0.5, max_stress=50.0, converged=False)

    pareto = identify_pareto([dp0, dp1, dp2, dp3])
    assert pareto == [0, 2]


def test_scale_study_loads():
    st = Study(name="ScaleTest", geometry="model.step")
    lc = LoadCase("LC1", loads=[
        Force("F1", [100.0, 0.0, -50.0]),
        Pressure("P1", value=2.0, gradient=[0.1, 0.0, 0.2]),
        Gravity(vector=[0.0, 0.0, -9810.0]),
        RemoteLoad("R1", point=[0.0, 0.0, 0.0], force=[10.0, 20.0, 30.0], moment=[1.0, 2.0, 3.0]),
        BearingLoad("B1", vector=[500.0, 0.0, 0.0]),
        Rotation(point=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], omega=100.0, alpha=50.0),
        ThermalLoad(value=120.0),
    ])
    st.load_cases.append(lc)

    scale_study_loads(st, factor=4.0)

    # Force: linearly 4x
    assert lc.loads[0].vector == [400.0, 0.0, -200.0]
    # Pressure: value and gradient 4x
    assert lc.loads[1].value == pytest.approx(8.0)
    assert lc.loads[1].gradient == pytest.approx([0.4, 0.0, 0.8])
    # Gravity: linearly 4x
    assert lc.loads[2].vector == [0.0, 0.0, -39240.0]
    # Remote: force i moment 4x
    assert lc.loads[3].force == [40.0, 80.0, 120.0]
    assert lc.loads[3].moment == [4.0, 8.0, 12.0]
    # Bearing: linearly 4x
    assert lc.loads[4].vector == [2000.0, 0.0, 0.0]
    # Rotation: omega scales with sqrt(factor) = 2x, alpha linearly = 4x
    assert lc.loads[5].omega == pytest.approx(200.0)
    assert lc.loads[5].alpha == pytest.approx(200.0)
    # ThermalLoad: the temperature in °C is NOT scaled
    assert lc.loads[6].value == 120.0


def test_pure_load_sweep_shortcut(tmp_path, monkeypatch):
    st = Study(name="LinearBeam", geometry="beam.step")
    mat = Material(name="STEEL", E=210000.0, nu=0.3, rho=7.85e-9, Re=250.0)
    st.materials["STEEL"] = mat
    st.part_materials["Part1"] = "STEEL"
    st.load_cases.append(LoadCase("Load1", [Force("end", [0.0, 0.0, -100.0])]))

    # Dummy mesh and results
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    tets = np.array([[0, 1, 2, 3]])
    mesh = Mesh(nodes, tets, np.zeros(1, dtype=np.int32), ["Part1"])

    U_base = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
    S_base = np.array([[50.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4)
    step_base = StepResult("Load1", U_base, S_base, reactions={"end": np.array([0.0, 0.0, 100.0])})
    mock_res = Results(mesh, [step_base], [], {})

    solve_count = 0

    def mock_run_study(study, workdir, cancel=None, mesh=None):
        nonlocal solve_count
        solve_count += 1
        return mock_res

    monkeypatch.setattr("fea_core.sweep.run_study", mock_run_study)

    spec = ParametricStudySpec(load_factors=[1.0, 2.0, 0.5])
    res = run_parametric_study(st, spec, tmp_path / "sweep_out")

    # A linear purely load sweep must solve the study exactly ONCE
    assert solve_count == 1
    assert len(res.points) == 3

    # DP0 (factor 1.0): max stress = 50 MPa
    assert res.points[0].max_stress == pytest.approx(50.0)
    # DP1 (factor 2.0): max stress = 100 MPa, max disp = 2x
    assert res.points[1].max_stress == pytest.approx(100.0)
    assert res.points[1].max_disp == pytest.approx(res.points[0].max_disp * 2.0)
    # DP2 (factor 0.5): max stress = 25 MPa
    assert res.points[2].max_stress == pytest.approx(25.0)


def test_part_nodes_element_indexing_fix(tmp_path, monkeypatch):
    """Check that min_safety does not index the nodes by the indices of the elements tet_part."""
    st = Study(name="MultiElemTest", geometry="model.step")
    mat1 = Material(name="M1", E=210000.0, nu=0.3, rho=7.85e-9, Re=300.0)
    mat2 = Material(name="M2", E=210000.0, nu=0.3, rho=7.85e-9, Re=150.0)
    st.materials["M1"] = mat1
    st.materials["M2"] = mat2
    st.part_materials["Part1"] = "M1"
    st.part_materials["Part2"] = "M2"
    st.load_cases.append(LoadCase("LC", [Force("F", [10.0, 0.0, 0.0])]))

    # Mesh with 6 nodes and 3 elements
    # Elements 0 and 1 belong to Part1 (tet_part=0)
    # Element 2 belongs to Part2 (tet_part=1)
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])
    tets = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
    ])
    tet_part = np.array([0, 0, 1], dtype=np.int32)
    mesh = Mesh(nodes, tets, tet_part, ["Part1", "Part2"])

    # Stresses at the nodes 0..5
    # Node 5 has a high stress of 100 MPa in Part2 (Re=150 -> safety = 150/100 = 1.5)
    # Nodes 0..4 have a stress of 50 MPa in Part1 (Re=300 -> safety = 300/50 = 6.0)
    S = np.zeros((6, 6))
    S[:5, 0] = 50.0
    S[5, 0] = 100.0
    step = StepResult("LC", np.zeros((6, 3)), S)
    mock_res = Results(mesh, [step], [], {})

    monkeypatch.setattr("fea_core.sweep.run_study", lambda st, wd, **kw: mock_res)

    spec = ParametricStudySpec(load_factors=[1.0])
    res = run_parametric_study(st, spec, tmp_path / "safety_test")

    # The minimum safety must be 1.5 (from Part2), neither infinity nor wrongly indexed
    assert res.points[0].min_safety == pytest.approx(1.5)
