"""Point mass in ccx (slow test, Gmsh + CalculiX): a rigid constraint with a rotary inertia and a distributing constraint pass
the statics and the *FREQUENCY (previously "*ERROR in add_bo_st / add_sm_st: coefficient should be 0" – the mass in a dependent node without
a matrix structure) and the participation factors Γ are correct also with the mass off the face (Σ Γ² ≤ the mass of the model;
ccx counts the rotation of the ROT node as displacements for a rigid constraint).
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _run(tmp_path, coupling: str, inertia: bool, modes: int = 6):
    from fea_core.geometry import Part, read_step, write_step
    from fea_core.selectors import find_faces
    from fea_core.study import Study, Fixed, LoadCase, Gravity, MeshSettings, Material
    from fea_core.dynamics.model import PointMass, Excitation
    from fea_core.solve import run_study
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    step = tmp_path / "m.step"
    write_step([Part("Blok", BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 10, 10, 30).Solid())], step)
    parts = read_step(step)
    st = Study("point mass", str(step))
    st.add_material(Material("ocel", 210000.0, 0.3, 7.85e-9, Rm=360.0), default=True)
    st.add_selection("dno", find_faces(parts, z=0))
    st.add_selection("vrch", find_faces(parts, z=30))
    st.constraints.append(Fixed("dno"))
    m = 1e-4
    J = m * 200 / 12
    st.masses.append(PointMass("vrch", m, point=[5, 5, 40], inertia=[J, J, J] if inertia else None, coupling=coupling))
    st.load_cases.append(LoadCase("g", [Gravity()]))
    st.dynamics.enabled = True
    st.dynamics.modes = modes
    st.dynamics.harmonic.enabled = True
    st.dynamics.harmonic.f_min, st.dynamics.harmonic.f_max = 10, 3000
    st.dynamics.harmonic.excitation = [Excitation(kind="base_acc", dof=1, unit="g")]
    st.mesh = MeshSettings(size=5.0)
    return run_study(st, tmp_path / "run", on_log=lambda s: None)


@pytest.mark.parametrize("coupling,inertia", [("rigid", True), ("distributing", False)])
def test_point_mass_runs_and_gamma(tmp_path, coupling, inertia):
    res = _run(tmp_path, coupling, inertia, modes=12)
    d = res.dynamics
    assert d.checks["mass_norm_error"] < 1e-3
    assert d.checks["gamma_error"] < 1e-3
    frac = (np.asarray(d.gamma)[:, :3] ** 2).sum(axis=0) / d.model_mass
    assert (frac <= 1.0 + 1e-3).all() and frac[0] > 0.8, frac
    assert res.steps[0].equilibrium["imbalance"] < 1e-3
