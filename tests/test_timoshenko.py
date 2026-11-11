"""Own solver of frame systems (fea_core/beams/timoshenko.py): element (stiffness, mass, K_G, equivalent loads),
condensation of the hinges, mechanism, temperature, natural frequencies and buckling, power, routing of the solver, storage of the results.
Without Gmsh and ccx (fast).
"""
import math
import time

import numpy as np
import pytest

from fea_core.beams import fem, timoshenko as T, sections as S
from fea_core.beams.fem import FrameError
from fea_core.beams.model import (FrameMember, FrameSupport, SectionDef, FrameNodeLoad, FrameMemberLoad,
                                  FrameTemperature)
from fea_core.beams.run import run_frame_tim
from fea_core.study import Study, Material, LoadCase

E, NU, RHO = 210000.0, 0.3, 7.85e-9
G = E / (2 * (1 + NU))


def _study(sec="IPE 200", length=2000.0, shear=True):
    st = Study("t", "")
    st.add_material(Material("Ocel S235", E, NU, RHO, Re=235.0, alpha=12e-6), default=True)
    fr = st.frame
    fr.settings.shear_deformation = shear
    fr.sections["S"] = SectionDef("S", library=sec)
    fr.add_node("A", [0, 0, 0])
    fr.add_node("B", [length, 0, 0])
    fr.members["m"] = FrameMember("m", "A", "B", "S")
    return st


def _el(shear=True, n=1, **kw):
    st = _study(shear=shear)
    for k, v in kw.items():
        setattr(st.frame.members["m"], k, v)
    st.frame.supports.append(FrameSupport("A"))
    fe = fem.build(st, segments=n)
    return st, fe, T.elements(fe, shear)


def test_stiffness_equals_strain_energy_of_shape_functions():
    """K from the shape functions (∫EI·θ'² + G·A_s·(v' − θ)² dx) = closed matrix – checks the shape functions and the signs."""
    st, fe, el = _el(length=1500.0) if False else _el()
    K = T.local_k(el)[0]
    xi, w = T.GAUSS3
    for idx, phi, EI, As, sgn in ((T.IDX_XY, el.phi_y, E * el.Iz, el.Asy, np.ones(4)),
                                  (T.IDX_XZ, el.phi_z, E * el.Iy, el.Asz, T.SGN_XZ)):
        Nv, dNv, Nth = T.shapes(xi, phi, el.L)
        L, p = el.L[0], phi[0]
        dNth = np.array([6 * (2 * xi - 1) / (L * L * (1 + p)), (6 * xi - (4 + p)) / (L * (1 + p)),
                         -6 * (2 * xi - 1) / (L * L * (1 + p)), (6 * xi - (2 - p)) / (L * (1 + p))])
        gam = dNv[0] - Nth[0]
        ke = L * ((dNth * w) @ dNth.T) * EI[0] + L * ((gam * w) @ gam.T) * G * As[0]
        ke = ke * (sgn[:, None] * sgn[None, :])
        assert np.allclose(K[np.ix_(idx, idx)], ke, rtol=1e-11, atol=1e-6 * np.abs(ke).max())


def test_stiffness_rigid_body_null_space():
    st, fe, el = _el()
    K = T.local_k(el)[0]
    for v in range(3):
        rb = np.zeros(12)
        rb[v] = rb[6 + v] = 1.0
        assert np.abs(K @ rb).max() < 1e-6
    rot = np.zeros(12)                                  # rigid rotation about z: v = θ·x
    rot[5] = rot[11] = 1.0
    rot[7] = el.L[0]
    assert np.abs(K @ rot).max() < 1e-6 * np.abs(K).max()


@pytest.mark.parametrize("shear", [False, True])
def test_mass_rigid_motions(shear):
    st, fe, el = _el(shear)
    M = T.local_mass(el)[0]
    L = el.L[0]
    rb = np.zeros(12)
    rb[1] = rb[7] = 1.0                                  # posuv v
    assert rb @ M @ rb == pytest.approx(RHO * el.A[0] * L, rel=1e-12)
    rot = np.zeros(12)
    rot[5] = rot[11] = 1.0                               # rotation about z about the node a: v = θx
    rot[7] = L
    assert rot @ M @ rot == pytest.approx(RHO * el.A[0] * L ** 3 / 3 + RHO * el.Iz[0] * L, rel=1e-12)
    tors = np.zeros(12)
    tors[3] = tors[9] = 1.0
    assert tors @ M @ tors == pytest.approx(RHO * (el.Iy[0] + el.Iz[0]) * L, rel=1e-12)


def test_geometric_stiffness_matches_classic_euler():
    st, fe, el = _el(False)
    L, N = el.L[0], -1000.0
    kg = T.local_kg(el, np.full((1, 3), N), False)[0]
    ref = N / (30 * L) * np.array([[36, 3 * L, -36, 3 * L], [3 * L, 4 * L * L, -3 * L, -L * L],
                                   [-36, -3 * L, 36, -3 * L], [3 * L, -L * L, -3 * L, 4 * L * L]])
    assert np.allclose(kg[np.ix_(T.IDX_XY, T.IDX_XY)], ref, rtol=1e-12)
    s = T.SGN_XZ
    assert np.allclose(kg[np.ix_(T.IDX_XZ, T.IDX_XZ)], ref * np.outer(s, s), rtol=1e-12)


def test_equivalent_loads_reduce_to_legacy_and_keep_resultant():
    st, fe, el = _el(False)
    qa, qb = np.array([[0.2, 1.0, -2.0]]), np.array([[-0.1, -3.0, 4.0]])
    f = T.eq_loads(el, qa, qb, np.zeros(1), np.zeros(1), np.zeros(1))[0]
    ref = fem._consistent(qa[0], qb[0], el.L[0], np.eye(3))
    assert np.allclose(f, ref, rtol=1e-12, atol=1e-9)
    st, fe, el = _el(True)                                 # with the shear a different distribution, the same resultant and moment
    f = T.eq_loads(el, qa, qb, np.zeros(1), np.zeros(1), np.zeros(1))[0]
    L = el.L[0]
    assert f[1] + f[7] == pytest.approx(L * (qa[0, 1] + qb[0, 1]) / 2, rel=1e-12)
    m_a = f[5] + f[11] + L * f[7]                          # moment about the node a: ∫ s·q ds
    assert m_a == pytest.approx(L * L * (qa[0, 1] + 2 * qb[0, 1]) / 6, rel=1e-12)


def test_hinge_condensation_propped_cantilever():
    """Clamping – hinge (E–B, distributed q): M_A = qL²/8, R_B = 3qL/8, M = 0 at the hinge."""
    st = _study(shear=False)
    L, q = 2000.0, 5.0
    st.frame.supports += [FrameSupport("A"), FrameSupport("B", [False, True, True, True, False, False])]
    st.frame.members["m"].releases_end = [False, False, False, False, True, False]
    st.load_cases.append(LoadCase("q", [FrameMemberLoad("m", [0, 0, -q])]))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        res = run_frame_tim(st, td)
    fr = res.step(0).frame
    assert fr["reactions"]["B"][2] == pytest.approx(3 * q * L / 8, rel=1e-12)
    assert fr["members"]["m"]["My"][0] == pytest.approx(q * L * L / 8, rel=1e-12)
    assert abs(fr["members"]["m"]["My"][-1]) < 1e-6
    assert res.step(0).equilibrium["imbalance"] < 1e-12


def test_mechanism_is_reported_with_node():
    st = _study()
    st.frame.members["m"].kind = "truss"
    st.frame.supports += [FrameSupport("A", [True, True, True, False, False, False])]      # B free → mechanism
    st.load_cases.append(LoadCase("F", [FrameNodeLoad("B", [0, 0, -10.0])]))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(FrameError, match="'B'"):
            run_frame_tim(st, td)
    st2 = _study()
    st2.frame.supports.append(FrameSupport("A", [True, True, True, False, False, False]))   # rotation about x free
    st2.load_cases.append(LoadCase("F", [FrameNodeLoad("B", [0, 0, -10.0])]))
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(FrameError, match="unstable"):
            run_frame_tim(st2, td)


def test_temperature_loads():
    st = _study()
    st.frame.supports += [FrameSupport("A"), FrameSupport("B")]
    st.load_cases += [LoadCase("dT", [FrameTemperature([], 40.0)]), LoadCase("g", [FrameTemperature(["m"], 0.0, 25.0, 10.0)])]
    sec = S.get("IPE 200")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        res = run_frame_tim(st, td)
    m0 = res.step(0).frame["members"]["m"]
    assert m0["N"][0] == pytest.approx(-E * sec.A * 12e-6 * 40, rel=1e-12)
    f = np.array(sec.fibers)
    m1 = res.step(1).frame["members"]["m"]
    assert m1["My"][0] == pytest.approx(-E * sec.Iy * 12e-6 * 25 / np.ptp(f[:, 1]), rel=1e-12)
    assert m1["Mz"][0] == pytest.approx(+E * sec.Iz * 12e-6 * 10 / np.ptp(f[:, 0]), rel=1e-12)
    # JSON and the change of the solver
    st2 = Study.from_dict(st.to_dict())
    assert st2.load_cases[1].loads[0].grad_z == 25.0 and st2.load_cases[1].loads[0].kind == "frame_temperature"
    st2.frame.settings.solver = "ccx_u1"
    from fea_core.beams.run import check
    assert any("built-in solver" in e for e in check(st2)[0])


def test_modal_simply_supported_timoshenko_and_buckling_euler():
    import verification.cases  # noqa: F401 – first (cyclic import)
    from verification.frame_cases import tim_ss_freq
    st = _study(length=4000.0)
    st.frame.supports += [FrameSupport("A", [True, True, True, True, False, False]),
                          FrameSupport("B", [False, True, True, False, False, False])]
    st.modal.enabled, st.modal.modes = True, 4
    st.load_cases.append(LoadCase("P", [FrameNodeLoad("B", [-1000.0, 0, 0])]))
    st.buckling.enabled, st.buckling.modes = True, 2
    st.frame.settings.shear_deformation = False
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        res = run_frame_tim(st, td)
    sec = S.get("IPE 200")
    f_ref = 1 / (2 * math.pi) * (math.pi / 4000.0) ** 2 * math.sqrt(E * sec.Iz / (RHO * sec.A))   # E–B (without rotary inertia)
    f1 = res.modes[0].values[0]
    assert f1 == pytest.approx(f_ref, rel=2e-4)                       # the rotary inertia lowers it by (π/L)²·i²/2
    assert res.modes[1].values[0] == pytest.approx(math.pi ** 2 * E * sec.Iz / 4000.0 ** 2 / 1000, rel=1e-4)
    m = res.modes[0]
    assert len(m.eff_mass) == len(m.values) and m.total_mass[0] == pytest.approx(RHO * sec.A * 4000.0, rel=1e-12)
    assert m.frame_shapes_own.shape[1] == len(m.frame_xyz)
    # exact Timoshenko (with shear and rotary inertia)
    st.frame.settings.shear_deformation = True
    st.buckling.enabled = False
    with tempfile.TemporaryDirectory() as td:
        res2 = run_frame_tim(st, td)
    assert res2.modes[0].values[0] == pytest.approx(tim_ss_freq(1, 4000.0, sec.A, sec.Av_y, sec.Iz), rel=1e-4)


def test_big_frame_is_fast_and_exact():
    """Spatial frame of ~800 members: assembly + solution + diagrams in units of seconds, equilibrium exact."""
    st = Study("big", "")
    st.add_material(Material("Ocel S235", E, NU, RHO, Re=235.0), default=True)
    fr = st.frame
    fr.sections["IPE"] = SectionDef("IPE", library="IPE 200")
    fr.sections["HEA"] = SectionDef("HEA", library="HEA 200")
    nx = ny = nz = 6
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                fr.add_node(f"n{i}_{j}_{k}", [i * 5000.0, j * 5000.0, k * 3500.0])
    n = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                if k < nz:
                    fr.members[f"c{n}"] = FrameMember(f"c{n}", f"n{i}_{j}_{k}", f"n{i}_{j}_{k + 1}", "HEA", up=[1, 0, 0])
                    n += 1
                if k > 0 and i < nx:
                    fr.members[f"x{n}"] = FrameMember(f"x{n}", f"n{i}_{j}_{k}", f"n{i + 1}_{j}_{k}", "IPE")
                    n += 1
                if k > 0 and j < ny:
                    fr.members[f"y{n}"] = FrameMember(f"y{n}", f"n{i}_{j}_{k}", f"n{i}_{j + 1}_{k}", "IPE")
                    n += 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            fr.supports.append(FrameSupport(f"n{i}_{j}_0"))
    loads = [FrameMemberLoad(k, [0, 0, -10.0]) for k in fr.members if k[0] in "xy"]
    loads += [FrameNodeLoad(f"n{i}_{j}_{nz}", [2000.0, 500.0, 0]) for j in range(ny + 1) for i in range(nx + 1)]
    st.load_cases.append(LoadCase("a", loads))
    assert len(fr.members) > 700
    t0 = time.time()
    fe = fem.build(st)
    model = T.TimModel(st, fe)
    out = model.solve_cases(st.load_cases)
    rec = T.step_record(model, out[0])
    dt = time.time() - t0
    assert dt < 5.0, f"{len(fr.members)} prutů: {dt:.2f} s"
    assert rec["node_residual"] < 1e-9
    Fa, Ma, Rf, Rm = model.resultants(out[0], np.zeros(3))
    assert np.abs(Fa + Rf).max() < 1e-8 * np.abs(Fa).max()
    assert np.abs(Ma + Rm).max() < 1e-8 * np.abs(Ma).max()


def test_solver_routing_and_results_roundtrip(tmp_path):
    from fea_core.beams.fem import solver_of
    from fea_core.results import Results
    st = _study()
    st.frame.supports.append(FrameSupport("A"))
    assert solver_of(st) == "timoshenko"
    st.frame.settings.solver = "ccx_u1"
    assert solver_of(st) == "ccx_u1" and solver_of(st, "timoshenko") == "timoshenko"
    st.frame.settings.solver = "timoshenko"
    st.geometry = "x.step"
    assert solver_of(st) == "ccx_u1"                       # mixed model
    st.geometry = ""
    st.load_cases.append(LoadCase("F", [FrameNodeLoad("B", [0, 0, -1000.0])]))
    st.modal.enabled, st.modal.modes = True, 3
    res = run_frame_tim(st, tmp_path)
    r2 = Results.load(tmp_path / "results.npz")
    m = r2.step(0).frame["members"]["m"]
    for k in ("s", "N", "Vy", "Vz", "T", "My", "Mz", "points", "u", "sigma_max", "tau_est", "sigma_eq", "section"):
        assert k in m
    assert len(m["s"]) == len(m["points"]) == len(m["u"])
    assert r2.step(0).frame["reactions"]["A"][2] == pytest.approx(1000.0, rel=1e-12)
    assert r2.modes[0].frame_shapes_own is not None and len(r2.modes[0].frame_xyz) == r2.modes[0].frame_shapes_own.shape[1]
    from fea_core.beams.run import eigen_fe
    assert len(eigen_fe(st).xyz) == len(r2.modes[0].frame_xyz)
    assert res.info["backend"] == "timoshenko" and "frame" in r2.info


def test_truss_member_multi_segment_buckling_and_modal():
    """A truss member (hinges at both ends, J = 0) as a pinned column on the subdivision for buckling and frequencies: without a singularity
    from the free torsion of the chain of elements; buckling = Euler, statics = purely axial.
    """
    import tempfile
    st = _study(length=3000.0, shear=False)
    st.frame.members["m"].kind = "truss"
    st.frame.supports += [FrameSupport("A", [True, True, True, False, False, False]),
                          FrameSupport("B", [False, True, True, False, False, False])]
    st.load_cases.append(LoadCase("P", [FrameNodeLoad("B", [-1000.0, 0, 0])]))
    st.buckling.enabled, st.modal.enabled = True, True
    st.modal.modes = 3
    with tempfile.TemporaryDirectory() as td:
        res = run_frame_tim(st, td)
    sec = S.get("IPE 200")
    assert res.modes[1].values[0] == pytest.approx(math.pi ** 2 * E * sec.Iz / 3000.0 ** 2 / 1000, rel=1e-4)
    m = res.step(0).frame["members"]["m"]
    assert m["N"][0] == pytest.approx(-1000.0, rel=1e-12) and max(abs(v) for v in m["My"]) < 1e-6
    assert res.modes[0].values[0] > 0
