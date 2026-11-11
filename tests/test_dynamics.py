"""Fast tests of the dynamics (fea_core.dynamics) without ccx: JSON model, tet10 mass, response on a synthetic basis,
maximum over the phase, saving of the results, check of the settings, version of the study file.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from fea_core.study import Study, Material, Fixed, Displacement, LoadCase
from fea_core.dynamics.model import (DynamicsSettings, Excitation, PointMass, SurfaceMass, Probe, G_MM)
from fea_core.dynamics import response as R
from fea_core.dynamics.check import check
from fea_core.dynamics.modal import zetas


def _study():
    st = Study("t", "x.step")
    st.add_material(Material("ocel", 210000.0, 0.3, 7.85e-9, Rm=360.0), default=True)
    return st


def test_dynamics_json_roundtrip():
    st = _study()
    ds = st.dynamics
    ds.enabled = True
    ds.harmonic.enabled = True
    ds.harmonic.excitation = [Excitation(kind="base_acc", dof=2, table=[[1.0, 1.0], [100.0, 2.0]])]
    ds.random.enabled = True
    ds.random.asd = [[20.0, 0.04], [2000.0, 0.04]]
    ds.probes = [Probe("p", point=[1.0, 2.0, 3.0])]
    st.masses = [PointMass("A", 1e-3, inertia=[1.0, 2.0, 3.0]), SurfaceMass("B", 2.5)]
    d = json.loads(json.dumps(st.to_dict()))
    assert d["version"] == 2 and set(d["features"]) == {"dynamics", "masses"}
    s2 = Study.from_dict(d)
    assert s2.dynamics.harmonic.excitation[0].dof == 2
    assert s2.dynamics.random.asd[1][0] == 2000.0
    assert isinstance(s2.masses[0], PointMass) and isinstance(s2.masses[1], SurfaceMass)
    assert s2.dynamics.probes[0].point == [1.0, 2.0, 3.0]


def test_unknown_kind_and_key():
    from fea_core.study import StudyFormatError
    st = _study()
    d = st.to_dict()
    d["novy_klic"] = {"x": 1}
    assert Study.from_dict(d).load_warnings
    d["constraints"] = [{"kind": "neznama_vazba", "selection": "a"}]
    with pytest.raises(StudyFormatError):
        Study.from_dict(d)


def test_old_file_without_f8_keys():
    st = _study()
    d = st.to_dict()
    for k in ("masses", "contacts", "contact_settings", "analysis", "mechanism", "dynamics", "features"):
        d.pop(k, None)
    d["version"] = 1
    s2 = Study.from_dict(d)
    assert not s2.dynamics.enabled and not s2.contacts and not s2.load_warnings


def test_tet10_consistent_mass():
    """Consistent tet10 mass: Σ M·1 = m, φᵀMφ for a rigid displacement = m."""
    from fea_core.mesh import Mesh
    from fea_core.dynamics.mass import mass_times, total_mass
    X = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]], float)
    mids = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    nodes = np.vstack([X, [(X[a] + X[b]) / 2 for a, b in mids]])
    mesh = Mesh(nodes, np.arange(10)[None, :], np.zeros(1, int), ["A"])
    st = _study()
    m = 7.85e-9 * 2 * 3 * 4 / 6
    assert total_mass(st, mesh) == pytest.approx(m, rel=1e-12)
    U = np.tile([1.0, 0, 0], (10, 1))
    MU = mass_times(st, mesh, U)
    assert MU[:, 0].sum() == pytest.approx(m, rel=1e-12)
    assert float((U * MU).sum()) == pytest.approx(m, rel=1e-12)


def test_rotary_points_reproduce_inertia():
    from fea_core.dynamics.mass import rotary_points
    pm = PointMass("A", 2.0, inertia=[[3.0, 0.2, 0], [0.2, 4.0, 0.1], [0, 0.1, 5.0]])
    p = np.array([1.0, 2.0, 3.0])
    pts = rotary_points(pm, p, 1.0)
    m = sum(mu for _, mu in pts)
    c = sum(mu * x for x, mu in pts) / m
    I = sum(mu * ((x - p) @ (x - p) * np.eye(3) - np.outer(x - p, x - p)) for x, mu in pts)
    assert m == pytest.approx(2.0)
    assert np.allclose(c, p)
    assert np.allclose(I, np.asarray(pm.inertia), atol=1e-12)


class _Basis:
    """Synthetic basis: 2 modes, 3 nodes."""
    def __init__(self):
        self.freq = np.array([10.0, 30.0])
        self.shapes = np.zeros((2, 3, 3))
        self.shapes[0, :, 2] = [0.0, 0.5, 1.0]
        self.shapes[1, :, 2] = [0.0, -0.8, 0.6]
        self.stress = np.zeros((2, 3, 6), np.float32)
        self.stress[0, :, 0] = [100.0, 50.0, 0.0]
        self.stress[1, :, 0] = [-40.0, 30.0, 0.0]
        self.gamma = np.zeros((2, 6))
        self.gamma[:, 2] = [0.7, 0.2]
        self.rf_sum = np.zeros((2, 3))

    @property
    def omega(self):
        return 2 * np.pi * self.freq

    @property
    def K(self):
        return 2


def test_harmonic_base_single_mode_analytic():
    b = _Basis()
    terms = R.Terms()
    e = Excitation(kind="base_acc", dof=3, unit="mm/s2")
    terms.items.append({"exc": e, "kind": "base_acc", "scale": 1.0, "phase": 0.0, "p": -b.gamma[:, 2],
                        "dir": np.array([0, 0, 1.0])})
    f = np.array([5.0, 10.0, 20.0])
    z = np.array([0.02, 0.02])
    hr = R.harmonic(b, terms, f, z, None, residual=False)
    w = 2 * np.pi * f
    for j in range(2):
        wj = b.omega[j]
        H = 1 / (wj ** 2 - w ** 2 + 2j * z[j] * wj * w)
        assert np.allclose(hr["q"][:, j], -b.gamma[j, 2] * H)
    assert np.allclose(hr["ub"][:, 2], -1.0 / w ** 2)      # u_b = −a_b/ω²


def test_max_over_phase_exact():
    rng = np.random.default_rng(1)
    a, bvec = rng.normal(size=(5, 6)), rng.normal(size=(5, 6))
    phi = np.linspace(0, 2 * np.pi, 20001)
    s = a[:, None, :] * np.cos(phi)[None, :, None] + bvec[:, None, :] * np.sin(phi)[None, :, None]
    vm = np.sqrt(np.einsum("npi,ij,npj->np", s, R.P_VM, s))
    assert np.allclose(R.max_over_phase(a, bvec, R.P_VM), vm.max(axis=1), rtol=1e-7)


def test_freq_grid_contains_resonances():
    f = R.freq_grid(1.0, 100.0, np.array([10.0, 50.0, 150.0]), np.array([0.02, 0.02, 0.02]), 5, 3.0)
    for x in (10.0, 50.0, 10 * 0.98, 10 * 1.02):
        assert np.min(np.abs(f - x)) < 1e-9
    assert f[0] == 1.0 and f[-1] == 100.0 and np.all(np.diff(f) > 0)


def test_time_terms_base_disp_hold():
    terms = R.Terms()
    e = Excitation(kind="base_disp", dof=1, unit="mm", table=[[0.0, 0.0], [1.0, 1.0]])
    terms.items.append({"exc": e, "kind": "base_disp", "scale": 1.0, "phase": 0.0, "p": np.array([-1.0]),
                        "dir": np.array([1.0, 0, 0])})
    t = np.linspace(0, 2, 21)
    P, C, ab, ub = R.time_terms(terms, t)
    assert np.isclose(ub[-1, 0], 1.0)                    # the displacement holds the last value
    dt = t[1] - t[0]
    assert np.isclose(ab[0, 0], 0.1 / dt ** 2)           # d(−Δt) = d(0) → a velocity jump in the first step
    assert np.allclose(ab[2:10, 0], 0.0)


def test_zetas_rayleigh():
    from fea_core.dynamics.model import DampingSpec
    d = DampingSpec(kind="rayleigh", zeta=0.03, f1=10.0, f2=100.0)
    z = zetas(d, np.array([10.0, 100.0, 31.6227766]))
    assert np.allclose(z[:2], 0.03)
    assert z[2] < 0.03


def test_check_base_symmetry_warning_and_errors():
    st = _study()
    st.constraints = [Fixed("a"), Displacement("sym", uy=0.0)]
    ds = st.dynamics
    ds.enabled = True
    ds.harmonic.enabled = True
    ds.harmonic.f_min, ds.harmonic.f_max = 100.0, 10.0
    ds.harmonic.excitation = [Excitation(kind="base_acc", dof=2)]
    e, w = check(st, [])
    assert any("f_min" in x for x in e)
    assert any("symmetry plane" in x for x in w)


def test_dynamics_results_save_load(tmp_path):
    from fea_core.dynamics.results import DynamicsResults
    b = _Basis()
    r = DynamicsResults(b.freq, np.array([0.02, 0.02]), b.shapes.astype(np.float32), b.stress, b.gamma,
                        np.zeros((2, 6)), b.rf_sum, {"x": 1.0}, 1e-3)
    r.harmonic = {"f": np.array([1.0, 2.0]), "q": np.ones((2, 2), complex), "c": np.zeros((2, 0), complex),
                  "ub": np.zeros((2, 3), complex), "res_index": np.zeros(0, np.int64),
                  "probes": {"p": {"U": np.ones((2, 3), complex)}}, "env_U": np.ones(3, np.float32),
                  "env_S": np.ones(3, np.float32), "env_S_at": np.zeros(3, np.int32)}
    p = tmp_path / "d.npz"
    r.save(p)
    r2 = DynamicsResults.load(p)
    assert np.allclose(r2.harmonic["q"], 1.0) and r2.harmonic["probes"]["p"]["U"].dtype.kind == "c"
    U, S = r2.harmonic_field(0, 0.0)
    assert np.allclose(U[:, 2], b.shapes[0, :, 2] + b.shapes[1, :, 2])
    assert r2.summary()["modes"] == 2


def test_inertia_load_matches_rotation():
    """Inertia with ω along the axis = Rotation (centrifugal force) and with α = Rotation α (inertial force)."""
    from fea_core.mesh import Mesh
    from fea_core.study import Inertia, Rotation
    from fea_core.loads import inertia_forces, rotation_forces
    X = np.array([[1, 0, 0], [3, 0, 0], [1, 2, 0], [1, 0, 2]], float)
    mids = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    nodes = np.vstack([X, [(X[a] + X[b]) / 2 for a, b in mids]])
    mesh = Mesh(nodes, np.arange(10)[None, :], np.zeros(1, int), ["A"])
    st = _study()
    Fi = inertia_forces(st, mesh, Inertia(omega=[0, 0, 50.0], alpha=[0, 0, 7.0]))
    Fr = rotation_forces(st, mesh, Rotation(axis=[0, 0, 1.0], omega=50.0, alpha=7.0))
    assert np.allclose(Fi, Fr, rtol=1e-12, atol=1e-18)


def test_envelope_and_sweep_kernels_match_numpy():
    """The numba core (dynamics._env) = the numpy path in batches: envelopes (complex and real, with residual vectors)
    and the sum of the damage from the sweep.
    """
    pytest.importorskip("numba")
    from types import SimpleNamespace
    from fea_core.dynamics import _env
    from fea_core.fatigue.spectral import inverse_life
    rng = np.random.default_rng(3)
    K, N, n, Rn = 5, 40, 30, 2
    basis = SimpleNamespace(K=K, shapes=rng.normal(size=(K, N, 3)).astype(np.float32),
                            stress=(100 * rng.normal(size=(K, N, 6))).astype(np.float32))
    res = {"U": rng.normal(size=(Rn, N, 3)).astype(np.float32), "S": (50 * rng.normal(size=(Rn, N, 6))).astype(np.float32)}
    Q = rng.normal(size=(n, K)) + 1j * rng.normal(size=(n, K))
    C = rng.normal(size=(n, Rn)) + 1j * rng.normal(size=(n, Rn))
    for cplx, q, c in ((True, Q, C), (False, Q.real, C.real)):
        a = R.envelopes(basis, res, q, c, cplx)                  # numba
        b = R.envelopes(basis, res, q, c, cplx, chunk=7)         # numpy in batches
        for k in ("U", "S"):
            assert np.allclose(a[k], b[k], rtol=2e-5)
            assert np.array_equal(a[k + "_at"], b[k + "_at"])
    sn = SimpleNamespace(SD=80.0, ND=2e6, k=5.0, k2=9.0)
    w = rng.random(n)
    ss = basis.stress.reshape(K, -1)
    sig = R.stress_amplitudes(ss, Q, C, res["S"].reshape(Rn, -1), True)
    ref = w @ inverse_life(sig, sn)
    par = np.array([[80.0], [2e6], [5.0], [9.0]]) * np.ones((4, N))
    got = _env.sweep_sum(basis.stress, res["S"], Q, C, w, *par)
    assert np.allclose(got, ref, rtol=1e-4)

def test_base_disp_analytic_ramp_and_hold():
    """Analytic check of 1-DOF: the ramp-and-hold of the base displacement has a known closed-form solution."""
    from fea_core.dynamics.modal_ode import integrate

    w = 20.0
    zeta = 0.0
    d0 = 0.05
    t_ramp = 0.2
    v0 = d0 / t_ramp

    def q_exact(t):
        res = np.zeros_like(t)
        m1 = t <= t_ramp
        res[m1] = - (v0 / w) * np.sin(w * t[m1])
        m2 = t > t_ramp
        tau = t[m2] - t_ramp
        q_tr = - (v0 / w) * np.sin(w * t_ramp)
        v_tr = v0 * (1.0 - np.cos(w * t_ramp))
        res[m2] = q_tr * np.cos(w * tau) + (v_tr / w) * np.sin(w * tau)
        return res

    dt = 0.01
    t = np.arange(0.0, 1.0 + 0.5 * dt, dt)
    P = np.zeros((len(t), 1))
    omega = np.array([w])
    z = np.array([zeta])
    impulses = [(0.0, np.array([-v0])), (t_ramp, np.array([v0]))]

    q, _, _ = integrate(dt, P, omega, z, impulses=impulses)
    qe = q_exact(t)
    assert np.allclose(q[:, 0], qe, atol=1e-12)


def test_base_disp_dt_independence():
    """Independence of the output time step: half dt changes the result < 1e-6."""
    from fea_core.dynamics.modal_ode import integrate

    w = 35.0
    z = np.array([0.02])
    d0 = 0.1
    t_ramp = 0.15
    v0 = d0 / t_ramp
    impulses = [(0.0, np.array([-v0])), (t_ramp, np.array([v0]))]

    dt1 = 0.005
    t1 = np.arange(0.0, 0.5 + 0.5 * dt1, dt1)
    P1 = np.zeros((len(t1), 1))
    q1, _, _ = integrate(dt1, P1, np.array([w]), z, impulses=impulses)

    dt2 = dt1 / 2
    t2 = np.arange(0.0, 0.5 + 0.5 * dt2, dt2)
    P2 = np.zeros((len(t2), 1))
    q2, _, _ = integrate(dt2, P2, np.array([w]), z, impulses=impulses)

    diff = np.abs(q1[:, 0] - q2[::2, 0])
    assert diff.max() < 1e-6


def test_prestressed_modal_reaction_equilibrium(tmp_path):
    """Balance of the modal reactions after the preload: K_total·φ consistent with −ω²·Σ M·φ."""
    from verification import cases as _  # noqa: F401 - avoid a cyclic import
    from verification.dynamics_cases import y9_build
    from fea_core.solve import run_study

    wd = tmp_path / "y9"
    wd.mkdir(parents=True, exist_ok=True)
    st, _ = y9_build(wd)
    st.dynamics.verify_ccx = False
    res = run_study(st, tmp_path / "y9" / "ccx")
    assert res.dynamics.checks["mode_equilibrium"] < 1e-3
    assert not any("modal reaction equilibrium" in w for w in res.dynamics.warnings)

