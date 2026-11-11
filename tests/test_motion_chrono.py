"""The Chrono solver in the mechanism (F8.9 C1 – fea_core.motion.chrono_worker in the environment tools/chrono_env): the same results
as Exudyn for the joints without contact (pendulum, crank), checks of the model (what Chrono does not support), contact in the summary and series.
Without the chrono module the calculation tests are skipped; the contact cases are verified by `python -m verification.run_all MB4C MB4R KK5`.
"""
import math

import numpy as np
import pytest

pytest.importorskip("exudyn")

from fea_core.motion.model import Body, MotionJoint, JointFriction, Driver, MotionLoad, MotionRun, MotionContact  # noqa: E402
from fea_core.motion.chrono_run import available  # noqa: E402
from tests.test_motion import _study, _crank, G  # noqa: E402

needs_chrono = pytest.mark.skipif(not available(), reason="modul chrono (tools/chrono_env) není nainstalován")


def _pendulum(st, solver):
    m, L = 2e-3, 500.0
    M = st.mechanism
    M.gravity = [0, -G, 0]
    M.bodies = [Body("rám", ground=True), Body("kyvadlo", mass=m, com=[L * math.sin(1.0), -L * math.cos(1.0), 0],
                                              inertia=(np.eye(3) * 1e-2).tolist())]
    M.joints = [MotionJoint("čep", "revolute", "rám", "kyvadlo", [0, 0, 0], [0, 0, 1])]
    M.run = MotionRun("dynamic", t_end=0.3, outputs=30, dt=2e-5, solver=solver)


def test_check_model_chrono_limits():
    from fea_core.motion.run import check_model
    st = _study()
    _pendulum(st, "exudyn")
    st.mechanism.contacts = [MotionContact("rám", "kyvadlo")]
    errs, _ = check_model(st)
    assert any("only by the Chrono solver" in e for e in errs)
    assert any("has no CAD parts" in e for e in errs)                      # the contact needs the shape from the CAD
    st.mechanism.contacts = []
    st.mechanism.run.solver = "chrono"
    st.mechanism.joints[0].friction = JointFriction("coulomb", mu_k=0.1, r_eff=5.0)
    st.mechanism.bodies[1].flexible = True
    st.mechanism.run.kind = "static"
    errs, _ = check_model(st)
    txt = " | ".join(errs)
    assert "friction and stops" in txt and "Craig–Bampton" in txt and "dynamic and kinematic" in txt


def test_contact_json_roundtrip():
    from fea_core.study import Study
    st = _study()
    _pendulum(st, "chrono")
    st.mechanism.contacts = [MotionContact("rám", "kyvadlo", mu=0.2, method="smc", young=2000.0, deflection=0.005)]
    st2 = Study.from_dict(st.to_dict())
    assert st2.mechanism.contacts == st.mechanism.contacts and st2.mechanism.run.solver == "chrono"


@needs_chrono
def test_chrono_pendulum_matches_exudyn(tmp_path):
    from fea_core.motion.run import run_motion
    res = {}
    for solver in ("exudyn", "chrono"):
        st = _study()
        _pendulum(st, solver)
        res[solver] = run_motion(st, tmp_path / solver)
    a, b = res["exudyn"].arrays, res["chrono"].arrays
    assert res["chrono"].meta["solver"] == "chrono"
    assert res["chrono"].meta["equilibrium"]["max_rel"] < 5e-4       # Euler: order of the step (dt 2e-5 → 8e-5)
    # Euler implicit (Chrono) × generalized-α (Exudyn): position 1e-3, reaction 2e-3 (order of the step)
    assert np.abs(b["pos"][:, 1] - a["pos"][:, 1]).max() < 1e-3 * 500
    Fa, Fb = np.linalg.norm(a["jF"][:, 0], axis=1), np.linalg.norm(b["jF"][:, 0], axis=1)
    assert np.abs(Fb - Fa).max() < 2e-3 * Fa.max()


@needs_chrono
def test_chrono_crank_kinematic(tmp_path):
    from fea_core.motion.run import run_motion
    st = _study()
    r, l, w = _crank(st)
    for b in st.mechanism.bodies[1:]:                             # Chrono: reasonable masses (1e-9 t is badly conditioned)
        b.mass = 1e-4
        b.inertia = (np.eye(3) * 1e-2).tolist()
    st.mechanism.joints[3].kind = "cylindrical"                   # no redundancy of the planar mechanism in 3D
    st.mechanism.joints[2].kind = "spherical"
    st.mechanism.run.solver = "chrono"
    st.mechanism.run.dt = 1e-5
    res = run_motion(st, tmp_path)
    k = 6
    phi = math.radians(60)
    x = r * math.cos(phi) + math.sqrt(l * l - (r * math.sin(phi)) ** 2)
    assert res.arrays["pos"][k, 3, 0] == pytest.approx(x, rel=1e-6)
    assert res.arrays["jq"][k, 0, 0] == pytest.approx(phi, rel=1e-9)
    assert res.meta["equilibrium"]["max_rel"] < 1e-3
    v = res.arrays["vel"][k, 3, 0]
    M_need = -(-1000.0) * v / w                                    # M·ω = −F·v (small masses)
    assert res.arrays["dQ"][k, 0] == pytest.approx(M_need, rel=2e-2)
