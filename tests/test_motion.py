"""Mechanisms (stage F8.3, F8.8 – fea_core.motion): fast tests without CalculiX and Gmsh.

Model JSON, drive courses, check of the mobility (Exudyn in the worker process), pendulum and crank (equilibrium of the bodies,
power balance), transfer to the FEM (load state with Inertia and the remote constraint).
"""
import math

import numpy as np
import pytest

pytest.importorskip("exudyn")

from fea_core.motion.laws import Law, DriverMotion  # noqa: E402
from fea_core.motion.model import (Body, MotionJoint, JointFriction, Driver, MotionLoad, MotionRun,  # noqa: E402
                                   ForceElement, Mechanism)
from fea_core.study import Study, Material  # noqa: E402

G = 9810.0


def _study(name="test"):
    st = Study(name, "")
    st.add_material(Material("Ocel", 210000.0, 0.3, 7.85e-9), default=True)
    return st


def _crank(st, outputs=36):
    r, l, w = 20.0, 80.0, 100.0
    m = 1e-9
    I3 = (np.eye(3) * 1e-9).tolist()
    M = st.mechanism
    M.gravity = None
    M.bodies = [Body("rám", ground=True), Body("klika", mass=m, com=[r / 2, 0, 0], inertia=I3),
                Body("ojnice", mass=m, com=[r + l / 2, 0, 0], inertia=I3), Body("píst", mass=m, com=[r + l, 0, 0], inertia=I3)]
    M.joints = [MotionJoint("J1", "revolute", "rám", "klika", [0, 0, 0], [0, 0, 1]),
                MotionJoint("J2", "revolute", "klika", "ojnice", [r, 0, 0], [0, 0, 1], selection_b="oko A"),
                MotionJoint("J3", "revolute", "ojnice", "píst", [r + l, 0, 0], [0, 0, 1], selection_a="oko B"),
                MotionJoint("J4", "prismatic", "rám", "píst", [r + l, 0, 0], [1, 0, 0])]
    M.drivers = [Driver("J1", "angular_velocity", "constant", [w], name="motor")]
    M.loads = [MotionLoad("píst", [r + l, 0, 0], [-1000.0, 0, 0], name="F")]
    M.run = MotionRun("kinematic", t_end=2 * math.pi / w, outputs=outputs)
    return r, l, w


# ------------------------------------------------------------------ model and laws
def test_mechanism_json_roundtrip():
    st = _study()
    _crank(st)
    st.mechanism.joints[0].friction = JointFriction("stribeck", mu_s=0.3, mu_k=0.2, r_eff=4.0)
    st.mechanism.forces = [ForceElement("spring", "spring", "rám", "píst", [0, 0, 0], [100, 0, 0], k=2.0,
                                        selection_b="oko B")]
    d = st.to_dict()
    assert "mechanism" in d["features"]
    st2 = Study.from_dict(d)
    assert st2.mechanism == st.mechanism
    assert isinstance(st2.mechanism.joints[0].friction, JointFriction)
    assert Mechanism.from_dict(st.mechanism.to_dict()) == st.mechanism


def test_driver_laws():
    sc = Law("scurve", [100.0, 50.0, 200.0, 2000.0, 0.5])
    T = sc.duration
    assert sc.value(0.0) == 0.0 and sc.value(0.5) == 0.0
    assert sc.value(0.5 + T + 1) == pytest.approx(100.0, rel=1e-12)
    ts = np.linspace(0.5, 0.5 + T, 2001)
    assert max(sc.d1(t) for t in ts) <= 50.0 * (1 + 1e-12)
    assert max(abs(sc.d2(t)) for t in ts) <= 200.0 * (1 + 1e-12)
    # short stroke: v_max is not reached, the profile still ends exactly at the stroke
    sc2 = Law("scurve", [1.0, 50.0, 200.0, 2000.0])
    assert sc2.value(sc2.duration) == pytest.approx(1.0, rel=1e-12)
    tab = Law("table", [[0, 0], [1, 2], [2, 2]])
    assert tab.integral(2.0) == pytest.approx(1.0 + 2.0)
    assert DriverMotion("velocity", tab).coord(1.5) == pytest.approx(1.0 + 1.0)
    p = Law("poly", [1.0, 2.0, 3.0], scale=1e-3)
    assert p.d2(5.0) == pytest.approx(6e-3) and p.integral(1.0) == pytest.approx((1 + 1 + 1) * 1e-3)
    s = Law("sine", [2.0, 1.0, 0.0, 0.0])
    assert s.integral(1.0) == pytest.approx(0.0, abs=1e-12)


def test_restitution_inverse():
    from fea_core.motion.run import restitution_kv, zeta_for_restitution
    for e in (0.1, 0.5, 0.8, 0.99):
        assert restitution_kv(zeta_for_restitution(e)) == pytest.approx(e, rel=1e-9)


# ------------------------------------------------------------------ worker (Exudyn)
def test_check_dof_planar_fourbar(tmp_path):
    from fea_core.motion.run import check_dof
    st = _study()
    _crank(st)
    d = check_dof(st, None, workdir=tmp_path)
    assert d["grubler"] == -2 and d["dof_free"] == 1 and d["redundant"] == 3 and d["dof_driven"] == 0
    assert any("AMBIGUOUS" in m for m in d["messages"])


def test_crank_kinematic(tmp_path):
    from fea_core.motion.run import run_motion
    st = _study()
    r, l, w = _crank(st)
    res = run_motion(st, tmp_path)
    k = 6                                                   # 36 outputs per revolution → 60°
    phi = math.radians(60)
    x = r * math.cos(phi) + math.sqrt(l * l - (r * math.sin(phi)) ** 2)
    assert res.arrays["pos"][k, 3, 0] == pytest.approx(x, rel=1e-9)
    assert res.meta["equilibrium"]["max_rel"] < 1e-8
    assert res.meta["power_balance"]["max_rel"] < 1e-6
    v = res.arrays["vel"][k, 3, 0]
    assert res.arrays["dQ"][k, 0] == pytest.approx(-(-1000.0) * v / w, rel=1e-4)   # M·ω = −F·v
    mx = res.find_max("reakce:J3")
    assert mx["value"] == pytest.approx(np.linalg.norm(res.arrays["jF"][:, 2], axis=1).max())
    s = res.series("moment pohonu:motor")
    assert s["unit"] == "N·mm" and s["values"].shape == (37, 1)


def test_pendulum_dynamic_with_friction(tmp_path):
    from fea_core.motion.run import run_motion
    st = _study()
    m, L = 2e-3, 500.0
    M = st.mechanism
    M.gravity = [0, -G, 0]
    M.bodies = [Body("rám", ground=True), Body("kyvadlo", mass=m, com=[L * math.sin(1.0), -L * math.cos(1.0), 0],
                                              inertia=(np.eye(3) * 1e-2).tolist())]
    M.joints = [MotionJoint("čep", "revolute", "rám", "kyvadlo", [0, 0, 0], [0, 0, 1],
                            friction=JointFriction("coulomb", mu_k=0.2, r_eff=10.0))]
    M.run = MotionRun("dynamic", t_end=0.4, outputs=40, dt=2e-4)
    res = run_motion(st, tmp_path)
    a = res.arrays
    assert res.meta["equilibrium"]["max_rel"] < 1e-7
    pb = res.meta["power_balance"]
    assert pb["max_rel"] < 1e-6 and pb["integral_rel"] < 1e-4
    assert a["wfr"][-1] < 0                                # friction is a loss
    E = a["ekin"] + a["epot"]
    assert E[-1] - E[0] == pytest.approx(a["wfr"][-1], rel=1e-4)


# ------------------------------------------------------------------ transfer to the FEM
def _tet10_mesh():
    from fea_core.mesh import Mesh
    c = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], float)
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]                 # order of the nodes C3D10
    nodes = np.vstack([c, [(c[i] + c[j]) / 2 for i, j in edges]])
    return Mesh(nodes, np.arange(10)[None, :], np.zeros(1, int), ["Díl"])


def test_inertia_unit_states_superpose():
    """Inertia: ω⊗ω decomposed into the 6 unit matrices W = the same as the direct calculation with ω; the resultant = −m·a_c."""
    from fea_core.loads import inertia_forces
    from fea_core.study import Inertia
    st = _study()
    mesh = _tet10_mesh()
    om, al, a0, p = np.array([3.0, -2.0, 5.0]), np.array([10.0, 0.5, -4.0]), np.array([100.0, -50.0, 20.0]), \
        np.array([1.0, 2.0, 3.0])
    direct = inertia_forces(st, mesh, Inertia(point=list(p), accel=list(a0), omega=list(om), alpha=list(al)))
    W = np.outer(om, om) - om @ om * np.eye(3)
    sup = inertia_forces(st, mesh, Inertia(point=list(p), accel=list(a0), alpha=list(al)))
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        E = np.zeros((3, 3))
        E[i, j] = E[j, i] = 1.0
        sup += W[i, j] * inertia_forces(st, mesh, Inertia(point=list(p), wmat=E.tolist()))
    assert np.abs(sup - direct).max() <= 1e-12 * np.abs(direct).max()
    V = 1000.0 / 6
    m = 7.85e-9 * V
    c = np.array([2.5, 2.5, 2.5])
    r = c - p
    acc = a0 + np.cross(al, r) + np.cross(om, np.cross(om, r))
    assert direct.sum(axis=0) == pytest.approx(-m * acc, rel=1e-10)


def test_to_fea_instant_and_history(tmp_path):
    from fea_core.motion.run import run_motion
    from fea_core.motion.to_fea import instant_loads, history_loads
    from fea_core.selectors import FaceSelector
    from fea_core.study import Inertia, RemoteLoad, RemoteSupport
    st = _study()
    _crank(st)
    fs = FaceSelector("Ojnice", "cylinder", 1.0, [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0, 1, 1, 1])
    st.add_selection("oko A", [fs])
    st.add_selection("oko B", [fs])
    st.mechanism.bodies[2].parts = ["Ojnice"]              # connecting-rod part (FEM) – the mass stays as given
    res = run_motion(st, tmp_path)
    k = 9                                                  # 90°
    info = instant_loads(st, res, k, body="ojnice")
    lc = next(x for x in st.load_cases if x.name == info["load_case"])
    rl = [x for x in lc.loads if isinstance(x, RemoteLoad)]
    inr = [x for x in lc.loads if isinstance(x, Inertia)]
    assert len(rl) == 1 and rl[0].selection == "oko B" and rl[0].coupling == "distributing"
    R = res.arrays["rot"][k, 2]
    F_on_rod = -res.arrays["jF"][k, 2]                     # the connecting rod is body_a of the joint J3
    assert np.allclose(rl[0].force, R.T @ F_on_rod, rtol=1e-12, atol=1e-9)
    assert inr and inr[0].parts == ["Ojnice"]
    assert np.allclose(inr[0].accel, R.T @ res.arrays["acc"][k, 2], rtol=1e-10, atol=1e-9)
    sup = [c for c in st.constraints if isinstance(c, RemoteSupport)]
    assert len(sup) == 1 and sup[0].selection == "oko A" and all(sup[0].dofs) and sup[0].coupling == "distributing"
    assert "oko A" in info["expected"]
    h = history_loads(st, res, body="ojnice")
    assert h["n_states"] == len(st.fatigue.history.channels)
    assert 0 < h["n_states"] <= 6 + 12
    names = {c.load_case for c in st.fatigue.history.channels}
    assert all(n in [x.name for x in st.load_cases] for n in names)


def test_motion_result_project_roundtrip(tmp_path):
    from fea_core.motion.results import MotionResult
    from fea_core.project import save_project, load_project, load_motion
    a = {"t": np.linspace(0, 1, 5), "pos": np.zeros((5, 1, 3))}
    res = MotionResult(a, {"bodies": [{"name": "A", "ground": False}], "kind": "dynamic"})
    st = _study("projekt")
    save_project(tmp_path / "p.fea", st, "", motion=res)
    study, step, results = load_project(tmp_path / "p.fea", tmp_path / "wd")
    m = load_motion(tmp_path / "wd")
    assert step is None and results is None and m is not None
    assert np.allclose(m.arrays["t"], a["t"]) and m.meta["bodies"][0]["name"] == "A"


# ------------------------------------------------------------------ proposal of the joints from the CAD
def _pin_parts():
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from fea_core.geometry import Part
    from verification.cases import box, cyl, solid
    plate = solid(BRepAlgoAPI_Cut(box(0, 0, 0, 60, 60, 10), cyl((30, 30, -1), (0, 0, 1), 5, 12)).Shape())
    lever = solid(BRepAlgoAPI_Fuse(box(25, 25, 10, 105, 10, 10), cyl((30, 30, 0), (0, 0, 1), 5, 20)).Shape())
    return [Part("Deska", plate), Part("Páka", lever)]


def test_joints_from_cad_pin_in_hole():
    from fea_core.motion.joints_cad import suggest_from_geometry, apply_suggestion
    parts = _pin_parts()
    st = _study()
    st.mechanism.bodies = [Body("rám", ["Deska"], ground=True), Body("páka", ["Páka"])]
    sug = suggest_from_geometry(st, parts)
    assert len(sug) == 1
    s = sug[0]
    j = s.joint
    assert j.kind == "revolute" and j.body_a == "rám" and j.body_b == "páka" and s.confidence >= 0.7
    assert abs(abs(j.axis[2]) - 1) < 1e-9 and np.allclose(j.origin[:2], [30, 30])
    j2 = apply_suggestion(st, parts, s)
    assert j2.selection_a in st.selections and j2.selection_b in st.selections
    assert st.selections[j2.selection_a].faces[0].part == "Deska"


def test_joints_from_fusion_metadata():
    from fea_core.motion.joints_cad import suggest_from_metadata
    st = _study()
    st.mechanism.bodies = [Body("rám", ["Deska"], ground=True), Body("páka", ["Páka"])]
    meta = {"joints": [{"name": "Rev1", "type": "revolute", "occurrence_one": "Deska:1", "occurrence_two": "Páka:1",
                        "origin": [3, 3, 1], "axis": [0, 0, 1], "limits": {"min": -90, "max": 90},
                        "units": {"length": "cm", "angle": "deg"}}]}
    s = suggest_from_metadata(st, meta)[0]
    assert s.joint.body_a == "rám" and s.joint.body_b == "páka" and s.joint.kind == "revolute"
    assert s.joint.origin == [30.0, 30.0, 10.0]
    assert s.joint.limits == pytest.approx([-math.pi / 2, math.pi / 2])
