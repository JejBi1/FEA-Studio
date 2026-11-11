"""Frame system (fea_core.beams): model ↔ JSON, discretisation, consistent loads, fast calculation of the cantilever
with the own solver (beams/timoshenko.py) and in ccx (element U1, without Gmsh, ~1 s), assessment EN 1993-1-1 from the result,
recognition of a prismatic body.
"""
import math

import numpy as np
import pytest

from fea_core.beams.fem import build, _consistent, FrameError
from fea_core.beams.model import FrameMember, FrameSupport, SectionDef, FrameNodeLoad, FrameMemberLoad
from fea_core.study import Study, Material, LoadCase

E = 210000.0


def _cantilever(**kw):
    st = Study("t", "")
    st.add_material(Material("Ocel S235", E, 0.3, 7.85e-9, Re=235.0), default=True)
    fr = st.frame
    fr.sections["IPE200"] = SectionDef("IPE200", library="IPE 200")
    fr.add_node("A", [0, 0, 0])
    fr.add_node("B", [2000, 0, 0])
    fr.members["m"] = FrameMember("m", "A", "B", "IPE200", **kw)
    fr.supports.append(FrameSupport("A"))
    return st


def test_json_roundtrip():
    st = _cantilever(through=None, releases_end=[False, False, False, False, True, False])
    st.load_cases.append(LoadCase("q", [FrameMemberLoad("m", [0, 0, -1.0], [0, 0, -2.0]), FrameNodeLoad("B", [1, 2, 3])]))
    st2 = Study.from_dict(st.to_dict())
    assert st2.frame.members["m"].releases_end[4] and st2.frame.sections["IPE200"].library == "IPE 200"
    assert type(st2.load_cases[0].loads[0]).__name__ == "FrameMemberLoad"
    assert st2.load_cases[0].loads[0].q_end == [0, 0, -2.0]


def test_build_split_and_release():
    st = _cantilever(releases_end=[False, False, False, False, True, False])
    fe = build(st, solver="ccx_u1")
    assert len(fe.segs) == st.frame.settings.segments_straight
    assert all(sg.bprime is not None for sg in fe.segs)               # IPE: J ≪ Ip → a pair of elements
    assert fe.end_node[("m", "end")] != fe.joint["B"]                   # hinge → its own end node
    with pytest.raises(FrameError):
        build(_cantilever(releases_start=[False, False, False, True, False, False],
                          releases_end=[False, False, False, True, False, False]))


def test_build_own_solver_is_lean():
    """Own solver: 1 element per straight member, a hinge = a flag of the release of the element (no nodes E / B', no equations)."""
    st = _cantilever(releases_end=[False, False, False, False, True, False])
    fe = build(st)
    assert len(fe.segs) == st.frame.settings.segments_static == 1
    assert fe.segs[0].bprime is None and not fe.eqs and len(fe.xyz) == 2
    assert fe.segs[0].rel.tolist() == [False] * 10 + [True, False]
    assert len(build(st, segments=8).segs) == 8
    assert build(st, solver="ccx_u1").segs[0].bprime is not None
    st.frame.settings.solver = "ccx_u1"
    assert len(build(st).segs) == st.frame.settings.segments_straight          # choice of the solver
    st2 = _cantilever()
    st2.geometry = "model.step"                                                 # mixed model → always U1
    assert len(build(st2).segs) == st2.frame.settings.segments_straight


def test_consistent_load_equivalence():
    R = np.eye(3)
    qa, qb = np.array([0, 1.0, -2.0]), np.array([0.5, -3.0, 4.0])
    f = _consistent(qa, qb, 100.0, R)
    F = 100 * (qa + qb) / 2
    assert np.allclose(f[:3] + f[6:9], F)
    # moment about the node a: f_b·L + m_a + m_b = ∫ s·q ds
    Ma = np.cross([100.0, 0, 0], f[6:9]) + f[3:6] + f[9:12]
    assert np.allclose(Ma, np.cross([1.0, 0, 0], 100 ** 2 * (qa + 2 * qb) / 6))


@pytest.mark.parametrize("solver", ["timoshenko", "ccx_u1"])
def test_cantilever_ccx(tmp_path, solver):
    from fea_core.solve import run_study
    from fea_core.beams import sections as S
    from fea_core.beams.assess import assess
    st = _cantilever()
    st.frame.settings.solver = solver
    st.load_cases.append(LoadCase("F", [FrameNodeLoad("B", [0, 0, -1000.0])]))
    res = run_study(st, tmp_path)
    fr = res.step(0).frame
    sec = S.get("IPE 200")
    assert fr["members"]["m"]["My"][0] == pytest.approx(2e6, rel=1e-4)
    assert fr["reactions"]["A"][2] == pytest.approx(1000.0, rel=1e-4)
    G = E / 2.6
    exact = 1000 * 2000 ** 3 / (3 * E * sec.Iy) + 1000 * 2000 / (G * sec.Av_z)
    assert -fr["nodes"]["B"][2] == pytest.approx(exact, rel=1e-9 if solver == "timoshenko" else 0.03)
    assert res.info["backend"] == ("timoshenko" if solver == "timoshenko" else "ccx")
    assert res.step(0).equilibrium["imbalance"] < 1e-4
    a = assess(st, res)
    assert 0 < a["m"]["utilization"] < 1 and a["_max"][1] == "m"
    from fea_core.results import Results
    r2 = Results.load(tmp_path / "results.npz")
    assert r2.step(0).frame["members"]["m"]["My"][0] == pytest.approx(2e6, rel=1e-4)
    assert "frame" in r2.info


def test_prismatic_recognition(tmp_path):
    from verification.cases import box
    from fea_core.geometry import Part, write_step, read_step
    from fea_core.beams.geometry import prismatic
    write_step([Part("Tyc", box(0, 0, 0, 500, 20, 10)), Part("Kostka", box(0, 0, 0, 10, 10, 10))], tmp_path / "m.step")
    parts = read_step(tmp_path / "m.step")
    pr = prismatic(parts[0])
    assert pr is not None and pr["length"] == pytest.approx(500.0)
    assert np.allclose(pr["start"], [0, 10, 5]) or np.allclose(pr["end"], [0, 10, 5])
    assert prismatic(parts[1])["length"] == pytest.approx(10.0)
