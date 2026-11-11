"""Fast tests of the remote constraint (fea_core.remote, writing of the .inp, reading of the reactions from the .dat) – without Gmsh and ccx.
The calculation with ccx is verified by the case V10 (verification/remote_cases.py).
"""
import numpy as np
import pytest

from fea_core.ccx.inp import write_inp
from fea_core.remote import (parse_dofs, dofs_text, rbe3, point_motion, distributing_equations, rigid_rows,
                             RemoteError)
from fea_core.selectors import FaceSelector
from fea_core.study import Study, Material, Displacement, RemoteSupport, Force, LoadCase, Selection
from tests.test_thermal import bar_mesh


def _study(constraints) -> Study:
    st = Study("R", "x.step")
    st.add_material(Material("Ocel", 210000.0, 0.3), default=True)
    sel = FaceSelector("Dil1", "plane", 1.0, [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0, 0, 0, 0])
    for name in ("x0", "x100", "y0"):
        st.selections[name] = Selection(name, [sel])
    st.constraints = list(constraints)
    st.load_cases = [LoadCase("F", [Force("x100", [0, 0, -100.0])])]
    return st


def test_parse_dofs():
    assert parse_dofs("ux,uy,uz") == [True, True, True, False, False, False]
    assert parse_dofs(["rz", "UX"]) == [True, False, False, False, False, True]
    assert parse_dofs([1, 0, 1, 0, 1, 0]) == [True, False, True, False, True, False]
    assert parse_dofs("all") == [True] * 6 and parse_dofs("posuvy") == [True] * 3 + [False] * 3
    assert dofs_text([True, False, False, False, False, True]) == "ux, rz" and dofs_text([False] * 6) == "none"
    with pytest.raises(RemoteError):
        parse_dofs("ux,uq")


def test_rbe3_reproduces_rigid_motion():
    """The RBE3 kinematics returns exactly the displacement and rotation of the point for a rigid motion of the face (also a point off the face)."""
    m = bar_mesh()
    rng = np.random.default_rng(3)
    for p in ([0.0, 5.0, 5.0], [-50.0, 12.0, -3.0]):
        t, w = rng.normal(size=3), 1e-3 * rng.normal(size=3)
        U = t + np.cross(w, m.nodes - p)
        assert np.allclose(point_motion(m, "x0", p, U), np.concatenate([t, w]), atol=1e-12)
    op = rbe3(m, "x0", [0, 5, 5])
    assert np.allclose(op["c"], [0, 5, 5]) and op["W"] == pytest.approx(100.0)       # weights = area of the face


def test_distributing_equations_rref():
    """Dependent DOFs: different, outside *BOUNDARY, each only in its own equation; the equations = the constrained components of RBE3 (rank),
    the rigid motion in the free components satisfies them.
    """
    m = bar_mesh()
    p = [-20.0, 5.0, 5.0]
    dofs = [True, False, True, True, False, True]
    st = _study([RemoteSupport("x0", p, dofs, "distributing"), Displacement("y0", uy=0.0)])
    from fea_core.cylinder import boundary_dofs
    bc = boundary_dofs(st, m)
    eqs = distributing_equations(st, m, bc)
    assert len(eqs) == 4
    deps = [(e[0][0], e[0][1]) for e in eqs]
    assert len(set(deps)) == 4 and not any(bc[n, d] for n, d in deps)
    for i, e in enumerate(eqs):
        assert e[0][2] == 1.0
        others = {(n, d) for k, ee in enumerate(eqs) if k != i for n, d, _ in ee}
        assert deps[i] not in others
    # rows = linear combination of the constrained rows of RBE3
    op = rbe3(m, "x0", p)
    cols = {(int(n), d): j for j, (n, d) in enumerate((n, d) for n in op["idx"] for d in range(3))}
    G = np.zeros((len(eqs), len(cols)))
    for i, e in enumerate(eqs):
        for n, d, c in e:
            G[i, cols[(n, d)]] = c
    A = op["A"].reshape(6, -1)
    assert np.linalg.matrix_rank(np.vstack([G, A[np.array(dofs)]]), tol=1e-9) == 4
    # the rigid motion in the free components (uy, ry) satisfies the equations, in the constrained ones not
    for k, free in ((1, True), (4, True), (0, False), (5, False)):
        t, w = np.zeros(3), np.zeros(3)
        (t if k < 3 else w)[k % 3] = 1.0
        U = t + np.cross(w, m.nodes - p)
        r = np.array([sum(c * U[n, d] for n, d, c in e) for e in eqs])
        assert (np.abs(r).max() < 1e-9) == free, (k, r)


def test_rigid_rows():
    c = RemoteSupport("x0", [10.0, 0, 0], [True] * 6)
    assert np.linalg.matrix_rank(rigid_rows(c, np.zeros(3))) == 6
    c.dofs = [True, True, True, False, False, False]
    assert np.linalg.matrix_rank(rigid_rows(c, np.zeros(3))) == 3


def test_write_inp_rigid_and_distributing(tmp_path):
    m = bar_mesh()
    st = _study([RemoteSupport("x0", [-50.0, 5.0, 5.0], [True, True, True, False, True, False], "rigid")])
    mp = write_inp(st, m, tmp_path / "r.inp")
    txt = (tmp_path / "r.inp").read_text()
    ref, rot = mp.remote["x0"]["ref"], mp.remote["x0"]["rot"]
    assert f"*RIGID BODY, NSET={mp.nsets['x0']}, REF NODE={ref}, ROT NODE={rot}" in txt
    assert "*NSET, NSET=NRSUP" in txt and "*NODE PRINT, NSET=NRSUP\nU, RF" in txt
    bnd = txt.split("*BOUNDARY\n")[1].split("*")[0]
    assert bnd == f"{ref}, 1, 1, 0.\n{ref}, 2, 2, 0.\n{ref}, 3, 3, 0.\n{rot}, 2, 2, 0.\n"
    # distributing: dense *EQUATION, no REF node
    st.constraints = [RemoteSupport("x0", [-50.0, 5.0, 5.0], [True] * 6, "distributing")]
    mp = write_inp(st, m, tmp_path / "d.inp")
    txt = (tmp_path / "d.inp").read_text()
    assert not mp.remote and "*RIGID BODY" not in txt and txt.count("RBE3") == 1
    eq = txt.split("(RBE3)\n*EQUATION\n")[1].split("*")[0].splitlines()
    assert int(eq[0]) > 10 and max(len(l.split(",")) for l in eq) <= 12      # ccx: at most 12 items per row
    # a rigid constraint sharing nodes with *BOUNDARY (edge x0 ∩ y0) → an understandable error, the distributing one passes
    st.constraints = [RemoteSupport("x0", [0, 5, 5], [True] * 6, "rigid"), Displacement("y0", uy=0.0)]
    with pytest.raises(RemoteError, match="shares nodes with support 'y0'"):
        write_inp(st, m, tmp_path / "e.inp")
    st.constraints[0].coupling = "distributing"
    write_inp(st, m, tmp_path / "e.inp")


def test_rigid_remote_loads_share_one_body(tmp_path):
    """More states with a rigid remote load on the same face = ONE *RIGID BODY (ccx does not attach a second rigid body on the same
    nodes – found by a cross-check, X4); a load from another point → the moment about the point of the body.
    """
    from fea_core.study import RemoteLoad
    m = bar_mesh()
    st = _study([Displacement("x0", ux=0.0, uy=0.0, uz=0.0)])
    p1, p2 = [150.0, 5.0, 5.0], [150.0, 25.0, 5.0]
    st.load_cases = [LoadCase("A", [RemoteLoad("x100", p1, [0, 0, -100.0], [1000.0, 0, 0], "rigid")]),
                     LoadCase("B", [RemoteLoad("x100", p2, [0, 0, -100.0], [0, 0, 0], "rigid")])]
    mp = write_inp(st, m, tmp_path / "r.inp")
    txt = (tmp_path / "r.inp").read_text()
    assert txt.count("*RIGID BODY") == 1
    a, b = mp.ref_nodes["0:0"], mp.ref_nodes["1:1"]
    assert a == b and txt.split("*NSET, NSET=NREF\n")[1].split("*")[0].strip().rstrip(",") == f"{a['ref']}, {a['rot']}"
    step_b = txt.split("*STEP")[2]
    cl = dict(((int(n), int(d)), float(v)) for n, d, v in
              (r.split(",") for r in step_b.split("*CLOAD, OP=NEW\n")[1].split("*")[0].splitlines()))
    M = np.cross(np.subtract(p2, p1), [0, 0, -100.0])          # (p2 − p1) × F = (2000, 0, 0)
    assert cl[(a["ref"], 3)] == -100.0 and cl[(a["rot"], 1)] == pytest.approx(M[0])
    assert (a["rot"], 2) not in cl and (a["rot"], 3) not in cl


def test_read_remote(tmp_path):
    """Reactions and displacements of the REF / ROT node from the .dat by steps (the U and RF blocks)."""
    from fea_core.ccx.inp import InpMap
    from fea_core.solve import read_remote
    blk = ("\r\n                        S T E P       {k}\r\n\r\n\r\n                                INCREMENT     1\r\n\r\n\r\n"
           " displacements (vx,vy,vz) for set NRSUP and time  0.1000000E+01\r\n\r\n"
           "        11  0.000000E+00  0.000000E+00  0.000000E+00\r\n"
           "        12  0.000000E+00  {r:.6E}  0.000000E+00\r\n\r\n"
           " forces (fx,fy,fz) for set NRSUP and time  0.1000000E+01\r\n\r\n"
           "        11 -2.564508E-09 -2.827267E-09  {f:.6E}\r\n"
           "        12  2.433770E-07 -2.000000E+04 -3.679141E-07\r\n")
    (tmp_path / "j.dat").write_bytes((blk.format(k=1, r=0.0, f=100.0) + blk.format(k=2, r=1.5e-3, f=-7.0)).encode())
    mp = InpMap()
    mp.remote = {"celo": {"ref": 11, "rot": 12}}
    out = read_remote(tmp_path / "j.dat", mp)
    assert len(out) == 2
    assert out[0]["celo"]["force"][2] == 100.0 and out[0]["celo"]["moment"][1] == -2e4
    assert out[1]["celo"]["force"][2] == -7.0 and out[1]["celo"]["rot"][1] == 1.5e-3


def test_check_remote_and_json():
    st = _study([RemoteSupport("x0", [0, 5], [True] * 6, "pevna")])
    from fea_core.solve import check_remote
    e, w = check_remote(st)
    assert any("3 coordinates" in x for x in e) and any("unknown coupling type" in x for x in e)
    st.constraints = [RemoteSupport("x0", [0, 5, 5], [False] * 6)]
    assert check_remote(st)[1]
    st2 = Study.from_dict(st.to_dict())
    assert isinstance(st2.constraints[0], RemoteSupport) and st2.constraints[0].coupling == "rigid"
