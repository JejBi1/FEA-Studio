"""Tests of the shell model (fea_core.shell). Fast: recognition of a plate / sheet / tube, mid-surface, extension of the T-joint, orientation, integration,
constraints and loads, writing of the .inp, results (without Gmsh and ccx).
Slow (slow): verification cases Sh1–Sh7 (verification/shell_cases.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from fea_core.geometry import Part, bbox, list_faces
from fea_core.shell import ShellSettings, ShellError
from fea_core.shell.mesh import ShellMesh, orient
from fea_core.shell.loads import (Target, tri6_integrals, constraint_dofs, pressure_loads, force_loads,
                                  gravity_loads, line3_gauss)
from verification.cases import box, cyl, solid


def rect_mesh(L=100.0, B=40.0, nx=10, ny=4, z=0.0):
    """Structured tri6 mesh of a rectangle (normal +z)."""
    X = np.linspace(0, L, 2 * nx + 1)
    Y = np.linspace(0, B, 2 * ny + 1)
    idx = np.arange(len(X) * len(Y)).reshape(len(Y), len(X))
    nodes = np.array([[x, y, z] for y in Y for x in X])
    tri = []
    for j in range(ny):
        for i in range(nx):
            a, b = idx[2 * j, 2 * i], idx[2 * j, 2 * i + 2]
            c, d = idx[2 * j + 2, 2 * i + 2], idx[2 * j + 2, 2 * i]
            ab, bc, cd, da = idx[2 * j, 2 * i + 1], idx[2 * j + 1, 2 * i + 2], idx[2 * j + 2, 2 * i + 1], idx[2 * j + 1, 2 * i]
            ac = idx[2 * j + 1, 2 * i + 1]
            tri += [[a, b, c, ab, bc, ac], [a, c, d, ac, cd, da]]
    tri = np.array(tri)
    return ShellMesh(nodes, tri, np.zeros(len(tri), np.int64), ["Deska"], np.array([5.0]), np.array([0.0]))


# ------------------------------------------------------------------ settings

def test_settings_roundtrip_and_check():
    s = ShellSettings(True, {"A": 5.0, "B": None}, 4.0, "bot")
    d = s.to_dict()
    s2 = ShellSettings.from_dict(d)
    assert s2 == s and ShellSettings.from_dict(None) == ShellSettings()
    assert ShellSettings.from_dict({"enabled": True, "neznamy": 1}).enabled
    assert not s.check()
    assert ShellSettings(offset="x").check() and ShellSettings(element="S8R").check()
    assert ShellSettings(parts={"A": -1.0}).check()


# ------------------------------------------------------------------ geometrie

def test_detect_plate():
    from fea_core.shell.geometry import detect_sheet
    info, why = detect_sheet(Part("Deska", solid(box(0, 0, 0, 200, 40, 5))))
    assert info is not None and why == ""
    assert info.kind == "plate" and info.thickness == pytest.approx(5.0)
    assert info.mid_area == pytest.approx(8000.0)
    assert np.allclose(info.normal, [0, 0, 1])                  # top = positive normal
    bb = bbox(info.mid)
    assert bb[2] == pytest.approx(2.5) and bb[5] == pytest.approx(2.5)
    assert len(info.sides) == 4


def test_detect_plate_with_hole_and_reject_block():
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from fea_core.shell.geometry import detect_sheet
    plate = solid(BRepAlgoAPI_Cut(box(0, 0, 0, 100, 100, 4), cyl((50, 50, -1), (0, 0, 1), 10, 6)).Shape())
    info, _ = detect_sheet(Part("D", plate))
    assert info.kind == "plate" and info.thickness == pytest.approx(4.0)
    assert info.mid_area == pytest.approx(10000 - np.pi * 100, rel=1e-6)
    assert len(info.sides) == 5                                 # 4 sides + hole
    info, why = detect_sheet(Part("K", solid(box(0, 0, 0, 50, 50, 40))))
    assert info is None and "too thick" in why


def test_detect_bent_sheet_and_tube():
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from fea_core.shell.geometry import detect_sheet, _area
    from verification.shell_cases import bent_sheet
    info, why = detect_sheet(Part("P", bent_sheet(2.0, 5.0, 100.0, 60.0, 40.0)))
    assert info is not None, why
    assert info.kind == "sheet" and info.thickness == pytest.approx(2.0)
    ref = 100 * 40 + 60 * 40 + np.pi / 2 * 6 * 40
    assert info.mid_area == pytest.approx(ref) and _area(info.mid) == pytest.approx(ref, rel=1e-6)
    tube = solid(BRepAlgoAPI_Cut(cyl((0, 0, 0), (0, 0, 1), 102, 300), cyl((0, 0, 0), (0, 0, 1), 98, 300)).Shape())
    info, why = detect_sheet(Part("T", tube))
    assert info is not None, why
    assert info.kind == "sheet" and info.thickness == pytest.approx(4.0)
    assert _area(info.mid) == pytest.approx(2 * np.pi * 100 * 300, rel=1e-6)


def test_connect_t_joint_and_lap_warning():
    from fea_core.shell.geometry import detect_sheet, connect_sheets
    A = Part("A", solid(box(0, 0, 0, 200, 40, 5)))
    B = Part("B", solid(box(97.5, 0, 5, 5, 40, 100)))
    sa, sb = detect_sheet(A)[0], detect_sheet(B)[0]
    w = connect_sheets([sa, sb], {"A": A, "B": B})
    assert not w and not sa.extensions and len(sb.extensions) == 1
    bb = bbox(sb.extensions[0])
    assert np.allclose(bb, [100, 0, 2.5, 100, 40, 5], atol=1e-6)    # the web extended to the mid-surface A
    assert len(list_faces(sb.mid_shape())) == 2
    C = Part("C", solid(box(50, 0, 5, 40, 40, 3)))
    sc = detect_sheet(C)[0]
    w = connect_sheets([detect_sheet(A)[0], sc], {"A": A, "C": C})
    assert w and "lap joint" in w[0]


def test_surface_sheet_needs_thickness():
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pln, gp_Pnt, gp_Dir
    from fea_core.shell.geometry import surface_sheet
    f = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0, 10, 0, 10).Face()
    with pytest.raises(ShellError):
        surface_sheet(Part("P", f), None)
    s = surface_sheet(Part("P", f), 2.0, "top")
    assert s.offset == 0.5 and s.thickness == 2.0 and s.mid_area == pytest.approx(100.0)


def test_read_step_surfaces(tmp_path):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pln, gp_Pnt, gp_Dir
    from fea_core.geometry import write_step, read_step
    from fea_core.shell.geometry import read_step_surfaces
    f = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 1), gp_Dir(0, 0, 1)), 0, 10, 0, 20).Face()
    p = tmp_path / "m.step"
    write_step([Part("Plech", f), Part("Kostka", solid(box(0, 0, 0, 5, 5, 5)))], p)
    surf = read_step_surfaces(p)
    assert [s.name for s in surf] == ["Plech"] and len(list_faces(surf[0].shape)) == 1
    assert [s.name for s in read_step(p)] == ["Kostka"]


# ------------------------------------------------------------------ mesh, integration

def test_orient_makes_consistent():
    m = rect_mesh()
    rng = np.random.default_rng(1)
    flip = rng.random(len(m.tri)) < 0.5
    m.tri[flip] = m.tri[flip][:, [0, 2, 1, 5, 4, 3]]
    n = orient(m, {0: np.array([0, 0, 1.0])})
    assert n == int(flip.sum())
    assert np.allclose(m.normals()[:, 2], 1.0)
    m.tri[:] = m.tri[:, [0, 2, 1, 5, 4, 3]]
    orient(m)                                                  # without a reference: most of the content stays
    assert np.allclose(m.normals()[:, 2], -1.0)


def test_tri6_integrals_and_boundary():
    m = rect_mesh(100, 40, 5, 2)
    nint, area = tri6_integrals(m.nodes, m.tri)
    assert area.sum() == pytest.approx(4000.0)
    assert np.allclose(nint[:, :3], 0.0, atol=1e-12) and np.allclose(nint[:, 3:], area[:, None] / 3)
    line, elem = m.boundary_edges()
    _, dl, _, N = line3_gauss(m.nodes, line)
    assert dl.sum() == pytest.approx(280.0)
    assert len(line) == 2 * (5 + 2)


# ------------------------------------------------------------------ constraints and loads

def _edge_target(m, cond, normal):
    line, _ = m.boundary_edges()
    x = m.nodes[line]
    k = np.nonzero(cond(x).all(axis=1))[0]
    return Target("e", edges=line[k], edge_normal=np.tile(normal, (len(k), 1)), edge_thick=np.full(len(k), 5.0),
                  side_edges=np.ones(len(k), bool))


def test_constraint_dofs_plate():
    from fea_core.study import Fixed, Displacement
    m = rect_mesh()
    nn = m.node_normals()
    tg = _edge_target(m, lambda x: np.abs(x[..., 0]) < 1e-9, [-1, 0, 0])
    d, fr = constraint_dofs(Fixed("e"), tg, m, nn)
    assert all(set(v) == set(range(1, 7)) for v in d.values()) and not fr
    d, fr = constraint_dofs(Displacement("e", uz=0.0), tg, m, nn)
    assert all(v == {3: 0.0} for v in d.values())              # hinge (simple support)
    d, fr = constraint_dofs(Displacement("e", ux=0.0), tg, m, nn)
    assert all(v == {1: 0.0, 5: 0.0} for v in d.values())      # symetrie: ux = 0, θy = 0
    d, fr = constraint_dofs(Displacement("e", ux=0.1, uy=0.0, uz=0.0), tg, m, nn)
    assert all(set(v) == set(range(1, 7)) and v[1] == 0.1 for v in d.values())


def test_constraint_dofs_oblique_frame():
    from fea_core.study import Displacement
    m = rect_mesh()
    c, s = np.cos(0.5), np.sin(0.5)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])          # a plate inclined about x
    m.nodes = m.nodes @ R.T
    nn = m.node_normals()
    tg = _edge_target(m, lambda x: np.abs(x[..., 0]) < 1e-9, [-1, 0, 0])
    d, fr = constraint_dofs(Displacement("e", uy=0.0), tg, m, nn)   # a = n × e_y ∥ x → global DOF 4
    assert all(v == {2: 0.0, 4: 0.0} for v in d.values()) and not fr
    d, fr = constraint_dofs(Displacement("e", ux=0.0), tg, m, nn)   # a = n × e_x = (0, c, s) → local axes
    v = next(iter(d))
    assert d[v] == {1: 0.0, 5: 0.0} and set(fr) == set(d)
    T = fr[v]
    assert np.allclose(T[:, 0], [1, 0, 0]) and np.allclose(T[:, 1], [0, c, s]) and np.allclose(T.T @ T, np.eye(3))
    assert abs(T[:, 1] @ nn[v]) < 1e-9                         # the axis of rotation lies in the plane of the shell
    with pytest.raises(ShellError):
        constraint_dofs(Displacement("e", ux=0.0, uy=0.0), tg, m, nn)


def test_pressure_force_gravity_totals():
    from fea_core.study import Pressure, Force, Gravity
    m = rect_mesh()
    tg = Target("top", elements=np.arange(len(m.tri)), elem_sign=np.ones(len(m.tri)))
    F = pressure_loads(Pressure("top", 0.01), tg, m)
    assert np.allclose(F.sum(axis=0), [0, 0, -0.01 * 4000])     # against the normal of the face (+z)
    Fv = pressure_loads(Pressure("top", 0.0, gradient=[1e-4, 0, 0], origin=[0, 0, 0]), tg, m)
    assert Fv.sum(axis=0)[2] == pytest.approx(-1e-4 * 50 * 4000)  # linear pressure: average p = g·L/2
    F = force_loads(Force("top", [1.0, 2.0, 3.0]), tg, m)
    assert np.allclose(F.sum(axis=0), [1, 2, 3])
    tge = _edge_target(m, lambda x: np.abs(x[..., 0] - 100) < 1e-9, [1, 0, 0])
    F = pressure_loads(Pressure("e", 2.0), tge, m)
    assert np.allclose(F.sum(axis=0), [-2.0 * 40 * 5, 0, 0])    # side: p·t·L against the normal
    rho_t = np.full(len(m.tri), 7.85e-9 * 5)
    G = gravity_loads(Gravity([0, 0, -9810.0]), m, rho_t)
    assert G.sum(axis=0)[2] == pytest.approx(-7.85e-9 * 5 * 4000 * 9810)
    from fea_core.study import Rotation
    from fea_core.shell.loads import rotation_loads
    Rf = rotation_loads(Rotation([0, 0, 0], [0, 0, 1], omega=100.0), m, rho_t)
    assert np.allclose(Rf.sum(axis=0), 7.85e-9 * 5 * 4000 * 1e4 * np.array([50.0, 20.0, 0.0]))   # ρ t A ω² r_c


def test_write_inp_transform(tmp_path):
    from fea_core.shell.inp import write_shell_inp
    from fea_core.study import Material
    m = rect_mesh(20, 10, 2, 1)
    T = np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(float)
    F = np.zeros((len(m.nodes), 3))
    F[0] = [1.0, 2.0, 3.0]
    write_shell_inp(tmp_path / "a.inp", m, {0: Material("Ocel", 210000, 0.3, 7.85e-9)}, {0: {1: 0.0, 5: 0.0}},
                    [("S", F)], "test", {0: T})
    txt = (tmp_path / "a.inp").read_text()
    assert "*ELEMENT, TYPE=S6, ELSET=P0" in txt and "*SHELL SECTION, ELSET=P0, MATERIAL=M1" in txt
    assert "*TRANSFORM, NSET=NT1, TYPE=R" in txt and "1, 5, 5, 0" in txt
    assert "1, 1, 2\n1, 2, 3\n1, 3, 1" in txt                  # force transferred to the local axes T


# ------------------------------------------------------------------ results

def test_frd_elements_and_results_roundtrip(tmp_path):
    from fea_core.shell.results import read_frd_elements, ShellResults, ShellStep
    frd = tmp_path / "x.frd"
    frd.write_text("    3C                             1                                     1\n"
                   " -1         1    5    0    1\n"
                   " -2        10        11        12        13        14        15        16        17        18        19\n"
                   " -2        20        21        22        23        24\n"
                   " -3\n", encoding="latin-1")
    els = read_frd_elements(frd)
    assert list(els) == [1] and list(els[1]) == list(range(10, 25))
    m = rect_mesh(20, 10, 2, 1)
    N = len(m.nodes)
    St = np.zeros((N, 6))
    St[:, 0] = m.nodes[:, 0]
    U = np.zeros((N, 3))
    U[:, 2] = -m.nodes[:, 0] / 20
    st = ShellStep("S", U, St, np.zeros((N, 6)), -St, {"v": np.arange(6.0)}, {"imbalance": 1e-9}, np.zeros(6))
    r = ShellResults(m, [st], m.nodes.mean(axis=0), {"v": np.array([0, 1])}, ["w"], {"a": 1})
    r.save(tmp_path / "res")
    r2 = ShellResults.load(tmp_path / "res")
    assert np.allclose(r2.step(0).U, U) and r2.warnings == ["w"] and list(r2.step("S").reactions["v"]) == list(range(6))
    assert r2.probe([7.0, 3.0, 0.0], "sxx", 0, "top")["value"] == pytest.approx(7.0)
    assert r2.probe([7.0, 3.0, 0.0], "sxx", 0, "bot")["value"] == pytest.approx(-7.0)
    assert np.allclose(r2.field("bending", 0), np.abs(m.nodes[:, 0]))
    assert r2.field("mises", 0, "max").max() == pytest.approx(20.0)
    sm = r2.summary()
    assert sm["steps"][0]["u_max"] == pytest.approx(1.0) and sm["parts"]["Deska"]["thickness"] == 5.0
    png = r2.render_png(tmp_path / "a.png", "mises", 0, "top", deform=True, size=(400, 300))
    assert png.exists() and png.stat().st_size > 1000


def test_no_qt_or_pyplot_import():
    import subprocess
    import sys
    code = ("import sys; import fea_core.shell.results, fea_core.shell.run, fea_core.shell.mesh_worker; "
            "assert 'matplotlib.pyplot' not in sys.modules and 'PySide6' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_mesh_worker_has_no_ocp():
    import subprocess
    import sys
    code = "import sys; import fea_core.shell.mesh_worker; assert not any(m.startswith('OCP') for m in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


# ------------------------------------------------------------------ slow: verification cases

@pytest.mark.slow
@pytest.mark.parametrize("key", ["Sh1", "Sh2", "Sh3", "Sh5", "Sh7"])
def test_shell_cases(key, tmp_path):
    from fea_core.shell import run_shell
    from verification.shell_cases import SHELL_CASES
    case = SHELL_CASES[key]
    study, parts = case.build(tmp_path)
    res = run_shell(study, tmp_path / "run", parts=parts, settings=case.settings)
    checks = case.evaluate(res, study, tmp_path)
    bad = [f"{c.name}: {c.value} vs {c.ref}" for c in checks if not c.ok]
    assert not bad, bad


def test_unsupported_items_rejected_before_mesh(tmp_path):
    from fea_core.shell import run_shell, check_study
    from fea_core.study import BearingLoad, CylindricalSupport, LoadCase
    from verification.shell_cases import sh1_build
    st, parts = sh1_build(tmp_path)
    assert check_study(st) == []
    st.load_cases = [LoadCase("B", [BearingLoad("konec", [0, 0, 1.0])])]
    st.constraints.append(CylindricalSupport("konec"))
    errs = check_study(st)
    assert len(errs) == 2 and "bearing" in errs[1] and "cylindrical" in errs[0]
    with pytest.raises(ShellError, match="not supported in the shell model"):
        run_shell(st, tmp_path / "run", parts=parts)
    assert not (tmp_path / "run" / "shell_mesh_job.json").exists()
