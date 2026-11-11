"""Tests of the 2D calculations (fea_core.twod). Fast: the frame of the section, the OCP section, the conversion of the coordinates, the orientation of the elements, the mapping of the edges
of the selections, the integration of the loads, writing of the .inp, Results2D (probe, segment, linearisation, saving, PNG) – without Gmsh and ccx.
Slow: the verification cases A1–A5P and an assembly of two parts with the weight, a force and several states (Gmsh + ccx).
"""
import math

import numpy as np
import pytest

from fea_core.twod.convert import selection_edges, weight_fn
from fea_core.twod.mesh2d import (FACE_NODES, Mesh2D, body_forces, boundary_lookup, edge_forces, edge_measure,
                                  orient)
from fea_core.twod.model import TwoDError, TwoDSettings, make_frame
from fea_core.twod.results import Results2D

E, NU = 210000.0, 0.3


# ------------------------------------------------------------------ helpers: structured mesh of a rectangle
def rect(x0, x1, y0, y1, nx, ny, cw=False):
    X = np.linspace(x0, x1, 2 * nx + 1)
    Y = np.linspace(y0, y1, 2 * ny + 1)
    idx = np.arange(len(X) * len(Y)).reshape(len(Y), len(X))
    nodes = np.array([[x, y] for y in Y for x in X])
    tri = []
    for j in range(ny):
        for i in range(nx):
            a, b = idx[2 * j, 2 * i], idx[2 * j, 2 * i + 2]
            c, d = idx[2 * j + 2, 2 * i + 2], idx[2 * j + 2, 2 * i]
            ab, bc = idx[2 * j, 2 * i + 1], idx[2 * j + 1, 2 * i + 2]
            cd, da, ac = idx[2 * j + 2, 2 * i + 1], idx[2 * j + 1, 2 * i], idx[2 * j + 1, 2 * i + 1]
            tri += [[a, b, c, ab, bc, ac], [a, c, d, ac, cd, da]]
    tri = np.array(tri)
    if cw:
        tri = tri[:, [0, 2, 1, 5, 4, 3]]
    return nodes.astype(float), tri


def rect_mesh(x0, x1, y0, y1, nx=4, ny=3, cw=False) -> Mesh2D:
    """Mesh2D of a rectangle: the boundary segments and the 4 curves (sides) with the sample points as from Gmsh."""
    nodes, tri = rect(x0, x1, y0, y1, nx, ny, cw)
    tri, _ = orient(nodes, tri)
    seg, curve = [], []
    sides = [(lambda p: abs(p[1] - y0) < 1e-9), (lambda p: abs(p[0] - x1) < 1e-9),
             (lambda p: abs(p[1] - y1) < 1e-9), (lambda p: abs(p[0] - x0) < 1e-9)]
    for t in tri:
        for k in range(3):
            a, b, m = t[FACE_NODES[k]]
            for ci, on in enumerate(sides):
                if on(nodes[a]) and on(nodes[b]):
                    seg.append([a, b, m])
                    curve.append(ci)
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    pts = np.array([[corners[c] + s * (corners[(c + 1) % 4] - corners[c]) for s in np.linspace(0, 1, 7)]
                    for c in range(4)])
    seg, elem, face = boundary_lookup(tri, np.array(seg))
    return Mesh2D(nodes, tri, np.zeros(len(tri), np.int64), ["Dil"], seg, elem, face, np.array(curve), pts, {})


# ------------------------------------------------------------------ frame and settings
def test_frame_axisym_roundtrip():
    fr = make_frame(TwoDSettings(kind="axisym", origin=[1, 2, 3], axis=[0, 0, 2], xdir=[1, 1, 0.5]))
    assert np.allclose(fr.e2, [0, 0, 1]) and np.allclose(fr.e1, np.array([1, 1, 0]) / math.sqrt(2))
    assert np.allclose(np.cross(fr.e1, fr.e2), fr.n)
    xy = np.array([[5.0, -2.0], [0.0, 7.0]])
    assert np.allclose(fr.to2d(fr.to3d(xy)), xy)
    # point outside the half-plane: r = distance from the axis
    p = np.array([1, 2, 3]) + np.array([0, -4.0, 1.5])
    assert np.allclose(fr.to2d(p), [4.0, 1.5])


def test_frame_plane_and_errors():
    fr = make_frame(TwoDSettings(kind="plane_stress", origin=[0, 0, 1], axis=[0, 0, 1], xdir=[0, 1, 0]))
    assert np.allclose(fr.e1, [0, 1, 0]) and np.allclose(fr.e2, [-1, 0, 0]) and np.allclose(fr.n, [0, 0, 1])
    assert np.allclose(fr.to2d([3, 4, 1]), [4, -3])
    with pytest.raises(TwoDError):
        make_frame(TwoDSettings(kind="axisym", axis=[0, 0, 1], xdir=[0, 0, 5]))
    with pytest.raises(TwoDError):
        make_frame(TwoDSettings(kind="plane_stress", thickness=0.0))
    with pytest.raises(TwoDError):
        make_frame(TwoDSettings(kind="shell"))


def test_settings_dict_ignores_unknown():
    s = TwoDSettings(True, "plane_strain", [1, 2, 3], [0, 1, 0], [1, 0, 0], 2.5, ["A"], 0.8)
    d = s.to_dict()
    d["novy_klic"] = 1
    assert TwoDSettings.from_dict(d) == s
    assert TwoDSettings.from_dict(None) == TwoDSettings()


# ------------------------------------------------------------------ OCP section
def _quarter_tube():
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
    from verification.cases import box, cyl, solid
    ring = BRepAlgoAPI_Cut(cyl((0, 0, 0), (0, 0, 1), 40, 10), cyl((0, 0, -1), (0, 0, 1), 20, 12)).Shape()
    return solid(BRepAlgoAPI_Common(ring, box(0, 0, -1, 41, 41, 12)).Shape())


def test_section_axisym_quarter_tube(tmp_path):
    from fea_core.geometry import Part
    from fea_core.twod.section import plane_edges, section_parts
    fr = make_frame(TwoDSettings(kind="axisym", axis=[0, 0, 1], xdir=[1, 1, 0]))
    sp, = section_parts([Part("T", _quarter_tube())], fr, tmp_path)
    assert sp.brep and sp.area == pytest.approx(200.0, rel=1e-9)
    assert sp.centroid == pytest.approx([30.0, 5.0], rel=1e-9)
    assert sp.volume / sp.model_volume == pytest.approx(0.25, rel=1e-6)          # sector 90°
    pts = np.vstack(plane_edges(sp.brep))
    assert pts[:, 0].min() == pytest.approx(20) and pts[:, 0].max() == pytest.approx(40)
    assert pts[:, 1].min() == pytest.approx(0) and pts[:, 1].max() == pytest.approx(10)


def test_section_plane_and_miss(tmp_path):
    from fea_core.geometry import Part
    from fea_core.twod.section import section_parts
    from verification.cases import box
    parts = [Part("A", box(0, 0, 0, 30, 20, 4)), Part("B", box(100, 0, 10, 5, 5, 5))]
    fr = make_frame(TwoDSettings(kind="plane_stress", origin=[0, 0, 2], axis=[0, 0, 1], xdir=[1, 0, 0], thickness=4))
    a, b = section_parts(parts, fr, tmp_path, thickness=4)
    assert a.area == pytest.approx(600.0) and a.model_volume == pytest.approx(a.volume)
    assert b.brep is None


# ------------------------------------------------------------------ mesh: orientation, edges, integration
def test_orient_and_boundary_lookup():
    m_cw = rect_mesh(0, 2, 0, 1, cw=True)
    m_ccw = rect_mesh(0, 2, 0, 1)
    for m in (m_cw, m_ccw):
        x = m.nodes[m.tri[:, :3]]
        area = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
        assert np.all(area > 0)
        # segments oriented by the element → the outward normal outwards from the rectangle
        mid = m.nodes[m.seg[:, 2]]
        t = m.nodes[m.seg[:, 1]] - m.nodes[m.seg[:, 0]]
        n = np.stack([t[:, 1], -t[:, 0]], 1)
        assert np.all(np.einsum("ij,ij->i", n, mid - [1.0, 0.5]) > 0)
        assert np.all(m.tri[m.seg_elem, FACE_NODES[m.seg_face - 1, 2]] == m.seg[:, 2])


def test_integration_axisym_and_plane():
    ri, ro, h, p, rho, g = 10.0, 20.0, 5.0, 3.0, 7.85e-9, 9810.0
    m = rect_mesh(ri, ro, 0, h, 3, 2)
    fa = make_frame(TwoDSettings(kind="axisym"))
    w = weight_fn(fa, 1.0)
    top = np.nonzero(m.seg_curve == 2)[0]
    F = edge_forces(m, top, lambda x, n: -p * n, w)
    assert F[:, 1].sum() == pytest.approx(-p * math.pi * (ro ** 2 - ri ** 2), rel=1e-12)
    assert edge_measure(m, top, w) == pytest.approx(math.pi * (ro ** 2 - ri ** 2), rel=1e-12)
    Fg = body_forces(m, np.full(len(m.tri), rho), lambda x: np.broadcast_to([0.0, -g], x.shape), w)
    assert -Fg[:, 1].sum() == pytest.approx(rho * g * math.pi * (ro ** 2 - ri ** 2) * h, rel=1e-12)
    fp = make_frame(TwoDSettings(kind="plane_stress", thickness=2.0))
    Fg = body_forces(m, np.full(len(m.tri), rho), lambda x: np.broadcast_to([0.0, -g], x.shape), weight_fn(fp, 2.0))
    assert -Fg[:, 1].sum() == pytest.approx(rho * g * (ro - ri) * h * 2.0, rel=1e-12)


def _box_study(tmp_path):
    from fea_core.geometry import Part
    from fea_core.selectors import find_faces
    from fea_core.study import Fixed, Force, LoadCase, MeshSettings, Pressure
    from verification.cases import _study, box
    st, parts = _study("box", tmp_path / "m.step", [Part("Dil", box(0, 0, 0, 20, 10, 4))], MeshSettings(size=2))
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("x20", find_faces(parts, x=20))
    st.add_selection("z0", find_faces(parts, z=0))           # in the plane of the section z = 0 → left out
    st.add_selection("y10", find_faces(parts, y=10))
    st.constraints.append(Fixed("x0"))
    st.load_cases.append(LoadCase("F", [Force("x20", [400.0, 0, 0]), Pressure("y10", 2.0)]))
    return st, parts


def test_selection_edges_mapping(tmp_path):
    st, parts = _box_study(tmp_path)
    m = rect_mesh(0, 20, 0, 10, 5, 3)
    fr = make_frame(TwoDSettings(kind="plane_stress", origin=[0, 0, 0], axis=[0, 0, 1], thickness=4))
    warn = []
    ed = selection_edges(st, parts, m, fr, ["x0", "x20", "z0", "y10"], warn)
    assert set(m.seg_curve[ed["x0"]]) == {3} and len(ed["x0"]) == 3
    assert set(m.seg_curve[ed["x20"]]) == {1}
    assert set(m.seg_curve[ed["y10"]]) == {2} and len(ed["y10"]) == 5
    assert len(ed["z0"]) == 0 and any("z0" in w and "section plane" in w for w in warn)


def test_convert_loads_and_inp(tmp_path):
    from fea_core.twod.convert import build_model
    from fea_core.twod.inp import write_inp
    st, parts = _box_study(tmp_path)
    m = rect_mesh(0, 20, 0, 10, 5, 3)
    fr = make_frame(TwoDSettings(kind="plane_stress", origin=[0, 0, 2], axis=[0, 0, 1], thickness=4))
    warn = []
    model = build_model(st, parts, m, fr, 4.0, warn)
    step = model.steps[0]
    # force 400 N on the area 10×4 → traction 10 MPa × length 10 × t 4 = 400 N (exactly F)
    assert step.cload[:, 0].sum() == pytest.approx(400.0, rel=1e-12)
    # pressure 2 MPa at y = 10 (length 20, t 4) → −160 N in y (only for the equilibrium, *DLOAD is written)
    assert step.F[:, 1].sum() == pytest.approx(-160.0, rel=1e-12) and len(step.dload) == 5
    assert all(face in (1, 2, 3) for _, face, _ in step.dload)
    inp = write_inp(model, tmp_path / "j.inp").read_text()
    assert "TYPE=CPS6" in inp and "*SOLID SECTION, ELSET=P1, MATERIAL=M1\n4\n" in inp and "P" in inp


def test_unsupported_load_raises(tmp_path):
    from fea_core.study import RemoteLoad
    from fea_core.twod.convert import build_model
    st, parts = _box_study(tmp_path)
    st.load_cases[0].loads.append(RemoteLoad("x20", [30, 5, 2], [0, 0, 10.0]))
    m = rect_mesh(0, 20, 0, 10, 5, 3)
    fr = make_frame(TwoDSettings(kind="plane_stress", origin=[0, 0, 2], axis=[0, 0, 1], thickness=4))
    with pytest.raises(TwoDError, match="not yet supported in 2D"):
        build_model(st, parts, m, fr, 4.0, [])


# ------------------------------------------------------------------ Results2D without ccx
def _fake_results(kind="plane_stress"):
    m = rect_mesh(0, 4, 0, 2, 4, 2)
    X = m.nodes
    U = np.column_stack([1e-3 * X[:, 0] ** 2, -2e-3 * X[:, 0] * X[:, 1]])      # a quadratic field = exact in tri6
    S = np.zeros((len(X), 6))
    S[:, 0] = 100.0 - 10 * X[:, 1]                                             # σx linear over the height
    S[:, 2] = 7.0
    step = {"name": "LC", "U": U, "S": S, "RF": np.zeros_like(U), "reactions": {"a": np.array([1.0, 2.0])},
            "equilibrium": {"imbalance": 0.0}}
    fr = {"origin": [0, 0, 0], "e1": [1, 0, 0], "e2": [0, 1, 0], "n": [0, 0, 1]}
    return Results2D(kind, X, m.tri, m.tri_part, m.part_names, [step], {"a": m.seg[:2]}, ["pozor"], fr, 3.0, {}, m)


def test_results_probe_line_linearize(tmp_path):
    r = _fake_results()
    p = [1.37, 0.81]
    assert r.probe(p, "U") == pytest.approx([1e-3 * 1.37 ** 2, -2e-3 * 1.37 * 0.81], rel=1e-10)
    assert r.probe(p, "sx") == pytest.approx(100 - 8.1, rel=1e-12)
    lv = r.line_values([1, 0], [1, 2], "sx", n=5)
    assert np.allclose(lv["values"], 100 - 10 * np.linspace(0, 2, 5))
    lin = r.linearize([1, 0], [1, 2])
    # SCL along y: σx is perpendicular to the SCL → membrane 90, bending ±10 (6/t²∫σ(t/2 − x) dx = 10)
    a, b = lin["membrane"]["global"], lin["bending"]["global"]
    assert a[0] == pytest.approx(90.0, rel=1e-9) and abs(b[0]) == pytest.approx(10.0, rel=1e-9)
    assert a[2] == pytest.approx(7.0)
    r.save(tmp_path / "r.npz")
    q = Results2D.load(tmp_path / "r.npz")
    assert np.array_equal(q.tri, r.tri) and q.steps[0]["reactions"]["a"].tolist() == [1.0, 2.0]
    assert q.mesh is not None and np.array_equal(q.mesh.seg, r.mesh.seg) and q.warnings == ["pozor"]
    assert q.summary()["steps"][0]["max_mises"] == pytest.approx(r.summary()["steps"][0]["max_mises"])
    out = r.render_png(tmp_path / "a.png", "sx", mesh_lines=True, scale=10)
    assert out.stat().st_size > 5000
    ra = _fake_results("axisym")
    assert ra.probe(p, "st") == pytest.approx(7.0)
    assert ra.render_png(tmp_path / "b.png", "mises").exists()


# ------------------------------------------------------------------ slow: Gmsh + ccx
@pytest.mark.slow
@pytest.mark.parametrize("key", ["A1", "A2", "A3", "A4", "A5", "A5P"])
def test_twod_verification(tmp_path, key):
    from fea_core.twod.run import run_2d
    import verification.cases  # noqa: F401 – order of the imports (cyclic import twod_cases ↔ cases)
    from verification.twod_cases import TWOD_CASES
    case = TWOD_CASES[key]
    st, parts = case.build(tmp_path)
    res = run_2d(st, tmp_path / "run", parts=parts)
    bad = [f"{c.name}: {c.value} vs {c.ref}" for c in case.evaluate(res, st, tmp_path) if not c.ok]
    assert not bad, bad


@pytest.mark.slow
def test_two_parts_gravity_force_steps(tmp_path):
    """Two coaxial cylinders on top of each other (steel + aluminium) – conforming mesh, the weight and a force in 2 states:
    the reaction of the bottom = the weight + the force over the whole circumference (axisym) and in plane, equilibrium, shared nodes of the interface.
    """
    from fea_core.geometry import Part
    from fea_core.selectors import find_faces
    from fea_core.study import Displacement, Force, Gravity, LoadCase, Material, MeshSettings
    from fea_core.twod.run import run_2d
    from verification.cases import RHO, _study, cyl
    al = Material("Al", 70000.0, 0.1, 2.7e-9)          # ν/E as steel → no transverse mismatch at the interface
    st, parts = _study("2 dily", tmp_path / "m.step", [Part("Spodek", cyl((0, 0, 0), (0, 0, 1), 10, 20)),
                                                       Part("Vrsek", cyl((0, 0, 20), (0, 0, 1), 10, 10))],
                       MeshSettings(size=1.5))
    st.add_material(al)
    st.part_materials["Vrsek"] = "Al"
    st.add_selection("dno", find_faces(parts, z=0))
    st.add_selection("vrch", find_faces(parts, z=30))
    st.constraints.append(Displacement("dno", uz=0.0))
    st.load_cases += [LoadCase("Tíha", [Gravity([0, 0, -9810.0])]),
                      LoadCase("Síla", [Force("vrch", [0, 0, -1000.0])])]
    W = 9810.0 * math.pi * 100 * (RHO * 20 + 2.7e-9 * 10)
    res = run_2d(st, tmp_path / "ax", parts=parts, settings=TwoDSettings(kind="axisym", size=1.5))
    assert res.step("Tíha")["reactions"]["dno"][1] == pytest.approx(W, rel=1e-6)
    assert res.step("Síla")["reactions"]["dno"][1] == pytest.approx(1000.0, rel=1e-6)
    # σz = −F/A in the middle of the bottom, the displacement of the top = the sum of the compression of both parts
    assert res.probe([3.0, 10.0], "sa", "Síla") == pytest.approx(-1000 / (math.pi * 100), rel=1e-3)
    uz = res.probe([0.0, 30.0], "U", "Síla")[1]
    ref = -1000 / (math.pi * 100) * (20 / E + 10 / 70000.0)
    assert uz == pytest.approx(ref, rel=1e-3)
    m = res.mesh
    shared = np.intersect1d(np.unique(m.tri[m.tri_part == 0]), np.unique(m.tri[m.tri_part == 1]))
    assert len(shared) == len(np.nonzero(np.abs(m.nodes[:, 1] - 20) < 1e-9)[0])
    # plane stress by the section x = 0 (rectangle 20 × 30, t = 5): ux = n is ignored, uy = e1 holds the displacement
    st.constraints.append(Displacement("dno", ux=0.0, uy=0.0))
    rp = run_2d(st, tmp_path / "pl", parts=parts,
                settings=TwoDSettings(kind="plane_stress", origin=[0, 0, 0], axis=[1, 0, 0], xdir=[0, 1, 0],
                                      thickness=5.0, size=1.5))
    assert rp.step("Tíha")["reactions"]["dno"][1] == pytest.approx(9810.0 * 20 * 5 * (RHO * 20 + 2.7e-9 * 10),
                                                                    rel=1e-6)
    assert any("selection area" in w for w in rp.warnings)        # force on a circular face → the 2D represents another share
    assert max(s["equilibrium"]["imbalance"] for s in rp.steps) < 1e-6
