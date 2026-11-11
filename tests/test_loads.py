"""Fast tests of the loads (without Gmsh and ccx): force on an edge / at a point, local and cylindrical coordinate system, variable
(linear, hydrostatic) pressure, rotation, JSON of the study, readiness check, writing of the .inp, descriptions.
Calculations against the analytics: verification V11–V14 (slow, python -m verification.run_all V11 V12 V13 V14).
"""
import math

import numpy as np
import pytest

from fea_core import csys
from fea_core.loads import (force_nodal, pressure_nodal, rotation_forces, nodal_loads, load_resultant,
                            line_quadrature)
from fea_core.mesh import Mesh
from fea_core.selectors import EdgeSelector, PointSelector, FaceSelector
from fea_core.study import (Study, Material, Force, Pressure, Rotation, Fixed, LoadCase, CoordinateSystem,
                            RemoteLoad, Gravity, BearingLoad, Selection)

KUHN = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])
RHO = 7.85e-9


def bar_mesh(L=100.0, a=10.0, nx=10, ny=2, nz=2) -> Mesh:
    """Bar L × a × a (tet10, midpoints on the segments). Selections x0, x100, y0, z10; edges: the upper edge of the face, a longitudinal edge;
    bodies: the centre of the face and 2 corners.
    """
    x, y, z = np.linspace(0, L, nx + 1), np.linspace(0, a, ny + 1), np.linspace(0, a, nz + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                tets += [[v[q] for q in t] for t in KUHN]
    tets = np.array(tets)
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    keys = np.sort(np.stack([tets[:, [p, q]] for p, q in edges], axis=1), axis=2).reshape(-1, 2)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    n0 = len(nodes)
    nodes = np.vstack([nodes, 0.5 * (nodes[uniq[:, 0]] + nodes[uniq[:, 1]])])
    tets = np.hstack([tets, n0 + inv.reshape(-1, 6)])
    m = Mesh(nodes, tets, np.zeros(len(tets), int), ["Tyc"])
    tri, _ = m.boundary_faces()
    c = m.nodes[tri[:, :3]].mean(axis=1)
    m.surfaces = {"x0": tri[np.abs(c[:, 0]) < 1e-9], "x100": tri[np.abs(c[:, 0] - L) < 1e-9],
                  "y0": tri[np.abs(c[:, 1]) < 1e-9], "z10": tri[np.abs(c[:, 2] - a) < 1e-9]}
    mid = {tuple(k): n0 + i for i, k in enumerate(uniq)}

    def line(ids):
        return np.array([[p, q, mid[tuple(sorted((p, q)))]] for p, q in zip(ids[:-1], ids[1:])])
    m.edges = {"hrana_cela": line([nid(nx, j, nz) for j in range(ny + 1)]),          # x = L, z = a, along y
               "hrana_horni": line([nid(i, 0, nz) for i in range(nx + 1)])}          # y = 0, z = a, along x
    m.points = {"stred": np.array([nid(nx, ny // 2, nz // 2)]), "rohy": np.array([nid(nx, 0, nz), nid(nx, ny, nz)])}
    return m


def bar_study() -> Study:
    st = Study("t", "x.step")
    st.add_material(Material("Ocel", 210000, 0.3, RHO), default=True)
    for s in ("x0", "x100", "y0", "z10"):
        st.selections[s] = Selection(s, [FaceSelector("Tyc", "plane", 1, [0, 0, 0], [0, 0, 0], [0, 0, 1], [0] * 6)])
    st.selections["hrana_cela"] = Selection("hrana_cela", edges=[EdgeSelector("Tyc", "line", 10, [100, 5, 10],
                                                                               [100, 5, 10], [100, 0, 10, 100, 10, 10])])
    st.selections["hrana_horni"] = Selection("hrana_horni", edges=[EdgeSelector("Tyc", "line", 100, [50, 0, 10],
                                                                                 [50, 0, 10], [0, 0, 10, 100, 0, 10])])
    st.selections["stred"] = Selection("stred", points=[PointSelector("Tyc", [100, 5, 5])])
    st.selections["rohy"] = Selection("rohy", points=[PointSelector("Tyc", [100, 0, 10]), PointSelector("Tyc", [100, 10, 10])])
    st.constraints.append(Fixed("x0"))
    return st


# ---------------------------------------------------------------- model a JSON
def test_json_roundtrip_new_loads():
    st = bar_study()
    st.coordinate_systems["V"] = CoordinateSystem("V", "cylindrical", [1, 2, 3], [1, 0, 0], [0, 1, 0])
    st.load_cases = [LoadCase("A", [Force("hrana_cela", [0, 0, -1], csys="V"), Force("stred", [1, 2, 3]),
                                    Pressure("y0", 0.1, [0, 0, -1e-5], [0, 0, 5], clip=True),
                                    Rotation([0, 0, 0], [0, 0, 1], 100.0, 5.0, ["Tyc"]),
                                    RemoteLoad("x100", [1, 2, 3], [1, 0, 0], [0, 0, 0], csys="V"),
                                    Gravity([0, 0, -9810], csys=None), BearingLoad("x100", [1, 0, 0])])]
    d = st.to_dict()
    st2 = Study.from_dict(d)
    assert st2.to_dict() == d
    assert st2.selections["hrana_cela"].entity == "edges" and st2.selections["rohy"].entity == "points"
    assert st2.selections["x0"].entity == "faces"
    assert st2.load_cases[0].loads[2].variable and st2.load_cases[0].loads[2].clip
    old = dict(d)                                                  # a file of an older version: without the new keys
    old.pop("coordinate_systems")
    assert Study.from_dict(old).coordinate_systems == {}
    with pytest.raises(ValueError):
        st.add_selection("smes", [FaceSelector("Tyc", "plane", 1, [0] * 3, [0] * 3, [0, 0, 1], [0] * 6)],
                         edges=[EdgeSelector("Tyc", "line", 1, [0] * 3, [0] * 3, [0] * 6)])


# ---------------------------------------------------------------- coordinate systems
def test_csys_cartesian_and_cylindrical():
    st = Study("t", "x.step")
    st.coordinate_systems["K"] = CoordinateSystem("K", "cartesian", [5, 5, 5], [1, 0, 0], [0, 1, 0.2])
    R = csys.matrix(st.coordinate_systems["K"])
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(R[2], [1, 0, 0]) and np.allclose(R[0], np.array([0, 1, 0.2]) / np.hypot(1, 0.2))  # x ⟂ z
    assert np.allclose(csys.vector_global(st, "K", [0, 0, 7]), [7, 0, 0])
    st.coordinate_systems["C"] = CoordinateSystem("C", "cylindrical", [0, 0, 0], [0, 0, 1], [1, 0, 0])
    X = np.array([[3, 0, 1], [0, 2, 5], [-1, -1, 0]])
    v = csys.field_global(st, "C", [1, 0, 0], X)                                   # radial
    assert np.allclose(v, np.c_[X[:, :2] / np.linalg.norm(X[:, :2], axis=1)[:, None], np.zeros(3)])
    t = csys.field_global(st, "C", [0, 1, 0], X)                                   # tangential = z × r
    assert np.allclose(t[0], [0, 1, 0]) and np.allclose(t[1], [-1, 0, 0])
    assert np.allclose(csys.field_global(st, "C", [0, 0, 2], np.array([[0, 0, 4]])), [[0, 0, 2]])   # axial and on the axis
    with pytest.raises(csys.CsysError):
        csys.field_global(st, "C", [1, 0, 0], np.array([[0, 0, 4]]))               # radial on the axis
    with pytest.raises(csys.CsysError):
        csys.vector_global(st, "C", [1, 0, 0])                                      # cylindrical without a point
    with pytest.raises(csys.CsysError):
        csys.vector_global(st, "neni", [1, 0, 0])
    assert csys.check(CoordinateSystem("Z", "cartesian", [0, 0, 0], [0, 0, 1], [0, 0, 2]))   # x ∥ z


# ---------------------------------------------------------------- force on an edge and at a point
def test_edge_force_uniform_per_length():
    m, st = bar_mesh(), bar_study()
    F = force_nodal(st, m, Force("hrana_horni", [0, 0, -100.0]))
    assert np.allclose(F.sum(axis=0), [0, 0, -100.0], atol=1e-12)
    seg = m.edges["hrana_horni"]
    h = 100.0 / len(seg)
    q = -1.0                                                     # N/mm
    ends = np.setdiff1d(seg[:, :2], seg[:, :2][1:-1].ravel())    # ends of the edge (only one segment)
    assert np.allclose(F[seg[:, 2], 2], 2 * q * h / 3)           # line3: midpoints 2/3, corners 1/6 (shared 2× 1/6)
    assert np.allclose(F[seg[0, 0], 2], q * h / 6) and np.allclose(F[seg[1, 0], 2], q * h / 3)
    M = np.cross(m.nodes, F).sum(axis=0)                         # resultant at the middle of the edge (50, 0, 10)
    assert np.allclose(M, np.cross([50, 0, 10], [0, 0, -100.0]), atol=1e-9)
    assert len(ends) >= 1


def test_line_quadrature_arc_length():
    t = np.linspace(0, np.pi / 2, 9)
    R = 10.0
    P = np.c_[R * np.cos(t), R * np.sin(t), np.zeros_like(t)]
    seg = np.array([[2 * i, 2 * i + 2, 2 * i + 1] for i in range(4)])          # line3 through the points of the arc
    N, x, wj = line_quadrature(P, seg)
    assert abs(wj.sum() - R * np.pi / 2) < 2e-4 * R


def test_point_force_split_and_cylindrical():
    m, st = bar_mesh(), bar_study()
    F = force_nodal(st, m, Force("rohy", [0, 0, -100.0]))
    assert np.allclose(F[m.points["rohy"], 2], -50.0) and np.isclose(F.sum(), -100.0)
    st.coordinate_systems["C"] = CoordinateSystem("C", "cylindrical", [100, 5, 5], [1, 0, 0], [0, 1, 0])
    F = force_nodal(st, m, Force("rohy", [10.0, 0, 0], csys="C"))                # radial from the axis of the face
    for n in m.points["rohy"]:
        r = m.nodes[n] - [100, 5, 5]
        assert np.allclose(F[n], 5.0 * r / np.linalg.norm(r))
    with pytest.raises(csys.CsysError):                                          # bod na ose
        force_nodal(st, m, Force("stred", [10.0, 0, 0], csys="C"))


def test_face_force_cartesian_csys_equals_rotated_global():
    m, st = bar_mesh(), bar_study()
    st.coordinate_systems["K"] = CoordinateSystem("K", "cartesian", [0, 0, 0], [1, 0, 0], [0, 1, 0])  # y_l = z
    a = force_nodal(st, m, Force("x100", [0, -100.0, 0], csys="K"))
    b = force_nodal(st, m, Force("x100", [0, 0, -100.0]))
    assert np.array_equal(a, b)


def test_face_force_cylindrical_torque():
    """Face of the bar 10×10 with the axis at its centre: tangential force F_θ → moment ∫τ·r dA, resultant ≈ 0."""
    m, st = bar_mesh(nx=2, ny=8, nz=8), bar_study()
    st.coordinate_systems["C"] = CoordinateSystem("C", "cylindrical", [100, 5, 5], [1, 0, 0], [0, 1, 0])
    Ft = 1000.0
    F = force_nodal(st, m, Force("x100", [0, Ft, 0], csys="C"))
    # ∫ r dA over the square a×a centred on the axis = a³/6·(√2 + ln(1 + √2))
    a = 10.0
    T = Ft / a ** 2 * a ** 3 / 6 * (math.sqrt(2) + math.log(1 + math.sqrt(2)))
    M = np.cross(m.nodes - [100, 5, 5], F).sum(axis=0)
    assert abs(M[0] - T) < 1e-4 * T, (M, T)
    assert np.linalg.norm(F.sum(axis=0)) < 1e-5 * Ft


# ---------------------------------------------------------------- variable pressure
def test_linear_pressure_resultant_and_moment():
    m = bar_mesh()
    p0, g, o = 0.5, np.array([0.0, 0.03, -0.02]), np.array([100.0, 5.0, 5.0])
    l = Pressure("x100", p0, g.tolist(), o.tolist())
    F = pressure_nodal(m, l)
    A = 100.0                                                     # face 10×10, normal +x → force −x
    assert np.allclose(F.sum(axis=0), [-p0 * A, 0, 0], atol=1e-10)              # the linear term through the centre = 0
    M = np.cross(m.nodes - o, F).sum(axis=0)
    I = 10 ** 4 / 12
    assert np.allclose(M, [0, -g[2] * I, g[1] * I], atol=1e-9)    # Σ r' × (−p n dA) = (0, −∫p z', ∫p y'), z' = z − 5
    c = pressure_nodal(m, Pressure("x100", p0))
    v = pressure_nodal(m, Pressure("x100", p0, [0, 0, 1e-300], [0, 0, 0]))
    assert np.allclose(c, v, atol=1e-14)


def test_hydrostatic_clip_integral():
    """Level x = 35 (it cuts the elements), weight in the direction +x: p = γ·max(0, x − 35) on the face y0 (100 × 10)."""
    m = bar_mesh()
    gam = 9.81e-6
    l = Pressure("y0", 0.0, [gam, 0, 0], [35, 0, 0], clip=True)
    F = pressure_nodal(m, l)
    P = gam * 10 * 65 ** 2 / 2
    assert abs(F[:, 1].sum() - P) < 1e-5 * P                      # plane y = 0, normal −y → force +y
    xc = 35 + 2 / 3 * 65                                          # centroid of the triangular distribution
    assert abs((m.nodes[:, 0] * F[:, 1]).sum() / F[:, 1].sum() - xc) < 1e-4


# ---------------------------------------------------------------- rotation
def test_rotation_forces_resultant_and_torque():
    m, st = bar_mesh(), bar_study()
    w, al = 100.0, 50.0
    l = Rotation([0, 0, 0], [0, 0, 2.0], w, al)
    F = rotation_forces(st, m, l)
    V = 100 * 10 * 10
    c = np.array([50, 5, 5])
    assert np.allclose(F.sum(axis=0), RHO * V * (w ** 2 * np.array([c[0], c[1], 0]) - al * np.cross([0, 0, 1], c)),
                       rtol=1e-12)
    Jz = RHO * (100 * 10 * 10) * ((100 ** 2 + 10 ** 2) / 3)        # ∫(x² + y²) dV, x ∈ [0, 100], y ∈ [0, 10]
    assert np.isclose(np.cross(m.nodes, F).sum(axis=0)[2], -al * Jz, rtol=1e-12)
    st2 = bar_study()
    assert np.allclose(rotation_forces(st2, m, Rotation([0, 0, 0], [0, 0, 1], w, 0, ["Jiny"])), 0)
    lc = LoadCase("R", [l])
    assert np.allclose(load_resultant(st, m, lc), nodal_loads(st, m, lc)[0].sum(axis=0))


# ---------------------------------------------------------------- check, .inp, descriptions
def test_check_loads_errors():
    from fea_core.solve import check_study
    st = bar_study()
    st.constraints.append(Fixed("hrana_cela"))
    st.load_cases = [LoadCase("A", [Pressure("stred", 1.0), Force("hrana_cela", [0, 0, 1], csys="neni"),
                                    RemoteLoad("x100", [0, 0, 0], [1, 0, 0], csys="C"),
                                    Rotation([0, 0, 0], [0, 0, 0], 1.0)])]
    st.coordinate_systems["C"] = CoordinateSystem("C", "cylindrical")
    st.materials["Ocel"].rho = 0.0
    errors, warnings = check_study(st, ["Tyc"])
    txt = "\n".join(errors)
    assert "Support 'hrana_cela'" in txt and "faces only" in txt
    assert "load 'stred'" in txt and "does not exist" in txt and "cylindrical coordinate system is only available for force" in txt
    assert "has no density" in txt and "axis" in txt
    assert any("singular" in w for w in warnings)


def test_inp_variable_pressure_rotation_points(tmp_path):
    from fea_core.ccx.inp import write_inp
    m, st = bar_mesh(), bar_study()
    st.load_cases = [LoadCase("A", [Pressure("z10", 1.0), Pressure("y0", 0.0, [0, 0, -1e-3], [0, 0, 10], clip=True),
                                    Rotation([0, 0, 0], [0, 0, 1], 10.0), Force("rohy", [0, 0, -100])])]
    write_inp(st, m, tmp_path / "a.inp")
    txt = (tmp_path / "a.inp").read_text()
    step = txt.split("*STEP")[1]
    assert ", P, " not in step                                    # also the constant pressure *CLOAD (not *DLOAD)
    cl = step.split("*CLOAD")[1].split("*")[0].strip().splitlines()
    F = np.zeros((len(m.nodes), 3))
    for row in cl:
        n, d, v = row.split(",")
        F[int(n) - 1, int(d) - 1] += float(v)
    ref = nodal_loads(st, m, st.load_cases[0])[0]
    assert np.allclose(F, ref, rtol=1e-10, atol=1e-12)


def test_load_texts():
    from fea_core.loads_text import load_text
    st = bar_study()
    st.coordinate_systems["C"] = CoordinateSystem("C", "cylindrical")
    assert "edge" in load_text(Force("hrana_cela", [0, 0, -1], csys="C"), st) and "(r, θ, z)" in \
        load_text(Force("hrana_cela", [0, 0, -1], csys="C"), st)
    assert "2 points of F/2 each" in load_text(Force("rohy", [0, 0, -1]), st)
    assert "hydrostatic" in load_text(Pressure("y0", 0, [0, 0, -9.81e-6], [0, 0, 5], True), st)
    assert "1000 kg/m³" in load_text(Pressure("y0", 0, [0, 0, -9.81e-6], [0, 0, 5], True), st)
    assert "rpm" in load_text(Rotation(omega=2 * math.pi), st)
