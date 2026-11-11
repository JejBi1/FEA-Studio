"""Fast tests of the cylindrical constraint, the bearing force and the section through the mesh (without Gmsh and ccx).

Mesh = structured annulus of hexahedra divided into 6 tetrahedra (conforming), optionally tet10.
"""
import numpy as np
import pytest

from fea_core.cylinder import (fit_cylinder, selection_axis, support_equations, projectors, bearing_load,
                               CylinderError)
from fea_core.mesh import Mesh
from fea_core.selectors import FaceSelector
from fea_core.study import (Study, Material, Displacement, CylindricalSupport, BearingLoad, Force, LoadCase)

KUHN = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])


def ring_mesh(ri=10.0, ro=20.0, h=5.0, nr=2, nt=48, nz=2, axis_rot=None, order=1):
    """Annulus with the axis z (or rotated by the matrix axis_rot). Selections: vnitrni (inner), vnejsi (outer), spodek (bottom)."""
    r = np.linspace(ri, ro, nr + 1)
    t = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    z = np.linspace(0, h, nz + 1)
    R, T, Z = np.meshgrid(r, t, z, indexing="ij")
    nodes = np.column_stack([(R * np.cos(T)).ravel(), (R * np.sin(T)).ravel(), Z.ravel()])
    nid = lambda i, j, k: (i * nt + (j % nt)) * (nz + 1) + k  # noqa: E731
    tets = []
    for i in range(nr):
        for j in range(nt):
            for k in range(nz):
                v = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                tets += [[v[a] for a in tt] for tt in KUHN]
    tets = np.array(tets)
    m = Mesh(nodes, tets, np.zeros(len(tets), int), ["Kruh"])
    if order == 2:
        m = _to_tet10(m)
    tri, _ = m.boundary_faces()
    x = m.nodes[tri[:, :3]].mean(axis=1)
    rr = np.hypot(x[:, 0], x[:, 1])
    m.surfaces = {"vnitrni": tri[np.abs(rr - ri) < 0.02 * ri], "vnejsi": tri[np.abs(rr - ro) < 0.01 * ro],
                  "spodek": tri[np.abs(x[:, 2]) < 1e-9]}
    if axis_rot is not None:
        m.nodes = m.nodes @ np.asarray(axis_rot).T
    return m


def _to_tet10(m: Mesh) -> Mesh:
    """Linear tetrahedra → tet10 (ccx order) with the midpoints of the edges on the segments."""
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    keys = np.sort(np.stack([m.tets[:, [a, b]] for a, b in edges], axis=1), axis=2).reshape(-1, 2)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    mids = 0.5 * (m.nodes[uniq[:, 0]] + m.nodes[uniq[:, 1]])
    nodes = np.vstack([m.nodes, mids])
    t10 = np.hstack([m.tets, len(m.nodes) + inv.reshape(-1, 6)])
    return Mesh(nodes, t10, m.tet_part, m.part_names)


def _rot():
    a = np.array([0.3, -0.5, 0.8])
    a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(0.7) * K + (1 - np.cos(0.7)) * K @ K


def test_fit_cylinder_exact_and_half():
    rng = np.random.default_rng(1)
    d0 = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
    e1 = np.cross(d0, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d0, e1)
    for span in (2 * np.pi, np.pi, 0.6):            # whole cylinder, half cylinder, narrow strip
        th = rng.random(200) * span
        zz = rng.random(200) * 30
        c = np.array([5.0, -3.0, 2.0])
        X = c + 7.5 * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2) + zz[:, None] * d0
        N = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
        cc, d, r, err = fit_cylinder(X, N)
        assert r == pytest.approx(7.5, rel=1e-9) and err < 1e-8
        assert abs(abs(d @ d0) - 1) < 1e-10
        off = (cc - c) - ((cc - c) @ d0) * d0
        assert np.linalg.norm(off) < 1e-8


def test_fit_rejects_plane():
    m = ring_mesh()
    with pytest.raises(CylinderError):
        selection_axis(m, "spodek")


def test_selection_axis_rotated():
    R = _rot()
    m = ring_mesh(axis_rot=R)
    c, d, r = selection_axis(m, "vnitrni")
    assert r == pytest.approx(10.0 * np.cos(np.pi / 48), rel=1e-3) or r == pytest.approx(10.0, rel=1e-3)
    assert abs(abs(d @ (R @ [0, 0, 1.0])) - 1) < 1e-9
    assert np.linalg.norm(np.cross(c, d)) < 1e-6        # the axis passes through the origin


def _rings(offsets, ri=10.0, order=1, rot=None):
    """Several annuli in one mesh (shifted by offsets, optionally rotated); selection "diry" (holes) = all
    inner faces, "d0", "d1" … = individual ones.
    """
    nodes, tets, surf = [], [], {}
    off = 0
    for k, o in enumerate(offsets):
        m = ring_mesh(ri=ri, order=order)
        X = m.nodes if rot is None else m.nodes @ np.asarray(rot).T
        nodes.append(X + np.asarray(o, float))
        tets.append(m.tets + off)
        surf[f"d{k}"] = m.surfaces["vnitrni"] + off
        off += len(m.nodes)
    mesh = Mesh(np.vstack(nodes), np.vstack(tets), np.zeros(sum(len(t) for t in tets), int), ["Kruh"])
    mesh.surfaces = dict(surf, diry=np.vstack(list(surf.values())))
    return mesh


@pytest.mark.parametrize("order", [1, 2])
def test_selection_axes_several_holes(order):
    from fea_core.cylinder import selection_axes
    offs = [(0, 0, 0), (60.0, 0, 0), (0, 70.0, 0), (60.0, 70.0, 0)]
    m = _rings(offs, order=order, rot=_rot())
    with pytest.raises(CylinderError):
        selection_axis(m, "diry")                         # the bearing force still wants one axis
    g = selection_axes(m, "diry")
    assert len(g) == 4
    ax = _rot() @ [0, 0, 1.0]
    centers = sorted(np.round(c - (c @ ax) * ax, 6).tolist() for _, c, _, _ in g)
    assert sorted(np.round(np.asarray(o) - (np.asarray(o) @ ax) * ax, 6).tolist() for o in offs) == centers
    assert all(abs(abs(d @ ax) - 1) < 1e-9 for _, _, d, _ in g)
    assert sum(len(n) for n, *_ in g) == len(m.surface_nodes("diry"))
    # one hole: the same result as selection_axis (without changing the behaviour)
    (n0, c0, d0, r0), = selection_axes(m, "d0")
    c1, d1, r1 = selection_axis(m, "d0")
    assert np.array_equal(c0, c1) and np.array_equal(d0, d1) and r0 == r1
    np.testing.assert_array_equal(n0, m.surface_nodes("d0"))


def test_selection_axes_coaxial_parts_merge_and_plane_rejected():
    from fea_core.cylinder import selection_axes
    m = _rings([(0, 0, 0), (0, 0, 20.0), (50.0, 0, 0)])  # two coaxial parts (with a gap) + one elsewhere
    g = selection_axes(m, "diry")
    assert sorted(len(n) for n, *_ in g) == sorted([2 * len(m.surface_nodes("d0")), len(m.surface_nodes("d2"))])
    plane = ring_mesh()
    with pytest.raises(CylinderError):
        selection_axes(plane, "spodek")
    # selection of a hole + the plane of another annulus (disconnected) → an error with the description of the part
    m3 = _rings([(0, 0, 0), (60.0, 0, 0)])
    sp = ring_mesh().surfaces["spodek"] + len(ring_mesh().nodes)
    m3.surfaces["mix"] = np.vstack([m3.surfaces["d0"], sp])
    with pytest.raises(CylinderError, match="not a cylindrical"):
        selection_axes(m3, "mix")


def test_projectors_several_holes_equal_separate():
    offs = [(0, 0, 0), (60.0, 0, 0), (0, 70.0, 0), (60.0, 70.0, 0)]
    m = _rings(offs, order=2)
    cyl = FaceSelector("Kruh", "cylinder", 1.0, [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0, 1, 1, 1])
    st1 = Study("t", "m.step")
    st1.add_selection("diry", [cyl])
    st1.constraints.append(CylindricalSupport("diry", True, True, False))
    st2 = Study("t", "m.step")
    for k in range(4):
        st2.add_selection(f"d{k}", [cyl])
        st2.constraints.append(CylindricalSupport(f"d{k}", True, True, False))
    (idx, P), = projectors(st1, m).values()
    P2 = {}
    for idk, Pk in projectors(st2, m).values():
        P2.update(zip(idk.tolist(), Pk))
    assert sorted(P2) == idx.tolist()
    assert all(np.array_equal(P[i], P2[n]) for i, n in enumerate(idx.tolist()))
    assert support_equations(st1, m) == support_equations(st2, m)


def _study(*constraints):
    st = Study("t", "m.step")
    st.add_material(Material("Ocel", 210000, 0.3, 7.85e-9))
    cyl = FaceSelector("Kruh", "cylinder", 1.0, [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0, 1, 1, 1])
    st.add_selection("vnitrni", [cyl])
    st.add_selection("spodek", [FaceSelector("Kruh", "plane", 1.0, [0, 0, 0], [0, 0, 0], [0, 0, -1],
                                             [0, 0, 0, 1, 1, 0])])
    st.constraints += list(constraints)
    return st


@pytest.mark.parametrize("flags", [(True, False, False), (True, True, False), (True, True, True), (False, True, True)])
def test_support_equations_with_boundary(flags):
    m = ring_mesh(axis_rot=_rot())
    st = _study(CylindricalSupport("vnitrni", *flags), Displacement("spodek", ux=0.0))
    eqs = support_equations(st, m)
    bnd = set(m.surface_nodes("spodek").tolist())
    rows = {}
    for n, terms in eqs:
        dep = terms[0][0]
        assert terms[0][1] == 1.0
        assert not (n in bnd and dep == 0)            # the dependent DOF is not in *BOUNDARY
        rows.setdefault(n, []).append((dep, terms))
    (idx, P), = projectors(st, m).values()
    Pn = dict(zip(idx.tolist(), P))
    for n, lst in rows.items():
        deps = [d for d, _ in lst]
        assert len(deps) == len(set(deps))            # each dependent DOF only in one equation
        for d, terms in lst:
            for d2, _ in lst:
                if d2 != d:
                    assert all(k != d2 for k, _ in terms[1:])   # the pivot of another equation does not occur
        # equations ⊂ envelope (constrained directions + DOF in *BOUNDARY), the number = rank
        for _, terms in lst:
            c = np.zeros(3)
            for k, v in terms:
                c[k] = v
            S = Pn[n] + (np.diag([1.0, 0, 0]) if n in bnd else 0)
            u, s, _ = np.linalg.svd(S)
            Q = u[:, s > 1e-9]
            assert np.linalg.norm(c - Q @ (Q.T @ c)) < 1e-9 * np.linalg.norm(c)
    # nodes off the boundary have as many equations as there are constrained components
    free = [n for n in idx.tolist() if n not in bnd]
    assert all(len(rows[n]) == sum(flags) for n in free)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("surf,F", [("vnitrni", [100.0, 0, 0]), ("vnitrni", [30.0, -40.0, 25.0]),
                                    ("vnejsi", [0.0, 80.0, 0.0]), ("vnitrni", [0, 0, 50.0])])
def test_bearing_resultant_and_side(surf, F, order):
    m = ring_mesh(order=order)
    f, info = bearing_load(m, surf, F)
    np.testing.assert_allclose(f.sum(axis=0), F, atol=1e-9 * max(np.linalg.norm(F), 1))
    M = np.cross(m.nodes - info["axis_point"], f).sum(axis=0)
    assert abs(M @ info["axis_dir"]) < 1e-9 * np.linalg.norm(F) * 10
    Fr = np.asarray(F) - (np.asarray(F) @ [0, 0, 1.0]) * np.array([0, 0, 1.0])
    if np.linalg.norm(Fr) > 0:
        fh = Fr / np.linalg.norm(Fr)
        loaded = np.nonzero(np.linalg.norm(f[:, :2], axis=1) > 1e-9)[0]
        side = m.nodes[loaded, :2] @ fh[:2]
        # hole: pushed to the side in the direction of the force; pin (outer surface): to the opposite side
        # (nodes of elements cut by the boundary of the loaded half may lie up to one element beyond it)
        cell = 1.01 * 2 * np.pi * 20.0 / 48
        assert np.all(side > -cell) if surf == "vnitrni" else np.all(side < cell)
        far = side < -cell if surf == "vnitrni" else side > cell
        assert np.sum(far) == 0 and np.sum(np.abs(side) > cell) > 5


def test_bearing_rejects_plane():
    m = ring_mesh()
    with pytest.raises(CylinderError):
        bearing_load(m, "spodek", [1.0, 0, 0])


def test_json_roundtrip_and_checks():
    from fea_core.solve import check_study
    from fea_core.solve_large import unsupported
    st = _study(CylindricalSupport("vnitrni", radial=True, tangential=False, axial=True))
    st.load_cases.append(LoadCase("LC", [BearingLoad("vnitrni", [1.0, 2.0, 0.0]), Force("spodek", [0, 0, 1.0])]))
    d = st.to_dict()
    st2 = Study.from_dict(d)
    assert st2.to_dict() == d
    assert isinstance(st2.constraints[0], CylindricalSupport) and st2.constraints[0].tangential is False
    assert isinstance(st2.load_cases[0].loads[0], BearingLoad)
    assert any("cylindrical" in u for u in unsupported(st2))
    st2.load_cases[0].loads.append(BearingLoad("spodek", [1.0, 0, 0]))
    errors, _ = check_study(st2, ["Kruh"])
    assert any("Bearing load" in e and "spodek" in e for e in errors)


# ------------------------------------------------------------------ section through the mesh (fea_core.section)

def _area(P):
    return 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1).sum()


@pytest.mark.parametrize("order", [1, 2])
def test_plane_section_area_and_linear_field(order):
    from fea_core.section import plane_section
    ri, ro, nt = 10.0, 20.0, 48
    m = ring_mesh(ri, ro, 5.0, nt=nt, order=order)
    f = m.nodes @ np.array([1.0, 2.0, 3.0])                 # linear field → interpolation exact
    P, V = plane_section(m.nodes, m.tets, f, [0, 0, 2.5], [0, 0, 1.0])
    poly = 0.5 * nt * np.sin(2 * np.pi / nt) * (ro ** 2 - ri ** 2)
    assert _area(P) == pytest.approx(poly, rel=1e-7)
    np.testing.assert_allclose(V, P @ np.array([1.0, 2.0, 3.0]), atol=1e-9)
    nn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    assert np.all(nn[:, 2] > 0)                               # normals in the direction n
    # inclined plane: values still exact, the points lie in the plane
    n = np.array([0.3, -0.2, 1.0])
    P, V = plane_section(m.nodes, m.tets, f, [1.0, 2.0, 2.0], n)
    assert len(P) > 0
    np.testing.assert_allclose((P - [1.0, 2.0, 2.0]) @ (n / np.linalg.norm(n)), 0, atol=1e-7)   # shift 1e-9·size
    np.testing.assert_allclose(V, P @ np.array([1.0, 2.0, 3.0]), atol=1e-9)


def test_plane_section_deformed_positions():
    from fea_core.section import plane_section
    m = ring_mesh(order=2)
    X = m.nodes + np.array([5.0, 0, 0])                     # shifted display, the section by the undeformed position
    P0, _ = plane_section(m.nodes, m.tets, m.nodes[:, 0], [0, 0, 2.5], [0, 0, 1])
    P1, _ = plane_section(m.nodes, m.tets, m.nodes[:, 0], [0, 0, 2.5], [0, 0, 1], X)
    np.testing.assert_allclose(P1, P0 + [5.0, 0, 0], atol=1e-12)


def test_clip_surface_partition():
    from fea_core.section import clip_surface
    m = ring_mesh(order=2)
    tri, _ = m.boundary_faces()
    f = m.nodes[:, 1]
    o, n = np.array([3.0, 0, 0]), np.array([1.0, 0.4, 0.2])
    Pa, Va = clip_surface(m.nodes, tri, f, o, n)
    Pb, Vb = clip_surface(m.nodes, tri, f, o, -n)
    from fea_core.section import SUBTRI
    total = _area(m.nodes[tri[:, SUBTRI].reshape(-1, 3)])
    assert _area(Pa) + _area(Pb) == pytest.approx(total, rel=1e-7)
    nu = n / np.linalg.norm(n)
    assert ((Pa - o) @ nu).max() < 1e-7 and ((Pb - o) @ nu).min() > -1e-7
    np.testing.assert_allclose(Va, Pa[:, :, 1], atol=1e-9)


def test_section_speed_50k():
    import time
    from fea_core.section import plane_section, clip_surface
    m = ring_mesh(10.0, 40.0, 20.0, nr=8, nt=160, nz=7, order=2)   # 53 760 tet10
    assert len(m.tets) > 50000
    tri, _ = m.boundary_faces()
    f = np.linalg.norm(m.nodes, axis=1)
    t0 = time.perf_counter()
    P, V = plane_section(m.nodes, m.tets, f, [0, 0, 0], [1.0, 0.3, 0.1])
    Q, W = clip_surface(m.nodes, tri, f, [0, 0, 0], [1.0, 0.3, 0.1])
    dt = time.perf_counter() - t0
    assert len(P) > 1000 and len(Q) > 1000
    assert dt < 1.5, dt


def test_section_outline_closed():
    from fea_core.section import plane_section, section_outline
    m = ring_mesh(order=2)
    P, _ = plane_section(m.nodes, m.tets, m.nodes[:, 0], [0, 0, 2.5], [0, 0, 1])
    E = section_outline(P)
    L = np.linalg.norm(E[:, 1] - E[:, 0], axis=1).sum()
    nt = 48
    ref = 2 * nt * np.sin(np.pi / nt) * (10.0 + 20.0)        # perimeters of the inner and outer polygon
    assert L == pytest.approx(ref, rel=1e-7)


# ------------------------------------------------------------------ plate with 4 holes (Gmsh + ccx)

@pytest.mark.slow
def test_plate_four_holes_one_selection(tmp_path):
    """One selection with 4 non-coaxial holes = 4 separate cylindrical constraints (the same mesh): displacements and the sum of the reactions."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from fea_core.geometry import Part
    from fea_core.selectors import find_faces
    from fea_core.solve import run_study
    from fea_core.study import MeshSettings
    from verification.cases import box, cyl, solid, _study as vstudy
    holes = [(20.0, 20.0), (100.0, 20.0), (20.0, 60.0), (100.0, 60.0)]
    shape = box(0, 0, 0, 120, 80, 10)
    for x, y in holes:
        shape = BRepAlgoAPI_Cut(shape, cyl((x, y, -1), (0, 0, 1), 6.0, 12)).Shape()
    st1, parts = vstudy("deska 4 diry", tmp_path / "model.step", [Part("Deska", solid(shape))], MeshSettings(size=6.0))
    st1.add_selection("diry", find_faces(parts, radius=6.0))
    for k, (x, y) in enumerate(holes):
        st1.add_selection(f"d{k}", find_faces(parts, radius=6.0, where=lambda p, x=x, y=y: abs(
            0.5 * (p["bbox"][0] + p["bbox"][3]) - x) < 7 and abs(0.5 * (p["bbox"][1] + p["bbox"][4]) - y) < 7))
        assert st1.selections[f"d{k}"].faces
    st1.add_selection("bok", find_faces(parts, x=120.0))
    st1.add_selection("vrch", find_faces(parts, z=10.0))
    F1, F2 = np.array([1000.0, 2000.0, 500.0]), np.array([0.0, 300.0, -1500.0])
    st1.load_cases.append(LoadCase("LC", [Force("bok", F1.tolist()), Force("vrch", F2.tolist())]))
    st2 = Study.from_dict(st1.to_dict())
    st1.constraints.append(CylindricalSupport("diry", True, True, True))
    st2.constraints += [CylindricalSupport(f"d{k}", True, True, True) for k in range(4)]
    r1 = run_study(st1, tmp_path / "a", parts=parts)
    r2 = run_study(st2, tmp_path / "b", mesh=r1.mesh)
    s1, s2 = r1.steps[0], r2.steps[0]
    F = F1 + F2
    np.testing.assert_allclose(s1.reactions["diry"], -F, rtol=0, atol=1e-5 * np.linalg.norm(F))
    assert s1.equilibrium["imbalance"] < 1e-5
    umax = float(np.abs(s2.U).max())
    assert float(np.abs(s1.U - s2.U).max()) <= 1e-9 * umax
    R2 = sum(s2.reactions[f"d{k}"] for k in range(4))
    np.testing.assert_allclose(s1.reactions["diry"], R2, rtol=0, atol=1e-9 * np.linalg.norm(F))
