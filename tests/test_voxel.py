"""Voxel detail of 3D printing (F5 B): path generator, preparation of the submodel, boundary conditions, writing of the .inp, storage and
image. Fast tests without Gmsh and ccx (analytic block, structured tet10 mesh, synthetic results); slow: the bending of
a beam of C3D8I and the generator (verification.voxel_cases X2, X4).
"""
import math

import numpy as np
import pytest

from fea_core import materials as matlib
from fea_core.mesh import Mesh
from fea_core.printing import PrintSpec
from fea_core.printing import voxel as V
from fea_core.printing import voxel_paths as vp
from fea_core.results import Results, StepResult
from fea_core.study import Study

PLA_NAME = "Tisk PLA Bambu Basic"


def box_mesh(name, x0, d, h):
    """Structured tet10 mesh of a block (Kuhn division of the cubes into 6 tetrahedra), node order of ccx."""
    kuhn = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    mids = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    n = [max(1, int(round(d[k] / h))) for k in range(3)]
    g = [np.linspace(x0[k], x0[k] + d[k], n[k] + 1) for k in range(3)]
    tets = []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                cube = np.array([[g[0][i + (q & 1)], g[1][j + ((q >> 1) & 1)], g[2][k + ((q >> 2) & 1)]]
                                 for q in range(8)])
                for t in kuhn:
                    X = cube[list(t)]
                    if np.linalg.det(X[1:] - X[0]) < 0:
                        X = X[[0, 2, 1, 3]]
                    tets.append(np.vstack([X] + [0.5 * (X[a] + X[b]) for a, b in mids]))
    P = np.concatenate(tets)
    uniq, inv = np.unique(np.round(P, 9), axis=0, return_inverse=True)
    return Mesh(uniq, inv.reshape(-1, 10).astype(np.int64), np.zeros(len(tets), np.int64), [name])


def _study(**spec):
    st = Study(name="t", geometry="x.step")
    st.add_material(matlib.by_name(PLA_NAME), default=True)
    st.printing.parts["A"] = PrintSpec("A", **spec)
    return st


def _results(mesh, eps=1e-3):
    """Linear field of the displacements (uniaxial tension x) and the print safety with the minimum at the point (6, 5, 3)."""
    x = mesh.nodes
    U = np.column_stack([eps * x[:, 0], -0.3 * eps * x[:, 1], -0.3 * eps * x[:, 2]])
    S = np.zeros((len(x), 6))
    S[:, 0] = 2680 * eps
    lam = 5.0 + np.linalg.norm(x - [6, 5, 3], axis=1)
    return Results(mesh, [StepResult("Tah", U, S, fields={"print_safety": lam})])


BOX = np.array([30.0, 30.0, 4.0])


def _inside_box(B):
    return lambda q: np.all((q > 0) & (q < B), axis=1)


def _gen(spec, B=BOX, dx=0.21):
    return vp.generate(spec, np.eye(3), np.zeros(3), -np.ones(3), B + 1, dx, _inside_box(B), np.zeros(3), B)


# ===================================================================== path generator
def test_perimeters_skin_layers():
    sp = PrintSpec("A", walls=3, line_width=0.42, layer_height=0.2, top_layers=4, bottom_layers=3, infill=0.15)
    g = _gen(sp)
    assert g.free_faces == ["x-", "x+", "y-", "y+", "z-", "z+"]
    assert sorted(np.unique(g.wall[g.wall >= 0]).tolist()) == [0, 1, 2]
    row = g.kind[:, g.shape[1] // 2, g.shape[2] // 2]
    assert (row == vp.PERIMETER).sum() == 12                      # 3 perimeters × 2 voxels (w/2) on each side
    col = g.kind[g.shape[0] // 2, g.shape[1] // 2, :]
    assert (col[:3] == vp.SKIN).all() and (col[-4:] == vp.SKIN).all()
    assert not np.isin(col[3:-4], (vp.SKIN,)).any()
    # direction of the perimeter = tangent of the outline: at the wall y = 0 along the axis 1 (angle 0), at the wall x = 0 along the axis 2 (90°)
    k = g.shape[2] // 2
    assert abs(math.sin(g.angle[g.shape[0] // 2, 0, k])) < 1e-6
    assert abs(math.cos(g.angle[0, g.shape[1] // 2, k])) < 1e-6
    # full layers ±45° alternately
    a0, a1 = g.angle[g.shape[0] // 2, g.shape[1] // 2, 0], g.angle[g.shape[0] // 2, g.shape[1] // 2, 1]
    assert {round(math.degrees(a0)), round(math.degrees(a1))} == {45, 135}


@pytest.mark.parametrize("pattern,rho,tol", [("grid", 0.15, 0.02), ("grid", 0.4, 0.02), ("rectilinear", 0.25, 0.02),
                                             ("gyroid", 0.2, 0.03), ("triangles", 0.2, 0.05)])
def test_infill_fraction(pattern, rho, tol):
    sp = PrintSpec("A", walls=2, top_layers=0, bottom_layers=0, infill=rho, pattern=pattern)
    fr = vp.infill_fraction(_gen(sp, np.array([30.0, 30.0, 4.0])))
    assert fr == pytest.approx(rho, rel=tol)


def test_gyroid_fraction_over_period():
    """Gyroid: the share in the layer changes with the height, the average over a period = density (±1 %)."""
    dx, w, rho = 0.21, 0.42, 0.15
    u = (np.arange(int(30 / dx)) + 0.5) * dx
    U, Vv = np.meshgrid(u, u, indexing="ij")
    P = 2 * math.pi / (2 * vp.gyroid_scale(rho) / w)
    fr = [vp.infill_layer("gyroid", rho, w, U, Vv, (L + 0.5) * 0.2, L)[0].mean() for L in range(int(P / 0.2) + 1)]
    assert np.mean(fr) == pytest.approx(rho, rel=0.01)


def test_full_and_empty_infill_and_thin():
    g = _gen(PrintSpec("A", walls=2, top_layers=0, bottom_layers=0, infill=1.0))
    core = g.inside & (g.kind != vp.PERIMETER)
    assert (g.kind[core] == vp.INFILL).all()
    g = _gen(PrintSpec("A", walls=2, top_layers=0, bottom_layers=0, infill=0.0))
    assert not (g.kind == vp.INFILL).any()
    # a rib 3 mm with 3 perimeters: inside 0.48 mm < 2w → full infill; switched off → sparse infill
    B = np.array([3.0, 20.0, 2.0])
    sp = PrintSpec("A", walls=3, top_layers=0, bottom_layers=0, infill=0.15)
    g = _gen(sp, B)
    core = g.inside & (g.kind != vp.PERIMETER)
    assert core.any() and (g.kind[core] == vp.THIN).all()
    sp.solid_thin = False
    assert not (_gen(sp, B).kind == vp.THIN).any()
    # a wide inside: the corners are not a thin place
    g = _gen(PrintSpec("A", walls=2, top_layers=0, bottom_layers=0, infill=0.15), np.array([10.0, 8.0, 1.0]))
    assert not (g.kind == vp.THIN).any()


def test_unsupported_pattern_warns():
    g = _gen(PrintSpec("A", walls=2, top_layers=0, bottom_layers=0, infill=0.2, pattern="honeycomb"),
             np.array([10.0, 10.0, 1.0]))
    assert g.warnings and "grid" in g.warnings[0]


# ===================================================================== model and boundary conditions
def test_build_model_hex_orientation():
    g = _gen(PrintSpec("A", walls=2, top_layers=1, bottom_layers=1, infill=0.2), np.array([3.0, 3.0, 1.0]))
    m = V.build_model(g)
    X = m["nodes"][m["hexes"]]
    # positive volume: (x1 − x0) × (x3 − x0) · (x4 − x0) > 0 at corner 0 (ccx order)
    vol = np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 3] - X[:, 0]), X[:, 4] - X[:, 0])
    assert (vol > 0).all()
    assert np.allclose(vol, np.prod(g.h))
    assert len(m["hexes"]) == int((g.kind > 0).sum())


def test_locate_points_and_interpolation():
    mesh = box_mesh("A", (0, 0, 0), (12, 10, 6), 2.0)
    loc = vp.tet_locator(mesh.nodes, mesh.tets)
    pts = np.array([[1, 1, 1], [11.9, 9.9, 5.9], [6, 5, 3], [12.5, 5, 3], [-0.1, 5, 3]])
    _, _, err = vp.locate_points(loc, pts)
    assert (err[:3] <= 1e-9).all() and (err[3:] > 1e-9).all()
    res = _results(mesh)
    U = V.interpolate_U(res, np.arange(len(mesh.tets)), loc, pts[:3], 0)
    assert np.allclose(U, np.column_stack([1e-3 * pts[:3, 0], -3e-4 * pts[:3, 1], -3e-4 * pts[:3, 2]]), atol=1e-14)


def test_setup_auto_center_and_bc_nodes():
    mesh = box_mesh("A", (0, 0, 0), (12, 10, 6), 1.0)
    res = _results(mesh)
    st = _study(walls=2, top_layers=3, bottom_layers=3, infill=0.2)
    spec = V.VoxelSpec(size=[3, 3, 2], margin=0.5)
    s = V._setup(st, res, spec)
    assert s.part == "A" and np.allclose(s.center, [6, 5, 3]) and s.auto["safety"] == pytest.approx(5.0)
    m = V.build_model(s.grid)
    bc, info = V._bc_nodes(st, res, s, m)
    nij = m["node_ijk"]
    n = np.array(s.grid.shape)
    on_face = np.any((nij == 0) | (nij == n), axis=1)
    assert set(bc.tolist()) == set(np.nonzero(on_face)[0].tolist())      # region inside the part: all 6 faces are cut
    assert s.grid.free_faces == []
    est = V._estimate_of(s, m, spec, 1, 0.0)
    assert est["dof"] == 3 * len(m["nodes"]) + 9 * len(m["hexes"]) and est["ok"]
    # region across the edge of the part: the face on the envelope of the part is a free surface, not a cut
    s2 = V._setup(st, res, V.VoxelSpec(center=[11.0, 5, 3], size=[3, 3, 2], margin=0.5))
    assert "x+" in s2.grid.free_faces
    m2 = V.build_model(s2.grid)
    bc2, _ = V._bc_nodes(st, res, s2, m2)
    assert not np.any(m2["node_ijk"][bc2, 0] == s2.grid.shape[0])
    # strop rovnic
    est = V._estimate_of(s, m, V.VoxelSpec(max_dof=10), 1, 0.0)
    assert not est["ok"] and "limit" in est["message"]


def test_setup_errors():
    mesh = box_mesh("A", (0, 0, 0), (6, 6, 6), 2.0)
    res = _results(mesh)
    st = _study(enabled=False)
    with pytest.raises(V.VoxelError):
        V._setup(st, res, V.VoxelSpec(center=[3, 3, 3]))
    st = _study()
    with pytest.raises(V.VoxelError):
        V._setup(st, res, V.VoxelSpec(center=[3, 3, 3], resolution="w/3"))
    res.steps[0].fields.clear()
    with pytest.raises(V.VoxelError):
        V._setup(st, res, V.VoxelSpec())


def test_drop_floating_islands():
    n = (6, 6, 4)
    kind = np.zeros(n, np.int8)
    kind[:, :, :2] = vp.PERIMETER          # plate connected to the boundary
    kind[2:4, 2:4, 3] = vp.INFILL          # island above the plate without a connecting face
    g = vp.VoxelGrid(np.eye(3), np.zeros(3), np.array([0.21, 0.21, 0.2]), kind, np.zeros(n), np.full(n, -1, np.int8),
                     kind > 0)
    m = V.build_model(g)
    bc = np.nonzero(m["node_ijk"][:, 0] == 0)[0]
    m2, dropped = V._drop_floating(g, m, bc)
    assert dropped == 4 and len(m2["hexes"]) == 6 * 6 * 2


# ===================================================================== stress, .inp, storage, image
def test_to_local_uniaxial_along_line():
    from fea_core.printing.infill import rotation
    R = rotation([0, 0, 1])
    axes = V.bin_axes(R, 36)
    for k in (0, 6, 9, 27):
        A = axes[k]
        e1 = A[0]
        sg = 7.0 * np.outer(e1, e1)
        S = np.array([sg[0, 0], sg[1, 1], sg[2, 2], sg[0, 1], sg[1, 2], sg[0, 2]])
        assert np.allclose(V.to_local(S, A), [7, 0, 0, 0, 0, 0], atol=1e-12)
    assert V._angle_bins(np.radians([0, 2.4, 2.6, 45, 179]), 36).tolist() == [0, 0, 1, 9, 0]


def test_write_inp(tmp_path):
    g = _gen(PrintSpec("A", walls=1, top_layers=0, bottom_layers=0, infill=1.0), np.array([1.0, 1.0, 0.4]))
    m = V.build_model(g)
    bins = V._angle_bins(g.angle[tuple(m["ijk"].T)], 36)
    axes = V.bin_axes(np.eye(3), 36)
    C = np.eye(6) * 1000
    bc = np.arange(4)
    V.write_inp(tmp_path / "v.inp", m["nodes"], m["hexes"], bins, axes, C, bc, [np.zeros((4, 3)), np.ones((4, 3))],
                step_names=["a", "b"])
    t = (tmp_path / "v.inp").read_text()
    assert t.count("*STEP") == 2 and "*BOUNDARY, OP=NEW" in t and "TYPE=C3D8I" in t
    assert t.count("*SOLID SECTION") == len(np.unique(bins)) == t.count("*ORIENTATION")
    assert "*EL PRINT, ELSET=EALL, GLOBAL=YES" in t
    assert t.count("\n1, 1, 1, 1\n") == 1                # node 1, DOF 1, value 1 in the 2nd step


def _fake_vr():
    g = _gen(PrintSpec("A", walls=2, top_layers=1, bottom_layers=1, infill=0.3), np.array([3.0, 3.0, 0.8]))
    m = V.build_model(g)
    E, N = len(m["hexes"]), len(m["nodes"])
    rng = np.random.default_rng(1)
    ijk = m["ijk"]
    st = V.VoxelStep("Tah", rng.normal(size=(N, 3)) * 1e-3, rng.normal(size=(N, 6)), rng.normal(size=(E, 6)),
                     rng.normal(size=(E, 6)), 1 + rng.random(E) * 5, 1 + rng.random(N) * 5)
    return V.VoxelResults(m["nodes"], m["hexes"], ijk, g.kind[tuple(ijk.T)], g.angle[tuple(ijk.T)],
                          g.wall[tuple(ijk.T)], np.zeros(E, np.int64), V.bin_axes(np.eye(3), 36), np.ones(E, bool),
                          np.eye(3), g.origin, g.h, g.shape, np.arange(10), [st], {"criterion": "tsai_wu"}, ["x"])


def test_save_load_roundtrip(tmp_path):
    vr = _fake_vr()
    vr.save(tmp_path / "d")
    b = V.VoxelResults.load(tmp_path / "d")
    assert b.shape == vr.shape and np.array_equal(b.hexes, vr.hexes) and b.warnings == ["x"]
    assert np.array_equal(b.step("Tah").safety_el, vr.steps[0].safety_el)


@pytest.mark.parametrize("field,z", [("kind", 1), ("safety", None), ("angle", 2), ("s33", 0)])
def test_render_png(tmp_path, field, z):
    p = V.render_png(_fake_vr(), tmp_path / f"{field}.png", field, slice_z=z)
    assert p.exists() and p.stat().st_size > 5000


def test_spec_json():
    s = V.VoxelSpec(center=[1, 2, 3], resolution="w/4", step="Tah")
    assert V.VoxelSpec.from_dict({**s.to_dict(), "novy_klic": 1}) == s


# ===================================================================== slow (ccx)
@pytest.mark.slow
def test_bending_c3d8i_exact(tmp_path):
    from verification.voxel_cases import x2
    checks, _ = x2(tmp_path)
    assert all(c.ok for c in checks), [(c.name, c.value, c.ref) for c in checks if not c.ok]


@pytest.mark.slow
def test_homogeneous_and_unidirectional(tmp_path):
    from verification.voxel_cases import x1
    checks, _ = x1(tmp_path)
    assert all(c.ok for c in checks), [(c.name, c.value, c.ref) for c in checks if not c.ok]
