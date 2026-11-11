"""The accelerated paths of the core give the same as the original (bit-for-bit): reading of the .frd and .dat, writing of the .inp, loads, caches of the
mesh, probes, saving of the results and of the project. Fast (without Gmsh and ccx); the real files from verification/out are used
if available (otherwise those parts are skipped).
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from fea_core.ccx import frd as F
from fea_core.mesh import Mesh, TET_FACES

ROOT = Path(__file__).resolve().parents[1]
VOUT = ROOT / "verification" / "out"
KUHN = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])


def same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def box_mesh(n=(4, 3, 2), seed=0) -> Mesh:
    """A block 40×30×20 from cubes of 6 tetrahedra each, tet10 (order of ccx), two parts (x < 20 / x ≥ 20), noisy
    coordinates (numbers with full precision). Selections x0, x1, z0 (faces of the surface).
    """
    nx, ny, nz = n
    g = np.stack(np.meshgrid(np.linspace(0, 40, nx + 1), np.linspace(0, 30, ny + 1), np.linspace(0, 20, nz + 1),
                             indexing="ij"), axis=-1).reshape(-1, 3)
    nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    tets, part = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                tets += [[v[a] for a in t] for t in KUHN]
                part += [int(i >= nx // 2)] * 6
    tets = np.array(tets)
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    keys = np.sort(np.stack([tets[:, [a, b]] for a, b in edges], axis=1), axis=2).reshape(-1, 2)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    nodes = np.vstack([g, 0.5 * (g[uniq[:, 0]] + g[uniq[:, 1]])])
    nodes = nodes + np.random.default_rng(seed).uniform(-0.3, 0.3, nodes.shape)
    t10 = np.hstack([tets, len(g) + inv.reshape(-1, 6)])
    m = Mesh(nodes, t10, np.array(part, np.int32), ["A", "B"])
    tri, _ = m.boundary_faces()
    c = m.nodes[tri[:, :3]].mean(axis=1)
    m.surfaces = {"x0": tri[c[:, 0] < 1], "x1": tri[c[:, 0] > 39], "z0": tri[c[:, 2] < 1]}
    return m


def study_for(m: Mesh, loads):
    from fea_core.study import Study, Material, Fixed, LoadCase, Selection
    st = Study("t", "x.step")
    st.add_material(Material("Ocel", 210000.0, 0.3, 7.85e-9), default=True)
    for k in m.surfaces:
        st.selections[k] = Selection(k, [])
    st.constraints.append(Fixed("x0"))
    st.load_cases.append(LoadCase("L", loads))
    return st


# ===================================================================== .frd
def _old_fixed_block(block, stride, nval, ids_cache):
    """The original conversion of the fixed columns (bytes → float64 in numpy)."""
    num_lines = len(block) // stride
    raw = np.frombuffer(block, dtype="S1").reshape(num_lines, stride)
    ids = raw[:, 3:13].copy().view("S10").ravel().astype(np.int64)
    vals = np.empty((num_lines, nval))
    for k in range(nval):
        vals[:, k] = raw[:, 13 + 12 * k: 25 + 12 * k].copy().view("S12").ravel().astype(np.float64)
    return ids, vals


def _read_frd_old(path, monkeypatch):
    with monkeypatch.context() as mp:
        mp.setattr(F, "_fixed_block", _old_fixed_block)
        mp.setattr(F, "SKIP_BLOCKS", set())
        return F.read_frd(path)


def _assert_frd_same(new, old):
    assert same(new.node_ids, old.node_ids) and same(new.coords, old.coords)
    ob = [b for b in old.blocks if b.name not in F.SKIP_BLOCKS]
    assert len(new.blocks) == len(ob)
    for a, b in zip(new.blocks, ob):
        assert (a.name, a.step, a.increment, a.time, a.components) == (b.name, b.step, b.increment, b.time,
                                                                         b.components)
        assert same(a.node_ids, b.node_ids) and same(a.values, b.values), a.name


def test_parse_e_fields_bitwise():
    rng = np.random.default_rng(5)
    v = rng.standard_normal(20000) * 10.0 ** rng.integers(-40, 40, 20000)
    v[:50] = 0.0
    v[50:60] = -0.0
    for nd in (5, 6):
        txt = "".join(f"{x:{nd + 7}.{nd}E}" for x in v) + "".join(
            s.rjust(nd + 7) for s in ("NaN", "inf", "-inf", "1.5E+02", "7E-3", "-2.5"))
        if nd == 5:
            txt += "1.00000E+100" + " 1.00000-100"[:0]           # 3-digit exponent (without a space)
        raw = np.frombuffer(txt.encode(), np.uint8).reshape(-1, nd + 7)
        ref = raw.copy().view(f"S{nd + 7}").ravel().astype(np.float64)
        got = F.parse_e_fields(raw, nd)
        assert same(got, ref)
    with pytest.raises(ValueError):
        F.parse_e_fields(np.frombuffer(b" 1.2345 E+02", np.uint8).reshape(1, 12), 5)


def _synthetic_frd(path: Path, n=300, seed=1):
    """A small .frd in the format of ccx (stuck negative numbers, an ERROR block, several steps)."""
    rng = np.random.default_rng(seed)
    out = ["    1C\n", f"    2C{n:30d}{1:37d}\n"]
    xyz = rng.standard_normal((n, 3)) * 100
    out += [f" -1{i + 1:10d}" + "".join(f"{x:12.5E}" for x in r) + "\n" for i, r in enumerate(xyz)]
    out.append(" -3\n")
    blocks = [("DISP", ["D1", "D2", "D3"], 3), ("STRESS", ["SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX"], 6),
              ("ERROR", ["STR(%)"], 1)]
    for step in (1, 2):
        for name, comps, nv in blocks:
            out.append(f"    1PSTEP{1:26d}{1:12d}{step:12d}\n")
            out.append(f"  100CL  101 1.000000000{n:12d}                     0    1           1\n")
            out.append(f" -4  {name:8s}{nv:4d}    1\n")
            out += [f" -5  {c:8s}    1    2    1    0\n" for c in comps]
            vals = rng.standard_normal((n, nv)) * 10.0 ** rng.integers(-30, 6, (n, nv))
            out += [f" -1{i + 1:10d}" + "".join(f"{x:12.5E}" for x in r) + "\n" for i, r in enumerate(vals)]
            out.append(" -3\n")
    out.append(" 9999\n")
    path.write_bytes("".join(out).encode())


def test_read_frd_synthetic(tmp_path, monkeypatch):
    p = tmp_path / "t.frd"
    _synthetic_frd(p)
    new = F.read_frd(p)
    old = _read_frd_old(p, monkeypatch)
    # the ERROR block (ccx stress error estimate) has been read since 14. 10. 2026 (the field stress_error)
    assert [b.name for b in old.blocks].count("ERROR") == 2 and [b.name for b in new.blocks].count("ERROR") == 2
    _assert_frd_same(new, old)
    assert new.steps() == [1, 2]


@pytest.mark.parametrize("rel", ["V1/run/job.frd", "S2/run/job.frd", "T2/run/thermal/case1.frd",
                                 "P1/run/job.frd"])
def test_read_frd_real(rel, monkeypatch):
    p = VOUT / rel
    if not p.exists():
        pytest.skip(f"chybí {p} (python -m verification.run_all)")
    _assert_frd_same(F.read_frd(p), _read_frd_old(p, monkeypatch))


# ===================================================================== .dat
def _synthetic_dat(path: Path, crlf: bool, n_el=400, seed=2, extra=""):
    rng = np.random.default_rng(seed)
    nl = "\r\n" if crlf else "\n"
    s = [nl + "                        S T E P       1" + nl * 3]
    for k in range(3):
        s.append(f" total force (fx,fy,fz) for set NSEL1 and time  0.{k}000000E+01{nl}{nl}"
                 f"       -1.359393E-10 -1.093814E-10 -2.000000E+04{nl}{nl}")
        s.append(f" displacements (vx,vy,vz) for set NREF and time  0.{k}000000E+01{nl}{nl}")
        s.append("".join(f"{65000 + j:10d}" + "".join(f" {x:13.6E}" for x in rng.standard_normal(3)) + nl
                         for j in range(2)) + nl)
        s.append(f" stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EJOINT and time  0.{k}E+01{nl}{nl}")
        order = rng.permutation(n_el) * 3 + 1
        for e in order:
            for ip in range(1, 5):
                s.append(f"{e:10d}{ip:4d}" + "".join(f" {x:13.6E}" for x in rng.standard_normal(6) * 100) + nl)
        s.append(nl)
    s.append(extra)
    path.write_bytes("".join(s).encode("latin-1"))


def _same_blocks(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert list(x) == list(y)
        for k in x:
            assert same(x[k], y[k]), k


@pytest.mark.parametrize("crlf", [False, True])
def test_dat_readers_synthetic(tmp_path, crlf):
    from fea_core.joints.forces import read_el_print, _read_el_print_lines
    from fea_core.solve import _read_printed, _read_printed_lines
    p = tmp_path / "job.dat"
    _synthetic_dat(p, crlf)
    _same_blocks(read_el_print(p, "EJOINT"), _read_el_print_lines(p, "EJOINT"))
    _same_blocks(_read_printed(p, "NREF"), _read_printed_lines(p, "NREF"))
    assert len(read_el_print(p, "ejoint")[0]) == 400               # name of the set in capitals


def test_dat_readers_irregular(tmp_path):
    """Unusual files: elements scattered (also repeatedly), a header without data, isolated \\\\r → the same as before."""
    from fea_core.joints.forces import read_el_print, _read_el_print_lines
    from fea_core.solve import _read_printed, _read_printed_lines
    head = " stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EJOINT and time  0.1E+01\n\n"
    row = lambda e, ip, v: f"{e:10d}{ip:4d}" + "".join(f" {v + i:13.6E}" for i in range(6)) + "\n"  # noqa: E731
    cases = {
        "rozhazene": head + row(5, 1, 1) + row(7, 1, 2) + row(5, 2, 3) + row(7, 2, 4) + "\n",
        "ruzne_ip": head + row(5, 1, 1) + row(5, 2, 2) + row(9, 1, 3) + "\n",
        "bez_dat": head + head + row(1, 1, 1) + "\n",
        "bez_mezery": head + row(1, 1, 1) + head + row(2, 1, 2) + "\n",
        "lone_cr": head + row(1, 1, 1) + "\r" + row(2, 1, 2) + "\n",
        "jiny_radek": head + row(1, 1, 1) + "    12   1  1.0E+00 2.0E+00 3.0E+00 4.0E+00 5.0E+00 6.0E+00\n\n",
        "konec_bez_nl": head + row(1, 1, 1) + row(2, 1, 2).rstrip("\n"),
    }
    for name, txt in cases.items():
        p = tmp_path / f"{name}.dat"
        p.write_bytes(txt.encode("latin-1"))
        _same_blocks(read_el_print(p, "EJOINT"), _read_el_print_lines(p, "EJOINT"))
    nref = (" displacements (vx,vy,vz) for set NREF and time  0.1E+01\n\n     17736  1.0E+00  2.0E+00  3.0E+00\n"
            " total displacements (vx,vy,vz) for set NREF and time  0.1E+01\n"
            " displacements (vx,vy,vz) for set NREF and time  0.2E+01\n     17736  4.0E+00\x0c  5.0E+00  6.0E+00\n")
    p = tmp_path / "nref.dat"
    p.write_bytes(nref.encode("latin-1"))
    _same_blocks(_read_printed(p, "NREF"), _read_printed_lines(p, "NREF"))


@pytest.mark.parametrize("case", ["S2", "W1", "S3", "V7"])
def test_dat_readers_real(case):
    from fea_core.joints.forces import read_el_print, _read_el_print_lines
    from fea_core.solve import _read_printed, _read_printed_lines
    p = VOUT / case / "run" / "job.dat"
    if not p.exists():
        pytest.skip(f"missing {p}")
    _same_blocks(read_el_print(p, "EJOINT"), _read_el_print_lines(p, "EJOINT"))
    _same_blocks(_read_printed(p, "NREF"), _read_printed_lines(p, "NREF"))


# ===================================================================== .inp
def _fmt(v):
    return f"{v:.12g}"


def _old_write_ids(f, ids, per_line=16):
    ids = np.asarray(ids, dtype=np.int64).ravel()
    for i in range(0, len(ids), per_line):
        f.write(", ".join(str(int(x)) for x in ids[i:i + per_line]) + ",\n")


def _old_write_mesh(f, mesh):
    f.write("*NODE, NSET=NALL\n")
    for i, (x, y, z) in enumerate(mesh.nodes):
        f.write(f"{i + 1}, {_fmt(x)}, {_fmt(y)}, {_fmt(z)}\n")
    names = []
    for pi, pname in enumerate(mesh.part_names):
        names.append(f"P{pi + 1}")
        f.write(f"** part {pi + 1}: {pname.encode('ascii', 'replace').decode()}\n")
        f.write(f"*ELEMENT, TYPE={mesh.element_type}, ELSET=P{pi + 1}\n")
        for ei in np.nonzero(mesh.tet_part == pi)[0]:
            f.write(f"{ei + 1}, " + ", ".join(str(n + 1) for n in mesh.tets[ei]) + "\n")
    f.write("*ELSET, ELSET=EALL\n" + ", ".join(names) + "\n")


def test_write_mesh_sets_bytes():
    from fea_core.ccx.inp import InpMap, write_mesh, write_sets, _write_ids, _write_table
    m = box_mesh()
    m.nodes[0] = [1e-17, -0.0, 123456789.123456789]               # extreme shapes of numbers
    m.nodes[1] = [1e21, -3.0, 2.5e-300]
    a, b = io.StringIO(), io.StringIO()
    write_mesh(a, m, InpMap())
    _old_write_mesh(b, m)
    assert a.getvalue() == b.getvalue()
    for ids in ([], [7], list(range(1, 17)), list(range(5, 60)), np.arange(40000) * 3 + 1):
        a, b = io.StringIO(), io.StringIO()
        _write_ids(a, ids)
        _old_write_ids(b, ids)
        assert a.getvalue() == b.getvalue()
    a, b = io.StringIO(), io.StringIO()
    sel = {"x0": "SEL1", "x1": "SEL2", "z0": "SEL3"}
    write_sets(a, m, InpMap(), {"x0", "z0"}, {"x1", "z0"}, sel)
    for name in ("x0", "z0"):
        b.write(f"** selection: {name}\n*NSET, NSET=N{sel[name]}\n")
        _old_write_ids(b, m.surface_nodes(name) + 1)
    for name in ("x1", "z0"):
        e, k = m.element_faces(name)
        b.write(f"*SURFACE, NAME=S{sel[name]}, TYPE=ELEMENT\n" + "".join(f"{x + 1}, S{y}\n" for x, y in zip(e, k)))
    assert a.getvalue() == b.getvalue()
    a = io.StringIO()
    vals = np.random.default_rng(3).standard_normal(50000) * 1e3
    _write_table(a, "%d, %.12g\n", [list(range(1, 50001)), vals.tolist()], 50000)
    assert a.getvalue() == "".join(f"{i + 1}, {_fmt(v)}\n" for i, v in enumerate(vals))


def test_write_inp_cload_same_as_dict(tmp_path):
    """Overlapping forces in one state: the dense summation = the original dictionary (the order of the loads, 0.0 + v)."""
    from fea_core.ccx.inp import write_inp, consistent_nodal_forces
    from fea_core.mesh import tri_areas
    from fea_core.study import Force
    m = box_mesh()
    m.surfaces["x1b"] = m.surfaces["x1"][: len(m.surfaces["x1"]) // 2]
    loads = [Force("x1", [10.0, -3.0, 0.0]), Force("x1b", [-1.0, 0.0, 7.5]), Force("z0", [0.0, 0.0, 0.1])]
    st = study_for(m, loads)
    write_inp(st, m, tmp_path / "a.inp")
    txt = (tmp_path / "a.inp").read_text()
    got = txt.split("*CLOAD\n", 1)[1].split("*NODE FILE", 1)[0]
    cload = {}
    for l in loads:
        tri = m.surfaces[l.selection]
        t = np.asarray(l.vector, float) / float(tri_areas(m.nodes, tri).sum())
        for n, fv in consistent_nodal_forces(m.nodes, tri, t).items():
            for d in range(3):
                if fv[d] != 0.0:
                    cload[(n + 1, d + 1)] = cload.get((n + 1, d + 1), 0.0) + fv[d]
    assert got == "".join(f"{n}, {d}, {_fmt(v)}\n" for (n, d), v in sorted(cload.items()))
    assert "TOTALS" not in txt


# ===================================================================== loads
def test_scatter_add_matches_add_at():
    from fea_core.loads import scatter_add
    rng = np.random.default_rng(4)
    F0 = rng.standard_normal((50, 3))
    idx = rng.integers(0, 50, (300, 6))
    vals = rng.standard_normal((300, 6, 3))
    ref = F0.copy()
    for i in range(6):
        np.add.at(ref, idx[:, i], vals[:, i])
    got = F0.copy()
    scatter_add(got, idx, vals)
    assert same(got, ref)


def _old_gravity(mesh, rho, g):
    from fea_core.loads import _TET_PTS, _TET_W, _tet10_dN
    from fea_core.results import tet10_shape
    F = np.zeros((len(mesh.nodes), 3))
    X = mesh.nodes[mesh.tets]
    fe = np.zeros((len(X), 10))
    vol = np.zeros(len(X))
    for p, w in zip(_TET_PTS, _TET_W):
        J = np.tensordot(_tet10_dN(*p), X, axes=(1, 1)).swapaxes(0, 1)
        det = (J[:, 0, 0] * (J[:, 1, 1] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 1])
               - J[:, 0, 1] * (J[:, 1, 0] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 0])
               + J[:, 0, 2] * (J[:, 1, 0] * J[:, 2, 1] - J[:, 1, 1] * J[:, 2, 0]))
        vol += w * det
        fe += (w * det * rho)[:, None] * tet10_shape(np.asarray(p))[None, :]
    for i in range(10):
        np.add.at(F, mesh.tets[:, i], fe[:, i, None] * g[None, :])
    return F, vol


def test_gravity_volumes_and_cache():
    from fea_core.loads import gravity_nodal_forces, element_volumes
    from fea_core.study import Gravity
    m = box_mesh()
    st = study_for(m, [Gravity()])
    g = np.array([0.0, -9810.0, 3.0])
    rho = np.full(len(m.tets), 7.85e-9)
    F_ref, vol_ref = _old_gravity(m, rho, g)
    assert same(gravity_nodal_forces(st, m, g), F_ref)
    assert same(element_volumes(m), vol_ref)
    v = element_volumes(m)
    v[:] = 0.0                                                     # a copy, the cache does not change
    assert same(element_volumes(m), vol_ref)
    m.nodes[:, 0] *= 2.0                                           # a change IN PLACE → recalculation
    assert np.allclose(element_volumes(m), 2 * vol_ref, rtol=1e-12)
    assert same(gravity_nodal_forces(st, m, g), _old_gravity(m, rho, g)[0])


def test_consistent_nodal_forces_matches_add_at():
    from fea_core.ccx.inp import consistent_nodal_forces
    from fea_core.mesh import tri_areas
    m = box_mesh()
    tri = m.surfaces["z0"]
    t = np.array([0.3, -1.1, 2.0])
    got = consistent_nodal_forces(m.nodes, tri, t)
    w_nodes = tri[:, 3:6]
    un, inv = np.unique(w_nodes, return_inverse=True)
    inv = inv.reshape(w_nodes.shape)
    acc = np.zeros((len(un), 3))
    for k in range(3):
        np.add.at(acc, inv[:, k], (tri_areas(m.nodes, tri) * (1.0 / 3.0))[:, None] * t[None, :])
    assert list(got) == un.tolist() and same(np.array(list(got.values())), acc)


# ===================================================================== mesh cache
def test_mesh_cache_invalidation_in_place():
    """tets overwritten IN PLACE (joints.fem.open_weld_roots, tests) → element_faces / boundary_faces again."""
    from fea_core.mesh import surface_element_faces

    def fresh(name):
        return surface_element_faces(m.tets, m.surfaces[name])
    m = box_mesh()
    before = {k: m.element_faces(k) for k in m.surfaces}         # fills the cache
    assert all(same(a, b) for k in m.surfaces for a, b in zip(before[k], fresh(k)))
    tri0, _ = m.boundary_faces()
    tri0[:] = -1                                                   # returns copies
    assert (m.boundary_faces()[0] >= 0).all()
    # rotate the corners of all elements in place (nodes unchanged): the same faces, different face numbers S1..S4
    m.tets[:, :4] = m.tets[:, [1, 2, 0, 3]]
    after = {k: m.element_faces(k) for k in m.surfaces}
    assert all(same(a, b) for k in m.surfaces for a, b in zip(after[k], fresh(k)))
    assert any(not same(before[k][1], after[k][1]) for k in m.surfaces)
    ref = Mesh(m.nodes, m.tets.copy(), m.tet_part, m.part_names)._boundary_faces()
    assert all(same(a, b) for a, b in zip(m.boundary_faces(), ref))
    # disconnected contact as open_weld_roots: new nodes (new field) + the tets of part B in place + node_alias
    m = box_mesh()
    n_b0 = len(m.boundary_faces()[0])
    shared = np.intersect1d(np.unique(m.tets[m.tet_part == 0]), np.unique(m.tets[m.tet_part == 1]))
    n0 = len(m.nodes)
    remap = np.arange(n0)
    remap[shared] = n0 + np.arange(len(shared))
    m.nodes = np.vstack([m.nodes, m.nodes[shared]])
    m.tets[m.tet_part == 1] = remap[m.tets[m.tet_part == 1]]
    assert len(m.boundary_faces()[0]) > n_b0                       # the contact is now a surface (without node_alias)
    m.node_alias = np.concatenate([np.arange(n0), shared])
    assert len(m.boundary_faces()[0]) == n_b0
    assert all(same(a, b) for a, b in zip(m.element_faces("x0"), fresh("x0")))   # x0 lies entirely on part A


def test_part_surface_packed_keys():
    from fea_core.printing.region import part_surface
    m = box_mesh()
    for p in (0, 1):
        el = np.nonzero(m.tet_part == p)[0]
        tri, nrm = part_surface(m, el)
        corners = m.tets[el, :4]
        flat = corners[:, TET_FACES].reshape(-1, 3)
        keys = np.sort(m.canonical()[flat], axis=1)
        _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        assert same(tri, flat[np.nonzero(cnt[inv.ravel()] == 1)[0]])


# ===================================================================== results, probes, saving
def _results(m):
    from fea_core.results import Results, StepResult
    rng = np.random.default_rng(6)
    n = len(m.nodes)
    st = StepResult("L", rng.standard_normal((n, 3)), rng.standard_normal((n, 6)), {"x0": np.array([1.0, 2, 3])},
                    {n + 1: np.array([0.1, 0.2, 0.3])})
    st.fields["print_safety"] = rng.random(n)
    return Results(m, [st], ["varování"], {"backend": "ccx"})


def _old_locate(res, p, cand):
    best, best_xi, best_err = -1, None, np.inf
    for e in np.atleast_1d(cand):
        pts = res.mesh.nodes[res.mesh.tets[e, :4]]
        a = pts[0]
        xi = np.linalg.solve((pts[1:] - a).T, p - a)
        err = max(-xi.min(), xi.sum() - 1.0, 0.0)
        if err < best_err:
            best, best_xi, best_err = int(e), xi, err
        if err <= 1e-9:
            break
    return best, best_xi


def test_locate_batch_and_cache():
    m = box_mesh()
    res = _results(m)
    pts = np.random.default_rng(7).uniform([-5, -5, -5], [45, 35, 25], size=(60, 3))   # a part outside the mesh
    for p in pts:
        e, xi = res.locate(p)
        _, cand = res._tree.query(p, k=min(64, len(m.tets)))
        e0, xi0 = _old_locate(res, p, cand)
        assert e == e0 and same(xi, xi0)
        e2, xi2 = res.locate(p.copy())                            # the same point → from memory, a copy
        xi2[:] = 9.0
        assert res.locate(p)[0] == e and same(res.locate(p)[1], xi0)
    u = res.probe(pts[3], "U")
    assert same(u, res.probe(pts[3], "U"))


def test_results_save_load_roundtrip(tmp_path):
    from fea_core.results import Results
    m = box_mesh()
    m.node_alias = np.arange(len(m.nodes))
    m.cell_data["print_shell"] = np.linspace(-1, 1, len(m.tets))
    res = _results(m)
    res.save(tmp_path / "results.npz")
    with zipfile.ZipFile(tmp_path / "results.npz") as z:          # without compression
        assert {i.compress_type for i in z.infolist()} == {zipfile.ZIP_STORED}
    with np.load(tmp_path / "results_mesh.npz") as z:
        assert z["tets"].dtype == np.int32                         # na disku int32
    r2 = Results.load(tmp_path / "results.npz")
    assert r2.mesh.tets.dtype == m.tets.dtype and same(r2.mesh.tets, m.tets) and same(r2.mesh.nodes, m.nodes)
    assert same(r2.mesh.tet_part, m.tet_part) and same(r2.mesh.node_alias, m.node_alias)
    assert all(same(r2.mesh.surfaces[k], m.surfaces[k]) for k in m.surfaces)
    assert same(r2.mesh.cell_data["print_shell"], m.cell_data["print_shell"])
    s0, s1 = res.steps[0], r2.steps[0]
    assert same(s1.U, s0.U) and same(s1.S, s0.S) and same(s1.fields["print_safety"], s0.fields["print_safety"])
    assert list(s1.ref_displacements) == list(s0.ref_displacements)
    assert all(same(s1.ref_displacements[k], s0.ref_displacements[k]) for k in s0.ref_displacements)
    assert r2.info.get("backend") == "ccx"


def test_load_old_compressed_format(tmp_path):
    """Files saved earlier (savez_compressed, int64 without a record of dtypes, without ref_displacements) are loaded."""
    import json
    from fea_core.results import Results
    m = box_mesh()
    meta_m = {"part_names": m.part_names, "surfaces": list(m.surfaces), "report": {}, "cell_data": []}
    arrays = {"nodes": m.nodes, "tets": m.tets, "tet_part": m.tet_part,
              **{f"surf_{i}": v for i, v in enumerate(m.surfaces.values())},
              "meta": np.frombuffer(json.dumps(meta_m).encode(), np.uint8)}
    np.savez_compressed(tmp_path / "results_mesh.npz", **arrays)
    n = len(m.nodes)
    meta = {"steps": [{"name": "L", "reactions": {}, "fields": []}], "warnings": []}
    np.savez_compressed(tmp_path / "results.npz", U_0=np.ones((n, 3)), S_0=np.ones((n, 6)),
                        meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
    r = Results.load(tmp_path / "results.npz")
    assert same(r.mesh.tets, m.tets) and r.steps[0].ref_displacements == {} and r.thermal == []


def test_project_storage_and_stale_results(tmp_path):
    from fea_core.project import save_project, load_project
    m = box_mesh()
    res = _results(m)
    st = study_for(m, [])
    step = tmp_path / "model.step"
    step.write_text("ISO-10303-21; " * 200)
    save_project(tmp_path / "a.fea", st, step, res)
    with zipfile.ZipFile(tmp_path / "a.fea") as z:
        kinds = {i.filename: i.compress_type for i in z.infolist()}
    assert kinds == {"model.step": zipfile.ZIP_DEFLATED, "study.json": zipfile.ZIP_DEFLATED,
                     "results.npz": zipfile.ZIP_STORED, "results_mesh.npz": zipfile.ZIP_STORED}
    wd = tmp_path / "open"
    _, _, r = load_project(tmp_path / "a.fea", wd)
    assert same(r.steps[0].S, res.steps[0].S)
    save_project(tmp_path / "b.fea", st, step, None)               # a project without results into the same folder
    _, _, r = load_project(tmp_path / "b.fea", wd)
    assert r is None and not (wd / "results.npz").exists()
