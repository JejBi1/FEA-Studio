"""Fast tests of the contacts (fea_core.contacts) without ccx: the model and the JSON, check of the settings, steps of the branches and
sequences, the scale of the state, reading of the CF from the .dat, an empty CONTACT block in the .frd, disconnection of a conforming joint.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from fea_core.study import Study, Material, Force, Pressure, LoadCase
from fea_core.contacts.model import ContactPair, AnalysisSettings, LoadStep, Motion, ContactSettings
from fea_core.contacts.check import check
from fea_core.contacts.run import step_plans, scaled_case


def _study():
    st = Study("t", "x.step")
    st.add_material(Material("ocel", 210000.0, 0.3, 7.85e-9), default=True)
    st.load_cases.append(LoadCase("F", [Force("a", [0, 0, 10.0]), Pressure("b", 2.0)]))
    return st


def test_contact_json_roundtrip():
    st = _study()
    st.contacts = [ContactPair("p", "A", "B", kind="frictional", mu=0.2, clearance=-0.01)]
    st.contact_settings = ContactSettings(method="mortar", nlgeom=True)
    st.analysis = AnalysisSettings(kind="nonlinear", branches=False, sequence=[
        LoadStep("s1", case="F", scale=0.5, prescribed=[Motion("r", "rot", 3, [[0, 0], [1, 0.1]])],
                 mu_change={"p": 0.3}, cycles=2)])
    d = json.loads(json.dumps(st.to_dict()))
    assert "contacts" in d["features"] and "analysis" in d["features"]
    s2 = Study.from_dict(d)
    assert s2.contacts[0].clearance == -0.01 and s2.contact_settings.method == "mortar"
    assert s2.analysis.sequence[0].prescribed[0].kind == "rot" and s2.analysis.sequence[0].cycles == 2
    assert s2.is_nonlinear()


def test_check_errors():
    st = _study()
    st.contacts = [ContactPair("p", "A", "B", kind="frictional", mu=0.0),
                   ContactPair("q", "A", "B", kind="interference", clearance=0.01),
                   ContactPair("r", "A", "A", kind="frictionless")]
    st.analysis = AnalysisSettings(kind="nonlinear", branches=False,
                                   sequence=[LoadStep("s", case="neni", mu_change={"x": 0.1},
                                                      prescribed=[Motion("nic")])])
    e, w = check(st, ["A", "B"])
    txt = "\n".join(e)
    for key in ("μ must be > 0", "negative radial", "self-contact", "'neni' does not exist",
                "non-existent pair 'x'", "rigid remote support"):
        assert key in txt, key


def test_step_plans_branches_and_sequence():
    st = _study()
    st.load_cases.append(LoadCase("G", []))
    plans = step_plans(st, preload=True)
    assert [p[0] for p in plans] == ["nl_b1", "nl_b2"]
    assert [s["kind"] for s in plans[0][1]] == ["preload", "case"] and plans[1][1][1]["case_index"] == 1
    st.analysis = AnalysisSettings(kind="nonlinear", branches=False,
                                   sequence=[LoadStep("a", case="F"), LoadStep("b", case="G", cycles=2)])
    (job, steps), = step_plans(st, preload=False)
    assert [s["name"] for s in steps] == ["a (cycle 1)", "b (cycle 1)", "b (cycle 2)"]


def test_scaled_case():
    st = _study()
    lc = scaled_case(st.load_cases[0], 0.5)
    assert lc.loads[0].vector == [0, 0, 5.0] and lc.loads[1].value == 1.0
    assert st.load_cases[0].loads[0].vector == [0, 0, 10.0]       # the original unchanged


def test_contact_print_parse(tmp_path):
    from fea_core.contacts.results import contact_print
    p = tmp_path / "x.dat"
    p.write_text("""
 statistics for slave set C1S, master set C1M and time  0.1000000E+01

   total surface force (fx,fy,fz) and moment about the origin (mx,my,mz)

    1.0E+00  2.0E+00 -3.0E+02  4.0E+00  5.0E+00  6.0E+00

   center of gravity and mean normal

    1.0E+01  2.0E+01  3.0E+01  0.0E+00  0.0E+00 -1.0E+00
""", encoding="latin-1")
    c = contact_print(p)
    assert len(c) == 1 and c[0]["slave"] == "C1S" and c[0]["time"] == 1.0
    assert np.allclose(c[0]["force"], [1, 2, -300]) and np.allclose(c[0]["normal"], [0, 0, -1])


def test_frd_empty_contact_block(tmp_path):
    from fea_core.ccx.frd import read_frd
    head = (" 100CL  101 1.000000000       2                     0    1           1\n"
            " -4  CONTACT     6    1\n"
            " -5  COPEN       1    1    1    0\n -5  CSLIP1      1    1    1    0\n -5  CSLIP2      1    1    1    0\n"
            " -5  CPRESS      1    1    1    0\n -5  CSHEAR1     1    1    1    0\n -5  CSHEAR2     1    1    1    0\n"
            " -3\n")
    disp = (" 100CL  102 1.000000000       2                     0    1           1\n"
            " -4  DISP        4    1\n -5  D1          1    2    1    0\n -5  D2          1    2    2    0\n"
            " -5  D3          1    2    3    0\n -5  ALL         1    2    0    0    1ALL\n"
            " -1         1 1.00000E+00 2.00000E+00 3.00000E+00\n"
            " -1         2 4.00000E+00 5.00000E+00 6.00000E+00\n -3\n")
    p = tmp_path / "x.frd"
    p.write_text("    1PSTEP                         1           1           1\n" + head + disp + "9999\n",
                 encoding="latin-1")
    f = read_frd(p)
    names = [b.name for b in f.blocks]
    assert names == ["CONTACT", "DISP"]
    assert len(f.blocks[0].node_ids) == 0
    assert np.allclose(f.get("DISP").values[1], [4, 5, 6])


def test_split_interface_two_tets():
    from fea_core.mesh import Mesh
    from fea_core.contacts.detect import split_interface
    # two tetrahedra with a common face (0, 1, 2), tet10 with the midpoints of the edges
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], float)
    edges = {}

    def mid(a, b):
        k = tuple(sorted((a, b)))
        if k not in edges:
            edges[k] = len(X) + len(edges)
        return edges[k]
    t1 = [0, 1, 2, 3]
    t2 = [0, 2, 1, 4]
    tets = []
    for t in (t1, t2):
        a, b, c, d = t
        tets.append([a, b, c, d, mid(a, b), mid(b, c), mid(c, a), mid(a, d), mid(b, d), mid(c, d)])
    mids = np.array([(X[a] + X[b]) / 2 for (a, b), _ in sorted(edges.items(), key=lambda kv: kv[1])])
    nodes = np.vstack([X, mids])
    mesh = Mesh(nodes, np.array(tets), np.array([0, 1]), ["A", "B"])
    n = split_interface(mesh, "A", "B")
    assert n == 6                                            # 3 corners + 3 midpoints of the common face
    assert len(mesh.nodes) == len(nodes) + 6
    assert not set(mesh.tets[0].tolist()) & set(mesh.tets[1].tolist())   # the parts do not share nodes any more
    a, b = mesh.surfaces["__split_A|B_a"], mesh.surfaces["__split_A|B_b"]
    # the same face geometry, opposite orientation (each side from its own element)
    assert np.allclose(np.sort(mesh.nodes[a].reshape(-1, 3), axis=0), np.sort(mesh.nodes[b].reshape(-1, 3), axis=0))
    assert set(mesh.canonical()[b].ravel().tolist()) == set(a.ravel().tolist())


# ------------------------------------------------------------------ A6: reliable measures of the contact pressure
def _tri6_square(n: int = 4, size: float = 4.0):
    """A square face of n × n cells (2 triangles tri6 per cell), a fine grid (2n+1)² nodes with the midpoints of the edges."""
    m = 2 * n + 1
    h = size / (2 * n)
    X = np.array([[i * h, j * h, 0.0] for j in range(m) for i in range(m)])
    idx = lambda i, j: j * m + i                                             # noqa: E731
    tris = []
    for cj in range(n):
        for ci in range(n):
            i, j = 2 * ci, 2 * cj
            for a, b, c in (((i, j), (i + 2, j), (i + 2, j + 2)), ((i, j), (i + 2, j + 2), (i, j + 2))):
                mid = lambda p, q: ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)  # noqa: E731
                tris.append([idx(*a), idx(*b), idx(*c), idx(*mid(a, b)), idx(*mid(b, c)), idx(*mid(c, a))])
    return X, np.array(tris, np.int64)


def _fake_mesh(X, held=()):
    from types import SimpleNamespace
    return SimpleNamespace(nodes=X, surface_nodes=lambda name: np.asarray(held, np.int64))


def test_free_edge_nodes_square():
    from fea_core.contacts.results import free_edge_nodes
    X, tri = _tri6_square(4)
    e = free_edge_nodes(tri)
    on_edge = (np.isclose(X[:, 0], 0) | np.isclose(X[:, 0], 4) | np.isclose(X[:, 1], 0) | np.isclose(X[:, 1], 4))
    assert set(e.tolist()) == set(np.nonzero(on_edge)[0].tolist())         # corners and midpoints of the edges of the perimeter, not the inside


def _metrics(p_all, fn=None, study=None, mesh=None):
    from fea_core.contacts.results import node_weights, pressure_metrics
    X, tri = _tri6_square(4)
    mesh = mesh or _fake_mesh(X)
    nodes, w = node_weights(mesh, tri)
    p = p_all[nodes]
    return pressure_metrics(study or _study(), mesh, tri, nodes, w, p, float((p * w).sum()) if fn is None else fn,
                            p_all), nodes


def test_pressure_metrics_uniform_and_peaks():
    X, tri = _tri6_square(4)
    p0 = np.full(len(X), 10.0)
    met, nodes = _metrics(p0)
    assert met["p_force_MPa"] == pytest.approx(10.0) and met["area_active_mm2"] == pytest.approx(16.0)
    assert not met["peak_at_edge"] and met["p_max_interior_MPa"] == pytest.approx(10.0)
    # a peak at the perimeter (midpoint of an edge) 3× above the inside → an edge peak
    edge_mid = int(next(n for n in nodes if np.isclose(X[n, 1], 0) and np.isclose(X[n, 0], 0.5)))
    p1 = p0.copy()
    p1[edge_mid] = 30.0
    met, _ = _metrics(p1)
    assert met["peak_at_edge"] and met["p_max_interior_MPa"] == pytest.approx(10.0) and met["peak_edge_ratio"] == pytest.approx(3.0)
    # the same peak in the middle of the face = not an edge peak
    center = int(next(n for n in nodes if np.allclose(X[n], [1.5, 1.5, 0.0])))
    p2 = p0.copy()
    p2[center] = 30.0
    met, _ = _metrics(p2)
    assert not met["peak_at_edge"] and met["p_max_interior_MPa"] == pytest.approx(30.0)


def test_pressure_metrics_symmetry_edge_is_not_free():
    """Nodes held by a constraint (a plane of symmetry) are not a free edge – a peak there is not conspicuous."""
    from fea_core.study import Displacement
    X, tri = _tri6_square(4)
    p = np.full(len(X), 10.0)
    on_x0 = np.nonzero(np.isclose(X[:, 0], 0))[0]
    spike = int(next(n for n in on_x0 if np.isclose(X[n, 1], 0.5)))
    p[spike] = 30.0
    st = _study()
    st.constraints.append(Displacement("sym", ux=0.0))
    met, _ = _metrics(p, study=st, mesh=_fake_mesh(X, held=on_x0))
    assert not met["peak_at_edge"]
    met_free, _ = _metrics(p)                                # without a constraint it is a free edge
    assert met_free["peak_at_edge"]


def test_pressure_force_uses_exact_force_and_threshold():
    """p_force = Fn / area of the nodes with p ≥ 5 % of the average: the noise at the contact boundary does not enlarge the area."""
    X, tri = _tri6_square(4)
    p = np.zeros(len(X))
    p[X[:, 0] <= 2.0 + 1e-9] = 10.0
    met0, _ = _metrics(p, fn=80.0)
    noisy = p.copy()
    noisy[np.isclose(X[:, 0], 3.0)] = 0.1                   # a barely visible noise (1 % of the pressure) beyond the contact boundary
    met, _ = _metrics(noisy, fn=80.0)                       # the exact force from CF
    assert met["area_active_mm2"] == pytest.approx(met0["area_active_mm2"])
    assert met["p_force_MPa"] == pytest.approx(80.0 / met0["area_active_mm2"])
    assert met["area_active_mm2"] == pytest.approx(8.0, rel=0.1)


def test_edge_peak_warnings_text():
    from types import SimpleNamespace
    from fea_core.contacts.results import edge_peak_warnings
    ok = SimpleNamespace(name="a", contacts={"P": {"peak_at_edge": False}})
    bad = SimpleNamespace(name="b", contacts={"P": {"peak_at_edge": True, "peak_edge_ratio": 5.4, "p_max_MPa": 579.0,
                                                    "p_max_interior_MPa": 107.0, "p_force_MPa": 26.8}})
    assert edge_peak_warnings([ok]) == []
    w = edge_peak_warnings([ok, bad, bad])
    assert len(w) == 1 and "Contact 'P'" in w[0] and "free edge" in w[0] and "579" in w[0] and "26.8" in w[0]


# ------------------------------------------------------------------ A7: rotation of a rigid body without NLGEOM
def test_rigid_rotation_warning():
    from fea_core.remote import rigid_rotations, rigid_rotation_note, RIGID_ROT_WARN
    from fea_core.study import RemoteLoad
    st = _study()
    st.load_cases[0].loads.append(RemoteLoad("vrch", [0, 0, 50.0], force=[0, 0, -1.0], coupling="rigid"))
    ref_nodes = {"0:0": {"ref": 101, "rot": 102}}
    refs = {101: np.array([1.0, 0, 0]), 102: np.array([0.0, 0.05, 0.0])}
    items = rigid_rotations(st, ref_nodes, refs, None)
    assert items == [("vrch", pytest.approx(0.05))]
    note = rigid_rotation_note(items, "Krok 1", nlgeom=False)
    assert note and "vrch" in note and "NLGEOM" in note and "distributing" in note
    assert rigid_rotation_note(items, "Krok 1", nlgeom=True) is None          # with large deformations it is fine
    assert rigid_rotation_note([("vrch", 0.5 * RIGID_ROT_WARN)], "Krok 1", nlgeom=False) is None
    # rigid remote constraint (RemoteSupport): the rotation from the reactions of the step
    sup = rigid_rotations(st, {}, None, {"pata": {"rot": np.array([0.0, 0.0, 0.04])}})
    assert sup == [("pata", pytest.approx(0.04))] and rigid_rotation_note(sup, "K", False)
