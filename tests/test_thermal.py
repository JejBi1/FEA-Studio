"""Fast tests of heat (without Gmsh and ccx): model ↔ JSON, integration of the nodal heats, writing of the .inp, balance,
temperature field for the mechanics, readiness check. Calculations against the analytics: verification T1–T7 (slow).
"""
import numpy as np
import pytest

from fea_core.mesh import Mesh
from fea_core.results import Results, ThermalResult, StepResult
from fea_core.study import Study, Material, LoadCase, ThermalLoad, Fixed
from fea_core.thermal.model import (ThermalCase, FixedTemperature, HeatFlux, HeatFlow, Convection, Radiation,
                                    HeatGeneration, ThermalSettings)

KUHN = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])


def bar_mesh(L=100.0, a=10.0, nx=10, ny=2, nz=2, order=2, parts=1) -> Mesh:
    """A bar L × a × a made of cubes of 6 tetrahedra each (tet10, midpoints on the segments). Selections x0, x100, the sides."""
    x, y, z = np.linspace(0, L, nx + 1), np.linspace(0, a, ny + 1), np.linspace(0, a, nz + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    tets, tp = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                for t in KUHN:
                    tets.append([v[q] for q in t])
                    tp.append(min(parts - 1, i * parts // nx))
    tets = np.array(tets)
    if order == 2:
        edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
        keys = np.sort(np.stack([tets[:, [p, q]] for p, q in edges], axis=1), axis=2).reshape(-1, 2)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        nodes = np.vstack([nodes, 0.5 * (nodes[uniq[:, 0]] + nodes[uniq[:, 1]])])
        tets = np.hstack([tets, len(nodes) - len(uniq) + inv.reshape(-1, 6)])
    m = Mesh(nodes, tets, np.array(tp), [f"Dil{i + 1}" for i in range(parts)])
    tri, _ = m.boundary_faces()
    c = m.nodes[tri[:, :3]].mean(axis=1)
    m.surfaces = {"x0": tri[np.abs(c[:, 0]) < 1e-9], "x100": tri[np.abs(c[:, 0] - L) < 1e-9],
                  "y0": tri[np.abs(c[:, 1]) < 1e-9]}
    return m


def study(parts=1) -> Study:
    st = Study("t", "x.step")
    st.add_material(Material("Ocel", 210000, 0.3, 7.85e-9, k=50.0, alpha=12e-6, cp=460.0), default=True)
    for s in ("x0", "x100", "y0"):
        st.selections[s] = None          # the mesh already has the selections; check() is called with part_names
    return st


def test_json_roundtrip():
    st = Study("t", "x.step")
    st.thermal = ThermalSettings([ThermalCase("A", [FixedTemperature("a", 100), HeatFlux("b", 5e3), HeatFlow("c", 2),
                                                    Convection("d", 10, 25), Radiation("e", 0.9, 30),
                                                    HeatGeneration("P", 4)], transient=True, time=60, dt=1)],
                                 reference_temperature=22.0, initial_temperature=40.0)
    st.load_cases = [LoadCase("L", [ThermalLoad(case="A", time=30.0), ThermalLoad(value=80.0, parts=["P"])])]
    d = st.to_dict()
    st2 = Study.from_dict(d)
    assert st2.to_dict() == d
    assert st2.thermal.t_initial == 40.0
    assert Study.from_dict({k: v for k, v in d.items() if k != "thermal"}).thermal.cases == []   # an older study


def test_face_and_volume_integration():
    from fea_core.thermal.loads import bc_nodal, curved_area
    m = bar_mesh()
    assert curved_area(m, "x100") == pytest.approx(100.0)
    f = bc_nodal(m, HeatFlux("x100", 1e4), None)                 # 1e4 W/m² · 100 mm² = 1 W
    assert f.sum() == pytest.approx(1.0, rel=1e-12)
    assert bc_nodal(m, HeatFlow("x100", 3.0), None).sum() == pytest.approx(3.0, rel=1e-12)
    T = 20 + 0.5 * m.nodes[:, 0]                                 # linear field: at the face x = 100 → 70 °C
    f = bc_nodal(m, Convection("x100", 200.0, 20.0), T)
    assert f.sum() == pytest.approx(200 * (20 - 70) * 1e-4, rel=1e-12)
    g = bc_nodal(m, HeatGeneration("Dil1", 7.0), None)
    assert g.sum() == pytest.approx(7.0, rel=1e-12)
    r = bc_nodal(m, Radiation("x100", 1.0, 20.0), np.full(len(m.nodes), 20.0))
    assert np.abs(r).max() < 1e-15


def test_heat_balance_and_reactions():
    """Synthetic RFL: 1D conduction T(x) linear, flux q at x100 → reaction at x0 = −q·A, balance 0."""
    from fea_core.thermal.loads import bc_nodal
    from fea_core.thermal.run import heat_flows
    m = bar_mesh()
    case = ThermalCase("A", [FixedTemperature("x0", 20.0), HeatFlux("x100", 1e4)])
    T = 20 + 0.2 * m.nodes[:, 0]
    f = bc_nodal(m, case.loads[1], T)
    rfl = f.copy()                                               # at the free nodes RFL = external heat
    fix = np.unique(m.surfaces["x0"])
    rfl[fix] = 0.0
    rfl[fix] = -f.sum() / len(fix)                               # the sum of RFL over the mesh = 0
    heat, bal = heat_flows(study(), m, case, T, rfl)
    q = {h["kind"]: h["Q_W"] for h in heat}
    assert q["heat_flux"] == pytest.approx(1.0)
    assert q["fixed_temperature"] == pytest.approx(-1.0)
    assert bal["imbalance"] < 1e-12


def test_write_thermal_inp(tmp_path):
    from fea_core.thermal.inp import write_model, write_case, time_stepping
    m = bar_mesh(parts=2)
    st = study(parts=2)
    st.thermal.cases = [ThermalCase("A", [FixedTemperature("x0", 100.0), Convection("x100", 500.0, 20.0),
                                          HeatFlux("y0", 1000.0), Radiation("x100", 0.5, 20.0),
                                          HeatGeneration("Dil2", 1.0)]),
                        ThermalCase("B", [FixedTemperature("x0", 100.0)], transient=True, time=95.0, dt=10.0,
                                    outputs=4)]
    mp = write_model(st, m, tmp_path / "model.inc")
    txt = (tmp_path / "model.inc").read_text()
    assert "*CONDUCTIVITY\n50\n" in txt and "*SPECIFIC HEAT\n460000000\n" in txt
    assert "STEFAN BOLTZMANN=5.670374419e-11" in txt
    write_case(st, m, mp, st.thermal.cases[0], tmp_path / "case1.inp", "case1")
    c1 = (tmp_path / "case1.inp").read_text()
    assert "*HEAT TRANSFER, STEADY STATE" in c1 and "*FILM" in c1 and "*RADIATE" in c1
    assert ", F" in c1 and ", 20, 0.5\n" in c1 and "P2, BF, " in c1
    assert ", S" in c1 and ", 1\n" in c1                       # 1000 W/m² = 1 mW/mm²
    tj = write_case(st, m, mp, st.thermal.cases[1], tmp_path / "case2.inp", "case2")
    assert (tj.increments, tj.frequency) == (12, 3)               # 10 steps → a multiple of 4 frames
    assert tj.dt == pytest.approx(95.0 / 12)
    assert time_stepping(ThermalCase("C", transient=True, time=10.0, dt=0.0, outputs=10)) == (100, 10, 0.1)


def test_node_temperatures_and_force_scale():
    from fea_core.thermal.field import node_temperatures, force_scale
    m = bar_mesh(parts=2)
    st = study(parts=2)
    st.thermal.reference_temperature = 20.0
    tr = ThermalResult("A", 20 + 0.5 * m.nodes[:, 0], np.zeros((len(m.nodes), 3)), True, [0.0, 10.0],
                       np.vstack([np.full(len(m.nodes), 20.0), 20 + 0.5 * m.nodes[:, 0]]))
    assert node_temperatures(st, m, LoadCase("L", []), {}) is None
    T = node_temperatures(st, m, LoadCase("L", [ThermalLoad(case="A", time=5.0)]), {"A": tr})
    assert T == pytest.approx(20 + 0.25 * m.nodes[:, 0])
    T = node_temperatures(st, m, LoadCase("L", [ThermalLoad(case="A"), ThermalLoad(value=100.0, parts=["Dil1"])]),
                          {"A": tr})
    d1 = np.unique(m.tets[m.tet_part == 0])
    d2 = np.unique(m.tets[m.tet_part == 1])
    only1 = np.setdiff1d(d1, d2)
    shared = np.intersect1d(d1, d2)
    assert np.all(T[only1] == 100.0)
    assert np.all(T[shared] == 100.0)                              # a single uniform temperature → overwrites the field
    T2 = node_temperatures(st, m, LoadCase("L", [ThermalLoad(value=100.0, parts=["Dil1"]),
                                                 ThermalLoad(value=50.0, parts=["Dil2"])]), {})
    assert np.all(T2[shared] == 75.0)                              # interface = average
    s = force_scale(st, m, np.full(len(m.nodes), 120.0))
    L = np.sqrt(100 ** 2 + 2 * 10 ** 2)
    assert s == pytest.approx(210000 * 12e-6 * 100 * 1e4 / L, rel=1e-9)


def test_check_messages():
    from fea_core.thermal.check import check, floating_parts
    st = Study("t", "x.step")
    st.add_material(Material("Bez tepla", 210000, 0.3, 7.85e-9), default=True)
    st.thermal.cases = [ThermalCase("A", [HeatFlux("x", 10.0)]),
                        ThermalCase("B", [FixedTemperature("x", 20.0)], transient=True, time=0.0),
                        ThermalCase("C", [Convection("x", -1.0, 20.0), Radiation("x", 1.5, 20.0)])]
    st.load_cases = [LoadCase("L", [ThermalLoad(case="X"), ThermalLoad()])]
    errors, warnings, used = check(st, ["Dil1"])
    text = "\n".join(errors)
    assert "thermal conductivity" in text and "specific heat c_p" in text
    assert "no way to dissipate heat" in text
    assert "transient duration must be positive" in text
    assert "heat transfer coefficient" in text and "emissivity" in text
    assert "'X' does not exist" in text and "neither a thermal case nor a temperature" in text
    assert "coefficient of thermal expansion" in text
    assert used == {"x"}
    m = bar_mesh(parts=2)
    m.surfaces["x0"] = m.surfaces["x0"]
    assert floating_parts(st, m, ThermalCase("D", [FixedTemperature("x0", 20.0)])) == []
    assert floating_parts(st, m, ThermalCase("D", [FixedTemperature("x0", 20.0)], transient=True)) == []
    m2 = bar_mesh(parts=2)
    m2.tets = m2.tets.copy()
    # disconnect the parts: move Dil2 so that it shares no nodes → without a heat sink
    off = len(m2.nodes)
    sel = m2.tet_part == 1
    used_nodes = np.unique(m2.tets[sel])
    remap = {int(n): off + i for i, n in enumerate(used_nodes)}
    m2.nodes = np.vstack([m2.nodes, m2.nodes[used_nodes]])
    m2.tets[sel] = np.vectorize(remap.get)(m2.tets[sel])
    assert floating_parts(st, m2, ThermalCase("D", [FixedTemperature("x0", 20.0)])) == ["Dil2"]


def test_results_save_load_thermal(tmp_path):
    m = bar_mesh(order=1)
    n = len(m.nodes)
    tr = ThermalResult("A", np.linspace(20, 30, n), np.ones((n, 3)), True, [1.0, 2.0],
                       np.vstack([np.full(n, 20.0), np.linspace(20, 30, n)]),
                       [{"kind": "fixed_temperature", "name": "x0", "Q_W": 1.5}], {"imbalance": 0.0})
    st = StepResult("L", np.zeros((n, 3)), np.zeros((n, 6)))
    st.fields["T"] = np.full(n, 50.0)
    res = Results(m, [st], [], {}, [tr])
    res.save(tmp_path / "results.npz")
    r2 = Results.load(tmp_path / "results.npz")
    t2 = r2.thermal_case("A")
    assert t2.transient and t2.times == [1.0, 2.0] and t2.heat == tr.heat
    assert np.allclose(t2.frames, tr.frames) and np.allclose(t2.T, tr.T)
    assert np.allclose(r2.step(0).fields["T"], 50.0)
    assert r2.summary()["thermal"][0]["T_max_C"] == pytest.approx(30.0)
    assert r2.probe(m.nodes[5], "T", "A")[0] == pytest.approx(tr.T[5])
    # heat-only study: without mechanical states
    r3 = Results(m, [], [], {}, [tr])
    r3.save(tmp_path / "r3.npz")
    assert Results.load(tmp_path / "r3.npz").steps == []


def test_thermal_only_readiness():
    from fea_core.solve import check_study
    st = Study("t", "x.step")
    st.add_material(Material("Ocel", 210000, 0.3, 7.85e-9, k=50.0, alpha=12e-6, cp=460.0), default=True)
    st.selections["x0"] = __import__("fea_core.study", fromlist=["Selection"]).Selection("x0", [])
    errors, _ = check_study(st, ["Dil1"])
    assert any("no load case or thermal case" in e for e in errors)
    st.thermal.cases = [ThermalCase("A", [FixedTemperature("x0", 20.0), HeatGeneration("Dil1", 1.0)])]
    errors, _ = check_study(st, ["Dil1"])
    assert errors == []                                           # without constraints – heat conduction only
    st.load_cases = [LoadCase("L", [ThermalLoad(case="A")])]
    errors, _ = check_study(st, ["Dil1"])
    assert any("no support" in e for e in errors)
    st.constraints.append(Fixed("x0"))
    assert check_study(st, ["Dil1"])[0] == []


def test_bonded_pairs():
    """Overview of the bonded parts (contacts): a bar of 2 parts → one pair with a cross-section area of 100 mm²."""
    from fea_core.contacts import bonded_pairs, edge_only
    m = bar_mesh(parts=2)
    pairs = bonded_pairs(m)
    assert len(pairs) == 1 and {pairs[0]["a"], pairs[0]["b"]} == {"Dil1", "Dil2"}
    assert pairs[0]["area"] == pytest.approx(100.0)
    assert edge_only(m) == []
    assert bonded_pairs(bar_mesh(parts=1)) == []
