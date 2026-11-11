"""Tests of the feedback of the topology optimisation into the FEM: element stiffnesses in the .inp, element energies from the displacements, filter,
the loop `TopologyOptimizer.run()` (without ccx through a substitute run_study, with ccx on a small cantilever).
"""
import io
import re
import time

import numpy as np
import pytest

from fea_core.ccx.inp import ElementScale, write_inp
from fea_core.loads import nodal_loads
from fea_core.mesh import Mesh
from fea_core.selectors import FaceSelector
from fea_core.study import Fixed, Force, LoadCase, Material, Study
from fea_core.topo import (MIN_STIFFNESS_RATIO, TopologyOptSpec, TopologyOptimizer, SensitivityFilter,
                           element_stiffness_scale, element_strain_energies)

KUHN = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])
STEEL = Material("Ocel", E=210000.0, nu=0.3, rho=7.85e-9, Re=235.0, Rm=360.0)


def _to_tet10(m: Mesh) -> Mesh:
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    keys = np.sort(np.stack([m.tets[:, [a, b]] for a, b in edges], axis=1), axis=2).reshape(-1, 2)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    mids = 0.5 * (m.nodes[uniq[:, 0]] + m.nodes[uniq[:, 1]])
    return Mesh(np.vstack([m.nodes, mids]), np.hstack([m.tets, len(m.nodes) + inv.reshape(-1, 6)]),
                m.tet_part, m.part_names)


def beam_mesh(nx=8, ny=3, nz=1, h=5.0) -> Mesh:
    """Cantilever nx*h x ny*h x nz*h of hexahedra, each into 6 tetrahedra (tet10); selections fix (x = 0), load (x = max)."""
    xs, ys, zs = (np.arange(n + 1) * h for n in (nx, ny, nz))
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k  # noqa: E731
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                tets += [[v[a] for a in tt] for tt in KUHN]
    tets = np.array(tets)
    m = _to_tet10(Mesh(nodes, tets, np.zeros(len(tets), int), ["Konzola"]))
    tri, _ = m.boundary_faces()
    x = m.nodes[tri[:, :3]].mean(axis=1)
    m.surfaces = {"fix": tri[np.abs(x[:, 0]) < 1e-9], "load": tri[np.abs(x[:, 0] - nx * h) < 1e-9]}
    return m


def beam_study(F=(0.0, -1000.0, 0.0)) -> Study:
    st = Study("t", "m.step")
    st.add_material(STEEL, default=True)
    sel = lambda n: FaceSelector("Konzola", "plane", 1.0, [0, 0, 0], [0, 0, 0], n, [0, 0, 0, 1, 1, 1])  # noqa: E731
    st.add_selection("fix", [sel([-1, 0, 0])])
    st.add_selection("load", [sel([1, 0, 0])])
    st.constraints.append(Fixed("fix"))
    st.load_cases.append(LoadCase("LC1", [Force("load", list(F))]))
    return st


# ------------------------------------------------------------------ filtr

def test_filter_matches_bruteforce_and_volume_weights():
    rng = np.random.default_rng(3)
    c = rng.random((300, 3)) * 20.0
    vol = 0.5 + rng.random(300)
    r = 4.0
    f = SensitivityFilter(c, r, volumes=vol)
    alpha = rng.random(300)
    got = f.filter(alpha, use_history=False)
    d = np.linalg.norm(c[:, None] - c[None], axis=2)
    w = np.maximum(0.0, r - d) * vol[None, :]
    ref = (w @ alpha) / w.sum(axis=1)
    np.testing.assert_allclose(got, ref, rtol=1e-12)
    # without volumes a plain weighted average
    g2 = SensitivityFilter(c, r).filter(alpha, use_history=False)
    w2 = np.maximum(0.0, r - d)
    np.testing.assert_allclose(g2, (w2 @ alpha) / w2.sum(axis=1), rtol=1e-12)


def test_filter_isolated_element_is_identity_and_fast():
    c = np.array([[0.0, 0, 0], [100.0, 0, 0]])
    np.testing.assert_allclose(SensitivityFilter(c, 3.0).filter(np.array([2.0, 5.0]), use_history=False), [2.0, 5.0])
    rng = np.random.default_rng(0)
    big = rng.random((60000, 3)) * 100.0          # ~ 14 neighbours in r = 3 mm
    t0 = time.time()
    SensitivityFilter(big, 3.0)
    assert time.time() - t0 < 8.0


# ------------------------------------------------------------------ energy of the elements

def test_element_strain_energies_uniform_strain():
    m = beam_mesh(nx=2, ny=1, nz=1)
    st = beam_study()
    eps = np.array([1e-3, -2e-4, 3e-4])                       # normal strains, + shear xy
    gam = 5e-4
    U = np.column_stack([eps[0] * m.nodes[:, 0] + 0.5 * gam * m.nodes[:, 1],
                         eps[1] * m.nodes[:, 1] + 0.5 * gam * m.nodes[:, 0],
                         eps[2] * m.nodes[:, 2]])
    e = np.array([eps[0], eps[1], eps[2], gam, 0.0, 0.0])
    from fea_core.solve_large import iso_stiffness
    C = iso_stiffness(STEEL.E, STEEL.nu)
    from fea_core.loads import element_volumes
    ref = 0.5 * e @ C @ e * element_volumes(m)
    np.testing.assert_allclose(element_strain_energies(st, m, [U]), ref, rtol=1e-10)
    np.testing.assert_allclose(element_strain_energies(st, m, [U, 2 * U]), 5 * ref, rtol=1e-10)   # sum of the states


def test_stiffness_scale_floor():
    s = element_stiffness_scale(np.array([1.0, 0.5, 1e-3, 0.0]), 3.0)
    np.testing.assert_allclose(s, [1.0, 0.125, MIN_STIFFNESS_RATIO, MIN_STIFFNESS_RATIO])


# ------------------------------------------------------------------ writing the .inp

def _inp(st, m, tmp_path, name, **kw):
    p = tmp_path / name
    write_inp(st, m, p, **kw)
    return p.read_bytes()


def test_inp_identical_without_scale_and_with_unit_scale(tmp_path):
    m, st = beam_mesh(), beam_study()
    base = _inp(st, m, tmp_path, "a.inp")
    assert _inp(st, m, tmp_path, "b.inp", element_scale=np.ones(len(m.tets))) == base
    with ElementScale(np.ones(len(m.tets))) as es:
        assert _inp(st, m, tmp_path, "c.inp") == base
        assert es.used == 1
    assert _inp(st, m, tmp_path, "d.inp") == base                 # after leaving the context without influence


def test_inp_scaled_elements_get_own_material_and_cover_all_elements_once(tmp_path):
    m, st = beam_mesh(), beam_study()
    scale = np.ones(len(m.tets))
    scale[::3] = 1e-6
    scale[1::7] = 0.25
    txt = "\n".join(_inp(st, m, tmp_path, "s.inp", element_scale=scale).decode("ascii").splitlines()) + "\n"
    # materials with multiples of E (E·1e-6 = 0.21, E·0.25 = 52500) and density
    assert re.search(r"\*ELASTIC\n0\.21,", txt)
    assert re.search(r"\*ELASTIC\n52500,", txt)
    # each element in exactly one *SOLID SECTION
    sets = {}
    for mt in re.finditer(r"\*ELSET, ELSET=(\w+)\n((?:[\d, ]+\n)+)", txt):
        sets[mt.group(1)] = [int(v) for v in re.findall(r"\d+", mt.group(2))]
    secs = re.findall(r"\*SOLID SECTION, ELSET=(\w+), MATERIAL=(\w+)", txt)
    assert secs and all(name != "P1" for name, _ in secs)            # the original uniform section of the part disappeared
    covered = [e for name, _ in secs for e in sets[name]]
    assert sorted(covered) == list(range(1, len(m.tets) + 1))
    for name, mat in secs:
        if mat.endswith("K0") or "K" not in mat:
            continue
    # elements with the same multiple share the material
    by_mat = {}
    for name, mat in secs:
        by_mat.setdefault(mat, []).extend(sets[name])
    for mat, els in by_mat.items():
        vals = {round(float(scale[e - 1]), 6) for e in els}
        assert len(vals) == 1, (mat, vals)


def test_inp_scale_rejects_bad_input(tmp_path):
    m, st = beam_mesh(), beam_study()
    with pytest.raises(ValueError):
        write_inp(st, m, tmp_path / "x.inp", element_scale=np.ones(3))
    bad = np.ones(len(m.tets))
    bad[0] = 0.0
    with pytest.raises(ValueError):
        write_inp(st, m, tmp_path / "y.inp", element_scale=bad)


# ------------------------------------------------------------------ the run() loop

def test_run_requires_study_and_load_case():
    m = beam_mesh()
    with pytest.raises(ValueError, match="study"):
        TopologyOptimizer(m, spec=TopologyOptSpec()).run()
    st = beam_study()
    st.load_cases.clear()
    with pytest.raises(ValueError, match="load case"):
        TopologyOptimizer(st, m, TopologyOptSpec()).run()
    st2 = beam_study()
    opt = TopologyOptimizer(st2, m, TopologyOptSpec())
    opt.mesh = _to_tet4(m)
    assert any("tet10" in e for e in opt.check_study())


def _to_tet4(m: Mesh) -> Mesh:
    return Mesh(m.nodes, m.tets[:, :4], m.tet_part, m.part_names)


class _FakeSteps:
    def __init__(self, U):
        self.steps = [type("S", (), {"U": U})()]


def test_run_feeds_densities_back_each_iteration(monkeypatch, tmp_path):
    """The loop passes to the calculation in every iteration the stiffnesses corresponding to the CURRENT densities (not always the original ones)."""
    import fea_core.ccx.inp as inp_mod
    import fea_core.solve as solve_mod

    m, st = beam_mesh(nx=6, ny=3), beam_study()
    seen = []

    def fake_run_study(study, workdir, **kw):
        es = inp_mod._ELEMENT_SCALE
        assert es is not None, "run_study běží mimo kontext ElementScale"
        es.used += 1                                           # as if the write_inp wrote it
        seen.append(es.scale.copy())
        assert study.solver.backend == "ccx"                   # NGSolve would silently ignore the multiples
        # displacements = the bending of the cantilever ~ x²; the stiffness of the elements does not affect them, it is enough for non-zero sensitivities
        x = m.nodes
        U = np.column_stack([np.zeros(len(x)), -1e-3 * x[:, 0] ** 2, np.zeros(len(x))])
        return _FakeSteps(U)

    monkeypatch.setattr(solve_mod, "run_study", fake_run_study)
    spec = TopologyOptSpec(target_volume_fraction=0.6, evolution_rate=0.1, filter_radius=7.0, max_iterations=4,
                           non_design_selections=["fix", "load"])
    st.solver.backend = "ngs_pardiso"
    opt = TopologyOptimizer(st, m, spec)
    opt.run(workdir=tmp_path)
    assert st.solver.backend == "ngs_pardiso"                  # the original study does not change
    assert len(seen) == len(opt.history) + 1                   # + closing evaluation of the resulting design
    assert np.all(seen[0] == 1.0)                              # first iteration: full material
    assert not np.array_equal(seen[1], seen[0])                # further ones already with the material removed
    assert not np.array_equal(seen[2], seen[1])
    np.testing.assert_allclose(seen[-1], element_stiffness_scale(opt.rho, spec.penalty))
    assert seen[1].min() == pytest.approx(MIN_STIFFNESS_RATIO)
    assert opt.initial_compliance and opt.final_compliance
    assert opt.non_design_mask.any() and np.all(opt.rho[opt.non_design_mask] == 1.0)


# ------------------------------------------------------------------ real ccx

@pytest.mark.slow
def test_run_with_ccx_small_cantilever(tmp_path):
    """The real loop with ccx: compliance of the full design = F·u, volume met, the result stiffer than an equally heavy strip."""
    from fea_core.solve import run_study
    m, st = beam_mesh(nx=10, ny=4, nz=1, h=5.0), beam_study()
    spec = TopologyOptSpec(target_volume_fraction=0.6, evolution_rate=0.08, filter_radius=4.0, max_iterations=10,
                           non_design_selections=["fix", "load"])
    opt = TopologyOptimizer(st, m, spec)

    # energy of the full design = ½ F·u (compliance F·u = 2 Σ U0)
    res = run_study(st, tmp_path / "full", mesh=m)
    F = nodal_loads(st, m, st.load_cases[0])[0]
    fu = float(np.sum(F * res.steps[0].U))
    u0 = opt.evaluate(tmp_path / "e0")
    assert opt.current_compliance(u0) == pytest.approx(fu, rel=2e-3)

    opt.run(workdir=tmp_path / "opt")
    assert opt.current_volume_fraction == pytest.approx(0.6, abs=0.02)
    assert opt.final_compliance > opt.initial_compliance
    # a strip of the same volume (the middle 3 of 4 rows in the height y – 75 %; 60 % is taken as the middle strip by the centroids)
    cy = opt.centroids[:, 1]
    keep = np.abs(cy - 10.0) <= 6.0
    naive = np.where(keep | opt.non_design_mask, 1.0, spec.min_density)
    opt.rho = naive
    c_naive = opt.current_compliance(opt.evaluate(tmp_path / "naive"))
    assert opt.current_volume_fraction >= 0.58
    opt_c = opt.final_compliance
    assert opt_c < c_naive
