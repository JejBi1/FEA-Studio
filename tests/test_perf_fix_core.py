"""Equivalence of the performance fixes (11. 11. 2026): caches and shortcuts give the same values as the original computation.

- printing.region.sample_points: precomputed Sobol table = scipy.stats.qmc (bitwise),
- cylinder.support_directions / support_equations: cache on the mesh = fresh computation, invalidated by a change,
- assessment.part_nodes = np.unique(tets of the part); diagnostics.hotspots without cached principal stresses = with them,
- section_forces: the one-step job text and the one-pass .dat reader,
- convergence.reduced_study: only what the quantity needs, the same automatic solver,
- memory.release_unused: garbage collection rate limit, the working set still released.
"""
from __future__ import annotations

import copy
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ 3D printing: Sobol points
def test_sobol_table_equals_scipy():
    try:
        from scipy.stats import qmc
    except Exception:  # noqa: BLE001
        pytest.skip("scipy.stats not available")
    from fea_core.printing._sobol import SOBOL_3D_SEED7
    from fea_core.printing.region import sample_points
    warnings.simplefilter("ignore")
    for n in (4, 7, 16, 32, 33, 64, 100, 256, 512):
        ref_u = qmc.Sobol(3, scramble=True, seed=7).random(n)
        assert np.array_equal(SOBOL_3D_SEED7[:n], ref_u), n
        u = np.sort(ref_u, axis=1)
        ref = np.diff(np.hstack([np.zeros((n, 1)), u, np.ones((n, 1))]), axis=1)
        assert np.array_equal(sample_points(n), ref), n
    assert sample_points(600).shape == (600, 4)                 # beyond the table: scipy


# ------------------------------------------------------------------ cylindrical support cache
def _ring_study():
    from tests.test_cylindrical import ring_mesh
    from fea_core.study import CylindricalSupport, Displacement, Study
    m = ring_mesh(order=2)
    st = Study(name="t", geometry="")
    st.constraints = [CylindricalSupport("vnitrni", radial=True, tangential=True, axial=False),
                      Displacement("spodek", uz=0.0)]
    return st, m


def test_cylinder_cache_equals_fresh():
    from fea_core import cylinder as cy
    st, m = _ring_study()
    eq1 = cy.support_equations(st, m)
    fresh = cy._support_equations(st, m, None)
    assert eq1 == fresh
    eq1[0][1].append((9, 9.0))                                    # the caller may change its copy
    assert cy.support_equations(st, m) == fresh
    d1 = cy.support_directions(st, m)
    d0 = cy._support_directions(st, m)
    for k in d0:
        assert np.array_equal(d1[k][0], d0[k][0]) and np.array_equal(d1[k][1], d0[k][1])
    extra = np.zeros((len(m.nodes), 3), bool)
    extra[:5] = True
    assert cy.support_equations(st, m, extra) == cy._support_equations(st, m, extra)
    st.constraints[0].tangential = False                          # changed constraint → recomputed
    assert cy.support_equations(st, m) == cy._support_equations(st, m, None)
    m.nodes = m.nodes * 1.5                                       # changed mesh → recomputed
    assert cy.support_equations(st, m) == cy._support_equations(st, m, None)


# ------------------------------------------------------------------ assessment / diagnostics
def test_part_nodes_cached():
    from fea_core.assessment import part_nodes
    from fea_core.mesh import Mesh
    rng = np.random.default_rng(3)
    tets = rng.integers(0, 500, (400, 10))
    m = Mesh(rng.random((500, 3)), tets, rng.integers(0, 3, 400), ["a", "b", "c"])
    for k in range(3):
        assert np.array_equal(part_nodes(m, k), np.unique(m.tets[m.tet_part == k]))
    m.tet_part[:] = 0                                             # change in place → recomputed
    assert np.array_equal(part_nodes(m, 0), np.unique(m.tets))
    assert len(part_nodes(m, 1)) == 0


def _example2(tmp_path):
    fea = ROOT / "examples" / "example2_assembly.fea"
    if not fea.exists():
        pytest.skip("example 2 not available")
    from fea_core.project import load_project
    from fea_core.mcp.session import Session
    study, step, res = load_project(fea, tmp_path / "proj")
    sess = Session(tmp_path / "diag")
    sess.study, sess.results, sess.mesh = study, res, res.mesh
    return sess


def test_hotspots_without_cached_principal_equal(tmp_path):
    from fea_core.mcp import diagnostics as dg
    from fea_core.assessment import assessment_summary
    sess = _example2(tmp_path)
    res = sess.results
    for k in range(len(res.steps)):
        st = res.steps[k]
        st._principal = None
        a = dg.hotspots(sess, k, 8, "mises", tags=True)
        assert st._principal is None                              # not computed for all nodes
        _ = st.principal
        b = dg.hotspots(sess, k, 8, "mises", tags=True)
        assert repr(a) == repr(b)
    st = res.steps[1]
    s1 = assessment_summary(sess.study, res.mesh, st)
    res.mesh._cache.clear()
    assert repr(s1) == repr(assessment_summary(sess.study, res.mesh, st))


# ------------------------------------------------------------------ section force helpers
INP = """*NODE, NSET=NALL
1, 0, 0, 0
*ELEMENT, TYPE=C3D10, ELSET=P1
1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
*ELSET, ELSET=EALL
P1
*STEP
*STATIC
*CLOAD
1, 1, 5.
*NODE FILE
U, RF
*EL FILE
S
*NODE PRINT, NSET=NREF
U
*END STEP
*STEP
*STATIC
*BOUNDARY, OP=NEW
1, 1, 3
*CLOAD, OP=NEW
1, 2, 7.
*NODE FILE
U, RF
*EL PRINT, ELSET=EJOINT
S
*END STEP
"""


def test_step_only_job():
    from fea_core.section_forces import _step_only_job
    job = _step_only_job(INP, 1, "ECUT")
    assert job.count("*STEP") == 1 and "1, 2, 7." in job and "1, 1, 5." not in job
    assert "*NODE FILE" not in job and "EJOINT" not in job and "U, RF" not in job
    assert job.rstrip().endswith("*EL PRINT, ELSET=ECUT\nS\n*END STEP")
    assert job.startswith(INP[:INP.index("*STEP")])
    assert _step_only_job(INP, 2, "ECUT") is None


def test_dat_prints_one_pass(tmp_path):
    from fea_core.joints.forces import read_el_print
    from fea_core.section_forces import _dat_prints
    rows = []
    for st in (1, 2):
        for es in ("ESECTION", "EJOINT"):
            rows.append(f" stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set {es} and time  0.1000000E+01\n\n")
            for e in (1, 2):
                for ip in (1, 2, 3, 4):
                    v = [st * 10 + e + ip * 0.1 + c for c in range(6)]
                    rows.append(f"{e:10d}{ip:4d}" + "".join(f" {x:13.6E}" for x in v) + "\n")
            rows.append("\n")
    p = tmp_path / "j.dat"
    p.write_text("".join(rows), encoding="latin-1")
    got = _dat_prints(p)
    assert set(got) == {"ESECTION", "EJOINT"}
    for es in ("ESECTION", "EJOINT"):
        ref = read_el_print(p, es)
        assert len(got[es]) == len(ref) == 2
        for a, b in zip(got[es], ref):
            assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.slow
def test_section_force_step_job_equals_main_run(tmp_path):
    """Exact stresses from the extra one-step ccx job (fallback 3) = those printed by the main run (Study.section_forces)."""
    import dataclasses
    import verification.cases  # noqa: F401 – import order of the verification modules
    from verification.cases import v1_build
    from fea_core.solve import run_study
    from fea_core.section_forces import section_forces
    st, parts = v1_build(tmp_path)
    lc2 = copy.deepcopy(st.load_cases[0])
    lc2.name = "LC2"
    st.load_cases.append(lc2)
    st.solver = dataclasses.replace(st.solver, backend="ccx")
    st.section_forces = True
    res = run_study(st, tmp_path / "run", parts=parts)
    k = 1
    ref = [section_forces(st, res.mesh, res.steps[k], [x, 10.0, 5.0], [1, 0, 0]) for x in (40.0, 120.0)]
    s = res.steps[k]
    s._sig_ip = None
    for f in (tmp_path / "run").glob("*.dat"):                   # no printed stresses → the extra job of this step
        f.unlink()
    got = [section_forces(st, res.mesh, s, [x, 10.0, 5.0], [1, 0, 0]) for x in (40.0, 120.0)]
    assert len(s._sig_ip) == len(res.mesh.tets)                  # all elements: further cuts without ccx
    for a, b in zip(ref, got):
        assert np.allclose(a["F_local"] + a["M_local"], b["F_local"] + b["M_local"], rtol=1e-9, atol=1e-6)   # other ccx job (threads) - .dat noise
    assert abs(got[0]["Vz"] + 100.0) < 1e-3


# ------------------------------------------------------------------ convergence
def _conv_study(n_cases=4):
    from fea_core.study import Force, LoadCase, Study, ThermalLoad
    from fea_core.thermal.model import ThermalCase
    st = Study(name="c", geometry="")
    st.load_cases = [LoadCase(f"LC{i}", [Force("f", [0, 0, -i])]) for i in range(n_cases)]
    st.thermal.cases = [ThermalCase("T0"), ThermalCase("T1")]
    st.load_cases[2].loads.append(ThermalLoad(case="T1"))
    st.modal.enabled = True
    st.buckling.enabled = True
    return st


def test_reduced_study():
    from fea_core import config
    from fea_core.convergence import ConvergenceSpec, reduced_study
    st = _conv_study()
    r, sp = reduced_study(st, ConvergenceSpec(quantity="max_mises", step=2))
    assert len(r.load_cases) == max(1, min(4, config.PARDISO_MIN_CASES))
    assert r.load_cases[sp.step].name == "LC2"
    assert [c.name for c in r.thermal.cases] == (["T1"] if any(lc.name == "LC2" for lc in r.load_cases) else [])
    assert not r.modal.enabled and not r.buckling.enabled
    assert r.solver.backend == st.solver.backend and len(st.load_cases) == 4      # the original untouched
    r, sp = reduced_study(st, ConvergenceSpec(quantity="frequency", step=1))
    assert r.load_cases == [] and r.thermal.cases == [] and r.modal.enabled and sp.step == 1
    r, sp = reduced_study(st, ConvergenceSpec(quantity="max_mises", step=9))       # invalid step: unchanged cases
    assert len(r.load_cases) == 4
    from fea_core.study import RemoteLoad
    st.load_cases[3].loads.append(RemoteLoad("f", [0, 0, 0], force=[1, 0, 0], coupling="rigid"))
    r, sp = reduced_study(st, ConvergenceSpec(quantity="max_mises", step=0))
    assert r.solver.backend == "ccx"                              # auto → ccx because of a dropped case stays ccx


# ------------------------------------------------------------------ memory
def test_release_rate_limit(monkeypatch):
    import gc
    from fea_core import memory
    monkeypatch.delenv("FEA_RELEASE_MEMORY", raising=False)
    calls = []
    monkeypatch.setattr(gc, "collect", lambda *a: calls.append(1) or 0)
    memory._last_gc[0] = -1e9
    memory.release_unused()
    memory.release_unused()
    assert len(calls) == 1
    memory.release_unused(force=True)
    assert len(calls) == 2
