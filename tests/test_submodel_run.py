"""Submodel of a solved study (fea_core.submodel_run): region, storage with the results, end-to-end run on a small plate
(transferred loads, equilibrium, unsupported conditions = error, saving in the .fea project)."""
from __future__ import annotations

import numpy as np
import pytest

from fea_core.mesh import Mesh
from fea_core.results import Results, StepResult
from fea_core.submodel_run import (SubmodelError, SubmodelRegion, SubmodelResult, load_submodels, save_submodels,
                                   sub_field)


def test_region_distance_and_check():
    b = SubmodelRegion(center=[0, 0, 0], shape="box", size=[10, 20, 30], mesh_size=1.0)
    d = b.signed_distance(np.array([[0, 0, 0], [5, 0, 0], [0, 11, 0], [4, 9, 14]]))
    assert np.allclose(d, [-5, 0, 1, -1])
    s = SubmodelRegion(center=[1, 1, 1], shape="sphere", radius=2.0, mesh_size=0.5)
    assert np.allclose(s.signed_distance(np.array([[1, 1, 1], [3, 1, 1]])), [-2, 0])
    b.check()
    s.check()
    for bad in (SubmodelRegion([0, 0, 0], "cube"), SubmodelRegion([0, 0], "box"),
                SubmodelRegion([0, 0, 0], "box", size=[10, -1, 10]), SubmodelRegion([0, 0, 0], "sphere", radius=1, mesh_size=1),
                SubmodelRegion([0, 0, 0], "box", size=[2, 2, 2], mesh_size=1.5)):
        with pytest.raises(SubmodelError):
            bad.check()
    assert SubmodelRegion.from_dict(s.to_dict()) == s


def _fake(n=10) -> SubmodelResult:
    rng = np.random.default_rng(1)
    nodes = rng.random((n, 3))
    mesh = Mesh(nodes, np.arange(10)[None, :], np.zeros(1, np.int64), ["Deska"])
    return SubmodelResult(SubmodelRegion([0.5, 0.5, 0.5], "box", [1, 1, 1], mesh_size=0.2, name="S"), "Tah", mesh,
                          rng.random((n, 3)), rng.random((n, 6)), np.array([0, 3]), np.array([1.0, 2.0]),
                          {"max_mises_MPa": 3.0, "cut_check": {"passed": True}}, ["w1"])


def test_save_load_roundtrip(tmp_path):
    a = _fake()
    save_submodels([a, a], tmp_path / "s.npz")
    b = load_submodels(tmp_path / "s.npz")
    assert len(b) == 2 and b[1].name == "S" and b[0].step_name == "Tah" and b[0].warnings == ["w1"]
    assert np.array_equal(b[0].U, a.U) and np.array_equal(b[0].S, a.S) and np.array_equal(b[0].cut_nodes, a.cut_nodes)
    assert b[0].check["passed"] and b[0].region == a.region
    assert np.allclose(sub_field(b[0], "mises"), sub_field(a, "mises"))


def test_results_save_keeps_submodels(tmp_path):
    a = _fake()
    res = Results(a.mesh, [StepResult("Tah", a.U, a.S)])
    res.save(tmp_path / "results.npz")
    assert not (tmp_path / "results_sub.npz").exists()
    res.submodels.append(a)
    res.save(tmp_path / "results.npz")
    r2 = Results.load(tmp_path / "results.npz")
    assert len(r2.submodels) == 1 and np.array_equal(r2.submodels[0].U, a.U)
    res.submodels.clear()
    res.save(tmp_path / "results.npz")                         # stale file is removed
    assert not (tmp_path / "results_sub.npz").exists() and Results.load(tmp_path / "results.npz").submodels == []


def test_load_rejects_bad_arrays(tmp_path):
    a = _fake()
    a.cut_nodes = np.array([0, 99])                            # node outside the mesh
    save_submodels([a], tmp_path / "s.npz")
    with pytest.raises(ValueError):
        load_submodels(tmp_path / "s.npz")


# ===================================================================== end-to-end (Gmsh + ccx)

@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from fea_core.geometry import Part, read_step, write_step
    from fea_core.selectors import find_faces
    from fea_core.solve import run_study
    from fea_core.study import Fixed, Force, LoadCase, Material, MeshSettings, Pressure, Study
    from verification.cases import box, cyl, solid
    d = tmp_path_factory.mktemp("sub")
    shape = solid(BRepAlgoAPI_Cut(box(0, 0, 0, 60, 40, 5), cyl((30, 20, -1), (0, 0, 1), 5, 7)).Shape())
    write_step([Part("Deska", shape)], d / "plate.step")
    parts = read_step(d / "plate.step")
    st = Study(name="P", geometry=str(d / "plate.step"), mesh=MeshSettings(size=5.0))
    st.add_material(Material("Ocel", 210000.0, 0.3, 7.85e-9, Re=235.0), default=True)
    st.add_selection("levy", find_faces(parts, x=0))
    st.add_selection("pravy", find_faces(parts, x=60))
    st.add_selection("horni", find_faces(parts, z=5))
    st.add_selection("otvor", find_faces(parts, radius=5))
    st.constraints.append(Fixed("levy"))
    st.load_cases.append(LoadCase("Tah", [Force("pravy", [2000.0, 0.0, 0.0]), Pressure("horni", 1.0)]))
    res = run_study(st, workdir=d / "run", parts=parts)
    return st, parts, res, d


@pytest.mark.slow
def test_run_submodel_transfers_loads(plate):
    from fea_core.submodel_run import run_submodel
    st, parts, res, d = plate
    region = SubmodelRegion([30, 20, 2.5], "box", [20, 20, 7], mesh_size=1.5)
    sr = run_submodel(st, parts, res, region, d / "s1")
    s = sr.summary
    a_top = 20 * 20 - np.pi * 25
    assert s["equilibrium"]["applied_force_N"][2] == pytest.approx(-1.0 * a_top, rel=2e-3)   # pressure inside the region
    assert s["equilibrium"]["imbalance"] < 1e-3
    # one coarse element through the bent 5 mm plate: the cut check may flag it – then the warning must say so
    assert s["cut_check"]["nodes"] > 0 and s["cut_check"]["mean_rel_diff"] < 0.5
    assert s["cut_check"]["passed"] == (not any(w.startswith("Cut check") for w in sr.warnings))
    assert {m["selection"] for m in s["mapped"]} == {"horni"}
    assert s["max_mises_MPa"] > 0 and s["global_max_mises_MPa"] > 0
    # submodel over the clamped face: the support is transferred, displacement there stays zero
    sr2 = run_submodel(st, parts, res, SubmodelRegion([3, 20, 2.5], "box", [10, 50, 8], mesh_size=1.5), d / "s2")
    assert any(m["selection"] == "levy" for m in sr2.summary["mapped"])
    on_clamp = np.abs(sr2.mesh.nodes[:, 0]) < 1e-6
    assert on_clamp.any() and np.abs(sr2.U[on_clamp]).max() < 1e-12


@pytest.mark.slow
def test_run_submodel_errors(plate):
    from dataclasses import replace
    from fea_core.study import BearingLoad, LoadCase
    from fea_core.submodel_run import run_submodel
    st, parts, res, d = plate
    with pytest.raises(SubmodelError, match="does not intersect"):
        run_submodel(st, parts, res, SubmodelRegion([300, 20, 2.5], "box", [10, 10, 10], mesh_size=1.0), d / "e1")
    with pytest.raises(SubmodelError, match="out of range"):
        run_submodel(st, parts, res, SubmodelRegion([30, 20, 2.5], "box", [20, 20, 7], mesh_size=1.5, step=3), d / "e2")
    st2 = replace(st, load_cases=[LoadCase("Tah", [BearingLoad("otvor", [100.0, 0.0, 0.0])])])
    with pytest.raises(SubmodelError, match="otvor"):
        run_submodel(st2, parts, res, SubmodelRegion([30, 20, 2.5], "box", [20, 20, 7], mesh_size=1.5), d / "e3")


@pytest.mark.slow
def test_submodel_saved_in_project(plate, tmp_path):
    from fea_core.project import load_project, save_project
    from fea_core.submodel_run import run_submodel
    st, parts, res, d = plate
    sr = run_submodel(st, parts, res, SubmodelRegion([30, 20, 2.5], "sphere", radius=8.0, mesh_size=1.5), d / "s3")
    res.submodels = [sr]
    try:
        save_project(tmp_path / "p.fea", st, st.geometry, res)
        _, _, r2 = load_project(tmp_path / "p.fea", tmp_path / "wd")
    finally:
        res.submodels = []
    assert len(r2.submodels) == 1
    assert r2.submodels[0].summary["max_mises_MPa"] == pytest.approx(sr.summary["max_mises_MPa"])
    assert r2.submodels[0].region.shape == "sphere"
    # Word report: a section with the cut check and the image of the submodel
    pytest.importorskip("docx")
    from fea_core.report import ReportOptions, build_report
    from fea_core.report.word import session_from
    sess = session_from(st, st.geometry, r2, parts)
    out = build_report(sess, tmp_path / "r.docx", ReportOptions(language="en"))
    import docx
    text = "\n".join(p.text for p in docx.Document(out["path"]).paragraphs)
    assert "Submodel" in text and not [w for w in out["warnings"] if "Submodel" in w], out["warnings"]
    assert any("submodel_0_mises" in p for p in out.get("images", [])) or out["figures"] >= 2
