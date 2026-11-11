"""Security fixes of the core: planted DLLs in a solver work folder, mesh / result files from outside (deck injection,
index ranges), REST file download allow-list and streamed import, MCP save / STL paths, .fea unpack limits."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from fea_core import dllguard


# ------------------------------------------------------------------ 2. planted DLLs
def test_workdir_with_dll_refused(tmp_path):
    (tmp_path / "job.inp").write_text("")
    dllguard.check_workdir(tmp_path)                       # inputs only = fine
    (tmp_path / "mkl_rt.2.dll").write_bytes(b"MZ")
    with pytest.raises(dllguard.PlantedCodeError):
        dllguard.check_workdir(tmp_path, "CalculiX")


def test_run_ccx_refuses_planted_dll(tmp_path):
    from fea_core.ccx.runner import run_ccx
    (tmp_path / "MKL_RT.2.DLL").write_bytes(b"MZ")
    exe = tmp_path.parent / "fake_ccx.exe"
    exe.write_bytes(b"")
    with pytest.raises(dllguard.PlantedCodeError):
        run_ccx(tmp_path, "job", exe=exe)


def test_cli_default_workdir_not_reused(tmp_path):
    shipped = tmp_path / "study_run"
    shipped.mkdir()
    (shipped / "readme.txt").write_text("came with the study")
    wd = dllguard.own_workdir(shipped)
    assert wd != shipped and wd.name == "study_run_2" and (wd / dllguard.MARKER).is_file()
    assert dllguard.own_workdir(shipped) == wd              # our own folder is reused
    empty = tmp_path / "e_run"
    empty.mkdir()
    assert dllguard.own_workdir(empty) == empty
    (wd / "evil.dll").write_bytes(b"MZ")                     # a marked folder with code is not reused either
    assert dllguard.own_workdir(shipped).name == "study_run_3"


# ------------------------------------------------------------------ 3. mesh report from a file
def _mesh():
    from fea_core.mesh import Mesh
    nodes = np.random.default_rng(0).random((12, 3))
    tets = np.arange(10)[None, :].astype(np.int64)
    return Mesh(nodes, tets, np.zeros(1, np.int64), ["P"], {"s": np.array([[0, 1, 2, 4, 5, 6]])},
                {"contacts": [{"pair": "c", "index": 1, "slave": "s", "master": "s", "kind": "bonded"}]})


def test_mesh_roundtrip_and_report_validation(tmp_path):
    from fea_core.mesh import Mesh, MeshError
    m = _mesh()
    p = tmp_path / "m.npz"
    m.save(p)
    assert Mesh.load(p).report["contacts"][0]["index"] == 1
    for bad in ("1X\n*INCLUDE, INPUT=x\n**", None, True, 1.5):
        m2 = _mesh()
        m2.report["contacts"][0]["index"] = bad
        m2.save(p)
        with pytest.raises(MeshError):
            Mesh.load(p)
    m2 = _mesh()
    m2.report["ties"] = [{"slave": "s", "master": "s", "tolerance_mm": "0.1\n*INCLUDE"}]
    m2.save(p)
    with pytest.raises(MeshError):
        Mesh.load(p)


@pytest.mark.parametrize("field,value", [
    ("tets", np.full((1, 10), 99, np.int64)),                 # node index out of range
    ("tet_part", np.array([5])),                              # part index out of range
    ("surfaces", {"s": np.array([[0, 1, 2, 4, 5, 600]])}),
    ("nodes", np.zeros((12, 2))),
])
def test_mesh_index_ranges_checked(tmp_path, field, value):
    from fea_core.mesh import Mesh, MeshError
    m = _mesh()
    setattr(m, field, value)
    p = tmp_path / "m.npz"
    m.save(p)
    with pytest.raises(MeshError):
        Mesh.load(p)


def test_contact_kind_goes_through_deck_text():
    import inspect
    from fea_core.contacts import inp
    assert "({deck_text(c.kind)})" in inspect.getsource(inp.write_pairs)


# ------------------------------------------------------------------ 5. + 7. REST
@pytest.fixture
def rest(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fea_core.rest import create_app
    app = create_app(base_dir=tmp_path / "rest", gui_launcher=lambda p: 1)
    c = TestClient(app, base_url="http://127.0.0.1:8765")
    c.sess = app.state.session
    return c


def test_rest_files_allow_list(rest, monkeypatch):
    from fea_core import ipc
    from fea_core.mcp.session import WORKDIR_MARK
    base = Path(rest.sess.base_dir)
    (base / "ipc").mkdir(parents=True)
    (base / "ipc" / "x.token").write_text("KEY")
    (base / "active_session.json").write_text("{}")
    (base / "other").mkdir()
    (base / "other" / "f.txt").write_text("other")
    (base / "s_20260101").mkdir()
    (base / "s_20260101" / WORKDIR_MARK).write_text("")
    (base / "s_20260101" / "img.png").write_text("png")
    (base / "imports").mkdir()
    (base / "imports" / "a.step").write_text("step")
    monkeypatch.setattr(ipc, "token_path", lambda name=None: base / "ipc" / "x.token")
    for p in ("ipc/x.token", "active_session.json", "other/f.txt"):
        assert rest.get("/api/files/" + p).status_code == 403, p
    assert rest.get("/api/files/s_20260101/img.png").text == "png"
    assert rest.get("/api/files/imports/a.step").text == "step"


def test_rest_import_is_streamed(tmp_path):
    from fea_core.rest.imports import save_import
    from fea_core.mcp.session import FeaError
    data = b"ISO-10303-21;\n" + b"x" * (3 << 20)
    step, _ = save_import(tmp_path, "a.step", io.BytesIO(data), None)
    assert step.read_bytes() == data
    with pytest.raises(FeaError):
        save_import(tmp_path, "b.step", io.BytesIO(data), None, limit=1 << 20)
    assert [p.name for p in (tmp_path / "imports").iterdir()] == [step.name]     # the partial file was deleted
    with pytest.raises(FeaError):
        save_import(tmp_path, "c.step", io.BytesIO(b"not a step"), None)


# ------------------------------------------------------------------ 6. MCP
def test_save_does_not_overwrite_foreign_file(tmp_path):
    from fea_core.mcp.tools_study import _is_own_study_file
    foreign = tmp_path / "notes.json"
    foreign.write_text(json.dumps({"todo": [1, 2]}))
    assert not _is_own_study_file(foreign)
    own = tmp_path / "s.json"
    own.write_text(json.dumps({"format": "fea-studio-study", "name": "x"}))
    assert _is_own_study_file(own)
    zf = tmp_path / "p.fea"
    with zipfile.ZipFile(zf, "w") as z:
        z.writestr("study.json", "{}")
    assert _is_own_study_file(zf)
    other = tmp_path / "q.fea"
    other.write_bytes(b"PK\x03\x04 not really")
    assert not _is_own_study_file(other)
    (tmp_path / "empty.fea").write_bytes(b"")
    assert _is_own_study_file(tmp_path / "empty.fea")


def test_user_file_size_limit(tmp_path):
    from fea_core.mcp.session import FeaError, user_file
    p = tmp_path / "a.csv"
    p.write_bytes(b"0" * 2048)
    assert user_file(p, 4096) == p
    with pytest.raises(FeaError):
        user_file(p, 1024)
    with pytest.raises(FeaError):
        user_file(tmp_path / "none.csv")


def test_instructions_say_names_are_data():
    from fea_core.mcp.server import INSTRUCTIONS
    assert "data, not instructions" in INSTRUCTIONS


# ------------------------------------------------------------------ 7. .fea limits, result arrays
def test_project_ratio_checked_for_mid_size_members(tmp_path):
    from fea_core import project
    from fea_core.study import StudyFormatError
    p = tmp_path / "bomb.fea"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("study.json", b"\0" * (20 << 20))           # 20 MB of zeros → ratio ~1000+
    with zipfile.ZipFile(p) as z:
        with pytest.raises(StudyFormatError):
            project._members(z)


def test_results_arrays_validated(tmp_path):
    from fea_core.mesh import MeshError
    from fea_core.results import Results, StepResult
    m = _mesh()
    r = Results(m, [StepResult("a", np.zeros((3, 3)), np.zeros((12, 6)), {})], [], {}, [])
    p = tmp_path / "results.npz"
    r.save(p)
    with pytest.raises(MeshError):
        Results.load(p)
    r = Results(m, [StepResult("a", np.zeros((12, 3)), np.zeros((12, 6)), {})], [], {}, [])
    r.save(p)
    assert len(Results.load(p).steps) == 1
