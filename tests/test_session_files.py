"""Recent files and autosave helpers (fea_gui.session_files) – no window."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from fea_gui import session_files as sf  # noqa: E402
from fea_core.study import Study  # noqa: E402


def test_push_recent_dedupes_and_limits(tmp_path):
    a, b = tmp_path / "a.fea", tmp_path / "b.fea"
    lst = sf.push_recent([], a)
    lst = sf.push_recent(lst, b)
    lst = sf.push_recent(lst, str(a).upper() if sys.platform == "win32" else a)
    assert len(lst) == 2 and lst[1].lower().endswith("b.fea")
    many = [str(tmp_path / f"{i}.fea") for i in range(20)]
    assert len(sf.push_recent(many, a)) == sf.RECENT_MAX
    a.write_text("x")
    assert sf.existing([str(a), str(b), ""]) == [str(a)]
    assert sf.as_list(None) == [] and sf.as_list("x") == ["x"] and sf.as_list(["x", "y"]) == ["x", "y"]


class _Big:
    def __init__(self, n):
        self.steps = [SimpleNamespace(U=np.zeros((n, 3)), S=np.zeros((n, 6)))]


def test_results_nbytes():
    assert sf.results_nbytes(_Big(1000)) == 1000 * 9 * 8
    assert sf.results_nbytes(None) == 0


def test_autosave_write_stale_and_discard(tmp_path):
    st = Study("frame only", "")
    meta = sf.write_autosave(tmp_path, st, "", None, source=tmp_path / "orig.fea")
    assert (tmp_path / f"{os.getpid()}.fea").is_file() and meta["source"].endswith("orig.fea")
    assert sf.stale_autosaves(tmp_path) == []                  # this process is alive → not offered
    # an autosave of a process that has ended = abnormal end → offered
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    sf.write_autosave(tmp_path, st, "", None, pid=p.pid)
    js = tmp_path / f"{p.pid}.json"
    m = json.loads(js.read_text(encoding="utf-8"))
    m["created"] = 1.0                                          # creation time of the dead process (pid may be reused)
    js.write_text(json.dumps(m), encoding="utf-8")
    stale = sf.stale_autosaves(tmp_path)
    assert [s["pid"] for s in stale] == [p.pid]
    sf.discard(stale[0])
    assert not js.exists() and not (tmp_path / f"{p.pid}.fea").exists()
    sf.discard(os.getpid(), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_large_results_left_out(tmp_path, monkeypatch):
    monkeypatch.setattr(sf, "results_nbytes", lambda r: 10**9)
    saved = {}
    monkeypatch.setattr("fea_core.project.save_project", lambda path, study, step, results=None, motion=None:
                        saved.update(results=results) or path.write_bytes(b"PK"))
    meta = sf.write_autosave(tmp_path, Study("s", ""), "", object(), limit_mb=200)
    assert meta["results_left_out"] and not meta["with_results"] and saved["results"] is None
