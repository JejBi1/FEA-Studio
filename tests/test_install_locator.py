"""install.json (the path to the program for the CAD macros) – written by the installer and only the own record removed (a fake APPDATA)."""
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def setup(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT / "installer"))
    try:
        loader = importlib.machinery.SourceFileLoader("fea_setup_locator_test", str(ROOT / "installer" / "fea_setup.py"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
    finally:
        sys.path.remove(str(ROOT / "installer"))
    monkeypatch.setattr(mod, "IS_WIN", True)
    monkeypatch.setenv("APPDATA", str(tmp_path))
    return mod


def test_locator_written_and_removed(setup, tmp_path):
    setup.write_locator()
    p = tmp_path / "FEAStudio" / "install.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert Path(d["home"]) == ROOT and d["version"]
    setup.remove_locator()
    assert not p.exists()


def test_foreign_locator_is_kept(setup, tmp_path):
    p = tmp_path / "FEAStudio" / "install.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"home": str(tmp_path / "jina_instalace")}), encoding="utf-8")
    setup.remove_locator()
    assert p.exists()                       # a record of another installation is not deleted
