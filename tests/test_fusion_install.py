"""Fusion add-in: installation into AddIns by the installer (a fake APPDATA) and the default path to FEA Studio."""
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FUSION = ROOT / "adapters" / "fusion" / "FEAStudio" / "FEAStudio.py"


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


@pytest.fixture()
def setup(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT / "installer"))
    try:
        mod = _load("fea_setup_fusion_test", ROOT / "installer" / "fea_setup.py")
    finally:
        sys.path.remove(str(ROOT / "installer"))
    monkeypatch.setattr(mod, "IS_WIN", True)
    monkeypatch.setenv("APPDATA", str(tmp_path))
    return mod


def test_addin_finds_fea_studio_from_project_folder(monkeypatch):
    fus = _load("fea_fusion_home", FUSION)
    for e in ("FEA_STUDIO_HOME", "FEA_REST_URL", "FEA_REST_TOKEN"):
        monkeypatch.delenv(e, raising=False)
    s = fus.load_settings(str(ROOT / "neexistuje.json"))
    assert Path(s["fea_studio_home"]) == ROOT


def test_install_without_fusion_fails(setup):
    assert setup.fusion_dir() is None
    with pytest.raises(RuntimeError):
        setup.install_fusion_addin()


def test_install_keeps_user_settings_and_removes(setup, tmp_path):
    (tmp_path / "Autodesk" / "Autodesk Fusion 360").mkdir(parents=True)
    dest = setup.install_fusion_addin()
    assert dest == tmp_path / "Autodesk" / "Autodesk Fusion 360" / "API" / "AddIns" / "FEAStudio"
    assert (dest / "FEAStudio.py").is_file() and (dest / "FEAStudio.manifest").is_file()
    assert (dest / "resources" / "SendToFEA" / "32x32.png").is_file()
    assert not (dest / "__pycache__").exists()
    cfg = json.loads((dest / "settings.json").read_text(encoding="utf-8"))
    assert Path(cfg["fea_studio_home"]) == ROOT
    # the user's own settings survive the new installation
    cfg["token"] = "tajne"
    cfg["fea_studio_home"] = "C:/jinde"
    (dest / "settings.json").write_text(json.dumps(cfg), encoding="utf-8")
    setup.install_fusion_addin()
    cfg = json.loads((dest / "settings.json").read_text(encoding="utf-8"))
    assert cfg["token"] == "tajne" and Path(cfg["fea_studio_home"]) == ROOT
    # the add-in loads the settings from a copy
    fus = _load("fea_fusion_copy", dest / "FEAStudio.py")
    assert fus.load_settings()["token"] == "tajne"
    setup.remove_fusion_addin()
    assert not dest.exists()
