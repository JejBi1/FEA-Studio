"""Restoration of the files shared by packages after `uv pip sync` (PySide6_Addons also deletes PySide6/__init__.py of PySide6_Essentials)."""
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def setup(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT / "installer"))
    try:
        loader = importlib.machinery.SourceFileLoader("fea_setup_shared_test", str(ROOT / "installer" / "fea_setup.py"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
    finally:
        sys.path.remove(str(ROOT / "installer"))
    site = tmp_path / "site"
    site.mkdir()
    vpy = tmp_path / "python.exe"
    vpy.write_text("")
    monkeypatch.setattr(mod, "VPY", vpy)
    monkeypatch.setattr(mod, "site_packages", lambda: site)
    calls = []
    monkeypatch.setattr(mod, "uv", lambda *a, **k: calls.append(a) or type("R", (), {"returncode": 0})())
    monkeypatch.setattr(mod, "info", lambda *a: None)
    return mod, site, calls


def _dist(site, name, ver, files):
    d = site / f"{name}-{ver}.dist-info"
    d.mkdir()
    (d / "RECORD").write_text("".join(f"{f},sha256=x,1\n" for f in files + [f"{d.name}/RECORD"]), encoding="utf-8")
    return d


def test_names_without_dist_info_suffix(setup):
    mod, site, _ = setup
    _dist(site, "PySide6_Essentials", "6.11.2", ["PySide6/__init__.py", "PySide6/QtCore.pyd"])
    assert set(mod.installed_records()) == {"pyside6_essentials"}


def test_package_sharing_files_with_removed_one_is_reinstalled(setup, tmp_path):
    mod, site, calls = setup
    _dist(site, "PySide6_Essentials", "6.11.2", ["PySide6/__init__.py", "PySide6/QtCore.pyd"])
    addons = _dist(site, "PySide6_Addons", "6.11.2", ["PySide6/__init__.py", "PySide6/QtWebEngineCore.pyd"])
    before = mod.installed_records()
    import shutil
    shutil.rmtree(addons)                                       # uv pip sync odebral PySide6_Addons
    resolved = tmp_path / "resolved.txt"
    resolved.write_text("numpy==1.26.4\npyside6-essentials==6.11.2\n", encoding="utf-8")
    mod.repair_shared_files(before, resolved)
    assert calls and "pyside6-essentials==6.11.2" in calls[0] and "--reinstall" in calls[0]


def test_pin_from_installed_version_when_missing_in_lock(setup, tmp_path):
    mod, site, calls = setup
    _dist(site, "PySide6_Essentials", "6.11.2", ["PySide6/__init__.py"])
    addons = _dist(site, "PySide6_Addons", "6.11.2", ["PySide6/__init__.py"])
    before = mod.installed_records()
    import shutil
    shutil.rmtree(addons)
    resolved = tmp_path / "resolved.txt"
    resolved.write_text("numpy==1.26.4\n", encoding="utf-8")
    mod.repair_shared_files(before, resolved)
    assert calls and "PySide6_Essentials==6.11.2" in calls[0]


def test_nothing_removed_nothing_reinstalled(setup, tmp_path):
    mod, site, calls = setup
    _dist(site, "PySide6_Essentials", "6.11.2", ["PySide6/__init__.py"])
    before = mod.installed_records()
    resolved = tmp_path / "resolved.txt"
    resolved.write_text("pyside6-essentials==6.11.2\n", encoding="utf-8")
    mod.repair_shared_files(before, resolved)
    assert calls == []
