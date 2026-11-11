"""Program language chosen at installation (language.txt in the user settings) and switchable later in the GUI."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def user_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return tmp_path


def test_core_store_and_read(user_dirs):
    from fea_core import i18n
    assert i18n.stored_language() is None
    i18n.store_language("cs")
    assert i18n.stored_language() == "cs"
    assert i18n.language_path().read_text(encoding="utf-8").strip() == "cs"
    i18n.store_language("xx")                         # unknown language: nothing written
    assert i18n.stored_language() == "cs"
    i18n.language_path().write_text("garbage", encoding="utf-8")
    assert i18n.stored_language() is None


def _setup_module():
    sys.path.insert(0, str(ROOT / "installer"))
    import fea_setup
    return fea_setup


def test_installer_writes_the_same_file_as_the_program(user_dirs, monkeypatch):
    fs = _setup_module()
    from fea_core import i18n
    assert fs.language_path() == i18n.language_path()
    monkeypatch.setattr(fs, "QUIET", True)
    fs.pick_language(SimpleNamespace(lang="en"))
    assert i18n.stored_language() == "en"
    fs.pick_language(SimpleNamespace(lang=None))      # update without --lang: the earlier choice (or the GUI switch) stays
    assert i18n.stored_language() == "en"
    i18n.store_language("cs")                         # switch in the program
    fs.pick_language(SimpleNamespace(lang=None))
    assert i18n.stored_language() == "cs"
    fs.pick_language(SimpleNamespace(lang="en"))      # a new installation with an explicit choice wins
    assert i18n.stored_language() == "en"


def test_installer_default_without_a_choice(user_dirs, monkeypatch):
    fs = _setup_module()
    from fea_core import i18n
    monkeypatch.setattr(fs, "QUIET", True)
    monkeypatch.setattr(fs, "CS", False)
    fs.pick_language(SimpleNamespace(lang=None))      # first installation, no question possible: by the system language
    assert i18n.stored_language() == "en"


def test_gui_prefers_the_stored_choice(user_dirs, monkeypatch):
    pytest.importorskip("PySide6")
    from fea_core import i18n
    from fea_gui import i18n as gi
    monkeypatch.delenv("FEA_LANG", raising=False)
    i18n.store_language("en")
    assert gi.saved_language() == "en"
    i18n.store_language("cs")
    assert gi.saved_language() == "cs"
    monkeypatch.setenv("FEA_LANG", "en")              # tests / explicit override
    assert gi.saved_language() == "en"


def test_setup_exe_passes_the_wizard_language():
    iss = (ROOT / "installer" / "FEAStudio.iss").read_text(encoding="utf-8-sig")
    assert "ShowLanguageDialog=yes" in iss and "' --lang ' + ActiveLanguage" in iss


def test_persisted_startup_translations(tmp_path, monkeypatch):
    """Translations of the start-up texts are saved and used at the next start without loading the catalogue; another catalogue = another file."""
    from fea_core import i18n
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("FEA_I18N_CACHE", "1")
    cat = tmp_path / "cs.json"
    cat.write_text('{"exact": {"Open model": "Otevřít model"}, "templates": {}}', encoding="utf-8")
    eng = i18n._Engine(False, cat)
    assert eng.tr("Open model") == "Otevřít model" and eng.loaded
    eng.write_persisted(dict(eng.cache))
    f = eng.persist_file()
    assert f is not None and f.exists()
    fresh = i18n._Engine(False, cat)
    assert fresh.tr("Open model") == "Otevřít model" and not fresh.loaded      # from the persisted file, catalogue not loaded
    assert fresh.tr("Something else") == "Something else" and fresh.loaded    # unknown text: the catalogue is loaded
    cat.write_text('{"exact": {"Open model": "Otevři model"}, "templates": {}}', encoding="utf-8")
    assert i18n._Engine(False, cat).tr("Open model") == "Otevři model"        # changed catalogue → the old file is not used
    monkeypatch.setenv("FEA_I18N_CACHE", "0")
    assert i18n._Engine(False, cat).persist_file() is None
