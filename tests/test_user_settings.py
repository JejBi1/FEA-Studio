"""Takeover of the GUI settings from an older version (a different organisation name in QSettings) – fea_gui/user_settings.py."""
import sys

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402

import fea_gui.user_settings as us  # noqa: E402

APP = "FEAStudioPytestSettings"
LEGACY = "FEAStudioPytestLegacyOrg"
NEW = "FEAStudioPytestNewOrg"


def _cleanup():
    for org in (LEGACY, NEW):
        s = QSettings(org, APP)
        s.clear()
        s.sync()
    if sys.platform == "win32":
        import winreg
        for org in (LEGACY, NEW):
            for key in (f"Software\\{org}\\{APP}", f"Software\\{org}"):
                try:
                    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)
                except OSError:
                    pass


@pytest.fixture
def names(monkeypatch):
    _cleanup()
    monkeypatch.setattr(us, "ORG", NEW)
    monkeypatch.setattr(us, "APP", APP)
    monkeypatch.setattr(us, "_done", False)
    yield
    _cleanup()


def test_legacy_settings_are_copied_once(names):
    old = QSettings(LEGACY, APP)
    old.setValue("language", "en")
    old.setValue("theme", "dark")
    old.sync()
    s = us.app_settings()
    assert s.value("language") == "en" and s.value("theme") == "dark"
    # another change of the old version is no longer taken over (the takeover only once), the old setting stays
    old.setValue("language", "cs")
    old.sync()
    us._done = False
    assert us.app_settings().value("language") == "en"
    assert QSettings(LEGACY, APP).value("theme") == "dark"


def test_no_legacy_settings(names):
    s = us.app_settings()
    assert s.value("language") is None and s.value(us._MIGRATED) is not None
