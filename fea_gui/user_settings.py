"""Stored GUI settings (QSettings: language, appearance, last folder, report, 3D mouse…) – one name for the whole program.

Older versions stored the settings under a different organisation name. At the first start the settings are taken over from the key
`<any organisation>/FEAStudio` (Windows: registry HKCU\\\\Software, elsewhere the Qt configuration folder); the old ones stay.
"""
from __future__ import annotations

import sys

from PySide6.QtCore import QSettings, QStandardPaths

ORG = APP = "FEAStudio"
_MIGRATED = "_settings_migrated"
_done = False


def app_settings() -> QSettings:
    _migrate_once()
    return QSettings(ORG, APP)


def _legacy_settings() -> QSettings | None:
    """Settings of an older version: the same application under a different organisation name (the first one found with some content)."""
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Software") as sw:
                orgs = []
                i = 0
                while True:
                    try:
                        orgs.append(winreg.EnumKey(sw, i))
                    except OSError:
                        break
                    i += 1
        except OSError:
            return None
        for org in orgs:
            if org == ORG:
                continue
            try:
                winreg.OpenKey(winreg.HKEY_CURRENT_USER, f"Software\\{org}\\{APP}").Close()
            except OSError:
                continue
            s = QSettings(org, APP)
            if s.allKeys():
                return s
        return None
    from pathlib import Path
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericConfigLocation)
    for p in sorted(Path(base).glob(f"*/{APP}.conf")) if base else []:
        if p.parent.name != ORG:
            s = QSettings(str(p), QSettings.Format.IniFormat)
            if s.allKeys():
                return s
    return None


def _migrate_once() -> None:
    global _done
    if _done:
        return
    _done = True
    new = QSettings(ORG, APP)
    if new.value(_MIGRATED) or new.allKeys():
        return
    try:
        old = _legacy_settings()
    except Exception:  # noqa: BLE001 – the settings are not essential, the program must be able to start
        old = None
    if old is not None:
        for k in old.allKeys():
            new.setValue(k, old.value(k))
    new.setValue(_MIGRATED, 1)
    new.sync()
