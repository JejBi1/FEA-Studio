"""Language of the installer's output (Czech / English) – shared by fea_setup.py, upgrade.py, migrations.py and slim.py.
Only the standard library. Czech if FEA_LANG / the national environment / LANG is Czech, otherwise English.
"""
from __future__ import annotations

import locale
import os


def _cs() -> bool:
    lang = os.environ.get("FEA_LANG") or (locale.getlocale()[0] or "") or os.environ.get("LANG", "")
    return lang.lower().startswith(("cs", "czech"))


CS = _cs()


def T(cs: str, en: str) -> str:
    return cs if CS else en
