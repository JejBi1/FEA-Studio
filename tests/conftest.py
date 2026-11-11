"""pytest setup shared by all tests.

The MCP server exposes only the `core` tool group by default (fea_core/mcp/groups.py); the tests use every tool, so they start
the server with all groups. Tests of the default / of the groups pass `groups_spec=` explicitly.

Installations with fewer program modules (CI with the `minimal` profile, fea_core/modules.json): a test that fails only because
an OPTIONAL module of the program is not installed (fea_core.modules.ModuleMissing, or an import of a package of such a module)
is reported as skipped. Real import errors (a package of an installed module, a typo) still fail.
"""
import os

import pytest

os.environ.setdefault("FEA_MCP_GROUPS", "all")
os.environ.setdefault("FEA_I18N_CACHE", "0")      # persisted start-up translations (fea_core.i18n) off – tests see the catalogue
os.environ.setdefault("FEA_MCP_NO_GUI", "1")      # MCP tools never run in an FEA Studio window open on this machine (fea_core.ipc)

@pytest.fixture(autouse=True)
def _private_user_dirs_outside_windows(tmp_path_factory, monkeypatch):
    """Outside Windows the user settings / data live in XDG folders (fea_core.config.user_config_dir); the tests isolate
    themselves by APPDATA / LOCALAPPDATA (Windows) → each test gets fresh XDG folders, the real ~/.config stays untouched."""
    if os.name != "nt":
        d = tmp_path_factory.mktemp("xdg")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(d / "config"))
        monkeypatch.setenv("XDG_DATA_HOME", str(d / "data"))


_OPTIONAL: dict[str, str] | None = None             # import name of a package → optional module that is NOT installed


def _optional_packages() -> dict[str, str]:
    global _OPTIONAL
    if _OPTIONAL is None:
        _OPTIONAL = {}
        try:
            from fea_core import modules
            for key, m in modules.all_modules().items():
                if m.required or modules.available(key):
                    continue
                names = [i.split(".")[0] for i in m.imports]
                names += [p.split("==")[0].split("[")[0].strip().lower().replace("-", "_") for p in m.pip + m.pip_nodeps]
                for n in names:
                    _OPTIONAL.setdefault(n.lower(), key)
        except Exception:  # noqa: BLE001 – without the table nothing is converted
            pass
    return _OPTIONAL


def missing_optional(exc: BaseException | None) -> str | None:
    """Key of the optional program module whose absence caused `exc` (None = a real error)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ == "ModuleMissing" and getattr(exc, "key", None):
            from fea_core import modules
            return exc.key if not modules.available(exc.key) else None
        if isinstance(exc, ModuleNotFoundError) and exc.name:
            key = _optional_packages().get(exc.name.split(".")[0].lower())
            if key:
                return key
        exc = exc.__cause__ or exc.__context__
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.failed and call.excinfo is not None:
        key = missing_optional(call.excinfo.value)
        if key:
            rep.outcome = "skipped"
            rep.longrepr = (str(item.path), (item.location[1] or 0) + 1,
                            f"Skipped: optional program module '{key}' is not installed")
