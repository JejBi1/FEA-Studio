"""Optional modules of the program: what is installed, what is missing and how to install it.

The core (module `core`) is small – STEP → mesh → CalculiX → results. The other parts (GUI, fatigue and 3D printing via Numba,
MKL PARDISO, large models, MCP, Word report…) are installed separately; the definitions are in `modules.json` (the installer reads them too).

    python -m fea_core.modules                 # overview
    python -m fea_core.modules install gui accel
    python -m fea_core.modules remove docs

Code that needs a module must not import it "blindly": `need("numba", "accel")` returns the module, or raises
`ModuleMissing` with a Czech / English message how to install it (the GUI and MCP show it as an error).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import config

DATA = Path(__file__).with_name("modules.json")


class ModuleMissing(ImportError):
    """An optional module of the program is missing (installed separately). The message contains the installation procedure."""

    def __init__(self, key: str, detail: str = ""):
        self.key = key
        m = get(key)
        title = m.title() if m else key
        msg = tr_msg(f"Missing module “{title}” ({key}). Install it: python -m fea_core.modules install {key} (or in the GUI: Tools → Program modules).")
        super().__init__(msg + (f" [{detail}]" if detail else ""))


def tr_msg(text: str) -> str:
    try:
        from .i18n import tr
        return tr(text)
    except Exception:  # noqa: BLE001 – i18n must not prevent reporting an error
        return text


@dataclass(frozen=True)
class Module:
    key: str
    titles: dict
    descs: dict
    requires: tuple = ()
    pip: tuple = ()
    pip_nodeps: tuple = ()
    imports: tuple = ()
    ext: tuple = ()
    check: str = ""
    required: bool = False
    size_mb: dict = field(default_factory=dict)

    def title(self, lang: str | None = None) -> str:
        lang = lang or _lang()
        return self.titles.get(lang) or self.titles.get("cs", self.key)

    def desc(self, lang: str | None = None) -> str:
        lang = lang or _lang()
        return self.descs.get(lang) or self.descs.get("cs", "")

    def size(self) -> int:
        return int(self.size_mb.get("win" if config.IS_WIN else "linux", 0))


def _lang() -> str:
    try:
        from . import i18n
        return i18n.language()
    except Exception:  # noqa: BLE001
        return "cs"


@lru_cache(maxsize=1)
def _load() -> tuple[dict, dict]:
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    mods = {}
    for k, d in raw["modules"].items():
        mods[k] = Module(k, d.get("title", {}), d.get("desc", {}), tuple(d.get("requires", ())), tuple(d.get("pip", ())),
                         tuple(d.get("pip_nodeps", ())), tuple(d.get("import", ())), tuple(d.get("ext", ())),
                         d.get("check", ""), bool(d.get("required", False)), d.get("size_mb", {}))
    return mods, raw.get("profiles", {})


def all_modules() -> dict[str, Module]:
    return dict(_load()[0])


def profiles() -> dict[str, list[str]]:
    return dict(_load()[1])


def get(key: str) -> Module | None:
    return _load()[0].get(key)


def _has_import(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):          # sub-package of the parent package that is missing
        return False


def _check(kind: str) -> bool:
    """Special checks of modules with an external component."""
    if kind == "pardiso":
        return config.mkl_rt().exists() and config.CCX_PARDISO
    if kind == "hypre":                        # without importing fea_core.hypre (numpy, scipy)
        from .solvers import hypre_missing
        return not hypre_missing()
    if kind == "chrono":
        from .motion.chrono_run import available as chrono_available
        return chrono_available()
    if kind == "native":                      # only the program code – switched on = in the list of modules of the installation
        return _installed("native")
    return True


def _installed(key: str) -> bool:
    """Is the module in the list of installed ones (.venv/fea_modules.json from the installer)? Without the file (development environment
    without the installer) = yes.
    """
    st = config.PROJECT_ROOT / ".venv" / "fea_modules.json"
    try:
        return key in json.loads(st.read_text(encoding="utf-8")).get("modules", [])
    except FileNotFoundError:
        return True
    except (OSError, ValueError):
        return False


def available(key: str) -> bool:
    """Is the module installed (imports and external components)? Fast – it imports nothing."""
    m = get(key)
    if m is None:
        return False
    if any(not _has_import(i) for i in m.imports):
        return False
    if "ccx_static" in m.ext and not (config.CCX_EXE.exists()):
        return False
    if "mystran" in m.ext and not (config.TOOLS_DIR / "mystran" / ("mystran" + config.EXE)).exists() and config.IS_WIN:
        return False
    return _check(m.check) if m.check else True


def status() -> dict[str, bool]:
    return {k: available(k) for k in _load()[0]}


def problem(key: str) -> str:
    """Why an installed module cannot be used: the Python packages of the module are there, but an external component or a module
    it depends on is missing ("Microsoft MPI (mpiexec) is missing"). Empty = the module is not installed at all or it is in order.
    """
    m = get(key)
    if m is None or available(key):
        return ""
    lib = (config.HYPRE_BIN / config.HYPRE_LIB_NAME).exists()
    if key == "hypre" and (_has_import("mpi4py") or lib):
        from .solvers import find_mpiexec
        miss = [] if _has_import("ngsolve") else ["module ngsolve"]
        miss += [] if lib else ["hypre library (.venv/hypre_env)"]
        miss += [] if _has_import("mpi4py") else ["package mpi4py"]
        if find_mpiexec() is None:
            miss.append("Microsoft MPI (mpiexec)" if config.IS_WIN else "MPI (mpiexec in the hypre environment)")
    elif key == "pardiso" and config.mkl_rt().exists():
        miss = [] if config.CCX_PARDISO else ["CalculiX with PARDISO (ccx_i4)"]
    elif m.imports and all(_has_import(i) for i in m.imports):
        miss = ["the CalculiX program (ccx)"] if "ccx_static" in m.ext and not config.CCX_EXE.exists() else []
        miss += [f"module {r}" for r in m.requires if not available(r)]
    else:
        return ""
    return ("missing " + ", ".join(miss)) if miss else ""


def missing(key: str, detail: str = "") -> ModuleMissing:
    return ModuleMissing(key, detail)


def need(import_name: str, key: str):
    """Import of an optional dependency with a clear error: `numba = need("numba", "accel")`."""
    try:
        return importlib.import_module(import_name)
    except ImportError as e:
        raise ModuleMissing(key, f"{type(e).__name__}: {e}") from e


def require(key: str) -> None:
    if not available(key):
        raise ModuleMissing(key)


def closure(keys: list[str]) -> list[str]:
    """Modules including the dependencies, in installation order."""
    mods = _load()[0]
    out: list[str] = []

    def add(k: str) -> None:
        if k not in mods:
            raise KeyError(f"unknown module: {k}")
        for r in mods[k].requires:
            add(r)
        if k not in out:
            out.append(k)

    for k in keys:
        add(k)
    return out


# ------------------------------------------------------------------ command line
def _setup_script() -> Path:
    return config.PROJECT_ROOT / "installer" / "fea_setup.py"


def run_setup(args: list[str]) -> int:
    """Runs the installer (installer/fea_setup.py) with this Python – installation / removal of modules."""
    script = _setup_script()
    if not script.exists():
        print("Installer (installer/fea_setup.py) not found – install the modules from the original package.")
        return 2
    return subprocess.call([sys.executable, str(script), *args], cwd=config.PROJECT_ROOT)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "list"
    if cmd in ("install", "remove", "slim", "doctor", "mpi"):          # mpi = Microsoft MPI only (fea_setup.py mpi)
        return run_setup([cmd, *argv[1:]])
    if cmd not in ("list", "ls"):
        print(__doc__)
        return 2
    st = status()
    print("FEA Studio modules (√ installed, – missing):\n")
    for k, m in _load()[0].items():
        print(f"  {'√' if st[k] else '–'} {k:8s} {m.title():66.66s} ~{m.size():4d} MB")
        print(f"             {m.desc()}")
    print("\nInstall: python -m fea_core.modules install <module> [<module> …]   (profiles: --profile standard|full)")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
