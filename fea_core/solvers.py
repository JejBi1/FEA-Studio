"""Availability of solvers: what is installed, what is missing and how to add it (without Qt, only light imports – nothing is loaded).

Keys such as `Study.solver.backend`: ccx | ngs_pardiso | ngs_hypre ("auto" chooses only from the available ones, see
`solve.choose_backend`). The GUI greys out the unavailable items according to `backend_status()`, the readiness check
(`solve.check_study` → `check_backend`) reports a manually chosen unavailable solver BEFORE meshing, MCP and REST list the state
in `fea_study_status` and `/api/status`.

Components (verified on an installation by modules, see fea_core/modules.json):
  ccx          CalculiX program (module core; ccx_i4 with PARDISO only with the module pardiso = Intel MKL)
  ngs_pardiso  module ngsolve (pip ngsolve) + Intel MKL (module pardiso)
  ngs_hypre    module ngsolve + module hypre (mpi4py + hypre library in .venv/hypre_env) + MPI: Windows Microsoft MPI
               (mpiexec, installed separately with administrator rights), Linux MPICH from the hypre environment
"""
from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

from . import config

KEYS = ("ccx", "ngs_pardiso", "ngs_hypre")
TITLES = {"auto": "Automatic", "ccx": "CalculiX", "ngs_pardiso": "NGSolve + PARDISO", "ngs_hypre": "NGSolve + hypre"}

# missing components: (description for the user, program module that installs it; "" = MS-MPI separately)
NGSOLVE = ("module ngsolve", "ngsolve")
MKL = ("module pardiso (Intel MKL)", "pardiso")
HYPRE = ("module hypre", "hypre")
MSMPI = ("Microsoft MPI (mpiexec)", "")
MPI_LINUX = ("MPI (mpiexec in the hypre environment)", "hypre")
CCX = ("the CalculiX program (ccx)", "core")


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):          # sub-package of the missing package
        return False


def find_mpiexec() -> str | None:
    """mpiexec: on Linux preferably from the hypre environment (the same MPI as the HYPRE library), otherwise from PATH; on Windows also
    from the MS-MPI folder (MSMPI_BIN, ProgramFiles/Microsoft MPI/Bin) – a program started before the installation of MS-MPI does not have a new PATH.
    The mpiexec from the conda hypre environment is NOT used on Windows (not verified).
    """
    own = config.MPIEXEC_DIR / ("mpiexec" + config.EXE)
    if not config.IS_WIN and own.exists():
        return str(own)
    w = config.which("mpiexec")
    if w:
        return w
    if config.IS_WIN:
        pf = os.environ.get("ProgramFiles") or os.environ.get("ProgramW6432") or "C:\\Program Files"
        for d in (os.environ.get("MSMPI_BIN", ""), str(Path(pf) / "Microsoft MPI" / "Bin")):
            if d and (Path(d) / "mpiexec.exe").is_file():
                return str(Path(d) / "mpiexec.exe")
    return None


def _missing(key: str) -> list[tuple[str, str]]:
    if key == "ccx":
        return [] if config.CCX_EXE.exists() else [CCX]
    out = [] if _has("ngsolve") else [NGSOLVE]
    if key == "ngs_pardiso":
        if not config.mkl_rt().exists():
            out.append(MKL)
    elif key == "ngs_hypre":
        if not _has("mpi4py") or not (config.HYPRE_BIN / config.HYPRE_LIB_NAME).exists():
            out.append(HYPRE)
        if find_mpiexec() is None:
            out.append(MSMPI if config.IS_WIN else MPI_LINUX)
    return out


def hypre_missing() -> list[str]:
    """Missing components of the NGSolve + hypre solver (empty = usable; `hypre.available()`)."""
    return [m[0] for m in _missing("ngs_hypre")]


def _fix(miss: list[tuple[str, str]]) -> str:
    """How to install the missing components (one sentence)."""
    if CCX in miss:
        return ("Run the installation again (python installer/fea_setup.py install) or set the variable FEA_CCX "
                "to the path of the ccx program.")
    mods = sorted({m for _, m in miss if m})
    parts = []
    if mods:
        parts.append(f"Tools → Program modules… (or python -m fea_core.modules install {' '.join(mods)})")
    if MSMPI in miss:
        parts.append("Microsoft MPI: Tools → Program modules… → hypre, or python installer/fea_setup.py mpi "
                     "(administrator rights)")
    if not parts:
        return "Run the repair installation (python installer/fea_setup.py install)."
    return "Install the missing parts: " + "; ".join(parts) + ". Then restart FEA Studio."


def install_hint() -> str:
    """One hint how to install all the missing solvers (union of the missing components), "" = nothing is missing."""
    miss: list[tuple[str, str]] = []
    for k in KEYS:
        for m in _missing(k):
            if m not in miss:
                miss.append(m)
    return _fix(miss) if miss else ""


def backend_status() -> dict[str, dict]:
    """{key: {"ok", "title", "missing": [component descriptions], "fix": how to install}} for ccx, ngs_pardiso, ngs_hypre."""
    out = {}
    for k in KEYS:
        miss = _missing(k)
        out[k] = {"ok": not miss, "title": TITLES[k], "missing": [m[0] for m in miss], "fix": _fix(miss) if miss else ""}
    return out


def linear_solvers() -> dict[str, bool]:
    """Linear solvers inside CalculiX (`Study.solver.solver`) that the found ccx can do (config.ccx_solvers)."""
    return config.ccx_solvers()


def linear_solver_note(name: str) -> str:
    """Why the CalculiX linear solver is not possible (empty = possible)."""
    if linear_solvers().get(str(name).upper(), True):
        return ""
    if str(name).upper() == "PARDISO":
        return "module pardiso is missing – Intel MKL, CalculiX ccx_i4"
    return "this CalculiX version does not contain it"


def backend_error(key: str) -> str | None:
    """Clear error when the solver `key` is not installed (None = available or "auto")."""
    if key == "auto" or key not in KEYS:
        return None
    miss = _missing(key)
    if not miss:
        return None
    what = ", ".join(m[0] for m in miss)
    if key == "ccx":
        return f"The “CalculiX” solver is not installed – missing {what}. {_fix(miss)}"
    return (f"The “{TITLES[key]}” solver is not installed – missing {what}. {_fix(miss)} Or choose the Automatic solver.")


def module_error(key: str, module: str) -> str:
    """Error of the helper process of the solver `key` (NGSolve, hypre) which lacks the Python module `module`
    (ModuleNotFoundError in the output of the process).
    """
    top = str(module).split(".")[0]
    comp = {"ngsolve": NGSOLVE, "netgen": NGSOLVE, "pyngcore": NGSOLVE, "mpi4py": HYPRE}.get(
        top, (f"Python module {top}", ""))
    return (f"The “{TITLES.get(key, key)}” solver cannot run – missing {comp[0]}. {_fix([comp])} Or choose the Automatic solver.")


def backend_ignored(study) -> bool:
    """The choice of the solver is not used: nonlinear calculation (always ccx), 2D section, shells, frames alone (own solver),
    a study without load states (heat, natural frequencies, dynamics – always ccx).
    """
    frame_only = study.frame.active and not str(study.geometry or "").strip()      # = beams.run.frame_only
    return (not study.load_cases or study.twod.enabled or study.shell.enabled or frame_only
            or study.is_nonlinear())


def check_backend(study, backend: str | None = None) -> list[str]:
    """Errors of a manually chosen solver (`backend` = override of the study setting, MCP): not installed / cannot do the study.
    Before meshing – so that one does not wait for the mesh needlessly. "auto" and a study where the choice is not used have no errors.
    """
    b = backend or study.solver.backend or "auto"
    if b == "auto" or backend_ignored(study):
        return []
    if b not in KEYS:
        return [f"Unknown solver “{b}” – choose Automatic, CalculiX, NGSolve + PARDISO or NGSolve + hypre."]
    err = backend_error(b)
    errs = [err] if err else []
    if b != "ccx":
        from .solve_large import unsupported
        why = unsupported(study)
        if why:
            errs.append(f"The “{TITLES[b]}” solver cannot handle this study: {', '.join(why)} – choose Automatic or CalculiX.")
    return errs


def summary() -> dict:
    """Brief state for MCP / REST: {key: "ok" | "missing …"} + the linear solvers of CalculiX."""
    st = backend_status()
    d = {k: ("ok" if v["ok"] else "missing " + ", ".join(v["missing"])) for k, v in st.items()}
    d["ccx_linear_solvers"] = [k for k, ok in linear_solvers().items() if ok]
    fixes = [v["fix"] for v in st.values() if v["fix"]]
    if fixes:
        d["fix"] = fixes[0] if len(set(fixes)) == 1 else " | ".join(dict.fromkeys(fixes))
    return d


def table() -> list[str]:
    """Overview of the solvers for `python -m fea_core.solvers` (installer: fea_setup.py doctor)."""
    lines = ["Solvers (√ available, – missing):"]
    for k, v in backend_status().items():
        lines.append((f"  {'√' if v['ok'] else '–'} {k:12s} {v['title']:18s}"
                      + ("" if v["ok"] else f" missing: {', '.join(v['missing'])}")).rstrip())
        if v["fix"]:
            lines.append(f"      {v['fix']}")
    have = linear_solvers()
    lines.append("  CalculiX system solver: " + ", ".join(f"{n} {'√' if ok else '–'}" for n, ok in have.items())
                 + f"  ({config.CCX_EXE.name})")
    mpi = find_mpiexec()
    lines.append(f"  mpiexec: {mpi or '–'}")
    return lines


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n".join(table()))
