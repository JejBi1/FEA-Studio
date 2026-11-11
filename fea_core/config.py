"""Paths to external programs and default settings (Windows and Linux).

Units throughout the application: mm, N, MPa, t (tonne), s.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import memory as _memory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IS_WIN = sys.platform == "win32"
EXE = ".exe" if IS_WIN else ""           # extension of executable files

TOOLS_DIR = PROJECT_ROOT / "tools"
CCX_TOOLS = TOOLS_DIR / "calculix"
# MKL from the pip package mkl==2023.2.0: Windows venv\Library\bin\mkl_rt.2.dll, Linux venv/lib/libmkl_rt.so.2
MKL_DIR = Path(sys.prefix) / "Library" / "bin" if IS_WIN else Path(sys.prefix) / "lib"
MKL_RT_NAME = "mkl_rt.2.dll" if IS_WIN else "libmkl_rt.so.2"
# hypre (conda-forge, the installer puts it into .venv/hypre_env): Windows Library\bin\HYPRE.dll, Linux lib/libHYPRE.so
HYPRE_ENV = PROJECT_ROOT / ".venv" / "hypre_env"
HYPRE_BIN = HYPRE_ENV / "Library" / "bin" if IS_WIN else HYPRE_ENV / "lib"
HYPRE_LIB_NAME = "HYPRE.dll" if IS_WIN else "libHYPRE.so"
MPIEXEC_DIR = HYPRE_ENV / "Library" / "bin" if IS_WIN else HYPRE_ENV / "bin"   # own mpiexec of the hypre environment
# Linux: system libraries that were missing from the system and that the installer downloaded (libGLU for gmsh, xcb for Qt…)
SYS_LIBS = PROJECT_ROOT / ".venv" / "sys_libs" / "lib"


def user_config_dir() -> Path:
    """User settings (material library, print presets): APPDATA/FEAStudio | $XDG_CONFIG_HOME/FEAStudio."""
    if IS_WIN:
        base = os.environ.get("APPDATA") or str(Path.home())
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "FEAStudio"


def user_data_dir() -> Path:
    """User data (examples, documents, MCP working folders): LOCALAPPDATA/FEAStudio | $XDG_DATA_HOME/FEAStudio."""
    if IS_WIN:
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "FEAStudio"


def mkl_rt() -> Path:
    return MKL_DIR / MKL_RT_NAME


def _find_tool(names: tuple[str, ...], root: Path | None = None) -> Path | None:
    """First existing file from `names` directly in `root` or in its subfolder (unpacked archives have a subfolder)."""
    root = root or CCX_TOOLS
    if not root.is_dir():
        return None
    for name in names:
        for p in [root / name, *sorted(root.glob(f"*/{name}"))]:
            if p.is_file():
                return p
    return None


def which(name: str) -> str | None:
    """Program from PATH WITHOUT the working folder (shutil.which on Windows searches it first – a planted .exe next to
    the project would run). Empty and relative PATH entries are skipped.
    """
    dirs = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d.strip() and os.path.isabs(d.strip())]
    return shutil.which(name, path=os.pathsep.join(dirs)) if not IS_WIN else _which_win(name, dirs)


def _which_win(name: str, dirs: list[str]) -> str | None:
    exts = [""] if Path(name).suffix else [e for e in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";") if e]
    for d in dirs:
        for e in exts:
            p = Path(d.strip().strip('"')) / (name + e)
            if p.is_file():
                return str(p)
    return None


def find_ccx() -> tuple[Path, bool]:
    """(path to ccx, can do PARDISO). FEA_CCX (+ FEA_CCX_PARDISO=1) > ccx_i4 with MKL > ccx_static > ccx in PATH.
    ccx_i4 needs MKL (module "pardiso"); without it ccx_static is used (PaStiX / SPOOLES, ~40 % slower).
    """
    env = os.environ.get("FEA_CCX")
    if env:
        return Path(env), os.environ.get("FEA_CCX_PARDISO", "") == "1" or "i4" in Path(env).name
    if mkl_rt().exists():
        p = _find_tool((f"ccx_i4{EXE}",))
        if p:
            return p, True
    p = _find_tool((f"ccx_static{EXE}", f"ccx_spooles{EXE}", f"ccx_2.23{EXE}"))
    if p:
        return p, False
    for name in ("ccx_2.23", "ccx"):
        w = which(name)
        if w:
            return Path(w), False
    return CCX_TOOLS / f"ccx_static{EXE}", False       # does not exist – the runner reports a clear error


CCX_EXE, CCX_PARDISO = find_ccx()
CCX_DIR = CCX_EXE.parent


def ccx_solvers() -> dict[str, bool]:
    """Linear solvers that the found ccx can use: PARDISO only ccx_i4 + MKL; PaStiX is built into the Windows binaries (ccx_static
    and ccx_i4 – verified in the binaries of calculix_2.23_4win), the Linux one (ccx_spooles, ccx_i4) not ("PASTIX library is not
    linked"), ccx from PATH unknown → no; SPOOLES all.
    """
    name = CCX_EXE.name.lower()
    pastix = name.startswith("ccx_static") or (IS_WIN and name.startswith("ccx_i4"))
    return {"PARDISO": CCX_PARDISO, "PASTIX": pastix, "SPOOLES": True}


KNOWN_SOLVERS = ("PARDISO", "PASTIX", "SPOOLES", "ITERATIVE SCALING", "ITERATIVE CHOLESKY")


def solver_name(requested: str = "PARDISO") -> str:
    """Linear solver for `*STATIC, SOLVER=`: the requested one if ccx can do it (ccx_solvers); otherwise PARDISO → PASTIX
    (ccx_static) / SPOOLES, PASTIX → SPOOLES. The solver actually used is reported by the calculation (Results.info["ccx_solver"]).
    """
    req = str(requested).strip().upper()
    if req not in KNOWN_SOLVERS:                # value from a foreign study file → only known constants, never input
        req = "PARDISO"
    have = ccx_solvers()
    if have.get(req, True):
        return req
    if req == "PARDISO" and have["PASTIX"] and CCX_EXE.name.lower().startswith("ccx_static"):
        return "PASTIX"
    return "SPOOLES"


def _physical_cores() -> int:
    """Physical cores (SMT / Hyper-Threading does not help the linear solver): FEA_THREADS > system > half of the logical ones.
    Windows: GetLogicalProcessorInformationEx (ctypes), Linux: /proc/cpuinfo – without importing psutil (~90 ms at start).
    """
    env = os.environ.get("FEA_THREADS", "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    try:
        if IS_WIN:
            import ctypes
            k32 = ctypes.windll.kernel32
            size = ctypes.c_ulong(0)
            k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))     # 0 = RelationProcessorCore
            buf = ctypes.create_string_buffer(size.value)
            if k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
                n, off = 0, 0
                while off < size.value:
                    n += int.from_bytes(buf.raw[off:off + 4], "little") == 0       # Relationship = RelationProcessorCore
                    off += int.from_bytes(buf.raw[off + 4:off + 8], "little")      # Size
                if n:
                    return n
        else:
            per_pkg: dict[str, int] = {}                  # physical id → number of cores of the package ("cpu cores")
            phys = "0"
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("cpu cores"):
                    per_pkg.setdefault(phys, int(line.split(":")[1]))
            if per_pkg:
                return sum(per_pkg.values())
    except Exception:  # noqa: BLE001 – only a refinement of the estimate
        pass
    return max(1, (os.cpu_count() or 2) // 2)


NUM_THREADS = _physical_cores()                   # notebook 8 cores / 16 threads: 8
LARGE_DOF = 600_000     # from this number of DOF the "auto" backend chooses NGSolve + hypre instead of ccx (PARDISO)
PARDISO_MIN_CASES = 2   # from this number of load states "auto" chooses NGSolve + PARDISO (one factorisation)
EQUILIBRIUM_TOL = 1e-3# relative imbalance of reactions and loads → calculation error

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)     # helper processes without a console window (Windows)


def _prepend_path(env: dict, var: str, *dirs: Path) -> None:
    cur = env.get(var, "")
    env[var] = os.pathsep.join([str(d) for d in dirs] + ([cur] if cur else []))


def child_env(env: dict | None = None) -> dict:
    """Environment of helper processes: on Linux adds the downloaded system libraries (.venv/sys_libs) to LD_LIBRARY_PATH."""
    e = dict(os.environ if env is None else env)
    if not IS_WIN and SYS_LIBS.is_dir():
        _prepend_path(e, "LD_LIBRARY_PATH", SYS_LIBS)
    return e


def run_logged(cmd: list, log: Path, cancel=None, env: dict | None = None, timeout: float | None = None) -> int | None:
    """Starts a helper process (cwd = project root) with output to the file `log` (not a pipe – it does not block)
    and waits for the end. Returns the return code; None = cancelled (cancel); after `timeout` [s] subprocess.TimeoutExpired.
    """
    _memory.release_unused()                 # the helper gets the memory that this process does not use (fea_core.memory)
    with open(log, "w", encoding="utf-8", errors="replace") as out:
        p = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=child_env(env), stdout=out, stderr=subprocess.STDOUT,
                             creationflags=NO_WINDOW)
        t0 = time.monotonic()
        while True:
            try:                             # returns as soon as the process ends (no fixed sleep after it)
                p.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel is not None and cancel.is_set():
                p.kill()
                p.wait()
                return None
            if timeout is not None and time.monotonic() - t0 > timeout:
                p.kill()
                p.wait()
                raise subprocess.TimeoutExpired(p.args, timeout)
    return p.returncode


def ccx_env(deterministic: bool | None = None) -> dict:
    """Environment for the ccx process (MKL libraries in PATH / LD_LIBRARY_PATH, number of threads).

    WARNING (verified 10. 10. 2026 on V1):
      - MKL 2024.2 + ccx_i4 + PARDISO with >= 8 threads gives a SILENTLY WRONG result
        (reaction -136.8 N instead of 100 N). MKL 2022.2 / 2023.2 / 2024.0 are fine
        → the venv has mkl==2023.2.0. That is why the equilibrium is checked after every calculation.
      - CCX_NPROC_EQUATION_SOLVER > 1 breaks SPOOLES → do not set.
      - deterministic: switches on conditional reproducibility of MKL (MKL_CBWR) and fixed
        OpenMP scheduling (OMP_DYNAMIC=FALSE, OMP_SCHEDULE=static, KMP_DETERMINISTIC_REDUCTION=yes).
        If not given, it follows the environment variable FEA_CCX_DETERMINISTIC.
    """
    env = child_env()
    _prepend_path(env, "PATH" if IS_WIN else "LD_LIBRARY_PATH", MKL_DIR, *([CCX_DIR] if not IS_WIN else []))
    n = str(NUM_THREADS)
    env.setdefault("OMP_NUM_THREADS", n)
    env.setdefault("MKL_NUM_THREADS", n)
    if not IS_WIN:
        env.setdefault("MKL_THREADING_LAYER", "GNU")       # ccx_i4 (gfortran) has libgomp in the process – MKL then picks the GNU layer
    env.pop("CCX_NPROC_EQUATION_SOLVER", None)
    env.setdefault("CCX_NPROC_STIFFNESS", n)
    env.setdefault("CCX_NPROC_RESULTS", n)

    if deterministic is None:
        deterministic = os.environ.get("FEA_CCX_DETERMINISTIC", "").strip().lower() in ("1", "true", "yes", "on")

    if deterministic:
        env.setdefault("MKL_CBWR", "AUTO,STRICT")
        env.setdefault("OMP_DYNAMIC", "FALSE")
        env.setdefault("OMP_SCHEDULE", "static")
        env.setdefault("KMP_DETERMINISTIC_REDUCTION", "yes")

    return env
