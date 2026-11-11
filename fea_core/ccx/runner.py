"""Running CalculiX as a separate process (progress, cancellation, errors)."""
from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .. import config, dllguard, memory


class CcxError(RuntimeError):
    pass


@dataclass
class CcxRun:
    returncode: int
    finished: bool            # ccx vypsal "Job finished"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    log_path: Path | None = None


def _env(threads: int | None, deterministic: bool | None = None) -> dict:
    env = config.ccx_env(deterministic=deterministic)
    if threads:
        n = str(max(1, int(threads)))
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CCX_NPROC_STIFFNESS", "CCX_NPROC_RESULTS"):
            env[k] = n
    return env


def run_ccx(workdir: str | Path, jobname: str, exe: Path | None = None,
            on_line: Callable[[str], None] | None = None,
            cancel: threading.Event | None = None, timeout: float | None = None,
            threads: int | None = None, deterministic: bool | None = None) -> CcxRun:
    """Runs `ccx -i jobname` in workdir. Output goes to jobname.log and to on_line.
    threads = number of ccx/MKL threads (None = config.NUM_THREADS; parallel jobs of ccx.parallel get fewer).
    deterministic = reproducible MKL/OpenMP summation (nonlinear jobs; linear ~25 % slower – see config.ccx_env).
    """
    workdir = Path(workdir)
    exe = Path(exe or config.CCX_EXE)
    if not exe.exists():
        raise CcxError(f"CalculiX not found: {exe}")
    if not threads:                          # share of the machine of this thread (independent jobs side by side)
        from .parallel import budget_threads
        threads = budget_threads()
    dllguard.check_workdir(workdir, "CalculiX")  # ccx runs with cwd = workdir: no planted DLLs (fea_core.dllguard)
    log_path = workdir / f"{jobname}.log"
    memory.release_unused()                  # ccx gets the memory that this process does not use (fea_core.memory)
    proc = subprocess.Popen(
        [str(exe), "-i", jobname], cwd=workdir, env=_env(threads, deterministic),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="latin-1", creationflags=config.NO_WINDOW,
    )
    errors, warnings, finished = [], [], False
    killer = None
    if cancel is not None:
        def watch():
            while proc.poll() is None:
                if cancel.wait(0.2):
                    proc.kill()
                    return
        killer = threading.Thread(target=watch, daemon=True)
        killer.start()
    with open(log_path, "w", encoding="utf-8") as log:
        for line in proc.stdout:
            log.write(line)
            s = line.strip()
            if "*ERROR" in s:
                errors.append(s)
            elif "*WARNING" in s:
                warnings.append(s)
            elif s.startswith("Job finished"):
                finished = True
            if on_line:
                on_line(line.rstrip("\n"))
    rc = proc.wait(timeout=timeout)
    if cancel is not None and cancel.is_set():
        raise CcxError("Analysis canceled by user")
    return CcxRun(rc, finished, errors, warnings, log_path)
