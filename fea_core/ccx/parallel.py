"""Parallel calculation of load states in ccx: each *STEP as a separate job over a common model.
Linear statics = steps independent of each other (constraints and loads are written again in every step, OP=NEW), but ccx
in every *STEP assembles and factorises the matrix again, and PARDISO factorisation scales badly with 8 threads. Several
jobs with fewer threads at once are therefore faster (verified 13. 10. 2026, scratch/opt_mech/split.py: example 2,
210 k DOF, 5 steps: 1 job × 8 threads 93.7 s → 5 jobs × 3 threads 56.8 s, U and S equal to 1e-8).

Job `job.inp` is split into `job_model.inc` (everything before the first *STEP) and `job_s{k}.inp` (*INCLUDE + k-th step);
the results of each job are read separately (.frd, .dat) and joined in step order (`read_frd_steps`).
The number of parallel jobs is limited by free memory (estimate of ccx PARDISO memory from the DOF) and by the number of cores.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from .. import config
from .runner import CcxError, CcxRun, run_ccx

MIN_THREADS = 3          # with fewer threads per job PARDISO slows down more than the concurrency gains (5 × 2 = 70 s)


def enabled() -> bool:
    """Parallel steps can be switched off by the environment variable FEA_CCX_PARALLEL=0 (debugging, comparison)."""
    import os
    return os.environ.get("FEA_CCX_PARALLEL", "1").strip() not in ("0", "false", "ne")


def ccx_memory_gb(ndof: int) -> float:
    """Estimate of peak ccx memory (PARDISO, tet10) [GB]: 214 k DOF 1.4 GB, 706 k DOF 6.8 GB (CLAUDE.md)."""
    return 0.2 + 1.2 * (max(ndof, 1) / 214e3) ** 1.32


def plan(n_steps: int, ndof: int, free_gb: float | None = None, cores: int | None = None) -> tuple[int, int]:
    """(number of parallel jobs, threads per job). (1, NUM_THREADS) = count the old way with one job."""
    cores = cores or config.NUM_THREADS
    if n_steps < 2 or cores < 2 * MIN_THREADS:
        return 1, cores
    if free_gb is None:
        try:
            import psutil
            free_gb = psutil.virtual_memory().available / 2**30
        except Exception:  # noqa: BLE001
            return 1, cores
    by_mem = int((0.8 * free_gb) // ccx_memory_gb(ndof))
    n = max(1, min(n_steps, by_mem, (2 * cores) // MIN_THREADS))     # even with SMT: 5 × 3 threads on 8 cores is the fastest
    if n < 2:
        return 1, cores
    return n, max(MIN_THREADS, -(-cores // n))


def split_steps(text: str) -> tuple[str, list[str]]:
    """(model – everything before the first *STEP, [text of step 1, …]) from the text of .inp. Searches only lines starting with *STEP."""
    lines = text.splitlines(True)
    starts = [i for i, ln in enumerate(lines) if ln[:5].upper() == "*STEP"]
    if not starts:
        raise ValueError("The .inp job has no *STEP")
    ends = starts[1:] + [len(lines)]
    return "".join(lines[:starts[0]]), ["".join(lines[a:b]) for a, b in zip(starts, ends)]


@dataclass
class ParallelRun:
    jobs: list[str]               # job names in step order (job_s0, job_s1, …)
    runs: list[CcxRun]
    workers: int
    threads: int


def run_steps(workdir: str | Path, job: str, ndof: int, exe: Path | None = None, on_line=None,
              cancel: threading.Event | None = None, workers: int | None = None,
              threads: int | None = None) -> ParallelRun:
    """Computes the steps of the job `job.inp` in parallel. Error of any job → CcxError (the others are cancelled)."""
    workdir = Path(workdir)
    text = (workdir / f"{job}.inp").read_text(encoding="latin-1")
    model, steps = split_steps(text)
    if workers is None or threads is None:
        w, t = plan(len(steps), ndof)
        workers, threads = workers or w, threads or t
    (workdir / f"{job}_model.inc").write_text(model, encoding="latin-1")
    jobs = []
    for k, body in enumerate(steps):
        name = f"{job}_s{k}"
        (workdir / f"{name}.inp").write_text(f"*INCLUDE, INPUT={job}_model.inc\n" + body, encoding="latin-1")
        jobs.append(name)
    stop = threading.Event()
    runs: list[CcxRun | None] = [None] * len(jobs)
    errs: list[str] = []
    sem = threading.Semaphore(workers)
    lock = threading.Lock()

    def watch():                                    # cancellation by the user → stop all jobs
        while not stop.is_set():
            if cancel is not None and cancel.wait(0.2):
                stop.set()
                return
            if cancel is None:
                stop.wait(0.5)

    def go(k: int, name: str):
        with sem:
            if stop.is_set():
                return
            try:
                prefix = f"[step {k + 1}] "
                r = run_ccx(workdir, name, exe=exe, cancel=stop, threads=threads,
                            on_line=(lambda s: on_line(prefix + s)) if on_line else None)
            except CcxError as e:
                with lock:
                    errs.append(str(e))
                stop.set()
                return
            runs[k] = r
            if not r.finished or r.errors:
                with lock:
                    errs.append(f"step {k + 1}: " + "; ".join(r.errors or [f"return code {r.returncode}"]))
                stop.set()

    w = threading.Thread(target=watch, daemon=True)
    w.start()
    th = [threading.Thread(target=go, args=(k, n), daemon=True) for k, n in enumerate(jobs)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if cancel is not None and cancel.is_set():
        stop.set()
        raise CcxError("Analysis canceled by user")
    stop.set()
    if errs:
        raise CcxError("CalculiX ended with an error:\n  " + "\n  ".join(errs))
    return ParallelRun(jobs, [r for r in runs if r is not None], workers, threads)


def read_frd_steps(workdir: str | Path, jobs: list[str]):
    """Joins the .frd of the jobs step by step into one FrdFile (step k+1 = job k), as if it were one job."""
    from .frd import read_frd
    out = None
    for k, name in enumerate(jobs):
        f = read_frd(Path(workdir) / f"{name}.frd")
        last = f.steps()[-1]
        blocks = [b for b in f.blocks if b.step == last]
        for b in blocks:
            b.step = k + 1
        if out is None:
            f.blocks = blocks
            out = f
        else:
            out.blocks.extend(blocks)
    return out
