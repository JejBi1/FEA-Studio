"""Independent ccx jobs of one calculation side by side (natural frequencies + buckling, Campbell speeds, print variants…).

Each job gets a share of the machine (`parallel.plan` over the number of jobs: free memory by the ccx estimate, ≥ 3
threads per job – PARDISO scales badly beyond a few threads, several smaller jobs are faster, see ccx/parallel.py).
The results come back in task order; the first error stops the other jobs (their ccx is killed) and is raised; the user's
cancel stops all of them. When the plan gives one job (little memory, few cores, FEA_CCX_PARALLEL=0) the tasks run one after
another in the calling thread exactly as before.
"""
from __future__ import annotations

import threading
from typing import Callable

from . import parallel
from .runner import CcxError


def run_tasks(tasks: list[Callable], ndof: int, cancel: threading.Event | None = None, on_line=None,
              share: tuple[int, int] | None = None) -> list:
    """tasks = [fn(cancel, on_line) → result]; on_line of task k gets the prefix "[step k] " when they run side by side
    (progress.CcxLines counts parallel jobs by it). share = (jobs, threads) instead of parallel.plan.
    """
    n = len(tasks)
    if share is not None:
        workers, threads = share
    else:
        workers, threads = parallel.plan(n, ndof) if n > 1 and parallel.enabled() else (1, 0)
    if workers < 2 or n < 2:
        return [t(cancel, on_line) for t in tasks]
    stop = threading.Event()
    results: list = [None] * n
    errors: list[BaseException] = []
    lock = threading.Lock()
    sem = threading.Semaphore(workers)

    def watch():                                    # the user's cancel → stop all jobs
        while not stop.is_set():
            if cancel is not None and cancel.wait(0.2):
                stop.set()
                return
            if cancel is None:
                stop.wait(0.5)

    def go(k: int):
        with sem:
            if stop.is_set():
                return
            line = (lambda s, p=f"[step {k + 1}] ": on_line(p + s)) if on_line else None
            try:
                with parallel.budget(1, threads):    # run_ccx inside the task: `threads` threads
                    results[k] = tasks[k](stop, line)
            except BaseException as e:  # noqa: BLE001 – raised below in the calling thread
                with lock:
                    errors.append(e)
                stop.set()

    w = threading.Thread(target=watch, daemon=True)
    w.start()
    th = [threading.Thread(target=go, args=(k,), daemon=True, name=f"fea-ccx-task-{k + 1}") for k in range(n)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    stop.set()
    if cancel is not None and cancel.is_set():
        raise CcxError("Analysis canceled by user")
    if errors:
        raise errors[0]
    return results
