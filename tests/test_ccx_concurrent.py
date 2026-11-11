"""Independent ccx jobs side by side (fea_core.ccx.concurrent, parallel.budget) – without ccx: fake tasks."""
import threading
import time

import pytest

from fea_core.ccx import parallel
from fea_core.ccx.concurrent import run_tasks
from fea_core.ccx.runner import CcxError


def _task(k, delay, log, fail=False):
    def t(cancel, on_line):
        log.append(("start", k, parallel.budget_threads()))
        end = time.time() + delay
        while time.time() < end:
            if cancel is not None and cancel.is_set():
                raise CcxError(f"task {k} killed")
            time.sleep(0.01)
        if on_line:
            on_line("Job finished")
        if fail:
            raise ValueError(f"task {k} failed")
        return k * 10
    return t


def test_results_in_task_order_and_prefixed_lines():
    log, lines = [], []
    tasks = [_task(k, d, log) for k, d in enumerate((0.3, 0.05, 0.15))]
    out = run_tasks(tasks, 1000, None, lines.append, share=(3, 4))
    assert out == [0, 10, 20]
    assert sorted(lines) == ["[step 1] Job finished", "[step 2] Job finished", "[step 3] Job finished"]
    assert all(t == 4 for _, _, t in log)              # every task in its own share of 4 threads


def test_serial_when_one_share():
    log, lines = [], []
    out = run_tasks([_task(0, 0.0, log), _task(1, 0.0, log)], 1000, None, lines.append, share=(1, 0))
    assert out == [0, 10] and lines == ["Job finished", "Job finished"]   # the old way: no prefix, caller's thread


def test_first_error_stops_the_others():
    log = []
    t0 = time.time()
    with pytest.raises(ValueError, match="task 1 failed"):
        run_tasks([_task(0, 5.0, log), _task(1, 0.05, log, fail=True)], 1000, None, None, share=(2, 3))
    assert time.time() - t0 < 2.0                      # task 0 was stopped, not waited for


def test_user_cancel_stops_all():
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()
    t0 = time.time()
    with pytest.raises(CcxError, match="canceled"):
        run_tasks([_task(0, 5.0, []), _task(1, 5.0, [])], 1000, cancel, None, share=(2, 3))
    assert time.time() - t0 < 2.0


def test_budget_plan_and_threads():
    assert parallel.budget_threads() is None
    with parallel.budget(2, 3):
        assert parallel.plan(5, 10**6) == (2, 3)            # inside the share, without a memory measurement
        assert parallel.plan(1, 10**6)[0] == 1
        assert parallel.budget_threads() == min(6, parallel.config.NUM_THREADS)
    assert parallel.budget_threads() is None
