import time
import threading
import pytest
from pathlib import Path
from fea_core.job_queue import JobQueue, Job
from fea_core.study import Study


def test_job_queue_basic(tmp_path):
    jq = JobQueue(max_workers=2)
    s1 = Study(name="Study_1", geometry="model1.step")
    s2 = Study(name="Study_2", geometry="model2.step")

    def mock_solve(st, out_dir):
        time.sleep(0.05)
        return "mock_res"

    j1 = jq.submit(s1, tmp_path / "j1", solve_fn=mock_solve)
    j2 = jq.submit(s2, tmp_path / "j2", solve_fn=mock_solve)

    assert j1.id in [j.id for j in jq.list_jobs()]
    assert j2.id in [j.id for j in jq.list_jobs()]

    jq.wait_all(timeout=5.0)

    assert j1.status == "completed"
    assert j2.status == "completed"
    assert j1.results == "mock_res"
    assert j2.results == "mock_res"
    assert j1.duration_s is not None and j1.duration_s >= 0.04

    summary = jq.summary()
    assert summary["total"] == 2
    assert summary["completed"] == 2
    assert summary["failed"] == 0
    jq.shutdown()


def test_job_queue_error_handling(tmp_path):
    jq = JobQueue(max_workers=1)
    s = Study(name="FailingStudy", geometry="model.step")

    def failing_solve(st, out_dir):
        raise ValueError("Chyba site")

    job = jq.submit(s, tmp_path / "fail", solve_fn=failing_solve)
    jq.wait_all(timeout=5.0)

    assert job.status == "failed"
    assert "Chyba site" in str(job.error)
    assert jq.summary()["failed"] == 1
    jq.shutdown()


def test_job_queue_cancel_running(tmp_path):
    jq = JobQueue(max_workers=1)
    s = Study(name="LongStudy", geometry="model.step")

    started = threading.Event()

    def long_solve(st, out_dir, cancel=None):
        started.set()
        if cancel:
            cancelled = cancel.wait(timeout=2.0)
            if cancelled:
                raise RuntimeError("CalculiX byl zrušen")
        return "finished"

    job = jq.submit(s, tmp_path / "long", solve_fn=long_solve)
    assert started.wait(timeout=1.0)

    res = jq.cancel(job.id)
    assert res is True

    jq.wait_all(timeout=5.0)
    assert job.status == "cancelled"
    assert jq.summary()["cancelled"] == 1
    jq.shutdown()


def test_job_queue_cancel_pending_and_wait_all(tmp_path):
    jq = JobQueue(max_workers=1)
    s1 = Study(name="Study_1", geometry="model.step")
    s2 = Study(name="Study_2", geometry="model.step")

    gate = threading.Event()

    def blocking_solve(st, out_dir, cancel=None):
        gate.wait(timeout=2.0)
        return "done1"

    def second_solve(st, out_dir, cancel=None):
        return "done2"

    j1 = jq.submit(s1, tmp_path / "j1", solve_fn=blocking_solve)
    j2 = jq.submit(s2, tmp_path / "j2", solve_fn=second_solve)

    # j2 is in the waiting queue → we cancel it before it starts
    res = jq.cancel(j2.id)
    assert res is True
    assert j2.status == "cancelled"

    gate.set()
    # wait_all must not fail on CancelledError from j2
    jq.wait_all(timeout=5.0)

    assert j1.status == "completed"
    assert j2.status == "cancelled"
    jq.shutdown()


def test_job_queue_keep_summary_only(tmp_path):
    class MockResultsWithSummary:
        def summary(self):
            return {"nodes": 120, "max_mises": 45.0}

    jq = JobQueue(max_workers=1, keep_summary_only=True)
    s = Study(name="SummaryStudy", geometry="model.step")

    job = jq.submit(s, tmp_path / "summary", solve_fn=lambda st, wd: MockResultsWithSummary())
    jq.wait_all(timeout=5.0)

    assert job.status == "completed"
    assert job.results is None
    assert job.summary == {"nodes": 120, "max_mises": 45.0}
    jq.shutdown()
