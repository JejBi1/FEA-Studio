"""Queue of FEA jobs (stage F9.8, F4).

Allows running several studies or parametric variants one after another in the background,
watching their state and taking over the results.
"""
from __future__ import annotations

import inspect
import time
import uuid
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor, CancelledError

from .study import Study
from .results import Results


@dataclass
class Job:
    id: str
    name: str
    study: Study
    out_dir: Path
    status: str = "pending"          # pending | running | completed | failed | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    results: Results | None = None
    summary: dict | None = None      # light summary without holding the full fields in memory
    error: str | None = None
    log: list[str] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or time.time()
        return round(end - self.started_at, 2)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "error": self.error,
            "out_dir": str(self.out_dir),
            "study_name": self.study.name if hasattr(self.study, "name") else "",
            "summary": self.summary,
        }


class JobQueue:
    """Manager of the job queue with sequential or parallel execution.

    Important note on max_workers > 1:
      Each parallel instance of the CalculiX solver (ccx) allocates RAM according to the number of DOF of the model
      (see `fea_core.ccx.parallel.ccx_memory_gb`: e.g. 214 k DOF allocates ~1.4 GB, 706 k DOF ~6.8 GB).
      With max_workers > 1 the memory consumption multiplies and the RAM may be exhausted.
      The default safe value is therefore max_workers=1.
    """

    def __init__(self, max_workers: int = 1, keep_results: bool = True, keep_summary_only: bool = False):
        self.max_workers = max(1, int(max_workers))
        self.keep_results = keep_results
        self.keep_summary_only = keep_summary_only
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="FEA_JobQueue")
        self._futures: dict[str, Any] = {}

    def submit(self, study: Study, out_dir: Path | str, name: str | None = None,
               solve_fn: Callable[..., Results] | None = None,
               keep_results: bool | None = None,
               keep_summary_only: bool | None = None) -> Job:
        """Puts a new calculation job into the queue."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:8]
        job_name = name or (study.name if hasattr(study, "name") and study.name else f"Job_{job_id}")
        job = Job(id=job_id, name=job_name, study=study, out_dir=out)

        with self._lock:
            self.jobs[job_id] = job

        should_keep = self.keep_results if keep_results is None else keep_results
        summary_only = self.keep_summary_only if keep_summary_only is None else keep_summary_only

        def _worker():
            with self._lock:
                if job.status == "cancelled":
                    return
                job.status = "running"
                job.started_at = time.time()
            try:
                fn = solve_fn
                if fn is None:
                    from .solve import run_study
                    fn = run_study

                sig = inspect.signature(fn)
                if "cancel" in sig.parameters:
                    res = fn(job.study, job.out_dir, cancel=job.cancel_event)
                else:
                    res = fn(job.study, job.out_dir)

                if job.cancel_event.is_set():
                    job.status = "cancelled"
                    job.error = "Job was cancelled by the user"
                    return

                if hasattr(res, "summary") and callable(res.summary):
                    job.summary = res.summary()

                if should_keep and not summary_only:
                    job.results = res
                else:
                    job.results = None

                job.status = "completed"
            except Exception as e:
                if job.cancel_event.is_set():
                    job.status = "cancelled"
                    job.error = "Job was cancelled by the user"
                else:
                    job.error = str(e)
                    job.status = "failed"
            finally:
                job.finished_at = time.time()

        future = self._executor.submit(_worker)
        self._futures[job_id] = future
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self.jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return list(self.jobs.values())

    def cancel(self, job_id: str) -> bool:
        """Cancels a waiting or running job in the queue."""
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            if job.status in ("completed", "failed", "cancelled"):
                return False

            job.cancel_event.set()
            fut = self._futures.get(job_id)
            if fut and fut.cancel():
                # the job was in the waiting queue and was cancelled successfully before it started
                job.status = "cancelled"
                job.finished_at = time.time()
                return True
            else:
                # the job is already running – it was signalled through cancel_event
                job.status = "cancelled"
                job.finished_at = time.time()
                return True

    def wait_all(self, timeout: float | None = None) -> bool:
        """Waits until all jobs in the queue have finished. Handles CancelledError of the cancelled futures."""
        t0 = time.time()
        for fut in list(self._futures.values()):
            rem = None if timeout is None else max(0.0, timeout - (time.time() - t0))
            try:
                fut.result(timeout=rem)
            except CancelledError:
                pass
            except Exception:
                pass
        return True

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """Shuts down the thread pool of the job queue."""
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def summary(self) -> dict:
        jobs = self.list_jobs()
        return {
            "total": len(jobs),
            "pending": sum(1 for j in jobs if j.status == "pending"),
            "running": sum(1 for j in jobs if j.status == "running"),
            "completed": sum(1 for j in jobs if j.status == "completed"),
            "failed": sum(1 for j in jobs if j.status == "failed"),
            "cancelled": sum(1 for j in jobs if j.status == "cancelled"),
        }
