"""Tools: asynchronous jobs for long calculations (contact, dynamics, mechanism) – specification F8, 4.9 point 4.

fea_job_start runs the calculation in a thread over the same session (Session) and returns the job id at once. The job holds the lock
of the session for the whole time of the calculation – other tools working with the study are refused ("Server je zaneprázdněn" = server busy, REST 409), the same
as with the blocking fea_solve_run; a second parallel calculation is not possible. fea_job_status / _cancel / _result do not need the lock
(the job state has its own lock) and answer even during the calculation (REST exempts them from the "busy" check – NO_LOCK_TOOLS).
Progress 0–1 and phases from fea_core.progress (run_study(on_progress=)), the last lines of the log and of the solver output.
"""
from __future__ import annotations

import itertools
import threading
import time
from collections import deque

from .common import out
from .session import FeaError, Session

NO_LOCK_TOOLS = {"fea_job_status", "fea_job_cancel", "fea_job_result"}
KIND_LABELS = {"solve": "study calculation", "motion": "mechanism calculation"}
DEFAULT_LIMIT_S = {"linear": 1800.0, "dynamics": 7200.0, "nonlinear": 14400.0, "motion": 3600.0}
_ids = itertools.count(1)


class Job:
    def __init__(self, jid: str, kind: str, timeout_s: float, detail: str):
        self.id, self.kind, self.timeout_s, self.detail = jid, kind, timeout_s, detail
        self.status = "starting"            # starting | running | done | error | cancelled | timeout
        self.t0 = time.time()
        self.t_end: float | None = None
        self.value, self.end, self.text = 0.0, 0.0, "Starting"
        self.log: deque[str] = deque(maxlen=300)
        self.solver: deque[str] = deque(maxlen=40)
        self.result: dict | None = None
        self.error: str | None = None
        self.cancel = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def progress(self, v, end=None, text: str = ""):
        with self.lock:
            self.value = max(self.value, min(float(v), 1.0))
            self.end = max(self.value, float(end) if end is not None else self.value)
            if text:
                self.text = str(text)

    def add_log(self, line: str):
        with self.lock:
            self.log.append(line)

    def add_line(self, line: str):
        with self.lock:
            self.solver.append(line.rstrip())

    @property
    def active(self) -> bool:
        return self.status in ("starting", "running")

    def snapshot(self, lines: int = 10) -> dict:
        with self.lock:
            el = (self.t_end or time.time()) - self.t0
            d = {"job_id": self.id, "kind": self.kind, "detail": self.detail, "status": self.status,
                 "progress": round(self.value, 4), "phase": self.text, "elapsed_s": round(el, 1),
                 "timeout_s": self.timeout_s, "log": list(self.log)[-lines:]}
            if self.status == "running":
                d["remaining_limit_s"] = round(max(self.timeout_s - el, 0.0), 1)
                if self.cancel.is_set():
                    d["cancelling"] = True
                if self.value > 0.02:
                    d["estimated_total_s"] = round(el / self.value, 0)
            if self.solver and self.status == "running":
                d["solver_tail"] = list(self.solver)[-5:]
            if self.error:
                d["error"] = self.error
            return d


def jobs(sess: Session) -> dict[str, Job]:
    return sess.__dict__.setdefault("_fea_jobs", {})


def _get(sess: Session, job_id: str | None) -> Job:
    js = jobs(sess)
    if not js:
        raise FeaError("No job – start fea_job_start(kind='solve')")
    if job_id is None:
        return list(js.values())[-1]
    if job_id not in js:
        raise FeaError(f"Job '{job_id}' does not exist (jobs: {', '.join(js)})")
    return js[job_id]


def default_limit(st, kind: str) -> tuple[float, str]:
    if kind == "motion":
        return DEFAULT_LIMIT_S["motion"], "mechanismus"
    if st is not None and st.is_nonlinear():
        return DEFAULT_LIMIT_S["nonlinear"], "nonlinear (contact)"
    if st is not None and st.dynamics.enabled:
        return DEFAULT_LIMIT_S["dynamics"], "dynamika"
    return DEFAULT_LIMIT_S["linear"], "linear"


def _motion_core(sess: Session, timeout_s: float, job: Job) -> dict:
    """Calculation of a mechanism (fea_core.motion) – result to sess.motion (shared by the motion tools)."""
    from .common import run_with_timeout
    st = sess.require_study()
    if not getattr(st, "mechanism", None) or not st.mechanism.active:
        raise FeaError("The study has no mechanism (bodies, joints, drives) – tools fea_motion_*")
    try:                                             # motion tools can have their own job body
        from . import tools_motion
        fn = getattr(tools_motion, "job_run", None)
    except ImportError:
        fn = None
    if fn is not None:
        return fn(sess, cancel=job.cancel, on_log=job.add_log, on_progress=job.progress, timeout_s=timeout_s)
    from ..motion.run import run_motion
    parts = None if sess.frame_only or not str(st.geometry or "").strip() else sess.parts

    def work(c):
        return run_motion(st, (sess.workdir or sess.base_dir) / "run", parts=parts, on_log=job.add_log, cancel=c,
                          on_progress=lambda v, text="": job.progress(v, v, text), timeout=timeout_s)
    res = run_with_timeout(work, timeout_s, "Mechanism", job.cancel)
    sess.motion = res
    return {"ok": True, "kind": "motion", "summary": res.summary(), "warnings": list(res.warnings)}


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_job_start(kind: str = "solve", backend: str = "auto", timeout_s: float | None = None) -> str:
        """Starts a long calculation ASYNCHRONOUSLY, returns job_id. kind: solve (whole study as fea_solve_run) | motion (mechanism). Track with fea_job_status, stop with fea_job_cancel, fetch with fea_job_result. Other study tools are busy meanwhile; one job at a time. timeout_s None = default per kind: linear 30 min, dynamics 2 h, contact/nonlinear 4 h, mechanism 1 h. backend as in fea_solve_run."""
        if kind not in KIND_LABELS:
            raise FeaError("kind: solve | motion")
        if backend not in ("auto", "ccx", "ngs_hypre", "ngs_pardiso"):
            raise FeaError("backend must be auto | ccx | ngs_hypre | ngs_pardiso")
        running = [j for j in jobs(sess).values() if j.active]
        if running:
            raise FeaError(f"Job {running[0].id} ({KIND_LABELS[running[0].kind]}) is already running – wait (fea_job_status) or cancel it (fea_job_cancel)")
        if sess.study is None:
            raise FeaError("No study is open – fea_study_new / fea_study_open")
        limit, detail = default_limit(sess.study, kind)
        if timeout_s is not None:
            if not timeout_s > 0:
                raise FeaError("timeout_s must be positive")
            limit = float(timeout_s)
        job = Job(f"job-{next(_ids)}", kind, limit, detail)

        def body():
            try:
                with sess.use(wait=5, op=f"job {job.id} ({KIND_LABELS[kind]})"):
                    job.status = "running"
                    job.ready.set()
                    if kind == "solve":
                        from .tools_results import solve_core
                        job.result = solve_core(sess, backend, limit, cancel=job.cancel, on_progress=job.progress,
                                                on_log=job.add_log, on_line=job.add_line)
                    else:
                        job.result = _motion_core(sess, limit, job)
                job.status = "done"
                job.progress(1.0, 1.0, "Done")
            except BaseException as e:  # noqa: BLE001 – the state of the job carries the error
                msg = str(e) if isinstance(e, FeaError) else f"{type(e).__name__}: {e}"
                job.error = msg
                job.status = ("cancelled" if job.cancel.is_set() else
                              "timeout" if "time limit" in msg else "error")
            finally:
                job.t_end = time.time()
                job.ready.set()
        js = jobs(sess)
        js[job.id] = job
        for old in [k for k, j in js.items() if not j.active][:-10]:       # history of at most 10 finished jobs
            del js[old]
        job.thread = threading.Thread(target=body, daemon=True, name=f"fea-job {job.id}")
        job.thread.start()
        job.ready.wait(15)
        if job.status == "error" and job.result is None and "busy" in (job.error or ""):
            del js[job.id]
            raise FeaError(job.error)
        d = job.snapshot()
        d["next"] = f"fea_job_status(job_id='{job.id}') every ~10–30 s, then fea_job_result(job_id='{job.id}')"
        return out(d)

    @mcp.tool(structured_output=False)
    def fea_job_status(job_id: str | None = None, log_lines: int = 10) -> str:
        """Job state (None = last): status starting|running|done|error|cancelled|timeout, progress 0-1, phase, elapsed_s, remaining limit, total estimate, last log_lines. Never blocks on the session lock."""
        j = _get(sess, job_id)
        d = j.snapshot(max(1, min(int(log_lines), 100)))
        if len(jobs(sess)) > 1 and job_id is None:
            d["jobs"] = [{"job_id": x.id, "kind": x.kind, "status": x.status} for x in jobs(sess).values()]
        return out(d)

    @mcp.tool(structured_output=False)
    def fea_job_cancel(job_id: str | None = None, wait_s: float = 30) -> str:
        """Cancels a running job (None = last) and kills its solver processes; no result is saved. Waits up to wait_s."""
        j = _get(sess, job_id)
        if not j.active:
            return out({**j.snapshot(), "note": "The job is no longer running"})
        j.cancel.set()
        if j.thread is not None:
            j.thread.join(max(0.0, float(wait_s)))
        return out(j.snapshot())

    @mcp.tool(structured_output=False)
    def fea_job_result(job_id: str | None = None) -> str:
        """Job result (None = last): after completion the same summary as fea_solve_run (results stay in the session for fea_results_*, fea_contact_*, fea_dyn_*, report). A running job returns its state; error/cancel = tool error."""
        j = _get(sess, job_id)
        if j.active:
            return out({**j.snapshot(), "note": "The job is still running – fea_job_status / wait"})
        if j.status != "done":
            raise FeaError(f"Job {j.id} ended with the state '{j.status}': {j.error}")
        return out({"job_id": j.id, "status": j.status, "elapsed_s": round((j.t_end or time.time()) - j.t0, 1),
                    **(j.result or {})})
