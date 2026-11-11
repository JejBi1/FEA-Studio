"""Progress of the calculation 0–1 (green bar in the GUI): `run_study(on_progress=)`.

The calculation has phases with weights according to the typical share of time (meshing, thermal states, mechanics, processing of results,
waiting for parallel jobs); phases that do not occur (a ready mesh, no heat) are left out. Reports
on_progress(value, end of phase, text): value = the part done, end of phase = how far the GUI bar may smoothly "catch up"
before the next report comes (inside a long phase, e.g. Gmsh or the factorisation, the core does not know the progress). The value only grows.

Inside the phases of CalculiX the progress is read from the output of ccx:
- `CcxLines` (mechanics): "STEP k", matrix assembly, factorisation, stresses, "Job finished"; parallel jobs
  (ccx/parallel.py, lines "[step k] …") = average over the jobs. The output of parallel ccx jobs arrives in batches
  (buffer of the pipe) – between the batches the GUI bar catches up.
- `ThermalLines` (heat conduction): steady state = "Job finished", transient = "actual step time" / time of the state;
  parallel states have the prefix "[name] " (thermal/run.py).
Parallel background jobs (transient thermal states, creep of printing) report progress "deferred": it is stored and shown
only when the main calculation waits for the job (phase background) – otherwise the bar would jump ahead before the end of the mechanics.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Callable

# phase weights (relative); run_study adjusts them to the job (transient thermal states last longer)
WEIGHTS = {"check": 0.02, "mesh": 0.25, "thermal": 0.12, "mech": 0.50, "eigen": 0.25, "post": 0.05,
           "background": 0.04, "dynamics": 0.30}
TRANSIENT_WEIGHT = 0.30                     # phase with a transient thermal state (ccx assembles the matrix at every step)

# progress inside one ccx step according to the output lines (the share of the step at the line containing the text); "stress calculation"
# occurs twice in a step – before the matrix assembly and after the factorisation (computing the stresses from the displacements)
CCX_MARKS = [("STEP", 0.03), ("setting up the structure", 0.10), ("stiffness/mass contributions", 0.25),
             ("Factoring the system", 0.35), ("Job finished", 1.0)]
AFTER_FACTOR = ("stress calculation", 0.90)
# messages of the run_study log → (phase, share of the phase, text of the bar); the start of the message
LOG_MARKS = [
    ("Loading geometry", "check", 0.3, "Loading geometry"),
    ("Meshing in progress (Gmsh)", "mesh", 0.0, "Meshing (Gmsh)"),
    ("Mesh reused", "mesh", 1.0, "Mesh reused"),
    ("Mesh:", "mesh", 1.0, "Mesh done"),
    ("3D printing: assignment", "mesh", 0.95, "3D printing: shell and infill"),
    ("Computing (CalculiX", "mech", 0.0, "Load cases (CalculiX)"),
    ("Assembling matrix", "mech", 0.10, "Assembling stiffness matrix (NGSolve)"),
    ("Solving system", "mech", 0.30, "Solving system of equations"),
    ("  factorization", "mech", 0.75, "System solved"),
    ("Computing stresses", "mech", 0.85, "Stress from displacements"),
    ("Joint forces", "post", 0.20, "Joint forces"),
    ("3D printing: criterion", "post", 0.50, "3D printing: failure criterion"),
]
_STEP = re.compile(r"^\s*STEP\s+(\d+)")
_JOB = re.compile("^\\[step (\\d+)\\] ")
_CASE = re.compile(r"^\[([^\]]+)\] ")
_TIME = re.compile(r"actual step time\s*=\s*([-+0-9.,eE]+)")


class Progress:
    """Tracks the phases of the calculation and reports on_progress(value 0–1, end of phase 0–1, text). Safe from several threads.
    phases = order of the phases; weights = override of the WEIGHTS.
    """

    def __init__(self, cb: Callable[[float, float, str], None] | None, phases: list[str],
                 weights: dict[str, float] | None = None):
        self.cb = cb
        w = {**WEIGHTS, **(weights or {})}
        total = sum(w[p] for p in phases) or 1.0
        self.span: dict[str, tuple[float, float]] = {}
        a = 0.0
        for p in phases:
            b = a + w[p] / total
            self.span[p] = (a, b)
            a = b
        self.value = 0.0
        self.active: str | None = None
        self.deferred: dict[str, float] = {}           # phase → share reported by a background job before the start of the phase
        self.lock = threading.Lock()
        self.times: dict[str, float] = {}              # durations of the phases [s] (Results.info["timings"]) – also without cb
        self._t_key: str | None = None
        self._t0 = 0.0

    def _clock(self, key: str | None):
        """Phase durations: switching to another phase closes the previous one (parallel background jobs are not counted separately)."""
        now = time.perf_counter()
        if self._t_key is not None:
            self.times[self._t_key] = self.times.get(self._t_key, 0.0) + now - self._t0
        self._t_key, self._t0 = key, now

    def timings(self) -> dict[str, float]:
        """Phase durations [s] in the order they ran (an open phase is counted up to now)."""
        out = dict(self.times)
        if self._t_key is not None:
            out[self._t_key] = out.get(self._t_key, 0.0) + time.perf_counter() - self._t0
        return {k: round(v, 3) for k, v in out.items()}

    def phase(self, key: str, text: str, frac: float = 0.0):
        """Phase key, done share frac (0–1) – the value only grows (parallel jobs must not move the bar back)."""
        if key in self.span and key != self._t_key:     # background jobs call phase only with the active phase (later)
            with self.lock:
                self._clock(key)
        if self.cb is None or key not in self.span:
            return
        a, b = self.span[key]
        with self.lock:
            self.active = key
            frac = max(frac, self.deferred.pop(key, 0.0))
            v = max(self.value, a + (b - a) * min(max(frac, 0.0), 1.0))
            self.value = v
        self.cb(v, b, text)

    def later(self, key: str, frac: float, text: str):
        """Progress of a background job: immediately only if the phase key is active, otherwise it is remembered (see the docstring of the module)."""
        with self.lock:
            if self.active != key:
                self.deferred[key] = max(self.deferred.get(key, 0.0), frac)
                return
        self.phase(key, text, frac)

    def finish(self, text: str = "Done"):
        self._clock(None)
        if self.cb is not None:
            self.value = 1.0
            self.cb(1.0, 1.0, text)

    def log(self, log: Callable[[str], None]) -> Callable[[str], None]:
        """Log which, according to LOG_MARKS, moves the progress and passes the message on."""
        def f(msg: str):
            for prefix, key, frac, text in LOG_MARKS:
                if msg.startswith(prefix):
                    self.phase(key, text, frac)
                    break
            log(msg)
        return f

    def lines(self, on_line, key: str, parser, background: bool = False) -> Callable[[str], None] | None:
        """on_solver_line which at the same time moves the progress of the phase key according to the parser (CcxLines / ThermalLines);
        background = background job (progress deferred, `later`).
        """
        if self.cb is None:
            return on_line
        if background:
            def report(frac, text):
                self.later(key, frac, text)
        else:
            def report(frac, text):
                self.phase(key, text, frac)
        return tee(on_line, parser, report)


class CcxLines:
    """Progress of ccx from the output lines: n_steps steps in one job, or n_jobs parallel jobs with one step each
    (lines with the prefix "[step k] "). Calling the instance with a line → share 0–1 and a description, or None (the line changes nothing).
    """

    def __init__(self, n_steps: int, label: str = "Load case"):
        self.n = max(n_steps, 1)
        self.label = label
        self.job_frac: dict[int, float] = {}          # job / step → done share of the step
        self.factored: set[int] = set()               # jobs / steps after the factorisation
        self.step = 0                                 # current step of one job (1…n)

    def __call__(self, line: str):
        m = _JOB.match(line)
        job = int(m.group(1)) if m else None
        body = line[m.end():] if m else line
        s = _STEP.match(body)
        if s and job is None:
            self.step = int(s.group(1))
        key = job if job is not None else max(self.step, 1)
        sub = next((f for mark, f in CCX_MARKS if mark in body), None)
        if "Factoring the system" in body:
            self.factored.add(key)
        elif sub is None and AFTER_FACTOR[0] in body and key in self.factored:
            sub = AFTER_FACTOR[1]
        if sub is None:
            return None
        if job is not None:                          # parallel jobs: each = one step
            self.job_frac[job] = max(self.job_frac.get(job, 0.0), sub)
            done = sum(self.job_frac.values()) / self.n
            finished = sum(1 for f in self.job_frac.values() if f >= 1.0)
            return min(done, 1.0), f"{self.label} – done {finished}/{self.n}"
        if "Job finished" in body:
            return 1.0, f"{self.label} {self.n}/{self.n}"
        return min((key - 1 + sub) / self.n, 1.0), f"{self.label} {key}/{self.n}"


class ThermalLines:
    """Progress of the thermal states: cases = [(name, duration of the transient state [s] | None)]. Parallel states have the
    prefix "[name] ", the sequential ones do not (then "Job finished" moves to the next state in order).
    """

    def __init__(self, cases: list[tuple[str, float | None]], label: str = "Heat conduction"):
        self.cases = cases
        self.label = label
        self.frac = {name: 0.0 for name, _ in cases}
        self.duration = dict(cases)
        self.k = 0                                     # current state without the prefix

    def __call__(self, line: str):
        m = _CASE.match(line)
        name = m.group(1) if m and m.group(1) in self.frac else None
        body = line[m.end():] if name else line
        prefixed = name is not None
        if name is None:
            if self.k >= len(self.cases):
                return None
            name = self.cases[self.k][0]
        if "Job finished" in body:
            self.frac[name] = 1.0
            if not prefixed:
                self.k += 1
        else:
            t = _TIME.search(body)
            dur = self.duration.get(name)
            if t is None or not dur:
                return None
            try:
                self.frac[name] = max(self.frac[name], min(float(t.group(1).replace(",", ".")) / dur, 0.99))
            except ValueError:
                return None
        done = sum(1 for f in self.frac.values() if f >= 1.0)
        return sum(self.frac.values()) / len(self.frac), f"{self.label} – done {done}/{len(self.frac)}"


def tee(on_line: Callable[[str], None] | None, parser: Callable[[str], tuple | None], report: Callable):
    """on_line which passes the line and its result (share, text) to report(share, text) at the same time."""
    def f(line: str):
        r = parser(line)
        if r is not None:
            report(*r)
        if on_line is not None:
            on_line(line)
    return f


class NonlinearLines:
    """Progress of a nonlinear calculation (contact, history): jobs jobs = [(name, [durations of steps])]; lines of parallel
    jobs with the prefix "[step k] " (k = order of the job from 1). Share of the job = (done steps + time of the current step
    "actual step time=") / total time; ccx writes the step time after every converted increment.
    """

    def __init__(self, jobs: list[tuple[str, list[float]]], label: str = "Nonlinear analysis"):
        self.jobs = jobs
        self.label = label
        self.step = {k: 0 for k in range(1, len(jobs) + 1)}
        self.frac = {k: 0.0 for k in range(1, len(jobs) + 1)}

    def __call__(self, line: str):
        m = _JOB.match(line)
        job = int(m.group(1)) if m else 1
        body = line[m.end():] if m else line
        if job not in self.step:
            return None
        durs = self.jobs[job - 1][1]
        s = _STEP.match(body)
        if s:
            self.step[job] = int(s.group(1))
            return None
        if "Job finished" in body:
            self.frac[job] = 1.0
        else:
            t = _TIME.search(body)
            k = self.step[job]
            if t is None or not k or k > len(durs):
                return None
            try:
                ts = float(t.group(1).replace(",", "."))
            except ValueError:
                return None
            self.frac[job] = max(self.frac[job], min((sum(durs[:k - 1]) + ts) / max(sum(durs), 1e-30), 0.99))
        done = sum(1 for f in self.frac.values() if f >= 1.0)
        return sum(self.frac.values()) / len(self.frac), f"{self.label} – done {done}/{len(self.frac)}"
