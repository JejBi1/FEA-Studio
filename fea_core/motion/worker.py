"""Working process of the mechanism: python -m fea_core.motion.worker JOB

Input JOB/job.json (all in SI – prepared by fea_core.motion.run), output JOB/result.npz (fields at the output times),
JOB/result.json (metadata, check of the freedom, statistics), progress and messages to stdout (log). Cancellation = the end
of the process. It must not import OCP nor fea_core.selectors (the library Exudyn runs only here).

Tasks (job["kind"]):
  kinematic – all the degrees of freedom prescribed by the drivers (inverse dynamics: reactions including the inertia) –
              at each output time static assemblies at t + (−2…2)·δ → velocity and acceleration by finite differences
              of the 4th order → equilibrium with the inertia forces (d'Alembert); the integrator generalized-α gave
              noise ~1e-3 in the accelerations (amplification of the error of the position 1/(β·h²), scratch/f8_motion/t_mb1_conv.py);
  static    – quasi-static equilibrium in a series of positions (the same without the inertia forces);
  dynamic   – direct dynamics (generalized-α, implicit), output at the times outputs.
job["stage"] = "check" → only the check of the freedom (JOB/check.json).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ARRAY_KEYS = ("pos", "rot", "vel", "omega", "acc", "alpha", "jF", "jM", "jP", "jPa", "jFl", "jMl", "jq", "jv", "jfr",
              "jlim", "jQ", "dQ", "dq", "dv", "fval", "flen", "fF", "fM", "fPa", "fPb", "lF", "lM", "lP")
SCALAR_KEYS = ("ekin", "epot")
WORKS = ("wdrv", "wfr", "wlim", "wdamp", "wbush", "wload")
POWER_KEYS = ("pdrv", "pfr", "plim", "pdamp", "pbush", "pload", "pelem")


def log(*a):
    print(*a, flush=True)


def settings(model, dense: bool):
    exu = model.exu
    ss = exu.SimulationSettings()
    ss.solution.file.write = False
    if dense:
        ss.linearSolver.solverType = exu.LinearSolverType.EigenDense
        ss.linearSolver.ignoreSingularJacobian = True      # redundant constraints (planar mechanisms in 3D)
    ss.timeIntegration.verboseMode = int(os.environ.get("FEA_MOTION_VERBOSE", "0"))     # debugging: 1–3
    ss.staticSolver.verboseMode = 0
    ss.staticSolver.useLoadFactor = False
    ss.staticSolver.newton.relativeTolerance = 1e-12
    ss.staticSolver.newton.absoluteTolerance = 1e-14
    ss.staticSolver.newton.maxIterations = 50
    ss.timeIntegration.newton.relativeTolerance = 1e-10
    ss.timeIntegration.newton.absoluteTolerance = 1e-12
    return ss


# ---------------------------------------------------------------- mobility check
def dof_check(job: dict) -> dict:
    """Rank of the Jacobian of the constraints (Exudyn ComputeSystemDegreeOfFreedom) without the drivers and with them + redundant constraints
    by joints (successive addition of the columns of the constraints and a test of the rank).
    """
    from .exudyn_adapter import Model
    out = {}
    for with_drv in (False, True):
        m = Model(job, with_drivers=with_drv, log=log)
        ss = settings(m, True)
        d = m.mbs.ComputeSystemDegreeOfFreedom(simulationSettings=ss, threshold=1e-10)
        key = "driven" if with_drv else "free"
        modal = sum(int(B.get("nmodes", 0)) for B in m.B if B.get("flex"))     # modal coordinates of the flexible bodies
        out[key] = {"dof": int(d["degreeOfFreedom"]) - modal, "redundant": int(d["redundantConstraints"]),
                    "modal": modal}
        out[key]["redundant_list"] = _redundant_rows(m, ss)
    return out


def _redundant_rows(m, ss) -> list[dict]:
    exu = m.exu
    solver = exu.MainSolverStatic()
    solver.InitializeSolver(m.mbs, ss)
    n2 = solver.GetODE2size()
    n12 = n2 + solver.GetODE1size()
    solver.ComputeJacobianODE2RHS(m.mbs, scalarFactor_ODE2=-1, scalarFactor_ODE2_t=0, scalarFactor_ODE1=0)
    solver.ComputeJacobianAE(m.mbs, scalarFactor_ODE2=1., scalarFactor_ODE2_t=0., scalarFactor_ODE1=1.,
                             velocityLevel=False)
    jac = np.array(solver.GetSystemJacobian())
    solver.FinalizeSolver(m.mbs, ss)
    Cq = jac[0:n2, n12:]                      # columns = constraints (AE)
    owner = {}
    labels = ["x", "y", "z", "rx", "ry", "rz"]
    for J in m.J:
        if J["generic"] is not None:
            for k, ae in enumerate(m.mbs.systemData.GetObjectLTGAE(J["generic"])):
                owner[ae] = (J["name"], labels[k] if k < 6 else str(k))
        for key in ("dist", "screw"):
            if J.get(key) is not None:
                for ae in m.mbs.systemData.GetObjectLTGAE(J[key]):
                    owner[ae] = (J["name"], "constraint")
        if J.get("coupling") is not None:
            for ae in m.mbs.systemData.GetObjectLTGAE(J["coupling"][0]):
                owner[ae] = (J["name"], "gearing")
    for d in m.drivers_kin:
        ae = m.mbs.systemData.GetObjectLTGAE(d["obj"])[d["axis"]]
        owner[ae] = (d["name"], "drive")
    cols = [c for c in range(Cq.shape[1]) if np.abs(Cq[:, c]).max() > 0]
    scale = max(np.abs(Cq).max(), 1e-300)
    basis = np.zeros((n2, 0))
    red = []
    for c in cols:
        v = Cq[:, c:c + 1] / scale
        if basis.shape[1]:
            coef, *_ = np.linalg.lstsq(basis, v, rcond=None)
            res = v - basis @ coef
        else:
            res = v
        if np.linalg.norm(res) <= 1e-8 * max(np.linalg.norm(v), 1e-300):
            if c in owner:
                red.append({"joint": owner[c][0], "axis": owner[c][1]})
        else:
            basis = np.hstack([basis, v])
    return red


# ---------------------------------------------------------------- calculation
class Runner:
    def __init__(self, job: dict):
        from .exudyn_adapter import Model, MotionError
        self.job = job
        self.kind = job.get("kind", "dynamic")
        self.m = Model(job, log=log)
        self.exu = self.m.exu
        n2 = len(self.m.mbs.systemData.GetODE2Coordinates())
        self.dense = n2 < 800
        self.ss = settings(self.m, self.dense)
        self.N = max(int(job.get("outputs", 200)), 1)
        self.t_end = float(job["t_end"])
        if not self.t_end > 0:
            raise MotionError("The simulation time t_end must be positive")
        self.dt_out = self.t_end / self.N
        h = job.get("dt") or min(self.dt_out, self.t_end / 1000.0)
        self.nsub = max(int(math.ceil(self.dt_out / h - 1e-9)), 1)
        self.h = self.dt_out / self.nsub
        self.frames: list[dict] = []
        self.work = {k: 0.0 for k in WORKS}
        self.works: list[dict] = []
        self.prevP = None
        self.t_prev = None
        self.next_k = 0
        self.notes: list[str] = []
        self.last_report = 0.0

    # ------------------------------------------------------------ helpers
    def static_at(self, t, assembly=False):
        """Static solution at the time t (drivers and loads evaluated at t); returns (ODE2, data) of the current state."""
        m = self.m
        m.t_override = t
        m.assembly = assembly
        if assembly:                                    # assembly without weight and loads
            for B in m.B:
                if "gload" in B:
                    m.mbs.SetLoadParameter(B["gload"], "loadVector", [0.0, 0.0, 0.0])
            self.ss.staticSolver.stabilizerODE2term = 1.0
        err = ""
        try:
            ok = m.mbs.SolveStatic(self.ss)
        except Exception as e:  # noqa: BLE001 – Exudyn reports a SolverError
            ok = False
            err = str(e)
        finally:
            m.t_override = None
            m.assembly = False
            if assembly:
                for B in m.B:
                    if "gload" in B:
                        m.mbs.SetLoadParameter(B["gload"], "loadVector", list(m.g))
                self.ss.staticSolver.stabilizerODE2term = 0.0
        if ok is False:
            from .exudyn_adapter import MotionError
            what = "assembly of the mechanism" if assembly else "static solution"
            raise MotionError(f"At time t = {t:.6g} s {what} failed (Newton did not converge) – the mechanism cannot be assembled (dead position, unsatisfiable joints, prescribed motion out of reach) or a free degree of freedom has no stiffness (quasi-static problem without drive / spring)"
                              + (f" [{err[:200]}]" if err else ""))
        sd = m.mbs.systemData
        CT = self.exu.ConfigurationType
        return np.array(sd.GetODE2Coordinates(CT.Current)), np.array(sd.GetDataCoordinates(CT.Current))

    def set_state(self, cfg, q, v=None, a=None, d=None):
        sd = self.m.mbs.systemData
        CT = getattr(self.exu.ConfigurationType, cfg)
        sd.SetODE2Coordinates(list(q), CT)
        if v is not None:
            sd.SetODE2Coordinates_t(list(v), CT)
        if a is not None:
            sd.SetODE2Coordinates_tt(list(a), CT)
        if d is not None and len(d):
            sd.SetDataCoordinates(list(d), CT)

    def progress(self, t):
        now = time.time()
        if now - self.last_report > 2.0:
            self.last_report = now
            log(f"progress {100.0 * max(t, 0.0) / self.t_end:5.1f} % (t = {t:.6g} s)")

    def needs_assembly(self) -> bool:
        if any(j.get("origin_b") is not None and np.linalg.norm(np.subtract(j["origin_b"], j["origin"])) > 1e-12
               for j in self.job["joints"]):
            return True
        return any(abs(d["motion"].coord(0.0)) > 1e-12 for d in self.m.drivers_kin)

    # ------------------------------------------------------------ tasks
    def run(self) -> None:
        if self.kind in ("kinematic", "static"):
            self.run_quasi(inertia=self.kind == "kinematic")
        else:
            self.run_dynamic()

    def run_quasi(self, inertia: bool):
        """Kinematic / quasi-static task by a series of static solutions of Exudyn: at each output time the position at 5
        times t + (−2, −1, 0, 1, 2)·δ → velocity and acceleration by finite differences of the 4th order, then the equilibrium at the time t
        with the inertia forces (d'Alembert – only kinematic) and with the velocities from the differences (dampers, friction); friction
        from the reaction of the joint by iteration of a fixed point. The accuracy does not depend on the integrator.
        """
        m = self.m
        m.lam_sign = -1.0
        delta = float(self.job.get("fd_step") or 0.05 * self.dt_out)
        has_fric = any(J["fric"] for J in m.J)
        if self.needs_assembly():
            q, d = self.static_at(0.0, assembly=True)
            self.set_state("Initial", q, d=d)
            self.notes.append("The CAD position did not satisfy the joints / drive – the mechanism was assembled before the analysis")
        rates: dict = {}
        for k in range(self.N + 1):
            t = k * self.dt_out
            Q, D = [], None
            m.set_inertia(False)
            m.kin_rates = rates
            for s in (0, -1, 1, -2, 2):                 # the centre of the first → the starting estimate for the others
                q, d = self.static_at(t + s * delta)
                Q.append(q)
                if s == 0:
                    D = d
                    self.set_state("Initial", q, d=d)
            q0, qm1, qp1, qm2, qp2 = Q
            v = (qm2 - 8 * qm1 + 8 * qp1 - qp2) / (12 * delta)
            a = (-qm2 + 16 * qm1 - 30 * q0 + 16 * qp1 - qp2) / (12 * delta * delta)
            self.set_state("Current", q0, v, a, D)
            rates = m.current_rates()
            m.kin_rates = rates
            if inertia:
                m.set_inertia(True)
            for _ in range(10 if has_fric else 1):     # equilibrium at the time t (friction from the reaction – a fixed point)
                old = [J["Rprev"].copy() for J in m.J]
                self.static_at(t)
                m.update_states()
                if not has_fric:
                    break
                ch = max((float(np.abs(J["Rprev"] - r0).max()) / max(float(np.abs(J["Rprev"]).max()), 1e-30)
                          for J, r0 in zip(m.J, old)), default=0.0)
                if ch < 1e-12:
                    break
            self.set_state("Current", q0, v, a, D)
            self.frames.append(m.snapshot(t))
            self.progress(t)
        m.set_inertia(False)
        m.kin_rates = None
        self.integrate_works()

    def integrate_works(self):
        """Work of the non-conservative forces from the powers at the output times (cubic spline – order 4)."""
        t = np.array([f["t"] for f in self.frames])
        Wk = {}
        for key, pk in zip(WORKS, POWER_KEYS):
            P = np.array([f[pk] for f in self.frames])
            if len(t) >= 4:
                from scipy.interpolate import CubicSpline
                W = CubicSpline(t, P).antiderivative()(t)
                Wk[key] = W - W[0]
            else:
                Wk[key] = np.concatenate([[0.0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(t))])
        self.works = [{key: float(Wk[key][i]) for key in WORKS} for i in range(len(t))]

    def run_dynamic(self):
        m = self.m
        ti = self.ss.timeIntegration
        ti.generalizedAlpha.spectralRadius = float(self.job.get("spectral_radius") or 0.8)
        h = self.h
        if self.needs_assembly():
            q0, d0 = self.static_at(0.0, assembly=True)
            self.set_state("Initial", q0, np.zeros_like(q0), None, d0)
            self.notes.append("The CAD position did not satisfy the joints / drive – the mechanism was assembled before the analysis")
        def mismatch(d):
            want = d["motion"].rate(0.0)
            try:
                have = m.marker_coord(m.J[d["joint"]][d["coord"]])[1]     # from the initial velocities of the bodies (v0, w0)
            except Exception:  # noqa: BLE001
                have = 0.0
            return abs(want - have) > 1e-9 * max(abs(want), 1e-12)
        if any(mismatch(d) for d in m.drivers_kin):
            self.notes.append("The drive velocity at time 0 does not match the initial body velocities (Body.v0, w0) – the velocity is corrected in the first step (initial accelerations and reactions are not valid); give initial velocities, use a run-up (S-curve) or a kinematic analysis")
        ti.generalizedAlpha.computeInitialAccelerations = True
        ti.startTime = 0.0
        ti.numberOfSteps = self.N * self.nsub
        ti.endTime = self.t_end
        ti.adaptiveStep = True
        self.next_k = 0

        # Exudyn calls PostStep also at t = 0 (the initial state after the calculation of the initial accelerations) and PreStep with the time of the END
        # of the step in the state at its start (scratch/f8_motion/dbg_steps.py) → the snapshots only in PostStep
        def post(mbs, t):
            # the powers and the snapshot in the state with which the step was computed (the friction from the reaction of the PREVIOUS step), then the new reaction
            P = m.powers(t)
            if self.prevP is not None:
                dt = t - self.t_prev
                for key in self.work:
                    self.work[key] += 0.5 * (self.prevP[key] + P[key]) * dt
            self.prevP = P
            if self.next_k <= self.N and t >= self.next_k * self.dt_out - 1e-6 * h:
                self.frames.append(m.snapshot(t))
                self.works.append(dict(self.work))
                self.next_k += 1
            m.update_states()
            self.t_prev = t
            self.progress(t)
            return True
        if any(J["fric"] for J in m.J):
            # the friction in the 1st step needs the reaction at t = 0: a short preliminary step (the state then returns to the initial one)
            n0, e0 = ti.numberOfSteps, ti.endTime
            ti.numberOfSteps, ti.endTime = 1, h
            m.mbs.SetPostStepUserFunction(lambda mbs_, t_: True)
            try:
                m.mbs.SolveDynamic(self.ss)
                m.update_states()
            except Exception as e:  # noqa: BLE001 – the lead is only an estimate of the normal force
                self.notes.append(f"Joint friction: the preliminary step for the normal force at time 0 failed ({str(e)[:120]}) – "
                                  f"the friction in the first step is computed without the initial reaction")
            ti.numberOfSteps, ti.endTime = n0, e0
        m.mbs.SetPostStepUserFunction(post)
        err = ""
        try:
            ok = m.mbs.SolveDynamic(self.ss)
        except Exception as e:  # noqa: BLE001
            ok = False
            err = str(e)[:200]
        if ok is False:
            from .exudyn_adapter import MotionError
            raise MotionError("Time integration did not converge (Newton) – shorten the step dt, check the supports (dead position of the mechanism, unassemblable joints) and the stop stiffness"
                              + (f" [{err}]" if err else ""))
        if len(self.frames) < self.N + 1:
            self.notes.append(f"Only {len(self.frames)} of {self.N + 1} output times stored (the step was shortened)")

    # ------------------------------------------------------------ output
    def arrays(self) -> dict:
        fr = self.frames
        out = {"t": np.array([f["t"] for f in fr])}
        for k in ARRAY_KEYS:
            out[k] = np.array([f[k] for f in fr])
        for k in SCALAR_KEYS + POWER_KEYS:
            out[k] = np.array([f[k] for f in fr])
        for k in WORKS:
            out[k] = np.array([w[k] for w in self.works])
        for i in (fr[0].get("flexq") or {}) if fr else {}:          # modal coordinates of the flexible bodies
            out[f"flexq{i}"] = np.array([f["flexq"][i] for f in fr])
        return out


def main(job_dir: str) -> int:
    jd = Path(job_dir)
    job = json.loads((jd / "job.json").read_text(encoding="utf-8"))
    t0 = time.time()
    try:
        from ..modules import need
        need("exudyn", "motion")
        if job.get("stage") == "check":
            res = dof_check(job)
            (jd / "check.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            return 0
        res = dof_check(job)
        log(f"mobility: without drives {res['free']['dof']}, with drives {res['driven']['dof']}, redundant constraints {res['free']['redundant']}")
        r = Runner(job)
        if job.get("kind") == "kinematic" and res["driven"]["dof"] != 0:
            from .exudyn_adapter import MotionError
            raise MotionError(f"A kinematic analysis must prescribe all degrees of freedom – after the drives the mechanism still has {res['driven']['dof']} free (add a drive or choose a dynamic analysis)")
        if r.kind == "dynamic":
            log(f"integration step {r.h:.4g} s, {r.N + 1} outputs")
        else:
            log(f"{r.N + 1} outputs (series of static solutions)")
        r.run()
        np.savez(jd / "result.npz", **r.arrays())
        if r.m.B and any(B.get("flex") for B in r.m.B):
            r.notes += [f"flexible body {B['name']}: natural frequencies with fixed interfaces "
                        + ", ".join(f"{x:.4g}" for x in B["freq"][:5]) + " Hz" for B in r.m.B if B.get("flex")]
        meta = {"ok": True, "dof": res, "notes": r.notes, "time_s": time.time() - t0,
                "h": r.h if r.kind == "dynamic" else None, "frames": len(r.frames)}
        (jd / "result.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        log(f"finished in {time.time() - t0:.2f} s")
        return 0
    except Exception as e:  # noqa: BLE001 – error into result.json for the main process
        log(traceback.format_exc())
        msg = str(e) if type(e).__name__ in ("MotionError", "LawError", "ModuleMissing") else f"{type(e).__name__}: {e}"
        (jd / "result.json").write_text(json.dumps({"ok": False, "error": msg}, ensure_ascii=False), encoding="utf-8")
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
