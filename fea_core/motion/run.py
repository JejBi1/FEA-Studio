"""Calculation of the mechanism: check, preparation of the task (SI), working process of Exudyn, reading and checking of the results.

check_dof(study, parts) – Grübler + rank of the Jacobian of the constraints (Exudyn in the worker), redundant constraints, drivers.
run_motion(study, workdir, parts=None, …) – MotionResult in the units of the project; check of the equilibrium of each body
at each output time (Σ F = m·a, Σ M = J·α + ω × J·ω, tolerance config.EQUILIBRIUM_TOL), power balance
(ΔE_k + ΔE_p = work of the drivers, loads, friction, dampers), check of the collisions (collisions.py) according to MotionRun.collisions.
"""
from __future__ import annotations

import json
import os
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .. import config
from .laws import KINEMATIC_QUANTITIES, Law, LawError
from .model import JOINT_CONSTRAINTS, JOINT_KINDS, DRIVER_QUANTITIES, FORCE_KINDS
from .results import MotionResult

MM = 1e-3            # mm → m
T2KG = 1e3           # t → kg
INERTIA = 1e-3       # t·mm² → kg·m²
NMM = 1e-3           # N·mm → N·m
CONTACT_TIME = 1e-3  # default duration of the impact on the stop [s] → stiffness of the stop
DRIVER_SCALE = {"angle": 1.0, "disp": MM, "velocity": MM, "angular_velocity": 1.0, "torque": NMM, "force": 1.0}
ROT_KINDS = ("revolute", "cylindrical", "screw", "pin_slot")
TRANS_KINDS = ("prismatic", "cylindrical", "screw", "pin_slot")


class MotionError(RuntimeError):
    """Comprehensible error of the mechanism (check, calculation)."""


# ---------------------------------------------------------------- model check
def check_model(study, parts=None, run: bool = True) -> tuple[list[str], list[str]]:
    """(errors, warnings) of the model of the mechanism without the calculation; run=False = without the check of the settings of the calculation (MotionRun)."""
    mech = study.mechanism
    errs, warns = [], []
    if not mech.bodies:
        return ["The mechanism has no body"], warns
    names = [b.name for b in mech.bodies]
    if len(set(names)) != len(names):
        errs.append("Body names are repeated")
    if not any(b.ground for b in mech.bodies) and not any(not j.body_a or not j.body_b for j in mech.joints):
        warns.append("The mechanism has no ground (ground body) nor a joint to the ground – free-flying bodies")
    pn = {p.name for p in parts} if parts is not None else None
    for b in mech.bodies:
        if pn is not None:
            miss = [p for p in b.parts if p not in pn]
            if miss:
                errs.append(f"Body '{b.name}': parts {', '.join(miss)} are not in the geometry")
        if not b.parts and not b.ground and (b.mass is None or b.inertia is None):
            errs.append(f"Body '{b.name}' has neither parts nor given mass and inertia")
    jn = [j.name for j in mech.joints]
    if len(set(jn)) != len(jn):
        errs.append("Joint names are repeated")
    for j in mech.joints:
        if j.kind not in JOINT_KINDS:
            errs.append(f"Joint '{j.name}': unknown type '{j.kind}'")
            continue
        if j.kind in ("gear", "rack_pinion"):
            for jj in (j.joint_a, j.joint_b):
                if jj not in jn:
                    errs.append(f"Joint '{j.name}': coupled joint '{jj}' does not exist")
            if not j.ratio:
                errs.append(f"Joint '{j.name}': the gear ratio is missing")
            continue
        for bn in (j.body_a, j.body_b):
            if bn and bn not in names:
                errs.append(f"Joint '{j.name}': body '{bn}' does not exist")
        if j.body_a == j.body_b:
            errs.append(f"Joint '{j.name}' connects a body with itself")
        if j.kind == "screw" and not j.lead:
            errs.append(f"Screw joint '{j.name}' has no lead value")
        if j.kind == "distance" and (j.point_a is None or j.point_b is None):
            errs.append(f"Link '{j.name}' needs points point_a and point_b")
        if j.limits is not None and (len(j.limits) != 2):
            errs.append(f"Joint '{j.name}': limits = [lower, upper]")
    for d in mech.drivers:
        if d.joint not in jn:
            errs.append(f"Driver '{d.name or d.joint}': joint '{d.joint}' does not exist")
        if d.quantity not in DRIVER_QUANTITIES:
            errs.append(f"Drive '{d.name or d.joint}': unknown quantity '{d.quantity}'")
        try:
            Law(d.law, d.data)
        except LawError as e:
            errs.append(f"Pohon '{d.name or d.joint}': {e}")
    for f in mech.forces:
        if f.kind not in FORCE_KINDS:
            errs.append(f"Force element '{f.name}': unknown type '{f.kind}'")
        if f.kind == "torsion" and f.joint not in jn:
            errs.append(f"Torsion spring '{f.name}': joint '{f.joint}' does not exist")
        for bn in (f.body_a, f.body_b):
            if f.kind != "torsion" and bn and bn not in names:
                errs.append(f"Force element '{f.name}': body '{bn}' does not exist")
    for ld in mech.loads:
        if ld.body not in names:
            errs.append(f"Load '{ld.name}': body '{ld.body}' does not exist")
    r = mech.run
    for b in mech.bodies:
        if b.flexible and not b.ground and not b.parts:
            errs.append(f"Flexible body '{b.name}' needs parts from the geometry (mesh)")
    for c in mech.contacts:
        for bn in (c.body_a, c.body_b):
            if bn not in names:
                errs.append(f"Contact '{c.name or c.body_a + '–' + c.body_b}': body '{bn}' does not exist")
            elif not next(b for b in mech.bodies if b.name == bn).parts:
                errs.append(f"Contact '{c.name or c.body_a + '–' + c.body_b}': body '{bn}' has no CAD parts (shape)")
        if c.method not in ("nsc", "smc"):
            errs.append(f"Kontakt '{c.name}': metoda nsc | smc")
        if c.mu < 0 or c.restitution < 0 or c.restitution > 1:
            errs.append(f"Kontakt '{c.name}': μ ≥ 0, restituce 0–1")
    if not run:
        return errs, warns
    if r.solver not in ("exudyn", "chrono"):
        errs.append(f"Unknown mechanism solver '{r.solver}' (exudyn | chrono)")
    if mech.contacts and r.solver != "chrono":
        errs.append("Body contact is computed only by the Chrono solver (MotionRun.solver = 'chrono')")
    if r.solver == "chrono":
        from .chrono_run import available, INSTALL_HINT
        if not available():
            errs.append(INSTALL_HINT)
        if r.kind == "static":
            errs.append("The Chrono solver computes only dynamic and kinematic analyses")
        if any(b.flexible and not b.ground for b in mech.bodies):
            errs.append("Flexible bodies (Craig–Bampton) are supported only by the Exudyn solver")
        for j in mech.joints:
            if j.kind in ("gear", "rack_pinion", "pin_slot"):
                errs.append(f"Joint '{j.name}' ({j.kind}) is supported only by the Exudyn solver")
            if j.friction is not None or j.limits is not None:
                errs.append(f"Joint '{j.name}': friction and stops in the joint are supported only by Exudyn (in Chrono, use shape contact)")
        for f in mech.forces:
            if f.kind not in ("spring", "damper", "spring_damper"):
                errs.append(f"Force element '{f.name}' ({f.kind}) is supported only by the Exudyn solver")
        if r.collisions:
            warns.append("The collision check (collisions) is redundant with the Chrono solver – bodies with contact cannot pass through each other")
    if any(b.flexible and not b.ground for b in mech.bodies) and r.kind != "dynamic":
        errs.append("Flexible bodies only in a dynamic analysis (kind='dynamic'; kinematic drives work there too)")
    if r.kind not in ("kinematic", "static", "dynamic"):
        errs.append(f"Unknown analysis kind '{r.kind}' (kinematic | static | dynamic)")
    if not r.t_end or r.t_end <= 0:
        errs.append("The simulation time t_end must be positive")
    if r.kind == "kinematic" and not any(d.quantity in KINEMATIC_QUANTITIES for d in mech.drivers):
        errs.append("A kinematic analysis needs a kinematic drive (angle, displacement or velocity)")
    return errs, warns


def grubler(study) -> dict:
    """Grübler–Kutzbach formula M = 6·(n − 1) − Σ constraints (n = bodies including the frame)."""
    mech = study.mechanism
    moving = [b for b in mech.bodies if not b.ground]
    c = sum(JOINT_CONSTRAINTS.get(j.kind, 0) for j in mech.joints)
    kin = sum(1 for d in mech.drivers if d.quantity in KINEMATIC_QUANTITIES)
    return {"moving_bodies": len(moving), "constraints": c, "dof": 6 * len(moving) - c, "kinematic_drivers": kin}


# ---------------------------------------------------------------- preparation of the task
def _need_parts(study) -> bool:
    return bool(study.mechanism.contacts) or any(
        ((b.mass is None or b.com is None or b.inertia is None) or b.flexible) and not b.ground
        for b in study.mechanism.bodies)


def _load_parts(study, parts, study_dir):
    if parts is not None or not _need_parts(study):
        return parts
    if not str(study.geometry or "").strip():
        raise MotionError("The bodies have no mass given and the study has no STEP geometry")
    from ..geometry import legacy_step_names, read_step
    return read_step(study.geometry_path(study_dir), legacy_names=legacy_step_names(study))


def body_props(study, parts) -> dict:
    """Mass properties of the bodies (mass.body_properties, or only the given values without geometry)."""
    mech = study.mechanism
    if parts is None:
        out = {}
        for b in mech.bodies:
            out[b.name] = {"mass": float(b.mass or 0.0), "com": np.asarray(b.com or [0, 0, 0], float),
                           "inertia": np.asarray(b.inertia or np.zeros((3, 3)), float).reshape(3, 3),
                           "parts": list(b.parts), "extra": [], "printed": [], "source": "user"}
        out["_warnings"], out["_check"] = [], {}
        return out
    from .mass import body_properties, MassError
    try:
        return body_properties(study, parts, include_extra=mech.run.include_extra_mass)
    except MassError as e:
        raise MotionError(str(e)) from None


def _auto_limit(j, bi, props, bodies) -> tuple[float, float]:
    """Stiffness and damping of the stop (SI): k = m_ef·(π/t_c)², c = 2ζ√(k·m_ef), ζ from the restitution e."""
    rot = j.kind in ("revolute", "cylindrical", "screw")
    b = bi.get(j.body_b, -1)
    if b < 0 or bodies[b].ground:
        b = bi.get(j.body_a, -1)
    p = props[bodies[b].name] if b >= 0 else None
    if p is None:
        m_ef = 1.0
    elif rot:
        n = np.asarray(j.axis, float) / np.linalg.norm(j.axis)
        r = (p["com"] - np.asarray(j.origin, float)) * MM
        rp = r - (r @ n) * n
        m_ef = float(n @ (p["inertia"] * INERTIA) @ n + p["mass"] * T2KG * (rp @ rp))
    else:
        m_ef = p["mass"] * T2KG
    m_ef = max(m_ef, 1e-12)
    if j.limit_stiffness:
        k = float(j.limit_stiffness) * (NMM if rot else 1e3)
    else:
        k = m_ef * (math.pi / CONTACT_TIME) ** 2
    zeta = zeta_for_restitution(float(j.restitution))
    return k, 2 * zeta * math.sqrt(k * m_ef)


def restitution_kv(zeta: float) -> float:
    """Coefficient of restitution of the linear stop (spring + damper, the force only pushes – the contact ends at zero force)."""
    if zeta <= 0:
        return 1.0
    if abs(zeta - 1) < 1e-9:                                          # critical damping: x = t·e^(−t), separation at t = 2
        return math.exp(-2.0)
    if zeta > 1:                                                      # overcritical: x = (e^(r1·t) − e^(r2·t))/(r1 − r2)
        q = math.sqrt(zeta * zeta - 1)
        r1, r2 = -zeta + q, -zeta - q
        arg = (1 + 2 * zeta * r2) / (1 + 2 * zeta * r1)
        if arg <= 0:
            return 0.0
        t = math.log(arg) / (r1 - r2)
        return -(r1 * math.exp(r1 * t) - r2 * math.exp(r2 * t)) / (r1 - r2)
    s = math.sqrt(1 - zeta * zeta)
    th = math.pi - math.atan2(2 * zeta * s, 1 - 2 * zeta * zeta)       # ω_d·t separation (ω²x + 2ζωẋ = 0)
    return -math.exp(-zeta * th / s) * (math.cos(th) - zeta / s * math.sin(th))


def zeta_for_restitution(e: float) -> float:
    """Relative damping of the stop ζ for the given coefficient of restitution e (bisection – e(ζ) decreases)."""
    e = min(max(e, 0.0), 1.0)
    if e >= 1:
        return 0.0
    if e <= 0:
        return 50.0
    lo, hi = 0.0, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if restitution_kv(mid) > e:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def prepare_job(study, parts=None, kind: str | None = None, jd=None, log=None) -> tuple[dict, dict, dict]:
    """(job in SI for the worker, meta in the units of the project, mass properties); jd = the component of the task (flexible bodies:
    the mesh and the files of Craig–Bampton, props["_flex"] = the data for flex.finish).
    """
    mech = study.mechanism
    run = mech.run
    props = body_props(study, parts)
    bodies = mech.bodies
    bi = {b.name: i for i, b in enumerate(bodies)}
    jbi = {j.name: i for i, j in enumerate(mech.joints)}

    def bidx(name):
        if not name:
            return -1
        return bi[name]
    jb, mj = [], []
    flex = {}
    for k, b in enumerate(bodies):
        p = props[b.name]
        jb.append({"name": b.name, "ground": bool(b.ground), "mass": p["mass"] * T2KG,
                   "com": list(p["com"] * MM), "inertia": (p["inertia"] * INERTIA).tolist(),
                   "v0": None if b.v0 is None else list(np.asarray(b.v0, float) * MM),
                   "w0": None if b.w0 is None else list(map(float, b.w0))})
        if b.flexible and not b.ground and jd is not None:
            from .flex import prepare, FlexError
            if parts is None:
                raise MotionError(f"Flexible body '{b.name}' needs geometry (mesh from CAD)")
            try:
                info = prepare(study, parts, b, k, Path(jd), log)
            except FlexError as e:
                raise MotionError(str(e)) from None
            flex[k] = info
            jb[-1].update({"flexible": True, **{key: v for key, v in info.items() if not key.startswith("_")
                                                and key != "mats"}})
    props["_flex"] = flex
    joints = []
    for j in mech.joints:
        d = {"name": j.name, "kind": j.kind, "a": bidx(j.body_a), "b": bidx(j.body_b),
             "origin": list(np.asarray(j.origin, float) * MM), "axis": list(map(float, j.axis))}
        if j.axis2 is not None:
            d["axis2"] = list(map(float, j.axis2))
        if j.kind != "distance" and j.point_b is not None:
            d["origin_b"] = list(np.asarray(j.point_b, float) * MM)
        if j.kind == "distance":
            d["point_a"] = list(np.asarray(j.point_a, float) * MM)
            d["point_b"] = list(np.asarray(j.point_b, float) * MM)
        if j.kind in ("gear", "rack_pinion"):
            d["joint_a"], d["joint_b"] = jbi[j.joint_a], jbi[j.joint_b]
            d["ratio"] = float(j.ratio) * (MM if j.kind == "rack_pinion" else 1.0)
        if j.kind == "screw":
            d["lead"] = float(j.lead) * MM
        prim_rot = j.kind in ROT_KINDS and j.kind != "pin_slot"
        if j.limits is not None:
            sc = 1.0 if prim_rot else MM
            d["limits"] = [None if v is None else float(v) * sc for v in j.limits]
            d["limit_k"], d["limit_c"] = _auto_limit(j, bi, props, bodies)
        if j.friction is not None:
            f = j.friction
            d["friction"] = {"model": f.model, "mu_s": f.mu_s, "mu_k": f.mu_k, "delta": f.delta,
                             "r_eff": f.r_eff * MM, "fit": f.fit, "preload": f.preload, "v_reg": f.v_reg * MM,
                             "c_v": {"rot": f.c_v * NMM, "trans": f.c_v * 1e3},
                             "v_s": {"rot": f.v_s, "trans": f.v_s * MM},
                             "sigma2": {"rot": f.sigma2 * NMM, "trans": f.sigma2 * 1e3}}
        joints.append(d)
        mj.append({"name": j.name, "kind": j.kind, "a": d["a"], "b": d["b"], "a_name": j.body_a or "",
                   "b_name": j.body_b or "", "origin": list(map(float, j.origin)),
                   "point_b": None if j.point_b is None else list(map(float, j.point_b)),
                   "axis": list(map(float, j.axis)), "axis2": None if j.axis2 is None else list(map(float, j.axis2)),
                   "selection_a": j.selection_a, "selection_b": j.selection_b,
                   "primary": "trans" if j.kind in ("prismatic", "pin_slot") else "rot" if prim_rot else None})
    forces, mf = [], []
    for f in mech.forces:
        d = {"name": f.name, "kind": f.kind, "a": bidx(f.body_a), "b": bidx(f.body_b),
             "point_a": list(np.asarray(f.point_a, float) * MM), "point_b": list(np.asarray(f.point_b, float) * MM)}
        if f.kind == "torsion":
            d.update(joint=jbi[f.joint], k=f.k * NMM, c=f.c * NMM, f0=f.f0 * NMM, phi0=float(f.l0 or 0.0))
        elif f.kind == "bushing":
            k6 = list(f.k6 or [0] * 6)
            c6 = list(f.c6 or [0] * 6)
            d["k6"] = [k6[i] * (1e3 if i < 3 else NMM) for i in range(6)]
            d["c6"] = [c6[i] * (1e3 if i < 3 else NMM) for i in range(6)]
        else:
            d.update(k=f.k * 1e3, c=f.c * 1e3, f0=f.f0, l0=None if f.l0 is None else f.l0 * MM)
            if f.kind == "gas_spring":
                d["table"] = [[s * MM, F] for s, F in (f.table or [])]
        forces.append(d)
        mf.append({"name": f.name, "kind": f.kind, "a": d["a"], "b": d["b"], "body_a": f.body_a, "body_b": f.body_b,
                   "selection_a": f.selection_a, "selection_b": f.selection_b, "joint": f.joint})
    drivers, md = [], []
    for d0 in mech.drivers:
        k = jbi[d0.joint]
        drivers.append({"name": d0.name or d0.joint, "joint": k, "quantity": d0.quantity, "law": d0.law,
                        "data": d0.data, "scale": DRIVER_SCALE[d0.quantity]})
    # order of the drivers in the results = kinematic, then force (as the worker)
    for d0 in sorted(mech.drivers, key=lambda x: x.quantity not in KINEMATIC_QUANTITIES):
        coord = "rot" if d0.quantity in ("angle", "angular_velocity", "torque") else "trans"
        md.append({"name": d0.name or d0.joint, "joint": jbi[d0.joint], "joint_name": d0.joint,
                   "quantity": d0.quantity, "coord": coord, "kinematic": d0.quantity in KINEMATIC_QUANTITIES})
    loads, ml = [], []
    for ld in mech.loads:
        b = bi[ld.body]
        if bodies[b].ground:
            continue
        loads.append({"name": ld.name, "body": b, "point": list(np.asarray(ld.point, float) * MM),
                      "force": list(map(float, ld.force)), "moment": list(np.asarray(ld.moment, float) * NMM),
                      "table": ld.table})
        ml.append({"name": ld.name or f"load {len(ml) + 1}", "body": b, "body_name": ld.body,
                   "selection": ld.selection})
    g = None if mech.gravity is None else list(np.asarray(mech.gravity, float) * MM)
    job = {"kind": kind or run.kind, "t_end": float(run.t_end), "outputs": int(run.outputs), "dt": run.dt,
           "spectral_radius": run.spectral_radius, "gravity": g, "bodies": jb, "joints": joints, "forces": forces,
           "drivers": drivers, "loads": loads}
    meta = {"kind": job["kind"], "gravity": mech.gravity,
            "bodies": [{"name": b.name, "ground": bool(b.ground), "mass": props[b.name]["mass"],
                        "com": list(map(float, props[b.name]["com"])),
                        "inertia": np.asarray(props[b.name]["inertia"], float).tolist(),
                        "parts": list(props[b.name]["parts"]), "extra": props[b.name]["extra"],
                        "printed": props[b.name]["printed"], "source": props[b.name]["source"],
                        "flexible": bool(b.flexible and not b.ground)} for b in bodies],
            "joints": mj, "forces": mf, "drivers": md, "loads": ml,
            "warnings": list(props.get("_warnings", [])), "mass_check": props.get("_check", {})}
    return job, meta, props


# ---------------------------------------------------------------- working process
def _run_worker(jd: Path, on_log=None, cancel=None, on_progress=None, timeout: float | None = None,
                chrono: bool = False) -> int | None:
    """Working process of Exudyn (Python venv) or Chrono (the interpreter tools/chrono_env, PYTHONPATH = root)."""
    log = jd / "worker.log"
    seen = 0
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if chrono:
        from .chrono_run import chrono_python
        cmd = [str(chrono_python()), "-m", "fea_core.motion.chrono_worker", str(jd)]
        env["PYTHONPATH"] = str(config.PROJECT_ROOT)
        for k in ("PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(k, None)
    else:
        cmd = [sys.executable, "-m", "fea_core.motion.worker", str(jd)]
    with open(log, "w", encoding="utf-8", errors="replace") as out:
        p = subprocess.Popen(cmd, cwd=config.PROJECT_ROOT, env=config.child_env(env), stdout=out,
                             stderr=subprocess.STDOUT, creationflags=config.NO_WINDOW)
        t0 = time.monotonic()
        while p.poll() is None:
            if cancel is not None and cancel.is_set():
                p.kill()
                p.wait()
                return None
            if timeout is not None and time.monotonic() - t0 > timeout:
                p.kill()
                p.wait()
                raise MotionError(f"The mechanism analysis exceeded the time limit of {timeout:g} s")
            seen = _pump(log, seen, on_log, on_progress)
            time.sleep(0.1)
    _pump(log, seen, on_log, on_progress)
    return p.returncode


def _pump(log: Path, seen: int, on_log, on_progress) -> int:
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return seen
    lines = text.splitlines()
    done = len(lines) if text.endswith("\n") else max(len(lines) - 1, 0)
    for line in lines[seen:done]:
        if line.startswith("progress") and on_progress is not None:
            try:
                on_progress(float(line.split()[1]) / 100.0, 1.0, line)      # as fea_core.progress
            except (ValueError, IndexError):
                pass
        elif on_log is not None and line.strip() and not line.startswith(("WARNING:", "  ")):
            on_log(line)
    return max(seen, done)


def check_dof(study, parts=None, study_dir=None, workdir=None) -> dict:
    """Check of the freedom: Grübler, rank of the Jacobian of the constraints (Exudyn), redundant constraints, drivers.

    Returns {grubler, dof_free, dof_driven, redundant, redundant_list, kinematic_drivers, ok, messages}.
    """
    errs, warns = check_model(study, parts, run=False)
    if errs:
        raise MotionError("Mechanism is incomplete:\n  " + "\n  ".join(errs))
    parts = _load_parts(study, parts, study_dir)
    import tempfile
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="fea_motion_"))
    jd = wd / "motion_check"
    jd.mkdir(parents=True, exist_ok=True)
    job, meta, _ = prepare_job(study, parts, jd=jd)
    job["stage"] = "check"
    (jd / "job.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    (jd / "result.json").unlink(missing_ok=True)
    rc = _run_worker(jd, timeout=120)
    res_file = jd / "check.json"
    if rc != 0 or not res_file.exists():
        try:
            err = json.loads((jd / "result.json").read_text(encoding="utf-8")).get("error")
        except (OSError, ValueError):
            err = (jd / "worker.log").read_text(encoding="utf-8", errors="replace")[-800:]
        raise MotionError(f"Kontrola volnosti selhala: {err}")
    d = json.loads(res_file.read_text(encoding="utf-8"))
    gr = grubler(study)
    return dof_report(gr, d, study, warns + meta["warnings"])


def dof_report(gr: dict, d: dict, study, warns: list[str]) -> dict:
    free, drv = d["free"], d["driven"]
    kin = gr["kinematic_drivers"]
    msgs = list(warns)
    red_list = free.get("redundant_list", [])
    out = {"grubler": gr["dof"], "dof_free": free["dof"], "dof_driven": drv["dof"], "redundant": free["redundant"],
           "redundant_list": red_list, "kinematic_drivers": kin, "messages": msgs}
    if gr["dof"] != free["dof"]:
        msgs.append(f"Grübler gives {gr['dof']} degrees of freedom, the constraint rank {free['dof']} – {free['redundant']} redundant constraints (typically a planar mechanism made of revolute joints only in 3D)")
    if free["redundant"]:
        joints = sorted({r["joint"] for r in red_list})
        msgs.append(f"Redundant constraints → joint reactions are AMBIGUOUS (the split between joints of rigid bodies is indeterminate; the sum is valid). Joints: {', '.join(joints) or '?'}. Remedies: (a) a non-redundant joint set (revolute + ball / cylindrical), (b) a compliant joint (bushing), (c) when transferring to FEA, give the redundant joints as supports (ToFea.support_joints) – FEA shares the load.")
    run_kind = study.mechanism.run.kind
    if run_kind == "kinematic" and drv["dof"] > 0:
        msgs.append(f"Drive missing: {drv['dof']} degrees of freedom remain free after the drives (a kinematic analysis must prescribe all)")
    if kin and drv["dof"] == free["dof"] - kin + (drv["redundant"] - free["redundant"]) and drv["redundant"] > free["redundant"]:
        msgs.append("The drive conflicts with the constraints (redundant drive) – it prescribes a coordinate already fixed by the joints")
    out["ok"] = not (run_kind == "kinematic" and drv["dof"] > 0) and drv["redundant"] <= free["redundant"]
    return out


# ---------------------------------------------------------------- calculation
def run_motion(study, workdir, parts=None, study_dir=None, on_log=None, cancel=None, on_progress=None,
               timeout: float | None = None) -> MotionResult:
    """Runs the calculation of the mechanism (study.mechanism.run) in the working process and returns a MotionResult.
    on_log(text) = messages of the worker; on_progress(value 0–1, end of phase 0–1, text) as fea_core.progress;
    cancel = threading.Event (ends the process); timeout [s].
    """
    t0 = time.time()
    errs, warns = check_model(study, parts)
    if errs:
        raise MotionError("Mechanism is incomplete:\n  " + "\n  ".join(errs))
    parts = _load_parts(study, parts, study_dir)
    wd = Path(workdir)
    jd = wd / "motion"
    jd.mkdir(parents=True, exist_ok=True)
    job, meta, props = prepare_job(study, parts, jd=jd, log=on_log)
    meta["warnings"] = warns + meta["warnings"]
    for name in ("result.json", "result.npz", "check.json"):
        (jd / name).unlink(missing_ok=True)
    chrono = study.mechanism.run.solver == "chrono"
    dof_chk = None
    if chrono:
        # the freedom and redundancy of the constraints are computed by Exudyn (the same model without contact), then the calculation in Chrono
        (jd / "job.json").write_text(json.dumps({**job, "stage": "check"}, ensure_ascii=False), encoding="utf-8")
        try:
            if _run_worker(jd, timeout=120) == 0 and (jd / "check.json").exists():
                dof_chk = json.loads((jd / "check.json").read_text(encoding="utf-8"))
        except MotionError:
            dof_chk = None
        from .chrono_run import prepare_contacts
        try:
            prepare_contacts(study, parts, job, meta, jd)
        except ValueError as e:
            raise MotionError(str(e)) from None
        # triangle mesh with the envelope r = 0.05 mm (sphere-swept; vertices shifted inward by r → the surface = the CAD): contact
        # with a mesh without the envelope (GIMPACT) a cam stuck after ~50° and penetrated (scratch/f8_motion/dbg_kk5.py); detection 0.5 mm
        job["chrono"] = {"envelope": 5e-4, "margin": 5e-5, "radius": 5e-5}
    (jd / "job.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    if on_log:
        on_log(f"Mechanism: {len(job['bodies'])} bodies, {len(job['joints'])} joints, {len(job['drivers'])} drives – { {'kinematic': 'kinematic', 'static': 'quasi-static', 'dynamic': 'dynamic'}[job['kind']]} problem"
               + (f", Chrono solver, {len(job.get('contacts') or [])} contacts" if chrono else ""))
    rc = _run_worker(jd, on_log, cancel, on_progress, timeout, chrono=chrono)
    if rc is None:
        raise MotionError("Analysis canceled by user")
    try:
        info = json.loads((jd / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        tail = (jd / "worker.log").read_text(encoding="utf-8", errors="replace")[-1500:]
        raise MotionError(f"The mechanism worker ended without a result (code {rc}):\n{tail}") from None
    if not info.get("ok"):
        raise MotionError(info.get("error") or "Mechanism analysis failed")
    with np.load(jd / "result.npz") as z:
        raw = {k: z[k] for k in z.files}
    arrays = to_project_units(raw, meta)
    if props.get("_flex"):
        from .flex import finish
        finish(arrays, meta, raw, props["_flex"])
    gr = grubler(study)
    if chrono:
        meta["solver"] = "chrono"
        meta["contacts_max"] = info.get("contacts_max")
        info["dof"] = dof_chk or {"free": {"dof": gr["dof"], "redundant": 0},
                                  "driven": {"dof": gr["dof"] - gr["kinematic_drivers"], "redundant": 0}}
    meta["dof"] = dof_report(gr, info["dof"], study, [])
    meta["warnings"] += meta["dof"]["messages"] + info.get("notes", [])
    meta["time_s"] = time.time() - t0
    meta["step_s"] = info.get("h")
    res = MotionResult(arrays, meta)
    res.meta["equilibrium"] = equilibrium(res)
    if not res.meta["equilibrium"]["ok"]:
        msg = (f"Body equilibrium check: max. imbalance {res.meta['equilibrium']['max_rel']:.2e} (body '{res.meta['equilibrium']['body']}', t = {res.meta['equilibrium']['t']:.4g} s)")
        if chrono:
            # Euler implicit (Chrono): the contact forces in the position at the start of the step, the accelerations from the difference of the velocities →
            # mismatch of the order h (MB4C: h 2.5e-4 → 2 %, h 5e-5 → 0.7 %), only a warning
            res.warnings.append(msg + " – Chrono solver (implicit Euler, error proportional to the step dt); shorten dt")
        elif res.meta["equilibrium"]["max_rel"] > 100 * config.EQUILIBRIUM_TOL:
            raise MotionError(msg + " – the result is wrong")
        else:
            res.warnings.append(msg)
    if job["kind"] != "static" and not props.get("_flex") and not chrono:      # Chrono: the contact and the Euler damp
        res.meta["power_balance"] = power_balance(res)
        if not res.meta["power_balance"]["ok"]:
            res.warnings.append(f"The power balance does not match (max. rel. {res.meta['power_balance']['max_rel']:.2e}) – shorten the step dt or refine the output")
    rc_ = study.mechanism.run
    if rc_.collisions:
        if parts is None and str(study.geometry or "").strip():
            from ..geometry import legacy_step_names, read_step
            parts = read_step(study.geometry_path(study_dir), legacy_names=legacy_step_names(study))
        if parts is not None:
            from .collisions import check_collisions
            col = check_collisions(study, parts, res, rc_.collisions, rc_.collision_tol)
            res.meta["collisions"] = col
            res.meta["collisions_summary"] = {k: v for k, v in col.items() if k not in ("t", "pairs")}
            res.meta["collisions_summary"]["pairs"] = [{k: v for k, v in p.items() if k != "distance"}
                                                       for p in col["pairs"]]
            if col.get("first"):
                f = col["first"]
                res.warnings.append(f"Collision of bodies '{f['a']}' and '{f['b']}' at t = {f['t']:.4g} s (check without contact – the bodies passed through each other in MBD)")
    res.save(jd / "results_motion.npz")
    if on_progress is not None:
        on_progress(1.0, 1.0, "Mechanism finished")
    return res


# ---------------------------------------------------------------- conversion of the units
def to_project_units(raw: dict, meta: dict) -> dict:
    """SI → mm, N, N·mm, t (outputs of the worker)."""
    a = {}
    k1 = 1e3
    N = len(raw["t"])
    for key in ("pos", "vel", "acc", "jP", "jPa", "fPa", "fPb", "lP"):
        a[key] = raw[key] * k1
    for key in ("rot", "omega", "alpha", "jF", "jFl", "fF", "lF", "t"):
        a[key] = raw[key] if key == "t" else raw[key] * 1.0
    for key in ("jM", "jMl", "fM", "lM"):
        a[key] = raw[key] * k1
    for key in ("ekin", "epot", "wdrv", "wfr", "wlim", "wdamp", "wbush", "wload", "pdrv", "pfr", "plim", "pdamp",
                "pbush", "pload", "pelem"):
        a[key] = raw[key] * k1                    # J → N·mm (mJ), W → N·mm/s
    nj = len(meta["joints"])
    sc = np.array([k1, 1.0])                      # [rot: N·m → N·mm, trans: N]
    a["jq"] = raw["jq"].reshape(N, nj, 2) * np.array([1.0, k1])
    a["jv"] = raw["jv"].reshape(N, nj, 2) * np.array([1.0, k1])
    a["jfr"] = raw["jfr"].reshape(N, nj, 2) * sc
    prim = np.array([k1 if j.get("primary") == "rot" else 1.0 for j in meta["joints"]])
    a["jlim"] = raw["jlim"].reshape(N, nj) * prim
    nd = len(meta["drivers"])
    dsc = np.array([k1 if d["coord"] == "rot" else 1.0 for d in meta["drivers"]])
    qsc = np.array([1.0 if d["coord"] == "rot" else k1 for d in meta["drivers"]])
    a["dQ"] = raw["dQ"].reshape(N, nd) * dsc
    a["dq"] = raw["dq"].reshape(N, nd) * qsc
    a["dv"] = raw["dv"].reshape(N, nd) * qsc
    nf = len(meta["forces"])
    tors = np.array([f["kind"] == "torsion" for f in meta["forces"]], bool)
    a["fval"] = raw["fval"].reshape(N, nf) * np.where(tors, k1, 1.0)
    a["flen"] = raw["flen"].reshape(N, nf) * np.where(tors, 1.0, k1)
    nb = len(meta["bodies"])
    for key in ("pos", "vel", "acc", "omega", "alpha"):
        a[key] = a[key].reshape(N, nb, 3)
    a["rot"] = a["rot"].reshape(N, nb, 3, 3)
    for key in ("jF", "jM", "jP", "jPa", "jFl", "jMl"):
        a[key] = a[key].reshape(N, nj, 3)
    for key in ("fF", "fM", "fPa", "fPb"):
        a[key] = a[key].reshape(N, nf, 3)
    nl = len(meta["loads"])
    for key in ("lF", "lM", "lP"):
        a[key] = a[key].reshape(N, nl, 3)
    if "cF" in raw:                               # contact (Chrono): resultant on the body, moment about the centroid
        a["cF"] = raw["cF"].reshape(N, nb, 3) * 1.0
        a["cM"] = raw["cM"].reshape(N, nb, 3) * k1
    return a


# ---------------------------------------------------------------- checks
def body_wrenches(res: MotionResult, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ΣF, ΣM about the centroid, scale) of the external forces on the body at all times (without the inertia)."""
    a, m = res.arrays, res.meta
    N = len(a["t"])
    c = a["pos"][:, i]
    B = m["bodies"][i]
    g = np.asarray(m.get("gravity") or [0, 0, 0], float)
    F = np.tile(B["mass"] * g, (N, 1))
    M = np.zeros((N, 3))
    scale = np.full(N, B["mass"] * np.linalg.norm(g))
    mscale = np.zeros(N)
    for j, J in enumerate(m["joints"]):
        for side, sg in (("b", 1.0), ("a", -1.0)):
            if J[side] != i:
                continue
            Fj = a["jF"][:, j]
            if side == "b":
                P, Mj = a["jP"][:, j], a["jM"][:, j]
            else:
                P = a["jPa"][:, j]
                Mj = -a["jM"][:, j] - np.cross(a["jP"][:, j] - a["jPa"][:, j], Fj)
                Fj = -Fj
            F += Fj
            M += np.cross(P - c, Fj) + Mj
            scale += np.linalg.norm(Fj, axis=1)
            mscale += np.linalg.norm(Mj, axis=1) + np.linalg.norm(P - c, axis=1) * np.linalg.norm(Fj, axis=1)
    for k, f in enumerate(m["forces"]):
        for side, sg in (("b", 1.0), ("a", -1.0)):
            if f[side] != i:
                continue
            Ff, Mf = a["fF"][:, k], a["fM"][:, k]
            if side == "b":
                P = a["fPb"][:, k]
            else:
                P = a["fPa"][:, k]
                Mf = -Mf - np.cross(a["fPb"][:, k] - a["fPa"][:, k], Ff)
                Ff = -Ff
            F += Ff
            M += np.cross(P - c, Ff) + Mf
            scale += np.linalg.norm(Ff, axis=1)
            mscale += np.linalg.norm(Mf, axis=1) + np.linalg.norm(P - c, axis=1) * np.linalg.norm(Ff, axis=1)
    if "cF" in a:                                 # contact force (Chrono) – at the centroid + moment about the centroid
        F += a["cF"][:, i]
        M += a["cM"][:, i]
        scale += np.linalg.norm(a["cF"][:, i], axis=1)
        mscale += np.linalg.norm(a["cM"][:, i], axis=1)
    for k, ld in enumerate(m["loads"]):
        if ld["body"] != i:
            continue
        F += a["lF"][:, k]
        M += np.cross(a["lP"][:, k] - c, a["lF"][:, k]) + a["lM"][:, k]
        scale += np.linalg.norm(a["lF"][:, k], axis=1)
        mscale += (np.linalg.norm(a["lM"][:, k], axis=1)
                   + np.linalg.norm(a["lP"][:, k] - c, axis=1) * np.linalg.norm(a["lF"][:, k], axis=1))
    return F, M, np.column_stack([scale, mscale])


def inertia_wrench(res: MotionResult, i: int) -> tuple[np.ndarray, np.ndarray]:
    """(m·a, J·α + ω × J·ω) of the body also in the global axes."""
    a, B = res.arrays, res.meta["bodies"][i]
    R = a["rot"][:, i]
    J0 = np.asarray(B["inertia"], float)
    Jg = np.einsum("nij,jk,nlk->nil", R, J0, R)
    w, al = a["omega"][:, i], a["alpha"][:, i]
    Jw = np.einsum("nij,nj->ni", Jg, w)
    return B["mass"] * a["acc"][:, i], np.einsum("nij,nj->ni", Jg, al) + np.cross(w, Jw)


def equilibrium(res: MotionResult) -> dict:
    """Equilibrium of each body at each output time (d'Alembert)."""
    worst = {"max_rel": 0.0, "body": None, "t": 0.0, "force_rel": 0.0, "moment_rel": 0.0}
    for i, B in enumerate(res.meta["bodies"]):
        if B["ground"] or B.get("flexible"):          # flexible body: frame ≠ centroid, the inertia of the shapes – not checked
            continue
        F, M, sc = body_wrenches(res, i)
        ma, dh = inertia_wrench(res, i)
        sF = np.maximum(sc[:, 0] + np.linalg.norm(ma, axis=1), 1e-300)
        # scale of the moments: Σ|r|·|F| + Σ|M| + |dH/dt|, at least the scale of the forces × 1 mm (forces at the centroid, zero moments);
        # both scales at least 1e-3 × their maximum during the run (the instants when almost nothing acts on the body)
        sM = np.maximum(sc[:, 1] + np.linalg.norm(dh, axis=1), sF * 1.0)
        sF = np.maximum(sF, 1e-3 * sF.max())
        sM = np.maximum(sM, 1e-3 * sM.max())
        eF = np.linalg.norm(F - ma, axis=1) / sF
        eM = np.linalg.norm(M - dh, axis=1) / sM
        k = int(np.argmax(np.maximum(eF, eM)))
        e = float(max(eF[k], eM[k]))
        if e > worst["max_rel"] or worst["body"] is None:
            worst = {"max_rel": e, "body": B["name"], "t": float(res.arrays["t"][k]), "force_rel": float(eF.max()),
                     "moment_rel": float(eM.max())}
    worst["ok"] = worst["max_rel"] <= config.EQUILIBRIUM_TOL
    worst["tolerance"] = config.EQUILIBRIUM_TOL
    return worst


def power_balance(res: MotionResult) -> dict:
    """Power balance at each output time: P_drivers + P_loads + P_friction + P_stops + P_elements =
    dE_k/dt + dE_weight/dt (dE_k/dt = Σ m·v·a + ω·J·α, dE_weight/dt = −Σ m·g·v) and integrally ΔE = Σ work.
    """
    a, m = res.arrays, res.meta
    g = np.asarray(m.get("gravity") or [0, 0, 0], float)
    dE = np.zeros(len(a["t"]))
    sc = np.zeros(len(a["t"]))
    for i, B in enumerate(m["bodies"]):
        if B["ground"]:
            continue
        R = a["rot"][:, i]
        Jg = np.einsum("nij,jk,nlk->nil", R, np.asarray(B["inertia"], float), R)
        w = a["omega"][:, i]
        t1 = B["mass"] * np.einsum("ni,ni->n", a["vel"][:, i], a["acc"][:, i])
        t2 = np.einsum("ni,nij,nj->n", w, Jg, a["alpha"][:, i])
        t3 = -B["mass"] * a["vel"][:, i] @ g
        dE += t1 + t2 + t3
        sc += np.abs(t1) + np.abs(t2) + np.abs(t3)
    keys = ("pdrv", "pload", "pfr", "plim", "pelem")
    P = sum(a[k] for k in keys)
    sc += sum(np.abs(a[k]) for k in keys)
    err = np.abs(P - dE) / np.maximum(sc.max(), 1e-300)
    k = int(np.argmax(err))
    out = {"max_rel": float(err[k]), "t": float(a["t"][k]), "ok": bool(err[k] <= 1e-3),
           "work_mJ": {kk: float(a[kk][-1]) for kk in ("wdrv", "wfr", "wlim", "wdamp", "wbush", "wload")},
           "dE_kin_mJ": float(a["ekin"][-1] - a["ekin"][0]), "dE_pot_mJ": float(a["epot"][-1] - a["epot"][0])}
    W = sum(a[kk] for kk in ("wdrv", "wfr", "wlim", "wdamp", "wbush", "wload"))
    E = a["ekin"] + a["epot"]
    escale = max(np.abs(a["ekin"]).max(), np.abs(E - E[0]).max(), *(np.abs(a[kk]).max() for kk in
                 ("wdrv", "wfr", "wlim", "wdamp", "wbush", "wload")), 1e-300)
    out["integral_rel"] = float(np.abs((E - E[0]) - W).max() / escale)
    if a["wfr"][-1] != 0 and a["wdrv"][-1] > 0:
        out["efficiency"] = float(1.0 + a["wfr"][-1] / a["wdrv"][-1])
    return out
