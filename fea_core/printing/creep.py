"""Estimate of the creep of printed parts (F4) – isochronous (secant) modulus in linear elasticity.

The long-term load states (PrintSettings.creep.cases) are computed AGAIN on the same mesh with the reduced
stiffnesses of the printed materials E_c = k_E(T)·E/(1 + k_φ(T)·φ(t)) (all moduli E, E_z, G_z; ν unchanged)
and the strengths × k_R(t) (creep rupture) – parameters of printing.props. The result = further steps
"<state> – creep <t> h, <T> °C" (displacements, stresses, safety of the printing against the long-term strength).

Limits of the method (to be stated in the report): linear viscoelasticity (up to ~30–35 % of the strength), constant load
over the whole time, no relaxation of the bolt preload in plastic, no moisture and ageing, uniform temperature.
Metal parts of the assembly do not creep (the redistribution of forces between the stiff and creeping parts is captured). With the same
ratio of stiffnesses of all parts the displacement of the long-term state is only the short-term one multiplied (stresses the same).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

STIFF = ("E", "E_z", "G_z")
STRENGTH = ("Rm", "Rm_z", "Rc", "Rc_z", "S_xy", "S_z")


def sustained_cases(study) -> list:
    cs = study.printing.creep
    if not cs.enabled:
        return []
    return [lc for lc in study.load_cases if lc.name in set(cs.cases)]


def check(study, printed: list[str]) -> tuple[list[str], list[str]]:
    """Check of the creep settings (errors, warnings)."""
    from .props import props_of
    errors, warnings = [], []
    cs = study.printing.creep
    if not cs.enabled or not printed:
        return errors, warnings
    names = {lc.name for lc in study.load_cases}
    missing = [c for c in cs.cases if c not in names]
    if missing:
        warnings.append("Creep: unknown load cases " + ", ".join(missing))
    if not sustained_cases(study):
        warnings.append("Creep is enabled, but no load case is marked as long-term")
    if cs.hours <= 0:
        errors.append("Creep: load duration must be positive")
    for mname in sorted({study.material_of(p).name for p in printed}):
        pr = props_of(study, study.materials[mname])
        fac = pr.creep_factors(cs.hours, cs.temperature)
        if not fac["valid"]:
            errors.append(f"Creep '{mname}': temperature {cs.temperature:g} °C is outside the validity of the estimate (up to {fac['T_max']:g} °C, near the glass transition) – the linear model cannot be used")
        if pr.creep_a <= 0:
            warnings.append(f"Creep '{mname}': the material has no creep parameters – the modulus will not be reduced")
    return errors, warnings


def creep_study(study):
    """Copy of the study for the long-term states: only these states, printed materials with reduced stiffness and strength.
    Returns (study, {material: factors}) or None.
    """
    from ..study import Study
    from .props import props_of
    cases = sustained_cases(study)
    if not cases:
        return None
    cs = study.printing.creep
    st = Study.from_dict(study.to_dict())
    st.printing.creep.enabled = False
    keep = {lc.name for lc in cases}
    st.load_cases = [lc for lc in st.load_cases if lc.name in keep]
    st.fatigue = None
    factors = {}
    for name, m in list(st.materials.items()):
        if not m.is_print:
            continue
        fac = props_of(study, study.materials[name]).creep_factors(cs.hours, cs.temperature)
        factors[name] = fac
        ks, kr = fac["stiffness"], fac["strength"]
        upd = {k: getattr(m, k) * ks for k in STIFF if getattr(m, k) is not None}
        upd.update({k: getattr(m, k) * kr for k in STRENGTH if getattr(m, k) is not None})
        st.materials[name] = replace(m, **upd)
    return st, factors


def step_name(case: str, hours: float, T: float) -> str:
    return f"{case} – creep {hours:g} h, {T:g} °C"


def start_creep(study, mesh, thermal, workdir, log, on_solver_line=None, cancel=None):
    """Long-term states in the background (in parallel with the mechanics – the same mesh only for reading, the printing signature does not change,
    the materials are not in the signature). Returns solve.Background or None (creep switched off / no long-term states).
    """
    made = creep_study(study)
    if made is None:
        return None
    from ..solve import Background, run_study
    st_c, _ = made
    cs = study.printing.creep
    log(f"3D printing: creep – long-term cases ({cs.hours:g} h, {cs.temperature:g} °C) in parallel")
    th = list(thermal)
    return Background(lambda stop, w: run_study(st_c, workdir / "creep", on_log=log, on_solver_line=on_solver_line,
                                                cancel=stop, mesh=mesh, thermal=th), cancel, "creep")


def add_creep_steps(study, mesh, res, workdir, log, on_solver_line=None, cancel=None, pending=None) -> None:
    """Computes the long-term states (the same mesh) and attaches them to res as further steps + a summary in info.
    pending = calculation started earlier (start_creep) → it only waits for the result.
    """
    made = creep_study(study)
    if made is None:
        return
    from ..solve import run_study
    from .props import props_of
    st_c, factors = made
    cs = study.printing.creep
    if pending is not None:
        rc, _ = pending.result()
    else:
        log(f"3D printing: creep – long-term cases ({cs.hours:g} h, {cs.temperature:g} °C)")
        rc = run_study(st_c, workdir / "creep", on_log=log, on_solver_line=on_solver_line, cancel=cancel, mesh=mesh,
                       thermal=list(res.thermal))
    base = {s.name: s for s in res.steps}
    summary = []
    for s in rc.steps[:len(st_c.load_cases)]:
        b = base.get(s.name)
        info = {"case": s.name, "hours": cs.hours, "temperature": cs.temperature,
                "factors": {k: {x: v[x] for x in ("phi", "k_E", "stiffness", "strength")} for k, v in factors.items()}}
        s.printing = dict(s.printing or {}, creep=info)
        row = dict(info, u_short=float(b.umag.max()) if b is not None else None, u_long=float(s.umag.max()))
        if b is not None and row["u_short"] > 0:
            row["ratio"] = row["u_long"] / row["u_short"]
        lam = [d["min_safety"] for d in (s.printing.get("parts") or {}).values()]
        row["min_safety_long"] = float(min(lam)) if lam else None
        # linear viscoelasticity only up to ~1/3 of the strength (short-term utilisation of the state)
        lam0 = [d["min_safety"] for d in ((b.printing or {}).get("parts") or {}).values()] if b is not None else []
        if lam0:
            util = 1.0 / min(lam0)
            row["utilization_short"] = util
            lim = min(props_of(study, study.materials[k]).creep_max_util for k in factors)
            if util > lim:
                res.warnings.append(
                    f"Creep '{s.name}': short-term print utilization {100 * util:.0f} % > {100 * lim:.0f} % – outside linear viscoelasticity, actual creep will be larger (the estimate is non-conservative).")
        s.name = step_name(s.name, cs.hours, cs.temperature)
        res.steps.append(s)
        summary.append(row)
    for w in rc.warnings:
        if w not in res.warnings:
            res.warnings.append(w)
    pinfo = res.info.setdefault("printing", {})
    pinfo["creep"] = {"hours": cs.hours, "temperature": cs.temperature, "cases": summary,
                      "method": "isochronous modulus E/(1 + φ), Findley φ = a·t^n, temperature factors; strength × k_R (creep rupture) – INDICATIVE"}
    for row in summary:
        if row["u_short"] is not None:
            log(f"  {row['case']}: max. displacement {row['u_short']:.4g} → {row['u_long']:.4g} mm "
                f"(×{row.get('ratio', np.nan):.3g})")
