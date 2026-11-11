"""Mesh convergence: the same study on successively finer meshes, the monitored quantity, Richardson extrapolation and GCI.

Refinement: globally (h_k = h₀ / r^k, also local sizes) or by a sphere around a point (MeshSettings.local with center/radius,
Gmsh field "Ball"; outside the sphere the original mesh). Quantities: max. von Mises / displacement (globally or in the sphere), value at a point
(probe), natural frequency of a mode, safety of 3D printing (minimum). Three or more levels with constant r:
  p = ln(|f₁ − f₂| / |f₂ − f₃|) / ln r,   f_ext = f₃ + (f₃ − f₂)/(r^p − 1),   GCI = 1.25·|(f₃ − f₂)/f₃|/(r^p − 1)
(Roache 1998, ASME V&V 20). A monotonically increasing value with p < 0.5 (or non-decreasing differences) → suspicion of a
singularity (sharp inner corner, clamped edge, force at a point/edge) – the stress there does not settle with refinement.
Additionally an estimate of the ccx error (field stress_error, STR % – gradient of the principal stress in neighbouring elements).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class ConvergenceSpec:
    quantity: str = "max_mises"        # max_mises | max_disp | probe | frequency | print_safety
    field: str = "mises"               # sonda: mises | s1 | sx… | ux uy uz umag
    point: list | None = None          # probe / centre of the sphere
    radius: float | None = None        # sphere: refinement + evaluation of the maximum inside (None = global)
    step: int = 0                      # load state (index) / mode for frequency
    levels: int = 3
    ratio: float = 1.5                 # ratio of the element sizes between the levels
    tol: float = 0.02                  # relative change of the last refinement for "converged"


def _value(res, spec: ConvergenceSpec) -> tuple[float, list | None]:
    q = spec.quantity
    if q == "frequency":
        modal = [m for m in res.modes if m.kind == "modal"]
        if not modal:
            raise ValueError("Frequency convergence: the study has natural frequencies disabled")
        return float(modal[0].values[spec.step]), None
    st = res.step(spec.step)
    X = res.mesh.nodes
    if q == "probe":
        if spec.point is None:
            raise ValueError("Convergence at a point: the point is missing")
        from .paths import line_values
        v = line_values(res, spec.point, spec.point, spec.field, spec.step, n=2)["values"][0]
        return float(v), list(map(float, spec.point))
    if q == "print_safety":
        vals = st.fields.get("print_safety")
        if vals is None:
            raise ValueError("Print safety convergence: the study has no printed parts")
        sel = _region(X, spec)
        i = sel[int(np.argmin(vals[sel]))]
        return float(vals[i]), X[i].tolist()
    vals = st.mises if q == "max_mises" else st.umag
    sel = _region(X, spec)
    i = sel[int(np.argmax(vals[sel]))]
    return float(vals[i]), X[i].tolist()


def _region(X, spec) -> np.ndarray:
    if spec.radius and spec.point is not None:
        sel = np.nonzero(np.linalg.norm(X - np.asarray(spec.point, float), axis=1) <= spec.radius)[0]
        if len(sel):
            return sel
    return np.arange(len(X))


def richardson(values: list[float], ratio: float) -> dict:
    """Order of convergence, extrapolated value and GCI from the last three levels (constant ratio r)."""
    out = {"order": None, "extrapolated": None, "gci": None, "monotonic": None}
    if len(values) < 3:
        return out
    f1, f2, f3 = values[-3:]
    d1, d2 = f2 - f1, f3 - f2
    out["monotonic"] = bool(d1 * d2 > 0)
    if not out["monotonic"] or abs(d2) < 1e-300:
        if abs(d2) <= 1e-12 * max(abs(f3), 1e-300):
            out.update(order=float("inf"), extrapolated=f3, gci=0.0)
        return out
    p = math.log(abs(d1 / d2)) / math.log(ratio)
    out["order"] = p
    if p > 0:
        out["extrapolated"] = f3 + d2 / (ratio ** p - 1)
        out["gci"] = 1.25 * abs(d2 / f3) / (ratio ** p - 1) if f3 else None
    return out


def reduced_study(study, spec: ConvergenceSpec):
    """Copy of the study with only what the monitored quantity needs on every level → (study, spec with the step index into it).

    - natural frequency: no load cases and thermal cases (the modal job has its own empty step), modal on, buckling off;
    - other quantities: the load case spec.step and as many others as keep the automatic solver choice the same
      (config.PARDISO_MIN_CASES: NGSolve + PARDISO from that many cases – a different solver would change the last digits), plus the
      thermal cases these load cases take their temperature from; natural frequencies, buckling off;
    - always off: dynamics, print creep, print variants / sweep (not computed by run_study anyway).
    An automatic solver that would be ccx because of a feature of a dropped load case (rigid remote load) is fixed to ccx.
    Nonlinear studies (load steps refer to the cases) keep all load cases. Load cases are independent linear steps, so the values
    of the kept case do not change.
    """
    import dataclasses
    from . import config
    st = copy.deepcopy(study)
    vspec = dataclasses.replace(spec)
    st.buckling.enabled = False
    st.modal.enabled = study.modal.enabled and spec.quantity == "frequency"
    if getattr(st, "dynamics", None) is not None:
        st.dynamics.enabled = False
    pr = getattr(st, "printing", None)
    if pr is not None and getattr(pr, "creep", None) is not None:
        pr.creep.enabled = False
    if st.solver.backend == "auto":
        from .solve_large import unsupported
        if unsupported(study):                     # ccx for any mesh size (also because of a dropped load case)
            st.solver = dataclasses.replace(st.solver, backend="ccx")
    nonlinear = study.is_nonlinear() or bool(getattr(study.analysis, "sequence", None))
    if spec.quantity == "frequency":
        if st.modal.enabled and not nonlinear:
            st.load_cases = []
            st.thermal.cases = []
        return st, vspec
    n = len(study.load_cases)
    if nonlinear or not 0 <= int(spec.step) < n:
        return st, vspec
    keep_n = max(1, min(n, int(config.PARDISO_MIN_CASES)))
    k = int(spec.step)
    idx = [k] + [i for i in range(n) if i != k][:keep_n - 1]
    idx.sort()
    st.load_cases = [st.load_cases[i] for i in idx]
    vspec.step = idx.index(k)
    need = {t.case for lc in st.load_cases for t in st.thermal_loads(lc) if t.case}
    st.thermal.cases = [c for c in st.thermal.cases if c.name in need]
    return st, vspec


def run_convergence(study, workdir, spec: ConvergenceSpec, study_dir=None, parts=None, on_log=None,
                    cancel=None, on_level=None) -> dict:
    """Computes the study on spec.levels mesh levels. on_level(k, record) after each level (GUI)."""
    from .solve import run_study
    log = on_log or (lambda s: None)
    workdir = Path(workdir)
    h0 = float(study.mesh.size)
    rows = []
    base, vspec = reduced_study(study, spec)
    if base is not study:
        log(f"Convergence: computing {len(base.load_cases)} of {len(study.load_cases)} load cases, "
            f"{len(base.thermal.cases)} of {len(study.thermal.cases)} thermal cases (only what the quantity needs)")
    for k in range(int(spec.levels)):
        if cancel is not None and cancel.is_set():
            break
        st = copy.deepcopy(base)
        st.fatigue = study.fatigue
        f = spec.ratio ** k
        if spec.radius and spec.point is not None:
            h = h0 / f
            if k:
                st.mesh.local = list(st.mesh.local) + [{"center": list(map(float, spec.point)),
                                                        "radius": float(spec.radius), "size": h}]
        else:
            h = h0 / f
            st.mesh.size = h
            if st.mesh.min_size:
                st.mesh.min_size = st.mesh.min_size / f
            st.mesh.local = [dict(loc, size=float(loc["size"]) / f) for loc in st.mesh.local]
        log(f"Convergence: level {k + 1}/{spec.levels}, element {h:.3g} mm")
        res = run_study(st, workdir / f"uroven_{k + 1}", study_dir=study_dir, on_log=log, cancel=cancel,
                        parts=parts)
        v, where = _value(res, vspec)
        err = None
        if spec.quantity not in ("frequency",) and res.steps:
            se = res.step(vspec.step).fields.get("stress_error")
            if se is not None:
                err = float(np.max(se[_region(res.mesh.nodes, spec)]))
        row = {"level": k + 1, "size_mm": h, "nodes": int(len(res.mesh.nodes)), "elements": int(len(res.mesh.tets)),
               "value": v, "at": where, "stress_error_pct": err}
        if rows:
            row["change"] = (v - rows[-1]["value"]) / rows[-1]["value"] if rows[-1]["value"] else None
        rows.append(row)
        if on_level is not None:
            on_level(k, row)
    vals = [r["value"] for r in rows]
    rich = richardson(vals, spec.ratio)
    last = abs(rows[-1].get("change") or 0.0) if len(rows) > 1 else None
    out = {"spec": asdict(spec), "levels": rows, **rich,
           "converged": bool(last is not None and last <= spec.tol), "warnings": []}
    increasing = len(vals) >= 3 and all(b > a for a, b in zip(vals, vals[1:]))
    if spec.quantity in ("max_mises", "probe") and increasing and (rich["order"] is None or rich["order"] < 0.5):
        out["warnings"].append("The value keeps growing with refinement and the differences do not decrease – probably a singularity (sharp internal corner, fixed support edge, point / edge force); stress at that location has no finite value – evaluate away from it or model a fillet")
    if rows and spec.quantity in ("max_mises", "max_disp", "print_safety") and len({tuple(np.round(r["at"], 1))
                                                                               for r in rows if r["at"]}) > 1:
        moved = max(np.linalg.norm(np.subtract(rows[-1]["at"], r["at"])) for r in rows if r["at"])
        if moved > 2 * rows[-1]["size_mm"]:
            out["warnings"].append(f"The location of the maximum shifts between levels (up to {moved:.3g} mm) – track the quantity at a fixed point or in a sphere")
    if out["converged"]:
        out["verdict"] = f"Converged: the last refinement changed the value by {100 * last:.2g} % (limit {100 * spec.tol:g} %)"
    elif last is not None:
        out["verdict"] = f"Not converged: the last refinement changed the value by {100 * last:.2g} % (limit {100 * spec.tol:g} %)"
    else:
        out["verdict"] = "Only one level – convergence cannot be assessed"
    return out
