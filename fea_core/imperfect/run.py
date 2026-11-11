"""Nonlinear buckling with a geometric imperfection (F9 C2): GNIA (elastic) / GMNIA (with material plasticity).

Procedure (one extra ccx job after the statics, on a COPY of the mesh with the nodes moved by the imperfection):
  1. linear buckling of the chosen load case (taken from the results when present, otherwise an own *BUCKLE job with elastic
     materials) → λ_cr and the modes φ_k (normalised max |u| = 1);
  2. imperfection w = e0 · Σ c_k φ_k / max|Σ c_k φ_k| (or a half-sine bow / a nodal shape from a file), e0 in mm or L/xxx;
  3. *STEP, NLGEOM with the load of the case × λ_max ramped by the step time (load control, automatic increments) – the
     plasticity of the materials (*PLASTIC) stays → GMNIA; every increment: U of the monitored node (*NODE PRINT) and the fields
     (.frd) → load–displacement curve, ultimate factor (fea_core.imperfect.curve), deformed shapes at chosen increments.

Verified ccx 2.23 behaviour (scratch/feat_buckle, 11. 11. 2026): no arc-length / Riks (keywords of *STATIC: SOLVER, DIRECT,
EXPLICIT, TIME RESET, TOTAL TIME AT START; the string RIKS is not in the executable) → load control; at a limit point the increments
shrink until the minimum increment and ccx stops with an error – the last converged increment = ultimate factor (lower bound). The
descending (post-limit) branch is not traced.
"""
from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np

from .curve import STATUS_TEXT, evaluate
from .model import ImperfectionError, NonlinearBucklingSettings
from .shape import imperfect_mesh, imperfection_field, reference_length

STEP_NAME = "Nonlinear buckling"
KIND = "nlbuckle"                    # eigen.ModeSet.kind of the deformed shapes (frame 0 = imperfection)
WARN_AMPLITUDE = ("Nonlinear buckling: the imperfection shape and amplitude e0 are the user's responsibility – take them from the "
                  "design code (e.g. EN 1993-1-5 Annex C, EN 1993-1-6 8.7 / 8.8 by the fabrication quality class, EN 1993-1-1 5.3.2 "
                  "for members); the result depends strongly on them")
WARN_LOAD_CONTROL = ("Nonlinear buckling: load control (CalculiX has no arc-length / Riks method) – the ultimate factor is the "
                     "last converged increment (a lower bound of the limit load); the post-buckling (descending) branch is not traced")


def settings_of(study) -> NonlinearBucklingSettings:
    s = getattr(study, "nlbuckling", None)
    return s if s is not None else NonlinearBucklingSettings()


def case_index(study, s: NonlinearBucklingSettings) -> int:
    names = [lc.name for lc in study.load_cases]
    for c in (s.case, study.buckling.case if study.buckling.enabled else None):
        if c is not None and c in names:
            return names.index(c)
    return 0


def check(study, part_names) -> tuple[list[str], list[str]]:
    """Readiness check of Study.nlbuckling (errors, warnings)."""
    s = settings_of(study)
    errors, warnings = [], []
    if not s.enabled:
        return errors, warnings
    names = [lc.name for lc in study.load_cases]
    if not names:
        errors.append("Nonlinear buckling: the study has no load case (its loads are multiplied by the load factor λ)")
    elif s.case is not None and s.case not in names:
        errors.append(f"Nonlinear buckling: load case '{s.case}' does not exist")
    if not study.constraints:
        errors.append("Nonlinear buckling: the study has no supports")
    if getattr(study.frame, "members", None) or getattr(study.twod, "enabled", False) \
            or getattr(study.shell, "enabled", False):
        errors.append("Nonlinear buckling is available for solid (tet10) models only – not for beams, 2D or shells")
    if s.imperfection not in ("modes", "bow", "file", "none"):
        errors.append(f"Nonlinear buckling: unknown imperfection kind '{s.imperfection}'")
    if s.imperfection == "modes":
        if not s.modes or any(int(m) < 1 for m in s.modes):
            errors.append("Nonlinear buckling: the imperfection modes must be numbers ≥ 1")
    if s.imperfection == "bow" and (s.bow_a is None or s.bow_b is None or s.bow_direction is None):
        errors.append("Nonlinear buckling: the bow imperfection needs the points a, b and a direction")
    if s.imperfection == "file" and not s.shape_file:
        errors.append("Nonlinear buckling: the imperfection file is not given")
    if s.imperfection != "none" and not (float(s.amplitude or 0) > 0 or float(s.amplitude_ratio or 0) > 0):
        errors.append("Nonlinear buckling: give the amplitude e0 > 0 mm or the ratio L/xxx")
    if s.max_factor is not None and not float(s.max_factor) > 0:
        errors.append("Nonlinear buckling: the maximum load factor must be > 0")
    if not 2 <= int(s.increments) <= 10000:
        errors.append("Nonlinear buckling: the number of increments must be 2–10000")
    if s.imperfection == "none":
        warnings.append("Nonlinear buckling without imperfection: a perfect structure under load control may pass the bifurcation without buckling – use an imperfection")
    warnings.append(WARN_AMPLITUDE)
    return errors, warnings


def _elastic_copy(study):
    """Shallow copy with materials without plasticity (linear buckling of the imperfection shape)."""
    st = copy.copy(study)
    st.materials = {k: (dataclasses.replace(m, plastic=None) if getattr(m, "plastic", None) else m)
                    for k, m in study.materials.items()}
    return st


def linear_modes(study, mesh, lc, n_modes: int, workdir: Path, log, on_line=None, cancel=None, thermal=None,
                 modes=None):
    """(λ list, shapes (k, N, 3)) of the linear buckling of case lc: from `modes` (Results.modes) when they contain the case
    with enough modes, otherwise an own *BUCKLE job with elastic materials."""
    for ms in modes or []:
        if ms.kind == "buckling" and ms.case == lc.name and len(ms.values) >= n_modes:
            return [float(v) for v in ms.values], np.asarray(ms.shapes, float)
    from ..eigen import BucklingSettings, ModalSettings, run_eigen
    st = _elastic_copy(study)
    st.modal = ModalSettings()
    st.buckling = BucklingSettings(True, lc.name, int(max(n_modes, 1)))
    out = run_eigen(st, mesh, Path(workdir), log, on_line, cancel, thermal)
    ms = [m for m in out if m.kind == "buckling"][0]
    return [float(v) for v in ms.values], np.asarray(ms.shapes, float)


def _monitor(mesh, w, s):
    """(node index, unit direction)."""
    if s.monitor_point is not None:
        node = int(np.argmin(np.linalg.norm(mesh.nodes - np.asarray(s.monitor_point, float), axis=1)))
    else:
        node = int(np.argmax(np.linalg.norm(w, axis=1)))
    if s.monitor_dir is not None:
        d = np.asarray(s.monitor_dir, float)
    else:
        d = np.asarray(w[node], float)
    n = float(np.linalg.norm(d))
    if not n > 0:
        d, n = np.array([0.0, 0.0, 1.0]), 1.0
    return node, d / n


def _mises(S: np.ndarray) -> np.ndarray:
    sx, sy, sz, txy, tyz, tzx = (S[:, i] for i in range(6))
    return np.sqrt(np.maximum(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                              + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2), 0.0))


def _yield_strength(study, mesh) -> float | None:
    vals = []
    excl = study.excluded_parts()
    for p in mesh.part_names:
        if p in excl:
            continue
        try:
            re_ = study.material_of(p).Re
        except KeyError:
            continue
        if re_:
            vals.append(float(re_))
    return min(vals) if vals else None


def run_nlbuckling(study, mesh, workdir, log, on_line=None, cancel=None, thermal=None, modes=None):
    """→ (eigen.ModeSet kind "nlbuckle" – frame 0 = imperfection w, then displacements u at chosen increments (from the
    imperfect geometry), values = λ; info dict for Results.info["nlbuckling"]; warnings)."""
    from ..ccx.frd import read_frd
    from ..ccx.inp import write_inp
    from ..ccx.runner import CcxError, run_ccx
    from ..config import CCX_EXE
    from ..contacts.model import AnalysisSettings, LoadStep
    from ..contacts.results import ccx_messages, print_series
    from ..contacts.run import step_bounds, step_plans
    from ..eigen import BucklingSettings, ModalSettings, ModeSet
    s = settings_of(study)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = [WARN_AMPLITUDE, WARN_LOAD_CONTROL]
    if not study.load_cases:
        raise ImperfectionError("Nonlinear buckling: the study has no load case")
    ci = case_index(study, s)
    lc = study.load_cases[ci]
    need = max([int(m) for m in s.modes] or [1]) if s.imperfection == "modes" else 1
    log(f"Nonlinear buckling: linear buckling modes of case '{lc.name}'")
    lams, shapes = linear_modes(study, mesh, lc, max(need, 1), workdir / "linear", log, on_line, cancel, thermal, modes)
    pos = [v for v in lams if v > 0]
    lam_cr = min(pos) if pos else None
    w = imperfection_field(mesh, s, shapes if s.imperfection == "modes" else None)
    e0 = float(np.linalg.norm(w, axis=1).max()) if len(w) else 0.0
    mesh_i = imperfect_mesh(mesh, w)
    if s.max_factor is not None:
        lam_max = float(s.max_factor)
    elif lam_cr is not None:
        lam_max = 1.1 * lam_cr
    else:
        raise ImperfectionError("Nonlinear buckling: no positive linear buckling factor – give the maximum load factor")
    node, d = _monitor(mesh, w, s)
    inc = int(s.increments)
    # sub-study: one nonlinear step = the case × λ_max, NLGEOM, load control
    st = copy.copy(study)
    st.contact_settings = dataclasses.replace(study.contact_settings, nlgeom=True)
    st.analysis = AnalysisSettings(kind="nonlinear", branches=False, sequence=[
        LoadStep(STEP_NAME, case=lc.name, scale=lam_max, dt0=1.0 / inc, dt_min=float(s.min_increment),
                 dt_max=min(1.0, 2.0 / inc), max_inc=int(s.max_increments))])
    st.modal, st.buckling = ModalSettings(), BucklingSettings()
    from ..joints.fem import bolt_models
    bolts = bolt_models(st, mesh_i) if st.active_bolts() else []
    preload = bool(st.joints.preload_case and any(b.split is not None for b in bolts))
    (job, steps), = step_plans(st, preload)
    job = "nlbuckle"
    temps = None
    if st.has_thermal_loads():
        from ..thermal.field import node_temperatures
        tmap = {t.name: t for t in (thermal or [])}
        temps = [node_temperatures(st, mesh_i, x, tmap) for x in st.load_cases]
    for ext in (".frd", ".dat", ".sta", ".cvg", ".log"):
        (workdir / f"{job}{ext}").unlink(missing_ok=True)
    try:
        write_inp(st, mesh_i, workdir / f"{job}.inp", None, temps, probe_nodes=[node],
                  nonlinear={"steps": steps, "part_nodes": {}})
    except ValueError as e:
        raise ImperfectionError(str(e)) from None
    text = (workdir / f"{job}.inp").read_text(encoding="utf-8")
    k_end = text.rfind("*END STEP")
    text = text[:k_end] + "*NODE PRINT, NSET=NPROBE\nU\n" + text[k_end:]
    (workdir / f"{job}.inp").write_text(text, encoding="utf-8")
    log(f"Nonlinear buckling (CalculiX NLGEOM, λ up to {lam_max:.4g}, imperfection e0 = {e0:.4g} mm)")
    exe = Path(study.solver.exe) if study.solver.exe else CCX_EXE
    r = run_ccx(workdir, job, exe=exe, on_line=on_line, cancel=cancel, deterministic=True)
    if cancel is not None and cancel.is_set():
        raise CcxError("Analysis canceled by user")
    finished = bool(r.finished and not r.errors)
    diag = ccx_messages(Path(r.log_path).read_text(encoding="latin-1", errors="replace")) if r.log_path else {}
    t0, t1 = step_bounds(steps)[-1]
    # curve: monitored node every increment (.dat)
    ser = [(t, blk) for t, kind, blk in print_series(workdir / f"{job}.dat", "NPROBE")
           if kind == "U" and t0 + 1e-12 < t <= t1 + 1e-9] if (workdir / f"{job}.dat").exists() else []
    lam_c = [0.0] + [(t - t0) * lam_max for t, _ in ser]
    u_vec = [np.zeros(3)] + [np.asarray(blk.get(node + 1, np.zeros(3)), float) for _, blk in ser]
    u_c = [float(v @ d) for v in u_vec]
    # fields per increment (.frd)
    frd = read_frd(workdir / f"{job}.frd") if (workdir / f"{job}.frd").exists() else None
    n = len(mesh.nodes)
    kstep = len(steps)
    disp = [b for b in (frd.blocks if frd else []) if b.name == "DISP" and b.step == kstep]
    strs = {round(b.time, 12): b for b in (frd.blocks if frd else []) if b.name == "STRESS" and b.step == kstep}
    if not disp:
        msg = "; ".join((r.errors or []) + diag.get("errors", [])) or f"code {r.returncode}"
        raise ImperfectionError(f"Nonlinear buckling: no converged increment ({msg}). Reduce the initial increment (more increments) or λ max")
    mises_max, frame_lam = [0.0], []
    for b in disp:
        frame_lam.append((float(b.time) - t0) * lam_max)
        sb = strs.get(round(b.time, 12))
        if sb is not None:
            ok = sb.node_ids <= n
            mises_max.append(float(_mises(sb.values[ok, :6]).max()) if ok.any() else 0.0)

    def field(b) -> np.ndarray:
        U = np.zeros((n, 3))
        ok = b.node_ids <= n
        U[b.node_ids[ok] - 1] = b.values[ok, :3]
        return U
    # curve from the .frd when the .dat print is missing (same increments)
    if len(lam_c) < 2:
        lam_c = [0.0] + frame_lam
        u_c = [0.0] + [float(field(b)[node] @ d) for b in disp]
    Re = _yield_strength(study, mesh)
    lam_m = [0.0] + frame_lam if len(mises_max) == len(frame_lam) + 1 else None
    ev = evaluate(lam_c, u_c, finished, lam_max, lam_cr)
    if lam_m is not None and Re:
        from .curve import first_crossing
        ev["first_yield_factor"] = first_crossing(lam_m, mises_max, Re)
        ev["first_yield_Re_MPa"] = Re
    # frames: evenly spaced increments + the last converged one (+ the ultimate)
    nf = max(int(s.frames), 1)
    nd = len(disp)
    pick = sorted(set(np.linspace(0, nd - 1, min(nf, nd)).round().astype(int).tolist()
                      + [nd - 1, int(np.argmax(frame_lam))]))
    shapes_out = np.zeros((len(pick) + 1, n, 3), np.float32)
    shapes_out[0] = w
    for j, i in enumerate(pick, 1):
        shapes_out[j] = field(disp[i])
    del frd, disp, strs
    values = [0.0] + [float(frame_lam[i]) for i in pick]
    if not finished:
        warnings.append(f"Nonlinear buckling: the solution stopped at λ = {lam_c[-1]:.4g} – "
                        + STATUS_TEXT.get(ev["status"], ev["status"]))
    elif ev["status"] == "max_factor":
        warnings.append(f"Nonlinear buckling: {STATUS_TEXT['max_factor']} (λ max = {lam_max:.4g})")
    if ev.get("first_yield_factor") is not None and not any(getattr(m, "is_plastic", False) for m in study.materials.values()):
        warnings.append("Nonlinear buckling (GNIA, elastic material): first yield uses the maximum nodal von Mises stress – "
                        "check that it is not a singularity at a support or a load point")
    info = {
        "case": lc.name, "case_index": ci, "kind": "GMNIA" if any(getattr(m, "is_plastic", False) for m in study.materials.values())
        else "GNIA", "linear_factors": lams[:max(need, 4)], "linear_factor": lam_cr,
        "imperfection": {"kind": s.imperfection, "modes": list(s.modes), "weights": list(s.weights), "e0_mm": e0,
                         "length_mm": reference_length(mesh, s), "amplitude_given_mm": float(s.amplitude or 0.0),
                         "ratio": float(s.amplitude_ratio or 0.0), "sign": float(s.sign or 1.0)},
        "monitor": {"node": node + 1, "point": mesh.nodes[node].tolist(), "direction": d.tolist()},
        "max_factor": lam_max, "finished": finished, "cutbacks": int(diag.get("cutbacks", 0) or 0),
        "curve": {"lambda": [float(x) for x in lam_c], "u_mm": [float(x) for x in u_c],
                  "lambda_frd": [0.0] + [float(x) for x in frame_lam] if lam_m is not None else [],
                  "mises_max_MPa": mises_max if lam_m is not None else []},
        "frames_lambda": values, "result": ev, "status_text": STATUS_TEXT.get(ev["status"], ev["status"]),
        "load_resultant_N": _resultant(st, mesh_i, lc), "warnings": warnings,
    }
    ms = ModeSet(KIND, f"Nonlinear buckling – {lc.name}", values, shapes_out, case=lc.name)
    return ms, info, warnings


def _resultant(study, mesh, lc) -> list[float]:
    """Resultant of the nodal forces of the case at λ = 1 [N] (curve in forces = λ × resultant)."""
    try:
        from ..loads import nodal_loads
        F = nodal_loads(study, mesh, lc)[0]
        return np.asarray(F, float).sum(axis=0).tolist()
    except Exception:  # noqa: BLE001 – informative only
        return []


def attach(study, mesh, res, workdir, log, on_line=None, cancel=None) -> None:
    """Hook of solve._add_eigen: runs the analysis when Study.nlbuckling is enabled and stores it in the results."""
    s = settings_of(study)
    if not s.enabled or not study.load_cases:
        return
    ms, info, warnings = run_nlbuckling(study, mesh, Path(workdir), log, on_line, cancel, res.thermal, res.modes)
    ms.info = info                                    # not stored with the shapes – the GUI reattaches it from Results.info
    res.modes.append(ms)
    res.info["nlbuckling"] = info
    for wtxt in warnings:
        if wtxt not in res.warnings:
            res.warnings.append(wtxt)


def info_of(res) -> dict | None:
    return (getattr(res, "info", None) or {}).get("nlbuckling")
