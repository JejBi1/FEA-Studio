"""Nonlinear calculation in ccx (F8.5, specification 4.5, 4.8, 4.9): contact, NLGEOM, load history.

- branches (default): each load state = a separate job [bolt preload] + the state, in parallel (as
  ccx/parallel), the result = the end of the state; without history → fatigue by superposition is not possible either;
- sequence: one job, steps according to Study.analysis.sequence (state × scale, prescribed motion, changes of
  friction and pairs, cycles), the result = the end of each step + frames of the increments (force–displacement curve);
- non-convergence: the last converted increment applies (partial result with a warning), diagnostics into the warnings;
- equilibrium: the contact forces are internal → the existing check (reactions + loads); with NLGEOM the moments
  about the deformed position; the reactions of the supports in compression only (GAPUNI) from *NODE PRINT of the fixed nodes.
"""
from __future__ import annotations

import re
import copy
import threading
from pathlib import Path

import numpy as np

from .results import (contact_fields, pair_summary, print_series, ccx_messages, ADVICE, contact_print, apply_cf,
                      edge_peak_warnings)


def scaled_case(lc, scale: float):
    """Copy of the load state with the load × scale (vectors, pressure, acceleration; rotation ω × √scale)."""
    if scale == 1.0:
        return lc
    out = copy.deepcopy(lc)
    for l in out.loads:
        for attr in ("vector", "force", "moment", "accel", "alpha"):
            v = getattr(l, attr, None)
            if isinstance(v, (list, tuple)):
                setattr(l, attr, [float(x) * scale for x in v])
            elif isinstance(v, (int, float)) and attr == "alpha":
                setattr(l, attr, float(v) * scale)
        if hasattr(l, "value") and isinstance(getattr(l, "value"), (int, float)) and l.kind == "pressure":
            l.value = float(l.value) * scale
            if l.gradient is not None:
                l.gradient = [float(g) * scale for g in l.gradient]
        if l.kind == "rotation":
            l.omega = float(l.omega) * np.sqrt(abs(scale))
        if l.kind == "inertia":
            l.omega = [float(x) * np.sqrt(abs(scale)) for x in l.omega]
    return out


def step_plans(study, preload: bool) -> list[tuple[str, list[dict]]]:
    """[(job, steps)] – branches: one job per state; sequence: one job."""
    from ..joints.fem import PRELOAD_CASE
    an = study.analysis
    names = [lc.name for lc in study.load_cases]
    pre = [{"name": PRELOAD_CASE, "kind": "preload", "case_index": None, "scale": 1.0, "spec": None}] if preload else []
    if an.branches or not an.sequence:
        return [(f"nl_b{k + 1}", pre + [{"name": lc.name, "kind": "case", "case_index": k, "scale": 1.0, "spec": None}])
                for k, lc in enumerate(study.load_cases)]
    steps = []
    seq = an.sequence
    cycles = max(int(max((s.cycles for s in seq), default=1)), 1)
    for c in range(cycles):
        for s in seq:
            if c > 0 and s.cycles <= c:
                continue
            ci = names.index(s.case) if s.case in names else None
            nm = s.name if cycles == 1 else f"{s.name} (cycle {c + 1})"
            steps.append({"name": nm, "kind": "case", "case_index": ci, "scale": float(s.scale), "spec": s})
    if preload and not any(s["spec"] is not None and s["spec"].bolts == "preload" for s in steps):
        steps = pre + steps
    return [("nl_seq", steps)]


def run_nonlinear(study, mesh, workdir: Path, warnings: list, log, on_line=None, cancel=None, thermal=None,
                  prog=None):
    """Mechanics nonlinearly → Results (StepResult per step / state + contact)."""
    from ..ccx.inp import write_inp
    from ..ccx.runner import run_ccx, CcxError
    from ..ccx.frd import read_frd
    from ..ccx import parallel
    from .. import config
    from ..results import Results
    from ..solve import constraint_masks, make_step, read_remote, _read_printed, StudyError
    from ..joints.fem import bolt_models
    from ..study import LoadCase
    workdir = Path(workdir)
    bolts = bolt_models(study, mesh) if study.active_bolts() else []
    preload = bool(study.joints.preload_case and any(b.split is not None for b in bolts))
    temps = None
    if study.has_thermal_loads():
        from ..thermal.field import node_temperatures
        tmap = {t.name: t for t in (thermal or [])}
        temps = [node_temperatures(study, mesh, lc, tmap) for lc in study.load_cases]
    plans = step_plans(study, preload)
    geom, joint_elems = None, None
    if study.active_bolts() or study.active_welds():
        from ..joints.evaluate import joint_geometry, required_elements
        geom = joint_geometry(study, mesh)
        warnings.extend(w for w in geom["warnings"] if w not in warnings)
        joint_elems = required_elements(geom)
    jobs = []
    for job, steps in plans:
        for ext in (".frd", ".dat", ".sta", ".cvg", ".log"):    # old outputs must not survive into the reading
            try:
                (workdir / f"{job}{ext}").unlink(missing_ok=True)
            except PermissionError:
                raise StudyError(f"Output {job}{ext} is held by another process (an analysis running in the same folder?) – wait for it to finish or terminate it") from None
        try:
            pn = {pname: np.unique(mesh.tets[mesh.tet_part == i]) for i, pname in enumerate(mesh.part_names)} \
                if study.analysis.initial_velocity else {}
            m = write_inp(study, mesh, workdir / f"{job}.inp", joint_elems, temps,
                          nonlinear={"steps": steps, "part_nodes": pn})
        except ValueError as e:
            raise StudyError(str(e)) from None
        jobs.append((job, steps, m))
    exe = Path(study.solver.exe) if study.solver.exe else config.CCX_EXE
    ndof = 3 * len(mesh.nodes)
    workers, threads = parallel.plan(len(jobs), ndof) if parallel.enabled() and len(jobs) > 1 else (1, 0)
    log(f"Calculating nonlinearly (CalculiX, {len(jobs)} {('tasks' if len(jobs) > 1 else 'task')}"
        + (f" in parallel: {workers} × {threads} threads" if workers > 1 else "") + ")")
    lines = on_line
    if prog is not None:                              # progress by the time of the steps (actual step time) over the jobs
        from ..progress import NonlinearLines
        lines = prog.lines(on_line, "mech", NonlinearLines([(j, [b[1] - b[0] for b in step_bounds(s)])
                                                            for j, s, _ in jobs]))
    runs: dict[str, object] = {}
    errs: dict[str, BaseException] = {}
    sem = threading.Semaphore(max(workers, 1))

    def go(job):
        k = [j for j, _, _ in jobs].index(job) + 1
        jl = (lambda line: lines(f"[step {k}] {line}")) if lines is not None and len(jobs) > 1 else lines
        with sem:
            if cancel is not None and cancel.is_set():
                return
            try:
                runs[job] = run_ccx(workdir, job, exe=exe, on_line=jl, cancel=cancel, deterministic=True,
                                    **({"threads": threads} if workers > 1 else {}))
            except BaseException as e:  # noqa: BLE001
                errs[job] = e
    th = [threading.Thread(target=go, args=(j,), daemon=True) for j, _, _ in jobs]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if cancel is not None and cancel.is_set():
        raise CcxError("Analysis canceled by user")
    n = len(mesh.nodes)
    masks = constraint_masks(study, mesh)
    steps_out = []
    plastic = any(getattr(mt, "is_plastic", False) for mt in study.materials.values())
    partial = {}
    info = {"backend": "ccx", "nonlinear": {"mode": "branches" if study.analysis.branches else "sequence",
                                           "jobs": [j for j, _, _ in jobs]}}
    for job, steps, m in jobs:
        r = runs.get(job)
        if job in errs:
            raise errs[job]
        frd_path = workdir / f"{job}.frd"
        frd = read_frd(frd_path) if frd_path.exists() else None
        done = sorted(frd.steps()) if frd is not None else []
        failed = r is None or not r.finished or bool(r.errors)
        if failed:
            log_text = Path(r.log_path).read_text(encoding="latin-1", errors="replace") if r is not None else ""
            diag = ccx_messages(log_text)
            if not done:
                raise CcxError(f"Nonlinear calculation ({job}) did not converge / ended with an error without a converted increment:\n  "
                               + "\n  ".join((r.errors if r else []) or diag["errors"] or ["?"]) + "\n" + ADVICE)
            partial[job] = {"last_step": int(done[-1]), "of_steps": len(steps), **diag}
            warnings.append(f"PARTIAL RESULT ({job}): analysis not completed – the last converged increment of step {done[-1]} of {len(steps)} applies (increment cutbacks {diag['cutbacks']}×). {ADVICE}")
        if r is not None:
            warnings.extend(w for w in r.warnings if "ignored" not in w.lower() and w not in warnings
                            and "no elastic constants" not in w)
        dat = workdir / f"{job}.dat"
        bounds = step_bounds(steps)
        refs = _step_ends(dat, "NREF", "U", bounds) if m.ref_nodes and dat.exists() else []
        rsup = _remote_ends(dat, m, bounds) if m.remote and dat.exists() else []
        gp = {c["pair"]: print_series(dat, c["ground_set"]) for c in m.contacts if c.get("ground_set")} \
            if dat.exists() else {}
        motion_curves = _motion_curves(study, m, dat, steps, bounds) if dat.exists() else {}
        cf = contact_print(dat) if dat.exists() and m.contacts else []
        sig_ip = []
        if m.el_print and dat.exists():
            from ..joints.forces import read_el_print
            sig_ip = read_el_print(dat, m.el_print)
        keep = range(len(steps)) if not study.analysis.branches or not study.analysis.sequence and len(steps) == 1 \
            else range(len(steps))
        for k in keep:
            s = steps[k]
            if k + 1 not in done:
                continue
            if study.analysis.branches and s["kind"] == "preload" and job != jobs[0][0]:
                continue                                  # the preload is enough from the first branch
            U, S, RF = (_nodal(frd, name, k + 1, n, nc) for name, nc in (("DISP", 3), ("STRESS", 6), ("FORC", 3)))
            cblk = _last(frd, "CONTACT", k + 1)
            if s["case_index"] is not None:
                lc = scaled_case(study.load_cases[s["case_index"]], s["scale"])
            else:
                lc = LoadCase(s["name"], [])
            lc = copy.copy(lc)
            lc.name = s["name"]
            extra = _gap_reactions(m, RF, mesh)
            ref = sum(b.preload for b in bolts) if s["kind"] == "preload" or preload else 0.0
            if s["case_index"] is not None and abs(s["scale"]) < 1e-12:
                # unloading to zero (the residual stress is internal): the scale of the equilibrium = the load of the state in full size,
                # otherwise the noise of RF (~1e-3 N) would be divided by an almost zero load and a valid result would be rejected
                from ..loads import nodal_loads
                ref = max(ref, float(np.linalg.norm(nodal_loads(study, mesh, study.load_cases[s["case_index"]])[0],
                                                    axis=1).sum()))
            # NLGEOM: moments about the deformed position x + u. Without NLGEOM the body is linear (undeformed position), but
            # the contact is evaluated in the current position (displacement of the slave against the master – R1: N·displacement) → for an imbalance
            # the deformed position is tried
            args = (study, mesh, lc, U, S, RF, masks, refs[k] if k < len(refs) else {}, ref,
                    rsup[k] if k < len(rsup) else None)
            dyn = bool(getattr(s.get("spec"), "dynamic", False))
            from ..remote import rigid_rotations, rigid_rotation_note
            rot_items = rigid_rotations(study, m.ref_nodes, refs[k] if k < len(refs) else {},
                                        rsup[k] if k < len(rsup) else None)
            rot_note = rigid_rotation_note(rot_items, s["name"], study.contact_settings.nlgeom)
            if rot_note and rot_note not in warnings:
                warnings.append(rot_note)
            try:
                try:
                    st = make_step(*args, extra=extra, x_eq=mesh.nodes + U if study.contact_settings.nlgeom else None,
                                   check=not dyn)
                except CcxError:
                    if study.contact_settings.nlgeom or not m.contacts:
                        raise
                    st = make_step(*args, extra=extra, x_eq=mesh.nodes + U)
            except CcxError as e:
                if job in partial and k + 1 == done[-1]:
                    warnings.append(f"Step '{s['name']}' (partial): {e}")
                    continue
                if rot_note and "are not in equilibrium" in str(e):
                    raise CcxError(f"{e}\n{rot_note}") from None
                raise
            fields = contact_fields(cblk, n)
            if m.contacts:
                st.contacts = pair_summary(study, mesh, mesh.report.get("contacts", []), fields, U, m.contacts)
                apply_cf(st.contacts, m.contacts, cf, bounds[k])
                hist = _cf_history(m.contacts, cf, bounds[k])
                if hist:
                    st.curves.update(hist)
                for kf in ("CPRESS", "COPEN", "CSLIP", "CSHEAR", "STATUS", "OPENING"):
                    st.fields[kf] = fields[kf]
            pe_blk = _last(frd, "PE", k + 1)
            if pe_blk is not None:
                try:
                    peeq = np.zeros(len(mesh.nodes))
                    ok = pe_blk.node_ids <= len(mesh.nodes)
                    peeq[pe_blk.node_ids[ok] - 1] = pe_blk.values[ok, 0]
                    st.fields["peeq"] = peeq
                except (IndexError, ValueError) as e:
                    warnings.append(f"Step '{s['name']}': equivalent plastic strain PEEQ could not be read ({e}) – "
                                    f"the plastic strain field is missing from the results")
            elif plastic:
                warnings.append(f"Step '{s['name']}': the .frd file has no equivalent plastic strain PEEQ – "
                                f"the plastic strain field is missing from the results (stresses and displacements are valid)")
            if dyn:
                st.equilibrium["dynamic"] = True
                warnings.append(f"Step '{s['name']}' (implicit dynamics): reaction equilibrium is not checked (inertia forces) – monitor the energy and the contact force history")
            if job in partial and k + 1 == done[-1]:
                st.partial = True
                st.fields_time = float(_last(frd, "DISP", k + 1).time)
            if motion_curves.get(k):
                st.curves = motion_curves[k]
            if geom is not None:
                from ..joints.evaluate import step_forces, to_json
                from ..loads import nodal_loads
                F = st.__dict__.pop("_F", None)
                if F is None:
                    F = nodal_loads(study, mesh, lc)[0]
                jf = step_forces(study, mesh, geom, RF, F, sig_ip[k] if k < len(sig_ip) else None)
                _clamp_from_contact(study, mesh, geom, jf, fields, getattr(st, "contacts", {}))
                st.joints = to_json(jf)
            steps_out.append(st)
    info["nonlinear"]["partial"] = partial
    warnings.extend(w for w in edge_peak_warnings(steps_out) if w not in warnings)
    try:
        from ..nl_convergence import read_convergence_job
        conv_hist = {}
        for j_name, *_ in jobs:                 # jobs = (job, steps, model) – a 2-tuple unpack failed silently before
            h = read_convergence_job(workdir, j_name)
            if h.iterations or h.increments:
                conv_hist[j_name] = h.to_dict()
        if conv_hist:
            info["nonlinear"]["convergence"] = conv_hist
    except Exception as e:              # diagnostics only – the results stay valid
        warnings.append(f"Convergence history of the nonlinear analysis could not be read ({type(e).__name__}: {e}) – "
                        f"the convergence monitor has no data for this run")
    from ..solve import print_results
    print_results(study, mesh, steps_out, info, warnings, log)
    warnings.append("Nonlinear analysis (contact / history): superposition of cases does not apply – fatigue by superposition is unavailable, safety factor = stress margin, not a load multiplier; contact stiffness is a penalty (K, λ) and μ is an estimate – verify the results by a sensitivity study")
    return Results(mesh, steps_out, warnings, info, thermal)


def _nodal(frd, name, step, n, nc):
    b = _last(frd, name, step)
    out = np.zeros((n, nc))
    if b is None:
        return out
    m = b.node_ids <= n
    out[b.node_ids[m] - 1] = b.values[m, :nc]
    return out


def _last(frd, name, step):
    cand = [b for b in frd.blocks if b.name == name and b.step == step]
    return cand[-1] if cand else None


def _gap_reactions(m, RF: np.ndarray, mesh) -> dict:
    """Reactions of the supports in compression only (GAPUNI): force on the part = −Σ RF of the nodes of the face on the side of the body (RF of the fixed nodes is
    doubled in ccx 2.23 – trial G1) → {pair: (force, points, forces)} for make_step(extra=); the RF of these nodes is thereby
    subtracted from the reaction (they are not in the constraints of the mesh).
    """
    out = {}
    for c in m.contacts:
        if not c.get("ground_set"):
            continue
        nodes = np.array(c["nodes"]) - 1
        F = -RF[nodes]
        out[c["pair"]] = (F.sum(axis=0), mesh.nodes[nodes], F)
    return out


def _motion_curves(study, m, dat: Path, steps, bounds=None) -> dict:
    """Force–displacement curves of the prescribed motions (C2): {step: {"target": {"t", "u" (displacement of REF), "F" (RF of REF)}}}."""
    out = {}
    if not m.remote:
        return out
    ser = print_series(dat, "NRSUP")
    if not ser:
        return out
    for k, s in enumerate(steps):
        spec = s.get("spec")
        if spec is None or not spec.prescribed:
            continue
        cur = {}
        for mo in spec.prescribed:
            info = m.remote.get(mo.target)
            if info is None:
                continue
            node = info["ref"] if mo.kind == "disp" else info["rot"]
            b = bounds[k] if bounds else (k, k + 1)
            t_u = [(t, blk[node]) for t, kind, blk in ser if kind == "U" and node in blk and _in(t, b)]
            t_f = [(t, blk[node]) for t, kind, blk in ser if kind == "RF" and node in blk and _in(t, b)]
            if not t_u:
                continue
            cur[mo.target] = {"t": [x[0] for x in t_u], "u": [x[1].tolist() for x in t_u],
                              "F": [x[1].tolist() for x in t_f], "dof": int(mo.dof), "kind": mo.kind}
        if cur:
            out[k] = cur
    return out


def step_bounds(steps) -> list[tuple[float, float]]:
    """(start, end) of the total time of ccx by steps: a static step lasts 1, a dynamic one `time` seconds."""
    out, t0 = [], 0.0
    for s in steps:
        spec = s.get("spec")
        d = float(spec.time) if spec is not None and getattr(spec, "dynamic", False) else 1.0
        out.append((t0, t0 + d))
        t0 += d
    return out


def _in(t: float, b: tuple[float, float]) -> bool:
    eps = 1e-9 * max(1.0, abs(b[1]))
    return b[0] + eps < t <= b[1] + eps


def _step_ends(dat: Path, setname: str, kind: str, bounds) -> list[dict]:
    """Output of *NODE PRINT at the END of each step (nonlinearly every increment is printed; the time of ccx is total)."""
    ser = [x for x in print_series(dat, setname) if x[1] == kind]
    out = []
    for b in bounds:
        cand = [blk for t, _, blk in ser if _in(t, b)]
        out.append(cand[-1] if cand else {})
    return out


def _remote_ends(dat: Path, m, bounds) -> list[dict]:
    """Reactions and displacements of the rigid remote constraints at the ends of the steps (as solve.read_remote)."""
    U = _step_ends(dat, "NRSUP", "U", bounds)
    RF = _step_ends(dat, "NRSUP", "RF", bounds)
    out = []
    for u, rf in zip(U, RF):
        d = {}
        for sel, info in m.remote.items():
            z = np.zeros(3)
            d[sel] = {"force": rf.get(info["ref"], z), "moment": rf.get(info["rot"], z),
                      "u": u.get(info["ref"], z), "rot": u.get(info["rot"], z)}
        out.append(d)
    return out


def _cf_history(params, cf: list[dict], b) -> dict:
    """Course of the force of a pair in the step from *CONTACT PRINT CF (by increments): {"contact:pair": {"t", "F", "Fn"}}."""
    out = {}
    for c in params:
        if not c.get("slave"):
            continue
        rows = [x for x in cf if x["slave"] == c["slave"] and x["master"] == c["master"] and _in(x["time"], b)]
        if len(rows) < 2:
            continue
        out[f"kontakt:{c['pair']}"] = {"t": [x["time"] for x in rows], "F": [x["force"].tolist() for x in rows],
                                       "Fn": [float(abs(x["force"] @ (x["normal"] / max(np.linalg.norm(x["normal"]),
                                                                                       1e-300)))) for x in rows]}
    return out


def _clamp_from_contact(study, mesh, geom, jf: dict, fields: dict, pairs: dict | None = None) -> None:
    """Bolt interface by contact: residual clamping force in the cell of the bolt = Σ CPRESS·area of the nodes of the interface (specification 4.6)
    and the fraction of the open area → planes[k]["F_clamp_contact"], ["open_fraction"]. The nodal CPRESS (extrapolation
    from the integration points) overestimates the integral → the scale is the exact force of the pair from the ccx CF (Fn_N / Fn_nodal_N).
    """
    if "CPRESS" not in fields:
        return
    from ..ccx.inp import consistent_nodal_forces
    from ..joints.forces import mesh_interfaces
    p = fields["CPRESS"]
    alias = mesh.canonical()
    dup = np.nonzero(alias != np.arange(len(alias)))[0]
    pmax = np.array(p, float)                       # the pressure is on the slave side of the disconnected interface – max(node, copy)
    np.maximum.at(pmax, alias[dup], pmax[dup])
    # and the other way round (copy ← original node): the nodes of the interface in a bolt cell may be copies, when the slave side is the original
    # (example 9: slave = upper part → F_clamp_contact came out 0 and open_fraction 1; K6 has the slave on the copies)
    pmax[dup] = np.maximum(pmax[dup], pmax[alias[dup]])
    ifs_all = mesh_interfaces(mesh)
    for b in study.active_bolts():
        planes = jf["bolts"].get(b.name, {}).get("planes", [])
        for pl, iface in zip(planes, geom["bolts"].get(b.name, [])):
            nodes = iface["nodes"]
            if not len(nodes) or iface["upper"] not in mesh.part_names or iface["lower"] not in mesh.part_names:
                continue
            ia, ib = mesh.part_names.index(iface["upper"]), mesh.part_names.index(iface["lower"])
            ifs = ifs_all.get((min(ia, ib), max(ia, ib)))
            if ifs is None:
                continue
            vals = pmax[np.asarray(nodes, np.int64)]
            w = consistent_nodal_forces(mesh.nodes, ifs["tri"], np.array([1.0, 0.0, 0.0]))
            wn = np.array([w.get(int(n), np.zeros(3))[0] for n in nodes])
            sc = 1.0
            for nm, pc in (pairs or {}).items():
                if {iface["upper"], iface["lower"]} <= set(re.sub(r"^(Interface|Spára) ", "", nm).split("–")) and \
                        pc.get("Fn_nodal_N") and pc.get("source") == "ccx CF":
                    sc = pc["Fn_N"] / pc["Fn_nodal_N"]
            pl["F_clamp_contact"] = float((vals * wn).sum()) * sc
            pl["open_fraction"] = float(wn[vals <= 1e-9].sum() / max(wn.sum(), 1e-30))
