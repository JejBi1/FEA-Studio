"""Tools: solving, summary of results, probes, places of maxima and headless PNG images."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from mcp.server.mcpserver import Image

from .common import out, run_with_timeout
from .diagnostics import (FIELDS, REVERSED, SIGNED, THERMAL_KEYS, field_key, field_values, hotspots, resolve_source,
                          results_summary, thermal_hotspots, thermal_values)
from .session import FeaError, Session, user_path


def _check_time(tr, time_s) -> list[str]:
    """Time of the snapshot of a transient state: check of the range, notes (for a steady state the time is not used)."""
    if time_s is None:
        return []
    t = float(time_s)
    if not tr.transient:
        return [f"Thermal state '{tr.name}' is steady – time_s is not used"]
    end = tr.times[-1] if tr.times else 0.0
    if not 0 <= t <= end * (1 + 1e-9):
        raise FeaError(f"time_s = {t:g} s is outside the transient state '{tr.name}' (0–{end:g} s)")
    if tr.times and t < tr.times[0]:
        return [f"Before the first stored frame (t = {tr.times[0]:g} s) the temperature of the first frame is taken – increase the outputs of the thermal state for a finer history"]
    return []


def _png_path(sess: Session, output_path: str | None, stem: str) -> Path:
    if output_path:
        p = user_path(output_path)
        if p.suffix.lower() != ".png":
            p = p.with_suffix(".png")
        return p
    return (sess.workdir or sess.base_dir) / "images" / f"{stem}_{time.strftime('%H%M%S')}.png"


def solve_core(sess: Session, backend: str = "auto", timeout_s: float = 600, cancel=None, on_progress=None,
               on_log=None, on_line=None) -> dict:
    """Body of the calculation (fea_solve_run and the asynchronous job fea_job_start): check of readiness → mesh (if missing) →
    run_study → results into the session → brief summary. The caller holds the lock of the session. cancel = external cancellation (job),
    on_progress(value 0–1, end of the phase, text) = progress (fea_core.progress), on_log / on_line = log / lines of the solver.
    """
    if backend not in ("auto", "ccx", "ngs_hypre", "ngs_pardiso"):
        raise FeaError("backend must be auto | ccx | ngs_hypre | ngs_pardiso")
    st = sess.require_study()
    from ..solvers import check_backend
    why = check_backend(st, backend)                # installed? (ngsolve, MKL, hypre, MS-MPI) can it handle the study? – before the mesh
    if why:
        raise FeaError("\n".join(why).replace("choose the Automatic solver", "use backend='auto'")
                       .replace("choose Automatic or CalculiX", "use backend='auto' or 'ccx'"))
    from .tools_mesh import readiness, do_mesh
    saved_backend = st.solver.backend               # the choice of the backend applies only to this run (also for the check)
    st.solver.backend = backend
    try:
        rd = readiness(sess)
    finally:
        st.solver.backend = saved_backend
    if not rd["ready"]:
        raise FeaError("The study is not ready for calculation:\n  " + "\n  ".join(rd["errors"]))
    t0 = time.time()
    mesh_info = None
    reduced = st.twod.enabled or st.shell.enabled          # the 2D section / shells have their own mesh
    meshing = sess.valid_mesh() is None and not sess.frame_only and not reduced     # only bars, no meshing
    w_mesh = 0.2 if meshing else 0.0                       # share of the meshing in the course of the task
    if meshing:
        if on_progress is not None:
            on_progress(0.0, w_mesh, "Meshing (Gmsh)")
        mesh_info = do_mesh(sess, timeout_s, cancel)
    mesh = sess.valid_mesh()
    logs: list[str] = []

    def log(line: str):
        logs.append(line)
        if on_log is not None:
            on_log(line)

    def prog(v, end=None, text=""):
        if on_progress is not None:
            on_progress(w_mesh + (1 - w_mesh) * float(v), w_mesh + (1 - w_mesh) * float(end if end is not None else v),
                        text)
    saved_backend = st.solver.backend               # the choice of the backend applies only to this run

    def work(c):
        from ..solve import run_study
        return run_study(st, sess.workdir / "run", on_log=log, cancel=c, mesh=mesh, parts=sess.parts,
                         on_solver_line=on_line, on_progress=prog if on_progress is not None else None)
    remaining = max(timeout_s - (time.time() - t0), 10.0)
    st.solver.backend = backend
    try:
        res = run_with_timeout(work, remaining, "Calculation", cancel)
    finally:
        st.solver.backend = saved_backend
        for line in logs:
            sess.add_log(line)
    sess.results = res
    sess.fatigue = None
    sess.joints = None
    sess.frame_check = None
    sess.voxel = None
    if st.cad_link is not None:                     # link with the CAD: the summary of the results goes into the current iteration
        try:
            from ..cad_link.history import record
            record(st, res, sess.parts)
        except Exception as e:  # noqa: BLE001 – the summary of the iteration does not discard the result
            sess.add_log(f"The iteration summary cannot be written: {e}")
    summ = results_summary(sess)
    steps = []
    for s in summ["steps"]:
        steps.append({k: s[k] for k in ("step", "max_displacement_mm", "max_von_mises_MPa", "reactions",
                                        "equilibrium", "min_safety_Re", "temperature_C", "note") if k in s})
        if s.get("frame"):                         # bar system: brief extremes
            steps[-1]["frame"] = {k: s["frame"][k] for k in ("max_node_displacement_mm", "max_sigma_eq_MPa",
                                                             "max_abs_moment_Nmm", "reactions")
                                  if k in s["frame"]}
    for s_out, s_res in zip(steps, res.steps):     # contact: forces of the pairs, partial result
        if s_res.contacts:
            s_out["contacts"] = {k: {kk: c.get(kk) for kk in ("Fn_N", "Ft_N", "area_contact_mm2", "p_max_MPa",
                                                              "open_max_mm", "slip_max_mm")}
                                 for k, c in s_res.contacts.items()}
        if s_res.partial:
            s_out["partial"] = True
    d = {"ok": True, "backend": res.info.get("backend"), "time_s": time.time() - t0,
         "mesh": summ["mesh"], "steps": steps, "warnings": res.warnings, "log": logs[-8:],
         "next": ("fea_frame_results, fea_frame_assess, fea_frame_render" if sess.frame_only else
                  "fea_results_summary, fea_results_hotspots, fea_results_render_image"
                  + (", fea_frame_results" if st.frame.active else ""))}
    if summ.get("thermal"):
        d["thermal"] = [{k: t[k] for k in ("thermal_case", "transient", "T_max_C", "T_min_C",
                                           "max_heat_flux_W_m2", "heat_flows_W", "balance") if k in t}
                        for t in summ["thermal"]]
        d["next"] += " (teplo: field='temperature' | 'heat_flux', thermal_case=…)"
    if summ.get("twod"):
        d["twod"] = summ["twod"]
        d["next"] = "fea_twod_results, fea_twod_render, fea_twod_probe, fea_twod_line, fea_twod_linearize"
    if summ.get("shell"):
        d["shell"] = summ["shell"]
        d["next"] = "fea_shell_results, fea_shell_render, fea_shell_probe"
    if summ.get("modes"):
        d["modes"] = summ["modes"]
        d["next"] += ", fea_results_modes, fea_results_render_mode"
        if any(m["kind"] == "buckling" for m in summ["modes"]):
            d["buckling_warning"] = ("Linear buckling without imperfections and plasticity – the real capacity is usually lower than λ × load")
    if res.info.get("nonlinear"):
        nl = res.info["nonlinear"]
        d["nonlinear"] = {k: nl.get(k) for k in ("mode", "jobs", "partial") if k in nl}
        d["nonlinear_warning"] = ("Nonlinear calculation (contact): superposition does not hold, μ is an estimate, K and λ are penalties – verify the results by a sensitivity study"
                                  + ("; PARTIAL RESULT – see warnings" if any(s.partial for s in res.steps) else ""))
        d["next"] += ", fea_contact_results, fea_contact_render, fea_results_history"
    if getattr(res, "dynamics", None) is not None:
        ds = res.dynamics.summary()
        d["dynamics"] = {k: ds[k] for k in ("modes", "frequencies_Hz", "harmonic", "transient", "random", "shock")
                         if k in ds}
        if isinstance(d["dynamics"].get("frequencies_Hz"), list):
            d["dynamics"]["frequencies_Hz"] = d["dynamics"]["frequencies_Hz"][:10]
        d["damping_warning"] = "Damping is an uncertain input (sensitivity ±50 %)"
        d["next"] += ", fea_dyn_results, fea_dyn_frf_get, fea_dyn_plot, fea_dyn_render"
    if mesh_info:
        d["mesh_generated"] = {k: mesh_info[k] for k in ("nodes", "elements", "min_scaled_jacobian", "time_s")}
    return d


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_solve_run(backend: str = "auto", timeout_s: float = 600) -> str:
        """Runs the calculation and WAITS: thermal states (ccx, heat balance check), statics of all load states (nonlinear with contacts), frequencies, dynamics; the mesh is made if missing/stale. A heat-only study needs no constraints or load states. For contact/nonlinear/dynamics prefer fea_job_start(kind='solve'). backend (statics): auto (ccx for what NGSolve cannot do - bolts, welds, cylindrical/remote/rigid constraints, *TIE, orthotropy, elastic support, point mass, beams; > 600k DOF NGSolve+hypre; >= 2 states ngs_pardiso when MKL+NGSolve available; else ccx PARDISO) | ccx | ngs_hypre | ngs_pardiso (one factorization for all states); heat and contact always ccx. Force/moment equilibrium (tol 1e-3) and steady heat balance are checked; an imbalance is an error and no result is returned. timeout_s covers meshing + solving. Returns a summary (max displacement, max von Mises, reactions, equilibrium; heat: T range, fluxes, balance)."""
        with sess.use(op="analysis", wait=5):
            return out(solve_core(sess, backend, timeout_s))

    @mcp.tool(structured_output=False)
    def fea_results_section_force(point: list[float], normal: list[float],
                                  parts: list[str] | None = None, step: int | str = 0) -> str:
        """Section force (free body cut): internal N, Vy, Vz, Mx, My, Mz about the cut centroid in the cut axes (x = normal). point [x,y,z] in the plane, normal [nx,ny,nz]; parts (None = all); step = mechanical state."""
        with sess.use():
            res = sess.require_results()
            if not res.steps:
                raise FeaError("The results contain no mechanical states")
            if not len(res.mesh.nodes):
                from .diagnostics import no_solid_hint
                raise FeaError(no_solid_hint(res))
            if len(point) != 3 or len(normal) != 3 or not any(normal):
                raise FeaError("point and normal must be vectors [x, y, z] (the normal nonzero)")
            i = sess.step_index(step)
            from ..section_forces import section_forces
            try:
                r = section_forces(sess.require_study(), res.mesh, res.steps[i], point, normal, parts)
            except ValueError as e:
                raise FeaError(str(e))
            r["step"] = res.steps[i].name
            return out(r)

    @mcp.tool(structured_output=False)
    def fea_results_summary(step: int | str | None = 0, thermal_case: int | str | None = None) -> str:
        """Results overview. step (index or name, None = all): max displacement (value, place, vector), ux/uy/uz ranges, max von Mises (place, part), s1 max / s3 min, reactions [N, N*mm] per constraint, equilibrium, safety to Re per part, top 5 places with singularity classification, temperature range for thermal loading. thermal_case (one, None = all): T max/min, max flux, temperatures per part, boundary-condition fluxes [W] (+ into body), heat balance, transient snapshots."""
        with sess.use():
            return out(results_summary(sess, step, thermal_case))

    @mcp.tool(structured_output=False)
    def fea_results_probe(point: list[float] | list[list[float]], step: int | str = 0,
                          thermal_case: int | str | None = None, time_s: float | None = None,
                          csys: str | dict | None = None, criterion: str | None = None,
                          limit_MPa: float | None = None) -> str:
        """Values at point(s) [x,y,z] mm (tet10 interpolation; outside the mesh = nearest element + warning). step: displacements [mm], stress components, von Mises, principal [MPa] (+ T for thermal loading). thermal_case (default first for heat-only studies): temperature [degC] and flux vector [W/m2]; time_s = transient snapshot (flux only at the end). csys: name/definition for transforming stress and displacement. criterion 'mises'|'tresca'|'mohr_coulomb' with limit_MPa gives a safety."""
        with sess.use():
            res = sess.require_results()
            if not len(res.mesh.nodes):
                from .diagnostics import no_solid_hint
                raise FeaError(no_solid_hint(res))
            pts = np.atleast_2d(np.asarray(point, float))
            if pts.shape[1] != 3:
                raise FeaError("point must be [x, y, z] or a list of such points")
            if len(pts) > 200:
                raise FeaError("At most 200 points at a time")
            from ..results import von_mises, principal, tet10_shape

            def where(p):
                e, xi = res.locate(p)
                err = max(-float(xi.min()), float(xi.sum()) - 1.0, 0.0)
                conn = res.mesh.tets[e]
                N = tet10_shape(xi) if len(conn) == 10 else np.array([1 - xi.sum(), *xi])
                row = {"point_mm": p, "inside_mesh": err < 1e-6,
                       "part": res.mesh.part_names[int(res.mesh.tet_part[e])]}
                if err >= 1e-6:
                    d = float(np.min(np.linalg.norm(res.mesh.nodes[conn] - p, axis=1)))
                    row["warning"] = f"the point lies outside the mesh (distance from the nearest element node {d:.3g} mm) – extrapolation"
                return row, N, conn
            if thermal_case is not None or not res.steps:
                k = sess.thermal_index(0 if thermal_case is None else thermal_case)
                tr = res.thermal[k]
                notes = _check_time(tr, time_s)
                T = tr.at_time(time_s)
                outl = []
                for p in pts:
                    row, N, conn = where(p)
                    q = N @ tr.flux[conn]
                    row.update({"T_C": float(N @ T[conn]), "heat_flux_W_m2": q,
                                "heat_flux_mag_W_m2": float(np.linalg.norm(q))})
                    outl.append(row)
                d = {"thermal_case": tr.name, "probes": outl}
                if tr.transient:
                    d["time_s"] = float(tr.times[-1]) if time_s is None and tr.times else time_s
                    if time_s is not None:
                        notes.append("the heat flux is from the end of the transient state")
                if notes:
                    d["notes"] = notes
                return out(d)
            i = sess.step_index(step)
            st = res.steps[i]
            cs_name = None
            study_cs = sess.study
            if csys is not None:
                if isinstance(csys, str):
                    if csys not in sess.study.coordinate_systems:
                        raise FeaError(f"Coordinate system '{csys}' does not exist in the study")
                    cs_name = csys
                elif isinstance(csys, dict):
                    from ..study import CoordinateSystem
                    d_cs = {"name": "_probe", **csys}
                    if "x_axis" not in d_cs:                    # any direction perpendicular to z (as fea_csys_create)
                        zz = np.asarray(d_cs.get("z_axis", [0.0, 0.0, 1.0]), float)
                        ref = [1.0, 0, 0] if abs(zz[0]) < 0.9 * max(np.linalg.norm(zz), 1e-300) else [0, 1.0, 0]
                        d_cs["x_axis"] = np.cross(np.cross(zz, ref), zz).tolist()
                    try:
                        cs_def = CoordinateSystem(**d_cs)
                    except TypeError as e:
                        raise FeaError(f"The csys definition is invalid: {e}")
                    from ..csys import check as check_cs
                    errs = check_cs(cs_def)
                    if errs:
                        raise FeaError(errs[0])
                    cs_name = cs_def.name
                    # a temporary definition only for this query (does not change the study)
                    import copy
                    study_cs = copy.copy(sess.study)
                    study_cs.coordinate_systems = {**sess.study.coordinate_systems, cs_name: cs_def}
                else:
                    raise FeaError("csys must be a system name or its definition")
            outl = []
            from ..csys import transform_stress, transform_displacements, stress_labels, displacement_labels
            from ..assessment import equivalent_stress, resolve_assessment
            for p in pts:
                row, N, conn = where(p)
                U = N @ st.U[conn]
                S = N @ st.S[conn]
                pr = principal(S)
                p_arr = np.asarray(p, float)
                if cs_name is not None:
                    try:
                        S_loc = transform_stress(study_cs, cs_name, S[None, :], p_arr[None, :], "point")[0]
                        U_loc = transform_displacements(study_cs, cs_name, U[None, :], p_arr[None, :], "point")[0]
                    except Exception as e:
                        raise FeaError(f"Transformation to the coordinate system '{cs_name}' at the point {p_arr.tolist()}: {e}")
                    slab = stress_labels(study_cs, cs_name)
                    ulab = displacement_labels(study_cs, cs_name)
                    row.update({"u_mm": U_loc, "u_mag_mm": float(np.linalg.norm(U)),
                                "u_labels": ulab,
                                "stress_MPa": dict(zip(slab, S_loc)),
                                "mises_MPa": float(von_mises(S)), "principal_MPa": pr,
                                "csys": cs_name})
                else:
                    row.update({"u_mm": U, "u_mag_mm": float(np.linalg.norm(U)),
                                "stress_MPa": dict(zip(("sxx", "syy", "szz", "sxy", "syz", "szx"), S)),
                                "mises_MPa": float(von_mises(S)), "principal_MPa": pr})
                if criterion:
                    if criterion not in ("mises", "tresca", "mohr_coulomb"):
                        raise FeaError("criterion must be mises | tresca | mohr_coulomb")
                    mat = sess.study.material_of(row.get("part"))
                    _, lim_type = resolve_assessment(sess.study, row.get("part"), criterion, None)
                    if criterion == "mohr_coulomb":
                        lim_type = "Rm"
                    lim = limit_MPa or getattr(mat, lim_type, None) or mat.Re or mat.Rm
                    rc = getattr(mat, "Rc", None)
                    sigma_eq, _ = equivalent_stress(S[None, :], criterion=criterion, Rc=rc, Rm=lim)
                    row["equivalent_stress_MPa"] = float(sigma_eq[0])
                    row["criterion"] = criterion
                    if lim:
                        row["limit_MPa"] = float(lim)
                        row["safety_factor"] = float(lim / max(float(sigma_eq[0]), 1e-12))
                if "T" in st.fields:
                    row["T_C"] = float(N @ st.fields["T"][conn])
                w = row.pop("warning", None)              # order of the keys as before (warnings at the end)
                if w:
                    row["warning"] = w
                outl.append(row)
            return out({"step": st.name, "probes": outl})

    @mcp.tool(structured_output=False)
    def fea_results_hotspots(top_n: int = 5, step: int | str = 0, field: str = "mises",
                             thermal_case: int | str | None = None, time_s: float | None = None) -> str:
        """Highest values of `field` (mises | s1 | s3 | tresca | displacement ...) in separate places (clusters of ~3 elements), classified as singularity (constraint edge, sharp inner corner, point force, bolt connection) / boundary-condition influenced / probable real concentration (fillet, hole), with advice (evaluate, refine). Heat: field 'temperature' (hottest + coldest places) or 'heat_flux' of thermal_case (time_s = snapshot); heat-only studies pick automatically."""
        if not 1 <= top_n <= 50:
            raise FeaError("top_n must be 1–50")
        with sess.use():
            res = sess.require_results()
            key = field_key(field)
            note = None
            if thermal_case is None and not res.steps and res.thermal and key not in THERMAL_KEYS:
                note = f"The results contain only thermal states – the hottest places are searched instead of '{field}'"
                key = "temperature"
            src, i = resolve_source(sess, key, step, thermal_case)
            if src == "thermal":
                notes = _check_time(res.thermal[i], time_s)
                d = thermal_hotspots(sess, i, top_n, key, time_s)
                if note:
                    notes.insert(0, note)
                if notes:
                    d["notes"] = notes
            else:
                d = hotspots(sess, i, top_n, key)
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_results_render_image(output_path: str | None = None, field: str = "mises", step: int | str = 0,
                                 view: str | list[float] = "iso", deformed: bool = True, scale: float | None = None,
                                 width: int = 1280, height: int = 720, show_mesh: bool = False, bands: int = 12,
                                 vmin: float | None = None, vmax: float | None = None, return_image: bool = True,
                                 thermal_case: int | str | None = None, time_s: float | None = None,
                                 section_point: list[float] | None = None, section_normal: list[float] | None = None):
        """Headless PNG render of a result (colour bands, legend, max/min markers, undeformed outline). field: mises | s1 | s2 | s3 | tresca | sxx..szx | displacement | ux | uy | uz | safety (Re/vM) | print_safety | print_index | fatigue_damage | fatigue_life | fatigue_safety | fatigue_amplitude | temperature [degC] | heat_flux | hfx | hfy | hfz [W/m2]. Heat: thermal_case, time_s (undeformed shape). view: iso | iso_back | iso_left | iso_bottom | front | back | left | right | top | bottom | multi (2x2) or camera direction [x,y,z]. scale: deformation (None = auto 5 % of size, 1 = real); vmin/vmax: fixed range; bands (12). section_point + section_normal: cut keeping (x-point).n <= 0, cut face coloured by the field (undeformed). Returns JSON (path, metadata) and the image when return_image."""
        from .render import render
        from .scenes import result_scene
        if not 2 <= bands <= 24:
            raise FeaError("bands must be 2–24")
        section = None
        if (section_point is None) != (section_normal is None):
            raise FeaError("A section needs both section_point and section_normal")
        if section_point is not None:
            if len(section_point) != 3 or len(section_normal) != 3 or not any(section_normal):
                raise FeaError("section_point and section_normal must be vectors [x, y, z] (the normal nonzero)")
            section = ([float(v) for v in section_point], [float(v) for v in section_normal])
        with sess.use(op="render"):
            res = sess.require_results()
            key = field_key(field)
            label, unit = FIELDS[key]
            src, i = resolve_source(sess, key, step, thermal_case)
            notes = []
            if src == "thermal":
                notes = _check_time(res.thermal[i], time_s)
                vals = thermal_values(sess, i, key, time_s)
                scene, meta = result_scene(sess, 0, key, vals, label, unit, key in REVERSED, False, None, bands,
                                           vmin, vmax, show_mesh, key in SIGNED, thermal=i, time_s=time_s,
                                           section=section)
                stem = f"{key}_T{i}"
            else:
                vals = field_values(sess, i, key)
                scene, meta = result_scene(sess, i, key, vals, label, unit, key in REVERSED, deformed, scale, bands,
                                           vmin, vmax, show_mesh, key in SIGNED, section=section)
                stem = f"{key}_{i}"
            if notes:
                meta["notes"] = notes
            p = _png_path(sess, output_path, stem)
            t0 = time.time()
            info = render(scene, p, view, width, height)
            meta.update(info)
            meta["render_time_s"] = time.time() - t0
            txt = out(meta)
            return [txt, Image(path=p)] if return_image else txt

    @mcp.tool(structured_output=False)
    def fea_model_render_image(output_path: str | None = None, view: str | list[float] = "multi",
                               load_case: str | None = None, highlight: str | list | dict | None = None,
                               show_face_ids: bool = False, face_ids_part: str | None = None,
                               width: int = 1280, height: int = 720, return_image: bool = True,
                               thermal_case: str | None = None):
        """Image of the model BEFORE solving, to check the setup: constraint faces (blue), loads (red/orange) with arrow values, remote point, gravity. highlight: faces (id/query/selection, purple); show_face_ids: label visible faces 'Díl#i' (face_ids_part limits to a part); view default multi (2x2). Heat: thermal_case draws only that thermal state (temperature purple, flux orange, convection turquoise, radiation yellow, source pink). Without load_case/thermal_case everything is drawn."""
        from .render import render
        from .scenes import model_scene
        with sess.use(op="render"):
            sess.require_study()
            scene = model_scene(sess, load_case, highlight, show_face_ids, face_ids_part, thermal_case)
            p = _png_path(sess, output_path, "model")
            info = render(scene, p, view, width, height)
            txt = out(info)
            return [txt, Image(path=p)] if return_image else txt

    @mcp.tool(structured_output=False)
    def fea_report_word(output_path: str | None = None, title: str | None = None, author: str = "",
                        company: str = "", subject: str = "", notes: str = "", required_safety: float = 1.0,
                        images: bool = True, assess_joints: bool = True, keep_images: bool = False,
                        language: str = "en") -> str:
        """Word (.docx) report: summary with assessment, input (parts, materials, constraints, loads, model image), mesh/solver, results per state (values, reactions, equilibrium, safety per part, maxima with singularity classification, maps), bolts/welds (assess_joints runs the assessment if needed), fatigue, printing (creep warning), assumptions. Without results only the input. subject = introduction, notes = author's conclusion (paragraphs separated by an empty line), required_safety to Re, language 'en' (default) | 'cs'. Returns the .docx path."""
        from ..report import ReportOptions, build_report
        if required_safety <= 0:
            raise FeaError("required_safety must be positive")
        if language not in ("cs", "en"):
            raise FeaError("language must be 'cs' or 'en'")
        with sess.use(op="protokol"):
            st = sess.require_study()
            if sess.results is not None and assess_joints and sess.joints is None and \
                    (st.active_bolts() or st.active_welds()):
                from ..joints.assess import assess_joints as run_assess
                sess.joints = run_assess(st, sess.results, sess.parts)
            if output_path:
                p = user_path(output_path)
            else:
                p = (sess.workdir or sess.base_dir) / f"protokol_{time.strftime('%Y%m%d_%H%M%S')}.docx"
            opts = ReportOptions(title=title, author=author, company=company, subject=subject, notes=notes,
                                 required_safety=float(required_safety), images=images, keep_images=keep_images,
                                 language=language)
            r = build_report(sess, p, opts, on_log=sess.add_log)
            r["with_results"] = sess.results is not None
            r["with_fatigue"] = sess.fatigue is not None
            r["with_joints"] = sess.joints is not None
            return out(r)

    @mcp.tool(structured_output=False)
    def fea_results_export(format: str = "vtu", output_dir: str | None = None,
                           step: int | str | None = None, dialect: str = "standard", elements: bool = False,
                           single_file: bool = False) -> str:
        """Exports results to VTU (ParaView) and/or CSV. format 'vtu'|'csv'|'all'; output_dir (None = export/ in the work folder); step: index from 0, name of a state/combination/envelope (None = all; the step_index column counts from 1); dialect 'standard' (comma, UTF-8) | 'excel' (semicolon, decimal comma, BOM); elements: also <name>_elements.csv (element_id, part_id, type, n1..n10, centroid, volume); single_file: all states in <name>_nodal_all.csv. Ids from 1 as in the solver input (node_id, element_id, part_id = ELSET P<n>). Files <name>_<NN>_<state>, combinations <name>_K<NN>_<name>, reactions <name>_reactions.csv (TOTAL row, moments about mref). VTU has NodeId, ElementId, PartNumber. Returns files and warnings."""
        if format not in ("vtu", "csv", "all"):
            raise FeaError("format must be 'vtu' | 'csv' | 'all'")
        if dialect not in ("standard", "excel"):
            raise FeaError("dialect must be 'standard' | 'excel'")
        with sess.use(op="export_results"):
            st = sess.require_study()
            if sess.results is None:
                raise FeaError("There are no results to export. Run the calculation first.")
            res = sess.results
            if not res.steps or not len(res.mesh.nodes):
                raise FeaError("The results contain no mechanical states on a 3D mesh that could be exported")
            out_p = user_path(output_dir) if output_dir else ((sess.workdir or sess.base_dir) / "export")
            out_p.mkdir(parents=True, exist_ok=True)
            files: list[str] = []
            warns: list[str] = []

            from ..combinations import compute_combination
            from ..export_vtu import (export_combination_vtu, export_results_vtu, export_step_vtu, result_file_name,
                                      safe_name)
            from ..export_csv import export_results_csv
            base = safe_name(st.name)

            # what is exported: states (StepResult) and combinations / envelopes from the study
            combs = list(getattr(st, "combinations", []) or [])
            names = [c.name for c in combs]
            select = None                                 # ("step", i) | ("comb", j) when step is given
            cache: dict = {}
            if step is not None:
                if isinstance(step, str) and step in names:
                    select = ("comb", names.index(step))
                    try:
                        compute_combination(st, res, combs[select[1]], cache)
                    except (ValueError, KeyError) as e:
                        raise FeaError(f"The combination '{step}' cannot be computed: {e}")
                else:
                    select = ("step", sess.step_index(step))

            if format in ("vtu", "all"):
                if select is None:
                    files.extend(str(f) for f in export_results_vtu(res.mesh, res, out_p, base_name=st.name,
                                                                    study=st, warnings=warns))
                elif select[0] == "step":
                    s = res.steps[select[1]]
                    vtu_path = out_p / result_file_name(base, "step", select[1], s.name, "vtu")
                    export_step_vtu(res.mesh, s, vtu_path)
                    files.append(str(vtu_path))
                else:
                    comb = combs[select[1]]
                    vtu_path = out_p / result_file_name(base, "comb", select[1], comb.name, "vtu")
                    export_combination_vtu(res.mesh, compute_combination(st, res, comb, cache), vtu_path)
                    files.append(str(vtu_path))

            if format in ("csv", "all"):
                csv_warns: list[str] = []
                files.extend(str(f) for f in export_results_csv(
                    res, out_p, base_name=st.name, study=st, select=select, dialect=dialect, elements=elements,
                    single_file=single_file, warnings=csv_warns, cache=cache))
                warns += [w for w in csv_warns if w not in warns]

            d = {"format": format, "output_dir": str(out_p), "exported_files": files, "count": len(files)}
            if format in ("csv", "all"):
                d["csv_dialect"] = dialect
            if warns:
                d["warnings"] = warns
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_convergence_history(job_name: str | None = None) -> str:
        """Convergence history of the nonlinear CalculiX run (.cvg, .sta, stdout): increments and iterations, force residuals and displacement corrections, node of the largest residual (with coordinates), reason of divergence. job_name e.g. 'nl_b1', 'nl_seq' (None = all)."""
        with sess.use(op="convergence_history"):
            sess.require_study()
            res = sess.results
            wd = sess.workdir or sess.base_dir
            from ..nl_convergence import read_convergence_job

            if res is not None and "nonlinear" in res.info and "convergence" in res.info["nonlinear"]:
                conv = res.info["nonlinear"]["convergence"]
                if job_name:
                    if job_name in conv:
                        return out(conv[job_name])
                    raise FeaError(f"Job '{job_name}' was not found in the convergence history: {list(conv.keys())}")
                return out(conv)

            if job_name:
                h = read_convergence_job(wd, job_name, mesh=res.mesh if res else None)
                if not h.iterations and not h.increments:
                    raise FeaError(f"No convergence data (.cvg/.sta) were found for the job '{job_name}' in the directory {wd}.")
                return out(h.to_dict())

            cvg_files = list(wd.glob("*.cvg"))
            if not cvg_files:
                raise FeaError(f"No convergence .cvg file was found in the directory {wd}.")
            all_h = {}
            for cf in cvg_files:
                jn = cf.stem
                h = read_convergence_job(wd, jn, mesh=res.mesh if res else None)
                all_h[jn] = h.to_dict()
            return out(all_h)

    @mcp.tool(structured_output=False)
    def fea_topo_opt_run(target_volume_fraction: float = 0.5, filter_radius: float = 6.0,
                         evolution_rate: float = 0.04, max_iterations: int = 60,
                         non_design_selections: list[str] | None = None,
                         output_stl: str | None = None, timeout_s: float = 3600) -> str:
        """Topology optimization (BESO) for lightweighting: each iteration re-solves on the same mesh with element stiffness from current densities; sensitivities = element strain energy summed over load states. Needs constraints, a load state, isotropic elastic material, linear statics; tens of solves, so use a coarse mesh. target_volume_fraction (0.4 = 40 %), filter_radius [mm] (min strut thickness), evolution_rate (per iteration), max_iterations, non_design_selections (material never removed; default all), output_stl (closed STL path), timeout_s. Returns statistics, volume, compliance (full vs result), history, STL path."""
        with sess.use(op="topo_opt"):
            st = sess.require_study()
            mesh = sess.valid_mesh()
            if mesh is None:
                from .tools_mesh import do_mesh
                do_mesh(sess, min(timeout_s, 600))            # returns only the message about the mesh; the mesh is in sess.mesh
                mesh = sess.mesh

            from ..topo import TopologyOptSpec, TopologyOptimizer, export_topo_stl
            from ..export_vtu import safe_name
            try:
                spec = TopologyOptSpec(
                    target_volume_fraction=float(target_volume_fraction),
                    filter_radius=float(filter_radius),
                    evolution_rate=float(evolution_rate),
                    max_iterations=int(max_iterations),
                    non_design_selections=(list(non_design_selections) if non_design_selections is not None
                                           else list(st.selections.keys())),
                )
                opt = TopologyOptimizer(st, mesh, spec)
            except ValueError as e:
                raise FeaError(str(e))
            errors = opt.check_study()
            if errors:
                raise FeaError("The topology optimization cannot be started:\n  " + "\n  ".join(errors))
            run_dir = (sess.workdir or sess.base_dir) / "topo_opt"
            run_dir.mkdir(parents=True, exist_ok=True)

            run_with_timeout(lambda c: opt.run(workdir=run_dir, cancel=c), timeout_s, "topology optimization")

            out_data = {
                "converged": bool(opt.history[-1].converged) if opt.history else False,
                "final_volume_fraction": float(opt.current_volume_fraction),
                "total_iterations": len(opt.history),
                "initial_compliance_Nmm": opt.initial_compliance,
                "final_compliance_Nmm": opt.final_compliance,
                "compliance_history_Nmm": [round(h.total_compliance, 6) for h in opt.history],
                "volume_history": [round(h.volume_fraction, 5) for h in opt.history],
            }

            res = getattr(sess, "results", None)
            if res is not None and res.mesh is mesh:                # results on the same mesh → save them with it
                from ..topo.results import attach
                attach(res, opt)
                out_data["attached_to_results"] = ("the result is attached to the study results (field topo_density, saved with the project)")

            stl_p = user_path(output_stl) if output_stl else (run_dir / f"{safe_name(st.name)}_topo_optimized.stl")
            if stl_p.suffix.lower() != ".stl":            # the client chooses the path – never write another file type
                stl_p = stl_p.with_name(stl_p.name + ".stl")
            stl_p.parent.mkdir(parents=True, exist_ok=True)
            export_topo_stl(mesh, opt.rho, stl_p)
            out_data["stl_path"] = str(stl_p)

            return out(out_data)

    @mcp.tool(structured_output=False)
    def fea_plasticity_assess(step_index: int = 0, standard: str = "en1993", material_class: str = "",
                              eps_Lu: float = 0.0, eps_cf: float = 0.0) -> str:
        """Plastic strain / local failure assessment of a state, per node and material. standard: en1993 (EN 1993-1-5 annex C, PEEQ <= limit_strain, default 5 %) | asme (ASME VIII-2 5.3.3.1, triaxial strain limit, Table 5.7). material_class ferritic | austenitic ('' = from the material name), eps_Lu 0 = Table 5.7, eps_cf = forming strain."""
        with sess.use(op="plasticity assessment"):
            res = sess.require_results()
            if not res.steps or step_index >= len(res.steps):
                raise FeaError(f"Step {step_index} does not exist")
            if res.steps[step_index].fields.get("peeq") is None:
                raise FeaError("There are no plastic strain data (PEEQ) in the given step. Was plasticity defined in the material?")
            from .. import plasticity_eval as PE
            try:
                a = PE.assess(sess.require_study(), res, step_index, standard,
                              PE.AsmeParams(material_class or None, eps_Lu=eps_Lu or None, eps_cf=eps_cf))
            except ValueError as e:
                raise FeaError(str(e)) from None
            d = PE.public(a)
            if d["standard"] == "asme":
                d["max_utilization"] = d["utilization"]       # key of the earlier version of the tool
            return out(d)

