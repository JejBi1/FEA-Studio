"""Tools: simplified models of the 3D study – 2D calculation by a section (fea_core.twod: axisymmetry CAX6, plane stress
CPS6, plane strain CPE6) and the shell model of plates and sheets (fea_core.shell: S6, unrolling C3D15).

Both are switched on in the study (Study.twod / Study.shell) and calculated in fea_solve_run (without a 3D mesh); the results are
in Results.twod (Results2D) or Results.shell (ShellResults), the 3D mesh of the result is empty. Constraints and loads
are entered on the faces of the 3D geometry as in the 3D calculation – they are converted to the edges of the section / the shell (what cannot be converted is reported).
"""
from __future__ import annotations

import math
import time

import numpy as np

from .common import out
from .session import FeaError, Session

TWOD_FIELDS = ("mises", "tresca", "s1", "s2", "s3", "sx", "sy", "sz", "sxy", "ux", "uy", "umag",
               "sr", "sa", "st", "srz", "ur", "ua")
SHELL_FIELDS = ("mises", "s1", "s3", "sxx", "syy", "szz", "sxy", "syz", "szx", "membrane", "bending", "u", "ux", "uy",
                "uz", "thickness")
SHELL_LAYERS = ("top", "mid", "bot", "max")
XDIR_HINT = ("for a sector (e.g. a quarter of a rotational part) choose xdir OUTSIDE the symmetry planes – e.g. [1, 1, 0] – so that the section plane passes through the material and not along the edge of the sector")


def _vec3(v, what: str) -> list[float]:
    try:
        a = [float(x) for x in v]
    except (TypeError, ValueError):
        raise FeaError(f"{what} must be [x, y, z]") from None
    if len(a) != 3 or not all(math.isfinite(x) for x in a):
        raise FeaError(f"{what} must be [x, y, z]")
    return a


def _twod(sess: Session):
    res = sess.require_results()
    r2 = getattr(res, "twod", None)
    if r2 is None:
        raise FeaError("The results contain no 2D calculation (fea_twod_set(enabled=True) + fea_solve_run)")
    return r2


def _shell(sess: Session):
    res = sess.require_results()
    rs = getattr(res, "shell", None)
    if rs is None:
        raise FeaError("The results contain no shell model (fea_shell_set(enabled=True) + fea_solve_run)")
    return rs


def _step_of(steps_names: list[str], step) -> int:
    if isinstance(step, str) and not step.strip().lstrip("-").isdigit():
        if step not in steps_names:
            raise FeaError(f"Load case '{step}' is not in the results (cases: {', '.join(steps_names)})")
        return steps_names.index(step)
    i = int(step)
    if not -len(steps_names) <= i < len(steps_names):
        raise FeaError(f"State index {i} out of range 0..{len(steps_names) - 1}")
    return i % len(steps_names)


def _pt2d(r2, p) -> np.ndarray:
    """Point 2D [x, y] or 3D [x, y, z] (converted into the plane of the section – for axisym the radius and the coordinate along the axis)."""
    try:
        q = [float(v) for v in p]
    except (TypeError, ValueError):
        raise FeaError("point = [x, y] in the section plane or [x, y, z] in 3D") from None
    if len(q) == 2:
        return np.asarray(q, float)
    if len(q) == 3:
        return np.asarray(r2.frame2d().to2d(np.asarray(q, float)), float)
    raise FeaError("point = [x, y] in the section plane or [x, y, z] in 3D")


def twod_labels(r2) -> dict:
    if r2.axisym:
        return {"axes": "2D x = radius r, y = coordinate along the axis; σ: sx = σr, sy = axial σ, sz = σθ (hoop)",
                "reactions": "R = [radial, axial] FOR THE WHOLE CIRCUMFERENCE [N] (the radial sum has no meaning of a resultant)"}
    return {"axes": "2D axes x, y of the section plane (x = xdir), sz perpendicular to the plane",
            "reactions": f"R = [x, y] [N] for the whole thickness {r2.thickness:g} mm"}


def twod_step(r2, i: int, field: str = "mises") -> dict:
    s = r2.steps[i]
    vals = r2.field(field, i)
    fr = r2.frame2d()
    jmax, jmin = int(np.nanargmax(vals)), int(np.nanargmin(vals))
    sm = r2.summary()["steps"][i]
    d = {"step": s["name"], "max_mises_MPa": sm["max_mises"], "max_mises_at_2d": sm["max_mises_at"],
         "max_displacement_mm": sm["max_disp"], "max_displacement_at_2d": sm["max_disp_at"],
         "field": field, "field_max": {"value": float(vals[jmax]), "at_2d": r2.nodes[jmax],
                                       "at_3d": fr.to3d(r2.nodes[jmax])},
         "field_min": {"value": float(vals[jmin]), "at_2d": r2.nodes[jmin], "at_3d": fr.to3d(r2.nodes[jmin])},
         "reactions_N": {k: list(map(float, v)) for k, v in s["reactions"].items()},
         "equilibrium": {"imbalance": float(s["equilibrium"].get("imbalance", 0.0)),
                         "ok": float(s["equilibrium"].get("imbalance", 0.0)) <= 1e-3}}
    from ..results import von_mises
    vm = von_mises(s["S"])
    parts = []
    for pi, pn in enumerate(r2.part_names):
        nodes = np.unique(r2.tri[r2.tri_part == pi])
        if len(nodes):
            parts.append({"part": pn, "max_mises_MPa": float(vm[nodes].max())})
    d["parts"] = parts
    return d


def shell_step(rs, i: int, field: str = "mises", layer: str = "max") -> dict:
    st = rs.steps[i]
    sm = rs.summary()["steps"][i]
    vals = rs.field(field, i, layer)
    jmax, jmin = int(np.nanargmax(vals)), int(np.nanargmin(vals))
    d = {"step": st.name, "max_displacement_mm": sm["u_max"], "max_displacement_at_mm": sm["u_max_at"],
         "mises_max_MPa": sm["mises_max"], "mises_max_at_mm": sm["mises_max_at"], "mises_top_max_MPa":
         sm["mises_top_max"], "mises_bot_max_MPa": sm["mises_bot_max"], "membrane_max_MPa": sm["membrane_max"],
         "bending_max_MPa": sm["bending_max"], "field": field, "layer": layer,
         "field_max": {"value": float(vals[jmax]), "at_mm": rs.nodes[jmax]},
         "field_min": {"value": float(vals[jmin]), "at_mm": rs.nodes[jmin]},
         "reactions": {k: {"force_N": list(map(float, r[:3])), "moment_Nmm": list(map(float, r[3:]))}
                       for k, r in st.reactions.items()},
         "reaction_moment_point_mm": rs.ref_point,
         "equilibrium": {"imbalance": float(st.equilibrium.get("imbalance", 0.0)),
                         "ok": float(st.equilibrium.get("imbalance", 0.0)) <= 1e-3}}
    from ..shell.results import von_mises
    vt, vb = von_mises(st.S_top), von_mises(st.S_bot)
    parts = []
    for pi, pn in enumerate(rs.mesh.part_names):
        nodes = np.unique(rs.tri[rs.mesh.tri_part == pi])
        if len(nodes):
            parts.append({"part": pn, "thickness_mm": float(rs.mesh.thickness[pi]),
                          "mises_top_max_MPa": float(vt[nodes].max()), "mises_bot_max_MPa": float(vb[nodes].max())})
    d["parts"] = parts
    return d


def register(mcp, sess: Session) -> None:

    # ================================================================= 2D section
    @mcp.tool(structured_output=False)
    def fea_twod_set(enabled: bool = True, kind: str = "axisym", origin: list[float] | None = None,
                     axis: list[float] | None = None, xdir: list[float] | None = None, thickness: float = 1.0,
                     parts: list[str] | None = None, size: float | None = None) -> str:
        """2D model by a SECTION of the 3D study (fast, exact for rotational/long parts). kind "axisym" (CAX6; origin on the axis, axis = rotation axis, xdir = radial direction of the half-plane) | "plane_stress" (CPS6, thickness [mm]) | "plane_strain" (CPE6); planar: origin in the plane, axis = plane normal, xdir = 2D x. parts (None = all), size [mm] (None = study mesh). Constraints/loads are given on 3D faces and converted to section edges (clamp, displacement, pressure, force, weight, rotation; others = error). Axisym forces/reactions are for the WHOLE circumference. For a sector choose xdir OUTSIDE the symmetry planes (e.g. [1,1,0]). enabled=False = 3D calculation."""
        from ..twod.model import KINDS, TwoDError, TwoDSettings, make_frame
        if kind not in KINDS:
            raise FeaError(f"kind must be {' | '.join(KINDS)}")
        with sess.use():
            st = sess.require_study()
            if sess.frame_only:
                raise FeaError("The study has no geometry – a 2D section needs a STEP")
            cur = st.twod
            s = TwoDSettings(enabled=bool(enabled), kind=kind,
                             origin=_vec3(origin, "origin") if origin is not None else list(cur.origin),
                             axis=_vec3(axis, "axis") if axis is not None else list(cur.axis),
                             xdir=_vec3(xdir, "xdir") if xdir is not None else list(cur.xdir),
                             thickness=float(thickness), parts=list(parts or []),
                             size=None if size is None else float(size))
            if size is not None and not size > 0:
                raise FeaError("size must be positive [mm]")
            for p in s.parts:
                if p not in sess.part_names():
                    raise FeaError(f"Part '{p}' does not exist")
            try:
                fr = make_frame(s)
            except TwoDError as e:
                raise FeaError(str(e)) from None
            if enabled and st.shell.enabled:
                raise FeaError("The shell model is on – first fea_shell_set(enabled=False)")
            st.twod = s
            sess.changed()
            d = {"twod": s.to_dict(), "frame": {"origin": fr.origin, "e1 (2D x)": fr.e1, "e2 (2D y)": fr.e2,
                                               "n (normal to the cut)": fr.n}, "hint_xdir": XDIR_HINT}
            if enabled:
                d["next"] = "fea_solve_run → fea_twod_results, fea_twod_render, fea_twod_probe, fea_twod_line"
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_twod_results(step: int | str = 0, field: str = "mises") -> str:
        """2D results: task kind, mesh, per state max von Mises and displacement (2D place), field extremes (2D + 3D point), reactions per selection (axisym: whole circumference), equilibrium, max stress per part, converted loads, section areas (3D/2D volume ratio), warnings. field: mises, tresca, s1-s3, sx, sy, sz, sxy, ux, uy, umag; axisym also sr, sa (axial), st (theta), srz, ur, ua."""
        if field not in TWOD_FIELDS:
            raise FeaError(f"field: {' | '.join(TWOD_FIELDS)}")
        with sess.use():
            r2 = _twod(sess)
            i = _step_of([s["name"] for s in r2.steps], step)
            sm = r2.summary()
            d = {"kind": sm["kind"], "kind_label": sm["kind_label"], "nodes": sm["nodes"], "elements": sm["elements"],
                 "parts": sm["parts"], "thickness_mm": sm["thickness_mm"], **twod_labels(r2),
                 "steps": [s["name"] for s in r2.steps], "result": twod_step(r2, i, field),
                 "sections": r2.info.get("sections"), "loads_converted": r2.info.get("loads"),
                 "warnings": r2.warnings}
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_twod_probe(point: list[float] | list[list[float]], step: int | str = 0) -> str:
        """2D values at point(s): [x,y] in the section plane or [x,y,z] in 3D (converted; axisym via radius and axial coordinate). Returns displacement, stress components, von Mises, principal stresses; outside the mesh = nearest element + warning."""
        from ..results import principal, von_mises
        with sess.use():
            r2 = _twod(sess)
            i = _step_of([s["name"] for s in r2.steps], step)
            pts = point if (isinstance(point, list) and point and isinstance(point[0], (list, tuple))) else [point]
            if len(pts) > 200:
                raise FeaError("At most 200 points at a time")
            names = ("sr", "sz_axial", "stheta", "srz", "syz", "szx") if r2.axisym else \
                ("sx", "sy", "sz", "sxy", "syz", "szx")
            rows = []
            for p in pts:
                q = _pt2d(r2, p)
                e, xi = r2.locate(q)
                S = r2.probe(q, "S", i)
                U = r2.probe(q, "U", i)
                row = {"point_2d": q, "u_mm": U, "stress_MPa": dict(zip(names, S)), "mises_MPa": float(von_mises(S)),
                       "principal_MPa": principal(S)}
                if min(xi.min(), 1 - xi.sum()) < -1e-6:
                    row["warning"] = "the point lies outside the 2D mesh – the value from the nearest element (extrapolation)"
                rows.append(row)
            return out({"step": r2.steps[i]["name"], "kind": r2.kind, "probes": rows})

    @mcp.tool(structured_output=False)
    def fea_twod_line(p0: list[float], p1: list[float], field: str = "mises", step: int | str = 0, n: int = 51) -> str:
        """2D field along p0 -> p1 (2D [x,y] or 3D points); outside the mesh = null. field as fea_twod_results. E.g. through-wall course."""
        if field not in TWOD_FIELDS:
            raise FeaError(f"field: {' | '.join(TWOD_FIELDS)}")
        if not 2 <= int(n) <= 2001:
            raise FeaError("n must be 2–2001")
        with sess.use():
            r2 = _twod(sess)
            i = _step_of([s["name"] for s in r2.steps], step)
            a, b = _pt2d(r2, p0), _pt2d(r2, p1)
            if np.linalg.norm(b - a) <= 0:
                raise FeaError("p0 and p1 must be different points")
            lv = r2.line_values(a, b, field, i, int(n))
            v = np.asarray(lv["values"], float)
            ok = np.isfinite(v)
            if not ok.any():
                raise FeaError("The line lies entirely outside the 2D mesh")
            j, k = int(np.nanargmax(v)), int(np.nanargmin(v))
            return out({"step": r2.steps[i]["name"], "field": field, "p0_2d": a, "p1_2d": b, "s_mm": lv["s"],
                        "values": v, "max": {"value": v[j], "s_mm": lv["s"][j]},
                        "min": {"value": v[k], "s_mm": lv["s"][k]}, "points_outside": int((~ok).sum())})

    @mcp.tool(structured_output=False)
    def fea_twod_linearize(p0: list[float], p1: list[float], step: int | str = 0, n: int = 41) -> str:
        """Stress linearization of the 2D result across the wall p0 -> p1 (ASME VIII-2 5-A, as fea_results_linearize): P_m, P_m + P_b, peak, von Mises/Tresca; axis b = z (axisym: circumferential). 2D or 3D points."""
        if not 3 <= int(n) <= 401:
            raise FeaError("n must be 3–401")
        with sess.use():
            r2 = _twod(sess)
            i = _step_of([s["name"] for s in r2.steps], step)
            a, b = _pt2d(r2, p0), _pt2d(r2, p1)
            try:
                lin = r2.linearize(a, b, i, int(n))
            except ValueError as e:
                raise FeaError(str(e)) from None
            d = {"step": r2.steps[i]["name"], "length_mm": lin["length_mm"], "points_2d": lin["points"],
                 "Pm_MPa": {k: lin["membrane"][k] for k in ("mises", "tresca", "local")},
                 "Pb_MPa": {"local": lin["bending"]["local"]},
                 "ends": {k: {"PmPb": {"mises": e["membrane_bending"]["mises"], "tresca": e["membrane_bending"]["tresca"]},
                              "total": {"mises": e["total"]["mises"]}, "peak": {"mises": e["peak"]["mises"]}}
                          for k, e in lin["ends"].items()},
                 "PmPb_mises_max_MPa": lin["PmPb_mises"], "total_mises_max_MPa": lin["total_mises"],
                 "axes_note": "local axes a (along the SCL), b = z / θ, c; Voigt aa bb cc ab bc ca",
                 "warnings": [lin["warning"]] if lin.get("warning") else []}
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_twod_render(field: str = "mises", step: int | str = 0, mirror: bool | None = None, scale: float = 0.0,
                        mesh_lines: bool = False, output_path: str | None = None, return_image: bool = True):
        """PNG of the 2D result: colour bands of `field` (as fea_twod_results), max/min, outline; mirror (axisym: mirrored half, None = yes); scale: deformation magnification (0 = undeformed); mesh_lines."""
        from mcp.server.mcpserver import Image
        from .tools_results import _png_path
        if field not in TWOD_FIELDS:
            raise FeaError(f"field: {' | '.join(TWOD_FIELDS)}")
        with sess.use(op="render"):
            r2 = _twod(sess)
            i = _step_of([s["name"] for s in r2.steps], step)
            p = _png_path(sess, output_path, f"twod_{field}")
            p.parent.mkdir(parents=True, exist_ok=True)
            r2.render_png(p, field, i, mirror=mirror, scale=float(scale), mesh_lines=bool(mesh_lines))
            vals = r2.field(field, i)
            txt = out({"path": str(p), "field": field, "step": r2.steps[i]["name"], "kind": r2.kind,
                       "max": float(np.nanmax(vals)), "min": float(np.nanmin(vals))})
            return [txt, Image(path=p)] if return_image else txt

    # ================================================================= shells
    @mcp.tool(structured_output=False)
    def fea_shell_set(enabled: bool = True, parts: dict | None = None, size: float | None = None,
                      offset: str = "mid") -> str:
        """Shell model of thin-walled parts (ccx S6 expanded to C3D15) instead of the 3D mesh. parts {part: thickness [mm] | null}: null = thickness from the body (plate/sheet recognized, see fea_shell_detect); a STEP surface part needs a thickness; None/{} = all. size [mm] (None = study mesh); offset of a surface part relative to the shell mid | top | bot. T/L joints joined by extending the mid-surface by t/2. Constraints: clamp, displacement; loads: force, pressure, weight, rotation (on 3D faces). No bolts, welds, heat, printing, orthotropy. enabled=False = 3D."""
        from ..shell.model import ShellSettings
        with sess.use():
            st = sess.require_study()
            if sess.frame_only:
                raise FeaError("The study has no geometry – shells need a STEP")
            pp = {}
            for k, v in (parts or {}).items():
                if k not in sess.part_names():
                    raise FeaError(f"Part '{k}' does not exist")
                pp[k] = None if v is None else float(v)
            s = ShellSettings(enabled=bool(enabled), parts=pp, size=None if size is None else float(size),
                              offset=offset)
            errs = s.check()
            if errs:
                raise FeaError("; ".join(errs))
            if enabled and st.twod.enabled:
                raise FeaError("The 2D calculation is on – first fea_twod_set(enabled=False)")
            st.shell = s
            sess.changed()
            d = {"shell": s.to_dict()}
            if enabled:
                from ..shell.run import check_study
                errs = check_study(st, s)
                if errs:
                    d["readiness_errors"] = errs
                d["next"] = "fea_shell_detect (kontrola desek), fea_solve_run → fea_shell_results, fea_shell_render"
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_shell_detect(part: str | None = None) -> str:
        """Checks whether a part is a constant-thickness plate/sheet for the shell model: kind (plate | sheet with bends), thickness t, mid-surface area, warnings, or the rejection reason. part None = all."""
        from ..shell.geometry import detect_sheet
        with sess.use(op="plate recognition"):
            sess.require_study()
            names = [part] if part else sess.part_names()
            outl = []
            for n in names:
                try:
                    p = sess.parts[sess.part_index(n)]
                except FeaError:
                    raise
                t0 = time.time()
                info, why = detect_sheet(p)
                row = {"part": n, "is_sheet": info is not None, "time_s": time.time() - t0}
                if info is not None:
                    row.update(kind=info.kind, thickness_mm=info.thickness, mid_area_mm2=info.mid_area,
                               warnings=info.warnings)
                else:
                    row["reason"] = why
                outl.append(row)
            return out({"parts": outl, "note": "the thickness can be given by hand in fea_shell_set(parts={part: t})"})

    @mcp.tool(structured_output=False)
    def fea_shell_results(step: int | str = 0, field: str = "mises", layer: str = "max") -> str:
        """Shell results: parts and thicknesses, extensions (T joints), S6 mesh; per state max displacement, von Mises top/bottom, membrane and bending ((top-bottom)/2), field extremes on `layer` (top|mid|bot|max), max per part, reactions per selection (force, moment about reaction_moment_point_mm), equilibrium, warnings. field: mises, s1, s3, sxx..szx, membrane, bending, u, ux, uy, uz, thickness. 'top' = side of the element normal."""
        if field not in SHELL_FIELDS:
            raise FeaError(f"field: {' | '.join(SHELL_FIELDS)}")
        if layer not in SHELL_LAYERS:
            raise FeaError(f"layer: {' | '.join(SHELL_LAYERS)}")
        with sess.use():
            rs = _shell(sess)
            i = _step_of([s.name for s in rs.steps], step)
            sm = rs.summary()
            return out({"nodes": sm["nodes"], "elements": sm["elements"], "element": "S6 (expansion to C3D15)",
                        "parts": sm["parts"], "sheets": rs.info.get("sheets"), "steps": [s.name for s in rs.steps],
                        "result": shell_step(rs, i, field, layer), "warnings": rs.warnings,
                        "units": "mm, N, MPa; reakce [Fx Fy Fz] N, [Mx My Mz] N·mm"})

    @mcp.tool(structured_output=False)
    def fea_shell_probe(point: list[float] | list[list[float]], field: str = "mises", step: int | str = 0,
                        layer: str = "top") -> str:
        """Shell field at point(s) [x,y,z] mm: nearest mid-surface point (reports distance), tri6 interpolation; layer top|mid|bot|max, field as fea_shell_results."""
        if field not in SHELL_FIELDS:
            raise FeaError(f"field: {' | '.join(SHELL_FIELDS)}")
        if layer not in SHELL_LAYERS:
            raise FeaError(f"layer: {' | '.join(SHELL_LAYERS)}")
        with sess.use():
            rs = _shell(sess)
            i = _step_of([s.name for s in rs.steps], step)
            pts = point if (isinstance(point, list) and point and isinstance(point[0], (list, tuple))) else [point]
            rows = [rs.probe(_vec3(p, "point"), field, i, layer) for p in pts[:200]]
            return out({"step": rs.steps[i].name, "probes": rows})

    @mcp.tool(structured_output=False)
    def fea_shell_render(field: str = "mises", step: int | str = 0, layer: str = "max", deform: bool | float = False,
                         output_path: str | None = None, return_image: bool = True):
        """PNG of the shell model: `field` on the mid-surface for `layer` (top|mid|bot|max); deform True (auto 5 % of size) | scale number | False."""
        from mcp.server.mcpserver import Image
        from .tools_results import _png_path
        if field not in SHELL_FIELDS:
            raise FeaError(f"field: {' | '.join(SHELL_FIELDS)}")
        if layer not in SHELL_LAYERS:
            raise FeaError(f"layer: {' | '.join(SHELL_LAYERS)}")
        with sess.use(op="render"):
            rs = _shell(sess)
            i = _step_of([s.name for s in rs.steps], step)
            p = _png_path(sess, output_path, f"shell_{field}_{layer}")
            p.parent.mkdir(parents=True, exist_ok=True)
            rs.render_png(p, field, i, layer, deform=deform)
            vals = rs.field(field, i, layer)
            txt = out({"path": str(p), "field": field, "layer": layer, "step": rs.steps[i].name,
                       "max": float(np.nanmax(vals)), "min": float(np.nanmin(vals))})
            return [txt, Image(path=p)] if return_image else txt
