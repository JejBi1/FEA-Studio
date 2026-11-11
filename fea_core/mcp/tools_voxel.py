"""Tools: voxel detail of 3D printing (F5 step 3, fea_core.printing.voxel) – local submodel of the beads
(perimeters, top/bottom layers, infill, thin places) in a box around the critical place, boundary conditions = displacements
of the homogenised solution, elements C3D8I in ccx, Tsai-Wu/Hill in the axes of the bead.

The result (VoxelResults) is kept in the session (sess.voxel) and discarded with a change of the study or of the results. The minimum
safety at an integration point is sensitive to the steps of the voxels (singularities, it falls with refinement) – the main data are
safety_avg (from the average stress of the element) and safety_p1 (1st percentile in the evaluated region).
"""
from __future__ import annotations

import math
import time

import numpy as np

from .common import out, run_with_timeout
from .session import FeaError, Session

FIELDS = ("kind", "safety", "angle", "mises", "umag", "s11", "s22", "s33", "t12", "t13", "t23")
NOTE_MIN = ("The minimum at an integration point typically lies at a voxel step / a corner of beads (a local singularity – it falls with refinement); for evaluation use safety_avg (average element stress) and safety_p1 (1st percentile)")
ORIENT = ("INDICATIVE: the paths are generated from the print settings (not from a slicer), the material = solid extrusion in the bead axes, the bead joints perfect; boundary conditions from the homogenized model")


def _spec(center, size_mm, resolution, margin_mm, part, step, element):
    from ..printing.voxel import VoxelSpec
    if resolution not in ("w/2", "w/4"):
        raise FeaError("resolution must be 'w/2' or 'w/4' (voxel size = line width / 2 or / 4)")
    if element not in ("C3D8I", "C3D8"):
        raise FeaError("element must be C3D8I (recommended) or C3D8")
    try:
        size = [float(v) for v in size_mm]
    except (TypeError, ValueError):
        raise FeaError("size_mm = [a, b, c] mm (in the print axes; c = in the build direction)") from None
    if len(size) != 3 or min(size) <= 0:
        raise FeaError("size_mm = 3 positive numbers [mm]")
    if not (math.isfinite(margin_mm) and 0 <= margin_mm <= 20):
        raise FeaError("margin_mm must be 0–20 mm")
    c = None
    if center is not None and center != "auto":
        try:
            c = [float(v) for v in center]
        except (TypeError, ValueError):
            raise FeaError("center = [x, y, z] mm or 'auto' (the place of the lowest print safety)") from None
        if len(c) != 3:
            raise FeaError("center = [x, y, z] mm")
    return VoxelSpec(center=c, size=size, resolution=resolution, part=part, step=step, margin=float(margin_mm),
                     element=element)


def _results(sess: Session):
    res = sess.require_results()
    if not any("print_safety" in s.fields for s in res.steps):
        raise FeaError("The results have no 3D printing model (a printed material with E_z and fea_solve_run)")
    return res


def _step_arg(sess: Session, step):
    return None if step is None else sess.step_index(step)


def voxel_summary(vr) -> dict:
    """Brief summary for the LLM / protocol: per state safety_avg, safety_p1, min. at the IP, homogenisation, ratios."""
    sm = vr.summary
    steps = []
    for d in sm.get("steps", []):
        r = {"step": d["name"], "safety_avg": d.get("min_safety_element_avg"), "safety_p1": d.get("safety_p1"),
             "min_safety_ip": d.get("min_safety"), "at_mm": d.get("at"), "kind": d.get("kind"),
             "angle_deg": d.get("angle_deg"), "stress_local_MPa": d.get("stress_local"),
             "homogenized_min_safety": d.get("homogenized_min_safety"),
             "max_displacement_mm": d.get("max_displacement_mm")}
        h = d.get("homogenized_min_safety")
        if h:
            r["ratio_avg_to_homogenized"] = (d.get("min_safety_element_avg") or float("nan")) / h
            r["ratio_ip_to_homogenized"] = d.get("ratio")
        if d.get("mean_stress_difference") is not None:
            r["mean_stress_difference"] = d["mean_stress_difference"]
            r["mean_stress_voxel_MPa"] = d.get("mean_stress_voxel")
            r["mean_stress_global_MPa"] = d.get("mean_stress_global")
        if d.get("creep"):
            r["creep"] = d["creep"]
        steps.append(r)
    return {"part": sm.get("part"), "center_mm": sm.get("center"), "size_mm": sm.get("size"), "auto": sm.get("auto"),
            "voxel_mm": sm.get("voxel_mm"), "shape": sm.get("shape"), "voxels": sm.get("voxels"),
            "dof": sm.get("dof"), "element": sm.get("element"), "kinds": sm.get("kinds"),
            "infill_fraction": sm.get("infill_fraction"), "criterion": sm.get("criterion"),
            "bc_nodes": sm.get("bc_nodes"), "time_s": sm.get("time_s"), "steps": steps,
            "notes": [NOTE_MIN, "mean_stress_difference = |σ̄_voxel − σ̄_homog| / |σ̄_homog| in the evaluation region – an indicator of whether the homogenization (calibration of infill / perimeters) matches the detail (≲ 10 % good)",
                      ORIENT],
            "warnings": list(vr.warnings)}


def critical_layer(vr, step: int = 0) -> int:
    """Layer (index z in the grid) of the element with the lowest safety_avg in the evaluated region."""
    st = vr.steps[step]
    v = st.safety_avg if st.safety_avg is not None else st.safety_el
    lam = np.where(vr.inner, v, np.inf)
    return int(vr.ijk[int(np.argmin(lam)), 2])


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_print_voxel_estimate(center: list[float] | str | None = "auto", size_mm: list[float] | None = None,
                                 resolution: str = "w/2", margin_mm: float = 1.0, part: str | None = None,
                                 step: int | str | None = None, element: str = "C3D8I") -> str:
        """Size estimate of the voxel print detail BEFORE running (needs results with the print model). center [x,y,z] or 'auto' (lowest print safety of the homogenised model); size_mm [a,b,c] in print axes (c = build; default [6,6,4]); resolution 'w/2' | 'w/4' (voxel = line width/2 or /4, height = layer); margin_mm (boundary margin, not evaluated); part; step (None = all); element C3D8I (recommended) | C3D8. Returns voxel counts by kind (perimeter, layer, infill, thin place), nodes, equations (C3D8I ~12/voxel), memory, time, ok + message."""
        from ..printing.voxel import VoxelError, estimate
        spec = _spec(center, size_mm or [6.0, 6.0, 4.0], resolution, margin_mm, part, None, element)
        with sess.use(op="estimate of the voxel detail"):
            st = sess.require_study()
            res = _results(sess)
            spec.step = _step_arg(sess, step)
            try:
                e = estimate(st, res, spec)
            except VoxelError as ex:
                raise FeaError(str(ex)) from None
            e["spec"] = spec.to_dict()
            e["next"] = ("fea_print_voxel_run with the same parameters" if e["ok"] else
                         "reduce size_mm / margin_mm or choose resolution='w/2'")
            return out(e)

    @mcp.tool(structured_output=False)
    def fea_print_voxel_run(center: list[float] | str | None = "auto", size_mm: list[float] | None = None,
                            resolution: str = "w/2", margin_mm: float = 1.0, part: str | None = None,
                            step: int | str | None = None, element: str = "C3D8I", timeout_s: float = 1800,
                            output_path: str | None = None, return_image: bool = True):
        """Voxel print detail (parameters as fea_print_voxel_estimate, + timeout_s, output_path, return_image): bead paths from the print settings, boundary displacements from the homogenised solution, ccx C3D8I, Tsai-Wu/Hill in bead axes. Per state returns safety_avg (element-average stress, MAIN) and safety_p1 (1st percentile), integration-point minimum (careful: voxel-step singularities), homogenised safety and ratio, detail-vs-homogenised stress difference (calibration indicator), voxel counts, warnings, and a section image through the critical layer. Result stays for fea_print_voxel_render. INDICATIVE."""
        from mcp.server.mcpserver import Image
        from ..printing.voxel import VoxelError, run_voxel, render_png
        spec = _spec(center, size_mm or [6.0, 6.0, 4.0], resolution, margin_mm, part, None, element)
        with sess.use(op="voxel print detail", wait=5):
            st = sess.require_study()
            res = _results(sess)
            spec.step = _step_arg(sess, step)
            wd = (sess.workdir or sess.base_dir) / f"voxel_{time.strftime('%H%M%S')}"
            logs: list[str] = []
            try:
                vr = run_with_timeout(lambda cancel: run_voxel(st, res, spec, wd, logs.append, cancel), timeout_s,
                                      "Voxel detail")
            except FeaError as e:
                if isinstance(e.__cause__, VoxelError):
                    raise FeaError(str(e.__cause__)) from None
                raise
            finally:
                for line in logs:
                    sess.add_log(line)
            sess.voxel = vr
            sess.voxel_spec = spec.to_dict()
            d = voxel_summary(vr)
            from .tools_results import _png_path
            p = _png_path(sess, output_path, "voxel_safety")
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                render_png(vr, p, "safety", critical_layer(vr), 0)
                d["image_path"] = str(p)
            except Exception as e:  # noqa: BLE001 – the image is a supplement
                d["image_error"] = str(e)
                p = None
            d["next"] = "fea_print_voxel_render(field='kind' | 'safety' | 's33' …, slice_z=…), fea_report_word"
            txt = out(d)
            return [txt, Image(path=p)] if (return_image and p is not None) else txt

    @mcp.tool(structured_output=False)
    def fea_print_voxel_render(field: str = "kind", slice_z: int | str | None = "auto", step: int | str = 0,
                               lines: bool = False, output_path: str | None = None, return_image: bool = True):
        """PNG of the last voxel detail. field: kind (perimeter/layer/infill/thin place) | safety | angle (line angle) | mises | umag | s11 s22 s33 t12 t13 t23 (bead axes: 1 line, 2 across, 3 build; s33, t13, t23 = layer bond). slice_z: layer index, 'auto' = lowest safety_avg, None = isometric surface. lines = bead direction ticks; step = state."""
        from mcp.server.mcpserver import Image
        from ..printing.voxel import render_png
        if field not in FIELDS:
            raise FeaError(f"field: {' | '.join(FIELDS)}")
        with sess.use(op="voxel render"):
            vr = getattr(sess, "voxel", None)
            if vr is None:
                raise FeaError("The voxel detail is not computed (fea_print_voxel_run; a change of the study / results discards it)")
            names = [s.name for s in vr.steps]
            if isinstance(step, str) and not step.strip().lstrip("-").isdigit():
                if step not in names:
                    raise FeaError(f"State '{step}' is not in the detail (states: {', '.join(names)})")
                k = names.index(step)
            else:
                k = int(step) % len(names)
            if slice_z == "auto":
                z = critical_layer(vr, k)
            elif slice_z is None:
                z = None
            else:
                z = int(slice_z)
                if not 0 <= z < vr.shape[2]:
                    raise FeaError(f"slice_z must be 0–{vr.shape[2] - 1} (layers in the region)")
            from .tools_results import _png_path
            p = _png_path(sess, output_path, f"voxel_{field}")
            p.parent.mkdir(parents=True, exist_ok=True)
            render_png(vr, p, field, z, k, lines=bool(lines))
            txt = out({"path": str(p), "field": field, "slice_z": z, "layers": vr.shape[2], "step": names[k]})
            return [txt, Image(path=p)] if return_image else txt
