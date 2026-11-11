"""Tools: results along a line (PNG graph) and linearisation of the stress across a wall (ASME VIII-2, 5-A) – fea_core.paths.

The stress and displacement fields are interpolated with the tet10 shape functions from the nodal S and U (paths.line_values – von Mises
and principal stresses from the interpolated tensor); other fields (safety, fatigue, temperature, heat flux, 3D printing,
error estimate) by interpolation of the nodal values. Points outside the body = null. Graph: matplotlib Agg without pyplot.
"""
from __future__ import annotations

import math
import time

import numpy as np
from mcp.server.mcpserver import Image

from .common import out
from .diagnostics import FIELDS, field_key, field_values, resolve_source, thermal_values
from .session import FeaError, Session

_STRESS = {"mises": "mises", "s1": "s1", "s2": "s2", "s3": "s3", "sxx": "sx", "syy": "sy", "szz": "sz",
           "sxy": "sxy", "syz": "syz", "szx": "szx"}
_DISP = {"displacement": "umag", "ux": "ux", "uy": "uy", "uz": "uz"}


def _vec3(v, what: str) -> np.ndarray:
    try:
        a = np.asarray([float(x) for x in v], float)
    except (TypeError, ValueError):
        raise FeaError(f"{what} must be [x, y, z] in mm") from None
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise FeaError(f"{what} must be [x, y, z] in mm")
    return a


def _interp(res, pts: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Nodal field at the points (shape functions of the element); points outside the mesh NaN (tolerance 2 % – curved surface)."""
    from ..paths import inside
    from ..results import tet10_shape
    outv = np.full(len(pts), np.nan)
    for i, p in enumerate(pts):
        if not inside(res, p, 0.02):
            continue
        e, xi = res.locate(p)
        conn = res.mesh.tets[e]
        N = tet10_shape(xi) if len(conn) == 10 else np.array([1 - xi.sum(), *xi])
        outv[i] = float(N @ vals[conn])
    return outv


def line_data(sess: Session, p0, p1, field: str, step, thermal_case, n: int) -> dict:
    """{"s", "points", "values", "label", "unit", "source"} of the values of the field along the line p0 → p1."""
    from ..paths import line_values
    res = sess.require_results()
    a, b = _vec3(p0, "p0"), _vec3(p1, "p1")
    L = float(np.linalg.norm(b - a))
    if L <= 0:
        raise FeaError("p0 and p1 must be different points")
    if not 2 <= int(n) <= 2001:
        raise FeaError("n must be 2–2001")
    key = field_key(field)
    label, unit = FIELDS[key]
    src, i = resolve_source(sess, key, step, thermal_case)
    if src == "step" and key in _STRESS:
        lv = line_values(res, a, b, _STRESS[key], i, int(n))
        vals = np.asarray(lv["values"], float)
    elif src == "step" and key in _DISP:
        lv = line_values(res, a, b, _DISP[key], i, int(n))
        vals = np.asarray(lv["values"], float)
    else:
        nodal = thermal_values(sess, i, key) if src == "thermal" else field_values(sess, i, key)
        t = np.linspace(0.0, 1.0, int(n))
        lv = {"s": (t * L).tolist(), "points": (a[None] + t[:, None] * (b - a)[None]).tolist()}
        vals = _interp(res, np.asarray(lv["points"]), np.asarray(nodal, float))
    name = res.thermal[i].name if src == "thermal" else res.steps[i].name
    return {"s": np.asarray(lv["s"], float), "points": np.asarray(lv["points"], float), "values": vals,
            "key": key, "label": label, "unit": unit, "source": ("thermal state " if src == "thermal" else "state ") + name,
            "length": L}


def line_plot(path, d: dict, title: str) -> None:
    """Graph of the value along the line (one series, no legend – the title names it); points outside the body = gap."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(9.6, 5.4), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(d["s"], d["values"], color="#1f5fbf", lw=2.0, marker="o" if len(d["s"]) <= 25 else None, ms=4)
    ax.set_xlabel("distance from p0 [mm]", color="#333")
    us = "" if d["unit"] == "–" else f" [{d['unit']}]"
    ax.set_ylabel(d["label"] + us, color="#333")
    ax.set_title(title, fontsize=11, loc="left", color="#111")
    ax.grid(True, color="#dddddd", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
    ax.tick_params(colors="#555555")
    v = d["values"]
    if np.isfinite(v).any():
        k = int(np.nanargmax(np.abs(v)))
        ax.annotate(f"{v[k]:.4g}", (d["s"][k], v[k]), textcoords="offset points", xytext=(6, 6), fontsize=9,
                    color="#333")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_results_line(p0: list[float], p1: list[float], field: str = "mises", step: int | str = 0, n: int = 51,
                         thermal_case: int | str | None = None, plot: bool = False, output_path: str | None = None,
                         return_image: bool = True):
        """Quantity along the line p0 -> p1 [mm] in n points (tet10 interpolation; stress from the interpolated tensor). field as in fea_results_render_image (+ stress_error); thermal_case for heat. Outside the body = null. Returns s_mm, values, min/max with positions; plot=True adds a PNG. E.g. through-wall stress, along a fillet."""
        with sess.use(op="line section"):
            d = line_data(sess, p0, p1, field, step, thermal_case, n)
            v = d["values"]
            ok = np.isfinite(v)
            r = {"field": d["key"], "unit": d["unit"], "source": d["source"], "length_mm": d["length"],
                 "n": len(v), "s_mm": d["s"], "values": v, "points_outside": int((~ok).sum())}
            if ok.any():
                imax, imin = int(np.nanargmax(v)), int(np.nanargmin(v))
                r["max"] = {"value": v[imax], "s_mm": d["s"][imax], "at_mm": d["points"][imax]}
                r["min"] = {"value": v[imin], "s_mm": d["s"][imin], "at_mm": d["points"][imin]}
                r["mean"] = float(np.nanmean(v))
            else:
                raise FeaError("The whole line lies outside the body – check the coordinates p0, p1 (mm)")
            if (~ok).any():
                r["note"] = "Points outside the body (a hole, a gap between parts, outside the model) have the value null"
            if not plot:
                return out(r)
            from .tools_results import _png_path
            p = _png_path(sess, output_path, f"line_{d['key']}")
            title = (f"{d['label']} – {d['source']}: line ({'; '.join((f'{x:.4g}' for x in _vec3(p0, 'p0')))}) → ({'; '.join((f'{x:.4g}' for x in _vec3(p1, 'p1')))}) mm")
            line_plot(p, d, title)
            r["image_path"] = str(p)
            txt = out(r)
            return [txt, Image(path=p)] if return_image else txt

    @mcp.tool(structured_output=False)
    def fea_results_linearize(p0: list[float], p1: list[float], step: int | str = 0, n: int = 41,
                              hint: list[float] | None = None, allowable_S_MPa: float | None = None) -> str:
        """Stress linearization on the line p0 -> p1 across the wall (ASME VIII-2 5-A): membrane P_m, bending P_b (normal stresses perpendicular to the SCL), peak F at both ends; von Mises and Tresca equivalents. The line must lie in the material, perpendicular to the wall. Tensors in global and SCL axes (a along SCL; hint = direction of axis b, e.g. [0,1,0]). allowable_S_MPa gives an informative check P_m <= S, P_m + P_b <= 1.5 S (5.2.2). Warns for < 3 elements across the thickness. Nodal (averaged) stresses."""
        from ..paths import linearize
        if not 3 <= int(n) <= 401:
            raise FeaError("n must be 3–401")
        a, b = _vec3(p0, "p0"), _vec3(p1, "p1")
        if np.linalg.norm(b - a) <= 0:
            raise FeaError("p0 and p1 must be different points")
        h = _vec3(hint, "hint") if hint is not None else None
        if h is not None and np.linalg.norm(h) == 0:
            raise FeaError("hint must be a nonzero vector")
        if allowable_S_MPa is not None and not (math.isfinite(allowable_S_MPa) and allowable_S_MPa > 0):
            raise FeaError("allowable_S_MPa must be positive [MPa]")
        with sess.use(op="linearizace"):
            sess.require_results()
            i = sess.step_index(step)
            t0 = time.time()
            try:
                lin = linearize(sess.results, a, b, i, int(n), h)
            except ValueError as e:
                raise FeaError(str(e)) from None
            ends = lin["ends"]
            r = {"step": sess.results.steps[i].name, "length_mm": lin["length_mm"], "axes_local": lin["axes"],
                 "axes_note": "rows = axes a (along the SCL), b, c in global coordinates; Voigt xx yy zz xy yz zx",
                 "Pm_MPa": {"mises": lin["membrane"]["mises"], "tresca": lin["membrane"]["tresca"],
                            "local": lin["membrane"]["local"], "global": lin["membrane"]["global"]},
                 "Pb_MPa": {"local": lin["bending"]["local"], "global": lin["bending"]["global"]},
                 "ends": {k: {"PmPb": {"mises": e["membrane_bending"]["mises"], "tresca": e["membrane_bending"]["tresca"]},
                              "total": {"mises": e["total"]["mises"], "tresca": e["total"]["tresca"]},
                              "peak": {"mises": e["peak"]["mises"], "tresca": e["peak"]["tresca"]}}
                          for k, e in ends.items()},
                 "PmPb_mises_max_MPa": lin["PmPb_mises"], "total_mises_max_MPa": lin["total_mises"],
                 "time_s": time.time() - t0, "warnings": []}
            if "warning" in lin:
                r["warnings"].append(lin["warning"])
            if allowable_S_MPa is not None:
                S = float(allowable_S_MPa)
                pm = lin["membrane"]["tresca"]
                pmpb = max(e["membrane_bending"]["tresca"] for e in ends.values())
                r["check_informative"] = {
                    "S_MPa": S, "Pm_tresca_over_S": pm / S, "PmPb_tresca_over_1_5S": pmpb / (1.5 * S),
                    "ok": pm <= S and pmpb <= 1.5 * S,
                    "note": "ASME VIII-2, 5.2.2 (primary stress; whether it is a primary / secondary stress is decided by the analyst – for supports and thermal stresses it is secondary Q with the limit 3·S)"}
            r["warnings"].append("Linearization only makes sense on a line perpendicular to the wall away from a singularity; in a sharp corner and at a support the peak stress depends on the mesh")
            return out(r)
