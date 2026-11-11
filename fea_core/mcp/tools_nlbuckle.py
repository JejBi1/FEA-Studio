"""Tools: nonlinear buckling with a geometric imperfection (GNIA / GMNIA) – fea_core.imperfect.

Settings in Study.nlbuckling (they do not change the mesh); the analysis runs in fea_solve_run after the statics (own ccx job on an
imperfect copy of the mesh). Results: Results.info["nlbuckling"] (curve, ultimate factor) + Results.modes set of kind "nlbuckle"
(frame 0 = imperfection, then displacements of the increments) – image with fea_results_render_mode(kind="nonlinear").
"""
from __future__ import annotations

import dataclasses
import math

from mcp.server.mcpserver import Image

from .common import out
from .session import FeaError, Session, user_path


def _vec(v, name: str):
    if v is None:
        return None
    try:
        vals = [float(x) for x in v]
    except (TypeError, ValueError):
        raise FeaError(f"{name} must be [x, y, z]") from None
    if len(vals) != 3 or not all(math.isfinite(x) for x in vals):
        raise FeaError(f"{name} must be [x, y, z]")
    return vals


def _summary(info: dict, points: int = 40) -> dict:
    c = info.get("curve") or {}
    lam, u = list(c.get("lambda") or []), list(c.get("u_mm") or [])
    step = max(1, len(lam) // points)
    idx = sorted(set(list(range(0, len(lam), step)) + ([len(lam) - 1] if lam else [])))
    r = info.get("result") or {}
    return {"kind": info.get("kind"), "case": info.get("case"), "ultimate_load_factor": r.get("ultimate_factor"),
            "status": r.get("status"), "status_text": info.get("status_text"),
            "linear_buckling_factor": info.get("linear_factor"), "knockdown_lambda_u_over_lambda_cr": r.get("knockdown"),
            "first_yield_factor": r.get("first_yield_factor"), "tangent_ratio_end": r.get("tangent_ratio_end"),
            "max_factor": info.get("max_factor"), "finished": info.get("finished"), "increments": r.get("increments"),
            "imperfection": info.get("imperfection"), "monitor": info.get("monitor"),
            "load_resultant_at_lambda_1_N": info.get("load_resultant_N"),
            "curve": {"lambda": [lam[i] for i in idx], "u_mm": [u[i] for i in idx], "points_total": len(lam)},
            "shape_frames_lambda": info.get("frames_lambda"), "warnings": info.get("warnings", [])}


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_nlbuckling_set(enabled: bool = True, case_name: str | None = None, imperfection: str = "modes",
                           modes: list[int] | None = None, weights: list[float] | None = None,
                           amplitude_mm: float | None = None, amplitude_ratio: float | None = None,
                           length_mm: float | None = None, sign: float = 1.0, bow_a: list[float] | None = None,
                           bow_b: list[float] | None = None, bow_direction: list[float] | None = None,
                           shape_file: str | None = None, max_factor: float | None = None, increments: int = 20,
                           monitor_point: list[float] | None = None, monitor_dir: list[float] | None = None) -> str:
        """Nonlinear buckling with imperfection (GNIA; GMNIA with plastic materials), run in fea_solve_run: imperfect mesh copy + NLGEOM load control on case_name x lambda.
        imperfection: modes (linear buckling modes `modes` 1-based x `weights`) | bow (half-sine between bow_a, bow_b towards bow_direction) | file (CSV x,y,z,ux,uy,uz) | none.
        Amplitude e0 = amplitude_mm, or L/amplitude_ratio (L = length_mm, default largest model size); sign -1 flips. max_factor = lambda max (None = 1.1 x linear lambda_cr); increments = initial count.
        monitor_point/monitor_dir = curve point (None = node of max imperfection). No arc-length in CalculiX: ultimate = last converged increment.
        The imperfection amplitude per design code (EN 1993-1-5 C, 1993-1-6, 1993-1-1 5.3) is the user's responsibility. enabled=False turns off."""
        from ..imperfect.model import IMPERFECTION_KINDS, NonlinearBucklingSettings
        kind = str(imperfection).strip().lower()
        if kind not in IMPERFECTION_KINDS:
            raise FeaError(f"imperfection must be one of {', '.join(IMPERFECTION_KINDS)}")
        if int(increments) != increments or not 2 <= int(increments) <= 10000:
            raise FeaError("increments must be an integer 2–10000")
        if max_factor is not None and not (math.isfinite(max_factor) and max_factor > 0):
            raise FeaError("max_factor must be > 0 (or None = 1.1 × linear λ_cr)")
        for v, nm in ((amplitude_mm, "amplitude_mm"), (amplitude_ratio, "amplitude_ratio"), (length_mm, "length_mm")):
            if v is not None and not (math.isfinite(v) and v > 0):
                raise FeaError(f"{nm} must be > 0")
        ms = [int(m) for m in (modes or [1])]
        if any(m < 1 or m > 50 for m in ms):
            raise FeaError("modes must be numbers 1–50")
        ws = [float(w) for w in (weights or [1.0] * len(ms))]
        if len(ws) != len(ms) or not all(math.isfinite(w) for w in ws):
            raise FeaError("weights must have one finite number per mode")
        if kind == "bow" and (bow_a is None or bow_b is None or bow_direction is None):
            raise FeaError("imperfection 'bow' needs bow_a, bow_b and bow_direction")
        path = None
        if kind == "file":
            if not shape_file:
                raise FeaError("imperfection 'file' needs shape_file")
            path = str(user_path(shape_file))
        with sess.use():
            st = sess.require_study()
            names = [lc.name for lc in st.load_cases]
            if enabled:
                if not names:
                    raise FeaError("Nonlinear buckling needs a load case with a load (fea_loadcase_create + fea_load_add_*)")
                if case_name is not None and case_name not in names:
                    raise FeaError(f"Load case '{case_name}' does not exist (cases: {', '.join(names)})")
            old = st.nlbuckling
            amp = float(amplitude_mm) if amplitude_mm else 0.0
            st.nlbuckling = dataclasses.replace(
                old if enabled else NonlinearBucklingSettings(), enabled=bool(enabled), case=case_name, imperfection=kind,
                modes=ms, weights=ws, amplitude=amp,
                amplitude_ratio=float(amplitude_ratio) if amplitude_ratio else (old.amplitude_ratio if not amp else 300.0),
                length=float(length_mm) if length_mm else None, sign=-1.0 if float(sign) < 0 else 1.0,
                bow_a=_vec(bow_a, "bow_a"), bow_b=_vec(bow_b, "bow_b"), bow_direction=_vec(bow_direction, "bow_direction"),
                shape_file=path, max_factor=float(max_factor) if max_factor else None, increments=int(increments),
                monitor_point=_vec(monitor_point, "monitor_point"), monitor_dir=_vec(monitor_dir, "monitor_dir"))
            if not enabled:
                st.nlbuckling = NonlinearBucklingSettings()
            sess.changed()                       # the mesh stays (settings are not in the mesh signature)
            d = {"nlbuckling": st.nlbuckling.to_dict()}
            if enabled:
                d["case_used"] = case_name or (st.buckling.case if st.buckling.enabled and st.buckling.case in names
                                               else names[0])
                plastic = any(getattr(m, "is_plastic", False) for m in st.materials.values())
                d["analysis"] = "GMNIA (plastic material)" if plastic else "GNIA (elastic material)"
                d["warning"] = ("The imperfection shape and amplitude are the user's responsibility (design code); "
                                "load control – the ultimate factor is the last converged increment")
                d["next"] = "fea_solve_run → fea_results_nlbuckling, fea_results_render_mode(kind='nonlinear', mode=…)"
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_results_nlbuckling(plot: bool = False, output_path: str | None = None, points: int = 40):
        """Nonlinear buckling result: ultimate load factor lambda_u, status (limit_point | stopped | max_factor), linear lambda_cr and knockdown, first yield factor, imperfection, monitored point, load-displacement curve (subsampled to `points`). plot=True adds a PNG of the curve."""
        from .tools_results import _png_path
        with sess.use(op="render" if plot else None):
            res = sess.require_results()
            info = (getattr(res, "info", None) or {}).get("nlbuckling")
            if not info:
                raise FeaError("The results contain no nonlinear buckling – enable fea_nlbuckling_set and run fea_solve_run")
            d = _summary(info, max(int(points), 2))
            if not plot:
                return out(d)
            from ..imperfect.plot import curve_png
            p = _png_path(sess, output_path, "nlbuckling_curve")
            curve_png(info, p)
            d["image"] = str(p)
            return [out(d), Image(path=p)]
