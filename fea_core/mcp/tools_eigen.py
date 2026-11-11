"""Tools: natural frequencies (ccx *FREQUENCY) and linear buckling (ccx *BUCKLE) – fea_core.eigen.

The settings are in Study.modal / Study.buckling (they do not change the mesh – the mesh signature does not contain them); the calculation runs in fea_solve_run on the same mesh as the statics (separate ccx jobs). A study with natural frequencies only does not need load states (without constraints = free body, the first 6 frequencies ≈ 0 Hz). Results Results.modes (eigen.ModeSet).
"""
from __future__ import annotations

import math
import time

from mcp.server.mcpserver import Image

from .common import out
from .diagnostics import modes_summary
from .session import FeaError, Session


def mode_set_index(sess: Session, kind: str | None) -> int:
    """Index of the set Results.modes by kind (modal | buckling; None = the first)."""
    res = sess.require_results()
    modes = getattr(res, "modes", None) or []
    if not modes:
        raise FeaError("The results contain neither natural frequencies nor buckling – enable fea_modal_set / fea_buckling_set and run fea_solve_run")
    if kind is None:
        return 0
    k = {"modal": "modal", "frequency": "modal", "frequency": "modal", "buckling": "buckling", "buckling": "buckling",
         "buckling": "buckling", "frekvence": "modal", "vzper": "buckling", "vzpěr": "buckling"}.get(str(kind).strip().lower())
    if k is None:
        raise FeaError("kind must be 'modal' (natural frequencies) or 'buckling'")
    for j, m in enumerate(modes):
        if m.kind == k:
            return j
    raise FeaError(f"The results do not contain {('natural frequencies' if k == 'modal' else 'buckling')} (they have: {', '.join((m.name for m in modes))})")


def _settings(st) -> dict:
    return {"modal": st.modal.to_dict(), "buckling": st.buckling.to_dict()}


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_modal_set(enabled: bool = True, modes: int = 10, f_min_Hz: float = 0.0,
                      f_max_Hz: float | None = None) -> str:
        """Natural frequencies/mode shapes (CalculiX *FREQUENCY, computed in fea_solve_run on the same mesh). modes, f_min_Hz/f_max_Hz range. Mass from rho [t/mm3]; stiffness from the study's constraints (none = free body, first 6 frequencies ~ 0 Hz). Loads do not change frequencies. A frequency-only study needs no load state. Results: fea_results_modes, fea_results_render_mode. enabled=False turns off."""
        if int(modes) != modes or not 1 <= int(modes) <= 200:
            raise FeaError("modes must be an integer 1–200")
        if not (math.isfinite(f_min_Hz) and f_min_Hz >= 0):
            raise FeaError("f_min_Hz must be ≥ 0")
        if f_max_Hz is not None and not (math.isfinite(f_max_Hz) and f_max_Hz > f_min_Hz):
            raise FeaError("f_max_Hz must be greater than f_min_Hz (or None = no limit)")
        with sess.use():
            st = sess.require_study()
            st.modal.enabled = bool(enabled)
            st.modal.modes = int(modes)
            st.modal.f_min = float(f_min_Hz)
            st.modal.f_max = float(f_max_Hz) if f_max_Hz is not None else None
            sess.changed()                       # the mesh stays (the mesh signature does not contain the frequency settings)
            d = _settings(st)
            notes = []
            if enabled:
                no_rho: list[str] = []
                for p in sess.part_names():
                    if p in st.excluded_parts():
                        continue
                    try:
                        m = st.material_of(p)
                    except KeyError:
                        continue
                    if not m.rho and m.name not in no_rho:
                        no_rho.append(m.name)
                if no_rho:
                    notes.append(f"Materials without density rho (natural frequencies cannot be computed): {', '.join(no_rho)}")
                if not st.constraints and not (st.frame.active and st.frame.supports):
                    notes.append("The study has no supports – a free body: the first 6 frequencies ≈ 0 Hz (rigid-body motions)")
                d["next"] = "fea_solve_run → fea_results_modes, fea_results_render_mode(mode=1)"
            if notes:
                d["notes"] = notes
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_buckling_set(enabled: bool = True, case_name: str | None = None, modes: int = 4) -> str:
        """Linear buckling (CalculiX *BUCKLE) for the load of state case_name (None = first; may be thermal): critical load = lambda x state load. Computed in fea_solve_run. modes = number of shapes. No imperfections or plasticity - real capacity is lower; design with EN 1993-1-1 6.3 (fea_ec3_member_check) or nonlinear analysis. enabled=False turns off."""
        if int(modes) != modes or not 1 <= int(modes) <= 50:
            raise FeaError("modes must be an integer 1–50")
        with sess.use():
            st = sess.require_study()
            names = [lc.name for lc in st.load_cases]
            if enabled:
                if not names:
                    raise FeaError("Buckling needs a load case with a load (fea_loadcase_create + fea_load_add_*)")
                if case_name is not None and case_name not in names:
                    raise FeaError(f"Load case '{case_name}' does not exist (cases: {', '.join(names)})")
            st.buckling.enabled = bool(enabled)
            st.buckling.case = case_name
            st.buckling.modes = int(modes)
            sess.changed()
            d = _settings(st)
            if enabled:
                d["case_used"] = case_name or names[0]
                d["warning"] = ("Linear buckling (no imperfections, no plasticity) – the real capacity is usually lower than λ × load")
                d["next"] = "fea_solve_run → fea_results_modes(kind='buckling'), fea_results_render_mode(kind='buckling')"
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_results_modes(kind: str | None = None) -> str:
        """Frequency/buckling results; kind 'modal' | 'buckling' | None. Modal: frequency [Hz], period, effective mass share x/y/z and cumulative (>= 90 % = enough modes), participation factors; ~0 Hz = rigid motions. Buckling: factors lambda, state load resultant, critical force lambda*|F|, always with a linear-buckling warning."""
        with sess.use():
            res = sess.require_results()
            if not getattr(res, "modes", None):
                mode_set_index(sess, None)                 # understandable error
            idx = range(len(res.modes)) if kind is None else [mode_set_index(sess, kind)]
            return out({"mode_sets": [modes_summary(sess, j) for j in idx]})

    @mcp.tool(structured_output=False)
    def fea_results_render_mode(mode: int = 1, kind: str = "modal", field: str = "displacement",
                                output_path: str | None = None, view: str | list[float] = "iso",
                                scale: float | None = None, width: int = 1280, height: int = 720,
                                show_mesh: bool = False, bands: int = 12, return_image: bool = True):
        """PNG of a mode shape: kind 'modal' | 'buckling', mode from 1. Shape normalized to max|u| = 1; field displacement | ux | uy | uz; scale in mm per unit (None = auto 8 % of size); view/width/height/bands/show_mesh as fea_results_render_image. Title shows frequency or lambda."""
        from .render import render
        from .scenes import result_scene
        from .tools_results import _png_path
        import numpy as np
        if not 2 <= bands <= 24:
            raise FeaError("bands must be 2–24")
        key = {"displacement": "displacement", "u": "displacement", "umag": "displacement", "ux": "ux", "uy": "uy",
               "uz": "uz"}.get(str(field).strip().lower())
        if key is None:
            raise FeaError("mode shape field: displacement | ux | uy | uz")
        with sess.use(op="render"):
            res = sess.require_results()
            j = mode_set_index(sess, kind)
            ms = res.modes[j]
            if not 1 <= int(mode) <= len(ms.values):
                raise FeaError(f"mode must be 1–{len(ms.values)} ({ms.name})")
            k = int(mode) - 1
            if not len(res.mesh.nodes) or sess.study.frame.members:      # beam system (also a mixed model)
                from .frame_scenes import mode_frame_scene
                scene, meta = mode_frame_scene(sess, j, k, scale, bands)
                p = _png_path(sess, output_path, f"mode_{ms.kind}_{k + 1}")
                meta.update(render(scene, p, view, width, height))
                if ms.kind == "buckling":
                    meta["warning"] = "Linear buckling – without imperfections and plasticity (the real capacity is usually lower)"
                if key != "displacement":
                    meta["note"] = "For members |u| is drawn (field ux/uy/uz only for solids)"
                txt = out(meta)
                return [txt, Image(path=p)] if return_image else txt
            U = np.asarray(ms.shapes[k], float)
            vals = np.linalg.norm(U, axis=1) if key == "displacement" else U[:, "xyz".index(key[1])]
            label = {"displacement": "Mode |u|", "ux": "Mode ux", "uy": "Mode uy", "uz": "Mode uz"}[key]
            scene, meta = result_scene(sess, 0, key, vals, label, "–", False, True, scale, bands, None, None,
                                       show_mesh, key != "displacement", mode=(j, k))
            p = _png_path(sess, output_path, f"mode_{ms.kind}_{k + 1}")
            t0 = time.time()
            meta.update(render(scene, p, view, width, height))
            meta["render_time_s"] = time.time() - t0
            if ms.kind == "buckling":
                meta["warning"] = "Linear buckling – without imperfections and plasticity (the real capacity is usually lower)"
            txt = out(meta)
            return [txt, Image(path=p)] if return_image else txt
