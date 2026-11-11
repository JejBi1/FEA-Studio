"""Tools: cyclic symmetry of a sector (fea_core.cyclic, F9 C3) – statics by constraint equations, natural frequencies by nodal
diameters, results expanded to the full 360° model."""
from __future__ import annotations

import math
import time

import numpy as np
from mcp.server.mcpserver import Image

from .common import out, guarded_selection
from .session import FeaError, Session


def _vec(v, what: str) -> list[float]:
    if v is None or len(v) != 3 or not all(math.isfinite(float(x)) for x in v):
        raise FeaError(f"{what} must be a vector [x, y, z]")
    return [float(x) for x in v]


def _settings(st) -> dict:
    c = st.cyclic
    return {"cyclic": None if c is None else c.to_dict()}


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_cyclic_set(left, right, sectors: int | None = None, axis_point: list[float] | None = None,
                       axis_dir: list[float] | None = None, nodal_diameters: list[int] | None = None,
                       expand: bool = True, enabled: bool = True) -> str:
        """Cyclic symmetry: the model is ONE sector, left/right = its two planar cut faces (selection name, face ids or query); right = left rotated by +360/sectors about the axis. Without sectors/axis they are computed from the planes of the faces. Statics: loads must be equal on every sector, reactions are per sector. Natural frequencies (fea_modal_set): per nodal diameter (nodal_diameters, None = 0..N/2). The mesh of the cut faces is made periodic. Results: fea_cyclic_results, fea_cyclic_render_full."""
        from ..cyclic import CyclicSymmetry, CyclicError, detect
        with sess.use():
            st = sess.require_study()
            ln = guarded_selection(sess, left, left if isinstance(left, str) and left in st.selections else "cyclic_left")
            rn = guarded_selection(sess, right, right if isinstance(right, str) and right in st.selections else "cyclic_right")
            if ln == rn:
                raise FeaError("left and right must be two different cut faces")
            notes = []
            try:
                auto = detect(st, ln, rn)
            except CyclicError as e:
                auto = None
                if sectors is None or axis_dir is None:
                    raise FeaError(f"{e} – give sectors, axis_point and axis_dir") from None
            n = int(sectors) if sectors is not None else int(auto["sectors"])
            if n != sectors and sectors is not None or n < 2 or n > 720:
                raise FeaError("sectors must be an integer 2–720")
            if auto is not None and not auto["exact"] and sectors is None:
                notes.append(f"The sector angle {auto['angle_deg']:.4f}° is not 360°/integer – rounded to N = {n}; check the geometry")
            ap = _vec(axis_point, "axis_point") if axis_point is not None else list(auto["axis_point"])
            ad = _vec(axis_dir, "axis_dir") if axis_dir is not None else list(auto["axis_dir"])
            if np.linalg.norm(ad) < 1e-12:
                raise FeaError("axis_dir must not be zero")
            if axis_dir is not None and auto is not None and np.dot(ad, auto["axis_dir"]) < 0:
                ad = [-v for v in ad]
                notes.append("axis_dir reversed so that right = left rotated by +360°/N")
            if nodal_diameters is not None:
                bad = [k for k in nodal_diameters if int(k) != k or not 0 <= int(k) <= n // 2]
                if bad:
                    raise FeaError(f"nodal_diameters must be integers 0–{n // 2}")
            st.cyclic = CyclicSymmetry(sectors=n, left=ln, right=rn, axis_point=ap, axis_dir=ad, enabled=bool(enabled),
                                       nodal_diameters=None if nodal_diameters is None else [int(k) for k in nodal_diameters],
                                       expand=bool(expand))
            sess.changed()
            d = _settings(st)
            d["sector_angle_deg"] = 360.0 / n
            if notes:
                d["notes"] = notes
            d["next"] = "fea_mesh_generate → fea_solve_run → fea_cyclic_results, fea_cyclic_render_full"
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_cyclic_get() -> str:
        """Cyclic symmetry settings of the study and, after solving, the summary (cut node pairs, equations)."""
        with sess.use():
            st = sess.require_study()
            d = _settings(st)
            if sess.results is not None and (sess.results.info or {}).get("cyclic"):
                d["solved"] = sess.results.info["cyclic"]
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_cyclic_remove() -> str:
        """Removes the cyclic symmetry (the model is computed as it is – one sector without the cut-face equations)."""
        with sess.use():
            st = sess.require_study()
            st.cyclic = None
            sess.changed()
            return out({"cyclic": None})

    @mcp.tool(structured_output=False)
    def fea_cyclic_results() -> str:
        """Cyclic results: per load state the per-sector reactions, the force/moment transmitted across the cut faces and the full-structure axial reaction (N x sector); natural frequencies as a table of nodal diameter, frequency [Hz] and multiplicity (2 = degenerate pair of the full structure)."""
        from ..cyclic.modal import table
        with sess.use():
            res = sess.require_results()
            info = (res.info or {}).get("cyclic")
            if not info:
                raise FeaError("The results have no cyclic symmetry (fea_cyclic_set, then fea_solve_run)")
            d = {"cyclic": info, "states": []}
            ax = np.asarray(info["axis_dir"], float)
            for s in res.steps:
                c = (s.equilibrium or {}).get("cyclic", {})
                tot = np.zeros(3)
                for v in s.reactions.values():
                    tot += np.asarray(v, float)
                d["states"].append({"name": s.name, "reactions_per_sector_N": {k: np.asarray(v).tolist()
                                                                                for k, v in s.reactions.items()},
                                    "full_structure_axial_reaction_N": float(info["sectors"] * tot @ ax),
                                    "cut_force_N": c.get("cut_force"), "cut_moment_Nmm": c.get("cut_moment"),
                                    "max_mises_MPa": float(s.mises.max()) if len(s.mises) else None})
            for m in res.modes:
                if getattr(m, "nodal_diameters", None) is not None:
                    d["frequencies"] = table(m)
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_cyclic_render_full(field: str = "mises", step: int | str = 0, mode: int | None = None,
                               output_path: str | None = None, view: str | list[float] = "iso",
                               scale: float | None = None, width: int = 1280, height: int = 720, bands: int = 12,
                               return_image: bool = True):
        """PNG of the result expanded to the full 360° model (sector rotated N times). field as fea_results_render_image; mode = number of a cyclic natural mode (1..) instead of a static state - shape of its nodal diameter around the circumference."""
        from ..cyclic.expand import expand_results
        from .diagnostics import FIELDS, REVERSED, SIGNED, field_key, field_values
        from .render import render
        from .scenes import result_scene
        from .tools_results import _png_path
        if not 2 <= bands <= 24:
            raise FeaError("bands must be 2–24")
        with sess.use(op="render"):
            res = sess.require_results()
            full = expand_results(res)
            if full is None:
                raise FeaError("The results have no cyclic symmetry (fea_cyclic_set, then fea_solve_run)")
            old = sess.results
            sess.results = full
            try:
                if mode is not None:
                    j = next((i for i, m in enumerate(full.modes) if m.kind == "modal"), None)
                    if j is None:
                        raise FeaError("The results have no natural frequencies (fea_modal_set)")
                    ms = full.modes[j]
                    if not 1 <= int(mode) <= len(ms.values):
                        raise FeaError(f"mode must be 1–{len(ms.values)}")
                    k = int(mode) - 1
                    vals = np.linalg.norm(np.asarray(ms.shapes[k], float), axis=1)
                    scene, meta = result_scene(sess, 0, "displacement", vals, "Mode |u|", "–", False, True, scale, bands,
                                               None, None, False, False, mode=(j, k))
                    nds = getattr(ms, "nodal_diameters", None)
                    meta["frequency_Hz"] = float(ms.values[k])
                    if nds is not None:
                        meta["nodal_diameter"] = int(nds[k])
                    stem = f"cyclic_mode_{k + 1}"
                else:
                    key = field_key(field)
                    label, unit = FIELDS[key]
                    names = [s.name for s in full.steps]
                    if isinstance(step, str):
                        if step not in names:
                            raise FeaError(f"State '{step}' does not exist (states: {', '.join(names)})")
                        i = names.index(step)
                    else:
                        i = int(step)
                        if not 0 <= i < len(names):
                            raise FeaError(f"step must be 0–{len(names) - 1}")
                    vals = field_values(sess, i, key)
                    scene, meta = result_scene(sess, i, key, vals, label, unit, key in REVERSED, True, scale, bands,
                                               None, None, False, key in SIGNED)
                    stem = f"cyclic_{key}_{i}"
                p = _png_path(sess, output_path, stem)
                t0 = time.time()
                meta.update(render(scene, p, view, width, height))
                meta["render_time_s"] = time.time() - t0
                meta["sectors"] = int(res.info["cyclic"]["sectors"])
            finally:
                sess.results = old
            txt = out(meta)
            return [txt, Image(path=p)] if return_image else txt
