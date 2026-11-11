"""Tools: submodel (C1, fea_core.submodel_run) – a finer local model of a box / sphere region of the solved 3D study, driven by the
global displacements on the cut faces; loads and supports inside the region are transferred (faces: Fixed, Displacement, Force,
Pressure; body: gravity, rotation, inertia). The results are kept in Results.submodels (saved with the project, discarded with the
results) and rendered by the same rasterizer as the 3D results.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from .common import out, run_with_timeout
from .session import FeaError, Session

NOTE_CUT = ("cut_check compares the submodel von Mises on the cut faces with the global field interpolated there: mean ≤ 10 % and "
            "max ≤ 25 % = the cut is far enough from the stress concentration; otherwise enlarge the region")


def sub_summary(sr, index: int) -> dict:
    """Brief summary of one submodel for the LLM / report."""
    s = sr.summary
    d = {"index": index, "name": sr.name, "step": sr.step_name, "region": s.get("region_text"), "parts": s.get("parts"),
         "nodes": s.get("nodes"), "elements": s.get("elements"), "mesh_size_mm": sr.region.mesh_size,
         "max_mises_MPa": s.get("max_mises_MPa"), "max_mises_at_mm": s.get("max_mises_at"),
         "max_s1_MPa": s.get("max_s1_MPa"), "min_s3_MPa": s.get("min_s3_MPa"),
         "max_displacement_mm": s.get("max_displacement_mm"),
         "global_max_mises_MPa": s.get("global_max_mises_MPa"), "ratio_to_global": s.get("ratio_to_global"),
         "cut_check": s.get("cut_check"), "equilibrium_imbalance": (s.get("equilibrium") or {}).get("imbalance"),
         "transferred": s.get("mapped"), "time_s": s.get("time_s")}
    if sr.warnings:
        d["warnings"] = list(sr.warnings)
    return d


def center_from_hotspots(diag_sess, step: int) -> list[float]:
    """Centre = place of the highest von Mises outside singularities (classification of fea_results_hotspots); all singular → the
    highest place."""
    from .diagnostics import hotspots
    hs = hotspots(diag_sess, step, top_n=8).get("hotspots") or []
    if not hs:
        raise FeaError("No stress maximum found for the automatic submodel centre")
    h = next((h for h in hs if h.get("classification") != "singularity"), hs[0])
    return [float(v) for v in h["xyz_mm"]]


def render_png(sr, study, path, key: str = "mises", view="iso", width: int = 1280, height: int = 720,
               deformed: bool = True, show_mesh: bool = False) -> dict:
    """PNG of the submodel field through the 3D result rasterizer (fea_core.mcp.scenes.result_scene)."""
    from ..submodel_run import sub_field
    from .diagnostics import FIELDS, REVERSED, SIGNED
    from .render import render
    from .scenes import result_scene
    label, unit = FIELDS[key]
    vals = sub_field(sr, key, study)
    shim = SimpleNamespace(results=sr.as_results(), study=SimpleNamespace(name=f"{study.name} – {sr.name}"))
    scene, meta = result_scene(shim, 0, key, vals, label, unit, key in REVERSED, deformed, None, 12, None, None,
                               show_mesh, key in SIGNED)
    scene.title[0] = f"Submodel – {scene.title[0]}"
    meta.update(render(scene, path, view, width, height))
    return meta


def _index(res, index: int) -> int:
    subs = getattr(res, "submodels", None) or []
    if not subs:
        raise FeaError("No submodel computed – use fea_submodel_run")
    k = int(index)
    if not -len(subs) <= k < len(subs):
        raise FeaError(f"Submodel index {index} out of range (0–{len(subs) - 1})")
    return k % len(subs)


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_submodel_run(center: list[float] | str = "max", shape: str = "box", size_mm: list[float] | float | None = None,
                         radius_mm: float | None = None, mesh_size_mm: float | None = None, step: int | str = 0,
                         name: str = "", timeout_s: float = 1800, output_path: str | None = None,
                         return_image: bool = True):
        """Finer local submodel of a box/sphere region of the solved 3D result, driven by the global displacements on the cut. center [x,y,z] mm or 'max' (highest von Mises outside singularities); shape 'box' (size_mm [dx,dy,dz] or one number, default 4× global element size) | 'sphere' (radius_mm, default 2× global size); mesh_size_mm default global/4; step = load case. Supports/loads inside the region are transferred (Fixed, Displacement, Force and Pressure on faces, gravity, rotation, inertia); others inside (bolts, bearing, remote, elastic, edge/point forces, thermal) = error. Returns max stresses (submodel vs global in the region), cut_check, equilibrium, image. Kept in the results (saved in .fea)."""
        from mcp.server.mcpserver import Image
        from ..submodel_run import SubmodelError, SubmodelRegion, run_submodel
        if shape not in ("box", "sphere"):
            raise FeaError("shape must be 'box' or 'sphere'")
        with sess.use(op="submodel", wait=5):
            st = sess.require_study()
            res = sess.require_results()
            k = sess.step_index(step)
            h_g = float(st.mesh.size)
            if isinstance(center, str):
                if center != "max":
                    raise FeaError("center = [x, y, z] mm or 'max'")
                c = center_from_hotspots(sess, k)
            else:
                try:
                    c = [float(v) for v in center]
                except (TypeError, ValueError):
                    raise FeaError("center = [x, y, z] mm or 'max'") from None
            if size_mm is None:
                size = [4.0 * h_g] * 3
            elif isinstance(size_mm, (int, float)):
                size = [float(size_mm)] * 3
            else:
                size = [float(v) for v in size_mm]
            region = SubmodelRegion(center=c, shape=shape, size=size, radius=float(radius_mm or 2.0 * h_g),
                                    mesh_size=float(mesh_size_mm or h_g / 4.0), step=k, name=name)
            try:
                region.check()
            except SubmodelError as e:
                raise FeaError(str(e)) from None
            parts = sess.parts
            wd = (sess.workdir or sess.base_dir) / f"submodel_{time.strftime('%H%M%S')}"
            logs: list[str] = []
            try:
                sr = run_with_timeout(lambda cancel: run_submodel(st, parts, res, region, wd, logs.append, cancel),
                                      timeout_s, "Submodel")
            except FeaError as e:
                if isinstance(e.__cause__, SubmodelError):
                    raise FeaError(str(e.__cause__)) from None
                raise
            finally:
                for line in logs:
                    sess.add_log(line)
            res.submodels.append(sr)
            d = sub_summary(sr, len(res.submodels) - 1)
            d["notes"] = [NOTE_CUT]
            p = None
            if return_image:
                from .tools_results import _png_path
                p = _png_path(sess, output_path, "submodel_mises")
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    render_png(sr, st, p, "mises")
                    d["image_path"] = str(p)
                except Exception as e:  # noqa: BLE001 – the image is a supplement
                    d["image_error"] = str(e)
                    p = None
            d["next"] = "fea_submodel_render(field=…), fea_study_save (keeps the submodel), fea_report_word"
            txt = out(d)
            return [txt, Image(path=p)] if p is not None else txt

    @mcp.tool(structured_output=False)
    def fea_submodel_results(index: int | None = None) -> str:
        """Submodels of the current results: summary per submodel (region, mesh, max stresses submodel vs global, cut check, equilibrium, transferred conditions, warnings). index = one submodel (default all)."""
        with sess.use(op="submodel results"):
            res = sess.require_results()
            subs = getattr(res, "submodels", None) or []
            if index is not None:
                k = _index(res, index)
                return out(sub_summary(subs[k], k))
            return out({"submodels": [sub_summary(s, i) for i, s in enumerate(subs)], "notes": [NOTE_CUT]}
                       if subs else {"submodels": [], "next": "fea_submodel_run"})

    @mcp.tool(structured_output=False)
    def fea_submodel_render(index: int = -1, field: str = "mises", view: str | list[float] = "iso",
                            deformed: bool = True, show_mesh: bool = False, width: int = 1280, height: int = 720,
                            output_path: str | None = None, return_image: bool = True):
        """PNG of a submodel result (index, default the last). field: mises | s1 s2 s3 tresca | sxx … szx | displacement ux uy uz | safety (Re/σ_vM); view as fea_results_render_image."""
        from mcp.server.mcpserver import Image
        from ..submodel_run import FIELD_KEYS
        from .diagnostics import field_key
        key = field_key(field)
        if key not in FIELD_KEYS:
            raise FeaError(f"field for a submodel: {' | '.join(FIELD_KEYS)}")
        with sess.use(op="submodel render"):
            res = sess.require_results()
            k = _index(res, index)
            sr = res.submodels[k]
            from .tools_results import _png_path
            p = _png_path(sess, output_path, f"submodel{k}_{key}")
            p.parent.mkdir(parents=True, exist_ok=True)
            meta = render_png(sr, sess.study, p, key, view, width, height, deformed, show_mesh)
            meta = {"index": k, "submodel": sr.name, "path": str(p),
                    **{a: (b.tolist() if isinstance(b, np.ndarray) else b) for a, b in meta.items()}}
            txt = out(meta)
            return [txt, Image(path=p)] if return_image else txt

    @mcp.tool(structured_output=False)
    def fea_submodel_delete(index: int) -> str:
        """Deletes a submodel from the results (index from fea_submodel_results)."""
        with sess.use(op="submodel delete"):
            res = sess.require_results()
            k = _index(res, index)
            sr = res.submodels.pop(k)
            return out({"deleted": sr.name, "remaining": len(res.submodels)})
