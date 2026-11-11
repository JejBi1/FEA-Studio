"""Report – submodels (fea_core.submodel_run, Results.submodels): region, mesh, max. stresses global vs submodel, transferred
conditions, cut check, equilibrium, image."""
from __future__ import annotations

from pathlib import Path

from .word import num, vec

KIND = {"fixed": "fixed support", "displacement": "prescribed displacement", "force": "force (same traction)",
        "pressure": "pressure", "gravity": "gravity (body force)", "rotation": "rotation (body force)",
        "inertia": "inertia (body force)"}


def results_sections(D, sess, o, img_dir: Path, images, warnings, n: int) -> int:
    """One section 3.n per submodel; returns the last number used."""
    res = sess.results
    for k, sr in enumerate(getattr(res, "submodels", None) or []):
        n += 1
        s = sr.summary
        D.h(f"3.{n} Submodel “{sr.name}”", 2)
        D.p("Finer local model of a region cut out of the meshed parts: on the cut faces the displacements of the global solution "
            "are prescribed (interpolated with the tet10 shape functions), supports and loads inside the region are transferred. "
            "Linear static analysis in CalculiX.", size=9)
        chk = s.get("cut_check") or {}
        eq = s.get("equilibrium") or {}
        rows = [("Load case", sr.step_name), ("Region", s.get("region_text", "")),
                ("Parts in the region", ", ".join(s.get("parts") or [])),
                ("Mesh", f"{s.get('nodes')} nodes, {s.get('elements')} C3D10 elements, element size {num(sr.region.mesh_size)} mm"
                 + (f", min. scaled Jacobian {num(s['min_scaled_jacobian'], 3)}" if s.get("min_scaled_jacobian") is not None
                    else "")),
                ("Max. σ_vM submodel", f"{num(s.get('max_mises_MPa'))} MPa at point {vec(s.get('max_mises_at') or (0, 0, 0), 5)} mm"),
                ("Max. σ1 / min. σ3 submodel", f"{num(s.get('max_s1_MPa'))} / {num(s.get('min_s3_MPa'))} MPa"),
                ("Max. displacement submodel", f"{num(s.get('max_displacement_mm'))} mm")]
        if s.get("global_max_mises_MPa") is not None:
            rows.append(("Max. σ_vM global model in the region",
                         f"{num(s['global_max_mises_MPa'])} MPa (element size {num(s.get('global_mesh_size_mm'))} mm)"
                         + (f", submodel / global = {num(s['ratio_to_global'], 4)}" if s.get("ratio_to_global") else "")))
        rows.append(("Cut check", f"mean |Δσ_vM| {num(100 * chk.get('mean_rel_diff', 0), 3)} % (limit {num(100 * chk.get('tol_mean', 0.1), 3)} %), "
                     f"max {num(100 * chk.get('max_rel_diff', 0), 3)} % (limit {num(100 * chk.get('tol_max', 0.25), 3)} %), "
                     f"{chk.get('nodes', 0)} cut nodes – " + ("OK" if chk.get("passed") else "EXCEEDED")))
        rows.append(("Equilibrium check", f"relative imbalance {num(eq.get('imbalance', 0.0), 2)} (loads applied inside the region "
                     f"{vec(eq.get('applied_force_N') or (0, 0, 0), 4)} N, cut reaction {vec(eq.get('cut_reaction_N') or (0, 0, 0), 4)} N)"))
        D.kv(rows)
        mp = s.get("mapped") or []
        if mp:
            D.table(["Selection", "transferred as", "resultant [N]"],
                    [[m.get("selection") or "–", KIND.get(m.get("kind"), m.get("kind")),
                      vec(m["force_N"], 4) if m.get("force_N") else "–"] for m in mp],
                    caption=f"Conditions transferred into the submodel “{sr.name}”", font=8)
        if not chk.get("passed", True):
            D.p("WARNING: the cut check exceeds the limit – the cut lies too close to the stress concentration or the global mesh is "
                "too coarse there; the submodel stress is not reliable. Enlarge the region.", bold=True, size=9,
                color=(0xC0, 0x00, 0x00))
        for w in sr.warnings:
            D.p(f"! {w}", italic=True, size=9)
        if o.images:
            try:
                from ..mcp.tools_submodel import render_png
                img_dir.mkdir(parents=True, exist_ok=True)
                p = img_dir / f"submodel_{k}_mises.png"
                render_png(sr, sess.study, p, "mises")
                images.append(str(p))
                D.image(p, f"Submodel “{sr.name}” – von Mises [MPa], load case “{sr.step_name}” (deformation scaled)")
            except Exception as e:  # noqa: BLE001 – the report is created without the image
                warnings.append(f"Submodel image {k + 1}: {e}")
    return n
