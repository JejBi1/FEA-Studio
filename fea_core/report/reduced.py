"""Report – simplified models of the 3D study: 2D calculation by a section (fea_core.twod, Results.twod) and the shell model
(fea_core.shell, Results.shell). Setup (kind of problem, plane of the section / parts and thicknesses), mesh, results by states
(extremes, reactions, equilibrium, pictures), what was left out in 2D, assumptions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .word import num, vec


def _ccx_solver(st) -> str:
    """The linear solver with which CalculiX actually computes (PARDISO without the pardiso module → PaStiX / SPOOLES)."""
    from ..config import solver_name
    return solver_name(st.solver.solver)


def mode(study) -> str | None:
    if study.twod.enabled:
        return "twod"
    if study.shell.enabled:
        return "shell"
    return None


def title_kind(study) -> str | None:
    m = mode(study)
    if m == "twod":
        from ..twod.model import KIND_LABEL
        return f"2D FEM section analysis ({KIND_LABEL.get(study.twod.kind, study.twod.kind)})"
    if m == "shell":
        return "FEM shell analysis"
    return None


def intro(study) -> str | None:
    m = mode(study)
    if m == "twod":
        return ("Linear static analysis on a 2D model: section of the 3D geometry by a plane, quadratic triangles (CAX6 / CPS6 / CPE6, CalculiX). Supports and loads defined on faces of the 3D model are converted to section edges. Units: mm, N, MPa.")
    if m == "shell":
        return ("Linear static analysis of a shell model: thin-walled parts replaced by a mid-surface with thickness (S6 elements, expanded by CalculiX to C3D15 solid elements), T and L plate joints connected by extending the mid-surface by t/2. Stress on the top and bottom surfaces, membrane and bending components. Units: mm, N, MPa.")
    return None


def input_section(D, sess, label: str) -> None:
    st = sess.study
    res = sess.results
    if mode(st) == "twod":
        from ..twod.model import KIND_LABEL, ELEMENT, make_frame
        s = st.twod
        D.h(f"{label} 2D section analysis", 2)
        fr = None
        try:
            fr = make_frame(s)
        except Exception:  # noqa: BLE001
            pass
        rows = [("Analysis type", f"{KIND_LABEL.get(s.kind, s.kind)} (element {ELEMENT.get(s.kind)})")]
        if s.kind == "axisym":
            rows += [("Rotation axis", f"through point {vec(s.origin)} mm, direction {vec(s.axis, 3)}"),
                     ("Section half-plane", f"radial direction {vec(s.xdir, 3)}" + (f" (e_r = {vec(fr.e1, 3)})" if fr else "")),
                     ("Forces and reactions", "for the full circumference (360°)")]
        else:
            rows += [("Section plane", f"through point {vec(s.origin)} mm, normal {vec(s.axis, 3)}, x axis {vec(s.xdir, 3)}"),
                     ("Thickness", f"{num(s.thickness)} mm")]
        rows.append(("Parts in section", ", ".join(s.parts) if s.parts else "all meshed"))
        rows.append(("Element size", f"{num(s.size or st.mesh.size)} mm"))
        D.kv(rows)
        r2 = getattr(res, "twod", None) if res is not None else None
        if r2 is not None:
            secs = r2.info.get("sections") or []
            if secs:
                D.table(["Part", "section area [mm²]", "3D volume [mm³]", "2D model volume [mm³]", "3D / 2D ratio"],
                        [[x["part"], num(x.get("area_mm2")), num(x.get("volume_3d")), num(x.get("volume_2d_model")),
                          num(x.get("ratio_3d_to_2d"), 4)] for x in secs],
                        caption="Part section (volume ratio ≈ 1 = the part is well represented in 2D; for a sector, the fraction of the circumference)", font=8)
            loads = r2.info.get("loads") or []
            if loads:
                D.table(["State", "selection", "type", "conversion to 2D"],
                        [[x.get("case"), x.get("selection"), x.get("kind"),
                          "; ".join(f"{k} = {num(v) if isinstance(v, (int, float)) else v}" for k, v in x.items()
                                    if k not in ("case", "selection", "kind"))] for x in loads],
                        caption="Loads converted to section edges", font=8)
            skipped = [w for w in r2.warnings if "omitted" in w or "not used" in w or "missing" in w]
            for w in skipped:
                D.p(f"Omitted in 2D: {w}", italic=True, size=9)
        return
    s = st.shell
    D.h(f"{label} Shell model", 2)
    rs = getattr(res, "shell", None) if res is not None else None
    rows = []
    if rs is not None:
        sheets = {x["part"]: x for x in rs.info.get("sheets", [])}
        sm = rs.summary()
        for pn, p in sm["parts"].items():
            sh = sheets.get(pn, {})
            rows.append([pn, {"plate": "plate", "sheet": "bent sheet", "surface": "face from STEP"}.get(
                sh.get("kind"), sh.get("kind", "–")), num(p["thickness"]), num(sh.get("mid_area")),
                sh.get("extensions", 0), p["elements"]])
        D.table(["Part", "type", "thickness t [mm]", "mid-surface area [mm²]", "extension (T-joints)", "S6 elements"], rows,
                caption="Shell model parts", font=8)
    else:
        parts = s.parts or {}
        D.kv([("Parts", ", ".join(f"{k} (t = {num(v)} mm)" if v else f"{k} (t from the body)" for k, v in parts.items())
               or "all parts (thickness from the body)"),
              ("Surface position of sheet parts", s.offset), ("Element size", f"{num(s.size or st.mesh.size)} mm")])


def mesh_rows(sess, res) -> list[tuple[str, str]]:
    st = sess.study
    if mode(st) == "twod":
        r2 = getattr(res, "twod", None) if res is not None else None
        from ..twod.model import ELEMENT
        txt = f"2D quadratic triangles {ELEMENT.get(st.twod.kind)}"
        if r2 is not None:
            m = r2.info.get("mesh") or {}
            txt += f": {r2.summary()['nodes']} nodes, {r2.summary()['elements']} elements"
            if m.get("min_scaled_jacobian") is not None:
                txt += f", min. scaled Jacobian {num(m['min_scaled_jacobian'], 3)}"
        return [("Mesh", txt), ("Solver", f"CalculiX 2.23, direct solver {_ccx_solver(st)}"),
                ("Equilibrium check", "loads vs. reactions in the section plane, tolerance 1·10⁻³")]
    rs = getattr(res, "shell", None) if res is not None else None
    txt = "S6 shell elements (tri6), expanded in ccx to C3D15 solid elements"
    if rs is not None:
        txt += f": {rs.summary()['nodes']} nodes, {rs.summary()['elements']} elements"
    return [("Mesh", txt), ("Solver", f"CalculiX 2.23, direct solver {_ccx_solver(st)}"),
            ("Equilibrium check", "forces and moments about the reference point, tolerance 1·10⁻³")]


def summary_rows(res) -> tuple[list[str], list[list]] | None:
    if getattr(res, "twod", None) is not None:
        t = res.twod.summary()
        return (["Load case", "max. displacement [mm]", "max. σ_vM [MPa]", "imbalance [–]"],
                [[s["name"], num(s["max_disp"]), num(s["max_mises"]), num(s["imbalance"], 2)] for s in t["steps"]])
    if getattr(res, "shell", None) is not None:
        t = res.shell.summary()
        return (["Load case", "max. displacement [mm]", "max. σ_vM (top/bottom) [MPa]", "membrane [MPa]",
                 "bending [MPa]", "imbalance [–]"],
                [[s["name"], num(s["u_max"]), num(s["mises_max"]), num(s["membrane_max"]), num(s["bending_max"]),
                  num(s["imbalance"], 2)] for s in t["steps"]])
    return None


def _img(fn, img_dir: Path, name: str, images, warnings) -> Path | None:
    try:
        img_dir.mkdir(parents=True, exist_ok=True)
        p = img_dir / name
        fn(p)
        images.append(str(p))
        return p
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Figure {name}: {e}")
        return None


def results_sections(D, sess, o, img_dir: Path, images, warnings, n: int) -> int:
    """Sections of the results by states (3.n …); returns the last number used."""
    from ..results import von_mises
    res = sess.results
    r2 = getattr(res, "twod", None)
    if r2 is not None:
        from ..mcp.tools_reduced import twod_labels
        lab = twod_labels(r2)
        names = ("σr", "σz (axial)", "σθ", "τrz") if r2.axisym else ("σx", "σy", "σz", "τxy")
        for i, s in enumerate(r2.steps):
            n += 1
            D.h(f"3.{n} Load case “{s['name']}” – 2D analysis", 2)
            sm = r2.summary()["steps"][i]
            fr = r2.frame2d()
            S = s["S"]
            rows = [("Max. displacement", f"{num(sm['max_disp'])} mm at 2D point {vec(sm['max_disp_at'], 5)} mm"),
                    ("Max. σ_vM", f"{num(sm['max_mises'])} MPa at 2D point {vec(sm['max_mises_at'], 5)} mm = 3D {vec(fr.to3d(np.asarray(sm['max_mises_at'])), 5)} mm")]
            for k, nm in enumerate(names):
                rows.append((f"Range {nm}", f"{num(float(S[:, k if k < 3 else 3].min()))} … "
                                             f"{num(float(S[:, k if k < 3 else 3].max()))} MPa"))
            D.kv(rows)
            rr = [[k, num(v[0]), num(v[1])] for k, v in s["reactions"].items()]
            if rr:
                D.table(["Selection", "R1 [N]", "R2 [N]"], rr, caption=f"Reaction – state “{s['name']}“ ({lab['reactions']})",
                        widths=(0.5, 0.25, 0.25), font=8)
            imb = float(s["equilibrium"].get("imbalance", 0.0))
            D.p(f"Equilibrium check: relative imbalance {num(imb, 2)} "
                + ("– OK." if imb <= 1e-3 else "– tolerance EXCEEDED!"), size=9)
            vm = von_mises(S)
            D.table(["Part", "max. σ_vM [MPa]"],
                    [[pn, num(float(vm[np.unique(r2.tri[r2.tri_part == pi])].max()))]
                     for pi, pn in enumerate(r2.part_names) if np.any(r2.tri_part == pi)],
                    caption=f"Stress by part – load case “{s['name']}”", widths=(0.6, 0.4), font=8)
            if o.images:
                for f, cap in (("mises", "von Mises"), ("st" if r2.axisym else "sx",
                                                         "hoop stress σθ" if r2.axisym else "stress σx")):
                    p = _img(lambda pth, f=f: r2.render_png(pth, f, i), img_dir, f"twod_{f}_{i}.png", images, warnings)
                    if p:
                        D.image(p, f"2D calculation – {cap}, state “{s['name']}”"
                                + (" (mirrored about the axis)" if r2.axisym else ""))
        return n
    rs = getattr(res, "shell", None)
    if rs is not None:
        for i, st in enumerate(rs.steps):
            n += 1
            D.h(f"3.{n} Load case “{st.name}” – shell model", 2)
            sm = rs.summary()["steps"][i]
            D.kv([("Max. displacement", f"{num(sm['u_max'])} mm at point {vec(sm['u_max_at'], 5)} mm"),
                  ("Max. σ_vM top / bottom surface", f"{num(sm['mises_top_max'])} / {num(sm['mises_bot_max'])} MPa"),
                  ("Max. σ_vM", f"{num(sm['mises_max'])} MPa at point {vec(sm['mises_max_at'], 5)} mm"),
                  ("Membrane component (mid-surface)", f"max. {num(sm['membrane_max'])} MPa"),
                  ("Bending component ((top − bottom)/2)", f"max. {num(sm['bending_max'])} MPa")])
            from ..shell.results import von_mises as vms
            vt, vb, vmid = vms(st.S_top), vms(st.S_bot), vms(st.S_mid)
            rows = []
            for pi, pn in enumerate(rs.mesh.part_names):
                nd = np.unique(rs.tri[rs.mesh.tri_part == pi])
                if len(nd):
                    rows.append([pn, num(float(rs.mesh.thickness[pi])), num(float(vt[nd].max())), num(float(vb[nd].max())),
                                 num(float(vmid[nd].max())), num(float(vms(st.bending)[nd].max()))])
            D.table(["Part", "t [mm]", "σ_vM top [MPa]", "σ_vM bottom [MPa]", "membrane [MPa]", "bending [MPa]"],
                    rows, caption=f"Stress by layer and part – load case “{st.name}”", font=8)
            rr = [[k, *[num(x) for x in r]] for k, r in st.reactions.items()]
            if rr:
                D.table(["Selection", "Fx [N]", "Fy [N]", "Fz [N]", "Mx [N·mm]", "My [N·mm]", "Mz [N·mm]"], rr,
                        caption=f"Reaction – state “{st.name}“ (moments about point {vec(rs.ref_point, 4)} mm)", font=8)
            imb = float(st.equilibrium.get("imbalance", 0.0))
            D.p(f"Equilibrium check (forces and moments): relative imbalance {num(imb, 2)} "
                + ("– OK." if imb <= 1e-3 else "– tolerance EXCEEDED!"), size=9)
            if o.images:
                for layer, cap in (("top", "top surface"), ("bot", "bottom surface")):
                    p = _img(lambda pth, layer=layer: rs.render_png(pth, "mises", i, layer, deform=True), img_dir,
                             f"shell_{layer}_{i}.png", images, warnings)
                    if p:
                        D.image(p, f"Shells – von Mises, {cap}, load case “{st.name}” (deformation scaled)")
        return n
    return n


def assumptions(study) -> list[str]:
    m = mode(study)
    if m == "twod":
        items = ["2D model: linear elasticity, small deformations; "
                 + ("axisymmetric geometry, material and load (no axis bending, no non-axisymmetric loads)"
                    if study.twod.kind == "axisym" else
                    "plane stress (σz = 0, thin plate)" if study.twod.kind == "plane_stress" else
                    "plane strain (εz = 0, long body)")
                 + "; supports and loads transferred from the faces of the 3D model to the edges of the section – loads that cannot be represented in 2D are reported as errors or warnings."]
        if study.twod.kind == "axisym":
            items.append("Axisymmetric analysis: forces and reactions are for the full circumference (360°); for a sector of a 3D model the result is for the whole ring.")
        return items
    if m == "shell":
        return ["Shell model: linear shell theory (Reissner–Mindlin, S6 elements expanded in CalculiX to C3D15 solid elements), small deformations, constant part thickness; the model does not capture stress normal to the mid-surface or local effects in corners and joints (welds, fillets).",
                "T and L plate joints are connected by extending the mid-surface by t/2 (rigid joint); bolts, welds, heat, 3D printing and orthotropy are not included in the shell model."]
    return []
