"""Report – frame system (Study.frame, fea_core.beams): setup (nodes, members, sections, supports, connections,
loads), results by states (extremes of the internal forces by members, reactions, pictures of the diagram My and of the deformation)
and the assessment according to EN 1993-1-1 (utilisation by members, governing row, detailed rows of the most utilised member,
coefficients γ, Winkler–Bach for curved bars). Works also for a study without geometry (only bars).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .word import num, vec

LOCAL = ("N", "Vy", "Vz", "T", "My", "Mz")
DOFS = ("ux", "uy", "uz", "rx", "ry", "rz")


def frame_only(study) -> bool:
    return study.frame.active and not str(study.geometry or "").strip()


def _lengths(study) -> dict:
    from ..mcp.frame_scenes import member_lengths
    try:
        return member_lengths(study)
    except Exception:  # noqa: BLE001
        return {}


def load_text(l) -> tuple[str, str, str]:
    """(kind, selection/node/member, value) of a load of a bar – for the tables of the load states."""
    if l.kind == "frame_node":
        val = f"F = {vec(l.force)} N" + (f"; M = {vec(l.moment)} N·mm" if any(l.moment) else "")
        return "point load on member node", f"node {l.node}", val
    if l.kind == "frame_temperature":
        val = f"ΔT = {num(l.dT)} K" + (f"; gradient in the z direction {num(l.grad_z)} K" if l.grad_z else "")             + (f"; gradient in the y direction {num(l.grad_y)} K" if l.grad_y else "")
        return "member temperature load", ("member " + ", ".join(l.members)) if l.members else "all members", val
    val = f"q = {vec(l.q)} N/mm" + ("" if l.q_end is None else f" → {vec(l.q_end)} N/mm (linear)")
    return ("distributed load of the member (" + ("member local axes" if l.axes == "local" else "global axes") + ")",
            f"member {l.member}", val)


def input_section(D, sess, label: str) -> None:
    """1.x Frame system: nodes, members, sections, supports, connections with bodies."""
    from ..mcp.frame_scenes import dofs_label
    st = sess.study
    fr = st.frame
    D.h(f"{label} Frame", 2)
    from ..beams.fem import solver_of
    if solver_of(st) == "timoshenko":
        D.p("Members: own solver with the exact Timoshenko element (float64"
            + ("" if fr.settings.shear_deformation else ", shear deformation off – Euler–Bernoulli") + f"), statics {fr.settings.segments_static} elements per straight member (exact), natural frequencies and buckling {fr.settings.segments_modal} elements, arc by ≤ {num(min(fr.settings.max_arc_angle, 2.0))}°; member axes x = from the start node to the end node, z = “up” in the section (web of the I profile), y = z × x (EN 1993). Internal forces N (+ tension), Vy, Vz, T, My, Mz in the local axes.", size=9)
    else:
        D.p("Members: beam element U1 (CalculiX, Timoshenko"
            + ("" if fr.settings.shear_deformation else " off – Euler–Bernoulli") + f"), {fr.settings.segments_straight} elements per straight member, arc by ≤ {num(fr.settings.max_arc_angle)}°; member axes x = from the start node to the end node, z = “up” in the section (web of the I profile), y = z × x (EN 1993). Internal forces N (+ tension), Vy, Vz, T, My, Mz in the local axes.", size=9)
    rows = [[k, *[num(v, 6) for v in n.xyz]] for k, n in fr.nodes.items()]
    D.table(["Node", "x [mm]", "y [mm]", "z [mm]"], rows, caption="Frame nodes", widths=(0.4, 0.2, 0.2, 0.2),
            font=8)
    Ls = _lengths(st)
    rows = []
    for m in fr.members.values():
        rel = []
        for end, r in (("start", m.releases_start), ("end", m.releases_end)):
            on = [n for n, x in zip(LOCAL, r) if x]
            if on:
                rel.append(f"{end}: {', '.join(on)}")
        typ = "truss (N only)" if m.kind == "truss" else ("arc" if m.through is not None else "beam")
        rows.append([m.name, f"{m.start} → {m.end}", m.section, m.material or st.default_material or "–",
                     num(Ls.get(m.name)), typ, "; ".join(rel) or "–"])
    D.table(["Member", "nodes", "cross-section", "material", "length [mm]", "type", "releases (hinges)"], rows,
            caption="Members", widths=(0.16, 0.14, 0.14, 0.14, 0.1, 0.13, 0.19), font=8)
    rows, srcs = [], []
    for k, sd in fr.sections.items():
        try:
            p = sd.resolve()
        except Exception as e:  # noqa: BLE001
            rows.append([k, "–", "–", "–", "–", "–", f"error: {e}"])
            continue
        rows.append([k, num(p.A), num(p.Iy), num(p.Iz), num(p.J), num(p.Wel_y), num(p.Wel_z)])
        if p.source:
            srcs.append(f"{k}: {p.source}")
    D.table(["Cross-section", "A [mm²]", "Iy [mm⁴]", "Iz [mm⁴]", "It [mm⁴]", "Wel,y [mm³]", "Wel,z [mm³]"], rows,
            caption="Member cross-sections (principal axes: y strong, z weak)", font=8)
    for t in srcs:
        D.p(t, size=8)
    rows = [[s.node, dofs_label(s.dofs, s.stiffness), ", ".join(n for n, on in zip(DOFS, s.dofs) if on) or "–",
             ", ".join(f"{n} = {num(k)}" for n, k in zip(DOFS, s.stiffness) if k) or "–"] for s in fr.supports]
    if rows:
        D.table(["Node", "supports", "constrained components (global)", "springs [N/mm, N·mm/rad]"], rows,
                caption="Frame supports", widths=(0.18, 0.2, 0.32, 0.3), font=8)
    if fr.connections:
        D.table(["Member node", "body faces (selection)", "constraint"],
                [[c.node, c.selection, "rigid (*COUPLING/*KINEMATIC)"] for c in fr.connections],
                caption="Member-to-body connections", font=8)


def mesh_rows(sess, res) -> list[tuple[str, str]]:
    info = (res.info.get("frame") if res is not None else None) or {}
    st = sess.study
    n_el = sum(len(v) for v in (info.get("segments") or {}).values())
    from ..beams.fem import solver_of
    tim = solver_of(st) == "timoshenko"
    txt = f"number of members {len(st.frame.members)}, " + ("built-in solver, exact Timoshenko element (float64)" if tim else
                                                     "U1 beam element (Timoshenko, CalculiX)")
    if n_el:
        txt += f", {n_el} elements, {len(info.get('xyz') or [])} nodes"
    if tim:
        return [("Members", txt), ("Member internal forces", "from element end forces K_e·u_e − f_eq (exact element matrix), diagrams and displacements along the member by integrating the forces (independent of the mesh), node equilibrium check")]
    return [("Members", txt), ("Member internal forces", "from element end forces K_U1·u − f_eq (actual element matrix from ccx), nodal equilibrium check")]


def _render(sess, i, quantity, img_dir: Path, view, warnings, util=None) -> Path | None:
    from ..mcp.frame_scenes import result_frame_scene
    from .images import render
    try:
        scene, _ = result_frame_scene(sess, i, quantity, None, 12, util)
        img_dir.mkdir(parents=True, exist_ok=True)
        p = img_dir / f"frame_{quantity}_{i}.png"
        render(scene, p, view, 1600, 900)
        return p
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Member plot ({quantity}, load case {i}): {e}")
        return None


def step_section(D, sess, i: int, label: str | None, o, img_dir, images, warnings, s: dict | None = None) -> None:
    """Results of the frame system in the state i (for a study without bodies the whole section of the state, for a mixed one a subsection)."""
    from ..mcp.diagnostics import frame_step_summary
    res = sess.results
    st = res.steps[i]
    if not st.frame:
        return
    fs = frame_step_summary(st)
    if label:
        D.h(f"{label} Load case “{st.name}” – frame", 2)
    else:
        D.p(f"Frame – load case “{st.name}”", bold=True)
    rows = []
    vmax = {g: max([abs(v.get(q, {}).get("value", 0.0)) for v in fs["members"].values() for q in qs] + [1e-30])
            for g, qs in (("F", ("N", "Vy", "Vz")), ("M", ("T", "My", "Mz")))}
    for m, v in fs["members"].items():
        def ex(q):
            d = v.get(q)
            if not d:
                return "–"
            x = d["value"]
            return num(0.0 if abs(x) < 1e-5 * vmax["F" if q in ("N", "Vy", "Vz") else "M"] else x + 0.0)
        rows.append([m, ex("N"), ex("Vy"), ex("Vz"), ex("T"), ex("My"), ex("Mz"), num(v.get("sigma_eq_max_MPa")),
                     num(v.get("u_max_mm"))])
    D.table(["Member", "N [N]", "Vy [N]", "Vz [N]", "T [N·mm]", "My [N·mm]", "Mz [N·mm]", "σ_eq [MPa]", "u max [mm]"],
            rows, caption=f"Internal force extremes per member (value with the largest absolute magnitude) – load case “{st.name}”",
            font=7)
    D.p("σ_eq = √(σ² + 3τ²): σ from N, My, Mz in the extreme fibers of the cross-section, τ estimated from V/Av and T/Wt (the check is governed by EN 1993-1-1, see chapter Member check).", size=8, italic=True)
    rr = []
    tf = 1e-5 * max([float(np.abs(np.asarray(r[:3], float)).max()) for r in fs["reactions"].values()] + [1e-30])
    tm = 1e-5 * max([float(np.abs(np.asarray(r[3:], float)).max()) for r in fs["reactions"].values()] + [1e-30])
    for k, r in fs["reactions"].items():
        r = [0.0 if abs(x) < tf else x + 0.0 for x in r[:3]] + [0.0 if abs(x) < tm else x + 0.0 for x in r[3:]]
        rr.append([k, *[num(x) for x in r]])
    if rr:
        D.table(["Node", "Rx [N]", "Ry [N]", "Rz [N]", "Mx [N·mm]", "My [N·mm]", "Mz [N·mm]"], rr,
                caption=f"Member support reactions – load case “{st.name}” (global axes)", font=8)
    cf = st.frame.get("coupling_forces") or {}
    if cf:
        D.table(["Node", "Fx [N]", "Fy [N]", "Fz [N]", "Mx [N·mm]", "My [N·mm]", "Mz [N·mm]"],
                [[k, *[num(x) for x in r]] for k, r in cf.items()],
                caption=f"Forces in member-to-body connections – load case “{st.name}”", font=8)
    eq = st.equilibrium or {}
    if eq and label:
        D.p(f"Equilibrium check: load {vec(eq.get('applied_exact') or (0, 0, 0))} N, sum of reactions {vec(eq.get('reaction_total') or (0, 0, 0))} N, relative imbalance {num(eq.get('imbalance'), 2)} "
            + ("– OK." if (eq.get("imbalance") or 0) <= 1e-3 else "– tolerance EXCEEDED!"), size=9)
    D.p(f"Max. displacement of member nodes {num(fs.get('max_node_displacement_mm'))} mm.", size=9)
    if o.images:
        for q, cap in (("My", "Bending moment My diagram"), ("deformed", "Deformed shape of members (color |u|)")):
            p = _render(sess, i, q, img_dir, o.image_view, warnings)
            if p:
                images.append(str(p))
                D.image(p, f"{cap} – state “{st.name}“")
        mz = max((v.get("Mz", {}).get("max_abs", 0.0) for v in fs["members"].values()), default=0.0)
        my = max((v.get("My", {}).get("max_abs", 0.0) for v in fs["members"].values()), default=0.0)
        if mz > 0.2 * max(my, 1e-30):
            p = _render(sess, i, "Mz", img_dir, o.image_view, warnings)
            if p:
                images.append(str(p))
                D.image(p, f"Bending moment Mz diagram – load case “{st.name}”")


def summary_rows(summ: dict) -> list[list]:
    rows = []
    for s in summ["steps"]:
        fr = s.get("frame")
        if not fr:
            continue
        eq = s.get("equilibrium") or {}
        sm = fr.get("max_sigma_eq_MPa") or {}
        rows.append([s["step"], num(fr.get("max_node_displacement_mm")), num(sm.get("value")), sm.get("member", "–"),
                     num((fr.get("max_abs_moment_Nmm") or {}).get("value")),
                     num(eq.get("imbalance_relative"), 2) if eq else "–"])
    return rows


def assess_section(D, sess, o, img_dir, images, warnings) -> None:
    """Assessment of the bars EN 1993-1-1 (beams.assess): overview of the utilisation, detailed rows of the most utilised bar."""
    import math
    from ..beams.assess import assess
    st, res = sess.study, sess.results
    if sess.frame_check is None:
        try:
            sess.frame_check = assess(st, res)
        except Exception as e:  # noqa: BLE001
            D.p(f"Member check failed: {e}", italic=True)
            warnings.append(f"Member check: {e}")
            return
    a = sess.frame_check
    s = st.frame.settings
    D.p(f"Check of steel members per ČSN EN 1993-1-1 (Czech National Annex): cross-section classification (Table 5.2), cross-section resistance at all points along the length (6.2), buckling (6.3.1), lateral-torsional buckling (6.3.2), interaction (6.3.3, Annex B). The internal forces of the load cases are taken as DESIGN values (load factors γF and combinations are specified by the user); γM0 = {num(s.gamma_M0, 3)}, γM1 = {num(s.gamma_M1, 3)}. Steel grade by material name, otherwise {s.steel_grade} (fy by thickness, Table 3.1). Buckling lengths and lateral-torsional buckling as specified (default: member length).")
    rows = []
    for k, v in sorted(((k, v) for k, v in a.items() if k != "_max"), key=lambda kv: -(kv[1].get("utilization") or 0)):
        u = v.get("utilization")
        rows.append([k, v.get("section", "–"), num(v.get("length_mm")), v.get("grade", "–"), v.get("section_class", "–"),
                     num(u, 3), "NOT ASSESSED" if u is None else ("PASS" if u <= 1 else "FAIL"),
                     v.get("governing", "–"), v.get("case", "–")])
    D.table(["Member", "cross-section", "length [mm]", "steel", "grade", "utilization [–]", "rating", "governing", "state"], rows,
            caption="Member check EN 1993-1-1 – utilization overview", status_col=6, font=7,
            widths=(0.13, 0.11, 0.08, 0.07, 0.05, 0.08, 0.12, 0.26, 0.1))
    checks = st.frame.checks
    if checks:
        D.table(["Member", "check inputs"],
                [[k, ", ".join(f"{p} = {num(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x}"
                               for p, x in c.items())] for k, c in checks.items()],
                caption="Buckling lengths, lateral-torsional buckling and other check parameters", widths=(0.25, 0.75), font=8)
    umax, mmax = a.get("_max", (0.0, None))
    if mmax is not None and a[mmax].get("rows"):
        v = a[mmax]
        D.p(f"Most utilised member “{mmax}” ({v.get('section')}, state “{v.get('case')}”): utilisation {num(umax, 3)} – "
            + ("PASS" if umax <= 1 else "FAIL"), bold=True)

        def val(x, unit):
            if unit == "N":
                return f"{num(x / 1e3)} kN"
            if unit == "N·mm":
                return f"{num(x / 1e6)} kN·m"
            return f"{num(x)} {unit}".strip()
        rr = [[r["name"], r["clause"], val(r["effect"], r["unit"]), val(r["resistance"], r["unit"]),
               num(r["utilization"], 3) if math.isfinite(r["utilization"]) else "–",
               num(r["x"]) if r.get("x") is not None else "–", r.get("note") or ""] for r in v["rows"]]
        D.table(["Assessment", "clause", "action effect", "resistance", "utilization", "x [mm]", "note"], rr,
                caption=f"Checks of member “{mmax}” (rows marked “(informative)” are not included in the utilization)", font=7,
                widths=(0.24, 0.09, 0.12, 0.12, 0.08, 0.07, 0.28))
    curved = [(k, v["curved"]) for k, v in a.items() if k != "_max" and v.get("curved")]
    for k, c in curved:
        D.p(f"Curved member “{k}” (R/h = {num(c['R_over_h'], 3)}): the bending stress in the plane of curvature differs from the straight-member value by K_i = {num(c['K_i'], 3)}× at the inner fiber (K_o = {num(c['K_o'], 3)}× at the outer fiber) (Winkler–Bach). EN 1993-1-1 does not cover curved members – the check is INFORMATIVE; verify the inner-fiber stress separately.", bold=True, size=9, color=(0xC0, 0x00, 0x00))
    for k, v in a.items():
        if k != "_max":
            for w in v.get("warnings", []):
                if not (v.get("curved") and w.startswith("Curved member")):
                    D.p(f"{k}: {w}", italic=True, size=8)
    if o.images and res.steps:
        p = _render(sess, 0, "utilization", img_dir, o.image_view, warnings, a)
        if p:
            images.append(str(p))
            D.image(p, "Member utilization per EN 1993-1-1 (maximum over all load cases)")


def assumptions(study) -> list[str]:
    from ..beams.fem import solver_of
    tim = solver_of(study) == "timoshenko"
    items = ["Members: linear beam theory (Timoshenko, " + ("exact element of the built-in solver" if tim else "U1 element")
             + "), small deformations, shear centre = centroid, no restrained torsion (warping) in FEM, member connections at nodes rigid unless a release (hinge) is given; lateral-torsional buckling and buckling assessed by EN 1993-1-1."]
    if any(m.through is not None for m in study.frame.members.values()):
        ang = min(study.frame.settings.max_arc_angle, 2.0) if tim else study.frame.settings.max_arc_angle
        items.append(f"Curved members: the arc is replaced by straight elements (≤ {num(ang)}°); for a small R/h ratio the Winkler–Bach stress distribution applies (informative in the check).")
    if study.buckling.enabled and frame_only(study):
        if tim:
            items.append("Buckling of the beam structure: linear, geometric stiffness from the axial forces of the load case (Engesser shear correction); no warping, lateral-torsional buckling or effect of the bending moments of the state – α_cr for EN 1993-1-1, 5.2.1.")
        else:
            items.append("Frame buckling is computed by MYSTRAN (SOL 105, CBAR) – factor α_cr for EN 1993-1-1, 5.2.1; it does not include spatial buckling with warping or lateral-torsional buckling.")
    return items
