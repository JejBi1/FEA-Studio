"""Word report: nonlinear buckling with imperfection (fea_core.imperfect, F9 C2) – called from word._modes_section."""
from __future__ import annotations

RED = (0xC0, 0x00, 0x00)


def summary_line(sess) -> str:
    """Bullet of the report summary: ultimate factor of the nonlinear buckling."""
    from .word import num
    info = ((sess.results.info or {}).get("nlbuckling") if sess.results is not None else None) or {}
    r = info.get("result") or {}
    return (f"Buckling (nonlinear, {info.get('kind', 'GNIA')}, state “{info.get('case', '–')}”): ultimate load factor "
            f"λ_u = {num(r.get('ultimate_factor'), 4)} – {info.get('status_text', '–')}; imperfection "
            f"e0 = {num((info.get('imperfection') or {}).get('e0_mm'), 3)} mm (amplitude per design code = user's responsibility)"
            + (" – LOSS OF STABILITY ALREADY AT THE APPLIED LOAD" if r.get("ultimate_factor") is not None
               and r["ultimate_factor"] < 1 and r.get("status") == "limit_point" else ""))


def nlbuckling_section(D, sess, j: int, num_label: str, o, img_dir, images, warnings, render_mode) -> None:
    from .word import num
    from ..imperfect.plot import curve_png
    res = sess.results
    ms = res.modes[j]
    info = (res.info or {}).get("nlbuckling") or {}
    r = info.get("result") or {}
    imp = info.get("imperfection") or {}
    D.h(f"{num_label} Buckling (nonlinear, {info.get('kind', 'GNIA')}) – case “{ms.case}”", 2)
    D.p("Geometrically nonlinear analysis (NLGEOM) of the structure with an initial geometric imperfection"
        + (" and material plasticity (GMNIA)" if info.get("kind") == "GMNIA" else " and elastic material (GNIA)")
        + ": the mesh nodes are moved by the imperfection shape, the loads of the case are increased by the load factor λ "
          "(load control with automatic increments) until the limit point or λ max.", size=9)
    kind_txt = {"modes": "linear buckling mode(s) " + ", ".join(
        f"{m}" + ("" if float(w) == 1.0 else f" × {num(float(w), 3)}") for m, w in zip(imp.get("modes", []), imp.get("weights", []))),
        "bow": "half-sine bow", "file": "nodal shape from a file", "none": "none (perfect geometry)"}.get(imp.get("kind"), "–")
    rows = [("Imperfection shape", kind_txt),
            ("Amplitude e0", f"{num(imp.get('e0_mm'), 4)} mm" + (
                f" (L / {num(imp.get('ratio'), 4)}, L = {num(imp.get('length_mm'), 4)} mm)"
                if not imp.get("amplitude_given_mm") and imp.get("kind") != "none" else "")),
            ("Linear buckling factor λ_cr", num(info.get("linear_factor"), 5)),
            ("Maximum load factor of the ramp", num(info.get("max_factor"), 5)),
            ("Ultimate load factor λ_u", num(r.get("ultimate_factor"), 5)),
            ("λ_u / λ_cr", num(r.get("knockdown"), 4)),
            ("Result", info.get("status_text", "–")),
            ("Converged increments", str(r.get("increments", "–"))),
            ("Monitored point", ", ".join(num(v, 5) for v in (info.get("monitor") or {}).get("point", [])) + " mm")]
    if r.get("first_yield_factor") is not None:
        rows.append(("First yield (max von Mises = Re)", f"λ = {num(r['first_yield_factor'], 5)}"))
    D.table(["quantity", "value"], [list(x) for x in rows], caption="Nonlinear buckling – summary", widths=(0.4, 0.6), font=8)
    F = info.get("load_resultant_N") or []
    if F:
        import math
        D.p(f"Load of the case at λ = 1: resultant |F| = {num(math.sqrt(sum(float(x) ** 2 for x in F)))} N; "
            f"ultimate resultant λ_u·|F| = {num((r.get('ultimate_factor') or 0.0) * math.sqrt(sum(float(x) ** 2 for x in F)))} N.",
            size=9)
    if o.images:
        try:
            img_dir.mkdir(parents=True, exist_ok=True)
            p = curve_png(info, img_dir / "nlbuckling_curve.png")
            images.append(str(p))
            D.image(p, "Load–displacement curve of the monitored point")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Nonlinear buckling curve figure: {e}")
        k = len(ms.values) - 1
        if k >= 1:
            p = render_mode(sess, j, k, img_dir, o.image_view, warnings)
            if p:
                images.append(str(p))
                D.image(p, f"Deformed shape at λ = {num(ms.values[k], 4)} (displacement from the imperfect geometry, scaled up)")
    D.p("WARNING – IMPERFECTION: the shape and the amplitude e0 of the imperfection are the user's responsibility and must be "
        "taken from the design code (e.g. EN 1993-1-5 Annex C, EN 1993-1-6 8.7 / 8.8 by the fabrication quality class, "
        "EN 1993-1-1 5.3.2 for members); the ultimate load depends strongly on them.", bold=True, size=9, color=RED)
    D.p("WARNING – LOAD CONTROL: CalculiX has no arc-length (Riks) method; the ultimate factor is the last converged increment "
        "(a lower bound of the limit load) and the post-buckling branch is not traced. Check the curve: a vanishing slope at the "
        "end confirms a limit point.", bold=True, size=9, color=RED)
