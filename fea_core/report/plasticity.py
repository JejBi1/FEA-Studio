"""Word report – chapter "Plasticity": material curves (the *PLASTIC points the solver got), max PEEQ and where,
EN 1993-1-5 annex C and ASME VIII-2 5.3.3.1 assessment per step (fea_core.plasticity_eval), residual stress of unloading
steps, PEEQ map and a red warning about the assumptions. Called from word._build (one call).
"""
from __future__ import annotations

from pathlib import Path

from .. import plasticity_eval as PE
from ..plasticity import PlasticitySpec

RED = (192, 0, 0)
MODEL_TEXT = {"bilinear": "bilinear", "multilinear": "multilinear (table)", "ramberg_osgood": "Ramberg-Osgood",
              "hollomon": "Hollomon"}
WARNING = ("WARNING – PLASTICITY: elastoplastic analysis with the von Mises yield condition and the hardening curve of the "
           "material model (true stress – plastic strain, constant stress beyond the last point); no damage, no rupture, "
           "no strain-rate and temperature dependence; the curve parameters are the main source of uncertainty. PEEQ "
           "is a nodal value extrapolated from integration points – at singularities (sharp inner corners, support edges, "
           "point loads) it grows with mesh refinement and must be assessed away from them. The EN 1993-1-5 annex C "
           "limit (5 %) and the ASME VIII-2 5.3.3.1 strain limit (Table 5.7 constants, nodal triaxiality) are checks of the "
           "computed strain only – they do not replace the complete design procedure of the standard (load factors, "
           "ratcheting, shakedown, buckling).")


def has_plasticity(study, res) -> bool:
    return bool(PE.plastic_materials(study)) or (res is not None and any("peeq" in s.fields for s in res.steps))


def _params(spec: PlasticitySpec, m, num) -> str:
    if spec.model == "bilinear" and not spec.table and spec.Et is not None:
        H = m.E * spec.Et / (m.E - spec.Et) if spec.Et < m.E else float("inf")
        return f"Re {num(m.Re)} MPa, Et {num(spec.Et)} MPa (H {num(H)} MPa)"
    if spec.model == "ramberg_osgood":
        return f"Re {num(m.Re)} MPa, K {num(spec.K_prime)} MPa, n {num(spec.n_prime)}"
    if spec.model == "hollomon":
        a = spec.A if spec.A is None or spec.A <= 1 else spec.A / 100
        return f"Re {num(m.Re)} MPa, Rm {num(spec.Rm)} MPa, A {num(100 * a if a else None)} %"
    return f"{len(spec.table)} points"


def _curve_png(mats: list, path: Path) -> bool:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        return False
    fig = Figure(figsize=(7, 3.6), dpi=150, tight_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    drawn = False
    for m in mats:
        try:
            eps, sig = PE.stress_strain(m)
        except ValueError:
            continue
        ax.plot(eps * 100.0, sig, label=m.name)
        drawn = True
    if not drawn:
        return False
    ax.set_xlabel("total strain ε [%]")
    ax.set_ylabel("true stress σ [MPa]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(str(path))
    return True


def section(D, sess, o, img_dir: Path, images: list, warnings: list, label: str, num, render_field) -> None:
    study, res = sess.require_study(), sess.results
    mats = list(PE.plastic_materials(study).values())
    D.p("Elastoplastic material (von Mises yield condition, nonlinear CalculiX analysis with load increments). "
        "The hardening curves below are exactly the *PLASTIC tables given to the solver.")
    D.h(f"{label}.1 Material curves", 2)
    rows = []
    for m in mats:
        s = PlasticitySpec.from_dict(m.plastic)
        try:
            n_pts = len(PE.curve(m))
            end = PE.curve(m)[-1][1]
        except ValueError:
            n_pts, end = 0, None
        rows.append([m.name, MODEL_TEXT.get(s.model, s.model), s.hardening, _params(s, m, num), str(n_pts),
                     num(end, 3) if end is not None else "–", f"{num(100 * s.limit_strain, 3)} %"])
    D.table(["Material", "Model", "Hardening", "Parameters", "Points", "Last ε_pl", "EN limit"], rows,
            caption="Plastic materials")
    if o.images and mats:
        p = Path(img_dir) / "plastic_curves.png"
        if _curve_png(mats, p):
            images.append(str(p))
            D.image(p, "Tensile curves of the material model (true stress – total strain, monotonic loading)")
    D.h(f"{label}.2 Assessment", 2)
    if res is None:
        D.p("Results are not available.", italic=True)
    else:
        try:
            summ = PE.summary(study, res)
        except ValueError as e:
            summ = []
            warnings.append(f"Plasticity assessment: {e}")
        if not summ:
            D.p("No result step contains the plastic strain PEEQ.", italic=True)
        rows = []
        for r in summ:
            en, am = r["en1993"], r["asme"]
            w = en["where_peeq"]
            x, y, z = w["xyz"]
            ok = en["ok"] and (am is None or am["ok"])
            rows.append([r["step"] + (" (partial)" if en["partial"] else ""), num(en["max_peeq"], 4),
                         f"{', '.join(w['parts'])} [{num(x)}; {num(y)}; {num(z)}]", num(en["utilization"], 3),
                         num(am["utilization"], 3) if am else "–", num(am["triaxiality_at_max"], 3) if am else "–",
                         num(en["residual"]["max_mises_MPa"], 4) if en.get("residual") else "–",
                         "PASS" if ok else "FAIL"])
        if rows:
            D.table(["Step", "max PEEQ", "Where (part [x; y; z] mm)", "EN 1993 util.", "ASME util.", "Triaxiality",
                     "Residual σ_vM max [MPa]", "Result"], rows, status_col=7,
                    caption="Plastic strain per step: EN 1993-1-5 annex C (PEEQ ≤ limit) and ASME VIII-2 5.3.3.1 "
                            "(PEEQ ≤ ε_L); utilization ≤ 1 passes")
            mrows = []
            last = summ[-1]
            for name, me in last["en1993"]["materials"].items():
                ma = ((last["asme"] or {}).get("materials") or {}).get(name, {})
                mrows.append([name, num(ma.get("R"), 3), num(ma.get("m2"), 3), num(ma.get("alpha_sl"), 3),
                              num(ma.get("eps_Lu"), 3), ma.get("eps_Lu_source", "–")])
            D.table(["Material", "R = Re/Rm", "m2", "α_sl", "ε_Lu", "ε_Lu from"], mrows,
                    caption="ASME VIII-2 Table 5.7 constants (ε_L = ε_Lu·exp[−α_sl/(1 + m2)·(σ_m/σ_e − 1/3)])")
            for r in summ:
                if r["asme_error"]:
                    D.p(f"ASME check not possible: {r['asme_error']}", italic=True)
                    break
            ex = sorted({mn for r in summ for mn in r["en1993"]["curve_end_exceeded"]})
            if ex:
                D.p("PEEQ exceeds the last point of the hardening curve (perfectly plastic beyond it): " + ", ".join(ex),
                    bold=True, color=RED)
            if o.images:
                i = max(summ, key=lambda r: r["en1993"]["max_peeq"])["index"]
                out = render_field(sess, i, "peeq", Path(img_dir), o.image_view, warnings)
                if out:
                    images.append(str(out[0]))
                    D.image(out[0], f"Equivalent plastic strain PEEQ – step “{res.steps[i].name}” (largest PEEQ)")
    D.p(WARNING, bold=True, size=9, color=RED)
