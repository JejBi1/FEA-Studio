"""Report – F8: contacts and nonlinear calculation (definition of the pairs, μ with its source and purpose, K, λ, interference, initial state,
formulation, sequence of steps; results by steps, maps of the pressure and of the state of the contact, force–displacement curves, partial results)
and structural dynamics (modes with m_eff, checks of the basis, damping, excitation and profiles, FRF, envelopes, settling time, RMS and 3σ,
spectral fatigue, shock and SRS, ccx etalon, added mass). Mandatory red warnings as for the creep of printing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..i18n import tr as _tr
from .word import num, vec

RED = (0xC0, 0x00, 0x00)
MAX_IMAGE_STEPS = 4


# ===================================================================== co protokol obsahuje

def has_contacts(study, res) -> bool:
    if study.contacts or study.analysis.nonlinear or study.contact_settings.nlgeom:
        return True
    return res is not None and any(s.contacts or s.partial for s in res.steps)


def has_dynamics(study, res) -> bool:
    return bool(study.dynamics.enabled or (res is not None and getattr(res, "dynamics", None) is not None))


def title_kind(study, therm: bool) -> str | None:
    nl = study.is_nonlinear()
    dyn = study.dynamics.enabled
    if nl and study.load_cases:
        return ("FEM nonlinear strength analysis (contact) and dynamics" if dyn else
                "FEM nonlinear strength analysis (contact)")
    if dyn and not study.load_cases:
        return "FEM dynamic analysis"
    if dyn:
        return "FEM strength and dynamic analysis" if not therm else "FEM strength, thermal and dynamic analysis"
    return None


def intro(study) -> list[str]:
    out = []
    if study.is_nonlinear() and study.load_cases:
        an = study.analysis
        out.append("Nonlinear static FEM analysis with contact (CalculiX): linear elastic material, "
                   + ("large deformations (NLGEOM)" if study.contact_settings.nlgeom else "small deformations")
                   + "; parts in contact pairs touch through contact surfaces (penalty contact, Coulomb "
                   "friction), the other parts are rigidly connected by the shared mesh. The load is applied in increments – "
                   + ("a step sequence with history (contact and friction state is carried from step to step)."
                      if an.nonlinear and not an.branches and an.sequence else
                      "each load case is a separate nonlinear analysis (branch).")
                   + " Units: mm, N, MPa, t, s.")
    if study.dynamics.enabled:
        out.append("Structural dynamics by modal superposition: mode shapes and modal stresses from CalculiX (*FREQUENCY, the same mesh and supports as the static analysis), modal response (harmonic, transient, random vibration, shock) by exact solution of the modal equations in the program; residual contribution of higher modes by static correction. Linear model (no contact), specified damping. Units: mm, N, MPa, t, s, Hz; acceleration in g (1 g = 9,810 mm/s²).")
    return out


def assumptions(study, res) -> list[str]:
    items = []
    if study.active_contacts() or study.analysis.nonlinear:
        items += [
            "Nonlinear analysis (contact): load cases cannot be superposed, the safety factor is a stress margin, not a load multiplier; fatigue by superposition of load cases is not available.",
            "Penalty contact: the normal stiffness K and the stick slope λ allow a small penetration p/K and elastic slip τ/λ; Coulomb friction with the specified (estimated) coefficient μ, independent of velocity and temperature.",
        ]
        if study.thermal.cases:
            from .thermal_extras import has_conductance
            items.append("In the thermal analysis, contact pairs with a contact conductance h_c have a temperature jump ΔT = q / h_c at the interface; the other contact pairs are joined by perfect thermal contact."
                         if has_conductance(study) else
                         "In the thermal analysis, parts in contact pairs are joined by perfect thermal contact (no contact conductance specified).")
    if study.dynamics.enabled:
        items += [
            "Dynamics: linear model, modal superposition with a truncated basis (higher modes statically / by missing mass), base excitation moves all supported nodes equally (rigid fixture).",
            "Damping is specified (estimate or measurement) – an uncertainty of ±50 % changes the resonance peaks proportionally to 1/ζ.",
        ]
    if getattr(study, "masses", None):
        items.append("Added masses (point / distributed) replace parts that are not modelled; a point mass with a rigid coupling stiffens the connected face.")
    return items


def summary_bullets(D, sess) -> None:
    res = sess.results
    if res is None:
        return
    st = sess.study
    part = [s.name for s in res.steps if s.partial]
    for s in res.steps:
        if s.contacts:
            tot = sum(float(c.get("Fn_N") or 0) for c in s.contacts.values())
            pm = max(float(c.get("p_max_MPa") or 0) for c in s.contacts.values())
            D.bullet(f"Contact – state “{s.name}“: Σ Fn = {num(tot)} N, max. pressure {num(pm)} MPa"
                     + (" – PARTIAL RESULT" if s.partial else "") + " (chapter Contacts)")
    if part:
        D.p("WARNING: nonlinear calculation did not converge in states / steps " + ", ".join(f"„{x}“" for x in part)
            + " – the results given are PARTIAL (last converted increment).", bold=True, size=9, color=RED)
    dr = getattr(res, "dynamics", None)
    if dr is not None:
        f = ", ".join(f"{num(v)} Hz" for v in dr.freq[:4])
        txt = _tr(f"Dynamic response: {dr.K} modes ({f}{(' …' if dr.K > 4 else '')})")
        if dr.harmonic is not None:
            txt += _tr(f"; harmonic response max. σ_vM {num(float(np.max(dr.harmonic['env_S'])))} MPa")
        if dr.random is not None:
            sm = dr.random.get("summary", {})
            txt += _tr(f"; random vibration 3σ σ_vM {num(sm.get('max_mises_3sigma_MPa'))} MPa, D = {num(sm.get('max_damage'), 3)}")
        if dr.shock is not None:
            txt += _tr(f"; shock max. σ_vM {num(dr.shock.get('summary', {}).get('max_mises_MPa'))} MPa")
        D.bullet(txt + " (chapter Dynamics; damping is an uncertain input)")
    elif st.dynamics.enabled:
        D.bullet("Dynamics is defined, results are not available.")


# ===================================================================== pictures and charts

def _field_image(sess, values, label, unit, src, img_dir: Path, name: str, warnings, parts=None, status=None,
                 U=None, view="iso"):
    from ..mcp.f8_scenes import field_scene
    from .images import render
    try:
        scene, _ = field_scene(sess, values, label, unit, src, U=U, parts=parts, status=status)
        img_dir.mkdir(parents=True, exist_ok=True)
        p = img_dir / f"{name}.png"
        render(scene, p, view, 1600, 900)
        return p
    except Exception as e:  # noqa: BLE001 – the report continues without the picture
        warnings.append(f"Figure {label} ({src}): {e}")
        return None


def _plot(fn, img_dir: Path, name: str, warnings, *args, **kw):
    try:
        img_dir.mkdir(parents=True, exist_ok=True)
        p = img_dir / f"{name}.png"
        fn(p, *args, **kw)
        return p
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Chart {name}: {e}")
        return None


# ===================================================================== kontakty

def contacts_section(D, sess, o, img_dir: Path, images: list, warnings: list, label: str) -> None:
    from ..mcp.f8_common import (INITIAL_LABELS, METHOD_LABELS, PURPOSE_LABELS, SPLIT_LABELS, mu_info,
                                 pair_effective, type_label)
    st = sess.study
    res = sess.results
    cs = st.contact_settings
    an = st.analysis
    D.h(f"{label}.1 Contact pairs and settings", 2)
    pairs = st.all_contacts() if hasattr(st, "all_contacts") else list(st.contacts)
    rows, sources = [], []
    for c in pairs:
        mi = mu_info(st, c)
        eff = pair_effective(st, c)
        mu_txt = num(mi.get("mu"), 3) if mi.get("mu") is not None else "–"
        if c.kind == "frictional":
            sources.append(f"„{c.name}“: μ = {mu_txt} – {mi['source']}"
                           + (f"; library range {num(mi['range'][0], 3)}–{num(mi['range'][1], 3)}"
                              if mi.get("range") and None not in mi["range"] else ""))
        rows.append([c.name + ("" if c.enabled else " (off)"), c.a, c.b or "ground (support)", type_label(c.kind),
                     mu_txt, PURPOSE_LABELS.get(c.purpose, c.purpose) if c.kind == "frictional" else "–",
                     num(eff.get("K_N_mm3"), 3) + (" (auto)" if eff.get("K_auto") else ""),
                     num(eff.get("lambda_N_mm3"), 3) + (" (auto)" if eff.get("lambda_auto") else "")
                     if c.kind in ("frictional", "rough", "no_separation") else "–",
                     num(c.clearance, 3) if c.clearance is not None else "–",
                     _tr(INITIAL_LABELS.get(c.initial, c.initial or "–"))
                     + (_tr(f" ({num(c.gap, 3)} mm)") if c.gap is not None else "")])
    if rows:
        D.table(["Contact pair", "side A", "side B", "type", "μ [–]", "μ purpose", "K [N/mm³]", "λ [N/mm³]",
                 "interference / clearance [mm]", "initial state"], rows,
                caption="Contact pairs (K = normal stiffness, λ = stick slope – penalty; interference radial, negative = interference)", font=7)
    else:
        D.p("The study has no contact pairs (nonlinear analysis without contact).", italic=True)
    for s in sources:
        D.p("Source of the friction coefficient – " + s, size=8)
    kv = [("Contact formulation", METHOD_LABELS.get(cs.method, cs.method) + " (CalculiX *CONTACT PAIR)"),
          ("Interface meshing", SPLIT_LABELS.get(cs.split, cs.split)),
          ("Large deformations (NLGEOM)", "yes" if cs.nlgeom else "no"),
          ("Calculation type", "step sequence (history)" if an.nonlinear and not an.branches and an.sequence
           else "branches – each load case separately after [bolt preload]")]
    if cs.stabilize != "none":
        kv.append(("Stabilization", f"weak springs {num(cs.stabilize_k)} N/mm" if cs.stabilize_k else "weak springs"))
    D.kv(kv)
    if an.nonlinear and not an.branches and an.sequence:
        rows = []
        for s in an.sequence:
            mot = "; ".join(_tr(f"{m.target}: displacement {'xyz'[int(m.dof) - 1]} {num(m.amplitude[-1][1], 3)} mm")
                            if m.kind == "disp" else
                            _tr(f"{m.target}: rotation {'xyz'[int(m.dof) - 1]} {num(m.amplitude[-1][1], 3)} rad")
                            for m in s.prescribed)
            rows.append([s.name, s.case or "–", num(s.scale, 3), {"lock": "locked", "preload": "preload",
                                                                  "free": "free"}.get(s.bolts, s.bolts),
                         ", ".join(s.contacts_off) or "–",
                         ", ".join(f"{k}: μ = {num(v, 3)}" for k, v in s.mu_change.items()) or "–", mot or "–",
                         f"{num(s.dt0, 2)} / {num(s.dt_min, 2)} / {num(s.dt_max, 2)}, max {s.max_inc}", s.cycles])
        D.table(["Step", "load case", "multiplier", "bolts", "pairs switched off", "friction change", "prescribed motion",
                 "increments dt0 / min / max", "cycles"], rows, caption="Load step sequence", font=7)
    if res is None:
        _contact_warnings(D, st, res)
        return
    # ---------------------------------------------------------------- results
    D.h(f"{label}.2 Contact results", 2)
    rows = []
    for s in res.steps:
        for name, c in (s.contacts or {}).items():
            rows.append([s.name + (_tr(" (PARTIAL)") if s.partial else ""), name, num(c.get("Fn_N")), num(c.get("Ft_N")),
                         f"{num(c.get('area_contact_mm2'))} ({num(100 * (c.get('contact_fraction') or 0), 3)} %)",
                         num(c.get("p_max_MPa")) + (" !" if c.get("peak_at_edge") else ""),
                         num(c.get("p_max_interior_MPa")),
                         num(c.get("p_force_MPa") if c.get("p_force_MPa") is not None else c.get("p_mean_MPa")),
                         num(c.get("open_max_mm"), 3), num(c.get("slip_max_mm"), 3),
                         f"{c.get('nodes_stick', 0)} / {c.get('nodes_slip', 0)} / {c.get('nodes_open', 0)}"])
    if rows:
        D.table(["State / step", "pair", "Fn [N]", "Ft [N]", "contact area [mm²] (fraction)", "p_max nodal [MPa]",
                 "p_max inside surface [MPa]", "p_mean = Fn/A [MPa]", "max. opening [mm]", "max. slip [mm]",
                 "nodes sticking / sliding / open"], rows,
                caption="Contact forces and pressure per case / step (Fn, Ft = sum over the pair surface; the mean pressure from the force is reliable, the peak nodal pressure depends on formulation and mesh)", font=7)
        for w in (x for x in res.warnings if x.startswith("Contact '") and "free edge" in x):
            D.p(w, bold=True, size=9, color=RED)
    nl = res.info.get("nonlinear") or {}
    for job, p in (nl.get("partial") or {}).items():
        D.p(f"PARTIAL RESULT ({job}): calculation not finished – the last converted increment of step {p.get('last_step')} of {p.get('of_steps')} applies, increment cut-back {p.get('cutbacks', 0)}×"
            + (f", last step time {num(p.get('last_step_time'), 3)}" if p.get("last_step_time") is not None else "")
            + ".", bold=True, size=9, color=RED)
    if o.images:
        from ..mcp.f8_scenes import contact_side_parts
        side = contact_side_parts(st, res.mesh, None, "slave") or contact_side_parts(st, res.mesh, None, "a")
        from ..mcp.f8_scenes import contact_view
        cview = contact_view(res.mesh)
        idx = [i for i, s in enumerate(res.steps) if "CPRESS" in s.fields]
        for i in idx[-MAX_IMAGE_STEPS:]:
            s = res.steps[i]
            src = f"state “{s.name}“" + (" – PARTIAL" if s.partial else "")
            p = _field_image(sess, s.fields["CPRESS"], "Contact pressure", "MPa", src, img_dir, f"cpress_{i}", warnings,
                             parts=side, view=cview)
            if p:
                images.append(str(p))
                D.image(p, f"Contact pressure – {src} (parts of the pair's slave surface)")
            if "STATUS" in s.fields:
                p = _field_image(sess, None, "Contact state", "–", src, img_dir, f"status_{i}", warnings, parts=side,
                                 status=s.fields["STATUS"], view=cview)
                if p:
                    images.append(str(p))
                    D.image(p, f"Contact status (sticking / sliding / open) – {src}")
        if len(idx) > MAX_IMAGE_STEPS:
            D.p(f"Maps only for the last {MAX_IMAGE_STEPS} cases / steps (total {len(idx)}).", italic=True, size=9)
    from ..mcp.tools_contact import curve_series
    curves = curve_series(res)
    if curves:
        rows = []
        for tg, c in curves.items():
            F = np.asarray(c["F"])
            k = int(np.argmax(np.abs(F))) if len(F) else 0
            tc = c["kind"] == "time"
            rows.append([tg, "pair force in time" if tc else "displacement" if c["kind"] == "disp" else "rotation",
                         "Fn" if tc else "xyz"[int(c["dof"]) - 1], len(F),
                         num(float(abs(F[k]))) if len(F) else "–", num(float(c["u"][k])) if len(F) else "–"])
        D.table(["Prescribed motion / pair", "type", "direction", "points", "max |F| [N / N·mm]", "at travel [mm / rad] (time [s])"],
                rows, caption="Force–travel curves (reaction of the driven constraint per increment) and pair force in time", font=8)
        if o.images:
            from ..mcp.f8_plots import xy_plot
            p = _plot(xy_plot, img_dir, "curves", warnings,
                      [{"x": c["u"], "y": c["F"], "label": tg} for tg, c in curves.items()],
                      "Force – travel of the prescribed motion", "travel [mm] / rotation [rad] / step time", "reaction [N] / [N·mm]")
            if p:
                images.append(str(p))
                D.image(p, "Force–travel curve of the prescribed motion")
    _contact_warnings(D, st, res)


def _contact_warnings(D, st, res) -> None:
    """Mandatory red warnings of the contact (always, like the creep in printing)."""
    D.p("WARNING – NONLINEAR ANALYSIS: contact results cannot be superposed or scaled linearly; each load case / step is a separate nonlinear analysis (a sequence depends on the history). The safety factor is a stress margin, not a load multiplier; fatigue by superposition of load cases is not valid.", bold=True, size=9, color=RED)
    fr = [c for c in st.active_contacts() if c.kind == "frictional"]
    if fr:
        D.p("WARNING – FRICTION COEFFICIENT: μ is an estimate (friction library, scatter between sources ±30 % or more) – check force transfer by friction with μ_min, resistance and drive forces with μ_max; report results for both bounds.",
            bold=True, size=9, color=RED)
    D.p("WARNING – PENALTY: the normal stiffness K and the stick slope λ are numerical parameters (penetration ≈ p/K, elastic slip ≈ τ/λ); joint stiffness and pressure distribution may be sensitive to them – check by changing K (×2).",
        bold=True, size=9, color=RED)
    partial = res is not None and any(s.partial for s in res.steps)
    D.p("WARNING – INCREMENTS AND CONVERGENCE: the load is applied in increments with automatic reduction; "
        + ("the analysis DID NOT CONVERGE – the results shown are PARTIAL (last converged increment), the full load was not carried." if partial else
           "the analysis converged to the end of all steps. Without convergence only the last converged increment is valid (partial result)."), bold=True, size=9, color=RED)


# ===================================================================== dynamika

def _damping_text(dmp) -> str:
    if dmp.kind == "rayleigh":
        return f"Rayleigh α·M + β·K, ζ = {num(100 * dmp.zeta, 3)} % at {num(dmp.f1)} and {num(dmp.f2)} Hz"
    if dmp.kind == "structural":
        eta = dmp.eta if dmp.eta else 2 * dmp.zeta
        return f"hysteretic (structural), loss factor η = {num(eta, 3)} (harmonic response only)"
    if dmp.kind == "per_material":
        return "by materials (weighted by strain energy): " + ", ".join(f"{k} {num(100 * v, 3)} %"
                                                                          for k, v in dmp.per_material.items())
    if dmp.table:
        return "modal, table ζ(f): " + ", ".join(f"{num(f)} Hz → {num(100 * z, 3)} %" for f, z in dmp.table)
    return f"modal, ζ = {num(100 * dmp.zeta, 3)} % for all modes"


def dynamics_section(D, sess, o, img_dir: Path, images: list, warnings: list, label: str) -> None:
    from ..mcp.f8_common import excitation_text, mass_text
    st = sess.study
    ds = st.dynamics
    res = sess.results
    dr = getattr(res, "dynamics", None) if res is not None else None
    D.h(f"{label}.1 Dynamics input", 2)
    # texts composed of parts are translated part by part (_tr) – the report is in English
    kv = [("Modal basis", _tr(f"{ds.modes} modes" + (f", up to {num(ds.f_max)} Hz" if ds.f_max else "")
                               + " (CalculiX *FREQUENCY, same mesh and supports as statics)")),
          ("Damping", _damping_text(ds.damping)),
          ("CalculiX reference", "yes (*STEADY STATE DYNAMICS / *MODAL DYNAMIC at probes)" if ds.verify_ccx else "no")]
    h, t, r, s, pr = ds.harmonic, ds.transient, ds.random, ds.shock, ds.prestress
    if h.enabled:
        kv.append(("Harmonic response", _tr(f"{num(h.f_min)}–{num(h.f_max)} Hz, {h.points} points between resonances") + "; "
                   + "; ".join(excitation_text(e, _tr) for e in h.excitation)
                   + (_tr(f"; sweep {num(h.sweep_rate)} oct/min × {num(h.sweeps)}") if h.sweep_rate else "")))
    if t.enabled:
        kv.append(("Transient response", _tr(f"0–{num(t.t_end)} s, step {num(t.dt)} s") + ", "
                   + _tr("modal (exact modal solution)" if t.method == "modal" else "direct integration ccx *DYNAMIC")
                   + "; " + "; ".join(excitation_text(e, _tr) for e in t.excitation)
                   + (_tr(f"; settling tolerance {num(t.settle_tol)} mm") if t.settle_tol else "")))
    if r.enabled:
        try:
            from ..dynamics.psd import asd_rms
            g = f", g_rms {num(asd_rms(r.asd), 3)} g"
        except Exception:  # noqa: BLE001
            g = ""
        d = _tr(f"direction ({'; '.join((num(v, 3) for v in r.direction))})" if r.direction else f"axis {'xyz'[int(r.dof) - 1]}")
        meth = _tr({"nb": "narrow-band", "dirlik": "Dirlik", "tb": "Tovo–Benasciutti"}.get(r.method, r.method))
        kv.append(("Random vibration", _tr(f"ASD {len(r.asd)} points") + g + f", {d}, {num(r.duration_h)} h, "
                   + _tr(f"spectral fatigue {meth}")))
    if s.enabled:
        d = _tr(f"direction ({'; '.join((num(v, 3) for v in s.direction))})" if s.direction else f"axis {'xyz'[int(s.dof) - 1]}")
        pname = _tr({"halfsine": "half-sine", "sawtooth": "sawtooth with terminal peak", "trapezoid": "trapezoid"}
                    .get(s.pulse, s.pulse))
        kv.append(("Shock", (_tr("specified response spectrum") if s.pulse == "srs_table" else
                           f"{pname} {num(s.A)} g, {num(1000 * s.tau)} ms")
                   + f", {d}, " + _tr(f"combination {s.combine.upper()}")
                   + (_tr(", missing mass (ZPA)") if s.missing_mass else "")))
    if pr.enabled:
        kv.append(("Preload", _tr(f"modal analysis after case “{pr.case}” (stress stiffening)")
                   + (_tr(f"; speeds {', '.join((num(v) for v in pr.rpm_list))} rpm") if pr.rotating else "")))
    D.kv(kv)
    if st.masses:
        D.table(["Added mass", "faces (selection)", "value"],
                [[m.name or m.selection, m.selection, mass_text(m, _tr)] for m in st.masses],
                caption="Added masses (parts not modelled, components)", widths=(0.25, 0.25, 0.5), font=8)
    if ds.probes:
        D.table(["Probe", "location", "quantity"],
                [[p.name, vec(p.point, 5) + " mm" if p.point is not None else f"selection {p.selection}",
                  {"U": "displacement", "S": "stress", "A": "acceleration"}.get(p.quantity, p.quantity)] for p in ds.probes],
                caption="Dynamics output probes", widths=(0.3, 0.5, 0.2), font=8)
    if o.images:
        _profile_plots(D, ds, dr, img_dir, images, warnings)
    if dr is None:
        D.p("Dynamics results are not available.", italic=True)
        _dyn_warnings(D)
        return
    # ---------------------------------------------------------------- modes and checks
    D.h(f"{label}.2 Modal basis", 2)
    from ..mcp.tools_dynamics import modes_table
    mt = modes_table(dr, 30)
    rows = [[m["mode"], num(m["f_Hz"]), num(100 * m["zeta"], 3), *[num(100 * v, 3) for v in m["eff_mass_fraction_xyz"]],
             " / ".join(num(100 * v, 3) for v in m["cumulative_xyz"])] for m in mt]
    D.table(["Mode", "f [Hz]", "ζ [%]", "m_eff x [%]", "m_eff y [%]", "m_eff z [%]", "Σ x / y / z [%]"], rows,
            caption=f"Natural frequencies and effective masses (model mass {num(1000 * dr.model_mass)} kg"
                    + (f"; {len(mt)} of {dr.K} modes shown" if dr.K > len(mt) else "") + ")", font=8)
    c = dr.checks or {}
    if c:
        D.kv([("Normalization φᵀMφ = 1 (own integration)", num(c.get("mass_norm_error"), 2)),
              ("Participation factors Γ vs. ccx", num(c.get("gamma_error"), 2)),
              ("Equilibrium of modal reactions", num(c.get("mode_equilibrium"), 2)),
              ("Mass coverage Σ m_eff x / y / z", " / ".join(f"{num(100 * v, 3)} %" for v in
                                                           c.get("mass_fraction_xyz", [])))])
    sub = 3
    if dr.harmonic is not None:
        D.h(f"{label}.{sub} Harmonic response", 2)
        sub += 1
        _harmonic(D, sess, dr, o, img_dir, images, warnings)
    if dr.transient is not None:
        D.h(f"{label}.{sub} Transient response", 2)
        sub += 1
        _transient(D, sess, dr, o, img_dir, images, warnings)
    if dr.random is not None:
        D.h(f"{label}.{sub} Random vibration", 2)
        sub += 1
        _random(D, sess, dr, o, img_dir, images, warnings)
    if dr.shock is not None:
        D.h(f"{label}.{sub} Shock", 2)
        sub += 1
        _shock(D, sess, dr, o, img_dir, images, warnings)
    et = dr.info.get("etalon") or {}
    if et:
        rows = []
        for k, v in et.items():
            if isinstance(v, dict):
                rows.append([{"harmonic": "harmonic (SSD)", "transient": "transient (MD)"}.get(k, k),
                             num(v.get("max_rel_error"), 2) if "max_rel_error" in v else "–",
                             v.get("note") or v.get("skipped") or v.get("error") or ""])
            else:
                rows.append([k, "–", str(v)])
        D.table(["CalculiX reference", "max. relative difference at probes", "note"], rows,
                caption="Check of the modal superposition against CalculiX", font=8)
    if dr.warnings:
        for w in dr.warnings:
            D.bullet(w)
    _dyn_warnings(D)


def _dyn_warnings(D) -> None:
    D.p("WARNING – DAMPING: damping is an uncertain input (sensitivity ±50 %); the resonant response is proportional to 1/ζ, the RMS of random vibration approximately to 1/√ζ. Verify the values by measurement (−3 dB bandwidth, logarithmic decrement) or report the results for both bounds.", bold=True, size=9, color=RED)


# ===================================================================== pohyb (mechanismus)

MOTION_KINDS = {"kinematic": "kinematics (prescribed drivers, inverse dynamics)",
                "static": "quasi-statics (equilibrium in a series of positions, no inertia)",
                "dynamic": "forward dynamics in time"}


def has_motion(sess) -> bool:
    return getattr(sess, "motion", None) is not None


def motion_section(D, sess, o, img_dir: Path, images: list, warnings: list, label: str) -> None:
    """Section Motion (specification F8 6.4): bodies, joints and drivers, maxima of the reactions and their moment, graphs of the reactions,
    conversion to FE, check of the equilibrium and of the power balance, warnings (redundancy, collisions).
    """
    from ..motion.model import JOINT_LABELS
    mr = sess.motion
    st = sess.study
    m = mr.meta
    s = mr.summary()
    solver = "Project Chrono" if (m.get("solver") or st.mechanism.run.solver) == "chrono" else "Exudyn"
    D.p(f"Rigid-body mechanism ({solver}): {MOTION_KINDS.get(s.get('kind'), s.get('kind') or '–')}, duration {num(s.get('t_end_s'))} s, {s.get('frames')} output times. Joint and driver forces include inertia, gravity, friction, stops and force elements. Units: mm, N, N·mm, s, rad.")
    # ---------------------------------------------------------------- setup
    D.h(f"{label}.1 Bodies, joints and drivers", 2)
    rows = []
    for b in m.get("bodies", []):
        rows.append([b["name"] + (" (ground)" if b.get("ground") else ""), ", ".join(b.get("parts") or []) or "–",
                     "–" if b.get("ground") else num(1e3 * float(b.get("mass") or 0.0))])
    if rows:
        D.table(["Body", "parts", "mass [kg]"], rows, caption="Mechanism bodies", font=8)
    rows = [[j["name"], JOINT_LABELS.get(j["kind"], j["kind"]), j.get("a_name") or "–", j.get("b_name") or "–",
             vec(j.get("origin") or [0, 0, 0]), vec(j.get("axis") or [0, 0, 0])] for j in m.get("joints", [])]
    if rows:
        D.table(["Joint", "type", "body A", "body B", "origin [mm]", "axis"], rows, caption="Joints", font=8)
    drv = {d.name or d.joint: d for d in st.mechanism.drivers}
    rows = []
    for d in s.get("drivers", []):
        sd = drv.get(d["driver"])
        rows.append([d["driver"], d.get("joint") or "–", d.get("quantity") or "–",
                     f"{sd.law} {sd.data}" if sd is not None else "–"])
    if rows:
        D.table(["Driver", "hinge", "quantity", "motion law"], rows, caption="Drivers", font=8)
    dof = m.get("dof") or {}
    if dof:
        D.p(f"Mobility (Grübler): {dof.get('grubler')}, free degrees {dof.get('dof_free')}, driven drives {dof.get('dof_driven')}, redundant constraints {dof.get('redundant', 0)}"
            + (f" ({', '.join(map(str, dof.get('redundant_list') or []))})" if dof.get("redundant_list") else "")
            + ".", size=9)
    # ---------------------------------------------------------------- results
    D.h(f"{label}.2 Joint reactions and drivers", 2)
    rows = [[j["joint"], JOINT_LABELS.get(j["kind"], j["kind"]), num(j["max_force_N"]), num(j["t_max_force_s"]),
             num(j["max_moment_Nmm"]), num(j.get("max_friction")) if j.get("max_friction") is not None else "–",
             num(j.get("max_stop_force")) if j.get("max_stop_force") is not None else "–"] for j in s.get("joints", [])]
    if rows:
        D.table(["Joint", "type", "max |F| [N]", "at time [s]", "max |M| [N·mm]", "max friction", "max stop force"], rows,
                caption="Maximum joint forces and moments (force on body B from body A, governing instant)",
                font=8)
    rows = [[d["driver"], d.get("joint") or "–", f"{num(d['max_abs'])} {d['unit']}", num(d["t_max_s"]),
             f"{num(d['rms'])} {d['unit']}", num(d["max_power_W"])] for d in s.get("drivers", [])]
    if rows:
        D.table(["Driver", "hinge", "max |M / F|", "at time [s]", "RMS", "max power [W]"], rows,
                caption="Driver forces and power", font=8)
    rows = [[b["body"], num(b["mass_kg"]), num(b["max_velocity_mm_s"]), num(b["max_acceleration_mm_s2"]),
             num(b["max_omega_rad_s"])] for b in s.get("bodies", [])]
    if rows:
        D.table(["Body", "mass [kg]", "max |v| [mm/s]", "max |a| [mm/s²]", "max |ω| [rad/s]"], rows,
                caption="Body motion (center of gravity)", font=8)
    if o.images and mr.joint_names:
        from ..mcp.f8_plots import xy_plot
        ser = []
        for name in mr.joint_names[:8]:
            try:
                x = mr.series(f"reaction:{name}")
                ser.append({"x": x["t"], "y": x["magnitude"], "label": name})
            except (KeyError, ValueError):
                continue
        if ser:
            p = _plot(xy_plot, img_dir, "motion_reactions", warnings, ser, "Joint reactions", "time [s]", "|F| [N]")
            if p:
                images.append(str(p))
                D.image(p, "Joint reaction magnitude over time")
        if mr.driver_names:
            ser = []
            for name in mr.driver_names[:8]:
                try:
                    x = mr.series(f"drive:{name}")
                    ser.append({"x": x["t"], "y": x["values"][:, 0], "label": f"{name} [{x['unit']}]"})
                except (KeyError, ValueError):
                    continue
            if ser:
                p = _plot(xy_plot, img_dir, "motion_drivers", warnings, ser, "Driver torque / force", "time [s]",
                          "M [N·mm] / F [N]")
                if p:
                    images.append(str(p))
                    D.image(p, "Driver torque / force over time")
    # ---------------------------------------------------------------- checks
    kv = []
    eq = m.get("equilibrium") or {}
    if eq:
        kv.append(("Body equilibrium (d'Alembert)", f"max. relative residual {num(eq.get('max_rel'), 3)} "
                   + ("– OK" if eq.get("ok") else "– FAILS") + f" (body {eq.get('body')}, t = {num(eq.get('t'))} s)"))
    pb = m.get("power_balance") or {}
    if pb:
        kv.append(("Power balance", f"max. relative error {num(pb.get('max_rel'), 3)}, integrated {num(pb.get('integral_rel'), 3)} " + ("– OK" if pb.get("ok") else "– DOES NOT MATCH")))
    elif solver == "Project Chrono":
        kv.append(("Power balance", "not computed (contact in Chrono)"))
    col = m.get("collisions_summary")
    if col:
        kv.append(("Collision check", str(col)))
    if kv:
        D.kv(kv)
    # ---------------------------------------------------------------- conversion to FE
    D.h(f"{label}.3 Transfer to FEA", 2)
    lcs = [lc for lc in st.load_cases if lc.name.startswith("MBD")
           or any(getattr(ld, "kind", "") == "inertia" for ld in lc.loads)]
    if lcs:
        D.table(["Load case", "load"], [[lc.name, ", ".join(sorted({ld.kind for ld in lc.loads}))]
                                                 for lc in lcs],
                caption="Load cases taken from the mechanism (joint reactions as remote loads, inertia as a body force)", font=8)
        tf = st.mechanism.run.to_fea
        how = {"instants": "selected instants", "max": "instant of the maximum",
               "history": "whole history (unit cases + fatigue)"}.get(tf.mode, tf.mode)
        D.p(f"Transmission: {how}" + (f" – quantity {tf.quantity}" if tf.mode == "max" and tf.quantity else "")
            + ". FEM support = joint of the body as a remote support; its reaction in FEM should equal the reaction from the mechanism (inertia relief check).", size=9)
    else:
        D.p("The mechanism loads were not transferred to FEA (fea_motion_export_loads / Motion step → Transfer to FEA).",
            italic=True)
    for w in mr.warnings:
        D.bullet(w)
    D.p("WARNING – MECHANISM: bodies are rigid (unless defined as flexible), joint clearances are not considered; joint friction is an estimate and redundant constraints distribute the reactions arbitrarily – in FEA the distribution is determined by stiffness. Verify the reactions at the governing instant with an FEA analysis.", bold=True, size=9, color=RED)


def _profile_plots(D, ds, dr, img_dir, images, warnings) -> None:
    from ..mcp.f8_plots import xy_plot
    h = ds.harmonic
    if h.enabled:
        for k, e in enumerate(h.excitation):
            if e.table and e.kind != "force":
                tab = np.asarray(e.table, float)
                p = _plot(xy_plot, img_dir, f"profile_h{k}", warnings,
                          [{"x": tab[:, 0], "y": tab[:, 1] * e.scale, "label": "profile"}],
                          "Harmonic response excitation profile", "frequency [Hz]", f"amplitude [{e.unit}]",
                          logx=True, logy=True)
                if p:
                    images.append(str(p))
                    D.image(p, "Input sine excitation profile")
    r = ds.random
    if r.enabled and len(r.asd) >= 2:
        a = np.asarray(r.asd, float)
        p = _plot(xy_plot, img_dir, "profile_asd", warnings, [{"x": a[:, 0], "y": a[:, 1], "label": "ASD"}],
                  "Input base acceleration ASD", "frequency [Hz]", "ASD [g²/Hz]", logx=True, logy=True)
        if p:
            images.append(str(p))
            D.image(p, "Input random vibration profile (ASD, log-log)")
    if dr is not None and dr.shock is not None and dr.shock.get("pulse_t") is not None:
        s = dr.shock
        p = _plot(xy_plot, img_dir, "profile_pulse", warnings,
                  [{"x": np.asarray(s["pulse_t"]) * 1000, "y": np.asarray(s["pulse_a"]) / 9810.0, "label": "pulse"}],
                  "Base acceleration pulse", "time [ms]", "a [g]")
        if p:
            images.append(str(p))
            D.image(p, "Input shock pulse")


def _probe_rows(dr, kind: str) -> list[list]:
    from ..mcp.tools_dynamics import probe_series
    rr = getattr(dr, kind)
    rows = []
    for name in (rr.get("probes") or {}):
        try:
            u = probe_series(dr, kind, name, "U")
            sm = probe_series(dr, kind, name, "S")
        except Exception:  # noqa: BLE001
            continue
        y = np.asarray(u["y"])
        if u["complex"]:
            from ..dynamics.response import max_over_phase
            mag = max_over_phase(y.real, -y.imag)
        else:
            mag = np.linalg.norm(y, axis=1)
        k = int(np.argmax(mag))
        ks = int(np.argmax(sm["y"]))
        row = [name, num(float(mag[k])), num(float(u["x"][k])), num(float(sm["y"][ks])), num(float(sm["x"][ks]))]
        if kind == "transient":
            st_ = (rr.get("settle") or {}).get(name, "–")
            row.append("does not settle" if st_ is None else num(st_) if st_ != "–" else "–")
        rows.append(row)
    return rows


def _harmonic(D, sess, dr, o, img_dir, images, warnings) -> None:
    h = dr.harmonic
    f = np.asarray(h["f"], float)
    i = int(np.argmax(h["env_S"]))
    D.kv([("Frequency range", f"{num(f[0])}–{num(f[-1])} Hz, {len(f)} frequencies"),
          ("Max. displacement amplitude (relative to the base)", f"{num(float(np.max(h['env_U'])))} mm"),
          ("Max. σ_vM amplitude (max over phase)", f"{num(float(h['env_S'][i]))} MPa at {num(float(f[int(h['env_S_at'][i])]))} Hz, point {vec(sess.results.mesh.nodes[i], 5)} mm")])
    rows = _probe_rows(dr, "harmonic")
    if rows:
        D.table(["Probe", "max. |u| [mm]", "at f [Hz]", "max. σ_vM [MPa]", "at f [Hz]"], rows,
                caption="Response at probes – harmonic", font=8)
    sw = h.get("sweep")
    if sw:
        sm = sw.get("summary", {})
        D.p(f"Fatigue from the sweep ({num(sm.get('rate_oct_min'))} oct/min, {num(sm.get('sweeps'))} passes): max. damage D = {num(sm.get('max_damage'), 3)} at point {vec(sm.get('at', (0, 0, 0)), 5)} mm – "
            + ("PASSES (D < 1)" if (sm.get("max_damage") or 0) < 1 else "FAILS (D ≥ 1)") + ".")
    if not o.images:
        return
    from ..mcp.tools_dynamics import probe_series
    from ..mcp.f8_plots import frf_plot
    for name in list(h.get("probes") or {})[:3]:
        try:
            s = probe_series(dr, "harmonic", name, "U")
        except Exception:  # noqa: BLE001
            continue
        y = np.asarray(s["y"])
        p = _plot(frf_plot, img_dir, f"frf_{len(images)}", warnings, s["x"],
                  [{"y": y[:, c], "label": f"u{'xyz'[c]}"} for c in range(3) if np.abs(y[:, c]).max() > 1e-9 * np.abs(y).max()],
                  f"Displacement FRF – probe “{name}“", "amplitude [mm]")
        if p:
            images.append(str(p))
            D.image(p, f"Frequency response (amplitude and phase) at probe “{name}”")
    p = _field_image(sess, h["env_S"], "von Mises amplitude envelope", "MPa",
                     f"harmonic response {num(f[0])}–{num(f[-1])} Hz", img_dir, "harm_env", warnings, view=o.image_view)
    if p:
        images.append(str(p))
        D.image(p, "Envelope of the von Mises stress amplitude over frequencies (max over phase)")


def _transient(D, sess, dr, o, img_dir, images, warnings) -> None:
    tr = dr.transient
    t = np.asarray(tr["t"], float)
    te = np.asarray(tr["t_env"], float)
    i = int(np.argmax(tr["env_S"]))
    D.kv([("Time / step", f"0–{num(t[-1])} s, {len(t)} time points"),
          ("Max. displacement (relative to the base)", f"{num(float(np.max(tr['env_U'])))} mm"),
          ("Max. σ_vM", f"{num(float(tr['env_S'][i]))} MPa at time {num(float(te[int(tr['env_S_at'][i])]))} s")])
    rows = _probe_rows(dr, "transient")
    if rows:
        D.table(["Probe", "max. |u| [mm]", "at time [s]", "max. σ_vM [MPa]", "at time [s]", "settling time [s]"], rows,
                caption="Response at probes – transient", font=8)
    if not o.images:
        return
    from ..mcp.tools_dynamics import probe_series
    from ..mcp.f8_plots import xy_plot
    for name in list(tr.get("probes") or {})[:2]:
        try:
            s = probe_series(dr, "transient", name, "U")
        except Exception:  # noqa: BLE001
            continue
        y = np.asarray(s["y"])
        p = _plot(xy_plot, img_dir, f"tr_{len(images)}", warnings,
                  [{"x": s["x"], "y": y[:, c], "label": f"u{'xyz'[c]}"} for c in range(3)],
                  f"Displacement over time – probe “{name}”", "time [s]", "displacement [mm]")
        if p:
            images.append(str(p))
            D.image(p, f"Displacement history at probe “{name}”")
    p = _field_image(sess, tr["env_S"], "von Mises envelope", "MPa", f"transient response 0–{num(t[-1])} s", img_dir,
                     "tr_env", warnings, view=o.image_view)
    if p:
        images.append(str(p))
        D.image(p, "Envelope of the von Mises stress over time")


def _random(D, sess, dr, o, img_dir, images, warnings) -> None:
    r = dr.random
    sm = r.get("summary", {})
    D.kv([("Input", f"g_rms {num(sm.get('g_rms_input'), 3)} g, duration {num(sm.get('duration_h'))} h, method {sm.get('method')}"),
          ("Max. RMS displacement", f"{num(sm.get('max_U_rms_mm'))} mm"),
          ("Max. RMS σ_vM / 3σ", f"{num(sm.get('max_mises_rms_MPa'))} / {num(sm.get('max_mises_3sigma_MPa'))} MPa at point {vec(sm.get('mises_rms_at', (0, 0, 0)), 5)} mm (Segalman – informative)"),
          ("Spectral fatigue", f"max. damage D = {num(sm.get('max_damage'), 3)}, min. life {num(sm.get('min_life_h'), 3)} h at point {vec(sm.get('damage_at', (0, 0, 0)), 5)} mm – "
                               + ("PASSES (D < 1)" if (sm.get("max_damage") or 0) < 1 else "FAILS (D ≥ 1)"))])
    if sm.get("note"):
        D.p(sm["note"], italic=True, size=9)
    if not o.images:
        return
    for key, label, unit, cap in (("S_rms", "RMS von Mises", "MPa", "RMS von Mises stress (1σ)"),
                                  ("D", "Spectral fatigue – log10 D", "log10 D", "Spectral fatigue damage (log10 D)")):
        vals = np.asarray(r[key], float)
        if key == "D":
            vals = np.log10(np.maximum(vals, 1e-12))
        p = _field_image(sess, vals, label, unit, "random vibration", img_dir, f"rnd_{key}", warnings, view=o.image_view)
        if p:
            images.append(str(p))
            D.image(p, cap)
    from ..mcp.f8_plots import xy_plot
    for name, s in list((r.get("probes") or {}).items())[:2]:
        a = np.asarray(s.get("A_psd_g2"), float)
        if a.ndim != 2:
            continue
        p = _plot(xy_plot, img_dir, f"psd_{len(images)}", warnings,
                  [{"x": r["f"], "y": a[:, c], "label": f"a{'xyz'[c]}"} for c in range(3) if a[:, c].max() > 0],
                  f"Absolute acceleration PSD – probe “{name}”", "frequency [Hz]", "PSD [g²/Hz]", logx=True, logy=True)
        if p:
            images.append(str(p))
            D.image(p, f"Acceleration PSD at probe “{name}”")


def _shock(D, sess, dr, o, img_dir, images, warnings) -> None:
    s = dr.shock
    sm = s.get("summary", {})
    pname = _tr({"halfsine": "half-sine", "sawtooth": "sawtooth with terminal peak", "trapezoid": "trapezoid",
                 "srs_table": "specified response spectrum"}.get(sm.get("pulse"), str(sm.get("pulse"))))
    rows = [("Pulse", f"{pname} {num(sm.get('A_g'))} g, {num(1000 * (sm.get('tau_s') or 0))} ms, ZPA "
                     f"{num(sm.get('zpa_g'))} g"),
            ("Mode combination", f"{str(sm.get('combine', '')).upper()}" + (_tr(" + missing mass")
                                                                         if sm.get("missing_mass") else "")),
            ("Max. displacement", f"{num(sm.get('max_U_mm'))} mm"),
            ("Max. σ_vM (response spectrum)", f"{num(sm.get('max_mises_MPa'))} MPa at point {vec(sm.get('at', (0, 0, 0)), 5)} mm")]
    if sm.get("transient_max_mises_MPa") is not None:
        rows.append(("Max. σ_vM (transient response to the pulse)", f"{num(sm['transient_max_mises_MPa'])} MPa"))
    D.kv(rows)
    if sm.get("note"):
        D.p(sm["note"], italic=True, size=9)
    if not o.images:
        return
    if s.get("srs_f") is not None:
        from ..mcp.f8_plots import xy_plot
        p = _plot(xy_plot, img_dir, "srs", warnings, [{"x": s["srs_f"], "y": np.asarray(s["srs"]) / 9810.0,
                                                       "label": "SRS"}],
                  "Shock response spectrum (absolute acceleration)", "frequency [Hz]", "S_a [g]", logx=True, logy=True)
        if p:
            images.append(str(p))
            D.image(p, "Shock response spectrum (SRS)")
    p = _field_image(sess, s["mises"], "von Mises – shock", "MPa", "shock (SRSS/CQC)", img_dir, "shock", warnings,
                     view=o.image_view)
    if p:
        images.append(str(p))
        D.image(p, "von Mises stress under shock (mode combination)")
