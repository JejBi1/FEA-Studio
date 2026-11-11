"""Advice according to the place of failure of the printed part (F5, A4) – INDICATIVE.

From the result (StepResult.printing: deciding region perimeter / shell / core, stress in the print axes or line axes
σ11 σ22 σ33 τ12 τ13 τ23) the component that decides is determined by part – its share in the failure index (|σ_i| / strength_i)²
with the strength by the sign – and from it the advice with a justification:
- joint of layers (σ33, τ13, τ23 ≥ 50 % of the index) → another build direction (the best of 6 axes by calculation: rank_build_directions);
- perimeter along the line → more perimeters; across the line → perimeters help little, build direction / rotation of the part;
- shell without a line (upper/lower full layers) → more upper/lower layers;
- core (infill) → higher density (estimate from the strength of the infill ~ ρ^m), for multiaxial stress a 3D pattern (gyroid / cubic);
- thin place (full infill) → the infill setting does not help, strengthen the geometry; switched-off full infill of thin places → switch it on;
- a part with a large reserve → the infill can be reduced.
The advice is to be verified by calculating the variant (printing.variants) and by a test – the infill model is indicative.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .model import AXIS_DIRS, PrintVariant

COMPONENTS = ("σ11", "σ22", "σ33", "τ12", "τ13", "τ23")
NOTE = "INDICATIVE – advice from the homogenized model; verify the benefit by analyzing a variant and by testing."
PATTERNS_3D = ("gyroid", "cubic", "adaptivecubic", "3dhoneycomb")
LAYER_SHARE = 0.5          # fraction of the joint of the layers in the failure index, from which the build direction is advised


def contributions(sig, m) -> np.ndarray:
    """|σ_i| / strength_i (strength by the sign for the normal components) in the axes of the material m (infill.Ortho)."""
    s = np.asarray(sig, float)
    X = [m.Xt if s[0] >= 0 else m.Xc, m.Ytv if s[1] >= 0 else m.Ycv, m.Zt if s[2] >= 0 else m.Zc,
         m.Sxy, m.Sz, m.Sz]
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.abs(s) / np.array(X, float)
    return np.nan_to_num(c, nan=0.0, posinf=0.0)


def governing(results) -> dict[str, dict]:
    """The worst place per printed part over all states (incl. the creep steps): {part: {step, step_index,
    min_safety, region, at, stress_local, thin, element}}.
    """
    out: dict[str, dict] = {}
    for i, st in enumerate(results.steps):
        for p, d in ((st.printing or {}).get("parts") or {}).items():
            if p not in out or d["min_safety"] < out[p]["min_safety"]:
                out[p] = dict(d, step=st.name, step_index=i)
    return out


def _region_material(study, part: str, region: str):
    from . import part_materials
    ms, ml, mc = part_materials(study, part)
    if region.startswith("core"):
        return mc
    return ml if region.startswith(("perimeter", "perimetr")) else ms


def _best_axes(results, g: dict, current) -> list[str]:
    """Axes ±X/±Y/±Z most perpendicular to the direction of the largest principal stress (global stress of the element) – the stress then lies
    in the plane of the layer. Without a simultaneous build direction.
    """
    from ..results import tensor
    e = g.get("element")
    if e is None:
        return []
    st = results.steps[g["step_index"]]
    S = st.S[results.mesh.tets[e]].mean(axis=0)
    w, v = np.linalg.eigh(tensor(S))
    n1 = v[:, int(np.argmax(np.abs(w)))]
    cur = np.asarray(current, float)
    cur = cur / np.linalg.norm(cur)
    score = {k: abs(float(np.dot(d, n1))) for k, d in AXIS_DIRS.items()}
    best = min(score.values())
    return [k for k, s in score.items() if s <= best + 0.05 and abs(float(np.dot(AXIS_DIRS[k], cur))) < 0.999]


def _item(part, kind, title, text, reason, g, priority, suggestion=None) -> dict:
    return {"part": part, "kind": kind, "title": title, "text": text, "reason": reason,
            "suggestion": suggestion or {}, "priority": priority, "step": g.get("step"),
            "min_safety": g.get("min_safety"), "region": g.get("region"), "at": g.get("at"), "note": NOTE,
            "orientational": True}


def _fmt(x: float, sig: int = 3) -> str:
    from fea_core.i18n import decimal
    return decimal(f"{x:.{sig}g}")


def advise(study, results, required_safety: float | None = None) -> list[dict]:
    """Advice per printed part (sorted: part, priority 1 = the most important). Item: part, kind (ok | fail | save |
    build_dir | walls | layers | infill | pattern | thin | solid_thin), title, text, reason (justification with numbers),
    suggestion (overrides of PrintSpec for the variant), priority, step, min_safety, region, at, note, orientational;
    for "infill" additionally sweep = {walls, infill} for prefilling the sweep (SweepSpec).
    """
    from .infill import FULL, PATTERNS
    k_req = float(required_safety if required_safety is not None else study.printing.required_safety)
    items: list[dict] = []
    for part, g in governing(results).items():
        spec = study.printing.spec(part)
        k = float(g["min_safety"])
        region = g.get("region", "")
        ks = _fmt(k)
        if not spec.enabled:
            items.append(_item(part, "ok" if k >= k_req else "thin", "Solid part",
                               f"Part is calculated as a full extrusion; k = {ks}"
                               + ("" if k >= k_req else " – print settings will no longer help; strengthen the geometry or choose a stronger material / another build direction."),
                               f"enabled = false, governed by {region}", g, 1 if k < k_req else 3))
            continue
        if k >= 2.0 * k_req:
            sug = {}
            if spec.infill > 0.1 and spec.infill < FULL:
                sug = {"infill": max(0.05, round(spec.infill / 2 / 0.05) * 0.05)}
            items.append(_item(part, "save", "Large margin – material can be saved",
                               f"k = {ks} ≥ 2 × requirement {_fmt(k_req)}. Try lower infill or fewer perimeters (variant, recalculation).", f"min. safety factor {ks}, governed by {region}", g, 3, sug))
            continue
        prio = 1
        sig = g.get("stress_local")
        if sig is None or k >= k_req:
            items.append(_item(part, "ok" if k >= k_req else "fail", "Passes" if k >= k_req else "Fails",
                               f"k = {ks} {('≥' if k >= k_req else '<')} {_fmt(k_req)} (governed by {region}).",
                               f"min. safety factor {ks} in case {g.get('step')}", g, 3 if k >= k_req else 1))
            continue
        m = _region_material(study, part, region)
        c = contributions(sig, m)
        tot = float((c ** 2).sum())
        if tot <= 0:
            continue
        share = c ** 2 / tot
        layer = float(share[2] + share[4] + share[5])
        dom = int(np.argmax(share))
        sig_txt = ", ".join(f"{n} {_fmt(float(v))}" for n, v in zip(COMPONENTS, sig))
        base_reason = (f"{region}, case {g.get('step')}: {sig_txt} MPa; governed by {COMPONENTS[dom]} ({100 * share[dom]:.0f} % of failure index)")
        if g.get("thin"):
            items.append(_item(part, "thin", "Thin region with solid infill",
                               "The critical location is thin (printed with solid infill) – perimeters and infill will not help here. Strengthen the geometry (thickness, fillets), choose a different build direction or a stronger material.",
                               base_reason, g, prio))
        if layer >= LAYER_SHARE:
            axes = _best_axes(results, g, spec.build_dir)
            items.append(_item(part, "build_dir", "Governed by layer bonding – different build direction",
                               f"Stress perpendicular to the layers and shear between the layers (σ33, τ13, τ23) make up {100 * layer:.0f} % of the failure index; the layer bond is the weakest point. Rotate the part so that the principal stress lies in the plane of the layers" + (f" – candidates: {', '.join(axes)}" if axes else "")
                               + "; the best of 6 axes is computed by “Rank build directions” (rank_build_directions).",
                               base_reason, g, prio, {"build_dir": list(AXIS_DIRS[axes[0]])} if axes else {}))
            continue
        if region.startswith("core"):
            _, kx, nx, kz, nz, sx, mx, sz, mz = PATTERNS.get(spec.pattern, PATTERNS["grid"])
            m_exp = mz if dom == 2 else mx
            target = spec.infill * (k_req / k) ** (1.0 / max(m_exp, 0.5))
            target = min(1.0, max(spec.infill + 0.05, np.ceil(target / 0.05) * 0.05))
            n_exp = nz if dom == 2 else nx
            sweep = {"walls": sorted({spec.walls, spec.walls + 2}),
                     "infill": sorted({round(spec.infill, 4), round(float(target), 4)})}
            items.append(_item(part, "infill", "Governed by infill – higher density",
                               f"The core with {100 * spec.infill:g} % {spec.pattern} infill fails. Infill strength grows roughly as ρ^{_fmt(m_exp, 2)} → upper estimate for k ≥ {_fmt(k_req)}: infill {100 * target:.0f} %. However, stiffness grows as ρ^{_fmt(n_exp, 2)} and a stiffer core takes more load – the actual benefit is usually smaller; verify with a perimeters × infill sweep.",
                               base_reason, g, prio, {"infill": float(target)}))
            items[-1]["sweep"] = sweep
            normal = share[:3]
            multi = int((normal >= 0.15).sum()) >= 2 or float(share[2] + share[4] + share[5]) >= 0.3
            if multi and spec.pattern not in PATTERNS_3D:
                parts_txt = ", ".join(f"{n} {100 * sh:.0f} %" for n, sh in zip(COMPONENTS, share) if sh >= 0.1)
                items.append(_item(part, "pattern", "Multiaxial loading of the infill – 3D pattern",
                                   f"The infill is loaded in several directions (fractions {parts_txt}); pattern {spec.pattern} carries mainly in the layer plane. Try gyroid or cubic infill (similar stiffness in all directions).", base_reason, g, prio + 1, {"pattern": "gyroid"}))
            if spec.walls < 3:
                items.append(_item(part, "walls", "A thicker shell relieves the infill",
                                   "More perimeters carry a larger part of the load and reduce the strain that loads the infill (the shell is much stiffer than the infill).",
                                   base_reason, g, prio + 1, {"walls": spec.walls + 2}))
            continue
        if region.startswith(("perimeter", "perimetr")):
            if dom == 1 or dom == 3:          # across the line / shear in the plane – the joint of neighbouring lines
                axes = _best_axes(results, g, spec.build_dir)
                items.append(_item(part, "build_dir", "Across the perimeter lines – more perimeters will help little",
                                   f"{COMPONENTS[dom]} governs (across lines / in-plane shear) – it is carried by the bond between adjacent lines. Rotate the part (build direction) so the stress runs along the lines (“Rank build directions”), or strengthen the geometry; perimeters only as a supplement.",
                                   base_reason, g, prio, {"build_dir": list(AXIS_DIRS[axes[0]])} if axes else {}))
            else:
                items.append(_item(part, "walls", "Governed by perimeter along the line – more perimeters",
                                   f"Stress along the perimeter line (σ11) governs; {spec.walls} perimeters × {_fmt(spec.line_width)} mm = {_fmt(spec.wall_thickness)} mm. Add perimeters (e.g. {spec.walls + 2}) – the load-bearing wall is then thicker and the infill behind it supports it less; verify the benefit with a variant.", base_reason, g, prio,
                                   {"walls": spec.walls + 2}))
            continue
        # shell without a line = upper/lower full layers (or a full extrusion)
        items.append(_item(part, "layers", "Governed by top/bottom solid layers – more layers",
                           f"A solid layer fails (top {spec.top_layers} / bottom {spec.bottom_layers} × {_fmt(spec.layer_height)} mm). Add top/bottom layers (e.g. +2), or perimeters.", base_reason, g, prio,
                           {"top_layers": spec.top_layers + 2, "bottom_layers": spec.bottom_layers + 2}))
    # switched-off full infill of thin places, while the part has thin places
    if results is not None and results.mesh is not None:
        from .region import thin_core
        mesh = results.mesh
        sh = mesh.cell_data.get("print_shell")
        for pi, pname in enumerate(mesh.part_names):
            spec = study.printing.spec(pname)
            if sh is None or not spec.enabled or getattr(spec, "solid_thin", True):
                continue
            el = np.nonzero((mesh.tet_part == pi) & (sh >= 0))[0]
            if not len(el):
                continue
            th, _ = thin_core(mesh, el, spec, sh[el])
            if th.any():
                g = governing(results).get(pname, {})
                items.append(_item(pname, "solid_thin", "Thin regions computed with sparse infill",
                                   f"{100 * th.mean():.0f} % of the part's elements are in regions narrower than {_fmt(spec.thin_limit)} mm (after subtracting perimeters) where sparse infill does not fit – the slicer fills them with solid infill. Enable “solid infill in thin regions”.",
                                   "solid_thin = false", g, 2, {"solid_thin": True}))
    items.sort(key=lambda d: (d["part"], d["priority"]))
    return items


def rank_build_directions(study, workdir: str | Path, part_names: list[str] | None = None,
                          directions: list[str] | None = None, study_dir=None, mesh=None, parts=None,
                          on_log: Callable[[str], None] | None = None, cancel=None,
                          on_progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Computes the build directions (default all 6 axes ±X/±Y/±Z) as variants on one mesh for the parts part_names (None =
    all printed) and sorts them by the min. safety of the printing. Returns {ranking: [{direction, build_dir, ok,
    min_safety, region, part, step, max_disp, mass_g, current, error}], best, current, run: VariantsRun}.
    """
    from .region import print_parts
    from .variants import BASE_NAME, run_variants
    dirs = list(directions) if directions else list(AXIS_DIRS)
    from .sweep import norm_dir
    dirs = [norm_dir(d) for d in dirs]
    if parts is None:
        from ..geometry import legacy_step_names, read_step
        parts = read_step(study.geometry_path(study_dir), legacy_names=legacy_step_names(study))
    names = [p.name for p in parts]
    targets = print_parts(study, part_names if part_names else names)
    if not targets:
        raise ValueError("No printed part (material with E_z) to compare build directions")
    vs = [PrintVariant(f"build {d}", {p: {"build_dir": list(AXIS_DIRS[d])} for p in targets}, "build direction")
          for d in dirs]
    vr = run_variants(study, workdir, study_dir, include_base=True, mesh=mesh, parts=parts, on_log=on_log,
                      cancel=cancel, on_progress=on_progress, keep_results=False, keep_files=False, variants=vs)
    cur = {p: np.asarray(study.printing.spec(p).build_dir, float) for p in targets}
    rows = []
    for d, v in zip(dirs, vs):
        o = vr.outcome(v.name)
        e = np.asarray(AXIS_DIRS[d])
        is_cur = all(np.allclose(c / np.linalg.norm(c), e) for c in cur.values())
        g = o.governing or {}
        rows.append({"direction": d, "build_dir": list(AXIS_DIRS[d]), "ok": o.ok, "error": o.error,
                     "min_safety": o.min_safety, "region": g.get("region"), "part": g.get("part"),
                     "step": g.get("step"), "max_disp": o.max_disp, "mass_g": o.mass_g, "current": is_cur})
    rows.sort(key=lambda r: (not r["ok"], -(r["min_safety"] or 0.0)))
    base = vr.outcome(BASE_NAME)
    return {"ranking": rows, "best": rows[0]["direction"] if rows and rows[0]["ok"] else None,
            "current": {"min_safety": base.min_safety if base else None,
                        "build_dir": {p: c.tolist() for p, c in cur.items()}},
            "parts": targets, "run": vr}
