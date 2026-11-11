"""Report – cyclic symmetry (fea_core.cyclic): sector, number of sectors, axis, cut faces and equations; per state the forces
transmitted across the cut faces and the full-structure axial reaction; natural frequencies by nodal diameter; pictures of the
results expanded to the full 360° model."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .word import num, vec


def has_cyclic(study, res) -> bool:
    return (getattr(study, "cyclic", None) is not None and study.cyclic.enabled) or \
        (res is not None and bool((res.info or {}).get("cyclic")))


def _render(sess, full, key: str, i: int, mode: tuple | None, img_dir: Path, view: str, warnings) -> Path | None:
    from ..mcp.diagnostics import FIELDS, REVERSED, SIGNED, field_values
    from ..mcp.render import render
    from ..mcp.scenes import result_scene
    old = sess.results
    sess.results = full
    try:
        if mode is not None:
            j, k = mode
            vals = np.linalg.norm(np.asarray(full.modes[j].shapes[k], float), axis=1)
            scene, _ = result_scene(sess, 0, "displacement", vals, "Mode |u|", "–", False, True, None, 12, None, None,
                                    False, False, mode=(j, k))
            name = f"cyclic_mode_{k + 1}.png"
        else:
            label, unit = FIELDS[key]
            scene, _ = result_scene(sess, i, key, field_values(sess, i, key), label, unit, key in REVERSED, True, None, 12,
                                    None, None, False, key in SIGNED)
            name = f"cyclic_{key}_{i}.png"
        img_dir.mkdir(parents=True, exist_ok=True)
        p = img_dir / name
        render(scene, p, view, 1600, 900)
        return p
    except Exception as e:  # noqa: BLE001 – the report continues without the picture
        warnings.append(f"Cyclic symmetry figure (360° model): {e}")
        return None
    finally:
        sess.results = old


def section(D, sess, o, img_dir, images, warnings, sec: str) -> None:
    from ..cyclic.expand import expand_results
    from ..cyclic.modal import table
    study, res = sess.study, sess.results
    spec = study.cyclic
    info = (res.info or {}).get("cyclic") if res is not None else None
    n = int(info["sectors"]) if info else int(spec.sectors)
    D.p("The model is one sector of a rotationally periodic part; the other sectors are identical and the loads and supports are the same on every sector. The two cut faces are meshed periodically (matching nodes after the rotation by 360°/N) and coupled by constraint equations in global coordinates u_left = Rᵀ·u_right; nodes on the axis cannot move perpendicular to it. Reactions and forces are given PER SECTOR; the full-structure axial reaction is N × the sector value. Natural frequencies are computed per nodal diameter (number of waves around the circumference, CalculiX cyclic symmetry).", size=9)
    rows = [("Number of sectors N", str(n)), ("Sector angle", f"{num(360.0 / n)}°"),
            ("Cut faces (left / right)", f"{(info or {}).get('left', spec.left if spec else '')} / {(info or {}).get('right', spec.right if spec else '')}"),
            ("Axis point [mm]", vec((info or {}).get("axis_point", spec.axis_point if spec else [0, 0, 0]))),
            ("Axis direction", vec((info or {}).get("axis_dir", spec.axis_dir if spec else [0, 0, 1])))]
    if info:
        rows += [("Matched node pairs on the cut faces", str(info["cut_pairs"])),
                 ("Nodes on the axis", str(info["axis_nodes"])),
                 ("Constraint equations", str(info["equations"]))]
    D.kv(rows)
    if res is None:
        return
    ax = np.asarray(info["axis_dir"], float) if info else np.array([0.0, 0.0, 1.0])
    trows = []
    for s in res.steps:
        c = (s.equilibrium or {}).get("cyclic") or {}
        tot = np.zeros(3)
        for v in s.reactions.values():
            tot += np.asarray(v, float)
        trows.append([s.name, vec(tot), num(float(n * tot @ ax)), vec(c.get("cut_force", [0, 0, 0])),
                      vec(c.get("cut_moment", [0, 0, 0]))])
    if trows:
        D.table(["case", "Σ reactions per sector [N]", "axial reaction 360° [N]", "force across the cuts [N]",
                 "moment across the cuts [N·mm]"], trows, caption="Cyclic symmetry – reactions and cut forces per sector",
                font=8)
    for m in res.modes:
        if getattr(m, "nodal_diameters", None) is None:
            continue
        rows = [[r["mode"], r["nodal_diameter"], num(r["frequency_Hz"]), r["multiplicity"]] for r in table(m)]
        D.table(["mode", "nodal diameter", "f [Hz]", "modes of the full structure"], rows,
                caption="Natural frequencies by nodal diameter (cyclic symmetry)", font=8)
        D.p("A frequency with nodal diameter 0 < ND < N/2 belongs to a pair of equal modes of the full structure (standing waves shifted by a quarter wave); ND = 0 and ND = N/2 are single modes.", size=9)
    if not o.images:
        return
    full = expand_results(res)
    if full is None:
        return
    for i, s in enumerate(full.steps[:2]):
        p = _render(sess, full, "mises", i, None, Path(img_dir), o.image_view, warnings)
        if p:
            images.append(str(p))
            D.image(p, f"Full 360° model (expanded sector) – {s.name}: equivalent von Mises stress")
    for j, m in enumerate(full.modes):
        if getattr(m, "nodal_diameters", None) is None:
            continue
        for k in range(min(3, len(m.values))):
            if m.values[k] <= 0:
                continue
            p = _render(sess, full, "displacement", 0, (j, k), Path(img_dir), o.image_view, warnings)
            if p:
                images.append(str(p))
                D.image(p, f"Mode {k + 1}: nodal diameter {m.nodal_diameters[k]}, f = {num(m.values[k])} Hz (360° model)")
