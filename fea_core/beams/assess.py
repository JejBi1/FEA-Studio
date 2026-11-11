"""Assessment of the bars of the frame system from the FE result: EN 1993-1-1 (fea_core.beams.ec3) and Winkler–Bach for arches.

Inputs of the assessment per bar in `Study.frame.checks[bar]` (all optional): Lcr_y, Lcr_z, ky, kz (default k = 1,
length of the bar), L_LT, C1, zg, lt_restrained, sway_y, sway_z, fabrication, grade. Steel: `grade` (S235…S460) or
Re of the material of the bar (fy = Re). Each load state is assessed; result per bar (maximum utilisation and rows).
For arches with R/h < 10 the Winkler–Bach stress (factors K_i, K_o from the outline of the section
in the plane of curvature) is added to the stress from the straight bar – EN 1993-1-1 does not cover curved bars, the row is informative.
"""
from __future__ import annotations


import numpy as np

from .ec3 import MemberForces, MemberSpec, check_member, EC3Error


def _grade(study, member) -> tuple[str, float | None]:
    fr = study.frame
    spec = fr.checks.get(member.name, {})
    if spec.get("grade"):
        return spec["grade"], None
    mat = study.materials.get(member.material or study.default_material or "")
    if mat is not None and mat.Re:
        name = mat.name.upper()
        for g in ("S460", "S420", "S355", "S275", "S235"):
            if g in name.replace(" ", ""):
                return g, None
        return fr.settings.steel_grade, float(mat.Re)
    return fr.settings.steel_grade, None


def member_spec(study, member, length: float) -> MemberSpec:
    c = dict(study.frame.checks.get(member.name, {}))
    grade, fy = _grade(study, member)
    kw = {k: c[k] for k in ("Lcr_y", "Lcr_z", "ky", "kz", "L_LT", "C1", "zg", "lt_restrained", "sway_y", "sway_z",
                            "fabrication", "Mcr", "k_LT") if k in c and c[k] is not None}
    mat = study.materials.get(member.material or study.default_material or "")
    E = float(mat.E) if mat is not None else 210000.0
    G = E / (2 * (1 + float(mat.nu))) if mat is not None else 81000.0
    return MemberSpec(L=length, grade=grade, fy=fy, E=E, G=G, gamma_M0=study.frame.settings.gamma_M0,
                      gamma_M1=study.frame.settings.gamma_M1, **kw)


def curved_factors(sec, R: float) -> dict | None:
    """Winkler–Bach K_i (inner fibre), K_o (outer) for bending in the plane of curvature: the outline of the section in the z direction
    (axis z of the section = radial, up="radial") – from the outline the width along the height is taken (polygon).
    None = not needed (R/h ≥ 10); an unusable outline raises (the caller reports it – dropping the correction silently
    would be non-conservative).
    """
    from .sections import outline
    from .analytic import polygon, curved_beam
    rings = outline(sec)
    P = np.asarray(rings[0] if isinstance(rings, (list, tuple)) and len(rings) and np.ndim(rings[0]) == 2 else rings,
                   float)
    h = float(P[:, 1].max() - P[:, 1].min())
    if h <= 0 or R / h >= 10:
        return None
    prof = polygon(P[:, [0, 1]])
    r = curved_beam(prof, M=1.0, R=R)
    return {"K_i": float(r.K_i), "K_o": float(r.K_o), "R_over_h": R / h}


def assess(study, results) -> dict:
    """Assessment of all bars in all states: {bar: {"utilization", "governing", "case", "rows", "warnings",
    "class", "curved"}} + {"_max": …}.
    """
    from .fem import build
    fe = build(study)
    out: dict = {}
    for mname, m in study.frame.members.items():
        sec = study.frame.sections[m.section].resolve()
        segs = fe.member_segs[mname]
        Lm = sum(fe.segs[i].L for i in segs)
        spec = member_spec(study, m, Lm)
        best = None
        warns: list[str] = []
        for st in results.steps:
            fr = st.frame or {}
            md = fr.get("members", {}).get(mname)
            if not md:
                continue
            forces = MemberForces(np.asarray(md["s"]), md["N"], md["Vy"], md["Vz"], md["T"], md["My"], md["Mz"])
            try:
                chk = check_member(sec, forces, spec)
            except EC3Error as e:
                warns.append(f"{st.name}: {e}")
                continue
            if best is None or chk.utilization > best[0].utilization:
                best = (chk, st.name)
        rec = {"length_mm": Lm, "section": sec.name, "warnings": warns}
        if best is not None:
            chk, case = best
            rec.update(utilization=chk.utilization, governing=chk.governing, case=case, x_crit=chk.x_crit,
                       section_class=chk.section_class, fy=chk.fy, grade=chk.grade,
                       rows=[{"name": r.name, "clause": r.clause, "effect": r.effect, "resistance": r.resistance,
                              "utilization": r.utilization, "unit": r.unit, "note": r.note, "x": r.x} for r in chk.rows],
                       warnings=warns + list(chk.warnings))
        if m.through is not None:
            from .fem import _arc
            fr_ = study.frame
            _, R, *_ = _arc(fr_.nodes[m.start].xyz, m.through, fr_.nodes[m.end].xyz)
            try:
                cf = curved_factors(sec, R)
            except Exception as e:  # noqa: BLE001 – unusual outline: reported, the member is flagged
                cf = None
                rec["curved"] = {"error": f"{type(e).__name__}: {e}"}
                rec["warnings"].append(
                    f"Curved member (R = {R:.4g} mm): the Winkler–Bach correction could not be computed ({e}). "
                    f"The stress at the inner fiber of a tightly curved member can be considerably higher than the "
                    f"straight-member assessment shows – this result is NOT conservative; check the member by hand or with a 3D model")
            if cf is not None:
                rec["curved"] = cf
                rec["warnings"].append(f"Curved member R/h = {cf['R_over_h']:.2g}: bending stress in the plane of curvature at the inner fiber is K_i = {cf['K_i']:.3g}× greater than for a straight member (Winkler–Bach, informative – EN 1993-1-1 does not cover curved members)")
        out[mname] = rec
    us = [(v.get("utilization", 0.0), k) for k, v in out.items()]
    out["_max"] = max(us) if us else (0.0, None)
    return out
