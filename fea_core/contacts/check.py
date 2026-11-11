"""Check of the contact pairs and of the nonlinear stepping before the calculation (F8.5, specification 4.2–4.5, 4.7, 6.2)."""
from __future__ import annotations

from .model import CONTACT_TYPES

METHODS = ("surface_to_surface", "mortar", "node_to_surface")


def check(study, part_names=None) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    cs = study.contact_settings
    act = study.active_contacts()
    names = set(part_names or [])
    if study.contacts:
        if cs.method not in METHODS:
            errors.append(f"Contacts: unknown formulation '{cs.method}' ({', '.join(METHODS)})")
        if cs.split not in ("separate", "conformal_split"):
            errors.append(f"Contacts: unknown interface meshing method '{cs.split}'")
        seen = set()
        for c in study.contacts:
            what = f"Contact '{c.name}'"
            if c.name in seen:
                errors.append(f"{what}: the pair name is used twice")
            seen.add(c.name)
            if c.kind not in CONTACT_TYPES:
                errors.append(f"{what}: unknown type '{c.kind}'")
                continue
            for side in ([c.a] if c.kind == "compression_only" and not c.b else [c.a, c.b]):
                if side not in study.selections and names and side not in names:
                    errors.append(f"{what}: '{side}' is neither a part nor a face selection")
                elif side in study.selections and study.selections[side].entity != "faces":
                    errors.append(f"{what}: selection '{side}' must contain faces")
            if c.b and c.a == c.b:
                errors.append(f"{what}: self-contact (same face / part on both sides) is not supported by ccx")
            if c.mu is not None and not c.mu > 0 and c.kind == "frictional":
                errors.append(f"{what}: the friction coefficient μ must be > 0 (no friction = type “Frictionless contact”)")
            for v, lab in ((c.stiffness, "normal stiffness K"), (c.stick_slope, "stick slope λ")):
                if v is not None and not v > 0:
                    errors.append(f"{what}: {lab} must be positive")
            if c.clearance is not None and cs.method == "node_to_surface":
                errors.append(f"{what}: interference / clearance (*CLEARANCE) works only with surface-to-surface or mortar")
            if c.kind == "interference" and c.clearance is None:
                warnings.append(f"{what}: interference without a value – the geometric overlap of the meshes is used (adjusting must not be enabled)")
            if c.kind == "interference" and c.clearance is not None and c.clearance > 0:
                errors.append(f"{what}: interference is entered as a negative radial value (−Δd/2)")
            if c.kind == "no_separation" and cs.method != "surface_to_surface":
                errors.append(f"{what}: no separation (TIED) works only with the surface-to-surface formulation")
            if c.kind == "interference" and c.adjust:
                errors.append(f"{what}: interference must not use adjusting into contact (the interference would vanish)")
            if c.kind == "frictional" and c.mu is None:
                warnings.append(f"{what}: μ from the friction library by purpose ({('slip – μ_min' if c.purpose == 'slip' else 'resistance – μ_max')}); μ is an estimate – compute both limits")
        if act and cs.method == "node_to_surface":
            warnings.append("Contacts: node-to-surface – contact pressure on quadratic elements is unusable (corners up to 3.6×), only the resultant")
        if act and cs.method == "mortar":
            warnings.append("Contacts: mortar – zero pressures may occur at the surface edge (verified by testing), ~2.5× slower")
    if act:
        if study.active_bolts() and study.contact_split() == "separate":
            from .detect import contact_parts
            clamped = {pl["part"] for b in study.active_bolts() for pl in b.plates} | \
                {b.head_part for b in study.active_bolts()} | {b.nut_part for b in study.active_bolts() if b.nut_part}
            hit = sorted(contact_parts(study) & clamped)
            if hit:
                errors.append("Contacts on bolted parts (" + ", ".join(hit[:3]) + ") require meshing the joint with “conforming mesh + split” (a separately meshed part cannot have a bolt imprint)")
        if any(c.kind in ("frictional", "rough") for c in act) and not study.analysis.sequence:
            warnings.append("Frictional contacts: a part held only by friction and loaded by a force may slip – the analysis then does not converge; use displacement control or add weak springs")
        if study.fatigue is not None and study.analysis.branches:
            warnings.append("Fatigue with contact: superposition does not apply and branches have no history – compute fatigue with a step sequence (≥ 2 cycles)")
        if study.twod.enabled or study.shell.enabled:
            errors.append("Contacts are so far available only for 3D bodies (not 2D section or shells)")
        if study.frame.active:
            errors.append("Contacts with a mixed model (members + bodies) are not supported")
    if act and study.active_bolts() and not cs.nlgeom:
        warnings.append("Preloaded bolts with contact without large deformations: ccx evaluates the contact in the deformed position, but the rest of the analysis is linear, so the effect of the joint rotation on the clamping force (P–Δ) is included only by half. Example K6 (M10, preload 19.8 kN, shear 1 kN): the joint rotation is 2.7 % smaller than with large deformations and 3.8 % smaller than the linear solution (Abaqus). Take the displacements and stresses of the joint with an uncertainty of a few percent; bolt forces and utilisation are not affected"
                        )
    if cs.nlgeom:
        if study.active_bolts():
            errors.append("Large deformations (NLGEOM) with bolts are not allowed (the spring connector has fixed directions, the equilibrium is violated by 0.1 % under a large transverse load)")
        if any(c.kind == "cylindrical" for c in study.constraints):
            errors.append("Large deformation (NLGEOM) with a cylindrical support (linear equations from the undeformed geometry) is not allowed – use a rigid remote support")
        if any(c.kind == "remote_support" and c.coupling == "distributing" for c in study.constraints):
            errors.append("Large deformation (NLGEOM) with a distributing remote support is not allowed – use a rigid one")
    an = study.analysis
    if an.kind not in ("linear", "nonlinear"):
        errors.append(f"Analysis type '{an.kind}' does not exist (linear | nonlinear)")
    if an.nonlinear and not an.branches:
        cases = {lc.name for lc in study.load_cases}
        pairs = {c.name: c for c in study.all_contacts()}
        from ..study import RemoteSupport
        rigid = {c.selection for c in study.constraints if isinstance(c, RemoteSupport) and c.coupling == "rigid"}
        if not an.sequence:
            errors.append("The step sequence is empty")
        for s in an.sequence:
            w = f"Step '{s.name}'"
            if s.case is not None and s.case not in cases:
                errors.append(f"{w}: load case '{s.case}' does not exist")
            if s.bolts not in ("preload", "lock", "free"):
                errors.append(f"{w}: bolts '{s.bolts}' (preload | lock | free)")
            for nm, mu in (s.mu_change or {}).items():
                c = pairs.get(nm)
                if c is None:
                    errors.append(f"{w}: friction change on a non-existent pair '{nm}'")
                elif not float(mu) > 0:
                    errors.append(f"{w}: μ must be > 0 (no friction = 1e-4)")
            for nm in s.contacts_off or []:
                if nm not in pairs:
                    errors.append(f"{w}: deactivation of a non-existent pair '{nm}'")
            for mo in s.prescribed:
                if mo.target not in rigid:
                    errors.append(f"{w}: prescribed motion '{mo.target}' – the target must be a rigid remote support")
                if mo.kind not in ("disp", "rot") or int(mo.dof) not in (1, 2, 3):
                    errors.append(f"{w}: motion kind disp|rot and dof 1–3")
            if not (0 < s.dt_min <= s.dt0 <= 1 and s.dt_min <= s.dt_max <= 1):
                errors.append(f"{w}: increment control must satisfy 0 < dt_min ≤ dt0 ≤ 1, dt_max ≤ 1")
            if s.cycles < 1:
                errors.append(f"{w}: number of cycles ≥ 1")
    return errors, warnings


def mesh_errors(study, mesh) -> list[str]:
    """After meshing: initial penetration of a pair that is not an interference (K8) → error with an offer of a solution."""
    out = []
    by = {c.name: c for c in study.all_contacts()}
    for r in mesh.report.get("contacts", []):
        c = by.get(r["pair"])
        if c is None or not r.get("master"):
            continue
        if r.get("state") == "penetration" and c.kind != "interference" and c.clearance is None and not c.adjust:
            out.append(f"Contact '{c.name}': initial penetration {-r['gap_min_mm']:.4g} mm (type “{c.kind}”). If it is a press fit, set the type to Interference; a small penetration from the model can be adjusted (Adjust into contact); otherwise fix the assembly in CAD (modelled thread, position error).")
    return out
