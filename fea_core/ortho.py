"""General orthotropic material (outside 3D printing): `Material.ortho` = {E1, E2, E3, nu12, nu13, nu23, G12, G13, G23
[, alpha1, alpha2, alpha3]} in the axes of the material, `Material.ortho_csys` = name of the coordinate system of the study (Cartesian:
axes 1, 2, 3 = x, y, z of the system; cylindrical: 1 = radial, 2 = tangential, 3 = axial; None = global axes).

ccx: *ELASTIC, TYPE=ENGINEERING CONSTANTS (ν_ij: ε_j = −ν_ij·σ_i/E_i) + *ORIENTATION with *SOLID SECTION;
SYSTEM=CYLINDRICAL = points a, b on the axis, X' radially, Y' tangentially, Z' from a to b (ccx 2.23 manual, 7.103). Thermal
expansion *EXPANSION, TYPE=ORTHO (otherwise isotropic alpha). Stresses in the .frd are in global axes (as for printing).
The NGSolve backend and MYSTRAN cannot do it yet (→ ccx / error).
"""
from __future__ import annotations

import numpy as np

KEYS = ("E1", "E2", "E3", "nu12", "nu13", "nu23", "G12", "G13", "G23")


def compliance(o: dict) -> np.ndarray:
    """Compliance matrix 6×6 (Voigt: 11 22 33 12 13 23, shear engineering)."""
    E1, E2, E3 = (float(o[k]) for k in ("E1", "E2", "E3"))
    n12, n13, n23 = (float(o[k]) for k in ("nu12", "nu13", "nu23"))
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
    S[0, 1] = S[1, 0] = -n12 / E1
    S[0, 2] = S[2, 0] = -n13 / E1
    S[1, 2] = S[2, 1] = -n23 / E2
    S[3, 3], S[4, 4], S[5, 5] = 1 / float(o["G12"]), 1 / float(o["G13"]), 1 / float(o["G23"])
    return S


def check(study, part_names) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    used = {study.part_materials.get(p, study.default_material) for p in part_names
            if p not in study.excluded_parts()}
    for name, m in study.materials.items():
        o = getattr(m, "ortho", None)
        if not o:
            continue
        what = f"Orthotropic material '{name}'"
        miss = [k for k in KEYS if o.get(k) is None]
        if miss:
            errors.append(f"{what}: missing {', '.join(miss)}")
            continue
        try:
            vals = [float(o[k]) for k in KEYS]
        except (TypeError, ValueError):
            errors.append(f"{what}: constants must be numbers")
            continue
        if min(vals[0:3] + vals[6:9]) <= 0:
            errors.append(f"{what}: moduli E and G must be positive")
            continue
        if np.linalg.eigvalsh(compliance(o)).min() <= 0:
            errors.append(f"{what}: constants are not physically admissible (compliance matrix is not positive definite – check ν: ν_ij² < E_i/E_j)")
        if m.is_print:
            errors.append(f"{what}: a material cannot be both orthotropic and printed (E_z) – use the 3D printing settings for printing")
        cs = getattr(m, "ortho_csys", None)
        if cs and cs not in study.coordinate_systems:
            errors.append(f"{what}: coordinate system '{cs}' does not exist")
        if name in used:
            warnings.append(f"{what}: the safety factor against Re / von Mises does not apply to an orthotropic material – assess stress in the material axes against its strengths")
    return errors, warnings


def axes(study, m) -> tuple[str, list[float]]:
    """(SYSTEM, 6 numbers of the row *ORIENTATION) for the axes of the material."""
    from .csys import matrix
    cs = study.coordinate_systems.get(getattr(m, "ortho_csys", None) or "")
    if cs is None:
        return "RECTANGULAR", [1.0, 0, 0, 0, 1.0, 0]
    if cs.kind == "cylindrical":
        o = np.asarray(cs.origin, float)
        z = np.asarray(cs.z_axis, float)
        z = z / np.linalg.norm(z)
        return "CYLINDRICAL", [*o.tolist(), *(o + z).tolist()]
    R = matrix(cs)
    return "RECTANGULAR", [*R[0].tolist(), *R[1].tolist()]


def material_lines(m, fmt, thermal: bool, t_ref: float) -> str:
    """*ELASTIC (ENGINEERING CONSTANTS) and *EXPANSION of the orthotropic material."""
    o = m.ortho
    v = [float(o[k]) for k in KEYS]
    s = "*ELASTIC, TYPE=ENGINEERING CONSTANTS\n" + ", ".join(fmt(x) for x in v[:8]) + f",\n{fmt(v[8])}, 0.\n"
    if thermal:
        a = [o.get(f"alpha{i}") for i in (1, 2, 3)]
        if all(x is not None for x in a):
            s += f"*EXPANSION, TYPE=ORTHO, ZERO={fmt(t_ref)}\n" + ", ".join(fmt(float(x)) for x in a) + "\n"
        else:
            s += f"*EXPANSION, ZERO={fmt(t_ref)}\n{fmt(m.alpha or 0.0)}\n"
    return s
