"""Check of the 3D printing settings before the calculation (complements solve.check_printing): ranges of numbers, strengths and positive
definiteness of the orthotropic stiffness (an incorrect ν_z with E_z ≪ E would otherwise fail only in ccx without a clear message).
"""
from __future__ import annotations

import numpy as np

from .infill import HOLLOW, core, line_solid, solid


def _posdef(o) -> bool:
    try:
        return bool(np.all(np.linalg.eigvalsh(o.stiffness()) > 0))
    except (np.linalg.LinAlgError, ZeroDivisionError, ValueError):
        return False


def check_settings(study, printed: list[str]) -> tuple[list[str], list[str]]:
    """(errors, warnings) for the printed parts `printed`."""
    from .props import family_of, props_of
    errors, warnings = [], []
    ps = study.printing
    if not printed:
        return errors, warnings
    if not (1 <= int(ps.bins) <= 50):
        errors.append(f"3D printing: the number of shell fraction levels (bins) must be 1–50, is {ps.bins}")
    if not (4 <= int(ps.samples) <= 512):
        errors.append(f"3D printing: the number of sampling points per element (samples) must be 4–512, is {ps.samples}")
    if not (1 <= int(ps.line_bins) <= 180):
        errors.append(f"3D printing: the number of line direction classes (line_bins) must be 1–180, is {ps.line_bins}")
    checked: set[str] = set()
    for p in printed:
        sp = ps.spec(p)
        m = study.material_of(p)
        if sp.enabled:
            bad = [n for n, v in (("perimeters", sp.walls), ("top layers", sp.top_layers),
                                  ("bottom layers", sp.bottom_layers)) if int(v) != v or v < 0]
            if bad:
                errors.append(f"3D printing '{p}': {', '.join(bad)} must be an integer ≥ 0")
            if sp.walls == 0 and sp.top_layers == 0 and sp.bottom_layers == 0:
                warnings.append(f"3D printing '{p}': no perimeters and solid layers – the entire part is infill")
            if 0.0 <= sp.infill < HOLLOW:
                warnings.append(f"3D printing '{p}': infill {100 * sp.infill:g} % = hollow part – only the shell (perimeters and solid layers) carries load, the cavity has negligible stiffness and is not assessed")
            if sp.pattern == "lightning" and sp.infill >= HOLLOW:
                warnings.append(f"3D printing '{p}': lightning infill supports only the top layers – the core has almost no stiffness (the iterative solver ngs_hypre then converges slowly; prefer ccx)")
            if sp.line_width > 2.0 or sp.layer_height > 1.0 or sp.layer_height > sp.line_width:
                warnings.append(f"3D printing '{p}': unusual layer {sp.layer_height:g} mm / line {sp.line_width:g} mm – check units (mm)")
        if m.name in checked:
            continue
        checked.add(m.name)
        if not m.E or m.E <= 0 or not m.E_z or m.E_z <= 0:
            errors.append(f"3D printing: material '{m.name}' must have positive moduli E and E_z")
            continue
        if not m.Rm:
            errors.append(f"3D printing: material '{m.name}' has no strength Rm – the failure criterion (Tsai-Wu/Hill) cannot be evaluated; enter Rm (and Rm_z) in the 3D printing or Materials step")
            continue
        if m.nu is None or not (-1.0 < m.nu < 0.5):
            errors.append(f"3D printing: material '{m.name}' has invalid ν = {m.nu}")
            continue
        if not _posdef(solid(m)):
            errors.append(f"3D printing: stiffness of material '{m.name}' is not positive definite – check ν_xz ({m.nu_z}) against E and E_z (ν_xz² < E/E_z·(1 − ν)/2 must hold)")
            continue
        pr = props_of(study, m)
        if not (0 < pr.line_E_ratio <= 1.5 and 0 < pr.line_R_ratio <= 1.5):
            errors.append(f"3D printing: material '{m.name}' – transverse/along line ratios must be in (0; 1.5]")
        elif not _posdef(line_solid(m, pr.line_E_ratio, pr.line_R_ratio)):
            errors.append(f"3D printing: perimeter stiffness of material '{m.name}' with line ratio {pr.line_E_ratio:g} is not positive definite")
        if family_of(m.name) == "PLA" and "PLA" not in m.name.upper():
            warnings.append(f"3D printing: the plastic family of material '{m.name}' was not recognized – creep, lines and S-N are taken as for PLA (indicative); adjust them in the 3D printing step")
    for p in printed:
        sp = ps.spec(p)
        if sp.enabled and sp.infill >= HOLLOW:
            try:
                if not _posdef(core(study.material_of(p), sp.pattern, sp.infill)):
                    errors.append(f"3D printing '{p}': stiffness of infill {sp.pattern} {100 * sp.infill:g} % is not positive definite")
            except (KeyError, ValueError, ZeroDivisionError):
                pass                       # an unknown pattern is reported by solve.check_printing
    return errors, warnings
