"""Check of the settings of the dynamics before the calculation (specification F8, 5a.3, decision 10, 4.7).

The base excitation is computed relatively (p_j = −Γ_j·a_b, Γ from ccx = φᵀ·M·r for the displacement of the whole model): it holds when
all constraints holding the excited direction move with the base. A plane of symmetry in the excited direction would be "excited" by that –
in ccx *BASE MOTION moreover the ROT nodes of the rigid bodies (verified scratch/f8_review/base_rot.py) → checked here.
"""
from __future__ import annotations

import numpy as np

from .model import DynamicsSettings

DAMPING_KINDS = ("modal", "rayleigh", "structural", "per_material")


def _dirs(study) -> set[int]:
    """Excited global directions of the base (0, 1, 2) – according to the main component of the direction."""
    ds = study.dynamics
    out = set()

    def add(dof, direction):
        if direction is not None:
            v = np.abs(np.asarray(direction, float))
            if v.max() > 0:
                for k in np.nonzero(v > 1e-9 * v.max())[0]:
                    out.add(int(k))
        elif dof:
            out.add(int(dof) - 1)
    for spec in (ds.harmonic, ds.transient):
        if spec.enabled:
            for e in spec.excitation:
                if e.kind in ("base_acc", "base_disp"):
                    add(e.dof, e.direction)
    for spec in (ds.random, ds.shock):
        if spec.enabled:
            add(spec.dof, spec.direction)
    return out


def check(study, part_names=None) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    ds: DynamicsSettings = study.dynamics
    if not ds.enabled:
        return errors, warnings
    if int(ds.modes) < 1:
        errors.append("Dynamics: the number of modes must be ≥ 1")
    if ds.f_max is not None and ds.f_max <= 0:
        errors.append("Dynamics: the upper frequency limit must be positive")
    excl = study.excluded_parts()
    for p in part_names or []:
        if p in excl:
            continue
        try:
            if not study.material_of(p).rho > 0:
                errors.append(f"Dynamics: the material of part '{p}' has no density ρ")
        except KeyError:
            pass
    d = ds.damping
    if d.kind not in DAMPING_KINDS:
        errors.append(f"Dynamics: unknown damping type '{d.kind}'")
    elif d.kind == "rayleigh" and not (d.f1 and d.f2 and d.f2 > d.f1 > 0):
        errors.append("Dynamics: Rayleigh damping needs two frequencies 0 < f1 < f2")
    elif d.kind == "structural" and not ((d.eta or 2 * d.zeta) > 0):
        errors.append("Dynamics: structural damping needs a loss factor η > 0")
    elif d.kind == "structural" and (ds.transient.enabled or ds.random.enabled or ds.shock.enabled):
        warnings.append("Dynamics: hysteretic (structural) damping applies only to the harmonic response – transient, PSD and shock use ζ = η/2")
    if not 0 <= float(d.zeta) < 1:
        errors.append("Dynamics: the damping ratio ζ must be 0 ≤ ζ < 1")
    if d.kind == "modal" and float(d.zeta) == 0 and not d.table:
        warnings.append("Dynamics: zero damping – the response at resonance is infinite")
    warnings.append("Dynamics: damping is an uncertain input (sensitivity ±50 %) – measurement recommended (impact hammer, decrement)")
    if not ds.active:
        warnings.append("Dynamics: no response type is enabled (harmonic, transient, random, shock) – only the modal basis will be computed")
    cases = [lc.name for lc in study.load_cases]
    h = ds.harmonic
    if h.enabled:
        if not (0 <= h.f_min < h.f_max):
            errors.append("Harmonic response: 0 ≤ f_min < f_max must hold")
        if not h.excitation:
            errors.append("Harmonic response: excitation is missing")
    t = ds.transient
    if t.enabled:
        if not (t.t_end > 0 and 0 < t.dt < t.t_end):
            errors.append("Transient response: 0 < dt < t_end must hold")
        elif t.t_end / t.dt > 2e6:
            errors.append("Transient response: too many steps (t_end/dt > 2·10⁶)")
        if not t.excitation:
            errors.append("Transient response: excitation is missing")
        if t.method == "direct":
            warnings.append("Transient response by direct integration (ccx *DYNAMIC) is ~25× slower than modal")
    for spec, what in ((h, "Harmonic response"), (t, "Transient response")):
        if not spec.enabled:
            continue
        for e in spec.excitation:
            if e.kind not in ("force", "base_acc", "base_disp"):
                errors.append(f"{what}: unknown excitation type '{e.kind}'")
            elif e.kind == "force" and e.case not in cases:
                errors.append(f"{what}: force excitation – load case '{e.case}' does not exist")
            elif e.kind != "force" and e.direction is None and e.dof not in (1, 2, 3):
                errors.append(f"{what}: the base excitation direction must be 1, 2 or 3 (x, y, z)")
            tab = np.asarray(e.table, float) if e.table else None
            if tab is not None and (tab.ndim != 2 or tab.shape[1] != 2 or np.any(np.diff(tab[:, 0]) <= 0)):
                errors.append(f"{what}: the history table must have pairs [x, value] with increasing x")
    r = ds.random
    if r.enabled:
        a = np.asarray(r.asd, float) if r.asd else np.zeros((0, 2))
        if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] != 2 or np.any(np.diff(a[:, 0]) <= 0) or np.any(a <= 0):
            errors.append("Random vibration: ASD needs ≥ 2 points [f Hz, g²/Hz] with increasing frequency and positive values")
        if r.method not in ("nb", "dirlik", "tb"):
            errors.append(f"Random vibration: unknown spectral fatigue method '{r.method}'")
        if not r.duration_h > 0:
            errors.append("Random vibration: the test duration must be positive")
    s = ds.shock
    if s.enabled:
        if s.pulse not in ("halfsine", "sawtooth", "trapezoid", "srs_table"):
            errors.append(f"Shock: unknown pulse '{s.pulse}'")
        elif s.pulse == "srs_table" and len(s.srs_table) < 2:
            errors.append("Shock: the response spectrum needs ≥ 2 points [f Hz, g]")
        elif s.pulse != "srs_table" and not (s.A > 0 and s.tau > 0):
            errors.append("Shock: amplitude A and duration τ must be positive")
        if s.combine not in ("srss", "cqc"):
            errors.append(f"Shock: unknown mode combination '{s.combine}'")
    pr = ds.prestress
    if pr.enabled and pr.case is not None and pr.case not in cases:
        errors.append(f"Prestressed modal analysis: load case '{pr.case}' does not exist")
    if pr.enabled and pr.rotating and not pr.rpm_list:
        errors.append("Campbell diagram: enter the speeds")
    # base excitation × constraints
    dirs = _dirs(study)
    if dirs:
        if not study.constraints:
            errors.append("Base excitation: the model has no supports (base) – a free body cannot be excited through the base")
        names = "xyz"
        for c in study.constraints:
            if c.kind == "displacement":
                held = [k for k, v in enumerate((c.ux, c.uy, c.uz)) if v is not None]
                if len(held) < 3 and set(held) & dirs:
                    warnings.append(f"Base excitation in direction {', '.join((names[k] for k in sorted(set(held) & dirs)))}: support '{c.selection}' fixes only some components – if it is a symmetry plane, the excitation must not move it (the result would be wrong); use the full model")
    # symmetry → the antisymmetric modes are missing
    sliding = [c.selection for c in study.constraints if c.kind == "displacement"
               and sum(v is not None for v in (c.ux, c.uy, c.uz)) < 3]
    if sliding:
        warnings.append("Dynamics: supports with only some components (" + ", ".join(sliding[:3]) + ") – if symmetry planes are involved, the antisymmetric modes are missing; use the whole model or calculate the antisymmetric boundary conditions too")
    if study.is_nonlinear():
        errors.append("Dynamics with contact / nonlinear analysis: modal superposition does not apply – disable contacts or dynamics (implicit dynamics with contact ccx *DYNAMIC is a separate analysis)")
    if study.twod.enabled or study.shell.enabled:
        errors.append("Dynamics: 2D section analysis and shells are not yet supported (only 3D bodies and members)")
    return errors, warnings


def memory_check(study, mesh) -> list[str]:
    """Estimate of the memory of ccx for natural frequencies (ARPACK ~ 1.5× the statics) against the free RAM (decision 10)."""
    from ..ccx.parallel import ccx_memory_gb
    from .. import config
    ndof = 3 * len(mesh.nodes)
    need = 1.5 * ccx_memory_gb(ndof)
    try:
        import psutil
        free = psutil.virtual_memory().available / 2**30
    except Exception:  # noqa: BLE001
        return []
    if need > 0.9 * free:
        return [f"Dynamics: estimated ccx memory {need:.1f} GB > free RAM {free:.1f} GB – coarsen the mesh or use a 2D / shell model"]
    if ndof > config.LARGE_DOF:
        return [f"Dynamics: large model ({ndof} DOF > {config.LARGE_DOF}) – computing the modes may take a long time"]
    return []
