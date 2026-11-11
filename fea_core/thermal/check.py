"""Check of the readiness of the thermal states and thermal loads (called by solve.check_study)."""
from __future__ import annotations

import numpy as np

from ..study import Study, ThermalLoad
from ..mesh import Mesh
from .model import (FixedTemperature, Convection, Radiation, CavityRadiation, ContactConductance, HeatGeneration,
                    ThermalCase, MAX_CAVITIES, is_heat_sink, is_cavity, bc_selections, cavity_key, ABSOLUTE_ZERO)


def _material(study: Study, part: str):
    try:
        return study.material_of(part)
    except KeyError:
        return None


def check(study: Study, part_names: list[str]) -> tuple[list[str], list[str], set[str]]:
    """(errors, warnings, selections used)."""
    errors, warnings, used = [], [], set()
    th = study.thermal
    excl = study.excluded_parts()
    meshed = [p for p in part_names if p not in excl]
    names = [c.name for c in th.cases]
    for n in {n for n in names if names.count(n) > 1}:
        errors.append(f"Thermal case '{n}' is defined more than once")
    if th.cases:
        for p in meshed:
            try:
                m = study.material_of(p)
            except KeyError:
                continue                                  # reported by check_study
            if not m.k or m.k <= 0:
                errors.append(f"Material '{m.name}' (part '{p}') has no thermal conductivity k [W/(m·K)]")
            if any(c.transient for c in th.cases) and (not m.cp or m.cp <= 0 or m.rho <= 0):
                errors.append(f"Material '{m.name}' (part '{p}') has no density or specific heat c_p (required for a transient case)")
    for c in th.cases:
        if not c.loads:
            errors.append(f"Thermal case '{c.name}' has no boundary condition")
            continue
        # a closed cavity (closed=True) does not remove heat to the surroundings – the steady state requires a heat sink (is_heat_sink)
        if not c.transient and not any(is_heat_sink(bc) for bc in c.loads):
            errors.append(f"Thermal case '{c.name}': the steady state has no way to dissipate heat – specify a prescribed temperature, heat transfer (convection) or radiation to the surroundings")
        if c.transient:
            if c.time <= 0:
                errors.append(f"Thermal case '{c.name}': transient duration must be positive")
            if c.dt < 0 or c.dt > c.time:
                errors.append(f"Thermal case '{c.name}': time step must be 0 (automatic) up to the case duration")
            if c.outputs < 1:
                errors.append(f"Thermal case '{c.name}': number of frames must be at least 1")
            elif c.time > 0 and c.time / c.step_size() > 20000:
                warnings.append(f"Thermal case '{c.name}': {c.time / c.step_size():.0f} time steps – the analysis will take long")
        has_cavity = False
        has_ambient_rad = False
        for bc in c.loads:
            sels = bc_selections(bc)
            sel = ", ".join(sels)
            used.update(sels)
            if isinstance(bc, ContactConductance):
                if bc.master:
                    used.add(bc.master)
                if not bc.conductance > 0:
                    errors.append(f"Thermal case '{c.name}': contact conductance at '{sel}' must be positive")
            if isinstance(bc, Convection) and bc.h <= 0:
                errors.append(f"Thermal case '{c.name}': heat transfer coefficient on '{sel}' must be positive")
            if isinstance(bc, (Radiation, CavityRadiation)):
                if not (0 < bc.emissivity <= 1):
                    errors.append(f"Thermal case '{c.name}': emissivity on '{sel}' must be in the interval (0; 1]")
                if is_cavity(bc):
                    has_cavity = True
                else:
                    has_ambient_rad = True
                    if bc.ambient <= ABSOLUTE_ZERO:
                        errors.append(f"Thermal case '{c.name}': ambient temperature on '{sel}' is below absolute zero")
            if isinstance(bc, Convection) and bc.ambient <= ABSOLUTE_ZERO:
                errors.append(f"Thermal case '{c.name}': ambient temperature on '{sel}' is below absolute zero")
            if isinstance(bc, FixedTemperature) and bc.value <= ABSOLUTE_ZERO:
                errors.append(f"Thermal case '{c.name}': temperature on '{sel}' is below absolute zero")
            if isinstance(bc, HeatGeneration):
                if bc.part not in part_names:
                    errors.append(f"Thermal case '{c.name}': part '{bc.part}' (heat generation) is not in the geometry")
                elif bc.part in excl:
                    errors.append(f"Thermal case '{c.name}': part '{bc.part}' (heat generation) is not meshed")
        if len({cavity_key(bc) for bc in c.loads if isinstance(bc, (Radiation, CavityRadiation)) and is_cavity(bc)}) > MAX_CAVITIES:
            errors.append(f"Thermal case '{c.name}': at most {MAX_CAVITIES} cavities (CalculiX distinguishes cavity names by three characters only)")
        if has_cavity:
            warnings.append(f"Thermal case '{c.name}': cavity radiation between faces (mutual irradiation), the problem is nonlinear (ccx iterations)")
        elif has_ambient_rad:
            warnings.append(f"Thermal case '{c.name}': radiation to ambient only (no surface-to-surface radiation), the problem is nonlinear (ccx iterations)")
    if th.conductances:                               # contact conductances of interfaces (fea_core.thermal.interfaces)
        from .interfaces import check_definitions
        e, w = check_definitions(study, part_names)
        errors += e
        warnings += w
        for c in th.conductances:
            used.update(s for s in (c.selection, c.master) if s in study.selections)
        if not th.cases:
            warnings.append("Contact conductance is defined but there is no thermal case – it has no effect")
    # thermal loads of the mechanical states
    if study.has_thermal_loads():
        for p in meshed:
            try:
                m = study.material_of(p)
            except KeyError:
                continue
            if not m.alpha or m.alpha <= 0:
                errors.append(f"Material '{m.name}' (part '{p}') has no coefficient of thermal expansion α [1/K] (thermal load)")
    for lc in study.load_cases:
        tl = [l for l in lc.loads if isinstance(l, ThermalLoad)]
        if sum(1 for l in tl if l.case) > 1:
            errors.append(f"Case '{lc.name}': at most one temperature field from a thermal case")
        for l in tl:
            if l.case:
                if l.case not in names:
                    errors.append(f"Case '{lc.name}': thermal case '{l.case}' does not exist")
                    continue
                c = th.case(l.case)
                if l.time is not None and c.transient and not (0 <= l.time <= c.time):
                    errors.append(f"Case '{lc.name}': time {l.time:g} s is outside transient case '{c.name}' (0–{c.time:g} s)")
            elif l.value is None:
                errors.append(f"Case '{lc.name}': thermal load has neither a thermal case nor a temperature")
            for p in l.parts:
                if p not in part_names:
                    errors.append(f"Case '{lc.name}': part '{p}' (temperature) is not in the geometry")
    return errors, warnings, used


def floating_parts(study: Study, mesh: Mesh, case: ThermalCase) -> list[str]:
    """Parts that in the given thermal state have no path to a heat sink (prescribed temperature, convection or radiation to the surroundings)
    – a steady state with heat has no solution (the temperature grows to infinity). Returns the list of names of the floating parts.
    Connections: conforming mesh (shared nodes), *TIE and contacts; bolts are not considered (they do not conduct heat).
    """
    if case.transient:
        return []
    from ..mesh import part_union
    owner, find, union = part_union(mesh)
    from ..ties import tied_part_pairs
    for a, b in tied_part_pairs(mesh):                # *TIE also connects thermally (thermal.inp.thermal_ties)
        union(a, b)
    for r in mesh.report.get("contacts", []):         # contact = perfect thermal contact (specification F8, 4.7)
        if r.get("master"):
            ps = set(np.unique(owner[np.unique(mesh.surfaces[r["slave"]])]).tolist())
            ps |= set(np.unique(owner[np.unique(mesh.surfaces[r["master"]])]).tolist())
            ps = sorted(ps - {-1})
            for q in ps[1:]:
                union(ps[0], q)
    # faces of the same cavity (radiation) and faces of the contact conductance (GAP CONDUCTANCE) connect the parts thermally
    cavities: dict[str, set[int]] = {}
    for bc in [*case.loads, *study.thermal.conductances]:
        if isinstance(bc, ContactConductance) and bc.selection in mesh.surfaces and bc.master in mesh.surfaces:
            ps = {int(q) for sel in (bc.selection, bc.master) for q in np.unique(owner[mesh.surface_nodes(sel)])} - {-1}
            ps = sorted(ps)
            for q in ps[1:]:
                union(ps[0], q)
        elif isinstance(bc, (Radiation, CavityRadiation)) and is_cavity(bc):
            grp = cavities.setdefault(cavity_key(bc), set())
            for sel in bc_selections(bc):
                if sel in mesh.surfaces:
                    grp |= {int(q) for q in np.unique(owner[mesh.surface_nodes(sel)])} - {-1}
    for grp in cavities.values():
        ps = sorted(grp)
        for q in ps[1:]:
            union(ps[0], q)
    sinks = set()
    for bc in case.loads:
        if not is_heat_sink(bc):
            continue
        for sel in bc_selections(bc):
            if sel in mesh.surfaces:
                for q in np.unique(owner[mesh.surface_nodes(sel)]):
                    if q >= 0:
                        sinks.add(find(int(q)))
    return [p for pi, p in enumerate(mesh.part_names) if find(pi) not in sinks]
