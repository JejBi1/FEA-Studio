"""Input of ccx for heat conduction: the model (mesh, selections, materials) once into model.inc, each thermal state
as a separate job (*INCLUDE + one *STEP) – a transient state thus always starts from the initial temperature.

Units of ccx: mm, N, s, t → power mW (conversions in thermal.model). Transient state: ccx does 2 iterations
(2 factorisations) per time step even for a linear problem → time ~ the number of steps (a small model ~0.1 s per step). Elements C3D10 (ccx in *HEAT TRANSFER
uses only the temperature DOF 11). Bolts (U1) are not written into the thermal problem – only the contacts of the parts conduct heat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import solver_name
from ..ccx.inp import InpMap, write_mesh, write_sets, _fmt, _write_ids
from ..mesh import Mesh
from ..study import Study
from .loads import curved_area, generation_density
from ..ccx import deck_text
from .model import (ThermalCase, FixedTemperature, HeatFlux, HeatFlow, Convection, Radiation,
                    CavityRadiation, ContactConductance, HeatGeneration,
                    K_TO_CCX, FLUX_TO_CCX, POWER_TO_CCX, CP_TO_CCX, STEFAN_BOLTZMANN_CCX, ABSOLUTE_ZERO,
                    MAX_CAVITIES, is_cavity, bc_selections, cavity_key)

MODEL = "model.inc"


@dataclass
class ThermalJob:
    name: str                     # name of the ccx job (case1 …)
    case: str
    increments: int = 1
    frequency: int = 1
    dt: float = 0.0
    fixed_sets: dict[str, str] = field(default_factory=dict)


def conductivity_fraction(study: Study, mesh: Mesh) -> np.ndarray:
    """Share of the full material per element (3D printing: shell + infill, otherwise 1) – the conductivity and the
    heat capacity are multiplied by this share (parallel conduction, INDICATIVE).
    """
    from ..printing import shell_of, element_density
    frac = np.ones(len(mesh.tets))
    if shell_of(mesh) is None:
        return frac
    rho = np.array([study.material_of(p).rho for p in mesh.part_names])[mesh.tet_part]
    ok = rho > 0
    frac[ok] = element_density(study, mesh)[ok] / rho[ok]
    return np.round(frac, 3)


def thermal_contact_conductances(study: Study, mesh: Mesh) -> list[tuple[str, str, float]]:
    """(slave, master, hc_W_m2K) of the contacts with a defined thermal conductance h_c."""
    out: list[tuple[str, str, float]] = []
    seen = set()

    # mapping from the detection of the contacts in the mesh (by 'pair' and 'name')
    contacts_report: dict[str, dict] = {}
    for r in mesh.report.get("contacts", []):
        if r.get("pair"):
            contacts_report[r["pair"]] = r
        if r.get("name"):
            contacts_report[r["name"]] = r

    # 1. Kontakty ze study.contacts s definovanou conductance
    for c in getattr(study, "contacts", []):
        if not getattr(c, "enabled", True):
            continue
        hc = getattr(c, "conductance", None)
        if hc is not None and float(hc) > 0:
            sl, ms = None, None
            if c.name in contacts_report:
                cr = contacts_report[c.name]
                sl, ms = cr.get("slave"), cr.get("master")
            elif c.a in mesh.surfaces and c.b in mesh.surfaces:
                sl, ms = c.a, c.b
            elif c.a in mesh.part_names or c.b in mesh.part_names:
                for r in mesh.report.get("contacts", []):
                    if (r.get("part_a") in (c.a, c.b) and r.get("part_b") in (c.a, c.b)) or r.get("name") == c.name:
                        sl, ms = r.get("slave"), r.get("master")
                        break
            if sl and ms and sl in mesh.surfaces and ms in mesh.surfaces and (sl, ms) not in seen:
                out.append((sl, ms, float(hc)))
                seen.add((sl, ms))

    # 2. Possibly ContactConductance in the loads
    for case in getattr(study.thermal, "cases", []):
        for bc in case.loads:
            if isinstance(bc, ContactConductance) and bc.conductance > 0:
                sl, ms = bc.selection, bc.master
                if sl in contacts_report:
                    cr = contacts_report[sl]
                    sl, ms = cr.get("slave"), cr.get("master")
                if sl and ms and sl in mesh.surfaces and ms in mesh.surfaces and (sl, ms) not in seen:
                    out.append((sl, ms, float(bc.conductance)))
                    seen.add((sl, ms))

    return out


def thermal_ties(mesh: Mesh, conductances: list[tuple[str, str, float]] | None = None) -> list[tuple[str, str, float]]:
    """(slave, master, tolerance) of the faces connected in the thermal problem: *TIE of the separately meshed parts and contact pairs
    with a separate mesh (perfect thermal contact – the pairs with a contact conductance are excluded).
    """
    cond_pairs = set()
    if conductances:
        for sl, ms, _ in conductances:
            cond_pairs.add((sl, ms))
            cond_pairs.add((ms, sl))

    out: list[tuple[str, str, float]] = []
    for t in mesh.report.get("ties", []):
        sl, ms = t["slave"], t["master"]
        if (sl, ms) not in cond_pairs:
            out.append((sl, ms, float(t["tolerance_mm"])))

    for r in mesh.report.get("contacts", []):
        sl, ms = r.get("slave"), r.get("master")
        if ms and not str(sl).startswith("__split") and (sl, ms) not in cond_pairs:
            gap = max(abs(r.get("gap_max_mm", 0.0)), abs(r.get("gap_min_mm", 0.0)))
            out.append((sl, ms, max(1.5 * gap, 0.05 * r.get("element_size_mm", 1.0), 1e-3)))

    return out


def split_equations(mesh: Mesh) -> list[tuple[int, int]]:
    """Pairs (copy, original) of the nodes of the disconnected conforming joints (contacts.detect.split_interface)."""
    if mesh.node_alias is None:
        return []
    split_nodes = set()
    for name, tri in mesh.surfaces.items():
        if name.startswith("__split_") and name.endswith("_b"):
            split_nodes |= set(np.unique(tri).tolist())
    alias = mesh.canonical()
    return [(n, int(alias[n])) for n in sorted(split_nodes) if int(alias[n]) != n]


def write_model(study: Study, mesh: Mesh, path: Path) -> InpMap:
    """Mesh, NSET/SURFACE of all the selections of the thermal states, materials (conductivity, density, capacity)."""
    m = InpMap()
    need_nset, need_surf = set(), set()
    for case in study.thermal.cases:
        for bc in case.loads:
            if isinstance(bc, FixedTemperature):
                need_nset.add(bc.selection)
            elif isinstance(bc, (HeatFlux, HeatFlow, Convection, Radiation, CavityRadiation)):
                need_surf.update(bc_selections(bc))

    # contact pairs with conductivity
    conductances = thermal_contact_conductances(study, mesh)
    for sl, ms, _ in conductances:
        need_surf.add(sl)
        need_surf.add(ms)

    # contact pairs without conductivity (*TIE)
    ties = thermal_ties(mesh, conductances)
    for sl, ms, _ in ties:
        need_surf.add(sl)
        need_surf.add(ms)

    # faces and NSETs
    # names of the sets generated (SEL<i> by the order of the selections of the study, X<k> internal faces of the mesh) – a selection name with a comma,
    # a space or diacritics would break the card of ccx, and two names could merge after conversion to ASCII
    ids = {name: f"SEL{i + 1}" for i, name in enumerate(study.selections)}
    for k, sel in enumerate(sorted((need_nset | need_surf) - ids.keys())):
        ids[sel] = f"X{k + 1}"
    for sel in sorted(need_nset):
        m.nsets[sel] = f"N_{ids[sel]}"
    for sel in sorted(need_surf):
        m.surfaces[sel] = f"S_{ids[sel]}"

    frac = conductivity_fraction(study, mesh)

    with open(path, "w", encoding="ascii", errors="replace") as f:
        f.write("** FEA Studio: heat conduction model (ccx)\n*HEADING\nModel\n")
        write_mesh(f, mesh, m)
        # NSET for the prescribed temperatures
        for sel, name in sorted(m.nsets.items()):
            f.write(f"*NSET, NSET={name}\n")
            _write_ids(f, mesh.surface_nodes(sel) + 1)
        # SURFACE for fluxes and convection
        for sel, name in sorted(m.surfaces.items()):
            el, fno = mesh.element_faces(sel)
            f.write(f"*SURFACE, NAME={name}, TYPE=ELEMENT\n" + "".join(f"{e + 1}, S{k}\n" for e, k in zip(el, fno)))

        # Materials: conductivity, for the transient also the density + capacity (scaled by the infill)
        any_trans = any(c.transient for c in study.thermal.cases)
        for i, name in enumerate(mesh.part_names):
            mat = study.material_of(name)
            p_frac = frac[mesh.tet_part == i]
            mean_frac = float(p_frac.mean()) if len(p_frac) else 1.0
            k_eff = max(mat.k * mean_frac, 1e-6) * K_TO_CCX
            f.write(f"*MATERIAL, NAME=MAT_{i + 1}\n*CONDUCTIVITY\n{_fmt(k_eff)}\n")
            if any_trans:
                rho = mat.rho
                cp = max(mat.cp * mean_frac, 1e-3) * CP_TO_CCX
                f.write(f"*DENSITY\n{_fmt(rho)}\n*SPECIFIC HEAT\n{_fmt(cp)}\n")
            f.write(f"*SOLID SECTION, ELSET=P{i + 1}, MATERIAL=MAT_{i + 1}\n")

        # Connection of separately meshed parts and perfect contacts
        for i, (sl, ms, tol) in enumerate(ties):
            f.write(f"*TIE, NAME=TIE_TH{i}, POSITION TOLERANCE={_fmt(tol)}\n"
                    f"{m.surfaces[sl]}, {m.surfaces[ms]}\n")

        # Thermal resistance of a contact via *SURFACE INTERACTION and *GAP CONDUCTANCE
        for i, (sl, ms, hc) in enumerate(conductances):
            f.write(f"*SURFACE INTERACTION, NAME=INT_TC{i}\n"
                    f"*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=TIED\n1000000.\n"
                    f"*GAP CONDUCTANCE\n"
                    f"{_fmt(hc * FLUX_TO_CCX)}\n"
                    f"*CONTACT PAIR, INTERACTION=INT_TC{i}, TYPE=SURFACE TO SURFACE\n"
                    f"{m.surfaces[sl]}, {m.surfaces[ms]}\n")

        eqs = split_equations(mesh)
        if eqs:                                           # disconnected conforming joint: T of the copy = T of the original
            f.write("** open gaps (contact): the same temperature on both sides\n*EQUATION\n")
            for a, b in eqs:
                f.write(f"2\n{a + 1}, 11, 1., {b + 1}, 11, -1.\n")
        f.write(f"*PHYSICAL CONSTANTS, ABSOLUTE ZERO={_fmt(ABSOLUTE_ZERO)}, "
                f"STEFAN BOLTZMANN={_fmt(STEFAN_BOLTZMANN_CCX)}\n")
        f.write(f"*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL, {_fmt(study.thermal.t_initial)}\n")
    return m


def time_stepping(case: ThermalCase) -> tuple[int, int, float]:
    """(number of increments, output every k-th, dt): number of increments = a multiple of the number of frames."""
    outs = max(1, int(case.outputs))
    n = max(1, math.ceil(case.time / case.step_size() - 1e-9))
    n = outs * max(1, math.ceil(n / outs))
    return n, n // outs, case.time / n


def cavity_codes(case: ThermalCase) -> dict[str, str]:
    """Name of the cavity → a unique code for ccx (C01, C02, …). ccx allows a cavity name of only 3 characters; shortening the user name
    (e.g. to the first 3 characters) would merge different cavities ("Dutina1" and "Dutina2") and the view factors would be computed over both. The user name goes only into a comment. At most 99 cavities per state.
    """
    out: dict[str, str] = {}
    for bc in case.loads:
        if isinstance(bc, (Radiation, CavityRadiation)) and is_cavity(bc):
            name = cavity_key(bc)
            if name not in out:
                if len(out) >= MAX_CAVITIES:
                    raise ValueError(f"Too many cavities (> {MAX_CAVITIES}) in thermal case '{case.name}' (CalculiX distinguishes cavity names by three characters only)")
                out[name] = f"C{len(out) + 1:02d}"
    return out


def _faces(mesh: Mesh, bc) -> tuple[np.ndarray, np.ndarray]:
    """(elements, face numbers) of all the selections of the condition (cavity radiation may have several selections)."""
    parts = [mesh.element_faces(sel) for sel in bc_selections(bc)]
    if len(parts) == 1:
        return parts[0]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def write_case(study: Study, mesh: Mesh, m: InpMap, case: ThermalCase, path: Path, job: str) -> ThermalJob:
    tj = ThermalJob(job, case.name)
    flux_lines, film_lines, rad_lines, bf_lines, bnd_lines = [], [], [], [], []
    cavity_decks: dict[tuple[str, str], list[str]] = {}

    cavity_map = cavity_codes(case)

    for bc in case.loads:
        if isinstance(bc, FixedTemperature):
            bnd_lines.append(f"{m.nsets[bc.selection]}, 11, 11, {_fmt(bc.value)}")
        elif isinstance(bc, (HeatFlux, HeatFlow)):
            q = bc.value * FLUX_TO_CCX if isinstance(bc, HeatFlux) else \
                bc.value * POWER_TO_CCX / curved_area(mesh, bc.selection)
            el, fno = mesh.element_faces(bc.selection)
            flux_lines += [f"{e + 1}, S{k}, {_fmt(q)}" for e, k in zip(el, fno)]
        elif isinstance(bc, Convection):
            el, fno = _faces(mesh, bc)
            film_lines += [f"{e + 1}, F{k}, {_fmt(bc.ambient)}, {_fmt(bc.h * FLUX_TO_CCX)}" for e, k in zip(el, fno)]
        elif isinstance(bc, (Radiation, CavityRadiation)):
            el, fno = _faces(mesh, bc)
            if is_cavity(bc):
                raw_cname = cavity_key(bc)
                code = cavity_map[raw_cname]
                sink = -300.0 if getattr(bc, "closed", False) else bc.ambient
                deck = cavity_decks.setdefault((code, raw_cname), [])
                deck += [f"{e + 1}, R{k}CR, {_fmt(sink)}, {_fmt(bc.emissivity)}" for e, k in zip(el, fno)]
            else:
                rad_lines += [f"{e + 1}, R{k}, {_fmt(bc.ambient)}, {_fmt(bc.emissivity)}" for e, k in zip(el, fno)]
        elif isinstance(bc, HeatGeneration):
            dens = generation_density(mesh, bc)            # W/mm³ per element (one part, constant)
            pi = mesh.part_names.index(bc.part)
            bf_lines.append(f"P{pi + 1}, BF, {_fmt(float(dens[mesh.tet_part == pi][0]) * POWER_TO_CCX)}")

    with open(path, "w", encoding="ascii", errors="replace") as f:
        f.write(f"** thermal case: {deck_text(case.name)}\n*INCLUDE, INPUT={MODEL}\n")
        if case.transient:
            n, freq, dt = time_stepping(case)
            tj.increments, tj.frequency, tj.dt = n, freq, dt
            f.write(f"*STEP, INC={n + 10}\n*HEAT TRANSFER, DIRECT, SOLVER={solver_name(study.solver.solver)}\n"
                    f"{_fmt(dt)}, {_fmt(case.time)}\n")
        else:
            f.write(f"*STEP, INC=100\n*HEAT TRANSFER, STEADY STATE, SOLVER={solver_name(study.solver.solver)}\n")
        if bnd_lines:
            f.write("*BOUNDARY\n" + "".join(l + "\n" for l in bnd_lines))
        if flux_lines or bf_lines:
            f.write("*DFLUX\n" + "".join(l + "\n" for l in flux_lines + bf_lines))
        if film_lines:
            f.write("*FILM\n" + "".join(l + "\n" for l in film_lines))
        if rad_lines:
            f.write("*RADIATE\n" + "".join(l + "\n" for l in rad_lines))
        for (code, raw_cname), lines in cavity_decks.items():
            f.write(f"** dutina {code}: {deck_text(raw_cname)}\n*RADIATE, CAVITY={code}\n" + "".join(l + "\n" for l in lines))
        fr = f", FREQUENCY={tj.frequency}" if case.transient else ""
        f.write(f"*NODE FILE{fr}\nNT, RFL\n*EL FILE{fr}\nHFL\n*END STEP\n")
    tj.fixed_sets = {bc.selection: m.nsets[bc.selection] for bc in case.loads if isinstance(bc, FixedTemperature)}
    return tj
