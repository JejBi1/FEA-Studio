"""Calculation of the thermal states in ccx (*HEAT TRANSFER) and the evaluation: temperature, flux, fluxes of the boundary
conditions and the check of the heat balance (an analogue of the equilibrium check of the mechanics).

Verified by a trial (scratch/thermal/exp1.py, 13. 10. 2026): RFL in the .frd = K·T (conduction only) in ALL nodes
– in the free nodes it equals the external heat (flux, convection, radiation, source), the sum over the mesh = 0.
The reaction of a prescribed temperature = RFL − the external heat in its nodes. DFLUX > 0 = heat into the body.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .. import config
from ..ccx.frd import read_frd
from ..ccx.runner import run_ccx, CcxError
from ..mesh import Mesh
from ..results import ThermalResult
from ..study import Study
from .inp import write_model, write_case, thermal_contact_conductances, MODEL
from .loads import bc_nodal, bc_label
from .model import (FixedTemperature, ThermalCase, POWER_TO_CCX, FLUX_TO_CCX, is_cavity, bc_selections,
                    cavity_key)

CAVITY_CLOSED_TOL = 0.02     # closed cavity: |net flux| / Σ|fluxes of the faces| above this limit = a warning (error of the view factors)

# warnings of ccx that mean nothing for heat conduction
_IGNORED = ("nonlinear geometric analysis", "ignored")
# *SURFACE INTERACTION (contact conductance) is read by ccx as a material without *CONDUCTIVITY → a harmless warning;
# it is ignored only in a study with a contact conductance (a missing λ of a part is reported by thermal.check as an error)
_IGNORED_GAP = ("no conductivity",)


def heat_flows(study: Study, mesh: Mesh, case: ThermalCase, T: np.ndarray, rfl: np.ndarray) -> tuple[list, dict]:
    """Fluxes of the boundary conditions [W] (+ into the body) and the balance. rfl = RFL from ccx converted to W.

    Cavity radiation (*RADIATE, CAVITY): the heat of the faces of the cavity depends on the view factors computed by ccx, and in Python
    we do not get them from T (bc_nodal returns None). In the free nodes, however, RFL = external heat, so the heat of the cavity in a node =
    RFL − the heat of the other conditions in the same node (the nodes with a prescribed temperature absorb it in the reaction). The balance thus still
    checks ALL the other nodes (sources, fluxes, convection, radiation to the surroundings) and the row of the cavity carries the net heat that
    the cavity supplies to the body: for an open cavity the exchange with the surroundings, for a closed one (view factors normalised to 1) it should be
    ≈ 0 – a larger deviation (`CAVITY_CLOSED_TOL`) is reported as a warning in bal["warnings"].
    """
    n = len(mesh.nodes)
    fext = np.zeros(n)
    rows = []
    cavities: dict[str, list] = {}
    for bc in case.loads:
        if is_cavity(bc):
            cavities.setdefault(cavity_key(bc), []).append(bc)
            continue
        f = bc_nodal(mesh, bc, T)
        if f is None:
            continue
        fext += f
        rows.append({"kind": bc.kind, "name": bc_label(bc), "Q_W": float(f.sum())})
    fixed = np.zeros(n, bool)
    owner = {}
    for bc in case.loads:
        if isinstance(bc, FixedTemperature):
            idx = mesh.surface_nodes(bc.selection)
            new = idx[~fixed[idx]]                 # a node shared by several selections is counted once (the first)
            fixed[new] = True
            owner[bc.selection] = new
    R = np.where(fixed, rfl - fext, 0.0)
    fix_rows = [{"kind": "fixed_temperature", "name": name, "Q_W": float(R[idx].sum())} for name, idx in owner.items()]
    warns: list[str] = []
    taken = fixed.copy()                           # free nodes not yet assigned to any cavity
    fcav = np.zeros(n)
    cav_rows, cav_faces = [], []
    for name, bcs in cavities.items():
        idx = np.unique(np.concatenate([mesh.surface_nodes(sel) for bc in bcs for sel in bc_selections(bc)]))
        idx = idx[~taken[idx]]
        taken[idx] = True
        f = np.zeros(n)
        f[idx] = (rfl - fext)[idx]
        fcav += f
        cav_rows.append({"kind": "cavity_radiation", "name": name, "Q_W": float(f.sum())})
        done = np.zeros(n, bool)                   # heat of the individual faces of the cavity (a shared edge node = first face)
        for bc in bcs:
            for sel in bc_selections(bc):
                ix = mesh.surface_nodes(sel)
                ix = ix[~done[ix]]
                done[ix] = True
                cav_faces.append({"cavity": name, "selection": sel, "emissivity": float(bc.emissivity),
                                  "Q_W": float(f[ix].sum())})
        throughput = float(np.abs(f[idx]).sum())
        closed = all(bool(getattr(bc, "closed", False)) for bc in bcs)
        if closed and throughput > 1e-12 and abs(f.sum()) / throughput > CAVITY_CLOSED_TOL:
            warns.append(f"Closed cavity '{name}': net heat of the faces {f.sum():.4g} W with exchange {throughput / 2:.4g} W ({100 * abs(f.sum()) / throughput:.2g} %) – the ccx view factors do not match (open cavity? set closed=False)")
    rows += cav_rows
    total = R.sum() + fext.sum() + fcav.sum()
    scale = sum(abs(r["Q_W"]) for r in rows + fix_rows)
    imb = abs(total) / scale if scale > 1e-12 else 0.0
    bal = {"residual_W": float(total), "imbalance": float(imb),
           "in_W": float(sum(r["Q_W"] for r in rows + fix_rows if r["Q_W"] > 0)),
           "out_W": float(-sum(r["Q_W"] for r in rows + fix_rows if r["Q_W"] < 0))}
    if warns:
        bal["warnings"] = warns
    if cav_faces:
        bal["cavity_faces"] = cav_faces             # + = heat into the body through the face (net radiation exchange)
    return fix_rows + rows, bal


def _nodal(block, n: int, ncomp: int) -> np.ndarray:
    out = np.zeros((n, ncomp))
    msk = block.node_ids <= n
    out[block.node_ids[msk] - 1] = block.values[msk, :ncomp]
    return out


def run_thermal(study: Study, mesh: Mesh, workdir: str | Path, warnings: list[str], log=None,
                on_solver_line=None, cancel: threading.Event | None = None,
                cases: list[str] | None = None) -> list[ThermalResult]:
    """Calculates the thermal states of the study (each as a separate ccx job; several states run in parallel – the jobs are independent,
    the transient state costs mainly the assembly in ccx, not the factorisation). cases = only these states (None = all),
    the order of the results = the order in the study.
    """
    from concurrent.futures import ThreadPoolExecutor
    log = log or (lambda s: None)
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    m = write_model(study, mesh, wd / MODEL)
    exe = Path(study.solver.exe) if study.solver.exe else config.CCX_EXE
    n = len(mesh.nodes)
    ignored = _IGNORED + (_IGNORED_GAP if thermal_contact_conductances(study, mesh) else ())
    todo = [(k, c) for k, c in enumerate(study.thermal.cases, 1) if cases is None or c.name in cases]
    jobs = []
    for k, case in todo:
        job = f"case{k}"
        tj = write_case(study, mesh, m, case, wd / f"{job}.inp", job)
        log(f"Heat conduction (CalculiX): {case.name}" + (f" – transient, {tj.increments} steps of {tj.dt:g} s"
                                                       if case.transient else ""))
        jobs.append((job, case))
    workers = max(1, min(len(jobs), config.NUM_THREADS // 2))
    threads = max(2, config.NUM_THREADS // workers) if workers > 1 else None
    stop = threading.Event()

    def solve(item):
        job, case = item
        if stop.is_set():
            return None
        lines = on_solver_line
        if on_solver_line and workers > 1:
            lines = lambda s, c=case.name: on_solver_line(f"[{c}] {s}")   # noqa: E731
        try:
            return run_ccx(wd, job, exe=exe, on_line=lines, cancel=cancel if workers == 1 else _either(cancel, stop),
                           threads=threads)
        except CcxError:
            stop.set()
            raise
    if workers == 1:
        runs = [solve(j) for j in jobs]
    else:
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(solve, j) for j in jobs]
            runs = []
            for f in futs:
                try:
                    runs.append(f.result())
                except CcxError:
                    stop.set()
                    raise
    if cancel is not None and cancel.is_set():
        raise CcxError("Analysis canceled by user")
    out = []
    for (job, case), run in zip(jobs, runs):
        if not run.finished or run.errors:
            raise CcxError(f"CalculiX (thermal state '{case.name}') ended with an error:\n  "
                           + "\n  ".join(run.errors or [f"return code {run.returncode}"]) + f"\nlog: {run.log_path}")
        for w in run.warnings:
            if not any(x in w.lower() for x in ignored) and w not in warnings:
                warnings.append(w)
        out.append(_result(study, mesh, case, wd / f"{job}.frd", n, log, warnings))
    return out


def _either(a: threading.Event | None, b: threading.Event) -> threading.Event:
    """Event set as soon as a or b is set (cancellation by the user / error of another parallel job)."""
    if a is None:
        return b
    both = threading.Event()

    def watch():
        while not both.is_set():
            if a.wait(0.2) or b.is_set():
                both.set()
    threading.Thread(target=watch, daemon=True).start()
    return both


def _result(study: Study, mesh: Mesh, case, frd_path: Path, n: int, log, warnings: list[str]) -> ThermalResult:
    frd = read_frd(frd_path)
    temps = [b for b in frd.blocks if b.name == "NDTEMP"]
    if not temps:
        raise CcxError(f"Thermal case '{case.name}': temperatures are missing in the results")
    T = _nodal(temps[-1], n, 1)[:, 0]
    flux = _nodal(frd.get("FLUX"), n, 3) / FLUX_TO_CCX          # mW/mm² → W/m²
    rfl = _nodal(frd.get("RFL"), n, 1)[:, 0] / POWER_TO_CCX     # mW → W
    heat, bal = heat_flows(study, mesh, case, T, rfl)
    tr = ThermalResult(case.name, T, flux, case.transient, heat=heat)
    if case.transient:
        # frame t = 0 = initial temperature (ccx does not write it) → at_time interpolates also before the 1st output
        tr.times = [0.0] + [float(b.time) for b in temps]
        tr.frames = np.vstack([np.full(n, float(study.thermal.t_initial))]
                              + [_nodal(b, n, 1)[:, 0] for b in temps])
        bal = {"note": "transient case – the balance includes stored heat, not checked",
               **({"cavity_faces": bal["cavity_faces"]} if "cavity_faces" in bal else {})}
    elif bal["imbalance"] > config.EQUILIBRIUM_TOL:
        raise CcxError(
            f"Thermal case '{case.name}': heat balance does not close (residual {bal['residual_W']:.4g} W, {100 * bal['imbalance']:.3g} % of fluxes). The result is invalid.")
    from .interfaces import interface_flows
    flows = interface_flows(study, mesh, T)        # heat through the contact conductance interfaces (end of the state)
    if flows:
        bal["interfaces"] = flows
    tr.balance = bal
    for w in bal.get("warnings", []):
        warnings.append(f"{case.name}: {w}")
    log(f"  {case.name}: T {T.min():.4g} … {T.max():.4g} °C" + ("" if case.transient else
                                                             f", balance {100 * bal['imbalance']:.2g} %"))
    return tr
