"""Backend for large models: NGSolve (stiffness matrix) + hypre (parallel PCG with AMG for elasticity).

The same tet10 mesh and the same loads as for ccx (loads.nodal_loads), the same evaluation of reactions
and the same equilibrium check (solve.make_step). NGSolve and hypre run in separate processes.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import config, hypre, memory
from .loads import nodal_loads, distributing_forces, remote_vectors
from .mesh import Mesh
from .study import Study, Displacement, Fixed, RemoteLoad


class LargeSolverError(RuntimeError):
    pass


def unsupported(study: Study) -> list[str]:
    """What the backend cannot do yet (then ccx is used)."""
    out = []
    for lc in study.load_cases:
        for l in lc.loads:
            if isinstance(l, RemoteLoad) and l.coupling == "rigid":
                out.append(f"rigid remote point support ({lc.name})")
    if any(getattr(c, "kind", "") == "cylindrical" for c in study.constraints):
        out.append("cylindrical support (constraint equations in local axes)")
    if any(getattr(c, "kind", "") == "remote_support" for c in study.constraints):
        out.append("remote support (rigid body / RBE3 constraint equations)")
    if study.frame.active:
        out.append("beam structure (ccx U1 elements)")
    if study.mesh.separate:
        out.append("parts joined by *TIE (meshed separately)")
    if any(getattr(m, "ortho", None) for m in study.materials.values()):
        out.append("general orthotropic material (ccx orientation)")
    if any(getattr(c, "kind", "") == "elastic" for c in study.constraints):
        out.append("elastic support (ccx springs)")
    if study.active_bolts():
        out.append("bolts (U1 beams and ccx constraints)")
    if study.active_welds():
        out.append("weld checks (stresses at integration points from ccx)")
    if any(getattr(m, "kind", "") == "point_mass" for m in getattr(study, "masses", [])):
        out.append("point mass (rigid coupling / ccx constraint equations)")
    if study.is_nonlinear():
        out.append("contact / nonlinear analysis (ccx only)")
    return out


# hypre memory (8 processes, tet10 elasticity), measured 12. 10. 2026: ~0.6 GB + 4.8 GB / mil. DOF
# default settings, ~0.6 GB + 3.4 GB / mil. DOF with aggressive coarsening (2.2× more iterations)
MEM_FAST = (0.6, 4.8)
MEM_LEAN = (0.6, 3.4)


def amg_for_memory(ndof: int) -> tuple[dict, str]:
    """Fast AMG setting if it fits into the free memory, otherwise the economical one; otherwise a clear error."""
    import psutil
    free = psutil.virtual_memory().available / 2**30
    mdof = ndof / 1e6
    fast = MEM_FAST[0] + MEM_FAST[1] * mdof
    lean = MEM_LEAN[0] + MEM_LEAN[1] * mdof
    if fast < 0.85 * free:
        return {}, f"fast mode, memory estimate {fast:.1f} GB"
    if lean < 0.85 * free:
        return {"agg_levels": 1}, f"memory-saving mode, memory estimate {lean:.1f} GB (available {free:.1f} GB)"
    raise LargeSolverError(
        f"Not enough free memory for the analysis: estimate {lean:.1f} GB, available {free:.1f} GB. Close other applications or coarsen the mesh.")


def iso_stiffness(E: float, nu: float) -> np.ndarray:
    """Isotropic C (6×6), Voigt xx yy zz xy xz yz, engineering shear."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] += 2 * mu
    C[np.arange(3, 6), np.arange(3, 6)] = mu
    return C


def element_stiffness_all(study: Study, mesh: Mesh) -> np.ndarray | None:
    """C (E, 6, 6) of all elements if the mesh contains printed parts (otherwise None – the isotropic path per part)."""
    from .printing import element_stiffness
    pr = element_stiffness(study, mesh)
    if pr is None:
        return None
    el, Cp = pr
    C = np.empty((len(mesh.tets), 6, 6))
    for pi, pname in enumerate(mesh.part_names):
        m = study.material_of(pname)
        C[mesh.tet_part == pi] = iso_stiffness(m.E, m.nu)
    C[el] = Cp
    return C


def _worker(stage: str, job: Path, log, backend: str = "ngs_hypre") -> None:
    memory.release_unused()                  # the NGSolve worker gets the memory that this process does not use
    p = subprocess.run([sys.executable, "-m", "fea_core.ngs.worker", stage, str(job)],
                       cwd=config.PROJECT_ROOT, capture_output=True, text=True, creationflags=config.NO_WINDOW)
    if p.returncode != 0:
        miss = re.search(r"No module named '([\w.]+)'", p.stderr or "")
        if miss:                                  # missing module ngsolve (installation without the module) – not a traceback printout
            from .solvers import module_error
            raise LargeSolverError(module_error(backend, miss.group(1)))
        raise LargeSolverError(f"NGSolve ({stage}) failed:\n{p.stderr[-3000:]}")


def run_ngs_hypre(study: Study, mesh: Mesh, workdir: Path, warnings: list[str], log,
                  cancel: threading.Event | None = None, thermal: list | None = None):
    return run_ngs(study, mesh, workdir, warnings, log, cancel, thermal, "hypre")


def run_ngs(study: Study, mesh: Mesh, workdir: Path, warnings: list[str], log,
            cancel: threading.Event | None = None, thermal: list | None = None, solver: str = "hypre"):
    """NGSolve assembles the matrix, the system is solved by hypre (AMG, iterative, per state) or PARDISO (direct, ONE
    factorisation for all states). Stresses and nodal forces from the displacements as in ccx (`recovery.recover`).
    thermal = results of the thermal states (ThermalResult) for the thermal load.
    """
    from .results import Results
    from .solve import constraint_masks, make_step
    from .recovery import recover
    bad = unsupported(study)
    if bad:
        raise LargeSolverError("The NGSolve solver cannot yet: " + ", ".join(bad) + " – use ccx.")
    from .solvers import backend_error
    err = backend_error(f"ngs_{solver}")          # missing ngsolve / MKL / hypre / MS-MPI → what to install
    if err:
        raise LargeSolverError(err)
    job = Path(workdir) / "ngs"
    job.mkdir(parents=True, exist_ok=True)
    for f in job.glob("*.npy"):
        f.unlink()
    mesh.save(job / "mesh.npz")
    mats = [{"E": study.material_of(p).E, "nu": study.material_of(p).nu, "alpha": study.material_of(p).alpha or 0.0}
            for p in mesh.part_names]
    (job / "materials.json").write_text(json.dumps(mats), encoding="utf-8")
    (job / "C_el.npy").unlink(missing_ok=True)
    pr = element_stiffness_all(study, mesh)
    if pr is not None:                        # 3D printing: C per element in global axes (the same groups as in ccx)
        np.save(job / "C_el.npy", pr)
    t0 = time.time()
    n = len(mesh.nodes)
    # constraints (the same masks as in ccx) and prescribed values
    masks = constraint_masks(study, mesh)
    fixed = np.zeros((n, 3), bool)
    gval = np.zeros((n, 3))
    for c in study.constraints:
        idx = mesh.surface_nodes(c.selection)
        if isinstance(c, Fixed):
            fixed[idx] = True
        elif isinstance(c, Displacement):
            for d, v in enumerate((c.ux, c.uy, c.uz)):
                if v is not None:
                    fixed[idx, d] = True
                    gval[idx, d] = v
    Fs = []
    for lc in study.load_cases:
        f = nodal_loads(study, mesh, lc)[0]
        for l in lc.loads:                         # distributing constraint = distribution of F and M to the nodes of the face
            if isinstance(l, RemoteLoad):
                f = f + distributing_forces(mesh, l.selection, l.point, *remote_vectors(study, l))
        Fs.append(f)
    np.save(job / "F.npy", np.column_stack([f.ravel() for f in Fs]))
    temps = [None] * len(study.load_cases)
    if study.has_thermal_loads():                 # thermal load: ΔT per node, the forces are computed by NGSolve
        from .thermal.field import node_temperatures
        tmap = {t.name: t for t in (thermal or [])}
        temps = [node_temperatures(study, mesh, lc, tmap) for lc in study.load_cases]
        t_ref = study.thermal.reference_temperature
        np.save(job / "dT.npy", np.column_stack([np.zeros(n) if t is None else t - t_ref for t in temps]))
    np.save(job / "fixed.npy", fixed.ravel())
    np.save(job / "g.npy", gval.ravel())
    del Fs
    log("Assembling stiffness matrix (NGSolve)")
    _worker("assemble", job, log, f"ngs_{solver}")   # the matrix goes directly into the files for hypre, the main process does not load it
    ainfo = json.loads((job / "assemble.json").read_text(encoding="utf-8"))
    if ainfo["rigid_body_residual"] > 1e-8:
        raise LargeSolverError(f"The stiffness matrix failed the rigid-body motion check ({ainfo['rigid_body_residual']:.2e})")
    if cancel is not None and cancel.is_set():
        raise LargeSolverError("Analysis canceled by user")
    if solver == "pardiso":
        from . import pardiso
        import psutil
        need = pardiso.memory_estimate_gb(3 * n, ainfo["nnz"])
        free = psutil.virtual_memory().available / 2**30
        if need > 0.85 * free:
            raise LargeSolverError(f"Not enough memory for the direct PARDISO solver (estimate {need:.1f} GB, available {free:.1f} GB) – use ngs_hypre (iterative) or coarsen the mesh.")
        log(f"Solving the system of {3 * n:,} equations directly (PARDISO".replace(",", " ")
            + f", one factorisation for {len(study.load_cases)} states, memory estimate {need:.1f} GB)")
        try:
            X, info = pardiso.run_prepared(job / "hypre", cancel=cancel)
        except pardiso.PardisoError as e:
            raise LargeSolverError(str(e)) from None
        log(f"  factorization {info['t_factor']:.1f} s ({info['factor_nnz'] / 1000000.0:.0f} M nonzeros), solve {info['t_solve']:.1f} s, rel. residual ≤ {max(info['rel_residual']):.1e}")
    else:
        nproc = study.solver.nproc or config.NUM_THREADS
        amg, mode = amg_for_memory(3 * n)
        log(f"Solving a system of {3 * n:,} equations (hypre BoomerAMG, {nproc} processes, {mode})".replace(",", " "))
        X, info = hypre.run_prepared(job / "hypre", nproc=nproc, amg=amg)
        info["amg_mode"] = mode
        for c, ci in zip(study.load_cases, info["cases"]):
            log(f"  {c.name}: {ci['iterations']} iterations, rel. residual {ci['rel_residual']:.1e}")
    if cancel is not None and cancel.is_set():
        raise LargeSolverError("Analysis canceled by user")
    log("Computing stresses (from displacements like ccx)")
    C_el = pr if pr is not None else None
    steps = []
    t_ref = study.thermal.reference_temperature
    for k, lc in enumerate(study.load_cases):
        U = X[:, k].reshape(n, 3)
        S, RF = recover(study, mesh, U, None if temps[k] is None else temps[k] - t_ref, C_el)
        ref = 0.0
        if temps[k] is not None:
            from .thermal.field import force_scale
            ref = force_scale(study, mesh, temps[k])
        st = make_step(study, mesh, lc, U, S, RF, masks, ref_force=ref)
        if temps[k] is not None:
            st.fields["T"] = temps[k]
        steps.append(st)
    for f in list(job.glob("*.npy")) + list((job / "hypre").glob("*.npy")):
        f.unlink(missing_ok=True)
    info.update({"assemble": ainfo, "t_total": time.time() - t0})
    out = {"backend": f"ngs_{solver}", solver: info}
    from .solve import print_results
    print_results(study, mesh, steps, out, warnings, log)
    return Results(mesh, steps, warnings, out)
