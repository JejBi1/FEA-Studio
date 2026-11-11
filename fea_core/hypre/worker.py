"""MPI process of the hypre solver (PCG + BoomerAMG for elasticity).  mpiexec -n P python -m fea_core.hypre.worker JOB

JOB contains (saved by fea_core.hypre.solve): indptr/indices/data.npy (CSR of the whole system, order node×3 components, Dirichlet already applied), rhs.npy (n × number of states), rbm.npy (n × 3 rotations),
job.json (limits of the rows per process, tolerance). Each process reads only its own rows (memmap).
Output: x_<rank>.npy, info.json (iterations, residuals, times).

AMG setting for elasticity taken from MFEM (HypreBoomerAMG::SetElasticityOptions):
system AMG (3 functions), nodal coarsening, interpolation vectors = rotations of the body (GM-2).
hypre 3.2 from conda-forge: HYPRE_Int and HYPRE_BigInt = 32 bit, MPI (MS-MPI), without OpenMP.
"""
from __future__ import annotations

import ctypes as C
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HYPRE_PARCSR = 5555
I32 = np.int32

# default settings of BoomerAMG (tuned 12. 10. 2026): threshold 0.25, without rotational vectors and without nodal
# coarsening, smoothing l1-Jacobi (18). l1-Gauss-Seidel (8) with the constraint of only some components of the displacement (V3,
# symmetry) broke the PCG – residual 1.0 after 7 iterations. Overrides job.json["amg"].
AMG_DEFAULTS = {"coarsen": 10, "agg_levels": 0, "theta": 0.25, "interp": 6, "pmax": 4, "relax": 18, "sweeps": 1,
                "nodal": 0, "nodal_diag": 1, "relax_coarse": 18, "interp_vec": 0, "qmax": 4}


def _p(a, ct):
    return a.ctypes.data_as(C.POINTER(ct))


def main(job: Path) -> None:
    from .. import config
    if config.IS_WIN:
        os.add_dll_directory(str(config.HYPRE_BIN))
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    H = C.CDLL(str(config.HYPRE_BIN / config.HYPRE_LIB_NAME))
    ch = C.c_int(MPI._handleof(comm))          # MS-MPI i MPICH: MPI_Comm = int
    meta = json.loads((job / "job.json").read_text(encoding="utf-8"))
    bounds = meta["bounds"]
    lo, hi = bounds[rank], bounds[rank + 1]    # rows [lo, hi)
    t0 = time.time()

    indptr = np.load(job / "indptr.npy", mmap_mode="r")
    indices = np.load(job / "indices.npy", mmap_mode="r")
    data = np.load(job / "data.npy", mmap_mode="r")
    ip = np.asarray(indptr[lo:hi + 1], dtype=np.int64)
    s, e = int(ip[0]), int(ip[-1])
    cols = np.ascontiguousarray(indices[s:e], dtype=I32)
    vals = np.ascontiguousarray(data[s:e], dtype=np.float64)
    ncols = np.ascontiguousarray(np.diff(ip), dtype=I32)
    rows = np.arange(lo, hi, dtype=I32)
    rhs_all = np.load(job / "rhs.npy", mmap_mode="r")
    rbm = np.load(job / "rbm.npy", mmap_mode="r")
    nloc = hi - lo

    H.HYPRE_Initialize()
    A, parA = C.c_void_p(), C.c_void_p()
    H.HYPRE_IJMatrixCreate(ch, lo, hi - 1, lo, hi - 1, C.byref(A))
    H.HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR)
    H.HYPRE_IJMatrixSetRowSizes(A, _p(ncols, C.c_int))
    H.HYPRE_IJMatrixInitialize(A)
    H.HYPRE_IJMatrixSetValues(A, C.c_int(nloc), _p(ncols, C.c_int), _p(rows, C.c_int),
                              _p(cols, C.c_int), _p(vals, C.c_double))
    H.HYPRE_IJMatrixAssemble(A)
    H.HYPRE_IJMatrixGetObject(A, C.byref(parA))
    del cols, vals

    def vector(values):
        v, pv = C.c_void_p(), C.c_void_p()
        H.HYPRE_IJVectorCreate(ch, lo, hi - 1, C.byref(v))
        H.HYPRE_IJVectorSetObjectType(v, HYPRE_PARCSR)
        H.HYPRE_IJVectorInitialize(v)
        arr = np.ascontiguousarray(values, dtype=np.float64)
        H.HYPRE_IJVectorSetValues(v, C.c_int(nloc), _p(rows, C.c_int), _p(arr, C.c_double))
        H.HYPRE_IJVectorAssemble(v)
        H.HYPRE_IJVectorGetObject(v, C.byref(pv))
        return v, pv

    # interpolation vectors: rotations of the body (the translation is covered by the system AMG)
    rbm_vecs = [vector(rbm[lo:hi, k]) for k in range(rbm.shape[1])]
    rbm_arr = (C.c_void_p * len(rbm_vecs))(*[pv for _, pv in rbm_vecs])
    t_load = time.time() - t0

    # ---------------------------------------------------------------- BoomerAMG (elasticity)
    amg = C.c_void_p()
    H.HYPRE_BoomerAMGCreate(C.byref(amg))
    o = dict(AMG_DEFAULTS, **meta.get("amg", {}))
    H.HYPRE_BoomerAMGSetCoarsenType(amg, int(o["coarsen"]))
    H.HYPRE_BoomerAMGSetAggNumLevels(amg, int(o["agg_levels"]))
    H.HYPRE_BoomerAMGSetStrongThreshold(amg, C.c_double(o["theta"]))
    H.HYPRE_BoomerAMGSetInterpType(amg, int(o["interp"]))
    H.HYPRE_BoomerAMGSetPMaxElmts(amg, int(o["pmax"]))
    H.HYPRE_BoomerAMGSetRelaxType(amg, int(o["relax"]))
    H.HYPRE_BoomerAMGSetNumSweeps(amg, int(o["sweeps"]))
    H.HYPRE_BoomerAMGSetNumFunctions(amg, 3)
    if o["nodal"]:
        H.HYPRE_BoomerAMGSetNodal(amg, int(o["nodal"]))
        H.HYPRE_BoomerAMGSetNodalDiag(amg, int(o["nodal_diag"]))
    H.HYPRE_BoomerAMGSetCycleRelaxType(amg, int(o["relax_coarse"]), 3)
    if o["interp_vec"]:
        H.HYPRE_BoomerAMGSetInterpVecVariant(amg, int(o["interp_vec"]))
        H.HYPRE_BoomerAMGSetInterpVecQMax(amg, int(o["qmax"]))
        H.HYPRE_BoomerAMGSetSmoothInterpVectors(amg, 1)
        H.HYPRE_BoomerAMGSetInterpVectors(amg, len(rbm_vecs), rbm_arr)
    if (job / "cpts.npy").exists():                 # forced C-points (p-multigrid: P2 → P1)
        cp = np.load(job / "cpts.npy")
        cp = np.ascontiguousarray(cp[(cp >= lo) & (cp < hi)], dtype=I32)
        H.HYPRE_BoomerAMGSetCPoints(amg, int(o.get("cpt_level", 1)), C.c_int(len(cp)), _p(cp, C.c_int))
        _keep = cp
    H.HYPRE_BoomerAMGSetMaxIter(amg, 1)
    H.HYPRE_BoomerAMGSetTol(amg, C.c_double(0.0))
    H.HYPRE_BoomerAMGSetPrintLevel(amg, 0)

    pcg = C.c_void_p()
    H.HYPRE_ParCSRPCGCreate(ch, C.byref(pcg))
    H.HYPRE_PCGSetMaxIter(pcg, int(meta.get("maxiter", 2000)))
    H.HYPRE_PCGSetTol(pcg, C.c_double(meta.get("tol", 1e-10)))
    H.HYPRE_PCGSetTwoNorm(pcg, 1)
    H.HYPRE_PCGSetPrintLevel(pcg, 0)
    H.HYPRE_PCGSetLogging(pcg, 1)
    solve_fn = C.cast(H.HYPRE_BoomerAMGSolve, C.c_void_p)
    setup_fn = C.cast(H.HYPRE_BoomerAMGSetup, C.c_void_p)
    H.HYPRE_PCGSetPrecond(pcg, solve_fn, setup_fn, amg)

    infos = []
    t1 = time.time()
    nrhs = rhs_all.shape[1]
    X = np.zeros((nloc, nrhs))
    for k in range(nrhs):
        b, pb = vector(rhs_all[lo:hi, k])
        x, px = vector(np.zeros(nloc))
        ts = time.time()
        if k == 0:
            H.HYPRE_ParCSRPCGSetup(pcg, parA, pb, px)
            H.HYPRE_ClearAllErrors()              # singular coarse levels (see MFEM) are not an error
            t_setup = time.time() - ts
            ts = time.time()
        H.HYPRE_ParCSRPCGSolve(pcg, parA, pb, px)
        it, res = C.c_int(), C.c_double()
        H.HYPRE_PCGGetNumIterations(pcg, C.byref(it))
        H.HYPRE_PCGGetFinalRelativeResidualNorm(pcg, C.byref(res))
        out = np.zeros(nloc)
        H.HYPRE_IJVectorGetValues(x, C.c_int(nloc), _p(rows, C.c_int), _p(out, C.c_double))
        X[:, k] = out
        infos.append({"iterations": it.value, "rel_residual": res.value, "t_solve": time.time() - ts})
        H.HYPRE_IJVectorDestroy(b)
        H.HYPRE_IJVectorDestroy(x)
    np.save(job / f"x_{rank}.npy", X)
    if sys.platform == "win32":
        import psutil
        peak = psutil.Process().memory_info().peak_wset / 2**30
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20          # kB → GB
    peaks = comm.gather(peak, root=0)
    comm.Barrier()
    if rank == 0:
        info = {"nproc": size, "n": int(bounds[-1]), "t_load": t_load, "t_setup": t_setup,
                "t_total_solve": time.time() - t1, "cases": infos, "peak_gb_per_rank": peaks,
                "peak_gb_total": float(sum(peaks))}
        (job / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    H.HYPRE_ParCSRPCGDestroy(pcg)
    H.HYPRE_BoomerAMGDestroy(amg)
    H.HYPRE_IJMatrixDestroy(A)
    H.HYPRE_Finalize()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
