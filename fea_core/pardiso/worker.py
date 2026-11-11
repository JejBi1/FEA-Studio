"""PARDISO process: python -m fea_core.pardiso.worker JOB → JOB/x.npy (n × states), JOB/pardiso_info.json.

Input: JOB/indptr.npy (int64), indices.npy (int32), data.npy (float64) = the whole symmetric matrix (constraints
eliminated symmetrically, the diagonal of the constrained DOF = d), rhs.npy (n × states). SPD matrix → mtype 2, upper triangle.
"""
from __future__ import annotations

import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def _mkl():
    from . import mkl_dll
    lib = ctypes.CDLL(str(mkl_dll()))
    i32p = ctypes.POINTER(ctypes.c_int32)
    lib.pardiso.argtypes = [ctypes.POINTER(ctypes.c_int64), i32p, i32p, i32p, i32p, i32p, ctypes.c_void_p, i32p, i32p,
                            i32p, i32p, i32p, i32p, ctypes.c_void_p, ctypes.c_void_p, i32p]
    lib.pardiso.restype = None
    return lib


class Pardiso:
    """MKL PARDISO (LP64, indices from 0) for a real SPD matrix – analysis + factorisation once, solution for several right-hand sides."""

    def __init__(self, K: sp.csr_matrix):
        self.lib = _mkl()
        U = sp.triu(K, format="csr")
        U.sort_indices()
        self.n = U.shape[0]
        self.a = np.ascontiguousarray(U.data, dtype=np.float64)
        self.ia = np.ascontiguousarray(U.indptr, dtype=np.int32)
        self.ja = np.ascontiguousarray(U.indices, dtype=np.int32)
        del U
        self.pt = np.zeros(64, dtype=np.int64)
        self.iparm = np.zeros(64, dtype=np.int32)
        ip = self.iparm
        ip[0] = 1            # own setting of iparm
        ip[1] = 3            # parallel nested dissection (METIS)
        ip[9] = 8            # perturbation of the pivots 1e-8
        ip[17] = -1          # return the number of non-zeros in the factor
        ip[34] = 1           # indexy od 0
        self.call(12, np.zeros(1), np.zeros(1), 1)

    def call(self, phase: int, b: np.ndarray, x: np.ndarray, nrhs: int) -> None:
        i = lambda v: ctypes.byref(ctypes.c_int32(v))   # noqa: E731
        p32 = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))   # noqa: E731
        err = ctypes.c_int32(0)
        self.lib.pardiso(self.pt.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), i(1), i(1), i(2), i(phase), i(self.n),
                         self.a.ctypes.data_as(ctypes.c_void_p), p32(self.ia), p32(self.ja), ctypes.POINTER(ctypes.c_int32)(),
                         i(nrhs), p32(self.iparm), i(0), b.ctypes.data_as(ctypes.c_void_p),
                         x.ctypes.data_as(ctypes.c_void_p), ctypes.byref(err))
        if err.value != 0:
            msg = {-2: "out of memory", -4: "zero pivot (matrix is not positive definite – free body?)"}
            raise RuntimeError(f"PARDISO phase {phase}: error {err.value} {msg.get(err.value, '')}")

    def solve(self, B: np.ndarray) -> np.ndarray:
        B = np.asfortranarray(B.reshape(self.n, -1), dtype=np.float64)
        X = np.zeros_like(B, order="F")
        self.call(33, B, X, B.shape[1])
        return X

    def release(self) -> None:
        self.call(-1, np.zeros(1), np.zeros(1), 1)


def main(job: Path) -> None:
    t0 = time.time()
    K = sp.csr_matrix((np.load(job / "data.npy", mmap_mode="r"), np.load(job / "indices.npy", mmap_mode="r"),
                       np.load(job / "indptr.npy")))
    rhs = np.load(job / "rhs.npy")
    t1 = time.time()
    ps = Pardiso(K)
    t2 = time.time()
    X = ps.solve(rhs)
    t3 = time.time()
    R = K @ X - rhs.reshape(X.shape)
    res = (np.linalg.norm(R, axis=0) / np.maximum(np.linalg.norm(rhs.reshape(X.shape), axis=0), 1e-300)).tolist()
    nnz_factor = int(ps.iparm[17])
    ps.release()
    np.save(job / "x.npy", X)
    (job / "pardiso_info.json").write_text(json.dumps({
        "n": int(K.shape[0]), "nnz": int(K.nnz), "factor_nnz": nnz_factor, "rel_residual": res,
        "t_load": t1 - t0, "t_factor": t2 - t1, "t_solve": t3 - t2, "cases": X.shape[1]}), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
