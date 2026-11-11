"""Direct solver MKL PARDISO for the system assembled by NGSolve (backend "ngs_pardiso").

One factorisation for all load states (ccx factorises again in every *STEP): plate + foot 197 k DOF,
3 states – ccx 34 s, NGSolve + PARDISO ~15 s (prototype scratch/pardiso_proto, 13. 10. 2026). The system is in the same files
as for hypre (JOB/hypre: indptr, indices, data, rhs – symmetric elimination of the constraints). The factorisation runs
in a separate process (`python -m fea_core.pardiso.worker JOB`) – a crash of MKL does not bring down the GUI, it can be cancelled.
MKL = pip package mkl 2023.2.0 in the venv (mkl_rt.2.dll; CAUTION: version 2024.2 gave wrong results in ccx with ≥ 8 threads).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .. import config


class PardisoError(RuntimeError):
    pass


def mkl_dll() -> Path:
    return config.mkl_rt()


def available() -> bool:
    return mkl_dll().exists()


def memory_estimate_gb(ndof: int, nnz: int) -> float:
    """Estimate of the memory of the PARDISO process [GB]: matrix (upper triangle) + factor (~570·n·(n/2e5)^(1/3) non-zeros,
    measured tet10 197 k DOF: 112 mil.) + working memory ×1.3.
    """
    factor = 570.0 * ndof * max(ndof / 2e5, 0.05) ** (1 / 3)
    return (12.0 * nnz / 2 + 1.3 * 8.0 * factor) / 2**30 + 0.3


def run_prepared(workdir: str | Path, threads: int | None = None, cancel=None) -> tuple[np.ndarray, dict]:
    """Solves the system in the workdir (indptr/indices/data/rhs.npy) → (X (n, states), info)."""
    wd = Path(workdir)
    if not available():
        raise PardisoError(f"MKL PARDISO not found ({mkl_dll()})")
    for f in ("x.npy", "pardiso_info.json"):
        (wd / f).unlink(missing_ok=True)
    env = config.ccx_env()
    env["MKL_NUM_THREADS"] = str(threads or config.NUM_THREADS)
    env["MKL_THREADING_LAYER"] = "INTEL"                 # Python process without libgomp: Intel OpenMP from the package intel-openmp
    log = wd / "pardiso.log"
    rc = config.run_logged([sys.executable, "-m", "fea_core.pardiso.worker", str(wd)], log, cancel, env)
    if rc is None:
        raise PardisoError("Analysis canceled by user")
    if rc != 0 or not (wd / "pardiso_info.json").exists():
        raise PardisoError(f"PARDISO failed (code {rc}):\n"
                           + log.read_text(encoding="utf-8", errors="replace")[-3000:])
    info = json.loads((wd / "pardiso_info.json").read_text(encoding="utf-8"))
    X = np.load(wd / "x.npy")
    bad = [r for r in info["rel_residual"] if not r <= 1e-8]
    if bad:
        raise PardisoError(f"PARDISO: large residual {bad} – the system is ill-conditioned (free body?)")
    return X, info
