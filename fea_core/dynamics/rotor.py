"""Rotors – Campbell diagram (stage F8.10, D5b of the specification F8): a loop over the speeds in ccx – the static step with the centrifugal
force (Rotation → consistent nodal forces) and the natural frequencies with the stress stiffness (*STEP, PERTURBATION),
the critical speeds from the intersections of f_j(n) with the multiples of the rotational frequency k·n/60.

Limitation (specification 5a.1): the system rotating with the body, only the stress stiffening (Southwell); the softening by the centrifugal
force (spin softening) and the Coriolis / gyroscopic term in the fixed system are not computed – ccx *COMPLEX FREQUENCY,
CORIOLIS is NOT VERIFIED. For shafts on bearings with precession use a specialised tool.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np


def campbell(study, mesh, workdir: Path, log, on_line=None, cancel=None, orders=(1, 2)) -> dict:
    from ..study import LoadCase, Rotation
    from .modal import run_basis
    pr = study.dynamics.prestress
    rpm = sorted(float(x) for x in pr.rpm_list)
    from ..ccx.concurrent import run_tasks

    def speed(k: int, n: float):
        def task(cancel_, on_line_):
            st = copy.copy(study)
            st.load_cases = [LoadCase("_rot", [Rotation(point=list(pr.axis_point), axis=list(pr.axis),
                                                        omega=2 * np.pi * n / 60.0)])] if n > 0 else \
                [LoadCase("_rot", [])]
            log(f"Campbell diagram: {n:g} rpm ({k + 1}/{len(rpm)})")
            return run_basis(st, mesh, Path(workdir) / f"campbell_{k + 1}", log, on_line_, cancel_,
                             prestress_case="_rot" if n > 0 else None, job="campbell")
        return task

    # the speeds are independent ccx jobs (own folders) → side by side as the machine allows (ccx.concurrent)
    bases = run_tasks([speed(k, n) for k, n in enumerate(rpm)], 3 * len(mesh.nodes), cancel, on_line)
    freqs = [b.freq for b in bases]
    K = min(len(f) for f in freqs)
    F = np.array([f[:K] for f in freqs])                  # (n_rpm, K)
    crit = []
    for order in orders:
        line = order * np.asarray(rpm) / 60.0
        for j in range(K):
            d = F[:, j] - line
            for i in range(len(rpm) - 1):
                if d[i] == 0 or d[i] * d[i + 1] < 0:
                    t = d[i] / (d[i] - d[i + 1]) if d[i] != d[i + 1] else 0.0
                    n_c = rpm[i] + t * (rpm[i + 1] - rpm[i])
                    crit.append({"order": order, "mode": j + 1, "rpm": float(n_c),
                                 "f_Hz": float(F[i, j] + t * (F[i + 1, j] - F[i, j]))})
    return {"rpm": np.asarray(rpm), "freq": F, "critical": crit, "orders": list(orders),
            "note": "rotating frame, stress stiffening only (no spin softening and Coriolis)"}
