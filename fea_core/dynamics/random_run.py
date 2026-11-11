"""Random vibrations (D3): ASD of the acceleration of the base → covariance matrix of the modal coordinates (CQC in the full form)
→ RMS of the displacements, RMS of von Mises (Segalman), spectral moments of the equivalent stress (Preumont & Piéfort) →
spectral fatigue (NB / Dirlik / TB) by nodes, response PSD in the probes. Modal superposition in Python (decision 3).
"""
from __future__ import annotations

import numpy as np

from . import psd
from .model import G_MM


def run_random(study, mesh, basis, zeta, out, log, warnings: list) -> dict:
    from .response import direction
    from ..fatigue.spectral import damage, life_hours
    from .fatigue_dyn import node_groups, print_warning
    r = study.dynamics.random
    d = direction(r)
    asd = np.asarray(r.asd, float)
    f = psd.freq_grid(float(asd[:, 0].min()), float(asd[:, 0].max()), basis.freq, zeta)
    W_g = psd.asd_interp(asd, f)
    W = psd.asd_to_mm(W_g)
    gam = basis.gamma[:, :3] @ d
    log(f"Dynamics: random vibration ({len(f)} frequencies, input g_rms {psd.asd_rms(asd):.4g} g)")
    Mi = psd.modal_moments(f, basis.freq, zeta, gam, W, powers=(0, 1, 2, 4))
    sh = np.ascontiguousarray(np.transpose(basis.shapes, (1, 2, 0)))          # (N, 3, K)
    st = np.ascontiguousarray(np.transpose(basis.stress, (1, 2, 0)))          # (N, 6, K)
    U_rms = psd.rms_components(sh, Mi[0])
    m = psd.equivalent_moments(st, Mi)
    vm_rms = np.sqrt(np.maximum(m[:, 0], 0.0))
    T = float(r.duration_h) * 3600.0
    D = np.zeros(len(mesh.nodes))
    for name, nodes, sn in node_groups(study, mesh):
        D[nodes] = damage(m[nodes], T, sn, r.method)
    warnings += [w for w in print_warning(study, mesh) if w not in warnings]
    from .fatigue_dyn import missing_sn
    miss = missing_sn(study, mesh)
    if miss:
        warnings.append("Spectral fatigue: material without Rm and without S-N (" + ", ".join(miss) + ") – fatigue is not calculated")
    probes = {}
    for op in out.probes:
        phi = np.einsum("p,kpi->ik", op["w"], basis.shapes[:, op["nodes"]].astype(float))     # (3, K)
        probes[op["name"]] = {"U_psd": psd.response_psd(f, basis.freq, zeta, gam, W, phi, kind="disp"),
                              "A_psd_g2": psd.response_psd(f, basis.freq, zeta, gam, W, phi, kind="acc_abs")
                              / G_MM ** 2}
    i = int(D.argmax())
    j = int(vm_rms.argmax())
    life = life_hours(D, T)
    summary = {"g_rms_input": float(psd.asd_rms(asd)), "direction": d.tolist(), "duration_h": float(r.duration_h),
               "method": r.method, "max_U_rms_mm": float(np.linalg.norm(U_rms, axis=1).max()),
               "max_mises_rms_MPa": float(vm_rms.max()), "max_mises_3sigma_MPa": float(3 * vm_rms.max()),
               "mises_rms_at": mesh.nodes[j].tolist(), "max_damage": float(D.max()),
               "min_life_h": float(life.min()), "damage_at": mesh.nodes[i].tolist(),
               "note": "RMS von Mises (Segalman) is not Gaussian – 3σ for information only; fatigue from the moments of the equivalent"}
    return {"f": f, "asd_g2": W_g, "U_rms": U_rms.astype(np.float32), "S_rms": vm_rms.astype(np.float32),
            "moments": m.astype(np.float32), "D": D.astype(np.float32), "life_h": np.minimum(life, 1e30).astype(np.float32),
            "probes": probes, "summary": summary}
