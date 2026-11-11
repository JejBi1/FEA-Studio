"""Shock (D4): pulse of the acceleration of the base → response spectrum (SRS, Smallwood) → peaks of the modes q_j = Γ_j·S_a(f_j)/ω_j²
→ combination SRSS / CQC (Der Kiureghian) by components + the missing mass (static response of the residual mass to the ZPA,
combination SRSS – US NRC RG 1.92) and optionally the transient response to the same pulse (exact solution of the equations of the modes).
"""
from __future__ import annotations

import numpy as np

from . import shock as SH
from .model import G_MM


def run_shock(study, mesh, basis, zeta, out, workdir, log, warnings: list, on_line=None, cancel=None,
              static: dict | None = None) -> dict:
    """static = the finished static solution of the unit acceleration in the direction of the shock (from the job of the static correction)."""
    from .response import direction, P_VM
    s = study.dynamics.shock
    d = direction(s)
    gam = basis.gamma[:, :3] @ d
    fn = basis.freq
    pulse_t = pulse_a = None
    if s.pulse == "srs_table":
        Sa = SH.srs_interp(s.srs_table, fn) * G_MM
        zpa = float(np.asarray(s.srs_table, float)[:, 1][-1]) * G_MM
    else:
        T_min = 1.0 / fn.max()
        dt = min(s.tau / 200.0, T_min / 50.0)
        pulse_t, pulse_a = SH.pulse(s.pulse, float(s.A) * G_MM, float(s.tau), dt, rise=s.rise)
        Sa = np.array([SH.srs(pulse_t, pulse_a, [f], zeta=float(z))["maximax"][0] for f, z in zip(fn, zeta)])
        zpa = float(np.abs(pulse_a).max())
    log(f"Dynamics: shock – response spectrum, {s.combine.upper()}, {basis.K} modes")
    q = SH.modal_peaks(fn, gam, Sa)                       # (K,) peaks of the modal coordinates with the sign of Γ
    rho = SH.cqc_rho(fn, zeta) if s.combine == "cqc" else None
    RU = basis.shapes.astype(float) * q[:, None, None]    # (K, N, 3)
    RS = basis.stress.astype(float) * q[:, None, None]    # (K, N, 6)
    U = SH.combine(RU, rho)
    S = SH.combine(RS, rho)
    mm_U = mm_S = None
    if s.missing_mass:
        st = static
        if st is None:
            from .modal import static_solutions
            from ..study import LoadCase, Gravity
            st = static_solutions(study, mesh, [LoadCase("_dyn_mm", [Gravity(vector=list(map(float, d)))])],
                                  workdir, log, on_line, cancel)[0]
        coef = gam / basis.omega ** 2
        mm_U = -(st["U"] - np.einsum("k,kni->ni", coef, basis.shapes)) * zpa
        mm_S = -(st["S"].astype(float) - np.einsum("k,kni->ni", coef, basis.stress.astype(float))) * zpa
        U = np.sqrt(U ** 2 + mm_U ** 2)
        S = np.sqrt(S ** 2 + mm_S ** 2)
    vm = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", S, P_VM, S), 0.0))
    res = {"Sa": Sa, "q_peak": q, "U": U.astype(np.float32), "S": S.astype(np.float32), "mises": vm.astype(np.float32),
           "zpa": zpa}
    if pulse_t is not None:
        f_srs = np.geomspace(max(fn.min() / 4, 1.0), fn.max() * 2, 120)
        sr = SH.srs(pulse_t, pulse_a, f_srs, zeta=float(np.median(zeta)))
        res.update(pulse_t=pulse_t, pulse_a=pulse_a, srs_f=f_srs, srs=sr["maximax"])
        if s.transient:
            from .modal_ode import integrate_base
            T_end = max(float(pulse_t[-1]) + 3.0 / fn.min(), 2 * float(s.tau))
            dt = float(pulse_t[1] - pulse_t[0])
            t = np.arange(0.0, T_end + 0.5 * dt, dt)
            a_b = np.interp(t, pulse_t, pulse_a, right=0.0)
            qt, _, qa = integrate_base(dt, a_b, gam, basis.omega, zeta)
            from .response import envelopes, probe_series
            stride = max(1, int(np.ceil(len(t) / 4000)))
            e = envelopes(basis, None, qt[::stride], np.zeros((len(t[::stride]), 0)), False)
            pr = probe_series(basis, None, out.probes, qt, np.zeros((len(t), 0)), qa)
            for op in out.probes:
                pr[op["name"]]["A"] = pr[op["name"]]["A"] + a_b[:, None] * d[None, :]
            res["time"] = {"t": t, "env_U": e["U"], "env_S": e["S"], "probes": pr}
    i = int(vm.argmax())
    res["summary"] = {"pulse": s.pulse, "A_g": float(s.A), "tau_s": float(s.tau), "combine": s.combine,
                      "missing_mass": bool(s.missing_mass), "zpa_g": zpa / G_MM,
                      "max_U_mm": float(np.linalg.norm(U, axis=1).max()), "max_mises_MPa": float(vm.max()),
                      "at": mesh.nodes[i].tolist(),
                      "note": "von Mises from combined components (SRSS/CQC) – conservative estimate",
                      **({"transient_max_mises_MPa": float(res["time"]["env_S"].max())} if "time" in res else {})}
    return res
