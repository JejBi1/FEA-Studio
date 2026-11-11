"""Calculation of the dynamics (Study.dynamics) on the mesh of the statics: modal basis (ccx) → residual static correction (ccx) →
harmonic / transient response, random vibrations, shock (modal superposition in Python, decision 3 of the specification F8)
→ DynamicsResults. Called by solve.run_study after the statics and the natural frequencies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .check import _dirs, memory_check
from .modal import run_basis, zetas, basis_warnings, static_solutions, DynamicsError
from .results import DynamicsResults
from . import response as R


def run_dynamics(study, mesh, workdir: Path, log, on_line=None, cancel=None) -> DynamicsResults:
    ds = study.dynamics
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    warnings = memory_check(study, mesh)
    pr = ds.prestress
    basis = run_basis(study, mesh, workdir, log, on_line, cancel,
                      prestress_case=(pr.case if pr.enabled and pr.case else None))
    if basis.K == 0:
        raise DynamicsError("Dynamics: the modal basis is empty (no mode with a positive frequency)")
    zeta = zetas(ds.damping, basis.freq)
    eta = (ds.damping.eta if ds.damping.eta else 2 * ds.damping.zeta) if ds.damping.kind == "structural" else None
    f_top = None
    if ds.harmonic.enabled:
        f_top = ds.harmonic.f_max
    if ds.random.enabled and ds.random.asd:
        f_top = max(f_top or 0.0, float(np.asarray(ds.random.asd, float)[:, 0].max()))
    warnings += basis_warnings(basis, _dirs(study), f_top)
    out = DynamicsResults(basis.freq, zeta, basis.shapes.astype(np.float32), basis.stress, basis.gamma,
                          basis.eff_mass, basis.rf_sum, basis.checks, basis.model_mass, basis.prestress)
    out.probes = R.probe_operators(study, mesh, ds.probes)
    # terms of the excitation of the harmonic and transient response + residual static vectors (one static job)
    groups, res_cases, res_labels = {}, [], []
    for name, spec in (("harmonic", ds.harmonic), ("transient", ds.transient)):
        if not spec.enabled or not spec.excitation:
            continue
        terms, w = R.build_terms(study, mesh, basis, spec.excitation)
        warnings += [x for x in w if x not in warnings]
        idx = []
        for it, lc in zip(terms.items, R.residual_cases(study, terms)):
            key = ("f", it["exc"].case) if it["kind"] == "force" else ("b", tuple(np.round(it["dir"], 12)))
            if key not in res_labels:
                res_labels.append(key)
                res_cases.append(lc)
            idx.append(res_labels.index(key))
        groups[name] = (terms, idx)
    use_res = (ds.harmonic.enabled and ds.harmonic.static_correction) or ds.transient.enabled
    n_res = len(res_cases) if use_res else 0
    if not use_res:
        res_cases, res_labels = [], []
    mm_key = mm_static = None
    if ds.shock.enabled and ds.shock.missing_mass:   # missing mass of the shock = the same state as the base (one job)
        mm_key = ("b", tuple(np.round(R.direction(ds.shock), 12)))
        if mm_key not in res_labels:
            from ..study import LoadCase, Gravity
            res_labels.append(mm_key)
            res_cases.append(LoadCase("_dyn_mm", [Gravity(vector=list(map(float, mm_key[1])))]))
    statics = []
    if res_cases:
        log(f"Dynamics: static correction of higher modes ({len(res_cases)} auxiliary cases, CalculiX)")
        statics = static_solutions(study, mesh, res_cases, workdir, log, on_line, cancel)
        if mm_key is not None:
            mm_static = statics[res_labels.index(mm_key)]
    if n_res:
        # residual vectors by the individual keys (the same key = the same vector)
        res_labels, res_cases = res_labels[:n_res], res_cases[:n_res]
        dummy = R.Terms()
        for key, lc in zip(res_labels, res_cases):
            if key[0] == "f":
                F, pts, _ = R.case_load(study, mesh, key[1])
                p = np.einsum("kni,ni->k", basis.shapes, F)
                from .modal import point_mass_terms
                pm = point_mass_terms(study, mesh, basis)
                for i, k, fv in pts:
                    p += pm[i][0][:, k, :] @ fv
                dummy.items.append({"kind": "force", "p": p})
            else:
                d = np.asarray(key[1], float)
                dummy.items.append({"kind": "base", "p": -(basis.gamma[:, :3] @ d)})
        res = R.residual_vectors(basis, dummy, statics[:n_res])
        res["labels"] = [f"{'force' if k[0] == 'f' else 'base'} {k[1]}" for k in res_labels]
        out.residual = res
    # ---------------------------------------------------------------- harmonic response
    if "harmonic" in groups:
        h = ds.harmonic
        terms, idx = groups["harmonic"]
        freq = R.freq_grid(h.f_min, h.f_max, basis.freq, zeta, h.points, h.bias)
        log(f"Dynamics: harmonic response ({len(freq)} frequencies, {basis.K} modes)")
        hr = R.harmonic(basis, terms, freq, zeta, eta, residual=out.residual is not None and h.static_correction)
        hr["res_index"] = np.asarray(idx if out.residual is not None and h.static_correction else [], np.int64)
        res_h = _subset(out.residual, hr["res_index"])
        hr.update(_env(basis, res_h, hr["q"], hr["c"], True))
        hr["probes"] = R.probe_series(basis, res_h, out.probes, hr["q"], hr["c"])
        hr["reaction"] = hr["q"] @ basis.rf_sum + (hr["c"] @ res_h["RF"] if res_h is not None and hr["c"].shape[1]
                                                   else 0.0)
        hr["labels"] = [_exc_label(it) for it in terms.items]
        _peak_probe(out, basis, res_h, hr, True)
        out.harmonic = hr
    # ---------------------------------------------------------------- transient response
    if "transient" in groups:
        tr_spec = ds.transient
        terms, idx = groups["transient"]
        t = np.arange(0.0, tr_spec.t_end + 0.5 * tr_spec.dt, tr_spec.dt)
        Tmin = 1.0 / basis.freq.max()
        if tr_spec.dt > Tmin / 20:
            warnings.append(f"Transient response: step {tr_spec.dt:.3g} s > T_min/20 = {Tmin / 20:.3g} s of the highest mode – the solution of the modal equations is exact, but maxima between output times may be missed")
        log(f"Dynamics: transient response ({len(t)} steps, {basis.K} modes)")
        P, C, ab, ub = R.time_terms(terms, t)
        from .modal_ode import integrate
        impulses = R.base_disp_impulses(terms)
        q, v, a = integrate(tr_spec.dt, P, basis.omega, zeta, impulses=impulses)
        tr = {"t": t, "q": q, "qdd": a, "c": C if out.residual is not None else np.zeros((len(t), 0)),
              "ab": ab, "ub": ub, "res_index": np.asarray(idx if out.residual is not None else [], np.int64)}
        res_t = _subset(out.residual, tr["res_index"])
        stride = max(1, int(np.ceil(len(t) / 4000)))
        tr["t_env"] = t[::stride]
        tr.update(_env(basis, res_t, q[::stride], tr["c"][::stride], False))
        tr["probes"] = R.probe_series(basis, res_t, out.probes, q, tr["c"], a)
        for op in out.probes:                          # absolute acceleration of the probe = modal + base
            tr["probes"][op["name"]]["A"] = tr["probes"][op["name"]]["A"] + ab
        tr["reaction"] = q @ basis.rf_sum + (tr["c"] @ res_t["RF"] if res_t is not None and tr["c"].shape[1] else 0.0)
        tr["labels"] = [_exc_label(it) for it in terms.items]
        if tr_spec.settle_tol:
            tr["settle"] = {name: R.settle_time(t, s["U"], float(tr_spec.settle_tol))
                            for name, s in tr["probes"].items()}
        _peak_probe(out, basis, res_t, tr, False)
        out.transient = tr
    # ---------------------------------------------------------------- random vibrations, shock
    if ds.random.enabled:
        from .random_run import run_random
        out.random = run_random(study, mesh, basis, zeta, out, log, warnings)
    if ds.shock.enabled:
        from .shock_run import run_shock
        out.shock = run_shock(study, mesh, basis, zeta, out, workdir, log, warnings, on_line, cancel,
                              static=mm_static)
    if ds.harmonic.enabled and ds.harmonic.sweep_rate and out.harmonic is not None:
        try:
            from .fatigue_dyn import sweep_fatigue
            out.harmonic["sweep"] = sweep_fatigue(study, mesh, out)
        except ImportError:
            pass
    if pr.enabled and pr.rotating and pr.rpm_list:
        from .rotor import campbell
        out.campbell = campbell(study, mesh, workdir, log, on_line, cancel)
    if ds.verify_ccx:
        from .etalon import run_etalon
        out.info["etalon"] = run_etalon(study, mesh, basis, zeta, out, workdir, log, on_line, cancel)
    out.warnings = warnings
    return out


def _subset(res: dict | None, idx: np.ndarray) -> dict | None:
    if res is None or not len(idx):
        return None
    return {"U": res["U"][idx], "S": res["S"][idx], "RF": res["RF"][idx]}


def _env(basis, res, Q, C, cplx) -> dict:
    e = R.envelopes(basis, res, Q, C if res is not None else np.zeros((len(Q), 0)), cplx)
    return {"env_U": e["U"], "env_S": e["S"], "env_U_at": e["U_at"], "env_S_at": e["S_at"]}


def _exc_label(it) -> str:
    e = it["exc"]
    if e.kind == "force":
        return f"force – case {e.case}"
    d = it["dir"]
    return ("acceleration" if e.kind == "base_acc" else "displacement") + " of the base in direction " + \
        ", ".join(f"{v:.3g}" for v in d)


def _peak_probe(out, basis, res, rr: dict, cplx: bool) -> None:
    """Probe "max" in the node with the largest envelope of the stress (the FRF / course even without the given probes)."""
    n = int(np.argmax(rr["env_S"]))
    op = {"name": "max σ (uzel %d)" % (n + 1), "nodes": np.array([n]), "w": np.array([1.0]),
          "point": basis.shapes.shape and None, "quantity": "S", "auto": True}
    op["point"] = None
    s = R.probe_series(basis, res, [op], rr["q"], rr["c"], rr.get("qdd"))
    if "qdd" in rr:
        s[op["name"]]["A"] = s[op["name"]]["A"] + rr["ab"]
    rr["probes"].update(s)
    rr["peak_node"] = n
