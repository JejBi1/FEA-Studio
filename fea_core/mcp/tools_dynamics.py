"""Tools: structural dynamics (F8, fea_core.dynamics) – modal basis from CalculiX and modal superposition in Python:
harmonic response (FRF, sine sweep), transient response (decay, settling time), random vibration (PSD,
RMS, 3σ, spectral fatigue), shock (SRS, SRSS/CQC, missing mass), prestressed modal analysis; added mass and probes.

Units in the names of the parameters: _Hz, _g (acceleration in g = 9810 mm/s²), _kg, _s, _mm. The settings are stored in
Study.dynamics and computed in fea_solve_run / fea_job_start(kind='solve') on the same mesh as the statics.
DAMPING IS AN UNCERTAIN INPUT (sensitivity ±50 %) – state this reservation with the dynamic results.
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
from mcp.server.mcpserver import Image

from .common import guarded_selection, out
from .f8_common import (PRESET_NOTE, PROFILE_PRESETS, dof_of, excitation_text, mass_describe, parse_excitation,
                        sine_profile_table, table2, vec3)
from .session import FeaError, Session

G = 9810.0
DAMPING_TYPICAL = [
    {"construction": "bolted steel structure", "zeta": "≈ 0,8 %", "source": "EN 1991-1-4 tab. F.2"},
    {"construction": "welded steel", "zeta": "≈ 0,3 %", "source": "EN 1991-1-4 tab. F.2 (UNVERIFIED)"},
    {"construction": "aluminium, monolith / assembly", "zeta": "0,1–0,3 % / 1–3 %", "source": "Harris (UNVERIFIED)"},
    {"construction": "PLA FDM (small beams)", "zeta": "1–2 % (max 1,63 %)", "source": "J. Vib. Control 2025"},
    {"construction": "populated PCB", "zeta": "1/(2√f_n): 5 % at 100 Hz, 2.5 % at 400 Hz",
     "source": "Steinberg 2000 (UNVERIFIED from the original)"},
    {"construction": "fixtures (bolts, guides)", "zeta": "2–5 %", "source": "estimate – measure with an impact hammer"},
    {"construction": "konvence SRS / Miles", "zeta": "5 % (Q = 10)", "source": "konvence norem"},
]
DAMPING_WARNING = ("Damping is an uncertain input (sensitivity ±50 %) – peaks at resonance are proportional to 1/ζ; measuring it is recommended (−3 dB bandwidth: ζ ≈ Δf/(2f_n), logarithmic decrement) or compute bounds.")
RENDER_FIELDS = ("harmonic_at", "harmonic_envelope", "transient_at", "transient_envelope", "rms", "3sigma", "damage",
                 "life", "sweep_damage", "shock", "mode")


def dyn_describe(st) -> dict:
    ds = st.dynamics
    d = {"enabled": ds.enabled, "active": ds.active, "modes": ds.modes, "f_max_Hz": ds.f_max,
         "verify_ccx": ds.verify_ccx, "damping": asdict(ds.damping)}
    h = ds.harmonic
    if h.enabled:
        d["harmonic"] = {"f_min_Hz": h.f_min, "f_max_Hz": h.f_max, "points": h.points, "bias": h.bias,
                         "excitation": [excitation_text(e) for e in h.excitation], "sweep_rate_oct_min": h.sweep_rate,
                         "sweeps": h.sweeps, "static_correction": h.static_correction}
    t = ds.transient
    if t.enabled:
        d["transient"] = {"t_end_s": t.t_end, "dt_s": t.dt, "method": t.method, "settle_tol_mm": t.settle_tol,
                          "excitation": [excitation_text(e) for e in t.excitation], "fatigue": t.fatigue}
    r = ds.random
    if r.enabled:
        d["random"] = {"asd_g2_Hz": r.asd, "dof": r.dof, "direction": r.direction, "duration_h": r.duration_h,
                       "method": r.method}
        try:
            from ..dynamics.psd import asd_rms
            d["random"]["grms_g"] = asd_rms(r.asd)
        except Exception:  # noqa: BLE001 – invalid ASD (reported by the check) / numba missing
            pass
    s = ds.shock
    if s.enabled:
        d["shock"] = {"pulse": s.pulse, "A_g": s.A, "tau_s": s.tau, "rise_s": s.rise, "dof": s.dof,
                      "direction": s.direction, "combine": s.combine, "missing_mass": s.missing_mass,
                      "transient": s.transient, **({"srs_table": s.srs_table} if s.srs_table else {})}
    p = ds.prestress
    if p.enabled:
        d["prestress"] = asdict(p)
    d["probes"] = [asdict(x) for x in ds.probes]
    d["masses"] = [mass_describe(m) for m in st.masses]
    return d


def _result(sess: Session):
    res = sess.require_results()
    dr = getattr(res, "dynamics", None)
    if dr is None:
        raise FeaError("The results contain no dynamics – fea_dyn_settings(enabled=True) + the response kind (fea_dyn_harmonic_set / transient / random / shock), then fea_solve_run or fea_job_start")
    return res, dr


def _check(sess: Session) -> tuple[list[str], list[str]]:
    from ..dynamics.check import check
    return check(sess.study, sess.part_names() if not sess.frame_only else [])


def _done(sess: Session, extra: dict | None = None) -> str:
    sess.changed()
    st = sess.study
    e, w = _check(sess)
    d = {"dynamics": dyn_describe(st), "errors": e, "warnings": w}
    if extra:
        d.update(extra)
    if st.dynamics.active:
        d["next"] = "fea_check_readiness → fea_job_start(kind='solve') (nebo fea_solve_run) → fea_dyn_results"
    return out(d)


def _base_exc(dof, direction, base_acc_g, sine_profile, f_min, f_max) -> list[dict] | None:
    if base_acc_g is None and sine_profile is None:
        return None
    item = {"kind": "base_acc", "unit": "g"}
    if direction is not None:
        item["direction"] = direction
    else:
        item["dof"] = dof if dof is not None else 3
    if sine_profile is not None:
        item["sine_profile"] = {"f_min_Hz": f_min, "f_max_Hz": f_max, **sine_profile}
    else:
        item["scale"] = float(base_acc_g)
    return [item]


def _probe_names(dr) -> list[str]:
    names = [p["name"] for p in dr.probes]
    for rr in (dr.harmonic, dr.transient):
        if rr is not None:
            names += [k for k in rr.get("probes", {}) if k not in names]
    return names


def _pick_kind(dr, kind):
    avail = [k for k in ("harmonic", "transient", "random", "shock") if getattr(dr, k) is not None]
    if not avail:
        raise FeaError("The dynamics results contain only the modal basis (no response)")
    if kind is None:
        return avail[0]
    if kind not in avail:
        raise FeaError(f"The results do not contain '{kind}' (they have: {', '.join(avail)})")
    return kind


def _probe(rr: dict, probe: str | None, names: list[str]):
    pr = rr.get("probes") or {}
    if not pr:
        raise FeaError("The result has no probes – fea_dyn_probe_add(name, point=…) and calculate again")
    if probe is None:
        probe = next(iter(pr))
    if probe not in pr:
        raise FeaError(f"Probe '{probe}' is not in the result (probes: {', '.join(pr)})")
    return probe, pr[probe]


def _stride(n: int, max_points: int, keep: list[int]) -> np.ndarray:
    idx = np.arange(0, n, max(1, int(math.ceil(n / max(max_points, 2)))))
    return np.unique(np.concatenate([idx, np.asarray([k for k in keep if 0 <= k < n], int), [n - 1]]))


def _vm(S) -> np.ndarray:
    from ..dynamics.response import P_VM
    S = np.asarray(S, float)
    return np.sqrt(np.maximum(np.einsum("ti,ij,tj->t", S, P_VM, S), 0.0))


def probe_series(dr, kind: str, probe: str | None, quantity: str) -> dict:
    """Series in a probe: {"x", "x_label", "y" (n, 3) | (n,), "complex", "unit", "label", "probe"}."""
    from ..dynamics.response import max_over_phase, P_VM
    rr = getattr(dr, kind)
    names = _probe_names(dr)
    q = quantity.upper()
    if kind in ("harmonic", "transient"):
        name, s = _probe(rr, probe, names)
        x = np.asarray(rr["f"] if kind == "harmonic" else rr["t"], float)
        cplx = kind == "harmonic"
        base = bool(np.any(np.abs(np.asarray(rr.get("ub", 0.0))) > 0) or np.any(np.abs(np.asarray(rr.get("ab", 0.0))) > 0))
        if q == "U":
            y, unit, lab = np.asarray(s["U"]), "mm", "displacement relative to the base" if base else "displacement"
        elif q == "A":
            if cplx:
                w = 2 * np.pi * x
                y, unit, lab = -(w ** 2)[:, None] * (np.asarray(s["U"]) + np.asarray(rr["ub"])) / G, "g", \
                    "absolute acceleration"
            else:
                y, unit, lab = np.asarray(s["A"], float) / G, "g", "absolute acceleration"
        elif q == "S":
            S = np.asarray(s["S"])
            y = max_over_phase(S.real, -S.imag, P_VM) if cplx else _vm(S)
            return {"x": x, "y": y, "complex": False, "unit": "MPa", "label": "von Mises" + (" (amplituda)" if cplx else ""),
                    "probe": name, "x_label": "frequency [Hz]" if cplx else "time [s]"}
        else:
            raise FeaError("quantity: U | A | S")
        return {"x": x, "y": y, "complex": cplx, "unit": unit, "label": lab, "probe": name,
                "x_label": "frequency [Hz]" if cplx else "time [s]"}
    if kind == "random":
        name, s = _probe(rr, probe, names)
        if q == "S":
            raise FeaError("Random vibration: the probe has only the PSD of displacement (U) and acceleration (A)")
        y = np.asarray(s["A_psd_g2"] if q == "A" else s["U_psd"], float)
        return {"x": np.asarray(rr["f"], float), "y": y, "complex": False, "psd": True,
                "unit": "g²/Hz" if q == "A" else "mm²/Hz", "label": "PSD " + ("acceleration" if q == "A" else "displacement"),
                "probe": name, "x_label": "frequency [Hz]"}
    tm = rr.get("time")
    if not tm:
        raise FeaError("Shock: the time history is not computed (fea_dyn_shock_set(transient=True))")
    name, s = _probe(tm, probe, names)
    x = np.asarray(tm["t"], float)
    if q == "U":
        y, unit, lab = np.asarray(s["U"], float), "mm", "displacement relative to the base"
    elif q == "A":
        y, unit, lab = np.asarray(s["A"], float) / G, "g", "absolute acceleration"
    else:
        return {"x": x, "y": _vm(s["S"]), "complex": False, "unit": "MPa", "label": "von Mises", "probe": name,
                "x_label": "time [s]"}
    return {"x": x, "y": y, "complex": False, "unit": unit, "label": lab, "probe": name, "x_label": "time [s]"}


def modes_table(dr, n: int = 50) -> list[dict]:
    m = dr.model_mass or 0.0
    rows, cum = [], np.zeros(3)
    for j in range(min(dr.K, n)):
        fr = dr.eff_mass[j, :3] / m if m > 0 else np.zeros(3)
        cum = cum + fr
        rows.append({"mode": j + 1, "f_Hz": float(dr.freq[j]), "zeta": float(dr.zeta[j]),
                     "eff_mass_fraction_xyz": fr, "cumulative_xyz": cum.copy()})
    return rows


def render_values(dr, field: str, quantity: str, frequency_Hz=None, time_s=None, mode: int = 1):
    """(values (N,), U for deformation | None, description, unit, source)."""
    from ..dynamics.response import P_VM
    disp = quantity in ("displacement", "u", "displacement", "posuv")
    if field == "mode":
        if not 1 <= int(mode) <= dr.K:
            raise FeaError(f"mode must be 1–{dr.K}")
        U = np.asarray(dr.shapes[int(mode) - 1], float)
        S = np.asarray(dr.stress[int(mode) - 1], float)
        vals = np.linalg.norm(U, axis=1) if disp else np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", S, P_VM, S), 0))
        return vals, U, ("Mode |u| (mass-normalized)" if disp else "Modal von Mises stress (mass-normalized)"), \
            "–", f"mode {int(mode)}: f = {dr.freq[int(mode) - 1]:.5g} Hz"
    if field in ("harmonic_at", "harmonic_envelope", "sweep_damage"):
        h = dr.harmonic
        if h is None:
            raise FeaError("The results have no harmonic response")
        if field == "sweep_damage":
            if "sweep" not in h:
                raise FeaError("Sweep fatigue is not computed (sweep_rate_oct_min > 0 and the material S-N)")
            return np.log10(np.maximum(np.asarray(h["sweep"]["D"], float), 1e-12)), None, \
                "Sweep fatigue – log10 D", "log10 D", f"{h['sweep']['summary'].get('sweeps', 1):g} passes"
        f = np.asarray(h["f"], float)
        if field == "harmonic_envelope":
            vals = np.asarray(h["env_U"] if disp else h["env_S"], float)
            return vals, None, ("Displacement amplitude envelope" if disp else "von Mises amplitude envelope"), \
                ("mm" if disp else "MPa"), f"harmonic response {f[0]:.4g}–{f[-1]:.4g} Hz (max over frequency)"
        if frequency_Hz is None:
            i = int(np.asarray(h["env_S_at"])[int(np.argmax(h["env_S"]))])
        else:
            i = int(np.argmin(np.abs(f - float(frequency_Hz))))
        um, vm = dr.harmonic_amplitude(i)
        Uc, _ = dr.harmonic_complex(i, stress=False)
        ph = max(range(0, 360, 15), key=lambda a: float(np.linalg.norm((Uc * np.exp(1j * np.deg2rad(a))).real,
                                                                       axis=1).max()))
        U = (Uc * np.exp(1j * np.deg2rad(ph))).real
        return (np.asarray(um if disp else vm, float), U,
                ("Displacement amplitude" if disp else "von Mises amplitude (max over phase)"), ("mm" if disp else "MPa"),
                f"harmonic response at f = {f[i]:.5g} Hz")
    if field in ("transient_at", "transient_envelope"):
        tr = dr.transient
        if tr is None:
            raise FeaError("The results have no transient response")
        if field == "transient_envelope":
            vals = np.asarray(tr["env_U"] if disp else tr["env_S"], float)
            return vals, None, ("Displacement envelope" if disp else "von Mises envelope"), ("mm" if disp else "MPa"), \
                f"transient response 0–{tr['t'][-1]:.4g} s (max over time)"
        t = np.asarray(tr["t"], float)
        if time_s is None:
            te = np.asarray(tr["t_env"], float)
            time_s = float(te[int(np.asarray(tr["env_S_at"])[int(np.argmax(tr["env_S"]))])])
        i = int(np.argmin(np.abs(t - float(time_s))))
        U, S = dr.transient_field(i)
        vals = np.linalg.norm(U, axis=1) if disp else np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", S, P_VM, S), 0))
        return vals, U, ("Displacement |u|" if disp else "von Mises"), ("mm" if disp else "MPa"), \
            f"transient response at time t = {t[i]:.5g} s"
    if field in ("rms", "3sigma", "damage", "life"):
        r = dr.random
        if r is None:
            raise FeaError("The results have no random vibration")
        if field == "damage":
            return np.log10(np.maximum(np.asarray(r["D"], float), 1e-12)), None, "Spectral fatigue – log10 D", \
                "log10 D", f"random vibration, {r['summary'].get('duration_h', 0):g} h, {r['summary'].get('method')}"
        if field == "life":
            return np.log10(np.minimum(np.maximum(np.asarray(r["life_h"], float), 1e-6), 1e12)), None, \
                "Life – log10 hours", "log10 h", "random vibration"
        k = 3.0 if field == "3sigma" else 1.0
        vals = (np.linalg.norm(np.asarray(r["U_rms"], float), axis=1) if disp else np.asarray(r["S_rms"], float)) * k
        lab = ("3σ " if k == 3 else "RMS ") + ("displacement" if disp else "von Mises (Segalman, informative)")
        return vals, None, lab, ("mm" if disp else "MPa"), "random vibration"
    if field == "shock":
        s = dr.shock
        if s is None:
            raise FeaError("The results have no shock")
        vals = np.linalg.norm(np.asarray(s["U"], float), axis=1) if disp else np.asarray(s["mises"], float)
        return vals, None, ("Displacement |u| – shock" if disp else "von Mises – shock (SRSS/CQC)"), ("mm" if disp else "MPa"), \
            f"shock {s['summary'].get('pulse')} {s['summary'].get('A_g', 0):g} g"
    raise FeaError(f"field: {' | '.join(RENDER_FIELDS)}")


def _time_table(shape: str, tau: float) -> list[list[float]]:
    """Normalised pulse (peak 1) as a table [[t, k], …] for the transient response."""
    if shape == "halfsine":
        t = np.linspace(0.0, tau, 41)
        return [[float(a), float(b)] for a, b in zip(t, np.sin(np.pi * t / tau))] + [[tau * 1.0001, 0.0]]
    if shape == "rect":
        return [[0.0, 1.0], [tau, 1.0], [tau * 1.0001, 0.0]]
    if shape == "sawtooth":
        return [[0.0, 0.0], [tau, 1.0], [tau * 1.0001, 0.0]]
    raise FeaError("pulse.shape: halfsine | rect | sawtooth")


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_dyn_settings(enabled: bool = True, modes: int | None = None, f_max_Hz: float | None = None,
                         verify_ccx: bool | None = None, max_frames: int | None = None) -> str:
        """Dynamics on/off and the modal basis (only `enabled`: lists settings). modes (cover 1.5-2x the highest excitation frequency and sum m_eff >= 90 % in the excited direction - checked), f_max_Hz, verify_ccx (extra ccx reference: *STEADY STATE DYNAMICS / *MODAL DYNAMIC at probes), max_frames (transient field snapshots). Mass from rho + added masses (fea_dyn_mass_add). Responses: fea_dyn_harmonic_set / transient_set / random_set / shock_set; damping fea_dyn_damping_set; probes fea_dyn_probe_add. Run with fea_job_start(kind='solve') (recommended) or fea_solve_run."""
        with sess.use():
            st = sess.require_study()
            ds = st.dynamics
            if modes is not None:
                if int(modes) != modes or not 1 <= int(modes) <= 500:
                    raise FeaError("modes must be an integer 1–500")
                ds.modes = int(modes)
            if f_max_Hz is not None:
                if not (math.isfinite(f_max_Hz) and f_max_Hz > 0):
                    raise FeaError("f_max_Hz must be positive")
                ds.f_max = float(f_max_Hz)
            if verify_ccx is not None:
                ds.verify_ccx = bool(verify_ccx)
            if max_frames is not None:
                if not 1 <= int(max_frames) <= 1000:
                    raise FeaError("max_frames must be 1–1000")
                ds.max_frames = int(max_frames)
            ds.enabled = bool(enabled)
            return _done(sess, {"damping_warning": DAMPING_WARNING})

    @mcp.tool(structured_output=False)
    def fea_dyn_damping_set(kind: str = "modal", zeta: float = 0.02, eta: float | None = None,
                            table: list[list[float]] | None = None, f1_Hz: float | None = None,
                            f2_Hz: float | None = None, per_material: dict | None = None) -> str:
        """Modal-system damping. kind modal (zeta of all modes: zeta or table [[f Hz, zeta], ...]) | rayleigh (alpha*M + beta*K, zeta at f1_Hz and f2_Hz) | structural (loss factor eta, harmonic only; None = 2*zeta; ccx *DAMPING STRUCTURAL takes eta, not zeta) | per_material {material: zeta} (modal-strain-energy weighted). Also returns indicative zeta by structure type with sources. Damping is UNCERTAIN - resonance response ~ 1/zeta."""
        if kind not in ("modal", "rayleigh", "structural", "per_material"):
            raise FeaError("kind: modal | rayleigh | structural | per_material")
        if not (math.isfinite(zeta) and 0 <= zeta < 1):
            raise FeaError("zeta must satisfy 0 ≤ ζ < 1 (e.g. 0.02 = 2 %)")
        with sess.use():
            st = sess.require_study()
            dmp = st.dynamics.damping
            tab = table2(table, "damping table", positive_x=True) if table else []
            if any(not 0 <= z < 1 for _, z in tab):
                raise FeaError("ζ in the table must satisfy 0 ≤ ζ < 1")
            if eta is not None and not (math.isfinite(eta) and eta > 0):
                raise FeaError("eta must be positive")
            if per_material is not None:
                bad = [m for m in per_material if m not in st.materials]
                if bad:
                    raise FeaError(f"Materials {', '.join(bad)} are not in the study")
                dmp.per_material = {str(k): float(v) for k, v in per_material.items()}
            dmp.kind, dmp.zeta, dmp.table = kind, float(zeta), tab
            dmp.eta = float(eta) if eta is not None else None
            dmp.f1 = float(f1_Hz) if f1_Hz is not None else None
            dmp.f2 = float(f2_Hz) if f2_Hz is not None else None
            return _done(sess, {"typical_zeta": DAMPING_TYPICAL, "damping_warning": DAMPING_WARNING})

    @mcp.tool(structured_output=False)
    def fea_dyn_harmonic_set(enabled: bool = True, f_min_Hz: float | None = None, f_max_Hz: float | None = None,
                             points: int | None = None, bias: float | None = None, excitation: list[dict] | None = None,
                             base_acc_g: float | None = None, dof: int | str | None = None,
                             direction: list[float] | None = None, sine_profile: dict | None = None,
                             sweep_rate_oct_min: float | None = None, sweeps: float | None = None,
                             static_correction: bool | None = None) -> str:
        """Harmonic (steady-state) response, FRF, f_min_Hz-f_max_Hz; points between natural frequencies (denser at resonances, bias). excitation = [{"kind": force (case x profile) | base_acc (unit g | mm/s2) | base_disp (mm), "dof": 1-3|"x", "direction": [x,y,z], "table": [[f Hz, multiple], ...], "scale", "phase_deg", "sine_profile"}], or shortcut base_acc_g + dof/direction, or sine_profile {"displacement_mm", "acceleration_g"} (IEC 60068-2-6: constant displacement below the crossover, acceleration above; values from the standard). sweep_rate_oct_min + sweeps = log-sweep fatigue (dn/df = 60/(R*ln 2)). static_correction = residual of higher modes. Results: fea_dyn_frf_get, fea_dyn_plot(kind='frf'), fea_dyn_render."""
        with sess.use():
            st = sess.require_study()
            h = st.dynamics.harmonic
            for v, nm in ((f_min_Hz, "f_min_Hz"), (f_max_Hz, "f_max_Hz"), (bias, "bias")):
                if v is not None and not (math.isfinite(v) and v >= 0):
                    raise FeaError(f"{nm} must be ≥ 0")
            f_lo = float(f_min_Hz) if f_min_Hz is not None else h.f_min
            f_hi = float(f_max_Hz) if f_max_Hz is not None else h.f_max
            if not 0 <= f_lo < f_hi:
                raise FeaError("0 ≤ f_min_Hz < f_max_Hz must hold")
            h.f_min, h.f_max = f_lo, f_hi
            if points is not None:
                if not 2 <= int(points) <= 500:
                    raise FeaError("points must be 2–500")
                h.points = int(points)
            if bias is not None:
                h.bias = float(bias)
            items = excitation if excitation is not None else _base_exc(dof, direction, base_acc_g, sine_profile,
                                                                        max(h.f_min, 1e-3), h.f_max)
            info = {}
            if items is not None:
                cases = [lc.name for lc in st.load_cases]
                h.excitation = [parse_excitation(it, cases, "harmonic") for it in items]
                if sine_profile is not None:
                    _, info = sine_profile_table(max(h.f_min, 1e-3), h.f_max, sine_profile.get("displacement_mm"),
                                                 sine_profile.get("acceleration_g"), sine_profile.get("segments"))
            if sweep_rate_oct_min is not None:
                if not sweep_rate_oct_min >= 0:
                    raise FeaError("sweep_rate_oct_min must be ≥ 0 (0 = no sweep fatigue)")
                h.sweep_rate = float(sweep_rate_oct_min)
            if sweeps is not None:
                h.sweeps = float(sweeps)
            if static_correction is not None:
                h.static_correction = bool(static_correction)
            h.enabled = bool(enabled)
            if enabled:
                st.dynamics.enabled = True
                if not h.excitation:
                    raise FeaError("A harmonic response needs an excitation (excitation, base_acc_g or sine_profile)")
            return _done(sess, {"sine_profile": info} if info else None)

    @mcp.tool(structured_output=False)
    def fea_dyn_transient_set(enabled: bool = True, t_end_s: float | None = None, dt_s: float | None = None,
                              method: str | None = None, excitation: list[dict] | None = None,
                              base_acc_g: float | None = None, dof: int | str | None = None,
                              direction: list[float] | None = None, pulse: dict | None = None,
                              settle_tol_mm: float | None = None, fatigue: bool | None = None) -> str:
        """Transient response 0-t_end_s, step dt_s (<= T_min/20 of the highest mode). method modal (exact, default) | direct (ccx *DYNAMIC, ~25x slower). excitation = [{"kind": force|base_acc|base_disp, "case", "dof", "direction", "table": [[t s, multiple], ...] (0 after the table end; step = [[0,1],[t_end,1]]), "unit", "scale"}], or shortcut base_acc_g + dof + pulse {"shape": halfsine|rect|sawtooth, "tau_s"}. settle_tol_mm: settling-time tolerance at probes. fatigue: fatigue from the history. Results: fea_dyn_frf_get(kind='transient'), fea_dyn_plot(kind='transient'), fea_dyn_render(field='transient_envelope')."""
        with sess.use():
            st = sess.require_study()
            t = st.dynamics.transient
            te = float(t_end_s) if t_end_s is not None else t.t_end
            dt = float(dt_s) if dt_s is not None else t.dt
            if not (te > 0 and 0 < dt < te):
                raise FeaError("0 < dt_s < t_end_s must hold")
            t.t_end, t.dt = te, dt
            if method is not None:
                if method not in ("modal", "direct"):
                    raise FeaError("method: modal | direct")
                t.method = method
            items = excitation
            if items is None and base_acc_g is not None:
                it = {"kind": "base_acc", "unit": "g", "scale": float(base_acc_g)}
                if direction is not None:
                    it["direction"] = direction
                else:
                    it["dof"] = dof if dof is not None else 3
                if pulse is not None:
                    it["table"] = _time_table(str(pulse.get("shape", "halfsine")), float(pulse.get("tau_s", 0.0)) or
                                              t.t_end / 10)
                items = [it]
            if items is not None:
                cases = [lc.name for lc in st.load_cases]
                t.excitation = [parse_excitation(x, cases, "transient") for x in items]
            if settle_tol_mm is not None:
                t.settle_tol = float(settle_tol_mm) if settle_tol_mm > 0 else None
            if fatigue is not None:
                t.fatigue = bool(fatigue)
            t.enabled = bool(enabled)
            if enabled:
                st.dynamics.enabled = True
                if not t.excitation:
                    raise FeaError("A transient response needs an excitation (excitation or base_acc_g)")
            return _done(sess)

    @mcp.tool(structured_output=False)
    def fea_dyn_random_set(enabled: bool = True, asd_g2_Hz: list[list[float]] | None = None,
                           points: list[list] | None = None, slopes_db_oct: list | None = None,
                           grms_g: float | None = None, dof: int | str | None = None,
                           direction: list[float] | None = None, duration_h: float | None = None,
                           method: str | None = None) -> str:
        """Random vibration (base-acceleration PSD, IEC 60068-2-64): asd_g2_Hz [[f Hz, g2/Hz], ...] (log-log) or standard notation points [[f, W|null], ...] + slopes_db_oct (computes missing W); grms_g rescales to total g_rms; dof/direction (one axis per run); duration_h per axis; method nb | dirlik | tb (spectral fatigue). Results: RMS and 3 sigma of displacement/stress, damage D, life (fea_dyn_render field rms|3sigma|damage|life), probe PSD (fea_dyn_plot kind='psd'). Take ASD values from the standard/specification."""
        with sess.use():
            st = sess.require_study()
            r = st.dynamics.random
            info = {}
            if asd_g2_Hz is not None or points is not None:
                try:
                    from ..dynamics.psd import asd_from_slopes, asd_rms
                except ImportError as e:
                    raise FeaError(f"Random vibration needs the accel module (numba): {e}") from None
                if asd_g2_Hz is not None:
                    tab = table2(asd_g2_Hz, "asd_g2_Hz", positive_x=True)
                    if any(w <= 0 for _, w in tab) or len(tab) < 2:
                        raise FeaError("The ASD needs ≥ 2 points with a positive g²/Hz value")
                    if grms_g is not None:
                        k = (float(grms_g) / asd_rms(tab)) ** 2
                        tab = [[f, w * k] for f, w in tab]
                else:
                    try:
                        tab = asd_from_slopes(points, slopes_db_oct, grms_g).tolist()
                    except ValueError as e:
                        raise FeaError(f"Profil ASD: {e}") from None
                r.asd = tab
                info["grms_g"] = asd_rms(tab)
            if direction is not None:
                r.direction = vec3(direction, "direction")
            elif dof is not None:
                r.dof, r.direction = dof_of(dof), None
            if duration_h is not None:
                if not duration_h > 0:
                    raise FeaError("duration_h must be positive")
                r.duration_h = float(duration_h)
            if method is not None:
                if method not in ("nb", "dirlik", "tb"):
                    raise FeaError("method: nb | dirlik | tb")
                r.method = method
            r.enabled = bool(enabled)
            if enabled:
                st.dynamics.enabled = True
                if len(r.asd) < 2:
                    raise FeaError("Enter the ASD (asd_g2_Hz or points + slopes_db_oct)")
            return _done(sess, info)

    @mcp.tool(structured_output=False)
    def fea_dyn_shock_set(enabled: bool = True, pulse: str | None = None, A_g: float | None = None,
                          tau_s: float | None = None, rise_s: float | None = None, dof: int | str | None = None,
                          direction: list[float] | None = None, srs_table: list[list[float]] | None = None,
                          combine: str | None = None, missing_mass: bool | None = None,
                          transient: bool | None = None) -> str:
        """Shock (IEC 60068-2-27) base-acceleration pulse: pulse halfsine | sawtooth (peak at end) | trapezoid (rise_s) | srs_table [[f Hz, S_a g], ...]; A_g peak [g], tau_s duration, dof/direction. Response by SRS: mode peaks combined by cqc | srss; missing_mass = ZPA static state; transient = also the exact transient response to the pulse. Take pulse values from the standard/specification."""
        with sess.use():
            st = sess.require_study()
            s = st.dynamics.shock
            if pulse is not None:
                if pulse not in ("halfsine", "sawtooth", "trapezoid", "srs_table"):
                    raise FeaError("pulse: halfsine | sawtooth | trapezoid | srs_table")
                s.pulse = pulse
            for v, nm in ((A_g, "A_g"), (tau_s, "tau_s"), (rise_s, "rise_s")):
                if v is not None and not (math.isfinite(v) and v > 0):
                    raise FeaError(f"{nm} must be positive")
            if A_g is not None:
                s.A = float(A_g)
            if tau_s is not None:
                s.tau = float(tau_s)
            if rise_s is not None:
                s.rise = float(rise_s)
            if direction is not None:
                s.direction = vec3(direction, "direction")
            elif dof is not None:
                s.dof, s.direction = dof_of(dof), None
            if srs_table is not None:
                s.srs_table = table2(srs_table, "srs_table", positive_x=True)
            if combine is not None:
                if combine not in ("cqc", "srss"):
                    raise FeaError("combine: cqc | srss")
                s.combine = combine
            if missing_mass is not None:
                s.missing_mass = bool(missing_mass)
            if transient is not None:
                s.transient = bool(transient)
            s.enabled = bool(enabled)
            if enabled:
                st.dynamics.enabled = True
            return _done(sess)

    @mcp.tool(structured_output=False)
    def fea_dyn_prestress_set(enabled: bool = True, case: str | None = None, nlgeom: bool = False,
                              rotating: bool = False, rpm_list: list[float] | None = None,
                              axis_point: list[float] | None = None, axis: list[float] | None = None,
                              coriolis: bool = True) -> str:
        """Prestressed modal analysis after the static state `case` (stress stiffening: tension, bolt preload, centrifugal; one preload state, no superposition). rotating + rpm_list [rpm] + axis_point/axis = Campbell diagram; nlgeom; coriolis."""
        with sess.use():
            st = sess.require_study()
            p = st.dynamics.prestress
            names = [lc.name for lc in st.load_cases]
            if enabled and case is not None and case not in names:
                raise FeaError(f"Load case '{case}' does not exist (cases: {', '.join(names) or 'none'})")
            p.enabled, p.case, p.nlgeom, p.rotating, p.coriolis = bool(enabled), case, bool(nlgeom), bool(rotating), bool(coriolis)
            if rpm_list is not None:
                p.rpm_list = [float(v) for v in rpm_list]
            if axis_point is not None:
                p.axis_point = [float(v) for v in axis_point]
            if axis is not None:
                p.axis = vec3(axis, "axis")
            if enabled:
                st.dynamics.enabled = True
            return _done(sess)

    @mcp.tool(structured_output=False)
    def fea_dyn_probe_add(name: str, point: list[float] | None = None, selection: str | list | dict | None = None,
                          quantity: str = "U") -> str:
        """Dynamics output probe (FRF, time history, PSD): point [mm] (shape-function interpolation) or selection (faces; nodal average). quantity U | S | A = default for plots (all are stored). The same name replaces. Without probes one is added at the largest stress-envelope node."""
        from ..dynamics.model import Probe
        if (point is None) == (selection is None):
            raise FeaError("Give exactly one of point / selection")
        if quantity.upper() not in ("U", "S", "A"):
            raise FeaError("quantity: U | S | A")
        with sess.use():
            st = sess.require_study()
            pr = Probe(name, quantity=quantity.upper())
            if point is not None:
                pr.point = vec3(point, "point") if any(point) else [0.0, 0.0, 0.0]
            else:
                if isinstance(selection, str) and selection in st.selections:
                    pr.selection = selection
                else:
                    old = [p for p in st.dynamics.probes if p.name == name]
                    pr.selection = guarded_selection(sess, selection, f"sonda {name}", old)
            ds = st.dynamics
            replaced = any(p.name == name for p in ds.probes)
            ds.probes = [p for p in ds.probes if p.name != name] + [pr]
            return _done(sess, {"probe": asdict(pr), "replaced": replaced})

    @mcp.tool(structured_output=False)
    def fea_dyn_probe_list() -> str:
        """Probes (name, point/selection, quantity)."""
        with sess.use():
            st = sess.require_study()
            return out({"probes": [asdict(p) for p in st.dynamics.probes]})

    @mcp.tool(structured_output=False)
    def fea_dyn_probe_remove(name: str) -> str:
        """Removes a probe ('*' = all)."""
        with sess.use():
            st = sess.require_study()
            ds = st.dynamics
            names = [p.name for p in ds.probes]
            if name != "*" and name not in names:
                raise FeaError(f"Probe '{name}' does not exist (probes: {', '.join(names) or 'none'})")
            ds.probes = [] if name == "*" else [p for p in ds.probes if p.name != name]
            return _done(sess, {"removed": names if name == "*" else [name]})

    @mcp.tool(structured_output=False)
    def fea_dyn_mass_add(name: str, selection: str | list | dict, mass_kg: float | None = None,
                         mass_per_area_kg_m2: float | None = None, point: list[float] | None = None,
                         inertia_kg_mm2: list | None = None, coupling: str = "rigid") -> str:
        """Added mass (motor, PCB mount, omitted part) on faces `selection`. Point mass: mass_kg [kg], point = centre of gravity [mm] (None = face centroid), inertia_kg_mm2 [Ixx,Iyy,Izz] or 3x3 about the CoG, coupling rigid (stiffens the faces) | distributing (RBE3, mass only). Surface mass: mass_per_area_kg_m2 spread over the area. Acts in statics (weight, rotation), frequencies, dynamics. The same name replaces."""
        from ..dynamics.model import PointMass, SurfaceMass
        if (mass_kg is None) == (mass_per_area_kg_m2 is None):
            raise FeaError("Give exactly one of mass_kg (point mass) / mass_per_area_kg_m2 (surface mass)")
        if coupling not in ("rigid", "distributing"):
            raise FeaError("coupling: rigid | distributing")
        with sess.use():
            st = sess.require_study()
            old = [m for m in st.masses if m.name == name]
            if isinstance(selection, str) and selection in st.selections:
                sel = selection
            else:
                sel = guarded_selection(sess, selection, name, old)     # the replaced mass may change the selection
            if mass_kg is not None:
                if not (math.isfinite(mass_kg) and mass_kg > 0):
                    raise FeaError("mass_kg must be positive")
                inertia = None
                if inertia_kg_mm2 is not None:
                    a = np.asarray(inertia_kg_mm2, float)
                    if a.shape not in ((3,), (3, 3)) or not np.all(np.isfinite(a)):
                        raise FeaError("inertia_kg_mm2 = [Ixx, Iyy, Izz] or a 3×3 matrix [kg·mm²]")
                    inertia = (a * 1e-3).tolist()                      # kg·mm² → t·mm²
                m = PointMass(sel, float(mass_kg) * 1e-3, point=vec3(point, "point") if point is not None else None,
                              inertia=inertia, coupling=coupling, name=name)
            else:
                if not (math.isfinite(mass_per_area_kg_m2) and mass_per_area_kg_m2 > 0):
                    raise FeaError("mass_per_area_kg_m2 must be positive")
                m = SurfaceMass(sel, float(mass_per_area_kg_m2), name=name)
            st.masses = [x for x in st.masses if x.name != name] + [m]
            sess.changed()
            from ..dynamics.mass import check
            e, w = check(st, sess.part_names())
            return out({"mass": mass_describe(m), "replaced": bool(old), "masses": len(st.masses), "errors": e,
                        "warnings": w})

    @mcp.tool(structured_output=False)
    def fea_dyn_mass_list() -> str:
        """Added masses (point [kg], surface [kg/m2])."""
        with sess.use():
            st = sess.require_study()
            return out({"masses": [mass_describe(m) for m in st.masses]})

    @mcp.tool(structured_output=False)
    def fea_dyn_mass_remove(name: str) -> str:
        """Removes an added mass ('*' = all); the face selection stays."""
        with sess.use():
            st = sess.require_study()
            names = [m.name for m in st.masses]
            if name != "*" and name not in names:
                raise FeaError(f"Mass '{name}' does not exist (masses: {', '.join(names) or 'none'})")
            st.masses = [] if name == "*" else [m for m in st.masses if m.name != name]
            sess.changed()
            return out({"removed": names if name == "*" else [name], "remaining": [m.name for m in st.masses]})

    @mcp.tool(structured_output=False)
    def fea_dyn_frf_get(probe: str | None = None, kind: str | None = None, quantity: str = "U",
                        max_points: int = 200, reaction: bool = False) -> str:
        """Probe response. kind: harmonic (FRF: frequency, amplitude, phase [deg], Re, Im per x,y,z, peak) | transient (history, peak, settling time) | random (PSD, RMS) | shock (history); None = first available. quantity U (relative to base [mm]) | A (absolute [g]) | S (von Mises [MPa]). probe (None = first; automatic 'max σ (uzel n)' at the largest stress envelope). reaction=True adds the summed constraint reaction [N] (harmonic/transient). max_points thins the series (peak kept)."""
        with sess.use():
            _, dr = _result(sess)
            kind = _pick_kind(dr, kind)
            s = probe_series(dr, kind, probe, quantity)
            x, y = s["x"], np.asarray(s["y"])
            mag = np.abs(y) if y.ndim == 1 else (np.linalg.norm(np.abs(y), axis=1) if s["complex"] else
                                                np.linalg.norm(y, axis=1))
            if s["complex"] and y.ndim == 2:
                from ..dynamics.response import max_over_phase
                mag = max_over_phase(y.real, -y.imag)
            k = int(np.argmax(mag))
            idx = _stride(len(x), int(max_points), [k])
            d = {"kind": kind, "probe": s["probe"], "quantity": s["label"], "unit": s["unit"],
                 "x_label": s["x_label"], "points": len(x), "x": x[idx]}
            if s.get("psd"):
                from ..dynamics.psd import trapz_weights
                w = trapz_weights(x)
                d["psd"] = y[idx]
                d["rms"] = np.sqrt(np.maximum(w @ y, 0.0))
                d["note"] = "RMS = √∫PSD df per component x, y, z; 3σ = 3 × RMS"
            elif s["complex"]:
                yy = y[idx]
                comps = {}
                for c, nm in enumerate("xyz"):
                    comps[nm] = {"amplitude": np.abs(yy[:, c]), "phase_deg": np.degrees(np.angle(yy[:, c])),
                                 "re": yy[:, c].real, "im": yy[:, c].imag}
                d["components"] = comps
                d["magnitude"] = mag[idx]
            else:
                d["y"] = y[idx]
            d["max"] = {"value": float(mag[k]), "at": float(x[k])}
            rr = getattr(dr, kind)
            if kind == "transient" and rr.get("settle"):
                d["settle_s"] = rr["settle"].get(s["probe"])
            if reaction and kind in ("harmonic", "transient"):
                R = np.asarray(rr["reaction"])
                if np.iscomplexobj(R):
                    d["reaction_amplitude_N"] = np.abs(R[idx])
                    d["reaction_max_N"] = np.abs(R).max(axis=0)
                else:
                    d["reaction_N"] = R[idx]
                    d["reaction_max_N"] = np.abs(R).max(axis=0)
            d["damping_warning"] = DAMPING_WARNING
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_dyn_results() -> str:
        """Dynamics summary: basis modes (f, zeta, effective mass shares x/y/z and cumulative), basis checks (mass normalization, Gamma, mode equilibrium, mass coverage), model mass, harmonic/transient maxima (where, at which f/t), random (RMS, 3 sigma, damage, life), shock, ccx reference (verify_ccx), warnings - always with the uncertain-damping caveat."""
        with sess.use():
            _, dr = _result(sess)
            d = dr.summary()
            d["modes_table"] = modes_table(dr)
            if dr.info.get("etalon"):
                d["etalon_ccx"] = dr.info["etalon"]
            d["damping_warning"] = DAMPING_WARNING
            d["next"] = ("fea_dyn_frf_get, fea_dyn_plot(kind='frf'|'transient'|'psd'|'srs'), "
                         "fea_dyn_render(field='harmonic_envelope'|'rms'|'damage'|…)")
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_dyn_render(field: str = "harmonic_envelope", quantity: str = "mises", frequency_Hz: float | None = None,
                       time_s: float | None = None, mode: int = 1, output_path: str | None = None,
                       view: str | list[float] = "iso", deformed: bool = True, scale: float | None = None,
                       width: int = 1280, height: int = 720, bands: int = 12, vmin: float | None = None,
                       vmax: float | None = None, show_mesh: bool = False, return_image: bool = True):
        """PNG of a dynamics field. field: harmonic_at (amplitude at frequency_Hz; None = frequency of max stress) | harmonic_envelope (max over frequency) | transient_at (time_s; None = time of max) | transient_envelope | rms | 3sigma | damage (log10 D) | life (log10 h) | sweep_damage | shock (SRSS/CQC) | mode (shape `mode` of the basis). quantity mises [MPa] | displacement [mm] (damage/life ignore it). view etc. as fea_results_render_image."""
        from .render import render
        from .f8_scenes import field_scene
        from .tools_results import _png_path
        if field not in RENDER_FIELDS:
            raise FeaError(f"field: {' | '.join(RENDER_FIELDS)}")
        q = str(quantity).strip().lower()
        if q not in ("mises", "displacement", "u", "displacement", "von_mises", "vm", "posuv"):
            raise FeaError("quantity: mises | displacement")
        q = "displacement" if q in ("displacement", "u", "displacement", "posuv") else "mises"
        if not 2 <= bands <= 24:
            raise FeaError("bands must be 2–24")
        with sess.use(op="render"):
            _, dr = _result(sess)
            vals, U, label, unit, src = render_values(dr, field, q, frequency_Hz, time_s, mode)
            info = [] if field in ("damage", "life", "sweep_damage", "mode") else \
                ["Damping is an uncertain input (sensitivity ±50 %)"]
            scene, meta = field_scene(sess, vals, label, unit, src, U=U if deformed else None, scale=scale,
                                      nbands=bands, vmin=vmin, vmax=vmax, show_mesh=show_mesh, info=info,
                                      mark_min=field == "life")
            p = _png_path(sess, output_path, f"dyn_{field}")
            meta.update(render(scene, p, view, width, height))
            meta["source"] = src
            txt = out(meta)
            return [txt, Image(path=p)] if return_image else txt

    @mcp.tool(structured_output=False)
    def fea_dyn_plot(kind: str = "frf", probe: str | None = None, quantity: str = "U", component: str = "all",
                     output_path: str | None = None, return_image: bool = True):
        """PNG graph. kind: frf (amplitude+phase at a probe, log f) | transient | psd (response PSD) | asd (input profile) | srs | pulse | shock_time | reaction (reaction amplitude vs f). quantity U | A | S; component all | x | y | z | mag."""
        from .f8_plots import frf_plot, xy_plot
        from .tools_results import _png_path
        comp = str(component).strip().lower()
        if comp not in ("all", "x", "y", "z", "mag"):
            raise FeaError("component: all | x | y | z | mag")
        with sess.use(op="graf"):
            _, dr = _result(sess)
            p = _png_path(sess, output_path, f"dyn_plot_{kind}")
            meta: dict = {"kind": kind}

            def comps(y, x, lab):
                y = np.asarray(y)
                if y.ndim == 1:
                    return [{"x": x, "y": y, "label": lab}]
                if comp == "mag":
                    if np.iscomplexobj(y):
                        from ..dynamics.response import max_over_phase
                        return [{"x": x, "y": max_over_phase(y.real, -y.imag), "label": f"|{lab}|"}]
                    return [{"x": x, "y": np.linalg.norm(y, axis=1), "label": f"|{lab}|"}]
                sel = "xyz" if comp == "all" else comp
                return [{"x": x, "y": y[:, "xyz".index(c)], "label": f"{lab} {c}"} for c in sel]
            if kind in ("frf", "transient", "psd", "shock_time"):
                src = {"frf": "harmonic", "transient": "transient", "psd": "random", "shock_time": "shock"}[kind]
                _pick_kind(dr, src)
                s = probe_series(dr, src, probe, quantity)
                ser = comps(s["y"], s["x"], s["label"])
                title = f"{s['label']} – sonda „{s['probe']}“"
                if kind == "frf" and s["complex"] and comp != "mag":
                    frf_plot(p, s["x"], [{"y": z["y"], "label": z["label"]} for z in ser], title, f"[{s['unit']}]")
                else:
                    xy_plot(p, [{**z, "y": np.abs(z["y"]) if np.iscomplexobj(z["y"]) else z["y"]} for z in ser],
                            title, s["x_label"], f"[{s['unit']}]", logx=kind in ("frf", "psd"),
                            logy=kind in ("frf", "psd"))
                meta["probe"] = s["probe"]
            elif kind == "asd":
                r = dr.random
                if r is None:
                    raise FeaError("The results have no random vibration")
                xy_plot(p, [{"x": r["f"], "y": r["asd_g2"], "label": "ASD"}],
                        f"Input ASD of the base acceleration (g_rms {r['summary'].get('g_rms_input', 0):.3g} g)",
                        "frequency [Hz]", "ASD [g²/Hz]", logx=True, logy=True)
            elif kind in ("srs", "pulse"):
                s = dr.shock
                if s is None or s.get("pulse_t") is None:
                    raise FeaError("The results have no shock pulse (pulse = srs_table has no time history)")
                if kind == "srs":
                    xy_plot(p, [{"x": s["srs_f"], "y": np.asarray(s["srs"]) / G, "label": "SRS"}],
                            "Shock response spectrum (absolute acceleration, maximax)", "frequency [Hz]", "S_a [g]",
                            logx=True, logy=True)
                else:
                    xy_plot(p, [{"x": s["pulse_t"], "y": np.asarray(s["pulse_a"]) / G, "label": "pulse"}],
                            f"Base acceleration pulse ({s['summary'].get('pulse')})", "time [s]", "a [g]")
            elif kind == "reaction":
                h = dr.harmonic
                if h is None:
                    raise FeaError("The results have no harmonic response")
                R = np.abs(np.asarray(h["reaction"]))
                xy_plot(p, [{"x": h["f"], "y": R[:, c], "label": f"R{'xyz'[c]}"} for c in range(3)],
                        "Amplitude of the sum of support reactions", "frequency [Hz]", "[N]", logx=True, logy=True)
            else:
                raise FeaError("kind: frf | transient | psd | asd | srs | pulse | shock_time | reaction")
            meta["image"] = str(p)
            txt = out(meta)
            return [txt, Image(path=p)] if return_image else txt

    @mcp.tool(structured_output=False)
    def fea_dyn_profile_presets(name: str | None = None) -> str:
        """Test-profile presets - STRUCTURE only (what to enter and with which tool): IEC 60068-2-6 (sine), -64 (random), -27 (shock), -47 (mounting), MIL-STD-810H (reference). Numerical values come from the standard/specification. name = one preset."""
        if name is not None:
            key = next((k for k in PROFILE_PRESETS if k.lower().replace(" ", "") == name.lower().replace(" ", "")), None)
            if key is None:
                raise FeaError(f"Preset '{name}' does not exist (presets: {', '.join(PROFILE_PRESETS)})")
            return out({"preset": key, **PROFILE_PRESETS[key], "note": PRESET_NOTE})
        return out({"presets": PROFILE_PRESETS, "note": PRESET_NOTE})
