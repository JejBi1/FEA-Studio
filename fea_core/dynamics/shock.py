"""Shocks and the response spectrum (D4) – pulses, SRS, the combination of the modes SRSS / CQC, the fall as an equivalent pulse.

CONVENTIONS:
  - Pulse of the acceleration of the base a(t) [mm/s² or g – the SRS returns the same units] on a uniform grid t_i = i·dt,
    between the samples linearly (as `modal_ode`).
  - SRS (shock response spectrum) = the maximum of the response of a 1-DOF oscillator (f_n, ζ) to the motion of the base a(t):
      kind="abs_acc"  absolute acceleration of the mass ẍ = −(2ζω·ż + ω²·z) (maximax = max |ẍ| over the whole time; the standard
                      of the norms, Q = 10 ↔ ζ = 5 %)
      kind="rel_disp" relative displacement z [mm] (pseudo-acceleration = ω²·z_max).
    Result: pos = max(y), neg = max(−y), maximax = max(pos, neg), primary = max|y| during the pulse (t ≤ t_pulse),
    residual = max|y| after the pulse (free decay; the calculation continues `tail_periods` periods after the end of the record with
    zero excitation).
  - The calculation = the EXACT solution for the piecewise linear input (ramp-invariant; the same as the recursive filter of Smallwood
    1981 – the agreement verified in a test); the step is in addition divided so that at least
    `points_per_period` samples fall on a period of the oscillator (the input is interpolated linearly between the samples = no change of the excitation);
    the sampling error of the peak ≤ 1 − cos(π/points_per_period) (50 → 0.2 %).
  - Peak of the mode from the SRS: q_j,max = Γ_j·S_a(f_j, ζ_j)/ω_j² (S_a = pseudo-acceleration, `modal_peaks`); the responses of the modes
    r_j = A_j·q_j,max (with the SIGN – Γ_j and the shape with the sign of ccx) are combined by SRSS or CQC (ρ_jk Der
    Kiureghian 1981): r = √(Σ_jk r_j·ρ_jk·r_k).
"""
from __future__ import annotations

import math

import numpy as np

from ..modules import need
from .modal_ode import coefficients
from .psd import loglog_interp

_numba = need("numba", "accel")
njit = _numba.njit
prange = _numba.prange

G_MM = 9810.0
PULSES = ("halfsine", "sawtooth", "trapezoid")
SRS_KINDS = ("abs_acc", "rel_disp")


# ------------------------------------------------------------------ pulzy

def pulse(kind: str, A: float, tau: float, dt: float, t_end: float | None = None, rise: float | None = None):
    """Pulse of the acceleration → (t, a): halfsine A·sin(πt/τ); sawtooth = a saw with an END peak A·t/τ (then a jump to 0
    during one step dt); trapezoid = rise `rise` (default 0.1·τ), a plateau, a descent `rise`, the total duration τ.
    t_end ≥ τ (default 2τ – the rest zero), dt [s].
    """
    if kind not in PULSES:
        raise ValueError(f"Unknown pulse '{kind}' (options: {', '.join(PULSES)})")
    if not (tau > 0 and dt > 0):
        raise ValueError("Pulse duration and time step must both be positive")
    t_end = 2.0 * tau if t_end is None else float(t_end)
    if t_end < tau:
        raise ValueError("The record must end after the end of the pulse")
    n = int(math.floor(t_end / dt + 1e-9)) + 1
    t = np.arange(n) * dt
    a = np.zeros(n)
    on = t <= tau * (1 + 1e-12)
    if kind == "halfsine":
        a[on] = A * np.sin(np.pi * t[on] / tau)
        a[np.isclose(t, tau, rtol=0, atol=1e-12 * tau)] = 0.0
    elif kind == "sawtooth":
        a[on] = A * t[on] / tau
    else:
        r = 0.1 * tau if rise is None else float(rise)
        if not 0 < r <= tau / 2:
            raise ValueError("Trapezoid rise time must be in (0; τ/2]")
        tt = t[on]
        a[on] = A * np.minimum(1.0, np.minimum(tt / r, (tau - tt) / r))
    return t, a


def drop_pulse(h: float, m: float, k: float, e: float = 0.0) -> dict:
    """A fall from the height h [mm] as an equivalent half-sine (a ROUGH estimate; the real impact = the direct dynamics with contact):
    v = √(2gh) [mm/s], τ = π·√(m/k) (m [t], k [N/mm] = stiffness of the contact), A = π·(1 + e)·v/(2τ) [mm/s²]
    (change of velocity (1 + e)·v = the area of the half-sine 2Aτ/π; e = coefficient of restitution). Returns v, tau, A, A_g.
    """
    if h < 0 or m <= 0 or k <= 0 or not 0 <= e <= 1:
        raise ValueError("Drop: h ≥ 0, m > 0, k > 0 and 0 ≤ e ≤ 1")
    v = math.sqrt(2.0 * G_MM * h)
    tau = math.pi * math.sqrt(m / k)
    A = math.pi * (1.0 + e) * v / (2.0 * tau)
    return {"v": v, "tau": tau, "A": A, "A_g": A / G_MM}


# ------------------------------------------------------------------ spektrum odezvy

@njit(parallel=True, cache=True)
def _srs_core(a, c, sub, ntail, i_pulse, two_zw, w2, absacc, out):
    T = a.shape[0]
    nf = c.shape[0]
    for j in prange(nf):
        c0, c1, c2, c3 = c[j, 0], c[j, 1], c[j, 2], c[j, 3]
        c4, c5, c6, c7 = c[j, 4], c[j, 5], c[j, 6], c[j, 7]
        m = sub[j]
        q = 0.0
        v = 0.0
        pos = 0.0
        neg = 0.0
        prim = 0.0
        res = 0.0
        for i in range(T - 1):
            a0 = a[i]
            da = (a[i + 1] - a0) / m
            for s in range(m):
                p0 = -(a0 + da * s)
                p1 = -(a0 + da * (s + 1))
                qn = c0 * q + c1 * v + c2 * p0 + c3 * p1
                v = c4 * q + c5 * v + c6 * p0 + c7 * p1
                q = qn
                y = -(two_zw[j] * v + w2[j] * q) if absacc else q
                if y > pos:
                    pos = y
                if -y > neg:
                    neg = -y
                ay = abs(y)
                if i < i_pulse:                      # end of interval t ≤ t_pulse
                    if ay > prim:
                        prim = ay
                else:
                    if ay > res:
                        res = ay
        for s in range(ntail[j]):
            qn = c0 * q + c1 * v
            v = c4 * q + c5 * v
            q = qn
            y = -(two_zw[j] * v + w2[j] * q) if absacc else q
            if y > pos:
                pos = y
            if -y > neg:
                neg = -y
            ay = abs(y)
            if ay > res:
                res = ay
        out[j, 0] = pos
        out[j, 1] = neg
        out[j, 2] = pos if pos > neg else neg
        out[j, 3] = prim
        out[j, 4] = res


def srs(t, a, f, zeta: float = 0.05, kind: str = "abs_acc", points_per_period: int = 50,
        t_pulse: float | None = None, tail_periods: float = 1.0) -> dict:
    """Response spectrum (SRS) of the record a(t) at the frequencies f [Hz] (see the conventions of the module).

    t (T,) uniformly [s], a (T,) acceleration of the base; zeta = relative damping (a scalar or (n_f,));
    t_pulse = end of the pulse for the division primary / residual (None = the last non-zero sample + dt).
    Returns a dict of arrays (n_f,): pos, neg, maximax, primary, residual (units: for abs_acc the units of a, for rel_disp mm when a is in mm/s²).
    """
    if kind not in SRS_KINDS:
        raise ValueError(f"Unknown spectrum type '{kind}' (options: {', '.join(SRS_KINDS)})")
    t = np.asarray(t, dtype=np.float64).ravel()
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64).ravel())
    if t.shape != a.shape or t.size < 2:
        raise ValueError("t and a must have the same length ≥ 2")
    dt = (t[-1] - t[0]) / (t.size - 1)
    if dt <= 0 or np.max(np.abs(np.diff(t) - dt)) > 1e-6 * dt:
        raise ValueError("The record must have a uniform time step")
    f = np.atleast_1d(np.asarray(f, dtype=np.float64)).ravel()
    if np.any(f <= 0):
        raise ValueError("SRS frequencies must be positive")
    ze = np.broadcast_to(np.asarray(zeta, dtype=np.float64), f.shape).copy()
    ppp = max(4, int(points_per_period))
    sub = np.maximum(1, np.ceil(ppp * f * dt - 1e-9)).astype(np.int64)
    h = dt / sub
    om = 2.0 * np.pi * f
    c = coefficients(h, om, ze)
    ntail = np.ceil(tail_periods / (f * h) - 1e-9).astype(np.int64)
    if t_pulse is None:
        nz = np.nonzero(np.abs(a) > 1e-12 * max(np.abs(a).max(), 1e-300))[0]
        i_pulse = int(min(nz[-1] + 1, a.size - 1)) if nz.size else 0
    else:
        i_pulse = int(round((float(t_pulse) - t[0]) / dt))
    # interval i (t_i → t_{i+1}) belongs to the pulse when i < i_pulse
    out = np.empty((f.size, 5))
    _srs_core(a, c, sub, ntail, i_pulse, 2.0 * ze * om, om * om, kind == "abs_acc", out)
    return {"f": f, "pos": out[:, 0], "neg": out[:, 1], "maximax": out[:, 2], "primary": out[:, 3],
            "residual": out[:, 4]}


def srs_interp(table, f) -> np.ndarray:
    """SRS table [[f Hz, S_a], …] at points f – log-log, outside the range 0."""
    return loglog_interp(table, f)


def modal_peaks(fn, gamma, Sa) -> np.ndarray:
    """Peaks of the modal coordinates q_j = Γ_j·S_a(f_j)/ω_j² (Sa (K,) in mm/s² = the pseudo-acceleration from the SRS)."""
    om = 2.0 * np.pi * np.asarray(fn, dtype=np.float64)
    return np.asarray(gamma, dtype=np.float64) * np.asarray(Sa, dtype=np.float64) / (om * om)


# ------------------------------------------------------------------ combination of the modes

def cqc_rho(fn, zeta) -> np.ndarray:
    """Correlation coefficients CQC (Der Kiureghian 1981) (K, K):
    ρ_ij = 8√(ζ_iζ_j)(ζ_i + rζ_j)r^{3/2} / ((1 − r²)² + 4ζ_iζ_j·r(1 + r²) + 4(ζ_i² + ζ_j²)r²), r = ω_j/ω_i.
    """
    w = np.atleast_1d(np.asarray(fn, dtype=np.float64)).ravel()
    z = np.broadcast_to(np.asarray(zeta, dtype=np.float64), w.shape)
    if np.any(w <= 0):
        raise ValueError("CQC: natural frequencies must be positive")
    r = w[None, :] / w[:, None]
    zi = z[:, None]
    zj = z[None, :]
    num = 8.0 * np.sqrt(zi * zj) * (zi + r * zj) * r ** 1.5
    den = (1.0 - r * r) ** 2 + 4.0 * zi * zj * r * (1.0 + r * r) + 4.0 * (zi * zi + zj * zj) * r * r
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(den > 0, num / den, 1.0)
    np.fill_diagonal(rho, 1.0)
    return 0.5 * (rho + rho.T)


@njit(parallel=True, cache=True)
def _cqc(R, rho, out):
    K, n = R.shape
    for i in prange(n):
        s = 0.0
        for j in range(K):
            rj = R[j, i]
            if rj == 0.0:
                continue
            t = 0.0
            for k in range(K):
                t += rho[j, k] * R[k, i]
            s += rj * t
        out[i] = math.sqrt(s) if s > 0.0 else 0.0


def combine(R, rho=None) -> np.ndarray:
    """Combination of the peaks of the modes: R (K, …) = responses of the modes with the sign (e.g. (K, n, c)); rho None = SRSS
    √(Σ r_j²), otherwise CQC √(Σ_jk r_j·ρ_jk·r_k) (numba over the nodes). Returns the shape of R without the first axis.
    """
    R = np.asarray(R, dtype=np.float64)
    K = R.shape[0]
    shape = R.shape[1:]
    if rho is None:
        return np.sqrt(np.sum(R * R, axis=0))
    rho = np.ascontiguousarray(rho, dtype=np.float64)
    if rho.shape != (K, K):
        raise ValueError(f"Matrix ρ must have shape ({K}, {K})")
    flat = np.ascontiguousarray(R.reshape(K, -1))
    out = np.empty(flat.shape[1])
    _cqc(flat, rho, out)
    return out.reshape(shape)


def warmup() -> None:
    """Precompilation of numba (fea_core.warmup)."""
    t, a = pulse("halfsine", 1.0, 0.01, 1e-3)
    srs(t, a, [10.0, 100.0], kind="abs_acc")
    srs(t, a, [10.0], kind="rel_disp")
    combine(np.ones((2, 3)), cqc_rho([10.0, 11.0], 0.05))
