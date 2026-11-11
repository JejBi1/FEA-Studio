"""Exact solution of the independent equations of the modes in time (modal transient response, D2; specification F8, decision 3a).

Equation of the mode j (shape normalised by the mass, φ_jᵀMφ_j = 1):

    q̈_j + 2·ζ_j·ω_j·q̇_j + ω_j²·q_j = p_j(t)

p_j(t) = φ_jᵀF(t) [N/√t … in the units of the project: mm/s² · √t] is PIECEWISE LINEAR between the samples of a uniform
grid t_i = i·dt (i = 0 … T−1); P[0] holds already at time 0 (a jump at t = 0 is thus exact, a jump inside the course is spread over one step dt). The solution is EXACT (Nigam & Jennings 1969, Duhamel integral; the same is done by ccx
*MODAL DYNAMIC, manual node315: "exact for piecewise linear loading"): the state [q, q̇] is transferred by the matrix
exp(A·dt) and two input vectors (the start and the end of the section). The coefficients are computed by the matrix exponential of the
extended system (Van Loan 1978) in the dimensionless time s = t/dt – valid for any ζ ≥ 0 (also ζ = 0, ζ ≥ 1)
and for ω = 0 (a rigid mode = double integration of p(t)); without division by small numbers (ω·dt → 0).

Damping: only viscous modal ζ_j (from the modal ζ, Rayleigh α + β·ω² or the weighting by materials – the conversion
is done by the caller). Hysteretic (structural) damping η has no sense in time (non-causal) → only in the harmonic response
(`psd.modal_frf(eta=)`).

Base excitation (`integrate_base`): q_j are RELATIVE modal coordinates (u_rel = u − r·u_base),
p_j = −Γ_j·a_b(t), Γ_j = φ_jᵀM·r = the participation factor (ccx `.dat`, the sign of ccx – the shape φ_j must have the
same sign; see specification 5a.3, scale s_j). The returned a = q̈ is the RELATIVE acceleration; the absolute acceleration
of the node = Σ φ_j·q̈_j + r·a_b.

When the excitation by the displacement of the base d(t) piecewise linear is given, the acceleration consists of Dirac impulses
Δv_k·δ(t − t_k) at the breaks of the path (A3). These impulses cause exact jumps of the velocity of the modal coordinate
Δq̇_j = −Γ_j·Δv_k (`impulses` in `integrate`), so the response is completely independent of the integration step dt.

Units: dt [s], ω [rad/s], q [mm·√t] (u = Σφ_j q_j in mm), p in the corresponding units; for the excitation by the base
a_b [mm/s²] (g = 9810 mm/s²). Arrays float64, no global state.
"""
from __future__ import annotations

import numpy as np

from ..modules import need

_numba = need("numba", "accel")      # if the Numba module is missing, a clear error with the installation procedure
njit = _numba.njit
prange = _numba.prange


def _as_modes(x, K: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        a = np.full(K, float(a))
    a = a.ravel()
    if a.shape[0] != K:
        raise ValueError(f"{name}: expected {K} values (number of modes), got {a.shape[0]}")
    return a


def coefficients(dt, omega, zeta) -> np.ndarray:
    """Coefficients of the exact step for K modes → array (K, 8):

        q₁ = c0·q₀ + c1·v₀ + c2·p₀ + c3·p₁
        v₁ = c4·q₀ + c5·v₀ + c6·p₀ + c7·p₁

    dt can be a scalar or an array (K,) (a different step by modes – the spectrum of the response with the division of the step).
    The extended system in the time s = t/dt with the state [q, w = dt·v, p̂ = dt²·p, dp̂/ds]:
    A = [[0, 1, 0, 0], [−H², −2ζH, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], H = ω·dt; E = exp(A).
    """
    from scipy.linalg import expm

    om = np.atleast_1d(np.asarray(omega, dtype=np.float64)).ravel()
    K = om.shape[0]
    ze = _as_modes(zeta, K, "zeta")
    h = _as_modes(dt, K, "dt")
    if np.any(h <= 0):
        raise ValueError("Time step dt must be positive")
    if np.any(om < 0) or np.any(ze < 0):
        raise ValueError("ω and ζ must not be negative")
    H = om * h
    A = np.zeros((K, 4, 4))
    A[:, 0, 1] = 1.0
    A[:, 1, 0] = -H * H
    A[:, 1, 1] = -2.0 * ze * H
    A[:, 1, 2] = 1.0
    A[:, 2, 3] = 1.0
    E = expm(A)
    c = np.empty((K, 8))
    c[:, 0] = E[:, 0, 0]
    c[:, 1] = h * E[:, 0, 1]
    c[:, 2] = h * h * (E[:, 0, 2] - E[:, 0, 3])
    c[:, 3] = h * h * E[:, 0, 3]
    c[:, 4] = E[:, 1, 0] / h
    c[:, 5] = E[:, 1, 1]
    c[:, 6] = h * (E[:, 1, 2] - E[:, 1, 3])
    c[:, 7] = h * E[:, 1, 3]
    return c


@njit(parallel=True, cache=True)
def _recur(P, c, two_zw, w2, q0, v0, q, v, a):
    T, K = P.shape
    for j in prange(K):
        qq = q0[j]
        vv = v0[j]
        c0, c1, c2, c3 = c[j, 0], c[j, 1], c[j, 2], c[j, 3]
        c4, c5, c6, c7 = c[j, 4], c[j, 5], c[j, 6], c[j, 7]
        q[0, j] = qq
        v[0, j] = vv
        a[0, j] = P[0, j] - two_zw[j] * vv - w2[j] * qq
        for i in range(1, T):
            p0 = P[i - 1, j]
            p1 = P[i, j]
            qn = c0 * qq + c1 * vv + c2 * p0 + c3 * p1
            vn = c4 * qq + c5 * vv + c6 * p0 + c7 * p1
            qq = qn
            vv = vn
            q[i, j] = qq
            v[i, j] = vv
            a[i, j] = p1 - two_zw[j] * vv - w2[j] * qq


@njit(parallel=True, cache=True)
def _recur_impulses(P, J, c, two_zw, w2, q0, v0, q, v, a):
    T, K = P.shape
    for j in prange(K):
        qq = q0[j]
        vv = v0[j] + J[0, j]
        c0, c1, c2, c3 = c[j, 0], c[j, 1], c[j, 2], c[j, 3]
        c4, c5, c6, c7 = c[j, 4], c[j, 5], c[j, 6], c[j, 7]
        q[0, j] = qq
        v[0, j] = vv
        a[0, j] = P[0, j] - two_zw[j] * vv - w2[j] * qq
        for i in range(1, T):
            p0 = P[i - 1, j]
            p1 = P[i, j]
            qn = c0 * qq + c1 * vv + c2 * p0 + c3 * p1
            vn = c4 * qq + c5 * vv + c6 * p0 + c7 * p1
            qq = qn
            vv = vn + J[i, j]
            q[i, j] = qq
            v[i, j] = vv
            a[i, j] = p1 - two_zw[j] * vv - w2[j] * qq


def _recur_substep(dt: float, P: np.ndarray, impulses_dict: dict, om: np.ndarray, ze: np.ndarray,
                   q0: np.ndarray, v0: np.ndarray, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> None:
    """General sub-stepping for the impulses outside the nodes of the uniform grid."""
    T, K = P.shape
    two_zw = 2.0 * ze * om
    w2 = om * om
    qq = q0.copy()
    vv = v0.copy()
    # Impulse at time 0
    if 0.0 in impulses_dict:
        vv += impulses_dict[0.0]
    q[0] = qq
    v[0] = vv
    a[0] = P[0] - two_zw * vv - w2 * qq

    for i in range(1, T):
        t_prev = (i - 1) * dt
        t_curr = i * dt
        # Find the impulses inside the interval (t_prev, t_curr]
        sub_events = sorted([tk for tk in impulses_dict if t_prev < tk <= t_curr + 1e-12 * dt])
        if not sub_events:
            c = coefficients(dt, om, ze)
            p0, p1 = P[i - 1], P[i]
            qn = c[:, 0] * qq + c[:, 1] * vv + c[:, 2] * p0 + c[:, 3] * p1
            vn = c[:, 4] * qq + c[:, 5] * vv + c[:, 6] * p0 + c[:, 7] * p1
            qq = qn
            vv = vn
        else:
            tau = t_prev
            p_tau = P[i - 1]
            for ev_t in sub_events:
                h = ev_t - tau
                if h > 1e-14:
                    c = coefficients(h, om, ze)
                    p_ev = P[i - 1] + ((ev_t - t_prev) / dt) * (P[i] - P[i - 1])
                    qn = c[:, 0] * qq + c[:, 1] * vv + c[:, 2] * p_tau + c[:, 3] * p_ev
                    vn = c[:, 4] * qq + c[:, 5] * vv + c[:, 6] * p_tau + c[:, 7] * p_ev
                    qq = qn
                    vv = vn
                    tau = ev_t
                    p_tau = p_ev
                vv += impulses_dict[ev_t]
            h_rem = t_curr - tau
            if h_rem > 1e-14:
                c = coefficients(h_rem, om, ze)
                p_curr = P[i]
                qn = c[:, 0] * qq + c[:, 1] * vv + c[:, 2] * p_tau + c[:, 3] * p_curr
                vn = c[:, 4] * qq + c[:, 5] * vv + c[:, 6] * p_tau + c[:, 7] * p_curr
                qq = qn
                vv = vn
        q[i] = qq
        v[i] = vv
        a[i] = P[i] - two_zw * vv - w2 * qq


def integrate(dt: float, P, omega, zeta, q0=None, v0=None, impulses=None):
    """Exact response of K independent modes to the piecewise linear excitation and Dirac impulses (A3).

    dt [s] = step of the uniform grid; P (T, K) = modal forces p_j(t_i) (or (T,) for one mode);
    omega (K,) [rad/s] (0 = rigid mode); zeta (K,) or a scalar (relative damping, ≥ 0); q0, v0 (K,) = initial
    state (default 0); impulses = optional list [(t_k, Δq̇_k)] or dictionary {t_k: Δq̇_k} of the jumps of the velocities of the
    modal coordinates from the Dirac impulses (e.g. the breaks of the displacement of the base).
    Returns (q, v, a) – each (T, K) (for the input (T,) the shape (T,)): modal displacement, velocity and acceleration
    a = p − 2ζωv − ω²q at the times t_i.
    """
    P = np.asarray(P, dtype=np.float64)
    one = P.ndim == 1
    if one:
        P = P[:, None]
    if P.ndim != 2 or P.shape[0] < 1:
        raise ValueError("P must have shape (T, K)")
    T, K = P.shape
    om = _as_modes(omega, K, "omega")
    ze = _as_modes(zeta, K, "zeta")
    q0 = np.zeros(K) if q0 is None else _as_modes(q0, K, "q0")
    v0 = np.zeros(K) if v0 is None else _as_modes(v0, K, "v0")
    q = np.empty((T, K))
    v = np.empty((T, K))
    a = np.empty((T, K))

    if impulses:
        # Convert impulses to the dict {t_k: Δq̇_k}
        imp_dict: dict[float, np.ndarray] = {}
        items = impulses.items() if isinstance(impulses, dict) else impulses
        for tk, dqv in items:
            arr = np.asarray(dqv, dtype=np.float64)
            if arr.ndim == 0:
                arr = np.full(K, float(arr))
            arr = arr.ravel()
            if arr.shape[0] != K:
                raise ValueError(f"impulses: expected {K} components, got {arr.shape[0]}")
            if np.all(np.abs(arr) < 1e-30):
                continue
            key = round(float(tk), 10)
            imp_dict[key] = imp_dict.get(key, np.zeros(K)) + arr

        # Check that all the impulses are at the nodes of the grid
        dt_val = float(dt)
        on_grid = True
        for tk in imp_dict:
            idx = int(round(tk / dt_val))
            if abs(tk - idx * dt_val) > 1e-8 * dt_val and abs(tk - idx * dt_val) > 1e-12:
                on_grid = False
                break

        if on_grid:
            J = np.zeros((T, K), dtype=np.float64)
            for tk, dqv in imp_dict.items():
                idx = int(round(tk / dt_val))
                if 0 <= idx < T:
                    J[idx] += dqv
            c = coefficients(dt_val, om, ze)
            _recur_impulses(np.ascontiguousarray(P), J, c, 2.0 * ze * om, om * om, q0, v0, q, v, a)
        else:
            _recur_substep(dt_val, np.ascontiguousarray(P), imp_dict, om, ze, q0, v0, q, v, a)
    else:
        c = coefficients(float(dt), om, ze)
        _recur(np.ascontiguousarray(P), c, 2.0 * ze * om, om * om, q0, v0, q, v, a)

    if one:
        return q[:, 0], v[:, 0], a[:, 0]
    return q, v, a


def integrate_base(dt: float, a_b, gamma, omega, zeta, q0=None, v0=None):
    """Excitation by the acceleration of the base a_b(t) (T,) [mm/s²] in the direction r: p_j = −Γ_j·a_b (Γ_j (K,) = φ_jᵀM·r with the sign
    of ccx). Returns the RELATIVE (q, v, a) – each (T, K).
    """
    a_b = np.asarray(a_b, dtype=np.float64).ravel()
    g = np.atleast_1d(np.asarray(gamma, dtype=np.float64)).ravel()
    P = -np.outer(a_b, g)
    return integrate(dt, P, omega, zeta, q0, v0)


def warmup() -> None:
    """Precompilation of numba (fea_core.warmup)."""
    integrate(1e-3, np.ones((4, 2)), [10.0, 0.0], 0.02)
    integrate(1e-3, np.ones((4, 2)), [10.0, 0.0], 0.02, impulses=[(0.0, [0.1, 0.2])])
