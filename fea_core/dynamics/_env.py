"""Response envelopes and damage from the sweep in numba (parallel over the nodes, frequencies / times inside): the array (n × N)
is never built – on a model of 23 k nodes, 6 k frequencies of 50 s → ~2 s (numpy by batches is limited by the memory).

Response in the node p and the point t: u = Σ_m Q[t, m]·φ_m(p) + Σ_r C[t, r]·ψ_r(p) (modes + residual vectors), complex
Q = Qr + i·Qi → max over the phase as response.max_over_phase, σ_vM via the quadratic form P_VM.
"""
from __future__ import annotations

import numpy as np

from ..modules import need

_numba = need("numba", "accel")
njit = _numba.njit
prange = _numba.prange


@njit(cache=True, inline="always")
def _vm2(a, b):
    """a·P_VM·b for the components SXX SYY SZZ SXY SYZ SZX."""
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
            - 0.5 * (a[0] * b[1] + a[1] * b[0] + a[1] * b[2] + a[2] * b[1] + a[2] * b[0] + a[0] * b[2])
            + 3.0 * (a[3] * b[3] + a[4] * b[4] + a[5] * b[5]))


@njit(cache=True, inline="always")
def _phase_max(A, B, C):
    v = 0.5 * (A + B) + np.sqrt(0.25 * (A - B) ** 2 + C * C)
    return np.sqrt(v) if v > 0.0 else 0.0


@njit(cache=True, inline="always")
def _load(p, sh, ru, ss, rs, Lu, Ls):
    """Shapes of the modes and of the residual vectors of the node p → local field (M, 3), (M, 6) float64."""
    K = sh.shape[0]
    for m in range(K):
        for c in range(3):
            Lu[m, c] = sh[m, p, c]
        for c in range(6):
            Ls[m, c] = ss[m, p, c]
    for r in range(ru.shape[0]):
        for c in range(3):
            Lu[K + r, c] = ru[r, p, c]
        for c in range(6):
            Ls[K + r, c] = rs[r, p, c]


@njit(cache=True, inline="always")
def _combine(Q, t, L, out, nc):
    for c in range(nc):
        out[c] = 0.0
    for m in range(Q.shape[1]):
        q = Q[t, m]
        if q != 0.0:
            for c in range(nc):
                out[c] += q * L[m, c]


@njit(parallel=True, cache=True)
def envelope_kernel(Qr, Qi, cplx, sh, ru, ss, rs, uenv, uarg, senv, sarg):
    """Envelope max |u| and max σ_vM over the points t (the first occurrence of the maximum). Qr, Qi (n, K + R) float64."""
    n, M = Qr.shape
    N = sh.shape[1]
    for p in prange(N):
        Lu = np.empty((M, 3))
        Ls = np.empty((M, 6))
        _load(p, sh, ru, ss, rs, Lu, Ls)
        ur = np.empty(3)
        ui = np.empty(3)
        sr = np.empty(6)
        si = np.empty(6)
        ue, se = 0.0, 0.0
        ua, sa = 0, 0
        for t in range(n):
            _combine(Qr, t, Lu, ur, 3)
            _combine(Qr, t, Ls, sr, 6)
            if cplx:
                _combine(Qi, t, Lu, ui, 3)
                _combine(Qi, t, Ls, si, 6)
                A = ur[0] * ur[0] + ur[1] * ur[1] + ur[2] * ur[2]
                B = ui[0] * ui[0] + ui[1] * ui[1] + ui[2] * ui[2]
                C = ur[0] * ui[0] + ur[1] * ui[1] + ur[2] * ui[2]
                uv = _phase_max(A, B, C)
                sv = _phase_max(_vm2(sr, sr), _vm2(si, si), _vm2(sr, si))
            else:
                uv = np.sqrt(ur[0] * ur[0] + ur[1] * ur[1] + ur[2] * ur[2])
                v = _vm2(sr, sr)
                sv = np.sqrt(v) if v > 0.0 else 0.0
            if uv > ue:
                ue, ua = uv, t
            if sv > se:
                se, sa = sv, t
        uenv[p], uarg[p], senv[p], sarg[p] = ue, ua, se, sa


@njit(parallel=True, cache=True)
def sweep_kernel(Qr, Qi, ss, rs, w, SD, ND, k, k2, D):
    """Σ_t w_t / N(σ_a(f_t)) over the nodes (σ_a = amplitude of σ_vM max over the phase); SD ≤ 0 = node without S-N (D = 0)."""
    n, M = Qr.shape
    N = ss.shape[1]
    K = ss.shape[0]
    for p in prange(N):
        if SD[p] <= 0.0:
            D[p] = 0.0
            continue
        Ls = np.empty((M, 6))
        for m in range(K):
            for c in range(6):
                Ls[m, c] = ss[m, p, c]
        for r in range(rs.shape[0]):
            for c in range(6):
                Ls[K + r, c] = rs[r, p, c]
        sr = np.empty(6)
        si = np.empty(6)
        acc = 0.0
        for t in range(n):
            _combine(Qr, t, Ls, sr, 6)
            _combine(Qi, t, Ls, si, 6)
            r = _phase_max(_vm2(sr, sr), _vm2(si, si), _vm2(sr, si)) / SD[p]
            if r >= 1.0:
                acc += w[t] * r ** k[p] / ND[p]
            elif k2[p] > 0.0:
                acc += w[t] * r ** k2[p] / ND[p]
        D[p] = acc


def _f32(a, N: int, c: int) -> np.ndarray:
    return np.ascontiguousarray(a, np.float32) if a is not None and len(a) else np.zeros((0, N, c), np.float32)


def coefficients(Q: np.ndarray, C: np.ndarray | None, complex_mode: bool) -> tuple[np.ndarray, np.ndarray]:
    """[Q | C] → (real, imaginary) part float64 (n, K + R)."""
    QC = Q if C is None or not C.shape[1] else np.concatenate([Q, C], axis=1)
    if complex_mode:
        return np.ascontiguousarray(QC.real, np.float64), np.ascontiguousarray(QC.imag, np.float64)
    QC = np.ascontiguousarray(QC.real if np.iscomplexobj(QC) else QC, np.float64)
    return QC, QC


def envelopes(shapes, stress, rU, rS, Q, C, complex_mode: bool) -> dict:
    """See response.envelopes; shapes (K, N, 3), stress (K, N, 6), rU/rS (R, N, ·) or None."""
    N = shapes.shape[1]
    use_res = rU is not None and C is not None and C.shape[1]
    Qr, Qi = coefficients(Q, C if use_res else None, complex_mode)
    out = {"U": np.zeros(N, np.float32), "S": np.zeros(N, np.float32),
           "U_at": np.zeros(N, np.int32), "S_at": np.zeros(N, np.int32)}
    if not len(Qr):
        return out
    envelope_kernel(Qr, Qi, bool(complex_mode), _f32(shapes, N, 3), _f32(rU if use_res else None, N, 3),
                    _f32(stress, N, 6), _f32(rS if use_res else None, N, 6), out["U"], out["U_at"], out["S"], out["S_at"])
    return out


def sweep_sum(stress, rS, Q, C, w, SD, ND, k, k2) -> np.ndarray:
    """Σ_f w_f/N(σ_a(f)) over the nodes (complex response Q, C)."""
    N = stress.shape[1]
    use_res = rS is not None and C is not None and C.shape[1]
    Qr, Qi = coefficients(Q, C if use_res else None, True)
    D = np.zeros(N)
    sweep_kernel(Qr, Qi, _f32(stress, N, 6), _f32(rS if use_res else None, N, 6), np.asarray(w, np.float64),
                 SD, ND, k, k2, D)
    return D


def warmup() -> None:
    """Precompilation (fea_core.warmup)."""
    rng = np.random.default_rng(0)
    sh, ss = rng.random((2, 3, 3)), rng.random((2, 3, 6))
    Q = rng.random((4, 2)) + 1j * rng.random((4, 2))
    C = rng.random((4, 1)) + 0j
    envelopes(sh, ss, rng.random((1, 3, 3)), rng.random((1, 3, 6)), Q, C, True)
    envelopes(sh, ss, None, None, Q.real, None, False)
    one = np.ones(3)
    sweep_sum(ss, None, Q, None, np.ones(4), one, one, one * 5, one * 0)
