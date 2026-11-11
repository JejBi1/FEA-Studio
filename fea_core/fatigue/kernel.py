"""Fatigue computing core (numba, parallel over the nodes).

For each node: σ(t) = Σ F[t,k]·S[k] → equivalent stress with sign → rainflow
(the same algorithm as the `rainflow` package, ASTM E1049) → mean-stress correction →
S-N → Miner. Safety = multiple of the stress at which the damage over the design number of
blocks equals 1 (the cycles do not change when the stress is multiplied, they only scale).
"""
from __future__ import annotations

import math

import numpy as np

from ..modules import need

_numba = need("numba", "accel")      # if the Numba module is missing, a clear error with the installation procedure
njit = _numba.njit
prange = _numba.prange

EQ_SIGNED_MISES = 0
EQ_MAX_PRINCIPAL = 1
MEAN_NONE = 0
MEAN_GOODMAN = 1
MEAN_GERBER = 2


@njit(cache=True)
def principal3(s0, s1, s2, s3, s4, s5):
    """Principal stresses e1 ≥ e2 ≥ e3 of the symmetric tensor (xx, yy, zz, xy, yz, zx)."""
    p1 = s3 * s3 + s4 * s4 + s5 * s5
    q = (s0 + s1 + s2) / 3.0
    a, b, c = s0 - q, s1 - q, s2 - q
    p2 = a * a + b * b + c * c + 2.0 * p1
    if p2 <= 1e-30 * (q * q + 1e-300):
        return q, q, q
    p = math.sqrt(p2 / 6.0)
    # det((A - qI)/p) / 2
    det = (a * (b * c - s4 * s4) - s3 * (s3 * c - s4 * s5) + s5 * (s3 * s4 - b * s5)) / (p * p * p)
    r = det / 2.0
    if r <= -1.0:
        phi = math.pi / 3.0
    elif r >= 1.0:
        phi = 0.0
    else:
        phi = math.acos(r) / 3.0
    e1 = q + 2.0 * p * math.cos(phi)
    e3 = q + 2.0 * p * math.cos(phi + 2.0 * math.pi / 3.0)
    e2 = 3.0 * q - e1 - e3
    return e1, e2, e3


@njit(cache=True)
def equivalent(s0, s1, s2, s3, s4, s5, method):
    e1, e2, e3 = principal3(s0, s1, s2, s3, s4, s5)
    dom = e1 if abs(e1) >= abs(e3) else e3
    if method == EQ_MAX_PRINCIPAL:
        return dom
    vm = math.sqrt(max(0.0, 0.5 * ((s0 - s1) ** 2 + (s1 - s2) ** 2 + (s2 - s0) ** 2)
                       + 3.0 * (s3 * s3 + s4 * s4 + s5 * s5)))
    return vm if dom >= 0 else -vm


@njit(cache=True)
def reversals(x, out):
    """Turning points (as rainflow.reversals). Returns the number written into out."""
    n = x.shape[0]
    if n < 2:
        if n == 1:
            out[0] = x[0]
            return 1
        return 0
    m = 0
    x_last = x[0]
    xc = x[1]
    d_last = xc - x_last
    out[m] = x_last
    m += 1
    x_next = xc
    for i in range(2, n):
        x_next = x[i]
        if x_next == xc:
            continue
        d_next = x_next - xc
        if d_last * d_next < 0:
            out[m] = xc
            m += 1
        x_last = xc
        xc = x_next
        d_last = d_next
    out[m] = x_next
    m += 1
    return m


@njit(cache=True)
def rainflow_cycles(x, rng, mean, cnt):
    """Rainflow (3-point method ASTM E1049 with the residue as half-cycles). Returns the number of cycles."""
    rev = np.empty(x.shape[0] + 1)
    m = reversals(x, rev)
    stack = np.empty(m)
    sp = 0                       # number of points in the stack
    start = 0                    # index of the first point (popleft)
    nc = 0
    for i in range(m):
        stack[sp] = rev[i]
        sp += 1
        while sp - start >= 3:
            X = abs(stack[sp - 2] - stack[sp - 1])
            Y = abs(stack[sp - 3] - stack[sp - 2])
            if X < Y:
                break
            if sp - start == 3:
                rng[nc] = Y
                mean[nc] = 0.5 * (stack[sp - 3] + stack[sp - 2])
                cnt[nc] = 0.5
                nc += 1
                start += 1
            else:
                rng[nc] = Y
                mean[nc] = 0.5 * (stack[sp - 3] + stack[sp - 2])
                cnt[nc] = 1.0
                nc += 1
                last = stack[sp - 1]
                sp -= 3
                stack[sp] = last
                sp += 1
    for j in range(start, sp - 1):
        rng[nc] = abs(stack[j] - stack[j + 1])
        mean[nc] = 0.5 * (stack[j] + stack[j + 1])
        cnt[nc] = 0.5
        nc += 1
    return nc


@njit(cache=True)
def close_block(x):
    """Repeated block: start and end at the point with the largest |value| → all cycles closed."""
    n = x.shape[0]
    j = 0
    best = -1.0
    for i in range(n):
        if abs(x[i]) > best:
            best = abs(x[i])
            j = i
    out = np.empty(n + 1)
    for i in range(n):
        out[i] = x[(j + i) % n]
    out[n] = x[j]
    return out


@njit(cache=True)
def cycles_N(Sa, SD, ND, k, k2):
    if Sa <= 0.0:
        return math.inf
    if Sa >= SD:
        return ND * (SD / Sa) ** k
    if k2 <= 0.0:
        return math.inf
    return ND * (SD / Sa) ** k2


@njit(cache=True)
def damage(rng, mean, cnt, nc, scale, SD, ND, k, k2, Rm, mean_method, mscale):
    """Miner. The ranges are multiplied by |scale|, the means by mscale (negative = opposite sign)."""
    D = 0.0
    scale = abs(scale)
    for i in range(nc):
        Sa = 0.5 * rng[i] * scale
        Sm = mean[i] * mscale
        if mean_method != MEAN_NONE and Sm > 0.0 and Rm > 0.0:
            r = Sm / Rm
            if r >= 1.0:
                return math.inf
            Sa = Sa / (1.0 - r) if mean_method == MEAN_GOODMAN else Sa / (1.0 - r * r)
        N = cycles_N(Sa, SD, ND, k, k2)
        if N < 1.0:
            N = 1.0
        D += cnt[i] / N
    return D


@njit(parallel=True, cache=True)
def fatigue_nodes(F, S, eq_method, mean_method, SD, ND, k, k2, Rm, repeats, closed,
                  out_D, out_sf, out_amp):
    """F (T, K) multiples, S (K, n, 6) stresses of the unit states at the nodes; parameters of the S-N per node."""
    T, K = F.shape
    n = S.shape[1]
    for j in prange(n):
        x = np.empty(T)
        for t in range(T):
            s0 = s1 = s2 = s3 = s4 = s5 = 0.0
            for kk in range(K):
                f = F[t, kk]
                s0 += f * S[kk, j, 0]
                s1 += f * S[kk, j, 1]
                s2 += f * S[kk, j, 2]
                s3 += f * S[kk, j, 3]
                s4 += f * S[kk, j, 4]
                s5 += f * S[kk, j, 5]
            x[t] = equivalent(s0, s1, s2, s3, s4, s5, eq_method)
        if closed:
            x = close_block(x)
        L = x.shape[0] + 1
        rng = np.empty(L)
        mean = np.empty(L)
        cnt = np.empty(L)
        nc = rainflow_cycles(x, rng, mean, cnt)
        nc = merge_cycles(rng, mean, cnt, nc)
        out_D[j], out_sf[j], out_amp[j] = node_result(rng, mean, cnt, nc, 1.0, 1.0, SD[j], ND[j], k[j],
                                                      k2[j], Rm[j], mean_method, repeats)


@njit(cache=True)
def node_result(rng, mean, cnt, nc, sr, sm, SD, ND, k, k2, Rm, mean_method, repeats):
    """(damage per block, safety, max. amplitude) from the cycles; ranges × sr, means × sm."""
    amax = 0.0
    for i in range(nc):
        if 0.5 * rng[i] > amax:
            amax = 0.5 * rng[i]
    amax *= abs(sr)
    D1 = damage(rng, mean, cnt, nc, sr, SD, ND, k, k2, Rm, mean_method, sm)
    # safety: s at which repeats · D(s) = 1 (D is non-decreasing in s)
    if nc == 0 or amax <= 0.0:
        return D1, math.inf, amax
    lo, hi = 0.0, 1.0
    while repeats * damage(rng, mean, cnt, nc, hi * sr, SD, ND, k, k2, Rm, mean_method, hi * sm) < 1.0:
        lo = hi
        hi *= 2.0
        if hi > 1e4:
            return D1, math.inf, amax
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if repeats * damage(rng, mean, cnt, nc, mid * sr, SD, ND, k, k2, Rm, mean_method, mid * sm) < 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6 * hi:
            break
    return D1, 0.5 * (lo + hi), amax


@njit(cache=True)
def merge_cycles(rng, mean, cnt, nc):
    """Merges the cycles with the same range and mean value (rel. 1e-12). Returns the new count."""
    if nc == 0:
        return 0
    scale = 0.0
    for i in range(nc):
        scale = max(scale, abs(rng[i]), abs(mean[i]))
    tol = 1e-12 * max(scale, 1e-300)
    order = np.argsort(rng[:nc] + 1e-6 * mean[:nc])
    r2, m2, c2 = rng[order].copy(), mean[order].copy(), cnt[order].copy()
    m = 0
    for i in range(nc):
        found = False
        j = m - 1
        while j >= 0 and r2[i] - rng[j] <= tol:   # candidates with the same range
            if abs(m2[i] - mean[j]) <= tol:
                cnt[j] += c2[i]
                found = True
                break
            j -= 1
        if not found:
            rng[m], mean[m], cnt[m] = r2[i], m2[i], c2[i]
            m += 1
    return m


@njit(parallel=True, cache=True)
def fatigue_nodes_proportional(f, Seff, eq_method, mean_method, SD, ND, k, k2, Rm, repeats, closed,
                               out_D, out_sf, out_amp):
    """Proportional loading: σ(t) = f(t)·Seff → x(t) = f(t)·eq(Seff); rainflow only once."""
    x = close_block(f) if closed else f.copy()
    L = x.shape[0] + 1
    rng = np.empty(L)
    mean = np.empty(L)
    cnt = np.empty(L)
    nc = rainflow_cycles(x, rng, mean, cnt)
    nc = merge_cycles(rng, mean, cnt, nc)
    for j in prange(Seff.shape[0]):
        e = equivalent(Seff[j, 0], Seff[j, 1], Seff[j, 2], Seff[j, 3], Seff[j, 4], Seff[j, 5], eq_method)
        out_D[j], out_sf[j], out_amp[j] = node_result(rng, mean, cnt, nc, e, e, SD[j], ND[j], k[j],
                                                      k2[j], Rm[j], mean_method, repeats)


def cycles_of(series: np.ndarray, closed: bool = False) -> np.ndarray:
    """Cycles of one history: array (n, 3) = range, mean value, count (0.5 = half cycle)."""
    x = np.ascontiguousarray(series, dtype=np.float64)
    if closed and len(x) > 1:
        x = close_block(x)
    L = len(x) + 1
    rng, mean, cnt = np.empty(L), np.empty(L), np.empty(L)
    nc = rainflow_cycles(x, rng, mean, cnt)
    return np.column_stack([rng[:nc], mean[:nc], cnt[:nc]])
