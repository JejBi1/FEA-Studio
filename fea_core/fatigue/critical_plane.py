"""Critical-plane method (numba, parallel over the nodes).

For each node the planes (normal n over the hemisphere) are searched and the plane with the largest damage is sought.
Criteria (values converted to the equivalent amplitude of the normal stress → S-N in the amplitudes σ):
  - CP_NORMAL  – normal stress σn(t) = n·σ(t)·n, rainflow of σn, mean-stress correction (Goodman/Gerber)
                 as in the other methods. Suitable for brittle materials / failure by tension (grey cast iron).
  - CP_FINDLEY – Findley (1959): f = τa + k·σn,max, σ_eq = f/λ, λ = (k + √(1+k²))/2 (for alternating tension
                 σ_eq = σa). k from the ratio of the fatigue limits s = f₋₁/t₋₁: k = (2−s)/(2√(s−1)); pure torsion σ_eq = s·τa.
                 The mean stress is in the term k·σn,max (Goodman/Gerber is not used).
Shear in the plane under non-proportional loading: projection of the shear vector onto the directions d in the plane (n_dir directions over
180°, then refinement), rainflow of the projection τ_d(t) in each direction, the worst direction is taken ("the longest projection"; for an
elliptical path = the major semi-axis = radius of the smallest circumscribed circle). σn,max of a cycle = the maximum σn in the time window
of the loop (from the start of the cycle to the return of the history to the initial level) – for a constant amplitude = the classic Findley.
Plane search: coarse grid (Fibonacci points on the hemisphere) + principal directions at the moment of the maximum stress, selection
of candidates by a cheap estimate (the range), then the exact damage and refinement (pattern search, step /2 per level).
Proportional loading σ(t) = f(t)·S: rainflow of f only once, planes analytically.
"""
from __future__ import annotations

import math

import numpy as np

from ..modules import need

_numba = need("numba", "accel")      # if the Numba module is missing, a clear error with the installation procedure
njit = _numba.njit
prange = _numba.prange

from .kernel import MEAN_NONE, MEAN_GOODMAN, rainflow_cycles, merge_cycles, node_result, principal3
from .settings import findley_k, findley_lambda  # noqa: F401 – part of the interface of the module

CP_NORMAL = 0
CP_FINDLEY = 1
N_CAND_PROP = 4       # proportional loading: number of candidate planes for the refinement


def hemisphere(n: int) -> np.ndarray:
    """n normals uniformly over the hemisphere z ≥ 0 (spherical Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    z = i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    return np.ascontiguousarray(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))


def plane_spacing(n: int) -> float:
    """Average angular pitch of the coarse grid of planes [rad]."""
    return math.sqrt(2.0 * math.pi / max(n, 1))


# ====================================================================== helpers
@njit(cache=True)
def basis(nx, ny, nz):
    """Orthonormal basis (a, b) of the plane with the normal n."""
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    if ax <= ay and ax <= az:
        ex, ey, ez = 1.0, 0.0, 0.0
    elif ay <= az:
        ex, ey, ez = 0.0, 1.0, 0.0
    else:
        ex, ey, ez = 0.0, 0.0, 1.0
    a0, a1, a2 = ny * ez - nz * ey, nz * ex - nx * ez, nx * ey - ny * ex
    L = math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
    a0, a1, a2 = a0 / L, a1 / L, a2 / L
    b0, b1, b2 = ny * a2 - nz * a1, nz * a0 - nx * a2, nx * a1 - ny * a0
    return a0, a1, a2, b0, b1, b2


@njit(cache=True)
def eigvec(s0, s1, s2, s3, s4, s5, lam, out):
    """Eigenvector of the symmetric tensor for the eigenvalue lam (the largest cross product of the rows)."""
    r00, r01, r02 = s0 - lam, s3, s5
    r10, r11, r12 = s3, s1 - lam, s4
    r20, r21, r22 = s5, s4, s2 - lam
    best = -1.0
    for c in range(3):
        if c == 0:
            x0, x1, x2 = r01 * r12 - r02 * r11, r02 * r10 - r00 * r12, r00 * r11 - r01 * r10
        elif c == 1:
            x0, x1, x2 = r01 * r22 - r02 * r21, r02 * r20 - r00 * r22, r00 * r21 - r01 * r20
        else:
            x0, x1, x2 = r11 * r22 - r12 * r21, r12 * r20 - r10 * r22, r10 * r21 - r11 * r20
        L = x0 * x0 + x1 * x1 + x2 * x2
        if L > best:
            best = L
            out[0], out[1], out[2] = x0, x1, x2
    L = math.sqrt(best)
    if L <= 0.0:
        out[0], out[1], out[2] = 0.0, 0.0, 1.0
    else:
        out[0] /= L
        out[1] /= L
        out[2] /= L


@njit(cache=True)
def seed_normals(s0, s1, s2, s3, s4, s5, seeds):
    """Principal directions of the tensor (3 normals) into seeds[0:3]."""
    e1, e2, e3 = principal3(s0, s1, s2, s3, s4, s5)
    v = np.empty(3)
    eigvec(s0, s1, s2, s3, s4, s5, e1, v)
    seeds[0, :] = v
    eigvec(s0, s1, s2, s3, s4, s5, e3, v)
    seeds[2, :] = v
    # third perpendicular to the first two
    seeds[1, 0] = seeds[0, 1] * seeds[2, 2] - seeds[0, 2] * seeds[2, 1]
    seeds[1, 1] = seeds[0, 2] * seeds[2, 0] - seeds[0, 0] * seeds[2, 2]
    seeds[1, 2] = seeds[0, 0] * seeds[2, 1] - seeds[0, 1] * seeds[2, 0]
    L = math.sqrt(seeds[1, 0] ** 2 + seeds[1, 1] ** 2 + seeds[1, 2] ** 2)
    if L < 1e-9:
        seeds[1, 0], seeds[1, 1], seeds[1, 2] = 0.0, 0.0, 1.0
    else:
        seeds[1, :] /= L


@njit(cache=True)
def dmg_pair(rng, mean, cnt, nc, scale, mscale, SD, ND, k, k2, Rm, mean_method):
    """(Miner D, fictitious D without the knee = Σ n·(Sa/SD)^k – the order of the planes also below the fatigue limit)."""
    D = 0.0
    Df = 0.0
    scale = abs(scale)
    for i in range(nc):
        Sa = 0.5 * rng[i] * scale
        Sm = mean[i] * mscale
        if mean_method != MEAN_NONE and Sm > 0.0 and Rm > 0.0:
            r = Sm / Rm
            if r >= 1.0:
                return math.inf, math.inf
            Sa = Sa / (1.0 - r) if mean_method == MEAN_GOODMAN else Sa / (1.0 - r * r)
        if Sa <= 0.0:
            continue
        q = Sa / SD
        qk = q ** k
        Df += cnt[i] * qk
        if Sa >= SD:
            D += cnt[i] * min(qk / ND, 1.0)          # N = ND·q^−k, N ≥ 1
        elif k2 > 0.0:
            D += cnt[i] * min(q ** k2 / ND, 1.0)
    return D, Df


@njit(cache=True)
def better(D, Df, bD, bDf):
    return D > bD or (D == bD and Df > bDf)


@njit(cache=True)
def pick_candidates(cand, order, nmax, cosmin, out):
    """The first nmax planes in the order order, mutually deviating by more than arccos(cosmin) (different vertices)."""
    m = 0
    for ii in range(order.shape[0]):
        i = order[ii]
        ok = True
        for jj in range(m):
            j = out[jj]
            if abs(cand[i, 0] * cand[j, 0] + cand[i, 1] * cand[j, 1] + cand[i, 2] * cand[j, 2]) > cosmin:
                ok = False
                break
        if ok:
            out[m] = i
            m += 1
            if m == nmax:
                break
    return m


# ====================================================================== rainflow with a time window
@njit(cache=True)
def reversals_idx(x, n, rev, ridx):
    """Turning points x[0:n] (as kernel.reversals) together with the sample indices."""
    if n < 2:
        if n == 1:
            rev[0] = x[0]
            ridx[0] = 0
            return 1
        return 0
    m = 0
    xc = x[1]
    ic = 1
    d_last = xc - x[0]
    rev[0] = x[0]
    ridx[0] = 0
    m = 1
    x_next = xc
    i_next = 1
    for i in range(2, n):
        x_next = x[i]
        i_next = i
        if x_next == xc:
            continue
        d_next = x_next - xc
        if d_last * d_next < 0:
            rev[m] = xc
            ridx[m] = ic
            m += 1
        xc = x_next
        ic = i
        d_last = d_next
    rev[m] = x_next
    ridx[m] = i_next
    m += 1
    return m


@njit(cache=True)
def window_max(x, y, n, ta, tb, tc, xa, xb):
    """max y from the start of the cycle to the point where x returns to the level xa behind tb (at the latest tc)."""
    t_end = tc + 1
    if xb > xa:
        for i in range(tb + 1, tc + 1):
            if x[i] <= xa:
                t_end = i
                break
    else:
        for i in range(tb + 1, tc + 1):
            if x[i] >= xa:
                t_end = i
                break
    m = y[ta]
    for i in range(ta + 1, min(t_end, tc + 1)):
        if y[i] > m:
            m = y[i]
    if t_end <= tc:              # y at the moment of the return to the level xa (linear between the samples)
        d = x[t_end] - x[t_end - 1]
        u = (xa - x[t_end - 1]) / d if d != 0.0 else 1.0
        yc = y[t_end - 1] + u * (y[t_end] - y[t_end - 1])
        if yc > m:
            m = yc
    return m


@njit(cache=True)
def window_max_residual(x, y, n, ta, tb, closed):
    """Residual half-cycle ta → tb. A closed block repeats → the loop closes in the next repetition: continues
    behind tb (across the end of the block to the start, x[n−1] = x[0]) until the return to the level x[ta]; otherwise only ta … tb.
    """
    m = y[ta]
    for i in range(ta + 1, tb + 1):
        if y[i] > m:
            m = y[i]
    if not closed or n < 3:
        return m
    xa = x[ta]
    up = x[tb] > xa
    prev = tb
    i = tb
    for _ in range(n):
        i = i + 1
        if i >= n:
            i = 1                       # x[n−1] = x[0] → next repetition from index 1
        if (up and x[i] <= xa) or ((not up) and x[i] >= xa):
            d = x[i] - x[prev]
            u = (xa - x[prev]) / d if d != 0.0 else 1.0
            yc = y[prev] + u * (y[i] - y[prev])
            if yc > m:
                m = yc
            break
        if y[i] > m:
            m = y[i]
        prev = i
    return m


@njit(cache=True)
def rainflow_win(x, y, n, rev, ridx, stk, rng, mean, cnt, yw, closed):
    """Rainflow x[0:n] (identical to kernel.rainflow_cycles) + yw = max y in the window of each loop."""
    m = reversals_idx(x, n, rev, ridx)
    sp = 0
    start = 0
    nc = 0
    for i in range(m):
        stk[sp] = i
        sp += 1
        while sp - start >= 3:
            p, q, r = stk[sp - 3], stk[sp - 2], stk[sp - 1]
            X = abs(rev[q] - rev[r])
            Y = abs(rev[p] - rev[q])
            if X < Y:
                break
            rng[nc] = Y
            mean[nc] = 0.5 * (rev[p] + rev[q])
            yw[nc] = window_max(x, y, n, ridx[p], ridx[q], ridx[r], rev[p], rev[q])
            if sp - start == 3:
                cnt[nc] = 0.5
                nc += 1
                start += 1
            else:
                cnt[nc] = 1.0
                nc += 1
                stk[sp - 3] = r
                sp -= 2
    for j in range(start, sp - 1):
        p, q = stk[j], stk[j + 1]
        rng[nc] = abs(rev[p] - rev[q])
        mean[nc] = 0.5 * (rev[p] + rev[q])
        yw[nc] = window_max_residual(x, y, n, ridx[p], ridx[q], closed)
        cnt[nc] = 0.5
        nc += 1
    return nc


@njit(cache=True)
def close_pair(x, y, T, closed, xc, yc):
    """Closed block (start and end at max |x|) into xc, yc; returns the length."""
    if not closed:
        for i in range(T):
            xc[i] = x[i]
            yc[i] = y[i]
        return T
    j = 0
    best = -1.0
    for i in range(T):
        if abs(x[i]) > best:
            best = abs(x[i])
            j = i
    for i in range(T):
        xc[i] = x[(j + i) % T]
        yc[i] = y[(j + i) % T]
    xc[T] = x[j]
    yc[T] = y[j]
    return T + 1


# ====================================================================== general (non-proportional) loading
# Working array of the node: w (W_ROWS, T+2) float, iw (2, T+2) int (indices of the turning points, a stack).
W_X, W_XC, W_YC, W_REV, W_RNG, W_MEAN, W_CNT, W_YW, W_RNG2, W_SN, W_TU, W_TV = range(12)
W_ROWS = 12


@njit(cache=True)
def project(G, B, cw, T, nx, ny, nz, ax, ay, az, bx, by, bz, sn, tu, tv):
    """σn(t) and the shear components into the directions a, b in the plane. σ(t) = Σ_k G[t, k]·B[k] (G = multiples of the states F and B = stresses of the states
    at the node, if there are ≤ 6 states, otherwise G = the history of σ and B = unit components) → projection only K numbers per time.
    """
    Kc = B.shape[0]
    for kk in range(Kc):
        s0, s1, s2, s3, s4, s5 = B[kk, 0], B[kk, 1], B[kk, 2], B[kk, 3], B[kk, 4], B[kk, 5]
        tx = s0 * nx + s3 * ny + s5 * nz
        ty = s3 * nx + s1 * ny + s4 * nz
        tz = s5 * nx + s4 * ny + s2 * nz
        cw[0, kk] = tx * nx + ty * ny + tz * nz
        cw[1, kk] = tx * ax + ty * ay + tz * az
        cw[2, kk] = tx * bx + ty * by + tz * bz
    for t in range(T):
        a0 = 0.0
        a1 = 0.0
        a2 = 0.0
        for kk in range(Kc):
            g = G[t, kk]
            a0 += g * cw[0, kk]
            a1 += g * cw[1, kk]
            a2 += g * cw[2, kk]
        sn[t] = a0
        tu[t] = a1
        tv[t] = a2


@njit(cache=True)
def rainflow_plain(x, n, rev, ridx, stk, rng, mean, cnt):
    """Rainflow x[0:n] without a window (criterion σn) – identical to kernel.rainflow_cycles, without allocations."""
    m = reversals_idx(x, n, rev, ridx)
    sp = 0
    start = 0
    nc = 0
    for i in range(m):
        stk[sp] = i
        sp += 1
        while sp - start >= 3:
            p, q, r = stk[sp - 3], stk[sp - 2], stk[sp - 1]
            X = abs(rev[q] - rev[r])
            Y = abs(rev[p] - rev[q])
            if X < Y:
                break
            rng[nc] = Y
            mean[nc] = 0.5 * (rev[p] + rev[q])
            if sp - start == 3:
                cnt[nc] = 0.5
                nc += 1
                start += 1
            else:
                cnt[nc] = 1.0
                nc += 1
                stk[sp - 3] = r
                sp -= 2
    for j in range(start, sp - 1):
        p, q = stk[j], stk[j + 1]
        rng[nc] = abs(rev[p] - rev[q])
        mean[nc] = 0.5 * (rev[p] + rev[q])
        cnt[nc] = 0.5
        nc += 1
    return nc


@njit(cache=True)
def eval_dir(T, phi, crit, kF, lam, closed, SD, ND, k, k2, Rm, mean_method, w, iw, final):
    """Cycles and damage in the plane (w[W_SN], w[W_TU], w[W_TV] already computed) for the shear direction phi.
    Returns (D, Df, nc); the cycles (equivalent range, mean, count) stay in w[W_RNG], w[W_MEAN], w[W_CNT].
    """
    sn, tu, tv = w[W_SN], w[W_TU], w[W_TV]
    xc, yc, rev = w[W_XC], w[W_YC], w[W_REV]
    rng, mean, cnt = w[W_RNG], w[W_MEAN], w[W_CNT]
    ridx, stk = iw[0], iw[1]
    if crit == CP_NORMAL:
        n = close_pair(sn, sn, T, closed, xc, yc)
        nc = rainflow_plain(xc, n, rev, ridx, stk, rng, mean, cnt)
        if final:
            nc = merge_cycles(rng, mean, cnt, nc)
        D, Df = dmg_pair(rng, mean, cnt, nc, 1.0, 1.0, SD, ND, k, k2, Rm, mean_method)
        return D, Df, nc
    x, yw, rng2 = w[W_X], w[W_YW], w[W_RNG2]
    c, s = math.cos(phi), math.sin(phi)
    for t in range(T):
        x[t] = c * tu[t] + s * tv[t]
    n = close_pair(x, sn, T, closed, xc, yc)
    nc = rainflow_win(xc, yc, n, rev, ridx, stk, rng2, mean, cnt, yw, closed)
    for i in range(nc):
        f = 0.5 * rng2[i] + kF * yw[i]           # Findley τa + k·σn,max
        rng[i] = 2.0 * max(f, 0.0) / lam         # as the range of the equivalent σ
        mean[i] = 0.0
    if final:
        nc = merge_cycles(rng, mean, cnt, nc)
    D, Df = dmg_pair(rng, mean, cnt, nc, 1.0, 1.0, SD, ND, k, k2, Rm, MEAN_NONE)
    return D, Df, nc


@njit(cache=True)
def eval_plane(G, B, cw, T, nx, ny, nz, ndirs, dphi, centered, rx, ry, rz, crit, kF, lam, closed, SD, ND, k, k2, Rm,
               mean_method, w, iw):
    """Exact damage of the plane n (the worst of the shear directions). Directions: centered → phi0 + (j − ndirs//2)·dphi,
    phi0 = projection of the reference direction r into the plane; otherwise j·dphi from the axis a. Returns (D, Df, dx, dy, dz).
    """
    ax, ay, az, bx, by, bz = basis(nx, ny, nz)
    project(G, B, cw, T, nx, ny, nz, ax, ay, az, bx, by, bz, w[W_SN], w[W_TU], w[W_TV])
    if crit == CP_NORMAL:
        D, Df, _ = eval_dir(T, 0.0, crit, kF, lam, closed, SD, ND, k, k2, Rm, mean_method, w, iw, False)
        return D, Df, 0.0, 0.0, 0.0
    phi0 = math.atan2(rx * bx + ry * by + rz * bz, rx * ax + ry * ay + rz * az) if centered else 0.0
    bD, bDf, bphi = -1.0, -1.0, phi0
    for j in range(ndirs):
        phi = phi0 + ((j - ndirs // 2) if centered else j) * dphi
        D, Df, _ = eval_dir(T, phi, crit, kF, lam, closed, SD, ND, k, k2, Rm, mean_method, w, iw, False)
        if better(D, Df, bD, bDf):
            bD, bDf, bphi = D, Df, phi
    c, s = math.cos(bphi), math.sin(bphi)
    return bD, bDf, c * ax + s * bx, c * ay + s * by, c * az + s * bz


@njit(cache=True)
def proxy_plane(G, B, cw, T, nx, ny, nz, crit, kF, lam, ndir, Rm, mean_method, w):
    """Cheap estimate (the range of the history without rainflow, exact for a constant amplitude) for the selection of candidates."""
    ax, ay, az, bx, by, bz = basis(nx, ny, nz)
    sn, tu, tv = w[W_SN], w[W_TU], w[W_TV]
    project(G, B, cw, T, nx, ny, nz, ax, ay, az, bx, by, bz, sn, tu, tv)
    smax, smin = sn[0], sn[0]
    for t in range(1, T):
        if sn[t] > smax:
            smax = sn[t]
        if sn[t] < smin:
            smin = sn[t]
    if crit == CP_NORMAL:
        a = 0.5 * (smax - smin)
        m = 0.5 * (smax + smin)
        if mean_method != MEAN_NONE and m > 0.0 and Rm > 0.0:
            r = m / Rm
            if r >= 1.0:
                return math.inf
            a = a / (1.0 - r) if mean_method == MEAN_GOODMAN else a / (1.0 - r * r)
        return a
    best = 0.0
    for j in range(ndir):
        phi = j * math.pi / ndir
        c, s = math.cos(phi), math.sin(phi)
        hi = c * tu[0] + s * tv[0]
        lo = hi
        for t in range(1, T):
            v = c * tu[t] + s * tv[t]
            if v > hi:
                hi = v
            elif v < lo:
                lo = v
        if hi - lo > best:
            best = hi - lo
    return (0.5 * best + kF * smax) / lam


@njit(cache=True)
def neighbour(nx, ny, nz, ax, ay, az, bx, by, bz, step, j, lev):
    """j-th of the 6 neighbours of the normal at the distance step (hexagon, rotated by 30° at odd levels)."""
    ang = j * math.pi / 3.0 + (math.pi / 6.0 if (lev % 2) else 0.0)
    ca, sa = math.cos(ang), math.sin(ang)
    cx = nx + step * (ca * ax + sa * bx)
    cy = ny + step * (ca * ay + sa * by)
    cz = nz + step * (ca * az + sa * bz)
    L = math.sqrt(cx * cx + cy * cy + cz * cz)
    return cx / L, cy / L, cz / L


@njit(cache=True)
def cp_node_general(G, B, cw, T, planes, step0, levels, ndir, ndir_proxy, ncand, exact_coarse, crit, kF, lam,
                    closed, SD, ND, k, k2, Rm, mean_method, repeats, w, iw):
    """Critical plane at the node, σ(t) = Σ_k G[t, k]·B[k]. Returns (D, sf, amp, nx, ny, nz, dx, dy, dz)."""
    P = planes.shape[0]
    Kc = B.shape[0]
    # seeds: principal directions at the moment of the largest stress
    tbest, vbest = 0, -1.0
    s = np.zeros(6)
    for t in range(T):
        for c in range(6):
            acc = 0.0
            for kk in range(Kc):
                acc += G[t, kk] * B[kk, c]
            s[c] = acc
        v = s[0] ** 2 + s[1] ** 2 + s[2] ** 2 + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
        if v > vbest:
            vbest, tbest = v, t
    for c in range(6):
        acc = 0.0
        for kk in range(Kc):
            acc += G[tbest, kk] * B[kk, c]
        s[c] = acc
    cand = np.empty((P + 3, 3))
    cand[:P] = planes
    seed_normals(s[0], s[1], s[2], s[3], s[4], s[5], cand[P:])
    nall = P + 3
    prox = np.empty(nall)
    if exact_coarse:          # exact damage already on the coarse mesh (fewer shear directions)
        proxf = np.empty(nall)
        for i in range(nall):
            prox[i], proxf[i], _, _, _ = eval_plane(G, B, cw, T, cand[i, 0], cand[i, 1], cand[i, 2], ndir_proxy,
                                                    math.pi / ndir_proxy, False, 0.0, 0.0, 0.0, crit, kF, lam,
                                                    closed, SD, ND, k, k2, Rm, mean_method, w, iw)
        if prox.max() <= 0.0:
            prox[:] = proxf
    else:                     # odhad z rozkmitu
        for i in range(nall):
            prox[i] = proxy_plane(G, B, cw, T, cand[i, 0], cand[i, 1], cand[i, 2], crit, kF, lam, ndir_proxy, Rm,
                                  mean_method, w)
    order = np.argsort(-prox)
    sel = np.empty(ncand, np.int64)
    nsel = pick_candidates(cand, order, ncand, math.cos(2.0 * step0), sel)
    nd_loc = 1                # when refining the plane only the direction taken from the previous plane (refined at the end)
    dphi0 = math.pi / ndir
    gD, gDf = -1.0, -1.0
    gx, gy, gz, gdx, gdy, gdz = 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
    for ci in range(nsel):
        i = sel[ci]
        nx, ny, nz = cand[i, 0], cand[i, 1], cand[i, 2]
        D, Df, dx, dy, dz = eval_plane(G, B, cw, T, nx, ny, nz, ndir, dphi0, False, 0.0, 0.0, 0.0, crit, kF, lam,
                                       closed, SD, ND, k, k2, Rm, mean_method, w, iw)
        step = step0
        dphi = dphi0
        for lev in range(levels):              # pattern search, the step is halved at each level
            step *= 0.5
            dphi *= 0.5
            for _mv in range(6):
                ax, ay, az, bx, by, bz = basis(nx, ny, nz)
                moved = False
                mx, my, mz, mdx, mdy, mdz = nx, ny, nz, dx, dy, dz
                for j in range(6):
                    cx, cy, cz = neighbour(nx, ny, nz, ax, ay, az, bx, by, bz, step, j, lev)
                    D2, Df2, ex, ey, ez = eval_plane(G, B, cw, T, cx, cy, cz, nd_loc, dphi, True, dx, dy, dz, crit, kF,
                                                     lam, closed, SD, ND, k, k2, Rm, mean_method, w, iw)
                    if better(D2, Df2, D, Df):
                        D, Df = D2, Df2
                        mx, my, mz, mdx, mdy, mdz = cx, cy, cz, ex, ey, ez
                        moved = True
                if not moved:
                    break
                nx, ny, nz, dx, dy, dz = mx, my, mz, mdx, mdy, mdz
            if crit == CP_FINDLEY:             # rotate the shear direction (±dphi) at the end of the level
                D2, Df2, ex, ey, ez = eval_plane(G, B, cw, T, nx, ny, nz, 3, dphi, True, dx, dy, dz, crit, kF, lam,
                                                 closed, SD, ND, k, k2, Rm, mean_method, w, iw)
                if better(D2, Df2, D, Df):
                    D, Df, dx, dy, dz = D2, Df2, ex, ey, ez
        if better(D, Df, gD, gDf):
            gD, gDf = D, Df
            gx, gy, gz, gdx, gdy, gdz = nx, ny, nz, dx, dy, dz
    if crit == CP_FINDLEY:                     # refinement of the shear direction in the found plane
        dphi = dphi0 * 0.5 ** levels
        for _ in range(3):
            dphi *= 0.5
            D2, Df2, ex, ey, ez = eval_plane(G, B, cw, T, gx, gy, gz, 3, dphi, True, gdx, gdy, gdz, crit, kF, lam,
                                             closed, SD, ND, k, k2, Rm, mean_method, w, iw)
            gdx, gdy, gdz = ex, ey, ez         # the middle direction = the current one → never worse
    # final cycles in the critical plane and direction → damage, safety, amplitude
    ax, ay, az, bx, by, bz = basis(gx, gy, gz)
    project(G, B, cw, T, gx, gy, gz, ax, ay, az, bx, by, bz, w[W_SN], w[W_TU], w[W_TV])
    phi = math.atan2(gdx * bx + gdy * by + gdz * bz, gdx * ax + gdy * ay + gdz * az)
    _, _, nc = eval_dir(T, phi, crit, kF, lam, closed, SD, ND, k, k2, Rm, mean_method, w, iw, True)
    mm = mean_method if crit == CP_NORMAL else MEAN_NONE
    D, sf, amp = node_result(w[W_RNG], w[W_MEAN], w[W_CNT], nc, 1.0, 1.0, SD, ND, k, k2, Rm, mm, repeats)
    return D, sf, amp, gx, gy, gz, gdx, gdy, gdz


@njit(parallel=True, cache=True)
def cp_nodes(F, S, planes, step0, levels, ndir, search, crit, kF, lam, mean_method, SD, ND, k, k2, Rm, repeats,
             closed, out_D, out_sf, out_amp, out_n, out_d):
    """Critical plane, general loading: F (T, K) multiples, S (K, n, 6); criterion and S-N per node.
    search = (number of candidates, shear directions on the coarse mesh, 1 = exact damage on the coarse mesh / 0 = estimate).
    """
    T, K = F.shape
    n = S.shape[1]
    L = T + 2
    ncand, ndir_proxy, exact_coarse = search[0], search[1], search[2] != 0
    for j in prange(n):
        if K <= 6:                       # projection over the unit states (K numbers per time)
            G = F
            B = S[:, j, :].copy()
        else:                            # history of σ (6 components per time)
            G = np.zeros((T, 6))
            for t in range(T):
                for kk in range(K):
                    f = F[t, kk]
                    if f != 0.0:
                        for c in range(6):
                            G[t, c] += f * S[kk, j, c]
            B = np.eye(6)
        cw = np.empty((3, B.shape[0]))
        w = np.empty((W_ROWS, L))
        iw = np.empty((2, L), np.int64)
        D, sf, amp, nx, ny, nz, dx, dy, dz = cp_node_general(
            G, B, cw, T, planes, step0, levels, ndir, ndir_proxy, ncand, exact_coarse, crit[j], kF[j], lam[j],
            closed, SD[j], ND[j], k[j], k2[j], Rm[j], mean_method, repeats, w, iw)
        out_D[j], out_sf[j], out_amp[j] = D, sf, amp
        out_n[j, 0], out_n[j, 1], out_n[j, 2] = nx, ny, nz
        out_d[j, 0], out_d[j, 1], out_d[j, 2] = dx, dy, dz


@njit(parallel=True, cache=True)
def cp_nodes_exhaustive(F, S, planes, ndir, crit, kF, lam, mean_method, SD, ND, k, k2, Rm, closed, out_D, out_n):
    """Reference search by brute force: exact damage on all planes × ndir directions (slow, for checking the fast search). Returns the max. damage per block and its normal."""
    T, K = F.shape
    n = S.shape[1]
    L = T + 2
    for j in prange(n):
        B = S[:, j, :].copy()
        cw = np.empty((3, K))
        w = np.empty((W_ROWS, L))
        iw = np.empty((2, L), np.int64)
        gD, gDf = -1.0, -1.0
        for i in range(planes.shape[0]):
            D, Df, _, _, _ = eval_plane(F, B, cw, T, planes[i, 0], planes[i, 1], planes[i, 2], ndir, math.pi / ndir,
                                        False, 0.0, 0.0, 0.0, crit[j], kF[j], lam[j], closed, SD[j], ND[j], k[j],
                                        k2[j], Rm[j], mean_method, w, iw)
            if better(D, Df, gD, gDf):
                gD, gDf = D, Df
                out_n[j, 0], out_n[j, 1], out_n[j, 2] = planes[i, 0], planes[i, 1], planes[i, 2]
        out_D[j] = gD


# ====================================================================== proportional loading
@njit(cache=True)
def prop_eval(S6, nx, ny, nz, rng, mean, cnt, nc, crit, kF, lam, SD, ND, k, k2, Rm, mean_method, br, bm):
    """Plane n at σ(t) = f(t)·S: σn = f·sn, shear = f·τ (a line, amplitude = |τ|·Δf/2).
    Findley: σn,max in the loop = sn·fm + |sn|·Δf/2. Returns (D, Df, sn, tau); for Findley the cycles in br, bm.
    """
    tx = S6[0] * nx + S6[3] * ny + S6[5] * nz
    ty = S6[3] * nx + S6[1] * ny + S6[4] * nz
    tz = S6[5] * nx + S6[4] * ny + S6[2] * nz
    sn = tx * nx + ty * ny + tz * nz
    tau = math.sqrt(max(0.0, tx * tx + ty * ty + tz * tz - sn * sn))
    if crit == CP_NORMAL:
        D, Df = dmg_pair(rng, mean, cnt, nc, sn, sn, SD, ND, k, k2, Rm, mean_method)
        return D, Df, sn, tau
    for i in range(nc):
        f = tau * 0.5 * rng[i] + kF * (sn * mean[i] + abs(sn) * 0.5 * rng[i])
        br[i] = 2.0 * max(f, 0.0) / lam
        bm[i] = 0.0
    D, Df = dmg_pair(br, bm, cnt, nc, 1.0, 1.0, SD, ND, k, k2, Rm, MEAN_NONE)
    return D, Df, sn, tau


@njit(parallel=True, cache=True)
def cp_nodes_proportional(f, Seff, planes, step0, levels, crit, kF, lam, mean_method, SD, ND, k, k2, Rm,
                          repeats, closed, out_D, out_sf, out_amp, out_n, out_d):
    """Proportional loading σ(t) = f(t)·Seff: rainflow of f only once, planes only over the cycles of f."""
    T = f.shape[0]
    x = np.empty(T + 1)
    y = np.empty(T + 1)
    n = close_pair(f, f, T, closed, x, y)
    L = n + 1
    rng = np.empty(L)
    mean = np.empty(L)
    cnt = np.empty(L)
    nc = rainflow_cycles(x[:n].copy(), rng, mean, cnt)
    nc = merge_cycles(rng, mean, cnt, nc)
    P = planes.shape[0]
    nall = P + 3
    for j in prange(Seff.shape[0]):
        S6 = Seff[j]
        cr, kf, lm = crit[j], kF[j], lam[j]
        br = np.empty(L)
        bm = np.empty(L)
        cand = np.empty((nall, 3))
        cand[:P] = planes
        seed_normals(S6[0], S6[1], S6[2], S6[3], S6[4], S6[5], cand[P:])
        if cr == CP_NORMAL:
            # σn = f·sn: the damage grows with |sn| at the same sign → maximum in the principal direction e1 or e3
            D1, Df1, _, _ = prop_eval(S6, cand[P, 0], cand[P, 1], cand[P, 2], rng, mean, cnt, nc, cr, kf, lm,
                                      SD[j], ND[j], k[j], k2[j], Rm[j], mean_method, br, bm)
            D3, Df3, _, _ = prop_eval(S6, cand[P + 2, 0], cand[P + 2, 1], cand[P + 2, 2], rng, mean, cnt, nc, cr, kf,
                                      lm, SD[j], ND[j], k[j], k2[j], Rm[j], mean_method, br, bm)
            i = P + 2 if better(D3, Df3, D1, Df1) else P
            gx, gy, gz = cand[i, 0], cand[i, 1], cand[i, 2]
            _, _, sn, tau = prop_eval(S6, gx, gy, gz, rng, mean, cnt, nc, cr, kf, lm, SD[j], ND[j], k[j], k2[j],
                                      Rm[j], mean_method, br, bm)
            out_D[j], out_sf[j], out_amp[j] = node_result(rng, mean, cnt, nc, sn, sn, SD[j], ND[j], k[j], k2[j],
                                                          Rm[j], mean_method, repeats)
            out_n[j, 0], out_n[j, 1], out_n[j, 2] = gx, gy, gz
            out_d[j, 0], out_d[j, 1], out_d[j, 2] = 0.0, 0.0, 0.0
            continue
        sc = np.empty(nall)
        scf = np.empty(nall)
        for i in range(nall):
            sc[i], scf[i], _, _ = prop_eval(S6, cand[i, 0], cand[i, 1], cand[i, 2], rng, mean, cnt, nc, cr, kf, lm,
                                            SD[j], ND[j], k[j], k2[j], Rm[j], mean_method, br, bm)
        order = np.argsort(-sc) if sc.max() > 0.0 else np.argsort(-scf)
        sel = np.empty(N_CAND_PROP, np.int64)
        nsel = pick_candidates(cand, order, N_CAND_PROP, math.cos(2.0 * step0), sel)
        gD, gDf = -1.0, -1.0
        gx, gy, gz = 0.0, 0.0, 1.0
        for ci in range(nsel):
            i = sel[ci]
            nx, ny, nz = cand[i, 0], cand[i, 1], cand[i, 2]
            D, Df = sc[i], scf[i]
            step = step0
            for lev in range(levels + 2):       # the planes are cheap → 2 levels finer
                step *= 0.5
                for _mv in range(8):
                    ax, ay, az, bx, by, bz = basis(nx, ny, nz)
                    moved = False
                    mx, my, mz = nx, ny, nz
                    for jj in range(6):
                        cx, cy, cz = neighbour(nx, ny, nz, ax, ay, az, bx, by, bz, step, jj, lev)
                        D2, Df2, _, _ = prop_eval(S6, cx, cy, cz, rng, mean, cnt, nc, cr, kf, lm, SD[j], ND[j],
                                                  k[j], k2[j], Rm[j], mean_method, br, bm)
                        if better(D2, Df2, D, Df):
                            D, Df, mx, my, mz = D2, Df2, cx, cy, cz
                            moved = True
                    if not moved:
                        break
                    nx, ny, nz = mx, my, mz
            if better(D, Df, gD, gDf):
                gD, gDf, gx, gy, gz = D, Df, nx, ny, nz
        # result in the critical plane
        _, _, sn, tau = prop_eval(S6, gx, gy, gz, rng, mean, cnt, nc, cr, kf, lm, SD[j], ND[j], k[j], k2[j],
                                  Rm[j], mean_method, br, bm)
        if cr == CP_NORMAL:
            out_D[j], out_sf[j], out_amp[j] = node_result(rng, mean, cnt, nc, sn, sn, SD[j], ND[j], k[j], k2[j],
                                                          Rm[j], mean_method, repeats)
        else:
            out_D[j], out_sf[j], out_amp[j] = node_result(br, bm, cnt, nc, 1.0, 1.0, SD[j], ND[j], k[j], k2[j],
                                                          Rm[j], MEAN_NONE, repeats)
        out_n[j, 0], out_n[j, 1], out_n[j, 2] = gx, gy, gz
        if tau > 1e-12 * (abs(sn) + 1e-300):
            tx = S6[0] * gx + S6[3] * gy + S6[5] * gz
            ty = S6[3] * gx + S6[1] * gy + S6[4] * gz
            tz = S6[5] * gx + S6[4] * gy + S6[2] * gz
            out_d[j, 0] = (tx - sn * gx) / tau
            out_d[j, 1] = (ty - sn * gy) / tau
            out_d[j, 2] = (tz - sn * gz) / tau
        else:
            out_d[j, 0], out_d[j, 1], out_d[j, 2] = 0.0, 0.0, 0.0


# ====================================================================== histories for the graph and the check
def plane_series(sig: np.ndarray, n, d) -> tuple[np.ndarray, np.ndarray]:
    """(σn(t), τ_d(t)) in the plane n and the shear direction d; sig (T, 6) = xx yy zz xy yz zx."""
    n = np.asarray(n, float)
    d = np.asarray(d, float)
    Tm = np.empty((len(sig), 3, 3))
    Tm[:, 0, 0], Tm[:, 1, 1], Tm[:, 2, 2] = sig[:, 0], sig[:, 1], sig[:, 2]
    Tm[:, 0, 1] = Tm[:, 1, 0] = sig[:, 3]
    Tm[:, 1, 2] = Tm[:, 2, 1] = sig[:, 4]
    Tm[:, 0, 2] = Tm[:, 2, 0] = sig[:, 5]
    tr = Tm @ n
    return tr @ n, tr @ d
