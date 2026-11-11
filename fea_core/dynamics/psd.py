"""Random vibrations (D3) and harmonic response (D1) by the modal superposition – numerical cores without the mesh and ccx.

CONVENTIONS (valid for the whole module):
  - frequency f [Hz], ω = 2πf [rad/s]; the PSDs are ONE-SIDED in Hz: ∫₀^∞ G(f) df = variance (RMS²).
  - the ASD of the acceleration is entered in g²/Hz (as in the standards IEC 60068-2-64); converted to (mm/s²)²/Hz only inside the computation
    (`G_MM` = 9810 mm/s², `asd_to_mm`). G_rms in g.
  - Modes normalised by the mass; modal FRF H_j(ω) = 1 / (ω_j² − ω² + 2i·ζ_j·ω_j·ω) [per unit modal force],
    with structural (hysteretic) damping H_j = 1 / (ω_j²·(1 + i·η_j) − ω²) (η = loss factor = 2ζ at the resonance).
  - Excitation by the acceleration of the base a_b in the direction r: modal force p_j = −Γ_j·a_b (Γ_j = φ_jᵀM·r with the sign of ccx) →
    relative displacement u_rel = Σ φ_j·q_j, q_j = −Γ_j·H_j·a_b. Modal transfers Z_j(f) according to `kind`:
        disp     −Γ_j·H_j                       relative displacement (and stress: σ = Σ σ_j·q_j)
        vel      −Γ_j·iω·H_j                    relative velocity
        acc_rel   Γ_j·ω²·H_j                    relative acceleration
        acc_abs   Γ_j·(ω_j² + 2iζ_jω_jω)·H_j    absolute acceleration (= relative + base for the complete basis;
                                                 with a truncated basis the "missing mass" residual is missing).
  - Modal moments M^(p)_jk = ∫ f^p·Z_j·Z_k*·W df (trapezium on a non-uniform grid, W in (mm/s²)²/Hz) – a Hermitian
    matrix K × K. The response r = Σ_j A_j·q_j with real modal coefficients A_j (component of the shape, modal stress) has the spectral moment
    m_p = Re(Aᵀ·M^(p)·A) = Aᵀ·Re(M^(p))·A = CQC in the FULL form (the correlation of close modes is in M).
  - Equivalent stress (Preumont & Piéfort 1994): G_eq(f) = tr(Q·G_σ(f)), Q = matrix of von Mises for the components of ccx
    in the order SXX SYY SZZ SXY SYZ SZX (the tensor shear components): σ_vm² = σᵀ·Q·σ. The m0 of the equivalent = RMS² of von Mises
    according to Segalman (Sandia 1997) – informative only (von Mises is not Gaussian); for the spectral fatigue
    (`fatigue.spectral`) the moments m0, m1, m2, m4 of the equivalent serve.
  - Grid: `freq_grid` (log base + densening around the resonances uniformly in arctg – the Lorentzian peak is thus integrated by the trapezium
    almost exactly), `ssd_grid` = points as ccx *STEADY STATE DYNAMICS (verified by the trial
    `scratch/f8_kernels/ssd_bias.py`: in each interval between the fixed points n points x = ½ + ½·sgn(y)·|y|^(1/bias),
    y = −1 + 2k/(n − 1); fixed points = the limits of the range and the natural frequencies inside, the same ones merge).
"""
from __future__ import annotations

import math

import numpy as np

from ..modules import need

_numba = need("numba", "accel")
njit = _numba.njit
prange = _numba.prange

G_MM = 9810.0                      # g [mm/s²]
LOG2_DB = 10.0 * math.log10(2.0)   # 3.0103 dB per octave = proportional to f¹

# matrix of von Mises for the components (SXX, SYY, SZZ, SXY, SYZ, SZX): σ_vm² = σᵀ·Q·σ
Q_MISES = np.array([[1.0, -0.5, -0.5, 0.0, 0.0, 0.0],
                    [-0.5, 1.0, -0.5, 0.0, 0.0, 0.0],
                    [-0.5, -0.5, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 3.0]])

KINDS = ("disp", "vel", "acc_rel", "acc_abs")


# ------------------------------------------------------------------ profil ASD

def _table(table) -> np.ndarray:
    t = np.asarray(table, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] != 2 or t.shape[0] < 1:
        raise ValueError("Table must have the form [[f Hz, value], …]")
    if np.any(t[:, 0] <= 0) or np.any(np.diff(t[:, 0]) <= 0):
        raise ValueError("Table frequencies must be positive and increasing")
    if np.any(t[:, 1] < 0):
        raise ValueError("Table values must not be negative")
    return t


def loglog_interp(table, f) -> np.ndarray:
    """Interpolation of a table [[f, y], …] between the points log-log (a section with a zero value linearly), outside the range 0."""
    t = _table(table)
    f = np.asarray(f, dtype=np.float64)
    out = np.zeros(f.shape)
    if t.shape[0] == 1:
        out[np.isclose(f, t[0, 0], rtol=1e-12)] = t[0, 1]
        return out
    inside = (f >= t[0, 0]) & (f <= t[-1, 0])
    fi = f[inside]
    i = np.clip(np.searchsorted(t[:, 0], fi, side="right") - 1, 0, t.shape[0] - 2)
    f1, f2 = t[i, 0], t[i + 1, 0]
    y1, y2 = t[i, 1], t[i + 1, 1]
    pos = (y1 > 0) & (y2 > 0)
    ratio = np.where(pos, y2, 1.0) / np.where(pos, y1, 1.0)
    b = np.log(ratio) / np.log(f2 / f1)                   # slope in log-log (0 in a section with a zero)
    ylog = y1 * (fi / f1) ** b
    ylin = y1 + (y2 - y1) * (fi - f1) / (f2 - f1)
    out[inside] = np.where(pos, ylog, ylin)
    return out


def asd_interp(table, f) -> np.ndarray:
    """ASD [[f Hz, g²/Hz], …] at the points f – between the points log-log (as in the standards), outside the range 0. The result g²/Hz."""
    return loglog_interp(table, f)


def asd_to_mm(W_g) -> np.ndarray:
    """g²/Hz → (mm/s²)²/Hz."""
    return np.asarray(W_g, dtype=np.float64) * G_MM * G_MM


def asd_from_slopes(points, slopes_db_oct, grms: float | None = None) -> np.ndarray:
    """Profile of the ASD from the points and slopes (the notation of the standards).

    points = [[f Hz, W g²/Hz or None], …] (increasing f); slopes_db_oct = slope of the section i (between the points i and i+1)
    in dB/oct or None (both ends given). Unknown values are computed W₂ = W₁·(f₂/f₁)^(s/3.0103) forwards and backwards.
    grms = target value of g_rms → the whole profile is rescaled. Returns a table (n, 2) [[f, g²/Hz], …].
    """
    pts = [[float(p[0]), None if p[1] is None else float(p[1])] for p in points]
    n = len(pts)
    if n < 2:
        raise ValueError("ASD profile needs at least 2 points")
    sl = list(slopes_db_oct) if slopes_db_oct is not None else [None] * (n - 1)
    if len(sl) != n - 1:
        raise ValueError(f"The number of slopes must be one less than the number of points ({n - 1})")
    for i in range(n - 1):
        if pts[i + 1][0] <= pts[i][0] or pts[i][0] <= 0:
            raise ValueError("Profile frequencies must be positive and increasing")
    changed = True
    while changed:
        changed = False
        for i in range(n - 1):
            s = sl[i]
            if s is None:
                continue
            r = (pts[i + 1][0] / pts[i][0]) ** (float(s) / LOG2_DB)
            if pts[i][1] is not None and pts[i + 1][1] is None:
                pts[i + 1][1] = pts[i][1] * r
                changed = True
            elif pts[i + 1][1] is not None and pts[i][1] is None:
                pts[i][1] = pts[i + 1][1] / r
                changed = True
    if any(p[1] is None for p in pts):
        raise ValueError("ASD profile cannot be completed – a value or slope is missing")
    t = np.array(pts, dtype=np.float64)
    if grms is not None:
        g0 = asd_rms(t)
        if g0 <= 0:
            raise ValueError("ASD profile has zero g_rms – it cannot be rescaled")
        t[:, 1] *= (float(grms) / g0) ** 2
    return t


def asd_rms(table) -> float:
    """g_rms of the profile = √(∫W df) – the exact integral of the log-log sections (W = W₁·(f/f₁)^b)."""
    t = _table(table)
    s = 0.0
    for i in range(t.shape[0] - 1):
        f1, f2 = t[i, 0], t[i + 1, 0]
        w1, w2 = t[i, 1], t[i + 1, 1]
        if w1 <= 0 or w2 <= 0:
            s += 0.5 * (w1 + w2) * (f2 - f1)
            continue
        b = math.log(w2 / w1) / math.log(f2 / f1)
        if abs(b + 1.0) < 1e-12:
            s += w1 * f1 * math.log(f2 / f1)
        else:
            s += w1 * f1 / (b + 1.0) * ((f2 / f1) ** (b + 1.0) - 1.0)
    return math.sqrt(s)


def miles(fn: float, Q: float, W: float) -> float:
    """Miles' formula G_rms = √(π/2·f_n·Q·W) [g] (W in g²/Hz, Q = 1/(2ζ)) – check of the 1-DOF absolute acceleration."""
    return math.sqrt(math.pi / 2.0 * fn * Q * W)


# ------------------------------------------------------------------ frequency grids

def freq_grid(f_min: float, f_max: float, fn=(), zeta=0.02, per_peak: int = 41, per_decade: int = 60,
              ratio: float = 1.03, span: float = 3.0) -> np.ndarray:
    """Non-uniform grid for the integration of the PSD.

    - log base: per_decade points per decade in [f_min, f_max];
    - at each natural frequency at the peak ±½ of the bandwidth (width = 2ζf_n, i.e. |x| ≤ 1, x = (f − f_n)/(ζf_n))
      per_peak points uniformly (a point exactly at the resonance);
    - further from the peak geometrically at the distance |f − f_n| with the ratio `ratio` up to 0.1·f_n below and span·f_n
      above (the Lorentzian tail |H|² ∝ 1/x² then has on each interval the same relative error ~(ratio − 1)²/2).
    The trapezium integrates |H|² with a relative error ~2e-4 (verified for ζ 0.5–10 %). zeta = a scalar or a field by the modes
    (for the structural damping pass η/2). f_min > 0.
    """
    if not (f_min > 0 and f_max > f_min):
        raise ValueError("Frequency range must satisfy 0 < f_min < f_max")
    fn = np.atleast_1d(np.asarray(fn, dtype=np.float64)).ravel()
    ze = np.broadcast_to(np.asarray(zeta, dtype=np.float64), fn.shape) if fn.size else np.empty(0)
    nb = max(2, int(math.ceil(math.log10(f_max / f_min) * per_decade)) + 1)
    parts = [np.geomspace(f_min, f_max, nb)]
    per_peak = max(3, int(per_peak) | 1)                   # odd → a point exactly at the resonance
    xs = np.linspace(-1.0, 1.0, per_peak)
    ratio = max(float(ratio), 1.001)
    for f0, z in zip(fn, ze):
        if not f0 > 0:
            continue
        z = max(float(z), 1e-4)
        d0 = z * f0
        g = [f0 * (1.0 + z * xs)]
        up = f0 * (max(span, 1.0) - 1.0)
        if up > d0:
            g.append(f0 + d0 * ratio ** np.arange(1, int(math.log(up / d0) / math.log(ratio)) + 2))
        dn = 0.9 * f0
        if dn > d0:
            g.append(f0 - d0 * ratio ** np.arange(1, int(math.log(dn / d0) / math.log(ratio)) + 1))
        g = np.concatenate(g)
        g = g[(g >= f_min) & (g <= f_max)]
        if g.size:
            parts.append(g)
    f = np.unique(np.concatenate(parts))
    keep = np.concatenate([[True], np.diff(f) > 1e-12 * f[1:]])
    return f[keep]


def ssd_grid(f_min: float, f_max: float, fn=(), points: int = 20, bias: float = 3.0) -> np.ndarray:
    """Points of the calculation as in ccx *STEADY STATE DYNAMICS (manual node351 + the trial): fixed points = f_min, the natural frequencies
    inside the range (the same ones merged), f_max; in each interval `points` points including the ends
    x = ½ + ½·sgn(y)·|y|^(1/bias), y = −1 + 2k/(points − 1). points < 2 → 20, bias < 1 → 3 (as ccx).
    In total (m + 1)(n − 2) + m + 2 points.
    """
    if not f_max > f_min:
        raise ValueError("Frequency range must satisfy f_min < f_max")
    n = int(points) if points >= 2 else 20
    b = float(bias) if bias >= 1.0 else 3.0
    fn = np.atleast_1d(np.asarray(fn, dtype=np.float64)).ravel()
    inner = np.unique(fn[(fn > f_min) & (fn < f_max)])
    fixed = np.concatenate([[f_min], inner, [f_max]])
    keep = np.concatenate([[True], np.diff(fixed) > 1e-9 * fixed[1:]])
    fixed = fixed[keep]
    y = -1.0 + 2.0 * np.arange(n) / (n - 1)
    x = 0.5 + 0.5 * np.sign(y) * np.abs(y) ** (1.0 / b)
    out = [fixed[:1]]
    for a, c in zip(fixed[:-1], fixed[1:]):
        out.append(a + (c - a) * x[1:])
    return np.concatenate(out)


def trapz_weights(f) -> np.ndarray:
    """Weights of the trapezium rule on a non-uniform grid (∫g df ≈ Σ w·g)."""
    f = np.asarray(f, dtype=np.float64)
    w = np.zeros_like(f)
    if f.size < 2:
        return w
    d = np.diff(f)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w


# ------------------------------------------------------------------ modal transfers and moments

def modal_frf(f, fn, zeta=0.0, eta=None) -> np.ndarray:
    """Modal FRF H (n_f, K) complex = 1/(ω_j² − ω² + 2iζ_jω_jω), or with eta (K,)/scalar
    1/(ω_j²(1 + iη_j) − ω²) (structural damping; zeta is then ignored).
    """
    f = np.atleast_1d(np.asarray(f, dtype=np.float64))
    wn = 2.0 * np.pi * np.atleast_1d(np.asarray(fn, dtype=np.float64)).ravel()
    w = 2.0 * np.pi * f[:, None]
    if eta is not None:
        et = np.broadcast_to(np.asarray(eta, dtype=np.float64), wn.shape)
        return 1.0 / (wn * wn * (1.0 + 1j * et) - w * w)
    ze = np.broadcast_to(np.asarray(zeta, dtype=np.float64), wn.shape)
    return 1.0 / (wn * wn - w * w + 2j * ze * wn * w)


def modal_transfer(f, fn, zeta, gamma, eta=None, kind: str = "disp") -> np.ndarray:
    """Modal transfers Z (n_f, K) from the acceleration of the base to the quantity `kind` (see the docstring of the module)."""
    if kind not in KINDS:
        raise ValueError(f"Unknown response type '{kind}' (options: {', '.join(KINDS)})")
    f = np.atleast_1d(np.asarray(f, dtype=np.float64))
    H = modal_frf(f, fn, zeta, eta)
    g = np.atleast_1d(np.asarray(gamma, dtype=np.float64)).ravel()
    w = 2.0 * np.pi * f[:, None]
    if kind == "disp":
        return -g * H
    if kind == "vel":
        return -g * 1j * w * H
    if kind == "acc_rel":
        return g * w * w * H
    wn = 2.0 * np.pi * np.atleast_1d(np.asarray(fn, dtype=np.float64)).ravel()
    if eta is not None:
        et = np.broadcast_to(np.asarray(eta, dtype=np.float64), wn.shape)
        return g * wn * wn * (1.0 + 1j * et) * H
    ze = np.broadcast_to(np.asarray(zeta, dtype=np.float64), wn.shape)
    return g * (wn * wn + 2j * ze * wn * w) * H


def modal_moments(f, fn, zeta, gamma, W_mm, powers=(0, 1, 2, 4), eta=None, kind: str = "disp") -> np.ndarray:
    """Modal spectral moments M (P, K, K) complex: M^(p)_jk = ∫ f^p·Z_j·Z_k*·W df (trapezium on the grid f).

    f (n_f,) [Hz] increasing (preferably `freq_grid`); fn (K,) [Hz]; zeta (K,)/scalar; gamma (K,) = Γ_j;
    W_mm (n_f,) = ASD of the base in (mm/s²)²/Hz at the points f; kind see the module. M is the covariance of the modal coordinates
    (q_j, or q̇_j, q̈_j): the spectral moment of the response r = Σ A_j·q_j is Aᵀ·Re(M^(p))·A in the units [A·q]²·Hz^p
    (A = component of the shape φ_j → mm²·Hz^p, A = modal stress σ_j → MPa²·Hz^p).
    """
    f = np.atleast_1d(np.asarray(f, dtype=np.float64))
    W = np.broadcast_to(np.asarray(W_mm, dtype=np.float64), f.shape)
    if np.any(W < 0):
        raise ValueError("ASD must not be negative")
    Z = modal_transfer(f, fn, zeta, gamma, eta, kind)
    Zw = Z * np.sqrt(W * trapz_weights(f))[:, None]
    P = len(powers)
    K = Z.shape[1]
    M = np.empty((P, K, K), dtype=np.complex128)
    for i, p in enumerate(powers):
        fp = np.ones_like(f) if p == 0 else f ** float(p)
        M[i] = (Zw * fp[:, None]).T @ np.conj(Zw)
    return M


def response_psd(f, fn, zeta, gamma, W_mm, A, eta=None, kind: str = "disp") -> np.ndarray:
    """PSD of the response at the points f for real modal coefficients A (c, K) (e.g. the components of the shapes in a probe) →
    (n_f, c) = |Σ_j A_j·Z_j|²·W (one-sided, [quantity]²/Hz).
    """
    Z = modal_transfer(f, fn, zeta, gamma, eta, kind)
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    R = Z @ A.T
    W = np.broadcast_to(np.asarray(W_mm, dtype=np.float64), np.shape(f))
    return (R.real ** 2 + R.imag ** 2) * W[:, None]


# ------------------------------------------------------------------ quadratic forms by nodes (numba)

@njit(parallel=True, cache=True)
def _quad_forms(A, M, out):
    n, C, K = A.shape
    P = M.shape[0]
    for i in prange(n):
        a = np.empty(K)
        for c in range(C):
            for k in range(K):
                a[k] = A[i, c, k]
            for p in range(P):
                s = 0.0
                for j in range(K):
                    t = 0.0
                    for k in range(K):
                        t += M[p, j, k] * a[k]
                    s += a[j] * t
                out[i, c, p] = s


@njit(parallel=True, cache=True)
def _eq_moments(S, M, Q, out):
    n = S.shape[0]
    K = S.shape[2]
    P = M.shape[0]
    for i in prange(n):
        s = np.empty((6, K))
        y = np.empty((6, K))
        for a in range(6):
            for k in range(K):
                s[a, k] = S[i, a, k]
        for a in range(6):
            for k in range(K):
                t = 0.0
                for b in range(6):
                    t += Q[a, b] * s[b, k]
                y[a, k] = t
        for p in range(P):
            tot = 0.0
            for a in range(6):
                for j in range(K):
                    t = 0.0
                    for k in range(K):
                        t += M[p, j, k] * y[a, k]
                    tot += s[a, j] * t
            out[i, p] = tot


def _real_sym(M) -> np.ndarray:
    M = np.asarray(M)
    if M.ndim == 2:
        M = M[None]
    R = np.ascontiguousarray(M.real, dtype=np.float64)
    return 0.5 * (R + np.transpose(R, (0, 2, 1)))


def _modal_array(A, last: int) -> np.ndarray:
    A = np.asarray(A)
    if A.dtype not in (np.float32, np.float64):
        A = A.astype(np.float64)
    if A.shape[-1] != last:
        raise ValueError(f"Number of modes of the array ({A.shape[-1]}) does not match the moment matrix ({last})")
    return np.ascontiguousarray(A)


def quad_forms(A, M) -> np.ndarray:
    """Quadratic forms by nodes: A (n, c, K) float32/64, M (P, K, K) (complex or real) → (n, c, P)
    = Aᵀ·Re(M^(p))·A. Spectral moments of the components of the response (CQC in the full form).
    """
    Mr = _real_sym(M)
    A = _modal_array(A, Mr.shape[1])
    if A.ndim != 3:
        raise ValueError("A must have shape (n, c, K)")
    out = np.empty((A.shape[0], A.shape[1], Mr.shape[0]))
    _quad_forms(A, Mr, out)
    return out


def response_rms(A, M0) -> np.ndarray:
    """RMS of the components of the response (n, c) = √(Re Σ_jk A_j·A_k·M0_jk); A (n, c, K), M0 (K, K) = modal moment p = 0."""
    q = quad_forms(A, M0)[:, :, 0]
    return np.sqrt(np.maximum(q, 0.0))


def rms_components(Smodal, M0) -> np.ndarray:
    """RMS of the components of the stress (n, 6) [MPa] from the modal stresses Smodal (n, 6, K)."""
    return response_rms(Smodal, M0)


def equivalent_moments(Smodal, Mi) -> np.ndarray:
    """Spectral moments of the equivalent von Mises stress (Preumont & Piéfort 1994) by the nodes → (n, P):
    m_p = Σ_ab Q_ab·Σ_jk S_aj·Re(M^(p))_jk·S_bk. Smodal (n, 6, K) float32/64 (order SXX SYY SZZ SXY SYZ SZX),
    Mi (P, K, K) from `modal_moments` (kind="disp").
    """
    Mr = _real_sym(Mi)
    S = _modal_array(Smodal, Mr.shape[1])
    if S.ndim != 3 or S.shape[1] != 6:
        raise ValueError("Modal stresses must have shape (n, 6, K)")
    out = np.empty((S.shape[0], Mr.shape[0]))
    _eq_moments(S, Mr, Q_MISES, out)
    return np.maximum(out, 0.0)


def vm_rms_segalman(Smodal, M0) -> np.ndarray:
    """RMS of von Mises according to Segalman (√E[σ_vm²] = √m0 of the equivalent) [MPa] – informative only (it is not Gaussian)."""
    return np.sqrt(equivalent_moments(Smodal, M0)[:, 0])


def warmup() -> None:
    """Precompilation of numba (fea_core.warmup) – float32 and float64."""
    f = freq_grid(10.0, 100.0, [50.0], 0.05, per_peak=5, per_decade=4)
    M = modal_moments(f, [50.0, 60.0], 0.05, [1.0, 0.5], np.ones_like(f))
    for dt in (np.float32, np.float64):
        S = np.ones((3, 6, 2), dtype=dt)
        equivalent_moments(S, M)
        rms_components(S, M[0])
