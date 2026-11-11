"""Analytical (closed-form) calculations of beams and curved bars – quick calculator and check of the FE of bars.

Numpy only, no stiffness matrix. Units mm, N, MPa (N/mm²), N·mm.

SIGNS of straight beams (`beam`, `continuous_beam`):
  x = axis of the beam from left to right, the loads POSITIVE DOWNWARDS (in the direction of gravity, −z),
  w  = deflection, positive DOWNWARDS (in the direction of the positive load),
  φ  = rotation of the section, for Euler–Bernoulli φ = dw/dx (positive = right-handed, the beam goes down to the right),
  M  = bending moment, positive = tension at the bottom (simply supported beam under gravity M > 0, clamping M < 0),
  V  = shear force, V = dM/dx (for a simply supported beam on the left V = +R_A),
  concentrated moment C positive = right-handed (looking along x to the right, z upwards) = jump of M(x) by +C,
  reaction R positive UPWARDS; M_A, M_B, M_i at the supports = bending moment of the section at the support (sign as M(x)),
  i.e. clamping under q gives M_A = −qL²/12.
  Timoshenko (shear=True): EI·φ' = −M, w' = φ + V/(κGA), clamping φ = 0 (the section, not the tangent).

Method of the simply supported beam: Macaulay (singular) functions – an exact closed form (what Roark
table 8.1 / Young & Budynas tabulate), unknown initial values V0, M0, φ0, w0 (and the reactions of the internal supports) from the boundary
conditions → a small linear system. Continuous beam: three-moment equation (Clapeyron) with EI per span, settlements of supports,
clamped ends, overhanging ends; the spans are then solved as simply supported beams with the support moments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial, pi, sqrt
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------------------------------------
# Loads
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Force:
    """Concentrated force P [N] at x = a [mm], positive downwards."""
    P: float
    a: float


@dataclass
class Moment:
    """Concentrated moment C [N·mm] at x = a, positive right-handed (jump of M(x) by +C)."""
    C: float
    a: float


@dataclass
class Distributed:
    """Distributed load [N/mm] linearly q1 at x = a → q2 at x = b (b None = end of the beam, q2 None = q1)."""
    q1: float
    a: float = 0.0
    b: float | None = None
    q2: float | None = None


def uniform(q: float, a: float = 0.0, b: float | None = None) -> Distributed:
    """Uniform load q [N/mm] on the section a–b (default the whole length)."""
    return Distributed(q, a, b, q)


def triangular(q: float, a: float, b: float, peak: str = "b") -> Distributed:
    """Triangular load: 0 → q (peak="b", apex at b) or q → 0 (peak="a")."""
    return Distributed(0.0, a, b, q) if peak == "b" else Distributed(q, a, b, 0.0)


# ---------------------------------------------------------------------------------------------------------
# Stiffnesses (numbers or SectionProps)
# ---------------------------------------------------------------------------------------------------------
def _stiffness(E, I=None, section=None, axis="y", shear=False, G=None, nu=0.3, A=None, A_s=None,
               kappa=5 / 6):
    """→ (EI, 1/(κGA) or 0). Bending about the axis y (load in z) takes Iy and the shear area Av_z, about z Iz, Av_y."""
    if I is None:
        if section is None:
            raise ValueError("Specify the second moment of area I or a cross-section (section).")
        I = section.Iy if axis == "y" else section.Iz
    if E is None or E <= 0 or I <= 0:
        raise ValueError("E and I must be positive.")
    cs = 0.0
    if shear:
        G = E / (2 * (1 + nu)) if G is None else G
        if A_s is None:
            av = 0.0 if section is None else (section.Av_z if axis == "y" else section.Av_y)
            area = A if A is not None else (section.A if section is not None else None)
            A_s = av if av > 0 else (kappa * area if area else None)
        if not A_s or A_s <= 0:
            raise ValueError("For shear deformation, specify A_s (shear area) or A (with kappa) or section.")
        cs = 1.0 / (G * A_s)
    return E * I, cs


# ---------------------------------------------------------------------------------------------------------
# Core: one field with Macaulay functions
# ---------------------------------------------------------------------------------------------------------
_SUP = {"free": ("V", "M"), "pinned": ("w", "M"), "fixed": ("w", "phi"), "guided": ("phi", "V")}
CASES = {"simple": ("pinned", "pinned"), "cantilever": ("fixed", "free"),
         "fixed": ("fixed", "fixed"), "propped": ("fixed", "pinned")}
_QI = {"V": 0, "M": 1, "phi": 2, "w": 3}


def _fk(d, k, incl):
    """<d>^k / k!; for k = 0 the Heaviside function (incl = the value at d = 0 is 1 – right limit)."""
    if k < 0:
        return np.zeros_like(d)
    on = (d > 0) | (incl & (d == 0)) if k == 0 else d > 0
    return np.where(on, np.abs(d) ** k / factorial(k), 0.0)


def _clip(ld: Distributed, x0: float, x1: float, L_full: float):
    """Distributed load cut to the section [x0, x1] in local coordinates (None = outside the section)."""
    a, b = ld.a, (L_full if ld.b is None else ld.b)
    q1 = ld.q1
    q2 = q1 if ld.q2 is None else ld.q2
    if b <= a:
        raise ValueError(f"Distributed load: end b = {b} must be beyond start a = {a}.")
    lo, hi = max(a, x0), min(b, x1)
    if hi <= lo:
        return None
    q = lambda x: q1 + (q2 - q1) * (x - a) / (b - a)  # noqa: E731
    return Distributed(q(lo), lo - x0, hi - x0, q(hi))


def _terms(loads, L):
    """Loads in local coordinates → terms (a, m, k): contribution to M(x) = m·<x − a>^k / k!."""
    t = []
    for ld in loads:
        if isinstance(ld, Force):
            t.append((ld.a, -ld.P, 1))
        elif isinstance(ld, Moment):
            t.append((ld.a, ld.C, 0))
        elif isinstance(ld, Distributed):
            c = _clip(ld, 0.0, L, L)
            if c is None:
                continue
            s = (c.q2 - c.q1) / (c.b - c.a)
            t += [(c.a, -c.q1, 2), (c.a, -s, 3), (c.b, c.q2, 2), (c.b, s, 3)]
        else:
            raise TypeError(f"Unknown load {ld!r}")
    return t


def _contrib(x, incl, terms, EI, cs):
    """(V, M, φ, w) from the Macaulay terms. The moment (k = 0) gives neither V nor the shear deflection."""
    V, M, phi, w = (np.zeros_like(x) for _ in range(4))
    for a, m, k in terms:
        d = x - a
        if k > 0:
            V += m * _fk(d, k - 1, incl)
            w += cs * m * _fk(d, k, incl)
        M += m * _fk(d, k, incl)
        phi -= m * _fk(d, k + 1, incl) / EI
        w -= m * _fk(d, k + 2, incl) / EI
    return np.array([V, M, phi, w])


class _Span:
    """One span of length L, constant EI: unknowns u = [V0, M0, φ0, w0, R_1 … R_m] (state at x = 0⁻)."""

    def __init__(self, L, loads, EI, cs, bcs, inner=(), inner_w=()):
        self.L, self.EI, self.cs = float(L), EI, cs
        self.terms = _terms(loads, L)
        self.inner = [float(s) for s in inner]
        n = 4 + len(self.inner)
        rows, rhs = [], []
        for end, q, val in bcs:          # boundary conditions (end 0|1, quantity, value)
            x = np.array([0.0 if end == 0 else self.L])
            inc = np.array([end == 1])   # on the left the state 0⁻ (without the load at x = 0), on the right L⁺ (incl. the load at L)
            rows.append(self._basis(x, inc)[_QI[q], 0])
            rhs.append(val - self._part(x, inc)[_QI[q], 0])
        for s, ws in zip(self.inner, inner_w):
            x, inc = np.array([s]), np.array([True])
            rows.append(self._basis(x, inc)[3, 0])
            rhs.append(ws - self._part(x, inc)[3, 0])
        A = np.array(rows).reshape(len(rows), n)
        sc = np.linalg.norm(A, axis=0)                # columns have different units (1/EI…) → scale
        sc[sc == 0] = 1.0
        As = A / sc
        if len(rows) != n or np.linalg.matrix_rank(As, tol=1e-10 * np.abs(As).max()) < n:
            raise ValueError("The beam is a mechanism or has unsuitable supports (singular boundary conditions).")
        self.u = np.linalg.solve(As, np.array(rhs)) / sc

    def _part(self, x, incl):
        return _contrib(x, incl, self.terms, self.EI, self.cs)

    def _basis(self, x, incl):
        EI, cs, z = self.EI, self.cs, np.zeros_like(x)
        cols = [np.array([np.ones_like(x), x, -x ** 2 / (2 * EI), -x ** 3 / (6 * EI) + cs * x]),   # V0
                np.array([z, np.ones_like(x), -x / EI, -x ** 2 / (2 * EI)]),                      # M0
                np.array([z, z, np.ones_like(x), x]),                                              # φ0
                np.array([z, z, z, np.ones_like(x)])]                                              # w0
        cols += [_contrib(x, incl, [(s, 1.0, 1)], EI, cs) for s in self.inner]   # upward reaction
        return np.stack(cols, axis=-1)            # (4, nx, nu)

    def eval(self, x, incl=None):
        x = np.atleast_1d(np.asarray(x, float))
        incl = np.ones(x.shape, bool) if incl is None else np.asarray(incl, bool)
        return self._part(x, incl) + self._basis(x, incl) @ self.u

    def jumps(self):
        """Positions of jumps of V or M inside the span (concentrated loads, internal supports)."""
        p = [a for a, m, k in self.terms if k <= 1] + self.inner
        return sorted({a for a in p if 0.0 < a < self.L})


# ---------------------------------------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------------------------------------
@dataclass
class BeamResult:
    x: np.ndarray            # points [mm] (at jumps twice: the left and the right limit)
    V: np.ndarray            # shear force [N]
    M: np.ndarray            # bending moment [N·mm]
    phi: np.ndarray          # rotation of the section [rad]
    w: np.ndarray            # deflection [mm], positive downwards
    reactions: dict          # R_A, M_A, R_B, M_B, R_1… [N, N·mm]
    extremes: dict           # {"w": {"max": (value, x), "min": …, "abs": …}, "M": …, "V": …, "phi": …}
    info: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    at: Callable = field(default=None, repr=False)   # at(x) → {"V", "M", "phi", "w"} (right limit)

    def summary(self) -> str:
        r = ", ".join(f"{k} = {v:.6g}" for k, v in self.reactions.items())
        e = self.extremes
        return (f"Reactions: {r}\nM max {e['M']['max'][0]:.6g} at x = {e['M']['max'][1]:.6g}, min {e['M']['min'][0]:.6g} at x = {e['M']['min'][1]:.6g}\nw abs. max {e['w']['abs'][0]:.6g} at x = {e['w']['abs'][1]:.6g}")


def _grid(L, n, jumps, extra=()):
    """Points 0…L + loads; at jumps a pair (left limit incl=False, right limit incl=True)."""
    xs = np.unique(np.concatenate([np.linspace(0.0, L, max(int(n), 2)),
                                   [p for p in extra if 0 <= p <= L], jumps]))
    inc = np.ones(xs.shape, bool)
    jx = np.array(jumps, float)
    xs = np.concatenate([xs, jx])
    inc = np.concatenate([inc, np.zeros(jx.shape, bool)])
    o = np.lexsort((inc, xs))
    return xs[o], inc[o]


def _extremes(x, vals, at):
    """Extremes on the grid refined by sampling the neighbourhood (4 rounds of 33 points → position to ~1e-5 of the grid step);
    the value is always a real value of the function, never worse than on the grid.
    """
    out = {}
    for qi, name in enumerate(("V", "M", "phi", "w")):
        y = vals[qi]
        d = {}
        for key, sgn in (("max", 1.0), ("min", -1.0)):
            i = int(np.argmax(sgn * y))
            best = (float(y[i]), float(x[i]))
            lo, hi = x[max(i - 1, 0)], x[min(i + 1, len(x) - 1)]
            if name != "V" and hi > lo:
                for _ in range(4):
                    t = np.linspace(lo, hi, 33)
                    f = at(t)[name]
                    k = int(np.argmax(sgn * f))
                    if sgn * f[k] > sgn * best[0]:
                        best = (float(f[k]), float(t[k]))
                    lo, hi = t[max(k - 1, 0)], t[min(k + 1, 32)]
            d[key] = best
        d["abs"] = d["max"] if abs(d["max"][0]) >= abs(d["min"][0]) else d["min"]
        out[name] = d
    return out


def _at_fn(segments):
    """Evaluation by sections: segments = [(x0, x1, fn(x_local) → (4, n))]."""
    def at(x):
        x = np.atleast_1d(np.asarray(x, float))
        out = np.zeros((4, x.size))
        for j, (x0, x1, fn) in enumerate(segments):
            last = j == len(segments) - 1
            m = (x >= x0) & ((x < x1) | (last & (x <= x1 + 1e-12 * max(abs(x1), 1.0))))
            if j == 0:
                m |= x < x0
            if m.any():
                out[:, m] = fn(x[m] - x0)
        r = {k: out[i] for i, k in enumerate(("V", "M", "phi", "w"))}
        return {k: float(v[0]) for k, v in r.items()} if out.shape[1] == 1 else r
    return at


# ---------------------------------------------------------------------------------------------------------
# 1. Simply supported beam (and in general a beam with internal hinged supports, constant EI)
# ---------------------------------------------------------------------------------------------------------
def beam(L: float, loads: Sequence = (), supports=("pinned", "pinned"), *, E: float, I: float | None = None,
         section=None, axis: str = "y", inner: Sequence[float] = (), settlement: Sequence[float] | None = None,
         shear: bool = False, G: float | None = None, nu: float = 0.3, A: float | None = None,
         A_s: float | None = None, kappa: float = 5 / 6, n: int = 201) -> BeamResult:
    """Beam of length L with constant EI, superposition of any number of loads (Force, Moment, Distributed).

    supports: a pair of end types (left, right) from "free" | "pinned" | "fixed" | "guided" (sliding clamping:
    φ = 0, V = 0) or the name of a case from CASES: "simple" (simply supported), "cantilever" (cantilever clamped on the left),
    "fixed" (clamped at both ends), "propped" (clamped on the left, hinged on the right).
    inner: positions of internal hinged supports [mm] (continuous beam with constant EI, overhanging ends = "free").
    settlement: prescribed deflections of the supports (positive downwards) in the order left end (if supported), internal, right.
    shear: shear deformation (Timoshenko) – G (default E/2(1+ν)), A_s (default section.Av_z / Av_y or κ·A).
    """
    if isinstance(supports, str):
        supports = CASES[supports]
    left, right = supports
    if left not in _SUP or right not in _SUP:
        raise ValueError(f"Unknown support type: {supports} (possible: {', '.join(_SUP)})")
    EI, cs = _stiffness(E, I, section, axis, shear, G, nu, A, A_s, kappa)
    inner = sorted(float(s) for s in inner)
    if any(not 0 < s < L for s in inner):
        raise ValueError("Interior supports must lie inside the beam (0 < x < L).")
    nsup = (left in ("pinned", "fixed")) + len(inner) + (right in ("pinned", "fixed"))
    dl = list(settlement) if settlement is not None else [0.0] * nsup
    if len(dl) != nsup:
        raise ValueError(f"settlement: expecting {nsup} values (supported ends and interior supports).")
    it = iter(dl)                       # order: left end, internal supports, right end
    d_left = next(it) if left in ("pinned", "fixed") else 0.0
    d_inner = [next(it) for _ in inner]
    d_right = next(it) if right in ("pinned", "fixed") else 0.0
    bcs = [(end, q, (d if q == "w" else 0.0))
           for end, typ, d in ((0, left, d_left), (1, right, d_right)) for q in _SUP[typ]]
    span = _Span(L, loads, EI, cs, bcs, inner, d_inner)
    x, inc = _grid(L, n, span.jumps(), [p for ld in loads for p in (getattr(ld, "a", 0), getattr(ld, "b", 0) or 0)])
    vals = span.eval(x, inc)
    at = _at_fn([(0.0, L, span.eval)])
    u = span.u
    endR = span.eval([L])[:, 0]
    reac = {}
    if left in ("pinned", "fixed"):
        reac["R_A"] = float(u[0])
    if left in ("fixed", "guided"):
        reac["M_A"] = float(u[1])
    for i, s in enumerate(inner):
        reac[f"R_{i + 1}"] = float(u[4 + i])
    if right in ("pinned", "fixed"):
        reac["R_B"] = float(-endR[0])
    if right in ("fixed", "guided"):
        reac["M_B"] = float(endR[1])
    warn = []
    if shear and cs * EI / L ** 2 > 0.1:
        warn.append("Shear deformation is significant (low slenderness) – beam theory is only approximate.")
    return BeamResult(x, *vals, reac, _extremes(x, vals, at),
                      {"EI": EI, "GA_s": (1 / cs if cs else None), "supports": (left, right),
                       "inner": inner}, warn, at)


# ---------------------------------------------------------------------------------------------------------
# 2. Continuous beam – three-moment equation (Clapeyron)
# ---------------------------------------------------------------------------------------------------------
def _per(v, n, name):
    if v is None or np.isscalar(v) or not isinstance(v, (list, tuple, np.ndarray)):
        return [v] * n
    if len(v) != n:
        raise ValueError(f"{name}: expecting {n} values (one per span).")
    return list(v)


def continuous_beam(spans: Sequence[float], loads: Sequence = (), *, E, I=None, section=None, axis: str = "y",
                    left: str = "pinned", right: str = "pinned", overhang=(0.0, 0.0),
                    settlement: Sequence[float] | None = None, n: int = 101) -> BeamResult:
    """Continuous beam of n spans – three-moment equation, Euler–Bernoulli.

    spans: lengths of the spans [mm]; E, I or section: one value or a list per span (overhanging ends have the EI
    of the adjacent span). left/right: "pinned" | "fixed" (end support). overhang: (overhang on the left, right) [mm]
    – only with a hinged end support. settlement: settlements of the supports [mm] (positive downwards), n + 1 values.
    loads: loads in the GLOBAL coordinate x (x = 0 at the left end of the beam including the overhang). A concentrated force directly
    over a support goes into the reaction, a concentrated moment over a support is a jump of the support moment.
    Reactions R_1 … R_{n+1} (from the left), M_i = bending moment over the support i (at support 1 to the right of it, otherwise to the left).
    """
    Ls = [float(s) for s in spans]
    ns = len(Ls)
    if ns < 1 or min(Ls) <= 0:
        raise ValueError("Specify at least one span with positive length.")
    cl, cr = (float(overhang[0]), float(overhang[1]))
    if (cl > 0 and left == "fixed") or (cr > 0 and right == "fixed"):
        raise ValueError("An overhanging end cannot be combined with a fixed end support.")
    for s in (left, right):
        if s not in ("pinned", "fixed"):
            raise ValueError("End support of a continuous beam: 'pinned' or 'fixed' (free end = overhang).")
    Es, Is, secs = _per(E, ns, "E"), _per(I, ns, "I"), _per(section, ns, "section")
    EIs = [_stiffness(Es[j], Is[j], secs[j], axis)[0] for j in range(ns)]
    xs = cl + np.concatenate([[0.0], np.cumsum(Ls)])            # support positions
    Ltot = xs[-1] + cr
    dl = np.zeros(ns + 1) if settlement is None else np.asarray(settlement, float)
    if dl.shape != (ns + 1,):
        raise ValueError(f"settlement: expecting {ns + 1} values.")
    tol = 1e-9 * Ltot
    P_sup, C_sup = np.zeros(ns + 1), np.zeros(ns + 1)
    seg_loads = {k: [] for k in range(-1, ns + 1)}               # −1 = overhang on the left, ns = overhang on the right
    bounds = [(0.0, cl)] + [(xs[j], xs[j + 1]) for j in range(ns)] + [(xs[-1], Ltot)]
    for ld in loads:
        if isinstance(ld, (Force, Moment)):
            near = np.nonzero(np.abs(xs - ld.a) < tol)[0]
            if near.size:
                if isinstance(ld, Force):
                    P_sup[near[0]] += ld.P
                else:
                    C_sup[near[0]] += ld.C
                continue
            if not -tol <= ld.a <= Ltot + tol:
                raise ValueError(f"Load outside the beam (x = {ld.a}).")
            for k, (x0, x1) in zip(range(-1, ns + 1), bounds):
                if x0 - tol <= ld.a <= x1 + tol and x1 > x0:
                    seg_loads[k].append(type(ld)(*(([ld.P] if isinstance(ld, Force) else [ld.C]) + [ld.a - x0])))
                    break
        else:
            for k, (x0, x1) in zip(range(-1, ns + 1), bounds):
                if x1 > x0:
                    c = _clip(ld, x0, x1, Ltot)
                    if c is not None:
                        seg_loads[k].append(c)
    # overhangs as cantilevers clamped at the end support → moment over the support (statically determinate)
    oh_l = _Span(cl, seg_loads[-1], EIs[0], 0.0, [(0, "V", 0), (0, "M", 0), (1, "w", 0), (1, "phi", 0)]) \
        if cl > 0 else None
    oh_r = _Span(cr, seg_loads[ns], EIs[-1], 0.0, [(0, "w", 0), (0, "phi", 0), (1, "V", 0), (1, "M", 0)]) \
        if cr > 0 else None
    ML0 = float(oh_l.eval([cl])[1, 0]) if oh_l else 0.0          # moment to the left of support 0
    MRn = float(oh_r.eval([0.0], [False])[1, 0]) if oh_r else 0.0  # moment to the right of support n

    def span_bc(j, Ma, Mb):
        return [(0, "w", dl[j]), (0, "M", Ma), (1, "w", dl[j + 1]), (1, "M", Mb)]

    # rotations of the ends of the spans from the load (simply supported span, M = 0, no settlement)
    th0 = []
    for j in range(ns):
        s0 = _Span(Ls[j], seg_loads[j], EIs[j], 0.0, [(0, "w", 0), (0, "M", 0), (1, "w", 0), (1, "M", 0)])
        th0.append((float(s0.eval([0.0], [False])[2, 0]), float(s0.eval([Ls[j]])[2, 0])))
    # unknowns X_i = moment to the left of support i (ML_i); MR_i = X_i + C_i
    nU = ns + 1
    A = np.zeros((nU, nU))
    b = np.zeros(nU)
    f = [Ls[j] / EIs[j] for j in range(ns)]
    rot = [(dl[j + 1] - dl[j]) / Ls[j] for j in range(ns)]
    # θ_a(j) = th0a + MR_j f/3 + ML_{j+1} f/6 + rot ;  θ_b(j) = th0b − MR_j f/6 − ML_{j+1} f/3 + rot
    for i in range(1, ns):              # continuity of rotation over an internal support: θ_b(i−1) = θ_a(i)
        jl, jr = i - 1, i
        A[i, jl] += f[jl] / 6           # MR_{i−1}
        A[i, i] += f[jl] / 3 + f[jr] / 3
        A[i, i + 1] += f[jr] / 6        # ML_{i+1}
        b[i] = (th0[jl][1] - th0[jr][0] + rot[jl] - rot[jr]
                - C_sup[jl] * f[jl] / 6 - C_sup[i] * f[jr] / 3)
    if left == "fixed":                 # θ_a(0) = 0
        A[0, 0], A[0, 1] = f[0] / 3, f[0] / 6
        b[0] = -th0[0][0] - rot[0] - C_sup[0] * f[0] / 3
    else:
        A[0, 0], b[0] = 1.0, ML0
    if right == "fixed":                # θ_b(n−1) = 0
        A[ns, ns - 1], A[ns, ns] = -f[-1] / 6, -f[-1] / 3
        b[ns] = -th0[-1][1] - rot[-1] + C_sup[ns - 1] * f[-1] / 6
    else:
        A[ns, ns], b[ns] = 1.0, MRn - C_sup[ns]
    X = np.linalg.solve(A, b)
    ML, MR = X, X + C_sup
    spans_s = [_Span(Ls[j], seg_loads[j], EIs[j], 0.0, span_bc(j, MR[j], ML[j + 1])) for j in range(ns)]
    th_sup0 = float(spans_s[0].eval([0.0], [False])[2, 0])
    th_supn = float(spans_s[-1].eval([Ls[-1]])[2, 0])

    def shifted(sp, d0, th, x_ref):
        def fn(x, incl=None):
            v = sp.eval(x, incl)
            v[2] += th
            v[3] += d0 + th * (np.atleast_1d(x) - x_ref)
            return v
        return fn

    segs = []                           # (x0, x1, fn, jumps)
    if oh_l:
        segs.append((0.0, cl, shifted(oh_l, dl[0], th_sup0, cl), oh_l.jumps()))
    for j, sp in enumerate(spans_s):
        segs.append((xs[j], xs[j + 1], sp.eval, sp.jumps()))
    if oh_r:
        segs.append((xs[-1], Ltot, shifted(oh_r, dl[-1], th_supn, 0.0), oh_r.jumps()))
    xx, vv = [], []
    for x0, x1, fn, jm in segs:
        xl, inc = _grid(x1 - x0, n, jm)
        xx.append(xl + x0)
        vv.append(fn(xl, inc))
    x = np.concatenate(xx)
    vals = np.concatenate(vv, axis=1)
    at = _at_fn([(x0, x1, fn) for x0, x1, fn, _ in segs])
    # reaction: R_i = V on the right − V on the left + force over the support
    reac = {}
    for i in range(ns + 1):
        vr = (float(spans_s[i].eval([0.0], [False])[0, 0]) if i < ns
              else (float(oh_r.eval([0.0], [False])[0, 0]) if oh_r else 0.0))
        vl = (float(spans_s[i - 1].eval([Ls[i - 1]])[0, 0]) if i > 0
              else (float(oh_l.eval([cl])[0, 0]) if oh_l else 0.0))
        reac[f"R_{i + 1}"] = vr - vl + P_sup[i]
    for i in range(ns + 1):
        reac[f"M_{i + 1}"] = float(MR[0] if i == 0 else ML[i])
    info = {"supports_x": xs.tolist(), "M_left": ML.tolist(), "M_right": MR.tolist(), "EI": EIs,
            "length": Ltot}
    return BeamResult(x, *vals, reac, _extremes(x, vals, at), info, [], at)


# ---------------------------------------------------------------------------------------------------------
# 3. Curved bar – Winkler–Bach
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Profile:
    """Section of a curved bar in the plane of curvature: y = radial distance from the inner fibre (outwards), width b(y).

    strips = trapezoidal strips (y1, y2, b1, b2, sign), circles = discs (y of centre, radius, sign);
    sign −1 = hole (tube).
    """
    strips: list = field(default_factory=list)
    circles: list = field(default_factory=list)
    name: str = ""

    def extent(self):
        lo = [s[0] for s in self.strips] + [c[0] - c[1] for c in self.circles]
        hi = [s[1] for s in self.strips] + [c[0] + c[1] for c in self.circles]
        return min(lo), max(hi)


def rect(b, h):
    return Profile([(0.0, h, b, b, 1)], name=f"rectangle {b}×{h}")


def trapezoid(b_in, b_out, h):
    """Trapezium: width b_in at the inner fibre, b_out at the outer one."""
    return Profile([(0.0, h, b_in, b_out, 1)], name="trapezoid")


def circle(d):
    return Profile(circles=[(d / 2, d / 2, 1)], name=f"circle Ø{d}")


def tube(D, t):
    return Profile(circles=[(D / 2, D / 2, 1), (D / 2, D / 2 - t, -1)], name=f"tube Ø{D}×{t}")


def tee(b_f, t_f, t_w, h, flange="inner"):
    """T section of total height h: flange b_f × t_f on the inner (flange="inner") or outer side, web t_w."""
    if flange == "inner":
        return Profile([(0.0, t_f, b_f, b_f, 1), (t_f, h, t_w, t_w, 1)], name="T")
    return Profile([(0.0, h - t_f, t_w, t_w, 1), (h - t_f, h, b_f, b_f, 1)], name="T")


def ishape(h, b_in, t_in, b_out, t_out, t_w):
    """I section: inner flange b_in × t_in, outer b_out × t_out, web t_w, total height h."""
    return Profile([(0.0, t_in, b_in, b_in, 1), (t_in, h - t_out, t_w, t_w, 1),
                    (h - t_out, h, b_out, b_out, 1)], name="I")


def strip_table(rows):
    """Table of strips [(y1, y2, b1, b2)] or width samples [(y, b)] (linear between the samples)."""
    rows = [tuple(map(float, r)) for r in rows]
    if rows and len(rows[0]) == 2:
        rows = [(y1, y2, b1, b2) for (y1, b1), (y2, b2) in zip(rows, rows[1:])]
    return Profile([(y1, y2, b1, b2, 1) for y1, y2, b1, b2 in rows if y2 > y1], name="width table")


def polygon(pts):
    """General (simple) polygon [(x, y)], y radially outwards; the width along the height is linear between the vertices."""
    P = np.asarray(pts, float)
    P = P - [0.0, P[:, 1].min()]
    Q = np.roll(P, -1, axis=0)

    def width(y):
        x1, y1, x2, y2 = P[:, 0], P[:, 1], Q[:, 0], Q[:, 1]
        m = (np.minimum(y1, y2) < y) & (np.maximum(y1, y2) > y)
        xc = np.sort(x1[m] + (y - y1[m]) * (x2[m] - x1[m]) / (y2[m] - y1[m]))
        return float(np.sum(xc[1::2] - xc[0::2]))

    ys = np.unique(P[:, 1])
    strips = []
    for y1, y2 in zip(ys, ys[1:]):
        w1, w3 = width(y1 + 0.25 * (y2 - y1)), width(y1 + 0.75 * (y2 - y1))
        strips.append((y1, y2, w1 - (w3 - w1) / 2, w3 + (w3 - w1) / 2, 1))
    return Profile(strips, name="polygon")


_GX, _GW = np.polynomial.legendre.leggauss(16)


def _strip_quad(y1, y2, shift, fn):
    """∫ fn(y, r) dy over [y1, y2], r = shift + y, in parts of width ≤ r/2 (16-point Gauss – 1/r is then exact
    to rounding). The integrand receives the local y → (y − y_T) without subtracting large radii.
    """
    tot, a = 0.0, y1
    while a < y2:
        b = min(y2, a + 0.5 * (a + shift))
        y = 0.5 * (b - a) * _GX + 0.5 * (a + b)
        tot += 0.5 * (b - a) * float(np.dot(_GW, fn(y, y + shift)))
        a = b
    return tot


@dataclass
class CurvedBeamResult:
    A: float                 # area [mm²]
    R: float                 # radius of the centroidal axis [mm]
    r_n: float               # radius of the neutral axis [mm] = A / ∫dA/r
    e: float                 # eccentricity R − r_n [mm] (> 0, the neutral axis is closer to the centre of curvature)
    r_i: float               # inner fibre [mm]
    r_o: float               # outer fibre [mm]
    I: float                 # second moment of area about the centroidal axis ⟂ to the plane of curvature [mm⁴]
    sigma_i: float           # stress of the inner fibre [MPa]
    sigma_o: float           # stress of the outer fibre [MPa]
    sigma_i_straight: float  # the same by the straight bar N/A + M·(r − R)/I
    sigma_o_straight: float
    K_i: float               # bending factor of the inner fibre (curved / straight), Roark table 9.1
    K_o: float
    N: float = 0.0
    M: float = 0.0
    warnings: list = field(default_factory=list)

    def stress(self, r):
        """σ(r) = N/A + M·(r − r_n) / (A·e·r)."""
        r = np.asarray(r, float)
        return self.N / self.A + self.M * (r - self.r_n) / (self.A * self.e * r)


def curved_beam(profile: Profile, M: float = 0.0, N: float = 0.0, *, r_inner: float | None = None,
                R: float | None = None) -> CurvedBeamResult:
    """Curved bar (Winkler–Bach): bending stress by the moment M and the axial force N (at the centroid).

    Position: r_inner = radius of the inner fibre, or R = radius of the centroidal axis.
    SIGNS: N > 0 tension; M > 0 increases curvature (the bar "closes") → outer fibres in tension, inner in compression.
    Stress σ(r) = N/A + M·(r − r_n) / (A·e·r), r_n = A / ∫dA/r, e = R − r_n.
    e is computed stably: Q = ∫(r − R)²/r dA, ∫dA/r = A/R + Q/R², e = R·Q / (A·R + Q) – without subtracting
    close numbers even for R/h → ∞ (e → I/(A·R)).
    """
    y0, y1 = profile.extent()
    h = y1 - y0
    if (r_inner is None) == (R is None):
        raise ValueError("Specify exactly one of r_inner (inner radius) or R (centroid radius).")
    # centroid at coordinate y
    A = Sy = 0.0
    for a, b_, b1, b2, sg in profile.strips:
        A += sg * 0.5 * (b1 + b2) * (b_ - a)
        Sy += sg * (b_ - a) * (b1 * (2 * a + b_) + b2 * (a + 2 * b_)) / 6
    for yc, c, sg in profile.circles:
        A += sg * pi * c * c
        Sy += sg * pi * c * c * yc
    if A <= 0:
        raise ValueError("The cross-section has non-positive area.")
    yg = Sy / A
    shift = (r_inner - y0) if r_inner is not None else (R - yg)
    Rg = yg + shift
    ri, ro = y0 + shift, y1 + shift
    if ri <= 0:
        raise ValueError(f"Inner fiber r_i = {ri:.6g} mm must have a positive radius.")
    Q = Ig = 0.0
    for a, b_, b1, b2, sg in profile.strips:
        bw = lambda y: b1 + (b2 - b1) * (y - a) / (b_ - a)  # noqa: E731
        Q += sg * _strip_quad(a, b_, shift, lambda y, r: bw(y) * (y - yg) ** 2 / r)
        Ig += sg * _strip_quad(a, b_, shift, lambda y, r: bw(y) * (y - yg) ** 2)
    for yc, c, sg in profile.circles:
        R0 = yc + shift
        s = sqrt((R0 - c) * (R0 + c))
        Sc = 2 * pi * c * c / (R0 + s)                  # ∫dA/r of a circle = 2π(R0 − √(R0² − c²))
        Q0 = R0 * pi * c ** 4 / (R0 + s) ** 2           # ∫(r − R0)²/r dA
        d = yc - yg
        Q += sg * (Q0 - 2 * d * Q0 / R0 + d * d * Sc)
        Ig += sg * (pi * c ** 4 / 4 + pi * c * c * d * d)
    e = Rg * Q / (A * Rg + Q)
    rn = Rg - e
    res = CurvedBeamResult(A, Rg, rn, e, ri, ro, Ig, 0, 0, 0, 0, 0, 0, N, M)
    ci, co = y0 - yg, y1 - yg                           # fibres from the centroid (ci < 0)
    bi = (ci + e) / (A * e * ri)                        # σ_i per unit moment (r − r_n = c + e)
    bo = (co + e) / (A * e * ro)
    res.sigma_i, res.sigma_o = N / A + M * bi, N / A + M * bo
    res.sigma_i_straight = N / A + M * ci / Ig
    res.sigma_o_straight = N / A + M * co / Ig
    res.K_i, res.K_o = bi / (ci / Ig), bo / (co / Ig)
    if Rg / h > 10:
        res.warnings.append(f"R/h = {Rg / h:.3g} > 10: the curvature effect is small (K_i = {res.K_i:.4f}), the straight-member formula is sufficient; e is computed stably.")
    return res


# ---------------------------------------------------------------------------------------------------------
# 4. Circular arcs and rings (Castigliano, numerically exact quadrature along the arc)
# ---------------------------------------------------------------------------------------------------------
@dataclass
class ArcResult:
    u_r: float               # displacement of the end radially (outwards from the centre) [mm]
    u_t: float               # displacement of the end tangentially (in the direction of increasing angle) [mm]
    rot: float               # rotation of the end [rad], positive anticlockwise
    ux: float                # global displacements (centre at the origin, clamping at (R, 0))
    uy: float
    M_fixed: float           # moment at the clamping [N·mm]
    M_absmax: float          # largest |M| along the arc
    compliance: np.ndarray   # 3×3 compliance of the end in (F_r, F_t, M)
    phi: np.ndarray = field(repr=False, default=None)   # angles of the sections [rad]
    M: np.ndarray = field(repr=False, default=None)     # M(φ)
    N: np.ndarray = field(repr=False, default=None)     # normal force (tension +)
    V: np.ndarray = field(repr=False, default=None)     # shear force (radial component)


def _arc_flex(EI, EA, kGA, w, m, nn, v):
    """Compliance C_ij = ∫ (m_i m_j/EI + n_i n_j/EA + v_i v_j/κGA) ds; m, nn, v = (3, nbodies), w = weights·ds."""
    C = (m * w) @ m.T / EI
    if EA:
        C += (nn * w) @ nn.T / EA
    if kGA:
        C += (v * w) @ v.T / kGA
    return C


def _arc_unit(R, th, phi):
    """Internal forces in the sections φ from the unit loads of the end θ: F_r, F_t (local axes of the end), M."""
    er = np.array([np.cos(th), np.sin(th)])
    et = np.array([-np.sin(th), np.cos(th)])
    rx = R * (np.cos(th) - np.cos(phi))           # arm = end − section
    ry = R * (np.sin(th) - np.sin(phi))
    t = np.array([-np.sin(phi), np.cos(phi)])     # tangent of the section (to the end)
    n_ = np.array([np.cos(phi), np.sin(phi)])
    m, nn, v = [], [], []
    for F in (er, et):
        m.append(rx * F[1] - ry * F[0])
        nn.append(F @ t)
        v.append(F @ n_)
    m.append(np.ones_like(phi))
    nn.append(np.zeros_like(phi))
    v.append(np.zeros_like(phi))
    return np.array(m), np.array(nn), np.array(v)


def _arc_props(E, I, section, axis, axial, shear, G, nu, A, A_s, kappa):
    EI, cs = _stiffness(E, I, section, axis, shear, G, nu, A, A_s, kappa)
    EA = None
    if axial:
        area = A if A is not None else (section.A if section is not None else None)
        if not area:
            raise ValueError("Give A or section for the axial compliance.")
        EA = E * area
    return EI, EA, (1 / cs if cs else None)


def arc_cantilever(R: float, theta: float, *, Fr: float = 0.0, Ft: float = 0.0, M: float = 0.0, E: float,
                   I: float | None = None, section=None, axis: str = "y", axial: bool = False,
                   shear: bool = False, G=None, nu: float = 0.3, A=None, A_s=None, kappa: float = 5 / 6,
                   n: int = 64) -> ArcResult:
    """Cantilever circular arc of radius R and central angle θ [rad]: clamped at the angle 0 (point (R, 0)), the arc
    anticlockwise; at the free end (angle θ) force F_r (radially outwards), F_t (tangentially in the direction of
    increasing angle) and moment M (anticlockwise). Castigliano with bending, optionally tension (axial) and shear (shear);
    Gauss quadrature with n points – the integrand is a trigonometric polynomial of degree 2 → exact.
    M(φ) positive anticlockwise (increases the curvature of the arc).
    """
    EI, EA, kGA = _arc_props(E, I, section, axis, axial, shear, G, nu, A, A_s, kappa)
    gx, gw = np.polynomial.legendre.leggauss(n)
    phi = 0.5 * theta * (gx + 1)
    m, nn, v = _arc_unit(R, theta, phi)
    C = _arc_flex(EI, EA, kGA, gw * 0.5 * theta * R, m, nn, v)
    P = np.array([Fr, Ft, M])
    ur, ut, rot = C @ P
    c, s = np.cos(theta), np.sin(theta)
    ph = np.linspace(0, theta, 181)
    mm, n2, v2 = _arc_unit(R, theta, ph)
    Mp = P @ mm
    return ArcResult(ur, ut, rot, ur * c - ut * s, ur * s + ut * c, float(Mp[0]), float(np.abs(Mp).max()), C,
                     ph, Mp, P @ n2, P @ v2)


@dataclass
class RingResult:
    dD_load: float           # change of diameter in the direction of the forces [mm] (negative = shortening)
    dD_perp: float           # change of diameter perpendicular [mm]
    M_load: float            # moment at the point of the force [N·mm] (M > 0 increases curvature, outer fibres in tension)
    M_side: float            # moment v 90° od sil
    N_side: float            # normal force at 90° [N] (compression −F/2)


def ring_pinch(F: float, R: float, *, E: float, I=None, section=None, axis="y", axial=False, shear=False,
               G=None, nu=0.3, A=None, A_s=None, kappa=5 / 6, n: int = 64) -> RingResult:
    """Thin circular ring (radius of the mid-line R) compressed by two opposite forces F (Roark table 9.2,
    case 1). A quarter of the ring, statically indeterminate moment M0 at the point of the force from the condition of zero rotation
    (Castigliano, numerical quadrature); optionally tension and shear. Bending alone: M_load = −FR/π,
    M_side = FR(1/2 − 1/π), ΔD_load = −(π/4 − 2/π)·FR³/EI, ΔD_perp = (2/π − 1/2)·FR³/EI.
    """
    EI, EA, kGA = _arc_props(E, I, section, axis, axial, shear, G, nu, A, A_s, kappa)
    gx, gw = np.polynomial.legendre.leggauss(n)
    a = 0.25 * pi * (gx + 1)                       # angle from the point of the force (top) along the circumference to the side
    w = gw * 0.25 * pi * R
    # right half, cut at the point of the force: force F/2 downwards, moment M0; section at the point R(sin a, cos a)
    m_F = (F / 2) * R * np.sin(a)                  # moment from F/2 (increases curvature at the sides)
    n_F = -(F / 2) * np.sin(a)                     # normal force (compression)
    v_F = (F / 2) * np.cos(a)
    m0 = np.ones_like(a)
    # ∂U/∂M0 = 0 (rotation on the symmetry axis is zero); neither N nor V depends on M0
    M0 = -np.sum(w * m_F) / np.sum(w * m0)
    Mt = M0 + m_F

    def work(mu, nu_, vu):
        """Displacement corresponding to the virtual load (mu, nu_, vu) on the quarter."""
        val = np.sum(w * Mt * mu) / EI
        if EA:
            val += np.sum(w * n_F * nu_) / EA
        if kGA:
            val += np.sum(w * v_F * vu) / kGA
        return val

    # Virtual states are enough on the statically determinate basic system (reduction theorem) – here with M0 = 0.
    # Approach of the points of the forces: unit pair of forces in the direction F (the whole ring = 4 quarters).
    dl = -4 * work(0.5 * R * np.sin(a), -0.5 * np.sin(a), 0.5 * np.cos(a))
    # Expansion perpendicular: unit pair of horizontal forces outwards at the sides; at the section at the top by symmetry tension 1/2,
    # at the section a: moment (1/2)·R(1 − cos a) (the same sign convention as m_F), N = (1/2)cos a.
    dp = 4 * work(0.5 * R * (1 - np.cos(a)), 0.5 * np.cos(a), 0.5 * np.sin(a))
    return RingResult(dl, dp, float(M0), float(M0 + (F / 2) * R), -F / 2)


# ---------------------------------------------------------------------------------------------------------
# 5. Buckling – Euler critical force
# ---------------------------------------------------------------------------------------------------------
_KL = 4.493409457909064                           # first root tan(x) = x (clamped–pinned)
BUCKLING_BETA = {"fixed-free": 2.0, "pinned-pinned": 1.0, "fixed-pinned": pi / _KL, "fixed-fixed": 0.5}


@dataclass
class BucklingResult:
    F_cr: float              # critical force [N]
    beta: float              # effective length factor
    L_cr: float              # buckling length β·L [mm]
    i: float | None          # radius of gyration [mm]
    slenderness: float | None   # slenderness λ = L_cr / i
    sigma_cr: float | None   # critical stress [MPa]
    rel_slenderness: float | None = None   # relative slenderness λ̄ = √(A·f_y / F_cr) (given f_y)


def euler_buckling(L: float, *, E: float, I: float | None = None, section=None, axis: str | None = None,
                   case: str = "pinned-pinned", beta: float | None = None, A: float | None = None,
                   fy: float | None = None) -> BucklingResult:
    """F_cr = π²EI / (βL)². case: fixed-free (β = 2), pinned-pinned (1), fixed-pinned (0.699 – exactly
    π/4.4934), fixed-fixed (0.5); beta overrides. With section and without axis the weaker axis is taken.
    """
    if I is None:
        if section is None:
            raise ValueError("Give I or section.")
        I = min(section.Iy, section.Iz) if axis is None else (section.Iy if axis == "y" else section.Iz)
    A = A if A is not None else (section.A if section is not None else None)
    b = BUCKLING_BETA[case] if beta is None else float(beta)
    Lcr = b * L
    Fcr = pi ** 2 * E * I / Lcr ** 2
    i = sqrt(I / A) if A else None
    return BucklingResult(Fcr, b, Lcr, i, (Lcr / i if i else None), (Fcr / A if A else None),
                          (sqrt(A * fy / Fcr) if (A and fy) else None))
