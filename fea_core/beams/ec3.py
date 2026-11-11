"""Assessment of steel bars according to EN 1993-1-1 (ČSN EN 1993-1-1 + Czech NP: γM0 = γM1 = 1.00, γM2 = 1.25).

Input: section `SectionProps` (EN 1993 axes: y strong, z weak), material (steel grade → fy by thickness, table 3.1),
the internal force distributions along the bar (`MemberForces`: x, N, Vy, Vz, T, My, Mz; N > 0 tension) and the buckling / lateral-torsional
lengths (`MemberSpec`). Output `MemberCheck`: rows of the assessment (`CheckRow`: name, clause, effect, resistance, utilisation, note),
the largest utilisation, the governing row and the worst section. Rows starting with INFO are not counted in the utilisation (as in
`joints/assess.py`).

Covered: classification table 5.2 (internal parts with α from the plastic distribution for N + M and ψ from the elastic one, outstands,
tubes d/t, angles), class 4 simplified for I and RHS (EN 1993-1-5, 4.4: A_eff for pure compression, W_eff for
pure bending, one step without iteration), section resistance 6.2.3–6.2.10 at all points x, buckling 6.3.1 (planar and spatial / torsional 6.3.1.4),
lateral-torsional buckling 6.3.2 (Mcr for doubly symmetric I, C1 from the moment diagram, 6.3.2.2 and
6.3.2.3 with the factor f), interaction 6.3.3 (equations 6.61, 6.62) with the k-factors of annex B (method 2), Cm table B.3.

Simplifications (see the notes of the rows and `MemberCheck.warnings`): restrained torsion of open sections is not assessed (only
τ of pure torsion), Mcr of monosymmetric sections (U, T, L) approximately by the formula for doubly symmetric ones, the flanges of I
in table 5.2 always as compressed, class 4 only for I and RHS, the moment diagrams for Cm / C1 over the whole length of the bar,
the class of the bar for the assessments 6.3 = the highest class along the length (conservative), C1 of a general diagram = Cb (AISC 360).
The loads are taken as design loads (the partial factors γF are given by the user).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .props import SectionProps

INFO = "(informative) "
E_STEEL = 210000.0        # [MPa]
G_STEEL = 81000.0
SQ3 = math.sqrt(3.0)

# imperfections table 6.1 (buckling) and table 6.3 (lateral-torsional)
ALPHA = {"a0": 0.13, "a": 0.21, "b": 0.34, "c": 0.49, "d": 0.76}

# table 3.1 EN 1993-1-1 (EN 10025-2/3/4): grade → ((fy, fu) for t ≤ 40, (fy, fu) for 40 < t ≤ 80) [MPa]
STEEL_GRADES = {
    235: ((235.0, 360.0), (215.0, 360.0)),
    275: ((275.0, 430.0), (255.0, 410.0)),
    355: ((355.0, 510.0), (335.0, 470.0)),
    420: ((420.0, 520.0), (390.0, 520.0)),
    450: ((440.0, 550.0), (410.0, 550.0)),
    460: ((460.0, 540.0), (430.0, 530.0)),
}

OPEN_SHAPES = ("I", "U", "T", "L")
CLOSED_SHAPES = ("CHS", "TUBE", "RHS")
SOLID_SHAPES = ("RECT", "CIRC")


class EC3Error(ValueError):
    """Clear error of the assessment (unsupported case, missing data)."""


# --------------------------------------------------------------------------- material

def _grade_number(grade) -> int:
    if isinstance(grade, (int, float)):
        return int(grade)
    m = re.search(r"S\s*(\d{3})", str(grade).upper()) or re.fullmatch(r"\s*(\d{3})\s*", str(grade))
    if not m:
        raise EC3Error(f"Unknown steel grade “{grade}” (expected S235, S275, S355, S420, S450, S460)")
    n = int(m.group(1))
    if n not in STEEL_GRADES:
        raise EC3Error(f"Steel grade S{n} is not in Table 3.1 of EN 1993-1-1 – specify fy manually")
    return n


def _grade_row(grade, t: float):
    n = _grade_number(grade)
    if t > 80.0 + 1e-9:
        raise EC3Error(f"Thickness {t:g} mm > 80 mm is outside Table 3.1 of EN 1993-1-1 – specify fy manually")
    return STEEL_GRADES[n][0 if t <= 40.0 + 1e-9 else 1]


def fy_of(grade, t: float) -> float:
    """Yield strength fy [MPa] according to table 3.1 EN 1993-1-1 (EN 10025-2): t ≤ 40 mm and 40 < t ≤ 80 mm."""
    return _grade_row(grade, t)[0]


def fu_of(grade, t: float) -> float:
    """Tensile strength fu [MPa] according to table 3.1 EN 1993-1-1 (rolled products EN 10025)."""
    return _grade_row(grade, t)[1]


# --------------------------------------------------------------------------- vstupy

@dataclass
class MemberForces:
    """Internal forces along the bar: x [mm], N [N] (> 0 tension), Vy, Vz [N], T [N·mm], My, Mz [N·mm]."""
    x: np.ndarray
    N: np.ndarray | None = None
    Vy: np.ndarray | None = None
    Vz: np.ndarray | None = None
    T: np.ndarray | None = None
    My: np.ndarray | None = None
    Mz: np.ndarray | None = None

    def __post_init__(self):
        self.x = np.atleast_1d(np.asarray(self.x, float))
        n = len(self.x)
        for k in ("N", "Vy", "Vz", "T", "My", "Mz"):
            v = getattr(self, k)
            v = np.zeros(n) if v is None else np.asarray(v, float)
            if v.ndim == 0:
                v = np.full(n, float(v))
            if len(v) != n:
                raise EC3Error(f"Diagram {k} has {len(v)} values, there are {n} x positions")
            setattr(self, k, v)
        if n == 0:
            raise EC3Error("Empty internal force diagram")

    @classmethod
    def from_dict(cls, d: dict) -> "MemberForces":
        return cls(**{k: d.get(k) for k in ("x", "N", "Vy", "Vz", "T", "My", "Mz")})


@dataclass
class MemberSpec:
    """Bar for the assessment (lengths in mm). Buckling lengths = Lcr, otherwise factor·L."""
    L: float
    grade: str = "S235"
    fy: float | None = None                   # overrides fy from table 3.1
    E: float = E_STEEL
    G: float = G_STEEL
    gamma_M0: float = 1.00
    gamma_M1: float = 1.00
    gamma_M2: float = 1.25                    # (only for completeness – weakening by holes is not assessed)
    fabrication: str = "auto"                 # rolled | welded | hot_finished | cold_formed | auto
    Lcr_y: float | None = None
    Lcr_z: float | None = None
    Lcr_T: float | None = None                # torsional buckling
    ky: float = 1.0
    kz: float = 1.0
    kT: float = 1.0
    L_LT: float | None = None                 # length between restraints against lateral-torsional buckling (None = L)
    k_LT: float = 1.0                         # factor k (rotation about z at the ends)
    kw: float = 1.0                           # factor kw (warping)
    C1: float | None = None                   # None = from the moment diagram My
    C2: float | None = None                   # None = 0 for a linear diagram, otherwise 0.45 (distributed load)
    zg: float = 0.0                           # point of application of the load above the shear centre (+ = destabilising) [mm]
    Mcr: float | None = None                  # given Mcr [N·mm] (overrides the calculation)
    lt_restrained: bool = False               # continuously restrained against lateral-torsional buckling
    lt_method: str = "auto"                   # auto | general (6.3.2.2) | rolled (6.3.2.3)
    lambda_LT0: float = 0.4                   # 6.3.2.3 (Czech NP = recommended values)
    beta_LT: float = 0.75
    sway_y: bool = False                      # buckling with sway → Cmy = 0.9 (table B.3)
    sway_z: bool = False
    Cmy: float | None = None
    Cmz: float | None = None
    CmLT: float | None = None
    load_type: str = "auto"                   # shape of the transverse load for Cm: auto | uniform | concentrated
    section_class: int | None = None          # forced class (own section)
    eta: float = 1.2                          # η EN 1993-1-5 (shear buckling of the web)


@dataclass
class CheckRow:
    name: str
    clause: str
    effect: float
    resistance: float
    utilization: float
    unit: str = ""                            # "N" | "N·mm" | "MPa" | ""
    note: str = ""
    x: float | None = None

    @property
    def informative(self) -> bool:
        return self.name.startswith(INFO)


@dataclass
class MemberCheck:
    section: str
    grade: str
    fy: float
    section_class: int
    classification: list[dict]
    rows: list[CheckRow] = field(default_factory=list)
    utilization: float = 0.0
    governing: str = ""
    x_crit: float | None = None
    data: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.utilization <= 1.0

    def row(self, prefix: str) -> CheckRow | None:
        """First row whose name starts with `prefix`."""
        return next((r for r in self.rows if r.name.startswith(prefix)), None)

    def text(self) -> str:
        return report_text(self)


# --------------------------------------------------------------------------- helpers

def _d(sec: SectionProps, *keys, default=None):
    for k in keys:
        v = sec.dims.get(k)
        if v is not None:
            return float(v)
    return default


def _need(sec: SectionProps, *keys) -> list[float]:
    out = []
    for k in keys:
        v = _d(sec, *k.split("|"))
        if v is None:
            raise EC3Error(f"Cross-section {sec.name} ({sec.shape}) has no dimension “{k}” required for EN 1993-1-1")
        out.append(v)
    return out


def _shape(sec: SectionProps) -> str:
    s = (sec.shape or "custom").upper()
    if s == "TUBE":
        return "CHS"
    return s if s in OPEN_SHAPES + CLOSED_SHAPES + SOLID_SHAPES else "custom"


def governing_thickness(sec: SectionProps) -> float | None:
    """Thickness deciding fy (table 3.1): I/U/T max(tf, tw), tubes and L t, solid sections the minimum dimension."""
    s = _shape(sec)
    if s in ("I", "U", "T"):
        return max(_d(sec, "tf", default=0.0), _d(sec, "tw", default=0.0)) or None
    if s in ("CHS", "RHS", "L"):
        return _d(sec, "t")
    if s == "RECT":
        h, b = _d(sec, "h"), _d(sec, "b")
        return min(h, b) if h and b else None
    if s == "CIRC":
        return _d(sec, "D", "d")
    return _d(sec, "t", "tf")


def _fmt(v: float, unit: str) -> str:
    if not math.isfinite(v):
        return "∞" if v > 0 else "–"
    if unit == "N":
        return f"{v / 1e3:.1f} kN"
    if unit == "N·mm":
        return f"{v / 1e6:.2f} kN·m"
    if unit == "MPa":
        return f"{v:.1f} MPa"
    return f"{v:.3f}"


def chi_buckling(lam: float, alpha: float) -> float:
    """χ according to 6.3.1.2 (6.49)."""
    if lam <= 0.2:
        return 1.0
    phi = 0.5 * (1.0 + alpha * (lam - 0.2) + lam * lam)
    return min(1.0, 1.0 / (phi + math.sqrt(max(phi * phi - lam * lam, 0.0))))


def chi_lt_general(lam: float, alpha: float) -> float:
    """χLT general case 6.3.2.2 (6.56)."""
    return chi_buckling(lam, alpha)


def chi_lt_rolled(lam: float, alpha: float, lam0: float = 0.4, beta: float = 0.75) -> float:
    """χLT rolled and equivalent welded sections 6.3.2.3 (6.57), without the factor f."""
    if lam <= lam0:
        return 1.0
    phi = 0.5 * (1.0 + alpha * (lam - lam0) + beta * lam * lam)
    chi = 1.0 / (phi + math.sqrt(max(phi * phi - beta * lam * lam, 0.0)))
    return min(chi, 1.0, 1.0 / (lam * lam))


def mcr_doubly_symmetric(E, G, Iz, It, Iw, L, C1=1.0, C2=0.0, zg=0.0, k=1.0, kw=1.0) -> float:
    """Elastic critical moment of lateral-torsional buckling of a doubly symmetric section [N·mm] (NCCI SN003, ČSN EN 1993-1-1 old annex F)."""
    kL = k * L
    a = math.pi ** 2 * E * Iz / kL ** 2
    root = math.sqrt((k / kw) ** 2 * Iw / Iz + kL ** 2 * G * It / (math.pi ** 2 * E * Iz) + (C2 * zg) ** 2)
    return C1 * a * (root - C2 * zg)


# --------------------------------------------------------------------------- moment diagram

def moment_diagram(x, M, load_type: str = "auto") -> dict:
    """Analysis of the moment diagram between the ends of the bar: linear (ψ) or with transverse loading (Mh, Ms, ψ)."""
    x = np.asarray(x, float)
    M = np.asarray(M, float)
    Mmax = float(np.max(np.abs(M))) if len(M) else 0.0
    if Mmax <= 0.0:
        return {"kind": "zero", "Mmax": 0.0, "psi": 1.0}
    if len(M) < 2 or x[-1] == x[0]:
        return {"kind": "linear", "Mmax": Mmax, "psi": 1.0, "Mh": float(M[0]), "single": True}
    s = (x - x[0]) / (x[-1] - x[0])
    lin = M[0] + (M[-1] - M[0]) * s
    dev = M - lin
    m1, m2 = float(M[0]), float(M[-1])
    if abs(m1) >= abs(m2):
        Mh, other = m1, m2
    else:
        Mh, other = m2, m1
    psi = other / Mh if Mh != 0.0 else 0.0
    if float(np.max(np.abs(dev))) <= 1e-3 * Mmax:
        return {"kind": "linear", "Mmax": Mmax, "psi": psi, "Mh": Mh}
    i = int(np.argmax(np.abs(dev)))
    Ms, sp, dm = float(M[i]), float(s[i]), float(dev[i])
    kind = load_type if load_type in ("uniform", "concentrated") else "uniform"
    if load_type == "auto" and 0.0 < sp < 1.0:
        par = dm * s * (1 - s) / (sp * (1 - sp))
        tri = np.where(s <= sp, dm * s / sp, dm * (1 - s) / (1 - sp))
        r_par = float(np.sqrt(np.mean((dev - par) ** 2)))
        r_tri = float(np.sqrt(np.mean((dev - tri) ** 2)))
        if r_tri < 0.5 * r_par:
            kind = "concentrated"
    return {"kind": kind, "Mmax": Mmax, "psi": psi, "Mh": Mh, "Ms": Ms, "x_s": float(x[i])}


def cm_factor(dg: dict) -> float:
    """Equivalent uniform moment factor Cm according to table B.3."""
    k = dg["kind"]
    if k == "zero":
        return 1.0
    psi = max(-1.0, min(1.0, dg["psi"]))
    if k == "linear":
        return max(0.6 + 0.4 * psi, 0.4)
    uni = k == "uniform"
    Mh, Ms = dg["Mh"], dg["Ms"]
    if abs(Ms) < abs(Mh):
        a_s = Ms / Mh
        if a_s >= 0:
            c = 0.2 + 0.8 * a_s
        elif psi >= 0:
            c = (0.1 - 0.8 * a_s) if uni else (-0.8 * a_s)
        else:
            c = (0.1 * (1 - psi) - 0.8 * a_s) if uni else (0.2 * (-psi) - 0.8 * a_s)
        return max(c, 0.4)
    a_h = Mh / Ms
    if a_h >= 0 or psi >= 0:
        return (0.95 + 0.05 * a_h) if uni else (0.90 + 0.10 * a_h)
    return (0.95 + 0.05 * a_h * (1 + 2 * psi)) if uni else (0.90 + 0.10 * a_h * (1 + 2 * psi))


def c1_factor(dg: dict, x=None, M=None) -> tuple[float, str]:
    """C1 from the moment diagram: linear 1.88 − 1.40ψ + 0.52ψ² ≤ 2.7; otherwise Cb (ANSI/AISC 360, F1-1) ≤ 2.7."""
    k = dg["kind"]
    if k == "zero":
        return 1.0, "no moment"
    if k == "linear":
        psi = dg["psi"]
        return min(1.88 - 1.40 * psi + 0.52 * psi * psi, 2.7), f"linear diagram ψ = {psi:.3f}"
    x = np.asarray(x, float)
    M = np.asarray(M, float)
    xq = x[0] + (x[-1] - x[0]) * np.array([0.25, 0.5, 0.75])
    ma, mb, mc = np.abs(np.interp(xq, x, M))
    mm = dg["Mmax"]
    cb = 12.5 * mm / (2.5 * mm + 3 * ma + 4 * mb + 3 * mc)
    return min(cb, 2.7), "general diagram – Cb = 12.5Mmax/(2.5Mmax + 3MA + 4MB + 3MC) (AISC 360, F1-1)"


# --------------------------------------------------------------------------- klasifikace (tab. 5.2)

def _internal_limits(alpha: float, psi: float, eps: float) -> tuple[float, float, float]:
    if alpha <= 0:
        l1 = l2 = math.inf
    elif alpha > 0.5:
        l1, l2 = 396 * eps / (13 * alpha - 1), 456 * eps / (13 * alpha - 1)
    else:
        l1, l2 = 36 * eps / alpha, 41.5 * eps / alpha
    l3 = 42 * eps / (0.67 + 0.33 * psi) if psi > -1 else 62 * eps * (1 - psi) * math.sqrt(-psi)
    return l1, l2, l3


def _cls(ratio: float, lim) -> int:
    for k, l in enumerate(lim, start=1):
        if ratio <= l * (1 + 1e-9):
            return k
    return 4


def _strips(sec: SectionProps, axis: str) -> tuple[list[tuple[float, float, float]], float]:
    """Flanges (z_bottom, z_top, width) of a symmetric section perpendicular to the bending axis + height; for the plastic neutral axis."""
    s = _shape(sec)
    if s in ("I", "U"):
        h, b, tw, tf = _need(sec, "h", "b", "tw", "tf")
        if axis == "y":
            hw = h - 2 * tf
            return [(-h / 2, -hw / 2, b), (-hw / 2, hw / 2, tw), (hw / 2, h / 2, b)], h
        return [(-b / 2, -tw / 2, 2 * tf), (-tw / 2, tw / 2, h), (tw / 2, b / 2, 2 * tf)], b   # I only
    if s == "RHS":
        h, b, t = _need(sec, "h", "b", "t")
        if axis == "z":
            h, b = b, h
        return [(-h / 2, -h / 2 + t, b), (-h / 2 + t, h / 2 - t, 2 * t), (h / 2 - t, h / 2, b)], h
    raise EC3Error(f"Plastic stress distribution is not available for shape {s}")


def plastic_neutral_axis(strips, height: float, area_c: float) -> float:
    """Plastic neutral axis z0 (compression above z0) for the axial force Nc = fy·area_c at the full plastic moment M_N,Rd."""
    def net(z0):
        n = 0.0
        for lo, hi, w in strips:
            n += w * (max(hi - max(lo, z0), 0.0) - max(min(hi, z0) - lo, 0.0))
        return n
    lo, hi = -height / 2, height / 2                       # net(lo) = +A, net(hi) = −A
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if net(mid) > area_c:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _alpha_web(sec, axis, Nc, M, c, fy):
    """α = share of the compressed part of an internal web of length c under the plastic distribution for N_Ed (table 5.2).

    Neutral axis from the equilibrium N_Ed = fy·(A_comp − A_tens) – for the web of an I in the range of the web α = (c/2 + N/(2·tw·fy))/c.
    """
    strips, height = _strips(sec, axis)
    z0 = plastic_neutral_axis(strips, height, Nc / fy)
    return min(max((c / 2 - z0) / c, 0.0), 1.0)


def _internal_part(sec, name, c, t, eps, fy, base, grad, axis, Nc, M, count=1):
    """Internal compressed part: elastically σ = base ± grad at the ends (compression +), plastically α from (Nc, M)."""
    s1, s2 = base + grad, base - grad
    p = {"part": name, "type": "internal", "c": c, "t": t, "ratio": c / t, "count": count}
    if s1 <= 1e-9 * max(abs(base), abs(grad), 1.0):
        p.update(load="tension", alpha=0.0, psi=1.0, lim=(math.inf,) * 3, cls=1)
        return p
    psi = s2 / s1
    if grad <= 1e-12 * max(abs(base), 1.0):
        alpha, load = 1.0, "compression"
    else:
        alpha = _alpha_web(sec, axis, Nc, M, c, fy)
        load = "bending" if abs(Nc) <= 1e-9 * abs(M) else "bending + axial force"
    lim = _internal_limits(alpha, psi, eps)
    p.update(load=load, alpha=alpha, psi=psi, lim=lim, cls=_cls(c / t, lim))
    return p


def _outstand_part(name, c, t, eps, compressed, count=1):
    p = {"part": name, "type": "outstand", "c": c, "t": t, "ratio": c / t, "count": count}
    if not compressed:
        p.update(load="tension", lim=(math.inf,) * 3, cls=1)
        return p
    lim = (9 * eps, 10 * eps, 14 * eps)
    p.update(load="compression", lim=lim, cls=_cls(c / t, lim))
    return p


@dataclass
class Classification:
    cls: int
    parts: list[dict]
    note: str = ""


def classify(sec: SectionProps, fy: float, N: float = 0.0, My: float = 0.0, Mz: float = 0.0) -> Classification:
    """Section class according to table 5.2 for the combination of N (> 0 tension), My, Mz."""
    eps = math.sqrt(235.0 / fy)
    s = _shape(sec)
    Nc = -N
    A = sec.A
    if sec.Wel_y > 0 and sec.Wel_z > 0:
        smax = Nc / A + abs(My) / sec.Wel_y + abs(Mz) / sec.Wel_z
        compressed = smax > 1e-9
    else:                                              # without Wel: compression whenever a moment or a compression acts
        compressed = Nc > 0 or My != 0 or Mz != 0
    if s in SOLID_SHAPES:
        return Classification(1, [], "solid cross-section – local buckling not a concern")
    if s == "custom":
        raise EC3Error(f"Cross-section {sec.name}: a general shape cannot be classified per Table 5.2 – specify the class manually")
    if not compressed:
        return Classification(1, [{"part": "entire cross-section", "load": "tension", "cls": 1, "ratio": 0.0,
                                     "lim": (math.inf,) * 3}], "no compression")
    parts = []
    if s in ("I", "U"):
        h, b, tw, tf = _need(sec, "h", "b", "tw", "tf")
        r = _d(sec, "r", default=0.0)
        cw = h - 2 * tf - 2 * r
        grad = abs(My) * (cw / 2) / sec.Iy if sec.Iy else 0.0
        parts.append(_internal_part(sec, "web", cw, tw, eps, fy, Nc / A, grad, "y", Nc, My))
        cf = (b - tw - 2 * r) / 2 if s == "I" else b - tw - r
        parts.append(_outstand_part("flange", cf, tf, eps, compressed, count=4 if s == "I" else 2))
    elif s == "T":
        h, b, tw, tf = _need(sec, "h", "b", "tw", "tf")
        r = _d(sec, "r", default=0.0)
        parts.append(_outstand_part("flange", (b - tw - 2 * r) / 2, tf, eps, compressed, count=2))
        parts.append(_outstand_part("web (outstand)", h - tf - r, tw, eps, compressed))
    elif s == "RHS":
        h, b, t = _need(sec, "h", "b", "t")
        ch, cb = h - 3 * t, b - 3 * t
        g_h = abs(My) * (ch / 2) / sec.Iy if sec.Iy else 0.0
        base_h = Nc / A + (abs(Mz) * (b - t) / 2 / sec.Iz if sec.Iz else 0.0)
        parts.append(_internal_part(sec, "wall h (web)", ch, t, eps, fy, base_h, g_h, "y", Nc, My, count=2))
        g_b = abs(Mz) * (cb / 2) / sec.Iz if sec.Iz else 0.0
        base_b = Nc / A + (abs(My) * (h - t) / 2 / sec.Iy if sec.Iy else 0.0)
        parts.append(_internal_part(sec, "wall b (flange)", cb, t, eps, fy, base_b, g_b, "z", Nc, Mz, count=2))
    elif s == "CHS":
        D, t = _need(sec, "D|d", "t")
        lim = (50 * eps ** 2, 70 * eps ** 2, 90 * eps ** 2)
        parts.append({"part": "tube d/t", "type": "tube", "c": D, "t": t, "ratio": D / t, "load": "compression",
                      "lim": lim, "cls": _cls(D / t, lim)})
    elif s == "L":
        h, b, t = _need(sec, "h", "b", "t")
        h, b = max(h, b), min(h, b)
        r1, r2 = h / t, (b + h) / (2 * t)
        ok3 = r1 <= 15 * eps * (1 + 1e-9) and r2 <= 11.5 * eps * (1 + 1e-9)
        parts.append({"part": "angle h/t", "type": "angle iron", "c": h, "t": t, "ratio": r1, "load": "compression",
                      "lim": (0.0, 0.0, 15 * eps), "cls": 3 if ok3 else 4})
        parts.append({"part": "angle (b + h)/2t", "type": "angle iron", "c": (b + h) / 2, "t": t, "ratio": r2,
                      "load": "compression", "lim": (0.0, 0.0, 11.5 * eps), "cls": 3 if ok3 else 4})
        return Classification(3 if ok3 else 4, parts, "angles: Table 5.2 sheet 3 – class 3 at most")
    else:
        raise EC3Error(f"Unknown cross-section shape {sec.shape}")
    return Classification(max(p["cls"] for p in parts), parts)


# --------------------------------------------------------------------------- class 4 (EN 1993-1-5, 4.4)

def _ksig_internal(psi: float) -> float:
    if psi >= 1:
        return 4.0
    if psi > 0:
        return 8.2 / (1.05 + psi)
    if abs(psi + 1) < 1e-9:
        return 23.9                                    # table 4.1 for ψ = −1
    if psi > -1:
        return 7.81 - 6.29 * psi + 9.78 * psi ** 2
    if psi > -3:
        return 5.98 * (1 - psi) ** 2
    return 5.98 * 16


def rho_internal(c_t: float, eps: float, psi: float) -> float:
    lp = c_t / (28.4 * eps * math.sqrt(_ksig_internal(psi)))
    if lp <= 0.5 + math.sqrt(max(0.085 - 0.055 * psi, 0.0)):
        return 1.0
    return min(1.0, (lp - 0.055 * (3 + psi)) / lp ** 2)


def rho_outstand(c_t: float, eps: float, ksig: float = 0.43) -> float:
    lp = c_t / (28.4 * eps * math.sqrt(ksig))
    if lp <= 0.748:
        return 1.0
    return min(1.0, (lp - 0.188) / lp ** 2)


def _bending_eff(I0, A, height, flange, webs, eps):
    """W_eff for pure bending: compressed flange (list of (ΔA, z, own I)) and webs (c, t, number, class-3 limit).

    Step: subtract the ineffective part of the flange → shift of the centroid → ψ of the web → ineffective part of the web (be1 = 0.4 beff).
    """
    dA = sum(a for a, _, _ in flange)
    S = -sum(a * z for a, z, _ in flange)
    Iorig = I0 - sum(a * z * z + i for a, z, i in flange)
    zc = S / (A - dA)
    for c, t, n, cls_fn in webs:
        top, bot = c / 2 - zc, -c / 2 - zc
        psi = bot / top
        if cls_fn(c / t, psi) < 4:
            continue
        rho = rho_internal(c / t, eps, psi)
        bc = c / (1 - psi)
        beff = rho * bc
        Lh = (1 - rho) * bc
        zh = c / 2 - 0.4 * beff - Lh / 2
        a = n * t * Lh
        dA += a
        S -= a * zh
        Iorig -= a * zh * zh + n * t * Lh ** 3 / 12
    Aeff = A - dA
    zc = S / Aeff
    Ieff = Iorig - Aeff * zc * zc
    return Ieff / (height / 2 + abs(zc)), zc


def effective_section(sec: SectionProps, fy: float) -> dict:
    """Effective section of class 4 (EN 1993-1-5, 4.4) – simplified for I (doubly symmetric) and RHS.

    A_eff for pure compression (ψ = 1), W_eff,y / W_eff,z for pure bending (one step, without iterating the position of the neutral axis).
    Shift of the centroid of the effective section in compression e_N = 0 (doubly symmetric).
    """
    s = _shape(sec)
    eps = math.sqrt(235.0 / fy)
    if s not in ("I", "RHS"):
        raise EC3Error(f"Class 4 – not supported for shape {sec.shape} ({sec.name}); effective cross-section is available only for I and RHS")

    def cls3_internal(ct, psi):
        return 4 if ct > _internal_limits(0.5, psi, eps)[2] * (1 + 1e-9) else 3

    out = {"eN_y": 0.0, "eN_z": 0.0}
    if s == "I":
        h, b, tw, tf = _need(sec, "h", "b", "tw", "tf")
        r = _d(sec, "r", default=0.0)
        cw, cf = h - 2 * tf - 2 * r, (b - tw - 2 * r) / 2
        # pure compression
        rw = rho_internal(cw / tw, eps, 1.0) if cw / tw > 42 * eps else 1.0
        rf = rho_outstand(cf / tf, eps) if cf / tf > 14 * eps else 1.0
        out["Aeff"] = sec.A - (1 - rw) * cw * tw - 4 * (1 - rf) * cf * tf
        out["rho_c"] = {"web": rw, "flange": rf}
        # bending My: compressed top flange (2 projecting parts), web ψ from the shifted centroid
        Lf = (1 - rf) * cf
        flange = [(2 * Lf * tf, (h - tf) / 2, 2 * Lf * tf ** 3 / 12)] if Lf > 0 else []
        out["Weff_y"], out["zc_y"] = _bending_eff(sec.Iy, sec.A, h, flange, [(cw, tw, 1, cls3_internal)], eps)
        # bending Mz: compressed halves of both flanges, stress grows towards the end (table 4.2, σ1 at the free edge)
        psi_f = (tw / 2 + r) / (b / 2)
        kf = 0.57 - 0.21 * psi_f + 0.07 * psi_f ** 2
        rz = rho_outstand(cf / tf, eps, kf) if cf / tf > 14 * eps else 1.0
        Lz = (1 - rz) * cf
        fl = [(2 * Lz * tf, b / 2 - Lz / 2, 2 * tf * Lz ** 3 / 12)] if Lz > 0 else []
        out["Weff_z"], out["zc_z"] = _bending_eff(sec.Iz, sec.A, b, fl, [], eps)
    else:
        h, b, t = _need(sec, "h", "b", "t")
        ch, cb = h - 3 * t, b - 3 * t
        rh = rho_internal(ch / t, eps, 1.0) if ch / t > 42 * eps else 1.0
        rb = rho_internal(cb / t, eps, 1.0) if cb / t > 42 * eps else 1.0
        out["Aeff"] = sec.A - 2 * (1 - rh) * ch * t - 2 * (1 - rb) * cb * t
        out["rho_c"] = {"wall h": rh, "wall b": rb}
        for ax, H, cf_, cw_, I0 in (("y", h, cb, ch, sec.Iy), ("z", b, ch, cb, sec.Iz)):
            rf = rho_internal(cf_ / t, eps, 1.0) if cf_ / t > 42 * eps else 1.0
            Lf = (1 - rf) * cf_
            flange = [(Lf * t, (H - t) / 2, Lf * t ** 3 / 12)] if Lf > 0 else []
            out[f"Weff_{ax}"], out[f"zc_{ax}"] = _bending_eff(I0, sec.A, H, flange, [(cw_, t, 2, cls3_internal)], eps)
    out["Weff_y"] = min(out["Weff_y"], sec.Wel_y or out["Weff_y"])
    out["Weff_z"] = min(out["Weff_z"], sec.Wel_z or out["Weff_z"])
    return out


# --------------------------------------------------------------------------- buckling and lateral-torsional buckling curves

def buckling_curves(sec: SectionProps, fy: float, fabrication: str) -> tuple[str, str, str]:
    """Buckling curves (y, z) according to table 6.2 + description."""
    s = _shape(sec)
    hi = fy >= 460 - 1e-9
    if s == "I":
        h, b, tf = _need(sec, "h", "b", "tf")
        if fabrication == "welded":
            return ("b", "c", "welded I, tf ≤ 40") if tf <= 40 else ("c", "d", "welded I, tf > 40")
        if h / b > 1.2:
            if tf <= 40:
                return ("a0", "a0", "rolled I, h/b > 1.2, tf ≤ 40, S460") if hi else \
                    ("a", "b", "rolled I, h/b > 1.2, tf ≤ 40")
            if tf <= 100:
                return ("a", "a", "rolled I, h/b > 1.2, 40 < tf ≤ 100, S460") if hi else \
                    ("b", "c", "rolled I, h/b > 1.2, 40 < tf ≤ 100")
        if tf <= 100:
            return ("a", "a", "rolled I, h/b ≤ 1.2, tf ≤ 100, S460") if hi else \
                ("b", "c", "rolled I, h/b ≤ 1.2, tf ≤ 100")
        return ("c", "c", "rolled I, tf > 100, S460") if hi else ("d", "d", "rolled I, tf > 100")
    if s in ("CHS", "RHS"):
        if fabrication == "hot_finished":
            return ("a0", "a0", "hot-finished hollow section, S460") if hi else \
                ("a", "a", "hot-finished hollow section")
        if fabrication == "welded" and s == "RHS":
            return ("b", "b", "welded box section")
        return ("c", "c", "cold-formed hollow section")
    if s == "L":
        return ("b", "b", "angle iron")
    if s in ("U", "T") or s in SOLID_SHAPES:
        return ("c", "c", "U, T and solid sections")
    return ("c", "c", "general cross-section (curve c assumed)")


def lt_curve(sec: SectionProps, fabrication: str, method: str) -> tuple[str, str]:
    """Lateral-torsional curve: table 6.4 (general case) or table 6.5 (rolled and equivalent welded sections)."""
    s = _shape(sec)
    hb = None
    if s == "I":
        h, b = _need(sec, "h", "b")
        hb = h / b
    welded = fabrication == "welded"
    if method == "rolled" and hb is not None:
        if welded:
            return ("c", "welded I, h/b ≤ 2 (Table 6.5)") if hb <= 2 else ("d", "welded I, h/b > 2 (Table 6.5)")
        return ("b", "rolled I, h/b ≤ 2 (Table 6.5)") if hb <= 2 else ("c", "rolled I, h/b > 2 (Table 6.5)")
    if hb is not None:
        if welded:
            return ("c", "welded I, h/b ≤ 2 (Table 6.4)") if hb <= 2 else ("d", "welded I, h/b > 2 (Table 6.4)")
        return ("a", "rolled I, h/b ≤ 2 (Table 6.4)") if hb <= 2 else ("b", "rolled I, h/b > 2 (Table 6.4)")
    return ("d", "other cross-sections (Table 6.4)")


# --------------------------------------------------------------------------- assessment of the bar

def _row_max(rows: list[CheckRow]) -> CheckRow | None:
    cand = [r for r in rows if not r.informative and math.isfinite(r.utilization)] or \
        [r for r in rows if not r.informative]
    return max(cand, key=lambda r: r.utilization) if cand else None


def _worst(name, clause, x, eff, res, unit, note_fn=None) -> CheckRow:
    eff = np.asarray(eff, float)
    res = np.asarray(res, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(eff == 0, 0.0, np.abs(eff) / res)
    i = int(np.argmax(u))
    note = note_fn(i) if note_fn else ""
    return CheckRow(name, clause, float(abs(eff[i])), float(res[i]), float(u[i]), unit, note, float(x[i]))


def check_member(section: SectionProps, forces: MemberForces | dict, member: MemberSpec) -> MemberCheck:
    """Assessment of a bar according to EN 1993-1-1: section at all points x (6.2), buckling, lateral-torsional buckling, interaction (6.3)."""
    sec = section
    if isinstance(forces, dict):
        forces = MemberForces.from_dict(forces)
    f = forces
    m = member
    s = _shape(sec)
    warns: list[str] = []
    data: dict = {}
    if sec.A <= 0 or sec.Iy <= 0 or sec.Iz <= 0:
        raise EC3Error(f"Cross-section {sec.name}: A, Iy or Iz is missing")
    # --- material
    t_gov = governing_thickness(sec)
    if m.fy:
        fy = float(m.fy)
    elif t_gov is None:
        fy = fy_of(m.grade, 0.0)
        warns.append(f"Governing cross-section thickness unknown – fy = {fy:g} MPa for t ≤ 40 mm")
    else:
        fy = fy_of(m.grade, t_gov)
    E, G = m.E, m.G
    g0, g1 = m.gamma_M0, m.gamma_M1
    eps = math.sqrt(235.0 / fy)
    fab = m.fabrication
    if fab == "auto":
        fab = "cold_formed" if s in ("CHS", "RHS") else "rolled"
    data.update(fy=fy, eps=eps, t=t_gov, fabrication=fab)
    x = f.x
    n = len(x)
    A = sec.A
    has_pl = sec.Wpl_y > 0 and sec.Wpl_z > 0
    if not has_pl:
        warns.append("The cross-section has no plastic moduli Wpl – elastic check (as class 3)")
    # --- classification at each point
    if m.section_class:
        cls_pts = [int(m.section_class)] * n
        cl_info = Classification(int(m.section_class), [], "class specified by user")
        cl_pts = [cl_info] * n
    elif s == "custom":
        cls_pts = [3] * n
        cl_info = Classification(3, [], "general cross-section – class 3 assumed")
        cl_pts = [cl_info] * n
        warns.append("General cross-section: class cannot be determined from Table 5.2 – class 3 (elastic) assumed; check local buckling of thin walls separately or specify the class")
    else:
        cl_pts = [classify(sec, fy, f.N[i], f.My[i], f.Mz[i]) for i in range(n)]
        cls_pts = [c.cls for c in cl_pts]
    if not has_pl:
        cls_pts = [max(c, 3) for c in cls_pts]
    cls_pts = np.array(cls_pts)
    i_cls = int(np.argmax(cls_pts))
    mcls = int(cls_pts[i_cls])
    # classification for pure compression (information)
    try:
        cl_c = classify(sec, fy, N=-1.0) if s not in ("custom",) and not m.section_class else None
    except EC3Error:
        cl_c = None
    data["class_compression"] = cl_c.cls if cl_c else None
    eff = None
    if (cls_pts == 4).any():
        eff = effective_section(sec, fy)
        data["effective"] = eff
        warns.append("Class 4: effective cross-section simplified (EN 1993-1-5, 4.4 – A_eff in compression, W_eff in bending, no iteration)")

    def W(cls, ax):
        if cls <= 2:
            return getattr(sec, f"Wpl_{ax}")
        if cls == 3:
            return getattr(sec, f"Wel_{ax}")
        return eff[f"Weff_{ax}"]

    Aeff = eff["Aeff"] if eff else A
    A_c = np.where(cls_pts == 4, Aeff, A)                       # area for compression at the points
    Wy = np.array([W(c, "y") for c in cls_pts])
    Wz = np.array([W(c, "z") for c in cls_pts])
    rows: list[CheckRow] = []
    cl = "EN 1993-1-1, "
    # --- informative: classification
    gov_cl = cl_pts[i_cls]
    if gov_cl.parts:
        p = max(gov_cl.parts, key=lambda p: p["ratio"] / p["lim"][2] if math.isfinite(p["lim"][2]) else 0.0)
        if math.isfinite(p["lim"][2]):
            rows.append(CheckRow(INFO + f"Classification – class {mcls} ({p['part']}, {p.get('load', '')})",
                                 cl + "5.5, tab. 5.2", p["ratio"], p["lim"][2], p["ratio"] / p["lim"][2], "",
                                 "c/t against the class 3 limit; " + "; ".join(
                                     f"{q['part']}: c/t = {q['ratio']:.2f} (class {q['cls']})" for q in gov_cl.parts),
                                 float(x[i_cls])))
    data["classification_points"] = cls_pts.tolist()
    # --- 6.2.3 / 6.2.4 axial force
    sec_rows: list[CheckRow] = []
    sec_u = np.zeros(n)

    def add(row: CheckRow, u_arr):
        sec_rows.append(row)
        np.maximum(sec_u, u_arr, out=sec_u)

    def ratio(eff_, res_):
        eff_ = np.abs(np.asarray(eff_, float))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(eff_ == 0, 0.0, eff_ / res_)

    Npl = A * fy / g0
    Nc_Rd = A_c * fy / g0
    Nt = np.where(f.N > 0, f.N, 0.0)
    Ncomp = np.where(f.N < 0, -f.N, 0.0)
    if Nt.any():
        add(_worst("Tension (6.2.3)", cl + "6.2.3 – N_pl,Rd = A·fy/γM0", x, Nt, np.full(n, Npl), "N",
                   lambda i: "no reduction by holes"), ratio(Nt, Npl))
    if Ncomp.any():
        add(_worst("Compression (6.2.4)", cl + "6.2.4 – N_c,Rd = A·fy/γM0" + (" (A_eff)" if eff else ""), x, Ncomp, Nc_Rd,
                   "N"), ratio(Ncomp, Nc_Rd))
    # --- 6.2.7 torsion, 6.2.6 shear
    tau_lim = fy / (SQ3 * g0)
    tau_t = np.zeros(n)
    if np.any(f.T):
        if sec.Wt > 0:
            tau_t = np.abs(f.T) / sec.Wt
            add(_worst("Torsion – uniform (6.2.7)", cl + "6.2.7 – τ_t = T/W_t ≤ fy/(√3·γM0)", x, tau_t,
                       np.full(n, tau_lim), "MPa"), ratio(tau_t, tau_lim))
        else:
            warns.append("The cross-section has no torsion modulus W_t – torsion not checked")
        if s not in CLOSED_SHAPES and s not in SOLID_SHAPES:
            warns.append("Open section in torsion: only uniform (St. Venant) torsion checked, warping torsion (bimoment) neglected")
    closed = s in CLOSED_SHAPES
    with np.errstate(invalid="ignore"):
        if closed:
            redT = np.clip(1 - tau_t / tau_lim, 0.0, 1.0)
        else:
            redT = np.sqrt(np.clip(1 - tau_t / (1.25 * tau_lim), 0.0, 1.0))
    Vpl = {}
    for ax, V in (("z", f.Vz), ("y", f.Vy)):
        Av = getattr(sec, f"Av_{ax}")
        Vpl[ax] = Av * fy / (SQ3 * g0) * redT
        if np.any(V):
            if Av <= 0:
                warns.append(f"The cross-section has no shear area A_v,{ax} – shear V{ax} not checked")
                continue
            add(_worst(f"Shear V{ax} (6.2.6)", cl + "6.2.6 – V_pl,Rd = A_v·(fy/√3)/γM0" +
                       (" (6.2.7(9) with torsion)" if np.any(f.T) else ""), x, V, Vpl[ax], "N"), ratio(V, Vpl[ax]))
    # shear buckling of the web (warning only)
    if np.any(f.Vz) and s in ("I", "U", "RHS"):
        if s == "RHS":
            hw, tw_ = _need(sec, "h", "t")[0] - 2 * _need(sec, "t")[0], _need(sec, "t")[0]
        else:
            h_, tf_, tw_ = _need(sec, "h", "tf", "tw")
            hw = h_ - 2 * tf_
        if hw / tw_ > 72 * eps / m.eta:
            warns.append(f"Web shear buckling: hw/tw = {hw / tw_:.1f} > 72ε/η = {72 * eps / m.eta:.1f} – check per EN 1993-1-5, Section 5 (not assessed)")
    # --- 6.2.5 bending, 6.2.8 bending + shear
    My_a, Mz_a = np.abs(f.My), np.abs(f.Mz)
    McRd = {"y": Wy * fy / g0, "z": Wz * fy / g0}
    MV = {}
    rho_v = {}
    for ax, M_, V_, vax in (("y", My_a, f.Vz, "z"), ("z", Mz_a, f.Vy, "y")):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where((Vpl[vax] > 0) & (np.abs(V_) > 0.5 * Vpl[vax]),
                         (2 * np.abs(V_) / Vpl[vax] - 1) ** 2, 0.0)
        r = np.where(np.isfinite(r), np.minimum(r, 1.0), 1.0)
        rho_v[ax] = r
        MV[ax] = McRd[ax] * (1 - r * _shear_w_fraction(sec, ax, cls_pts, Wy if ax == "y" else Wz))
        MV[ax] = np.maximum(MV[ax], 0.0)
        if M_.any():
            add(_worst(f"Bending M{ax} (6.2.5)", cl + "6.2.5 – M_c,Rd = W·fy/γM0", x, M_, McRd[ax], "N·mm",
                       lambda i, ax=ax: f"class {cls_pts[i]}, W = {(Wy if ax == 'y' else Wz)[i] / 1000.0:.1f} cm³"),
                ratio(M_, McRd[ax]))
            if (r > 0).any():
                add(_worst(f"Bending M{ax} and shear V{vax} (6.2.8)", cl + "6.2.8 – ρ = (2V_Ed/V_pl,Rd − 1)²", x,
                           np.where(r > 0, M_, 0.0), np.where(r > 0, MV[ax], np.inf), "N·mm",
                           lambda i, ax=ax: f"ρ = {rho_v[ax][i]:.3f}; reduced fy in the shear area"),
                    ratio(np.where(r > 0, M_, 0.0), MV[ax]))
    # --- 6.2.9 bending + axial force
    if np.any(f.N) and (My_a.any() or Mz_a.any()):
        u_nm, notes = _nm_interaction(sec, s, fy, g0, f.N, My_a, Mz_a, cls_pts, A_c, MV, eff)
        i = int(np.argmax(u_nm))
        add(CheckRow("Bending and axial force (6.2.9)", cl + "6.2.9", float(u_nm[i]), 1.0, float(u_nm[i]), "",
                     notes[i], float(x[i])), u_nm)
    if (rho_v["y"] > 0).any() or (rho_v["z"] > 0).any():
        if np.any(f.N):
            warns.append("6.2.10: effect of shear in N + M interaction simplified – plastic moments replaced by M_V,Rd")
    rows += sec_rows
    i_sec = int(np.argmax(sec_u)) if n else 0
    data["section_utilization"] = sec_u.tolist()
    # --- 6.3 stabilita
    Nc_max = float(Ncomp.max()) if n else 0.0
    My_max, Mz_max = float(My_a.max()), float(Mz_a.max())
    Lcr_y = m.Lcr_y if m.Lcr_y else m.ky * m.L
    Lcr_z = m.Lcr_z if m.Lcr_z else m.kz * m.L
    Lcr_T = m.Lcr_T if m.Lcr_T else m.kT * m.L
    NRk = (Aeff if mcls == 4 else A) * fy
    WyM, WzM = W(mcls, "y"), W(mcls, "z")
    MyRk, MzRk = WyM * fy, WzM * fy
    cy, cz, cdesc = buckling_curves(sec, fy, fab)
    Ncr_y = math.pi ** 2 * E * sec.Iy / Lcr_y ** 2
    Ncr_z = math.pi ** 2 * E * sec.Iz / Lcr_z ** 2
    lam_y, lam_z = math.sqrt(NRk / Ncr_y), math.sqrt(NRk / Ncr_z)
    chi_y, chi_z = chi_buckling(lam_y, ALPHA[cy]), chi_buckling(lam_z, ALPHA[cz])
    data.update(Lcr_y=Lcr_y, Lcr_z=Lcr_z, Ncr_y=Ncr_y, Ncr_z=Ncr_z, lambda_y=lam_y, lambda_z=lam_z,
                curve_y=cy, curve_z=cz, curve_note=cdesc, chi_y=chi_y, chi_z=chi_z, member_class=mcls, NRk=NRk)
    if Nc_max > 0:
        for ax, chi, lam, Ncr, cur in (("y", chi_y, lam_y, Ncr_y, cy), ("z", chi_z, lam_z, Ncr_z, cz)):
            note = f"Ncr = {Ncr / 1000.0:.1f} kN, λ̄ = {lam:.3f}, curve {cur} (α = {ALPHA[cur]}), χ = {chi:.3f}"
            if Nc_max <= 0.04 * Ncr or lam <= 0.2:
                note += "; per 6.3.1.2(4) buckling may be neglected"
            rows.append(CheckRow(f"Buckling about {ax} (6.3.1)", cl + "6.3.1 – N_b,Rd = χ·A·fy/γM1", Nc_max,
                                 chi * NRk / g1, Nc_max / (chi * NRk / g1), "N", note))
        if s in OPEN_SHAPES:
            Ncr_T, Ncr_TF = _torsional_ncr(sec, E, G, Ncr_y, Ncr_z, Lcr_T)
            lam_T = math.sqrt(NRk / Ncr_TF)
            chi_T = chi_buckling(lam_T, ALPHA[cz])
            data.update(Lcr_T=Lcr_T, Ncr_T=Ncr_T, Ncr_TF=Ncr_TF, lambda_T=lam_T, chi_T=chi_T)
            rows.append(CheckRow("Torsional / torsional-flexural buckling (6.3.1.4)", cl + "6.3.1.4 – curve for axis z",
                                 Nc_max, chi_T * NRk / g1, Nc_max / (chi_T * NRk / g1), "N",
                                 f"Ncr,T = {Ncr_T / 1e3:.1f} kN, Ncr,TF = {Ncr_TF / 1e3:.1f} kN, "
                                 f"λ̄_T = {lam_T:.3f}, χ = {chi_T:.3f}"))
    # lateral-torsional buckling
    lt_relevant = (My_max > 0 and not m.lt_restrained and s not in CLOSED_SHAPES and s != "CIRC"
                   and sec.Iz < sec.Iy * (1 - 1e-6))
    chi_LT = 1.0
    dg_y = moment_diagram(x, f.My, m.load_type)
    dg_z = moment_diagram(x, f.Mz, m.load_type)
    data.update(diagram_y=dg_y, diagram_z=dg_z)
    if lt_relevant:
        L_LT = m.L_LT or m.L
        if m.C1:
            C1, c1note = float(m.C1), "C1 specified"
        else:
            C1, c1note = c1_factor(dg_y, x, f.My)
        C2 = m.C2 if m.C2 is not None else (0.0 if dg_y["kind"] in ("linear", "zero") else 0.45)
        if m.Mcr:
            Mcr = float(m.Mcr)
        else:
            Mcr = mcr_doubly_symmetric(E, G, sec.Iz, sec.J, sec.Iw, L_LT, C1, C2, m.zg, m.k_LT, m.kw)
            if s in ("U", "T", "L"):
                warns.append(f"Lateral-torsional buckling of section {s}: Mcr approximated by the formula for a doubly symmetric section (without factor C3 and z_j) – verify")
        lam_LT = math.sqrt(MyRk / Mcr)
        method = m.lt_method
        if method == "auto":
            method = "rolled" if s == "I" else "general"
        cur, cnote = lt_curve(sec, fab, method)
        aLT = ALPHA[cur]
        fmod, kc = 1.0, None
        if method == "rolled":
            lam0, beta = m.lambda_LT0, m.beta_LT
            chi0 = chi_lt_rolled(lam_LT, aLT, lam0, beta)
            if dg_y["kind"] == "linear" and not m.C1:
                kc = 1.0 / (1.33 - 0.33 * dg_y["psi"])
            else:
                kc = 1.0 / math.sqrt(C1)
            fmod = min(1.0, 1 - 0.5 * (1 - kc) * (1 - 2 * (lam_LT - 0.8) ** 2))
            chi_LT = min(1.0, chi0 / fmod, 1.0 / lam_LT ** 2)
        else:
            lam0 = 0.2
            chi0 = chi_lt_general(lam_LT, aLT)
            chi_LT = chi0
        ignore = lam_LT <= lam0 or My_max <= lam0 ** 2 * Mcr
        if ignore:
            chi_LT = 1.0
        MbRd = chi_LT * MyRk / g1
        data.update(L_LT=L_LT, C1=C1, C2=C2, Mcr=Mcr, lambda_LT=lam_LT, curve_LT=cur, chi_LT=chi_LT,
                    chi_LT0=chi0, f=fmod, kc=kc, lt_method=method, Mb_Rd=MbRd)
        note = (f"Mcr = {Mcr / 1000000.0:.1f} kN·m (C1 = {C1:.3f} – {c1note}), λ̄LT = {lam_LT:.3f}, curve {cur} ({cnote}), χLT = {chi_LT:.3f}" + (f", kc = {kc:.3f}, f = {fmod:.3f}" if kc else ""))
        if ignore:
            note += "; per 6.3.2.2(4) lateral-torsional buckling may be neglected"
        rows.append(CheckRow("Lateral-torsional buckling (6.3.2)", cl + ("6.3.2.3" if method == "rolled" else "6.3.2.2") +
                             " – M_b,Rd = χLT·Wy·fy/γM1", My_max, MbRd, My_max / MbRd, "N·mm", note))
    elif My_max > 0:
        why = ("secured against lateral-torsional buckling" if m.lt_restrained else
               "closed / circular section – lateral-torsional buckling not a concern" if s in CLOSED_SHAPES or s == "CIRC" else
               "Iz ≥ Iy – lateral-torsional buckling not a concern")
        data["lt_note"] = why
    # interaction 6.3.3 (Annex B)
    if Nc_max > 0 and (My_max > 0 or Mz_max > 0):
        susceptible = (s in OPEN_SHAPES or s == "custom") and not m.lt_restrained
        Cmy = m.Cmy if m.Cmy else (0.9 if m.sway_y else cm_factor(dg_y))
        Cmz = m.Cmz if m.Cmz else (0.9 if m.sway_z else cm_factor(dg_z))
        CmLT = m.CmLT if m.CmLT else max(cm_factor(dg_y), 0.4)
        nY = Nc_max / (chi_y * NRk / g1)
        nZ = Nc_max / (chi_z * NRk / g1)
        k = annex_b_factors(mcls, lam_y, lam_z, nY, nZ, Cmy, Cmz, CmLT, susceptible,
                            open_section=s in OPEN_SHAPES or s == "custom")
        dMy = Nc_max * (eff["eN_z"] if eff and mcls == 4 else 0.0)
        dMz = Nc_max * (eff["eN_y"] if eff and mcls == 4 else 0.0)
        ty = (My_max + dMy) / (chi_LT * MyRk / g1)
        tz = (Mz_max + dMz) / (MzRk / g1)
        u61 = nY + k["kyy"] * ty + k["kyz"] * tz
        u62 = nZ + k["kzy"] * ty + k["kzz"] * tz
        data.update(Cmy=Cmy, Cmz=Cmz, CmLT=CmLT, nY=nY, nZ=nZ, annex_b=k, susceptible=susceptible,
                    interaction=(u61, u62))
        tab = "B.2" if susceptible else "B.1"
        knote = ", ".join(f"{kk} = {v:.3f}" for kk, v in k.items())
        base = (f"table {tab}, class {mcls}; Cmy = {Cmy:.3f}, Cmz = {Cmz:.3f}" +
                (f", CmLT = {CmLT:.3f}" if susceptible else "") + f", χLT = {chi_LT:.3f}; {knote}")
        rows.append(CheckRow("N + M interaction – buckling about y (6.61)", cl + "6.3.3, equation (6.61), annex B",
                             u61, 1.0, u61, "", base))
        rows.append(CheckRow("N + M interaction – buckling about z (6.62)", cl + "6.3.3, equation (6.62), annex B",
                             u62, 1.0, u62, "", base))
    gov = _row_max(rows)
    out = MemberCheck(sec.name, str(m.grade), fy, mcls, gov_cl.parts, rows,
                      gov.utilization if gov else 0.0, gov.name if gov else "",
                      float(x[i_sec]) if sec_rows else None, data, warns)
    return out


def _shear_w_fraction(sec, ax, cls_pts, W_arr) -> np.ndarray:
    """Share of the section modulus belonging to the shear area (6.2.8(3) – reduced fy in the shear area).

    I, U (axis y) and RHS: the webs in the direction of shear (6.30: ρ·A_w²/(4t_w)); others simplified A_v/A (conservative).
    """
    s = _shape(sec)
    plastic = cls_pts <= 2
    Wv_pl = Wv_el = None
    if s in ("I", "U") and ax == "y":
        h, tw, tf = _need(sec, "h", "tw", "tf")
        hw = h - 2 * tf
        Wv_pl, Wv_el = tw * hw ** 2 / 4, tw * hw ** 3 / (6 * h)
    elif s == "I" and ax == "z":
        b, tf = _need(sec, "b", "tf")
        Wv_pl, Wv_el = 2 * tf * b ** 2 / 4, tf * b ** 2 / 3
    elif s == "RHS":
        h, b, t = _need(sec, "h", "b", "t")
        if ax == "z":
            h, b = b, h
        hw = h - 2 * t
        Wv_pl, Wv_el = 2 * t * hw ** 2 / 4, t * hw ** 3 / (3 * h)
    if Wv_pl is None:
        Av = getattr(sec, "Av_z" if ax == "y" else "Av_y")
        return np.full(len(cls_pts), min(Av / sec.A, 1.0) if Av > 0 else 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fr = np.where(plastic, Wv_pl, Wv_el) / W_arr
    return np.clip(np.nan_to_num(fr, nan=1.0), 0.0, 1.0)


def _nm_interaction(sec, s, fy, g0, N, My, Mz, cls_pts, A_c, MV, eff):
    """6.2.9 at each point → (utilisation, notes)."""
    n = len(N)
    u = np.zeros(n)
    notes = []
    A = sec.A
    Npl = A * fy / g0
    for i in range(n):
        c = cls_pts[i]
        Ni, myi, mzi = abs(N[i]), My[i], Mz[i]
        mvy, mvz = MV["y"][i], MV["z"][i]
        if c <= 2 and s in ("I", "RHS", "CHS"):
            nn = Ni / Npl
            if nn >= 1:
                u[i], txt = nn, f"n = {nn:.3f} ≥ 1"
                notes.append(txt)
                continue
            if s == "I":
                h, b, tw, tf = _need(sec, "h", "b", "tw", "tf")
                hw = h - 2 * tf
                a = min((A - 2 * b * tf) / A, 0.5)
                if Ni <= 0.25 * Npl and Ni <= 0.5 * hw * tw * fy / g0:
                    MNy = mvy
                else:
                    MNy = min(mvy * (1 - nn) / (1 - 0.5 * a), mvy)
                if Ni <= hw * tw * fy / g0 or nn <= a:
                    MNz = mvz
                else:
                    MNz = mvz * (1 - ((nn - a) / (1 - a)) ** 2)
                al, be = 2.0, max(5 * nn, 1.0)
                txt = f"I class {c}: n = {nn:.3f}, a = {a:.3f}"
            elif s == "RHS":
                h, b, t = _need(sec, "h", "b", "t")
                aw = min((A - 2 * b * t) / A, 0.5)
                af = min((A - 2 * h * t) / A, 0.5)
                MNy = min(mvy * (1 - nn) / (1 - 0.5 * aw), mvy)
                MNz = min(mvz * (1 - nn) / (1 - 0.5 * af), mvz)
                al = be = min(1.66 / (1 - 1.13 * nn ** 2), 6.0)
                txt = f"RHS class {c}: n = {nn:.3f}, a_w = {aw:.3f}, a_f = {af:.3f}"
            else:
                MNy = mvy * (1 - nn ** 1.7)
                MNz = mvz * (1 - nn ** 1.7)
                al = be = 2.0
                txt = f"CHS class {c}: n = {nn:.3f}"
            ry = myi / MNy if myi else 0.0
            rz = mzi / MNz if mzi else 0.0
            if myi and mzi:
                u[i] = ry ** al + rz ** be
                txt += f", (My/MN,y,Rd)^{al:.2f} + (Mz/MN,z,Rd)^{be:.2f}"
            else:
                u[i] = max(ry, rz)
                txt += ", M_Ed/M_N,Rd"
            txt += f"; MN,y,Rd = {MNy / 1e6:.2f}, MN,z,Rd = {MNz / 1e6:.2f} kN·m"
        else:
            NRd = (A_c[i] if N[i] < 0 else A) * fy / g0
            if c == 4:
                dMy = Ni * eff["eN_z"]
                dMz = Ni * eff["eN_y"]
            else:
                dMy = dMz = 0.0
            u[i] = Ni / NRd + (myi + dMy) / mvy + (mzi + dMz) / mvz if mvy > 0 and mvz > 0 else math.inf
            txt = {3: "class 3: σ = N/A + My/Wel,y + Mz/Wel,z ≤ fy/γM0 (6.42)",
                   4: "class 4: N/(A_eff·fy) + (My + N·e_N)/(W_eff,y·fy) + … (6.44)"}.get(
                int(c), f"class {c}: linear sum N/N_Rd + My/M_y,Rd + Mz/M_z,Rd (6.2.1(7), conservative)")
        notes.append(txt)
    return u, notes


def _torsional_ncr(sec, E, G, Ncr_y, Ncr_z, Lcr_T) -> tuple[float, float]:
    """Ncr,T (torsion) and Ncr,TF (spatial buckling with the shear centre ys, zs away from the centroid; Timoshenko & Gere).

    The offset of the shear centre ys couples torsion with buckling in the z direction (bending about y, Ncr,y), zs with bending about z.
    Monosymmetric section → quadratic equation (β = 1 − (y0/i0)²), general → the smallest root of the 3×3 system.
    """
    y0, z0 = sec.ys, sec.zs
    i02 = (sec.Iy + sec.Iz) / sec.A + y0 * y0 + z0 * z0
    NT = (G * sec.J + math.pi ** 2 * E * sec.Iw / Lcr_T ** 2) / i02
    tiny = 1e-6 * math.sqrt(i02)
    if abs(y0) < tiny and abs(z0) < tiny:
        return NT, NT
    if abs(y0) < tiny or abs(z0) < tiny:
        Na, e = (Ncr_y, y0) if abs(z0) < tiny else (Ncr_z, z0)
        beta = 1 - e * e / i02
        S = Na + NT
        return NT, (S - math.sqrt(max(S * S - 4 * beta * Na * NT, 0.0))) / (2 * beta)
    K = np.diag([Ncr_z, Ncr_y, i02 * NT])
    B = np.array([[1.0, 0.0, -z0], [0.0, 1.0, y0], [-z0, y0, i02]])
    ev = np.real(np.linalg.eigvals(np.linalg.solve(B, K)))
    return NT, float(ev[ev > 0].min())


def annex_b_factors(cls: int, lam_y: float, lam_z: float, nY: float, nZ: float, Cmy: float, Cmz: float,
                    CmLT: float, susceptible: bool, open_section: bool = True) -> dict:
    """Interaction factors kyy, kyz, kzy, kzz – annex B, table B.1 (non-sway in torsion) / B.2."""
    plastic = cls <= 2
    if plastic:
        kyy = Cmy * min(1 + (lam_y - 0.2) * nY, 1 + 0.8 * nY)
        if open_section:
            kzz = Cmz * min(1 + (2 * lam_z - 0.6) * nZ, 1 + 1.4 * nZ)
        else:
            kzz = Cmz * min(1 + (lam_z - 0.2) * nZ, 1 + 0.8 * nZ)
        kyz = 0.6 * kzz
        kzy = 0.6 * kyy
    else:
        kyy = Cmy * min(1 + 0.6 * lam_y * nY, 1 + 0.6 * nY)
        kzz = Cmz * min(1 + 0.6 * lam_z * nZ, 1 + 0.6 * nZ)
        kyz = kzz
        kzy = 0.8 * kyy
    if susceptible:
        c = 0.1 if plastic else 0.05
        kzy = 1 - c * min(lam_z, 1.0) * nZ / (CmLT - 0.25)
        if plastic and lam_z < 0.4:
            kzy = min(0.6 + lam_z, 1 - 0.1 * lam_z * nZ / (CmLT - 0.25))
    return {"kyy": kyy, "kyz": kyz, "kzy": kzy, "kzz": kzz}


def report_text(c: MemberCheck) -> str:
    """Text printout of the assessment (CLI, MCP, report)."""
    st = "PASS" if c.ok else "FAIL"
    lines = [f"{c.section} {c.grade} (fy = {c.fy:g} MPa), class {c.section_class}: utilization {c.utilization:.3f} – {st}; governing: {c.governing}"]
    for r in c.rows:
        xs = f" [x = {r.x:g} mm]" if r.x is not None else ""
        if r.unit:
            val = f"{_fmt(r.effect, r.unit)} / {_fmt(r.resistance, r.unit)}"
        else:
            val = f"{r.effect:.3f} / {r.resistance:.4g}"
        lines.append(f"  {r.name}: {val} → {r.utilization:.3f}{xs}  ({r.clause}{'; ' + r.note if r.note else ''})")
    lines += [f"  ! {w}" for w in c.warnings]
    return "\n".join(lines)
