"""Effective properties of the shell and the infill of the printed part (homogenisation, F4).

Material axes: 1, 2 = the plane of the layer (XY of the printer), 3 = build direction Z. Shell = full extrusion
(properties from Material), core = infill with density ρ: E_i,core = E_i · k_i · ρ^n_i, strength
X_i,core = X_i · s_i · ρ^m_i (Gibson–Ashby: tension-loaded lattices n ≈ 1, bending n ≈ 2–3).
The coefficients of the infills are INDICATIVE (literature, own estimate) – they are calibrated by tests (calibrate.py).
2D infills (grid, triangles, honeycomb…) have vertical walls → in the direction Z they carry with the whole section of the walls (k_z = 1).
Stiffness in Voigt notation: σ = C·ε, order 11, 22, 33, 12, 13, 23, engineering shear (γ = 2ε),
as *ELASTIC, TYPE=ORTHO in ccx (D1212 = G12; verified 12. 10. 2026 against ENGINEERING CONSTANTS).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                # study imports printing.model → no cyclic import
    from ..study import Material

# key: (description, k_xy, n_xy, k_z, n_z, s_xy, m_xy, s_z, m_z)
PATTERNS: dict[str, tuple] = {
    "grid":          ("Grid",               0.35, 1.0, 1.0, 1.0, 0.35, 1.0, 0.9, 1.0),
    "rectilinear":   ("Rectilinear",     0.35, 1.0, 0.5, 1.5, 0.35, 1.0, 0.5, 1.5),
    "alignedrectilinear": ("Aligned rectilinear",    0.35, 1.0, 0.5, 1.5, 0.35, 1.0, 0.5, 1.5),
    "zigzag":        ("Zig-zag",            0.35, 1.0, 0.5, 1.5, 0.35, 1.0, 0.5, 1.5),
    "triangles":     ("Triangles",                0.33, 1.0, 1.0, 1.0, 0.33, 1.0, 0.9, 1.0),
    "tri-hexagon":   ("Triangles-hexagons",  0.30, 1.0, 1.0, 1.0, 0.30, 1.0, 0.9, 1.0),
    "honeycomb":     ("Honeycomb",         1.50, 3.0, 1.0, 1.0, 0.50, 2.0, 0.9, 1.0),
    "cubic":         ("Cubic",             0.30, 1.3, 0.30, 1.3, 0.30, 1.3, 0.30, 1.3),
    "adaptivecubic": ("Adaptive cubic",           0.30, 1.3, 0.30, 1.3, 0.30, 1.3, 0.30, 1.3),
    "crosshatch":    ("Crosshatch",      0.30, 1.5, 0.30, 1.5, 0.30, 1.5, 0.30, 1.5),
    "3dhoneycomb":   ("3D honeycomb",                  0.30, 1.5, 0.30, 1.5, 0.30, 1.5, 0.30, 1.5),
    "gyroid":        ("Gyroid",                      1.00, 1.7, 1.00, 1.7, 0.80, 1.5, 0.80, 1.5),
    "concentric":    ("Concentric",     0.50, 1.0, 1.0, 1.0, 0.50, 1.0, 0.9, 1.0),
    "lightning":     ("Lightning – non-structural", 0.001, 1.0, 0.001, 1.0, 0.001, 1.0, 0.001, 1.0),
}
FULL = 0.99            # from this density the core is full (the slicer prints a full infill)

# In-plane anisotropy of the infills (F5, INDICATIVE): G12 of the core / E_xy of the core in the print axes 1, 2.
# Line infills (grid, rectilinear, …) have lines in two families ±45° to the print axes (Bambu Studio: the infill direction
# 45°). From the symmetry ±45° E1 = E2 (the table values are kept = tensile tests along the print axes, i.e. 45° to the lines),
# but the shear in the axes 1-2 is tension / compression of the lines – the lattice carries by tension (Gibson & Ashby, Cellular Solids, 2nd ed. 1997, ch. 4;
# a square lattice loaded along the struts E = ρ/2·E_s, in shear under 45° G = ρ/4·E_s). Therefore G12 = (ρ/4)/(k_xy·ρ)·E1 =
# 0.25/0.35·E1 ≈ 0.71·E1 (isotropically it would be E1/(2(1 + ν)) ≈ 0.37·E1). Check: the modulus along the lines (45° to the axes)
# then comes out ≈ 0.52·ρ·E_s ≈ ρ/2·E_s as for the ideal lattice. The shear strength S_xy is NOT increased (conservative).
# Infills without the ±45° families (triangles, honeycomb, gyroid, cubic, concentric) are approximately isotropic in the plane
# (Gibson & Ashby: the triangular and hexagonal lattice are isotropic) → isotropic G12 (None).
INPLANE_SHEAR: dict[str, float] = {
    "grid": 0.25 / 0.35, "rectilinear": 0.25 / 0.35, "alignedrectilinear": 0.25 / 0.35, "zigzag": 0.25 / 0.35,
}


def pattern_label(key: str) -> str:
    return PATTERNS[key][0] if key in PATTERNS else key


def ratios(pattern: str, density: float) -> dict[str, float]:
    """Ratios of the properties of the core to the full extrusion: E_xy, E_z, X_xy, X_z."""
    if density >= FULL:
        return {"E_xy": 1.0, "E_z": 1.0, "X_xy": 1.0, "X_z": 1.0}
    if pattern not in PATTERNS:
        raise KeyError(f"Unknown infill type '{pattern}'")
    _, kx, nx, kz, nz, sx, mx, sz, mz = PATTERNS[pattern]
    r = max(float(density), 0.0)
    return {"E_xy": kx * r ** nx, "E_z": kz * r ** nz, "X_xy": sx * r ** mx, "X_z": sz * r ** mz}


@dataclass
class Ortho:
    """Orthotropic material in the print axes + strengths (MPa). Without E2/Yt/Yc/G12v transversely isotropic (axis 3).
    Perimeters with line anisotropy: axis 1 = along the line, 2 = across the line in the plane of the layer, 3 = build.
    """
    E1: float
    E3: float
    nu12: float
    nu13: float              # ε3 = −ν13·σ1/E1 (a ε3 = −ν13·σ2/E2)
    G13: float
    Xt: float
    Xc: float
    Zt: float
    Zc: float
    Sxy: float
    Sz: float
    E2: float | None = None  # None = E1
    Yt: float | None = None  # tensile strength in direction 2 (None = Xt)
    Yc: float | None = None
    G12v: float | None = None    # None = E1/(2(1 + ν12))

    @property
    def G12(self) -> float:
        return self.G12v if self.G12v is not None else self.E1 / (2 * (1 + self.nu12))

    @property
    def E2v(self) -> float:
        return self.E2 if self.E2 is not None else self.E1

    @property
    def Ytv(self) -> float:
        return self.Yt if self.Yt is not None else self.Xt

    @property
    def Ycv(self) -> float:
        return self.Yc if self.Yc is not None else self.Xc

    def stiffness(self) -> np.ndarray:
        """C (6×6) in the print axes, Voigt 11 22 33 12 13 23 (engineering shear)."""
        E1, E2, E3, n12, n13 = self.E1, self.E2v, self.E3, self.nu12, self.nu13
        S = np.zeros((6, 6))
        S[:3, :3] = [[1 / E1, -n12 / E1, -n13 / E1],
                     [-n12 / E1, 1 / E2, -n13 / E2],
                     [-n13 / E1, -n13 / E2, 1 / E3]]
        S[3, 3], S[4, 4], S[5, 5] = 1 / self.G12, 1 / self.G13, 1 / self.G13
        return np.linalg.inv(S)


def solid(mat: Material) -> Ortho:
    """Full extrusion (shell). Missing values: ν_z = ν, G_z = E_z/(2(1+ν)), Rc = Rm, S = R/√3."""
    if not mat.is_print:
        raise ValueError(f"Material '{mat.name}' is not a printed material (E_z missing)")
    Xt = mat.Rm or 1.0
    Zt = mat.Rm_z or Xt
    nu_z = mat.nu_z if mat.nu_z is not None else mat.nu
    return Ortho(E1=mat.E, E3=mat.E_z, nu12=mat.nu, nu13=nu_z,
                 G13=mat.G_z or mat.E_z / (2 * (1 + mat.nu)),
                 Xt=Xt, Xc=mat.Rc or Xt, Zt=Zt, Zc=mat.Rc_z or mat.Rc or Zt,
                 Sxy=mat.S_xy or Xt / math.sqrt(3), Sz=mat.S_z or Zt / math.sqrt(3))


def line_solid(mat: Material, rE: float, rR: float) -> Ortho:
    """Perimeter with line anisotropy: axis 1 = along the line, 2 = across (in the plane of the layer), 3 = build.
    E, Rm of the material (XY) = average along and across: along = 2/(1 + r)·XY, across = r·along. Compression, shear
    in the plane (G12 = E/(2(1 + ν)) from the average) and the properties in the direction Z unchanged. ν_LT = ν (ν_TL = ν·E_T/E_L).
    """
    s = solid(mat)
    if rE == 1.0 and rR == 1.0:
        return s
    from dataclasses import replace
    EL = 2 * s.E1 / (1 + rE)
    XL = 2 * s.Xt / (1 + rR)
    return replace(s, E1=EL, E2=rE * EL, G12v=s.G12, Xt=XL, Yt=rR * XL, Yc=s.Xc)


HOLLOW = 0.005          # infill below 0.5 % = hollow part (shell only)
HOLLOW_STIFFNESS = 1e-4  # stiffness of the cavity relative to the full extrusion (only so that the system is not singular)


def core(mat: Material, pattern: str, density: float) -> Ortho:
    """Core with infill. ν13 is scaled so that ν13·ν31 stays as for the full material (positive definiteness).
    Infill < HOLLOW = cavity: stiffness × HOLLOW_STIFFNESS, strength infinite (the cavity is not assessed).
    """
    s = solid(mat)
    if density < HOLLOW:
        k, inf = HOLLOW_STIFFNESS, math.inf
        return Ortho(E1=s.E1 * k, E3=s.E3 * k, nu12=s.nu12, nu13=s.nu13, G13=s.G13 * k,
                     Xt=inf, Xc=inf, Zt=inf, Zc=inf, Sxy=inf, Sz=inf)
    r = ratios(pattern, density)
    E1, E3 = s.E1 * r["E_xy"], s.E3 * r["E_z"]
    nu13 = s.nu13 * math.sqrt((r["E_xy"] / r["E_z"]))
    g = math.sqrt(r["E_xy"] * r["E_z"])
    sg = math.sqrt(r["X_xy"] * r["X_z"])
    g12 = INPLANE_SHEAR.get(pattern) if density < FULL else None
    return Ortho(E1=E1, E3=E3, nu12=s.nu12, nu13=nu13, G13=s.G13 * g,
                 Xt=s.Xt * r["X_xy"], Xc=s.Xc * r["X_xy"], Zt=s.Zt * r["X_z"], Zc=s.Zc * r["X_z"],
                 Sxy=s.Sxy * r["X_xy"], Sz=s.Sz * sg, G12v=None if g12 is None else g12 * E1)


def density_ratio(pattern: str, density: float) -> float:
    """Ratio of the density of the core to the full material (= volume fraction of the infill)."""
    return 1.0 if density >= FULL else float(density)


def rotation(build_dir) -> np.ndarray:
    """Matrix R (rows = print axes 1, 2, 3 in global coordinates); axis 3 = build direction."""
    e3 = np.asarray(build_dir, float)
    e3 = e3 / np.linalg.norm(e3)
    ref = np.array([1.0, 0, 0]) if abs(e3[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = ref - (ref @ e3) * e3
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    return np.vstack([e1, e2, e3])


def strain_to_local(eps: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Strain (…, 6) in Voigt notation 11 22 33 12 13 23 (engineering shear) into the axes R."""
    T = np.empty(eps.shape[:-1] + (3, 3))
    T[..., 0, 0], T[..., 1, 1], T[..., 2, 2] = eps[..., 0], eps[..., 1], eps[..., 2]
    T[..., 0, 1] = T[..., 1, 0] = 0.5 * eps[..., 3]
    T[..., 0, 2] = T[..., 2, 0] = 0.5 * eps[..., 4]
    T[..., 1, 2] = T[..., 2, 1] = 0.5 * eps[..., 5]
    L = np.einsum("ai,...ij,bj->...ab", R, T, R)
    return np.stack([L[..., 0, 0], L[..., 1, 1], L[..., 2, 2],
                     2 * L[..., 0, 1], 2 * L[..., 0, 2], 2 * L[..., 1, 2]], axis=-1)


def strain_matrix(R: np.ndarray) -> np.ndarray:
    """T (6×6): ε_local = T·ε_global (Voigt 11 22 33 12 13 23, engineering shear), R = axes (rows)."""
    T = np.empty((6, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = 1.0
        T[:, j] = strain_to_local(e, R)
    return T
