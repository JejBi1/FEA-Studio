"""Library of bar sections (F7): rolled profiles, parametric shapes and own sections.

The library stores ONLY THE DIMENSIONS of the profiles according to the standards; the properties are computed:
- by exact formulas (I rolled and welded doubly symmetric, rectangle, circle, CHS, RHS/SHS tubes) – area
and second moments of area with the roundings (4 − π)·r², Wpl with the contributions of the roundings, I_t by the ArcelorMittal formula,
RHS by EN 10219-2 / EN 10210-2, annex (h_p, A_h, K, C_t);
- by exact integration of the outline (Green's theorem) for shapes without a simple formula (U, L, T, unsymmetric I) – the outline
with roundings (arcs in 32 segments per quarter circle, area error < 1e-5), main axes, Wpl by shifting the plastic neutral axis;
I_t = flanges (Roark) + replacement of the junction calibrated by FE (sectionproperties), I_w and the shear centre thin-walled;
- numerically through `sectionproperties` (FE of the section) for an own section (`custom_section`, `section_from_face`).

The torsional modulus Wt (τ = T/Wt): for the library and parametric shapes nominal (open I_t/t_max, RHS C_t
according to EN 10219-2, solid shapes exact); for an own section the peak of the FE (incl. the concentration in the corners – 30–60 % higher
stress than nominal). I_w and the shear centre of U, L, T are thin-walled (against FE: U ±8 %, shear centre 3–7 %).

Axes (EN 1993): y = main strong, z = weak, origin at the centroid; `alpha` = angle of the main axis y from the axis y0 of the input
(for L profiles). Units mm, mm², mm⁴, mm⁶. Shear areas according to EN 1993-1-1, 6.2.6(3) with η = 1.0 (on the safe side –
η = 1.2 according to EN 1993-1-5 depends on the steel, which the library does not know).

Frame of the input outline (y0, z0): I, RHS, CHS, rectangle, circle – centre; U – back of the web at y0 = 0, flanges towards +y0,
z0 symmetric; L – corner at the origin, arm b along +y0, arm h along +z0; T – bottom of the web z0 = 0, flange on top.
"""
from __future__ import annotations

import copy
import math
import re
from functools import lru_cache

import numpy as np

from .props import SectionProps

ETA = 1.0          # η for the shear area of rolled I (EN 1993-1-1, 6.2.6(3)a) – conservative
NSEG = 32          # number of arc segments per quarter circle along the outline for the integration
FIBER_SEG = 2      # number of arc segments per quarter circle at the points for the stresses (fibres)

SRC_IPE = "EN 10365:2017 (IPE), properties calculated from dimensions"
SRC_HE = "EN 10365:2017 (HE A/B/M, formerly EU 53-62), properties calculated from dimensions"
SRC_UPE = "EN 10365:2017 (UPE, DIN 1026-2), properties calculated from dimensions"
SRC_UPN = "EN 10365:2017 (UPN, DIN 1026-1) – inner flange face slope 8 % / 5 %, properties calculated from dimensions"
SRC_L = "EN 10056-1:2017 (L), properties calculated from dimensions"
SRC_CHS = "EN 10219-2 / EN 10210-2 (CHS), properties calculated from dimensions"
SRC_CF = "EN 10219-2:2019 (cold-formed, r_o = 2t / 2.5t / 3t, r_i = r_o − t), properties per the standard's annex"
SRC_HF = "EN 10210-2:2019 (hot-finished, r_o = 1.5t, r_i = t), properties per the standard's annex"


class SectionError(ValueError):
    """Unknown section or invalid dimensions."""


# ---------------------------------------------------------------- dimension tables (dimensions only!)
# I profiles: h, b, tw, tf, r [mm]
IPE = {
    80: (80, 46, 3.8, 5.2, 5), 100: (100, 55, 4.1, 5.7, 7), 120: (120, 64, 4.4, 6.3, 7),
    140: (140, 73, 4.7, 6.9, 7), 160: (160, 82, 5.0, 7.4, 9), 180: (180, 91, 5.3, 8.0, 9),
    200: (200, 100, 5.6, 8.5, 12), 220: (220, 110, 5.9, 9.2, 12), 240: (240, 120, 6.2, 9.8, 15),
    270: (270, 135, 6.6, 10.2, 15), 300: (300, 150, 7.1, 10.7, 15), 330: (330, 160, 7.5, 11.5, 18),
    360: (360, 170, 8.0, 12.7, 18), 400: (400, 180, 8.6, 13.5, 21), 450: (450, 190, 9.4, 14.6, 21),
    500: (500, 200, 10.2, 16.0, 21), 550: (550, 210, 11.1, 17.2, 24), 600: (600, 220, 12.0, 19.0, 24),
}
HEA = {
    100: (96, 100, 5.0, 8.0, 12), 120: (114, 120, 5.0, 8.0, 12), 140: (133, 140, 5.5, 8.5, 12),
    160: (152, 160, 6.0, 9.0, 15), 180: (171, 180, 6.0, 9.5, 15), 200: (190, 200, 6.5, 10.0, 18),
    220: (210, 220, 7.0, 11.0, 18), 240: (230, 240, 7.5, 12.0, 21), 260: (250, 260, 7.5, 12.5, 24),
    280: (270, 280, 8.0, 13.0, 24), 300: (290, 300, 8.5, 14.0, 27), 320: (310, 300, 9.0, 15.5, 27),
    340: (330, 300, 9.5, 16.5, 27), 360: (350, 300, 10.0, 17.5, 27), 400: (390, 300, 11.0, 19.0, 27),
    450: (440, 300, 11.5, 21.0, 27), 500: (490, 300, 12.0, 23.0, 27), 550: (540, 300, 12.5, 24.0, 27),
    600: (590, 300, 13.0, 25.0, 27), 650: (640, 300, 13.5, 26.0, 27), 700: (690, 300, 14.5, 27.0, 27),
    800: (790, 300, 15.0, 28.0, 30), 900: (890, 300, 16.0, 30.0, 30), 1000: (990, 300, 16.5, 31.0, 30),
}
HEB = {
    100: (100, 100, 6.0, 10.0, 12), 120: (120, 120, 6.5, 11.0, 12), 140: (140, 140, 7.0, 12.0, 12),
    160: (160, 160, 8.0, 13.0, 15), 180: (180, 180, 8.5, 14.0, 15), 200: (200, 200, 9.0, 15.0, 18),
    220: (220, 220, 9.5, 16.0, 18), 240: (240, 240, 10.0, 17.0, 21), 260: (260, 260, 10.0, 17.5, 24),
    280: (280, 280, 10.5, 18.0, 24), 300: (300, 300, 11.0, 19.0, 27), 320: (320, 300, 11.5, 20.5, 27),
    340: (340, 300, 12.0, 21.5, 27), 360: (360, 300, 12.5, 22.5, 27), 400: (400, 300, 13.5, 24.0, 27),
    450: (450, 300, 14.0, 26.0, 27), 500: (500, 300, 14.5, 28.0, 27), 550: (550, 300, 15.0, 29.0, 27),
    600: (600, 300, 15.5, 30.0, 27),
}
HEM = {
    100: (120, 106, 12.0, 20.0, 12), 120: (140, 126, 12.5, 21.0, 12), 140: (160, 146, 13.0, 22.0, 12),
    160: (180, 166, 14.0, 23.0, 15), 180: (200, 186, 14.5, 24.0, 15), 200: (220, 206, 15.0, 25.0, 18),
    220: (240, 226, 15.5, 26.0, 18), 240: (270, 248, 18.0, 32.0, 21), 260: (290, 268, 18.0, 32.5, 24),
    280: (310, 288, 18.5, 33.0, 24), 300: (340, 310, 21.0, 39.0, 27), 320: (359, 309, 21.0, 40.0, 27),
    340: (377, 309, 21.0, 40.0, 27), 360: (395, 308, 21.0, 40.0, 27), 400: (432, 307, 21.0, 40.0, 27),
    450: (478, 307, 21.0, 40.0, 27), 500: (524, 306, 21.0, 40.0, 27), 550: (572, 306, 21.0, 40.0, 27),
    600: (620, 305, 21.0, 40.0, 27),
}
# U with parallel flanges: h, b, tw, tf, r
UPE = {
    80: (80, 50, 4.0, 7.0, 10), 100: (100, 55, 4.5, 7.5, 10), 120: (120, 60, 5.0, 8.0, 12),
    140: (140, 65, 5.0, 9.0, 12), 160: (160, 70, 5.5, 9.5, 12), 180: (180, 75, 5.5, 10.5, 12),
    200: (200, 80, 6.0, 11.0, 13), 220: (220, 85, 6.5, 12.0, 13), 240: (240, 90, 7.0, 12.5, 15),
    270: (270, 95, 7.5, 13.5, 15), 300: (300, 100, 9.5, 15.0, 15), 330: (330, 105, 11.0, 16.0, 18),
    360: (360, 110, 12.0, 17.0, 18), 400: (400, 115, 13.5, 18.0, 18),
}
# U with sloping flanges: h, b, tw, tf (thickness at the middle of the outstand of the flange), r1, r2
UPN = {
    80: (80, 45, 6.0, 8.0, 8.0, 4.0), 100: (100, 50, 6.0, 8.5, 8.5, 4.5), 120: (120, 55, 7.0, 9.0, 9.0, 4.5),
    140: (140, 60, 7.0, 10.0, 10.0, 5.0), 160: (160, 65, 7.5, 10.5, 10.5, 5.5),
    180: (180, 70, 8.0, 11.0, 11.0, 5.5), 200: (200, 75, 8.5, 11.5, 11.5, 6.0),
    220: (220, 80, 9.0, 12.5, 12.5, 6.5), 240: (240, 85, 9.5, 13.0, 13.0, 6.5),
    260: (260, 90, 10.0, 14.0, 14.0, 7.0), 280: (280, 95, 10.0, 15.0, 15.0, 7.5),
    300: (300, 100, 10.0, 16.0, 16.0, 8.0), 320: (320, 100, 14.0, 17.5, 17.5, 8.75),
    350: (350, 100, 14.0, 16.0, 16.0, 8.0), 380: (380, 102, 13.5, 16.0, 16.0, 8.0),
    400: (400, 110, 14.0, 18.0, 18.0, 9.0),
}
# L equal-leg EN 10056-1: (a, t) -> r1 (r2 = r1/2)
L_EQUAL = {
    (20, 3): 3.5, (25, 3): 3.5, (25, 4): 3.5, (30, 3): 5, (30, 4): 5, (35, 4): 5, (40, 4): 6, (40, 5): 6,
    (45, 4.5): 7, (50, 4): 7, (50, 5): 7, (50, 6): 7, (60, 5): 8, (60, 6): 8, (60, 8): 8, (65, 7): 9,
    (70, 6): 9, (70, 7): 9, (75, 6): 9, (75, 8): 9, (80, 8): 10, (80, 10): 10, (90, 7): 11, (90, 8): 11,
    (90, 9): 11, (90, 10): 11, (100, 8): 12, (100, 10): 12, (100, 12): 12, (120, 10): 13, (120, 12): 13,
    (130, 12): 14, (150, 10): 16, (150, 12): 16, (150, 15): 16, (160, 15): 17, (180, 16): 18,
    (180, 18): 18, (200, 16): 18, (200, 20): 18, (200, 24): 18,
}
# L unequal-leg EN 10056-1: (a, b, t) -> r1 (r2 = r1/2); the radii are less certain (effect on A < 0.5 %)
L_UNEQUAL = {
    (30, 20, 3): 4, (40, 25, 4): 4, (45, 30, 4): 4.5, (50, 30, 5): 5, (60, 30, 5): 6, (60, 40, 5): 6,
    (60, 40, 6): 6, (65, 50, 5): 6, (70, 50, 6): 7, (75, 50, 6): 7, (75, 50, 8): 7, (80, 40, 6): 7,
    (80, 40, 8): 7, (80, 60, 7): 8, (100, 50, 6): 9, (100, 50, 8): 9, (100, 65, 7): 10, (100, 75, 8): 10,
    (100, 75, 10): 10, (120, 80, 8): 11, (120, 80, 10): 11, (120, 80, 12): 11, (150, 75, 9): 12,
    (150, 90, 10): 12, (150, 100, 10): 12, (200, 100, 10): 15, (200, 150, 12): 15,
}
CHS_SIZES = {
    21.3: (2.0, 2.3, 2.6, 3.2), 26.9: (2.0, 2.3, 2.6, 3.2), 33.7: (2.0, 2.6, 3.2, 4.0),
    42.4: (2.0, 2.6, 3.2, 4.0), 48.3: (2.0, 2.6, 3.2, 4.0, 5.0), 60.3: (2.0, 2.6, 3.2, 4.0, 5.0),
    76.1: (2.6, 3.2, 4.0, 5.0), 88.9: (3.2, 4.0, 5.0, 6.3), 101.6: (3.2, 4.0, 5.0, 6.3),
    114.3: (3.2, 4.0, 5.0, 6.3), 139.7: (4.0, 5.0, 6.3, 8.0, 10.0), 168.3: (4.0, 5.0, 6.3, 8.0, 10.0),
    193.7: (5.0, 6.3, 8.0, 10.0), 219.1: (5.0, 6.3, 8.0, 10.0, 12.5), 244.5: (6.3, 8.0, 10.0, 12.5),
    273.0: (6.3, 8.0, 10.0, 12.5), 323.9: (6.3, 8.0, 10.0, 12.5), 355.6: (8.0, 10.0, 12.5, 16.0),
    406.4: (8.0, 10.0, 12.5, 16.0), 457.0: (10.0, 12.5, 16.0), 508.0: (10.0, 12.5, 16.0),
}
SHS_SIZES = {
    20: (2.0,), 25: (2.0, 2.5), 30: (2.0, 2.5, 3.0), 40: (2.0, 2.5, 3.0, 4.0), 50: (2.0, 2.5, 3.0, 4.0, 5.0),
    60: (2.0, 2.5, 3.0, 4.0, 5.0), 70: (3.0, 4.0, 5.0), 80: (3.0, 4.0, 5.0, 6.0), 90: (4.0, 5.0, 6.0),
    100: (3.0, 4.0, 5.0, 6.0, 8.0), 120: (4.0, 5.0, 6.0, 8.0, 10.0), 140: (5.0, 6.0, 8.0, 10.0),
    150: (5.0, 6.0, 8.0, 10.0), 160: (5.0, 6.0, 8.0, 10.0), 180: (6.0, 8.0, 10.0),
    200: (6.0, 8.0, 10.0, 12.5), 250: (8.0, 10.0, 12.5), 300: (8.0, 10.0, 12.5), 400: (10.0, 12.5),
}
RHS_SIZES = {
    (40, 20): (2.0, 3.0), (50, 30): (2.0, 3.0), (60, 40): (2.0, 3.0, 4.0), (80, 40): (3.0, 4.0),
    (80, 60): (3.0, 4.0), (100, 50): (3.0, 4.0, 5.0), (100, 60): (4.0, 5.0), (120, 60): (4.0, 5.0),
    (120, 80): (4.0, 5.0, 6.0), (140, 80): (4.0, 5.0, 6.0), (150, 100): (5.0, 6.0, 8.0), (160, 80): (5.0, 6.0),
    (180, 100): (6.0, 8.0), (200, 100): (5.0, 6.0, 8.0, 10.0), (250, 150): (6.0, 8.0, 10.0),
    (300, 200): (8.0, 10.0, 12.5),
}
HF_MIN = 40        # hot-finished tubes (EN 10210) in the library from the size 40 mm


# ---------------------------------------------------------------- number format in the names
def _n(v: float) -> str:
    """Number in the name in Czech format: 60,3 / 5 / 12,5."""
    s = f"{float(v):.4g}"
    return s.replace(".", ",")


# ---------------------------------------------------------------- obrys a integrace
def _rounded(points, radii, nseg: int = NSEG) -> np.ndarray:
    """Polygon with rounded corners: corners `points` (CCW), radius per corner (0 = sharp); convex and concave alike."""
    P = np.asarray(points, float)
    n = len(P)
    out = []
    for i in range(n):
        r = float(radii[i]) if radii is not None else 0.0
        p = P[i]
        if r <= 0:
            out.append(p)
            continue
        d1 = P[i - 1] - p
        d2 = P[(i + 1) % n] - p
        l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
        d1, d2 = d1 / l1, d2 / l2
        cosang = float(np.clip(d1 @ d2, -1.0, 1.0))
        half = math.acos(cosang) / 2.0
        if half < 1e-9 or abs(half - math.pi / 2) < 1e-12:
            out.append(p)
            continue
        dist = r / math.tan(half)
        if dist > min(l1, l2) + 1e-9:
            raise SectionError(f"fillet r = {r} does not fit into the corner")
        t1, t2 = p + d1 * dist, p + d2 * dist
        bis = d1 + d2
        bis /= np.linalg.norm(bis)
        c = p + bis * (r / math.sin(half))
        a1 = math.atan2(t1[1] - c[1], t1[0] - c[0])
        a2 = math.atan2(t2[1] - c[1], t2[0] - c[0])
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi     # the shorter arc
        k = max(1, int(math.ceil(abs(da) / (math.pi / 2) * nseg)))
        for j in range(k + 1):
            a = a1 + da * j / k
            out.append(c + r * np.array([math.cos(a), math.sin(a)]))
    return np.asarray(out)


def _circle(R: float, n: int) -> np.ndarray:
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([R * np.cos(a), R * np.sin(a)])


def _ring_int(P: np.ndarray):
    """Integrals over a polygon (sign by the orientation): A, ∫y, ∫z, ∫y², ∫z², ∫yz."""
    y, z = P[:, 0], P[:, 1]
    y1, z1 = np.roll(y, -1), np.roll(z, -1)
    cr = y * z1 - y1 * z
    A = cr.sum() / 2.0
    Qy = ((y + y1) * cr).sum() / 6.0
    Qz = ((z + z1) * cr).sum() / 6.0
    Iyy = ((y * y + y * y1 + y1 * y1) * cr).sum() / 12.0
    Izz = ((z * z + z * z1 + z1 * z1) * cr).sum() / 12.0
    Iyz = ((y * z1 + 2 * y * z + 2 * y1 * z1 + y1 * z) * cr).sum() / 24.0
    return np.array([A, Qy, Qz, Iyy, Izz, Iyz])


def _orient(rings):
    """Outer outline CCW, holes CW."""
    out = []
    for i, R in enumerate(rings):
        R = np.asarray(R, float)
        if len(R) > 1 and np.allclose(R[0], R[-1]):
            R = R[:-1]
        a = _ring_int(R)[0]
        if (i == 0 and a < 0) or (i > 0 and a > 0):
            R = R[::-1]
        out.append(R)
    return out


def _integrals(rings):
    return sum(_ring_int(R) for R in rings)


def _clip_above(P: np.ndarray, c: float) -> np.ndarray:
    """Sutherland–Hodgman: part of the polygon with z ≥ c (correct for the integrals also for non-convex ones)."""
    z = P[:, 1] - c
    Pn, zn = np.roll(P, -1, axis=0), np.roll(z, -1)
    keep = z >= 0
    cross = keep != (zn >= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(cross, z / (z - zn), 0.0)
    X = P + t[:, None] * (Pn - P)
    # on the edge i: [vertex i, if above], [intersection, if the side changes]
    pts = np.stack([P, X], axis=1).reshape(-1, 2)
    m = np.stack([keep, cross], axis=1).reshape(-1)
    return pts[m]


def _wpl_about_y(rings) -> float:
    """Plastic modulus about the axis y (stress ∝ z) – in the centroidal axes: Wpl = 2·∫_{z≥c} z dA, A(z≥c) = A/2."""
    A = _integrals(rings)[0]
    zs = np.concatenate([R[:, 1] for R in rings])
    lo, hi = float(zs.min()), float(zs.max())

    def above(c):
        tot = np.zeros(6)
        for R in rings:
            Q = _clip_above(R, c)
            if len(Q) >= 3:
                tot += _ring_int(Q)
        return tot

    for _ in range(60):
        c = 0.5 * (lo + hi)
        if above(c)[0] > A / 2:
            lo = c
        else:
            hi = c
        if hi - lo < 1e-9 * (zs.max() - zs.min()):
            break
    return 2.0 * above(0.5 * (lo + hi))[2]


def _hull(P: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain), CCW."""
    pts = sorted(set(map(tuple, np.round(P, 9))))
    if len(pts) <= 2:
        return np.asarray(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def _rot(P, c, alpha):
    """To the main axes: p' = R(α)ᵀ·(p − c), axis y' = (cos α, sin α)."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    Q = np.asarray(P, float) - c
    return np.column_stack([Q[:, 0] * ca + Q[:, 1] * sa, -Q[:, 0] * sa + Q[:, 1] * ca])


def polygon_props(rings):
    """Geometric properties of the outline (outer + holes, any orientation): dict A, c, alpha, Iy, Iz, Wel, Wpl.

    Main axes: y = strong. Wel/Wpl about the main axes. Exact for a polygon.
    """
    rings = _orient(rings)
    A, Qy, Qz, Iyy, Izz, Iyz = _integrals(rings)
    if A <= 0:
        raise SectionError("contour has zero area")
    c = np.array([Qy / A, Qz / A])
    Iz0 = Iyy - A * c[0] ** 2            # ∫y² about the centroid = moment about the axis z0
    Iy0 = Izz - A * c[1] ** 2            # ∫z² = moment k ose y0
    Iyz0 = Iyz - A * c[0] * c[1]
    if abs(Iyz0) < 1e-7 * (Iy0 + Iz0):         # symmetric shape (noise of the rounding of the outline)
        Iyz0 = 0.0
    alpha = 0.5 * math.atan2(-2.0 * Iyz0, Iy0 - Iz0)
    if abs(Iy0 - Iz0) < 1e-9 * (Iy0 + Iz0) and Iyz0 == 0.0:
        alpha = 0.0
    m, R = 0.5 * (Iy0 + Iz0), math.hypot(0.5 * (Iy0 - Iz0), Iyz0)
    Iy, Iz = m + R, m - R
    pr = [_rot(Rg, c, alpha) for Rg in rings]
    allp = np.vstack(pr)
    zmax, ymax = float(np.abs(allp[:, 1]).max()), float(np.abs(allp[:, 0]).max())
    Wpl_y = _wpl_about_y(pr)
    Wpl_z = _wpl_about_y([np.column_stack([-R_[:, 1], R_[:, 0]]) for R_ in pr])  # z' → y', y' → −z'
    return dict(A=A, c=c, alpha=alpha, Iy=Iy, Iz=Iz, Iy0=Iy0, Iz0=Iz0, Iyz0=Iyz0,
                Wel_y=Iy / zmax, Wel_z=Iz / ymax, Wpl_y=Wpl_y, Wpl_z=Wpl_z, rings=pr)


# ---------------------------------------------------------------- outlines of the shapes within the input (y0, z0)
def _rings(shape: str, d: dict, nseg: int = NSEG):
    s = shape
    if s == "I":
        h, b, tw, tf, r = d["h"], d["b"], d["tw"], d["tf"], d.get("r", 0.0)
        b2, tf2 = d.get("b2", b), d.get("tf2", tf)      # bottom flange
        r = r or 0.0
        y1, y2 = b / 2, b2 / 2
        zt, zb = h / 2, -h / 2
        pts = [(-y2, zb), (y2, zb), (y2, zb + tf2), (tw / 2, zb + tf2), (tw / 2, zt - tf), (y1, zt - tf),
               (y1, zt), (-y1, zt), (-y1, zt - tf), (-tw / 2, zt - tf), (-tw / 2, zb + tf2), (-y2, zb + tf2)]
        rad = [0, 0, 0, r, r, 0, 0, 0, 0, r, r, 0]
        return [_rounded(pts, rad, nseg)]
    if s == "U":
        if d.get("sheet"):
            # U from sheet: thickness t, inner bending radius ri
            h, b, t, ri = d["h"], d["b"], d["t"], d.get("ri", d["t"])
            ro = ri + t
            pts = [(0, -h / 2), (b, -h / 2), (b, -h / 2 + t), (t, -h / 2 + t), (t, h / 2 - t), (b, h / 2 - t),
                   (b, h / 2), (0, h / 2)]
            return [_rounded(pts, [ro, 0, 0, ri, ri, 0, 0, ro], nseg)]
        h, b, tw, tf, r = d["h"], d["b"], d["tw"], d["tf"], d.get("r", 0.0)
        slope, r2 = d.get("slope", 0.0), d.get("r2", 0.0)
        if slope:
            # tf = thickness at the place y0 = tf_y (DIN 1026-1: h ≤ 300 at b/2, h > 300 at the middle of the outstand)
            ym = d.get("tf_y", tw + (b - tw) / 2)
            t_tip = tf - slope * (b - ym)
            t_root = tf + slope * (ym - tw)
        else:
            t_tip = t_root = tf
        pts = [(0, -h / 2), (b, -h / 2), (b, -h / 2 + t_tip), (tw, -h / 2 + t_root), (tw, h / 2 - t_root),
               (b, h / 2 - t_tip), (b, h / 2), (0, h / 2)]
        return [_rounded(pts, [0, 0, r2, r, r, r2, 0, 0], nseg)]
    if s == "L":
        h, b, t = d["h"], d["b"], d["t"]
        r1, r2 = d.get("r1", 0.0), d.get("r2", 0.0)
        pts = [(0, 0), (b, 0), (b, t), (t, t), (t, h), (0, h)]
        return [_rounded(pts, [0, 0, r2, r1, r2, 0], nseg)]
    if s == "T":
        h, b, tw, tf, r = d["h"], d["b"], d["tw"], d["tf"], d.get("r", 0.0)
        pts = [(-tw / 2, 0), (tw / 2, 0), (tw / 2, h - tf), (b / 2, h - tf), (b / 2, h), (-b / 2, h),
               (-b / 2, h - tf), (-tw / 2, h - tf)]
        return [_rounded(pts, [0, 0, r, 0, 0, 0, 0, r], nseg)]
    if s == "RECT":
        b, h = d["b"], d["h"]
        return [np.array([(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)])]
    if s == "CIRC":
        return [_circle(d["d"] / 2, 4 * nseg)]
    if s in ("CHS", "TUBE"):
        D, t = d["D"], d["t"]
        return [_circle(D / 2, 4 * nseg), _circle(D / 2 - t, 4 * nseg)[::-1]]
    if s == "RHS":
        h, b, t, ro, ri = d["h"], d["b"], d["t"], d["ro"], d["ri"]
        o = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)]
        i = [(-b / 2 + t, -h / 2 + t), (b / 2 - t, -h / 2 + t), (b / 2 - t, h / 2 - t), (-b / 2 + t, h / 2 - t)]
        return [_rounded(o, [ro] * 4, nseg), _rounded(i, [ri] * 4, nseg)[::-1]]
    if s == "custom":
        return [np.asarray(d["outer"], float)] + [np.asarray(hh, float) for hh in d.get("holes", [])]
    raise SectionError(f"unknown cross-section shape “{shape}”")


def outline(sec: SectionProps, nseg: int = 8):
    """Outline of the section in its main axes (origin at the centroid): [outer CCW, holes CW] – for drawing in the GUI."""
    rings = _orient(_rings(sec.shape, sec.dims, nseg))
    A, Qy, Qz = _integrals(rings)[:3]
    c = np.array([Qy / A, Qz / A])
    return [_rot(R, c, sec.alpha) for R in rings]


def _area(shape: str, dims: dict) -> float:
    return float(_integrals(_orient(_rings(shape, dims)))[0])


def _fibers_from(shape, dims, c, alpha):
    rings = _orient(_rings(shape, dims, FIBER_SEG))
    return [tuple(map(float, p)) for p in _hull(_rot(rings[0], c, alpha))]


# ---------------------------------------------------------------- helper formulas
def _spandrel(r: float):
    """Corner rounding (square r×r without the quarter circle): area, distance of the centroid from the corner vertex, I about its own axis."""
    if r <= 0:
        return 0.0, 0.0, 0.0
    As = (1.0 - math.pi / 4.0) * r * r
    dc = r / (6.0 * (1.0 - math.pi / 4.0))          # from the centre of the circle
    I_own = r ** 4 * (1.0 / 3.0 - math.pi / 16.0) - As * dc * dc
    return As, r - dc, I_own


def _rect_torsion(a: float, b: float, nterms: int = 50):
    """Solid rectangle (Saint-Venant, series): J = β·a·b³, Wt = J/(b·k); a ≥ b."""
    if b > a:
        a, b = b, a
    s1 = s2 = 0.0
    for i in range(nterms):
        n = 2 * i + 1
        x = n * math.pi * a / (2 * b)
        s1 += math.tanh(x) / n ** 5
        s2 += (1.0 / (n * n * math.cosh(x))) if x < 700 else 0.0
    J = a * b ** 3 / 3.0 * (1.0 - 192.0 * b / (math.pi ** 5 * a) * s1)
    k = 1.0 - 8.0 / math.pi ** 2 * s2
    return J, J / (b * k)


def _strip_J(a: float, t: float, free_ends: int) -> float:
    """Strip a × t (Roark): 2 free ends 1/3 − 0.21·t/a·(1 − t⁴/12a⁴), 1 end 1/3 − 0.105·t/a·(1 − t⁴/192a⁴)."""
    if free_ends == 2:
        return a * t ** 3 * (1 / 3 - 0.21 * t / a * (1 - t ** 4 / (12 * a ** 4)))
    if free_ends == 1:
        return a * t ** 3 * (1 / 3 - 0.105 * t / a * (1 - t ** 4 / (192 * a ** 4)))
    return a * t ** 3 / 3


# Replacement of the junction for I_t of L, U, T shapes: j·t⁴ = I_t(FE) − flanges (Roark: 2 / 1 / 0 free ends) for long
# outstands; FE of sectionproperties 3.10 (6-node elements, mesh t²/25), cubic in ρ = r/tf for the table τ = tw/tf,
# linear between τ (scratch/beams_sections/calib.py, fit.py; fit error < 1e-3·t⁴). The Roark α·D⁴ formulas differed from the
# FE by −80 … +40 % of the replacement (I_t L 50×5 +4 %, UPE 80 −5 %).
_J_L = (0.06245, 0.26907, 0.14178, 0.18126)                 # L (the same thickness of the legs), ρ = r1/t
_J_U = {0.3: (0.05085, 0.07923, 0.01534, 0.00464), 0.5: (0.05064, 0.13534, 0.04347, 0.02402),
        0.7: (0.05013, 0.19824, 0.07504, 0.06653), 0.85: (0.05004, 0.24807, 0.09984, 0.11609),
        1.0: (0.05009, 0.30046, 0.1235, 0.18179)}          # corner of a U (flange tf, web tw ≤ tf)
_J_T = {0.3: (0.15895, 0.30089, 0.09559, 0.01976), 0.5: (0.16022, 0.4013, 0.15283, 0.06219),
        0.7: (0.15925, 0.5173, 0.20616, 0.13476), 1.0: (0.1538, 0.71691, 0.27461, 0.30421),
        1.4: (0.1606, 0.96121, 0.39498, 0.61054), 2.0: (0.17489, 1.34549, 0.553, 1.10162)}  # styk T


def _joint(table, tau: float, rho: float) -> float:
    """Replacement of the junction j(τ, ρ) [·t⁴] from the calibration table (ρ limited to 0 … 2.5, τ to the range of the table)."""
    rho = min(max(rho, 0.0), 2.5)
    if isinstance(table, tuple):
        return float(np.polyval(table, rho))
    taus = sorted(table)
    vals = [float(np.polyval(table[t], rho)) for t in taus]
    return float(np.interp(tau, taus, vals))


def _T_joint(tf: float, tw: float, r: float) -> float:
    """Replacement of the junction T (ArcelorMittal / El Darwish & Johnston): α·D⁴, D = ((tf + r)² + tw(r + tw/4))/(2r + tf)."""
    alpha = tw / tf * (0.145 + 0.1 * r / tf)
    D = ((tf + r) ** 2 + tw * (r + tw / 4)) / (2 * r + tf)
    return alpha * D ** 4


def _swap_to_strong(p: SectionProps) -> SectionProps:
    """For doubly symmetric shapes given in (y0, z0): if the strong axis is z0, rotate by 90° (alpha = π/2)."""
    if p.Iz > p.Iy * (1 + 1e-12):
        p.Iy, p.Iz = p.Iz, p.Iy
        p.Wel_y, p.Wel_z = p.Wel_z, p.Wel_y
        p.Wpl_y, p.Wpl_z = p.Wpl_z, p.Wpl_y
        p.Av_y, p.Av_z = p.Av_z, p.Av_y
        p.ys, p.zs = p.zs, -p.ys
        p.fibers = [(z, -y) for (y, z) in p.fibers]
        p.alpha = math.pi / 2
    return p


# ---------------------------------------------------------------- shapes – exact formulas
def i_rolled(name, h, b, tw, tf, r, source="", fab="rolled") -> SectionProps:
    """Doubly symmetric I with the radius r between the web and the flange (rolled; r = 0 = welded)."""
    hw = h - 2 * tf
    if hw <= 0 or tw >= b or min(h, b, tw, tf) <= 0:
        raise SectionError("invalid I section dimensions")
    As, e, I_own = _spandrel(r)
    zf = h / 2 - tf - e                  # centroid of the corner from the centroid of the section
    yf = tw / 2 + e
    A = 2 * b * tf + hw * tw + 4 * As
    Iy = (b * h ** 3 - (b - tw) * hw ** 3) / 12 + 4 * (I_own + As * zf ** 2)
    Iz = (2 * tf * b ** 3 + hw * tw ** 3) / 12 + 4 * (I_own + As * yf ** 2)
    Wpl_y = tw * hw ** 2 / 4 + b * tf * (h - tf) + 4 * As * zf
    Wpl_z = tf * b ** 2 / 2 + hw * tw ** 2 / 4 + 4 * As * yf
    if r > 0:
        # ArcelorMittal: I_t = 2/3(b − 0,63tf)tf³ + 1/3(h − 2tf)tw³ + 2(tw/tf)(0,145 + 0,1r/tf)·D⁴
        J = 2 / 3 * (b - 0.63 * tf) * tf ** 3 + hw * tw ** 3 / 3 + 2 * _T_joint(tf, tw, r)
        Avz = max(A - 2 * b * tf + (tw + 2 * r) * tf, ETA * hw * tw)
    else:
        J = 2 / 3 * (b - 0.63 * tf) * tf ** 3 + hw * tw ** 3 / 3
        Avz = ETA * hw * tw              # welded (EN 1993-1-1, 6.2.6(3)d)
    Iw = tf * b ** 3 * (h - tf) ** 2 / 24  # only the flanges (as the tables of the manufacturers)
    p = SectionProps(name=name, shape="I", dims=dict(h=h, b=b, tw=tw, tf=tf, r=r, fab=fab),
                     A=A, Iy=Iy, Iz=Iz, J=J, Iw=Iw, Wel_y=Iy / (h / 2), Wel_z=Iz / (b / 2),
                     Wpl_y=Wpl_y, Wpl_z=Wpl_z, Av_y=A - hw * tw, Av_z=Avz, Wt=J / max(tf, tw),
                     fibers=[(b / 2, h / 2), (-b / 2, h / 2), (-b / 2, -h / 2), (b / 2, -h / 2)],
                     source=source)
    return p


def rect(b: float, h: float, name: str = "") -> SectionProps:
    """Solid rectangle b (along y0) × h (along z0)."""
    if b <= 0 or h <= 0:
        raise SectionError("invalid rectangle dimensions")
    J, Wt = _rect_torsion(max(b, h), min(b, h))
    A = b * h
    p = SectionProps(name=name or f"Rectangle {_n(b)}×{_n(h)}", shape="RECT", dims=dict(b=b, h=h),
                     A=A, Iy=b * h ** 3 / 12, Iz=h * b ** 3 / 12, J=J, Iw=0.0,
                     Wel_y=b * h * h / 6, Wel_z=h * b * b / 6, Wpl_y=b * h * h / 4, Wpl_z=h * b * b / 4,
                     Av_y=A, Av_z=A, Wt=Wt,
                     fibers=[(b / 2, h / 2), (-b / 2, h / 2), (-b / 2, -h / 2), (b / 2, -h / 2)],
                     source="solid rectangle (Saint-Venant, series)")
    return _swap_to_strong(p)


def _circle_fibers(R: float, n: int = 16):
    return [tuple(map(float, p)) for p in _circle(R, n)]


def circ(d: float, name: str = "") -> SectionProps:
    """Solid circle of diameter d."""
    if d <= 0:
        raise SectionError("invalid diameter")
    A = math.pi * d * d / 4
    I = math.pi * d ** 4 / 64
    return SectionProps(name=name or f"Circle Ø{_n(d)}", shape="CIRC", dims=dict(d=d), A=A, Iy=I, Iz=I,
                        J=2 * I, Iw=0.0, Wel_y=I / (d / 2), Wel_z=I / (d / 2), Wpl_y=d ** 3 / 6,
                        Wpl_z=d ** 3 / 6, Av_y=A, Av_z=A, Wt=math.pi * d ** 3 / 16,
                        fibers=_circle_fibers(d / 2), source="solid circle")


def tube(D: float, t: float, name: str = "", shape: str = "TUBE", source: str = "tube (exact formulas)",
         fab: str = "") -> SectionProps:
    """Circular tube D × t."""
    if D <= 0 or t <= 0 or 2 * t > D:
        raise SectionError("invalid tube dimensions")
    d = D - 2 * t
    A = math.pi * (D * D - d * d) / 4
    I = math.pi * (D ** 4 - d ** 4) / 64
    dims = dict(D=D, t=t)
    if fab:
        dims["fab"] = fab
    return SectionProps(name=name or f"Tube {_n(D)}×{_n(t)}", shape=shape, dims=dims, A=A, Iy=I, Iz=I,
                        J=2 * I, Iw=0.0, Wel_y=I / (D / 2), Wel_z=I / (D / 2), Wpl_y=(D ** 3 - d ** 3) / 6,
                        Wpl_z=(D ** 3 - d ** 3) / 6, Av_y=2 * A / math.pi, Av_z=2 * A / math.pi,
                        Wt=2 * I / (D / 2), fibers=_circle_fibers(D / 2), source=source)


def rhs_radii(t: float, fab: str = "cold"):
    """Corner radii of tubes: EN 10219-2 (cold) r_o = 2t (t ≤ 6), 2.5t (≤ 10), 3t; EN 10210-2 (hot) 1.5t / t."""
    if fab == "hot":
        return 1.5 * t, t
    ro = 2.0 * t if t <= 6 else (2.5 * t if t <= 10 else 3.0 * t)
    return ro, ro - t


def rhs(h: float, b: float, t: float, ro: float | None = None, ri: float | None = None, name: str = "",
        fab: str = "cold", source: str = "") -> SectionProps:
    """Rectangular / square tube h (along z0) × b (along y0) × t, corners r_o / r_i (default according to EN 10219)."""
    if min(h, b, t) <= 0 or 2 * t >= min(h, b):
        raise SectionError("invalid rectangular tube dimensions")
    if ro is None:
        ro, ri0 = rhs_radii(t, fab)
        ri = ri0 if ri is None else ri
    if ri is None:
        ri = max(ro - t, 0.0)
    if ro > min(h, b) / 2 or ri > min(h, b) / 2 - t:
        raise SectionError("tube corner radius is greater than half the side")
    Ao, eo, Io = _spandrel(ro)
    Ai, ei, Ii = _spandrel(ri)
    hi, bi = h - 2 * t, b - 2 * t
    A = 2 * t * (b + h - 2 * t) - (4 - math.pi) * (ro * ro - ri * ri)
    # moment: full rectangles − corners of the outer corners + corners of the inner corners (the hole is smaller)
    Iy = (b * h ** 3 - bi * hi ** 3) / 12 - 4 * (Io + Ao * (h / 2 - eo) ** 2) + 4 * (Ii + Ai * (hi / 2 - ei) ** 2)
    Iz = (h * b ** 3 - hi * bi ** 3) / 12 - 4 * (Io + Ao * (b / 2 - eo) ** 2) + 4 * (Ii + Ai * (bi / 2 - ei) ** 2)
    Wpl_y = (b * h * h / 4 - 4 * Ao * (h / 2 - eo)) - (bi * hi * hi / 4 - 4 * Ai * (hi / 2 - ei))
    Wpl_z = (h * b * b / 4 - 4 * Ao * (b / 2 - eo)) - (hi * bi * bi / 4 - 4 * Ai * (bi / 2 - ei))
    # EN 10219-2 / 10210-2, annex: I_t = t³h_p/3 + 2K·A_h, C_t = I_t/(t + K/t)
    Rc = (ro + ri) / 2
    hp = 2 * ((b - t) + (h - t)) - 2 * Rc * (4 - math.pi)
    Ah = (b - t) * (h - t) - Rc * Rc * (4 - math.pi)
    K = 2 * Ah * t / hp
    J = t ** 3 * hp / 3 + 2 * K * Ah
    Wt = J / (t + K / t)
    sq = abs(h - b) < 1e-9
    nm = name or (f"SHS {_n(h)}×{_n(t)}" if sq else f"RHS {_n(h)}×{_n(b)}×{_n(t)}")
    dims = dict(h=h, b=b, t=t, ro=ro, ri=ri, fab=fab)
    p = SectionProps(name=nm, shape="RHS", dims=dims, A=A, Iy=Iy, Iz=Iz, J=J, Iw=0.0,
                     Wel_y=Iy / (h / 2), Wel_z=Iz / (b / 2), Wpl_y=Wpl_y, Wpl_z=Wpl_z,
                     Av_z=A * h / (b + h), Av_y=A * b / (b + h), Wt=Wt,
                     source=source or (SRC_HF if fab == "hot" else SRC_CF if fab == "cold" else
                                       "rectangular tube (EN 10219-2 formulas, annex)"))
    p.fibers = _fibers_from("RHS", dims, np.zeros(2), 0.0)
    return _swap_to_strong(p)


# ---------------------------------------------------------------- tvary – integrace obrysu
def _from_outline(name, shape, dims, J, Iw, Av_y0, Av_z0, sc0, source, t_max) -> SectionProps:
    """Properties from the outline; Av_y0/Av_z0 and the shear centre sc0 in the axes of the input (y0, z0) are converted to the main axes."""
    g = polygon_props(_rings(shape, dims))
    c, al = g["c"], g["alpha"]
    sc = _rot(np.array([sc0], float), c, al)[0]
    if abs(al) < 1e-9:
        Avy, Avz = Av_y0, Av_z0
    else:
        # shear areas of L / unsymmetric shapes are approximate – in the main axes we take the smaller
        Avy = Avz = min(Av_y0, Av_z0)
    return SectionProps(name=name, shape=shape, dims=dims, A=g["A"], Iy=g["Iy"], Iz=g["Iz"], J=J, Iw=Iw,
                        Wel_y=g["Wel_y"], Wel_z=g["Wel_z"], Wpl_y=g["Wpl_y"], Wpl_z=g["Wpl_z"],
                        Av_y=Avy, Av_z=Avz, Wt=J / t_max if t_max > 0 else 0.0, alpha=al,
                        ys=float(sc[0]), zs=float(sc[1]), fibers=_fibers_from(shape, dims, c, al),
                        source=source)


def _channel_thin(hp, bp, tw, tfun, n: int = 2000):
    """Thin-walled U with a variable flange thickness t(y) (y from the axis of the web): shear centre e behind the axis of the web and
    I_w (sectorial coordinate about the shear centre: web ω = e·z, flange ω = h'/2·(e − y)).
    """
    dy = bp / n
    y = (np.arange(n) + 0.5) * dy
    t = tfun(y)
    Iy = tw * hp ** 3 / 12 + 2 * (hp / 2) ** 2 * t.sum() * dy
    e = hp ** 2 / (2 * Iy) * (t * y).sum() * dy
    Iw = e * e * tw * hp ** 3 / 12 + 2 * (hp / 2) ** 2 * ((e - y) ** 2 * t).sum() * dy
    return float(e), float(Iw)


def channel(name, h, b, tw, tf, r, slope=0.0, r2=0.0, tf_y=None, source="", fab="rolled") -> SectionProps:
    """Rolled U (UPE: parallel flanges; UPN: slope `slope` of the inner face, tf at y0 = tf_y from the back)."""
    if min(h, b, tw, tf) <= 0 or 2 * tf >= h or tw >= b:
        raise SectionError("invalid U section dimensions")
    dims = dict(h=h, b=b, tw=tw, tf=tf, r=r, fab=fab)
    if slope:
        dims.update(slope=slope, r2=r2, tf_y=tw + (b - tw) / 2 if tf_y is None else tf_y)
    hw = h - 2 * tf
    if slope:
        # sloping flange: a strip of thickness at the middle of the outstand t_m + correction to the mean t³ (linear thickness),
        # web between the roots of the flanges, replacement of the corner with the flange thickness at the web t_r
        c = b - tw
        tm = tf - slope * (tw + c / 2 - dims["tf_y"])
        tr = tm + slope * c / 2
        t2 = tm - slope * c / 2
        k3 = (tr ** 4 - t2 ** 4) / (4 * (tr - t2)) / tm ** 3
        J = (2 * (_strip_J(b, tm, 2) + (k3 - 1) * c * tm ** 3 / 3) + (h - 2 * tr) * tw ** 3 / 3
             + 2 * _joint(_J_U, tw / tr, r / tr) * tr ** 4)
    else:
        # I_t: flange b × tf (2 free ends), web between the flanges, 2 corners (calibrated replacement of the junction)
        J = 2 * _strip_J(b, tf, 2) + hw * tw ** 3 / 3 + 2 * _joint(_J_U, tw / tf, r / tf) * tf ** 4
    bp, hp = b - tw / 2, h - tf
    if slope:
        ty = dims["tf_y"]
        e, Iw = _channel_thin(hp, bp, tw, lambda y: tf - slope * (np.maximum(y + tw / 2, tw) - ty))
    else:
        e = 3 * bp * bp * tf / (6 * bp * tf + hp * tw)      # shear centre behind the axis of the web
        Iw = tf * bp ** 3 * hp ** 2 / 12 * (3 * bp * tf + 2 * hp * tw) / (6 * bp * tf + hp * tw)
    g_A = _area("U", dims)
    Avz = g_A - 2 * b * tf + (tw + r) * tf                    # EN 1993-1-1, 6.2.6(3)b
    Avy = g_A - hw * tw                                       # 6.2.6(3)e
    return _from_outline(name, "U", dims, J, Iw, Avy, Avz, (tw / 2 - e, 0.0), source, max(tf, tw))


def angle(name, h, b, t, r1=0.0, r2=0.0, source="", fab="rolled") -> SectionProps:
    """L profile: arm h along z0, arm b along y0, thickness t; r1 root, r2 ends of the arms."""
    if min(h, b, t) <= 0 or t >= min(h, b):
        raise SectionError("invalid L section dimensions")
    dims = dict(h=h, b=b, t=t, r1=r1, r2=r2, fab=fab)
    # arm b whole (2 free ends), arm h from the inner face of the arm b (1 free end), replacement of the corner
    J = _strip_J(b, t, 2) + _strip_J(h - t, t, 1) + _joint(_J_L, 1.0, r1 / t) * t ** 4
    bp, hp = b - t / 2, h - t / 2
    Iw = t ** 3 / 36 * (bp ** 3 + hp ** 3)                    # small (Kollbrunner), usually neglected
    # shear areas approximately the area of the arm (in the main axes of L the smaller of them, see _from_outline)
    return _from_outline(name, "L", dims, J, Iw, b * t, h * t, (t / 2, t / 2), source, t)


def tee(h, b, tw, tf, r=0.0, name="", source="", fab="welded") -> SectionProps:
    """T profile: height h, flange b × tf on top, web tw, radius r."""
    if min(h, b, tw, tf) <= 0 or tf >= h or tw >= b:
        raise SectionError("invalid T section dimensions")
    dims = dict(h=h, b=b, tw=tw, tf=tf, r=r, fab=fab)
    c = h - tf
    J = _strip_J(b, tf, 2) + _strip_J(c, tw, 1) + _joint(_J_T, tw / tf, r / tf) * tf ** 4
    hp = h - tf / 2
    Iw = tf ** 3 * b ** 3 / 144 + tw ** 3 * hp ** 3 / 36
    g_A = _area("T", dims)
    if fab == "rolled":
        Avz = 0.9 * (g_A - b * tf)                            # EN 1993-1-1, 6.2.6(3)c (AC)
    else:
        Avz = ETA * c * tw
    return _from_outline(name or f"T {_n(h)}×{_n(b)}×{_n(tw)}×{_n(tf)}", "T", dims, J, Iw, b * tf, Avz,
                         (0.0, h - tf / 2), source or "T section (contour integration, I_t Roark)",
                         max(tf, tw))


def welded_i(h, b, tw, tf, b2=None, tf2=None, name="", source="") -> SectionProps:
    """Welded I (without roundings), also unsymmetric: upper flange b × tf, lower b2 × tf2."""
    b2 = b if b2 is None else b2
    tf2 = tf if tf2 is None else tf2
    if abs(b2 - b) < 1e-12 and abs(tf2 - tf) < 1e-12:
        return i_rolled(name or f"Welded I {_n(h)}×{_n(b)}×{_n(tw)}×{_n(tf)}", h, b, tw, tf, 0.0,
                        source or "welded I (exact formulas)", fab="welded")
    if min(h, b, b2, tw, tf, tf2) <= 0 or tf + tf2 >= h or tw >= min(b, b2):
        raise SectionError("invalid welded I dimensions")
    dims = dict(h=h, b=b, tw=tw, tf=tf, b2=b2, tf2=tf2, r=0.0, fab="welded")
    hw = h - tf - tf2
    J = (_strip_J(b, tf, 2) + _strip_J(b2, tf2, 2) + hw * tw ** 3 / 3)
    hp = h - tf / 2 - tf2 / 2                                  # distance of the centroids of the flanges
    I1, I2 = tf * b ** 3 / 12, tf2 * b2 ** 3 / 12
    Iw = hp * hp * I1 * I2 / (I1 + I2)
    z_top = h / 2 - tf / 2
    zs0 = z_top - hp * I2 / (I1 + I2)                          # shear centre closer to the stiffer flange
    A = b * tf + b2 * tf2 + hw * tw
    return _from_outline(name or f"Welded I {_n(h)}×{_n(b)}/{_n(b2)}×{_n(tw)}×{_n(tf)}/{_n(tf2)}", "I", dims,
                         J, Iw, A - hw * tw, ETA * hw * tw, (0.0, zs0),
                         source or "welded monosymmetric I (contour integration)", max(tf, tf2, tw))


def sheet_channel(h, b, t, ri=None, name="") -> SectionProps:
    """U (C) from sheet of thickness t, inner bending radius ri (default t); thin-walled I_t, I_w, shear centre."""
    ri = t if ri is None else ri
    if min(h, b, t) <= 0 or 2 * (t + ri) >= h or t + ri >= b:
        raise SectionError("invalid dimensions of U from sheet")
    dims = dict(h=h, b=b, t=t, ri=ri, sheet=True, fab="cold")
    rm = ri + t / 2
    # mid-line: two flanges, a web, two arcs
    Ls = 2 * (b - t / 2 - rm) + (h - t - 2 * rm) + 2 * (math.pi / 2 * rm)
    J = Ls * t ** 3 / 3
    bp, hp = b - t / 2, h - t
    e = 3 * bp * bp / (6 * bp + hp)
    Iw = t * bp ** 3 * hp ** 2 / 12 * (3 * bp + 2 * hp) / (6 * bp + hp)
    hw = h - 2 * t
    g_A = _area("U", dims)
    return _from_outline(name or f"U sheet {_n(h)}×{_n(b)}×{_n(t)}", "U", dims, J, Iw, g_A - hw * t,
                         ETA * hw * t, (t / 2 - e, 0.0), "U from sheet (contour integration, thin-walled I_t, I_w)", t)


# ---------------------------------------------------------------- library: names, families, search
def _std_name(fam: str, key) -> str:
    if fam in ("IPE", "HEA", "HEB", "HEM", "UPE", "UPN"):
        return f"{fam} {key}"
    if fam == "L":
        return f"L {_n(key[0])}×{_n(key[1])}"
    if fam == "LU":
        return f"L {_n(key[0])}×{_n(key[1])}×{_n(key[2])}"
    if fam == "CHS":
        return f"CHS {_n(key[0])}×{_n(key[1])}"
    if fam in ("SHS", "SHS_HF"):
        return f"SHS {_n(key[0])}×{_n(key[1])}" + (" HF" if fam == "SHS_HF" else "")
    if fam in ("RHS", "RHS_HF"):
        return f"RHS {_n(key[0])}×{_n(key[1])}×{_n(key[2])}" + (" HF" if fam == "RHS_HF" else "")
    raise KeyError(fam)


FAMILY_TITLES = {
    "IPE": "IPE", "HEA": "HEA", "HEB": "HEB", "HEM": "HEM", "UPE": "UPE", "UPN": "UPN",
    "L": "L equal", "LU": "L unequal", "CHS": "CHS (circular tubes)",
    "SHS": "SHS (EN 10219, cold-formed)", "RHS": "RHS (EN 10219, cold-formed)",
    "SHS_HF": "SHS (EN 10210, hot-finished)", "RHS_HF": "RHS (EN 10210, hot-finished)",
}


def _family_keys(fam: str):
    if fam in ("IPE", "HEA", "HEB", "HEM", "UPE", "UPN"):
        return list(globals()[fam].keys())
    if fam == "L":
        return list(L_EQUAL.keys())
    if fam == "LU":
        return list(L_UNEQUAL.keys())
    if fam == "CHS":
        return [(D, t) for D, ts in CHS_SIZES.items() for t in ts]
    if fam in ("SHS", "SHS_HF"):
        return [(a, t) for a, ts in SHS_SIZES.items() for t in ts if fam == "SHS" or a >= HF_MIN]
    if fam in ("RHS", "RHS_HF"):
        return [(h, b, t) for (h, b), ts in RHS_SIZES.items() for t in ts if fam == "RHS" or b >= HF_MIN]
    raise KeyError(fam)


def families() -> dict:
    """Series of the library: {name of the series: [names of the profiles]}."""
    return {FAMILY_TITLES[f]: [_std_name(f, k) for k in _family_keys(f)] for f in FAMILY_TITLES}


def _num(s: str) -> float:
    return float(s.replace(",", "."))


_NUM = r"(\d+(?:\.\d+)?)"


def _parse(text: str):
    """Text → (family, key, in_library). SectionError exception if the notation is not understood."""
    s = str(text).strip().upper()
    s = s.replace(",", ".").replace("×", "X").replace("*", "X").replace("Ø", "").replace("Ø", "")
    s = re.sub(r"\s+", "", s)
    fab = None
    m = re.search(r"(EN)?10210(-2)?|HF$|\(HF\)", s)
    if m:
        fab = "hot"
        s = s[:m.start()] + s[m.end():]
    m = re.search(r"(EN)?10219(-2)?|CF$|\(CF\)", s)
    if m:
        fab = "cold"
        s = s[:m.start()] + s[m.end():]
    s = s.strip("-/()")
    # I a U profily
    m = re.fullmatch(r"(IPE|HEA|HEB|HEM|UPE|UPN|U)" + _NUM, s) or None
    if m:
        fam, v = m.group(1), _num(m.group(2))
        fam = "UPN" if fam == "U" else fam
        key = int(v) if v == int(v) else v
        return fam, key, key in globals()[fam]
    m = re.fullmatch(r"HE" + _NUM + r"(A|B|M)", s) or re.fullmatch(r"HE(A|B|M)" + _NUM, s)
    if m:
        g = m.groups()
        letter, v = (g[1], g[0]) if g[1] in "ABM" else (g[0], g[1])
        fam = "HE" + letter
        key = int(_num(v))
        return fam, key, key in globals()[fam]
    # L
    m = re.fullmatch(r"L" + _NUM + "X" + _NUM + "(?:X" + _NUM + ")?", s)
    if m:
        a, b, c = m.groups()
        a, b = _num(a), _num(b)
        a, b = (int(x) if x == int(x) else x for x in (a, b))
        if c is None:
            return "L", (a, b), (a, b) in L_EQUAL
        c = _num(c)
        c = int(c) if c == int(c) else c
        if a == b:
            return "L", (a, c), (a, c) in L_EQUAL
        a, b = max(a, b), min(a, b)
        return "LU", (a, b, c), (a, b, c) in L_UNEQUAL
    # circular tubes
    m = re.fullmatch(r"(?:CHS|TR|TRUBKA|ROURA|O)" + _NUM + "X" + _NUM, s)
    if m:
        D, t = _num(m.group(1)), _num(m.group(2))
        D = _chs_D(D)
        return "CHS", (D, t), t in CHS_SIZES.get(D, ())
    # square and rectangular tubes (Czech "jekl", "TR 4HR", "TR OBD")
    m = re.fullmatch(r"(?:SHS|RHS|JEKL|TR4HR|TROBD|4HR|OBD|TRJEKL)" + _NUM + "X" + _NUM + "(?:X" + _NUM + ")?", s)
    if m:
        a, b, c = m.groups()
        a, b = _num(a), _num(b)
        if c is None:
            sq, h, bb, t = True, a, a, b
        else:
            h, bb, t = max(a, b), min(a, b), _num(c)
            sq = h == bb
        hot = fab == "hot"
        h, bb = (int(x) if x == int(x) else x for x in (h, bb))
        if sq:
            fam = "SHS_HF" if hot else "SHS"
            key = (h, t)
            inlib = t in SHS_SIZES.get(h, ()) and (not hot or h >= HF_MIN)
        else:
            fam = "RHS_HF" if hot else "RHS"
            key = (h, bb, t)
            inlib = t in RHS_SIZES.get((h, bb), ()) and (not hot or bb >= HF_MIN)
        return fam, key, inlib
    raise SectionError(f"unknown cross-section “{text}”")


def _chs_D(D: float) -> float:
    """Notation 273 / 457 / 508 → key of the table (273.0)."""
    for k in CHS_SIZES:
        if abs(k - D) < 1e-6:
            return k
    return D


def _build(fam: str, key, inlib: bool) -> SectionProps:
    name = _std_name(fam, key)
    if fam in ("IPE", "HEA", "HEB", "HEM"):
        if not inlib:
            raise SectionError(f"{name} is not in the library ({fam} {min(globals()[fam])}–{max(globals()[fam])})")
        h, b, tw, tf, r = globals()[fam][key]
        return i_rolled(name, h, b, tw, tf, r, SRC_IPE if fam == "IPE" else SRC_HE)
    if fam == "UPE":
        if not inlib:
            raise SectionError(f"{name} is not in the library")
        h, b, tw, tf, r = UPE[key]
        return channel(name, h, b, tw, tf, r, source=SRC_UPE)
    if fam == "UPN":
        if not inlib:
            raise SectionError(f"{name} is not in the library")
        h, b, tw, tf, r1, r2 = UPN[key]
        # DIN 1026-1: h ≤ 300 slope 8 %, tf at b/2 from the back; h > 300 slope 5 %, tf at the middle of the outstand (b − tw)/2
        if h <= 300:
            return channel(name, h, b, tw, tf, r1, slope=0.08, r2=r2, tf_y=b / 2, source=SRC_UPN)
        return channel(name, h, b, tw, tf, r1, slope=0.05, r2=r2, tf_y=tw + (b - tw) / 2, source=SRC_UPN)
    if fam in ("L", "LU"):
        if not inlib:
            raise SectionError(f"{name} is not in the EN 10056-1 library (fillet radii unknown) – use parametric('L', h=, b=, t=, r1=, r2=)")
        if fam == "L":
            a, t = key
            r1 = L_EQUAL[key]
            return angle(name, a, a, t, r1, r1 / 2, SRC_L)
        a, b, t = key
        r1 = L_UNEQUAL[key]
        return angle(name, a, b, t, r1, r1 / 2, SRC_L + "; radii r1, r2 of unequal angles are less certain")
    if fam == "CHS":
        D, t = key
        src = SRC_CHS if inlib else "circular tube (dimension outside the library range)"
        return tube(D, t, name=name, shape="CHS", source=src, fab="hot/cold")
    if fam in ("SHS", "SHS_HF", "RHS", "RHS_HF"):
        hot = fam.endswith("_HF")
        h, b, t = (key[0], key[0], key[1]) if fam.startswith("SHS") else key
        fab = "hot" if hot else "cold"
        src = (SRC_HF if hot else SRC_CF) + ("" if inlib else "; dimension outside the library range")
        return rhs(h, b, t, name=name, fab=fab, source=src)
    raise SectionError(f"unknown series {fam}")


@lru_cache(maxsize=1024)
def _get_cached(fam, key, inlib) -> SectionProps:
    return _build(fam, key, inlib)


def get(name: str) -> SectionProps:
    """Section from the library by name ("IPE 200", "HE 200 B", "L50x5", "TR 60,3×4", "JEKL 100x100x5" …).

    Tubes (CHS, SHS, RHS) outside the series of the library are computed from the dimensions (a note in `source`). Returns a copy.
    """
    fam, key, inlib = _parse(name)
    return copy.deepcopy(_get_cached(fam, key, inlib))


def canonical(name: str) -> str:
    """Normalised name of the profile ("IPE200" → "IPE 200")."""
    fam, key, _ = _parse(name)
    return _std_name(fam, key)


def search(text: str, limit: int = 50) -> list[str]:
    """Names of the library profiles matching the text (first an exact match of the notation, then a substring)."""
    out = []
    try:
        fam, key, inlib = _parse(text)
        if inlib:
            out.append(_std_name(fam, key))
    except SectionError:
        pass
    norm = lambda s: re.sub(r"\s+", "", s.upper().replace(",", ".").replace("×", "X"))  # noqa: E731
    q = norm(text)
    q2 = q
    m = re.fullmatch(r"HE(\d*)(A|B|M)", q)
    if m:
        q2 = f"HE{m.group(2)}{m.group(1)}"
    for f in FAMILY_TITLES:
        for k in _family_keys(f):
            n = _std_name(f, k)
            if n not in out and (q in norm(n) or q2 in norm(n)):
                out.append(n)
                if len(out) >= limit:
                    return out
    return out


def parametric(shape: str, **d) -> SectionProps:
    """Parametric section. Shapes and dimensions [mm]:

    RECT (b, h) | CIRC (d) | TUBE (D, t) | RHS (h, b, t, ro=None, ri=None, fab="cold"|"hot") |
    I (h, b, tw, tf, r=0, b2=None, tf2=None – welded / unsymmetric) | T (h, b, tw, tf, r=0) |
    U (h, b, t, ri=None – from sheet) or U (h, b, tw, tf, r=0) | L (h, b, t, r1=0, r2=0).
    """
    s = shape.strip().upper()
    nm = d.pop("name", "")
    if s in ("RECT", "OBDELNIK"):
        return rect(d["b"], d["h"], name=nm)
    if s in ("CIRC", "KRUH"):
        return circ(d["d"], name=nm)
    if s in ("TUBE", "CHS", "TRUBKA"):
        return tube(d["D"], d["t"], name=nm)
    if s in ("RHS", "SHS"):
        b = d.get("b", d["h"])
        return rhs(d["h"], b, d["t"], d.get("ro"), d.get("ri"), name=nm, fab=d.get("fab", "cold"))
    if s == "I":
        r = d.get("r", 0.0) or 0.0
        if r > 0 and d.get("b2") is None and d.get("tf2") is None:
            return i_rolled(nm or f"I {_n(d['h'])}×{_n(d['b'])}×{_n(d['tw'])}×{_n(d['tf'])}", d["h"], d["b"],
                            d["tw"], d["tf"], r, "I with fillets (exact formulas)")
        if r > 0:
            raise SectionError("asymmetric I with fillets is not supported (use custom_section)")
        return welded_i(d["h"], d["b"], d["tw"], d["tf"], d.get("b2"), d.get("tf2"), name=nm)
    if s == "T":
        return tee(d["h"], d["b"], d["tw"], d["tf"], d.get("r", 0.0), name=nm,
                   fab="rolled" if d.get("r", 0.0) else "welded")
    if s in ("U", "C"):
        if "t" in d:
            return sheet_channel(d["h"], d["b"], d["t"], d.get("ri"), name=nm)
        return channel(nm or f"U {_n(d['h'])}×{_n(d['b'])}×{_n(d['tw'])}×{_n(d['tf'])}", d["h"], d["b"],
                       d["tw"], d["tf"], d.get("r", 0.0), source="U section (contour integration)", fab="welded")
    if s == "L":
        h = d["h"]
        return angle(nm or f"L {_n(h)}×{_n(d.get('b', h))}×{_n(d['t'])}", h, d.get("b", h), d["t"],
                     d.get("r1", 0.0), d.get("r2", 0.0), source="L section (contour integration)")
    raise SectionError(f"unknown parametric shape “{shape}”")


# ---------------------------------------------------------------- own section (sectionproperties)
def custom_section(name: str, outer, holes=(), mesh_size: float | None = None) -> SectionProps:
    """Own section from the outline [(y, z), …] and holes – FE of the section `sectionproperties` (I_t, I_w, shear centre,
    shear areas, plastic moduli). `mesh_size` = max. element area [mm²] (default A/400, ≤ t²/2).

    Importing sectionproperties takes ~3 s, the calculation 1–3 s → only on request, not in `get()`.
    """
    from shapely.geometry import Polygon as ShPolygon
    from .section_mesh import install
    install()                     # mesh of the section from Gmsh instead of Triangle (licence) – before the first import of sectionproperties
    from sectionproperties.pre.geometry import Geometry
    from sectionproperties.analysis import Section

    outer = [tuple(map(float, p)) for p in outer]
    holes = [[tuple(map(float, p)) for p in h] for h in holes]
    poly = ShPolygon(outer, holes)
    if not poly.is_valid or poly.area <= 0:
        raise SectionError(f"cross-section contour “{name}” is not a valid polygon")
    geom = Geometry(poly)
    # default element size: A/400, for thin walls at most t²/2 (t ≈ 2A/perimeter); 6-node elements → I_t
    # with refinement it changes < 0.1 % (verified on L 50×5: A/150 … A/1500)
    t_est = 2.0 * poly.area / poly.length
    ms = mesh_size or min(poly.area / 400.0, 0.5 * t_est * t_est)
    geom.create_mesh(mesh_sizes=ms)
    sec = Section(geom)
    sec.calculate_geometric_properties()
    sec.calculate_warping_properties()
    sec.calculate_plastic_properties()
    A = sec.get_area()
    cx, cy = sec.get_c()
    I11, I22 = sec.get_ip()
    alpha = math.radians(sec.get_phi())
    A11, A22 = sec.get_as_p()
    S11, S22 = sec.get_sp()
    # sectionproperties: axis 11 at the angle phi from x; our y = strong. If the strong axis is 22, swap
    if I22 > I11:
        I11, I22, A11, A22, S11, S22 = I22, I11, A22, A11, S22, S11
        alpha += math.pi / 2
    alpha = (alpha + math.pi / 2) % math.pi - math.pi / 2     # the direction of the axis is unsigned → (−π/2; π/2]
    ixx, iyy, ixy = sec.get_ic()
    if abs(ixy) < 1e-5 * (ixx + iyy):                       # symmetric section: noise of the angle → exactly 0 / 90°
        alpha = 0.0 if ixx >= iyy * (1 - 1e-7) else math.pi / 2
    J = sec.get_j()
    Iw = sec.get_gamma()
    xs, ys_ = sec.get_sc()
    sc = _rot(np.array([[xs, ys_]]), np.array([cx, cy]), alpha)[0]
    rings = [np.asarray(outer)] + [np.asarray(h) for h in holes]
    pr = [_rot(R, np.array([cx, cy]), alpha) for R in _orient(rings)]
    allp = np.vstack(pr)
    zmax, ymax = float(np.abs(allp[:, 1]).max()), float(np.abs(allp[:, 0]).max())
    # τ_max from the unit torque → Wt = 1/τ_max
    wt_note = ""
    try:
        st = sec.calculate_stress(mzz=1.0).get_stress()[0]
        tau = float(np.max(np.abs(st["sig_zxy_mzz"])))
        Wt = 1.0 / tau if tau > 0 else 0.0
    except Exception as e:       # noqa: BLE001 – Wt = 0 would silently drop the torsional shear from the checks
        Wt = 0.0
        wt_note = f"; torsional stress not computed ({type(e).__name__})"
    if Wt <= 0 and J > 0:
        # conservative estimate τ_max ≈ T·r_max/J (exact for a circle, an upper bound for thin open sections)
        Wt = J / max(float(np.hypot(allp[:, 0], allp[:, 1]).max()), 1e-12)
        wt_note = (wt_note or "; torsional stress not computed") + " – W_t = J/r_max (conservative estimate)"
    # A_s from sectionproperties: A_s11 = shear area for a force along the axis 11 (= y)
    return SectionProps(name=name, shape="custom", dims=dict(outer=[list(p) for p in outer],
                                                             holes=[[list(p) for p in h] for h in holes]),
                        A=A, Iy=I11, Iz=I22, J=J, Iw=Iw, Wel_y=I11 / zmax, Wel_z=I22 / ymax,
                        Wpl_y=S11, Wpl_z=S22, Av_y=A11, Av_z=A22, Wt=Wt, alpha=alpha,
                        ys=float(sc[0]), zs=float(sc[1]),
                        fibers=[tuple(map(float, p)) for p in _hull(pr[0])],
                        source=f"custom cross-section – sectionproperties + Gmsh (FEM, max. element area {ms:.3g} mm²){wt_note}")


def _wire_points(wire, face, defl: float):
    """Points of the wire of the face in order (BRepTools_WireExplorer), arcs and curves discretised by the deflection defl."""
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection
    from OCP.TopAbs import TopAbs_REVERSED

    pts = []
    ex = BRepTools_WireExplorer(wire, face)
    while ex.More():
        e = ex.Current()
        c = BRepAdaptor_Curve(e)
        u1, u2 = c.FirstParameter(), c.LastParameter()
        dis = GCPnts_QuasiUniformDeflection(c, defl, u1, u2)
        if dis.IsDone() and dis.NbPoints() >= 2:
            seg = [dis.Value(i) for i in range(1, dis.NbPoints() + 1)]
        else:
            seg = [c.Value(u1), c.Value(u2)]
        seg = [(p.X(), p.Y(), p.Z()) for p in seg]
        if ex.Orientation() == TopAbs_REVERSED:
            seg.reverse()
        pts.extend(seg[:-1])          # end = start of the next edge
        ex.Next()
    return np.asarray(pts)


def section_from_face(face, name: str = "Custom cross-section", mesh_size: float | None = None,
                      deflection: float | None = None):
    """Section from a planar face of OCP (TopoDS_Face). Returns (SectionProps, placement):
    placement = dict(centroid, normal, y_axis, z_axis) in 3D [mm] – the main axes of the section (y strong).
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepTools import BRepTools
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_WIRE, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    surf = BRepAdaptor_Surface(face)
    if surf.GetType() != GeomAbs_Plane:
        raise SectionError("cross-section area is not planar")
    pos = surf.Plane().Position()
    O = np.array([pos.Location().X(), pos.Location().Y(), pos.Location().Z()])
    X = np.array([pos.XDirection().X(), pos.XDirection().Y(), pos.XDirection().Z()])
    Y = np.array([pos.YDirection().X(), pos.YDirection().Y(), pos.YDirection().Z()])
    N = np.cross(X, Y)
    if face.Orientation() == TopAbs_REVERSED:
        N, Y = -N, -Y              # right-handed system (X, Y, N) with N = outward normal of the face
    if deflection is None:
        box = Bnd_Box()
        BRepBndLib.Add_s(face, box)
        p1, p2 = box.CornerMin(), box.CornerMax()
        deflection = 1e-4 * p1.Distance(p2)
    outer_w = BRepTools.OuterWire_s(face)
    rings_3d, outer = [], None
    ex = TopExp_Explorer(face, TopAbs_WIRE)
    while ex.More():
        w = TopoDS.Wire(ex.Current())
        P = _wire_points(w, face, deflection)
        Q = np.column_stack([(P - O) @ X, (P - O) @ Y])
        if w.IsSame(outer_w):
            outer = Q
        else:
            rings_3d.append(Q)
        ex.Next()
    if outer is None:
        raise SectionError("the area has no outer contour")
    rings = _orient([outer] + rings_3d)
    sec = custom_section(name, rings[0].tolist(), [h.tolist() for h in rings[1:]], mesh_size)
    A, Qy, Qz = _integrals(rings)[:3]
    c2 = np.array([Qy / A, Qz / A])
    ca, sa = math.cos(sec.alpha), math.sin(sec.alpha)
    yax = ca * X + sa * Y
    zax = np.cross(N, yax)
    place = dict(centroid=tuple(O + c2[0] * X + c2[1] * Y), normal=tuple(N), y_axis=tuple(yax),
                 z_axis=tuple(zax))
    return sec, place
