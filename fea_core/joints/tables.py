"""Tables of standard fasteners (nominal values, metric coarse thread).

Sources: ISO 261/724 (thread), ISO 898-1 (strength classes, minimum Rm/Rp0.2), ISO 3506-1 (stainless),
ISO 4014/4017 (hexagon head), ISO 4762 (cylindrical head with an inner hexagon),
ISO 4032 (nuts), ISO 7089 (washers), ISO 273 (clearance holes), EN 1993-1-8 table 3.1.
Dimensions in mm, strengths in MPa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# coarse pitch ISO 261
PITCH = {
    2: 0.4, 2.5: 0.45, 3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25, 10: 1.5, 12: 1.75, 14: 2.0, 16: 2.0,
    18: 2.5, 20: 2.5, 22: 2.5, 24: 3.0, 27: 3.0, 30: 3.5, 33: 3.5, 36: 4.0, 39: 4.0, 42: 4.5,
}

# hexagon head ISO 4014/4017: s (key width), k (height of the head), dw (min. diameter of the bearing face)
HEX_HEAD = {
    3: (5.5, 2.0, 4.57), 4: (7.0, 2.8, 5.88), 5: (8.0, 3.5, 6.88), 6: (10.0, 4.0, 8.88),
    8: (13.0, 5.3, 11.63), 10: (16.0, 6.4, 14.63), 12: (18.0, 7.5, 16.63), 14: (21.0, 8.8, 19.64),
    16: (24.0, 10.0, 22.49), 18: (27.0, 11.5, 25.34), 20: (30.0, 12.5, 28.19), 22: (34.0, 14.0, 31.71),
    24: (36.0, 15.0, 33.61), 27: (41.0, 17.0, 38.0), 30: (46.0, 18.7, 42.75), 33: (50.0, 21.0, 46.55),
    36: (55.0, 22.5, 51.11), 39: (60.0, 25.0, 55.86), 42: (65.0, 26.0, 59.95),
}
# DIN 931/933 (older): for M10, M12, M14, M22 a different key width
DIN_HEX_S = {10: 17.0, 12: 19.0, 14: 22.0, 22: 32.0}

# cylindrical head ISO 4762: dk (head diameter), k (= d), s (inner hexagon), dw (min. bearing diameter)
SOCKET_HEAD = {
    3: (5.5, 3.0, 2.5, 5.07), 4: (7.0, 4.0, 3.0, 6.53), 5: (8.5, 5.0, 4.0, 8.03), 6: (10.0, 6.0, 5.0, 9.38),
    8: (13.0, 8.0, 6.0, 12.33), 10: (16.0, 10.0, 8.0, 15.33), 12: (18.0, 12.0, 10.0, 17.23),
    14: (21.0, 14.0, 12.0, 20.17), 16: (24.0, 16.0, 14.0, 23.17), 18: (27.0, 18.0, 14.0, 25.87),
    20: (30.0, 20.0, 17.0, 28.87), 22: (33.0, 22.0, 17.0, 31.81), 24: (36.0, 24.0, 19.0, 34.81),
    27: (40.0, 27.0, 19.0, 38.61), 30: (45.0, 30.0, 22.0, 43.61), 33: (50.0, 33.0, 24.0, 48.6),
    36: (54.0, 36.0, 27.0, 52.54),
}

# nut ISO 4032: m (height); s and dw as for the hexagon head
NUT_M = {
    3: 2.4, 4: 3.2, 5: 4.7, 6: 5.2, 8: 6.8, 10: 8.4, 12: 10.8, 14: 12.8, 16: 14.8, 18: 15.8, 20: 18.0,
    22: 19.4, 24: 21.5, 27: 23.8, 30: 25.6, 33: 28.7, 36: 31.0, 39: 33.4, 42: 34.0,
}

# washers ISO 7089 / 7090 / DIN 125-A: d1 (inner), d2 (outer), h
WASHER = {
    3: (3.2, 7.0, 0.5), 4: (4.3, 9.0, 0.8), 5: (5.3, 10.0, 1.0), 6: (6.4, 12.0, 1.6), 8: (8.4, 16.0, 1.6),
    10: (10.5, 20.0, 2.0), 12: (13.0, 24.0, 2.5), 14: (15.0, 28.0, 2.5), 16: (17.0, 30.0, 3.0),
    18: (19.0, 34.0, 3.0), 20: (21.0, 37.0, 3.0), 22: (23.0, 39.0, 3.0), 24: (25.0, 44.0, 4.0),
    27: (28.0, 50.0, 4.0), 30: (31.0, 56.0, 4.0), 33: (34.0, 60.0, 5.0), 36: (37.0, 66.0, 5.0),
}
# large washers ISO 7093-1
WASHER_LARGE = {
    3: (3.2, 9.0, 0.8), 4: (4.3, 12.0, 1.0), 5: (5.3, 15.0, 1.2), 6: (6.4, 18.0, 1.6), 8: (8.4, 24.0, 2.0),
    10: (10.5, 30.0, 2.5), 12: (13.0, 37.0, 3.0), 14: (15.0, 44.0, 3.0), 16: (17.0, 50.0, 3.0),
    20: (22.0, 60.0, 4.0), 24: (26.0, 72.0, 5.0), 30: (33.0, 92.0, 6.0), 36: (39.0, 110.0, 8.0),
}

# clearance holes ISO 273: (fine, medium, coarse)
CLEARANCE = {
    3: (3.2, 3.4, 3.6), 4: (4.3, 4.5, 4.8), 5: (5.3, 5.5, 5.8), 6: (6.4, 6.6, 7.0), 8: (8.4, 9.0, 10.0),
    10: (10.5, 11.0, 12.0), 12: (13.0, 13.5, 14.5), 14: (15.0, 15.5, 16.5), 16: (17.0, 17.5, 18.5),
    18: (19.0, 20.0, 21.0), 20: (21.0, 22.0, 24.0), 22: (23.0, 24.0, 26.0), 24: (25.0, 26.0, 28.0),
    27: (28.0, 30.0, 32.0), 30: (31.0, 33.0, 35.0), 33: (34.0, 36.0, 38.0), 36: (37.0, 39.0, 42.0),
    39: (40.0, 42.0, 45.0), 42: (43.0, 45.0, 48.0),
}


@dataclass(frozen=True)
class StrengthClass:
    name: str
    Rm: float          # min. tensile strength (ISO 898-1 / ISO 3506-1)
    Rp02: float        # min. mez kluzu Rp0,2 (resp. ReL)
    fub: float | None  # nominal f_ub according to EN 1993-1-8 table 3.1 (None = the class is not there)
    fyb: float | None
    E: float = 210000.0
    stainless: bool = False
    source: str = "ISO 898-1"


def strength_class(name: str, d: float = 10.0) -> StrengthClass:
    """Strength class of the bolt; for 8.8 it depends on the diameter (d ≤ 16 / > 16 mm)."""
    n = name.strip().upper().replace(",", ".")
    if n == "8.8":
        if d <= 16:
            return StrengthClass("8.8", 800, 640, 800, 640)
        return StrengthClass("8.8", 830, 660, 800, 640)
    if n not in CLASSES:
        raise KeyError(f"Unknown strength class '{name}'")
    return CLASSES[n]


CLASSES = {
    "4.6": StrengthClass("4.6", 400, 240, 400, 240),
    "4.8": StrengthClass("4.8", 420, 340, 400, 320),
    "5.6": StrengthClass("5.6", 500, 300, 500, 300),
    "5.8": StrengthClass("5.8", 520, 420, 500, 400),
    "6.8": StrengthClass("6.8", 600, 480, 600, 480),
    "9.8": StrengthClass("9.8", 900, 720, None, None),
    "10.9": StrengthClass("10.9", 1040, 940, 1000, 900),
    "12.9": StrengthClass("12.9", 1220, 1100, None, None),
    "A2-50": StrengthClass("A2-50", 500, 210, 500, 210, 200000.0, True, "ISO 3506-1"),
    "A2-70": StrengthClass("A2-70", 700, 450, 700, 450, 200000.0, True, "ISO 3506-1"),
    "A2-80": StrengthClass("A2-80", 800, 600, 800, 600, 200000.0, True, "ISO 3506-1"),
    "A4-50": StrengthClass("A4-50", 500, 210, 500, 210, 200000.0, True, "ISO 3506-1"),
    "A4-70": StrengthClass("A4-70", 700, 450, 700, 450, 200000.0, True, "ISO 3506-1"),
    "A4-80": StrengthClass("A4-80", 800, 600, 800, 600, 200000.0, True, "ISO 3506-1"),
}
CLASS_NAMES = ["4.6", "4.8", "5.6", "5.8", "6.8", "8.8", "9.8", "10.9", "12.9",
               "A2-50", "A2-70", "A2-80", "A4-50", "A4-70", "A4-80"]


@dataclass(frozen=True)
class Thread:
    """Metric thread: d nominal, P pitch, d2 mean, d3 minor (bolt), As calculation section."""
    d: float
    P: float

    @property
    def d2(self) -> float:
        return self.d - 0.649519 * self.P

    @property
    def d3(self) -> float:
        return self.d - 1.226869 * self.P

    @property
    def d1(self) -> float:                       # small diameter of the nut
        return self.d - 1.082532 * self.P

    @property
    def ds(self) -> float:                       # diameter of the calculation section
        return 0.5 * (self.d2 + self.d3)

    @property
    def As(self) -> float:
        return math.pi / 4 * self.ds ** 2

    @property
    def Ad3(self) -> float:
        return math.pi / 4 * self.d3 ** 2

    @property
    def AN(self) -> float:                       # nominal section of the shank
        return math.pi / 4 * self.d ** 2


def size_key(d: float) -> float:
    """Normalised size key (int, if d is an integer)."""
    return int(d) if abs(d - round(d)) < 1e-9 else d


def thread(d: float, P: float | None = None) -> Thread:
    k = size_key(d)
    if P is None:
        if k not in PITCH:
            raise KeyError(f"Size M{d:g} is not in the coarse thread table")
        P = PITCH[k]
    return Thread(float(d), float(P))


def head_dims(d: float, head: str, standard: str = "") -> dict:
    """Dimensions of the head: {'dw', 'k', 's', 'lsk'} (lsk = equivalent length of the head for the compliance, VDI 2230)."""
    k = size_key(d)
    if head == "socket":
        dk, kk, s, dw = SOCKET_HEAD.get(k, (1.5 * d, d, 0.75 * d, 1.45 * d))
        return {"dw": dw, "k": kk, "s": s, "dk": dk, "lsk": 0.4 * d}
    s, kk, dw = HEX_HEAD.get(k, (1.6 * d, 0.65 * d, 1.45 * d))
    if standard.upper().startswith("DIN") and k in DIN_HEX_S:
        s = DIN_HEX_S[k]
    return {"dw": dw, "k": kk, "s": s, "lsk": 0.5 * d}


def nut_dims(d: float, standard: str = "") -> dict:
    k = size_key(d)
    s, _, dw = HEX_HEAD.get(k, (1.6 * d, 0.65 * d, 1.45 * d))
    m = NUT_M.get(k, 0.85 * d)
    if standard.upper().replace(" ", "") == "DIN934":
        m = 0.8 * d
        s = DIN_HEX_S.get(k, s)
    return {"dw": dw, "m": m, "s": s}


def washer_dims(d: float, large: bool = False) -> dict:
    k = size_key(d)
    tab = WASHER_LARGE if large else WASHER
    d1, d2, h = tab.get(k, (1.05 * d, 2.0 * d, 0.2 * d))
    return {"d1": d1, "d2": d2, "h": h}


def clearance_class(d: float, hole: float) -> str:
    """Class of the clearance of the hole according to ISO 273 (or 'smaller than fine' / 'larger than coarse')."""
    k = size_key(d)
    if k not in CLEARANCE:
        return "?"
    fine, med, coarse = CLEARANCE[k]
    tol = 0.02
    if hole < d - tol:
        return "thread / smaller than d"
    if hole < fine - tol:
        return "fitted / smaller than fine"
    if hole <= fine + tol:
        return "fine"
    if hole <= med + tol:
        return "medium"
    if hole <= coarse + tol:
        return "coarse"
    return "larger than coarse"


def shank_length(d: float, L: float | None, standard: str) -> float:
    """Length of the plain shank (0 = thread up to the head). ISO 4014/4762/DIN 931/912: b = 2d + 6/12/25."""
    std = standard.upper().replace(" ", "")
    if L is None or std in ("ISO4017", "DIN933", "ISO7380", "ISO10642", "ISO7046", "ISO1207", "ISO4026"):
        return 0.0
    if std in ("ISO4762", "DIN912"):
        b = 2 * d + 12
    else:
        b = 2 * d + 6 if L <= 125 else (2 * d + 12 if L <= 200 else 2 * d + 25)
    return max(0.0, L - b)


# ------------------------------------------------------------------ filler metals of welds (chapter 5.3)
# (Re, Rm) minimum values of the pure weld metal [MPa]; EN 1993-1-8, 4.2: the filler must have both Re and Rm
# ≥ the base material (matching / over-matching) – then the f_u of the base material is assessed, otherwise f_u = Rm of the filler.
FILLERS: dict[str, tuple[float, float, str]] = {
    "G 38 (ISO 14341-A, MAG wire)": (380.0, 470.0, "ISO 14341-A: Re ≥ 380, Rm 470–600 MPa"),
    "G 42 – G3Si1 (ISO 14341-A, MAG wire)": (420.0, 500.0, "ISO 14341-A: Re ≥ 420, Rm 500–640 MPa"),
    "G 46 – G4Si1 (ISO 14341-A, MAG wire)": (460.0, 530.0, "ISO 14341-A: Re ≥ 460, Rm 530–680 MPa"),
    "E 38 (ISO 2560-A, coated electrode)": (380.0, 470.0, "ISO 2560-A: Re ≥ 380, Rm 470–600 MPa"),
    "E 42 (ISO 2560-A, coated electrode)": (420.0, 500.0, "ISO 2560-A: Re ≥ 420, Rm 500–640 MPa"),
    "E 46 (ISO 2560-A, coated electrode)": (460.0, 530.0, "ISO 2560-A: Re ≥ 460, Rm 530–680 MPa"),
    "ER70S-6 (AWS A5.18)": (400.0, 480.0, "AWS A5.18: Re ≥ 400 (58 ksi), Rm ≥ 480 MPa (70 ksi)"),
    "G 19 9 L Si – 308LSi (ISO 14343-A, stainless)": (320.0, 510.0, "ISO 14343-A: Re ≥ 320, Rm ≥ 510 MPa"),
    "G 19 12 3 L Si – 316LSi (ISO 14343-A, stainless)": (320.0, 510.0, "ISO 14343-A: Re ≥ 320, Rm ≥ 510 MPa"),
}


def filler(name: str) -> tuple[float, float, str] | None:
    """Filler metal by name (also just the beginning, e.g. "G 42", "ER70S-6", "316LSi")."""
    if not name:
        return None
    if name in FILLERS:
        return FILLERS[name]
    key = name.strip().upper().replace(" ", "")
    for k, v in FILLERS.items():
        if key and (k.upper().replace(" ", "").startswith(key) or key in k.upper().replace(" ", "")):
            return v
    return None
