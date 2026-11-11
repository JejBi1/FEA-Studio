"""Definition of the bolts and welds in the study (serialisable to JSON).

A bolt in the FE: a U1 beam (Timoshenko, ccx) in the axis of the hole from the bearing face of the head (on the clamped part,
under the washer) to the bearing face of the nut, resp. to the middle of the screwed-in length. The ends of the beam are
connected by a rigid constraint (*KINEMATIC through an auxiliary node) to the annulus under the head / nut, at the thread
to the cylindrical face of the hole. The parts of the bolt, nut and washers are not meshed.
Position on the axis: s [mm] from the point `origin` in the direction `axis` (from the head towards the nut).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields

import numpy as np

from . import tables


@dataclass
class BoltSpec:
    name: str
    standard: str = ""
    size: float = 10.0                  # nominal diameter d [mm]
    pitch: float | None = None          # None = coarse thread
    length: float | None = None         # nominal length of the bolt
    strength: str = "8.8"
    head: str = "hex"                   # hex | socket | countersunk
    connection: str = "nut"             # nut | tapped | fitted
    origin: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    head_part: str = ""                 # part under the head (washer)
    head_dw: float = 0.0                # outer diameter of the bearing face on head_part
    nut_part: str = ""                  # part under the nut / part with the thread
    nut_s: float = 0.0                  # bearing plane of the nut, resp. start of the thread in the part
    nut_dw: float = 0.0                 # outer diameter of the bearing face on nut_part (nut only)
    thread_end: float = 0.0             # end of the screwing-in (tapped only)
    nut_m: float = 0.0                  # height of the nut from CAD (0 = according to the ISO 4032 table) – stripping of the thread
    plates: list[dict] = field(default_factory=list)   # [{"part", "s0", "s1", "hole"}] od hlavy
    washers: list[dict] = field(default_factory=list)  # [{"part", "side", "h", "d1", "d2"}]
    cad_parts: list[str] = field(default_factory=list)  # bolt, nut, washers (not meshed)
    # assembly and assessment
    preload_mode: str = "auto"          # automatic (ν = 0.9 when tightening) | force (F_M,max [N]) | torque (M_A [N·m])
    preload: float = 0.0
    alpha_A: float = 1.6                # tightening factor (torque wrench 1.4–1.6)
    mu_G: float = 0.12                  # thread friction (min.)
    mu_K: float = 0.12                  # friction under the head (min.)
    mu_T: float = 0.15                  # friction in the joint (slip)
    roughness: str = "10-40"            # Rz [µm] for the settlement: "<10" | "10-40" | "40-160"
    thread_in_shear: bool = True        # the shear plane passes through the thread
    DA: float | None = None             # equivalent outer diameter of the clamped parts D'_A (None = full cone)
    head_cone: float = 0.0              # countersunk head: depth of the cone [mm] (origin = top face of the head)
    shank_d: float | None = None        # fitted bolt: shank diameter [mm] (None = smallest hole, otherwise d + 1)
    enabled: bool = True
    note: str = ""

    # ------------------------------------------------------------ derived
    @property
    def thread(self) -> tables.Thread:
        return tables.thread(self.size, self.pitch)

    @property
    def e(self) -> np.ndarray:
        a = np.asarray(self.axis, float)
        return a / np.linalg.norm(a)

    def point(self, s: float) -> np.ndarray:
        return np.asarray(self.origin, float) + s * self.e

    def washer_h(self, side: str) -> float:
        return sum(w["h"] for w in self.washers if w["side"] == side)

    @property
    def grip(self) -> float:
        """Clamping length l_K including the washers."""
        end = self.nut_s if self.connection != "fitted" else self.plates[-1]["s1"] if self.plates else 0.0
        return end + self.washer_h("head") + (self.washer_h("nut") if self.connection == "nut" else 0.0)

    @property
    def hole(self) -> float | None:
        holes = [p["hole"] for p in self.plates if p.get("hole")]
        return min(holes) if holes else None

    @property
    def shank(self) -> float:
        """Shank diameter of a fitted bolt (fitted hole = shank; DIN 609 / 7968 shank d + 1 mm)."""
        if self.shank_d:
            return float(self.shank_d)
        return float(self.hole or (self.size + 1.0))

    @property
    def designation(self) -> str:
        s = f"M{self.size:g}"
        if self.pitch:
            s += f"×{self.pitch:g}"
        if self.length:
            s += f"×{self.length:g}"
        return f"{self.standard} {s} {self.strength}".strip()

    def beam_end(self) -> float:
        """Position of the end of the beam on the axis (nut: bearing plane, thread: middle of the screwing)."""
        if self.connection == "tapped":
            return 0.5 * (self.nut_s + self.thread_end)
        if self.connection == "fitted":
            return self.plates[-1]["s1"] if self.plates else self.nut_s
        return self.nut_s

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BoltSpec":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class WeldSpec:
    name: str
    part: str                           # body of the weld in CAD
    a: float | None = None              # effective height [mm]; None = from the geometry
    length: float | None = None         # length [mm]; None = from the geometry
    segments: int = 10                  # number of segments along the length for the forces
    fat_sigma: float = 36.0             # FAT for σ_w (root of a fillet weld, EN 1993-1-9 table 8.5 / IIW)
    fat_tau: float = 80.0               # FAT for τ_w
    sn_curve: str = "EN 1993-1-9"       # EN 1993-1-9 | IIW
    gamma_Mf: float = 1.15              # partial factor of the fatigue strength
    open_root: bool = True              # do not connect the direct contact of the welded parts (the load goes through the weld)
    toe: bool = True                    # fatigue of the toe of the weld (hot-spot IIW, fillet weld only)
    fat_toe: float = 90.0               # FAT hot-spot: 90 load-carrying fillet weld, 100 non-load-carrying (IIW, EN 1993-1-9 table B.1)
    toe_t: float | None = None          # thickness of the sheet at the toe for the extrapolation 0.4t/1.0t; None = from the mesh
    filler: str = ""                    # filler metal (tables.FILLERS); "" = matching the base metal (EN 1993-1-8, 4.2)
    enabled: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WeldSpec":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class JointSettings:
    gamma_M2: float = 1.25              # EN 1993-1-8 (bolts, welds)
    gamma_M3: float = 1.25              # friction joints (ULS)
    category: str = "A"                 # EN 1993-1-8: A (shear/bearing) | B/C (friction)
    slip_mu: float = 0.3                # friction coefficient for F_s,Rd (surface class C)
    preload_case: bool = True           # add the load state "Bolt preload" (FE)
    shear_split: str = "group"          # shear force per bolt: group = group of bolts (uniformly + torsion
                                        # by the polar moment, EN 1993-1-8) | fe = by the cells of the joined joint
    bolt_elements: int = 64             # number of U1 elements per bolt (min. fem.MIN_BOLT_ELEMENTS)
    bolt_mesh: float = 0.25             # local element size at the bolts = multiple of d
    weld_mesh: float = 1.0              # local element size in the weld = multiple of a (estimate)
    weld_colors: list[str] = field(default_factory=list)   # colours of the welds in CAD ("#rrggbb") – the recognition rule
    weld_color_tol: float = 0.03        # colour tolerance (RGB distance 0–1)
    contact_joint: bool = False         # joint of clamped parts as a contact with friction μ_T (F8.5b) – opening, slip

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "JointSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


PRELOAD_CASE = "Bolt preload"
PRELOAD_CASE_LEGACY = "Předpětí šroubů"      # name in results saved by versions up to 1.4 (Czech source texts)


def is_preload(name: str) -> bool:
    """True for the name of the bolt-preload state (also the Czech name stored in older projects)."""
    return name in (PRELOAD_CASE, PRELOAD_CASE_LEGACY)
