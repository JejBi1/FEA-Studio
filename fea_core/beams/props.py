"""Section properties – the common interface of the section library, the FE of bars and the assessment according to EN 1993-1-1.

Axes of the section (EN 1993): y = main (strong) axis, z = minor (weak) axis; origin at the centroid. Values to the main
central axes (for unsymmetric sections, e.g. L, `alpha` is the angle of the main axis y from the axis y0 of the input of the section).
Units mm, mm², mm⁴, mm⁶.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class SectionProps:
    name: str                          # "IPE 200", "CHS 60.3×4", "Own: Profil 1"
    shape: str                         # I | U | L | CHS | RHS | RECT | CIRC | TUBE | T | custom
    dims: dict = field(default_factory=dict)   # dimensions of the shape [mm]: h, b, tw, tf, r, t, d, …
    A: float = 0.0                     # area [mm²]
    Iy: float = 0.0                    # second moment of area about the main axis y [mm⁴]
    Iz: float = 0.0                    # about the main axis z [mm⁴]
    J: float = 0.0                     # torsional constant in pure torsion I_t [mm⁴]
    Iw: float = 0.0                    # warping constant I_w [mm⁶] (0 = neglected / closed)
    Wel_y: float = 0.0                 # elastic section modulus [mm³] (min. over the extreme fibres)
    Wel_z: float = 0.0
    Wpl_y: float = 0.0                 # plastic section modulus [mm³]
    Wpl_z: float = 0.0
    Av_y: float = 0.0                  # shear area for V_y [mm²] (EN 1993-1-1, 6.2.6)
    Av_z: float = 0.0                  # shear area for V_z [mm²]
    Wt: float = 0.0                    # torsional modulus τ_max = T/Wt [mm³]
    alpha: float = 0.0                 # angle of the main axis y from the axis y0 of the input [rad] (L profiles)
    ys: float = 0.0                    # shear centre relative to the centroid in the main axes [mm]
    zs: float = 0.0
    fibers: list = field(default_factory=list)   # points for stresses [(y, z), …] in the main axes [mm]
    source: str = ""                   # source (standard, calculation by sectionproperties, …)

    @property
    def iy(self) -> float:
        return (self.Iy / self.A) ** 0.5 if self.A > 0 else 0.0

    @property
    def iz(self) -> float:
        return (self.Iz / self.A) ** 0.5 if self.A > 0 else 0.0

    @property
    def Ip(self) -> float:
        """Polar moment about the centroid Iy + Iz [mm⁴]."""
        return self.Iy + self.Iz

    def mass_per_m(self, rho: float = 7.85e-9) -> float:
        """Mass [kg/m] at the density rho [t/mm³]."""
        return self.A * 1000.0 * rho * 1000.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fibers"] = [list(map(float, p)) for p in self.fibers]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SectionProps":
        from dataclasses import fields
        names = {f.name for f in fields(cls)}
        out = cls(**{k: v for k, v in (d or {}).items() if k in names})
        out.fibers = [tuple(p) for p in out.fibers]
        return out
