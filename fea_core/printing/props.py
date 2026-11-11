"""Local voxel detail of 3D printing (F5 B): submodel of beads around a critical place.

Procedure:
  1. region = box in the print axes (centre + size + margin) around the given point or the place of the lowest print safety
     of the homogenised model (StepResult.fields["print_safety"]);
  2. paths from the print settings in voxels (voxel_paths.generate): perimeters, top/bottom layers, thin places, infill;
  3. hex8 elements C3D8I (incompatible modes – bending of thin beads without locking) in ccx, material = full extrusion
     in the axes of the line (printing.part_materials: perimeter `ml`, axes 1 = line, 2 = across, 3 = build – the weak layer bond
     is in E_z, Rm_z), *ELASTIC TYPE=ORTHO per classes of the line angle with *ORIENTATION (as F4);
  4. boundary conditions = displacements of the global (homogenised) solution interpolated by tet10 shape functions at the nodes
     of the cut faces of the box (NOT on the free surface of the part) + at the nodes in contact with other parts and at constraint faces → *BOUNDARY,
     one *STEP per load state;
  5. stresses at integration points (*EL PRINT, GLOBAL=YES) → bead axes → Tsai-Wu / Tsai-Hill (failure.safety),
     safety in the assessment region (without margin), comparison with the homogenised model in the same region.

Verified ccx 2.23 conventions (scratch/voxel/exp1.py, exp2.py, 14. 10. 2026):
  * C3D8I has 3 internal nodes of incompatible modes as GLOBAL unknowns (ccx does not write them to .frd): equations
    ≈ 3·nodes + 9·elements (75 k voxels → 914 k equations), C3D8 only 3·nodes;
  * *ORIENTATION RECTANGULAR a = axis 1, b = axis 2 of the material; S in .frd and *EL PRINT GLOBAL=YES in global axes,
    *EL PRINT without GLOBAL (default NO) in material axes; .dat order xx yy zz xy xz yz (read_el_print reorders);
  * nodal S in .frd are extrapolations (for C3D8I a deviation of ~4e-5 from the exact homogeneous state) – safety is therefore
    computed from integration points (.dat, precision 1e-7).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict, fields


@dataclass
class PrintProps:
    family: str = "PLA"
    # anisotropy of the line in the perimeters (across / along the line in the plane of the layer; 1 = off)
    line_E_ratio: float = 1.0
    line_R_ratio: float = 1.0
    # creep: φ(t) = a·t^n (t [h], 23 °C); table [T °C, k_E, k_φ]; creep rupture k_R = 1 − β·log10(1 + t)
    creep_a: float = 0.0
    creep_n: float = 0.4
    creep_T: list = field(default_factory=lambda: [[23.0, 1.0, 1.0]])
    rupture_beta: float = 0.0
    rupture_min: float = 0.3
    creep_max_util: float = 0.35      # utilisation (1/λ) above this value → nonlinear creep (warning)
    # S-N of the print (amplitudes, XY = in the plane of the layer): SD at ND, slopes k, k2; SD_z/SD_xy (None = Rm_z/Rm)
    sn_SD_ratio: float = 0.10         # SD = share of Rm (if sn_SD is not given)
    sn_SD: float | None = None        # MPa (takes precedence over sn_SD_ratio)
    sn_ND: float = 2e6
    sn_k: float = 5.5
    sn_k2: float = 10.0
    sn_z_ratio: float | None = None
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PrintProps":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    # ---------------------------------------------------------------- creep
    def temperature_factors(self, T: float) -> tuple[float, float, bool]:
        """(k_E, k_φ, valid) for the temperature T [°C]; below the first row of the table = the first row."""
        tab = sorted([list(map(float, r)) for r in self.creep_T])
        if T <= tab[0][0]:
            return tab[0][1], tab[0][2], True
        for (t0, e0, p0), (t1, e1, p1) in zip(tab, tab[1:]):
            if T <= t1:
                w = (T - t0) / (t1 - t0)
                return e0 + w * (e1 - e0), p0 + w * (p1 - p0), True
        return tab[-1][1], tab[-1][2], False

    def creep_phi(self, hours: float) -> float:
        return self.creep_a * max(float(hours), 0.0) ** self.creep_n

    def creep_factors(self, hours: float, T: float) -> dict:
        """Multiple of the stiffnesses (k_E/(1 + k_φ·φ)) and strengths (creep rupture) for the time and temperature."""
        kE, kphi, ok = self.temperature_factors(T)
        phi = kphi * self.creep_phi(hours)
        kR = max(self.rupture_min, 1.0 - self.rupture_beta * math.log10(1.0 + max(hours, 0.0)))
        return {"phi": phi, "k_E": kE, "stiffness": kE / (1.0 + phi), "strength": kR, "valid": ok,
                "T_max": max(float(r[0]) for r in self.creep_T)}

    # ---------------------------------------------------------------- S-N
    def sn_xy(self, Rm: float | None) -> float:
        if self.sn_SD:
            return float(self.sn_SD)
        if not Rm:
            raise ValueError("The print S-N curve requires the material Rm or a specified amplitude SD")
        return self.sn_SD_ratio * Rm


_EZEH = ("S-N: Ezeh & Susmel, Int. J. Fatigue 126 (2019) 319–326 (FFF PLA, k = 5.5, SD(2e6) ≈ 0.1 Rm, P_s 95 %)")
_LINE = ("lines: Song et al., Mater. Des. 123 (2017); Khosravani et al., Sci. Rep. 12 (2022) 1016 (90°/0° raster: Rm 0.54–0.58; E ratio 0.8 estimated)")
DEFAULTS: dict[str, PrintProps] = {
    "PLA": PrintProps(
        "PLA", line_E_ratio=0.8, line_R_ratio=0.55, creep_a=0.04, creep_n=0.4,
        creep_T=[[23.0, 1.0, 1.0], [30.0, 0.97, 1.5], [40.0, 0.92, 3.0], [45.0, 0.85, 5.0]],
        rupture_beta=0.10, source="INDICATIVE – " + _LINE + "; creep: estimate (Findley, Waseem et al. 2020), Tg ≈ 55–60 °C → valid up to 45 °C; " + _EZEH),
    "PETG": PrintProps(
        "PETG", line_E_ratio=0.85, line_R_ratio=0.6, creep_a=0.03, creep_n=0.4,
        creep_T=[[23.0, 1.0, 1.0], [40.0, 0.95, 1.6], [50.0, 0.9, 2.5], [60.0, 0.8, 4.5]],
        rupture_beta=0.08, source="INDICATIVE – lines as for PLA (less pronounced); creep: estimate from PMC12349189 (PETG, 21 °C), Tg ≈ 78–80 °C → valid up to 60 °C; S-N taken from PLA"),
    "ABS": PrintProps(
        "ABS", line_E_ratio=0.85, line_R_ratio=0.6, creep_a=0.03, creep_n=0.35,
        creep_T=[[23.0, 1.0, 1.0], [50.0, 0.92, 2.0], [70.0, 0.8, 4.0]],
        rupture_beta=0.08, source="INDICATIVE – estimate based on PETG (Tg of ABS ≈ 105 °C), S-N taken from PLA"),
    "PC": PrintProps(
        "PC", line_E_ratio=0.90, line_R_ratio=0.65, creep_a=0.02, creep_n=0.32,
        creep_T=[[23.0, 1.0, 1.0], [60.0, 0.95, 1.5], [80.0, 0.88, 2.5], [100.0, 0.78, 4.5]],
        rupture_beta=0.06, source="INDICATIVE – polycarbonate (Tg ≈ 115–125 °C); creep: Findley model, S-N from literature"),
    "PA": PrintProps(
        "PA", line_E_ratio=0.85, line_R_ratio=0.55, creep_a=0.035, creep_n=0.38,
        creep_T=[[23.0, 1.0, 1.0], [40.0, 0.93, 1.8], [60.0, 0.85, 3.0], [80.0, 0.72, 5.0]],
        rupture_beta=0.09, source="INDICATIVE – polyamide PA / PA-CF; creep: Findley model, S-N estimated from literature"),
    "TPU": PrintProps(
        "TPU", line_E_ratio=0.95, line_R_ratio=0.85, creep_a=0.06, creep_n=0.45,
        creep_T=[[23.0, 1.0, 1.0], [40.0, 0.90, 2.0], [60.0, 0.75, 4.0]],
        rupture_beta=0.12, source="INDICATIVE – TPU elastomer 95A; creep: Findley model, high toughness"),
    "PEEK": PrintProps(
        "PEEK", line_E_ratio=0.90, line_R_ratio=0.60, creep_a=0.015, creep_n=0.30,
        creep_T=[[23.0, 1.0, 1.0], [80.0, 0.95, 1.4], [120.0, 0.88, 2.2], [140.0, 0.75, 4.0]],
        rupture_beta=0.05, source="INDICATIVE – PEEK (Tg ≈ 143 °C, Tm ≈ 343 °C); high strength and temperature resistance"),
    "PEI": PrintProps(
        "PEI", line_E_ratio=0.90, line_R_ratio=0.60, creep_a=0.02, creep_n=0.32,
        creep_T=[[23.0, 1.0, 1.0], [80.0, 0.95, 1.4], [120.0, 0.90, 2.0], [150.0, 0.80, 3.5]],
        rupture_beta=0.05, source="INDICATIVE – PEI Ultem (Tg ≈ 186 °C); self-extinguishing aerospace thermoplastic"),
}


def family_of(name: str) -> str:
    u = name.upper()
    for key in ("PETG", "ABS", "ASA", "TPU", "PEEK", "PEI", "ULTEM"):
        if key in u:
            if key == "ASA":
                return "ABS"
            if key == "ULTEM":
                return "PEI"
            return key
    toks = re.findall(r"[A-Z0-9]+", u)
    if "PC" in toks:
        return "PC"
    if any(k in u for k in ("PAHT", "PA-CF", "PA6", "PA11", "PA12", "NYLON")) or "PA" in toks:
        return "PA"
    return "PLA"


def default_props(mat_name: str) -> PrintProps:
    return PrintProps.from_dict(DEFAULTS[family_of(mat_name)].to_dict())


def props_of(study, mat) -> PrintProps:
    """Properties of the printing of the material: default according to the family + override from the study (PrintSettings.materials)."""
    p = default_props(mat.name).to_dict()
    p.update((study.printing.materials.get(mat.name) or {}) if study is not None else {})
    return PrintProps.from_dict(p)
