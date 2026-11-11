"""Fatigue settings in the study (without numba – it is imported with the study)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

from .history import LoadHistory
from .sn import SURFACE_LABELS

METHOD_LABELS = {
    "sn": "S-N curve (stress-life, high-cycle)",
    "en": "ε-N curve (strain-life, low-cycle)",
}

EQUIVALENT_LABELS = {
    "signed_mises": "signed von Mises (sign of the dominant principal stress)",
    "max_principal": "maximum principal stress (largest absolute value)",
    "critical_plane": "critical plane (rainflow in every plane, worst plane; Findley / normal stress)",
}
MEAN_LABELS = {
    "none": "no correction",
    "goodman": "Goodman",
    "gerber": "Gerber",
    "morrow": "Morrow",
    "swt": "Smith-Watson-Topper (SWT)",
}
NOTCH_RULE_LABELS = {
    "neuber": "Neuber's rule (σ·ε = σ_el²/E)",
    "glinka": "Glinka ESED method (Wel = σ_el²/(2E))",
}
CP_CRITERION_LABELS = {
    "auto": "by material (ductile → Findley, brittle → normal stress)",
    "findley": "Findley (shear + normal stress in the plane)",
    "normal": "normal stress in the plane (tensile failure, brittle materials)",
}
CP_RATIO_DUCTILE = math.sqrt(3.0)   # f₋₁/t₋₁ of ductile metals (von Mises, Shigley; steel of tests ~1.6–1.8)


def findley_k(ratio: float) -> float:
    """k of the Findley criterion from the ratio of the fatigue limits s = f₋₁/t₋₁ (1 < s ≤ 2): max_θ f under alternating tension σa is
    λ·σa, λ = (k + √(1+k²))/2, under alternating torsion τa·√(1+k²) → s = 2√(1+k²)/(k + √(1+k²)).
    """
    if not 1.0 < ratio <= 2.0:
        raise ValueError(f"endurance limit ratio f₋₁/t₋₁ = {ratio:g} is outside the range of the Findley criterion (1; 2] – for brittle materials (f₋₁/t₋₁ ≈ 1) choose the normal stress criterion")
    return (2.0 - ratio) / (2.0 * math.sqrt(ratio - 1.0))


def findley_lambda(k: float) -> float:
    """λ: the Findley parameter under alternating tension σa is λ·σa."""
    return 0.5 * (k + math.sqrt(1.0 + k * k))


@dataclass
class FatigueSettings:
    history: LoadHistory = field(default_factory=LoadHistory)
    method: str = "sn"                # sn (Stress-Life) | en (Strain-Life)
    equivalent: str = "signed_mises"  # method of the multiaxiality (EQUIVALENT_LABELS)
    mean_correction: str = "goodman"
    repeats: float = 1e6              # design number of blocks (repetitions of the load curve)
    closed_blocks: bool = True        # the block repeats → close the rest of the rainflow
    surface: str = "obrobeny"
    reliability: float = 50.0
    size_d: float | None = None       # effective diameter for the size factor [mm]
    k_load: float = 1.0
    k_temp: float = 1.0
    k_misc: float = 1.0
    sn: dict[str, dict] = field(default_factory=dict)   # material -> given S-N (otherwise estimate from Rm)
    # critical plane (equivalent = "critical_plane")
    cp_criterion: str = "auto"        # auto | findley | normal (CP_CRITERION_LABELS)
    cp_ratio: dict[str, float] = field(default_factory=dict)   # material -> f₋₁/t₋₁ (otherwise according to the class)
    cp_planes: int = 100              # planes of the coarse grid on the hemisphere (100 ≈ 14°, refinement to ~1°)
    cp_refine: int = 4                # number of refinement levels (step /2)
    cp_dirs: int = 12                 # shear directions in the plane (per 180°), then refinement
    cp_thorough: bool = False         # thorough search (4× more planes, more candidates; ~3× slower)
    # Strain-Life ε-N (method = "en")
    en_notch_rule: str = "neuber"     # neuber | glinka
    en_mean_correction: str = "morrow"  # none | morrow | swt
    en_params: dict[str, dict] = field(default_factory=dict)  # material -> parameters of ε-N

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FatigueSettings":
        d = dict(d)
        hist = LoadHistory.from_dict(d.pop("history", {}))
        names = set(cls.__dataclass_fields__)
        return cls(history=hist, **{k: v for k, v in d.items() if k in names})

    @property
    def critical_plane(self) -> bool:
        return self.equivalent == "critical_plane"

    def search_params(self) -> tuple[int, int, int, int, int, int]:
        """(planes, levels, directions, candidates, directions on the coarse mesh, exactly on the coarse mesh)."""
        planes = max(int(self.cp_planes), 12)
        if self.cp_thorough:
            return 4 * planes, max(int(self.cp_refine), 0), max(int(self.cp_dirs), 4), 6, 6, 1
        return planes, max(int(self.cp_refine), 0), max(int(self.cp_dirs), 4), 3, 4, 1

    def search_text(self) -> str:
        planes, levels, dirs, ncand, _, _ = self.search_params()
        h = math.degrees(math.sqrt(2.0 * math.pi / planes))      # = critical_plane.plane_spacing
        return (f"plane search: {planes} planes on the hemisphere (spacing ~{h:.0f}°) + principal directions, {ncand} candidates refined to ~{h / 2 ** levels:.1f}°; shear up to {dirs} in-plane directions + refinement"
                + (" (thorough search)" if self.cp_thorough else ""))

    def describe(self) -> list[str]:
        if self.method == "en":
            rule_text = NOTCH_RULE_LABELS.get(self.en_notch_rule, self.en_notch_rule)
            mean_key = self.mean_correction if self.mean_correction in ("morrow", "swt", "none") else self.en_mean_correction
            mean_text = MEAN_LABELS.get(mean_key, mean_key)
            lines = [
                f"Fatigue method: {METHOD_LABELS['en']}",
                f"Equivalent stress: {EQUIVALENT_LABELS.get(self.equivalent, self.equivalent)}",
                f"Notch correction: {rule_text}",
                f"Mean stress correction: {mean_text}",
                "Life curve: Coffin-Manson-Basquin with the Ramberg-Osgood cyclic curve",
                "Cycle counting: rainflow ASTM E1049" + (", repeated block closed" if self.closed_blocks else ""),
                "Damage accumulation: linear (Palmgren–Miner), D = 1 = failure",
                f"Design number of blocks: {self.repeats:g}",
            ]
            return lines

        if self.critical_plane:
            lines = [
                f"Multiaxiality: critical plane method – criterion {CP_CRITERION_LABELS[self.cp_criterion]}; plane with the greatest damage per block"]
            if self.cp_criterion != "normal":
                lines.append(
                    "Findley (1959): f = τa + k·σn,max, equivalent amplitude σ = f/λ, λ = (k + √(1+k²))/2 (fully reversed tension → σa, pure torsion → s·τa), k = (2 − s)/(2√(s − 1)) from the endurance limit ratio s = f₋₁/t₋₁; τa = rainflow of the shear projection onto an in-plane direction (worst direction), σn,max = maximum normal stress within the cycle loop")
            if self.cp_criterion != "findley":
                lines.append("Normal stress: σn(t) = n·σ(t)·n in the plane, rainflow of σn, S-N directly")
            lines += [
                "Search: " + self.search_text().removeprefix("plane search: "),
                f"Mean stress correction: {MEAN_LABELS[self.mean_correction]} only for the normal stress criterion" + (" (compressive mean stress without beneficial effect)" if self.mean_correction != "none" else "")
                + "; for Findley the mean stress is in the term k·σn,max",
            ]
        else:
            lines = [
                f"Equivalent stress: {EQUIVALENT_LABELS[self.equivalent]}",
                f"Mean stress correction: {MEAN_LABELS[self.mean_correction]}"
                + (" (compressive mean stress without beneficial effect)" if self.mean_correction != "none" else ""),
            ]
        return lines + [
            "Cycle counting: rainflow ASTM E1049" + (", repeated block closed" if self.closed_blocks else ""),
            "Damage accumulation: linear (Palmgren–Miner), D = 1 = failure",
            f"Surface: {SURFACE_LABELS.get(self.surface, self.surface)}, reliability {self.reliability:g} %",
            f"Design number of blocks: {self.repeats:g}",
        ]
