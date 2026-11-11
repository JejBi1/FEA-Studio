"""Specification of the topology optimisation (BESO / SIMP)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class TopologyOptSpec:
    """Parameters of the topology optimisation.

    target_volume_fraction: target volume fraction (e.g. 0.5 = 50 % of the initial volume)
    evolution_rate: rate of the evolution of the volume per iteration (e.g. 0.02 = 2 % per step)
    filter_radius: radius of the filter of the sensitivities in mm
    penalty: penalty exponent of the stiffness p (E = E0 * rho^p)
    max_iterations: maximum number of iterations
    non_design_selections: list of names of selections (faces, edges, points) whose elements must not be removed
    min_density: minimum density of the void elements (e.g. 1e-3 to keep the stiffness matrix regular)
    target_step: index of the load state for the calculation of the sensitivities (only when all_load_cases = False)
    all_load_cases: True = sensitivities from the sum of the compliances of all load states
    method: optimisation method ("beso" | "simp")
    tol: convergence criterion (e.g. relative change of the compliance or the densities)
    """
    target_volume_fraction: float = 0.5
    evolution_rate: float = 0.02
    filter_radius: float = 6.0
    penalty: float = 3.0
    max_iterations: int = 50
    non_design_selections: list[str] = field(default_factory=list)
    min_density: float = 1e-3
    target_step: int = 0
    all_load_cases: bool = True
    method: str = "beso"
    tol: float = 1e-3

    def __post_init__(self) -> None:
        if not (0.0 < self.target_volume_fraction <= 1.0):
            raise ValueError(f"target_volume_fraction must be in the interval (0, 1], entered: {self.target_volume_fraction}")
        if self.evolution_rate <= 0.0 or self.evolution_rate > 1.0:
            raise ValueError(f"evolution_rate must be in the interval (0, 1], entered: {self.evolution_rate}")
        if self.filter_radius <= 0.0:
            raise ValueError(f"filter_radius must be > 0, entered: {self.filter_radius}")
        if self.penalty < 1.0:
            raise ValueError(f"penalty exponent must be >= 1.0, entered: {self.penalty}")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, entered: {self.max_iterations}")
        if self.min_density < 0.0:
            raise ValueError(f"min_density must be >= 0.0, entered: {self.min_density}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TopologyOptSpec":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
