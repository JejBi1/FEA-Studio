"""Settings of the nonlinear buckling analysis with a geometric imperfection (F9 C2, GNIA / GMNIA): Study.nlbuckling."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

IMPERFECTION_KINDS = ("modes", "bow", "file", "none")


@dataclass
class NonlinearBucklingSettings:
    """Nonlinear buckling (NLGEOM, load control with automatic increments) of an imperfect copy of the mesh.

    case        load case whose loads are multiplied by the load factor λ (None = the case of the linear buckling, else the first)
    imperfection modes = combination of linear buckling modes of the case | bow = half-sine bow between points a, b in the
                direction `bow_direction` | file = nodal shape from a CSV (x, y, z, ux, uy, uz) | none = perfect geometry
    modes, weights  1-based buckling modes and their coefficients (sign included); the shape is scaled to max |w| = e0
    amplitude   e0 [mm] (> 0) – otherwise e0 = L / amplitude_ratio with L = `length` (None = largest bounding-box dimension)
    sign        +1 / −1: direction of the imperfection
    max_factor  λ_max of the load ramp (None = 1.1 × the first linear buckling factor)
    increments  initial number of increments to λ_max (initial increment 1/increments, maximum 2/increments)
    max_increments, min_increment  ccx limits (min. increment as a fraction of the ramp)
    monitor_point, monitor_dir  monitored node (nearest mesh node) and direction (None = node of max |w|, direction of w there)
    frames      number of deformed shapes kept for display (evenly over the increments + the last one)
    """
    enabled: bool = False
    case: str | None = None
    imperfection: str = "modes"
    modes: list = field(default_factory=lambda: [1])
    weights: list = field(default_factory=lambda: [1.0])
    amplitude: float = 0.0
    amplitude_ratio: float = 300.0
    length: float | None = None
    sign: float = 1.0
    bow_a: list | None = None
    bow_b: list | None = None
    bow_direction: list | None = None
    shape_file: str | None = None
    max_factor: float | None = None
    increments: int = 20
    max_increments: int = 400
    min_increment: float = 1e-5
    monitor_point: list | None = None
    monitor_dir: list | None = None
    frames: int = 12

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "NonlinearBucklingSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


class ImperfectionError(RuntimeError):
    pass
