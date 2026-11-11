"""Settings of the 2D calculation (section of the 3D study) and the conversion of the coordinates section plane ↔ 3D.

Kinds of problems (elements of ccx):
  axisym        axisymmetric problem (CAX6): the half-plane of the section contains the axis of rotation `axis` through the point `origin`, the radial
                direction `xdir` (component ⟂ to the axis). 2D x = radius r ≥ 0, y = coordinate along the axis. Loads and reactions
                OVER THE WHOLE CIRCUMFERENCE (ccx – verified scratch/twod/exp1.py). Stresses: SXX = σr, SYY = axial σ, SZZ = σθ.
  plane_stress  plane stress (CPS6), plane_strain plane strain (CPE6): the plane of the section through the point `origin`
                with the normal `axis`, axis x = `xdir` (projection into the plane), y = axis × x. Thickness `thickness` [mm].

Frame (Frame2D): e1 = 2D axis x, e2 = 2D axis y, n = e1 × e2 (perpendicular to the plane of the section; for axisym the circumferential direction in the plane of the section
with the opposite sign to a × e_r – the sign does not matter). 3D point of the plane = origin + x·e1 + y·e2.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

KINDS = ("axisym", "plane_stress", "plane_strain")
ELEMENT = {"axisym": "CAX6", "plane_stress": "CPS6", "plane_strain": "CPE6"}
KIND_LABEL = {"axisym": "axisymmetric problem", "plane_stress": "plane stress",
              "plane_strain": "plane strain"}
AXIS_LABELS = {"axisym": ("r", "z"), "plane_stress": ("x", "y"), "plane_strain": ("x", "y")}


class TwoDError(ValueError):
    """Clear error of the 2D calculation (a setting that cannot be done in 2D)."""


@dataclass
class TwoDSettings:
    enabled: bool = False
    kind: str = "axisym"                  # axisym | plane_stress | plane_strain
    origin: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])   # axisym: axis of rotation; planar: normal of the plane
    xdir: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])   # axisym: radial direction of the half-plane; planar: axis x
    thickness: float = 1.0                # thickness of the planar problem [mm] (axisym does not use it)
    parts: list[str] = field(default_factory=list)                       # empty = all meshed parts
    size: float | None = None             # velikost prvku [mm]; None = Study.mesh.size

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "TwoDSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


def _unit(v, what: str) -> np.ndarray:
    v = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        raise TwoDError(f"2D: {what} has zero length")
    return v / n


@dataclass(frozen=True)
class Frame2D:
    kind: str
    origin: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    n: np.ndarray

    @property
    def axisym(self) -> bool:
        return self.kind == "axisym"

    def to3d(self, xy) -> np.ndarray:
        """2D points (…, 2) → 3D points in the plane of the section (…, 3)."""
        xy = np.asarray(xy, float)
        return self.origin + xy[..., :1] * self.e1 + xy[..., 1:2] * self.e2

    def to2d(self, p) -> np.ndarray:
        """3D points (…, 3) → 2D. Axisym: (distance from the axis, coordinate along the axis) – also outside the half-plane of the section."""
        d = np.asarray(p, float) - self.origin
        y = d @ self.e2
        if self.axisym:
            radial = d - y[..., None] * self.e2
            return np.stack([np.linalg.norm(radial, axis=-1), y], axis=-1)
        return np.stack([d @ self.e1, y], axis=-1)

    def vec2d(self, v) -> np.ndarray:
        """Components of a 3D vector in the plane of the section (e1, e2) – for axisym in the half-plane of the section (radial, axial)."""
        v = np.asarray(v, float)
        return np.stack([v @ self.e1, v @ self.e2], axis=-1)

    def vec3d(self, v2) -> np.ndarray:
        v2 = np.asarray(v2, float)
        return v2[..., :1] * self.e1 + v2[..., 1:2] * self.e2


def make_frame(s: TwoDSettings) -> Frame2D:
    if s.kind not in KINDS:
        raise TwoDError(f"2D: unknown problem type '{s.kind}' (axisym | plane_stress | plane_strain)")
    a = _unit(s.axis, "axis" if s.kind == "axisym" else "plane normal")
    x = np.asarray(s.xdir, float).reshape(3)
    x = x - (x @ a) * a
    if np.linalg.norm(x) < 1e-9:
        raise TwoDError("2D: the xdir direction is parallel to the axis / plane normal – choose another direction")
    x = x / np.linalg.norm(x)
    o = np.asarray(s.origin, float).reshape(3)
    if s.kind == "axisym":
        e1, e2 = x, a
        n = np.cross(e1, e2)
    else:
        if not (float(s.thickness) > 0):
            raise TwoDError("2D: plane problem thickness must be positive")
        e1, n = x, a
        e2 = np.cross(n, e1)
    return Frame2D(s.kind, o, e1, e2, n)
