"""Splitting a face for loads and supports (Split Face / Partition / Imprint) (A10).

Makes it possible to create an imprint (e.g. a circle Ø20 for a punch, a rectangle for a foot, a washer)
on any planar face of a body.
The imprint is fragmented into the surface of the body through OpenCascade (occ.fragment in Gmsh)
and becomes a full face with its own name in the selections (Named Selection),
to which a pressure, a force, supports and contacts can be applied.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Imprint:
    """Definition of an imprint on a face of a part.

    name: name of the created face selection (e.g. 'otisk_razniku')
    part: name of the part on which the imprint lies
    center: centre of the imprint [x, y, z] in global coordinates [mm]
    normal: normal of the face [nx, ny, nz]
    shape: 'circle' | 'rectangle'
    radius: radius of the circular imprint [mm] (for shape='circle')
    dx: length of the rectangle [mm] (for shape='rectangle')
    dy: width of the rectangle [mm] (for shape='rectangle')
    size: optional size of the mesh elements on the imprint [mm] (None = global)
    """
    name: str
    part: str
    center: list[float]
    normal: list[float]
    shape: str = "circle"
    radius: float = 10.0
    dx: float = 20.0
    dy: float = 20.0
    size: float | None = None

    def to_dict(self) -> dict:
        """Dictionary identical to `dataclasses.asdict` (this is how Study.to_dict writes it), only with numbers of type float."""
        return {
            "name": self.name,
            "part": self.part,
            "center": [float(v) for v in self.center],
            "normal": [float(v) for v in self.normal],
            "shape": self.shape,
            "radius": float(self.radius),
            "dx": float(self.dx),
            "dy": float(self.dy),
            "size": float(self.size) if self.size is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Imprint:
        return cls(
            name=d["name"],
            part=d["part"],
            center=list(d["center"]),
            normal=list(d["normal"]),
            shape=d.get("shape", "circle"),
            radius=float(d.get("radius", 10.0)),
            dx=float(d.get("dx", 20.0)),
            dy=float(d.get("dy", 20.0)),
            size=float(d["size"]) if d.get("size") is not None else None,
        )

    def to_mesh_dict(self) -> dict:
        """Input for the mesh generator (worker.py) – the same dictionary as `to_dict` (it also matches Study.to_dict)."""
        return self.to_dict()
