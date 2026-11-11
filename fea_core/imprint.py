"""Splitting a face for loads and supports (Split Face / Partition / Imprint) (A10).

Makes it possible to create an imprint (a circle Ø20 for a punch, a rectangle for a foot, a polygon for a pad of any outline)
on a planar face of a body. The imprint is fragmented into the surface of the body through OpenCascade (occ.fragment in Gmsh)
and becomes a full face with its own name in the selections (Named Selection), to which a pressure, a force, supports and
contacts can be applied.

Face frame (rectangle axes, polygon 2D coordinates): n = unit normal, x' = n × (1, 0, 0) (or n × (0, 1, 0) when n is close to
the X axis), y' = n × x' – the same frame as the mesh worker uses for the rectangle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SHAPES = ("circle", "rectangle", "polygon")


class ImprintError(ValueError):
    """Invalid imprint definition."""


def face_frame(normal) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n, x', y') of the imprint frame for the face normal (see the module docstring)."""
    n = np.asarray(normal, float)
    nn = float(np.linalg.norm(n))
    n = n / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0])
    xa = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    xa /= np.linalg.norm(xa)
    ya = np.cross(n, xa)
    ya /= np.linalg.norm(ya)
    return n, xa, ya


def polygon_area_centroid(pts2d) -> tuple[float, np.ndarray]:
    """Signed area (shoelace) and centroid of a simple polygon in 2D."""
    p = np.asarray(pts2d, float)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    a = 0.5 * float(cr.sum())
    if abs(a) < 1e-300:
        return 0.0, p.mean(axis=0)
    cx = float(((x + xn) * cr).sum()) / (6.0 * a)
    cy = float(((y + yn) * cr).sum()) / (6.0 * a)
    return a, np.array([cx, cy])


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def check_polygon(pts2d) -> None:
    """ImprintError for fewer than 3 points, repeated points, zero area or a self-intersecting outline."""
    p = np.asarray(pts2d, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        raise ImprintError("A polygon imprint needs at least 3 points [u, v]")
    if not np.all(np.isfinite(p)):
        raise ImprintError("Polygon points must be finite numbers")
    if len(p) > 200:
        raise ImprintError("A polygon imprint may have at most 200 points")
    size = float(np.ptp(p, axis=0).max())
    d = np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1)
    if size <= 0 or d.min() <= 1e-9 * size:
        raise ImprintError("Polygon points must not repeat (consecutive points coincide)")
    area, _ = polygon_area_centroid(p)
    if abs(area) <= 1e-9 * size * size:
        raise ImprintError("The polygon has zero area (points on a line)")
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) in (1, n - 1):
                continue
            if _segments_cross(p[i], p[(i + 1) % n], p[j], p[(j + 1) % n]):
                raise ImprintError("The polygon outline intersects itself")


@dataclass
class Imprint:
    """Definition of an imprint on a face of a part.

    name: name of the created face selection (e.g. 'otisk_razniku')
    part: name of the part on which the imprint lies
    center: centre of the imprint [x, y, z] in global coordinates [mm] (polygon: origin of the 2D points)
    normal: normal of the face [nx, ny, nz]
    shape: 'circle' | 'rectangle' | 'polygon'
    radius: radius of the circular imprint [mm] (for shape='circle')
    dx: length of the rectangle [mm] (for shape='rectangle')
    dy: width of the rectangle [mm] (for shape='rectangle')
    size: optional size of the mesh elements on the imprint [mm] (None = global)
    points: polygon outline [[u, v], …] in the face frame around `center` [mm] (for shape='polygon')
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
    points: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Dictionary as written in the study JSON; `points` only for a polygon (older files are written unchanged)."""
        d = {
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
        if self.shape == "polygon":
            d["points"] = [[float(u), float(v)] for u, v in self.points]
        return d

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
            points=[[float(u), float(v)] for u, v in d.get("points") or []],
        )

    @classmethod
    def polygon_from_3d(cls, name: str, part: str, points3d, normal=None, size: float | None = None) -> "Imprint":
        """Polygon imprint from 3D points (e.g. picked on the face): normal = given, otherwise the best-fit plane (Newell);
        centre = centroid of the points; the points are projected onto the plane and stored as 2D points of the face frame."""
        p = np.asarray(points3d, float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3:
            raise ImprintError("A polygon imprint needs at least 3 points [x, y, z]")
        c = p.mean(axis=0)
        if normal is None:
            nw = np.zeros(3)
            for a, b in zip(p, np.roll(p, -1, axis=0)):
                nw += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
            if np.linalg.norm(nw) < 1e-12:
                raise ImprintError("The points lie on a line – give the face normal")
            normal = nw
        n, xa, ya = face_frame(normal)
        q = p - c
        off = np.abs(q @ n).max()
        span = float(np.ptp(p, axis=0).max())
        if off > 1e-3 * max(span, 1e-9) + 1e-6:
            raise ImprintError(f"The points do not lie in one plane (deviation {off:.3g} mm) – a polygon imprint lies on a planar face")
        pts = np.column_stack([q @ xa, q @ ya])
        imp = cls(name=name, part=part, center=[float(v) for v in c], normal=[float(v) for v in n], shape="polygon",
                  size=size, points=[[float(u), float(v)] for u, v in pts])
        imp.validate()
        return imp

    def validate(self) -> None:
        if self.shape not in SHAPES:
            raise ImprintError(f"Unknown imprint shape '{self.shape}' ({' | '.join(SHAPES)})")
        if self.shape == "circle" and not self.radius > 0:
            raise ImprintError("The imprint radius must be positive")
        if self.shape == "rectangle" and not (self.dx > 0 and self.dy > 0):
            raise ImprintError("The rectangle dimensions must be positive")
        if self.shape == "polygon":
            check_polygon(self.points)

    def points_3d(self) -> np.ndarray:
        """Polygon outline in global coordinates [mm]."""
        _, xa, ya = face_frame(self.normal)
        c = np.asarray(self.center, float)
        p = np.asarray(self.points, float).reshape(-1, 2)
        return c + p[:, :1] * xa + p[:, 1:2] * ya

    def area(self) -> float:
        if self.shape == "circle":
            return float(np.pi * self.radius ** 2)
        if self.shape == "rectangle":
            return float(self.dx * self.dy)
        return abs(polygon_area_centroid(self.points)[0])

    def to_mesh_dict(self) -> dict:
        """Input for the mesh generator (worker.py): `to_dict` + the polygon outline in global coordinates (`points3d`)."""
        d = self.to_dict()
        if self.shape == "polygon":
            d["points3d"] = self.points_3d().tolist()
        return d
