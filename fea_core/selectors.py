"""Face selectors: part + face geometry (not an index), so that they survive a change of the model.

A selector holds: the name of the part, the face type, area, centroid, a point on the face with its normal
and the envelope. Matching is done by area, centroid and envelope with a tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np

if TYPE_CHECKING:            # OCP (~1 s) is loaded only when searching faces – the study and the calculation do not need it
    from .geometry import Part


def _from_known(cls, d: dict):
    """Dataclass from a dictionary – unknown keys (a file from a newer version) are ignored."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class FaceSelector:
    part: str
    type: str
    area: float
    centroid: list[float]
    point: list[float]
    normal: list[float]
    bbox: list[float]

    @classmethod
    def from_props(cls, part: str, p: dict) -> "FaceSelector":
        return cls(part, p["type"], float(p["area"]),
                   [float(v) for v in p["centroid"]], [float(v) for v in p["point"]],
                   [float(v) for v in p["normal"]], [float(v) for v in p["bbox"]])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FaceSelector":
        return _from_known(cls, d)

    def describe(self) -> str:
        c = ", ".join(f"{v:.2f}" for v in self.centroid)
        return f"{self.part}: {self.type} A={self.area:.2f} mm² centroid=({c})"


@dataclass
class EdgeSelector:
    """Edge of a part (for an edge load): curve type, length, centroid, point at the middle of the parameter, envelope.
    Matched in the same way as a face (length instead of area, see mesh.worker).
    """
    part: str
    type: str
    length: float
    centroid: list[float]
    point: list[float]
    bbox: list[float]

    @classmethod
    def from_props(cls, part: str, p: dict) -> "EdgeSelector":
        return cls(part, p["type"], float(p["length"]), [float(v) for v in p["centroid"]],
                   [float(v) for v in p["point"]], [float(v) for v in p["bbox"]])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EdgeSelector":
        return _from_known(cls, d)

    def describe(self) -> str:
        c = ", ".join(f"{v:.2f}" for v in self.centroid)
        return f"{self.part}: edge {self.type} L={self.length:.2f} mm centroid=({c})"


@dataclass
class PointSelector:
    """Point on the surface of a part (vertex or any point of a face / edge) for a load at a point. Meshing projects the point
    onto the nearest face of the part and inserts it into the mesh as a node (occ.fragment, see mesh.worker).
    """
    part: str
    point: list[float]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PointSelector":
        return cls(str(d["part"]), [float(v) for v in d["point"]])

    def describe(self) -> str:
        c = ", ".join(f"{v:.2f}" for v in self.point)
        return f"{self.part}: point ({c})"


def match_score(sel: dict, cand: dict, bbox_weight: float = 1.0) -> float:
    """Measure of mismatch (0 = identical) of a face (keys area, centroid, bbox) or an edge (length instead of area; the scale
    of the distances is also the length of the edge). bbox_weight < 1 = the envelope only a weak criterion (Gmsh, see mesh.worker).
    """
    key = "length" if "length" in sel else "area"
    a0, a1 = sel[key], cand[key]
    bb = np.asarray(sel["bbox"], float)
    size = max(np.linalg.norm(bb[3:] - bb[:3]), a0 if key == "length" else 0.0, 1e-9)
    da = abs(a0 - a1) / max(a0, 1e-12)
    dc = np.linalg.norm(np.asarray(sel["centroid"]) - np.asarray(cand["centroid"])) / size
    db = np.abs(bb - np.asarray(cand["bbox"], float)).max() / size
    return max(da, dc, bbox_weight * db)


MATCH_TOL = 1e-4


def resolve(sel: FaceSelector, parts: list[Part], face_cache: dict | None = None) -> tuple[int, int] | None:
    """Finds the face of a selector in the geometry → (part index, face index), or None."""
    for pi, p in enumerate(parts):
        if p.name != sel.part:
            continue
        props = face_cache.get(pi) if face_cache is not None else None
        if props is None:
            from .geometry import face_props
            props = [face_props(f) for f in p.faces()]
            if face_cache is not None:
                face_cache[pi] = props
        sd = sel.to_dict()
        scored = sorted((match_score(sd, fp), fi) for fi, fp in enumerate(props))
        if scored and scored[0][0] < MATCH_TOL:
            return pi, scored[0][1]
    return None


def in_planes(bb, xyz, tol: float) -> bool:
    """The envelope bb lies entirely in the planes x=..., y=..., z=... (None = no condition)."""
    return all(v is None or (abs(bb[k] - v) <= tol * max(1, abs(v)) + 1e-7
                             and abs(bb[k + 3] - v) <= tol * max(1, abs(v)) + 1e-7)
               for k, v in enumerate(xyz))


def _radius_ok(r, radius) -> bool:
    return r is not None and abs(r - radius) <= 1e-6 * max(1, radius)


def face_ok(p: dict, type=None, normal=None, xyz=(None, None, None), radius=None, tol: float = 1e-6) -> bool:
    """Basic conditions for finding faces: type, outward normal of a planar face (unit), planes of the envelope, radius
    of a cylinder (p["radius"], None = not a cylinder).
    """
    if type is not None and p["type"] != type:
        return False
    if normal is not None and (p["type"] != "plane" or float(np.dot(p["normal"], normal)) < 1 - 1e-6):
        return False
    if not in_planes(p["bbox"], xyz, tol):
        return False
    return radius is None or _radius_ok(p.get("radius"), radius)


def edge_ok(p: dict, type=None, xyz=(None, None, None), radius=None, direction=None, tol: float = 1e-6) -> bool:
    """Basic conditions for finding edges: type, planes of the envelope, radius of the circle, parallelism of the segment with a direction
    (unit vector).
    """
    if type is not None and p["type"] != type:
        return False
    if not in_planes(p["bbox"], xyz, tol):
        return False
    if radius is not None and not _radius_ok(p.get("radius"), radius):
        return False
    if direction is not None:
        t = np.asarray(p["end"]) - np.asarray(p["start"])
        if p["type"] != "line" or np.linalg.norm(t) == 0 or \
                abs(abs(float(direction @ t)) / np.linalg.norm(t) - 1) > 1e-6:
            return False
    return True


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def find_faces(parts: Iterable[Part], part: str | None = None, type: str | None = None,
               normal: Iterable[float] | None = None, x: float | None = None,
               y: float | None = None, z: float | None = None, radius: float | None = None,
               where: Callable[[dict], bool] | None = None, tol: float = 1e-6) -> list[FaceSelector]:
    """Selects faces according to conditions (for scripts, the CLI and verification tasks).

    normal  – outward normal at a point of the face (planar faces)
    x, y, z – the face lies entirely in the plane x=..., y=..., z=... (according to the envelope)
    radius  – cylindrical face of the given radius (it is not derived from the envelope / area,
              it is computed through OCC)
    """
    from .geometry import cylinder_axis, face_props
    n = None if normal is None else unit(normal)
    out = []
    for p in parts:
        if part is not None and p.name != part:
            continue
        for f in p.faces():
            props = face_props(f)
            if radius is not None:
                cyl = cylinder_axis(f)
                props["radius"] = None if cyl is None else cyl[2]
            if not face_ok(props, type, n, (x, y, z), radius, tol):
                continue
            if where is not None and not where(props):
                continue
            out.append(FaceSelector.from_props(p.name, props))
    return out


def resolve_edge(sel: EdgeSelector, parts: list[Part], edge_cache: dict | None = None) -> tuple[int, int] | None:
    """Finds the edge of a selector in the geometry → (part index, index of the edge in geometry.list_edges), or None."""
    from .geometry import edge_props, list_edges
    for pi, p in enumerate(parts):
        if p.name != sel.part:
            continue
        props = edge_cache.get(pi) if edge_cache is not None else None
        if props is None:
            props = [edge_props(e) for e in list_edges(p.shape)]
            if edge_cache is not None:
                edge_cache[pi] = props
        sd = sel.to_dict()
        scored = sorted((match_score(sd, ep), ei) for ei, ep in enumerate(props))
        if scored and scored[0][0] < MATCH_TOL:
            return pi, scored[0][1]
    return None


def find_edges(parts: Iterable[Part], part: str | None = None, type: str | None = None,
               x: float | None = None, y: float | None = None, z: float | None = None,
               radius: float | None = None, direction: Iterable[float] | None = None,
               where: Callable[[dict], bool] | None = None, tol: float = 1e-6) -> list[EdgeSelector]:
    """Selects edges according to conditions (scripts, MCP, verification tasks).

    x, y, z   – the edge lies entirely in the plane x=..., y=..., z=... (according to the envelope); two coordinates = a line
    radius    – circle / arc of the given radius
    direction – segment parallel to a direction
    """
    from .geometry import edge_props, list_edges
    d = None if direction is None else unit(direction)
    out = []
    for p in parts:
        if part is not None and p.name != part:
            continue
        for e in list_edges(p.shape):
            props = edge_props(e)
            if not edge_ok(props, type, (x, y, z), radius, d, tol):
                continue
            if where is not None and not where(props):
                continue
            out.append(EdgeSelector.from_props(p.name, props))
    return out
