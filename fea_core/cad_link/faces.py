"""Properties of the faces and edges of the parts for the remapping (cache over geometry.face_props / edge_props).

An extension compared with the selector: axis and radius of a cylinder / cone / sphere, plane (normal, distance from the origin) –
these are used to find the "same" face after a change of the dimensions.
"""
from __future__ import annotations

import numpy as np


def _surface_extra(face) -> dict:
    """Axis / radius / plane of the face (BRepAdaptor_Surface)."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_Torus
    s = BRepAdaptor_Surface(face)
    t = s.GetType()

    def xyz(p):
        return np.array([p.X(), p.Y(), p.Z()])
    out: dict = {}
    if t == GeomAbs_Plane:
        pl = s.Plane()
        ax = pl.Axis()
        out["axis_point"], out["axis"] = xyz(ax.Location()), xyz(ax.Direction())
    elif t in (GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Torus):
        g = {GeomAbs_Cylinder: s.Cylinder, GeomAbs_Cone: s.Cone, GeomAbs_Torus: s.Torus}[t]()
        ax = g.Axis()
        out["axis_point"], out["axis"] = xyz(ax.Location()), xyz(ax.Direction())
        if t == GeomAbs_Cylinder:
            out["radius"] = float(g.Radius())
        elif t == GeomAbs_Cone:
            out["radius"] = float(g.RefRadius())
            out["angle"] = float(g.SemiAngle())
        else:
            out["radius"] = float(g.MajorRadius())
            out["radius2"] = float(g.MinorRadius())
    elif t == GeomAbs_Sphere:
        sp = s.Sphere()
        out["axis_point"] = xyz(sp.Location())
        out["radius"] = float(sp.Radius())
    return out


class PartFaces:
    """Faces and edges of one part: list of properties (lazily), length scale = diagonal of the envelope of the part."""

    def __init__(self, part):
        from ..geometry import bbox
        self.part = part
        self.name = part.name
        bb = bbox(part.shape)
        self.bbox = bb
        self.size = max(float(np.linalg.norm(bb[3:] - bb[:3])), 1e-9)
        self._faces = None
        self._face_objs = None
        self._edges = None
        self._edge_objs = None

    @property
    def face_objs(self) -> list:
        if self._face_objs is None:
            self._face_objs = self.part.faces()
        return self._face_objs

    @property
    def faces(self) -> list[dict]:
        if self._faces is None:
            from ..geometry import face_props
            out = []
            for f in self.face_objs:
                d = face_props(f)
                try:
                    d.update(_surface_extra(f))
                except Exception:  # noqa: BLE001 – unusual face: only the basic properties
                    pass
                out.append(d)
            self._faces = out
        return self._faces

    @property
    def edge_objs(self) -> list:
        if self._edge_objs is None:
            from ..geometry import list_edges
            self._edge_objs = list_edges(self.part.shape)
        return self._edge_objs

    @property
    def edges(self) -> list[dict]:
        if self._edges is None:
            from ..geometry import edge_props
            self._edges = [edge_props(e) for e in self.edge_objs]
        return self._edges


class Geometry:
    """Parts of one version of the geometry with a cache of the properties of the faces and edges."""

    def __init__(self, parts: list):
        self.parts = list(parts)
        self._pf: dict[int, PartFaces] = {}
        self.index = {p.name: i for i, p in enumerate(self.parts)}

    def part(self, name_or_index) -> PartFaces | None:
        i = self.index.get(name_or_index) if isinstance(name_or_index, str) else name_or_index
        if i is None or not 0 <= i < len(self.parts):
            return None
        if i not in self._pf:
            self._pf[i] = PartFaces(self.parts[i])
        return self._pf[i]

    @property
    def size(self) -> float:
        """Diagonal of the whole assembly."""
        if not self.parts:
            return 1.0
        from ..geometry import bbox
        bb = np.array([bbox(p.shape) for p in self.parts])
        return max(float(np.linalg.norm(bb[:, 3:].max(0) - bb[:, :3].min(0))), 1e-9)
