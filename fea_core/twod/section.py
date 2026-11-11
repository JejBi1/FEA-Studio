"""Section of the 3D parts by a plane / half-plane (OCP, main process) and conversion to the plane x-y for the 2D mesh.

Procedure: the planar section face (axisym: the half-plane r ≥ 0 containing the axis and the radial direction xdir; planar: the plane of the normal
`axis` through origin) over the envelope of the part → BRepAlgoAPI_Common(body, face) = the part of the face inside the body → gp_Trsf
(SetTransformation(gp_Ax3(origin, n, e1)) = coordinates in the frame of the section: x = e1, y = e2, z = n ≈ 0) → BREP for Gmsh.

Edges of the selections: an edge of the section belongs to the selection when ALL its sampling points (converted back to 3D) lie on the faces of the
selection (distance < tol – exact geometry of OCP / Gmsh, not the nodes of the mesh; a tangentially adjoining face is thus not matched).
A planar face of the selection lying in the plane of the section (a plane of symmetry containing the section) is left out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import Frame2D, TwoDError


@dataclass
class SectionPart:
    name: str
    brep: str | None                      # BREP of the faces of the section in the plane x-y (None = the part does not intersect the section)
    faces: int = 0
    area: float = 0.0                     # area of the section [mm²]
    centroid: list[float] = field(default_factory=lambda: [0.0, 0.0])   # centroid of the section in 2D
    volume: float = 0.0                   # volume of the 3D part [mm³]
    model_volume: float = 0.0             # volume that the 2D model represents (axisym 2π·r_c·A, planar A·t)


def _ax3(fr: Frame2D):
    from OCP.gp import gp_Ax3, gp_Dir, gp_Pnt
    return gp_Ax3(gp_Pnt(*map(float, fr.origin)), gp_Dir(*map(float, fr.n)), gp_Dir(*map(float, fr.e1)))


def cutting_face(fr: Frame2D, box: np.ndarray):
    """Planar section face covering the envelope box (6,) with a reserve; for axisym only r ≥ 0."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pln
    lo, hi = box[:3], box[3:]
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    corners = np.vstack([corners, fr.origin])
    uv = fr.to2d(corners)
    margin = 0.05 * float(np.linalg.norm(hi - lo)) + 1.0
    if fr.axisym:
        umin, umax = 0.0, float(uv[:, 0].max()) + margin
    else:
        umin, umax = float(uv[:, 0].min()) - margin, float(uv[:, 0].max()) + margin
    vmin, vmax = float(uv[:, 1].min()) - margin, float(uv[:, 1].max()) + margin
    mk = BRepBuilderAPI_MakeFace(gp_Pln(_ax3(fr)), umin, umax, vmin, vmax)
    if not mk.IsDone():
        raise TwoDError("2D: the section face cannot be created")
    return mk.Face()


def section_shape(shape, fr: Frame2D):
    """Faces of the section of the body (3D position) → (TopoDS_Shape with the faces, list of the faces)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from ..geometry import bbox, list_faces
    face = cutting_face(fr, bbox(shape))
    op = BRepAlgoAPI_Common(shape, face)
    op.Build()
    if not op.IsDone():
        raise TwoDError("2D: intersecting the part with the section plane failed (OCCT)")
    res = op.Shape()
    return res, list_faces(res)


def to_plane(shape, fr: Frame2D):
    """Shape in 3D → coordinates of the frame of the section (x = e1, y = e2, z = n)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    T = gp_Trsf()
    T.SetTransformation(_ax3(fr))
    return BRepBuilderAPI_Transform(shape, T, True).Shape()


def _surface_props(shape) -> tuple[float, np.ndarray]:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    p = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, p)
    c = p.CentreOfMass()
    return float(p.Mass()), np.array([c.X(), c.Y(), c.Z()])


def section_parts(parts, fr: Frame2D, workdir: str | Path, thickness: float = 1.0) -> list[SectionPart]:
    """Section of all the parts; BREP (plane x-y) into workdir/twod_part_XXX.brep. A part without an intersection has brep None."""
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    from ..geometry import write_brep
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, p in enumerate(parts):
        res, faces = section_shape(p.shape, fr)
        sp = SectionPart(p.name, None, volume=float(p.volume()))
        if faces:
            comp = TopoDS_Compound()
            b = BRep_Builder()
            b.MakeCompound(comp)
            for f in faces:
                b.Add(comp, f)
            flat = to_plane(comp, fr)
            area, c = _surface_props(flat)
            if area > 1e-12:
                path = workdir / f"twod_part_{i:03d}.brep"
                write_brep(flat, path)
                sp.brep, sp.faces, sp.area = str(path), len(faces), area
                sp.centroid = [float(c[0]), float(c[1])]
                sp.model_volume = (2 * np.pi * float(c[0]) * area) if fr.axisym else area * float(thickness)
        out.append(sp)
    return out


def plane_edges(brep: str) -> list[np.ndarray]:
    """Edges of the faces of the section (BREP in the plane x-y): sampling points (7, 2) – parameters 0, 1/6 … 1 (for the refinement of the mesh)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape
    from ..geometry import list_edges
    shape = TopoDS_Shape()
    BRepTools.Read_s(shape, str(brep), BRep_Builder())
    out = []
    for e in list_edges(shape):
        ad = BRepAdaptor_Curve(e)
        t0, t1 = ad.FirstParameter(), ad.LastParameter()
        pts = []
        for a in np.linspace(0.0, 1.0, 7):
            q = ad.Value(t0 + a * (t1 - t0))
            pts.append([q.X(), q.Y()])
        out.append(np.array(pts))
    return out


# ------------------------------------------------------------------ edges on the faces of the selections

class FaceSet:
    """Faces of OCP of the selections for the test "does the point lie on the face": envelopes + BRepExtrema (geometry.project_to_surface)."""

    def __init__(self, faces: list, fr: Frame2D, tol: float):
        from ..geometry import bbox
        self.tol = tol
        self.faces, self.boxes, self.skipped = [], [], 0
        for f in faces:
            if _in_section_plane(f, fr, tol):
                self.skipped += 1
                continue
            self.faces.append(f)
            self.boxes.append(bbox(f))

    def contains(self, p: np.ndarray) -> bool:
        from ..geometry import project_to_surface
        for f, b in zip(self.faces, self.boxes):
            if np.any(p < b[:3] - self.tol) or np.any(p > b[3:] + self.tol):
                continue
            try:
                _, d = project_to_surface(f, p)
            except ValueError:
                continue
            if d <= self.tol:
                return True
        return False

    def contains_all(self, pts: np.ndarray) -> bool:
        return bool(self.faces) and all(self.contains(p) for p in pts)


def _in_section_plane(face, fr: Frame2D, tol: float) -> bool:
    """Planar face lying in the plane of the section (e.g. the plane of symmetry in which the section is led)."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane
    s = BRepAdaptor_Surface(face)
    if s.GetType() != GeomAbs_Plane:
        return False
    pl = s.Plane()
    d, loc = pl.Axis().Direction(), pl.Location()
    nrm = np.array([d.X(), d.Y(), d.Z()])
    p = np.array([loc.X(), loc.Y(), loc.Z()])
    return abs(abs(float(nrm @ fr.n)) - 1.0) < 1e-9 and abs(float((p - fr.origin) @ fr.n)) <= tol


def selection_faces(study, parts, names) -> dict[str, list]:
    """Faces of OCP of the selections of the study (selectors.resolve) – only the parts in `parts`. A selection without faces → an empty list."""
    from ..selectors import resolve
    cache: dict = {}
    out = {}
    for name in names:
        sel = study.selections.get(name)
        faces = []
        if sel is not None:
            for fs in sel.faces:
                hit = resolve(fs, parts, cache)
                if hit is not None:
                    faces.append(parts[hit[0]].faces()[hit[1]])
        out[name] = faces
    return out
