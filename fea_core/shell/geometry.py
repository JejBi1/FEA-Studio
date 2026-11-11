"""Geometry of the shells (OCP): recognition of the plate / sheet in the body, mid face, planar parts from STEP,
extension of the mid face towards the mid surface of the neighbouring part (T joint).

Plate = a body with two parallel planar faces of opposite normal (largest area) at the distance t;
"skin" = the faces connected to them tangentially (G1) – for a sheet with bends also the cylindrical faces of the bends. Check of the constant thickness:
volume V = t·(A_upper + A_lower)/2 (holds exactly for planar plates and cylindrical bends, holes do not matter). Mid face:
planar plate = the upper face shifted by −t/2·n, otherwise an offset of the upper skin (BRepOffsetAPI_MakeOffsetShape, simple).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_Transform, BRepBuilderAPI_MakeVertex
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.GeomAbs import GeomAbs_Plane
from OCP.GeomLProp import GeomLProp_SLProps
from OCP.GProp import GProp_GProps
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.TopAbs import TopAbs_EDGE, TopAbs_IN, TopAbs_ON, TopAbs_REVERSED, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape
from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Trsf, gp_Vec

from ..geometry import Part, bbox, face_props, list_edges, list_faces, xyz_of
from .model import OFFSETS, ShellError

TANGENT_COS = np.cos(np.radians(1.0))     # tangential (G1) connection of the faces of the skin
MIN_SLENDER = 5.0                         # √A / t below this limit it is not a plate
THICK_SLENDER = 10.0                      # √A / t below this limit a warning (thick plate)


@dataclass
class SheetInfo:
    """Part of the shell model: mid face + thickness + division of the faces of the original body."""
    part: str
    kind: str                         # plate | sheet | surface
    thickness: float
    mid: TopoDS_Shape                 # mid (reference) face – compound of faces
    top: list[int] = field(default_factory=list)      # indices of the faces of the body (list_faces) of the upper skin (normal outwards)
    bottom: list[int] = field(default_factory=list)   # lower skin
    sides: list[int] = field(default_factory=list)    # side faces (edges of the shell)
    normal: np.ndarray | None = None  # planar plate: outward normal of the upper face (= normal of the shell)
    offset: float = 0.0               # ccx OFFSET (planar parts from STEP), for solids always 0 (mid face)
    mid_area: float = 0.0
    warnings: list[str] = field(default_factory=list)
    extensions: list[TopoDS_Shape] = field(default_factory=list)   # extension towards the neighbouring parts (T joint)

    def mid_shape(self) -> TopoDS_Shape:
        """Mid face including the extension (for meshing)."""
        return _compound(list_faces(self.mid) + [f for e in self.extensions for f in list_faces(e)])


# ------------------------------------------------------------------ OCP helpers

def _compound(shapes) -> TopoDS_Compound:
    c = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(c)
    for s in shapes:
        b.Add(c, s)
    return c


def _area(shape) -> float:
    p = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, p)
    return p.Mass()


def _volume(shape) -> float:
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, p)
    return p.Mass()


def face_normal_at(face, point, tol: float = 1e-7):
    """(outward normal respecting the orientation of the face, distance of the point from the face, position relative to the boundaries of the face)
    for a point projected onto the face. Position: True = the projection lies on the face (IN/ON), False = outside (beyond the boundary).
    """
    surf = BRep_Tool.Surface_s(face)
    sas = ShapeAnalysis_Surface(surf)
    p = gp_Pnt(*map(float, point))
    uv = sas.ValueOfUV(p, tol)
    q = surf.Value(uv.X(), uv.Y())
    pr = GeomLProp_SLProps(surf, uv.X(), uv.Y(), 1, 1e-9)
    if pr.IsNormalDefined():
        n = pr.Normal()
        nn = np.array([n.X(), n.Y(), n.Z()])
    else:
        nn = np.zeros(3)
    if face.Orientation() == TopAbs_REVERSED:
        nn = -nn
    dist = float(np.linalg.norm(xyz_of(q) - np.asarray(point, float)))
    return nn, dist, uv


def on_face(face, point, tol: float) -> tuple[bool, float, np.ndarray]:
    """Does the projection of the point lie inside the face (with a tolerance)? → (inside, distance from the face, outward normal)."""
    n, dist, uv = face_normal_at(face, point)
    st = BRepClass_FaceClassifier(face, gp_Pnt2d(uv.X(), uv.Y()), max(tol, 1e-7)).State()
    return st in (TopAbs_IN, TopAbs_ON), dist, n


def _edge_face_map(shape) -> dict:
    """hash(edge) → [(edge, [indices of the faces])] (OCP 8 has no IndexedDataMapOfShapeListOfShape)."""
    out: dict[int, list] = {}
    for fi, f in enumerate(list_faces(shape)):
        exp = TopExp_Explorer(f, TopAbs_EDGE)
        while exp.More():
            e = TopoDS.Edge(exp.Current())
            bucket = out.setdefault(hash(e), [])
            for item in bucket:
                if item[0].IsSame(e):
                    if fi not in item[1]:
                        item[1].append(fi)
                    break
            else:
                bucket.append((e, [fi]))
            exp.Next()
    return out


def _edge_mid(edge) -> np.ndarray:
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    ad = BRepAdaptor_Curve(edge)
    return xyz_of(ad.Value(0.5 * (ad.FirstParameter() + ad.LastParameter())))


def _grow_skin(faces, start: int, emap, exclude: set[int]) -> list[int]:
    """Faces connected tangentially to the face start (the same outward normal at the middle of the common edge)."""
    skin, todo = {start}, [start]
    adj: dict[int, list[tuple]] = {}
    for bucket in emap.values():
        for e, fl in bucket:
            if len(fl) == 2:
                a, b = fl
                adj.setdefault(a, []).append((b, e))
                adj.setdefault(b, []).append((a, e))
    while todo:
        fi = todo.pop()
        for fj, e in adj.get(fi, []):
            if fj in skin or fj in exclude:
                continue
            m = _edge_mid(e)
            n1, _, _ = face_normal_at(faces[fi], m)
            n2, _, _ = face_normal_at(faces[fj], m)
            if float(n1 @ n2) >= TANGENT_COS:
                skin.add(fj)
                todo.append(fj)
    return sorted(skin)


def _translate(shape, vec) -> TopoDS_Shape:
    tr = gp_Trsf()
    tr.SetTranslation(gp_Vec(*map(float, vec)))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


# ------------------------------------------------------------------ recognition of a plate

def _partners(props, faces, f0) -> list[tuple[int, float]]:
    """Opposite faces to f0 and the thickness: plane → parallel planes with the opposite normal behind it; cylinder → coaxial
    cylinders of another radius. Sorted by area, descending.
    """
    from ..geometry import cylinder_axis
    p0 = props[f0]
    out = []
    if p0["type"] == "plane":
        n0 = np.asarray(p0["normal"], float)
        for j, pj in enumerate(props):
            if j == f0 or pj["type"] != "plane" or float(np.asarray(pj["normal"]) @ n0) > -1 + 1e-6:
                continue
            d = -float((np.asarray(pj["point"], float) - np.asarray(p0["point"], float)) @ n0)
            if d > 0:
                out.append((j, d))
    elif p0["type"] == "cylinder":
        c0, a0, r0 = cylinder_axis(faces[f0])
        for j, pj in enumerate(props):
            if j == f0 or pj["type"] != "cylinder":
                continue
            c, a, r = cylinder_axis(faces[j])
            off = (c - c0) - ((c - c0) @ a0) * a0
            if abs(abs(float(a @ a0)) - 1) < 1e-9 and np.linalg.norm(off) <= 1e-6 * max(1.0, r0) and                     abs(r - r0) > 1e-6 * r0:
                out.append((j, abs(r - r0)))
    return sorted(out, key=lambda jt: -props[jt[0]]["area"])


def detect_sheet(part: Part) -> tuple[SheetInfo | None, str]:
    """Recognises a plate / sheet of constant thickness in the body → (SheetInfo, "") or (None, reason in Czech).

    Tries the largest faces (plane / cylinder) and their opposite faces (parallel plane, coaxial cylinder).
    """
    faces = list_faces(part.shape)
    if not faces:
        return None, "part has no faces"
    props = [face_props(f) for f in faces]
    order = sorted((i for i, p in enumerate(props) if p["type"] in ("plane", "cylinder")),
                   key=lambda i: -props[i]["area"])
    if not order:
        return None, "the part has no planar or cylindrical faces"
    emap = _edge_face_map(part.shape)
    vol = _volume(part.shape)
    reason = "the part has no pair of parallel faces (plane / coaxial cylinder) at a constant distance"
    tried = 0
    for f0 in order:
        for f1, t in _partners(props, faces, f0)[:3]:
            tried += 1
            a, b = f0, f1
            if _prefer_top(props[b], props[a]):
                a, b = b, a                       # "upper" = positive normal (z, y, x), for a cylinder the outer face
            info, why = _try_sheet(part, faces, props, emap, vol, a, b, t)
            if info is not None:
                return info, ""
            reason = why
            if tried >= 8:
                return None, reason
    return None, reason


def _prefer_top(p: dict, q: dict) -> bool:
    """Should the face p be the upper one rather than q? Plane: the "more positive" normal (z, then y, then x); cylinder: larger area."""
    if p["type"] == "plane" and q["type"] == "plane":
        for k in (2, 1, 0):
            if abs(p["normal"][k] - q["normal"][k]) > 1e-9:
                return p["normal"][k] > q["normal"][k]
        return False
    return p["area"] > q["area"]


def _try_sheet(part, faces, props, emap, vol, f0, f1, t):
    top = _grow_skin(faces, f0, emap, {f1})
    if f1 in top:
        return None, "top and bottom faces are tangentially connected (not a plate)"
    bottom = _grow_skin(faces, f1, emap, set(top))
    if set(top) & set(bottom):
        return None, "top and bottom faces are tangentially connected (not a plate)"
    a_top = sum(props[i]["area"] for i in top)
    a_bot = sum(props[i]["area"] for i in bottom)
    v_est = t * 0.5 * (a_top + a_bot)
    if vol <= 0 or abs(v_est - vol) > 0.02 * vol:
        return None, (f"thickness is not constant (volume {vol:.6g} mm³, t·(A_t + A_b)/2 = {v_est:.6g} mm³ at t = {t:.6g} mm)")
    mid_area = 0.5 * (a_top + a_bot)
    slender = np.sqrt(mid_area) / t
    if slender < MIN_SLENDER:
        return None, f"part is too thick for a shell (√A / t = {slender:.2g} < {MIN_SLENDER:g})"
    sides = [i for i in range(len(faces)) if i not in top and i not in bottom]
    warnings = []
    if slender < THICK_SLENDER:
        warnings.append(f"Part '{part.name}': thick plate (√A / t = {slender:.2g}) – shell theory is only approximate, verify with a 3D model")
    planar = all(props[i]["type"] == "plane" for i in top)
    if planar:
        n0 = np.asarray(props[f0]["normal"], float)
        p0 = np.asarray(props[f0]["point"], float)
        coplanar = all(abs(float((np.asarray(props[i]["point"]) - p0) @ n0)) <= 1e-6 * max(1.0, t) and
                       float(np.asarray(props[i]["normal"]) @ n0) > 1 - 1e-9 for i in top)
        if not coplanar:
            return None, "top faces are not coplanar"
        mid = _compound([_translate(faces[i], -0.5 * t * n0) for i in top])
        kind, normal = "plate", n0
    else:
        try:
            mid = _offset_skin([faces[i] for i in top], -0.5 * t)
        except ShellError as e:
            return None, str(e)
        kind, normal = "sheet", None
        am = _area(mid)
        if abs(am - mid_area) > 0.01 * mid_area:
            return None, (f"the sheet mid-surface could not be created (area {am:.6g} mm² instead of {mid_area:.6g} mm²)")
    return SheetInfo(part.name, kind, float(t), mid, top, bottom, sides, normal, 0.0, float(mid_area),
                     warnings), ""


def _offset_skin(faces, offset: float) -> TopoDS_Shape:
    """Offset of the skin of a sheet (sewn faces) by `offset` along the outward normal (negative = inwards)."""
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
    sew = BRepBuilderAPI_Sewing(1e-6)
    for f in faces:
        sew.Add(f)
    sew.Perform()
    shell = sew.SewedShape()
    mk = BRepOffsetAPI_MakeOffsetShape()
    mk.PerformBySimple(shell, offset)
    if not mk.IsDone():
        raise ShellError("Sheet skin offset failed (BRepOffsetAPI_MakeOffsetShape)")
    return _compound(list_faces(mk.Shape()))


def surface_sheet(part: Part, thickness: float | None, offset: str = "mid") -> SheetInfo:
    """Planar part from STEP (faces without volume) with a thickness given by the user."""
    if thickness is None or not thickness > 0:
        raise ShellError(f"Surface part '{part.name}' needs a shell thickness (ShellSettings.parts)")
    faces = list_faces(part.shape)
    if not faces:
        raise ShellError(f"Surface part '{part.name}' has no faces")
    return SheetInfo(part.name, "surface", float(thickness), _compound(faces), list(range(len(faces))), [], [],
                     None, OFFSETS[offset], float(sum(_area(f) for f in faces)))


# ------------------------------------------------------------------ planar parts from STEP

def read_step_surfaces(path: str | Path) -> list[Part]:
    """Faces and shells without volume from STEP (fea_core.geometry.read_step skips them) as parts – Part.shape
    = compound of the free faces (in the global position). Names as read_step (instance, otherwise component).
    """
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDF import TDF_Label
    from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from ..geometry import _label_name, generic_name, check_step_external_refs, check_reader_external_refs
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")
    check_step_external_refs(path)                   # external file references only inside the folder (security)
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise IOError(f"Cannot read STEP: {path}")
    check_reader_external_refs(reader)
    reader.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    parts: list[Part] = []
    used: dict[str, int] = {}

    generic = generic_name

    def free_faces(shape):
        solid_faces = []
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        while exp.More():
            solid_faces += list_faces(exp.Current())
            exp.Next()
        out = []
        for f in list_faces(shape):
            if not any(f.IsSame(g) for g in solid_faces):
                out.append(f)
        return out

    def add(name, shape, loc):
        ff = free_faces(shape)
        if not ff:
            return
        n = name or "Face"
        if n in used:
            used[n] += 1
            n = f"{n} ({used[n]})"
        else:
            used[n] = 1
        parts.append(Part(n, _compound([f.Moved(loc) for f in ff])))

    def walk(label, loc, inst_name, parent=""):
        ref = TDF_Label()
        if st.IsReference_s(label) and st.GetReferredShape_s(label, ref):
            loc = loc.Multiplied(st.GetLocation_s(label))
            name = _label_name(label)
            if generic(name):
                name = _label_name(ref)
            walk(ref, loc, name, parent)
            return
        if st.IsAssembly_s(label):
            comps = TDF_LabelSequence()
            st.GetComponents_s(label, comps)
            own = inst_name if not generic(inst_name) else _label_name(label)
            for i in range(1, comps.Length() + 1):
                walk(comps.Value(i), loc, "", own if not generic(own) else parent)
            return
        name = inst_name if not generic(inst_name) else _label_name(label)
        if generic(name) and parent:
            name = parent
        add(name, st.GetShape_s(label), loc)

    roots = TDF_LabelSequence()
    st.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        walk(roots.Value(i), TopLoc_Location(), "")
    return parts


def write_step_surfaces(parts: list[Part], path: str | Path) -> None:
    """Writes the parts (bodies and faces) into a STEP with names (tests, verification tasks)."""
    from ..geometry import write_step
    write_step(parts, path)


# ------------------------------------------------------------------ spoje desek (T spoj)

def _dist(a, b) -> float:
    ds = BRepExtrema_DistShapeShape(a, b)
    return float(ds.Value()) if ds.IsDone() and ds.NbSolution() else np.inf


def _edge_on(edge, face, tol: float) -> bool:
    """Does the edge lie entirely on the face? (start, middle and end within the tolerance)"""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    ad = BRepAdaptor_Curve(edge)
    t0, t1 = ad.FirstParameter(), ad.LastParameter()
    for t in (t0, 0.5 * (t0 + t1), t1):
        v = BRepBuilderAPI_MakeVertex(ad.Value(t)).Vertex()
        if _dist(v, face) > tol:
            return False
    return True


def _boxes_touch(a, b, gap: float) -> bool:
    ba, bb = bbox(a), bbox(b)
    return bool(np.all(ba[:3] <= bb[3:] + gap) and np.all(bb[:3] <= ba[3:] + gap))


def connect_sheets(sheets: list[SheetInfo], parts: dict[str, Part]) -> list[str]:
    """Extends the mid faces of the plates which stand with a side face on the skin of another plate (T / L joint) by half
    the thickness of that plate – the mid faces then intersect and the mesh is conforming (fragmentation in Gmsh). Returns a warning
    (parts joined only by a face on a face – lap joint – the shell model does not connect them).
    """
    warnings = []
    solid = [s for s in sheets if s.kind != "surface"]
    for P in solid:
        pf = list_faces(parts[P.part].shape)
        for Q in solid:
            if Q is P:
                continue
            qshape = parts[Q.part].shape
            tol = 1e-4 * max(P.thickness, Q.thickness)
            if not _boxes_touch(parts[P.part].shape, qshape, 10 * tol):
                continue
            qf = list_faces(qshape)
            qskin = [qf[i] for i in Q.top + Q.bottom]
            # side face P on skin Q → extension
            for si in P.sides:
                fs = pf[si]
                for g in qskin:
                    if not _boxes_touch(fs, g, 10 * tol) or _dist(fs, g) > tol:
                        continue
                    pp = face_props(fs)
                    ns = np.asarray(pp["normal"], float)
                    ng, dg, _ = face_normal_at(g, pp["point"])
                    if dg > tol or float(ns @ ng) > -1 + 1e-6:
                        continue                         # only a touch by an edge / non-matching normal
                    for e in list_edges(P.mid):
                        if _edge_on(e, fs, tol):
                            pr = BRepPrimAPI_MakePrism(e, gp_Vec(*(0.5 * Q.thickness * ns))).Shape()
                            P.extensions.append(pr)
                    break
            # skin P on skin Q → lap joint (not connected)
            for i in P.top + P.bottom:
                for g in qskin:
                    if _boxes_touch(pf[i], g, 10 * tol) and _dist(pf[i], g) <= tol:
                        pp = face_props(pf[i])
                        ng, dg, _ = face_normal_at(g, pp["point"])
                        if dg <= tol and float(np.asarray(pp["normal"]) @ ng) < -1 + 1e-6 and P.part < Q.part:
                            warnings.append(f"Parts '{P.part}' and '{Q.part}' touch over a face (lap joint) – they are not connected in the shell model")
                        break
    return list(dict.fromkeys(warnings))


def project_point(shape, point) -> tuple[np.ndarray, float]:
    """Nearest point of the shape (face / edge) to a point and the distance."""
    v = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, point))).Vertex()
    ds = BRepExtrema_DistShapeShape(v, shape)
    if not ds.IsDone() or not ds.NbSolution():
        raise ShellError("Point cannot be projected onto the shape")
    return xyz_of(ds.PointOnShape2(1)), float(ds.Value())


def is_planar(face) -> bool:
    return BRepAdaptor_Surface(face).GetType() == GeomAbs_Plane
