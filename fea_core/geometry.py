"""Geometry through OCP (OpenCascade): reading / writing STEP with names, face properties.

A part (Part) = one body (solid) in the global position. Repeated instances
in an assembly are separate parts with a sequence number in the name.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp, BRepGProp_Face
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.Bnd import Bnd_Box
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Pnt, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence     # OCP 8: sekvence v OCP.collections
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.XCAFDoc import XCAFDoc_DocumentTool


def _cast(ns, name):
    """OCP 8 has TopoDS.Solid, OCP 7 has TopoDS.Solid_s (the core calls TopoDS.* directly; this stays for fea_gui)."""
    return getattr(ns, name, None) or getattr(ns, name + '_s')


def _quiet_occt() -> None:
    """Mutes the STEP transfer statistics that OCCT prints to stdout."""
    from OCP.Message import Message, Message_Gravity
    ps = Message.DefaultMessenger_s().Printers()
    if ps.Size():
        ps.First().SetTraceLevel(Message_Gravity.Message_Fail)


_quiet_occt()

_SURF_TYPES = {
    GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus",
}


GENERIC_NAMES = {"SOLID", "BODY", "SOLID1", "BODY1", "TELESO", "TĚLESO", "PART", "COMPONENT", "SHAPE", "COMPOUND"}


def generic_name(n: str) -> bool:
    """Name without information: an instance without a name (OCCT "=>[0:1:1:14]"), a generic body from CAD or a SolidWorks occurrence
    without a name – NEXT_ASSEMBLY_USAGE_OCCURRENCE('NAUO1',' ',' ',…): OCCT then takes the id "NAUO1" (instead of the part name).
    """
    s = (n or "").strip()
    if not s or s.startswith("=>[") or s.upper() in GENERIC_NAMES:
        return True
    return s[:4].upper() == "NAUO" and s[4:].isdigit()


def _generic_name_v1(n: str) -> bool:
    """Rule of version 1.0.0: the SolidWorks occurrence "NAUO1" was taken as the part name (`legacy_names`)."""
    return not n or n.startswith("=>[") or n.strip().upper() in GENERIC_NAMES


# part name from the id of a SolidWorks occurrence of an older version: "NAUO3", several bodies "NAUO3_2", same name "NAUO3 (2)"
_LEGACY_NAUO = re.compile(r"NAUO\d+(?:_\d+)?(?: \(\d+\))?")


def legacy_step_names(study) -> bool:
    """Does the study refer to parts named in the old way after the SolidWorks occurrence id ("NAUO1"…)?

    Version 1.0.0 gave the parts of a STEP from SOLIDWORKS (occurrence without a name) the names "NAUO1", "NAUO2"…; newer versions use the name
    of the part. A saved study (project .fea, JSON, MCP) with the old names is therefore read the old way (`read_step(legacy_names=
    True)`) – materials of the parts, face selections, bolts and the mesh of saved results stay valid.
    """
    if study is None:
        return False
    try:
        data = study.to_dict()
    except Exception:  # noqa: BLE001 – the check is only a supplement
        return False
    stack = [data]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            for k, x in v.items():
                if isinstance(k, str) and _LEGACY_NAUO.fullmatch(k):
                    return True
                stack.append(x)
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
        elif isinstance(v, str) and _LEGACY_NAUO.fullmatch(v):
            return True
    return False


@dataclass
class Part:
    name: str
    shape: TopoDS_Shape  # TopoDS_Solid in the global position
    color: tuple[float, float, float] | None = None   # colour from CAD (XCAF, RGB 0–1), None = no colour
    layers: list[str] = field(default_factory=list)   # vrstvy z CAD (XCAF)
    attributes: list[str] = field(default_factory=list)   # values of user attributes (STEP metadata)

    def faces(self) -> list:
        return list_faces(self.shape)

    def volume(self) -> float:
        """Volume [mm³]; cached (shapes are immutable – a new shape assigned to `shape` is recomputed)."""
        c = self.__dict__.get("_volume_cache")
        if c is not None and c[0] is self.shape:
            return c[1]
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(self.shape, props)
        v = props.Mass()
        self.__dict__["_volume_cache"] = (self.shape, v)
        return v


# --------------------------------------------------------------------- STEP

def _label_name(label: TDF_Label) -> str:
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString()
    return ""


def _solids(shape: TopoDS_Shape) -> list[TopoDS_Shape]:
    out = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        out.append(TopoDS.Solid(exp.Current()))
        exp.Next()
    return out


def _label_color(ct, label: TDF_Label) -> tuple[float, float, float] | None:
    """Colour of the XCAF label (face, otherwise general)."""
    from OCP.Quantity import Quantity_Color
    from OCP.XCAFDoc import XCAFDoc_ColorType
    c = Quantity_Color()
    for t in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
        if ct.GetColor_s(label, t, c):
            return (round(c.Red(), 4), round(c.Green(), 4), round(c.Blue(), 4))
    return None


def _shape_color(ct, shape) -> tuple[float, float, float] | None:
    """Colour of the body from the shape, otherwise the prevailing colour of the faces (by area) – CAD often colours only the faces."""
    from OCP.Quantity import Quantity_Color
    from OCP.XCAFDoc import XCAFDoc_ColorType
    c = Quantity_Color()
    types = (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen)
    for t in types:
        if ct.GetColor(shape, t, c):
            return (round(c.Red(), 4), round(c.Green(), 4), round(c.Blue(), 4))
    votes: dict[tuple, float] = {}
    for f in list_faces(shape):
        for t in types:
            if ct.GetColor(f, t, c):
                k = (round(c.Red(), 4), round(c.Green(), 4), round(c.Blue(), 4))
                pr = GProp_GProps()
                BRepGProp.SurfaceProperties_s(f, pr)
                votes[k] = votes.get(k, 0.0) + pr.Mass()
                break
    return max(votes, key=votes.get) if votes else None


def _label_layers(lt, label: TDF_Label) -> list[str]:
    try:
        hs = lt.GetLayers(label)
        return [hs.Value(i).ToExtString() for i in range(1, hs.Length() + 1)]
    except Exception:  # noqa: BLE001 – older OCP / label without layers
        return []


def _label_attributes(label: TDF_Label) -> list[str]:
    """Values of text user attributes (STEP property_definition → TDataStd_NamedData)."""
    try:
        from OCP.TDataStd import TDataStd_NamedData
        nd = TDataStd_NamedData()
        if not label.FindAttribute(TDataStd_NamedData.GetID_s(), nd):
            return []
        nd.LoadDeferredData()
        if not nd.HasStrings():
            return []
        return [v.ToExtString() for v in nd.GetStringsContainer()]   # iteration gives only values
    except Exception:  # noqa: BLE001
        return []


def _free_faces(shape: TopoDS_Shape) -> int:
    """Number of shape faces that lie in no body (faces / shells without volume)."""
    n_all = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        n_all += 1
        exp.Next()
    n_solid = 0
    for so in _solids(shape):
        exp = TopExp_Explorer(so, TopAbs_FACE)
        while exp.More():
            n_solid += 1
            exp.Next()
    return max(0, n_all - n_solid)


def check_parts(parts: list[Part]) -> list[str]:
    """Geometry check after import (BRepCheck, volume): warning in Czech."""
    from OCP.BRepCheck import BRepCheck_Analyzer
    out = []
    for p in parts:
        try:
            if not BRepCheck_Analyzer(p.shape).IsValid():
                out.append(f"Part '{p.name}': the body is not topologically valid (BRepCheck) – meshing may fail, fix the model in CAD")
        except Exception:  # noqa: BLE001 – the check is only a supplement
            pass
        try:
            v = p.volume()
        except Exception:  # noqa: BLE001
            v = None
        if v is not None and v <= 0:
            out.append(f"Part '{p.name}': zero or negative volume ({v:.3g} mm³) – inverted faces or an open body")
    return out


# Entities whose strings OCCT may take an external file name from (DOCUMENT_FILE, EXTERNAL_SOURCE,
# APPLIED_EXTERNAL_IDENTIFICATION_ASSIGNMENT, …): every entity whose type name contains one of these words is scanned.
_EXT_KEYWORD = re.compile(rb"(?:EXTERNAL|DOCUMENT|IDENTIFICATION)[A-Z0-9_]*\s*\(", re.I)
# one token of an entity body: a string literal (with '' inside), the end of the entity, or a run of other characters
_STEP_TOKEN = re.compile(rb"'(?:[^']|'')*'|;|[^';]+")
_ESCAPE = re.compile(r"\\X2\\((?:[0-9A-Fa-f]{4})*)\\X0\\|\\X4\\((?:[0-9A-Fa-f]{8})*)\\X0\\|\\X\\([0-9A-Fa-f]{2})|"
                     r"\\S\\(.)|\\P[A-Za-z]?\\|\\\\")
_MAX_ENTITY = 1 << 20                     # one entity body longer than 1 MB = not a reference, stop scanning it


def decode_step_string(raw: str) -> str:
    """ISO 10303-21 string literal content → text (\\X\\hh, \\X2\\…\\X0\\, \\X4\\…\\X0\\, \\S\\c, \\P?\\, \\\\, '')."""
    def one(m):
        if m.group(1) is not None:
            h = m.group(1)
            return "".join(chr(int(h[i:i + 4], 16)) for i in range(0, len(h), 4))
        if m.group(2) is not None:
            h = m.group(2)
            return "".join(chr(min(int(h[i:i + 8], 16), 0x10FFFF)) for i in range(0, len(h), 8))
        if m.group(3) is not None:
            return chr(int(m.group(3), 16))
        if m.group(4) is not None:
            return chr((ord(m.group(4)) + 128) % 256)
        if m.group(0) == "\\\\":
            return "\\"
        return ""                                       # \P?\ code page switch
    return _ESCAPE.sub(one, raw.replace("''", "'"))


def _bad_reference(name: str) -> bool:
    """True = a file name OCCT must not open: anything but a relative path inside the folder of the STEP."""
    from .pathcheck import is_unc
    t = name.strip()
    if not t:
        return False
    if any(ord(c) < 32 or ord(c) == 127 for c in t):
        return True                                     # control characters (also undecoded escape remnants)
    if is_unc(t) or ":" in t or t[0] in "/\\":
        return True                                     # UNC / NT namespace, drive (also "C:x"), absolute, URL scheme, ADS
    if re.search(r"\\[XSP]", t, re.I):
        return True                                     # an escape sequence OCCT may decode differently than we do
    return ".." in re.split(r"[\\/]", t)


def _refuse(name: str):
    raise ValueError(f"The STEP file references a file outside its folder ({name[:120]!r}) – for security reasons it is "
                     "not loaded; export a STEP without external references (a single file) from your CAD")


def _entity_strings(mm, pos: int):
    """String literals of the entity body starting at pos (outside a string), up to its ';'."""
    out = []
    end = min(len(mm), pos + _MAX_ENTITY)
    for m in _STEP_TOKEN.finditer(mm, pos, end):
        tok = m.group(0)
        if tok == b";":
            break
        if tok[:1] == b"'":
            out.append(tok[1:-1])
    return out


def _keyword_positions(mm):
    """Positions of EXTERNAL / DOCUMENT / IDENTIFICATION (upper or lower case) – bytes.find is ~30× faster than a regex
    scan of a large file; mixed case is left to the check of the names OCCT resolved (check_reader_external_refs)."""
    out = set()
    for kw in (b"EXTERNAL", b"DOCUMENT", b"IDENTIFICATION"):
        for w in (kw, kw.lower()):
            k = mm.find(w)
            while k >= 0:
                out.add(k)
                k = mm.find(w, k + 1)
    return sorted(out)


def check_step_external_refs(path: str | Path) -> None:
    """STEP references to external files may point only into the folder of the file (OCCT opens them itself when reading).
    A network path (UNC, NT namespace), an absolute path or ".." → ValueError: opening \\\\\\\\server\\\\… would make
    Windows log in to a foreign server (leak of the NTLM hash), and an absolute path would bring in a foreign file.

    Text pre-check before OCCT parses the file: all string literals of every entity whose type name contains EXTERNAL,
    DOCUMENT or IDENTIFICATION, with the ISO 10303-21 escapes decoded. `check_reader_external_refs` then checks the names
    OCCT itself resolved (after ReadFile, before Transfer) – that one is authoritative.
    """
    import mmap
    with open(path, "rb") as f:
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:                       # empty file
            return
        with mm:
            for pos in _keyword_positions(mm):
                kw = _EXT_KEYWORD.match(mm, pos)
                if kw is None:
                    continue
                for lit in _entity_strings(mm, kw.end()):
                    t = decode_step_string(lit.decode("latin-1"))
                    if _bad_reference(t):
                        _refuse(t)


def check_reader_external_refs(reader) -> None:
    """External file references as OCCT resolved them (STEPConstruct_ExternRefs over the work session of a
    STEPCAFControl_Reader after ReadFile, before Transfer – Transfer opens exactly these names)."""
    from OCP.STEPConstruct import STEPConstruct_ExternRefs
    refs = STEPConstruct_ExternRefs(reader.Reader().WS())
    if not refs.LoadExternRefs():
        return
    for i in range(1, refs.NbExternRefs() + 1):
        name = refs.FileName(i)
        if name is None:
            continue
        name = name if isinstance(name, str) else str(name)
        if _bad_reference(name):
            _refuse(name)


def read_step(path: str | Path, report: list[str] | None = None, legacy_names: bool = False) -> list[Part]:
    """Reads a STEP assembly, returns the list of bodies with names and global position.

    In addition (XCAF): colour of the body (instance > component > shape > prevailing colour of faces), layers and values of
    user attributes – for recognising welds by colour / layer / the attribute "weld".
    report: list to which the geometry check warnings are added (faces without a body, invalid bodies).
    legacy_names: part names as in version 1.0.0 (SolidWorks occurrence "NAUO1" instead of the part name) – for saved
    studies with the old names, see `legacy_step_names`.
    """
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    reader.SetLayerMode(True)
    if hasattr(reader, "SetMetaMode"):
        reader.SetMetaMode(True)
    Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")
    check_step_external_refs(path)
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise IOError(f"Cannot read STEP: {path}")
    check_reader_external_refs(reader)
    reader.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    lt = XCAFDoc_DocumentTool.LayerTool_s(doc.Main())

    parts: list[Part] = []
    used: dict[str, int] = {}

    skipped: list[str] = []

    def add(name: str, shape: TopoDS_Shape, loc: TopLoc_Location, color, layers, attrs):
        if report is not None and _free_faces(shape):
            skipped.append(name or "?")
        raw = _solids(shape)                           # search colours on the shape without position (matching by labels)
        solids = [r.Moved(loc) for r in raw]
        for i, s in enumerate(solids):
            n = name or "Solid"
            if len(solids) > 1:
                n = f"{n}_{i + 1}"
            if n in used:
                used[n] += 1
                n = f"{n} ({used[n]})"
            else:
                used[n] = 1
            col = color
            if col is None:
                try:
                    col = _shape_color(ct, raw[i] if len(raw) > 1 else shape)
                except Exception:  # noqa: BLE001 – colour is only a supplement
                    col = None
            parts.append(Part(n, s, col, list(dict.fromkeys(layers)), list(dict.fromkeys(attrs))))

    generic = _generic_name_v1 if legacy_names else generic_name

    def walk(label: TDF_Label, loc: TopLoc_Location, inst_name: str, parent: str = "", color=None,
             layers=(), attrs=()):
        layers = list(layers) + _label_layers(lt, label)
        attrs = list(attrs) + _label_attributes(label)
        color = color or _label_color(ct, label)             # the colour of the instance takes precedence
        ref = TDF_Label()
        if st.IsReference_s(label) and st.GetReferredShape_s(label, ref):
            loc = loc.Multiplied(st.GetLocation_s(label))
            name = _label_name(label)
            if generic(name):
                name = _label_name(ref)
            walk(ref, loc, name, parent, color, layers, attrs)
            return
        if st.IsAssembly_s(label):
            comps = TDF_LabelSequence()
            st.GetComponents_s(label, comps)
            own = inst_name if not generic(inst_name) else _label_name(label)
            for i in range(1, comps.Length() + 1):
                walk(comps.Value(i), loc, "", own if not generic(own) else parent, color, layers, attrs)
            return
        shape = st.GetShape_s(label)
        name = inst_name if not generic(inst_name) else _label_name(label)
        if generic(name) and parent:                   # Fusion: component "Rám" (frame) with body "SOLID"
            name = parent
        add(name, shape, loc, color, layers, attrs)

    roots = TDF_LabelSequence()
    st.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        walk(roots.Value(i), TopLoc_Location(), "")
    if report is not None:
        if skipped:
            report.append("The STEP contains faces or shells without volume (ignored, FEM uses only solids): "
                          + ", ".join(dict.fromkeys(skipped)))
        if not parts:
            report.append("The STEP file contains no solid – export solid bodies from CAD, not surfaces")
        report.extend(check_parts(parts))
    return parts


def write_step(parts: list[Part], path: str | Path) -> None:
    """Writes bodies to a STEP with names, colours, layers and attributes (for verification tasks and tests)."""
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    lt = XCAFDoc_DocumentTool.LayerTool_s(doc.Main())
    meta = False
    for p in parts:
        lab = st.AddShape(p.shape, False)
        TDataStd_Name.Set_s(lab, TCollection_ExtendedString(p.name, True))   # True = UTF-8 (diakritika)
        if p.color is not None:
            from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
            from OCP.XCAFDoc import XCAFDoc_ColorType
            ct.SetColor(lab, Quantity_Color(*map(float, p.color), Quantity_TOC_RGB), XCAFDoc_ColorType.XCAFDoc_ColorSurf)
        for ly in p.layers:
            lt.SetLayer(lab, TCollection_ExtendedString(ly, True), False)
        if p.attributes:
            from OCP.TDataStd import TDataStd_NamedData
            nd = TDataStd_NamedData.Set_s(lab)
            for i, v in enumerate(p.attributes, 1):
                nd.SetString(TCollection_ExtendedString(f"attr{i}"), TCollection_ExtendedString(v, True))
            meta = True
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    w = STEPCAFControl_Writer()
    w.SetNameMode(True)
    w.SetColorMode(True)
    w.SetLayerMode(True)
    if meta and hasattr(w, "SetMetadataMode"):
        w.SetMetadataMode(True)
    if not w.Transfer(doc, STEPControl_AsIs):
        raise IOError("STEP conversion failed")
    if w.Write(str(path)) != IFSelect_RetDone:
        raise IOError(f"The STEP cannot be written: {path}")


def write_brep(shape: TopoDS_Shape, path: str | Path) -> None:
    BRepTools.Write_s(shape, str(path))


# --------------------------------------------------------------------- faces

def list_faces(shape: TopoDS_Shape) -> list:
    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        out.append(TopoDS.Face(exp.Current()))
        exp.Next()
    return out


def list_edges(shape: TopoDS_Shape) -> list:
    """Unique edges of the shape (index = order of the first occurrence in TopExp_Explorer – the same as FeaView.part_edges)."""
    from OCP.TopAbs import TopAbs_EDGE
    out, seen = [], {}
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        e = TopoDS.Edge(exp.Current())
        bucket = seen.setdefault(hash(e), [])
        if not any(e.IsSame(o) for o in bucket):
            bucket.append(e)
            out.append(e)
        exp.Next()
    return out


_CURVE_TYPES = None


def edge_props(edge) -> dict:
    """Edge properties for a selector: type (line | circle | ellipse | curve), length, centroid, point at the middle
    of the parameter, envelope; for a circle / arc the centre, axis and radius.
    """
    global _CURVE_TYPES
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Line
    if _CURVE_TYPES is None:
        _CURVE_TYPES = {GeomAbs_Line: "line", GeomAbs_Circle: "circle", GeomAbs_Ellipse: "ellipse"}
    props = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, props)
    c = props.CentreOfMass()
    ad = BRepAdaptor_Curve(edge)
    t0, t1 = ad.FirstParameter(), ad.LastParameter()

    def xyz(p):
        return np.array([p.X(), p.Y(), p.Z()])
    d = {"type": _CURVE_TYPES.get(ad.GetType(), "curve"), "length": float(props.Mass()),
         "centroid": np.array([c.X(), c.Y(), c.Z()]), "point": xyz(ad.Value(0.5 * (t0 + t1))),
         "start": xyz(ad.Value(t0)), "end": xyz(ad.Value(t1)), "bbox": bbox(edge)}
    if d["type"] == "circle":
        circ = ad.Circle()
        ax = circ.Axis().Direction()
        d.update(center=xyz(circ.Location()), axis=np.array([ax.X(), ax.Y(), ax.Z()]), radius=float(circ.Radius()))
    return d


def edge_polyline(edge, n: int = 24) -> np.ndarray:
    """Points of the edge evenly in the parameter (drawing), (n + 1, 3); a line only the end points."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Line
    ad = BRepAdaptor_Curve(edge)
    t0, t1 = ad.FirstParameter(), ad.LastParameter()
    ts = [t0, t1] if ad.GetType() == GeomAbs_Line else np.linspace(t0, t1, n + 1)
    return np.array([xyz_of(ad.Value(float(t))) for t in ts])


def project_to_surface(shape: TopoDS_Shape, point) -> tuple[np.ndarray, float]:
    """Nearest point of the shape surface to a point and the distance [mm] (a point for a load "at a point" must lie on the surface)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    v = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, point))).Vertex()
    best, best_d = None, np.inf
    for f in list_faces(shape):                    # by faces – the solid itself would return 0 for a point inside
        ds = BRepExtrema_DistShapeShape(v, f)
        if ds.IsDone() and ds.NbSolution() and ds.Value() < best_d:
            best_d = ds.Value()
            best = xyz_of(ds.PointOnShape2(1))
    if best is None:
        raise ValueError("The point cannot be projected onto the part surface")
    return best, float(best_d)


def xyz_of(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()])


def bbox(shape: TopoDS_Shape) -> np.ndarray:
    b = Bnd_Box()
    b.SetGap(0.0)
    BRepBndLib.AddOptimal_s(shape, b, False, False)
    lo, hi = b.CornerMin(), b.CornerMax()  # Bnd_Box.Get() in OCP 8 cannot be converted
    return np.array([lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z()])


def cylinder_axis(face) -> tuple[np.ndarray, np.ndarray, float] | None:
    """(point on the axis, unit direction of the axis, radius) of a cylindrical face, otherwise None."""
    s = BRepAdaptor_Surface(face)
    if s.GetType() != GeomAbs_Cylinder:
        return None
    c = s.Cylinder()
    ax = c.Axis()
    p, d = ax.Location(), ax.Direction()
    return np.array([p.X(), p.Y(), p.Z()]), np.array([d.X(), d.Y(), d.Z()]), c.Radius()


def coaxial_cylinders(faces: list, index: int, tol: float = 1e-6) -> list[int]:
    """Indices of all cylindrical faces coaxial with faces[index] and of the same radius (a whole hole / pin)."""
    ref = cylinder_axis(faces[index])
    if ref is None:
        return [index]
    p0, d0, r0 = ref
    out = []
    for i, f in enumerate(faces):
        cyl = cylinder_axis(f)
        if cyl is None:
            continue
        p, d, r = cyl
        if abs(r - r0) > tol * max(1.0, r0) or abs(abs(float(d @ d0)) - 1.0) > 1e-9:
            continue
        off = (p - p0) - ((p - p0) @ d0) * d0          # distance of axes
        if np.linalg.norm(off) <= tol * max(1.0, r0) + 1e-9:
            out.append(i)
    return out


def face_props(face) -> dict:
    """Geometric properties of a face for a selector.

    point/normal = a point lying ON the face (near the centroid) and the outward normal at it.
    """
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    area = props.Mass()
    c = props.CentreOfMass()
    centroid = np.array([c.X(), c.Y(), c.Z()])

    # point on a face: triangulation node nearest to the centroid
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None:
        BRepMesh_IncrementalMesh(face, max(np.sqrt(area) / 20, 1e-3), False, 0.3)
        tri = BRep_Tool.Triangulation_s(face, loc)
    trsf = loc.Transformation()
    best, best_d, best_uv = None, np.inf, None
    for i in range(1, tri.NbNodes() + 1):
        p = tri.Node(i).Transformed(trsf)
        q = np.array([p.X(), p.Y(), p.Z()])
        d = np.linalg.norm(q - centroid)
        if d < best_d:
            best, best_d = q, d
            best_uv = tri.UVNode(i) if tri.HasUVNodes() else None
    normal = np.zeros(3)
    if best_uv is not None:
        gf = BRepGProp_Face(face)
        pnt, vec = gp_Pnt(), gp_Vec()
        gf.Normal(best_uv.X(), best_uv.Y(), pnt, vec)
        if vec.Magnitude() > 0:
            normal = np.array([vec.X(), vec.Y(), vec.Z()]) / vec.Magnitude()
    stype = _SURF_TYPES.get(BRepAdaptor_Surface(face).GetType(), "other")
    return {
        "area": area,
        "centroid": centroid,
        "point": best,
        "normal": normal,
        "bbox": bbox(face),
        "type": stype,
    }
