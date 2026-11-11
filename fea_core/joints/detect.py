"""Recognition of bolted joints and welds from the geometry of the assembly (OCP, main process).

Bolts: parts named by a standard (naming.parse_name) → axis from the cylindrical face of the shank,
the bearing face of the head (planar face ⟂ axis with the longest continuation of the bolt behind it), washers
and nuts coaxial with the bolt, clamped parts by the intersections of lines parallel to the axis (in the middle of the
bearing annulus), hole diameters from coaxial cylindrical faces. Without a nut = a bolt into a thread.
Welds: parts named "svar", "weld", "W_…"; contact faces = faces of the weld lying on another part.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.BRepIntCurveSurface import BRepIntCurveSurface_Inter
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_IN, TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

from ..geometry import Part, bbox, cylinder_axis, face_props
from . import tables
from .model import BoltSpec, WeldSpec
from .naming import FastenerName, parse_name

WELD_NAME_RE = re.compile(r"(svar|weld|schwei|^w_|^w\d|[\s_-]w\d)", re.I)
WELD_TAG_RE = re.compile(r"(svar|weld|schwei)", re.I)          # layer / user attribute from CAD


@dataclass
class Suggestion:
    """Recognised joint to be confirmed by the user (chapter 13: automatic recognition is always shown)."""
    kind: str                           # bolt | weld
    spec: object                        # BoltSpec | WeldSpec
    source: str                         # name | shape | colour | layer | attribute
    confidence: float = 1.0
    reason: str = ""
    accept: bool = True                 # default tick in the confirmation list

    @property
    def label(self) -> str:
        s = self.spec
        if self.kind == "bolt":
            return f"{s.name}: {s.designation} – {', '.join(s.cad_parts)}"
        return f"{s.name}: {s.part}"


@dataclass
class DetectReport:
    bolts: list[BoltSpec] = field(default_factory=list)                # by names (reliable)
    welds: list[WeldSpec] = field(default_factory=list)                # by rules (name, colour, layer, attribute)
    fasteners: dict[str, FastenerName] = field(default_factory=dict)   # part -> recognised name
    warnings: list[str] = field(default_factory=list)
    unused: list[str] = field(default_factory=list)                    # nuts / washers without a bolt
    shapes: dict = field(default_factory=dict)                         # part -> shape.ShapeGuess (without a standard in the name)
    suggestions: list[Suggestion] = field(default_factory=list)        # everything recognised, for confirmation
    rejected: list[str] = field(default_factory=list)                  # shape like a bolt, but no joint found (info)

    @property
    def shape_bolts(self) -> list[BoltSpec]:
        """Bolts where at least one component was recognised only by shape (they require confirmation)."""
        return [s.spec for s in self.suggestions if s.kind == "bolt" and s.source == "shape"]


def _hex_rgb(h: str) -> tuple[float, float, float] | None:
    h = h.strip().lstrip("#")
    if len(h) != 6:
        return None
    try:
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return None


def rgb_hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(round(255 * max(0.0, min(1.0, v)))) for v in c)


def weld_rule(part: Part, settings=None) -> tuple[str, str] | None:
    """Rule for a weld: (source, reason) or None. Name (svar, weld, W_…), layer / attribute "weld", colour."""
    if WELD_NAME_RE.search(part.name):
        return "name", f"name “{part.name}”"
    for ly in getattr(part, "layers", []) or []:
        if WELD_TAG_RE.search(ly):
            return "layer", f"layer “{ly}”"
    for at in getattr(part, "attributes", []) or []:
        if WELD_TAG_RE.search(at):
            return "attribute", f"attribute “{at}”"
    col = getattr(part, "color", None)
    if settings is not None and col is not None:
        for h in settings.weld_colors:
            ref = _hex_rgb(h)
            if ref is not None and float(np.linalg.norm(np.subtract(col, ref))) <= settings.weld_color_tol + 1e-9:
                return "color", f"color {rgb_hex(col)} ≈ {h}"
    return None


# ===================================================================== OCP helpers

def _tri_nodes(shape, defl: float | None = None) -> np.ndarray:
    """Nodes of the triangulation (global coordinates) – for the projections onto the axis."""
    from ..geometry import list_faces
    pts = []
    faces = list_faces(shape)
    if defl is None:
        bb = bbox(shape)
        defl = max(1e-3, 2e-3 * float(np.linalg.norm(bb[3:] - bb[:3])))
    for f in faces:
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            BRepMesh_IncrementalMesh(f, defl, False, 0.3)
            tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i).Transformed(trsf)
            pts.append((p.X(), p.Y(), p.Z()))
    return np.asarray(pts, float).reshape(-1, 3)


def _face_samples(face, n: int = 6) -> np.ndarray:
    """Points lying inside a face: centroids of the largest triangles of the triangulation projected onto the face."""
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None:
        BRepMesh_IncrementalMesh(face, 0.05, False, 0.3)
        tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None or tri.NbTriangles() == 0:
        return np.zeros((0, 3))
    trsf = loc.Transformation()
    nodes = np.array([[tri.Node(i).Transformed(trsf).X(), tri.Node(i).Transformed(trsf).Y(),
                       tri.Node(i).Transformed(trsf).Z()] for i in range(1, tri.NbNodes() + 1)])
    tris = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)]) - 1
    P = nodes[tris]
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    order = np.argsort(-area)[:n]
    surf = BRep_Tool.Surface_s(face)
    out = []
    for c in P[order].mean(axis=1):
        pr = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, c)), surf)
        if pr.NbPoints():
            q = pr.NearestPoint()
            out.append((q.X(), q.Y(), q.Z()))
        else:
            out.append(tuple(c))
    return np.asarray(out, float)


def _dist_to_shape(p, shape) -> float:
    v = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, p))).Vertex()
    d = BRepExtrema_DistShapeShape(v, shape)
    return d.Value() if d.IsDone() else math.inf


_CLS: dict[int, tuple] = {}                 # id(shape) -> (shape, classifier) – loading the body is expensive
_BB: dict[int, tuple] = {}


def _inside(shape, p, tol=1e-6) -> bool:
    k = id(shape)
    hit = _CLS.get(k)
    if hit is None or hit[0] is not shape:
        if len(_CLS) > 64:
            _CLS.clear()
        hit = (shape, BRepClass3d_SolidClassifier(shape))
        _CLS[k] = hit
    c = hit[1]
    c.Perform(gp_Pnt(*map(float, p)), tol)
    return c.State() == TopAbs_IN


def _bbox(shape) -> np.ndarray:
    """bbox with a cache (the recognition asks about the same parts many times)."""
    k = id(shape)
    hit = _BB.get(k)
    if hit is None or hit[0] is not shape:
        if len(_BB) > 4096:
            _BB.clear()
        hit = (shape, bbox(shape))
        _BB[k] = hit
    return hit[1]


def _line_intervals(shape, p0, e, s_lo: float, s_hi: float) -> list[tuple[float, float]]:
    """Segments of the line p0 + s·e inside the body (s in [s_lo, s_hi])."""
    inter = BRepIntCurveSurface_Inter()
    inter.Init(shape, gp_Lin(gp_Pnt(*map(float, p0)), gp_Dir(*map(float, e))), 1e-7)
    ws = []
    while inter.More():
        ws.append(inter.W())
        inter.Next()
    ws = sorted(set(round(w, 9) for w in ws))
    cuts = [s_lo] + [w for w in ws if s_lo < w < s_hi] + [s_hi]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 1e-6:
            continue
        if _inside(shape, np.asarray(p0) + 0.5 * (a + b) * np.asarray(e)):
            if out and abs(out[-1][1] - a) < 1e-6:
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
    return out


def _planar_faces_perp(part: Part, p0, e, tol_ang=1e-4) -> list[dict]:
    """Planar faces perpendicular to the axis: position s, outward normal (±1 with respect to e), radii (min, max)."""
    out = []
    for f in part.faces():
        pr = face_props(f)
        if pr["type"] != "plane":
            continue
        n = np.asarray(pr["normal"], float)
        c = float(n @ e)
        if abs(abs(c) - 1.0) > tol_ang:
            continue
        nodes = _tri_nodes(f)
        rel = nodes - p0
        s = float(np.median(rel @ e))
        r = np.linalg.norm(rel - np.outer(rel @ e, e), axis=1)
        out.append({"face": f, "s": s, "sign": 1 if c > 0 else -1, "rmin": float(r.min()),
                    "rmax": float(r.max()), "area": pr["area"]})
    return out


def _coaxial_cylinders(part: Part, p0, e, tol: float) -> list[dict]:
    out = []
    for f in part.faces():
        cyl = cylinder_axis(f)
        if cyl is None:
            continue
        p, d, r = cyl
        if abs(abs(float(d @ e)) - 1.0) > 1e-6:
            continue
        off = (p - p0) - ((p - p0) @ e) * e
        if np.linalg.norm(off) > tol:
            continue
        nodes = _tri_nodes(f)
        s = (nodes - p0) @ e
        reversed_ = f.Orientation() == TopAbs_REVERSED
        out.append({"face": f, "r": r, "s0": float(s.min()), "s1": float(s.max()), "hole": None,
                    "reversed": reversed_})
    return out


def coaxial_cones(part: Part, p0, e, tol: float) -> list[dict]:
    """Conical faces coaxial with the axis: range s, radius at the ends (countersunk heads and countersinks)."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cone
    out = []
    for f in part.faces():
        sf = BRepAdaptor_Surface(f)
        if sf.GetType() != GeomAbs_Cone:
            continue
        ax = sf.Cone().Axis()
        d = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
        p = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
        if abs(abs(float(d @ e)) - 1.0) > 1e-6:
            continue
        off = (p - p0) - ((p - p0) @ e) * e
        if np.linalg.norm(off) > tol:
            continue
        rel = _tri_nodes(f) - p0
        s = rel @ e
        r = np.linalg.norm(rel - np.outer(s, e), axis=1)
        i0, i1 = int(np.argmin(s)), int(np.argmax(s))
        out.append({"face": f, "s0": float(s.min()), "s1": float(s.max()), "r_s0": float(r[i0]),
                    "r_s1": float(r[i1]), "rmax": float(r.max())})
    return out


def _slot_width(part: Part, p0, e, s0: float, s1: float, tol: float, d: float) -> float | None:
    """Slot: two parallel half-cylindrical faces of the same radius, axes close to the axis of the bolt → width 2r."""
    best = None
    for f in part.faces():
        cyl = cylinder_axis(f)
        if cyl is None:
            continue
        p, dd, r = cyl
        if abs(abs(float(dd @ e)) - 1.0) > 1e-6 or r < 0.45 * d or r > 1.5 * d:
            continue
        off = np.linalg.norm((p - p0) - ((p - p0) @ e) * e)
        if off > 2.5 * r:
            continue
        sv = (_tri_nodes(f) - p0) @ e
        if min(sv.max(), s1) - max(sv.min(), s0) > 0.5 * (s1 - s0):
            best = 2 * r if best is None else min(best, 2 * r)
    return best


def part_axis(part: Part, d_hint: float | None) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Axis of the bolt: cylindrical face with the radius nearest d/2 (largest area), otherwise the main axis of inertia."""
    best = None
    for f in part.faces():
        cyl = cylinder_axis(f)
        if cyl is None:
            continue
        p, d, r = cyl
        area = face_props(f)["area"]
        score = (abs(r - d_hint / 2) / max(d_hint, 1e-9), -area) if d_hint else (0.0, -area)
        if best is None or score < best[0]:
            best = (score, p, d, r)
    if best is not None and (d_hint is None or best[0][0] < 0.15):
        return np.asarray(best[1], float), np.asarray(best[2], float) / np.linalg.norm(best[2]), float(best[3])
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(part.shape, props)
    c = props.CentreOfMass()
    pp = props.PrincipalProperties()
    moms = [pp.FirstAxisOfInertia(), pp.SecondAxisOfInertia(), pp.ThirdAxisOfInertia()]
    vals = pp.Moments()
    k = int(np.argmin(vals))
    ax = moms[k]
    return (np.array([c.X(), c.Y(), c.Z()]), np.array([ax.X(), ax.Y(), ax.Z()]),
            (d_hint or 0.0) / 2)


def _extent(shape, p0, e) -> tuple[float, float]:
    s = (_tri_nodes(shape) - p0) @ e
    return float(s.min()), float(s.max())


def _coaxial_part(part: Part, p0, e, tol: float) -> bool:
    for f in part.faces():
        cyl = cylinder_axis(f)
        if cyl is None:
            continue
        p, d, _ = cyl
        if abs(abs(float(d @ e)) - 1.0) < 1e-6:
            off = (p - p0) - ((p - p0) @ e) * e
            if np.linalg.norm(off) <= tol:
                return True
    return False


# ===================================================================== bolts

def _clamp_stack(parts: list[Part], skip: set[str], p0, e, r_probe: float, s_lo: float, s_hi: float,
                 bolt_bb) -> dict[str, list[tuple[float, float]]]:
    """Parts intersected by lines parallel to the axis at the distance r_probe (4 directions)."""
    t1 = np.cross(e, [1.0, 0, 0] if abs(e[0]) < 0.9 else [0, 1.0, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(e, t1)
    out: dict[str, list[tuple[float, float]]] = {}
    lo = np.minimum(p0 + s_lo * e, p0 + s_hi * e) - r_probe - 1.0
    hi = np.maximum(p0 + s_lo * e, p0 + s_hi * e) + r_probe + 1.0
    for part in parts:
        if part.name in skip:
            continue
        bb = _bbox(part.shape)
        if np.any(bb[3:] < lo) or np.any(bb[:3] > hi):
            continue
        ivs = []
        for t in (t1, -t1, t2, -t2):
            ivs += _line_intervals(part.shape, p0 + r_probe * t, e, s_lo, s_hi)
        if ivs:
            ivs.sort()
            merged = [list(ivs[0])]
            for a, b in ivs[1:]:
                if a <= merged[-1][1] + 1e-6:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            out[part.name] = [tuple(m) for m in merged]
    return out


def _hole(part: Part, p0, e, s0: float, s1: float, tol: float) -> float | None:
    """Diameter of a coaxial hole in a part on the section [s0, s1] (the cylinder with the largest overlap)."""
    best, best_ov = None, 0.0
    for c in _coaxial_cylinders(part, p0, e, tol):
        ov = min(c["s1"], s1) - max(c["s0"], s0)
        if ov > best_ov + 1e-9:
            best, best_ov = 2 * c["r"], ov
    return best


def detect(parts: list[Part], existing: list[BoltSpec] | None = None, settings=None,
           shapes: bool = True) -> DetectReport:
    """Recognises bolts (by names, without a standard in the name by shape) and welds (name, layer, attribute, colour).

    rep.bolts = bolts whose components all have a standard in the name; bolts with a nut / washer / shank
    recognised only by shape are only in rep.suggestions (source "shape") – they are added only after confirmation.
    settings = JointSettings (weld colours); shapes = False → by names only (fast).
    """
    from .shape import analyze, HEAD_LABEL as HEAD_LABEL_CZ
    rep = DetectReport()
    names: dict[str, FastenerName] = {}
    for p in parts:
        fn = parse_name(p.name)
        if fn is not None:
            names[p.name] = fn
    rep.fasteners = dict(names)
    weld_src = {}
    for p in parts:
        if p.name not in names:
            r = weld_rule(p, settings)
            if r is not None:
                weld_src[p.name] = r
    if shapes:
        for p in parts:
            if p.name in weld_src:
                continue
            fn = names.get(p.name)
            if fn is not None:
                # name without a standard ("Šroub M4x17"): head type from the shape (otherwise a hexagon would be assumed)
                if fn.kind == "bolt" and not fn.standard:
                    g = analyze(p)
                    if g is not None and g.kind == "bolt" and g.d == fn.d and g.head and g.head != fn.head:
                        fn.head = g.head
                        rep.warnings.append(f"{p.name}: name without a standard – {HEAD_LABEL_CZ.get(g.head, g.head)} by shape ({g.reason})")
                continue
            g = analyze(p)
            if g is not None:
                rep.shapes[p.name] = g
                names[p.name] = g.fastener_name()
    fastener_parts = set(names)
    used: set[str] = set()
    k = len(existing or [])
    for p in parts:
        fn = names.get(p.name)
        if fn is None or fn.kind not in ("bolt", "fitted"):
            continue
        by_shape = p.name in rep.shapes
        local: list[str] = []
        try:
            b = _bolt_from_part(p, fn, parts, names, fastener_parts, local, f"Bolt {k + 1}")
        except Exception as ex:  # noqa: BLE001 – one bolt must not bring down the whole recognition
            local.append(f"{p.name}: bolt could not be recognized ({ex})")
            b = None
        shaped = [n for n in (b.cad_parts if b else []) if n in rep.shapes]
        if by_shape:
            g = rep.shapes[p.name]
            if b is None:                                # shape like a bolt, but without clamped parts / holes
                rep.rejected.append(f"{p.name}: by shape {g.describe()} ({g.reason}), but the joint was not found – "
                                    + "; ".join(local))
                continue
            b.note = f"recognized by shape ({g.reason}); strength class unknown – 8.8 assumed"
            local = [w for w in local if "strength class missing in the name" not in w]
        rep.warnings += local
        if b is None:
            continue
        k += 1
        used.update(b.cad_parts)
        if shaped:
            conf = min(rep.shapes[n].confidence for n in shaped)
            why = "; ".join(f"{n}: {rep.shapes[n].describe()} – {rep.shapes[n].reason}" for n in shaped)
            rep.suggestions.append(Suggestion("bolt", b, "shape", conf, why, accept=conf >= 0.7))
            if not by_shape:
                b.note = (b.note + "; " if b.note else "") + "nut/washer recognised by shape"
        else:
            rep.bolts.append(b)
            rep.suggestions.append(Suggestion("bolt", b, "name", 1.0, f"name “{p.name}”"))
    rep.unused = [n for n, fn in names.items() if fn.kind in ("nut", "washer", "washer_large") and n not in used
                  and n not in rep.shapes]
    for n in rep.unused:
        rep.warnings.append(f"{n}: {names[n].describe()} does not belong to any recognized bolt")
    for p in parts:
        if p.name in weld_src:
            src, why = weld_src[p.name]
            w = WeldSpec(f"Weld {len(rep.welds) + 1}", p.name, note=f"weld by rule: {why}")
            rep.welds.append(w)
            rep.suggestions.append(Suggestion("weld", w, src, 1.0, why))
    return rep


def _bolt_from_part(part: Part, fn: FastenerName, parts: list[Part], names: dict, fastener_parts: set,
                    warnings: list, bname: str) -> BoltSpec | None:
    d = fn.d
    th = tables.thread(d, fn.pitch)
    ax = part_axis(part, d)
    if ax is None:
        warnings.append(f"{part.name}: bolt axis not found")
        return None
    p_ax, e, r_geo = ax
    tol = max(0.05, 0.02 * d)
    smin, smax = _extent(part.shape, p_ax, e)
    # bearing face of the head: plane ⟂ axis, radius > 0.6 d, behind it the longest part of the bolt
    cands = [f for f in _planar_faces_perp(part, p_ax, e) if f["rmax"] > 0.6 * d]
    best = None
    head_cone = 0.0
    for f in cands:
        beyond = (smax - f["s"]) if f["sign"] > 0 else (f["s"] - smin)
        if best is None or beyond > best[0]:
            best = (beyond, f)
    cones = coaxial_cones(part, p_ax, e, tol) if (best is None or best[0] < 0.5 * d) else []
    if cones:
        # countersunk head: top face of the head = large end of the cone, e = from the large end towards the tip
        c = max(cones, key=lambda k: k["rmax"])
        if c["r_s1"] > c["r_s0"]:
            e = -e
            s_top = -c["s1"]
            smin, smax = -smax, -smin
        else:
            s_top = c["s0"]
        s_bear = s_top
        head_cone = c["s1"] - c["s0"]
        dk = 2 * c["rmax"]
    elif best is None or best[0] < 0.5 * d:
        warnings.append(f"{part.name}: head bearing face not found")
        return None
    else:
        bf = best[1]
        if bf["sign"] < 0:                           # e = from the head towards the tip
            e = -e
            s_bear = -bf["s"]
            smin, smax = -smax, -smin
        else:
            s_bear = bf["s"]
    p0 = p_ax + s_bear * e                        # point on the axis in the bearing plane of the head (countersunk: top face)
    s_tip = smax - s_bear
    head = tables.head_dims(d, fn.head, fn.standard)
    if head_cone:
        head = dict(head, dw=dk)
    cad = [part.name]
    # washers and nuts coaxial with the bolt
    washers, nut = [], None
    for q in parts:
        fq = names.get(q.name)
        if fq is None or q.name == part.name or fq.kind not in ("nut", "washer", "washer_large"):
            continue
        if not _coaxial_part(q, p0, e, tol):
            continue
        a, b = _extent(q.shape, p0, e)
        if b < -tol or a > s_tip + tol:
            continue
        if fq.kind == "nut":
            if nut is None or a < nut[1]:
                nut = (q, a, b, fq)
        else:
            dims = tables.washer_dims(fq.d or d, fq.kind == "washer_large")
            r_out = max(np.linalg.norm(((_tri_nodes(q.shape) - p0) - np.outer((_tri_nodes(q.shape) - p0) @ e, e)),
                                       axis=1))
            washers.append({"part": q.name, "s0": a, "s1": b, "h": b - a, "d1": dims["d1"],
                            "d2": float(2 * r_out)})
    # washers under the head (adjoining the bearing plane) and under the nut
    washers.sort(key=lambda w: w["s0"])
    head_w, s_plate = [], 0.0
    for w in washers:
        if abs(w["s0"] - s_plate) <= tol:
            head_w.append(w)
            s_plate = w["s1"]
    nut_w = []
    if nut is not None:
        s_n = nut[1]
        for w in reversed(washers):
            if w in head_w:
                continue
            if abs(w["s1"] - s_n) <= tol:
                nut_w.append(w)
                s_n = w["s0"]
    else:
        s_n = s_tip
    other_w = [w for w in washers if w not in head_w and w not in nut_w]
    for w in other_w:
        warnings.append(f"{bname}: washer '{w['part']}' sits on the bolt but does not seat against the head or nut")
    # clamped parts: lines in the middle of the bearing annulus
    dw_head = head_w[-1]["d2"] if head_w else head["dw"]
    r_probe = 0.25 * (tables.CLEARANCE.get(tables.size_key(d), (1.1 * d,) * 3)[2] + dw_head)
    skip = fastener_parts | {part.name}
    stack = _clamp_stack(parts, skip, p0, e, r_probe, s_plate - tol, s_n + tol, None)
    plates = []
    for name, ivs in stack.items():
        for a, b in ivs:
            plates.append({"part": name, "s0": a - s_plate, "s1": b - s_plate})
    plates.sort(key=lambda q: q["s0"])
    origin = p0 + s_plate * e
    connection = "fitted" if fn.kind == "fitted" else ("nut" if nut is not None else "tapped")
    thread_end = 0.0
    nut_s = (s_n - s_plate) if nut is not None else 0.0
    if not plates:
        warnings.append(f"{bname} ({part.name}): no clamped parts found")
        return None
    if head_cone and plates[0]["s0"] <= head_cone + tol:
        plates[0]["s0"] = 0.0                     # countersunk head lies in the countersink of the first part
    elif abs(plates[0]["s0"]) > tol:
        warnings.append(f"{bname}: there is a {plates[0]['s0']:.3g} mm gap between the head and part '{plates[0]['part']}' – the head does not seat")
    if connection == "tapped":
        last = plates[-1]
        if last["s1"] < s_tip - s_plate - tol:
            warnings.append(f"{bname}: bolt end does not reach into any threaded part")
            return None
        tapped = plates.pop()
        nut_s = tapped["s0"]
        thread_end = min(s_tip - s_plate, tapped["s1"])
        nut_part = tapped["part"]
        if not plates:
            warnings.append(f"{bname}: bolt into thread without a clamped part")
            return None
    else:
        nut_part = plates[-1]["part"]
        if nut is not None and abs(plates[-1]["s1"] - nut_s) > tol:
            warnings.append(f"{bname}: nut does not seat on part '{nut_part}' (gap {nut_s - plates[-1]['s1']:.3g} mm)")
    # continuous sections of the same part, holes
    pq = {p.name: p for p in parts}
    for pl in plates:
        pl["hole"] = _hole(pq[pl["part"]], origin, e, pl["s0"], pl["s1"], tol)
        if pl["hole"] is None:
            w = _slot_width(pq[pl["part"]], origin, e, pl["s0"], pl["s1"], tol, d)
            if w is not None:
                pl["hole"], pl["slot"] = w, True
            else:
                warnings.append(f"{bname}: no coaxial hole or slot found in part '{pl['part']}'")
    for i in range(len(plates) - 1):
        gap = plates[i + 1]["s0"] - plates[i]["s1"]
        if gap > tol:
            warnings.append(f"{bname}: there is a {gap:.3g} mm clearance in CAD between parts '{plates[i]['part']}' and '{plates[i + 1]['part']}' – FEA clamped stack (parts attached to the bolt, clearance closed after tightening)")
    if nut is not None:
        cad.append(nut[0].name)
        nd = tables.nut_dims(d, nut[3].standard)
        nut_dw = nut_w[-1]["d2"] if nut_w else nd["dw"]
    else:
        nut_dw = 0.0
    wl = [{"part": w["part"], "side": "head", "h": w["h"], "d1": w["d1"], "d2": w["d2"]} for w in head_w]
    wl += [{"part": w["part"], "side": "nut", "h": w["h"], "d1": w["d1"], "d2": w["d2"]} for w in nut_w]
    cad += [w["part"] for w in wl]
    strength = fn.strength
    note = ""
    if strength is None:
        strength = "8.8"
        note = "strength class missing in the name – 8.8 assumed"
        warnings.append(f"{bname} ({part.name}): {note}")
    length = fn.length or round(smax - smin, 3)
    b = BoltSpec(name=bname, standard=fn.standard, size=d, pitch=fn.pitch, length=length, strength=strength,
                 head=fn.head, connection=connection, origin=[float(v) for v in origin],
                 axis=[float(v) for v in e], head_part=plates[0]["part"], head_dw=float(dw_head),
                 nut_part=nut_part, nut_s=float(nut_s), nut_dw=float(nut_dw), thread_end=float(thread_end),
                 plates=plates, washers=wl, cad_parts=cad, note=note, head_cone=float(head_cone),
                 nut_m=float(nut[2] - nut[1]) if nut is not None else 0.0)
    hole = b.hole
    if hole is not None and hole >= b.head_dw - 1e-6:
        warnings.append(f"{bname}: hole Ø{hole:g} is larger than the bearing face Ø{b.head_dw:g}")
    if connection == "tapped" and b.thread_end - b.nut_s < 0.8 * d:
        warnings.append(f"{bname}: short thread engagement {b.thread_end - b.nut_s:.3g} mm (< 0.8 d)")
    _ = th
    return b


def bolt_from_hole(parts: list[Part], part_index: int, face_index: int, size: float, strength: str = "8.8",
                   standard: str = "ISO 4017", washers: bool = True, name: str = "Bolt") -> tuple[BoltSpec | None, list[str]]:
    """Bolt with a nut in a selected hole (model without bolts): all the coaxial parts with the hole are clamped."""
    warnings: list[str] = []
    face = parts[part_index].faces()[face_index]
    cyl = cylinder_axis(face)
    if cyl is None:
        return None, ["The selected face is not a cylindrical hole"]
    p, e, r = cyl
    e = e / np.linalg.norm(e)
    d = float(size)
    tol = max(0.05, 0.02 * d)
    head = tables.head_dims(d, "hex", standard)
    wd = tables.washer_dims(d)
    dw = wd["d2"] if washers else head["dw"]
    r_probe = 0.5 * (r + 0.5 * dw)
    span = 10 * max(_bbox(q.shape)[3:].max() - _bbox(q.shape)[:3].min() for q in parts)
    stack = _clamp_stack(parts, set(), p, e, r_probe, -span, span, None)
    ivs = sorted((a, b, n) for n, iv in stack.items() for a, b in iv
                 if _hole(next(q for q in parts if q.name == n), p, e, a, b, tol))
    # continuous block around the chosen face
    s_face = float(((_tri_nodes(face) - p) @ e).mean())
    blk = [iv for iv in ivs if iv[0] - tol <= s_face <= iv[1] + tol]
    if not blk:
        return None, ["The selected hole does not lie in any part"]
    lo, hi = blk[0][0], blk[0][1]
    changed = True
    while changed:
        changed = False
        for a, b, _ in ivs:
            if a <= hi + tol and b > hi + tol:
                hi, changed = b, True
            if b >= lo - tol and a < lo - tol:
                lo, changed = a, True
    plates = [{"part": n, "s0": a - lo, "s1": b - lo,
               "hole": _hole(next(q for q in parts if q.name == n), p, e, a, b, tol)}
              for a, b, n in ivs if a >= lo - tol and b <= hi + tol]
    if len({pl["part"] for pl in plates}) < 1:
        return None, ["No clamped parts found"]
    wl = []
    if washers:
        wl = [{"part": "", "side": "head", "h": wd["h"], "d1": wd["d1"], "d2": wd["d2"]},
              {"part": "", "side": "nut", "h": wd["h"], "d1": wd["d1"], "d2": wd["d2"]}]
    origin = p + lo * e
    nd = tables.nut_dims(d, standard)
    b = BoltSpec(name=name, standard=standard, size=d, strength=strength, head="hex", connection="nut",
                 origin=[float(v) for v in origin], axis=[float(v) for v in e], head_part=plates[0]["part"],
                 head_dw=float(dw), nut_part=plates[-1]["part"], nut_s=float(hi - lo),
                 nut_dw=float(wd["d2"] if washers else nd["dw"]), plates=plates, washers=wl, cad_parts=[],
                 note="entered manually from the hole (bolt is not in the model)")
    if b.hole and b.hole >= b.head_dw:
        warnings.append(f"{name}: hole Ø{b.hole:g} is larger than the bearing face Ø{b.head_dw:g}")
    return b, warnings


def edge_distances(part: Part, center, axis, direction, hole_r: float) -> tuple[float, float]:
    """(e1, e2): distance of the centre of the hole from the edge of the part in the direction `direction` and the smallest perpendicular to it.

    Rays in the mid-plane of the part (center = point on the axis at the middle of the thickness); the edge = the first exit
    from the material behind the hole. Infinity if the ray does not leave the part (open shape).
    """
    c = np.asarray(center, float)
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    u = np.asarray(direction, float)
    u = u - (u @ ax) * ax
    if np.linalg.norm(u) < 1e-12:
        return math.inf, math.inf
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    size = float(np.linalg.norm(_bbox(part.shape)[3:] - _bbox(part.shape)[:3]))

    def ray(dvec) -> float:
        """The last exit from the material = edge of the part (holes of other bolts are skipped, those are handled by p1/p2)."""
        ivs = [iv for iv in _line_intervals(part.shape, c, dvec, hole_r * 0.5, 2 * size + hole_r)
               if iv[1] > hole_r + 1e-6]
        return ivs[-1][1] if ivs else math.inf
    return ray(u), min(ray(v), ray(-v))


# ===================================================================== svary

def weld_geometry(parts: list[Part], weld_part: str) -> dict:
    """Contact faces of the weld with the other parts and an estimate of a, lengths (for display before the calculation).

    Fillet weld with two contact faces enclosing the angle θ (legs z1, z2, length L):
      V = ½ z1 z2 sin θ · L, A1 = z1 L, A2 = z2 L  →  L = A1 A2 sin θ / (2V),
      a = z1 z2 sin θ / √(z1² + z2² − 2 z1 z2 cos θ).
    """
    pw = next(p for p in parts if p.name == weld_part)
    vol = pw.volume()
    bbw = bbox(pw.shape)
    size = float(np.linalg.norm(bbw[3:] - bbw[:3]))
    tol = max(1e-3, 1e-4 * size)
    groups: dict[str, dict] = {}
    for f in pw.faces():
        pts = _face_samples(f)
        if len(pts) == 0:
            continue
        pr = face_props(f)
        for q in parts:
            if q.name == weld_part:
                continue
            bb = _bbox(q.shape)
            if np.any(bb[3:] < bbw[:3] - tol) or np.any(bb[:3] > bbw[3:] + tol):
                continue
            if all(_dist_to_shape(x, q.shape) <= tol for x in pts):
                g = groups.setdefault(q.name, {"area": 0.0, "n": np.zeros(3)})
                g["area"] += pr["area"]
                g["n"] += pr["area"] * np.asarray(pr["normal"], float)
                break
    out = {"part": weld_part, "volume": vol, "fusion": {k: v["area"] for k, v in groups.items()},
           "kind": "?", "a": None, "length": None, "theta_deg": None}
    if len(groups) >= 2:
        (n1, g1), (n2, g2) = sorted(groups.items(), key=lambda kv: -kv[1]["area"])[:2]
        u1 = g1["n"] / max(np.linalg.norm(g1["n"]), 1e-30)
        u2 = g2["n"] / max(np.linalg.norm(g2["n"]), 1e-30)
        cos_t = float(np.clip(-u1 @ u2, -1, 1))
        if np.linalg.norm(g1["n"]) < 0.3 * g1["area"] or np.linalg.norm(g2["n"]) < 0.3 * g2["area"]:
            cos_t = 0.0                           # curved contact face (weld around a tube) – 90° assumed
        if cos_t < -0.95:                         # parallel contact faces = butt weld
            out["kind"] = "butt"
        else:
            sin_t = math.sqrt(max(1 - cos_t ** 2, 1e-12))
            A1, A2 = g1["area"], g2["area"]
            L = A1 * A2 * sin_t / (2 * vol)
            z1, z2 = A1 / L, A2 / L
            a = z1 * z2 * sin_t / math.sqrt(z1 ** 2 + z2 ** 2 - 2 * z1 * z2 * cos_t)
            out.update(kind="fillet", a=a, length=L, z=[z1, z2], theta_deg=math.degrees(math.acos(cos_t)))
    if out["kind"] == "butt" or len(groups) == 1:
        A1 = max(g["area"] for g in groups.values()) if groups else 0.0
        ext = np.sort(bbw[3:] - bbw[:3])
        L = float(ext[-1])
        out.update(kind="butt", length=L, a=A1 / L if L > 0 else None)
    return out
