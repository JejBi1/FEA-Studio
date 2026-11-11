"""3D window of FEA Studio: OccView from the CAD viewer + parts, face selection, face sets, mesh, contours.

The window keeps the face selection itself (list of (part, face)); it does not rely on the AIS selection – it is
stable on redraw and can be set by the program (highlighting a constraint in the tree).

Selection filter (pick_filter): "face" = face, "body" = all faces of a part, "hole" = whole hole / pin
(coaxial cylindrical faces of the same radius), "edge" = edge (in "faces" mode a click adds / removes the edge from the edge
selection – force on an edge; always info too: length, type, for a circle the centre and radius – signal edgePicked), "point" = point
(a vertex, or a point on a face projected exactly onto the CAD face – force to a point). The identity of a selection is always (part index,
face / edge index) – also for repeated parts (each part has its own AIS object and a shape in global position). In
"model" mode (steps without face selection) a click selects whole parts (signal partsPicked), the Edge filter only informs there.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from fea_core.i18n import decimal

import numpy as np
from PySide6.QtCore import Qt, Signal

from OCP.AIS import AIS_ColorScale, AIS_Shape, AIS_ViewCubeOwner, AIS_TextLabel
from OCP.Aspect import Aspect_TOTP_LEFT_UPPER, Aspect_TOCSD_USER, Aspect_TOM_BALL
from OCP.BRep import BRep_Builder
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Graphic3d import (Graphic3d_TMF_2d, Graphic3d_TransformPers, Graphic3d_ZLayerId_Top,
                           Graphic3d_ZLayerId_TopOSD)
from OCP.Prs3d import Prs3d_PointAspect
from OCP.TCollection import TCollection_ExtendedString

from fea_core.i18n import tr
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Compound
from OCP.collections import Vec2_int as Graphic3d_Vec2i

from .contour import BAND_COLORS, contour_bands, triangle_bands, triangles_face
from .viewer3d import OccView, rgb

# light part colours (the user does not want dark models)
PART_COLORS = [(214, 219, 226), (250, 226, 150), (190, 215, 240), (205, 232, 200), (240, 205, 205),
               (225, 210, 240), (245, 222, 195), (200, 230, 230)]
SEL_COLOR = (255, 140, 0)
EDGE_COLOR = (230, 0, 120)
FACE_MODE = AIS_Shape.SelectionMode_s(TopAbs_FACE)
EDGE_MODE = AIS_Shape.SelectionMode_s(TopAbs_EDGE)
VERTEX_MODE = AIS_Shape.SelectionMode_s(TopAbs_VERTEX)
PICK_FILTERS = ("face", "body", "hole", "edge", "point")


def edge_info(edge) -> dict:
    """Description of an edge for the status bar: type, length, end points and the centre of the edge; for a circle / arc the centre,
    radius, axis and angle, for an ellipse the centre and semi-axes. "point" = point for a remote load (circle / ellipse centre,
    otherwise the edge centre).
    """
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Line

    def xyz(p):
        return np.array([p.X(), p.Y(), p.Z()])
    c = BRepAdaptor_Curve(edge)
    t0, t1 = c.FirstParameter(), c.LastParameter()
    props = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, props)
    d = {"length": float(props.Mass()), "start": xyz(c.Value(t0)), "end": xyz(c.Value(t1)),
         "mid": xyz(c.Value(0.5 * (t0 + t1)))}
    ct = c.GetType()
    if ct == GeomAbs_Line:
        d["type"], d["label"] = "line", "line"
    elif ct == GeomAbs_Circle:
        circ = c.Circle()
        ang = math.degrees(abs(t1 - t0))
        full = abs(ang - 360.0) < 1e-6
        ax = circ.Axis().Direction()
        d.update(type="circle", label="circle" if full else decimal(f"arc {ang:.4g}°"),
                 center=xyz(circ.Location()), radius=float(circ.Radius()), axis=np.array([ax.X(), ax.Y(), ax.Z()]),
                 angle=ang, full=full)
    elif ct == GeomAbs_Ellipse:
        el = c.Ellipse()
        d.update(type="ellipse", label="elipsa", center=xyz(el.Location()), major=float(el.MajorRadius()),
                 minor=float(el.MinorRadius()))
    else:
        d["type"], d["label"] = "curve", "curve (spline)"
    d["point"] = d.get("center", d["mid"])
    return d


def _compound(shapes):
    b = BRep_Builder()
    c = TopoDS_Compound()
    b.MakeCompound(c)
    for s in shapes:
        b.Add(c, s)
    return c


class FeaView(OccView):
    facesPicked = Signal(list)     # [(part index, face index)] – current face selection
    probed = Signal(object)        # (x, y, z) click point on the results
    partsPicked = Signal(list)     # [part index] – click on parts in "model" mode (Materials step)
    edgePicked = Signal(object)    # edge description (edge_info + "part", "edge") or None (Edge filter)
    edgesPicked = Signal(list)     # [(part index, edge index)] – edge selection (Edge filter in faces mode)
    pointsPicked = Signal(list)    # [(part index, (x, y, z))] – point selection (Point filter)
    pickNote = Signal(str)         # short message about the selection (e.g. the Hole filter outside a cylindrical face)


    # ------------------------------------------------------------ batched redraw
    _batch_depth = 0
    _batch_dirty = False

    @contextmanager
    def batch(self):
        """Display changes without continuous redraw – one redraw at the end (opening a project called
        UpdateCurrentViewer 51× = 2.2 s).
        """
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._batch_dirty:
                self._batch_dirty = False
                self.ctx.UpdateCurrentViewer()

    def _update(self):
        if self._batch_depth:
            self._batch_dirty = True
        else:
            self.ctx.UpdateCurrentViewer()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parts = []
        self.part_ais = []
        self.part_faces = []           # part -> list of TopoDS_Face
        self.mode = "model"            # "model" | "faces" | "results"
        self.pick_filter = "face"      # "face" | "body" | "hole" | "edge" (see PICK_FILTERS)
        self._applied_pick_mode = None
        self._parts_transparency = 0.0
        self._sel: list[tuple[int, int]] = []
        self._sel_objs = []
        self._sets: dict[str, list] = {}
        self._result_objs = []
        self._marks: dict[str, list] = {}   # named markers in the results (a new one replaces the old one)
        self._scale = None
        self._edges: dict[int, list] = {}   # part -> unique edges (TopExp_Explorer order = edge index)
        self._edge = None                   # selected edge (edge_info)
        self._edge_objs = []
        self._sel_edges: list[tuple[int, int]] = []          # edge selection (force on an edge)
        self._sel_points: list[tuple[int, tuple]] = []       # point selection (force to a point)
        self._sel_ep_objs = []
        self.parts_visible = True
        self._swallow_release = False     # release of the button after a double click (otherwise it would select the face again)
        self._persist: dict[str, list] = {}  # persistent objects (frame system) – not deleted by clear_sets / show_sets
        self.extra_bbox = None               # (lo, hi) of other objects (members) for model_size
        self.screen_pick = None              # fn(x, y in device pixels) → True = click handled (member / node selection)
        self.selectionChanged.connect(self._on_ais_selection)
        self.zoomSelected.connect(self._on_double_click)

    # ------------------------------------------------------------ parts
    def set_parts(self, parts):
        for o in self.part_ais:
            self.ctx.Remove(o, False)
        self.clear_sets()
        self.clear_results()
        self._sel = []
        self._show_sel()
        self.clear_edge(emit=True)
        self._edges = {}
        self._sel_edges, self._sel_points = [], []
        self.parts, self.part_ais, self.part_faces = list(parts), [], []
        self._ghost = set()
        self._hidden = set()
        self.parts_visible = True
        self._parts_transparency = 0.0
        self._applied_pick_mode = None
        for i, p in enumerate(self.parts):
            from fea_core.geometry import bbox
            bb = bbox(p.shape)
            size = float(np.max(bb[3:] - bb[:3]))
            # Display mesh only (not the calculation FEM mesh): a coarser tolerance speeds up loading of assemblies and reduces the triangle count
            # while keeping a smooth CAD preview.
            BRepMesh_IncrementalMesh(p.shape, min(max(size * 0.002, 0.02), 0.8), False, 0.4, True)
            o = AIS_Shape(p.shape)
            o.SetColor(rgb(*self.part_color(i)))
            self.apply_edge_style(o, self.display_style == "shaded_edges")
            self.ctx.Display(o, 0 if self.display_style == "wire" else 1, 0, False)
            self.part_ais.append(o)
            self.part_faces.append(p.faces())
        self._apply_mode()
        self._show_ep()
        self.fit_all()

    def part_color(self, i: int):
        cols = getattr(self, "_part_colors", None)
        if cols and i < len(cols) and cols[i] is not None:
            return cols[i]
        return PART_COLORS[i % len(PART_COLORS)]

    display_style = "shaded_edges"          # shaded_edges | shaded | wire (bar above 3D: Shaded ▾)
    user_transparency = 0.0                 # part transparency chosen by the user (bar above 3D: Transparent ▾)

    def set_display_style(self, style: str):
        """Part style: shaded with face edges, shaded only, wire model."""
        if style == self.display_style:
            return
        self.display_style = style
        for i, o in enumerate(self.part_ais):
            self.apply_edge_style(o, style == "shaded_edges")
            self.ctx.SetDisplayMode(o, 0 if style == "wire" else 1, False)
            self.ctx.Redisplay(o, False)
        self._update()

    def set_user_transparency(self, t: float):
        """Transparency of all parts (joints and holes inside an assembly are visible); 0 = opaque."""
        t = float(t)
        if t == self.user_transparency:
            return
        self.user_transparency = t
        ghost = getattr(self, "_ghost", set())
        for i, o in enumerate(self.part_ais):
            tr = max(self._parts_transparency, t, self.GHOST if i in ghost else 0.0)
            o.SetTransparency(tr) if tr else o.UnsetTransparency()
            self.ctx.Redisplay(o, False)
        self._update()

    def set_part_colors(self, colors):
        """Part colours (RGB list per part, None = default colour by order) – e.g. colour by material."""
        new = list(colors) if colors else None
        if new == getattr(self, "_part_colors", None):
            return
        self._part_colors = new
        for i, o in enumerate(self.part_ais):
            o.SetColor(rgb(*self.part_color(i)))
            self.ctx.Redisplay(o, False)
        self._update()

    def set_part_poses(self, poses):
        """Mechanism animation (Motion step): {part index: (R 3×3, T)} – x = R·X + T as a local transformation of the AIS
        part; None or empty = all parts back to the CAD position.
        """
        from OCP.gp import gp_Trsf
        from OCP.TopLoc import TopLoc_Location
        for i, o in enumerate(self.part_ais):
            tr = gp_Trsf()
            if poses and i in poses:
                R, T = poses[i]
                R = np.asarray(R, float)
                tr.SetValues(R[0, 0], R[0, 1], R[0, 2], float(T[0]), R[1, 0], R[1, 1], R[1, 2], float(T[1]),
                             R[2, 0], R[2, 1], R[2, 2], float(T[2]))
            self.ctx.SetLocation(o, TopLoc_Location(tr))
        self._part_posed = bool(poses)
        self._update()

    GHOST = 0.85                   # transparency of the parts excluded from the calculation

    def set_ghost(self, indices):
        """Parts excluded from the calculation shown translucent (kept across steps – show_parts keeps it)."""
        self._ghost = set(indices)
        for i, o in enumerate(self.part_ais):
            if i in self._ghost:
                o.SetTransparency(self.GHOST)
            elif o.IsTransparent():
                o.UnsetTransparency()
            if self.ctx.IsDisplayed(o):
                self.ctx.Redisplay(o, False)
        self._update()

    def set_hidden(self, indices):
        """Parts hidden in the 3D view (display only; kept across steps – show_parts does not show them)."""
        hidden = set(indices)
        if hidden == getattr(self, "_hidden", set()):
            return
        self._hidden = hidden
        if self.parts_visible:
            ghost = getattr(self, "_ghost", set())
            for i, o in enumerate(self.part_ais):
                if i in hidden:
                    if self.ctx.IsDisplayed(o):
                        self.ctx.Erase(o, False)
                elif not self.ctx.IsDisplayed(o):
                    tr = max(self._parts_transparency, self.GHOST if i in ghost else 0.0)
                    o.SetTransparency(tr) if tr else o.UnsetTransparency()
                    self.ctx.Display(o, 1, 0, False)
        self._applied_pick_mode = None
        self._apply_mode()
        self._update()

    def show_parts(self, on: bool, transparency: float = 0.0):
        on = bool(on)
        transparency = float(transparency)
        if on == self.parts_visible and transparency == self._parts_transparency:
            return
        self.parts_visible = on
        self._parts_transparency = transparency
        self._applied_pick_mode = None
        ghost = getattr(self, "_ghost", set())
        hidden = getattr(self, "_hidden", set())
        for i, o in enumerate(self.part_ais):
            if on and i not in hidden:
                tr = max(transparency, self.user_transparency, self.GHOST if i in ghost else 0.0)
                o.SetTransparency(tr) if tr else o.UnsetTransparency()
                self.ctx.Display(o, 1, 0, False)
            else:
                self.ctx.Erase(o, False)
        self._apply_mode()
        self._update()

    def set_mode(self, mode: str):
        """model = selection of whole parts, faces = selection of faces, results = probe on the contours."""
        self.mode = mode
        if mode == "results":
            self.clear_edge(emit=True)
        if mode != "faces" and (self._sel_edges or self._sel_points):   # edge / point selection only in the steps with selection
            self._sel_edges, self._sel_points = [], []
            self._show_ep()
        self._apply_mode()

    def set_filter(self, kind: str):
        """Selection filter by click: face | body | hole | edge (the face selection stays, only what a click selects changes)."""
        if kind not in PICK_FILTERS:
            raise ValueError(kind)
        self.pick_filter = kind
        if kind != "edge":
            self.clear_edge()
        self._apply_mode()

    def _edge_picking(self) -> bool:
        return self.pick_filter == "edge" and self.mode in ("faces", "model")

    def _point_picking(self) -> bool:
        return self.pick_filter == "point" and self.mode == "faces"

    def _apply_mode(self):
        # Transitions between steps often keep the selection mode the same. Keep the clearing of the
        # selection, but do not switch the modes of all AIS objects in OCCT needlessly.
        self.ctx.ClearSelected(False)
        signature = (self.mode, self.pick_filter)
        if signature == self._applied_pick_mode:
            return
        edge = self._edge_picking()
        for o in self.part_ais:
            if not self.ctx.IsDisplayed(o):
                continue
            self.ctx.Deactivate(o)
            if edge:
                self.ctx.Activate(o, EDGE_MODE)
            elif self._point_picking():                   # a vertex has priority, otherwise a point on the face
                self.ctx.Activate(o, VERTEX_MODE)
                self.ctx.Activate(o, FACE_MODE)
            elif self.mode == "faces":
                self.ctx.Activate(o, 0 if self.pick_filter == "body" else FACE_MODE)
            elif self.mode == "model":
                self.ctx.Activate(o, 0)
        if hasattr(self.ctx, "SetPixelTolerance"):      # an edge / vertex is easier to hit with a larger tolerance
            self.ctx.SetPixelTolerance(int(6 * self.dpr()) if edge or self._point_picking() else 3)
        self._applied_pick_mode = signature

    # ------------------------------------------------------------ selection of whole parts (model mode)
    def selected_parts(self) -> list[int]:
        return [self.part_ais.index(o) for o in self.selected_objects() if o in self.part_ais]

    def set_selected_parts(self, indices):
        """Highlight parts as selected (synchronisation with the part list in the Materials step); only in model mode."""
        if self.mode != "model" or self._edge_picking():
            return
        self.ctx.ClearSelected(False)
        for i in indices:
            if 0 <= i < len(self.part_ais) and self.ctx.IsDisplayed(self.part_ais[i]):
                self.ctx.AddOrRemoveSelected(self.part_ais[i], False)
        self._update()

    def _on_ais_selection(self):
        if self.mode == "model" and not self._edge_picking():
            self.partsPicked.emit(self.selected_parts())

    # ------------------------------------------------------------ edge selection (Edge filter)
    def part_edges(self, pi: int) -> list:
        """Unique edges of a part (index = order of first occurrence in TopExp_Explorer – stable for a given shape)."""
        if pi not in self._edges:
            from fea_core.geometry import _cast
            from OCP.TopoDS import TopoDS
            as_edge = _cast(TopoDS, "Edge")
            out, seen = [], {}
            exp = TopExp_Explorer(self.parts[pi].shape, TopAbs_EDGE)
            while exp.More():
                e = as_edge(exp.Current())
                bucket = seen.setdefault(hash(e), [])
                if not any(e.IsSame(o) for o in bucket):
                    bucket.append(e)
                    out.append(e)
                exp.Next()
            self._edges[pi] = out
        return self._edges[pi]

    def selected_edge(self) -> dict | None:
        return self._edge

    def select_edge(self, pi: int, ei: int, emit: bool = True):
        """Programmatic edge selection (part, edge index)."""
        e = self.part_edges(pi)[ei]
        info = edge_info(e)
        info.update(part=pi, edge=ei, shape=e)
        self._edge = info
        self._show_edge()
        if emit:
            self.edgePicked.emit(info)

    def clear_edge(self, emit: bool = False):
        for o in self._edge_objs:
            self.ctx.Remove(o, False)
        self._edge_objs = []
        had = self._edge is not None
        self._edge = None
        self._update()
        if emit and had:
            self.edgePicked.emit(None)

    def _pick_edge(self):
        """Click with the Edge filter: information about the edge; in faces mode it also adds / removes the edge in the edge selection."""
        owner = self.ctx.DetectedOwner() if self.ctx.HasDetected() else None
        obj = self.ctx.DetectedInteractive() if owner is not None else None
        collect = self.mode == "faces"
        if owner is None or obj not in self.part_ais or not hasattr(owner, "Shape") or not owner.HasShape() \
                or owner.Shape().ShapeType() != TopAbs_EDGE:
            self.clear_edge(emit=True)
            if collect and self._sel_edges:              # click on empty space = clear the edge selection
                self.set_selected_edges([], emit=True)
            return
        pi = self.part_ais.index(obj)
        s = owner.Shape()
        ei = next((k for k, e in enumerate(self.part_edges(pi)) if e.IsSame(s)), None)
        if ei is None:
            self.clear_edge(emit=True)
            return
        try:
            self.select_edge(pi, ei)
        except Exception as ex:  # noqa: BLE001 – an exception in the mouse handler would crash Qt
            self.clear_edge(emit=True)
            self.pickNote.emit(f"The edge cannot be evaluated: {ex}")
        if collect:
            key = (pi, ei)
            sel = [e for e in self._sel_edges if e != key] if key in self._sel_edges else self._sel_edges + [key]
            self.set_selected_edges(sel, emit=True)

    # ------------------------------------------------------------ edge and point selection (force on an edge / to a point)
    def selected_edges(self) -> list[tuple[int, int]]:
        return list(self._sel_edges)

    def set_selected_edges(self, edges, emit=False):
        self._sel_edges = list(dict.fromkeys(edges))
        self._show_ep()
        if emit:
            self.edgesPicked.emit(list(self._sel_edges))

    def selected_points(self) -> list[tuple[int, tuple]]:
        return list(self._sel_points)

    def set_selected_points(self, points, emit=False):
        self._sel_points = [(int(pi), tuple(float(c) for c in q)) for pi, q in points]
        self._show_ep()
        if emit:
            self.pointsPicked.emit(list(self._sel_points))

    def clear_selection(self):
        """Clear the face, edge and point selection (after creating a study item)."""
        self._sel = []
        self._show_sel()
        self.set_selected_edges([], emit=True)
        self.set_selected_points([], emit=True)

    def _pick_point(self):
        """Click with the Point filter: a vertex (exact), or a face point projected onto the CAD face; a repeated click on the same
        point removes it, a click on empty space clears the point selection.
        """
        from fea_core.geometry import project_to_surface
        owner = self.ctx.DetectedOwner() if self.ctx.HasDetected() else None
        obj = self.ctx.DetectedInteractive() if owner is not None else None
        if owner is None or obj not in self.part_ais or not hasattr(owner, "Shape") or not owner.HasShape():
            self.set_selected_points([], emit=True)
            return
        pi = self.part_ais.index(obj)
        sh = owner.Shape()
        try:
            if sh.ShapeType() == TopAbs_VERTEX:
                from OCP.BRep import BRep_Tool
                from OCP.TopoDS import TopoDS
                from fea_core.geometry import _cast
                q = BRep_Tool.Pnt_s(_cast(TopoDS, "Vertex")(sh))
                xyz = (q.X(), q.Y(), q.Z())
            elif sh.ShapeType() == TopAbs_FACE:
                pp = self._picked_point()
                if pp is None:
                    return
                proj, _ = project_to_surface(sh, (pp.X(), pp.Y(), pp.Z()))
                xyz = tuple(float(c) for c in proj)
            else:
                return
        except Exception as ex:  # noqa: BLE001 – an exception in the mouse handler would crash Qt
            self.pickNote.emit(f"The point cannot be evaluated: {ex}")
            return
        tol = 1e-3 * max(self.model_size(), 1.0) * 0.01
        keep = [(i, q) for i, q in self._sel_points if not (i == pi and np.linalg.norm(np.subtract(q, xyz)) <= tol)]
        if len(keep) == len(self._sel_points):
            keep.append((pi, xyz))
        self.set_selected_points(keep, emit=True)
        self.pickNote.emit("Bod (" + "; ".join(decimal(f"{c:.4f}") for c in xyz) + ") mm – " +
                           ("vertex" if sh.ShapeType() == TopAbs_VERTEX else "on face"))

    def _show_ep(self):
        """Highlight of selected edges (strong orange) and points (sphere)."""
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        from OCP.gp import gp_Pnt
        for o in self._sel_ep_objs:
            self.ctx.Remove(o, False)
        self._sel_ep_objs = []
        if self._sel_edges:
            shapes = []
            for pi, ei in self._sel_edges:
                if 0 <= pi < len(self.parts) and 0 <= ei < len(self.part_edges(pi)):
                    shapes.append(self.part_edges(pi)[ei])
            if shapes:
                o = AIS_Shape(_compound(shapes))
                o.SetColor(rgb(*SEL_COLOR))
                o.SetWidth(5.0)
                o.SetZLayer(Graphic3d_ZLayerId_TopOSD)
                self.ctx.Display(o, 0, -1, False)
                self.ctx.Deactivate(o)
                self._sel_ep_objs.append(o)
        for _, q in self._sel_points:
            v = AIS_Shape(BRepBuilderAPI_MakeVertex(gp_Pnt(*q)).Vertex())
            v.Attributes().SetPointAspect(Prs3d_PointAspect(Aspect_TOM_BALL, rgb(*SEL_COLOR), 4.0))
            v.SetZLayer(Graphic3d_ZLayerId_TopOSD)
            self.ctx.Display(v, 0, -1, False)
            self.ctx.Deactivate(v)
            self._sel_ep_objs.append(v)
        self._update()

    def _text_label(self, text: str, p, color) -> AIS_TextLabel:
        """Label in 3D above the model (point p = gp_Pnt), displayed and not selectable."""
        lbl = AIS_TextLabel()
        lbl.SetText(TCollection_ExtendedString(tr(text), True))
        lbl.SetPosition(p)
        lbl.SetColor(rgb(*color))
        lbl.SetHeight(14 * self.dpr())
        lbl.SetFont("Arial")
        lbl.SetZLayer(Graphic3d_ZLayerId_TopOSD)
        self.ctx.Display(lbl, False)
        self.ctx.Deactivate(lbl)
        return lbl

    def _show_edge(self):
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        from OCP.gp import gp_Pnt
        for o in self._edge_objs:
            self.ctx.Remove(o, False)
        self._edge_objs = []
        info = self._edge
        if info is not None:
            o = AIS_Shape(info["shape"])
            o.SetColor(rgb(*EDGE_COLOR))
            o.SetWidth(4.0)
            o.SetZLayer(Graphic3d_ZLayerId_TopOSD)
            self.ctx.Display(o, 0, -1, False)
            self.ctx.Deactivate(o)
            self._edge_objs.append(o)
            p = gp_Pnt(*map(float, info["point"]))
            v = AIS_Shape(BRepBuilderAPI_MakeVertex(p).Vertex())
            v.Attributes().SetPointAspect(Prs3d_PointAspect(Aspect_TOM_BALL, rgb(*EDGE_COLOR), 3.0))
            v.SetZLayer(Graphic3d_ZLayerId_TopOSD)
            self.ctx.Display(v, 0, -1, False)
            self.ctx.Deactivate(v)
            self._edge_objs.append(v)
            if "center" in info:              # centre of a circle / ellipse with a label
                txt = decimal(f"R {info['radius']:.4g}" if "radius" in info else "center")
                self._edge_objs.append(self._text_label(txt, p, (150, 0, 80)))
        self._update()

    # ------------------------------------------------------------ face selection
    def selected_faces(self) -> list[tuple[int, int]]:
        return list(self._sel)

    def set_selected_faces(self, faces, emit=False):
        self._sel = list(dict.fromkeys(faces))
        self._show_sel()
        if emit:
            self.facesPicked.emit(list(self._sel))

    def _show_sel(self):
        for o in self._sel_objs:
            self.ctx.Remove(o, False)
        self._sel_objs = []
        if self._sel:
            faces = [self.part_faces[pi][fi] for pi, fi in self._sel]
            o = AIS_Shape(_compound(faces))
            o.SetColor(rgb(*SEL_COLOR))
            o.SetTransparency(0.25)
            o.SetZLayer(Graphic3d_ZLayerId_Top)
            self.ctx.Display(o, 1, -1, False)
            self.ctx.Deactivate(o)
            self._sel_objs.append(o)
        self._update()

    def _face_at_cursor(self):
        owner = self.ctx.DetectedOwner() if self.ctx.HasDetected() else None
        if owner is None or not hasattr(owner, "Shape") or not owner.HasShape():
            return None
        obj = self.ctx.DetectedInteractive()
        if obj not in self.part_ais:
            return None
        pi = self.part_ais.index(obj)
        s = owner.Shape()
        if s.ShapeType() != TopAbs_FACE:
            return None
        for fi, f in enumerate(self.part_faces[pi]):
            if f.IsSame(s):
                return pi, fi
        return None

    def _part_at_cursor(self):
        if not self.ctx.HasDetected():
            return None
        obj = self.ctx.DetectedInteractive()
        return self.part_ais.index(obj) if obj in self.part_ais else None

    def _group_at_cursor(self):
        """Faces that a click selects according to the filter: [face] | all faces of the part | whole hole; None = empty."""
        if self.pick_filter == "body":
            pi = self._part_at_cursor()
            return None if pi is None else [(pi, fi) for fi in range(len(self.part_faces[pi]))]
        hit = self._face_at_cursor()
        if hit is None:
            return None
        if self.pick_filter == "hole":
            from fea_core.geometry import coaxial_cylinders, cylinder_axis
            pi, fi = hit
            if cylinder_axis(self.part_faces[pi][fi]) is None:
                self.pickNote.emit("Hole filter: click the cylindrical face of a hole or pin.")
                return []
            return [(pi, j) for j in coaxial_cylinders(self.part_faces[pi], fi)]
        return [hit]

    # ------------------------------------------------------------ zoom to selection
    def has_selection(self) -> bool:
        return bool(self._sel or self._sel_edges or self._sel_points or self._edge is not None
                    or (self.mode == "model" and self.selected_parts()))

    def selection_box(self):
        """Bounding box (Bnd_Box) of the selection: faces, edges, points, selected edge (Edge filter), in model mode whole parts;
        None = nothing selected.
        """
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        from OCP.gp import gp_Pnt
        box = Bnd_Box()
        shapes = [self.part_faces[pi][fi] for pi, fi in self._sel
                  if 0 <= pi < len(self.part_faces) and 0 <= fi < len(self.part_faces[pi])]
        shapes += [self.part_edges(pi)[ei] for pi, ei in self._sel_edges
                   if 0 <= pi < len(self.parts) and 0 <= ei < len(self.part_edges(pi))]
        if self._edge is not None:
            shapes.append(self._edge["shape"])
        if self.mode == "model":
            shapes += [self.parts[i].shape for i in self.selected_parts() if 0 <= i < len(self.parts)]
        for s in shapes:
            BRepBndLib.Add_s(s, box)
        for _, q in self._sel_points:
            box.Add(gp_Pnt(*map(float, q)))
        if box.IsVoid():
            return None
        if self._sel_points and not shapes:             # a point alone: neighbourhood ~5 % of the model
            box.Enlarge(0.05 * max(self.model_size(), 1.0))
        return box

    def zoom_selection(self):
        """Zoom to the selection (double click, context menu); without a selection show all."""
        box = self.selection_box()
        if box is None:
            self.fit_all()
        else:
            self.fit_box(box)

    def _on_double_click(self):
        """Double click = zoom to the selection (step with face / part selection); in results a double click does not change the view."""
        if self.mode in ("faces", "model"):
            self.zoom_selection()

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton and self.mode in ("faces", "model"):
            self._swallow_release = True               # the first click has already selected – the second release does not change the selection
            if self.mode == "faces" and not self._edge_picking() and not self._point_picking():
                # faces under the cursor must stay selected even though the second click of the double click has already switched the selection
                # (the order of press/release/dblclick events differs by platform and QTest)
                x, y = self._px(e.position().toPoint())
                self.ctx.MoveTo(x, y, self.view, False)
                group = self._group_at_cursor()
                if group and any(g not in self._sel for g in group):
                    self._sel += [g for g in group if g not in self._sel]
                    self._show_sel()
                    self.facesPicked.emit(list(self._sel))
        super().mouseDoubleClickEvent(e)

    def mousePressEvent(self, e):
        self._swallow_release = False                  # a new press = a new click
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if self._swallow_release and e.button() == Qt.LeftButton:
            self._swallow_release = False
            self._dragging = False
            self._press_pos = None
            return
        edge = self._edge_picking()
        if e.button() != Qt.LeftButton or (self.mode == "model" and not edge) or self._dragging:
            return super().mouseReleaseEvent(e)
        x, y = self._px(e.position().toPoint())
        self._dragging = False
        self._press_pos = None
        if self.screen_pick is not None and self.mode != "results":
            try:
                if self.screen_pick(x, y):           # node / member of the frame under the cursor
                    return
            except Exception:  # noqa: BLE001 – an exception in the mouse handler would crash Qt
                pass
        self.ctx.MoveTo(x, y, self.view, False)
        owner = self.ctx.DetectedOwner() if self.ctx.HasDetected() else None
        if owner is not None and isinstance(owner, AIS_ViewCubeOwner):
            self.cube.HandleClick(owner)
            self._anim.start()
            return
        if edge:
            self._pick_edge()
        elif self._point_picking():
            self._pick_point()
        elif self.mode == "faces":
            group = self._group_at_cursor()
            if group is None:                      # click on empty space = clear the selection
                self._sel = []
            elif group and all(g in self._sel for g in group):
                self._sel = [s for s in self._sel if s not in group]
            else:
                self._sel += [g for g in group if g not in self._sel]
            self._show_sel()
            self.facesPicked.emit(list(self._sel))
        elif self.mode == "results":
            p = self._picked_point()
            if p is not None and owner is not None:
                self.probed.emit((p.X(), p.Y(), p.Z()))

    # ------------------------------------------------------------ coloured face sets (constraints, loads)
    def set_face_set(self, key: str, faces, color, label: str | None = None, transparency=0.35):
        self.remove_face_set(key)
        objs = []
        shapes = [self.part_faces[pi][fi] for pi, fi in faces]
        if shapes:
            o = AIS_Shape(_compound(shapes))
            o.SetColor(rgb(*color))
            o.SetTransparency(transparency)
            o.SetZLayer(Graphic3d_ZLayerId_Top)
            self.ctx.Display(o, 1, -1, False)
            self.ctx.Deactivate(o)
            objs.append(o)
            if label:
                from fea_core.geometry import face_props
                from OCP.gp import gp_Pnt
                p = face_props(shapes[0])["point"]
                objs.append(self._text_label(label, gp_Pnt(*map(float, p)), [int(c * 0.6) for c in color]))
        self._sets[key] = objs
        self._update()

    def model_size(self) -> float:
        from fea_core.geometry import bbox
        boxes = [np.asarray(bbox(p.shape), float) for p in self.parts]
        if self.extra_bbox is not None:
            lo, hi = self.extra_bbox
            boxes.append(np.r_[np.asarray(lo, float), np.asarray(hi, float)])
        if not boxes:
            return 100.0
        bb = np.array(boxes)
        return float(np.linalg.norm(bb[:, 3:].max(axis=0) - bb[:, :3].min(axis=0))) or 100.0

    def project(self, xyz) -> np.ndarray:
        """3D points (n, 3) → device pixels (n, 2) of the current view (for a click selection without AIS)."""
        P = np.atleast_2d(np.asarray(xyz, float))
        out = np.zeros((len(P), 2))
        for i, q in enumerate(P):
            out[i] = self.view.Convert(float(q[0]), float(q[1]), float(q[2]))
        return out

    @staticmethod
    def place_on_face(point, direction, normal, L):
        """Arrow tip point outside the material: if the force pushes into the face, the tip is on the face; otherwise the arrow starts from the face."""
        d = np.asarray(direction, float)
        d = d / max(np.linalg.norm(d), 1e-30)
        n = np.asarray(normal, float)
        p = np.asarray(point, float) + 0.03 * L * n          # kousek nad plochou
        return p if d @ n < -1e-6 else p + d * L

    def add_arrows(self, key: str, arrows, color, double: bool = False, persist: bool = False):
        """Arrows (tip point, direction, length) to the set `key`. double = moment (double tip)."""
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone
        from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
        shapes = []
        for p, d, L in arrows:
            d = np.asarray(d, float)
            nd = np.linalg.norm(d)
            if nd < 1e-12 or L <= 0:
                continue
            d = d / nd
            tip = np.asarray(p, float)
            head = 0.28 * L
            r = 0.09 * L
            tail = tip - d * L
            base = tip - d * head
            shapes.append(BRepBuilderAPI_MakeEdge(gp_Pnt(*tail), gp_Pnt(*base)).Edge())
            shapes.append(BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(*base), gp_Dir(*d)), r, 0.0, head).Shape())
            if double:
                b2 = base - d * head
                shapes.append(BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(*b2), gp_Dir(*d)), r, 0.0, head).Shape())
        if not shapes:
            return
        c = _compound(shapes)
        BRepMesh_IncrementalMesh(c, max(0.01, 0.02 * float(arrows[0][2])), False, 0.5, True)
        o = AIS_Shape(c)
        o.SetColor(rgb(*color))
        o.SetWidth(2.5)
        o.SetZLayer(Graphic3d_ZLayerId_Top)
        self.ctx.Display(o, 1, -1, False)
        self.ctx.Deactivate(o)
        (self._persist if persist else self._sets).setdefault(key, []).append(o)
        self._update()

    def add_lines(self, key: str, segments, color, width: float = 2.0, label: str | None = None,
                  persist: bool = False):
        """Segments [(p0, p1)] to the set `key` (bolt axes), optionally with a label at the first one. persist = persistent object
        (frame system – stays across steps, removed by clear_persist).
        """
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCP.gp import gp_Pnt
        shapes = [BRepBuilderAPI_MakeEdge(gp_Pnt(*map(float, a)), gp_Pnt(*map(float, b))).Edge()
                  for a, b in segments if np.linalg.norm(np.subtract(b, a)) > 1e-9]
        if not shapes:
            return
        o = AIS_Shape(_compound(shapes))
        o.SetColor(rgb(*color))
        o.SetWidth(width)
        o.SetZLayer(Graphic3d_ZLayerId_Top)
        self.ctx.Display(o, 1, -1, False)
        self.ctx.Deactivate(o)
        objs = [o]
        if label:
            objs.append(self._text_label(label, gp_Pnt(*map(float, segments[0][0])), [int(c * 0.6) for c in color]))
        (self._persist if persist else self._sets).setdefault(key, []).extend(objs)
        self._update()

    def add_points(self, key: str, pts, color, labels=None, persist: bool = True, size: float = 5.0):
        """Points (spheres) with optional labels – nodes of the frame system."""
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        from OCP.gp import gp_Pnt
        store = self._persist if persist else self._sets
        pts = [np.asarray(q, float) for q in pts]
        if not pts:
            return
        o = AIS_Shape(_compound([BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, q))).Vertex() for q in pts]))
        o.Attributes().SetPointAspect(Prs3d_PointAspect(Aspect_TOM_BALL, rgb(*color), size))
        o.SetZLayer(Graphic3d_ZLayerId_Top)
        self.ctx.Display(o, 0, -1, False)
        self.ctx.Deactivate(o)
        objs = [o]
        for q, t in zip(pts, labels or []):
            if t:
                objs.append(self._text_label(t, gp_Pnt(*map(float, q)), [int(c * 0.6) for c in color]))
        store.setdefault(key, []).extend(objs)
        self._update()

    def clear_persist(self, prefix: str = ""):
        for k in [k for k in self._persist if k.startswith(prefix)]:
            for o in self._persist.pop(k):
                self.ctx.Remove(o, False)
        self._update()

    def show_persist(self, on: bool):
        for objs in self._persist.values():
            for o in objs:
                if on:
                    self.ctx.Display(o, False)
                    self.ctx.Deactivate(o)
                else:
                    self.ctx.Erase(o, False)
        self._update()

    def result_lines(self, segments, values=None, edges=None, colors=None, color=(30, 30, 30), width: float = 4.0,
                     pickable: bool = True):
        """Segments (k, 2, 3) in the results (frames): with values (k,) and band boundaries → band colours (one object per
        band), otherwise one colour. pickable = click on the line = probe (signal probed).
        """
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCP.gp import gp_Pnt
        seg = np.asarray(segments, float).reshape(-1, 2, 3)
        if not len(seg):
            return
        if values is not None and edges is not None and colors is not None:
            k = np.clip(np.searchsorted(np.asarray(edges, float)[1:-1], np.asarray(values, float), side="right"), 0,
                        len(colors) - 1)
            groups = [(colors[b], seg[k == b]) for b in range(len(colors))]
        else:
            groups = [(color, seg)]
        for col, S in groups:
            shapes = [BRepBuilderAPI_MakeEdge(gp_Pnt(*map(float, a)), gp_Pnt(*map(float, b))).Edge()
                      for a, b in S if np.linalg.norm(b - a) > 1e-12]
            if not shapes:
                continue
            o = AIS_Shape(_compound(shapes))
            o.SetColor(rgb(*col))
            o.SetWidth(width)
            self._no_highlight(o)
            self.ctx.Display(o, 1, 0 if pickable else -1, False)
            if not pickable:
                self.ctx.Deactivate(o)
            self._result_objs.append(o)
        self._update()

    def remove_face_set(self, key: str):
        for o in self._sets.pop(key, []):
            self.ctx.Remove(o, False)

    def clear_sets(self):
        for k in list(self._sets):
            self.remove_face_set(k)
        self._update()

    def show_sets(self, on: bool):
        for objs in self._sets.values():
            for o in objs:
                if on:
                    self.ctx.Display(o, False)
                    self.ctx.Deactivate(o)
                else:
                    self.ctx.Erase(o, False)
        self._update()

    # ------------------------------------------------------------ mesh and results
    def show_mesh(self, nodes: np.ndarray, tri: np.ndarray, color=(214, 219, 226)):
        """Mesh surface (corner triangles) with the element edges."""
        self.clear_results()
        P = nodes[tri[:, :3]]
        face = triangles_face(P)
        if face is None:
            return
        o = AIS_Shape(face)
        o.Attributes().SetAutoTriangulation(False)
        o.SetColor(rgb(*color))
        asp = o.Attributes().ShadingAspect().Aspect()
        asp.SetDrawEdges(True)
        asp.SetEdgeColor(rgb(40, 50, 70))
        asp.SetEdgeWidth(1.0)
        self.ctx.Display(o, 1, -1, False)
        self.ctx.Deactivate(o)
        self._result_objs.append(o)
        self._update()

    def show_contour(self, X: np.ndarray, tri: np.ndarray, values: np.ndarray, edges: np.ndarray,
                     title: str, fmt: str = "%.4g", edges_on: bool = False, reverse: bool = False,
                     show_scale: bool = True, tris=None, outline=None):
        """reverse = inverted colours (small value in red, e.g. safety).

        tris = (P, V) ready linear triangles (trimmed surface + section through the mesh, fea_core.section) instead of the
        surface X/tri; outline = segments (k, 2, 3) of the section outline (dark lines).
        """
        self.clear_results()
        colors = BAND_COLORS[::-1] if reverse else BAND_COLORS
        n = len(edges) - 1
        colors = [colors[round(k * (len(colors) - 1) / max(n - 1, 1))] for k in range(n)]
        bands = triangle_bands(tris[0], tris[1], edges) if tris is not None else contour_bands(X, tri, values, edges)
        for k, P in enumerate(bands):
            face = triangles_face(P)
            if face is None:
                continue
            o = AIS_Shape(face)
            o.Attributes().SetAutoTriangulation(False)
            o.SetColor(rgb(*colors[k]))
            o.SetMaterial(self._matte())
            self._no_highlight(o)
            if edges_on:
                asp = o.Attributes().ShadingAspect().Aspect()
                asp.SetDrawEdges(True)
                asp.SetEdgeColor(rgb(60, 60, 60))
            self.ctx.Display(o, 1, 0, False)
            self._result_objs.append(o)
        if outline is not None and len(outline):
            self._show_segments(outline, (30, 30, 30), 1.5)
        if show_scale:
            self._show_scale(edges, title, fmt, colors)
        self._update()
        return colors

    def _no_highlight(self, o):
        """Contour band without hover or selection highlight: a highlighted band would look like another value.
        It stays detectable (the probe takes the click point) – the highlight draws a completely transparent style.
        """
        q = getattr(self, "_quiet_style", None)
        if q is None:
            from OCP.Prs3d import Prs3d_Drawer
            q = Prs3d_Drawer()
            q.SetLink(self.ctx.HighlightStyle())
            q.SetColor(rgb(80, 200, 255))
            q.SetDisplayMode(1)
            q.SetTransparency(1.0)
            self._quiet_style = q
        o.SetDynamicHilightAttributes(q)
        o.SetHilightAttributes(q)

    def _show_segments(self, seg: np.ndarray, color, width: float):
        """Segments (k, 2, 3) as one results object (section outline)."""
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCP.gp import gp_Pnt
        shapes = [BRepBuilderAPI_MakeEdge(gp_Pnt(*map(float, a)), gp_Pnt(*map(float, b))).Edge()
                  for a, b in seg if np.linalg.norm(b - a) > 1e-12]
        if not shapes:
            return
        o = AIS_Shape(_compound(shapes))
        o.SetColor(rgb(*color))
        o.SetWidth(width)
        self.ctx.Display(o, 1, -1, False)
        self.ctx.Deactivate(o)
        self._result_objs.append(o)

    @staticmethod
    def _matte():
        from OCP.Graphic3d import Graphic3d_MaterialAspect, Graphic3d_NameOfMaterial_UserDefined
        m = Graphic3d_MaterialAspect(Graphic3d_NameOfMaterial_UserDefined)
        m.SetSpecularColor(rgb(25, 25, 25))
        m.SetShininess(0.1)
        return m

    def _show_scale(self, edges, title, fmt, colors):
        from OCP.collections import (Sequence_Quantity_Color as Aspect_SequenceOfColor,
                                     Sequence_TCollection_ExtendedString as TColStd_SequenceOfExtendedString)
        if self._scale is not None:
            self.ctx.Remove(self._scale, False)
        cs = AIS_ColorScale()
        n = len(edges) - 1
        cs.SetRange(float(edges[0]), float(edges[-1]))
        cs.SetNumberOfIntervals(n)
        cols = Aspect_SequenceOfColor()
        for k in range(n):
            cols.Append(rgb(*colors[k]))
        cs.SetColorType(Aspect_TOCSD_USER)
        cs.SetColors(cols)
        labels = TColStd_SequenceOfExtendedString()
        for v in edges:
            labels.Append(TCollection_ExtendedString(decimal(fmt % v), True))
        cs.SetLabelType(Aspect_TOCSD_USER)
        cs.SetLabelAtBorder(True)
        cs.SetLabels(labels)
        cs.SetTitle(TCollection_ExtendedString(title, True))
        s = self.dpr()
        W, H = int(150 * s), int(380 * s)
        cs.SetTextHeight(int(13 * s))
        cs.SetColor(rgb(20, 22, 28))
        cs.SetSize(W, H)
        cs.SetBreadth(int(22 * s))
        # labels with a light background, so that they can be read through the model (own background face
        # drawn over the legend – AIS_ColorScale manages its layer itself)
        try:
            from OCP.Aspect import Aspect_TODT_SUBTITLE
            asp = cs.Attributes().TextAspect().Aspect()
            asp.SetFont("Arial")
            asp.SetDisplayType(Aspect_TODT_SUBTITLE)
            asp.SetColorSubTitle(rgb(250, 251, 253))
        except Exception:  # noqa: BLE001 – appearance only
            pass
        cs.SetTransformPersistence(Graphic3d_TransformPers(
            Graphic3d_TMF_2d, Aspect_TOTP_LEFT_UPPER, Graphic3d_Vec2i(int(14 * s), int(430 * s))))
        cs.SetZLayer(Graphic3d_ZLayerId_TopOSD)
        self.ctx.Display(cs, False)
        self.ctx.Deactivate(cs)
        self._scale = cs

    def mark(self, xyz, text, color=(20, 20, 20), key: str | None = None, below: bool = False):
        """Point marker with a label in the results; key = named marker (a previous one with the same key disappears),
        below = label below the point (does not cover the "max" label at the same place).
        """
        from OCP.gp import gp_Pnt
        if key is not None:
            for o in self._marks.pop(key, []):
                self.ctx.Remove(o, False)
                if o in self._result_objs:
                    self._result_objs.remove(o)
        p = gp_Pnt(*map(float, xyz))
        n = len(self._overlay)
        self.show_point(p, text, color)
        new = self._overlay[n:]
        if below and new and isinstance(new[-1], AIS_TextLabel):
            from OCP.Graphic3d import Graphic3d_VTA_TOP
            new[-1].SetVJustification(Graphic3d_VTA_TOP)
            self.ctx.Redisplay(new[-1], True)
        self._result_objs += new
        del self._overlay[n:]
        if key is not None:
            self._marks[key] = new

    def result_polyline(self, key: str, points, color=(20, 20, 20), width: float = 3.0):
        """Polyline over the results (progress along a line) in an overlay layer – visible also inside the body; the same key
        replaces the previous line, clear_results deletes it.
        """
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCP.gp import gp_Pnt
        for o in self._marks.pop(key, []):
            self.ctx.Remove(o, False)
            if o in self._result_objs:
                self._result_objs.remove(o)
        P = [np.asarray(p, float) for p in points]
        shapes = [BRepBuilderAPI_MakeEdge(gp_Pnt(*map(float, a)), gp_Pnt(*map(float, b))).Edge()
                  for a, b in zip(P, P[1:]) if np.linalg.norm(b - a) > 1e-12]
        if not shapes:
            self._update()
            return
        o = AIS_Shape(_compound(shapes))
        o.SetColor(rgb(*color))
        o.SetWidth(width)
        o.SetZLayer(Graphic3d_ZLayerId_TopOSD)
        self.ctx.Display(o, 1, -1, False)
        self.ctx.Deactivate(o)
        self._result_objs.append(o)
        self._marks[key] = [o]
        self._update()

    def clear_results(self):
        for o in self._result_objs:
            self.ctx.Remove(o, False)
        self._result_objs = []
        self._marks = {}
        if self._scale is not None:
            self.ctx.Remove(self._scale, False)
            self._scale = None
        self._update()
