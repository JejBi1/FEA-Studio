"""OCCT V3d view embedded in a Qt widget."""
import ctypes
from fea_core.i18n import decimal
import math
import sys
import time

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtWidgets import QWidget

from OCP.AIS import (AIS_InteractiveContext, AIS_SelectionScheme_Replace, AIS_SelectionScheme_XOR, AIS_Shape,
                     AIS_TextLabel, AIS_ViewCube, AIS_ViewCubeOwner)
from OCP.Aspect import (Aspect_DisplayConnection, Aspect_GradientFillMethod_Vertical, Aspect_TOL_SOLID, Aspect_TOM_BALL,
                        Aspect_TOTP_LEFT_LOWER, Aspect_TOTP_RIGHT_UPPER)
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
from OCP.Graphic3d import (Graphic3d_Camera, Graphic3d_ClipPlane, Graphic3d_HTA_CENTER, Graphic3d_SequenceOfHClipPlane,
                           Graphic3d_TMF_2d, Graphic3d_TMF_TriedronPers, Graphic3d_TransformPers, Graphic3d_VTA_BOTTOM,
                           Graphic3d_ZLayerId_TopOSD)
from OCP.TopoDS import TopoDS_Compound
from OCP.gp import gp_Ax1, gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec
from OCP.collections import Vec2_int as Graphic3d_Vec2i
from OCP.OpenGl import OpenGl_GraphicDriver
from OCP.Prs3d import Prs3d_LineAspect, Prs3d_PointAspect
from OCP.Quantity import Quantity_Color, Quantity_TOC_sRGB
from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
from OCP.V3d import (V3d_TypeOfOrientation_Zup_AxoRight, V3d_TypeOfOrientation_Zup_Back,
                     V3d_TypeOfOrientation_Zup_Bottom, V3d_TypeOfOrientation_Zup_Front,
                     V3d_TypeOfOrientation_Zup_Left, V3d_TypeOfOrientation_Zup_Right,
                     V3d_TypeOfOrientation_Zup_Top, V3d_Xneg, V3d_Xpos, V3d_Yneg, V3d_Ypos,
                     V3d_Zneg, V3d_Zpos, V3d_Viewer)

from fea_core.i18n import tr

if sys.platform == "win32":
    from OCP.WNT import WNT_Window
else:
    from OCP.Xw import Xw_Window

ORIENT = {
    "iso": V3d_TypeOfOrientation_Zup_AxoRight,
    "front": V3d_TypeOfOrientation_Zup_Front,
    "back": V3d_TypeOfOrientation_Zup_Back,
    "top": V3d_TypeOfOrientation_Zup_Top,
    "bottom": V3d_TypeOfOrientation_Zup_Bottom,
    "left": V3d_TypeOfOrientation_Zup_Left,
    "right": V3d_TypeOfOrientation_Zup_Right,
}


def rgb(r, g, b):
    return Quantity_Color(r / 255.0, g / 255.0, b / 255.0, Quantity_TOC_sRGB)


def nice_step(raw):
    """1 / 2 / 5 × 10^n not smaller than raw."""
    if raw <= 0:
        return 1.0
    e = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * e >= raw - 1e-12:
            return m * e
    return 10.0 * e


def fmt_val(v):
    if abs(v) >= 1000:
        s = f"{v:,.0f}".replace(",", " ")
    else:
        s = f"{v:.6g}"
    return decimal(s)


def _capsule(ptr):
    f = ctypes.pythonapi.PyCapsule_New
    f.restype = ctypes.py_object
    f.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return f(ptr, None, None)


class OccView(QWidget):
    selectionChanged = Signal()           # selection changed by clicking in 3D
    contextMenuAt = Signal(QPoint)        # right click (no drag) at global pos
    zoomSelected = Signal()               # double click on a part

    def __init__(self, parent=None):
        super().__init__(parent)
        for a in (Qt.WA_NativeWindow, Qt.WA_PaintOnScreen, Qt.WA_NoSystemBackground):
            self.setAttribute(a)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(200, 200)

        self._disp = Aspect_DisplayConnection()
        self.driver = OpenGl_GraphicDriver(self._disp, True)
        self.viewer = V3d_Viewer(self.driver)
        self.viewer.SetDefaultLights()
        self.viewer.SetLightOn()
        self.ctx = AIS_InteractiveContext(self.viewer)
        self.view = self.viewer.CreateView()
        self._window = None
        self._press_pos = None
        self._press_btn = None
        self._dragging = False
        self._last = None
        self._sm_last = 0.0               # 3D mouse: time of the last motion, its rotation pivot
        self._sm_pivot = None
        self._overlay = []                # point markers with labels (show_point)
        self._bar_objs = []
        self._bar_key = None
        self._dark = False
        self.clip = None
        self._anim = QTimer(self)
        self._anim.setInterval(15)
        self._anim.timeout.connect(self._animate)
        self._edge_aspect = Prs3d_LineAspect(rgb(25, 25, 30), Aspect_TOL_SOLID, 1.0)
        dr = self.ctx.DefaultDrawer()
        dr.SetAutoTriangulation(False)
        hs = self.ctx.HighlightStyle()
        hs.SetColor(rgb(80, 200, 255))
        hs.SetDisplayMode(1)
        ss = self.ctx.SelectionStyle()
        ss.SetColor(rgb(255, 140, 0))
        ss.SetDisplayMode(1)
        self.ctx.SetPixelTolerance(3) if hasattr(self.ctx, "SetPixelTolerance") else None

    # ------------------------------------------------------------ Qt plumbing
    def paintEngine(self):                # WA_PaintOnScreen: OCCT draws into the native window, Qt must not
        return None

    def dpr(self):
        return self.devicePixelRatioF()

    def _px(self, pos):
        r = self.dpr()
        return int(pos.x() * r), int(pos.y() * r)

    def showEvent(self, e):
        super().showEvent(e)
        if self._window is None:
            if sys.platform == "win32":
                self._window = WNT_Window(_capsule(int(self.winId())))
            else:                                    # X11 (Wayland sessions run Qt via XWayland, see main)
                self._window = Xw_Window(self._disp, int(self.winId()))
            self.view.SetWindow(self._window)
            if not self._window.IsMapped():
                self._window.Map()
            self._setup_view()
            QTimer.singleShot(0, self._force_resize)
            QTimer.singleShot(300, self._force_resize)

    def _force_resize(self):
        self._size = None
        self.redraw()

    def _setup_view(self):
        v = self.view
        from . import theme
        self.set_dark_background(theme.is_dark(), redraw=False)
        rp = v.ChangeRenderingParams()
        rp.NbMsaaSamples = 4
        rp.IsAntialiasingEnabled = True
        v.SetProj(ORIENT["iso"])
        cube = AIS_ViewCube()
        s = self.dpr()
        cube.SetSize(55 * s)
        cube.SetFontHeight(11 * s)
        cube.SetBoxColor(rgb(225, 228, 235))
        cube.SetTransformPersistence(Graphic3d_TransformPers(
            Graphic3d_TMF_TriedronPers, Aspect_TOTP_RIGHT_UPPER, Graphic3d_Vec2i(int(90 * s), int(90 * s))))
        for side, txt in ((V3d_Yneg, "Front"), (V3d_Ypos, "Back"), (V3d_Zpos, "Top"),
                          (V3d_Zneg, "Bottom"), (V3d_Xneg, "Left"), (V3d_Xpos, "Right")):
            cube.SetBoxSideLabel(side, TCollection_AsciiString(tr(txt)))
        cube.SetDuration(0.35)
        cube.SetFitSelected(False)
        noclip = Graphic3d_SequenceOfHClipPlane()
        noclip.SetOverrideGlobal(True)
        cube.SetClipPlanes(noclip)
        self.cube = cube
        self.ctx.Display(cube, False)
        v.MustBeResized()

    def set_dark_background(self, dark, redraw=True):
        from . import theme                        # background gradient according to the theme of the program (fea_gui.theme)
        c = theme.DARK if dark else theme.LIGHT
        top, bottom = c["view_top"], c["view_bottom"]
        self.view.SetBgGradientColors(rgb(*top), rgb(*bottom), Aspect_GradientFillMethod_Vertical, False)
        self._dark = dark
        self._bar_key = None
        if redraw:
            self.redraw()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._window is not None and not getattr(self, "_resize_pending", False):
            # one resize + redraw per event-loop pass: at the start of the window (layout, docks, toolbars) several resizes came
            # in a row, each with MustBeResized + Redraw (~0.2 s); redraw() itself resizes when the size changed
            self._resize_pending = True
            QTimer.singleShot(0, self._resize_redraw)

    def _resize_redraw(self):
        self._resize_pending = False
        self.redraw()

    def paintEvent(self, e):
        self.redraw()

    def redraw(self):
        if self._window is None:
            return
        cur = (int(self.width() * self.dpr()), int(self.height() * self.dpr()))
        if getattr(self, "_size", None) != cur:
            self._size = cur
            self._window.DoResize()
            self.view.MustBeResized()
        self._update_scale_bar()
        self.view.Redraw()

    # ------------------------------------------------------------ scale bar ("ruler")
    def _remove_bar(self):
        for o in self._bar_objs:
            self.ctx.Remove(o, False)
        self._bar_objs = []

    def _update_scale_bar(self):
        cam = self.view.Camera()
        persp = cam.ProjectionType() != Graphic3d_Camera.Projection_e.Projection_Orthographic
        # ortho Scale() is the view height for wide windows but the view WIDTH for tall ones
        h = min(self._size) if getattr(self, "_size", None) else 0
        if persp or h <= 0:
            if self._bar_objs:
                self._remove_bar()
                self._bar_key = None
            return
        upp = cam.Scale() / h                      # model units per pixel (ortho)
        key = (round(math.log(upp), 6), self._dark, h)
        if key == self._bar_key:
            return
        self._bar_key = key
        self._remove_bar()
        s = self.dpr()
        length = nice_step(110 * s * upp)
        L = length / upp
        tk = 7 * s
        b = BRep_Builder()
        comp = TopoDS_Compound()
        b.MakeCompound(comp)
        segs = [((0, 0), (L, 0)), ((0, 0), (0, tk)), ((L, 0), (L, tk)), ((L / 2, 0), (L / 2, tk * 0.6))]
        for i in range(1, 10):
            if i != 5:
                segs.append(((L * i / 10, 0), (L * i / 10, tk * 0.35)))
        for (x0, y0), (x1, y1) in segs:
            b.Add(comp, BRepBuilderAPI_MakeEdge(gp_Pnt(x0, y0, 0), gp_Pnt(x1, y1, 0)).Edge())
        col = rgb(235, 235, 235) if self._dark else rgb(30, 32, 38)
        pers = Graphic3d_TransformPers(Graphic3d_TMF_2d, Aspect_TOTP_LEFT_LOWER,
                                       Graphic3d_Vec2i(int(24 * s), int(22 * s)))
        bar = AIS_Shape(comp)
        bar.SetColor(col)
        bar.SetWidth(1.6)
        lbl = AIS_TextLabel()
        lbl.SetText(TCollection_ExtendedString(f"{fmt_val(length)} mm", True))
        lbl.SetPosition(gp_Pnt(L / 2, tk + 3 * s, 0))
        lbl.SetHJustification(Graphic3d_HTA_CENTER)
        lbl.SetVJustification(Graphic3d_VTA_BOTTOM)
        lbl.SetColor(col)
        lbl.SetHeight(13 * s)
        noclip = Graphic3d_SequenceOfHClipPlane()
        noclip.SetOverrideGlobal(True)
        for o in (bar, lbl):
            o.SetTransformPersistence(pers)
            o.SetZLayer(Graphic3d_ZLayerId_TopOSD)
            o.SetClipPlanes(noclip)
            self.ctx.Display(o, False)
            self.ctx.Deactivate(o)
            self._bar_objs.append(o)

    def apply_edge_style(self, obj, on):
        dr = obj.Attributes()
        dr.SetFaceBoundaryDraw(on)
        if on:
            dr.SetFaceBoundaryAspect(self._edge_aspect)

    # ------------------------------------------------------------ camera
    def fit_all(self):
        self.view.FitAll(0.04, False)
        self.view.ZFitAll()
        self.redraw()

    def fit_box(self, box):
        if box is None or box.IsVoid():
            return
        self.view.FitAll(box, 0.15, False)
        self.view.ZFitAll()
        self.redraw()

    def set_orientation(self, key):
        self.view.SetProj(ORIENT[key])
        self.fit_all()

    def _animate(self):
        if not self.cube.UpdateAnimation(True):
            self._anim.stop()
        self.redraw()                              # cube animation re-fits -> scale bar follows

    def _zoom_at(self, x, y, factor):
        cam = self.view.Camera()
        w, h = self.width() * self.dpr(), self.height() * self.dpr()
        c = cam.Center()
        zc = cam.Project(c).Z()
        p = cam.UnProject(gp_Pnt(2.0 * x / w - 1.0, 1.0 - 2.0 * y / h, zc))
        e = cam.Eye()
        k = 1.0 / factor
        nc = gp_Pnt(p.X() + (c.X() - p.X()) * k, p.Y() + (c.Y() - p.Y()) * k, p.Z() + (c.Z() - p.Z()) * k)
        if cam.ProjectionType() == Graphic3d_Camera.Projection_e.Projection_Orthographic:
            d = gp_Vec(c, nc)
            cam.SetScale(cam.Scale() * k)
            cam.SetCenter(nc)
            cam.SetEye(e.Translated(d))
        else:                                      # perspective: move the eye towards the cursor point
            cam.SetCenter(nc)
            cam.SetEye(gp_Pnt(p.X() + (e.X() - p.X()) * k, p.Y() + (e.Y() - p.Y()) * k,
                              p.Z() + (e.Z() - p.Z()) * k))
        self.view.Invalidate()
        self.redraw()

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, e):
        self.setFocus()
        self._press_pos = e.position().toPoint()
        self._press_btn = e.button()
        self._dragging = False
        x, y = self._px(self._press_pos)
        self._last = (x, y)
        ctrl = bool(e.modifiers() & Qt.ControlModifier)
        # Onshape: RMB drag = rotate about the point under the cursor, MMB / Ctrl+RMB = pan, LMB = select
        if e.button() == Qt.RightButton and not ctrl:
            self._mode = "rotate"
            self._pivot = self._pick_pivot(x, y)
        elif e.button() in (Qt.MiddleButton, Qt.RightButton):
            self._mode = "pan"
        else:
            self._mode = None

    def _pick_pivot(self, x, y):
        """Model point under the cursor, else the centre of the view."""
        self.ctx.MoveTo(x, y, self.view, False)
        p = None
        if self.ctx.HasDetected() and not isinstance(self.ctx.DetectedOwner(), AIS_ViewCubeOwner):
            p = self._picked_point()
        self.ctx.ClearDetected(False) if hasattr(self.ctx, "ClearDetected") else None
        return p if p is not None else self.view.Camera().Center()

    # ------------------------------------------------------------ 3D mouse (3Dconnexion)
    SM_ROT = math.radians(150.0)           # rad/s at full deflection, speed 1
    SM_PAN = 1100.0                        # px/s
    SM_ZOOM = 2.2                          # e-folds/s

    def space_motion(self, v, dt):
        """Object mode like the 3Dconnexion default: the model follows the cap. v = (tx, ty, tz, rx, ry, rz) in the
        screen frame (X right, Y up, Z towards the user), rotations about the pivot picked when a motion starts."""
        if self._dragging or self._window is None:
            return
        now = time.monotonic()
        if now - self._sm_last > 0.3 or self._sm_pivot is None:
            s = self.dpr()
            self._sm_pivot = self._pick_pivot(int(self.width() * s / 2), int(self.height() * s / 2))
        self._sm_last = now
        tx, ty, tz, rx, ry, rz = v
        cam = self.view.Camera()
        d, up = cam.Direction(), cam.Up()
        right = gp_Vec(d.Crossed(up))
        if rx or ry or rz:                         # model rotates by w -> camera by -w about the pivot
            w = right.Multiplied(rx).Added(gp_Vec(up).Multiplied(ry)).Added(gp_Vec(d).Multiplied(-rz))
            mag = w.Magnitude()
            if mag > 1e-12:
                t = gp_Trsf()
                t.SetRotation(gp_Ax1(self._sm_pivot, gp_Dir(w)), -mag * self.SM_ROT * dt)
                cam.Transform(t)
                d, up = cam.Direction(), cam.Up()
                right = gp_Vec(d.Crossed(up))
        if tx or ty:                               # model moves right / up -> camera the other way
            k = self.SM_PAN * self.dpr() * dt * self.world_per_pixel()
            t = gp_Trsf()
            t.SetTranslation(right.Multiplied(-tx * k).Added(gp_Vec(up).Multiplied(-ty * k)))
            cam.Transform(t)
        if tz:                                     # pulled towards the user = closer
            s = self.dpr()
            self._zoom_at(self.width() * s / 2, self.height() * s / 2, math.exp(tz * self.SM_ZOOM * dt))
            return
        self.view.Invalidate()
        self.redraw()

    def roll(self, ang):
        """Rotate the view about the viewing direction (3D mouse Roll keys)."""
        cam = self.view.Camera()
        t = gp_Trsf()
        t.SetRotation(gp_Ax1(cam.Center(), cam.Direction()), ang)
        cam.Transform(t)
        self.view.Invalidate()
        self.redraw()

    def orbit(self, ang_h, ang_v, pivot=None):
        """Rotate the camera about pivot: ang_h around the screen vertical, ang_v around the screen horizontal."""
        cam = self.view.Camera()
        pivot = pivot or cam.Center()
        d = cam.Direction()
        up = cam.Up()
        side = gp_Dir(d.Crossed(up))
        t1 = gp_Trsf()
        t1.SetRotation(gp_Ax1(pivot, up), ang_h)
        t2 = gp_Trsf()
        t2.SetRotation(gp_Ax1(pivot, side), ang_v)
        cam.Transform(t2.Multiplied(t1))
        self.view.Invalidate()
        self.redraw()

    def mouseMoveEvent(self, e):
        pos = e.position().toPoint()
        x, y = self._px(pos)
        if e.buttons() == Qt.NoButton:
            self.ctx.MoveTo(x, y, self.view, True)
            return
        if self._press_pos is None:
            return
        if not self._dragging:
            if (pos - self._press_pos).manhattanLength() < 4:
                return
            self._dragging = True
        lx, ly = self._last
        if self._mode == "rotate":
            k = 0.006 / self.dpr()                       # rad per physical pixel (~180° per 520 px)
            self.orbit(-(x - lx) * k, -(y - ly) * k, self._pivot)
        elif self._mode == "pan":
            self.view.Pan(x - lx, ly - y, 1.0, True)
        self._last = (x, y)

    def keyPressEvent(self, e):
        # Onshape: arrows rotate 15°, Shift+arrows 90°
        mods = e.modifiers()
        shift = bool(mods & Qt.ShiftModifier)
        other = bool(mods & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier))
        step = math.radians(90 if shift else 15)
        keys = {Qt.Key_Left: (step, 0), Qt.Key_Right: (-step, 0), Qt.Key_Up: (0, step), Qt.Key_Down: (0, -step)}
        if e.key() in keys and not other:
            self.orbit(*keys[e.key()])
        else:
            super().keyPressEvent(e)

    def mouseReleaseEvent(self, e):
        pos = e.position().toPoint()
        x, y = self._px(pos)
        was_drag = self._dragging
        self._dragging = False
        self._press_pos = None
        if was_drag:
            return
        if e.button() == Qt.LeftButton:
            self.ctx.MoveTo(x, y, self.view, False)
            owner = self.ctx.DetectedOwner() if self.ctx.HasDetected() else None
            if owner is not None and isinstance(owner, AIS_ViewCubeOwner):
                self.cube.HandleClick(owner)
                self._anim.start()
                return
            if owner is None:
                self.ctx.ClearSelected(False)
            else:
                scheme = AIS_SelectionScheme_XOR if e.modifiers() & Qt.ControlModifier else AIS_SelectionScheme_Replace
                self.ctx.SelectDetected(scheme)
            self.ctx.UpdateCurrentViewer()
            self.selectionChanged.emit()
        elif e.button() == Qt.RightButton:
            self.ctx.MoveTo(x, y, self.view, False)
            self.contextMenuAt.emit(self.mapToGlobal(pos))

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.selectionChanged.emit()
            self.zoomSelected.emit()           # the window knows the located shapes (box incl. assembly placement)

    def wheelEvent(self, e):
        x, y = self._px(e.position().toPoint())
        d = e.angleDelta().y()
        if d == 0 or time.monotonic() - self._sm_last < 0.4:   # 3DxWare may emulate the wheel while moving
            return
        self._zoom_at(x, y, 1.2 ** (d / 120.0))

    def leaveEvent(self, e):
        if hasattr(self.ctx, "ClearDetected"):
            self.ctx.ClearDetected(True)

    # ------------------------------------------------------------ selection helpers
    def selected_objects(self):
        out = []
        self.ctx.InitSelected()
        while self.ctx.MoreSelected():
            out.append(self.ctx.SelectedInteractive())
            self.ctx.NextSelected()
        return out

    def _picked_point(self):
        sel = self.ctx.MainSelector()
        if sel.NbPicked() < 1:
            return None
        return sel.PickedPoint(1)

    def world_per_pixel(self):
        try:
            return self.view.Convert(int(100 * self.dpr())) / (100 * self.dpr())
        except Exception:  # noqa: BLE001
            return 0.0

    # ------------------------------------------------------------ section plane
    def set_section_plane(self, origin, normal):
        """Section of the model by a general plane (OCCT clip plane with filling); keeps the side the normal points to."""
        pln = gp_Pln(gp_Pnt(*map(float, origin)), gp_Dir(*map(float, normal)))
        if self.clip is None:
            self.clip = Graphic3d_ClipPlane(pln)
            self.clip.SetCapping(True)
            self.clip.SetUseObjectMaterial(True)
            self.clip.SetCappingHatchOff()
            self.view.AddClipPlane(self.clip)
        else:
            self.clip.SetEquation(pln)
        self.redraw()

    def clear_section(self):
        if self.clip is not None:
            self.view.RemoveClipPlane(self.clip)
            self.clip = None
        self.redraw()

    # ------------------------------------------------------------ point markers
    def show_point(self, p, text, color=(0, 120, 255)):
        v = AIS_Shape(BRepBuilderAPI_MakeVertex(p).Vertex())
        v.Attributes().SetPointAspect(Prs3d_PointAspect(Aspect_TOM_BALL, rgb(*color), 4.0))
        v.SetZLayer(Graphic3d_ZLayerId_TopOSD)
        self.ctx.Display(v, 0, -1, False)
        self.ctx.Deactivate(v)
        lbl = AIS_TextLabel()
        lbl.SetText(TCollection_ExtendedString(text, True))
        lbl.SetPosition(p)
        lbl.SetColor(rgb(*color))
        lbl.SetHeight(15 * self.dpr())
        lbl.SetZLayer(Graphic3d_ZLayerId_TopOSD)
        self.ctx.Display(lbl, False)
        self.ctx.Deactivate(lbl)
        self._overlay += [v, lbl]
        self.ctx.UpdateCurrentViewer()

    def screenshot(self, path):
        return self.view.Dump(path)
