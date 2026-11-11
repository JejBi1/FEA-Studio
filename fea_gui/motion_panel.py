"""Motion panel (step after Constraints, F8 – spec 3, 6.3): kinematics and dynamics of mechanisms of rigid bodies.

Tabs: Bodies (parts by click in 3D, mass properties, frame, flexible body) | Joints (proposal from the geometry / Fusion metadata
with confirmation, by hand – type, bodies, axis from a face, limits, restitution, friction) | Drives and forces (laws incl.
the S-curve with a preview, force elements, external loads, weight) | Calculation (kind, time, outputs, collisions, solver, mobility
check, start) | Results (summary, animation – slider and play, series graphs, "Find maximum", transfer to the FEM).

The GUI edits Study.mechanism; the calculation is done by fea_core.motion (a separate Exudyn process). The motion result
(MotionResult) is held by the main window (MainWindow.motion_result) and saved in the project (results_motion.npz). A change
of the mechanism discards the result (the mechanism signature without the FEM transfer settings).
Project units: mm, N, N·mm, s, rad; mass in the GUI in kg (converted to t), inertia in kg·m² (converted to t·mm²).
"""
from __future__ import annotations

from html import escape
import json

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSlider, QSpinBox, QStackedWidget, QTabWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from fea_core.motion.model import (JOINT_KINDS, JOINT_LABELS, Body, Driver, ForceElement, JointFriction, MotionJoint,
                                   MotionLoad)

from .safe_browser import SafeTextBrowser
from .contact_panel import EditTable
from .widgets import LazyPlot, TaskProgress, dspin, fit_tabs, fmt, panel_header, scroll_body, scroll_to, table, Section, help_tip

APP = "FEA Studio"
GROUND = "(ground)"
QUANTITY_LABELS = {"angle": "angle [rad]", "disp": "displacement [mm]", "velocity": "velocity [mm/s]",
                   "angular_velocity": "angular velocity [rad/s]", "torque": "torque [N·mm]", "force": "force [N]"}
LAW_LABELS = {"constant": "constant", "table": "table in time", "sine": "sine", "poly": "polynomial",
              "scurve": "S-curve (limited jerk)"}
FORCE_LABELS = {"spring": "spring", "damper": "damper", "spring_damper": "spring + damper",
                "torsion": "torsion spring (in a joint)", "gas_spring": "gas spring (table)",
                "bushing": "bushing (6 stiffnesses)"}
FRICTION_LABELS = {"coulomb": "Coulomb (μ·|R|·r)", "viscous": "viscous (c·v)", "stribeck": "Stribeck",
                   "bearing": "rolling bearing (M = μ·F·d/2)", "thread": "power screw"}
FIT_LABELS = {"clearance": "with clearance (line contact)", "worn": "worn-in (p ~ cos θ)", "fitted": "fitted (uniform pressure)"}
RUN_KINDS = {"kinematic": "kinematic (prescribed drivers, forces incl. inertia)",
             "static": "quasi-static (equilibrium in positions, no inertia)",
             "dynamic": "dynamic (direct time integration)"}
JOINT_TIPS = {"fixed": "Rigid: bodies connected without motion (6 constraints)",
              "revolute": "Revolute: rotation about an axis (pin, hinge) – 1 degree of freedom",
              "prismatic": "Slider: translation along an axis (guide) – 1 degree of freedom",
              "cylindrical": "Cylindrical: rotation about the axis and translation along it – 2 degrees of freedom",
              "pin_slot": "Pin slot: rotation about the axis and translation along the slot (axis 2) – 2 degrees of freedom",
              "planar": "Planar: translation in the plane normal to the axis and rotation about the axis – 3 degrees of freedom",
              "spherical": "Ball: rotation about a point – 3 degrees of freedom",
              "universal": "Universal: two rotations about axes perpendicular to the joint axis (rotation about the axis is transmitted) – 2 degrees of freedom",
              "screw": "Screw: rotation coupled with translation by the lead – 1 degree of freedom",
              "gear": "Gear: ω_b = ratio · ω_a between two revolute joints",
              "rack_pinion": "Rack and pinion: translation = ratio [mm/rad] · rotation",
              "distance": "Distance (link): constant distance of points A and B – 1 constraint"}
JOINT_COLORS = {"revolute": (220, 60, 40), "prismatic": (40, 120, 220), "cylindrical": (150, 60, 200),
                "spherical": (230, 140, 20), "fixed": (90, 90, 90)}


def _f(v):
    return None if v is None or abs(v) < 1e-300 else float(v)


def _vec(vals, lo=-1e7, hi=1e7, dec=4, tip="", names="xyz", detailed=""):
    """Three coordinate fields in one row: (list of fields, layout)."""
    h = QHBoxLayout()
    spins = []
    for v, c in zip(vals, names):
        s = dspin(float(v), lo, hi, dec, 1.0)
        short = f"{tip} – component {c}" if tip else f"Component {c}"
        det = f"{detailed} – component {c}" if detailed else ""
        help_tip(s, short, det)
        spins.append(s)
        h.addWidget(s)
    return spins, h


def _vals(spins):
    return [float(s.value()) for s in spins]


def _set(spins, vals):
    for s, v in zip(spins, vals or [0, 0, 0]):
        s.setValue(float(v))


def _btn(text, tip, slot, lay=None, detailed=""):
    b = QPushButton(text)
    if not detailed and (len(tip) > 80 or "\n" in tip):
        from .guide import short_of
        help_tip(b, short_of(tip, 80), tip)
    else:
        help_tip(b, tip, detailed)
    b.clicked.connect(slot)
    if lay is not None:
        lay.addWidget(b)
    return b


def mech_signature(study) -> str:
    """Signature of the inputs of the mechanism calculation (without the FEM transfer settings) – a change discards the motion result."""
    d = study.mechanism.to_dict()
    d.get("run", {}).pop("to_fea", None)
    return json.dumps({"m": d, "g": str(study.geometry or "")}, sort_keys=True, default=str)


class SuggestionDialog(QDialog):
    """Confirmation of the proposed joints (from the geometry or the Fusion metadata): tick which ones to accept; a click on a row
    highlights the faces of the proposal in 3D.
    """

    def __init__(self, panel, suggestions):
        super().__init__(panel)
        self.panel, self.sug = panel, list(suggestions)
        self.setWindowTitle("Joint suggestions")
        v = QVBoxLayout(self)
        lbl = QLabel("The program suggests joints – check the type and bodies, tick the ones you want to accept. Clicking a row highlights the faces in 3D.")
        lbl.setWordWrap(True)
        v.addWidget(lbl)
        self.table = table(["Accept", "Joint", "Type", "Body A", "Body B", "Confidence", "Reason"], 200,
                           stretch_last=True)
        help_tip(self.table, "Proposed joints: confidence and reason; ticked ones are added after confirmation",
                 "Suggested joints: confidence 0–1 and reason. The ticked ones are added after confirmation (with face selections for the transfer to FEA).")
        for s in self.sug:
            r = self.table.rowCount()
            self.table.insertRow(r)
            it = QTableWidgetItem("")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if s.confidence >= 0.6 else Qt.Unchecked)
            self.table.setItem(r, 0, it)
            j = s.joint
            for c, t in enumerate((j.name, JOINT_LABELS.get(j.kind, j.kind), j.body_a or GROUND, j.body_b or GROUND,
                                   fmt(s.confidence, 2), s.reason), 1):
                self.table.setItem(r, c, QTableWidgetItem(str(t)))
        self.table.currentCellChanged.connect(lambda r, *_: self.show_row(r))
        v.addWidget(self.table)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Accept ticked")
        bb.button(QDialogButtonBox.Ok).setToolTip("Add the ticked joints to the mechanism")
        bb.button(QDialogButtonBox.Cancel).setText("Cancel")
        bb.button(QDialogButtonBox.Cancel).setToolTip("Close without adding joints")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.resize(820, 420)

    def show_row(self, r):
        if not (0 <= r < len(self.sug)):
            return
        win = self.panel.win
        names = [p.name for p in win.parts]
        s = self.sug[r]
        v = win.view
        v.clear_sets()
        for key, faces, col in (("sug:a", s.faces_a, (40, 160, 230)), ("sug:b", s.faces_b, (230, 120, 30))):
            fl = [(names.index(pn), k) for pn, k in faces if pn in names]
            if fl:
                v.set_face_set(key, fl, col, transparency=0.2)
        j = s.joint
        L = 0.3 * v.model_size()
        o, a = np.asarray(j.origin, float), np.asarray(j.axis, float)
        a = a / max(np.linalg.norm(a), 1e-30)
        v.add_lines("sug:axis", [(o - L * a, o + L * a)], (220, 30, 30), 3.0, label=j.name)
        v.show_sets(True)

    def accepted_suggestions(self):
        return [s for r, s in enumerate(self.sug) if self.table.item(r, 0).checkState() == Qt.Checked]


class ToFeaDialog(QDialog):
    """Transfer of the motion result to the FEM: instant(s) / maximum of a quantity / the whole course, body, bearing joints."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        res = panel.win.motion_result
        st = panel.study
        cfg = st.mechanism.run.to_fea
        self.setWindowTitle("Transfer motion to FEA")
        f = QFormLayout(self)
        intro = QLabel("Creates load cases of the body (d'Alembert): reactions of joints and force elements as remote loads (RBE3) on the joint faces, inertia and gravity as the “Inertia” load; one joint is the support (remote support) – its reaction in FEA must come out as in MBD (check).")
        intro.setWordWrap(True)
        f.addRow(intro)
        self.mode = QComboBox()
        for k, t in (("instants", "instant(s) – times"), ("max", "instant of the maximum of a quantity"),
                     ("history", "whole history → unit cases + fatigue channels")):
            self.mode.addItem(t, k)
        self.mode.setCurrentIndex(max(0, self.mode.findData(cfg.mode)))
        help_tip(self.mode, "Instant, maximum or whole history – which steps result from the motion",
                 "Instant: load step at the given times; maximum: the instant of the largest value of the chosen quantity;<br>whole history: unit steps (6 components per joint + inertia) and the channels of the load curve for fatigue (superposition)")
        f.addRow("Type", self.mode)
        self.times = QLineEdit("; ".join(fmt(t, 6) for t in (cfg.times or [panel.current_time()])))
        help_tip(self.times, "Instants [s] separated by semicolons (default = time of the animation slider)",
                 "Times of the instants [s] separated by semicolons (e.g. 0.01; 0.025); default = time of the animation slider")
        f.addRow("Times [s]", self.times)
        self.qty = QComboBox()
        self.qty.setEditable(True)
        for q in panel.series_names():
            self.qty.addItem(q)
        if cfg.quantity:
            self.qty.setEditText(cfg.quantity)
        self.qty.setToolTip("Quantity whose maximum (magnitude) determines the instant – e.g. reakce:Kloub 1")
        f.addRow("Quantity of the maximum", self.qty)
        self.body = QComboBox()
        for b in res.meta.get("bodies", []):
            if not b.get("ground"):
                self.body.addItem(b["name"])
        self.body.setToolTip("Body for which the load cases are created (each body separately)")
        self.body.currentIndexChanged.connect(lambda _: self.fill_joints())
        f.addRow("Body", self.body)
        self.sup = QListWidget()
        self.sup.setMaximumHeight(100)
        help_tip(self.sup, "Support joints: the first ticked = remote support, the others = redundant",
                 "Support joints: the first ticked = remote support (6 components; its reaction is not loaded – inertia relief check);<br>other (redundant) joints are also supports – the FE model distributes the reactions among them. If nothing is ticked, the first joint of the body is used.")
        f.addRow("Support joints", self.sup)
        self.kind = QComboBox()
        self.kind.addItem("remote load RBE3 (recommended)", "remote_distributing")
        self.kind.addItem("bearing load on pins (instants only)", "bearing")
        self.kind.setCurrentIndex(max(0, self.kind.findData(cfg.load_kind)))
        help_tip(self.kind, "Remote load (linear) or bearing load (instants only)",
                 "Joint load: remote load (linear – also for the whole history), or bearing load (cosine pressure on the facing half of the hole, instants only – not superposable)")
        f.addRow("Joint loads", self.kind)
        self.fill_joints(cfg.support_joints)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Create load cases")
        bb.button(QDialogButtonBox.Ok).setToolTip("Transfer and add the load cases (and support constraints) to the study")
        bb.button(QDialogButtonBox.Cancel).setText("Cancel")
        bb.button(QDialogButtonBox.Cancel).setToolTip("Close without transfer")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self.resize(560, 460)

    def fill_joints(self, checked=None):
        res = self.panel.win.motion_result
        names = res.body_names
        b = self.body.currentText()
        bi = names.index(b) if b in names else -1
        prev = checked if checked is not None else [self.sup.item(i).text() for i in range(self.sup.count())
                                                     if self.sup.item(i).checkState() == Qt.Checked]
        self.sup.clear()
        for J in res.meta.get("joints", []):
            if bi >= 0 and bi in (J.get("a"), J.get("b")) and J["kind"] not in ("gear", "rack_pinion"):
                it = QListWidgetItem(J["name"])
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if J["name"] in (prev or []) else Qt.Unchecked)
                self.sup.addItem(it)

    def options(self) -> dict:
        times = []
        for x in self.times.text().replace(",", ".").replace(";", " ").split():
            times.append(float(x))
        return {"mode": self.mode.currentData(), "times": times, "quantity": self.qty.currentText().strip() or None,
                "body": self.body.currentText() or None,
                "support_joints": [self.sup.item(i).text() for i in range(self.sup.count())
                                   if self.sup.item(i).checkState() == Qt.Checked] or None,
                "load_kind": self.kind.currentData()}


class MotionPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self._filling = False
        self._thread = None
        self._jsel = {"a": None, "b": None}           # faces of the joint being edited (selections)
        self._fsel = {"a": None, "b": None}           # faces of the force element being edited
        self._lsel = None                             # face of the load being edited
        self._props = {}                              # mass properties of the bodies (body_props)
        self._exports = []                            # transfers to the FEM (check of inertia relief after the calculation)
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)
        self._scroll, lay = scroll_body(self)
        lay.addWidget(panel_header("Motion", "Kinematics and dynamics of rigid-body mechanisms – joints, drivers, animation, reactions, transfer to FEA (optional)"))
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        self.info.setToolTip("Mechanism state: bodies, joints, drivers and the result of the model check")
        lay.addWidget(self.info)
        self.tabs = QTabWidget()
        self.tabs.setToolTip("Procedure from the left: bodies → joints → drivers and forces → analysis → results")
        lay.addWidget(self.tabs)
        for title, build, tip in (("Bodies", self._tab_bodies, "Bodies = parts or groups of parts, ground"),
                                  ("Joints", self._tab_joints, "Joints between bodies (from geometry or manually)"),
                                  ("Drivers and forces", self._tab_drivers, "Joint drivers, force elements, external loads, gravity"),
                                  ("Calculation", self._tab_run, "Analysis type, time, outputs, collisions, DOF check, start"),
                                  ("Results", self._tab_results, "Animation, charts, maximum, transfer to FEA")):
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(4, 6, 4, 4)
            build(v)
            v.addStretch(1)
            self.tabs.addTab(w, title)
            self.tabs.setTabToolTip(self.tabs.count() - 1, tip)
        fit_tabs(self.tabs)
        lay.addStretch(1)

    # ================================================================== stavba karet
    def _tab_bodies(self, lay):
        self.b_table = table(["Body", "Parts", "Ground", "Mass [kg]", "Centre of gravity [mm]", "Source"],
                             placeholder="No bodies yet – use “Body for each part” or enter one below",
                             rows=(3, 8))
        help_tip(self.b_table, "Mechanism bodies: parts, frame, mass and centre of mass – click = edit below",
                 "Bodies of the mechanism: parts (rigidly connected), ground = fixed body, mass and centre of gravity from CAD (× ρ) or entered. Click = edit below, the parts are highlighted in 3D.")
        self.b_table.currentCellChanged.connect(lambda r, *_: self.on_body_row(r))
        lay.addWidget(self.b_table)
        h = QHBoxLayout()
        _btn("Body for each part", "Create a body from each part not yet in any body", self.bodies_from_parts, h,
             detailed="Create a body from each part that is not in any body yet (the first = ground if the ground is missing)")
        self.btn_props = _btn("Compute masses", "Body mass, center of gravity and inertia from CAD", self.compute_props, h,
                              detailed="Mass, centre of gravity and inertia of the bodies from CAD (volume × density; printed parts indicatively by shell and infill)")
        lay.addLayout(h)
        g = Section("Body", key="motion.body", collapsed=False, tip="Edit the selected / enter a new body")
        f = QFormLayout(g.body)
        self.b_name = QLineEdit()
        self.b_name.setToolTip("Body name (joints and results refer to it)")
        f.addRow("Name", self.b_name)
        self.b_parts = QListWidget()
        self.b_parts.setMaximumHeight(110)
        self.b_parts.setToolTip("Parts of the body (ticked) – rigidly connected, they move together")
        f.addRow("Parts", self.b_parts)
        hb = QHBoxLayout()
        _btn("Parts from 3D selection", "Check parts whose faces are selected in 3D", self.parts_from_selection, hb,
             detailed="Tick the parts whose faces are selected in 3D (filter Body B selects a whole part)")
        f.addRow(hb)
        self.b_ground = QCheckBox("ground (fixed body)")
        self.b_ground.setToolTip("Ground = fixed surroundings (stand, plate); joints to the ground hold the mechanism")
        f.addRow(self.b_ground)
        self.b_override = QCheckBox("enter mass and inertia manually")
        help_tip(self.b_override, "Enter the mass, centre of mass and inertia manually instead of from CAD",
                 "Checked = mass, centre of gravity and inertia from the fields below (body without parts or purchased part); otherwise from CAD")
        f.addRow(self.b_override)
        self.b_mass = dspin(1.0, 0.0, 1e9, 6, 0.1, " kg")
        self.b_mass.setToolTip("Mass of the body [kg]")
        f.addRow("Mass", self.b_mass)
        self.b_com, hc = _vec([0, 0, 0], tip="Centre of gravity [mm]")
        f.addRow("Centre of gravity [mm]", hc)
        self.b_I, hi = _vec([0, 0, 0], 0.0, 1e9, 8, "Moment of inertia about the centre of gravity [kg·m²]", ("Ixx", "Iyy", "Izz"))
        f.addRow("Inertia [kg·m²]", hi)
        self.b_v0, hv = _vec([0, 0, 0], tip="Initial velocity of the centre of gravity [mm/s] (dynamic analysis)")
        f.addRow("v₀ [mm/s]", hv)
        self.b_w0, hw = _vec([0, 0, 0], tip="Initial angular velocity [rad/s] (dynamic analysis)")
        f.addRow("ω₀ [rad/s]", hw)
        self.b_flex = QCheckBox("flexible body (Craig–Bampton, dynamic analysis only)")
        help_tip(self.b_flex, "Flexible body (Craig–Bampton) – dynamic analysis only, isotropic material",
                 "Body as flexible (fixed-interface modes + static interface modes); interface = joint faces of the body. Isotropic materials only, dynamic analysis.")
        f.addRow(self.b_flex)
        hf = QHBoxLayout()
        self.b_modes = QSpinBox()
        self.b_modes.setRange(1, 200)
        self.b_modes.setValue(10)
        self.b_modes.setToolTip("Number of fixed-interface modes in addition to the static interface modes")
        self.b_iface = QComboBox()
        self.b_iface.addItem("rigid interface (RBE2)", "rbe2")
        self.b_iface.addItem("distributing interface (RBE3)", "rbe3")
        self.b_iface.setToolTip("Connection of the joint to the face of the flexible body: rigid or distributing")
        self.b_zeta = dspin(0.0, 0.0, 50.0, 3, 0.5, " %")
        self.b_zeta.setToolTip("Damping ratio of the 1st mode of the flexible body [%] (stiffness proportional)")
        for w in (QLabel("modes"), self.b_modes, self.b_iface, QLabel("ζ"), self.b_zeta):
            hf.addWidget(w)
        f.addRow(hf)
        hb = QHBoxLayout()
        self.btn_body_add = _btn("Add body", "New body from the form", self.add_body, hb)
        self.btn_body_upd = _btn("Save changes", "Overwrite the selected body with the form values", self.update_body, hb)
        self.btn_body_del = _btn("Delete", "Delete selected body", self.delete_body, hb,
                                 detailed="Delete the selected body (joints and drivers referring to it remain – the check reports them)")
        f.addRow(hb)
        lay.addWidget(g)
        self.b_info = QLabel("")
        self.b_info.setWordWrap(True)
        lay.addWidget(self.b_info)

    def _tab_joints(self, lay):
        self.j_table = table(["Joint", "Type", "A", "B", "Origin [mm]", "Axis", "Limits", "Friction", "Source"],
                             placeholder="No joints yet – create them from geometry or enter one below",
                             rows=(3, 8))
        help_tip(self.j_table, "Joints between bodies: type, bodies, origin, axis, limits, friction – click = edit",
                 "Joints between bodies: type, bodies A and B, origin and axis (global), limits (stops), friction. Click = edit below and the axis in 3D.")
        self.j_table.currentCellChanged.connect(lambda r, *_: self.on_joint_row(r))
        lay.addWidget(self.j_table)
        h = QHBoxLayout()
        self.btn_suggest = _btn("Suggest from geometry", "Find coaxial cylindrical faces and suggest revolute / cylindrical joints",
                                self.suggest_geometry, h,
                                detailed="Find coaxial cylindrical faces of parts of different bodies (pin in a hole) and suggest revolute / cylindrical joints – with confirmation")
        _btn("From Fusion metadata…", "Joints from the Fusion export metadata (JSON with the “joints” key) – with confirmation",
             self.suggest_metadata, h)
        lay.addLayout(h)
        g = Section("Joint", key="motion.joint", collapsed=False, tip="Edit the selected / enter a new joint")
        f = QFormLayout(g.body)
        self.j_name = QLineEdit()
        self.j_name.setToolTip("Joint name (reactions in the results are under this name)")
        f.addRow("Name", self.j_name)
        self.j_kind = QComboBox()
        for k in JOINT_KINDS:
            self.j_kind.addItem(JOINT_LABELS[k], k)
            self.j_kind.setItemData(self.j_kind.count() - 1, JOINT_TIPS[k], Qt.ToolTipRole)
        help_tip(self.j_kind, "Joint type (names as in Fusion)",
                 "Joint type (names as in Fusion):<br>" + "<br>".join(JOINT_TIPS[k] for k in JOINT_KINDS))
        self.j_kind.currentIndexChanged.connect(lambda _: self.on_joint_kind())
        f.addRow("Type", self.j_kind)
        hb = QHBoxLayout()
        self.j_a, self.j_b = QComboBox(), QComboBox()
        self.j_a.setToolTip("Body A (the joint reaction is given as the force on body B from body A)")
        self.j_b.setToolTip("Body B (moving side of the joint)")
        hb.addWidget(self.j_a, 1)
        hb.addWidget(QLabel("→"))
        hb.addWidget(self.j_b, 1)
        f.addRow("Bodies A → B", hb)
        self.j_o, ho = _vec([0, 0, 0], tip="Joint origin [mm]")
        f.addRow("Origin [mm]", ho)
        self.j_ax, ha = _vec([0, 0, 1], tip="Joint axis (direction of rotation / translation)")
        f.addRow("Axis", ha)
        hz = QHBoxLayout()
        _btn("Axis from selection", "Determine joint axis and origin from 3D face, edge or point selection", self.axis_from_selection, hz,
             detailed="From a face selected in 3D: cylinder (hole, pin) → axis and origin on the axis at the face; plane → normal and centroid; point / edge (filter Point, Edge) → origin")
        f.addRow(hz)
        self.j_ax2, ha2 = _vec([1, 0, 0], tip="Slot direction (pin slot) / second axis")
        self.j_ax2_lbl = QLabel("Axis 2 (slot)")
        f.addRow(self.j_ax2_lbl, ha2)
        hr = QHBoxLayout()
        self.j_ratio = dspin(0.0, -1e6, 1e6, 6, 0.1)
        help_tip(self.j_ratio, "Gear ratio: ω_b = ratio · ω_a (gears), displacement = ratio · rotation",
                 "Ratio: gear ω_b = ratio · ω_a; rack and pinion translation = ratio [mm/rad] · rotation")
        self.j_lead = dspin(0.0, -1e6, 1e6, 5, 0.5, " mm")
        self.j_lead.setToolTip("Lead of the screw joint [mm per revolution]")
        for w in (QLabel("ratio"), self.j_ratio, QLabel("lead"), self.j_lead):
            hr.addWidget(w)
        self.j_ratio_lbl = QLabel("Transmission")
        f.addRow(self.j_ratio_lbl, hr)
        hj = QHBoxLayout()
        self.j_ja, self.j_jb = QComboBox(), QComboBox()
        self.j_ja.setToolTip("Coupled joint A (gear / rack and pinion)")
        self.j_jb.setToolTip("Coupled joint B (gear / rack and pinion)")
        hj.addWidget(self.j_ja, 1)
        hj.addWidget(self.j_jb, 1)
        self.j_jj_lbl = QLabel("Coupled joints")
        f.addRow(self.j_jj_lbl, hj)
        self.j_pa, hpa = _vec([0, 0, 0], tip="Point A of the link [mm] (on body A)")
        self.j_pb, hpb = _vec([0, 0, 0], tip="Point B of the link [mm] (on body B)")
        self.j_pa_lbl, self.j_pb_lbl = QLabel("Point A [mm]"), QLabel("Point B [mm]")
        f.addRow(self.j_pa_lbl, hpa)
        f.addRow(self.j_pb_lbl, hpb)
        hl = QHBoxLayout()
        self.j_lim = QCheckBox("limits")
        self.j_lim.setToolTip("Joint stops: lower and upper limit of the main coordinate (rad for rotation, mm for translation)")
        self.j_lo = dspin(-1.0, -1e6, 1e6, 5, 0.1)
        self.j_lo.setToolTip("Lower limit (stop) [rad | mm]")
        self.j_hi = dspin(1.0, -1e6, 1e6, 5, 0.1)
        self.j_hi.setToolTip("Upper limit (stop) [rad | mm]")
        for w in (self.j_lim, self.j_lo, self.j_hi):
            hl.addWidget(w)
        f.addRow("Stops", hl)
        hs = QHBoxLayout()
        self.j_kst = dspin(0.0, 0.0, 1e12, 3, 1e3, special="automatic")
        help_tip(self.j_kst, "Stop stiffness [N/mm | N·mm/rad]; 0 = automatic",
                 "Stop stiffness [N/mm | N·mm/rad]; 0 = automatic (impact duration ~1 ms by inertia)")
        self.j_rest = dspin(0.5, 0.0, 1.0, 3, 0.05)
        self.j_rest.setToolTip("Coefficient of restitution e at the stop: 0 = plastic impact, 1 = elastic rebound")
        for w in (QLabel("stiffness"), self.j_kst, QLabel("restitution e"), self.j_rest):
            hs.addWidget(w)
        f.addRow(hs)
        hp = QHBoxLayout()
        self.j_sel_lbl = QLabel("")
        self.j_sel_lbl.setWordWrap(True)
        _btn("Face A from selection", "Joint face on body A (hole, pin) from the 3D selection – for transferring the reaction to FEA",
             lambda: self.joint_face("a"), hp)
        _btn("Face B from selection", "Joint face on body B from the 3D selection – for transferring the reaction to FEA",
             lambda: self.joint_face("b"), hp)
        f.addRow(hp)
        f.addRow(self.j_sel_lbl)
        self.j_fr = QGroupBox("Joint friction")
        self.j_fr.setCheckable(True)
        self.j_fr.setChecked(False)
        help_tip(self.j_fr, "Friction in the joint as a moment or force from the reaction",
                 "Joint friction as a torque / force from the reaction (Coulomb, viscous, Stribeck, bearing, screw)")
        ff = QFormLayout(self.j_fr)
        self.fr_model = QComboBox()
        for k, t in FRICTION_LABELS.items():
            self.fr_model.addItem(t, k)
        help_tip(self.fr_model, "Friction model: Coulomb, viscous, Stribeck, rolling bearing or screw",
                 "Friction model: Coulomb μ·|R|·r (pin), viscous c·v, Stribeck (drop from static to kinetic), rolling bearing, lead screw")
        ff.addRow("Model", self.fr_model)
        hm = QHBoxLayout()
        self.fr_mus = dspin(0.15, 0.0, 3.0, 4, 0.01)
        self.fr_mus.setToolTip("Static friction coefficient μ_s (friction library: help Friction)")
        self.fr_muk = dspin(0.1, 0.0, 3.0, 4, 0.01)
        self.fr_muk.setToolTip("Kinetic friction coefficient μ_k (≤ μ_s)")
        self.fr_r = dspin(5.0, 0.0, 1e5, 3, 0.5, " mm")
        self.fr_r.setToolTip("Effective friction radius r [mm] (pin: radius, bearing: d/2)")
        for w in (QLabel("μ_s"), self.fr_mus, QLabel("μ_k"), self.fr_muk, QLabel("r"), self.fr_r):
            hm.addWidget(w)
        ff.addRow(hm)
        hv = QHBoxLayout()
        self.fr_cv = dspin(0.0, 0.0, 1e9, 5, 0.01)
        self.fr_cv.setToolTip("Viscous coefficient c_v [N·s/mm | N·mm·s/rad]")
        self.fr_vs = dspin(1.0, 1e-9, 1e6, 4, 0.1)
        self.fr_vs.setToolTip("Stribeck velocity v_s [mm/s | rad/s]")
        self.fr_pre = dspin(0.0, 0.0, 1e9, 3, 10.0, " N")
        self.fr_pre.setToolTip("Constant normal force [N] (seal, preload) in addition to the reaction")
        for w in (QLabel("c_v"), self.fr_cv, QLabel("v_s"), self.fr_vs, QLabel("preload"), self.fr_pre):
            hv.addWidget(w)
        ff.addRow(hv)
        self.fr_fit = QComboBox()
        for k, t in FIT_LABELS.items():
            self.fr_fit.addItem(t, k)
        help_tip(self.fr_fit, "Pressure distribution in the pin – affects the friction moment (μ'·|R|·r)",
                 "Pressure distribution in the pin: clearance μ/√(1+μ²), worn-in 4/π·…, fitted π/2·… (torque = μ'·|R|·r)")
        ff.addRow("Pin", self.fr_fit)
        f.addRow(self.j_fr)
        hb = QHBoxLayout()
        self.btn_joint_add = _btn("Add joint", "New joint from the form", self.add_joint, hb)
        _btn("Save changes", "Overwrite the selected joint with the form values", self.update_joint, hb)
        _btn("Delete", "Delete the selected joint", self.delete_joint, hb)
        f.addRow(hb)
        lay.addWidget(g)
        self.on_joint_kind()

    def _tab_drivers(self, lay):
        lay.addWidget(QLabel("<b>Pohony</b>"))
        self.d_table = table(["Driver", "Joint", "Quantity", "Law", "Data"],
                             placeholder="No drivers yet",
                             rows=(3, 8))
        help_tip(self.d_table, "Joint drivers: kinematic or force – click = edit and preview of the profile",
                 "Joint drivers: kinematic (angle, displacement, velocity) or force (torque, force). Click = edit and history preview.")
        self.d_table.currentCellChanged.connect(lambda r, *_: self.on_driver_row(r))
        lay.addWidget(self.d_table)
        g = Section("Driver", key="motion.driver", collapsed=False, tip="Joint driver: quantity and law in time")
        f = QFormLayout(g.body)
        self.d_name = QLineEdit()
        self.d_name.setToolTip("Driver name (e.g. Motor)")
        f.addRow("Name", self.d_name)
        self.d_joint = QComboBox()
        self.d_joint.setToolTip("Driven joint")
        f.addRow("Joint", self.d_joint)
        self.d_qty = QComboBox()
        for k, t in QUANTITY_LABELS.items():
            self.d_qty.addItem(t, k)
        help_tip(self.d_qty, "Driver quantity: angle, displacement, velocity (kinematic) or torque, force",
                 "Driver quantity: angle / displacement / velocity (kinematic driver – prescribed motion) or torque / force (force driver)")
        f.addRow("Quantity", self.d_qty)
        self.d_law = QComboBox()
        for k, t in LAW_LABELS.items():
            self.d_law.addItem(t, k)
        help_tip(self.d_law, "Law in time: constant, table, sine, polynomial, S-curve",
                 "Law in time: constant, table [t, value], sine, polynomial c0 + c1·t + …, S-curve (axis profile from rest to rest with limited jerk)")
        self.d_law.currentIndexChanged.connect(lambda _: self.on_law())
        f.addRow("Law", self.d_law)
        self.d_stack = QStackedWidget()
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 0, 0, 0)
        self.d_const = dspin(1.0, -1e9, 1e9, 6, 1.0)
        self.d_const.setToolTip("Driver value (in the unit of the quantity)")
        hl.addWidget(self.d_const)
        self.d_stack.addWidget(w)
        self.d_tab = EditTable(["t [s]", "value"], "Driver table [t, value] – linear between points",
                               detailed="Driver table [t, value] – linear between points, outside the range the end value")
        self.d_stack.addWidget(self.d_tab)
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 0, 0, 0)
        self.d_sine = [dspin(v, -1e9, 1e9, 5, 0.1) for v in (1.0, 1.0, 0.0, 0.0)]
        for s, t in zip(self.d_sine, ("Amplitude", "Frequency [Hz]", "Phase [rad]", "Mean")):
            s.setToolTip(f"Sine: {t}")
            hl.addWidget(s)
        self.d_stack.addWidget(w)
        self.d_poly = QLineEdit("0; 1")
        self.d_poly.setToolTip("Polynomial coefficients c0; c1; c2; … (value = c0 + c1·t + c2·t² + …)")
        self.d_stack.addWidget(self.d_poly)
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(0, 0, 0, 0)
        self.d_sc = [dspin(v, -1e9, 1e9, 5, 1.0) for v in (100.0, 200.0, 2000.0, 50000.0, 0.0)]
        for s, t in zip(self.d_sc, ("Stroke (travel / angle)", "v_max", "a_max", "j_max (jerk)", "Start t_start [s]")):
            s.setToolTip(f"S-curve: {t}")
            hl.addWidget(s)
        self.d_stack.addWidget(w)
        f.addRow("Data", self.d_stack)
        self.d_plot = LazyPlot()
        self.d_plot.setToolTip("Driver history preview: value, 1st and 2nd derivative over the analysis time")
        f.addRow(self.d_plot)
        hb = QHBoxLayout()
        _btn("Preview", "Plot the law history (value, velocity, acceleration) up to the analysis time", self.preview_law, hb)
        self.btn_drv_add = _btn("Add driver", "New driver from the form", self.add_driver, hb)
        _btn("Save changes", "Overwrite the selected driver", self.update_driver, hb)
        _btn("Delete", "Delete the selected driver", self.delete_driver, hb)
        f.addRow(hb)
        lay.addWidget(g)

        lay.addWidget(QLabel("<b>Force elements</b>"))
        self.f_table = table(["Element", "Type", "A", "B", "k", "c"], placeholder="No force elements yet", rows=(3, 8))
        help_tip(self.f_table, "Springs, dampers, torsion and gas springs and bushings – click = edit",
                 "Springs, dampers, torsion springs, gas springs and bushings between bodies. Click = edit.")
        self.f_table.currentCellChanged.connect(lambda r, *_: self.on_force_row(r))
        lay.addWidget(self.f_table)
        g = Section("Force element", key="motion.force", collapsed=True, tip="Force element between bodies A and B")
        f = QFormLayout(g.body)
        hn = QHBoxLayout()
        self.f_name = QLineEdit()
        self.f_name.setToolTip("Element name")
        self.f_kind = QComboBox()
        for k, t in FORCE_LABELS.items():
            self.f_kind.addItem(t, k)
        help_tip(self.f_kind, "Force element type: spring, damper, torsion, gas spring, bushing",
                 "Type: spring / damper (axis = line through the points, F = f0 + k·(l − l0) + c·l̇), torsion spring in a joint, gas spring (table F(stroke)), bushing (6 stiffnesses at point A)")
        hn.addWidget(self.f_name, 1)
        hn.addWidget(self.f_kind, 1)
        f.addRow("Name, type", hn)
        hb2 = QHBoxLayout()
        self.f_a, self.f_b = QComboBox(), QComboBox()
        self.f_a.setToolTip("Body A")
        self.f_b.setToolTip("Body B")
        hb2.addWidget(self.f_a, 1)
        hb2.addWidget(self.f_b, 1)
        f.addRow("Bodies A, B", hb2)
        self.f_pa, h1 = _vec([0, 0, 0], tip="Attachment point A [mm]")
        self.f_pb, h2 = _vec([0, 0, 0], tip="Attachment point B [mm]")
        _btn("from selection", "Point A from the 3D selection (point, edge centre, face centroid)", lambda: self.point_to(self.f_pa), h1)
        _btn("from selection", "Point B from the 3D selection (point, edge centre, face centroid)", lambda: self.point_to(self.f_pb), h2)
        f.addRow("Point A [mm]", h1)
        f.addRow("Point B [mm]", h2)
        hk = QHBoxLayout()
        self.f_k = dspin(0.0, -1e12, 1e12, 5, 1.0)
        self.f_k.setToolTip("Stiffness k [N/mm | N·mm/rad for a torsion spring]")
        self.f_c = dspin(0.0, 0.0, 1e12, 5, 0.1)
        self.f_c.setToolTip("Damping c [N·s/mm | N·mm·s/rad]")
        self.f_f0 = dspin(0.0, -1e9, 1e9, 4, 1.0)
        self.f_f0.setToolTip("Preload f0 [N | N·mm] (force at l = l0)")
        self.f_l0 = dspin(0.0, 0.0, 1e9, 4, 1.0, special="from CAD")
        self.f_l0.setToolTip("Free length l0 [mm] (0 = length in the CAD position)")
        for w in (QLabel("k"), self.f_k, QLabel("c"), self.f_c, QLabel("f0"), self.f_f0, QLabel("l0"), self.f_l0):
            hk.addWidget(w)
        f.addRow(hk)
        self.f_joint = QComboBox()
        self.f_joint.setToolTip("Joint of the torsion spring (torque about the joint axis)")
        f.addRow("Joint (torque)", self.f_joint)
        self.f_tab = EditTable(["stroke [mm]", "force [N]"], "Gas spring: table F(stroke), positive force pushes")
        f.addRow(self.f_tab)
        self.f_k6 = QLineEdit("")
        self.f_k6.setToolTip("Bushing: 6 stiffnesses k_x; k_y; k_z [N/mm]; k_rx; k_ry; k_rz [N·mm/rad]")
        self.f_c6 = QLineEdit("")
        self.f_c6.setToolTip("Bushing: 6 damping values (same order as the stiffnesses)")
        f.addRow("Bushing k (6)", self.f_k6)
        f.addRow("Bushing c (6)", self.f_c6)
        hs = QHBoxLayout()
        _btn("Face A from selection", "Attachment face on body A (for transferring the element force to FEA)",
             lambda: self.force_face("a"), hs)
        _btn("Face B from selection", "Attachment face on body B (for transferring the element force to FEA)",
             lambda: self.force_face("b"), hs)
        f.addRow(hs)
        self.f_sel_lbl = QLabel("")
        f.addRow(self.f_sel_lbl)
        hb = QHBoxLayout()
        _btn("Add element", "New force element from the form", self.add_force, hb)
        _btn("Save changes", "Overwrite the selected element", self.update_force, hb)
        _btn("Delete", "Delete the selected element", self.delete_force, hb)
        f.addRow(hb)
        lay.addWidget(g)

        lay.addWidget(QLabel("<b>External loads and gravity</b>"))
        self.l_table = table(["Loads", "Body", "Force [N]", "Moment [N·mm]", "Waveform"],
                             placeholder="No loads yet",
                             rows=(3, 8))
        help_tip(self.l_table, "External forces and moments on bodies, profile multiplied by the table over time – click = edit",
                 "External forces and moments on bodies (at a point, global), history by a table multiplier in time. Click = edit.")
        self.l_table.currentCellChanged.connect(lambda r, *_: self.on_load_row(r))
        lay.addWidget(self.l_table)
        g = Section("Loads", key="motion.load", collapsed=True, tip="External force / moment on a body")
        f = QFormLayout(g.body)
        hn = QHBoxLayout()
        self.l_name = QLineEdit()
        self.l_name.setToolTip("Load name")
        self.l_body = QComboBox()
        self.l_body.setToolTip("Body the load acts on")
        hn.addWidget(self.l_name, 1)
        hn.addWidget(self.l_body, 1)
        f.addRow("Name, body", hn)
        self.l_pt, h1 = _vec([0, 0, 0], tip="Point of action [mm]")
        _btn("from selection", "Point of action from the 3D selection", lambda: self.point_to(self.l_pt), h1)
        f.addRow("Point [mm]", h1)
        self.l_F, h2 = _vec([0, 0, 0], tip="Force [N]")
        f.addRow("Force [N]", h2)
        self.l_M, h3 = _vec([0, 0, 0], tip="Moment [N·mm]")
        f.addRow("Moment [N·mm]", h3)
        self.l_tab = EditTable(["t [s]", "multiplier"], "Load history: multiplier of force and moment in time [t, k]",
                               detailed="Load history: multiplier of force and moment in time [t, k]; empty table = always 1")
        f.addRow(self.l_tab)
        hs = QHBoxLayout()
        _btn("Face from selection", "Face the load acts on (for transfer to FEA)", self.load_face, hs)
        self.l_sel_lbl = QLabel("")
        hs.addWidget(self.l_sel_lbl, 1)
        f.addRow(hs)
        hb = QHBoxLayout()
        _btn("Add load", "New load from the form", self.add_load, hb)
        _btn("Save changes", "Overwrite the selected load", self.update_load, hb)
        _btn("Delete", "Delete the selected load", self.delete_load, hb)
        f.addRow(hb)
        lay.addWidget(g)
        hg = QHBoxLayout()
        self.g_on = QCheckBox("gravity")
        self.g_on.setToolTip("Include gravity (gravitational acceleration) in the mechanism analysis")
        hg.addWidget(self.g_on)
        self.g_vec, hv = _vec([0, 0, -9810.0], tip="Gravitational acceleration [mm/s²]")
        hg.addLayout(hv)
        lay.addLayout(hg)
        self.g_on.toggled.connect(lambda _: self.on_gravity())
        for s in self.g_vec:
            s.editingFinished.connect(self.on_gravity)

    def _tab_run(self, lay):
        g = Section("Analysis settings",
                    key="motion.run",
                    collapsed=False,
                    tip="Analysis type, duration, outputs, solver, collision check")
        f = QFormLayout(g.body)
        self.r_kind = QComboBox()
        for k, t in RUN_KINDS.items():
            self.r_kind.addItem(t, k)
        help_tip(self.r_kind, "Analysis type: kinematic, quasi-static or dynamic",
                 "Kinematic: drivers prescribe the motion, forces from inverse dynamics (incl. inertia).<br>Quasi-static: equilibrium at a series of positions without inertia.<br>Dynamic: direct integration of the equations of motion (force drivers, run-down, impacts at stops, flexible bodies).")
        f.addRow("Type", self.r_kind)
        ht = QHBoxLayout()
        self.r_tend = dspin(1.0, 1e-9, 1e6, 7, 0.1, " s")
        self.r_tend.setToolTip("Analysis duration t_end [s]")
        self.r_out = QSpinBox()
        self.r_out.setRange(2, 1000000)
        self.r_out.setValue(200)
        self.r_out.setToolTip("Number of output times (animation frames and chart points)")
        self.r_dt = dspin(0.0, 0.0, 1e3, 8, 1e-4, " s", special="automatic")
        self.r_dt.setToolTip("Integration step [s] (0 = automatic by outputs)")
        for w in (QLabel("t_end"), self.r_tend, QLabel("outputs"), self.r_out, QLabel("dt"), self.r_dt):
            ht.addWidget(w)
        f.addRow(ht)
        self.r_solver = QComboBox()
        self.r_solver.addItem("Exudyn (default)", "exudyn")
        self.r_solver.addItem("Chrono (contact of bodies)", "chrono")
        help_tip(self.r_solver, "MBD solver: Exudyn (default), or Chrono with contact between bodies",
                 "MBD solver: Exudyn (joints, drivers, friction, stops, flexible bodies); Chrono (additionally contact of bodies from CAD) – optional module")
        f.addRow("Solver", self.r_solver)
        self.r_rho = dspin(0.0, 0.0, 1.0, 3, 0.05, special="automatic")
        help_tip(self.r_rho, "Spectral radius ρ∞ of the integrator (0 = automatic)",
                 "Spectral radius ρ∞ of the generalized-α integrator (0 = auto: kinematic 0.6, dynamic 0.8; smaller = more numerical damping)")
        f.addRow("ρ∞", self.r_rho)
        self.r_extra = QCheckBox("mass of omitted parts and bolts to the nearest body")
        self.r_extra.setToolTip("Parts outside the bodies (bolts, omitted parts) add their mass to the nearest body")
        f.addRow(self.r_extra)
        hc = QHBoxLayout()
        self.r_col = QComboBox()
        self.r_col.addItem("no collision check", None)
        self.r_col.addItem("all pairs of bodies", "all")
        self.r_col.addItem("selected pairs", "pairs")
        help_tip(self.r_col, "Collision check of part shapes over time (no contact – report only)",
                 "Collision check without contact: minimum distance of part shapes in time (bodies pass through each other in MBD – the collision is only reported)")
        self.r_ctol = dspin(0.0, 0.0, 1e3, 3, 0.1, " mm")
        self.r_ctol.setToolTip("Collision tolerance [mm] – a distance smaller than the tolerance is reported as a collision")
        hc.addWidget(self.r_col, 1)
        hc.addWidget(QLabel("tolerance"))
        hc.addWidget(self.r_ctol)
        f.addRow("Collisions", hc)
        self.r_pairs = QListWidget()
        self.r_pairs.setMaximumHeight(90)
        self.r_pairs.setToolTip("Pairs of bodies checked for collisions (only the “selected pairs” option)")
        f.addRow(self.r_pairs)
        lay.addWidget(g)
        for w in (self.r_kind, self.r_solver, self.r_col):
            w.currentIndexChanged.connect(lambda _: self.on_run())
        self.r_extra.toggled.connect(lambda _: self.on_run())
        self.r_pairs.itemChanged.connect(lambda _: self.on_run())
        for w in (self.r_tend, self.r_out, self.r_dt, self.r_rho, self.r_ctol):
            w.editingFinished.connect(self.on_run)
            w.valueChanged.connect(lambda *_, _w=w: None if _w.hasFocus() else self.on_run())
        h = QHBoxLayout()
        self.btn_dof = _btn("Mobility check", "Mechanism mobility check: Grübler, constraint rank and redundancy", self.check_dof, h,
                            detailed="Grübler, constraint rank (Exudyn), redundant constraints and missing drivers – in a thread, result in the Messages dock")
        self.btn_run = _btn("Run motion", "Run mechanism computation (standalone process)", self.run_motion, h,
                            detailed="Start the mechanism analysis (separate process) – progress in the bar, then the Results tab")
        self.btn_cancel = _btn("Cancel", "Interrupt the mechanism analysis", self.cancel, h)
        self.btn_cancel.setEnabled(False)
        lay.addLayout(h)
        self.progress = TaskProgress()
        lay.addWidget(self.progress)
        self.run_info = QLabel("")
        self.run_info.setWordWrap(True)
        self.run_info.setTextFormat(Qt.RichText)
        lay.addWidget(self.run_info)

    def _tab_results(self, lay):
        self.res_info = SafeTextBrowser()
        self.res_info.setMinimumHeight(140)
        help_tip(self.res_info, "Motion result summary: maxima, reactions, balance, collisions, warnings",
                 "Summary of the motion result: maxima of velocities and accelerations of bodies, joint reactions, drivers, equilibrium and power balance checks, collisions, warnings")
        lay.addWidget(self.res_info)
        g = Section("Animation",
                    key="motion.anim",
                    collapsed=False,
                    tip="Position of parts in time – slider and playback; afterwards back to the CAD position")
        v = QVBoxLayout(g.body)
        h = QHBoxLayout()
        self.a_slider = QSlider(Qt.Horizontal)
        self.a_slider.setToolTip("Animation time – the parts of the bodies move to the position at this time")
        self.a_slider.valueChanged.connect(lambda _: self.show_frame())
        self.a_time = QLabel("")
        self.a_time.setMinimumWidth(90)
        h.addWidget(self.a_slider, 1)
        h.addWidget(self.a_time)
        v.addLayout(h)
        h = QHBoxLayout()
        self.a_play = QPushButton("► Play")
        self.a_play.setCheckable(True)
        self.a_play.setToolTip("Play / stop the animation (in a loop)")
        self.a_play.toggled.connect(self.play)
        h.addWidget(self.a_play)
        self.a_speed = QComboBox()
        for k, t in ((1, "1 frame / step"), (2, "2 frames / step"), (5, "5 frames / step"), (10, "10 frames / step")):
            self.a_speed.addItem(t, k)
        self.a_speed.setToolTip("Playback speed (number of output times per animation step, 25 steps/s)")
        h.addWidget(self.a_speed)
        _btn("CAD position", "Return the parts to the CAD position (end of animation)", self.reset_pose, h)
        v.addLayout(h)
        lay.addWidget(g)
        g = Section("Chart", key="motion.graph", collapsed=False, tip="History of any result quantity in time")
        f = QFormLayout(g.body)
        hq = QHBoxLayout()
        self.g_qty = QComboBox()
        help_tip(self.g_qty, "Graph quantity: position, velocity, acceleration, reaction, drivers, energy…",
                 "Quantity: position / velocity / acceleration of bodies, joint reactions and moments, joint coordinates, drivers, elements, energy, work, distance (collisions)")
        self.g_item = QComboBox()
        self.g_item.setToolTip("Body / joint / driver / element the quantity belongs to")
        self.g_comp = QComboBox()
        self.g_comp.setToolTip("Component (all, vector magnitude or one component)")
        for w in (self.g_qty, self.g_item, self.g_comp):
            hq.addWidget(w, 1)
        f.addRow(hq)
        self.g_qty.currentIndexChanged.connect(lambda _: self.fill_items())
        self.g_item.currentIndexChanged.connect(lambda _: self.fill_comps())
        self.g_comp.currentIndexChanged.connect(lambda _: self.draw())
        self.g_plot = LazyPlot()
        self.g_plot.setToolTip("Chart of the selected quantity; vertical line = animation time")
        f.addRow(self.g_plot)
        hm = QHBoxLayout()
        _btn("Find maximum", "Instant of the largest magnitude of the chosen quantity – the animation slider jumps to this time",
             self.find_max, hm)
        self.g_max = QLabel("")
        self.g_max.setWordWrap(True)
        hm.addWidget(self.g_max, 1)
        f.addRow(hm)
        lay.addWidget(g)
        h = QHBoxLayout()
        self.btn_fea = _btn("Transfer to FEA…", "Convert motion results to FEA load cases", self.to_fea, h,
                            detailed="Instant / maximum / whole history → load cases of the body (joint reactions as remote loads, inertia), support by a remote support")
        _btn("Support check", "After the FEA analysis: support reaction from FEA versus the joint reaction from MBD (inertia relief)",
             self.check_relief, h)
        lay.addLayout(h)
        self.fea_info = QLabel("")
        self.fea_info.setWordWrap(True)
        self.fea_info.setTextFormat(Qt.RichText)
        lay.addWidget(self.fea_info)

    def scroll_to(self, w):
        scroll_to(self._scroll, w)

    # ================================================================== study → form
    @property
    def study(self):
        return self.win.study

    @property
    def mech(self):
        return self.win.study.mechanism

    def body_names(self) -> list[str]:
        return [b.name for b in self.mech.bodies] if self.study is not None else []

    def refresh(self):
        st = self.study
        if st is None:
            return
        res = self.win.motion_result
        if res is not None and getattr(self.win, "_motion_sig", None) != mech_signature(st):
            self.win.motion_result = None                # the mechanism changed – the result is not valid
            self.reset_pose()
            self.win.log_msg("Mechanism changed – motion result discarded, run the analysis again.")
        self._filling = True
        try:
            self.fill_bodies()
            self.fill_joints()
            self.fill_drivers()
            self.fill_forces()
            self.fill_loads()
            m = self.mech
            self.g_on.setChecked(m.gravity is not None)
            _set(self.g_vec, m.gravity or [0, 0, -9810.0])
            r = m.run
            self.r_kind.setCurrentIndex(max(0, self.r_kind.findData(r.kind)))
            self.r_tend.setValue(r.t_end)
            self.r_out.setValue(int(r.outputs))
            self.r_dt.setValue(r.dt or 0.0)
            self.r_solver.setCurrentIndex(max(0, self.r_solver.findData(r.solver)))
            self.r_rho.setValue(r.spectral_radius or 0.0)
            self.r_extra.setChecked(r.include_extra_mass)
            col = r.collisions
            self.r_col.setCurrentIndex(0 if not col else 1 if col == "all" else 2)
            self.r_ctol.setValue(r.collision_tol or 0.0)
            pairs = {frozenset(p) for p in col} if isinstance(col, list) else set()
            self.r_pairs.clear()
            names = self.body_names()
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    it = QListWidgetItem(f"{a} | {b}")
                    it.setData(Qt.UserRole, [a, b])
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Checked if frozenset((a, b)) in pairs else Qt.Unchecked)
                    self.r_pairs.addItem(it)
            self.r_pairs.setEnabled(self.r_col.currentData() == "pairs")
            self._chrono_state()
        finally:
            self._filling = False
        self.fill_results()
        self.update_info()

    def _chrono_state(self):
        try:
            from fea_core import modules
            ok = modules.available("chrono")
        except Exception:  # noqa: BLE001
            ok = False
        it = self.r_solver.model().item(1)
        it.setEnabled(ok)
        it.setToolTip("" if ok else "The Chrono module is not installed (Tools → Program modules)")

    def fill_bodies(self):
        m = self.mech
        t = self.b_table
        cur = t.currentRow()
        t.blockSignals(True)
        t.setRowCount(0)
        for b in m.bodies:
            r = t.rowCount()
            t.insertRow(r)
            p = self._props.get(b.name)
            mass = 1e3 * b.mass if b.mass is not None else (1e3 * p["mass"] if p else None)
            com = b.com if b.com is not None else (p["com"] if p else None)
            src = "entered" if b.mass is not None else ("CAD" + (" (printing indicative)" if p and p.get("printed") else "")
                                                       if p else "–")
            vals = [b.name, ", ".join(b.parts) or "–", "yes" if b.ground else "",
                    fmt(mass, 4) if mass is not None and not b.ground else "–",
                    "(" + "; ".join(fmt(float(c), 4) for c in com) + ")" if com is not None and not b.ground else "–",
                    src + (" · flexible" if b.flexible else "")]
            for k, v in enumerate(vals):
                t.setItem(r, k, QTableWidgetItem(v))
        if 0 <= cur < t.rowCount():
            t.selectRow(cur)
        t.blockSignals(False)
        names = self.body_names()
        for cb in (self.j_a, self.j_b, self.f_a, self.f_b, self.l_body):
            txt = cb.currentText()
            cb.clear()
            if cb in (self.j_a, self.j_b, self.f_a, self.f_b):
                cb.addItem(GROUND, "")
            for n in names:
                cb.addItem(n, n)
            k = cb.findText(txt)
            cb.setCurrentIndex(max(k, 0))
        self.b_parts.clear()
        for p in self.win.solid_parts():
            it = QListWidgetItem(p.name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.b_parts.addItem(it)
        r = self.b_table.currentRow()
        if 0 <= r < len(m.bodies):
            self._load_body(r)

    def fill_joints(self):
        m = self.mech
        t = self.j_table
        cur = t.currentRow()
        t.blockSignals(True)
        t.setRowCount(0)
        for j in m.joints:
            r = t.rowCount()
            t.insertRow(r)
            lim = "–" if not j.limits else " … ".join("–" if x is None else fmt(float(x), 4) for x in j.limits)
            vals = [j.name, JOINT_LABELS.get(j.kind, j.kind), j.body_a or GROUND, j.body_b or GROUND,
                    "(" + "; ".join(fmt(float(c), 4) for c in j.origin) + ")",
                    "(" + "; ".join(fmt(float(c), 3) for c in j.axis) + ")", lim,
                    FRICTION_LABELS.get(j.friction.model, "") if j.friction else "–",
                    {"user": "manual", "cad": "from geometry", "fusion": "Fusion"}.get(j.source, j.source)]
            for k, v in enumerate(vals):
                t.setItem(r, k, QTableWidgetItem(v))
        if 0 <= cur < t.rowCount():
            t.selectRow(cur)
        t.blockSignals(False)
        jn = [j.name for j in m.joints]
        for cb in (self.d_joint, self.j_ja, self.j_jb, self.f_joint):
            txt = cb.currentText()
            cb.clear()
            cb.addItems(jn)
            if txt in jn:
                cb.setCurrentText(txt)

    def fill_drivers(self):
        t = self.d_table
        t.blockSignals(True)
        t.setRowCount(0)
        for d in self.mech.drivers:
            r = t.rowCount()
            t.insertRow(r)
            data = d.data if d.law != "table" else f"{len(d.data)} points"
            for k, v in enumerate((d.name or "–", d.joint, QUANTITY_LABELS.get(d.quantity, d.quantity),
                                   LAW_LABELS.get(d.law, d.law), str(data))):
                t.setItem(r, k, QTableWidgetItem(v))
        t.blockSignals(False)

    def fill_forces(self):
        t = self.f_table
        t.blockSignals(True)
        t.setRowCount(0)
        for e in self.mech.forces:
            r = t.rowCount()
            t.insertRow(r)
            for k, v in enumerate((e.name, FORCE_LABELS.get(e.kind, e.kind), e.body_a or GROUND, e.body_b or GROUND,
                                   fmt(e.k), fmt(e.c))):
                t.setItem(r, k, QTableWidgetItem(v))
        t.blockSignals(False)

    def fill_loads(self):
        t = self.l_table
        t.blockSignals(True)
        t.setRowCount(0)
        for ld in self.mech.loads:
            r = t.rowCount()
            t.insertRow(r)
            for k, v in enumerate((ld.name or "–", ld.body, "(" + "; ".join(fmt(float(x)) for x in ld.force) + ")",
                                   "(" + "; ".join(fmt(float(x)) for x in ld.moment) + ")",
                                   f"{len(ld.table)} points" if ld.table else "constant")):
                t.setItem(r, k, QTableWidgetItem(v))
        t.blockSignals(False)

    def update_info(self):
        st = self.study
        if st is None:
            return
        m = self.mech
        if not m.bodies:
            self.info.setText("No mechanism (optional). Procedure: bodies → joints → drivers → analysis. Results (reactions, inertia) can be transferred to FEA as load cases.")
            return
        from fea_core.motion.run import check_model, grubler
        try:
            errs, warns = check_model(st, self.win.parts or None)
            gr = grubler(st)
        except Exception as e:  # noqa: BLE001
            errs, warns, gr = [f"The mechanism check failed: {e}"], [], {"dof": "?"}
        rows = [f"Bodies {len(m.bodies)}, joints {len(m.joints)}, drivers {len(m.drivers)}; Grübler: {gr.get('dof')} degrees of freedom"]
        rows += [f"<span style='color:#d23b3b'>● {escape(e)}</span>" for e in errs]
        rows += [f"<span style='color:#b36b00'>● {escape(w)}</span>" for w in warns]
        if self.win.motion_result is not None:
            rows.append("<span style='color:#2e9e4f'>● The motion result is computed (Results tab).</span>")
        self.info.setText("<br>".join(rows))

    def errors(self) -> tuple[list[str], list[str]]:
        st = self.study
        if st is None or not self.mech.bodies:
            return [], []
        from fea_core.motion.run import check_model
        try:
            return check_model(st, self.win.parts or None)
        except Exception as e:  # noqa: BLE001
            return [str(e)], []

    def color(self) -> str:
        from .main import GREEN, GREY, ORANGE, RED
        if self.study is None or not self.mech.bodies:
            return GREY
        errs, warns = self.errors()
        if errs:
            return RED
        return GREEN if self.win.motion_result is not None else ORANGE

    # ================================================================== bodies
    def _checked_parts(self) -> list[str]:
        return [self.b_parts.item(i).text() for i in range(self.b_parts.count())
                if self.b_parts.item(i).checkState() == Qt.Checked]

    def parts_from_selection(self):
        names = {self.win.parts[pi].name for pi, _ in self.win.view.selected_faces()}
        names |= {self.win.parts[pi].name for pi in self.win.view.selected_parts()}
        if not names:
            QMessageBox.information(self, APP, "Select faces of parts in 3D (filter Body B selects a whole part).")
            return
        for i in range(self.b_parts.count()):
            it = self.b_parts.item(i)
            if it.text() in names:
                it.setCheckState(Qt.Checked)

    def body_from_form(self) -> Body | None:
        name = self.b_name.text().strip()
        if not name:
            QMessageBox.warning(self, APP, "Enter the body name.")
            return None
        b = Body(name, self._checked_parts(), self.b_ground.isChecked())
        if self.b_override.isChecked():
            b.mass = 1e-3 * self.b_mass.value()
            b.com = _vals(self.b_com)
            b.inertia = np.diag([1e3 * x for x in _vals(self.b_I)]).tolist()       # kg·m² → t·mm²
        v0, w0 = _vals(self.b_v0), _vals(self.b_w0)
        b.v0 = v0 if any(v0) else None
        b.w0 = w0 if any(w0) else None
        b.flexible = self.b_flex.isChecked()
        b.modes = self.b_modes.value()
        b.interface = self.b_iface.currentData()
        b.zeta = self.b_zeta.value() / 100.0
        if not b.parts and not b.ground and not self.b_override.isChecked():
            QMessageBox.warning(self, APP, "A body without parts needs an entered mass and inertia (tick “enter manually”), or tick parts.")
            return None
        return b

    def add_body(self):
        if self.study is None:
            return
        b = self.body_from_form()
        if b is None:
            return
        if b.name in self.body_names():
            QMessageBox.warning(self, APP, f"Body '{b.name}' already exists.")
            return
        self.mech.bodies.append(b)
        self.win.changed()
        self.b_table.selectRow(len(self.mech.bodies) - 1)

    def update_body(self):
        r = self.b_table.currentRow()
        if not (0 <= r < len(self.mech.bodies)):
            QMessageBox.warning(self, APP, "Select a body in the table.")
            return
        b = self.body_from_form()
        if b is None:
            return
        old = self.mech.bodies[r]
        if b.name != old.name:
            if b.name in self.body_names():
                QMessageBox.warning(self, APP, f"Body '{b.name}' already exists.")
                return
            self._rename_body(old.name, b.name)
        b.color = old.color
        self.mech.bodies[r] = b
        self.win.changed()
        self.b_table.selectRow(r)

    def _rename_body(self, old, new):
        m = self.mech
        for j in m.joints:
            j.body_a = new if j.body_a == old else j.body_a
            j.body_b = new if j.body_b == old else j.body_b
        for f in m.forces:
            f.body_a = new if f.body_a == old else f.body_a
            f.body_b = new if f.body_b == old else f.body_b
        for ld in m.loads:
            ld.body = new if ld.body == old else ld.body

    def delete_body(self):
        r = self.b_table.currentRow()
        if 0 <= r < len(self.mech.bodies):
            self.mech.bodies.pop(r)
            self.win.changed()

    def bodies_from_parts(self):
        if self.study is None:
            return
        used = {p for b in self.mech.bodies for p in b.parts}
        names = self.body_names()
        added = 0
        for p in self.win.solid_parts():
            if p.name in used or p.name in names:
                continue
            ground = not any(b.ground for b in self.mech.bodies)
            self.mech.bodies.append(Body(p.name, [p.name], ground))
            added += 1
        if added:
            self.win.changed()
            self.win.log_msg(f"Bodies added from parts: {added} (the first without ground = ground – check it)")

    def compute_props(self):
        st = self.study
        if st is None or not self.mech.bodies:
            return
        from fea_core.motion.run import body_props
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            props = body_props(st, self.win.solid_parts() or None)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, APP, f"Mass properties cannot be computed:\n{e}")
            return
        QApplication.restoreOverrideCursor()
        self._props = {k: v for k, v in props.items() if not k.startswith("_")}
        warn = props.get("_warnings", [])
        self.b_info.setText("Masses from CAD (volume × material ρ)" + (": " + "; ".join(warn) if warn else ".")
                            + " Printed parts approximately.")
        self.fill_bodies()

    def on_body_row(self, r):
        if 0 <= r < len(self.mech.bodies):
            self._load_body(r)
            self.win.refresh_sets()

    def _load_body(self, r):
        b = self.mech.bodies[r]
        self.b_name.setText(b.name)
        for i in range(self.b_parts.count()):
            it = self.b_parts.item(i)
            it.setCheckState(Qt.Checked if it.text() in b.parts else Qt.Unchecked)
        self.b_ground.setChecked(b.ground)
        self.b_override.setChecked(b.mass is not None)
        p = self._props.get(b.name)
        mass = b.mass if b.mass is not None else (p["mass"] if p else 0.0)
        self.b_mass.setValue(1e3 * float(mass or 0.0))
        _set(self.b_com, b.com if b.com is not None else (list(p["com"]) if p else [0, 0, 0]))
        I = np.asarray(b.inertia if b.inertia is not None else (p["inertia"] if p else np.zeros((3, 3))), float)
        I = np.diag(I.reshape(3, 3)) if I.size == 9 else I
        _set(self.b_I, [x / 1e3 for x in I])
        _set(self.b_v0, b.v0 or [0, 0, 0])
        _set(self.b_w0, b.w0 or [0, 0, 0])
        self.b_flex.setChecked(b.flexible)
        self.b_modes.setValue(int(b.modes))
        self.b_iface.setCurrentIndex(max(0, self.b_iface.findData(b.interface)))
        self.b_zeta.setValue(100.0 * b.zeta)

    # ================================================================== joints
    def on_joint_kind(self):
        k = self.j_kind.currentData()
        for w in (self.j_ax2_lbl, *self.j_ax2):
            w.setVisible(k in ("pin_slot", "universal"))
        for w in (self.j_ratio_lbl, self.j_ratio, self.j_lead):
            w.setVisible(k in ("gear", "rack_pinion", "screw"))
        for w in (self.j_jj_lbl, self.j_ja, self.j_jb):
            w.setVisible(k in ("gear", "rack_pinion"))
        for w in (self.j_pa_lbl, self.j_pb_lbl, *self.j_pa, *self.j_pb):
            w.setVisible(k == "distance")

    def joint_from_form(self) -> MotionJoint | None:
        name = self.j_name.text().strip() or f"Joint {len(self.mech.joints) + 1}"
        k = self.j_kind.currentData()
        ax = np.asarray(_vals(self.j_ax), float)
        if np.linalg.norm(ax) < 1e-12 and k not in ("gear", "rack_pinion", "distance", "spherical", "fixed"):
            QMessageBox.warning(self, APP, "The joint axis must not be zero.")
            return None
        j = MotionJoint(name, k, self.j_a.currentData() or "", self.j_b.currentData() or "", _vals(self.j_o),
                        (ax / max(np.linalg.norm(ax), 1e-30)).tolist() if np.linalg.norm(ax) > 0 else [0, 0, 1])
        if k in ("pin_slot", "universal"):
            j.axis2 = _vals(self.j_ax2)
        if k in ("gear", "rack_pinion"):
            j.ratio = _f(self.j_ratio.value())
            j.joint_a, j.joint_b = self.j_ja.currentText() or None, self.j_jb.currentText() or None
        if k == "screw":
            j.lead = _f(self.j_lead.value())
        if k == "distance":
            j.point_a, j.point_b = _vals(self.j_pa), _vals(self.j_pb)
        if self.j_lim.isChecked():
            j.limits = [self.j_lo.value(), self.j_hi.value()]
        j.limit_stiffness = _f(self.j_kst.value())
        j.restitution = self.j_rest.value()
        j.selection_a, j.selection_b = self._jsel["a"], self._jsel["b"]
        if self.j_fr.isChecked():
            j.friction = JointFriction(self.fr_model.currentData(), self.fr_mus.value(), self.fr_muk.value(),
                                       self.fr_cv.value(), self.fr_vs.value(), r_eff=self.fr_r.value(),
                                       fit=self.fr_fit.currentData(), preload=self.fr_pre.value())
        return j

    def add_joint(self):
        if self.study is None:
            return
        j = self.joint_from_form()
        if j is None:
            return
        if j.name in [x.name for x in self.mech.joints]:
            QMessageBox.warning(self, APP, f"Joint '{j.name}' already exists.")
            return
        self.mech.joints.append(j)
        self.win.changed()
        self.j_table.selectRow(len(self.mech.joints) - 1)

    def update_joint(self):
        r = self.j_table.currentRow()
        if not (0 <= r < len(self.mech.joints)):
            QMessageBox.warning(self, APP, "Select a joint in the table.")
            return
        j = self.joint_from_form()
        if j is None:
            return
        old = self.mech.joints[r]
        j.source = old.source
        if j.name != old.name:
            for d in self.mech.drivers:
                d.joint = j.name if d.joint == old.name else d.joint
        self.mech.joints[r] = j
        self.win.changed()
        self.j_table.selectRow(r)

    def delete_joint(self):
        r = self.j_table.currentRow()
        if 0 <= r < len(self.mech.joints):
            j = self.mech.joints[r]
            for s in (j.selection_a, j.selection_b):
                if s:
                    self.win._drop_selection(s)
            self.mech.joints.pop(r)
            self.win.changed()

    def on_joint_row(self, r):
        if not (0 <= r < len(self.mech.joints)):
            return
        j = self.mech.joints[r]
        self.j_name.setText(j.name)
        self.j_kind.setCurrentIndex(max(0, self.j_kind.findData(j.kind)))
        self.j_a.setCurrentIndex(max(0, self.j_a.findData(j.body_a or "")))
        self.j_b.setCurrentIndex(max(0, self.j_b.findData(j.body_b or "")))
        _set(self.j_o, j.origin)
        _set(self.j_ax, j.axis)
        _set(self.j_ax2, j.axis2 or [1, 0, 0])
        self.j_ratio.setValue(j.ratio or 0.0)
        self.j_lead.setValue(j.lead or 0.0)
        if j.joint_a:
            self.j_ja.setCurrentText(j.joint_a)
        if j.joint_b:
            self.j_jb.setCurrentText(j.joint_b)
        _set(self.j_pa, j.point_a or [0, 0, 0])
        _set(self.j_pb, j.point_b or [0, 0, 0])
        self.j_lim.setChecked(bool(j.limits))
        if j.limits:
            self.j_lo.setValue(float(j.limits[0]) if j.limits[0] is not None else -1e6)
            self.j_hi.setValue(float(j.limits[1]) if j.limits[1] is not None else 1e6)
        self.j_kst.setValue(j.limit_stiffness or 0.0)
        self.j_rest.setValue(j.restitution)
        self._jsel = {"a": j.selection_a, "b": j.selection_b}
        self._show_jsel()
        fr = j.friction
        self.j_fr.setChecked(fr is not None)
        if fr is not None:
            self.fr_model.setCurrentIndex(max(0, self.fr_model.findData(fr.model)))
            self.fr_mus.setValue(fr.mu_s)
            self.fr_muk.setValue(fr.mu_k)
            self.fr_cv.setValue(fr.c_v)
            self.fr_vs.setValue(fr.v_s)
            self.fr_r.setValue(fr.r_eff)
            self.fr_fit.setCurrentIndex(max(0, self.fr_fit.findData(fr.fit)))
            self.fr_pre.setValue(fr.preload)
        self.on_joint_kind()
        self.win.refresh_sets()

    def _show_jsel(self):
        self.j_sel_lbl.setText(f"Joint faces: A = {self._jsel['a'] or '–'}, B = {self._jsel['b'] or '–'} (for transferring reactions to FEA)")

    def joint_face(self, side):
        sels = self.win.need_faces()
        if not sels:
            return
        name = self.win.new_selection(f"Joint {side.upper()}", sels)
        self.win.view.set_selected_faces([])
        self._jsel[side] = name
        self._show_jsel()
        self.win.log_msg(f"Joint face {side.upper()}: {name} – save the joint (“Add joint” / “Save changes”).")

    def axis_from_selection(self):
        """Axis and origin of the joint from the 3D selection (cylinder → axis, plane → normal, point / edge → origin)."""
        win, v = self.win, self.win.view
        faces = v.selected_faces()
        pts = v.selected_points()
        e = v.selected_edge()
        if faces:
            from fea_core.geometry import cylinder_axis
            pi, fi = faces[0]
            props = win.face_prop(pi, fi)
            cyl = cylinder_axis(v.part_faces[pi][fi])
            if cyl is not None:
                p, d, _r = cyl
                c = np.asarray(props["centroid"], float)
                o = p + ((c - p) @ d) * d
                _set(self.j_o, o)
                _set(self.j_ax, d)
                return
            _set(self.j_o, props["centroid"])
            _set(self.j_ax, props["normal"])
            return
        if pts:
            _set(self.j_o, pts[-1][1])
            return
        if e is not None:
            _set(self.j_o, e["point"])
            if e.get("axis") is not None:
                _set(self.j_ax, e["axis"])
            return
        QMessageBox.information(self, APP, "Select a cylindrical face (hole, pin), a planar face, a point or an edge in 3D.")

    def point_to(self, spins):
        v = self.win.view
        p = None
        if v.selected_points():
            p = v.selected_points()[-1][1]
        elif v.selected_edge() is not None:
            p = v.selected_edge()["point"]
        elif v.selected_faces():
            sels = self.win.selected_selectors()
            w = np.array([s.area for s in sels])
            p = (np.array([s.centroid for s in sels]) * w[:, None]).sum(axis=0) / max(w.sum(), 1e-30)
        if p is None:
            QMessageBox.information(self, APP, "Select a point (Point filter V), an edge (Edge filter E) or faces in 3D.")
            return
        _set(spins, p)

    def suggest_geometry(self):
        st = self.study
        if st is None or self._thread is not None:
            return
        if len([b for b in self.mech.bodies if b.parts]) < 2:
            QMessageBox.information(self, APP, "First define bodies with parts (Bodies tab) – joints are searched between parts of different bodies.")
            return
        step = self.win.step_path
        from fea_core.study import Study

        def job(_th):
            from fea_core.geometry import legacy_step_names, read_step
            from fea_core.motion.joints_cad import suggest_from_geometry
            # parts are reloaded in a thread (triangulation of shapes shown in 3D would collide with the GUI)
            study = Study.from_dict(st.to_dict())
            return suggest_from_geometry(study, read_step(step, legacy_names=legacy_step_names(study)))
        self._start(job, self._on_suggest, "Searching for joints in the geometry…")

    def suggest_metadata(self):
        st = self.study
        if st is None:
            return
        from PySide6.QtWidgets import QFileDialog
        p, _ = QFileDialog.getOpenFileName(self, "Fusion metadata", str(self.win.step_path.parent if
                                                                          self.win.step_path else ""), "JSON (*.json)")
        if not p:
            return
        try:
            from pathlib import Path
            from fea_core.motion.joints_cad import suggest_from_metadata
            sug = suggest_from_metadata(st, json.loads(Path(p).read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP, f"Metadata cannot be loaded:\n{e}")
            return
        self._on_suggest(sug)

    def _on_suggest(self, sug):
        if not sug:
            QMessageBox.information(self, APP, "No joint found (coaxial cylindrical faces of parts of different bodies). Enter the joints manually.")
            return
        dlg = SuggestionDialog(self, sug)
        self.suggest_dialog = dlg                        # for tests
        ok = dlg.exec() == QDialog.Accepted
        acc = dlg.accepted_suggestions() if ok else []
        self.suggest_dialog = None
        if acc:
            self.accept_suggestions(acc)
        self.win.refresh_sets()

    def accept_suggestions(self, acc):
        from fea_core.motion.joints_cad import apply_suggestion
        n = 0
        for s in acc:
            try:
                apply_suggestion(self.study, self.win.parts, s)
                n += 1
            except Exception as e:  # noqa: BLE001
                self.win.log_msg(f"Suggested joint {s.joint.name} cannot be accepted: {e}")
        if n:
            self.win.resolve_all()
            self.win.changed()
            self.win.log_msg(f"Joints added from suggestions: {n}")

    # ================================================================== pohony
    def on_law(self):
        self.d_stack.setCurrentIndex(list(LAW_LABELS).index(self.d_law.currentData()))

    def law_data(self):
        k = self.d_law.currentData()
        if k == "constant":
            return [self.d_const.value()]
        if k == "table":
            return [[float(a), float(b)] for a, b in self.d_tab.values()]
        if k == "sine":
            return _vals(self.d_sine)
        if k == "poly":
            return [float(x) for x in self.d_poly.text().replace(",", ".").replace(";", " ").split()]
        return _vals(self.d_sc)

    def driver_from_form(self) -> Driver | None:
        if not self.d_joint.currentText():
            QMessageBox.warning(self, APP, "The driver needs a joint (Joints tab).")
            return None
        try:
            data = self.law_data()
            from fea_core.motion.laws import Law
            Law(self.d_law.currentData(), data)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP, f"Law data are not valid: {e}")
            return None
        return Driver(self.d_joint.currentText(), self.d_qty.currentData(), self.d_law.currentData(), data,
                      self.d_name.text().strip())

    def add_driver(self):
        d = self.driver_from_form()
        if d is not None:
            self.mech.drivers.append(d)
            self.win.changed()

    def update_driver(self):
        r = self.d_table.currentRow()
        if not (0 <= r < len(self.mech.drivers)):
            QMessageBox.warning(self, APP, "Select a drive in the table.")
            return
        d = self.driver_from_form()
        if d is not None:
            self.mech.drivers[r] = d
            self.win.changed()

    def delete_driver(self):
        r = self.d_table.currentRow()
        if 0 <= r < len(self.mech.drivers):
            self.mech.drivers.pop(r)
            self.win.changed()

    def on_driver_row(self, r):
        if not (0 <= r < len(self.mech.drivers)):
            return
        d = self.mech.drivers[r]
        self.d_name.setText(d.name)
        self.d_joint.setCurrentText(d.joint)
        self.d_qty.setCurrentIndex(max(0, self.d_qty.findData(d.quantity)))
        self.d_law.setCurrentIndex(max(0, self.d_law.findData(d.law)))
        self.on_law()
        data = d.data
        if d.law == "constant":
            self.d_const.setValue(float(np.ravel(data)[0]) if len(np.ravel(data)) else 0.0)
        elif d.law == "table":
            self.d_tab.set_rows(data)
        elif d.law == "sine":
            _set(self.d_sine, (list(data) + [0, 0, 0, 0])[:4])
        elif d.law == "poly":
            self.d_poly.setText("; ".join(fmt(float(x), 8) for x in data))
        else:
            _set(self.d_sc, (list(data) + [0])[:5])
        self.preview_law()

    def preview_law(self):
        try:
            from fea_core.motion.laws import Law
            law = Law(self.d_law.currentData(), self.law_data())
        except Exception as e:  # noqa: BLE001
            ax = self.d_plot.axes()
            ax.text(0.5, 0.5, f"Invalid data: {e}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            self.d_plot.canvas.draw_idle()
            return
        t_end = max(self.mech.run.t_end, law.duration + (float(self.d_sc[4].value()) if law.kind == "scurve" else 0))
        t = np.linspace(0.0, t_end, 400)
        self.d_plot._ensure()
        fig = self.d_plot.fig
        fig.clear()
        axs = fig.subplots(3, 1, sharex=True)
        for ax, fn, lab in zip(axs, (law.value, law.d1, law.d2), ("value", "1st derivative", "2nd derivative")):
            ax.plot(t, [fn(x) for x in t], color="#1f77b4", lw=1.2)
            ax.set_ylabel(lab, fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
        axs[-1].set_xlabel("t [s]", fontsize=7)
        self.d_plot.canvas.draw_idle()

    # ================================================================== force elements
    def force_from_form(self) -> ForceElement | None:
        name = self.f_name.text().strip() or f"Element {len(self.mech.forces) + 1}"
        kind = self.f_kind.currentData()
        e = ForceElement(name, kind, self.f_a.currentData() or "", self.f_b.currentData() or "", _vals(self.f_pa),
                         _vals(self.f_pb), self.f_k.value(), self.f_c.value(), self.f_f0.value(),
                         _f(self.f_l0.value()))
        if kind == "torsion":
            e.joint = self.f_joint.currentText() or None
        if kind == "gas_spring":
            try:
                e.table = [[float(a), float(b)] for a, b in self.f_tab.values()]
            except (TypeError, ValueError):
                QMessageBox.warning(self, APP, "Gas spring table: fill in pairs of numbers.")
                return None
        if kind == "bushing":
            try:
                e.k6 = [float(x) for x in self.f_k6.text().replace(",", ".").replace(";", " ").split()]
                e.c6 = [float(x) for x in self.f_c6.text().replace(",", ".").replace(";", " ").split()] or [0.0] * 6
            except ValueError:
                QMessageBox.warning(self, APP, "Bushing: enter 6 numbers.")
                return None
            if len(e.k6) != 6 or len(e.c6) != 6:
                QMessageBox.warning(self, APP, "Bushing: enter 6 stiffnesses and 6 damping values.")
                return None
        e.selection_a, e.selection_b = self._fsel["a"], self._fsel["b"]
        return e

    def add_force(self):
        e = self.force_from_form()
        if e is not None:
            self.mech.forces.append(e)
            self.win.changed()

    def update_force(self):
        r = self.f_table.currentRow()
        if not (0 <= r < len(self.mech.forces)):
            QMessageBox.warning(self, APP, "Select an element in the table.")
            return
        e = self.force_from_form()
        if e is not None:
            self.mech.forces[r] = e
            self.win.changed()

    def delete_force(self):
        r = self.f_table.currentRow()
        if 0 <= r < len(self.mech.forces):
            e = self.mech.forces[r]
            for s in (e.selection_a, e.selection_b):
                if s:
                    self.win._drop_selection(s)
            self.mech.forces.pop(r)
            self.win.changed()

    def on_force_row(self, r):
        if not (0 <= r < len(self.mech.forces)):
            return
        e = self.mech.forces[r]
        self.f_name.setText(e.name)
        self.f_kind.setCurrentIndex(max(0, self.f_kind.findData(e.kind)))
        self.f_a.setCurrentIndex(max(0, self.f_a.findData(e.body_a or "")))
        self.f_b.setCurrentIndex(max(0, self.f_b.findData(e.body_b or "")))
        _set(self.f_pa, e.point_a)
        _set(self.f_pb, e.point_b)
        self.f_k.setValue(e.k)
        self.f_c.setValue(e.c)
        self.f_f0.setValue(e.f0)
        self.f_l0.setValue(e.l0 or 0.0)
        if e.joint:
            self.f_joint.setCurrentText(e.joint)
        self.f_tab.set_rows(e.table or [])
        self.f_k6.setText("; ".join(fmt(float(x), 6) for x in (e.k6 or [])))
        self.f_c6.setText("; ".join(fmt(float(x), 6) for x in (e.c6 or [])))
        self._fsel = {"a": e.selection_a, "b": e.selection_b}
        self.f_sel_lbl.setText(f"Attachment faces: A = {e.selection_a or '–'}, B = {e.selection_b or '–'}")

    def force_face(self, side):
        sels = self.win.need_faces()
        if not sels:
            return
        name = self.win.new_selection(f"Element {side.upper()}", sels)
        self.win.view.set_selected_faces([])
        self._fsel[side] = name
        self.f_sel_lbl.setText(f"Attachment faces: A = {self._fsel['a'] or '–'}, B = {self._fsel['b'] or '–'}")

    # ================================================================== loads
    def load_from_form(self) -> MotionLoad | None:
        if not self.l_body.currentText():
            QMessageBox.warning(self, APP, "The load needs a body.")
            return None
        try:
            tab = [[float(a), float(b)] for a, b in self.l_tab.values()]
        except (TypeError, ValueError):
            QMessageBox.warning(self, APP, "Load history: fill in pairs of numbers.")
            return None
        return MotionLoad(self.l_body.currentText(), _vals(self.l_pt), _vals(self.l_F), _vals(self.l_M), tab or None,
                          self.l_name.text().strip() or f"Load {len(self.mech.loads) + 1}", self._lsel)

    def add_load(self):
        ld = self.load_from_form()
        if ld is not None:
            self.mech.loads.append(ld)
            self.win.changed()

    def update_load(self):
        r = self.l_table.currentRow()
        if not (0 <= r < len(self.mech.loads)):
            QMessageBox.warning(self, APP, "Select a load in the table.")
            return
        ld = self.load_from_form()
        if ld is not None:
            self.mech.loads[r] = ld
            self.win.changed()

    def delete_load(self):
        r = self.l_table.currentRow()
        if 0 <= r < len(self.mech.loads):
            ld = self.mech.loads[r]
            if ld.selection:
                self.win._drop_selection(ld.selection)
            self.mech.loads.pop(r)
            self.win.changed()

    def on_load_row(self, r):
        if not (0 <= r < len(self.mech.loads)):
            return
        ld = self.mech.loads[r]
        self.l_name.setText(ld.name)
        self.l_body.setCurrentText(ld.body)
        _set(self.l_pt, ld.point)
        _set(self.l_F, ld.force)
        _set(self.l_M, ld.moment)
        self.l_tab.set_rows(ld.table or [])
        self._lsel = ld.selection
        self.l_sel_lbl.setText(ld.selection or "")

    def load_face(self):
        sels = self.win.need_faces()
        if not sels:
            return
        self._lsel = self.win.new_selection("Motion load", sels)
        self.win.view.set_selected_faces([])
        self.l_sel_lbl.setText(self._lsel)

    def on_gravity(self):
        if self._filling or self.study is None:
            return
        g = _vals(self.g_vec) if self.g_on.isChecked() else None
        if g != self.mech.gravity:
            self.mech.gravity = g
            self.win.changed()

    # ================================================================== calculation
    def on_run(self):
        if self._filling or self.study is None:
            return
        r = self.mech.run
        before = json.dumps(r.__dict__, default=str, sort_keys=True)
        r.kind = self.r_kind.currentData()
        r.t_end = self.r_tend.value()
        r.outputs = self.r_out.value()
        r.dt = _f(self.r_dt.value())
        r.solver = self.r_solver.currentData()
        r.spectral_radius = _f(self.r_rho.value())
        r.include_extra_mass = self.r_extra.isChecked()
        col = self.r_col.currentData()
        if col == "pairs":
            r.collisions = [self.r_pairs.item(i).data(Qt.UserRole) for i in range(self.r_pairs.count())
                            if self.r_pairs.item(i).checkState() == Qt.Checked] or None
        else:
            r.collisions = col
        r.collision_tol = self.r_ctol.value()
        self.r_pairs.setEnabled(col == "pairs")
        if json.dumps(r.__dict__, default=str, sort_keys=True) != before:
            self.win.changed()

    def _start(self, job, done, text):
        from .workers import TaskThread
        t = TaskThread(job, self)
        t.done.connect(done)
        t.failed.connect(self._failed)
        t.finished.connect(self._finished)
        self._thread = t
        self.win.register_aux(t)
        for b in (self.btn_dof, self.btn_run, self.btn_suggest):
            b.setEnabled(False)
        self.win.log_msg(text)
        t.start()
        return t

    def _finished(self):
        self.win.unregister_aux(self._thread)
        self._thread = None
        for b in (self.btn_dof, self.btn_run, self.btn_suggest):
            b.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _failed(self, text):
        cancelled = "cancelled" in text.lower()
        self.progress.finish("cancelled" if cancelled else "error", "Cancelled" if cancelled else "Error – reason in the log")
        self.win.log_msg(f"Motion ERROR: {text}")
        self.win.messages.set_source("pohyb", [("warning" if cancelled else "error", text)], "Motion")
        if not cancelled:
            self.win.dock_messages.raise_()
            QMessageBox.warning(self, APP, f"Mechanism analysis failed:\n\n{text}")

    def _study_copy(self):
        from fea_core.study import Study
        st = Study.from_dict(self.study.to_dict())
        st.geometry = self.win.geometry_str()
        return st

    def check_dof(self):
        if self.study is None or self._thread is not None:
            return
        st = self._study_copy()
        wd = self.win._workdir / "motion_dof"

        def job(_th):
            from fea_core.motion.run import check_dof
            return check_dof(st, None, workdir=wd)
        self._start(job, self._on_dof, "Mechanism mobility check…")

    def _on_dof(self, d):
        self.last_dof = d
        ok = d.get("ok")
        text = (f"Grübler {d['grubler']}, actual mobility {d['dof_free']}, after drivers {d['dof_driven']}, redundant constraints {d['redundant']}")
        self.run_info.setText(("<span style='color:#2e9e4f'>● " if ok else "<span style='color:#d23b3b'>● ") + text
                              + "</span><br>" + "<br>".join(escape(str(x)) for x in d.get("messages", [])))
        items = [("info" if ok else "warning", "Mobility check: " + text)]
        items += [("warning", m) for m in d.get("messages", [])]
        self.win.messages.set_source("pohyb", items, "Motion")
        self.win.log_msg("Mobility check: " + text)

    def run_motion(self):
        if self.study is None or self._thread is not None:
            return
        from fea_core.motion.run import check_model
        errs, _ = check_model(self.study, self.win.parts or None)
        if errs:
            QMessageBox.warning(self, APP, "The mechanism is incomplete:\n\n" + "\n".join(errs))
            return
        st = self._study_copy()
        sig = mech_signature(self.study)
        import time
        wd = self.win._workdir / f"motion_{int(time.time() * 1000)}"
        t = None

        def job(th):
            from fea_core.motion.run import run_motion
            return run_motion(st, wd, None, on_log=th.log.emit, cancel=th.cancel,
                              on_progress=lambda v, e, txt: th.progress.emit(int(1000 * v), int(1000 * e), txt))
        t = self._start(job, lambda res: self._on_result(res, sig), "Computing the mechanism motion…")
        t.log.connect(self.win.log_msg)
        t.progress.connect(lambda v, e, txt: self.progress.report(v / 1000.0, e / 1000.0, txt))
        self.btn_cancel.setEnabled(True)
        self.progress.start("Mechanism (Exudyn)", 0.05)

    def cancel(self):
        if self._thread is not None:
            self._thread.cancel.set()
            self.win.log_msg("Cancelling the mechanism analysis…")

    def _on_result(self, res, sig):
        if self.study is None or mech_signature(self.study) != sig:
            self.progress.finish("cancelled", "Result discarded – the mechanism changed")
            self.win.log_msg("The mechanism changed during the analysis – result discarded.")
            return
        self.win.motion_result = res
        self.win._motion_sig = sig
        self.win.dirty = True
        self.win.update_title()
        self.progress.finish("done", f"Done in {fmt(res.meta.get('time_s', 0.0), 3)} s")
        items = [("info", f"Motion computed: {len(res.arrays['t'])} outputs up to {fmt(float(res.arrays['t'][-1]), 4)} s")]
        items += [("warning", w) for w in res.warnings]
        self.win.messages.set_source("pohyb", items, "Motion")
        self.win.log_msg(f"Motion computed ({len(res.arrays['t'])} outputs).")
        self.fill_results()
        self.update_info()
        self.win.refresh_flow()
        self.win.refresh_tree()
        self.tabs.setCurrentIndex(4)

    # ================================================================== results
    def fill_results(self):
        res = self.win.motion_result
        has = res is not None
        for w in (self.a_slider, self.a_play, self.btn_fea, self.g_qty, self.g_item, self.g_comp):
            w.setEnabled(has)
        if not has:
            self.res_info.setHtml("<i>No motion result yet – the Results tab is filled after the analysis (Analysis tab → Run motion).</i>")
            self.a_time.setText("")
            return
        n = len(res.arrays["t"])
        self.a_slider.blockSignals(True)
        self.a_slider.setRange(0, n - 1)
        self.a_slider.blockSignals(False)
        self.res_info.setHtml(self.summary_html(res))
        cur = self.g_qty.currentText()
        self.g_qty.blockSignals(True)
        self.g_qty.clear()
        from fea_core.motion.results import MotionResult
        qs = [q for q, spec in MotionResult.SERIES.items() if self._items_for(spec[0])]
        qs += ["energy", "work"] + (["distance"] if (res.meta.get("collisions") or {}).get("pairs") else [])
        self.g_qty.addItems(qs)
        if cur in qs:
            self.g_qty.setCurrentText(cur)
        else:
            self.g_qty.setCurrentText("reaction" if "reaction" in qs else qs[0])
        self.g_qty.blockSignals(False)
        self.fill_items()
        self.a_time.setText(f"t = {fmt(float(res.arrays['t'][self.a_slider.value()]), 4)} s")

    def summary_html(self, res) -> str:
        s = res.summary()
        rows = [f"<b>{ {'kinematic': 'Kinematic', 'static': 'Quasi-static', 'dynamic': 'Dynamic'}.get(s.get('kind'), '')} analysis</b> up to {fmt(s['t_end_s'], 4)} s, {s['frames']} outputs, computation {fmt(s.get('time_s') or 0, 3)} s"]
        for b in s.get("bodies", []):
            rows.append(f"{escape(str(b['body']))}: {fmt(b['mass_kg'], 4)} kg, v_max {fmt(b['max_velocity_mm_s'])} mm/s, a_max "
                        f"{fmt(b['max_acceleration_mm_s2'])} mm/s², ω_max {fmt(b['max_omega_rad_s'])} rad/s")
        for j in s.get("joints", []):
            rows.append(f"Joint {escape(str(j['joint']))}: max |F| <b>{fmt(j['max_force_N'])} N</b> (t = {fmt(j['t_max_force_s'], 4)} s), "
                        f"max |M| {fmt(j['max_moment_Nmm'])} N·mm")
        for d in s.get("drivers", []):
            rows.append(f"Driver {escape(str(d['driver']))}: max {fmt(d['max_abs'])} {escape(str(d['unit']))} (t = {fmt(d['t_max_s'], 4)} s), RMS {fmt(d['rms'])}, max power {fmt(d['max_power_W'])} W")
        eq = s.get("equilibrium") or {}
        if eq:
            rows.append(f"Body equilibrium: max. imbalance {fmt(eq.get('max_rel'), 3)}")
        pb = s.get("power_balance") or {}
        if pb:
            rows.append(f"Power balance: max. rel. {fmt(pb.get('max_rel'), 3)}")
        col = s.get("collisions_summary") or {}
        if col.get("first"):
            f = col["first"]
            rows.append(f"<span style='color:#d23b3b'><b>Collision</b> {f['a']} – {f['b']} at t = {fmt(f['t'], 4)} s</span>")
        elif col:
            rows.append("Collisions: none")
        for w in s.get("warnings", []):
            rows.append(f"<span style='color:#b36b00'>! {w}</span>")
        return "<br>".join(rows)

    def _items_for(self, kind) -> list[str]:
        res = self.win.motion_result
        if res is None:
            return []
        if kind == "body":
            return [b["name"] for b in res.meta.get("bodies", []) if not b.get("ground")]
        return {"joint": res.joint_names, "driver": res.driver_names, "force": res.force_names}.get(kind, [])

    def series_names(self) -> list[str]:
        """Names of the series for the maximum (transfer to the FEM): reactions and moments of the joints, drives, elements."""
        res = self.win.motion_result
        if res is None:
            return []
        out = [f"reaction:{j}" for j in res.joint_names] + [f"joint moment:{j}" for j in res.joint_names]
        out += [f"drive:{d}" for d in res.driver_names] + [f"element force:{f}" for f in res.force_names]
        return out

    def fill_items(self):
        res = self.win.motion_result
        if res is None:
            return
        from fea_core.motion.results import MotionResult
        q = self.g_qty.currentText()
        spec = MotionResult.SERIES.get(q)
        cur = self.g_item.currentText()
        self.g_item.blockSignals(True)
        self.g_item.clear()
        if spec is not None:
            self.g_item.addItems(self._items_for(spec[0]))
        elif q == "distance":
            self.g_item.addItems([f"{p['a']}|{p['b']}" for p in (res.meta.get("collisions") or {}).get("pairs", [])])
        if cur and self.g_item.findText(cur) >= 0:
            self.g_item.setCurrentText(cur)
        self.g_item.blockSignals(False)
        self.g_item.setEnabled(self.g_item.count() > 0)
        self.fill_comps()

    def series_name(self) -> str:
        q = self.g_qty.currentText()
        it = self.g_item.currentText()
        return f"{q}:{it}" if it else q

    def fill_comps(self):
        res = self.win.motion_result
        self.g_comp.blockSignals(True)
        self.g_comp.clear()
        try:
            s = res.series(self.series_name()) if res is not None else None
        except Exception:  # noqa: BLE001
            s = None
        if s is not None:
            self.g_comp.addItem("all components", None)
            if "magnitude" in s:
                self.g_comp.addItem("magnitude", "mag")
            for k, lab in enumerate(s["labels"]):
                self.g_comp.addItem(lab, k)
        self.g_comp.blockSignals(False)
        self.draw()

    def draw(self):
        res = self.win.motion_result
        if res is None or (not self.isVisible() and self.g_plot.canvas is None):
            return
        ax = self.g_plot.axes()
        try:
            s = res.series(self.series_name())
        except Exception as e:  # noqa: BLE001
            ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes, fontsize=8)
            self.g_plot.canvas.draw_idle()
            return
        c = self.g_comp.currentData()
        t = s["t"]
        if c == "mag":
            ax.plot(t, s["magnitude"], label="magnitude", lw=1.3)
        elif c is None:
            for k, lab in enumerate(s["labels"]):
                ax.plot(t, s["values"][:, k], label=lab, lw=1.2)
        else:
            ax.plot(t, s["values"][:, int(c)], label=s["labels"][int(c)], lw=1.3)
        k = self.a_slider.value()
        if 0 <= k < len(t):
            ax.axvline(float(t[k]), color="#888888", lw=0.8, ls=":")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(f"{self.g_qty.currentText()} [{s['unit']}]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        self.g_plot.canvas.draw_idle()

    def find_max(self):
        res = self.win.motion_result
        if res is None:
            return
        c = self.g_comp.currentData()
        try:
            mx = res.find_max(self.series_name(), int(c) if isinstance(c, int) else None)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP, f"Maximum cannot be found: {e}")
            return
        self.last_max = mx
        self.g_max.setText(f"Maximum {escape(str(self.series_name()))}: <b>{fmt(mx['value'])} {escape(str(mx['unit']))}</b> at t = {fmt(mx['t'], 5)} s")
        self.a_slider.setValue(mx["index"])

    # ------------------------------------------------------------------ animace
    def current_time(self) -> float:
        res = self.win.motion_result
        if res is None:
            return 0.0
        return float(res.arrays["t"][min(self.a_slider.value(), len(res.arrays["t"]) - 1)])

    def show_frame(self):
        res = self.win.motion_result
        if res is None:
            return
        k = self.a_slider.value()
        self.a_time.setText(f"t = {fmt(float(res.arrays['t'][k]), 4)} s")
        names = [p.name for p in self.win.parts]
        poses = {}
        for bi, b in enumerate(res.meta.get("bodies", [])):
            if b.get("ground"):
                continue
            R, T = res.pose(bi, k)
            for pn in list(b.get("parts", [])) + list(b.get("extra", [])):
                if pn in names:
                    poses[names.index(pn)] = (R, T)
        v = self.win.view
        if poses:
            v.show_sets(False)
        v.set_part_poses(poses)
        if self.g_plot.canvas is not None and not self.a_play.isChecked():
            self.draw()

    def play(self, on):
        self.a_play.setText("■ Stop" if on else "► Play")
        if on and self.win.motion_result is not None:
            self._timer.start()
        else:
            self._timer.stop()
            self.draw()

    def _tick(self):
        s = self.a_slider
        if s.maximum() <= 0:
            self.a_play.setChecked(False)
            return
        k = s.value() + int(self.a_speed.currentData() or 1)
        s.setValue(k if k <= s.maximum() else 0)

    def reset_pose(self):
        """Parts back to the CAD position (end of the animation, leaving the step)."""
        if getattr(self, "a_play", None) is not None and self.a_play.isChecked():
            self.a_play.setChecked(False)
        v = self.win.view
        if getattr(v, "_part_posed", False):
            v.set_part_poses(None)

    # ------------------------------------------------------------------ transfer to the FEM
    def to_fea(self):
        res = self.win.motion_result
        if res is None or self.study is None:
            return
        dlg = ToFeaDialog(self)
        self.fea_dialog = dlg                            # for tests
        ok = dlg.exec() == QDialog.Accepted
        self.fea_dialog = None
        if ok:
            self.export(dlg.options())

    def export(self, opt: dict) -> dict | None:
        from fea_core.motion.to_fea import export_loads
        st = self.study
        cfg = st.mechanism.run.to_fea
        cfg.mode, cfg.times, cfg.quantity = opt["mode"], opt["times"], opt["quantity"]
        cfg.support_joints, cfg.load_kind = list(opt["support_joints"] or []), opt["load_kind"]
        try:
            out = export_loads(st, self.win.motion_result, opt["mode"], opt["times"] or None, opt["quantity"],
                               opt["body"], None, opt["support_joints"], opt["load_kind"])
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP, f"Transfer to FEA is not possible:\n{e}")
            return None
        self.win._motion_sig = mech_signature(st)          # changing the transfer settings does not invalidate the motion result
        self._exports = list(out.get("cases", []))
        notes = [n for c in out.get("cases", []) for n in c.get("notes", [])] + list(out.get("notes", []))
        lcs = out.get("load_cases", [])
        self.fea_info.setText(f"Load cases created: <b>{len(lcs)}</b> ({escape(', '.join(lcs[:6]))}"
                              + ("…" if len(lcs) > 6 else "") + "). Support = the remote support of the joint (Supports step)."
                              + "".join(f"<br><small>{escape(n)}</small>" for n in notes))
        self.win.log_msg(f"Motion → FEM: {len(lcs)} load cases" + (" + fatigue channels" if opt["mode"] == "history"
                                                                         else ""))
        self.win.resolve_all()
        self.win.changed()
        return out

    def check_relief(self):
        res = self.win.results
        if res is None or not self._exports:
            QMessageBox.information(self, APP, "First transfer the motion to FEA and run the analysis (Solve step).")
            return
        from fea_core.motion.to_fea import check_relief
        rows = []
        for info in self._exports:
            try:
                step = res.step(info["load_case"])
            except Exception:  # noqa: BLE001
                continue
            r = check_relief(info, step)
            ok = r["max_rel"] < 1e-2
            rows.append(f"{info['load_case']}: difference of the FEM × MBD support reaction <b>{fmt(100 * r['max_rel'], 3)} %</b> "
                        + ("<span style='color:#2e9e4f'>OK</span>" if ok else
                           "<span style='color:#d23b3b'>does not match – check the masses and joint faces</span>"))
        self.fea_info.setText("<br>".join(rows) or "The FEA results do not contain the transferred cases.")

    # ================================================================== 3D
    def show_3d(self):
        st = self.study
        if st is None:
            return
        win, v = self.win, self.win.view
        L = 0.25 * v.model_size()
        names = [p.name for p in win.parts]
        r = self.b_table.currentRow()
        if 0 <= r < len(self.mech.bodies) and self.tabs.currentIndex() == 0:
            b = self.mech.bodies[r]
            fl = [(names.index(pn), fi) for pn in b.parts if pn in names
                  for fi in range(len(v.part_faces[names.index(pn)]))]
            if fl:
                v.set_face_set("mb:body", fl, (230, 140, 20), label=b.name, transparency=0.45)
        jr = self.j_table.currentRow()
        for k, j in enumerate(self.mech.joints):
            if j.kind in ("gear", "rack_pinion"):
                continue
            col = JOINT_COLORS.get(j.kind, (40, 160, 160))
            o, a = np.asarray(j.origin, float), np.asarray(j.axis, float)
            a = a / max(np.linalg.norm(a), 1e-30)
            seg = [(o - L * a, o + L * a)]
            if j.kind == "distance" and j.point_a is not None and j.point_b is not None:
                seg = [(np.asarray(j.point_a, float), np.asarray(j.point_b, float))]
            v.add_lines(f"mj:{k}", seg, col, 4.0 if k == jr else 2.5, label=j.name)
            if k == jr:
                for s in (j.selection_a, j.selection_b):
                    if s:
                        v.set_face_set(f"mj:{k}:{s}", win.sel_faces.get(s, []), col, transparency=0.3)
        for k, e in enumerate(self.mech.forces):
            if e.kind != "torsion":
                v.add_lines(f"mf:{k}", [(np.asarray(e.point_a, float), np.asarray(e.point_b, float))],
                            (20, 150, 80), 2.0, label=e.name)
        for k, ld in enumerate(self.mech.loads):
            if np.linalg.norm(ld.force) > 0:
                v.add_arrows(f"ml:{k}", [(ld.point, ld.force, 0.5 * L)], (220, 40, 40))
