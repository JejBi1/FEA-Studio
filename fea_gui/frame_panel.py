"""Step "Frames": editor of the frame system (Study.frame – fea_core.beams.model): nodes, members (straight and curved,
hinges, truss), sections (EN library, parametric shapes, own one from a planar CAD face), supports and springs,
connection to a face of a body, EN 1993-1-1 assessment; taking over members from a STEP wire model and replacing a part with a member.
Drawing of the system in 3D (persistent objects of FeaView) and selection of a node / member by click (FeaView.screen_pick).
"""
from __future__ import annotations

import json
import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QSpinBox,
                               QSizePolicy, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from fea_core.beams.model import (DOF_NAMES, LOCAL_NAMES, FrameConnection, FrameMember, FrameNode, FrameSupport,
                                  SectionDef)
from fea_core.i18n import tr

from . import theme
from .widgets import LazyPlot, dspin, fit_tabs, fmt, panel_header, scroll_body, table, Section, help_tip

C_MEMBER = (45, 70, 110)
C_TRUSS = (95, 130, 170)
C_SEL = (255, 140, 0)
C_NODE = (25, 25, 25)
C_SUPPORT = (40, 110, 230)
C_RELEASE = (200, 40, 40)
C_CONN = (190, 60, 200)
C_FLOAD = (220, 40, 40)
KEY = "frame"
GRADES = ("S235", "S275", "S355", "S420", "S460")
# parametric shapes (fea_core.beams.sections.parametric): dimensions and labels
PARAM = {"RECT": ("rectangle", ("b", "h")), "CIRC": ("circle (solid bar)", ("d",)), "TUBE": ("tube", ("D", "t")),
         "RHS": ("RHS (rectangular hollow section)", ("h", "b", "t")), "I": ("Welded I", ("h", "b", "tw", "tf")),
         "T": ("T", ("h", "b", "tw", "tf")), "U": ("Sheet-metal U", ("h", "b", "t")), "L": ("L angle", ("h", "b", "t"))}
DIM_TIPS = {"b": "width b [mm]", "h": "height h [mm]", "d": "diameter d [mm]", "D": "outer diameter D [mm]",
            "t": "wall thickness t [mm]", "tw": "web thickness t_w [mm]", "tf": "flange thickness t_f [mm]"}
PROPS_ROWS = [("A", "area A", "mm²"), ("Iy", "I_y (strong axis)", "mm⁴"), ("Iz", "I_z (weak axis)", "mm⁴"),
              ("J", "I_t (torsion)", "mm⁴"), ("Iw", "I_w (warping)", "mm⁶"), ("Wel_y", "W_el,y", "mm³"),
              ("Wel_z", "W_el,z", "mm³"), ("Wpl_y", "W_pl,y", "mm³"), ("Wpl_z", "W_pl,z", "mm³"),
              ("Av_y", "A_v,y (shear)", "mm²"), ("Av_z", "A_v,z (shear)", "mm²")]


def frame_signature(study) -> str:
    return json.dumps(study.frame.to_dict(), sort_keys=True, default=str) + "|" + json.dumps(
        {k: [m.E, m.nu, m.rho] for k, m in study.materials.items()}) + f"|{study.default_material}"


def member_polyline(study, m, n_arc: int = 24) -> list[np.ndarray]:
    """Points of the member axis (line or arc through `through`) – drawing without FEM discretisation."""
    fr = study.frame
    a, b = np.asarray(fr.nodes[m.start].xyz, float), np.asarray(fr.nodes[m.end].xyz, float)
    if m.through is None:
        return [a, b]
    try:
        from fea_core.beams.fem import _arc
        cen, r, n, e1, e2, th = _arc(a, m.through, b)
        return [cen + r * (math.cos(t) * e1 + math.sin(t) * e2) for t in np.linspace(0, th, n_arc + 1)]
    except Exception:  # noqa: BLE001 – points on a line → segments through the arc point
        return [a, np.asarray(m.through, float), b]


def member_axes(study, m) -> np.ndarray | None:
    """Local axes of the member (rows x, y, z) at half the length (straight member) – for the local load arrows."""
    try:
        from fea_core.beams.fem import _axes
        P = member_polyline(study, m)
        k = len(P) // 2
        sec = study.frame.sections[m.section].resolve()
        return _axes(P[min(k + 1, len(P) - 1)] - P[max(k - 1, 0)], m.up, m.roll, sec.alpha)
    except Exception:  # noqa: BLE001
        return None


def _num_list(text: str) -> list[float] | None:
    t = text.replace(";", " ").replace(",", ".").split()
    try:
        return [float(x) for x in t]
    except ValueError:
        return None


class FramePanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self._filling = False
        self._sel = ("", None)                   # ("node" | "member", name) – highlight in 3D
        self._sec_preview = None                 # SectionProps of the preview (library / shape / area)
        self.scroll, lay = scroll_body(self)
        lay.addWidget(panel_header("Members", "Frame: nodes, members (beams, trusses, arcs), cross-sections, supports; EN 1993-1-1 check"))
        h = QVBoxLayout()                       # buttons one below the other – a long text would not fit side by side
        b = QPushButton("Members from STEP wireframe…")
        help_tip(b, "Reads a wireframe model (free edges) from STEP and creates nodes and members from it",
                 "Loads free edges (arcs, curves outside solids) from a STEP file and creates nodes<br>and members with the section chosen on the Sections tab (end points within 0.001 mm are merged)")
        b.clicked.connect(self.from_wires)
        h.addWidget(b)
        b = QPushButton("Replace part with member")
        help_tip(b, "Replaces a part (bar, profile) selected in 3D with a member using the section of its end face",
                 "Part whose face you selected in 3D (prismatic body – bar, profile) replaced by a member:<br>section from the end face (library profile if it matches), nodes at the centroids of the end faces, connections to touching<br>faces of other parts. The part is left out of the mesh.")
        b.clicked.connect(self.replace_part)
        self.btn_replace = b
        h.addWidget(b)
        lay.addLayout(h)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.tabs = QTabWidget()
        help_tip(self.tabs, "Nodes, members, sections, supports and connections; selecting in the table and in 3D is linked",
                 "Frame parts – clicking a table row highlights the node / member in 3D, clicking in 3D selects the row")
        lay.addWidget(self.tabs)
        self._tab_nodes()
        self._tab_members()
        self._tab_sections()
        self._tab_supports()
        self._tab_connections()
        self._tab_checks()
        self.tabs.currentChanged.connect(lambda _: self.show_3d())
        lay.addStretch(1)

    # ================================================================== stavba karet
    def _buttons(self, form, specs):
        h = QHBoxLayout()
        for txt, tip, slot in specs:
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            h.addWidget(b)
        form.addRow(h) if isinstance(form, QFormLayout) else form.addLayout(h)

    @staticmethod
    def _section(v, title, key, tip, collapsed=False, form=False):
        """Section of a tab (collapsible, can be opened in a separate window); returns its content layout."""
        g = Section(title, key=f"frame.{key}", tip=tip, collapsed=collapsed)
        lay = QFormLayout(g.body) if form else QVBoxLayout(g.body)
        lay.setContentsMargins(0, 0, 0, 0)
        v.addWidget(g)
        return lay

    def _tab_nodes(self):
        w = QWidget()
        v = QVBoxLayout(w)
        tv = self._section(v, "Node list", "nodes", "Nodes in global coordinates – click a row = load into the form")
        self.t_nodes = table(["Node", "x [mm]", "y [mm]", "z [mm]"],
                             stretch_last=True,
                             placeholder="No nodes yet – enter a position below or load a wireframe model",
                             rows=(3, 10))
        help_tip(self.t_nodes, "Nodes in global coordinates – click = load into the form",
                 "Frame nodes (global coordinates). Click = load into the form and highlight in 3D; clicking a node in 3D selects the row.")
        self.t_nodes.currentCellChanged.connect(lambda r, *_: self.on_node_row(r))
        tv.addWidget(self.t_nodes)
        f = self._section(v, "Node", "node_form", "Name and position of a new or the selected node", form=True)
        self.n_name = QLineEdit()
        self.n_name.setToolTip("Node name (unique; referenced by members, supports and loads)")
        f.addRow("Name", self.n_name)
        h = QHBoxLayout()
        self.n_xyz = [dspin(0.0, -1e8, 1e8, 3) for _ in range(3)]
        for s, ax in zip(self.n_xyz, "xyz"):
            s.setToolTip(f"Node coordinate {ax} [mm]")
            h.addWidget(s)
        self.win.register_point_input(self.n_xyz)
        f.addRow("Position [mm]", h)
        self._buttons(f, [("Add node", "New node with the name and position from the form", self.add_node),
                          ("Save changes", "Overwrite the position (and name) of the node selected in the table; members and supports stay attached", self.update_node),
                          ("Delete", "Delete the node selected in the table (unused only – delete its members first)",
                           self.delete_node)])
        v.addStretch(1)
        self.tabs.addTab(w, "Nodes")

    def _tab_members(self):
        w = QWidget()
        v = QVBoxLayout(w)
        tv = self._section(v, "Member list", "members", "Members of the frame – click a row = load into the form")
        self.t_members = table(["Member", "Start", "End", "Cross-section", "Material", "Type", "L [mm]"],
                               stretch_last=True,
                               placeholder="No members yet – first enter nodes and a section, then a member below",
                               rows=(3, 10))
        help_tip(self.t_members, "Members: nodes, section, material, type and length – click = load into the form",
                 "Members: start and end nodes, section, material, type (beam / truss member),<br>length. Click = load into the form; clicking a member in 3D selects its row.")
        self.t_members.currentCellChanged.connect(lambda r, *_: self.on_member_row(r))
        tv.addWidget(self.t_members)
        f = self._section(v, "Member", "member_form", "Name, end nodes, cross-section, material and type", form=True)
        self.m_name = QLineEdit()
        self.m_name.setToolTip("Member name (unique; internal forces and the check are listed under it)")
        f.addRow("Name", self.m_name)
        h = QHBoxLayout()
        self.m_start, self.m_end = QComboBox(), QComboBox()
        self.m_start.setToolTip("Member start node – the member local x axis runs from start to end")
        self.m_end.setToolTip("Member end node")
        h.addWidget(self.m_start)
        h.addWidget(QLabel("→"))
        h.addWidget(self.m_end)
        f.addRow("Nodes", h)
        self.m_section = QComboBox()
        self.m_section.setToolTip("Member cross-section (Cross-sections tab)")
        f.addRow("Cross-section", self.m_section)
        self.m_material = QComboBox()
        help_tip(self.m_material, "Material of the member from the study; “default” = the study's default material",
                 "Material of the member from the study (Materials step); “default” = the study's default material.<br>For the EN 1993 check steel is taken from the name (S235…S460) or fy = Re.")
        f.addRow("Material", self.m_material)
        self.m_kind = QComboBox()
        self.m_kind.addItem("beam (bending, torsion)", "beam")
        self.m_kind.addItem("truss member (axial force only)", "truss")
        help_tip(self.m_kind, "A beam carries N, V, T, M; a truss member carries axial force only",
                 "A beam carries N, V, T, M; a truss member carries axial force only (hinges at both ends)")
        f.addRow("Type", self.m_kind)
        f = self._section(v, "Orientation, arc and elements", "member_geom",
                          "Cross-section z axis, rotation, curved member and number of elements", collapsed=True, form=True)
        h = QHBoxLayout()
        self.m_up = QComboBox()
        for txt, key in (("automatic (global Z, vertical member X)", "auto"), ("vector", "vec"),
                         ("arc plane normal", "normal"), ("toward arc center", "radial")):
            self.m_up.addItem(txt, key)
        help_tip(self.m_up, "Orientation of the section's z axis (web of an I-profile “up”) – projection perpendicular to the member axis",
                 "Orientation of the section's z axis (web of an I-profile “up”): projection of the vector perpendicular to the member axis.<br>For an arc, the normal of the arc plane or the direction to the arc centre (section depth in the plane of curvature).")
        self.m_up.currentIndexChanged.connect(lambda _: self._sync_member_form())
        h.addWidget(self.m_up, 1)
        self.m_upv = [dspin(v, -1, 1, 3, 0.1) for v in (0, 0, 1)]
        for s, ax in zip(self.m_upv, "xyz"):
            s.setToolTip(f"Component {ax} of the “up” vector (cross-section z axis)")
            h.addWidget(s)
        f.addRow("Cross-section z axis", h)
        self.m_roll = dspin(0.0, -360, 360, 2, 15.0, " °")
        self.m_roll.setToolTip("Cross-section rotation about the member axis [°] (positive counterclockwise about x)")
        f.addRow("Rotation", self.m_roll)
        h = QHBoxLayout()
        self.m_arc = QCheckBox("arc through point")
        self.m_arc.setToolTip("Curved member: circular arc from the start through the given point to the end")
        self.m_arc.toggled.connect(lambda _: self._sync_member_form())
        h.addWidget(self.m_arc)
        self.m_thr = [dspin(0.0, -1e8, 1e8, 3) for _ in range(3)]
        for s, ax in zip(self.m_thr, "xyz"):
            s.setToolTip(f"Coordinate {ax} of the point on the arc [mm] (between start and end)")
            h.addWidget(s)
        self.win.register_point_input(self.m_thr)
        f.addRow("Arc", h)
        self.m_segs = QSpinBox()
        self.m_segs.setRange(0, 500)
        self.m_segs.setSpecialValueText("automatic")
        help_tip(self.m_segs, "Number of U1 elements per member (default: 20 for a beam, 1 for a truss member)",
                 "Number of U1 elements per member (automatic = 20 per straight beam, 1 per truss member, arc every 5°)")
        f.addRow("Elements", self.m_segs)
        f = self._section(v, "Hinges (end releases)", "member_rel", "Released components at the member ends in local axes",
                          collapsed=True, form=True)
        g = QGridLayout()
        g.addWidget(QLabel("hinges"), 0, 0)
        self.m_rel = {"start": [], "end": []}
        for c, n in enumerate(LOCAL_NAMES):
            g.addWidget(QLabel(n), 0, c + 1)
        for r, end in enumerate(("start", "end")):
            g.addWidget(QLabel("start" if end == "start" else "end"), r + 1, 0)
            for c, n in enumerate(LOCAL_NAMES):
                cb = QCheckBox()
                cb.setToolTip(f"Release {n} at the {('start' if end == 'start' else 'end')} of the member (hinge in member local axes; My + Mz = bending hinge)")
                g.addWidget(cb, r + 1, c + 1)
                self.m_rel[end].append(cb)
        f.addRow(g)
        h = QHBoxLayout()
        for txt, tip, rel in (("hinge at start", "Release My and Mz at the start (hinged connection)", ("start",)),
                              ("hinges at both ends", "Release My and Mz at both ends (pendulum member, purlin)",
                               ("start", "end")),
                              ("no hinges", "Clear all releases", ())):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, rel=rel: self.set_hinges(rel))
            h.addWidget(b)
        f.addRow(h)
        self._buttons(v, [("Add member", "New member from the form", self.add_member),
                          ("Save changes", "Overwrite the member selected in the table", self.update_member),
                          ("Delete", "Delete the member selected in the table together with its loads and check", self.delete_member)])
        v.addStretch(1)
        self.tabs.addTab(w, "Members")

    def _tab_sections(self):
        w = QWidget()
        v = QVBoxLayout(w)
        tv = self._section(v, "Cross-sections of the study", "sections", "Cross-sections used by members – click = preview")
        self.t_sections = table(["Cross-section", "Input", "A [mm²]", "Iy [mm⁴]", "Iz [mm⁴]"],
                                stretch_last=True,
                                placeholder="No sections yet – choose a profile in the New section group",
                                rows=(3, 10))
        self.t_sections.setToolTip("Study cross-sections. Click = outline preview and properties.")
        self.t_sections.currentCellChanged.connect(lambda r, *_: self.on_section_row(r))
        tv.addWidget(self.t_sections)
        g = Section("New cross-section",
                    key="frame.new_section",
                    collapsed=False,
                    tip="EN library cross-section, parametric shape, or custom from a planar CAD face")
        f = QFormLayout(g.body)
        self.s_src = QComboBox()
        for txt, key in (("library (EN 10365, EN 10056, EN 10219/10210)", "lib"), ("parametric shape", "param"),
                         ("from planar CAD face (selected in 3D)", "face")):
            self.s_src.addItem(txt, key)
        help_tip(self.s_src, "Section source: library profile, parametric shape, or CAD face",
                 "Section source: rolled profile from the library (dimensions only, properties by formulas), parametric<br>shape from dimensions, or a custom section from a planar face of a part (FE section analysis – I_t, I_w)")
        self.s_src.currentIndexChanged.connect(lambda _: self._sync_section_form())
        f.addRow("Source", self.s_src)
        self.s_family = QComboBox()
        self.s_family.setToolTip("Library section series (IPE, HEA, HEB, UPE, L, CHS, SHS, RHS …)")
        self.s_family.currentIndexChanged.connect(lambda _: self.on_family())
        self.s_lbl_family = QLabel("Series")
        f.addRow(self.s_lbl_family, self.s_family)
        self.s_find = QLineEdit()
        self.s_find.setPlaceholderText("search: IPE 200, HEB200, L50x5, CHS 60.3x4 …")
        self.s_find.setClearButtonEnabled(True)
        self.s_find.setToolTip("Search the library for a profile by name (spaces, commas and × are ignored)")
        self.s_find.textChanged.connect(self.on_find)
        self.s_lbl_find = QLabel("Search")
        f.addRow(self.s_lbl_find, self.s_find)
        self.s_list = QListWidget()
        self.s_list.setMaximumHeight(110)
        self.s_list.setToolTip("Profiles of the series / search results – click = preview and properties")
        self.s_list.currentTextChanged.connect(lambda _: self.preview_section())
        self.s_lbl_list = QLabel("Profile")
        f.addRow(self.s_lbl_list, self.s_list)
        self.s_shape = QComboBox()
        for k, (txt, _) in PARAM.items():
            self.s_shape.addItem(txt, k)
        self.s_shape.setToolTip("Parametric cross-section shape")
        self.s_shape.currentIndexChanged.connect(lambda _: self._sync_section_form())
        self.s_lbl_shape = QLabel("Shape")
        f.addRow(self.s_lbl_shape, self.s_shape)
        self.s_dims = {}
        hd = QGridLayout()
        for i, (k, val) in enumerate((("b", 50.0), ("h", 100.0), ("d", 30.0), ("D", 60.0), ("t", 4.0), ("tw", 6.0),
                                      ("tf", 10.0))):
            lb = QLabel(k)
            sp = dspin(val, 0.01, 1e5, 2, 1.0)
            sp.setToolTip(DIM_TIPS[k])
            sp.valueChanged.connect(lambda _: self.preview_section())
            hd.addWidget(lb, i // 4, 2 * (i % 4))
            hd.addWidget(sp, i // 4, 2 * (i % 4) + 1)
            self.s_dims[k] = (lb, sp)
        self.s_lbl_dims = QLabel("Dimensions [mm]")
        f.addRow(self.s_lbl_dims, hd)
        self.s_name = QLineEdit()
        self.s_name.setToolTip("Cross-section name in the study (empty = profile / shape name)")
        f.addRow("Name in study", self.s_name)
        self._buttons(f, [("Preview", "Calculate properties and draw the outline (for a CAD face, an FE cross-section analysis, ~3 s)",
                           self.preview_section),
                          ("Add cross-section", "Add the cross-section from the form to the study", self.add_section),
                          ("Delete", "Delete the cross-section selected in the table (unused only)", self.delete_section)])
        v.addWidget(g)
        pv = self._section(v, "Preview and properties", "section_preview", "Outline of the cross-section and its properties")
        h = QHBoxLayout()
        self.s_plot = LazyPlot()
        self.s_plot.setMinimumHeight(200)
        help_tip(self.s_plot, "Section outline in the principal axes, origin at the centroid",
                 "Cross-section outline in principal axes (y horizontal = strong axis, z vertical), origin at the centroid")
        h.addWidget(self.s_plot, 1)
        self.s_props = table(["Property", "Value"],
                             stretch_last=True,
                             placeholder="Choose a profile or calculate a section preview",
                             rows=(3, 10))
        help_tip(self.s_props, "Section properties: area, moments, I_t, I_w, section moduli, shear areas",
                 "Cross-section properties: area, moments of inertia about principal axes, I_t, I_w, section moduli and shear areas (EN 1993-1-1)")
        h.addWidget(self.s_props, 1)
        pv.addLayout(h)
        self.s_note = QLabel("")
        self.s_note.setWordWrap(True)
        self.s_note.setProperty("muted", True)
        pv.addWidget(self.s_note)
        v.addStretch(1)
        self.tabs.addTab(w, "Cross-sections")

    def _tab_supports(self):
        w = QWidget()
        v = QVBoxLayout(w)
        tv = self._section(v, "Support list", "supports", "Node supports and springs – click a row = load into the form")
        self.t_sup = table(["Node", "Constrained components", "Springs"],
                           stretch_last=True,
                           placeholder="No supports yet – choose a node and the restrained components below",
                           rows=(3, 10))
        self.t_sup.setToolTip("Node supports: constrained displacement and rotation components (global axes) and springs")
        self.t_sup.currentCellChanged.connect(lambda r, *_: self.on_support_row(r))
        tv.addWidget(self.t_sup)
        f = self._section(v, "Support", "support_form", "Node, constrained components and springs", form=True)
        self.u_node = QComboBox()
        self.u_node.setToolTip("Node with support")
        f.addRow("Node", self.u_node)
        g = QGridLayout()
        self.u_dof, self.u_k = [], []
        for c, n in enumerate(DOF_NAMES):
            cb = QCheckBox(n)
            cb.setChecked(True)
            cb.setToolTip(("displacement" if c < 3 else "rotation") + f" {n} constrained (zero)")
            k = dspin(0.0, 0.0, 1e15, 3, 100.0, special="–")
            k.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)     # width by the minimum, not by 1e15 in the text
            help_tip(k, "Spring instead of a support: stiffness " + ("[N/mm]" if c < 3 else "[N·mm/rad]") + f" in component {n}",
                     "Spring stiffness in the component: value > 0 = elastic support even without the constraint ticked; “–” = no spring.")
            g.addWidget(cb, 0, c)
            g.addWidget(k, 1, c)
            self.u_dof.append(cb)
            self.u_k.append(k)
        f.addRow(g)
        h = QHBoxLayout()
        for txt, tip, dofs in (("fixed", "All 6 components constrained", [True] * 6),
                               ("hinge", "Displacements constrained, rotations free", [True] * 3 + [False] * 3),
                               ("sliding hinge (uz)", "Only vertical displacement uz constrained", [False, False, True] + [False] * 3)):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, d=dofs: [cb.setChecked(x) for cb, x in zip(self.u_dof, d)])
            h.addWidget(b)
        f.addRow(h)
        self._buttons(f, [("Add / save support", "Node support from the form (an existing support of the node is overwritten)",
                           self.save_support),
                          ("Delete", "Delete the support selected in the table", self.delete_support)])
        v.addStretch(1)
        self.tabs.addTab(w, "Support")

    def _tab_connections(self):
        w = QWidget()
        v = QVBoxLayout(w)
        lb = QLabel("Member node attached to a body face by a rigid coupling (like a bolt): select the part faces and the node in 3D.")
        lb.setWordWrap(True)
        v.addWidget(lb)
        self.t_conn = table(["Node", "Faces (selection)", "Constraint"],
                            stretch_last=True,
                            placeholder="No connections yet – select part faces in 3D and a node below",
                            rows=(3, 10))
        self.t_conn.setToolTip("Connection of member nodes to body faces (rigid *KINEMATIC coupling to an auxiliary node)")
        self.t_conn.currentCellChanged.connect(lambda r, *_: self.on_conn_row(r))
        tv = self._section(v, "Connection list", "connections", "Member nodes connected to body faces")
        tv.addWidget(self.t_conn)
        f = self._section(v, "Connection", "conn_form", "Node connected to the faces selected in 3D", form=True)
        self.c_node = QComboBox()
        self.c_node.setToolTip("Member node to be connected to the selected body faces")
        f.addRow("Node", self.c_node)
        self._buttons(f, [("Connect to selected faces", "New connection: node ↔ faces selected in 3D (Face filter)",
                           self.add_connection),
                          ("Delete", "Delete the connection selected in the table", self.delete_connection)])
        v.addStretch(1)
        self.tabs.addTab(w, "Connection to body")

    def _tab_checks(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = Section("Common settings",
                    key="frame.common",
                    collapsed=True,
                    tip="Steel and partial factors for the EN 1993-1-1 check, member subdivision into elements")
        f = QFormLayout(g.body)
        self.k_grade = QComboBox()
        self.k_grade.addItems(GRADES)
        help_tip(self.k_grade, "Steel for the check if it cannot be determined from the material name (S235…S460)",
                 "Steel for the check if it cannot be determined from the member's material name (S235…S460);<br>fy by thickness (EN 1993-1-1, tab. 3.1). A material with Re and no grade → fy = Re.")
        f.addRow("Steel (default)", self.k_grade)
        self.k_g0 = dspin(1.0, 1.0, 2.0, 2, 0.05)
        self.k_g0.setToolTip("γ_M0 – cross-section resistance (EN 1993-1-1, 6.1; Czech NA 1.00)")
        self.k_g1 = dspin(1.0, 1.0, 2.0, 2, 0.05)
        self.k_g1.setToolTip("γ_M1 – member stability, buckling and lateral-torsional buckling (Czech NA 1.00)")
        f.addRow("γ_M0", self.k_g0)
        f.addRow("γ_M1", self.k_g1)
        self.k_solver = QComboBox()
        self.k_solver.addItem("Built-in solver (exact Timoshenko element)", "timoshenko")
        self.k_solver.addItem("CalculiX (U1 element) – comparison", "ccx_u1")
        help_tip(self.k_solver, "Own solver (exact Timoshenko) or CalculiX U1 for comparison",
                 "Beam structure alone: the built-in solver (default) has the exact Timoshenko matrix, full-precision internal forces, buckling, natural frequencies and temperature. CalculiX U1 is for comparison only. A mixed model with solids always uses CalculiX (U1).")
        f.addRow("Beam solver", self.k_solver)
        self.k_segs = QSpinBox()
        self.k_segs.setRange(1, 200)
        help_tip(self.k_segs, "CalculiX U1 only: elements per straight beam (default 20)",
                 "CalculiX U1 only: elements per straight beam (default 20 – the shear deformation of U1 converges ~9 %/n²). The built-in solver is exact even with 1 element per member.")
        f.addRow("Elements per member (CalculiX U1)", self.k_segs)
        self.k_arc = dspin(5.0, 0.5, 45.0, 1, 1.0, " °")
        self.k_arc.setToolTip("Maximum angle of one arc element [°]")
        f.addRow("Arc element max.", self.k_arc)
        self.k_shear = QCheckBox("shear deformation (Timoshenko)")
        self.k_shear.setToolTip("On = Timoshenko member (κ = A_v,z / A); off = Euler–Bernoulli")
        f.addRow(self.k_shear)
        for wdg in (self.k_g0, self.k_g1, self.k_arc):
            wdg.editingFinished.connect(self.save_settings)
        self.k_segs.editingFinished.connect(self.save_settings)
        self.k_solver.currentIndexChanged.connect(lambda _: self.save_settings())
        self.k_grade.currentIndexChanged.connect(lambda _: self.save_settings())
        self.k_shear.toggled.connect(lambda _: self.save_settings())
        v.addWidget(g)
        g = Section("Member check (effective lengths, lateral-torsional buckling)",
                    key="frame.check",
                    collapsed=False,
                    tip="Buckling lengths and lateral-torsional buckling of the selected member; empty field = default",
                    help="EN 1993-1-1 check inputs for the selected member (Study.frame.checks); empty = default")
        f = QFormLayout(g.body)
        self.k_member = QComboBox()
        self.k_member.setToolTip("Member whose check inputs are being edited")
        self.k_member.currentIndexChanged.connect(lambda _: self.load_check())
        f.addRow("Member", self.k_member)
        self.k_vals = {}
        for key, lbl, tip in (("Lcr_y", "L_cr,y [mm]", "Buckling length for buckling perpendicular to the y axis (bending about y)"),
                              ("Lcr_z", "L_cr,z [mm]", "Buckling length for buckling perpendicular to the z axis"),
                              ("L_LT", "L_LT [mm]", "Length between lateral-torsional buckling restraints"),
                              ("C1", "C1", "Moment diagram factor for M_cr (empty = from the My diagram)"),
                              ("zg", "z_g [mm]", "Load application point above the shear center (+ = destabilizing, on the top flange)")):
            sp = dspin(0.0, -1e7, 1e8, 3, 100.0, special="automatic (member length)" if key != "zg" else "0")
            if key == "C1":
                sp.setSpecialValueText("from My diagram")
            sp.setToolTip(tip)
            f.addRow(lbl, sp)
            self.k_vals[key] = sp
        self.k_lt = QCheckBox("continuously restrained against lateral-torsional buckling")
        self.k_lt.setToolTip("Compression flange continuously restrained (roof cladding) – lateral-torsional buckling is not checked")
        self.k_sway_y = QCheckBox("sway joints (y)")
        self.k_sway_y.setToolTip("Sway buckling about y → C_my = 0.9 (Annex B, Table B.3)")
        self.k_sway_z = QCheckBox("sway joints (z)")
        self.k_sway_z.setToolTip("Sway buckling about z → C_mz = 0.9")
        h = QHBoxLayout()
        for c in (self.k_lt, self.k_sway_y, self.k_sway_z):
            h.addWidget(c)
        f.addRow(h)
        self.k_fab = QComboBox()
        for txt, key in (("automatic", "auto"), ("rolled", "rolled"), ("welded", "welded"),
                         ("hot-finished (tubes)", "hot_finished"), ("cold-formed", "cold_formed")):
            self.k_fab.addItem(txt, key)
        self.k_fab.setToolTip("Cross-section fabrication – buckling curve selection (EN 1993-1-1, Table 6.2)")
        f.addRow("Fabrication", self.k_fab)
        self._buttons(f, [("Save member inputs", "Save the check inputs of the selected member", self.save_check),
                          ("Default", "Delete custom member inputs (effective lengths = member length)", self.reset_check)])
        v.addWidget(g)
        v.addStretch(1)
        self.tabs.addTab(w, "EN 1993 check")
        fit_tabs(self.tabs)

    # ================================================================== obnova
    @property
    def frame(self):
        return self.win.study.frame if self.win.study is not None else None

    def refresh(self):
        st = self.win.study
        if st is None:
            return
        fr = st.frame
        self._filling = True
        try:
            def fill(t, rows, keep=None):
                cur = t.item(t.currentRow(), 0).text() if t.currentRow() >= 0 and t.item(t.currentRow(), 0) else keep
                t.blockSignals(True)
                t.setRowCount(len(rows))
                for r, row in enumerate(rows):
                    for c, txt in enumerate(row):
                        it = QTableWidgetItem(str(txt))
                        if c and isinstance(txt, str) and txt[:1].lstrip("-").isdigit():
                            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        t.setItem(r, c, it)
                k = next((r for r, row in enumerate(rows) if str(row[0]) == cur), -1)
                if k >= 0:
                    t.setCurrentCell(k, 0)
                t.blockSignals(False)
            fill(self.t_nodes, [(n.name, *(fmt(float(c), 6) for c in n.xyz)) for n in fr.nodes.values()])
            rows = []
            for m in fr.members.values():
                P = member_polyline(st, m) if m.start in fr.nodes and m.end in fr.nodes else []
                L = sum(float(np.linalg.norm(b - a)) for a, b in zip(P, P[1:])) if P else 0.0
                rows.append((m.name, m.start, m.end, m.section, m.material or "default",
                             "truss" if m.kind == "truss" else ("arc" if m.through is not None else "beam"),
                             fmt(L, 5)))
            fill(self.t_members, rows)
            srows = []
            for s in fr.sections.values():
                try:
                    p = s.resolve()
                    srows.append((s.name, s.library or (f"mode {s.shape}" if s.shape else "custom"), fmt(p.A, 5),
                                  fmt(p.Iy, 5), fmt(p.Iz, 5)))
                except Exception as e:  # noqa: BLE001
                    srows.append((s.name, f"error: {e}", "–", "–", "–"))
            fill(self.t_sections, srows)
            fill(self.t_sup, [(s.node, " ".join(n for n, on in zip(DOF_NAMES, s.dofs) if on) or "–",
                               ", ".join(f"{n} {fmt(k, 4)}" for n, k in zip(DOF_NAMES, s.stiffness) if k) or "–")
                              for s in fr.supports])
            fill(self.t_conn, [(c.node, c.selection, "rigid" if c.coupling == "rigid" else c.coupling)
                               for c in fr.connections])
            nodes = list(fr.nodes)
            for cb in (self.m_start, self.m_end, self.u_node, self.c_node):
                cur = cb.currentText()
                cb.blockSignals(True)
                cb.clear()
                cb.addItems(nodes)
                if cur in nodes:
                    cb.setCurrentText(cur)
                cb.blockSignals(False)
            cur = self.m_section.currentText()
            self.m_section.clear()
            self.m_section.addItems(list(fr.sections))
            if cur in fr.sections:
                self.m_section.setCurrentText(cur)
            cur = self.m_material.currentData()
            self.m_material.clear()
            self.m_material.addItem("study default material", None)
            for k in st.materials:
                self.m_material.addItem(k, k)
            self.m_material.setCurrentIndex(max(0, self.m_material.findData(cur)))
            cur = self.k_member.currentText()
            self.k_member.blockSignals(True)
            self.k_member.clear()
            self.k_member.addItems(list(fr.members))
            if cur in fr.members:
                self.k_member.setCurrentText(cur)
            self.k_member.blockSignals(False)
            s = fr.settings
            self.k_grade.setCurrentText(s.steel_grade)
            self.k_g0.setValue(s.gamma_M0)
            self.k_g1.setValue(s.gamma_M1)
            self.k_segs.setValue(int(s.segments_straight))
            self.k_solver.setCurrentIndex(max(0, self.k_solver.findData(s.solver)))
            self.k_arc.setValue(float(s.max_arc_angle))
            self.k_shear.setChecked(bool(s.shear_deformation))
            if self.s_family.count() == 0:
                from fea_core.beams.sections import families
                self._families = families()
                self.s_family.addItems(list(self._families))
        finally:
            self._filling = False
        self.load_check()
        self._sync_member_form()
        self._sync_section_form()
        self.btn_replace.setEnabled(bool(self.win.parts))
        nm = len(fr.members)
        # parts are translated separately (optional add-ons in a row would not be split by the catalogue template)
        txt = tr(f"Nodes {len(fr.nodes)}, members {nm}, sections {len(fr.sections)}, supports {len(fr.supports)}")
        if fr.connections:
            txt += tr(f", connection to body {len(fr.connections)}")
        if not nm:
            txt += tr(" – enter nodes, cross-section and members (or load them from a STEP wireframe).")
        self.info.setText(txt)

    def color(self, errors) -> str:
        from .main import GREEN, GREY, RED
        fr = self.frame
        if fr is None or not fr.members:
            return GREY
        return RED if any(e.startswith(("Member", "Support", "Connection to body")) for e in errors) else GREEN

    # ================================================================== nodes
    def on_node_row(self, r):
        names = list(self.frame.nodes) if self.frame else []
        if 0 <= r < len(names):
            n = self.frame.nodes[names[r]]
            self.n_name.setText(n.name)
            for s, v in zip(self.n_xyz, n.xyz):
                s.setValue(float(v))
            self._sel = ("node", n.name)
            self.show_3d()

    def _free_name(self, prefix, used):
        k = 1
        while f"{prefix}{k}" in used:
            k += 1
        return f"{prefix}{k}"

    def add_node(self):
        fr = self.frame
        name = self.n_name.text().strip() or self._free_name("N", fr.nodes)
        if name in fr.nodes:
            name = self._free_name(name + "_", fr.nodes)
        fr.add_node(name, [s.value() for s in self.n_xyz])
        self._sel = ("node", name)
        self.n_name.setText(self._free_name("N", fr.nodes))
        self.win.changed()

    def update_node(self):
        fr = self.frame
        r = self.t_nodes.currentRow()
        names = list(fr.nodes)
        if not 0 <= r < len(names):
            QMessageBox.information(self, "Members", "Select a node in the table.")
            return
        old = names[r]
        new = self.n_name.text().strip() or old
        if new != old and new in fr.nodes:
            QMessageBox.information(self, "Members", f"Node '{new}' already exists.")
            return
        xyz = [s.value() for s in self.n_xyz]
        fr.nodes = {(new if k == old else k): (FrameNode(new, xyz) if k == old else v) for k, v in fr.nodes.items()}
        if new != old:                                  # renaming: member references, saving, loads
            for m in fr.members.values():
                m.start = new if m.start == old else m.start
                m.end = new if m.end == old else m.end
            for s in fr.supports:
                s.node = new if s.node == old else s.node
            for c in fr.connections:
                c.node = new if c.node == old else c.node
            for lc in self.win.study.load_cases:
                for l in lc.loads:
                    if getattr(l, "kind", "") == "frame_node" and l.node == old:
                        l.node = new
        self._sel = ("node", new)
        self.win.changed()

    def delete_node(self):
        fr = self.frame
        r = self.t_nodes.currentRow()
        names = list(fr.nodes)
        if not 0 <= r < len(names):
            return
        n = names[r]
        users = [m.name for m in fr.members.values() if n in (m.start, m.end)]
        if users:
            QMessageBox.information(self, "Members", f"Node '{n}' is used by members: {', '.join(users[:5])} – delete them first.")
            return
        del fr.nodes[n]
        fr.supports = [s for s in fr.supports if s.node != n]
        fr.connections = [c for c in fr.connections if c.node != n]
        for lc in self.win.study.load_cases:
            lc.loads = [l for l in lc.loads if not (getattr(l, "kind", "") == "frame_node" and l.node == n)]
        self.win.changed()

    # ================================================================== members
    def _sync_member_form(self):
        up = self.m_up.currentData()
        for s in self.m_upv:
            s.setVisible(up == "vec")
        for s in self.m_thr:
            s.setEnabled(self.m_arc.isChecked())

    def set_hinges(self, ends):
        for end in ("start", "end"):
            for k, cb in enumerate(self.m_rel[end]):
                cb.setChecked(end in ends and k >= 4)

    def on_member_row(self, r):
        fr = self.frame
        names = list(fr.members) if fr else []
        if not 0 <= r < len(names):
            return
        m = fr.members[names[r]]
        self.m_name.setText(m.name)
        self.m_start.setCurrentText(m.start)
        self.m_end.setCurrentText(m.end)
        self.m_section.setCurrentText(m.section)
        self.m_material.setCurrentIndex(max(0, self.m_material.findData(m.material)))
        self.m_kind.setCurrentIndex(max(0, self.m_kind.findData(m.kind)))
        if isinstance(m.up, str):
            self.m_up.setCurrentIndex(self.m_up.findData(m.up))
        elif m.up is not None:
            self.m_up.setCurrentIndex(self.m_up.findData("vec"))
            for s, v in zip(self.m_upv, m.up):
                s.setValue(float(v))
        else:
            self.m_up.setCurrentIndex(0)
        self.m_roll.setValue(float(m.roll))
        self.m_arc.setChecked(m.through is not None)
        if m.through is not None:
            for s, v in zip(self.m_thr, m.through):
                s.setValue(float(v))
        self.m_segs.setValue(int(m.segments))
        for end, rel in (("start", m.releases_start), ("end", m.releases_end)):
            for cb, on in zip(self.m_rel[end], rel):
                cb.setChecked(bool(on))
        self._sel = ("member", m.name)
        self.show_3d()

    def member_from_form(self, name) -> FrameMember | None:
        a, b = self.m_start.currentText(), self.m_end.currentText()
        if not a or not b or a == b:
            QMessageBox.information(self, "Members", "Select two different start and end nodes (Nodes tab).")
            return None
        if not self.m_section.currentText():
            QMessageBox.information(self, "Members", "Add a cross-section first (Cross-sections tab).")
            return None
        up = self.m_up.currentData()
        upv = None if up == "auto" else [s.value() for s in self.m_upv] if up == "vec" else up
        if isinstance(upv, list) and np.linalg.norm(upv) < 1e-12:
            QMessageBox.information(self, "Members", "The “up” vector must not be zero.")
            return None
        return FrameMember(name, a, b, self.m_section.currentText(), self.m_material.currentData(), upv,
                           self.m_roll.value(), [s.value() for s in self.m_thr] if self.m_arc.isChecked() else None,
                           [cb.isChecked() for cb in self.m_rel["start"]], [cb.isChecked() for cb in self.m_rel["end"]],
                           int(self.m_segs.value()), self.m_kind.currentData())

    def add_member(self):
        fr = self.frame
        name = self.m_name.text().strip() or self._free_name("P", fr.members)
        if name in fr.members:
            name = self._free_name(name + "_", fr.members)
        m = self.member_from_form(name)
        if m is None:
            return
        fr.members[name] = m
        self._sel = ("member", name)
        self.m_name.setText(self._free_name("P", fr.members))
        self.win.changed()

    def update_member(self):
        fr = self.frame
        r = self.t_members.currentRow()
        names = list(fr.members)
        if not 0 <= r < len(names):
            QMessageBox.information(self, "Members", "Select a member in the table.")
            return
        old = names[r]
        new = self.m_name.text().strip() or old
        if new != old and new in fr.members:
            QMessageBox.information(self, "Members", f"Member '{new}' already exists.")
            return
        m = self.member_from_form(new)
        if m is None:
            return
        fr.members = {(new if k == old else k): (m if k == old else v) for k, v in fr.members.items()}
        if new != old:
            if old in fr.checks:
                fr.checks[new] = fr.checks.pop(old)
            for lc in self.win.study.load_cases:
                for l in lc.loads:
                    if getattr(l, "kind", "") == "frame_member" and l.member == old:
                        l.member = new
        self._sel = ("member", new)
        self.win.changed()

    def delete_member(self):
        fr = self.frame
        r = self.t_members.currentRow()
        names = list(fr.members)
        if not 0 <= r < len(names):
            return
        n = names[r]
        del fr.members[n]
        fr.checks.pop(n, None)
        for lc in self.win.study.load_cases:
            lc.loads = [l for l in lc.loads if not (getattr(l, "kind", "") == "frame_member" and l.member == n)]
        self.win.changed()

    # ================================================================== sections
    def _sync_section_form(self):
        src = self.s_src.currentData()
        for w in (self.s_family, self.s_lbl_family, self.s_find, self.s_lbl_find, self.s_list, self.s_lbl_list):
            w.setVisible(src == "lib")
        for w in (self.s_shape, self.s_lbl_shape, self.s_lbl_dims):
            w.setVisible(src == "param")
        dims = PARAM[self.s_shape.currentData()][1] if src == "param" else ()
        for k, (lb, sp) in self.s_dims.items():
            lb.setVisible(k in dims)
            sp.setVisible(k in dims)

    def on_family(self):
        if self._filling or not getattr(self, "_families", None):
            return
        self.s_list.clear()
        self.s_list.addItems(self._families.get(self.s_family.currentText(), []))

    def on_find(self, text):
        from fea_core.beams.sections import search
        self.s_list.clear()
        if text.strip():
            self.s_list.addItems(search(text, 40))
        else:
            self.on_family()
        if self.s_list.count():
            self.s_list.setCurrentRow(0)

    def section_def(self, compute=False) -> tuple[SectionDef, object] | None:
        """(SectionDef, SectionProps | None) from the form; compute = compute the properties (always for a CAD face)."""
        src = self.s_src.currentData()
        if src == "lib":
            it = self.s_list.currentItem()
            if it is None:
                QMessageBox.information(self, "Members", "Select a section in the list (series or search).")
                return None
            name = self.s_name.text().strip() or it.text()
            sd = SectionDef(name, library=it.text())
        elif src == "param":
            shape = self.s_shape.currentData()
            dims = {k: self.s_dims[k][1].value() for k in PARAM[shape][1]}
            name = self.s_name.text().strip() or f"{shape} " + "×".join(fmt(v, 4) for v in dims.values())
            sd = SectionDef(name, shape=shape, dims=dims)
        else:
            faces = self.win.view.selected_faces()
            if not faces:
                QMessageBox.information(self, "Members", "Select one planar cross-section face in 3D (the end face of the profile).")
                return None
            pi, fi = faces[0]
            from fea_core.beams.sections import section_from_face
            name = self.s_name.text().strip() or f"Cross-section {self.win.parts[pi].name}"
            try:
                sec, _place = section_from_face(self.win.view.part_faces[pi][fi], name=name)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Members", f"The cross-section from the face cannot be calculated:\n{e}")
                return None
            return SectionDef(name, props=sec.to_dict()), sec
        if not compute:
            return sd, None
        try:
            return sd, sd.resolve()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Members", f"The cross-section cannot be calculated:\n{e}")
            return None

    def preview_section(self, *_):
        if self._filling:
            return
        src = self.s_src.currentData()
        if src == "face" and self.sender() is not None and not isinstance(self.sender(), QPushButton):
            return                                         # FEM of the section only on the button
        if src == "lib" and self.s_list.currentItem() is None:
            return
        r = self.section_def(compute=True)
        if r is not None:
            self.show_props(r[1])

    def show_props(self, sec):
        self._sec_preview = sec
        rows = [(lbl + f" [{unit}]", fmt(float(getattr(sec, k)), 5)) for k, lbl, unit in PROPS_ROWS]
        rows.append(("steel mass [kg/m]", fmt(sec.mass_per_m(), 4)))
        self.s_props.setRowCount(len(rows))
        for r, (a, b) in enumerate(rows):
            self.s_props.setItem(r, 0, QTableWidgetItem(a))
            it = QTableWidgetItem(b)
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.s_props.setItem(r, 1, it)
        self.s_note.setText(f"{sec.name}: {sec.source}" if sec.source else sec.name)
        ax = self.s_plot.axes()
        try:
            from fea_core.beams.sections import outline
            for k, R in enumerate(outline(sec)):
                R = np.vstack([R, R[:1]])
                ax.fill(R[:, 0], R[:, 1], color="#93b4e6" if k == 0 else "white", ec="#2a4a7a", lw=1.2)
        except Exception:  # noqa: BLE001 – own section without an outline in the library
            if sec.fibers:
                F = np.asarray(sec.fibers, float)
                ax.plot(F[:, 0], F[:, 1], "o", color="#2a4a7a")
        ax.axhline(0, color="#888", lw=0.6)
        ax.axvline(0, color="#888", lw=0.6)
        ax.set_aspect("equal")
        ax.set_xlabel("y [mm]")
        ax.set_ylabel("z [mm]")
        self.s_plot.canvas.draw_idle()

    def on_section_row(self, r):
        fr = self.frame
        names = list(fr.sections) if fr else []
        if 0 <= r < len(names):
            try:
                self.show_props(fr.sections[names[r]].resolve())
            except Exception as e:  # noqa: BLE001
                self.s_note.setText(f"The cross-section cannot be calculated: {e}")

    def add_section(self):
        r = self.section_def(compute=True)
        if r is None:
            return
        sd, sec = r
        fr = self.frame
        if sd.name in fr.sections:
            sd.name = self._free_name(sd.name + " ", fr.sections)
        fr.sections[sd.name] = sd
        if sec is not None:
            self.show_props(sec)
        self.m_section.setCurrentText(sd.name)
        self.win.changed()
        self.m_section.setCurrentText(sd.name)

    def delete_section(self):
        fr = self.frame
        r = self.t_sections.currentRow()
        names = list(fr.sections)
        if not 0 <= r < len(names):
            return
        n = names[r]
        users = [m.name for m in fr.members.values() if m.section == n]
        if users:
            QMessageBox.information(self, "Members", f"Cross-section '{n}' is used by members: {', '.join(users[:5])}.")
            return
        del fr.sections[n]
        self.win.changed()

    # ================================================================== supports, connection
    def on_support_row(self, r):
        fr = self.frame
        if fr and 0 <= r < len(fr.supports):
            s = fr.supports[r]
            self.u_node.setCurrentText(s.node)
            for cb, on in zip(self.u_dof, s.dofs):
                cb.setChecked(bool(on))
            for sp, k in zip(self.u_k, list(s.stiffness) + [0.0] * 6):
                sp.setValue(float(k))
            self._sel = ("node", s.node)
            self.show_3d()

    def save_support(self):
        fr = self.frame
        node = self.u_node.currentText()
        if not node:
            QMessageBox.information(self, "Members", "Define nodes first.")
            return
        dofs = [cb.isChecked() for cb in self.u_dof]
        k = [sp.value() for sp in self.u_k]
        if not any(dofs) and not any(k):
            QMessageBox.information(self, "Members", "Check at least one constrained component or enter a spring.")
            return
        fr.supports = [s for s in fr.supports if s.node != node] + [FrameSupport(node, dofs, k)]
        self.win.changed()

    def delete_support(self):
        fr = self.frame
        r = self.t_sup.currentRow()
        if 0 <= r < len(fr.supports):
            fr.supports.pop(r)
            self.win.changed()

    def on_conn_row(self, r):
        fr = self.frame
        if fr and 0 <= r < len(fr.connections):
            c = fr.connections[r]
            self.win.view.set_selected_faces(self.win.sel_faces.get(c.selection, []))
            self._sel = ("node", c.node)
            self.show_3d()

    def add_connection(self):
        node = self.c_node.currentText()
        if not node:
            QMessageBox.information(self, "Members", "Define nodes first.")
            return
        sels = self.win.need_faces()
        if not sels:
            return
        name = self.win.new_selection("Member connection", sels)
        self.frame.connections.append(FrameConnection(node, name))
        self.win.view.set_selected_faces([])
        self.win.changed(invalidate_mesh=True)

    def delete_connection(self):
        fr = self.frame
        r = self.t_conn.currentRow()
        if 0 <= r < len(fr.connections):
            c = fr.connections.pop(r)
            self.win._drop_selection(c.selection)
            self.win.changed(invalidate_mesh=True)

    # ================================================================== assessment
    def save_settings(self):
        if self._filling or self.frame is None:
            return
        s = self.frame.settings
        new = dict(steel_grade=self.k_grade.currentText(), gamma_M0=self.k_g0.value(), gamma_M1=self.k_g1.value(),
                   segments_straight=int(self.k_segs.value()), max_arc_angle=self.k_arc.value(),
                   shear_deformation=self.k_shear.isChecked(), solver=self.k_solver.currentData() or "timoshenko")
        if all(getattr(s, k) == v for k, v in new.items()):
            return
        for k, v in new.items():
            setattr(s, k, v)
        self.win.changed()

    def load_check(self):
        fr = self.frame
        c = fr.checks.get(self.k_member.currentText(), {}) if fr else {}
        for k, sp in self.k_vals.items():
            sp.setValue(float(c.get(k) or 0.0))
        self.k_lt.setChecked(bool(c.get("lt_restrained")))
        self.k_sway_y.setChecked(bool(c.get("sway_y")))
        self.k_sway_z.setChecked(bool(c.get("sway_z")))
        self.k_fab.setCurrentIndex(max(0, self.k_fab.findData(c.get("fabrication", "auto"))))

    def save_check(self):
        name = self.k_member.currentText()
        if not name:
            return
        c = {k: sp.value() for k, sp in self.k_vals.items() if sp.value()}
        for k, cb in (("lt_restrained", self.k_lt), ("sway_y", self.k_sway_y), ("sway_z", self.k_sway_z)):
            if cb.isChecked():
                c[k] = True
        if self.k_fab.currentData() != "auto":
            c["fabrication"] = self.k_fab.currentData()
        if c:
            self.frame.checks[name] = c
        else:
            self.frame.checks.pop(name, None)
        self.win.changed(invalidate_results=False)     # assessment inputs do not change the FEM results
        self.win.frame_assessment_changed()

    def reset_check(self):
        name = self.k_member.currentText()
        if name and name in self.frame.checks:
            del self.frame.checks[name]
            self.win.changed(invalidate_results=False)
            self.win.frame_assessment_changed()
        self.load_check()

    # ================================================================== taking over geometry
    def from_wires(self):
        st = self.win.study
        if st is None:
            return
        sec = self.m_section.currentText()
        if not sec:
            QMessageBox.information(self,
                                    "Members",
                                    "Add a cross-section first (Cross-sections tab) – it will be used for new members.")
            return
        last = self.win.settings.value("last_dir", "")
        p, _ = QFileDialog.getOpenFileName(self, "STEP wireframe", last, "STEP (*.step *.stp)")
        if p:
            self.import_wires(p, sec)

    def import_wires(self, path, section) -> list[str]:
        from fea_core.beams.geometry import frame_from_wires, wire_edges
        try:
            edges = wire_edges(path)
            if not edges:
                QMessageBox.information(self, "Members", "The STEP file contains no free edges (wireframe) outside bodies.")
                return []
            new = frame_from_wires(self.win.study, edges, section)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Members", f"The wireframe cannot be imported:\n{e}")
            return []
        self.win.log_msg(f"Members from wireframe {path}: {len(new)} members (cross-section {section})")
        self.win.changed()
        return new

    def replace_part(self):
        win = self.win
        faces = win.view.selected_faces()
        if not faces:
            QMessageBox.information(self, "Members", "Select a face in 3D of the part to be replaced by a member.")
            return
        name = win.parts[faces[0][0]].name
        self.replace_part_named(name)

    def replace_part_named(self, name):
        from fea_core.beams.geometry import member_from_part
        win = self.win
        try:
            d = member_from_part(win.study, win.parts, name)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Members", str(e))
            return None
        win.resolve_all()
        win.view.set_selected_faces([])
        win.log_msg(f"Part '{name}' replaced by a member: section {d['section']}" + (" (library)" if d["library"] else "")
                    + f", L = {fmt(d['length_mm'], 5)} mm, connection with {len(d['connections'])} ends")
        win.changed(invalidate_mesh=True)
        return d

    # ================================================================== 3D
    def bbox(self):
        fr = self.frame
        if fr is None or not fr.nodes:
            return None
        X = np.array([n.xyz for n in fr.nodes.values()], float)
        return X.min(axis=0), X.max(axis=0)

    def show_3d(self, loads=None):
        """Frame system in 3D (persistent objects): members, nodes, supports, hinges, connections, and possibly loads of the state."""
        win = self.win
        v = win.view
        v.clear_persist(KEY)
        st = win.study
        fr = self.frame
        v.extra_bbox = self.bbox()
        if fr is None or not fr.nodes:
            return
        in_stage = win.STAGES[win.stage] == "Members"
        L = 0.06 * v.model_size()
        sel_kind, sel_name = self._sel if in_stage else ("", None)
        normal, truss, hl = [], [], []
        for m in fr.members.values():
            if m.start not in fr.nodes or m.end not in fr.nodes:
                continue
            P = member_polyline(st, m)
            seg = list(zip(P, P[1:]))
            (hl if sel_kind == "member" and m.name == sel_name else truss if m.kind == "truss" else normal).extend(seg)
        if normal:
            v.add_lines(f"{KEY}:m", normal, C_MEMBER, 3.0, persist=True)
        if truss:
            v.add_lines(f"{KEY}:t", truss, C_TRUSS, 2.0, persist=True)
        if hl:
            v.add_lines(f"{KEY}:sel", hl, C_SEL, 5.0, label=sel_name, persist=True)
        names = list(fr.nodes)
        X = [np.asarray(fr.nodes[n].xyz, float) for n in names]
        show_lbl = in_stage and len(names) <= 80
        v.add_points(f"{KEY}:n", X, C_NODE, names if show_lbl else None, size=4.0)
        if sel_kind == "node" and sel_name in fr.nodes:
            v.add_points(f"{KEY}:nsel", [fr.nodes[sel_name].xyz], C_SEL, None, size=7.0)
        # supports: triangle under the node (clamping + square), springs as a zigzag
        seg, springs = [], []
        for s in fr.supports:
            if s.node not in fr.nodes:
                continue
            p = np.asarray(fr.nodes[s.node].xyz, float)
            ez, ex = np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
            a, b, c = p - L * ez + 0.6 * L * ex, p - L * ez - 0.6 * L * ex, p
            seg += [(a, b), (b, c), (c, a)]
            if all(s.dofs[3:]):
                seg += [(a - 0.25 * L * ez, b - 0.25 * L * ez), (a, a - 0.25 * L * ez), (b, b - 0.25 * L * ez)]
            for d, k in enumerate(s.stiffness[:3]):
                if k:
                    springs += win.spring_segments(p, -np.eye(3)[d], 1.2 * L)
        if seg:
            v.add_lines(f"{KEY}:sup", seg, C_SUPPORT, 2.0, persist=True)
        if springs:
            v.add_lines(f"{KEY}:spr", springs, (20, 150, 120), 2.0, persist=True)
        # hinges: a ring a little away from the end of the member
        ring = []
        for m in fr.members.values():
            if m.start not in fr.nodes or m.end not in fr.nodes or m.kind == "truss":
                continue
            P = member_polyline(st, m)
            for end, rel in (("start", m.releases_start), ("end", m.releases_end)):
                if not any(rel[4:]):
                    continue
                q0, q1 = (P[0], P[1]) if end == "start" else (P[-1], P[-2])
                d = q1 - q0
                ln = float(np.linalg.norm(d))
                if ln < 1e-9:
                    continue
                d /= ln
                c = q0 + min(0.6 * L, 0.2 * ln) * d
                e1 = np.cross(d, [0, 0, 1.0]) if abs(d[2]) < 0.9 else np.cross(d, [1.0, 0, 0])
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(d, e1)
                t = np.linspace(0, 2 * np.pi, 13)
                Q = [c + 0.25 * L * (math.cos(a) * e1 + math.sin(a) * e2) for a in t]
                ring += list(zip(Q, Q[1:]))
        if ring:
            v.add_lines(f"{KEY}:rel", ring, C_RELEASE, 2.0, persist=True)
        cross = []
        for c in fr.connections:
            if c.node in fr.nodes:
                p = np.asarray(fr.nodes[c.node].xyz, float)
                cross += [(p - 0.4 * L * e, p + 0.4 * L * e) for e in np.eye(3)]
        if cross:
            v.add_lines(f"{KEY}:conn", cross, C_CONN, 3.0, persist=True)
        if loads is not None:
            self.show_loads(loads, L)
        v._update()

    def show_loads(self, lc, L):
        """Load arrows of the members of the load state (forces at nodes, distributed loads along the length)."""
        st = self.win.study
        fr = st.frame
        v = self.win.view
        arr, mom = [], []
        for l in lc.loads:
            k = getattr(l, "kind", "")
            if k == "frame_node" and l.node in fr.nodes:
                p = np.asarray(fr.nodes[l.node].xyz, float)
                if np.linalg.norm(l.force) > 0:
                    arr.append((p, np.asarray(l.force, float), 2.0 * L))
                if np.linalg.norm(l.moment) > 0:
                    mom.append((p, np.asarray(l.moment, float), 2.0 * L))
            elif k == "frame_member" and l.member in fr.members:
                m = fr.members[l.member]
                if m.start not in fr.nodes or m.end not in fr.nodes:
                    continue
                P = member_polyline(st, m)
                R = member_axes(st, m) if l.axes == "local" else None
                q0 = np.asarray(l.q, float)
                q1 = np.asarray(l.q_end if l.q_end is not None else l.q, float)
                qmax = max(np.linalg.norm(q0), np.linalg.norm(q1), 1e-30)
                cum = np.r_[0, np.cumsum([np.linalg.norm(b - a) for a, b in zip(P, P[1:])])]
                for t in np.linspace(0.0, 1.0, 7):
                    s = t * cum[-1]
                    i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(P) - 2)
                    u = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-30)
                    p = P[i] + u * (P[i + 1] - P[i])
                    q = (1 - t) * q0 + t * q1
                    d = R.T @ q if R is not None else q
                    if np.linalg.norm(d) > 0:
                        arr.append((p, d, 1.6 * L * max(np.linalg.norm(q) / qmax, 0.2)))
        if arr:
            v.add_arrows(f"{KEY}:load", arr, C_FLOAD, persist=True)
        if mom:
            v.add_arrows(f"{KEY}:mom", mom, C_FLOAD, double=True, persist=True)

    def pick(self, x, y) -> tuple[str, str] | None:
        """Node / member under the cursor (device pixels): the nearest node within 10 px, otherwise a member within 7 px."""
        fr = self.frame
        if fr is None or not fr.nodes:
            return None
        v = self.win.view
        tol = 10 * v.dpr()
        names = list(fr.nodes)
        P2 = v.project([fr.nodes[n].xyz for n in names])
        d = np.linalg.norm(P2 - [x, y], axis=1)
        k = int(np.argmin(d))
        if d[k] <= tol:
            return "node", names[k]
        best = (0.7 * tol, None)
        for m in fr.members.values():
            if m.start not in fr.nodes or m.end not in fr.nodes:
                continue
            Q = v.project(member_polyline(self.win.study, m, 12))
            for a, b in zip(Q, Q[1:]):
                ab = b - a
                t = float(np.clip(((np.array([x, y]) - a) @ ab) / max(ab @ ab, 1e-12), 0, 1))
                dd = float(np.linalg.norm(a + t * ab - [x, y]))
                if dd < best[0]:
                    best = (dd, m.name)
        return ("member", best[1]) if best[1] else None

    def select(self, kind, name):
        """Selection of a node / member (click in 3D) → table row and form."""
        fr = self.frame
        if kind == "node":
            names = list(fr.nodes)
            tab, t = 0, self.t_nodes
            if self.tabs.currentIndex() in (3, 4):        # supports / connection: node into the form
                (self.u_node if self.tabs.currentIndex() == 3 else self.c_node).setCurrentText(name)
                self._sel = ("node", name)
                self.show_3d()
                return
        else:
            names = list(fr.members)
            tab, t = 1, self.t_members
            if self.tabs.currentIndex() == 5:
                self.k_member.setCurrentText(name)
                self._sel = ("member", name)
                self.show_3d()
                return
        self.tabs.setCurrentIndex(tab)
        if name in names:
            t.setCurrentCell(names.index(name), 0)
