"""Joints panel: bolts (recognition from the names, by hand into a hole) and welds (designation of the parts), assembly
parameters and assessment, utilisation tables in the bottom dock and highlighting in 3D.

The assessment (VDI 2230, EN 1993-1-8, EN 1993-1-9 / IIW) is computed from the FEM results in the main thread
(takes a fraction of a second) – after the calculation and when the step is shown, if the load curve has changed.
"""
from __future__ import annotations

import math
from html import escape

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QHBoxLayout, QHeaderView, QInputDialog, QLabel, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

from fea_core.joints import tables
from fea_core.joints.model import WeldSpec

from .safe_browser import SafeTextBrowser
from . import theme
from .widgets import C_HEX, decimal, dspin, fit_rows, fit_tabs, fmt, panel_header, scroll_body, set_placeholder, table, Section, help_tip

GREEN, ORANGE, RED, GREY = "#2e9e4f", "#e69a1a", "#d23b3b", "#9aa1ab"
C_BOLT = (70, 100, 150)
C_WELD = (235, 140, 40)
CONNECTION = {"nut": "with nut", "tapped": "into thread", "fitted": "fitted"}
PRELOAD_MODES = [("auto", "automatic – min(F_M,zul; pressure under head)"),
                  ("force", "force F_M,max [N]"), ("torque", "tightening torque M_A [N·m]")]
PRELOAD_AUTO_TIP = ("Automatic assembly preload F_M,max = min(F_M,zul; 0.9·p_G·A_p) (VDI 2230):\n• F_M,zul – permissible assembly preload of the bolt (reduced stress when tightening 90 % Rp0.2),\n• 0.9·p_G·A_p – 90 % of the bearing capacity of the surface under the head / nut (limit pressure p_G of the\n  clamped part material × area A_p); for plastics (PLA, PETG) and soft alloys this limit governs.\nWhich limit governed is shown in the bolt description below the form.")
# joint fields by effect on the calculation and the decision functions change_level are in the core (shared with MCP).
from fea_core.joints.levels import (BOLT_ASSESS, BOLT_FE, BOLT_PRELOAD, WELD_ASSESS, WELD_FE,  # noqa: F401,E402
                                    change_level)
CATEGORIES = [("A", "A – shear and bearing"), ("B", "B – slip-resistant (SLS)"), ("C", "C – slip-resistant (ULS)")]


def util_color(u) -> str:
    if u is None or not math.isfinite(u):
        return GREY
    if u <= 0.9:
        return GREEN
    if u <= 1.0:
        return ORANGE
    return RED


def util_rgb(u) -> tuple[int, int, int]:
    c = QColor(util_color(u))
    return c.red(), c.green(), c.blue()


TMP_SUFFIX = " (from model)"


def _select_or_add(combo, data, label: str) -> None:
    """Selects the item with the data `data` (for a menu without data by text); a missing one is added temporarily with the suffix "(from model)".
    Temporary items of the previous bolt are removed first.
    """
    for i in range(combo.count() - 1, -1, -1):
        if combo.itemText(i).endswith(TMP_SUFFIX):
            combo.removeItem(i)
    i = combo.findData(data)
    if i < 0 and isinstance(data, str):
        i = combo.findText(data)
    if i < 0:
        combo.addItem(label + TMP_SUFFIX, data)
        i = combo.count() - 1
    combo.setCurrentIndex(i)


class JointsPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.assessment = None
        self._weld_geo: dict[str, dict] = {}
        self._filling = False
        _, lay = scroll_body(self)
        lay.addWidget(panel_header("Joints", "Bolts (VDI 2230, EN 1993-1-8) and welds (EN 1993-1-8, 1-9)"))
        h = QHBoxLayout()
        b = QPushButton("Detect joints…")
        theme.set_primary(b)
        b.setToolTip("Finds bolts, nuts, washers and welds in the model – suggestions to confirm")
        b.setWhatsThis("<b>Recognize joints</b><p>Scans the model parts and suggests bolts (bolt + nut / washer, or a bolt into a thread) and welds. Bolts with a standard in the part name are reliable; parts without a standard are recognized by shape (shank, head, hex) with a stated confidence.</p><p>Welds are found by name, layer or CAD attribute, or by weld colour (field “Weld colour in CAD”). Suggestions are shown in a table to confirm – selecting a row highlights the parts in 3D. Recognized bolts are not meshed; a preloaded beam replaces them.</p><p>Names with a standard (e.g. “ISO 4017 - M8 x 30 - 8.8”, “ISO 7089 - 8,4 - 200 HV”) are taken reliably; a suggestion by shape or colour is shown for confirmation and added to the existing ones.</p>")
        b.clicked.connect(self.detect)
        h.addWidget(b)
        b = QPushButton("Bolt into selected hole…")
        b.setToolTip("Model without bolts: a bolt into a selected cylindrical hole (selection in 3D)")
        b.setWhatsThis("<b>Bolt into selected hole</b><p>For a model in which the bolts are not drawn. Select the cylindrical face of the hole in 3D and confirm the size (proposed from the fine clearance hole of ISO 273) and the strength grade.</p><p>The bolt clamps all parts with a coaxial hole; in the FEM it is replaced by a beam with stiffness per VDI 2230 connected to the annulus under the head and nut.</p>")
        b.clicked.connect(self.bolt_from_hole)
        h.addWidget(b)
        lay.addLayout(h)

        # ---------------------------------------------------------- bolts
        g = Section("Bolts",
                    key="joints.bolts",
                    collapsed=False,
                    tip="Bolted joints: assembly parameters, preload (VDI 2230), checks",
                    help="Bolted joints of the study: assembly parameters and preload (VDI 2230), checks per EN 1993-1-8")
        v = QVBoxLayout(g.body)
        self.bolt_list = QListWidget()
        self.bolt_list.setMaximumHeight(130)
        set_placeholder(self.bolt_list, "No bolts yet – use 'Detect joints' or add from CAD")
        fit_rows(self.bolt_list, 2, 8)
        self.bolt_list.setToolTip("Bolts of the study: designation, joint type, l_K; dot colour = utilization")
        self.bolt_list.setWhatsThis("<b>List of bolts</b><p>Each bolt is replaced in the FE model by a beam (bolt parts, nuts and washers are not meshed) connected to the bearing faces under the head and nut, or to the thread in the hole.</p><p>After selecting a row you can change the size, class, preload and friction in the form; changes take effect only with the “Save bolt changes” button. The detailed check of the selected bolt is below in the panel.</p><p>Dot colour = utilization (green ≤ 0.9, orange ≤ 1, red > 1, grey = off). Click = edit in the form below, the bolt parts are highlighted in 3D.</p>")
        self.bolt_list.currentRowChanged.connect(self.on_bolt_row)
        v.addWidget(self.bolt_list)
        self.bolt_form = QWidget()
        vf = QVBoxLayout(self.bolt_form)
        vf.setContentsMargins(0, 0, 0, 0)
        f = QFormLayout()
        self.b_size = QComboBox()
        for k in tables.PITCH:
            if k >= 3:
                self.b_size.addItem(f"M{k:g}", float(k))
        help_tip(self.b_size, "Nominal thread diameter (coarse pitch, ISO 261)",
                 "Nominal diameter of the metric thread (coarse pitch ISO 261).<br>Changing the size after saving requires a new mesh (bearing-face imprint).")
        self.b_class = QComboBox()
        self.b_class.addItems(tables.CLASS_NAMES)
        help_tip(self.b_class, "Strength class ISO 898-1 (steel) or ISO 3506 (stainless)",
                 "Strength class: steel ISO 898-1 (4.6 … 12.9; e.g. 8.8 = Rm 800, Re 0.8·Rm),<br>stainless ISO 3506 (A2/A4-50/70/80). Defines F_M,max, the beam stiffness and f_ub of the checks.")
        hs = QHBoxLayout()
        hs.addWidget(self.b_size, 1)
        hs.addWidget(QLabel("grade"))
        hs.addWidget(self.b_class, 1)
        f.addRow("Size", hs)
        self.b_mode = QComboBox()
        for key, txt in PRELOAD_MODES:
            self.b_mode.addItem(txt, key)
        self.b_mode.setItemData(0, PRELOAD_AUTO_TIP, Qt.ToolTipRole)
        help_tip(self.b_mode, "How the bolt preload is specified",
                 PRELOAD_AUTO_TIP + "<br><br>Force F_M,max or moment M_A = the specified tightening; F_M,min = F_M,max / α_A.")
        self.b_mode.currentIndexChanged.connect(self.on_mode)
        f.addRow("Preload", self.b_mode)
        self.b_pre = dspin(0.0, 0.0, 1e7, 1, 100.0)
        help_tip(self.b_pre, "Preload: force F_M,max [N], or tightening torque M_A [N·m], as selected",
                 "Preload value according to the choice above: force F_M,max [N] or tightening torque M_A [N·m]<br>(converted to force via the friction μ_G, μ_K). Not entered for “automatic”.")
        f.addRow("Value", self.b_pre)
        self.b_alpha = dspin(1.6, 1.0, 4.0, 2, 0.1)
        help_tip(self.b_alpha, "Tightening factor α_A (torque wrench 1.4–1.6, impact wrench 2.5–4)",
                 "Tightening factor α_A: torque wrench 1.4–1.6, impact wrench 2.5–4, tightening to yield / by angle 1.0–1.2")
        f.addRow("α_A", self.b_alpha)
        self.b_muG, self.b_muK, self.b_muT = (dspin(x, 0.02, 0.6, 3, 0.01) for x in (0.12, 0.12, 0.15))
        help_tip(self.b_muG, "Thread friction coefficient μ_G (lowest value per VDI 2230)",
                 "Thread friction coefficient μ_G (VDI 2230, lowest value) – relation M_A ↔ F_M and<br>torsion during tightening (F_M,max). Default 0.12; lower friction = higher force at the same M_A.")
        help_tip(self.b_muK, "Friction coefficient under head / nut μ_K (VDI 2230)",
                 "Friction coefficient under head / nut μ_K (VDI 2230, lowest value) – relation<br>M_A ↔ F_M. Default 0.12.")
        hm = QHBoxLayout()
        for lbl, w in (("thread μ_G", self.b_muG), ("head μ_K", self.b_muK)):
            hm.addWidget(QLabel(lbl))
            hm.addWidget(w, 1)
        f.addRow("Friction", hm)
        help_tip(self.b_muT, "Friction coefficient in the joint interface μ_T for the slip check (VDI 2230)",
                 "Friction coefficient in the interface between clamped parts μ_T – slip check per VDI 2230<br>(default 0.15). Slip per EN 1993-1-8 (categories B, C) takes μ from “Check settings”.")
        f.addRow("Friction in joint interface μ_T", self.b_muT)
        self.b_shank = dspin(0.0, 0.0, 200.0, 2, 0.5, " mm")
        self.b_shank.setSpecialValueText("by hole (otherwise d + 1)")
        help_tip(self.b_shank, "Shank diameter of a fitted bolt d_sch (0 = from the smallest hole)",
                 "Fitted bolt: shank diameter d_sch (DIN 609 / 7968: d + 1 mm). 0 = smallest hole of the clamped parts, without a hole d + 1 mm.")
        self.b_shank_lbl = QLabel("Shank diameter")
        f.addRow(self.b_shank_lbl, self.b_shank)
        self.b_shank.setVisible(False)               # only for a fitted bolt (on_bolt_row)
        self.b_shank_lbl.setVisible(False)
        self.b_thread = QCheckBox("shear plane in thread")
        help_tip(self.b_thread, "Shear plane passes through the thread (otherwise through the shank)",
                 "Shear per EN 1993-1-8: checked = shear plane passes through the thread<br>(area A_s, α_v 0.6 / 0.5 by class), otherwise through the shank (area A, α_v = 0.6).<br>Not used for a fitted bolt (shank shear).")
        self.b_enabled = QCheckBox("include with bolt")
        self.b_enabled.setToolTip("Disabled bolt: the bolt, nut and washer parts are meshed as ordinary bodies")
        f.addRow(self.b_thread)
        f.addRow(self.b_enabled)
        vf.addLayout(f)
        hb = QHBoxLayout()
        b = QPushButton("Save bolt changes")
        help_tip(b, "Save the form values to the selected bolt",
                 "Writes the form values to the bolt selected in the list. Depending on the change, only the check is recalculated,<br>or results are discarded (friction, preload, class), or even the mesh (size).")
        b.clicked.connect(self.save_bolt)
        hb.addWidget(b)
        b = QPushButton("Delete")
        help_tip(b, "Remove the selected bolt from the study (new mesh)",
                 "Remove the selected bolt from the study (the bolt parts are then meshed as ordinary bodies; new mesh)")
        b.clicked.connect(self.delete_bolt)
        hb.addWidget(b)
        vf.addLayout(hb)
        self.bolt_info = QLabel("")
        self.bolt_info.setWordWrap(True)
        self.bolt_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        vf.addWidget(self.bolt_info)
        v.addWidget(self.bolt_form)
        self.bolt_form.setVisible(False)
        lay.addWidget(g)

        # ---------------------------------------------------------- svary
        g = Section("Welds",
                    key="joints.welds",
                    collapsed=False,
                    tip="Welds as bodies: EN 1993-1-8 check and EN 1993-1-9 / IIW fatigue",
                    help="Welds as bodies in the model: check by the directional method (EN 1993-1-8) and fatigue (EN 1993-1-9 / IIW)")
        v = QVBoxLayout(g.body)
        b = QPushButton("Mark the part of the selected face as a weld")
        b.setToolTip("The weld is a body in the model: click any weld face in 3D")
        b.clicked.connect(self.mark_weld)
        v.addWidget(b)
        self.weld_list = QListWidget()
        self.weld_list.setMaximumHeight(110)
        set_placeholder(self.weld_list, "No welds yet – mark body in 3D or use CAD color")
        fit_rows(self.weld_list, 2, 8)
        self.weld_list.setToolTip("Welds of the study: name and body; dot colour = utilization")
        self.weld_list.setWhatsThis("<b>List of welds</b><p>A weld is a separate body of the model (fillet or butt). Forces are computed in sections along the root line from the stresses in the weld elements and converted to σ_perp, τ_perp, τ_par in the effective section a·L.</p><p>After selecting a row, edit the parameters in the form and save with the “Save weld changes” button; below the form are the weld type, a, L and the parts in contact from the CAD geometry.</p><p>Dot colour = utilization (green ≤ 0.9, orange ≤ 1, red > 1, grey = not checked). Click = edit in the form below, the weld body is highlighted in 3D.</p>")
        self.weld_list.currentRowChanged.connect(self.on_weld_row)
        v.addWidget(self.weld_list)
        self.weld_form = QWidget()
        vwf = QVBoxLayout(self.weld_form)
        vwf.setContentsMargins(0, 0, 0, 0)
        f = QFormLayout()
        self.w_a = dspin(0.0, 0.0, 100.0, 2, 0.5, " mm")
        self.w_a.setSpecialValueText("from geometry")
        help_tip(self.w_a, "Effective throat thickness a of a fillet weld [mm]; 0 = from geometry",
                 "Effective throat thickness a (height of the inscribed triangle of a fillet weld) [mm].<br>0 = from the weld body geometry (fillet: a = A1·A2·sinθ / (2V)).")
        f.addRow("Effective throat a", self.w_a)
        self.w_seg = QSpinBox()
        self.w_seg.setRange(1, 200)
        help_tip(self.w_seg, "Number of sections along the weld for forces and stresses (default 10)",
                 "Number of sections along the weld in which forces and stresses are computed (default 10).<br>More sections = finer profile along the length, but fewer elements per section.")
        f.addRow("Segments along length", self.w_seg)
        self.w_fs = dspin(36.0, 1.0, 500.0, 1, 1.0, " MPa")
        self.w_ft = dspin(80.0, 1.0, 500.0, 1, 1.0, " MPa")
        self.w_fs.setToolTip("FAT for σ_w = √(σ_perp² + τ_perp²): fillet weld root 36 (EN 1993-1-9 Table 8.5, IIW)")
        self.w_ft.setToolTip("FAT for τ_w = τ_par: fillet weld carrying shear 80 (m = 5)")
        f.addRow("FAT σ_w", self.w_fs)
        f.addRow("FAT τ_w", self.w_ft)
        self.w_curve = QComboBox()
        self.w_curve.addItems(["EN 1993-1-9", "IIW"])
        help_tip(self.w_curve, "S-N curve shape of the weld: EN 1993-1-9 or IIW",
                 "S-N curve shape of the weld (FAT = stress range at 2·10⁶ cycles):<br>EN 1993-1-9 – m = 3, from 5·10⁶ m = 5, cut-off at 10⁸ (shear m = 5);<br>IIW – m = 3 (shear 5), beyond the knee at 10⁷ (shear 10⁸) m = 22.")
        self.w_gamma = dspin(1.15, 1.0, 2.0, 2, 0.05)
        self.w_gamma.setToolTip("Partial factor for fatigue strength (EN 1993-1-9 Table 3.1): 1.0–1.35")
        f.addRow("S-N curve", self.w_curve)
        f.addRow("γ_Mf", self.w_gamma)
        self.w_toe = QCheckBox("weld toe fatigue (hot-spot IIW)")
        help_tip(self.w_toe, "Weld toe fatigue – structural stress extrapolated from the surface",
                 "Structural stress in the plate at the fillet weld toe, extrapolated from the surface<br>(IIW type “a”: 0.4t and 1.0t, for a coarse mesh 0.5t and 1.5t) – NOT the maximum at a node.<br>For accuracy the mesh at the toe should be ≤ 0.4t (local refinement).")
        f.addRow(self.w_toe)
        self.w_fat_toe = QComboBox()
        self.w_fat_toe.setEditable(True)
        for val, txt in ((90.0, "90 – load-carrying fillet weld"), (100.0, "100 – non-load-carrying fillet / butt weld"),
                         (112.0, "112 – full-penetration butt weld (hot-spot)")):
            self.w_fat_toe.addItem(f"{txt}", val)
        help_tip(self.w_fat_toe, "FAT for the hot-spot stress (IIW, EN 1993-1-9 tab. B.1); a number can be typed",
                 "FAT for structural stress (hot-spot) – IIW, EN 1993-1-9 Table B.1. A number can be typed in.")
        f.addRow("Toe FAT", self.w_fat_toe)
        self.w_toe_t = dspin(0.0, 0.0, 500.0, 2, 1.0, " mm")
        self.w_toe_t.setSpecialValueText("from mesh")
        self.w_toe_t.setToolTip("Plate thickness at the toe for reference points (0 = by ray from the mesh)")
        f.addRow("Plate t at toe", self.w_toe_t)
        self.w_filler = QComboBox()
        self.w_filler.addItem("matching the base material", "")
        from fea_core.joints.tables import FILLERS
        for name, (re_, rm, _) in FILLERS.items():
            self.w_filler.addItem(f"{name} – Re {re_:g} / Rm {rm:g} MPa", name)
        help_tip(self.w_filler, "Filler metal of the weld: same as base metal, or overmatching",
                 "Filler metal (EN 1993-1-8, 4.2: should be the same as or overmatching the<br>base metal; undermatching → check with f_u = Rm of the filler metal)")
        f.addRow("Filler metal", self.w_filler)
        self.w_open = QCheckBox("separate contact of welded parts")
        help_tip(self.w_open, "Separate the contact of welded parts (root gap, the load goes through the weld)",
                 "Direct contact of parts joined by a weld is not connected (root gap) – the whole load goes<br>through the weld, as EN 1993-1-8 assumes. Contact pressure is then not transmitted (conservative).")
        self.w_enabled = QCheckBox("assess the weld")
        help_tip(self.w_enabled, "Off: the weld is not assessed and the contact is not opened",
                 "Off: the weld is not checked and the contact of welded parts is not separated –<br>the weld body stays in the mesh as an ordinary part bonded to its surroundings.")
        f.addRow(self.w_open)
        f.addRow(self.w_enabled)
        vwf.addLayout(f)
        hb = QHBoxLayout()
        b = QPushButton("Save weld changes")
        help_tip(b, "Save the form values to the selected weld",
                 "Writes the form values to the weld selected in the list. Fatigue parameters only<br>recalculate the check; sections and separation of contact discard results, or the mesh.")
        b.clicked.connect(self.save_weld)
        hb.addWidget(b)
        b = QPushButton("Delete")
        help_tip(b, "Remove the selected weld from the study (new mesh)",
                 "Remove the selected weld from the study (the body stays in the model as an ordinary part; new mesh)")
        b.clicked.connect(self.delete_weld)
        hb.addWidget(b)
        vwf.addLayout(hb)
        self.weld_info = QLabel("")
        self.weld_info.setWordWrap(True)
        vwf.addWidget(self.weld_info)
        v.addWidget(self.weld_form)
        self.weld_form.setVisible(False)
        hc = QHBoxLayout()
        hc.addWidget(QLabel("Weld color in CAD"))
        self.w_color = QComboBox()
        help_tip(self.w_color, "Parts of this colour are marked as welds on recognition",
                 "Parts of this colour (e.g. welds coloured in Fusion) are marked as welds on recognition.<br>List = part colours in the model with the count of parts; “(do not use)” = rule off.")
        self.w_color.currentIndexChanged.connect(self.on_weld_color)
        hc.addWidget(self.w_color, 1)
        v.addLayout(hc)
        lay.addWidget(g)

        # ---------------------------------------------------------- excluded parts
        g = self.ign_group = Section("Parts excluded from the analysis",
                    key="joints.ignored",
                    collapsed=True,
                    tip="Parts that are neither meshed nor calculated (their mass and stiffness are then missing)")
        v = QVBoxLayout(g.body)
        lb = QLabel("Checked parts are not meshed (magnets, pins, free parts without connection). Transparent in 3D.")
        lb.setWordWrap(True)
        v.addWidget(lb)
        self.ign_list = QListWidget()
        self.ign_list.setMaximumHeight(150)
        help_tip(self.ign_list, "Parts excluded from the calculation – tick a part to exclude it (new mesh)",
                 "All parts of the model; tick a part to exclude it from the calculation (new mesh).<br>Bolt parts are replaced by a beam and are not meshed even when unticked.")
        self.ign_list.itemChanged.connect(self.on_ignored_item)
        v.addWidget(self.ign_list)
        hb = QHBoxLayout()
        b = QPushButton("Exclude the part of the selected face")
        b.setToolTip("Select any face of a part in 3D (also several parts) and exclude it from the analysis")
        b.clicked.connect(lambda: self.ignore_selected(True))
        hb.addWidget(b)
        b = QPushButton("Restore")
        help_tip(b, "Return to the calculation the part selected in 3D (or untick it in the list)",
                 "Restore to the analysis the part whose face you have selected in 3D (or uncheck the part in the list)")
        b.clicked.connect(lambda: self.ignore_selected(False))
        hb.addWidget(b)
        v.addLayout(hb)
        lay.addWidget(g)

        # ---------------------------------------------------------- bonded parts (shared mesh)
        g = self.bond_group = Section("Connected parts (shared mesh)",
                    key="joints.bonded",
                    collapsed=True,
                    tip="Parts permanently bonded by shared mesh nodes – check after meshing",
                    help="Pairs of parts rigidly connected by shared mesh nodes (face contact) – check after meshing")
        v = QVBoxLayout(g.body)
        lb = QLabel("Parts that touch along a face are rigidly connected after meshing (shared nodes, no contact). Check that this is intended – otherwise exclude the part or modify the geometry.")
        lb.setWordWrap(True)
        v.addWidget(lb)
        self.bond_table = table(["Part A", "Part B", "Contact [mm²]"],
                                stretch_last=True,
                                placeholder="Joints are found after the mesh is created (Mesh step)",
                                rows=(3, 8))
        help_tip(self.bond_table, "Parts joined by the shared mesh and the contact area [mm²]; click = highlight",
                 "Parts joined by a shared mesh and the area of the common contact face [mm²].<br>Clicking a row highlights both parts in 3D; remove an unwanted connection with the buttons below.<br>Filled after the mesh is created (Mesh step) or after the calculation.")
        self.bond_table.cellClicked.connect(lambda r, _c: self.on_bond_row(r))
        v.addWidget(self.bond_table)
        self.bond_info = QLabel("")
        self.bond_info.setWordWrap(True)
        v.addWidget(self.bond_info)
        hb = QHBoxLayout()
        for txt, k in (("Exclude part A", 0), ("Exclude part B", 1)):
            b = QPushButton(txt)
            b.setToolTip("Exclude from the analysis the part of the selected row (list “Parts excluded from the analysis”)")
            b.clicked.connect(lambda _=False, k=k: self.ignore_bond_part(k))
            hb.addWidget(b)
        v.addLayout(hb)
        lay.addWidget(g)
        self._bonds: list[dict] = []
        self._bond_key = None

        # ---------------------------------------------------------- parts meshed separately (*TIE)
        g = self.tie_group = Section("Parts meshed separately (*TIE connection)",
                    key="joints.tie",
                    collapsed=True,
                    tip="Separate mesh of the part, connected by a *TIE constraint to neighbouring parts (CalculiX only)",
                    help="Parts meshed separately from the assembly (own independent mesh) and connected to neighbouring parts<br>by a *TIE constraint (nodes of the separate part tied to the neighbour's surface) – for parts that Gmsh<br>does not join with the assembly (tangent contact, “PLC Error”), or for a different mesh density.")
        v = QVBoxLayout(g.body)
        lb = QLabel("A checked part gets its own mesh and is tied to the parts it touches with a *TIE constraint (rigid connection without shared nodes). CalculiX solver only.")
        lb.setWordWrap(True)
        v.addWidget(lb)
        self.sep_list = QListWidget()
        self.sep_list.setMaximumHeight(130)
        help_tip(self.sep_list, "Tick to mesh the part separately and connect it by *TIE to its neighbours",
                 "Tick a part to mesh it separately and connect it by *TIE to its neighbours (new mesh).<br>Contact is searched after meshing: faces of the part's surface within the tolerance of another part's surface.")
        self.sep_list.setWhatsThis("<b>Mesh separately (*TIE)</b><p>Normally Gmsh joins the assembly parts that touch along a face with shared nodes. When that is not possible (tangent contact of a cylinder in a slot, error “PLC Error: segment and facet intersect”) or you want to give a part a different mesh, check it: it gets its own mesh and is tied to all parts it touches with a *TIE constraint.</p><p>The connections found (part, contact face) are shown in the table below the list after the mesh is generated. Contact face 0 = the part is not connected to anything.</p>")
        self.sep_list.itemChanged.connect(self.on_separate_item)
        v.addWidget(self.sep_list)
        hb = QHBoxLayout()
        hb.addWidget(QLabel("Contact tolerance"))
        self.tie_tol = dspin(0.0, 0.0, 100.0, 4, 0.05, " mm", special="auto (5 % of the element)")
        help_tip(self.tie_tol, "Contact tolerance for *TIE [mm]; auto = 5 % of the element size",
                 "Largest distance of a separate part's face from a neighbour's surface that is still joined by *TIE [mm].<br>Auto = 5 % of the global element size. Increase it for CAD with a small gap between parts.")
        self.tie_tol.editingFinished.connect(self.on_tie_tol)
        hb.addWidget(self.tie_tol, 1)
        v.addLayout(hb)
        self.tie_table = table(["Part", "Contact [mm²]", "Tolerance [mm]"],
                               stretch_last=True,
                               placeholder="No parts meshed separately",
                               rows=(3, 8))
        help_tip(self.tie_table, "*TIE connections found after meshing: separately meshed part, contact area [mm²] and tolerance used",
                 "*TIE connections found after meshing: separately meshed part, contact area<br>and tolerance used. Clicking a row highlights the part in 3D.")
        self.tie_table.cellClicked.connect(lambda r, _c: self.on_tie_row(r))
        v.addWidget(self.tie_table)
        self.tie_info = QLabel("")
        self.tie_info.setWordWrap(True)
        v.addWidget(self.tie_info)
        lay.addWidget(g)
        self._ties: list[dict] = []

        # ---------------------------------------------------------- settings
        g = Section("Check settings",
                    key="joints.settings",
                    collapsed=True,
                    tip="Common settings of the bolt and weld checks (EN 1993-1-8) and the preload state in FE",
                    help="Common check settings for all bolts and welds (EN 1993-1-8) and the preload case in the FEM")
        f = QFormLayout(g.body)
        self.s_cat = QComboBox()
        for key, txt in CATEGORIES:
            self.s_cat.addItem(txt, key)
        help_tip(self.s_cat, "Shear joint category per EN 1993-1-8 (A, B, C)",
                 "Shear joint category per EN 1993-1-8 (tab. 3.2): A – transverse force carried by shear<br>and bearing (slip informative only); B – no slip at SLS, C – no slip at ULS<br>(friction joint, check F_s,Rd with μ below and γ_M3).")
        f.addRow("Bolt category (EN 1993-1-8)", self.s_cat)
        self.s_split = QComboBox()
        self.s_split.addItem("bolt group – uniform + torsion", "group")
        self.s_split.addItem("per FEM (connected interface, elastic)", "fe")
        help_tip(self.s_split, "Distribution of the transverse force between bolts: group, or per FE",
                 "Distribution of the transverse force among the bolts of the same joint. A bonded joint in FE concentrates shear<br>toward the ends of the lap (as a friction joint without slip); after slip (category A)<br>the force is distributed among the bolts – EN 1993-1-8 assumes a bolt group.")
        f.addRow("Transverse force per bolt", self.s_split)
        self.s_mu = dspin(0.3, 0.05, 0.6, 2, 0.05)
        self.s_mu.setToolTip("Friction coefficient for F_s,Rd: surface class A 0.5, B 0.4, C 0.3, D 0.2")
        f.addRow("μ for slip (B, C)", self.s_mu)
        self.s_g2 = dspin(1.25, 1.0, 2.0, 2, 0.05)
        help_tip(self.s_g2, "Partial factor γ_M2 (EN 1993-1-8, tab. 2.1: 1.25)",
                 "Partial safety factor γ_M2 (EN 1993-1-8, tab. 2.1: 1.25) – resistance of bolts<br>in shear, tension and bearing, and of welds. The National Annex may prescribe another value.")
        f.addRow("γ_M2", self.s_g2)
        self.s_pre = QCheckBox("“Bolt preload” case in the FEM analysis")
        self.s_pre.setToolTip("Separate load case with the preload F_V of all bolts (clamping stress)")
        f.addRow(self.s_pre)
        for w in (self.s_cat, self.s_split, self.s_mu, self.s_g2):
            (w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged).connect(self.on_settings)
        self.s_pre.toggled.connect(self.on_settings)
        lay.addWidget(g)

        # ---------------------------------------------------------- assessment
        hb = QHBoxLayout()
        b = QPushButton("Re-check")
        b.setToolTip("Recompute the check from the results (e.g. after changing the fatigue load history)")
        b.clicked.connect(lambda: self.compute(force=True))
        hb.addWidget(b)
        hb.addStretch(1)
        lay.addLayout(hb)
        self.detail = SafeTextBrowser()
        self.detail.setMinimumHeight(220)
        self.detail.setVisible(False)
        help_tip(self.detail, "Detailed check of the selected bolt or weld",
                 "Detailed check of the selected bolt / weld: utilization of the individual checks<br>(grey “informative” lines are not counted in the utilization), warnings and load steps")
        lay.addWidget(self.detail)
        lay.addStretch(1)
        self._build_tables()
        self.on_mode(0)

    # ================================================================== bottom tables
    def _build_tables(self):
        self.tabs = QTabWidget()
        self.t_bolts = table(["Bolt", "Designation", "Joint", "F_SA max [N]", "F_Q max [N]", "Φ", "Utilization",
                              "Governing"], stretch_last=True, placeholder="Bolts will be checked after the calculation (Solve step)", rows=(3, 8))
        self.t_welds = table(["Weld", "Part", "a [mm]", "L [mm]", "σ_eq max [MPa]", "D fatigue", "Utilization",
                              "Governing"], stretch_last=True, placeholder="Welds will be checked after the calculation (Solve step)", rows=(3, 8))
        help_tip(self.t_bolts, "Bolt check from the results (maxima over steps); click = detail",
                 "Bolt check from the results (maxima over steps): F_SA = additional axial bolt force,<br>F_Q = transverse force in the joint, Φ = F_SA / F_A (VDI 2230), utilization = worst check<br>(≤ 1 is OK; colour: green ≤ 0.9, orange ≤ 1, red &gt; 1).<br>Clicking a row opens the bolt in the Joints step with the detailed check.")
        help_tip(self.t_welds, "Weld check from the results; click = detail in Joints",
                 "Weld check from the results: a, L from geometry, σ_eq = √(σ_perp² + 3(τ_perp² + τ_par²)),<br>D fatigue = damage (worse of root and toe), utilization = worst check (≤ 1 is OK).<br>Clicking a row opens the weld in the Joints step with the detailed check.")
        self.t_bolts.cellClicked.connect(lambda r, _c: self.select_joint("bolt", r))
        self.t_welds.cellClicked.connect(lambda r, _c: self.select_joint("weld", r))
        self.tabs.addTab(self.t_bolts, "Bolts")
        self.tabs.addTab(self.t_welds, "Welds")
        fit_tabs(self.tabs)

    def fill_tables(self):
        st = self.win.study
        a = self.assessment
        bolt_chk = {c.name: c for c in a.bolts} if a else {}
        weld_chk = {c.name: c for c in a.welds} if a else {}
        bolts = st.bolts if st else []
        self.t_bolts.setRowCount(len(bolts))
        for r, b in enumerate(bolts):
            c = bolt_chk.get(b.name)
            vdi = c.data.get("vdi", {}) if c else {}
            vals = [b.name, b.designation, CONNECTION.get(b.connection, b.connection),
                    fmt(vdi.get("F_SA"), 5) if c else "–", fmt(vdi.get("F_Q"), 5) if c else "–",
                    fmt(vdi.get("phi"), 3) if c else "–",
                    fmt(c.utilization, 3) if c else ("not computed" if not b.enabled else "–"),
                    c.governing if c else ""]
            self._fill_row(self.t_bolts, r, vals, c.utilization if c else None)
        welds = st.welds if st else []
        self.t_welds.setRowCount(len(welds))
        for r, w in enumerate(welds):
            c = weld_chk.get(w.name)
            d = c.data if c else {}
            seq = None
            for pc in d.get("per_case", []):
                import numpy as np
                S = np.asarray(pc["stress"])
                if len(S):
                    e = float(np.sqrt(S[:, 0] ** 2 + 3 * (S[:, 1] ** 2 + S[:, 2] ** 2)).max())
                    seq = e if seq is None else max(seq, e)
            fat = (d.get("fatigue") or {}).get("damage")
            toe_d = ((d.get("toe") or {}).get("fatigue") or {}).get("damage")
            if toe_d is not None:
                fat = toe_d if fat is None else max(fat, toe_d)          # root and toe
            vals = [w.name, w.part, fmt(d.get("a"), 3), fmt(d.get("length"), 4), fmt(seq, 4), fmt(fat, 3),
                    fmt(c.utilization, 3) if c else ("not checked" if not w.enabled else "–"),
                    c.governing if c else ""]
            self._fill_row(self.t_welds, r, vals, c.utilization if c else None)
        n_b = sum(1 for c in bolt_chk.values() if not c.ok)
        n_w = sum(1 for c in weld_chk.values() if not c.ok)
        self.tabs.setTabText(0, f"Bolts ({len(bolts)})" + (f" – {n_b} ×" if n_b else ""))
        self.tabs.setTabText(1, f"Welds ({len(welds)})" + (f" – {n_w} ×" if n_w else ""))

    @staticmethod
    def _fill_row(table, r, vals, util):
        bg = QBrush(QColor(util_color(util)).lighter(170)) if util is not None else None
        for col, txt in enumerate(vals):
            it = QTableWidgetItem(str(txt))
            if col in (3, 4, 5, 6):
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if bg is not None and col == 6:
                it.setBackground(bg)
                it.setForeground(QBrush(QColor("#000000")))
            table.setItem(r, col, it)

    # ================================================================== obnova
    def refresh(self):
        st = self.win.study
        if getattr(self, "_parts_id", None) != id(self.win.parts):      # new model → weld geometry again
            self._parts_id = id(self.win.parts)
            self._weld_geo = {}
        self._filling = True
        try:
            if st:
                js = st.joints
                self.s_cat.setCurrentIndex(max(0, self.s_cat.findData(js.category)))
                self.s_split.setCurrentIndex(max(0, self.s_split.findData(js.shear_split)))
                self.s_mu.setValue(js.slip_mu)
                self.s_g2.setValue(js.gamma_M2)
                self.s_pre.setChecked(js.preload_case)
        finally:
            self._filling = False
        self._fill_lists()
        self._fill_colors()
        self._fill_ignored()
        self._fill_bonds()
        self._fill_ties()
        self._update_badges()
        self.apply_ghost()
        self.fill_tables()

    def _update_badges(self):
        """Counts in the header of collapsed sections (badge on the right, empty = hidden)."""
        n_ign = sum(1 for i in range(self.ign_list.count()) if self.ign_list.item(i).checkState() == Qt.Checked)
        self.ign_group.set_badge(str(n_ign) if n_ign else "")
        self.bond_group.set_badge(str(len(self._bonds)) if self._bonds else "")
        self.tie_group.set_badge(str(len(self._ties)) if self._ties else "")

    # ================================================================== bonded parts (fea_core.contacts)
    def _bond_mesh(self):
        win = self.win
        if win.mesh is not None:
            return win.mesh
        return win.results.mesh if win.results is not None else None

    def _fill_bonds(self):
        """Pairs of parts rigidly connected by a shared mesh (+ suspicious connections only by an edge / point) to be confirmed."""
        mesh = self._bond_mesh()
        key = None if mesh is None else (id(mesh), len(mesh.nodes), len(mesh.tets))
        if key == self._bond_key:
            return
        self._bond_key = key
        self._bonds = []
        self.bond_table.setRowCount(0)
        if mesh is None:
            self.bond_info.setText("<i>Shown after the mesh is generated (Mesh step) or after the analysis.</i>")
            return
        from fea_core.contacts import bonded_pairs, edge_only
        try:
            self._bonds = bonded_pairs(mesh)
            edges = edge_only(mesh)
        except Exception as e:  # noqa: BLE001
            self.bond_info.setText(f"<span style='color:{RED}'>The connection of parts cannot be determined: {escape(str(e))}</span>")
            return
        self.bond_table.setRowCount(len(self._bonds))
        for r, d in enumerate(self._bonds):
            for c, txt in enumerate((d["a"], d["b"], fmt(d["area"], 4))):
                it = QTableWidgetItem(txt)
                if c == 2:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.bond_table.setItem(r, c, it)
        rows = [f"Connected pairs: {len(self._bonds)}" if self._bonds else
                ("No parts are connected by the mesh." if len(mesh.part_names) > 1 else "The model has a single part.")]
        if edges:
            rows.append("<span style='color:#b36b00'>! Connection by an edge or a point only (a few shared nodes – a singularity, almost a hinge; usually unintended): "
                        + ", ".join(f"{escape(str(a))} – {escape(str(b))}" for a, b in edges) + "</span>")
        self.bond_info.setText("<br>".join(rows))
        if edges:
            self.win.log_msg("Warning: parts connected by an edge / a point only: " + ", ".join(f"{a} – {b}" for a, b in edges))

    # ================================================================== parts meshed separately (fea_core.ties)
    def _fill_ties(self):
        st = self.win.study
        excl = st.excluded_parts() if st else set()
        sep = set(st.mesh.separate) if st else set()
        self.sep_list.blockSignals(True)
        self.sep_list.clear()
        for p in self.win.parts or []:
            if p.name in excl and p.name not in sep:
                continue
            it = QListWidgetItem(p.name)
            it.setData(Qt.UserRole, p.name)
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if p.name in sep else Qt.Unchecked)
            it.setToolTip("mesh separately (*TIE connection)")
            self.sep_list.addItem(it)
        self.sep_list.blockSignals(False)
        self.tie_tol.blockSignals(True)
        self.tie_tol.setValue(float(st.mesh.tie_tolerance or 0.0) if st else 0.0)
        self.tie_tol.blockSignals(False)
        mesh = self._bond_mesh()
        self._ties = list(mesh.report.get("ties", [])) if mesh is not None else []
        self.tie_table.setRowCount(len(self._ties))
        for r, t in enumerate(self._ties):
            for c, txt in enumerate((t["part"], fmt(t["area_mm2"], 5), fmt(t["tolerance_mm"], 3))):
                it = QTableWidgetItem(txt)
                if c:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tie_table.setItem(r, c, it)
        if not sep:
            self.tie_info.setText("")
        elif mesh is None:
            self.tie_info.setText("<i>*TIE connections are found after the mesh is generated (Mesh step) or during the analysis.</i>")
        else:
            tied = {t["part"] for t in self._ties}
            free = [p for p in sorted(sep) if p not in tied and p in mesh.part_names]
            self.tie_info.setText(f"*TIE connection: {len(self._ties)}" + (
                f"<br><span style='color:#b36b00'>! No connection (does not touch another part within tolerance): {escape(', '.join(free))}</span>" if free else ""))

    def on_separate_item(self, it):
        st = self.win.study
        if st is None:
            return
        name = it.data(Qt.UserRole)
        on = it.checkState() == Qt.Checked
        cur = list(st.mesh.separate)
        new = (cur + [name] if name not in cur else cur) if on else [n for n in cur if n != name]
        if new == cur:
            return
        st.mesh.separate = new
        self.win.log_msg(("Mesh separately (*TIE): " if on else "Mesh with assembly: ") + name)
        self.win.changed(invalidate_mesh=True)

    def on_tie_tol(self):
        st = self.win.study
        if st is None:
            return
        v = self.tie_tol.value() or None
        if v != st.mesh.tie_tolerance:
            st.mesh.tie_tolerance = v
            self.win.changed(invalidate_mesh=True)

    def on_tie_row(self, r):
        if 0 <= r < len(self._ties) and self.active():
            self.win.view.set_selected_faces(self.part_faces([self._ties[r]["part"]]))

    def on_bond_row(self, r):
        if not (0 <= r < len(self._bonds)) or not self.active():
            return
        d = self._bonds[r]
        self.win.view.set_selected_faces(self.part_faces([d["a"], d["b"]]))

    def ignore_bond_part(self, k: int):
        r = self.bond_table.currentRow()
        if not (0 <= r < len(self._bonds)):
            from .main import msg
            msg(self.win, "Joints", "Select a pair of connected parts in the table.")
            return
        self.set_ignored([self._bonds[r]["a" if k == 0 else "b"]], True)

    # ================================================================== weld colour, excluded parts
    def _fill_colors(self):
        from fea_core.joints.detect import rgb_hex
        st = self.win.study
        cnt: dict[str, int] = {}
        for p in self.win.parts or []:
            if getattr(p, "color", None) is not None:
                h = rgb_hex(p.color)
                cnt[h] = cnt.get(h, 0) + 1
        cur = st.joints.weld_colors[0] if st and st.joints.weld_colors else ""
        self.w_color.blockSignals(True)
        self.w_color.clear()
        self.w_color.addItem("(do not use)", "")
        for h, n in sorted(cnt.items(), key=lambda kv: -kv[1]):
            from PySide6.QtGui import QIcon, QPixmap
            pm = QPixmap(14, 14)
            pm.fill(QColor(h))
            self.w_color.addItem(QIcon(pm), f"{h} ({n} parts)", h)
        if cur and self.w_color.findData(cur) < 0:
            self.w_color.addItem(f"{cur} (not in model)", cur)
        self.w_color.setCurrentIndex(max(0, self.w_color.findData(cur)))
        self.w_color.blockSignals(False)

    def on_weld_color(self, _):
        st = self.win.study
        if st is None or self._filling:
            return
        h = self.w_color.currentData() or ""
        new = [h] if h else []
        if new != st.joints.weld_colors:
            mesh_ok = self._mesh_valid()
            st.joints.weld_colors = new
            self._keep_mesh(mesh_ok)             # recognition rule – does not change the mesh or the results
            self.win.dirty = True
            self.win.push_undo()
            self.win.update_title()
            self.win.log_msg(f"Weld color in CAD: {h or 'do not use'} (applied when joints are detected)")

    def _fill_ignored(self):
        st = self.win.study
        excl = {p for b in st.active_bolts() for p in b.cad_parts} if st else set()
        ign = set(st.ignored_parts) if st else set()
        self.ign_list.blockSignals(True)
        self.ign_list.clear()
        for p in self.win.parts or []:
            it = QListWidgetItem(p.name + ("  (bolt – replaced by a beam)" if p.name in excl else ""))
            it.setData(Qt.UserRole, p.name)
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if p.name in ign else Qt.Unchecked)
            self.ign_list.addItem(it)
        self.ign_list.blockSignals(False)

    def set_ignored(self, names: list[str], on: bool):
        self.win.set_ignored(names, on)

    def refresh_ignored(self):
        """The main window calls after a change of the excluded parts (also from 3D or the tree): the list and the badge are rewritten."""
        self._fill_ignored()
        self._update_badges()

    def on_ignored_item(self, it):
        self.set_ignored([it.data(Qt.UserRole)], it.checkState() == Qt.Checked)

    def ignore_selected(self, on: bool):
        from .main import msg
        names = list(dict.fromkeys(self.win.parts[pi].name for pi, _ in self.win.view.selected_faces()))
        if not names:
            msg(self.win, "Joints", "Select any face of a part in 3D.")
            return
        self.win.view.set_selected_faces([])
        self.set_ignored(names, on)

    def apply_ghost(self):
        """Excluded parts translucent in 3D (in all steps with a model)."""
        self.win.apply_ghost()

    def offer_ignore(self, text: str) -> bool:
        """After the error "free bodies" it offers to exclude the unconnected parts. True = excluded."""
        st = self.win.study
        if st is None or "free bodies" not in text:
            return False
        names = {p.name for p in self.win.parts or []}
        lines = [ln.strip() for ln in text.splitlines()]
        free = [ln for ln in lines if ln in names]
        if not free:
            return False
        r = QMessageBox.question(self.win, "Free parts",
                                 "These parts are not connected to a support (no contact, bolt or weld):\n\n  "
                                 + "\n  ".join(free[:20]) + ("\n  …" if len(free) > 20 else "")
                                 + "\n\nExclude them from the calculation? (Parts excluded from the calculation – the Joints step)")
        if r != QMessageBox.Yes:
            return False
        self.set_ignored(free, True)
        return True

    def _fill_lists(self):
        from .main import dot_icon
        st = self.win.study
        chk = {c.name: c for c in self.assessment.all()} if self.assessment else {}
        rb, rw = self.bolt_list.currentRow(), self.weld_list.currentRow()
        self.bolt_list.blockSignals(True)
        self.bolt_list.clear()
        for b in (st.bolts if st else []):
            c = chk.get(b.name)
            col = util_color(c.utilization) if c else (GREY if not b.enabled else C_HEX(C_BOLT))
            it = QListWidgetItem(dot_icon(col), f"{b.name}: {b.designation} – {CONNECTION.get(b.connection, '')}, "
                                                f"l_K {fmt(b.grip, 4)} mm" + ("" if b.enabled else " (off)"))
            self.bolt_list.addItem(it)
        self.bolt_list.blockSignals(False)
        self.weld_list.blockSignals(True)
        self.weld_list.clear()
        for w in (st.welds if st else []):
            c = chk.get(w.name)
            col = util_color(c.utilization) if c else (GREY if not w.enabled else C_HEX(C_WELD))
            self.weld_list.addItem(QListWidgetItem(dot_icon(col), f"{w.name}: {w.part}"
                                                   + ("" if w.enabled else " (not assessed)")))
        self.weld_list.blockSignals(False)
        if st and st.bolts:
            self.bolt_list.setCurrentRow(min(max(rb, 0), len(st.bolts) - 1))
        else:
            self.bolt_list.setCurrentRow(-1)
            self.bolt_form.setVisible(False)
        if st and st.welds:
            self.weld_list.setCurrentRow(min(max(rw, 0), len(st.welds) - 1))
        else:
            self.weld_list.setCurrentRow(-1)
            self.weld_form.setVisible(False)

    # ================================================================== bolts
    def current_bolt(self):
        st = self.win.study
        r = self.bolt_list.currentRow()
        return st.bolts[r] if st and 0 <= r < len(st.bolts) else None

    def on_mode(self, _):
        mode = self.b_mode.currentData()
        self.b_pre.setEnabled(mode != "auto")
        self.b_pre.setSuffix({"force": " N", "torque": " N·m"}.get(mode, ""))

    def on_bolt_row(self, row):
        b = self.current_bolt()
        if b is None:
            self.bolt_form.setVisible(False)
            self.bolt_info.setText("")
            return
        self.bolt_form.setVisible(True)
        self._filling = True
        try:
            # size / class outside the list (below M3, a class from a newer version…) is shown temporarily, so that saving does not
            # silently change the bolt to the first item of the list
            _select_or_add(self.b_size, float(b.size), f"M{float(b.size):g}")
            _select_or_add(self.b_class, b.strength, b.strength)
            self.b_mode.setCurrentIndex(max(0, self.b_mode.findData(b.preload_mode)))
            self.b_pre.setValue(b.preload)
            self.b_alpha.setValue(b.alpha_A)
            self.b_muG.setValue(b.mu_G)
            self.b_muK.setValue(b.mu_K)
            self.b_muT.setValue(b.mu_T)
            self.b_thread.setChecked(b.thread_in_shear)
            self.b_enabled.setChecked(b.enabled)
            fitted = b.connection == "fitted"
            self.b_shank.setValue(float(getattr(b, "shank_d", None) or 0.0))
            self.b_shank.setVisible(fitted)
            self.b_shank_lbl.setVisible(fitted)
        finally:
            self._filling = False
        self.on_mode(0)
        self.show_bolt_info(b)
        self.show_detail(b.name)
        if self.active():
            self.win.view.set_selected_faces(self.part_faces(b.cad_parts))

    def active(self) -> bool:
        """Is the Joints step shown? (otherwise do not touch the face selection in 3D)"""
        return getattr(self.win, "stage", -1) == self.win.STAGE["Joints"]

    def show_bolt_info(self, b):
        from fea_core.joints.fem import bolt_basics
        rows = [f"<b>{escape(str(b.name))}</b> – {escape(str(b.designation))}, {CONNECTION.get(b.connection, b.connection)}"]
        plates = ", ".join(f"{escape(str(p['part']))} (t {fmt(p['s1'] - p['s0'], 3)}, Ø{fmt(p.get('hole'), 3)})" for p in b.plates)
        rows.append(f"Clamped: {plates}; washers: {len(b.washers)}")
        if b.connection == "tapped":
            rows.append(f"Thread in part '{escape(str(b.nut_part))}', engaged {fmt(b.thread_end - b.nut_s, 3)} mm")
        if b.connection == "fitted":
            rows.append(f"Fitted: shank Ø{fmt(b.shank, 3)} mm" + ("" if b.shank_d else
                                                                 " (from hole, otherwise d + 1)"))
        try:
            bas = bolt_basics(self.win.study, b)
            rows.append(f"l_K = {fmt(bas['lK'], 4)} mm, F_M,zul = {fmt(bas['F_Mzul'], 4)} N, "
                        f"M_A = {fmt(bas['M_A'], 3)} N·m, F_M = {fmt(bas['F_Mmin'], 4)}–{fmt(bas['F_Mmax'], 4)} N, "
                        f"F_Z = {fmt(bas['F_Z'], 3)} N, Φ_K = {fmt(bas['phi_K'], 3)}")
            if bas.get("preload_mode") == "auto" and bas.get("preload_limit"):
                rows.append(f"Automatic preload is limited by: {escape(str(bas['preload_limit']))}")
            if b.hole:
                rows.append(f"Hole Ø{fmt(b.hole, 3)}: {tables.clearance_class(b.size, b.hole)} clearance (ISO 273)")
        except Exception as e:  # noqa: BLE001
            rows.append(f"<span style='color:{RED}'>{escape(str(e))}</span>")
        if b.note:
            rows.append(f"<i>{escape(str(b.note))}</i>")
        self.bolt_info.setText("<br>".join(rows))

    def save_bolt(self):
        b = self.current_bolt()
        if b is None or self._filling:
            return
        old, mesh_ok = b.to_dict(), self._mesh_valid()
        b.size = float(self.b_size.currentData())
        b.pitch = None if b.size in tables.PITCH else b.pitch
        b.strength = self.b_class.currentData() or self.b_class.currentText()
        b.preload_mode = self.b_mode.currentData()
        b.preload = self.b_pre.value()
        b.alpha_A = self.b_alpha.value()
        b.mu_G, b.mu_K, b.mu_T = self.b_muG.value(), self.b_muK.value(), self.b_muT.value()
        b.thread_in_shear = self.b_thread.isChecked()
        b.enabled = self.b_enabled.isChecked()
        if b.connection == "fitted":
            b.shank_d = self.b_shank.value() or None
        row = self.bolt_list.currentRow()
        self._apply_change(change_level("bolt", old, b.to_dict(), self.win.study.joints.preload_case), mesh_ok)
        self.bolt_list.setCurrentRow(row)

    # ------------------------------------------------------------------ effect of the changes on the mesh and results
    def _mesh_valid(self) -> bool:
        """Does the mesh apply to the current study? (call BEFORE the study is changed)"""
        win = self.win
        return win.mesh is not None and win.valid_mesh() is not None

    def _keep_mesh(self, mesh_ok: bool):
        """The change does not change the meshing inputs: the mesh valid before the change stays valid (the signature MESH_KEYS contains
        the bolts, welds and joint settings, so it would otherwise be meshed again).
        """
        win = self.win
        if mesh_ok and win.mesh is not None:
            win._mesh_sig = win.study_signature(win.MESH_KEYS)

    def _apply_change(self, level: str, mesh_ok: bool):
        """Maps the change of a joint by its effect (change_level): mesh / FEM calculation / assessment only."""
        win = self.win
        if not level:
            return
        if level == "mesh":
            win.changed(invalidate_mesh=True)
        elif level == "fe":
            win.changed()                       # results discarded, the mesh stays
            self._keep_mesh(mesh_ok)
        else:
            self._assessment_changed(mesh_ok)

    def _assessment_changed(self, mesh_ok: bool):
        """A change only for the assessment: the mesh and the FEM results stay, the assessment is recomputed."""
        win = self.win
        win.dirty = True
        win.push_undo()
        win.update_title()
        self._keep_mesh(mesh_ok)
        self.compute(force=True)
        if self.active():
            win.refresh_sets()

    def delete_bolt(self):
        b = self.current_bolt()
        if b is None:
            return
        self.win.study.bolts.remove(b)
        self.bolt_form.setVisible(False)
        self.win.changed(invalidate_mesh=True)

    def detect(self):
        from fea_core.joints.detect import detect
        st = self.win.study
        if st is None:
            return
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rep = detect(self.win.parts, st.bolts, settings=st.joints)
        finally:
            QApplication.restoreOverrideCursor()
        known_parts = {p for b in st.bolts for p in b.cad_parts}
        known_w = {w.part for w in st.welds}
        sugg = [s for s in rep.suggestions
                if (s.kind == "bolt" and not set(s.spec.cad_parts) & known_parts)
                or (s.kind == "weld" and s.spec.part not in known_w)]
        lines = [f"Detected: {sum((s.kind == 'bolt' for s in rep.suggestions))} bolts ({len(rep.shape_bolts)} by shape), {len(rep.welds)} welds; {len(sugg)} new"]
        lines += [f"  {n}: {fn.describe()}" for n, fn in rep.fasteners.items()]
        lines += [f"Warning: {w}" for w in rep.warnings]
        for ln in lines:
            self.win.log_msg(ln)
        if not sugg:
            from .main import msg
            msg(self.win, "Joints", "\n".join(lines[:1] + [f"• {w}" for w in rep.warnings[:15]]
                                             + ([] if rep.suggestions else
                                                ["No part has a bolt standard or weld designation in its name, and none matches a bolt with nut / into thread by shape."])
                                             + [f"• {r}" for r in rep.rejected[:5]]),
                QMessageBox.Information)
            return
        dlg = DetectDialog(self.win, sugg, rep.rejected)
        dlg.table.currentCellChanged.connect(
            lambda r, *_: self.active() and 0 <= r < len(sugg) and self.win.view.set_selected_faces(
                self.part_faces(sugg[r].spec.cad_parts if sugg[r].kind == "bolt" else [sugg[r].spec.part])))
        if not exec_dialog(dlg):
            self.win.view.set_selected_faces([])
            return
        chosen = dlg.selected()
        nb, nw = len(st.bolts), len(st.welds)
        for s in chosen:
            if s.kind == "bolt":
                nb += 1
                k = nb
                while any(b.name == f"Bolt {k}" for b in st.bolts):
                    k += 1
                s.spec.name = f"Bolt {k}"
                st.bolts.append(s.spec)
            else:
                nw += 1
                k = nw
                while any(w.name == f"Weld {k}" for w in st.welds):
                    k += 1
                s.spec.name = f"Weld {k}"
                st.welds.append(s.spec)
        self.win.log_msg(f"Added: {sum((s.kind == 'bolt' for s in chosen))} bolts, {sum((s.kind == 'weld' for s in chosen))} welds (confirmed from {len(sugg)} proposals)")
        self.win.view.set_selected_faces([])
        if chosen:
            self.win.changed(invalidate_mesh=True)

    def bolt_from_hole(self):
        from fea_core.joints.detect import bolt_from_hole
        from fea_core.geometry import cylinder_axis
        from .main import msg
        sel = self.win.view.selected_faces()
        hole = next(((pi, fi) for pi, fi in sel if cylinder_axis(self.win.view.part_faces[pi][fi]) is not None), None)
        if hole is None:
            msg(self.win, "Joints", "Select the cylindrical face of the hole in 3D (the Joints step is in face selection mode).")
            return
        r = cylinder_axis(self.win.view.part_faces[hole[0]][hole[1]])[2]
        # the largest bolt whose fine clearance hole (ISO 273) fits into the hole
        guess = max((k for k in tables.CLEARANCE if tables.CLEARANCE[k][0] <= 2 * r + 1e-6), default=8)
        items = [f"M{k:g}" for k in tables.CLEARANCE]
        size, ok = QInputDialog.getItem(self.win, "Bolt into hole", f"Hole Ø{2 * r:.2f} mm – bolt size:",
                                        items, items.index(f"M{guess:g}"), False)
        if not ok:
            return
        cls, ok = QInputDialog.getItem(self.win, "Bolt into hole", "Strength grade:", tables.CLASS_NAMES,
                                       tables.CLASS_NAMES.index("8.8"), False)
        if not ok:
            return
        st = self.win.study
        b, warns = bolt_from_hole(self.win.parts, hole[0], hole[1], float(size[1:]), cls,
                                  name=f"Bolt {len(st.bolts) + 1}")
        for w in warns:
            self.win.log_msg(f"Warning: {w}")
        if b is None:
            msg(self.win, "Joints", "\n".join(warns) or "Cannot create bolt")
            return
        st.bolts.append(b)
        self.win.view.set_selected_faces([])
        self.win.log_msg(f"{b.name}: {b.designation} into a hole Ø{fmt(2 * r, 4)}, clamped "
                         + ", ".join(p["part"] for p in b.plates))
        self.win.changed(invalidate_mesh=True)

    # ================================================================== svary
    def current_weld(self):
        st = self.win.study
        r = self.weld_list.currentRow()
        return st.welds[r] if st and 0 <= r < len(st.welds) else None

    def weld_geo(self, w) -> dict | None:
        g = self._weld_geo.get(w.part)
        if g is None:
            from fea_core.joints.detect import weld_geometry
            try:
                g = weld_geometry(self.win.parts, w.part)
            except Exception as e:  # noqa: BLE001
                g = {"error": str(e)}
            self._weld_geo[w.part] = g
        return g

    def on_weld_row(self, row):
        w = self.current_weld()
        if w is None:
            self.weld_form.setVisible(False)
            self.weld_info.setText("")
            return
        self.weld_form.setVisible(True)
        self._filling = True
        try:
            self.w_a.setValue(w.a or 0.0)
            self.w_seg.setValue(w.segments)
            self.w_fs.setValue(w.fat_sigma)
            self.w_ft.setValue(w.fat_tau)
            self.w_curve.setCurrentText(w.sn_curve)
            self.w_gamma.setValue(w.gamma_Mf)
            self.w_open.setChecked(w.open_root)
            self.w_enabled.setChecked(w.enabled)
            self.w_toe.setChecked(w.toe)
            k = self.w_fat_toe.findData(float(w.fat_toe))
            if k >= 0:
                self.w_fat_toe.setCurrentIndex(k)
            else:
                self.w_fat_toe.setEditText(fmt(w.fat_toe))
            self.w_toe_t.setValue(w.toe_t or 0.0)
            self._set_filler(w.filler or "")
        finally:
            self._filling = False
        g = self.weld_geo(w)
        if "error" in g:
            txt = f"Cannot determine the weld geometry: {g['error']}"
        else:
            kind = {"fillet": "fillet", "butt": "butt"}.get(g["kind"], "?")
            txt = (f"{kind}, a = {fmt(g.get('a'), 3)} mm, L = {fmt(g.get('length'), 4)} mm"
                   + (f", angle {fmt(g.get('theta_deg'), 3)}°" if g.get("theta_deg") else "")
                   + f"<br>styk s: {', '.join(g['fusion']) or '–'} (z geometrie CAD)")
            if not g["fusion"]:
                txt += f"<br><span style='color:{RED}'>The weld does not touch any part!</span>"
        self.weld_info.setText(txt)
        self.show_detail(w.name)
        if self.active():
            self.win.view.set_selected_faces(self.part_faces([w.part]))

    def _set_filler(self, name: str):
        """Filler material into the menu. An unknown name (e.g. from a file of a newer version) is shown only temporarily
        as the last item "unknown: …" – it disappears when another weld is selected; the menu does not grow permanently.
        """
        cb = self.w_filler
        cb.blockSignals(True)
        try:
            k = cb.count() - 1
            if k >= 0 and cb.itemData(k, Qt.UserRole + 1):        # temporary item of the previous weld
                cb.removeItem(k)
            k = cb.findData(name)
            if k < 0:
                cb.addItem(f"unknown: {name} (not in the table – checked as matching)", name)
                k = cb.count() - 1
                cb.setItemData(k, True, Qt.UserRole + 1)
            cb.setCurrentIndex(k)
        finally:
            cb.blockSignals(False)

    def save_weld(self):
        w = self.current_weld()
        if w is None or self._filling:
            return
        old, mesh_ok = w.to_dict(), self._mesh_valid()
        w.a = self.w_a.value() or None
        w.segments = self.w_seg.value()
        w.fat_sigma, w.fat_tau = self.w_fs.value(), self.w_ft.value()
        w.sn_curve = self.w_curve.currentText()
        w.gamma_Mf = self.w_gamma.value()
        w.open_root = self.w_open.isChecked()
        w.enabled = self.w_enabled.isChecked()
        w.toe = self.w_toe.isChecked()
        txt = self.w_fat_toe.currentText()
        k = self.w_fat_toe.findText(txt)
        try:
            w.fat_toe = float(self.w_fat_toe.itemData(k)) if k >= 0 else float(txt.split()[0].replace(",", "."))
        except (ValueError, IndexError):
            pass
        w.toe_t = self.w_toe_t.value() or None
        w.filler = self.w_filler.currentData() or ""
        row = self.weld_list.currentRow()
        self._apply_change(change_level("weld", old, w.to_dict(), self.win.study.joints.preload_case), mesh_ok)
        self.weld_list.setCurrentRow(row)

    def delete_weld(self):
        w = self.current_weld()
        if w is None:
            return
        self.win.study.welds.remove(w)
        self.weld_form.setVisible(False)
        self.win.changed(invalidate_mesh=True)

    def mark_weld(self):
        from .main import msg
        st = self.win.study
        sel = self.win.view.selected_faces()
        names = list(dict.fromkeys(self.win.parts[pi].name for pi, _ in sel))
        if not names:
            msg(self.win, "Joints", "Select a face of the weld body in 3D.")
            return
        excl = st.excluded_parts()
        known = {w.part for w in st.welds}
        added = 0
        for n in names:
            if n in excl:
                self.win.log_msg(f"'{n}' is part of a bolt – cannot be marked as a weld")
                continue
            if n in known:
                continue
            k = len(st.welds) + 1
            while any(w.name == f"Weld {k}" for w in st.welds):
                k += 1
            st.welds.append(WeldSpec(f"Weld {k}", n))
            added += 1
        self.win.view.set_selected_faces([])
        if added:
            self.win.changed(invalidate_mesh=True)

    # ================================================================== settings
    def on_settings(self, *_):
        st = self.win.study
        if st is None or self._filling:
            return
        js = st.joints
        new = (self.s_cat.currentData(), self.s_mu.value(), self.s_g2.value(), self.s_pre.isChecked(),
               self.s_split.currentData())
        if new == (js.category, js.slip_mu, js.gamma_M2, js.preload_case, js.shear_split):
            return
        fe = new[3] != js.preload_case
        mesh_ok = self._mesh_valid()
        js.category, js.slip_mu, js.gamma_M2, js.preload_case, js.shear_split = new
        # the preload state changes the FEM model (beam section, extra step), not the mesh; the others only the assessment
        self._apply_change("fe" if fe else "assess", mesh_ok)

    # ================================================================== assessment
    def on_results(self):
        self.assessment = None
        self.compute(force=True)

    def compute(self, force=False):
        win = self.win
        if win.study is None or win.results is None or not win.results.steps or \
                not (win.study.active_bolts() or win.study.active_welds()):
            self.assessment = None
            self.refresh()
            return
        if self.assessment is not None and not force:
            return
        from fea_core.joints.assess import assess_joints
        try:
            self.assessment = assess_joints(win.study, win.results, win.parts)
        except Exception as e:  # noqa: BLE001
            self.assessment = None
            win.log_msg(f"Joint check ERROR: {e}")
        if self.assessment is not None:
            bad = [c for c in self.assessment.all() if not c.ok]
            win.log_msg(f"Joint assessment: {len(self.assessment.bolts)} bolts, {len(self.assessment.welds)} welds"
                        + (f", FAILS: {', '.join((c.name for c in bad))}" if bad else ", all pass"))
            for w in self.assessment.warnings:
                win.log_msg(f"Warning: {w}")
        self.refresh()
        win.refresh_tree()
        win.refresh_flow()

    def show_detail(self, name):
        a = self.assessment
        c = next((x for x in a.all() if x.name == name), None) if a else None
        if c is None:
            msg = ("The check will be available after the analysis." if self.win.results is None else
                   "The joint has no results (recompute the study).")
            self.detail.setHtml(f"<i>{msg}</i>")
            self.detail.setVisible(self.win.results is not None)
            return
        self.detail.setVisible(True)
        rows = [f"<b>{escape(str(c.name))}</b> – {escape(str(c.label))}<br>Utilization <b style='color:{util_color(c.utilization)}'>{fmt(c.utilization, 3)}</b>, governing: {escape(str(c.governing))}",
                "<table cellspacing=3>"]
        for name_, u, txt in c.rows:
            info = name_.startswith("(informative)")
            col = GREY if info else util_color(u)
            rows.append(f"<tr><td align=right style='color:{col}'><b>{fmt(u, 3)}</b></td>"
                        f"<td>{escape(str(name_))}<br><small>{escape(str(txt))}</small></td></tr>")
        rows.append("</table>")
        if c.warnings:
            rows.append("<span style='color:#b36b00'>! " + "<br>! ".join(escape(str(w)) for w in c.warnings) + "</span>")
        if self.assessment.cases:
            rows.append(f"<small>Load cases: {escape(', '.join(str(x) for x in self.assessment.cases))}"
                        + ("; fatigue from load history" if self.assessment.fatigue else "; no fatigue (no load history)")
                        + ". Loads are taken as design loads (γ_F in the input).</small>")
        self.detail.setHtml("".join(rows))

    def select_joint(self, kind, row):
        self.win.set_stage(self.win.STAGE["Joints"])
        if kind == "bolt":
            self.bolt_list.setCurrentRow(row)
        else:
            self.weld_list.setCurrentRow(row)

    # ================================================================== 3D
    def part_faces(self, names) -> list[tuple[int, int]]:
        v = self.win.view
        out = []
        for pi, p in enumerate(self.win.parts):
            if p.name in names:
                out += [(pi, fi) for fi in range(len(v.part_faces[pi]))]
        return out

    def show_3d(self):
        """Bolts and welds in colour (by utilisation, if assessed), bolt axes with labels."""
        st = self.win.study
        v = self.win.view
        if st is None:
            return
        chk = {c.name: c for c in self.assessment.all()} if self.assessment else {}
        L = 0.04 * v.model_size()
        for b in st.bolts:
            c = chk.get(b.name)
            col = util_rgb(c.utilization) if c else (C_BOLT if b.enabled else (150, 150, 150))
            faces = self.part_faces(b.cad_parts)
            key = f"j:{b.name}"
            if faces:
                v.set_face_set(key, faces, col, label=b.name, transparency=0.15)
            else:
                v.remove_face_set(key)
            p0, p1 = b.point(-L), b.point(b.beam_end() + L)
            v.add_lines(key, [(p0, p1)], col, 3.0, label=None if faces else b.name)
        for w in st.welds:
            c = chk.get(w.name)
            col = util_rgb(c.utilization) if c else (C_WELD if w.enabled else (150, 150, 150))
            faces = self.part_faces([w.part])
            if faces:
                v.set_face_set(f"j:{w.name}", faces, col, label=w.name, transparency=0.1)


SOURCE_TXT = {"name": "name", "shape": "shape", "color": "CAD color", "layer": "CAD layer", "attribute": "CAD attribute"}


class DetectDialog(QDialog):
    """Confirmation of the recognised joints (chapter 13 of the spec): each proposal with its source, certainty and reason.
    Bolts by name and welds by rules are preselected; by shape only with certainty ≥ 0.7.
    """

    def __init__(self, parent, suggestions, rejected=()):
        super().__init__(parent)
        self.setWindowTitle("Detected joints – confirmation")
        self.resize(980, 460)
        self.sugg = list(suggestions)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Checked joints are added to the study. Check the detection by shape and color (selecting a row highlights the parts in 3D)."))
        self.table = QTableWidget(len(self.sugg), 5)
        self.table.setHorizontalHeaderLabels(["Add", "Joint", "Source", "Confidence", "Reason"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        help_tip(self.table, "Joint suggestions: ticked ones are added; “By” = recognition source",
                 "Joint suggestions: ticked ones in the first column are added. “By” = recognition source (name,<br>shape, colour / layer / CAD attribute), confidence 0–1 (green ≥ 0.85, orange ≥ 0.7,<br>red lower). Selecting a row highlights the parts in 3D.")
        self.table.setWhatsThis("<b>Confirm detected joints</b><p>Pre-checked are bolts by the standard in the name, welds by the rules (name, layer, attribute, color) and bolts by shape with a confidence of at least 0.7. The “Reason” column explains why the part was recognized.</p><p>Check the detection by shape and color in 3D (selecting a row highlights the parts) and uncheck unwanted proposals.</p>")
        for r, s in enumerate(self.sugg):
            it = QTableWidgetItem("")
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if s.accept else Qt.Unchecked)
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QTableWidgetItem(s.label))
            self.table.setItem(r, 2, QTableWidgetItem(SOURCE_TXT.get(s.source, s.source)))
            c = QTableWidgetItem(decimal(f"{s.confidence:.2f}"))
            c.setForeground(QBrush(QColor(GREEN if s.confidence >= 0.85 else (ORANGE if s.confidence >= 0.7 else RED))))
            self.table.setItem(r, 3, c)
            self.table.setItem(r, 4, QTableWidgetItem(s.reason))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table, 1)
        if rejected:
            info = QLabel("<small>Not accepted (bolt-like shape but without a joint): " + "<br>".join(escape(str(x)) for x in rejected[:8])
                          + ("<br>…" if len(rejected) > 8 else "") + "</small>")
            info.setWordWrap(True)
            lay.addWidget(info)
        hb = QHBoxLayout()
        for txt, val, tip in (("Select all", True, "Check all proposals"),
                              ("Clear selection", False, "Uncheck all proposals")):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, v=val: self.set_all(v))
            hb.addWidget(b)
        hb.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Add selected")
        bb.button(QDialogButtonBox.Ok).setToolTip("Add the checked bolts and welds to the study (new mesh)")
        bb.button(QDialogButtonBox.Cancel).setToolTip("Close without adding joints")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        hb.addWidget(bb)
        lay.addLayout(hb)

    def set_all(self, on: bool):
        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def selected(self) -> list:
        return [s for r, s in enumerate(self.sugg) if self.table.item(r, 0).checkState() == Qt.Checked]


def exec_dialog(dlg) -> bool:
    """Modal dialog (the GUI test replaces it – snapshot and confirmation without waiting for the user)."""
    return dlg.exec() == QDialog.Accepted
