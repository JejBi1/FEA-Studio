"""Fatigue panel: load curve, settings, calculation, damage map and the worst places."""
from __future__ import annotations

import numpy as np
from html import escape
from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QMessageBox,
                               QPushButton, QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from fea_core.fatigue.history import Channel, read_csv_points
from fea_core.fatigue.settings import (CP_CRITERION_LABELS, CP_RATIO_DUCTILE, EQUIVALENT_LABELS, MEAN_LABELS,
                                       NOTCH_RULE_LABELS, FatigueSettings)
from fea_core.fatigue.sn import RELIABILITY, SURFACE_LABELS

from .safe_browser import SafeTextBrowser
from .legend import Legend
from . import theme
from .widgets import LazyPlot, Section, decimal, fit_rows, fmt, help_tip, panel_header, scroll_body, set_placeholder
from .workers import TaskThread

EQUIVALENT_SHORT = {"signed_mises": "signed von Mises", "max_principal": "max. principal stress",
                    "critical_plane": "critical plane"}
CP_CRITERION_SHORT = {"auto": "by material", "findley": "Findley", "normal": "normal stress"}
METHOD_SHORT = {"sn": "S-N (stress-life, high-cycle)", "en": "ε-N (strain-life, low-cycle)"}
# mean stress corrections valid for each method (fatigue.analysis.run_fatigue rejects the others)
MEAN_OF_METHOD = {"sn": ("none", "goodman", "gerber"), "en": ("none", "morrow", "swt")}
NOTCH_SHORT = {"neuber": "Neuber", "glinka": "Glinka (ESED)"}
QUANTITIES = [("Damage D over design life (log₁₀)", "–"),
              ("Life [blocks] (log₁₀)", "–"),
              ("Fatigue safety factor (stress multiplier)", "–"),
              ("Equivalent stress amplitude", "MPa")]


def num_edit(val, tip="", long=""):
    e = QLineEdit(f"{val:g}")
    v = QDoubleValidator()
    v.setNotation(QDoubleValidator.ScientificNotation)
    e.setValidator(v)
    help_tip(e, tip, long)
    return e


def num(e: QLineEdit, default=0.0):
    try:
        return float(e.text().replace(",", "."))
    except ValueError:
        return default


class Plot(LazyPlot):
    """Small chart – course of the multiples / stress at a node."""
    FIGSIZE = (4, 2.2)
    MIN_HEIGHT = 190

    def clear(self):
        if self.canvas is not None:
            self.fig.clear()
            self.canvas.draw_idle()

    def draw(self, t, series: dict, ylabel: str, title: str = ""):
        ax = self.axes()
        for name, y in series.items():
            ax.plot(t, y, lw=1.0, label=name)
        ax.set_xlabel("time in block")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=9)
        if len(series) > 1:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        self.canvas.draw_idle()


class FatiguePanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.result = None
        self._thread = None
        self._filling = False
        self.scroll, lay = scroll_body(self)
        lay.addWidget(panel_header("Fatigue", "Stress = Σ multiplier(t) × load case (superposition), rainflow, Miner"))

        # ---------------------------------------------------------- curve
        g = Section("Load history (one block)", key="fatigue.curve", collapsed=False,
                    tip="Defines the load history and its repetition within one block.",
                    help="<b>Load history</b><p>Each channel multiplies the stress of the selected load case over time. The channels are summed and the block is repeated according to the design number of blocks.</p>")
        v = QVBoxLayout(g.body)
        self.ch_list = QListWidget()
        set_placeholder(self.ch_list, "No channel yet – add one below")
        fit_rows(self.ch_list, 2, 5)
        self.ch_list.setToolTip("Load history channels (superposition)")
        self.ch_list.setWhatsThis(
            "<b>Load history channels</b><p>Each static load case serves as a unit case. A channel specifies the multiplier applied to its stresses over the block time (sine or points). The stress at a node at time t is the sum of multipliers × stresses of the cases (linear superposition).</p><p>Procedure: select a case, history and values, “Add channel”; repeat for another case. The phase between sines determines whether the loading is proportional (same phase) or non-proportional – then use the critical plane method. A constant multiplier (amplitude 0, mean 1) gives mean stress, e.g. from bolt preload.</p>")
        self.ch_list.currentRowChanged.connect(self.on_channel_row)
        v.addWidget(self.ch_list)
        f = QFormLayout()
        self.ch_lc = QComboBox()
        help_tip(self.ch_lc, "Load case whose stress the channel multiplies",
                 "Load case whose stress the channel multiplies.\n“Bolt preload” with constant multiplier 1 (amplitude 0, mean 1) = mean stress from preload.")
        self.ch_kind = QComboBox()
        self.ch_kind.addItems(["Sine", "Points (table / CSV)"])
        help_tip(self.ch_kind, "History type: sine or points",
                 "Sine: multiplier = mean + amplitude·sin(2π·oscillations·t/T + phase), T = block length.\nPoints: table (time, multiplier) with linear interpolation between points, or CSV import.")
        self.ch_kind.currentIndexChanged.connect(lambda i: self.kind_stack.setCurrentIndex(i))
        f.addRow("Load case", self.ch_lc)
        f.addRow("Waveform", self.ch_kind)
        v.addLayout(f)
        self.kind_stack = QStackedWidget()
        w = QWidget()
        fs = QFormLayout(w)
        fs.setContentsMargins(0, 0, 0, 0)
        self.s_amp = num_edit(1.0, "Case multiplier amplitude")
        self.s_mean = num_edit(0.0, "Mean value of the multiplier")
        self.s_cyc = num_edit(1, "Number of sine cycles in one block",
                              "Number of oscillations per block. Repetition is set by the design number of blocks – oscillations within a block increase the number of time points (24 per oscillation) and slow down the computation")
        self.s_phase = num_edit(0.0, "Phase shift [°]")
        for lbl, e in (("Amplitude", self.s_amp), ("Mean", self.s_mean), ("Cycles per block", self.s_cyc),
                       ("Phase [°]", self.s_phase)):
            fs.addRow(lbl, e)
        self.kind_stack.addWidget(w)
        w = QWidget()
        vp = QVBoxLayout(w)
        vp.setContentsMargins(0, 0, 0, 0)
        self.pt_table = QTableWidget(0, 2)
        self.pt_table.setHorizontalHeaderLabels(["time", "multiplier"])
        self.pt_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        set_placeholder(self.pt_table, "No history points entered yet.")
        fit_rows(self.pt_table, 2, 8)
        help_tip(self.pt_table, "History points: time and multiplier, linear in between",
                 "History points: time and case multiplier, linear between points (sorted by time).\nDouble-click = edit; decimal comma or point. At least 2 points.\nRainflow is independent of rate – only the order of peaks matters.")
        vp.addWidget(self.pt_table)
        h = QHBoxLayout()
        for txt, slot, tip in (
                ("+ row", self.add_row, "Add a point at the end of the table"),
                ("− row", self.del_row, "Delete selected table row"),
                ("Import CSV…", self.import_csv,
                 "Load points from CSV: two columns (time; multiplier) or one\nTwo columns (time; multiplier) or one (multipliers, time = order).\nSeparator ; or tab (then decimal comma), or comma with decimal point; the header is skipped.")):
            b = QPushButton(txt)
            short, _, rest = tip.partition("\n")
            help_tip(b, short, tip) if rest else b.setToolTip(tip)
            b.clicked.connect(slot)
            h.addWidget(b)
        vp.addLayout(h)
        self.kind_stack.addWidget(w)
        v.addWidget(self.kind_stack)
        h = QHBoxLayout()
        for txt, slot, tip in (
                ("Add channel", self.add_channel, "Add a new channel from the form (case + waveform)"),
                ("Save channel", self.save_channel,
                 "Overwrite the selected channel with the values from the form (with no selection = add a new one)"),
                ("Delete", self.delete_channel, "Delete selected channel")):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            h.addWidget(b)
        v.addLayout(h)
        f = QFormLayout()
        self.repeats = num_edit(1e6, "How many times the block repeats over the life")
        self.repeats.editingFinished.connect(self.on_settings)
        self.duration = num_edit(1.0, "Block length (sine time axis)")
        self.duration.editingFinished.connect(self.on_settings)
        f.addRow("Design number of blocks", self.repeats)
        f.addRow("Block length", self.duration)
        v.addLayout(f)
        self.plot = Plot()
        self.plot.setVisible(False)
        v.addWidget(self.plot)
        lay.addWidget(g)

        # ---------------------------------------------------------- method and material
        g = Section("Method and material", key="fatigue.method", collapsed=False,
                    tip="Selects the fatigue method, material and S-N curve.",
                    help="<b>Method and material</b><p>Selects the multiaxial stress conversion, mean stress correction, Marin factors and the S-N curve of the material.</p>")
        f = QFormLayout(g.body)
        self.method = QComboBox()
        for k, lbl in METHOD_SHORT.items():
            self.method.addItem(lbl, k)
        help_tip(self.method, "Fatigue method: S-N (stress-life) or ε-N (strain-life)",
                 "S-N: stress amplitude against the S-N curve with Marin factors – high-cycle fatigue (elastic stresses).\n"
                 "ε-N: local elastic-plastic strain from the elastic FE stress (Neuber / Glinka) on the cyclic\n"
                 "Ramberg-Osgood curve and Coffin-Manson-Basquin life – low-cycle fatigue with local yielding.\n"
                 "ε-N parameters are estimated from Rm (Bäumel-Seeger uniform material law) – verify by tests;\n"
                 "measured parameters only via the Python API / study file (FatigueSettings.en_params). Not for 3D printing.")
        self.method.currentIndexChanged.connect(self.on_method)
        self.notch = QComboBox()
        for k, lbl in NOTCH_RULE_LABELS.items():
            self.notch.addItem(NOTCH_SHORT[k], k)
            self.notch.setItemData(self.notch.count() - 1, lbl, Qt.ToolTipRole)
        help_tip(self.notch, "ε-N: conversion of the elastic stress to the local elastic-plastic strain",
                 "Neuber (σ·ε = σ_el²/E) – the usual, more conservative choice;\nGlinka (equal strain energy density) – lower local strains.")
        self.notch.currentIndexChanged.connect(self.on_settings)
        self.eq = QComboBox()
        for k, lbl in EQUIVALENT_LABELS.items():
            self.eq.addItem(EQUIVALENT_SHORT[k], k)
            self.eq.setItemData(self.eq.count() - 1, lbl, Qt.ToolTipRole)
        self.eq.setToolTip("Conversion of multiaxial stress to equivalent stress")
        self.eq.setWhatsThis(
            "<b>Multiaxiality (method)</b><p><i>Signed von Mises</i> = von Mises with the sign of the dominant principal stress; <i>max. principal stress</i> = principal stress with the largest absolute value. Both reduce the stress state at each time to a single number to which rainflow and the Palmgren–Miner rule are applied – valid for proportional loading (principal directions do not change).</p><p>For pure shear (σ1 = −σ3) the sign is ambiguous and for non-proportional loading (different channel phases) the principal directions rotate – then choose the <i>critical plane</i>: rainflow on many planes (Findley or normal stress), the plane with the most damage governs.</p>")
        self.mean = QComboBox()
        for k, lbl in MEAN_LABELS.items():
            self.mean.addItem(lbl, k)
        help_tip(self.mean, "Mean stress correction of the cycle (Goodman, Gerber)",
                 "Correction for tensile mean stress Sm of a cycle (needs the material Rm):\nGoodman Sa/(1 − Sm/Rm) – more conservative; Gerber Sa/(1 − (Sm/Rm)²);\ncompressive mean stress is not credited favorably. Not used for the Findley\ncriterion (mean stress is in the k·σn,max term).")
        self.closed = QCheckBox("block repeats (close rainflow residue)")
        help_tip(self.closed, "Repeating block: all cycles are closed",
                 "On: the block repeats – the history starts and ends at the largest |value|, so all cycles are closed\n(including the largest range across block boundaries).\nOff: one-time history, the rainflow residue is counted as half cycles.")
        self.surface = QComboBox()
        for k, lbl in SURFACE_LABELS.items():
            self.surface.addItem(lbl, k)
        help_tip(self.surface, "Part surface → Marin factor ka",
                 "Part surface → Marin factor ka = a·Rm^b (Shigley, Table 6-2).\nPolished = 1 (test specimen); a rougher surface lowers the endurance limit, more so for high-strength materials.\nNot used for 3D printing.")
        self.rel = QComboBox()
        for r in RELIABILITY:
            self.rel.addItem(f"{r:g} %", r)
        help_tip(self.rel, "Required reliability → factor ke",
                 "Required reliability → factor ke (Shigley, Table 6-5):\n50 % = 1 (mean S-N curve), 90 % = 0.897, 99 % = 0.814, 99.99 % = 0.702.")
        self.size_d = QDoubleSpinBox()
        self.size_d.setRange(0, 1000)
        self.size_d.setSuffix(" mm")
        self.size_d.setSpecialValueText("ignore")
        help_tip(self.size_d, "Effective cross-section diameter for the size factor kb",
                 "Effective cross-section diameter for size factor kb (Shigley):\nkb = 1.24·d^−0.107 for 2.79–51 mm, 1.51·d^−0.157 above 51 mm (up to 254 mm).\n0 = ignore (kb = 1). Not used for 3D printing.")
        self.k_misc = QDoubleSpinBox()
        self.k_misc.setRange(0.05, 2.0)
        self.k_misc.setDecimals(3)
        self.k_misc.setSingleStep(0.05)
        help_tip(self.k_misc, "Other influences kf on the fatigue limit (1 = none)",
                 "Other effects kf – additional factor on the endurance limit (corrosion, coatings, experience…), 1 = no effect.\nThis is not a notch factor – notches are already included in the FE stress.")
        # critical plane – parameters (visible only for this method)
        self.cp_box = Section("Critical plane", key="fatigue.critical_plane", collapsed=True,
                              tip="Sets the critical plane search for fatigue.",
                              help="<b>Critical plane</b><p>Sets the criterion and the density of the plane search. Used only when the critical plane method is selected.</p>")
        fc = QFormLayout(self.cp_box.body)
        self.cp_crit = QComboBox()
        for k_, lbl in CP_CRITERION_LABELS.items():
            self.cp_crit.addItem(CP_CRITERION_SHORT[k_], k_)
            self.cp_crit.setItemData(self.cp_crit.count() - 1, lbl, Qt.ToolTipRole)
        help_tip(self.cp_crit, "Critical plane criterion: Findley or normal stress",
                 "Findley (1959): τa + k·σn,max on the plane – ductile materials (shear cracks).\nNormal stress: rainflow of σn on the plane – brittle materials (gray cast iron).\nBy material: gray cast iron → normal stress, others → Findley.")
        self.cp_ratio = QLineEdit()
        self.cp_ratio.setPlaceholderText(f"default √3 = {CP_RATIO_DUCTILE:.3f}")
        help_tip(self.cp_ratio, "Fatigue limit ratio f₋₁/t₋₁ for the k of the Findley criterion",
                 "Ratio of fully reversed tension / fully reversed torsion endurance limits f₋₁/t₋₁ of the material selected for S-N (1 < s ≤ 2).\nDetermines k of the Findley criterion: k = (2 − s)/(2√(s − 1)); √3 = von Mises (ductile steels, tests ~1.6–1.8). Empty = default.")
        self.cp_ratio.editingFinished.connect(self.on_cp_ratio)
        self.cp_planes = QSpinBox()
        self.cp_planes.setRange(20, 2000)
        self.cp_planes.setSingleStep(20)
        help_tip(self.cp_planes, "Number of coarse-grid planes on the hemisphere",
                 "Number of coarse-grid planes on the hemisphere (100 ≈ 14° spacing); candidates are refined by 4 levels to ~1°")
        self.cp_dirs = QSpinBox()
        self.cp_dirs.setRange(4, 72)
        self.cp_dirs.setToolTip("Number of shear directions in the plane (per 180°) for the Findley criterion; the best one is refined")
        self.cp_thorough = QCheckBox("thorough search (4× more planes, more candidates; ~3× slower)")
        help_tip(self.cp_thorough, "Thorough search: 4× more planes, 6 candidates",
                 "4× more coarse-grid planes and 6 candidates to refine instead of 3.\nTo check whether the normal search missed a worse plane.")
        self.cp_info = QLabel("")
        self.cp_info.setWordWrap(True)
        for lbl, wdg in (("Criterion", self.cp_crit), ("f₋₁/t₋₁ (S-N material)", self.cp_ratio),
                         ("Planes (coarse grid)", self.cp_planes), ("Shear directions", self.cp_dirs), ("", self.cp_thorough),
                         ("", self.cp_info)):
            fc.addRow(lbl, wdg)
        self.cp_crit.currentIndexChanged.connect(self.on_settings)
        self.cp_planes.valueChanged.connect(self.on_settings)
        self.cp_dirs.valueChanged.connect(self.on_settings)
        self.cp_thorough.toggled.connect(self.on_settings)
        f.addRow("Fatigue method", self.method)
        f.addRow("Notch rule (ε-N)", self.notch)
        for lbl, wdg in (("Multiaxiality (method)", self.eq), ("", self.cp_box), ("Mean stress", self.mean),
                         ("", self.closed), ("Surface (ka)", self.surface), ("Reliability (ke)", self.rel),
                         ("Diameter for kb", self.size_d), ("Other effects kf", self.k_misc)):
            f.addRow(lbl, wdg)
        for wdg in (self.eq, self.mean, self.surface, self.rel):
            wdg.currentIndexChanged.connect(self.on_settings)
        self.closed.toggled.connect(self.on_settings)
        self.size_d.valueChanged.connect(self.on_settings)
        self.k_misc.valueChanged.connect(self.on_settings)
        # S-N of the material
        self.sn_mat = QComboBox()
        help_tip(self.sn_mat, "Material for which the S-N curve is entered",
                 "Material whose S-N curve is displayed and entered (and ratio f₋₁/t₋₁ for the critical plane).\nOnly materials of meshed parts; a node shared by parts takes the worse S-N.")
        self.sn_mat.currentIndexChanged.connect(self.fill_sn)
        self.sn_own = QCheckBox("custom S-N (otherwise estimate from Rm)")
        help_tip(self.sn_own, "Own S-N curve instead of the estimate from Rm",
                 "On: S-N entered below (SD, ND, k, k2) or taken from PragTic.\nOff: estimate from Rm (Shigley: steel 0.5·Rm at 10⁶, Al 0.4·Rm at 5·10⁸, plastic only approximate),\nfor 3D printing the print S-N. Marin factors are applied in both cases.")
        self.sn_own.toggled.connect(self.on_sn)
        self.sn_SD = num_edit(100, "Amplitude at the knee point [MPa] (without Marin factors)")
        self.sn_ND = num_edit(1e6, "Cycles at knee point")
        self.sn_k = num_edit(5, "Slope above knee")
        self.sn_k2 = num_edit(0, "Slope below knee (0 = endurance limit, Haibach 2k−1)")
        for e in (self.sn_SD, self.sn_ND, self.sn_k, self.sn_k2):
            e.editingFinished.connect(self.on_sn)
        f.addRow("S-N for material", self.sn_mat)
        h = QHBoxLayout()
        h.addWidget(self.sn_own, 1)
        b = self.sn_db = QPushButton("From PragTic database…")
        help_tip(b, "Fatigue properties from the PragTic database with sources",
                 "Fatigue properties from the literature (pragtic.com, J. Papuga): endurance limits, Basquin, Manson-Coffin, static values – with sources")
        b.clicked.connect(self.sn_from_db)
        h.addWidget(b)
        f.addRow("", h)
        h = QHBoxLayout()
        for lbl, e in (("SD", self.sn_SD), ("ND", self.sn_ND), ("k", self.sn_k), ("k2", self.sn_k2)):
            h.addWidget(QLabel(lbl))
            h.addWidget(e)
        f.addRow(h)
        self.sn_info = QLabel("")
        self.sn_info.setWordWrap(True)
        f.addRow(self.sn_info)
        lay.addWidget(g)

        # ---------------------------------------------------------- calculation and results
        self.btn = QPushButton("Run fatigue analysis")
        theme.set_primary(self.btn)
        self.btn.setToolTip("Computes fatigue from the superposition of static results")
        self.btn.setWhatsThis(
            "<b>Compute fatigue</b><p>At each node, the stress history over the block is assembled from load case stresses and channel multipliers, converted to equivalent stress (or stresses on planes), rainflow (ASTM E1049) counts the cycles, the mean stress correction converts them to amplitudes and the Palmgren–Miner rule sums the damage per the S-N curve with the Marin factors included.</p><p>Result: damage D for the design number of blocks (D ≥ 1 = failure), life in blocks and fatigue safety factor (stress multiple at which D = 1). The result is not saved – any settings change discards it, recomputation is fast.</p>")
        self.btn.clicked.connect(self.compute)
        lay.addWidget(self.btn)
        g = Section("Fatigue results", key="fatigue.results", collapsed=False,
                    tip="Shows the fatigue map, the worst nodes and calculation details.",
                    help="<b>Fatigue results</b><p>Map of the selected quantity, table of the worst nodes and description of the method used for the report.</p>")
        v = QVBoxLayout(g.body)
        self.qty = QComboBox()
        for name, _ in QUANTITIES:
            self.qty.addItem(name)
        help_tip(self.qty, "Quantity shown in the fatigue map",
                 "Quantity in the map:\nlog₁₀ D – damage over the design number of blocks (D ≥ 1, i.e. log D ≥ 0 = failure);\nlog₁₀ of life in blocks; safety factor = stress multiplier at which D = 1 (displayed up to 10);\namplitude of the equivalent stress [MPa].")
        self.qty.currentIndexChanged.connect(lambda _: self.show_map())
        v.addWidget(self.qty)
        self.legend = Legend()
        self.legend.setMinimumHeight(260)
        self.legend.hide()
        v.addWidget(self.legend)
        self.worst = QTableWidget(0, 7)
        self.worst.setHorizontalHeaderLabels(["x", "y", "z", "Total D", "life [blk.]", "SF", "plane normal"])
        self.worst.horizontalHeaderItem(6).setToolTip("Critical plane: normal n (nx, ny, nz); purple segment in 3D")
        from .widgets import copy_menu
        copy_menu(self.worst)                     # Ctrl+C / context menu: the whole table
        hh = self.worst.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setStretchLastSection(True)
        self.worst.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.worst.setEditTriggers(QAbstractItemView.NoEditTriggers)
        set_placeholder(self.worst, "Results appear after the fatigue calculation.")
        fit_rows(self.worst, 2, 10)
        help_tip(self.worst, "10 worst nodes by damage",
                 "10 worst nodes by damage: coordinates [mm], D, life in blocks, safety factor.\nClick a row = marker in 3D and node stress history plot (for critical plane σn and in-plane shear).")
        self.worst.currentCellChanged.connect(lambda r, *_: self.on_worst(r))
        v.addWidget(self.worst)
        self.info = SafeTextBrowser()
        self.info.setMinimumHeight(220)
        self.info.setVisible(False)
        v.addWidget(self.info)
        lay.addWidget(g)
        lay.addStretch(1)

    # ================================================================== settings ↔ study
    def settings(self) -> FatigueSettings:
        st = self.win.study
        if st.fatigue is None:
            st.fatigue = FatigueSettings()
        return st.fatigue

    def case_names(self) -> list[str]:
        """States for the curve channels: load states + "Bolt preload" (Results step, if it is on) – a channel
        with a constant multiple of 1 gives the mean stress from the preload.
        """
        st = self.win.study
        if st is None:
            return []
        names = [lc.name for lc in st.load_cases]
        if st.joints.preload_case and st.active_bolts():
            from fea_core.joints.model import PRELOAD_CASE
            names.append(PRELOAD_CASE)
        return names

    def refresh(self):
        """Fill the controls from the study."""
        st = self.win.study
        if st is None:
            return
        self._filling = True
        try:
            fs = st.fatigue or FatigueSettings()
            cur = self.ch_lc.currentText()
            self.ch_lc.clear()
            self.ch_lc.addItems(self.case_names())
            if cur:
                self.ch_lc.setCurrentText(cur)
            self.ch_list.clear()
            for ch in fs.history.channels:
                self.ch_list.addItem(self.describe_channel(ch))
            self.repeats.setText(f"{fs.repeats:g}")
            self.duration.setText(f"{fs.history.duration:g}")
            self.method.setCurrentIndex(max(0, self.method.findData(fs.method)))
            self.notch.setCurrentIndex(max(0, self.notch.findData(fs.en_notch_rule)))
            self._apply_method(fs.method)
            self.eq.setCurrentIndex(max(0, self.eq.findData(fs.equivalent)))
            i_mean = self.mean.findData(fs.mean_correction)
            if i_mean >= 0:                    # otherwise the default of the method chosen by _apply_method stays
                self.mean.setCurrentIndex(i_mean)
            self.closed.setChecked(fs.closed_blocks)
            self.surface.setCurrentIndex(max(0, self.surface.findData(fs.surface)))
            self.rel.setCurrentIndex(max(0, self.rel.findData(fs.reliability)))
            self.size_d.setValue(fs.size_d or 0.0)
            self.k_misc.setValue(fs.k_misc)
            self.cp_crit.setCurrentIndex(max(0, self.cp_crit.findData(fs.cp_criterion)))
            self.cp_planes.setValue(int(fs.cp_planes))
            self.cp_dirs.setValue(int(fs.cp_dirs))
            self.cp_thorough.setChecked(bool(fs.cp_thorough))
            self.cp_box.setVisible(fs.critical_plane)
            mats = self.fatigue_materials()
            cur = self.sn_mat.currentText()
            self.sn_mat.clear()
            self.sn_mat.addItems(mats)
            if cur in mats:
                self.sn_mat.setCurrentText(cur)
        finally:
            self._filling = False
        self.fill_sn()
        self.plot_history()
        has_steps = self.win.results is not None and bool(self.win.results.steps)
        self.btn.setEnabled(has_steps and self._thread is None)
        if self.result is None:
            self.legend.hide()
            self.info.setHtml("")
            self.info.setVisible(False)

    def fatigue_materials(self) -> list[str]:
        """Materials of the parts that are meshed and assessed for fatigue: with results the parts of the mesh, otherwise the parts of the model without bolts,
        nuts and washers replaced by a beam and without the parts excluded from the calculation (Study.excluded_parts). A part
        without a material is skipped (the readiness check reports the error).
        """
        st = self.win.study
        res = self.win.results
        if res is not None and getattr(res, "mesh", None) is not None:
            names = list(res.mesh.part_names)
        else:
            excl = st.excluded_parts()
            names = [p.name for p in self.win.parts or [] if p.name not in excl]
        mats = set()
        for n in names:
            try:
                mats.add(st.material_of(n).name)
            except KeyError:
                continue
        return sorted(mats)

    @staticmethod
    def describe_channel(ch: Channel) -> str:
        if ch.kind == "sine":
            return (f"{ch.load_case}: sine, mean {fmt(ch.mean)}, amplitude {fmt(ch.amplitude)}, {fmt(ch.cycles)} oscillations, phase {fmt(ch.phase_deg)}°")
        return f"{ch.load_case}: {len(ch.points)} points"

    def on_settings(self, *_):
        if self._filling or self.win.study is None:
            return
        fs = self.settings()
        fs.repeats = max(num(self.repeats, 1.0), 1e-12)
        fs.history.duration = max(num(self.duration, 1.0), 1e-12)
        fs.method = self.method.currentData() or "sn"
        fs.en_notch_rule = self.notch.currentData() or "neuber"
        if fs.method == "en" and self.eq.currentData() == "critical_plane":
            self.eq.setCurrentIndex(max(0, self.eq.findData("signed_mises")))   # ε-N: no critical plane
        fs.equivalent = self.eq.currentData()
        fs.mean_correction = self.mean.currentData()
        if fs.method == "en":
            fs.en_mean_correction = fs.mean_correction
        fs.closed_blocks = self.closed.isChecked()
        fs.surface = self.surface.currentData()
        fs.reliability = float(self.rel.currentData())
        fs.size_d = self.size_d.value() or None
        fs.k_misc = self.k_misc.value()
        fs.cp_criterion = self.cp_crit.currentData()
        fs.cp_planes = self.cp_planes.value()
        fs.cp_dirs = self.cp_dirs.value()
        fs.cp_thorough = self.cp_thorough.isChecked()
        self.cp_box.setVisible(fs.critical_plane)
        self.changed()

    def _apply_method(self, method: str) -> None:
        """Controls valid for the method: mean stress corrections, notch rule, no critical plane and Marin factors for ε-N."""
        en = method == "en"
        keep = self.mean.currentData()
        old = self.mean.blockSignals(True)
        try:
            self.mean.clear()
            for k in MEAN_OF_METHOD["en" if en else "sn"]:
                self.mean.addItem(MEAN_LABELS[k], k)
            i = self.mean.findData(keep)
            self.mean.setCurrentIndex(i if i >= 0 else self.mean.findData("morrow" if en else "goodman"))
        finally:
            self.mean.blockSignals(old)
        self.notch.setEnabled(en)
        item = self.eq.model().item(self.eq.findData("critical_plane"))
        if item is not None:
            item.setEnabled(not en)
        for w in (self.surface, self.rel, self.size_d, self.k_misc, self.sn_own, self.sn_db):
            w.setEnabled(not en)

    def on_method(self, *_):
        if self._filling or self.win.study is None:
            return
        self._apply_method(self.method.currentData())
        self.on_settings()
        self.fill_sn()

    def on_cp_ratio(self):
        """Ratio f₋₁/t₋₁ for the material selected at S-N (empty = default by material class)."""
        if self._filling or self.win.study is None:
            return
        name = self.sn_mat.currentText()
        if not name:
            return
        fs = self.settings()
        txt = self.cp_ratio.text().strip()
        if not txt:
            if fs.cp_ratio.pop(name, None) is not None:
                self.changed()
            return
        val = num(self.cp_ratio, 0.0)
        if not 1.0 < val <= 2.0:
            QMessageBox.warning(self, "FEA Studio", "The ratio f₋₁/t₋₁ must be in the range (1; 2] – for brittle materials choose the normal stress criterion.")
            self.fill_sn()
            return
        if fs.cp_ratio.get(name) != val:
            fs.cp_ratio[name] = val
            self.changed()

    def fill_cp(self):
        """Label of the critical plane parameters for the material selected at S-N."""
        st = self.win.study
        name = self.sn_mat.currentText()
        fs = (st.fatigue if st is not None else None) or FatigueSettings()
        prev, self._filling = self._filling, True
        try:
            r = fs.cp_ratio.get(name)
            self.cp_ratio.setText(decimal(f"{r:g}") if r else "")
        finally:
            self._filling = prev
        if st is None or name not in st.materials:
            self.cp_info.setText("")
            return
        from fea_core.fatigue.analysis import cp_params, cp_text
        try:
            p = cp_params(st.materials[name], fs)
            self.cp_info.setText(f"{escape(str(name))}: {escape(str(cp_text(p)))}<br><i>{escape(str(fs.search_text()))}</i>")
        except ValueError as e:
            self.cp_info.setText(f"<span style='color:#d23b3b'>{escape(str(e))}</span>")

    def changed(self):
        self.result = None
        self.legend.hide()
        self.info.setVisible(False)
        self.worst.setRowCount(0)
        self.win.dirty = True
        self.win.push_undo()
        self.win.update_title()
        self.fill_sn()
        self.plot_history()

    # ================================================================== channels
    def channel_from_form(self) -> Channel | None:
        lc = self.ch_lc.currentText()
        if not lc:
            QMessageBox.warning(self, "FEA Studio", "The study has no load case.")
            return None
        if self.ch_kind.currentIndex() == 0:
            return Channel(lc, "sine", amplitude=num(self.s_amp, 1.0), mean=num(self.s_mean),
                           cycles=max(num(self.s_cyc, 1.0), 0.5), phase_deg=num(self.s_phase))
        pts = []
        for r in range(self.pt_table.rowCount()):
            try:
                t = float(self.pt_table.item(r, 0).text().replace(",", "."))
                fval = float(self.pt_table.item(r, 1).text().replace(",", "."))
            except (AttributeError, ValueError):
                continue
            pts.append([t, fval])
        if len(pts) < 2:
            QMessageBox.warning(self, "FEA Studio", "Enter at least 2 points (time, multiplier).")
            return None
        return Channel(lc, "points", points=sorted(pts))

    def add_channel(self):
        ch = self.channel_from_form()
        if ch:
            self.settings().history.channels.append(ch)
            self.refresh_channels(len(self.settings().history.channels) - 1)

    def save_channel(self):
        row = self.ch_list.currentRow()
        chans = self.settings().history.channels
        if not (0 <= row < len(chans)):
            return self.add_channel()
        ch = self.channel_from_form()
        if ch:
            chans[row] = ch
            self.refresh_channels(row)

    def delete_channel(self):
        row = self.ch_list.currentRow()
        chans = self.settings().history.channels
        if 0 <= row < len(chans):
            chans.pop(row)
            self.refresh_channels(min(row, len(chans) - 1))

    def refresh_channels(self, row):
        self.ch_list.clear()
        for ch in self.settings().history.channels:
            self.ch_list.addItem(self.describe_channel(ch))
        self.ch_list.setCurrentRow(row)
        self.changed()
        self.win.refresh_flow()

    def on_channel_row(self, row):
        st = self.win.study
        if st is None or st.fatigue is None or not (0 <= row < len(st.fatigue.history.channels)):
            return
        ch = st.fatigue.history.channels[row]
        self.ch_lc.setCurrentText(ch.load_case)
        self.ch_kind.setCurrentIndex(0 if ch.kind == "sine" else 1)
        self.s_amp.setText(f"{ch.amplitude:g}")
        self.s_mean.setText(f"{ch.mean:g}")
        self.s_cyc.setText(f"{ch.cycles:g}")
        self.s_phase.setText(f"{ch.phase_deg:g}")
        self.set_points(ch.points)

    def set_points(self, pts):
        self.pt_table.setRowCount(0)
        for t, f in pts:
            self.add_row(t, f)

    def add_row(self, t=None, f=None):
        r = self.pt_table.rowCount()
        if t is None:
            t = r * 1.0
            f = 0.0
        self.pt_table.insertRow(r)
        self.pt_table.setItem(r, 0, QTableWidgetItem(f"{t:g}"))
        self.pt_table.setItem(r, 1, QTableWidgetItem(f"{f:g}"))

    def del_row(self):
        r = self.pt_table.currentRow()
        if r >= 0:
            self.pt_table.removeRow(r)

    def import_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "Import load history", "", "CSV (*.csv *.txt);;All (*.*)")
        if not p:
            return
        try:
            pts = read_csv_points(p)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "FEA Studio", f"Cannot load CSV:\n{e}")
            return
        self.ch_kind.setCurrentIndex(1)
        self.set_points(pts)

    def plot_history(self):
        st = self.win.study
        if st is None or st.fatigue is None or not st.fatigue.history.channels:
            self.plot.clear()
            self.plot.setVisible(False)
            return
        lcs = self.case_names()
        try:
            t, F = st.fatigue.history.factors(lcs)
        except KeyError as e:
            self.info.setHtml(f"<span style='color:#d23b3b'>{e}</span>")
            self.info.setVisible(True)
            self.plot.clear()
            self.plot.setVisible(False)
            return
        used = [k for k in range(len(lcs)) if np.any(F[:, k])]
        if not used:
            self.plot.clear()
            self.plot.setVisible(False)
            return
        self.plot.setVisible(True)
        self.plot.draw(t, {lcs[k]: F[:, k] for k in used}, "case multiplier", "Load history")

    # ================================================================== S-N
    def fill_sn(self, *_):
        self.fill_cp()
        st = self.win.study
        name = self.sn_mat.currentText()
        if st is None or not name or name not in st.materials:
            self.sn_info.setText("")
            return
        from fea_core.fatigue.analysis import sn_for_material
        from fea_core.fatigue.sn import estimate_sn, SNCurve
        mat = st.materials[name]
        fs = st.fatigue or FatigueSettings()
        if fs.method == "en":
            self._fill_en(mat, fs)
            return
        own = fs.sn.get(name)
        self._filling = True
        try:
            self.sn_own.setChecked(own is not None)
            try:
                if own:
                    base = SNCurve(**own)
                elif mat.is_print:                      # printing: S-N of printing (printing.props), not the metal estimate
                    from fea_core.printing.fatigue import sn_for_print
                    import copy
                    fs0 = copy.deepcopy(fs)
                    fs0.sn.pop(name, None)
                    d, _ = sn_for_print(st, mat, fs0)
                    m0 = fs0.k_load * fs0.k_temp * fs0.k_misc
                    base = SNCurve(SD=d["SD"] / m0, ND=d["ND"], k=d["k"], k2=d["k2"], source=d["source"])
                else:
                    base = estimate_sn(mat.E, mat.Rm)
            except ValueError as e:
                self.sn_info.setText(f"<span style='color:#d23b3b'>{e}</span>")
                return
            for e, v in ((self.sn_SD, base.SD), (self.sn_ND, base.ND), (self.sn_k, base.k), (self.sn_k2, base.k2)):
                e.setText(f"{v:.4g}")
                e.setEnabled(own is not None)
        finally:
            self._filling = False
        if mat.is_print:
            from fea_core.printing.fatigue import sn_for_print
            d, info = sn_for_print(st, mat, fs)
            m = info["marin"]
            pi = info["print"]
            self.sn_info.setText(
                f"Used (3D printing, XY plane): SD = {fmt(d['SD'])} MPa at {fmt(d['ND'])} cycles, k = {fmt(d['k'], 3)}, k2 = {(fmt(d['k2'], 3) if d['k2'] > 0 else 'fatigue limit')}; perpendicular to layers SD_z = {fmt(pi['SD_z'])} MPa<br>Factors: kc {fmt(m['kc'], 3)} · kd {fmt(m['kd'], 3)} · kf {fmt(m['kf'], 3)} (surface and size are not used for printing)<br>{escape(str(pi['method']))}<br><i>{escape(str(d['source']))}</i>")
            return
        sn, m = sn_for_material(mat, fs)
        self.sn_info.setText(
            f"Used: SD = {fmt(sn.SD)} MPa at {fmt(sn.ND)} cycles, k = {fmt(sn.k, 3)}, k2 = {(fmt(sn.k2, 3) if sn.k2 > 0 else 'fatigue limit')}<br>Marin: ka {fmt(m['ka'], 3)} · kb {fmt(m['kb'], 3)} · ke {fmt(m['ke'], 3)} · kf {fmt(m['kf'], 3)} = {fmt(m['k'], 3)}<br><i>{base.source}</i>")

    def _fill_en(self, mat, fs) -> None:
        """ε-N: parameters used for the material (given in the study or estimated from Rm)."""
        from fea_core.fatigue.analysis import en_for_material
        for e in (self.sn_SD, self.sn_ND, self.sn_k, self.sn_k2):
            e.setEnabled(False)
        try:
            p = en_for_material(mat, fs)
        except ValueError as e:
            self.sn_info.setText(f"<span style='color:#d23b3b'>{escape(str(e))}</span>")
            return
        self.sn_info.setText(
            f"ε-N used: σ'f = {fmt(p.sigma_f_prime)} MPa, b = {fmt(p.b, 3)}, ε'f = {fmt(p.eps_f_prime, 3)}, c = {fmt(p.c, 3)}, "
            f"K' = {fmt(p.K_prime)} MPa, n' = {fmt(p.n_prime, 3)}, E = {fmt(p.E)} MPa<br>"
            f"The S-N curve and the Marin factors are not used by the ε-N method.<br><i>{escape(str(p.source))}</i>")

    def on_sn(self, *_):
        if self._filling or self.win.study is None:
            return
        name = self.sn_mat.currentText()
        fs = self.settings()
        if self.sn_own.isChecked():
            new = {"SD": num(self.sn_SD, 100), "ND": num(self.sn_ND, 1e6), "k": num(self.sn_k, 5),
                   "k2": num(self.sn_k2, 0)}
            old = fs.sn.get(name)
            if old and all(f"{old[k]:.4g}" == f"{v:.4g}" for k, v in new.items()):
                return                                  # field only left without a change – keep the source
            fs.sn[name] = {**new, "source": "user-defined"}
        else:
            fs.sn.pop(name, None)
        self.changed()

    def sn_from_db(self):
        st = self.win.study
        name = self.sn_mat.currentText()
        if st is None or name not in st.materials:
            QMessageBox.warning(self, "FEA Studio", "The study has no material assigned to the parts.")
            return
        from .pragtic_dialog import PragticDialog
        mat = st.materials[name]
        try:
            dlg = PragticDialog(name, mat.Rm, self)
        except FileNotFoundError as e:
            QMessageBox.warning(self, "FEA Studio", str(e))
            return
        if not dlg.exec():
            return
        self.settings().sn[name] = dlg.sn.to_dict()
        msg = f"Fatigue: S-N of material {name} from the PragTic database – {dlg.sn.source}"
        if dlg.take_static:
            from fea_core.fatigue import pragtic
            s = pragtic.summary(dlg.record)
            mat.Rm = s["Rm"] or mat.Rm
            mat.Re = s["Re"] or mat.Re
            mat.source = (mat.source + "; " if mat.source else "") + f"Rm, Re: PragTic #{dlg.record['id']}"
            msg += f"; Rm = {fmt(mat.Rm)} MPa, Re = {fmt(mat.Re)} MPa"
        self.win.log_msg(msg)
        self.changed()
        self.win.refresh_tree()

    # ================================================================== calculation
    def compute(self):
        win = self.win
        if win.results is None or not win.results.steps:
            QMessageBox.warning(self, "FEA Studio", "First run the static analysis (Solve step) – the results contain no mechanical load case.")
            return
        fs = self.settings()
        if not fs.history.channels:
            QMessageBox.warning(self, "FEA Studio", "Add at least one load history channel.")
            return
        from fea_core.study import Study
        self.btn.setEnabled(False)
        self.btn.setText("Computing fatigue…")
        win.log_msg("Fatigue: calculation (the first run compiles numba, may take ~20 s"
                    + (", critical plane up to ~1 min" if fs.critical_plane else "") + ")…")
        study, results = Study.from_dict(win.study.to_dict()), win.results

        def job(_t):
            from fea_core.fatigue.analysis import run_fatigue
            return run_fatigue(study, results)
        t = TaskThread(job, self)
        t.done.connect(self.on_done)
        t.failed.connect(self.on_failed)
        t.finished.connect(self.on_finished)
        self._thread = t
        t.start()

    def on_finished(self):
        self._thread = None
        self.btn.setText("Run fatigue analysis")
        self.btn.setEnabled(self.win.results is not None and bool(self.win.results.steps))

    def on_failed(self, text):
        self.win.log_msg(f"Fatigue ERROR: {text}")
        QMessageBox.critical(self, "FEA Studio", f"Fatigue analysis failed:\n\n{text}")

    def on_done(self, fr):
        self.result = fr
        self.win.log_msg(f"Fatigue analysis finished in {fr.time_s:.2f} s ({len(fr.t)} time points).")
        self.win.refresh_flow()
        self.win.refresh_tree()
        self.show_map()

    # ================================================================== display
    def field(self, qi):
        fr = self.result
        if qi == 0:
            D = fr.total_damage
            pos = D[D > 0]
            floor = pos.max() * 1e-6 if len(pos) else 1e-12   # 6 decades below the maximum, D = 0 → lower bound
            return np.log10(np.maximum(D, floor))
        if qi == 1:
            life = fr.life_blocks
            fin = life[np.isfinite(life)]
            cap = min(fin.max(), fin.min() * 1e12) if len(fin) else 1e12
            return np.log10(np.minimum(life, cap))
        if qi == 2:
            return np.minimum(fr.safety, 10.0)
        return fr.amplitude

    def show_map(self):
        win = self.win
        if self.result is None or win.results is None:
            return
        from .contour import band_edges
        m = win.results.mesh
        tri, _ = m.boundary_faces()
        surf = np.unique(tri)
        qi = self.qty.currentIndex()
        vals = self.field(qi)
        sv = vals[surf]
        edges = band_edges(float(sv.min()), float(sv.max()), 12)
        name, unit = QUANTITIES[qi]
        reverse = qi in (1, 2)                # short life / safety = red
        v = win.view
        v.show_parts(False)
        v.show_sets(False)
        v.set_mode("results")
        colors = v.show_contour(m.nodes, tri, vals, edges, name, reverse=reverse, show_scale=False)
        self.legend.set_data(name, edges, colors, lambda x: fmt(float(x)))
        self.legend.show()
        fr = self.result
        worst = fr.worst(m, 10)
        self.worst.setRowCount(len(worst))
        for r, w in enumerate(worst):
            for c, val in enumerate([*w["xyz"], w["total_damage"], w["life_blocks"], w["safety"]]):
                self.worst.setItem(r, c, QTableWidgetItem(fmt(float(val), 4)))
            self.worst.setItem(r, 6, QTableWidgetItem(self.normal_text(w["normal"])))
        self.worst.setColumnHidden(6, fr.normal is None)
        if worst:
            v.mark(worst[0]["xyz"], f"D = {fmt(worst[0]['total_damage'], 3)}", (170, 0, 0))
            self.mark_plane(worst[0])
        from scipy.spatial import cKDTree
        win._probe = (cKDTree(m.nodes[surf]), surf, sv, unit, name)     # values in the same order as the tree items
        win._probe_exact = None
        self.fill_info()

    @staticmethod
    def normal_text(n) -> str:
        if n is None:
            return ""
        n = np.asarray(n, float)
        if n[np.argmax(np.abs(n))] < 0:          # n and −n = the same plane → the largest component positive
            n = -n
        return "(" + "; ".join(decimal(f"{x:.3f}") for x in n) + ")"

    def mark_plane(self, w: dict):
        """Critical plane in 3D: purple segment in the direction of the normal (+ thinner in the direction of the shear)."""
        if w.get("normal") is None:
            return
        v = self.win.view
        p = np.asarray(w["xyz"], float)
        L = 0.06 * v.model_size()
        n = np.asarray(w["normal"], float)
        segs = [(p - n * L, p + n * L)]
        d = np.asarray(w.get("shear_dir") or (0, 0, 0), float)
        v.add_lines("_cp_plane", segs, (150, 0, 170), 3.0, "n")
        if np.any(d):
            v.add_lines("_cp_plane", [(p - d * 0.6 * L, p + d * 0.6 * L)], (0, 140, 170), 2.0, "τ")
        v._result_objs += v._sets.pop("_cp_plane", [])      # disappears with the results (clear_results)

    def fill_info(self):
        fr = self.result
        D = fr.total_damage
        i = int(np.argmax(D))
        verdict = "PASS" if D[i] < 1 else "<span style='color:#d23b3b'>FAIL</span>"
        rows = [f"<b>Max. damage:</b> {fmt(float(D[i]), 4)} ({verdict}, D &lt; 1)",
                f"Min. life: {fmt(float(fr.life_blocks[i]))} blocks (design {fmt(fr.repeats)})",
                f"Min. fatigue safety factor (stress multiplier): {fmt(float(np.min(fr.safety)), 3)}", "",
                "<b>Method</b> (state in the report):"]
        rows += [f"• {escape(str(s))}" for s in fr.method_lines()]
        if fr.normal is not None:
            w0 = fr.worst(self.win.results.mesh, 1)[0]
            rows.append(f"• Critical plane at the worst location: n = {self.normal_text(w0['normal'])} (purple in 3D; amplitude = equivalent amplitude of the normal stress)")
        rows.append("<br><b>S-N:</b>")
        for name, info in fr.sn_used.items():
            sn = info["sn"]
            rows.append(f"• {escape(str(name))}: SD = {fmt(sn['SD'])} MPa at {fmt(sn['ND'])}, k = {fmt(sn['k'], 3)}, k2 = {(fmt(sn['k2'], 3) if sn['k2'] > 0 else 'fatigue limit')} – {escape(str(sn['source']))}")
        if np.any(D == 0):
            rows.append(f"Nodes without damage (below the endurance limit): {fmt(100 * float(np.mean(D == 0)), 3)} % – at the lower bound in the log D map.")
        warn = ["The worst locations at sharp internal corners and at support edges are singularities – evaluate fatigue there using a refined mesh or a notch method, not the nodal maximum.",
                "Assessment by local stress of a smooth specimen; welds and bolts are assessed separately (FAT, VDI 2230)."]
        if any("PLA" in n or "PETG" in n for n in fr.sn_used):
            warn.append("Printed plastics: the linear model does not capture creep; the S-N curve is only a calibration estimate.")
        warn += [escape(str(w)) for w in fr.warnings]
        rows.append("<br><span style='color:#b36b00'>! " + "<br>! ".join(warn) + "</span>")
        self.info.setHtml("<br>".join(rows))
        self.info.setVisible(True)

    def on_worst(self, row):
        if self.result is None or row < 0:
            return
        w = self.result.worst(self.win.results.mesh, 10)
        if row >= len(w):
            return
        node = w[row]["node"]
        from fea_core.fatigue.analysis import node_history, node_plane_history
        from fea_core.fatigue import kernel
        fr = self.result
        self.plot.setVisible(True)
        if fr.normal is not None:             # critical plane: σn and shear in the critical plane
            t, series = node_plane_history(self.win.results, fr, node)
            self.plot.draw(t, series, "MPa", f"Node {node + 1}: plane n = {self.normal_text(w[row]['normal'])}, "
                           f"σekv,a = {fmt(w[row]['amplitude'])} MPa")
            self.win.view.mark(w[row]["xyz"], f"#{row + 1}", (20, 20, 20))
            self.mark_plane(w[row])
            return
        t, eq = node_history(self.win.results, fr.settings, node)
        cyc = kernel.cycles_of(eq, fr.settings.closed_blocks)
        self.plot.draw(t, {"equiv. stress": eq}, "MPa",
                       f"Node {node + 1}: {fmt(float(cyc[:, 2].sum()))} cycles, max. Sa = {fmt(float(cyc[:, 0].max() / 2) if len(cyc) else 0.0)} MPa")
        self.win.view.mark(w[row]["xyz"], f"#{row + 1}", (20, 20, 20))
