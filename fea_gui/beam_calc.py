"""Analytical beam calculator (Tools menu, fea_core.beams.analytic and ec3): simply supported / clamped / continuous beam
(graphs V, M, w, reactions, extremes), curved bar (Winkler–Bach), Euler buckling and the EN 1993-1-1 member assessment with manually
entered forces. No FEM – closed formulas (Macaulay, Clapeyron).
"""
from __future__ import annotations

import html

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QCompleter, QDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .safe_browser import SafeTextBrowser
from . import theme
from .widgets import LazyPlot, dspin, fmt, table

SUPPORTS = [("hinge", "pinned"), ("fixed", "fixed"), ("free end", "free"), ("sliding fixed", "guided")]
LOAD_KINDS = [("distributed q [N/mm]", "q"), ("force P [N]", "P"), ("moment C [N·mm]", "C")]
CASES = [("hinge – hinge (β = 1)", "pinned-pinned"), ("fixed – free (β = 2)", "fixed-free"),
         ("fixed – pinned (β = 0.7)", "fixed-pinned"), ("fixed – fixed (β = 0.5)", "fixed-fixed")]
CURVED = {"rectangle": ("b", "h"), "circle": ("d",), "tube": ("D", "t"), "T (flange inside)": ("b", "tf", "tw", "h")}


def _section_edit(tip):
    e = QLineEdit()
    e.setPlaceholderText("library profile, e.g. IPE 200 (empty = enter I, A)")
    e.setToolTip(tip)
    try:
        from fea_core.beams.sections import families
        names = [n for v in families().values() for n in v]
        c = QCompleter(names)
        c.setCaseSensitivity(Qt.CaseInsensitive)
        c.setFilterMode(Qt.MatchContains)
        e.setCompleter(c)
    except Exception:  # noqa: BLE001
        pass
    return e


def _get_section(text):
    from fea_core.beams.sections import get
    t = text.strip()
    return get(t) if t else None


class BeamCalcDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Beam calculator (analytical)")
        self.setModal(False)
        self.resize(760, 900)
        lay = QVBoxLayout(self)
        intro = QLabel("Closed-form formulas without FEM: beam (Macaulay), continuous beam (three-moment equation), curved member (Winkler–Bach), Euler buckling and EN 1993-1-1 member check. Sign convention: load and deflection positive downward, M positive = tension at the bottom.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.tabs = QTabWidget()
        self.tabs.setToolTip("Calculation type")
        lay.addWidget(self.tabs, 1)
        self._tab_beam()
        self._tab_curved()
        self._tab_euler()
        self._tab_ec3()
        b = QPushButton("Close")
        b.setToolTip("Close calculator")
        b.clicked.connect(self.close)
        h = QHBoxLayout()
        h.addStretch(1)
        h.addWidget(b)
        lay.addLayout(h)

    # ------------------------------------------------------------------ beam
    def _tab_beam(self):
        w = QWidget()
        v = QVBoxLayout(w)
        f = QFormLayout()
        self.b_spans = QLineEdit("6000")
        self.b_spans.setToolTip("Beam length [mm]; several spans separated by semicolons = continuous beam (e.g. 4000; 5000)")
        f.addRow("Spans [mm]", self.b_spans)
        h = QHBoxLayout()
        self.b_left, self.b_right = QComboBox(), QComboBox()
        for cb, tip in ((self.b_left, "Left end"), (self.b_right, "Right end")):
            for t, k in SUPPORTS:
                cb.addItem(t, k)
            cb.setToolTip(tip + ": hinge, clamp, free end (cantilever), sliding clamp (φ = 0, V = 0).\nContinuous beam: end support hinge or clamp.")
            h.addWidget(cb)
        f.addRow("Supports (left, right)", h)
        self.b_sec = _section_edit("Library cross-section (E·I from the profile, stress M/W_el); empty = I entered below")
        f.addRow("Cross-section", self.b_sec)
        h = QHBoxLayout()
        self.b_axis = QComboBox()
        self.b_axis.addItem("strong axis y", "y")
        self.b_axis.addItem("weak axis z", "z")
        self.b_axis.setToolTip("Bending axis of the library cross-section")
        h.addWidget(self.b_axis)
        self.b_I = dspin(1e7, 1e-3, 1e15, 0, 1e6, " mm⁴")
        self.b_I.setToolTip("Moment of inertia I [mm⁴] (only without a library cross-section)")
        h.addWidget(self.b_I)
        f.addRow("Axis / I", h)
        self.b_E = dspin(210000.0, 1.0, 1e7, 0, 1000.0, " MPa")
        self.b_E.setToolTip("Young's modulus E [MPa]")
        f.addRow("E", self.b_E)
        v.addLayout(f)
        self.b_loads = QTableWidget(1, 5)
        self.b_loads.setHorizontalHeaderLabels(["Type", "Value", "a [mm]", "b [mm]", "q at the end"])
        self.b_loads.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.b_loads.setMaximumHeight(150)
        self.b_loads.setToolTip("Loads (positive downward): distributed q from a to b (empty b = to the end, q at end = linear\nvariation), point force P at x = a, point moment C at x = a (positive clockwise).\nFor a continuous beam, x is measured from the left end of the beam.")
        self._load_row(0, "q", 10.0)
        v.addWidget(self.b_loads)
        h = QHBoxLayout()
        for txt, tip, slot in (("+ load", "Add a load row", lambda: self._load_row(self.b_loads.rowCount())),
                               ("− load", "Remove the selected load row", self._del_load)):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            h.addWidget(b)
        b = QPushButton("Calculate")
        theme.set_primary(b)
        b.setToolTip("Calculate V, M, w diagrams, reactions and extremes")
        b.clicked.connect(self.calc_beam)
        h.addWidget(b)
        v.addLayout(h)
        self.b_plot = LazyPlot()
        self.b_plot.setMinimumHeight(330)
        self.b_plot.setToolTip("Shear force V [kN], bending moment M [kN·m] and deflection w [mm] (positive downward)")
        v.addWidget(self.b_plot, 1)
        self.b_out = SafeTextBrowser()
        self.b_out.setMaximumHeight(150)
        self.b_out.setToolTip("Reactions, extremes and stresses (with a library cross-section)")
        v.addWidget(self.b_out)
        self.tabs.addTab(w, "Beam")

    def _load_row(self, r, kind="P", val=1000.0):
        t = self.b_loads
        if r >= t.rowCount():
            t.insertRow(r)
        cb = QComboBox()
        for txt, k in LOAD_KINDS:
            cb.addItem(txt, k)
        cb.setCurrentIndex(max(0, cb.findData(kind)))
        cb.setToolTip("Load kind")
        t.setCellWidget(r, 0, cb)
        for c, txt in enumerate((fmt(val), "0", "", ""), start=1):
            t.setItem(r, c, QTableWidgetItem(txt))

    def _del_load(self):
        r = self.b_loads.currentRow()
        if r >= 0:
            self.b_loads.removeRow(r)

    @staticmethod
    def _f(item, default=None):
        if item is None or not item.text().strip():
            return default
        return float(item.text().replace(",", ".").replace(" ", ""))

    def beam_loads(self):
        from fea_core.beams import analytic as A
        out = []
        t = self.b_loads
        for r in range(t.rowCount()):
            k = t.cellWidget(r, 0).currentData()
            val = self._f(t.item(r, 1), 0.0)
            a = self._f(t.item(r, 2), 0.0)
            b = self._f(t.item(r, 3))
            q2 = self._f(t.item(r, 4))
            if k == "q":
                out.append(A.Distributed(val, a, b, q2 if q2 is not None else val))
            elif k == "P":
                out.append(A.Force(val, a))
            else:
                out.append(A.Moment(val, a))
        return out

    def calc_beam(self):
        from fea_core.beams import analytic as A
        try:
            spans = [float(x.replace(",", ".")) for x in self.b_spans.text().replace(";", " ").split()]
            sec = _get_section(self.b_sec.text())
            kw = dict(E=self.b_E.value(), axis=self.b_axis.currentData())
            if sec is not None:
                kw["section"] = sec
            else:
                kw["I"] = self.b_I.value()
            loads = self.beam_loads()
            if len(spans) == 1:
                r = A.beam(spans[0], loads, (self.b_left.currentData(), self.b_right.currentData()), **kw)
            else:
                r = A.continuous_beam(spans, loads, left=self.b_left.currentData(), right=self.b_right.currentData(),
                                      **kw)
        except Exception as e:  # noqa: BLE001
            self.b_out.setHtml(f"<span style='color:#d23b3b'>{html.escape(str(e))}</span>")
            return None
        self.b_plot._ensure()
        fig = self.b_plot.fig
        fig.clear()
        axs = fig.subplots(3, 1, sharex=True)
        for ax, y, lbl, col in ((axs[0], r.V / 1e3, "V [kN]", "#2e9e4f"), (axs[1], r.M / 1e6, "M [kN·m]", "#2563eb"),
                                (axs[2], r.w, "w [mm]", "#d23b3b")):
            ax.plot(r.x, y, color=col, lw=1.6)
            ax.fill_between(r.x, y, alpha=0.15, color=col)
            ax.axhline(0, color="#888", lw=0.6)
            ax.set_ylabel(lbl)
            ax.grid(True, alpha=0.3)
        axs[1].invert_yaxis()                          # positive moment (tension at the bottom) drawn downwards
        axs[2].invert_yaxis()
        axs[2].set_xlabel("x [mm]")
        self.b_plot.canvas.draw_idle()
        e = r.extremes
        rows = ["<b>Reakce</b>: " + ", ".join(f"{k} = {fmt(float(v))}" for k, v in r.reactions.items()),
                f"M: max {fmt(e['M']['max'][0] / 1e6)} kN·m (x = {fmt(e['M']['max'][1])} mm), min "
                f"{fmt(e['M']['min'][0] / 1e6)} kN·m (x = {fmt(e['M']['min'][1])} mm)",
                f"V: |V| max {fmt(e['V']['abs'][0] / 1e3)} kN (x = {fmt(e['V']['abs'][1])} mm)",
                f"w: |w| max <b>{fmt(e['w']['abs'][0])} mm</b> (x = {fmt(e['w']['abs'][1])} mm)"]
        if sec is not None:
            W = sec.Wel_y if self.b_axis.currentData() == "y" else sec.Wel_z
            Mm = max(abs(e["M"]["max"][0]), abs(e["M"]["min"][0]))
            rows.append(f"σ = |M|max / W_el = <b>{fmt(Mm / W)} MPa</b> ({html.escape(sec.name)}, W_el = {fmt(W)} mm³)")
        rows += [f"<span style='color:#b36b00'>! {html.escape(w)}</span>" for w in r.warnings]
        self.b_out.setHtml("<br>".join(rows))
        self.last_beam = r
        return r

    # ------------------------------------------------------------------ curved bar
    def _tab_curved(self):
        w = QWidget()
        f = QFormLayout(w)
        self.c_prof = QComboBox()
        self.c_prof.addItems(list(CURVED))
        self.c_prof.setToolTip("Cross-section shape in the plane of curvature (height h radial)")
        self.c_prof.currentIndexChanged.connect(lambda _: self._sync_curved())
        f.addRow("Cross-section", self.c_prof)
        self.c_dims = {}
        h = QHBoxLayout()
        for k, val in (("b", 20.0), ("h", 40.0), ("d", 30.0), ("D", 60.0), ("t", 5.0), ("tf", 8.0), ("tw", 6.0)):
            lb = QLabel(k)
            sp = dspin(val, 0.01, 1e5, 2, 1.0)
            sp.setToolTip(f"Dimension {k} [mm]")
            h.addWidget(lb)
            h.addWidget(sp)
            self.c_dims[k] = (lb, sp)
        f.addRow("Dimensions [mm]", h)
        self.c_R = dspin(60.0, 0.01, 1e7, 3, 5.0, " mm")
        self.c_R.setToolTip("Centroidal axis radius R [mm] (center of curvature → cross-section centroid)")
        f.addRow("Centroid R", self.c_R)
        self.c_M = dspin(1e6, -1e12, 1e12, 0, 1e5, " N·mm")
        self.c_M.setToolTip("Bending moment M [N·mm]; positive increases curvature (inner fiber in compression)")
        f.addRow("M", self.c_M)
        self.c_N = dspin(0.0, -1e10, 1e10, 0, 1000.0, " N")
        self.c_N.setToolTip("Axial force N [N] at the centroid (positive tension)")
        f.addRow("N", self.c_N)
        b = QPushButton("Calculate")
        theme.set_primary(b)
        b.setToolTip("Inner and outer fiber stress per Winkler–Bach, compared with a straight member")
        b.clicked.connect(self.calc_curved)
        f.addRow(b)
        self.c_out = SafeTextBrowser()
        self.c_out.setToolTip("Result: neutral axis position, fiber stresses, factors K_i, K_o")
        f.addRow(self.c_out)
        self.tabs.addTab(w, "Curved member")
        self._sync_curved()

    def _sync_curved(self):
        need = CURVED[self.c_prof.currentText()]
        for k, (lb, sp) in self.c_dims.items():
            lb.setVisible(k in need)
            sp.setVisible(k in need)

    def calc_curved(self):
        from fea_core.beams import analytic as A
        d = {k: sp.value() for k, (lb, sp) in self.c_dims.items()}
        p = self.c_prof.currentText()
        try:
            prof = (A.rect(d["b"], d["h"]) if p == "rectangle" else A.circle(d["d"]) if p == "circle" else
                    A.tube(d["D"], d["t"]) if p == "tube" else A.tee(d["b"], d["tf"], d["tw"], d["h"]))
            r = A.curved_beam(prof, M=self.c_M.value(), N=self.c_N.value(), R=self.c_R.value())
        except Exception as e:  # noqa: BLE001
            self.c_out.setHtml(f"<span style='color:#d23b3b'>{html.escape(str(e))}</span>")
            return None
        self.c_out.setHtml("<br>".join([
            f"A = {fmt(r.A)} mm², R = {fmt(r.R)} mm, neutral axis r_n = {fmt(r.r_n)} mm (e = {fmt(r.e, 4)} mm)",
            f"Inner fiber r_i = {fmt(r.r_i)} mm: <b>σ_i = {fmt(r.sigma_i)} MPa</b> (straight member {fmt(r.sigma_i_straight)} MPa, K_i = {fmt(r.K_i, 4)})",
            f"Outer fiber r_o = {fmt(r.r_o)} mm: <b>σ_o = {fmt(r.sigma_o)} MPa</b> (straight member {fmt(r.sigma_o_straight)} MPa, K_o = {fmt(r.K_o, 4)})",
            "<i>Winkler–Bach (Roark, Table 9.1): for R/h < 10 the inner-fiber stress is significantly higher than for a straight member.</i>"]))
        return r

    # ------------------------------------------------------------------ buckling
    def _tab_euler(self):
        w = QWidget()
        f = QFormLayout(w)
        self.e_L = dspin(3000.0, 1.0, 1e7, 1, 100.0, " mm")
        self.e_L.setToolTip("Member length L [mm]")
        f.addRow("L", self.e_L)
        self.e_case = QComboBox()
        for t, k in CASES:
            self.e_case.addItem(t, k)
        self.e_case.setToolTip("End conditions – effective length factor β (L_cr = β·L)")
        f.addRow("Support", self.e_case)
        self.e_sec = _section_edit("Library cross-section (weak axis); empty = I and A below")
        f.addRow("Cross-section", self.e_sec)
        h = QHBoxLayout()
        self.e_I = dspin(1e6, 1e-3, 1e15, 0, 1e5, " mm⁴")
        self.e_I.setToolTip("Moment of inertia I [mm⁴] (when no library cross-section is used)")
        self.e_A = dspin(1000.0, 1e-3, 1e9, 1, 100.0, " mm²")
        self.e_A.setToolTip("Area A [mm²] (slenderness, critical stress)")
        h.addWidget(self.e_I)
        h.addWidget(self.e_A)
        f.addRow("I, A", h)
        self.e_E = dspin(210000.0, 1.0, 1e7, 0, 1000.0, " MPa")
        self.e_E.setToolTip("Young's modulus E [MPa]")
        f.addRow("E", self.e_E)
        self.e_fy = dspin(235.0, 0.0, 2000.0, 1, 5.0, " MPa", special="not specified")
        self.e_fy.setToolTip("Yield strength f_y [MPa] – relative slenderness λ̄ = √(A·f_y / F_cr)")
        f.addRow("f_y", self.e_fy)
        b = QPushButton("Calculate")
        theme.set_primary(b)
        b.setToolTip("Euler critical force F_cr = π²EI / (βL)²")
        b.clicked.connect(self.calc_euler)
        f.addRow(b)
        self.e_out = SafeTextBrowser()
        self.e_out.setToolTip("Critical force, effective length, slenderness, critical stress")
        f.addRow(self.e_out)
        self.tabs.addTab(w, "Buckling (Euler)")

    def calc_euler(self):
        from fea_core.beams.analytic import euler_buckling
        try:
            sec = _get_section(self.e_sec.text())
            kw = dict(E=self.e_E.value(), case=self.e_case.currentData(), fy=self.e_fy.value() or None)
            if sec is not None:
                kw["section"] = sec
            else:
                kw.update(I=self.e_I.value(), A=self.e_A.value())
            r = euler_buckling(self.e_L.value(), **kw)
        except Exception as e:  # noqa: BLE001
            self.e_out.setHtml(f"<span style='color:#d23b3b'>{html.escape(str(e))}</span>")
            return None
        rows = [f"<b>F_cr = {fmt(r.F_cr / 1e3)} kN</b> (β = {fmt(r.beta, 3)}, L_cr = {fmt(r.L_cr)} mm)"]
        if r.slenderness:
            rows.append(f"i = {fmt(r.i)} mm, slenderness λ = {fmt(r.slenderness)}, σ_cr = {fmt(r.sigma_cr)} MPa")
        if r.rel_slenderness:
            rows.append(f"Relative slenderness λ̄ = {fmt(r.rel_slenderness, 3)}"
                        + (" – below 0.2 buckling does not govern (EN 1993-1-1, 6.3.1.2(4))" if r.rel_slenderness < 0.2 else ""))
        rows.append("<i>Ideal elastic buckling – the design resistance per EN 1993-1-1 (χ) is lower (Check tab).</i>")
        self.e_out.setHtml("<br>".join(rows))
        return r

    # ------------------------------------------------------------------ posudek EN 1993
    def _tab_ec3(self):
        w = QWidget()
        v = QVBoxLayout(w)
        f = QFormLayout()
        self.k_sec = _section_edit("Library cross-section (the check needs the profile dimensions)")
        self.k_sec.setText("IPE 200")
        f.addRow("Cross-section", self.k_sec)
        self.k_grade = QComboBox()
        self.k_grade.addItems(["S235", "S275", "S355", "S420", "S460"])
        self.k_grade.setToolTip("Steel – f_y by thickness (EN 1993-1-1, Table 3.1)")
        f.addRow("Steel", self.k_grade)
        self.k_L = dspin(4000.0, 1.0, 1e7, 1, 100.0, " mm")
        self.k_L.setToolTip("Member length L [mm]")
        f.addRow("L", self.k_L)
        h = QHBoxLayout()
        self.k_lcy = dspin(0.0, 0.0, 1e8, 1, 100.0, " mm", special="= L")
        self.k_lcy.setToolTip("Effective length L_cr,y [mm] (buckling perpendicular to the y axis)")
        self.k_lcz = dspin(0.0, 0.0, 1e8, 1, 100.0, " mm", special="= L")
        self.k_lcz.setToolTip("Effective length L_cr,z [mm]")
        self.k_llt = dspin(0.0, 0.0, 1e8, 1, 100.0, " mm", special="= L")
        self.k_llt.setToolTip("Length between lateral-torsional buckling restraints L_LT [mm]")
        for s in (self.k_lcy, self.k_lcz, self.k_llt):
            h.addWidget(s)
        f.addRow("L_cr,y / L_cr,z / L_LT", h)
        self.k_N = dspin(-100e3, -1e10, 1e10, 0, 1000.0, " N")
        self.k_N.setToolTip("Axial force N [N] (positive tension, negative compression) – constant along the length")
        f.addRow("N", self.k_N)
        h = QHBoxLayout()
        self.k_My1 = dspin(20e6, -1e13, 1e13, 0, 1e5, " N·mm")
        self.k_My1.setToolTip("Moment My at member start [N·mm] (linear variation to the value at the end)")
        self.k_My2 = dspin(0.0, -1e13, 1e13, 0, 1e5, " N·mm")
        self.k_My2.setToolTip("Moment My at the member end [N·mm]")
        h.addWidget(self.k_My1)
        h.addWidget(self.k_My2)
        f.addRow("My (start, end)", h)
        h = QHBoxLayout()
        self.k_Mz = dspin(0.0, -1e13, 1e13, 0, 1e5, " N·mm")
        self.k_Mz.setToolTip("Moment Mz [N·mm] – constant along the length")
        self.k_Vz = dspin(0.0, -1e10, 1e10, 0, 1000.0, " N")
        self.k_Vz.setToolTip("Shear force Vz [N] – constant along the length")
        h.addWidget(self.k_Mz)
        h.addWidget(self.k_Vz)
        f.addRow("Mz, Vz", h)
        b = QPushButton("Check")
        theme.set_primary(b)
        b.setToolTip("EN 1993-1-1 check: cross-section class, resistance (6.2), buckling, lateral-torsional buckling and interaction (6.3)")
        b.clicked.connect(self.calc_ec3)
        f.addRow(b)
        v.addLayout(f)
        self.k_rows = table(["Assessment", "Cl.", "Action effect", "Resistance", "Utilization"], 200)
        self.k_rows.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.k_rows.setToolTip("Check rows – utilization ≤ 1 passes; informative rows are not included in the result")
        v.addWidget(self.k_rows, 1)
        self.k_out = QLabel("")
        self.k_out.setWordWrap(True)
        v.addWidget(self.k_out)
        self.tabs.addTab(w, "EN 1993 member check")

    def calc_ec3(self):
        from fea_core.beams.ec3 import MemberForces, MemberSpec, check_member
        try:
            sec = _get_section(self.k_sec.text())
            if sec is None:
                raise ValueError("Select a library cross-section")
            L = self.k_L.value()
            x = np.linspace(0.0, L, 21)
            My = self.k_My1.value() + (self.k_My2.value() - self.k_My1.value()) * x / L
            forces = MemberForces(x, N=self.k_N.value(), Vz=self.k_Vz.value(), My=My, Mz=self.k_Mz.value())
            spec = MemberSpec(L=L, grade=self.k_grade.currentText(), Lcr_y=self.k_lcy.value() or None,
                              Lcr_z=self.k_lcz.value() or None, L_LT=self.k_llt.value() or None)
            chk = check_member(sec, forces, spec)
        except Exception as e:  # noqa: BLE001
            self.k_out.setText(f"<span style='color:#d23b3b'>{html.escape(str(e))}</span>")
            return None
        t = self.k_rows
        t.setRowCount(len(chk.rows))
        for r, row in enumerate(chk.rows):
            for c, txt in enumerate((row.name, row.clause, f"{fmt(row.effect, 4)} {row.unit}", fmt(row.resistance, 4),
                                     fmt(row.utilization, 3))):
                it = QTableWidgetItem(txt)
                if c >= 2:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(r, c, it)
        col = "#d23b3b" if chk.utilization > 1 else "#2e9e4f"
        self.k_out.setText(f"<b>Utilisation <span style='color:{col}'>{fmt(chk.utilization, 3)}</span></b> – governed by {html.escape(chk.governing)}; section class {chk.section_class}, f_y = {fmt(chk.fy)} MPa"
                           + "".join(f"<br><span style='color:#b36b00'>! {html.escape(w)}</span>" for w in chk.warnings))
        return chk
