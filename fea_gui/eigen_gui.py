"""Natural frequencies and linear buckling in the GUI (fea_core.eigen): settings in the Solve step, mode descriptions and a table of
frequencies with effective-mass fractions in the results. The shapes are normalised (max |u| = 1) – the colour map shows the shape,
not the real size of the displacement.
"""
from __future__ import annotations

import html

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHeaderView, QLabel, QSpinBox,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from fea_core.eigen import BucklingSettings, ModalSettings

from .widgets import Section, dspin, fmt, help_tip, table

# quantities of the mode shape display (normalised displacement)
MODE_QUANTITIES = [("Mode – total displacement (normalized)", "–"), ("Mode – X displacement (normalized)", "–"),
                   ("Mode – Y displacement (normalized)", "–"), ("Mode – Z displacement (normalized)", "–")]
LINEAR_NOTE = ("Buckling is LINEAR (stress stiffness only, no imperfections, plasticity or large deformations) – the actual capacity is usually lower than λ·load. For design use the code procedure (e.g. EN 1993-1-1, 6.3) or a nonlinear analysis.")


def mode_value_text(ms, k: int) -> str:
    """„f = 209,4 Hz“ / „λ = 21,65“."""
    v = float(ms.values[k])
    return f"f = {fmt(v, 5)} Hz" if ms.kind == "modal" else f"λ = {fmt(v, 5)}"


def mode_label(ms, k: int) -> str:
    """Results menu item: "Vlastní tvar 1: f = 209,4 Hz" / "Tvar vzpěru 1 (Tlak): λ = 21,65" (the Czech labels the program shows)."""
    if ms.kind == "modal":
        return f"Mode shape {k + 1}: {mode_value_text(ms, k)}"
    return f"Buckling mode {k + 1} ({ms.case}): {mode_value_text(ms, k)}"


def mass_fractions(ms) -> np.ndarray | None:
    """Effective-mass fractions (k, 3) in the directions x, y, z, or None."""
    if not ms.eff_mass or not ms.total_mass:
        return None
    tot = np.asarray(ms.total_mass, float)[:3]
    return np.asarray(ms.eff_mass, float)[:, :3] / np.where(tot > 0, tot, 1.0)


def pct(c) -> str:
    """Mass fraction in % (noise below 0.005 % = 0)."""
    v = 100 * float(c)
    return fmt(0.0 if abs(v) < 5e-3 else v, 3)


def mode_quantity(shape: np.ndarray, qi: int) -> np.ndarray:
    if qi <= 0:
        return np.linalg.norm(shape, axis=1)
    return shape[:, qi - 1]


def mode_info_html(ms, k: int, study=None) -> str:
    """Summary of the selected shape for the results panel."""
    v = float(ms.values[k])
    rows = []
    if ms.kind == "modal":
        rows.append(f"<b>Mode shape {k + 1}</b> – natural frequency <b>{fmt(v, 5)} Hz</b> (ω = {fmt(2 * np.pi * v, 5)} rad/s, period {(fmt(1 / v, 4) if v > 0 else '∞')} s)")
        fr = mass_fractions(ms)
        if fr is not None:
            rows.append("Effective mass fraction of the mode: x {} %, y {} %, z {} %".format(
                *(pct(c) for c in fr[k])))
            cum = fr[:k + 1].sum(axis=0)
            rows.append("Sum of modes 1–{}: x {} %, y {} %, z {} %".format(k + 1, *(pct(c) for c in cum)))
        if ms.total_mass:
            rows.append(f"Total model mass: {fmt(1000 * float(ms.total_mass[0]), 5)} kg")
        if v < max(1e-3 * float(np.max(ms.values)), 1e-2):
            rows.append("<span style='color:#b36b00'>! Frequency ≈ 0 Hz – rigid-body motion (model is unsupported or a part is free).</span>")
        rows.append("<i>The shape is normalized (max |u| = 1) – colors show the mode shape, not actual displacements. The effective mass fraction tells how strongly the mode is excited by acceleration in the given direction (for seismic and base excitation a sum ≥ 90 % is required).</i>")
    else:
        rows.append(f"<b>Buckling mode {k + 1}</b> – critical load factor <b>λ = {fmt(v, 5)}</b>")
        rows.append(f"Loss of stability at λ × load of the state “{html.escape(str(ms.case))}”"
                    + (" (negative λ = under REVERSED load)." if v < 0 else "."))
        if 0 < v < 1:
            rows.append("<span style='color:#d23b3b'>! λ < 1 – the structure loses stability under the applied load.</span>")
        rows.append(f"<span style='color:#b36b00'>! {LINEAR_NOTE}</span>")
        rows.append("<i>The shape is normalized (max |u| = 1).</i>")
    return "<br>".join(rows)


class EigenBox(QWidget):
    """The groups "Natural frequency" and "Linear buckling" in the Solve step (Study.modal, Study.buckling)."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        self._filling = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.m_on = QCheckBox("compute natural frequencies and mode shapes")
        help_tip(self.m_on, "After the statics compute natural frequencies, mode shapes and effective mass",
                 "<b>Natural frequencies</b><p>Computes the lowest natural frequencies and mode shapes of the model with the study's supports. The mass is taken from the material densities (the analysis cannot run without a density). Results: Results step → the “Mode shape …” list and the <i>Mode shapes</i> tab with effective mass fractions.</p><p>A study may contain only natural frequencies, without load cases.</p><p>On = the natural frequencies (Hz), mode shapes and effective mass fractions are computed after the statics. Without supports = free body (the first 6 frequencies ≈ 0 Hz).</p>")
        g = Section("Natural frequencies", check=self.m_on,
                    tip="Modal analysis: natural frequencies and mode shapes (ccx *FREQUENCY)",
                    help="Modal analysis (ccx *FREQUENCY): natural frequencies and mode shapes with the study's supports and the mass from the material densities. It can also run without load cases.")
        f = QFormLayout(g.body)
        self.m_modes = QSpinBox()
        self.m_modes.setRange(1, 200)
        self.m_modes.setValue(10)
        self.m_modes.setToolTip("Number of lowest natural frequencies to find (usually 6–20)")
        f.addRow("Number of modes", self.m_modes)
        self.m_fmin = dspin(0.0, 0.0, 1e7, 2, 10.0, " Hz")
        self.m_fmin.setToolTip("Lower bound of searched frequencies [Hz] (0 = from the lowest)")
        self.m_fmax = dspin(0.0, 0.0, 1e7, 2, 100.0, " Hz", special="no limit")
        self.m_fmax.setToolTip("Upper bound of searched frequencies [Hz] (no limit = mode count only)")
        f.addRow("From", self.m_fmin)
        f.addRow("To", self.m_fmax)
        lay.addWidget(g)
        self.modal_group = g
        self.b_on = QCheckBox("compute buckling (critical load factor λ)")
        help_tip(self.b_on, "Compute the load multiplier λ at which the structure loses stability (linear)",
                 "<b>Linear buckling</b><p>Computes factors λ: load of the selected case × λ = critical load (loss of stability). Negative λ = buckling under reversed load.</p><p><b>Note:</b> the linear analysis does not include initial imperfections, plasticity, or large deformations – the actual capacity is usually significantly lower. For design use the code procedure (EN 1993-1-1, 6.3) or a nonlinear analysis.</p>")
        g = Section("Buckling (linear)", check=self.b_on,
                    tip="Linear stability: critical load factors λ and buckling modes (ccx *BUCKLE)",
                    help="Linear stability (ccx *BUCKLE): critical load factors λ and buckling modes for the loads of one load case.")
        f = QFormLayout(g.body)
        self.b_case = QComboBox()
        self.b_case.setToolTip("Load case whose loads are multiplied by factor λ (first = first case of the study)")
        f.addRow("Load case", self.b_case)
        self.b_modes = QSpinBox()
        self.b_modes.setRange(1, 50)
        self.b_modes.setValue(4)
        self.b_modes.setToolTip("Number of buckling modes (lowest factors λ)")
        f.addRow("Number of modes", self.b_modes)
        self.b_note = QLabel(LINEAR_NOTE)
        self.b_note.setWordWrap(True)
        self.b_note.setStyleSheet("color:#b36b00")
        f.addRow(self.b_note)
        lay.addWidget(g)
        self.buckling_group = g
        for w in (self.m_on, self.b_on):
            w.toggled.connect(self.on_change)
        for w in (self.m_modes, self.m_fmin, self.m_fmax, self.b_modes):
            w.valueChanged.connect(self.on_change)
        self.b_case.currentIndexChanged.connect(self.on_change)

    def refresh(self):
        st = self.win.study
        if st is None:
            return
        self._filling = True
        try:
            m, b = st.modal, st.buckling
            self.m_on.setChecked(bool(m.enabled))
            self.m_modes.setValue(int(m.modes))
            self.m_fmin.setValue(float(m.f_min or 0.0))
            self.m_fmax.setValue(float(m.f_max or 0.0))
            self.b_on.setChecked(bool(b.enabled))
            self.b_modes.setValue(int(b.modes))
            self.b_case.clear()
            self.b_case.addItem("first load case", None)
            for lc in st.load_cases:
                self.b_case.addItem(lc.name, lc.name)
            if b.case is not None and self.b_case.findData(b.case) < 0:
                self.b_case.addItem(f"{b.case} (does not exist)", b.case)
            self.b_case.setCurrentIndex(max(0, self.b_case.findData(b.case)) if b.case is not None else 0)
        finally:
            self._filling = False
        self._enable()

    def _enable(self):
        for w in (self.m_modes, self.m_fmin, self.m_fmax):
            w.setEnabled(self.m_on.isChecked())
        for w in (self.b_case, self.b_modes):
            w.setEnabled(self.b_on.isChecked())
        self.b_note.setVisible(self.b_on.isChecked())

    def settings(self) -> tuple[ModalSettings, BucklingSettings]:
        return (ModalSettings(self.m_on.isChecked(), int(self.m_modes.value()), float(self.m_fmin.value()),
                              float(self.m_fmax.value()) or None),
                BucklingSettings(self.b_on.isChecked(), self.b_case.currentData(), int(self.b_modes.value())))

    def on_change(self, *_):
        self._enable()
        st = self.win.study
        if st is None or self._filling:
            return
        m, b = self.settings()
        if m.to_dict() == st.modal.to_dict() and b.to_dict() == st.buckling.to_dict():
            return
        st.modal, st.buckling = m, b
        self.win.changed()


class ModesTable(QWidget):
    """The "Mode shapes" tab in the results: shapes, frequencies / λ, effective-mass fractions and their sums."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        v = QVBoxLayout(self)
        v.setContentsMargins(2, 4, 2, 2)
        self.table = table(["Shape", "f [Hz] / λ", "m_x [%]", "m_y [%]", "m_z [%]", "Σ m_x", "Σ m_y", "Σ m_z"], 160)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        help_tip(self.table, "Natural frequencies [Hz] (buckling: λ) and effective mass fractions in x, y, z",
                 "Natural frequencies [Hz] (buckling: factors λ) and effective mass fractions in the x, y, z directions (m_x = how much mass takes part in the vibration along the x axis; Σ = sum from the 1st mode – 90 % or more is required for base excitation). A click on a row shows the mode shape in 3D.")
        self.table.currentCellChanged.connect(lambda r, *_: self.on_row(r))
        v.addWidget(self.table)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        v.addWidget(self.note)
        self._rows: list[tuple[int, int]] = []

    def fill(self, results):
        t = self.table
        t.blockSignals(True)
        t.setRowCount(0)
        self._rows = []
        notes = []
        for j, ms in enumerate(results.modes if results is not None else []):
            fr = mass_fractions(ms)
            cum = np.cumsum(fr, axis=0) if fr is not None else None
            for k in range(len(ms.values)):
                r = t.rowCount()
                t.insertRow(r)
                name = f"{k + 1}" if ms.kind == "modal" else f"buckling {k + 1} ({ms.case})"
                vals = [name, fmt(float(ms.values[k]), 5)]
                if fr is not None and k < len(fr):
                    vals += [pct(c) for c in fr[k]] + [pct(c) for c in cum[k]]
                else:
                    vals += ["–"] * 6
                for c, txt in enumerate(vals):
                    it = QTableWidgetItem(txt)
                    if c:
                        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    t.setItem(r, c, it)
                self._rows.append((j, k))
            if ms.kind == "modal" and cum is not None and len(cum):
                notes.append("Sum of the effective mass fractions of {} modes: x {} %, y {} %, z {} %".format(
                    len(cum), *(pct(c) for c in cum[-1])))
            if ms.kind == "buckling":
                notes.append(f"<span style='color:#b36b00'>! {LINEAR_NOTE}</span>")
        t.blockSignals(False)
        self.note.setText("<br>".join(notes))

    def on_row(self, r):
        if 0 <= r < len(self._rows):
            j, k = self._rows[r]
            self.win.select_result(f"mode:{j}:{k}")

    def select(self, j: int, k: int):
        if (j, k) in self._rows:
            self.table.blockSignals(True)
            self.table.selectRow(self._rows.index((j, k)))
            self.table.blockSignals(False)
