"""Natural frequencies and linear buckling in the GUI (fea_core.eigen): settings in the Solve step, mode descriptions and a table of
frequencies with effective-mass fractions in the results. The shapes are normalised (max |u| = 1) – the colour map shows the shape,
not the real size of the displacement.
"""
from __future__ import annotations

import html

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHeaderView, QLabel, QLineEdit, QSpinBox,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from fea_core.eigen import BucklingSettings, ModalSettings

from .widgets import LazyPlot, Section, dspin, fmt, help_tip, table

# quantities of the mode shape display (normalised displacement)
MODE_QUANTITIES = [("Mode – total displacement (normalized)", "–"), ("Mode – X displacement (normalized)", "–"),
                   ("Mode – Y displacement (normalized)", "–"), ("Mode – Z displacement (normalized)", "–")]
LINEAR_NOTE = ("Buckling is LINEAR (stress stiffness only, no imperfections, plasticity or large deformations) – the actual capacity is usually lower than λ·load. For design use the code procedure (e.g. EN 1993-1-1, 6.3) or a nonlinear analysis.")


# nonlinear buckling (fea_core.imperfect): real displacements of the increments from the imperfect geometry (frame 0 = imperfection)
NLB_QUANTITIES = [("Nonlinear buckling – total displacement", "mm"), ("Nonlinear buckling – X displacement", "mm"),
                  ("Nonlinear buckling – Y displacement", "mm"), ("Nonlinear buckling – Z displacement", "mm")]
NLB_KIND = "nlbuckle"
NLB_NOTE = ("The imperfection shape and amplitude e0 are your responsibility – take them from the design code "
            "(EN 1993-1-5 Annex C, EN 1993-1-6, EN 1993-1-1 5.3). Load control: the ultimate factor is the last converged increment.")


def mode_quantities(ms) -> list[tuple[str, str]]:
    """Quantity menu of a mode set: normalised shapes, or real displacements of the nonlinear buckling increments."""
    return NLB_QUANTITIES if getattr(ms, "kind", "") == NLB_KIND else MODE_QUANTITIES


def mode_value_text(ms, k: int) -> str:
    """„f = 209,4 Hz“ / „λ = 21,65“."""
    v = float(ms.values[k])
    return f"f = {fmt(v, 5)} Hz" if ms.kind == "modal" else f"λ = {fmt(v, 5)}"


def mode_label(ms, k: int) -> str:
    """Results menu item: "Vlastní tvar 1: f = 209,4 Hz" / "Tvar vzpěru 1 (Tlak): λ = 21,65" (the Czech labels the program shows)."""
    if ms.kind == "modal":
        nds = getattr(ms, "nodal_diameters", None)        # cyclic symmetry: nodal diameter of the mode
        if nds is not None and k < len(nds):
            return f"Mode shape {k + 1} (ND {int(nds[k])}): {mode_value_text(ms, k)}"
        return f"Mode shape {k + 1}: {mode_value_text(ms, k)}"
    if ms.kind == NLB_KIND:
        if k == 0:
            return f"Nonlinear buckling ({ms.case}): imperfection"
        return f"Nonlinear buckling ({ms.case}): {mode_value_text(ms, k)}"
    return f"Buckling mode {k + 1} ({ms.case}): {mode_value_text(ms, k)}"


def nlb_info_html(ms, k: int, info: dict | None) -> str:
    """Summary of a nonlinear buckling frame + the ultimate factor."""
    rows = []
    umax = float(np.linalg.norm(np.asarray(ms.shapes[k], float), axis=1).max()) if len(ms.shapes[k]) else 0.0
    if k == 0:
        rows.append(f"<b>Imperfection</b> (initial shape added to the mesh): max {fmt(umax, 4)} mm")
    else:
        rows.append(f"<b>Nonlinear buckling</b> – load factor <b>λ = {fmt(float(ms.values[k]), 5)}</b>, "
                    f"max displacement {fmt(umax, 4)} mm (from the imperfect geometry)")
    if info:
        r = info.get("result") or {}
        if r.get("ultimate_factor") is not None:
            rows.append(f"Ultimate load factor <b>λ_u = {fmt(float(r['ultimate_factor']), 5)}</b> ({info.get('kind', '')}) – "
                        f"{html.escape(str(info.get('status_text', '')))}")
        if info.get("linear_factor"):
            rows.append(f"Linear buckling λ_cr = {fmt(float(info['linear_factor']), 5)}"
                        + (f", λ_u/λ_cr = {fmt(float(r['knockdown']), 4)}" if r.get("knockdown") else ""))
        if r.get("first_yield_factor") is not None:
            rows.append(f"First yield (max von Mises = Re) at λ = {fmt(float(r['first_yield_factor']), 5)}")
        imp = info.get("imperfection") or {}
        if imp:
            rows.append(f"Imperfection e0 = {fmt(float(imp.get('e0_mm', 0.0)), 4)} mm ({html.escape(str(imp.get('kind', '')))})")
    rows.append(f"<span style='color:#b36b00'>! {NLB_NOTE}</span>")
    return "<br>".join(rows)


def parse_modes(text: str) -> tuple[list[int], list[float]]:
    """"1" | "1, 2*0.5" | "1, 3*-0.3" → ([1, 2], [1.0, 0.5]); ValueError on a bad entry."""
    modes, weights = [], []
    for tok in (text or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "*" in tok:
            a, b = tok.split("*", 1)
            m, w = int(a.strip()), float(b.strip().replace(",", "."))
        else:
            m, w = int(tok), 1.0
        if m < 1:
            raise ValueError(tok)
        modes.append(m)
        weights.append(w)
    if not modes:
        raise ValueError(text)
    return modes, weights


def format_modes(modes, weights) -> str:
    out = []
    for i, m in enumerate(modes or [1]):
        w = float(weights[i]) if weights and i < len(weights) else 1.0
        out.append(f"{int(m)}" if w == 1.0 else f"{int(m)}*{w:g}")
    return ", ".join(out)


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
    if ms.kind == NLB_KIND:
        return nlb_info_html(ms, k, getattr(ms, "info", None))
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
        self._build_nonlinear(lay)
        for w in (self.m_on, self.b_on):
            w.toggled.connect(self.on_change)
        for w in (self.m_modes, self.m_fmin, self.m_fmax, self.b_modes):
            w.valueChanged.connect(self.on_change)
        self.b_case.currentIndexChanged.connect(self.on_change)

    def _build_nonlinear(self, lay):
        """Section "Nonlinear buckling with imperfection" (Study.nlbuckling, fea_core.imperfect)."""
        self.n_on = QCheckBox("compute nonlinear buckling with imperfection (GNIA / GMNIA)")
        help_tip(self.n_on, "Large-deformation analysis of the imperfect structure up to the ultimate load factor",
                 "<b>Nonlinear buckling with imperfection</b><p>The mesh is copied and its nodes are moved by the imperfection "
                 "(scaled linear buckling mode(s) of the case, amplitude e0). Then the load of the case is increased by the load "
                 "factor λ with large deformations (NLGEOM); with plastic materials this is a GMNIA, otherwise a GNIA. Results: "
                 "the load–displacement curve of the monitored point, the ultimate factor λ_u and the deformed shapes.</p>"
                 "<p>CalculiX has no arc-length (Riks) method – load control is used: the ultimate factor is the last converged "
                 "increment (a lower bound), the post-buckling branch is not traced.</p>"
                 "<p><b>The imperfection shape and amplitude are your responsibility</b> (EN 1993-1-5 Annex C, EN 1993-1-6, "
                 "EN 1993-1-1 5.3).</p>")
        g = Section("Buckling (nonlinear, imperfection)", check=self.n_on,
                    tip="Nonlinear buckling: imperfect geometry, large deformations, load–displacement curve",
                    help="Nonlinear buckling (GNIA / GMNIA): imperfection from the buckling modes, NLGEOM with load control, "
                         "ultimate load factor and load–displacement curve.")
        f = QFormLayout(g.body)
        self.n_case = QComboBox()
        self.n_case.setToolTip("Load case multiplied by the load factor λ")
        f.addRow("Load case", self.n_case)
        self.n_kind = QComboBox()
        for txt, key in (("buckling mode(s) of the case", "modes"), ("half-sine bow (set by script)", "bow"),
                         ("shape from a CSV file (set by script)", "file"), ("none (perfect geometry)", "none")):
            self.n_kind.addItem(txt, key)
        self.n_kind.setToolTip("Shape of the initial geometric imperfection")
        f.addRow("Imperfection", self.n_kind)
        self.n_modes = QLineEdit("1")
        help_tip(self.n_modes, "Buckling modes and weights, e.g. 1 or 1, 2*0.5 (negative weight = opposite sign)",
                 "Numbers of the linear buckling modes combined into the imperfection, optionally with a weight after *: "
                 "<i>1</i>, <i>1, 2*0.5</i>, <i>2*-1</i>. The combination is scaled so that its maximum is e0.")
        f.addRow("Modes", self.n_modes)
        self.n_amp_kind = QComboBox()
        self.n_amp_kind.addItem("e0 in mm", "mm")
        self.n_amp_kind.addItem("e0 = L / ratio", "ratio")
        self.n_amp_kind.setToolTip("Give the amplitude e0 directly or as a fraction of the length L")
        f.addRow("Amplitude", self.n_amp_kind)
        self.n_amp = dspin(1.0, 0.0, 1e4, 4, 0.1, " mm")
        self.n_amp.setToolTip("Imperfection amplitude e0 = max displacement of the shape [mm]")
        f.addRow("e0", self.n_amp)
        self.n_ratio = dspin(300.0, 1.0, 1e6, 1, 50.0)
        self.n_ratio.setToolTip("e0 = L / ratio (e.g. 1000 for a member bow, 200 for plates)")
        f.addRow("L /", self.n_ratio)
        self.n_len = dspin(0.0, 0.0, 1e7, 2, 10.0, " mm", special="largest model dimension")
        self.n_len.setToolTip("Reference length L for the ratio (0 = largest dimension of the model)")
        f.addRow("Length L", self.n_len)
        self.n_flip = QCheckBox("opposite direction of the imperfection")
        self.n_flip.setToolTip("Multiply the imperfection by −1")
        f.addRow(self.n_flip)
        self.n_max = dspin(0.0, 0.0, 1e9, 4, 1.0, special="1.1 × linear λ_cr")
        self.n_max.setToolTip("Maximum load factor of the ramp (0 = 1.1 × first linear buckling factor)")
        f.addRow("λ max", self.n_max)
        self.n_inc = QSpinBox()
        self.n_inc.setRange(2, 10000)
        self.n_inc.setValue(20)
        self.n_inc.setToolTip("Initial number of increments to λ max (ccx adapts them automatically)")
        f.addRow("Increments", self.n_inc)
        self.n_mon = QLineEdit("")
        self.n_mon.setPlaceholderText("automatic (max imperfection)")
        help_tip(self.n_mon, "Monitored point x, y, z [mm] (empty = node of the largest imperfection)",
                 "Point of the load–displacement curve; the nearest mesh node is used. Empty = node with the largest imperfection, "
                 "direction = direction of the imperfection there.")
        f.addRow("Monitored point", self.n_mon)
        self.n_note = QLabel(NLB_NOTE)
        self.n_note.setWordWrap(True)
        self.n_note.setStyleSheet("color:#b36b00")
        f.addRow(self.n_note)
        lay.addWidget(g)
        self.nlbuckling_group = g
        self.n_on.toggled.connect(self.on_change)
        self.n_flip.toggled.connect(self.on_change)
        for w in (self.n_amp, self.n_ratio, self.n_len, self.n_max, self.n_inc):
            w.valueChanged.connect(self.on_change)
        for w in (self.n_case, self.n_kind, self.n_amp_kind):
            w.currentIndexChanged.connect(self.on_change)
        for w in (self.n_modes, self.n_mon):
            w.editingFinished.connect(self.on_change)

    def _nl_widgets(self):
        return (self.n_case, self.n_kind, self.n_modes, self.n_amp_kind, self.n_amp, self.n_ratio, self.n_len, self.n_flip,
                self.n_max, self.n_inc, self.n_mon)

    def _fill_nonlinear(self, st):
        s = st.nlbuckling
        self.n_on.setChecked(bool(s.enabled))
        self.n_case.clear()
        self.n_case.addItem("as linear buckling / first case", None)
        for lc in st.load_cases:
            self.n_case.addItem(lc.name, lc.name)
        self.n_case.setProperty("_i18n_done", True)
        if s.case is not None and self.n_case.findData(s.case) < 0:
            self.n_case.addItem(f"{s.case} (does not exist)", s.case)
        self.n_case.setCurrentIndex(max(0, self.n_case.findData(s.case)) if s.case is not None else 0)
        self.n_kind.setCurrentIndex(max(0, self.n_kind.findData(s.imperfection)))
        self.n_modes.setText(format_modes(s.modes, s.weights))
        by_mm = bool(s.amplitude and float(s.amplitude) > 0)
        self.n_amp_kind.setCurrentIndex(0 if by_mm else 1)
        if by_mm:
            self.n_amp.setValue(float(s.amplitude))
        self.n_ratio.setValue(float(s.amplitude_ratio or 300.0))
        self.n_len.setValue(float(s.length or 0.0))
        self.n_flip.setChecked(float(s.sign or 1.0) < 0)
        self.n_max.setValue(float(s.max_factor or 0.0))
        self.n_inc.setValue(int(s.increments))
        self.n_mon.setText("" if s.monitor_point is None else ", ".join(f"{float(v):g}" for v in s.monitor_point))

    def nonlinear_settings(self, old):
        """NonlinearBucklingSettings from the form (fields the form does not show are kept from `old`)."""
        import dataclasses
        try:
            modes, weights = parse_modes(self.n_modes.text())
        except ValueError:
            modes, weights = list(old.modes), list(old.weights)
        mon = old.monitor_point
        txt = self.n_mon.text().strip()
        if not txt:
            mon = None
        else:
            try:
                vals = [float(v) for v in txt.replace(";", ",").split(",") if v.strip()]
                mon = vals if len(vals) == 3 else old.monitor_point
            except ValueError:
                pass
        by_mm = self.n_amp_kind.currentData() == "mm"
        return dataclasses.replace(
            old, enabled=self.n_on.isChecked(), case=self.n_case.currentData(), imperfection=self.n_kind.currentData(),
            modes=modes, weights=weights, amplitude=float(self.n_amp.value()) if by_mm else 0.0,
            amplitude_ratio=float(self.n_ratio.value()), length=float(self.n_len.value()) or None,
            sign=-1.0 if self.n_flip.isChecked() else 1.0, max_factor=float(self.n_max.value()) or None,
            increments=int(self.n_inc.value()), monitor_point=mon,
            monitor_dir=old.monitor_dir if mon == old.monitor_point else None)

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
            self._fill_nonlinear(st)
        finally:
            self._filling = False
        self._enable()

    def _enable(self):
        for w in (self.m_modes, self.m_fmin, self.m_fmax):
            w.setEnabled(self.m_on.isChecked())
        for w in (self.b_case, self.b_modes):
            w.setEnabled(self.b_on.isChecked())
        self.b_note.setVisible(self.b_on.isChecked())
        on = self.n_on.isChecked()
        for w in self._nl_widgets():
            w.setEnabled(on)
        if on:
            by_mm = self.n_amp_kind.currentData() == "mm"
            self.n_amp.setEnabled(by_mm)
            self.n_ratio.setEnabled(not by_mm)
            self.n_len.setEnabled(not by_mm)
            self.n_modes.setEnabled(self.n_kind.currentData() == "modes")
        self.n_note.setVisible(on)

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
        n = self.nonlinear_settings(st.nlbuckling) if (self.n_on.isChecked() or st.nlbuckling.enabled) else st.nlbuckling
        if m.to_dict() == st.modal.to_dict() and b.to_dict() == st.buckling.to_dict() \
                and n.to_dict() == st.nlbuckling.to_dict():
            return
        st.modal, st.buckling, st.nlbuckling = m, b, n
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
        self.nl_label = QLabel("")                    # nonlinear buckling: ultimate factor + curve
        self.nl_label.setWordWrap(True)
        v.addWidget(self.nl_label)
        self.nl_plot = LazyPlot()
        self.nl_plot.setToolTip("Load factor λ over the displacement of the monitored point")
        v.addWidget(self.nl_plot)
        self.nl_label.hide()
        self.nl_plot.hide()
        self._rows: list[tuple[int, int]] = []

    def plot_curve(self, info: dict) -> None:
        """Load–displacement curve of the nonlinear buckling (λ over u of the monitored point), λ_u and λ_cr marked."""
        c = info.get("curve") or {}
        lam, u = np.asarray(c.get("lambda") or [], float), np.asarray(c.get("u_mm") or [], float)
        if not len(lam):
            self.nl_plot.hide()
            return
        self.nl_plot.show()
        ax = self.nl_plot.axes()
        ax.plot(np.abs(u), lam, "o-", ms=3, lw=1.4, color="#2f6fd0", label="λ(u)")
        r = info.get("result") or {}
        if r.get("ultimate_factor") is not None:
            i = int(r.get("ultimate_index", int(np.argmax(lam))))
            ax.plot([abs(u[i])], [lam[i]], "s", color="#d23b3b", label=f"λ_u = {fmt(float(r['ultimate_factor']), 4)}")
        if info.get("linear_factor"):
            ax.axhline(float(info["linear_factor"]), ls="--", lw=1, color="#888888",
                       label=f"λ_cr = {fmt(float(info['linear_factor']), 4)}")
        ax.set_xlabel("Displacement of the monitored point [mm]")
        ax.set_ylabel("Load factor λ")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        self.nl_plot.canvas.draw_idle()

    def fill(self, results):
        t = self.table
        t.blockSignals(True)
        t.setRowCount(0)
        self._rows = []
        notes = []
        nl_info = ((getattr(results, "info", None) or {}).get("nlbuckling") if results is not None else None)
        for j, ms in enumerate(results.modes if results is not None else []):
            fr = mass_fractions(ms)
            cum = np.cumsum(fr, axis=0) if fr is not None else None
            if ms.kind == NLB_KIND and getattr(ms, "info", None) is None:
                ms.info = nl_info                     # Results.load does not attach it to the shapes
            for k in range(len(ms.values)):
                r = t.rowCount()
                t.insertRow(r)
                if ms.kind == NLB_KIND:
                    name = "nonlinear: imperfection" if k == 0 else f"nonlinear, increment {k}"
                else:
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
        if nl_info:
            r = nl_info.get("result") or {}
            txt = (f"<b>Nonlinear buckling ({html.escape(str(nl_info.get('kind', '')))})</b>: ultimate load factor "
                   f"<b>λ_u = {fmt(r.get('ultimate_factor'), 5)}</b> – {html.escape(str(nl_info.get('status_text', '')))}")
            if r.get("first_yield_factor") is not None:
                txt += f"<br>First yield at λ = {fmt(float(r['first_yield_factor']), 5)}"
            self.nl_label.setText(txt + f"<br><span style='color:#b36b00'>! {NLB_NOTE}</span>")
            self.nl_label.show()
            self.plot_curve(nl_info)
        else:
            self.nl_label.hide()
            self.nl_plot.hide()

    def on_row(self, r):
        if 0 <= r < len(self._rows):
            j, k = self._rows[r]
            self.win.select_result(f"mode:{j}:{k}")

    def select(self, j: int, k: int):
        if (j, k) in self._rows:
            self.table.blockSignals(True)
            self.table.selectRow(self._rows.index((j, k)))
            self.table.blockSignals(False)
