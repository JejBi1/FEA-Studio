"""Dynamics panel (step after Loads, F8 – spec 5a, 6.3): modal basis, damping (+ calculator from measurement), harmonic and transient
response with excitation (force / base), random vibration (ASD), shock (pulse / SRS), prestressed modal analysis, output probes
and added mass (point / surface).

The GUI only edits Study.dynamics and Study.masses; the calculation is done by fea_core.dynamics (modal superposition over the modes
from CalculiX; it runs with the whole study in the Solve step). Units: acceleration g or mm/s² (g = 9810 mm/s²), ASD
g²/Hz, mass kg (converted to t of the project), inertia kg·m² (converted to t·mm²).

The added mass is in this step (not in Materials / Loads): it is a property of the model that acts in statics
(weight, rotation), in the natural frequencies and in the dynamics – it is entered here, where it is needed most.
"""
from __future__ import annotations

from html import escape
import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QMessageBox, QPushButton, QSpinBox, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from fea_core.dynamics.model import Excitation, PointMass, Probe, SurfaceMass, G_MM

from .contact_panel import EditTable
from .widgets import (LazyPlot, Section, dspin, fit_rows, fmt, help_tip, panel_header,
                      scroll_body, scroll_to, set_placeholder, table)

APP = "FEA Studio"
C_MASS = (150, 60, 200)
C_PROBE = (20, 130, 60)
DAMPING_KINDS = [("modal", "modal ζ (all modes / table)"), ("rayleigh", "Rayleigh α·M + β·K (two frequencies)"),
                 ("structural", "structural (hysteretic) η – harmonic only"),
                 ("per_material", "per material (strain energy)")]
EXC_KINDS = [("force", "force (load case)"), ("base_acc", "base acceleration"), ("base_disp", "base displacement")]
UNITS = {"force": [("N", "× state")], "base_acc": [("g", "g"), ("mm/s2", "mm/s²")], "base_disp": [("mm", "mm")]}
DIRS = ["x", "y", "z", "custom direction"]
RANDOM_METHODS = [("dirlik", "Dirlik (1985)"), ("tb", "Tovo–Benasciutti (2005)"), ("nb", "narrow-band (Rayleigh)")]
PULSES = [("halfsine", "half-sine"), ("sawtooth", "sawtooth with terminal peak"), ("trapezoid", "trapezoid"),
          ("srs_table", "response spectrum (SRS table)")]
QTY = [("U", "displacement"), ("S", "stress"), ("A", "acceleration")]
UNCERTAIN = ("<span style='color:#d23b3b'><b>Damping is an uncertain input (sensitivity ±50 %)</b> – the response at resonance is inversely proportional to damping. Where it matters, measure it (hammer test, decay) – calculator “From measurement…”.</span>")
# test profiles – STRUCTURE only (decision 13 of the F8 spec): the numbers are entered by the user from the standard / specification
PROFILES = [
    ("", "– select a profile –", ""),
    ("60068-2-6", "IEC 60068-2-6 (Fc) – sinusoidal vibration",
     "Harmonic response with base acceleration excitation: frequency range, segment of constant displacement x̂ and constant acceleration a (crossover frequency f_c = √(a/x̂)/2π), sweep rate [oct/min] and number of sweeps → sweep fatigue. Enter the values from the standard / specification (history generator in the harmonic response excitation)."),
    ("60068-2-64", "IEC 60068-2-64 (Fh) – random vibration",
     "Random vibration: ASD points [Hz, g²/Hz] and slopes dB/oct, g_rms, test duration per axis, excitation axis. Enter the values from the standard / specification into the ASD editor."),
    ("60068-2-27", "IEC 60068-2-27 (Ea) – single shock",
     "Shock: pulse shape (half-sine, trapezoid, sawtooth), peak acceleration A [g], duration τ, axis; the number of shocks is informative for strength. Enter the values from the standard / specification."),
    ("60068-2-47", "IEC 60068-2-47 – specimen mounting",
     "The fixture must have its first natural frequency outside the tested range (criterion per the standard – UNVERIFIED). Check f₁ in the results (mode table)."),
]


def _f(v):
    return None if v is None or abs(v) < 1e-300 else float(v)


def excitation_text(e: Excitation) -> tuple[str, str]:
    kind = dict(EXC_KINDS).get(e.kind, e.kind)
    if e.kind == "force":
        where = f"state {e.case or '?'}"
    elif e.direction is not None:
        where = "direction (" + "; ".join(fmt(float(x), 3) for x in e.direction) + ")"
    else:
        where = f"axis {DIRS[int(e.dof or 3) - 1]}"
    return kind, where


class DampingCalcDialog(QDialog):
    """Relative damping from a measurement (fea_core.calc.damping): bandwidth −3 dB and logarithmic decrement; "Use"
    writes ζ into the Dynamics panel. Indicative values of ζ with sources (spec 5a.4).
    """

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.setWindowTitle("Damping from measurement")
        self.setModal(False)
        self.zeta = None
        v = QVBoxLayout(self)
        g = QGroupBox("Half-power bandwidth −3 dB (measured FRF)")
        help_tip(g, "−3 dB bandwidth from resonance curve",
                 "Frequency f₁ < f_n < f₂ where the amplitude drops to 1/√2 of the peak → ζ ≈ (f₂ − f₁)/(2f_n)")
        f = QFormLayout(g)
        self.fn = dspin(100.0, 1e-6, 1e7, 4, 1.0, " Hz")
        self.fn.setToolTip("Resonance frequency f_n [Hz] (peak of the measured FRF)")
        self.f1 = dspin(98.0, 1e-6, 1e7, 4, 1.0, " Hz")
        self.f1.setToolTip("Lower frequency f₁ [Hz] where amplitude = peak/√2 (−3 dB)")
        self.f2 = dspin(102.0, 1e-6, 1e7, 4, 1.0, " Hz")
        self.f2.setToolTip("Upper frequency f₂ [Hz] where amplitude = peak/√2 (−3 dB)")
        f.addRow("f_n", self.fn)
        f.addRow("f₁ (−3 dB)", self.f1)
        f.addRow("f₂ (−3 dB)", self.f2)
        b = QPushButton("Compute ζ")
        b.setToolTip("ζ ≈ (f₂ − f₁)/(2 f_n) – half-power method (Chopra ch. 3)")
        b.clicked.connect(self.calc_hp)
        self.btn_hp = b
        f.addRow(b)
        v.addWidget(g)
        g = QGroupBox("Logarithmic decrement (decay)")
        help_tip(g, "Free vibration decay after impact from amplitude x₀ and x_n",
                 "Free decay after an impact: amplitude x₀ and x_n n periods later → δ = ln(x₀/x_n)/n, ζ = δ/√(4π² + δ²)")
        f = QFormLayout(g)
        self.x0 = dspin(1.0, 1e-12, 1e12, 5, 0.1)
        self.x0.setToolTip("Amplitude x₀ (any unit – mm, g, V)")
        self.xn = dspin(0.5, 1e-12, 1e12, 5, 0.1)
        self.xn.setToolTip("Amplitude x_n n periods later (same unit as x₀)")
        self.n = QSpinBox()
        self.n.setRange(1, 10000)
        self.n.setValue(5)
        self.n.setToolTip("Number of periods between x₀ and x_n")
        f.addRow("x₀", self.x0)
        f.addRow("x_n", self.xn)
        f.addRow("n periods", self.n)
        b = QPushButton("Compute ζ")
        b.setToolTip("δ = ln(x₀/x_n)/n, ζ = δ/√(4π² + δ²) (Chopra ch. 2.2)")
        b.clicked.connect(self.calc_ld)
        self.btn_ld = b
        f.addRow(b)
        v.addWidget(g)
        self.out = QLabel("")
        self.out.setWordWrap(True)
        self.out.setTextFormat(Qt.RichText)
        v.addWidget(self.out)
        self.tab = table(["Structure", "ζ [%]", "Source"], 160, stretch_last=True,
                           placeholder="Table of typical damping values", rows=(4, 10))
        help_tip(self.tab, "Typical damping ratio ζ by structure type",
                 "Typical damping ratio ζ by structure type with sources (spec F8, 5a.4) – estimate only, experimental measurement takes precedence.")
        try:
            from fea_core.calc.damping import TABLE
            for name, (lo, hi), src in TABLE:
                r = self.tab.rowCount()
                self.tab.insertRow(r)
                for c, t in enumerate((name, fmt(lo, 3) if lo == hi else f"{fmt(lo, 3)}–{fmt(hi, 3)}", src)):
                    self.tab.setItem(r, c, QTableWidgetItem(t))
        except ImportError:
            pass
        v.addWidget(self.tab)
        h = QHBoxLayout()
        self.btn_use = QPushButton("Use ζ")
        self.btn_use.setToolTip("Write the computed ζ as the modal damping of the study (Dynamics step)")
        self.btn_use.setEnabled(False)
        self.btn_use.clicked.connect(self.use)
        b = QPushButton("Close")
        b.setToolTip("Close calculator")
        b.clicked.connect(self.close)
        h.addStretch(1)
        h.addWidget(self.btn_use)
        h.addWidget(b)
        v.addLayout(h)
        self.resize(560, 640)

    def _show(self, fn, *args):
        try:
            r = fn(*args)
        except Exception as e:  # noqa: BLE001 – CalcError
            self.out.setText(f"<span style='color:#d23b3b'>{e}</span>")
            self.zeta = None
            self.btn_use.setEnabled(False)
            return
        val = r.get("values", r)
        self.zeta = float(val["zeta"])
        warn = "".join(f"<br><span style='color:#b36b00'>! {w}</span>" for w in r.get("warnings", []))
        self.out.setText(f"ζ = <b>{fmt(100 * self.zeta, 4)} %</b>, Q = {fmt(val['Q'], 4)}, η = {fmt(val['eta'], 4)}"
                         + (f", δ = {fmt(val['delta'], 4)}" if "delta" in val else "")
                         + f"<br><small>{r.get('formula', '')}; {r.get('source', '')}</small>{warn}")
        self.btn_use.setEnabled(True)

    def calc_hp(self):
        from fea_core.calc.damping import half_power
        self._show(half_power, self.fn.value(), self.f1.value(), self.f2.value())

    def calc_ld(self):
        from fea_core.calc.damping import log_decrement
        self._show(log_decrement, self.x0.value(), self.xn.value(), self.n.value())

    def use(self):
        if self.zeta is not None:
            self.panel.set_zeta(self.zeta)


class ExcitationEditor(QGroupBox):
    """Excitation of the harmonic (domain "f") or transient (domain "t") response: the list of Excitation items and the form."""

    def __init__(self, panel, domain: str):
        super().__init__("Excitation")
        self.panel, self.domain = panel, domain
        self.items: list[Excitation] = []
        help_tip(self, "Excitation entries by force or base motion",
                 "Excitation items: force = load case × profile, or base motion (acceleration / displacement) in a direction; profile as a table " + ("by frequency" if domain == "f" else "in time"))
        f = QFormLayout(self)
        cols = ["Type", "Case / direction", "Unit", "Scale"] + (["Phase [°]"] if domain == "f" else []) + ["Waveform"]
        self.list = table(cols, 90, placeholder="No excitation yet – enter in the form below", rows=(2, 6))
        help_tip(self.list, "Excitation list – click loads entry into form",
                 "Excitation – click = load into the form. Several items are added together (forces with different phase).")
        self.list.currentCellChanged.connect(lambda r, *_: self.on_row(r))
        f.addRow(self.list)
        self.kind = QComboBox()
        for k, t in EXC_KINDS:
            self.kind.addItem(t, k)
        help_tip(self.kind, "Excitation by force (load case) or base motion",
                 "Force = load case (Loads step) × time history × scale; base acceleration / displacement = motion of all supports in direction (results relative to base, absolute optional in results).")
        self.kind.currentIndexChanged.connect(self.on_kind)
        f.addRow("Type", self.kind)
        self.case = QComboBox()
        self.case.setToolTip("Load case whose load is the amplitude of the force excitation")
        f.addRow("State", self.case)
        hd = QHBoxLayout()
        self.dir = QComboBox()
        self.dir.addItems(DIRS)
        self.dir.setCurrentIndex(2)
        self.dir.setToolTip("Direction of base motion: global axis x, y, z or a custom direction (unit vector)")
        self.dir.currentIndexChanged.connect(self.on_kind)
        hd.addWidget(self.dir)
        self.dvec = [dspin(v, -1e6, 1e6, 4, 0.1) for v in (0.0, 0.0, 1.0)]
        for s_, c in zip(self.dvec, "xyz"):
            s_.setToolTip(f"Component {c} of the custom excitation direction (normalized to unit length)")
            hd.addWidget(s_)
        f.addRow("Direction", hd)
        hu = QHBoxLayout()
        self.unit = QComboBox()
        help_tip(self.unit, "Amplitude unit (g, mm/s², mm, case multiplier)",
                 "Amplitude unit: g (= 9810 mm/s²) or mm/s² for acceleration, mm for displacement, load multiplier for force.")
        self.scale = dspin(1.0, -1e9, 1e9, 5, 0.1)
        self.scale.setToolTip("Excitation amplitude (scale) in the unit; the table history is multiplied by it")
        hu.addWidget(self.scale, 1)
        hu.addWidget(self.unit)
        f.addRow("Amplitude", hu)
        self.phase = dspin(0.0, -360.0, 360.0, 2, 15.0, " °")
        self.phase.setToolTip("Excitation phase [°] (forces with different phase – e.g. 90° = shifted by a quarter period)")
        if domain == "f":
            f.addRow("Phase", self.phase)
        x = "f [Hz]" if domain == "f" else "t [s]"
        self.tab = EditTable([x, "multiplier"], f"Amplitude profile {('by frequency' if domain == 'f' else 'in time')}: pairs [{x}, multiple] linear between the points"
                                             + ("; empty table = constant 1" if domain == "f" else
                                                "; 0 after the end of the table, empty = step 1"))
        f.addRow(self.tab)
        hr = QHBoxLayout()
        for txt, slot, tip in (("+ row", self.tab.add_row, "Add history row"),
                               ("− row", self.tab.remove_row, "Delete the selected history row")):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            hr.addWidget(b)
        f.addRow(hr)
        # presets of the course
        hp = QHBoxLayout()
        self.preset = QComboBox()
        if domain == "t":
            for k, t in (("step", "step"), ("rect", "rectangular pulse"), ("halfsine", "half-sine pulse"),
                         ("scurve", "displacement S-curve (axis start)")):
                self.preset.addItem(t, k)
            help_tip(self.preset, "Time history preset: step, pulse, or S-curve",
                     "Time history preset: step (constant from t = 0), rectangular pulse of duration τ, half-sine of duration τ, base displacement S-curve (axis motion by distance = amplitude over duration τ, cycloid).")
        else:
            self.preset.addItem("IEC 60068-2-6: const. displacement / acceleration", "iec6")
            help_tip(self.preset, "Sine test profile per IEC 60068-2-6",
                     "Sine test profile: below crossover frequency constant displacement x̂, above it constant acceleration a (excitation amplitude = a in g; table = frequency factor). Specify x̂ and a from standard / test spec.")
        hp.addWidget(self.preset, 1)
        self.p_tau = dspin(0.0, 0.0, 1e6, 6, 0.001, " s" if domain == "t" else " mm")
        self.p_tau.setToolTip("Duration τ of the pulse / start [s]" if domain == "t" else
                              "Displacement amplitude x̂ [mm] in the constant-displacement segment (from the standard)")
        hp.addWidget(QLabel("τ" if domain == "t" else "x̂"))
        hp.addWidget(self.p_tau)
        b = QPushButton("Insert history")
        b.setToolTip("Fill the history table by the preset (for the S-curve also the type base displacement)")
        b.clicked.connect(self.apply_preset)
        self.btn_preset = b
        hp.addWidget(b)
        f.addRow(hp)
        hb = QHBoxLayout()
        for txt, slot, tip in (("Add excitation", self.add, "Add an excitation item from the form"),
                               ("Save changes", self.update, "Overwrite the selected item with the form"),
                               ("Delete", self.delete, "Delete the selected excitation item")):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            hb.addWidget(b)
        f.addRow(hb)
        self.on_kind()

    def set_cases(self, names):
        cur = self.case.currentText()
        self.case.clear()
        self.case.addItems(names)
        if cur in names:
            self.case.setCurrentText(cur)

    def set_items(self, items):
        self.items = list(items)
        self.fill()

    def fill(self):
        t = self.list
        t.blockSignals(True)
        cur = t.currentRow()
        t.setRowCount(0)
        for e in self.items:
            r = t.rowCount()
            t.insertRow(r)
            kind, where = excitation_text(e)
            unit = {"g": "g", "mm/s2": "mm/s²", "mm": "mm", "N": "× state"}.get(e.unit, e.unit)
            vals = [kind, where, unit, fmt(e.scale, 5)] + ([fmt(e.phase, 4)] if self.domain == "f" else []) + \
                [f"{len(e.table)} points" if e.table else ("constant" if self.domain == "f" else "step")]
            for k, v in enumerate(vals):
                t.setItem(r, k, QTableWidgetItem(v))
        if 0 <= cur < t.rowCount():
            t.selectRow(cur)
        t.blockSignals(False)

    def on_kind(self, *_):
        k = self.kind.currentData()
        self.case.setEnabled(k == "force")
        self.dir.setEnabled(k != "force")
        custom = k != "force" and self.dir.currentIndex() == 3
        for s_ in self.dvec:
            s_.setEnabled(custom)
        cur = self.unit.currentData()
        self.unit.blockSignals(True)
        self.unit.clear()
        for u, t in UNITS.get(k, []):
            self.unit.addItem(t, u)
        i = self.unit.findData(cur)
        self.unit.setCurrentIndex(max(i, 0))
        self.unit.blockSignals(False)

    def from_form(self) -> Excitation | None:
        k = self.kind.currentData()
        if k == "force" and not self.case.currentText():
            QMessageBox.warning(self, APP, "Force excitation needs a load case (Loads step).")
            return None
        try:
            tab = [[float(a), float(b)] for a, b in self.tab.values()]
        except (TypeError, ValueError):
            QMessageBox.warning(self, APP, "History: fill in pairs of numbers.")
            return None
        if tab and any(b[0] <= a[0] for a, b in zip(tab, tab[1:])):
            QMessageBox.warning(self, APP, "History: the first column must increase.")
            return None
        e = Excitation(kind=k, case=self.case.currentText() if k == "force" else None, table=tab,
                       unit=self.unit.currentData() or "g", scale=self.scale.value(),
                       phase=self.phase.value() if self.domain == "f" else 0.0)
        if k == "force":
            e.dof, e.unit = None, "N"
        elif self.dir.currentIndex() == 3:
            d = np.array([s_.value() for s_ in self.dvec], float)
            if np.linalg.norm(d) < 1e-12:
                QMessageBox.warning(self, APP, "The custom excitation direction must not be zero.")
                return None
            e.dof, e.direction = None, (d / np.linalg.norm(d)).tolist()
        else:
            e.dof = self.dir.currentIndex() + 1
        return e

    def on_row(self, r):
        if not (0 <= r < len(self.items)):
            return
        e = self.items[r]
        self.kind.setCurrentIndex(max(0, self.kind.findData(e.kind)))
        if e.case:
            self.case.setCurrentText(e.case)
        if e.direction is not None:
            self.dir.setCurrentIndex(3)
            for s_, v in zip(self.dvec, e.direction):
                s_.setValue(float(v))
        elif e.dof:
            self.dir.setCurrentIndex(int(e.dof) - 1)
        self.on_kind()
        self.unit.setCurrentIndex(max(0, self.unit.findData(e.unit)))
        self.scale.setValue(e.scale)
        self.phase.setValue(e.phase)
        self.tab.set_rows(e.table)

    def add(self):
        e = self.from_form()
        if e is not None:
            self.items.append(e)
            self.panel.excitation_changed(self)
            self.list.selectRow(len(self.items) - 1)

    def update(self):
        r = self.list.currentRow()
        if not (0 <= r < len(self.items)):
            QMessageBox.warning(self, APP, "Select an excitation item in the table.")
            return
        e = self.from_form()
        if e is not None:
            self.items[r] = e
            self.panel.excitation_changed(self)

    def delete(self):
        r = self.list.currentRow()
        if 0 <= r < len(self.items):
            self.items.pop(r)
            self.panel.excitation_changed(self)

    def apply_preset(self):
        k = self.preset.currentData()
        tau = self.p_tau.value()
        if k == "step":
            t_end = self.panel.tr_tend.value()
            self.tab.set_rows([[0.0, 1.0], [t_end, 1.0]])
            return
        if k == "iec6":
            self._iec6(tau)
            return
        if tau <= 0:
            QMessageBox.warning(self, APP, "Enter the pulse / ramp-up duration τ (> 0).")
            return
        dt = max(self.panel.tr_dt.value(), tau / 1000.0)
        if k == "rect":
            rows = [[0.0, 1.0], [tau, 1.0], [tau + dt, 0.0]]
        elif k == "halfsine":
            t = np.linspace(0.0, tau, 41)
            rows = [[float(x), float(math.sin(math.pi * x / tau))] for x in t] + [[tau + dt, 0.0]]
        else:                                   # S-curve of the motion (cycloid): s(t) = t/τ − sin(2πt/τ)/(2π)
            t = np.linspace(0.0, tau, 51)
            rows = [[float(x), float(x / tau - math.sin(2 * math.pi * x / tau) / (2 * math.pi))] for x in t]
            rows[-1][1] = 1.0
            self.kind.setCurrentIndex(self.kind.findData("base_disp"))
            self.on_kind()
        self.tab.set_rows(rows)

    def _iec6(self, xhat):
        p = self.panel
        f0, f1 = p.h_fmin.value(), p.h_fmax.value()
        a = self.scale.value()
        if xhat <= 0 or a <= 0 or not 0 < f0 < f1:
            QMessageBox.warning(self, APP, "IEC 60068-2-6 profile: enter from the standard / specification the displacement amplitude x̂ [mm] (field x̂), the acceleration amplitude a [g] (Amplitude) and the frequency range of the harmonic response f_min > 0.")
            return
        fc = math.sqrt(a * G_MM / xhat) / (2 * math.pi)
        fs = sorted(set([f0, f1] + ([fc] if f0 < fc < f1 else []) +
                        list(np.geomspace(f0, min(fc, f1), 16)) if fc > f0 else [f0, f1]))
        rows = [[float(f), float(min(1.0, xhat * (2 * math.pi * f) ** 2 / (a * G_MM)))] for f in fs]
        self.kind.setCurrentIndex(self.kind.findData("base_acc"))
        self.on_kind()
        self.unit.setCurrentIndex(max(0, self.unit.findData("g")))
        self.tab.set_rows(rows)
        p.win.log_msg(f"IEC 60068-2-6 profile: crossover frequency f_c = {fmt(fc, 4)} Hz (x̂ {fmt(xhat)} mm, a {fmt(a)} g) – save the excitation with the “Add excitation” button")


class DynamicsPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self._filling = False
        self._asd: list = []
        self._calc = None
        self._last_study = None
        self._scroll, lay = scroll_body(self)
        lay.addWidget(panel_header("Dynamics", "Natural vibration, harmonic and transient response, random vibration, shock and added mass – optional"))
        self.warn = QLabel(UNCERTAIN)
        self.warn.setWordWrap(True)
        self.warn.setTextFormat(Qt.RichText)
        help_tip(self.warn, "Damping is an uncertain input with ±50% sensitivity",
                 "Damping is entered by estimate – dynamic response strongly depends on it (±50% sensitivity).")
        lay.addWidget(self.warn)

        # ---------------------------------------------------------- modal basis
        self.on = QCheckBox("compute dynamics")
        help_tip(self.on, "Enable structural dynamics (natural frequencies and responses)",
                 "<b>Dynamics</b><p>The program computes the mode shapes (CalculiX *FREQUENCY) and from them by modal superposition the harmonic, transient and random response and shock. The static correction adds the contribution of higher modes. Results: fields at frequency / time, envelopes, RMS, damage, charts of FRF and time histories in probes.</p><p>Choose the number of modes to cover 1.5–2× the highest excitation frequency and a sum of effective masses in the excited direction ≥ 90 % (the program checks it).</p>")
        self.sec_modal = Section("Modal basis", key="dyn.modal", check=self.on,
                                 tip="Mode shapes from CalculiX and modal basis setup",
                                 help=self.on.whatsThis())
        f = QFormLayout(self.sec_modal.body)
        hm = QHBoxLayout()
        self.modes = QSpinBox()
        self.modes.setRange(1, 1000)
        self.modes.setValue(30)
        help_tip(self.modes, "Number of mode shapes in modal basis",
                 "Number of mode shapes in modal basis (recommended to cover 1.5–2× highest excitation frequency).")
        self.fmax = dspin(0.0, 0.0, 1e7, 2, 100.0, " Hz", special="no limit")
        self.fmax.setToolTip("Upper limit of the natural frequency search [Hz] (0 = number of modes only)")
        hm.addWidget(QLabel("modes"))
        hm.addWidget(self.modes, 1)
        hm.addWidget(QLabel("f_max"))
        hm.addWidget(self.fmax, 1)
        f.addRow(hm)
        self.verify = QCheckBox("verify in probes with CalculiX (reference)")
        help_tip(self.verify, "Independent probe verification via CalculiX solve",
                 "Additional ccx *STEADY STATE DYNAMICS / *MODAL DYNAMIC at probes – independent check of modal superposition (longer run).")
        f.addRow(self.verify)
        self.frames = QSpinBox()
        self.frames.setRange(1, 10000)
        self.frames.setValue(50)
        self.frames.setToolTip("Number of transient response field frames stored in the results")
        f.addRow("Transient response frames", self.frames)
        lay.addWidget(self.sec_modal)

        # ---------------------------------------------------------- damping
        self.sec_damping = Section("Damping", key="dyn.damping", collapsed=False,
                                   tip="Modal system damping for response computation",
                                   help="<b>Damping</b><p>Modal system damping is an uncertain input; resonance response is inversely proportional to it. Measure it where critical (impact hammer, decay) – \"From measurement…\" calculator.</p>")
        f = QFormLayout(self.sec_damping.body)
        self.f_damping = f
        self.d_kind = QComboBox()
        for k, t in DAMPING_KINDS:
            self.d_kind.addItem(t, k)
        help_tip(self.d_kind, "Method for specifying modal system damping",
                 "Modal ζ: uniform for all modes, or table vs. frequency.<br>Rayleigh: α·M + β·K from ζ at f₁ and f₂.<br>Structural: loss factor η (harmonic response only; η = 2ζ).<br>By materials: material ζ weighted by mode strain energy.")
        f.addRow("Type", self.d_kind)
        hz = QHBoxLayout()
        self.d_zeta = dspin(2.0, 0.0, 99.0, 4, 0.5, " %")
        help_tip(self.d_zeta, "Modal system damping ratio ζ [%]",
                 "Damping ratio ζ [%] (welded steel ~0.3%, bolted ~1–3%, PLA 1–2%, assembled PCB 2.5–5%, SRS convention 5%).")
        hz.addWidget(self.d_zeta, 1)
        b = QPushButton("From measurement…")
        help_tip(b, "Damping calculator from measurement (bandwidth, decrement)",
                 "Damping calculator from measurement: −3 dB bandwidth, logarithmic decrement; typical values.")
        b.clicked.connect(self.open_calc)
        self.btn_calc = b
        hz.addWidget(b)
        self.hz_widget = QWidget()
        self.hz_widget.setLayout(hz)
        self.hz_widget.setContentsMargins(0, 0, 0, 0)
        f.addRow("ζ", self.hz_widget)
        self.d_eta = dspin(0.0, 0.0, 2.0, 4, 0.01, special="= 2ζ")
        help_tip(self.d_eta, "Loss factor η for structural damping (0 = 2ζ)",
                 "Loss factor η (structural damping, harmonic response only); 0 = 2ζ.<br>Note: η = 2ζ – at resonance F/(kη) = F/(2kζ).")
        f.addRow("η", self.d_eta)
        hf = QHBoxLayout()
        self.d_f1 = dspin(0.0, 0.0, 1e7, 3, 10.0, " Hz")
        self.d_f1.setToolTip("Rayleigh: lower frequency f₁ [Hz] where the given ζ applies")
        self.d_f2 = dspin(0.0, 0.0, 1e7, 3, 10.0, " Hz")
        help_tip(self.d_f2, "Upper frequency f₂ [Hz] where specified ζ applies",
                 "Rayleigh: upper frequency f₂ [Hz] where specified ζ applies (damping is lower between f₁ and f₂).")
        hf.addWidget(QLabel("f₁"))
        hf.addWidget(self.d_f1, 1)
        hf.addWidget(QLabel("f₂"))
        hf.addWidget(self.d_f2, 1)
        self.hf_widget = QWidget()
        self.hf_widget.setLayout(hf)
        self.hf_widget.setContentsMargins(0, 0, 0, 0)
        f.addRow("Rayleigh", self.hf_widget)
        self.d_table = EditTable(["f [Hz]", "ζ [%]"], "Modal ζ by mode frequency",
                                 detailed="Modal ζ vs. mode frequency (linear between points); empty table = ζ for all modes")
        self.d_table.setMinimumHeight(80)
        fit_rows(self.d_table, 2, 8)
        set_placeholder(self.d_table, "Uniform ζ for all modes (add rows for frequency dependence)")
        f.addRow("Table ζ(f)", self.d_table)
        self.d_mat = EditTable(["Material", "ζ [%]"], "Damping ζ of each material [%]",
                               detailed="Per material: ζ of each material [%] (weighted by the strain energy in the mode – Johnson & Kienholz 1982)")
        self.d_mat.setMinimumHeight(80)
        fit_rows(self.d_mat, 2, 6)
        set_placeholder(self.d_mat, "No materials yet")
        f.addRow("ζ of materials", self.d_mat)
        lay.addWidget(self.sec_damping)

        # ---------------------------------------------------------- test profiles
        self.sec_profile = Section("Test profile (structure)", key="dyn.profile", collapsed=True,
                                   tip="Response preset per IEC 60068-2-x test standard",
                                   help="<b>Test profile</b><p>Response preset per test standard – template only; enter numerical values from standard or test specification.</p>")
        f = QFormLayout(self.sec_profile.body)
        self.profile = QComboBox()
        for k, t, d in PROFILES:
            self.profile.addItem(t, k)
            if d:
                self.profile.setItemData(self.profile.count() - 1, d, Qt.ToolTipRole)
        help_tip(self.profile, "Response preset per IEC 60068-2-x test standard",
                 "Environmental testing standards IEC 60068-2-x: enables the appropriate response and highlights inputs. Numerical levels, ranges, and durations are NOT included – specify from standard or test spec.")
        self.profile.currentIndexChanged.connect(self.on_profile_combo)
        f.addRow("Standard", self.profile)
        self.profile_note = QLabel("")
        self.profile_note.setWordWrap(True)
        f.addRow(self.profile_note)
        b = QPushButton("Prepare structure")
        help_tip(b, "Enable dynamics and corresponding response per standard",
                 "Enable dynamics and corresponding response (without numbers – enter from standard).")
        b.clicked.connect(self.apply_profile)
        self.btn_profile = b
        f.addRow(b)
        lay.addWidget(self.sec_profile)

        # ---------------------------------------------------------- harmonic
        self.h_on = QCheckBox("compute harmonic response")
        help_tip(self.h_on, "Enable harmonic response (modal superposition across frequency range)",
                 "Steady-state vibration under sine excitation across frequency range: FRF, field at frequency, envelope, sweep fatigue.")
        self.sec_harmonic = Section("Harmonic response (sine)", key="dyn.harmonic", check=self.h_on,
                                    tip="Steady-state vibration under sine excitation across frequency range",
                                    help="<b>Harmonic response</b><p>Steady-state vibration under sine excitation across frequency range: FRF, field at frequency, envelope, sweep fatigue.</p>")
        f = QFormLayout(self.sec_harmonic.body)
        hr = QHBoxLayout()
        self.h_fmin = dspin(5.0, 0.0, 1e7, 3, 1.0, " Hz")
        self.h_fmin.setToolTip("Lower frequency of the range [Hz]")
        self.h_fmax = dspin(500.0, 0.0, 1e7, 3, 10.0, " Hz")
        self.h_fmax.setToolTip("Upper frequency of the range [Hz]")
        hr.addWidget(self.h_fmin, 1)
        hr.addWidget(QLabel("–"))
        hr.addWidget(self.h_fmax, 1)
        f.addRow("Range", hr)
        hp = QHBoxLayout()
        self.h_points = QSpinBox()
        self.h_points.setRange(2, 10000)
        self.h_points.setValue(20)
        self.h_points.setToolTip("Number of points between natural frequencies (as in ccx – denser near resonances)")
        self.h_bias = dspin(3.0, 1.0, 100.0, 2, 0.5)
        self.h_bias.setToolTip("Clustering of points towards natural frequencies (ccx BIAS, 1 = uniform)")
        hp.addWidget(QLabel("points"))
        hp.addWidget(self.h_points, 1)
        hp.addWidget(QLabel("bias"))
        hp.addWidget(self.h_bias, 1)
        f.addRow(hp)
        self.h_static = QCheckBox("static correction of higher modes")
        help_tip(self.h_static, "Static correction for truncated higher modes",
                 "Contribution of truncated higher modes as static response (mode acceleration) – more accurate stresses.")
        f.addRow(self.h_static)
        hs = QHBoxLayout()
        self.h_rate = dspin(1.0, 0.0, 1e3, 3, 0.5, " okt/min", special="no fatigue")
        help_tip(self.h_rate, "Logarithmic sweep rate [oct/min]",
                 "Logarithmic sweep rate [oct/min] → sweep fatigue (dn/df = 60/(R·ln 2) cycles/Hz); 0 = do not compute sweep fatigue.")
        self.h_sweeps = QSpinBox()
        self.h_sweeps.setRange(1, 100000)
        self.h_sweeps.setToolTip("Number of sweeps (up and down = 2)")
        hs.addWidget(self.h_rate, 1)
        hs.addWidget(QLabel("sweeps"))
        hs.addWidget(self.h_sweeps)
        f.addRow("Sweep", hs)
        self.h_exc = ExcitationEditor(self, "f")
        f.addRow(self.h_exc)
        lay.addWidget(self.sec_harmonic)

        # ---------------------------------------------------------- transient
        self.tr_on = QCheckBox("compute transient response")
        help_tip(self.tr_on, "Enable transient response in time domain",
                 "Enable transient response (exact solution of modal equations for piecewise linear excitation).")
        self.sec_transient = Section("Transient response (in time)", key="dyn.transient", check=self.tr_on,
                                     tip="Response to time-history excitation: step, pulse, axis start",
                                     help="<b>Transient response</b><p>Response to time-history excitation: step, pulse, axis start (settling time), probe time traces.</p>")
        f = QFormLayout(self.sec_transient.body)
        ht = QHBoxLayout()
        self.tr_tend = dspin(0.1, 1e-9, 1e6, 6, 0.01, " s")
        self.tr_tend.setToolTip("Analysis duration t_end [s]")
        self.tr_dt = dspin(1e-4, 1e-12, 1e3, 8, 1e-4, " s")
        self.tr_dt.setToolTip("Output time step dt [s] – recommended ≤ T_min/20 of the highest important mode")
        ht.addWidget(QLabel("t_end"))
        ht.addWidget(self.tr_tend, 1)
        ht.addWidget(QLabel("dt"))
        ht.addWidget(self.tr_dt, 1)
        f.addRow(ht)
        hm2 = QHBoxLayout()
        self.tr_method = QComboBox()
        self.tr_method.addItem("modal (exact, fast)", "modal")
        self.tr_method.addItem("direct integration ccx *DYNAMIC (HHT-α)", "direct")
        help_tip(self.tr_method, "Solution method: modal superposition or direct integration",
                 "Modal = exact solution of modal equations; direct integration only for non-linearities (~25× slower).")
        self.tr_alpha = dspin(-0.05, -1 / 3, 0.0, 3, 0.01)
        help_tip(self.tr_alpha, "Numerical damping parameter HHT-α (−1/3 to 0)",
                 "HHT-α for direct integration (−1/3 to 0, default −0.05 – high frequency numerical damping).")
        hm2.addWidget(self.tr_method, 1)
        hm2.addWidget(QLabel("α"))
        hm2.addWidget(self.tr_alpha)
        f.addRow("Method", hm2)
        self.tr_settle = dspin(0.0, 0.0, 1e6, 6, 0.001, " mm", special="do not compute")
        help_tip(self.tr_settle, "Settling tolerance [mm] at probes",
                 "Settling tolerance [mm] at probes – duration after which displacement stays within ±tolerance (decay after axis move).")
        f.addRow("Settling tolerance", self.tr_settle)
        self.tr_fat = QCheckBox("fatigue from the history (modal stresses as unit cases)")
        self.tr_fat.setToolTip("Prepare fatigue from the transient response (q_j(t) as load history channels)")
        f.addRow(self.tr_fat)
        self.tr_exc = ExcitationEditor(self, "t")
        f.addRow(self.tr_exc)
        lay.addWidget(self.sec_transient)

        # ---------------------------------------------------------- random vibration
        self.r_on = QCheckBox("compute random vibration")
        help_tip(self.r_on, "Enable random vibration (modal superposition from ASD)",
                 "Enable random vibration (modal superposition in Python – full CQC formulation).")
        self.sec_random = Section("Random vibration (ASD)", key="dyn.random", check=self.r_on,
                                  tip="Base acceleration spectral density (ASD) → RMS displacement and stress",
                                  help="<b>Random vibration</b><p>Base acceleration spectral density (ASD) → RMS displacements and stresses, 3σ, spectral fatigue, life.</p>")
        f = QFormLayout(self.sec_random.body)
        self.r_tab = EditTable(["f [Hz]", "W [g²/Hz]", "slope to next [dB/oct]"],
                               "Base acceleration ASD profile: points [f, W]",
                               detailed="ASD profile as in standards: points [f, W]; missing W is computed from section slope (dB/oct) to adjacent point.<br>Button 'Compute and use' saves profile.")
        fit_rows(self.r_tab, 2, 6)
        set_placeholder(self.r_tab, "Enter the profile points [f, W]")
        f.addRow(self.r_tab)
        hr2 = QHBoxLayout()
        for txt, slot, tip in (("+ row", self.r_tab.add_row, "Add profile point"),
                               ("− row", self.r_tab.remove_row, "Delete the selected profile point")):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            hr2.addWidget(b)
        self.r_grms_target = dspin(0.0, 0.0, 1e4, 4, 0.5, " g", special="no rescaling")
        self.r_grms_target.setToolTip("Target g_rms – the whole profile is rescaled (0 = keep the levels)")
        hr2.addWidget(QLabel("g_rms"))
        hr2.addWidget(self.r_grms_target)
        b = QPushButton("Compute and apply")
        help_tip(b, "Compute profile from slopes and save to study",
                 "Calculate missing values from slopes, rescale to g_rms, and save profile to study.")
        b.clicked.connect(self.apply_asd)
        self.btn_asd = b
        hr2.addWidget(b)
        f.addRow(hr2)
        self.r_info = QLabel("")
        self.r_info.setWordWrap(True)
        f.addRow(self.r_info)
        self.r_plot = LazyPlot()
        self.r_plot.setToolTip("ASD profile on a log-log scale")
        f.addRow(self.r_plot)
        hd = QHBoxLayout()
        self.r_dir = QComboBox()
        self.r_dir.addItems(DIRS[:3])
        self.r_dir.setCurrentIndex(2)
        self.r_dir.setToolTip("Axis of random vibration excitation (global x, y, z)")
        self.r_dur = dspin(1.0, 1e-6, 1e9, 4, 0.5, " h")
        self.r_dur.setToolTip("Test duration per axis [h] → spectral fatigue and life")
        hd.addWidget(self.r_dir)
        hd.addWidget(QLabel("duration"))
        hd.addWidget(self.r_dur, 1)
        f.addRow("Axis", hd)
        self.r_method = QComboBox()
        for k, t in RANDOM_METHODS:
            self.r_method.addItem(t, k)
        help_tip(self.r_method, "Spectral fatigue method (Dirlik, Tovo–Benasciutti)",
                 "Spectral fatigue: Dirlik (default, broadband), Tovo–Benasciutti, narrow-band (conservative).")
        f.addRow("Fatigue method", self.r_method)
        lay.addWidget(self.sec_random)

        # ---------------------------------------------------------- shock
        self.s_on = QCheckBox("compute shock")
        help_tip(self.s_on, "Enable shock response spectrum (SRS) analysis",
                 "Enable shock response spectrum analysis (and optionally transient pulse response).")
        self.sec_shock = Section("Shock (response spectrum)", key="dyn.shock", check=self.s_on,
                                 tip="Base acceleration pulse → response spectrum (SRS) → modal peaks",
                                 help="<b>Shock</b><p>Base acceleration pulse → response spectrum (SRS) → modal peaks, SRSS / CQC combination.</p>")
        f = QFormLayout(self.sec_shock.body)
        self.s_pulse = QComboBox()
        for k, t in PULSES:
            self.s_pulse.addItem(t, k)
        self.s_pulse.setToolTip("Shape of the base acceleration pulse, or the response spectrum directly (SRS table)")
        f.addRow("Pulse", self.s_pulse)
        hs2 = QHBoxLayout()
        self.s_A = dspin(0.0, 0.0, 1e6, 3, 5.0, " g")
        self.s_A.setToolTip("Peak acceleration of the pulse A [g] (from the standard / specification)")
        self.s_tau = dspin(0.0, 0.0, 1e4, 4, 1.0, " ms")
        self.s_tau.setToolTip("Pulse duration τ [ms] (from the standard / specification)")
        self.s_rise = dspin(0.0, 0.0, 1e4, 4, 0.5, " ms", special="default")
        self.s_rise.setToolTip("Trapezoid: rise time [ms] (0 = default)")
        for lbl, w in (("A", self.s_A), ("τ", self.s_tau), ("rise", self.s_rise)):
            hs2.addWidget(QLabel(lbl))
            hs2.addWidget(w, 1)
        f.addRow(hs2)
        hs3 = QHBoxLayout()
        self.s_dir = QComboBox()
        self.s_dir.addItems(DIRS[:3])
        self.s_dir.setCurrentIndex(2)
        self.s_dir.setToolTip("Shock axis (global x, y, z)")
        self.s_comb = QComboBox()
        self.s_comb.addItem("CQC (closely spaced modes)", "cqc")
        self.s_comb.addItem("SRSS", "srss")
        help_tip(self.s_comb, "Combination of modal peaks (CQC or SRSS)",
                 "Combination of modal peaks: CQC (Der Kiureghian 1981, closely spaced mode correlation) or SRSS.")
        hs3.addWidget(self.s_dir)
        hs3.addWidget(self.s_comb, 1)
        f.addRow("Axis, combination", hs3)
        self.s_mm = QCheckBox("missing mass (ZPA) as a static case")
        help_tip(self.s_mm, "Missing mass (ZPA) as static case",
                 "Mass not captured in modes × ZPA as static response, combined with SRSS (US NRC RG 1.92).")
        f.addRow(self.s_mm)
        self.s_tr = QCheckBox("also transient response to the pulse")
        self.s_tr.setToolTip("Exact time response to the same pulse (histories in probes, envelope)")
        f.addRow(self.s_tr)
        self.s_srs = EditTable(["f [Hz]", "S_a [g]"], "SRS response spectrum: pairs [f, S_a]",
                               detailed="Response spectrum (pulse “SRS table”): pairs [f, S_a] – the last value = ZPA")
        self.s_srs.setMinimumHeight(80)
        fit_rows(self.s_srs, 2, 6)
        set_placeholder(self.s_srs, "Enter the response spectrum points [f, S_a]")
        f.addRow("SRS", self.s_srs)
        lay.addWidget(self.sec_shock)

        # ---------------------------------------------------------- preload
        self.p_on = QCheckBox("modes after a static case")
        help_tip(self.p_on, "Modal basis after selected static load case",
                 "Modal basis from stiffness at the end of selected load case (*STEP, PERTURBATION).")
        self.sec_prestress = Section("Prestressed modal analysis", key="dyn.prestress", check=self.p_on,
                                     tip="Mode shapes after static load case (stress stiffening)",
                                     help="<b>Prestressed modal analysis</b><p>Mode shapes after static case (stress stiffening – bolt preload, tension, centrifugal force).</p>")
        f = QFormLayout(self.sec_prestress.body)
        self.p_nlgeom = QCheckBox("prestress with NLGEOM")
        help_tip(self.p_nlgeom, "Preload with geometric non-linearity NLGEOM",
                 "Preload with geometric non-linearity (CalculiX computes stress stiffness on reference geometry; Abaqus updates geometry).")
        self.p_case = QComboBox()
        self.p_case.setToolTip("Preload load case")
        f.addRow(self.p_nlgeom)
        f.addRow("State", self.p_case)
        lay.addWidget(self.sec_prestress)

        # ---------------------------------------------------------- sondy
        self.sec_probes = Section("Output probes", key="dyn.probes", collapsed=True,
                                  tip="Points and faces for recording FRF, time history, and response PSD",
                                  help="<b>Output probes</b><p>Points / faces where FRF, time history, and response PSD are recorded (result plots).</p>")
        f = QFormLayout(self.sec_probes.body)
        self.pr_list = QListWidget()
        self.pr_list.setMaximumHeight(90)
        set_placeholder(self.pr_list, "No probes yet – add a point or faces")
        help_tip(self.pr_list, "Output probe list for FRF, time history, and PSD plots",
                 "Output probes: FRF, time history, and PSD are saved at these locations (+ automatically at peak stress node).")
        self.pr_list.currentRowChanged.connect(self.on_probe_row)
        f.addRow(self.pr_list)
        hn = QHBoxLayout()
        self.pr_name = QLineEdit()
        self.pr_name.setToolTip("Probe name (empty = Probe n)")
        self.pr_q = QComboBox()
        for k, t in QTY:
            self.pr_q.addItem(t, k)
        self.pr_q.setToolTip("Monitored quantity of the probe: displacement, stress, acceleration")
        hn.addWidget(self.pr_name, 1)
        hn.addWidget(self.pr_q)
        f.addRow("Name", hn)
        hpt = QHBoxLayout()
        self.pr_pt = [dspin(0.0, -1e7, 1e7, 4, 1.0) for _ in range(3)]
        for s_, c in zip(self.pr_pt, "xyz"):
            s_.setToolTip(f"Coordinate {c} of the probe point [mm] (projected onto the mesh)")
            hpt.addWidget(s_)
        b = QPushButton("From selection")
        help_tip(b, "Get point coordinates from 3D selection",
                 "Point from 3D: vertex / surface point (Vertex filter V), edge or circle center (Edge filter E), otherwise center of gravity of selected faces.")
        b.clicked.connect(self.probe_point_from_selection)
        self.btn_pr_pt = b
        hpt.addWidget(b)
        f.addRow("Point [mm]", hpt)
        self.win.register_point_input(self.pr_pt)
        hb = QHBoxLayout()
        for txt, slot, tip, attr in (("Probe at point", self.add_probe_point, "Add a probe at the given point",
                                      "btn_pr_add"),
                                     ("Probe from faces", self.add_probe_faces, "Add a probe as the average of the nodes of the faces selected in 3D", "btn_pr_faces"),
                                     ("Delete", self.delete_probe, "Delete the selected probe", "btn_pr_del")):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            setattr(self, attr, b)
            hb.addWidget(b)
        f.addRow(hb)
        lay.addWidget(self.sec_probes)

        # ---------------------------------------------------------- added mass
        self.sec_masses = Section("Added mass", key="dyn.masses", collapsed=True,
                                  tip="Mass of unmodeled parts acting in static and dynamic analysis",
                                  help="<b>Added mass</b><p>A <i>point mass</i> (motor, purchased part) is connected to the selected faces at the centre of gravity (point) – by a rigid coupling (also carries rotary inertia, stiffens the face) or a distributing one (translational mass only, stiffness unchanged). A <i>surface mass</i> [kg/m²] is distributed over the face area (populated PCB, coating, insulation).</p><p>The mass acts in all analyses: gravity and rotation in statics, natural frequencies, dynamics. Check: the sum of the model mass in the dynamics results.</p>")
        f = QFormLayout(self.sec_masses.body)
        self.m_list = QListWidget()
        self.m_list.setMaximumHeight(90)
        set_placeholder(self.m_list, "No added masses yet – select faces in 3D")
        self.m_list.setToolTip("Added masses of the study. Click = faces in 3D and values into the form.")
        self.m_list.currentRowChanged.connect(self.on_mass_row)
        f.addRow(self.m_list)
        self.m_kind = QComboBox()
        self.m_kind.addItem("point mass (connected to faces)", "point_mass")
        self.m_kind.addItem("surface mass [kg/m²]", "surface_mass")
        help_tip(self.m_kind, "Added mass type: point mass at center of gravity or distributed area mass",
                 "Point: mass at a point coupled to selected faces; distributed: kg/m² spread across surface area.")
        self.m_kind.currentIndexChanged.connect(self.on_mass_kind)
        f.addRow("Type", self.m_kind)
        self.m_name = QLineEdit()
        self.m_name.setToolTip("Mass name (e.g. Motor, PCB) – empty = by the face selection")
        f.addRow("Name", self.m_name)
        self.m_mass = dspin(1.0, 0.0, 1e9, 4, 0.1, " kg")
        self.m_mass.setToolTip("Mass of the point mass [kg] (converted to t of the project)")
        f.addRow("Mass", self.m_mass)
        self.m_area = dspin(1.0, 0.0, 1e9, 4, 0.5, " kg/m²")
        self.m_area.setToolTip("Surface mass [kg/m²] (populated PCB roughly 1–5 kg/m² in addition to the board)")
        f.addRow("Surface mass", self.m_area)
        hpt = QHBoxLayout()
        self.m_auto = QCheckBox("face centroid")
        self.m_auto.setChecked(True)
        self.m_auto.setToolTip("Centre of gravity of the mass = centroid of the selected faces (otherwise the given point)")
        hpt.addWidget(self.m_auto)
        self.m_pt = [dspin(0.0, -1e7, 1e7, 4, 1.0) for _ in range(3)]
        for s_, c in zip(self.m_pt, "xyz"):
            s_.setToolTip(f"Coordinate {c} of the centre of gravity of the mass [mm]")
            hpt.addWidget(s_)
        self.m_auto.toggled.connect(lambda on: [s_.setEnabled(not on) for s_ in self.m_pt])
        for s_ in self.m_pt:
            s_.setEnabled(False)
        f.addRow("Centre of gravity [mm]", hpt)
        self.win.register_point_input(self.m_pt, self.m_auto)
        hi = QHBoxLayout()
        self.m_I = [dspin(0.0, 0.0, 1e9, 6, 0.001) for _ in range(3)]
        for s_, c in zip(self.m_I, ("Ixx", "Iyy", "Izz")):
            s_.setToolTip(f"Moment of inertia {c} about the centre of gravity [kg·m²] (0 = no rotary inertia)")
            hi.addWidget(s_)
        f.addRow("Inertia [kg·m²]", hi)
        self.m_coup = QComboBox()
        self.m_coup.addItem("rigid (also rotary inertia)", "rigid")
        self.m_coup.addItem("distributing (RBE3, does not add stiffness)", "distributing")
        help_tip(self.m_coup, "Coupling method of point mass to faces",
                 "Coupling to faces: rigid coupling transfers rotational inertia (stiffens surface), distributing transfers translational mass only.")
        f.addRow("Constraint", self.m_coup)
        hb = QHBoxLayout()
        for txt, slot, tip, attr in (("Add from selection", self.add_mass, "Add mass to the faces selected in 3D",
                                      "btn_m_add"),
                                     ("Save changes", self.update_mass, "Overwrite the values of the selected mass (faces are kept)",
                                      "btn_m_upd"),
                                     ("Faces from selection", self.mass_faces, "Replace the faces of the selected mass with the selection",
                                      "btn_m_faces"),
                                     ("Delete", self.delete_mass, "Delete the selected mass", "btn_m_del")):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            setattr(self, attr, b)
            hb.addWidget(b)
        f.addRow(hb)
        self.m_info = QLabel("")
        self.m_info.setWordWrap(True)
        f.addRow(self.m_info)
        lay.addWidget(self.sec_masses)

        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        lay.addWidget(self.info)
        lay.addStretch(1)
        # writing to the study
        for w in (self.on, self.verify, self.h_on, self.h_static, self.tr_on, self.tr_fat, self.r_on, self.s_on,
                  self.s_mm, self.s_tr, self.p_on, self.p_nlgeom):
            w.toggled.connect(lambda _: self.apply())
        for w in (self.d_kind, self.tr_method, self.r_dir, self.r_method, self.s_pulse, self.s_dir, self.s_comb,
                  self.p_case):
            w.currentIndexChanged.connect(lambda _: self.apply())
        for w in (self.modes, self.fmax, self.frames, self.d_zeta, self.d_eta, self.d_f1, self.d_f2, self.h_fmin,
                  self.h_fmax, self.h_points, self.h_bias, self.h_rate, self.h_sweeps, self.tr_tend, self.tr_dt,
                  self.tr_alpha, self.tr_settle, self.r_dur, self.s_A, self.s_tau, self.s_rise):
            self._commit_on_edit(w, self.apply)
        for t in (self.d_table, self.d_mat, self.s_srs):
            t.itemChanged.connect(lambda *_: self.apply())
        self.on_mass_kind()

    def scroll_to(self, w):
        """Scrolls the panel content so that the element w is visible (guide, test snapshots)."""
        scroll_to(self._scroll, w)

    @staticmethod
    def _commit_on_edit(spin, slot):
        spin.editingFinished.connect(slot)
        spin.valueChanged.connect(lambda *_: None if spin.hasFocus() else slot())

    @property
    def study(self):
        return self.win.study

    # ================================================================== study → form
    def refresh(self):
        st = self.study
        if st is None:
            return
        ds = st.dynamics
        self._filling = True
        try:
            for t in (self.d_table, self.d_mat, self.s_srs):
                t.blockSignals(True)
            self.on.setChecked(ds.enabled)
            self.modes.setValue(int(ds.modes))
            self.fmax.setValue(ds.f_max or 0.0)
            self.verify.setChecked(ds.verify_ccx)
            self.frames.setValue(int(ds.max_frames))
            d = ds.damping
            self.d_kind.setCurrentIndex(max(0, self.d_kind.findData(d.kind)))
            self.d_zeta.setValue(100 * float(d.zeta))
            self.d_eta.setValue(d.eta or 0.0)
            self.d_f1.setValue(d.f1 or 0.0)
            self.d_f2.setValue(d.f2 or 0.0)
            self.d_table.set_rows([[r[0], 100 * r[1]] for r in d.table])
            mats = sorted(st.materials)
            self.d_mat.set_rows([[m, 100 * d.per_material[m] if m in d.per_material else None] for m in mats])
            for r in range(self.d_mat.rowCount()):
                it = self.d_mat.item(r, 0)
                if it is not None:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            cases = [lc.name for lc in st.load_cases]
            h = ds.harmonic
            self.h_on.setChecked(h.enabled)
            self.h_fmin.setValue(h.f_min)
            self.h_fmax.setValue(h.f_max)
            self.h_points.setValue(int(h.points))
            self.h_bias.setValue(h.bias)
            self.h_static.setChecked(h.static_correction)
            self.h_rate.setValue(h.sweep_rate or 0.0)
            self.h_sweeps.setValue(max(int(h.sweeps), 1))
            self.h_exc.set_cases(cases)
            self.h_exc.set_items(h.excitation)
            t = ds.transient
            self.tr_on.setChecked(t.enabled)
            self.tr_tend.setValue(t.t_end)
            self.tr_dt.setValue(t.dt)
            self.tr_method.setCurrentIndex(max(0, self.tr_method.findData(t.method)))
            self.tr_alpha.setValue(t.alpha)
            self.tr_settle.setValue(t.settle_tol or 0.0)
            self.tr_fat.setChecked(t.fatigue)
            self.tr_exc.set_cases(cases)
            self.tr_exc.set_items(t.excitation)
            r = ds.random
            self.r_on.setChecked(r.enabled)
            self._asd = [list(map(float, x)) for x in r.asd]
            if [row[:2] for row in self.r_tab.values()] != self._asd:
                self.r_tab.set_rows([[f_, w_, None] for f_, w_ in self._asd])
            self.r_dir.setCurrentIndex(int(r.dof or 3) - 1)
            self.r_dur.setValue(r.duration_h)
            self.r_method.setCurrentIndex(max(0, self.r_method.findData(r.method)))
            s = ds.shock
            self.s_on.setChecked(s.enabled)
            self.s_pulse.setCurrentIndex(max(0, self.s_pulse.findData(s.pulse)))
            self.s_A.setValue(s.A)
            self.s_tau.setValue(1000 * s.tau)
            self.s_rise.setValue(1000 * (s.rise or 0.0))
            self.s_dir.setCurrentIndex(int(s.dof or 3) - 1)
            self.s_comb.setCurrentIndex(max(0, self.s_comb.findData(s.combine)))
            self.s_mm.setChecked(s.missing_mass)
            self.s_tr.setChecked(s.transient)
            self.s_srs.set_rows(s.srs_table)
            p = ds.prestress
            self.p_on.setChecked(p.enabled)
            self.p_nlgeom.setChecked(getattr(p, "nlgeom", False))
            self.p_case.clear()
            self.p_case.addItems(cases)
            if p.case in cases:
                self.p_case.setCurrentText(p.case)
            self.fill_probes()
            self.fill_masses()
        finally:
            for t in (self.d_table, self.d_mat, self.s_srs):
                t.blockSignals(False)
            self._filling = False
        if st is not self._last_study:
            self._last_study = st
            if st.masses:
                self.sec_masses.set_open(True)
        self.update_enabled()
        self.update_asd_info()
        self.update_info()

    def update_enabled(self):
        k = self.d_kind.currentData()
        self.d_eta.setEnabled(k == "structural")
        self.d_f1.setEnabled(k == "rayleigh")
        self.d_f2.setEnabled(k == "rayleigh")
        self.d_table.setEnabled(k == "modal")
        self.d_mat.setEnabled(k == "per_material")
        # Rule 3: setRowVisible instead of setEnabled
        self.f_damping.setRowVisible(self.d_eta, k == "structural")
        self.f_damping.setRowVisible(self.hf_widget, k == "rayleigh")
        self.f_damping.setRowVisible(self.d_table, k == "modal")
        self.f_damping.setRowVisible(self.d_mat, k == "per_material")
        self.tr_alpha.setEnabled(self.tr_method.currentData() == "direct")
        srs = self.s_pulse.currentData() == "srs_table"
        self.s_srs.setEnabled(srs)
        for w in (self.s_A, self.s_tau, self.s_tr):
            w.setEnabled(not srs)
        self.s_rise.setEnabled(self.s_pulse.currentData() == "trapezoid")
        self.p_case.setEnabled(self.p_on.isChecked())
        self.p_nlgeom.setEnabled(self.p_on.isChecked())

    def update_info(self):
        st = self.study
        if st is None:
            return
        from fea_core.dynamics.check import check
        from fea_core.dynamics.mass import check as check_mass
        try:
            e1, w1 = check(st, [p.name for p in self.win.parts])
            e2, w2 = check_mass(st, [p.name for p in self.win.parts])
        except Exception as e:  # noqa: BLE001
            e1, w1, e2, w2 = [f"The dynamics check failed: {e}"], [], [], []
        rows = [f"<span style='color:#d23b3b'>● {escape(x)}</span>" for x in e1 + e2]
        rows += [f"<span style='color:#b36b00'>● {escape(x)}</span>" for x in w1 + w2 if "uncertain input" not in x]
        if st.dynamics.enabled and not (e1 + e2):
            rows.append("<span style='color:#2e9e4f'>● Dynamics is defined without errors.</span>")
        self.info.setText("<br>".join(rows))

    def color(self, errors, warnings) -> str:
        from .main import GREEN, GREY, ORANGE, RED
        st = self.study
        if st is None or not (st.dynamics.enabled or st.masses):
            return GREY
        pre = ("Dynamics", "Harmonic", "Transient", "Random vibration", "Shock", "Added mass", "Base excitation",
               "Prestressed modal", "Campbell")
        if any(e.startswith(pre) for e in errors):
            return RED
        return ORANGE if any(w.startswith(pre) and "uncertain input" not in w for w in warnings) else GREEN

    # ================================================================== form → study
    def apply(self):
        st = self.study
        if self._filling or st is None:
            return
        self.update_enabled()
        ds = st.dynamics
        before = ds.to_dict()
        ds.enabled = self.on.isChecked()
        ds.modes = self.modes.value()
        ds.f_max = _f(self.fmax.value())
        ds.verify_ccx = self.verify.isChecked()
        ds.max_frames = self.frames.value()
        d = ds.damping
        d.kind = self.d_kind.currentData()
        d.zeta = self.d_zeta.value() / 100.0
        d.eta = _f(self.d_eta.value())
        d.f1, d.f2 = _f(self.d_f1.value()), _f(self.d_f2.value())
        try:
            d.table = [[float(a), float(b) / 100.0] for a, b in self.d_table.values() if a is not None and b is not None]
            d.per_material = {m: float(z) / 100.0 for m, z in self.d_mat.values([str, float]) if z is not None}
        except (TypeError, ValueError):
            pass
        h = ds.harmonic
        h.enabled = self.h_on.isChecked()
        h.f_min, h.f_max = self.h_fmin.value(), self.h_fmax.value()
        h.points, h.bias = self.h_points.value(), self.h_bias.value()
        h.static_correction = self.h_static.isChecked()
        h.sweep_rate = self.h_rate.value()
        h.sweeps = self.h_sweeps.value()
        h.excitation = list(self.h_exc.items)
        t = ds.transient
        t.enabled = self.tr_on.isChecked()
        t.t_end, t.dt = self.tr_tend.value(), self.tr_dt.value()
        t.method = self.tr_method.currentData()
        t.alpha = self.tr_alpha.value()
        t.settle_tol = _f(self.tr_settle.value())
        t.fatigue = self.tr_fat.isChecked()
        t.excitation = list(self.tr_exc.items)
        r = ds.random
        r.enabled = self.r_on.isChecked()
        r.asd = [list(x) for x in self._asd]
        r.dof = self.r_dir.currentIndex() + 1
        r.direction = None
        r.duration_h = self.r_dur.value()
        r.method = self.r_method.currentData()
        s = ds.shock
        s.enabled = self.s_on.isChecked()
        s.pulse = self.s_pulse.currentData()
        s.A = self.s_A.value()
        s.tau = self.s_tau.value() / 1000.0
        s.rise = _f(self.s_rise.value() / 1000.0)
        s.dof = self.s_dir.currentIndex() + 1
        s.direction = None
        s.combine = self.s_comb.currentData()
        s.missing_mass = self.s_mm.isChecked()
        s.transient = self.s_tr.isChecked()
        try:
            s.srs_table = [[float(a), float(b)] for a, b in self.s_srs.values() if a is not None and b is not None]
        except (TypeError, ValueError):
            pass
        p = ds.prestress
        p.enabled = self.p_on.isChecked()
        p.nlgeom = self.p_nlgeom.isChecked()
        p.case = self.p_case.currentText() or None
        if ds.to_dict() != before:
            self.win.changed()

    def excitation_changed(self, editor):
        self.apply()
        editor.fill()

    def set_zeta(self, zeta: float):
        """ζ from the measurement calculator → modal damping."""
        self.d_kind.setCurrentIndex(self.d_kind.findData("modal"))
        self.d_zeta.setValue(100.0 * zeta)
        self.apply()
        self.win.log_msg(f"Damping from measurement: ζ = {fmt(100 * zeta, 4)} %")

    def open_calc(self):
        if self._calc is None:
            self._calc = DampingCalcDialog(self)
        self._calc.show()
        self._calc.raise_()

    # ------------------------------------------------------------------ ASD
    def apply_asd(self):
        from fea_core.dynamics.psd import asd_from_slopes
        try:
            rows = self.r_tab.values()
            pts = [[r[0], r[1]] for r in rows]
            if any(p[0] is None for p in pts):
                raise ValueError("Each profile point needs a frequency")
            slopes = [rows[i][2] if len(rows[i]) > 2 else None for i in range(len(rows) - 1)]
            g = self.r_grms_target.value()
            tab = asd_from_slopes(pts, slopes, g if g > 0 else None)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, APP, f"The ASD profile cannot be built: {e}\n\nEnter points [f, W] from the standard / specification; a missing W is completed from the segment slope in dB/oct.")
            return
        self._asd = [[float(a), float(b)] for a, b in tab]
        self.r_tab.set_rows([[a, b, None] for a, b in self._asd])
        self.apply()
        self.update_asd_info()

    def update_asd_info(self):
        if not self._asd or len(self._asd) < 2:
            self.r_info.setText("The ASD profile is not defined yet – points [f, W] from the standard / specification, then “Compute and apply”.")
            return
        from fea_core.dynamics.psd import asd_rms
        try:
            g = asd_rms(self._asd)
        except Exception:  # noqa: BLE001
            g = float("nan")
        a = np.asarray(self._asd, float)
        sl = [10 * math.log10(a[i + 1, 1] / a[i, 1]) / math.log2(a[i + 1, 0] / a[i, 0]) for i in range(len(a) - 1)
              if a[i, 1] > 0 and a[i + 1, 1] > 0]
        self.r_info.setText(f"g_rms = <b>{fmt(g, 4)} g</b>, {len(a)} points {fmt(a[0, 0])}–{fmt(a[-1, 0])} Hz; slopes "
                            + ", ".join(f"{fmt(s, 3)}" for s in sl) + " dB/oct")
        if self.isVisible():
            self.plot_asd()

    def plot_asd(self):
        if not self._asd or len(self._asd) < 2:
            return
        ax = self.r_plot.axes()
        a = np.asarray(self._asd, float)
        ax.loglog(a[:, 0], a[:, 1], "o-", color="#1f77b4")
        ax.set_xlabel("f [Hz]")
        ax.set_ylabel("ASD [g²/Hz]")
        ax.grid(True, which="both", alpha=0.3)
        self.r_plot.canvas.draw_idle()

    def showEvent(self, e):
        super().showEvent(e)
        if self._asd and len(self._asd) >= 2:
            self.plot_asd()

    # ------------------------------------------------------------------ profily
    def on_profile_combo(self, *_):
        k = self.profile.currentIndex()
        self.profile_note.setText(f"<small>{PROFILES[k][2]}</small>" if 0 <= k < len(PROFILES) else "")

    def apply_profile(self):
        st = self.study
        key = self.profile.currentData()
        if st is None or not key:
            return
        ds = st.dynamics
        ds.enabled = True
        if key == "60068-2-6":
            ds.harmonic.enabled = True
            if not ds.harmonic.excitation:
                ds.harmonic.excitation = [Excitation(kind="base_acc", dof=3, unit="g", scale=0.0)]
        elif key == "60068-2-64":
            ds.random.enabled = True
        elif key == "60068-2-27":
            ds.shock.enabled = True
            if ds.shock.pulse == "srs_table":
                ds.shock.pulse = "halfsine"
            ds.shock.A, ds.shock.tau = 0.0, 0.0
        self.win.changed()
        self.win.log_msg(f"Profile {self.profile.currentText()}: structure prepared – enter the values from the standard / test specification (the program does not contain them).")

    # ================================================================== sondy
    def fill_probes(self):
        self.pr_list.blockSignals(True)
        self.pr_list.clear()
        for p in self.study.dynamics.probes:
            where = ("(" + "; ".join(fmt(float(x)) for x in p.point) + ") mm") if p.point is not None else \
                f"selection {p.selection}"
            self.pr_list.addItem(f"{p.name}: {dict(QTY).get(p.quantity, p.quantity)}, {where}")
        self.pr_list.blockSignals(False)
        n = len(self.study.dynamics.probes)
        self.sec_probes.set_badge("" if n == 0 else ("1 probe" if n == 1 else (f"{n} probes" if n < 5 else f"{n} probes")))

    def _probe_name(self) -> str:
        names = {p.name for p in self.study.dynamics.probes}
        n = self.pr_name.text().strip()
        if n and n not in names:
            return n
        k = 1
        while f"Probe {k}" in names:
            k += 1
        return f"Probe {k}"

    def probe_point_from_selection(self):
        v = self.win.view
        p = None
        pts = v.selected_points()
        e = v.selected_edge()
        if pts:
            p = pts[-1][1]
        elif e is not None:
            p = e["point"]
        elif v.selected_edges():
            p = self.win.edge_prop(*v.selected_edges()[-1])["point"]
        elif v.selected_faces():
            sels = self.win.selected_selectors()
            w = np.array([s.area for s in sels])
            p = (np.array([s.centroid for s in sels]) * w[:, None]).sum(axis=0) / max(w.sum(), 1e-30)
        if p is None:
            QMessageBox.information(self, APP, "Select a point (Point filter V), an edge (Edge filter E) or faces in 3D.")
            return
        for s_, x in zip(self.pr_pt, p):
            s_.setValue(float(x))

    def add_probe_point(self):
        st = self.study
        if st is None:
            return
        st.dynamics.probes.append(Probe(self._probe_name(), point=[s_.value() for s_ in self.pr_pt],
                                        quantity=self.pr_q.currentData()))
        self.win.changed()

    def add_probe_faces(self):
        st = self.study
        if st is None:
            return
        sels = self.win.need_faces()
        if not sels:
            return
        name = self.win.new_selection("Probe", sels)
        self.win.view.set_selected_faces([])
        st.dynamics.probes.append(Probe(self._probe_name(), selection=name, quantity=self.pr_q.currentData()))
        self.win.changed()

    def delete_probe(self):
        st = self.study
        r = self.pr_list.currentRow()
        if st is None or not (0 <= r < len(st.dynamics.probes)):
            return
        p = st.dynamics.probes[r]
        if p.selection:
            self.win._drop_selection(p.selection)          # before removal (this probe is counted too)
        st.dynamics.probes.pop(r)
        self.win.changed()

    def on_probe_row(self, r):
        st = self.study
        if st is None or not (0 <= r < len(st.dynamics.probes)):
            return
        p = st.dynamics.probes[r]
        self.pr_name.setText(p.name)
        self.pr_q.setCurrentIndex(max(0, self.pr_q.findData(p.quantity)))
        if p.point is not None:
            for s_, x in zip(self.pr_pt, p.point):
                s_.setValue(float(x))
        if p.selection:
            self.win.view.set_selected_faces(self.win.sel_faces.get(p.selection, []))

    # ================================================================== added mass
    def fill_masses(self):
        self.m_list.blockSignals(True)
        cur = self.m_list.currentRow()
        self.m_list.clear()
        tot = 0.0
        for m in self.study.masses:
            if isinstance(m, PointMass):
                tot += 1000 * m.mass
                self.m_list.addItem(f"{m.name or m.selection}: {fmt(1000 * m.mass)} kg "
                                    f"({'rigid' if m.coupling == 'rigid' else 'distributing'} constraint"
                                    + (", inertia" if m.inertia else "") + ")")
            else:
                self.m_list.addItem(f"{m.name or m.selection}: {fmt(m.mass_per_area)} kg/m²")
        if 0 <= cur < self.m_list.count():
            self.m_list.setCurrentRow(cur)
        self.m_list.blockSignals(False)
        n_s = sum(isinstance(m, SurfaceMass) for m in self.study.masses)
        self.m_info.setText((f"Point masses total {fmt(tot)} kg" if tot else "No point masses")
                            + (f", surface masses: {n_s}" if n_s else "")
                            + ". It also acts in statics (weight, rotation) and in natural frequencies.")
        n = len(self.study.masses)
        self.sec_masses.set_badge("" if n == 0 else ("1 mass" if n == 1 else (f"{n} masses" if n < 5 else f"{n} masses")))

    def on_mass_kind(self, *_):
        pt = self.m_kind.currentData() == "point_mass"
        for w in [self.m_mass, self.m_auto, self.m_coup] + self.m_I:
            w.setEnabled(pt)
        for s_ in self.m_pt:
            s_.setEnabled(pt and not self.m_auto.isChecked())
        self.m_area.setEnabled(not pt)

    def mass_from_form(self, selection: str):
        name = self.m_name.text().strip()
        if self.m_kind.currentData() == "surface_mass":
            if self.m_area.value() <= 0:
                QMessageBox.warning(self, APP, "The surface mass must be positive.")
                return None
            return SurfaceMass(selection, self.m_area.value(), name=name)
        if self.m_mass.value() <= 0:
            QMessageBox.warning(self, APP, "The mass must be positive.")
            return None
        I = [1e3 * s_.value() for s_ in self.m_I]           # kg·m² → t·mm²
        return PointMass(selection, 1e-3 * self.m_mass.value(),
                         point=None if self.m_auto.isChecked() else [s_.value() for s_ in self.m_pt],
                         inertia=I if any(x > 0 for x in I) else None, coupling=self.m_coup.currentData(), name=name)

    def add_mass(self):
        st = self.study
        if st is None:
            return
        sels = self.win.need_faces()
        if not sels:
            return
        m = self.mass_from_form("")
        if m is None:
            return
        m.selection = self.win.new_selection("Mass", sels)
        self.win.view.set_selected_faces([])
        st.masses.append(m)
        self.sec_masses.set_open(True)
        self.win.changed()
        self.m_list.setCurrentRow(len(st.masses) - 1)

    def update_mass(self):
        st = self.study
        r = self.m_list.currentRow()
        if st is None or not (0 <= r < len(st.masses)):
            QMessageBox.warning(self, APP, "Select a mass in the list.")
            return
        m = self.mass_from_form(st.masses[r].selection)
        if m is not None:
            st.masses[r] = m
            self.win.changed()
            self.m_list.setCurrentRow(r)

    def mass_faces(self):
        st = self.study
        r = self.m_list.currentRow()
        if st is None or not (0 <= r < len(st.masses)):
            QMessageBox.warning(self, APP, "Select a mass in the list.")
            return
        sels = self.win.need_faces()
        if not sels:
            return
        name = st.masses[r].selection
        st.selections[name].faces = sels
        self.win.sel_faces[name] = self.win.view.selected_faces()
        self.win.changed(invalidate_mesh=True)

    def delete_mass(self):
        st = self.study
        r = self.m_list.currentRow()
        if st is None or not (0 <= r < len(st.masses)):
            return
        self.win._drop_selection(st.masses[r].selection)    # before removal (this mass is counted too)
        st.masses.pop(r)
        self.win.changed()

    def on_mass_row(self, r):
        st = self.study
        if st is None or not (0 <= r < len(st.masses)):
            return
        m = st.masses[r]
        self.m_name.setText(m.name)
        self.m_kind.setCurrentIndex(self.m_kind.findData(m.kind))
        if isinstance(m, PointMass):
            self.m_mass.setValue(1000 * m.mass)
            self.m_auto.setChecked(m.point is None)
            if m.point is not None:
                for s_, x in zip(self.m_pt, m.point):
                    s_.setValue(float(x))
            I = m.inertia or [0, 0, 0]
            I = np.diag(np.asarray(I, float)) if np.asarray(I).ndim == 2 else np.asarray(I, float)
            for s_, x in zip(self.m_I, I):
                s_.setValue(float(x) / 1e3)
            self.m_coup.setCurrentIndex(max(0, self.m_coup.findData(m.coupling)))
        else:
            self.m_area.setValue(m.mass_per_area)
        self.on_mass_kind()
        self.win.view.set_selected_faces(self.win.sel_faces.get(m.selection, []))

    # ================================================================== 3D
    def show_3d(self):
        st = self.study
        win = self.win
        v = win.view
        L = 0.04 * v.model_size()
        for k, m in enumerate(st.masses):
            faces = win.sel_faces.get(m.selection, [])
            lbl = m.name or m.selection
            v.set_face_set(f"m:{k}", faces, C_MASS, label=lbl)
            if isinstance(m, PointMass):
                if m.point is not None:
                    p = np.asarray(m.point, float)
                elif faces:
                    props = [win.face_prop(pi, fi) for pi, fi in faces]
                    w = np.array([q["area"] for q in props])
                    p = (np.array([q["centroid"] for q in props]) * w[:, None]).sum(axis=0) / max(w.sum(), 1e-30)
                else:
                    continue
                seg = [(p - L * e, p + L * e) for e in np.eye(3)]
                for pi, fi in faces[:8]:
                    seg.append((p, np.asarray(win.arrow_point(win.face_prop(pi, fi)), float)))
                v.add_lines(f"m:{k}", seg, C_MASS, 2.0, label=f"{lbl} {fmt(1000 * m.mass)} kg")
        for k, p in enumerate(st.dynamics.probes):
            if p.point is not None:
                q = np.asarray(p.point, float)
                v.add_lines(f"pr:{k}", [(q - L * e, q + L * e) for e in np.eye(3)], C_PROBE, 2.5, label=p.name)
            elif p.selection:
                v.set_face_set(f"pr:{k}", win.sel_faces.get(p.selection, []), C_PROBE, label=p.name)
