"""Plasticity of a metal in the Material properties window: switch, model (bilinear / multilinear table / Ramberg-Osgood /
Hollomon), hardening (isotropic / kinematic), EN 1993 strain limit, check by `fea_core.plasticity.validate_material` and the
curve the solver gets (`plasticity_eval.stress_strain`). The values go to `Material.plastic` = `PlasticitySpec.to_dict()`
(the only storage – the same dict MCP `custom_props["plastic"]` fills and `ccx/inp.py` writes as *PLASTIC).
"""
from __future__ import annotations

import html
import re

from PySide6.QtWidgets import (QApplication, QComboBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QCheckBox)

from fea_core.i18n import tr
from fea_core.plasticity import PlasticitySpec, validate_material

from .widgets import LazyPlot, Section, dspin, fmt, help_tip

MODELS = (("bilinear", "Bilinear (Re + tangent modulus)"), ("multilinear", "Multilinear (table σ – ε_pl)"),
          ("ramberg_osgood", "Ramberg-Osgood (K, n)"), ("hollomon", "Hollomon (from Re, Rm, A)"))
HARDENING = (("isotropic", "Isotropic"), ("kinematic", "Kinematic (Bauschinger, needs NLGEOM)"))
HELP = ("<b>Plasticity (elastoplastic metal)</b><p>Tick to compute the material beyond the yield point: von Mises yield "
        "condition, hardening by the chosen curve. The study then runs as a <b>nonlinear</b> CalculiX analysis (no "
        "superposition of load cases, no NGSolve); results get the equivalent plastic strain PEEQ.</p>"
        "<ul><li><b>Bilinear</b>: yield at Re (field above), then the tangent modulus E<sub>t</sub> of the tensile curve "
        "(dσ/dε) or the plastic modulus H = dσ/dε<sub>pl</sub> = E·E<sub>t</sub>/(E − E<sub>t</sub>).</li>"
        "<li><b>Multilinear</b>: table of TRUE stress and PLASTIC strain, the first row = yield point with ε<sub>pl</sub> = 0. "
        "Paste from Excel / CSV; an engineering tensile curve (σ, total ε) can be converted (valid up to necking).</li>"
        "<li><b>Ramberg-Osgood</b>: σ = K·(ε<sub>pl</sub> + (Re/K)<sup>1/n</sup>)<sup>n</sup> from Re.</li>"
        "<li><b>Hollomon</b>: power curve through Re and Rm at the uniform elongation ≈ A/2 (estimate).</li></ul>"
        "<p>Beyond the last point of the curve the stress stays constant (perfectly plastic). Kinematic hardening is "
        "verified for the bilinear curve and switches large deformations (NLGEOM) on. Residual stress after unloading needs a "
        "load history: <i>Add unloading step</i> (sequence in the Contacts step).</p>")


def parse_table(text: str) -> list[tuple[float, float]]:
    """Two numeric columns from pasted text (Excel tab, CSV with ';' or ',', spaces). Rows that are not two numbers
    (headings) are skipped. With ';' or tab the comma is the decimal point; with only commas one comma = column separator.
    """
    rows = []
    for line in text.replace("\r", "").split("\n"):
        line = line.strip()
        if not line:
            continue
        if "\t" in line or ";" in line:
            cells = [c.replace(",", ".") for c in re.split(r"[\t;]", line)]
        elif "," in line and "." in line:
            cells = line.split(",")
        elif line.count(",") == 1:
            cells = line.split(",")
        elif "," in line:
            raise ValueError(tr("Ambiguous row (commas are both decimal points and separators): {0}").format(line))
        else:
            cells = line.split()
        cells = [c.strip() for c in cells if c.strip()]
        if len(cells) < 2:
            continue
        try:
            rows.append((float(cells[0]), float(cells[1])))
        except ValueError:
            continue                                        # heading row
    if not rows:
        raise ValueError(tr("No rows of two numbers found"))
    return rows


class PlasticityBox(Section):
    """Collapsible section with a tick in the header (as OrthoBox): tick = the material is elastoplastic."""

    def __init__(self, win):
        self.win = win
        self.check = QCheckBox()
        self.check.setToolTip("Tick to make the material elastoplastic (nonlinear analysis, PEEQ)")
        inner = QWidget()
        super().__init__("Plasticity (elastoplastic metal)", inner, check=self.check,
                         tip="Tick to make the material elastoplastic (nonlinear analysis, PEEQ)", help=HELP)
        self.inner = inner
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 0, 0, 0)
        f = QFormLayout()
        self.model = QComboBox()
        for k, t in MODELS:
            self.model.addItem(t, k)
        help_tip(self.model, "Hardening curve after the yield point",
                 "Bilinear: Re + tangent modulus. Multilinear: table of true stress – plastic strain. Ramberg-Osgood: K, n. "
                 "Hollomon: estimate from Re, Rm and the elongation A.")
        f.addRow("Model", self.model)
        self.hard = QComboBox()
        for k, t in HARDENING:
            self.hard.addItem(t, k)
        help_tip(self.hard, "Isotropic (monotonic loading) or kinematic (reversed / cyclic loading)",
                 "Isotropic: the yield surface grows equally in all directions – monotonic loading. Kinematic: the surface "
                 "moves (Bauschinger effect) – load reversal, cycles. Kinematic is computed correctly by CalculiX only with "
                 "large deformations (NLGEOM) – it is switched on automatically.")
        f.addRow("Hardening", self.hard)
        v.addLayout(f)

        self.pages = QStackedWidget()
        # bilinear
        pb = QWidget()
        fb = QFormLayout(pb)
        fb.setContentsMargins(0, 0, 0, 0)
        self.bi_kind = QComboBox()
        self.bi_kind.addItem("tangent modulus Et (dσ/dε of the tensile curve)", "Et")
        self.bi_kind.addItem("plastic modulus H (dσ/dε_pl)", "H")
        help_tip(self.bi_kind, "Enter the slope as Et (tensile curve) or H (plastic strain)",
                 "E_t = slope of the tensile curve above Re (dσ/dε, total strain); H = dσ/dε_pl = E·E_t/(E − E_t). "
                 "Stored as E_t. E_t = 0 = perfectly plastic.")
        fb.addRow("Slope given as", self.bi_kind)
        self.bi_val = dspin(2100.0, 0.0, 1e7, 1, 100.0, " MPa")
        help_tip(self.bi_val, "Hardening slope above the yield point Re (0 = perfectly plastic)")
        fb.addRow("Slope", self.bi_val)
        self.bi_note = QLabel()
        self.bi_note.setProperty("muted", True)
        self.bi_note.setWordWrap(True)
        fb.addRow(self.bi_note)
        self.pages.addWidget(pb)
        # multilinear
        pm = QWidget()
        vm = QVBoxLayout(pm)
        vm.setContentsMargins(0, 0, 0, 0)
        self.tab = QTableWidget(0, 2)
        self.tab.setHorizontalHeaderLabels(["σ true [MPa]", "ε_pl [–]"])
        self.tab.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tab.verticalHeader().setVisible(False)
        self.tab.setMinimumHeight(130)
        help_tip(self.tab, "Hardening table: true stress and plastic strain, 1st row = yield (ε_pl 0)",
                 "Rows of true stress [MPa] and equivalent plastic strain [–]; the first row is the yield point with "
                 "ε_pl = 0, plastic strains increase, stresses do not decrease. Ctrl+V pastes from Excel / CSV.")
        vm.addWidget(self.tab)
        h = QHBoxLayout()
        for text, tip, slot in (("+ Row", "Add a row below the last one", self.add_row),
                                ("− Row", "Remove the selected rows", self.remove_rows),
                                ("Paste", "Replace the table with two columns from the clipboard (Excel, CSV)", self.paste),
                                ("Engineering → true", "Convert the table from an engineering tensile curve (σ, total ε)",
                                 self.convert_engineering)):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            h.addWidget(b)
        h.addStretch(1)
        vm.addLayout(h)
        self.tab_msg = QLabel()
        self.tab_msg.setWordWrap(True)
        vm.addWidget(self.tab_msg)
        self.pages.addWidget(pm)
        # Ramberg-Osgood
        pr = QWidget()
        fr = QFormLayout(pr)
        fr.setContentsMargins(0, 0, 0, 0)
        self.ro_K = dspin(700.0, 0.0, 1e6, 1, 10.0, " MPa")
        help_tip(self.ro_K, "Ramberg-Osgood strength coefficient K (> Re)")
        self.ro_n = dspin(0.2, 0.0, 2.0, 4, 0.01)
        help_tip(self.ro_n, "Ramberg-Osgood hardening exponent n (steel ~0.1–0.25)")
        fr.addRow("K", self.ro_K)
        fr.addRow("n", self.ro_n)
        self.pages.addWidget(pr)
        # Hollomon
        ph = QWidget()
        fh = QFormLayout(ph)
        fh.setContentsMargins(0, 0, 0, 0)
        self.ho_Rm = dspin(400.0, 0.0, 1e5, 0, 10.0, " MPa")
        help_tip(self.ho_Rm, "Tensile strength Rm for the Hollomon curve (> Re)")
        self.ho_A = dspin(20.0, 0.0, 100.0, 1, 1.0, " %")
        help_tip(self.ho_A, "Elongation at fracture A (uniform elongation estimated as A/2)")
        fh.addRow("Rm", self.ho_Rm)
        fh.addRow("A", self.ho_A)
        self.pages.addWidget(ph)
        v.addWidget(self.pages)

        f2 = QFormLayout()
        self.limit = dspin(5.0, 0.1, 99.0, 2, 0.5, " %")
        help_tip(self.limit, "EN 1993-1-5 annex C limit of the plastic strain (default 5 %)",
                 "Limit of the equivalent plastic strain for the EN 1993-1-5 annex C check in Results (5 % for structural "
                 "steel); the ASME VIII-2 check has its own triaxial limit.")
        f2.addRow("Strain limit", self.limit)
        v.addLayout(f2)
        self.status = QLabel()
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        self.plot = LazyPlot()
        help_tip(self.plot, "Tensile curve the solver gets (true stress – total strain)")
        v.addWidget(self.plot)
        h2 = QHBoxLayout()
        self.btn_unload = QPushButton("Add unloading step")
        help_tip(self.btn_unload, "Load history: cases in sequence + a final zero-load step (residual stress)",
                 "Creates the load history (Contacts step → Nonlinear analysis – steps): the load cases one after another "
                 "and a final step with scale 0. The stress of that step = residual stress after unloading.")
        self.btn_unload.clicked.connect(self.add_unloading)
        h2.addWidget(self.btn_unload)
        h2.addStretch(1)
        v.addLayout(h2)

        self._filling = False
        self.model.currentIndexChanged.connect(self._on_model)
        for w in (self.hard, self.bi_kind):
            w.currentIndexChanged.connect(self._on_edit)
        self.bi_kind.currentIndexChanged.connect(self._on_bi_kind)
        for w in (self.bi_val, self.ro_K, self.ro_n, self.ho_Rm, self.ho_A, self.limit):
            w.valueChanged.connect(self._on_edit)
        self.tab.itemChanged.connect(self._on_edit)
        self.check.toggled.connect(self._on_toggle)
        for w in ("m_E", "m_Re", "m_Rm"):
            sp = getattr(win, w, None)
            if sp is not None:
                sp.valueChanged.connect(self._on_edit)
        self._on_model()

    # ------------------------------------------------------------------ state
    def isChecked(self) -> bool:
        return self.check.isChecked()

    def setChecked(self, on: bool) -> None:
        self.check.setChecked(bool(on))

    def _E(self) -> float:
        return float(self.win.m_E.value())

    def _on_toggle(self, on: bool):
        self.inner.setEnabled(True)
        self.refresh()

    def _on_model(self, *_):
        self.pages.setCurrentIndex(max(self.model.currentIndex(), 0))
        if self.model.currentData() == "multilinear" and self.tab.rowCount() == 0 and not self._filling:
            self._fill_table_from_curve()
        self._on_edit()

    def _on_bi_kind(self, *_):
        """Keep the same physical slope when switching Et ↔ H."""
        if self._filling:
            return
        E, x = self._E(), self.bi_val.value()
        self._filling = True
        if self.bi_kind.currentData() == "H":            # was Et
            self.bi_val.setValue(E * x / (E - x) if x < E else 0.0)
        else:
            self.bi_val.setValue(E * x / (E + x))
        self._filling = False
        self._on_edit()

    def _on_edit(self, *_):
        if not self._filling:
            self.refresh()

    def _et(self) -> float:
        x = self.bi_val.value()
        if self.bi_kind.currentData() == "H":
            E = self._E()
            return E * x / (E + x)
        return x

    def table_rows(self) -> list[tuple[float, float]]:
        rows = []
        for r in range(self.tab.rowCount()):
            cells = [self.tab.item(r, c) for c in (0, 1)]
            txt = [(c.text() if c is not None else "").strip().replace(",", ".") for c in cells]
            if not any(txt):
                continue
            try:
                rows.append((float(txt[0]), float(txt[1])))
            except ValueError:
                raise ValueError(tr("Row {0} of the table is not two numbers").format(r + 1)) from None
        return rows

    def set_table(self, rows) -> None:
        self.tab.blockSignals(True)
        self.tab.setRowCount(0)
        for s, ep in rows:
            r = self.tab.rowCount()
            self.tab.insertRow(r)
            self.tab.setItem(r, 0, QTableWidgetItem(f"{s:.6g}"))
            self.tab.setItem(r, 1, QTableWidgetItem(f"{ep:.6g}"))
        self.tab.blockSignals(False)

    # ------------------------------------------------------------------ material ↔ form
    def values(self) -> dict | None:
        """Material.plastic (PlasticitySpec.to_dict) or None (elastic). ValueError for an unreadable table."""
        if not self.isChecked():
            return None
        model = self.model.currentData()
        spec = PlasticitySpec(model=model, hardening=self.hard.currentData(), limit_strain=self.limit.value() / 100.0)
        if model == "bilinear":
            spec.Et = float(self._et())
        elif model == "multilinear":
            spec.table = self.table_rows()
        elif model == "ramberg_osgood":
            spec.K_prime, spec.n_prime = float(self.ro_K.value()), float(self.ro_n.value())
        elif model == "hollomon":
            spec.Rm, spec.A = float(self.ho_Rm.value()), float(self.ho_A.value()) / 100.0
        return spec.to_dict()

    def load(self, m) -> None:
        """Fills the form from a material (elastic → switched off, defaults from Re / Rm of the material)."""
        p = getattr(m, "plastic", None) if isinstance(getattr(m, "plastic", None), dict) else None
        self._filling = True
        try:
            self.check.blockSignals(True)
            self.check.setChecked(bool(p))
            self.check.blockSignals(False)
            self.set_open(bool(p))
            spec = PlasticitySpec.from_dict(p or {})
            self._set_combo(self.model, "multilinear" if (p and spec.table and spec.model == "bilinear") else spec.model)
            self._set_combo(self.hard, spec.hardening.lower())
            self.bi_kind.setCurrentIndex(0)
            self.bi_val.setValue(spec.Et if spec.Et is not None else float(m.E) / 100.0)
            self.set_table(spec.table)
            self.ro_K.setValue(spec.K_prime or 700.0)
            self.ro_n.setValue(spec.n_prime or 0.2)
            rm = spec.Rm or m.Rm or ((m.Re or 0.0) * 1.25)
            self.ho_Rm.setValue(rm or 400.0)
            a = spec.A if spec.A is not None else 0.20
            self.ho_A.setValue(a if a > 1.0 else a * 100.0)
            self.limit.setValue((spec.limit_strain or 0.05) * 100.0)
            self.pages.setCurrentIndex(max(self.model.currentIndex(), 0))
        finally:
            self._filling = False
        self.refresh()

    @staticmethod
    def _set_combo(cb, key):
        i = cb.findData(key)
        cb.setCurrentIndex(i if i >= 0 else 0)

    def _trial_material(self):
        """Material from the main form + this box (for the check and the curve)."""
        from fea_core.study import Material
        w = self.win
        return Material(w.m_name.text().strip() or "material", self._E(), float(w.m_nu.value()),
                        Re=w.m_Re.value() or None, Rm=w.m_Rm.value() or None, plastic=self.values())

    def describe(self, m) -> str:
        p = getattr(m, "plastic", None)
        if not isinstance(p, dict) or not p:
            return ""
        s = PlasticitySpec.from_dict(p)
        model = dict(MODELS).get(s.model, s.model)
        hard = dict(HARDENING).get(s.hardening.lower(), s.hardening)
        extra = ""
        if s.model == "bilinear" and s.Et is not None:
            extra = f", Et {fmt(s.Et)} MPa"
        elif s.table:
            extra = f", {len(s.table)} " + tr("points")
        return tr("plasticity") + f": {tr(model)}, {tr(hard)}{extra}"

    # ------------------------------------------------------------------ check and curve
    def refresh(self) -> None:
        """Check (plasticity.validate_material) + curve. An elastic material clears the status."""
        self.bi_note.setText(self._bi_text())
        if not self.isChecked():
            self.status.setText("")
            self.tab_msg.setText("")
            return
        try:
            m = self._trial_material()
        except ValueError as e:
            self.status.setText(f"<span style='color:#d23b3b'>● {html.escape(str(e))}</span>")
            return
        errors, warnings = validate_material(m)
        rows = [f"<span style='color:#d23b3b'>● {html.escape(tr(e))}</span>" for e in errors]
        rows += [f"<span style='color:#b36b00'>● {html.escape(tr(w))}</span>" for w in warnings]
        if not errors:
            rows.append("<span style='color:#2e9e4f'>● " + html.escape(tr("Plasticity definition is complete – the study runs nonlinear (CalculiX).")) + "</span>")
        self.status.setText("<br>".join(rows))
        if not errors:
            self._draw(m)

    def _bi_text(self) -> str:
        E, et = self._E(), self._et()
        if et >= E:
            return ""
        H = E * et / (E - et) if et > 0 else 0.0
        re_ = self.win.m_Re.value()
        return tr("Yield at Re = {0} MPa (field above); Et = {1} MPa, H = {2} MPa").format(
            fmt(re_), f"{et:.6g}", f"{H:.6g}")

    def _draw(self, m) -> None:
        if not self.isVisible():
            self._pending = m
            return
        from fea_core.plasticity_eval import stress_strain
        try:
            eps, sig = stress_strain(m)
        except ValueError:
            return
        ax = self.plot.axes()
        ax.plot(eps * 100.0, sig, color="#c0392b")
        ax.set_xlabel(tr("total strain ε [%]"))
        ax.set_ylabel(tr("true stress σ [MPa]"))
        ax.grid(True, alpha=0.3)
        self.plot.canvas.draw_idle()

    def showEvent(self, e):
        super().showEvent(e)
        if getattr(self, "_pending", None) is not None:
            m, self._pending = self._pending, None
            self._draw(m)

    # ------------------------------------------------------------------ table helpers
    def _fill_table_from_curve(self):
        """Switching to the table: start from the current Re (bilinear points), so the user edits a valid curve."""
        re_ = self.win.m_Re.value()
        if re_ > 0:
            from fea_core.plasticity import make_bilinear_table
            try:
                self.set_table(make_bilinear_table(re_, self._et(), E=self._E()))
            except ValueError:
                self.set_table([(re_, 0.0)])

    def add_row(self):
        r = self.tab.rowCount()
        self.tab.insertRow(r)

    def remove_rows(self):
        for r in sorted({i.row() for i in self.tab.selectedIndexes()}, reverse=True):
            self.tab.removeRow(r)
        self.refresh()

    def paste(self):
        self.paste_text(QApplication.clipboard().text())

    def paste_text(self, text: str):
        try:
            rows = parse_table(text)
        except ValueError as e:
            self.tab_msg.setText(f"<span style='color:#d23b3b'>{html.escape(str(e))}</span>")
            return
        self.set_table(rows)
        self.tab_msg.setText(html.escape(tr("{0} rows pasted").format(len(rows))))
        self.refresh()

    def convert_engineering(self):
        from fea_core.plasticity_eval import true_table_from_engineering
        try:
            rows = true_table_from_engineering(self.table_rows(), self._E())
        except ValueError as e:
            self.tab_msg.setText(f"<span style='color:#d23b3b'>{html.escape(str(e))}</span>")
            return
        self.set_table(rows)
        self.tab_msg.setText(html.escape(tr("Converted from the engineering curve (σ, total ε) to true stress and plastic strain")))
        self.refresh()

    def keyPressEvent(self, e):
        from PySide6.QtGui import QKeySequence
        if e.matches(QKeySequence.Paste) and self.tab.hasFocus():
            self.paste()
            return
        super().keyPressEvent(e)

    # ------------------------------------------------------------------ load history
    def add_unloading(self):
        st = self.win.study
        if st is None:
            return
        from fea_core.plasticity_eval import add_unloading_step
        if add_unloading_step(st):
            self.win.changed()
            self.win.log_msg(tr("Load history: unloading step added (Contacts step → Nonlinear analysis – steps)."))
        else:
            self.win.log_msg(tr("The load history already ends with an unloading step (or the study has no load case)."))
