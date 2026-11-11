"""Results → "Plasticity assessment…": max PEEQ and where, EN 1993-1-5 annex C strain limit, ASME VIII-2 5.3.3.1 local
failure (triaxial strain limit, Table 5.7), residual stress of unloading steps. Computation in `fea_core.plasticity_eval`
(same as MCP fea_plasticity_assess and the Word report).
"""
from __future__ import annotations

import html

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QPushButton,
                               QTableWidgetItem, QVBoxLayout)

from fea_core import plasticity_eval as PE
from fea_core.i18n import tr

from .widgets import Section, dspin, fmt, help_tip, table

ASSUMPTIONS = ("PEEQ is the nodal value extrapolated and averaged by CalculiX; at singularities (sharp inner corners, support "
               "edges, point loads) it grows with mesh refinement – assess away from them. The curve is a material model "
               "(true stress), beyond its last point the stress stays constant. The ASME check uses the nodal stress "
               "triaxiality and Table 5.7 constants – it does not replace the full design procedure.")


def _item(text, align_right=False):
    it = QTableWidgetItem(str(text))
    if align_right:
        from PySide6.QtCore import Qt
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return it


def where_text(w: dict) -> str:
    x, y, z = w["xyz"]
    return f"{', '.join(w['parts'])} [{fmt(x, 4)}; {fmt(y, 4)}; {fmt(z, 4)}] mm"


class PlasticityDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Plasticity assessment")
        self.resize(820, 600)
        lay = QVBoxLayout(self)
        intro = QLabel("Equivalent plastic strain PEEQ per load step: EN 1993-1-5 annex C (PEEQ ≤ limit of the material, "
                       "default 5 %) and ASME VIII-2 5.3.3.1 local failure (PEEQ + ε_cf ≤ ε_L, triaxial limit).")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        g = Section("ASME VIII-2 parameters", key="plast.asme", detachable=False,
                    tip="Material class and strain limits of ASME VIII-2 Table 5.7",
                    help="ε<sub>L</sub> = ε<sub>Lu</sub>·exp[−α<sub>sl</sub>/(1 + m<sub>2</sub>)·((σ<sub>1</sub> + σ<sub>2</sub> + "
                         "σ<sub>3</sub>)/(3σ<sub>e</sub>) − 1/3)], R = Re/Rm. Table 5.7: ferritic steel m<sub>2</sub> = "
                         "0.60(1 − R), α<sub>sl</sub> = 2.2; austenitic stainless steel / nickel alloy m<sub>2</sub> = "
                         "0.75(1 − R), α<sub>sl</sub> = 0.6. ε<sub>Lu</sub> = m<sub>2</sub> without a specified elongation, "
                         "otherwise max(m<sub>2</sub>, ln(1 + A)). ε<sub>cf</sub> = forming strain (cold work).")
        f = QFormLayout(g.body)
        self.cls = QComboBox()
        self.cls.addItem("from the material name", "")
        for k, c in PE.ASME_CLASSES.items():
            self.cls.addItem(c["label"], k)
        help_tip(self.cls, "ASME Table 5.7 material class (m2, α_sl); auto = from the material name")
        f.addRow("Material class", self.cls)
        self.eps_lu = dspin(0.0, 0.0, 5.0, 4, 0.01, special="Table 5.7")
        help_tip(self.eps_lu, "Uniaxial strain limit ε_Lu (Table 5.7 = m2 or ln(1 + A))")
        f.addRow("ε_Lu", self.eps_lu)
        self.eps_cf = dspin(0.0, 0.0, 1.0, 4, 0.01)
        help_tip(self.eps_cf, "Forming strain ε_cf added to PEEQ (cold-formed parts; 0 = none)")
        f.addRow("ε_cf", self.eps_cf)
        lay.addWidget(g)

        self.steps = table(["Step", "max PEEQ", "Where (part, x; y; z)", "EN 1993 util.", "ASME util.", "Triaxiality",
                            "Residual σ_vM max [MPa]"], placeholder="No step with plastic strain PEEQ", rows=(2, 8))
        help_tip(self.steps, "Assessment per step (utilization ≤ 1 = passes); double-click = show PEEQ in 3D")
        self.steps.cellDoubleClicked.connect(lambda r, _c: self.show_step(r))
        lay.addWidget(self.steps)
        self.mats = table(["Material", "max PEEQ", "EN limit", "R = Re/Rm", "m2", "α_sl", "ε_Lu", "ASME util."],
                          placeholder="No plastic material", rows=(1, 5))
        help_tip(self.mats, "Per material of the selected step: limits and constants used")
        lay.addWidget(self.mats)
        self.info = QLabel()
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        note = QLabel(ASSUMPTIONS)
        note.setWordWrap(True)
        note.setStyleSheet("color:#d23b3b")
        lay.addWidget(note)
        h = QHBoxLayout()
        b = QPushButton("Show PEEQ in 3D")
        b.setToolTip("Show the equivalent plastic strain of the selected step in the results view")
        b.clicked.connect(lambda: self.show_step(self.steps.currentRow()))
        h.addWidget(b)
        h.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        h.addWidget(bb)
        lay.addLayout(h)
        for w in (self.eps_lu, self.eps_cf):
            w.valueChanged.connect(lambda _: self.recompute())
        self.cls.currentIndexChanged.connect(lambda _: self.recompute())
        self.steps.currentCellChanged.connect(lambda r, *_: self.fill_materials(r))
        self.rows: list[dict] = []
        self.recompute()

    def params(self) -> PE.AsmeParams:
        return PE.AsmeParams(self.cls.currentData() or None, eps_Lu=self.eps_lu.value() or None, eps_cf=self.eps_cf.value())

    def recompute(self):
        res, st = self.win.results, self.win.study
        try:
            self.rows = PE.summary(st, res, self.params()) if res is not None and st is not None else []
        except ValueError as e:
            self.rows = []
            self.info.setText(f"<span style='color:#d23b3b'>{html.escape(tr(str(e)))}</span>")
            return
        t = self.steps
        t.setRowCount(0)
        bad_asme = []
        for row in self.rows:
            en, am = row["en1993"], row["asme"]
            r = t.rowCount()
            t.insertRow(r)
            name = QTableWidgetItem(row["step"] + (" " + tr("(partial)") if en["partial"] else ""))
            t.setItem(r, 0, name)
            t.setItem(r, 1, _item(fmt(en["max_peeq"], 4), True))
            t.setItem(r, 2, _item(where_text(en["where_peeq"])))
            t.setItem(r, 3, _item(fmt(en["utilization"], 3) + ("" if en["ok"] else " !"), True))
            t.setItem(r, 4, _item((fmt(am["utilization"], 3) + ("" if am["ok"] else " !")) if am else "–", True))
            t.setItem(r, 5, _item(fmt(am["triaxiality_at_max"], 3) if am else "–", True))
            resid = en.get("residual")
            t.setItem(r, 6, _item(fmt(resid["max_mises_MPa"], 4) if resid else "–", True))
            if row["asme_error"]:
                bad_asme.append(row["asme_error"])
        msgs = []
        ex = sorted({m for row in self.rows for m in row["en1993"]["curve_end_exceeded"]})
        if ex:
            msgs.append(tr("PEEQ exceeds the last point of the hardening curve (stress then stays constant): ")
                        + html.escape(", ".join(ex)))
        if bad_asme:
            msgs.append(tr("ASME check not possible: ") + html.escape(tr(bad_asme[0])))
        if any(r["en1993"]["partial"] for r in self.rows):
            msgs.append(tr("Partial result (the analysis did not converge to the end) – the last converged increment is assessed."))
        self.info.setText("<br>".join(f"<span style='color:#b36b00'>● {m}</span>" for m in msgs))
        if t.rowCount():
            t.selectRow(0)
            self.fill_materials(0)

    def fill_materials(self, r: int):
        t = self.mats
        t.setRowCount(0)
        if not 0 <= r < len(self.rows):
            return
        en, am = self.rows[r]["en1993"], self.rows[r]["asme"]
        for name, me in en["materials"].items():
            ma = (am or {}).get("materials", {}).get(name, {})
            i = t.rowCount()
            t.insertRow(i)
            it = QTableWidgetItem(name)
            t.setItem(i, 0, it)
            t.setItem(i, 1, _item(fmt(me["max_peeq"], 4), True))
            t.setItem(i, 2, _item(fmt(me["limit_strain"], 3), True))
            for c, k in ((3, "R"), (4, "m2"), (5, "alpha_sl"), (6, "eps_Lu"), (7, "utilization")):
                v = ma.get(k)
                t.setItem(i, c, _item(fmt(v, 3) if v is not None else "–", True))

    def show_step(self, r: int):
        if not 0 <= r < len(self.rows):
            return
        w = self.win
        k = w.r_step.findData(f"step:{self.rows[r]['index']}")
        if k >= 0:
            w.r_step.setCurrentIndex(k)
        q = w.r_qty.findText("Equivalent plastic strain PEEQ")
        if q >= 0:
            w.r_qty.setCurrentIndex(q)
        w.show_result()
