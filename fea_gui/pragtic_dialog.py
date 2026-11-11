"""Choice of S-N from the PragTic material database (J. Papuga) – search, all values of the record with the sources."""
from __future__ import annotations

import math
from html import escape

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QSplitter, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)

from fea_core.fatigue import pragtic
from fea_core.i18n import decimal

from .safe_browser import SafeTextBrowser
from .widgets import help_tip, table


def _f(v, dec=4):
    if v is None:
        return ""
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    return decimal(f"{v:.{dec}g}")


class PragticDialog(QDialog):
    """After acceptance: `self.sn` (SNCurve), `self.record` (dict), `self.take_static` (take Rm, Re over)."""

    COLS = ["Match", "Designation", "Treatment", "Rm", "Re", "E", "Fatigue data"]

    def __init__(self, material_name: str, mat_rm: float | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"S-N from the PragTic database – material “{material_name}”")
        self.resize(1100, 700)
        self.mat_rm = mat_rm
        self.sn = self.record = None
        self.take_static = False
        self.recs: list[dict] = []
        lay = QVBoxLayout(self)
        db = pragtic.database()
        info = QLabel(f"Material database <b>PragTic</b> (Jan Papuga, CTU Prague) – {len(db['records'])} records from the literature, downloaded {db['downloaded']}. The values are from smooth test bars; surface, size and reliability are accounted for by Marin factors in the Fatigue panel.")
        info.setWordWrap(True)
        lay.addWidget(info)
        cite = QLabel(f"<small><b>Citation:</b> {escape(pragtic.citation(db))}<br>{escape(pragtic.NOTICE)} <a href='{pragtic.URL}'>{pragtic.URL}</a></small>")
        cite.setWordWrap(True)
        cite.setOpenExternalLinks(True)
        cite.setTextInteractionFlags(Qt.TextBrowserInteraction)
        cite.setToolTip("Data source – cite it in your calculation documentation too (the Word report adds it itself)")
        lay.addWidget(cite)
        h = QHBoxLayout()
        self.q = QLineEdit()
        self.q.setPlaceholderText("Designation per any standard (12050, C45, 1.0503, AISI 1045, S45C, St 52-3, EN AW-2024…), optionally + treatment (quenched)")
        self.q.setClearButtonEnabled(True)
        help_tip(self.q, "Search by material grade (the converter finds equivalents)",
                 "Database search: designation per any standard – the grade converter finds equivalents\n(ČSN, EN, W.Nr., AISI, JIS…) and similar materials; optionally a treatment text.\nPrefilled with the study material designation.")
        self.q.textChanged.connect(self.fill)
        self.sn_only = QCheckBox("only with S-N data (tension-compression / bending)")
        self.sn_only.setChecked(True)
        help_tip(self.sn_only, "Only records with S-N data",
                 "Show only records with S-N data (fatigue limit / Basquin in tension-compression or bending).\nOff = also records with only static or torsion values.")
        self.sn_only.toggled.connect(self.fill)
        h.addWidget(self.q, 1)
        h.addWidget(self.sn_only)
        lay.addLayout(h)
        self.conv = QLabel()
        self.conv.setWordWrap(True)
        self.conv.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.conv)

        split = QSplitter(Qt.Horizontal)
        self.table = table(self.COLS)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        help_tip(self.table, "Records found; click = details, double click = use S-N",
                 "Records found. Match: match = same designation per the converter, ≈ match = approximate equivalent,\ntext = text-only match, similar (italic) = same family, close composition and Rm.\nClick = details on the right, double-click = use S-N.")
        self.table.currentCellChanged.connect(lambda r, *_: self.on_row(r))
        self.table.cellDoubleClicked.connect(lambda *_: self.accept_sn())
        split.addWidget(self.table)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.details = SafeTextBrowser()
        self.details.setOpenExternalLinks(True)
        self.details.setToolTip("All values of the selected record with sources (links open in the browser)")
        rv.addWidget(self.details, 1)
        rv.addWidget(QLabel("<b>S-N curve for the analysis</b>"))
        self.variant = QComboBox()
        help_tip(self.variant, "S-N variant from the record (first = recommended)",
                 "S-N variant from the record (first = recommended): tension-compression fatigue limit, Basquin curve, bending limit…\nIt is converted to SD, ND, k, k2 without Marin factors – those are applied in the Fatigue panel.")
        self.variant.currentIndexChanged.connect(self.on_variant)
        rv.addWidget(self.variant)
        self.sn_text = QLabel()
        self.sn_text.setWordWrap(True)
        rv.addWidget(self.sn_text)
        self.static = QCheckBox("also take over Rm and Re into the material (mean stress correction uses Rm)")
        help_tip(self.static, "Overwrites Rm and Re of the material with the record values",
                 "Overwrites Rm and Re of the study material with the record values (the source is appended to the material).\nAlso affects the static evaluation (safety factor against Re) and the surface factor ka.")
        rv.addWidget(self.static)
        split.addWidget(right)
        split.setSizes([520, 580])
        lay.addWidget(split, 1)
        bb = QDialogButtonBox()
        self.ok = bb.addButton("Use S-N", QDialogButtonBox.AcceptRole)
        self.ok.setToolTip("Use the selected S-N variant for the material (saved in the study's fatigue settings)")
        bb.addButton("Cancel", QDialogButtonBox.RejectRole).setToolTip("Close without changes")
        bb.accepted.connect(self.accept_sn)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        from fea_core import grades
        g = grades.grade_of(material_name)          # pre-fill with the material designation of the study
        self.q.setText(g.key if g else "")
        self.fill()

    def fill(self, *_):
        hits, exact, similar = pragtic.find(self.q.text(), self.sn_only.isChecked())
        self.show_conversion(hits, len(exact), len(similar))
        self.recs = exact + [r for r, _, _ in similar]
        self.table.setRowCount(len(self.recs))
        italic = self.table.font()
        italic.setItalic(True)
        for r, rec in enumerate(self.recs):
            s = pragtic.summary(rec)
            mark = rec["mark"] + (f" ({rec['alt_mark']})" if rec.get("alt_mark") else "")
            sim = r >= len(exact)
            if sim:
                _, d, why = similar[r - len(exact)]
                rel = "similar"
                tip = f"Similar material (distance {_f(d, 2)}): {why or 'same family, similar composition'}"
            else:
                h = pragtic.grade_of_record(rec)
                if h and any(h[0] is g for g, _ in hits):
                    rel = ("≈ match" if h[1] else "match")
                else:
                    rel = "text" if hits else ""
                tip = f"{h[0].key}: {h[0].summary()}" if h else ""
            for c, txt in enumerate([rel, mark, rec.get("treatment", ""), _f(s["Rm"]), _f(s["Re"]), _f(s["E"], 6),
                                     s["available"]]):
                it = QTableWidgetItem(txt)
                if c in (3, 4, 5):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if sim:
                    it.setFont(italic)
                it.setToolTip(tip)
                self.table.setItem(r, c, it)
        self.table.blockSignals(True)                   # row 0 stays → Qt does not emit the change
        self.table.setCurrentCell(0 if self.recs else -1, 0)
        self.table.blockSignals(False)
        self.on_row(self.table.currentRow())

    def show_conversion(self, hits, n_exact, n_similar):
        if not self.q.text().strip():
            self.conv.setText("")
            return
        if not hits:
            self.conv.setText("<i>The designation is not in the converter – only the text in the database is searched.</i>")
            return
        g, approx = hits[0]
        txt = (f"<b>{g.key}</b> ({g.family_label}){' – approximate match only' if approx else ''}: "
               f"{g.summary(html=True)}")
        if g.note:
            txt += f" <small>({g.note})</small>"
        if len(hits) > 1:
            txt += "<br>Also: " + ", ".join(f"{h.key}{' (≈)' if a else ''}" for h, a in hits[1:4])
        txt += f"<br>In the database: {n_exact} matching records, {n_similar} similar (italic)."
        self.conv.setText(txt)

    def on_row(self, row):
        self.variant.blockSignals(True)
        self.variant.clear()
        if not (0 <= row < len(self.recs)):
            self.record = None
            self.details.setHtml("<i>Nothing found.</i>")
        else:
            self.record = rec = self.recs[row]
            self.details.setHtml(pragtic.details_html(rec))
            self._options = pragtic.sn_options(rec)
            for lbl, _ in self._options:
                self.variant.addItem(lbl)
            s = pragtic.summary(rec)
            self.static.setEnabled(bool(s["Rm"]))
        self.variant.blockSignals(False)
        self.on_variant()

    def on_variant(self, *_):
        i = self.variant.currentIndex()
        if self.record is None or i < 0:
            self.sn_text.setText("<span style='color:#b36b00'>The record has no S-N data (only static or torsion values) – select another one.</span>" if self.record else "")
            self.ok.setEnabled(False)
            return
        sn = self._options[i][1]
        rm = pragtic.summary(self.record)["Rm"]
        txt = (f"SD = {_f(sn.SD)} MPa at ND = {_f(sn.ND)} cycles, k = {_f(sn.k, 3)}, k2 = {(_f(sn.k2, 3) if sn.k2 > 0 else 'fatigue limit')}")
        if not 2.5 <= sn.k <= 30:
            txt += ("<br><span style='color:#b36b00'>Unusual S-N slope (typically k = 3–25 for metals) – verify the values in the original source.</span>")
        if rm and self.mat_rm and abs(rm - self.mat_rm) > 0.1 * self.mat_rm:
            txt += (f"<br><span style='color:#b36b00'>The record's Rm {_f(rm)} MPa differs from the material's Rm {_f(self.mat_rm)} MPa by more than 10 % – consider whether it is a corresponding material condition.</span>")
        self.sn_text.setText(txt)
        self.ok.setEnabled(True)

    def accept_sn(self):
        i = self.variant.currentIndex()
        if self.record is None or i < 0:
            return
        self.sn = self._options[i][1]
        self.take_static = self.static.isEnabled() and self.static.isChecked()
        self.accept()
