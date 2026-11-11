""""New geometry – remapping the settings" window (fea_core.cad_link) and the table of the geometry iterations."""
from __future__ import annotations

from html import escape

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QTableWidgetItem,
                               QTabWidget, QVBoxLayout)

from fea_core.cad_link.remap import STATUS_LABEL
from .widgets import fmt, table

COLORS = {"same": "#3a9d5d", "cad": "#3a7bd5", "probable": "#e0a800", "split": "#e07b00",
          "ambiguous": "#d23b3b", "lost": "#d23b3b", "shifted": "#e0a800", "review": "#d23b3b"}
PART_METHOD = {"cad": "by CAD", "name": "same name", "name_key": "name without suffix",
               "geometry": "by geometry", "added": "new part", "removed": "removed part"}
OPTIONAL = ("probable", "split")


def _status_item(status: str, text: str | None = None) -> QTableWidgetItem:
    it = QTableWidgetItem(text or STATUS_LABEL.get(status, status))
    it.setBackground(QBrush(QColor(COLORS.get(status, "#999999")).lighter(170)))
    it.setForeground(QBrush(QColor("#000000")))
    return it


class RemapDialog(QDialog):
    """Remapping proposal: selections (state, who uses them), parts, moved points, notes. Uncertain proposals can be
    rejected (untick) – the selection then keeps the old faces and the readiness check reports it.
    """

    def __init__(self, parent, report, title: str):
        super().__init__(parent)
        self.report = report
        self.setWindowTitle("New geometry – remapping the setup")
        self.resize(900, 560)
        lay = QVBoxLayout(self)
        counts = ", ".join(f"{STATUS_LABEL[k]} {v}" for k, v in report.counts().items())
        head = QLabel(f"<b>{escape(title)}</b><br>Selections: {counts or 'none'}. Green and blue = certain matches, yellow = proposal to review (uncheck to reject it), red = not found – fix the selection manually after loading.")
        head.setWordWrap(True)
        lay.addWidget(head)
        tabs = QTabWidget()
        lay.addWidget(tabs, 1)
        # selections
        self.sel_table = table(["Selection", "State", "Used by", "Note"], stretch_last=True)
        rows = [s for s in report.selections]
        self.sel_table.setRowCount(len(rows))
        for r, s in enumerate(rows):
            name = QTableWidgetItem(s.name)
            if s.status in OPTIONAL:
                name.setFlags(name.flags() | Qt.ItemIsUserCheckable)
                name.setCheckState(Qt.Checked)
            self.sel_table.setItem(r, 0, name)
            self.sel_table.setItem(r, 1, _status_item(s.status))
            self.sel_table.setItem(r, 2, QTableWidgetItem("; ".join(s.users) or "–"))
            notes = "; ".join(dict.fromkeys(it.note for it in s.items if it.note))
            self.sel_table.setItem(r, 3, QTableWidgetItem(notes))
        self.sel_table.setToolTip("Checked yellow proposals are applied; unchecked ones stay for review")
        tabs.addTab(self.sel_table, f"Selections ({len(rows)})")
        # parts
        pt = table(["Original part", "New part", "Matching", "Geometry change"], stretch_last=True)
        pt.setRowCount(len(report.parts))
        for r, p in enumerate(report.parts):
            pt.setItem(r, 0, QTableWidgetItem(p.old or "–"))
            pt.setItem(r, 1, QTableWidgetItem(p.new or "–"))
            st = "same" if p.certain else "probable" if p.method == "geometry" else "review"
            pt.setItem(r, 2, _status_item(st, PART_METHOD.get(p.method, p.method)))
            d = p.to_dict()
            ch = (f"volume {fmt(d['volume_change_pct'], 3)} %" if "volume_change_pct" in d and p.changed
                  else "changed" if p.changed else "no changes")
            pt.setItem(r, 3, QTableWidgetItem(ch))
        tabs.addTab(pt, f"Parts ({len(report.parts)})")
        # points and coordinates
        self.anc_table = table(["Where", "State", "Before [mm]", "After [mm]", "Note"], stretch_last=True)
        anchors = [(k, a) for k, a in enumerate(report.anchors) if a.status != "same"]
        self._anchors = anchors
        self.anc_table.setRowCount(len(anchors))
        for r, (_, a) in enumerate(anchors):
            w = QTableWidgetItem(a.where)
            if a.status == "shifted":
                w.setFlags(w.flags() | Qt.ItemIsUserCheckable)
                w.setCheckState(Qt.Checked)
            self.anc_table.setItem(r, 0, w)
            self.anc_table.setItem(r, 1, _status_item(a.status, "shifted" if a.status == "shifted" else "check"))
            self.anc_table.setItem(r, 2, QTableWidgetItem(", ".join(fmt(v, 5) for v in a.old)))
            self.anc_table.setItem(r, 3, QTableWidgetItem(", ".join(fmt(v, 5) for v in a.new)))
            self.anc_table.setItem(r, 4, QTableWidgetItem(a.note))
        tabs.addTab(self.anc_table, f"Points ({len(anchors)})")
        if report.notes:
            nl = QLabel()
            nl.setTextFormat(Qt.PlainText)                 # the notes carry the part names from a foreign CAD (format first:
            nl.setText("\n".join("• " + n for n in report.notes))     # QLabel(text) would already parse them as rich text)
            nl.setWordWrap(True)
            nl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            tabs.addTab(nl, f"Notes ({len(report.notes)})")
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Use new geometry")
        bb.button(QDialogButtonBox.Ok).setToolTip("Load the new geometry with the remapped setup")
        bb.button(QDialogButtonBox.Cancel).setToolTip("Keep the current geometry unchanged")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def rejected_selections(self) -> set[str]:
        out = set()
        for r in range(self.sel_table.rowCount()):
            it = self.sel_table.item(r, 0)
            if it.flags() & Qt.ItemIsUserCheckable and it.checkState() != Qt.Checked:
                out.add(it.text())
        return out

    def rejected_anchors(self) -> set[int]:
        out = set()
        for r, (k, _) in enumerate(self._anchors):
            it = self.anc_table.item(r, 0)
            if it.flags() & Qt.ItemIsUserCheckable and it.checkState() != Qt.Checked:
                out.add(k)
        return out


KPI_COLS = [("vm_max_clean", "σ max outside singularities [MPa]"), ("vm_max", "σ max [MPa]"),
            ("u_max", "max displacement [mm]"), ("k_min", "safety factor to Re"), ("mass_kg", "mass [kg]"),
            ("f1_Hz", "1st frequency [Hz]")]


class IterationsDialog(QDialog):
    """Table of the geometry iterations with the difference against the previous iteration and a chart of the chosen quantity."""

    def __init__(self, parent, study):
        super().__init__(parent)
        from fea_core.cad_link.history import cases_of
        self.link = study.cad_link
        self.setWindowTitle("Geometry iterations")
        self.resize(1000, 560)
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Load case:"))
        self.case = QComboBox()
        self.case.addItems(cases_of(self.link) or ["–"])
        self.case.setToolTip("Load case shown in the table and the chart")
        row.addWidget(self.case)
        row.addWidget(QLabel("Chart:"))
        self.kpi = QComboBox()
        self.kpi.addItems([t for _, t in KPI_COLS])
        self.kpi.setToolTip("Quantity in the chart versus iteration number")
        row.addWidget(self.kpi)
        row.addStretch(1)
        lay.addLayout(row)
        self.tab = table(["Iteration", "Time", "Changes", "CAD parameters", "Selections"] + [t for _, t in KPI_COLS],
                         stretch_last=False)
        lay.addWidget(self.tab, 1)
        from .widgets import LazyPlot
        self.plot = LazyPlot(self)
        lay.addWidget(self.plot)
        self.case.currentIndexChanged.connect(self.refresh)
        self.kpi.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def refresh(self):
        from fea_core.cad_link.history import table as rows_of
        case = self.case.currentText()
        rows = rows_of(self.link, None if case == "–" else case)
        self.tab.setRowCount(len(rows))
        for r, row in enumerate(rows):
            vals = [str(row["n"]), (row["time"] or "").replace("T", " "), "; ".join(row["changes"]) or "–",
                    ", ".join(f"{k} = {v}" for k, v in row["params"].items()) or "–",
                    ", ".join(f"{STATUS_LABEL.get(k, k)} {v}" for k, v in row["remap"].items()) or "–"]
            for c, v in enumerate(vals):
                self.tab.setItem(r, c, QTableWidgetItem(v))
            for c, (key, _) in enumerate(KPI_COLS, len(vals)):
                v = row.get(key)
                d = (row.get("delta_pct") or {}).get(key)
                txt = "–" if v is None else fmt(v, 5) + (f" ({'+' if d >= 0 else ''}{fmt(d, 3)} %)" if d is not None else "")
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tab.setItem(r, c, it)
        key = KPI_COLS[self.kpi.currentIndex()][0]
        pts = [(row["n"], row[key]) for row in rows if row.get(key) is not None]
        ax = self.plot.axes()
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-")
        ax.set_xlabel("iteration")
        ax.set_ylabel(KPI_COLS[self.kpi.currentIndex()][1])
        ax.grid(True, alpha=0.3)
        self.plot.canvas.draw_idle()
