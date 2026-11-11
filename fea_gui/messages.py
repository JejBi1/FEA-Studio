"""Messages table (spec chapter 13): errors, warnings and information from the check of the study, the mesh, the solver and the results.

Each source has its own set of messages (`set_source` replaces it – the study check is refreshed after every change, the mesh after
meshing, the results after the calculation); `add` adds one message. The messages stay in the solver log too (the main window keeps writing there).
A double click on a row switches to the step where the message is corrected, if it is clear (`stage_for`).
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .widgets import _paste_safe, help_tip, table

LEVELS = {"error": ("error", "#d23b3b"), "warning": ("warning", "#e69a1a"), "info": ("information", "#2e9e4f")}
ORDER = {"error": 0, "warning": 1, "info": 2}
SOURCES = ("geometry check", "CAD import", "study check", "mesh", "solver", "results")
MODEL_SOURCES = ("study check", "geometry check", "CAD import")     # "Clear" does not delete them (they belong to the model)
FILTERS = [("all", "All"), ("problems", "Errors and warnings"), ("error", "Errors only"), ("warning", "Warnings only"),
           ("info", "Information only")]

# keywords of the text → step of the progress bar (first match; order from the most specific)
STAGE_WORDS = [
    (("Mechanism", "mechanism", "Joint '", "Drive '", "Force element", "Body '", "Link", "Grübler",
      "Redundant", "Missing driver", "Mobility check", "Motion ", "Collision of bodies",
      "Mechanism", "mechanism", "Joint '", "Drive '", "Motion"), "Motion"),
    (("Contact", "contact", "Contact", "contact", "Step sequence", "Step '", "Large deformations", "Nonlinear analysis"), "Contacts"),
    (("Dynamics", "Harmonic response", "Transient response", "Random vibration", "Shock", "Added mass",
      "Base excitation", "Prestressed modal", "Spectral fatigue", "Dynamics"), "Dynamics"),
    (("Orthotropic", "material", "Material", "material", "Material"), "Materials"),
    (("3D printing", "Print", "printed", "Creep", "creep", "3D print", "Print", "print"), "3D printing"),
    (("*TIE", "Bolt", "bolt", "bolt", "Weld", "weld", "Weld", "weld", "free bodies", "free bodies", "joint"), "Joints"),
    (("Elastic support", "Cylindrical support", "Remote support", "Elastic support", "support", "Support", "rigid-body", "constraint", "Constraint", "rigid-body motion"), "Constraints"),
    (("Thermal case", "thermal", "Heat", "Thermal", "thermal", "Heat"), "Heat"),
    (("Natural frequencies", "Buckling", "buckling", "Buckling", "buckling"), "Solve"),
    (("State '", "Load case", "load case", "load", "Loads", "Load", "load", "Selection '", "Selection '"), "Loads"),
    (("mesh", "Mesh", "mesh", "Mesh", "element", "element", "Gmsh"), "Mesh"),
]


def stage_for(text: str, source: str) -> str | None:
    """The step where the message is corrected / where it is visible (None = unclear)."""
    for words, stage in STAGE_WORDS:
        if any(w in text for w in words):
            return stage
    return {"mesh": "Mesh", "results": "Results", "solver": "Solve", "study check": "Solve"}.get(source)


def _dot(color: str) -> QIcon:
    pm = QPixmap(12, 12)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(1, 1, 10, 10)
    p.end()
    return QIcon(pm)


class MessagesPanel(QWidget):
    """Content of the "Messages" dock: level filter, copying, table (level, source, text, time)."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.dock = None                         # QDockWidget (the title with the counts is set by _update_title)
        self._items: list[dict] = []             # {"level", "source", "text", "time", "stage"}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(3)
        h = QHBoxLayout()
        self.filter = QComboBox()
        for key, txt in FILTERS:
            self.filter.addItem(txt, key)
        self.filter.setToolTip("Which messages to show: all, errors and warnings only, or a single level")
        self.filter.currentIndexChanged.connect(lambda _: self._fill())
        h.addWidget(QLabel("Show"))
        h.addWidget(self.filter)
        self.count = QLabel("")
        self.count.setProperty("muted", True)
        h.addWidget(self.count, 1)
        b = QPushButton("Copy")
        help_tip(b, "Copy selected message to clipboard",
                 "Copy selected message (or all displayed if none selected) to clipboard as tab-delimited text – can be pasted into email or Excel.")
        b.clicked.connect(self.copy)
        h.addWidget(b)
        self.btn_copy = b
        b = QPushButton("Clear")
        help_tip(b, "Clear mesh, solver and result messages",
                 "Clear mesh, solver and result messages (study check messages are refreshed automatically after every study change). The solver log is not cleared.")
        b.clicked.connect(self.clear)
        h.addWidget(b)
        lay.addLayout(h)
        self.table = table(["Level", "Source", "Text", "Time"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setToolTip("Program messages: errors, warnings and information from checking, meshing and solver")
        self.table.setWhatsThis("<b>Messages</b><p>Overview of everything the program reports: <i>study check</i> (refreshed after every change), <i>mesh</i> (meshing warnings, *TIE connections), <i>solver</i> (analysis errors, run time) and <i>results</i> (result warnings – singularities, buckling linearity…).</p><p>Double-clicking a row opens the workflow step where the problem is fixed. The filter at the top selects the level; “Copy” copies the text to the clipboard.</p>")
        self.table.cellDoubleClicked.connect(lambda r, _c: self.goto(r))
        lay.addWidget(self.table)
        self._rows: list[dict] = []              # displayed messages (order of the table rows)

    # ------------------------------------------------------------------ filling
    def set_source(self, source: str, items, stage: str | None = None, unique: bool = False):
        """Replaces the messages of the source `source` by the list [(level, text)]; the time stays with the messages that already existed.
        unique = leave out the texts that another source already reports (result warnings = warnings of the study check).
        """
        if unique:
            other = {m["text"] for m in self._items if m["source"] != source}
            items = [(lv, t) for lv, t in items if str(t) not in other]
        old = {(m["level"], m["text"]): m["time"] for m in self._items if m["source"] == source}
        now = time.strftime("%H:%M:%S")
        keep = [m for m in self._items if m["source"] != source]
        new = [{"level": lv, "source": source, "text": str(t), "time": old.get((lv, str(t)), now),
                "stage": stage or stage_for(str(t), source)} for lv, t in items]
        if [(m["level"], m["text"]) for m in self._items if m["source"] == source] == [(m["level"], m["text"])
                                                                                        for m in new]:
            return                                # unchanged – the table is not redrawn (refresh after every change)
        self._items = keep + new
        self._fill()

    def add(self, level: str, source: str, text: str, stage: str | None = None):
        self._items.append({"level": level, "source": source, "text": str(text), "time": time.strftime("%H:%M:%S"),
                            "stage": stage or stage_for(str(text), source)})
        self._fill()

    def clear(self):
        self._items = [m for m in self._items if m["source"] in MODEL_SOURCES]
        self._fill()

    def messages(self, level: str | None = None, source: str | None = None) -> list[dict]:
        return [dict(m) for m in self._items if (level is None or m["level"] == level)
                and (source is None or m["source"] == source)]

    def _visible(self, m) -> bool:
        f = self.filter.currentData()
        return f == "all" or (f == "problems" and m["level"] != "info") or m["level"] == f

    def _fill(self):
        rows = sorted((m for m in self._items if self._visible(m)),
                      key=lambda m: (ORDER.get(m["level"], 3), SOURCES.index(m["source"]) if m["source"] in SOURCES
                                     else 9))
        self._rows = rows
        t = self.table
        t.setUpdatesEnabled(False)
        t.setRowCount(len(rows))
        for r, m in enumerate(rows):
            name, col = LEVELS.get(m["level"], (m["level"], "#9aa1ab"))
            it = QTableWidgetItem(_dot(col), name)
            t.setItem(r, 0, it)
            t.setItem(r, 1, QTableWidgetItem(m["source"]))
            tx = QTableWidgetItem(m["text"])
            tx.setToolTip(m["text"] + (f"\n\nDouble-click: step {m['stage']}" if m['stage'] else ""))  # tip-ok
            t.setItem(r, 2, tx)
            t.setItem(r, 3, QTableWidgetItem(m["time"]))
        t.setUpdatesEnabled(True)
        n = {lv: sum(1 for m in self._items if m["level"] == lv) for lv in LEVELS}
        self.count.setText(f"errors {n['error']}, warnings {n['warning']}, information {n['info']}")
        if self.dock is not None:
            extra = [f"{n['error']} ×"] if n["error"] else []
            extra += [f"{n['warning']} !"] if n["warning"] else []
            self.dock.setWindowTitle("Messages" + (f" ({', '.join(extra)})" if extra else ""))

    # ------------------------------------------------------------------ akce
    def goto(self, r: int):
        if not (0 <= r < len(self._rows)):
            return
        stage = self._rows[r]["stage"]
        if stage and stage in self.win.STAGE:
            self.win.set_stage(self.win.STAGE[stage])

    def copy(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}) or list(range(len(self._rows)))
        # message texts carry names from foreign files: no tabs / line breaks inside a cell, no formula start (= + - @) for Excel
        text = "\n".join("\t".join(_paste_safe(str(v)) for v in (LEVELS.get(m["level"], (m["level"],))[0], m["source"],
                                                                 m["text"], m["time"]))
                         for m in (self._rows[r] for r in rows if r < len(self._rows)))
        QApplication.clipboard().setText(text)
        return text
