"""Legend of the colour bands as a Qt widget (always readable, does not cover the model)."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from fea_core.i18n import tr


class Legend(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = ""
        self.edges = []
        self.colors = []
        self.note = ""                   # note to the right of the scale (e.g. the maximum is a singularity)
        self.setMinimumHeight(330)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_data(self, title, edges, colors, fmt):
        self.title = tr(title)
        self.colors = list(colors)
        self.labels = [fmt(v) for v in edges]
        self.edges = list(edges)
        self.update()

    def set_note(self, text: str):
        self.note = text or ""
        self.setToolTip(self.note)
        self.update()

    def clear(self):
        self.edges = []
        self.note = ""
        self.setToolTip("")
        self.update()

    def paintEvent(self, e):
        if not self.edges:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pal = self.palette()
        fg = pal.windowText().color()
        f = QFont(self.font())
        f.setBold(True)
        p.setFont(f)
        p.setPen(fg)
        p.drawText(QRectF(4, 0, self.width() - 8, 20), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        p.setFont(self.font())
        n = len(self.colors)
        top, bottom = 30.0, self.height() - 10.0
        h = (bottom - top) / n
        x0, w = 8.0, 26.0
        for k in range(n):               # highest value at the top
            c = self.colors[k]
            y = bottom - (k + 1) * h
            p.fillRect(QRectF(x0, y, w, h), QColor(*c))
        p.setPen(QPen(QColor(60, 60, 60), 1))
        p.drawRect(QRectF(x0, top, w, bottom - top))
        p.setPen(fg)
        for k, lab in enumerate(self.labels):
            y = bottom - k * h
            p.drawLine(int(x0 + w), int(y), int(x0 + w + 5), int(y))
            p.drawText(QRectF(x0 + w + 9, y - 9, self.width() - x0 - w - 12, 18),
                       Qt.AlignLeft | Qt.AlignVCenter, lab)
        if self.note:                    # note to the scale, to the right of the labels (at the maximum, at the top)
            fm = p.fontMetrics()
            x = x0 + w + 9 + max((fm.horizontalAdvance(lab) for lab in self.labels), default=0) + 18
            if self.width() - x >= 110:
                p.setPen(QColor(200, 120, 10))
                p.drawText(QRectF(x, top - 6, self.width() - x - 6, bottom - top),
                           Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, "! " + self.note)
        p.end()
