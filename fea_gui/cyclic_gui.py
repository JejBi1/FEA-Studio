"""Cyclic symmetry in the GUI (fea_core.cyclic): section in the Constraints step (cut faces, number of sectors, axis, nodal
diameters), 3D markers of the cut faces and the axis, and the results shown as the full 360° model (expanded sector)."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox

from fea_core.i18n import tr

from .widgets import Section, dspin, fmt, help_tip

C_CYCLIC = (150, 60, 200)
KEY = "cyc:"


class CyclicBox(Section):
    """Section "Cyclic symmetry (sector)" (Study.cyclic); the tick in the header = the model is one sector."""

    def __init__(self, win):
        on = QCheckBox("model is one sector")
        on.setToolTip("On = the model is one sector of a rotationally periodic part (N identical sectors)")
        super().__init__("Cyclic symmetry (sector)", check=on, key="con.cyclic", collapsed=True,
                         tip="One sector of a periodic part: cut faces tied by equations, 360° results",
                         help="The model is one sector of a rotationally periodic part (flange with N holes, impeller, disc). The two cut faces are meshed identically and tied by constraint equations; loads and supports must be the same on every sector. Statics: reactions per sector. Natural frequencies: by nodal diameter (number of waves around the circumference). Results can be shown as the full 360° model.")
        self.on = on
        self.win = win
        self._filling = False
        f = QFormLayout(self.body)

        def face_row(attr: str, label: str, tip: str):
            h = QHBoxLayout()
            combo = QComboBox()
            combo.setProperty("_i18n_done", True)          # selection names are not translated
            help_tip(combo, tip)
            combo.currentIndexChanged.connect(lambda _: self.on_change())
            b = QPushButton("From selected faces")
            b.setToolTip("Create the selection from the faces selected in 3D")
            b.clicked.connect(lambda: self.from_view(combo, label))
            h.addWidget(combo, 1)
            h.addWidget(b)
            setattr(self, attr, combo)
            f.addRow(label, h)
        face_row("left", "Cut face 1", "First cut face of the sector (planar, through the axis)")
        face_row("right", "Cut face 2", "Second cut face = first one rotated by +360°/N")
        b = QPushButton("Axis and N from the cut faces")
        b.setToolTip("Axis = intersection of the two cut planes, N = 360° / sector angle")
        b.clicked.connect(self.detect)
        f.addRow(b)
        self.n = QSpinBox()
        self.n.setRange(2, 720)
        self.n.setValue(8)
        help_tip(self.n, "Number of sectors N in 360° (sector angle = 360°/N)")
        self.n.editingFinished.connect(self.on_change)
        f.addRow("Number of sectors", self.n)

        def vec(vals, tip):
            h = QHBoxLayout()
            sp = [dspin(v, -1e7, 1e7, 4) for v in vals]
            for s, ax in zip(sp, "xyz"):
                s.setToolTip(tr(tip).format(ax))
                s.editingFinished.connect(self.on_change)
                h.addWidget(s)
            return sp, h
        self.pt, h = vec((0, 0, 0), "{} coordinate of a point on the symmetry axis [mm]")
        f.addRow("Axis point [mm]", h)
        self.dir, h = vec((0, 0, 1), "{} component of the symmetry axis direction")
        f.addRow("Axis direction", h)
        self.nds = QLineEdit()
        self.nds.setPlaceholderText(tr("all (0 … N/2)"))
        help_tip(self.nds, "Nodal diameters for natural frequencies, e.g. 0, 2-4 (empty = all)",
                 "Nodal diameters (number of waves around the circumference) for which the natural frequencies are computed, e.g. “0, 2-4”. Empty = all from 0 to N/2.")
        self.nds.editingFinished.connect(self.on_change)
        f.addRow("Nodal diameters", self.nds)
        self.expand = QCheckBox("show results as the full 360° model")
        self.expand.setChecked(True)
        self.expand.setToolTip("Results step: the sector is rotated N times (the sector results are not changed)")
        self.expand.toggled.connect(self.on_expand)
        f.addRow(self.expand)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setProperty("muted", True)
        f.addRow(self.note)
        self.on.toggled.connect(lambda _: self.on_change())

    # ------------------------------------------------------------------ study ↔ form
    def _fill_combos(self, st):
        names = [k for k, s in st.selections.items() if s.faces]
        for combo in (self.left, self.right):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("")
            combo.addItems(names)
            combo.setCurrentIndex(max(0, combo.findText(cur)))
            combo.blockSignals(False)

    def refresh(self):
        st = self.win.study
        if st is None:
            return
        self._filling = True
        try:
            self._fill_combos(st)
            c = st.cyclic
            self.on.setChecked(c is not None and c.enabled)
            if c is not None:
                self.left.setCurrentIndex(max(0, self.left.findText(c.left)))
                self.right.setCurrentIndex(max(0, self.right.findText(c.right)))
                self.n.setValue(int(c.sectors))
                for sp, v in zip(self.pt, c.axis_point):
                    sp.setValue(float(v))
                for sp, v in zip(self.dir, c.axis_dir):
                    sp.setValue(float(v))
                self.nds.setText("" if c.nodal_diameters is None else ", ".join(str(k) for k in c.nodal_diameters))
                self.expand.setChecked(bool(c.expand))
                self.note.setText(tr("Sector angle {0}°; loads and supports must be the same on every sector.").format(
                    fmt(360.0 / int(c.sectors))))
            else:
                self.note.setText("")
        finally:
            self._filling = False

    def _parse_nds(self) -> list[int] | None:
        txt = self.nds.text().strip()
        if not txt:
            return None
        out = []
        for part in txt.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                out += list(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return sorted(set(out))

    def on_change(self):
        if self._filling or self.win.study is None:
            return
        from fea_core.cyclic import CyclicSymmetry
        st = self.win.study
        try:
            nds = self._parse_nds()
        except ValueError:
            self.note.setText(tr("Nodal diameters: write integers separated by commas, ranges as 2-4"))
            return
        if not self.on.isChecked():
            if st.cyclic is not None and st.cyclic.enabled:
                st.cyclic.enabled = False
                self.win.changed()
            return
        new = CyclicSymmetry(sectors=int(self.n.value()), left=self.left.currentText(), right=self.right.currentText(),
                             axis_point=[s.value() for s in self.pt], axis_dir=[s.value() for s in self.dir],
                             enabled=True, nodal_diameters=nds, expand=self.expand.isChecked())
        if st.cyclic != new:
            st.cyclic = new
            self.win.changed()

    def on_expand(self, on: bool):
        """Only the display of the results – the results stay valid."""
        if self._filling or self.win.study is None or self.win.study.cyclic is None:
            return
        self.win.study.cyclic.expand = bool(on)
        self.win.dirty = True
        self.win.show_result()

    def from_view(self, combo: QComboBox, label: str):
        sels = self.win.need_faces()
        if not sels:
            return
        name = self.win.new_selection(tr("Cyclic") + " " + tr(label).lower(), sels)
        self.win.view.set_selected_faces([])
        self._fill_combos(self.win.study)
        combo.setCurrentIndex(max(0, combo.findText(name)))
        self.win.changed()

    def detect(self):
        from fea_core.cyclic import CyclicError, detect
        st = self.win.study
        if st is None:
            return
        try:
            d = detect(st, self.left.currentText(), self.right.currentText())
        except CyclicError as e:
            self.note.setText(str(e))
            return
        self._filling = True
        try:
            self.n.setValue(int(d["sectors"]))
            for sp, v in zip(self.pt, d["axis_point"]):
                sp.setValue(float(v))
            for sp, v in zip(self.dir, d["axis_dir"]):
                sp.setValue(float(v))
            self.on.setChecked(True)
        finally:
            self._filling = False
        self.on_change()
        if not d["exact"]:
            self.note.setText(tr("The sector angle {0}° is not 360°/integer – N rounded to {1}").format(
                fmt(d["angle_deg"]), d["sectors"]))


# ------------------------------------------------------------------ 3D and results

def show_3d(win) -> None:
    """Constraints step: the two cut faces and the symmetry axis."""
    st = win.study
    c = getattr(st, "cyclic", None) if st is not None else None
    if c is None or not c.enabled:
        return
    v = win.view
    for name in (c.left, c.right):
        if name in st.selections:
            v.set_face_set(f"{KEY}{name}", win.sel_faces.get(name, []), C_CYCLIC, label=f"{name} (N = {c.sectors})")
    d = np.asarray(c.axis_dir, float)
    if np.linalg.norm(d) > 0:
        d = d / np.linalg.norm(d)
        p0 = np.asarray(c.axis_point, float)
        half = 0.6 * v.model_size()
        v.add_lines(f"{KEY}axis", [(p0 - half * d, p0 + half * d)], C_CYCLIC, 2.5,
                    label=tr("Cyclic symmetry axis"))


def full_results(win):
    """Results of the full 360° model for display (None = show the sector)."""
    res = win.results
    st = win.study
    if res is None or st is None or not (res.info or {}).get("cyclic"):
        return None
    if st.cyclic is not None and not st.cyclic.expand:
        return None
    try:
        from fea_core.cyclic.expand import expand_results
        return expand_results(res)
    except Exception as e:  # noqa: BLE001 – the sector is shown instead
        win.log_msg(f"Cyclic symmetry: the 360° model cannot be shown ({e})")
        return None
