"""2D calculation by a section in the GUI (fea_core.twod): settings group in the Solve step (Study.twod), preview of the section plane in 3D
and display of the Results2D results in the 3D window (2D mesh placed in the section plane, contours, probe, reactions).
"""
from __future__ import annotations

import html

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QListWidget,
                               QMessageBox, QPushButton, QVBoxLayout, QWidget)

from fea_core.i18n import tr
from fea_core.twod.model import KIND_LABEL, TwoDSettings, make_frame

from .widgets import Section, check_item, checked, dspin, fmt, help_tip

C_PLANE = (230, 120, 20)
KEY = "twod:plane"
# quantities of the 2D result: (name, unit, field Results2D.field)
AXI_Q = [("von Mises", "MPa", "mises"), ("σθ hoop", "MPa", "st"), ("σr radial", "MPa", "sr"),
         ("σz axial", "MPa", "sa"), ("τrz", "MPa", "srz"), ("1st principal stress S1", "MPa", "s1"),
         ("3rd principal stress S3", "MPa", "s3"), ("Tresca", "MPa", "tresca"), ("Total displacement", "mm", "umag"),
         ("Displacement ur (radial)", "mm", "ur"), ("Displacement uz (axial)", "mm", "ua")]
PLANE_Q = [("von Mises", "MPa", "mises"), ("Stress σx", "MPa", "sx"), ("Stress σy", "MPa", "sy"),
           ("Stress σz", "MPa", "sz"), ("Stress τxy", "MPa", "sxy"), ("1st principal stress S1", "MPa", "s1"),
           ("3rd principal stress S3", "MPa", "s3"), ("Tresca", "MPa", "tresca"), ("Total displacement", "mm", "umag"),
           ("Displacement ux", "mm", "ux"), ("Displacement uy", "mm", "uy")]


def quantities(r2) -> list[tuple[str, str, str]]:
    return AXI_Q if r2.axisym else PLANE_Q


class TwoDBox(Section):
    """Section "2D section calculation" (Study.twod); the tick in the header = calculate 2D instead of 3D."""

    def __init__(self, win):
        on = QCheckBox("analyze in 2D instead of 3D")
        on.setToolTip("On = Run analysis solves the 2D section (results “2D: …”); the 3D mesh is not generated")
        super().__init__("2D section analysis (axisymmetric / plane problem)", check=on,
                         tip="Instead of the 3D model, a 2D section of the parts is analyzed (CAX6 / CPS6 / CPE6)",
                         help="Instead of the 3D model, a 2D section of the parts by a plane is analyzed: an axisymmetric problem (CAX6 – pipes, vessels, flanges), plane stress (CPS6 – thin plates) or plane strain (CPE6 – long bodies). Supports and loads of the 3D faces are converted to the edges of the section. Fast and accurate for a fine mesh.")
        self.on = on
        self.win = win
        self._filling = False
        self.setWhatsThis("<b>2D section analysis</b><p>Parts are cut by a plane and a 2D problem is solved: <i>axisymmetric</i> (the half-plane contains the axis of rotation, x = radius, loads and reactions for the full circumference), <i>plane stress</i> (thin plate of thickness t) or <i>plane strain</i> (long body). Supports, pressure, force, gravity and rotation are transferred from 3D faces to the section edges.</p><p>For sectors (a model of only part of the circumference with symmetry), the section half-plane must lie <b>inside</b> the sector – away from the symmetry planes, e.g. direction r = (1; 1; 0) for a quarter.</p>")
        f = QFormLayout(self.body)
        self.kind = QComboBox()
        for k, t in KIND_LABEL.items():
            self.kind.addItem(t, k)
        help_tip(self.kind, "Axisymmetric, plane stress (thin plate) or plane strain",
                 "Axisymmetric problem (axis of rotation), plane stress (thin plate, σz = 0) or plane strain (long body, εz = 0)")
        f.addRow("Problem type", self.kind)

        def vec(vals, tips):
            h = QHBoxLayout()
            sp = [dspin(v, -1e7, 1e7, 4) for v in vals]
            for s, ax in zip(sp, "xyz"):
                s.setToolTip(tr(tips).format(ax))      # the catalogue has a template with "{}"
                h.addWidget(s)
            return sp, h
        self.origin, h = vec((0, 0, 0), "{} coordinate [mm] of a point on the section plane (for axisymmetric, a point on the axis of rotation)")
        b = QPushButton("Cylinder z axis")
        b.setToolTip("Point and axis from the selected cylindrical face (pipe, flange) – the axis of rotation of the axisymmetric model")
        b.clicked.connect(self.axis_from_selection)
        h.addWidget(b)
        self.win.register_point_input(self.origin)
        f.addRow("Point [mm]", h)
        self.axis, h = vec((0, 0, 1), "{} component: for axisymmetric, the direction of the axis of rotation; for a plane problem, the normal of the section plane")
        self.axis_lbl = QLabel("Rotation axis")
        f.addRow(self.axis_lbl, h)
        self.xdir, h = vec((1, 0, 0), "{} component of the 2D x axis direction: for axisymmetric, the radial direction of the section half-plane (for sectors\nINSIDE the sector, away from symmetry planes – e.g. (1; 1; 0)); for a plane problem, the x axis of the section")
        f.addRow("x direction (radial)", h)
        self.thick = dspin(1.0, 1e-6, 1e6, 3, 1.0, " mm")
        self.thick.setToolTip("Thickness of the plane problem [mm] – forces and reactions are for the full thickness")
        self.thick_lbl = QLabel("Thickness")
        f.addRow(self.thick_lbl, self.thick)
        self.size = dspin(0.0, 0.0, 1e5, 3, 0.5, " mm", special="same as 3D mesh")
        help_tip(self.size, "2D mesh element size [mm] (0 = global size of the Mesh step)",
                 "2D mesh element size [mm] (0 = global element size of the Mesh step). The 2D mesh can be much finer than the 3D one.")
        f.addRow("Element size", self.size)
        self.parts = QListWidget()
        self.parts.setMaximumHeight(80)
        self.parts.setToolTip("Checked parts are sectioned (none checked = all meshed parts)")
        self.parts.itemChanged.connect(lambda _: self.on_change())
        f.addRow("Parts", self.parts)
        self.preview = QCheckBox("preview section plane in 3D")
        self.preview.setChecked(True)
        self.preview.setToolTip("The section plane (half-plane) and axis are shown in orange in the 3D view of the Solve step")
        self.preview.toggled.connect(lambda _: self.win.refresh_sets())
        f.addRow(self.preview)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setProperty("muted", True)
        f.addRow(self.note)
        for w in self.origin + self.axis + self.xdir:
            w.editingFinished.connect(self.on_change)
        for w in (self.thick, self.size):
            w.editingFinished.connect(self.on_change)
        self.on.toggled.connect(lambda _: self.on_change())
        self.kind.currentIndexChanged.connect(lambda _: self.on_change())

    def refresh(self):
        st = self.win.study
        if st is None:
            return
        s = st.twod
        self._filling = True
        try:
            self.on.setChecked(bool(s.enabled))
            self.kind.setCurrentIndex(max(0, self.kind.findData(s.kind)))
            for sp, v in zip(self.origin, s.origin):
                sp.setValue(float(v))
            for sp, v in zip(self.axis, s.axis):
                sp.setValue(float(v))
            for sp, v in zip(self.xdir, s.xdir):
                sp.setValue(float(v))
            self.thick.setValue(float(s.thickness))
            self.size.setValue(float(s.size or 0.0))
            self.parts.clear()
            excl = st.excluded_parts()
            for p in self.win.parts:
                if p.name not in excl:
                    check_item(self.parts, p.name, p.name in s.parts)
        finally:
            self._filling = False
        self._sync()

    def _sync(self):
        axi = self.kind.currentData() == "axisym"
        self.axis_lbl.setText("Rotation axis" if axi else "Plane normal")
        for w in (self.thick, self.thick_lbl):
            w.setVisible(not axi)
        on = self.on.isChecked()
        for w in [self.kind, self.thick, self.size, self.parts, self.preview] + self.origin + self.axis + self.xdir:
            w.setEnabled(on)
        try:
            fr = make_frame(self.settings())
            self.note.setText(("Half-plane of the section: r along (" + "; ".join(fmt(float(c), 3) for c in fr.e1) + "), z axis ("
                               + "; ".join(fmt(float(c), 3) for c in fr.e2) + ")") if axi else
                              ("Section plane: x (" + "; ".join(fmt(float(c), 3) for c in fr.e1) + "), y ("
                               + "; ".join(fmt(float(c), 3) for c in fr.e2) + ")"))
        except Exception as e:  # noqa: BLE001
            self.note.setText(f"<span style='color:#d23b3b'>{html.escape(str(e))}</span>")

    def settings(self) -> TwoDSettings:
        return TwoDSettings(self.on.isChecked(), self.kind.currentData(), [s.value() for s in self.origin],
                            [s.value() for s in self.axis], [s.value() for s in self.xdir], self.thick.value(),
                            checked(self.parts), self.size.value() or None)

    def on_change(self, *_):
        self._sync()
        st = self.win.study
        if st is None or self._filling:
            return
        new = self.settings()
        if new.to_dict() == st.twod.to_dict():
            return
        if new.enabled and st.shell.enabled:            # 2D and shells exclude each other
            from fea_core.shell.model import ShellSettings
            QMessageBox.information(self, "2D", "The shell model is enabled – 2D and shells are mutually exclusive; turning off the shell model.")
            st.shell = ShellSettings(False, st.shell.parts, st.shell.size, st.shell.offset, st.shell.element)
        st.twod = new
        self.win.changed()

    def axis_from_selection(self):
        from fea_core.geometry import cylinder_axis
        for pi, fi in self.win.view.selected_faces():
            ax = cylinder_axis(self.win.view.part_faces[pi][fi])
            if ax is not None:
                for sp, v in zip(self.origin, ax[0]):
                    sp.setValue(float(v) + 0.0)
                for sp, v in zip(self.axis, ax[1]):
                    sp.setValue(round(float(v), 9) + 0.0)
                self.kind.setCurrentIndex(self.kind.findData("axisym"))
                self.on_change()
                return
        QMessageBox.information(self, "2D", "Select a cylindrical face in 3D (Face or Hole filter).")

    def draw_preview(self):
        """Section plane / half-plane and axis into the 3D window sets (Solve step)."""
        win = self.win
        st = win.study
        if st is None or not st.twod.enabled or not self.preview.isChecked():
            return
        try:
            fr = make_frame(st.twod)
        except Exception:  # noqa: BLE001
            return
        Lm = 0.6 * win.view.model_size()
        o = fr.origin
        if fr.axisym:
            q = [o - Lm * fr.e2, o - Lm * fr.e2 + Lm * fr.e1, o + Lm * fr.e2 + Lm * fr.e1, o + Lm * fr.e2]
            seg = list(zip(q, q[1:]))
            win.view.add_lines(KEY, [(o - 1.2 * Lm * fr.e2, o + 1.2 * Lm * fr.e2)], (90, 90, 90), 2.0, label="axis")
            win.view.add_lines(KEY, seg, C_PLANE, 2.5, label="section half-plane (r ≥ 0)")
        else:
            q = [o + a * Lm * fr.e1 + b * Lm * fr.e2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
            win.view.add_lines(KEY, list(zip(q, q[1:])), C_PLANE, 2.5, label="section plane")
        win.view.add_arrows(KEY, [(o + 0.5 * Lm * fr.e1, fr.e1, 0.4 * Lm)], C_PLANE)


def draw(win, r2, i, qi, n_bands):
    """Contours of the 2D result in the section plane in the 3D window. Returns (title, bounds, colours, X3, values, scale, U3)."""
    from .contour import band_edges
    name, unit, key = quantities(r2)[min(qi, len(quantities(r2)) - 1)]
    fr = r2.frame2d()
    vals = np.asarray(r2.field(key, i), float)
    U2 = np.asarray(r2.step(i)["U"], float)
    U3 = fr.vec3d(U2)
    scale = win.deformation_scale(U3)
    X3 = fr.to3d(r2.nodes + scale * U2)
    edges = band_edges(float(vals.min()), float(vals.max()), n_bands)
    colors = win.view.show_contour(X3, r2.tri, vals, edges, f"{name} [{unit}]", edges_on=win.r_edges.isChecked(),
                                   show_scale=False)
    return f"2D {name} [{unit}]", edges, colors, X3, vals, scale, U3, key


def info_html(win, r2, i, key) -> str:
    s = r2.step(i)
    fr = r2.frame2d()
    ax = ("r", "z") if r2.axisym else ("x", "y")
    vm = r2.mises(i)
    um = np.linalg.norm(s["U"], axis=1)
    a, b = int(np.argmax(vm)), int(np.argmax(um))
    rows = [f"<b>2D: {html.escape(s['name'])}</b> – {KIND_LABEL[r2.kind]}"
            + ("" if r2.axisym else f", thickness {fmt(r2.thickness)} mm"),
            f"2D mesh: {len(r2.nodes):,} nodes, {len(r2.tri):,} tri6 elements".replace(",", " "),
            f"Max. von Mises: <b>{fmt(float(vm[a]))} MPa</b> at point {ax[0]} = {fmt(float(r2.nodes[a, 0]), 5)}, {ax[1]} = {fmt(float(r2.nodes[a, 1]), 5)} mm",
            f"Max. displacement: <b>{fmt(float(um[b]))} mm</b>"]
    if s.get("reactions"):
        rows.append("<br><b>Reakce</b> [N]" + (" – over the full circumference; the radial sum is not a meaningful resultant"
                                               if r2.axisym else " – over the full thickness") + ":")
        for k, r in s["reactions"].items():
            rows.append(f"{html.escape(k)}: F{ax[0]} {fmt(float(r[0]))}, F{ax[1]} {fmt(float(r[1]))}")
    eq = s.get("equilibrium") or {}
    if "imbalance" in eq:
        rows.append(f"Equilibrium check: imbalance {fmt(100 * float(eq['imbalance']), 2)} %")
    rows.append(f"<i>2D points: {ax[0]} along (" + "; ".join(fmt(float(c), 3) for c in fr.e1) + f"), {ax[1]} along ("
                + "; ".join(fmt(float(c), 3) for c in fr.e2) + ") from the point (" + "; ".join(fmt(float(c), 4)
                                                                          for c in fr.origin) + ")</i>")
    if r2.warnings:
        rows.append("<br><span style='color:#b36b00'>! " + "<br>! ".join(html.escape(w) for w in r2.warnings)
                    + "</span>")
    return "<br>".join(rows)
