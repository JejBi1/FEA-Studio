"""Shell model in the GUI (fea_core.shell): settings group in the Solve step (Study.shell – parts with thickness,
recognition of plate / sheet, plate parts from STEP, position of the surface) and display of the ShellResults results on the mid-surface
in the 3D window (contours by layer top / middle / bottom / max, membrane and bending stress, probe, reactions).
"""
from __future__ import annotations

import html

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QPushButton, QTableWidgetItem)

from fea_core.shell.model import OFFSETS, ShellSettings

from .widgets import Section, dspin, fmt, help_tip, table

OFFSET_LABELS = {"mid": "mid surface", "top": "top surface (shell below it)",
                 "bot": "bottom surface (shell above it)"}
KIND_LABELS = {"plate": "plate", "sheet": "sheet metal (with bends)", "surface": "surface part"}
# quantities of the shell result: (name, unit, field ShellResults.field, layer)
SHELL_Q = [("von Mises – max of surfaces", "MPa", "mises", "max"), ("von Mises – top surface", "MPa", "mises", "top"),
           ("von Mises – mid-surface", "MPa", "mises", "mid"), ("von Mises – bottom surface", "MPa", "mises", "bot"),
           ("Membrane stress (von Mises)", "MPa", "membrane", "max"),
           ("Bending stress (von Mises)", "MPa", "bending", "max"),
           ("1st principal stress S1 – max", "MPa", "s1", "max"), ("3rd principal stress S3 – min", "MPa", "s3", "max"),
           ("Stress σx – top surface", "MPa", "sxx", "top"), ("Stress σx – bottom surface", "MPa", "sxx", "bot"),
           ("Stress σy – top surface", "MPa", "syy", "top"), ("Stress σy – bottom surface", "MPa", "syy", "bot"),
           ("Total displacement", "mm", "u", "max"), ("Displacement X", "mm", "ux", "max"), ("Displacement Y", "mm", "uy", "max"),
           ("Displacement Z", "mm", "uz", "max"), ("Thickness", "mm", "thickness", "max")]


class ShellBox(Section):
    """Section "Shell model" (Study.shell); the tick in the header = calculate shells instead of 3D.
    2D section and shells exclude each other.
    """

    def __init__(self, win):
        on = QCheckBox("analyze the shell model instead of 3D")
        on.setToolTip("On = Run analysis computes shells (results “Shells: …”), no 3D mesh is created")
        super().__init__("Shell model (plates, sheet metal, surfaces)", check=on,
                         tip="Thin-walled parts as shells (mid-surface with thickness, S6 elements)",
                         help="Thin-walled parts (plates, sheet metal) are analyzed as shells – a mid-surface with thickness (S6 elements in CalculiX). Many fewer elements than a 3D mesh of a thin wall, stress on both the top and bottom surface. Surface parts from STEP (faces without volume) need a thickness.")
        self.on = on
        self.win = win
        self._filling = False
        self._detect: dict[str, tuple] = {}          # part → (SheetInfo | None, reason) – cache of the recognition
        self._surface_names: set[str] = set()
        self.setWhatsThis("<b>Shell model</b><p>Parts are converted to a mid-surface with thickness: a <i>plate</i> (two parallel planar faces), a <i>sheet metal part</i> with cylindrical bends, or a <i>surface part</i> from STEP (a face without volume – you enter the thickness). Plates joined in a T / L shape are connected by extending the mid-surface.</p><p>Supports (fixed, displacement) and loads (force, pressure, gravity, rotation) are transferred from 3D faces / edges. Bolts, welds, heat, 3D printing and orthotropy are not available in shells. Cannot be combined with a 2D section.</p>")
        f = QFormLayout(self.body)
        self.table = table(["Part", "Type", "Model thickness [mm]", "Entered t [mm]", "Note"],
                           placeholder="The model has no parts", rows=(3, 8))
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        help_tip(self.table, "Shell model parts: kind, thickness from geometry and entered thickness",
                 "Shell model parts: the detected kind (plate / sheet metal / surface part), thickness from the geometry and the entered thickness (empty = from geometry). Ticked = the part is in the shell model; an unrecognized part shows the reason. A click on a row loads the part into the form.")
        self.table.currentCellChanged.connect(lambda r, *_: self.on_row(r))
        self.table.itemChanged.connect(self.on_item)
        f.addRow(self.table)
        h = QHBoxLayout()
        self.t = dspin(0.0, 0.0, 1e4, 3, 0.5, " mm", special="from geometry")
        help_tip(self.t, "Shell thickness of the selected part [mm] (“from geometry” = detected)",
                 "Shell thickness of the selected part [mm]; “from geometry” = thickness of the detected plate / sheet (a surface part from STEP needs a thickness)")
        h.addWidget(self.t, 1)
        b = QPushButton("Set thickness")
        b.setToolTip("Save the thickness of the part selected in the table")
        b.clicked.connect(self.set_thickness)
        h.addWidget(b)
        f.addRow("Part thickness", h)
        self.offset = QComboBox()
        for k in OFFSETS:
            self.offset.addItem(OFFSET_LABELS[k], k)
        help_tip(self.offset, "Surface parts from STEP: the CAD surface is the mid, top or bottom surface",
                 "Surface parts from STEP: the CAD surface is the mid, top or bottom surface of the shell (for plates made from solids the mid-surface is always used)")
        self.offset.currentIndexChanged.connect(lambda _: self.on_change())
        f.addRow("Face position", self.offset)
        self.size = dspin(0.0, 0.0, 1e5, 3, 0.5, " mm", special="same as 3D mesh")
        self.size.setToolTip("Element size of the shell mesh [mm] (0 = global size from the Mesh step)")
        self.size.editingFinished.connect(self.on_change)
        f.addRow("Element size", self.size)
        h = QHBoxLayout()
        b = QPushButton("Detect plates")
        b.setToolTip("Re-detect plates and sheet metal in bodies (kind, thickness, reason for rejection)")
        b.clicked.connect(lambda: self.refresh(force=True))
        h.addWidget(b)
        b = QPushButton("Load surface parts from STEP")
        help_tip(b, "Add faces and shells without volume from STEP (shown in 3D)",
                 "Add the faces and shells without volume from STEP to the model (the normal loading skips them) – they are shown in 3D and you can select edges and faces on them for supports and loads")
        b.clicked.connect(self.load_surfaces)
        self.btn_surfaces = b
        h.addWidget(b)
        f.addRow(h)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setProperty("muted", True)
        f.addRow(self.note)
        self.on.toggled.connect(lambda _: self.on_change())

    # ------------------------------------------------------------------ obnova
    def detect(self, part):
        """(SheetInfo | None, reason) of the part – cache by the identity of the shape."""
        key = (part.name, id(part.shape))
        if key not in self._detect:
            if part.name in self.win.surface_names:
                self._detect[key] = (None, "surface part from STEP – enter the thickness")
            else:
                from fea_core.shell.geometry import detect_sheet
                try:
                    self._detect[key] = detect_sheet(part)
                except Exception as e:  # noqa: BLE001
                    self._detect[key] = (None, str(e))
        return self._detect[key]

    def refresh(self, force=False):
        st = self.win.study
        if st is None:
            return
        if force:
            self._detect = {}
        s = st.shell
        self._filling = True
        try:
            self.on.setChecked(bool(s.enabled))
            self.offset.setCurrentIndex(max(0, self.offset.findData(s.offset)))
            self.size.setValue(float(s.size or 0.0))
            excl = set(st.ignored_parts)
            parts = [p for p in self.win.parts if p.name not in excl]
            self.table.blockSignals(True)
            self.table.setRowCount(len(parts))
            for r, p in enumerate(parts):
                surf = p.name in self.win.surface_names
                if s.enabled or force or surf:
                    info, why = self.detect(p)
                else:
                    info, why = None, "(detected when enabled)"
                inside = (not s.parts) or p.name in s.parts
                kind = "surface part" if surf else KIND_LABELS.get(info.kind, info.kind) if info else "–"
                t_auto = fmt(info.thickness, 5) if info else "–"
                t_set = s.parts.get(p.name)
                note = (why if info is None else "; ".join(info.warnings) or
                        ("" if surf else f"mid-surface {fmt(info.mid_area, 5)} mm²"))
                vals = [p.name, kind, t_auto, fmt(t_set, 5) if t_set else ("–" if not surf else "enter a value!"), note]
                for c, txt in enumerate(vals):
                    it = QTableWidgetItem(txt)
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                    if c == 0:
                        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                        it.setCheckState(Qt.Checked if inside else Qt.Unchecked)
                        it.setData(Qt.UserRole, p.name)
                    if c in (2, 3):
                        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    if c == 4 and info is None and not surf and s.enabled:
                        from PySide6.QtGui import QBrush, QColor
                        it.setForeground(QBrush(QColor("#d23b3b")))
                    self.table.setItem(r, c, it)
            self.table.blockSignals(False)
        finally:
            self._filling = False
        twod = st.twod.enabled
        self.note.setText(("<span style='color:#d23b3b'>The 2D section analysis is enabled – shells and 2D are mutually exclusive (2D is being analyzed).</span> " if twod and s.enabled else "")
                          + ("Empty list of checked items = all parts. " if not s.parts else "")
                          + "Supports: clamp and displacement; loads: force, pressure, weight, rotation.")

    def settings_from_form(self) -> ShellSettings:
        st = self.win.study
        old = st.shell
        parts = {}
        all_on = True
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            name = it.data(Qt.UserRole)
            if it.checkState() == Qt.Checked:
                parts[name] = old.parts.get(name)
            else:
                all_on = False
        if all_on and all(v is None for v in parts.values()) and not old.parts:
            parts = {}                                   # all parts, thicknesses from the geometry
        return ShellSettings(self.on.isChecked(), parts, self.size.value() or None, self.offset.currentData(),
                             old.element)

    def on_change(self, *_):
        st = self.win.study
        if st is None or self._filling:
            return
        new = self.settings_from_form()
        if new.to_dict() == st.shell.to_dict():
            return
        if new.enabled and st.twod.enabled:
            QMessageBox.information(self, "Shells", "The 2D section analysis is enabled – shells and 2D are mutually exclusive; turning off the 2D analysis.")
            st.twod.enabled = False
        st.shell = new
        self.win.changed()

    def on_item(self, it):
        if self._filling or it.column() != 0:
            return
        self.on_change()

    def on_row(self, r):
        it = self.table.item(r, 0) if r >= 0 else None
        if it is None or self.win.study is None:
            return
        self.t.setValue(float(self.win.study.shell.parts.get(it.data(Qt.UserRole)) or 0.0))

    def set_thickness(self):
        r = self.table.currentRow()
        it = self.table.item(r, 0) if r >= 0 else None
        if it is None:
            QMessageBox.information(self, "Shells", "Select a part in the table.")
            return
        st = self.win.study
        name = it.data(Qt.UserRole)
        parts = dict(st.shell.parts) if st.shell.parts else {
            self.table.item(k, 0).data(Qt.UserRole): None for k in range(self.table.rowCount())}
        parts[name] = self.t.value() or None
        st.shell = ShellSettings(st.shell.enabled, parts, st.shell.size, st.shell.offset, st.shell.element)
        self.win.changed()

    def load_surfaces(self):
        n = self.win.load_surface_parts()
        if n == 0:
            QMessageBox.information(self, "Shells", "The STEP file contains no faces or shells without volume.")


def draw(win, rs, i, qi, n_bands):
    """Contours of the shell result on the mid-surface. Returns (title, bounds, colours, X, values, scale, U, field, layer)."""
    from .contour import band_edges
    name, unit, key, layer = SHELL_Q[min(qi, len(SHELL_Q) - 1)]
    st = rs.step(i)
    vals = np.asarray(rs.field(key, i, layer), float)
    U = np.asarray(st.U, float)
    scale = win.deformation_scale(U)
    X = rs.nodes + scale * U
    edges = band_edges(float(vals.min()), float(vals.max()), n_bands)
    colors = win.view.show_contour(X, rs.tri, vals, edges, f"{name} [{unit}]", edges_on=win.r_edges.isChecked(),
                                   show_scale=False)
    return f"{name} [{unit}]", edges, colors, X, vals, scale, U, key, layer


def info_html(win, rs, i) -> str:
    st = rs.step(i)
    sm = rs.summary()["steps"][i]
    rows = [f"<b>Shells: {html.escape(st.name)}</b> – {len(rs.nodes):,} nodes, {len(rs.tri):,} S6 elements".replace(",", " "),
            f"Max. displacement: <b>{fmt(sm['u_max'])} mm</b>",
            f"Max. von Mises: <b>{fmt(sm['mises_max'])} MPa</b> (top surface {fmt(sm['mises_top_max'])}, bottom {fmt(sm['mises_bot_max'])} MPa) at the point (" + ", ".join(fmt(float(c), 5) for c in sm["mises_max_at"]) + ") mm",
            f"Membrane stress max. {fmt(sm['membrane_max'])} MPa, bending stress max. {fmt(sm['bending_max'])} MPa"]
    rows.append("<br><b>Parts</b>: " + ", ".join(
        f"{html.escape(n)} t = {fmt(d['thickness'], 4)} mm" + (f" (offset {fmt(d['offset'], 3)}·t)" if d['offset'] else "")
        for n, d in rs.summary()["parts"].items()))
    if st.reactions:
        rows.append("<br><b>Reactions</b> [N, N·mm about the ref. point (" + "; ".join(fmt(float(c), 4) for c in rs.ref_point)
                    + ") mm]:")
        rows.append("<table cellspacing=3><tr><th align=left>selection</th>" + "".join(
            f"<th align=right>{c}</th>" for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")) + "</tr>")
        for k, r in st.reactions.items():
            rows.append(f"<tr><td>{html.escape(k)}</td>" + "".join(f"<td align=right>{fmt(float(x))}</td>" for x in r)
                        + "</tr>")
        rows.append("</table>")
    if "imbalance" in st.equilibrium:
        rows.append(f"Equilibrium check: imbalance {fmt(100 * float(st.equilibrium['imbalance']), 2)} %")
    rows.append("<i>Contours on the mid-surface; top / bottom surface stress according to the quantity layer. Bending = (top − bottom)/2, membrane = mid-surface.</i>")
    if rs.warnings:
        rows.append("<br><span style='color:#b36b00'>! " + "<br>! ".join(html.escape(w) for w in rs.warnings)
                    + "</span>")
    return win._join_rows(rows)
