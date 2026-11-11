"""Dialog for local coordinate systems of the loads (Study.coordinate_systems): Cartesian / cylindrical, origin and axes,
taking over from a selection (cylindrical face or circle → axis of the cylinder, planar face → normal as the z axis).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QMessageBox, QPushButton, QVBoxLayout)

from fea_core.csys import check, describe
from fea_core.i18n import tr
from fea_core.study import CoordinateSystem

from .widgets import dspin

KINDS = [("cartesian", "Cartesian (x, y, z)"), ("cylindrical", "cylindrical (r, θ, z)")]


class CsysDialog(QDialog):
    """List of the systems at the top left, the form below it. Changes go straight into the study (win.changed())."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Local coordinate systems")
        self.resize(560, 520)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Coordinate systems for load components. Cylindrical: components (r, θ, z) – radial from the axis,\ntangential (counterclockwise about the z axis), along the axis; the direction varies over the face / edge.\nThe system also defines axes 1, 2, 3 of an orthotropic material (Materials step)."))
        self.lst = QListWidget()
        self.lst.setToolTip("Study coordinate systems. Click = load into the form (edit → “Add / save”).\nThe system is chosen for force, remote force, bearing load and gravity (cylindrical only for force).")
        self.lst.currentRowChanged.connect(self.on_row)
        lay.addWidget(self.lst)
        f = QFormLayout()
        self.name = QLineEdit()
        self.name.setToolTip("System name – loads refer to it; saving with the same name overwrites the system")
        f.addRow("Name", self.name)
        self.kind = QComboBox()
        for k, t in KINDS:
            self.kind.addItem(t, k)
        self.kind.setToolTip("Cartesian: components along the system's x, y, z axes.\nCylindrical: components (r, θ, z) – radial from the z axis, tangential (counterclockwise about z), along the axis;\nthe direction varies point to point. Only for forces on faces, edges, and points.")
        f.addRow("Type", self.kind)
        self.vecs = {}
        tips = {"origin": "System origin – {} coordinate [mm] in the global system",
                "z_axis": "z axis (for cylindrical, the cylinder axis) – {} component of the direction vector (length is irrelevant)",
                "x_axis": "x axis (for cylindrical, direction θ = 0) – {} component of the direction vector;\northogonality to the z axis is computed automatically (must not be parallel to the z axis)"}
        for key, label, val in (("origin", "Origin [mm]", (0, 0, 0)), ("z_axis", "z axis (cylinder axis)", (0, 0, 1)),
                                ("x_axis", "x axis (θ = 0)", (1, 0, 0))):
            h = QHBoxLayout()
            sp = [dspin(v, -1e7, 1e7, 4) for v in val]
            for s, comp in zip(sp, "xyz"):
                s.setToolTip(tr(tips[key]).format(comp))      # the catalogue has a template with "{}"
            for s in sp:
                h.addWidget(s)
            self.vecs[key] = sp
            f.addRow(label, h)
        lay.addLayout(f)
        h = QHBoxLayout()
        for txt, slot, tip in (
                ("From selection", self.from_selection, "Cylindrical face or circle → origin and z axis = cylinder axis;\nplanar face → origin at centroid, z axis = normal"),
                ("Add / save", self.save, "Save system (same name = overwrite)"),
                ("Delete", self.delete, "Delete the selected system (must not be used by any load)")):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            h.addWidget(b)
        lay.addLayout(h)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.button(QDialogButtonBox.Close).setToolTip("Close – saved systems are already in the study; unsaved form edits are discarded")
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.refresh()

    def refresh(self, select: str | None = None):
        st = self.win.study
        self.lst.blockSignals(True)
        self.lst.clear()
        for c in st.coordinate_systems.values():
            self.lst.addItem(describe(c))
        self.lst.blockSignals(False)
        names = list(st.coordinate_systems)
        if select in names:
            self.lst.setCurrentRow(names.index(select))
        elif names:
            self.lst.setCurrentRow(0)
        else:
            self.name.setText(self._free_name())

    def _free_name(self) -> str:
        k = 1
        while f"System {k}" in self.win.study.coordinate_systems:
            k += 1
        return f"System {k}"

    def on_row(self, row):
        names = list(self.win.study.coordinate_systems)
        if not 0 <= row < len(names):
            return
        c = self.win.study.coordinate_systems[names[row]]
        self.name.setText(c.name)
        self.kind.setCurrentIndex(max(self.kind.findData(c.kind), 0))
        for key in ("origin", "z_axis", "x_axis"):
            for s, v in zip(self.vecs[key], getattr(c, key)):
                s.setValue(float(v))

    def current(self) -> CoordinateSystem:
        return CoordinateSystem(self.name.text().strip(), self.kind.currentData(),
                                *[[s.value() for s in self.vecs[k]] for k in ("origin", "z_axis", "x_axis")])

    def from_selection(self):
        """Axis from the selection: circle (Edge filter) / cylindrical face → axis of the cylinder; planar face → normal."""
        from fea_core.geometry import cylinder_axis
        w = self.win
        o = z = None
        cyl = False
        edge = w.view.selected_edge()
        if edge is not None and "center" in edge and "axis" in edge:
            o, z, cyl = np.asarray(edge["center"], float), np.asarray(edge["axis"], float), True
        else:
            for pi, fi in w.view.selected_faces():
                ax = cylinder_axis(w.view.part_faces[pi][fi])
                if ax is not None:
                    o, z, cyl = ax[0], ax[1], True
                    break
                p = w.face_prop(pi, fi)
                if p["type"] == "plane":
                    o, z = np.asarray(p["centroid"], float), np.asarray(p["normal"], float)
                    break
        if o is None:
            QMessageBox.information(self, "Coordinate system", "Select a cylindrical face, a circle (Edge filter) or a planar face.")
            return
        for s, v in zip(self.vecs["origin"], o):
            s.setValue(float(v) + 0.0)
        for s, v in zip(self.vecs["z_axis"], z):
            s.setValue(round(float(v), 9) + 0.0)
        x = np.cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0])
        x = np.cross(x, z)
        for s, v in zip(self.vecs["x_axis"], x / np.linalg.norm(x)):
            s.setValue(round(float(v), 9) + 0.0)
        if cyl:
            self.kind.setCurrentIndex(self.kind.findData("cylindrical"))

    def save(self):
        c = self.current()
        if not c.name:
            QMessageBox.warning(self, "Coordinate system", "Enter a name.")
            return
        errs = check(c)
        if errs:
            QMessageBox.warning(self, "Coordinate system", errs[0])
            return
        st = self.win.study
        users = [l for lc in st.load_cases for l in lc.loads if getattr(l, "csys", None) == c.name]
        if c.kind == "cylindrical" and any(l.kind != "force" for l in users):
            QMessageBox.warning(self, "Coordinate system", "The system is also used by a load other than force – it must stay Cartesian.")
            return
        st.coordinate_systems[c.name] = c
        self.win.changed()
        self.refresh(c.name)
        self.info.setText(f"Saved: {describe(c)}")

    def delete(self):
        st = self.win.study
        names = list(st.coordinate_systems)
        row = self.lst.currentRow()
        if not 0 <= row < len(names):
            return
        name = names[row]
        users = [f"{getattr(l, 'selection', l.kind)} ({lc.name})" for lc in st.load_cases for l in lc.loads
                 if getattr(l, "csys", None) == name]
        if users:
            QMessageBox.warning(self, "Coordinate system", f"System '{name}' is used by load {users[0]} – change or delete the load first.")
            return
        mats = [m.name for m in st.materials.values() if getattr(m, "ortho_csys", None) == name]
        if mats:
            QMessageBox.warning(self, "Coordinate system", f"System '{name}' defines the axes of orthotropic material '{mats[0]}' – change its axes first (Materials step).")
            return
        del st.coordinate_systems[name]
        self.win.changed()
        self.refresh()
