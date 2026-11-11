"""Dialog for the selection by criteria (ANSYS Named Selection Worksheet, F3).

Allows finding and selecting faces (or edges) by:
- part (a specific part or all parts)
- face type (plane, cylinder, cone, sphere, torus, bspline)
- cylinder radius with tolerance
- direction of the plane normal
- coordinate planes and envelope limits (X, Y, Z)
- face area
Preview in the 3D window of the model and creation of a named selection (Study.selections).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QVBoxLayout)

from fea_core.i18n import tr
from fea_core.selectors import find_faces
from . import theme
from .widgets import dspin, fmt


class WorksheetDialog(QDialog):
    """Selection worksheet by criteria (Worksheet)."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle(tr("Selection by Criteria (Worksheet)"))
        self.resize(520, 560)
        lay = QVBoxLayout(self)

        intro = QLabel(tr(
            "Find model faces by geometric criteria (type, radius, normal, coordinate range, area). The result is displayed in 3D and saved as a named selection."
        ))
        intro.setWordWrap(True)
        lay.addWidget(intro)

        f = QFormLayout()

        # Name of the new selection
        self.name_edit = QLineEdit(self._default_name())
        self.name_edit.setToolTip(tr("Name of the created selection"))
        f.addRow(tr("Selection name"), self.name_edit)

        # Part
        self.part_combo = QComboBox()
        self.part_combo.addItem(tr("(all parts)"), None)
        if hasattr(win, "parts") and win.parts:
            for p in win.parts:
                self.part_combo.addItem(p.name, p.name)
        self.part_combo.setToolTip(tr("Search faces only in the selected part or throughout the assembly"))
        f.addRow(tr("Part"), self.part_combo)

        # Face type
        self.type_combo = QComboBox()
        types = [
            (tr("(all types)"), None),
            (tr("plane"), "plane"),
            (tr("cylinder"), "cylinder"),
            (tr("cone"), "cone"),
            (tr("sphere"), "sphere"),
            (tr("torus"), "torus"),
            (tr("freeform face (bspline)"), "bspline"),
        ]
        for label, val in types:
            self.type_combo.addItem(label, val)
        self.type_combo.setToolTip(tr("Geometric face type in CAD model"))
        f.addRow(tr("Face type"), self.type_combo)

        # Radius limit
        self.chk_radius = QCheckBox(tr("Limit cylinder radius"))
        self.chk_radius.setToolTip(tr("Applies to cylindrical faces (e.g. holes of given radius)"))
        self.rad_val = dspin(10.0, 0.001, 1e5, 3, 1.0, " mm")
        self.rad_val.setToolTip(tr("Target cylindrical face radius [mm]"))
        self.rad_tol = dspin(0.1, 1e-4, 100.0, 3, 0.1, " mm")
        self.rad_tol.setToolTip(tr("Cylindrical face radius tolerance [mm]"))
        hr = QHBoxLayout()
        hr.addWidget(self.rad_val)
        hr.addWidget(QLabel(tr("± tol.")))
        hr.addWidget(self.rad_tol)
        f.addRow(self.chk_radius, hr)

        # Normal limit
        self.chk_normal = QCheckBox(tr("Limit normal"))
        self.chk_normal.setToolTip(tr("Outer normal of planar face (e.g. top face: nx=0, ny=0, nz=1)"))
        self.nx = dspin(0.0, -1.0, 1.0, 3, 1.0)
        self.nx.setToolTip(tr("X component of planar face normal vector"))
        self.ny = dspin(0.0, -1.0, 1.0, 3, 1.0)
        self.ny.setToolTip(tr("Y component of planar face normal vector"))
        self.nz = dspin(1.0, -1.0, 1.0, 3, 1.0)
        self.nz.setToolTip(tr("Z component of planar face normal vector"))
        hn = QHBoxLayout()
        for s in (self.nx, self.ny, self.nz):
            hn.addWidget(s)
        f.addRow(self.chk_normal, hn)

        # Coordinate limits X, Y, Z
        self.chk_x = QCheckBox(tr("Limit X [mm]"))
        self.chk_x.setToolTip(tr("Filter faces by body X coordinate range"))
        self.x_min = dspin(-1000.0, -1e6, 1e6, 2, 10.0)
        self.x_min.setToolTip(tr("Minimum X coordinate of face bounding box [mm]"))
        self.x_max = dspin(1000.0, -1e6, 1e6, 2, 10.0)
        self.x_max.setToolTip(tr("Maximum X coordinate of face bounding box [mm]"))
        hx = QHBoxLayout()
        hx.addWidget(QLabel(tr("od"))); hx.addWidget(self.x_min)
        hx.addWidget(QLabel(tr("do"))); hx.addWidget(self.x_max)
        f.addRow(self.chk_x, hx)

        self.chk_y = QCheckBox(tr("Limit Y [mm]"))
        self.chk_y.setToolTip(tr("Filter faces by body Y coordinate range"))
        self.y_min = dspin(-1000.0, -1e6, 1e6, 2, 10.0)
        self.y_min.setToolTip(tr("Minimum Y coordinate of face bounding box [mm]"))
        self.y_max = dspin(1000.0, -1e6, 1e6, 2, 10.0)
        self.y_max.setToolTip(tr("Maximum Y coordinate of face bounding box [mm]"))
        hy = QHBoxLayout()
        hy.addWidget(QLabel(tr("od"))); hy.addWidget(self.y_min)
        hy.addWidget(QLabel(tr("do"))); hy.addWidget(self.y_max)
        f.addRow(self.chk_y, hy)

        self.chk_z = QCheckBox(tr("Limit Z [mm]"))
        self.chk_z.setToolTip(tr("Filter faces by body Z coordinate range"))
        self.z_min = dspin(-1000.0, -1e6, 1e6, 2, 10.0)
        self.z_min.setToolTip(tr("Minimum Z coordinate of face bounding box [mm]"))
        self.z_max = dspin(1000.0, -1e6, 1e6, 2, 10.0)
        self.z_max.setToolTip(tr("Maximum Z coordinate of face bounding box [mm]"))
        hz = QHBoxLayout()
        hz.addWidget(QLabel(tr("od"))); hz.addWidget(self.z_min)
        hz.addWidget(QLabel(tr("do"))); hz.addWidget(self.z_max)
        f.addRow(self.chk_z, hz)

        # Area limit
        self.chk_area = QCheckBox(tr("Limit area A [mm²]"))
        self.chk_area.setToolTip(tr("Filter faces by surface area"))
        self.area_min = dspin(0.0, 0.0, 1e9, 2, 10.0)
        self.area_min.setToolTip(tr("Minimum face area [mm²]"))
        self.area_max = dspin(1e6, 0.0, 1e9, 2, 10.0)
        self.area_max.setToolTip(tr("Maximum face area [mm²]"))
        ha = QHBoxLayout()
        ha.addWidget(QLabel(tr("od"))); ha.addWidget(self.area_min)
        ha.addWidget(QLabel(tr("do"))); ha.addWidget(self.area_max)
        f.addRow(self.chk_area, ha)

        lay.addLayout(f)

        # Search state
        self.lbl_status = QLabel(tr("Enter criteria and click Preview"))
        self.lbl_status.setProperty("muted", True)
        lay.addWidget(self.lbl_status)

        # Action buttons
        hb = QHBoxLayout()
        b_find = QPushButton(tr("Preview / Search"))
        b_find.setToolTip(tr("Find faces matching criteria and highlight them in 3D"))
        b_find.clicked.connect(self.find_matching)
        hb.addWidget(b_find)

        b_create = QPushButton(tr("Create selection"))
        theme.set_primary(b_create)
        b_create.setToolTip(tr("Save found faces as named study selection"))
        b_create.clicked.connect(self.create_selection)
        hb.addWidget(b_create)

        b_close = QPushButton(tr("Close"))
        b_close.setToolTip(tr("Close dialog without saving selection"))
        b_close.clicked.connect(self.reject)
        hb.addWidget(b_close)

        lay.addLayout(hb)

        self.found_selectors = []
        self.found_tuples = []

    def _default_name(self) -> str:
        sels = getattr(self.win.study, "selections", {}) if getattr(self.win, "study", None) else {}
        k = 1
        prefix = tr("Criteria selection")
        while f"{prefix} {k}" in sels:
            k += 1
        return f"{prefix} {k}"

    def find_matching(self):
        """Finds the faces and highlights them in the 3D window."""
        if not getattr(self.win, "parts", None):
            self.lbl_status.setText(tr("No geometry is loaded"))
            self.lbl_status.setProperty("muted", True)
            self.lbl_status.style().unpolish(self.lbl_status)
            self.lbl_status.style().polish(self.lbl_status)
            return

        part_filter = self.part_combo.currentData()
        type_filter = self.type_combo.currentData()

        # Built where filter
        def where_fn(props: dict) -> bool:
            # Radius
            if self.chk_radius.isChecked():
                r = props.get("radius")
                if r is None or abs(r - self.rad_val.value()) > self.rad_tol.value():
                    return False

            # Normal
            if self.chk_normal.isChecked():
                n_req = np.array([self.nx.value(), self.ny.value(), self.nz.value()], float)
                norm_len = np.linalg.norm(n_req)
                if norm_len > 1e-6:
                    n_req = n_req / norm_len
                    n_act = np.asarray(props.get("normal", [0, 0, 1]), float)
                    if np.dot(n_req, n_act) < 0.99:
                        return False

            # Coordinates
            bb = props.get("bbox")
            if bb:
                # bb = [xmin, ymin, zmin, xmax, ymax, zmax]
                if self.chk_x.isChecked():
                    if bb[3] < self.x_min.value() or bb[0] > self.x_max.value():
                        return False
                if self.chk_y.isChecked():
                    if bb[4] < self.y_min.value() or bb[1] > self.y_max.value():
                        return False
                if self.chk_z.isChecked():
                    if bb[5] < self.z_min.value() or bb[2] > self.z_max.value():
                        return False

            # Obsah
            if self.chk_area.isChecked():
                a = props.get("area", 0.0)
                if a < self.area_min.value() or a > self.area_max.value():
                    return False

            return True

        rad_arg = self.rad_val.value() if self.chk_radius.isChecked() else None

        self.found_selectors = find_faces(
            self.win.parts,
            part=part_filter,
            type=type_filter,
            radius=rad_arg,
            where=where_fn
        )

        # Match selectors with the 3D indices (part_idx, face_idx) for highlighting
        tuples = []
        from fea_core.selectors import match_score
        for pi, p in enumerate(self.win.parts):
            if part_filter is not None and p.name != part_filter:
                continue
            cand_props = [self.win.face_prop(pi, fi) for fi in range(len(list(p.faces())))]
            for sel in self.found_selectors:
                if sel.part != p.name:
                    continue
                sd = sel.to_dict()
                best = None
                best_score = 1e9
                for fi, cp in enumerate(cand_props):
                    sc = match_score(sd, cp)
                    if sc < best_score:
                        best_score = sc
                        best = fi
                if best is not None and best_score < 0.05:
                    tuples.append((pi, best))

        self.found_tuples = tuples
        self.win.view.set_selected_faces(tuples)

        n = len(self.found_selectors)
        tot_area = sum(s.area for s in self.found_selectors)
        self.lbl_status.setText(tr(f"Found {n} faces, total area A = {fmt(tot_area)} mm²"))
        self.lbl_status.setProperty("muted", False)
        self.lbl_status.style().unpolish(self.lbl_status)
        self.lbl_status.style().polish(self.lbl_status)

    def create_selection(self):
        """Creates a named selection in the model."""
        if not self.found_selectors:
            self.find_matching()
        if not self.found_selectors:
            QMessageBox.information(self, tr("Information"), tr("No faces matching the specified criteria were found."))
            return

        name = self.name_edit.text().strip() or self._default_name()
        self.win.study.add_selection(name, self.found_selectors)
        self.win.sel_faces[name] = self.found_tuples
        self.win.changed()
        if hasattr(self.win, "refresh_sets"):
            self.win.refresh_sets()
        self.accept()
