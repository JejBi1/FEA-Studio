"""Dialog for the force by a cut (Free Body Cut, A3).

Allows defining the cutting plane (point, normal), choosing the load state and the parts, computing the internal forces and moments N, Vy, Vz, Mx, My, Mz about the centroid of the cut and showing them
in a clear table with the option to copy it to the clipboard. The calculation runs in the background (`TaskThread`).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from fea_core.i18n import tr
from fea_core.section_forces import section_forces
from . import theme
from .widgets import dspin, fmt
from .workers import TaskThread


class SectionForceDialog(QDialog):
    """Dialog for the force by a cut (Free Body Cut)."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle(tr("Section force (Free Body Cut)"))
        self.resize(600, 700)
        lay = QVBoxLayout(self)
        self._thread: TaskThread | None = None        # calculation in the background
        self.last_result = None

        intro = QLabel(tr(
            "Computes the resultant internal forces and moments in a cut through the body by a plane (point, normal). Components N, Vy, Vz, Mx, My, Mz refer to the centroid of the cut in the local cut coordinate system (x axis = cut normal n)."
        ))
        intro.setWordWrap(True)
        lay.addWidget(intro)

        # ---------------------------------------------------------- Cutting plane
        g_plane = QGroupBox(tr("Section plane"))
        fp = QFormLayout(g_plane)

        # Point on the plane
        self.px = dspin(0.0, -1e6, 1e6, 3, 10.0)
        self.py = dspin(0.0, -1e6, 1e6, 3, 10.0)
        self.pz = dspin(0.0, -1e6, 1e6, 3, 10.0)
        hp = QHBoxLayout()
        for s, a in ((self.px, "X"), (self.py, "Y"), (self.pz, "Z")):
            s.setToolTip(tr(f"{a} coordinate of a point lying in the cut plane [mm] (global coordinate system)"))
            hp.addWidget(s)
        fp.addRow(tr("Cut point [X, Y, Z] mm"), hp)

        # Normal
        self.nx = dspin(1.0, -1.0, 1.0, 4, 1.0)
        self.ny = dspin(0.0, -1.0, 1.0, 4, 1.0)
        self.nz = dspin(0.0, -1.0, 1.0, 4, 1.0)
        hn = QHBoxLayout()
        for s, a in ((self.nx, "x"), (self.ny, "y"), (self.nz, "z")):
            s.setToolTip(tr(f"{a} component of the cut plane normal (the vector is normalized before the computation; the x axis of the local cut system points along the normal)"))
            hn.addWidget(s)
        fp.addRow(tr("Cut normal [nx, ny, nz]"), hn)

        # Preset buttons
        h_pre = QHBoxLayout()
        b_x = QPushButton(tr("YZ plane (normal X)"))
        b_x.setToolTip(tr("Sets the cut normal to the X axis (cut by the YZ plane); the cut point is unchanged"))
        b_x.clicked.connect(lambda: self._set_normal(1, 0, 0))
        b_y = QPushButton(tr("XZ plane (normal Y)"))
        b_y.setToolTip(tr("Sets the cut normal to the Y axis (cut by the XZ plane); the cut point is unchanged"))
        b_y.clicked.connect(lambda: self._set_normal(0, 1, 0))
        b_z = QPushButton(tr("XY plane (normal Z)"))
        b_z.setToolTip(tr("Sets the cut normal to the Z axis (cut by the XY plane); the cut point is unchanged"))
        b_z.clicked.connect(lambda: self._set_normal(0, 0, 1))
        h_pre.addWidget(b_x)
        h_pre.addWidget(b_y)
        h_pre.addWidget(b_z)
        fp.addRow(tr("Plane presets"), h_pre)

        if hasattr(win, "section_plane"):
            b_active = QPushButton(tr("Take from active model section"))
            b_active.setToolTip(tr("Takes the point and normal from the section shown in the results panel (section position slider)"))
            b_active.clicked.connect(self._take_from_active_section)
            fp.addRow("", b_active)

        lay.addWidget(g_plane)

        # ---------------------------------------------------------- State and parts
        f_opt = QFormLayout()
        self.step_combo = QComboBox()
        self.step_combo.setToolTip(tr("Load case whose results (displacements and stresses) are used for the cut"))
        if hasattr(win, "results") and win.results and win.results.steps:
            for i, st in enumerate(win.results.steps):
                self.step_combo.addItem(st.name, i)
        f_opt.addRow(tr("Load case"), self.step_combo)
        lay.addLayout(f_opt)

        lay.addWidget(QLabel(tr("Parts included in the cut (unchecked parts are omitted):")))
        self.part_list = QListWidget()
        self.part_list.setMaximumHeight(90)
        self.part_list.setToolTip(tr("Parts whose elements are included in the cut (checked); other parts are omitted"))
        if hasattr(win, "results") and win.results:
            for pn in win.results.mesh.part_names:
                item = QListWidgetItem(pn)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.part_list.addItem(item)
        lay.addWidget(self.part_list)

        # ---------------------------------------------------------- Calculate button
        h_act = QHBoxLayout()
        self.b_calc = QPushButton(tr("Compute section force"))
        theme.set_primary(self.b_calc)
        self.b_calc.setToolTip(tr("Computes the resultant force and moment in the cut (about the cut centroid, in the local cut system); a large mesh takes a few seconds"))
        self.b_calc.clicked.connect(self.compute)
        h_act.addWidget(self.b_calc)

        self.b_copy = QPushButton(tr("Copy to clipboard"))
        self.b_copy.setToolTip(tr("Copies the results table (tab-separated) to the clipboard, e.g. for Excel"))
        self.b_copy.setEnabled(False)
        self.b_copy.clicked.connect(self.copy_to_clipboard)
        h_act.addWidget(self.b_copy)
        lay.addLayout(h_act)

        # ---------------------------------------------------------- Results table
        self.tbl = QTableWidget()
        self.tbl.setToolTip(tr("Computed force and moment components in the cut: N = normal force, Vy and Vz = shear forces, Mx = torque, My and Mz = bending moments"))
        self.tbl.setColumnCount(3)
        self.tbl.setHorizontalHeaderLabels([tr("Component"), tr("Value"), tr("Unit")])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        from .widgets import copy_menu
        copy_menu(self.tbl)                       # Ctrl+C / context menu: the whole table
        lay.addWidget(self.tbl)

        self.lbl_info = QLabel("")
        self.lbl_info.setProperty("muted", True)
        self.lbl_info.setWordWrap(True)
        lay.addWidget(self.lbl_info)

        b_close = QPushButton(tr("Close"))
        b_close.setToolTip(tr("Closes the dialog (a running computation is discarded)"))
        b_close.clicked.connect(self.close)
        h_end = QHBoxLayout()
        h_end.addStretch(1)
        h_end.addWidget(b_close)
        lay.addLayout(h_end)

        # Try to take over the active cut immediately on opening
        self._take_from_active_section()

    def _set_normal(self, nx, ny, nz):
        self.nx.setValue(nx)
        self.ny.setValue(ny)
        self.nz.setValue(nz)

    def _take_from_active_section(self):
        if hasattr(self.win, "section_plane"):
            plane = self.win.section_plane()
            if plane is not None:
                pt, n = plane
                self.px.setValue(float(pt[0]))
                self.py.setValue(float(pt[1]))
                self.pz.setValue(float(pt[2]))
                self.nx.setValue(float(n[0]))
                self.ny.setValue(float(n[1]))
                self.nz.setValue(float(n[2]))

    # ------------------------------------------------------------------ calculation in the background
    def compute(self):
        """Starts the calculation of the force by a cut in the background (for a large mesh it takes seconds)."""
        if self._thread is not None and self._thread.isRunning():
            return
        res = getattr(self.win, "results", None)
        if not res or not res.steps:
            self.lbl_info.setText("! " + tr("No results of mechanical load cases are available"))
            return

        step_idx = self.step_combo.currentData()
        if step_idx is None:
            self.lbl_info.setText("! " + tr("Select a load case"))
            return
        point = [self.px.value(), self.py.value(), self.pz.value()]
        normal = [self.nx.value(), self.ny.value(), self.nz.value()]

        # Selected parts
        selected_parts = []
        for i in range(self.part_list.count()):
            item = self.part_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_parts.append(item.text())

        if not selected_parts:
            self.lbl_info.setText("! " + tr("Select at least one part"))
            return

        study, mesh, step = self.win.study, res.mesh, res.steps[step_idx]

        def job(t):
            r = section_forces(study, mesh, step, point, normal, selected_parts)
            return None if t.cancel.is_set() else r

        self._busy(True)
        self.lbl_info.setText(tr("Computing section force…"))
        thread = TaskThread(job, self)
        thread.done.connect(lambda r, th=thread: self._on_done(th, r))
        thread.failed.connect(lambda err, th=thread: self._on_failed(th, err))
        self._thread = thread
        thread.start()

    def _busy(self, on: bool):
        """A further one cannot be started during the calculation and a stale result is not copied."""
        self.b_calc.setEnabled(not on)
        self.b_copy.setEnabled(not on and self.last_result is not None)

    def _on_done(self, th, r):
        if th is not self._thread:                    # stale thread (the dialog was closed)
            return
        self._busy(False)
        if r is None:
            return
        self.last_result = r
        self.b_copy.setEnabled(True)
        self._fill_table(r)

    def _on_failed(self, th, err: str):
        if th is not self._thread:
            return
        self._busy(False)
        self.lbl_info.setText("! " + tr("The section force could not be computed:") + " " + err)
        QMessageBox.warning(self, tr("Section force"), err)

    def _stop_thread(self):
        """Discards the result of the unfinished calculation and waits for the end of the thread (so that it does not touch the deleted widgets)."""
        th = self._thread
        self._thread = None                           # late signals are ignored
        if th is not None and th.isRunning():
            th.cancel.set()
            th.wait(15000)

    def closeEvent(self, event):
        self._stop_thread()
        super().closeEvent(event)

    def reject(self):
        self._stop_thread()
        super().reject()

    def _fill_table(self, r: dict):
        Fg, Mg = np.asarray(r["F_global"], float), np.asarray(r["M_global"], float)
        c = r["centroid"]
        rows = [
            (tr("Normal force N (tension / compression)"), fmt(r["N"], 4), "N"),
            (tr("Shear force Vy"), fmt(r["Vy"], 4), "N"),
            (tr("Shear force Vz"), fmt(r["Vz"], 4), "N"),
            (tr("Total shear force V"), fmt(np.hypot(r["Vy"], r["Vz"]), 4), "N"),
            (tr("Total force F_tot"), fmt(float(np.linalg.norm(Fg)), 4), "N"),
            (tr("Torque Mx"), fmt(r["Mx"], 4), "N·mm"),
            (tr("Bending moment My"), fmt(r["My"], 4), "N·mm"),
            (tr("Bending moment Mz"), fmt(r["Mz"], 4), "N·mm"),
            (tr("Total bending moment M_bend"), fmt(np.hypot(r["My"], r["Mz"]), 4), "N·mm"),
            (tr("Total moment M_tot"), fmt(float(np.linalg.norm(Mg)), 4), "N·mm"),
            (tr("Cut centroid X"), fmt(c[0], 3), "mm"),
            (tr("Cut centroid Y"), fmt(c[1], 3), "mm"),
            (tr("Cut centroid Z"), fmt(c[2], 3), "mm"),
            (tr("Number of nodes in cut"), str(r["cut_nodes_count"]), "–"),
        ]

        self.tbl.setRowCount(len(rows))
        for i, (name, val, unit) in enumerate(rows):
            item0 = QTableWidgetItem(name)
            item1 = QTableWidgetItem(val)
            item1.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            item2 = QTableWidgetItem(unit)
            self.tbl.setItem(i, 0, item0)
            self.tbl.setItem(i, 1, item1)
            self.tbl.setItem(i, 2, item2)
        self.tbl.resizeColumnsToContents()

        self.lbl_info.setText("√ " + tr(
            f"Cut completed successfully: {r['cut_nodes_count']} nodes, centroid ({fmt(c[0])}; {fmt(c[1])}; {fmt(c[2])}) mm. The internal forces keep the free body in equilibrium."
        ))

    def copy_to_clipboard(self):
        """Copies the results table to the system clipboard."""
        if not self.last_result:
            return
        lines = [f"{tr('Component')}\t{tr('Value')}\t{tr('Unit')}"]
        for r in range(self.tbl.rowCount()):
            c0 = self.tbl.item(r, 0).text()
            c1 = self.tbl.item(r, 1).text()
            c2 = self.tbl.item(r, 2).text()
            lines.append(f"{c0}\t{c1}\t{c2}")
        QApplication.clipboard().setText("\n".join(lines))
        self.lbl_info.setText(tr("Results copied to the clipboard."))
