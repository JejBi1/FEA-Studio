"""Dialog for evaluating and exporting the principal stress vectors (A5: display of results).

The vectors are computed in the background (eigenvalues of the stress tensor at each node); a change of the scale or count
is computed after a short pause (300 ms), so that nothing is computed after every typed character.
CSV export: solver node number (node_id), part, position, principal stress number, magnitude, unit direction and state –
CSV format as in the results export dialog (`export_dialog.csv_dialect`).
"""
from __future__ import annotations

import numpy as np

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QDoubleSpinBox,
    QPushButton, QComboBox, QFileDialog, QMessageBox, QGroupBox, QSpinBox
)

from fea_core.post_vectors import principal_vector_segments
from . import theme
from .widgets import help_tip
from .workers import TaskThread


class VectorsDialog(QDialog):
    def __init__(self, parent, results, current_step_idx: int = 0, step=None, label: str | None = None):
        """step = another result on the mesh `results` (e.g. a combination – StepResult) instead of the state current_step_idx,
        label = its description in the dialog (e.g. "K01: name").
        """
        super().__init__(parent)
        self.setWindowTitle("Principal stress vectors (A5)")
        self.resize(460, 280)
        self.results = results
        self.step_idx = current_step_idx
        if step is None:
            step = results.step(current_step_idx) if results and results.steps else None
        self.step = step
        self.mesh = results.mesh if results else None
        if label is None and step is not None:
            label = f"{current_step_idx + 1}: {step.name}"

        self.seg_data = None
        self._thread: TaskThread | None = None

        # deferred recalculation after a change of the scale / count (one run after the value settles)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self.update_vectors)

        lay = QVBoxLayout(self)

        g = QGroupBox("Principal stress vector settings")
        f = QFormLayout(g)

        self.cb_which = QComboBox()
        self.cb_which.addItems([
            "σ1 – largest tensile stress",
            "σ2 – intermediate principal stress",
            "σ3 – largest compressive stress"
        ])
        self.cb_which.setToolTip("Principal stress: σ1 largest (tension), σ2 middle, σ3 smallest (compression)")
        self.cb_which.currentIndexChanged.connect(self.update_vectors)
        f.addRow("Direction / component:", self.cb_which)

        self.lbl_step = QLabel(label or "–")
        self.lbl_step.setToolTip("Case or combination the vectors are computed from (number: name)")
        f.addRow("Result:", self.lbl_step)

        self.sp_scale = QDoubleSpinBox()
        self.sp_scale.setRange(0.001, 1000.0)
        self.sp_scale.setValue(0.1)
        self.sp_scale.setSuffix(" mm/MPa")
        self.sp_scale.setToolTip("Vector length per 1 MPa of stress [mm/MPa]; changes only the lengths of exported segments")
        self.sp_scale.valueChanged.connect(self._schedule)
        self.sp_scale.editingFinished.connect(self._flush)
        f.addRow("Length scale:", self.sp_scale)

        self.sp_max = QSpinBox()
        self.sp_max.setRange(50, 50000)
        self.sp_max.setValue(2000)
        help_tip(self.sp_max, "Maximum number of vectors (nodes with the highest stress + a sample of the rest)",
                 "Maximum number of vectors: half are the nodes with the highest stress and the rest a uniform sample; nodes below 5 % of the maximum are skipped.")
        self.sp_max.valueChanged.connect(self._schedule)
        self.sp_max.editingFinished.connect(self._flush)
        f.addRow("Max. number of vectors:", self.sp_max)

        self.lbl_summary = QLabel("Number of vectors: -")
        self.lbl_summary.setProperty("muted", True)
        self.lbl_summary.setToolTip("Number of vectors and their split into tensile (σ > 0) and compressive (σ < 0)")
        f.addRow("Statistics:", self.lbl_summary)

        lay.addWidget(g)

        h_btn = QHBoxLayout()
        self.btn_export = QPushButton("Export CSV...")
        theme.set_primary(self.btn_export)
        self.btn_export.setToolTip("Saves the node, part, position, magnitude and direction of each vector to CSV")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_close = QPushButton("Close")
        self.btn_close.setToolTip("Close the principal stress vectors dialog")
        self.btn_close.clicked.connect(self.accept)
        h_btn.addWidget(self.btn_export)
        h_btn.addWidget(self.btn_close)
        lay.addLayout(h_btn)

        self.update_vectors()

    # ------------------------------------------------------------------ recalculation in the background
    def _schedule(self, _val=None):
        self._timer.start()

    def _flush(self):
        """End of editing of a field: recompute at once (only when a deferred recalculation is waiting)."""
        if self._timer.isActive():
            self._timer.stop()
            self.update_vectors()

    def update_vectors(self):
        self._timer.stop()
        if self.mesh is None or self.step is None:
            return
        which = self.cb_which.currentIndex() + 1
        scale = self.sp_scale.value()
        max_pts = self.sp_max.value()
        nodes, S = self.mesh.nodes, self.step.S

        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel.set()                 # the result of an older calculation is discarded

        self.lbl_summary.setText("Computing vectors…")
        self.btn_export.setEnabled(False)

        def job(t):
            seg = principal_vector_segments(nodes, S, which=which, scale=scale, max_points=max_pts)
            seg["which"] = which                      # number of the principal stress of the calculation (the choice could have changed)
            return None if t.cancel.is_set() else seg

        thread = TaskThread(job, self)
        thread.done.connect(lambda seg, th=thread: self._on_done(th, seg))
        thread.failed.connect(lambda err, th=thread: self._on_failed(th, err))
        self._thread = thread
        thread.start()

    def _on_done(self, th, seg):
        if th is not self._thread or seg is None:     # stale calculation
            return
        self.seg_data = seg
        n = len(seg["starts"])
        n_tens = int(np.sum(seg["is_tension"]))
        n_comp = n - n_tens
        self.lbl_summary.setText(f"Total: {n} (Tension: {n_tens}, Compression: {n_comp})")
        self.btn_export.setEnabled(n > 0)

    def _on_failed(self, th, err: str):
        if th is not self._thread:
            return
        self.seg_data = None
        self.lbl_summary.setText("! The vectors could not be computed")
        QMessageBox.warning(self, "Principal stress vectors", err)

    def _cleanup_thread(self):
        """Stops the deferred recalculation and waits for the thread (it must not touch the deleted widgets)."""
        self._timer.stop()
        th = self._thread
        self._thread = None                           # late signals are ignored
        if th is not None and th.isRunning():
            th.cancel.set()
            th.wait(15000)

    def closeEvent(self, event):
        self._cleanup_thread()
        super().closeEvent(event)

    def reject(self):
        self._cleanup_thread()
        super().reject()

    def accept(self):
        self._cleanup_thread()
        super().accept()

    # ------------------------------------------------------------------ export
    def export_csv(self):
        if self.seg_data is None or len(self.seg_data["starts"]) == 0:
            QMessageBox.warning(self, "Error", "There are no vectors to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export stress vectors to CSV", "vektory_napeti.csv", "CSV files (*.csv)")
        if not path:
            return
        try:
            self.write_csv(path)
            QMessageBox.information(self, "Export completed", f"The vectors were saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def write_csv(self, path):
        """Vectors to CSV: node_id (solver node number), part, x, y, z, principal (1 | 2 | 3), sigma_MPa, unit
        direction nx, ny, nz, type (tension | compression), step_name and the ends of the segment (start, end).
        """
        from fea_core.export_csv import node_parts, write_csv
        from .export_dialog import csv_dialect
        seg = self.seg_data
        idx = np.asarray(seg["indices"], dtype=np.int64)
        X = np.asarray(self.mesh.nodes, float)[idx]
        dirs = seg.get("directions")
        if dirs is None:                              # older result without directions: from the segment
            v = np.asarray(seg["ends"], float) - np.asarray(seg["starts"], float)
            dirs = v / np.maximum(np.linalg.norm(v, axis=1), 1e-300)[:, None]
        first, _ = node_parts(self.mesh)
        names = list(self.mesh.part_names)
        which = int(seg.get("which", self.cb_which.currentIndex() + 1))
        step_name = self.step.name if self.step is not None else ""
        header = ["node_id", "part", "x_mm", "y_mm", "z_mm", "principal", "sigma_MPa", "nx", "ny", "nz", "type",
                  "step_name", "x_start_mm", "y_start_mm", "z_start_mm", "x_end_mm", "y_end_mm", "z_end_mm"]
        rows = []
        for k, i in enumerate(idx.tolist()):
            pi = int(first[i])
            rows.append([i + 1, names[pi] if 0 <= pi < len(names) else "", *X[k], which, float(seg["values"][k]),
                         *dirs[k], "tension" if seg["is_tension"][k] else "compression", step_name,
                         *seg["starts"][k], *seg["ends"][k]])
        return write_csv(path, header, rows, csv_dialect(self.parent()), fmt="%.6f")
