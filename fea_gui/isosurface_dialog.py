"""Dialog for displaying and exporting isosurfaces of the results (A5: display of results)."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QDoubleSpinBox,
    QPushButton, QComboBox, QFileDialog, QMessageBox, QGroupBox
)
from PySide6.QtCore import Qt, QTimer

from fea_core.isosurface import extract_isosurface
from . import theme
from .workers import TaskThread


class IsosurfaceDialog(QDialog):
    def __init__(self, parent, results, current_step_idx: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Result Isosurface (Marching Tetrahedra, A5)")
        self.resize(450, 260)
        self.results = results
        self.step_idx = current_step_idx
        self.step = results.step(current_step_idx) if results and results.steps else None
        self.mesh = results.mesh if results else None

        self.iso_verts = None
        self.iso_faces = None
        self._thread: TaskThread | None = None

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self.update_isosurface)

        lay = QVBoxLayout(self)

        g = QGroupBox("Isosurface settings")
        f = QFormLayout(g)

        self.cb_field = QComboBox()
        self.cb_field.setToolTip("Select result field for isosurface generation (von Mises stress, principal stress)")
        self.cb_field.addItems(["von Mises stress [MPa]", "Principal stress σ1 [MPa]", "Principal stress σ3 [MPa]"])
        self.cb_field.currentIndexChanged.connect(self.on_field_changed)
        f.addRow("Field:", self.cb_field)

        self.lbl_range = QLabel("")
        self.lbl_range.setProperty("muted", True)
        f.addRow("Value range:", self.lbl_range)

        self.sp_val = QDoubleSpinBox()
        self.sp_val.setToolTip("Field value for isosurface generation via Marching Tetrahedra")
        self.sp_val.setDecimals(2)
        self.sp_val.valueChanged.connect(self._on_value_changed)
        self.sp_val.editingFinished.connect(self._on_editing_finished)
        f.addRow("Isosurface value:", self.sp_val)

        self.lbl_info = QLabel("Not calculated yet")
        self.lbl_info.setProperty("muted", True)
        f.addRow("Mesh status:", self.lbl_info)

        lay.addWidget(g)

        h_btn = QHBoxLayout()
        self.btn_export = QPushButton("Export STL...")
        theme.set_primary(self.btn_export)
        self.btn_export.setToolTip("Export created isosurface to 3D STL file")
        self.btn_export.clicked.connect(self.export_stl)

        self.btn_close = QPushButton("Close")
        self.btn_close.setToolTip("Close isosurface dialog")
        self.btn_close.clicked.connect(self.accept)

        h_btn.addWidget(self.btn_export)
        h_btn.addWidget(self.btn_close)
        lay.addLayout(h_btn)

        self.on_field_changed()

    def _on_value_changed(self, _val: float):
        self._timer.start()

    def _on_editing_finished(self):
        self._timer.stop()
        self.update_isosurface()

    def get_field_values(self) -> np.ndarray:
        if self.step is None:
            return np.zeros(0)
        idx = self.cb_field.currentIndex()
        if idx == 0:
            return self.step.mises
        elif idx == 1:
            return self.step.principal[:, 0]
        else:
            return self.step.principal[:, 2]

    def on_field_changed(self):
        vals = self.get_field_values()
        if len(vals) == 0:
            return
        vmin, vmax = float(vals.min()), float(vals.max())
        self.lbl_range.setText(f"{vmin:.2f} to {vmax:.2f}")
        self.sp_val.blockSignals(True)
        self.sp_val.setRange(vmin, vmax)
        self.sp_val.setValue(0.5 * (vmin + vmax))
        self.sp_val.blockSignals(False)
        self.update_isosurface()

    def update_isosurface(self):
        if self.mesh is None or self.step is None:
            return
        vals = self.get_field_values()
        if len(vals) == 0:
            return
        c = float(self.sp_val.value())

        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel.set()

        self.lbl_info.setText("Computing isosurface…")
        self.lbl_info.setProperty("muted", True)
        self.lbl_info.style().unpolish(self.lbl_info)
        self.lbl_info.style().polish(self.lbl_info)

        nodes = self.mesh.nodes
        tets = self.mesh.tets

        def job(t):
            verts, faces = extract_isosurface(nodes, tets, vals, c)
            if t.cancel.is_set():
                return None
            return verts, faces

        thread = TaskThread(job, self)
        thread.done.connect(self._on_iso_done)
        thread.failed.connect(self._on_iso_failed)
        self._thread = thread
        thread.start()

    def _on_iso_done(self, result):
        if result is None:
            return
        self.iso_verts, self.iso_faces = result
        nv = len(self.iso_verts)
        nf = len(self.iso_faces)
        self.lbl_info.setText(f"{nf} triangles, {nv} vertices")
        self.lbl_info.setProperty("muted", False)
        self.lbl_info.style().unpolish(self.lbl_info)
        self.lbl_info.style().polish(self.lbl_info)

    def _on_iso_failed(self, err: str):
        self.lbl_info.setText(f"Error: {err}")
        self.lbl_info.setProperty("muted", True)
        self.lbl_info.style().unpolish(self.lbl_info)
        self.lbl_info.style().polish(self.lbl_info)

    def closeEvent(self, event):
        self._cleanup_thread()
        super().closeEvent(event)

    def reject(self):
        self._cleanup_thread()
        super().reject()

    def accept(self):
        self._cleanup_thread()
        super().accept()

    def _cleanup_thread(self):
        self._timer.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel.set()
            self._thread.wait(500)

    def export_stl(self):
        if self.iso_verts is None or self.iso_faces is None or len(self.iso_faces) == 0:
            QMessageBox.warning(self, "Error", "No isosurface was generated for the specified value.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Isosurface STL", "isoplocha.stl", "STL files (*.stl)")
        if not path:
            return
        try:
            # Simple ASCII STL export
            with open(path, "w", encoding="ascii") as f:
                f.write("solid isosurface\n")
                for tri in self.iso_faces:
                    p0, p1, p2 = self.iso_verts[tri[0]], self.iso_verts[tri[1]], self.iso_verts[tri[2]]
                    n = np.cross(p1 - p0, p2 - p0)
                    norm = np.linalg.norm(n)
                    if norm > 1e-12:
                        n /= norm
                    else:
                        n = np.array([0.0, 0.0, 1.0])
                    f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                    f.write("    outer loop\n")
                    f.write(f"      vertex {p0[0]:.6e} {p0[1]:.6e} {p0[2]:.6e}\n")
                    f.write(f"      vertex {p1[0]:.6e} {p1[1]:.6e} {p1[2]:.6e}\n")
                    f.write(f"      vertex {p2[0]:.6e} {p2[1]:.6e} {p2[2]:.6e}\n")
                    f.write("    endloop\n")
                    f.write("  endfacet\n")
                f.write("endsolid isosurface\n")
            QMessageBox.information(self, "Export completed", f"Isosurface was saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))
