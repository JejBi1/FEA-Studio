"""Topology optimisation dialog (BESO / lightweighting for 3D printing, C4a)."""
from __future__ import annotations

import html
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QDoubleSpinBox,
    QPushButton, QComboBox, QFileDialog, QMessageBox, QGroupBox, QSpinBox,
    QProgressBar, QListWidget, QListWidgetItem, QTextBrowser
)

from .safe_browser import SafeTextBrowser
from . import theme
from .widgets import help_tip


class TopoWorker(QThread):
    progress = Signal(int, int, float, float)  # iter, max_iter, vol_frac, compliance
    finished_opt = Signal(object)              # TopologyOptimizer
    error = Signal(str)

    def __init__(self, study, mesh, spec):
        super().__init__()
        self.study = study
        self.mesh = mesh
        self.spec = spec
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()                      # also terminates a running ccx

    def run(self):
        try:
            from fea_core.topo.beso import TopologyOptimizer
            optimizer = TopologyOptimizer(self.study, self.mesh, self.spec)

            def on_iter(it, max_it, vf, comp):
                if self._cancel.is_set():
                    raise RuntimeError("The optimization was cancelled by the user")
                self.progress.emit(it, max_it, vf, comp)

            res = optimizer.run(callback=on_iter, cancel=self._cancel)    # same path as MCP fea_topo_opt_run
            self.finished_opt.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class TopologyOptDialog(QDialog):
    def __init__(self, parent, study, mesh, results=None):
        super().__init__(parent)
        self.setWindowTitle("Topology optimization (BESO / 3D printing, C4a)")
        self.resize(520, 580)
        self.study = study
        self.mesh = mesh
        self.results = results
        self.worker = None
        self.opt_result = None

        lay = QVBoxLayout(self)

        g_params = QGroupBox("Optimization goals and parameters")
        self.g_params = g_params
        f = QFormLayout(g_params)

        self.sp_target_vol = QDoubleSpinBox()
        self.sp_target_vol.setRange(5.0, 95.0)
        self.sp_target_vol.setValue(40.0)
        self.sp_target_vol.setSuffix(" %")
        self.sp_target_vol.setToolTip("Target volume after lightening relative to the original volume")
        f.addRow("Target volume:", self.sp_target_vol)

        self.sp_filter_r = QDoubleSpinBox()
        self.sp_filter_r.setRange(0.5, 100.0)
        self.sp_filter_r.setValue(6.0)
        self.sp_filter_r.setSuffix(" mm")
        self.sp_filter_r.setToolTip("Radius of the spatial sensitivity filter (controls the minimum strut thickness)")
        f.addRow("Min. thickness (filter R):", self.sp_filter_r)

        self.sp_er = QDoubleSpinBox()
        self.sp_er.setRange(0.5, 10.0)
        self.sp_er.setValue(4.0)
        self.sp_er.setSuffix(" %/step")
        help_tip(self.sp_er, "Maximum share of volume removed in one step (ER)",
                 "Evolutionary rate of material removal in one step. The target is reached after at least ln(target) / ln(1 − ER) steps, e.g. about 23 steps for ER 4 % and a 40 % target; the maximum number of iterations must be larger.")
        f.addRow("Removal rate (ER):", self.sp_er)

        self.sp_max_iter = QSpinBox()
        self.sp_max_iter.setRange(5, 200)
        self.sp_max_iter.setValue(60)
        self.sp_max_iter.setToolTip("Maximum number of optimization iterations")
        f.addRow("Max. iterations:", self.sp_max_iter)

        lay.addWidget(g_params)

        g_nondesign = QGroupBox("Protected regions (non-design regions)")
        g_nondesign.setToolTip("Regions from which material must not be removed (supports, loads, imprints, bolts)")
        v_nd = QVBoxLayout(g_nondesign)
        self.list_nd = QListWidget()
        self.list_nd.setSelectionMode(QListWidget.MultiSelection)
        self.list_nd.setToolTip("Face selections whose surroundings are not lightened (supports, loads, bolts); click to select, click again to deselect. All are selected by default")
        # filled with the selections available in the study
        for sname in self.study.selections:
            item = QListWidgetItem(sname)
            item.setSelected(True)  # All existing selections (constraints and forces) are protected by default
            self.list_nd.addItem(item)
        v_nd.addWidget(self.list_nd)
        lay.addWidget(g_nondesign)

        g_progress = QGroupBox("Optimization progress")
        v_prg = QVBoxLayout(g_progress)
        self.bar_progress = QProgressBar()
        self.bar_progress.setRange(0, 100)
        self.bar_progress.setValue(0)
        self.bar_progress.setToolTip("Progress by completed iterations out of the maximum allowed (may finish earlier when the design converges)")
        v_prg.addWidget(self.bar_progress)

        self.txt_log = SafeTextBrowser()
        self.txt_log.setMinimumHeight(120)
        self.txt_log.setToolTip("Optimization log: volume and compliance in every iteration")
        v_prg.addWidget(self.txt_log)
        lay.addWidget(g_progress)

        h_btn = QHBoxLayout()
        self.btn_run = QPushButton("Start optimization")
        theme.set_primary(self.btn_run)
        self.btn_run.setToolTip("Starts the iterative BESO optimization (one FEA run per iteration); while running, the button stops it")
        self.btn_run.clicked.connect(self.start_optimization)
        self.btn_show = QPushButton("Show final shape")
        help_tip(self.btn_show, "Shows the lightened shape in the 3D view (Results step)",
                 "Switches the main window to the quantity Topology – final shape (isosurface of density 0.5). Available when the optimization result is stored in the study results.")
        self.btn_show.setEnabled(False)
        self.btn_show.clicked.connect(self.show_shape)
        self.btn_export = QPushButton("Export lightened STL...")
        self.btn_export.setToolTip("Saves the resulting lightened shape as a smooth STL body (density isosurface 0.5)")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_stl)
        self.btn_close = QPushButton("Close")
        self.btn_close.setToolTip("Closes the dialog; a running optimization is stopped first")
        self.btn_close.clicked.connect(self.close)

        h_btn.addWidget(self.btn_run)
        h_btn.addWidget(self.btn_show)
        h_btn.addWidget(self.btn_export)
        h_btn.addWidget(self.btn_close)
        lay.addLayout(h_btn)

        self._show_existing()

    def _same_mesh(self) -> bool:
        """The study results are on the same mesh on which the optimisation runs (only then can the result be attached)."""
        return self.results is not None and self.results.mesh is self.mesh

    def _show_existing(self):
        """A result that already has a topology: summary to the log, enables display and export from the saved densities."""
        if not self._same_mesh():
            return
        from fea_core.topo.results import element_densities
        if element_densities(self.results) is None:
            return
        t = (self.results.info or {}).get("topology") or {}
        self.btn_export.setEnabled(True)
        self.btn_show.setEnabled(True)
        self.txt_log.append("<b>The results already contain a topology optimization:</b>")
        self.txt_log.append(f"Iterations: {t.get('iterations', '?')}, target volume {100 * float(t.get('target_volume_fraction', 0)):.1f} %, reached {100 * float(t.get('final_volume_fraction', 0)):.1f} %")
        c0, c1 = t.get("initial_compliance_Nmm"), t.get("final_compliance_Nmm")
        if c0 and c1:
            self.txt_log.append(f"Compliance: full design {c0:.4e} N·mm, result {c1:.4e} N·mm (×{c1 / c0:.2f})")
        self.txt_log.append("Running a new optimization overwrites the result.<br>")

    def show_shape(self):
        main = self.parent()
        if hasattr(main, "topo_attached"):
            main.topo_attached()
            self.accept()

    def start_optimization(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()                      # the button is restored only after the thread has really finished
            self.btn_run.setEnabled(False)
            self.btn_run.setText("Stopping…")
            return

        from fea_core.topo.spec import TopologyOptSpec

        nd_selected = [item.text() for item in self.list_nd.selectedItems()]
        spec = TopologyOptSpec(
            target_volume_fraction=self.sp_target_vol.value() / 100.0,
            evolution_rate=self.sp_er.value() / 100.0,
            filter_radius=self.sp_filter_r.value(),
            max_iterations=self.sp_max_iter.value(),
            non_design_selections=nd_selected,
        )

        self.txt_log.clear()
        self.txt_log.append(f"<b>BESO topology optimization started:</b>")
        self.txt_log.append(f"Target: {spec.target_volume_fraction * 100:.1f} % of the original volume")
        self.txt_log.append(f"Filter R: {spec.filter_radius:.1f} mm, ER: {spec.evolution_rate * 100:.1f} %")
        self.txt_log.append(f"Protected regions: {(', '.join(nd_selected) if nd_selected else 'none')}<br>")

        self.btn_run.setText("Stop")
        self.btn_run.setToolTip("Stops the running optimization (the iteration in progress is finished)")
        self.btn_export.setEnabled(False)
        self.g_params.setEnabled(False)
        self.list_nd.setEnabled(False)
        self.bar_progress.setValue(0)

        self.worker = TopoWorker(self.study, self.mesh, spec)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_opt.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self._on_thread_finished)      # QThread.finished: the thread has really ended
        self.worker.start()

    def _on_thread_finished(self):
        """The thread finished (success, error or cancellation): restores the controls."""
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Start optimization")
        self.btn_run.setToolTip("Starts the iterative BESO optimization (one FEA run per iteration); while running, the button stops it")
        self.g_params.setEnabled(True)
        self.list_nd.setEnabled(True)

    def _stop_worker(self) -> bool:
        """Cancels the running optimisation (also terminates ccx) and waits for the thread; without this the signals would reach the closed
        dialog. Returns False when the thread did not finish within 60 s (the dialog is then not closed).
        """
        w = self.worker
        if w is None or not w.isRunning():
            return True
        for sig in (w.progress, w.finished_opt, w.error, w.finished):
            try:
                sig.disconnect()
            except (RuntimeError, TypeError):
                pass
        w.cancel()
        if w.wait(60000):
            return True
        # the thread did not stop: restore the connection so that the dialog keeps showing the state
        w.progress.connect(self.on_progress)
        w.finished_opt.connect(self.on_finished)
        w.error.connect(self.on_error)
        w.finished.connect(self._on_thread_finished)
        return False

    def closeEvent(self, event):
        if not self._stop_worker():
            self.txt_log.append("<br>! The optimization could not be stopped; please wait for the iteration to finish.")
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self):                                 # Esc and the window button
        if not self._stop_worker():
            self.txt_log.append("<br>! The optimization could not be stopped; please wait for the iteration to finish.")
            return
        super().reject()

    def accept(self):
        if not self._stop_worker():
            return
        super().accept()

    def on_progress(self, it, max_it, vf, comp):
        pct = int(it / max(max_it, 1) * 100)
        self.bar_progress.setValue(pct)
        self.txt_log.append(
            f"Iteration {it}/{max_it}: Volume = {vf * 100:.1f} %, Compliance = {comp:.4e} N·mm"
        )

    def on_finished(self, opt_result):
        self.opt_result = opt_result
        self.btn_export.setEnabled(True)
        self.bar_progress.setValue(100)
        self.txt_log.append("<br><b>√ Topology optimization completed successfully.</b>")
        vf_fin = opt_result.current_volume_fraction
        self.txt_log.append(f"Final volume reached: {vf_fin * 100:.1f} %")
        c0, c1 = opt_result.initial_compliance, opt_result.final_compliance
        if c0 and c1:
            self.txt_log.append(f"Compliance: full design {c0:.4e} N·mm, result {c1:.4e} N·mm (×{c1 / c0:.2f} at volume {vf_fin * 100:.0f} %)")
        if self._same_mesh():
            try:
                from fea_core.topo.results import attach
                attach(self.results, opt_result)
            except Exception as e:  # noqa: BLE001 – the result stays at least for the STL export
                self.txt_log.append(f"! The result could not be attached to the study results: {html.escape(str(e))}")
            else:
                self.btn_show.setEnabled(True)
                self.txt_log.append("The result is stored in the study results (quantities Topology – density / final shape); it is saved with the project.")
                main = self.parent()
                if hasattr(main, "topo_attached"):
                    main.topo_attached()
        else:
            self.txt_log.append("The study results are on a different mesh – the result can only be exported to STL.")

    def on_error(self, err_msg):
        if "cancelled" in err_msg.lower():
            self.txt_log.append("<br><b>The optimization was stopped.</b>")
            return
        self.txt_log.append(f"<br><b>! Optimization error:</b> {html.escape(err_msg).replace(chr(10), '<br>')}")
        QMessageBox.critical(self, "Optimization error", err_msg)

    def export_stl(self):
        rho = self.opt_result.densities if self.opt_result is not None else None
        if rho is None and self._same_mesh():
            from fea_core.topo.results import element_densities
            rho = element_densities(self.results)          # saved densities from the results (0 / 1)
        if rho is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save optimized STL", "odlehceny_model.stl", "STL files (*.stl)"
        )
        if not path:
            return
        try:
            from fea_core.topo.export import export_topo_stl
            export_topo_stl(self.mesh, rho, path)
            QMessageBox.information(self, "STL export", f"The smooth body was saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))
