""""Mesh convergence" dialog (fea_core.convergence): the same study on gradually finer meshes (globally or in a sphere
around a point), a running table of the levels, a chart value × element size, Richardson extrapolation, GCI and the verdict.
The calculation runs in a thread over a COPY of the study – neither the study nor the results of the main window change.
"""
from __future__ import annotations

import time
from html import escape

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                               QPushButton, QSpinBox, QTableWidgetItem, QVBoxLayout)

from .safe_browser import SafeTextBrowser
from . import theme
from .widgets import LazyPlot, TaskProgress, dspin, fmt, table

QUANTITIES = [("max_mises", "max. von Mises [MPa]"), ("max_disp", "max. displacement [mm]"), ("probe", "value at point"),
              ("frequency", "natural frequency [Hz]"), ("print_safety", "min. 3D-print safety factor [–]")]
PROBE_FIELDS = [("mises", "von Mises"), ("s1", "1st principal stress S1"), ("s3", "3rd principal stress S3"),
                ("sx", "σx"), ("sy", "σy"), ("sz", "σz"), ("umag", "displacement |u|"), ("ux", "ux"), ("uy", "uy"),
                ("uz", "uz")]


class ConvergenceDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Mesh convergence")
        self.setModal(False)
        self.resize(640, 820)
        self.thread = None
        self.result = None
        self.rows: list[dict] = []
        lay = QVBoxLayout(self)
        intro = QLabel("The study is solved on several progressively finer meshes (element ÷ ratio at each level, globally or only in a sphere around the point) and the selected quantity is monitored. The main window result is not changed.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        f = QFormLayout()
        self.qty = QComboBox()
        for k, t in QUANTITIES:
            self.qty.addItem(t, k)
        self.qty.setToolTip("Monitored quantity: maximum von Mises / displacement (whole model or sphere), value at a point\n(probe), natural frequency of the selected mode, lowest 3D-print safety factor")
        self.qty.currentIndexChanged.connect(lambda _: self._sync())
        f.addRow("Quantity", self.qty)
        self.field = QComboBox()
        for k, t in PROBE_FIELDS:
            self.field.addItem(t, k)
        self.field.setToolTip("Probe quantity at the point (only “value at point”)")
        self.field_lbl = QLabel("Quantity at point")
        f.addRow(self.field_lbl, self.field)
        self.step = QComboBox()
        self.step.setToolTip("Load case in which the quantity is evaluated")
        self.step_lbl = QLabel("Load case")
        f.addRow(self.step_lbl, self.step)
        self.mode = QSpinBox()
        self.mode.setRange(1, 200)
        self.mode.setToolTip("Mode shape number (1 = lowest frequency); natural frequencies must be enabled in the Solve step")
        self.mode_lbl = QLabel("Mode")
        f.addRow(self.mode_lbl, self.mode)
        h = QHBoxLayout()
        self.pt = [dspin(0.0, -1e7, 1e7, 4) for _ in range(3)]
        for s, ax in zip(self.pt, "xyz"):
            s.setToolTip(f"{ax} coordinate [mm]: probe point, or center of the refinement and maximum evaluation sphere")
            h.addWidget(s)
        self.btn_pick = QPushButton("Click in 3D")
        self.btn_pick.setCheckable(True)
        self.btn_pick.setToolTip("Point = point clicked on the model in the 3D view in the Results step (undeformed position)")
        self.btn_pick.clicked.connect(self.start_pick)
        h.addWidget(self.btn_pick)
        b = QPushButton("Maximum location")
        b.setToolTip("Point = node with the highest value of the displayed quantity in the results (“max” marker)")
        b.clicked.connect(self.point_from_max)
        h.addWidget(b)
        self.pt_lbl = QLabel("Point [mm]")
        f.addRow(self.pt_lbl, h)
        self.radius = dspin(0.0, 0.0, 1e6, 3, 1.0, " mm", special="global")
        self.radius.setToolTip("Sphere radius around the point: the mesh is refined only inside it and the maximum is searched only inside it\n(“global” = refine the whole mesh, maximum from the whole model)")
        self.radius.valueChanged.connect(lambda _: self._sync())
        f.addRow("Sphere radius", self.radius)
        self.levels = QSpinBox()
        self.levels.setRange(2, 6)
        self.levels.setValue(3)
        self.levels.setToolTip("Number of mesh levels (analyses). Convergence order, extrapolation and GCI need at least 3.")
        f.addRow("Levels", self.levels)
        self.ratio = dspin(1.5, 1.1, 3.0, 2, 0.1)
        self.ratio.setToolTip("Element size ratio between adjacent levels (1.5 = each level has elements one third smaller;\nelement count grows ~ r³ – with 3 levels and r = 1.5 about 11× more elements on the last)")
        f.addRow("Refinement ratio r", self.ratio)
        self.tol = dspin(2.0, 0.1, 50.0, 1, 0.5, " %")
        self.tol.setToolTip("Relative change limit of the last refinement for the “converged” verdict")
        f.addRow("Change limit", self.tol)
        lay.addLayout(f)
        self.est = QLabel("")
        self.est.setWordWrap(True)
        self.est.setProperty("muted", True)
        lay.addWidget(self.est)
        for w in (self.levels, self.ratio):
            w.valueChanged.connect(lambda _: self._sync())
        h = QHBoxLayout()
        self.btn_run = QPushButton("Run convergence")
        theme.set_primary(self.btn_run)
        self.btn_run.setToolTip("Run all level analyses in the background (each level = mesh + analysis of the whole study)")
        self.btn_run.clicked.connect(self.run)
        h.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip("Abort the running level analysis (completed levels stay in the table)")
        self.btn_cancel.clicked.connect(self.cancel)
        h.addWidget(self.btn_cancel)
        lay.addLayout(h)
        self.progress = TaskProgress()
        lay.addWidget(self.progress)
        self.tab = table(["Level", "Element [mm]", "Nodes", "Elements", "Value", "Change [%]", "ccx error [%]"], 130)
        self.tab.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tab.setToolTip("Mesh levels: element size, number of nodes and elements, monitored value, change from the previous\nlevel and ccx stress error estimate (maximum of the “Stress error estimate” field in the region)")
        lay.addWidget(self.tab)
        self.plot_w = LazyPlot()
        self.plot_w.setToolTip("Monitored quantity vs. element size (coarse mesh on the right); dashed line = extrapolated value")
        lay.addWidget(self.plot_w, 1)
        self.out = SafeTextBrowser()
        self.out.setMinimumHeight(110)
        self.out.setToolTip("Result: verdict, convergence order p, extrapolated value (Richardson), GCI and warnings")
        lay.addWidget(self.out)
        bb = QHBoxLayout()
        bb.addStretch(1)
        b = QPushButton("Close")
        b.setToolTip("Close the dialog (a running analysis is cancelled)")
        b.clicked.connect(self.close)
        bb.addWidget(b)
        lay.addLayout(bb)
        self.refresh()

    # ------------------------------------------------------------------ form
    def refresh(self):
        st = self.win.study
        self.step.clear()
        for i, lc in enumerate(st.load_cases if st else []):
            if lc.loads:
                self.step.addItem(lc.name, i)
        cur = self.win.current_result()
        if cur and cur[0] == "step" and self.win.results is not None:
            name = self.win.results.steps[cur[1]].name
            k = self.step.findText(name)
            if k >= 0:
                self.step.setCurrentIndex(k)
        if st is not None and st.modal.enabled:
            self.mode.setMaximum(max(1, int(st.modal.modes)))
        self._sync()

    def _sync(self):
        q = self.qty.currentData()
        probe = q == "probe"
        for w in (self.field, self.field_lbl):
            w.setVisible(probe)
        for w in (self.mode, self.mode_lbl):
            w.setVisible(q == "frequency")
        for w in (self.step, self.step_lbl):
            w.setVisible(q != "frequency")
        st = self.win.study
        if st is None:
            return
        r, n = self.ratio.value(), self.levels.value()
        h0 = float(st.mesh.size)
        where = (f"in sphere R = {fmt(self.radius.value(), 4)} mm" if self.radius.value() > 0 else "in the whole model")
        n_el = ""
        m = self.win.mesh
        if m is not None and self.radius.value() <= 0:
            n_el = f"; elements on the last level ~{fmt(float(round(len(m.tets) * r ** (3 * (n - 1)), -2)), 8)}"
        self.est.setText(f"Element sizes {where}: " + ", ".join(f"{fmt(h0 / r ** k, 4)} mm" for k in range(n))
                         + n_el + ".")

    def spec(self):
        from fea_core.convergence import ConvergenceSpec
        q = self.qty.currentData()
        pt = [s.value() for s in self.pt]
        rad = self.radius.value() or None
        step = (self.mode.value() - 1) if q == "frequency" else (self.step.currentData() or 0)
        return ConvergenceSpec(quantity=q, field=self.field.currentData() or "mises",
                               point=pt if (q == "probe" or rad) else None, radius=rad, step=int(step),
                               levels=int(self.levels.value()), ratio=float(self.ratio.value()),
                               tol=float(self.tol.value()) / 100.0)

    def set_point(self, p):
        for s, v in zip(self.pt, p):
            s.setValue(float(v) + 0.0)

    def start_pick(self, on=True):
        if not on:
            if getattr(self.win, "_pick_hook", None) == self.on_pick:
                self.win._pick_hook = None
            return
        if self.win.results is None:
            self.btn_pick.setChecked(False)
            QMessageBox.information(self, self.windowTitle(), "The point is picked by clicking in the results – run the analysis first.")
            return
        if self.win.STAGES[self.win.stage] != "Results":
            self.win.set_stage(self.win.STAGE["Results"])
        self.win._pick_hook = self.on_pick
        self.win.statusBar().showMessage("Click on the model in 3D – convergence point", 8000)

    def on_pick(self, p):
        self.btn_pick.setChecked(False)
        self.win._pick_hook = None
        self.set_point(p)

    def point_from_max(self):
        pr = getattr(self.win, "_probe", None)
        if pr is None:
            QMessageBox.information(self, self.windowTitle(), "Display a result in the Results step.")
            return
        tree, ids, vals, _, _ = pr
        j = int(np.nanargmax(vals))
        self.set_point(self.win.undeformed(tree.data[j]))

    # ------------------------------------------------------------------ calculation
    def run(self):
        from .workers import TaskThread
        win = self.win
        if win.study is None or self.thread is not None:
            return
        if win._thread is not None:
            QMessageBox.information(self, self.windowTitle(), "Meshing or analysis in progress – wait for it to finish.")
            return
        spec = self.spec()
        if spec.quantity == "probe" and spec.point is None:
            QMessageBox.information(self, self.windowTitle(), "Enter the probe point.")
            return
        if spec.quantity == "frequency" and not win.study.modal.enabled:
            QMessageBox.information(self, self.windowTitle(), "Natural frequencies are not enabled (Solve step).")
            return
        if spec.quantity != "frequency" and self.step.count() == 0:
            QMessageBox.information(self, self.windowTitle(), "The study has no load case with loads.")
            return
        errors, _ = win.check()
        if errors:
            QMessageBox.warning(self, self.windowTitle(), "The study is not ready:\n\n" + "\n".join(errors))
            return
        st = win.study_for_solve()
        st.geometry = win.geometry_str()
        if spec.quantity != "frequency":           # state index after skipping the empty states (study_for_solve)
            name = win.study.load_cases[spec.step].name
            spec.step = next((i for i, lc in enumerate(st.load_cases) if lc.name == name), 0)
        wd = win._workdir / f"konvergence_{int(time.time() * 1000)}"

        def job(t):
            from fea_core.convergence import run_convergence
            return run_convergence(st, wd, spec, on_log=t.log.emit, cancel=t.cancel,
                                   on_level=lambda k, row: t.outcome.emit((k, row)))
        self.rows = []
        self.result = None
        self.tab.setRowCount(0)
        self.out.setHtml("")
        t = TaskThread(job, self)
        t.log.connect(win.log_msg)
        t.outcome.connect(self.on_level)
        t.done.connect(self.on_done)
        t.failed.connect(self.on_failed)
        t.finished.connect(self.on_finished)
        self.thread = t
        win.register_aux(t)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.start(f"Level 1/{spec.levels}", 1.0 / spec.levels)
        self._levels = spec.levels
        win.log_msg(f"Mesh convergence: {spec.levels} levels, ratio {spec.ratio:g}, quantity {spec.quantity}")
        t.start()

    def cancel(self):
        if self.thread is not None:
            self.thread.cancel.set()
            from .workers import kill_mesh_workers
            kill_mesh_workers()
            self.win.log_msg("Cancelling mesh convergence…")

    def on_level(self, item):
        k, row = item
        self.rows.append(row)
        self._fill_table()
        self._plot()
        n = self._levels
        if k + 1 < n:
            self.progress.report((k + 1) / n, (k + 2) / n, f"Level {k + 2}/{n}")

    def _fill_table(self):
        t = self.tab
        t.setRowCount(len(self.rows))
        for r, row in enumerate(self.rows):
            ch = row.get("change")
            vals = [str(row["level"]), fmt(row["size_mm"], 4), fmt(float(row["nodes"]), 6), fmt(float(row["elements"]), 6),
                    fmt(row["value"], 6), fmt(100 * ch, 3) if ch is not None else "–",
                    fmt(row.get("stress_error_pct"), 3)]
            for c, txt in enumerate(vals):
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(r, c, it)

    def _plot(self):
        if not self.rows:
            return
        h = [r["size_mm"] for r in self.rows]
        v = [r["value"] for r in self.rows]
        ax = self.plot_w.axes()
        ax.plot(h, v, "o-", color="#2563eb", lw=1.8)
        ext = (self.result or {}).get("extrapolated")
        if ext is not None and np.isfinite(ext):
            ax.axhline(ext, color="#e69a1a", ls="--", lw=1.2, label="extrapolation")
            ax.legend(loc="best", fontsize=8)
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.set_xlabel("element size [mm] (coarser on the right)")
        ax.set_ylabel(self.qty.currentText())
        ax.grid(True, alpha=0.3)
        self.plot_w.canvas.draw_idle()

    def on_done(self, out):
        self.result = out
        self.rows = list(out.get("levels", []))
        self._fill_table()
        self._plot()
        rows = [f"<b>{escape(str(out.get('verdict', '')))}</b>"]
        if out.get("order") is not None:
            rows.append(f"Convergence order p = {fmt(out['order'], 3)}")
        if out.get("extrapolated") is not None:
            rows.append(f"Extrapolated value (Richardson): {fmt(out['extrapolated'], 6)}")
        if out.get("gci") is not None:
            rows.append(f"GCI (grid convergence index, Roache): {fmt(100 * out['gci'], 3)} %")
        for w in out.get("warnings", []):
            rows.append(f"<span style='color:#b36b00'>! {escape(str(w))}</span>")
            self.win.messages.add("warning", "results", "Mesh convergence: " + w, "Mesh")
        self.out.setHtml("<br>".join(rows))
        self.win.messages.add("info" if out.get("converged") else "warning", "results",
                              "Mesh convergence: " + out.get("verdict", ""), "Mesh")
        self.progress.finish("done", "Convergence complete")
        self.win.log_msg("Mesh convergence: " + out.get("verdict", ""))

    def on_failed(self, text):
        cancelled = self.thread is not None and self.thread.cancel.is_set()
        self.progress.finish("cancelled" if cancelled else "error", "Cancelled" if cancelled else "Analysis failed")
        self.out.setHtml(f"<span style='color:#d23b3b'>{text}</span>")
        self.win.log_msg(f"Mesh convergence ERROR: {text}")
        if not cancelled:
            self.win.messages.add("error", "solver", f"Mesh convergence failed: {text}")

    def on_finished(self):
        self.win.unregister_aux(self.thread)
        self.thread = None
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def closeEvent(self, e):
        if self.thread is not None:
            self.cancel()
            self.thread.wait(10000)
        if getattr(self.win, "_pick_hook", None) == self.on_pick:
            self.win._pick_hook = None
        super().closeEvent(e)
