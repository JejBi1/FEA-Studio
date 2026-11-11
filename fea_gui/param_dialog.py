""""Parametric study" dialog (fea_core.sweep, C4b): parameters (load scale, material, material property, mesh size, 3D print
setting) as a grid or a list of values, calculation in a thread over COPIES of the study, live table and chart of the KPI
(max. stress outside singularities, displacement, safety, mass, natural frequency), Pareto front, take over a design point.
The study and the results of the main window do not change until a point is taken over.
"""
from __future__ import annotations

import math
import time
from html import escape

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSpinBox, QTableWidgetItem, QVBoxLayout)

from fea_core.sweep import (MATERIAL_FIELDS, PRINT_FIELDS, ParametricStudySpec, SweepError, SweepParam, apply_params,
                            check_spec, kpi_columns, parse_values, _linear_scalable)
from . import theme
from .widgets import LazyPlot, TaskProgress, fmt, help_tip, table

KINDS = [("load_factor", "Load factor (all load cases)"), ("case_factor", "Load factor of one load case"),
         ("material", "Material of a part"), ("material_prop", "Material property"), ("mesh_size", "Mesh size [mm]"),
         ("print", "3D print setting")]
PLACEHOLDER = {"material": "material names separated by ;", "print": "e.g. 2, 3, 4 or 0.1:0.4:4"}


def _names(combo: QComboBox) -> None:
    combo.setProperty("_i18n_done", True)          # names of cases / parts / materials are not translated


class ParamRow:
    """One parameter row of the table: kind, target, property and values (widgets in the cells)."""

    def __init__(self, dlg, row: int):
        self.dlg = dlg
        t = dlg.ptab
        self.kind = QComboBox()
        for k, txt in KINDS:
            self.kind.addItem(txt, k)
        help_tip(self.kind, "What is swept: load scale, material, material property, mesh size, print setting")
        self.target = QComboBox()
        _names(self.target)
        help_tip(self.target, "Load case, part or material the parameter applies to")
        self.field = QComboBox()
        _names(self.field)
        help_tip(self.field, "Material property or print setting")
        self.values = QLineEdit()
        help_tip(self.values, "Values: list “1, 1.5, 2” or range “start:stop:n” (n values incl. both ends)")
        for c, w in enumerate((self.kind, self.target, self.field, self.values)):
            t.setCellWidget(row, c, w)
        self.kind.currentIndexChanged.connect(self._fill)
        self.values.textChanged.connect(lambda _: dlg.update_estimate())
        self.target.currentIndexChanged.connect(lambda _: dlg.update_estimate())
        self.field.currentIndexChanged.connect(lambda _: self._sync_placeholder())
        self._fill()

    def _fill(self):
        st = self.dlg.win.study
        k = self.kind.currentData()
        self.target.clear()
        self.field.clear()
        if st is not None:
            if k == "case_factor":
                self.target.addItems([lc.name for lc in st.load_cases if lc.loads])
            elif k == "material":
                self.target.addItems([p.name for p in self.dlg.win.solid_parts()] if self.dlg.win.parts else
                                     list(st.part_materials))
            elif k == "material_prop":
                self.target.addItems(list(st.materials))
                self.field.addItems(list(MATERIAL_FIELDS))
            elif k == "print":
                self.target.addItems(list(st.printing.parts))
                self.field.addItems(list(PRINT_FIELDS))
        self.target.setEnabled(self.target.count() > 0)
        self.field.setEnabled(self.field.count() > 0)
        self._sync_placeholder()
        self.dlg.update_estimate()

    def _sync_placeholder(self):
        k = self.kind.currentData()
        if k == "material" and self.dlg.win.study is not None:
            self.values.setPlaceholderText("; ".join(list(self.dlg.win.study.materials)[:4]))
        elif k == "print" and self.field.currentText() == "pattern":
            self.values.setPlaceholderText("grid; gyroid; cubic")
        else:
            self.values.setPlaceholderText(PLACEHOLDER.get(k, "e.g. 0.5, 1, 1.5 or 0.5:2:4"))

    def param(self) -> SweepParam:
        k = self.kind.currentData()
        numeric = not (k == "material" or (k == "print" and self.field.currentText() == "pattern"))
        vals = parse_values(self.values.text(), numeric)
        return SweepParam(kind=k, values=vals, target=self.target.currentText() if self.target.isEnabled() else "",
                          field=self.field.currentText() if self.field.isEnabled() else "")


class ParamDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Parametric study")
        self.setModal(False)
        self.resize(820, 860)
        self.thread = None
        self.result = None
        self.points = []
        self.rows: list[ParamRow] = []
        lay = QVBoxLayout(self)
        intro = QLabel("Each design point is a full analysis of a COPY of the study with the given parameter values. The table "
                       "and the chart show the key results; the study of the main window changes only when you take over a "
                       "point. CAD parameters are not swept (the geometry is not regenerated) – use Replace geometry for that.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.ptab = table(["Parameter", "Applies to", "Property", "Values"], 120)
        self.ptab.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        help_tip(self.ptab, "Swept parameters (one row each); grid = all combinations, list = i-th values together")
        lay.addWidget(self.ptab)
        h = QHBoxLayout()
        b = QPushButton("Add parameter")
        help_tip(b, "Add a parameter row")
        b.clicked.connect(self.add_row)
        h.addWidget(b)
        b = QPushButton("Remove parameter")
        help_tip(b, "Remove the selected parameter row")
        b.clicked.connect(self.remove_row)
        h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        f = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("Grid – all combinations", "grid")
        self.mode.addItem("List – i-th values together", "list")
        help_tip(self.mode, "Grid: every combination of the values; list: the n-th values of all parameters together")
        self.mode.currentIndexChanged.connect(lambda _: self.update_estimate())
        f.addRow("Design points", self.mode)
        self.max_points = QSpinBox()
        self.max_points.setRange(1, 200)
        self.max_points.setValue(30)
        help_tip(self.max_points, "Upper limit of design points (each point = one analysis)")
        self.max_points.valueChanged.connect(lambda _: self.update_estimate())
        f.addRow("At most", self.max_points)
        lay.addLayout(f)
        self.est = QLabel("")
        self.est.setWordWrap(True)
        self.est.setProperty("muted", True)
        lay.addWidget(self.est)
        h = QHBoxLayout()
        self.btn_run = QPushButton("Run parametric study")
        theme.set_primary(self.btn_run)
        help_tip(self.btn_run, "Solve all design points in the background (copies of the study)")
        self.btn_run.clicked.connect(self.run)
        h.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        help_tip(self.btn_cancel, "Abort the running point and stop (finished points stay in the table)")
        self.btn_cancel.clicked.connect(self.cancel)
        h.addWidget(self.btn_cancel)
        lay.addLayout(h)
        self.progress = TaskProgress()
        lay.addWidget(self.progress)
        self.rtab = table(["DP"], 160)
        self.rtab.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.rtab.setSelectionMode(QAbstractItemView.SingleSelection)
        help_tip(self.rtab, "Design points and their KPI; * = Pareto front (lighter and less stressed)")
        lay.addWidget(self.rtab)
        h = QHBoxLayout()
        self.x_axis = QComboBox()
        help_tip(self.x_axis, "Horizontal axis of the chart: a parameter or a result")
        self.y_axis = QComboBox()
        help_tip(self.y_axis, "Vertical axis of the chart: a result (KPI)")
        for a, t in kpi_columns():
            self.y_axis.addItem(t, a)
        self.x_axis.currentIndexChanged.connect(lambda _: self._plot())
        self.y_axis.currentIndexChanged.connect(lambda _: self._plot())
        h.addWidget(QLabel("Chart:"))
        h.addWidget(self.y_axis, 1)
        h.addWidget(QLabel("vs."))
        h.addWidget(self.x_axis, 1)
        lay.addLayout(h)
        self.plot_w = LazyPlot()
        help_tip(self.plot_w, "Chosen result vs. parameter; orange = Pareto front")
        lay.addWidget(self.plot_w, 1)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        bb = QHBoxLayout()
        self.btn_apply = QPushButton("Take over selected point")
        help_tip(self.btn_apply, "Copy the parameters of the selected point into the study (results are discarded)")
        self.btn_apply.clicked.connect(self.apply_selected)
        self.btn_apply.setEnabled(False)
        bb.addWidget(self.btn_apply)
        bb.addStretch(1)
        b = QPushButton("Close")
        help_tip(b, "Close the dialog (a running study is cancelled)")
        b.clicked.connect(self.close)
        bb.addWidget(b)
        lay.addLayout(bb)
        self.rtab.itemSelectionChanged.connect(lambda: self.btn_apply.setEnabled(self._selected() is not None
                                                                                  and self.thread is None))
        prev = getattr(win, "param_study", None)
        self.add_row()
        if prev is not None and getattr(prev, "points", None):
            self.result = prev
            self.points = list(prev.points)
            self._fill_results()

    # ------------------------------------------------------------------ parameters
    def add_row(self):
        r = self.ptab.rowCount()
        self.ptab.insertRow(r)
        self.rows.append(ParamRow(self, r))
        self.update_estimate()

    def remove_row(self):
        r = self.ptab.currentRow()
        if r < 0:
            r = self.ptab.rowCount() - 1
        if r < 0:
            return
        self.ptab.removeRow(r)
        del self.rows[r]
        self.update_estimate()

    def spec(self) -> ParametricStudySpec:
        return ParametricStudySpec.from_params([row.param() for row in self.rows], mode=self.mode.currentData(),
                                               max_points=self.max_points.value())

    def update_estimate(self):
        st = self.win.study
        if st is None:
            self.est.setText("Open a study first.")
            return
        try:
            spec = self.spec()
            n = spec.total()
            errs = check_spec(st, spec) if spec.all_params() else ["Enter the values of at least one parameter."]
        except SweepError as e:
            self.est.setText(str(e))
            return
        if errs:
            self.est.setText("; ".join(errs[:3]))
            return
        load_only = all(p.kind in ("load_factor", "case_factor") for p in spec.all_params())
        lim = min(n, self.max_points.value())
        if load_only and _linear_scalable(st):
            self.est.setText(f"{lim} design points – linear study with load factors only: one analysis, the points are scaled exactly.")
        else:
            self.est.setText(f"{lim} design points = {lim} analyses" + (f" (of {n} combinations)" if n > lim else "")
                             + ". The mesh is reused while the mesh size does not change.")

    # ------------------------------------------------------------------ calculation
    def run(self):
        from .workers import TaskThread
        win = self.win
        if win.study is None or self.thread is not None:
            return
        if win.busy():
            QMessageBox.information(self, self.windowTitle(), "Meshing or analysis in progress – wait for it to finish.")
            return
        try:
            spec = self.spec()
        except SweepError as e:
            QMessageBox.warning(self, self.windowTitle(), str(e))
            return
        errs = check_spec(win.study, spec) if spec.all_params() else ["Enter the values of at least one parameter."]
        if errs:
            QMessageBox.warning(self, self.windowTitle(), "\n".join(errs))
            return
        errors, _ = win.check()
        if errors:
            QMessageBox.warning(self, self.windowTitle(), "The study is not ready:\n\n" + "\n".join(errors))
            return
        st = win.study_for_solve()
        st.geometry = win.geometry_str()
        parts = win.solid_parts() if win.parts else None
        wd = win._workdir / f"param_{int(time.time() * 1000)}"
        total = min(spec.total(), spec.max_points)
        self._total = total

        def job(t):
            from fea_core.sweep import run_parametric_study
            return run_parametric_study(st, spec, wd, parts=parts, cancel=t.cancel, on_log=t.log.emit,
                                        on_point=t.outcome.emit)
        self.points = []
        self.result = None
        self._keys = list(spec.combinations()[0].keys())
        self._fill_results()
        t = TaskThread(job, self)
        t.log.connect(win.log_msg)
        t.outcome.connect(self.on_point)
        t.done.connect(self.on_done)
        t.failed.connect(self.on_failed)
        t.finished.connect(self.on_finished)
        self.thread = t
        win.register_aux(t)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_apply.setEnabled(False)
        self.progress.start(f"Point 1/{total}", 1.0 / max(total, 1))
        win.log_msg(f"Parametric study: {total} design points")
        t.start()

    def cancel(self):
        if self.thread is not None:
            self.thread.cancel.set()
            self.win.log_msg("Cancelling the parametric study…")

    def on_point(self, dp):
        self.points.append(dp)
        self._fill_results()
        n = max(self._total, 1)
        k = len(self.points)
        if k < n:
            self.progress.report(k / n, (k + 1) / n, f"Point {k + 1}/{n}")

    def on_done(self, res):
        self.result = res
        self.points = list(res.points)
        self.win.param_study = res                   # Word report (section Parametric study)
        self._fill_results()
        ok = sum(p.converged for p in res.points)
        txt = f"{ok} of {len(res.points)} points solved"
        if res.linear_shortcut:
            txt += " (one linear analysis, scaled)"
        lines = [escape(txt)] + [f"<span style='color:#b36b00'>! {escape(str(w))}</span>" for w in res.warnings[:6]]
        self.note.setText("<br>".join(lines))
        self.progress.finish("cancelled" if res.cancelled else "done", "Cancelled" if res.cancelled else "Parametric study done")
        self.win.log_msg("Parametric study: " + txt)
        self.win.messages.add("info", "results", "Parametric study: " + txt, "Solve")

    def on_failed(self, text):
        self.progress.finish("error", "Parametric study failed")
        self.note.setText(f"<span style='color:#d23b3b'>{escape(text)}</span>")
        self.win.log_msg(f"Parametric study ERROR: {text}")

    def on_finished(self):
        self.win.unregister_aux(self.thread)
        self.thread = None
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_apply.setEnabled(self._selected() is not None)

    # ------------------------------------------------------------------ table, chart
    def _columns(self):
        return [(a, t) for a, t in kpi_columns()
                if any(getattr(p, a, None) is not None and getattr(p, a) != math.inf for p in self.points if p.converged)]

    def _fill_results(self):
        from fea_core.report.param import param_label
        keys = list(getattr(self.result, "keys", None) or getattr(self, "_keys", None) or
                    (list(self.points[0].params) if self.points else []))
        self._keys = keys
        cols = self._columns()
        pareto = set(self.result.pareto_indices) if self.result is not None else set()
        t = self.rtab
        t.setColumnCount(2 + len(keys) + len(cols))
        t.setHorizontalHeaderLabels(["DP"] + [param_label(k) for k in keys] + [c for _, c in cols] + ["Status"])
        t.setRowCount(len(self.points))
        for r, p in enumerate(self.points):
            vals = [f"{p.index}{' *' if p.index in pareto else ''}"]
            vals += [fmt(v, 5) if isinstance(v, (int, float)) else str(v) for v in (p.params.get(k) for k in keys)]
            vals += [fmt(getattr(p, a), 5) if p.converged else "–" for a, _ in cols]
            vals.append("OK" if p.converged else p.error[:120])
            for c, txt in enumerate(vals):
                it = QTableWidgetItem(txt)
                it.setTextAlignment((Qt.AlignLeft if c == len(vals) - 1 else Qt.AlignRight) | Qt.AlignVCenter)
                t.setItem(r, c, it)
        t.setProperty("_i18n_done", True)
        cur = self.x_axis.currentData()
        self.x_axis.blockSignals(True)
        self.x_axis.clear()
        for k in keys:
            self.x_axis.addItem(param_label(k), ("param", k))
        for a, txt in kpi_columns():
            self.x_axis.addItem(txt, ("kpi", a))
        i = self.x_axis.findData(cur) if cur is not None else 0
        self.x_axis.setCurrentIndex(max(i, 0))
        self.x_axis.blockSignals(False)
        if self.y_axis.currentIndex() < 0:
            self.y_axis.setCurrentIndex(0)
        self._plot()

    def _xy(self, p):
        kind, key = self.x_axis.currentData() or ("param", None)
        x = p.params.get(key) if kind == "param" else getattr(p, key, None)
        y = getattr(p, self.y_axis.currentData() or "max_stress", None)
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return None
        return (x, y) if math.isfinite(x) and math.isfinite(y) else None

    def _plot(self):
        pts = [(p, self._xy(p)) for p in self.points if p.converged]
        pts = [(p, xy) for p, xy in pts if xy is not None]
        if not pts and self.plot_w.canvas is None:
            return
        ax = self.plot_w.axes()
        pareto = set(self.result.pareto_indices) if self.result is not None else set()
        if pts:
            xs = [xy[0] for _, xy in pts]
            ys = [xy[1] for _, xy in pts]
            ax.plot(xs, ys, "o", color="#2563eb", ms=5)
            for p, (x, y) in pts:
                if p.index in pareto:
                    ax.plot([x], [y], "o", color="#e69a1a", ms=8, mfc="none", mew=1.6)
                ax.annotate(str(p.index), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(self.x_axis.currentText())
        ax.set_ylabel(self.y_axis.currentText())
        ax.grid(True, alpha=0.3)
        self.plot_w.canvas.draw_idle()

    def _selected(self):
        r = self.rtab.currentRow()
        return self.points[r] if 0 <= r < len(self.points) and self.rtab.selectedItems() else None

    def apply_selected(self):
        dp = self._selected()
        win = self.win
        if dp is None or win.study is None:
            return
        if win.busy():
            QMessageBox.information(self, self.windowTitle(), "Meshing or analysis in progress – wait for it to finish.")
            return
        try:
            new = apply_params(win.study, dp.params)
        except SweepError as e:
            QMessageBox.warning(self, self.windowTitle(), str(e))
            return
        st = win.study                                 # in place: undo / redo and the panels keep the same study object
        st.load_cases, st.materials, st.part_materials = new.load_cases, new.materials, new.part_materials
        st.mesh, st.printing = new.mesh, new.printing
        win.changed()
        win.log_msg(f"Parametric study: design point {dp.index} taken over – run the analysis")
        self.note.setText(f"Design point {dp.index} taken over into the study – run the analysis (Solve step).")

    def closeEvent(self, e):
        if self.thread is not None:
            self.cancel()
            self.thread.wait(15000)
        super().closeEvent(e)
