""""Print detail (voxels)" dialog in the results of a 3D print (fea_core.printing.voxel): region (centre by a click in 3D
or the place of the lowest print safety, size, margin, resolution w/2 | w/4, state, element), estimate of the size before the calculation,
calculation in a thread with cancellation, image of the section through a layer (slider, bead direction, field), table of the summary by states
and the box of the region in 3D. The result is INDICATIVE (paths from the print settings, not from G-code).
"""
from __future__ import annotations

import html
import tempfile
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QPushButton, QScrollArea, QSlider, QSplitter, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from . import theme
from .widgets import TaskProgress, dspin, fmt, table

FIELDS = [("kind", "voxel kind (perimeter / layer / infill / thin region)"), ("safety", "safety factor (Tsai-Wu / Hill)"),
          ("angle", "line angle [°]"), ("mises", "von Mises [MPa]"), ("umag", "displacement [mm]"),
          ("s11", "σ11 along bead [MPa]"), ("s22", "σ22 across bead [MPa]"), ("s33", "σ33 normal to layers [MPa]"),
          ("t12", "τ12 [MPa]"), ("t13", "τ13 [MPa]"), ("t23", "τ23 [MPa]")]
WARN = ("INDICATIVE: bead paths are generated from the print settings (not from a slicer G-code), boundary conditions from the homogenized model, solid-extrusion material. The minimum at the integration point decreases with refinement (voxel steps and bead corners are singularities) – evaluate the safety factor from the element average (safety_avg) and the 1st percentile.")
C_BOX = (230, 120, 20)


class VoxelDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Print detail (voxels)")
        self.setModal(False)
        self.resize(1000, 1000)
        self.thread = None
        self.vr = None
        self.est = None
        self._tmp = Path(tempfile.mkdtemp(prefix="fea_voxel_"))
        lay = QVBoxLayout(self)
        intro = QLabel("Local voxel model of beads (perimeters, layers, infill) around the critical location of the printed part: hex8 C3D8I elements with solid-extrusion material in the bead axes, boundary displacements from the whole-part result.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        warn = QLabel(WARN)
        warn.setWordWrap(True)
        warn.setStyleSheet("color:#d23b3b")
        lay.addWidget(warn)
        f = QFormLayout()
        h = QHBoxLayout()
        self.c = [dspin(0.0, -1e7, 1e7, 3) for _ in range(3)]
        for s, ax in zip(self.c, "xyz"):
            s.setToolTip(f"{ax} coordinate of the detail region center [mm] (global)")
            h.addWidget(s)
        b = QPushButton("Click in 3D")
        b.setCheckable(True)
        b.setToolTip("Center = point clicked on the model in the results (undeformed position)")
        b.clicked.connect(self.start_pick)
        self.btn_pick = b
        h.addWidget(b)
        b = QPushButton("Lowest safety factor")
        b.setToolTip("Center = node with the lowest print safety factor of the homogenized model (all cases)")
        b.clicked.connect(self.auto_center)
        h.addWidget(b)
        f.addRow("Center [mm]", h)
        self.auto = QCheckBox("automatic (location of lowest safety factor during analysis)")
        self.auto.setChecked(True)
        self.auto.setToolTip("On = the center is determined during the analysis from the lowest print safety factor of the selected case")
        self.auto.toggled.connect(lambda on: [s.setEnabled(not on) for s in self.c])
        for s in self.c:
            s.setEnabled(False)
        f.addRow(self.auto)
        h = QHBoxLayout()
        self.sz = [dspin(v, 0.2, 100.0, 2, 0.5, " mm") for v in (6.0, 6.0, 4.0)]
        for s, t in zip(self.sz, ("print axis 1", "print axis 2", "build direction")):
            s.setToolTip(f"Size of the evaluation region in direction {t} [mm] (part print axes)")
            h.addWidget(s)
        f.addRow("Size [mm]", h)
        self.margin = dspin(1.0, 0.0, 20.0, 2, 0.25, " mm")
        self.margin.setToolTip("Margin around the region [mm] – carries the boundary conditions from the global solution, not included in the evaluation (≥ 2–3 line widths)")
        f.addRow("Margin", self.margin)
        self.res = QComboBox()
        self.res.addItem("w/2 (voxel = half the line width)", "w/2")
        self.res.addItem("w/4 (finer, ~8× more voxels)", "w/4")
        self.res.setToolTip("Resolution in the layer plane: voxel = line width / 2 or / 4; in the build direction = layer height")
        f.addRow("Resolution", self.res)
        self.step = QComboBox()
        self.step.setToolTip("Load case (all = one analysis, one step per case)")
        f.addRow("State", self.step)
        self.elem = QComboBox()
        self.elem.addItem("C3D8I (incompatible modes – accurate bead bending)", "C3D8I")
        self.elem.addItem("C3D8 (cheaper, locks in bending of thin beads)", "C3D8")
        self.elem.setToolTip("Voxel element: C3D8I ≈ 12 equations per voxel (recommended), C3D8 ≈ 3 equations per voxel")
        f.addRow("Element", self.elem)
        lay.addLayout(f)
        h = QHBoxLayout()
        b = QPushButton("Estimate size")
        b.setToolTip("Estimate voxel counts per kind, equations, memory and ccx time in advance (without running)")
        b.clicked.connect(self.estimate)
        h.addWidget(b)
        self.btn_run = QPushButton("Run detail")
        theme.set_primary(self.btn_run)
        self.btn_run.setToolTip("Generate toolpaths, boundary conditions and compute the voxel detail in CalculiX (in the background)")
        self.btn_run.clicked.connect(self.run)
        h.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip("Cancel detail analysis")
        self.btn_cancel.clicked.connect(self.cancel)
        h.addWidget(self.btn_cancel)
        self.box3d = QCheckBox("region box in 3D")
        self.box3d.setChecked(True)
        self.box3d.setToolTip("Orange: evaluation region of the detail in the 3D results window")
        self.box3d.toggled.connect(lambda _: self.draw_box())
        h.addWidget(self.box3d)
        h.addStretch(1)
        lay.addLayout(h)
        self.est_lbl = QLabel("")
        self.est_lbl.setWordWrap(True)
        lay.addWidget(self.est_lbl)
        self.progress = TaskProgress()
        lay.addWidget(self.progress)
        sp = QSplitter(Qt.Vertical)
        top = QWidget()
        tv = QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        h = QHBoxLayout()
        self.field = QComboBox()
        for k, t in FIELDS:
            self.field.addItem(t, k)
        self.field.setToolTip("Field displayed per voxel (stress in the bead axes: 1 = along the line, 2 = across, 3 = build direction)")
        self.field.currentIndexChanged.connect(lambda _: self.render())
        h.addWidget(QLabel("Field"))
        h.addWidget(self.field, 1)
        self.rstep = QComboBox()
        self.rstep.setToolTip("Detail result case")
        self.rstep.currentIndexChanged.connect(lambda _: self.render())
        h.addWidget(self.rstep)
        self.lines = QCheckBox("bead direction")
        self.lines.setChecked(True)
        self.lines.setToolTip("Short dashes along the bead direction in the section (every 2nd voxel)")
        self.lines.toggled.connect(lambda _: self.render())
        h.addWidget(self.lines)
        tv.addLayout(h)
        h = QHBoxLayout()
        self.slice = QSlider(Qt.Horizontal)
        self.slice.setToolTip("Section layer (layer index within the detail region, counted from the build plate)")
        self.slice.valueChanged.connect(lambda _: self.render())
        self.slice_lbl = QLabel("layer –")
        h.addWidget(QLabel("Layer"))
        h.addWidget(self.slice, 1)
        h.addWidget(self.slice_lbl)
        tv.addLayout(h)
        self.img = QLabel("")
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setToolTip("Section of the voxel model through a layer (print axes 1, 2)")
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.img)
        tv.addWidget(sc, 1)
        sp.addWidget(top)
        bot = QWidget()
        bv = QVBoxLayout(bot)
        bv.setContentsMargins(0, 0, 0, 0)
        self.tab = table(["State", "Safety factor, element average", "1st percentile", "Min. at int. point", "Homogenized",
                          "Ratio", "Avg. σ difference [%]", "Minimum location"], 120)
        self.tab.horizontalHeader().setSectionResizeMode(7, QHeaderView.Stretch)
        self.tab.setToolTip("Summary per case: key value = lowest safety factor from the element-average stress and the 1st percentile\nof the safety factor at integration points; the minimum at an integration point is sensitive to voxel steps\n(singularities). Homogenized = min. safety factor of the whole model in the same region, ratio = detail\n/ homogenized, difference of the average stress over the region (boundary condition check).")
        bv.addWidget(self.tab)
        self.out = QLabel("")
        self.out.setWordWrap(True)
        self.out.setTextFormat(Qt.RichText)
        bv.addWidget(self.out)
        sp.addWidget(bot)
        sp.setSizes([560, 220])
        lay.addWidget(sp, 1)
        h = QHBoxLayout()
        h.addStretch(1)
        b = QPushButton("Close")
        b.setToolTip("Close the dialog (a running analysis is cancelled)")
        b.clicked.connect(self.close)
        h.addWidget(b)
        lay.addLayout(h)
        self.refresh()

    # ------------------------------------------------------------------ form
    def refresh(self):
        res = self.win.results
        self.step.clear()
        self.step.addItem("all cases", None)
        for i, s in enumerate(res.steps if res is not None else []):
            if "print_safety" in s.fields:
                self.step.addItem(s.name, i)
        cur = self.win.current_result()
        if cur and cur[0] == "step":
            k = self.step.findData(cur[1])
            if k >= 0:
                self.step.setCurrentIndex(k)

    def spec(self):
        from fea_core.printing.voxel import VoxelSpec
        return VoxelSpec(center=None if self.auto.isChecked() else [s.value() for s in self.c],
                         size=[s.value() for s in self.sz], resolution=self.res.currentData(),
                         step=self.step.currentData(), margin=self.margin.value(), element=self.elem.currentData())

    def start_pick(self, on=True):
        if not on:
            if getattr(self.win, "_pick_hook", None) == self.on_pick:
                self.win._pick_hook = None
            return
        if self.win.STAGES[self.win.stage] != "Results":
            self.win.set_stage(self.win.STAGE["Results"])
        self.win._pick_hook = self.on_pick
        self.win.statusBar().showMessage("Click the printed part in 3D – center of the voxel detail", 8000)

    def on_pick(self, p):
        self.btn_pick.setChecked(False)
        self.win._pick_hook = None
        self.auto.setChecked(False)
        for s, v in zip(self.c, p):
            s.setValue(float(v) + 0.0)
        self.draw_box()

    def auto_center(self):
        from fea_core.printing.voxel import auto_center
        try:
            x, part, case, lam = auto_center(self.win.study, self.win.results, step=self.step.currentData())
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, self.windowTitle(), str(e))
            return
        self.auto.setChecked(False)
        for s, v in zip(self.c, x):
            s.setValue(float(v))
        self.est_lbl.setText(f"Lowest print safety factor {fmt(lam, 4)} – part {html.escape(part)}, case {html.escape(case)}")
        self.draw_box()

    def estimate(self):
        from fea_core.printing.voxel import estimate
        try:
            e = estimate(self.win.study, self.win.results, self.spec())
        except Exception as ex:  # noqa: BLE001
            self.est_lbl.setText(f"<span style='color:#d23b3b'>{html.escape(str(ex))}</span>")
            self.est = None
            return None
        self.est = e
        kinds = ", ".join(f"{k} {v:,}".replace(",", " ") for k, v in e.get("kinds", {}).items())
        col = "#2e9e4f" if e["ok"] else "#d23b3b"
        self.est_lbl.setText(
            f"<b><span style='color:{col}'>{('can be computed' if e['ok'] else 'NOT POSSIBLE')}</span></b> – part {html.escape(str(e['part']))}: {e['voxels']:,} voxels ({kinds}), {e['nodes']:,} nodes, ".replace(",", " ")
            + f"{e['dof']:,} equations".replace(",", " ")
            + f", memory ~{fmt(e['memory_gb'], 3)} GB, time ~{fmt(e['time_s'], 3)} s, voxel "
            + " × ".join(fmt(float(v), 3) for v in e["voxel_mm"]) + " mm"
            + (f"<br><span style='color:#d23b3b'>{html.escape(e['message'])}</span>" if e.get("message") else ""))
        if e.get("center") is not None and self.auto.isChecked():
            for s, v in zip(self.c, e["center"]):
                s.setValue(float(v))
        self.draw_box(e)
        return e

    # ------------------------------------------------------------------ calculation
    def run(self):
        from .workers import TaskThread
        win = self.win
        if self.thread is not None:
            return
        e = self.estimate()
        if e is None or not e["ok"]:
            return
        spec = self.spec()
        st, res = win.study, win.results
        wd = win._workdir / f"voxel_{int(time.time() * 1000)}"

        def job(t):
            from fea_core.printing.voxel import run_voxel
            return run_voxel(st, res, spec, wd, on_log=t.log.emit, cancel=t.cancel)
        t = TaskThread(job, self)
        t.log.connect(win.log_msg)
        t.done.connect(self.on_done)
        t.failed.connect(self.on_failed)
        t.finished.connect(self.on_finished)
        self.thread = t
        win.register_aux(t)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.start("Voxel detail (toolpaths, CalculiX C3D8I)", 0.95)
        t.start()

    def cancel(self):
        if self.thread is not None:
            self.thread.cancel.set()
            self.win.log_msg("Cancelling voxel detail…")

    def on_failed(self, text):
        cancelled = self.thread is not None and self.thread.cancel.is_set()
        self.progress.finish("cancelled" if cancelled else "error", "Cancelled" if cancelled else "Detail failed")
        self.out.setText(f"<span style='color:#d23b3b'>{html.escape(text)}</span>")
        self.win.log_msg(f"VOXEL DETAIL ERROR: {text}")
        if not cancelled:
            self.win.messages.add("error", "solver", f"Voxel detail failed: {text}", "Results")

    def on_finished(self):
        self.win.unregister_aux(self.thread)
        self.thread = None
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def on_done(self, vr):
        self.vr = vr
        self.progress.finish("done", "Detail finished")
        sm = vr.summary
        self.rstep.blockSignals(True)
        self.rstep.clear()
        for i, s in enumerate(vr.steps):
            self.rstep.addItem(s.name, i)
        self.rstep.blockSignals(False)
        nz = int(vr.shape[2])
        self.slice.blockSignals(True)
        self.slice.setRange(0, max(nz - 1, 0))
        self.slice.setValue(nz // 2)
        self.slice.blockSignals(False)
        t = self.tab
        t.setRowCount(len(sm.get("steps", [])))
        for r, d in enumerate(sm.get("steps", [])):
            vals = [d["name"], fmt(d.get("min_safety_element_avg"), 4), fmt(d.get("safety_p1"), 4),
                    fmt(d.get("min_safety"), 4) + " *", fmt(d.get("homogenized_min_safety"), 4),
                    fmt(d.get("ratio"), 3), fmt(100 * d["mean_stress_difference"], 3)
                    if d.get("mean_stress_difference") is not None else "–",
                    f"{d.get('kind', '')}, linie {fmt(d.get('angle_deg'), 3)}° (" +
                    "; ".join(fmt(float(c), 4) for c in d.get("at", [])) + ")"]
            for c, txt in enumerate(vals):
                it = QTableWidgetItem(txt)
                if 0 < c < 7:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(r, c, it)
        def th(n):                                # space as the thousands separator (only in the number, not in the text)
            return f"{n:,}".replace(",", " ")
        rows = [f"Voxels {th(sm['voxels'])}, equations {th(sm['dof'])}, element {sm['element']}, time {fmt(sm['time_s'], 3)} s; kinds: " + ", ".join(f"{k} {th(v)}" for k, v in sm["kinds"].items())
                + f"; infill fraction {fmt(100 * sm.get('infill_fraction', 0), 3)} %",
                "* minimum at the integration point – sensitive to voxel steps and bead corners (singularities), decreases with refinement; the key values are the element average and the 1st percentile."]
        for w in vr.warnings:
            rows.append(f"<span style='color:#b36b00'>! {html.escape(w)}</span>")
        rows.append(f"<span style='color:#d23b3b'>{html.escape(WARN)}</span>")
        self.out.setText("<br>".join(rows))
        self.win.log_msg("Voxel detail: " + "; ".join(
            f"{d['name']} element average {fmt(d.get('min_safety_element_avg'), 4)}, p1 {fmt(d.get('safety_p1'), 4)}"
            for d in sm.get("steps", [])))
        self.render()
        self.draw_box()

    # ------------------------------------------------------------------ display
    def render(self):
        vr = self.vr
        if vr is None:
            return
        from fea_core.printing.voxel import render_png
        k = int(self.slice.value())
        self.slice_lbl.setText(f"layer {k} / {int(vr.shape[2]) - 1}")
        path = self._tmp / f"rez_{int(time.time() * 1000)}.png"
        try:
            render_png(vr, path, field=self.field.currentData(), slice_z=k, step=max(self.rstep.currentIndex(), 0),
                       lines=self.lines.isChecked())
        except Exception as e:  # noqa: BLE001
            self.img.setText(f"Cannot render the image: {e}")
            return
        pm = QPixmap(str(path))
        self.img.setPixmap(pm)
        self.last_png = path

    def box_corners(self, est=None):
        """8 corners of the assessment region (global) from the detail result, or approximately from the centre and the size."""
        vr = self.vr
        if vr is not None and vr.summary.get("center") is not None:
            c = np.asarray(vr.summary["center"], float)
            size = np.asarray(vr.summary["size"], float)
            R = np.asarray(vr.R, float)
        else:
            c = np.asarray(est["center"] if est else [s.value() for s in self.c], float)
            size = np.array([s.value() for s in self.sz])
            R = np.eye(3)
            try:
                d = np.asarray(self.win.study.printing.spec(est["part"] if est else "").build_dir, float)
                if np.linalg.norm(d) > 0:                      # axis 3 = build direction, axes 1, 2 in the plane of the layer
                    z = d / np.linalg.norm(d)
                    x = np.cross([0, 1.0, 0], z) if abs(z[1]) < 0.9 else np.cross([1.0, 0, 0], z)
                    x /= np.linalg.norm(x)
                    R = np.array([x, np.cross(z, x), z])
            except Exception:  # noqa: BLE001
                pass
        q0 = c @ R.T - size / 2
        out = []
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    out.append((q0 + np.array([i, j, k]) * size) @ R)
        return np.array(out)

    def draw_box(self, est=None):
        win = self.win
        v = win.view
        for o in v._marks.pop("voxel_box", []):
            v.ctx.Remove(o, False)
            if o in v._result_objs:
                v._result_objs.remove(o)
        if not self.box3d.isChecked() or win.results is None or win.STAGES[win.stage] != "Results":
            v._update()
            return
        P = self.box_corners(est)
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        n0 = len(v._result_objs)
        v.result_lines([(P[a], P[b]) for a, b in edges], color=C_BOX, width=3.0, pickable=False)
        v._marks["voxel_box"] = v._result_objs[n0:]
        v._update()

    def closeEvent(self, e):
        if self.thread is not None:
            self.cancel()
            self.thread.wait(10000)
        if getattr(self.win, "_pick_hook", None) == self.on_pick:
            self.win._pick_hook = None
        super().closeEvent(e)
