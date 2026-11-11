""""Progress along a line" dialog in the results (fea_core.paths): values of a quantity along the segment A → B (plot, line in 3D),
linearisation of the stress on the segment through the wall (membrane + bending + peak, von Mises and Tresca, ASME VIII-2 5-A) and CSV export.
The points are entered as numbers or by a click on the model in 3D (the point is converted to the undeformed position).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QPushButton, QSpinBox, QTableWidgetItem, QVBoxLayout)

from . import theme
from .widgets import LazyPlot, dspin, fmt, table

# quantities of the mechanical state: key fea_core.paths.line_values → (description, unit)
FIELDS = [("mises", "von Mises", "MPa"), ("s1", "1st principal stress S1", "MPa"), ("s2", "2nd principal stress S2", "MPa"),
          ("s3", "3rd principal stress S3", "MPa"), ("sx", "stress σx", "MPa"), ("sy", "stress σy", "MPa"),
          ("sz", "stress σz", "MPa"), ("sxy", "stress τxy", "MPa"), ("syz", "stress τyz", "MPa"),
          ("szx", "stress τzx", "MPa"), ("umag", "displacement |u|", "mm"), ("ux", "displacement ux", "mm"), ("uy", "displacement uy", "mm"),
          ("uz", "displacement uz", "mm")]
EXTRA = {"stress_error": ("stress error estimate", "%"), "print_safety": ("3D printing – safety factor", "–"),
         "print_index": ("3D printing – failure index", "–")}
LIN_ROWS = [("Membrane P_m", lambda d: d["membrane"]),
            ("P_m + P_b – end A", lambda d: d["ends"]["start"]["membrane_bending"]),
            ("P_m + P_b – end B", lambda d: d["ends"]["end"]["membrane_bending"]),
            ("Peak F – end A", lambda d: d["ends"]["start"]["peak"]),
            ("Peak F – end B", lambda d: d["ends"]["end"]["peak"]),
            ("Total – end A", lambda d: d["ends"]["start"]["total"]),
            ("Total – end B", lambda d: d["ends"]["end"]["total"])]
C_LINE = (20, 120, 220)


class PathDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Line profile and stress linearization")
        self.setModal(False)
        self.resize(640, 900)
        self.data = None                      # result of line_values
        self.lin = None                       # result of linearize
        self._picking = None                  # "A" | "B" – waiting for a click in 3D
        lay = QVBoxLayout(self)
        intro = QLabel("Values of the quantity along segment A → B (interpolated in mesh elements, points outside the body are omitted). Stress linearization: segment through the wall thickness (SCL) – both ends on the surface.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        f = QFormLayout()
        self.pts = {}
        self.pick_btn = {}
        for key in ("A", "B"):
            h = QHBoxLayout()
            sp = [dspin(0.0, -1e7, 1e7, 4) for _ in range(3)]
            for s, ax in zip(sp, "xyz"):
                s.setToolTip(f"{ax} coordinate of point {key} [mm] in the undeformed position of the model")
                s.valueChanged.connect(self._points_changed)
                h.addWidget(s)
            b = QPushButton("Click in 3D")
            b.setCheckable(True)
            b.setToolTip(f"Point {key} = point clicked on the model in the 3D window (surface or section; converted to the undeformed position).\nClicking the button again cancels the selection.")
            b.clicked.connect(lambda on, k=key: self.start_pick(k, on))
            h.addWidget(b)
            self.pts[key] = sp
            self.pick_btn[key] = b
            f.addRow(f"Point {key} [mm]", h)
        hb = QHBoxLayout()
        b = QPushButton("Swap A ↔ B")
        b.setToolTip("Swap the start and end of the segment (direction of the s axis in the plot and the linearization ends)")
        b.clicked.connect(self.swap)
        hb.addWidget(b)
        b = QPushButton("A = place of the maximum")
        b.setToolTip("Point A = node with the highest value of the displayed quantity (“max” marker in 3D) – default start of a path from the stress concentration into the material")
        b.clicked.connect(self.a_from_max)
        hb.addWidget(b)
        hb.addStretch(1)
        f.addRow(hb)
        self.step = QComboBox()
        self.step.setToolTip("Load case (or thermal case “Heat: …”) whose results are evaluated")
        self.step.currentIndexChanged.connect(lambda _: self._fill_fields())
        f.addRow("State", self.step)
        self.field = QComboBox()
        self.field.setToolTip("Quantity along the line: stress (von Mises, principal, components in global axes), displacements,\nfor a thermal case temperature, optionally the stress error estimate and 3D print safety factor")
        f.addRow("Quantity", self.field)
        self.n = QSpinBox()
        self.n.setRange(3, 1001)
        self.n.setValue(51)
        self.n.setToolTip("Number of evaluation points evenly spaced on the segment (more = smoother plot, takes longer)")
        f.addRow("Number of points", self.n)
        lay.addLayout(f)
        h = QHBoxLayout()
        self.btn_plot = QPushButton("Plot profile")
        theme.set_primary(self.btn_plot)
        self.btn_plot.setToolTip("Compute the values along segment A → B, draw the plot and the line in 3D")
        self.btn_plot.clicked.connect(self.plot)
        h.addWidget(self.btn_plot)
        self.btn_lin = QPushButton("Stress linearization")
        self.btn_lin.setToolTip("Decomposition of the stress on a segment through the wall into membrane P_m, bending P_b and peak F\n(ASME BPVC VIII-2, Annex 5-A, stress integration); von Mises and Tresca equivalents.\nThe entire segment must lie in the material.")
        self.btn_lin.setWhatsThis("<b>Stress linearization</b><p>An SCL (stress classification line) segment running through the wall thickness (both ends on the surface). Membrane stress P<sub>m</sub> = average through the thickness, bending P<sub>b</sub> = linear part (only components normal to the segment), peak F = remainder at the ends. Evaluated per ASME VIII-2 / EN 13445-3 (P<sub>m</sub> ≤ S, P<sub>m</sub> + P<sub>b</sub> ≤ 1.5 S).</p><p>There should be at least 3–4 elements through the thickness, otherwise the program warns.</p>")
        self.btn_lin.clicked.connect(self.linearize)
        h.addWidget(self.btn_lin)
        self.btn_csv = QPushButton("Export CSV…")
        self.btn_csv.setToolTip("Save the line plot (point, s, x, y, z, value) and the linearization to a CSV file")
        self.btn_csv.clicked.connect(self.export_dialog)
        h.addWidget(self.btn_csv)
        lay.addLayout(h)
        self.plot_w = LazyPlot()
        self.plot_w.setToolTip("Profile of the quantity along the segment: s = distance from point A [mm]")
        lay.addWidget(self.plot_w, 1)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.lin_table = table(["", "von Mises [MPa]", "Tresca [MPa]"], 235)
        self.lin_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.lin_table.setToolTip("Stress linearization: membrane P_m, membrane + bending P_m + P_b at both ends,\npeak F and total stress at the ends – von Mises and Tresca equivalents [MPa]")
        lay.addWidget(self.lin_table)
        self.lin_info = QLabel("")
        self.lin_info.setWordWrap(True)
        lay.addWidget(self.lin_info)
        bb = QHBoxLayout()
        bb.addStretch(1)
        b = QPushButton("Close")
        b.setToolTip("Close the dialog (the line in 3D disappears)")
        b.clicked.connect(self.close)
        bb.addWidget(b)
        lay.addLayout(bb)
        self.refresh()

    # ------------------------------------------------------------------ obsah
    def refresh(self):
        """Menu of the states according to the results of the main window (default = the displayed state)."""
        res = self.win.results
        cur = self.step.currentData()
        self.step.blockSignals(True)
        self.step.clear()
        if res is not None:
            for i, s in enumerate(res.steps):
                self.step.addItem(s.name, ("step", i))
            for t in res.thermal:
                self.step.addItem(f"Thermal: {t.name}", ("thermal", t.name))
            if res.twod is not None:
                for i, s2 in enumerate(res.twod.steps):
                    self.step.addItem(f"2D: {s2['name']}", ("twod", i))
        want = cur
        if want is None:
            c = self.win.current_result()
            if c and c[0] == "step":
                want = ("step", c[1])
            elif c and c[0] == "thermal":
                want = ("thermal", res.thermal[c[1]].name)
            elif c and c[0] == "twod":
                want = ("twod", c[1])
        k = next((i for i in range(self.step.count()) if self.step.itemData(i) == want), 0)
        self.step.setCurrentIndex(k)
        self.step.blockSignals(False)
        self._fill_fields()
        if res is not None and all(s.value() == 0 for s in self.pts["A"] + self.pts["B"]):
            X = res.mesh.nodes if len(res.mesh.nodes) else res.twod.frame2d().to3d(res.twod.nodes)
            lo, hi = X.min(axis=0), X.max(axis=0)                                  # default: the diagonal of the model
            c = 0.5 * (lo + hi)
            ax = int(np.argmax(hi - lo))
            a, b = c.copy(), c.copy()
            a[ax], b[ax] = lo[ax], hi[ax]
            self.set_point("A", a)
            self.set_point("B", b)

    def _fill_fields(self):
        cur = self.field.currentData()
        self.field.blockSignals(True)
        self.field.clear()
        d = self.step.currentData()
        res = self.win.results
        if d and d[0] == "thermal":
            self.field.addItem("temperature [°C]", ("T", "temperature", "°C"))
        elif d and d[0] == "twod" and res is not None:
            from .twod_gui import quantities
            for name, unit, k in quantities(res.twod):
                self.field.addItem(f"{name} [{unit}]", (k, name, unit))
        elif d and res is not None:
            for k, name, unit in FIELDS:
                self.field.addItem(f"{name} [{unit}]", (k, name, unit))
            st = res.steps[d[1]]
            for k, (name, unit) in EXTRA.items():
                if k in st.fields:
                    self.field.addItem(f"{name} [{unit}]", (k, name, unit))
        i = next((i for i in range(self.field.count()) if cur and self.field.itemData(i)[0] == cur[0]), 0)
        self.field.setCurrentIndex(i)
        self.field.blockSignals(False)
        self.btn_lin.setEnabled(bool(d) and d[0] in ("step", "twod"))

    def point(self, key) -> np.ndarray:
        return np.array([s.value() for s in self.pts[key]])

    def set_point(self, key, p):
        for s, v in zip(self.pts[key], p):
            s.blockSignals(True)
            s.setValue(float(v) + 0.0)
            s.blockSignals(False)
        self._points_changed()

    def swap(self):
        a, b = self.point("A"), self.point("B")
        self.set_point("A", b)
        self.set_point("B", a)

    def a_from_max(self):
        pr = getattr(self.win, "_probe", None)
        if pr is None:
            QMessageBox.information(self, self.windowTitle(), "Display a result in the Results step.")
            return
        tree, ids, vals, _, _ = pr
        j = int(np.nanargmax(vals))
        self.set_point("A", self.win.undeformed(tree.data[j]))

    def _points_changed(self, *_):
        self.draw_line()

    # ------------------------------------------------------------------ point selection in 3D
    def start_pick(self, key, on=True):
        for k, b in self.pick_btn.items():
            if k != key:
                b.setChecked(False)
        if not on:
            self.stop_pick()
            return
        if self.win.results is None:
            self.pick_btn[key].setChecked(False)
            return
        if self.win.STAGES[self.win.stage] != "Results":
            self.win.set_stage(self.win.STAGE["Results"])
        self._picking = key
        self.win._pick_hook = self.on_pick
        self.win.statusBar().showMessage(f"Click the model in 3D – point {key} of the line profile", 8000)

    def stop_pick(self):
        self._picking = None
        for b in self.pick_btn.values():
            b.setChecked(False)
        if getattr(self.win, "_pick_hook", None) == self.on_pick:
            self.win._pick_hook = None

    def on_pick(self, p):
        key = self._picking
        self.stop_pick()
        if key:
            self.set_point(key, p)

    def draw_line(self):
        """Line A → B and markers A, B in 3D (in the displayed – possibly deformed – position)."""
        win = self.win
        if win.results is None or win.STAGES[win.stage] != "Results" or not self.isVisible():
            return
        a, b = self.point("A"), self.point("B")
        pts = [a + t * (b - a) for t in np.linspace(0.0, 1.0, 21)]
        disp = [win.displayed(p) for p in pts]
        win.view.result_polyline("path_line", disp, C_LINE, 3.5)
        win.view.mark(disp[0], "A", C_LINE, key="path_a")
        win.view.mark(disp[-1], "B", C_LINE, key="path_b")
        win.view._update()

    def closeEvent(self, e):
        self.stop_pick()
        v = self.win.view
        for k in ("path_line", "path_a", "path_b"):
            for o in v._marks.pop(k, []):
                v.ctx.Remove(o, False)
                if o in v._result_objs:
                    v._result_objs.remove(o)
        v._update()
        super().closeEvent(e)

    # ------------------------------------------------------------------ calculation
    def _step_key(self):
        d = self.step.currentData()
        return d[1] if d else 0

    def plot(self):
        from fea_core.paths import line_values
        res = self.win.results
        if res is None or self.field.currentData() is None:
            return
        a, b = self.point("A"), self.point("B")
        if np.linalg.norm(b - a) < 1e-9:
            QMessageBox.information(self, self.windowTitle(), "Points A and B coincide – enter two different points.")
            return
        key, name, unit = self.field.currentData()
        d = self.step.currentData()
        try:
            if d and d[0] == "twod":                       # 2D: points in the section plane → coordinates of the section
                fr = res.twod.frame2d()
                self.data = res.twod.line_values(fr.to2d(a), fr.to2d(b), key, d[1], int(self.n.value()))
                self.data["points"] = fr.to3d(np.asarray(self.data["points"], float)).tolist()
            else:
                self.data = line_values(res, a, b, key, self._step_key(), int(self.n.value()))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, self.windowTitle(), f"The profile cannot be computed:\n{e}")
            return
        self.data["label"] = (name, unit)
        s = np.asarray(self.data["s"])
        v = np.asarray(self.data["values"], float)
        ax = self.plot_w.axes()
        ax.plot(s, v, "-", color="#2563eb", lw=1.8)
        ax.set_xlabel("s from point A [mm]")
        ax.set_ylabel(f"{name} [{unit}]")
        ax.grid(True, alpha=0.3)
        self.plot_w.canvas.draw_idle()
        ok = np.isfinite(v)
        if ok.any():
            i, j = int(np.nanargmax(v)), int(np.nanargmin(v))
            txt = (f"Length {fmt(float(s[-1]), 5)} mm; max {fmt(float(v[i]), 5)} {unit} at s = {fmt(float(s[i]), 4)} mm, min {fmt(float(v[j]), 5)} {unit} at s = {fmt(float(s[j]), 4)} mm")
            if not ok.all():
                txt += f"; {int((~ok).sum())} points outside the body (omitted)"
        else:
            txt = "The entire segment lies outside the body – check points A and B (undeformed position)."
        self.info.setText(txt)
        self.draw_line()
        self.win.log_msg(f"Line profile: {name} – {txt}")

    def linearize(self):
        from fea_core.paths import linearize
        res = self.win.results
        d = self.step.currentData()
        if res is None or not d or d[0] not in ("step", "twod"):
            return
        try:
            if d[0] == "twod":
                fr = res.twod.frame2d()
                self.lin = res.twod.linearize(fr.to2d(self.point("A")), fr.to2d(self.point("B")), d[1])
            else:
                self.lin = linearize(res, self.point("A"), self.point("B"), d[1])
        except Exception as e:  # noqa: BLE001 – a segment outside the body
            self.lin = None
            self.lin_table.setRowCount(0)
            self.lin_info.setText(f"<span style='color:#d23b3b'>{e}</span>")
            return
        t = self.lin_table
        t.setRowCount(len(LIN_ROWS))
        for r, (lbl, get) in enumerate(LIN_ROWS):
            q = get(self.lin)
            for c, txt in enumerate((lbl, fmt(q["mises"], 5), fmt(q["tresca"], 5))):
                it = QTableWidgetItem(txt)
                if c:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(r, c, it)
        rows = [f"Thickness (SCL length) {fmt(self.lin['length_mm'], 5)} mm; P_m = {fmt(self.lin['Pm_mises'], 5)} MPa, P_m + P_b = {fmt(self.lin['PmPb_mises'], 5)} MPa (larger of the ends, von Mises)."]
        if self.lin.get("warning"):
            rows.append(f"<span style='color:#b36b00'>! {self.lin['warning']}</span>")
            self.win.messages.add("warning", "results", "Linearizace: " + self.lin["warning"], "Mesh")
        rows.append("<i>Nodal stresses are averaged; bending only from components normal to the segment (ASME VIII-2, 5-A).</i>")
        self.lin_info.setText("<br>".join(rows))
        self.win.log_msg(f"Stress linearization: P_m {fmt(self.lin['Pm_mises'], 5)} MPa, P_m + P_b {fmt(self.lin['PmPb_mises'], 5)} MPa, total {fmt(self.lin['total_mises'], 5)} MPa")

    def export_dialog(self):
        if self.data is None and self.lin is None:
            QMessageBox.information(self, self.windowTitle(), "Plot the profile or compute the linearization first.")
            return
        last = self.win.settings.value("last_dir", "")
        p, _ = QFileDialog.getSaveFileName(self, "Export CSV", str(Path(last) / "prubeh.csv") if last else "prubeh.csv",
                                           "CSV (*.csv)")
        if p:
            self.export_csv(p)

    def export_csv(self, path):
        """Course and result of the linearisation to CSV (format as in the results export dialog – Czech for Excel: semicolon,
        decimal comma): point = point number from 1, inside = 1 point in the material / 0 outside the body (value empty).
        """
        from fea_core.export_csv import write_csv
        from fea_core.i18n import tr
        from .export_dialog import csv_dialect
        rows = []
        if self.data is not None:
            name, unit = self.data.get("label", (self.data["field"], ""))
            rows.append(["point", "s [mm]", "x [mm]", "y [mm]", "z [mm]", f"{tr(name)} [{unit}]", "inside"])
            for k, (s, p, v) in enumerate(zip(self.data["s"], self.data["points"],
                                              np.asarray(self.data["values"], float))):
                rows.append([k + 1, s, *p, v, int(bool(np.isfinite(v)))])
        if self.lin is not None:
            if rows:
                rows.append([])
            rows.append([tr("Stress linearization"), "von Mises [MPa]", "Tresca [MPa]"])
            for lbl, get in LIN_ROWS:
                q = get(self.lin)
                rows.append([tr(lbl), q["mises"], q["tresca"]])
            rows.append([tr("SCL length [mm]"), self.lin["length_mm"]])
        write_csv(path, None, rows, csv_dialect(self.win))
        self.win.log_msg(f"Line profile saved to {path}")
        return path
