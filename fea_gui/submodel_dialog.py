""""Submodel…" dialog in the Results step (fea_core.submodel_run): region box (centre + size) or sphere (centre + radius),
centre by a click in 3D / the selected row of the highest-stress table / the maximum outside singularities, element size,
calculation in a thread with progress and cancellation, list of the computed submodels. The submodel result is its own item
"Submodel: …" in the result menu (r_step data "sub:k"), drawn on its own mesh with the global parts transparent around it, and it is
saved with the results in the .fea project (results_sub.npz).
"""
from __future__ import annotations

import html
import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QTableWidgetItem, QVBoxLayout, QWidget)

from . import theme
from .widgets import TaskProgress, dspin, fmt, help_tip, table

# quantities of a submodel result: (menu name, unit, key of fea_core.submodel_run.sub_field | "cut")
QUANTS = [("von Mises", "MPa", "mises"), ("1st principal stress S1", "MPa", "s1"), ("3rd principal stress S3", "MPa", "s3"),
          ("Stress σx", "MPa", "sxx"), ("Stress σy", "MPa", "syy"), ("Stress σz", "MPa", "szz"),
          ("Total displacement", "mm", "displacement"), ("Safety factor to Re", "–", "safety"),
          ("Cut check – |Δσ_vM| to the global model", "MPa", "cut")]
QLIST = [(n, u) for n, u, _ in QUANTS]
C_BOX = (20, 150, 200)
NOTE = "Submodel: finer local model; global parts transparent for orientation, cut faces driven by the global displacement"


# ===================================================================== result menu and display (called from MainWindow)

def r_step_items(results) -> list[tuple[str, str]]:
    return [(f"Submodel: {sr.name}", f"sub:{k}") for k, sr in enumerate(getattr(results, "submodels", None) or [])]


def current(results, rest) -> tuple | None:
    subs = getattr(results, "submodels", None) or []
    try:
        k = int(rest[0])
    except (IndexError, ValueError):
        return None
    return ("sub", k) if 0 <= k < len(subs) else None


def values(win, sr, key: str) -> np.ndarray:
    if key == "cut":
        v = np.zeros(len(sr.mesh.nodes))
        v[sr.cut_nodes] = np.abs(sr.step().mises[sr.cut_nodes] - sr.glob_cut_mises)
        return v
    from fea_core.submodel_run import sub_field
    return np.asarray(sub_field(sr, key, win.study), float)


def _scale(win, sr) -> float:
    U = sr.U
    umax = float(np.linalg.norm(U, axis=1).max()) if len(U) else 0.0
    s = {0: 0.0, 2: 1.0, 3: 10.0, 4: 100.0, 5: 1000.0}.get(win.r_scale.currentIndex())
    if s is None:                                    # automatic: max. displacement = 5 % of the submodel diagonal
        diag = float(np.linalg.norm(np.ptp(sr.mesh.nodes, axis=0)))
        s = 0.05 * diag / umax if umax > 0 else 0.0
    return s


def show(win, k: int, qi: int) -> None:
    """Draws the submodel result k (quantity qi of QUANTS) in the 3D view, legend, max/min markers, probe and summary."""
    from fea_core.submodel_run import SubmodelError
    from .contour import band_edges
    from .main import NearestPoints
    sr = win.results.submodels[k]
    name, unit, key = QUANTS[min(max(qi, 0), len(QUANTS) - 1)]
    win.clear_hotspots()
    try:
        vals = values(win, sr, key)
    except SubmodelError as e:
        win.res_info.setHtml(f"<i>{html.escape(str(e))}</i>")
        win.view.clear_results()
        win.legend.clear()
        return
    m = sr.mesh
    scale = 0.0 if key == "cut" else _scale(win, sr)
    X = m.nodes + scale * sr.U
    tri, _ = m.boundary_faces()
    surf = np.unique(tri)
    sv = vals[surf]
    vmin, vmax = float(sv.min()), float(sv.max())
    tris = outline = None
    probe_pts, probe_vals = X[surf], sv
    plane = win.section_plane()
    if plane is not None:
        from fea_core.section import clip_surface, plane_section, section_outline
        o, nrm = plane
        Ps, Vs = clip_surface(m.nodes, tri, vals, o, nrm, X)
        Pc, Vc = plane_section(m.nodes, m.tets, vals, o, nrm, X)
        tris = (np.concatenate([Ps, Pc]), np.concatenate([Vs, Vc]))
        outline = section_outline(Pc)
        if len(Vc):
            vmin, vmax = min(vmin, float(Vc.min())), max(vmax, float(Vc.max()))
        kept = (m.nodes[surf] - o) @ nrm <= 0
        probe_pts = np.vstack([X[surf][kept], Pc.reshape(-1, 3)])
        probe_vals = np.concatenate([sv[kept], Vc.ravel()])
    edges = band_edges(vmin, vmax, win.r_bands.value())
    title = f"{name} [{unit}]"
    colors = win.view.show_contour(X, tri, vals, edges, title, edges_on=win.r_edges.isChecked(), reverse=key == "safety",
                                   show_scale=False, tris=tris, outline=outline)
    win.legend.set_data(title, edges, colors, lambda v: fmt(float(v)))
    win.legend.set_note(NOTE)
    win.view.show_parts(True, 0.85)                  # the global model around the submodel (orientation)
    win.view.set_mode("results")
    if len(probe_vals):
        jmax, jmin = int(np.argmax(probe_vals)), int(np.argmin(probe_vals))
        if key != "safety":
            win.view.mark(probe_pts[jmax], f"max {fmt(float(probe_vals[jmax]))}", (170, 0, 0))
        if key not in ("mises", "displacement", "cut"):
            win.view.mark(probe_pts[jmin], f"min {fmt(float(probe_vals[jmin]))}",
                          (170, 0, 0) if key == "safety" else (0, 60, 170))
    win._r_view = (0.0, None)                         # dialogs mark points without the global deformation
    win._probe = (NearestPoints(probe_pts), np.full(len(probe_pts), -1), probe_vals, unit, name)
    tree = NearestPoints(X[surf])
    res_sub = sr.as_results()

    def exact(p, _v=vals):
        _, j = tree.query(p)
        q = np.asarray(p, float) - scale * sr.U[surf[j]]          # undeformed position of the clicked point
        from fea_core.results import tet10_shape
        e, xi = res_sub.locate(q)
        return float(tet10_shape(xi) @ _v[m.tets[e]])
    win._probe_exact = exact
    win.res_info.setHtml(info_html(win, sr, scale))


def info_html(win, sr, scale: float = 0.0) -> str:
    s = sr.summary
    chk = s.get("cut_check") or {}
    eq = s.get("equilibrium") or {}
    ok = chk.get("passed", False)
    col = "#2e9e4f" if ok else "#d23b3b"
    rows = [f"<b>Submodel: {html.escape(sr.name)}</b> – load case {html.escape(sr.step_name)}",
            f"Region: {html.escape(s.get('region_text', ''))}, parts {html.escape(', '.join(s.get('parts') or []))}",
            f"Mesh: {s.get('nodes', 0):,} nodes, {s.get('elements', 0):,} C3D10 elements, element size {fmt(sr.region.mesh_size, 3)} mm".replace(",", " "),
            f"Max. von Mises: <b>{fmt(s.get('max_mises_MPa'))} MPa</b> at the point {win._pt(s.get('max_mises_at') or (0, 0, 0), 4)} mm",
            f"Global model in the region: max. von Mises {fmt(s.get('global_max_mises_MPa'))} MPa (element size {fmt(s.get('global_mesh_size_mm'), 3)} mm)"
            + (f", submodel / global = {fmt(s.get('ratio_to_global'), 3)}" if s.get("ratio_to_global") else ""),
            f"S1 max. {fmt(s.get('max_s1_MPa'))} MPa, S3 min. {fmt(s.get('min_s3_MPa'))} MPa, max. displacement {fmt(s.get('max_displacement_mm'))} mm",
            f"<span style='color:{col}'>Cut check: mean difference {fmt(100 * chk.get('mean_rel_diff', 0), 3)} % "
            f"(limit {fmt(100 * chk.get('tol_mean', 0.1), 3)} %), max. {fmt(100 * chk.get('max_rel_diff', 0), 3)} % "
            f"(limit {fmt(100 * chk.get('tol_max', 0.25), 3)} %) – {'OK' if ok else 'EXCEEDED'}</span>",
            f"Equilibrium check: imbalance {fmt(eq.get('imbalance', 0.0), 2)}"]
    mp = s.get("mapped") or []
    if mp:
        rows.append("Transferred into the region: " + html.escape(", ".join(
            f"{m.get('selection') or m.get('kind')} ({m.get('kind')})" for m in mp)))
    if scale:
        rows.append(f"Deformation scale: {fmt(scale, 3)}×")
    if sr.warnings:
        rows.append("<span style='color:#b36b00'>! " + "<br>! ".join(html.escape(w) for w in sr.warnings) + "</span>")
    rows.append("<i>The submodel is valid only if the cut check passes: the cut must lie away from the stress concentration.</i>")
    return "<br>".join(rows)


# ===================================================================== dialog

class SubmodelDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Submodel")
        self.setModal(False)
        self.resize(760, 760)
        self.thread = None
        lay = QVBoxLayout(self)
        intro = QLabel("Finer local model of a region of the solved model: the parts are cut by a box or a sphere and meshed finer, the global displacement is prescribed on the cut faces; supports and face loads, gravity and rotation inside the region are transferred.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        f = QFormLayout()
        self.step = QComboBox()
        help_tip(self.step, "Load case of the global result driving the submodel")
        f.addRow("Load case", self.step)
        self.shape = QComboBox()
        self.shape.addItem("Box (centre + size)", "box")
        self.shape.addItem("Sphere (centre + radius)", "sphere")
        help_tip(self.shape, "Shape of the submodel region (axes of the box = global axes)")
        self.shape.currentIndexChanged.connect(self._shape_changed)
        f.addRow("Region", self.shape)
        h = QHBoxLayout()
        self.c = [dspin(0.0, -1e7, 1e7, 3) for _ in range(3)]
        for s, ax in zip(self.c, "xyz"):
            help_tip(s, f"{ax} coordinate of the region centre [mm] (global)")
            s.valueChanged.connect(lambda _: self.draw_region())
            h.addWidget(s)
        f.addRow("Centre [mm]", h)
        h = QHBoxLayout()
        b = QPushButton("Click in 3D")
        b.setCheckable(True)
        help_tip(b, "Centre = point clicked on the model in the results (undeformed position)")
        b.clicked.connect(self.start_pick)
        self.btn_pick = b
        h.addWidget(b)
        b = QPushButton("Selected hotspot")
        help_tip(b, "Centre = location selected in the Highest stresses table (else the 1st row)")
        b.clicked.connect(self.from_hotspot_row)
        h.addWidget(b)
        b = QPushButton("Max. stress")
        help_tip(b, "Centre = highest von Mises outside singularities of the selected load case")
        b.clicked.connect(self.from_max)
        h.addWidget(b)
        h.addStretch(1)
        f.addRow("", h)
        self.box_w = QWidget()
        hb = QHBoxLayout(self.box_w)
        hb.setContentsMargins(0, 0, 0, 0)
        self.sz = [dspin(20.0, 0.01, 1e6, 3, 1.0, " mm") for _ in range(3)]
        for s, ax in zip(self.sz, "xyz"):
            help_tip(s, f"Box size in the global {ax} direction [mm]")
            s.valueChanged.connect(lambda _: self.draw_region())
            hb.addWidget(s)
        f.addRow("Box size", self.box_w)
        self.radius = dspin(10.0, 0.01, 1e6, 3, 1.0, " mm")
        help_tip(self.radius, "Sphere radius [mm]")
        self.radius.valueChanged.connect(lambda _: self.draw_region())
        f.addRow("Radius", self.radius)
        self.h = dspin(1.0, 0.001, 1e5, 3, 0.1, " mm")
        help_tip(self.h, "Element size of the submodel mesh [mm] (default = global size / 4)")
        f.addRow("Element size", self.h)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Submodel 1")
        help_tip(self.name, "Name of the submodel in the result menu (optional)")
        f.addRow("Name", self.name)
        self.form = f
        lay.addLayout(f)
        h = QHBoxLayout()
        self.btn_run = QPushButton("Run submodel")
        theme.set_primary(self.btn_run)
        help_tip(self.btn_run, "Cut, mesh and solve the submodel in CalculiX (in the background)")
        self.btn_run.clicked.connect(self.run)
        h.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        help_tip(self.btn_cancel, "Cancel the submodel analysis")
        self.btn_cancel.clicked.connect(self.cancel)
        h.addWidget(self.btn_cancel)
        self.show3d = QCheckBox("region in 3D")
        self.show3d.setChecked(True)
        help_tip(self.show3d, "Blue: outline of the submodel region in the 3D view")
        self.show3d.toggled.connect(lambda _: self.draw_region())
        h.addWidget(self.show3d)
        h.addStretch(1)
        lay.addLayout(h)
        self.progress = TaskProgress()
        lay.addWidget(self.progress)
        self.out = QLabel("")
        self.out.setWordWrap(True)
        self.out.setTextFormat(Qt.RichText)
        lay.addWidget(self.out)
        self.tab = table(["Name", "Load case", "Region", "Max. σ_vM [MPa]", "Global [MPa]", "Cut check [%]"], 140,
                         placeholder="No submodel computed yet")
        help_tip(self.tab, "Computed submodels; double-click = show the result in 3D")
        self.tab.cellDoubleClicked.connect(lambda r, _c: self.show_row(r))
        lay.addWidget(self.tab, 1)
        h = QHBoxLayout()
        b = QPushButton("Show")
        help_tip(b, "Show the selected submodel result in 3D (result menu item Submodel: …)")
        b.clicked.connect(lambda: self.show_row(self.tab.currentRow()))
        h.addWidget(b)
        b = QPushButton("Delete")
        help_tip(b, "Delete the selected submodel from the results")
        b.clicked.connect(self.delete_row)
        h.addWidget(b)
        h.addStretch(1)
        b = QPushButton("Close")
        help_tip(b, "Close the dialog (a running analysis is cancelled)")
        b.clicked.connect(self.close)
        h.addWidget(b)
        lay.addLayout(h)
        self.refresh()
        self.defaults()
        self._shape_changed()

    # ------------------------------------------------------------------ form
    def refresh(self):
        res = self.win.results
        cur = self.step.currentData()
        self.step.blockSignals(True)
        self.step.clear()
        self.step.setProperty("_i18n_done", True)              # load case names = study data
        for i, s in enumerate(res.steps if res is not None else []):
            self.step.addItem(s.name, i)
        c = self.win.current_result()
        want = cur if cur is not None else (c[1] if c and c[0] == "step" else 0)
        k = self.step.findData(want)
        self.step.setCurrentIndex(max(k, 0))
        self.step.blockSignals(False)
        self.fill_table()

    def defaults(self):
        st = self.win.study
        hg = float(st.mesh.size) if st is not None else 5.0
        for s in self.sz:
            s.setValue(4.0 * hg)
        self.radius.setValue(2.0 * hg)
        self.h.setValue(hg / 4.0)

    def _shape_changed(self):
        box = self.shape.currentData() == "box"
        self.form.setRowVisible(self.box_w, box)
        self.form.setRowVisible(self.radius, not box)
        self.draw_region()

    def region(self):
        from fea_core.submodel_run import SubmodelRegion
        return SubmodelRegion(center=[s.value() for s in self.c], shape=self.shape.currentData(),
                              size=[s.value() for s in self.sz], radius=self.radius.value(), mesh_size=self.h.value(),
                              step=int(self.step.currentData() or 0), name=self.name.text().strip())

    def set_center(self, p):
        for s, v in zip(self.c, p):
            s.blockSignals(True)
            s.setValue(float(v) + 0.0)
            s.blockSignals(False)
        self.draw_region()

    def start_pick(self, on=True):
        if not on:
            if getattr(self.win, "_pick_hook", None) == self.on_pick:
                self.win._pick_hook = None
            return
        if self.win.STAGES[self.win.stage] != "Results":
            self.win.set_stage(self.win.STAGE["Results"])
        self.win._pick_hook = self.on_pick
        self.win.statusBar().showMessage("Click the model in 3D – centre of the submodel region", 8000)

    def on_pick(self, p):
        self.btn_pick.setChecked(False)
        self.win._pick_hook = None
        self.set_center(p)

    def from_hotspot_row(self):
        rows = getattr(self.win, "_hs_rows", None) or []
        if not rows:
            QMessageBox.information(self, self.windowTitle(), "The Highest stresses table is empty – show a stress quantity of a load case in the results.")
            return
        r = self.win.hs_table.currentRow()
        h = rows[r if 0 <= r < len(rows) else 0][0]
        self.set_center(h["xyz_mm"])

    def from_max(self):
        from fea_core.mcp.tools_submodel import center_from_hotspots
        try:
            p = center_from_hotspots(self.win.diag_session(), int(self.step.currentData() or 0))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, self.windowTitle(), str(e))
            return
        self.set_center(p)

    # ------------------------------------------------------------------ calculation
    def run(self):
        from fea_core.submodel_run import SubmodelError
        from .workers import TaskThread
        win = self.win
        if self.thread is not None or win.results is None:
            return
        region = self.region()
        try:
            region.check()
        except SubmodelError as e:
            self.out.setText(f"<span style='color:#d23b3b'>{html.escape(str(e))}</span>")
            return
        if not region.name:
            region.name = f"Submodel {len(win.results.submodels) + 1}"
        st, res, step = win.study, win.results, win.step_path
        from fea_core.geometry import legacy_step_names
        legacy = legacy_step_names(st)
        wd = win._workdir / f"submodel_{int(time.time() * 1000)}"

        def job(t):
            from fea_core.geometry import read_step
            from fea_core.submodel_run import run_submodel
            # parts are reloaded in the thread – Boolean operations over shapes shown in 3D would collide with the view
            parts = read_step(step, legacy_names=legacy) if step is not None else []
            return run_submodel(st, parts, res, region, wd, on_log=t.log.emit, cancel=t.cancel)
        t = TaskThread(job, self)
        t.log.connect(self.on_log)
        t.done.connect(self.on_done)
        t.failed.connect(self.on_failed)
        t.finished.connect(self.on_finished)
        self.thread = t
        self._res = res
        win.register_aux(t)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.out.setText("")
        self.progress.start("Submodel – cutting the parts", 0.1)
        t.start()

    def on_log(self, text):
        self.win.log_msg(text)
        for mark, v, end, phase in (("Submodel: meshing", 0.1, 0.45, "Submodel – meshing"),
                                    ("Submodel mesh:", 0.45, 0.5, "Submodel – boundary conditions"),
                                    ("Submodel: CalculiX", 0.5, 0.95, "Submodel – CalculiX")):
            if text.startswith(mark):
                self.progress.report(v, end, phase)

    def cancel(self):
        if self.thread is not None:
            self.thread.cancel.set()
            self.win.log_msg("Cancelling the submodel…")

    def on_failed(self, text):
        cancelled = self.thread is not None and self.thread.cancel.is_set()
        self.progress.finish("cancelled" if cancelled else "error", "Cancelled" if cancelled else "Submodel failed")
        self.out.setText(f"<span style='color:#d23b3b'>{html.escape(text)}</span>")
        self.win.log_msg(f"SUBMODEL ERROR: {text}")
        if not cancelled:
            self.win.messages.add("error", "solver", f"Submodel failed: {text}", "Results")

    def on_finished(self):
        self.win.unregister_aux(self.thread)
        self.thread = None
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def on_done(self, sr):
        win = self.win
        if win.results is not self._res:                  # the results were discarded during the run
            self.progress.finish("cancelled", "Results changed – submodel discarded")
            return
        win.results.submodels.append(sr)
        self.progress.finish("done", "Submodel finished")
        chk = sr.check
        txt = [f"<b>{html.escape(sr.name)}</b>: max. von Mises {fmt(sr.summary.get('max_mises_MPa'))} MPa "
               f"(global model {fmt(sr.summary.get('global_max_mises_MPa'))} MPa)"]
        if not chk.get("passed", False):
            txt.append(f"<span style='color:#d23b3b'>! Cut check exceeded: mean difference {fmt(100 * chk.get('mean_rel_diff', 0), 3)} %, max. {fmt(100 * chk.get('max_rel_diff', 0), 3)} % – enlarge the region.</span>")
            win.messages.add("warning", "solver", f"Submodel {sr.name}: cut check exceeded – enlarge the region", "Results")
        for w in sr.warnings:
            if not w.startswith("Cut check"):
                txt.append(f"<span style='color:#b36b00'>! {html.escape(w)}</span>")
        self.out.setText("<br>".join(txt))
        win.dirty = True                                   # results change without discarding them (saved with the project)
        win.update_title()
        win.refresh_all()
        self.fill_table()
        self.show_row(len(win.results.submodels) - 1)

    # ------------------------------------------------------------------ list
    def fill_table(self):
        subs = getattr(self.win.results, "submodels", None) or []
        t = self.tab
        t.setRowCount(len(subs))
        for r, sr in enumerate(subs):
            s = sr.summary
            chk = sr.check
            vals = [sr.name, sr.step_name, s.get("region_text", ""), fmt(s.get("max_mises_MPa")),
                    fmt(s.get("global_max_mises_MPa")), fmt(100 * chk.get("mean_rel_diff", 0), 3)]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if c >= 3:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if c == 5 and not chk.get("passed", False):
                    it.setForeground(QColor("#d23b3b"))
                t.setItem(r, c, it)

    def show_row(self, r):
        subs = getattr(self.win.results, "submodels", None) or []
        if not 0 <= r < len(subs):
            return
        self.win.select_result(f"sub:{r}")

    def delete_row(self):
        win = self.win
        subs = getattr(win.results, "submodels", None) or []
        r = self.tab.currentRow()
        if not 0 <= r < len(subs):
            return
        subs.pop(r)
        win.dirty = True
        win.update_title()
        win.refresh_all()
        self.fill_table()
        win.show_result()

    # ------------------------------------------------------------------ region in 3D
    def draw_region(self):
        win = self.win
        v = win.view
        for o in v._marks.pop("submodel_region", []):
            v.ctx.Remove(o, False)
            if o in v._result_objs:
                v._result_objs.remove(o)
        if not getattr(self, "show3d", None) or not self.show3d.isChecked() or win.results is None \
                or win.STAGES[win.stage] != "Results":
            v._update()
            return
        c = np.array([s.value() for s in self.c])
        segs = []
        if self.shape.currentData() == "box":
            hs = 0.5 * np.array([s.value() for s in self.sz])
            P = np.array([c + hs * np.array([i, j, k]) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
            for a, b in ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)):
                segs.append((P[a], P[b]))
        else:
            r = self.radius.value()
            t = np.linspace(0, 2 * np.pi, 49)
            for e1, e2 in ((0, 1), (1, 2), (0, 2)):
                pts = np.zeros((len(t), 3))
                pts[:, e1], pts[:, e2] = r * np.cos(t), r * np.sin(t)
                pts += c
                segs += [(pts[i], pts[i + 1]) for i in range(len(t) - 1)]
        n0 = len(v._result_objs)
        v.result_lines(segs, color=C_BOX, width=2.5, pickable=False)
        v._marks["submodel_region"] = v._result_objs[n0:]
        v._update()

    def closeEvent(self, e):
        if self.thread is not None:
            self.cancel()
            self.thread.wait(10000)
        if getattr(self.win, "_pick_hook", None) == self.on_pick:
            self.win._pick_hook = None
        self.show3d.setChecked(False)
        super().closeEvent(e)
