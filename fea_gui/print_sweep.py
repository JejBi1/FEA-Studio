"""Sweep of the print settings and advice by the place of failure (F5, step 2) in the 3D printing step.

SweepBox: grid perimeters × infill × pattern × build direction (fea_core.printing.sweep) – time estimate, calculation in a thread
on one mesh, table and mass × safety chart with the Pareto boundary and recommendation, taking a variant into the study
or into the variants. The definition is in the study (PrintSettings.sweep) – a change does not discard the study results.
AdviceBox: advice by the place of failure from the results (fea_core.printing.advice.advise, INDICATIVE) – try as a
variant, sweep the infill; ranking of the 6 build directions by calculation (rank_build_directions) with taking over the best direction.
"""
from __future__ import annotations

import html
import math
import re
from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPalette
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QMessageBox, QProgressBar, QPushButton, QSpinBox, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from fea_core.printing.infill import PATTERNS, pattern_label
from fea_core.printing.model import AXIS_DIRS, PrintVariant, SweepSpec
from fea_core.printing.variants import BASE_NAME

from .print_variants import (BAD_BG, OK_BG, PARETO_BG, PREV_NAME, REC_BG, WARN_BG, _Runner, adopt_new_mesh,
                             adopt_results, adopt_variant, status_color, status_brush, study_key)
from .widgets import (LazyPlot, check_item, checked, fit_rows, fmt_dash as fmt, help_tip, set_checks,
                      set_placeholder, table)
from .workers import TaskThread

DIR_TEXT = {"+X": "+X", "-X": "−X", "+Y": "+Y", "-Y": "−Y", "+Z": "+Z", "-Z": "−Z"}
KIND_BG = {"ok": OK_BG, "save": PARETO_BG, "fail": BAD_BG, "thin": BAD_BG}
KIND_BG_OTHER = WARN_BG
SUG_LABEL = {"walls": "perimeters", "infill": "infill", "pattern": "pattern", "build_dir": "build",
             "top_layers": "top layers", "bottom_layers": "bottom layers", "solid_thin": "solid thin regions"}
NAME_ROLE = Qt.UserRole
SORT_ROLE = Qt.UserRole + 1


# Section = the shared collapsible section from widgets (the name stays in the module because of imports)


class NumItem(QTableWidgetItem):
    """Cell sorted by number (SORT_ROLE), not by text with a decimal comma."""

    def __init__(self, text: str, value=None):
        super().__init__(text)
        v = value if value is not None and not (isinstance(value, float) and math.isnan(value)) else -math.inf
        self.setData(SORT_ROLE, float(v))

    def __lt__(self, other):
        a, b = self.data(SORT_ROLE), other.data(SORT_ROLE)
        if a is None or b is None:
            return super().__lt__(other)
        return a < b


def parse_list(text: str, integer: bool = False, what: str = "value") -> list[float]:
    """"2, 3, 4" / "10 20 40" / "7,5; 15" (with a semicolon the comma is decimal) → numbers. ValueError with a Czech message."""
    t = text.strip()
    if not t:
        return []
    if ";" in t:
        items = [x.strip().replace(",", ".") for x in t.split(";")]
    else:
        items = re.split(r"[,\s]+", t)
    out = []
    for x in items:
        if not x:
            continue
        try:
            v = float(x)
        except ValueError:
            raise ValueError(f"{what}: '{x}' is not a number (separate with a comma or space, decimal comma only with a semicolon: 7,5; 15)") from None
        if integer and v != int(v):
            raise ValueError(f"{what}: '{x}' is not an integer")
        out.append(int(v) if integer else v)
    return out


def join_list(vals, integer: bool = False) -> str:
    if integer or all(float(v) == int(v) for v in vals):
        return ", ".join(str(int(round(v))) for v in vals)
    return "; ".join(fmt(float(v), 6) for v in vals)


def unique_name(names: set[str], base: str) -> str:
    if base not in names:
        return base
    k = 2
    while f"{base} ({k})" in names:
        k += 1
    return f"{base} ({k})"


def measured_single(*runs) -> float | None:
    """Measured time of one variant calculation [s] from the last valid calculations of variants / sweep."""
    ts = [o.time_s for r in runs if r is not None for o in r.outcomes if o.ok and o.time_s > 0]
    return float(sum(ts) / len(ts)) if ts else None


# ====================================================================== graf
class SweepPlot(LazyPlot):
    """Chart mass × min. safety of the sweep variants: Pareto boundary, required safety, recommended
    and the default variant. Click on a point = selection of a table row (pointClicked).
    """
    pointClicked = Signal(str)
    FIGSIZE = (4, 2.8)
    MIN_HEIGHT = 260

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pts = []
        self._ax = None

    def _ensure(self):
        if self.canvas is None:
            super()._ensure()
            self.canvas.mpl_connect("button_press_event", self._on_click)

    def draw(self, rows: list[dict], k_req: float, title: str = ""):
        ax = self._ax = self.axes()
        pts = [r for r in rows if r["ok"] and r["mass_g"] is not None and r["min_safety"] is not None]
        self._pts = [(r["name"], r["mass_g"], r["min_safety"]) for r in pts]
        good = [r for r in pts if r["min_safety"] >= k_req]
        bad = [r for r in pts if r["min_safety"] < k_req]
        if good:
            ax.scatter([r["mass_g"] for r in good], [r["min_safety"] for r in good], s=22, color="#1f77b4",
                       zorder=3, label=f"k ≥ {fmt(k_req, 3)}")
        if bad:
            ax.scatter([r["mass_g"] for r in bad], [r["min_safety"] for r in bad], s=22, color="#c62828",
                       zorder=3, label=f"k < {fmt(k_req, 3)}")
        front = sorted((r for r in pts if r["pareto"]), key=lambda r: r["mass_g"])
        if front:
            ax.plot([r["mass_g"] for r in front], [r["min_safety"] for r in front], "-", color="#ef6c00", lw=1.4,
                    zorder=2, label="Pareto front")
            ax.scatter([r["mass_g"] for r in front], [r["min_safety"] for r in front], s=40, facecolor="none",
                       edgecolor="#ef6c00", lw=1.2, zorder=4)
        for r in pts:
            if r["base"]:
                ax.scatter([r["mass_g"]], [r["min_safety"]], s=60, marker="s", facecolor="none", edgecolor="black",
                           lw=1.2, zorder=5, label="study settings")
            if r["recommended"]:
                ax.scatter([r["mass_g"]], [r["min_safety"]], s=110, marker="*", color="#2e7d32", edgecolor="black",
                           lw=0.6, zorder=6, label="recommended")
                ax.annotate(r["name"], (r["mass_g"], r["min_safety"]), fontsize=7, xytext=(5, 4),
                            textcoords="offset points")
        ax.axhline(k_req, color="#555", lw=0.9, ls="--", label=f"required safety factor {fmt(k_req, 3)}")
        ax.set_xlabel("mass of printed parts [g]")
        ax.set_ylabel("min. print safety factor [–]")
        if title:
            ax.set_title(title, fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="best", framealpha=0.85)
        self.canvas.draw_idle()

    def _on_click(self, ev):
        if ev.inaxes is not self._ax or not self._pts or ev.x is None:
            return
        import numpy as np
        xy = self._ax.transData.transform(np.array([[m, k] for _, m, k in self._pts]))
        d = np.hypot(xy[:, 0] - ev.x, xy[:, 1] - ev.y)
        i = int(np.argmin(d))
        if d[i] <= 12:
            self.pointClicked.emit(self._pts[i][0])


# ====================================================================== sweep
class SweepBox(QWidget, _Runner):
    """Content of the "Parameter sweep" section. panel = PrintPanel."""

    TITLE = "Print sweep"
    COLS = ["Variant", "Perimeters", "Infill", "Pattern", "Z direction", "Min. safety factor", "Mass [g]",
            "Max. displacement [mm]", "Governing", "Pareto"]

    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.win = panel.win
        self.run_result = None          # VariantsRun s .sweep
        self._run_key = None
        self._mesh_key = None
        self._thread = None
        self._filling = False
        self._live = []                 # finished variants during the calculation (running table)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 0, 0, 0)
        note = QLabel("Grid of print settings (perimeters × infill × pattern × build direction) computed on a <b>single mesh</b>; empty parameter = value from the study. The lightest variant with a safety factor ≥ the required one is recommended (Variants above). Variant results are not kept – adopt the selected one (recalculation).")
        note.setWordWrap(True)
        lay.addWidget(note)
        f = QFormLayout()
        self.parts = QListWidget()
        set_placeholder(self.parts, "No printed part yet")
        fit_rows(self.parts, 2, 4)
        help_tip(self.parts, "Parts whose settings the sweep changes (none = all)",
                 "Parts whose settings the sweep changes (all the same). Nothing checked = all printed parts.")
        self.parts.itemChanged.connect(self.store)
        f.addRow("Parts", self.parts)
        self.walls = QLineEdit()
        self.walls.setPlaceholderText("e.g. 2, 3, 4  (empty = from study)")
        self.walls.setToolTip("Perimeter counts separated by a comma or space")
        self.walls.editingFinished.connect(self.store)
        f.addRow("Perimeters", self.walls)
        self.infill = QLineEdit()
        self.infill.setPlaceholderText("e.g. 10, 15, 25  (empty = from study)")
        help_tip(self.infill, "Infill densities in %, separated by comma or space",
                 "Infill densities in % separated by commas or spaces; use semicolons if you type a decimal comma (7,5; 15; 30)")
        self.infill.editingFinished.connect(self.store)
        f.addRow("Infill [%]", self.infill)
        self.patterns = QListWidget()
        self.patterns.setMaximumHeight(96)
        self.patterns.setToolTip("Checked infill patterns are swept; none = pattern from the study")
        for k in PATTERNS:
            check_item(self.patterns, pattern_label(k), False, k)
        self.patterns.itemChanged.connect(self.store)
        f.addRow("Infill patterns", self.patterns)
        hd = QHBoxLayout()
        self.dirs = {}
        for k in AXIS_DIRS:
            c = QCheckBox(DIR_TEXT[k])
            c.setToolTip(f"Build direction (printer Z) {DIR_TEXT[k]} in model coordinates")
            c.toggled.connect(self.store)
            self.dirs[k] = c
            hd.addWidget(c)
        hd.addStretch(1)
        f.addRow("Build direction", hd)
        lbl = QLabel("<i>nothing checked = current build direction of the parts</i>")
        f.addRow("", lbl)
        self.max_n = QSpinBox()
        from fea_core.printing.sweep import HARD_LIMIT
        self.max_n.setRange(1, HARD_LIMIT)
        self.max_n.setValue(60)
        self.max_n.setToolTip(f"Variant count limit (protection against a long computation), at most {HARD_LIMIT}")
        self.max_n.editingFinished.connect(self.store)
        f.addRow("Max. number of variants", self.max_n)
        lay.addLayout(f)
        self.count = QLabel()
        self.count.setWordWrap(True)
        lay.addWidget(self.count)
        h = QHBoxLayout()
        self.btn_est = QPushButton("Estimate")
        help_tip(self.btn_est, "Number of variants and estimated computation time",
                 "Number of variants and estimated run time (from the previous variants run time, otherwise from the mesh DOF count)")
        self.btn_est.clicked.connect(self.do_estimate)
        h.addWidget(self.btn_est)
        self._add_run_buttons(h, "Compute sweep", self.compute,
                               "Study settings + all grid variants on a single mesh (no re-meshing)")
        self.btn_run.setWhatsThis(
            "<b>Parameter sweep</b><p>Computes all combinations of the given perimeters, infills, patterns and build directions (an empty parameter = the value from the study) for the checked parts, all on a single mesh. The results are shown in a table and a mass × safety factor plot with the Pareto front; the lightest variant with a safety factor ≥ required is recommended (field in the Variants section).</p><p>Use <i>Estimate</i> before running – the number of analyses grows multiplicatively. You can adopt the selected variant into the study or add it to the variants (recomputed with results).</p>")
        h.addStretch(1)
        lay.addLayout(h)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.hide()
        lay.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.table = table(self.COLS, placeholder="Sweep results appear after the computation", rows=(3, 10))
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)   # grid order, sorting by click
        help_tip(self.table, "Green = recommended, * = Pareto front; click to sort",
                 "Green = recommended, * = on the Pareto front (nothing lighter is safer); click a header to sort")
        lay.addWidget(self.table)
        self.reason = QLabel()
        self.reason.setWordWrap(True)
        lay.addWidget(self.reason)
        self.plot = SweepPlot()
        self.plot.hide()
        self.plot.pointClicked.connect(self.select_name)
        lay.addWidget(self.plot)
        h = QHBoxLayout()
        b = QPushButton("Adopt selected variant")
        help_tip(b, "Settings of the selected variant → study (results are discarded)",
                 "Settings of the selected variant → study (study results are discarded – recompute)")
        b.clicked.connect(self.adopt)
        h.addWidget(b)
        b = QPushButton("Add to variants…")
        b.setToolTip("Selected sweep variant → variants table above (full analysis with results)")
        b.clicked.connect(self.add_variant)
        h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)

    # ------------------------------------------------------------------ stav
    def printed(self) -> list[str]:
        from fea_core.printing.region import print_parts
        return print_parts(self.study, self.panel.part_names()) if self.study is not None else []

    def _sync_buttons(self):
        _Runner._sync_buttons(self)
        self.btn_est.setEnabled(self._thread is None)

    def refresh(self):
        st = self.study
        if st is None:
            return
        sw = st.printing.sweep
        self._filling = True
        try:
            self.parts.blockSignals(True)
            self.parts.clear()
            for p in self.printed():
                check_item(self.parts, p, p in sw.parts)
            self.parts.blockSignals(False)
            if not self.walls.hasFocus():
                self.walls.setText(join_list(sw.walls, integer=True))
            if not self.infill.hasFocus():
                self.infill.setText(join_list([100.0 * x for x in sw.infill]))
            self.patterns.blockSignals(True)
            set_checks(self.patterns, sw.patterns, Qt.UserRole)
            self.patterns.blockSignals(False)
            for k, c in self.dirs.items():
                c.blockSignals(True)
                c.setChecked(k in sw.build_dirs)
                c.blockSignals(False)
            self.max_n.blockSignals(True)
            self.max_n.setValue(int(sw.max_variants))
            self.max_n.blockSignals(False)
        finally:
            self._filling = False
        self.update_count()
        if self._thread is None:
            self.show_results()
        self._sync_buttons()

    # ------------------------------------------------------------------ definice
    def spec_from_form(self) -> SweepSpec:
        """SweepSpec from the form (ValueError with a Czech message for a wrong number)."""
        walls = parse_list(self.walls.text(), integer=True, what="Perimeters")
        infill = parse_list(self.infill.text(), what="Infill")
        if any(w < 0 for w in walls):
            raise ValueError("Perimeters must be ≥ 0")
        if any(not 0.0 <= x <= 100.0 for x in infill):
            raise ValueError("Infill must be 0–100 %")
        dirs = [k for k, c in self.dirs.items() if c.isChecked()]
        return SweepSpec(checked(self.parts), [int(w) for w in walls], [x / 100.0 for x in infill],
                         checked(self.patterns, Qt.UserRole), dirs, int(self.max_n.value()))

    def store(self, *_):
        """Form → Study.printing.sweep (a study change with a snapshot for undo/redo, results stay)."""
        if self._filling or self.study is None:
            return
        try:
            sw = self.spec_from_form()
        except ValueError as e:
            self.count.setText(f"<span style='color:#c62828'>{html.escape(str(e))}</span>")
            return
        st = self.study
        if sw.to_dict() != st.printing.sweep.to_dict():
            st.printing.sweep = sw
            self.win.changed(invalidate_results=False)
        self.update_count()

    def prefill(self, parts=None, walls=None, infill=None, patterns=None, build_dirs=None, open_section=True):
        """Pre-fills the form (advice "Sweep infill") and saves to the study; infill 0–1."""
        st = self.study
        if st is None:
            return
        old = st.printing.sweep
        sw = SweepSpec(list(parts if parts is not None else old.parts),
                       [int(w) for w in (walls if walls is not None else old.walls)],
                       [float(x) for x in (infill if infill is not None else old.infill)],
                       list(patterns if patterns is not None else []),
                       list(build_dirs if build_dirs is not None else []), int(old.max_variants))
        if sw.to_dict() != old.to_dict():
            st.printing.sweep = sw
            self.win.changed(invalidate_results=False)
        else:
            self.refresh()
        if open_section:
            sec = getattr(self.panel, "sweep_section", None)
            if sec is not None:
                sec.set_open(True)
                self.panel.scroll_to(sec)

    def update_count(self):
        st = self.study
        if st is None:
            return
        from fea_core.printing.sweep import count
        sw = st.printing.sweep
        n = count(sw)
        if n == 0:
            self.count.setText("<i>Enter at least one parameter (perimeters, infill, pattern or build direction).</i>")
        elif n > sw.max_variants:
            self.count.setText(f"<span style='color:#c62828'>{n} variants – more than the allowed {sw.max_variants}; narrow the range or increase the max. number.</span>")
        else:
            self.count.setText(f"{n} variants + study settings = {n + 1} computations (time estimate: Estimate button).")

    def do_estimate(self) -> dict | None:
        st = self.study
        if st is None:
            return None
        self.store()
        from fea_core.printing.sweep import check, estimate
        errs = check(st, st.printing.sweep, self.panel.part_names())
        mesh = self.win.valid_mesh()
        single = measured_single(self.valid_run(), self.panel.variants_box.valid_run())
        try:
            est = estimate(st, st.printing.sweep, mesh=mesh, single_s=single)
        except Exception as e:  # noqa: BLE001
            self.status.setText(f"<span style='color:#c62828'>{html.escape(str(e))}</span>")
            return None
        rows = [f"<b>{est['count']} variants</b> + the study settings = {est['runs']} calculations"
                + (f" ({fmt(est['dof'] / 1000, 4)} k DOF)" if est.get("dof") else "")]
        if est.get("total_s") is not None:
            rows.append(f"estimate {fmt(est['total_s'] / 60, 3)} min (one computation ~{fmt(est['single_s'], 3)} s, {est['source']})")
        else:
            rows.append("time unknown – no mesh (create a mesh in the Mesh step or solve the study)")
        if est["over_limit"]:
            rows.append(f"<span style='color:#c62828'>! exceeds the max. number of variants {est['max_variants']}</span>")
        for e in errs:
            if "at most" not in e:
                rows.append(f"<span style='color:#c62828'>! {html.escape(e)}</span>")
        self.status.setText("<br>".join(rows))
        return est

    # ------------------------------------------------------------------ calculation
    def compute(self):
        win = self.win
        st = self.study
        if st is None or self._thread is not None or win._thread is not None:
            return
        self.store()
        from fea_core.printing.sweep import check, estimate
        names = self.panel.part_names()
        errs = check(st, st.printing.sweep, names)
        if errs:
            QMessageBox.warning(self, self.TITLE, "\n".join(errs))
            return
        if not self._ready():
            return
        mesh = win.valid_mesh()
        single = measured_single(self.valid_run(), self.panel.variants_box.valid_run())
        est = estimate(st, st.printing.sweep, mesh=mesh, single_s=single)
        if est.get("total_s") and est["total_s"] > 600 and not win.ask(
                f"The sweep ({est['runs']} computations) will take an estimated {fmt(est['total_s'] / 60, 3)} min. Run it?"):
            return
        self._run_key, wd = self._begin("sweep")
        sw = SweepSpec.from_dict(st.printing.sweep.to_dict())
        study, parts = win.study_for_solve(), win.parts
        win.log_msg(f"Print sweep: {est['runs']} calculations in the folder {wd}"
                    + (" (mesh reused)" if mesh is not None else " (mesh first)"))

        def job(t):
            from fea_core.printing.sweep import run_sweep
            return run_sweep(study, wd, sw=sw, mesh=mesh, parts=parts, on_log=t.log.emit,
                             on_solver_line=t.solver_line.emit, cancel=t.cancel, on_progress=t.progress.emit,
                             on_outcome=t.outcome.emit, single_s=single)
        self._live = []
        self.run_result = None
        t = TaskThread(job, self)
        t.outcome.connect(self.on_outcome)
        self._start(t, "Computing sweep…")

    def on_outcome(self, o):
        self._live.append(o)
        rows = [{"name": x.name, "ok": x.ok, "error": x.error, "mass_g": x.mass_g, "min_safety": x.min_safety,
                 "max_disp": x.max_disp, "part": (x.governing or {}).get("part"),
                 "region": (x.governing or {}).get("region"), "step": (x.governing or {}).get("step"),
                 "pareto": False, "recommended": False, "base": x.name == BASE_NAME, "settings": x.overrides}
                for x in self._live]
        self.fill_table(rows, {x.name: x for x in self._live}, None, self.study.printing.required_safety)

    def on_done(self, run):
        self._store_run(run, "sweep results", "computations")

    # ------------------------------------------------------------------ results
    def fill_table(self, rows, outcomes: dict, run, k_req: float):
        target = (run.sweep or {}).get("parts") if run is not None and run.sweep else None
        bold = QFont()
        bold.setBold(True)
        cur = self.selected_name()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for r in rows:
            o = outcomes.get(r["name"])
            eff = (o.settings if o is not None else {}) or {}
            sp = eff.get(target[0]) if target else next(iter(eff.values()), None)
            sp = sp or {}
            i = self.table.rowCount()
            self.table.insertRow(i)
            if r["ok"]:
                bg = REC_BG if r["recommended"] else (OK_BG if r["min_safety"] >= k_req else BAD_BG)
            else:
                bg = BAD_BG
            name = ("* " if r["pareto"] else "") + r["name"] + ("  (recommended)" if r["recommended"] else "")
            cells = [
                QTableWidgetItem(name),
                NumItem(str(sp.get("walls", "")) if sp.get("enabled", True) else "solid", sp.get("walls")),
                NumItem(f"{fmt(100 * sp['infill'], 3)} %" if "infill" in sp else "", 100 * sp.get("infill", 0)),
                QTableWidgetItem(pattern_label(sp["pattern"]) if "pattern" in sp else ""),
                QTableWidgetItem(self.panel.dir_text(sp["build_dir"]) if "build_dir" in sp else ""),
                NumItem(fmt(r["min_safety"], 3) if r["ok"] else "error", r["min_safety"]),
                NumItem(fmt(r["mass_g"]), r["mass_g"]),
                NumItem(fmt(r["max_disp"]), r["max_disp"]),
                QTableWidgetItem(f"{r['part']} · {r['region']} · {r['step']}" if r["ok"] else r["error"]),
                QTableWidgetItem("*" if r["pareto"] else ""),
            ]
            cells[0].setData(NAME_ROLE, r["name"])
            for c, it in enumerate(cells):
                it.setBackground(status_brush(PARETO_BG if (r["pareto"] and bg == OK_BG) else bg))
                if r["recommended"] or (r["pareto"] and c == 9):
                    it.setFont(bold)
                if not r["ok"] and c == 8:
                    it.setToolTip(r["error"])
                self.table.setItem(i, c, it)
        self.table.setSortingEnabled(True)
        if cur:
            self.select_name(cur)

    def show_results(self):
        run = self.valid_run()
        if run is None:
            self.table.setRowCount(0)
            self.reason.setText("<i>The sweep results do not match the current study – recompute.</i>"
                                if self.run_result is not None else "")
            self.plot.hide()
            return
        from fea_core.printing.sweep import points
        rows = points(run)
        self.fill_table(rows, {o.name: o for o in run.outcomes}, run, run.required_safety)
        front = run.sweep.get("pareto", []) if run.sweep else run.pareto()
        self.reason.setText(html.escape(run.reason) + f"<br><i>Pareto front: {len(front)} variants (*). Mesh {fmt(run.mesh_info.get('nodes'), 6)} nodes, {fmt(run.time_s, 3)} s in total. Print safety = the multiple of the load to failure (indicative infill factors – calibrate with tests).</i>")
        self.plot.show()
        self.plot.draw(rows, run.required_safety)

    def selected_name(self) -> str | None:
        r = self.table.currentRow()
        it = self.table.item(r, 0) if r >= 0 else None
        return it.data(NAME_ROLE) if it is not None else None

    def select_name(self, name: str):
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None and it.data(NAME_ROLE) == name:
                self.table.selectRow(r)
                self.table.scrollToItem(it)
                return

    def _selected_outcome(self):
        run = self.valid_run()
        name = self.selected_name()
        if run is None or name is None:
            QMessageBox.information(self, self.TITLE, "Select a computed variant in the sweep table.")
            return None, None
        return run, run.outcome(name)

    def adopt(self):
        run, o = self._selected_outcome()
        if o is None:
            return
        if o.name in (BASE_NAME, PREV_NAME):
            QMessageBox.information(self, self.TITLE, "This is the study setting (default row) – nothing to adopt.")
            return
        from fea_core.printing.sweep import to_variant
        win, st = self.win, self.study
        adopt_variant(win, run, to_variant(run, o.name))
        adopt_results(win, o)
        win.log_msg(f"Sweep variant '{o.name}' taken over"
                    + ("" if win.results is not None else " – run the study again (Solve step)."))
        self._run_key = study_key(st)                  # the other variants stay valid (same mesh and loads)
        win.refresh_all()

    def add_variant(self):
        run, o = self._selected_outcome()
        if o is None:
            return
        if o.name in (BASE_NAME, PREV_NAME):
            QMessageBox.information(self, self.TITLE, "The study settings are not added to the variants.")
            return
        from fea_core.printing.sweep import to_variant
        vb = self.panel.variants_box
        name = vb._name("Sweep variant", unique_name({v.name for v in vb.variants()}, o.name))
        if name and vb._store(to_variant(run, o.name, name)):
            self.win.log_msg(f"Sweep variant '{o.name}' added to the variants as '{name}'.")


# ====================================================================== rada
class AdviceBox(QWidget, _Runner):
    """Content of the "Advice by the place of failure" section. panel = PrintPanel."""

    TITLE = "Build directions"
    RCOLS = ["Build direction", "Min. safety factor", "Mass [g]", "Max. displacement [mm]", "Governing"]

    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.win = panel.win
        self.items: list[dict] = []
        self._items_results = None       # Results from which the advice is made
        self.rank = None                 # result of rank_build_directions
        self._rank_key = None
        self._mesh_key = None
        self._thread = None
        self.rank_directions = None      # None = all 6 axes (tests may limit them)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 0, 0, 0)
        note = QLabel("From analysis results: which location and stress component governs (layer bond, perimeter along / across the line, solid layers, infill, thin region) and what to try. <b>INDICATIVE</b> – verify the benefit by computing the variant and by testing.")
        note.setWordWrap(True)
        lay.addWidget(note)
        h = QHBoxLayout()
        self.btn_advise = QPushButton("Advise")
        self.btn_advise.setToolTip("Advice per printed part from the latest results (no computation)")
        self.btn_advise.setWhatsThis(
            "<b>Advise</b><p>From the results of the last analysis it finds, for each printed part, the governing location and stress component in the print axes (layer bond, perimeter along / across the line, solid layers, infill, thin region) and suggests what to change: perimeters, infill and pattern, build direction.</p><p>The suggestion can be added as a variant with one button or prefill the sweep. The advice is INDICATIVE – verify the benefit by a variant analysis and by testing.</p>")
        self.btn_advise.clicked.connect(self.advise)
        h.addWidget(self.btn_advise)
        self._add_run_buttons(h, "Rank build directions", self.compute_rank,
                               "Ranks the 6 build directions by print safety\nComputes all 6 build directions (±X, ±Y, ±Z) as variants on one mesh for the parts selected in the list (none selected = all printed) and ranks them by safety.")
        h.addStretch(1)
        lay.addLayout(h)
        self.progress = QProgressBar()
        self.progress.hide()
        lay.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.list_box = QWidget()
        self.list_lay = QVBoxLayout(self.list_box)
        self.list_lay.setContentsMargins(0, 0, 0, 0)
        self.list_lay.setSpacing(4)
        lay.addWidget(self.list_box)
        self.rank_label = QLabel()
        self.rank_label.setWordWrap(True)
        lay.addWidget(self.rank_label)
        self.rtable = table(self.RCOLS, placeholder="The ranking of directions appears after the computation", rows=(3, 8))
        help_tip(self.rtable, "Ranking of build directions by minimum print safety",
                 "Build directions ranked by minimum print safety factor (best in bold, “current” = direction in the study).\nSelect a row and click “Adopt selected build direction”.")
        self.rtable.hide()
        lay.addWidget(self.rtable)
        self.btn_rank_adopt = QPushButton("Adopt selected build direction")
        help_tip(self.btn_rank_adopt, "Build direction of the selected row → parts in the study",
                 "Build direction of the selected row → compared parts in the study (results are discarded – recompute)")
        self.btn_rank_adopt.clicked.connect(self.adopt_rank)
        self.btn_rank_adopt.hide()
        lay.addWidget(self.btn_rank_adopt)

    def refresh(self):
        if self.items and self._items_results is not self.win.results:
            self.items = []
            self._items_results = None
            self._render_items()
            self.status.setText("<i>The results changed – press Advise for new advice.</i>")
        if self._thread is None:
            self.show_rank()
        self._sync_buttons()

    # ------------------------------------------------------------------ rada
    def advise(self):
        st, res = self.study, self.win.results
        if st is None:
            return
        if res is None or not res.info.get("printing"):
            QMessageBox.information(self, "Advice", "First solve the study with printed parts (Solve step).")
            return
        from fea_core.printing.advice import advise
        try:
            self.items = advise(st, res)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Advice", f"Cannot build the advice:\n{e}")
            return
        self._items_results = res
        self._render_items()
        self.status.setText(f"{len(self.items)} recommendations for {len({i['part'] for i in self.items})} printed parts (required safety factor {fmt(st.printing.required_safety, 3)}).")

    def _render_items(self):
        while self.list_lay.count():
            w = self.list_lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for item in self.items:
            self.list_lay.addWidget(self._item_widget(item))
        if self.items:
            n = QLabel(f"<i>{html.escape(self.items[0].get('note', ''))}</i>")
            n.setWordWrap(True)
            n.setProperty("muted", True)
            self.list_lay.addWidget(n)

    def _item_widget(self, item: dict) -> QWidget:
        fr = QFrame()
        fr.setFrameShape(QFrame.StyledPanel)
        bg = KIND_BG.get(item["kind"], KIND_BG_OTHER)
        fr.setObjectName("advice")              # transparent shade by the kind of advice; text and buttons from the theme
        pal = fr.palette()
        pal.setColor(QPalette.Window, status_color(bg))
        fr.setPalette(pal)
        fr.setAutoFillBackground(True)
        v = QVBoxLayout(fr)
        v.setContentsMargins(6, 4, 6, 4)
        k = item.get("min_safety")
        head = (f"<b>{html.escape(item['part'])}: {html.escape(item['title'])}</b>"
                + (f" <small>(k = {fmt(k, 3)}, {html.escape(str(item.get('step') or ''))})</small>"
                   if k is not None else ""))
        lab = QLabel(f"{head}<br>{html.escape(item['text'])}<br><i>Why: {html.escape(item['reason'])}</i>")
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(lab)
        sug = item.get("suggestion") or {}
        if sug or item.get("sweep"):
            h = QHBoxLayout()
            if sug:
                b = QPushButton("Try as variant")
                b.setToolTip("Adds a variant with the recommended settings (" + self.suggestion_text(sug)
                             + ") to the table of variants – then “Compute variants”")
                b.clicked.connect(lambda _=False, it=item: self.try_variant(it))
                h.addWidget(b)
            if item.get("sweep"):
                b = QPushButton("Sweep infill")
                b.setToolTip("Prefills the perimeters × infill sweep for this part (Parameter sweep section)")
                b.clicked.connect(lambda _=False, it=item: self.sweep_item(it))
                h.addWidget(b)
            h.addStretch(1)
            v.addLayout(h)
        return fr

    def suggestion_text(self, sug: dict) -> str:
        out = []
        for k, val in sug.items():
            if k == "infill":
                s = f"{fmt(100 * float(val), 3)} %"
            elif k == "build_dir":
                s = self.panel.dir_text(val)
            elif k == "pattern":
                s = pattern_label(val)
            elif isinstance(val, bool):
                s = "yes" if val else "ne"
            else:
                s = str(val)
            out.append(f"{SUG_LABEL.get(k, k)} {s}")
        return ", ".join(out)

    def try_variant(self, item: dict) -> bool:
        vb = self.panel.variants_box
        sug = dict(item.get("suggestion") or {})
        if not sug or self.study is None:
            return False
        name = unique_name({v.name for v in vb.variants()},
                           f"Advice {item['part']}: {self.suggestion_text(sug)}")
        ok = vb._store(PrintVariant(name, {item["part"]: sug}, "from failure-location advice"))
        if ok:
            self.status.setText(f"Variant '{html.escape(name)}' added – compute it in the Variants section (“Compute variants”).")
            self.win.log_msg(f"Advice → variant '{name}'")
        return ok

    def sweep_item(self, item: dict):
        sw = item.get("sweep") or {}
        self.panel.sweep_box.prefill(parts=[item["part"]], walls=sw.get("walls", []), infill=sw.get("infill", []))

    # ------------------------------------------------------------------ build directions
    def target_parts(self) -> list[str]:
        return self.panel.variants_box._target_parts()

    def compute_rank(self):
        win = self.win
        st = self.study
        if st is None or self._thread is not None or win._thread is not None:
            return
        targets = self.target_parts()
        if not targets:
            QMessageBox.information(self, self.TITLE, "No part has a printed material.")
            return
        if not self._ready():
            return
        self._rank_key, wd = self._begin("build_dirs")
        mesh = win.valid_mesh()
        study, parts = win.study_for_solve(), win.parts
        dirs = list(self.rank_directions) if self.rank_directions else None
        win.log_msg(f"Build directions ({', '.join(targets)}): {len(dirs or AXIS_DIRS)} directions + study settings")

        def job(t):
            from fea_core.printing.advice import rank_build_directions
            return rank_build_directions(study, wd, part_names=targets, directions=dirs, mesh=mesh, parts=parts,
                                         on_log=t.log.emit, cancel=t.cancel, on_progress=t.progress.emit)
        self.rank = None
        self._start(TaskThread(job, self), "Computing build directions…")

    def on_done(self, res):
        if self.study is None or study_key(self.study) != self._rank_key:
            self.status.setText("The study changed during the computation – direction ranking discarded.")
            return
        self.rank = res
        adopt_new_mesh(self.win, res.get("run"), self._mesh_key)
        best = res.get("best")
        self.status.setText("Build directions computed" + (f" – best {DIR_TEXT.get(best, best)}." if best else "."))
        self.show_rank()

    def valid_rank(self):
        if self.rank is None or self.study is None or self._rank_key != study_key(self.study):
            return None
        return self.rank

    def show_rank(self):
        res = self.valid_rank()
        self.rtable.setRowCount(0)
        if res is None:
            self.rtable.hide()
            self.btn_rank_adopt.hide()
            self.rank_label.setText("<i>The ranking of build directions does not match the current study – recompute.</i>"
                                    if self.rank is not None else "")
            return
        bold = QFont()
        bold.setBold(True)
        k_req = self.study.printing.required_safety
        for r in res["ranking"]:
            i = self.rtable.rowCount()
            self.rtable.insertRow(i)
            best = r["direction"] == res.get("best")
            d = DIR_TEXT.get(r["direction"], r["direction"]) + ("  (current)" if r["current"] else "") \
                + ("  ← best" if best else "")
            if r["ok"]:
                row = [d, fmt(r["min_safety"], 3), fmt(r["mass_g"]), fmt(r["max_disp"]),
                       f"{r['part']} · {r['region']} · {r['step']}"]
                bg = REC_BG if best else (OK_BG if r["min_safety"] >= k_req else BAD_BG)
            else:
                row = [d, "error", "", "", r["error"]]
                bg = BAD_BG
            for c, txt in enumerate(row):
                it = QTableWidgetItem(txt)
                it.setBackground(status_brush(bg))
                if best:
                    it.setFont(bold)
                if c == 0:
                    it.setData(NAME_ROLE, r["direction"])
                self.rtable.setItem(i, c, it)
        cur = res["current"].get("min_safety")
        self.rank_label.setText(f"Parts: {html.escape(', '.join(res['parts']))}. Current setting: k = {fmt(cur, 3)}. The mass changes only slightly with direction (different distribution of solid layers and perimeters). Indicative – supports and surface quality are not assessed.")
        self.rtable.show()
        self.btn_rank_adopt.show()

    def adopt_rank(self):
        res = self.valid_rank()
        r = self.rtable.currentRow()
        it = self.rtable.item(r, 0) if r >= 0 else None
        if res is None or it is None:
            QMessageBox.information(self, self.TITLE, "Select a direction in the table.")
            return
        key = it.data(NAME_ROLE)
        d = list(AXIS_DIRS[key])
        st = self.study
        for p in res["parts"]:
            st.printing.parts[p] = replace(st.printing.spec(p), part=p, build_dir=list(d))
        for row in res["ranking"]:
            row["current"] = row["direction"] == key
            if row["current"]:
                res["current"]["min_safety"] = row["min_safety"]
        self.win.changed()
        self._rank_key = study_key(st)                 # order stays valid (only another current direction)
        self.win.log_msg(f"Build direction {DIR_TEXT[key]} adopted for {', '.join(res['parts'])} – recompute the study.")
        self.panel.refresh()
