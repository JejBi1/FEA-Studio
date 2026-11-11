"""Variants of the internal structure of 3D printing (F5) in the 3D printing step: definition of the variants from the form of the print settings,
calculation of all variants on one mesh (fea_core.printing.variants, thread), table of results, recommendation and a
mass × safety chart. Taking over a variant = settings into the study + its results (without recalculation).
"""
from __future__ import annotations

import json
from html import escape
import time

from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton,
                               QTableWidgetItem, QVBoxLayout)

from fea_core.printing import PrintVariant
from fea_core.printing.infill import pattern_label
from fea_core.printing.variants import BASE_NAME, SPEC_KEYS, apply_variant, effective, validate

from .widgets import LazyPlot, Section, dspin, fmt_dash as fmt, help_tip, table
from .workers import TaskThread, kill_mesh_workers

# states of the result rows = transparent shades (the table background shows through, the text stays in the theme colour → readable
# in both the light and the dark theme)
OK_BG, BAD_BG, REC_BG, PARETO_BG, WARN_BG = range(5)
_TINT = {OK_BG: (46, 160, 67, 45), REC_BG: (46, 160, 67, 110), BAD_BG: (210, 59, 59, 55),
         PARETO_BG: (37, 99, 235, 55), WARN_BG: (230, 150, 0, 60)}


def status_color(kind: int) -> QColor:
    return QColor(*_TINT[kind])


def status_brush(kind: int) -> QBrush:
    return QBrush(status_color(kind))


PREV_NAME = "Original study settings"


def study_key(study) -> str:
    """Study signature without the variant / sweep definitions and the fatigue settings (the results of the variants stay valid until they change)."""
    d = study.to_dict()
    d.pop("fatigue", None)
    d.get("printing", {}).pop("variants", None)
    d.get("printing", {}).pop("required_safety", None)
    d.get("printing", {}).pop("sweep", None)            # the sweep definition does not change the results
    return json.dumps(d, sort_keys=True, default=str)


def adopt_new_mesh(win, run, mesh_key):
    """Mesh newly created during the variant calculation → main window (further calculations without Gmsh)."""
    base = run.outcome(BASE_NAME) if run is not None else None
    if base is not None and base.results is not None and win.valid_mesh() is None \
            and win.study_signature(win.MESH_KEYS) == mesh_key:
        win.mesh = base.results.mesh
        win._mesh_sig = mesh_key


def adopt_variant(win, run, v: PrintVariant):
    """Settings of a variant → study (the study results belonged to another setting); the study settings row in the results
    now describes the original setting.
    """
    st = win.study
    new = apply_variant(st, v)
    st.printing.parts = new.printing.parts
    st.part_materials = new.part_materials
    win.changed()
    prev = run.outcome(BASE_NAME)
    if prev is not None:
        prev.name = PREV_NAME
        if run.sweep:
            run.sweep["pareto"] = [PREV_NAME if n == BASE_NAME else n for n in run.sweep.get("pareto", [])]
    if run.recommended == BASE_NAME:
        run.recommended = PREV_NAME


def adopt_results(win, o) -> bool:
    """Results of a variant → main window (without recalculation); False = the variant has no results."""
    if not (o.ok and o.results is not None):
        return False
    win.results = o.results
    win.mesh = o.results.mesh
    win._mesh_sig = win.study_signature(win.MESH_KEYS)
    return True


class _Runner:
    """Common background calculation of variants (VariantsBox, SweepBox, AdviceBox): blocks meshing and calculation in the main
    window. The child has win, status, btn_run, btn_cancel, _thread, run_result / _run_key, _mesh_key, possibly progress.
    """
    TITLE = ""
    PROGRESS_TEXT = "Computing {i}/{n}: {name}…"
    progress = None                                      # QProgressBar (VariantsBox does not have it)

    @property
    def study(self):
        return self.win.study

    def valid_run(self):
        """Results of the variants, if they match the current study."""
        if self.run_result is None or self.study is None or self._run_key != study_key(self.study):
            return None
        return self.run_result

    def _add_run_buttons(self, h, text: str, slot, tip: str):
        """Calculate (btn_run) and Cancel (btn_cancel) buttons into the row h."""
        self.btn_run = QPushButton(text)
        short, _, rest = tip.partition("\n")
        help_tip(self.btn_run, short, tip) if rest else self.btn_run.setToolTip(tip)
        self.btn_run.clicked.connect(slot)
        h.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("Cancel")
        help_tip(self.btn_cancel, "Stops the running background computation",
                 "Stops the running background computation (including running meshing). Active only during a computation.")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        h.addWidget(self.btn_cancel)

    def _sync_buttons(self):
        busy = self._thread is not None
        self.btn_run.setEnabled(not busy and self.win._thread is None)
        self.btn_cancel.setEnabled(busy)

    def _ready(self) -> bool:
        win = self.win
        if win.study is None or win._thread is not None or self._thread is not None:
            return False
        errors, _ = win.check()
        errors += win.rigid_errors()
        if errors:
            QMessageBox.warning(self, self.TITLE, "The study is not ready:\n\n" + "\n".join(errors))
            return False
        return True

    def _begin(self, prefix: str):
        """Start: geometry into the study, signature of the meshing inputs; returns (study key, work folder of the calculation)."""
        win = self.win
        win.study.geometry = str(win.step_path)
        self._mesh_key = win.study_signature(win.MESH_KEYS)
        return study_key(win.study), win._workdir / f"{prefix}_{int(time.time() * 1000)}"

    def _start(self, t: TaskThread, status: str):
        win = self.win
        t.log.connect(win.log_msg)
        t.solver_line.connect(lambda s: win.log.appendPlainText(s) if s.strip() else None)
        t.progress.connect(self._on_progress)
        t.done.connect(self.on_done)
        t.failed.connect(self.on_failed)
        t.finished.connect(self._on_finished)
        self._thread = t
        win._thread = t                                  # blocks meshing and calculation in the main window
        if self.progress is not None:
            self.progress.setValue(0)
            self.progress.show()
        self.status.setText(status)
        self._sync_buttons()
        t.start()

    def _on_progress(self, i, n, name):
        if self.progress is not None:
            self.progress.setRange(0, max(n, 1))
            self.progress.setValue(i)
        self.status.setText(self.PROGRESS_TEXT.format(i=i + 1, n=n, name=name))

    def cancel(self):
        if self._thread is not None:
            self._thread.cancel.set()
            kill_mesh_workers()
            self.status.setText("Cancelling…")

    def _on_finished(self):
        self._thread = None
        if self.progress is not None:
            self.progress.hide()
        self.win._thread_finished()
        self._refresh_after()

    def _refresh_after(self):
        self.win.print_panel.refresh()

    def on_failed(self, text: str):
        t = self._thread
        if t is not None and t.cancel.is_set():
            self.status.setText("Cancelled.")
            self.win.log_msg(f"{self.TITLE}: cancelled by the user.")
            return
        self.status.setText(f"{self.TITLE}: computation failed.")
        self.win.log_msg(f"ERROR ({self.TITLE}): {text}")
        QMessageBox.critical(self, self.TITLE, f"Analysis failed:\n\n{text}")

    def _store_run(self, run, lost: str, unit: str):
        """Finished variant calculation → run_result (only if the study has not changed in the meantime), mesh and results table."""
        if self.study is None or study_key(self.study) != self._run_key:
            self.status.setText(f"The study changed during the computation – {lost} discarded.")
            return
        self.run_result = run
        ok = sum(o.ok for o in run.outcomes)
        self.status.setText(f"Done: {ok}/{len(run.outcomes)} {unit} in {fmt(run.time_s, 3)} s.")
        adopt_new_mesh(self.win, run, self._mesh_key)
        self.show_results()


class Plot(LazyPlot):
    """Chart mass × min. safety of the variants."""

    def draw(self, run):
        ax = self.axes()
        pts = [o for o in run.outcomes if o.ok and o.mass_g is not None]
        for o in pts:
            rec = o.name == run.recommended
            ax.scatter([o.mass_g], [o.min_safety], s=55 if rec else 30, zorder=3,
                       color="#2e7d32" if rec else ("#1f77b4" if o.min_safety >= run.required_safety else "#c62828"),
                       edgecolor="black" if rec else "none")
            ax.annotate(o.name, (o.mass_g, o.min_safety), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.axhline(run.required_safety, color="#555", lw=0.8, ls="--")
        ax.set_xlabel("mass of printed parts [g]")
        ax.set_ylabel("min. print safety factor [–]")
        ax.grid(alpha=0.3)
        self.canvas.draw_idle()


class VariantsBox(Section, _Runner):
    """Group in the 3D printing panel. panel = PrintPanel (print settings form, part list)."""

    TITLE = "Print variants"
    PROGRESS_TEXT = "Computing variant {i}/{n}: {name}…"
    COLS = ["Variant", "Parts", "Perimeters", "Solid layers h/d", "Infill", "Line / layer", "Z direction", "Material"]
    RCOLS = ["Variant", "Min. safety factor", "Mass [g]", "Max. displacement [mm]", "Governing", "Time [s]"]

    def __init__(self, panel):
        super().__init__("Internal structure variants – strength comparison", key="print.variants",
                         collapsed=True, tip="Compares several print settings on one mesh.",
                         help="<b>Print variants</b><p>Compares settings of perimeters, infill, pattern, build direction and material on one mesh. The results show safety, mass and displacement.</p>")
        self.panel = panel
        self.win = panel.win
        self.run_result = None             # VariantsRun
        self._run_key = None               # study_key when the calculation starts
        self._mesh_key = None              # signature of the meshing inputs when it starts
        self._thread = None
        lay = QVBoxLayout(self.body)
        note = QLabel("A variant = print settings (form above) for the selected parts. All variants are computed on a single mesh and compared with the study settings; the lightest one that meets the required safety factor is recommended.")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.table = table(self.COLS, placeholder="No variants yet – add them from the form",
                           rows=(3, 8))
        self.table.setToolTip("Defined print variants")
        self.table.setWhatsThis(
            "<b>Print variants</b><p>A variant = an override of the print settings (and optionally the printed material) for the selected parts; the other parts stay as in the study. Create a new variant by setting up the “Print settings of selected parts” form, selecting the parts and clicking <i>Add from form…</i>.</p><p><i>Compute variants</i> computes the study settings and all variants on a single mesh (no re-meshing) and compares the results in the table below. Variant definitions are saved with the study; study results are not changed.</p>")
        self.table.doubleClicked.connect(lambda _: self.load_form())
        lay.addWidget(self.table)
        h = QGridLayout()                       # 3 buttons per row – the print panel fits a narrower window
        for k, (txt, tip, fn) in enumerate((
                ("Add from form…", "New variant from the settings in the form for the selected parts\nNo parts selected = all printed parts; the material comes from the printed material selection.", self.add),
                ("Update from form", "Overwrites the selected variant with the settings from the form (for the selected parts)", self.update),
                ("To form", "Loads the settings of the selected variant into the form (double-click)", self.load_form),
                ("Copy…", "Copy of the selected variant under a new name", self.copy),
                ("Remove", "Removes the selected variant", self.remove))):
            b = QPushButton(txt)
            short, _, rest = tip.partition("\n")
            help_tip(b, short, tip) if rest else b.setToolTip(tip)
            b.clicked.connect(fn)
            h.addWidget(b, k // 3, k % 3)
        lay.addLayout(h)
        h = QHBoxLayout()
        h.addWidget(QLabel("Required safety factor"))
        self.k_req = dspin(0.1, 0.1, 100.0, 2, 0.1)
        self.k_req.setToolTip("Min. print safety factor (Tsai-Wu/Hill, load multiplier) for variant recommendation")
        self.k_req.editingFinished.connect(self.on_k_req)
        h.addWidget(self.k_req)
        self._add_run_buttons(h, "Compute variants", self.compute,
                               "Study settings + all variants on a single mesh (no re-meshing)")
        h.addStretch(1)
        lay.addLayout(h)
        self.status = QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.rtable = table(self.RCOLS, placeholder="Results appear after the variants are computed", rows=(3, 8))
        help_tip(self.rtable, "Variant results: safety, mass, displacement, failure location",
                 "Results of the last variants run: min. print safety factor, mass of printed parts, max. displacement and governing location (part · region · case).\nDark green = recommended, green = meets the required safety factor, red = does not meet it / error.")
        lay.addWidget(self.rtable)
        self.reason = QLabel()
        self.reason.setWordWrap(True)
        lay.addWidget(self.reason)
        self.plot = Plot()
        self.plot.hide()
        lay.addWidget(self.plot)
        b = QPushButton("Adopt the selected variant into the study (with its results)")
        help_tip(b, "Adopts the variant into the study together with its results",
                 "Print settings of the selected variant → study; its results → Results step (no recalculation)")
        b.clicked.connect(self.adopt)
        lay.addWidget(b)

    # ================================================================== stav
    def variants(self) -> list[PrintVariant]:
        return self.study.printing.variants if self.study is not None else []

    def current(self) -> PrintVariant | None:
        r = self.table.currentRow()
        v = self.variants()
        return v[r] if 0 <= r < len(v) else None

    def refresh(self):
        st = self.study
        if st is None:
            return
        self.k_req.blockSignals(True)
        self.k_req.setValue(st.printing.required_safety)
        self.k_req.blockSignals(False)
        names = self.panel.part_names()
        cur = self.table.currentRow()
        self.table.setRowCount(0)
        for v in self.variants():
            eff = effective(st, v, names)
            ov = [p for p in v.parts if p in eff]
            vals = [eff[p] for p in ov] or list(eff.values())

            def col(fn):
                xs = {fn(d) for d in vals}
                return xs.pop() if len(xs) == 1 else "various"
            row = [v.name, ", ".join(ov) if ov else "–",
                   col(lambda d: str(d["walls"]) if d["enabled"] else "solid"),
                   col(lambda d: f"{d['top_layers']} / {d['bottom_layers']}"),
                   col(lambda d: f"{fmt(100 * d['infill'], 3)} % {pattern_label(d['pattern'])}"),
                   col(lambda d: f"{fmt(d['line_width'], 3)} / {fmt(d['layer_height'], 3)} mm"),
                   col(lambda d: self.panel.dir_text(d["build_dir"])),
                   col(lambda d: d["material"])]
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, txt in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(txt))
        if 0 <= cur < self.table.rowCount():
            self.table.selectRow(cur)
        self.show_results()
        self._sync_buttons()

    def show_results(self):
        run = self.valid_run()
        self.rtable.setRowCount(0)
        if run is None:
            self.reason.setText("<i>Variant results do not match the current study – recompute.</i>"
                                if self.run_result is not None else "")
            self.plot.hide()
            return
        bold = QFont()
        bold.setBold(True)
        for o in run.outcomes:
            r = self.rtable.rowCount()
            self.rtable.insertRow(r)
            if o.ok:
                g = o.governing
                row = [o.name, fmt(o.min_safety, 3), fmt(o.mass_g), fmt(o.max_disp),
                       f"{g.get('part')} · {g.get('region')} · {g.get('step')}", fmt(o.time_s, 3)]
                bg = REC_BG if o.name == run.recommended else (OK_BG if o.min_safety >= run.required_safety else BAD_BG)
            else:
                row = [o.name, "error", "", "", o.error, fmt(o.time_s, 3)]
                bg = BAD_BG
            for c, txt in enumerate(row):
                it = QTableWidgetItem(txt)
                it.setBackground(status_brush(bg))
                if o.name == run.recommended:
                    it.setFont(bold)
                if not o.ok and c == 4:
                    it.setToolTip(o.error)
                self.rtable.setItem(r, c, it)
        self.reason.setText(escape(str(run.reason)) + f"<br><i>Mesh {fmt(run.mesh_info.get('nodes'), 6)} nodes ({('new' if run.mesh_info.get('meshed') else 'adopted')}), {fmt(run.time_s, 3)} s in total. Print safety = the multiple of the load to failure by the criterion (indicative infill factors – calibrate with tests).</i>")
        self.plot.show()
        self.plot.draw(run)

    # ================================================================== definice
    def _target_parts(self) -> list[str]:
        from fea_core.printing.region import print_parts
        names = self.panel.part_names()
        printed = print_parts(self.study, names)
        sel = [p for p in self.panel.selected_parts() if p in printed]
        return sel or printed

    def _form_overrides(self, parts: list[str]) -> dict[str, dict] | None:
        """Settings from the form (all fields) + the printed material of the selection, if it differs from the part."""
        from .print_panel import print_materials
        st = self.study
        mname = self.panel.mat.currentText()
        m = next((x for x in print_materials(st) if x.name == mname), None)
        out = {}
        for p in parts:
            sp = self.panel.spec_from_ui(p)
            ov = {k: getattr(sp, k) for k in SPEC_KEYS}
            if m is not None and m.name != st.material_of(p).name:
                if m.name not in st.materials:
                    from fea_core.study import Material
                    st.materials[m.name] = Material(**m.__dict__)
                ov["material"] = m.name
            out[p] = ov
        return out

    def _name(self, title: str, default: str) -> str | None:
        names = {v.name for v in self.variants()} | {BASE_NAME}
        while True:
            name, ok = QInputDialog.getText(self, title, "Variant name:", text=default)
            if not ok:
                return None
            name = name.strip()
            if name and name not in names:
                return name
            QMessageBox.warning(self, title, "The name must be non-empty and unique.")

    def _store(self, v: PrintVariant, index: int | None = None):
        errs = validate(self.study, v, self.panel.part_names())
        if errs:
            QMessageBox.warning(self, self.TITLE, "\n".join(errs))
            return False
        lst = self.study.printing.variants
        if index is None:
            lst.append(v)
        else:
            lst[index] = v
        self.win.changed(invalidate_results=False)
        if index is None:
            self.table.selectRow(len(lst) - 1)
        return True

    def add(self):
        if self.study is None:
            return
        parts = self._target_parts()
        if not parts:
            QMessageBox.information(self, self.TITLE, "No part has a printed material.")
            return
        sp = self.panel.spec_from_ui("")
        default = (f"{sp.walls} perimeters, {fmt(100 * sp.infill, 3)} % {pattern_label(sp.pattern)}"
                   if sp.enabled else "solid part")
        name = self._name("New variant", default)
        if name:
            self._store(PrintVariant(name, self._form_overrides(parts)))

    def update(self):
        v = self.current()
        if v is None:
            QMessageBox.information(self, self.TITLE, "Select a variant in the table.")
            return
        parts = dict(v.parts)
        parts.update(self._form_overrides(self._target_parts()))
        self._store(PrintVariant(v.name, parts, v.note), self.table.currentRow())

    def load_form(self):
        v = self.current()
        if v is None:
            return
        parts = self._target_parts()
        p = next((x for x in parts if x in v.parts), next(iter(v.parts), None))
        if p is None:
            return
        st = apply_variant(self.study, v)
        self.panel.fill_spec(st.printing.spec(p))
        m = st.material_of(p)
        if m.is_print:
            self.panel.mat.setCurrentText(m.name)

    def copy(self):
        v = self.current()
        if v is None:
            return
        name = self._name("Variant copy", v.name + " (kopie)")
        if name:
            self._store(PrintVariant(name, {k: dict(d) for k, d in v.parts.items()}, v.note))

    def remove(self):
        v = self.current()
        if v is None:
            return
        self.study.printing.variants = [x for x in self.variants() if x is not v]
        self.win.changed(invalidate_results=False)

    def on_k_req(self):
        if self.study is None or abs(self.k_req.value() - self.study.printing.required_safety) < 1e-12:
            return
        self.study.printing.required_safety = float(self.k_req.value())
        run = self.run_result
        if run is not None:
            from fea_core.printing.variants import recommend
            run.required_safety = self.study.printing.required_safety
            run.recommended, run.reason = recommend(run.outcomes, run.required_safety)
        self.win.changed(invalidate_results=False)

    # ================================================================== calculation
    def compute(self):
        win = self.win
        st = self.study
        if st is None or win._thread is not None or self._thread is not None:
            return
        if not self._target_parts():
            QMessageBox.information(self, self.TITLE, "No part has a printed material.")
            return
        if not self._ready():
            return
        self._run_key, wd = self._begin("variants")
        mesh = win.valid_mesh()
        win.log_msg(f"Print variants: {len(st.printing.variants) + 1} calculations in the folder {wd}"
                    + (" (mesh reused)" if mesh is not None else " (mesh first)"))
        study, parts = win.study_for_solve(), win.parts

        def job(t):
            from fea_core.printing.variants import run_variants
            return run_variants(study, wd, mesh=mesh, parts=parts, on_log=t.log.emit,
                                on_solver_line=t.solver_line.emit, cancel=t.cancel, on_progress=t.progress.emit)
        self._start(TaskThread(job, self), "Computing variants…")

    def _refresh_after(self):
        self.refresh()

    def on_failed(self, text):
        self.status.setText("Variants computation failed.")
        self.win.log_msg(f"Print variants ERROR: {text}")
        QMessageBox.critical(self, self.TITLE, f"Variants computation failed:\n\n{text}")

    def on_done(self, run):
        self._store_run(run, "variant results", "variant")

    def adopt(self):
        run = self.valid_run()
        r = self.rtable.currentRow()
        if run is None or not (0 <= r < len(run.outcomes)):
            QMessageBox.information(self, self.TITLE, "Select a computed variant in the results table.")
            return
        o = run.outcomes[r]
        win = self.win
        st = self.study
        if o.name == PREV_NAME:
            QMessageBox.information(self, self.TITLE, "The original settings are no longer in the study – adopt it as a variant or set it manually.")
            return
        if o.name != BASE_NAME:
            v = next((x for x in st.printing.variants if x.name == o.name), None)
            if v is None:
                return
            adopt_variant(win, run, v)
        if adopt_results(win, o):
            win.log_msg(f"Print variant '{o.name}' adopted with its results.")
        self._run_key = study_key(st)              # the other variants stay valid (same mesh and loads)
        win.refresh_all()
