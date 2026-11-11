"""Dialog for exporting the results to VTU (ParaView) and CSV (A6). The files are written in the background (`TaskThread`).

File names and columns are the same as in the MCP `fea_results_export` (fea_core.export_csv.export_results_csv):
node, element, part and state numbers from 1 as in the solver input (.inp).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout
)

from fea_core.export_csv import DIALECTS as CSV_DIALECTS, EXCEL, STANDARD
from . import theme
from .widgets import help_tip
from .workers import TaskThread

DIALECTS = [("Excel – semicolon and decimal comma", EXCEL.name), ("Standard – comma and decimal point", STANDARD.name)]


def csv_dialect(win=None) -> str:
    """CSV format in the GUI: the one last chosen in the export dialog (QSettings "csv_dialect"), otherwise by the language of the program
    (Czech – Excel: semicolon and decimal comma; English – standard: comma and decimal point).
    """
    from fea_core.i18n import language
    settings = getattr(win, "settings", None)
    v = settings.value("csv_dialect", "") if settings is not None else ""
    if v in CSV_DIALECTS:
        return v
    return EXCEL.name if language() == "cs" else STANDARD.name


class _Cancelled(Exception):
    """Export interrupted by the user (closing the dialog) between the files."""


class ExportResultsDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self._thread: TaskThread | None = None
        self._running = False
        self.setWindowTitle("Export results (VTU, CSV)")
        self.resize(560, 380)

        lay = QVBoxLayout(self)

        lbl = QLabel("<b>Export of results</b><p>Mesh with results to VTU for ParaView, nodal values, elements and reactions to CSV. Node, element, part and case numbers start at 1 as in the solver input (.inp).</p>")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        f = QFormLayout()

        # Target folder
        h_dir = QHBoxLayout()
        default_dir = str(win.workdir / "export" if getattr(win, "workdir", None) else Path.cwd() / "export")
        self.txt_dir = QLineEdit(default_dir)
        self.txt_dir.setToolTip("Folder the files are written to (created if it does not exist)")
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.setToolTip("Choose the export target folder in a dialog")
        self.btn_browse.clicked.connect(self._browse)
        h_dir.addWidget(self.txt_dir, 1)
        h_dir.addWidget(self.btn_browse)
        f.addRow("Target folder:", h_dir)

        # Formats
        self.cb_vtu = QCheckBox("VTK UnstructuredGrid (.vtu + .pvd for ParaView)")
        self.cb_vtu.setChecked(True)
        help_tip(self.cb_vtu, "tet10 mesh with displacements, stresses and node, element, part numbers for ParaView",
                 "<b>VTU</b><p>Quadratic tet10 mesh, displacements, stresses, von Mises, principal stresses and other case fields (temperature, contact, 3D printing). Solver numbers are in the arrays <i>NodeId</i>, <i>ElementId</i> and <i>PartNumber</i> (from 1); ParaView’s own “Point ID” starts at 0.</p>")
        f.addRow("VTU export:", self.cb_vtu)

        self.cb_csv_nodal = QCheckBox("Nodal values to CSV (.csv)")
        self.cb_csv_nodal.setChecked(True)
        help_tip(self.cb_csv_nodal, "Coordinates, displacements and stresses per node (node_id = solver node number)",
                 "<b>Nodal values</b><p>Columns node_id, part_id, part, n_parts (number of parts sharing the node), x, y, z, displacements, stress components, von Mises, principal stresses and other case fields. A missing value = an empty cell.</p>")
        f.addRow("CSV nodes:", self.cb_csv_nodal)

        self.cb_single = QCheckBox("All cases in one file")
        help_tip(self.cb_single, "Nodal values of all cases one below another with columns step_index and step_name",
                 "<b>One file</b><p>Instead of a file per case, one file <i>…_nodal_all.csv</i> (row = node × case). Combinations and envelopes are still written separately. Applies only to “All cases and combinations”.</p>")
        f.addRow("", self.cb_single)

        self.cb_elements = QCheckBox("Element table (element and node numbers)")
        help_tip(self.cb_elements, "Element number, part, nodes n1…n10, centroid and volume (as in the solver .inp)",
                 "<b>Element table</b><p>File <i>…_elements.csv</i>: element_id, part_id, part, element_type, nodes n1…n10 in CalculiX order, centroid and volume. For one selected case also the nodal values averaged over the element (these are not integration-point stresses).</p>")
        f.addRow("CSV elements:", self.cb_elements)

        self.cb_csv_rf = QCheckBox("Reactions table to CSV (_reactions.csv)")
        self.cb_csv_rf.setChecked(True)
        help_tip(self.cb_csv_rf, "Reaction forces and moments per case, with a TOTAL row (total reaction)",
                 "<b>Reactions</b><p>Row = case × support: step_index, step_name, selection, constraint_kind, Fx–Fz, F_mag, Mx–Mz, M_mag, moment point mref_x/y/z (centre of the mesh bounding box), result_kind (case / combination), combination_index. Each case ends with a TOTAL row = total reaction from the equilibrium check (a node shared by two supports counted once).</p>")
        f.addRow("CSV reactions:", self.cb_csv_rf)

        self.combo_dialect = QComboBox()
        for text, key in DIALECTS:
            self.combo_dialect.addItem(text, key)
        self.combo_dialect.setCurrentIndex(max(0, self.combo_dialect.findData(csv_dialect(win))))
        help_tip(self.combo_dialect, "Excel: semicolon and decimal comma; standard: comma and decimal point",
                 "<b>CSV format</b><p><i>Excel</i> – semicolon, decimal comma, UTF-8 with BOM: Czech Excel opens the file directly in columns. <i>Standard</i> – comma, decimal point, UTF-8 (pandas, ParaView, scripts, English Excel).</p>")
        f.addRow("CSV format:", self.combo_dialect)

        # Volba stavu
        self.combo_step = QComboBox()
        help_tip(self.combo_step, "Exported case, combination or envelope; “All” = each separately",
                 "Which load case, combination or envelope is exported (VTU, CSV of nodes, elements and reactions); “All” writes each case and combination to its own file.")
        self.combo_step.addItem("All cases and combinations", None)
        if win.results:
            for i, st in enumerate(win.results.steps):
                self.combo_step.addItem(f"State {i + 1}: {st.name}", ("step", i))
            if hasattr(win.study, "combinations") and win.study.combinations:
                for j, comb in enumerate(win.study.combinations):
                    prefix = "Envelope" if comb.envelope else "Combination"
                    self.combo_step.addItem(f"{prefix} K{j + 1:02d}: {comb.name}", ("comb", j))
        self.combo_step.currentIndexChanged.connect(self._sync)
        f.addRow("Load case:", self.combo_step)

        lay.addLayout(f)

        self.lbl_state = QLabel("")
        self.lbl_state.setProperty("muted", True)
        self.lbl_state.setWordWrap(True)
        lay.addWidget(self.lbl_state)

        # Buttons
        h_btn = QHBoxLayout()
        self.btn_export = QPushButton("Export")
        theme.set_primary(self.btn_export)
        self.btn_export.setToolTip("Writes the selected files to the target folder (in background)")
        self.btn_export.clicked.connect(self._export)
        self.btn_close = QPushButton("Close")
        self.btn_close.setToolTip("Close the dialog (a running export stops after the current file)")
        self.btn_close.clicked.connect(self.accept)
        h_btn.addStretch(1)
        h_btn.addWidget(self.btn_export)
        h_btn.addWidget(self.btn_close)
        lay.addLayout(h_btn)
        self._sync()

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Choose export folder", self.txt_dir.text())
        if d:
            self.txt_dir.setText(d)

    def _sync(self, *_):
        """"All states in one file" makes sense only when exporting all states."""
        self.cb_single.setEnabled(not self._running and self.combo_step.currentData() is None)

    # ------------------------------------------------------------------ export in the background
    def _export(self):
        if self._thread is not None and self._thread.isRunning():
            return
        text = self.txt_dir.text().strip()
        if not text:
            QMessageBox.warning(self, "Error", "Enter the export target folder.")
            return
        out_dir = Path(text)

        res = self.win.results
        if not res or not res.mesh or not len(res.mesh.nodes):
            QMessageBox.warning(self, "Error", "There are no results to export.")
            return

        study = self.win.study
        sel_step = self.combo_step.currentData()
        opts = {"vtu": self.cb_vtu.isChecked(), "nodal": self.cb_csv_nodal.isChecked(),
                "reactions": self.cb_csv_rf.isChecked(), "elements": self.cb_elements.isChecked(),
                "single_file": self.cb_single.isChecked() and sel_step is None,
                "dialect": self.combo_dialect.currentData() or STANDARD.name}
        if not (opts["vtu"] or opts["nodal"] or opts["reactions"] or opts["elements"]):
            QMessageBox.information(self, "Export", "Check at least one export format.")
            return
        settings = getattr(self.win, "settings", None)
        if settings is not None:
            settings.setValue("csv_dialect", opts["dialect"])

        def job(t):
            return self._write_files(t, res, study, out_dir, sel_step, opts)

        self._busy(True)
        self.lbl_state.setText("Exporting files…")
        thread = TaskThread(job, self)
        thread.done.connect(lambda r, th=thread, d=out_dir: self._on_done(th, d, r))
        thread.failed.connect(lambda err, th=thread: self._on_failed(th, err))
        self._thread = thread
        thread.start()

    @staticmethod
    def _write_files(t, res, study, out_dir: Path, sel_step, opts: dict):
        """The actual writing of the files (runs in a thread, the GUI does not touch it). Returns (list of files, warnings).
        opts: vtu, nodal, reactions, elements, single_file (bool), dialect ("excel" | "standard").
        """
        from fea_core.export_vtu import (safe_name, export_results_vtu, export_step_vtu, export_combination_vtu,
                                         result_file_name)
        from fea_core.export_csv import export_results_csv
        from fea_core.combinations import compute_combination

        def check():
            if t.cancel.is_set():
                raise _Cancelled()

        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = safe_name(study.name if study else "results")
        exported: list[Path] = []
        warns: list[str] = []
        combs = list(getattr(study, "combinations", []) or [])
        cache: dict = {}

        if opts.get("vtu"):
            check()
            if sel_step is None:
                exported.extend(export_results_vtu(res.mesh, res, out_dir, base_name=prefix, study=study,
                                                   warnings=warns))
            elif sel_step[0] == "step":
                s = res.steps[sel_step[1]]
                p = out_dir / result_file_name(prefix, "step", sel_step[1], s.name, "vtu")
                export_step_vtu(res.mesh, s, p)
                exported.append(p)
            elif sel_step[0] == "comb":
                comb = combs[sel_step[1]]
                p = out_dir / result_file_name(prefix, "comb", sel_step[1], comb.name, "vtu")
                export_combination_vtu(res.mesh, compute_combination(study, res, comb, cache), p)
                exported.append(p)

        if opts.get("nodal") or opts.get("reactions") or opts.get("elements"):
            csv_warns: list[str] = []
            exported.extend(export_results_csv(
                res, out_dir, base_name=prefix, study=study, select=sel_step, dialect=opts.get("dialect"),
                nodal=bool(opts.get("nodal")), reactions=bool(opts.get("reactions")),
                elements=bool(opts.get("elements")), single_file=bool(opts.get("single_file")),
                warnings=csv_warns, check=check, cache=cache))
            warns += [w for w in csv_warns if w not in warns]

        return exported, warns

    def _busy(self, on: bool):
        """During the writing the options cannot be changed or another export started."""
        self._running = on
        for w in (self.btn_export, self.btn_browse, self.txt_dir, self.cb_vtu, self.cb_csv_nodal,
                  self.cb_csv_rf, self.cb_elements, self.cb_single, self.combo_dialect, self.combo_step):
            w.setEnabled(not on)
        if not on:
            self._sync()

    def _on_done(self, th, out_dir: Path, result):
        if th is not self._thread:                    # stale thread (the dialog was closed)
            return
        self._busy(False)
        exported, warns = result
        self.lbl_state.setText(f"√ {len(exported)} files exported.")
        msg = f"{len(exported)} files were exported successfully to:\n{out_dir}\n\n"
        msg += "\n".join(f.name for f in exported[:8])
        if len(exported) > 8:
            msg += f"\n... and {len(exported) - 8} more files."
        if warns:
            msg += "\n\nWarnings:\n" + "\n".join(warns)
        QMessageBox.information(self, "Export completed", msg)

    def _on_failed(self, th, err: str):
        if th is not self._thread:
            return
        self._busy(False)
        self.lbl_state.setText("! Export failed.")
        QMessageBox.critical(self, "Export error", f"Export failed:\n{err}")

    def _stop_thread(self):
        """Interrupts the export after the current file is finished and waits for the thread (it must not touch the closed dialog)."""
        th = self._thread
        self._thread = None                           # late signals are ignored
        if th is not None and th.isRunning():
            th.cancel.set()
            th.wait(60000)

    def closeEvent(self, event):
        self._stop_thread()
        super().closeEvent(event)

    def reject(self):
        self._stop_thread()
        super().reject()

    def accept(self):
        self._stop_thread()
        super().accept()
