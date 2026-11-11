"""Program modules (Tools → Program modules…): what is installed, installing and removing the optional parts.

Installs and removes with the same installer as the command line (installer/fea_setup.py, `python -m fea_core.modules`); it runs
as a separate process, its output is visible in the window. After a change the program must be restarted.
The hypre module on Windows needs Microsoft MPI (system installation, UAC) – the installer runs with --quiet, so MS-MPI
is installed only with explicit consent (--msmpi) after a question in this window.
"""
from __future__ import annotations

import sys

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPlainTextEdit, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from fea_core import config, modules


class ModulesDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.setWindowTitle("Program modules")
        self.setModal(False)
        self.resize(900, 640)
        self.proc: QProcess | None = None
        lay = QVBoxLayout(self)
        intro = QLabel("The program core (FEM engine) is small; the other parts are installed separately. Tick the modules you want and press “Apply changes”. The installation downloads packages from the internet; restart the program afterwards.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Module", "Description", "Size", "State"])
        self.table.setToolTip("Program modules. Ticked = installed / will be installed.")
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(True)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        lay.addWidget(self.table, 3)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setToolTip("Installer output")
        lay.addWidget(self.log, 2)
        row = QHBoxLayout()
        self.total = QLabel("")
        row.addWidget(self.total, 1)
        self.apply_btn = QPushButton("Apply changes")
        self.apply_btn.setToolTip("Installs the ticked modules and removes the unticked ones (including modules that depend on them)")
        self.apply_btn.clicked.connect(self.apply)
        row.addWidget(self.apply_btn)
        close = QPushButton("Close")
        close.setToolTip("Closes the window")
        close.clicked.connect(self.close)
        row.addWidget(close)
        lay.addLayout(row)
        self.status = modules.status()
        self.fill()
        self.table.itemChanged.connect(self.on_changed)
        self.update_total()

    def fill(self):
        self.table.blockSignals(True)
        mods = modules.all_modules()
        self.table.setRowCount(len(mods))
        for r, (k, m) in enumerate(mods.items()):
            have = self.status.get(k, False)
            why = "" if have else modules.problem(k)   # installed, but it cannot be used (hypre without MS-MPI…)
            name = QTableWidgetItem(f"{k} – {m.title()}")
            name.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled if not m.required else Qt.ItemIsEnabled)
            # a non-working module stays ticked → "Apply changes" installs it (for hypre with MS-MPI too, after a question)
            name.setCheckState(Qt.Checked if have or why or m.required else Qt.Unchecked)
            name.setData(Qt.UserRole, k)
            self.table.setItem(r, 0, name)
            self.table.setItem(r, 1, QTableWidgetItem(m.desc()))
            self.table.setItem(r, 2, QTableWidgetItem(f"~{m.size()} MB"))
            state = QTableWidgetItem("installed" if have else f"not working – {why}" if why else "not installed")
            if why:
                state.setToolTip("Installed but not usable – “Apply changes” adds the missing parts")
            self.table.setItem(r, 3, state)
            for c in (1, 2, 3):
                self.table.item(r, c).setFlags(Qt.ItemIsEnabled)
        self.table.resizeRowsToContents()
        self.table.blockSignals(False)

    def on_changed(self, item):
        """Ticking a module ticks its dependencies too; unticking unticks the modules that depend on it."""
        if item.column() != 0:
            return
        key = item.data(Qt.UserRole)
        mods = modules.all_modules()
        on = item.checkState() == Qt.Checked
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            k = it.data(Qt.UserRole)
            if on and k in modules.closure([key]) and k != key:
                it.setCheckState(Qt.Checked)
            elif not on and key in modules.closure([k]) and k != key and not mods[k].required:
                it.setCheckState(Qt.Unchecked)
        self.table.blockSignals(False)
        self.update_total()

    def wanted(self) -> list[str]:
        return [self.table.item(r, 0).data(Qt.UserRole) for r in range(self.table.rowCount())
                if self.table.item(r, 0).checkState() == Qt.Checked]

    def update_total(self, *_):
        add = [k for k in modules.closure(self.wanted()) if not self.status.get(k, False)]
        drop = [k for k, v in self.status.items() if v and k not in modules.closure(self.wanted())]
        mods = modules.all_modules()
        txt = []
        if add:
            txt.append(f"install: {', '.join(add)} (~{sum((mods[k].size() for k in add))} MB)")
        if drop:
            txt.append(f"remove: {', '.join(drop)}")
        self.total.setText("; ".join(txt) or "no changes")
        self.apply_btn.setEnabled(bool(add or drop) and self.proc is None)

    def apply(self):
        want = modules.closure(self.wanted())
        add = [k for k in want if not self.status.get(k, False)]
        drop = [k for k, v in self.status.items() if v and k not in want]
        steps = []
        if drop:
            steps.append(["remove", *drop])
        if add:
            steps.append(["install", *add, "--no-check", "--no-shortcuts"] + (["--msmpi"] if self.ask_msmpi(add) else []))
        self.queue = steps
        self.failed = False
        self.log.clear()
        self.apply_btn.setEnabled(False)
        self.next_step()

    def ask_msmpi(self, add: list[str]) -> bool:
        """Windows + installing the hypre module without mpiexec: consent to install Microsoft MPI (into the system, UAC)."""
        from fea_core.solvers import find_mpiexec
        if not config.IS_WIN or "hypre" not in add or find_mpiexec() is not None:
            return False
        r = QMessageBox.question(
            self, "Program modules",
            "The NGSolve + hypre solver needs Microsoft MPI (mpiexec), which is missing in the system.\n\nDownload and install it now? It is installed into the system – Windows asks for administrator rights (UAC).\nWithout it the hypre module is installed but the NGSolve + hypre solver will not work (it can be added later: python installer/fea_setup.py mpi).",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        return r == QMessageBox.Yes

    def next_step(self):
        if not self.queue:
            self.proc = None
            self.status = modules.status()
            self.fill()
            self.update_total()
            self.log.appendPlainText("\nDone. Restart the program for the changes to take effect.")
            if not getattr(self, "failed", False):
                QMessageBox.information(self, "Program modules",
                                        "The module changes are done. Restart FEA Studio for them to take effect (new solvers appear after a restart).")
            return
        args = self.queue.pop(0)
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.setWorkingDirectory(str(config.PROJECT_ROOT))
        self.proc.readyReadStandardOutput.connect(
            lambda: self.log.appendPlainText(bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace").rstrip()))
        self.proc.finished.connect(self.step_done)
        self.log.appendPlainText(f"> fea_setup.py {' '.join(args)}")
        self.proc.start(sys.executable, [str(config.PROJECT_ROOT / "installer" / "fea_setup.py"), "--quiet", *args])

    def step_done(self, code, _status):
        if code != 0:
            self.log.appendPlainText(f"\nError (code {code}) – the remaining steps are skipped.")
            self.queue = []
            self.failed = True
        self.next_step()
