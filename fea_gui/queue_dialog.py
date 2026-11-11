""""Job queue" dialog (fea_core.job_queue): several .fea projects solved one after another, each in its own background
process (`python -m fea_core solve-project`); the solved project is saved as `<name>_results.fea` next to the source. The queue
lives in the main window while the program runs (closing the dialog does not stop it); quitting the program kills the
running solver process.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                               QPushButton, QTableWidgetItem, QVBoxLayout)

from . import theme
from .widgets import help_tip, table

STATUS = {"pending": "waiting", "running": "running", "completed": "done", "failed": "failed", "cancelled": "cancelled"}


def queue_of(win):
    """The job queue of the main window (created at the first use; one job at a time – RAM goes to the solver)."""
    q = getattr(win, "job_queue", None)
    if q is None:
        from PySide6.QtCore import QCoreApplication
        from fea_core.job_queue import JobQueue
        q = win.job_queue = JobQueue(max_workers=1)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(lambda: _stop_all(q))
    return q


def _stop_all(q) -> None:
    for j in q.list_jobs():
        if j.status in ("pending", "running"):
            q.cancel(j.id)
    q.shutdown(wait=False, cancel_futures=True)


class QueueDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.q = queue_of(win)
        self.setWindowTitle("Job queue")
        self.setModal(False)
        self.resize(760, 420)
        lay = QVBoxLayout(self)
        intro = QLabel("Projects (.fea) are solved one after another in background processes – you can keep working. "
                       "Each solved project is saved as “<name>_results.fea” next to the source; the source is not changed.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        h = QHBoxLayout()
        b = QPushButton("Add projects…")
        theme.set_primary(b)
        help_tip(b, "Choose .fea projects to solve in the background")
        b.clicked.connect(self.add_files)
        h.addWidget(b)
        b = QPushButton("Add current project")
        help_tip(b, "Queue the open project (it must be saved)")
        b.clicked.connect(self.add_current)
        h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        self.tab = table(["Project", "Status", "Time [s]", "Result / error"], 200)
        self.tab.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.tab.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tab.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tab.setProperty("_i18n_done", True)
        help_tip(self.tab, "Queued projects: status, solve time and the result file or the error")
        self.tab.cellDoubleClicked.connect(lambda *_: self.open_result())
        lay.addWidget(self.tab, 1)
        h = QHBoxLayout()
        self.btn_open = QPushButton("Open result")
        help_tip(self.btn_open, "Open the solved project of the selected job in the main window")
        self.btn_open.clicked.connect(self.open_result)
        h.addWidget(self.btn_open)
        self.btn_log = QPushButton("Show log")
        help_tip(self.btn_log, "Open the solver output of the selected job")
        self.btn_log.clicked.connect(self.show_log)
        h.addWidget(self.btn_log)
        self.btn_cancel = QPushButton("Cancel job")
        help_tip(self.btn_cancel, "Cancel the selected waiting or running job (the solver process is killed)")
        self.btn_cancel.clicked.connect(self.cancel_selected)
        h.addWidget(self.btn_cancel)
        h.addStretch(1)
        b = QPushButton("Close")
        help_tip(b, "Close the dialog – the queue keeps running")
        b.clicked.connect(self.close)
        h.addWidget(b)
        lay.addLayout(h)
        self.summary = QLabel("")
        self.summary.setProperty("muted", True)
        lay.addWidget(self.summary)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.tab.itemSelectionChanged.connect(self._sync)
        self.refresh()

    # ------------------------------------------------------------------ jobs
    def add(self, paths) -> int:
        n = 0
        for p in paths:
            try:
                self.q.submit_project(Path(p))
                n += 1
            except ValueError as e:
                QMessageBox.warning(self, self.windowTitle(), str(e))
        if n:
            self.win.log_msg(f"Job queue: {n} project(s) added")
        self.refresh()
        return n

    def add_files(self):
        last = self.win.settings.value("last_dir", "")
        files, _ = QFileDialog.getOpenFileNames(self, "Projects to solve", last, "FEA project (*.fea)")
        if files:
            self.add(files)

    def add_current(self):
        w = self.win
        if w.study is None:
            return
        if w.project_path is None or w.dirty:
            r = QMessageBox.question(self, self.windowTitle(), "The project must be saved first. Save it now?",
                                     QMessageBox.Save | QMessageBox.Cancel)
            if r != QMessageBox.Save or not w.save():
                return
        self.add([w.project_path])

    def jobs(self):
        return sorted(self.q.list_jobs(), key=lambda j: j.created_at)

    def _selected(self):
        r = self.tab.currentRow()
        js = self.jobs()
        return js[r] if 0 <= r < len(js) and self.tab.selectedItems() else None

    def _sync(self):
        j = self._selected()
        self.btn_open.setEnabled(j is not None and j.status == "completed")
        self.btn_log.setEnabled(j is not None and j.log_path is not None and j.log_path.is_file())
        self.btn_cancel.setEnabled(j is not None and j.status in ("pending", "running"))

    def refresh(self):
        js = self.jobs()
        t = self.tab
        if t.rowCount() != len(js):
            t.setRowCount(len(js))
        for r, j in enumerate(js):
            dur = j.duration_s
            if j.status == "completed":
                res = str(j.output)
            else:
                lines = (j.error or "").splitlines()
                res = lines[-1] if lines else ""
            vals = [j.name, STATUS.get(j.status, j.status), "" if dur is None else f"{dur:.0f}", res]
            for c, txt in enumerate(vals):
                it = t.item(r, c)
                if it is None:
                    t.setItem(r, c, QTableWidgetItem(txt))
                elif it.text() != txt:
                    it.setText(txt)
            t.item(r, 0).setToolTip(str(j.source or ""))
        s = self.q.summary()
        self.summary.setText(f"{s['total']} jobs: {s['running']} running, {s['pending']} waiting, {s['completed']} done, "
                             f"{s['failed']} failed, {s['cancelled']} cancelled")
        self._sync()

    def open_result(self):
        j = self._selected()
        if j is None or j.status != "completed" or j.output is None:
            return
        self.win.open_file(str(j.output))

    def show_log(self):
        j = self._selected()
        if j is not None and j.log_path is not None and j.log_path.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(j.log_path)))

    def cancel_selected(self):
        j = self._selected()
        if j is not None and self.q.cancel(j.id):
            self.win.log_msg(f"Job queue: {j.name} cancelled")
        self.refresh()

    def closeEvent(self, e):
        self.timer.stop()
        super().closeEvent(e)


def wait_idle(q, timeout: float = 60.0) -> bool:
    """Waits until no job is waiting or running (tests)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not any(j.status in ("pending", "running") for j in q.list_jobs()):
            return True
        time.sleep(0.05)
    return False
