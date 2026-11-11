"""Scripting Python console in the GUI (F1 / Workbench & CAE ergonomics).

Allows interactive writing and running of Python scripts directly over the open model:
  - Global variables: study, mesh, results, window, fea_core, np, Path
  - Running code (Ctrl+Enter or the Run button)
  - Record of the output and error messages
  - Buttons Load script... / Save script...
"""
from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QPlainTextEdit,
    QSplitter, QFileDialog, QLabel, QMessageBox
)

from . import theme


class ConsolePanel(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        splitter = QSplitter(Qt.Vertical)

        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)

        # Output console
        out_w = QWidget()
        out_lay = QVBoxLayout(out_w)
        out_lay.setContentsMargins(0, 0, 0, 0)
        out_lay.setSpacing(2)
        lbl_out = QLabel("Output:")
        lbl_out.setProperty("muted", True)
        out_lay.addWidget(lbl_out)
        self.output_edit = QPlainTextEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setFont(mono)
        self.output_edit.setToolTip("Output of the executed Python script (standard output and error messages)")
        out_lay.addWidget(self.output_edit)
        splitter.addWidget(out_w)

        # Code editor
        in_w = QWidget()
        in_lay = QVBoxLayout(in_w)
        in_lay.setContentsMargins(0, 0, 0, 0)
        in_lay.setSpacing(2)
        lbl_in = QLabel("Python script (study, mesh, results, fea_core):")
        lbl_in.setProperty("muted", True)
        in_lay.addWidget(lbl_in)
        self.input_edit = QPlainTextEdit()
        self.input_edit.setFont(mono)
        self.input_edit.setToolTip("Python script editor – enter code and press Run (Ctrl+Enter)")
        self.input_edit.setPlaceholderText("# Write Python code and press Run (Ctrl+Enter)\nprint('Study name:', study.name if study else 'None')")
        in_lay.addWidget(self.input_edit)

        # Button bar
        btn_bar = QHBoxLayout()
        btn_run = QPushButton("Run (Ctrl+Enter)")
        theme.set_primary(btn_run)
        btn_run.setToolTip("Run the specified Python script in application environment (Ctrl+Enter)")
        btn_run.clicked.connect(self.run_code)
        btn_bar.addWidget(btn_run)

        btn_clear = QPushButton("Clear output")
        btn_clear.setToolTip("Clear output console text")
        btn_clear.clicked.connect(self.output_edit.clear)
        btn_bar.addWidget(btn_clear)

        btn_load = QPushButton("Open script...")
        btn_load.setToolTip("Load Python script from .py file")
        btn_load.clicked.connect(self.load_script)
        btn_bar.addWidget(btn_load)

        btn_save = QPushButton("Save script...")
        btn_save.setToolTip("Save current editor content to .py file")
        btn_save.clicked.connect(self.save_script)
        btn_bar.addWidget(btn_save)

        btn_bar.addStretch(1)
        in_lay.addLayout(btn_bar)

        splitter.addWidget(in_w)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        lay.addWidget(splitter)

        # Keyboard shortcuts for running
        self._sc_return = QShortcut(QKeySequence("Ctrl+Return"), self)
        self._sc_return.activated.connect(self.run_code)
        self._sc_enter = QShortcut(QKeySequence("Ctrl+Enter"), self)
        self._sc_enter.activated.connect(self.run_code)

    def write_output(self, text: str, error: bool = False):
        cursor = self.output_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.output_edit.setTextCursor(cursor)
        self.output_edit.insertPlainText(text)
        cursor.movePosition(QTextCursor.End)
        self.output_edit.setTextCursor(cursor)

    def run_code(self):
        code = self.input_edit.toPlainText().strip()
        if not code:
            return

        import numpy as np
        import fea_core

        env = {
            "window": self.main_window,
            "study": getattr(self.main_window, "study", None),
            "mesh": getattr(self.main_window, "mesh", None) or (self.main_window.results.mesh if getattr(self.main_window, "results", None) else None),
            "results": getattr(self.main_window, "results", None),
            "fea_core": fea_core,
            "np": np,
            "Path": Path,
        }

        # Redirection of stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sio = io.StringIO()
        sys.stdout = sio
        sys.stderr = sio

        self.write_output("\n>>> Running script...\n")
        try:
            exec(code, env)
            output = sio.getvalue()
            if output:
                self.write_output(output)
            self.write_output(">>> Done.\n")
        except SystemExit:
            output = sio.getvalue()
            if output:
                self.write_output(output)
            self.write_output(">>> Script exited (sys.exit).\n")
        except Exception:
            output = sio.getvalue()
            if output:
                self.write_output(output)
            err = traceback.format_exc()
            self.write_output(f"ERROR: {err}\n", error=True)
        except BaseException as e:
            output = sio.getvalue()
            if output:
                self.write_output(output)
            self.write_output(f"ABORTED: {e}\n", error=True)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def load_script(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open Python Script", "", "Python scripts (*.py);;All files (*.*)")
        if fn:
            try:
                txt = Path(fn).read_text(encoding="utf-8", errors="replace")
                self.input_edit.setPlainText(txt)
            except Exception as e:
                QMessageBox.critical(self, "Read error", f"The file cannot be opened:\n{e}")

    def save_script(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save Python Script", "", "Python scripts (*.py);;All files (*.*)")
        if fn:
            try:
                Path(fn).write_text(self.input_edit.toPlainText(), encoding="utf-8")
            except Exception as e:
                QMessageBox.critical(self, "Write error", f"Cannot save file:\n{e}")
