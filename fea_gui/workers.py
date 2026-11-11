"""Long jobs outside the GUI thread (mesh and calculation run in separate processes as before)."""
from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal


def kill_mesh_workers() -> int:
    """Backup for cancellation: terminates the Gmsh meshing processes (python -m fea_core.mesh.worker) started by this program.
    Normally it is enough to set `cancel` of the thread – generate_mesh ends the process itself. Returns the number of terminated processes.
    """
    try:
        import psutil
    except ImportError:
        return 0
    n = 0
    for p in psutil.Process().children(recursive=True):
        try:
            if any("fea_core.mesh.worker" in a for a in p.cmdline()):
                p.kill()
                n += 1
        except psutil.Error:
            pass
    return n


def kill_ccx_processes() -> int:
    """"Stop and keep the results" (F8, spec 4.9): terminates the running CalculiX (ccx*) processes of this program WITHOUT
    a cancellation setting – the nonlinear calculation then ends as a non-convergence and the core returns the last converted increment
    (StepResult.partial). Returns the number of terminated processes.
    """
    try:
        import psutil
    except ImportError:
        return 0
    n = 0
    for p in psutil.Process().children(recursive=True):
        try:
            if p.name().lower().startswith("ccx"):
                p.kill()
                n += 1
        except psutil.Error:
            pass
    return n


class TaskThread(QThread):
    """Calculation in the background: fn(thread) → result; signals for the log, solver lines, progress and finished variants."""
    log = Signal(str)
    solver_line = Signal(str)
    progress = Signal(int, int, str)
    outcome = Signal(object)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.cancel = threading.Event()

    def run(self):
        try:
            self.done.emit(self.fn(self))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e) or e.__class__.__name__)


class MeshThread(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, study, parts, workdir, parent=None):
        super().__init__(parent)
        self.study, self.parts, self.workdir = study, parts, workdir
        self.cancel = threading.Event()

    def run(self):
        try:
            from fea_core.solve import mesh_study           # also with the bolts (imprints, without the bolt parts) and welds
            mesh, warnings = mesh_study(self.study, self.parts, self.workdir, self.cancel)
            mesh.report["study_warnings"] = list(warnings)     # taking the mesh over into the calculation (prepare_mesh)
            mesh.report.setdefault("warnings", [])
            for w in warnings:
                if w not in mesh.report["warnings"]:
                    mesh.report["warnings"].append(w)
            self.done.emit(mesh)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e) or e.__class__.__name__)


class SolveThread(QThread):
    log = Signal(str)
    solver_line = Signal(str)
    progress = Signal(float, float, str)          # value 0–1, end of the phase, text (fea_core.progress)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, study, workdir, parent=None, mesh=None, parts=None):
        super().__init__(parent)
        self.study, self.workdir = study, workdir
        self.mesh, self.parts = mesh, parts         # finished mesh (meshing inputs unchanged) → without Gmsh
        self.cancel = threading.Event()

    def run(self):
        try:
            from fea_core.solve import run_study
            res = run_study(self.study, self.workdir, on_log=self.log.emit,
                            on_solver_line=self.solver_line.emit, cancel=self.cancel, mesh=self.mesh, parts=self.parts,
                            on_progress=self.progress.emit)
            self.done.emit(res)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e) or e.__class__.__name__)
