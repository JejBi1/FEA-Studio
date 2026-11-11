"""Natural frequencies / buckling and dynamics side by side with the mechanical states (run_study).

The three are independent ccx jobs on the same mesh (eigen.run_eigen and dynamics.run_dynamics compute their own prestress
steps, they do not use the static result), so they need not wait for the statics. The machine is split by
`ccx.parallel.plan` over all their jobs (load cases + modal + buckling + dynamics): the mechanics gets the rest of the
share (`parallel.budget` – its own parallel steps plan inside it), the side jobs run in `solve.Background` threads with
their own share. Results, warnings and errors are taken in the order of the serial calculation (mechanics, eigen,
dynamics): an error of the mechanics stops the side jobs; the side job results are joined after the mechanics, their
progress is deferred to their phases (progress.Progress.later), their log lines do not move the bar.
Serial (the old way) when there is nothing beside the mechanics, the plan gives one job (memory, cores,
FEA_CCX_PARALLEL=0) or the mechanics does not run in ccx (NGSolve has its own threads and memory).
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

from .ccx import parallel


class SideJobs:
    def __init__(self, study, mesh, workdir, log, on_line, cancel, thermal, prog=None):
        self.study, self.mesh, self.prog = study, mesh, prog
        self.eigen_bg = self.dyn_bg = None
        self.mech_share = None
        n_eig = int(study.modal.enabled) + int(study.buckling.enabled and bool(study.load_cases))
        dyn = bool(study.dynamics.enabled)
        if not (n_eig or dyn) or not study.load_cases or not parallel.enabled():
            return
        from .solve import choose_backend
        if not study.is_nonlinear() and choose_backend(study, mesh) != "ccx":
            return
        ndof = 3 * len(mesh.nodes)
        n_side = n_eig + int(dyn)
        workers, threads = parallel.plan(len(study.load_cases) + n_side, ndof)
        if workers < 2:
            return
        side = min(n_side, workers - 1)                 # at least one share stays for the mechanics
        self.mech_share = (workers - side, threads)
        from .progress import CcxLines
        from .solve import Background
        workdir = Path(workdir)

        def lines(key, parser):
            return prog.lines(on_line, key, parser, True) if prog is not None else on_line

        def eigen(stop, warnings):
            from .eigen import run_eigen
            return run_eigen(study, mesh, workdir / "eigen", log, eig_lines, stop, thermal)

        def dynamics(stop, warnings):
            from .dynamics.run import run_dynamics
            return run_dynamics(study, mesh, workdir / "dynamics", log, dyn_lines, stop)

        eig_lines = lines("eigen", CcxLines(n_eig)) if n_eig else None
        dyn_lines = lines("dynamics", CcxLines(2, "Dynamics")) if dyn else None
        if n_eig and dyn and side >= 2:
            self.eigen_bg = Background(_shared(eigen, side - 1, threads), cancel, "eigen")
            self.dyn_bg = Background(_shared(dynamics, 1, threads), cancel, "dynamics")
        elif n_eig and dyn:                             # one share for both: one after another beside the mechanics
            both = Background(_shared(lambda s, w: (eigen(s, w), dynamics(s, w)), side, threads), cancel,
                              "eigen + dynamics")
            self.eigen_bg = self.dyn_bg = both
        elif n_eig:
            self.eigen_bg = Background(_shared(eigen, side, threads), cancel, "eigen")
        else:
            self.dyn_bg = Background(_shared(dynamics, side, threads), cancel, "dynamics")
        log(f"Natural frequencies / dynamics computed concurrently with the load cases "
            f"({workers} CalculiX jobs × {threads} threads)")

    @property
    def active(self) -> bool:
        return self.mech_share is not None

    def mech_budget(self):
        """Share of the machine for the mechanical states (context manager)."""
        return parallel.budget(*self.mech_share) if self.active else nullcontext()

    def abort(self):
        for bg in {id(b): b for b in (self.eigen_bg, self.dyn_bg) if b is not None}.values():
            bg.abort()

    def join(self, res) -> None:
        """Results of the side jobs into res (as solve._add_eigen / _add_dynamics); errors in the serial order."""
        from .solve import StudyError
        from .eigen import EigenError
        from .dynamics.modal import DynamicsError
        prog = self.prog
        both = self.eigen_bg is not None and self.eigen_bg is self.dyn_bg
        out = None
        if self.eigen_bg is not None:
            if prog is not None:
                prog.phase("eigen", "Natural frequencies / buckling (CalculiX)")
            try:
                out, _ = self.eigen_bg.result()
            except (EigenError, DynamicsError) as e:
                self.abort()
                raise StudyError(str(e)) from None
            except BaseException:
                self.abort()
                raise
            res.modes = out[0] if both else out
        if self.dyn_bg is not None:
            if prog is not None:
                prog.phase("dynamics", "Dynamics (CalculiX + modal superposition)")
            try:
                dyn = out[1] if both else self.dyn_bg.result()[0]
            except DynamicsError as e:
                raise StudyError(str(e)) from None
            res.dynamics = dyn
            for w in res.dynamics.warnings:
                if w not in res.warnings:
                    res.warnings.append(w)


def _shared(fn, workers: int, threads: int):
    """fn(stop, warnings) run inside a share of the machine (parallel.budget is per thread → set in the job's thread)."""
    def run(stop, warnings):
        with parallel.budget(workers, threads):
            return fn(stop, warnings)
    return run
