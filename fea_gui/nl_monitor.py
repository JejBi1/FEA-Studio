"""Convergence monitor of a nonlinear analysis (A7) in the Solve step: residual force and displacement correction per
iteration of CalculiX, live during the run (the .cvg / .sta files in the work folder of the running analysis) and after it from
the saved results (`Results.info["nonlinear"]["convergence"]`, fea_core.nl_convergence)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QVBoxLayout

from .widgets import LazyPlot, Section, help_tip

POLL_MS = 1500


def histories_from_results(results) -> dict[str, dict]:
    """{job: ConvergenceHistory dict} saved with a nonlinear result (empty for a linear one)."""
    info = getattr(results, "info", None) or {}
    nl = info.get("nonlinear") or {}
    conv = nl.get("convergence") or {}
    return {str(k): v for k, v in conv.items() if isinstance(v, dict)}


def histories_from_folder(workdir) -> dict[str, dict]:
    """Live histories of the ccx jobs in the work folder of the running analysis (every *.cvg)."""
    from fea_core.nl_convergence import read_convergence_job
    out = {}
    wd = Path(workdir)
    try:
        cvgs = sorted(wd.glob("*.cvg"))
    except OSError:
        return out
    for cvg in cvgs:
        try:
            h = read_convergence_job(wd, cvg.stem)
        except (OSError, ValueError):
            continue
        if h.iterations or h.increments:
            out[cvg.stem] = h.to_dict()
    return out


def series(hist: dict) -> tuple[list[float], list[float], list[int]]:
    """Residual force [%], displacement correction [%] per iteration (in order) and the indices where a new increment starts."""
    rf, cd, starts = [], [], []
    prev = None
    for k, it in enumerate(hist.get("iterations") or []):
        key = (it.get("step"), it.get("inc"), it.get("att"))
        if key != prev:
            starts.append(k)
            prev = key
        rf.append(float(it.get("resid_force_pct") or 0.0))
        cd.append(float(it.get("corr_disp_pct") or 0.0))
    return rf, cd, starts


def summary_text(hists: dict[str, dict], live: bool) -> str:
    if not hists:
        return ("Waiting for the first iteration of CalculiX…" if live else
                "No convergence data (linear analysis, or results from an older version).")
    parts = []
    for job, h in hists.items():
        its = h.get("iterations") or []
        incs = h.get("increments") or []
        last = its[-1] if its else {}
        txt = (f"{job}: {len(incs)} increments, {len(its)} iterations, {h.get('cutbacks', 0)} cutbacks"
               + (f", last residual {float(last.get('resid_force_pct') or 0):.3g} %" if last else ""))
        if incs:
            txt += f", time {float(incs[-1].get('tot_time') or 0):.4g}"
        if not live and h.get("status_message"):
            txt += f" – {h['status_message']}"
        parts.append(txt)
    return "\n".join(parts)


class NlMonitor(Section):
    """Section of the Solve panel: chart of the iteration history (log scale) + a summary line. Visible for a nonlinear study or
    when the results carry convergence data; it polls the running analysis every POLL_MS."""

    def __init__(self, main):
        self.body_w = None
        super().__init__("Convergence (nonlinear analysis)", key="solve.nl_monitor", collapsed=False,
                         tip="Residual force and displacement correction per CalculiX iteration",
                         help="<b>Convergence monitor</b><p>For a nonlinear analysis (contacts, plasticity, NLGEOM, load "
                              "sequences) CalculiX iterates every increment until the residual force and the displacement "
                              "correction fall below the tolerance. The chart shows both per iteration (logarithmic scale); "
                              "a dotted line marks the start of a new increment or a cutback (repeated attempt with a smaller "
                              "increment). Many cutbacks or a residual that stops falling point to a contact that opens and "
                              "closes, a missing support or too large a step.</p><p>During the analysis the chart follows the "
                              "running job; after it the history is kept with the results.</p>")
        self.main = main
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(0, 0, 0, 0)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setProperty("muted", True)
        help_tip(self.info, "Increments, iterations, cutbacks and the last residual of each CalculiX job")
        lay.addWidget(self.info)
        self.plot = LazyPlot()
        help_tip(self.plot, "Residual force and displacement correction [%] per iteration (log scale)")
        lay.addWidget(self.plot)
        self._shown = None
        self.timer = QTimer(self)
        self.timer.setInterval(POLL_MS)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.toggled.connect(self._on_toggled)
        self.setVisible(False)

    def _on_toggled(self, opened):
        if opened:
            self._shown = None
            self.refresh()

    def _running_workdir(self):
        t = getattr(self.main, "_thread", None)
        wd = getattr(t, "workdir", None)
        st = getattr(t, "study", None)
        if wd is not None and st is not None and st.is_nonlinear():
            return wd
        return None

    def refresh(self):
        m = self.main
        wd = self._running_workdir()
        live = wd is not None
        hists = histories_from_folder(wd) if live else histories_from_results(getattr(m, "results", None))
        study = getattr(m, "study", None)
        show = live or bool(hists) or (study is not None and study.is_nonlinear())
        if self.isHidden() == show:                       # isHidden = hidden by setVisible(False) (not by the parent)
            self.setVisible(show)
        if not show:
            return
        key = (live, tuple((j, len(h.get("iterations") or [])) for j, h in hists.items()))
        if key == self._shown:
            return
        self._shown = key
        self.info.setText(summary_text(hists, live))
        self.set_badge(f"{sum(len(h.get('iterations') or []) for h in hists.values())} it." if hists else "")
        if self.body.isHidden():                          # collapsed: drawn when expanded (_on_toggled)
            return
        self._draw(hists)

    def _draw(self, hists):
        if not hists:
            if self.plot.canvas is not None:
                self.plot.axes()
                self.plot.canvas.draw_idle()
            return
        ax = self.plot.axes()
        off = 0
        colors = ["#2563eb", "#e69a1a", "#16a34a", "#d23b3b"]
        for n, (job, h) in enumerate(hists.items()):
            rf, cd, starts = series(h)
            x = list(range(off + 1, off + len(rf) + 1))
            c = colors[n % len(colors)]
            lab = f" ({job})" if len(hists) > 1 else ""
            ax.semilogy(x, [max(v, 1e-12) for v in rf], "o-", ms=3, lw=1.4, color=c, label="residual force" + lab)
            ax.semilogy(x, [max(v, 1e-12) for v in cd], "s--", ms=3, lw=1.0, color=c, alpha=0.6,
                        label="displacement correction" + lab)
            for s in starts[1:]:
                ax.axvline(off + s + 0.5, color="#888888", lw=0.6, ls=":")
            off += len(rf)
        ax.set_xlabel("iteration (all increments)")
        ax.set_ylabel("[%]")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=7)
        self.plot.canvas.draw_idle()
