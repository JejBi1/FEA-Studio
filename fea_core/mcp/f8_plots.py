"""Graphs F8 to PNG (matplotlib Agg without pyplot – as mcp/render.py): force–displacement curves, FRF (amplitude and phase in
two separate graphs above each other), courses in time, PSD, SRS. One axis y per graph, the legend from 2 series.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# fixed order of the colours of the series (the identity of the series = the order in the specification, not the size)
SERIES = ["#1f5fbf", "#d9822b", "#2e8b57", "#7b4fb3", "#b8403a", "#4f8f9f", "#8c6d1f", "#5f5f5f"]


def _style(ax, title: str | None, xlabel: str, ylabel: str):
    if title:
        ax.set_title(title, fontsize=11, loc="left", color="#111")
    ax.set_xlabel(xlabel, color="#333")
    ax.set_ylabel(ylabel, color="#333")
    ax.grid(True, color="#dddddd", lw=0.8, which="major")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")
    ax.tick_params(colors="#555555")


def _fig(rows: int = 1, size=(9.6, 5.4)):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=size, dpi=100)
    FigureCanvasAgg(fig)
    axes = [fig.add_subplot(rows, 1, i + 1) for i in range(rows)]
    return fig, axes


def _save(fig, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(p))
    return p


def xy_plot(path, series: list[dict], title: str, xlabel: str, ylabel: str, logx: bool = False,
            logy: bool = False, hlines: list[tuple[float, str]] | None = None, size=(9.6, 5.4)) -> Path:
    """series = [{"x", "y", "label"}]; hlines = horizontal reference lines (value, description)."""
    fig, (ax,) = _fig(1, size)
    for i, s in enumerate(series):
        x, y = np.asarray(s["x"], float), np.asarray(s["y"], float)
        if logy:
            y = np.where(y > 0, y, np.nan)
        ax.plot(x, y, color=SERIES[i % len(SERIES)], lw=2.0, label=s.get("label"),
                marker="o" if len(x) <= 25 else None, ms=4)
    for v, lab in hlines or []:
        ax.axhline(v, color="#888888", lw=1.0, ls="--")
        ax.annotate(lab, (0.01, v), xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points",
                    fontsize=8, color="#555")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    _style(ax, title, xlabel, ylabel)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def frf_plot(path, f, H: list[dict], title: str, ylabel: str, logy: bool = True) -> Path:
    """FRF: H = [{"y" complex, "label"}] → amplitude (top) and phase [°] (bottom), log frequency."""
    fig, (a1, a2) = _fig(2, (9.6, 7.2))
    f = np.asarray(f, float)
    for i, s in enumerate(H):
        y = np.asarray(s["y"])
        col = SERIES[i % len(SERIES)]
        amp = np.abs(y)
        a1.plot(f, np.where(amp > 0, amp, np.nan) if logy else amp, color=col, lw=2.0, label=s.get("label"))
        a2.plot(f, np.degrees(np.angle(y)), color=col, lw=2.0, label=s.get("label"))
    for ax in (a1, a2):
        ax.set_xscale("log")
    if logy:
        a1.set_yscale("log")
    _style(a1, title, "frequency [Hz]", ylabel)
    _style(a2, None, "frequency [Hz]", "phase [°]")
    a2.set_ylim(-185, 185)
    a2.set_yticks([-180, -90, 0, 90, 180])
    if len(H) > 1:
        a1.legend(frameon=False, fontsize=9)
    return _save(fig, path)
