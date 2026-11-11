"""PNG of the load–displacement curve of the nonlinear buckling (matplotlib Agg without pyplot – MCP, Word report)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def curve_png(info: dict, path, width_px: int = 1100, height_px: int = 700, title: str | None = None) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from ..i18n import tr
    c = info.get("curve") or {}
    lam = np.asarray(c.get("lambda") or [], float)
    u = np.abs(np.asarray(c.get("u_mm") or [], float))
    dpi = 110
    fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, tight_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(u, lam, "o-", ms=3, lw=1.5, color="#2f6fd0", label=tr("load factor λ"))
    r = info.get("result") or {}
    if r.get("ultimate_factor") is not None and len(lam):
        i = int(r.get("ultimate_index", int(np.argmax(lam))))
        ax.plot([u[min(i, len(u) - 1)]], [lam[min(i, len(lam) - 1)]], "s", color="#d23b3b", ms=7,
                label=f"λ_u = {float(r['ultimate_factor']):.4g}")
    if info.get("linear_factor"):
        ax.axhline(float(info["linear_factor"]), ls="--", lw=1, color="#777777",
                   label=f"λ_cr ({tr('linear')}) = {float(info['linear_factor']):.4g}")
    if r.get("first_yield_factor") is not None:
        ax.axhline(float(r["first_yield_factor"]), ls=":", lw=1, color="#b36b00",
                   label=f"{tr('first yield')} λ = {float(r['first_yield_factor']):.4g}")
    ax.set_xlabel(tr("Displacement of the monitored point [mm]"))
    ax.set_ylabel(tr("Load factor λ"))
    ax.set_title(title or tr("Nonlinear buckling – load–displacement curve"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = Path(path)
    fig.savefig(path)
    return path
