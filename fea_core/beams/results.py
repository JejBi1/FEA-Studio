"""Summaries of the results of the frame system (MCP, GUI, report)."""
from __future__ import annotations

import numpy as np

from .model import LOCAL_NAMES


def frame_summary(frame: dict) -> dict:
    """Extremes of the internal forces and stresses by bars, max. displacement of the nodes, reactions."""
    out = {"members": {}, "reactions": frame.get("reactions", {})}
    for name, m in frame.get("members", {}).items():
        d = {}
        for k in LOCAL_NAMES:
            v = np.asarray(m.get(k, []), float)
            if len(v):
                i = int(np.argmax(np.abs(v)))
                d[k] = {"max_abs": float(abs(v[i])), "value": float(v[i]), "s_mm": float(m["s"][i])}
        sig = np.asarray(m.get("sigma_eq", []), float)
        if len(sig):
            i = int(np.argmax(sig))
            d["sigma_eq_max_MPa"] = float(sig[i])
            d["sigma_eq_at_s_mm"] = float(m["s"][i])
        u = np.asarray(m.get("u", []), float)
        if len(u):
            d["u_max_mm"] = float(np.linalg.norm(u, axis=1).max())
        out["members"][name] = d
    nodes = frame.get("nodes", {})
    if nodes:
        U = np.array([v[:3] for v in nodes.values()])
        out["max_node_displacement_mm"] = float(np.linalg.norm(U, axis=1).max())
    out["node_residual"] = frame.get("node_residual")
    return out
