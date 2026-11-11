"""Summary of the results of one iteration of the geometry (KPI) – for the table of iterations and the comparison of versions.

The maximum stress is also given without the singularities (diagnostics.hotspots): the value in a sharp corner or at the edge of a clamping
grows with the refinement of the mesh and the comparison of the versions would be misleading. Reactions by selections (the names of the selections survive the update).
"""
from __future__ import annotations

import math
import tempfile
import time

import numpy as np


def _session(study, results, parts):
    from ..mcp.session import Session
    s = Session(base_dir=tempfile.gettempdir())
    s.study, s.results = study, results
    s._parts = list(parts) if parts is not None else []
    return s


def _r(x, sig: int = 5):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.{sig}g}") if x != 0 else 0.0


def kpis(study, results, parts=None) -> dict:
    """KPI of the results: per load state the max. von Mises (all / outside the singularities), max. displacement, min. safety
    against Re and the part, reactions of the selections; mass; natural frequencies; mesh.
    """
    from ..mcp import diagnostics as dg
    sess = _session(study, results, parts)
    m = results.mesh
    out: dict = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "mesh": {"nodes": int(len(m.nodes)), "elements": int(len(m.tets)),
                          "size": _r(study.mesh.size)}}
    cases = []
    for i, st in enumerate(results.steps):
        try:
            s = dg.step_summary(sess, i, with_hotspots=False)
        except Exception as e:  # noqa: BLE001 – the summary is a supplement
            cases.append({"name": st.name, "error": str(e)})
            continue
        row = {"name": st.name,
               "vm_max": _r((s.get("max_von_mises_MPa") or {}).get("value")),
               "u_max": _r((s.get("max_displacement_mm") or {}).get("value")),
               "k_min": _r(s.get("min_safety_Re"))}
        parts_rows = [p for p in s.get("parts") or [] if p.get("safety_Re") is not None]
        if parts_rows:
            worst = min(parts_rows, key=lambda p: p["safety_Re"])
            row["k_part"] = worst["part"]
        if len(m.nodes):
            try:
                hs = dg.hotspots(sess, i, 5)["hotspots"]
                clean = [h for h in hs if h["classification"] != "singularity"]
                if hs and hs[0]["classification"] == "singularity":
                    row["vm_max_clean"] = _r(clean[0]["mises_MPa"]) if clean else None
                    row["vm_singular"] = True
                else:
                    row["vm_max_clean"] = row["vm_max"]
            except Exception:  # noqa: BLE001
                pass
        row["reactions"] = {k: [_r(v) for v in (r.get("force_N") if r.get("force_N") is not None else [])]
                            for k, r in (s.get("reactions") or {}).items()}
        cases.append(row)
    if cases:
        out["cases"] = cases
    try:
        from ..loads import part_volumes
        vols = part_volumes(m)
        mass = 0.0
        for pi, pn in enumerate(m.part_names):
            try:
                mass += float(vols[pi]) * float(study.material_of(pn).rho or 0.0)
            except KeyError:
                pass
        if mass > 0:
            out["mass_kg"] = _r(mass * 1000.0)
    except Exception:  # noqa: BLE001
        pass
    modes = getattr(results, "modes", None) or []
    for ms in modes:
        if ms.kind == "modal":
            out["frequencies_Hz"] = [_r(v) for v in list(ms.values)[:6]]
        elif ms.kind == "buckling":
            out["buckling"] = [_r(v) for v in list(ms.values)[:3]]
    return out


def record(study, results, parts=None) -> bool:
    """Writes the KPI of the results into the current iteration of the linked study (Study.cad_link). Returns whether it was written."""
    link = getattr(study, "cad_link", None)
    if link is None or not link.iterations or results is None:
        return False
    link.iterations[-1].kpi = kpis(study, results, parts)
    return True


# ===================================================================== table of iterations

COLUMNS = (("vm_max_clean", "σ max (outside singularities) [MPa]"), ("vm_max", "σ max [MPa]"),
           ("u_max", "max displacement [mm]"), ("k_min", "safety factor to Re"))


def table(link, case: str | None = None) -> list[dict]:
    """Rows of the table of iterations: number, time, source, changes, parameters and KPI of the chosen state (None = the first) + the difference
    from the previous iteration with results [%].
    """
    rows = []
    prev = None
    for it in link.iterations:
        row = {"n": it.n, "time": it.time, "trigger": it.trigger, "geometry": it.geometry,
               "changes": list(it.changes), "params": {k: v.get("expression", v.get("value"))
                                                        for k, v in it.params.items()},
               "remap": dict(it.remap), "note": it.note}
        k = it.kpi or {}
        cs = k.get("cases") or []
        c = next((x for x in cs if x.get("name") == case), cs[0] if cs else None) if cs else None
        for key, _ in COLUMNS:
            row[key] = c.get(key) if c else None
        row["mass_kg"] = k.get("mass_kg")
        row["f1_Hz"] = (k.get("frequencies_Hz") or [None])[0]
        row["case"] = c.get("name") if c else None
        if prev is not None:
            delta = {}
            for key in [kk for kk, _ in COLUMNS] + ["mass_kg", "f1_Hz"]:
                a, b = prev.get(key), row.get(key)
                if a not in (None, 0) and b is not None:
                    delta[key] = 100.0 * (b / a - 1.0)
            row["delta_pct"] = delta
        if c is not None:
            prev = row
        rows.append(row)
    return rows


def cases_of(link) -> list[str]:
    names = []
    for it in link.iterations:
        for c in (it.kpi or {}).get("cases") or []:
            if c.get("name") and c["name"] not in names:
                names.append(c["name"])
    return names


def mesh_size_changed(link) -> bool:
    sizes = {(it.kpi or {}).get("mesh", {}).get("size") for it in link.iterations if it.kpi}
    return len(sizes - {None}) > 1


def finite(vals) -> list[float]:
    return [float(v) for v in vals if v is not None and np.isfinite(v)]
