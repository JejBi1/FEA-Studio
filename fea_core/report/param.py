"""Report section "Parametric study" (fea_core.sweep): table of the design points and their KPI, Pareto front."""
from __future__ import annotations

from .word import num

PARAM_LABELS = {"load_factor": "load factor", "mesh_size": "mesh size [mm]"}


def has_param(sess) -> bool:
    res = getattr(sess, "param_study", None)
    return res is not None and bool(getattr(res, "points", None))


def param_label(key: str) -> str:
    """Readable column name of a parameter key (sweep.SweepParam.key)."""
    if key in PARAM_LABELS:
        return PARAM_LABELS[key]
    kind, _, rest = key.partition(":")
    if kind == "case_factor":
        return f"factor of {rest}"
    if kind == "material":
        return f"material of {rest}"
    if kind in ("prop", "print"):
        target, _, fld = rest.rpartition(":")
        return f"{target}: {fld}"
    return key


def _value(v) -> str:
    return num(v, 4) if isinstance(v, (int, float)) else str(v)


def param_section(D, sess) -> None:
    from ..sweep import kpi_columns
    res = sess.param_study
    cols = [(a, t) for a, t in kpi_columns()
            if any(getattr(p, a, None) not in (None, float("inf")) for p in res.points if p.converged)]
    D.p(f"The study was solved for {len(res.points)} design points (DP) on copies of the study; the parameters and the key results "
        "(KPI) are in the table. Maximum stress outside singularities = highest von Mises stress after leaving out the places "
        "classified as singularities (sharp corners, support edges, point loads).")
    if res.linear_shortcut:
        D.p("The study is linear and only the load scale changes: one analysis was scaled exactly (superposition).",
            italic=True)
    keys = list(res.keys)
    header = ["DP"] + [param_label(k) for k in keys] + [t for _, t in cols] + ["Pareto"]
    rows = []
    for p in res.points:
        if not p.converged:
            rows.append([str(p.index)] + [_value(p.params.get(k)) for k in keys] + ["error: " + p.error[:80]]
                        + [""] * (len(cols) - 1) + [""])
            continue
        rows.append([str(p.index)] + [_value(p.params.get(k)) for k in keys] + [num(getattr(p, a), 4) for a, _ in cols]
                    + ["*" if p.index in res.pareto_indices else ""])
    D.table(header, rows, caption="Parametric study – design points and KPI", font=8)
    if res.pareto_indices:
        D.p("Pareto front (no other point is both lighter and less stressed): "
            + ", ".join(f"DP {i}" for i in res.pareto_indices) + ".")
    for w in res.warnings[:5]:
        D.p(w, italic=True, size=9)
