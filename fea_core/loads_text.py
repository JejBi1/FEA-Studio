"""Description of loads in Czech (MCP, GUI, report): force on faces / edges / points, local system, variable pressure,
rotation. The numbers are formatted by `num` (the report has its own – decimal comma).
"""
from __future__ import annotations

import math

import numpy as np


def _num(v: float) -> str:
    return f"{float(v):.4g}"


def vec_text(v, num=_num) -> str:
    return "(" + "; ".join(num(float(x)) for x in v) + ")"


def entity_of(l, st) -> str:
    """Kind of load selection: faces | edges | points (without a study a face)."""
    sel = st.selections.get(getattr(l, "selection", None)) if st is not None else None
    return sel.entity if sel is not None else "faces"


def csys_suffix(l, st) -> str:
    name = getattr(l, "csys", None)
    if not name:
        return ""
    cs = st.coordinate_systems.get(name) if st is not None else None
    if cs is not None and cs.kind == "cylindrical":
        return f" in system “{name}” (r, θ, z)"
    return f" in system “{name}”"


def force_kind(l, st) -> str:
    return {"faces": "force (uniform over face)", "edges": "force on edge (uniform along length)",
            "points": "force at point"}.get(entity_of(l, st), "force")


def pressure_text(l, num=_num) -> str:
    if not getattr(l, "variable", False):
        return f"p = {num(l.value)} MPa"
    g = np.asarray(l.gradient, float)
    o = l.origin if l.origin is not None else [0.0, 0.0, 0.0]
    if l.clip:
        gn = float(np.linalg.norm(g))
        rho_kg = gn / 9810.0 * 1e12                   # ρ·g → ρ [kg/m³] at g = 9810 mm/s²
        base = f"hydrostatic: p = {num(l.value)} MPa + {num(gn)} MPa/mm × depth below the surface"
        return (base + f" (level at point {vec_text(o, num)} mm, weight in direction {vec_text(g / gn, num)}; ≈ liquid {num(rho_kg)} kg/m³ at g = 9.81 m/s²)")
    return f"linear: p = {num(l.value)} MPa + {vec_text(g, num)} MPa/mm · (x − {vec_text(o, num)} mm)"


def rotation_text(l, num=_num) -> str:
    parts = []
    if l.omega:
        parts.append(f"ω = {num(l.omega)} rad/s ({num(l.omega * 60 / (2 * math.pi))} rpm)")
    if l.alpha:
        parts.append(f"α = {num(l.alpha)} rad/s²")
    who = "parts " + ", ".join(l.parts) if l.parts else "all parts"
    return (", ".join(parts) or "no rotation") + f"; axis at point {vec_text(l.point, num)} mm, direction {vec_text(l.axis, num)}; {who}"


def inertia_text(l, num=_num) -> str:
    """Inertia forces of the rigid motion (study.Inertia)."""
    parts = []
    for lab, v, u in (("a₀", l.accel, "mm/s²"), ("ω", l.omega, "rad/s"), ("α", l.alpha, "rad/s²")):
        if any(float(x) for x in v):
            parts.append(f"{lab} = {vec_text(v, num)} {u}")
    who = "parts " + ", ".join(l.parts) if l.parts else "all parts"
    return (", ".join(parts) or "no motion") + f"; point {vec_text(l.point, num)} mm; {who}"


def load_text(l, st=None, num=_num) -> str:
    """Value of a load with units (one sentence)."""
    k = getattr(l, "kind", "")
    if k == "force":
        n = len(st.selections[l.selection].points) if entity_of(l, st) == "points" else 0
        each = f", {n} points of F/{n} each" if n > 1 else ""
        return f"{force_kind(l, st)}: F = {vec_text(l.vector, num)} N{csys_suffix(l, st)}{each}"
    if k == "pressure":
        return pressure_text(l, num)
    if k == "rotation":
        return "rotation: " + rotation_text(l, num)
    if k == "inertia":
        return "inertia: " + inertia_text(l, num)
    if k == "remote":
        cpl = "distributing (RBE3)" if l.coupling == "distributing" else "rigid"
        return (f"remote load ({cpl}), point {vec_text(l.point, num)} mm: F = {vec_text(l.force, num)} N, M = {vec_text(l.moment, num)} N·mm{csys_suffix(l, st)}")
    if k == "bearing":
        return f"bearing load F = {vec_text(l.vector, num)} N{csys_suffix(l, st)}"
    if k == "gravity":
        return f"gravity g = {vec_text(l.vector, num)} mm/s²{csys_suffix(l, st)}"
    return k


def constraint_text(c, st=None, num=_num) -> str:
    """Description of a constraint with units (one sentence) – MCP, GUI, report."""
    k = getattr(c, "kind", "")
    if k == "fixed":
        return "fixed support (ux = uy = uz = 0)"
    if k == "displacement":
        parts = [f"{n} = {num(v)} mm" for n, v in zip(("ux", "uy", "uz"), (c.ux, c.uy, c.uz)) if v is not None]
        return "prescribed displacement: " + (", ".join(parts) or "no component")
    if k == "cylindrical":
        on = [n for n, v in (("radially", c.radial), ("tangentially", c.tangential), ("axially", c.axial)) if v]
        return "cylindrical support: " + (", ".join(on) or "no component")
    if k == "remote_support":
        names = ("ux", "uy", "uz", "rx", "ry", "rz")
        on = [n for n, v in zip(names, c.dofs) if v]
        cpl = "rigid" if c.coupling == "rigid" else "distributing"
        return f"remote support ({cpl}) at point {vec_text(c.point, num)} mm: " + (", ".join(on) or "nothing")
    if k == "elastic":
        unit = "N/mm³" if c.per_area else "N/mm (total)"
        d = "normal to face" if c.direction == "normal" else "in all directions"
        return f"elastic support k = {num(c.stiffness)} {unit}, {d}"
    return k
