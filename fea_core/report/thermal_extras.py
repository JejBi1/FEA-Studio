"""Report – heat: contact conductances of interfaces (D1) and cavity radiation (D2): setup tables, heat through the
interfaces and through the cavity faces (ThermalResult.balance["interfaces"] / ["cavity_faces"], fea_core.thermal.run).
"""
from __future__ import annotations

from .word import num


def has_conductance(study) -> bool:
    return bool(study.thermal.conductances) or any(c.conductance for c in study.active_contacts())


def contact_sentence(study) -> str:
    """How the parts touch thermally (introduction of the heat conduction and the contact assumptions)."""
    if has_conductance(study):
        return ("parts with a shared mesh and separately meshed parts tied by *TIE are in perfect thermal contact, interfaces with a "
                "contact conductance h_c have a temperature jump ΔT = q / h_c (CalculiX *GAP CONDUCTANCE), the other contact pairs are in perfect thermal contact. ")
    if study.active_contacts():
        return "parts with a shared mesh and contact pairs are in perfect thermal contact (no contact conductance specified). "
    return "parts in perfect thermal contact. "


def input_tables(D, study) -> None:
    """Setup: interface conductances (all thermal cases) – after the boundary conditions of the thermal cases."""
    rows = [[c.name, c.selection, c.master, num(c.conductance), num(c.resistance, 3), "Heat step"]
            for c in study.thermal.conductances]
    rows += [[c.name, c.a, c.b, num(c.conductance), num(1.0 / c.conductance, 3), "contact pair"]
             for c in study.active_contacts() if c.conductance]
    if not rows:
        return
    D.table(["Interface", "Side A", "Side B", "h_c [W/(m²·K)]", "R_c [m²·K/W]", "defined in"], rows,
            caption="Thermal contact conductances (apply to all thermal cases)", widths=(0.2, 0.18, 0.18, 0.15, 0.14, 0.15), font=8)
    sep = sorted({p for c in study.thermal.conductances for p in (c.selection, c.master) if p in study.mesh.separate})
    D.p("Temperature jump at the interface ΔT = q / h_c. h_c depends on pressure, roughness and filling of the joint – the values are estimates unless measured."
        + (f" Parts meshed separately for the interface: {', '.join(sep)} (mechanically tied by *TIE)." if sep else ""), size=9)


def result_tables(D, tr) -> None:
    """Heat through the interfaces and through the cavity faces of a thermal case."""
    bal = tr.balance or {}
    ifc = bal.get("interfaces") or []
    if ifc:
        D.table(["Interface", "from", "to", "h_c [W/(m²·K)]", "area [mm²]", "mean ΔT [K]", "heat Q [W]"],
                [[r["name"], r["from"], r["to"], num(r["h_W_m2K"]), num(r["area_mm2"]), num(r["dT_mean_K"]), num(r["Q_W"])]
                 for r in ifc],
                caption=f"Heat through contact conductance interfaces – thermal case “{tr.name}” (positive = from → to"
                        + (", at the end of the state" if tr.transient else "") + ")",
                widths=(0.18, 0.15, 0.15, 0.13, 0.13, 0.12, 0.14), font=8)
    cav = bal.get("cavity_faces") or []
    if cav:
        D.table(["Cavity", "Face selection", "ε [–]", "heat Q [W]"],
                [[r["cavity"], r["selection"], num(r["emissivity"], 3), num(r["Q_W"])] for r in cav],
                caption=f"Cavity radiation per face – thermal case “{tr.name}” (positive = absorbed by the body, negative = emitted)",
                widths=(0.25, 0.35, 0.15, 0.25), font=8)
