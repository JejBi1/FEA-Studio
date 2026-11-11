"""Calibration of printed material from tensile tests (ISO 527) – F4.

Test specimens (FDM, PLA) have a shell and infill like the parts, therefore from the measured
E and Rm the FULL extrusion is computed back by the same model as the calculation uses (iso-strain in the section):
    E_meas·A = E_full·(A_s + A_j·k_E),   Rm_meas·A = R_full·(A_s + A_j·k_R)
k_E, k_R = infill ratios (infill.ratios) for the direction of the load. Section of the measured part b × h:
  - "xy" (the specimen lies on the bed, tension in the plane of the layer): perimeters along the sides of the width b (in the plane),
    upper/lower layers across the thickness h,
  - "z" (the specimen stands, tension perpendicular to the layers): perimeters all around the perimeter of the section b × h.
Tangent modulus according to ISO 527-1: secant between ε = 0.05 % and 0.25 %.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field, asdict, fields, replace
from pathlib import Path

import numpy as np

from .infill import ratios
from .model import PrintSpec


@dataclass
class TensileTest:
    name: str
    direction: str = "xy"           # xy | z
    E: float = 0.0                  # measured modulus [MPa] (nominal, for the whole section b × h)
    Rm: float = 0.0                 # measured strength [MPa]
    width: float = 10.0             # b [mm] (ISO 527-2 1A/1B: 10 mm)
    thickness: float = 4.0          # h [mm]
    walls: int = 2
    line_width: float = 0.42
    top_layers: int = 5
    bottom_layers: int = 3
    layer_height: float = 0.2
    infill: float = 1.0
    pattern: str = "grid"
    enabled: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TensileTest":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def from_spec(cls, name: str, direction: str, spec: PrintSpec, **kw) -> "TensileTest":
        return cls(name, direction, walls=spec.walls, line_width=spec.line_width, top_layers=spec.top_layers,
                   bottom_layers=spec.bottom_layers, layer_height=spec.layer_height, infill=spec.infill,
                   pattern=spec.pattern, **kw)


def section3(t: TensileTest) -> tuple[float, float, float]:
    """(A_perimeters, A_full layers, A_core) of the measured section [mm²]. For the specimen "xy" the perimeters along the sides of the width
    through the whole thickness (along the tension), the full layers between them; for "z" the perimeters go around the whole perimeter, the full layers
    do not form the section (they are only at the ends of the specimen).
    """
    A = t.width * t.thickness
    w = t.walls * t.line_width
    if t.direction == "z":
        core = max(t.width - 2 * w, 0.0) * max(t.thickness - 2 * w, 0.0)
        return A - core, 0.0, core
    inner = max(t.width - 2 * w, 0.0)
    core = inner * max(t.thickness - (t.top_layers + t.bottom_layers) * t.layer_height, 0.0)
    per = A - inner * t.thickness
    return per, A - per - core, core


def section(t: TensileTest) -> tuple[float, float]:
    """(A_shell, A_core) of the measured section [mm²]."""
    per, lay, core = section3(t)
    return per + lay, core


def factors(t: TensileTest, line_E: float = 1.0, line_R: float = 1.0) -> tuple[float, float]:
    """(φ_E, φ_R): measured value = full (XY) · φ (share of the load-carrying section according to the infill model).

    line_E, line_R = ratios across/along the line (printing.props): the perimeters of the specimen "xy" are stressed along the line,
    where the calculation takes E_L = 2/(1 + r)·E_xy (line_solid) – the calibration thus gives E_xy consistent with the calculation.
    """
    per, lay, core = section3(t)
    A = per + lay + core
    r = ratios(t.pattern, t.infill)
    if t.direction == "z":
        return (per + core * r["E_z"]) / A, (per + core * r["X_z"]) / A
    cE, cR = 2.0 / (1.0 + line_E), 2.0 / (1.0 + line_R)
    return (per * cE + lay + core * r["E_xy"]) / A, (per * cR + lay + core * r["X_xy"]) / A


def iso527_modulus(strain: np.ndarray, stress: np.ndarray, e1: float = 0.0005, e2: float = 0.0025) -> float:
    """Modulus of elasticity as the secant between ε1 and ε2 (ISO 527-1, strain, not %)."""
    strain, stress = np.asarray(strain, float), np.asarray(stress, float)
    k = int(np.argmax(stress))                       # only the ascending part of the curve
    s, e = stress[: k + 1], strain[: k + 1]
    if e.max() < e2:
        raise ValueError(f"The curve does not reach a strain of {100 * e2:g} %")
    if e.min() > e1:
        raise ValueError(f"The curve starts only at a strain of {100 * e.min():.3g} % (> {100 * e1:g} %) – the modulus per ISO 527-1 cannot be determined; check extensometer zeroing")
    o = np.argsort(e, kind="stable")
    s1, s2 = np.interp([e1, e2], e[o], s[o])
    return float((s2 - s1) / (e2 - e1))


def read_curve(path: str | Path, area: float, gauge_length: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """CSV/TXT with two columns: force [N] and elongation [mm] (gauge_length = L0 of the extensometer),
    or force [N] and strain [%] (gauge_length = None). Separator , ; tab; decimal comma allowed.
    Returns (strain [-], stress [MPa]).
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:2000], delimiters=",;\t ") if text.strip() else csv.excel
    except csv.Error:                                # the sniffer did not recognise the separator → by the characters in the file
        class dialect(csv.excel):                    # noqa: N801
            delimiter = ";" if ";" in text else "\t" if "\t" in text else "," if "." in text else " "
            skipinitialspace = True
    rows = []
    for row in csv.reader(text.splitlines(), dialect):
        vals = [c.strip().replace(",", ".") for c in row if c.strip()]
        try:
            if len(vals) >= 2:
                rows.append((float(vals[0]), float(vals[1])))
        except ValueError:
            continue                                 # heading
    if len(rows) < 5:
        raise ValueError("The file does not contain two numeric columns (force, elongation)")
    F, x = np.array(rows).T
    strain = x / gauge_length if gauge_length else x / 100.0
    return strain, F / area


@dataclass
class Calibration:
    material: object                  # Material (calibrated)
    rows: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def calibrate(base, tests: list[TensileTest], name: str | None = None, props=None) -> Calibration:
    """Calibrated printed material from the measured specimens (average over the specimens of each direction).

    Without tests in the direction Z the ratio Z/XY stays according to the default material. ν, the compressive and shear strengths
    are recomputed by the same ratio as the corresponding tensile values (a tensile test does not determine them).
    props = printing.props.PrintProps (anisotropy of the line of the perimeters); None = default according to the material family.
    """
    if not base.is_print:
        raise ValueError(f"Default material '{base.name}' is not a printed material (E_z missing)")
    if props is None:
        from .props import default_props
        props = default_props(base.name)
    rE, rR = props.line_E_ratio, props.line_R_ratio
    rows, warns = [], []
    by_dir: dict[str, dict[str, list[float]]] = {"xy": {"E": [], "R": []}, "z": {"E": [], "R": []}}
    for t in tests:
        if not t.enabled:
            continue
        fe, fr = factors(t, rE, rR)
        d = "z" if t.direction == "z" else "xy"
        row = {"name": t.name, "direction": d, "E_meas": t.E, "Rm_meas": t.Rm, "phi_E": fe, "phi_R": fr,
               "E_solid": t.E / fe if t.E > 0 else None, "R_solid": t.Rm / fr if t.Rm > 0 else None}
        rows.append(row)
        if row["E_solid"]:
            by_dir[d]["E"].append(row["E_solid"])
        if row["R_solid"]:
            by_dir[d]["R"].append(row["R_solid"])
        if fe < 0.6:
            warns.append(f"{t.name}: the infill carries a large part of the cross-section (φ_E = {fe:.2f}) – calibration of the solid material depends on the estimated infill properties; for calibration print specimens with 100 % infill.")

    def mean(v):
        return float(np.mean(v)) if v else None

    def cv(v):
        return float(np.std(v, ddof=1) / np.mean(v)) if len(v) > 1 else None
    E_xy, E_z = mean(by_dir["xy"]["E"]), mean(by_dir["z"]["E"])
    R_xy, R_z = mean(by_dir["xy"]["R"]), mean(by_dir["z"]["R"])
    if E_xy is None and E_z is None and R_xy is None and R_z is None:
        raise ValueError("No valid test (enter E or Rm)")
    zr_E = base.E_z / base.E
    zr_R = (base.Rm_z or base.Rm) / base.Rm if base.Rm else 1.0
    if E_xy is None:
        E_xy = (E_z / zr_E) if E_z else base.E
    if E_z is None:
        E_z = E_xy * zr_E
        warns.append("No tests in the Z direction – E_z taken by ratio from the default material.")
    if R_xy is None:
        R_xy = (R_z / zr_R) if R_z else base.Rm
    if R_z is None:
        R_z = R_xy * zr_R
        warns.append("No tests in the Z direction – interlayer strength taken by ratio from the default material.")
    sx = R_xy / base.Rm if base.Rm else 1.0
    sz = R_z / (base.Rm_z or base.Rm) if base.Rm else 1.0

    def sc(v, k):
        return None if v is None else v * k
    mat = replace(base, name=name or f"{base.name} – calibration", E=E_xy, E_z=E_z, Rm=R_xy, Rm_z=R_z,
                  G_z=sc(base.G_z, E_z / base.E_z), Rc=sc(base.Rc, sx), Rc_z=sc(base.Rc_z, sz),
                  S_xy=sc(base.S_xy, sx), S_z=sc(base.S_z, sz),
                  source=f"ISO 527 calibration ({len(rows)} specimens) from '{base.name}'")
    summary = {"E_xy": E_xy, "E_z": E_z, "Rm_xy": R_xy, "Rm_z": R_z,
               "cv_E_xy": cv(by_dir["xy"]["E"]), "cv_E_z": cv(by_dir["z"]["E"]),
               "cv_R_xy": cv(by_dir["xy"]["R"]), "cv_R_z": cv(by_dir["z"]["R"]),
               "n_xy": len(by_dir["xy"]["E"] or by_dir["xy"]["R"]), "n_z": len(by_dir["z"]["E"] or by_dir["z"]["R"]),
               "z_ratio_R": R_z / R_xy if R_xy else math.nan}
    return Calibration(mat, rows, summary, warns)
