"""Parametric study and Design of Experiments (DOE / C4b).

A systematic sweep of the inputs of a study over a grid (all combinations) or a list (the i-th values together):
  - load scale: all load cases together (`load_factor`) or one case (`case_factor:<case>`)
  - material of a part (`material:<part>`) or a property of a material (`prop:<material>:<field>`, E, nu, rho, Re, Rm, …)
  - global mesh size (`mesh_size`)
  - 3D print settings of a part (`print:<part>:<field>`, walls, infill, pattern, …)
CAD parameters are not swept: the core cannot regenerate the geometry from parameters (the CAD link only replaces a STEP).

KPI of every design point (DP) = `cad_link.history.kpis` (max. von Mises also outside singularities, displacement, safety against
Re, mass, natural frequencies) + the lowest 3D print safety factor. A linear study with only load factors is solved once and the
KPI are scaled exactly (superposition); everything else is solved per point on one mesh while the mesh signature does not change.
"""
from __future__ import annotations

import copy
import itertools
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .assessment import part_nodes as _part_nodes
from .solve import run_study
from .study import Study

KINDS = ("load_factor", "case_factor", "material", "material_prop", "mesh_size", "print")
MATERIAL_FIELDS = ("E", "nu", "rho", "Re", "Rm", "alpha", "k", "E_z", "Rm_z")
PRINT_FIELDS = {"walls": int, "top_layers": int, "bottom_layers": int, "infill": float, "layer_height": float,
                "line_width": float, "pattern": str}
MAX_POINTS = 200                                  # hard ceiling (each point = one analysis)


class SweepError(ValueError):
    """Invalid specification of the parametric study."""


@dataclass
class SweepParam:
    """One swept input. kind: load_factor | case_factor (target = case) | material (target = part, values = material
    names) | material_prop (target = material, field = property) | mesh_size | print (target = part, field = print setting)."""
    kind: str
    values: list = field(default_factory=list)
    target: str = ""
    field: str = ""

    @property
    def key(self) -> str:
        if self.kind == "case_factor":
            return f"case_factor:{self.target}"
        if self.kind == "material":
            return f"material:{self.target}"
        if self.kind == "material_prop":
            return f"prop:{self.target}:{self.field}"
        if self.kind == "print":
            return f"print:{self.target}:{self.field}"
        return self.kind

    def to_dict(self) -> dict:
        return {"kind": self.kind, "target": self.target, "field": self.field, "values": list(self.values)}

    @classmethod
    def from_dict(cls, d: dict) -> "SweepParam":
        return cls(kind=str(d.get("kind", "")), values=list(d.get("values") or []), target=str(d.get("target") or ""),
                   field=str(d.get("field") or ""))


@dataclass
class DesignPoint:
    index: int
    params: dict[str, Any]
    max_stress: float = 0.0          # max. von Mises over the load cases [MPa] (all nodes)
    max_disp: float = 0.0            # max. |U| [mm]
    mass: float = 0.0                # total mass [kg]
    min_safety: float = float("inf") # minimum safety factor against Re (Re / max. von Mises per part)
    converged: bool = True
    error: str = ""
    max_stress_clean: float | None = None    # max. von Mises outside singularities (diagnostics.hotspots) [MPa]
    freq1: float | None = None               # 1st natural frequency [Hz]
    print_safety: float | None = None        # lowest 3D print safety factor (Tsai-Wu / Hill)
    kpi: dict = field(default_factory=dict)  # full KPI per load case (cad_link.history.kpis)

    def to_dict(self) -> dict:
        def num(v):
            return None if v is None or not math.isfinite(float(v)) else float(v)
        return {
            "index": self.index,
            "params": self.params,
            "max_stress": self.max_stress,
            "max_stress_clean": num(self.max_stress_clean),
            "max_disp": self.max_disp,
            "mass": self.mass,
            "min_safety": num(self.min_safety),
            "freq1": num(self.freq1),
            "print_safety": num(self.print_safety),
            "converged": self.converged,
            "error": self.error,
        }


@dataclass
class ParametricStudySpec:
    """Specification of the sweep. Legacy fields (load_factors, materials, mesh_sizes) and `params` are combined:
    mode "grid" = all combinations, "list" = the i-th values of all parameters together (equal lengths)."""
    load_factors: list[float] = field(default_factory=lambda: [1.0])
    materials: dict[str, list[str]] = field(default_factory=dict)  # part_name -> [mat_name1, mat_name2]
    mesh_sizes: list[float] = field(default_factory=list)          # mesh sizes [mm]
    max_points: int = 50                                          # safety ceiling of the number of points
    warnings: list[str] = field(default_factory=list)
    params: list[SweepParam] = field(default_factory=list)
    mode: str = "grid"                                            # grid | list

    def all_params(self) -> list[SweepParam]:
        out = []
        if len(self.load_factors) > 1 or (len(self.load_factors) == 1 and self.load_factors[0] != 1.0):
            out.append(SweepParam("load_factor", list(self.load_factors)))
        for part, mats in self.materials.items():
            if mats:
                out.append(SweepParam("material", list(mats), target=part))
        if len(self.mesh_sizes) >= 1:
            out.append(SweepParam("mesh_size", list(self.mesh_sizes)))
        return out + [p for p in self.params if p.values]

    def total(self) -> int:
        ps = self.all_params()
        if not ps:
            return 1
        if self.mode == "list":
            return len(ps[0].values)
        return int(np.prod([len(p.values) for p in ps]))

    def combinations(self) -> list[dict[str, Any]]:
        ps = self.all_params()
        if not ps:
            return [{}]                      # no variability, only the starting point
        keys = [p.key for p in ps]
        if len(set(keys)) != len(keys):
            raise SweepError("A parameter is given twice: " + ", ".join(sorted({k for k in keys if keys.count(k) > 1})))
        limit = max(1, min(int(self.max_points), MAX_POINTS))
        if self.mode == "list":
            n = {len(p.values) for p in ps}
            if len(n) != 1:
                raise SweepError("List mode needs the same number of values for every parameter")
            rows = [dict(zip(keys, vals)) for vals in zip(*[p.values for p in ps])]
        elif self.mode == "grid":
            rows = []
            for prod in itertools.product(*[p.values for p in ps]):
                rows.append(dict(zip(keys, prod)))
                if len(rows) >= limit:
                    break
        else:
            raise SweepError(f"Unknown mode '{self.mode}' (grid | list)")
        total = self.total()
        if total > limit:
            self.warnings = [
                f"The number of parameter combinations ({total}) exceeded the limit max_points={limit}. The point list was truncated to the first {limit} points."
            ]
            rows = rows[:limit]
        else:
            self.warnings = []
        return rows

    def to_dict(self) -> dict:
        return {"mode": self.mode, "max_points": self.max_points, "params": [p.to_dict() for p in self.all_params()]}

    @classmethod
    def from_params(cls, params: list[SweepParam | dict], mode: str = "grid", max_points: int = 50) -> "ParametricStudySpec":
        ps = [p if isinstance(p, SweepParam) else SweepParam.from_dict(p) for p in params]
        return cls(load_factors=[1.0], params=ps, mode=mode, max_points=max_points)


@dataclass
class ParametricStudyResult:
    points: list[DesignPoint]
    pareto_indices: list[int] = field(default_factory=list)
    baseline_index: int = 0
    warnings: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)     # parameter keys (table columns)
    linear_shortcut: bool = False                     # one analysis, KPI scaled
    cancelled: bool = False

    def to_dict(self) -> dict:
        return {
            "points": [p.to_dict() for p in self.points],
            "pareto_indices": self.pareto_indices,
            "baseline_index": self.baseline_index,
            "total_points": len(self.points),
            "parameters": self.keys,
            "linear_shortcut": self.linear_shortcut,
            "cancelled": self.cancelled,
            "warnings": self.warnings,
        }

    def summary_table(self) -> str:
        lines = ["| DP | Parameters | Max σ [MPa] | Max u [mm] | Mass [kg] | Safety | Status |",
                 "|---|---|---|---|---|---|---|"]
        for p in self.points:
            p_str = ", ".join(f"{k}={v}" for k, v in p.params.items()) if p.params else "default"
            status = "OK" if p.converged else f"ERROR ({p.error})"
            star = " *" if p.index in self.pareto_indices else ""
            lines.append(f"| DP{p.index}{star} | {p_str} | {p.max_stress:.1f} | {p.max_disp:.3f} | {p.mass:.3f} | {p.min_safety:.2f} | {status} |")
        return "\n".join(lines)


def identify_pareto(points: list[DesignPoint]) -> list[int]:
    """Identifies the points of the Pareto front (minimisation of mass and stress while keeping the safety)."""
    valid = [p for p in points if p.converged]
    if not valid:
        return []

    def stress(p):
        return p.max_stress_clean if p.max_stress_clean is not None else p.max_stress
    pareto = []
    for p in valid:
        dominated = False
        for other in valid:
            if other.index == p.index:
                continue
            # other dominates p if it is at least as good in all objectives and sharply better in at least one
            if (other.mass <= p.mass and stress(other) <= stress(p) and
                    (other.mass < p.mass or stress(other) < stress(p))):
                dominated = True
                break
        if not dominated:
            pareto.append(p.index)
    return sorted(pareto)


# --------------------------------------------------------------------- values from text (GUI, MCP)
_RANGE = re.compile(r"^\s*([-+0-9.eE]+)\s*:\s*([-+0-9.eE]+)\s*:\s*(\d+)\s*$")


def parse_values(text: str, numeric: bool = True) -> list:
    """Values from text: "1, 1.5, 2" (also ";" or spaces; decimal comma accepted when ";" separates) or a range
    "start:stop:n" (n evenly spaced values including both ends). numeric=False = names (material, infill pattern)."""
    text = (text or "").strip()
    if not text:
        return []
    if not numeric:
        return [t.strip() for t in re.split(r"[;\n]|,(?!\d)", text) if t.strip()]
    m = _RANGE.match(text)
    if m:
        a, b, n = float(m.group(1)), float(m.group(2)), int(m.group(3))
        if not 1 <= n <= MAX_POINTS:
            raise SweepError(f"Range '{text}': the number of values must be 1–{MAX_POINTS}")
        return [float(f"{v:.12g}") for v in (np.linspace(a, b, n) if n > 1 else [a])]
    if ";" in text:
        items = [t.strip().replace(",", ".") for t in text.split(";")]
    else:
        items = [t for t in re.split(r"[,\s]+", text) if t]
    try:
        return [float(t) for t in items if t]
    except ValueError:
        raise SweepError(f"Cannot read the numbers '{text}' (use '1, 2, 3' or 'start:stop:n')") from None


# --------------------------------------------------------------------- application of the parameters
def _scale_loads(loads, factor: float) -> None:
    for l in loads:
        kind = getattr(l, "kind", "")
        if kind in ("force", "bearing", "gravity"):
            if hasattr(l, "vector") and l.vector:
                l.vector = [float(v) * factor for v in l.vector]
        elif kind == "pressure":
            if hasattr(l, "value"):
                l.value = float(l.value) * factor
            if hasattr(l, "gradient") and l.gradient:
                l.gradient = [float(g) * factor for g in l.gradient]
        elif kind == "remote":
            if hasattr(l, "force") and l.force:
                l.force = [float(v) * factor for v in l.force]
            if hasattr(l, "moment") and l.moment:
                l.moment = [float(v) * factor for v in l.moment]
        elif kind == "rotation":
            if factor < 0 and getattr(l, "omega", 0.0) != 0.0:
                raise ValueError(f"Negative load_factor={factor} cannot be used for rotation with a nonzero angular velocity omega")
            if hasattr(l, "omega"):
                l.omega = float(l.omega) * math.sqrt(factor) if factor >= 0 else float(l.omega)
            if hasattr(l, "alpha"):
                l.alpha = float(l.alpha) * factor
        elif kind == "inertia":
            if hasattr(l, "accel") and l.accel:
                l.accel = [float(a) * factor for a in l.accel]
            if hasattr(l, "alpha") and l.alpha:
                l.alpha = [float(a) * factor for a in l.alpha]
            if hasattr(l, "omega") and l.omega:
                if factor < 0 and any(w != 0.0 for w in l.omega):
                    raise ValueError(f"Negative load_factor={factor} cannot be used for inertia with a nonzero omega")
                l.omega = [float(w) * math.sqrt(factor) for w in l.omega] if factor >= 0 else l.omega
            if hasattr(l, "wmat") and l.wmat:
                l.wmat = [[float(v) * factor for v in row] for row in l.wmat]
        elif kind == "frame_node":
            if hasattr(l, "force") and l.force:
                l.force = [float(v) * factor for v in l.force]
            if hasattr(l, "moment") and l.moment:
                l.moment = [float(v) * factor for v in l.moment]
        elif kind == "frame_member":
            if hasattr(l, "q") and l.q:
                l.q = [float(v) * factor for v in l.q]
            if hasattr(l, "q_end") and l.q_end:
                l.q_end = [float(v) * factor for v in l.q_end]
        elif kind == "frame_temperature":
            l.dT, l.grad_z, l.grad_y = (float(v) * factor for v in (l.dT, l.grad_z, l.grad_y))
        elif kind == "thermal":
            pass                             # the temperature in °C is not a force load
        else:
            if hasattr(l, "vector") and l.vector:
                l.vector = [float(v) * factor for v in l.vector]
            elif hasattr(l, "value") and isinstance(l.value, (int, float)):
                l.value = float(l.value) * factor


def scale_study_loads(study: Study, factor: float, cases: list[str] | None = None) -> None:
    """Scales the loads of the study (or only of the named load cases) by `factor` according to the kind of load.

    Scaling rules:
      - Force, BearingLoad, Gravity, Pressure (value and gradient), RemoteLoad, frame loads: linearly.
      - Rotation: omega × sqrt(factor) (centrifugal force ~ omega²), alpha × factor; Inertia likewise (wmat × factor).
      - ThermalLoad: NOT SCALED (temperature in °C is not a force load).
    """
    for lc in study.load_cases:
        if cases is None or lc.name in cases:
            _scale_loads(lc.loads, factor)


def apply_params(study: Study, params: dict[str, Any]) -> Study:
    """Copy of the study with the parameters of one design point applied (used for the analysis and for taking over a variant)."""
    st = copy.deepcopy(study)
    for k, val in params.items():
        if k == "load_factor":
            scale_study_loads(st, float(val))
        elif k.startswith("case_factor:"):
            name = k.split(":", 1)[1]
            if name not in {lc.name for lc in st.load_cases}:
                raise SweepError(f"Load case '{name}' does not exist")
            scale_study_loads(st, float(val), [name])
        elif k.startswith("material:"):
            part = k.split(":", 1)[1]
            if str(val) not in st.materials:
                raise SweepError(f"Material '{val}' is not in the study")
            st.part_materials[part] = str(val)
        elif k.startswith("prop:"):
            _, mat, fld = k.split(":", 2)
            if mat not in st.materials:
                raise SweepError(f"Material '{mat}' is not in the study")
            if fld not in MATERIAL_FIELDS:
                raise SweepError(f"Material property '{fld}' cannot be swept ({', '.join(MATERIAL_FIELDS)})")
            st.materials[mat] = replace(st.materials[mat], **{fld: float(val)})
        elif k == "mesh_size":
            if not float(val) > 0:
                raise SweepError("The mesh size must be positive")
            st.mesh.size = float(val)
        elif k.startswith("print:"):
            _, part, fld = k.split(":", 2)
            spec = st.printing.parts.get(part)
            if spec is None:
                raise SweepError(f"Part '{part}' has no 3D print settings")
            if fld not in PRINT_FIELDS:
                raise SweepError(f"Print setting '{fld}' cannot be swept ({', '.join(PRINT_FIELDS)})")
            v = PRINT_FIELDS[fld](val) if PRINT_FIELDS[fld] is not int else int(round(float(val)))
            if fld == "infill" and not 0.0 <= float(v) <= 1.0:
                raise SweepError("Infill must be 0–1")
            setattr(spec, fld, v)
        else:
            raise SweepError(f"Unknown parameter '{k}'")
    return st


def check_spec(study: Study, spec: ParametricStudySpec) -> list[str]:
    """Errors of the specification against the study (empty = ready)."""
    errors = []
    cases = {lc.name for lc in study.load_cases}
    for p in spec.all_params():
        if p.kind not in KINDS:
            errors.append(f"Unknown parameter kind '{p.kind}' ({', '.join(KINDS)})")
            continue
        if not p.values:
            errors.append(f"Parameter {p.key}: no values")
        if p.kind == "case_factor" and p.target not in cases:
            errors.append(f"Load case '{p.target}' does not exist")
        elif p.kind == "material":
            for v in p.values:
                if str(v) not in study.materials:
                    errors.append(f"Material '{v}' is not in the study (add it in the Materials step first)")
        elif p.kind == "material_prop":
            if p.target not in study.materials:
                errors.append(f"Material '{p.target}' is not in the study")
            if p.field not in MATERIAL_FIELDS:
                errors.append(f"Material property '{p.field}' cannot be swept ({', '.join(MATERIAL_FIELDS)})")
            if p.field in ("E", "rho") and any(float(v) <= 0 for v in p.values):
                errors.append(f"{p.field} must be positive")
            if p.field == "nu" and any(not -1.0 < float(v) < 0.5 for v in p.values):
                errors.append("Poisson's ratio must be in (−1; 0.5)")
        elif p.kind == "mesh_size" and any(float(v) <= 0 for v in p.values):
            errors.append("The mesh size must be positive")
        elif p.kind == "print":
            if p.target not in study.printing.parts:
                errors.append(f"Part '{p.target}' has no 3D print settings")
            if p.field not in PRINT_FIELDS:
                errors.append(f"Print setting '{p.field}' cannot be swept ({', '.join(PRINT_FIELDS)})")
            elif p.field == "infill" and any(not 0 <= float(v) <= 1 for v in p.values):
                errors.append("Infill must be 0–1 (e.g. 0.2 = 20 %)")
    try:
        spec.combinations()
    except SweepError as e:
        errors.append(str(e))
    return errors


def _linear_scalable(study: Study) -> bool:
    """Can the KPI of load factors be scaled from one linear analysis exactly (superposition)? Not with temperatures,
    prescribed displacements, bolt preload, 3D printing (safety criterion is not linear), buckling and dynamics."""
    if study.is_nonlinear() or study.has_thermal_loads() or study.active_bolts():
        return False
    if study.buckling.enabled or study.dynamics.enabled or study.printing.parts:
        return False
    for c in study.constraints:
        if getattr(c, "kind", "") == "displacement" and any(v not in (None, 0, 0.0) for v in (c.ux, c.uy, c.uz)):
            return False
    return True


# --------------------------------------------------------------------- KPI
def _mass_kg(study: Study, mesh) -> float:
    from .loads import element_volumes
    vols = element_volumes(mesh)
    tot = 0.0
    for p_idx, p_name in enumerate(mesh.part_names):
        try:
            mat = study.material_of(p_name)
        except KeyError:
            mat = None
        rho = mat.rho if mat and mat.rho else 7.85e-9
        tot += float(np.sum(vols[mesh.tet_part == p_idx])) * rho
    return tot * 1000.0


def _evaluate(dp: DesignPoint, study: Study, res, parts=None) -> None:
    from .results import von_mises
    s_vm_max = u_max = 0.0
    min_s = float("inf")
    print_min = None
    for step in res.steps:
        vm = von_mises(step.S)
        s_vm_max = max(s_vm_max, float(np.max(vm)) if len(vm) else 0.0)
        if len(step.U):
            u_max = max(u_max, float(np.max(np.linalg.norm(step.U, axis=1))))
        for p_idx, p_name in enumerate(res.mesh.part_names):
            try:
                mat = study.material_of(p_name)
            except KeyError:
                mat = None
            if mat and mat.Re and mat.Re > 0:
                nodes = _part_nodes(res.mesh, p_idx)
                if len(nodes) > 0:
                    vmax = float(np.max(vm[nodes]))
                    if vmax > 1e-3:
                        min_s = min(min_s, mat.Re / vmax)
        ps = (getattr(step, "fields", None) or {}).get("print_safety")
        if ps is not None and len(ps):
            v = float(np.nanmin(ps))
            print_min = v if print_min is None else min(print_min, v)
    dp.max_stress = s_vm_max
    dp.max_disp = u_max
    dp.mass = _mass_kg(study, res.mesh)
    dp.min_safety = min_s
    dp.print_safety = print_min
    try:
        from .cad_link.history import kpis
        k = kpis(study, res, parts)
    except Exception:  # noqa: BLE001 – the extra KPI must not discard the point
        k = {}
    dp.kpi = k
    clean = [c.get("vm_max_clean") for c in k.get("cases") or [] if c.get("vm_max_clean") is not None]
    dp.max_stress_clean = max(clean) if clean else None
    freqs = k.get("frequencies_Hz") or []
    dp.freq1 = freqs[0] if freqs else None


def _scaled_point(dp: DesignPoint, base: DesignPoint, case_names: list[str], params: dict) -> None:
    """KPI of a linear study with load factors from the baseline: per case × |factor| (stress, displacement), Re / … ."""
    g = float(params.get("load_factor", 1.0))
    f = [abs(g * float(params.get(f"case_factor:{n}", 1.0))) for n in case_names]
    cases = (base.kpi or {}).get("cases") or []
    steps = base.kpi.get("_steps") or []          # per-case maxima of the baseline (filled by run_parametric_study)
    dp.mass = base.mass
    dp.freq1 = base.freq1
    if steps and len(steps) == len(f):
        dp.max_stress = max(s["vm"] * fi for s, fi in zip(steps, f))
        dp.max_disp = max(s["u"] * fi for s, fi in zip(steps, f))
        ks = [s["k"] / fi for s, fi in zip(steps, f) if s["k"] is not None and fi > 0]
        dp.min_safety = min(ks) if ks else float("inf")
    clean = [c.get("vm_max_clean") * fi for c, fi in zip(cases, f) if c.get("vm_max_clean") is not None]
    dp.max_stress_clean = max(clean) if clean else None
    dp.kpi = {"cases": [{k: (v * fi if k in ("vm_max", "vm_max_clean", "u_max") and v is not None else v)
                         for k, v in c.items() if k != "reactions"} for c, fi in zip(cases, f)]}


def _step_maxima(study: Study, res) -> list[dict]:
    from .results import von_mises
    out = []
    for step in res.steps:
        vm = von_mises(step.S)
        k = None
        for p_idx, p_name in enumerate(res.mesh.part_names):
            try:
                mat = study.material_of(p_name)
            except KeyError:
                mat = None
            if mat and mat.Re and mat.Re > 0:
                nodes = _part_nodes(res.mesh, p_idx)
                if len(nodes):
                    vmax = float(np.max(vm[nodes]))
                    if vmax > 1e-3:
                        k = mat.Re / vmax if k is None else min(k, mat.Re / vmax)
        out.append({"vm": float(np.max(vm)) if len(vm) else 0.0,
                    "u": float(np.max(np.linalg.norm(step.U, axis=1))) if len(step.U) else 0.0, "k": k})
    return out


def run_parametric_study(study: Study, spec: ParametricStudySpec, base_workdir: Path,
                         on_progress: Callable[[int, int], None] | None = None, *, parts=None, cancel=None,
                         on_log: Callable[[str], None] | None = None,
                         on_point: Callable[[DesignPoint], None] | None = None,
                         study_dir: str | Path | None = None) -> ParametricStudyResult:
    """Runs the parametric sweep of the study (a copy per point – the study is not changed).

    parts = geometry already loaded (not read again per point); cancel = threading.Event (stops between points and in the
    solver); on_point(dp) after every finished point (GUI table). Work folders of the points are kept in base_workdir.
    """
    errors = check_spec(study, spec)
    if errors:
        raise SweepError("; ".join(errors))
    combos = spec.combinations()
    total = len(combos)
    keys = list(combos[0].keys()) if combos else []
    points: list[DesignPoint] = []
    base_workdir = Path(base_workdir)
    base_workdir.mkdir(parents=True, exist_ok=True)
    warnings = list(getattr(spec, "warnings", []))
    log = on_log or (lambda s: None)
    kw = {"study_dir": study_dir} if study_dir is not None else {}

    def solve(st, wd, mesh=None):
        return run_study(st, wd, cancel=cancel, mesh=mesh, parts=parts if mesh is None else None, on_log=on_log, **kw)

    def add_warnings(ws):
        warnings.extend(w for w in ws if w not in warnings)

    load_only = all(all(k == "load_factor" or k.startswith("case_factor:") for k in p) for p in combos)
    if load_only and _linear_scalable(study):
        log(f"Parametric study: linear study with load factors only – one analysis, {total} points scaled")
        res_base = solve(copy.deepcopy(study), base_workdir / "baseline")
        add_warnings(res_base.warnings)
        base = DesignPoint(index=-1, params={})
        _evaluate(base, study, res_base, parts)
        base.kpi["_steps"] = _step_maxima(study, res_base)
        names = [s.name for s in res_base.steps]
        for idx, params in enumerate(combos):
            if on_progress:
                on_progress(idx, total)
            dp = DesignPoint(index=idx, params=params)
            try:
                _scaled_point(dp, base, names, params)
            except Exception as e:  # noqa: BLE001
                dp.converged, dp.error = False, str(e)
            points.append(dp)
            if on_point:
                on_point(dp)
        return ParametricStudyResult(points=points, pareto_indices=identify_pareto(points), baseline_index=0,
                                     warnings=warnings, keys=keys, linear_shortcut=True)

    from .signature import mesh_signature
    cached_mesh = None
    cached_sig = None
    stopped = False
    for idx, params in enumerate(combos):
        if cancel is not None and cancel.is_set():
            stopped = True
            break
        if on_progress:
            on_progress(idx, total)
        dp = DesignPoint(index=idx, params=params)
        log(f"Parametric study: point {idx + 1}/{total} " + ", ".join(f"{k} = {v}" for k, v in params.items()))
        try:
            st = apply_params(study, params)
            sig = mesh_signature(st)
            mesh_arg = cached_mesh if (cached_mesh is not None and cached_sig == sig) else None
            res = solve(st, base_workdir / f"DP_{idx:03d}", mesh=mesh_arg)
            cached_mesh, cached_sig = res.mesh, sig
            add_warnings(res.warnings)
            _evaluate(dp, st, res, parts)
        except Exception as e:  # noqa: BLE001 – one failed point does not stop the others
            if cancel is not None and cancel.is_set():
                stopped = True
                break
            dp.converged = False
            dp.error = str(e)
        points.append(dp)
        if on_point:
            on_point(dp)
    if stopped:
        warnings.append(f"Parametric study cancelled after {len(points)} of {total} points")
    return ParametricStudyResult(points=points, pareto_indices=identify_pareto(points), baseline_index=0,
                                 warnings=warnings, keys=keys, cancelled=stopped)


def kpi_columns() -> list[tuple[str, str]]:
    """(attribute, label) of the KPI columns of the table / report / chart."""
    return [("max_stress_clean", "max. σ outside singularities [MPa]"), ("max_stress", "max. von Mises [MPa]"),
            ("max_disp", "max. displacement [mm]"), ("min_safety", "min. safety to Re [–]"), ("mass", "mass [kg]"),
            ("freq1", "f1 [Hz]"), ("print_safety", "3D print safety [–]")]
