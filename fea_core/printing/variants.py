"""Variants of the internal structure of the printing (F5): the same mesh, different perimeters / layers / infill / build direction / material.

The mesh does not depend on the print settings (printing.region.ensure recomputes only the assignment of the shell and the infill, ~s) → all
variants are computed on ONE Gmsh mesh and differ only in the materials of the elements. Each variant = a separate run_study
(also the creep according to the study). Output: min. safety of the printing (Tsai-Wu/Hill) and where, max. displacement, mass of the printed parts;
recommendation = the lightest variant with k ≥ PrintSettings.required_safety.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .model import PrintSpec, PrintVariant

BASE_NAME = "Current settings"
# overrides: fields of PrintSpec (without part, note) + another printed material of the part
SPEC_KEYS = ("build_dir", "layer_height", "line_width", "walls", "top_layers", "bottom_layers", "infill", "pattern",
             "enabled", "solid_thin", "thin_width")
OVERRIDE_KEYS = SPEC_KEYS + ("material",)


class VariantError(ValueError):
    pass


def normalize(ov: dict) -> dict:
    """Overrides of a part: infill in % → 0–1, numbers of layers whole, build direction as a list of 3 numbers."""
    out = {}
    for k, v in (ov or {}).items():
        if k in ("walls", "top_layers", "bottom_layers"):
            v = int(round(float(v)))
        elif k in ("layer_height", "line_width", "thin_width"):
            v = float(v)
        elif k == "infill":
            v = float(v)
            v = v / 100.0 if v > 1.0 else v
        elif k == "build_dir":
            v = [float(x) for x in v]
        elif k in ("enabled", "solid_thin"):
            v = bool(v)
        elif k in ("pattern", "material"):
            v = str(v)
        out[k] = v
    return out


def validate(study, v: PrintVariant, part_names: list[str] | None = None) -> list[str]:
    """Errors of the definition of the variant (empty list = all right)."""
    from .infill import PATTERNS
    errs = []
    if not v.name.strip():
        errs.append("A variant must have a name")
    if v.name == BASE_NAME:
        errs.append(f"The name '{BASE_NAME}' is reserved for the study settings")
    for part, ov in v.parts.items():
        if part_names is not None and part not in part_names:
            errs.append(f"Variant '{v.name}': part '{part}' does not exist")
        try:
            ov = normalize({k: val for k, val in ov.items() if k in OVERRIDE_KEYS}) | {
                k: val for k, val in ov.items() if k not in OVERRIDE_KEYS}
        except (TypeError, ValueError):
            errs.append(f"Variant '{v.name}', part '{part}': invalid setting value ({ov})")
            continue
        for k, val in ov.items():
            if k not in OVERRIDE_KEYS:
                errs.append(f"Variant '{v.name}', part '{part}': unknown setting '{k}' (allowed: {', '.join(OVERRIDE_KEYS)})")
            elif k == "pattern" and val not in PATTERNS:
                errs.append(f"Variant '{v.name}', part '{part}': unknown infill pattern '{val}' ({' | '.join(PATTERNS)})")
            elif k == "material":
                m = study.materials.get(val)
                if m is None:
                    errs.append(f"Variant '{v.name}', part '{part}': material '{val}' is not in the study")
                elif not m.is_print:
                    errs.append(f"Variant '{v.name}', part '{part}': material '{val}' is not a printed material (E_z missing)")
            elif k == "infill" and not (0.0 <= float(val) <= 1.0):
                errs.append(f"Variant '{v.name}', part '{part}': infill must be 0–100 %")
            elif k == "build_dir" and (len(val) != 3 or sum(float(x) ** 2 for x in val) == 0):
                errs.append(f"Variant '{v.name}', part '{part}': build direction must be a nonzero vector [x, y, z]")
            elif k in ("layer_height", "line_width") and float(val) <= 0:
                errs.append(f"Variant '{v.name}', part '{part}': {k} must be positive")
            elif k == "thin_width" and float(val) < 0:
                errs.append(f"Variant '{v.name}', part '{part}': thin_width must be ≥ 0 (0 = automatic)")
    return errs


def apply_variant(study, v: PrintVariant):
    """Copy of the study with the print settings of the variant (the definitions of the variants in the copy are not)."""
    from ..study import Study
    st = Study.from_dict(study.to_dict())
    st.printing.variants = []
    for part, ov in v.parts.items():
        ov = normalize(ov)
        if "material" in ov:
            st.part_materials[part] = ov["material"]
        d = st.printing.spec(part).to_dict()
        d.update({k: val for k, val in ov.items() if k in SPEC_KEYS})
        d["part"] = part
        st.printing.parts[part] = PrintSpec.from_dict(d)
    return st


def effective(study, v: PrintVariant, part_names: list[str]) -> dict[str, dict]:
    """Resulting settings of the printed parts of the variant {part: {material, PrintSpec fields}} (for tables)."""
    from .region import print_parts
    st = apply_variant(study, v)
    out = {}
    for p in print_parts(st, part_names):
        sp = st.printing.spec(p)
        out[p] = dict({k: getattr(sp, k) for k in SPEC_KEYS}, material=st.material_of(p).name)
    return out


@dataclass
class VariantOutcome:
    name: str
    overrides: dict = field(default_factory=dict)        # overrides per part (definition of the variant)
    settings: dict = field(default_factory=dict)         # resulting settings of the printed parts
    ok: bool = False
    error: str = ""
    min_safety: float | None = None                      # min. over the states and the printed parts
    governing: dict = field(default_factory=dict)        # state, part, region, place, stress in the axes of the print
    steps: list = field(default_factory=list)            # [{step, min_safety, part, region, max_disp}]
    parts: dict = field(default_factory=dict)            # part -> {min_safety, mass_g, volume_mm3, shell_fraction}
    mass_g: float | None = None                          # printed parts in total
    max_disp: float | None = None                        # max. |u| over the states [mm]
    max_disp_step: str = ""
    time_s: float = 0.0
    workdir: str = ""
    warnings: list = field(default_factory=list)
    results: object = None                               # Results (not saved to JSON)

    def to_json(self) -> dict:
        d = {k: getattr(self, k) for k in ("name", "overrides", "settings", "ok", "error", "min_safety", "governing",
                                           "steps", "parts", "mass_g", "max_disp", "max_disp_step", "time_s",
                                           "warnings")}
        return d


@dataclass
class VariantsRun:
    outcomes: list[VariantOutcome]
    required_safety: float
    recommended: str | None = None
    reason: str = ""
    mesh_info: dict = field(default_factory=dict)
    time_s: float = 0.0
    sweep: dict | None = None            # sweep (printing.sweep): definition, Pareto front, estimate of the time

    def outcome(self, name: str) -> VariantOutcome | None:
        return next((o for o in self.outcomes if o.name == name), None)

    def to_json(self) -> dict:
        d = {"required_safety": self.required_safety, "recommended": self.recommended, "reason": self.reason,
             "mesh": self.mesh_info, "time_s": self.time_s, "variants": [o.to_json() for o in self.outcomes]}
        if self.sweep is not None:
            d["sweep"] = self.sweep
        return d

    def pareto(self) -> list[str]:
        return pareto(self.outcomes)


def summarize(res, out: VariantOutcome) -> None:
    """Summary of the results of one variant into out (printing safety, displacements, mass)."""
    best = None
    for st in res.steps:
        parts = (st.printing or {}).get("parts") or {}
        row = {"step": st.name, "max_disp": float(st.umag.max()) if len(st.U) else 0.0}
        if parts:
            pn = min(parts, key=lambda p: parts[p]["min_safety"])
            row.update(min_safety=float(parts[pn]["min_safety"]), part=pn, region=parts[pn].get("region", ""))
            for p, d in parts.items():
                q = out.parts.setdefault(p, {})
                q["min_safety"] = min(q.get("min_safety", float("inf")), float(d["min_safety"]))
            if best is None or row["min_safety"] < best[0]:
                best = (row["min_safety"], st.name, pn, parts[pn])
        out.steps.append(row)
        if out.max_disp is None or row["max_disp"] > out.max_disp:
            out.max_disp, out.max_disp_step = row["max_disp"], st.name
    if best is not None:
        out.min_safety = best[0]
        out.governing = {"step": best[1], "part": best[2], "region": best[3].get("region", ""),
                         "at": best[3].get("at"), "stress_local": best[3].get("stress_local")}
    info = (res.info.get("printing") or {}).get("parts") or {}
    for p, d in info.items():
        q = out.parts.setdefault(p, {})
        q.update(mass_g=d.get("mass_g"), volume_mm3=d.get("volume_mm3"), shell_fraction=d.get("shell_fraction"))
    masses = [d["mass_g"] for d in info.values() if d.get("mass_g") is not None]
    out.mass_g = float(sum(masses)) if masses else None


def recommend(outcomes: list[VariantOutcome], k_req: float) -> tuple[str | None, str]:
    """The lightest variant with k ≥ k_req; when none satisfies it, the one with the highest safety (with an explanation)."""
    ok = [o for o in outcomes if o.ok and o.min_safety is not None]
    if not ok:
        return None, "No variant was analyzed."
    good = [o for o in ok if o.min_safety >= k_req]
    if good:
        o = min(good, key=lambda o: (o.mass_g if o.mass_g is not None else float("inf"), -o.min_safety))
        return o.name, (f"Lightest variant with safety factor k ≥ {k_req:g}: '{o.name}' (k = {o.min_safety:.3g}, {o.mass_g:.4g} g)" if o.mass_g is not None else f"'{o.name}' (k = {o.min_safety:.3g})")
    o = max(ok, key=lambda o: o.min_safety)
    return o.name, (f"No variant satisfies k ≥ {k_req:g}; the highest safety factor is '{o.name}' (k = {o.min_safety:.3g}) – consider more perimeters, denser infill or a different build direction.")


def pareto(outcomes: list[VariantOutcome]) -> list[str]:
    """Pareto front (min. mass, max. safety): the computed variants that no other beats in both
    at the same time (lighter or equally heavy and safer or equally safe, at least one strictly). Sorted by mass.
    """
    ok = [o for o in outcomes if o.ok and o.min_safety is not None and o.mass_g is not None]
    front = []
    for o in ok:
        dom = any(q is not o and q.mass_g <= o.mass_g and q.min_safety >= o.min_safety
                  and (q.mass_g < o.mass_g or q.min_safety > o.min_safety) for q in ok)
        if not dom:
            front.append(o)
    front.sort(key=lambda o: (o.mass_g, -o.min_safety))
    out, seen = [], set()
    for o in front:                                     # identical points (the same mass and safety) only once
        key = (round(o.mass_g, 9), round(o.min_safety, 9))
        if key not in seen:
            seen.add(key)
            out.append(o.name)
    return out


def _mesh_copy(mesh):
    """Shallow copy of the mesh with its own cell_data and report (nodes and elements shared, the topology cache too)."""
    m = copy.copy(mesh)
    m.cell_data = dict(mesh.cell_data)
    m.report = dict(mesh.report)
    return m


def selected(study, names: list[str] | None, include_base: bool = True) -> list[PrintVariant]:
    """Variants to compute: the settings of the study (BASE_NAME) + the chosen ones (None = all) in the order of the study."""
    out = [PrintVariant(BASE_NAME)] if include_base else []
    have = {v.name: v for v in study.printing.variants}
    if names is None:
        out += list(study.printing.variants)
    else:
        missing = [n for n in names if n not in have and n != BASE_NAME]
        if missing:
            raise VariantError(f"Variants not found: {', '.join(missing)}")
        out += [have[n] for n in names if n != BASE_NAME]
    return out


def run_variants(study, workdir: str | Path, study_dir: str | Path | None = None, names: list[str] | None = None,
                 include_base: bool = True, mesh=None, parts=None,
                 on_log: Callable[[str], None] | None = None, on_solver_line: Callable[[str], None] | None = None,
                 cancel=None, on_progress: Callable[[int, int, str], None] | None = None,
                 keep_results: bool = True, keep_files: bool = True, variants: list[PrintVariant] | None = None,
                 on_outcome: Callable[[VariantOutcome], None] | None = None) -> VariantsRun:
    """Computes the variants of the printing on one mesh. mesh = a finished mesh of the same study (otherwise it is meshed once),
    parts = the loaded geometry (OCP; otherwise it is loaded from the STEP). An error of one variant does not stop the calculation (ok=False),
    cancellation (cancel) does. variants = own list of variants instead of the definitions of the study (sweep, build directions).
    keep_results=False keeps in memory only the results of the settings of the study (BASE_NAME), keep_files=False deletes the working
    folders of the other variants (sweep of dozens of variants). on_outcome = a finished variant (running table).
    """
    from ..solve import prepare_mesh, run_study, StudyError
    log = on_log or (lambda s: None)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    if variants is None:
        todo = selected(study, names, include_base)
    else:
        todo = ([PrintVariant(BASE_NAME)] if include_base else []) + list(variants)
    if not todo:
        raise VariantError("No variant to analyze")
    if parts is None:
        from ..geometry import legacy_step_names, read_step
        geo = study.geometry_path(study_dir)
        log(f"Loading geometry {geo.name}")
        parts = read_step(geo, legacy_names=legacy_step_names(study))
    names_all = [p.name for p in parts]
    errs = [e for v in todo if v.name != BASE_NAME for e in validate(study, v, names_all)]
    if errs:
        raise VariantError("Invalid print variants:\n  " + "\n  ".join(errs))
    t0 = time.time()
    meshed = mesh is None
    mesh, mw = prepare_mesh(study, workdir, study_dir, log, mesh, parts, cancel)
    if meshed:
        mesh.report["study_warnings"] = [w for w in mw if w not in mesh.report.get("print_warnings", [])]
    mesh_info = {"nodes": int(len(mesh.nodes)), "elements": int(len(mesh.tets)), "meshed": meshed,
                 "time_s": time.time() - t0 if meshed else 0.0}
    outcomes = []
    for i, v in enumerate(todo):
        if cancel is not None and cancel.is_set():
            raise StudyError("Variant analysis cancelled by user")
        if on_progress:
            on_progress(i, len(todo), v.name)
        log(f"Variant {i + 1}/{len(todo)}: {v.name}")
        out = VariantOutcome(v.name, {k: dict(d) for k, d in v.parts.items()}, effective(study, v, names_all))
        wd = workdir / f"v{i:02d}"
        out.workdir = str(wd)
        t1 = time.time()
        try:
            res = run_study(apply_variant(study, v), wd, study_dir, on_log=log, on_solver_line=on_solver_line,
                            cancel=cancel, mesh=_mesh_copy(mesh), parts=parts)
            summarize(res, out)
            out.ok = out.min_safety is not None
            if not out.ok:
                out.error = "The result contains no printed parts (material with E_z)"
            out.warnings = list(res.warnings)
            if keep_results or v.name == BASE_NAME:
                out.results = res
        except Exception as e:  # noqa: BLE001 – an error of a variant (e.g. a non-positive-definite infill) does not stop the others
            if cancel is not None and cancel.is_set():
                raise
            out.error = str(e) or e.__class__.__name__
            log(f"  error: {out.error}")
        out.time_s = time.time() - t1
        if not keep_files and v.name != BASE_NAME:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)
            out.workdir = ""
        if on_outcome is not None:
            on_outcome(out)
        if out.ok:
            log(f"  min. safety factor {out.min_safety:.3g} ({out.governing.get('part')}, {out.governing.get('region')}), max. displacement {out.max_disp:.4g} mm" + (f", {out.mass_g:.4g} g" if out.mass_g is not None else ""))
        outcomes.append(out)
    k_req = float(study.printing.required_safety)
    rec, why = recommend(outcomes, k_req)
    log(why)
    return VariantsRun(outcomes, k_req, rec, why, mesh_info, time.time() - t_all)
