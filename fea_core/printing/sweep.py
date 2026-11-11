"""Sweep of the print settings (F5, A3): grid perimeters × infill × pattern × build direction → variants of printing
computed on ONE mesh (printing.variants.run_variants), points mass × safety, Pareto front and recommendation
(the lightest variant with k ≥ PrintSettings.required_safety). No optimiser; before the start an estimate of the time
(number of variants × time of one calculation) and a hard limit of the number of variants.

The definition is in the study (PrintSettings.sweep = SweepSpec, JSON); the same overrides are given to all parts of SweepSpec.parts
(empty = all printed). The results of the variants of the sweep are not kept in memory (only the settings of the study) and their working
folders are deleted – the chosen variant is taken over by to_variant() as a normal variant of the study (recomputation).
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable

from .model import AXIS_DIRS, PrintVariant, SweepSpec
from .variants import BASE_NAME, VariantsRun, pareto, run_variants, validate

HARD_LIMIT = 200          # limit SweepSpec.max_variants (default 60)


class SweepError(ValueError):
    pass


def norm_dir(d) -> str:
    """'+x', 'x', '-Z', [0, 0, 1] → key AXIS_DIRS ('+X' … '-Z')."""
    if isinstance(d, (list, tuple)):
        for k, v in AXIS_DIRS.items():
            if [float(x) for x in d] == v:
                return k
        raise SweepError(f"Build direction {list(d)} is not a ±X/±Y/±Z axis")
    s = str(d).strip().upper().replace("−", "-")
    if s in ("X", "Y", "Z"):
        s = "+" + s
    if s not in AXIS_DIRS:
        raise SweepError(f"Unknown build direction '{d}' (allowed: {', '.join(AXIS_DIRS)})")
    return s


def normalized(sw: SweepSpec) -> SweepSpec:
    """Copy with infill 0–1 (value > 1 = %), whole perimeters, directions as keys AXIS_DIRS, without duplicates."""
    def uniq(xs):
        return list(dict.fromkeys(xs))
    try:
        walls = uniq(int(round(float(w))) for w in sw.walls)
        infill = uniq(float(x) / 100.0 if float(x) > 1.0 else float(x) for x in sw.infill)
    except (TypeError, ValueError) as e:
        raise SweepError(f"Sweep: invalid number ({e})") from None
    return SweepSpec(list(sw.parts), walls, infill, uniq(str(p) for p in sw.patterns),
                     uniq(norm_dir(d) for d in sw.build_dirs), int(sw.max_variants))


def target_parts(study, sw: SweepSpec, part_names: list[str]) -> list[str]:
    from .region import print_parts
    printed = print_parts(study, part_names)
    return [p for p in sw.parts if p in printed] if sw.parts else printed


def count(sw: SweepSpec) -> int:
    """Number of variants of the sweep (without the settings of the study)."""
    if sw.is_empty():
        return 0
    n = 1
    for xs in (sw.walls, sw.infill, sw.patterns, sw.build_dirs):
        n *= max(1, len(xs))
    return n


def check(study, sw: SweepSpec | None = None, part_names: list[str] | None = None) -> list[str]:
    """Errors of the definition of the sweep (empty list = all right)."""
    from .infill import PATTERNS
    from .region import print_parts
    sw = sw if sw is not None else study.printing.sweep
    errs = []
    try:
        sw = normalized(sw)
    except SweepError as e:
        return [str(e)]
    if sw.is_empty():
        errs.append("The sweep has no parameter (perimeters, infill, pattern or build direction)")
    if any(w < 0 for w in sw.walls):
        errs.append("Sweep: perimeters must be an integer ≥ 0")
    if any(not 0.0 <= x <= 1.0 for x in sw.infill):
        errs.append("Sweep: infill must be 0–100 %")
    bad = [p for p in sw.patterns if p not in PATTERNS]
    if bad:
        errs.append(f"Sweep: unknown infill pattern {', '.join(bad)} ({' | '.join(PATTERNS)})")
    if not 1 <= sw.max_variants <= HARD_LIMIT:
        errs.append(f"Sweep: max. number of variants must be 1–{HARD_LIMIT}")
    n = count(sw)
    if n > sw.max_variants:
        errs.append(f"The sweep has {n} variants, at most {sw.max_variants} are allowed – narrow the range (fewer perimeter, infill, pattern or direction values) or increase the max. number of variants")
    if part_names is not None:
        printed = print_parts(study, part_names)
        if not printed:
            errs.append("The study has no printed parts (material with E_z) – the sweep has nothing to vary")
        miss = [p for p in sw.parts if p not in printed]
        if miss:
            errs.append(f"Sweep: parts {', '.join(miss)} do not exist or are not printed")
    return errs


def variant_name(walls=None, infill=None, pattern=None, build_dir=None) -> str:
    bits = []
    if walls is not None:
        bits.append(f"{walls} walls")
    if infill is not None:
        bits.append(f"{100 * infill:g} %")
    if pattern is not None:
        bits.append(pattern)
    if build_dir is not None:
        bits.append(f"build {build_dir}")
    return " · ".join(bits)


def sweep_variants(study, sw: SweepSpec | None = None, part_names: list[str] | None = None) -> list[PrintVariant]:
    """Variants of the grid (order: perimeters, infill, pattern, direction – the last one changes the fastest)."""
    sw = normalized(sw if sw is not None else study.printing.sweep)
    parts = target_parts(study, sw, part_names) if part_names is not None else (
        list(sw.parts) or list(study.printing.parts))
    out = []
    for w, f, pat, d in itertools.product(sw.walls or [None], sw.infill or [None], sw.patterns or [None],
                                          sw.build_dirs or [None]):
        ov = {}
        if w is not None:
            ov["walls"] = w
        if f is not None:
            ov["infill"] = f
        if pat is not None:
            ov["pattern"] = pat
        if d is not None:
            ov["build_dir"] = list(AXIS_DIRS[d])
        out.append(PrintVariant(variant_name(w, f, pat, d), {p: dict(ov) for p in parts}, "sweep"))
    return out


def single_solve_estimate(study, mesh) -> float:
    """Estimate of the time of one calculation of a variant [s] from the number of DOF and states (ccx PARDISO, notebook 8 cores: 213 k DOF ≈ 20 s
    per state; parallel steps ≈ n^0.6; assignment of the printing ~3 s / 100 k elements; creep ~ +60 %). INDICATIVE.
    """
    dof = 3.0 * len(mesh.nodes)
    steps = max(1, len(study.load_cases))
    t = max(0.5, 20.0 * (dof / 2.13e5) ** 1.3) * steps ** 0.6 + 2.0 + 3.0 * len(mesh.tets) / 1e5
    cs = study.printing.creep
    if cs.enabled and cs.cases:
        t *= 1.6
    return float(t)


def estimate(study, sw: SweepSpec | None = None, mesh=None, single_s: float | None = None,
             include_base: bool = True) -> dict:
    """Estimate before the start: number of variants, time of one calculation (single_s = measured, otherwise an estimate from the DOF of the mesh),
    total time. Without a mesh and without single_s the time is unknown (None).
    """
    sw = normalized(sw if sw is not None else study.printing.sweep)
    n = count(sw)
    runs = n + (1 if include_base else 0)
    src = "measured"
    if single_s is None and mesh is not None:
        single_s, src = single_solve_estimate(study, mesh), "estimated from DOF count"
    d = {"count": n, "runs": runs, "max_variants": sw.max_variants, "over_limit": n > sw.max_variants,
         "single_s": single_s, "total_s": None if single_s is None else single_s * runs,
         "source": src if single_s is not None else "unknown (no mesh)"}
    if mesh is not None:
        d["dof"] = int(3 * len(mesh.nodes))
    return d


def run_sweep(study, workdir: str | Path, sw: SweepSpec | None = None, study_dir: str | Path | None = None,
              mesh=None, parts=None, on_log: Callable[[str], None] | None = None,
              on_solver_line: Callable[[str], None] | None = None, cancel=None,
              on_progress: Callable[[int, int, str], None] | None = None,
              on_outcome=None, single_s: float | None = None, keep_results: bool = False) -> VariantsRun:
    """Computes the sweep (+ the settings of the study as a reference) on one mesh. Result = VariantsRun with .sweep =
    {definition, parts, count, pareto, estimate, variants: {name: overrides}}; recommendation as for the variants.
    The results (Results) hold only the settings of the study (keep_results=False), the working folders of the variants are deleted.
    """
    sw = normalized(sw if sw is not None else study.printing.sweep)
    if parts is None:
        from ..geometry import legacy_step_names, read_step
        parts = read_step(study.geometry_path(study_dir), legacy_names=legacy_step_names(study))
    names = [p.name for p in parts]
    errs = check(study, sw, names)
    if errs:
        raise SweepError("\n".join(errs))
    vs = sweep_variants(study, sw, names)
    for v in vs:
        e = validate(study, v, names)
        if e:
            raise SweepError("\n".join(e))
    est = estimate(study, sw, mesh, single_s)
    log = on_log or (lambda s: None)
    log(f"Print sweep: {len(vs)} variants + study settings"
        + (f", estimate {est['total_s'] / 60:.1f} min" if est.get("total_s") else ""))
    vr = run_variants(study, workdir, study_dir, include_base=True, mesh=mesh, parts=parts, on_log=on_log,
                      on_solver_line=on_solver_line, cancel=cancel, on_progress=on_progress,
                      keep_results=keep_results, keep_files=False, variants=vs, on_outcome=on_outcome)
    vr.sweep = {"definition": sw.to_dict(), "parts": target_parts(study, sw, names), "count": len(vs),
                "pareto": pareto(vr.outcomes), "estimate": est,
                "variants": {v.name: v.parts for v in vs}}
    return vr


def points(vr: VariantsRun) -> list[dict]:
    """Points for a chart / table: name, mass, min. safety, max. displacement, deciding region, Pareto, recommended."""
    front = set(vr.sweep["pareto"]) if vr.sweep else set(pareto(vr.outcomes))
    out = []
    for o in vr.outcomes:
        g = o.governing or {}
        out.append({"name": o.name, "ok": o.ok, "error": o.error, "mass_g": o.mass_g, "min_safety": o.min_safety,
                    "max_disp": o.max_disp, "part": g.get("part"), "region": g.get("region"),
                    "step": g.get("step"), "pareto": o.name in front, "recommended": o.name == vr.recommended,
                    "base": o.name == BASE_NAME, "settings": o.overrides})
    return out


def to_variant(vr: VariantsRun, name: str, new_name: str | None = None) -> PrintVariant:
    """Variant of the sweep as a normal variant of the study (for taking over / a full calculation)."""
    o = vr.outcome(name)
    if o is None or o.name == BASE_NAME:
        raise SweepError(f"Sweep variant '{name}' does not exist")
    return PrintVariant(new_name or name, {k: dict(v) for k, v in o.overrides.items()}, "from sweep")
