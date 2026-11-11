"""Tools: 3D printing (F4) – overview of the settings, creep, properties of the printed materials, calibration from tensile tests,
a cut with the share of the shell – and variants of the internal structure (F5) – wrappers over fea_core.printing.

A variant = overrides of the printing settings per part (perimeters, top/bottom layers, infill, pattern, line width, layer height,
build direction, printed material). All variants are calculated on one mesh (the printing settings do not change the mesh).
Changing the definitions of the variants does not discard the results of the study (it does not change the calculation of the settings of the study). Other changes of the printing settings
discard the results (sess.changed()), but the mesh stays (the mesh signature does not contain printing; the assignment of the shell and the infill is
recalculated by printing.region.ensure).
"""
from __future__ import annotations

import math
import time
from dataclasses import fields

import numpy as np

from ..printing.props import PrintProps
from .common import out, run_with_timeout
from .session import FeaError, Session

# PrintSpec field that can be changed (part is not changed)
_INT_KEYS = ("walls", "top_layers", "bottom_layers")
_POS_KEYS = ("layer_height", "line_width")
SPEC_KEYS = ("build_dir", "layer_height", "line_width", "walls", "top_layers", "bottom_layers", "infill",
             "infill_percent", "pattern", "enabled", "note", "solid_thin", "thin_width", "tolerance")


def _num(k: str, v, what: str = "") -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise FeaError(f"{what}{k} must be a number (given {v!r})")
    return float(v)


def _infill(v, key: str, notes: list[str], what: str = "") -> float:
    """Infill density 0–1; infill in (1; 100] = percent (note), infill_percent = always percent."""
    x = _num(key, v, what)
    if key == "infill_percent":
        if not 0 <= x <= 100:
            raise FeaError(f"{what}infill_percent must be 0–100 %")
        return x / 100.0
    if 0 <= x <= 1:
        return x
    if 1 < x <= 100:
        notes.append(f"{what}infill = {x:g} > 1 taken as {x:g} % (density {x / 100:g}); give the density as 0–1 or infill_percent")
        return x / 100.0
    raise FeaError(f"{what}infill must be 0–1 (or infill_percent 0–100), given {x:g}")


def spec_changes(settings: dict, what: str = "") -> tuple[dict, list[str]]:
    """Verified changes of PrintSpec from the parameters of the tool (types, ranges, infill in %) → (changes, notes)."""
    from ..printing.infill import PATTERNS
    if not isinstance(settings, dict):
        raise FeaError("settings must be an object {key: value}")
    ch = {k: v for k, v in settings.items() if k != "part"}
    bad = set(ch) - set(SPEC_KEYS)
    if bad:
        raise FeaError(f"{what}unknown print setting {', '.join(sorted(bad))} (allowed: {', '.join(SPEC_KEYS)})")
    if "infill" in ch and "infill_percent" in ch:
        raise FeaError(f"{what}give only infill (0–1) or infill_percent (0–100), not both")
    out_, notes = {}, []
    for k, v in ch.items():
        if k == "build_dir":
            if not isinstance(v, (list, tuple)) or len(v) != 3:
                raise FeaError(f"{what}build_dir must be a list of 3 numbers [x, y, z] (the printer Z axis in model coordinates), given {v!r}")
            vec = [_num("build_dir", x, what) for x in v]
            if math.hypot(*vec) == 0:
                raise FeaError(f"{what}build_dir must be a nonzero vector")
            out_[k] = vec
        elif k in _INT_KEYS:
            x = _num(k, v, what)
            if x != int(x) or x < 0:
                raise FeaError(f"{what}{k} must be an integer ≥ 0 (given {v!r})")
            out_[k] = int(x)
        elif k in _POS_KEYS:
            x = _num(k, v, what)
            if x <= 0:
                raise FeaError(f"{what}{k} must be positive [mm]")
            out_[k] = x
        elif k in ("infill", "infill_percent"):
            out_["infill"] = _infill(v, k, notes, what)
        elif k == "pattern":
            if v not in PATTERNS:
                raise FeaError(f"{what}pattern must be {' | '.join(PATTERNS)}")
            out_[k] = v
        elif k in ("enabled", "solid_thin"):
            if not isinstance(v, bool):
                raise FeaError(f"{what}{k} must be true / false")
            out_[k] = v
        elif k == "thin_width":
            x = _num(k, v, what)
            if x < 0:
                raise FeaError(f"{what}thin_width must be ≥ 0 mm (0 = automatic 2 × line width)")
            out_[k] = x
        elif k == "tolerance":
            x = _num(k, v, what)
            if not 0 <= x <= 1.0:
                raise FeaError(f"{what}tolerance must be 0–1 mm (loss of material at the print surface, typically 0.05–0.2)")
            out_[k] = x
        elif k == "note":
            out_[k] = str(v)
    return out_, notes


def _printed(sess: Session) -> list[str]:
    from ..printing.region import print_parts
    return print_parts(sess.require_study(), sess.part_names())


def _print_material(sess: Session, name: str):
    """Printed material of the study (or of the library) by name."""
    st = sess.require_study()
    m = st.materials.get(name)
    if m is None:
        from ..materials import by_name
        try:
            m = by_name(name)
        except KeyError:
            raise FeaError(f"Material '{name}' is neither in the study nor in the library (fea_materials_list query='Tisk')") \
                from None
    if not m.is_print:
        raise FeaError(f"Material '{name}' is not printed (E_z is missing) – it has no print properties")
    return m


def _regions(mat, spec, pr) -> dict:
    """Effective properties of the regions of the part: full extrusion, perimeters (along / across the line), core with infill."""
    from ..printing.infill import HOLLOW, core, density_ratio, line_solid, pattern_label, solid
    s, ls, c = solid(mat), line_solid(mat, pr.line_E_ratio, pr.line_R_ratio), core(mat, spec.pattern, spec.infill)
    return {
        "thin_regions": {"solid_thin": spec.solid_thin, "limit_mm": spec.thin_limit},
        "shell_thickness_mm": {"walls": spec.walls * spec.line_width,
                               "top": spec.top_layers * spec.layer_height,
                               "bottom": spec.bottom_layers * spec.layer_height},
        "solid": {"E_xy": s.E1, "E_z": s.E3, "G_z": s.G13, "nu": s.nu12, "nu_z": s.nu13, "Rm_xy": s.Xt,
                  "Rc_xy": s.Xc, "Rm_z": s.Zt, "Rc_z": s.Zc, "S_xy": s.Sxy, "S_z": s.Sz},
        "perimeter_line": {"E_along": ls.E1, "E_across": ls.E2v, "Rm_along": ls.Xt, "Rm_across": ls.Ytv},
        "core": {"pattern": pattern_label(spec.pattern), "infill": spec.infill,
                 "density_ratio": density_ratio(spec.pattern, spec.infill), "hollow": spec.infill < HOLLOW,
                 "E_xy": c.E1, "E_z": c.E3, "G_xy": c.G12, "G_z": c.G13, "nu_z": c.nu13, "Rm_xy": c.Xt, "Rc_xy": c.Xc,
                 "Rm_z": c.Zt, "Rc_z": c.Zc, "S_xy": c.Sxy, "S_z": c.Sz},
        "units": "MPa, mm; Rm/Rc = tensile/compressive strength, S = in shear; xy = layer plane, z = build direction",
    }


def _props_dict(st, m) -> dict:
    from ..printing.props import family_of, props_of
    pr = props_of(st, m)
    d = {"family": family_of(m.name), "overrides": dict(st.printing.materials.get(m.name) or {}),
         "props": pr.to_dict()}
    cs = st.printing.creep
    if cs.enabled:
        d["creep_factors"] = pr.creep_factors(cs.hours, cs.temperature)
    return d


PROPS_KEYS = tuple(f.name for f in fields(PrintProps) if f.name != "family")
_PROPS_POS = ("creep_n", "sn_ND", "sn_k", "sn_SD_ratio", "line_E_ratio", "line_R_ratio")
_PROPS_NONNEG = ("creep_a", "rupture_beta", "sn_k2", "creep_max_util")
_PROPS_OPT = ("sn_SD", "sn_z_ratio")


def _check_props(ov: dict) -> dict:
    """Verified overrides of PrintProps (keys, types, ranges)."""
    bad = set(ov) - set(PROPS_KEYS)
    if bad:
        hint = " (family is not overwritten – the family is determined from the material name)" if "family" in bad else ""
        raise FeaError(f"Unknown print properties {', '.join(sorted(bad))}{hint}. Allowed: {', '.join(PROPS_KEYS)}")
    outd = {}
    for k, v in ov.items():
        if k == "source":
            outd[k] = str(v)
        elif k == "creep_T":
            try:
                rows = [[float(x) for x in r] for r in v]
            except (TypeError, ValueError):
                rows = None
            if not rows or any(len(r) != 3 for r in rows):
                raise FeaError("creep_T = a table [[T °C, k_E, k_φ], …] (at least one row)")
            if any(r[1] <= 0 or r[2] < 0 for r in rows):
                raise FeaError("creep_T: k_E must be positive and k_φ ≥ 0")
            outd[k] = sorted(rows)
        elif k in _PROPS_OPT and v is None:
            outd[k] = None
        else:
            x = _num(k, v)
            if k in _PROPS_POS and x <= 0:
                raise FeaError(f"{k} must be positive")
            if k in _PROPS_NONNEG and x < 0:
                raise FeaError(f"{k} must be ≥ 0")
            if k in ("line_E_ratio", "line_R_ratio") and x > 1.5:
                raise FeaError(f"{k} = across / along the line (typically 0.5–1), given {x:g}")
            if k == "rupture_min" and not 0 < x <= 1:
                raise FeaError("rupture_min must be in (0; 1]")
            if k in _PROPS_OPT and x <= 0:
                raise FeaError(f"{k} must be positive or null")
            outd[k] = x
    return outd


def _shell_mesh(sess: Session, timeout_s: float):
    """Mesh with the assignment of printing (results → their mesh, otherwise the valid mesh of the study, otherwise it is created)."""
    from ..printing.region import ensure
    st = sess.require_study()
    mesh = sess.results.mesh if sess.results is not None else sess.valid_mesh()
    info = None
    if mesh is None:
        from .tools_mesh import do_mesh
        info = do_mesh(sess, timeout_s)
        mesh = sess.valid_mesh()
    warnings = ensure(st, mesh)
    if "print_shell" not in mesh.cell_data:
        raise FeaError("The mesh has no printed parts (a material with E_z) – there is no shell fraction to show")
    return mesh, warnings, info


def _row(o) -> dict:
    """Brief line of the result of a variant."""
    d = {"variant": o.name, "ok": o.ok}
    if not o.ok:
        d["error"] = o.error
        return d
    d.update(min_safety=o.min_safety, governing=o.governing, max_displacement_mm=o.max_disp,
             max_displacement_step=o.max_disp_step, mass_g=o.mass_g,
             parts={p: {"min_safety": q.get("min_safety"), "mass_g": q.get("mass_g"),
                        "shell_fraction": q.get("shell_fraction")} for p, q in o.parts.items()},
             settings=o.settings, time_s=o.time_s)
    return d


def _measured_single(sess: Session) -> float | None:
    """Average time of one variant from the last calculation of the variants (estimate of the sweep)."""
    vr = sess.variants
    if vr is None:
        return None
    ts = [o.time_s for o in vr.outcomes if o.ok and o.time_s > 0]
    return float(sum(ts) / len(ts)) if ts else None


def _prepare_run(sess: Session, timeout_s: float):
    """Check of readiness and the mesh for the calculation of the variants (the mesh is not re-meshed if it is valid)."""
    st = sess.require_study()
    if not _printed(sess):
        raise FeaError("The study has no printed parts (a material with E_z)")
    if not st.load_cases:
        raise FeaError("The study has no load cases – the print variants have nothing to compare")
    from .tools_mesh import readiness, do_mesh
    rd = readiness(sess)
    if not rd["ready"]:
        raise FeaError("The study is not ready for calculation:\n  " + "\n  ".join(rd["errors"]))
    if sess.valid_mesh() is None:
        do_mesh(sess, timeout_s)
    return st, sess.valid_mesh()


def _run_job(sess: Session, timeout_s: float, what: str, fn) -> tuple:
    """Check, mesh and calculation fn(st, mesh, on_log, cancel) with a time limit (log into the session) → (result, t0)."""
    t0 = time.time()
    st, mesh = _prepare_run(sess, timeout_s)
    logs: list[str] = []
    try:
        r = run_with_timeout(lambda cancel: fn(st, mesh, logs.append, cancel),
                             max(timeout_s - (time.time() - t0), 10.0), what)
    except FeaError:
        raise
    except Exception as e:  # noqa: BLE001 – VariantError, SweepError, StudyError → text for the LLM
        raise FeaError(str(e)) from None
    finally:
        for line in logs:
            sess.add_log(line)
    return r, t0


def _adopt_base(sess: Session, vr) -> None:
    """Results of the settings of the study from the calculation of the variants = results of the study."""
    sess.variants = vr
    base = vr.outcome("Current settings")
    if base is not None and base.ok and base.results is not None:
        sess.results = base.results
        sess.voxel = None
        sess.fatigue = None
        sess.joints = None


def register(mcp, sess: Session) -> None:

    # ------------------------------------------------------------------ settings of printing (F4)
    @mcp.tool(structured_output=False)
    def fea_print_get(part: str | None = None) -> str:
        """3D-print settings overview (part = one): material, build direction, layer, line, perimeters, top/bottom layers, infill, pattern, shell thicknesses [mm], effective properties of regions (full extrusion, perimeters along/across, infill core - INDICATIVE), criterion, bins/samples/line_bins, creep, required_safety, calibration/variant counts, PrintProps (line anisotropy, creep, S-N) with sources and overrides."""
        from ..printing.props import props_of
        from ..printing.region import print_parts
        with sess.use():
            st = sess.require_study()
            names = sess.part_names()
            printed = print_parts(st, names)
            if part is not None:
                if part not in names:
                    raise FeaError(f"Part '{part}' does not exist. Parts: {', '.join(names)}")
                show = [part]
            else:
                show = printed + [p for p in st.printing.parts if p not in printed and p in names]
            parts = {}
            for p in show:
                spec = st.printing.spec(p)
                try:
                    m = st.material_of(p)
                except KeyError:
                    m = None
                d = {"material": m.name if m is not None else None, "printed": p in printed,
                     "has_own_settings": p in st.printing.parts, "settings": spec.to_dict()}
                if p in printed:
                    if spec.enabled:
                        d.update(_regions(m, spec, props_of(st, m)))
                    else:
                        d["note"] = "enabled = false → the whole part as solid extrusion"
                else:
                    d["note"] = "the material is not printed (E_z is missing) – the setting has no effect"
                parts[p] = d
            ps = st.printing
            mats = sorted({st.material_of(p).name for p in printed}
                          | {k for k, m in st.materials.items() if m.is_print})
            return out({"parts": parts, "criterion": ps.criterion, "bins": ps.bins, "samples": ps.samples,
                        "line_bins": ps.line_bins, "required_safety": ps.required_safety,
                        "creep": ps.creep.to_dict(), "tests": len(ps.tests),
                        "variants": [v.name for v in ps.variants],
                        "materials": {k: _props_dict(st, st.materials[k]) for k in mats},
                        "printed_parts": printed,
                        "tools": "fea_print_set, fea_print_creep, fea_print_props, fea_print_calibrate, "
                                 "fea_print_render_shell, fea_print_presets, fea_print_variant_set, fea_print_sweep, "
                                 "fea_print_advice, fea_print_rank_build_direction"})

    @mcp.tool(structured_output=False)
    def fea_print_creep(enabled: bool | None = None, hours: float | None = None, temperature: float | None = None,
                        cases: list[str] | None = None) -> str:
        """INDICATIVE creep estimate of printed parts (isochronous modulus): long-term states `cases` are recomputed in fea_solve_run with E/(1 + phi(t,T)) and creep-rupture strength (extra steps '<state> - creep t h, T degC'). hours (1000 h ~ 6 weeks, 8760 = 1 year), temperature [degC] (PLA <= 45, PETG <= 60); omitted items unchanged. Valid up to ~35 % of strength, no bolt-preload relaxation. Material parameters: fea_print_props."""
        from ..printing.creep import check
        with sess.use():
            st = sess.require_study()
            cs = st.printing.creep
            if hours is not None and not _num("hours", hours) > 0:
                raise FeaError("hours must be positive [h]")
            if temperature is not None:
                _num("temperature", temperature)
            if cases is not None:
                known = [lc.name for lc in st.load_cases]
                bad = [c for c in cases if c not in known]
                if bad:
                    raise FeaError(f"Unknown load cases {', '.join(bad)} (cases: {', '.join(known) or 'none'})")
            if enabled is not None:
                cs.enabled = bool(enabled)
            if hours is not None:
                cs.hours = float(hours)
            if temperature is not None:
                cs.temperature = float(temperature)
            if cases is not None:
                cs.cases = list(dict.fromkeys(cases))
            sess.changed()                     # the mesh stays (printing is not in the mesh signature)
            printed = _printed(sess)
            errors, warnings = check(st, printed)
            if cs.enabled and not printed:
                warnings.append("The study has no printed parts – creep is not computed")
            mats = sorted({st.material_of(p).name for p in printed})
            return out({"creep": cs.to_dict(), "errors": errors, "warnings": warnings,
                        "factors": {m: _props_dict(st, st.materials[m]).get("creep_factors") for m in mats}
                        if cs.enabled else {},
                        "next": "fea_solve_run (the long-term states are recomputed)" if cs.enabled else None})

    @mcp.tool(structured_output=False)
    def fea_print_props(material: str, overrides: dict | None = None, reset: bool = False) -> str:
        """Print properties of a material (PrintProps; INDICATIVE family defaults PLA/PETG/ABS by name). overrides: line_E_ratio, line_R_ratio (across/along the line, 1 = isotropic), creep_a, creep_n (phi = a*t^n, t [h]), creep_T [[T degC, k_E, k_phi], ...], rupture_beta, rupture_min (k_R = 1 - beta*log10(1+t)), creep_max_util, sn_SD_ratio | sn_SD [MPa], sn_ND, sn_k, sn_k2, sn_z_ratio, source. reset=True drops earlier overrides first. Returns resulting properties, overrides, default family."""
        from ..printing.props import default_props
        with sess.use():
            st = sess.require_study()
            m = st.materials.get(material)
            if m is None:
                raise FeaError(f"Material '{material}' is not in the study (study: {', '.join(st.materials) or 'none'}) – assign it with fea_materials_assign")
            if not m.is_print:
                raise FeaError(f"Material '{material}' is not printed (E_z is missing) – it has no print properties")
            if overrides is None and not reset:
                raise FeaError("Enter overrides or reset=True")
            ov = _check_props(dict(overrides or {}))
            if reset:
                st.printing.materials.pop(material, None)
            if ov:
                st.printing.materials.setdefault(material, {}).update(ov)
            sess.changed()
            d = _props_dict(st, m)
            d["defaults"] = default_props(material).to_dict()
            d["material"] = material
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_print_calibrate(tests: list[dict], base_material: str, name: str | None = None,
                            assign_parts: list[str] | None = None) -> str:
        """Calibrates a printed material from ISO 527 tensile tests: back-calculates the FULL extrusion using the same shell/infill model as the solver, creates a new material (name, default '<base> - kalibrace') and optionally assigns it to assign_parts ('*' = parts with base_material); tests are stored in the study. tests: [{name, direction 'xy' (flat) | 'z' (upright), E [MPa], Rm [MPa] (on whole cross-section), width, thickness [mm], walls, line_width, top_layers, bottom_layers, layer_height, infill 0-1 or %, pattern}]; defaults ISO 527 1A 10x4 mm, 0.2 mm standard, 100 % infill. Without Z tests the Z/XY ratio comes from base_material. Returns averages, CV, per-specimen phi_E, phi_R and warnings."""
        from dataclasses import asdict
        from ..printing.calibrate import TensileTest, calibrate
        from ..printing.props import props_of
        from .tools_materials import material_row
        with sess.use():
            st = sess.require_study()
            base = _print_material(sess, base_material)
            if not isinstance(tests, list) or not tests:
                raise FeaError("tests = a non-empty list of tests [{direction, E, Rm, width, thickness, …}]")
            allowed = {f.name for f in fields(TensileTest)}
            tt, notes = [], []
            for i, t in enumerate(tests):
                what = f"Test {i + 1}: "
                if not isinstance(t, dict):
                    raise FeaError(f"{what}must be an object")
                t = dict(t)
                bad = set(t) - allowed - {"infill_percent"}
                if bad:
                    raise FeaError(f"{what}unknown items {', '.join(sorted(bad))} (allowed: {', '.join(sorted(allowed))})")
                t.setdefault("name", f"Specimen {i + 1}")
                if t.get("direction", "xy") not in ("xy", "z"):
                    raise FeaError(f"{what}direction must be xy (lying) | z (standing)")
                for k in ("E", "Rm"):
                    if k in t and _num(k, t[k], what) < 0:
                        raise FeaError(f"{what}{k} must not be negative")
                for k in ("width", "thickness"):
                    if k in t and _num(k, t[k], what) <= 0:
                        raise FeaError(f"{what}{k} must be positive [mm]")
                spec_keys = {k: t.pop(k) for k in ("walls", "top_layers", "bottom_layers", "layer_height",
                                                   "line_width", "infill", "infill_percent", "pattern") if k in t}
                ch, nn = spec_changes(spec_keys, what)
                notes += nn
                t.update(ch)
                if not (_num("E", t.get("E", 0.0), what) > 0 or _num("Rm", t.get("Rm", 0.0), what) > 0):
                    raise FeaError(f"{what}give the measured E or Rm")
                tt.append(TensileTest.from_dict(t))
            try:
                cal = calibrate(base, tt, name, props=props_of(st, base))
            except (ValueError, KeyError) as e:
                raise FeaError(f"Kalibrace selhala: {e}") from None
            m = cal.material
            from ..materials import LIBRARY
            if m.name in {x.name for x in LIBRARY}:
                raise FeaError(f"The name '{m.name}' belongs to a built-in material – choose another name")
            replaced = m.name in st.materials
            names = sess.part_names()
            targets: list[str] = []
            if assign_parts:
                if list(assign_parts) == ["*"]:
                    for p in names:
                        try:
                            if st.material_of(p).name == base.name:
                                targets.append(p)
                        except KeyError:
                            pass
                else:
                    bad = [p for p in assign_parts if p not in names]
                    if bad:
                        raise FeaError(f"Parts {', '.join(bad)} do not exist. Parts: {', '.join(names)}")
                    targets = list(assign_parts)
            st.materials[m.name] = m
            if base.name in st.printing.materials and m.name not in st.printing.materials:
                st.printing.materials[m.name] = dict(st.printing.materials[base.name])   # overrides of PrintProps
                notes.append(f"The print property overrides from '{base.name}' were taken over for '{m.name}' as well")
            for p in targets:
                st.part_materials[p] = m.name
            st.printing.tests = [x.to_dict() for x in tt]
            sess.changed()
            return out({"material": material_row(m), "replaced": replaced, "assigned_parts": targets,
                        "summary": cal.summary, "rows": cal.rows, "warnings": cal.warnings, "notes": notes,
                        "solid_props": {k: v for k, v in asdict(m).items()
                                        if k in ("E", "E_z", "G_z", "nu", "nu_z", "Rm", "Rm_z", "Rc", "Rc_z",
                                                 "S_xy", "S_z") and v is not None},
                        "next": "fea_materials_assign(part, '" + m.name + "')" if not targets else "fea_solve_run"})

    @mcp.tool(structured_output=False)
    def fea_print_render_shell(output_path: str | None = None, axis: str | list[float] = "z",
                               position: float | None = None, view: str | list[float] | None = None,
                               width: int = 1280, height: int = 720, return_image: bool = True,
                               timeout_s: float = 600):
        """PNG cut through the printed-part mesh showing shell fraction per element (0 core/infill, 1 shell/full layers, between = boundary element, unprinted grey) - checks whether the mesh resolves the shell. No results needed (uses results mesh, valid study mesh or builds one, timeout_s). axis x|y|z|normal vector, position on the axis [mm] (None = centre of printed parts), side against the normal stays; view as fea_results_render_image (None = perpendicular to cut)."""
        from mcp.server.mcpserver import Image
        from .render import render
        from .scenes import print_shell_scene
        from .tools_results import _png_path
        if isinstance(axis, str):
            key = axis.strip().lower().lstrip("+")
            neg = key.startswith("-")
            key = key.lstrip("-")
            if key not in ("x", "y", "z"):
                raise FeaError("axis must be x | y | z or a vector [x, y, z]")
            n = np.eye(3)["xyz".index(key)] * (-1.0 if neg else 1.0)
        else:
            if len(axis) != 3:
                raise FeaError("axis as a vector must have 3 components")
            n = np.array([_num("axis", v) for v in axis])
            if np.linalg.norm(n) == 0:
                raise FeaError("axis must be a nonzero vector")
            n = n / np.linalg.norm(n)
        with sess.use(op="print section render", wait=5):
            st = sess.require_study()
            mesh, warnings, minfo = _shell_mesh(sess, timeout_s)
            sh = mesh.cell_data["print_shell"]
            el = sh >= 0
            X = mesh.nodes[np.unique(mesh.tets[el, :4])]
            proj = X @ n
            if position is None:
                pos = 0.5 * (float(proj.min()) + float(proj.max()))
            else:
                pos = _num("position", position)
                if not proj.min() < pos < proj.max():
                    raise FeaError(f"position = {pos:g} mm is outside the printed parts (on the axis {float(proj.min()):.4g} to "
                                   f"{float(proj.max()):.4g} mm)")
            origin = n * pos
            lines = []
            for p in _printed(sess):
                sp = st.printing.spec(p)
                lines.append(f"{p}: {sp.walls}× {sp.line_width:g} mm perimeters, layers {sp.top_layers}/{sp.bottom_layers} × {sp.layer_height:g} mm, infill {100 * sp.infill:g} % {sp.pattern}"
                             + ("" if sp.enabled else " (solid part)"))
            try:
                scene, meta = print_shell_scene(st, mesh, origin, n, lines[:4])
            except ValueError as e:
                raise FeaError(str(e)) from None
            if view is None:                   # perpendicular to the cut, slightly from above and from the side (spatial impression)
                up = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, -1.0, 0.0])
                up = up - (up @ n) * n
                up /= np.linalg.norm(up)
                view = (n + 0.4 * up + 0.3 * np.cross(up, n)).tolist()
            p = _png_path(sess, output_path, "print_shell")
            try:
                info = render(scene, p, view, width, height)
            except ValueError as e:
                raise FeaError(str(e)) from None
            meta.update(info)
            meta["warnings"] = warnings
            if minfo:
                meta["mesh_generated"] = {k: minfo[k] for k in ("nodes", "elements", "min_scaled_jacobian", "time_s")
                                          if k in minfo}
            txt = out(meta)
            return [txt, Image(path=p)] if return_image else txt

    # ------------------------------------------------------------------ varianty (F5)
    @mcp.tool(structured_output=False)
    def fea_print_variant_set(name: str, settings: dict | None = None, part_settings: dict | None = None,
                              copy_from: str | None = None, note: str = "") -> str:
        """Creates/replaces a print variant (computed on the study mesh). settings = overrides for ALL printed parts; part_settings = {part: overrides} (priority). Keys: walls, top_layers, bottom_layers, infill (0-1 or %), pattern, line_width, layer_height [mm], build_dir [x,y,z], enabled (False = solid), material (another printed material). copy_from starts from a variant. Unspecified = study settings (fea_print_set)."""
        from ..printing.model import PrintVariant
        from ..printing.variants import BASE_NAME, normalize, validate, effective
        with sess.use():
            st = sess.require_study()
            printed = _printed(sess)
            if not printed:
                raise FeaError("The study has no printed parts (a material with E_z) – the print variants have nothing to change")
            parts: dict[str, dict] = {}
            if copy_from:
                src = next((v for v in st.printing.variants if v.name == copy_from), None)
                if src is None and copy_from != BASE_NAME:
                    raise FeaError(f"Variant '{copy_from}' does not exist")
                parts = {k: dict(v) for k, v in (src.parts if src else {}).items()}
            for p in printed:
                if settings:
                    parts.setdefault(p, {}).update(settings)
            for p, ov in (part_settings or {}).items():
                parts.setdefault(p, {}).update(ov or {})
            try:
                parts = {p: normalize(ov) for p, ov in parts.items() if ov}
            except (TypeError, ValueError) as e:
                raise FeaError(f"Invalid setting value: {e}") from None
            v = PrintVariant(name, parts, note)
            errs = validate(st, v, sess.part_names())
            if errs:
                raise FeaError("\n".join(errs))
            lst = st.printing.variants
            i = next((k for k, x in enumerate(lst) if x.name == name), None)
            if i is None:
                lst.append(v)
            else:
                lst[i] = v
            sess.variants = None                          # results of the variants no longer match the definitions
            return out({"variant": name, "replaced": i is not None, "overrides": parts,
                        "settings": effective(st, v, sess.part_names()), "count": len(lst),
                        "next": "fea_print_variants_run"})

    @mcp.tool(structured_output=False)
    def fea_print_variant_list() -> str:
        """Variants: overrides, resulting part settings, required safety, last results."""
        from ..printing.variants import effective
        from ..printing.model import PrintVariant
        with sess.use():
            st = sess.require_study()
            names = sess.part_names()
            d = {"required_safety": st.printing.required_safety,
                 "study_settings": effective(st, PrintVariant("_"), names),
                 "variants": [{"name": v.name, "overrides": v.parts, "note": v.note,
                               "settings": effective(st, v, names)} for v in st.printing.variants]}
            if sess.variants is not None:
                d["last_run"] = {"recommended": sess.variants.recommended, "reason": sess.variants.reason,
                                 "results": [_row(o) for o in sess.variants.outcomes]}
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_print_variant_remove(name: str) -> str:
        """Removes a variant (name or '*' = all)."""
        with sess.use():
            st = sess.require_study()
            before = len(st.printing.variants)
            st.printing.variants = [] if name == "*" else [v for v in st.printing.variants if v.name != name]
            if len(st.printing.variants) == before:
                raise FeaError(f"Variant '{name}' does not exist")
            sess.variants = None
            return out({"removed": before - len(st.printing.variants),
                        "variants": [v.name for v in st.printing.variants]})

    @mcp.tool(structured_output=False)
    def fea_print_variants_run(names: list[str] | None = None, required_safety: float | None = None,
                               include_study: bool = True, timeout_s: float = 3600) -> str:
        """Computes variants (names, default all) + the study settings (include_study) on ONE mesh: shell/infill assignment -> solve -> Tsai-Wu/Hill (+ creep per study). Table: min print safety and where (state, part, region), max displacement, printed mass; recommendation = lightest variant with k >= required_safety (stored). The study-settings result becomes the study results; other variants via fea_print_variant_apply."""
        with sess.use(op="print variants calculation", wait=5):
            st = sess.require_study()
            if not _printed(sess):
                raise FeaError("The study has no printed parts (a material with E_z)")
            if required_safety is not None:
                if required_safety <= 0:
                    raise FeaError("required_safety must be positive")
                st.printing.required_safety = float(required_safety)
            from ..printing.variants import run_variants
            vr, t0 = _run_job(sess, timeout_s, "Print variants calculation", lambda st, mesh, log, cancel: run_variants(
                st, sess.workdir / "variants", names=names, include_base=include_study, mesh=mesh, parts=sess.parts,
                on_log=log, cancel=cancel))
            _adopt_base(sess, vr)
            return out({"required_safety": vr.required_safety, "recommended": vr.recommended, "reason": vr.reason,
                        "time_s": time.time() - t0, "mesh": vr.mesh_info,
                        "results": [_row(o) for o in vr.outcomes],
                        "next": "fea_print_variant_apply(name) takes the variant over into the study (with its results)"})

    @mcp.tool(structured_output=False)
    def fea_print_variant_apply(name: str, keep_variant: bool = True) -> str:
        """Copies a variant's settings into the study (fea_print_set per part); if it was computed in the last fea_print_variants_run its results become the study results. keep_variant=False removes it from the list."""
        from ..printing.variants import apply_variant
        with sess.use():
            st = sess.require_study()
            v = next((x for x in st.printing.variants if x.name == name), None)
            o = sess.variants.outcome(name) if sess.variants is not None else None
            if v is None and o is not None and o.name != "Current settings":
                from ..printing.model import PrintVariant
                v = PrintVariant(name, {k: dict(d) for k, d in o.overrides.items()})   # sweep / build directions
            if v is None:
                raise FeaError(f"Variant '{name}' does not exist")
            new = apply_variant(st, v)
            st.printing.parts = new.printing.parts
            st.part_materials = new.part_materials
            if not keep_variant:
                st.printing.variants = [x for x in st.printing.variants if x.name != name]
            vr = sess.variants
            sess.changed()
            reused = False
            if o is not None and o.ok and o.results is not None and sess.valid_mesh() is not None:
                sess.results = o.results
                sess.mesh = o.results.mesh                # mesh of the variant (the assignment of printing matches the settings)
                sess.mesh_sig = sess.mesh_signature()
                reused = True
            sess.variants = vr
            return out({"applied": name, "results_reused": reused,
                        "print": {p: s.to_dict() for p, s in st.printing.parts.items()},
                        "part_materials": st.part_materials,
                        "next": "fea_print_results" if reused else "fea_solve_run"})

    # ------------------------------------------------------------------ sweep, advice, build direction, presets (F5)
    @mcp.tool(structured_output=False)
    def fea_print_sweep(walls: list[int] | None = None, infill: list[float] | None = None,
                        patterns: list[str] | None = None, build_dirs: list[str] | None = None,
                        parts: list[str] | None = None, max_variants: int | None = None,
                        required_safety: float | None = None, run: bool = True, add_recommended: bool = False,
                        timeout_s: float = 7200) -> str:
        """Sweep grid: walls x infill (0-1 or %) x patterns x build_dirs (+X -X +Y -Y +Z -Z) for `parts` (None = all printed); omitted = study value; no arguments = definition stored in the study (given ones are stored). All variants + study settings on ONE mesh; first an estimate of time; more than max_variants (default 60) = error; run=False = estimate only. Result: points (mass, min print safety, max displacement, governing region), Pareto front, recommendation = lightest with k >= required_safety; add_recommended=True adds it to the variants (take over via fea_print_variant_apply)."""
        from ..printing.model import SweepSpec
        from ..printing.sweep import SweepError, check, estimate, normalized, points, run_sweep, to_variant
        with sess.use(op="print sweep", wait=5):
            st = sess.require_study()
            old = st.printing.sweep
            given = any(x is not None for x in (walls, infill, patterns, build_dirs))
            sw = SweepSpec(list(parts) if parts is not None else ([] if given else list(old.parts)),
                           list(walls) if walls is not None else ([] if given else list(old.walls)),
                           list(infill) if infill is not None else ([] if given else list(old.infill)),
                           list(patterns) if patterns is not None else ([] if given else list(old.patterns)),
                           list(build_dirs) if build_dirs is not None else ([] if given else list(old.build_dirs)),
                           int(max_variants) if max_variants is not None else old.max_variants)
            try:
                sw = normalized(sw)
            except SweepError as e:
                raise FeaError(str(e)) from None
            errs = check(st, sw, sess.part_names())
            if errs:
                raise FeaError("\n".join(errs))
            if required_safety is not None:
                if _num("required_safety", required_safety) <= 0:
                    raise FeaError("required_safety must be positive")
                st.printing.required_safety = float(required_safety)
            st.printing.sweep = sw
            single = _measured_single(sess)
            if not run:
                est = estimate(st, sw, sess.valid_mesh(), single)
                return out({"sweep": sw.to_dict(), "estimate": est,
                            "next": "fea_print_sweep(run=True) starts the calculation" + (
                                "" if est.get("total_s") else " (duration unknown – the mesh does not exist yet: fea_mesh_generate)")})
            vr, t0 = _run_job(sess, timeout_s, "Print sweep", lambda st, mesh, log, cancel: run_sweep(
                st, sess.workdir / "sweep", sw, mesh=mesh, parts=sess.parts, on_log=log, cancel=cancel,
                single_s=single))
            _adopt_base(sess, vr)
            added = None
            if add_recommended and vr.recommended and vr.recommended != "Current settings":
                v = to_variant(vr, vr.recommended)
                st.printing.variants = [x for x in st.printing.variants if x.name != v.name] + [v]
                added = v.name
            rows = [{k: v for k, v in r.items() if k != "error" or v} for r in points(vr)]
            return out({"required_safety": vr.required_safety, "recommended": vr.recommended, "reason": vr.reason,
                        "pareto": vr.sweep["pareto"], "estimate": vr.sweep["estimate"],
                        "time_s": time.time() - t0, "mesh": vr.mesh_info, "points": rows,
                        "added_variant": added,
                        "next": "fea_print_variant_apply(name) takes the variant over (the results are recomputed by fea_solve_run); fea_print_advice = advice from the failure location"})

    @mcp.tool(structured_output=False)
    def fea_print_advice(required_safety: float | None = None) -> str:
        """INDICATIVE advice from the failure location of printed parts: governing region (perimeter / full layers / infill / thin place) and stress in print axes (s11 s22 in layer plane; s33, t13, t23 = layer bond) -> more perimeters, more top/bottom layers, denser infill, gyroid under multiaxial stress, another build direction (fea_print_rank_build_direction), solid thin places, material saving where reserve is large. suggestion = overrides for fea_print_variant_set."""
        from ..printing.advice import advise
        with sess.use():
            st = sess.require_study()
            res = sess.require_results()
            if not any((s.printing or {}).get("parts") for s in res.steps):
                raise FeaError("The results contain no printed parts (a material with E_z)")
            if required_safety is not None and _num("required_safety", required_safety) <= 0:
                raise FeaError("required_safety must be positive")
            k_req = float(required_safety) if required_safety is not None else st.printing.required_safety
            items = advise(st, res, k_req)
            return out({"required_safety": k_req, "advice": items,
                        "note": "INDICATIVE – verify the benefit with a variant (fea_print_variant_set + fea_print_variants_run) and a test",
                        "next": "fea_print_variant_set(name, part_settings={part: suggestion})"})

    @mcp.tool(structured_output=False)
    def fea_print_rank_build_direction(parts: list[str] | None = None, directions: list[str] | None = None,
                                       timeout_s: float = 3600) -> str:
        """Ranks build directions by calculation: for `parts` (None = all printed) runs `directions` (default all 6: +X -X +Y -Y +Z -Z) as variants on one mesh (+ study settings), sorted by min print safety (safety, region, displacement, mass); returns the best and current. Take over: fea_print_variant_apply('stavba +X' = build +X) or fea_print_set(part, settings={build_dir: [x,y,z]})."""
        from ..printing.advice import rank_build_directions
        from ..printing.sweep import SweepError, norm_dir
        with sess.use(op="build direction comparison", wait=5):
            names = sess.part_names()
            if parts:
                bad = [p for p in parts if p not in names]
                if bad:
                    raise FeaError(f"Parts {', '.join(bad)} do not exist. Parts: {', '.join(names)}")
            try:
                dirs = [norm_dir(d) for d in directions] if directions else None
            except SweepError as e:
                raise FeaError(str(e)) from None
            r, t0 = _run_job(sess, timeout_s, "Build direction comparison", lambda st, mesh, log, cancel:
                             rank_build_directions(st, sess.workdir / "build_dirs", parts, dirs, mesh=mesh,
                                                   parts=sess.parts, on_log=log, cancel=cancel))
            _adopt_base(sess, r["run"])
            return out({"ranking": r["ranking"], "best": r["best"], "current": r["current"], "parts": r["parts"],
                        "time_s": time.time() - t0,
                        "next": "fea_print_variant_apply('stavba <direction>') or fea_print_set(part, settings={build_dir: [x, y, z]})"})

    @mcp.tool(structured_output=False)
    def fea_print_presets(action: str = "list", name: str | None = None, settings: dict | None = None,
                          from_part: str | None = None, parts: list[str] | None = None, note: str = "") -> str:
        """Print setting presets (user library print_presets.json; built-in only '3D tisk standard'). action: list | save (name + settings {layer_height, line_width, walls, top_layers, bottom_layers, infill 0-1 or %, pattern, solid_thin, thin_width} or from_part, settings then modify it) | apply (name -> parts; '*'/None = all printed; build direction kept) | delete. Preset values are user input - never invent manufacturer numbers."""
        from ..printing import presets as pz
        act = (action or "list").strip().lower()
        if act not in ("list", "save", "apply", "delete"):
            raise FeaError("action must be list | save | apply | delete")
        try:
            if act == "list":
                return out({"presets": [dict(p.to_dict(), builtin=p.builtin) for p in pz.list_presets()],
                            "file": str(pz.presets_path())})
            if not name:
                raise FeaError("Give the preset name")
            if act == "delete":
                pz.delete_preset(name)
                return out({"deleted": name, "presets": [p.name for p in pz.list_presets()]})
            if act == "save":
                if from_part is not None:
                    with sess.use():
                        st = sess.require_study()
                        if from_part not in sess.part_names():
                            raise FeaError(f"Part '{from_part}' does not exist")
                        vals = pz.preset_from_spec(st.printing.spec(from_part))
                    vals.update(settings or {})
                else:
                    vals = dict(settings or {})
                pr = pz.save_preset(name, vals, note)
                return out({"saved": pr.to_dict(), "file": str(pz.presets_path())})
            with sess.use():
                st = sess.require_study()
                targets = _printed(sess) if not parts or list(parts) == ["*"] else list(parts)
                bad = [p for p in targets if p not in sess.part_names()]
                if bad:
                    raise FeaError(f"Parts {', '.join(bad)} do not exist")
                if not targets:
                    raise FeaError("No printed part (a material with E_z) – give parts")
                pz.apply_preset(st, name, targets)
                sess.changed()                     # the mesh stays (printing is not in the mesh signature)
                return out({"applied": name, "parts": {p: st.printing.parts[p].to_dict() for p in targets}})
        except pz.PresetError as e:
            raise FeaError(str(e)) from None

    @mcp.tool(structured_output=False)
    def fea_print_refine(n_spots: int = 3, use_results: bool = True, timeout_s: float = 1800) -> str:
        """Two-step mesh refinement for printing (shell stress depends on a coarse mesh): from the results (use_results and valid results, else the 1st run) finds <= n_spots places of lowest print safety, refines spheres to ~shell thickness (MeshSettings.local) and re-solves. Returns min safety before/after, spheres, minimum location, sharp-inner-edge singularity (min_safety_regular = minimum away from the edge) and warnings. The refined study and results are kept in the session."""
        from ..printing.refine import two_step
        if not 1 <= int(n_spots) <= 10:
            raise FeaError("n_spots must be 1–10")
        with sess.use(op="print refinement", wait=5):
            st = sess.require_study()
            if not _printed(sess):
                raise FeaError("The study has no printed parts (a material with E_z)")
            from .tools_mesh import readiness
            rd = readiness(sess)
            if not rd["ready"]:
                raise FeaError("The study is not ready for calculation:\n  " + "\n  ".join(rd["errors"]))
            first = sess.results if use_results and sess.results is not None else None
            wd = sess.workdir / f"zjemneni_tisku_{time.strftime('%H%M%S')}"
            logs: list[str] = []
            t0 = time.time()
            try:
                r = run_with_timeout(lambda cancel: two_step(st, wd, results=first, n_spots=int(n_spots),
                                                             parts=sess.parts, on_log=logs.append, cancel=cancel),
                                     timeout_s, "Print refinement")
            finally:
                for line in logs:
                    sess.add_log(line)
            sess.study = r["study"]                          # spheres of refinement in the mesh settings
            sess.results = r["second"]
            sess.mesh, sess.mesh_sig, sess.mesh_time = r["second"].mesh, sess.mesh_signature(), None
            sess.fatigue = sess.joints = sess.variants = sess.voxel = None
            d = {k: r.get(k) for k in ("min_safety_before", "min_safety_after", "change", "nodes_before",
                                       "nodes_after", "at_after", "singular", "min_safety_regular", "warnings")
                 if r.get(k) is not None}
            d["balls"] = r["balls"]
            d["time_s"] = time.time() - t0
            d["adopted"] = ("The refined study (spheres in MeshSettings.local) and the results of the 2nd calculation are now the session results (fea_print_results, fea_results_render_image field='print_safety')")
            return out(d)
