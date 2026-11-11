"""Tools: management of the study and files (new from a STEP, opening .fea/.json, saving, overview)."""
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from .common import out, selection_info
from .session import FeaError, Session, user_path


def describe_constraint(st, c) -> dict:
    d = {k: v for k, v in asdict(c).items() if v is not None}
    d["name"] = c.selection
    if c.kind == "remote_support":
        from ..remote import dofs_text, coupling_text
        d["dofs_fixed"] = dofs_text(c.dofs)
        d["description"] = (f"remote support ({coupling_text(c.coupling)}) at the point [{', '.join((f'{v:g}' for v in c.point))}] mm, constrained components: {dofs_text(c.dofs)}")
    else:
        from ..loads_text import constraint_text
        d["description"] = constraint_text(c, st)
    return d


def describe_load(l, st=None) -> dict:
    """Load for the output of the MCP; st = study (kind of selection of faces / edges / points, coordinate system)."""
    from ..loads_text import load_text
    d = {k: v for k, v in asdict(l).items() if v is not None and not (k == "clip" and v is False)}
    if "selection" in d:
        d["name"] = d["selection"]
        if st is not None and l.selection in st.selections:
            d["applied_to"] = {"faces": "faces", "edges": "edges", "points": "body"}.get(
                st.selections[l.selection].entity, st.selections[l.selection].entity)
    elif l.kind == "thermal":
        from .diagnostics import thermal_load_text
        d["name"] = "temperature"
        d["description"] = thermal_load_text(l)
    elif l.kind == "rotation":
        d["name"] = "rotation"
    elif l.kind == "inertia":
        d["name"] = l.name or "inertia"
    if l.kind in ("force", "pressure", "rotation", "remote", "bearing", "gravity", "inertia"):
        d["description"] = load_text(l, st)
    return d


def material_props(m) -> dict:
    d = {"E_MPa": m.E, "nu": m.nu, "rho_t_mm3": m.rho, "Re_MPa": m.Re, "Rm_MPa": m.Rm}
    for k, key in (("k", "k_W_mK"), ("alpha", "alpha_1_K"), ("cp", "cp_J_kgK")):
        if getattr(m, k) is not None:
            d[key] = getattr(m, k)
    if getattr(m, "ortho", None):
        d["orthotropic"] = dict(m.ortho)
        d["ortho_csys"] = m.ortho_csys
    return d


def study_status(sess: Session) -> dict:
    from ..solvers import summary as solvers_summary
    st = sess.require_study()
    parts = []
    excl = st.excluded_parts()
    for p in sess.parts:
        try:
            m = st.material_of(p.name).name
        except KeyError:
            m = None
        from ..geometry import bbox
        bb = bbox(p.shape)
        d = {"name": p.name, "material": m, "size_mm": bb[3:] - bb[:3]}
        if p.name in excl:
            d["excluded"] = "bolt/omitted – not meshed"
        parts.append(d)
    mesh = sess.valid_mesh()
    res = sess.results
    d = {
        "study": st.name, "step_file": str(sess.step_path) if sess.step_path else None, "saved_as": str(sess.study_path) if sess.study_path else None,
        "workdir": str(sess.workdir), "parts": parts,
        "materials": {k: material_props(m) for k, m in st.materials.items()},
        "default_material": st.default_material,
        "selections": [selection_info(sess, k) for k in st.selections],
        "constraints": [describe_constraint(st, c) for c in st.constraints],
        "load_cases": [{"name": lc.name, "loads": [describe_load(l) for l in lc.loads]} for lc in st.load_cases],
        "mesh_settings": asdict(st.mesh),
        "modal": st.modal.to_dict(), "buckling": st.buckling.to_dict(),
        "mesh": ({"nodes": len(mesh.nodes), "elements": len(mesh.tets), "dof": 3 * len(mesh.nodes),
                  "min_scaled_jacobian": mesh.report.get("min_scaled_jacobian")} if mesh is not None
                 else ("outdated (the study changed since meshing)" if sess.mesh is not None else "none")),
        "solver": asdict(st.solver),
        "solvers": solvers_summary(),               # availability of the backends (what is missing and how to install it)
        "results": ({"steps": [s.name for s in res.steps], "thermal_cases": [t.name for t in res.thermal],
                     "backend": res.info.get("backend"), "nodes": len(res.mesh.nodes)} if res is not None else "nejsou"),
        "fatigue_settings": bool(st.fatigue), "fatigue_result": sess.fatigue is not None,
        "bolts": len(st.bolts), "welds": len(st.welds), "ignored_parts": st.ignored_parts,
        "printed_parts": list(st.printing.parts),
        "units": "mm, N, MPa, t (tuna), s; hustota t/mm³ (ocel 7.85e-9), g = 9810 mm/s²",
    }
    if st.thermal.cases or st.has_thermal_loads():
        from .tools_thermal import thermal_overview, UNITS
        d["thermal"] = thermal_overview(st)
        d["units"] += "; " + UNITS
    if st.frame.active or st.frame.nodes:
        fr = st.frame
        d["frame"] = {"nodes": len(fr.nodes), "members": len(fr.members), "sections": list(fr.sections),
                      "supports": len(fr.supports), "connections": len(fr.connections),
                      "frame_only": sess.frame_only, "detail": "fea_frame_list"}
    if sess.geometry_warnings:
        d["geometry_warnings"] = sess.geometry_warnings
    return d


def _open_json(sess: Session, path: Path, workdir) -> None:
    from ..project import load_study_json          # the same loading as in the GUI (geometry relative to the folder of the file)
    try:
        st, geo = load_study_json(path)            # geo None = bar system only
    except (FileNotFoundError, ValueError) as e:   # STEP missing, not a study (StudyFormatError), faulty JSON
        raise FeaError(str(e)) from None
    sess.reset()
    sess.study, sess.step_path, sess.study_path = st, geo, path
    sess.workdir = sess.new_workdir(st.name, workdir)


def _cad_metadata(st, step: Path, part_names: list[str]) -> dict | None:
    """Metadata from the CAD next to the STEP (<name>.json) as in the GUI (fea_core.cad_import): matching of the parts and materials
    from the library. None = there is no metadata file; an error of the metadata does not block the study (only a warning).
    """
    from .. import cad_import
    warn: list[str] = []
    try:
        meta = cad_import.load_sidecar(step, warn)
        if meta is None:
            return {"warnings": warn} if warn else None
        res = cad_import.apply_metadata(st, part_names, meta)
    except Exception as e:  # noqa: BLE001 – the metadata is only a supplement
        return {"warnings": [f"CAD metadata cannot be used: {e}"]}
    d = {"file": cad_import.sidecar_path(step).name, "summary": res.summary(), "assigned": res.assigned,
         "not_in_library": res.not_in_library, "hints": res.hints, "no_material": res.no_material,
         "no_metadata": res.no_metadata, "unmatched": res.unmatched, "warnings": warn}
    d = {k: v for k, v in d.items() if v not in ({}, [], None)}
    if res.not_in_library or res.no_material or res.no_metadata:
        d["next"] = "parts without a material from the library: fea_materials_assign (fea_materials_list query=…)"
    return d


def native_step(p: Path, warnings: list[str]) -> Path:
    """Native file (.ipt, .iam, .f3d, .f3z) → STEP in the cache (module native); errors as FeaError."""
    from ..modules import ModuleMissing
    from ..native_cad import NativeError, to_step
    try:
        return to_step(p, warnings)
    except (NativeError, ModuleMissing) as e:
        raise FeaError(str(e)) from None


def _is_own_study_file(p: Path) -> bool:
    """An existing file that fea_study_save may overwrite: an empty file, a .fea project (zip with study.json) or a study
    .json written by FEA Studio – an agent must not destroy an arbitrary user file by "saving" over it."""
    try:
        if not p.is_file():
            return False
        if p.stat().st_size == 0:
            return True
        if p.suffix.lower() == ".fea":
            import zipfile
            if not zipfile.is_zipfile(p):
                return False
            with zipfile.ZipFile(p) as z:
                return "study.json" in z.namelist()
        if p.stat().st_size > 64 << 20:
            return False
        import json
        d = json.loads(p.read_text(encoding="utf-8"))
        return isinstance(d, dict) and (d.get("format") == "fea-studio-study"
                                        or {"load_cases", "materials", "geometry"} <= d.keys())
    except (OSError, ValueError):
        return False


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_study_new(step_path: str | None = None, name: str | None = None, workdir: str | None = None,
                      material: str | None = None, apply_cad_metadata: bool = True) -> str:
        """New study from a STEP file (part names kept). step_path: absolute path to .step/.stp (.ipt/.iam/.f3d/.f3z are converted to STEP without CAD); None/"" = study WITHOUT geometry (beam/frame systems, fea_frame_*). name, workdir (default user data/mcp/<name>_<time>), material (catalogue name for all parts). apply_cad_metadata: read <name>.json next to the STEP (parts matched, library materials assigned; result in cad_metadata). Units: mm, N, MPa, t, s. Returns parts with dimensions and face counts; on error the current study stays open."""
        from ..study import Study
        mat = None                                     # the material is checked BEFORE the current study is discarded
        if material:
            from ..materials import by_name
            try:
                mat = by_name(material)
            except KeyError:
                raise FeaError(f"Material '{material}' is not in the catalogue (fea_materials_list)") from None
        if not str(step_path or "").strip():           # bar system only
            with sess.use():
                sess.reset()
                st = Study(name=name or "Frame", geometry="")
                sess.study, sess.step_path = st, None
                sess.workdir = sess.new_workdir(st.name, workdir)
                if mat is not None:
                    st.add_material(mat, default=True)
                return out({"study": st.name, "workdir": str(sess.workdir), "parts": [], "frame_only": True,
                            "next": "fea_materials_assign(part='*', …) → fea_frame_section_set → fea_frame_node_set → "
                                    "fea_frame_member_set → fea_frame_support_set → fea_frame_load_node / "
                                    "fea_frame_load_member → fea_frame_render → fea_solve_run → fea_frame_results, "
                                    "fea_frame_assess"})
        p = user_path(step_path)
        if not p.exists():
            raise FeaError(f"File does not exist: {p}")
        native_warn: list[str] = []
        if p.suffix.lower() not in (".step", ".stp"):
            from ..native_cad import is_native
            if not is_native(p):
                raise FeaError("Expected a STEP file (.step / .stp), an Inventor part or assembly (.ipt, .iam) or a Fusion design (.f3d, .f3z)")
            p = native_step(p, native_warn)
        with sess.use(op="loading the STEP"):
            from ..geometry import read_step
            report: list[str] = []
            try:                                       # load first – on an error the current study stays
                parts = read_step(p.resolve(), report=report)
            except Exception as e:
                raise FeaError(f"Cannot read STEP: {e}") from e
            if not parts:
                raise FeaError(f"The STEP cannot be loaded: the STEP contains no solid: {p}")
            sess.reset()
            st = Study(name=name or p.stem, geometry=str(p.resolve()))
            sess.study, sess.step_path = st, p.resolve()
            sess._parts, sess.geometry_warnings = parts, native_warn + report
            sess.workdir = sess.new_workdir(st.name, workdir)
            if mat is not None:
                st.add_material(mat, default=True)
            cad = _cad_metadata(st, p, [x.name for x in parts]) if apply_cad_metadata else None
            if apply_cad_metadata:                   # fasteners of the assembly of the CAD → virtual bolts
                from .. import cad_import
                try:
                    added, bw = cad_import.apply_fasteners(st, parts, cad_import.load_sidecar(p))
                except Exception as e:  # noqa: BLE001
                    added, bw = [], [f"Virtual bolts from the assembly cannot be created: {e}"]
                if added or bw:
                    cad = dict(cad or {})
                    if added:
                        cad["virtual_bolts"] = added
                        cad["next_bolts"] = "bolt assessment: fea_joints_list → fea_solve_run → fea_joints_assess"
                    if bw:
                        cad["bolt_warnings"] = bw
            from .tools_geometry import part_info
            d = {"study": st.name, "workdir": str(sess.workdir),
                 "parts": [part_info(sess, i) for i in range(len(parts))],
                 "next": "fea_materials_assign → fea_geometry_find_faces → fea_constraint_add_fixed → "
                         "fea_loadcase_create + fea_load_add_force → fea_mesh_generate → fea_solve_run "
                         "(teplo: fea_thermal_case_create + fea_thermal_add_*)"}
            if cad:
                d["cad_metadata"] = cad
            if sess.geometry_warnings:
                d["geometry_warnings"] = sess.geometry_warnings
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_study_open(path: str, workdir: str | None = None) -> str:
        """Opens a project .fea (STEP + study + results) or study.json (geometry path relative to the file). Returns fea_study_status. On failure the current study stays open."""
        p = user_path(path)
        if not p.exists():
            raise FeaError(f"File does not exist: {p}")
        with sess.use(op="opening the study"):
            if p.suffix.lower() == ".fea":
                from ..project import load_motion, load_project
                wd = sess.new_workdir(p.stem, workdir)
                try:                                   # unpack and load first, the session changes only after success
                    st, step, res = load_project(p, wd / "project")
                    motion = load_motion(wd / "project")
                except Exception as e:
                    if not workdir:                    # automatically created folder – nothing stays in it
                        import shutil
                        shutil.rmtree(wd, ignore_errors=True)
                    raise FeaError(f"The project cannot be opened ({p.name}): {type(e).__name__}: {e}") from e
                sess.reset()
                sess.study, sess.study_path, sess.workdir = st, p, wd
                sess.step_path = Path(step) if step else None     # a project with bars only has no STEP
                sess.results = res
                sess.motion = motion
            elif p.suffix.lower() == ".json":
                _open_json(sess, p, workdir)
            else:
                raise FeaError("Supported formats: .fea (project) or .json (study.json)")
            return out(study_status(sess))

    @mcp.tool(structured_output=False)
    def fea_study_save(path: str) -> str:
        """Saves the study: .fea = project (STEP + study.json + results), .json = study definition only (relative geometry path where possible)."""
        p = user_path(path)
        with sess.use(op="saving"):
            st = sess.require_study()
            if p.suffix.lower() in (".fea", ".json") and p.exists() and not _is_own_study_file(p):
                raise FeaError(f"{p} exists and is not a study saved by FEA Studio – it is not overwritten; choose another name")
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.suffix.lower() == ".fea":
                from ..project import save_project
                save_project(p, st, sess.step_path or "", sess.results, motion=sess.motion)
            elif p.suffix.lower() == ".json":
                from ..study import Study
                cp = Study.from_dict(st.to_dict())
                if sess.step_path is None:                   # bar system only
                    cp.geometry = ""
                else:
                    try:
                        cp.geometry = os.path.relpath(sess.step_path, p.parent.resolve())
                    except ValueError:                         # another drive
                        cp.geometry = str(sess.step_path)
                cp.save(p)
            else:
                raise FeaError("The extension must be .fea or .json")
            sess.study_path = p
            fea = p.suffix.lower() == ".fea"
            return out({"saved": str(p.resolve()), "with_results": fea and sess.results is not None,
                        "with_motion": fea and sess.motion is not None})

    @mcp.tool(structured_output=False)
    def fea_study_status() -> str:
        """Overview: parts, materials, selections, constraints, load states, loads, mesh/solver settings and backend availability, results, bolts/welds, printing."""
        with sess.use():
            return out(study_status(sess))

    @mcp.tool(structured_output=False)
    def fea_study_set_parts_ignored(parts: list[str], ignored: bool = True) -> str:
        """Excludes parts from the calculation (magnets, pins, purchased parts) or returns them (ignored=False)."""
        with sess.use():
            st = sess.require_study()
            names = sess.part_names()
            for p in parts:
                if p not in names:
                    raise FeaError(f"Part '{p}' does not exist")
            cur = set(st.ignored_parts)
            cur = cur | set(parts) if ignored else cur - set(parts)
            st.ignored_parts = [p for p in names if p in cur]
            sess.changed(mesh=True)
            return out({"ignored_parts": st.ignored_parts})

    @mcp.tool(structured_output=False)
    def fea_study_set_parts_hidden(parts: list[str], hidden: bool = True, only: bool = False) -> str:
        """Hides parts in the GUI 3D view and images (display only; calculation unchanged) or shows them (hidden=False). only=True: show ONLY the listed parts."""
        with sess.use():
            st = sess.require_study()
            names = sess.part_names()
            for p in parts:
                if p not in names:
                    raise FeaError(f"Part '{p}' does not exist")
            if only:
                cur = set(names) - set(parts)
            else:
                cur = set(st.hidden_parts)
                cur = cur | set(parts) if hidden else cur - set(parts)
            st.hidden_parts = [p for p in names if p in cur]
            return out({"hidden_parts": st.hidden_parts})
