"""Command line of FEA Studio (without GUI).

python -m fea_core faces model.step [--part NAME]   list of parts and faces (for making selections)
python -m fea_core check study.json                 check of the study readiness
python -m fea_core run study.json [-o WORKDIR]      mesh + calculation (incl. thermal states) + result summary
python -m fea_core fatigue study.json [--results R] fatigue according to the load curve in the study
python -m fea_core detect model.step [--add study.json]  recognition of bolts (names, shape) and welds (name, colour, layer, attribute)
python -m fea_core joints study.json [--results R]  assessment of bolts (VDI 2230, EN 1993-1-8) and welds
python -m fea_core report study.json|project.fea    Word report (.docx), also for a study with heat only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _workdir(given, default: Path) -> Path:
    """Work folder of a run: the one given on the command line, otherwise a folder created by FEA Studio – an existing
    "<study>_run" folder that came with the study is not reused (it could carry planted libraries, fea_core.dllguard)."""
    if given:
        return Path(given)
    from .dllguard import own_workdir
    wd = own_workdir(default)
    if wd != Path(default):
        print(f"note: {default} was not created by FEA Studio – using {wd}")
    return wd


def cmd_faces(a) -> int:
    from .geometry import read_step, face_props
    for p in read_step(a.step):
        if a.part and p.name != a.part:
            continue
        print(f"{p.name}  (objem {p.volume():.3f} mm³)")
        for i, f in enumerate(p.faces()):
            d = face_props(f)
            c = ", ".join(f"{v:.3f}" for v in d["centroid"])
            nrm = ", ".join(f"{v:.3f}" for v in d["normal"])
            print(f"  {i:4d} {d['type']:<9} A={d['area']:12.4f} mm²  centroid=({c})  n=({nrm})")
    return 0


def cmd_check(a) -> int:
    from .geometry import legacy_step_names, read_step
    from .solve import check_study
    from .study import Study
    st = Study.load(a.study)
    parts = read_step(st.geometry_path(Path(a.study).parent), legacy_names=legacy_step_names(st))
    errors, warnings = check_study(st, [p.name for p in parts])
    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    print("The study is ready for calculation." if not errors else "The study is NOT ready.")
    return 1 if errors else 0


def cmd_run(a) -> int:
    from .solve import run_study
    from .study import Study
    st = Study.load(a.study)
    wd = _workdir(a.output, Path(a.study).parent / (Path(a.study).stem + "_run"))
    res = run_study(st, wd, study_dir=Path(a.study).parent, on_log=print,
                    on_solver_line=print if a.verbose else None)
    summ = res.summary()
    for s in res.steps:
        summ_step = next(x for x in summ["steps"] if x["name"] == s.name)
        summ_step["equilibrium_imbalance"] = s.equilibrium.get("imbalance")
    for line in thermal_lines(res):
        print(line)
    print(json.dumps(summ, indent=2, ensure_ascii=False))
    (wd / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def thermal_lines(res) -> list[str]:
    """Text summary of the thermal states: temperatures, max. flux, heat of the boundary conditions, balance."""
    from .mcp.diagnostics import THERMAL_LABELS
    out = []
    for t in res.thermal:
        kind = f"transient, end t = {t.times[-1]:g} s" if t.transient and t.times else "steady-state"
        out.append(f"Thermal state '{t.name}' ({kind}): T {t.T.min():.4g} … {t.T.max():.4g} °C, max. heat flux {t.flux_mag.max():.4g} W/m²")
        for h in t.heat:
            out.append(f"  heat '{h['name']}' ({THERMAL_LABELS.get(h['kind'], h['kind'])}): {h['Q_W']:+.4g} W (+ into the body)")
        bal = t.balance or {}
        if "imbalance" in bal:
            out.append(f"  balance: input {bal.get('in_W', 0):.4g} W, output {bal.get('out_W', 0):.4g} W, imbalance {bal['imbalance']:.2e}")
        elif bal.get("note"):
            out.append(f"  {bal['note']}")
    return out


def cmd_fatigue(a) -> int:
    import numpy as np
    from .fatigue.analysis import run_fatigue
    from .results import Results
    from .solve import run_study
    from .study import Study
    st = Study.load(a.study)
    if st.fatigue is None:
        print("ERROR: the study has no fatigue set (load curve)", file=sys.stderr)
        return 1
    if a.results:
        res = Results.load(a.results)
    else:
        wd = _workdir(a.output, Path(a.study).parent / (Path(a.study).stem + "_run"))
        res = run_study(st, wd, study_dir=Path(a.study).parent, on_log=print)
    fr = run_fatigue(st, res)
    print("\n".join(fr.method_lines()))
    for name, info in fr.sn_used.items():
        sn, m = info["sn"], info["marin"]
        print(f"S-N {name}: SD = {sn['SD']:.1f} MPa at {sn['ND']:.3g} cycles, k = {sn['k']:.2f}, k2 = {sn['k2'] or 'fatigue limit'}; Marin k = {m['k']:.3f} ({sn['source']})")
    for w in fr.warnings:
        print(f"warning: {w}")
    print(f"\nWorst locations (calculation {fr.time_s:.2f} s, {len(fr.t)} time points):")
    print(f"{'node':>8} {'x':>9} {'y':>9} {'z':>9} {'Sa,max':>9} {'Total D':>11} {'life [blocks]':>14} {'safety factor':>10}")
    for w in fr.worst(res.mesh, a.top):
        x, y, z = w["xyz"]
        print(f"{w['node'] + 1:8d} {x:9.3f} {y:9.3f} {z:9.3f} {w['amplitude']:9.2f} {w['total_damage']:11.4g} "
              f"{w['life_blocks']:14.4g} {w['safety']:10.3f}"
              + ("   n = ({:.3f}, {:.3f}, {:.3f})".format(*w["normal"]) if w["normal"] is not None else ""))
    wd.mkdir(parents=True, exist_ok=True)
    extra = {"normal": fr.normal, "shear_dir": fr.shear_dir} if fr.normal is not None else {}
    np.savez_compressed(wd / "fatigue.npz", damage=fr.damage, safety=fr.safety, amplitude=fr.amplitude, **extra)
    return 0


def cmd_detect(a) -> int:
    from .geometry import read_step
    from .joints.detect import detect, weld_geometry
    from .joints.model import JointSettings
    parts = read_step(a.step)
    js = JointSettings(weld_colors=list(a.weld_color or []))
    st0 = None
    if a.add:
        from .study import Study
        st0 = Study.load(a.add)
        if not a.weld_color:
            js = st0.joints
    rep = detect(parts, settings=js, shapes=not a.no_shape)
    for n, fn in rep.fasteners.items():
        print(f"  {n}: {fn.describe()}")
    for s in rep.suggestions:
        if s.kind != "bolt":
            continue
        b = s.spec
        holes = ", ".join(f"{p['part']} Ø{p['hole']:g}" if p.get("hole") else p["part"] for p in b.plates)
        tag = "" if s.source == "name" else f" [by shape, certainty {s.confidence:.2f} – CONFIRM: {s.reason}]"
        print(f"{b.name}: {b.designation}, {b.connection}, l_K = {b.grip:.2f} mm, clamped: {holes}{tag}")
    for r in rep.rejected:
        print(f"  (not accepted) {r}")
    for w in rep.welds:
        g = weld_geometry(parts, w.part)
        a_txt = f"a = {g['a']:.2f} mm" if g.get("a") else "a = ?"
        L_txt = f"L = {g['length']:.1f} mm" if g.get("length") else ""
        print(f"{w.name}: part '{w.part}', {g['kind']}, {a_txt}, {L_txt}, contact with: {', '.join(g['fusion'])} ({w.note})")
    for w in rep.warnings:
        print(f"warning: {w}")
    if a.add:
        st = st0
        shaped = [s.spec for s in rep.suggestions if s.kind == "bolt" and s.source == "shape" and s.accept]
        st.bolts = rep.bolts + (shaped if a.accept_shape else [])
        st.welds = rep.welds
        st.save(a.add)
        print(f"Bolts and welds written to {a.add}" + (f" (incl. {len(shaped)} by shape)" if a.accept_shape and shaped
                                                   else (f"; {len(shaped)} bolts by shape NOT WRITTEN – confirm with --accept-shape" if shaped else "")))
    return 0


def cmd_joints(a) -> int:
    from .geometry import legacy_step_names, read_step
    from .joints.assess import assess_joints, report_text
    from .results import Results
    from .solve import run_study
    from .study import Study
    st = Study.load(a.study)
    if not st.active_bolts() and not st.active_welds():
        print("The study has no bolts or welds (python -m fea_core detect model.step --add study.json)")
        return 1
    res = Results.load(a.results) if a.results else run_study(
        st, _workdir(a.output, Path(a.study).parent / (Path(a.study).stem + "_run")), study_dir=Path(a.study).parent,
        on_log=print)
    print(report_text(assess_joints(st, res, read_step(st.geometry_path(Path(a.study).parent),
                                                       legacy_names=legacy_step_names(st)))))
    return 0


def cmd_report(a) -> int:
    """Word report from a study (study.json) or a project (.fea with results)."""
    import tempfile
    from .geometry import legacy_step_names, read_step
    from .report import ReportOptions, build_report, session_from
    from .results import Results
    from .solve import run_study
    from .study import Study
    src = Path(a.source)
    motion = None
    if src.suffix.lower() == ".fea":
        from .project import load_motion, load_project
        tmp = Path(tempfile.mkdtemp(prefix="fea_report_"))
        st, step, res = load_project(src, tmp)
        motion = load_motion(tmp)                       # result of a mechanism → section Motion
        base = src.parent
    else:
        st = Study.load(src)
        base = src.parent
        step = st.geometry_path(base)
        res = None
    if a.results:
        res = Results.load(a.results)
    parts = read_step(step, legacy_names=legacy_step_names(st))
    if res is None and not a.no_solve:
        wd = _workdir(a.workdir, base / (src.stem + "_run"))
        res = run_study(st, wd, study_dir=base, on_log=print, parts=parts)
    fr = ja = None
    if res is not None and st.fatigue is not None and st.fatigue.history.channels and not a.no_fatigue:
        from .fatigue.analysis import run_fatigue
        fr = run_fatigue(st, res)
    if res is not None and (st.active_bolts() or st.active_welds()):
        from .joints.assess import assess_joints
        ja = assess_joints(st, res, parts)
    sess = session_from(st, step, res, parts, fr, ja, motion=motion)
    out = Path(a.output) if a.output else base / f"{src.stem}_protokol.docx"
    opts = ReportOptions(title=a.title, author=a.author or "", company=a.company or "",
                         required_safety=a.safety, images=not a.no_images, keep_images=a.keep_images,
                         language=a.lang)
    r = build_report(sess, out, opts, on_log=print)
    for w in r["warnings"]:
        print(f"warning: {w}")
    print(f"Report: {r['path']} ({r['figures']} images, {r['tables']} tables)")
    return 0


def cmd_export(a) -> int:
    """Results of a solved .fea project (or results.npz + study.json) → VTU (.vtu + .pvd) and / or CSV."""
    import tempfile
    from .export_vtu import export_results_vtu, safe_name
    from .export_csv import export_results_csv
    from .results import Results
    from .study import Study
    src = Path(a.source)
    if src.suffix.lower() == ".fea":
        from .project import load_project
        tmp = Path(tempfile.mkdtemp(prefix="fea_export_"))
        st, _, res = load_project(src, tmp)
    else:
        st = Study.load(src)
        res = Results.load(a.results) if a.results else None
    if res is None:
        print("ERROR: the project has no results – solve it first (python -m fea_core solve-project)", file=sys.stderr)
        return 1
    if not len(res.mesh.nodes):
        print("ERROR: the results have no 3D mesh (frame, 2D or shell model) – VTU / CSV export is for solid results",
              file=sys.stderr)
        return 1
    out = Path(a.out) if a.out else src.parent / f"{src.stem}_export"
    out.mkdir(parents=True, exist_ok=True)
    base = safe_name(st.name or src.stem)
    warn: list[str] = []
    files: list[Path] = []
    if a.format in ("vtu", "all"):
        files += export_results_vtu(res.mesh, res, out, base, study=st, warnings=warn)
    if a.format in ("csv", "all"):
        files += export_results_csv(res, out, base, study=st, dialect=a.dialect, elements=a.elements, warnings=warn)
    for w in warn:
        print(f"warning: {w}")
    for p in files:
        print(p)
    print(f"{len(files)} files written to {out}")
    return 0


def cmd_solve_project(a) -> int:
    """Solve a .fea project and save it with the results (another file by default – the original stays unchanged)."""
    import tempfile
    from .project import load_motion, load_project, save_project
    from .solve import run_study
    from .study import Study
    src = Path(a.source)
    tmp = Path(tempfile.mkdtemp(prefix="fea_solve_"))
    import shutil
    try:
        st, step, _ = load_project(src, tmp)
        calc = Study.from_dict(st.to_dict())      # as the GUI: empty load cases left out with heat / frequencies only
        if calc.thermal.cases or calc.modal.enabled or calc.dynamics.enabled:
            calc.load_cases = [lc for lc in calc.load_cases if lc.loads]
        res = run_study(calc, tmp / "run", study_dir=tmp, on_log=lambda s: print(s, flush=True))
        out = Path(a.output) if a.output else src.with_name(f"{src.stem}_results.fea")
        save_project(out, st, step if step is not None else "", res, motion=load_motion(tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for w in res.warnings:
        print(f"warning: {w}", flush=True)
    print(f"Saved: {out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fea_core", description="FEA Studio – calculation core")
    from . import __version__
    ap.add_argument("--version", action="version",
                    version=f"FEA Studio {__version__} – GPL-3.0-or-later with additional permissions (LICENSE-EXCEPTIONS.md); "
                            "NO WARRANTY, see LICENSE")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("faces", help="list of STEP parts and faces")
    p.add_argument("step")
    p.add_argument("--part")
    p.set_defaults(func=cmd_faces)
    p = sub.add_parser("check", help="check the study readiness")
    p.add_argument("study")
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("run", help="run the study calculation")
    p.add_argument("study")
    p.add_argument("-o", "--output", help="calculation work folder")
    p.add_argument("-v", "--verbose", action="store_true", help="vypisovat log ccx")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("fatigue", help="fatigue according to the load history in the study")
    p.add_argument("study")
    p.add_argument("-o", "--output", help="calculation work folder")
    p.add_argument("--results", help="use the saved results.npz instead of a new calculation")
    p.add_argument("--top", type=int, default=10, help="number of worst locations")
    p.set_defaults(func=cmd_fatigue)
    p = sub.add_parser("detect", help="recognition of bolts (names, shape) and welds (name, colour, layer, attribute)")
    p.add_argument("step")
    p.add_argument("--add", help="write the detected joints to study.json")
    p.add_argument("--accept-shape", action="store_true",
                   help="also write the bolts detected by shape (certainty ≥ 0.7) – after checking the listing")
    p.add_argument("--no-shape", action="store_true", help="by names only (no shape recognition)")
    p.add_argument("--weld-color", action="append", help="weld colour in CAD (#rrggbb), can be repeated")
    p.set_defaults(func=cmd_detect)
    p = sub.add_parser("joints", help="assessment of bolts and welds")
    p.add_argument("study")
    p.add_argument("-o", "--output", help="calculation work folder")
    p.add_argument("--results", help="use the saved results.npz instead of a new calculation")
    p.set_defaults(func=cmd_joints)
    p = sub.add_parser("report", help="calculation report to Word (.docx)")
    p.add_argument("source", help="study.json or .fea project")
    p.add_argument("-o", "--output", help="path to the .docx (default <name>_report.docx)")
    p.add_argument("--results", help="use the saved results.npz (otherwise from the .fea or a new calculation)")
    p.add_argument("--workdir", help="calculation work folder")
    p.add_argument("--no-solve", action="store_true", help="do not compute results – the report contains only the setup")
    p.add_argument("--no-fatigue", action="store_true", help="do not calculate fatigue even if the study has it")
    p.add_argument("--no-images", action="store_true", help="without images")
    p.add_argument("--keep-images", action="store_true", help="ponechat PNG vedle protokolu")
    p.add_argument("--title")
    p.add_argument("--author")
    p.add_argument("--company")
    p.add_argument("--safety", type=float, default=1.0, help="required safety against Re (default 1.0)")
    p.add_argument("--lang", choices=["cs", "en"], default="", help="report language (default: program language, FEA_LANG, otherwise English)")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("export", help="results of a solved project to VTU (ParaView) and / or CSV")
    p.add_argument("source", help=".fea project with results (or study.json with --results)")
    p.add_argument("--format", choices=["vtu", "csv", "all"], default="vtu")
    p.add_argument("--out", help="output folder (default <name>_export next to the project)")
    p.add_argument("--results", help="results.npz (for a study.json source)")
    p.add_argument("--dialect", choices=["standard", "excel"], default="standard", help="CSV dialect (excel = ; and decimal comma)")
    p.add_argument("--elements", action="store_true", help="also the element table (CSV)")
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("solve-project", help="solve a .fea project and save it with the results")
    p.add_argument("source", help=".fea project")
    p.add_argument("-o", "--output", help="output .fea (default <name>_results.fea next to the project)")
    p.set_defaults(func=cmd_solve_project)
    a = ap.parse_args(argv)
    try:
        return a.func(a)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
