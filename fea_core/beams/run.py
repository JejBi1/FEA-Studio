"""Calculation of the frame system alone (study without STEP geometry – Study.geometry == "") and the common evaluation of the
bars for mixed models (bars + bodies, see ccx/inp.write_inp and solve._run_mechanics).

The system alone: own solver with the exact Timoshenko element (`timoshenko.py`, `FrameSettings.solver = "timoshenko"`,
default); `solver = "ccx_u1"` = the original path through the U1 element in ccx (comparison) and MYSTRAN for buckling. Mixed models with bodies
always go through ccx U1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import solver_name
from .fem import (FrameError, build, write, boundary_lines, cload_lines, recover, read_displacements,  # noqa: F401
                  load_resultant, solver_of)
from ..ccx import deck_text


def frame_only(study) -> bool:
    return study.frame.active and not str(study.geometry or "").strip()


def empty_mesh():
    from ..mesh import Mesh
    return Mesh(np.zeros((0, 3)), np.zeros((0, 10), np.int64), np.zeros(0, np.int64), [], {},
                {"nodes": 0, "elements": 0, "element_type": "C3D10", "min_scaled_jacobian": 1.0})


def fe_info(fe, base_node: int) -> dict:
    """Geometry of the bars for the display (FE nodes, segments by bars, nodes of the system)."""
    return {"xyz": [list(map(float, p)) for p in fe.xyz], "base_node": base_node,
            "segments": {m: [[fe.segs[i].a, fe.segs[i].b] for i in segs] for m, segs in fe.member_segs.items()},
            "joints": dict(fe.joint)}


def frame_equilibrium(study, fe, lc, rec, c0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(force of the loads, moment of the loads about c0, force of the reactions, moment of the reactions about c0) of the frame system."""
    Fa, Ma = load_resultant(study, fe, lc, c0)
    X = np.array(fe.xyz)
    Rf, Rm = np.zeros(3), np.zeros(3)
    for name, r in rec["reactions"].items():
        p = X[fe.joint[name]]
        r = np.asarray(r)
        Rf += r[:3]
        Rm += r[3:] + np.cross(p - c0, r[:3])
    return Fa, Ma, Rf, Rm


def eigen_fe(study):
    """Discretisation on which the shapes of the natural vibrations and buckling lie (`ModeSet.frame_shapes_own`, `frame_xyz`): the own
    solver `FrameSettings.segments_modal` elements per straight bar, ccx_u1 (buckling in MYSTRAN) 4 elements (`BUCKLE_SEGMENTS`).
    """
    if solver_of(study) == "timoshenko":
        return build(study, segments=max(int(study.frame.settings.segments_modal), 2))
    from .mystran_frame import BUCKLE_SEGMENTS
    return build(study, solver="timoshenko", segments=BUCKLE_SEGMENTS[0])


def check(study) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    fr = study.frame
    if not fr.active:
        return errors, warnings
    tim = solver_of(study) == "timoshenko"
    try:
        build(study)
    except FrameError as e:
        errors.append(str(e))
    except Exception as e:  # noqa: BLE001 – section from the library etc.
        errors.append(f"Frame structure: {e}")
    if not fr.supports and not fr.connections and frame_only(study):
        errors.append("The frame structure has no supports")
    if study.has_thermal_loads() and not study.geometry:
        errors.append("Member temperature loads are entered as 'Member temperature' (dT, gradient), not as a temperature of solid parts")
    elif study.has_thermal_loads():
        errors.append("Thermal loading of members is not yet supported (the U1 element in ccx does not compute thermal expansion correctly)")
    if not tim and any(l.kind == "frame_temperature" for lc in study.load_cases for l in lc.loads):
        errors.append("Member temperature loads are computed only by the built-in solver (FrameSettings.solver = timoshenko) – the U1 element in ccx does not handle thermal expansion correctly")
    if study.buckling.enabled and not frame_only(study):
        errors.append("Buckling of a model with members and bodies is not yet supported – the U1 element (members) in ccx has no stress stiffness; buckling of a frame structure alone is computed by MYSTRAN, check member stability per EN 1993-1-1")
    if study.buckling.enabled and frame_only(study) and not tim:
        from ..mystran import MYSTRAN_EXE
        if not MYSTRAN_EXE.exists():
            errors.append(f"Frame structure buckling is computed by MYSTRAN – not found ({MYSTRAN_EXE})")
    for m in fr.members.values():
        if m.through is not None and m.kind == "truss":
            errors.append(f"Member '{m.name}': a truss member cannot be curved")
    return errors, warnings


def _fmt(v: float) -> str:
    return f"{v:.12g}"


def run_frame_tim(study, workdir, on_log=None, cancel=None):
    """Statics, natural frequencies and linear buckling of the frame system alone by the own solver (timoshenko.py)."""
    import time
    from ..config import EQUILIBRIUM_TOL
    from ..results import Results, StepResult
    from ..solve import StudyError, check_study
    from . import timoshenko as T
    log = on_log or (lambda s: None)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    errors, warnings = check_study(study, [])
    if errors:
        raise StudyError("Study is not ready:\n  " + "\n  ".join(errors))
    t0 = time.time()
    fe = build(study)
    steps, modes = [], []
    X = np.array(fe.xyz)
    c0 = 0.5 * (X.min(axis=0) + X.max(axis=0))
    if study.load_cases:
        log(f"Solving the beam structure (Timoshenko element, {len(fe.segs)} elements, {6 * len(fe.xyz)} DOF)")
        model = T.TimModel(study, fe)
        for lc, res in zip(study.load_cases, model.solve_cases(study.load_cases)):
            rec = T.step_record(model, res)
            Fa, Ma, Rf, Rm = model.resultants(res, c0)
            fres, mres = Fa + Rf, Ma + Rm
            size = float(np.linalg.norm(X.max(axis=0) - X.min(axis=0))) or 1.0
            # scale = size of the individual reactions (thermal load: the loads and the sum of the reactions are ≈ 0)
            rs = [np.asarray(r, float) for r in rec["reactions"].values()]
            f0 = max(float(np.linalg.norm(Fa)), sum(float(np.linalg.norm(r[:3])) for r in rs))
            m0 = max(float(np.linalg.norm(Ma)), sum(float(np.linalg.norm(r[3:])) for r in rs))
            fl = 1e-9 * model.tension_scale                 # floor: force at a relative strain 1e-9 (noise of zero reactions)
            fs, ms_ = max(f0, m0 / size, fl), max(m0, f0 * size, fl * size)
            imb = max(float(np.linalg.norm(fres)) / fs, float(np.linalg.norm(mres)) / ms_)
            st = StepResult(lc.name, np.zeros((0, 3)), np.zeros((0, 6)))
            st.frame = rec
            st.equilibrium = {"force_residual": fres.tolist(), "moment_residual": mres.tolist(),
                              "reaction_total": Rf.tolist(), "reaction_moment_total": Rm.tolist(),
                              "applied_exact": Fa.tolist(), "imbalance": imb, "node_residual": rec["node_residual"]}
            if imb > EQUILIBRIUM_TOL:
                raise FrameError(f"Case '{lc.name}': member reactions are not in equilibrium with the load (imbalance {100 * imb:.3g} %)")
            steps.append(st)
    if study.modal.enabled or (study.buckling.enabled and study.load_cases):
        fe_e = eigen_fe(study)
        model_e = T.TimModel(study, fe_e)
        xyz_e = [list(map(float, p)) for p in fe_e.xyz]
    if study.modal.enabled:
        from ..eigen import ModeSet
        ms = study.modal
        log(f"Natural frequencies of the beam structure ({int(ms.modes)} modes, {len(fe_e.segs)} elements)")
        d = T.modal(model_e, int(ms.modes), float(ms.f_min or 0.0), ms.f_max)
        k = len(d["freq"])
        mset = ModeSet("modal", "Natural frequencies", d["freq"], np.zeros((k, 0, 3), np.float32),
                       participation=d["participation"], eff_mass=d["eff_mass"], total_mass=d["total_mass"])
        mset.frame_shapes = None
        mset.frame_xyz = xyz_e
        mset.frame_shapes_own = d["shapes"]
        modes.append(mset)
    if study.buckling.enabled and study.load_cases:
        from ..eigen import ModeSet
        names = [c.name for c in study.load_cases]
        lc = study.load_cases[names.index(study.buckling.case) if study.buckling.case in names else 0]
        log(f"Buckling of the beam structure (load case '{lc.name}', {len(fe_e.segs)} elements)")
        b = T.buckling(model_e, lc, int(study.buckling.modes))
        bs = ModeSet("buckling", f"Buckling – {lc.name}", b["factors"], np.zeros((len(b["factors"]), 0, 3), np.float32),
                     case=lc.name)
        bs.frame_shapes = None
        bs.frame_xyz = xyz_e
        bs.frame_shapes_own = b["shapes"]
        modes.append(bs)
        warnings.append("Buckling of the beam structure is linear (geometric stiffness from the axial forces of the state, no imperfections, Engesser shear correction, no warping or lateral-torsional buckling) – α_cr for EN 1993-1-1, 5.2.1; check the resistance of the members for flexural and lateral-torsional buckling per 6.3 (member check)")
    res = Results(empty_mesh(), steps, warnings, {"backend": "timoshenko", "frame": fe_info(fe, 1)}, [], modes)
    res.save(workdir / "results.npz")
    log(f"Done ({time.time() - t0:.2f} s)")
    return res


def _write_model(f, study, fe, mass: bool) -> None:
    f.write(f"** FEA Studio – {deck_text(study.name)} (prutova soustava)\n** jednotky: mm, N, MPa, t, s\n*HEADING\n"
            + deck_text(study.name) + "\n")
    write(f, study, fe, 1, 1, _fmt, mass=mass)


def write_u1_static_inp(study, fe, path) -> None:
    """ccx input with U1 elements (statics of all load states, one *STEP per state) – `fe` from build(study, solver="ccx_u1")."""
    with open(path, "w", encoding="ascii", errors="replace") as f:
        _write_model(f, study, fe, mass=False)
        bl = boundary_lines(fe, 1)
        for si, lc in enumerate(study.load_cases):
            op = "" if si == 0 else ", OP=NEW"
            f.write(f"** load case {si + 1}\n*STEP\n*STATIC, SOLVER={solver_name(study.solver.solver)}\n")
            if bl:
                f.write(f"*BOUNDARY{op}\n" + "\n".join(bl) + "\n")
            cl = cload_lines(study, fe, lc, 1, _fmt)
            if cl or op:
                f.write(f"*CLOAD{op}\n" + "\n".join(cl) + ("\n" if cl else ""))
            f.write("*NODE PRINT, NSET=NFRAME\nU\n*END STEP\n")


def write_u1_modal_inp(study, fe, path) -> None:
    """ccx input with U1 elements for natural frequencies (*FREQUENCY)."""
    ms = study.modal
    with open(path, "w", encoding="ascii", errors="replace") as f:
        _write_model(f, study, fe, mass=True)
        bl = boundary_lines(fe, 1)
        line = f"{int(ms.modes)}, {float(ms.f_min or 0.0):.12g}" + (f", {float(ms.f_max):.12g}" if ms.f_max else "")
        f.write(f"*STEP\n*FREQUENCY, SOLVER={solver_name(study.solver.solver)}\n{line}\n")
        if bl:
            f.write("*BOUNDARY\n" + "\n".join(bl) + "\n")
        f.write("*NODE PRINT, NSET=NFRAME\nU\n*END STEP\n")


def run_frame(study, workdir, on_log=None, on_solver_line=None, cancel=None):
    """Statics (+ natural frequencies) of the frame system alone: own solver (default) or ccx U1 (`FrameSettings.solver`).
    Returns Results with an empty mesh of bodies.
    """
    if solver_of(study) == "timoshenko":
        return run_frame_tim(study, workdir, on_log, cancel)
    from ..ccx.runner import run_ccx, CcxError
    from ..config import CCX_EXE, EQUILIBRIUM_TOL
    from ..results import Results, StepResult
    from ..solve import StudyError, check_study
    log = on_log or (lambda s: None)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    errors, warnings = check_study(study, [])
    if errors:
        raise StudyError("Study is not ready:\n  " + "\n  ".join(errors))
    fe = build(study)
    exe = Path(study.solver.exe) if study.solver.exe else CCX_EXE
    nf = len(fe.xyz)

    def run(job):
        r = run_ccx(workdir, job, exe=exe, on_line=on_solver_line, cancel=cancel)
        if not r.finished or r.errors:
            raise CcxError("CalculiX ended with an error:\n  " + "\n  ".join(r.errors or [f"code {r.returncode}"])
                           + f"\nlog: {r.log_path}")
        return r
    steps = []
    X = np.array(fe.xyz)
    c0 = 0.5 * (X.min(axis=0) + X.max(axis=0))
    if study.load_cases:
        log(f"Computing frame structure (CalculiX, {len(fe.segs)} U1 elements)")
        write_u1_static_inp(study, fe, workdir / "frame.inp")
        r = run("frame")
        warnings += [w for w in r.warnings if "ignored" not in w.lower()]
        Us = read_displacements(workdir / "frame.dat", 1, nf)
        from .fem import u1_matrices
        u1_matrices(fe, workdir / "u1k", exe)          # real matrices of U1 → consistent internal forces
        if len(Us) != len(study.load_cases):
            raise CcxError(f"The number of steps in .dat ({len(Us)}) does not match the study")
        for lc, U6 in zip(study.load_cases, Us):
            rec = recover(study, fe, lc, U6)
            Fa, Ma, Rf, Rm = frame_equilibrium(study, fe, lc, rec, c0)
            fres, mres = Fa + Rf, Ma + Rm
            size = float(np.linalg.norm(X.max(axis=0) - X.min(axis=0))) or 1.0
            fs = max(float(np.linalg.norm(Fa)), float(np.linalg.norm(Rf)), 1e-30)
            ms_ = max(float(np.linalg.norm(Ma)), float(np.linalg.norm(Rm)), fs * size, 1e-30)
            imb = max(float(np.linalg.norm(fres)) / fs if fs > 1e-20 else 0.0, float(np.linalg.norm(mres)) / ms_)
            st = StepResult(lc.name, np.zeros((0, 3)), np.zeros((0, 6)))
            st.frame = rec
            st.equilibrium = {"force_residual": fres.tolist(), "moment_residual": mres.tolist(),
                              "reaction_total": Rf.tolist(), "reaction_moment_total": Rm.tolist(),
                              "applied_exact": Fa.tolist(), "imbalance": imb, "node_residual": rec["node_residual"]}
            if imb > EQUILIBRIUM_TOL:
                raise CcxError(f"Case '{lc.name}': member reactions are not in equilibrium with the load (imbalance {100 * imb:.3g} %)")
            if rec["node_residual"] > 1e-3:
                warnings.append(f"Case '{lc.name}': residual imbalance at member nodes {100 * rec['node_residual']:.2g} % (force recovery by Timoshenko matrix × U1 element in shear) – increase the number of elements per member")
            steps.append(st)
    modes = []
    if study.modal.enabled:
        from ..eigen import ModeSet
        ms = study.modal
        log(f"Natural frequencies of the frame structure ({int(ms.modes)} modes)")
        write_u1_modal_inp(study, fe, workdir / "frame_modal.inp")
        run("frame_modal")
        from ..eigen import read_dat
        d = read_dat(workdir / "frame_modal.dat", "modal")
        shapes = read_displacements(workdir / "frame_modal.dat", 1, nf)
        k = len(d["freq"])
        fs_ = np.zeros((k, nf, 3), np.float32)
        for j, U6 in enumerate(shapes[:k]):
            m_ = np.abs(U6[:, :3]).max()
            fs_[j] = U6[:, :3] / m_ if m_ > 0 else 0.0
        mset = ModeSet("modal", "Natural frequencies", d["freq"], np.zeros((k, 0, 3), np.float32),
                       participation=d["participation"], eff_mass=d["eff_mass"], total_mass=d["total"])
        mset.frame_shapes = fs_
        modes.append(mset)
    if study.buckling.enabled and study.load_cases:
        from ..eigen import ModeSet
        from .mystran_frame import buckling
        log("Frame structure buckling (MYSTRAN SOL 105)")
        b = buckling(study, workdir / "buckle")
        bs = ModeSet("buckling", f"Buckling – {b['case']}", b["factors"], np.zeros((len(b["factors"]), 0, 3), np.float32),
                     case=b["case"])
        bs.frame_shapes = None                       # shapes on the own subdivision (4 elements) – other nodes than the statics
        bs.frame_xyz = b["xyz"]
        bs.frame_shapes_own = b["frame_shapes"]
        modes.append(bs)
        warnings += b["warnings"]
        warnings.append("Frame structure buckling is linear (MYSTRAN SOL 105) – α_cr for EN 1993-1-1, 5.2.1; check member resistance for buckling and lateral-torsional buckling per 6.3 (member checks)")
    res = Results(empty_mesh(), steps, warnings, {"backend": "ccx", "frame": fe_info(fe, 1)}, [], modes)
    res.save(workdir / "results.npz")
    log("Done")
    return res
