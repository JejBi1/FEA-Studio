"""2D calculation as a simplification of the 3D study by a section: section (OCP) → 2D mesh (Gmsh, separate process) → conversion of the constraints and loads
→ ccx (CAX6 / CPS6 / CPE6) → results + check of the equilibrium.

    res = run_2d(study, workdir, settings=TwoDSettings(kind="axisym", axis=[0, 0, 1], xdir=[1, 1, 0]))

Equilibrium: reaction = RF − F in the constrained DOF (RF in the .frd = K·u in all nodes, for CAX over the whole circumference – verified
scratch/twod/exp1.py, exp2.py). Planar problems: forces in the plane + moment perpendicular to it; axisym only the axial component (radial forces around the circumference
cancel each other). Imbalance > config.EQUILIBRIUM_TOL = error, the result is not returned.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .. import config
from .convert import AXIS_SET, Model2D, build_model
from .mesh2d import generate_mesh2d
from .model import KIND_LABEL, TwoDError, TwoDSettings, make_frame
from .results import Results2D


def _settings(study, settings) -> TwoDSettings:
    s = settings if settings is not None else getattr(study, "twod", None)
    if s is None:
        raise TwoDError("The study has no 2D analysis settings")
    return TwoDSettings.from_dict(s) if isinstance(s, dict) else s


def _local_refinement(study, chosen, sections, fr, warnings) -> list[dict]:
    """Local refinement of the study (Study.mesh.local) converted to 2D: selection → points of the edges of the section on its faces, sphere →
    a circle with the centre in the plane of the section.
    """
    from .section import FaceSet, plane_edges, selection_faces
    out = []
    locs = list(getattr(study.mesh, "local", []) or [])
    names = [l["selection"] for l in locs if l.get("selection")]
    faces = selection_faces(study, chosen, names) if names else {}
    for loc in locs:
        if loc.get("center") is not None:
            out.append({"center": fr.to2d(np.asarray(loc["center"], float)).tolist(), "radius": float(loc["radius"]),
                        "size": float(loc["size"])})
            continue
        name = loc.get("selection")
        if not name:
            continue
        pts = []
        for sp in sections:
            if not sp.brep:
                continue
            edges = plane_edges(sp.brep)
            ext = max(float(np.ptp(np.vstack(edges), axis=0).max()), 1.0) if edges else 1.0
            fs = FaceSet(faces.get(name, []), fr, max(1e-5 * ext, 1e-5))
            for e in edges:
                if fs.contains_all(fr.to3d(e)):
                    pts.append(e[3].tolist())
        if pts:
            out.append({"selection": name, "points": pts, "size": float(loc["size"])})
        else:
            warnings.append(f"Local refinement for selection '{name}': the selection has no edge in the section")
    return out


def _equilibrium(model: Model2D, F: np.ndarray, R: np.ndarray) -> dict:
    x = model.mesh.nodes - model.mesh.nodes.mean(axis=0)
    f_scale = max(np.linalg.norm(F, axis=1).sum(), np.linalg.norm(R, axis=1).sum(), 1e-30)
    if model.frame.axisym:
        fres = np.array([0.0, F[:, 1].sum() + R[:, 1].sum()])
        return {"force_residual": fres.tolist(), "moment_residual": 0.0,
                "reaction_total": [0.0, float(R[:, 1].sum())], "imbalance": float(abs(fres[1]) / f_scale)}
    fres = F.sum(axis=0) + R.sum(axis=0)
    cz = lambda A: x[:, 0] * A[:, 1] - x[:, 1] * A[:, 0]       # noqa: E731
    mres = float(cz(F).sum() + cz(R).sum())
    m_scale = max(np.abs(cz(F)).sum(), np.abs(cz(R)).sum(), 1e-30)
    return {"force_residual": fres.tolist(), "moment_residual": mres, "reaction_total": R.sum(axis=0).tolist(),
            "imbalance": float(max(np.linalg.norm(fres) / f_scale, abs(mres) / m_scale))}


def _owners(model: Model2D) -> np.ndarray:
    """Owner of the constrained DOF (index of the constraint; a node in several constraints belongs to the first), −1 = free."""
    own = np.full((len(model.mesh.nodes), 2), -1, dtype=np.int64)
    for k, bc in enumerate(model.bcs):
        for dof in bc.dofs:
            col = own[bc.nodes, dof - 1]
            free = col < 0
            own[bc.nodes[free], dof - 1] = k
    return own


def run_2d(study, workdir: str | Path, parts=None, study_dir=None, on_log=None, cancel=None,
           settings: TwoDSettings | dict | None = None, timeout: float | None = None) -> Results2D:
    """2D calculation of the study. settings = TwoDSettings (None = study.twod). parts = parts of OCP (None = load the STEP of the study)."""
    from ..ccx.frd import read_frd
    from ..ccx.runner import run_ccx
    from .inp import write_inp
    from .section import section_parts
    t0 = time.time()
    log = on_log or (lambda m: None)
    s = _settings(study, settings)
    fr = make_frame(s)
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    if parts is None:
        from ..geometry import legacy_step_names, read_step
        log("Loading geometry…")
        parts = read_step(study.geometry_path(study_dir), legacy_names=legacy_step_names(study))
    names = [p.name for p in parts]
    for n in s.parts:
        if n not in names:
            raise TwoDError(f"2D: part '{n}' is not in the geometry")
    excl = study.excluded_parts()
    chosen = [p for p in parts if p.name not in excl and (not s.parts or p.name in s.parts)]
    if not chosen:
        raise TwoDError("2D: no part to analyze")
    log(f"2D section ({KIND_LABEL[fr.kind]})…")
    sections = section_parts(chosen, fr, wd / "mesh", s.thickness)
    for sp in sections:
        if not sp.brep:
            warnings.append(f"Part '{sp.name}' does not intersect the section plane – missing in 2D")
    cut = [sp for sp in sections if sp.brep]
    if not cut:
        raise TwoDError("2D: the section plane does not intersect any part (check the point, axis and xdir direction)")
    used = [p for p in chosen if p.name in {sp.name for sp in cut}]
    for sp in cut:
        try:
            study.material_of(sp.name)
        except KeyError:
            raise TwoDError(f"Part '{sp.name}' has no material") from None
    ms = study.mesh
    mset = {"size": float(s.size or ms.size), "min_size": ms.min_size, "curvature_points": ms.curvature_points,
            "order": 2, "local": _local_refinement(study, used, cut, fr, warnings)}
    log(f"2D mesh (element size {mset['size']:g} mm)…")
    mesh = generate_mesh2d(cut, mset, wd / "mesh", cancel=cancel, timeout=timeout, axis_clamp=fr.axisym)
    warnings += mesh.report.get("warnings", [])
    log(f"2D mesh: {len(mesh.nodes)} nodes, {len(mesh.tri)} tri6 elements")
    model = build_model(study, used, mesh, fr, s.thickness, warnings)
    inp = write_inp(model, wd / "job2d.inp", study.solver.solver)
    log(f"ccx: {inp.name}")
    exe = Path(study.solver.exe) if study.solver.exe else None
    run = run_ccx(wd, "job2d", exe=exe, cancel=cancel, timeout=timeout)
    if run.errors or not run.finished:
        tail = (wd / "job2d.log").read_text(encoding="utf-8", errors="replace")[-2500:]
        raise TwoDError("ccx calculation (2D) failed: " + ("; ".join(dict.fromkeys(run.errors[:5])) or tail))
    frd = read_frd(wd / "job2d.frd")
    n = len(mesh.nodes)
    own = _owners(model)
    steps = []
    for si, st in enumerate(model.steps):
        def nodal(name, nc):
            b = frd.get(name, si + 1)
            out = np.zeros((n, nc))
            mask = b.node_ids <= n
            out[b.node_ids[mask] - 1] = b.values[mask, :nc]
            return out
        U, RF, S = nodal("DISP", 2), nodal("FORC", 2), nodal("STRESS", 6)
        Rn = np.where(own >= 0, RF - st.F, 0.0)
        reactions: dict[str, np.ndarray] = {}
        for k, bc in enumerate(model.bcs):
            if bc.selection == AXIS_SET:
                continue
            r = np.where(own == k, Rn, 0.0).sum(axis=0)
            reactions[bc.selection] = reactions.get(bc.selection, np.zeros(2)) + r
        eq = _equilibrium(model, st.F, Rn)
        if eq["imbalance"] > config.EQUILIBRIUM_TOL:
            raise TwoDError(f"Case '{st.name}': imbalance between reactions and loads {eq['imbalance']:.2e} > {config.EQUILIBRIUM_TOL:g} – the 2D result is wrong (supports / mechanism?)")
        steps.append({"name": st.name, "U": U, "S": S, "RF": RF, "reactions": reactions, "equilibrium": eq})
    info = dict(model.info)
    info["sections"] = [{"part": sp.name, "area_mm2": sp.area, "centroid": sp.centroid, "volume_3d": sp.volume,
                         "volume_2d_model": sp.model_volume,
                         "ratio_3d_to_2d": sp.volume / sp.model_volume if sp.model_volume else None} for sp in cut]
    info["mesh"] = {k: mesh.report.get(k) for k in ("nodes", "elements", "min_scaled_jacobian", "flipped")}
    info["time_s"] = time.time() - t0
    frame = {"origin": fr.origin.tolist(), "e1": fr.e1.tolist(), "e2": fr.e2.tolist(), "n": fr.n.tolist()}
    res = Results2D(fr.kind, mesh.nodes, mesh.tri, mesh.tri_part, mesh.part_names, steps,
                    {k: mesh.seg[v] for k, v in model.edges.items()}, list(dict.fromkeys(warnings)), frame,
                    float(s.thickness), info, mesh)
    log(f"2D analysis done ({info['time_s']:.1f} s)")
    return res
