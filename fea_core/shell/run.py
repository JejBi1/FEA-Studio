"""run_shell: shell calculation of the 3D study (plates / sheets / faces → S6 in ccx).

Procedure: parts → SheetInfo (plate from a body / sheet / planar part) → extension towards the neighbouring plates (T joint) →
Gmsh mesh (separate process) → conversion of the selections, constraints and loads → .inp → ccx → expanded nodes → results →
check of the equilibrium (of forces and moments; imbalance > config.EQUILIBRIUM_TOL = error).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .. import config
from .model import ShellError, ShellSettings

SUPPORTED_LOADS = {"force", "pressure", "gravity", "rotation"}
SUPPORTED_CONSTRAINTS = {"fixed", "displacement"}


def _settings(study, settings) -> ShellSettings:
    s = settings if settings is not None else getattr(study, "shell", None)
    if s is None:
        s = ShellSettings(enabled=True)
    elif isinstance(s, dict):
        s = ShellSettings.from_dict(s)
    errs = s.check()
    if errs:
        raise ShellError("; ".join(errs))
    return s


def check_study(study, settings: ShellSettings | dict | None = None) -> list[str]:
    """Readiness check of the study for the shell model (without geometry and mesh) → errors in Czech."""
    errs = []
    try:
        s = _settings(study, settings)
    except ShellError as e:
        return [str(e)]
    del s
    if study.active_bolts() or study.active_welds():
        errs.append("Bolts and welds are not supported in the shell model (turn them off or run a 3D analysis)")
    if not study.constraints:
        errs.append("The shell model has no support")
    for c in study.constraints:
        kind = getattr(c, "kind", "")
        if kind not in SUPPORTED_CONSTRAINTS:
            errs.append(f"Support type '{kind}' is not supported in the shell model (only fixed support and prescribed displacement)")
        elif c.selection not in study.selections:
            errs.append(f"Support: selection '{c.selection}' does not exist")
    if not study.load_cases:
        errs.append("The study has no load case")
    for lc in study.load_cases:
        for ld in lc.loads:
            kind = getattr(ld, "kind", "")
            if kind not in SUPPORTED_LOADS:
                errs.append(f"Case '{lc.name}': load type '{kind}' is not supported in the shell model ({', '.join(sorted(SUPPORTED_LOADS))} only)")
                continue
            if getattr(ld, "csys", None):
                errs.append(f"Case '{lc.name}': loads in a local coordinate system are not supported in the shell model")
            sel = getattr(ld, "selection", None)
            if sel is not None and sel not in study.selections:
                errs.append(f"Case '{lc.name}': selection '{sel}' does not exist")
    if getattr(study.thermal, "cases", None):
        errs.append("Thermal cases are not supported in the shell model")
    return list(dict.fromkeys(errs))


def prepare_sheets(study, parts, settings: ShellSettings, geometry_path=None, log=print):
    """SheetInfo of the parts of the shell model + the parts (name → Part) + warnings."""
    from .geometry import detect_sheet, read_step_surfaces, surface_sheet, connect_sheets
    warnings: list[str] = []
    ignored = set(getattr(study, "ignored_parts", []) or [])
    solids = {p.name: p for p in parts if p.name not in ignored}
    wanted = dict(settings.parts)
    surfaces = {}
    missing = [n for n in wanted if n not in solids and n not in ignored]
    if missing:
        if geometry_path is None or not Path(geometry_path).exists():
            raise ShellError(f"Parts {missing} are not among the bodies and no STEP with surfaces is available")
        surfaces = {p.name: p for p in read_step_surfaces(geometry_path)}
        lost = [n for n in missing if n not in surfaces]
        if lost:
            raise ShellError(f"Parts {lost} are not in the geometry (neither bodies nor surfaces)")
    names = list(wanted) if wanted else list(solids)
    if not wanted:
        names = list(solids)
    else:
        rest = [n for n in solids if n not in wanted]
        if rest:
            raise ShellError(f"Bodies {rest} are not in the shell model – add them to ShellSettings.parts or exclude them from the analysis (ignored_parts)")
    sheets, all_parts = [], {}
    for n in names:
        if n in solids:
            info, why = detect_sheet(solids[n])
            if info is None:
                raise ShellError(f"Part '{n}' cannot be converted to a shell: {why}")
            t = wanted.get(n)
            if t is not None and abs(t - info.thickness) > 1e-3 * info.thickness:
                warnings.append(f"Part '{n}': the specified thickness {t:g} mm differs from the body thickness {info.thickness:.6g} mm – the specified value is used")
                info.thickness = float(t)
            warnings += info.warnings
            all_parts[n] = solids[n]
        else:
            info = surface_sheet(surfaces[n], wanted.get(n), settings.offset)
            all_parts[n] = surfaces[n]
        sheets.append(info)
    if not sheets:
        raise ShellError("The shell model has no part")
    warnings += connect_sheets(sheets, all_parts)
    return sheets, all_parts, warnings


def _components(mesh) -> np.ndarray:
    """Number of the connected part of the mesh for each node (connection through the elements)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    N = len(mesh.nodes)
    a = mesh.tri[:, [0, 1, 2, 3, 4, 5]]
    rows = np.repeat(a[:, 0], 5)
    cols = a[:, 1:].ravel()
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    return connected_components(g, directed=False)[1]


def build_model(study, mesh, sheets, parts):
    """Constraints and loads on the shell mesh → (bcs, frames, owner, steps, sel_nodes, materials)."""
    from .loads import Mapper, constraint_dofs, force_loads, pressure_loads, gravity_loads, rotation_loads
    mapper = Mapper(mesh, {s.part: s for s in sheets}, parts)
    targets: dict = {}

    def target(name):
        if name not in targets:
            if name not in study.selections:
                raise ShellError(f"Selection '{name}' does not exist in the study")
            targets[name] = mapper.target(name, study.selections[name])
        return targets[name]
    nnorm = mesh.node_normals()
    bcs: dict[int, dict[int, float]] = {}
    frames: dict[int, np.ndarray] = {}
    owner: dict[int, str] = {}
    sel_nodes: dict[str, np.ndarray] = {}
    for c in study.constraints:
        kind = getattr(c, "kind", "")
        if kind not in SUPPORTED_CONSTRAINTS:
            raise ShellError(f"Support type '{kind}' is not supported in the shell model (only fixed support and prescribed displacement)")
        tg = target(c.selection)
        dofs, fr = constraint_dofs(c, tg, mesh, nnorm)
        sel_nodes[c.selection] = np.asarray(sorted(dofs), np.int64)
        for v, T in fr.items():
            if v in bcs or v in frames:
                raise ShellError(f"Node on the boundary of selections '{owner.get(v)}' and '{c.selection}': a skewed side-face support in local axes cannot be combined with another support at the same node")
            frames[v] = T
        for v, d in dofs.items():
            if v in frames and v not in fr:
                raise ShellError(f"Node on the boundary of selections '{owner.get(v)}' and '{c.selection}': a skewed side-face support in local axes cannot be combined with another support at the same node")
            cur = bcs.setdefault(v, {})
            for k, val in d.items():
                if k in cur and abs(cur[k] - val) > 1e-12 * max(1.0, abs(val)):
                    raise ShellError(f"The node on the boundary of selections '{owner.get(v)}' and '{c.selection}' has two different prescribed values for DOF {k}")
                cur[k] = val
            owner.setdefault(v, c.selection)
    if not bcs:
        raise ShellError("The shell model has no support")
    # nodes with a constrained drilling rotation without the other rotations → add them (ccx: drilling only with all rotations)
    for v, d in bcs.items():
        rot = [k for k in (4, 5, 6) if k in d]
        if rot and v not in frames:
            n = nnorm[v]
            for k in rot:
                e = np.zeros(3)
                e[k - 4] = 1.0
                if abs(float(e @ n)) > 1 - 1e-6 and len(rot) < 3:
                    del d[k]                         # the rotation about the normal alone is not constrained (ccx removes it anyway)
    comp = _components(mesh)
    fixed_comp = set(comp[list(bcs)].tolist())
    free = sorted(set(comp.tolist()) - fixed_comp)
    if free:
        names = sorted({mesh.part_names[int(mesh.tri_part[e])] for e in range(len(mesh.tri))
                        if comp[mesh.tri[e, 0]] in free})
        raise ShellError(f"Free parts without support in the shell model: {names} (not connected to the rest – overlap / gap between plates?)")
    rho = np.zeros(len(mesh.part_names))
    mats = {}
    for i, n in enumerate(mesh.part_names):
        m = study.material_of(n)
        if getattr(m, "is_print", False) or getattr(m, "is_ortho", False):
            raise ShellError(f"Part '{n}': printed / orthotropic material is not supported in the shell model")
        mats[i] = m
        rho[i] = float(m.rho or 0.0)
    rho_t = (rho * mesh.thickness)[mesh.tri_part]
    steps = []
    for lc in study.load_cases:
        F = np.zeros((len(mesh.nodes), 3))
        for ld in lc.loads:
            kind = getattr(ld, "kind", "")
            if kind == "force":
                F += force_loads(ld, target(ld.selection), mesh)
            elif kind == "pressure":
                F += pressure_loads(ld, target(ld.selection), mesh)
            elif kind == "gravity":
                if not np.any(rho_t > 0):
                    raise ShellError("Gravity: materials have no density")
                F += gravity_loads(ld, mesh, rho_t)
            elif kind == "rotation":
                F += rotation_loads(ld, mesh, rho_t)
            else:
                raise ShellError(f"Load type '{kind}' is not supported in the shell model ({', '.join(sorted(SUPPORTED_LOADS))} only)")
        steps.append((lc.name, F))
    if not steps:
        raise ShellError("The study has no load case")
    return bcs, frames, owner, steps, sel_nodes, mats


def _equilibrium(x_app, F, x_exp_by_node, RF_by_node, owner, ref) -> tuple[dict, dict, np.ndarray]:
    """Reactions by the selections of the constraints [F, M about ref] and the equilibrium. x_exp_by_node / RF_by_node: node → (k, 3)."""
    Fa = F.sum(axis=0)
    Ma = np.cross(x_app - ref, F).sum(axis=0)
    reac: dict[str, np.ndarray] = {}
    rnorm, mnorm = 0.0, 0.0
    for v, sel in owner.items():
        xr, rf = x_exp_by_node[v], RF_by_node[v]
        R = rf.sum(axis=0) - F[v]
        M = np.cross(xr - ref, rf).sum(axis=0) - np.cross(x_app[v] - ref, F[v])
        r = reac.setdefault(sel, np.zeros(6))
        r[:3] += R
        r[3:] += M
        rnorm += float(np.linalg.norm(R))
        mnorm += float(np.linalg.norm(M))
    Rt = sum((r for r in reac.values()), np.zeros(6))
    fres = Fa + Rt[:3]
    mres = Ma + Rt[3:]
    size = float(np.linalg.norm(x_app.max(axis=0) - x_app.min(axis=0))) or 1.0
    f_scale = max(float(np.linalg.norm(F, axis=1).sum()), rnorm, 1e-30)
    m_scale = max(float(np.linalg.norm(np.cross(x_app - ref, F), axis=1).sum()), mnorm, f_scale * size * 1e-3,
                  1e-30)
    imb = max(float(np.linalg.norm(fres)) / f_scale, float(np.linalg.norm(mres)) / m_scale)
    eq = {"force_residual": fres.tolist(), "moment_residual": mres.tolist(), "reaction_total": Rt[:3].tolist(),
          "reaction_moment_total": Rt[3:].tolist(), "applied_force": Fa.tolist(), "applied_moment": Ma.tolist(),
          "imbalance": imb}
    return reac, eq, np.r_[Fa, Ma]


def run_shell(study, workdir, parts=None, study_dir=None, on_log=None, cancel=None,
              settings: ShellSettings | dict | None = None, mesh=None):
    """Shell calculation of the study → ShellResults (also saves workdir/shell_results.npz + .json).

    parts = bodies (geometry.read_step) – None = load from the STEP of the study; settings = ShellSettings (None =
    Study.shell, otherwise all parts automatically); mesh = a finished ShellMesh (not meshed again).
    """
    from ..ccx.frd import read_frd
    from ..ccx.runner import run_ccx
    from .inp import write_shell_inp
    from .mesh import generate_shell_mesh
    from .results import Expansion, ShellResults, ShellStep, _gather, _node_mean
    log = on_log or (lambda s: None)
    t0 = time.time()
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    s = _settings(study, settings)
    errs = check_study(study, s)
    if errs:
        raise ShellError("; ".join(errs))
    geo = study.geometry_path(study_dir) if getattr(study, "geometry", "") else None
    if parts is None:
        from ..geometry import legacy_step_names, read_step
        if geo is None or not Path(geo).exists():
            raise ShellError(f"Study geometry not found: {geo}")
        parts = read_step(geo, legacy_names=legacy_step_names(study))
    log("Shells: detecting plates and mid surfaces")
    sheets, all_parts, warnings = prepare_sheets(study, parts, s, geo, log)
    for sh in sheets:
        log(f"  {sh.part}: {sh.kind}, t = {sh.thickness:.6g} mm, mid-surface area {sh.mid_area:.6g} mm²"
            + (f", extension {len(sh.extensions)}" if sh.extensions else ""))
    if cancel is not None and cancel.is_set():
        raise ShellError("Analysis canceled by user")
    size = float(s.size or study.mesh.size)
    if mesh is None:
        log(f"Shells: mesh (element size {size:g} mm)")
        mesh = generate_shell_mesh(sheets, size, wd, getattr(study.mesh, "curvature_points", 12),
                                   getattr(study.mesh, "threads", 0), cancel=cancel)
        warnings += mesh.report.get("warnings", [])
    log(f"  {len(mesh.nodes)} nodes, {len(mesh.tri)} S6 elements, min. SICN {mesh.report.get('min_sicn', 1):.3f}")
    tmax = float(mesh.thickness.max())
    hmean = float(np.mean(np.linalg.norm(mesh.nodes[mesh.tri[:, 0]] - mesh.nodes[mesh.tri[:, 1]], axis=1)))
    if hmean < 0.2 * tmax:
        warnings.append(f"Shell elements ({hmean:.3g} mm) are smaller than one fifth of the thickness ({tmax:.3g} mm) – the expanded C3D15 elements are elongated through the thickness, consider a 3D model")
    bcs, frames, owner, steps, sel_nodes, mats = build_model(study, mesh, sheets, all_parts)
    write_shell_inp(wd / "shell.inp", mesh, mats, bcs, steps, study.name, frames,
                    getattr(study.solver, "solver", "PARDISO") or "PARDISO")
    log(f"Shells: ccx ({len(steps)} cases)")
    run = run_ccx(wd, "shell", exe=getattr(study.solver, "exe", None) or None, on_line=None, cancel=cancel)
    if run.errors or not run.finished:
        tail = (wd / "shell.log").read_text(encoding="utf-8", errors="replace")[-2500:]
        raise ShellError("CalculiX (shells) failed: " + "; ".join(run.errors[:3]) + "\n" + tail)
    frd = read_frd(wd / "shell.frd")
    exp = Expansion.from_frd(mesh, wd / "shell.frd")
    xyz_b = _gather(exp.bot, frd.coords, frd.node_ids)
    xyz_t = _gather(exp.top, frd.coords, frd.node_ids)
    n_el = mesh.normals()
    side = np.einsum("mkc,mc->mk", xyz_t - xyz_b, n_el)
    if np.any(side <= 0):
        raise ShellError("ccx expanded nodes do not match the element normal (top/bottom surface)")
    nn = mesh.node_normals()
    off_t = (mesh.offset * mesh.thickness)[mesh.tri_part]
    off_node = np.zeros(len(mesh.nodes))
    for k in range(6):
        off_node[mesh.tri[:, k]] = off_t
    x_app = mesh.nodes - off_node[:, None] * nn
    ref = mesh.nodes.mean(axis=0)
    # expanded nodes belonging to a shell node (also the nodes doubled in a hinge – "knot") → reaction
    corner_mid = {}
    els3c = None
    from .results import read_frd_elements
    els3c = read_frd_elements(wd / "shell.frd")
    for e in range(len(mesh.tri)):
        c = els3c[e + 1]
        for k in range(3):
            corner_mid.setdefault(int(mesh.tri[e, k]), set()).update((int(c[k]), int(c[k + 3]), int(c[9 + k])))
        for k in range(3):
            corner_mid.setdefault(int(mesh.tri[e, 3 + k]), set()).update((int(c[6 + k]), int(c[12 + k])))
    lut = np.full(int(frd.node_ids.max()) + 1, -1, np.int64)
    lut[frd.node_ids] = np.arange(len(frd.node_ids))
    out_steps = []
    for k, (name, F) in enumerate(steps):
        sn = k + 1
        U = frd.get("DISP", sn)
        S = frd.get("STRESS", sn)
        RF = frd.get("FORC", sn)
        ub = _gather(exp.bot, U.values[:, :3], U.node_ids)
        ut = _gather(exp.top, U.values[:, :3], U.node_ids)
        sb = _gather(exp.bot, S.values[:, :6], S.node_ids)
        stp = _gather(exp.top, S.values[:, :6], S.node_ids)
        o = (mesh.offset[mesh.tri_part])[:, None, None]
        u_ref = 0.5 * (ub + ut) + o * (ut - ub)
        Un = _node_mean(mesh, u_ref)
        St, Sb = _node_mean(mesh, stp), _node_mean(mesh, sb)
        rf_lut = np.zeros((len(lut), 3))
        rf_lut[RF.node_ids] = RF.values[:, :3]
        xe, re = {}, {}
        for v in owner:
            ids = np.fromiter(corner_mid[v], np.int64)
            xe[v] = frd.coords[lut[ids]]
            re[v] = rf_lut[ids]
        reac, eq, applied = _equilibrium(x_app, F, xe, re, owner, ref)
        if eq["imbalance"] > config.EQUILIBRIUM_TOL:
            raise ShellError(f"Case '{name}': reaction/load imbalance {eq['imbalance']:.3g} (> {config.EQUILIBRIUM_TOL:g}) – the shell model result is not reliable")
        out_steps.append(ShellStep(name, Un, St, 0.5 * (St + Sb), Sb, reac, eq, applied))
    res = ShellResults(mesh, out_steps, ref, sel_nodes, list(dict.fromkeys(warnings)),
                       {"time_s": time.time() - t0, "size": size, "ccx_warnings": run.warnings[:20],
                        "sheets": [{"part": sh.part, "kind": sh.kind, "thickness": sh.thickness,
                                    "mid_area": sh.mid_area, "extensions": len(sh.extensions)} for sh in sheets]})
    res.save(wd / "shell_results")
    log(f"Shells: done in {time.time() - t0:.1f} s")
    return res
