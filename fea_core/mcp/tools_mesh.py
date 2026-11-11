"""Tools: meshing (Gmsh tet10) and the check of readiness (materials, constraints, rigid motions)."""
from __future__ import annotations

import math
import time
from dataclasses import asdict

import numpy as np

from .common import out, resolve_selection, run_with_timeout
from .session import FeaError, Session

DOF_PER_H3 = 26.0        # tet10: ≈ 8.7 nodes per h³ of volume (V1: 22 226 nodes, 40 000 mm³, h = 2.5)


def estimate_dof(sess: Session, size: float) -> int:
    excl = sess.study.excluded_parts()
    vol = sum(p.volume() for p in sess.parts if p.name not in excl)
    return int(DOF_PER_H3 * vol / max(size, 1e-9) ** 3)


def auto_size(sess: Session) -> float:
    """Default element size: at most 5 mm and at least 3 elements across the thinnest dimension of the envelope of the part."""
    from ..geometry import bbox
    excl = sess.study.excluded_parts()
    dims = [float(np.min(bbox(p.shape)[3:] - bbox(p.shape)[:3])) for p in sess.parts if p.name not in excl]
    h = min([5.0] + [d / 3.0 for d in dims if d > 0])
    return float(f"{h:.2g}")


# ===================================================================== rigid motions

def _face_points(sess: Session, sel, nmax: int = 60) -> np.ndarray:
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from ..selectors import resolve
    for pi in range(len(sess.parts)):
        sess.face_props(pi)
    r = resolve(sel, sess.parts, sess._face_props)
    if r is None:
        return np.asarray([sel.centroid], float)
    face = sess.faces_of(r[0])[r[1]]
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None:
        return np.asarray([sel.point, sel.centroid], float)
    trsf = loc.Transformation()
    n = tri.NbNodes()
    idx = np.unique(np.linspace(1, n, min(n, nmax)).astype(int))
    pts = []
    for i in idx:
        p = tri.Node(int(i)).Transformed(trsf)
        pts.append((p.X(), p.Y(), p.Z()))
    return np.asarray(pts, float)


def _describe_modes(rows: np.ndarray, center: np.ndarray) -> list[str]:
    if len(rows) == 0:
        return ["all 3 displacements and 3 rotations (no support)"]
    _, sv, vt = np.linalg.svd(rows, full_matrices=True)
    tol = 1e-8 * max(sv.max(), 1e-30)
    rank = int((sv > tol).sum())
    modes = []
    for v in vt[rank:]:
        a, w = v[:3], v[3:]
        if np.linalg.norm(w) < 1e-6 * max(np.linalg.norm(a), 1e-30):
            a = a / np.linalg.norm(a)
            modes.append(f"displacement in direction ({', '.join((f'{x:.3g}' for x in a))})")
        else:
            p = np.cross(w, a) / (w @ w) + center
            wn = w / np.linalg.norm(w)
            modes.append(f"rotation about the axis ({', '.join(f'{x:.3g}' for x in wn)}) through the point "
                         f"({', '.join(f'{x:.4g}' for x in p)}) mm")
    return modes


def _cyl_dirs(c, P: np.ndarray, d: np.ndarray, c0: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cylindrical constraint: blocks (points, directions (n, 3)) of the constrained components in the axes of the cylinder."""
    from ..cylinder import radial_frames
    er, et = radial_frames(P, c0, d)
    dirs = {"r": er, "t": et, "a": np.broadcast_to(d, er.shape)}
    return [(P, dirs[k]) for k, on in (("r", c.radial), ("t", c.tangential), ("a", c.axial)) if on]


def _cad_axis(sess: Session, faces) -> tuple[np.ndarray, np.ndarray] | None:
    """Axis of the cylinder (point, direction) from the CAD geometry of the first cylindrical face of faces (cache face_props); the check of rigid
    motions calls it for each face – the selection of a cylindrical constraint may contain several non-coaxial holes.
    """
    from ..selectors import resolve
    for pi in range(len(sess.parts)):
        sess.face_props(pi)
    for f in faces:
        r = resolve(f, sess.parts, sess._face_props)
        if r is None:
            continue
        d = sess._face_props[r[0]][r[1]]
        if "axis_dir" in d:
            return np.asarray(d["axis_point"], float), np.asarray(d["axis_dir"], float)
    return None


def rigid_body_check(sess: Session) -> dict:
    """Does the constraint block all 6 rigid motions? Rows [e, x × e] for the constrained directions e → rank 6.

    With the mesh exactly (DOF masks of the nodes, solve.constraint_masks; a cylindrical constraint in the directions of the axes of the cylinder
    z cylinder.support_directions), without a mesh from the points of the triangulation of the faces (axis of the cylinder by fitting).
    The connection of the parts (free bodies) is checked only after meshing (solve.floating_parts).
    Remote constraint: constrained components at the point (displacement in the direction e = [e, (p − c) × e], rotation = [0, e]) –
    the nodes of the face are not counted (the face only "converges" at the point, remote.rigid_rows).
    """
    from ..study import Fixed, Displacement, CylindricalSupport, RemoteSupport, ElasticSupport
    from ..remote import rigid_rows
    st = sess.study
    mesh = sess.valid_mesh()
    approx = False
    fallback = ""
    blocks: list[tuple[np.ndarray, np.ndarray]] = []       # (points (n, 3), directions (n, 3))
    rng = np.random.default_rng(0)
    cyl = {c.selection for c in st.constraints if isinstance(c, CylindricalSupport)}
    remote = [c for c in st.constraints if isinstance(c, RemoteSupport)]
    elastic = [c for c in st.constraints if isinstance(c, ElasticSupport)]
    skip = cyl | {c.selection for c in remote} | {c.selection for c in elastic}
    if mesh is not None:
        from ..solve import constraint_masks
        from ..cylinder import support_directions
        try:
            msk = np.zeros((len(mesh.nodes), 3), bool)
            for k, m in constraint_masks(st, mesh).items():
                if k not in skip:
                    msk |= m
            idx = np.nonzero(msk.any(axis=1))[0]
            if len(idx) > 3000:
                idx = idx[rng.choice(len(idx), 3000, replace=False)]
            for d in range(3):
                sel = idx[msk[idx, d]]
                blocks.append((mesh.nodes[sel], np.broadcast_to(np.eye(3)[d], (len(sel), 3))))
            from ..springs import table as spring_table
            for t in spring_table(st, mesh):           # elastic support: normal, or all directions
                pick = np.arange(len(t["idx"]))
                if len(pick) > 1000:
                    pick = rng.choice(len(pick), 1000, replace=False)
                pts = mesh.nodes[t["idx"][pick]]
                dirs = [t["normal"][pick]] if t["direction"] == "normal" else \
                    [np.broadcast_to(np.eye(3)[d], pts.shape) for d in range(3)]
                blocks += [(pts, D) for D in dirs]
            for idx, D in support_directions(st, mesh).values():
                if len(idx) > 1000:
                    pick = rng.choice(len(idx), 1000, replace=False)
                    idx, D = idx[pick], D[pick]
                for k in range(D.shape[1]):
                    blocks.append((mesh.nodes[idx], D[:, k]))
            method = "mesh (support nodes)"
        except Exception as e:  # noqa: BLE001 – new kind of constraint without a mask / error of the axis → geometric check
            mesh, blocks = None, []
            fallback = f"the check on the mesh failed ({type(e).__name__}: {e}) – geometric check used instead"
    if mesh is None:
        method = "geometry (face points of supports)"
        if fallback:
            method += f"; {fallback}"
            approx = True
        for c in st.constraints:
            if c.selection not in st.selections or isinstance(c, RemoteSupport):
                continue
            P = np.vstack([_face_points(sess, f) for f in st.selections[c.selection].faces])
            if isinstance(c, CylindricalSupport):
                # each face with its own axis from the CAD (the selection may contain several non-coaxial holes)
                cyl_blocks, ok = [], True
                for f in st.selections[c.selection].faces:
                    ax = _cad_axis(sess, [f])
                    if ax is None:
                        ok = False
                        break
                    cyl_blocks += _cyl_dirs(c, _face_points(sess, f), ax[1], ax[0])
                if ok and cyl_blocks:
                    blocks += cyl_blocks
                    continue
                approx = True                     # the axis from the CAD unknown → approximately as clamping
                comp = [0, 1, 2]
            elif isinstance(c, Fixed):
                comp = [0, 1, 2]
            elif isinstance(c, Displacement):
                comp = [d for d, v in enumerate((c.ux, c.uy, c.uz)) if v is not None]
            elif isinstance(c, ElasticSupport) and c.direction == "all":
                comp = [0, 1, 2]
            elif isinstance(c, ElasticSupport) and all(f.type == "plane" for f in st.selections[c.selection].faces):
                for f in st.selections[c.selection].faces:            # only perpendicular to a planar face
                    Pf = _face_points(sess, f)
                    nrm = np.asarray(f.normal, float)
                    blocks.append((Pf, np.broadcast_to(nrm / max(np.linalg.norm(nrm), 1e-30), Pf.shape)))
                continue
            else:
                approx = True
                comp = [0, 1, 2]
            for d in comp:
                blocks.append((P, np.broadcast_to(np.eye(3)[d], P.shape)))
    blocks = [(P, E) for P, E in blocks if len(P)]
    if blocks:
        P = np.vstack([b[0] for b in blocks])
        E = np.vstack([b[1] for b in blocks])
        center = P.mean(axis=0)
        rows = np.hstack([E, np.cross(P - center, E)])
    else:
        center = np.mean([c.point for c in remote], axis=0) if remote else np.zeros(3)
        rows = np.zeros((0, 6))
    if remote:
        rows = np.vstack([rows] + [rigid_rows(c, center) for c in remote])
    frows, fcenter = frame_rows(st, center if blocks or remote else None)
    if len(frows):
        if not blocks and not remote:
            center = fcenter
        rows = np.vstack([rows, frows])
        method += " + member supports"
    modes = _describe_modes(rows, center)
    d = {"method": method, "free_rigid_modes": modes, "ok": not modes}
    if fallback:
        d["note"] = f"Rigid-body check: {fallback} – the result is approximate."
    elif approx:
        d["note"] = "The study contains a support that the geometric check cannot evaluate exactly – the result is approximate."
    return d


def frame_rows(st, center=None) -> tuple[np.ndarray, np.ndarray]:
    """Rows [e, (p − c) × e] of the displacements and [0, e] of the rotations of the constrained (also elastic) components of the support of the bar system.
    Internal mechanisms (chains of hinges) cannot be detected this way – only the rigid motions of the whole.
    """
    fr = st.frame
    if center is None:
        center = np.mean([n.xyz for n in fr.nodes.values()], axis=0) if fr.nodes else np.zeros(3)
    rows = []
    for sup in fr.supports:
        if sup.node not in fr.nodes:
            continue
        p = np.asarray(fr.nodes[sup.node].xyz, float)
        for d in range(6):
            k = sup.stiffness[d] if d < len(sup.stiffness) else 0.0
            if not (sup.dofs[d] or k > 0):
                continue
            e = np.eye(3)[d % 3]
            rows.append(np.r_[e, np.cross(p - center, e)] if d < 3 else np.r_[np.zeros(3), e])
    return (np.asarray(rows, float).reshape(-1, 6), np.asarray(center, float))


def readiness(sess: Session) -> dict:
    """Check of readiness. A study with heat only (thermal states, no load state) does not need constraints –
    the check of rigid motions is not done for it; instead the check of the heat outflow (with the mesh).
    """
    from ..solve import check_study
    st = sess.require_study()
    names = sess.part_names()
    excl = st.excluded_parts()
    meshed = [p for p in names if p not in excl]          # bolts and left-out parts are not meshed
    errors, warnings = check_study(st, names)
    for lc in st.load_cases:
        for l in lc.loads:
            vec = getattr(l, "vector", None)
            if l.kind == "force" and vec is not None and not np.any(vec):
                warnings.append(f"Case '{lc.name}': force '{l.selection}' is zero")
            if l.kind == "gravity":
                for p in meshed:
                    try:
                        if not st.material_of(p).rho:
                            warnings.append(f"Gravity in case '{lc.name}': the material of part '{p}' has no density")
                    except KeyError:
                        pass
    if st.load_cases:
        for p in meshed:
            try:
                if st.material_of(p).Re is None and not st.material_of(p).is_print:
                    warnings.append(f"The material of part '{p}' has no Re – the safety factor to yield will not be computed")
            except KeyError:
                pass
    rb = None
    reduced = None
    if st.twod.enabled:                                    # 2D section: the frame of the plane, the 3D check of rigid motions does not hold
        from ..twod.model import TwoDError, make_frame
        try:
            make_frame(st.twod)
        except TwoDError as e:
            errors.append(str(e))
        reduced = "2D section analysis (fea_twod_set) – the conversion of supports and loads to section edges is checked during the analysis"
    if st.shell.enabled:
        from ..shell.run import check_study as shell_check
        errors += [e for e in shell_check(st, st.shell) if e not in errors]
        reduced = "Shell model (fea_shell_set) – plate recognition and load conversion are checked during the analysis"
    if reduced is None and (st.constraints or (st.frame.active and st.frame.supports)) and st.load_cases \
            and not errors:
        rb = rigid_body_check(sess)
        contact = bool(st.active_contacts())
        if not rb["ok"]:
            msg = ("The supports do not prevent rigid-body motions (the problem would be singular): " + "; ".join(rb["free_rigid_modes"]))
            if contact:                                     # a contact (also a support in compression only) may hold a part – only a warning
                msg += (" – the study has contacts: a contact is one-sided and does not replace a support (without clamping non-convergence threatens); consider weak springs fea_contact_settings(stabilize='springs')")
                rb["note"] = "contact – a warning only"
            (warnings if rb.get("note") else errors).append(msg)
    mesh = sess.valid_mesh()
    if mesh is not None:
        if st.load_cases:
            from ..solve import floating_parts
            fl = floating_parts(st, mesh)
            if fl:
                errors.append("Free parts without a connection to a support: " + ", ".join(fl))
        if st.thermal.cases and not errors:
            from ..thermal.check import floating_parts as thermal_floating
            for case in st.thermal.cases:
                try:
                    fl = thermal_floating(st, mesh, case)
                except Exception as e:  # noqa: BLE001 – never a silent pass: the check could not be done
                    warnings.append(f"Thermal case '{case.name}': the check of heat removal could not be done "
                                    f"({type(e).__name__}: {e}) – parts without a heat sink would not be detected before the analysis")
                    fl = []
                if fl:
                    errors.append(f"Thermal case '{case.name}': parts without heat removal (no path to a prescribed temperature, convection or radiation): {', '.join(fl)}")
    size = st.mesh.size
    est = estimate_dof(sess, size) if mesh is None else 3 * len(mesh.nodes)
    if est > 2_000_000 and st.load_cases:
        warnings.append(f"Estimate {est / 1000000.0:.1f} million DOF – the analysis will be long and memory-intensive, consider a larger element size")
    d = {"ready": not errors, "errors": errors, "warnings": warnings, "rigid_body": rb,
         "mesh": ("ready" if mesh is not None else "none – fea_solve_run will create it (or fea_mesh_generate)"),
         "dof": est, "dof_is_estimate": mesh is None}
    if st.thermal.cases:
        d["thermal_cases"] = [c.name for c in st.thermal.cases]
        d["temperature_dof"] = est // 3                   # heat conduction: 1 unknown (temperature) per node
        if not st.load_cases:
            d["note"] = "Heat-conduction-only study – supports and load cases are not needed"
    if reduced:
        d["note"] = reduced
        d["mesh"] = "The 2D / shell mesh is generated during the analysis"
    if st.modal.enabled or st.buckling.enabled:
        d["eigen"] = {"modal": st.modal.to_dict() if st.modal.enabled else None,
                      "buckling": st.buckling.to_dict() if st.buckling.enabled else None}
        if st.modal.enabled and not st.load_cases:
            d["note"] = ("A study with natural frequencies only – load cases are not needed"
                         + ("" if st.constraints else "; no supports = free body (6 zero frequencies)"))
    return d


def mesh_report(sess: Session, mesh, warnings: list[str], elapsed: float) -> dict:
    rep = mesh.report
    st = sess.study
    per_part = {pn: int((mesh.tet_part == i).sum()) for i, pn in enumerate(mesh.part_names)}
    dof = 3 * len(mesh.nodes)
    backend = st.solver.backend
    if backend == "auto":                      # the same rule as the calculation (solve.choose_backend)
        from ..solve import choose_backend
        backend = ("ccx (nonlinear – contact)" if st.is_nonlinear() else f"{choose_backend(st, mesh)} (auto)")
    sj = rep.get("min_scaled_jacobian")
    d = {"nodes": len(mesh.nodes), "elements": len(mesh.tets), "element_type": mesh.element_type, "dof": dof,
         "element_size_mm": st.mesh.size, "min_scaled_jacobian": sj,
         "quality": ("good" if sj is not None and sj >= 0.3 else "weaker (min SJ < 0.3) – consider a finer mesh"
                     if sj is not None and sj > 0 else "poor"),
         "elements_per_part": per_part,
         "selections_triangles": {k: int(len(v)) for k, v in mesh.surfaces.items()},
         "time_s": elapsed, "solver_backend": backend,
         "solve_time_hint": ("~20 s / 1.4 GB for 200k DOF (ccx PARDISO), ~2 min / 7 GB for 700k DOF"),
         "warnings": warnings}
    if len(mesh.part_names) > 1:
        from ..contacts import bonded_pairs, edge_only
        d["bonded_pairs"] = [{"a": p["a"], "b": p["b"], "area_mm2": p["area"], "faces": p["faces"]}
                             for p in bonded_pairs(mesh)]
        d["bonded_note"] = "parts sharing mesh faces are rigidly connected (no contact, also thermally)"
        for a, b in edge_only(mesh):
            warnings.append(f"Parts '{a}' and '{b}' touch only along an edge/at a point – connection through a few nodes (singularity, near mechanism); modify the geometry or exclude a part")
    if st.thermal.cases and not st.load_cases:
        d["solver_backend"] = "ccx (heat conduction)"
        d["temperature_dof"] = len(mesh.nodes)
    ties = rep.get("ties") or []
    if ties or st.mesh.separate:
        d["separate_parts"] = list(st.mesh.separate)
        d["ties"] = [{k: t.get(k) for k in ("part", "slave", "master", "area_mm2", "tolerance_mm")} for t in ties]
        d["ties_note"] = ("separately meshed parts are connected to the others by a *TIE constraint (slave face = separate part); ccx only")
    balls = [loc for loc in st.mesh.local if "center" in loc]
    if balls:
        d["refinement_spheres"] = balls
    return d


def ball(center, radius, size) -> dict:
    """Verified sphere of local refinement {"center", "radius", "size"} (field of Gmsh Ball)."""
    try:
        c = [float(v) for v in center]
    except (TypeError, ValueError):
        raise FeaError("center must be [x, y, z] in mm") from None
    if len(c) != 3 or not all(np.isfinite(c)):
        raise FeaError("center must be [x, y, z] in mm")
    if not (radius and float(radius) > 0 and size and float(size) > 0):
        raise FeaError("radius and size must be positive [mm]")
    return {"center": c, "radius": float(radius), "size": float(size)}


def do_mesh(sess: Session, timeout_s: float, cancel_event=None) -> dict:
    from ..solve import mesh_study
    st = sess.study
    parts = sess.parts
    wd = sess.workdir / "mesh"

    def work(cancel):
        return mesh_study(st, parts, wd, cancel)          # cancellation / limit ends the Gmsh process at once
    t0 = time.time()
    mesh, warnings = run_with_timeout(work, timeout_s, "Meshing", cancel_event)
    mesh.report["study_warnings"] = list(warnings)
    sess.mesh, sess.mesh_sig, sess.mesh_time = mesh, sess.mesh_signature(), time.time() - t0
    return mesh_report(sess, mesh, warnings, time.time() - t0)


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_mesh_generate(global_size: float | None = None, element_order: int = 2, grading: list | None = None,
                          min_size: float | None = None, curvature_points: int | None = None,
                          timeout_s: float = 600) -> str:
        """tet10 mesh (Gmsh HXT, Delaunay fallback) in a separate process. global_size [mm] (None = previous; new study: auto, <= 5 mm and >= 3 elements across the thinnest part); element_order 2 = tet10 (recommended), 1 = tet4 (too stiff); grading (replaces existing): [{"selection": id/query/name, "size": mm}] (transition within 4x global) or spheres {"center","radius","size"} (see fea_mesh_refine_point); min_size; curvature_points (elements per 2*pi on curved faces, 0 = off). Returns nodes/elements/DOF, minSJ, warnings, solver_backend. The mesh is reused by fea_solve_run until geometry, selections, mesh settings, joints, ignored parts or contacts change (materials and printing settings do not)."""
        if element_order not in (1, 2):
            raise FeaError("element_order must be 1 or 2")
        with sess.use(op="meshing", wait=5):
            st = sess.require_study()
            if global_size is not None:
                if global_size <= 0:
                    raise FeaError("global_size must be positive [mm]")
                st.mesh.size = float(global_size)
            elif st.mesh.size == 5.0 and sess.mesh is None:
                st.mesh.size = auto_size(sess)
            st.mesh.order = int(element_order)
            if min_size is not None:
                st.mesh.min_size = float(min_size) if min_size > 0 else None
            if curvature_points is not None:
                st.mesh.curvature_points = int(curvature_points)
            if grading is not None:
                loc = []
                for i, g in enumerate(grading):
                    if isinstance(g, dict) and "center" in g:            # koule kolem bodu
                        loc.append(ball(g.get("center"), g.get("radius"), g.get("size")))
                        continue
                    if not isinstance(g, dict) or "selection" not in g or "size" not in g:
                        raise FeaError('grading: items of the form {"selection": …, "size": mm} or sphere {"center": [x, y, z], "radius": mm, "size": mm}')
                    sel = g["selection"]
                    nm = sel if isinstance(sel, str) and sel in st.selections else g.get("name") or f"zjemneni_{i + 1}"
                    nm = resolve_selection(sess, sel, nm)
                    loc.append({"selection": nm, "size": float(g["size"])})
                st.mesh.local = loc
            est = estimate_dof(sess, st.mesh.size)
            if est > 6_000_000:
                raise FeaError(f"The estimate of {est / 1000000.0:.1f} million DOF is beyond this computer's capability – increase global_size")
            sess.changed()
            d = do_mesh(sess, timeout_s)
            d["mesh_settings"] = asdict(st.mesh)
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_check_readiness() -> str:
        """Pre-run check: materials, selections, load states, bolts/welds, printing, and enough constraints against rigid motion (with a mesh exact, incl. free parts). Returns ready, errors, warnings, DOF estimate."""
        with sess.use():
            return out(readiness(sess))

    @mcp.tool(structured_output=False)
    def fea_mesh_info() -> str:
        """Current mesh (valid study mesh, else the results mesh): nodes, elements, DOF, minSJ, elements per part, bonded parts, *TIE pairs (part, face, tolerance), refinement spheres, settings."""
        with sess.use():
            st = sess.require_study()
            mesh = sess.valid_mesh()
            src = "valid study mesh"
            if mesh is None and sess.results is not None:
                mesh, src = sess.results.mesh, "mesh of the latest results"
            if mesh is None:
                raise FeaError("No mesh – fea_mesh_generate (or fea_solve_run creates it)"
                               + (" (existing mesh is outdated – the study has changed since meshing)" if sess.mesh else ""))
            d = mesh_report(sess, mesh, list(mesh.report.get("study_warnings", [])), sess.mesh_time or 0.0)
            d.pop("time_s", None)
            d["source"] = src
            d["mesh_settings"] = asdict(st.mesh)
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_mesh_separate_parts(parts: list[str], separate: bool = True, tie_tolerance_mm: float | None = None) -> str:
        """Meshes parts SEPARATELY and joins them to the rest by *TIE (use when Gmsh cannot merge parts, e.g. tangential contact, "PLC Error", or for a different fineness). Contact walls found within tie_tolerance_mm (None = 5 % of element size); walls with nodes in a constraint are skipped (warning). separate=False reverts. ccx only; changes the mesh."""
        if tie_tolerance_mm is not None and not (math.isfinite(tie_tolerance_mm) and tie_tolerance_mm > 0):
            raise FeaError("tie_tolerance_mm must be positive [mm] (or None = 5 % of element size)")
        with sess.use():
            st = sess.require_study()
            names = sess.part_names()
            for p in parts:
                if p not in names:
                    raise FeaError(f"Part '{p}' does not exist. Parts: {', '.join(names)}")
            cur = set(st.mesh.separate)
            cur = cur | set(parts) if separate else cur - set(parts)
            st.mesh.separate = [p for p in names if p in cur]
            if tie_tolerance_mm is not None:
                st.mesh.tie_tolerance = float(tie_tolerance_mm)
            sess.changed()                                   # the mesh signature contains the mesh settings → new mesh
            d = {"separate_parts": st.mesh.separate, "tie_tolerance_mm": st.mesh.tie_tolerance,
                 "note": "The mesh will be regenerated (fea_mesh_generate / fea_solve_run); *TIE couplings are then listed in fea_mesh_info"}
            if len(st.mesh.separate) == len([p for p in names if p not in st.excluded_parts()]) and len(names) > 1:
                d["warning"] = ("All parts are meshed separately – *TIE joins each with the others; usually only parts that do not connect to the assembly need it")
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_mesh_refine_point(center: list[float] | None = None, radius_mm: float | None = None,
                              size_mm: float | None = None, remove_all: bool = False) -> str:
        """Local refinement in a sphere (center [mm], radius_mm, size_mm; smooth transition), added to other refinement. Typically at the stress maximum from fea_results_hotspots; size ~ 1/3-1/4 of the fillet radius. remove_all=True clears the spheres. Changes the mesh."""
        with sess.use():
            st = sess.require_study()
            if remove_all:
                n0 = len(st.mesh.local)
                st.mesh.local = [loc for loc in st.mesh.local if "center" not in loc]
                if len(st.mesh.local) != n0:
                    sess.changed()
                return out({"removed": n0 - len(st.mesh.local), "local": st.mesh.local})
            if center is None:
                raise FeaError("Enter center [x, y, z], radius_mm and size_mm (or remove_all=True)")
            b = ball(center, radius_mm, size_mm)
            if b["size"] >= st.mesh.size:
                raise FeaError(f"size_mm {b['size']:g} must be smaller than the global element size {st.mesh.size:g} mm")
            est = estimate_dof(sess, st.mesh.size) + int(DOF_PER_H3 * 4.19 * b["radius"] ** 3 / b["size"] ** 3)
            if est > 6_000_000:
                raise FeaError(f"Estimate {est / 1000000.0:.1f} million DOF – reduce the sphere or increase size_mm")
            st.mesh.local = list(st.mesh.local) + [b]
            sess.changed()
            return out({"added": b, "refinement_spheres": [loc for loc in st.mesh.local if "center" in loc],
                        "dof_estimate": est, "note": "The mesh will be regenerated (fea_mesh_generate / fea_solve_run)"})

    @mcp.tool(structured_output=False)
    def fea_mesh_convergence(quantity: str = "max_mises", field: str = "mises", point: list[float] | None = None,
                             radius_mm: float | None = None, step: int | str = 0, mode: int = 1, levels: int = 3,
                             ratio: float = 1.5, tol: float = 0.02, timeout_s: float = 1800) -> str:
        """Mesh convergence study on `levels` finer meshes (size ratio `ratio`) in a copy of the study: Richardson extrapolation (order p, GCI per Roache, ASME V&V 20) and verdict (last change <= tol). quantity: max_mises | max_disp (model or sphere) | probe (field at `point`: mises, s1, sx.., ux, umag) | frequency (mode `mode`, needs fea_modal_set) | print_safety. point + radius_mm refines ONLY that sphere (cheaper); without radius_mm global (DOF ~ ratio^3 per level). step selects the state. Warns about singularities (value keeps growing) and a wandering maximum. timeout_s covers all levels."""
        from ..convergence import ConvergenceSpec, run_convergence
        if quantity not in ("max_mises", "max_disp", "probe", "frequency", "print_safety"):
            raise FeaError("quantity must be max_mises | max_disp | probe | frequency | print_safety")
        if not 2 <= int(levels) <= 6:
            raise FeaError("levels must be 2–6 (Richardson needs 3)")
        if not 1.1 <= float(ratio) <= 3.0:
            raise FeaError("ratio must be 1.1–3 (element size ratio between levels)")
        if not 0 < float(tol) < 1:
            raise FeaError("tol must be in (0; 1), e.g. 0.02 = 2 %")
        if quantity == "probe" and point is None:
            raise FeaError("quantity='probe' requires point [x, y, z]")
        if radius_mm is not None and (point is None or not radius_mm > 0):
            raise FeaError("radius_mm requires point and must be positive")
        if quantity == "probe" and field not in ("mises", "s1", "s2", "s3", "sx", "sy", "sz", "sxy", "syz", "szx",
                                                 "ux", "uy", "uz", "umag"):
            raise FeaError("field for probe: mises | s1 | s2 | s3 | sx … szx | ux | uy | uz | umag")
        with sess.use(op="mesh convergence", wait=5):
            st = sess.require_study()
            rd = readiness(sess)
            if not rd["ready"]:
                raise FeaError("The study is not ready for calculation:\n  " + "\n  ".join(rd["errors"]))
            if quantity == "frequency":
                if not st.modal.enabled:
                    raise FeaError("quantity='frequency' requires natural frequencies to be enabled (fea_modal_set)")
                if not 1 <= int(mode) <= int(st.modal.modes):
                    raise FeaError(f"mode must be 1–{st.modal.modes}")
                k = int(mode) - 1
            elif not st.load_cases:
                raise FeaError("The study has no load case")
            else:
                names = [lc.name for lc in st.load_cases]
                if isinstance(step, str) and not step.strip().lstrip("-").isdigit():
                    if step not in names:
                        raise FeaError(f"Load case '{step}' does not exist (cases: {', '.join(names)})")
                    k = names.index(step)
                else:
                    k = int(step) % len(names)
            pt = [float(v) for v in point] if point is not None else None
            if radius_mm is None:
                est = estimate_dof(sess, st.mesh.size / float(ratio) ** (int(levels) - 1))
                if est > 6_000_000:
                    raise FeaError(f"Finest level ≈ {est / 1000000.0:.1f} million DOF – reduce levels / ratio or refine only a sphere (point + radius_mm)")
            spec = ConvergenceSpec(quantity=quantity, field=field, point=pt,
                                   radius=float(radius_mm) if radius_mm else None, step=k, levels=int(levels),
                                   ratio=float(ratio), tol=float(tol))
            wd = sess.workdir / f"konvergence_{time.strftime('%H%M%S')}"
            logs: list[str] = []
            t0 = time.time()
            try:
                r = run_with_timeout(lambda cancel: run_convergence(st, wd, spec, parts=sess.parts, on_log=logs.append,
                                                                    cancel=cancel), timeout_s, "Mesh convergence")
            finally:
                for line in logs:
                    sess.add_log(line)
            r["time_s"] = time.time() - t0
            r["workdir"] = str(wd)
            r["note"] = "Results and mesh of the main study are unchanged (levels were computed in a copy of the study)"
            if quantity in ("max_mises", "probe") and not r.get("converged"):
                r["hint"] = ("Non-converging stress: check fea_results_hotspots (singularity?), monitor the quantity away from the corner / support or the displacements; for a real concentration refine further (point + radius_mm)")
            return out(r)
