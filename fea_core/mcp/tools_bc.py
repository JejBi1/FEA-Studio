"""Tools: constraints (boundary conditions), load states and loads.

The layer add_constraint / add_load is general according to the registries study.CONSTRAINT_KINDS / LOAD_KINDS:
a new kind = one thin envelope of the tool (the parameters of a dataclass) – see fea_constraint_add_fixed.
The name of the constraint / load = the name of its selection of faces (the reactions in the results are by the selection).
"""
from __future__ import annotations

import math

import numpy as np

from .common import (out, resolve_faces, resolve_selection, same_faces, selection_info, selection_users,  # noqa: F401
                     resolve_edges, resolve_points, guard_shared_selection)    # guard_shared_selection also for the modules of F8
from .session import FeaError, Session
from .tools_study import describe_constraint, describe_load

THERMAL_LOAD_NAMES = {"teplota", "thermal", "temperature", "teplotní zatížení", "teplotni zatizeni"}
GRAVITY_LOAD_NAMES = {"gravity", "tíha", "tiha", "vlastní tíha", "vlastni tiha"}
ROTATION_LOAD_NAMES = {"rotation", "otáčení", "otaceni", "rotace", "odstředivá síla", "centrifugal"}


def _vec(v, what: str, allow_none: bool = False) -> list[float] | None:
    if v is None:
        if allow_none:
            return None
        raise FeaError(f"{what}: missing vector [x, y, z]")
    try:
        out_v = [float(x) for x in v]
    except (TypeError, ValueError):
        raise FeaError(f"{what}: a vector of three numbers [x, y, z] expected") from None
    if len(out_v) != 3 or not all(math.isfinite(x) for x in out_v):
        raise FeaError(f"{what}: a vector of three finite numbers [x, y, z] expected")
    return out_v


def _users(st, sel: str) -> list[tuple[str, object]]:
    u = [("constraint", c) for c in st.constraints if c.selection == sel]
    u += [(f"load:{lc.name}", l) for lc in st.load_cases for l in lc.loads if getattr(l, "selection", None) == sel]
    return u


def _claim_name(st, name: str, allowed: list) -> None:
    """The name of the selection must not belong to another constraint / load (the reactions and the loads are by the selection)."""
    others = [w for w, o in _users(st, name) if not any(o is a for a in allowed)]
    if others:
        raise FeaError(f"Name '{name}' is already used by {others[0].replace('load:', 'load in state ')} – choose another name")


def add_constraint(sess: Session, kind: str, name: str, selection, **params) -> dict:
    """General addition of a constraint of the kind kind (key of study.CONSTRAINT_KINDS); the same name = replacement."""
    from ..study import CONSTRAINT_KINDS
    st = sess.require_study()
    if kind not in CONSTRAINT_KINDS:
        raise FeaError(f"Unknown constraint kind '{kind}' (known: {', '.join(CONSTRAINT_KINDS)})")
    old = [c for c in st.constraints if c.selection == name]
    _claim_name(st, name, old)
    guard_shared_selection(sess, name, selection, old)
    sel = resolve_selection(sess, selection, name)
    obj = CONSTRAINT_KINDS[kind](selection=sel, **params)
    st.constraints = [c for c in st.constraints if c.selection != sel] + [obj]
    sess.changed()
    return {"constraint": describe_constraint(st, obj), "selection": selection_info(sess, sel),
            "replaced": bool(old)}


def get_case(st, case_name: str, create: bool = True):
    for lc in st.load_cases:
        if lc.name == case_name:
            return lc, False
    if not create:
        raise FeaError(f"Load case '{case_name}' does not exist (cases: {', '.join((l.name for l in st.load_cases))})")
    from ..study import LoadCase
    lc = LoadCase(case_name)
    st.load_cases.append(lc)
    return lc, True


def resolve_entity_selection(sess: Session, entity: str, selection, name: str, allowed: list = (),
                             part: str | None = None) -> tuple[str, dict]:
    """Selection of edges / points for a load: an existing selection of the same kind (selection = its name), otherwise a new selection
    name from the edges (resolve_edges) / points (resolve_points). Returns (name of the selection, additional information).
    """
    st = sess.require_study()
    if isinstance(selection, str) and selection in st.selections and getattr(st.selections[selection], entity):
        return selection, {}
    if name in st.selections:
        users = [w for w, o in selection_users(st, name) if not any(o is a for a in allowed)]
        if users:
            raise FeaError(f"Selection '{name}' is already used by {users[0]} – choose another name")
    extra = {}
    if entity == "edges":
        st.add_selection(name, edges=resolve_edges(sess, selection))
    else:
        pts, dists = resolve_points(sess, selection, part)
        st.add_selection(name, points=pts)
        if max(dists, default=0.0) > 1e-6:
            extra["projected_mm"] = dists                 # the point moved onto the surface of the part
    return name, extra


def _entity_items(sel) -> list:
    """Edges / points of the selection as the input for a new selection (a copy under another name)."""
    if sel.edges:
        return [e.to_dict() for e in sel.edges]
    return [{"point": p.point, "part": p.part} for p in sel.points]


def check_csys(st, csys: str | None, cylindrical_ok: bool = False) -> None:
    if csys is None:
        return
    cs = st.coordinate_systems.get(csys)
    if cs is None:
        raise FeaError(f"Coordinate system '{csys}' does not exist (systems: {', '.join(st.coordinate_systems) or 'none'} – create it with fea_csys_create)")
    if cs.kind == "cylindrical" and not cylindrical_ok:
        raise FeaError(f"The cylindrical system '{csys}' cannot be used here (the direction would depend on the point) – only for a force; use a Cartesian one")


def add_load(sess: Session, case_name: str, kind: str, name: str, selection=None, entity: str = "faces",
             part: str | None = None, **params) -> dict:
    """General addition of a load of the kind kind (key of study.LOAD_KINDS) to the state (creates it if it is missing).
    entity = "faces" (selection of faces) | "edges" (selection of edges) | "points" (selection of points, part = part of the points).
    """
    from ..study import LOAD_KINDS
    st = sess.require_study()
    if kind not in LOAD_KINDS:
        raise FeaError(f"Unknown load kind '{kind}' (known: {', '.join(LOAD_KINDS)})")
    check_csys(st, params.get("csys"), kind == "force")
    lc, created = get_case(st, case_name)
    klass = LOAD_KINDS[kind]
    has_sel = "selection" in klass.__dataclass_fields__
    extra = {}
    if has_sel:
        old = [l for l in lc.loads if getattr(l, "selection", None) == name]
        _claim_name(st, name, old)
        if entity == "faces":
            if isinstance(selection, str) and selection in st.selections and                     st.selections[selection].entity in ("edges", "points"):
                if kind != "force":
                    raise FeaError(f"Selection '{selection}' contains edges or points – this kind of load works on faces only")
                sel, extra = resolve_entity_selection(sess, st.selections[selection].entity,
                                                      selection if name == selection else
                                                      _entity_items(st.selections[selection]), name, old, part)
            else:
                guard_shared_selection(sess, name, selection, old)
                sel = resolve_selection(sess, selection, name)
        else:
            sel, extra = resolve_entity_selection(sess, entity, selection, name, old, part)
        obj = klass(selection=sel, **params)
        lc.loads = [l for l in lc.loads if getattr(l, "selection", None) != sel] + [obj]
        info = selection_info(sess, sel)
    else:                                                 # weight: one per state
        old = [l for l in lc.loads if l.kind == kind]
        obj = klass(**params)
        lc.loads = [l for l in lc.loads if l.kind != kind] + [obj]
        info = None
    sess.changed()
    d = {"load_case": lc.name, "load": describe_load(obj, st), "replaced": bool(old), "case_created": created}
    if info:
        d["selection"] = info
    d.update(extra)
    if has_sel and st.selections[obj.selection].entity in ("edges", "points"):
        d["warning"] = ("A force on an edge / at a point is singular in a solid model – the stress and displacement near it grow with mesh refinement; evaluate away from the point of application.")
    return d


def register(mcp, sess: Session) -> None:

    # ------------------------------------------------------------------ constraints
    @mcp.tool(structured_output=False)
    def fea_constraint_add_fixed(name: str, selection: str | list | dict) -> str:
        """Clamps faces (all displacements 0). name = constraint name (= its selection; reactions reported under it). selection: ids from fea_geometry_find_faces, query {"x": 0} or selection name. The clamped edge is a stress singularity."""
        with sess.use():
            return out(add_constraint(sess, "fixed", name, selection))

    @mcp.tool(structured_output=False)
    def fea_constraint_add_displacement(name: str, selection: str | list | dict, ux: float | None = None,
                                        uy: float | None = None, uz: float | None = None) -> str:
        """Prescribed displacement [mm] per global axis; None = free. 0 = sliding support/symmetry plane (e.g. uz=0 on z=0), non-zero = forced displacement."""
        if ux is None and uy is None and uz is None:
            raise FeaError("Give at least one component ux, uy, uz (None = free)")
        with sess.use():
            return out(add_constraint(sess, "displacement", name, selection, ux=ux, uy=uy, uz=uz))

    @mcp.tool(structured_output=False)
    def fea_constraint_add_cylindrical(name: str, selection: str | list | dict, radial: bool = True,
                                       tangential: bool = True, axial: bool = True) -> str:
        """Cylindrical support of a hole/pin (Ansys Cylindrical Support): zero displacement in the chosen components of the cylinder axes - radial, tangential (rotation about the axis), axial. selection = cylindrical faces (half-cylinders OK; several non-coaxial holes allowed, each gets its own axis, reaction = sum). Pin in a hole: radial=True, tangential=False (then hold the rotation otherwise). ccx only; must not share nodes with a remote point or bolt."""
        if not (radial or tangential or axial):
            raise FeaError("A cylindrical support without a constrained component holds nothing – choose radial, tangential or axial")
        with sess.use():
            return out(add_constraint(sess, "cylindrical", name, selection, radial=bool(radial),
                                      tangential=bool(tangential), axial=bool(axial)))

    @mcp.tool(structured_output=False)
    def fea_constraint_add_remote(name: str, selection: str | list | dict, point: list[float] | None = None,
                                  dofs: str | list = "ux,uy,uz,rx,ry,rz", coupling: str = "rigid") -> str:
        """Remote displacement: faces coupled to `point` [mm] (None = selection centroid); components in dofs are zero (ux,uy,uz,rx,ry,rz about the point; list of names or 6 bools; "ux,uy,uz" = hinge). coupling "rigid" (face does not deform; must not share nodes with a clamp/displacement) or "distributing" (RBE3, face deforms; small faces, large ones give an error). Reaction: force and moment ABOUT THE POINT (fea_results_summary -> remote_supports). ccx only."""
        if coupling not in ("distributing", "rigid"):
            raise FeaError("coupling must be 'rigid' or 'distributing'")
        from ..remote import parse_dofs, RemoteError
        try:
            d = parse_dofs(dofs)
        except RemoteError as e:
            raise FeaError(str(e)) from None
        if not any(d):
            raise FeaError("A remote support without a constrained component holds nothing – give dofs (e.g. 'ux,uy,uz')")
        with sess.use():
            if point is None:
                faces = resolve_faces(sess, selection)
                area = sum(f.area for f in faces)
                p = (sum(np.asarray(f.centroid) * f.area for f in faces) / max(area, 1e-30)).tolist()
            else:
                p = _vec(point, "point")
            return out(add_constraint(sess, "remote_support", name, selection, point=[float(v) for v in p],
                                      dofs=d, coupling=coupling))

    @mcp.tool(structured_output=False)
    def fea_constraint_add_elastic_support(name: str, selection: str | list | dict, stiffness: float,
                                           direction: str = "normal", per_area: bool = True) -> str:
        """Elastic (Winkler) support. per_area=True: stiffness k [N/mm3] (p = k*u); False: total K [N/mm] spread by area. direction "normal" (perpendicular only) or "all" (x,y,z). Typical k: rubber 1-10 (E/t), wood/soil 0.01-0.2, concrete 10-100. Acts in tension and compression; reaction = -k*u. Normal-only support does not stop in-plane rigid motion - add another constraint. ccx only."""
        if not (isinstance(stiffness, (int, float)) and math.isfinite(stiffness) and stiffness > 0):
            raise FeaError("stiffness must be a positive number (N/mm³ with per_area, otherwise N/mm)")
        if direction not in ("normal", "all"):
            raise FeaError("direction must be 'normal' (perpendicular to the face) or 'all' (x, y, z)")
        with sess.use():
            return out(add_constraint(sess, "elastic", name, selection, stiffness=float(stiffness),
                                      direction=direction, per_area=bool(per_area)))

    @mcp.tool(structured_output=False)
    def fea_constraint_list() -> str:
        """Constraints (kind, name = selection, parameters, faces)."""
        with sess.use():
            st = sess.require_study()
            return out({"constraints": [{**describe_constraint(st, c), "selection_info": selection_info(sess, c.selection)}
                                        for c in st.constraints if c.selection in st.selections]})

    @mcp.tool(structured_output=False)
    def fea_constraint_remove(name: str) -> str:
        """Removes a constraint by name (the selection stays)."""
        with sess.use():
            st = sess.require_study()
            n0 = len(st.constraints)
            st.constraints = [c for c in st.constraints if c.selection != name]
            if len(st.constraints) == n0:
                raise FeaError(f"Constraint '{name}' does not exist")
            sess.changed()
            return out({"removed": name, "constraints": [c.selection for c in st.constraints]})

    # ------------------------------------------------------------------ load states
    @mcp.tool(structured_output=False)
    def fea_loadcase_create(name: str) -> str:
        """Creates a load state (one solver step; states are superposed in fatigue)."""
        with sess.use():
            st = sess.require_study()
            lc, created = get_case(st, name)
            if created:
                sess.changed()
            return out({"load_case": name, "created": created, "load_cases": [l.name for l in st.load_cases]})

    @mcp.tool(structured_output=False)
    def fea_loadcase_delete(name: str) -> str:
        """Deletes a load state with its loads."""
        with sess.use():
            st = sess.require_study()
            lc, _ = get_case(st, name, create=False)
            st.load_cases.remove(lc)
            sess.changed()
            return out({"deleted": name, "load_cases": [l.name for l in st.load_cases]})

    # ------------------------------------------------------------------ loads
    @mcp.tool(structured_output=False)
    def fea_load_add_force(case_name: str, name: str, selection: str | list | dict, force: list[float],
                           csys: str | None = None) -> str:
        """Total force [N] [fx,fy,fz] spread uniformly over faces (or along an existing edge selection / onto point selections). The state case_name is created if missing; the same name in a state replaces. csys: local system (fea_csys_create); Cartesian = components in its axes; cylindrical = (F_r, F_theta, F_z) at each point, force density F/A, resultant generally != F (tangential component around a full circumference = torque)."""
        f = _vec(force, "force")
        with sess.use():
            return out(add_load(sess, case_name, "force", name, selection, vector=f, csys=csys or None))

    @mcp.tool(structured_output=False)
    def fea_load_add_edge_force(case_name: str, name: str, edges: str | list | dict, force: list[float],
                                csys: str | None = None) -> str:
        """Force [N] along edges (line load F/L). edges: id 'Part#e<i>', list, query {"x": 200, "z": 10} or edge selection name. csys as in fea_load_add_force. An edge force is a singularity - do not trust the stress at the edge (or load a narrow face)."""
        f = _vec(force, "force")
        with sess.use():
            return out(add_load(sess, case_name, "force", name, edges, entity="edges", vector=f, csys=csys or None))

    @mcp.tool(structured_output=False)
    def fea_load_add_point_force(case_name: str, name: str, points: list | dict | str, force: list[float],
                                 part: str | None = None, csys: str | None = None) -> str:
        """Force [N] at points on the surface, F/n per point. points: [x,y,z], list, {"point","part"} or point selection name; projected onto the nearest surface of `part` (None = nearest part) and meshed as a node. csys as in fea_load_add_force. A point force is a singularity (stress around it is meaningless); use a small face or remote load."""
        f = _vec(force, "force")
        with sess.use():
            return out(add_load(sess, case_name, "force", name, points, entity="points", part=part, vector=f,
                                csys=csys or None))

    @mcp.tool(structured_output=False)
    def fea_load_add_pressure(case_name: str, name: str, selection: str | list | dict, pressure: float,
                              gradient: list[float] | None = None, origin: list[float] | None = None,
                              clip: bool = False) -> str:
        """Pressure [MPa] on faces; positive acts into the material, negative = suction. Linear: p(x) = pressure + gradient*(x - origin), gradient [MPa/mm] [gx,gy,gz], origin [mm]; clip=True keeps only the positive linear term (liquid level). For liquids see fea_load_add_hydrostatic."""
        if not math.isfinite(pressure):
            raise FeaError("pressure must be a finite number [MPa]")
        g = _vec(gradient, "gradient", True)
        o = _vec(origin, "origin", True)
        with sess.use():
            return out(add_load(sess, case_name, "pressure", name, selection, value=float(pressure),
                                gradient=g, origin=o if g else None, clip=bool(clip) if g else False))

    @mcp.tool(structured_output=False)
    def fea_load_add_hydrostatic(case_name: str, name: str, selection: str | list | dict,
                                 surface_point: list[float], density_kg_m3: float = 1000.0, g: float = 9810.0,
                                 gravity_direction: list[float] | None = None, surface_pressure: float = 0.0) -> str:
        """Hydrostatic pressure on faces: p = surface_pressure [MPa] + rho*g*depth below the level; above it only surface_pressure. surface_point [mm] on the level; gravity_direction default [0,0,-1] (level perpendicular to it); g [mm/s2]; density_kg_m3 (water 1000). Acts into the material."""
        d = np.asarray(_vec(gravity_direction if gravity_direction is not None else [0, 0, -1], "gravity_direction"))
        if np.linalg.norm(d) == 0:
            raise FeaError("gravity_direction must not be a zero vector")
        if not (density_kg_m3 > 0 and g > 0):
            raise FeaError("density_kg_m3 and g must be positive")
        grad = (d / np.linalg.norm(d) * float(density_kg_m3) * 1e-12 * float(g)).tolist()
        with sess.use():
            return out(add_load(sess, case_name, "pressure", name, selection, value=float(surface_pressure),
                                gradient=grad, origin=_vec(surface_point, "surface_point"), clip=True))

    @mcp.tool(structured_output=False)
    def fea_load_add_gravity(case_name: str, name: str = "Gravity", direction: list[float] | None = None,
                             g: float = 9810.0) -> str:
        """Self-weight: acceleration g [mm/s2] (9810 = 1 g) along direction (default [0,0,-1]). Materials need rho [t/mm3]. One per state (replaces); name is a label."""
        dvec = np.asarray(_vec(direction if direction is not None else [0, 0, -1], "direction"))
        if np.linalg.norm(dvec) == 0:
            raise FeaError("direction must not be a zero vector")
        with sess.use():
            st = sess.require_study()
            d = add_load(sess, case_name, "gravity", name, vector=(dvec / np.linalg.norm(dvec) * float(g)).tolist())
            no_rho = [m.name for m in st.materials.values() if not m.rho]
            if no_rho:
                d["warning"] = f"Materials without density (the weight is zero): {', '.join(no_rho)}"
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_load_add_remote(case_name: str, name: str, selection: str | list | dict, point: list[float],
                            force: list[float] | None = None, moment: list[float] | None = None,
                            coupling: str = "distributing", csys: str | None = None) -> str:
        """Force [N] and moment [N*mm] at remote `point` [mm] carried to the faces: coupling "distributing" (RBE3, no extra stiffness, recommended) or "rigid" (stiffens the face; ccx only). csys: Cartesian local system for force/moment components."""
        if coupling not in ("distributing", "rigid"):
            raise FeaError("coupling must be 'distributing' or 'rigid'")
        p = _vec(point, "point")
        f = _vec(force, "force", True) or [0.0, 0.0, 0.0]
        m = _vec(moment, "moment", True) or [0.0, 0.0, 0.0]
        if not any(f) and not any(m):
            raise FeaError("Enter force or moment")
        with sess.use():
            return out(add_load(sess, case_name, "remote", name, selection, point=p, force=f, moment=m,
                                coupling=coupling, csys=csys or None))

    @mcp.tool(structured_output=False)
    def fea_load_add_bearing(case_name: str, name: str, selection: str | list | dict, force: list[float],
                             csys: str | None = None) -> str:
        """Bearing load [N] on a cylindrical hole/pin face (Ansys Bearing Load): the radial component = pressure ~ cos(theta) on the half facing the force (resultant exactly F, zero moment about the axis), axial component = uniform shear. csys: Cartesian local system."""
        f = _vec(force, "force")
        with sess.use():
            return out(add_load(sess, case_name, "bearing", name, selection, vector=f, csys=csys or None))

    @mcp.tool(structured_output=False)
    def fea_load_add_rotation(case_name: str, axis_point: list[float], axis_direction: list[float],
                              rpm: float | None = None, omega: float | None = None, alpha: float = 0.0,
                              parts: list[str] | None = None, name: str = "Rotation") -> str:
        """Rotation of parts about an axis (axis_point [mm], axis_direction): centrifugal force from rpm OR omega [rad/s], inertia force from alpha [rad/s2] (positive = accelerating right-handed; drive moment J*alpha appears as constraint reaction). parts: rotating parts (None = all). Needs density. Static in the rotating frame (no Coriolis, no centrifugal softening). One per state; name is a label."""
        if rpm is not None and omega is not None:
            raise FeaError("Give rpm OR omega, not both")
        w = float(omega) if omega is not None else (float(rpm) * 2 * math.pi / 60 if rpm is not None else 0.0)
        p = _vec(axis_point, "axis_point")
        a = np.asarray(_vec(axis_direction, "axis_direction"))
        if np.linalg.norm(a) == 0:
            raise FeaError("axis_direction must not be a zero vector")
        if not (math.isfinite(w) and math.isfinite(alpha)) or (w == 0 and alpha == 0):
            raise FeaError("Give a nonzero speed (rpm / omega) or the acceleration alpha")
        with sess.use():
            st = sess.require_study()
            for pn in parts or []:
                sess.part_index(pn)
            d = add_load(sess, case_name, "rotation", name, point=p, axis=(a / np.linalg.norm(a)).tolist(),
                         omega=w, alpha=float(alpha), parts=list(parts or []))
            no_rho = set()
            for pn in (parts or [x.name for x in sess.parts]):
                try:
                    m = st.material_of(pn)
                except KeyError:
                    continue                               # missing material is reported by the readiness check
                if pn not in st.excluded_parts() and not m.rho:
                    no_rho.add(m.name)
            if no_rho:
                d["warning"] = f"Materials without density (the force is zero / calculation error): {', '.join(no_rho)}"
            return out(d)

    # ------------------------------------------------------------------ coordinate systems
    @mcp.tool(structured_output=False)
    def fea_csys_create(name: str, kind: str = "cartesian", origin: list[float] | None = None,
                        z_axis: list[float] | None = None, x_axis: list[float] | None = None,
                        from_face: str | dict | None = None) -> str:
        """Local coordinate system for load components (csys of forces). kind "cartesian" or "cylindrical" (r, theta, z; axis = z_axis through origin; theta=0 along x_axis, projected perpendicular to z). from_face: cylindrical face (id or query) -> origin and z_axis from its axis. The same name replaces."""
        from ..csys import check as check_cs
        from ..study import CoordinateSystem
        if kind not in ("cartesian", "cylindrical"):
            raise FeaError("kind must be 'cartesian' or 'cylindrical'")
        with sess.use():
            st = sess.require_study()
            o = _vec(origin, "origin", True)
            z = _vec(z_axis, "z_axis", True)
            if from_face is not None:
                faces = resolve_faces(sess, from_face)
                from ..selectors import resolve
                r = resolve(faces[0], sess.parts, sess._face_props)
                fp = sess.face_props(r[0])[r[1]] if r is not None else None
                if fp is None or fp.get("radius") is None:
                    raise FeaError("from_face must be a cylindrical face")
                o = o or [float(v) for v in fp["axis_point"]]
                z = z or [float(v) for v in fp["axis_dir"]]
            z = z or [0.0, 0.0, 1.0]
            x = _vec(x_axis, "x_axis", True)
            if x is None:                                  # any direction perpendicular to z
                zz = np.asarray(z, float)
                x = np.cross(zz, [1.0, 0, 0] if abs(zz[0]) < 0.9 * np.linalg.norm(zz) else [0, 1.0, 0]).tolist()
                x = np.cross(x, zz).tolist()
            cs = CoordinateSystem(name, kind, o or [0.0, 0.0, 0.0], z, x)
            errs = check_cs(cs)
            if errs:
                raise FeaError(errs[0])
            old = name in st.coordinate_systems
            st.coordinate_systems[name] = cs
            sess.changed()
            from ..csys import matrix, describe
            return out({"csys": name, "replaced": old, "description": describe(cs), "axes_global": matrix(cs)})

    @mcp.tool(structured_output=False)
    def fea_csys_list() -> str:
        """Coordinate systems (kind, origin, axes) and the loads using them."""
        from ..csys import describe, matrix
        with sess.use():
            st = sess.require_study()
            return out({"coordinate_systems": [
                {"name": k, "kind": c.kind, "origin": c.origin, "axes_global": matrix(c), "description": describe(c),
                 "used_by": [f"{l.kind} '{getattr(l, 'selection', l.kind)}' in state '{lc.name}'"
                             for lc in st.load_cases for l in lc.loads if getattr(l, "csys", None) == k]}
                for k, c in st.coordinate_systems.items()]})

    @mcp.tool(structured_output=False)
    def fea_csys_delete(name: str) -> str:
        """Deletes a coordinate system (refused while a load uses it)."""
        with sess.use():
            st = sess.require_study()
            if name not in st.coordinate_systems:
                raise FeaError(f"Coordinate system '{name}' does not exist")
            users = [f"load '{getattr(l, 'selection', l.kind)}' in state '{lc.name}'" for lc in st.load_cases
                     for l in lc.loads if getattr(l, "csys", None) == name]
            users += [f"material '{m.name}' (orthotropy axes)" for m in st.materials.values()
                      if getattr(m, "ortho_csys", None) == name]
            if users:
                raise FeaError(f"Coordinate system '{name}' is used by {users[0]} – change / remove that first")
            del st.coordinate_systems[name]
            sess.changed()
            return out({"deleted": name, "coordinate_systems": list(st.coordinate_systems)})

    @mcp.tool(structured_output=False)
    def fea_load_list(case_name: str | None = None) -> str:
        """Load states and their loads (optionally one state)."""
        with sess.use():
            st = sess.require_study()
            cases = [get_case(st, case_name, create=False)[0]] if case_name else st.load_cases
            return out({"load_cases": [{"name": lc.name, "loads": [describe_load(l, st) for l in lc.loads]}
                                       for lc in cases]})

    @mcp.tool(structured_output=False)
    def fea_load_remove(case_name: str, name: str) -> str:
        """Removes a load from a state by name (weight: 'gravity' or 'Tíha'; rotation: 'rotation' or 'otáčení'; thermal load: 'teplota' or the thermal state name - removes all thermal loads of the state)."""
        with sess.use():
            st = sess.require_study()
            lc, _ = get_case(st, case_name, create=False)
            n0 = len(lc.loads)
            lc.loads = [l for l in lc.loads if getattr(l, "selection", None) != name]
            if len(lc.loads) == n0:                       # the weight and the temperature have no selection – the name is only a label
                if name.strip().lower() in THERMAL_LOAD_NAMES or \
                        any(l.kind == "thermal" and l.case == name for l in lc.loads):
                    lc.loads = [l for l in lc.loads if l.kind != "thermal"]
                elif name.strip().lower() in ROTATION_LOAD_NAMES:
                    lc.loads = [l for l in lc.loads if l.kind != "rotation"]
                elif name.strip().lower() in GRAVITY_LOAD_NAMES:      # otherwise any unknown name would delete the weight
                    lc.loads = [l for l in lc.loads if hasattr(l, "selection") or l.kind != "gravity"]
            if len(lc.loads) == n0:
                raise FeaError(f"Load '{name}' in state '{case_name}' does not exist")
            sess.changed()
            return out({"removed": name, "load_case": case_name, "loads": [describe_load(l, st) for l in lc.loads]})

    @mcp.tool(structured_output=False)
    def fea_combination_add(name: str, factors: dict[str, float],
                            standard: str = "CSN_EN_1990",
                            design_situation: str = "STR_GEO_6_10",
                            description: str = "",
                            envelope: bool = False,
                            envelope_type: str = "max",
                            envelope_combinations: list[str] | None = None,
                            envelope_field: str = "von_mises") -> str:
        """Adds a load-state combination or envelope (post-processing only; results are kept). Linear superposition of U, S, RF; von Mises and safety from the combined tensor. factors {state: factor}; standard 'CSN_EN_1990'; design_situation 'STR_GEO_6_10' | 'STR_GEO_6_10a_6_10b' | 'EQU' | 'SLS_CHAR' | 'SLS_FREQ' | 'SLS_QUASI' | 'custom'; description. Envelope: envelope=True, envelope_type max|min|abs_max, envelope_combinations (list), envelope_field von_mises|umag|s1|s3|safety (worst safety = min)."""
        with sess.use(op="combination_add"):
            st = sess.require_study()
            if any(c.name == name for c in st.combinations):
                raise FeaError(f"Combination '{name}' already exists")
            from ..combinations import Combination, validate_combination
            comb = Combination(
                name=name,
                factors={k: float(v) for k, v in factors.items()},
                standard=standard,
                design_situation=design_situation,
                description=description,
                envelope=envelope,
                envelope_type=envelope_type,
                envelope_combinations=list(envelope_combinations) if envelope_combinations else [],
                envelope_field=envelope_field,
            )
            try:
                validate_combination(st, comb)
            except ValueError as e:
                raise FeaError(str(e))
            st.combinations.append(comb)
            return out({"added_combination": comb.to_dict(), "total_combinations": len(st.combinations)})

    @mcp.tool(structured_output=False)
    def fea_combination_generate_en1990(cases: list[dict], formula: str = "6.10") -> str:
        """Generates EN 1990 + national annex combinations. cases: [{"name","action_type": permanent|variable, "subtype"?, "category"?, "leading"?}] e.g. {"name":"Uzitne","action_type":"variable","category":"B","leading":true}, {"name":"Snih","action_type":"variable","subtype":"snow_below_1000m"}. formula '6.10' | '6.10a_6.10b' | 'all'. Adds them to the study and returns them."""
        with sess.use(op="combination_generate"):
            st = sess.require_study()
            from ..combinations import generate_en1990_combinations, validate_combination
            known = [lc.name for lc in st.load_cases]
            bad = [c.get("name") for c in cases if c.get("name") not in known]
            if bad:
                raise FeaError(f"Load cases {bad} do not exist in the study. Cases of the study: {known}")
            try:
                combs = generate_en1990_combinations(cases, formula=formula)
                for c in combs:
                    validate_combination(st, c, others=combs)
            except (ValueError, KeyError) as e:
                raise FeaError(str(e))
            for c in combs:
                st.combinations = [old for old in st.combinations if old.name != c.name]
                st.combinations.append(c)
            return out({"generated_count": len(combs), "combinations": [c.to_dict() for c in combs]})

    @mcp.tool(structured_output=False)
    def fea_combination_list() -> str:
        """Combinations and envelopes of the study."""
        with sess.use():
            st = sess.require_study()
            return out({"combinations": [c.to_dict() for c in st.combinations]})

    @mcp.tool(structured_output=False)
    def fea_combination_delete(name: str) -> str:
        """Deletes a combination by name."""
        with sess.use(op="combination_delete"):
            st = sess.require_study()
            idx = next((i for i, c in enumerate(st.combinations) if c.name == name), None)
            if idx is None:
                raise FeaError(f"Combination '{name}' does not exist")
            users = [c.name for c in st.combinations if c.envelope and name in c.envelope_combinations]
            if users:
                raise FeaError(f"Combination '{name}' is used in the envelope {users} – delete the envelope first")
            deleted = st.combinations.pop(idx)
            return out({"deleted": deleted.to_dict(), "remaining": len(st.combinations)})

