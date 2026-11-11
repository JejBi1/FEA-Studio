"""Tools: mechanism of rigid and flexible bodies and contact (stage F8.3, F8.8, F8.9 C1 – fea_core.motion; Exudyn,
Chrono for contact, both in a separate process).

Procedure: fea_motion_body_add (bodies = parts, frame ground) → fea_motion_joints_from_cad (proposal of joints from coaxial
cylinders, accept=True) or fea_motion_joint_add → fea_motion_driver_add / fea_motion_force_add / fea_motion_load_add
(→ fea_motion_contact_add + fea_motion_settings(solver='chrono') for the contact of shapes; flexible=True in
fea_motion_body_add for a flexible body) → fea_motion_check_dof → fea_motion_settings → fea_motion_run →
fea_motion_results / fea_motion_render →
fea_motion_export_loads (instant / maximum / whole course → load states of the FEM and channels of fatigue) → fea_solve_run.
Units of the project: mm, N, N·mm, s, rad; mass in the tools in kg (internally t). The result is in the session (sess.motion).
"""
from __future__ import annotations

import math
import time

import numpy as np
from mcp.server.mcpserver import Image

from .common import guarded_selection, out, run_with_timeout
from .session import FeaError, Session, user_file

KG = 1e-3                     # kg → t


def _mech(sess: Session):
    return sess.require_study().mechanism


def _changed(sess: Session):
    """Change of the mechanism: the motion result does not hold (the FEM results stay – the motion does not change them)."""
    sess.motion = None


def _vec(v, name, n=3):
    if v is None:
        return None
    try:
        a = [float(x) for x in v]
    except (TypeError, ValueError):
        raise FeaError(f"{name} must be a list of {n} numbers") from None
    if len(a) != n or not all(math.isfinite(x) for x in a):
        raise FeaError(f"{name} must be a list of {n} finite numbers")
    return a


def _body_props(sess: Session) -> dict:
    from ..motion.run import body_props, MotionError, _need_parts
    st = sess.require_study()
    try:
        return body_props(st, sess.parts if _need_parts(st) and not sess.frame_only else None)
    except MotionError as e:
        raise FeaError(str(e)) from None


def _body_dict(b, props) -> dict:
    p = props.get(b.name) or {}
    d = {"name": b.name, "parts": b.parts, "ground": b.ground}
    if getattr(b, "flexible", False) and not b.ground:
        d["flexible"] = {"modes": b.modes, "interface": b.interface, "mesh_size_mm": b.mesh_size, "zeta": b.zeta}
    if p and not b.ground:
        d.update(mass_kg=p["mass"] / KG, com_mm=[float(x) for x in p["com"]],
                 inertia_kgmm2=(np.asarray(p["inertia"]) / KG).round(6).tolist(), source=p["source"])
        if p.get("extra"):
            d["extra_parts"] = p["extra"]
        if p.get("printed"):
            d["printed_estimate"] = p["printed"]
    return d


def _axis_from_faces(sess: Session, sel_name: str):
    """(point on the axis, direction, radius) of the cylindrical faces of the selection from the CAD (the first cylinder)."""
    from ..geometry import cylinder_axis
    from ..selectors import resolve
    st = sess.require_study()
    for f in st.selections[sel_name].faces:
        hit = resolve(f, sess.parts)
        if hit is None:
            continue
        c = cylinder_axis(sess.faces_of(hit[0])[hit[1]])
        if c is not None:
            p, d, r = c
            return np.asarray(p, float), np.asarray(d, float), float(r)
    return None


def _project_on_axis(p, d, pts):
    d = d / np.linalg.norm(d)
    s = [(np.asarray(x) - p) @ d for x in pts]
    return p + d * (0.5 * (min(s) + max(s))) if s else p


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_motion_body_add(name: str, parts: list[str] | None = None, ground: bool = False,
                            mass_kg: float | None = None, com_mm: list[float] | None = None,
                            inertia_kgmm2: list | None = None, v0_mm_s: list[float] | None = None,
                            w0_rad_s: list[float] | None = None, flexible: bool = False, modes: int = 10,
                            interface: str = "rbe2", mesh_size_mm: float | None = None, zeta: float = 0.0) -> str:
        """Adds/replaces a mechanism body = part or group of parts (rigidly joined); ground=True = frame. Mass, centre of gravity and inertia come from CAD (volume x rho; printed parts INDICATIVE); ignored parts and bolt parts join the nearest body. mass_kg, com_mm, inertia_kgmm2 (3x3 about the CoG) override - a body without parts needs all. v0_mm_s, w0_rad_s: initial velocities (dynamic only). flexible=True: Craig-Bampton FFRF body (dynamic, exudyn, isotropic materials only): mesh_size_mm (None = study mesh), interface = faces of the body's joints (faces_a/faces_b in fea_motion_joint_add; a joint without faces = error) rbe2 | rbe3, modes with clamped interfaces, zeta of the 1st mode. Results: series "napětí:<body>" (max vM) and "deformace:<body>"."""
        from ..motion.model import Body
        if interface not in ("rbe2", "rbe3"):
            raise FeaError("interface must be rbe2 | rbe3")
        if flexible and not 1 <= int(modes) <= 200:
            raise FeaError("modes 1–200")
        with sess.use():
            st = sess.require_study()
            mech = st.mechanism
            parts = list(parts or [])
            if parts and not sess.frame_only:
                names = sess.part_names()
                bad = [p for p in parts if p not in names]
                if bad:
                    raise FeaError(f"Parts {', '.join(bad)} are not in the geometry (parts: {', '.join(names)})")
            for b in mech.bodies:
                if b.name != name:
                    dup = set(b.parts) & set(parts)
                    if dup:
                        raise FeaError(f"Parts {', '.join(sorted(dup))} already belong to the body '{b.name}'")
            J = None
            if inertia_kgmm2 is not None:
                J = np.asarray(inertia_kgmm2, float)
                if J.size == 3:
                    J = np.diag(J)
                if J.shape != (3, 3):
                    raise FeaError("inertia_kgmm2 = a 3×3 matrix or 3 principal moments")
                J = (J * KG).tolist()
            if not parts and not ground and (mass_kg is None or com_mm is None or J is None):
                raise FeaError("A body without parts needs mass_kg, com_mm and inertia_kgmm2")
            if flexible and (ground or not parts):
                raise FeaError("A flexible body needs parts from CAD and must not be a frame")
            b = Body(name, parts, ground, None if mass_kg is None else float(mass_kg) * KG, _vec(com_mm, "com_mm"), J,
                     v0=_vec(v0_mm_s, "v0_mm_s"), w0=_vec(w0_rad_s, "w0_rad_s"), flexible=bool(flexible),
                     modes=int(modes), interface=interface,
                     mesh_size=None if mesh_size_mm is None else float(mesh_size_mm), zeta=float(zeta))
            mech.bodies = [x for x in mech.bodies if x.name != name] + [b]
            _changed(sess)
            props = _body_props(sess)
            return out({"body": _body_dict(b, props), "bodies": [x.name for x in mech.bodies],
                        "warnings": props.get("_warnings", []),
                        "next": "fea_motion_joints_from_cad (joint proposal) or fea_motion_joint_add"})

    @mcp.tool(structured_output=False)
    def fea_motion_body_list() -> str:
        """Bodies with mass properties (kg, mm, kg*mm2), parts outside bodies, mass check against CAD."""
        with sess.use():
            mech = _mech(sess)
            props = _body_props(sess)
            chk = props.get("_check", {})
            return out({"bodies": [_body_dict(b, props) for b in mech.bodies],
                        "mass_total_kg": (chk.get("bodies_t") or 0.0) / KG,
                        "mass_cad_kg": None if chk.get("cad_t") is None else chk["cad_t"] / KG,
                        "unassigned_parts": chk.get("unassigned", []), "warnings": props.get("_warnings", [])})

    @mcp.tool(structured_output=False)
    def fea_motion_body_remove(name: str, cascade: bool = False) -> str:
        """Removes a body; cascade=True also removes joints, force elements, drivers and loads that use it."""
        with sess.use():
            mech = _mech(sess)
            if name not in [b.name for b in mech.bodies]:
                raise FeaError(f"Body '{name}' does not exist")
            js = [j.name for j in mech.joints if name in (j.body_a, j.body_b)]
            fs = [f.name for f in mech.forces if name in (f.body_a, f.body_b)]
            ls = [x.name for x in mech.loads if x.body == name]
            cs = [c.name or f"{c.body_a}–{c.body_b}" for c in mech.contacts if name in (c.body_a, c.body_b)]
            if (js or fs or ls or cs) and not cascade:
                raise FeaError(f"The body is used by: joints {js}, elements {fs}, loads {ls}, contacts {cs} – cascade=True removes them")
            mech.contacts = [c for c in mech.contacts if name not in (c.body_a, c.body_b)]
            mech.bodies = [b for b in mech.bodies if b.name != name]
            mech.joints = [j for j in mech.joints if j.name not in js]
            mech.forces = [f for f in mech.forces if f.name not in fs]
            mech.loads = [x for x in mech.loads if x.body != name]
            mech.drivers = [d for d in mech.drivers if d.joint in [j.name for j in mech.joints]]
            _changed(sess)
            return out({"removed": name, "joints_removed": js, "forces_removed": fs, "loads_removed": ls})

    @mcp.tool(structured_output=False)
    def fea_motion_joint_add(name: str, kind: str, body_a: str = "", body_b: str = "",
                             origin_mm: list[float] | None = None, axis: list[float] | None = None,
                             axis2: list[float] | None = None, faces_a=None, faces_b=None,
                             ratio: float | None = None, lead_mm: float | None = None,
                             joint_a: str | None = None, joint_b: str | None = None,
                             point_a_mm: list[float] | None = None, point_b_mm: list[float] | None = None,
                             limits: list | None = None, limit_stiffness: float | None = None,
                             restitution: float = 0.5, friction: dict | None = None) -> str:
        """Adds/replaces a joint between body_a and body_b ("" = world). kind: fixed | revolute | prismatic | cylindrical | pin_slot (axis2 = slot direction) | planar (axis = normal) | spherical | universal (axis on A, axis2 on B) | screw (lead_mm) | gear (joint_a, joint_b rotary, ratio = w_b/w_a) | rack_pinion (joint_a pinion, joint_b rack, ratio = mm/rad) | distance (point_a_mm - point_b_mm). faces_a/faces_b: joint faces on the bodies (selection | id | query); a cylindrical face fills axis and origin; also used for the FEM transfer. point_b_mm = joint point on body B in the CAD position (!= origin: the mechanism is assembled). limits [lower, upper] (rad | mm from the CAD position) = stops; limit_stiffness (None = from inertia, impact ~1 ms), restitution. friction {model coulomb|stribeck|viscous|bearing|thread, mu_k, mu_s, r_eff mm, fit clearance|worn|fitted, c_v, v_s, v_reg mm/s, preload N}: torque mu'*|R|*r from the previous step's reaction."""
        from ..motion.model import JOINT_KINDS, MotionJoint, JointFriction
        if kind not in JOINT_KINDS:
            raise FeaError(f"kind must be one of: {', '.join(JOINT_KINDS)}")
        with sess.use():
            st = sess.require_study()
            mech = st.mechanism
            names = [b.name for b in mech.bodies]
            for b in (body_a, body_b):
                if b and b not in names:
                    raise FeaError(f"Body '{b}' does not exist (bodies: {', '.join(names)})")
            old = [j for j in mech.joints if j.name == name]          # the joint being replaced may be changed by selection
            sel_a = guarded_selection(sess, faces_a, f"{name} – A", old) if faces_a is not None else None
            sel_b = guarded_selection(sess, faces_b, f"{name} – B", old) if faces_b is not None else None
            o, ax = _vec(origin_mm, "origin_mm"), _vec(axis, "axis")
            notes = []
            if (o is None or ax is None) and kind not in ("gear", "rack_pinion", "distance"):
                cyl = None
                for s in (sel_a, sel_b):
                    if s and cyl is None:
                        cyl = _axis_from_faces(sess, s)
                if cyl is None:
                    raise FeaError("Give origin_mm and axis, or the cylindrical faces faces_a / faces_b (axis from CAD)")
                p, d, r = cyl
                if ax is None:
                    ax = [float(x) for x in d]
                if o is None:
                    pts = [np.asarray(f.centroid, float) for s in (sel_a, sel_b) if s for f in st.selections[s].faces]
                    o = [float(x) for x in _project_on_axis(p, d, pts)]
                notes.append(f"axis and origin from a cylindrical face (radius {r:.4g} mm)")
            fr = None
            if friction:
                bad = set(friction) - {f for f in JointFriction.__dataclass_fields__}
                if bad:
                    raise FeaError(f"Unknown friction keys: {', '.join(sorted(bad))}")
                fr = JointFriction.from_dict(friction)
            j = MotionJoint(name, kind, body_a, body_b, o or [0.0, 0.0, 0.0], ax or [0.0, 0.0, 1.0],
                            axis2=_vec(axis2, "axis2"), selection_a=sel_a, selection_b=sel_b,
                            ratio=None if ratio is None else float(ratio), lead=None if lead_mm is None else float(lead_mm),
                            joint_a=joint_a, joint_b=joint_b, point_a=_vec(point_a_mm, "point_a_mm"),
                            point_b=_vec(point_b_mm, "point_b_mm"), limits=limits, friction=fr,
                            limit_stiffness=limit_stiffness, restitution=float(restitution))
            mech.joints = [x for x in mech.joints if x.name != name] + [j]
            _changed(sess)
            return out({"joint": j.to_dict(), "notes": notes, "next": "fea_motion_check_dof"})

    @mcp.tool(structured_output=False)
    def fea_motion_joint_list() -> str:
        """Joints, force elements, drivers, loads of the mechanism."""
        with sess.use():
            m = _mech(sess)
            return out({"joints": [j.to_dict() for j in m.joints], "forces": [vars(f) for f in m.forces],
                        "drivers": [vars(d) for d in m.drivers], "loads": [vars(x) for x in m.loads],
                        "gravity_mm_s2": m.gravity, "run": {k: v for k, v in vars(m.run).items() if k != "to_fea"}})

    @mcp.tool(structured_output=False)
    def fea_motion_joint_remove(name: str) -> str:
        """Removes a joint (with its drivers, torsion springs and gear couplings)."""
        with sess.use():
            m = _mech(sess)
            if name not in [j.name for j in m.joints]:
                raise FeaError(f"Joint '{name}' does not exist")
            m.joints = [j for j in m.joints if j.name != name and name not in (j.joint_a, j.joint_b)]
            m.drivers = [d for d in m.drivers if d.joint != name]
            m.forces = [f for f in m.forces if f.joint != name]
            _changed(sess)
            return out({"removed": name, "joints": [j.name for j in m.joints]})

    @mcp.tool(structured_output=False)
    def fea_motion_remove(kind: str, name: str) -> str:
        """Removes kind 'force' (force element), 'driver' (driver or joint name), 'load' or 'contact' (name or "A-B")."""
        with sess.use():
            m = _mech(sess)
            if kind == "contact":
                before = len(m.contacts)
                m.contacts = [c for c in m.contacts if name not in (c.name, f"{c.body_a}–{c.body_b}",
                                                                     f"{c.body_b}–{c.body_a}")]
                if len(m.contacts) == before:
                    raise FeaError(f"'{name}' not found")
                _changed(sess)
                return out({"removed": name, "kind": kind})
            if kind == "force":
                before = len(m.forces)
                m.forces = [f for f in m.forces if f.name != name]
            elif kind == "driver":
                before = len(m.drivers)
                m.drivers = [d for d in m.drivers if (d.name or d.joint) != name and d.joint != name]
            elif kind == "load":
                before = len(m.loads)
                m.loads = [x for x in m.loads if x.name != name]
            else:
                raise FeaError("kind must be force | driver | load | contact")
            after = len({"force": m.forces, "driver": m.drivers, "load": m.loads}[kind])
            if after == before:
                raise FeaError(f"'{name}' not found")
            _changed(sess)
            return out({"removed": name, "kind": kind})

    @mcp.tool(structured_output=False)
    def fea_motion_joints_from_cad(accept: bool = False, min_confidence: float = 0.7,
                                   metadata_json: str | None = None) -> str:
        """Proposes joints from geometry: coaxial cylindrical faces of different bodies (pin in hole, same radius, axial overlap) -> revolute (with axial support) or cylindrical; clusters along an axis = separate joints (2 hinges = redundancy). metadata_json: Fusion metadata with key "joints" (fea_core.motion.joints_cad). Returns proposals with certainty and reason; accept=True creates those >= min_confidence (face selections + joints)."""
        from ..motion import joints_cad
        with sess.use(op="joint proposal"):
            st = sess.require_study()
            if len(st.mechanism.bodies) < 2:
                raise FeaError("First define at least 2 bodies (fea_motion_body_add)")
            sug = joints_cad.suggest_from_geometry(st, sess.parts)
            if metadata_json:
                import json
                try:
                    meta = json.loads(user_file(metadata_json, 16 << 20, "Metadata").read_text(encoding="utf-8"))
                except (OSError, ValueError) as e:
                    raise FeaError(f"The metadata cannot be loaded: {e}") from None
                sug += joints_cad.suggest_from_metadata(st, meta)
            applied = []
            if accept:
                for s in sug:
                    if s.confidence >= min_confidence:
                        j = joints_cad.apply_suggestion(st, sess.parts, s)
                        applied.append(j.name)
                if applied:
                    _changed(sess)
            return out({"suggestions": [s.to_dict() for s in sug], "accepted": applied,
                        "next": "fea_motion_check_dof" if applied else "accept=True or fea_motion_joint_add"})

    @mcp.tool(structured_output=False)
    def fea_motion_force_add(name: str, kind: str, body_a: str = "", body_b: str = "",
                             point_a_mm: list[float] | None = None, point_b_mm: list[float] | None = None,
                             k: float = 0.0, c: float = 0.0, f0: float = 0.0, l0: float | None = None,
                             joint: str | None = None, table: list | None = None, k6: list[float] | None = None,
                             c6: list[float] | None = None, faces_a=None, faces_b=None) -> str:
        """Force element. spring | damper | spring_damper: F = f0 + k*(L - l0) + c*dL/dt [N, N/mm, N*s/mm], tension > 0, l0 None = CAD length | gas_spring (table [[compression mm, force N], ...]) | torsion (on `joint`: M = f0 + k*(phi - l0) + c*dphi/dt) | bushing (k6, c6 = 3 translational [N/mm] + 3 rotational [N*mm/rad] at point_a_mm). faces_a/faces_b: attachment faces (FEM transfer)."""
        from ..motion.model import FORCE_KINDS, ForceElement
        if kind not in FORCE_KINDS:
            raise FeaError(f"kind must be one of: {', '.join(FORCE_KINDS)}")
        with sess.use():
            st = sess.require_study()
            m = st.mechanism
            if kind == "torsion" and joint not in [j.name for j in m.joints]:
                raise FeaError("A torsion spring needs an existing revolute joint (joint)")
            if kind != "torsion" and (point_a_mm is None or point_b_mm is None) and kind != "bushing":
                raise FeaError("Give point_a_mm and point_b_mm (the attachment points of the element)")
            old = [x for x in m.forces if x.name == name]
            sa = guarded_selection(sess, faces_a, f"{name} – A", old) if faces_a is not None else None
            sb = guarded_selection(sess, faces_b, f"{name} – B", old) if faces_b is not None else None
            f = ForceElement(name, kind, body_a, body_b, _vec(point_a_mm, "point_a_mm") or [0.0] * 3,
                             _vec(point_b_mm, "point_b_mm") or _vec(point_a_mm, "point_a_mm") or [0.0] * 3,
                             float(k), float(c), float(f0), l0, joint, table, k6, c6, sa, sb)
            m.forces = [x for x in m.forces if x.name != name] + [f]
            _changed(sess)
            return out({"force": vars(f)})

    @mcp.tool(structured_output=False)
    def fea_motion_driver_add(joint: str, quantity: str = "angle", law: str = "constant", data: list | None = None,
                              name: str = "") -> str:
        """Joint driver. quantity: angle [rad] | disp [mm] | angular_velocity [rad/s] | velocity [mm/s] (kinematic, from the CAD position) | torque [N*mm] | force [N] (kinetic). law: constant [value] | table [[t, value], ...] | sine [amplitude, f Hz, phase rad, mean] | poly [c0, c1, ...] | scurve [stroke, v_max, a_max, j_max, t_start] (jerk-limited axis profile)."""
        from ..motion.laws import Law, LawError
        from ..motion.model import DRIVER_QUANTITIES, Driver
        if quantity not in DRIVER_QUANTITIES:
            raise FeaError(f"quantity must be one of: {', '.join(DRIVER_QUANTITIES)}")
        try:
            Law(law, data if data is not None else [0.0])
        except LawError as e:
            raise FeaError(str(e)) from None
        with sess.use():
            m = _mech(sess)
            if joint not in [j.name for j in m.joints]:
                raise FeaError(f"Joint '{joint}' does not exist")
            d = Driver(joint, quantity, law, list(data if data is not None else [0.0]), name or joint)
            m.drivers = [x for x in m.drivers if (x.name or x.joint) != d.name] + [d]
            _changed(sess)
            return out({"driver": vars(d), "drivers": [x.name or x.joint for x in m.drivers]})

    @mcp.tool(structured_output=False)
    def fea_motion_load_add(body: str, point_mm: list[float], force_N: list[float] | None = None,
                            moment_Nmm: list[float] | None = None, table: list | None = None, name: str = "",
                            faces=None) -> str:
        """External force/moment on a body at a point (global axes), time course = table [[t, k], ...] multiplier (None = constant). faces: face the load acts on (FEM transfer)."""
        from ..motion.model import MotionLoad
        with sess.use():
            st = sess.require_study()
            m = st.mechanism
            if body not in [b.name for b in m.bodies]:
                raise FeaError(f"Body '{body}' does not exist")
            nm = name or f"load {len(m.loads) + 1}"
            old = [x for x in m.loads if x.name == nm]
            sel = guarded_selection(sess, faces, nm, old) if faces is not None else None
            ld = MotionLoad(body, _vec(point_mm, "point_mm"), _vec(force_N, "force_N") or [0.0] * 3,
                            _vec(moment_Nmm, "moment_Nmm") or [0.0] * 3, table, nm, sel)
            m.loads = [x for x in m.loads if x.name != nm] + [ld]
            _changed(sess)
            return out({"load": vars(ld)})

    @mcp.tool(structured_output=False)
    def fea_motion_contact_add(body_a: str, body_b: str, mu: float = 0.3, restitution: float = 0.0,
                               rolling_mm: float = 0.0, method: str = "nsc", young_MPa: float | None = None,
                               deflection_mm: float = 0.01, name: str = "") -> str:
        """Body-body contact (Chrono solver only: fea_motion_settings(solver='chrono'), module "chrono"): shapes = triangulated CAD parts (deflection_mm chordal deviation); method nsc (hard, default) | smc (penalty, young_MPa None = part E); mu, restitution 0-1, rolling_mm. Contact material belongs to the Chrono body (smaller friction wins; conflicts warn). Series "kontaktní síla:<body>", "kontaktní moment:<body>" (about CoG). Verified MB4C/MB4R, KK5."""
        from ..motion.model import MotionContact
        if method not in ("nsc", "smc"):
            raise FeaError("method must be nsc | smc")
        if mu < 0 or not 0 <= restitution <= 1 or deflection_mm <= 0:
            raise FeaError("mu ≥ 0, restitution 0–1, deflection_mm > 0")
        with sess.use():
            m = _mech(sess)
            names = {b.name: b for b in m.bodies}
            for bn in (body_a, body_b):
                if bn not in names:
                    raise FeaError(f"Body '{bn}' does not exist")
                if not names[bn].parts:
                    raise FeaError(f"Body '{bn}' has no parts from CAD – a contact needs a shape")
            if body_a == body_b:
                raise FeaError("A contact connects two different bodies")
            nm = name or f"{body_a}–{body_b}"
            c = MotionContact(body_a, body_b, float(mu), float(restitution), float(rolling_mm), method,
                              None if young_MPa is None else float(young_MPa), float(deflection_mm), nm)
            m.contacts = [x for x in m.contacts if (x.name or f"{x.body_a}–{x.body_b}") != nm] + [c]
            _changed(sess)
            from ..motion.chrono_run import available, INSTALL_HINT
            d = {"contact": vars(c), "contacts": [x.name or f"{x.body_a}–{x.body_b}" for x in m.contacts],
                 "solver": m.run.solver, "chrono_available": available()}
            if m.run.solver != "chrono":
                d["next"] = "fea_motion_settings(solver='chrono', kind='dynamic') – only Chrono computes the contact"
            if not d["chrono_available"]:
                d["warning"] = INSTALL_HINT
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_motion_settings(kind: str | None = None, t_end_s: float | None = None, outputs: int | None = None,
                            dt_s: float | None = None, gravity_mm_s2: list[float] | None = None,
                            no_gravity: bool = False, collisions=None, collision_tol_mm: float | None = None,
                            include_extra_mass: bool | None = None, solver: str | None = None,
                            spectral_radius: float | None = None) -> str:
        """Mechanism calculation settings. kind kinematic (all freedoms by drivers; reactions include inertia) | static (quasi-static positions) | dynamic. t_end_s, outputs, dt_s (None = auto), gravity_mm_s2 (default [0,0,-9810]; no_gravity), collisions [[body, body], ...] | "all" (no contact, check only), include_extra_mass, spectral_radius (generalized-alpha rho_inf; 1 = no damping). solver: exudyn (default; rigid+flexible, joint friction, stops, gears) | chrono (CAD-shape CONTACT, fea_motion_contact_add; dynamic/kinematic, rigid bodies, frictionless joints, no stops; implicit Euler needs dt_s 1e-4..2.5e-4; optional module)."""
        with sess.use():
            m = _mech(sess)
            r = m.run
            if solver is not None:
                if solver not in ("exudyn", "chrono"):
                    raise FeaError("solver must be exudyn | chrono")
                if solver == "chrono":
                    from ..motion.chrono_run import available, INSTALL_HINT
                    if not available():
                        raise FeaError(INSTALL_HINT)
                r.solver = solver
            if spectral_radius is not None:
                if not 0 <= spectral_radius <= 1:
                    raise FeaError("spectral_radius 0–1")
                r.spectral_radius = float(spectral_radius)
            if kind is not None:
                if kind not in ("kinematic", "static", "dynamic"):
                    raise FeaError("kind must be kinematic | static | dynamic")
                r.kind = kind
            if t_end_s is not None:
                if not t_end_s > 0:
                    raise FeaError("t_end_s must be positive")
                r.t_end = float(t_end_s)
            if outputs is not None:
                if not 1 <= int(outputs) <= 100000:
                    raise FeaError("outputs 1–100000")
                r.outputs = int(outputs)
            if dt_s is not None:
                r.dt = float(dt_s) if dt_s > 0 else None
            if no_gravity:
                m.gravity = None
            elif gravity_mm_s2 is not None:
                m.gravity = _vec(gravity_mm_s2, "gravity_mm_s2")
            if collisions is not None:
                r.collisions = collisions if collisions else None
            if collision_tol_mm is not None:
                r.collision_tol = float(collision_tol_mm)
            if include_extra_mass is not None:
                r.include_extra_mass = bool(include_extra_mass)
            _changed(sess)
            from ..motion.chrono_run import available
            return out({"run": {k: v for k, v in vars(r).items() if k != "to_fea"}, "gravity_mm_s2": m.gravity,
                        "contacts": [vars(c) for c in m.contacts], "chrono_available": available(),
                        "flexible_bodies": [b.name for b in m.bodies if b.flexible and not b.ground]})

    @mcp.tool(structured_output=False)
    def fea_motion_check_dof(timeout_s: float = 120) -> str:
        """Freedom check: Grubler count, real DOF from the constraint Jacobian rank (Exudyn), redundant joint constraints (reactions then AMBIGUOUS, with a fix proposal), DOF left after drivers (kinematic needs 0)."""
        from ..motion.run import check_dof, MotionError
        with sess.use(op="mobility check", wait=5):
            st = sess.require_study()
            parts = None if sess.frame_only else sess.parts
            wd = sess.workdir or sess.new_workdir(st.name)

            def work(cancel):
                return check_dof(st, parts, workdir=wd)
            try:
                d = run_with_timeout(work, timeout_s, "Mobility check")
            except MotionError as e:
                raise FeaError(str(e)) from None
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_motion_run(timeout_s: float = 600) -> str:
        """Runs the mechanism (fea_motion_settings) in a separate process (Exudyn, or Chrono). Checks body equilibrium at each time (sum F = m*a, sum M = dH/dt; Chrono: mismatch ~dt only warns) and the power balance (Exudyn rigid bodies); collisions per settings. Flexible bodies: mesh + Craig-Bampton first (NGSolve), summary "flexible". Returns max joint reactions, driver moments, contact forces, energies, warnings (redundancy, collisions)."""
        from ..motion.run import run_motion, MotionError
        with sess.use(op="mechanism calculation", wait=5):
            st = sess.require_study()
            parts = None if sess.frame_only else sess.parts
            wd = sess.workdir or sess.new_workdir(st.name)
            logs: list[str] = []

            def work(cancel):
                return run_motion(st, wd, parts=parts, on_log=logs.append, cancel=cancel, timeout=timeout_s)
            t0 = time.time()
            try:
                res = run_with_timeout(work, timeout_s + 30, "Mechanism calculation")
            except MotionError as e:
                raise FeaError(str(e)) from None
            finally:
                for line in logs:
                    sess.add_log(line)
            sess.motion = res
            d = res.summary()
            d["time_s"] = time.time() - t0
            d["next"] = ("fea_motion_results(series=['reaction:<joint>']), fea_motion_render, "
                         "fea_motion_export_loads(mode='max' | 'time' | 'history')")
            return out(d)

    def _require():
        if sess.motion is None:
            raise FeaError("There is no mechanism result – run fea_motion_run (it is discarded after a change of the mechanism)")
        return sess.motion

    @mcp.tool(structured_output=False)
    def fea_motion_results(series: list[str] | None = None, find_max: str | None = None, max_points: int = 60,
                           time_s: float | None = None) -> str:
        """Mechanism results: summary; series = ["quantity:item", ...] resampled to max_points. Quantities (item): position, velocity, acceleration, angular velocity, angular acceleration (body); reaction, joint moment, reaction in joint frame, moment in joint frame, angle, translation, joint velocity, friction, stop (joint); drive, drive moment, drive force, drive power (driver); element force, element length (force element); energy, work; distance:A|B; stress, deformation (flexible body: max vM [MPa], max |u| [mm]); contact force, contact moment (body, Chrono). Czech names of older versions (reakce, pohon, energie…) are accepted. find_max = quantity -> instant of its maximum; time_s -> all joints' state at that instant (global and joint-frame reactions)."""
        with sess.use():
            res = _require()
            d = res.summary()
            if series:
                d["series"] = {}
                for s in series:
                    try:
                        ser = res.series(s)
                    except KeyError as e:
                        raise FeaError(str(e).strip("'\"")) from None
                    n = len(ser["t"])
                    idx = np.unique(np.linspace(0, n - 1, min(max(int(max_points), 2), n)).astype(int))
                    rec = {"unit": ser["unit"], "labels": ser["labels"], "t_s": ser["t"][idx].tolist(),
                           "values": ser["values"][idx].tolist(), "max_abs": float(np.abs(ser["values"]).max())}
                    if "magnitude" in ser:
                        rec["magnitude_max"] = float(ser["magnitude"].max())
                    d["series"][s] = rec
            if find_max:
                try:
                    d["maximum"] = res.find_max(find_max)
                except KeyError as e:
                    raise FeaError(str(e).strip("'\"")) from None
            if time_s is not None:
                k = res.time_index(time_s)
                a = res.arrays
                d["instant"] = {"t_s": float(a["t"][k]), "joints": [
                    {"joint": j, "force_N": a["jF"][k, i].tolist(), "moment_Nmm": a["jM"][k, i].tolist(),
                     "force_local_N": a["jFl"][k, i].tolist(), "moment_local_Nmm": a["jMl"][k, i].tolist(),
                     "point_mm": a["jP"][k, i].tolist()} for i, j in enumerate(res.joint_names)],
                    "bodies": [{"body": b, "position_mm": a["pos"][k, i].tolist(), "acc_mm_s2": a["acc"][k, i].tolist(),
                                "omega_rad_s": a["omega"][k, i].tolist(), "alpha_rad_s2": a["alpha"][k, i].tolist()}
                               for i, b in enumerate(res.body_names)]}
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_motion_render(quantity: str = "energy", output_path: str | None = None, time_s: float | None = None,
                          width: int = 1280, height: int = 720, return_image: bool = True):
        """PNG graph of a mechanism quantity vs time; quantity as the series in fea_motion_results ("reaction:Joint 1", "drive moment:Motor", "energy"); vectors show components + magnitude. quantity='scheme' (or 'poloha') = mechanism scheme (joint points, CoGs in the plane of largest spread) at time_s."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from .tools_results import _png_path
        with sess.use(op="render"):
            res = _require()
            fig = Figure(figsize=(width / 100, height / 100), dpi=100)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            meta = {"quantity": quantity}
            if quantity.strip().lower() in ("scheme", "poloha", "location"):
                a = res.arrays
                k = res.time_index(time_s if time_s is not None else 0.0)
                P = np.concatenate([a["pos"][:, :, :].reshape(-1, 3), a["jP"].reshape(-1, 3)])
                spread = np.ptp(P, axis=0)
                i0, i1 = np.argsort(-spread)[:2]
                lab = "xyz"
                for j, J in enumerate(res.meta["joints"]):
                    for side in ("a", "b"):
                        bi = J[side]
                        if bi is None or bi < 0 or res.meta["bodies"][bi]["ground"]:
                            continue
                        c = a["pos"][k, bi]
                        p = a["jP"][k, j] if side == "b" else a["jPa"][k, j]
                        ax.plot([c[i0], p[i0]], [c[i1], p[i1]], "-", color="#4C72B0", lw=2)
                ax.plot(a["jP"][k, :, i0], a["jP"][k, :, i1], "o", color="#C44E52", label="hinges")
                ax.plot(a["pos"][k, :, i0], a["pos"][k, :, i1], "s", color="#55A868", label="centre of mass")
                for i, b in enumerate(res.body_names):
                    ax.annotate(b, (a["pos"][k, i, i0], a["pos"][k, i, i1]), fontsize=8)
                ax.set_aspect("equal")
                ax.set_xlabel(f"{lab[i0]} [mm]")
                ax.set_ylabel(f"{lab[i1]} [mm]")
                ax.set_title(f"Mechanism position, t = {a['t'][k]:.4g} s")
                ax.legend()
                meta["t_s"] = float(a["t"][k])
            else:
                try:
                    s = res.series(quantity)
                except KeyError as e:
                    raise FeaError(str(e).strip("'\"")) from None
                for c in range(s["values"].shape[1]):
                    ax.plot(s["t"], s["values"][:, c], lw=1.2, label=s["labels"][c] if c < len(s["labels"]) else str(c))
                if "magnitude" in s:
                    ax.plot(s["t"], s["magnitude"], "k--", lw=1.0, label="magnitude")
                ax.set_xlabel("t [s]")
                ax.set_ylabel(f"{quantity} [{s['unit']}]")
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_title(quantity)
                meta["max_abs"] = float(np.abs(s["values"]).max())
            p = _png_path(sess, output_path, "motion")
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p)
            meta["path"] = str(p)
            txt = out(meta)
            return [txt, Image(path=p)] if return_image else txt

    @mcp.tool(structured_output=False)
    def fea_motion_export_loads(mode: str = "max", times_s: list[float] | None = None, quantity: str | None = None,
                                body: str | None = None, parts: list[str] | None = None,
                                support_joints: list[str] | None = None, load_kind: str = "remote_distributing") -> str:
        """Transfers the mechanism result to FEM for one body (`body`; points or `parts`). mode 'time' (times_s) | 'max' (instant of the maximum of `quantity`, e.g. 'reakce:Kloub 1') | 'history' (whole course -> unit states: 6 RBE3 remote-load components per joint + 12 inertia states; fatigue channels = component courses in body axes -> Study.fatigue; no BearingLoad - not superposable). An instant = load state "MBD t = ... s": joint reactions as distributing RemoteLoad on joint faces, inertia and weight as Inertia; support = first joint (support_joints) as distributing RemoteSupport (6 components; FEM reaction should equal the MBD one = inertia relief); more support_joints = redundant constraints. load_kind 'bearing' = instants as BearingLoad (+ RBE3 moment). Then fea_solve_run."""
        from ..motion.to_fea import export_loads, ToFeaError
        mmap = {"time": "instants", "instants": "instants", "max": "max", "history": "history"}
        if mode not in mmap:
            raise FeaError("mode must be time | max | history")
        with sess.use():
            st = sess.require_study()
            res = _require()
            try:
                info = export_loads(st, res, mmap[mode], times_s, quantity, body, parts, support_joints, load_kind)
            except (ToFeaError, KeyError) as e:
                raise FeaError(str(e).strip("'\"")) from None
            motion = sess.motion
            sess.changed()                         # new load states → the FEM results do not hold
            sess.motion = motion
            if mode == "history":
                info["estimate"] = (f"{info['n_states']} unit states; time ≈ states × the time of one CalculiX state (remote support → CalculiX, parallel steps)")
            info["next"] = "fea_solve_run" + (" → fea_fatigue_run" if mode == "history" else "")
            return out(info)
