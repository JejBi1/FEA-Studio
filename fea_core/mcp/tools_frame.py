"""Tools: bar system (frames, trusses, curved bars) – Study.frame (fea_core.beams.model).

A study without geometry (fea_study_new without step_path) = bars only (calculation beams.run.run_frame, own solver with the exact
Timoshenko element beams.timoshenko; FrameSettings.solver = "ccx_u1" = the U1 element in ccx);
with a STEP it is a mixed model (bars + bodies, connection of a bar node to a face of a body, always ccx U1). The loads of the bars are in the
load states (FrameNodeLoad, FrameMemberLoad, FrameTemperature); the self-weight (fea_load_add_gravity) acts on the bars too. Results
StepResult.frame (internal forces along the bars, reactions, node displacements), assessment EN 1993-1-1 beams.assess.
Axes of a bar (EN 1993): x = axis from the start node to the end, z = "up" in the section (web of an I profile), y = z × x;
up = vector whose projection perpendicular to the axis gives z (default global Z; for a vertical bar global X).
"""
from __future__ import annotations

import math
import time

import numpy as np
from mcp.server.mcpserver import Image

from .common import guarded_selection, out
from .frame_scenes import member_lengths
from .session import FeaError, Session, user_path

LOCAL = ("N", "Vy", "Vz", "T", "My", "Mz")
DOFS = ("ux", "uy", "uz", "rx", "ry", "rz")
CHECK_KEYS = {"Lcr_y": "buckling length about y [mm]", "Lcr_z": "buckling length about z [mm]",
              "ky": "buckling length factor y", "kz": "buckling length factor z",
              "L_LT": "length between lateral-torsional restraints [mm]", "C1": "moment diagram factor C1",
              "zg": "load application point above the shear centre [mm] (+ destabilizing)", "k_LT": "lateral-torsional buckling factor k",
              "Mcr": "given Mcr [N·mm]", "lt_restrained": "continuously restrained against lateral-torsional buckling",
              "sway_y": "buckling with sway y (Cmy = 0.9)", "sway_z": "buckling with sway z",
              "fabrication": "rolled | welded | hot_finished | cold_formed | auto", "grade": "S235 … S460"}


def _vec3(v, what: str) -> list[float]:
    try:
        a = [float(x) for x in v]
    except (TypeError, ValueError):
        raise FeaError(f"{what} must be [x, y, z]") from None
    if len(a) != 3 or not all(math.isfinite(x) for x in a):
        raise FeaError(f"{what} must be [x, y, z] (3 finite numbers)")
    return a


def flags(v, names: tuple, what: str, presets: dict | None = None) -> list[bool]:
    """6 flags from the notation: 6 bools | list of names | text "a, b" | preset."""
    presets = presets or {}
    if v is None:
        return [False] * 6
    if isinstance(v, str):
        key = v.strip().lower()
        if key in presets:
            return list(presets[key])
        v = [x.strip() for x in v.replace(";", ",").split(",") if x.strip()]
    if isinstance(v, (list, tuple)) and len(v) == 6 and all(isinstance(x, bool) for x in v):
        return [bool(x) for x in v]
    if isinstance(v, (list, tuple)):
        low = [n.lower() for n in names]
        outf = [False] * 6
        for x in v:
            if not isinstance(x, str) or x.strip().lower() not in low:
                raise FeaError(f"{what}: unknown component {x!r} (allowed: {', '.join(names)}"
                               + (f"; presets {', '.join(presets)}" if presets else "") + ")")
            outf[low.index(x.strip().lower())] = True
        return outf
    raise FeaError(f"{what}: give 6 true/false values, a list of components ({', '.join(names)}) or a preset")


SUPPORT_PRESETS = {"fixed": [True] * 6, "vetknuti": [True] * 6, "all": [True] * 6,
                   "pinned": [True, True, True, False, False, False], "hinge": [True, True, True, False, False, False],
                   "none": [False] * 6, "vetknutí": [True] * 6, "kloub": [True, True, True, False, False, False]}
RELEASE_PRESETS = {"hinge": [False, False, False, False, True, True],
                   "none": [False] * 6, "": [False] * 6, "kloub": [False, False, False, False, True, True]}


def releases_text(rel) -> str:
    on = [n for n, r in zip(LOCAL, rel) if r]
    return ", ".join(on) if on else "–"


def member_dict(study, m, L: float | None = None) -> dict:
    d = {"name": m.name, "start": m.start, "end": m.end, "section": m.section,
         "material": m.material or study.default_material, "kind": m.kind}
    if L is not None:
        d["length_mm"] = L
    if m.up is not None:
        d["up"] = m.up
    if m.roll:
        d["roll_deg"] = m.roll
    if m.through is not None:
        d["through_mm"] = m.through
    if any(m.releases_start):
        d["releases_start"] = releases_text(m.releases_start)
    if any(m.releases_end):
        d["releases_end"] = releases_text(m.releases_end)
    if m.segments:
        d["segments"] = m.segments
    if m.name in study.frame.checks:
        d["checks"] = study.frame.checks[m.name]
    return d


def frame_load_text(l) -> str:
    if l.kind == "frame_temperature":
        return (f"{'member ' + ', '.join(l.members) if l.members else 'all members'}: ΔT = {l.dT:g} K"
                + (f", gradient in the z direction {l.grad_z:g} K" if l.grad_z else "")
                + (f", gradient in the y direction {l.grad_y:g} K" if l.grad_y else ""))
    if l.kind == "frame_node":
        return (f"node {l.node}: F = [{', '.join(f'{v:g}' for v in l.force)}] N"
                + (f", M = [{', '.join(f'{v:g}' for v in l.moment)}] N·mm" if any(l.moment) else ""))
    q = f"[{', '.join(f'{v:g}' for v in l.q)}]"
    q2 = "" if l.q_end is None else f" → [{', '.join(f'{v:g}' for v in l.q_end)}]"
    return f"member {l.member}: q = {q}{q2} N/mm ({'local axes x, y, z' if l.axes == 'local' else 'global axes'})"


def frame_overview(sess: Session) -> dict:
    st = sess.require_study()
    fr = st.frame
    Ls = member_lengths(st) if fr.members else {}
    secs = {}
    for k, sd in fr.sections.items():
        try:
            p = sd.resolve()
            secs[k] = {"source": sd.library or sd.shape or "custom", "A_mm2": p.A, "Iy_mm4": p.Iy, "Iz_mm4": p.Iz,
                       "It_mm4": p.J, "Wel_y_mm3": p.Wel_y, "Wel_z_mm3": p.Wel_z}
        except Exception as e:  # noqa: BLE001
            secs[k] = {"error": str(e)}
    loads = []
    for lc in st.load_cases:
        for l in lc.loads:
            if l.kind in ("frame_node", "frame_member", "frame_temperature"):
                loads.append({"case": lc.name, "kind": l.kind, "description": frame_load_text(l)})
    from .frame_scenes import dofs_label
    return {"frame_only": sess.frame_only,
            "nodes": {k: n.xyz for k, n in fr.nodes.items()},
            "members": [member_dict(st, m, Ls.get(m.name)) for m in fr.members.values()],
            "sections": secs,
            "supports": [{"node": s.node, "fixed": [n for n, on in zip(DOFS, s.dofs) if on],
                          "stiffness": {n: k for n, k in zip(DOFS, s.stiffness) if k}, "description":
                          dofs_label(s.dofs, s.stiffness)} for s in fr.supports],
            "connections": [{"node": c.node, "selection": c.selection, "coupling": c.coupling} for c in fr.connections],
            "loads": loads, "settings": fr.settings.__dict__, "checks": fr.checks,
            "total_length_mm": float(sum(Ls.values())),
            "units": "mm, N, N/mm (distributed load), N·mm; member axes x (from start), y, z (up)"}


def _check_build(sess: Session, undo) -> None:
    """The discretisation (beams.fem.build) must pass – otherwise it returns the change (undo) and reports the error."""
    from .frame_scenes import fe_of
    st = sess.study
    if not st.frame.members:
        return
    try:
        fe_of(st)
    except FeaError:
        undo()
        raise


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_frame_section_set(name: str, library: str | None = None, shape: str | None = None,
                              dims: dict | None = None) -> str:
        """Section of the study's bars: library name ("IPE 200", "HE 200 B", "L50x5", "CHS 76.1x3.6", "RHS 100x50x4"; see fea_section_search) OR shape + dims (RECT b h | CIRC d | TUBE D t | RHS h b t | I h b tw tf r | T | U | L; mm). The same name replaces. Returns properties. A library rectangle has strong y along the longer side."""
        from ..beams.model import SectionDef
        from .tools_beams import section_dict
        if (library is None) == (shape is None):
            raise FeaError("Give library (profile name) OR shape + dims")
        with sess.use():
            st = sess.require_study()
            sd = SectionDef(name, library=library, shape=shape, dims=dict(dims or {}))
            try:
                sec = sd.resolve()
            except KeyError as e:
                raise FeaError(f"The parametric section is missing the dimension {e}") from None
            except (ValueError, TypeError) as e:
                raise FeaError(f"Section '{name}': {e}") from None
            old = st.frame.sections.get(name)
            st.frame.sections[name] = sd
            sess.changed()
            return out({"section": name, "replaced": old is not None, "props": section_dict(sec),
                        "used_by": [m.name for m in st.frame.members.values() if m.section == name]})

    @mcp.tool(structured_output=False)
    def fea_frame_node_set(nodes: dict | list) -> str:
        """Nodes [mm]: {"A1": [0,0,0], "A2": [0,0,5000]} or [{"name","xyz"}]. An existing node is moved (bars follow)."""
        items = nodes.items() if isinstance(nodes, dict) else [(d.get("name"), d.get("xyz")) for d in nodes
                                                               if isinstance(d, dict)]
        items = list(items)
        if not items:
            raise FeaError('nodes: {"name": [x, y, z], …} or [{"name": …, "xyz": [x, y, z]}, …]')
        with sess.use():
            st = sess.require_study()
            fr = st.frame
            old = {k: list(v.xyz) for k, v in fr.nodes.items()}
            created, moved = [], []
            for name, xyz in items:
                if not name or not isinstance(name, str):
                    raise FeaError("A node must have a name (text)")
                p = _vec3(xyz, f"Node '{name}'")
                (moved if name in fr.nodes else created).append(name)
                fr.add_node(name, p)

            def undo():
                for k in created:
                    fr.nodes.pop(k, None)
                for k in moved:
                    fr.add_node(k, old[k])
            _check_build(sess, undo)
            sess.changed()
            return out({"created": created, "moved": moved, "nodes": len(fr.nodes)})

    @mcp.tool(structured_output=False)
    def fea_frame_member_set(name: str, start: str | None = None, end: str | None = None, section: str | None = None,
                             material: str | None = None, up: list[float] | str | None = None,
                             roll_deg: float | None = None, through: list[float] | None = None,
                             releases_start: list | str | None = None, releases_end: list | str | None = None,
                             segments: int | None = None, kind: str | None = None) -> str:
        """Bar start -> end (new: start, end, section required; existing: only changed parameters). section = study section name; material (None = default; catalogue names are added). up = orientation of the section z axis ([x,y,z]; for arcs "normal"|"radial"); roll_deg; through = point on the arc (curved bar start-through-end; [] = straight again); releases_start/releases_end: components N,Vy,Vz,T,My,Mz, 6 booleans, "kloub" (= My,Mz) or "none"; kind "beam"|"truss" (axial force only); segments (0 = auto: straight 20, arc every 5 deg)."""
        from ..beams.model import FrameMember
        if kind is not None and kind not in ("beam", "truss"):
            raise FeaError("kind must be 'beam' or 'truss'")
        if segments is not None and (int(segments) != segments or not 0 <= int(segments) <= 500):
            raise FeaError("segments must be an integer 0–500 (0 = automatic)")
        with sess.use():
            st = sess.require_study()
            fr = st.frame
            old = fr.members.get(name)
            if old is None:
                if not (start and end and section):
                    raise FeaError("A new member needs start, end and section")
                m = FrameMember(name, start, end, section)
            else:
                import copy
                m = copy.deepcopy(old)
            for k, v in (("start", start), ("end", end)):
                if v is not None:
                    if v not in fr.nodes:
                        raise FeaError(f"Node '{v}' does not exist (fea_frame_node_set)")
                    setattr(m, k, v)
            if m.start == m.end:
                raise FeaError("A member must connect two different nodes")
            if section is not None:
                if section not in fr.sections:
                    raise FeaError(f"Section '{section}' does not exist (fea_frame_section_set; sections: {', '.join(fr.sections) or 'none'})")
                m.section = section
            if material is not None:
                if material not in st.materials:
                    from ..materials import by_name
                    try:
                        st.add_material(by_name(material))
                    except KeyError:
                        raise FeaError(f"Material '{material}' is neither in the study nor in the catalogue") from None
                m.material = material
            if up is not None:
                m.up = up if isinstance(up, str) else _vec3(up, "up")
                if isinstance(up, str) and up not in ("normal", "radial"):
                    raise FeaError("up as text only 'normal' or 'radial' (arc)")
            if roll_deg is not None:
                m.roll = float(roll_deg)
            if through is not None:
                m.through = _vec3(through, "through") if len(through) else None
            if releases_start is not None:
                m.releases_start = flags(releases_start, LOCAL, "releases_start", RELEASE_PRESETS)
            if releases_end is not None:
                m.releases_end = flags(releases_end, LOCAL, "releases_end", RELEASE_PRESETS)
            if segments is not None:
                m.segments = int(segments)
            if kind is not None:
                m.kind = kind
            if m.material is None and st.default_material is None:
                raise FeaError("The member has no material – give material or a default material (fea_materials_assign part='*')")
            fr.members[name] = m

            def undo():
                if old is None:
                    fr.members.pop(name, None)
                else:
                    fr.members[name] = old
            _check_build(sess, undo)
            sess.changed()
            return out({"member": member_dict(st, m, member_lengths(st).get(name)), "replaced": old is not None})

    @mcp.tool(structured_output=False)
    def fea_frame_support_set(node: str, dofs: list | str = "fixed", stiffness: list[float] | dict | None = None) -> str:
        """Node support (global axes): dofs "fixed" (all 6) | "pinned" (ux,uy,uz) | list of ux,uy,uz,rx,ry,rz | 6 booleans. stiffness = elastic instead of rigid: 6 values [N/mm x3, N*mm/rad x3] or {"uz": 1000, ...} (component > 0 is elastic). The same node replaces. Nodes where only hinges meet (truss) do not hold rotation."""
        with sess.use():
            st = sess.require_study()
            fr = st.frame
            if node not in fr.nodes:
                raise FeaError(f"Node '{node}' does not exist")
            f = flags(dofs, DOFS, "dofs", SUPPORT_PRESETS)
            k = [0.0] * 6
            if isinstance(stiffness, dict):
                for n_, v in stiffness.items():
                    if n_ not in DOFS:
                        raise FeaError(f"stiffness: unknown component '{n_}' ({', '.join(DOFS)})")
                    k[DOFS.index(n_)] = float(v)
            elif stiffness is not None:
                if len(stiffness) != 6:
                    raise FeaError("stiffness: 6 values (ux uy uz [N/mm], rx ry rz [N·mm/rad]) or {component: k}")
                k = [float(v) for v in stiffness]
            if any(v < 0 or not math.isfinite(v) for v in k):
                raise FeaError("stiffness must be ≥ 0")
            for i in range(6):
                if k[i] > 0:
                    f[i] = False                       # the elastic component is not a rigid constraint
            if not any(f) and not any(k):
                raise FeaError("A support with neither a constrained nor a spring component holds nothing")
            from ..beams.model import FrameSupport
            old = [s for s in fr.supports if s.node == node]
            fr.supports = [s for s in fr.supports if s.node != node] + [FrameSupport(node, f, k)]
            sess.changed()
            from .frame_scenes import dofs_label
            return out({"support": {"node": node, "fixed": [n for n, on in zip(DOFS, f) if on],
                                    "stiffness": {n: v for n, v in zip(DOFS, k) if v},
                                    "description": dofs_label(f, k)}, "replaced": bool(old)})

    @mcp.tool(structured_output=False)
    def fea_frame_connection_set(node: str, selection: str | list | dict, name: str | None = None) -> str:
        """Rigidly connects a bar node to a body face (mixed model, STEP required; *COUPLING/*KINEMATIC). selection: faces (id, query, selection name); name of the new selection (default 'prut_<node>'). The same node replaces."""
        from ..beams.model import FrameConnection
        with sess.use():
            st = sess.require_study()
            if sess.frame_only:
                raise FeaError("The study has no geometry – connection to a solid only in a model with a STEP")
            if node not in st.frame.nodes:
                raise FeaError(f"Node '{node}' does not exist")
            old = [c for c in st.frame.connections if c.node == node]          # the connection being replaced may be changed by selection
            sel = guarded_selection(sess, selection, name or f"prut_{node}", old)
            st.frame.connections = [c for c in st.frame.connections if c.node != node] + [FrameConnection(node, sel)]
            sess.changed()
            from .common import selection_info
            return out({"connection": {"node": node, "selection": sel}, "selection_info": selection_info(sess, sel)})

    @mcp.tool(structured_output=False)
    def fea_frame_remove(kind: str, names: list[str] | str, case_name: str | None = None) -> str:
        """Removes frame items. kind node | member | section | support | connection | node_load | member_load; names (nodes, bars, sections; supports/connections/node_loads use node names; member_load bar names); case_name for loads (None = all states). Nodes/sections in use cannot be deleted; deleting a bar deletes its loads and check settings."""
        names = [names] if isinstance(names, str) else list(names)
        with sess.use():
            st = sess.require_study()
            fr = st.frame
            removed = []
            if kind == "node":
                for n_ in names:
                    if n_ not in fr.nodes:
                        raise FeaError(f"Node '{n_}' does not exist")
                    used = [m.name for m in fr.members.values() if n_ in (m.start, m.end)]
                    if used:
                        raise FeaError(f"Node '{n_}' is used by members {', '.join(used)} – remove them first")
                for n_ in names:
                    fr.nodes.pop(n_)
                    fr.supports = [s for s in fr.supports if s.node != n_]
                    fr.connections = [c for c in fr.connections if c.node != n_]
                    for lc in st.load_cases:
                        lc.loads = [l for l in lc.loads if not (l.kind == "frame_node" and l.node == n_)]
                    removed.append(n_)
            elif kind == "member":
                for n_ in names:
                    if n_ not in fr.members:
                        raise FeaError(f"Member '{n_}' does not exist")
                for n_ in names:
                    fr.members.pop(n_)
                    fr.checks.pop(n_, None)
                    for lc in st.load_cases:
                        lc.loads = [l for l in lc.loads if not (l.kind == "frame_member" and l.member == n_)]
                        for l in lc.loads:                     # the temperature of the nominal bars: remove the bar from the list
                            if l.kind == "frame_temperature" and n_ in l.members:
                                l.members = [x for x in l.members if x != n_]
                    removed.append(n_)
            elif kind == "section":
                for n_ in names:
                    if n_ not in fr.sections:
                        raise FeaError(f"Cross-section '{n_}' does not exist")
                    used = [m.name for m in fr.members.values() if m.section == n_]
                    if used:
                        raise FeaError(f"Section '{n_}' is used by members {', '.join(used)}")
                for n_ in names:
                    fr.sections.pop(n_)
                    removed.append(n_)
            elif kind in ("support", "connection"):
                attr = "supports" if kind == "support" else "connections"
                before = getattr(fr, attr)
                setattr(fr, attr, [x for x in before if x.node not in names])
                removed = [x.node for x in before if x.node in names]
            elif kind in ("node_load", "member_load"):
                k_ = "frame_node" if kind == "node_load" else "frame_member"
                key = "node" if kind == "node_load" else "member"
                for lc in st.load_cases:
                    if case_name is not None and lc.name != case_name:
                        continue
                    keep = []
                    for l in lc.loads:
                        if l.kind == k_ and getattr(l, key) in names:
                            removed.append(f"{lc.name}: {getattr(l, key)}")
                        else:
                            keep.append(l)
                    lc.loads = keep
            else:
                raise FeaError("kind must be node | member | section | support | connection | node_load | member_load")
            if not removed:
                raise FeaError(f"Nothing to remove ({kind}: {', '.join(names)})")
            sess.changed()
            return out({"removed": removed, "kind": kind, "members": len(fr.members), "nodes": len(fr.nodes)})

    @mcp.tool(structured_output=False)
    def fea_frame_settings(segments_straight: int | None = None, max_arc_angle_deg: float | None = None,
                           shear_deformation: bool | None = None, steel_grade: str | None = None,
                           gamma_M0: float | None = None, gamma_M1: float | None = None,
                           solver: str | None = None, segments_static: int | None = None,
                           segments_modal: int | None = None, buckling_torsion: bool | None = None) -> str:
        """Frame settings (none = list): solver "timoshenko" (default own solver, exact element, float64) | "ccx_u1" (comparison only; mixed models with bodies always U1); segments_static (own solver, default 1 - exact statics), segments_modal (frequencies/buckling, 8), buckling_torsion (St. Venant term N*i_p^2, default False), segments_straight (ccx_u1 only, 20), max_arc_angle_deg (5; own solver <= 2), shear_deformation (False = Euler-Bernoulli), steel_grade (S235..S460 for EN 1993-1-1; else by material/Re), gamma_M0, gamma_M1."""
        with sess.use():
            st = sess.require_study()
            s = st.frame.settings
            ch = False
            if solver is not None:
                if solver not in ("timoshenko", "ccx_u1"):
                    raise FeaError("solver must be 'timoshenko' or 'ccx_u1'")
                s.solver, ch = solver, True
            for k_, v_, lim in (("segments_static", segments_static, 50), ("segments_modal", segments_modal, 100)):
                if v_ is not None:
                    if int(v_) != v_ or not 1 <= v_ <= lim:
                        raise FeaError(f"{k_} must be 1–{lim}")
                    setattr(s, k_, int(v_))
                    ch = True
            if buckling_torsion is not None:
                s.buckling_torsion, ch = bool(buckling_torsion), True
            if segments_straight is not None:
                if int(segments_straight) != segments_straight or not 1 <= segments_straight <= 200:
                    raise FeaError("segments_straight must be 1–200")
                s.segments_straight, ch = int(segments_straight), True
            if max_arc_angle_deg is not None:
                if not 0.5 <= max_arc_angle_deg <= 30:
                    raise FeaError("max_arc_angle_deg must be 0.5–30°")
                s.max_arc_angle, ch = float(max_arc_angle_deg), True
            if shear_deformation is not None:
                s.shear_deformation, ch = bool(shear_deformation), True
            if steel_grade is not None:
                from ..beams.ec3 import STEEL_GRADES
                g = steel_grade.strip().upper().replace(" ", "")
                if not g.startswith("S") or not g[1:4].isdigit() or int(g[1:4]) not in STEEL_GRADES:
                    raise FeaError(f"steel_grade: S{' | S'.join(map(str, STEEL_GRADES))}")
                s.steel_grade, ch = g[:4], True
            for k, v in (("gamma_M0", gamma_M0), ("gamma_M1", gamma_M1)):
                if v is not None:
                    if not 0.8 <= v <= 2.0:
                        raise FeaError(f"{k} must be 0.8–2.0")
                    setattr(s, k, float(v))
                    ch = True
            if ch:
                sess.changed()
            return out({"settings": s.__dict__})

    @mcp.tool(structured_output=False)
    def fea_frame_list() -> str:
        """Frame overview: nodes [mm], bars (nodes, section, material, length, orientation, hinges, arc), sections (A, Iy, Iz, It, Wel), supports, body connections, bar loads per state, settings, checks."""
        with sess.use():
            return out(frame_overview(sess))

    @mcp.tool(structured_output=False)
    def fea_frame_load_node(case_name: str, node: str, force: list[float] | None = None,
                            moment: list[float] | None = None) -> str:
        """Concentrated force [N] / moment [N*mm] at a node (global axes). Creates the state if missing; the same node in a state replaces."""
        from ..beams.model import FrameNodeLoad
        from .tools_bc import get_case
        f = _vec3(force, "force") if force is not None else [0.0, 0.0, 0.0]
        m = _vec3(moment, "moment") if moment is not None else [0.0, 0.0, 0.0]
        if not any(f) and not any(m):
            raise FeaError("Enter force or moment")
        with sess.use():
            st = sess.require_study()
            if node not in st.frame.nodes:
                raise FeaError(f"Node '{node}' does not exist")
            lc, created = get_case(st, case_name)
            old = [l for l in lc.loads if l.kind == "frame_node" and l.node == node]
            lc.loads = [l for l in lc.loads if not (l.kind == "frame_node" and l.node == node)] + \
                [FrameNodeLoad(node, f, m)]
            sess.changed()
            return out({"load_case": lc.name, "load": frame_load_text(lc.loads[-1]), "replaced": bool(old),
                        "case_created": created})

    @mcp.tool(structured_output=False)
    def fea_frame_load_member(case_name: str, member: str, q: list[float], q_end: list[float] | None = None,
                              axes: str = "global") -> str:
        """Distributed bar load [N/mm] (1 kN/m = 1 N/mm): q at start, q_end at end (linear; None = uniform), axes "global" (snow [0,0,-q]) or "local" (bar axes). On arcs along the arc length. Creates the state if missing; the same bar replaces. Self-weight: fea_load_add_gravity."""
        from ..beams.model import FrameMemberLoad
        from .tools_bc import get_case
        if axes not in ("global", "local"):
            raise FeaError("axes must be 'global' or 'local'")
        qa = _vec3(q, "q")
        qb = _vec3(q_end, "q_end") if q_end is not None else None
        if not any(qa) and not (qb and any(qb)):
            raise FeaError("The load is zero")
        with sess.use():
            st = sess.require_study()
            if member not in st.frame.members:
                raise FeaError(f"Member '{member}' does not exist")
            lc, created = get_case(st, case_name)
            old = [l for l in lc.loads if l.kind == "frame_member" and l.member == member]
            lc.loads = [l for l in lc.loads if not (l.kind == "frame_member" and l.member == member)] + \
                [FrameMemberLoad(member, qa, qb, axes)]
            L = member_lengths(st).get(member)
            sess.changed()
            d = {"load_case": lc.name, "load": frame_load_text(lc.loads[-1]), "replaced": bool(old),
                 "case_created": created}
            if L:
                qm = np.asarray(qa) if qb is None else 0.5 * (np.asarray(qa) + np.asarray(qb))
                d["resultant_N"] = (qm * L).tolist()
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_frame_load_temperature(case_name: str, dT: float = 0.0, grad_z: float = 0.0, grad_y: float = 0.0,
                                   members: list[str] | None = None) -> str:
        """Temperature load of bars (own solver only): dT [K] uniform (restrained: N = -E*A*alpha*dT), grad_z = T(+z edge) - T(-z edge) [K] (bending My), grad_y likewise (Mz). members (None = all); alpha from the material. Creates the state if missing; the same group replaces."""
        from ..beams.model import FrameTemperature
        from .tools_bc import get_case
        if not (dT or grad_z or grad_y):
            raise FeaError("Enter dT or a gradient")
        with sess.use():
            st = sess.require_study()
            if st.frame.settings.solver == "ccx_u1":
                raise FeaError("Member temperature loads are computed only by the built-in solver (fea_frame_settings(solver='timoshenko'))")
            names = list(members or [])
            for n_ in names:
                if n_ not in st.frame.members:
                    raise FeaError(f"Member '{n_}' does not exist")
            lc, created = get_case(st, case_name)
            old = [l for l in lc.loads if l.kind == "frame_temperature" and set(l.members) == set(names)]
            lc.loads = [l for l in lc.loads if not any(l is o for o in old)] + \
                [FrameTemperature(names, float(dT), float(grad_z), float(grad_y))]
            sess.changed()
            return out({"load_case": lc.name, "load": frame_load_text(lc.loads[-1]), "replaced": bool(old),
                        "case_created": created})

    @mcp.tool(structured_output=False)
    def fea_frame_from_wires(section: str, step_path: str | None = None, tol_mm: float = 1e-3, prefix: str = "") -> str:
        """Bars from the STEP wire model (free edges outside bodies): segments -> straight, arcs -> curved, full circle -> 2 arcs, other curves -> polyline. step_path (None = study STEP); end points within tol_mm merged into nodes (also existing). All get `section` (change via fea_frame_member_set). Add supports and loads afterwards."""
        from pathlib import Path
        from ..beams.geometry import wire_edges, frame_from_wires
        with sess.use(op="members from the wire model"):
            st = sess.require_study()
            path = user_path(step_path) if step_path else sess.step_path
            if path is None or not Path(path).exists():
                raise FeaError(f"STEP file not found: {path}")
            if section not in st.frame.sections:
                raise FeaError(f"Section '{section}' does not exist (fea_frame_section_set)")
            if st.default_material is None:
                raise FeaError("A default material first (fea_materials_assign part='*')")
            try:
                edges = wire_edges(path)
            except IOError as e:
                raise FeaError(str(e)) from None
            if not edges:
                raise FeaError("The STEP contains no free edges (wire model) – convert solids with fea_frame_from_part")
            n0 = len(st.frame.nodes)
            new = frame_from_wires(st, edges, section, float(tol_mm), prefix)
            sess.changed()
            types = {}
            for e in edges:
                types[e["type"]] = types.get(e["type"], 0) + 1
            return out({"edges": types, "members_created": new, "nodes_created": len(st.frame.nodes) - n0,
                        "next": "fea_frame_support_set, fea_frame_load_*, fea_frame_render"})

    @mcp.tool(structured_output=False)
    def fea_frame_from_part(part: str, section_name: str | None = None, use_library: bool = True,
                            connect: bool = True) -> str:
        """Replaces a straight prismatic part with a bar: section from the end face (library profile if A, Iy, Iz match), nodes at end-face centroids; connect=True rigidly joins end faces touching other parts. The part is ignored in the mesh. For slender parts of an assembly."""
        from ..beams.geometry import member_from_part
        with sess.use(op="replacing a part with a member"):
            st = sess.require_study()
            if part not in sess.part_names():
                raise FeaError(f"Part '{part}' does not exist")
            try:
                d = member_from_part(st, sess.parts, part, section_name, use_library, connect)
            except ValueError as e:
                raise FeaError(str(e)) from None
            sess.changed(mesh=True)
            d["ignored_parts"] = st.ignored_parts
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_frame_checks_set(member: str, settings: dict | None = None, replace: bool = False) -> str:
        """EN 1993-1-1 check parameters of a bar ('*' = all): Lcr_y, Lcr_z [mm] (else ky, kz x length, default 1), L_LT, C1, zg, k_LT, Mcr [N*mm], lt_restrained, sway_y, sway_z, fabrication, grade. replace=True replaces (else supplements); settings None + replace = delete. Run with fea_frame_assess."""
        with sess.use():
            st = sess.require_study()
            fr = st.frame
            targets = list(fr.members) if member == "*" else [member]
            for m in targets:
                if m not in fr.members:
                    raise FeaError(f"Member '{m}' does not exist")
            ch = dict(settings or {})
            bad = set(ch) - set(CHECK_KEYS)
            if bad:
                raise FeaError(f"Unknown assessment parameters: {', '.join(sorted(bad))} (allowed: {', '.join(CHECK_KEYS)})")
            for k, v in ch.items():
                if k in ("lt_restrained", "sway_y", "sway_z"):
                    if not isinstance(v, bool):
                        raise FeaError(f"{k} must be true / false")
                elif k == "fabrication":
                    if v not in ("rolled", "welded", "hot_finished", "cold_formed", "auto"):
                        raise FeaError("fabrication: rolled | welded | hot_finished | cold_formed | auto")
                elif k == "grade":
                    ch[k] = str(v).strip().upper().replace(" ", "")
                elif v is not None:
                    if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
                        raise FeaError(f"{k} must be a positive number")
            for m in targets:
                cur = {} if replace else dict(fr.checks.get(m, {}))
                cur.update(ch)
                if cur:
                    fr.checks[m] = cur
                else:
                    fr.checks.pop(m, None)
            sess.frame_check = None
            return out({"checks": {m: fr.checks.get(m, {}) for m in targets}, "meaning": CHECK_KEYS})

    @mcp.tool(structured_output=False)
    def fea_frame_results(step: int | str = 0, member: str | None = None, points: int = 21) -> str:
        """Frame results for a state: per bar extremes of N, Vy, Vz, T, My, Mz (value, position s), max sigma_eq (sqrt(sigma^2 + 3 tau^2), estimate), max displacement; support reactions [N, N*mm], node displacements/rotations, body-connection forces, equilibrium. member = course along the bar at `points` points. Signs: N > 0 tension; bar axes x (start->end), y, z (up)."""
        from .diagnostics import frame_step_summary
        with sess.use():
            res = sess.require_results()
            i = sess.step_index(step)
            st = res.steps[i]
            if not st.frame:
                raise FeaError(f"State '{st.name}' has no member results (a study without a frame system?)")
            fs = frame_step_summary(st)
            d = {"step": st.name, "members": fs["members"],
                 "reactions": {k: {"force_N": r[:3], "moment_Nmm": r[3:]} for k, r in fs["reactions"].items()},
                 "nodes": {k: {"u_mm": v[:3], "rot_rad": v[3:]} for k, v in st.frame.get("nodes", {}).items()},
                 "max_node_displacement_mm": fs.get("max_node_displacement_mm"),
                 "max_sigma_eq_MPa": fs.get("max_sigma_eq_MPa"), "max_abs_moment_Nmm": fs.get("max_abs_moment_Nmm")}
            if st.frame.get("coupling_forces"):
                d["connection_forces"] = {k: {"force_N": r[:3], "moment_Nmm": r[3:]}
                                          for k, r in st.frame["coupling_forces"].items()}
            eq = st.equilibrium or {}
            if eq:
                d["equilibrium"] = {"applied_load_N": eq.get("applied_exact"), "reaction_total_N": eq.get("reaction_total"),
                                    "imbalance_relative": eq.get("imbalance"), "ok": (eq.get("imbalance") or 0) <= 1e-3}
            if member is not None:
                md = st.frame["members"].get(member)
                if md is None:
                    raise FeaError(f"Member '{member}' is not in the results (members: {', '.join(st.frame['members'])})")
                s = np.asarray(md["s"], float)
                n = len(s)
                k = max(2, min(int(points), n))
                idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
                d["member_course"] = {"member": member, "section": md.get("section"), "s_mm": s[idx],
                                      **{q: np.asarray(md[q], float)[idx] for q in LOCAL},
                                      "sigma_eq_MPa": np.asarray(md["sigma_eq"], float)[idx],
                                      "u_mm": np.asarray(md["u"], float)[idx]}
            d["units"] = "N, N·mm, MPa, mm; N > 0 tah"
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_frame_assess(member: str | None = None, rows: bool = True) -> str:
        """EN 1993-1-1 check of bars over all load states (FEM internal forces as design forces): classification, section resistance, buckling, LTB, interaction; Winkler-Bach for arches R/h < 10 (information). Lengths and LTB from fea_frame_checks_set (default bar length, k = 1). Returns utilisation per bar (governing check and state), the maximum, and detailed rows for `member` (None = most utilised); rows=False omits them."""
        from ..beams.assess import assess
        with sess.use(op="member assessment"):
            st = sess.require_study()
            res = sess.require_results()
            if not st.frame.members or not any(s.frame for s in res.steps):
                raise FeaError("The results contain no members (fea_solve_run with a load case)")
            if sess.frame_check is None:
                try:
                    sess.frame_check = assess(st, res)
                except Exception as e:  # noqa: BLE001 – EC3Error, section
                    raise FeaError(f"Member check: {e}") from None
            a = sess.frame_check
            umax, mmax = a.get("_max", (0.0, None))
            table = []
            for k, v in a.items():
                if k == "_max":
                    continue
                u = v.get("utilization")
                table.append({"member": k, "section": v.get("section"), "length_mm": v.get("length_mm"),
                              "utilization": u, "status": ("NOT ASSESSED" if u is None else
                                                           "PASS" if u <= 1 else "FAIL"),
                              "governing": v.get("governing"), "case": v.get("case"),
                              "section_class": v.get("section_class"), "grade": v.get("grade"),
                              **({"curved": v["curved"]} if v.get("curved") else {}),
                              **({"warnings": v["warnings"]} if v.get("warnings") else {})})
            table.sort(key=lambda r: -(r["utilization"] or 0))
            d = {"max_utilization": umax, "governing_member": mmax,
                 "status": "PASS" if umax <= 1 else "FAIL", "members": table,
                 "gamma": {"M0": st.frame.settings.gamma_M0, "M1": st.frame.settings.gamma_M1},
                 "note": "Loads from the states are taken as design loads (the γF factors are given by the user); rows (informative) are not counted in the utilization; curved members are not covered by EN 1993-1-1 (Winkler–Bach)."}
            target = member or mmax
            if rows and target is not None:
                if target not in a:
                    raise FeaError(f"Member '{target}' does not exist")
                v = a[target]
                d["detail"] = {"member": target, "rows": [
                    {"check": r["name"], "clause": r["clause"], "effect": r["effect"], "resistance": r["resistance"],
                     "unit": r["unit"], "utilization": r["utilization"] if math.isfinite(r["utilization"]) else None,
                     "x_mm": r["x"], **({"note": r["note"]} if r.get("note") else {})} for r in v.get("rows", [])]}
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_frame_render(quantity: str = "model", step: int | str = 0, output_path: str | None = None,
                         view: str | list[float] = "iso", scale: float | None = None, width: int = 1280,
                         height: int = 720, bands: int = 12, show_names: bool = True, return_image: bool = True,
                         load_case: str | None = None):
        """PNG of the frame (headless; mixed models show bodies grey). quantity: model (before solving: sections, supports, hinges, connections, loads of load_case, None = all; show_names) | deformed (colour |u|, scale = magnification) | My | Mz | N | Vy | Vz | T (diagram perpendicular to the bar, scale = mm per unit) | sigma | utilization (runs fea_frame_assess). view etc. as fea_results_render_image."""
        from .render import render
        from .frame_scenes import frame_model_scene, result_frame_scene
        from .tools_results import _png_path
        if not 2 <= bands <= 24:
            raise FeaError("bands must be 2–24")
        with sess.use(op="render"):
            st = sess.require_study()
            if not st.frame.members:
                raise FeaError("The study has no members (fea_frame_member_set)")
            if quantity == "model":
                lc = load_case
                if lc is not None and lc not in [x.name for x in st.load_cases]:
                    raise FeaError(f"Load case '{lc}' does not exist")
                if sess.frame_only:
                    scene = frame_model_scene(sess, lc, show_names)
                else:
                    from .scenes import model_scene
                    scene = model_scene(sess, lc, None, False, None)
                meta = {"quantity": "model", "load_case": lc}
            else:
                res = sess.require_results()
                i = sess.step_index(step)
                util = None
                if quantity == "utilization":
                    from ..beams.assess import assess
                    if sess.frame_check is None:
                        sess.frame_check = assess(st, res)
                    util = sess.frame_check
                scene, meta = result_frame_scene(sess, i, quantity, scale, bands, util)
            p = _png_path(sess, output_path, f"frame_{quantity}")
            t0 = time.time()
            meta.update(render(scene, p, view, width, height))
            meta["render_time_s"] = time.time() - t0
            txt = out(meta)
            return [txt, Image(path=p)] if return_image else txt
