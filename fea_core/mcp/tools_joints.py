"""Tools: bolts and welds (F3) and 3D printing (F4) – slim wrappers over fea_core.joints / fea_core.printing."""
from __future__ import annotations

import math
from dataclasses import fields

from .common import out
from .session import FeaError, Session


def bolt_row(b) -> dict:
    return {"name": b.name, "designation": b.designation, "enabled": b.enabled, "connection": b.connection,
            "strength": b.strength, "grip_mm": b.grip, "head_part": b.head_part, "nut_part": b.nut_part,
            "plates": [p["part"] for p in b.plates], "preload_mode": b.preload_mode, "preload": b.preload,
            "cad_parts": b.cad_parts}


def weld_row(w) -> dict:
    return {"name": w.name, "part": w.part, "enabled": w.enabled, "a_mm": w.a, "length_mm": w.length,
            "fat_sigma": w.fat_sigma, "fat_tau": w.fat_tau, "sn_curve": w.sn_curve}


def _apply(obj, changes: dict, what: str) -> list[str]:
    names = {f.name for f in fields(obj)}
    bad = set(changes) - names
    if bad:
        raise FeaError(f"{what}: unknown items {', '.join(sorted(bad))} (allowed: {', '.join(sorted(names - {'name'}))})")
    for k, v in changes.items():
        setattr(obj, k, v)
    return sorted(changes)


def register(mcp, sess: Session) -> None:

    # ------------------------------------------------------------------ bolts and welds
    @mcp.tool(structured_output=False)
    def fea_joints_detect(apply: bool = True, accept_shape: bool | list[str] = False) -> str:
        """Recognizes bolts (from part names such as 'ISO 4017 - M8 x 30 - 8.8', nuts, washers; by shape if the name has no standard) and welds (name 'svar'/'weld', layer, attribute, colour weld_colors) and, with apply, writes them to the study. Shape-only bolts are suggestions (certainty, reason): accept_shape=True adds those >= 0.7, a list of names ('Šroub 3') adds exactly those. A bolt becomes a preloaded beam in the FEM; bolt/nut/washer parts are not meshed."""
        from ..joints.detect import detect, weld_geometry
        with sess.use(op="joint recognition"):
            st = sess.require_study()
            rep = detect(sess.parts, st.bolts, settings=st.joints)        # colours of welds from the study (as in the GUI)
            welds = []
            for w in rep.welds:
                g = weld_geometry(sess.parts, w.part)
                welds.append({**weld_row(w), "kind": g.get("kind"), "a_geom_mm": g.get("a"),
                              "length_geom_mm": g.get("length"), "fusion_parts": g.get("fusion")})
            shaped = [s for s in rep.suggestions if s.kind == "bolt" and s.source == "shape"]
            if isinstance(accept_shape, list):
                known = {s.spec.name for s in shaped}
                bad = [n for n in accept_shape if n not in known]
                if bad:
                    raise FeaError(f"Bolt proposal {', '.join(bad)} does not exist (proposals by shape: {', '.join(sorted(known)) or 'none'})")
                accepted = [s.spec for s in shaped if s.spec.name in accept_shape]
            else:
                accepted = [s.spec for s in shaped if s.accept] if accept_shape else []
            bolts = rep.bolts + accepted
            if apply:
                if st.bolts != bolts or st.welds != rep.welds:        # without a change do not discard the mesh or the results
                    st.bolts, st.welds = bolts, rep.welds
                    sess.changed(mesh=True)
            d = {"bolts": [bolt_row(b) for b in bolts], "welds": welds, "warnings": rep.warnings,
                 "unused_fasteners": rep.unused, "applied": apply}
            if shaped:
                d["suggestions"] = [{"bolt": bolt_row(s.spec), "source": s.source, "confidence": s.confidence,
                                     "reason": s.reason, "accept_default": s.accept,
                                     "accepted": any(s.spec is a for a in accepted)} for s in shaped]
                if not accepted:
                    d["next"] = ("bolts by shape were not added – after checking, fea_joints_detect(accept_shape=True) or accept_shape=['Šroub …']")
            if rep.rejected:
                d["rejected"] = rep.rejected
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_joints_list() -> str:
        """Bolts, welds and assessment settings (EN 1993-1-8 category, gamma_M2, preload)."""
        with sess.use():
            st = sess.require_study()
            return out({"bolts": [bolt_row(b) for b in st.bolts], "welds": [weld_row(w) for w in st.welds],
                        "settings": st.joints.to_dict()})

    def _effect(level: str, mesh_ok: bool) -> str:
        """Projects the adjustment of the joint according to the effect (fea_core.joints.levels, as the GUI step Joints does)."""
        if level == "mesh":
            sess.changed(mesh=True)
            return "new mesh and FEM analysis"
        if level == "fe":
            sess.changed()                           # results removed, mesh stays
        elif level == "assess":
            sess.joints = None                       # only the assessment – the mesh and the FEM results stay
        if level and mesh_ok:                        # the mesh signature contains the joints as a whole → restore (the mesh stays valid)
            sess.mesh_sig = sess.mesh_signature()
        return {"fe": "new FEM calculation (the mesh stays)", "assess": "assessment only – the mesh and FEM results stay (fea_joints_assess)"}.get(level, "no changes")

    @mcp.tool(structured_output=False)
    def fea_joints_update(name: str, changes: dict) -> str:
        """Changes a bolt/weld by name ('*settings*' = common settings). Examples: {"enabled": false}; {"strength": "10.9"}; {"preload_mode": "force", "preload": 20000} [N]; {"preload_mode": "torque", "preload": 25} [N*m]; {"alpha_A": 1.4, "mu_G": 0.12}; weld {"a": 4, "fat_sigma": 36}; settings {"category": "C", "slip_mu": 0.3, "gamma_M2": 1.25}. Assessment-only changes (gamma_M2, category, slip mu, FAT...) keep mesh and results - just run fea_joints_assess; the answer says what needs recomputing."""
        from ..joints.levels import change_level, settings_level
        with sess.use():
            st = sess.require_study()
            mesh_ok = sess.valid_mesh() is not None
            if name == "*settings*":
                old = st.joints.to_dict()
                done = _apply(st.joints, changes, "joint settings")
                eff = _effect(settings_level(old, st.joints.to_dict()), mesh_ok)
                return out({"settings": st.joints.to_dict(), "changed": done, "effect": eff})
            for b in st.bolts:
                if b.name == name:
                    if "preload_mode" in changes and changes["preload_mode"] not in ("auto", "force", "torque"):
                        raise FeaError("preload_mode must be auto | force | torque")
                    old = b.to_dict()
                    done = _apply(b, changes, f"bolt {name}")
                    eff = _effect(change_level("bolt", old, b.to_dict(), st.joints.preload_case), mesh_ok)
                    return out({"bolt": bolt_row(b), "changed": done, "effect": eff})
            for w in st.welds:
                if w.name == name:
                    old = w.to_dict()
                    done = _apply(w, changes, f"weld {name}")
                    eff = _effect(change_level("weld", old, w.to_dict(), st.joints.preload_case), mesh_ok)
                    return out({"weld": weld_row(w), "changed": done, "effect": eff})
            raise FeaError(f"Neither a bolt nor a weld '{name}' exists")

    @mcp.tool(structured_output=False)
    def fea_joints_assess() -> str:
        """Assesses joints from the results: bolts VDI 2230 + EN 1993-1-8 (tension, shear, bearing, slip, fatigue), welds directional method + fatigue (EN 1993-1-9 / IIW). Returns utilisation (<= 1 passes) and the governing check."""
        from ..joints.assess import assess_joints
        with sess.use(op="joint assessment"):
            st = sess.require_study()
            res = sess.require_results()
            if not st.active_bolts() and not st.active_welds():
                raise FeaError("The study has no active bolts or welds (fea_joints_detect)")
            a = assess_joints(st, res, sess.parts)
            sess.joints = a

            def row(c):
                u = c.utilization
                return {"name": c.name, "label": c.label, "utilization": u if math.isfinite(u) else None,
                        "status": "PASS" if c.ok else ("FAIL" if math.isfinite(u) else "NOT ASSESSED"),
                        "governing": c.governing,
                        "checks": [{"check": n, "utilization": v if math.isfinite(v) else None, "detail": t}
                                   for n, v, t in c.rows],
                        "warnings": c.warnings}
            return out({"cases": a.cases, "fatigue_included": a.fatigue, "bolts": [row(c) for c in a.bolts],
                        "welds": [row(c) for c in a.welds], "warnings": a.warnings})

    # ------------------------------------------------------------------ 3D tisk
    @mcp.tool(structured_output=False)
    def fea_print_set(part: str | None = None, settings: dict | None = None, criterion: str | None = None) -> str:
        """3D-printing settings of a part with a printed material (E_z, e.g. 'Tisk PLA Bambu Basic'). part name, '*' = all printed parts, None = only criterion. settings: build_dir [x,y,z], layer_height, line_width [mm], walls, top_layers, bottom_layers, infill 0-1 (1-100 taken as %) or infill_percent, pattern (grid, gyroid, ...), enabled (False = solid part), tolerance [mm] (free surfaces move inward by that; constraint/load faces stay), solid_thin, thin_width, note. criterion tsai_wu | tsai_hill (all parts). Changes do not re-mesh but discard results. See fea_print_get."""
        from ..printing.model import PrintSpec
        from ..printing.region import print_parts
        from .tools_print import spec_changes
        with sess.use():
            st = sess.require_study()
            if part is None and settings:
                raise FeaError("settings requires part (a part name or '*' = all printed parts)")
            if part is None and criterion is None:
                raise FeaError("Enter part + settings or criterion")
            ch, notes = spec_changes(settings or {})
            if criterion is not None and criterion not in ("tsai_wu", "tsai_hill"):
                raise FeaError("criterion must be tsai_wu | tsai_hill")
            names = sess.part_names()
            if part == "*":
                targets = print_parts(st, names)
                if not targets:
                    raise FeaError("No part has a printed material (E_z) – assign one with fea_materials_assign (fea_materials_list query='Tisk')")
            elif part is not None:
                if part not in names:
                    raise FeaError(f"Part '{part}' does not exist. Parts: {', '.join(names)}")
                targets = [part]
            else:
                targets = []
            for p in targets:
                spec = st.printing.parts.get(p) or PrintSpec(p)
                _apply(spec, ch, "print settings")
                st.printing.parts[p] = spec
            if criterion is not None:
                st.printing.criterion = criterion
            sess.changed()                     # the mesh stays, the assignment of the printing is recalculated (region.ensure)
            d: dict = {"criterion": st.printing.criterion}
            if part is not None and part != "*":
                d = {"part": part, "print": st.printing.parts[part].to_dict(), **d}
            elif targets:
                d = {"parts": {p: st.printing.parts[p].to_dict() for p in targets}, **d}
            not_printed = [p for p in targets if p not in print_parts(st, [p])]
            if not_printed:
                d["warning"] = ("The part material is not a printed one (E_z missing) – the settings take effect only after a printed material is assigned (fea_materials_list query='Tisk').")
            if notes:
                d["notes"] = notes
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_print_results(step: int | str = 0) -> str:
        """3D-printing results: criterion (Tsai-Wu/Hill), min safety per part and place, stresses in print axes (11, 22 in layer plane, 33 across), shell share, mass. Maps: print_safety / print_index."""
        with sess.use():
            res = sess.require_results()
            i = sess.step_index(step)
            st = res.steps[i]
            if not st.printing:
                raise FeaError("The result contains no printed parts (material with E_z)")
            return out({"step": st.name, "printing": st.printing, "parts_info": res.info.get("printing"),
                        "warnings": [w for w in res.warnings if "tisk" in w.lower() or "PLA" in w]})
