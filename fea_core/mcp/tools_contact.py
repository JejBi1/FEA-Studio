"""Tools: contacts of assemblies, friction and nonlinear stepping (F8, fea_core.contacts).

A contact pair = two parts (all their faces in contact) or two selections of faces; types as in Ansys / Fusion (bonded,
no separation, frictionless, frictional, rough, interference, compression only, no interaction). A calculation with contact is
NONLINEAR (superposition of states does not hold, fatigue by superposition is not available) and runs only in CalculiX – long runs
via fea_job_start. μ is an estimate (friction library by purpose: joint slip → μ_min, mechanism resistance → μ_max), the normal
stiffness K and the stick slope λ are penalties – check the results for sensitivity.
"""
from __future__ import annotations

import math
import time

import numpy as np
from mcp.server.mcpserver import Image

from .common import guarded_selection, out
from .f8_common import METHOD_LABELS, SPLIT_LABELS, contact_type, find_pair, pair_describe, type_label
from .session import FeaError, Session

CONTACT_FIELDS = {   # result field: (key of StepResult.fields, description, unit); Czech names stay accepted as aliases
    "state": ("STATUS", "Contact state", "–"), "status": ("STATUS", "Contact state", "–"), "stav": ("STATUS", "Contact state", "–"),
    "pressure": ("CPRESS", "Contact pressure", "MPa"), "compression": ("CPRESS", "Contact pressure", "MPa"),
    "cpress": ("CPRESS", "Contact pressure", "MPa"), "tlak": ("CPRESS", "Contact pressure", "MPa"),
    "gap": ("OPENING", "Joint opening (gap)", "mm"), "opening": ("OPENING", "Joint opening (gap)", "mm"),
    "mezera": ("OPENING", "Joint opening (gap)", "mm"),
    "penetration": ("COPEN", "Penetration (penalty)", "mm"), "copen": ("COPEN", "Penetration (penalty)", "mm"),
    "prunik": ("COPEN", "Penetration (penalty)", "mm"), "průnik": ("COPEN", "Penetration (penalty)", "mm"),
    "slip": ("CSLIP", "Slip", "mm"), "cslip": ("CSLIP", "Slip", "mm"), "prokluz": ("CSLIP", "Slip", "mm"),
    "friction": ("CSHEAR", "Frictional (shear) stress", "MPa"), "shear": ("CSHEAR", "Frictional (shear) stress", "MPa"),
    "cshear": ("CSHEAR", "Frictional (shear) stress", "MPa"), "treni": ("CSHEAR", "Frictional (shear) stress", "MPa"),
    "tření": ("CSHEAR", "Frictional (shear) stress", "MPa"),
}
STEP_KEYS = {"name", "case", "scale", "ramp", "bolts", "contacts_off", "mu_change", "prescribed", "dt0", "dt_min",
             "dt_max", "max_inc", "direct", "output_every", "cycles"}
MOTION_KEYS = {"target", "kind", "dof", "amplitude", "csys"}


def _side(sess: Session, pair: str, side: str, value, pc=None) -> str:
    """Side of a pair: part name, existing selection of faces, or faces (id / query / list) → new selection
    (does not overwrite a selection used by another item of the study; pc = the pair being edited).
    """
    st = sess.require_study()
    if isinstance(value, str):
        if value in st.selections:
            if st.selections[value].entity != "faces":
                raise FeaError(f"Selection '{value}' must contain faces")
            return value
        if not sess.frame_only and value in sess.part_names():
            return value
    return guarded_selection(sess, value, f"{pair}_{side}", [pc] if pc is not None else [])


def _check_msgs(sess: Session, name: str | None = None) -> tuple[list[str], list[str]]:
    from ..contacts.check import check
    st = sess.study
    e, w = check(st, sess.part_names() if not sess.frame_only else [])
    if name is not None:
        key = f"'{name}'"
        e = [x for x in e if key in x or not x.startswith("Contact '")]
        w = [x for x in w if key in x or not x.startswith("Contact '")]
    return e, w


def contact_steps(sess: Session) -> list[int]:
    res = sess.require_results()
    idx = [i for i, s in enumerate(res.steps) if s.contacts or s.curves or s.partial]
    if not idx and not (res.info.get("nonlinear") or {}):
        raise FeaError("The results contain no contact or nonlinear steps – define pairs with fea_contact_set / fea_contact_detect(apply=True) and run fea_job_start(kind='solve') or fea_solve_run")
    return idx or list(range(len(res.steps)))


def history_rows(res) -> list[dict]:
    """Values per step (sequence / branches): displacement, stress, reactions, contact forces, partial result."""
    rows = []
    for s in res.steps:
        r = {"step": s.name, "max_displacement_mm": float(s.umag.max()) if len(s.umag) else None,
             "max_mises_MPa": float(s.mises.max()) if len(s.mises) else None}
        if s.reactions:
            r["reactions_N"] = {k: np.asarray(v, float).tolist() for k, v in s.reactions.items()}
        if s.contacts:
            r["contacts"] = {k: {kk: c.get(kk) for kk in ("Fn_N", "Ft_N", "area_contact_mm2", "p_max_MPa", "p_force_MPa",
                                                             "p_max_interior_MPa", "peak_at_edge", "open_max_mm",
                                                             "slip_max_mm")}
                             for k, c in s.contacts.items()}
        if s.partial:
            r["partial"] = True
        rows.append(r)
    return rows


def curve_series(res, target: str | None = None) -> dict:
    """Force–displacement curves of the prescribed motions joined over the steps: {target: {"t", "u", "F", "dof", "kind", "steps"}}.
    u and F are the components in the direction dof (displacement [mm] / rotation [rad], force [N] / moment [N·mm]).
    """
    outc: dict = {}
    for s in res.steps:
        for tg, c in (s.curves or {}).items():
            if target is not None and tg != target:
                continue
            time_curve = "u" not in c            # time course of the pair force ("contact:<pair>": t, F, Fn) – without the path
            d = outc.setdefault(tg, {"t": [], "u": [], "F": [], "dof": c.get("dof", 1),
                                     "kind": "time" if time_curve else c.get("kind", "disp"), "steps": []})
            if time_curve:
                x = np.asarray(c["t"], float)
                F = np.asarray(c.get("Fn", []), float)
                n = min(len(x), len(F))
                d["t"] += x[:n].tolist()
                d["u"] += x[:n].tolist()          # x axis = time of the step
                d["F"] += F[:n].tolist()
                d["steps"].append({"step": s.name, "points": n, "partial": bool(s.partial)})
                continue
            k = int(c.get("dof", 1)) - 1
            u = np.asarray(c["u"], float).reshape(-1, 3)
            F = np.asarray(c["F"], float).reshape(-1, 3)
            n = min(len(u), len(F), len(c["t"]))
            d["t"] += list(map(float, c["t"][:n]))
            d["u"] += u[:n, k].tolist()
            d["F"] += F[:n, k].tolist()
            d["steps"].append({"step": s.name, "points": n, "partial": bool(s.partial)})
    return outc


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_contact_detect(parts: list[str] | None = None, tol_mm: float | None = None, apply: bool = False,
                           type: str | None = None) -> str:
        """Finds contacts between parts in the CAD geometry (before meshing): pairs within tol_mm (None = 0.2 x element size, min 0.05) with state contact/gap/penetration, minimum distance, contact area [mm2], penetration volume/depth and touching face ids. Returns a PROPOSAL; apply=True adds the pairs (existing pairs of the same parts are skipped). type for all (None = auto: contact -> bonded (shared mesh), gap -> frictionless, penetration -> interference - CHECK, it is often a modelled thread or an assembly error). Adjust with fea_contact_set(pair, type=...). parts limits the search."""
        with sess.use(op="searching for contacts"):
            st = sess.require_study()
            from ..contacts.detect import find_touching, default_tol
            names = sess.part_names()
            if parts:
                bad = [p for p in parts if p not in names]
                if bad:
                    raise FeaError(f"Parts {', '.join(bad)} do not exist (parts: {', '.join(names)})")
            tol = float(tol_mm) if tol_mm is not None else default_tol(st)
            if not tol > 0:
                raise FeaError("tol_mm must be positive")
            forced = contact_type(type) if type is not None else None
            t0 = time.time()
            hits = find_touching(sess.parts, tol, names=set(parts) if parts else None)
            cache = sess.__dict__.setdefault("_f8_touch", {})
            existing = {frozenset((c.a, c.b)) for c in st.contacts}
            props, added = [], []
            for h in hits:
                cache[(h["a"], h["b"], round(tol, 9))] = h
                kind = forced or {"contact": "bonded", "gap": "frictionless", "penetration": "interference"}[h["state"]]
                d = {"a": h["a"], "b": h["b"], "state": h["state"], "distance_mm": h["distance"], "area_mm2": h["area"],
                     "proposed_type": kind, "proposed_type_label": type_label(kind),
                     "faces_a": [f"{h['a']}#{i}" for i in h["faces_a"][:20]],
                     "faces_b": [f"{h['b']}#{i}" for i in h["faces_b"][:20]]}
                if h["state"] == "penetration":
                    d.update(overlap_volume_mm3=h["overlap_volume"], depth_mm=h["depth"],
                             warning="Initial penetration – a press fit (type interference), a modelled thread or an assembly error; for a type other than interference the calculation ends with an error")
                d["exists"] = frozenset((h["a"], h["b"])) in existing
                props.append(d)
                if apply and not d["exists"]:
                    from ..contacts.model import ContactPair
                    name = f"{h['a']} – {h['b']}"
                    k = 2
                    while any(c.name == name for c in st.contacts):
                        name = f"{h['a']} – {h['b']} ({k})"
                        k += 1
                    st.contacts.append(ContactPair(name, h["a"], h["b"], kind=kind, initial=h["state"],
                                                   gap=-h["depth"] if h["state"] == "penetration" else h["distance"],
                                                   area=h["area"], source="auto"))
                    added.append(name)
            if added:
                sess.changed()
            d = {"tol_mm": tol, "pairs": props, "time_s": time.time() - t0, "applied": added}
            if not apply and props:
                d["next"] = "fea_contact_detect(apply=True) adds the proposed pairs; then fea_contact_set(pair, type=…, mu=…)"
            if not props:
                d["note"] = "No parts touch within the tolerance"
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_contact_set(pair: str, type: str | None = None, a: str | list | dict | None = None,
                        b: str | list | dict | None = None, mu: float | None = None, purpose: str | None = None,
                        friction_pair: str | None = None, stiffness: float | None = None,
                        stick_slope: float | None = None, clearance: float | None = None, adjust: bool | None = None,
                        slave: str | None = None, enabled: bool | None = None, reset: list[str] | None = None) -> str:
        """Creates/changes contact pair `pair`; a new pair needs a, b, type. a, b = part (all its contacting faces), selection name or faces (id/query) -> creates selections '<pair>_a' / '<pair>_b'. type: bonded | no_separation (slides) | frictionless | frictional | rough (no slip) | interference (press fit) | compression_only (without b = support to ground) | none. mu > 0 (None = from the friction library by purpose: slip -> mu_min, resistance -> mu_max; friction_pair = library pair, see fea_calc_friction_library). stiffness K [N/mm3] (None = 50*min E), stick_slope lambda [N/mm3] (None = K/100): penalties, always written explicitly. clearance [mm] RADIAL (interference = -dd/2) overrides the geometric gap of the pair. adjust: move slave nodes into contact (never for interference). slave 'a'|'b' (None = finer mesh). enabled. reset: list of mu, stiffness, stick_slope, clearance, slave, friction_pair to return to automatic. Returns the pair with effective values (mu + source, K, lambda) and its check."""
        from ..contacts.model import ContactPair
        with sess.use():
            st = sess.require_study()
            pc = next((c for c in st.contacts if c.name == pair), None)
            new = pc is None
            if new:
                if a is None or (b is None and contact_type(type or "frictionless") != "compression_only"):
                    raise FeaError("A new contact pair needs both a and b (part / face selection / faces); b may be missing only for type compression_only (support in compression only to the ground)")
                if type is None:
                    raise FeaError("A new contact pair needs type (bonded | no_separation | frictionless | frictional | rough | interference | compression_only | none)")
                pc = ContactPair(pair, "", "")
            kind = contact_type(type) if type is not None else pc.kind
            if mu is not None and not (math.isfinite(mu) and mu > 0):
                raise FeaError("mu must be positive (no friction = type 'frictionless')")
            if purpose is not None and purpose not in ("slip", "resistance"):
                raise FeaError("purpose: slip (prokluz spoje → μ_min) | resistance (odpor mechanismu → μ_max)")
            for v, nm in ((stiffness, "stiffness"), (stick_slope, "stick_slope")):
                if v is not None and not (math.isfinite(v) and v > 0):
                    raise FeaError(f"{nm} must be positive [N/mm³]")
            if clearance is not None and not math.isfinite(clearance):
                raise FeaError("clearance must be a number [mm]")
            if slave is not None and slave not in ("a", "b"):
                raise FeaError("slave: 'a' | 'b' (None = automaticky)")
            if friction_pair is not None:
                from ..calc import friction_lib as fl
                try:
                    friction_pair = fl.get(friction_pair).name
                except Exception as e:  # noqa: BLE001
                    raise FeaError(str(e)) from None
            bad = set(reset or []) - {"mu", "stiffness", "stick_slope", "clearance", "slave", "friction_pair"}
            if bad:
                raise FeaError(f"reset: unknown parameters {', '.join(sorted(bad))}")
            if a is not None:
                pc.a = _side(sess, pair, "a", a, pc)
            if b is not None:
                pc.b = _side(sess, pair, "b", b, pc) if b != "" else ""
            pc.kind = kind
            for attr, v in (("mu", mu), ("purpose", purpose), ("stiffness", stiffness), ("stick_slope", stick_slope),
                            ("clearance", clearance), ("adjust", adjust), ("slave", slave), ("enabled", enabled),
                            ("materials", friction_pair)):
                if v is not None:
                    setattr(pc, attr, v)
            for r in reset or []:
                setattr(pc, "materials" if r == "friction_pair" else r, None)
            if kind == "interference" and pc.adjust:
                raise FeaError("adjust must not be used with an interference (the adjustment would remove the interference)")
            pc.source = "user"
            if new:
                st.contacts.append(pc)
            sess.changed()
            e, w = _check_msgs(sess, pair)
            d = {"pair": pair_describe(st, pc), "created": new, "nonlinear": st.is_nonlinear(), "errors": e,
                 "warnings": w}
            if pc.kind in ("frictional", "rough", "interference", "no_separation", "frictionless", "compression_only"):
                d["note"] = ("Contact = a nonlinear calculation in CalculiX only (superposition of states does not hold). Long run: fea_job_start(kind='solve') → fea_job_status → fea_job_result.")
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_contact_list() -> str:
        """Contact pairs with effective values (type, mu + source, K, lambda, interference, initial state), settings (formulation, interface meshing, NLGEOM) and the analysis kind (linear/nonlinear, branches/sequence)."""
        with sess.use():
            st = sess.require_study()
            cs = st.contact_settings
            an = st.analysis
            d = {"pairs": [pair_describe(st, c) for c in st.contacts],
                 "settings": {**cs.__dict__, "method_label": METHOD_LABELS.get(cs.method, cs.method),
                              "split_label": SPLIT_LABELS.get(cs.split, cs.split)},
                 "analysis": {"kind": an.kind, "branches": an.branches, "steps": [s.name for s in an.sequence],
                              "stop_keep": an.stop_keep},
                 "nonlinear": st.is_nonlinear()}
            auto = [c for c in st.all_contacts() if c.name not in {x.name for x in st.contacts}]
            if auto:
                d["automatic_pairs"] = [pair_describe(st, c) for c in auto]
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_contact_remove(pair: str) -> str:
        """Removes a pair by name ('*' = all); selections '<pair>_a/_b' stay. Friction changes and switch-offs referring to it in the step sequence are removed."""
        with sess.use():
            st = sess.require_study()
            names = [c.name for c in st.contacts] if pair == "*" else [find_pair(st, pair).name]
            st.contacts = [c for c in st.contacts if c.name not in names]
            for s in st.analysis.sequence:
                s.mu_change = {k: v for k, v in s.mu_change.items() if k not in names}
                s.contacts_off = [k for k in s.contacts_off if k not in names]
            sess.changed()
            return out({"removed": names, "remaining": [c.name for c in st.contacts], "nonlinear": st.is_nonlinear()})

    @mcp.tool(structured_output=False)
    def fea_contact_check() -> str:
        """Pre-run contact/stepping check: formulation x pair type, mu > 0, K, lambda, interference, bolted parts, slip without a constraint (force-controlled part held only by friction does not converge), step sequence. After meshing also the initial state of pairs (slave/master faces, gap/penetration) and initial penetration for non-interference types (error + fix)."""
        with sess.use():
            st = sess.require_study()
            e, w = _check_msgs(sess)
            d = {"ok": not e, "errors": e, "warnings": w, "pairs": len(st.contacts),
                 "active_pairs": [c.name for c in st.active_contacts()], "nonlinear": st.is_nonlinear()}
            mesh = sess.valid_mesh()
            if mesh is not None:
                from ..contacts.check import mesh_errors
                me = mesh_errors(st, mesh)
                d["errors"] += me
                d["ok"] = not d["errors"]
                d["mesh_pairs"] = [{k: r.get(k) for k in ("pair", "slave", "master", "state", "gap_min_mm",
                                                         "area_mm2") if r.get(k) is not None}
                                   for r in (mesh.report or {}).get("contacts", [])]
            else:
                d["mesh"] = "no mesh – the initial state of the pairs in the mesh is checked after fea_mesh_generate"
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_contact_settings(method: str | None = None, split: str | None = None, detect_tol_mm: float | None = None,
                             stabilize: str | None = None, stabilize_k: float | None = None,
                             nlgeom: bool | None = None, pressure: str | None = None) -> str:
        """Study contact settings (none = list). method: surface_to_surface (default, penalty, pressure +-0.1 %) | mortar (hard contact, interference; ~2.5x slower) | node_to_surface (friction convergence only; pressure with tet10 unusable). split: separate (pair parts meshed separately without *TIE - required for interference) | conformal_split (conforming mesh + opened interface - bolted parts). detect_tol_mm: face-finding tolerance. stabilize: none | springs (weak springs, stabilize_k N/mm). nlgeom: large deformations (snap-fit, cam). pressure: dead | follower (needs NLGEOM)."""
        with sess.use():
            st = sess.require_study()
            cs = st.contact_settings
            if method is not None and method not in METHOD_LABELS:
                raise FeaError("method: surface_to_surface | mortar | node_to_surface")
            if split is not None and split not in SPLIT_LABELS:
                raise FeaError("split: separate | conformal_split")
            if stabilize is not None and stabilize not in ("none", "springs"):
                raise FeaError("stabilize: none | springs")
            if pressure is not None and pressure not in ("dead", "follower"):
                raise FeaError("pressure: dead | follower")
            for v, nm in ((detect_tol_mm, "detect_tol_mm"), (stabilize_k, "stabilize_k")):
                if v is not None and not (math.isfinite(v) and v > 0):
                    raise FeaError(f"{nm} must be positive")
            changed = False
            for attr, v in (("method", method), ("split", split), ("detect_tol", detect_tol_mm),
                            ("stabilize", stabilize), ("stabilize_k", stabilize_k), ("nlgeom", nlgeom),
                            ("pressure", pressure)):
                if v is not None and getattr(cs, attr) != v:
                    setattr(cs, attr, v)
                    changed = True
            if changed:
                sess.changed()
            e, w = _check_msgs(sess)
            return out({"settings": {**cs.__dict__, "method_label": METHOD_LABELS.get(cs.method),
                                     "split_label": SPLIT_LABELS.get(cs.split)},
                        "nonlinear": st.is_nonlinear(), "errors": e, "warnings": w})

    @mcp.tool(structured_output=False)
    def fea_analysis_set_sequence(kind: str = "nonlinear", branches: bool = True, steps: list[dict] | None = None,
                                  stop_keep: bool = True) -> str:
        """Analysis kind and load history. kind linear (superposition) | nonlinear. Nonlinear: branches=True: each load state separately after [bolt preload] (parallel, no history, no fatigue); branches=False: SEQUENCE `steps` (one job, state kept after each step). Step {"name", "case" (state or null), "scale", "ramp": linear|step, "bolts": preload|lock|free, "contacts_off": [pairs], "mu_change": {pair: mu}, "prescribed": [{"target": selection with a rigid remote constraint (fea_constraint_add_remote, coupling rigid), "kind": disp|rot, "dof": 1-3, "amplitude": [[t 0-1, value mm or rad], ...]}], "dt0", "dt_min", "dt_max" (fraction of step), "max_inc", "output_every", "cycles" (>= 2 for fatigue with friction)}. Prescribed motion gives a force-displacement curve (fea_results_history). stop_keep: keep the last converged increment on non-convergence (partial result)."""
        from ..contacts.model import LoadStep, Motion
        if kind not in ("linear", "nonlinear"):
            raise FeaError("kind: linear | nonlinear")
        with sess.use():
            st = sess.require_study()
            seq = []
            for i, s in enumerate(steps or []):
                if not isinstance(s, dict):
                    raise FeaError("Each step must be an object {\"name\": …, \"case\": …}")
                bad = set(s) - STEP_KEYS
                if bad:
                    raise FeaError(f"Step {i + 1}: unknown keys {', '.join(sorted(bad))} (allowed: {', '.join(sorted(STEP_KEYS))})")
                motions = []
                for mo in s.get("prescribed") or []:
                    if not isinstance(mo, dict) or "target" not in mo:
                        raise FeaError(f"Step {i + 1}: the prescribed motion must be {{\"target\", \"kind\", \"dof\", \"amplitude\"}}")
                    badm = set(mo) - MOTION_KEYS
                    if badm:
                        raise FeaError(f"Step {i + 1}: unknown motion keys {', '.join(sorted(badm))}")
                    mm = Motion.from_dict(mo)
                    try:
                        mm.amplitude = [[float(t), float(v)] for t, v in mm.amplitude]
                        mm.dof = int(mm.dof)
                    except (TypeError, ValueError):
                        raise FeaError(f"Step {i + 1}: amplitude = [[t, value], …], dof 1–3") from None
                    motions.append(mm)
                ls = LoadStep.from_dict({**{k: v for k, v in s.items() if k != "prescribed"},
                                         "name": s.get("name") or f"Step {i + 1}"})
                ls.prescribed = motions
                ls.mu_change = {str(k): float(v) for k, v in (ls.mu_change or {}).items()}
                seq.append(ls)
            an = st.analysis
            an.kind = kind
            an.branches = bool(branches)
            if steps is not None:
                an.sequence = seq
            an.stop_keep = bool(stop_keep)
            sess.changed()
            e, w = _check_msgs(sess)
            d = {"analysis": an.to_dict(), "nonlinear": st.is_nonlinear(), "errors": e, "warnings": w}
            if kind == "nonlinear":
                d["note"] = ("Nonlinear calculation: superposition of states does not hold, it runs in CalculiX only – use fea_job_start(kind='solve') and watch fea_job_status.")
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_contact_results(step: int | str | None = None, pair: str | None = None) -> str:
        """Contact results per step/state, per pair: normal force Fn [N], friction Ft [N], contact area [mm2] and share, max/mean pressure [MPa], max opening and slip [mm], node counts open/slipping/sticking, mu used; PARTIAL-result flag (non-convergence: last converged increment, with diagnostics) and force-displacement curves. Reliable pressure: p_force_MPa = Fn (ccx CF) / area_active_mm2; nodal p_max_MPa depends on formulation and mesh; p_max_interior_MPa excludes the edges, p_face_max_MPa is the largest wall-averaged value; peak_at_edge = peak at a free edge > 10 % above the interior (warning in edge_peak_warnings)."""
        with sess.use():
            res = sess.require_results()
            idx = contact_steps(sess) if step is None else [sess.step_index(step)]
            rows = []
            for i in idx:
                s = res.steps[i]
                c = s.contacts
                if pair is not None:
                    if pair not in c:
                        raise FeaError(f"Pair '{pair}' is not in the results of state '{s.name}' (pairs: {', '.join(c) or 'none'})")
                    c = {pair: c[pair]}
                r = {"step": s.name, "contacts": c}
                if s.partial:
                    r["partial"] = True
                    r["partial_time"] = getattr(s, "fields_time", None)
                if s.curves:
                    r["curves"] = {k: {"points": len(v.get("t", [])), "dof": v.get("dof"), "kind": v.get("kind")}
                                   for k, v in s.curves.items()}
                eq = s.equilibrium or {}
                if eq:
                    r["equilibrium_imbalance"] = eq.get("imbalance")
                rows.append(r)
            nl = res.info.get("nonlinear") or {}
            d = {"steps": rows, "nonlinear": {k: nl.get(k) for k in ("mode", "jobs", "partial") if k in nl}}
            pw = [w for w in res.warnings if w.startswith("PARTIAL")]
            if pw:
                d["partial_warnings"] = pw
            ew = [w for w in res.warnings if w.startswith("Contact '") and "free edge" in w]
            if ew:
                d["edge_peak_warnings"] = ew
            d["warning"] = ("Nonlinear calculation: μ is an estimate (compute the bounds μ_min/μ_max), K and λ are penalties – verify the results by a sensitivity study; superposition of states does not hold.")
            d["next"] = "fea_contact_render(field='pressure'|'state'|'gap'|'slip'|'friction'), fea_results_history"
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_contact_render(field: str = "pressure", step: int | str = 0, pair: str | None = None, side: str = "slave",
                           parts: list[str] | None = None, output_path: str | None = None,
                           view: str | list[float] = "auto", deformed: bool = False, scale: float | None = None,
                           width: int = 1280, height: int = 720, bands: int = 12, vmin: float | None = None,
                           vmax: float | None = None, show_mesh: bool = False, return_image: bool = True):
        """PNG of contact quantities on part surfaces. field: status (open/slipping/sticking) | pressure (CPRESS MPa) | gap (opening of the deformed mesh, mm) | copen (penalty penetration) | slip (CSLIP mm) | shear (CSHEAR MPa). Draws one side: side slave (default) | master | a | b; pair limits it; or parts list. view auto (contact face) or as fea_results_render_image; deformed + scale; vmin/vmax/bands/show_mesh."""
        from .render import render
        from .f8_scenes import field_scene
        from .tools_results import _png_path
        key = CONTACT_FIELDS.get(str(field).strip().lower())
        if key is None:
            raise FeaError("field: state | pressure | gap | penetration | slip | friction")
        if not 2 <= bands <= 24:
            raise FeaError("bands must be 2–24")
        if side not in ("slave", "master", "a", "b"):
            raise FeaError("side: slave | master | a | b")
        with sess.use(op="render"):
            res = sess.require_results()
            st = sess.study
            i = sess.step_index(step)
            s = res.steps[i]
            fk, label, unit = key
            if fk not in s.fields:
                raise FeaError(f"State '{s.name}' has no contact fields (a calculation without contact?) – fea_contact_results")
            if parts is None:
                from .f8_scenes import contact_side_parts
                pairs = [find_pair(st, pair)] if pair is not None else None
                parts = contact_side_parts(st, res.mesh, pairs, side) or \
                    contact_side_parts(st, res.mesh, pairs, "a" if side in ("slave", "a") else "b")
            src = f"state “{s.name}“" + (" – PARTIAL RESULT" if s.partial else "")
            if fk == "STATUS":
                scene, meta = field_scene(sess, None, label, unit, src, U=s.U if deformed else None, scale=scale,
                                          parts=parts, status=s.fields["STATUS"], show_mesh=show_mesh)
            else:
                scene, meta = field_scene(sess, s.fields[fk], label, unit, src, U=s.U if deformed else None,
                                          scale=scale, parts=parts, nbands=bands, vmin=vmin, vmax=vmax,
                                          show_mesh=show_mesh)
            p = _png_path(sess, output_path, f"contact_{fk.lower()}_{i}")
            if isinstance(view, str) and view.strip().lower() == "auto":
                from .f8_scenes import contact_view
                view = contact_view(res.mesh, [find_pair(st, pair)] if pair is not None else None,
                                    side if side in ("slave", "master") else "slave")
            meta.update(render(scene, p, view, width, height))
            meta["step"] = s.name
            if s.partial:
                meta["warning"] = "Partial result (non-convergence) – the last converted increment applies"
            txt = out(meta)
            return [txt, Image(path=p)] if return_image else txt

    @mcp.tool(structured_output=False)
    def fea_results_history(target: str | None = None, plot: bool = False, output_path: str | None = None,
                            max_points: int = 400, return_image: bool = True):
        """Nonlinear history: (1) per step/state values (max displacement, max von Mises, reactions, Fn/Ft, area, p_max, opening, slip, partial flag); (2) force-displacement curves of prescribed motions (fea_analysis_set_sequence) joined over steps: path u [mm]/rotation [rad], reaction F [N]/moment [N*mm], max |F| and where (assembly/separation force). target = one constraint; plot=True adds F(u) PNG."""
        with sess.use(op="progress"):
            res = sess.require_results()
            curves = curve_series(res, target)
            if target is not None and not curves:
                raise FeaError(f"Curve '{target}' is not in the results (a prescribed motion in the step sequence?)")
            d = {"steps": history_rows(res)}
            cd = {}
            for tg, c in curves.items():
                u, F = np.asarray(c["u"]), np.asarray(c["F"])
                k = int(np.argmax(np.abs(F))) if len(F) else 0
                stride = max(1, int(math.ceil(len(u) / max(int(max_points), 2))))
                cd[tg] = {"dof": c["dof"], "kind": c["kind"], "points": len(u), "u": u[::stride], "F": F[::stride],
                          "t": np.asarray(c["t"])[::stride], "max_abs_F": float(abs(F[k])) if len(F) else None,
                          "at_u": float(u[k]) if len(u) else None, "steps": c["steps"],
                          "units": "t [s], Fn [N] (pair force in time)" if c["kind"] == "time"
                          else "u [mm], F [N]" if c["kind"] == "disp" else "u [rad], F [N·mm]"}
            if cd:
                d["curves"] = cd
            if not plot:
                return out(d)
            if not cd:
                raise FeaError("The results have no force–displacement curves (a plot needs a prescribed motion in the sequence)")
            from .f8_plots import xy_plot
            from .tools_results import _png_path
            p = _png_path(sess, output_path, "history")
            kinds = {c["kind"] for c in curves.values()}
            xy_plot(p, [{"x": c["u"], "y": c["F"], "label": tg} for tg, c in curves.items()],
                    "Force – travel of the prescribed motion", "displacement [mm]" if kinds == {"disp"} else "displacement / rotation / time",
                    "reakce [N]" if kinds == {"disp"} else "reaction [N] / [N·mm]")
            d["image"] = str(p)
            txt = out(d)
            return [txt, Image(path=p)] if return_image else txt
