"""Transfer of the result of the mechanism into the linear FEM (specification 3.5).

An instant t → the load state of the body (d'Alembert) in the COORDINATES OF THE BODY (position from the CAD, x_CAD = Rᵀ·(x − c(t)) + c_CAD):
  - reactions of the joints (total force + moment of the joint including the driver, friction and stop) → RemoteLoad(coupling="distributing")
    on the face of the joint on this body (MotionJoint.selection_a / selection_b) at the point of the joint;
  - force elements (springs, dampers, bushings) and external loads → RemoteLoad on their faces (selection_a/_b,
    MotionLoad.selection); the torsion spring is already in the moment of the joint;
  - inertia + weight → ONE load Inertia(point = centroid, accel = Rᵀ(a − g), omega = Rᵀω, alpha = Rᵀα, parts
    of the body) – the weight is in accel (−g), therefore Gravity is not added (not to count twice);
  - support: the first joint from support_joints (or the first joint of the body) = RemoteSupport(coupling="distributing",
    6 components) and its reaction from the MBD is NOT loaded (a remote constraint and a remote load must not share nodes) – the reaction of the FE
    in the support must come out equal to the reaction of the joint from the MBD (= check of the inertia relief: the rest ≈ 0, `expected`); the other
    support_joints (redundant joints) are also constraints (without load) – the distribution among them is determined by the FE.
Maximum of the quantity → the instant of the maximum (MotionResult.find_max). The whole course (history) → unit states: 6 components
of the RemoteLoad distributing per joint / element (BearingLoad NOT – the pressure only on the facing half of the hole, it is not linear),
the inertia 3 states a₀ (= (a − g) in the axes of the body), 3 states α, 6 states W (W = ω·ωᵀ − |ω|²·I, the symmetric unit
matrix) and the fatigue channels (time courses of the components in the axes of the body) → Study.fatigue.history (superposition).
"""
from __future__ import annotations

import numpy as np

from ..study import Inertia, LoadCase, RemoteLoad, RemoteSupport, BearingLoad
from .model import JOINT_LABELS

COMP = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
DOF_MASK = {"revolute": (1, 1, 1, 1, 1, 0), "cylindrical": (1, 1, 0, 1, 1, 0), "prismatic": (1, 1, 0, 1, 1, 1),
            "spherical": (1, 1, 1, 0, 0, 0), "fixed": (1, 1, 1, 1, 1, 1), "screw": (1, 1, 0, 1, 1, 0)}


class ToFeaError(ValueError):
    """The transfer to FE is not possible (a missing face of a joint, parts outside one body…)."""


# ---------------------------------------------------------------- selection of the body and the items
def _body_of(study, res, parts: list[str] | None, body: str | None) -> int:
    names = res.body_names
    if body:
        if body not in names:
            raise ToFeaError(f"Body '{body}' is not in the mechanism results")
        return names.index(body)
    if not parts:
        moving = [i for i, b in enumerate(res.meta["bodies"]) if not b["ground"]]
        if len(moving) == 1:
            return moving[0]
        raise ToFeaError("Specify the body or parts (ToFea.parts) for which the load cases are created")
    owners = set()
    for p in parts:
        found = [i for i, b in enumerate(res.meta["bodies"]) if p in b["parts"] or p in b.get("extra", [])]
        if not found:
            raise ToFeaError(f"Part '{p}' belongs to no mechanism body")
        owners.add(found[0])
    if len(owners) > 1:
        raise ToFeaError("The parts belong to different bodies – transfer to FEA is per body (each body separately)")
    return owners.pop()


def _attached(res, i) -> list[tuple[int, str]]:
    """Joints of the body i: [(index of the joint, "a" | "b")] (a gear without its own position is left out)."""
    out = []
    for j, J in enumerate(res.meta["joints"]):
        if J["kind"] in ("gear", "rack_pinion"):
            continue
        if J["b"] == i:
            out.append((j, "b"))
        elif J["a"] == i:
            out.append((j, "a"))
    return out


def _joint_point(J, side) -> np.ndarray:
    """Point of the joint on the body in the position of the CAD."""
    if side == "b" and J.get("point_b") is not None:
        return np.asarray(J["point_b"], float)
    return np.asarray(J["origin"], float)


def _joint_selection(J, side) -> str | None:
    return J.get("selection_b") if side == "b" else J.get("selection_a")


def _joint_wrench(res, j, side, k=None):
    """(F, M about the point of the joint on this body) – globally, at all times (k = None) or at the time k."""
    a = res.arrays
    sl = slice(None) if k is None else k
    F, M = a["jF"][sl, j], a["jM"][sl, j]
    if side == "b":
        return F, M
    return -F, -M - np.cross(a["jP"][sl, j] - a["jPa"][sl, j], F)


def _force_items(study, res, i):
    """Force elements and loads on the body i: [(name, selection, point CAD, function wrench(k|None) → (F, M))]."""
    a = res.arrays
    items = []
    mech = study.mechanism
    for k, f in enumerate(res.meta["forces"]):
        if f["kind"] == "torsion":
            continue                                    # the moment of the torsion spring is in the moment of the joint
        fe = next((x for x in mech.forces if x.name == f["name"]), None)
        for side in ("b", "a"):
            if f[side] != i:
                continue
            sel = (fe.selection_b if side == "b" else fe.selection_a) if fe else None
            pt = np.asarray((fe.point_b if side == "b" else fe.point_a) if fe else [0, 0, 0], float)
            if f["kind"] == "bushing":
                pt = np.asarray(fe.point_a, float)

            def wr(kk=None, _k=k, _s=side):
                sl = slice(None) if kk is None else kk
                F, M = a["fF"][sl, _k], a["fM"][sl, _k]
                if _s == "b":
                    return F, M
                return -F, -M - np.cross(a["fPb"][sl, _k] - a["fPa"][sl, _k], F)
            items.append((f["name"], sel, pt, wr, "element"))
    for k, ld in enumerate(res.meta["loads"]):
        if ld["body"] != i:
            continue
        ml = next((x for x in mech.loads if (x.name or "") == ld["name"]), None)
        pt = np.asarray(ml.point if ml else [0, 0, 0], float)

        def wr(kk=None, _k=k):
            sl = slice(None) if kk is None else kk
            return a["lF"][sl, _k], a["lM"][sl, _k]
        items.append((ld["name"], ld.get("selection"), pt, wr, "load"))
    return items


def _fe_parts(study, res, i) -> list[str]:
    B = res.meta["bodies"][i]
    excl = study.excluded_parts()
    return [p for p in B["parts"] if p not in excl]


def _support_constraint(J, side, sel, pt, first: bool, notes: list) -> RemoteSupport:
    dofs = [True] * 6
    if not first:
        ax = np.asarray(J["axis"], float)
        ax = ax / np.linalg.norm(ax)
        k = int(np.argmax(np.abs(ax)))
        mask = DOF_MASK.get(J["kind"])
        if mask is not None and abs(abs(ax[k]) - 1) < 1e-9:
            # frame of the joint: z = axis → components in the global axes (the axis of the joint = global axis k)
            other = [x for x in range(3) if x != k]
            dofs = [True] * 6
            dofs[k] = bool(mask[2])
            dofs[3 + k] = bool(mask[5])
            for x in other:
                dofs[x] = bool(mask[0])
                dofs[3 + x] = bool(mask[3])
        else:
            notes.append(f"Joint '{J['name']}': the axis is not parallel to a global axis – the FEA support holds all 6 components")
    return RemoteSupport(selection=sel, point=[float(x) for x in pt], dofs=dofs, coupling="distributing")


def _prepare(study, res, body, parts, support_joints):
    i = _body_of(study, res, parts, body)
    B = res.meta["bodies"][i]
    if B["ground"]:
        raise ToFeaError(f"Body '{B['name']}' is the ground – it does not move, it has no motion loads")
    att = _attached(res, i)
    if not att:
        raise ToFeaError(f"Body '{B['name']}' has no joint – it cannot be supported")
    jn = res.joint_names
    sup = list(support_joints or [])
    for s in sup:
        if s not in jn:
            raise ToFeaError(f"Support joint '{s}' does not exist")
        if not any(jn[j] == s for j, _ in att):
            raise ToFeaError(f"Support joint '{s}' does not belong to body '{B['name']}'")
    if not sup:
        sup = [jn[att[0][0]]]
    for j, side in att:
        J = res.meta["joints"][j]
        if not _joint_selection(J, side):
            raise ToFeaError(f"Joint '{J['name']}' has no face on body '{B['name']}' (selection_{side}) – select the joint face (hole, pin) that carries the reaction")
        if _joint_selection(J, side) not in study.selections:
            raise ToFeaError(f"Selection '{_joint_selection(J, side)}' of joint '{J['name']}' is not in the study")
    fe_parts = _fe_parts(study, res, i)
    if not fe_parts:
        raise ToFeaError(f"Body '{B['name']}' has no part in the FEA model")
    return i, B, att, sup, fe_parts


def _remove_old(study, names: set[str], sup_sel: set[str]):
    study.load_cases = [lc for lc in study.load_cases if lc.name not in names]
    study.constraints = [c for c in study.constraints
                         if not (isinstance(c, RemoteSupport) and c.selection in sup_sel and c.coupling == "distributing")]


def _set_case(study, name, loads):
    for k, lc in enumerate(study.load_cases):
        if lc.name == name:
            study.load_cases[k] = LoadCase(name, loads)
            return
    study.load_cases.append(LoadCase(name, loads))


def _check_items(items, B):
    for name, sel, pt, wr, what in items:
        F, M = wr(None)
        if (np.abs(F).max() > 0 or np.abs(M).max() > 0) and not sel:
            raise ToFeaError(f"{what.capitalize()} '{name}' acts on body '{B['name']}' but has no face (selection) – give the attachment face")


# ---------------------------------------------------------------- instants
def instant_loads(study, res, k: int, body=None, parts=None, support_joints=None, load_kind="remote_distributing",
                  name: str | None = None) -> dict:
    """Load state at the time res.t[k] (see the module). Returns {load_case, t, expected, supports, notes}."""
    a, meta = res.arrays, res.meta
    i, B, att, sup, fe_parts = _prepare(study, res, body, parts, support_joints)
    items = _force_items(study, res, i)
    _check_items(items, B)
    R = a["rot"][k, i]
    g = np.asarray(meta.get("gravity") or [0, 0, 0], float)
    c0 = np.asarray(B["com"], float)
    loads = []
    notes = []
    expected = {}
    jn = res.joint_names
    multi = len(sup) > 1
    for j, side in att:
        J = meta["joints"][j]
        F, M = _joint_wrench(res, j, side, k)
        Fc, Mc = R.T @ F, R.T @ M
        sel, pt = _joint_selection(J, side), _joint_point(J, side)
        if J["name"] in sup:
            expected[sel] = {"force": Fc.tolist(), "moment": Mc.tolist(), "point": pt.tolist(), "joint": J["name"]}
            continue
        if load_kind == "bearing" and J["kind"] in ("revolute", "cylindrical", "pin_slot"):
            loads.append(BearingLoad(selection=sel, vector=Fc.tolist()))
            if np.linalg.norm(Mc) > 1e-12 * max(np.linalg.norm(Fc), 1.0):
                loads.append(RemoteLoad(selection=sel, point=pt.tolist(), force=[0.0, 0.0, 0.0], moment=Mc.tolist(),
                                        coupling="distributing"))
        else:
            loads.append(RemoteLoad(selection=sel, point=pt.tolist(), force=Fc.tolist(), moment=Mc.tolist(),
                                    coupling="distributing"))
    for nm, sel, pt, wr, what in items:
        F, M = wr(k)
        if np.linalg.norm(F) == 0 and np.linalg.norm(M) == 0:
            continue
        loads.append(RemoteLoad(selection=sel, point=pt.tolist(), force=(R.T @ F).tolist(), moment=(R.T @ M).tolist(),
                                coupling="distributing"))
    acc = R.T @ (a["acc"][k, i] - g)
    loads.append(Inertia(point=c0.tolist(), accel=acc.tolist(), omega=(R.T @ a["omega"][k, i]).tolist(),
                         alpha=(R.T @ a["alpha"][k, i]).tolist(), parts=list(fe_parts),
                         name="inertia and gravity (MBD)"))
    t = float(a["t"][k])
    name = name or f"MBD t = {t:.6g} s" + (f" – {B['name']}" if len([b for b in meta["bodies"] if not b["ground"]]) > 1
                                           else "")
    _set_case(study, name, loads)
    sup_sel = {_joint_selection(meta["joints"][jn.index(s)], dict(att)[jn.index(s)]) for s in sup}
    study.constraints = [c for c in study.constraints if not (isinstance(c, RemoteSupport) and c.selection in sup_sel)]
    supports = []
    for n, s in enumerate(sup):
        j = jn.index(s)
        side = dict(att)[j]
        J = meta["joints"][j]
        c = _support_constraint(J, side, _joint_selection(J, side), _joint_point(J, side), n == 0 or not multi, notes)
        if n == 0:
            c.dofs = [True] * 6
        study.constraints.append(c)
        supports.append(c.selection)
    if multi:
        notes.append("Several support joints (redundant joints) – FEA determines how the reactions are shared; the MBD split is ambiguous, only the sum is compared")
    if B.get("printed"):
        notes.append("The mass of printed parts in MBD is approximate – the FEA support reaction may differ from MBD by the difference in inertia forces (density from the print settings on the mesh)")
    return {"load_case": name, "t": t, "index": int(k), "body": B["name"], "supports": supports,
            "expected": expected, "notes": notes, "parts": fe_parts}


def check_relief(info: dict, step) -> dict:
    """Check of the inertia relief: reaction of the FE in the support (StepResult.remote) = reaction of the joint from the MBD.
    Returns {max_rel, sum_force_N (FE), sum_force_mbd_N, scale}; max_rel against Σ|F| of the loads.
    """
    exp = info["expected"]
    tot_mkp = np.zeros(3)
    tot_mbd = np.zeros(3)
    scale = 0.0
    for sel, e in exp.items():
        r = step.remote.get(sel)
        if r is None:
            continue
        tot_mkp += np.asarray(r["force"], float)
        tot_mbd += np.asarray(e["force"], float)
        scale += np.linalg.norm(e["force"])
    lc_scale = scale + sum(np.linalg.norm(v) for v in step.reactions.values()) if step.reactions else scale
    rel = float(np.linalg.norm(tot_mkp - tot_mbd) / max(lc_scale, 1e-300))
    return {"max_rel": rel, "sum_force_N": tot_mkp.tolist(), "sum_force_mbd_N": tot_mbd.tolist(), "scale": lc_scale}


# ---------------------------------------------------------------- course → unit states
def history_loads(study, res, body=None, parts=None, support_joints=None, prefix: str = "MBD", skip_zero=True) -> dict:
    """Unit states + channels of the fatigue (see the module). Returns {load_cases, channels, n_states, notes}."""
    from ..fatigue.history import Channel, LoadHistory
    from ..fatigue.settings import FatigueSettings
    a, meta = res.arrays, res.meta
    i, B, att, sup, fe_parts = _prepare(study, res, body, parts, support_joints)
    items = _force_items(study, res, i)
    _check_items(items, B)
    t = a["t"]
    R = a["rot"][:, i]
    RT = np.transpose(R, (0, 2, 1))
    g = np.asarray(meta.get("gravity") or [0, 0, 0], float)
    c0 = np.asarray(B["com"], float)
    jn = res.joint_names
    cases: list[tuple[str, list, np.ndarray]] = []       # (name, load, channel (N,))
    series = []
    for j, side in att:
        J = meta["joints"][j]
        if J["name"] in sup:
            continue
        F, M = _joint_wrench(res, j, side)
        series.append((J["name"], _joint_selection(J, side), _joint_point(J, side), F, M))
    for nm, sel, pt, wr, what in items:
        F, M = wr(None)
        series.append((nm, sel, pt, F, M))
    for nm, sel, pt, F, M in series:
        Fc = np.einsum("nij,nj->ni", RT, F)
        Mc = np.einsum("nij,nj->ni", RT, M)
        vals = np.column_stack([Fc, Mc])
        for c in range(6):
            unit = [0.0] * 3
            unit[c % 3] = 1.0
            ld = RemoteLoad(selection=sel, point=[float(x) for x in pt], coupling="distributing",
                            force=unit if c < 3 else [0.0, 0.0, 0.0], moment=unit if c >= 3 else [0.0, 0.0, 0.0])
            cases.append((f"{prefix} {nm} {COMP[c]} ({'1 N' if c < 3 else '1 N·mm'})", [ld], vals[:, c]))
    acc = np.einsum("nij,nj->ni", RT, a["acc"][:, i] - g)
    om = np.einsum("nij,nj->ni", RT, a["omega"][:, i])
    al = np.einsum("nij,nj->ni", RT, a["alpha"][:, i])
    W = np.einsum("ni,nj->nij", om, om) - np.einsum("ni,ni->n", om, om)[:, None, None] * np.eye(3)[None]
    lab = "xyz"
    for c in range(3):
        unit = [0.0] * 3
        unit[c] = 1.0
        cases.append((f"{prefix} inertia a0{lab[c]} (1 mm/s²)",
                      [Inertia(point=c0.tolist(), accel=unit, parts=list(fe_parts))], acc[:, c]))
    for c in range(3):
        unit = [0.0] * 3
        unit[c] = 1.0
        cases.append((f"{prefix} inertia α{lab[c]} (1 rad/s²)",
                      [Inertia(point=c0.tolist(), alpha=unit, parts=list(fe_parts))], al[:, c]))
    for p, q in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        E = np.zeros((3, 3))
        E[p, q] = E[q, p] = 1.0
        cases.append((f"{prefix} inertia W{lab[p]}{lab[q]} (ω_iω_j = 1)",
                      [Inertia(point=c0.tolist(), wmat=E.tolist(), parts=list(fe_parts))], W[:, p, q]))
    peak = max((np.abs(ch).max() for _, _, ch in cases if len(ch)), default=0.0)
    notes = []
    names = []
    channels = []
    skipped = []
    old = {nm for nm, _, _ in cases}
    _remove_old(study, old, set())
    for nm, loads, ch in cases:
        if skip_zero and np.abs(ch).max() <= 1e-12 * max(peak, 1e-300):
            skipped.append(nm)
            continue
        _set_case(study, nm, loads)
        names.append(nm)
        channels.append(Channel(load_case=nm, kind="points", points=[[float(tt), float(v)] for tt, v in zip(t, ch)]))
    # storage (the same as for the instants)
    sup_sel = set()
    for n, s in enumerate(sup):
        j = jn.index(s)
        side = dict(att)[j]
        J = meta["joints"][j]
        sel = _joint_selection(J, side)
        sup_sel.add(sel)
    study.constraints = [c for c in study.constraints if not (isinstance(c, RemoteSupport) and c.selection in sup_sel)]
    for n, s in enumerate(sup):
        j = jn.index(s)
        side = dict(att)[j]
        J = meta["joints"][j]
        c = _support_constraint(J, side, _joint_selection(J, side), _joint_point(J, side), n == 0 or len(sup) == 1, notes)
        if n == 0:
            c.dofs = [True] * 6
        study.constraints.append(c)
    hist = LoadHistory(channels=channels, duration=float(t[-1] - t[0]) if len(t) > 1 else 1.0)
    if study.fatigue is None:
        study.fatigue = FatigueSettings(history=hist)
    else:
        study.fatigue.history = hist
    if skipped:
        notes.append(f"{len(skipped)} zero components skipped (planar motion etc.)")
    notes.append("The remote support forces the CalculiX solver (NGSolve cannot handle it) – number of cases × time per case")
    return {"load_cases": names, "n_states": len(names), "skipped": skipped, "body": B["name"],
            "supports": sorted(sup_sel), "notes": notes, "parts": fe_parts, "duration_s": hist.duration}


def superpose(res_fe, study, t: float) -> np.ndarray:
    """Stress (N, 6) by the superposition of the unit states with the multiples of the channels of Study.fatigue.history at the time t."""
    hist = study.fatigue.history
    names = [s.name for s in res_fe.steps]
    S = np.zeros_like(res_fe.steps[0].S)
    for ch in hist.channels:
        if ch.load_case not in names:
            continue
        f = float(ch.value(np.array([t]), hist.duration)[0])
        S += f * res_fe.steps[names.index(ch.load_case)].S
    return S


# ---------------------------------------------------------------- entry point
def export_loads(study, res, mode: str | None = None, times: list | None = None, quantity: str | None = None,
                 body: str | None = None, parts: list[str] | None = None, support_joints: list[str] | None = None,
                 load_kind: str | None = None) -> dict:
    """Transfer according to Study.mechanism.run.to_fea (the arguments overwrite it): mode instants | max | history."""
    cfg = study.mechanism.run.to_fea
    mode = mode or cfg.mode
    parts = parts if parts is not None else (cfg.parts or None)
    support_joints = support_joints if support_joints is not None else (cfg.support_joints or None)
    load_kind = load_kind or cfg.load_kind
    if mode == "history":
        out = history_loads(study, res, body, parts, support_joints)
        out["mode"] = "history"
        if load_kind == "bearing":
            out["notes"].append("Full history: BearingLoad is not superposable – a remote load (RBE3) is used")
        return out
    if mode == "max":
        q = quantity or cfg.quantity
        if not q:
            i, B, att, sup, _ = _prepare(study, res, body, parts, support_joints)
            q = "reaction:" + res.joint_names[att[0][0]]
        mx = res.find_max(q)
        info = instant_loads(study, res, mx["index"], body, parts, support_joints, load_kind,
                             name=f"MBD max {q} (t = {mx['t']:.6g} s)")
        return {"mode": "max", "maximum": mx, "cases": [info], "load_cases": [info["load_case"]]}
    tl = list(times if times is not None else cfg.times)
    if not tl:
        raise ToFeaError("Specify the instant times (times)")
    infos = [instant_loads(study, res, res.time_index(t), body, parts, support_joints, load_kind) for t in tl]
    return {"mode": "instants", "cases": infos, "load_cases": [x["load_case"] for x in infos]}


def describe_joint(J) -> str:
    return f"{J['name']} ({JOINT_LABELS.get(J['kind'], J['kind'])})"
