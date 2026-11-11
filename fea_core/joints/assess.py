"""Assessment of bolts (VDI 2230, EN 1993-1-8) and welds (EN 1993-1-8 directional method, fatigue
EN 1993-1-9 / IIW) from the FE results.

The static assessments take each load state separately (the state "Bolt preload" is not computed – the preload
is analytical in VDI) and the worst one decides. Fatigue superposes the states according to the load curve of the study
(Study.fatigue): rainflow of the forces in the bolt / stresses in the effective section of the weld, Palmgren–Miner.
The design values of EN 1993-1-8 assume that the loads in the study are design loads (the load factors γ_F are given by the user in the loads).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import tables
from .fem import bolt_basics, bolt_materials
from .model import BoltSpec, WeldSpec, PRELOAD_CASE, PRELOAD_CASE_LEGACY, is_preload
from .vdi2230 import assess as vdi_assess, fatigue_damage as vdi_fatigue


@dataclass
class JointCheck:
    kind: str                         # "bolt" | "weld"
    name: str
    label: str
    utilization: float                # max. utilisation (≤ 1 passes)
    governing: str
    rows: list[tuple[str, float, str]] = field(default_factory=list)   # (assessment, utilisation, details)
    data: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.utilization <= 1.0


@dataclass
class JointsAssessment:
    bolts: list[JointCheck] = field(default_factory=list)
    welds: list[JointCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cases: list[str] = field(default_factory=list)
    fatigue: bool = False

    def all(self) -> list[JointCheck]:
        return self.bolts + self.welds


INFO = "(informative) "


def _rows_max(rows) -> tuple[float, str]:
    """Largest utilisation; rows starting with INFO are not counted."""
    rows = [r for r in rows if not r[0].startswith(INFO) and math.isfinite(r[1])] or \
        [r for r in rows if not r[0].startswith(INFO)]
    if not rows:
        return 0.0, ""
    u, name = max((r[1], r[0]) for r in rows)
    return u, name


def _perp(v, e):
    v = np.asarray(v, float)
    return v - (v @ e) * e


def _cases(results) -> list[int]:
    return [i for i, s in enumerate(results.steps) if not is_preload(s.name)]


def _fatigue_factors(study, results, idx: list[int], why: list | None = None):
    """(F, repeats, closed) of the fatigue history, None = no joint fatigue; why (list) receives the reason of an error."""
    st = study.fatigue
    if st is None or not st.history.channels:
        return None
    names = [results.steps[i].name for i in idx]
    try:
        t, F = st.history.factors(names)
    except Exception as e:  # noqa: BLE001 – e.g. the channel refers to a state that does not exist → reported by assess_joints
        if why is not None:
            why.append(str(e))
        return None
    if len(t) < 2 or not np.any(F):
        return None
    return F, float(st.repeats), bool(st.closed_blocks)


def _cycles(series: np.ndarray, closed: bool) -> np.ndarray:
    from ..fatigue.kernel import cycles_of
    return cycles_of(np.asarray(series, float), closed)


def _dominant(series2: np.ndarray) -> np.ndarray:
    """Projection of the vector history (T, 2|3) onto the direction of the largest change (exact for proportional loading)."""
    X = np.asarray(series2, float)
    if not np.any(X):
        return np.zeros(len(X))
    Xc = X - X.mean(axis=0)
    if not np.any(Xc):
        u = X[np.argmax(np.linalg.norm(X, axis=1))]
    else:
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        u = vt[0]
    u = u / max(np.linalg.norm(u), 1e-300)
    return X @ u


# ===================================================================== bolts

def group_shear(bolts: list[BoltSpec], forces: dict[str, dict]) -> None:
    """Shear force per bolt by the group method (EN 1993-1-8, rigid plate): for each pair of parts the force and the moment transferred
    through the joint plane by all cells and bolts are summed and distributed:
    Q_i = Q/n + M_t × r_i / Σ r²  (r_i from the centroid of the group in the plane of the joint).
    Changes forces[bolt]["planes"][k]: "Q" (group), "Q_fe" (by plane), "F_Q".
    """
    groups: dict[tuple[str, str], list[tuple[BoltSpec, int]]] = {}
    for b in bolts:
        c = forces.get(b.name)
        if not c:
            continue
        for k, p in enumerate(c["planes"]):
            groups.setdefault(tuple(sorted((p["upper"], p["lower"]))), []).append((b, k))
    for members in groups.values():
        e = members[0][0].e
        if any(abs(float(b.e @ e)) < 0.999 for b, _ in members):
            continue                                   # different directions of the axes – the group method does not hold
        ref = forces[members[0][0].name]["planes"][members[0][1]]["upper"]   # forces on this part
        sg = [1.0 if forces[b.name]["planes"][k]["upper"] == ref else -1.0 for b, k in members]
        pts = np.array([b.point(forces[b.name]["planes"][k]["s"]) for b, k in members])
        c0 = pts.mean(axis=0)
        Ftot, Mt = np.zeros(3), 0.0
        for (b, k), p, s in zip(members, pts, sg):
            pl = forces[b.name]["planes"][k]
            F = s * np.asarray(pl["F_tot"])
            Ftot += F
            Mt += float((s * np.asarray(pl["M_tot"]) + np.cross(p - c0, F)) @ e)
        Q = _perp(Ftot, e) / len(members)
        r = np.array([_perp(p - c0, e) for p in pts])
        J = float((r * r).sum())
        for (b, k), ri, s in zip(members, r, sg):
            pl = forces[b.name]["planes"][k]
            qi = s * (Q + (np.cross(e * Mt, ri) / J if J > 1e-12 else 0.0))
            pl["Q_fe"] = pl["Q"]
            pl["Q"] = qi
            pl["F_Q"] = float(np.linalg.norm(qi))
            pl["group"] = len(members)


def bolt_case_forces(b: BoltSpec, sj: dict) -> dict:
    """Forces of one state: ends of the bolt and the planes of the joints (see joints.evaluate)."""
    e = b.e
    ends = sj.get("ends", [])
    planes = []
    for pl in sj.get("planes", []):
        Fb, Fi = np.asarray(pl["F_bolt"]), np.asarray(pl["F_int"])
        Ft = Fb + Fi
        planes.append({
            "s": pl["s"], "upper": pl["upper"], "lower": pl["lower"], "area": pl["area"],
            "F_tot": Ft, "M_tot": np.asarray(pl["M_int"]) + np.asarray(pl["M_bolt"]),
            "F_A": float(Ft @ e), "F_SA": float(Fb @ e), "F_PA": float(Fi @ e),
            "Q": _perp(Ft, e), "F_Q": float(np.linalg.norm(_perp(Ft, e))),
            "Q_bolt": float(np.linalg.norm(_perp(Fb, e))),
            "M_b": float(np.linalg.norm(_perp(pl["M_bolt"], e))),
        })
    F_SA = float(np.asarray(ends[0]["F"]) @ e) if ends else 0.0
    Mb_ends = [float(np.linalg.norm(_perp(en["M"], e))) for en in ends]
    return {"planes": planes, "F_SA": F_SA, "M_b": max(Mb_ends + [p["M_b"] for p in planes] + [0.0]),
            "ends": ends}


def _bearing_geometry(b: BoltSpec, all_bolts: list[BoltSpec], parts, plate: dict, u: np.ndarray, d0: float):
    """e1, e2, p1, p2 for the bearing of the sheet in the direction of the force of the bolt on the sheet u."""
    from .detect import edge_distances
    pmap = {p.name: p for p in parts}
    part = pmap.get(plate["part"])
    if part is None or np.linalg.norm(u) < 1e-12:
        return None
    e = b.e
    c = b.point(0.5 * (plate["s0"] + plate["s1"]))
    e1, e2 = edge_distances(part, c, e, u, 0.5 * d0)
    uu = _perp(u, e)
    uu /= np.linalg.norm(uu)
    vv = np.cross(e, uu)
    p1 = p2 = math.inf
    for o in all_bolts:
        if o is b or plate["part"] not in [pl["part"] for pl in o.plates]:
            continue
        dlt = _perp(np.asarray(o.origin) - np.asarray(b.origin), e)
        a, l = float(dlt @ uu), float(dlt @ vv)
        if a > 0.5 * d0 and abs(l) < d0:
            p1 = min(p1, a)
        if abs(a) < d0 and abs(l) > 0.5 * d0:
            p2 = min(p2, abs(l))
    return {"e1": e1, "e2": e2, "p1": p1, "p2": p2}


def _preload_state(study, results):
    """Nonlinear calculation: bolt forces at the end of the preload step (state "Bolt preload" or a step of the sequence
    with bolts = preload) – the results of the states contain the preload, the VDI / EN assessment waits for the operating (additional) forces.
    """
    if not study.is_nonlinear():
        return None
    names = {PRELOAD_CASE, PRELOAD_CASE_LEGACY} | {s.name for s in getattr(study.analysis, "sequence", []) or []
                              if getattr(s, "bolts", "") == "preload"}
    for s in results.steps:
        if s.name in names and (s.joints or {}).get("bolts"):
            return s.joints["bolts"]
    return None


def _minus_preload(sj: dict, pre: dict) -> dict:
    """Forces of a state without the forces after the preload (ends of the bolt and the planes of the joints, the same order)."""
    out = dict(sj)
    if len(sj.get("ends", [])) == len(pre.get("ends", [])):
        out["ends"] = [{**a, "F": (np.asarray(a["F"]) - np.asarray(b["F"])).tolist(),
                        "M": (np.asarray(a["M"]) - np.asarray(b["M"])).tolist()}
                       for a, b in zip(sj["ends"], pre["ends"])]
    if len(sj.get("planes", [])) == len(pre.get("planes", [])):
        out["planes"] = [{**a, **{k: (np.asarray(a[k]) - np.asarray(b[k])).tolist()
                                  for k in ("F_bolt", "F_int", "M_int", "M_bolt") if k in a and k in b}}
                         for a, b in zip(sj["planes"], pre["planes"])]
    return out


def case_forces(study, results, idx: list[int]) -> list[tuple[str, dict]]:
    """Forces of all bolts by states; the shear force according to Study.joints.shear_split. In a nonlinear calculation (the states
    contain the preload) the forces after the preload are subtracted – the assessment works with the operating forces as in superposition
    (earlier the preload was counted twice: example 9 – utilisation 1.39 instead of ~0.9).
    """
    bolts = study.active_bolts()
    pre = _preload_state(study, results)
    out = []
    for i in idx:
        bj = results.steps[i].joints.get("bolts", {})
        fb = {b.name: bolt_case_forces(b, _minus_preload(bj[b.name], pre[b.name])
                                       if pre is not None and b.name in pre else bj[b.name])
              for b in bolts if b.name in bj}
        if study.joints.shear_split == "group":
            group_shear(bolts, fb)
        out.append((results.steps[i].name, fb))
    return out


def assess_bolt(study, b: BoltSpec, results, idx: list[int], parts=None, ff=None, forces=None) -> JointCheck:
    js = study.joints
    th = b.thread
    sc = tables.strength_class(b.strength, b.size)
    bas = bolt_basics(study, b)
    E_S, plate_E, E_M, Rp, Rm_plate = bolt_materials(study, b)
    warns = []
    forces = forces if forces is not None else case_forces(study, results, idx)
    per_case = [(name, fb[b.name]) for name, fb in forces if b.name in fb]
    from .fem import has_gap
    package = has_gap(b) and b.connection != "fitted"
    if package and per_case:
        # a bundle with a clearance in CAD: in the FE the whole bundle is connected to the beam → the beam also carries the share of the clamped
        # parts. The axial force in the bolt by VDI: F_SA = Φ_n·F_A, Φ_n = n·Φ_K (n = 0.5), the bending of the beam ≠ the bolt.
        phi_n = 0.5 * bas["phi_K"]
        for _, c in per_case:
            for p in c["planes"]:
                p["F_SA"], p["F_PA"] = phi_n * p["F_A"], (1 - phi_n) * p["F_A"]
                p["M_b"] = 0.0
            if c["planes"]:
                c["F_SA"] = max((p["F_SA"] for p in c["planes"]), key=abs)
            c["M_b"] = 0.0
            for en in c["ends"]:
                en["M"] = [0.0, 0.0, 0.0]
        warns.append(f"Clearance between clamped parts in CAD: FEA clamped stack, bolt load share Φ_n = 0.5·Φ_K = {phi_n:.3f} (VDI), bolt bending from FEA is not assessed")
    if any(E < 20000 for E in plate_E.values()):
        warns.append("Clamped plastic parts: settlement and creep (preload loss over time) are not covered by VDI 2230 – the actual residual clamp force will be lower; limit the tightening torque, consider metal inserts")
    if not per_case:
        return JointCheck("bolt", b.name, b.designation, math.nan, "no results", warnings=["Bolt has no forces from the analysis"])
    # --- maxima over the states
    def mx(key, src="planes"):
        vals = [p[key] for _, c in per_case for p in c[src]] if src == "planes" else [c[key] for _, c in per_case]
        return max(vals + [0.0])
    w = {"F_A": mx("F_A"), "F_SA": max(mx("F_SA", "bolt"), 0.0), "F_PA": mx("F_PA"), "F_Q": mx("F_Q"),
         "M_b": mx("M_b", "bolt")}
    # --- fatigue (VDI): σ(t) = F_SA(t)/A_S + bending (projection onto the main direction)/W_b
    fat = None
    if ff is not None:
        F, repeats, closed = ff
        e = b.e
        fsa = np.array([c["F_SA"] for _, c in per_case])
        mvec = np.array([_perp(c["ends"][0]["M"], e) if c["ends"] else np.zeros(3) for _, c in per_case])
        W_b = math.pi * th.ds ** 3 / 32
        sig = F @ fsa / th.As + _dominant(F @ mvec) / W_b
        cyc = _cycles(sig, closed)
        if len(cyc):
            fd = vdi_fatigue((cyc[:, 0] / 2).tolist(), cyc[:, 2].tolist(), b.size)
            fat = {"sigma_a": fd["sigma_a"], "damage": fd["damage"] * repeats, "damage_block": fd["damage"],
                   "repeats": repeats}
    v = vdi_assess(b, bas, w, Rm_plate, min(plate_E.values()) if plate_E else 210000.0, fat)
    rows = []
    for name, u, txt in v["checks"]:
        if name.startswith("Slip") and js.category == "A":
            name = INFO + name + " – category A connection transmits the shear force by bearing"
        if name.startswith("Assembly") and bas["preload_mode"] == "auto":
            name = INFO + name + " – automatic preload min(F_M,zul; 0.9·p_G·A_p)"
        elif b.connection == "fitted" and name.startswith("Slip") and not name.startswith(INFO):
            name = INFO + name + " – fitted bolt transmits the shear force through the shank"
        rows.append((f"VDI 2230: {name}" if not name.startswith(INFO) else INFO + "VDI 2230: " + name[len(INFO):],
                     u, txt))
    if sc.stainless and fat:
        warns.append("Stainless steel bolt: VDI σ_ASV applies to steel bolts – fatigue is approximate only")
    strip = thread_strip_rows(study, b, v["F_Smax"], rows, warns)
    # --- EN 1993-1-8
    ec = {}
    fub = sc.fub or sc.Rm
    if sc.fub is None:
        warns.append(f"Class {sc.name} is not in EN 1993-1-8 – f_ub = Rm = {sc.Rm:g} MPa used")
    g2 = js.gamma_M2
    fitted = b.connection == "fitted"
    if fitted:                                   # fitted bolt: shear of the shank, α_v = 0.6 (EN 1993-1-8, 3.6.1)
        A_v, alpha_v = math.pi * b.shank ** 2 / 4, 0.6
    elif b.thread_in_shear:
        A_v = th.As
        alpha_v = 0.5 if (sc.stainless or sc.name in ("4.8", "5.8", "6.8", "10.9", "12.9", "9.8")) else 0.6
    else:
        A_v, alpha_v = th.AN, 0.6
    Fv_Rd = alpha_v * fub * A_v / g2
    k2 = 0.63 if b.head == "countersunk" else 0.9
    Ft_Rd = k2 * fub * th.As / g2
    d0 = b.shank if fitted else (b.hole or (b.size + (1 if b.size <= 14 else (2 if b.size <= 24 else 3))))
    d_b = b.shank if fitted else b.size          # diameter for bearing (fitted = the shank)
    n_planes = max(len(per_case[0][1]["planes"]), 1)
    Fv_Ed_planes = [max(c["planes"][k]["F_Q"] for _, c in per_case) if c_has(per_case, k) else 0.0
                    for k in range(n_planes)]
    Fv_Ed = max(Fv_Ed_planes + [0.0])
    Ft_Ed = max(max(max(p["F_A"], p["F_SA"]) for p in c["planes"]) if c["planes"] else c["F_SA"]
                for _, c in per_case)
    Ft_Ed = max(Ft_Ed, 0.0)
    split = ("bolt group: Q/n + torsion" if any(p.get("group", 1) > 1 for _, c in per_case for p in c["planes"])
             else ("single bolt in the joint interface" if js.shear_split == "group" else "from the FEA bonded interface"))
    rows.append(("EN 1993-1-8: shear F_v,Ed ≤ F_v,Rd" + (" (fitted bolt shank)" if fitted else ""), Fv_Ed / Fv_Rd,
                 f"{Fv_Ed:.0f} / {Fv_Rd:.0f} N (α_v = {alpha_v:g}, A = {A_v:.1f} mm²"
                 + (f", d_s = {b.shank:g} mm" if fitted else "") + f"; {split})"))
    rows.append(("EN 1993-1-8: tah F_t,Ed ≤ F_t,Rd", Ft_Ed / Ft_Rd, f"{Ft_Ed:.0f} / {Ft_Rd:.0f} N"))
    rows.append(("EN 1993-1-8: shear + tension", Fv_Ed / Fv_Rd + Ft_Ed / (1.4 * Ft_Rd), "F_v/F_v,Rd + F_t/(1,4 F_t,Rd)"))
    # punching through by the head / nut
    head = tables.head_dims(b.size, b.head, b.standard)
    if b.head == "socket":
        dm_head = head.get("dk", head["dw"])
    else:
        dm_head = 0.5 * (head["s"] + head["s"] * 2 / math.sqrt(3))
    nut = tables.nut_dims(b.size)
    dm_nut = 0.5 * (nut["s"] + nut["s"] * 2 / math.sqrt(3))
    for pl, dm, side in ((b.plates[0], dm_head, "head"),
                         (b.plates[-1] if b.connection == "nut" else None, dm_nut, "nut")):
        if pl is None:
            continue
        try:
            fu_p = study.material_of(pl["part"]).Rm
        except KeyError:
            fu_p = None
        if not fu_p:
            continue
        t_p = pl["s1"] - pl["s0"]
        Bp = 0.6 * math.pi * dm * t_p * fu_p / g2
        rows.append((f"EN 1993-1-8: punching shear ({side}, {pl['part']})", Ft_Ed / Bp, f"{Ft_Ed:.0f} / {Bp:.0f} N"))
    # bearing of the walls of the holes (direction of the force from the state with the largest shear)
    if per_case[0][1]["planes"]:
        gov = max(per_case, key=lambda kc: max(p["F_Q"] for p in kc[1]["planes"]))[1]
        Q = [p["Q"] for p in gov["planes"]]
        plates = b.plates + ([{"part": b.nut_part, "s0": b.nut_s, "s1": b.thread_end}]
                             if b.connection == "tapped" else [])
        zero = np.zeros(3)
        for k, pl in enumerate(plates):
            # force of the bolt on the sheet k = transfer by the plane below it − transfer by the plane above it
            Fk = (Q[k] if k < len(Q) else zero) - (Q[k - 1] if k >= 1 else zero)
            Fb_Ed = float(np.linalg.norm(Fk))
            if Fb_Ed < 1e-9:
                continue
            try:
                fu_p = study.material_of(pl["part"]).Rm
            except KeyError:
                fu_p = None
            if not fu_p:
                warns.append(f"Bearing '{pl['part']}' not assessed – material has no Rm")
                continue
            t = pl["s1"] - pl["s0"]
            geo = _bearing_geometry(b, study.active_bolts(), parts, pl, Fk, d0) if parts else None
            if geo is None:
                alpha_d, k1, gtxt = 1.0, 2.5, "no edge geometry (α_d = 1, k1 = 2.5) – approximate only"
                warns.append(f"Bearing '{pl['part']}': edge distances not determined")
            else:
                e1, e2, p1, p2 = geo["e1"], geo["e2"], geo["p1"], geo["p2"]
                alpha_d = e1 / (3 * d0) if math.isfinite(e1) else 1.0
                if math.isfinite(p1):
                    alpha_d = min(alpha_d, p1 / (3 * d0) - 0.25)
                k1 = 2.5
                if math.isfinite(e2):
                    k1 = min(k1, 2.8 * e2 / d0 - 1.7)
                if math.isfinite(p2):
                    k1 = min(k1, 1.4 * p2 / d0 - 1.7)
                gtxt = (f"e1 = {e1:.1f}, e2 = {e2:.1f}" + (f", p1 = {p1:.1f}" if math.isfinite(p1) else "")
                        + (f", p2 = {p2:.1f}" if math.isfinite(p2) else "") + f" mm, d0 = {d0:.1f} mm")
                for nm, val, lim in (("e1", e1, 1.2), ("e2", e2, 1.2), ("p1", p1, 2.2), ("p2", p2, 2.4)):
                    if math.isfinite(val) and val < lim * d0:
                        warns.append(f"'{pl['part']}': {nm} = {val:.1f} mm < {lim:g}·d0 = {lim * d0:.1f} mm (EN 1993-1-8 Table 3.3)")
            alpha_b = max(min(alpha_d, fub / fu_p, 1.0), 0.0)
            k1 = max(k1, 0.0)
            Fb_Rd = k1 * alpha_b * fu_p * d_b * t / g2
            u = Fb_Ed / Fb_Rd if Fb_Rd > 0 else math.inf
            rows.append((f"EN 1993-1-8: bearing '{pl['part']}' (t = {t:g} mm)", u,
                         f"{Fb_Ed:.0f} / {Fb_Rd:.0f} N, k1 = {k1:.2f}, α_b = {alpha_b:.2f}; {gtxt}"))
    # friction joint (category B/C)
    if js.category in ("B", "C"):
        Fp_C = 0.7 * fub * th.As
        n_f = n_planes
        hole_ok = d0 <= b.size + (2 if b.size <= 24 else 3) + 1e-6
        ks = 1.0 if hole_ok else 0.85
        Fs_Rd = ks * n_f * js.slip_mu * (Fp_C - 0.8 * Ft_Ed) / js.gamma_M3
        Fv_tot = sum(Fv_Ed_planes)
        rows.append((f"EN 1993-1-8: slip (cat. {js.category})", Fv_tot / Fs_Rd if Fs_Rd > 0 else math.inf,
                     f"{Fv_tot:.0f} / {Fs_Rd:.0f} N (μ = {js.slip_mu:g}, n = {n_f}, k_s = {ks:g}, "
                     f"F_p,C = {Fp_C:.0f} N)"))
    ec.update({"Fv_Ed": Fv_Ed, "Fv_Rd": Fv_Rd, "Ft_Ed": Ft_Ed, "Ft_Rd": Ft_Rd, "d0": d0})
    util, gov_name = _rows_max(rows)
    data = {"vdi": v, "basics": bas, "ec3": ec, "stripping": strip, "per_case": [
        {"case": n, "F_SA": c["F_SA"], "M_b": c["M_b"],
         "planes": [{k: (val.tolist() if isinstance(val, np.ndarray) else val) for k, val in p.items()}
                    for p in c["planes"]]} for n, c in per_case]}
    if b.note:
        warns.append(b.note)
    return JointCheck("bolt", b.name, b.designation, util, gov_name, rows, data, warns)


def thread_strip_rows(study, b: BoltSpec, F_Smax: float, rows: list, warns: list) -> dict | None:
    """Stripping of the thread (VDI 2230): the nut (assumption: the class matches the bolt, ISO 898-2) or the internal thread of the part.

    Rows: (1) F_S,max = F_M,max + F_SA ≤ min(F_mGM, F_mGS) at the real length, (2) thread length m ≥ m_ges,min
    (the bolt breaks before the thread strips – tensile failure mode; for plastic only informative).
    Effective length m_eff = m − 0.8 P (chamfer at the start of the thread).
    """
    from .vdi2230 import stripping, tau_ratio_bolt, tau_ratio_part
    if b.connection not in ("nut", "tapped"):
        return None
    th = b.thread
    sc = tables.strength_class(b.strength, b.size)
    if b.connection == "nut":
        nd = tables.nut_dims(b.size, b.standard)
        m = b.nut_m or nd["m"]
        Rm_M, tau_M, mat = sc.Rm, tau_ratio_part(sc.E)[0], f"nut of class matching {sc.name} (ISO 898-2)"
        s_nut, where = nd["s"], "nut"
        plastic = False
    else:
        m = b.thread_end - b.nut_s
        try:
            mm = study.material_of(b.nut_part)
        except KeyError:
            return None
        if not mm.Rm:
            warns.append(f"Thread stripping in '{b.nut_part}' not assessed – material has no Rm")
            return None
        tau_M, kind = tau_ratio_part(mm.E)
        Rm_M, mat, s_nut, where = mm.Rm, f"{mm.name} ({kind})", None, f"internal thread '{b.nut_part}'"
        plastic = mm.E < 20000
    m_eff = m - 0.8 * th.P
    if m_eff <= 0:
        rows.append((f"VDI 2230: thread stripping ({where})", math.inf, f"thread length {m:.3g} mm ≤ 0.8 P"))
        return None
    r = stripping(th.d, th.P, m_eff, sc.Rm, tau_ratio_bolt(sc.name), Rm_M, tau_M, s_nut, th.As)
    r.update(m=m, material=mat, where=where)
    F_G = min(r["F_mGM"], r["F_mGS"])
    rows.append((f"VDI 2230: thread stripping ({where}): F_S,max ≤ F_m,G", F_Smax / F_G,
                 f"{F_Smax:.0f} / {F_G:.0f} N (governing: {r['governs']}; F_mGM = {r['F_mGM']:.0f}, F_mGS = {r['F_mGS']:.0f} N, m_eff = {m_eff:.3g} mm, R_S = {r['R_S']:.2f}, C1 = {r['C1']:.3f}, C2 = {r['C2']:.3f}, C3 = {r['C3']:.3f}; {mat})"))
    name = f"VDI 2230: thread length m ≥ m_ges,min ({where} – the bolt breaks before the thread strips)"
    rows.append(((INFO + name) if plastic else name, r["m_ges_min"] / m,
                 f"m = {m:.3g} mm, m_ges,min = {r['m_ges_min']:.3g} mm (F_mS = R_m·A_S = {r['F_mS']:.0f} N)"))
    if r["note"]:
        warns.append(f"Thread stripping: {r['note']}")
    if plastic:
        warns.append(f"Thread in plastic '{b.nut_part}': stripping approximate only (creep, thread-forming screw, temperature effect) – use threaded inserts for repeated assembly")
    return r


def c_has(per_case, k) -> bool:
    return all(len(c["planes"]) > k for _, c in per_case)


# ===================================================================== svary

def beta_w(Re: float | None) -> float:
    """Correlation factor β_w (EN 1993-1-8 table 4.1) according to the yield strength."""
    if not Re:
        return 1.0
    if Re <= 235 + 1e-6:
        return 0.8
    if Re <= 275 + 1e-6:
        return 0.85
    if Re <= 355 + 1e-6:
        return 0.9
    return 1.0


def weld_sn_damage(ranges: np.ndarray, counts: np.ndarray, fat: float, shear: bool, curve: str,
                   gamma: float) -> float:
    """Damage from the stress ranges according to the S-N curve of the detail (EN 1993-1-9 or IIW)."""
    D = 0.0
    for dS, n in zip(ranges, counts):
        s = gamma * dS
        if s <= 0:
            continue
        if curve == "IIW":
            if shear:
                N = 2e6 * (fat / s) ** 5
                if N > 1e8:
                    s_k = fat * (2e6 / 1e8) ** 0.2
                    N = 1e8 * (s_k / s) ** 22
            else:
                N = 2e6 * (fat / s) ** 3
                if N > 1e7:
                    s_k = fat * (2e6 / 1e7) ** (1 / 3)
                    N = 1e7 * (s_k / s) ** 22
        elif shear:                                   # EN 1993-1-9: m = 5, fatigue limit at 10⁸
            s_L = 0.457 * fat
            if s < s_L:
                continue
            N = 2e6 * (fat / s) ** 5
        else:                                         # m = 3 up to 5·10⁶, m = 5 up to 10⁸, then the cut-off
            s_D = 0.737 * fat
            s_L = 0.549 * s_D
            if s >= s_D:
                N = 2e6 * (fat / s) ** 3
            elif s >= s_L:
                N = 5e6 * (s_D / s) ** 5
            else:
                continue
        D += n / N
    return D


def toe_hotspot(results, segments: list[dict], idx: list[int]) -> tuple[np.ndarray, list[dict]]:
    """Structural stress at the toe of the weld (IIW hot-spot, type "a") by states: (K, nseg, 2) – σ ⟂ to the toe
    (component u·σ·u in the direction from the weld along the sheet surface) extrapolated from the reference points; NaN = not assessed.
    Stresses at the points = nodal stresses of ccx interpolated by the shape functions of the element of the sheet.
    """
    from ..results import tet10_shape
    nseg = len(segments)
    out = np.full((len(idx), nseg, 2), np.nan)
    meta: list[dict] = [{} for _ in range(2)]
    tets = results.mesh.tets
    for k, sg in enumerate(segments):
        for j, tp in enumerate(sg.get("toe", [])[:2]):
            meta[j].setdefault("part", tp.get("part"))
            if not tp.get("ok") or "elems" not in tp:
                continue
            u = np.asarray(tp["u"])
            N = [tet10_shape(np.asarray(xi)) for xi in tp["xi"]]
            conn = [tets[el] for el in tp["elems"]]
            for a, i in enumerate(idx):
                S = results.steps[i].S
                val = 0.0
                for Nk, ck, wk in zip(N, conn, tp["w"]):
                    s = Nk @ S[ck]                              # xx yy zz xy yz zx
                    T = np.array([[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]])
                    val += wk * float(u @ T @ u)
                out[a, k, j] = val
            for key in ("t", "h", "mesh"):
                meta[j].setdefault(key + "s", []).append(tp.get(key))
    return out, meta


def assess_weld(study, w: WeldSpec, results, idx: list[int], ff=None) -> JointCheck:
    js = study.joints
    info = (results.info.get("joints") or {}).get("welds", {}).get(w.name)
    warns = []
    if info is None:
        return JointCheck("weld", w.name, w.part, math.nan, "no results", warnings=["Weld has no results"])
    a, L, kind = info["a"], info["length"], info["kind"]
    label = f"{'fillet' if kind == 'fillet' else 'butt'} a = {a:.2f} mm, L = {L:.1f} mm"
    mats = []
    for pn in (info.get("part_a"), info.get("part_b")):
        if pn:
            try:
                mats.append(study.material_of(pn))
            except KeyError:
                pass
    if not mats:
        return JointCheck("weld", w.name, label, math.nan, "material missing", warnings=["Joined parts have no material"])
    weak = min(mats, key=lambda m: m.Rm or 1e9)
    fu = weak.Rm or 360.0
    bw = beta_w(weak.Re)
    if getattr(w, "filler", ""):                  # under-strength filler metal → f_u = Rm of the filler (EN 1993-1-8, 4.2)
        fl = tables.filler(w.filler)
        if fl is None:
            warns.append(f"Filler metal '{w.filler}' is not in the table – assumed to match the base metal")
        elif fl[1] < fu or (weak.Re and fl[0] < weak.Re):
            warns.append(f"Filler metal {w.filler} (Re {fl[0]:g}, Rm {fl[1]:g} MPa) is weaker than the base metal – EN 1993-1-8, 4.2 requires matching or overmatching; assessed with f_u = {min(fu, fl[1]):g} MPa")
            fu = min(fu, fl[1])
    if weak.E < 150000:
        warns.append(f"{weak.name}: not steel – EN 1993-1-8 assessment is approximate only")
    if not weak.Rm:
        warns.append("Joined material has no Rm – f_u = 360 MPa used")
    g2 = js.gamma_M2
    f_eq = fu / (bw * g2)
    f_perp = 0.9 * fu / g2
    rows = []
    worst = None
    seg_util = []
    per_case = []
    for i in idx:
        wj = results.steps[i].joints.get("welds", {}).get(w.name)
        if not wj:
            continue
        S = np.asarray(wj["stress"])               # (nseg, 3): σ⊥, τ⊥, τ∥
        eq = np.sqrt(S[:, 0] ** 2 + 3 * (S[:, 1] ** 2 + S[:, 2] ** 2))
        if kind == "fillet":
            u = np.maximum(eq / f_eq, np.abs(S[:, 0]) / f_perp)
        else:
            Re = weak.Re or 0.6 * fu
            u = eq / Re                                # full penetration: as the base material (γ_M0 = 1.0)
        seg_util.append(u)
        k = int(np.argmax(u))
        per_case.append({"case": results.steps[i].name, "stress": S.tolist(), "util": u.tolist(),
                         "imbalance": wj.get("imbalance"),
                         "F_total": (np.asarray(wj["F_a"]) - np.asarray(wj["F_b"])).tolist()})
        if worst is None or u[k] > worst[0]:
            worst = (float(u[k]), results.steps[i].name, k, S[k].tolist(), float(eq[k]))
    if worst is None:
        return JointCheck("weld", w.name, label, math.nan, "no results", warnings=["Weld has no forces from the analysis"])
    u, case, k, (sp, tp, tl), seq = worst
    if kind == "fillet":
        rows.append(("EN 1993-1-8 directional method: √(σ_perp² + 3(τ_perp² + τ_par²)) ≤ f_u/(β_w γ_M2)", seq / f_eq,
                     f"{seq:.1f} / {f_eq:.1f} MPa (case '{case}', segment {k + 1}; σ_perp = {sp:.1f}, τ_perp = {tp:.1f}, τ_par = {tl:.1f} MPa; β_w = {bw:g}, f_u = {fu:g} MPa)"))
        rows.append(("EN 1993-1-8: σ_perp ≤ 0,9 f_u/γ_M2", abs(sp) / f_perp, f"{abs(sp):.1f} / {f_perp:.1f} MPa"))
        if a < 3.0:
            warns.append(f"a = {a:.2f} mm < 3 mm (EN 1993-1-8, 4.5.2)")
        # sheet thicknesses at the toe (by a ray from the surface, evaluate._toe_points): usual limit of a fillet weld and ≤ 0.7·t_min
        ts = [tp.get("t") for sg in (info.get("segments") or []) for tp in (sg.get("toe") or [])
              if tp.get("t") and tp.get("ok", True)]
        if ts:
            t_min = float(min(ts))
            if a > 0.7 * t_min:
                warns.append(f"a = {a:.2f} mm > 0.7·t_min = {0.7 * t_min:.2f} mm (thinnest plate at the weld t = {t_min:.2f} mm) – the weld is oversized relative to the plate, check dimension a")
        if L < max(30.0, 6 * a):
            warns.append(f"Length {L:.1f} mm < max(30 mm; 6a) – the weld should not be counted (EN 1993-1-8, 4.5.1)")
    else:
        Re = weak.Re or 0.6 * fu
        rows.append(("Full-penetration butt weld: σ_eq ≤ f_y (same as base material)", seq / Re,
                     f"{seq:.1f} / {Re:.1f} MPa (case '{case}', segment {k + 1})"))
    imb = max((pc["imbalance"] or 0) for pc in per_case)
    if imb > 1e-3:
        warns.append(f"Force equilibrium on weld {imb:.1e} – verify the result with a finer mesh")
    # --- fatigue (FAT): stress ranges σ_w = projection of (σ⊥, τ⊥) onto the main direction of change, τ_w = τ∥
    fat = None
    if ff is not None:
        F, repeats, closed = ff
        Sk = np.stack([np.asarray(results.steps[i].joints["welds"][w.name]["stress"]) for i in idx
                       if w.name in results.steps[i].joints.get("welds", {})])   # (K, nseg, 3)
        nseg = Sk.shape[1]
        D_seg = np.zeros(nseg)
        Ds, Dt = np.zeros(nseg), np.zeros(nseg)
        dmax_s, dmax_t = np.zeros(nseg), np.zeros(nseg)
        for j in range(nseg):
            hist = F @ Sk[:, j, :]                  # (T, 3)
            ps = _dominant(hist[:, :2])
            cs = _cycles(ps, closed)
            ct = _cycles(hist[:, 2], closed)
            if len(cs):
                Ds[j] = weld_sn_damage(cs[:, 0], cs[:, 2], w.fat_sigma, False, w.sn_curve, w.gamma_Mf) * repeats
                dmax_s[j] = cs[:, 0].max()
            if len(ct):
                Dt[j] = weld_sn_damage(ct[:, 0], ct[:, 2], w.fat_tau, True, w.sn_curve, w.gamma_Mf) * repeats
                dmax_t[j] = ct[:, 0].max()
            D_seg[j] = Ds[j] + Dt[j]
        j = int(np.argmax(D_seg))
        fat = {"damage": float(D_seg[j]), "segment": j, "D_sigma": float(Ds[j]), "D_tau": float(Dt[j]),
               "range_sigma": float(dmax_s[j]), "range_tau": float(dmax_t[j]), "repeats": repeats,
               "damage_segments": D_seg.tolist()}
        rows.append((f"Fatigue {w.sn_curve}: D = D_σ + D_τ ≤ 1 (FAT {w.fat_sigma:g} / {w.fat_tau:g}, γ_Mf = {w.gamma_Mf:g})",
                     float(D_seg[j]),
                     f"segment {j + 1}: Δσ_w = {dmax_s[j]:.1f} MPa (D_σ = {Ds[j]:.3g}), Δτ_w = {dmax_t[j]:.1f} MPa (D_τ = {Dt[j]:.3g}), {repeats:g} blocks"))
    # --- toe of the weld: structural stress (IIW hot-spot) in the sheet at the toe, fatigue FAT_hs (fillet weld only)
    toe = None
    segs = info.get("segments") or []
    if kind == "fillet" and getattr(w, "toe", True) and segs:
        if segs[0].get("toe_error"):
            warns.append(f"Weld root not assessed: {segs[0]['toe_error']}")
        hs, meta = toe_hotspot(results, segs, [i for i in idx if w.name in results.steps[i].joints.get("welds", {})])
        valid = ~np.isnan(hs)
        if valid.any():
            toe = {"hotspot": np.where(valid, hs, 0.0).tolist(), "parts": [m.get("part") for m in meta],
                   "t": [float(np.nanmedian([x for x in m.get("ts", []) if x] or [np.nan])) for m in meta],
                   "mesh": [sorted({x for x in m.get("meshs", []) if x}) for m in meta],
                   "h": [float(np.nanmax([x for x in m.get("hs", []) if x] or [np.nan])) for m in meta]}
            for pc, row in zip(per_case, np.where(valid, hs, np.nan)):
                pc["toe_hotspot"] = [[None if np.isnan(v) else float(v) for v in r] for r in row]
            for j, m in enumerate(meta):
                t_j, h_j = toe["t"][j], toe["h"][j]
                if "coarse" in toe["mesh"][j]:
                    warns.append(f"Toe on '{m.get('part')}': mesh at the toe is coarse (h ≈ {h_j:.2g} mm > 0.6t, t = {t_j:.3g}) – hot-spot from points 0.5t/1.5t (IIW, coarse mesh); for accuracy refine to h ≤ 0.4t")
            if (~valid).any() and valid.any():
                warns.append("Weld toe: in some segments the reference points lie outside the plate (short overlap) – not assessed")
            k_max = np.unravel_index(np.nanargmax(np.abs(np.where(valid, hs, 0.0))), hs.shape)
            toe["max"] = {"case": results.steps[idx[k_max[0]]].name if k_max[0] < len(idx) else "",
                          "segment": int(k_max[1]), "side": int(k_max[2]), "value": float(hs[k_max])}
            if ff is not None:
                F, repeats, closed = ff
                curve = w.sn_curve
                Dmax, best = 0.0, None
                for j in range(hs.shape[2]):
                    t_j = toe["t"][j]
                    n_t = 0.3 if curve == "IIW" else 0.2
                    fat_j = w.fat_toe * ((25.0 / t_j) ** n_t if np.isfinite(t_j) and t_j > 25.0 else 1.0)
                    for k in range(hs.shape[1]):
                        if not valid[:, k, j].all():
                            continue
                        cyc = _cycles(F @ hs[:, k, j], closed)
                        if not len(cyc):
                            continue
                        D = weld_sn_damage(cyc[:, 0], cyc[:, 2], fat_j, False, curve, w.gamma_Mf) * repeats
                        if best is None or D > Dmax:
                            Dmax, best = D, (j, k, float(cyc[:, 0].max()), fat_j)
                if best is not None:
                    j, k, rng, fat_j = best
                    toe["fatigue"] = {"damage": Dmax, "side": j, "segment": k, "range": rng, "fat": fat_j}
                    rows.append((f"Weld toe fatigue {curve} (hot-spot, FAT {fat_j:.3g}, γ_Mf = {w.gamma_Mf:g}): D ≤ 1",
                                 Dmax, f"toe on '{toe['parts'][j]}', segment {k + 1}: Δσ_hs = {rng:.1f} MPa, t = {toe['t'][j]:.3g} mm, {repeats:g} blocks"))
                    if fat is not None:
                        fat["damage_toe"] = Dmax
    elif kind == "butt":
        warns.append("Butt weld toe is not assessed (weld cross-section only)")
    util, gov = _rows_max(rows)
    data = {"a": a, "length": L, "kind": kind, "a_source": info.get("a_source"), "f_u": fu, "beta_w": bw,
            "parts": [info.get("part_a"), info.get("part_b")], "per_case": per_case, "fatigue": fat,
            "segments": info.get("segments"), "toe": toe}
    return JointCheck("weld", w.name, label, util, gov, rows, data, warns)


# ===================================================================== intermittent weld

def _weld_line(info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(start, end, direction) of the root of the weld from the segments of the geometry (x = centre of the segment, e = tangent, s0/s1 = position)."""
    seg = info.get("segments") or []
    if not seg:
        return None
    x0, e0 = np.asarray(seg[0]["x"], float), np.asarray(seg[0]["e"], float)
    x1, e1 = np.asarray(seg[-1]["x"], float), np.asarray(seg[-1]["e"], float)
    a = x0 - e0 * 0.5 * (seg[0]["s1"] - seg[0]["s0"])
    b = x1 + e1 * 0.5 * (seg[-1]["s1"] - seg[-1]["s0"])
    d = b - a
    n = np.linalg.norm(d)
    return (a, b, d / n) if n > 0 else None


def intermittent_groups(study, results) -> list[dict]:
    """Intermittent welds: fillet welds (bodies) between the same pair of parts on one line (parallel, roots
    within max(2a; 3 mm) of the common line). Returns [{"welds": [names by position], "gaps": [mm], "t_min", "a"}].
    """
    winfo = (results.info.get("joints") or {}).get("welds", {})
    cand = []
    for w in study.active_welds():
        info = winfo.get(w.name)
        if not info or info.get("kind") != "fillet" or not info.get("part_b"):
            continue
        line = _weld_line(info)
        if line is None:
            continue
        ts = [tp.get("t") for sg in info.get("segments") or [] for tp in (sg.get("toe") or [])
              if tp.get("t") and tp.get("ok", True)]
        cand.append({"name": w.name, "pair": frozenset((info["part_a"], info["part_b"])), "line": line,
                     "a": float(info["a"]), "t": min(ts) if ts else None})
    groups, used = [], set()
    for i, c in enumerate(cand):
        if c["name"] in used:
            continue
        a0, _, e = c["line"]
        tol = max(0.25 * c["a"], 1.0)                    # segments of the same weld lie on the same line (not the opposite weld)
        members = [c]
        for d in cand[i + 1:]:
            if d["name"] in used or d["pair"] != c["pair"] or abs(float(d["line"][2] @ e)) < 0.995:
                continue
            off = [p - a0 - ((p - a0) @ e) * e for p in d["line"][:2]]
            if max(np.linalg.norm(o) for o in off) <= tol:
                members.append(d)
        if len(members) < 2:
            continue
        span = []
        for m in members:
            sa, sb = sorted(float((p - a0) @ e) for p in m["line"][:2])
            span.append((sa, sb, m))
        span.sort(key=lambda r: r[0])
        gaps = [max(0.0, span[k + 1][0] - span[k][1]) for k in range(len(span) - 1)]
        ts = [m["t"] for m in members if m["t"]]
        groups.append({"welds": [r[2]["name"] for r in span], "gaps": gaps, "t_min": min(ts) if ts else None,
                       "lengths": [r[1] - r[0] for r in span], "a": min(m["a"] for m in members)})
        used.update(m["name"] for m in members)
    return groups


def _intermittent_rows(g: dict) -> tuple[list[tuple[str, float, str]], list[str]]:
    """EN 1993-1-8, 4.3.2.1: gap L2 ≤ min(12t; 200 mm) (compression, shear), L1 ≤ min(16t; 200 mm) (tension).
    The width b (0.25 b, end segments ≥ 0.75 b) is not known from the mesh → a warning.
    """
    rows, warns = [], []
    gap = max(g["gaps"]) if g["gaps"] else 0.0
    t = g["t_min"]
    lim2 = min(12 * t, 200.0) if t else 200.0
    lim1 = min(16 * t, 200.0) if t else 200.0
    txt = (f"{len(g['welds'])} segments ({', '.join(g['welds'])}), largest gap {gap:.1f} mm"
           + (f", t_min = {t:.1f} mm" if t else ", t not determined – 200 mm only"))
    rows.append((INFO + "EN 1993-1-8 4.3.2.1: intermittent weld – gap ≤ min(12t; 200 mm) (compression/shear)",
                 gap / lim2, txt))
    rows.append((INFO + "EN 1993-1-8 4.3.2.1: intermittent weld – gap ≤ min(16t; 200 mm) (tension)", gap / lim1, txt))
    if gap > lim2:
        warns.append(f"Intermittent weld: gap {gap:.1f} mm > min(12t; 200) = {lim2:.1f} mm – not acceptable in compression / shear members (EN 1993-1-8, 4.3.2.1)")
    warns.append("Intermittent weld: also check the gap ≤ 0.25·b and the end segments ≥ 0.75·b (part width b) and do not use it in a corrosive environment (EN 1993-1-8, 4.3.2.1)")
    return rows, warns


# ===================================================================== all

def assess_joints(study, results, parts=None) -> JointsAssessment:
    """Assesses all active bolts and welds. parts (OCP) = edge distances for the bearing."""
    idx = _cases(results)
    out = JointsAssessment(cases=[results.steps[i].name for i in idx])
    why: list[str] = []
    ff = _fatigue_factors(study, results, idx, why)
    out.fatigue = ff is not None
    forces = case_forces(study, results, idx) if study.active_bolts() else []
    for b in study.active_bolts():
        try:
            out.bolts.append(assess_bolt(study, b, results, idx, parts, ff, forces))
        except Exception as e:  # noqa: BLE001 – one joint must not bring down the assessment of the others
            out.bolts.append(JointCheck("bolt", b.name, b.designation, math.nan, f"error: {e}"))
    for w in study.active_welds():
        try:
            out.welds.append(assess_weld(study, w, results, idx, ff))
        except Exception as e:  # noqa: BLE001
            out.welds.append(JointCheck("weld", w.name, w.part, math.nan, f"error: {e}"))
    try:
        groups = intermittent_groups(study, results)
    except Exception as e:  # noqa: BLE001 – a supplement, the assessment of the welds holds without it
        groups = []
        out.warnings.append(f"Intermittent welds could not be recognised ({e}) – the EN 1993-1-8 4.3.2.1 gap check is missing")
    by_name = {c.name: c for c in out.welds}
    for g in groups:
        rows, warns = _intermittent_rows(g)
        for name in g["welds"]:
            c = by_name.get(name)
            if c is not None:
                c.rows += rows
                c.warnings += [w for w in warns if w not in c.warnings]
                c.data["intermittent"] = g
    if not out.fatigue and study.fatigue is not None and study.fatigue.history.channels:
        out.warnings.append("Load history does not match the load cases – joint fatigue not assessed"
                            + (f": {why[0]}" if why else ""))
    return out


def report_text(a: JointsAssessment) -> str:
    """Text summary (CLI, log)."""
    lines = []
    for c in a.all():
        st = "PASS" if c.ok else ("FAIL" if math.isfinite(c.utilization) else "NOT ASSESSED")
        lines.append(f"{c.name} ({c.label}): utilization {c.utilization:.2f} – {st}; governing: {c.governing}")
        for name, u, txt in c.rows:
            lines.append(f"    {u:6.2f}  {name}: {txt}")
        for w in c.warnings:
            lines.append(f"    ! {w}")
    for w in a.warnings:
        lines.append(f"! {w}")
    return "\n".join(lines)
