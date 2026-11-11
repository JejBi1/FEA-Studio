"""Verification cases F3 – bolts and welds (balance of forces, beam stiffness, standard formulas).

S1  Two plates 50×50×10 with a gap of 1 mm joined only by an M10 bolt (without preload): tension and shear
    → the force in the bolt = the load (exactly), elongation of the beam = F·(δ_S + δ_washers) from VDI.
S2  Lap joint M10 8.8 (gap joined), shear 10 kN, preload in the FEM state:
    → transfer by the gap + the bolt = 10 kN, the preload F_V exactly, EN 1993-1-8 and VDI against a hand calculation.
S3  Two bolts M12 in a row, shear 20 kN → group method 10 kN per bolt, bearing of the inner bolt (p1).
W1  T-joint, two fillet welds a = 5, L = 100 (contact web–flange separated):
    → the sum of the forces of the segments = the load (tension 20 kN, longitudinal shear 10 kN), τ∥ = F/(2aL) = 10 MPa, and a from the mesh,
    fatigue (sine × tension) = Miner from the stress range of the worst segment (FAT 36 / 80).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from fea_core.geometry import Part, read_step
from fea_core.joints import tables
from fea_core.joints.detect import detect
from fea_core.results import Results
from fea_core.selectors import find_faces
from fea_core.study import Study, Fixed, Force, Gravity, RemoteLoad, LoadCase, MeshSettings
from verification import joint_models as jm
from verification.cases import Case, Check, _study, box, cyl, solid


# ------------------------------------------------------------------ S1
def s1_build(wd: Path):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    d, t, gap = 10.0, 10.0, 1.0
    dh = tables.CLEARANCE[10][1]
    A = solid(BRepAlgoAPI_Cut(box(-25, -25, 0, 50, 50, t), cyl((0, 0, -1), (0, 0, 1), dh / 2, t + 2)).Shape())
    B = solid(BRepAlgoAPI_Cut(box(-25, -25, -t - gap, 50, 50, t), cyl((0, 0, -t - gap - 1), (0, 0, 1), dh / 2, t + 2)).Shape())
    w = tables.washer_dims(d)
    nut = tables.nut_dims(d)
    L = 40.0
    geo = [Part("Deska A", A), Part("Deska B", B),
           Part("ISO 4017 - M10 x 40 - 8.8", jm.hex_bolt(0, 0, t + w["h"], d, L)),
           Part("ISO 7089 - 10 - 200 HV", jm.washer(0, 0, t, d, up=True)),
           Part("ISO 7089 - 10 - 200 HV", jm.washer(0, 0, -t - gap, d, up=False)),
           Part("ISO 4032 - M10 - 8", jm.hex_nut(0, 0, -t - gap - w["h"], d))]
    st, parts = _study("S1 sroub se sparou", wd / "model.step", geo, MeshSettings(size=4.0))
    st.bolts = detect(parts).bolts
    st.joints.preload_case = False
    st.add_selection("bokyB", find_faces(parts, part="Deska B", x=-25) + find_faces(parts, part="Deska B", x=25))
    st.add_selection("bokyA", find_faces(parts, part="Deska A", x=-25) + find_faces(parts, part="Deska A", x=25))
    st.constraints.append(Fixed("bokyB"))
    st.load_cases += [LoadCase("Tah", [Force("bokyA", [0, 0, 10000.0])]),
                      LoadCase("Smyk", [Force("bokyA", [10000.0, 0, 0])])]
    _ = nut
    return st, parts


def _rigid_mean(res: Results, key: str, step: int) -> np.ndarray:
    idx = res.mesh.surface_nodes(key)
    return res.step(step).U[idx].mean(axis=0)


def s1_eval(res: Results, st: Study):
    from fea_core.joints.assess import bolt_case_forces
    from fea_core.joints.fem import bolt_basics
    b = st.bolts[0]
    e = b.e
    tah = bolt_case_forces(b, res.step("Tah").joints["bolts"][b.name])
    smyk = bolt_case_forces(b, res.step("Smyk").joints["bolts"][b.name])
    bas = bolt_basics(st, b)
    dW = sum(w["h"] / (bas["E_S"] * math.pi / 4 * (w["d2"] ** 2 - w["d1"] ** 2)) for w in b.washers)
    # elongation of the beam = difference of the displacements of the rigid bearing faces (head, nut) along the axis
    du = (_rigid_mean(res, "__b1_n", 0) - _rigid_mean(res, "__b1_h", 0)) @ e
    # tension acts against e (separation of plate A from B) → F_SA > 0, elongation > 0
    return [
        Check("tah: síla ve šroubu F_SA = zatížení", tah["F_SA"], 10000.0, 1e-4, "N"),
        Check("tah: F_A přenesená rovinou spáry", tah["planes"][0]["F_A"], 10000.0, 1e-4, "N"),
        Check("smyk: příčná síla ve šroubu = zatížení", smyk["planes"][0]["Q_bolt"], 10000.0, 1e-4, "N"),
        Check("protažení nosníku = F·(δ_S + δ_podložky)", du, 10000.0 * (bas["delta_S"] + dW), 1e-3, "mm"),
    ]


# ------------------------------------------------------------------ S2
def s2_build(wd: Path):
    st, parts = _study("S2 preplatovany spoj", wd / "model.step", jm.lap_joint(), MeshSettings(size=5.0))
    st.bolts = detect(parts).bolts
    st.add_selection("vetknuti", find_faces(parts, part="Deska B", x=160))
    st.add_selection("konecA", find_faces(parts, part="Deska A", x=0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Smyk 10 kN", [Force("konecA", [-10000.0, 0, 0])]))
    st.load_cases.append(LoadCase("Tíha", [Gravity([0.0, 0.0, -9810.0])]))   # U1 cannot carry GRAV → nodal forces
    return st, parts


def s2_eval(res: Results, st: Study):
    from fea_core.joints.assess import assess_joints, bolt_case_forces
    from fea_core.joints.fem import PRELOAD_CASE, bolt_basics
    b = st.bolts[0]
    c = bolt_case_forces(b, res.step("Smyk 10 kN").joints["bolts"][b.name])
    pre = bolt_case_forces(b, res.step(PRELOAD_CASE).joints["bolts"][b.name])
    bas = bolt_basics(st, b)
    parts = read_step(st.geometry)
    a = assess_joints(st, res, parts).bolts[0]
    rows = {r[0]: r for r in a.rows}
    th = tables.thread(10)
    # hand calculation
    k = 1.5 / (math.pi * th.d2) + 1.155 * 0.12
    FMzul = th.As * 0.9 * 640 / math.sqrt(1 + 3 * (1.5 * th.d2 / th.ds * k) ** 2)
    Fv_Rd = 0.6 * 800 * th.As / 1.25
    Fb_Rd = 2.5 * (20 / (3 * 11.0)) * 360 * 10 * 10 / 1.25
    shear = next(r for n, r in rows.items() if n.startswith("EN 1993-1-8: shear F_v"))
    bear = next(r for n, r in rows.items() if "bearing 'Deska A'" in n)
    Fv = float(shear[2].split("/")[1].split("N")[0])
    Fb = float(bear[2].split("/")[1].split("N")[0])
    from fea_core.loads import part_volumes
    weight = 9810.0 * sum(v * st.material_of(p).rho for p, v in zip(res.mesh.part_names, part_volumes(res.mesh)))
    return [
        Check("tíha (uzlové síly, model s U1): reakce Fz = m·g", res.step("Tíha").reactions["vetknuti"][2],
              weight, 1e-4, "N"),                     # RF in the .frd has 6 valid digits
        Check("smyk přenesený rovinou spáry (spára + šroub)", c["planes"][0]["F_Q"], 10000.0, 1e-4, "N"),
        Check("předpětí: síla ve šroubu = F_V (VDI)", pre["F_SA"], bas["F_Vnom"], 1e-4, "N"),
        Check("předpětí: tlak ve spáře = F_V", -pre["planes"][0]["F_PA"], bas["F_Vnom"], 1e-3, "N"),
        Check("VDI F_M,zul (M10 8.8, μ_G = 0,12)", bas["F_Mzul"], FMzul, 1e-9, "N"),
        Check("EN 1993-1-8 F_v,Rd (závit, α_v = 0,6)", Fv, Fv_Rd, 1e-3, "N"),
        Check("EN 1993-1-8 F_b,Rd (e1 = e2 = 20 z geometrie)", Fb, Fb_Rd, 1e-3, "N"),
    ]


# ------------------------------------------------------------------ S4
def s4_build(wd: Path):
    """S2 without the fastening parts in the model: the bolt, washers and nut only as records of the CAD assembly (parts not loaded,
    Content Center) → a virtual bolt `joints.virtual`; the results and assessments must be the same as for S2.
    """
    from fea_core.joints.naming import parse_name
    from fea_core.joints.virtual import bolts_from_fasteners, fastener_record
    full = jm.lap_joint()
    st, parts = _study("S4 virtualni sroub", wd / "model.step", full[:2], MeshSettings(size=5.0))
    wh = tables.washer_dims(10.0)["h"]

    def rec(name, z):
        m = [1, 0, 0, 8.0, 0, 1, 0, 2.0, 0, 0, 1, z / 10.0, 0, 0, 0, 1]
        return fastener_record(parse_name(name), name + ":1", name + ".ipt", m)
    recs = [rec(full[2].name, 10.0 + wh), rec(full[3].name, 10.0), rec(full[4].name, -10.0),
            rec(full[5].name, -10.0 - wh)]
    st.bolts, warn = bolts_from_fasteners(parts, recs)
    if len(st.bolts) != 1 or warn:
        raise RuntimeError(f"virtuální šroub: {len(st.bolts)} šroubů, upozornění {warn}")
    st.add_selection("vetknuti", find_faces(parts, part="Deska B", x=160))
    st.add_selection("konecA", find_faces(parts, part="Deska A", x=0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Smyk 10 kN", [Force("konecA", [-10000.0, 0, 0])]))
    st.load_cases.append(LoadCase("Tíha", [Gravity([0.0, 0.0, -9810.0])]))
    return st, parts


# ------------------------------------------------------------------ S3
def s3_build(wd: Path):
    st, parts = _study("S3 dva srouby", wd / "model.step", jm.lap_joint(d=12.0, nbolts=2, pitch=40.0),
                       MeshSettings(size=5.0))
    st.bolts = detect(parts).bolts
    st.joints.preload_case = False
    st.add_selection("vetknuti", find_faces(parts, part="Deska B", x=200))
    st.add_selection("konecA", find_faces(parts, part="Deska A", x=0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Smyk 20 kN", [Force("konecA", [-20000.0, 0, 0])]))
    return st, parts


def s3_eval(res: Results, st: Study):
    from fea_core.joints.assess import assess_joints, case_forces
    forces = case_forces(st, res, [0])[0][1]
    q = [forces[b.name]["planes"][0]["F_Q"] for b in st.bolts]
    q_fe = sum(np.asarray(forces[b.name]["planes"][0]["Q_fe"]) for b in st.bolts)
    a = assess_joints(st, res, read_step(st.geometry))
    row = next(r for r in a.bolts[0].rows if "bearing 'Deska A'" in r[0])
    Fb = float(row[2].split("/")[1].split("N")[0])
    d0, fu = 13.5, 360.0
    alpha = min(60 / (3 * d0), 40 / (3 * d0) - 0.25, 800 / fu, 1.0)        # inner bolt: pitch p1 = 40
    k1 = min(2.8 * 20 / d0 - 1.7, 2.5)
    return [
        Check("spára + šrouby přenesou celkem (MKP)", float(np.linalg.norm(q_fe)), 20000.0, 1e-4, "N"),
        Check("skupina: šroub 1 = F/n", q[0], 10000.0, 1e-4, "N"),
        Check("skupina: šroub 2 = F/n", q[1], 10000.0, 1e-4, "N"),
        Check("F_b,Rd deska A, šroub 1 (e1 = 60, p1 = 40, e2 = 20)", Fb, k1 * alpha * fu * 12 * 10 / 1.25, 1e-3, "N"),
    ]


# ------------------------------------------------------------------ W1
def w1_build(wd: Path):
    st, parts = _study("W1 T-spoj koutove svary", wd / "model.step", jm.t_joint_fillet(), MeshSettings(size=5.0))
    st.welds = detect(parts).welds
    st.add_selection("pata", find_faces(parts, part="Pasnice", z=-10))
    st.add_selection("hlava", find_faces(parts, part="Stojina", z=100))
    st.constraints.append(Fixed("pata"))
    st.load_cases += [LoadCase("Tah 20 kN", [Force("hlava", [0, 0, 20000.0])]),
                      LoadCase("Podélný smyk 10 kN", [RemoteLoad("hlava", [0, 50, 0], force=[0, 10000.0, 0])])]
    from fea_core.fatigue.history import Channel, LoadHistory
    from fea_core.fatigue.settings import FatigueSettings
    st.fatigue = FatigueSettings(LoadHistory([Channel("Tah 20 kN", "sine", amplitude=1.0, cycles=1000)]),
                                 repeats=1000.0)
    return st, parts


def w1_eval(res: Results, st: Study):
    from fea_core.joints.assess import assess_joints, weld_sn_damage
    tah = res.step("Tah 20 kN").joints["welds"]
    smyk = res.step("Podélný smyk 10 kN").joints["welds"]
    Tz = sum(np.asarray(w["T"]).sum(axis=0)[2] for w in tah.values())
    Ty = sum(np.asarray(w["T"]).sum(axis=0)[1] for w in smyk.values())
    tau = np.mean([np.asarray(w["stress"])[:, 2].mean() for w in smyk.values()])
    info = res.info["joints"]["welds"]
    # fatigue: sine ±1 × the state "Tension", 1000 cycles per block, 1000 blocks → Δσ_w = 2|(σ⊥, τ⊥)|, Δτ_w = 2|τ∥|
    ws = st.welds[0]
    fat = next(c for c in assess_joints(st, res).welds if c.name == ws.name).data["fatigue"]
    S = np.asarray(tah[ws.name]["stress"])[fat["segment"]]
    D_ref = 1e6 * (weld_sn_damage(np.array([2 * np.hypot(S[0], S[1])]), np.array([1.0]), ws.fat_sigma, False,
                                  ws.sn_curve, ws.gamma_Mf)
                   + weld_sn_damage(np.array([2 * abs(S[2])]), np.array([1.0]), ws.fat_tau, True,
                                    ws.sn_curve, ws.gamma_Mf))
    return [
        Check("únava svaru: D nejhoršího úseku = Miner z rozkmitu (FAT 36/80)", fat["damage"], D_ref, 1e-6, ""),
        Check("tah: součet sil úseků obou svarů", Tz, 20000.0, 1e-4, "N"),
        Check("podélný smyk: součet sil úseků", Ty, 10000.0, 1e-4, "N"),
        Check("podélný smyk: průměrné τ∥ = F/(2aL)", abs(float(tau)), 10.0, 1e-3, "MPa"),
        Check("účinná výška a ze sítě", info["Weld 1"]["a"], 5.0, 1e-4, "mm"),
        Check("délka svaru ze sítě", info["Weld 1"]["length"], 100.0, 1e-4, "mm"),
    ]


# ------------------------------------------------------------------ W2
W2_F, W2_L, W2_B, W2_T = 1000.0, 200.0, 40.0, 10.0


def w2_build(wd: Path):
    from fea_core.study import Material
    st, parts = _study("W2 pata svaru hot-spot", wd / "model.step", jm.plate_attachment(), MeshSettings(size=3.0))
    st.welds = detect(parts).welds
    # compliant welds (E = 1 % of steel): the attachment plate does not stiffen → the surface stress of the plate = beam theory
    soft = Material("Svar poddajný (ověření)", 2100.0, 0.3, 7.85e-9, Re=235.0, Rm=360.0)
    st.materials[soft.name] = soft
    for w in st.welds:
        st.part_materials[w.part] = soft.name
    st.add_selection("vetknuti", find_faces(parts, part="Plech", x=0))
    st.add_selection("konec", find_faces(parts, part="Plech", x=W2_L))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Ohyb", [Force("konec", [0, 0, -W2_F])]))
    from fea_core.fatigue.history import Channel, LoadHistory
    from fea_core.fatigue.settings import FatigueSettings
    st.fatigue = FatigueSettings(LoadHistory([Channel("Ohyb", "sine", amplitude=1.0, cycles=1000)]), repeats=1000.0)
    return st, parts


def w2_eval(res: Results, st: Study):
    """A cantilever with the weight F at the end: the surface of the sheet σ_x(x) = 6F(L − x)/(b t²) is linear in x → the IIW extrapolation
    (1.67 σ(0.4t) − 0.67 σ(1.0t)) gives exactly the stress at the toe x_toe = x_stub ± (t_stub/2 + a√2).
    """
    from fea_core.joints.assess import assess_joints, weld_sn_damage
    a = assess_joints(st, res)
    checks = []
    zleg = 4.0 * math.sqrt(2)
    for c in a.welds:
        toe = c.data["toe"]
        j = toe["parts"].index("Plech")
        hs = np.asarray(toe["hotspot"])[0, :, j]
        right = "vpravo" in st.welds[[w.name for w in st.welds].index(c.name)].part
        x_toe = 100.0 + (5.0 + zleg) * (1 if right else -1)
        ref = 6 * W2_F * (W2_L - x_toe) / (W2_B * W2_T ** 2)
        checks.append(Check(f"hot-spot v patě {'vpravo' if right else 'vlevo'} (x = {x_toe:.2f}) = σ nosníku",
                            float(hs.mean()), ref, 1e-2, "MPa"))
        checks.append(Check(f"tloušťka plechu u paty ze sítě ({'vpravo' if right else 'vlevo'})", toe["t"][j],
                            W2_T, 1e-5, "mm"))
        if right:
            f = toe["fatigue"]
            hs_k = float(np.asarray(toe["hotspot"])[0, f["segment"], f["side"]])
            w = st.welds[0]
            D_ref = 1e6 * weld_sn_damage(np.array([2 * abs(hs_k)]), np.array([1.0]), w.fat_toe, False, w.sn_curve,
                                         w.gamma_Mf)
            checks.append(Check("únava paty: D = Miner z rozkmitu Δσ_hs (FAT 90)", f["damage"], D_ref, 1e-6, ""))
    return checks


JOINT_CASES: dict[str, Case] = {
    "W2": Case("W2", "Pata koutového svaru: hot-spot IIW (0,4t/1,0t) na ohýbaném plechu", w2_build, w2_eval),
    "S1": Case("S1", "Šroub M10 se spárou 1 mm (jen šroub): tah a smyk, tuhost nosníku", s1_build, s1_eval),
    "S2": Case("S2", "Přeplátovaný spoj M10 8.8: smyk, předpětí, VDI a EN 1993-1-8", s2_build, s2_eval),
    "S4": Case("S4", "Virtuální šroub ze sestavy CAD (součásti nenačtené): jako S2", s4_build, s2_eval),
    "S3": Case("S3", "Dva šrouby M12: rozdělení na skupinu šroubů, otlačení s roztečí", s3_build, s3_eval),
    "W1": Case("W1", "T-spoj, koutové svary a5: tah a podélný smyk", w1_build, w1_eval),
}
