"""Verification cases of 3D printing (F4): homogenised model, orthotropy by the build direction, failure criteria.

P1  two bars 100×10×10 of full extrusion (PLA Bambu), tension 10 MPa along x; bar A printed standing
    (build = x → tension perpendicular to the layers), bar B lying (build = z). Analytically: u = σL/E_z, resp. σL/E_xy,
    transverse contraction ν13·σ/E (check of the convention of ν), Tsai-Hill λ = Zt/σ, resp. Xt/σ.
P2  block 20×20×40 (build z), 4 perimeters × 0.5 mm, 5/3 layers × 0.2 mm, grid 20 %, tension 1 MPa along z:
    shell and core in parallel (mixing rule in the section), top/bottom layers in series;
    + self-weight: reaction = ρ·g·(V_shell + 0.2·V_core). Check of the shell share from the mesh.
P3  anisotropy of the line of the perimeters (build z, 2 × 0.42 mm, without full layers): wall 30×1.6×6 (entirely of perimeters,
    line along x) and plate 30×4×6 (perimeters + grid 40 %). Tension 5 MPa along x (along the line), along y
    (across the line, only the wall) and along z; the strain measured by probes in the middle part: ε = σ/E_L, σ/E_T, σ/E_z,
    plate = laminate (shell 1.68 ∥ core 2.32, common ε_x, ε_z); Tsai-Hill λ = X_L/σ, X_T/σ.
    (Tension of the plate in Z with pressure on the face without full layers cannot be checked by hand – the pressure on the soft core is transferred by shear.)
P4  creep: bar 100×10×10 (solid, lying), tension 5 MPa as a long-term state 1000 h / 30 °C:
    displacement × (1 + k_φ·φ)/k_E (all parts printed → exact), safety × k_R.
P5  fatigue of the printing on the bars P1: alternating tension ±8 MPa (1 cycle per block, without a mean-stress correction): lying (tension XY)
    D = (σa/SD)^k/ND, standing (tension Z, Hill equivalent) σ_eq = σa·Rm/Rm_z; safety = SD·ND^(1/k)/σ_eq.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fea_core import materials as matlib
from fea_core.geometry import Part
from fea_core.printing import PrintSpec
from fea_core.printing.infill import ratios
from fea_core.results import Results
from fea_core.selectors import find_faces
from fea_core.study import Study, Displacement, Pressure, Gravity, LoadCase, MeshSettings
from verification.cases import Case, Check, _study, box

PLA = matlib.by_name("Tisk PLA Bambu Basic")
SIG = 10.0


def _rollers(st, parts):
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("y0", find_faces(parts, y=0) + find_faces(parts, y=20))
    st.add_selection("z0", find_faces(parts, z=0))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0), Displacement("z0", uz=0.0)]


# ------------------------------------------------------------------ P1
def p1_build(wd: Path):
    geo = [Part("Tyc nastojato", box(0, 0, 0, 100, 10, 10)), Part("Tyc naležato", box(0, 20, 0, 100, 10, 10))]
    st, parts = _study("P1 ortotropie tisku", wd / "model.step", geo, MeshSettings(size=3.0))
    st.add_material(PLA, default=True)
    st.materials.pop("Ocel S235")
    st.printing.criterion = "tsai_hill"
    st.printing.parts["Tyc nastojato"] = PrintSpec("Tyc nastojato", build_dir=[1, 0, 0], enabled=False)
    st.printing.parts["Tyc naležato"] = PrintSpec("Tyc naležato", build_dir=[0, 0, 1], enabled=False)
    _rollers(st, parts)
    st.add_selection("cela", find_faces(parts, x=100))
    st.load_cases.append(LoadCase("Tah 10 MPa", [Pressure("cela", -SIG)]))
    return st, parts


def p1_eval(res: Results, st: Study):
    s = res.step(0)
    x = res.mesh.nodes
    nu13 = PLA.nu_z if PLA.nu_z is not None else PLA.nu
    a = (np.abs(x[:, 0] - 100) < 1e-6) & (x[:, 1] < 15)
    b = (np.abs(x[:, 0] - 100) < 1e-6) & (x[:, 1] > 15)
    ya = np.abs(x[:, 1] - 10) < 1e-6                   # side of the bar A (y = 10)
    zb = (np.abs(x[:, 2] - 10) < 1e-6) & (x[:, 1] > 15)   # top face of the bar B
    pa, pb = s.printing["parts"]["Tyc nastojato"], s.printing["parts"]["Tyc naležato"]
    return [
        Check("A (tah ⟂ vrstvám): u = σL/E_z", float(s.U[a, 0].mean()), SIG * 100 / PLA.E_z, 0.001, "mm"),
        Check("B (tah v rovině vrstvy): u = σL/E_xy", float(s.U[b, 0].mean()), SIG * 100 / PLA.E, 0.001, "mm"),
        Check("A příčně: uy = −ν13·σ/E_xy·10", float(-s.U[ya, 1].mean()), nu13 * SIG / PLA.E * 10, 0.01, "mm"),
        Check("B svisle: uz = −ν13·σ/E_xy·10", float(-s.U[zb, 2].mean()), nu13 * SIG / PLA.E * 10, 0.01, "mm"),
        Check("σx (globálně) v tyči A", res.probe([50, 5, 5], "S")[0], SIG, 0.001, "MPa"),
        Check("Tsai-Hill A: λ = Rm_z/σ", pa["min_safety"], PLA.Rm_z / SIG, 0.002, ""),
        Check("Tsai-Hill B: λ = Rm/σ", pb["min_safety"], PLA.Rm / SIG, 0.002, ""),
        Check("reakce Fx (obě tyče)", -s.reactions["x0"][0], 2 * SIG * 100, 0.001, "N"),
    ]


# ------------------------------------------------------------------ P2
P2 = PrintSpec("Hranol", build_dir=[0, 0, 1], layer_height=0.2, line_width=0.5, walls=4, top_layers=5,
               bottom_layers=3, infill=0.2, pattern="grid")
P2_B, P2_H, P2_DU = 10.0, 20.0, 0.01


def p2_build(wd: Path):
    st, parts = _study("P2 skorepina a vypln", wd / "model.step", [Part("Hranol", box(0, 0, 0, P2_B, P2_B, P2_H))],
                       MeshSettings(size=0.6))
    st.add_material(PLA, default=True)
    st.materials.pop("Ocel S235")
    st.printing.parts["Hranol"] = P2
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("y0", find_faces(parts, y=0))
    st.add_selection("z0", find_faces(parts, z=0))
    st.add_selection("vrch", find_faces(parts, z=P2_H))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0), Displacement("z0", uz=0.0)]
    # tension by rigid jaws (prescribed displacement of the top) – isostrain in the section as in the mixing rule;
    # under a uniform pressure the 1 mm plate would transfer the load from the soft core into the walls (bending, +3.6 % of the displacement)
    st.constraints.append(Displacement("vrch", uz=P2_DU))
    st.load_cases.append(LoadCase("Tah čelistmi", []))
    st.load_cases.append(LoadCase("Tíha", [Gravity([0, 0, -9810.0])]))
    return st, parts


def p2_reference() -> dict:
    w = P2.wall_thickness
    A = P2_B ** 2
    Ac = (P2_B - 2 * w) ** 2
    As = A - Ac
    t_end = (P2.top_layers + P2.bottom_layers) * P2.layer_height
    L_mid = P2_H - t_end
    r = ratios(P2.pattern, P2.infill)
    E_mid = PLA.E_z * (As + Ac * r["E_z"]) / A
    F = P2_DU * A / (t_end / PLA.E_z + L_mid / E_mid)
    V_s = As * L_mid + A * t_end
    V_c = Ac * L_mid
    return {"F": F, "shell_fraction": V_s / (V_s + V_c),
            "weight": PLA.rho * 9810.0 * (V_s + P2.infill * V_c)}


def p2_eval(res: Results, st: Study):
    ref = p2_reference()
    s, g = res.step(0), res.step(1)
    summ = res.info["printing"]["parts"]["Hranol"]
    return [
        Check("síla v čelistech (skořepina ∥ jádro, vrstvy sériově)", s.reactions["vrch"][2], ref["F"], 0.03, "N"),
        Check("objemový podíl skořepiny ze sítě", summ["shell_fraction"], ref["shell_fraction"], 0.03, ""),
        Check("vlastní tíha: ΣFz reakcí − tah = ρ·g·(V_s + ρ_v·V_j)",
              g.reactions["z0"][2] + g.reactions["vrch"][2], ref["weight"], 0.03, "N"),
    ]


PRINT_CASES: dict[str, Case] = {
    "P1": Case("P1", "Tisk: tyče nastojato / naležato, ortotropie a Tsai-Hill", p1_build, p1_eval),
    "P2": Case("P2", "Tisk: hranol se skořepinou a výplní 20 %, tah + vlastní tíha", p2_build, p2_eval),
}


# ------------------------------------------------------------------ P3
P3_SPEC = dict(build_dir=[0, 0, 1], layer_height=0.2, line_width=0.42, walls=2, top_layers=0, bottom_layers=0,
               infill=0.4, pattern="grid")
P3_L, P3_H, P3_TA, P3_TB, P3_YB, P3_S = 30.0, 6.0, 1.6, 4.0, 10.0, 5.0


def p3_build(wd: Path):
    geo = [Part("Stena", box(0, 0, 0, P3_L, P3_TA, P3_H)), Part("Deska", box(0, P3_YB, 0, P3_L, P3_TB, P3_H))]
    st, parts = _study("P3 anizotropie linie", wd / "model.step", geo, MeshSettings(size=0.6))
    st.add_material(PLA, default=True)
    st.materials.pop("Ocel S235")
    st.printing.criterion = "tsai_hill"
    for p in ("Stena", "Deska"):
        st.printing.parts[p] = PrintSpec(p, **P3_SPEC)
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("y0", find_faces(parts, y=0) + find_faces(parts, y=P3_YB))
    st.add_selection("z0", find_faces(parts, z=0))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0), Displacement("z0", uz=0.0)]
    st.add_selection("xL", find_faces(parts, x=P3_L))
    st.add_selection("yA", find_faces(parts, y=P3_TA))
    st.add_selection("zH", find_faces(parts, z=P3_H))
    st.load_cases += [LoadCase("Tah podél linie (x)", [Pressure("xL", -P3_S)]),
                      LoadCase("Tah napříč linie (y)", [Pressure("yA", -P3_S)]),
                      LoadCase("Tah Z", [Pressure("zH", -P3_S)])]
    return st, parts


def p3_reference() -> dict:
    from fea_core.printing.infill import line_solid, core
    from fea_core.printing.props import default_props
    pr = default_props(PLA.name)
    ml = line_solid(PLA, pr.line_E_ratio, pr.line_R_ratio)
    mc = core(PLA, P3_SPEC["pattern"], P3_SPEC["infill"])
    t = P3_SPEC["walls"] * P3_SPEC["line_width"] * 2
    # plate = laminate in the direction y (shell t, core P3_TB − t): ε_x, ε_z common, σ_y = 0 in each layer,
    # Σ t·σ_x = σ·P3_TB, Σ t·σ_z = 0 (the different transverse contraction of the shell and the core is captured)
    A = np.zeros((2, 2))
    for C, tt in ((ml.stiffness(), t), (mc.stiffness(), P3_TB - t)):
        a, b = [0, 2], [1]
        Q = C[np.ix_(a, a)] - C[np.ix_(a, b)] @ np.linalg.inv(C[np.ix_(b, b)]) @ C[np.ix_(b, a)]
        A += tt * Q
    ex = np.linalg.solve(A, [P3_S * P3_TB, 0.0])[0]
    L, T = P3_L, P3_TB
    return {"E_L": ml.E1, "E_T": ml.E2v, "X_L": ml.Xt, "X_T": ml.Ytv, "eps_B": ex,
            "eps_B_mix": P3_S * P3_TB / (t * ml.E1 + (P3_TB - t) * mc.E1),
            "shell_B": 1 - (L - t) * (T - t) / (L * T)}


def _strain(res, step, a, b, comp):
    ua, ub = res.probe(a, "U", step)[comp], res.probe(b, "U", step)[comp]
    return (ub - ua) / np.linalg.norm(np.subtract(b, a))


def _mid_safety(res, step, y0, y1):
    x = res.mesh.nodes
    sel = (np.abs(x[:, 0] - P3_L / 2) < 6) & (x[:, 1] >= y0 - 1e-6) & (x[:, 1] <= y1 + 1e-6)         & (np.abs(x[:, 2] - P3_H / 2) < 2)
    return float(np.nanmin(res.step(step).fields["print_safety"][sel]))


def p3_eval(res: Results, st: Study):
    r = p3_reference()
    ym, zm, yb = P3_TA / 2, P3_H / 2, P3_YB + P3_TB / 2
    return [
        Check("stěna podél linie: ε_x = σ/E_L", _strain(res, 0, [10, ym, zm], [20, ym, zm], 0), P3_S / r["E_L"], 0.005, ""),
        Check("stěna napříč linie: ε_y = σ/E_T", _strain(res, 1, [15, 0, zm], [15, P3_TA, zm], 1), P3_S / r["E_T"], 0.005, ""),
        Check("stěna Z: ε_z = σ/E_z", _strain(res, 2, [15, ym, 1.5], [15, ym, 4.5], 2), P3_S / PLA.E_z, 0.005, ""),
        Check("deska podél linie: ε_x (laminát skořepina + jádro)", _strain(res, 0, [12, yb, zm], [18, yb, zm], 0),
              r["eps_B"], 0.025, ""),   # mesh 0.6: 1.8 %, mesh 0.4: 1.2 % (transition shell–core in the elements)
        Check("  (jen směšovací pravidlo E = Σt·E/t, bez příčné vazby)", _strain(res, 0, [12, yb, zm], [18, yb, zm], 0),
              r["eps_B_mix"], 0.05, ""),
        Check("deska: objemový podíl skořepiny ze sítě", res.info["printing"]["parts"]["Deska"]["shell_fraction"],
              r["shell_B"], 0.03, ""),
        Check("Tsai-Hill stěna podél linie: λ = X_L/σ", _mid_safety(res, 0, 0, P3_TA), r["X_L"] / P3_S, 0.01, ""),
        Check("Tsai-Hill stěna napříč linie: λ = X_T/σ", _mid_safety(res, 1, 0, P3_TA), r["X_T"] / P3_S, 0.01, ""),
    ]


# ------------------------------------------------------------------ P4
P4_HOURS, P4_T, P4_S = 1000.0, 30.0, 5.0


def p4_build(wd: Path):
    st, parts = _study("P4 tečení", wd / "model.step", [Part("Tyc", box(0, 0, 0, 100, 10, 10))], MeshSettings(size=3.0))
    st.add_material(PLA, default=True)
    st.materials.pop("Ocel S235")
    st.printing.criterion = "tsai_hill"
    st.printing.parts["Tyc"] = PrintSpec("Tyc", build_dir=[0, 0, 1], enabled=False)
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("y0", find_faces(parts, y=0))
    st.add_selection("z0", find_faces(parts, z=0))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0), Displacement("z0", uz=0.0)]
    st.add_selection("cela", find_faces(parts, x=100))
    st.load_cases.append(LoadCase("Trvalý tah", [Pressure("cela", -P4_S)]))
    st.printing.creep.enabled = True
    st.printing.creep.hours = P4_HOURS
    st.printing.creep.temperature = P4_T
    st.printing.creep.cases = ["Trvalý tah"]
    return st, parts


def p4_eval(res: Results, st: Study):
    from fea_core.printing.props import default_props
    fac = default_props(PLA.name).creep_factors(P4_HOURS, P4_T)
    s0, s1 = res.step(0), res.step(1)
    x = res.mesh.nodes
    end = np.abs(x[:, 0] - 100) < 1e-6
    u0, u1 = float(s0.U[end, 0].mean()), float(s1.U[end, 0].mean())
    l0 = s0.printing["parts"]["Tyc"]["min_safety"]
    l1 = s1.printing["parts"]["Tyc"]["min_safety"]
    # ATTENTION: the creep factors k_E, k_R come from the SAME table (`default_props().creep_factors`) as in the solver
    # → the creep checks verify only the correct wiring (factor × stiffness / strength in the case, units, the choice of the state),
    # NOT the physics of the creep of PLA (that is an indicative estimate, see printing/props.py). There is no physical reference for PLA.
    return [
        Check("krátkodobě: u = σL/E_xy", u0, P4_S * 100 / PLA.E, 0.001, "mm"),
        Check("tečení (zapojení tabulky creep_factors): u = σL/(k_E·E_xy)", u1, P4_S * 100 / (PLA.E * fac["stiffness"]),
              0.001, "mm"),
        Check("tečení (zapojení tabulky creep_factors): λ = k_R·Rm/σ", l1, fac["strength"] * PLA.Rm / P4_S, 0.002, ""),
        Check("krátkodobě: λ = Rm/σ", l0, PLA.Rm / P4_S, 0.002, ""),
        Check("tečení: reakce = σ·A", -s1.reactions["x0"][0], P4_S * 100, 0.001, "N"),
    ]


PRINT_CASES.update({
    "P3": Case("P3", "Tisk: anizotropie linie perimetrů (podél / napříč linie, směs s výplní)", p3_build, p3_eval),
    "P4": Case("P4", "Tisk: tečení – izochronní modul a dlouhodobá pevnost (1000 h, 30 °C)", p4_build, p4_eval),
})


# ------------------------------------------------------------------ P5
P5_A = 0.8          # multiple of the state P1 (10 MPa) → amplitude 8 MPa


def p5_build(wd: Path):
    from fea_core.fatigue.history import Channel, LoadHistory
    from fea_core.fatigue.settings import FatigueSettings
    st, parts = p1_build(wd)
    st.fatigue = FatigueSettings(history=LoadHistory([Channel("Tah 10 MPa", "sine", amplitude=P5_A, cycles=1)]),
                                 mean_correction="none", repeats=1.0)
    return st, parts


def p5_eval(res: Results, st: Study):
    from fea_core.fatigue.analysis import run_fatigue
    from fea_core.printing.props import default_props
    fr = run_fatigue(st, res)
    pr = default_props(PLA.name)
    SD = pr.sn_xy(PLA.Rm)
    x = res.mesh.nodes
    mid = np.abs(x[:, 0] - 50) < 20
    a, b = mid & (x[:, 1] < 15), mid & (x[:, 1] > 15)
    sa = P5_A * SIG
    eq_a = sa * PLA.Rm / PLA.Rm_z
    D = lambda s: (s / SD) ** pr.sn_k / pr.sn_ND
    sf = lambda s: SD * pr.sn_ND ** (1 / pr.sn_k) / s
    return [
        Check("naležato (XY): D = (σa/SD)^k/ND", float(np.median(fr.damage[b])), D(sa), 0.002, ""),
        Check("nastojato (Z): D = (σa·Rm/Rm_z/SD)^k/ND", float(np.median(fr.damage[a])), D(eq_a), 0.002, ""),
        Check("naležato: bezpečnost SD·ND^(1/k)/σa", float(np.median(fr.safety[b])), sf(sa), 0.002, ""),
        Check("nastojato: bezpečnost SD·ND^(1/k)/σ_eq", float(np.median(fr.safety[a])), sf(eq_a), 0.002, ""),
        Check("amplituda ekvivalentu naležato", float(np.median(fr.amplitude[b])), sa, 0.002, "MPa"),
    ]


PRINT_CASES["P5"] = Case("P5", "Tisk: únava – S-N tisku a Hillův ekvivalent v osách tisku (XY / Z)", p5_build, p5_eval)


# ------------------------------------------------------------------ P6
# Thin places (F5): plate 2 mm standing up (build z, 2 perimeters × 0.42 = 1.68 mm, infill 15 %) – the inside 0.32 mm
# with sparse infill is not filled → the slicer and the model both print full infill; the thick block does not change. Sweep of perimeters ×
# infill on a tensile bar: more perimeters / infill → smaller displacement and larger mass, recommendation = the lightest with k ≥ k_req.
P6_SPEC = dict(build_dir=[0, 0, 1], layer_height=0.2, line_width=0.42, walls=2, top_layers=5, bottom_layers=3,
               infill=0.15, pattern="grid")
P6_T, P6_LY, P6_LZ, P6_B = 2.0, 30.0, 20.0, 12.0
P6_SWEEP = dict(walls=[1, 3], infill=[0.1, 0.4])
P6_KREQ = 7.0


def p6_build(wd: Path):
    geo = [Part("Deska", box(0, 0, 0, P6_T, P6_LY, P6_LZ)), Part("Blok", box(10, 0, 0, P6_B, P6_B, P6_B))]
    st, parts = _study("P6 tenka mista", wd / "model.step", geo, MeshSettings(size=1.0))
    st.add_material(PLA, default=True)
    st.materials.pop("Ocel S235")
    for p in ("Deska", "Blok"):
        st.printing.parts[p] = PrintSpec(p, **P6_SPEC)
    st.add_selection("x0", find_faces(parts, x=0) + find_faces(parts, x=10))
    st.add_selection("y0", find_faces(parts, y=0))
    st.add_selection("z0", find_faces(parts, z=0))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0), Displacement("z0", uz=0.0)]
    st.add_selection("konce", find_faces(parts, y=P6_LY) + find_faces(parts, y=P6_B))
    st.load_cases.append(LoadCase("Tah 1 MPa", [Pressure("konce", -1.0)]))
    return st, parts


def _p6_bar(wd: Path):
    st, parts = _study("P6 rozmitani", wd / "tyc.step", [Part("Tyc", box(0, 0, 0, 60, 10, 10))], MeshSettings(size=2.5))
    st.add_material(PLA, default=True)
    st.materials.pop("Ocel S235")
    st.printing.parts["Tyc"] = PrintSpec("Tyc", **P6_SPEC)
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("y0", find_faces(parts, y=0))
    st.add_selection("z0", find_faces(parts, z=0))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0), Displacement("z0", uz=0.0)]
    st.add_selection("cela", find_faces(parts, x=60))
    st.load_cases.append(LoadCase("Tah 2 MPa", [Pressure("cela", -2.0)]))
    st.printing.required_safety = P6_KREQ
    from fea_core.printing.model import SweepSpec
    st.printing.sweep = SweepSpec(**P6_SWEEP)
    return st, parts


def p6_eval(res: Results, st: Study):
    import copy
    from fea_core.printing.failure import part_summary
    from fea_core.printing.region import assign
    from fea_core.printing.sweep import run_sweep
    info = res.info["printing"]["parts"]
    d, b = info["Deska"], info["Blok"]
    V = P6_T * P6_LY * P6_LZ
    # without the full infill of thin places: the shell = the whole volume − core (the inside 0.32 mm between the perimeters, without layers)
    w, tl = P6_SPEC["walls"] * P6_SPEC["line_width"], (P6_SPEC["top_layers"] + P6_SPEC["bottom_layers"]) * 0.2
    core = (P6_T - 2 * w) * (P6_LY - 2 * w) * (P6_LZ - tl)
    st2 = copy.deepcopy(st)
    for p in ("Deska", "Blok"):
        st2.printing.parts[p].solid_thin = False
    m2 = copy.deepcopy(res.mesh)
    assign(st2, m2)
    off = part_summary(st2, m2)
    # sweep on a tensile bar (own mesh, all variants on it)
    wd = Path(st.geometry).parent / "sweep"
    bar, parts = _p6_bar(wd.parent)
    vr = run_sweep(bar, wd, parts=parts)
    o = {x.name: x for x in vr.outcomes if x.name != "Current settings"}
    key = {(ov["Tyc"]["walls"], ov["Tyc"]["infill"]): n for n, ov in vr.sweep["variants"].items()}
    mono_u = mono_m = 0
    for f in P6_SWEEP["infill"]:
        a, c = o[key[(1, f)]], o[key[(3, f)]]
        mono_u += c.max_disp < a.max_disp
        mono_m += c.mass_g > a.mass_g
    for wl in P6_SWEEP["walls"]:
        a, c = o[key[(wl, 0.1)]], o[key[(wl, 0.4)]]
        mono_u += c.max_disp < a.max_disp
        mono_m += c.mass_g > a.mass_g
    ok = [x for x in vr.outcomes if x.ok]
    good = [x for x in ok if x.min_safety >= P6_KREQ]
    expect = min(good, key=lambda x: x.mass_g).name if good else max(ok, key=lambda x: x.min_safety).name
    front = vr.sweep["pareto"]
    ends = {min(ok, key=lambda x: (x.mass_g, -x.min_safety)).name, max(ok, key=lambda x: (x.min_safety, -x.mass_g)).name}
    return [
        Check("deska 2 mm: podíl plného materiálu (tenké místo → plná výplň)", d["shell_fraction"], 1.0, 1e-9, ""),
        Check("deska: hmotnost = ρ·V (plná)", d["mass_g"], PLA.rho * V * 1e6, 1e-6, "g"),
        Check("deska bez plné výplně tenkých míst: skořepina = 1 − V_jádra/V", off["Deska"]["shell_fraction"],
              1 - core / V, 0.03, ""),
        Check("blok 12 mm: podíl skořepiny beze změny (s / bez plné výplně tenkých míst)", b["shell_fraction"],
              off["Blok"]["shell_fraction"], 1e-12, ""),
        Check("blok: objem tenkých míst = 0 (1 + podíl)", 1.0 + b["thin_fraction"], 1.0, 1e-12, ""),
        Check("rozmítání: všechny varianty spočteny", float(len(ok)), float(len(vr.outcomes)), 0.0, ""),
        Check("rozmítání: víc perimetrů / výplně → menší posuv (4 dvojice)", float(mono_u), 4.0, 0.0, ""),
        Check("rozmítání: víc perimetrů / výplně → větší hmotnost (4 dvojice)", float(mono_m), 4.0, 0.0, ""),
        Check("doporučení = nejlehčí s k ≥ k_req", float(vr.recommended == expect), 1.0, 0.0, ""),
        Check("Pareto obsahuje nejlehčí a nejbezpečnější", float(len(ends & set(front))), float(len(ends)), 0.0, ""),
    ]


PRINT_CASES["P6"] = Case("P6", "Tisk: tenká místa s plnou výplní, rozmítání perimetry × výplň (monotonie, Pareto)",
                         p6_build, p6_eval)


# ------------------------------------------------------------------ P7 tolerance tisku (F7)
TOL7 = 0.25


def p7_build(wd: Path):
    """Two identical bars 100×10×10 lying (full extrusion), the second with a printing tolerance δ = 0.25 mm (the free surface
    moved by δ inwards): tension 10 MPa on the face → elongation × A₀/A_ef and the safety ÷ the same. The bearing faces (planar
    constraints y, z, the face x = 0) and the loaded face stand (the forces act on the real material) → only the two free side
    walls retreat: A_ef = (10 − δ)². The pressure on the face acts on the reduced section → force p·A_ef, the stress in the bar unchanged!
    Therefore the compliance (u/F) and the safety related to the force are compared.
    """
    geo = [Part("Tyc bez tolerance", box(0, 0, 0, 100, 10, 10)), Part("Tyc s toleranci", box(0, 20, 0, 100, 10, 10))]
    st, parts = _study("P7 tolerance tisku", wd / "model.step", geo, MeshSettings(size=1.5))
    st.add_material(PLA, default=True)
    st.materials.pop("Ocel S235")
    st.printing.criterion = "tsai_hill"
    st.printing.parts["Tyc bez tolerance"] = PrintSpec("Tyc bez tolerance", build_dir=[0, 0, 1], enabled=False)
    st.printing.parts["Tyc s toleranci"] = PrintSpec("Tyc s toleranci", build_dir=[0, 0, 1], enabled=False,
                                                     tolerance=TOL7)
    _rollers(st, parts)
    st.add_selection("cela", find_faces(parts, x=100))
    st.add_selection("cela_b", find_faces(parts, part="Tyc s toleranci", x=100))
    st.load_cases.append(LoadCase("Tah 10 MPa", [Pressure("cela", -SIG)]))
    return st, parts


def p7_eval(res: Results, st: Study):
    s = res.step(0)
    x = res.mesh.nodes
    a = (np.abs(x[:, 0] - 100) < 1e-6) & (x[:, 1] < 15)
    b = (np.abs(x[:, 0] - 100) < 1e-6) & (x[:, 1] > 15)
    A0, A1 = 100.0, (10 - TOL7) ** 2
    ratio = float(s.U[b, 0].mean() / s.U[a, 0].mean())     # the same pressure → the same stress → the same elongation
    ref = 1.0
    pa, pb = s.printing["parts"]["Tyc bez tolerance"], s.printing["parts"]["Tyc s toleranci"]
    vol = res.info["printing"]["parts"] if res.info.get("printing") else {}
    out = [
        Check("prodloužení s tolerancí / bez (stejný tlak = stejné napětí)", ratio, 1.0, 0.002, ""),
        Check("σx uprostřed tyče s tolerancí = p", res.probe([50, 25, 5], "S")[0], SIG, 0.002, "MPa"),
        Check("min. bezpečnost s tolerancí / bez (min. přes uzly, šum posunutých prvků u hran)",
              pb["min_safety"] / pa["min_safety"], 1.0, 0.03, ""),
        Check("obsah zatíženého čela s tolerancí = (10 − δ)²", res.mesh.surface_area("cela_b"), A1, 0.002, "mm²"),
        Check("tyč bez tolerance: u = σL/E (beze změny)", float(s.U[a, 0].mean()), SIG * 100 / PLA.E, 0.001, "mm"),
    ]
    if vol:
        m0, m1 = vol["Tyc bez tolerance"]["mass_g"], vol["Tyc s toleranci"]["mass_g"]
        out.append(Check("hmotnost s tolerancí / bez = A_ef/A₀", m1 / m0, A1 / A0, 0.002, ""))
    return out


PRINT_CASES["P7"] = Case("P7", "Tolerance tisku: vrstva δ bez materiálu (tuhost, bezpečnost, hmotnost)", p7_build, p7_eval)
