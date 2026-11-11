"""Verification of the elastic support (fea_core.springs, ElasticSupport) and of separately meshed parts (*TIE).

V15 (a) cube 10×10×10 on a foundation k = 100 N/mm³ normal, pressure 1 MPa on top: u_z of the bottom = −p/k, reaction = p·A;
        the same with the total stiffness K = k·A (per_area = False) and in all directions;
    (b) long beam 1000×20×20 on a Winkler foundation k = 1 N/mm³ (k_b = k·b), force P = 1 kN in the middle:
        Hetényi w₀ = P·λ/(2k_b), M(x) = P/(4λ)·e^(−λx)(cos λx − sin λx), λ = (k_b/(4EI))^¼ (λL = 6.5 → "infinite"
        beam); stress 2h from the force (directly under it the local effect of the load).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.study import Displacement, ElasticSupport, Force, LoadCase, MeshSettings, Pressure
from verification.cases import Case, Check, E, _study, box

K_A = 100.0


def v15_build(wd: Path):
    st, parts = _study("V15 pruzne ulozeni", wd / "model.step",
                       [Part("Kostka", box(0, 0, 0, 10, 10, 10)), Part("Nosnik", box(100, 0, 0, 1000, 20, 20)),
                        Part("Zatez", box(595, 0, 20, 10, 20, 1))], MeshSettings(size=5.0))
    # the load strip ("Zatez") is connected to the beam by a shared mesh – the force is distributed over 10 mm of length
    st.add_selection("dno", find_faces(parts, normal=[0, 0, -1], where=lambda p: p["centroid"][0] < 50))
    st.add_selection("vrch", find_faces(parts, normal=[0, 0, 1], where=lambda p: p["centroid"][0] < 50))
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("y0", find_faces(parts, y=0, where=lambda p: p["centroid"][0] < 50))
    st.add_selection("nosnik_dno", find_faces(parts, z=0, where=lambda p: p["centroid"][0] > 50))
    st.add_selection("nosnik_x0", find_faces(parts, x=100))
    st.add_selection("nosnik_y0", find_faces(parts, y=0, where=lambda p: p["centroid"][0] > 50 and p["area"] > 1000))
    st.add_selection("zatez", find_faces(parts, z=21))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0),
                       ElasticSupport("dno", K_A, "normal"),
                       Displacement("nosnik_x0", ux=0.0), Displacement("nosnik_y0", uy=0.0),
                       ElasticSupport("nosnik_dno", 1.0, "normal")]
    st.load_cases.append(LoadCase("Tlak a síla", [Pressure("vrch", 1.0), Force("zatez", [0, 0, -1000.0])]))
    return st, parts


def v15_eval(res, st):
    s = res.step(0)
    x = res.mesh.nodes
    bot = np.nonzero((np.abs(x[:, 2]) < 1e-6) & (x[:, 0] < 50))[0]
    kb = 1.0 * 20.0
    EI = E * 20.0 * 20.0 ** 3 / 12
    lam = (kb / (4 * EI)) ** 0.25
    w0 = 1000.0 * lam / (2 * kb)
    M0 = 1000.0 / (4 * lam)
    mid = np.nonzero((np.abs(x[:, 0] - 600) < 1e-6) & (np.abs(x[:, 2]) < 1e-6))[0]
    xx = 40.0                                     # 2h from the force (directly under the force the local effect of the load – Saint-Venant)
    Mx = M0 * np.exp(-lam * xx) * (np.cos(lam * xx) - np.sin(lam * xx))
    sb = res.probe([600.0 + xx, 10.0, 0.0], "S")[0]
    stop = res.probe([600.0 + xx, 10.0, 20.0], "S")[0]
    return [
        Check("(a) u_z dna kostky = −p/k", -float(s.U[bot, 2].mean()), 1.0 / K_A, 0.01, "mm"),
        Check("(a) reakce podloží kostky = p·A", float(s.reactions["dno"][2]), 100.0, 1e-4, "N"),
        Check("(b) průhyb pod silou w₀ = Pλ/(2k_b) (Hetényi)", -float(s.U[mid, 2].mean()), w0, 0.03, "mm"),
        Check("(b) σx dolní vlákno 40 mm od síly = M/W (Hetényi)", sb, Mx / (20.0 * 20.0 ** 2 / 6), 0.03, "MPa"),
        Check("(b) σx horní vlákno 40 mm od síly = −M/W", -stop, Mx / (20.0 * 20.0 ** 2 / 6), 0.03, "MPa"),
        Check("(b) reakce podloží nosníku = P", float(s.reactions["nosnik_dno"][2]), 1000.0, 1e-4, "N"),
        Check("rovnováha (nerovnováha + 1)", 1.0 + s.equilibrium["imbalance"], 1.0, 1e-6, ""),
    ]


SUPPORT_CASES = {
    "V15": Case("V15", "Pružné uložení: kostka na podloží, nosník na Winklerově podloží (Hetényi)", v15_build, v15_eval),
}


def v17_build(wd: Path):
    """Cantilever V6 (two bodies A, B) – part B meshed separately and finer (non-conforming mesh), connection *TIE."""
    from verification.cases import v1_build
    st, parts = v1_build(wd, split=True)
    st.name = "V17 TIE delena konzola"
    st.mesh.separate = ["Konzola B"]
    st.add_selection("b_cela", find_faces(parts, part="Konzola B", normal=[0, 1, 0]))
    st.mesh.local = [{"selection": "b_cela", "size": 1.6}]
    return st, parts


def v17_eval(res, st):
    s = res.step(0)
    x = res.mesh.nodes
    end = np.nonzero(np.abs(x[:, 0] - 200) < 1e-6)[0]
    ties = res.mesh.report.get("ties", [])
    return [
        Check("plocha *TIE = průřez 20×10", ties[0]["area_mm2"] if ties else 0.0, 200.0, 1e-3, "mm²"),
        Check("průhyb konce (Timoshenko, jako V1)", -float(s.U[end, 2].mean()), 0.7634, 0.03, "mm"),
        Check("σx horní vlákno x = 50 (díl A)", res.probe([50, 10, 10], "S")[0], 45.0, 0.05, "MPa"),
        Check("σx horní vlákno x = 150 (díl B, jiná síť)", res.probe([150, 10, 10], "S")[0], 15.0, 0.05, "MPa"),
        Check("reakce Fz", float(s.reactions["vetknuti"][2]), 100.0, 1e-4, "N"),
    ]


SUPPORT_CASES["V17"] = Case("V17", "Samostatně síťovaný díl spojený *TIE (dělená konzola, nekonformní síť)",
                            v17_build, v17_eval)


# ------------------------------------------------------------------ V18 progress along a line and stress linearisation
def v18_build(wd: Path):
    """Quarter of a thick-walled tube V4 (ri 20, ro 40, p = 100 MPa) + cantilever V1 (bending) shifted to x = 100."""
    import math as _m
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
    from verification.cases import cyl, solid, RI, RO, V4_H, P4
    from fea_core.study import Fixed
    ring = BRepAlgoAPI_Cut(cyl((0, 0, 0), (0, 0, 1), RO, V4_H), cyl((0, 0, -1), (0, 0, 1), RI, V4_H + 2)).Shape()
    quarter = solid(BRepAlgoAPI_Common(ring, box(0, 0, -1, RO + 1, RO + 1, V4_H + 2)).Shape())
    st, parts = _study("V18 linearizace", wd / "model.step",
                       [Part("Trubka", quarter), Part("Konzola", box(100, 0, 0, 200, 20, 10))], MeshSettings(size=2.5))
    st.add_selection("sym_x", find_faces(parts, part="Trubka", x=0))
    st.add_selection("sym_y", find_faces(parts, part="Trubka", y=0))
    st.add_selection("sym_z", find_faces(parts, part="Trubka", z=0))
    st.add_selection("vnitrni", find_faces(parts, radius=RI))
    st.add_selection("vetknuti", find_faces(parts, x=100))
    st.add_selection("konec", find_faces(parts, x=300))
    st.constraints += [Displacement("sym_x", ux=0.0), Displacement("sym_y", uy=0.0), Displacement("sym_z", uz=0.0),
                       Fixed("vetknuti")]
    st.load_cases.append(LoadCase("Tlak a ohyb", [Pressure("vnitrni", P4), Force("konec", [0, 0, -100.0])]))
    return st, parts


def v18_eval(res, st):
    import math as _m
    from scipy.integrate import quad
    from fea_core.paths import linearize, line_values
    from verification.cases import RI, RO, V4_H, P4
    a = P4 * RI ** 2 / (RO ** 2 - RI ** 2)
    sth = lambda r: a * (1 + RO ** 2 / r ** 2)
    t = RO - RI
    m_ref = quad(sth, RI, RO)[0] / t
    b_ref = 6 / t ** 2 * quad(lambda r: sth(r) * (t / 2 - (r - RI)), RI, RO)[0]
    phi = _m.radians(45)
    e = np.array([_m.cos(phi), _m.sin(phi), 0.0])
    p0, p1 = e * RI + [0, 0, V4_H / 2], e * RO + [0, 0, V4_H / 2]
    hoop = np.array([-_m.sin(phi), _m.cos(phi), 0.0])
    lin = linearize(res, p0, p1, hint=hoop)           # axis b = circumferential
    lb = linearize(res, [200, 10, 0], [200, 10, 10], hint=[1, 0, 0])   # cantilever x = 200 (half): b = axis x
    lv = line_values(res, [200, 10, 0], [200, 10, 10], "sx", n=11)
    return [
        Check("trubka: membrána σθ = p·ri/t (rovnováha)", lin["membrane"]["local"][1], m_ref, 0.01, "MPa"),
        Check("trubka: ohyb σθ = 6/t²∫σθ(t/2 − x)dx (Lamé)", lin["bending"]["local"][1], b_ref, 0.02, "MPa"),
        Check("trubka: membrána σr = (−p)…0 průměr", lin["membrane"]["local"][0],
              quad(lambda r: a * (1 - RO ** 2 / r ** 2), RI, RO)[0] / t, 0.02, "MPa"),
        Check("konzola x = 200: ohyb σx na spodním vlákně = −M/W (M = 100·100)", -lb["bending"]["local"][1], 30.0,
              0.02, "MPa"),
        Check("konzola: membrána σx ≈ 0 (+30)", 30.0 + lb["membrane"]["local"][1], 30.0, 0.005, "MPa"),
        Check("konzola: σx podél výšky – spodní vlákno (tlak)", -float(lv["values"][0]), 30.0, 0.03, "MPa"),
    ]


SUPPORT_CASES["V18"] = Case("V18", "Výsledky podél čáry a linearizace napětí (Lamé, nosník)", v18_build, v18_eval)
