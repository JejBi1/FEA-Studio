"""Verification of the frame system (fea_core.beams): the own solver with the exact Timoshenko element (beams/timoshenko.py);
the mixed model body + member (F8) stays on ccx U1.

F1 cantilever IPE 200, L = 2 m: force at the end (Timoshenko, M = FL), torsion T (φ = TL/GJ, J ≠ Ip), distributed q – 1 element is EXACT.
F2 simply supported beam HEB 200, L = 4 m, q = 10 N/mm: w = 5qL⁴/384EI + qL²/(8κGA), M = qL²/8, reaction qL/2.
F3 frame (portal) with clamped feet, horizontal force H at the level of the crossbar: slope–deflection method (axially rigid members,
   E–B): M_feet = (Hh/2)(3k + 1)/(6k + 1), M_corner = (Hh/2)·3k/(6k + 1), k = I_b·h/(I_c·L).
F4 quarter-circle cantilever (arc R = 500, CHS 60.3×4) with a force at the end: Castigliano (bending + tension + shear).
F5 truss pair of members (V) with a force at the apex: N = P/(2 sin θ).
F6 Gerber beam (hinge in the span): M_clamping = P/2·L₁, moment in the hinge 0.
F7 natural frequencies of the simply supported beam IPE 200: exact Timoshenko solution (shear + rotary inertia), torsion 1/4 wave.
F9 linear buckling: Euler with the Engesser shear correction P_E/(1 + P_E/GA_s).
F10 accuracy of 1 element (strong / weak axis, rotated section, spring) and independence of the subdivision of the member.
F11 spatial frame (hinges, truss members, rotated sections, spring, loads in local and global axes):
    equilibrium of each node from the end forces, integration of the displacements along the member = displacement of the node. F11M = the same against MYSTRAN CBAR.
F12 torsional and bending frequencies: O(h²) convergence of the consistent mass.
F13 linear buckling: Euler (pinned column, cantilever, pinned–clamped) 2–8 elements.
F14 temperature: restrained bar N = −EAαΔT, free expansion, gradient across the section.
"""
from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path

import numpy as np

from fea_core.beams import sections as S
from fea_core.beams.model import (FrameMember, FrameSupport, SectionDef, FrameNodeLoad, FrameMemberLoad,
                                  FrameTemperature)
from fea_core.study import Study, Material, LoadCase, Gravity
from verification.cases import Case, Check, E, NU, RHO

G = E / (2 * (1 + NU))
ALPHA_T = 12e-6


def _study(name: str) -> Study:
    st = Study(name, "")
    st.add_material(Material("Ocel S235", E, NU, RHO, Re=235.0, Rm=360.0, alpha=ALPHA_T), default=True)
    return st


def _fr(res, k=0):
    return res.step(k).frame


def _again(st, mod=None):
    """Another calculation of the same study (other subdivision, other settings) by the own solver – without writing to out."""
    from fea_core.beams.run import run_frame_tim
    s2 = copy.deepcopy(st)
    if mod is not None:
        mod(s2)
    with tempfile.TemporaryDirectory() as td:
        return run_frame_tim(s2, td)


def tim_ss_freq(n: int, L: float, A: float, As: float, I: float) -> float:
    """Exact natural frequency [Hz] of the n-th bending mode of a simply supported Timoshenko beam (ρ²I/κG)ω⁴ − [ρA + ρI(1 + E/κG)λ²]ω² +
    EIλ⁴ = 0, λ = nπ/L, κG = G·A_s/A (Han, Benaroya, Wei 1999) – the lower root.
    """
    lam = n * math.pi / L
    kG = G * As / A
    a = RHO ** 2 * I / kG
    b = -(RHO * A + RHO * I * (1 + E / kG) * lam ** 2)
    c = E * I * lam ** 4
    return math.sqrt((-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)) / (2 * math.pi)


# ------------------------------------------------------------------ F1
def f1_build(wd: Path):
    st = _study("F1 konzola IPE 200")
    fr = st.frame
    fr.sections["IPE200"] = SectionDef("IPE200", library="IPE 200")
    fr.add_node("A", [0, 0, 0])
    fr.add_node("B", [2000, 0, 0])
    fr.members["m1"] = FrameMember("m1", "A", "B", "IPE200")
    fr.supports.append(FrameSupport("A"))
    st.load_cases += [LoadCase("Síla", [FrameNodeLoad("B", [0, 0, -1000.0])]),
                      LoadCase("Krut", [FrameNodeLoad("B", moment=[1e5, 0, 0])]),
                      LoadCase("Spojité", [FrameMemberLoad("m1", [0, 0, -1.0])])]
    return st, []


def f1_eval(res, st):
    sec = S.get("IPE 200")
    L, k = 2000.0, sec.Av_z / sec.A
    s0, s1, s2 = _fr(res, 0), _fr(res, 1), _fr(res, 2)
    m0, m2 = s0["members"]["m1"], s2["members"]["m1"]
    return [
        Check("průhyb síla FL³/3EI + FL/κGA (1 prvek)", -s0["nodes"]["B"][2],
              1000 * L ** 3 / (3 * E * sec.Iy) + 1000 * L / (k * G * sec.A), 1e-9, "mm"),
        Check("M kořen = F·L", m0["My"][0], 1000 * L, 1e-10, "N·mm"),
        Check("posouvající síla Vz = −F", -m0["Vz"][0], 1000.0, 1e-10, "N"),
        Check("reakce Fz", s0["reactions"]["A"][2], 1000.0, 1e-10, "N"),
        Check("zkroucení T·L/(G·J) (J ≠ Ip)", s1["nodes"]["B"][3], 1e5 * L / (G * sec.J), 1e-9, "rad"),
        Check("krouticí moment T", s1["members"]["m1"]["T"][0], 1e5, 1e-10, "N·mm"),
        Check("spojité: průhyb qL⁴/8EI + qL²/2κGA", -s2["nodes"]["B"][2],
              L ** 4 / (8 * E * sec.Iy) + L ** 2 / (2 * k * G * sec.A), 1e-9, "mm"),
        Check("spojité: M kořen = qL²/2", m2["My"][0], L * L / 2, 1e-10, "N·mm"),
        Check("spojité: posuv z obrazce na konci prutu = posuv uzlu", m2["u"][-1][2], s2["nodes"]["B"][2], 1e-9, "mm"),
    ]


# ------------------------------------------------------------------ F2
def f2_build(wd: Path):
    st = _study("F2 prosty nosnik HEB 200")
    fr = st.frame
    fr.sections["HEB200"] = SectionDef("HEB200", library="HEB 200")
    fr.add_node("A", [0, 0, 0])
    fr.add_node("B", [4000, 0, 0])
    fr.members["m1"] = FrameMember("m1", "A", "B", "HEB200")
    fr.supports += [FrameSupport("A", [True, True, True, True, False, False]),
                    FrameSupport("B", [False, True, True, False, False, False])]
    st.load_cases.append(LoadCase("q", [FrameMemberLoad("m1", [0, 0, -10.0])]))
    return st, []


def f2_eval(res, st):
    sec = S.get("HEB 200")
    L, q, k = 4000.0, 10.0, sec.Av_z / sec.A
    m = _fr(res)["members"]["m1"]
    s = np.asarray(m["s"])
    i = int(np.argmin(np.abs(s - L / 2)))
    w = np.asarray(m["u"])[i, 2]
    return [
        Check("průhyb uprostřed 5qL⁴/384EI + qL²/8κGA (1 prvek)", -w,
              5 * q * L ** 4 / (384 * E * sec.Iy) + q * L ** 2 / (8 * k * G * sec.A), 1e-9, "mm"),
        Check("M uprostřed = qL²/8 (tah dole: My < 0)", -m["My"][i], q * L * L / 8, 1e-10, "N·mm"),
        Check("max. |M| z obrazce = qL²/8 (bod V = 0)", float(np.abs(m["My"]).max()), q * L * L / 8, 1e-10, "N·mm"),
        Check("reakce A = qL/2", _fr(res)["reactions"]["A"][2], q * L / 2, 1e-10, "N"),
        Check("reakce B = qL/2", _fr(res)["reactions"]["B"][2], q * L / 2, 1e-10, "N"),
    ]


# ------------------------------------------------------------------ F3
H3, L3, h3 = 10000.0, 6000.0, 4000.0


def _rigid_axial(name: str, lib: str) -> SectionDef:
    """Section with the area ×1e6: an axially rigid member (the slope–deflection method assumes incompressible members)."""
    d = S.get(lib).to_dict()
    d["A"] *= 1e6
    return SectionDef(name, props=d)


def f3_build(wd: Path):
    st = _study("F3 portalovy ram")
    fr = st.frame
    fr.settings.shear_deformation = False
    fr.sections["IPE300"] = _rigid_axial("IPE300", "IPE 300")
    fr.sections["IPE400"] = _rigid_axial("IPE400", "IPE 400")
    for n, p in (("A", [0, 0, 0]), ("B", [0, 0, h3]), ("C", [L3, 0, h3]), ("D", [L3, 0, 0])):
        fr.add_node(n, p)
    # frame in the plane xz: bending in the plane → the axis y of the section ⟂ to the plane of the frame (up = the axis z of the section in the plane of the frame)
    fr.members["sloup1"] = FrameMember("sloup1", "A", "B", "IPE300", up=[1, 0, 0])
    fr.members["pricel"] = FrameMember("pricel", "B", "C", "IPE400", up=[0, 0, 1])
    fr.members["sloup2"] = FrameMember("sloup2", "D", "C", "IPE300", up=[1, 0, 0])
    fr.supports += [FrameSupport("A"), FrameSupport("D")]
    st.load_cases.append(LoadCase("H", [FrameNodeLoad("B", [H3, 0, 0])]))
    return st, []


def f3_eval(res, st):
    Ic, Ib = S.get("IPE 300").Iy, S.get("IPE 400").Iy
    k = Ib * h3 / (Ic * L3)
    Mb_ref = H3 * h3 / 2 * (3 * k + 1) / (6 * k + 1)
    Mt_ref = H3 * h3 / 2 * 3 * k / (6 * k + 1)
    m = _fr(res)["members"]["sloup1"]
    m2 = _fr(res)["members"]["sloup2"]
    return [
        Check("moment v patě sloupu (sklon–průhyb, osově tuhé pruty)", abs(m["My"][0]), Mb_ref, 1e-5, "N·mm"),
        Check("moment v rohu rámu", abs(m["My"][-1]), Mt_ref, 1e-5, "N·mm"),
        Check("moment v patě druhého sloupu", abs(m2["My"][0]), Mb_ref, 1e-5, "N·mm"),
        # A ×1e6 worsens the conditioning (stiffness ratio 1e6) → equilibrium ~1e-9 instead of 1e-12
        Check("součet vodorovných reakcí = H", -(_fr(res)["reactions"]["A"][0] + _fr(res)["reactions"]["D"][0]), H3, 1e-7, "N"),
        Check("zbytek rovnováhy v uzlech", res.step(0).frame["node_residual"], 0.0, 1e-8, ""),
    ]


# ------------------------------------------------------------------ F4
R4 = 500.0


def f4_build(wd: Path):
    st = _study("F4 obloukova konzola")
    fr = st.frame
    fr.settings.max_arc_angle = 0.5                # chords: error O(h²) – 0.5° for a tight comparison
    fr.sections["CHS"] = SectionDef("CHS", library="CHS 60.3x4")
    fr.add_node("A", [R4, 0, 0])
    fr.add_node("B", [0, R4, 0])
    c = math.cos(math.pi / 4) * R4
    fr.members["oblouk"] = FrameMember("oblouk", "A", "B", "CHS", up="normal", through=[c, c, 0])
    fr.supports.append(FrameSupport("A"))
    # at the end (angle 90°): tangent in the direction of increasing angle = −x, radially outwards = +y
    st.load_cases += [LoadCase("Ft", [FrameNodeLoad("B", [-100.0, 0, 0])]),
                      LoadCase("Fr", [FrameNodeLoad("B", [0, 100.0, 0])])]
    return st, []


def f4_eval(res, st):
    from fea_core.beams.analytic import arc_cantilever
    sec = S.get("CHS 60.3x4")
    out = []
    coarse = _again(st, lambda s: setattr(s.frame.settings, "max_arc_angle", 5.0))      # the own solver divides at most every 2°
    for k, (name, kw) in enumerate((("Ft", {"Ft": 100.0}), ("Fr", {"Fr": 100.0}))):
        a = arc_cantilever(R4, math.pi / 2, E=E, I=sec.Iz, A=sec.A, A_s=sec.Av_z, nu=NU, axial=True, shear=True, **kw)
        u = _fr(res, k)["nodes"]["B"]
        uc = _fr(coarse, k)["nodes"]["B"]
        out += [Check(f"{name}: posuv konce ux (Castigliano, 0,5°)", u[0], a.ux, 2e-4, "mm"),
                Check(f"{name}: posuv konce uy (Castigliano, 0,5°)", u[1], a.uy, 2e-4, "mm"),
                Check(f"{name}: posuv konce uy (Castigliano, 2° – chyba tětiv)", uc[1], a.uy, 2e-3, "mm")]
    return out


# ------------------------------------------------------------------ F5
def f5_build(wd: Path):
    st = _study("F5 prihrada V")
    fr = st.frame
    fr.sections["T"] = SectionDef("T", shape="CIRC", dims={"d": 20.0})
    for n, p in (("A", [0, 0, 0]), ("B", [2000, 0, 0]), ("C", [1000, 0, 1500])):
        fr.add_node(n, p)
    fr.members["AC"] = FrameMember("AC", "A", "C", "T", kind="truss")
    fr.members["BC"] = FrameMember("BC", "B", "C", "T", kind="truss")
    fr.supports += [FrameSupport("A", [True, True, True, False, False, False]),
                    FrameSupport("B", [True, True, True, False, False, False]),
                    FrameSupport("C", [False, True, False, False, False, False])]
    st.load_cases.append(LoadCase("P", [FrameNodeLoad("C", [0, 0, -10000.0])]))
    return st, []


def f5_eval(res, st):
    th = math.atan2(1500, 1000)
    N = -10000.0 / (2 * math.sin(th))
    m = _fr(res)["members"]
    # displacement of the vertex: δ = N·L/(EA) (uniform members at an angle θ): u_z = P·L/(2·EA·sin²θ)·... → check via the energy
    L = math.hypot(1000, 1500)
    A = math.pi * 20.0 ** 2 / 4
    uz = _fr(res)["nodes"]["C"][2]
    return [Check("osová síla AC = −P/(2 sin θ)", m["AC"]["N"][0], N, 1e-10, "N"),
            Check("osová síla BC", m["BC"]["N"][0], N, 1e-10, "N"),
            Check("ohybový moment v příhradovém prutu = 0 (abs.)", float(np.abs(m["AC"]["My"]).max()), 0.0, 1e-6, "N·mm"),
            Check("svislý posuv vrcholu = −N·L/(EA·sin θ)", uz, N * L / (E * A) / math.sin(th), 1e-9, "mm")]


# ------------------------------------------------------------------ F6
def f6_build(wd: Path):
    st = _study("F6 Gerberuv nosnik")
    fr = st.frame
    fr.sections["IPE200"] = SectionDef("IPE200", library="IPE 200")
    for n, p in (("A", [0, 0, 0]), ("C", [2000, 0, 0]), ("D", [3000, 0, 0]), ("B", [4000, 0, 0])):
        fr.add_node(n, p)
    fr.members["AC"] = FrameMember("AC", "A", "C", "IPE200", releases_end=[False, False, False, False, True, False])
    fr.members["CD"] = FrameMember("CD", "C", "D", "IPE200")
    fr.members["DB"] = FrameMember("DB", "D", "B", "IPE200")
    fr.supports += [FrameSupport("A"), FrameSupport("B", [False, True, True, True, False, False])]
    st.load_cases.append(LoadCase("P", [FrameNodeLoad("D", [0, 0, -1000.0])]))
    return st, []


def f6_eval(res, st):
    m = _fr(res)["members"]
    return [Check("moment ve vetknutí = P/2·L₁", m["AC"]["My"][0], 500.0 * 2000, 1e-10, "N·mm"),
            Check("moment v kloubu = 0 (abs.)", abs(m["AC"]["My"][-1]), 0.0, 1e-6, "N·mm"),
            Check("reakce B = P/2", _fr(res)["reactions"]["B"][2], 500.0, 1e-10, "N"),
            Check("reakce A = P/2", _fr(res)["reactions"]["A"][2], 500.0, 1e-10, "N")]


# ------------------------------------------------------------------ F7
def f7_build(wd: Path):
    st, _ = f2_build(wd)
    st.name = "F7 vlastni frekvence prosteho nosniku"
    st.frame.sections["IPE200"] = SectionDef("IPE200", library="IPE 200")
    st.frame.members["m1"].section = "IPE200"
    st.load_cases = []
    st.modal.enabled = True
    st.modal.modes = 12
    return st, []


def _nearest(vals, ref):
    return min(vals, key=lambda v: abs(v - ref))


def f7_eval(res, st):
    sec = S.get("IPE 200")
    L = 4000.0
    f = res.modes[0].values
    ft = 1 / (4 * L) * math.sqrt(G * sec.J / (RHO * (sec.Iy + sec.Iz)))     # torsion: rx constrained only at A
    w = [tim_ss_freq(n, L, sec.A, sec.Av_y, sec.Iz) for n in (1, 2, 3)]       # weak axis (shear V_y: A_sy)
    s = [tim_ss_freq(n, L, sec.A, sec.Av_z, sec.Iy) for n in (1, 2)]          # strong axis (A_sz)
    return [Check("f₁ slabá osa (Timoshenko, 8 prvků)", _nearest(f, w[0]), w[0], 1e-4, "Hz"),
            Check("f₂ slabá osa", _nearest(f, w[1]), w[1], 1e-3, "Hz"),
            Check("f₃ slabá osa", _nearest(f, w[2]), w[2], 5e-3, "Hz"),
            Check("f₁ silná osa (smyk + rotační setrvačnost)", _nearest(f, s[0]), s[0], 5e-4, "Hz"),
            Check("f torze (A vázané, B volné): 1/(4L)·√(GJ/ρIp)", _torsional(res)[0], ft, 3e-3, "Hz")]


def _torsional(res):
    """Frequencies of the modes without translational effective mass (torsion)."""
    m = res.modes[0]
    tot = max(m.total_mass[:3]) if m.total_mass else 1.0
    return [f for f, em in zip(m.values, m.eff_mass) if sum(em[:3]) < 1e-6 * tot] or [float("nan")]


FRAME_CASES = {
    "F1": Case("F1", "Pruty: konzola IPE 200 (síla, krut s J, spojité) – přesný Timoshenko, 1 prvek", f1_build, f1_eval),
    "F2": Case("F2", "Pruty: prostý nosník HEB 200 se spojitým zatížením", f2_build, f2_eval),
    "F3": Case("F3", "Pruty: portálový rám (sklon–průhyb)", f3_build, f3_eval),
    "F4": Case("F4", "Pruty: čtvrtkruhový oblouk (Castigliano)", f4_build, f4_eval),
    "F5": Case("F5", "Pruty: příhradová dvojice (osové síly)", f5_build, f5_eval),
    "F6": Case("F6", "Pruty: Gerberův nosník s kloubem", f6_build, f6_eval),
    "F7": Case("F7", "Pruty: vlastní frekvence prostého nosníku (Timoshenko, torze)", f7_build, f7_eval),
}


# ------------------------------------------------------------------ F8 mixed model body + member
def f8_build(wd: Path):
    """Body cantilever V1 (200×20×10, clamped at x = 0) and at the face x = 200 a rigidly connected member of the same section
    (rectangle 20×10) up to x = 400; force 100 N at the end of the member → cantilever L = 400: w ≈ PL³/3EI + PL/κGA (the rigid face
    of the connection stiffens locally), reactions, moment at the connection point = P·200. Mixed model: the member stays on ccx U1.
    """
    from fea_core.beams.model import FrameConnection
    from verification.cases import v1_build
    st, parts = v1_build(wd)
    st.name = "F8 teleso + prut"
    st.load_cases = []
    fr = st.frame
    fr.sections["R"] = SectionDef("R", shape="RECT", dims={"b": 20.0, "h": 10.0})
    fr.add_node("J", [200, 10, 5])
    fr.add_node("K", [400, 10, 5])
    fr.members["prut"] = FrameMember("prut", "J", "K", "R")
    fr.connections.append(FrameConnection("J", "konec"))
    st.load_cases.append(LoadCase("Síla na konci prutu", [FrameNodeLoad("K", [0, 0, -100.0])]))
    return st, parts


def f8_eval(res, st):
    I, A, L = 20 * 10 ** 3 / 12, 200.0, 400.0
    w = 100 * L ** 3 / (3 * E * I) + 100 * L / (5 / 6 * G * A)
    fr_ = res.step(0).frame
    s = res.step(0)
    return [
        Check("průhyb konce prutu (konzola 400, E–B + smyk)", -fr_["nodes"]["K"][2], w, 0.03, "mm"),
        # rectangle 20×10: the library gives the strong axis y along the dimension 20 (rotation of the principal axes) → vertical bending = Mz
        Check("moment v prutu u spojení = P·200", abs(fr_["members"]["prut"]["Mz"][0]), 100.0 * 200, 1e-3, "N·mm"),
        Check("reakce vetknutí tělesa Fz = P", float(s.reactions["vetknuti"][2]), 100.0, 1e-3, "N"),
        Check("síla prut → těleso Fz (.dat 6 číslic)", -fr_["coupling_forces"]["J"][2], -100.0, 5e-3, "N"),
    ]


FRAME_CASES["F8"] = Case("F8", "Pruty + tělesa: prut tuze připojený k ploše konzoly (ccx U1)", f8_build, f8_eval)


# ------------------------------------------------------------------ F9 buckling (Euler + Engesser)
def f9_build(wd: Path):
    """Buckling: cantilever IPE 200 L = 2 m (β = 2) and a pinned column HEA 200 L = 5 m (β = 1) in one system (two
    independent parts), pressure 1 kN on each → the lowest λ = min(π²EI_z/(β L)²)/F, with the shear P_E/(1 + P_E/GA_s).
    """
    st = _study("F9 vzper prutu")
    fr = st.frame
    fr.sections["IPE200"] = SectionDef("IPE200", library="IPE 200")
    fr.sections["HEA200"] = SectionDef("HEA200", library="HEA 200")
    fr.add_node("A", [0, 0, 0])
    fr.add_node("B", [2000, 0, 0])
    fr.members["konzola"] = FrameMember("konzola", "A", "B", "IPE200")
    fr.add_node("C", [0, 1000, 0])
    fr.add_node("D", [5000, 1000, 0])
    fr.members["sloup"] = FrameMember("sloup", "C", "D", "HEA200")
    fr.supports += [FrameSupport("A"), FrameSupport("C", [True, True, True, True, False, False]),
                    FrameSupport("D", [False, True, True, False, False, False])]
    st.load_cases.append(LoadCase("Tlak", [FrameNodeLoad("B", [-1000.0, 0, 0]), FrameNodeLoad("D", [-1000.0, 0, 0])]))
    st.buckling.enabled = True
    st.buckling.modes = 4
    return st, []


def f9_eval(res, st):
    s1, s2 = S.get("IPE 200"), S.get("HEA 200")
    pe_c = math.pi ** 2 * E * s1.Iz / (2 * 2000.0) ** 2
    pe_s = math.pi ** 2 * E * s2.Iz / 5000.0 ** 2
    lam_c = pe_c / (1 + pe_c / (G * s1.Av_y)) / 1000                       # Engesser (shear compliance)
    lam_s = pe_s / (1 + pe_s / (G * s2.Av_y)) / 1000
    vals = res.modes[-1].values
    near = lambda r: min(vals, key=lambda v: abs(v - r))
    return [Check("vzpěr konzoly β = 2 (Euler + Engesser, 8 prvků)", near(lam_c), lam_c, 2e-4, ""),
            Check("vzpěr kloubového sloupu β = 1 (Euler + Engesser)", near(lam_s), lam_s, 2e-4, ""),
            Check("statika: tlak = reakce", -res.step(0).frame["reactions"]["A"][0], -1000.0, 1e-10, "N")]


FRAME_CASES["F9"] = Case("F9", "Pruty: lineární vzpěr (Euler + Engesserova oprava smyku)", f9_build, f9_eval)


def f9m_build(wd: Path):
    """Original route: statics of ccx U1 + buckling of MYSTRAN (SOL 105, CBAR, without shear) – cross-check of the own solver."""
    st, parts = f9_build(wd)
    st.name = "F9M vzper MYSTRAN"
    st.frame.settings.solver = "ccx_u1"
    st.frame.settings.shear_deformation = False
    return st, parts


def f9m_eval(res, st):
    r2 = _again(st, lambda s: setattr(s.frame.settings, "solver", "timoshenko"))
    a, b = np.array(sorted(res.modes[-1].values)), np.array(sorted(r2.modes[-1].values))
    n = min(len(a), len(b), 2)
    return [Check(f"MYSTRAN (4 prvky CBAR) × vlastní řešič: λ{j + 1}", a[j], b[j], 1e-3, "") for j in range(n)]


FRAME_CASES["F9M"] = Case("F9M", "Pruty: vzpěr – křížová kontrola MYSTRAN (SOL 105) × vlastní řešič", f9m_build, f9m_eval)


def f3m_build(wd: Path):
    st, parts = f3_build(wd)
    st.name = "F3M portal + MYSTRAN"
    st.frame.sections["IPE300"] = SectionDef("IPE300", library="IPE 300")       # real area (CBAR computes the axial deformation)
    st.frame.sections["IPE400"] = SectionDef("IPE400", library="IPE 400")
    return st, parts


def f3m_eval(res, st):
    from fea_core.beams.mystran_frame import static_check
    with tempfile.TemporaryDirectory() as td:
        r = static_check(st, res, Path(td))
    return [Check("MYSTRAN CBAR × vlastní řešič: max. rozdíl posuvů uzlů / max. posuv", r[0]["du_rel"], 0.0, 1e-6, "")]


FRAME_CASES["F3M"] = Case("F3M", "Pruty: křížová kontrola rámu F3 v MYSTRAN (CBAR)", f3m_build, f3m_eval)


# ------------------------------------------------------------------ F10 accuracy of one element, rotated section, springs
def f10_build(wd: Path):
    """Cantilever IPE 200 L = 2 m (1 element): force in the strong and in the weak axis; the same cantilever with the section rotated by 30°;
    beam clamped into a spring (displacements held, rotation about y on the spring k_θ) and self-weight.
    """
    st = _study("F10 presnost 1 prvku")
    fr = st.frame
    fr.sections["IPE200"] = SectionDef("IPE200", library="IPE 200")
    fr.add_node("A", [0, 0, 0])
    fr.add_node("B", [2000, 0, 0])
    fr.members["m1"] = FrameMember("m1", "A", "B", "IPE200")
    fr.supports.append(FrameSupport("A"))
    st.load_cases += [LoadCase("Fz", [FrameNodeLoad("B", [0, 0, -1000.0])]),
                      LoadCase("Fy", [FrameNodeLoad("B", [0, 1000.0, 0])]),
                      LoadCase("tíha", [Gravity([0, 0, -9810.0])])]
    return st, []


def _cant_defl(F, L, I, As, EI_shear=True):
    return F * L ** 3 / (3 * E * I) + (F * L / (G * As) if EI_shear else 0.0)


def f10_eval(res, st):
    sec = S.get("IPE 200")
    L = 2000.0
    out = []
    uz = _fr(res, 0)["nodes"]["B"]
    uy = _fr(res, 1)["nodes"]["B"]
    out += [Check("silná osa: w = FL³/3EI_y + FL/(G·A_sz)", -uz[2], _cant_defl(1000, L, sec.Iy, sec.Av_z), 1e-10, "mm"),
            Check("silná osa: natočení konce θy = FL²/2EI (kladné kolem y = dolů)", uz[4], 1000 * L * L / (2 * E * sec.Iy),
                  1e-10, "rad"),
            Check("slabá osa: v = FL³/3EI_z + FL/(G·A_sy)", uy[1], _cant_defl(1000, L, sec.Iz, sec.Av_y), 1e-10, "mm")]
    # self-weight: q = ρ·A·g
    q = RHO * sec.A * 9810.0
    out.append(Check("tíha: průhyb konce qL⁴/8EI + qL²/2G·A_sz", -_fr(res, 2)["nodes"]["B"][2],
                     q * L ** 4 / (8 * E * sec.Iy) + q * L ** 2 / (2 * G * sec.Av_z), 1e-10, "mm"))
    # independence of the subdivision of the member (1 × 4 × 16 elements) and of the rotated section
    for n in (4, 16):
        r2 = _again(st, lambda s, n=n: setattr(s.frame.members["m1"], "segments", n))
        out.append(Check(f"{n} prvků: průhyb konce = 1 prvek", -_fr(r2, 0)["nodes"]["B"][2], -uz[2], 1e-10, "mm"))
        out.append(Check(f"{n} prvků: M kořen = F·L", _fr(r2, 0)["members"]["m1"]["My"][0], 1000 * L, 1e-10, "N·mm"))
    ro = math.radians(30.0)

    def rolled(s):
        s.frame.members["m1"].roll = 30.0
        s.load_cases = [LoadCase("Fz", [FrameNodeLoad("B", [0, 0, -1000.0])])]
    r3 = _again(st, rolled)
    ub = _fr(r3)["nodes"]["B"]
    # local axes of the member in x: y₀ = z × x = (0, 1, 0), z₀ = (0, 0, 1); roll φ: y' = c·y₀ + s·z₀, z' = −s·y₀ + c·z₀
    ey = np.array([0, math.cos(ro), math.sin(ro)])
    ez = np.array([0, -math.sin(ro), math.cos(ro)])
    F = np.array([0, 0, -1000.0])
    wz = _cant_defl(F @ ez, L, sec.Iy, sec.Av_z)                              # displacement in the direction z' (strong axis)
    wy = _cant_defl(F @ ey, L, sec.Iz, sec.Av_y)                              # displacement in the direction y' (weak axis)
    pred = wz * ez + wy * ey
    out += [Check("natočený průřez 30°: posuv uy", ub[1], pred[1], 1e-10, "mm"),
            Check("natočený průřez 30°: posuv uz", ub[2], pred[2], 1e-10, "mm")]
    # rotational spring at the lower end: support A only the displacements + spring rz... (about y: k_θy)
    kth = 5e9

    def spring(s):
        s.frame.supports = [FrameSupport("A", [True, True, True, True, False, True], [0, 0, 0, 0, kth, 0])]
        s.load_cases = [LoadCase("Fz", [FrameNodeLoad("B", [0, 0, -1000.0])])]
    r4 = _again(st, spring)
    f4 = _fr(r4)
    th = 1000 * L / kth                              # rotation about y: M = F·L on the spring
    out += [Check("pružina k_θy: průhyb = F L³/3EI + FL/GA + θ·L", -f4["nodes"]["B"][2],
                  _cant_defl(1000, L, sec.Iy, sec.Av_z) + th * L, 1e-9, "mm"),
            Check("pružina k_θy: reakční moment = −k·θ = F·L", abs(f4["reactions"]["A"][4]), 1000 * L, 1e-9, "N·mm")]
    return out


FRAME_CASES["F10"] = Case("F10", "Pruty: přesnost 1 prvku, natočený průřez, rotační pružina, nezávislost na dělení",
                          f10_build, f10_eval)


# ------------------------------------------------------------------ F11 spatial frame
def f11_build(wd: Path, spring: bool = True, scale: float = 0.02):
    """Spatial frame: column HEA, crossbar IPE rotated by 15° with a hinge (My) at the end, arm of a rectangle in local
    axes, column with released torsion, three truss members into one node; support by clamping and (optionally) a spring.
    """
    st = _study("F11 prostorovy ram")
    fr = st.frame
    fr.sections["IPE"] = SectionDef("IPE", library="IPE 200")
    fr.sections["HEA"] = SectionDef("HEA", library="HEA 160")
    fr.sections["R"] = SectionDef("R", shape="RECT", dims={"b": 30.0, "h": 60.0})
    fr.sections["T"] = SectionDef("T", library="CHS 48.3x3.2")
    for n, p in (("A", [0, 0, 0]), ("B", [0, 0, 3000]), ("C", [4000, 0, 3500]), ("D", [4000, 2500, 3000]),
                 ("E", [4000, 2500, 0]), ("F", [2000, 1200, 3300])):
        fr.add_node(n, p)
    fr.members["col1"] = FrameMember("col1", "A", "B", "HEA", up=[1, 0, 0])
    fr.members["beam"] = FrameMember("beam", "B", "C", "IPE", roll=15.0, releases_end=[False, False, False, False, True, False])
    fr.members["arm"] = FrameMember("arm", "C", "D", "R", up=[0, 0, 1])
    fr.members["col2"] = FrameMember("col2", "E", "D", "HEA", up=[1, 0, 0], releases_end=[False, False, False, True, False, False])
    fr.members["diag"] = FrameMember("diag", "B", "F", "T", kind="truss")
    fr.members["diag2"] = FrameMember("diag2", "F", "D", "T", kind="truss")
    fr.members["diag3"] = FrameMember("diag3", "F", "A", "T", kind="truss")
    fr.supports.append(FrameSupport("A"))
    fr.supports.append(FrameSupport("E", [True, True, True, False, False, False], [0, 0, 0, 0, 0, 5e9]) if spring
                       else FrameSupport("E"))
    k = scale
    st.load_cases.append(LoadCase("mix", [FrameNodeLoad("C", [3000 * k, -2000 * k, -8000 * k], [1e6 * k, 2e6 * k, -3e6 * k]),
                                          FrameMemberLoad("beam", [200 / 1000 * k, 0, -5.0 * k], [0, 100 / 1000 * k, -9.0 * k]),
                                          FrameMemberLoad("arm", [0, 0, -3.0 * k], None, "local"),
                                          FrameNodeLoad("F", [0, 500 * k, 0])]))
    st.load_cases.append(LoadCase("tíha", [Gravity([0, 0, -9810.0])]))
    return st, []


def _member_end_forces(fe, rec, mname, which):
    """Forces and moments ON THE MEMBER from the surroundings at the end (0 = start, −1 = end), global axes, from the diagrams of the internal forces."""
    segs = fe.member_segs[mname]
    sg = fe.segs[segs[0] if which == 0 else segs[-1]]
    m = rec["members"][mname]
    N, Vy, Vz, T, My, Mz = (m[k][which] for k in ("N", "Vy", "Vz", "T", "My", "Mz"))
    sign = -1.0 if which == 0 else 1.0
    return sign * (sg.R.T @ np.array([N, Vy, Vz])), sign * (sg.R.T @ np.array([T, My, Mz]))


def f11_eval(res, st):
    from fea_core.beams.fem import build
    fe = build(st)
    X = {n: np.array(v.xyz) for n, v in st.frame.nodes.items()}
    out = []
    for k, lc in enumerate(st.load_cases):
        rec = _fr(res, k)
        bal = {n: np.zeros(6) for n in X}
        for mn, m in st.frame.members.items():
            for which, key in ((0, m.start), (-1, m.end)):
                F, M = _member_end_forces(fe, rec, mn, which)
                bal[key][:3] -= F
                bal[key][3:] -= M
        scale = 1e-30
        for l in lc.loads:
            if isinstance(l, FrameNodeLoad):
                bal[l.node][:3] += l.force
                bal[l.node][3:] += l.moment
                scale = max(scale, float(np.abs(l.force).max()), float(np.abs(l.moment).max()) / 3000.0)
        for n, r in rec["reactions"].items():
            bal[n] += np.asarray(r)
            scale = max(scale, float(np.abs(r[:3]).max()), float(np.abs(r[3:]).max()) / 3000.0)
        resid = max(float(np.abs(v[:3]).max()) for v in bal.values())
        resm = max(float(np.abs(v[3:]).max()) for v in bal.values()) / 3000.0
        out += [Check(f"{lc.name}: rovnováha sil v uzlech z koncových sil prutů / měřítko", resid / scale, 0.0, 1e-10, ""),
                Check(f"{lc.name}: rovnováha momentů v uzlech / měřítko", resm / scale, 0.0, 1e-10, ""),
                Check(f"{lc.name}: M_y v kloubu příčle = 0", abs(rec["members"]["beam"]["My"][-1]) / (scale * 3000.0), 0.0, 1e-10, ""),
                Check(f"{lc.name}: T v kloubu sloupu 2 = 0", abs(rec["members"]["col2"]["T"][-1]) / (scale * 3000.0), 0.0, 1e-10, "")]
        du = 0.0
        for mn, m in st.frame.members.items():
            u = np.asarray(rec["members"][mn]["u"])
            du = max(du, float(np.abs(u[0] - rec["nodes"][m.start][:3]).max()),
                     float(np.abs(u[-1] - rec["nodes"][m.end][:3]).max()))
        umax = max(float(np.abs(v[:3]).max()) for v in rec["nodes"].values())
        out.append(Check(f"{lc.name}: posuv z obrazce na koncích prutů = posuv uzlů", du / umax, 0.0, 1e-9, ""))
        out.append(Check(f"{lc.name}: celková rovnováha (síly + momenty)", res.step(k).equilibrium["imbalance"], 0.0, 1e-10, ""))
    return out


FRAME_CASES["F11"] = Case("F11", "Pruty: prostorový rám (klouby, příhradové pruty, pružina) – rovnováha uzlů", f11_build, f11_eval)


def f11m_build(wd: Path):
    st, parts = f11_build(wd, spring=False)
    st.name = "F11M prostorovy ram + MYSTRAN"
    st.frame.settings.shear_deformation = False           # CBAR without shear
    st.load_cases = st.load_cases[:1]
    return st, parts


def f11m_eval(res, st):
    from fea_core.beams.mystran_frame import static_check
    with tempfile.TemporaryDirectory() as td:
        r = static_check(st, res, Path(td))
    return [Check("MYSTRAN CBAR × vlastní řešič (prostorový rám): max. rozdíl posuvů uzlů / max. posuv", r[0]["du_rel"],
                  0.0, 1e-6, "")]


FRAME_CASES["F11M"] = Case("F11M", "Pruty: křížová kontrola prostorového rámu v MYSTRAN (CBAR)", f11m_build, f11m_eval)


# ------------------------------------------------------------------ F12 frekvence a konvergence
def f12_build(wd: Path):
    """Cantilever of a tube CHS 60.3×4, L = 1 m: torsional modes f = (2k − 1)/(4L)·√(G/ρ) (J = Ip) and bending modes
    of the cantilever (Timoshenko) – convergence with the subdivision (O(h²) of the mass and the torsional stiffness).
    """
    st = _study("F12 frekvence konzoly")
    fr = st.frame
    fr.sections["CHS"] = SectionDef("CHS", library="CHS 60.3x4")
    fr.add_node("A", [0, 0, 0])
    fr.add_node("B", [1000, 0, 0])
    fr.members["m"] = FrameMember("m", "A", "B", "CHS")
    fr.supports.append(FrameSupport("A"))
    st.modal.enabled = True
    st.modal.modes = 6
    return st, []


def f12_eval(res, st):
    sec = S.get("CHS 60.3x4")
    L = 1000.0
    ft = 1 / (4 * L) * math.sqrt(G * sec.J / (RHO * (sec.Iy + sec.Iz)))
    out = []
    errs = []
    for n in (2, 4, 8, 16):
        r = _again(st, lambda s, n=n: setattr(s.frame.settings, "segments_modal", n))
        m = r.modes[0]
        tors = [f for f, em in zip(m.values, m.eff_mass) if sum(em[:3]) < 1e-6 * max(m.total_mass[:3])]
        errs.append(tors[0] / ft - 1)
    out += [Check("torze 1. mód, 4 prvky (O(h²))", ft * (1 + errs[1]), ft, 8e-3, "Hz"),
            Check("torze 1. mód, 8 prvků (O(h²))", ft * (1 + errs[2]), ft, 2.2e-3, "Hz"),
            Check("torze 1. mód, 16 prvků", ft * (1 + errs[3]), ft, 6e-4, "Hz"),
            Check("řád konvergence torze (poměr chyb 8 → 16 prvků = 4)", errs[2] / errs[3], 4.0, 0.05, "")]
    # bending mode of the cantilever: check against a fine subdivision and Euler–Bernoulli with a shear correction (Timoshenko, 64 elements)
    ref = _again(st, lambda s: setattr(s.frame.settings, "segments_modal", 64)).modes[0].values
    eb = 1.8751040687 ** 2 / (2 * math.pi * L * L) * math.sqrt(E * sec.Iy / (RHO * sec.A))
    out += [Check("ohyb 1. mód konzoly, 8 prvků = 64 prvků", min(res.modes[0].values, key=lambda v: abs(v - eb)),
                  min(ref, key=lambda v: abs(v - eb)), 1e-4, "Hz"),
            Check("ohyb 1. mód konzoly = Euler–Bernoulli (štíhlý prut, malý smyk)", min(ref, key=lambda v: abs(v - eb)), eb,
                  8e-3, "Hz")]
    return out


FRAME_CASES["F12"] = Case("F12", "Pruty: torzní a ohybové frekvence konzoly, konvergence", f12_build, f12_eval)


# ------------------------------------------------------------------ F13 buckling
def f13_build(wd: Path):
    st = _study("F13 vzper sloupu")
    fr = st.frame
    fr.sections["IPE200"] = SectionDef("IPE200", library="IPE 200")
    fr.add_node("A", [0, 0, 0])
    fr.add_node("B", [3000, 0, 0])
    fr.members["m"] = FrameMember("m", "A", "B", "IPE200")
    fr.supports += [FrameSupport("A", [True, True, True, True, False, False]),
                    FrameSupport("B", [False, True, True, False, False, False])]
    st.load_cases.append(LoadCase("P", [FrameNodeLoad("B", [-1000.0, 0, 0])]))
    st.buckling.enabled = True
    st.buckling.modes = 3
    return st, []


def f13_eval(res, st):
    sec = S.get("IPE 200")
    L = 3000.0
    EI = E * sec.Iz
    out = []

    def lam(support, nseg, shear):
        def mod(s):
            s.frame.settings.segments_modal = nseg
            s.frame.settings.shear_deformation = shear
            if support == "cant":
                s.frame.supports = [FrameSupport("A")]
            elif support == "pf":
                s.frame.supports = [FrameSupport("A"), FrameSupport("B", [False, True, True, False, False, False])]
        return _again(st, mod).modes[-1].values[0]
    pe = lambda beta: math.pi ** 2 * EI / (beta * L) ** 2 / 1000          # noqa: E731 – λ for F = 1 kN
    out += [Check("kloubový sloup, E–B, 4 prvky", lam("pin", 4, False), pe(1.0), 1e-3, ""),
            Check("kloubový sloup, E–B, 8 prvků", lam("pin", 8, False), pe(1.0), 1e-4, ""),
            Check("konzola, E–B, 2 prvky", lam("cant", 2, False), pe(2.0), 1e-3, ""),
            Check("konzola, E–B, 4 prvky", lam("cant", 4, False), pe(2.0), 1e-4, ""),
            Check("kloub–vetknutí, E–B, 8 prvků (λ_kr = 20,1907·EI/L²)", lam("pf", 8, False),
                  20.190729 * EI / L ** 2 / 1000, 5e-4, "")]
    for sup, beta in (("pin", 1.0), ("cant", 2.0)):
        p = pe(beta) * 1000
        eng = p / (1 + p / (G * sec.Av_y)) / 1000
        out.append(Check(f"{'kloubový sloup' if sup == 'pin' else 'konzola'} s posunem: Engesser P_E/(1 + P_E/GA_s), 8 prvků",
                         lam(sup, 8, True), eng, 2e-4, ""))
    return out


FRAME_CASES["F13"] = Case("F13", "Pruty: lineární vzpěr – Euler (2–8 prvků) a Engesser", f13_build, f13_eval)


# ------------------------------------------------------------------ F14 teplota
def f14_build(wd: Path):
    """Bar IPE 200 L = 2 m clamped at both ends: ΔT = 50 K (N = −EAαΔT); gradient 30 K across the height (M = −EIαΔT_g/h);
    the same for a simply supported beam (deflection αΔT_g L²/(8h)) and a cantilever (elongation αΔT·L).
    """
    st = _study("F14 teplota prutu")
    fr = st.frame
    fr.sections["S"] = SectionDef("S", library="IPE 200")
    fr.add_node("A", [0, 0, 0])
    fr.add_node("B", [2000, 0, 0])
    fr.members["m"] = FrameMember("m", "A", "B", "S")
    fr.supports += [FrameSupport("A"), FrameSupport("B")]
    st.load_cases += [LoadCase("dT", [FrameTemperature([], 50.0)]),
                      LoadCase("gradient", [FrameTemperature(["m"], 0.0, 30.0)])]
    return st, []


def f14_eval(res, st):
    sec = S.get("IPE 200")
    L, dT, g = 2000.0, 50.0, 30.0
    hz = float(np.ptp(np.array(sec.fibers)[:, 1]))
    ea = E * sec.A * ALPHA_T * dT
    r0, r1 = _fr(res, 0), _fr(res, 1)
    out = [Check("vetknutá tyč: N = −E·A·α·ΔT", r0["members"]["m"]["N"][0], -ea, 1e-12, "N"),
           Check("vetknutá tyč: reakce = E·A·α·ΔT", r0["reactions"]["A"][0], ea, 1e-12, "N"),
           Check("vetknutá tyč: posuv uzlu B = 0 (abs.)", abs(r0["nodes"]["B"][0]), 0.0, 1e-12, "mm"),
           Check("vetknutý nosník: M_y = −E·I·α·ΔT_g/h", r1["members"]["m"]["My"][0], -E * sec.Iy * ALPHA_T * g / hz, 1e-12, "N·mm")]
    free = _again(st, lambda s: setattr(s.frame, "supports",
                                        [FrameSupport("A"), FrameSupport("B", [False, True, True, True, True, True])]))
    out.append(Check("volná tyč: prodloužení α·ΔT·L", _fr(free, 0)["nodes"]["B"][0], ALPHA_T * dT * L, 1e-12, "mm"))
    out.append(Check("volná tyč: N = 0 (abs.)", abs(_fr(free, 0)["members"]["m"]["N"][0]), 0.0, 1e-8, "N"))

    def ss(s):
        s.frame.supports = [FrameSupport("A", [True, True, True, True, False, False]),
                            FrameSupport("B", [False, True, True, False, False, False])]
        s.frame.members["m"].segments = 4
    r = _again(st, ss)
    m = _fr(r, 1)["members"]["m"]
    i = int(np.argmin(np.abs(np.asarray(m["s"]) - L / 2)))
    out.append(Check("prostý nosník s gradientem: průhyb uprostřed α·ΔT_g·L²/(8h)", m["u"][i][2], ALPHA_T * g / hz * L * L / 8,
                     1e-9, "mm"))
    out.append(Check("prostý nosník s gradientem: M = 0 (abs.)", float(np.abs(m["My"]).max()), 0.0, 1e-6, "N·mm"))
    return out


FRAME_CASES["F14"] = Case("F14", "Pruty: teplota – bráněná roztažnost, gradient, volná tyč", f14_build, f14_eval)


def f11a_build(wd: Path):
    """F11 without the rotational spring (variant for an external cross-check that has no rotational spring)."""
    st, parts = f11_build(wd, spring=False)
    st.name = "F11A prostorovy ram bez pruziny"
    return st, parts


FRAME_CASES["F11A"] = Case("F11A", "Pruty: prostorový rám bez pružiny (varianta pro křížovou kontrolu)", f11a_build, f11_eval)
