"""Verification cases of heat (T1–T7): heat conduction in ccx and thermal stress, comparison with the analytic solution.

T1  bar 100×10×10 from two parts (steel 0–50, aluminium 50–100), sides insulated – series resistances:
    (a) T = 100 °C at x = 0, convection h = 500 W/(m²·K), 20 °C at x = 100;
    (b) T = 20 °C at x = 0, flux 10 000 W/m² into the face x = 100; (c) the same as a power 1 W.
T2  quarter of a tube ri 20 / ro 40 / h 10: T(ri) = 100, T(ro) = 20 → T(r) logarithmic, Q = 2πkh·ΔT/ln(ro/ri)/4.
T3  bar 100×10×10, both ends 20 °C, source 10 W in the volume → parabola, T_max = T0 + q'''L²/(8k), ends 5 W each.
T4  bar 100×10×10, ΔT = 100 K uniformly: (a) free expansion (symmetry) u = αΔT·L, σ = 0;
    (b) both ends ux = 0 → σx = −EαΔT, reaction = EαΔT·A.
T5  thermal stress from the field T2 (steady state of ccx → statics), plane strain (uz = 0 on both faces, symmetry):
    σr, σθ, u_r according to Timoshenko & Goodier (integrals of T·r numerically), σz = ν(σr + σθ) − EαΔT.
T6  transient conduction: bar 100 mm, start 20 °C, at x = 0 a jump to 120 °C, others insulated – Fourier series.
T7  radiation: bar k = 5 W/(m·K), x = 0 at 300 °C, the face x = 100 radiates (ε = 0.8) to 20 °C – the root k(T0 − T_L)/L = εσ(T_L⁴ − T∞⁴).
TH8 two plates in series with a contact pair conductance h_c (series resistances), TH8B the same as a bonded interface
    (ThermalSettings.conductances, one part meshed separately) + tension through the kept mechanical *TIE;
TH9 / TH9C radiation in a closed cavity between two plates (Radiation cavity=True / CavityRadiation, ε 0.8 / 0.5),
    TH9C also checks the heat of each cavity face.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut

from fea_core.geometry import Part
from fea_core.results import Results
from fea_core.selectors import find_faces
from fea_core.study import Study, Material, Displacement, LoadCase, MeshSettings, ThermalLoad
from fea_core.thermal.model import (ThermalCase, FixedTemperature, HeatFlux, HeatFlow, Convection, Radiation,
                                    HeatGeneration, STEFAN_BOLTZMANN, ABSOLUTE_ZERO)
from verification.cases import Case, Check, E, NU, box, cyl, solid, _study

K_ST, K_AL = 50.0, 170.0
ALPHA = 12e-6
STEEL_T = Material("Ocel (teplo)", E, NU, 7.85e-9, Re=235.0, Rm=360.0, k=K_ST, alpha=ALPHA, cp=460.0,
                   source="ověřovací – kulatá čísla")
AL_T = Material("Hliník (teplo)", 70000.0, 0.33, 2.7e-9, k=K_AL, alpha=23e-6, cp=900.0, source="ověřovací")


def _bar_study(name: str, wd: Path, parts: list[Part], size: float = 4.0) -> tuple[Study, list]:
    st, parts = _study(name, wd / "model.step", parts, MeshSettings(size=size))
    st.materials.clear()
    st.default_material = None
    st.add_material(STEEL_T, default=True)
    for s, kw in {"x0": dict(x=0), "x100": dict(x=100), "y0": dict(y=0), "z0": dict(z=0)}.items():
        st.add_selection(s, find_faces(parts, **kw))
    return st, parts


def _at_x(res: Results, x: float, case: int | str = 0) -> float:
    n = res.mesh.nodes
    idx = np.nonzero(np.abs(n[:, 0] - x) < 1e-6)[0]
    return float(res.thermal_case(case).T[idx].mean())


def _q(res: Results, case, kind: str) -> float:
    return sum(h["Q_W"] for h in res.thermal_case(case).heat if h["kind"] == kind)


# ------------------------------------------------------------------ T1
T1_H, T1_Q = 500.0, 10000.0
A1 = 100e-6              # m²
R_ST, R_AL = 0.05 / K_ST, 0.05 / K_AL


def t1_build(wd: Path):
    st, parts = _bar_study("T1 tyc ocel-hlinik", wd, [Part("Ocel", box(0, 0, 0, 50, 10, 10)),
                                                    Part("Hlinik", box(50, 0, 0, 50, 10, 10))])
    st.add_material(AL_T)
    st.part_materials["Hlinik"] = AL_T.name
    st.thermal.cases = [
        ThermalCase("a) 100 °C + konvekce", [FixedTemperature("x0", 100.0), Convection("x100", T1_H, 20.0)]),
        ThermalCase("b) tok 10 kW/m²", [FixedTemperature("x0", 20.0), HeatFlux("x100", T1_Q)]),
        ThermalCase("c) výkon 1 W", [FixedTemperature("x0", 20.0), HeatFlow("x100", T1_Q * A1)]),
    ]
    return st, parts


def t1_eval(res: Results, st: Study):
    q = (100.0 - 20.0) / (R_ST + R_AL + 1 / T1_H)          # W/m²
    th = res.thermal_case(0)
    return [
        Check("(a) T rozhraní x = 50", _at_x(res, 50, 0), 100 - q * R_ST, 1e-4, "°C"),
        Check("(a) T čela x = 100", _at_x(res, 100, 0), 100 - q * (R_ST + R_AL), 1e-4, "°C"),
        Check("(a) teplo předepsané teploty", _q(res, 0, "fixed_temperature"), q * A1, 1e-3, "W"),
        Check("(a) teplo konvekcí", -_q(res, 0, "convection"), q * A1, 1e-3, "W"),
        Check("(a) tok q_x v oceli", float(th.flux[np.argmin(np.abs(res.mesh.nodes[:, 0] - 25)), 0]), q, 1e-3, "W/m²"),
        Check("(b) T čela x = 100", _at_x(res, 100, 1), 20 + T1_Q * (R_ST + R_AL), 1e-4, "°C"),
        Check("(c) T čela x = 100 (výkon 1 W)", _at_x(res, 100, 2), 20 + T1_Q * (R_ST + R_AL), 1e-4, "°C"),
        Check("(c) teplo předepsané teploty", -_q(res, 2, "fixed_temperature"), 1.0, 1e-3, "W"),
    ]


# ------------------------------------------------------------------ T2 / T5
RI, RO, H2 = 20.0, 40.0, 10.0
TI, TO = 100.0, 20.0


def _quarter_pipe():
    ring = BRepAlgoAPI_Cut(cyl((0, 0, 0), (0, 0, 1), RO, H2), cyl((0, 0, -1), (0, 0, 1), RI, H2 + 2)).Shape()
    return solid(BRepAlgoAPI_Common(ring, box(0, 0, -1, RO + 1, RO + 1, H2 + 2)).Shape())


def t2_build(wd: Path, stress: bool = False):
    st, parts = _study("T5 trubka teplotni napeti" if stress else "T2 trubka vedeni", wd / "model.step",
                       [Part("Trubka", _quarter_pipe())], MeshSettings(size=2.5))
    st.materials.clear()
    st.default_material = None
    st.add_material(STEEL_T, default=True)
    st.add_selection("vnitrni", find_faces(parts, radius=RI))
    st.add_selection("vnejsi", find_faces(parts, radius=RO))
    st.thermal.cases = [ThermalCase("Radiální vedení", [FixedTemperature("vnitrni", TI), FixedTemperature("vnejsi", TO)])]
    if stress:
        st.thermal.reference_temperature = TO
        st.add_selection("sym_x", find_faces(parts, x=0))
        st.add_selection("sym_y", find_faces(parts, y=0))
        st.add_selection("cela", find_faces(parts, z=0) + find_faces(parts, z=H2))
        st.constraints += [Displacement("sym_x", ux=0.0), Displacement("sym_y", uy=0.0), Displacement("cela", uz=0.0)]
        st.load_cases.append(LoadCase("Teplotní napětí", [ThermalLoad(case="Radiální vedení")]))
    return st, parts


def _t_exact(r):
    return TO + (TI - TO) * np.log(RO / r) / math.log(RO / RI)


def t2_eval(res: Results, st: Study):
    th = res.thermal_case(0)
    phi = math.radians(30)
    T30 = float(res.probe([30 * math.cos(phi), 30 * math.sin(phi), 5.0], "T", 0)[0])
    Q = 2 * math.pi * K_ST * (H2 * 1e-3) * (TI - TO) / math.log(RO / RI) / 4
    p = np.array([30 * math.cos(phi), 30 * math.sin(phi), 5.0])
    qr = float(res.probe(p, "HFL", 0) @ np.array([math.cos(phi), math.sin(phi), 0.0]))
    return [
        Check("T(r = 30)", T30, float(_t_exact(30.0)), 1e-3, "°C"),
        Check("radiální tok q_r(30) = k·ΔT/(r·ln(ro/ri))", qr, K_ST * (TI - TO) / (0.030 * math.log(RO / RI)), 0.01, "W/m²"),
        Check("teplo vnitřní plochou", next(h["Q_W"] for h in th.heat if h["name"] == "vnitrni"), Q, 2e-3, "W"),
        Check("tepelná bilance (1 + zbytek)", 1 + th.balance["imbalance"], 1.0, 1e-4, ""),
    ]


def _timoshenko(r: float) -> tuple[float, float, float, float]:
    """Plane strain of a hollow cylinder, T* = T − T_ref: σr, σθ, σz, u_r (Timoshenko & Goodier, §150)."""
    from scipy.integrate import quad
    a, b = RI, RO
    Tst = lambda x: (_t_exact(x) - TO) * x
    Iab = quad(Tst, a, b)[0]
    Iar = quad(Tst, a, r)[0]
    c = ALPHA * E / (1 - NU)
    sr = c / r ** 2 * ((r ** 2 - a ** 2) / (b ** 2 - a ** 2) * Iab - Iar)
    st = c / r ** 2 * ((r ** 2 + a ** 2) / (b ** 2 - a ** 2) * Iab + Iar - (_t_exact(r) - TO) * r ** 2)
    sz = NU * (sr + st) - ALPHA * E * (_t_exact(r) - TO)
    u = (1 + NU) / (1 - NU) * ALPHA / r * (Iar + ((1 - 2 * NU) * r ** 2 + a ** 2) / (b ** 2 - a ** 2) * Iab)
    return sr, st, sz, u


def t5_eval(res: Results, st: Study):
    from fea_core.results import tensor
    phi = math.radians(45)
    er = np.array([math.cos(phi), math.sin(phi), 0.0])
    et = np.array([-math.sin(phi), math.cos(phi), 0.0])
    out = []
    for r in (RI, 30.0, RO):
        p = [r * math.cos(phi), r * math.sin(phi), 5.0]
        S = tensor(res.probe(p, "S"))
        U = res.probe(p, "U")
        sr, sth, sz, u = _timoshenko(r)
        out.append(Check(f"σθ(r = {r:g})", float(et @ S @ et), sth, 0.03, "MPa"))
        if r == 30.0:
            out.append(Check("σr(r = 30)", float(er @ S @ er), sr, 0.05, "MPa"))
            out.append(Check("u_r(r = 30)", float(U @ er), u, 0.01, "mm"))
        if r == RI:
            out.append(Check(f"σz(r = {r:g})", float(S[2, 2]), sz, 0.03, "MPa"))
    s = res.step(0)
    out.append(Check("rovnováha (1 + nerovnováha)", 1 + s.equilibrium["imbalance"], 1.0, 1e-3, ""))
    return out


# ------------------------------------------------------------------ T3
P3 = 10.0      # W


def t3_build(wd: Path):
    st, parts = _bar_study("T3 tyc zdroj tepla", wd, [Part("Tyc", box(0, 0, 0, 100, 10, 10))])
    st.thermal.cases = [ThermalCase("Zdroj 10 W", [FixedTemperature("x0", 20.0), FixedTemperature("x100", 20.0),
                                                  HeatGeneration("Tyc", P3)])]
    return st, parts


def t3_eval(res: Results, st: Study):
    qv = P3 / 1e-5                                   # W/m³
    tmax = 20 + qv * 0.1 ** 2 / (8 * K_ST)
    th = res.thermal_case(0)
    ends = [h["Q_W"] for h in th.heat if h["kind"] == "fixed_temperature"]
    return [
        Check("T uprostřed x = 50", _at_x(res, 50), tmax, 1e-4, "°C"),
        Check("T x = 25", _at_x(res, 25), 20 + qv / (2 * K_ST) * 0.025 * 0.075, 1e-4, "°C"),
        Check("teplo koncem x = 0", -ends[0], P3 / 2, 1e-3, "W"),
        Check("teplo koncem x = 100", -ends[1], P3 / 2, 1e-3, "W"),
        Check("zdroj (integrace)", _q(res, 0, "heat_generation"), P3, 1e-9, "W"),
    ]


# ------------------------------------------------------------------ T4
DT4 = 100.0


def t4_build(wd: Path):
    st, parts = _bar_study("T4 tyc roztaznost", wd, [Part("Tyc", box(0, 0, 0, 100, 10, 10))])
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0), Displacement("z0", uz=0.0)]
    # state (b) adds the displacement ux = 0 at x = 100 only through the second study – here the superposition is verified:
    st.load_cases = [LoadCase("ΔT = 100 K", [ThermalLoad(value=20.0 + DT4)]),
                     LoadCase("Bez teploty", [ThermalLoad(value=20.0)])]
    return st, parts


def t4_eval(res: Results, st: Study):
    from verification.cases import Check as C
    n = res.mesh.nodes
    end = np.abs(n[:, 0] - 100) < 1e-6
    s0 = res.step(0)
    out = [
        C("(a) volná roztažnost ux(L) = αΔT·L", float(s0.U[end, 0].mean()), ALPHA * DT4 * 100, 1e-5, "mm"),
        C("(a) bez napětí: 1 + max σ_vM/(EαΔT)", 1 + float(s0.mises.max()) / (E * ALPHA * DT4), 1.0, 1e-5, ""),
        C("(a) teplota uložena do výsledku", float(s0.fields["T"].max()), 20 + DT4, 1e-12, "°C"),
        C("(a) stav bez ΔT: 1 + max |u|", 1 + float(res.step(1).umag.max()), 1.0, 1e-12, "mm"),
    ]
    # (b) restrained expansion – a separate calculation
    from fea_core.solve import run_study
    st2 = Study.from_dict(st.to_dict())
    st2.constraints.append(Displacement("x100", ux=0.0))
    st2.load_cases = [LoadCase("ΔT bráněná", [ThermalLoad(value=20.0 + DT4)])]
    r2 = run_study(st2, Path(__file__).resolve().parent / "out" / "T4" / "b", mesh=res.mesh)
    s2 = r2.step(0)
    sx = float(r2.probe([50, 5, 5], "S")[0])
    out += [
        C("(b) bráněná: σx = −EαΔT", sx, -E * ALPHA * DT4, 1e-4, "MPa"),
        C("(b) reakce = EαΔT·A", float(s2.reactions["x100"][0]), -E * ALPHA * DT4 * 100, 1e-4, "N"),
    ]
    return out


# ------------------------------------------------------------------ T6
T6_TIME, T6_DT = 600.0, 4.0


def t6_build(wd: Path):
    st, parts = _bar_study("T6 prechodove vedeni", wd, [Part("Tyc", box(0, 0, 0, 100, 10, 10))])
    st.thermal.initial_temperature = 20.0
    st.thermal.cases = [ThermalCase("Skok 120 °C", [FixedTemperature("x0", 120.0)], transient=True,
                                    time=T6_TIME, dt=T6_DT, outputs=6)]
    return st, parts


def _t6_exact(x: float, t: float) -> float:
    kap = K_ST / (STEEL_T.rho * 1e12 * STEEL_T.cp) * 1e6       # mm²/s (ρ v kg/m³)
    L = 100.0
    s = sum(4 / ((2 * n + 1) * math.pi) * math.sin((2 * n + 1) * math.pi * x / (2 * L))
            * math.exp(-((2 * n + 1) * math.pi / (2 * L)) ** 2 * kap * t) for n in range(400))
    return 120.0 + (20.0 - 120.0) * s


def t6_eval(res: Results, st: Study):
    th = res.thermal_case(0)
    n = res.mesh.nodes
    out = [Check("počet snímků (+ počáteční t = 0)", len(th.times), 7, 0, ""),
           Check("snímek t = 0 = počáteční teplota", float(th.at_time(0.0).max()), 20.0, 1e-12, "°C"),
           Check("t = 50 s: interpolace mezi t = 0 a 1. snímkem", float(th.at_time(50.0).max()),
                 float(0.5 * (20.0 + th.frames[1].max())), 1e-12, "°C"),
           Check("čas posledního snímku", th.times[-1], T6_TIME, 1e-9, "s")]
    for t in (300.0, 600.0):
        T = th.at_time(t)
        for x in (50.0, 100.0):
            idx = np.abs(n[:, 0] - x) < 1e-6
            out.append(Check(f"ΔT(x = {x:g}, t = {t:g} s)", float(T[idx].mean()) - 20.0, _t6_exact(x, t) - 20.0,
                             0.01, "K"))
    return out


# ------------------------------------------------------------------ T7
K7, T70, EPS7 = 5.0, 300.0, 0.8


def t7_build(wd: Path):
    st, parts = _bar_study("T7 salani", wd, [Part("Tyc", box(0, 0, 0, 100, 10, 10))])
    m = Material("Ocel k = 5 (teplo)", E, NU, 7.85e-9, k=K7, alpha=ALPHA, cp=460.0, source="ověřovací")
    st.materials.clear()
    st.add_material(m, default=True)
    st.thermal.cases = [ThermalCase("Sálání", [FixedTemperature("x0", T70), Radiation("x100", EPS7, 20.0)])]
    return st, parts


def _t7_exact() -> float:
    from scipy.optimize import brentq
    z = ABSOLUTE_ZERO
    f = lambda T: K7 / 0.1 * (T70 - T) - EPS7 * STEFAN_BOLTZMANN * ((T - z) ** 4 - (20.0 - z) ** 4)
    return brentq(f, 20.0, T70)


def t7_eval(res: Results, st: Study):
    TL = _t7_exact()
    th = res.thermal_case(0)
    return [
        Check("T čela (sálání)", _at_x(res, 100), TL, 1e-4, "°C"),
        Check("teplo sáláním", -_q(res, 0, "radiation"), K7 / 0.1 * (T70 - TL) * A1, 2e-3, "W"),
        Check("tepelná bilance (1 + zbytek)", 1 + th.balance["imbalance"], 1.0, 1e-3, ""),
    ]


# ------------------------------------------------------------------ TH8
HC8 = 1000.0  # W/(m²·K) contact conductance of the joint
A8 = 100e-6   # m² (10x10 mm²)


def th8_build(wd: Path):
    p1 = Part("Platna1", box(0, 0, 0, 50, 10, 10))
    p2 = Part("Platna2", box(50, 0, 0, 50, 10, 10))
    st, parts = _bar_study("TH8 odpor styku", wd, [p1, p2], size=4.0)
    st.materials.clear()
    st.add_material(STEEL_T, default=True)
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("x100", find_faces(parts, x=100))
    st.add_selection("x50_a", find_faces([parts[0]], x=50))
    st.add_selection("x50_b", find_faces([parts[1]], x=50))
    from fea_core.contacts.model import ContactPair
    st.contacts = [ContactPair("Styk_desek", "x50_a", "x50_b", conductance=HC8)]
    st.thermal.cases = [
        ThermalCase("Odpor styku", [FixedTemperature("x0", 100.0), FixedTemperature("x100", 20.0)])
    ]
    return st, parts


def th8_eval(res: Results, st: Study):
    # Analytic solution of the series resistances:
    # R1 = 0.05 / (50 · 1e-4) = 10 K/W
    # Rc = 1 / (1000 · 1e-4) = 10 K/W
    # R2 = 0.05 / (50 · 1e-4) = 10 K/W
    # R_tot = 30 K/W
    # Q = (100 - 20) / 30 = 80 / 30 = 2.66667 W
    # q = Q / A = 26666.67 W/m²
    # ΔT_gap = q / hc = 26.6667 K
    # T(50-) = 100 - Q·R1 = 73.3333 °C
    # T(50+) = 20 + Q·R2 = 46.6667 °C
    R1 = 0.05 / (K_ST * A8)
    Rc = 1.0 / (HC8 * A8)
    R2 = 0.05 / (K_ST * A8)
    R_tot = R1 + Rc + R2
    Q_exact = (100.0 - 20.0) / R_tot
    q_exact = Q_exact / A8
    dT_gap_exact = q_exact / HC8
    T_a_exact = 100.0 - Q_exact * R1
    T_b_exact = 20.0 + Q_exact * R2

    th = res.thermal_case(0)
    el1_nodes = res.mesh.surface_nodes("x50_a")
    el2_nodes = res.mesh.surface_nodes("x50_b")
    T_a = float(th.T[el1_nodes].mean())
    T_b = float(th.T[el2_nodes].mean())
    dT_num = T_a - T_b

    ends = {h["name"]: h["Q_W"] for h in th.heat if h["kind"] == "fixed_temperature"}
    ifc = (th.balance.get("interfaces") or [{}])[0]

    return [
        Check("heat through the interface Q = h_c·∫ΔT dA (own integration)", abs(ifc.get("Q_W", 0.0)), Q_exact, 1e-3, "W"),
        Check("T rozhraní x = 50- (slave)", T_a, T_a_exact, 1e-4, "°C"),
        Check("T rozhraní x = 50+ (master)", T_b, T_b_exact, 1e-4, "°C"),
        Check("ΔT styku (q / h_c)", dT_num, dT_gap_exact, 1e-3, "K"),
        Check("teplo vstupující na x = 0", ends.get("x0", 0.0), Q_exact, 1e-3, "W"),
        Check("teplo odvedené na x = 100", -ends.get("x100", 0.0), Q_exact, 1e-3, "W"),
        Check("tepelná bilance (1 + zbytek)", 1 + th.balance["imbalance"], 1.0, 1e-3, ""),
    ]


# ------------------------------------------------------------------ TH9
EPS9 = 0.8
A9 = 400e-6  # m² (20x20 mm²)
T9_HOT, T9_COLD = 200.0, 20.0


def th9_build(wd: Path):
    p1 = Part("Platna1", box(0, 0, 0, 10, 20, 20))
    p2 = Part("Platna2", box(12, 0, 0, 10, 20, 20))
    st, parts = _study("TH9 salani mezi deskami", wd / "model.step", [p1, p2], MeshSettings(size=6.0))
    high_k = Material("VysokaVodivost", E, NU, 7.85e-9, k=10000.0, cp=460.0, source="ověřovací")
    st.materials.clear()
    st.add_material(high_k, default=True)

    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("x22", find_faces(parts, x=22))
    st.add_selection("cav_hot", find_faces([parts[0]], x=10))
    st.add_selection("cav_cold", find_faces([parts[1]], x=12))

    st.thermal.cases = [
        ThermalCase("Dutinové sálání", [
            FixedTemperature("x0", T9_HOT),
            FixedTemperature("x22", T9_COLD),
            Radiation("cav_hot", emissivity=EPS9, cavity=True, cavity_name="CAV", closed=True),
            Radiation("cav_cold", emissivity=EPS9, cavity=True, cavity_name="CAV", closed=True),
        ])
    ]
    return st, parts


def th9_eval(res: Results, st: Study):
    # Analytic radiation between two parallel plates:
    # q = ε_eff · σ · (T1⁴ - T2⁴)
    # where ε_eff = 1 / (1/ε1 + 1/ε2 - 1)
    T1_k = T9_HOT - ABSOLUTE_ZERO
    T2_k = T9_COLD - ABSOLUTE_ZERO
    eps_eff = 1.0 / (1.0 / EPS9 + 1.0 / EPS9 - 1.0)
    q_exact = eps_eff * STEFAN_BOLTZMANN * (T1_k ** 4 - T2_k ** 4)
    Q_exact = q_exact * A9

    th = res.thermal_case(0)
    ends = {h["name"]: h["Q_W"] for h in th.heat if h["kind"] == "fixed_temperature"}
    q_in = ends.get("x0", 0.0)
    q_out = -ends.get("x22", 0.0)

    return [
        Check("tepelný tok sáláním Q", q_in, Q_exact, 1e-3, "W"),
        Check("teplo odvedené deskou 2", q_out, Q_exact, 1e-3, "W"),
        Check("tepelná bilance (1 + zbytek)", 1 + th.balance["imbalance"], 1.0, 1e-3, ""),
    ]


# ------------------------------------------------------------------ TH8B
F8B = 1000.0   # N tension of the bar (the mechanical *TIE of the separately meshed part must stay)


def th8b_build(wd: Path):
    """TH8 as a bonded interface of two parts (no contact pair): ThermalSettings.conductances with the part names,
    one part meshed separately (the Heat step / fea_thermal_add_contact_conductance do the same) + a tension load case."""
    from fea_core.thermal.model import ContactConductance
    from fea_core.study import Force
    p1 = Part("Platna1", box(0, 0, 0, 50, 10, 10))
    p2 = Part("Platna2", box(50, 0, 0, 50, 10, 10))
    st, parts = _bar_study("TH8B bonded interface conductance", wd, [p1, p2], size=4.0)
    st.mesh.separate = ["Platna2"]
    st.thermal.conductances = [ContactConductance("Platna1", "Platna2", conductance=HC8, name="Plate joint")]
    st.thermal.cases = [ThermalCase("Contact resistance", [FixedTemperature("x0", 100.0), FixedTemperature("x100", 20.0)])]
    # symmetry planes only on part 1 – the *TIE slave (separate part 2) must not have supported nodes (ties.detect leaves
    # such faces out); part 2 is held by the *TIE alone → uniaxial tension in both parts
    st.add_selection("y0a", find_faces([parts[0]], y=0))
    st.add_selection("z0a", find_faces([parts[0]], z=0))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0a", uy=0.0), Displacement("z0a", uz=0.0)]
    st.load_cases = [LoadCase("Tension", [Force("x100", [F8B, 0.0, 0.0])])]
    return st, parts


def _interface_T(res: Results, part: str) -> float:
    m = res.mesh
    pi = m.part_names.index(part)
    own = np.unique(m.tets[m.tet_part == pi])
    idx = own[np.abs(m.nodes[own, 0] - 50.0) < 1e-6]
    return float(res.thermal_case(0).T[idx].mean())


def th8b_eval(res: Results, st: Study):
    R = 0.05 / (K_ST * A8)
    Q = 80.0 / (2 * R + 1.0 / (HC8 * A8))
    th = res.thermal_case(0)
    ends = {h["name"]: h["Q_W"] for h in th.heat if h["kind"] == "fixed_temperature"}
    ifc = (th.balance.get("interfaces") or [{}])[0]
    n = res.mesh.nodes
    end = np.abs(n[:, 0] - 100) < 1e-6
    u = float(res.step(0).U[end, 0].mean())
    return [
        Check("T at x = 50 in part 1", _interface_T(res, "Platna1"), 100.0 - Q * R, 1e-4, "°C"),
        Check("T at x = 50 in part 2", _interface_T(res, "Platna2"), 20.0 + Q * R, 1e-4, "°C"),
        Check("heat in at x = 0", ends.get("x0", 0.0), Q, 1e-3, "W"),
        Check(f"heat through the interface {ifc.get('from')} → {ifc.get('to')} (own integration)",
              ifc.get("Q_W", 0.0) if ifc.get("from") == "Platna1" else 0.0, Q, 1e-3, "W"),
        Check("ΔT of the interface = q / h_c", ifc.get("dT_mean_K", 0.0), Q / A8 / HC8, 1e-3, "K"),
        Check("heat balance (1 + residual)", 1 + th.balance["imbalance"], 1.0, 1e-3, ""),
        Check("mechanics: u(L) = F·L/(E·A) (*TIE kept)", u, F8B * 100.0 / (E * 100.0), 1e-4, "mm"),
    ]


# ------------------------------------------------------------------ TH9C
EPS9H, EPS9C = 0.8, 0.5


def th9c_build(wd: Path):
    """TH9 with the CavityRadiation condition (the Heat step / fea_thermal_add_cavity) and a different ε per face."""
    from fea_core.thermal.model import CavityRadiation
    st, parts = th9_build(wd)
    st.name = "TH9C cavity two emissivities"
    case = st.thermal.cases[0]
    case.loads = [FixedTemperature("x0", T9_HOT), FixedTemperature("x22", T9_COLD),
                  CavityRadiation("cav_hot", emissivity=EPS9H, cavity_name="Gap", closed=True),
                  CavityRadiation("cav_cold", emissivity=EPS9C, cavity_name="Gap", closed=True)]
    return st, parts


def th9c_eval(res: Results, st: Study):
    eps = 1.0 / (1.0 / EPS9H + 1.0 / EPS9C - 1.0)
    Q = eps * STEFAN_BOLTZMANN * ((T9_HOT - ABSOLUTE_ZERO) ** 4 - (T9_COLD - ABSOLUTE_ZERO) ** 4) * A9
    th = res.thermal_case(0)
    ends = {h["name"]: h["Q_W"] for h in th.heat if h["kind"] == "fixed_temperature"}
    faces = {f["selection"]: f["Q_W"] for f in th.balance.get("cavity_faces", [])}
    return [
        Check("radiated heat Q (ε 0.8 / 0.5)", ends.get("x0", 0.0), Q, 1e-3, "W"),
        Check("cavity face 'cav_hot' gives off Q", -faces.get("cav_hot", 0.0), Q, 1e-3, "W"),
        Check("cavity face 'cav_cold' receives Q", faces.get("cav_cold", 0.0), Q, 1e-3, "W"),
        Check("heat balance (1 + residual)", 1 + th.balance["imbalance"], 1.0, 1e-3, ""),
    ]


THERMAL_CASES: dict[str, Case] = {
    "TH8B": Case("TH8B", "Heat: bonded interface with contact conductance (separate mesh, ΔT = q / h_c)", th8b_build, th8b_eval),
    "TH9C": Case("TH9C", "Heat: cavity radiation with two emissivities (ε_eff = 1/(1/ε1 + 1/ε2 − 1))", th9c_build, th9c_eval),
    "T1": Case("T1", "Teplo: tyč ocel + hliník, konvekce / tok / výkon (sériové odpory)", t1_build, t1_eval),
    "T2": Case("T2", "Teplo: trubka ri 20 / ro 40, radiální vedení (logaritmický profil)", t2_build, t2_eval),
    "T3": Case("T3", "Teplo: tyč s vnitřním zdrojem 10 W (parabolický profil)", t3_build, t3_eval),
    "T4": Case("T4", "Teplotní napětí: volná a bráněná roztažnost tyče, ΔT = 100 K", t4_build, t4_eval),
    "T5": Case("T5", "Teplotní napětí trubky z pole T2 (rovinná deformace, Timoshenko)",
               lambda wd: t2_build(wd, stress=True), t5_eval),
    "T6": Case("T6", "Teplo: přechodové vedení tyče po skoku teploty (Fourierova řada)", t6_build, t6_eval),
    "T7": Case("T7", "Teplo: sálání čela tyče do okolí (nelineární, kořen rovnice)", t7_build, t7_eval),
    "TH8": Case("TH8", "Teplo: dvě desky v sérii s tepelným odporem styku (ΔT = q / h_c)", th8_build, th8_eval),
    "TH9": Case("TH9", "Teplo: sálání mezi plochami (dutinové sálání, Stefan-Boltzmann)", th9_build, th9_eval),
    "T8": Case("T8", "Teplo: dvě desky v sérii s tepelným odporem styku (alias TH8)", th8_build, th8_eval),
    "T9": Case("T9", "Teplo: sálání mezi plochami (alias TH9)", th9_build, th9_eval),
}

