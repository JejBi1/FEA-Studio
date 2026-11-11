"""Verification cases F0 (V1–V7): steel E = 210 GPa, ν = 0.3; geometry from OCP through a STEP.

Criterion: displacements within 3 %, stress at a non-singular point within 5 %.
Run: python -m verification.run_all [V1 V3 ...]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

from fea_core.geometry import Part, write_step, read_step, _solids
from fea_core.results import Results
from fea_core.selectors import find_faces
from fea_core.study import (Study, Material, Fixed, Displacement, Force, Pressure,
                            Gravity, RemoteLoad, LoadCase, MeshSettings)

E, NU, RHO = 210000.0, 0.3, 7.85e-9
STEEL = Material("Ocel S235", E, NU, RHO, Re=235.0, Rm=360.0, source="EN 10025-2")


# ------------------------------------------------------------------ geometrie

def box(x0, y0, z0, dx, dy, dz):
    return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), dx, dy, dz).Shape()


def cyl(p, d, r, h):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*p), gp_Dir(*d)), r, h).Shape()


def solid(shape):
    s = _solids(shape)
    assert len(s) == 1, f"čekáno 1 těleso, je {len(s)}"
    return s[0]


# ------------------------------------------------------------------ checks

@dataclass
class Check:
    name: str
    value: float
    ref: float
    tol: float          # relative
    unit: str = ""

    @property
    def error(self) -> float:
        if abs(self.ref) < 1e-12: return abs(self.value - self.ref)
        return abs(self.value - self.ref) / abs(self.ref)

    @property
    def ok(self) -> bool:
        return self.error <= self.tol


class AtLeast(Check):
    """One-sided check "value ≥ reference" (error 0 when satisfied; tol relative to |ref|)."""

    @property
    def error(self) -> float:
        return max(0.0, (self.ref - self.value) / abs(self.ref))


class AtMost(Check):
    """One-sided check "value ≤ reference" (error 0 when satisfied; tol relative to |ref|)."""

    @property
    def error(self) -> float:
        return max(0.0, (self.value - self.ref) / abs(self.ref))


@dataclass
class Case:
    key: str
    title: str
    build: Callable[[Path], tuple[Study, list[Part]]]
    evaluate: Callable[[Results, Study], list[Check]]
    checks: list[Check] = field(default_factory=list)
    run: Callable | None = None       # own calculation instead of run_study: run(study, workdir, parts) → result


def _study(name: str, step: Path, parts: list[Part], mesh: MeshSettings) -> tuple[Study, list[Part]]:
    write_step(parts, step)
    parts = read_step(step)  # always through a STEP, as for the user
    st = Study(name=name, geometry=str(step), mesh=mesh)
    st.add_material(STEEL, default=True)
    return st, parts


# ------------------------------------------------------------------ V1
def v1_build(wd: Path, split: bool = False):
    if split:
        geo = [Part("Konzola A", box(0, 0, 0, 100, 20, 10)), Part("Konzola B", box(100, 0, 0, 100, 20, 10))]
    else:
        geo = [Part("Konzola", box(0, 0, 0, 200, 20, 10))]
    st, parts = _study("V6 delena konzola" if split else "V1 konzola", wd / "model.step", geo,
                       MeshSettings(size=2.5))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=200))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Sila 100 N", [Force("konec", [0, 0, -100.0])]))
    return st, parts


def _tip_deflection(res: Results) -> float:
    m = res.mesh
    end = np.nonzero(np.abs(m.nodes[:, 0] - 200) < 1e-6)[0]
    return float(res.step(0).U[end, 2].mean())


def v1_eval(res: Results, st: Study):
    s = res.step(0)
    sx_top = res.probe([100, 10, 10], "S")[0]
    sx_bot = res.probe([100, 10, 0], "S")[0]
    return [
        Check("průhyb konce (Timoshenko)", -_tip_deflection(res), 0.7634, 0.03, "mm"),
        Check("σx horní vlákno x=100", sx_top, 30.0, 0.05, "MPa"),
        Check("σx dolní vlákno x=100", -sx_bot, 30.0, 0.05, "MPa"),
        Check("reakce Fz", s.reactions["vetknuti"][2], 100.0, 0.001, "N"),
    ]


# ------------------------------------------------------------------ V2
def v2_build(wd: Path):
    st, parts = _study("V2 tyc tlak", wd / "model.step", [Part("Tyc", box(0, 0, 0, 100, 10, 10))],
                       MeshSettings(size=3.0))
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("y0", find_faces(parts, y=0))
    st.add_selection("z0", find_faces(parts, z=0))
    st.add_selection("celo", find_faces(parts, x=100))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0), Displacement("z0", uz=0.0)]
    st.load_cases.append(LoadCase("Tlak 100 MPa", [Pressure("celo", 100.0)]))
    return st, parts


def v2_eval(res: Results, st: Study):
    s = res.step(0)
    end = np.nonzero(np.abs(res.mesh.nodes[:, 0] - 100) < 1e-6)[0]
    return [
        Check("σx (střed)", res.probe([50, 5, 5], "S")[0], -100.0, 0.001, "MPa"),
        Check("σx max |odchylka| v uzlech", float(np.abs(s.S[:, 0]).max()), 100.0, 0.001, "MPa"),
        Check("posuv čela ux", float(s.U[end, 0].mean()), -100 * 100 / E, 0.001, "mm"),
        Check("reakce Fx", s.reactions["x0"][0], 100 * 10 * 10, 0.001, "N"),
    ]


# ------------------------------------------------------------------ V3
V3_L, V3_W, V3_T, V3_R = 150.0, 50.0, 2.5, 5.0   # 1/8 of the model: half the length, half the width, half the thickness


def v3_build(wd: Path):
    plate = solid(BRepAlgoAPI_Cut(box(0, 0, 0, V3_L, V3_W, V3_T), cyl((0, 0, -1), (0, 0, 1), V3_R, V3_T + 2)).Shape())
    st, parts = _study("V3 deska s otvorem", wd / "model.step", [Part("Deska", plate)],
                       MeshSettings(size=4.0, local=[{"selection": "otvor", "size": 0.35}]))
    st.add_selection("sym_x", find_faces(parts, x=0))
    st.add_selection("sym_y", find_faces(parts, y=0))
    st.add_selection("sym_z", find_faces(parts, z=0))
    st.add_selection("konec", find_faces(parts, x=V3_L))
    st.add_selection("otvor", find_faces(parts, radius=V3_R))
    st.constraints += [Displacement("sym_x", ux=0.0), Displacement("sym_y", uy=0.0), Displacement("sym_z", uz=0.0)]
    st.load_cases.append(LoadCase("Tah 100 MPa", [Pressure("konec", -100.0)]))
    return st, parts


def v3_eval(res: Results, st: Study):
    s = res.step(0)
    n = res.mesh.nodes
    edge = np.nonzero((np.abs(n[:, 0]) < 1e-6) & (np.abs(n[:, 1] - V3_R) < 1e-3))[0]
    smax_mid = float(s.S[edge, 0].max())   # at the hole in the plane x=0 (maximum in the mid-plane)
    smax_avg = float(s.S[edge, 0].mean())  # average over the thickness ~ plane stress
    kt = 3 - 3.14 * 0.1 + 3.667 * 0.01 - 1.527 * 0.001  # Heywood/Peterson, d/W = 0.1
    ref = kt * 100 * 100 / 90
    return [
        Check("σx max na otvoru, průměr po tl. (Kt,n=2,729)", smax_avg, ref, 0.05, "MPa"),
        Check("σx max na otvoru, střední rovina", smax_mid, ref, 0.05, "MPa"),
        Check("reakce Fx (síla 100 MPa·50·2,5)", -s.reactions["sym_x"][0], 100 * V3_W * V3_T, 0.001, "N"),
    ]


# ------------------------------------------------------------------ V4
RI, RO, V4_H, P4 = 20.0, 40.0, 10.0, 100.0


def v4_build(wd: Path):
    ring = BRepAlgoAPI_Cut(cyl((0, 0, 0), (0, 0, 1), RO, V4_H), cyl((0, 0, -1), (0, 0, 1), RI, V4_H + 2)).Shape()
    quarter = solid(BRepAlgoAPI_Common(ring, box(0, 0, -1, RO + 1, RO + 1, V4_H + 2)).Shape())
    st, parts = _study("V4 trubka Lame", wd / "model.step", [Part("Trubka", quarter)],
                       MeshSettings(size=2.5))
    st.add_selection("sym_x", find_faces(parts, x=0))
    st.add_selection("sym_y", find_faces(parts, y=0))
    st.add_selection("sym_z", find_faces(parts, z=0))
    st.add_selection("vnitrni", find_faces(parts, radius=RI))
    st.constraints += [Displacement("sym_x", ux=0.0), Displacement("sym_y", uy=0.0), Displacement("sym_z", uz=0.0)]
    st.load_cases.append(LoadCase("Vnitřní tlak 100 MPa", [Pressure("vnitrni", P4)]))
    return st, parts


def _hoop(res: Results, r: float, phi: float, z: float) -> tuple[float, float]:
    p = [r * math.cos(phi), r * math.sin(phi), z]
    S = res.probe(p, "S")
    from fea_core.results import tensor, von_mises
    T = tensor(S)
    t = np.array([-math.sin(phi), math.cos(phi), 0.0])
    return float(t @ T @ t), float(von_mises(S))


def v4_eval(res: Results, st: Study):
    s = res.step(0)
    n = res.mesh.nodes
    a = P4 * RI ** 2 / (RO ** 2 - RI ** 2)
    sth_i, sth_o = a * (1 + RO ** 2 / RI ** 2), a * (1 + RO ** 2 / RO ** 2)
    vm_i = math.sqrt(sth_i ** 2 + P4 ** 2 + sth_i * P4)
    u_i = RI / E * (sth_i + NU * P4)
    phi = math.radians(45)
    hi, vmi = _hoop(res, RI, phi, V4_H / 2)
    ho, _ = _hoop(res, RO, phi, V4_H / 2)
    r = np.hypot(n[:, 0], n[:, 1])
    inner = np.nonzero(np.abs(r - RI) < 1e-3)[0]
    ur = (s.U[inner, 0] * n[inner, 0] + s.U[inner, 1] * n[inner, 1]) / r[inner]
    return [
        Check("σθ na ri", hi, sth_i, 0.05, "MPa"),
        Check("σθ na ro", ho, sth_o, 0.05, "MPa"),
        Check("von Mises na ri", vmi, vm_i, 0.05, "MPa"),
        Check("u(ri) průměr", float(ur.mean()), u_i, 0.03, "mm"),
    ]


# ------------------------------------------------------------------ V5
def v5_build(wd: Path):
    st, parts = _study("V5 sloup vlastni tihou", wd / "model.step", [Part("Sloup", box(0, 0, 0, 100, 100, 1000))],
                       MeshSettings(size=25.0))
    st.add_selection("pata", find_faces(parts, z=0))
    st.constraints.append(Fixed("pata"))
    st.load_cases.append(LoadCase("Vlastní tíha", [Gravity([0, 0, -9810.0])]))
    return st, parts


def v5_eval(res: Results, st: Study):
    return [Check("reakce Fz", res.step(0).reactions["pata"][2], 100 * 100 * 1000 * RHO * 9810, 0.001, "N")]


# ------------------------------------------------------------------ V6
_V1_REF: dict = {}


def v6_eval(res: Results, st: Study):
    checks = v1_eval(res, st)
    n = res.mesh.nodes
    iface = np.nonzero(np.abs(n[:, 0] - 100) < 1e-6)[0]
    # conforming mesh: nodes of the interface shared by both parts
    m = res.mesh
    in_a = np.zeros(len(n), bool)
    in_b = np.zeros(len(n), bool)
    in_a[m.tets[m.tet_part == 0].ravel()] = True
    in_b[m.tets[m.tet_part == 1].ravel()] = True
    shared = int(np.count_nonzero(in_a & in_b))
    checks.append(Check("sdílené uzly na rozhraní / uzly v rovině x=100", shared, len(iface), 0.0, ""))
    # deflection against V1 (one body, another mesh): the reference is computed here at the first use (the check does not depend on whether
    # V1 ran earlier in the same session); run_all may pass it ready
    if "tip" not in _V1_REF:
        import tempfile
        from fea_core.solve import run_study
        with tempfile.TemporaryDirectory() as td:
            st1, _ = v1_build(Path(td))
            _V1_REF["tip"] = -_tip_deflection(run_study(st1, Path(td) / "run"))
    checks.append(Check("průhyb proti V1 (jiná síť)", -_tip_deflection(res), _V1_REF["tip"], 0.005, "mm"))
    return checks


# ------------------------------------------------------------------ V7
D7, L7, T7 = 20.0, 200.0, 100_000.0  # mm, mm, N·mm


def v7_build(wd: Path, coupling: str = "distributing"):
    st, parts = _study(f"V7 hridel krut ({coupling})", wd / "model.step",
                       [Part("Hridel", cyl((0, 0, 0), (1, 0, 0), D7 / 2, L7))], MeshSettings(size=2.5))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=L7))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Krouticí moment 100 N·m",
                                  [RemoteLoad("konec", [L7, 0, 0], moment=[T7, 0, 0], coupling=coupling)]))
    return st, parts


def _twist(res: Results, x: float) -> float:
    n = res.mesh.nodes
    U = res.step(0).U
    r = np.hypot(n[:, 1], n[:, 2])
    sel = np.nonzero((np.abs(n[:, 0] - x) < 1e-6) & (np.abs(r - D7 / 2) < 1e-3))[0]
    # u_θ / r, θ around the axis x
    ut = (-U[sel, 1] * n[sel, 2] + U[sel, 2] * n[sel, 1]) / r[sel]
    return float((ut / r[sel]).mean())


def v7_eval(res: Results, st: Study):
    G = E / (2 * (1 + NU))
    J = math.pi * D7 ** 4 / 32
    tau = 16 * T7 / (math.pi * D7 ** 3)
    from fea_core.results import von_mises
    taus = [float(von_mises(res.probe([100, (D7 / 2) * math.cos(a), (D7 / 2) * math.sin(a)], "S"))) / math.sqrt(3)
            for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    s = res.step(0)
    return [
        Check("τ na povrchu x=100 (průměr po obvodu)", float(np.mean(taus)), tau, 0.05, "MPa"),
        Check("zkroucení konce x=200", math.degrees(_twist(res, L7)), math.degrees(T7 * L7 / (G * J)), 0.03, "°"),
        Check("zkroucení mezi x=50 a 150 (×2)", math.degrees(2 * (_twist(res, 150) - _twist(res, 50))),
              math.degrees(T7 * L7 / (G * J)), 0.03, "°"),
        Check("reakční moment Mx ve vetknutí", -float(s.reaction_moments["vetknuti"][0]), T7, 0.001, "N·mm"),
    ]


# ------------------------------------------------------------------ U1, U2 (fatigue)
U_SN = {"SD": 20.0, "ND": 1e6, "k": 5.0, "k2": 0.0, "source": "testovací S-N"}
U_CYCLES, U_REPEATS = 1000, 100


def _fatigue_settings(channels):
    from fea_core.fatigue.history import LoadHistory
    from fea_core.fatigue.settings import FatigueSettings
    return FatigueSettings(LoadHistory(channels), equivalent="max_principal", mean_correction="none",
                           repeats=U_REPEATS, surface="lesteny", sn={STEEL.name: dict(U_SN)})


def u1_build(wd: Path):
    from fea_core.fatigue.history import Channel
    st, parts = v1_build(wd)
    st.name = "U1 konzola strida"
    st.fatigue = _fatigue_settings([Channel(st.load_cases[0].name, "sine", amplitude=1.0, cycles=U_CYCLES)])
    return st, parts


def _nearest(res, p):
    return int(np.argmin(np.linalg.norm(res.mesh.nodes - np.asarray(p, float), axis=1)))


def _miner(amp):
    N = U_SN["ND"] * (U_SN["SD"] / amp) ** U_SN["k"]
    return U_CYCLES / N


def u1_eval(res: Results, st: Study):
    from fea_core.fatigue.analysis import run_fatigue
    fr = run_fatigue(st, res)
    j = _nearest(res, [100, 10, 10])
    a = float(fr.amplitude[j])
    D_ref = _miner(a)
    return [
        Check("amplituda σ v polovině délky (horní vlákno)", a, 30.0, 0.05, "MPa"),
        Check("poškození za blok = Miner z amplitudy uzlu", float(fr.damage[j]), D_ref, 1e-6, ""),
        Check("bezpečnost = (1/(R·D))^(1/k)", float(fr.safety[j]), (1 / (U_REPEATS * D_ref)) ** (1 / U_SN["k"]),
              1e-4, ""),
        Check("poškození proti teorii (σa = 30 MPa)", float(fr.damage[j]), _miner(30.0), 0.25, ""),
    ]


def u2_build(wd: Path):
    from fea_core.fatigue.history import Channel
    st, parts = v1_build(wd)
    st.name = "U2 konzola dva stavy"
    st.load_cases = [LoadCase("Fz", [Force("konec", [0, 0, -100.0])]),
                     LoadCase("Fy", [Force("konec", [0, 100.0, 0])])]
    st.fatigue = _fatigue_settings([Channel("Fz", "sine", amplitude=1.0, cycles=U_CYCLES),
                                    Channel("Fy", "sine", amplitude=1.0, cycles=U_CYCLES)])
    return st, parts


def u2_eval(res: Results, st: Study):
    from fea_core.fatigue.analysis import run_fatigue
    fr = run_fatigue(st, res)
    j = _nearest(res, [100, 0, 10])          # roh: tah od Fz (30 MPa) i od Fy (15 MPa)
    sx = res.step("Fz").S[j, 0] + res.step("Fy").S[j, 0]
    return [
        Check("amplituda v rohu x=100 (30 + 15 MPa)", float(fr.amplitude[j]), 45.0, 0.05, "MPa"),
        Check("superpozice: amplituda = |σx(Fz) + σx(Fy)| v uzlu", float(fr.amplitude[j]), abs(float(sx)), 0.02, "MPa"),
        Check("poškození za blok = Miner z amplitudy uzlu", float(fr.damage[j]), _miner(float(fr.amplitude[j])),
              1e-6, ""),
    ]


# ------------------------------------------------------------------ U3 (fatigue – critical plane)
SIG3 = 80.0                              # tah σ = F/A [MPa]
F3 = SIG3 * math.pi * D7 ** 2 / 4        # axial force [N]


def u3_build(wd: Path):
    from fea_core.fatigue.history import Channel
    st, parts = v7_build(wd)
    st.name = "U3 hridel tah + krut, kriticka rovina"
    st.load_cases = [LoadCase("Krut", [RemoteLoad("konec", [L7, 0, 0], moment=[T7, 0, 0])]),
                     LoadCase("Tah", [Force("konec", [F3, 0, 0])])]
    st.fatigue = _fatigue_settings([Channel("Tah", "sine", amplitude=1.0, cycles=U_CYCLES)])
    st.fatigue.equivalent = "critical_plane"
    return st, parts


def _tensor(s):
    return np.array([[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]])


def _cp_reference(A, B, ratio: float) -> tuple[float, float]:
    """Independent check (numpy, analytic in time): σ(t) = A·sin ωt + B·cos ωt (A, B components xx…zx).
    In the plane n: σn = a·sin + b·cos → amplitude = σn,max = |(a, b)|; the shear vector describes an ellipse
    τ(t) = τA·sin + τB·cos → τa = semi-major axis = largest singular value of [τA τB] (worst direction).
    Planes: hemisphere 200 000 + refinement around the maximum (±1°, step 0.02°) → (Findley σeq, max σn amplitude).
    """
    from fea_core.fatigue.critical_plane import hemisphere
    from fea_core.fatigue.settings import findley_k, findley_lambda
    k, TA, TB = findley_k(ratio), _tensor(A), _tensor(B)

    def values(N):
        tA, tB = N @ TA, N @ TB
        a, b = np.einsum("pi,pi->p", tA, N), np.einsum("pi,pi->p", tB, N)
        sA, sB = tA - a[:, None] * N, tB - b[:, None] * N
        g11, g22, g12 = np.einsum("pi,pi->p", sA, sA), np.einsum("pi,pi->p", sB, sB), np.einsum("pi,pi->p", sA, sB)
        tau = np.sqrt(0.5 * (g11 + g22) + np.sqrt(0.25 * (g11 - g22) ** 2 + g12 ** 2))
        sn = np.hypot(a, b)
        return (tau + k * sn) / findley_lambda(k), sn

    out = []
    for which in (0, 1):
        N = hemisphere(200000)
        v = values(N)[which]
        n0 = N[np.argmax(v)]
        e = np.cross(n0, [1.0, 0, 0] if abs(n0[0]) < 0.9 else [0, 1.0, 0])
        e /= np.linalg.norm(e)
        e2 = np.cross(n0, e)
        g = np.radians(np.linspace(-1, 1, 101))
        G1, G2 = np.meshgrid(g, g)
        M = n0 + G1.reshape(-1, 1) * e + G2.reshape(-1, 1) * e2
        M /= np.linalg.norm(M, axis=1)[:, None]
        out.append(max(float(v.max()), float(values(M)[which].max())))
    return out[0], out[1]


def u3_eval(res: Results, st: Study):
    from dataclasses import replace
    from fea_core.fatigue.analysis import run_fatigue
    from fea_core.fatigue.history import Channel, LoadHistory
    from fea_core.fatigue.settings import CP_RATIO_DUCTILE as s3
    tau = 16 * T7 / (math.pi * D7 ** 3)
    j = _nearest(res, [100, 0, D7 / 2])                     # surface at the half length

    def run(channels, **kw):
        fs = replace(st.fatigue, history=LoadHistory(channels), **kw)
        return run_fatigue(replace(st, fatigue=fs), res)

    S_krut, S_tah = res.step("Krut").S[j], res.step("Tah").S[j]
    zero = np.zeros(6)
    tah = [Channel("Tah", "sine", amplitude=1.0, cycles=5)]
    krut = [Channel("Krut", "sine", amplitude=1.0, cycles=5)]
    op = [Channel("Tah", "sine", amplitude=1.0, cycles=5, samples_per_cycle=72),
          Channel("Krut", "sine", amplitude=1.0, cycles=5, phase_deg=90.0, samples_per_cycle=72)]
    # (a) uniaxial tension: σn criterion = max. principal stress in all nodes, Findley = σa
    a_n = run(tah, cp_criterion="normal")
    a_p = run(tah, equivalent="max_principal")
    a_f = run(tah, cp_criterion="findley")
    big = a_p.amplitude > 0.05 * a_p.amplitude.max()
    dev = float(np.max(np.abs(a_n.amplitude[big] / a_p.amplitude[big] - 1)))
    devD = float(np.max(np.abs(a_n.damage[big] / a_p.damage[big] - 1)))
    # (b) pure torsion σ1 = −σ3
    b_n = run(krut, cp_criterion="normal")
    b_f = run(krut, cp_criterion="findley")
    tb_f, tb_n = _cp_reference(S_krut, zero, s3)
    # (c) tension + torsion 90° out of phase (hand calculation: plane ⟂ to the axis, τa = τ, σn,max = σ)
    c_f = run(op, cp_criterion="findley")
    c_n = run(op, cp_criterion="normal")
    tc_f, tc_n = _cp_reference(S_tah, S_krut, s3)       # sin(ωt + 90°) = cos ωt
    from fea_core.fatigue.settings import findley_k, findley_lambda
    k = findley_k(s3)
    th = np.linspace(0, np.pi, 200001)
    c2, s2 = np.cos(2 * th), np.sin(2 * th)
    hand_f = float(((np.hypot(SIG3 * s2 / 2, tau * c2) + k * np.hypot(SIG3 * (1 + c2) / 2, tau * s2))
                    / findley_lambda(k)).max())
    hand_n = float(np.hypot(SIG3 * (1 + c2) / 2, tau * s2).max())
    return [
        Check("(a) tah: σn krit. rovina / max. hlavní, amplituda (max. odch.)", 1 + dev, 1.0, 1e-4, ""),
        Check("(a) tah: σn krit. rovina / max. hlavní, poškození (max. odch.)", 1 + devD, 1.0, 5e-4, ""),
        Check("(a) tah: Findley σekv = σa (uzel na povrchu)", float(a_f.amplitude[j]), float(a_p.amplitude[j]),
              1e-4, "MPa"),
        Check("(a) tah: σa na povrchu = F/A", float(a_f.amplitude[j]), SIG3, 0.03, "MPa"),
        Check("(b) krut: σn krit. rovina σa = τ (teorie)", float(b_n.amplitude[j]), tau, 0.05, "MPa"),
        Check("(b) krut: σn krit. rovina = numpy kontrola uzlu", float(b_n.amplitude[j]), tb_n, 1e-4, "MPa"),
        Check("(b) krut: Findley σekv = √3·τ (teorie)", float(b_f.amplitude[j]), s3 * tau, 0.05, "MPa"),
        Check("(b) krut: Findley = numpy kontrola uzlu", float(b_f.amplitude[j]), tb_f, 1e-4, "MPa"),
        Check("(c) 90° mimo fázi: Findley = ruční výpočet", float(c_f.amplitude[j]), hand_f, 0.05, "MPa"),
        Check("(c) 90° mimo fázi: Findley = numpy kontrola uzlu", float(c_f.amplitude[j]), tc_f, 1e-3, "MPa"),
        Check("(c) 90° mimo fázi: σn krit. rovina = ruční výpočet", float(c_n.amplitude[j]), hand_n, 0.05, "MPa"),
        Check("(c) 90° mimo fázi: σn krit. rovina = numpy kontrola", float(c_n.amplitude[j]), tc_n, 1e-3, "MPa"),
        Check("(c) poškození = Miner z amplitudy (5 cyklů/blok)", float(c_f.damage[j]),
              5 / (U_SN["ND"] * (U_SN["SD"] / float(c_f.amplitude[j])) ** U_SN["k"]), 1e-6, ""),
    ]


CASES: dict[str, Case] = {
    "V1": Case("V1", "Konzola 200×20×10, 100 N na konci", v1_build, v1_eval),
    "V2": Case("V2", "Tyč 10×10×100, tlak 100 MPa", v2_build, v2_eval),
    "V3": Case("V3", "Deska š. 100 tl. 5 s otvorem Ø10, tah 100 MPa (1/8 modelu)", v3_build, v3_eval),
    "V4": Case("V4", "Trubka ri 20 / ro 40, vnitřní tlak 100 MPa (Lamé)", v4_build, v4_eval),
    "V5": Case("V5", "Sloup 100×100×1000 vlastní tíhou", v5_build, v5_eval),
    "V6": Case("V6", "V1 rozdělená na dvě spojená tělesa", lambda wd: v1_build(wd, split=True), v6_eval),
    "V7": Case("V7", "Hřídel Ø20×200, T = 100 N·m (distribuční vazba)", v7_build, v7_eval),
    "V7R": Case("V7R", "Hřídel Ø20×200, T = 100 N·m (tuhá vazba)", lambda wd: v7_build(wd, "rigid"), v7_eval),
    "U1": Case("U1", "Únava: konzola V1, ±100 N, 1000 kmitů/blok, 100 bloků", u1_build, u1_eval),
    "U2": Case("U2", "Únava: konzola, dva jednotkové stavy Fz + Fy ve fázi (superpozice)", u2_build, u2_eval),
    "U3": Case("U3", "Únava – kritická rovina: hřídel Ø20, tah 80 MPa / krut 100 N·m / obojí 90° mimo fázi",
               u3_build, u3_eval),
}

from verification.joint_cases import JOINT_CASES  # noqa: E402 – bolts and welds (F3)

CASES.update(JOINT_CASES)

from verification.print_cases import PRINT_CASES  # noqa: E402 – 3D tisk (F4)

CASES.update(PRINT_CASES)

from verification.cyl_cases import CYL_CASES  # noqa: E402 – cylindrical support, bearing force

CASES.update(CYL_CASES)

from verification.thermal_cases import THERMAL_CASES  # noqa: E402 – heat (heat conduction, thermal stress)

CASES.update(THERMAL_CASES)

from verification.remote_cases import REMOTE_CASES  # noqa: E402 – remote constraint (rigid / distributing)

CASES.update(REMOTE_CASES)
from verification.load_cases import LOAD_CASES  # noqa: E402

CASES.update(LOAD_CASES)
from verification.eigen_cases import EIGEN_CASES  # noqa: E402 – natural frequencies and buckling (F7)

CASES.update(EIGEN_CASES)
from verification.support_cases import SUPPORT_CASES  # noqa: E402 – elastic support, *TIE (F7)

CASES.update(SUPPORT_CASES)
from verification.ortho_cases import ORTHO_CASES  # noqa: E402 – orthotropic material (F7)

CASES.update(ORTHO_CASES)
from verification.frame_cases import FRAME_CASES  # noqa: E402 – frame system (F7)

CASES.update(FRAME_CASES)


def _twod_wrap(case: Case) -> Case:
    """2D cases (verification.twod_cases): run_study returns Results with .twod; evaluate gets Results2D and the folder of
    the case (A1 compares with the 3D calculation V4 in the same folder).
    """
    def build(wd: Path):
        st, parts = case.build(wd)
        st.twod.enabled = True
        return st, parts

    def evaluate(res, st):
        return case.evaluate(res.twod, st, Path(st.geometry).parent)
    return Case(case.key, case.title, build, evaluate)


from verification.twod_cases import TWOD_CASES  # noqa: E402 – 2D calculations by a section (F7)

CASES.update({k: _twod_wrap(c) for k, c in TWOD_CASES.items()})


def _shell_wrap(case) -> Case:
    """Shell cases (verification.shell_cases): run_study returns Results with .shell."""
    def build(wd: Path):
        from fea_core.shell.model import ShellSettings
        st, parts = case.build(wd)
        if getattr(case, "settings", None):
            st.shell = ShellSettings.from_dict(case.settings)
        st.shell.enabled = True
        return st, parts

    def evaluate(res, st):
        return case.evaluate(res.shell, st, Path(st.geometry).parent)
    return Case(case.key, case.title, build, evaluate)


from verification.shell_cases import SHELL_CASES  # noqa: E402 – shells S6 (F7)

CASES.update({k: _shell_wrap(c) for k, c in SHELL_CASES.items()})
try:
    from verification.motion_cases import MOTION_CASES  # noqa: E402 – mechanismy (etapa F8.3, F8.8)
    CASES.update(MOTION_CASES)
except ModuleNotFoundError as _e:          # in progress (parallel work) – the other cases must be runnable
    if "motion_cases" not in str(_e):
        raise
from verification.dynamics_cases import DYN_CASES  # noqa: E402 – dynamika konstrukce (etapa F8.1)

CASES.update(DYN_CASES)
from verification.contact_cases import CONTACT_CASES  # noqa: E402 – contact and friction (stage F8.5)

CASES.update(CONTACT_CASES)

from verification.f9_cases import F9_CASES  # noqa: E402 – etapa F9.1

CASES.update(F9_CASES)

from verification.topo_cases import TOPO_CASES  # noqa: E402 – topology optimisation (stage F9.7)

CASES.update(TOPO_CASES)

from verification.plasticity_cases import CASES as PLASTIC_CASES  # noqa: E402 – plasticity PL1–PL4 (the import error must not be hidden)

CASES.update(PLASTIC_CASES)

from fea_core.modules import ModuleMissing  # noqa: E402

try:                                       # ε-N fatigue (U4) needs the accel module (Numba); without it only this case is missing
    from verification.fatigue_cases import FATIGUE_CASES  # noqa: E402 – etapa F9.4
except ModuleMissing as _e:
    print(f"(úloha U4 vynechána: {_e})")
    FATIGUE_CASES = {}
CASES.update(FATIGUE_CASES)
