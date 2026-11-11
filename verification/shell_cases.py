"""Verification cases of the shell model (fea_core.shell, ccx S6): Sh1–Sh6.

Run: python -m verification.shell_cases [Sh1 Sh2 ...]   (table as verification.run_all,
output verification/out/shell_report.json). Steel E = 210 GPa, ν = 0.3.

Sh1 cantilever from a plate 200×40×5 (force at the end): deflection FL³/3EI, stress of the top face M/W, reaction.
Sh2 square plate 400×400×5 hinged along the perimeter, uniform pressure (Timoshenko & Woinowsky-Krieger,
    w = 0.00406 q a⁴/D, M = 0.0479 q a²).
Sh3 cylindrical shell R = 100, t = 4 with internal pressure (σθ = pR/t, u_r = pR²/(Et)); checks the cylindrical skin,
    the offset of the mid-surface and the constraint of the side in local axes (*TRANSFORM).
Sh4 plate with a hole (cantilever) against the 3D solid model of the same plate (run_study).
Sh5 T-beam from two plates (extension of the mid-surface of the web to the flange) against the 3D model.
Sh6 sheet with a 90° bend (offset of the skin) against the 3D model.
Sh7 plate part from a STEP with a thickness given by the user (mid / bottom surface = OFFSET).
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from verification.cases import Check, STEEL, E, NU, box, cyl, solid
from fea_core.geometry import Part, write_step, read_step
from fea_core.selectors import find_faces, PointSelector
from fea_core.study import Study, Fixed, Displacement, Force, Pressure, Gravity, LoadCase, MeshSettings

OUT = Path(__file__).resolve().parent / "out"


@dataclass
class ShellCase:
    key: str
    title: str
    build: Callable[[Path], tuple[Study, list]]
    evaluate: Callable                     # (ShellResults, Study, workdir) → list[Check]
    settings: dict | None = None           # ShellSettings as a dict (None = all parts automatically)
    checks: list[Check] = field(default_factory=list)


def _study(name: str, wd: Path, geo: list[Part], mesh_size: float) -> tuple[Study, list[Part]]:
    step = wd / "model.step"
    write_step(geo, step)
    parts = read_step(step)
    st = Study(name=name, geometry=str(step), mesh=MeshSettings(size=mesh_size))
    st.add_material(STEEL, default=True)
    return st, parts


def _at(res, cond) -> np.ndarray:
    x = res.mesh.nodes
    return np.nonzero(cond(x[:, 0], x[:, 1], x[:, 2]))[0]


def _run3d(st: Study, wd: Path):
    """The same solid 3D (a copy of the study with the shells switched off – run_study would otherwise choose the shell model)."""
    from fea_core.solve import run_study
    st3 = Study.from_dict(st.to_dict())
    st3.shell.enabled = False
    return run_study(st3, wd / "run3d")


# ------------------------------------------------------------------ Sh1
SH1_L, SH1_B, SH1_T, SH1_F = 200.0, 40.0, 5.0, 100.0


def sh1_build(wd: Path):
    st, parts = _study("Sh1 konzola deska", wd, [Part("Deska", box(0, 0, 0, SH1_L, SH1_B, SH1_T))], 5.0)
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=SH1_L))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Síla 100 N", [Force("konec", [0, 0, -SH1_F])]))
    st.load_cases.append(LoadCase("Tíha", [Gravity([0, 0, -9810.0])]))
    return st, parts


def sh1_eval(res, st, wd):
    s = res.step(0)
    tip = _at(res, lambda x, y, z: np.abs(x - SH1_L) < 1e-6)
    I = SH1_B * SH1_T ** 3 / 12
    G = E / (2 * (1 + NU))
    w_ref = SH1_F * SH1_L ** 3 / (3 * E * I) + SH1_F * SH1_L / (5 / 6 * G * SH1_B * SH1_T)
    W = SH1_B * SH1_T ** 2 / 6
    pr = res.probe([SH1_L / 2, SH1_B / 2, SH1_T / 2], "sxx", 0, "top")
    pb = res.probe([SH1_L / 2, SH1_B / 2, SH1_T / 2], "sxx", 0, "bot")
    r = s.reactions["vetknuti"]
    return [
        # beam theory: a plate with b/t = 8 is stiffer at the clamping (the transverse curvature is restrained) – the 3D body gives −1.8 %
        Check("průhyb konce (Timoshenko, nosník)", -float(s.U[tip, 2].mean()), w_ref, 0.03, "mm"),
        Check("σx horní plocha x = L/2 (M/W)", pr["value"], SH1_F * SH1_L / 2 / W, 0.02, "MPa"),
        Check("σx dolní plocha x = L/2", -pb["value"], SH1_F * SH1_L / 2 / W, 0.02, "MPa"),
        Check("1 + membránové σx / M/W (≈ 1)", 1 + res.probe([SH1_L / 2, SH1_B / 2, 0], "sxx", 0, "mid")["value"] /
              (SH1_F * SH1_L / 2 / W), 1.0, 0.01, ""),
        Check("reakce Fz", r[2], SH1_F, 1e-3, "N"),
        Check("reakce My k ref. bodu (rameno síly)", -r[4] / SH1_F + res.ref_point[0],
              SH1_L, 1e-3, "mm"),
        Check("rovnováha (1 − nerovnováha)", 1 - s.equilibrium["imbalance"], 1.0, 1e-4, ""),
        Check("tíha: reakce Fz = ρ·t·A·g", res.step(1).reactions["vetknuti"][2],
              STEEL.rho * SH1_T * SH1_L * SH1_B * 9810.0, 1e-3, "N"),
    ] + _cmp3d(res, st, wd, lambda x, y, z: np.abs(x - SH1_L) < 1e-6, "konce", tol=0.02)[:1]


# ------------------------------------------------------------------ Sh2
SH2_A, SH2_T, SH2_Q = 400.0, 5.0, 0.01


def sh2_build(wd: Path):
    st, parts = _study("Sh2 deska kloubove", wd, [Part("Deska", box(0, 0, 0, SH2_A, SH2_A, SH2_T))], 10.0)
    sides = find_faces(parts, x=0) + find_faces(parts, x=SH2_A) + find_faces(parts, y=0) + find_faces(parts, y=SH2_A)
    st.add_selection("obvod", sides)
    st.add_selection("horni", find_faces(parts, z=SH2_T))
    st.add_selection("roh0", points=[PointSelector("Deska", [0, 0, SH2_T / 2])])
    st.add_selection("roh1", points=[PointSelector("Deska", [SH2_A, 0, SH2_T / 2])])
    st.constraints += [Displacement("obvod", uz=0.0), Displacement("roh0", ux=0.0, uy=0.0),
                       Displacement("roh1", uy=0.0)]
    st.load_cases.append(LoadCase("Tlak 0,01 MPa", [Pressure("horni", SH2_Q)]))
    return st, parts


def sh2_eval(res, st, wd):
    s = res.step(0)
    D = E * SH2_T ** 3 / (12 * (1 - NU ** 2))
    c = [SH2_A / 2, SH2_A / 2, SH2_T / 2]
    w = res.probe(c, "uz", 0)["value"]
    sx_top = res.probe(c, "sxx", 0, "top")["value"]
    sy_bot = res.probe(c, "syy", 0, "bot")["value"]
    m = 0.0479 * SH2_Q * SH2_A ** 2
    rz = sum(r[2] for r in s.reactions.values())
    return [
        Check("průhyb středu (Timoshenko 0,00406 q a⁴/D)", -w, 0.00406 * SH2_Q * SH2_A ** 4 / D, 0.03, "mm"),
        Check("σx horní plocha střed (6M/t², M = 0,0479 q a²)", -sx_top, 6 * m / SH2_T ** 2, 0.03, "MPa"),
        Check("σy dolní plocha střed", sy_bot, 6 * m / SH2_T ** 2, 0.03, "MPa"),
        Check("reakce Fz = q a²", rz, SH2_Q * SH2_A ** 2, 1e-3, "N"),
        Check("rovnováha (1 − nerovnováha)", 1 - s.equilibrium["imbalance"], 1.0, 1e-4, ""),
    ]


# ------------------------------------------------------------------ Sh3
SH3_R, SH3_T, SH3_L, SH3_P = 100.0, 4.0, 300.0, 1.0


def sh3_build(wd: Path):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    ro, ri = SH3_R + SH3_T / 2, SH3_R - SH3_T / 2
    tube = solid(BRepAlgoAPI_Cut(cyl((0, 0, 0), (0, 0, 1), ro, SH3_L), cyl((0, 0, 0), (0, 0, 1), ri, SH3_L)).Shape())
    st, parts = _study("Sh3 valec vnitrni tlak", wd, [Part("Trubka", tube)], 10.0)
    st.add_selection("vnitrni", find_faces(parts, type="cylinder", radius=ri))
    st.add_selection("celo", find_faces(parts, z=0))
    st.add_selection("A", points=[PointSelector("Trubka", [SH3_R, 0, SH3_L])])
    st.add_selection("B", points=[PointSelector("Trubka", [0, SH3_R, SH3_L])])
    st.add_selection("C", points=[PointSelector("Trubka", [-SH3_R, 0, SH3_L])])
    st.constraints += [Displacement("celo", uz=0.0), Displacement("A", uy=0.0), Displacement("B", ux=0.0),
                       Displacement("C", uy=0.0)]
    st.load_cases.append(LoadCase("Vnitřní tlak 1 MPa", [Pressure("vnitrni", SH3_P)]))
    return st, parts


def sh3_eval(res, st, wd):
    """Lamé (open ends, σz = 0): σθ(r) = A + B/r², u_r = ((1 − ν)A r + (1 + ν)B/r)/E. The shell is linear through the thickness →
    the mid-surface = the average of σθ(ri), σθ(ro); the bending = half of the difference (outer − inner).
    """
    s = res.step(0)
    x = res.mesh.nodes
    ro, ri = SH3_R + SH3_T / 2, SH3_R - SH3_T / 2
    A = SH3_P * ri ** 2 / (ro ** 2 - ri ** 2)
    B = A * ro ** 2
    sth = lambda r: A + B / r ** 2           # noqa: E731
    sel = np.nonzero((x[:, 2] > 0.3 * SH3_L) & (x[:, 2] < 0.7 * SH3_L))[0]
    er = x[sel, :2] / np.linalg.norm(x[sel, :2], axis=1, keepdims=True)
    et = np.column_stack([-er[:, 1], er[:, 0]])

    def hoop(S):
        return S[:, 0] * et[:, 0] ** 2 + S[:, 1] * et[:, 1] ** 2 + 2 * S[:, 3] * et[:, 0] * et[:, 1]
    hm = hoop(s.S_mid[sel])
    hb = hoop(s.bending[sel])
    ur = np.einsum("ij,ij->i", s.U[sel, :2], er)
    r_mid = np.linalg.norm(x[sel, :2], axis=1)
    return [
        Check("σθ střední plocha (Lamé, průměr ri/ro ≈ p·ri/t)", float(hm.mean()), 0.5 * (sth(ri) + sth(ro)), 0.01,
              "MPa"),
        Check("σθ rozptyl (max/průměr)", float(hm.max() / hm.mean()), 1.0, 0.01, ""),
        Check("σθ ohybová složka (Lamé (ro − ri)/2)", float(hb.mean()), 0.5 * (sth(ro) - sth(ri)), 0.05, "MPa"),
        Check("u_r střední plochy (Lamé)", float(ur.mean()),
              ((1 - NU) * A * SH3_R + (1 + NU) * B / SH3_R) / E, 0.01, "mm"),
        Check("σθ tenkostěnně pR/t (informativně, 3 %)", float(hm.mean()), SH3_P * SH3_R / SH3_T, 0.03, "MPa"),
        Check("poloměr střední plochy", float(r_mid.mean()), SH3_R, 1e-6, "mm"),
        Check("rovnováha (1 − nerovnováha)", 1 - s.equilibrium["imbalance"], 1.0, 1e-4, ""),
    ]


# ------------------------------------------------------------------ Sh4
SH4 = dict(L=200.0, B=100.0, T=5.0, D=40.0, F=200.0)


def sh4_build(wd: Path):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    L, B, T, D = SH4["L"], SH4["B"], SH4["T"], SH4["D"]
    plate = solid(BRepAlgoAPI_Cut(box(0, 0, 0, L, B, T), cyl((L / 2, B / 2, -1), (0, 0, 1), D / 2, T + 2)).Shape())
    st, parts = _study("Sh4 deska s dirou", wd, [Part("Deska", plate)], 2.5)
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=L))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Síla", [Force("konec", [0, 0, -SH4["F"]])]))
    return st, parts


def _cmp3d(res, st, wd, tip_cond, label, stress_pt=None, tol=0.03):
    r3 = _run3d(st, wd)
    s = res.step(0)
    m3 = r3.mesh
    tip3 = np.nonzero(tip_cond(m3.nodes[:, 0], m3.nodes[:, 1], m3.nodes[:, 2]))[0]
    tip = _at(res, tip_cond)
    w3 = float(r3.step(0).U[tip3, 2].mean())
    w = float(s.U[tip, 2].mean())
    out = [Check(f"průhyb {label} skořepina / 3D", w, w3, tol, "mm")]
    if stress_pt is not None:
        p3, ps, comp = stress_pt
        v3 = float(r3.probe(p3, "S")[comp])
        vs = res.probe(ps, ["sxx", "syy", "szz"][comp], 0, "top")["value"]
        out.append(Check(f"σ{'xyz'[comp]} povrch v {p3} skořepina / 3D", vs, v3, 0.05, "MPa"))
    out.append(Check("reakce Fz skořepina / 3D", sum(r[2] for r in s.reactions.values()),
                     sum(r[2] for r in r3.step(0).reactions.values()), 1e-3, "N"))
    out.append(Check("rovnováha (1 − nerovnováha)", 1 - s.equilibrium["imbalance"], 1.0, 1e-4, ""))
    return out


def sh4_eval(res, st, wd):
    L, B, T = SH4["L"], SH4["B"], SH4["T"]
    return _cmp3d(res, st, wd, lambda x, y, z: np.abs(x - L) < 1e-6, "konce",
                  ([L / 4, B / 2, T], [L / 4, B / 2, T / 2], 0))


# ------------------------------------------------------------------ Sh5 T-beam
SH5 = dict(L=300.0, BF=60.0, TF=5.0, HW=50.0, TW=4.0, F=500.0)


def sh5_build(wd: Path):
    L, BF, TF, HW, TW = SH5["L"], SH5["BF"], SH5["TF"], SH5["HW"], SH5["TW"]
    geo = [Part("Pasnice", box(0, 0, 0, L, BF, TF)), Part("Stojina", box(0, BF / 2 - TW / 2, TF, L, TW, HW))]
    st, parts = _study("Sh5 T-nosnik", wd, geo, 4.0)
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=L))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Síla", [Force("konec", [0, 0, -SH5["F"]])]))
    return st, parts


def sh5_eval(res, st, wd):
    L = SH5["L"]
    return _cmp3d(res, st, wd, lambda x, y, z: np.abs(x - L) < 1e-6, "konce")


# ------------------------------------------------------------------ Sh6 sheet with a bend
SH6 = dict(T=2.0, R=5.0, L1=100.0, L2=60.0, W=40.0, F=20.0)


def bent_sheet(t, R, L1, L2, W):
    """Sheet: horizontal part z ∈ [0, t], a 90° bend (inner radius R) at x = L1, vertical part upwards."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    a = box(0, 0, 0, L1, W, t)
    ax = gp_Ax2(gp_Pnt(L1, W, R + t), gp_Dir(0, -1, 0), gp_Dir(0, 0, -1))
    ring = BRepAlgoAPI_Cut(BRepPrimAPI_MakeCylinder(ax, R + t, W, math.pi / 2).Shape(),
                           BRepPrimAPI_MakeCylinder(ax, R, W, math.pi / 2).Shape()).Shape()
    b = box(L1 + R, 0, R + t, t, W, L2)
    s = BRepAlgoAPI_Fuse(BRepAlgoAPI_Fuse(a, ring).Shape(), b).Shape()
    u = ShapeUpgrade_UnifySameDomain(s, True, True, True)
    u.Build()
    return solid(u.Shape())


def sh6_build(wd: Path):
    t, R, L1, L2, W = SH6["T"], SH6["R"], SH6["L1"], SH6["L2"], SH6["W"]
    st, parts = _study("Sh6 plech s ohybem", wd, [Part("Plech", bent_sheet(t, R, L1, L2, W))], 2.0)
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, z=R + t + L2))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Síla", [Force("konec", [-SH6["F"], 0, 0])]))
    return st, parts


def sh6_eval(res, st, wd):
    t, R, L1, L2, W = SH6["T"], SH6["R"], SH6["L1"], SH6["L2"], SH6["W"]
    ztop = R + t + L2
    r3 = _run3d(st, wd)
    s = res.step(0)
    tip3 = np.nonzero(np.abs(r3.mesh.nodes[:, 2] - ztop) < 1e-6)[0]
    tip = _at(res, lambda x, y, z: np.abs(z - ztop) < 1e-6)
    ux3, uz3 = r3.step(0).U[tip3, 0].mean(), r3.step(0).U[tip3, 2].mean()
    ux, uz = s.U[tip, 0].mean(), s.U[tip, 2].mean()
    return [Check("posuv konce ux skořepina / 3D", float(ux), float(ux3), 0.03, "mm"),
            Check("posuv konce uz skořepina / 3D", float(uz), float(uz3), 0.03, "mm"),
            Check("reakce Fx", s.reactions["vetknuti"][0], SH6["F"], 1e-3, "N"),
            Check("rovnováha (1 − nerovnováha)", 1 - s.equilibrium["imbalance"], 1.0, 1e-4, "")]


# ------------------------------------------------------------------ Sh7 plate part from a STEP
def _face_rect(z, L, B):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pln, gp_Pnt, gp_Dir
    return BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, z), gp_Dir(0, 0, 1)), 0, L, 0, B).Face()


def sh7_build(wd: Path, z: float = SH1_T / 2, name: str = "Sh7 plosny dil"):
    from fea_core.selectors import find_edges
    from fea_core.shell.geometry import read_step_surfaces
    step = wd / f"model_{z:g}.step"
    write_step([Part("Plocha", _face_rect(z, SH1_L, SH1_B))], step)
    surf = read_step_surfaces(step)
    st = Study(name=name, geometry=str(step), mesh=MeshSettings(size=5.0))
    st.add_material(STEEL, default=True)
    st.add_selection("vetknuti", edges=find_edges(surf, x=0))
    st.add_selection("konec", edges=find_edges(surf, x=SH1_L))
    st.add_selection("plocha", find_faces(surf))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Síla 100 N", [Force("konec", [0, 0, -SH1_F])]))
    st.load_cases.append(LoadCase("Tlak", [Pressure("plocha", 0.001)]))
    return st, []


def sh7_eval(res, st, wd):
    """Plate part (mid-surface) = cantilever Sh1; the same part as the bottom surface (offset bot) gives the same results."""
    from fea_core.shell import run_shell
    s = res.step(0)
    tip = _at(res, lambda x, y, z: np.abs(x - SH1_L) < 1e-6)
    st2, _ = sh7_build(wd, 0.0, "Sh7 plosny dil (dolni plocha)")
    r2 = run_shell(st2, wd / "run_bot", settings={"enabled": True, "parts": {"Plocha": SH1_T}, "offset": "bot"})
    tip2 = _at(r2, lambda x, y, z: np.abs(x - SH1_L) < 1e-6)
    W = SH1_B * SH1_T ** 2 / 6
    p = [SH1_L / 2, SH1_B / 2, SH1_T / 2]
    q = SH1_B * SH1_L * 0.001
    return [
        Check("průhyb konce (3D model Sh1, 2,9887 mm)", -float(s.U[tip, 2].mean()), 2.98867, 0.02, "mm"),
        Check("σx horní plocha x = L/2 (M/W)", res.probe(p, "sxx", 0, "top")["value"], SH1_F * SH1_L / 2 / W, 0.02,
              "MPa"),
        Check("tlak: reakce Fz = p·A (kladný tlak proti normále plochy)", res.step(1).reactions["vetknuti"][2], q,
              1e-3, "N"),
        Check("offset bot: průhyb konce / střední plocha", float(r2.step(0).U[tip2, 2].mean()),
              float(s.U[tip, 2].mean()), 2e-3, "mm"),
        Check("offset bot: σx horní plocha", r2.probe([SH1_L / 2, SH1_B / 2, 0], "sxx", 0, "top")["value"],
              res.probe(p, "sxx", 0, "top")["value"], 5e-3, "MPa"),
        Check("rovnováha (1 − nerovnováha)", 1 - s.equilibrium["imbalance"], 1.0, 1e-4, ""),
    ]


SHELL_CASES = {c.key: c for c in [
    ShellCase("Sh1", "Konzola z desky 200×40×5, síla na konci", sh1_build, sh1_eval),
    ShellCase("Sh2", "Čtvercová deska kloubově po obvodu, rovnoměrný tlak", sh2_build, sh2_eval),
    ShellCase("Sh3", "Válcová skořepina s vnitřním tlakem (σθ = pR/t)", sh3_build, sh3_eval),
    ShellCase("Sh4", "Deska s dírou – skořepina proti 3D modelu", sh4_build, sh4_eval),
    ShellCase("Sh5", "T-nosník ze dvou desek – skořepina proti 3D modelu", sh5_build, sh5_eval),
    ShellCase("Sh6", "Plech s ohybem 90° – skořepina proti 3D modelu", sh6_build, sh6_eval),
    ShellCase("Sh7", "Plošný díl ze STEP (tloušťka zadaná, offset střed / dolní plocha)", sh7_build, sh7_eval,
              {"enabled": True, "parts": {"Plocha": SH1_T}}),
]}


def run(keys: list[str]) -> bool:
    from fea_core.shell import run_shell
    all_ok = True
    report = {}
    for key in keys:
        case = SHELL_CASES[key]
        wd = OUT / f"shell_{key}"
        wd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"\n=== {key}: {case.title}")
        try:
            study, parts = case.build(wd)
            study.save(wd / "study.json")
            res = run_shell(study, wd / "run", parts=parts, settings=case.settings)
            t_shell = time.time() - t0
            checks = case.evaluate(res, study, wd)
        except Exception as e:  # noqa: BLE001 – continue with the other cases
            import traceback
            traceback.print_exc()
            print(f"  CHYBA: {e}")
            all_ok = False
            report[key] = {"error": str(e)}
            continue
        dt = time.time() - t0
        print(f"  síť: {len(res.mesh.nodes)} uzlů, {len(res.mesh.tri)} prvků S6, skořepiny {t_shell:.1f} s, "
              f"celkem {dt:.1f} s")
        for w in res.warnings:
            print(f"  varování: {w}")
        for c in checks:
            flag = "OK  " if c.ok else "CHYBA"
            print(f"  [{flag}] {c.name:<52} {c.value:12.6g} {c.unit:<4} ref {c.ref:12.6g}  "
                  f"odch. {100 * c.error:6.3f} % (limit {100 * c.tol:g} %)")
            all_ok &= c.ok
        report[key] = {"title": case.title, "time_s": dt, "nodes": int(len(res.mesh.nodes)),
                       "elements": int(len(res.mesh.tri)), "warnings": res.warnings,
                       "checks": [{"name": c.name, "value": float(c.value), "ref": float(c.ref),
                                   "error": float(c.error), "tol": c.tol, "unit": c.unit, "ok": bool(c.ok)}
                                  for c in checks]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shell_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nVÝSLEDEK:", "všechny kontroly prošly" if all_ok else "některé kontroly NEPROŠLY")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1:] or list(SHELL_CASES)) else 1)
