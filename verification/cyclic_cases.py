"""Cyclic symmetry (F9 C3, fea_core.cyclic) – verification cases CY1, CY2.

CY1 statics, one study with two 60° sectors (N = 6) of two parts:
    "Priruba" – flange ring ri 20, ro 100, t 10 with one hole Ø20 at r = 60 in the middle of the sector, clamped bore;
        (a) rotation ω = 1000 rad/s about z, (b) pressure 5 MPa on the top face + 20 MPa in the hole.
        Reference = THE FULL 360° MODEL on the same mesh (the sector mesh rotated 6×, nodes of the cut faces merged), computed
        by ccx without any constraint equations: displacements and stresses ≤ 1e-5 of the maximum, reactions of the full model
        = 6 × the per-sector reactions (axial force), applied load exactly.
    "Kotouc" – thin disc ri 20, ro 100, t 4 (z 30–34), cylindrical support of the bore only tangential (u_θ = 0 – holds the
        rigid rotation, no reaction for a radial load), u_z = 0 on its bottom face; (a) ω = 1000 rad/s: plane-stress
        Lamé solution σθ(ri), σθ(ro), u_r(ro) within 1 % (thin disc, 3D deviation ~ (t/r)², as V14).
CY2 natural frequencies of a free slender ring (mean radius 100, radial width 3, height 12, N = 8): ccx cyclic symmetry by
    nodal diameters 0–4 against the full 360° model on the same mesh (every cyclic frequency is in the full spectrum
    within 1e-5 with the right multiplicity) and the in-plane bending frequencies of a thin ring (Hoppe / Love):
        ω_n = √(EI/(ρA R⁴)) · n(n² − 1)/√(n² + 1), I = h·w³/12 (bending in the plane of the ring), n = 2, 3, 4 within 2 %.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

from fea_core.cyclic import CyclicSymmetry
from fea_core.geometry import Part
from fea_core.results import Results, tensor
from fea_core.selectors import find_faces
from fea_core.study import (Study, Fixed, Displacement, CylindricalSupport, Pressure, Rotation, LoadCase, MeshSettings)
from verification.cases import Case, Check, E, NU, RHO, cyl, solid, _study

N1 = 6
TH1 = 2 * math.pi / N1
RI, RO, T1 = 20.0, 100.0, 10.0
RH, RHOLE = 60.0, 10.0
ZD, TD = 30.0, 4.0
W = 1000.0


def wedge(r: float, z0: float, h: float, angle: float):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, z0), gp_Dir(0, 0, 1)), r, h, angle).Shape()


def ring_sector(ri: float, ro: float, z0: float, h: float, angle: float):
    return BRepAlgoAPI_Cut(wedge(ro, z0, h, angle), cyl((0, 0, z0 - 1), (0, 0, 1), ri, h + 2)).Shape()


def cy1_build(wd: Path):
    flange = BRepAlgoAPI_Cut(ring_sector(RI, RO, 0.0, T1, TH1),
                             cyl((RH * math.cos(TH1 / 2), RH * math.sin(TH1 / 2), -1), (0, 0, 1), RHOLE, T1 + 2)).Shape()
    geo = [Part("Priruba", solid(flange)), Part("Kotouc", solid(ring_sector(RI, RO, ZD, TD, TH1)))]
    st, parts = _study("CY1 cyklicka symetrie", wd / "model.step", geo, MeshSettings(size=6.0))
    left = [f for f in find_faces(parts, y=0)]
    nr = np.array([-math.sin(TH1), math.cos(TH1), 0.0])
    right = [f for p in parts for f in find_faces([p]) if f.type == "plane" and abs(abs(np.dot(f.normal, nr)) - 1) < 1e-9
             and abs(np.dot(f.centroid, nr)) < 1e-6]
    st.add_selection("rez_L", left)
    st.add_selection("rez_P", right)
    st.add_selection("otvor_hridele", [f for f in find_faces(parts, part="Priruba", radius=RI)])
    st.add_selection("otvor_sroubu", [f for f in find_faces(parts, part="Priruba", radius=RHOLE)])
    st.add_selection("horni", find_faces(parts, part="Priruba", z=T1))
    st.add_selection("kotouc_otvor", [f for f in find_faces(parts, part="Kotouc", radius=RI)])
    st.add_selection("kotouc_spodek", find_faces(parts, part="Kotouc", z=ZD))
    st.constraints += [Fixed("otvor_hridele"),
                       CylindricalSupport("kotouc_otvor", radial=False, tangential=True, axial=False),
                       Displacement("kotouc_spodek", uz=0.0)]
    st.load_cases += [
        LoadCase("Otaceni w", [Rotation([0, 0, 0], [0, 0, 1], omega=W)]),
        LoadCase("Tlak", [Pressure("horni", 5.0), Pressure("otvor_sroubu", 20.0)]),
    ]
    st.cyclic = CyclicSymmetry(sectors=N1, left="rez_L", right="rez_P")
    return st, parts


def _full_model(study: Study, res: Results, wd: Path) -> Results:
    """The full 360° model on the rotated sector mesh, computed without cyclic equations (independent reference)."""
    from fea_core.cyclic.expand import expand_mesh
    from fea_core.solve import run_study
    full_mesh, _, _ = expand_mesh(res.mesh, res.info["cyclic"])
    st = copy.deepcopy(study)
    st.cyclic = None
    st.selections.pop(study.cyclic.left, None)
    st.selections.pop(study.cyclic.right, None)
    st.modal.modes = 40 if study.modal.enabled else st.modal.modes
    return run_study(st, wd, mesh=full_mesh)


def cy_run(study: Study, wd: Path, parts):
    from fea_core.solve import run_study
    res = run_study(study, wd / "sector", parts=parts)
    res._full = _full_model(study, res, wd / "full")
    return res


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def cy1_eval(res: Results, st: Study):
    from fea_core.cyclic.expand import expand_results
    full = res._full
    ex = expand_results(res)
    out = []
    pi = res.mesh.part_names.index("Priruba")
    nodes_p = np.unique(ex.mesh.tets[ex.mesh.tet_part == pi])
    for k, s in enumerate(res.steps):
        sf, se = full.step(k), ex.step(k)
        out += [Check(f"{s.name}: |U sektor − U 360°| / max|U| (+1)", 1 + _rel(se.U[nodes_p], sf.U[nodes_p]), 1.0, 1e-5),
                Check(f"{s.name}: |σ sektor − σ 360°| / max|σ| (+1)", 1 + _rel(se.S[nodes_p], sf.S[nodes_p]), 1.0, 1e-4),
                Check(f"{s.name}: max σ_vM sektor = 360°", float(s.mises[np.unique(res.mesh.tets[res.mesh.tet_part == pi])].max()),
                      float(sf.mises[nodes_p].max()), 1e-5, "MPa"),
                Check(f"{s.name}: rovnováha sektoru (nerovnováha + 1)", 1 + s.equilibrium["imbalance"], 1.0, 1e-5)]
    s1, f1 = res.step(1), full.step(1)
    a_top = (RO ** 2 - RI ** 2) * TH1 / 2 - math.pi * RHOLE ** 2
    out += [Check("Tlak: reakce Fz 360° = 6 × sektor", float(f1.reactions["otvor_hridele"][2]),
                  N1 * float(s1.reactions["otvor_hridele"][2]), 1e-5, "N"),
            Check("Tlak: reakce Fz sektoru = tlak × plocha", float(s1.reactions["otvor_hridele"][2]), 5.0 * a_top, 1e-4, "N")]
    # disc: plane-stress Lamé (rotating disc with a free bore – the tangential support carries nothing)
    rw = RHO * W ** 2
    sth_i = rw / 4 * ((3 + NU) * RO ** 2 + (1 - NU) * RI ** 2)
    sth_o = rw / 4 * ((1 - NU) * RO ** 2 + (3 + NU) * RI ** 2)
    phi = TH1 / 2

    def hoop(r):
        p = [r * math.cos(phi), r * math.sin(phi), ZD + TD / 2]
        T = tensor(res.probe(p, "S", 0))
        t = np.array([-math.sin(phi), math.cos(phi), 0.0])
        return float(t @ T @ t)
    n = res.mesh.nodes
    di = res.mesh.part_names.index("Kotouc")
    disc = np.unique(res.mesh.tets[res.mesh.tet_part == di])
    outer = disc[np.abs(np.hypot(n[disc, 0], n[disc, 1]) - RO) < 1e-3]
    s0 = res.step(0)
    ur = (s0.U[outer, 0] * n[outer, 0] + s0.U[outer, 1] * n[outer, 1]) / RO
    out += [Check("Kotouč ω: σθ(ri) (Lamé)", hoop(RI), sth_i, 0.01, "MPa"),
            Check("Kotouč ω: σθ(ro) (Lamé)", hoop(RO), sth_o, 0.01, "MPa"),
            Check("Kotouč ω: u_r(ro) = ro·σθ(ro)/E", float(ur.mean()), RO * sth_o / E, 0.01, "mm"),
            Check("Kotouč ω: tečná reakce otvoru ≈ 0 (1 + |R|/ρω²V r)",
                  1 + float(np.linalg.norm(s0.reactions["kotouc_otvor"])) / (rw * TH1 / 2 * (RO ** 2 - RI ** 2) * TD * RO),
                  1.0, 1e-3)]
    return out


# ------------------------------------------------------------------ CY2 modal of a free ring
N2 = 8
TH2 = 2 * math.pi / N2
R2, W2, H2 = 100.0, 3.0, 12.0


def cy2_build(wd: Path):
    geo = [Part("Prstenec", solid(ring_sector(R2 - W2 / 2, R2 + W2 / 2, 0.0, H2, TH2)))]
    st, parts = _study("CY2 cyklicka symetrie vlastni frekvence", wd / "model.step", geo, MeshSettings(size=2.0))
    nr = np.array([-math.sin(TH2), math.cos(TH2), 0.0])
    st.add_selection("rez_L", find_faces(parts, y=0))
    st.add_selection("rez_P", [f for f in find_faces(parts) if f.type == "plane"
                               and abs(abs(np.dot(f.normal, nr)) - 1) < 1e-9 and abs(np.dot(f.centroid, nr)) < 1e-6])
    st.modal.enabled = True
    st.modal.modes = 4
    st.cyclic = CyclicSymmetry(sectors=N2, left="rez_L", right="rez_P")
    return st, parts


def cy2_eval(res: Results, st: Study):
    ms = res.modes[0]
    full = np.array(sorted(v for v in res._full.modes[0].values if v > 1.0))
    # every ND gives its lowest k modes → below the lowest "highest computed frequency" of all NDs the cyclic spectrum
    # (each frequency × multiplicity) must be the complete spectrum of the full model
    top = {}
    for f, nd in zip(ms.values, ms.nodal_diameters):
        top[nd] = max(top.get(nd, 0.0), f)
    cut = min(min(top.values()), float(full.max())) * (1 + 1e-6)
    cyc = np.array(sorted(f for f, m in zip(ms.values, ms.multiplicity) for _ in range(m) if 1.0 < f <= cut))
    ref = full[full <= cut]
    worst = float(np.max(np.abs(cyc - ref) / ref)) if len(cyc) == len(ref) else 1.0
    out = [Check(f"počet frekvencí do {cut:.0f} Hz (cyklicky × násobnost = model 360°)", float(len(cyc)), float(len(ref)), 0.0),
           Check("cyklické frekvence = spektrum modelu 360° (max rel. rozdíl + 1)", 1 + worst, 1.0, 1e-5)]
    I = H2 * W2 ** 3 / 12
    A = H2 * W2
    base = math.sqrt(E * I / (RHO * A * R2 ** 4))
    for nd in (2, 3, 4):
        fs = [f for f, k in zip(ms.values, ms.nodal_diameters) if k == nd]
        fn = base * nd * (nd * nd - 1) / math.sqrt(nd * nd + 1) / (2 * math.pi)
        out.append(Check(f"ND {nd}: ohyb prstence v rovině (Love)", float(min(fs)), fn, 0.02, "Hz"))
    return out


CYCLIC_CASES = {
    "CY1": Case("CY1", "Cyklická symetrie: statika sektoru = model 360°, rotující kotouč (Lamé)", cy1_build, cy1_eval,
                run=cy_run),
    "CY2": Case("CY2", "Cyklická symetrie: vlastní frekvence po uzlových průměrech = model 360°, prstenec", cy2_build,
                cy2_eval, run=cy_run),
}
