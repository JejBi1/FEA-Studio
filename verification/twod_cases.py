"""Verification cases of the 2D calculations (a section of the 3D study): axisymmetric CAX6, plane stress CPS6, plane
strain CPE6.

  A1  V4 tube (Lamé, internal pressure) as axisymmetric + comparison with the 3D calculation V4
  A2  thick-walled sphere with internal pressure (Lamé) – axisymmetric from 1/8 of the model
  A3  V3 plate with a hole (Kt Heywood) as plane stress
  A4  thick-walled cylinder in plane strain (Lamé, σz = ν(σr + σθ))
  A5  rotating thin disc (axisymmetric) and a quarter of the disc (plane stress), centrifugal force

Run: python -m verification.twod_cases [A1 A2 …]  (table as verification.run_all, out/twod_report.json)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCP.gp import gp_Pnt

from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.study import Displacement, LoadCase, MeshSettings, Pressure, Rotation
from fea_core.twod.model import TwoDSettings
from verification.cases import (Case, Check, E, NU, RHO, RI, RO, V4_H, P4, V3_R, V3_W, V3_T, _study, box, cyl,
                                solid, v3_build, v4_build)

OUT = Path(__file__).resolve().parent / "out"


# ------------------------------------------------------------------ A1 – V4 tube as axisymmetric
def a1_build(wd: Path):
    st, parts = v4_build(wd)
    st.name = "A1 trubka Lame (axisym)"
    st.twod = TwoDSettings(True, "axisym", [0, 0, 0], [0, 0, 1], [1, 1, 0], size=1.0)   # half-plane at 45° inside the sector
    return st, parts


def _lame_tube():
    a = P4 * RI ** 2 / (RO ** 2 - RI ** 2)
    sth_i, sth_o = a * (1 + RO ** 2 / RI ** 2), 2 * a
    return a, sth_i, sth_o


_V4_3D: dict = {}


def _v4_3d(wd: Path) -> dict:
    """3D result of V4 on the same geometry (run_study) – σθ, von Mises and u_r at ri at half the height."""
    if not _V4_3D:
        from fea_core.solve import run_study
        from verification.cases import _hoop
        (wd / "v4_3d").mkdir(parents=True, exist_ok=True)
        st, _ = v4_build(wd / "v4_3d")
        res = run_study(st, wd / "v4_3d" / "run")
        phi = math.radians(45)
        hi, vmi = _hoop(res, RI, phi, V4_H / 2)
        ho, _ = _hoop(res, RO, phi, V4_H / 2)
        p = np.array([RI * math.cos(phi), RI * math.sin(phi), V4_H / 2])
        u = res.probe(p, "U")
        _V4_3D.update(sth_i=hi, sth_o=ho, vm_i=vmi, u_i=float(u[0] * math.cos(phi) + u[1] * math.sin(phi)))
    return _V4_3D


def a1_eval(res, st, wd: Path):
    a, sth_i, sth_o = _lame_tube()
    vm_i = math.sqrt(sth_i ** 2 + P4 ** 2 + sth_i * P4)
    u_i = RI / E * (sth_i + NU * P4)
    zc = V4_H / 2
    hi = float(res.probe([RI, zc], "st"))
    ho = float(res.probe([RO, zc], "st"))
    vmi = float(res.probe([RI, zc], "mises"))
    ui = float(res.probe([RI, zc], "U")[0])
    r3 = _v4_3d(wd)
    return [
        Check("σθ na ri (Lamé)", hi, sth_i, 0.01, "MPa"),
        Check("σθ na ro (Lamé)", ho, sth_o, 0.01, "MPa"),
        Check("von Mises na ri", vmi, vm_i, 0.01, "MPa"),
        Check("u_r(ri)", ui, u_i, 0.01, "mm"),
        Check("σr na ri = −p", -float(res.probe([RI, zc], "sr")), P4, 0.01, "MPa"),
        Check("σθ na ri proti 3D (V4)", hi, r3["sth_i"], 0.03, "MPa"),
        Check("von Mises na ri proti 3D (V4)", vmi, r3["vm_i"], 0.03, "MPa"),
        Check("u_r(ri) proti 3D (V4)", ui, r3["u_i"], 0.01, "mm"),
    ]


# ------------------------------------------------------------------ A2 – koule
def a2_build(wd: Path):
    shell = BRepAlgoAPI_Cut(BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 0), RO).Shape(),
                            BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 0), RI).Shape()).Shape()
    octant = solid(BRepAlgoAPI_Common(shell, box(0, 0, 0, RO + 1, RO + 1, RO + 1)).Shape())
    st, parts = _study("A2 koule Lame", wd / "model.step", [Part("Koule", octant)], MeshSettings(size=1.0))
    st.add_selection("sym_x", find_faces(parts, x=0))
    st.add_selection("sym_y", find_faces(parts, y=0))
    st.add_selection("sym_z", find_faces(parts, z=0))
    st.add_selection("vnitrni", find_faces(parts, type="sphere",
                                           where=lambda p: abs(np.linalg.norm(p["point"]) - RI) < 1e-3))
    st.constraints += [Displacement("sym_x", ux=0.0), Displacement("sym_y", uy=0.0), Displacement("sym_z", uz=0.0)]
    st.load_cases.append(LoadCase("Vnitřní tlak 100 MPa", [Pressure("vnitrni", P4)]))
    st.twod = TwoDSettings(True, "axisym", [0, 0, 0], [0, 0, 1], [1, 1, 0])
    return st, parts


def a2_eval(res, st, wd: Path):
    k = RO ** 3 - RI ** 3
    sth_i = P4 * (RO ** 3 + 2 * RI ** 3) / (2 * k)
    sth_o = P4 * 3 * RI ** 3 / (2 * k)
    u_i = P4 * RI ** 3 / k * RI / E * ((1 - 2 * NU) + (1 + NU) * RO ** 3 / (2 * RI ** 3))
    c = math.cos(math.radians(45))
    pi_, po = np.array([RI * c, RI * c]), np.array([RO * c, RO * c])          # polar angle 45°
    u = res.probe(pi_, "U")
    eq = res.probe([RI, 0.0], "U")                                            # equator (z = 0)
    return [
        Check("σθ na ri, polární úhel 45° (Lamé koule)", float(res.probe(pi_, "st")), sth_i, 0.01, "MPa"),
        Check("σθ na ro, polární úhel 45°", float(res.probe(po, "st")), sth_o, 0.01, "MPa"),
        Check("σθ na ri, rovník", float(res.probe([RI, 0.0], "st")), sth_i, 0.01, "MPa"),
        Check("|u|(ri), polární úhel 45°", float(np.linalg.norm(u)), u_i, 0.01, "mm"),
        Check("u_r(ri) na rovníku", float(eq[0]), u_i, 0.01, "mm"),
        Check("σ meridiánové = σθ (tečné v bodě 45°)",
              float(_tangential(res.probe(pi_, "S"), pi_)), sth_i, 0.01, "MPa"),
    ]


def _tangential(S, p):
    """Meridional stress at the point p (r, z): the component of the tensor (r, z) in the direction of the tangent to the circle."""
    t = np.array([-p[1], p[0]]) / np.linalg.norm(p)
    T = np.array([[S[0], S[3]], [S[3], S[1]]])
    return t @ T @ t


# ------------------------------------------------------------------ A3 – V3 plate as plane stress
def a3_build(wd: Path):
    st, parts = v3_build(wd)
    st.name = "A3 deska s otvorem (rovinna napjatost)"
    st.twod = TwoDSettings(True, "plane_stress", [0, 0, V3_T / 2], [0, 0, 1], [1, 0, 0], thickness=V3_T)
    return st, parts


def a3_eval(res, st, wd: Path):
    kt = 3 - 3.14 * 0.1 + 3.667 * 0.01 - 1.527 * 0.001
    ref = kt * 100 * 100 / 90
    s = res.step(0)
    return [
        Check("σx max na otvoru (Kt Heywood, d/W = 0,1)", float(res.probe([0.0, V3_R], "sx")), ref, 0.03, "MPa"),
        Check("σx max v uzlech = na otvoru", float(s["S"][:, 0].max()), ref, 0.03, "MPa"),
        Check("reakce Fx sym_x (100 MPa·50·2,5)", -float(s["reactions"]["sym_x"][0]), 100 * V3_W * V3_T, 0.001, "N"),
    ]


# ------------------------------------------------------------------ A4 – cylinder in plane strain
A4_T = 1.0


def a4_build(wd: Path):
    ring = BRepAlgoAPI_Cut(cyl((0, 0, 0), (0, 0, 1), RO, V4_H), cyl((0, 0, -1), (0, 0, 1), RI, V4_H + 2)).Shape()
    quarter = solid(BRepAlgoAPI_Common(ring, box(0, 0, -1, RO + 1, RO + 1, V4_H + 2)).Shape())
    st, parts = _study("A4 valec rovinna deformace", wd / "model.step", [Part("Valec", quarter)], MeshSettings(size=1.0))
    st.add_selection("sym_x", find_faces(parts, x=0))
    st.add_selection("sym_y", find_faces(parts, y=0))
    st.add_selection("vnitrni", find_faces(parts, radius=RI))
    st.constraints += [Displacement("sym_x", ux=0.0), Displacement("sym_y", uy=0.0)]
    st.load_cases.append(LoadCase("Vnitřní tlak 100 MPa", [Pressure("vnitrni", P4)]))
    st.twod = TwoDSettings(True, "plane_strain", [0, 0, V4_H / 2], [0, 0, 1], [1, 0, 0], thickness=A4_T)
    return st, parts


def _hoop2d(res, r, phi=math.radians(45)):
    p = np.array([r * math.cos(phi), r * math.sin(phi)])
    S = res.probe(p, "S")
    t = np.array([-math.sin(phi), math.cos(phi)])
    T = np.array([[S[0], S[3]], [S[3], S[1]]])
    return float(t @ T @ t), float(S[2]), p


def a4_eval(res, st, wd: Path):
    a, sth_i, sth_o = _lame_tube()
    b = P4 * RI ** 2 * RO ** 2 / (RO ** 2 - RI ** 2)
    u_i = (1 + NU) / E * ((1 - 2 * NU) * a * RI + b / RI)
    hi, szi, p = _hoop2d(res, RI)
    ho, szo, _ = _hoop2d(res, RO)
    u = res.probe(p, "U")
    s = res.step(0)
    return [
        Check("σθ na ri (Lamé)", hi, sth_i, 0.01, "MPa"),
        Check("σθ na ro (Lamé)", ho, sth_o, 0.01, "MPa"),
        Check("σz na ri = ν(σr + σθ)", szi, NU * (sth_i - P4), 0.01, "MPa"),
        Check("σz na ro = ν(σr + σθ)", szo, NU * sth_o, 0.01, "MPa"),
        Check("u_r(ri) rovinná deformace", float(np.linalg.norm(u)), u_i, 0.01, "mm"),
        Check("reakce |Fx| sym_x = p·ri·t", abs(float(s["reactions"]["sym_x"][0])), P4 * RI * A4_T, 0.001, "N"),
    ]


# ------------------------------------------------------------------ A5 – rotating disc
A5_R, A5_H, A5_W = 100.0, 4.0, 1000.0           # mm, mm, rad/s


def a5_build(wd: Path):
    disk = cyl((0, 0, 0), (0, 0, 1), A5_R, A5_H)
    st, parts = _study("A5 rotujici disk (axisym)", wd / "model.step", [Part("Disk", solid(disk))],
                       MeshSettings(size=2.0))
    st.add_selection("dno", find_faces(parts, z=0))
    st.constraints.append(Displacement("dno", uz=0.0))
    st.load_cases.append(LoadCase("Otáčení 1000 rad/s", [Rotation([0, 0, 0], [0, 0, 1], omega=A5_W)]))
    st.twod = TwoDSettings(True, "axisym", [0, 0, 0], [0, 0, 1], [1, 0, 0])
    return st, parts


def _disk_ref(r):
    c = RHO * A5_W ** 2
    sr = (3 + NU) / 8 * c * (A5_R ** 2 - r ** 2)
    st = (3 + NU) / 8 * c * A5_R ** 2 - (1 + 3 * NU) / 8 * c * r ** 2
    return sr, st


def a5_eval(res, st, wd: Path):
    zc = A5_H / 2
    s0r, s0t = _disk_ref(0.0)
    _, sRt = _disk_ref(A5_R)
    u_R = A5_R / E * sRt
    s50r, s50t = _disk_ref(50.0)
    return [
        Check("σr = σθ ve středu (tenký disk)", float(res.probe([0.0, zc], "sr")), s0r, 0.01, "MPa"),
        Check("σθ ve středu", float(res.probe([0.0, zc], "st")), s0t, 0.01, "MPa"),
        Check("σr v r = 50", float(res.probe([50.0, zc], "sr")), s50r, 0.01, "MPa"),
        Check("σθ v r = 50", float(res.probe([50.0, zc], "st")), s50t, 0.01, "MPa"),
        Check("σθ na obvodu", float(res.probe([A5_R, zc], "st")), sRt, 0.01, "MPa"),
        Check("u_r na obvodu", float(res.probe([A5_R, zc], "U")[0]), u_R, 0.01, "mm"),
    ]


def a5p_build(wd: Path):
    q = solid(BRepAlgoAPI_Common(cyl((0, 0, 0), (0, 0, 1), A5_R, A5_H), box(0, 0, -1, A5_R + 1, A5_R + 1, A5_H + 2)).Shape())
    st, parts = _study("A5P rotujici ctvrt disku (rovinna napjatost)", wd / "model.step", [Part("Disk", q)],
                       MeshSettings(size=2.0))
    st.add_selection("sym_x", find_faces(parts, x=0))
    st.add_selection("sym_y", find_faces(parts, y=0))
    st.constraints += [Displacement("sym_x", ux=0.0), Displacement("sym_y", uy=0.0)]
    st.load_cases.append(LoadCase("Otáčení 1000 rad/s", [Rotation([0, 0, 0], [0, 0, 1], omega=A5_W)]))
    st.twod = TwoDSettings(True, "plane_stress", [0, 0, A5_H / 2], [0, 0, 1], [1, 0, 0], thickness=A5_H)
    return st, parts


def a5p_eval(res, st, wd: Path):
    s0r, _ = _disk_ref(0.0)
    s50r, s50t = _disk_ref(50.0)
    _, sRt = _disk_ref(A5_R)
    h50, _, _ = _hoop2d(res, 50.0)
    hR, _, p = _hoop2d(res, A5_R)
    s = res.step(0)
    # reaction sym_x (x = 0): force in x carried by the section = ∫σθ dr · t = ∫ … (the y half of the disc) → ρω²·t·R³/3
    Fx = RHO * A5_W ** 2 * A5_H * A5_R ** 3 / 3
    return [
        Check("σx ve středu (= σr = σθ)", float(res.probe([0.0, 0.0], "sx")), s0r, 0.01, "MPa"),
        Check("σθ v r = 50", h50, s50t, 0.01, "MPa"),
        Check("σθ na obvodu", hR, sRt, 0.01, "MPa"),
        Check("u_r na obvodu", float(np.linalg.norm(res.probe(p, "U"))), A5_R / E * sRt, 0.01, "mm"),
        Check("reakce |Fx| sym_x = ρω²tR³/3", abs(float(s["reactions"]["sym_x"][0])), Fx, 0.002, "N"),
    ]


TWOD_CASES: dict[str, Case] = {
    "A1": Case("A1", "V4 trubka ri 20 / ro 40, p = 100 MPa jako rotačně symetrická úloha (+ 3D V4)", a1_build, a1_eval),
    "A2": Case("A2", "Tlustostěnná koule ri 20 / ro 40, p = 100 MPa (axisym z 1/8 modelu)", a2_build, a2_eval),
    "A3": Case("A3", "V3 deska š. 100 s otvorem Ø10, tah 100 MPa jako rovinná napjatost", a3_build, a3_eval),
    "A4": Case("A4", "Tlustostěnný válec ri 20 / ro 40, p = 100 MPa, rovinná deformace", a4_build, a4_eval),
    "A5": Case("A5", "Rotující disk R 100, t 4, ω = 1000 rad/s (axisym)", a5_build, a5_eval),
    "A5P": Case("A5P", "Rotující čtvrt disku R 100, t 4, ω = 1000 rad/s (rovinná napjatost)", a5p_build, a5p_eval),
}


def run(keys: list[str]) -> bool:
    from fea_core.twod.run import run_2d
    all_ok, report = True, {}
    for key in keys:
        case = TWOD_CASES[key]
        wd = OUT / f"twod_{key}"
        wd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"\n=== {key}: {case.title}")
        try:
            study, parts = case.build(wd)
            res = run_2d(study, wd / "run", parts=parts)
            res.save(wd / "results2d.npz")
            res.render_png(wd / "mises.png")
            checks = case.evaluate(res, study, wd)
        except Exception as e:  # noqa: BLE001 – continue with the other cases
            import traceback
            traceback.print_exc()
            print(f"  CHYBA: {e}")
            all_ok = False
            report[key] = {"error": str(e)}
            continue
        dt = time.time() - t0
        mi = res.info.get("mesh", {})
        print(f"  síť: {mi.get('nodes')} uzlů, {mi.get('elements')} prvků tri6, min SJ "
              f"{(mi.get('min_scaled_jacobian') or 0):.3f}, čas {dt:.1f} s, nerovnováha "
              f"{max(s['equilibrium']['imbalance'] for s in res.steps):.1e}")
        for w in res.warnings:
            print(f"  varování: {w}")
        for c in checks:
            flag = "OK  " if c.ok else "CHYBA"
            print(f"  [{flag}] {c.name:<52} {c.value:12.6g} {c.unit:<4} ref {c.ref:12.6g}  "
                  f"odch. {100 * c.error:6.3f} % (limit {100 * c.tol:g} %)")
            all_ok &= c.ok
        report[key] = {"title": case.title, "time_s": dt, "nodes": mi.get("nodes"), "elements": mi.get("elements"),
                       "warnings": res.warnings,
                       "checks": [{"name": c.name, "value": float(c.value), "ref": float(c.ref), "error": float(c.error),
                                   "tol": c.tol, "unit": c.unit, "ok": bool(c.ok)} for c in checks]}
    (OUT / "twod_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nVÝSLEDEK:", "všechny kontroly prošly" if all_ok else "některé kontroly NEPROŠLY")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1:] or list(TWOD_CASES)) else 1)
