"""Verification of contact and friction (stage F8.5, fea_core.contacts) – cases K, R of the F8 spec (chapter 7). Steel E = 210 GPa, ν = 0.3.

K1  2× block 20×20×10, pressure 10 MPa, contact without friction, symmetric supports: p = 10 MPa everywhere, u_z = 2ph/E + p/K;
    a) separate meshes (non-conforming), b) conforming mesh with the joint split.
K4  press fit shaft Ø20 / hub Ø40, Δd = 0.02 (*CLEARANCE −0.01 radially), an eighth of the model: p = 78.75 MPa,
    σ_t,hub = 131.25 MPa, u_r of the hub +7.375 µm, shaft −2.625 µm (Lamé, plane stress = 3D with free ends).
K7  cantilever 10×10×200 over a rigid cylinder R 5 at x = 150 with a gap 0.5, F = 200 N at the end: contact from F_c = 51.85 N,
    reaction of the cylinder 222.2 N (E–B, statically indeterminate), K7n the same with NLGEOM (check of the equilibrium at the deformed position).
K8  blocks with an initial penetration 0.01 of the pair "without friction" → an understandable error before the calculation.
R1  block pressed by N = 1000 N, moved by a rigid constraint by 0.05 mm, μ 0.3: friction plateau T = μN = 300 N.
G1  block on a support only in compression (GAPUNI), pressure 10 MPa: reaction = 4000 N, displacement = ph/E.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut

from fea_core.study import (Displacement, Pressure, LoadCase, MeshSettings, Fixed, Force, Material, RemoteSupport)
from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.contacts.model import ContactPair, LoadStep, Motion
from verification.cases import Case, Check, E, _study, box, cyl, solid


def _blocks(wd: Path, name: str, gap: float = 0.0, size: float = 2.5):
    st, parts = _study(name, wd / "model.step",
                       [Part("A", box(0, 0, 0, 20, 20, 10)), Part("B", box(0, 0, 10 + gap, 20, 20, 10))],
                       MeshSettings(size=size))
    st.add_selection("dno", find_faces(parts, z=0))
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("y0", find_faces(parts, y=0))
    st.add_selection("vrch", find_faces(parts, z=20 + gap))
    return st, parts


def k1_build(wd: Path, split: str = "separate"):
    st, parts = _blocks(wd, "K1 patch test kontaktu")
    st.constraints += [Displacement("dno", uz=0.0), Displacement("x0", ux=0.0), Displacement("y0", uy=0.0)]
    st.load_cases.append(LoadCase("p", [Pressure("vrch", 10.0)]))
    st.contacts.append(ContactPair("A-B", "A", "B", kind="frictionless", stiffness=1e6))
    st.contact_settings.split = split
    return st, parts


def k1_eval(res, st):
    s = res.steps[0]
    m = res.mesh
    sl = np.unique(m.surfaces[m.report["contacts"][0]["slave"]])
    p = s.fields["CPRESS"][sl]
    top = np.nonzero(np.abs(m.nodes[:, 2] - 20) < 1e-6)[0]
    uz = float(s.U[top, 2].mean())
    c = s.contacts["A-B"]
    return [Check("tlak ve styku – průměr (10 MPa)", float(p.mean()), 10.0, 1e-3, "MPa"),
            Check("tlak ve styku – max. odchylka", float(np.abs(p - 10).max()) / 10 + 1, 1.0, 5e-3, ""),
            Check("normálová síla páru = p·A", c["Fn_N"], 4000.0, 1e-3, "N"),
            Check("u_z vrchu = 2ph/E + p/K", -uz, 2 * 10 * 10 / E + 10 / 1e6, 1e-3, "mm"),
            Check("rovnováha (1 + nerovnováha)", 1 + s.equilibrium["imbalance"], 1.0, 1e-5, "")]


def k1c_build(wd: Path):
    st, parts = k1_build(wd, "conformal_split")
    st.name = "K1c patch test – konformni spara"
    return st, parts


# ------------------------------------------------------------------ K4 press fit
RI, RO, LH = 10.0, 20.0, 10.0


def k4_build(wd: Path):
    q = box(0, 0, 0, 30, 30, LH)
    shaft = solid(BRepAlgoAPI_Common(cyl((0, 0, 0), (0, 0, 1), RI, LH), q).Shape())
    hub = solid(BRepAlgoAPI_Common(BRepAlgoAPI_Cut(cyl((0, 0, 0), (0, 0, 1), RO, LH),
                                                   cyl((0, 0, 0), (0, 0, 1), RI, LH)).Shape(), q).Shape())
    st, parts = _study("K4 lisovany spoj", wd / "model.step", [Part("Hridel", shaft), Part("Naboj", hub)],
                       MeshSettings(size=1.25))
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("y0", find_faces(parts, y=0))
    st.add_selection("z0", find_faces(parts, z=0))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0), Displacement("z0", uz=0.0)]
    st.load_cases.append(LoadCase("Lis", []))
    st.contacts.append(ContactPair("lis", "Hridel", "Naboj", kind="interference", clearance=-0.01))
    st.contact_settings.detect_tol = 0.05
    return st, parts


def k4_eval(res, st):
    s = res.steps[0]
    m = res.mesh
    X = m.nodes
    r = np.hypot(X[:, 0], X[:, 1])
    hub = np.unique(m.tets[m.tet_part == m.part_names.index("Naboj")])
    shaft = np.unique(m.tets[m.tet_part == m.part_names.index("Hridel")])
    mid = np.abs(X[:, 2] - 0.0) < 1e-6                       # plane of symmetry z = 0 (middle of the length)
    hi = hub[(np.abs(r[hub] - RI) < 1e-3) & mid[hub]]
    sh = shaft[(np.abs(r[shaft] - RI) < 1e-3) & mid[shaft]]
    ho = hub[(np.abs(r[hub] - RO) < 1e-3) & mid[hub]]
    ur = lambda idx: float(np.mean((s.U[idx, 0] * X[idx, 0] + s.U[idx, 1] * X[idx, 1]) / r[idx]))  # noqa: E731
    # circumferential stress on the inner surface of the hub
    t = np.arctan2(X[hi, 1], X[hi, 0])
    sxx, syy, sxy = s.S[hi, 0], s.S[hi, 1], s.S[hi, 3]
    st_t = sxx * np.sin(t) ** 2 + syy * np.cos(t) ** 2 - 2 * sxy * np.sin(t) * np.cos(t)
    c = s.contacts["lis"]
    area = math.pi / 2 * RI * LH                               # quarter of the cylinder, half of the length
    p_ref = 0.01 * E * (RO ** 2 - RI ** 2) / (2 * RI * RO ** 2)       # δ_r·E·(ro² − ri²)/(2·ri·ro²) = 78,75
    return [Check("tlak ve spoji p (vážený průměr uzlového CPRESS)", c["p_mean_MPa"], p_ref, 2.5e-2, "MPa"),
            Check("σ_t náboje na vnitřním povrchu", float(st_t.mean()), 131.25, 5e-3, "MPa"),
            Check("u_r náboje vnitřní povrch", ur(hi) * 1000, 7.375, 1e-2, "µm"),
            Check("u_r hřídele", -ur(sh) * 1000, 2.625, 2e-2, "µm"),
            Check("u_r náboje vnější povrch (Lamé 2p·ri²·ro/(E(ro² − ri²)))", ur(ho) * 1000,
                  2 * p_ref * RI ** 2 * RO / (E * (RO ** 2 - RI ** 2)) * 1000, 1e-2, "µm")]


# ------------------------------------------------------------------ K7 cantilever on a cylinder
LB, AX, GAP, RC = 200.0, 150.0, 0.5, 5.0
I_K7 = 10.0 ** 4 / 12


def k7_build(wd: Path, nlgeom: bool = False):
    roller = cyl((AX, -5.0, -GAP - RC), (0, 1, 0), RC, 20.0)
    st, parts = _study("K7n konzola na valci NLGEOM" if nlgeom else "K7 konzola na valci", wd / "model.step",
                       [Part("Nosnik", box(0, 0, 0, LB, 10, 10)), Part("Valec", roller)], MeshSettings(size=2.5))
    st.materials["Tuhý válec"] = Material("Tuhý válec", 2.1e8, 0.3, 7.85e-9)
    st.part_materials["Valec"] = "Tuhý válec"
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=LB))
    st.add_selection("valec_cela", find_faces(parts, y=-5.0) + find_faces(parts, y=15.0))
    st.constraints += [Fixed("vetknuti"), Fixed("valec_cela")]
    st.load_cases.append(LoadCase("F 200 N", [Force("konec", [0, 0, -200.0])]))
    st.contacts.append(ContactPair("nosnik-valec", "Nosnik", "Valec", kind="frictionless"))
    st.contact_settings.detect_tol = 1.0
    st.contact_settings.nlgeom = nlgeom
    return st, parts


def k7_eval(res, st):
    s = res.steps[0]
    F = 200.0
    a = AX
    k = 3 * E * I_K7 / a ** 3                                   # stiffness of the cantilever at the place of the cylinder
    # cantilever with a support with a gap: R from the compatibility at x = a: (F·a²(3L − a)/(6EI) − R·a³/(3EI)) = gap
    R = (F * a ** 2 * (3 * LB - a) / (6 * E * I_K7) - GAP) / (a ** 3 / (3 * E * I_K7))
    Fc = GAP * 6 * E * I_K7 / (a ** 2 * (3 * LB - a))
    c = s.contacts["nosnik-valec"]
    return [Check("síla ve styku (E–B, mezera 0,5)", c["Fn_N"], R, 1.5e-2, "N"),
            Check("mez styku F_c (informativně 51,85 N)", Fc, 51.85, 1e-3, "N"),
            Check("rovnováha (1 + nerovnováha)", 1 + s.equilibrium["imbalance"], 1.0, 1e-3, "")]


def k7n_build(wd: Path):
    return k7_build(wd, nlgeom=True)


def k8_build(wd: Path):
    st, parts = _study("K8 pocatecni prunik", wd / "model.step",
                       [Part("A", box(0, 0, 0, 20, 20, 10)), Part("B", box(0, 0, 9.99, 20, 20, 10))],
                       MeshSettings(size=5.0))
    st.add_selection("dno", find_faces(parts, z=0))
    st.constraints.append(Fixed("dno"))
    st.add_selection("vrch", find_faces(parts, z=19.99))
    st.load_cases.append(LoadCase("p", [Pressure("vrch", 1.0)]))
    st.contacts.append(ContactPair("A-B", "A", "B", kind="frictionless"))
    return st, parts


def r1_build(wd: Path):
    st, parts = _blocks(wd, "R1 plato treni")
    st.add_selection("bok", find_faces(parts, x=0.0)[1:] if False else [f for f in find_faces(parts, x=0.0)
                                                                         if f.part == "B"])
    st.add_selection("dnoA", find_faces(parts, z=0))
    st.constraints = [Fixed("dnoA"), RemoteSupport("bok", [0.0, 10.0, 15.0], dofs=[True, True, False, True, True, True])]
    st.load_cases.append(LoadCase("prit", [Pressure("vrch", 2.5)]))
    st.contacts.append(ContactPair("A-B", "A", "B", kind="frictional", mu=0.3))
    an = st.analysis
    an.kind, an.branches = "nonlinear", False
    an.sequence = [LoadStep("Přítlak", case="prit"),
                   LoadStep("Posun", case="prit", prescribed=[Motion("bok", "disp", 1, [[0.0, 0.0], [1.0, 0.05]])],
                            dt0=0.05, dt_max=0.1)]
    return st, parts


def r1_eval(res, st):
    s = res.steps[-1]
    T = abs(float(s.remote["bok"]["force"][0]))
    c = s.contacts["A-B"]
    return [Check("plató tření T = μN (300 N)", T, 300.0, 1e-3, "N"),
            Check("třecí síla páru Σ τ·A", c["Ft_N"], 300.0, 5e-3, "N"),
            Check("normálová síla páru N", c["Fn_N"], 1000.0, 1e-3, "N")]


def g1_build(wd: Path):
    st, parts = _study("G1 podpora jen v tlaku", wd / "model.step", [Part("A", box(0, 0, 0, 20, 20, 10))],
                       MeshSettings(size=2.5))
    st.add_selection("dno", find_faces(parts, z=0))
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("y0", find_faces(parts, y=0))
    st.add_selection("vrch", find_faces(parts, z=10))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0)]
    st.load_cases.append(LoadCase("p", [Pressure("vrch", 10.0)]))
    st.contacts.append(ContactPair("podlozka", "dno", "", kind="compression_only", stiffness=1e6))
    return st, parts


def g1_eval(res, st):
    s = res.steps[0]
    top = np.nonzero(np.abs(res.mesh.nodes[:, 2] - 10) < 1e-6)[0]
    return [Check("reakce podpory jen v tlaku = p·A", float(s.reactions["podlozka"][2]), 4000.0, 1e-3, "N"),
            Check("posuv vrchu = ph/E + p/K", -float(s.U[top, 2].mean()), 10 * 10 / E + 10 / 1e6, 2e-3, "mm"),
            Check("rovnováha (1 + nerovnováha)", 1 + s.equilibrium["imbalance"], 1.0, 1e-4, "")]


def k8_run(study, workdir, parts):
    """K8: the calculation must end with an understandable error BEFORE the solver; returns the error text (None = the calculation ran)."""
    from types import SimpleNamespace
    from fea_core.solve import run_study, StudyError
    for f in Path(workdir).glob("nl_*.inp"):              # leftovers from an earlier run would distort the check "without ccx"
        f.unlink()
    try:
        run_study(study, workdir, parts=parts)
    except StudyError as e:
        return SimpleNamespace(error=str(e), warnings=[], mesh=None)
    return SimpleNamespace(error=None, warnings=[], mesh=None)


def k8_eval(res, st):
    msg = res.error or ""
    return [Check("K8: výpočet skončil chybou před řešičem (1 = ano)", float(res.error is not None), 1.0, 0.0, ""),
            Check("K8: chyba jmenuje průnik a pár A-B (1 = ano)", float("penetration" in msg and "A-B" in msg), 1.0, 0.0, ""),
            Check("K8: žádný výstup ccx (nl_b1.inp) nevznikl (1 = ano)",
                  float(not list(Path(st.geometry).parent.glob("**/nl_b*.inp"))), 1.0, 0.0, "")]


CONTACT_CASES = {
    "K1": Case("K1", "Patch test kontaktu – samostatné sítě", k1_build, k1_eval),
    "K1c": Case("K1c", "Patch test kontaktu – konformní síť, rozpojená spára", k1c_build, k1_eval),
    "K4": Case("K4", "Lisovaný spoj (Lamé)", k4_build, k4_eval),
    "K7": Case("K7", "Konzola nad tuhým válcem s mezerou", k7_build, k7_eval),
    "K7n": Case("K7n", "Konzola nad válcem s NLGEOM", k7n_build, k7_eval),
    "R1": Case("R1", "Plató tření při řízeném posuvu", r1_build, r1_eval),
    "G1": Case("G1", "Podpora jen v tlaku (GAPUNI)", g1_build, g1_eval),
    "K8": Case("K8", "Počáteční průnik u páru bez tření → chyba před výpočtem", k8_build, k8_eval, run=k8_run),
}


# ------------------------------------------------------------------ K6 bolted joint with a gap with contact
def k6_build(wd: Path):
    from fea_core.joints.detect import detect
    from verification import joint_models as jm
    st, parts = _study("K6 sroub se sparou kontaktem", wd / "model.step", jm.lap_joint(), MeshSettings(size=5.0))
    st.bolts = detect(parts).bolts
    st.add_selection("vetknuti", find_faces(parts, part="Deska B", x=160))
    st.add_selection("konecA", find_faces(parts, part="Deska A", x=0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Smyk 1 kN", [Force("konecA", [-1000.0, 0, 0])]))
    st.load_cases.append(LoadCase("Smyk 10 kN", [Force("konecA", [-10000.0, 0, 0])]))
    st.joints.contact_joint = True
    st.analysis.kind = "nonlinear"
    return st, parts


def k6_eval(res, st):
    from fea_core.joints.assess import bolt_case_forces
    from fea_core.joints.fem import PRELOAD_CASE, bolt_basics
    b = st.bolts[0]
    bas = bolt_basics(st, b)
    pre = res.step(PRELOAD_CASE)
    pf = bolt_case_forces(b, pre.joints["bolts"][b.name])
    pl = pre.joints["bolts"][b.name]["planes"][0]
    s1 = res.step("Smyk 1 kN")
    c1 = bolt_case_forces(b, s1.joints["bolts"][b.name])
    s10 = res.step("Smyk 10 kN")
    pair = next(iter(s10.contacts))
    cc = s10.contacts[pair]
    return [Check("předpětí: síla ve šroubu = F_V", pf["F_SA"], bas["F_Vnom"], 1e-3, "N"),
            Check("předpětí: svěrná síla z kontaktního tlaku = F_V", pl.get("F_clamp_contact", 0.0), bas["F_Vnom"],
                  2e-2, "N"),
            Check("smyk 1 kN (lepí): síla rovinou spáry", c1["planes"][0]["F_Q"], 1000.0, 1e-3, "N"),
            Check("smyk 1 kN: spoj lepí – Ft < μ_T·Fn (max(1, Ft/μFn))", max(1.0, s1.contacts[pair]["Ft_N"]
                  / (0.15 * s1.contacts[pair]["Fn_N"])), 1.0, 1e-9, ""),
            Check("smyk 10 kN (prokluz): Ft = μ_T·Fn", cc["Ft_N"], 0.15 * cc["Fn_N"], 2e-2, "N")]


CONTACT_CASES["K6"] = Case("K6", "Šroubový spoj M10, spára kontaktem (předpětí, lepení, prokluz)", k6_build, k6_eval)


# ------------------------------------------------------------------ KK4 impact of a sphere on a rigid plate (*DYNAMIC + contact)
def kk4_build(wd: Path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Pnt
    R, g = 10.0, 0.01
    q = box(0, 0, -20, 30, 30, 60)
    ball = solid(BRepAlgoAPI_Common(BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, R + g), R).Shape(), q).Shape())
    plate = box(0, 0, -10, 15, 15, 10)
    ms = MeshSettings(size=2.0, local=[{"center": [0.0, 0.0, 0.0], "radius": 1.2, "size": 0.12}])
    st, parts = _study("KK4 raz koule", wd / "model.step", [Part("Koule", ball), Part("Deska", plate)], ms)
    st.materials["Tuhá deska"] = Material("Tuhá deska", 2.1e9, 0.3, 7.85e-9)
    st.part_materials["Deska"] = "Tuhá deska"
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("y0", find_faces(parts, y=0))
    st.add_selection("dno", find_faces(parts, z=-10))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0), Fixed("dno")]
    st.load_cases.append(LoadCase("Bez zatížení", []))
    st.contacts.append(ContactPair("dopad", "Koule", "Deska", kind="frictionless", stiffness=5e8))
    st.contact_settings.detect_tol = 0.5
    an = st.analysis
    an.kind, an.branches = "nonlinear", False
    an.initial_velocity = {"Koule": [0.0, 0.0, -1000.0]}
    an.sequence = [LoadStep("Ráz", case=None, dynamic=True, time=8e-5, alpha=0.0, dt0=5e-7, dt_min=1e-10,
                            dt_max=5e-7, max_inc=2000)]
    return st, parts


def kk4_eval(res, st):
    s = res.steps[-1]
    h = s.curves.get("kontakt:dopad")
    t = np.asarray(h["t"])
    F = 4.0 * np.asarray(h["Fn"])                              # quarter model
    on = t[F > 1e-3 * F.max()]
    E_ = 210000.0 / (1 - 0.3 ** 2)
    m = 7.85e-9 * 4 / 3 * math.pi * 10.0 ** 3
    d = (15 * m * 1000.0 ** 2 / (16 * E_ * math.sqrt(10.0))) ** 0.4
    Fmax = 4 / 3 * E_ * math.sqrt(10.0) * d ** 1.5
    return [Check("max. kontaktní síla (Hertz 2310 N)", float(F.max()), Fmax, 3e-2, "N"),
            Check("doba rázu (Hertz 52,3 µs)", float(on.max() - on.min()) * 1e6, 2.94 * d / 1000.0 * 1e6, 5e-2, "µs")]


CONTACT_CASES["KK4"] = Case("KK4", "Ráz koule na tuhou desku (*DYNAMIC + kontakt)", kk4_build, kk4_eval)


# ------------------------------------------------------------------ KK2 snap-fit hook (C2: NLGEOM + friction + motion)
def _prism_xz(pts, y0, y1):
    """Prism from a polygon in the plane xz extruded in y."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCP.gp import gp_Pnt, gp_Vec
    poly = BRepBuilderAPI_MakePolygon()
    for x, z in pts:
        poly.Add(gp_Pnt(x, y0, z))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    return BRepPrimAPI_MakePrism(face, gp_Vec(0, y1 - y0, 0)).Shape()


KK2_L, KK2_H, KK2_B, KK2_E, KK2_Y, KK2_A, KK2_MU = 15.0, 1.5, 5.0, 2000.0, 0.5, 30.0, 0.3


def kk2_build(wd: Path):
    ta = math.tan(math.radians(KK2_A))
    # wedge: the lower face descends to the right and passes the edge of the hook end (15; 1.5)
    z = lambda x: KK2_H - (x - KK2_L) * ta                      # noqa: E731
    wedge = _prism_xz([(14.0, z(14.0)), (17.5, z(17.5)), (17.5, 4.0), (14.0, 4.0)], -1.0, KK2_B + 1.0)
    st, parts = _study("KK2 zacvakavaci hacek", wd / "model.step",
                       [Part("Hacek", box(0, 0, 0, KK2_L, KK2_B, KK2_H)), Part("Klin", solid(wedge))],
                       MeshSettings(size=0.5))
    st.materials.clear()
    st.default_material = None
    st.add_material(Material("PLA lineárně", KK2_E, 0.35, 1.24e-9), default=True)
    st.materials["Tuhý klín"] = Material("Tuhý klín", 2.0e6, 0.3, 7.85e-9)
    st.part_materials["Klin"] = "Tuhý klín"
    st.add_selection("koren", find_faces(parts, x=0))
    st.add_selection("klin_vrch", find_faces(parts, part="Klin", z=4.0))
    st.constraints += [Fixed("koren"), RemoteSupport("klin_vrch", [15.75, 2.5, 4.0], coupling="rigid")]
    # large sliding: the faces of the pair by selection (the whole ramp of the wedge, the top and the face of the hook) – automatic search within a tolerance
    # around the initial position, otherwise the contact point would be "lost" after the displacement
    n = [math.sin(math.radians(KK2_A)), 0.0, math.cos(math.radians(KK2_A))]   # ± normal of the ramp
    ramp = [f for f in find_faces(parts, part="Klin", type="plane")
            if abs(abs(np.dot(f.normal, n)) - 1) < 1e-6]
    st.add_selection("rampa", ramp)
    st.add_selection("hacek_konec", find_faces(parts, part="Hacek", z=KK2_H) + find_faces(parts, part="Hacek", x=KK2_L))
    st.contacts.append(ContactPair("hacek-klin", "hacek_konec", "rampa", kind="frictional", mu=KK2_MU,
                                   stiffness=50 * KK2_E, slave="a"))
    st.contact_settings.nlgeom = True
    st.contact_settings.detect_tol = 0.3
    an = st.analysis
    an.kind, an.branches = "nonlinear", False
    s_end = KK2_Y / ta
    an.sequence = [LoadStep("Zasunutí", case=None, dt0=0.02, dt_min=1e-6, dt_max=0.05, max_inc=500,
                            prescribed=[Motion("klin_vrch", "disp", 1, [[0.0, 0.0], [1.0, -s_end]])])]
    return st, parts


def kk2_eval(res, st):
    """The corner of the hook end slides along the plane of the wedge → the normal of the contact = the normal of the wedge (the rotation of the end does not apply; the correction
    1.467 N holds for the approach of the hook along the edge of the counterpart): W/P = tan(α + ρ), ρ = arctg μ. P = bending force for the deflection y
    – a wide section (b/h = 3.3) lies between the beam 3EIy/l³ (E) and the plate (E/(1 − ν²)).
    """
    s = res.steps[-1]
    W = abs(float(s.remote["klin_vrch"]["force"][0]))
    P = abs(float(s.remote["klin_vrch"]["force"][2]))
    I = KK2_B * KK2_H ** 3 / 12
    P_beam = 3 * KK2_E * I * KK2_Y / KK2_L ** 3                       # Bayer 1,25 N
    P_plate = P_beam / (1 - 0.35 ** 2)
    curve = s.curves.get("klin_vrch", {})
    return [Check("W/P = tan(α + arctg μ) (1,0610)", W / P, math.tan(math.radians(KK2_A) + math.atan(KK2_MU)), 5e-3, ""),
            Check("P mezi nosníkem a deskou (1 = uvnitř)", float(P_beam * 0.98 <= P <= P_plate * 1.02), 1.0, 0.0, ""),
            Check("montážní síla W (Bayer: P·(μ + tg α)/(1 − μ·tg α) s P z MKP)", W,
                  P * (KK2_MU + math.tan(math.radians(KK2_A))) / (1 - KK2_MU * math.tan(math.radians(KK2_A))), 5e-3, "N"),
            Check("křivka síla–dráha má přírůstky (≥ 5)", float(len(curve.get("t", [])) >= 5), 1.0, 0.0, "")]


CONTACT_CASES["KK2"] = Case("KK2", "Zacvakávací háček – NLGEOM, tření, předepsaný pohyb", kk2_build, kk2_eval)


# ------------------------------------------------------------------ KT1 heat conduction through the contact (spec 4.7)
def _kt1(wd: Path, split: str):
    from fea_core.thermal.model import ThermalCase, FixedTemperature
    st, parts = _blocks(wd, f"KT1 teplo pres kontakt ({split})")
    st.materials["Ocel S235"].k = 53.3
    st.thermal.cases.append(ThermalCase("ustaleny", [FixedTemperature("dno", 0.0), FixedTemperature("vrch", 100.0)]))
    st.constraints += [Displacement("dno", uz=0.0), Displacement("x0", ux=0.0), Displacement("y0", uy=0.0)]
    st.load_cases.append(LoadCase("p", [Pressure("vrch", 1.0)]))
    st.contacts.append(ContactPair("A-B", "A", "B", kind="frictionless"))
    st.contact_settings.split = split
    return st, parts


def kt1_eval(res, st):
    th = res.thermal[0]
    Q = max(abs(h["Q_W"]) for h in th.heat)
    mid = np.abs(res.mesh.nodes[:, 2] - 10) < 1e-6
    return [Check("tepelný tok přes kontakt Q = k·A·ΔT/L (106,6 W)", Q, 53.3 * 400e-6 * 100 / 0.02, 1e-3, "W"),
            Check("teplota rozhraní (50 °C)", float(th.T[mid].mean()), 50.0, 1e-3, "°C")]


CONTACT_CASES["KT1"] = Case("KT1", "Vedení tepla přes kontakt (samostatné sítě, *TIE v tepelné úloze)",
                            lambda wd: _kt1(wd, "separate"), kt1_eval)
CONTACT_CASES["KT1c"] = Case("KT1c", "Vedení tepla přes rozpojenou konformní spáru (rovnice DOF 11)",
                             lambda wd: _kt1(wd, "conformal_split"), kt1_eval)


# ------------------------------------------------------------------ K2 Hertz koule na desce (statika)
def k2_build(wd: Path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Pnt
    R = 10.0
    q = box(0, 0, 0, 30, 30, R)                                  # lower hemisphere, ¼ model
    ball = solid(BRepAlgoAPI_Common(BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, R), R).Shape(), q).Shape())
    plate = box(0, 0, -20, 20, 20, 20)
    ms = MeshSettings(size=2.0, local=[{"center": [0.0, 0.0, 0.0], "radius": 1.0, "size": 0.07}])
    st, parts = _study("K2 Hertz koule", wd / "model.step", [Part("Koule", ball), Part("Deska", plate)], ms)
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("y0", find_faces(parts, y=0))
    st.add_selection("dno", find_faces(parts, z=-20))
    st.add_selection("vrch", find_faces(parts, part="Koule", z=R))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0), Fixed("dno")]
    st.load_cases.append(LoadCase("F 1000 N", [Force("vrch", [0.0, 0.0, -250.0])]))
    st.contacts.append(ContactPair("hertz", "Koule", "Deska", kind="frictionless", slave="a"))
    st.contact_settings.method = "mortar"
    st.contact_settings.detect_tol = 0.3
    return st, parts


def k2_eval(res, st):
    s = res.steps[0]
    m = res.mesh
    sl = np.unique(m.surfaces[m.report["contacts"][0]["slave"]])
    p = s.fields["CPRESS"][sl]
    r = np.hypot(m.nodes[sl, 0], m.nodes[sl, 1])
    Es = 210000.0 / (2 * (1 - 0.3 ** 2))
    a = (3 * 1000.0 * 10.0 / (4 * Es)) ** (1 / 3)
    p0 = 3 * 1000.0 / (2 * math.pi * a ** 2)
    on = r[p > 0.02 * p.max()]
    h = 0.07
    return [Check("p₀ Hertz (2953 MPa)", float(p.max()), p0, 3e-2, "MPa"),
            Check("poloměr styku a (0,4021 mm, ±1 prvek)", float(on.max()), a, h / a, "mm")]


CONTACT_CASES["K2"] = Case("K2", "Hertz – koule na desce (mortar)", k2_build, k2_eval)
