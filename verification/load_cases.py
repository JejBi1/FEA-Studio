"""Verification cases of loads (13. 10. 2026): force on an edge and at a point (V11), local and cylindrical coordinate system (V12),
variable (linear and hydrostatic) pressure (V13), rotation – centrifugal force and angular acceleration (V14).

V11 cantilever V1 200×20×10 clamped at x = 0, one mesh for all states:
    (a) force −100 N on the upper edge of the face x = 200 → deflection of the end (the face average) = Timoshenko V1 (a local indentation under the
        edge affects the face average little), reactions Fz and My = F·L exactly;
    (b) −200 N along the length of the upper front edge (y = 0, z = 10): q = 1 N/mm, deflection qL⁴/(8EI) + qL²/(2κGA)
        (the face average – the torsion from the eccentricity cancels in the average), reactions Mx = q·L·b/2 (eccentricity), My = qL²/2;
    (c) −100 N at a point in the middle of the face (an inserted node) → deflection ≈ V1, reactions exactly;
    (d) −100 N at 2 points (the vertices of the upper edge of the face) 50 N each.
V12 shaft V7 Ø20 × 200 clamped at x = 0:
    (a) tangential force F_θ = 15 kN on the face in the cylindrical system (axis x): uniform tangential density → T = ⅔·F_θ·R
        = 100 N·m → torsion T·L/(GJ) as V7, reaction Mx = T;
    (b) radial force F_r = −p·A along the shell (inwards) = pressure p on the same mesh (the same nodal forces except the integration:
        *DLOAD of ccx × Gauss 6 points → agreement of the displacements ~1e-6);
    (c) Cartesian system (z = global x, x = global y → y = global z): components (0, −100, 0) = global
        (0, 0, −100) → displacements identical to the state in the global axes (exactly).
V13 wall of a tank (clamped beam) b = 100, t = 20, H = 1000, water ρ = 1000 kg/m³, g = 9810 mm/s²:
    (a) full level (z = H): q(z) = ρ·g·b·(H − z), deflection of the top q₀H⁴/(30EI) (+ shear q₀H²/(3κGA)), reaction
        F = ρgbH²/2, moment ρgbH³/6;
    (b) level z = h = 500 (clipped): w = q₀h⁴/(30EI) + q₀h³/(24EI)·(H − h) (+ shear), reaction ρgbh²/2, moment ρgbh³/6;
    (c) linear pressure without clipping p = p₀ + g·(x − o) with NEGATIVE values (tension) – reaction = ∫p dA and the moment exactly;
    (d) an almost constant variable pressure (gradient 1e-12) = *DLOAD of the constant pressure on the same mesh (nodal forces
        *CLOAD from the Gauss integration = the integration of ccx).
V14 rotation:
    (a) annulus ri = 20, ro = 100, thickness 4 (1/8 of the model, symmetry), ω = 1000 rad/s about z: plane stress
        σθ(ri) = ρω²/4·((3 + ν)ro² + (1 − ν)ri²), σθ(ro) = ρω²/4·((1 − ν)ro² + (3 + ν)ri²), u_r(ro) = ro·σθ(ro)/E
        (a thin disc – 3D deviation ~ (t/r)²);
    (b) bar 100 × 4 × 4 clamped at x = 0 (axis through the point 0), α = 1e5 rad/s²: transverse load q(x) = ρAαx →
        deflection of the end 11q₀L⁴/(120EI) (+ shear q₀L²/(3κGA)), reaction Fy = ρAαL²/2, Mz = α·J_z exactly;
    (c) the same bar, ω = 1000 rad/s: axial force N(0) = ρω²AL²/2 (reaction Fx), elongation ρω²L³/(3E);
    (d) assembly of two parts, only one rotates (parts) → the reaction only from its mass.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut

from fea_core.geometry import Part
from fea_core.results import Results, tensor
from fea_core.selectors import find_faces, find_edges, PointSelector
from fea_core.study import (Study, Fixed, Displacement, Force, Pressure, Rotation, LoadCase, MeshSettings,
                            CoordinateSystem)
from verification.cases import Case, Check, E, NU, RHO, box, cyl, solid, _study

G = E / (2 * (1 + NU))
KAPPA = 5 / 6


def _one(v: float) -> float:
    return 1.0 + float(v)


def _eq_checks(res: Results, tag: str = "") -> list[Check]:
    return [Check(f"{tag}{s.name}: rovnováha (nerovnováha + 1)", _one(s.equilibrium["imbalance"]), 1.0, 1e-5, "")
            for s in res.steps]


# ------------------------------------------------------------------ V11 edge and point
L1, B1, H1 = 200.0, 20.0, 10.0
I1 = B1 * H1 ** 3 / 12


def v11_build(wd: Path):
    st, parts = _study("V11 sila na hranu a do bodu", wd / "model.step", [Part("Konzola", box(0, 0, 0, L1, B1, H1))],
                       MeshSettings(size=2.5))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("hrana_cela", edges=find_edges(parts, x=L1, z=H1))
    st.add_selection("hrana_horni", edges=find_edges(parts, y=0, z=H1))
    st.add_selection("bod_stred", points=[PointSelector("Konzola", [L1, B1 / 2, H1 / 2])])
    st.add_selection("body_rohy", points=[PointSelector("Konzola", [L1, 0, H1]), PointSelector("Konzola", [L1, B1, H1])])
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases += [
        LoadCase("Hrana čela −100 N", [Force("hrana_cela", [0, 0, -100.0])]),
        LoadCase("Horní hrana −200 N", [Force("hrana_horni", [0, 0, -200.0])]),
        LoadCase("Bod −100 N", [Force("bod_stred", [0, 0, -100.0])]),
        LoadCase("Dva body −100 N", [Force("body_rohy", [0, 0, -100.0])]),
    ]
    return st, parts


def _tip(res: Results, k: int) -> float:
    m = res.mesh
    end = np.nonzero(np.abs(m.nodes[:, 0] - L1) < 1e-6)[0]
    return float(res.step(k).U[end, 2].mean())


def v11_eval(res: Results, st: Study):
    tim = 100 * L1 ** 3 / (3 * E * I1) + 100 * L1 / (KAPPA * G * B1 * H1)
    q = 1.0
    w_q = q * L1 ** 4 / (8 * E * I1) + q * L1 ** 2 / (2 * KAPPA * G * B1 * H1)
    m = res.mesh
    out = [
        # edge of length B1 = 20 mm, mesh size 2.5 mm → exactly B1 / 2.5 = 8 segments (line3)
        Check("síť: úseček horní hrany čela (line3)", len(m.edges["hrana_cela"]), round(B1 / st.mesh.size), 0.0, ""),
        Check("síť: délka hrany čela ze sítě", float(np.linalg.norm(
            m.nodes[m.edges["hrana_cela"][:, 1]] - m.nodes[m.edges["hrana_cela"][:, 0]], axis=1).sum()), B1, 1e-9, "mm"),
        Check("síť: bod středu čela = uzel (vzdálenost + 1)",
              _one(np.linalg.norm(m.nodes[m.points["bod_stred"][0]] - [L1, B1 / 2, H1 / 2])), 1.0, 1e-9, ""),
    ]
    for k, tag, w_ref in ((0, "(a) hrana čela", tim), (2, "(c) bod", tim), (3, "(d) 2 body", tim)):
        s = res.step(k)
        out += [
            Check(f"{tag}: průhyb konce (Timoshenko)", -_tip(res, k), w_ref, 0.03, "mm"),
            Check(f"{tag}: reakce Fz", float(s.reactions["vetknuti"][2]), 100.0, 1e-5, "N"),
            Check(f"{tag}: reakční moment My = −F·L (k počátku)", float(s.reaction_moments["vetknuti"][1]
                                                                     + np.cross(s.moment_point, s.reactions["vetknuti"])[1]),
                  -100.0 * L1, 1e-5, "N·mm"),
        ]
    s = res.step(1)
    Mr = np.asarray(s.reaction_moments["vetknuti"]) + np.cross(s.moment_point, s.reactions["vetknuti"])  # to the origin
    out += [
        Check("(b) horní hrana: průhyb konce qL⁴/8EI + smyk", -_tip(res, 1), w_q, 0.03, "mm"),
        Check("(b) reakce Fz = q·L", float(s.reactions["vetknuti"][2]), 200.0, 1e-5, "N"),
        Check("(b) reakční moment My = −qL²/2 (k počátku)", float(Mr[1]), -q * L1 ** 2 / 2, 1e-5, "N·mm"),
        Check("(b) reakční moment Mx = 0 (k počátku; hrana y = 0) + 1", _one(Mr[0] / (q * L1 ** 2 / 2)), 1.0, 1e-5, ""),
    ]
    from fea_core.loads import force_nodal
    pts = m.points["body_rohy"]
    Fd = force_nodal(st, m, st.load_cases[3].loads[0])
    out.append(Check("(d) uzlové síly bodů po −50 N", float(Fd[pts, 2].mean()), -50.0, 1e-12, "N"))
    return out + _eq_checks(res)


# ------------------------------------------------------------------ V12 coordinate systems
D2, L2 = 20.0, 200.0
R2 = D2 / 2
T2 = 100_000.0
FT2 = 1.5 * T2 / R2                       # F_θ: T = ⅔·F_θ·R
P2 = 10.0


def v12_build(wd: Path):
    st, parts = _study("V12 mistni a valcovy system", wd / "model.step",
                       [Part("Hridel", cyl((0, 0, 0), (1, 0, 0), R2, L2))], MeshSettings(size=2.5))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=L2))
    st.add_selection("plast", find_faces(parts, radius=R2))
    st.coordinate_systems["valec"] = CoordinateSystem("valec", "cylindrical", [0, 0, 0], [1, 0, 0], [0, 1, 0])
    st.coordinate_systems["otoceny"] = CoordinateSystem("otoceny", "cartesian", [50, 0, 0], [1, 0, 0], [0, 1, 0])
    st.constraints.append(Fixed("vetknuti"))
    A = 2 * math.pi * R2 * L2
    st.load_cases += [
        LoadCase("Tečná síla na čelo", [Force("konec", [0, FT2, 0], csys="valec")]),
        LoadCase("Radiální síla na plášť", [Force("plast", [-P2 * A, 0, 0], csys="valec")]),
        LoadCase("Tlak na plášť", [Pressure("plast", P2)]),
        LoadCase("Ohyb v místním systému", [Force("konec", [0, -100.0, 0], csys="otoceny")]),
        LoadCase("Ohyb v globálních osách", [Force("konec", [0, 0, -100.0])]),
    ]
    return st, parts


def _twist(res: Results, k: int, x: float, r: float = 0.8 * R2) -> float:
    """Rotation of the section x: average u_θ/r in 24 points of the circle r (probes) – the bending from the residual resultant cancels."""
    th = []
    for a in np.linspace(0, 2 * math.pi, 24, endpoint=False):
        p = np.array([x, r * math.cos(a), r * math.sin(a)])
        u = res.probe(p, "U", k)
        th.append((-u[1] * p[2] + u[2] * p[1]) / r ** 2)
    return float(np.mean(th))


def _ur_mean(res: Results, k: int) -> float:
    n = res.mesh.nodes
    r = np.hypot(n[:, 1], n[:, 2])
    sel = np.nonzero(np.abs(r - R2) < 1e-3)[0]
    U = res.step(k).U
    return float(((U[sel, 1] * n[sel, 1] + U[sel, 2] * n[sel, 2]) / r[sel]).mean())


def v12_eval(res: Results, st: Study):
    J = math.pi * D2 ** 4 / 32
    s0 = res.step(0)
    Mx = float(s0.reaction_moments["vetknuti"][0] + np.cross(s0.moment_point, s0.reactions["vetknuti"])[0])
    U1, U2 = res.step(1).U, res.step(2).U
    U3, U4 = res.step(3).U, res.step(4).U
    return [
        Check("(a) zkroucení mezi x = 50 a 150 (×2) = T·L/(GJ)", math.degrees(2 * (_twist(res, 0, 150) - _twist(res, 0, 50))),
              math.degrees(T2 * L2 / (G * J)), 0.03, "°"),
        Check("(a) reakční moment Mx = ⅔·F_θ·R", -Mx, T2, 1e-3, "N·mm"),
        Check("(a) výslednice tečné síly ≈ 0 (1 + |R|/F_θ)",
              _one(np.linalg.norm(s0.reactions["vetknuti"]) / FT2), 1.0, 1e-5, ""),
        # radial from the axis ≠ normal of the curved tri6 mesh (deviation of the order (h/R)²/12) → agreement 1e-3, not 1e-6
        Check("(b) radiální síla = tlak: max |Δu| / max |u| + 1",
              _one(np.abs(U1 - U2).max() / np.abs(U2).max()), 1.0, 2e-3, ""),
        Check("(b) radiální síla = tlak: průměr u_r pláště",
              float(_ur_mean(res, 1)), float(_ur_mean(res, 2)), 1e-4, "mm"),
        Check("(c) kartézský místní systém = globální: max |Δu| / max |u| + 1",
              _one(np.abs(U3 - U4).max() / np.abs(U4).max()), 1.0, 1e-9, ""),
    ] + _eq_checks(res)


# ------------------------------------------------------------------ V13 variable pressure
B3, T3, H3 = 20.0, 20.0, 1000.0               # slender beam (a wide wall would behave as a plate)
RHO_W = 1000e-12                          # voda 1000 kg/m³ = 1e-9 t/mm³
G_ACC = 9810.0
GAM = RHO_W * G_ACC                       # ρ·g [MPa/mm] = 9,81e-6
H3B = 500.0
I3 = B3 * T3 ** 3 / 12


def v13_build(wd: Path):
    st, parts = _study("V13 promenny tlak", wd / "model.step", [Part("Stena", box(0, 0, 0, B3, T3, H3))],
                       MeshSettings(size=5.0))
    st.add_selection("pata", find_faces(parts, z=0))
    st.add_selection("voda", find_faces(parts, y=0))           # outward normal −y → the pressure pushes into +y
    st.constraints.append(Fixed("pata"))
    st.load_cases += [
        LoadCase("Hladina H", [Pressure("voda", 0.0, [0, 0, -GAM], [0, 0, H3], clip=True)]),
        LoadCase("Hladina h", [Pressure("voda", 0.0, [0, 0, -GAM], [0, 0, H3B], clip=True)]),
        LoadCase("Lineární se zápornými hodnotami", [Pressure("voda", 0.01, [3e-4, 0, -1e-5], [10, 0, 100])]),
        LoadCase("Proměnný ≈ konstantní", [Pressure("voda", 0.01, [0, 0, 1e-12], [0, 0, 0])]),
        LoadCase("Konstantní", [Pressure("voda", 0.01)]),
    ]
    return st, parts


def _top_deflection(res: Results, k: int) -> float:
    n = res.mesh.nodes
    top = np.nonzero(np.abs(n[:, 2] - H3) < 1e-6)[0]
    return float(res.step(k).U[top, 1].mean())


def v13_eval(res: Results, st: Study):
    out = []
    for k, h, tag in ((0, H3, "(a) plná hladina"), (1, H3B, "(b) hladina h = 500")):
        q0 = GAM * B3 * h
        w = q0 * h ** 4 / (30 * E * I3) + q0 * h ** 3 / (24 * E * I3) * (H3 - h) + q0 * h ** 2 / (3 * KAPPA * G * B3 * T3)
        s = res.step(k)
        Mr = np.asarray(s.reaction_moments["pata"]) + np.cross(s.moment_point, s.reactions["pata"])
        out += [
            Check(f"{tag}: průhyb vrcholu", _top_deflection(res, k), w, 0.02, "mm"),
            Check(f"{tag}: reakce Fy = −ρgbh²/2", -float(s.reactions["pata"][1]), GAM * B3 * h ** 2 / 2, 1e-4, "N"),
            Check(f"{tag}: reakční moment Mx = ρgbh³/6 (k počátku)", float(Mr[0]), GAM * B3 * h ** 3 / 6, 1e-4, "N·mm"),
        ]
    # (c) linear pressure: ∫p dA and the moment on the plane y = 0 (x ∈ [0, B], z ∈ [0, H]) analytically
    p0, gx, gz, ox, oz = 0.01, 3e-4, -1e-5, 10.0, 100.0
    A = B3 * H3
    P = p0 * A + gx * (B3 ** 2 / 2 * H3 - ox * A) + gz * (H3 ** 2 / 2 * B3 - oz * A)          # ∫p dA
    Pz = p0 * B3 * H3 ** 2 / 2 + gx * (B3 ** 2 / 2 * H3 ** 2 / 2 - ox * B3 * H3 ** 2 / 2) + \
        gz * (B3 * H3 ** 3 / 3 - oz * B3 * H3 ** 2 / 2)                                     # ∫p·z dA
    s = res.step(2)
    Mr = np.asarray(s.reaction_moments["pata"]) + np.cross(s.moment_point, s.reactions["pata"])
    out += [
        Check("(c) lineární: reakce Fy = −∫p dA", -float(s.reactions["pata"][1]), P, 1e-5, "N"),
        Check("(c) lineární: reakční moment Mx = ∫p·z dA", float(Mr[0]), Pz, 1e-5, "N·mm"),
        Check("(d) proměnný (*CLOAD) = konstantní (*DLOAD): max |Δu| / max |u| + 1",
              _one(np.abs(res.step(3).U - res.step(4).U).max() / np.abs(res.step(4).U).max()), 1.0, 1e-5, ""),
    ]
    return out + _eq_checks(res)


# ------------------------------------------------------------------ V14 rotation
RI4, RO4, T4 = 20.0, 100.0, 4.0
W4 = 1000.0
L4, A4 = 100.0, 4.0
AL4 = 1e5


def v14_build(wd: Path):
    ring = BRepAlgoAPI_Cut(cyl((0, 0, 0), (0, 0, 1), RO4, T4 / 2), cyl((0, 0, -1), (0, 0, 1), RI4, T4 + 2)).Shape()
    quarter = solid(BRepAlgoAPI_Common(ring, box(0, 0, -1, RO4 + 1, RO4 + 1, T4 + 2)).Shape())
    geo = [Part("Kotouc", quarter),
           Part("Tyc", box(200, -A4 / 2, -A4 / 2, L4, A4, A4)),
           Part("Tyc2", box(200, 20 - A4 / 2, -A4 / 2, L4, A4, A4)),      # 2nd bar, only the first one rotates (d)
           Part("Patka2", box(190, 20 - 5, -5, 10, 10, 10))]
    st, parts = _study("V14 otaceni", wd / "model.step", geo, MeshSettings(size=4.0))
    st.add_selection("sym_x", find_faces(parts, part="Kotouc", x=0))
    st.add_selection("sym_y", find_faces(parts, part="Kotouc", y=0))
    st.add_selection("sym_z", find_faces(parts, part="Kotouc", z=0))
    st.add_selection("tyc_koren", find_faces(parts, part="Tyc", x=200))
    st.add_selection("tyc2_koren", find_faces(parts, part="Patka2", x=190))
    st.constraints += [Displacement("sym_x", ux=0.0), Displacement("sym_y", uy=0.0), Displacement("sym_z", uz=0.0),
                       Fixed("tyc_koren"), Fixed("tyc2_koren")]
    st.mesh.local = [{"selection": "sym_z", "size": 2.0}]
    st.load_cases += [
        LoadCase("Kotouč ω", [Rotation([0, 0, 0], [0, 0, 1], omega=W4, parts=["Kotouc"])]),
        LoadCase("Tyč α", [Rotation([200, 0, 0], [0, 0, 1], alpha=AL4, parts=["Tyc"])]),
        LoadCase("Tyč ω", [Rotation([200, 0, 0], [0, 0, 1], omega=W4, parts=["Tyc"])]),
        LoadCase("Vše ω", [Rotation([200, 0, 0], [0, 0, 1], omega=W4)]),
    ]
    return st, parts


def v14_eval(res: Results, st: Study):
    rw = RHO * W4 ** 2
    sth_i = rw / 4 * ((3 + NU) * RO4 ** 2 + (1 - NU) * RI4 ** 2)
    sth_o = rw / 4 * ((1 - NU) * RO4 ** 2 + (3 + NU) * RI4 ** 2)
    phi = math.radians(45)

    def hoop(r):
        p = [r * math.cos(phi), r * math.sin(phi), 0.0]
        T = tensor(res.probe(p, "S", 0))
        t = np.array([-math.sin(phi), math.cos(phi), 0.0])
        return float(t @ T @ t)
    n = res.mesh.nodes
    s0 = res.step(0)
    disc = np.unique(res.mesh.tets[res.mesh.tet_part == res.mesh.part_names.index("Kotouc")])
    outer = disc[np.abs(np.hypot(n[disc, 0], n[disc, 1]) - RO4) < 1e-3]
    ur = (s0.U[outer, 0] * n[outer, 0] + s0.U[outer, 1] * n[outer, 1]) / RO4
    out = [
        Check("(a) kotouč: σθ(ri)", hoop(RI4), sth_i, 0.01, "MPa"),
        Check("(a) kotouč: σθ(ro)", hoop(RO4), sth_o, 0.01, "MPa"),
        Check("(a) kotouč: u_r(ro) = ro·σθ(ro)/E", float(ur.mean()), RO4 * sth_o / E, 0.01, "mm"),
    ]
    # (b) bar, angular acceleration: q(x) = ρAαx (x from the axis), inertial force −ρα(a × r) = ρα(y, −x, 0)
    A = A4 * A4
    I = A4 ** 4 / 12
    q0 = RHO * A * AL4 * L4
    w = 11 * q0 * L4 ** 4 / (120 * E * I) + q0 * L4 ** 2 / (3 * KAPPA * G * A)
    tip = np.nonzero(np.abs(n[:, 0] - (200 + L4)) < 1e-6)[0]
    tip = tip[np.abs(n[tip, 1]) < A4]                       # only the first bar
    s1 = res.step(1)
    Jz = RHO * A4 * A4 * (L4 ** 3 / 3 + L4 * A4 ** 2 / 12)
    c = np.array([200.0, 0, 0])
    Mz = float(s1.reaction_moments["tyc_koren"][2] + np.cross(s1.moment_point - c, s1.reactions["tyc_koren"])[2])
    out += [
        Check("(b) tyč α: průhyb konce 11q₀L⁴/120EI + smyk", -float(s1.U[tip, 1].mean()), w, 0.02, "mm"),
        Check("(b) tyč α: reakce Fy = ρAαL²/2", float(s1.reactions["tyc_koren"][1]), RHO * A * AL4 * L4 ** 2 / 2, 1e-4, "N"),
        Check("(b) tyč α: reakční moment Mz = α·J_z (k ose)", Mz, AL4 * Jz, 1e-4, "N·mm"),
    ]
    s2 = res.step(2)
    out += [
        Check("(c) tyč ω: reakce Fx = −ρω²AL²/2", -float(s2.reactions["tyc_koren"][0]), rw * A * L4 ** 2 / 2, 1e-4, "N"),
        Check("(c) tyč ω: prodloužení ρω²L³/(3E)", float(s2.U[tip, 0].mean()), rw * L4 ** 3 / (3 * E), 0.01, "mm"),
        Check("(c) jen díl Tyc: reakce 2. tyče ≈ 0 (1 + |R2|/R1)",
              _one(np.linalg.norm(s2.reactions["tyc2_koren"]) / np.linalg.norm(s2.reactions["tyc_koren"])), 1.0, 1e-6, ""),
    ]
    s3 = res.step(3)
    out.append(Check("(d) všechny díly: reakce Fx tyče = (c)", float(s3.reactions["tyc_koren"][0]),
                     float(s2.reactions["tyc_koren"][0]), 1e-6, "N"))
    return out + _eq_checks(res)


LOAD_CASES = {
    "V11": Case("V11", "Síla na hranu a do bodu (konzola V1)", v11_build, v11_eval),
    "V12": Case("V12", "Místní a válcový souřadný systém (hřídel V7)", v12_build, v12_eval),
    "V13": Case("V13", "Proměnný tlak: hydrostatika s hladinou, lineární tlak", v13_build, v13_eval),
    "V14": Case("V14", "Otáčení: rotující kotouč, úhlové zrychlení tyče, jen vybrané díly", v14_build, v14_eval),
}
