"""Verification of the general orthotropic material (fea_core.ortho).

V16 (a) bar 100×10×10, material axes rotated by 30° about z (Cartesian system), tension σ = 10 MPa: the strain
        in the middle = the transformed compliance (ε_x, ε_y and the shear γ_xy from the rotation);
    (b) ring r = 90–110, height 10, cylindrically orthotropic (E_r = 10 GPa, E_θ = 100 GPa, ν13 = ν23 = 0 → plane
        stress exactly), internal pressure 10 MPa: Lekhnitskii u = A·r^k + B·r^−k, k = √(Q22/Q11).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.study import (CoordinateSystem, CylindricalSupport, Displacement, Fixed, LoadCase, Material,
                             MeshSettings, Pressure)
from verification.cases import Case, Check, _study, box, cyl
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

ORTHO_A = {"E1": 100000.0, "E2": 20000.0, "E3": 10000.0, "nu12": 0.3, "nu13": 0.25, "nu23": 0.4,
           "G12": 5000.0, "G13": 4000.0, "G23": 3000.0}
ORTHO_B = {"E1": 10000.0, "E2": 100000.0, "E3": 10000.0, "nu12": 0.03, "nu13": 0.0, "nu23": 0.0,
           "G12": 5000.0, "G13": 4000.0, "G23": 4000.0}
THETA = math.radians(30.0)
SIG = 10.0
RI, RO, P = 90.0, 110.0, 10.0


def v16_build(wd: Path):
    ring = BRepAlgoAPI_Cut(cyl((300, 0, 0), (0, 0, 1), RO, 10), cyl((300, 0, 0), (0, 0, 1), RI, 10)).Shape()
    st, parts = _study("V16 ortotropni material", wd / "model.step",
                       [Part("Tyc", box(0, 0, 0, 100, 10, 10)), Part("Prstenec", ring)], MeshSettings(size=2.5))
    st.coordinate_systems["mat30"] = CoordinateSystem("mat30", "cartesian", [0, 0, 0], [0, 0, 1],
                                                      [math.cos(THETA), math.sin(THETA), 0])
    st.coordinate_systems["valec"] = CoordinateSystem("valec", "cylindrical", [300, 0, 0], [0, 0, 1], [1, 0, 0])
    st.add_material(Material("Ortotropní A", ORTHO_A["E1"], ORTHO_A["nu12"], ortho=ORTHO_A, ortho_csys="mat30"),
                    default=True)
    st.add_material(Material("Ortotropní B", ORTHO_B["E1"], ORTHO_B["nu12"], ortho=ORTHO_B, ortho_csys="valec"))
    st.part_materials["Prstenec"] = "Ortotropní B"
    st.add_selection("x0", find_faces(parts, part="Tyc", x=0))
    st.add_selection("x1", find_faces(parts, part="Tyc", x=100))
    st.add_selection("vnitrni", find_faces(parts, part="Prstenec", radius=RI))
    st.add_selection("dno", find_faces(parts, part="Prstenec", z=0))
    st.add_selection("vnejsi", find_faces(parts, part="Prstenec", radius=RO))
    # ring: the tangential component of the cylindrical support on the outer shell holds the displacements in the plane xy and the rotation about the axis
    # and does not prevent the axially symmetric deformation (the tangential displacement is zero in it)
    st.constraints += [Fixed("x0"), Displacement("dno", uz=0.0),
                       CylindricalSupport("vnejsi", radial=False, tangential=True, axial=False)]
    st.load_cases.append(LoadCase("Tah a tlak", [Pressure("x1", -SIG), Pressure("vnitrni", P)]))
    return st, parts


def _strain(res, p, h=4.0):
    """Displacement gradient at the point p by central differences (the state is homogeneous → exact for tet10)."""
    G = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        G[:, j] = (res.probe(np.add(p, e), "U") - res.probe(np.subtract(p, e), "U")) / (2 * h)
    return 0.5 * (G + G.T)


def _expected_a():
    from fea_core.ortho import compliance
    S = compliance(ORTHO_A)
    c, s = math.cos(THETA), math.sin(THETA)
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])          # rows = material axes in the global ones
    sig = np.zeros((3, 3))
    sig[0, 0] = SIG
    sm = R @ sig @ R.T
    v = np.array([sm[0, 0], sm[1, 1], sm[2, 2], sm[0, 1], sm[0, 2], sm[1, 2]])
    e = S @ v
    em = np.array([[e[0], e[3] / 2, e[4] / 2], [e[3] / 2, e[1], e[5] / 2], [e[4] / 2, e[5] / 2, e[2]]])
    return R.T @ em @ R


def _lekhnitskii(r):
    o = ORTHO_B
    Sp = np.array([[1 / o["E1"], -o["nu12"] / o["E1"]], [-o["nu12"] / o["E1"], 1 / o["E2"]]])
    Q = np.linalg.inv(Sp)
    k = math.sqrt(Q[1, 1] / Q[0, 0])

    def sr(A, B, rr):          # σ_r = Q11 u' + Q12 u/r, u = A r^k + B r^-k
        return (Q[0, 0] * (A * k * rr ** (k - 1) - B * k * rr ** (-k - 1))
                + Q[0, 1] * (A * rr ** (k - 1) + B * rr ** (-k - 1)))
    M = np.array([[sr(1, 0, RI), sr(0, 1, RI)], [sr(1, 0, RO), sr(0, 1, RO)]])
    A, B = np.linalg.solve(M, [-P, 0.0])
    return A * r ** k + B * r ** (-k)


def v16_eval(res, st):
    e = _strain(res, [50.0, 5.0, 5.0])
    ref = _expected_a()
    s = res.step(0)
    x = res.mesh.nodes
    rr = np.hypot(x[:, 0] - 300, x[:, 1])
    ring = (rr > RI - 1) & (np.abs(x[:, 0] - 300) < RO + 1)
    out_n = np.nonzero(ring & (np.abs(rr - RO) < 1e-6))[0]
    ur = ((s.U[out_n, 0] * (x[out_n, 0] - 300) + s.U[out_n, 1] * x[out_n, 1]) / rr[out_n]).mean()
    return [
        Check("(a) ε_x tyče s natočenou ortotropií", e[0, 0], ref[0, 0], 0.01, ""),
        Check("(a) ε_y (Poisson v natočených osách)", e[1, 1], ref[1, 1], 0.02, ""),
        Check("(a) γ_xy/2 (smyk od natočení os)", e[0, 1], ref[0, 1], 0.02, ""),
        Check("(b) u_r vnějšího pláště prstence (Lekhnitskii)", float(ur), _lekhnitskii(RO), 0.01, "mm"),
    ]


ORTHO_CASES = {
    "V16": Case("V16", "Ortotropní materiál: natočené osy (tah), válcová ortotropie (prstenec)", v16_build, v16_eval),
}
