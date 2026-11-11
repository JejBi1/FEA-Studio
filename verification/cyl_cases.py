"""Verification cases of the cylindrical support (V8) and the bearing force (V9).

V8: quarter of a tube ri 20 / ro 40 / h 10, symmetry x, y, z, the inner surface constrained ONLY RADIALLY
    (tangentially and axially free), external pressure p. Lamé with u(ri) = 0: σr + σθ = const → εz const., σz = 0
    → plane stress is the exact 3D solution (free ends, axial sliding along the inner surface).
        σr = A − B/r², σθ = A + B/r², u = r/E·[(1−ν)A + (1+ν)B/r²]
        u(ri) = 0 → B = −(1−ν)/(1+ν)·A·ri², σr(ro) = −p → A = −p / (1 + (1−ν)ri²/((1+ν)ro²))
        σr(ri) = 2A/(1+ν), σθ(ri) = ν·σr(ri), uz(h) = −2νA·h/E
    Edges shared with the symmetry planes verify the coincidence of *EQUATION with *BOUNDARY.
V9: plate 100×60×10 with a hole Ø20 (eye), clamped end x = 100, bearing force in the hole:
    resultant of the nodal forces = F (1e-9), moment about the hole axis 0, pressure p = p0·cos θ on the facing
    half (p0 = 4F/(π·d·h)), reaction in the clamping = −F (accuracy of .frd 6 digits).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut

from fea_core.geometry import Part
from fea_core.results import Results
from fea_core.selectors import find_faces
from fea_core.study import (Study, Displacement, Fixed, Pressure, CylindricalSupport, BearingLoad, LoadCase,
                            MeshSettings)
from verification.cases import Case, Check, E, NU, box, cyl, solid, _study

# ------------------------------------------------------------------ V8
R8I, R8O, H8, P8 = 20.0, 40.0, 10.0, 100.0


def _lame8():
    A = -P8 / (1 + (1 - NU) * R8I ** 2 / ((1 + NU) * R8O ** 2))
    B = -(1 - NU) * A * R8I ** 2 / (1 + NU)
    return A, B


def v8_build(wd: Path):
    ring = BRepAlgoAPI_Cut(cyl((0, 0, 0), (0, 0, 1), R8O, H8), cyl((0, 0, -1), (0, 0, 1), R8I, H8 + 2)).Shape()
    quarter = solid(BRepAlgoAPI_Common(ring, box(0, 0, -1, R8O + 1, R8O + 1, H8 + 2)).Shape())
    st, parts = _study("V8 trubka valcova vazba", wd / "model.step", [Part("Trubka", quarter)], MeshSettings(size=2.5))
    st.add_selection("sym_x", find_faces(parts, x=0))
    st.add_selection("sym_y", find_faces(parts, y=0))
    st.add_selection("sym_z", find_faces(parts, z=0))
    st.add_selection("vnitrni", find_faces(parts, radius=R8I))
    st.add_selection("vnejsi", find_faces(parts, radius=R8O))
    st.constraints += [Displacement("sym_x", ux=0.0), Displacement("sym_y", uy=0.0), Displacement("sym_z", uz=0.0),
                       CylindricalSupport("vnitrni", radial=True, tangential=False, axial=False)]
    st.load_cases.append(LoadCase("Vnější tlak 100 MPa", [Pressure("vnejsi", P8)]))
    return st, parts


def _polar(res: Results, r: float, phi: float, z: float) -> tuple[float, float, float]:
    from fea_core.results import tensor
    T = tensor(res.probe([r * math.cos(phi), r * math.sin(phi), z], "S"))
    er = np.array([math.cos(phi), math.sin(phi), 0.0])
    et = np.array([-math.sin(phi), math.cos(phi), 0.0])
    return float(er @ T @ er), float(et @ T @ et), float(T[2, 2])


def v8_eval(res: Results, st: Study):
    s = res.step(0)
    n = res.mesh.nodes
    A, B = _lame8()
    sr_i, st_i = 2 * A / (1 + NU), 2 * NU * A / (1 + NU)
    u_o = R8O / E * ((1 - NU) * A + (1 + NU) * B / R8O ** 2)
    uz_h = -2 * NU * A * H8 / E
    r = np.hypot(n[:, 0], n[:, 1])
    outer = np.nonzero(np.abs(r - R8O) < 1e-3)[0]
    inner = np.nonzero(np.abs(r - R8I) < 1e-3)[0]
    top_in = inner[np.abs(n[inner, 2] - H8) < 1e-6]
    ur = lambda idx: (s.U[idx, 0] * n[idx, 0] + s.U[idx, 1] * n[idx, 1]) / r[idx]   # noqa: E731
    sr, sth, _ = _polar(res, R8I, math.radians(45), H8 / 2)
    _, sth_o, _ = _polar(res, R8O, math.radians(45), H8 / 2)
    # the tangential displacement at ri must be zero only from the equilibrium (the constraint does not hold it): symmetry → u_θ = 0
    ut = (-s.U[inner, 0] * n[inner, 1] + s.U[inner, 1] * n[inner, 0]) / r[inner]
    return [
        Check("u_r(ro) průměr", float(ur(outer).mean()), u_o, 0.01, "mm"),
        Check("vazba: 1 + max|u_r(ri)| / |u_r(ro)|", 1 + float(np.abs(ur(inner)).max()) / abs(u_o), 1.0, 1e-6, ""),
        Check("σr na ri (45°, z = h/2)", sr, sr_i, 0.05, "MPa"),
        Check("σθ na ri = ν·σr (εθ = 0)", sth, st_i, 0.05, "MPa"),
        Check("σθ na ro", sth_o, A + B / R8O ** 2, 0.05, "MPa"),
        Check("uz horní hrany díry (osově volná)", float(s.U[top_in, 2].mean()), uz_h, 0.01, "mm"),
        Check("tečně volná: 1 + max|u_θ(ri)| / |u_r(ro)|", 1 + float(np.abs(ut).max()) / abs(u_o), 1.0, 1e-3, ""),
        Check("reakce vazby Fx = −σr(ri)·ri·h", float(s.reactions["vnitrni"][0]), -sr_i * R8I * H8, 0.01, "N"),
        Check("reakce vazby Fz (osově volná) / Fx + 1", 1 + float(s.reactions["vnitrni"][2]) / (-sr_i * R8I * H8),
              1.0, 1e-4, ""),
    ]


# ------------------------------------------------------------------ V9
L9, W9, T9, D9, XC9 = 100.0, 60.0, 10.0, 20.0, 30.0
V9_CASES = [("Fx −10 kN", [-10000.0, 0.0, 0.0]), ("šikmo (−6; 8) kN", [-6000.0, 8000.0, 0.0]),
            ("s osovou složkou", [-3000.0, 0.0, 1500.0])]


def v9_build(wd: Path):
    plate = solid(BRepAlgoAPI_Cut(box(0, 0, 0, L9, W9, T9), cyl((XC9, W9 / 2, -1), (0, 0, 1), D9 / 2, T9 + 2)).Shape())
    st, parts = _study("V9 oko loziskova sila", wd / "model.step", [Part("Oko", plate)],
                       MeshSettings(size=4.0, local=[{"selection": "dira", "size": 1.5}]))
    st.add_selection("vetknuti", find_faces(parts, x=L9))
    st.add_selection("dira", find_faces(parts, radius=D9 / 2))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases = [LoadCase(name, [BearingLoad("dira", vec)]) for name, vec in V9_CASES]
    return st, parts


def v9_eval(res: Results, st: Study):
    from fea_core.cylinder import bearing_load
    out = []
    mesh = res.mesh
    c_ref = np.array([XC9, W9 / 2, 0.0])
    for k, (name, vec) in enumerate(V9_CASES):
        F = np.asarray(vec)
        f, info = bearing_load(mesh, "dira", F)
        Fn = float(np.linalg.norm(F))
        c, d = info["axis_point"], info["axis_dir"]
        M = np.cross(mesh.nodes - c, f).sum(axis=0)
        s = res.step(k)
        out += [
            Check(f"[{name}] výslednice uzlových sil |ΣF − F| / |F| + 1", 1 + float(np.linalg.norm(f.sum(axis=0) - F)) / Fn,
                  1.0, 1e-9, ""),
            Check(f"[{name}] moment k ose |M·e| / (|F|·r) + 1", 1 + abs(float(M @ d)) / (Fn * D9 / 2), 1.0, 1e-9, ""),
            Check(f"[{name}] reakce ve vetknutí = −F", float(-s.reactions["vetknuti"] @ F / Fn), Fn, 1e-5, "N"),
        ]
        if k == 0:
            off = np.linalg.norm((c - c_ref) - ((c - c_ref) @ d) * d)
            out.append(Check("osa díry ze sítě (odchylka středu + 1)", 1 + float(off), 1.0, 1e-6, "mm"))
        Fr = F - (F @ d) * d
        if np.linalg.norm(Fr) > 0:
            fh = Fr / np.linalg.norm(Fr)
            x, p, dA = info["x"], info["p"], info["dA"]
            v = x - c
            v -= np.outer(v @ d, d)
            cos = (v / np.linalg.norm(v, axis=1)[:, None]) @ fh
            p0 = 4 * float(np.linalg.norm(Fr)) / (math.pi * D9 * T9)
            ref = p0 * np.maximum(cos, 0.0)
            out += [
                Check(f"[{name}] tlak = p0·cos θ: 1 + max|p − p0 cos θ| / p0", 1 + float(np.abs(p - ref).max()) / p0,
                      1.0, 2e-3, ""),
                Check(f"[{name}] ∫p dA = 4·F_r/π", float((p * dA).sum()), 4 * float(np.linalg.norm(Fr)) / math.pi,
                      2e-3, "N"),
            ]
    return out


CYL_CASES = {
    "V8": Case("V8", "Trubka ri 20 / ro 40, vnitřní plocha radiálně vázaná (válcová vazba), vnější tlak 100 MPa",
               v8_build, v8_eval),
    "V9": Case("V9", "Oko 100×60×10 s dírou Ø20, ložisková síla (3 stavy)", v9_build, v9_eval),
}
