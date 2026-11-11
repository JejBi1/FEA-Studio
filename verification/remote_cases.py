"""Verification case of the remote constraint (V10): cantilever V1 200×20×10, steel; all variants on the SAME mesh.

(a) clamping of the face x = 0 replaced by a RIGID remote constraint with all 6 components, point at the centroid of the face:
    a rigid body with zero displacement and rotation of the point = all nodes of the face zero → kinematically THE SAME as the clamping
    → deflection = V1 (Fixed) on the same mesh up to rounding (.frd 6 digits), reaction in the point
    F = 100 N, M = F·L = 20 000 N·mm (RF REF / ROT of the node from .dat – measured, not recomputed).
(b) point 50 mm beyond the face: M = F·(L + 50); deflection unchanged (the face is still entirely zero).
(a', b') the SAME with a DISTRIBUTING constraint (RBE3 *EQUATION): the face may deform (warping, lateral contraction),
    only the weighted average of the displacement and rotation is held → deflection closer to the Timoshenko beam, the reactions the same; the displacement
    of the point from the RBE3 kinematics = 0; the result does not depend on the position of the point (u_p = u_c + θ×(p − c), θ = 0 → u_c = 0).
(c) simply supported beam: the face x = 0 with a remote constraint of ux, uy, uz only (rotations free = hinge at the centroid of the face),
    the end x = 200 uy = uz = 0 (sliding hinge, rotation about y, z free), pressure p = 1 MPa on the top face
    (q = p·b = 20 N/mm): deflection in the middle 5qL⁴/(384EI) + qL²/(8κGA), rotation of the end qL³/(24EI),
    reaction in the point qL/2 (statically determinate in the results), moment in the point 0 (free rotations).
    Both the distributing (c) and the rigid (c').
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from fea_core.geometry import Part
from fea_core.results import Results
from fea_core.selectors import find_faces
from fea_core.solve import run_study
from fea_core.study import (Study, Fixed, Displacement, Force, Pressure, RemoteSupport, LoadCase, MeshSettings)
from verification.cases import Case, Check, E, NU, box, _study

L10, B10, H10, F10, P10 = 200.0, 20.0, 10.0, 100.0, 1.0
C10 = [0.0, B10 / 2, H10 / 2]                 # centroid of the face x = 0
OFF10 = 50.0                                   # a point behind the face (b)
OUT = Path(__file__).resolve().parent / "out" / "V10"
I10 = B10 * H10 ** 3 / 12
G10 = E / (2 * (1 + NU))
KAPPA = 5 / 6


def v10_build(wd: Path):
    st, parts = _study("V10 vzdalena vazba", wd / "model.step", [Part("Konzola", box(0, 0, 0, L10, B10, H10))],
                       MeshSettings(size=2.5))
    st.add_selection("celo", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=L10))
    st.add_selection("horni", find_faces(parts, z=H10))
    st.constraints.append(RemoteSupport("celo", list(C10), [True] * 6, "rigid"))
    st.load_cases.append(LoadCase("Sila 100 N", [Force("konec", [0, 0, -F10])]))
    return st, parts


def _variant(st: Study, res: Results, name: str, constraints, loads=None) -> Results:
    s2 = deepcopy(st)
    s2.name = f"V10 {name}"
    s2.constraints = list(constraints)
    if loads is not None:
        s2.load_cases = [LoadCase(loads[0], loads[1])]
    return run_study(s2, OUT / name, mesh=res.mesh)          # the same mesh (no new meshing)


def _tip(res: Results) -> float:
    end = np.nonzero(np.abs(res.mesh.nodes[:, 0] - L10) < 1e-6)[0]
    return float(res.step(0).U[end, 2].mean())


def _one(v: float) -> float:
    """1 + deviation (for a check "should be 0" with a relative tolerance)."""
    return 1.0 + float(v)


def v10_eval(res: Results, st: Study):
    s = res.step(0)
    tim = F10 * L10 ** 3 / (3 * E * I10) + F10 * L10 / (KAPPA * G10 * B10 * H10)
    off = [-OFF10, C10[1], C10[2]]
    ref = _variant(st, res, "vetknuti", [Fixed("celo")])
    rb = _variant(st, res, "tuha_mimo", [RemoteSupport("celo", off, [True] * 6, "rigid")])
    da = _variant(st, res, "distribucni", [RemoteSupport("celo", list(C10), [True] * 6, "distributing")])
    db = _variant(st, res, "distribucni_mimo", [RemoteSupport("celo", off, [True] * 6, "distributing")])
    tip_ref = _tip(ref)
    out = []
    for tag, r, arm in (("(a) tuhá, bod v těžišti", res, L10), ("(b) tuhá, bod 50 mm za čelem", rb, L10 + OFF10),
                        ("(a') distribuční, těžiště", da, L10), ("(b') distribuční, 50 mm za čelem", db, L10 + OFF10)):
        sr = r.step(0)
        rem = sr.remote["celo"]
        out += [
            Check(f"{tag}: reakce Fz v bodě", float(rem["force"][2]), F10, 1e-4, "N"),
            Check(f"{tag}: reakce My v bodě = −F·rameno", -float(rem["moment"][1]), F10 * arm, 1e-4, "N·mm"),
            # distributing: sum of RF − F over ~140 nodes, RF in .frd has 6 valid digits → noise ~1e-5·F
            Check(f"{tag}: |Fx, Fy, Mx, Mz| / F, /(F·L) + 1",
                  _one(max(np.abs(rem["force"][:2]).max() / F10, abs(rem["moment"][0]) / (F10 * arm),
                           abs(rem["moment"][2]) / (F10 * arm))), 1.0, 1e-4, ""),
            Check(f"{tag}: posuv + natočení bodu ≈ 0 (1 + |u|/|w_konec|)",
                  _one(max(np.abs(rem["u"]).max(), np.abs(rem["rot"]).max() * L10) / abs(_tip(r))), 1.0, 1e-6, ""),
            Check(f"{tag}: rovnováha (nerovnováha + 1)", _one(sr.equilibrium["imbalance"]), 1.0, 1e-4, ""),
        ]
    out += [
        Check("(a) průhyb = vetknutí na stejné síti", -_tip(res), -tip_ref, 1e-5, "mm"),
        Check("(a) průhyb konce (Timoshenko)", -_tip(res), tim, 0.03, "mm"),
        Check("(a) reakce výběru = reakce v bodě (StepResult.reactions)", float(s.reactions["celo"][2]), F10, 1e-4, "N"),
        Check("(b) průhyb = (a)", -_tip(rb), -_tip(res), 1e-5, "mm"),
        Check("(a') průhyb konce (Timoshenko)", -_tip(da), tim, 0.01, "mm"),
        Check("(b') průhyb = (a') (nezávisí na poloze bodu)", -_tip(db), -_tip(da), 1e-5, "mm"),
    ]
    # (c) simply supported beam: a hinge (only displacements) at the centroid of the face + a sliding hinge at the end, pressure on the top face
    q = P10 * B10
    w_mid = 5 * q * L10 ** 4 / (384 * E * I10) + q * L10 ** 2 / (8 * KAPPA * G10 * B10 * H10)
    th_end = q * L10 ** 3 / (24 * E * I10)
    for tag, cpl in (("(c) distribuční kloub", "distributing"), ("(c') tuhý kloub", "rigid")):
        r = _variant(st, res, f"prosty_{cpl}",
                     [RemoteSupport("celo", list(C10), [True, True, True, False, False, False], cpl),
                      Displacement("konec", uy=0.0, uz=0.0)],
                     ("Tlak 1 MPa", [Pressure("horni", P10)]))
        sr = r.step(0)
        rem = sr.remote["celo"]
        w = -float(r.probe([L10 / 2, B10 / 2, H10 / 2], "U")[2])
        out += [
            Check(f"{tag}: průhyb uprostřed (Timoshenko)", w, w_mid, 0.03, "mm"),
            Check(f"{tag}: natočení bodu θy = qL³/(24EI)", float(rem["rot"][1]), th_end, 0.03, "rad"),
            Check(f"{tag}: reakce Fz v bodě = qL/2", float(rem["force"][2]), q * L10 / 2, 1e-4, "N"),
            Check(f"{tag}: moment v bodě ≈ 0 (1 + |M| / (qL/2·L))",
                  _one(np.abs(rem["moment"]).max() / (q * L10 / 2 * L10)), 1.0, 1e-5, ""),
            Check(f"{tag}: reakce konce Fz = qL/2", float(sr.reactions["konec"][2]), q * L10 / 2, 1e-4, "N"),
        ]
    return out


REMOTE_CASES = {
    "V10": Case("V10", "Vzdálená vazba: konzola V1 (tuhá / distribuční, bod mimo plochu), prostý nosník s kloubem",
                v10_build, v10_eval),
}
