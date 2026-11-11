"""Verification of the parametric study (C4b) and of the polygon imprint (A10):
- PS1: cantilever V1 swept over load factor × Young's modulus (grid 2 × 2, one mesh) and a pure load sweep (one analysis,
  scaled): tip deflection = Timoshenko · factor · E0/E (linear elasticity), mass = ρ·V of the box (exact), the scaled points equal
  separately solved ones.
- IM2: L-shaped polygon imprint on the plate of IM1 with pressure 1 MPa: imprint area = shoelace area (straight edges → exact),
  reaction Fz = p·A, moment of the reaction about the clamped edge = p·A·x_c (centroid of the polygon, analytic).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fea_core.geometry import Part
from fea_core.imprint import Imprint, face_frame, polygon_area_centroid
from fea_core.mesh import tri_areas
from fea_core.results import Results
from fea_core.selectors import find_faces
from fea_core.study import Fixed, LoadCase, MeshSettings, Pressure, Study

from .cases import Case, Check, _study, box, v1_build, E

PARAM_CASES: dict[str, Case] = {}

V1_TIP = 0.7634            # Timoshenko tip deflection of V1 (cases.v1_eval) [mm]
V1_MASS = 200 * 20 * 10 * 7.85e-9 * 1000.0     # kg


# ===================================================================== PS1
class _Out(dict):
    """Both runs for the evaluation; .warnings for run_all."""

    @property
    def warnings(self):
        return list(self['grid'].warnings) + list(self['linear'].warnings)


def ps1_run(study: Study, workdir: Path, parts):
    from fea_core.sweep import ParametricStudySpec, SweepParam, run_parametric_study
    mat = study.default_material
    grid = ParametricStudySpec.from_params([SweepParam("load_factor", [1.0, 2.0]),
                                            SweepParam("material_prop", [E, E / 3.0], target=mat, field="E")])
    r_grid = run_parametric_study(study, grid, workdir / "grid", parts=parts)
    loads = ParametricStudySpec.from_params([SweepParam("load_factor", [0.5, 3.0])])
    r_lin = run_parametric_study(study, loads, workdir / "linear", parts=parts)
    return _Out(grid=r_grid, linear=r_lin)


def _tip(dp) -> float:
    c = (dp.kpi.get("cases") or [{}])[0]
    return float(c.get("u_max") or dp.max_disp)


def ps1_eval(out, st: Study):
    g, lin = out["grid"], out["linear"]
    checks = []
    for dp in g.points:
        f, e = dp.params["load_factor"], dp.params[f"prop:{st.default_material}:E"]
        ref = V1_TIP * f * E / e
        checks.append(Check(f"PS1 max. displacement F×{f:g}, E = {e:.0f} (Timoshenko)", dp.max_disp, ref, 0.03, "mm"))
    base = next(dp for dp in g.points if dp.params["load_factor"] == 1.0 and dp.params[f"prop:{st.default_material}:E"] == E)
    for dp in lin.points:
        checks.append(Check(f"PS1 scaled point F×{dp.params['load_factor']:g} = factor × solved point", dp.max_disp,
                            base.max_disp * dp.params["load_factor"], 1e-6, "mm"))
    checks.append(Check("PS1 mass of the beam (ρ·V)", base.mass, V1_MASS, 1e-6, "kg"))
    checks.append(Check("PS1 linear sweep solved once (shortcut used)", float(lin.linear_shortcut), 1.0, 0.0))
    checks.append(Check("PS1 all grid points converged", float(sum(p.converged for p in g.points)), 4.0, 0.0))
    return checks


PARAM_CASES["PS1"] = Case("PS1", "Parametricka studie: konzola V1 – zatizeni × modul E a linearni sken", v1_build,
                          ps1_eval, run=ps1_run)


# ===================================================================== IM2
IM2_POLY = [[-15.0, -10.0], [15.0, -10.0], [15.0, 0.0], [0.0, 0.0], [0.0, 10.0], [-15.0, 10.0]]
IM2_CENTER = [50.0, 50.0, 10.0]


def im2_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    st, parts = _study("IM2 otisk mnohouhelniku", wd / "model.step", [Part("Deska", box(0, 0, 0, 100, 100, 10))],
                       MeshSettings(size=5.0))
    st.imprints.append(Imprint("pad_L", "Deska", IM2_CENTER, [0.0, 0.0, 1.0], shape="polygon", points=IM2_POLY, size=2.0))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Tlak_pad", [Pressure("pad_L", 1.0)]))
    return st, parts


def im2_eval(res: Results, st: Study):
    imp = st.imprints[0]
    area, c2 = polygon_area_centroid(IM2_POLY)
    _, xa, ya = face_frame(imp.normal)
    centroid = np.asarray(IM2_CENTER) + c2[0] * xa + c2[1] * ya        # centroid of the pressure (analytic)
    step = res.step(0)
    m = res.mesh
    a_mesh = float(np.sum(tri_areas(m.nodes, m.surfaces["pad_L"])))
    F = 1.0 * area
    load = np.array([0.0, 0.0, -F])                                     # pressure 1 MPa into the top face
    rz = float(step.reactions["vetknuti"][2])
    # equilibrium about the moment point c0 of the reactions: M_reaction = −(x_c − c0) × F_load
    c0 = np.asarray(step.moment_point, float)
    m_ref = -np.cross(centroid - c0, load)
    m_fem = np.asarray(step.reaction_moments["vetknuti"], float)
    return [
        Check("IM2 imprint area = shoelace area (straight edges)", a_mesh, area, 1e-6, "mm²"),
        Check("IM2 reaction Fz = p·A", rz, F, 0.002, "N"),
        Check("IM2 reaction moment Mx about c0 = −((x_c − c0) × F)x", float(m_fem[0]), float(m_ref[0]), 0.005, "N·mm"),
        Check("IM2 reaction moment My about c0 = −((x_c − c0) × F)y", float(m_fem[1]), float(m_ref[1]), 0.005, "N·mm"),
    ]

PARAM_CASES["IM2"] = Case("IM2", "Otisk mnohouhelniku (L): plocha, reakce a tezisko tlaku 1 MPa", im2_build, im2_eval)
