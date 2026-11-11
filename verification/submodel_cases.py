"""Verification case SM2 – submodel of a solved study (fea_core.submodel_run, the path used by GUI / MCP / report).

Thick plate 160 × 80 × 10 mm with a hole Ø20, clamped at x = −80, one load case combining every load kind the submodel
transfers: tension 20 kN on the end face x = +80, pressure 0.1 MPa on the top face (partly inside the region), internal pressure
20 MPa in the hole (inside the region) and gravity 100 g (body force inside the region).

Independent reference = a FINE FULL MODEL of the same geometry (standard run_study, local sphere refinement around the hole with
the same element size as the submodel, 1.2 mm). The submodel is driven by a COARSE global model (8 mm). Checks:
  * max. von Mises in the region: submodel vs the fine full model (same element size at the hole – tolerance 3 %, the meshes differ);
  * displacement at the hole (interpolated) vs the fine model (0.5 %);
  * resultant of the loads applied inside the region = p·A_top(region) + ρ·g·V(region) by hand (0.5 %, hole pressure ~0);
  * cut check (mean |Δσ_vM| on the cut < 10 %);
  * the coarse global model underestimates the peak (informative: the submodel must be closer to the fine model than the global one).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.study import Fixed, Force, Gravity, LoadCase, MeshSettings, Pressure, Study

from .cases import AtMost, Case, Check, RHO, _study

SM2_L, SM2_W, SM2_T, SM2_R = 160.0, 80.0, 10.0, 10.0
SM2_F, SM2_P_TOP, SM2_P_HOLE, SM2_G = 20000.0, 0.1, 20.0, 981000.0
SM2_BOX = (40.0, 40.0, 14.0)
SM2_H = 1.2
SM2_SPHERE = 19.0                 # second region: sphere around the hole centre (spherical cut faces)


def _plate():
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    b = BRepPrimAPI_MakeBox(gp_Ax2(gp_Pnt(-SM2_L / 2, -SM2_W / 2, 0.0), gp_Dir(0, 0, 1)), SM2_L, SM2_W, SM2_T).Shape()
    c = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -1.0), gp_Dir(0, 0, 1)), SM2_R, SM2_T + 2.0).Shape()
    op = BRepAlgoAPI_Cut(b, c)
    op.Build()
    return Part("Deska", op.Shape())


def _setup(st: Study, parts) -> None:
    st.add_selection("clamp", find_faces(parts, x=-SM2_L / 2))
    st.add_selection("end", find_faces(parts, x=SM2_L / 2))
    st.add_selection("top", find_faces(parts, z=SM2_T))
    st.add_selection("hole", find_faces(parts, radius=SM2_R))
    st.constraints.append(Fixed("clamp"))
    st.load_cases.append(LoadCase("Combined", [Force("end", [SM2_F, 0.0, 0.0]), Pressure("top", SM2_P_TOP),
                                               Pressure("hole", SM2_P_HOLE), Gravity([0.0, 0.0, -SM2_G])]))


def sm2_build(wd: Path):
    wd.mkdir(parents=True, exist_ok=True)
    st, parts = _study("SM2 global plate", wd / "model.step", [_plate()], MeshSettings(size=8.0))
    _setup(st, parts)
    return st, parts


def sm2_run(study: Study, workdir: Path, parts):
    from dataclasses import replace
    from fea_core.solve import run_study
    from fea_core.submodel_run import SubmodelRegion, run_submodel
    res_g = run_study(study, workdir=workdir / "run_global", parts=parts)
    fine = replace(study, mesh=replace(study.mesh, size=8.0,
                                       local=[{"center": [0.0, 0.0, SM2_T / 2], "radius": 22.0, "size": SM2_H}]))
    res_f = run_study(fine, workdir=workdir / "run_fine", parts=parts)
    region = SubmodelRegion(center=[0.0, 0.0, SM2_T / 2], shape="box", size=list(SM2_BOX), mesh_size=SM2_H, step=0)
    sub = run_submodel(study, parts, res_g, region, workdir / "run_submodel")
    sphere = SubmodelRegion(center=[0.0, 0.0, SM2_T / 2], shape="sphere", radius=SM2_SPHERE, mesh_size=SM2_H, step=0)
    sub_s = run_submodel(study, parts, res_g, sphere, workdir / "run_submodel_sphere")
    from types import SimpleNamespace
    return SimpleNamespace(glob=res_g, fine=res_f, sub=sub, sub_s=sub_s,
                           warnings=res_g.warnings + sub.warnings + sub_s.warnings)


def sm2_eval(res, st: Study):
    if not hasattr(res, "sub"):                          # without its own run there is no submodel → the checks FAIL
        nan = float("nan")
        return [Check("SM2 submodel max σ_vM vs fine full model", nan, 1.0, 0.03, "MPa")]
    from fea_core.submodel_run import SubmodelRegion
    sub, glob, fine = res.sub, res.glob, res.fine
    region: SubmodelRegion = sub.region
    tol = 1e-6
    fm = fine.mesh
    inside = np.nonzero(region.signed_distance(fm.nodes) <= tol)[0]
    inside = np.intersect1d(inside, np.unique(fm.tets))
    vm_fine = float(fine.step(0).mises[inside].max())
    vm_glob = float(glob.step(0).mises[np.intersect1d(np.nonzero(region.signed_distance(glob.mesh.nodes) <= tol)[0],
                                                      np.unique(glob.mesh.tets))].max())
    vm_sub = float(sub.summary["max_mises_MPa"])
    p = np.array([-SM2_R, 0.0, SM2_T])                    # hole edge on the top face (largest bending + tension)
    u_fine = np.asarray(fine.probe(p, "U", 0))
    u_sub = np.asarray(sub.as_results().probe(p, "U", 0))
    # loads inside the region by hand: top-face pressure on the region part of the face, weight of the region volume
    a_top = SM2_BOX[0] * SM2_BOX[1] - math.pi * SM2_R ** 2
    v_reg = a_top * SM2_T
    fz_ref = -(SM2_P_TOP * a_top + RHO * SM2_G * v_reg)
    fz = float(sub.summary["equilibrium"]["applied_force_N"][2])
    chk = sub.check
    sub_s = res.sub_s
    ins = np.intersect1d(np.nonzero(sub_s.region.signed_distance(fm.nodes) <= tol)[0], np.unique(fm.tets))
    vm_fine_s = float(fine.step(0).mises[ins].max())
    return [
        Check("SM2 koule: submodel max σ_vM vs jemny plny model", float(sub_s.summary["max_mises_MPa"]), vm_fine_s,
              0.03, "MPa"),
        Check("SM2 koule: kontrola rezu < 10 %", sub_s.check["mean_rel_diff"], 0.0, 0.10, "-"),
        Check("SM2 submodel max σ_vM vs fine full model (h 1,2 mm)", vm_sub, vm_fine, 0.03, "MPa"),
        Check("SM2 posuv na hrane otvoru vs jemny model", float(np.linalg.norm(u_sub)), float(np.linalg.norm(u_fine)),
              0.005, "mm"),
        Check("SM2 vysledna zatizeni v oblasti (tlak + tize) vs rucni vypocet", fz, fz_ref, 0.005, "N"),
        Check("SM2 kontrola rezu: stredni odchylka napeti < 10 %", chk["mean_rel_diff"], 0.0, 0.10, "-"),
        AtMost("SM2 submodel blize jemnemu modelu nez hruby globalni", abs(vm_sub - vm_fine),
               abs(vm_glob - vm_fine), 0.0, "MPa"),
    ]


SUBMODEL_CASES: dict[str, Case] = {
    "SM2": Case("SM2", "Submodel ze studie: deska s otvorem, tah + tlak + tlak v otvoru + tize, vs jemny plny model",
                sm2_build, sm2_eval, run=sm2_run),
}
