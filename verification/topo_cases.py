"""Verification case TO1 for topology optimisation (stage F9.7).

TO1: cantilever 120 x 40 x 10 mm (MBB / Michell) with BESO.
- Clamping at x = 0, vertical force Fy = -1000 N at x = 120; both faces are non-design (rho = 1).
- Target: 50 % of the volume. The real loop `TopologyOptimizer.run()`: in each iteration ccx with the stiffness of the elements E·max(rho^3, 1e-6)
  (`fea_core.ccx.inp.ElementScale`, no edits of the .inp text), sensitivities from the energy of the elements from the displacements.
- Checks: the volume reached, the compliance grows only a little (against the full cantilever) and is markedly LOWER than for a
  "naive" design of the same volume (the middle strip of half the height, the same FEM), stabilisation of the compliance after reaching the volume,
  the non-design regions, a watertight STL (each edge in 2 triangles, orientation, positive volume).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from fea_core.geometry import Part
from fea_core.results import Results
from fea_core.selectors import find_faces
from fea_core.solve import run_study
from fea_core.study import Fixed, Force, LoadCase, MeshSettings, Study
from fea_core.topo import TopologyOptSpec, TopologyOptimizer, topo_surface, write_binary_stl

from .cases import AtLeast, AtMost, Case, Check, _study, box, solid

TOPO_CASES: dict[str, Case] = {}


# ===================================================================== TO1
def to1_build(wd: Path) -> tuple[Study, list[Part]]:
    """TO1: cantilever 120 x 40 x 10 mm, clamping at x = 0, Fy = -1000 N at x = 120, mesh 5 mm."""
    plate = solid(box(0.0, 0.0, 0.0, 120.0, 40.0, 10.0))
    st, parts = _study("TO1 MBB konzola", wd / "model.step", [Part("MBB_Konzola", plate)], MeshSettings(size=5.0))
    st.add_selection("vetknuti", find_faces(parts, x=0.0))
    st.add_selection("sila", find_faces(parts, x=120.0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Svisla sila", [Force("sila", [0.0, -1000.0, 0.0])]))
    return st, parts


def _closed_report(faces: np.ndarray) -> tuple[int, int]:
    """(edges with a wrong number of triangles, edges without an oppositely oriented pair)."""
    und, directed = Counter(), Counter()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            und[frozenset((int(u), int(v)))] += 1
            directed[(int(u), int(v))] += 1
    bad_count = sum(1 for n in und.values() if n != 2)
    bad_orient = sum(1 for (u, v), n in directed.items() if n != 1 or directed.get((v, u), 0) != 1)
    return bad_count, bad_orient


def to1_run(study: Study, wd: Path, parts: list[Part]) -> Results:
    """Initial mesh (run_study) → BESO loop `run()` → comparison with the naive design → watertight STL."""
    wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    res0 = run_study(study, wd / "iter_0", parts=parts)
    mesh = res0.mesh

    spec = TopologyOptSpec(
        target_volume_fraction=0.50,
        evolution_rate=0.06,
        filter_radius=8.0,
        penalty=3.0,
        max_iterations=15,
        non_design_selections=["vetknuti", "sila"],
        min_density=1e-3,
    )
    opt = TopologyOptimizer(study, mesh, spec)
    opt.run(workdir=wd / "opt")

    # naive design of the same volume: the middle strip of half the height (y), the same FEM and the same energy
    optimum_rho = opt.rho.copy()
    band = np.abs(opt.centroids[:, 1] - 20.0) <= 10.0
    naive = np.where(band | opt.non_design_mask, 1.0, spec.min_density)
    opt.rho = naive
    naive_vf = opt.current_volume_fraction
    naive_compliance = opt.current_compliance(opt.evaluate(wd / "naive"))
    opt.rho = optimum_rho

    # watertight STL
    verts, faces = topo_surface(mesh, opt.rho, 0.5)
    stl_path = wd / "TO1_mbb_optimized.stl"
    write_binary_stl(stl_path, verts, faces)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    stl_volume = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)

    setattr(res0, "opt", opt)
    setattr(res0, "stl_path", stl_path)
    setattr(res0, "naive", {"vf": naive_vf, "compliance": naive_compliance})
    setattr(res0, "stl_report", {"faces": len(faces), "closed": _closed_report(faces), "volume": stl_volume})
    return res0


def to1_eval(res: Results, st: Study) -> list[Check]:
    opt = getattr(res, "opt", None)
    if opt is None:
        raise RuntimeError("Optimalizátor TopologyOptimizer nebyl nalezen ve výsledcích.")
    spec = opt.spec
    hist = opt.history
    c_hist = np.array([h.total_compliance for h in hist])
    c0, c_fin = opt.initial_compliance, opt.final_compliance
    naive = res.naive
    stl = res.stl_report

    # after the target volume is reached the compliance must not grow substantially (the end against the best value since reaching the volume)
    reached = [i for i, h in enumerate(hist) if abs(h.volume_fraction - spec.target_volume_fraction) <= 0.01]
    tail = c_hist[reached[0]:] if reached else c_hist
    drift = float(c_fin / np.min(np.append(tail, c_fin)) - 1.0)

    # an empty mask of non-design = error (previously a substitute value 1.0 silently satisfied the check)
    nd_mean = float(np.mean(opt.rho[opt.non_design_mask])) if opt.non_design_mask.any() else float("nan")
    # STL: isosurface ρ = 0.5 on the nodal densities (weighted transfer element → node) vs the volume of the elements with ρ > 0.5; the isosurface
    # lies inside the elements at the transition, the difference of the volumes is of the order of one layer of elements at the boundary (measured −0.9 %)
    solid_vol = float(np.sum(opt.volumes * (opt.rho > 0.5)))
    ratio = float(c_fin / c0)

    return [
        Check("TO1 Cilovy objemovy zlomek (cil 50 %)", opt.current_volume_fraction, spec.target_volume_fraction, 0.005, "-"),
        # Removing material cannot reduce the compliance (≥ 1), and a uniform thinning of the plate to 50 % of the thickness (an admissible design of
        # the same volume) has exactly 2× the compliance → the optimum must not be worse than 2× the full cantilever.
        AtLeast("TO1 Poddajnost optimum / plna konzola (>= 1)", ratio, 1.0, 1e-6, "-"),
        AtMost("TO1 Poddajnost optimum / plna konzola (<= 2 = rovnomerne ztenceni na 50 %)", ratio, 2.0, 1e-6, "-"),
        AtMost("TO1 Poddajnost optimalizovaneho / naivniho pasu stejneho objemu (< 1)",
               float(c_fin / naive["compliance"]), 1.0, 1e-6, "-"),
        AtLeast("TO1 Naivni pas ma aspon tolik materialu jako optimum (spravedlive srovnani)",
                naive["vf"], opt.current_volume_fraction - 0.01, 1e-6, "-"),
        AtMost("TO1 Poddajnost po dosazeni objemu: konec / nejlepsi (<= +5 %)", drift, 0.05, 1e-6, "-"),
        Check("TO1 Non-design oblasti zachovany (rho=1.0)", nd_mean, 1.0, 0.001, "-"),
        Check("TO1 STL: vodotesny (kazda hrana ve 2 trojuhelnicich)", float(stl["closed"][0] == 0), 1.0, 0.001, "-"),
        Check("TO1 STL: orientace plochy konzistentni a kladny objem",
              float(stl["closed"][1] == 0 and stl["volume"] > 0), 1.0, 0.001, "-"),
        Check("TO1 STL: objem uzavrene plochy ~ objem pevnych prvku (+-5 %)", stl["volume"], solid_vol, 0.05, "mm3"),
    ]


TOPO_CASES["TO1"] = Case(
    "TO1",
    "Konzola 120x40x10 mm – BESO se zpětnou vazbou do MKP, srovnání s naivním návrhem, vodotěsný STL",
    to1_build,
    to1_eval,
    run=to1_run,
)
