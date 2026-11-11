"""Verification of the voxel detail of 3D printing (F5 B, fea_core.printing.voxel).  python -m verification.voxel_cases [X1 …]

X1  homogeneous field: (a) bar 40×10×10 of full extrusion (100 % infill, line ratios 1), tension 10 MPa – the detail in the middle
    through the whole chain (mesh, global ccx, BC from the tet10 shape functions, voxels, ccx C3D8I): σ = global,
    safety = homogenised; (b) unidirectional block of beads (anisotropy of the line of PLA 0.8 / 0.55) at the angle 0° and 30° with the displacements of
    uniaxial stress along the line → σ11 = σ0, E along the line = 2/(1 + r)·E_xy exactly.
X2  beam of voxels C3D8I in pure bending (prescribed rotation of the ends, 40 × 4 × 6 voxels) against the exact solution of
    the orthotropic beam: deflection κL²/2, σ11 = −E1·κ·z at the integration points, moment E1·I·κ; C3D8 informatively (locking).
X3  block P2 (verification.print_cases, 4 perimeters × 0.5 mm, grid 20 %, tension by jaws): the voxel detail of the middle
    part (the whole section, height 4 mm + margin 1 mm) against the homogenised model – force through the section, safety.
X4  path generator (without ccx): number and width of the perimeters, top/bottom layers, share of infill grid / gyroid / rectilinear.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from fea_core import materials as matlib
from fea_core.printing import PrintSpec, voxel as V, voxel_paths as vp
from fea_core.printing.infill import line_solid, solid
from verification.cases import Check

OUT = Path(__file__).resolve().parent / "out" / "voxel"
PLA = matlib.by_name("Tisk PLA Bambu Basic")


# ------------------------------------------------------------------ X1
def _bar_study(wd: Path):
    from fea_core.geometry import Part
    from fea_core.selectors import find_faces
    from fea_core.study import Displacement, Pressure, LoadCase, MeshSettings
    from verification.cases import _study, box
    st, parts = _study("X1 tyc", wd / "model.step", [Part("Tyc", box(0, 0, 0, 40, 10, 10))], MeshSettings(size=2.5))
    st.add_material(PLA, default=True)
    st.materials.pop("Ocel S235")
    st.printing.parts["Tyc"] = PrintSpec("Tyc", build_dir=[0, 0, 1], infill=1.0, walls=2, top_layers=3,
                                         bottom_layers=3)
    st.printing.materials[PLA.name] = {"line_E_ratio": 1.0, "line_R_ratio": 1.0}
    st.add_selection("x0", find_faces(parts, x=0))
    st.add_selection("y0", find_faces(parts, y=0))
    st.add_selection("z0", find_faces(parts, z=0))
    st.add_selection("cela", find_faces(parts, x=40))
    st.constraints += [Displacement("x0", ux=0.0), Displacement("y0", uy=0.0), Displacement("z0", uz=0.0)]
    st.load_cases.append(LoadCase("Tah 10 MPa", [Pressure("cela", -10.0)]))
    return st


def _block(angle: float, n=(12, 12, 8), dx=0.21, dz=0.2):
    g = vp.VoxelGrid(R=np.eye(3), origin=np.zeros(3), h=np.array([dx, dx, dz]),
                     kind=np.full(n, vp.PERIMETER, np.int8), angle=np.full(n, angle), wall=np.zeros(n, np.int8),
                     inside=np.ones(n, bool))
    return g, V.build_model(g)


def x1(wd: Path) -> tuple[list[Check], dict]:
    from fea_core.solve import run_study
    st = _bar_study(wd)
    res = run_study(st, wd / "global")
    t0 = time.time()
    spec = V.VoxelSpec(center=[20, 5, 5], size=[6, 6, 4], margin=1.0)
    vr = V.run_voxel(st, res, spec, wd / "voxel")
    t_vox = time.time() - t0
    s = vr.step(0)
    sm = vr.summary["steps"][0]
    sxx = s.S_el[:, 0]
    ms = solid(PLA)
    from fea_core.printing.failure import safety
    lam_hom = float(safety(np.array([10.0, 0, 0, 0, 0, 0]), ms, st.printing.criterion))
    # bead at an angle θ: σ in the axes of the line from the global σxx = 10 MPa; Tsai-Wu with Rc ≠ Rm and S = Rm/√3 is not invariant in the plane
    # of the layers under rotation → the worst class of the angle decides (layers ±45°), the homogenisation assesses in the print axes
    sg = np.array([10.0, 0, 0, 0, 0, 0])
    lam_ref = float(min(safety(V.to_local(sg, vr.axes[k]), ms, st.printing.criterion) for k in np.unique(vr.bins)))
    other = float(np.abs(s.S_el[:, 1:]).max())
    checks = [
        Check("(a) σxx voxelů = globální 10 MPa (průměr)", float(sxx.mean()), 10.0, 1e-4, "MPa"),
        Check("(a) σxx voxelů max. odchylka", float(10 + np.abs(sxx - 10).max()), 10.0, 1e-3, "MPa"),
        Check("(a) ostatní složky ≈ 0 (max |σ|)", 10 + other, 10.0, 1e-3, "MPa"),
        Check("(a) bezpečnost detailu = min. Tsai-Wu(σ v osách housenek)", sm["min_safety"], lam_ref, 1e-3, ""),
        Check("(a) bezpečnost homogenizovaná v oblasti = Tsai-Wu(10 MPa)", sm["homogenized_min_safety"], lam_hom,
              2e-3, ""),
        Check("(a) průměr σxx přes oblast: voxely × globální", sm["mean_stress_voxel"][0],
              sm["mean_stress_global"][0], 1e-3, "MPa"),
    ]
    info = {"a_voxels": vr.summary["voxels"], "a_dof": vr.summary["dof"], "a_time_s": round(t_vox, 1)}
    # (b) unidirectional block with the anisotropy of the line
    ml = line_solid(PLA, 0.8, 0.55)
    C = ml.stiffness()
    Sm = np.linalg.inv(C)
    sig0 = 5.0
    for deg in (0.0, 30.0):
        a = np.radians(deg)
        g, m = _block(a)
        axes = V.bin_axes(np.eye(3), 36)
        b = V._angle_bins(np.full(len(m["hexes"]), a), 36)
        eps_l = Sm @ np.array([sig0, 0, 0, 0, 0, 0])            # axes of the line 11 22 33 12 13 23 (engineering shear)
        A = axes[b[0]]
        El = np.array([[eps_l[0], eps_l[3] / 2, eps_l[4] / 2], [eps_l[3] / 2, eps_l[1], eps_l[5] / 2],
                       [eps_l[4] / 2, eps_l[5] / 2, eps_l[2]]])
        Eg = A.T @ El @ A                                       # strain globally
        U = m["nodes"] @ Eg.T
        nij = m["node_ijk"]
        bnd = np.nonzero(np.any((nij == 0) | (nij == np.array(g.shape)), axis=1))[0]
        raw = V.solve_voxels(wd / f"blok{int(deg)}", m["nodes"], m["hexes"], b, axes, C, bnd, [U[bnd]])
        vs = V.evaluate_steps(raw, ["blok"], b, axes, ml, "tsai_wu", m["hexes"], len(m["nodes"]))[0]
        s11 = V.to_local(raw[0]["S_ip"], axes[b][:, None])[..., 0]
        E_L = float(s11.mean() / eps_l[0])
        checks += [
            Check(f"(b) {deg:.0f}°: σ11 podél linie = σ0", float(s11.mean()), sig0, 1e-5, "MPa"),
            Check(f"(b) {deg:.0f}°: E podél linie = 2/(1+0,8)·E_xy", E_L, 2 * PLA.E / 1.8, 1e-5, "MPa"),
        ]
        if deg == 0.0:
            from fea_core.printing.failure import safety
            checks.append(Check("(b) 0°: Tsai-Wu λ = X_L/σ0", float(vs.safety_el.min()),
                                float(safety(np.array([sig0, 0, 0, 0, 0, 0]), ml, "tsai_wu")), 1e-5, ""))
    return checks, info


# ------------------------------------------------------------------ X2
def x2(wd: Path) -> tuple[list[Check], dict]:
    ml = line_solid(PLA, 0.8, 0.55)
    C = ml.stiffness()
    Sm = np.linalg.inv(C)
    E1 = 1 / Sm[0, 0]
    nu12, nu13 = -Sm[1, 0] * E1, -Sm[2, 0] * E1
    nx, ny, nz = 40, 4, 6
    g, m = _block(0.0, (nx, ny, nz))
    X = m["nodes"]
    L, B, H = nx * g.h[0], ny * g.h[1], nz * g.h[2]
    yc, zc = B / 2, H / 2
    kap = 1e-3
    x, eta, zeta = X[:, 0], X[:, 1] - yc, X[:, 2] - zc
    Uex = np.column_stack([-kap * x * zeta, nu12 * kap * zeta * eta,
                           kap * x ** 2 / 2 + nu13 * kap * zeta ** 2 / 2 - nu12 * kap * eta ** 2 / 2])
    nij = m["node_ijk"]
    x0, x1n = nij[:, 0] == 0, nij[:, 0] == nx
    A = np.nonzero(x0 & (nij[:, 1] == ny // 2) & (nij[:, 2] == nz // 2))[0][0]
    Bn = np.nonzero(x0 & (nij[:, 1] == ny) & (nij[:, 2] == nz // 2))[0][0]
    bc = np.nonzero(x0 | x1n)[0]
    dofs = np.zeros((len(bc), 3), bool)
    dofs[:, 0] = True
    dofs[bc == A] = True
    dofs[bc == Bn, 2] = True
    axes = V.bin_axes(np.eye(3), 36)
    b = np.zeros(len(m["hexes"]), np.int64)
    tip = np.nonzero(x1n & (nij[:, 1] == ny // 2) & (nij[:, 2] == nz // 2))[0][0]
    out, info = [], {}
    for et in ("C3D8I", "C3D8"):
        raw = V.solve_voxels(wd / et, X, m["hexes"], b, axes, C, bc, [Uex[bc]], element=et, bc_dofs=dofs)[0]
        w_tip = raw["U"][tip, 2]
        # integration points 2×2×2: z_ip = z_centre ± dz/(2√3); the order of the points of ccx – comparison by sorting per element
        zc_el = X[m["hexes"]].mean(axis=1)[:, 2] - zc
        s11 = raw["S_ip"][..., 0]                              # axis 1 = x (angle 0)
        dz = g.h[2] / (2 * np.sqrt(3))
        ex_lo, ex_hi = -E1 * kap * (zc_el - dz), -E1 * kap * (zc_el + dz)
        s_sorted = np.sort(s11, axis=1)
        ex = np.sort(np.column_stack([ex_lo] * 4 + [ex_hi] * 4), axis=1)
        err = float(np.abs(s_sorted - ex).max() / (E1 * kap * H / 2))
        # moment by Gauss integration with 2 points over the height (exact for σ linear in z): σ ascending ↔ z descending
        z_ip = np.column_stack([zc_el + dz] * 4 + [zc_el - dz] * 4)
        M = -float((s_sorted * z_ip).sum() / 8 * g.h[0] * g.h[1] * g.h[2] / L)
        I = B * H ** 3 / 12
        if et == "C3D8I":
            out += [Check("C3D8I průhyb konce = κL²/2", float(w_tip), kap * L ** 2 / 2, 1e-4, "mm"),
                    Check("C3D8I σ11 v int. bodech (max chyba / σ_max)", 1 + err, 1.0, 1e-4, ""),
                    Check("C3D8I moment Σσ·z·dV/L = E1·I·κ", M, E1 * I * kap, 1e-4, "N·mm")]
        else:
            info["C3D8_tip_ratio"] = float(w_tip / (kap * L ** 2 / 2))
            info["C3D8_moment_ratio"] = float(M / (E1 * I * kap))
            info["C3D8_sigma_error"] = err
            # C3D8 (locking) is only informative – it is stored in info, not as a check
    return out, info


# ------------------------------------------------------------------ X3
def x3(wd: Path) -> tuple[list[Check], dict]:
    from fea_core.solve import run_study
    from verification import print_cases as P
    st, _ = P.p2_build(wd)
    t0 = time.time()
    res = run_study(st, wd / "global")
    t_glob = time.time() - t0
    spec = V.VoxelSpec(center=[5.0, 5.0, 10.0], size=[12.0, 12.0, 4.0], margin=1.0, step=0)
    est = V.estimate(st, res, spec)
    t0 = time.time()
    vr = V.run_voxel(st, res, spec, wd / "voxel")
    t_vox = time.time() - t0
    vr.save(wd / "voxel" / "detail")
    for fld, z in (("kind", vr.shape[2] // 2), ("safety", vr.shape[2] // 2), ("safety", None)):
        V.render_png(vr, wd / f"detail_{fld}_{'iso' if z is None else z}.png", fld, slice_z=z)
    s = vr.step(0)
    sm = vr.summary["steps"][0]
    # force through the section: layers of the assessment region, Σ σzz·dx·dy (element = voxel)
    layers = np.unique(vr.ijk[vr.inner, 2])
    A_vox = vr.h[0] * vr.h[1]
    Fz = np.array([(s.S_el[(vr.ijk[:, 2] == k), 2]).sum() * A_vox for k in layers])
    F_glob = float(res.step(0).reactions["vrch"][2])
    # mixing rule with the real share of voxels: σzz = E_z·ε in the beads (along the build all the same)
    cnt = np.array([(vr.ijk[:, 2] == k).sum() for k in layers])
    nodes_top = vr.nodes[:, 2].max()
    nodes_bot = vr.nodes[:, 2].min()
    top = np.abs(vr.nodes[:, 2] - nodes_top) < 1e-9
    bot = np.abs(vr.nodes[:, 2] - nodes_bot) < 1e-9
    eps = float((s.U[top, 2].mean() - s.U[bot, 2].mean()) / (nodes_top - nodes_bot))
    F_mix = float(PLA.E_z * eps * cnt.mean() * A_vox)
    perim = vr.summary["kinds"]["perimetr"]
    # homogenised safety of the pure shell in the region: nodes whose elements all have a shell share of 1
    mesh = res.mesh
    f = mesh.cell_data["print_shell"]
    bad = np.zeros(len(mesh.nodes), bool)
    bad[np.unique(mesh.tets[f < 1])] = True
    zn = mesh.nodes[:, 2]
    sel = ~bad & (zn >= 8) & (zn <= 12)
    ps = res.step(0).fields["print_safety"][sel]
    lam_shell = float(np.nanmedian(ps))
    lam_shell_min = float(np.nanmin(ps))
    lam_vox_per = float(np.median(s.safety_el[vr.inner & (vr.kind == vp.PERIMETER)]))
    ref = P.p2_reference()
    g_frac = vr.summary["infill_fraction"]
    checks = [
        Check("síla průřezem: voxely × homogenizovaný (reakce)", float(Fz.mean()), F_glob, 0.03, "N"),
        Check("síla průřezem: voxely × E_z·ε·A_voxelů", float(Fz.mean()), F_mix, 0.03, "N"),
        Check("síla stejná ve všech vrstvách (max/min)", float(Fz.max() / Fz.min()), 1.0, 0.01, ""),
        Check("podíl výplně voxelů / zadání 20 %", g_frac, 0.2, 0.05, ""),
        Check("bezpečnost perimetrů: medián voxelů × medián skořepiny (f = 1)", lam_vox_per, lam_shell, 0.05, ""),
        # min. safety of the voxels × min. of the shell and the ratio detail / homogenised min. are not checks: the minimum at an
        # integration point depends on the steps of the voxels (singularity), they are only in info (voxel_min, homogenized_*_min).
    ]
    info = {"global_time_s": round(t_glob, 1), "voxel_time_s": round(t_vox, 1), "estimate": est,
            "summary_step": sm, "F_ref_mix_P2": ref["F"], "perimeter_voxels": perim,
            "F_layers": Fz.tolist(), "eps": eps, "voxel_min": sm["min_safety"],
            "voxel_min_element_avg": sm["min_safety_element_avg"], "homogenized_min": sm.get("homogenized_min_safety"),
            "homogenized_shell_min": lam_shell_min, "homogenized_shell_median": lam_shell,
            "voxel_perimeter_median": lam_vox_per}
    return checks, info


# ------------------------------------------------------------------ X4
def x4(wd: Path) -> tuple[list[Check], dict]:
    out, info = [], {}
    B = np.array([30.0, 30.0, 4.0])
    inside = lambda q: np.all((q > 0) & (q < B), axis=1)
    for pat, rho in (("grid", 0.15), ("gyroid", 0.2), ("rectilinear", 0.25)):
        sp = PrintSpec("A", walls=3, line_width=0.42, layer_height=0.2, top_layers=4, bottom_layers=3, infill=rho,
                       pattern=pat)
        g = vp.generate(sp, np.eye(3), np.zeros(3), -np.ones(3), B + 1, 0.21, inside, np.zeros(3), B)
        fr = vp.infill_fraction(g)
        tol = 0.02 if pat != "gyroid" else 0.03       # gyroid: the share in the layer changes with the height (period ~5 mm)
        out.append(Check(f"podíl výplně {pat} {rho:.0%}", fr, rho, tol, ""))
        info[pat] = fr
    sp = PrintSpec("A", walls=3, line_width=0.42, layer_height=0.2, top_layers=4, bottom_layers=3, infill=0.15)
    g = vp.generate(sp, np.eye(3), np.zeros(3), -np.ones(3), B + 1, 0.21, inside, np.zeros(3), B)
    mid = g.kind[:, :, g.shape[2] // 2]
    row = mid[:, g.shape[1] // 2]
    out += [
        Check("počet perimetrů = 3", len(np.unique(g.wall[g.wall >= 0])), 3, 0.0, ""),
        # 3·0.42 / 0.21 = 6 voxels per side; the edge of the strip falls into a voxel arbitrarily → quantisation ±1 voxel (1/6)
        Check("šířka perimetrů (voxely na straně, 3·w / (w/2))", (row == vp.PERIMETER).sum() / 2, 6, 1 / 6, "vox"),
        Check("horní vrstvy = 4", int((g.kind[g.shape[0] // 2, g.shape[1] // 2, -4:] == vp.SKIN).sum()), 4, 0.0, ""),
        Check("dolní vrstvy = 3", int((g.kind[g.shape[0] // 2, g.shape[1] // 2, :4] == vp.SKIN).sum()), 3, 0.0, ""),
    ]
    # thin place: wall 1.6 mm (inside 1.6 − 2·3·0.42 < 0) and rib 3 mm (inside 0.48 < 2w) → full infill
    Bt = np.array([3.0, 20.0, 2.0])
    g = vp.generate(sp, np.eye(3), np.zeros(3), -np.ones(3), Bt + 1, 0.21,
                    lambda q: np.all((q > 0) & (q < Bt), axis=1), np.zeros(3), Bt)
    core = g.inside & ~np.isin(g.kind, (vp.PERIMETER, vp.SKIN))
    out.append(Check("žebro 3 mm: vnitřek bez perimetrů = plná výplň", float((g.kind[core] == vp.THIN).mean()), 1.0,
                     0.0, ""))
    return out, info


CASES = {"X1": ("Homogenní pole: celý řetězec + jednosměrný blok", x1),
         "X2": ("Nosník C3D8I v čistém ohybu (přesné řešení)", x2),
         "X3": ("Hranol P2: voxelový detail × homogenizovaný model", x3),
         "X4": ("Generátor drah: perimetry, vrstvy, výplň", x4)}


def run(keys: list[str]) -> bool:
    ok_all = True
    report = {}
    for key in keys:
        title, fn = CASES[key]
        wd = OUT / key
        wd.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {key}: {title}")
        t0 = time.time()
        try:
            checks, info = fn(wd)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  CHYBA: {e}")
            ok_all = False
            report[key] = {"error": str(e)}
            continue
        dt = time.time() - t0
        for c in checks:
            flag = "OK  " if c.ok else "CHYBA"
            print(f"  [{flag}] {c.name:<56} {c.value:12.6g} {c.unit:<4} ref {c.ref:12.6g}  "
                  f"odch. {100 * c.error:7.3f} % (limit {100 * c.tol:g} %)")
            ok_all &= c.ok
        print(f"  čas {dt:.1f} s; " + json.dumps({k: v for k, v in info.items() if not isinstance(v, (dict, list))},
                                                  ensure_ascii=False, default=float))
        report[key] = {"title": title, "time_s": dt, "info": info,
                       "checks": [{"name": c.name, "value": float(c.value), "ref": float(c.ref),
                                   "error": float(c.error), "tol": c.tol, "unit": c.unit, "ok": bool(c.ok)}
                                  for c in checks]}
    (OUT / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("\nVÝSLEDEK:", "všechny kontroly prošly" if ok_all else "některé kontroly NEPROŠLY")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1:] or list(CASES)) else 1)
