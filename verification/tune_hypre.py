"""Tuning of BoomerAMG on a stored system.  python -m verification.tune_hypre h [--np N] [--prep]

--prep: mesh (plate + foot, size h) → NGSolve K → loads, Dirichlet → out_bench/sys_h.npz
without --prep: runs through the variants of the AMG settings and prints the iterations, times and memory.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

OUT = Path(__file__).resolve().parent / "out_bench"

VARIANTS = {
    "mfem_elasticity": {},
    "vychozi": {},
    "agg1": {"agg_levels": 1},
    "agg1_pmax2": {"agg_levels": 1, "pmax": 2},
    "agg2": {"agg_levels": 2},
    "bez_rbm": {"interp_vec": 0},
    "theta_0.25": {"theta": 0.25},
    "theta_0.8": {"theta": 0.8},
    "sweeps_2": {"sweeps": 2},
    "bez_nodal": {"nodal": 0},
    "interp_ext+e_18": {"interp": 18},
    "agg_1": {"agg_levels": 1},
    "relax_sgs_6": {"relax": 6},
    "jacobi18": {"relax": 18, "relax_coarse": 18},
    "jacobi18_bez_nodal": {"nodal": 0, "relax": 18, "relax_coarse": 18},
    "jacobi18_sweeps2": {"relax": 18, "relax_coarse": 18, "sweeps": 2},
    "jacobi18_bez_nodal_sw2": {"nodal": 0, "relax": 18, "relax_coarse": 18, "sweeps": 2},
}


def prep(h: float):
    from fea_core.geometry import read_step
    from fea_core.selectors import find_faces
    from fea_core.mesh import generate_mesh
    from fea_core.study import Study, Fixed, Force, LoadCase
    from fea_core import materials, hypre
    from fea_core.loads import nodal_loads
    from verification.bench_scaling import model
    import subprocess
    step = model(OUT)
    parts = read_step(step)
    st = Study("tune", str(step))
    st.add_material(materials.default())
    st.add_selection("fix", find_faces(parts, part="Patka", x=0))
    st.add_selection("F", find_faces(parts, part="Deska", x=200))
    st.constraints.append(Fixed("fix"))
    st.load_cases.append(LoadCase("LC", [Force("F", [0, 0, -500.0])]))
    job = OUT / f"tune_{h}"
    job.mkdir(parents=True, exist_ok=True)
    sels = {k: [f.to_dict() for f in s.faces] for k, s in st.selections.items()}
    t = time.time()
    m = generate_mesh(parts, sels, {"size": h}, job / "mesh")
    print(f"síť {time.time() - t:.1f} s, {len(m.nodes)} uzlů", flush=True)
    m.save(job / "mesh.npz")
    (job / "materials.json").write_text(json.dumps([{"E": 210000.0, "nu": 0.3}] * len(m.part_names)))
    t = time.time()
    subprocess.run([sys.executable, "-m", "fea_core.ngs.worker", "assemble", str(job)], check=True)
    print(f"sestavení {time.time() - t:.1f} s", json.loads((job / "assemble.json").read_text()), flush=True)
    K = sp.load_npz(job / "K.npz")
    F = nodal_loads(st, m, st.load_cases[0])[0].ravel()
    fixed = np.zeros((len(m.nodes), 3), bool)
    fixed[m.surface_nodes("fix")] = True
    Kb, Fb = hypre.apply_dirichlet(K, F, fixed.ravel(), np.zeros(fixed.size))
    sp.save_npz(OUT / f"sys_{h}.npz", Kb)
    np.save(OUT / f"rhs_{h}.npy", Fb)
    np.save(OUT / f"rbm_{h}.npy", hypre.rotation_modes(m.nodes))


def tune(h: float, nproc: int, names: list[str]):
    from fea_core import hypre
    K = sp.load_npz(OUT / f"sys_{h}.npz").tocsr()
    F = np.load(OUT / f"rhs_{h}.npy")
    R = np.load(OUT / f"rbm_{h}.npy")
    ref = None
    for name in names:
        t = time.time()
        try:
            X, info = hypre.solve(K, F, R, OUT / f"hyp_{h}", nproc=nproc, amg=VARIANTS[name], maxiter=3000)
            c = info["cases"][0]
            if ref is None:
                ref = X
            d = float(np.abs(X - ref).max() / np.abs(ref).max())
            print(f"{name:18s} iter {c['iterations']:5d}  setup {info['t_setup']:6.1f} s  řešení {c['t_solve']:6.1f} s"
                  f"  celkem {time.time() - t:6.1f} s  paměť {info['peak_gb_total']:5.1f} GB  Δx {d:.1e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name:18s} CHYBA {str(e)[:200]}", flush=True)


def hier_S(m):
    """u_nodes = S·u_hier (scalar): corners unchanged, middle of the edge = average of the ends + B_MID·c."""
    from fea_core.ngs.worker import CCX_EDGES, B_MID
    N = len(m.nodes)
    corner = np.unique(m.tets[:, :4])
    rows, cols, vals = list(corner), list(corner), [1.0] * len(corner)
    pairs = {}
    for a, b, c in CCX_EDGES:
        for x, y, z in zip(m.tets[:, a], m.tets[:, b], m.tets[:, c]):
            pairs[int(z)] = (int(x), int(y))
    for mnode, (x, y) in pairs.items():
        rows += [mnode, mnode, mnode]
        cols += [x, y, mnode]
        vals += [0.5, 0.5, B_MID]
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N)), corner


def tune2(h: float, nproc: int):
    from fea_core import hypre
    from fea_core.mesh import Mesh
    job = OUT / f"tune_{h}"
    m = Mesh.load(job / "mesh.npz")
    K = sp.load_npz(job / "K.npz").tocsr()
    F = np.load(OUT / f"rhs_{h}.npy")[:, 0]
    R = np.load(OUT / f"rbm_{h}.npy")
    fixed = np.zeros((len(m.nodes), 3), bool)
    fixed[m.surface_nodes("fix")] = True
    fixed = fixed.ravel()
    F[fixed] = 0.0
    S1, corner = hier_S(m)
    Sv = sp.kron(S1, sp.identity(3), format="csr")
    Kn, Fn = hypre.apply_dirichlet(K, F, fixed, np.zeros(fixed.size))
    Kh, Fh = hypre.apply_dirichlet((Sv.T @ K @ Sv).tocsr(), Sv.T @ F, fixed, np.zeros(fixed.size))
    Rh = R.copy()
    is_mid = np.ones(len(m.nodes), bool)
    is_mid[corner] = False
    Rh[np.repeat(is_mid, 3)] = 0.0
    cp = (3 * corner[:, None] + np.arange(3)[None, :]).ravel()
    ref = None
    cases = [("uzlová (výchozí)", Kn, Fn, R, None, {}, False),
             ("hierarch. (výchozí)", Kh, Fh, Rh, None, {}, True),
             ("uzlová, theta 0.25", Kn, Fn, R, None, {"theta": 0.25}, False),
             ("uzlová + C-body P1", Kn, Fn, R, cp, {"theta": 0.25}, False),
             ("hierarch., theta 0.25", Kh, Fh, Rh, None, {"theta": 0.25}, True),
             ("hierarch. + C-body P1", Kh, Fh, Rh, cp, {"theta": 0.25}, True),
             ("hierarch. + C-body, theta 0.5", Kh, Fh, Rh, cp, {}, True)]
    for name, A, b, B, cps, opts, hier in cases:
        t = time.time()
        try:
            X, info = hypre.solve(A, b, B, OUT / f"hyp_{h}", nproc=nproc, amg=opts, maxiter=3000, cpoints=cps)
            Xn = Sv @ X[:, 0] if hier else X[:, 0]
            if ref is None:
                ref = Xn
            c = info["cases"][0]
            print(f"{name:30s} iter {c['iterations']:5d}  setup {info['t_setup']:6.1f} s  řešení {c['t_solve']:6.1f} s"
                  f"  celkem {time.time() - t:6.1f} s  paměť {info['peak_gb_total']:5.1f} GB"
                  f"  Δx {float(np.abs(Xn - ref).max() / np.abs(ref).max()):.1e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name:30s} CHYBA {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    if "--pmg" in sys.argv:
        tune2(float(sys.argv[1]), int(sys.argv[sys.argv.index("--np") + 1]) if "--np" in sys.argv else 8)
        sys.exit(0)
    h = float(sys.argv[1])
    nproc = int(sys.argv[sys.argv.index("--np") + 1]) if "--np" in sys.argv else 8
    if "--prep" in sys.argv:
        prep(h)
    else:
        only = [a for a in sys.argv[2:] if a in VARIANTS]
        tune(h, nproc, only or list(VARIANTS))
