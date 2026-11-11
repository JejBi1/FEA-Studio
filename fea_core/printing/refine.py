"""Two-step refinement of the mesh for 3D printing (F5 step 4, docs/projekt/F5_plan.md A5).

A coarse mesh against the shell (perimeters ~0.8 mm) gives a stress in the shell that depends on the mesh. A global refinement to the thickness of the shell is
unaffordable (F4: 507 k nodes, 15 GB) → after the first calculation the places with the lowest safety of the printing are found (separated into clusters, ≤ n_spots)
and in the spheres around them the mesh is refined to ~ the thickness of the shell (MeshSettings.local with center/radius,
Gmsh field "Ball"), then a second calculation. The result compares the min. safety.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np


def spots(study, results, n_spots: int = 3, steps=None) -> list[dict]:
    """Refinement spheres [{"center", "radius", "size", "part", "safety"}] at the places of the lowest printing safety."""
    mesh = results.mesh
    X = mesh.nodes
    worst = np.full(len(X), np.inf)
    for k, st in enumerate(results.steps):
        if steps is not None and k not in steps:
            continue
        v = st.fields.get("print_safety")
        if v is not None:
            worst = np.minimum(worst, v)
    if not np.isfinite(worst).any():
        return []
    # part of the node (the last one by order) and its shell thickness
    owner = np.full(len(X), -1)
    for pi in range(len(mesh.part_names)):
        owner[np.unique(mesh.tets[mesh.tet_part == pi])] = pi
    h_glob = float(study.mesh.size)
    out: list[dict] = []
    for i in np.argsort(worst):
        if not np.isfinite(worst[i]) or len(out) >= n_spots:
            break
        part = mesh.part_names[owner[i]] if owner[i] >= 0 else None
        sp = study.printing.spec(part) if part else None
        t = min(sp.wall_thickness, sp.top_layers * sp.layer_height) if sp else 0.8
        size = max(min(t, h_glob / 2), 0.25)
        radius = max(4 * t, 2.5 * h_glob)
        if any(np.linalg.norm(X[i] - np.asarray(o["center"])) < 1.5 * o["radius"] for o in out):
            continue
        out.append({"center": X[i].tolist(), "radius": float(radius), "size": float(size), "part": part,
                    "safety": float(worst[i])})
    return out


def refined_study(study, balls: list[dict]):
    st = copy.deepcopy(study)
    st.mesh.local = list(st.mesh.local) + [{"center": b["center"], "radius": b["radius"], "size": b["size"]}
                                           for b in balls]
    return st


def two_step(study, workdir, results=None, n_spots: int = 3, study_dir=None, parts=None, on_log=None,
             cancel=None) -> dict:
    """1st calculation (if results are not there) → refinement spheres → 2nd calculation. Returns {"first", "second", "balls", "study",
    "min_safety_before", "min_safety_after", "change"}.
    """
    from ..solve import run_study
    log = on_log or (lambda s: None)
    workdir = Path(workdir)
    if results is None:
        log("Print refinement: 1st analysis")
        results = run_study(study, workdir / "krok_1", study_dir=study_dir, on_log=log, cancel=cancel, parts=parts)
    balls = spots(study, results, n_spots)
    if not balls:
        raise ValueError("Print refinement: results have no print safety factor (study without printed parts)")
    st2 = refined_study(study, balls)
    log(f"Print refinement: 2nd analysis with refinement at {len(balls)} locations (element {min((b['size'] for b in balls)):.2g} mm)")
    res2 = run_study(st2, workdir / "krok_2", study_dir=study_dir, on_log=log, cancel=cancel, parts=parts)

    def mins(res):
        best = (np.inf, None)
        for s in res.steps:
            v = s.fields.get("print_safety")
            # unprinted nodes have NaN (the printing safety is not evaluated in them)
            if v is not None and np.isfinite(v).any() and float(np.nanmin(v)) < best[0]:
                best = (float(np.nanmin(v)), int(np.nanargmin(v)))
        return best
    (a, _), (b, nb) = mins(results), mins(res2)
    out = {"first": results, "second": res2, "balls": balls, "study": st2, "min_safety_before": a,
           "min_safety_after": b, "change": (b - a) / a if a else None,
           "nodes_before": int(len(results.mesh.nodes)), "nodes_after": int(len(res2.mesh.nodes)),
           "at_after": res2.mesh.nodes[nb].tolist() if nb is not None else None, "warnings": []}
    # minimum at a sharp inner edge = singularity (the stress grows with refinement, the value depends on the mesh)
    from ..mcp.diagnostics import mesh_cache
    re_idx, re_tree = mesh_cache(res2).reentrant
    if re_tree is not None and nb is not None:
        d, _ = re_tree.query(res2.mesh.nodes[nb])
        h = min(b_["size"] for b_ in balls)
        if d <= 1.5 * h:
            out["singular"] = True
            X = res2.mesh.nodes
            dist, _ = re_tree.query(X)
            far = dist > 1.5 * h
            vals = [s.fields["print_safety"][far] for s in res2.steps if "print_safety" in s.fields]
            vals = [v for v in vals if np.isfinite(v).any()]
            if far.any() and vals:
                out["min_safety_regular"] = float(min(np.nanmin(v) for v in vals))
            out["warnings"].append(
                f"After refinement the minimum safety factor lies at a sharp internal edge ({d:.2g} mm) – singularity: the value decreases with every refinement and has no finite limit. The actual corner fillet governs (add a radius in CAD), or evaluate away from the edge; do not use the corner value alone for design.")
    if out["change"] is not None and out["change"] < -0.1 and not out.get("singular"):
        out["warnings"].append(f"Refinement reduced the minimum safety factor by {-100 * out['change']:.0f} % – the coarse mesh did not capture the shell; rely on the finer analysis (or refine again).")
    return out
