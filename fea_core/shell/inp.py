"""Writing the ccx input for the shell model (S6, *SHELL SECTION, *BOUNDARY, *CLOAD by steps).

Conventions (verified by trials scratch/shell/exp1.py, exp2.py, ccx 2.23):
  • S6 is expanded to C3D15 (OUTPUT=3D – default): the .frd contains ONLY the expanded nodes (new numbers > the max. node)
    and the elements 3C (the same numbers as S6, type 5 = penta15) in the order of the frd: [0:3] corners at −t/2, [3:6] corners at +t/2,
    [6:9] midpoints of the edges at −t/2, [9:12] corners at the mid face, [12:15] midpoints of the edges at +t/2; "+" = normal of the element
    (x1 − x0) × (x2 − x0).
  • *BOUNDARY DOF 1–3 at a shell node = hinge (the rotation free), DOF 4–6 = rotations (ccx mid-surface rotation MPC);
    the rotation about the normal (drilling) cannot be constrained alone, only with all rotations.
  • *CLOAD at a shell node is split between the expanded nodes (1/6-2/3-1/6 corner, 1/2-1/2 midpoint); the sum of the RF
    of the expanded nodes of the constrained node = the reaction (+ the load in it).
  • *DLOAD P on S6 = force +p·n (in the direction of the normal of the element) – not used, the pressure goes through *CLOAD.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import solver_name
from ..ccx import deck_text


def _lines(vals, per=8):
    vals = list(vals)
    return [", ".join(str(v) for v in vals[k:k + per]) for k in range(0, len(vals), per)]


def write_shell_inp(path, mesh, materials: dict[int, object], bcs: dict[int, dict[int, float]],
                    steps: list[tuple[str, np.ndarray]], title: str = "FEA Studio – shells",
                    frames: dict[int, np.ndarray] | None = None, solver: str = "PARDISO") -> None:
    """materials: index of the part → Material (E, nu, rho); bcs: node (0-based) → {DOF: value};
    steps: [(name, F (N, 3))] – one *STEP per load state (forces in global axes);
    frames: node → local axes (3, 3) (*TRANSFORM, TYPE=R; constraints of the node in the local axes, forces converted).
    """
    frames = frames or {}
    L = ["*HEADING", deck_text(title), "*NODE, NSET=NALL"]
    L += ["%d, %.12g, %.12g, %.12g" % (i + 1, x, y, z) for i, (x, y, z) in enumerate(mesh.nodes)]
    for pi in range(len(mesh.part_names)):
        el = np.nonzero(mesh.tri_part == pi)[0]
        if not len(el):
            continue
        L.append(f"*ELEMENT, TYPE=S6, ELSET=P{pi}")
        tri = mesh.tri[el] + 1
        L += ["%d, %d, %d, %d, %d, %d, %d" % (e + 1, *t) for e, t in zip(el, tri)]
    used = [pi for pi in range(len(mesh.part_names)) if np.any(mesh.tri_part == pi)]
    L.append("*ELSET, ELSET=EALL")
    L += _lines([f"P{pi}" for pi in used])
    mats: dict[str, str] = {}
    for pi in used:
        m = materials[pi]
        if m.name not in mats:
            key = f"M{len(mats) + 1}"
            mats[m.name] = key
            L += [f"*MATERIAL, NAME={key}", "*ELASTIC", f"{float(m.E):.12g}, {float(m.nu):.12g}"]
            if m.rho:
                L += ["*DENSITY", f"{float(m.rho):.12g}"]
    for pi in used:
        off = float(mesh.offset[pi])
        L.append(f"*SHELL SECTION, ELSET=P{pi}, MATERIAL={mats[materials[pi].name]}"
                 + (f", OFFSET={off:g}" if off else ""))
        L.append(f"{float(mesh.thickness[pi]):.12g}")
    for v, T in sorted(frames.items()):
        L += [f"*NSET, NSET=NT{v + 1}", str(v + 1), f"*TRANSFORM, NSET=NT{v + 1}, TYPE=R",
              ", ".join("%.12g" % c for c in (*T[:, 0], *T[:, 1]))]
    if bcs:
        L.append("*BOUNDARY")
        for v in sorted(bcs):
            for d in sorted(bcs[v]):
                L.append("%d, %d, %d, %.12g" % (v + 1, d, d, bcs[v][d]))
    for k, (name, F) in enumerate(steps):
        L += ["*STEP", f"*STATIC, SOLVER={solver_name(solver)}"]       # without SOLVER ccx chooses PaStiX (10× slower)
        L.append("*CLOAD, OP=NEW")
        if frames:
            F = F.copy()
            for v, T in frames.items():
                F[v] = T.T @ F[v]                      # force in the local axes of the node
        nz = np.nonzero(np.abs(F) > 0)
        for v, c in zip(*nz):
            L.append("%d, %d, %.12g" % (v + 1, c + 1, F[v, c]))
        if not len(nz[0]):
            L.append("1, 1, 0.")                       # state without load (only prescribed displacements)
        L += ["*NODE FILE", "U, RF", "*EL FILE", "S"]
        L.append("*END STEP")
    Path(path).write_text("\n".join(L) + "\n", encoding="ascii", errors="replace")
