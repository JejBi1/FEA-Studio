"""Writing of the ccx input for the 2D problem (CAX6 / CPS6 / CPE6), nodes in the plane x-y (z = 0).

One load state = one *STEP (constraints and loads OP=NEW from the 2nd step, as in 3D). Constant pressure = *DLOAD Pk
(face k of the element), other loads = consistent nodal forces *CLOAD (for CAX over the whole circumference, for CPS/CPE over the whole thickness –
verified scratch/twod/exp1.py). *SOLID SECTION: for CPS/CPE the thickness on the next line, for CAX it is not written.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import solver_name
from .convert import Model2D
from .model import ELEMENT
from ..ccx import deck_text


def _fmt(v: float) -> str:
    return f"{float(v):.12g}"


def _ascii(s: str) -> str:
    return deck_text(s)


def _ids(f, ids) -> None:
    ids = [int(i) + 1 for i in ids]
    for k in range(0, len(ids), 12):
        f.write(", ".join(map(str, ids[k:k + 12])) + "\n")


def write_inp(model: Model2D, path: str | Path, solver: str = "PARDISO") -> Path:
    path = Path(path)
    m = model.mesh
    et = ELEMENT[model.kind]
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(f"*HEADING\nFEA Studio 2D ({et})\n*NODE, NSET=NALL\n")
        for i, (x, y) in enumerate(m.nodes):
            f.write(f"{i + 1}, {_fmt(x)}, {_fmt(y)}, 0\n")
        for pi, _ in enumerate(m.part_names):
            idx = np.nonzero(m.tri_part == pi)[0]
            if not len(idx):
                continue
            f.write(f"*ELEMENT, TYPE={et}, ELSET=P{pi + 1}\n")
            for e in idx:
                f.write(f"{e + 1}, " + ", ".join(str(int(v) + 1) for v in m.tri[e]) + "\n")
        mats: dict[str, int] = {}
        for pi, mat in enumerate(model.materials):
            if mat.name not in mats:
                mats[mat.name] = len(mats) + 1
                f.write(f"** {_ascii(mat.name)}\n*MATERIAL, NAME=M{mats[mat.name]}\n*ELASTIC\n"
                        f"{_fmt(mat.E)}, {_fmt(mat.nu)}\n")
                if mat.rho:
                    f.write(f"*DENSITY\n{_fmt(mat.rho)}\n")
        for pi, mat in enumerate(model.materials):
            if not np.any(m.tri_part == pi):
                continue
            f.write(f"** part {_ascii(m.part_names[pi])}\n*SOLID SECTION, ELSET=P{pi + 1}, MATERIAL=M{mats[mat.name]}\n")
            if model.kind != "axisym":
                f.write(f"{_fmt(model.thickness)}\n")
        for k, bc in enumerate(model.bcs):
            f.write(f"** constraint {_ascii(bc.selection)}\n*NSET, NSET=C{k + 1}\n")
            _ids(f, bc.nodes)
        for si, st in enumerate(model.steps):
            op = "" if si == 0 else ", OP=NEW"
            f.write(f"** load case {si + 1}: {_ascii(st.name)}\n*STEP\n*STATIC, SOLVER={solver_name(solver)}\n")
            f.write(f"*BOUNDARY{op}\n")
            for k, bc in enumerate(model.bcs):
                for dof, v in sorted(bc.dofs.items()):
                    f.write(f"C{k + 1}, {dof}, {dof}, {_fmt(v)}\n")
            nn, dd = np.nonzero(st.cload != 0.0)
            if len(nn) or op:
                f.write(f"*CLOAD{op}\n")
                for a, b in zip(nn, dd):
                    f.write(f"{a + 1}, {b + 1}, {_fmt(st.cload[a, b])}\n")
            if st.dload or op:
                f.write(f"*DLOAD{op}\n")
                for e, face, p in st.dload:
                    f.write(f"{e + 1}, P{face}, {_fmt(p)}\n")
            f.write("*NODE FILE\nU, RF\n*EL FILE\nS\n*END STEP\n")
    return path
