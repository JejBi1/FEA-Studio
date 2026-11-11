"""3D printing: NGSolve + hypre against ccx on the SAME mesh (P1–P5).  python -m verification.print_ngs_check [P1 P3 …]

Orthotropic material per element (C in global axes, the same groups as *ORIENTATION in ccx), the weight with the density
per element. It compares the displacements (max |ΔU| / max |U|), the nodal stresses (max |ΔS| / max |S| – the projection to the nodes
differs between ccx and NGSolve, at the material jumps), the reactions and the print safety.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from fea_core.solve import run_study
from verification import cases as _C  # noqa: F401 – first the cases (cyclic import of print_cases)
from verification.print_cases import PRINT_CASES

OUT = Path(__file__).resolve().parent / "out" / "print_ngs"


def compare(key: str, nproc: int = 4) -> list[str]:
    case = PRINT_CASES[key]
    wd = OUT / key
    wd.mkdir(parents=True, exist_ok=True)
    st, _ = case.build(wd)
    st.solver.backend = "ccx"
    rc = run_study(st, wd / "ccx")
    st.solver.backend = "ngs_hypre"
    st.solver.nproc = nproc
    rn = run_study(st, wd / "ngs", mesh=rc.mesh)
    out = []
    for a, b in zip(rc.steps, rn.steps):
        du = np.abs(a.U - b.U).max() / np.abs(a.U).max()
        ds = np.abs(a.S - b.S).max() / np.abs(a.S).max()
        ds_med = np.median(np.abs(a.S - b.S).max(axis=1)) / np.abs(a.S).max()
        rs = max(np.abs(v).max() for v in a.reactions.values())
        dr = max(np.abs(a.reactions[k] - b.reactions[k]).max() for k in a.reactions) / rs
        la = [d["min_safety"] for d in a.printing.get("parts", {}).values()]
        lb = [d["min_safety"] for d in b.printing.get("parts", {}).values()]
        dl = max(abs(x - y) / x for x, y in zip(la, lb)) if la else 0.0
        out.append(f"{key} {a.name:<40} U {du:.1e}  S max {ds:.1e} (medián {ds_med:.1e})  reakce {dr:.1e}  "
                   f"λ tisku {dl:.1e}")
    try:
        chk = case.evaluate(rn, st)
        out += [f"   NGSolve: [{'OK' if c.ok else 'CHYBA'}] {c.name}: {c.value:.6g} ref {c.ref:.6g} "
                f"({100 * c.error:.3f} %)" for c in chk]
    except Exception as e:                       # noqa: BLE001
        out.append(f"   NGSolve vyhodnocení: {e}")
    return out


if __name__ == "__main__":
    for k in sys.argv[1:] or ["P1", "P2", "P3", "P4"]:
        for line in compare(k):
            print(line, flush=True)
