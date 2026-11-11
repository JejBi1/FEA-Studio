"""Cross-check V1–V7 in MYSTRAN: the same (coarser) mesh in ccx and MYSTRAN, comparison of the displacements and reactions.

  python -m verification.mystran_check [V1 V3 ...] [--coarse 2.0] [--jobs N]

MYSTRAN calculates on one core and its processing of the elements is slow (~9 ms/node; it is neither the solver of the system
nor the disk – verified 12. 10. 2026), therefore:
  - the mesh is coarsened (element size × coarse); the analytic values are checked by run_all,
  - the ccx cases run one after another (they use all cores themselves), the MYSTRAN cases then in parallel.
Criterion: max |Δu| / max |u| < 1e-3, |ΔR| / |R| < 1e-3.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fea_core.mystran import compare, prepare, run_mystran
from fea_core.solve import run_study
from verification import cases as C

OUT = Path(__file__).resolve().parent / "out_mystran"
TOL = 1e-3


def run(keys: list[str], coarse: float, jobs: int) -> bool:
    t_all = time.time()
    ok_all = True
    report = {}
    ready = []                                        # (key, study, results, workdir)
    for key in keys:                                  # 1) ccx one after another
        case = C.CASES[key]
        wd = OUT / key
        wd.mkdir(parents=True, exist_ok=True)
        try:
            st, _ = case.build(wd)
            st.mesh.size *= coarse
            for loc in st.mesh.local:
                loc["size"] *= coarse
            res = run_study(st, wd / "ccx")
            ready.append((key, st, res, prepare(st, res, wd / "mystran")))
            print(f"ccx {key}: {len(res.mesh.nodes)} uzlů", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"=== {key}: CHYBA ccx: {e}")
            report[key] = {"error": str(e)}
            ok_all = False

    def solve(item):                                  # 2) MYSTRAN in parallel
        t0 = time.time()
        try:
            run_mystran(item[3])
            return item[0], None, time.time() - t0
        except Exception as e:  # noqa: BLE001
            return item[0], str(e), time.time() - t0

    t_my = time.time()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        done = {k: (err, dt) for k, err, dt in pool.map(solve, ready)}
    print(f"MYSTRAN: {len(ready)} úloh souběžně ({jobs} najednou) za {time.time() - t_my:.1f} s")

    for key, st, res, wd in ready:                    # 3) comparison
        err, dt = done[key]
        print(f"\n=== {key}: {C.CASES[key].title}  ({len(res.mesh.nodes)} uzlů, MYSTRAN {dt:.1f} s)")
        if err:
            print(f"  CHYBA: {err}")
            report[key] = {"error": err}
            ok_all = False
            continue
        rows = compare(st, res, wd)
        for r in rows:
            ok = r["du_rel"] < TOL and r["dr_rel"] < TOL
            ok_all &= ok
            print(f"  [{'OK  ' if ok else 'CHYBA'}] {r['case']}: max u ccx {r['u_max_ccx']:.6g} / MYSTRAN "
                  f"{r['u_max_mystran']:.6g} mm, Δu {r['du_rel']:.2e}, ΔR {r['dr_rel']:.2e}")
        report[key] = rows
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nVÝSLEDEK: {'ccx a MYSTRAN se shodují' if ok_all else 'NESHODA'}  (celkem {time.time() - t_all:.0f} s)")
    return ok_all


if __name__ == "__main__":
    args = sys.argv[1:]
    opts = {"--coarse": 2.0, "--jobs": max(1, (os.cpu_count() or 2) // 2)}
    for k in list(opts):
        if k in args:
            i = args.index(k)
            opts[k] = type(opts[k])(args[i + 1])
            del args[i:i + 2]
    keys = args or [k for k in C.CASES if k.startswith("V")]
    sys.exit(0 if run(keys, opts["--coarse"], int(opts["--jobs"])) else 1)
