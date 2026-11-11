"""Runs the verification cases and prints a table.  python -m verification.run_all [V1 V2 ...]"""
from __future__ import annotations

import json
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
from pathlib import Path

from fea_core.solve import run_study
from verification import cases as C

OUT = Path(__file__).resolve().parent / "out"


def run(keys: list[str], backend: str = "ccx") -> bool:
    all_ok = True
    report = {}
    for key in keys:
        case = C.CASES[key]
        wd = OUT / (key if backend == "ccx" else f"{key}_{backend}")
        wd.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"\n=== {key}: {case.title}")
        try:
            study, parts = case.build(wd)
            study.solver.backend = backend
            study.save(wd / "study.json")
            if case.run is not None:                   # own calculation (mechanisms – run_motion)
                res = case.run(study, wd / "run", parts)
            else:
                res = run_study(study, wd / "run")
            checks = case.evaluate(res, study)
        except Exception as e:  # continue with the other cases
            import traceback
            traceback.print_exc()
            print(f"  CHYBA: {e}")
            all_ok = False
            report[key] = {"error": str(e)}
            continue
        if key == "V1":
            C._V1_REF["tip"] = -C._tip_deflection(res)
        dt = time.time() - t0
        mesh = getattr(res, "mesh", None)
        if mesh is not None and getattr(mesh, "report", None):
            print(f"  síť: {mesh.report['nodes']} uzlů, {mesh.report['elements']} prvků, "
                  f"min SJ {mesh.report['min_scaled_jacobian']:.3f}, čas {dt:.1f} s")
        else:
            print(f"  čas {dt:.1f} s")
        for w in res.warnings:
            print(f"  varování: {w}")
        for c in checks:
            flag = "OK  " if c.ok else "CHYBA"
            print(f"  [{flag}] {c.name:<52} {c.value:12.6g} {c.unit:<4} ref {c.ref:12.6g}  "
                  f"odch. {100 * c.error:6.3f} % (limit {100 * c.tol:g} %)")
            all_ok &= c.ok
        rep = mesh.report if mesh is not None and getattr(mesh, "report", None) else {}
        report[key] = {
            "title": case.title, "time_s": dt, "nodes": rep.get("nodes"),
            "elements": rep.get("elements"), "warnings": res.warnings,
            "checks": [{"name": c.name, "value": float(c.value), "ref": float(c.ref), "error": float(c.error),
                        "tol": c.tol, "unit": c.unit, "ok": bool(c.ok)} for c in checks],
        }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nVÝSLEDEK:", "všechny kontroly prošly" if all_ok else "některé kontroly NEPROŠLY")
    return all_ok


if __name__ == "__main__":
    args = sys.argv[1:]
    backend = "ccx"
    if "--backend" in args:
        i = args.index("--backend")
        backend = args[i + 1]
        del args[i:i + 2]
    keys = args or list(C.CASES)
    sys.exit(0 if run(keys, backend) else 1)
