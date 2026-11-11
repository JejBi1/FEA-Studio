"""Check of the installation and precompilation of numba (called by the installer): `python -m fea_core.warmup [--quick]`.

The numba functions (@njit(cache=True): fatigue – rainflow and critical plane, 3D printing – shell distances and printing fatigue,
the image rasteriser of MCP and the report, dynamics – mode equations, modal moments of the PSD, response spectrum) are compiled
at the FIRST use – the critical plane takes ~1 min. The numba cache is in the __pycache__ of the package, so one run after
installation is enough; the user then does not wait for the first fatigue or report.

Procedure: verification tasks V1 (statics, CalculiX), U3 (fatigue, critical plane), P5 (3D printing, printing fatigue) and the Word
report from V1 (images = rasteriser). --quick = only V1 (check of the installation without precompilation).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="only the check calculation V1")
    a = ap.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from verification.run_all import run
    t0 = time.time()
    keys = ["V1"] if a.quick else ["V1", "U3", "P5"]
    print(f"Check calculations {' '.join(keys)}"
          + ("" if a.quick else " (the first run compiles numba – may take several minutes)") + "…", flush=True)
    ok = run(keys)
    if ok and not a.quick:
        print("Precompiling dynamics kernels (modal response, PSD, response spectrum)…", flush=True)
        try:
            _dynamics_kernels()
        except Exception as e:  # noqa: BLE001 – precompilation is not a condition of functionality
            print(f"  the dynamics kernels could not be precompiled ({e}) – they will be compiled at first use")
        print("Precompiling images (Word report from V1)…", flush=True)
        try:
            _report_v1(root)
        except Exception as e:  # noqa: BLE001 – precompilation is not a condition of functionality
            print(f"  the report could not be created ({e}) – images will be compiled at first use")
    print(f"{'Done' if ok else 'ERROR'} in {time.time() - t0:.0f} s", flush=True)
    return 0 if ok else 1


def _dynamics_kernels() -> None:
    """Short calls of the numba functions of dynamics on small data (numba cache in __pycache__)."""
    from .dynamics import modal_ode, psd, shock, _env
    modal_ode.warmup()
    psd.warmup()
    shock.warmup()
    _env.warmup()


def _report_v1(root: Path) -> None:
    from .cli import main as cli_main
    study = root / "verification" / "out" / "V1" / "study.json"
    with tempfile.TemporaryDirectory(prefix="fea_warmup_") as tmp:
        rc = cli_main(["report", str(study), "-o", str(Path(tmp) / "V1.docx")])
        if rc:
            raise RuntimeError(f"return code {rc}")


if __name__ == "__main__":
    sys.exit(main())
