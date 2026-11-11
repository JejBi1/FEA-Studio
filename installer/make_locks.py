"""Developer tool: recreates the version locks installer/locks/<platform>.txt (the whole set of packages of all modules).

    python installer/make_locks.py [--uv path/to/uv]

A lock = the exact versions of all packages including dependencies for Python 3.11 (uv pip compile, works also for another platform
than the one it runs on). The installer uses it as constraints when installing any subset of modules, so every
user gets the same verified versions. mkl must stay 2023.2.0 (see CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = {"windows-x86_64": "x86_64-pc-windows-msvc", "linux-x86_64": "x86_64-manylinux_2_35"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uv", default=str(ROOT / "tools" / "uv" / ("uv.exe" if sys.platform == "win32" else "uv")))
    ap.add_argument("--upgrade", action="store_true", help="re-resolve the versions (otherwise the current lock is kept)")
    args = ap.parse_args()
    mods = json.loads((ROOT / "fea_core" / "modules.json").read_text(encoding="utf-8"))["modules"]
    reqs = sorted({r for m in mods.values() for r in m.get("pip", [])})
    nodeps = sorted({r for m in mods.values() for r in m.get("pip_nodeps", [])})
    (ROOT / "installer" / "locks").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        rin = Path(td) / "req.in"
        rin.write_text("\n".join(reqs) + "\n", encoding="utf-8")
        for name, plat in PLATFORMS.items():
            out = ROOT / "installer" / "locks" / f"{name}.txt"
            cmd = [args.uv, "pip", "compile", str(rin), "--python-version", "3.11", "--python-platform", plat,
                   "--no-header", "--no-annotate", "-o", str(out), "-q"]
            if not args.upgrade and out.exists():             # the existing verified versions are kept
                cmd += ["-c", str(out)]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
            lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
            lines += nodeps
            head = ("# Version lock for " + name + " (Python 3.11) – created by installer/make_locks.py from fea_core/modules.json.\n"
                    "# The lines at the end (sectionproperties, pyNastran, cadquery-ocp, vtk…) are installed without dependencies.\n")
            out.write_text(head + "\n".join(sorted(set(lines), key=str.lower)) + "\n", encoding="utf-8")
            print(f"{name}: {len(lines)} packages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
