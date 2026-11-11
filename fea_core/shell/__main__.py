"""CLI: python -m fea_core.shell study.json [--out folder] [--png field] [--thickness part=t …]

Shell calculation of the study (settings from the key "shell" in study.json, otherwise all parts automatically).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fea_core.shell", description="Shell model of the study (ccx S6)")
    ap.add_argument("study")
    ap.add_argument("--out", default=None, help="working folder (default <study>_shell)")
    ap.add_argument("--png", default="mises", help="image field (mises, u, membrane, bending …; '' = no image)")
    ap.add_argument("--thickness", nargs="*", default=[], help="part=thickness (sheet parts from STEP)")
    ap.add_argument("--size", type=float, default=None, help="element size [mm]")
    a = ap.parse_args(argv)
    from ..study import Study
    from .model import ShellSettings, ShellError
    from .run import run_shell
    path = Path(a.study)
    raw = json.loads(path.read_text(encoding="utf-8"))
    st = Study.from_dict(raw)
    s = ShellSettings.from_dict(raw.get("shell") or {"enabled": True})
    for item in a.thickness:
        name, _, val = item.rpartition("=")
        s.parts[name] = float(val)
    if a.size:
        s.size = a.size
    out = Path(a.out) if a.out else path.with_name(path.stem + "_shell")
    try:
        res = run_shell(st, out, study_dir=path.parent, on_log=print, settings=s)
    except ShellError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(json.dumps(res.summary(), indent=1, ensure_ascii=False))
    if a.png:
        for k in range(len(res.steps)):
            print("image:", res.render_png(out / f"shell_{a.png}_{k + 1}.png", a.png, k, "max", deform=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
