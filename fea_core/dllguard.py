"""Work folders of helper programs started with cwd = job folder (CalculiX, MYSTRAN): no planted libraries.

Windows resolves the DLLs of a program (e.g. mkl_rt.2.dll of ccx_i4) also from the CURRENT folder. A work folder that
came from outside (a "<study>_run" folder shipped next to a study, a work folder chosen by an MCP / REST client) could
therefore carry a library that the solver would load and run. Before such a program starts, the folder must not
contain any executable code (`check_workdir`). The CLI default work folder is a folder created by FEA Studio
(`own_workdir`).
"""
from __future__ import annotations

import os
from pathlib import Path

# files that a program started in the folder could load or run (DLL search order, side-by-side manifests, .local)
CODE_SUFFIXES = (".dll", ".exe", ".pyd", ".so", ".sys", ".ocx", ".cpl", ".drv", ".com", ".scr", ".manifest",
                 ".local", ".bat", ".cmd", ".ps1")
MARKER = ".fea_workdir"                       # written into work folders that FEA Studio created


class PlantedCodeError(RuntimeError):
    pass


def foreign_code(workdir: str | Path) -> list[str]:
    """Names of files / folders in workdir (not recursive – the DLL search looks only at the folder itself) that could be
    loaded as code by a program started there."""
    out = []
    try:
        with os.scandir(workdir) as it:
            for e in it:
                n = e.name.lower()
                if n.endswith(CODE_SUFFIXES):
                    out.append(e.name)
    except FileNotFoundError:
        return []
    return sorted(out)


def check_workdir(workdir: str | Path, program: str = "the solver") -> None:
    """PlantedCodeError when the work folder holds executable code (DLL planting)."""
    bad = foreign_code(workdir)
    if bad:
        shown = ", ".join(bad[:5]) + (" …" if len(bad) > 5 else "")
        raise PlantedCodeError(
            f"The work folder {workdir} contains program files ({shown}) that {program} could load – for security "
            "reasons the calculation is not started; use an empty work folder")


def mark(workdir: str | Path) -> None:
    try:
        Path(workdir, MARKER).write_text("FEA Studio work folder\n", encoding="utf-8")
    except OSError:
        pass


def own_workdir(default: str | Path) -> Path:
    """Default work folder of a command line run: `default` when it does not exist, is empty or was created by FEA Studio
    (marker); otherwise a new sibling `<name>_2`, `<name>_3`, … – a folder that came with the study is not reused."""
    base = Path(default)
    cand = base
    k = 1
    while True:
        if not cand.exists():
            cand.mkdir(parents=True)
            mark(cand)
            return cand
        if cand.is_dir():
            if (cand / MARKER).is_file() and not foreign_code(cand):
                return cand
            try:
                empty = not any(os.scandir(cand))
            except OSError:
                empty = False
            if empty:
                mark(cand)
                return cand
        k += 1
        cand = base.with_name(f"{base.name}_{k}")
