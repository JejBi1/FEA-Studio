"""Shell model of thin-walled parts (ccx S6) as a simplification of the 3D study.

API: run_shell(study, workdir, parts=None, study_dir=None, on_log=None, cancel=None) → ShellResults;
ShellSettings (Study.shell); detect_sheet / read_step_surfaces (OCP). The imports are lazy – the meshing process
(fea_core.shell.mesh_worker) must not load OCP.
"""
from .model import ShellSettings, ShellError, OFFSETS   # noqa: F401 (without OCP)

_LAZY = {"run_shell": "run", "prepare_sheets": "run", "check_study": "run", "ShellResults": "results", "ShellStep": "results",
         "ShellMesh": "mesh", "generate_shell_mesh": "mesh", "detect_sheet": "geometry",
         "read_step_surfaces": "geometry", "surface_sheet": "geometry", "SheetInfo": "geometry",
         "connect_sheets": "geometry"}

__all__ = ["ShellSettings", "ShellError", "OFFSETS", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(name)
