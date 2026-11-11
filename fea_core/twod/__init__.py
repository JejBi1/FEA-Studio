"""2D calculations as a simplification of the 3D study by a section: axisymmetric problem (CAX6), plane stress (CPS6) and plane
strain (CPE6) in CalculiX.

    from fea_core.twod import TwoDSettings, run_2d
    res = run_2d(study, "work", settings=TwoDSettings(kind="axisym", origin=[0, 0, 0], axis=[0, 0, 1], xdir=[1, 1, 0]))
    res.probe([20, 5], "st")            # σθ at the point r = 20, z = 5
    res.render_png("vm.png")

Modules: model (settings, the frame of the section), section (section by OCP), worker (Gmsh – a separate process without OCP), mesh2d (mesh,
integration), convert (constraints and loads), inp (ccx), run (run_2d), results (Results2D). The imports are lazy (the worker
must not load OCP).
"""
_LAZY = {"TwoDSettings": "model", "TwoDError": "model", "Frame2D": "model", "make_frame": "model",
         "KIND_LABEL": "model", "run_2d": "run", "Results2D": "results"}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(name)
