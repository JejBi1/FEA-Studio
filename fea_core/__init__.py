"""FEA Studio – calculation core without GUI (linear statics and contacts, dynamics, mechanisms, heat; CalculiX).

Units: mm, N, MPa, t, s.

Imports are lazy: the meshing process (fea_core.mesh.worker) must not load OCP,
so that two OpenCascade libraries (OCP and gmsh) never meet in one process.
"""
__version__ = "1.4.0"

_LAZY = {
    "Study": "study", "Material": "study", "Selection": "study", "Fixed": "study",
    "Displacement": "study", "Force": "study", "Pressure": "study", "Gravity": "study",
    "RemoteLoad": "study", "CylindricalSupport": "study", "BearingLoad": "study", "LoadCase": "study", "MeshSettings": "study", "SolverSettings": "study",
    "ThermalLoad": "study", "Rotation": "study", "CoordinateSystem": "study", "RemoteSupport": "study",
    "ThermalSettings": "thermal.model", "ThermalCase": "thermal.model", "FixedTemperature": "thermal.model",
    "HeatFlux": "thermal.model", "HeatFlow": "thermal.model", "Convection": "thermal.model",
    "Radiation": "thermal.model", "HeatGeneration": "thermal.model",
    "FaceSelector": "selectors", "find_faces": "selectors", "EdgeSelector": "selectors", "find_edges": "selectors",
    "PointSelector": "selectors",
    "read_step": "geometry", "write_step": "geometry", "Part": "geometry",
    "run_study": "solve",
    "TopologyOptSpec": "topo", "TopologyOptimizer": "topo",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(name)
