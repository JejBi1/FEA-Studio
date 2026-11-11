"""Node temperatures for a mechanical load state with ThermalLoad (thermal stresses)."""
from __future__ import annotations

import numpy as np

from ..mesh import Mesh
from ..study import Study, LoadCase, ThermalLoad


def node_temperatures(study: Study, mesh: Mesh, lc: LoadCase, thermal: dict | None) -> np.ndarray | None:
    """Temperature of the mesh nodes [°C] of the state lc, or None (a state without a thermal load).

    Field from a thermal state (ThermalLoad.case, frame at the time .time) + uniform temperatures of the parts (they overwrite the field;
    a node at the interface of parts with different temperatures gets the average). thermal = {name of the state: ThermalResult}.
    """
    loads = [l for l in lc.loads if isinstance(l, ThermalLoad)]
    if not loads:
        return None
    T = np.full(len(mesh.nodes), float(study.thermal.reference_temperature))
    for l in loads:
        if l.case:
            if not thermal or l.case not in thermal:
                raise KeyError(f"Case '{lc.name}': result of thermal case '{l.case}' is missing")
            T = np.array(thermal[l.case].at_time(l.time), dtype=float)
    acc = np.zeros(len(mesh.nodes))
    cnt = np.zeros(len(mesh.nodes))
    for l in loads:
        if l.value is None or l.case:
            continue
        parts = l.parts or list(mesh.part_names)
        for p in parts:
            if p not in mesh.part_names:
                continue
            nodes = np.unique(mesh.tets[mesh.tet_part == mesh.part_names.index(p)])
            acc[nodes] += float(l.value)
            cnt[nodes] += 1
    msk = cnt > 0
    T[msk] = acc[msk] / cnt[msk]
    return T


def force_scale(study: Study, mesh: Mesh, T: np.ndarray) -> float:
    """Scale of the forces of the thermal load [N] for the equilibrium check (the external forces are zero):
    Σ E·α·|ΔT|·V / L (L = diagonal of the model) ≈ the force of fully prevented expansion.
    """
    from ..loads import element_volumes
    E = np.array([study.material_of(p).E for p in mesh.part_names])[mesh.tet_part]
    a = np.array([study.material_of(p).alpha or 0.0 for p in mesh.part_names])[mesh.tet_part]
    dT = np.abs(T[mesh.tets[:, :4]].mean(axis=1) - study.thermal.reference_temperature)
    L = float(np.linalg.norm(mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0))) or 1.0
    return float((E * a * dT * element_volumes(mesh)).sum() / L)
