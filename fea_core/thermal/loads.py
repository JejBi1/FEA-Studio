"""Nodal heats of the boundary conditions [W] (consistent integration of tri6 / tet10) – for the heat balance
and the reactions of the prescribed temperatures. The same role as fea_core.loads for the mechanics: ccx computes its own, we
independently check that the heat in the nodes matches (the balance), and we distribute the flux over the boundary conditions.
"""
from __future__ import annotations

import numpy as np

from ..loads import _TRI_PTS, _TRI_W, _tri6_dN, _tri6_N, _TET_PTS, _TET_W, _tet10_dN, element_volumes
from ..mesh import Mesh
from .model import (FixedTemperature, HeatFlux, HeatFlow, Convection, Radiation, CavityRadiation,
                    ContactConductance, HeatGeneration, STEFAN_BOLTZMANN, ABSOLUTE_ZERO, is_cavity)

MM2_TO_M2 = 1e-6


def face_quadrature(mesh: Mesh, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss integration over the curved faces of a selection: (tri, N (P, n), dA (K, P) [mm²])."""
    tri = mesh.surfaces[name]
    X = mesh.nodes[tri]
    if tri.shape[1] == 6:
        N = np.array([_tri6_N(xi, eta) for xi, eta in _TRI_PTS])
        dA = np.empty((len(tri), len(_TRI_PTS)))
        for p, ((xi, eta), w) in enumerate(zip(_TRI_PTS, _TRI_W)):
            dxi, deta = _tri6_dN(xi, eta)
            a = np.einsum("i,kij->kj", dxi, X)
            b = np.einsum("i,kij->kj", deta, X)
            dA[:, p] = w * np.linalg.norm(np.cross(a, b), axis=1)
    else:
        N = np.full((1, 3), 1 / 3)
        dA = 0.5 * np.linalg.norm(np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), axis=1)[:, None]
    return tri, N, dA


def curved_area(mesh: Mesh, name: str) -> float:
    return float(face_quadrature(mesh, name)[2].sum())


def _face_nodal(mesh: Mesh, name: str, integrand, T: np.ndarray | None) -> np.ndarray:
    """∫ g(T) N_i dA over the faces of the selection → (N,) [units of g · mm²]."""
    tri, N, dA = face_quadrature(mesh, name)
    Tp = np.zeros(dA.shape) if T is None else T[tri] @ N.T     # (K, P)
    g = integrand(Tp) * dA                                      # (K, P)
    fe = g @ N                                                  # (K, n)
    return np.bincount(tri.ravel(), weights=fe.ravel(), minlength=len(mesh.nodes))


def volume_nodal(mesh: Mesh, density: np.ndarray) -> np.ndarray:
    """∫ q N_i dV for a volume density q per element (M,) → (N,) (exact integration of tet10)."""
    from ..results import tet10_shape
    out = np.zeros(len(mesh.nodes))
    sel = np.nonzero(density != 0)[0]
    if len(sel) == 0:
        return out
    tets = mesh.tets[sel]
    X = mesh.nodes[tets]
    q = density[sel]
    if tets.shape[1] == 10:
        fe = np.zeros((len(sel), 10))
        for p, w in zip(_TET_PTS, _TET_W):
            J = np.tensordot(_tet10_dN(*p), X, axes=(1, 1)).swapaxes(0, 1)
            fe += (w * np.linalg.det(J) * q)[:, None] * tet10_shape(np.asarray(p))[None, :]
    else:
        vol = np.abs(np.einsum("mi,mi->m", X[:, 1] - X[:, 0], np.cross(X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]))) / 6
        fe = np.repeat((vol * q / 4)[:, None], 4, axis=1)
    out += np.bincount(tets.ravel(), weights=fe.ravel(), minlength=len(mesh.nodes))
    return out


def generation_density(mesh: Mesh, bc: HeatGeneration) -> np.ndarray:
    """Volume source per element [W/mm³] (the power uniform in the volume of the part)."""
    vol = element_volumes(mesh)
    pi = mesh.part_names.index(bc.part)
    msk = mesh.tet_part == pi
    out = np.zeros(len(mesh.tets))
    out[msk] = bc.value / vol[msk].sum()
    return out


def bc_nodal(mesh: Mesh, bc, T: np.ndarray | None) -> np.ndarray | None:
    """Nodal heat [W] (+ into the body) of one boundary condition at the temperature field T; a prescribed temperature → None."""
    if isinstance(bc, FixedTemperature):
        return None
    if isinstance(bc, ContactConductance):
        return None
    if isinstance(bc, HeatFlux):
        return _face_nodal(mesh, bc.selection, lambda Tp: np.full(Tp.shape, bc.value * MM2_TO_M2), T)
    if isinstance(bc, HeatFlow):
        q = bc.value / curved_area(mesh, bc.selection)
        return _face_nodal(mesh, bc.selection, lambda Tp: np.full(Tp.shape, q), T)
    if isinstance(bc, Convection):
        return _face_nodal(mesh, bc.selection, lambda Tp: bc.h * MM2_TO_M2 * (bc.ambient - Tp), T)
    if isinstance(bc, (Radiation, CavityRadiation)):
        if is_cavity(bc):
            return None  # cavity radiation: the view factors are known only to ccx → the heat of the cavity see run.heat_flows (from RFL)
        z = ABSOLUTE_ZERO
        c = bc.emissivity * STEFAN_BOLTZMANN * MM2_TO_M2
        return _face_nodal(mesh, bc.selection, lambda Tp: c * ((bc.ambient - z) ** 4 - (Tp - z) ** 4), T)
    if isinstance(bc, HeatGeneration):
        return volume_nodal(mesh, generation_density(mesh, bc))
    raise TypeError(f"Unknown thermal boundary condition {type(bc).__name__}")


def bc_label(bc) -> str:
    return getattr(bc, "selection", None) or getattr(bc, "part", "")
