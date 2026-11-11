"""Core of the topology optimisation (BESO / SIMP) in FEA Studio.

Main class TopologyOptimizer:
- Management of the element densities rho_e (0 or 1 in BESO, continuous in SIMP).
- Identification of the non-design elements (elements containing nodes of the non_design_selections have rho = 1.0 fixed).
- Gradual reduction of the target volume in each iteration: V_k+1 = max(V_target, V_k * (1 - ER)).
- Threshold value by bisection for the assignment of the elements (solid vs void).
- Real feedback to the FE: in each iteration the stiffness of the elements is set to E_e = E_0 * max(rho_e^p, 1e-6)
  (`fea_core.ccx.inp.ElementScale`, the void elements have their own *MATERIAL) and ccx is run again on the same mesh.
- Sensitivities from the strain energy of the elements computed from the displacements (strain at the 4 Gauss points of tet10, stiffness
  of the base material) – NOT from the nodally averaged stresses (those would mix void and solid neighbours).
  Several load states: the sum of the compliances.
- Record of the history of the iterations: volume fraction, total compliance, removed and added elements.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Any

import numpy as np

from ..loads import GAUSS4, W_GAUSS4, _tet10_dN, element_volumes, inv3
from .filter import SensitivityFilter
from .spec import TopologyOptSpec

# Smallest ratio E_void / E_0 sent to ccx (rho^p with min_density 1e-3 and p = 3 would give 1e-9 – bad conditioning)
MIN_STIFFNESS_RATIO = 1e-6

if TYPE_CHECKING:
    from ..mesh import Mesh
    from ..results import StepResult, Results
    from ..study import Material, Study


@dataclass
class OptimizationIteration:
    """Record of one iteration of the topology optimisation."""
    iteration: int
    target_volume_fraction: float
    volume_fraction: float
    total_compliance: float
    elements_removed: int
    elements_added: int
    solid_elements: int
    void_elements: int
    threshold: float
    change_rate: float = 0.0
    converged: bool = False

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_tet_volumes(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Volumes of the tetrahedra tets (M, 4|10) from the nodes nodes (N, 3) – `loads.element_volumes` over a temporary mesh."""
    if len(tets) == 0:
        return np.zeros(0, dtype=float)
    from ..mesh import Mesh
    return element_volumes(Mesh(np.asarray(nodes, dtype=float), np.asarray(tets), np.zeros(len(tets), dtype=int), ["_"]))


def compute_tet_centroids(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Centroids of the tetrahedra tets (M, 4|10) = mean of the 4 corner nodes."""
    if len(tets) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(nodes)[np.asarray(tets)[:, :4]].mean(axis=1)


def compute_mesh_element_volumes_and_centroids(mesh: "Mesh") -> tuple[np.ndarray, np.ndarray]:
    """Returns (volumes, centroids) for all elements of the mesh Mesh (volumes from `loads.element_volumes`, centroids from the corners)."""
    vols = element_volumes(mesh)
    cents = np.asarray(mesh.nodes)[mesh.tets[:, :4]].mean(axis=1) if len(mesh.tets) else np.zeros((0, 3))
    return vols, cents


def identify_non_design_elements(mesh: "Mesh", non_design_selections: list[str]) -> np.ndarray:
    """Identifies the elements that contain nodes of the given non-design selections.

    Elements containing at least one node of the given selections have rho = 1.0 fixed.
    Returns: boolean array (M,) where True = non-design element.
    """
    num_elements = len(mesh.tets)
    mask = np.zeros(num_elements, dtype=bool)
    if not non_design_selections or num_elements == 0:
        return mask

    chunks: list[np.ndarray] = []
    for name in non_design_selections:
        for attr in ("surfaces", "edges", "points"):
            group = getattr(mesh, attr, None)
            if group and name in group:
                chunks.append(np.asarray(group[name]).ravel())
    if not chunks:
        return mask

    node_arr = np.unique(np.concatenate(chunks))
    return np.isin(mesh.tets, node_arr).any(axis=1)


def compute_strain_energy_density(S: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Computes the strain energy density u0 [mJ/mm³ = MPa] from the stress tensor in the nodes.

    S: (N, 6) stress components: Sxx, Syy, Szz, Sxy, Syz, Szx
    u0 = 1/(2E) * [Sxx^2 + Syy^2 + Szz^2 - 2*nu*(Sxx*Syy + Syy*Szz + Szz*Sxx) + 2*(1+nu)*(Sxy^2 + Syz^2 + Szx^2)]
    """
    sx, sy, sz = S[:, 0], S[:, 1], S[:, 2]
    txy, tyz, tzx = S[:, 3], S[:, 4], S[:, 5]

    u0 = (
        sx ** 2 + sy ** 2 + sz ** 2
        - 2.0 * nu * (sx * sy + sy * sz + sz * sx)
        + 2.0 * (1.0 + nu) * (txy ** 2 + tyz ** 2 + tzx ** 2)
    ) / (2.0 * max(E, 1e-12))
    return np.maximum(0.0, u0)


def compute_element_strain_energies(
    mesh: "Mesh",
    step_result: "StepResult",
    E: float = 210000.0,
    nu: float = 0.3,
    volumes: np.ndarray | None = None,
) -> np.ndarray:
    """Approximate strain energy U_e [N*mm = mJ] of the elements from the nodal stresses: U_e = u0_elem * V_e, u0_elem =
    the mean of the energy density of the nodes of the element.

    CAUTION: the nodal stresses are averages over the neighbouring elements, at the void / solid interface they mix both sides. The optimisation
    does not use this (see `element_strain_energies` from the displacements); the function stays as a quick estimate without displacements.
    """
    if len(mesh.tets) == 0:
        return np.zeros(0, dtype=float)

    if volumes is None:
        volumes = compute_tet_volumes(mesh.nodes, mesh.tets)

    # Nodal strain energy density
    u0_nodal = compute_strain_energy_density(step_result.S, E=E, nu=nu)

    # Average u0 per element
    elem_u0 = np.mean(u0_nodal[mesh.tets], axis=1)
    return elem_u0 * volumes


def element_stiffness_scale(rho: np.ndarray, penalty: float, min_ratio: float = MIN_STIFFNESS_RATIO) -> np.ndarray:
    """Multiple of the stiffness of the elements E_e / E_0 = max(rho_e^p, min_ratio) – the input `ElementScale` for the writing of the .inp."""
    return np.maximum(np.power(np.maximum(np.asarray(rho, dtype=float), 0.0), penalty), min_ratio)


def element_strain_energies(study, mesh: "Mesh", displacements: list[np.ndarray],
                            C: np.ndarray | None = None) -> np.ndarray:
    """Strain energy of the elements U0_e = Σ_states ½ ∫ εᵀ C0 ε dV [N·mm] with the stiffness of the BASE material (without penalisation).

    Strains from the displacements at the 4 Gauss points of tet10 (points gauss3d4 as in ccx, `loads.GAUSS4`), weights |J|/24 – the energy
    of one element is not mixed with its neighbours (unlike the nodally averaged stresses). displacements = displacements (N, 3)
    per load states, C = (M, 6, 6) stiffnesses of the elements (default `recovery.element_stiffness`).
    """
    from ..recovery import element_stiffness, _chunks
    if mesh.tets.shape[1] != 10:
        raise ValueError("Element strain energies are only available for tet10 (C3D10)")
    if C is None:
        C = element_stiffness(study, mesh)
    out = np.zeros(len(mesh.tets))
    dN = [_tet10_dN(*p) for p in GAUSS4]
    for lo, hi in _chunks(len(mesh.tets)):
        conn = mesh.tets[lo:hi]
        X = mesh.nodes[conn]
        Ce = C[lo:hi]
        for g in range(len(GAUSS4)):
            J = np.tensordot(dN[g], X, axes=(1, 1)).swapaxes(0, 1)           # (E, 3, 3)
            dNx = inv3(J) @ dN[g]                                            # (E, 3, 10)
            w = W_GAUSS4 * np.abs(np.linalg.det(J))
            for U in displacements:
                G = dNx @ U[conn]                                            # ∂u_k/∂x_a
                eps = np.stack([G[:, 0, 0], G[:, 1, 1], G[:, 2, 2], G[:, 0, 1] + G[:, 1, 0],
                                G[:, 0, 2] + G[:, 2, 0], G[:, 1, 2] + G[:, 2, 1]], axis=1)
                out[lo:hi] += 0.5 * w * np.einsum("ei,ei->e", eps, np.einsum("eij,ej->ei", Ce, eps))
    return out


def find_threshold_bisection(
    sensitivities: np.ndarray,
    volumes: np.ndarray,
    target_volume: float,
    non_design_mask: np.ndarray | None = None,
    min_density: float = 1e-3,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> tuple[float, np.ndarray]:
    """Determines the threshold value by bisection for the assignment of the elements (solid vs void).

    For BESO:
        rho_e = 1.0 for alpha_e >= th
        rho_e = min_density for alpha_e < th
    Elements with non_design_mask == True always have rho_e = 1.0.

    Parameters:
        sensitivities: (M,) filtered sensitivities of the elements alpha
        volumes: (M,) volumes of the elements
        target_volume: target total volume of the solid elements
        non_design_mask: (M,) mask of the non-design elements (True = permanently solid)
        min_density: density of the void elements (e.g. 1e-3)
        max_iter: maximum number of bisection iterations
        tol: relative tolerance of the agreement of the volume
    Returns:
        (threshold, rho): the found threshold and the resulting array of densities (M,)
    """
    n = len(sensitivities)
    if non_design_mask is None:
        non_design_mask = np.zeros(n, dtype=bool)

    full_rho = np.ones(n, dtype=float)
    if n == 0:
        return 0.0, full_rho

    nd_vol = float(np.sum(volumes[non_design_mask]))
    d_mask = ~non_design_mask

    # If all elements are non-design
    if not np.any(d_mask):
        return 0.0, full_rho

    d_vols = volumes[d_mask]
    d_sens = sensitivities[d_mask]
    d_total_vol = float(np.sum(d_vols))
    d_target_vol = max(0.0, target_volume - nd_vol)

    # Limit cases
    if d_target_vol >= d_total_vol:
        full_rho[d_mask] = 1.0
        return float(np.min(d_sens)) - 1.0, full_rho

    min_vol = d_total_vol * min_density
    if d_target_vol <= min_vol:
        full_rho[d_mask] = min_density
        return float(np.max(d_sens)) + 1.0, full_rho

    # Bisection on the sensitivities of the design elements
    th_min = float(np.min(d_sens))
    th_max = float(np.max(d_sens))
    best_th = th_min
    best_diff = float("inf")
    best_rho_d = np.ones(len(d_sens), dtype=float)

    for _ in range(max_iter):
        th = 0.5 * (th_min + th_max)
        rho_d = np.where(d_sens >= th, 1.0, min_density)
        cur_v = float(np.sum(d_vols * rho_d))
        diff = abs(cur_v - d_target_vol)

        if diff < best_diff:
            best_diff = diff
            best_th = th
            best_rho_d = rho_d

        if diff <= tol * max(d_target_vol, 1e-12) or (th_max - th_min) <= 1e-12 * (abs(th_max) + abs(th_min) + 1e-12):
            break

        if cur_v > d_target_vol:
            th_min = th
        else:
            th_max = th

    full_rho[d_mask] = best_rho_d
    return best_th, full_rho


def find_threshold_simp(
    sensitivities: np.ndarray,
    volumes: np.ndarray,
    target_volume: float,
    current_rho: np.ndarray,
    non_design_mask: np.ndarray | None = None,
    min_density: float = 1e-3,
    eta: float = 0.5,
    move: float = 0.2,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> tuple[float, np.ndarray]:
    """Bisection for the determination of the Lagrange multiplier in SIMP (Optimality Criteria)."""
    n = len(sensitivities)
    if non_design_mask is None:
        non_design_mask = np.zeros(n, dtype=bool)

    full_rho = current_rho.copy()
    d_mask = ~non_design_mask
    if not np.any(d_mask):
        full_rho[non_design_mask] = 1.0
        return 0.0, full_rho

    d_vols = volumes[d_mask]
    d_sens = np.maximum(sensitivities[d_mask], 1e-12)
    d_rho = current_rho[d_mask]
    nd_vol = float(np.sum(volumes[non_design_mask]))
    d_target_vol = max(0.0, target_volume - nd_vol)

    l1 = 1e-12
    l2 = float(np.max(d_sens) * 1e4)
    best_l = l1
    best_diff = float("inf")
    best_rho_d = d_rho.copy()

    for _ in range(max_iter):
        lmid = 0.5 * (l1 + l2)
        step_factor = (d_sens / lmid) ** eta
        rho_candidate = d_rho * step_factor
        rho_new = np.clip(
            rho_candidate,
            np.maximum(min_density, d_rho - move),
            np.minimum(1.0, d_rho + move),
        )
        cur_v = float(np.sum(d_vols * rho_new))
        diff = abs(cur_v - d_target_vol)

        if diff < best_diff:
            best_diff = diff
            best_l = lmid
            best_rho_d = rho_new

        if diff <= tol * max(d_target_vol, 1e-12) or (l2 - l1) <= 1e-12 * (l2 + l1):
            break

        if cur_v > d_target_vol:
            l1 = lmid
        else:
            l2 = lmid

    full_rho[d_mask] = best_rho_d
    full_rho[non_design_mask] = 1.0
    return best_l, full_rho


class TopologyOptimizer:
    """Main class for the topology optimisation (BESO / SIMP)."""

    def __init__(
        self,
        *args: Any,
        spec: TopologyOptSpec | None = None,
        material: "Material | None" = None,
        base_E: float = 210000.0,
        nu: float = 0.3,
        **kwargs: Any,
    ) -> None:
        """Initialisation of TopologyOptimizer.

        Can be called in two ways:
            TopologyOptimizer(mesh, spec=spec, ...)
            TopologyOptimizer(study, mesh, spec)
        """
        self.study: "Study | None" = None

        if len(args) >= 2 and (hasattr(args[0], "load_cases") or hasattr(args[0], "materials")):
            # (study, mesh, spec)
            self.study = args[0]
            self.mesh: "Mesh" = args[1]
            self.spec = args[2] if len(args) > 2 and args[2] is not None else (spec or TopologyOptSpec())
        elif len(args) >= 1:
            self.mesh = args[0]
            self.spec = args[1] if len(args) > 1 and args[1] is not None else (spec or TopologyOptSpec())
        else:
            raise ValueError("TopologyOptimizer requires at least the mesh parameter")

        # Material properties
        if self.study is not None and hasattr(self.study, "materials") and self.study.materials:
            if isinstance(self.study.materials, dict):
                first_mat = next(iter(self.study.materials.values()))
            else:
                first_mat = self.study.materials[0]
            self.base_E = float(first_mat.E)
            self.nu = float(first_mat.nu)
        elif material is not None:
            self.base_E = float(material.E)
            self.nu = float(material.nu)
        else:
            self.base_E = float(base_E)
            self.nu = float(nu)

        self.num_elements = len(self.mesh.tets)

        # Geometry of the elements (volumes and centroids)
        self.volumes, self.centroids = compute_mesh_element_volumes_and_centroids(self.mesh)
        self.total_mesh_volume = float(np.sum(self.volumes))

        # Management of the element densities (initial state: full material rho = 1.0)
        self.rho = np.ones(self.num_elements, dtype=float)

        # Identification of the non-design elements
        self.non_design_mask = identify_non_design_elements(
            self.mesh, self.spec.non_design_selections
        )
        self.rho[self.non_design_mask] = 1.0

        # Spatial filter of the sensitivities
        self.filter = SensitivityFilter(
            self.centroids,
            filter_radius=self.spec.filter_radius,
            volumes=self.volumes,
        )

        self.initial_compliance: float | None = None    # compliance of the full design (1st iteration) [N·mm]
        self.final_compliance: float | None = None      # compliance of the resulting design (after `run`) [N·mm]
        self._C0: np.ndarray | None = None       # stiffnesses of the base material per element (cache)

        # Counter and history
        self.iteration = 0
        self.history: list[OptimizationIteration] = []

    @property
    def densities(self) -> np.ndarray:
        """Alias for self.rho."""
        return self.rho

    @densities.setter
    def densities(self, val: np.ndarray) -> None:
        self.rho = np.ascontiguousarray(val, dtype=float)

    @property
    def current_volume_fraction(self) -> float:
        """Current ratio of the volume of the solid elements to the total volume of the mesh."""
        if self.total_mesh_volume <= 0.0:
            return 1.0
        return float(np.sum(self.volumes * self.rho)) / self.total_mesh_volume

    @property
    def solid_elements_count(self) -> int:
        """Number of elements considered solid (rho > 0.5)."""
        return int(np.count_nonzero(self.rho > 0.5))

    @property
    def void_elements_count(self) -> int:
        """Number of elements considered void (rho <= 0.5)."""
        return self.num_elements - self.solid_elements_count

    def compute_next_target_volume_fraction(self, current_vf: float | None = None) -> float:
        """Gradual reduction of the target volume in each iteration:

        V_k+1 = max(V_target, V_k * (1 - ER)).
        """
        if current_vf is None:
            current_vf = self.current_volume_fraction
        target_vf = self.spec.target_volume_fraction
        er = self.spec.evolution_rate
        next_vf = max(target_vf, current_vf * (1.0 - er))
        return float(next_vf)

    def stiffness_scale(self) -> np.ndarray:
        """E_e / E_0 = max(rho_e^p, 1e-6) – multiples of the stiffness of the elements for the FE (`ElementScale`)."""
        return element_stiffness_scale(self.rho, self.spec.penalty)

    def get_element_young_moduli(self, base_E: float | None = None) -> np.ndarray:
        """Application of the change of stiffness: E_e = E_0 * rho_e^p (void elements: rho_e >= min_density, stiffness >= 1e-6 * E_0)."""
        E0 = self.base_E if base_E is None else float(base_E)
        p = self.spec.penalty
        eff_rho = np.maximum(self.rho, self.spec.min_density)
        return E0 * np.maximum(eff_rho ** p, MIN_STIFFNESS_RATIO)

    def compute_sensitivities_from_strain_energies(self, strain_energies: np.ndarray) -> np.ndarray:
        """Sensitivity numbers alpha_e from the energies U0_e of the elements computed with the stiffness of the BASE material (see
        `element_strain_energies`) at the current densities (those with which the FE ran).

        Compliance c = Σ x_e^p · 2 U0_e, ∂c/∂x_e = −2 p x_e^(p−1) U0_e. Per unit volume (non-uniform mesh):
            BESO: alpha_e = x_eff^(p−1) · U0_e / V_e   (solid x = 1; a void has the sensitivity from ITS OWN strain)
            SIMP: alpha_e = p · x_eff^(p−1) · U0_e / V_e
        x_eff = (E_e/E_0)^(1/p) is the stiffness actually used in the FE (because of the lower bound 1e-6).
        """
        vols = np.maximum(self.volumes, 1e-12)
        u0_elem = np.asarray(strain_energies, dtype=float) / vols
        p = self.spec.penalty
        x_eff = self.stiffness_scale() ** (1.0 / p)
        sens = x_eff ** (p - 1.0) * u0_elem
        if self.spec.method.lower() != "beso":
            sens = p * sens
        return sens

    def current_compliance(self, strain_energies: np.ndarray) -> float:
        """Compliance f·u = Σ_e 2 · (E_e/E_0) · U0_e at the current densities [N·mm]."""
        return float(2.0 * np.sum(self.stiffness_scale() * np.asarray(strain_energies, dtype=float)))

    def step(
        self,
        sensitivities: np.ndarray | None = None,
        strain_energies: np.ndarray | None = None,
    ) -> OptimizationIteration:
        """Performs one step (iteration) of the topology optimisation.

        Parameters:
            sensitivities: direct vector of the sensitivities of the elements (M,) (per unit volume)
            strain_energies: strain energies of the elements U0_e (M,) from the FE with the densities `self.rho`
                (`element_strain_energies`, stiffness of the base material)
        Returns:
            OptimizationIteration with the summary of the iteration
        """
        # 1. Sensitivities and compliance (before changing rho, they belong to the densities with which the FE ran)
        if strain_energies is not None:
            u_elem = np.ascontiguousarray(strain_energies, dtype=float)
            if len(u_elem) != self.num_elements:
                raise ValueError(f"Element energy count {len(u_elem)}, mesh elements {self.num_elements}")
            total_compliance = self.current_compliance(u_elem)
            raw_sens = self.compute_sensitivities_from_strain_energies(u_elem)
        elif sensitivities is not None:
            raw_sens = np.ascontiguousarray(sensitivities, dtype=float)
            u_elem = raw_sens * self.volumes
            total_compliance = float(np.sum(u_elem))
        else:
            raise ValueError("strain_energies or sensitivities must be passed to the step() method")

        self.iteration += 1
        prev_rho = self.rho.copy()
        current_vf = self.current_volume_fraction

        # 2. Filtering of the sensitivities (spatial filter + averaging over the history)
        filtered_sens = self.filter.filter(raw_sens, use_history=True)

        # 3. Target volume for this iteration: V_k+1 = max(V_target, V_k * (1 - ER))
        next_target_vf = self.compute_next_target_volume_fraction(current_vf)
        target_vol = next_target_vf * self.total_mesh_volume

        # 4. Threshold value by bisection for the assignment of the elements (solid vs void)
        if self.spec.method.lower() == "beso":
            th, new_rho = find_threshold_bisection(
                filtered_sens,
                volumes=self.volumes,
                target_volume=target_vol,
                non_design_mask=self.non_design_mask,
                min_density=self.spec.min_density,
            )
        else:  # SIMP
            th, new_rho = find_threshold_simp(
                filtered_sens,
                volumes=self.volumes,
                target_volume=target_vol,
                current_rho=self.rho,
                non_design_mask=self.non_design_mask,
                min_density=self.spec.min_density,
            )

        # Non-design elements are fixed solid
        new_rho[self.non_design_mask] = 1.0
        self.rho = new_rho

        # 5. Statistics of the changes
        was_solid = prev_rho > 0.5
        is_solid = self.rho > 0.5
        removed = int(np.count_nonzero(was_solid & ~is_solid))
        added = int(np.count_nonzero(~was_solid & is_solid))
        change_rate = (removed + added) / max(self.num_elements, 1)

        new_vf = self.current_volume_fraction

        # 6. Convergence check
        converged = self.check_convergence(change_rate, new_vf)

        rec = OptimizationIteration(
            iteration=self.iteration,
            target_volume_fraction=next_target_vf,
            volume_fraction=new_vf,
            total_compliance=total_compliance,
            elements_removed=removed,
            elements_added=added,
            solid_elements=int(np.count_nonzero(is_solid)),
            void_elements=int(np.count_nonzero(~is_solid)),
            threshold=th,
            change_rate=change_rate,
            converged=converged,
        )
        self.history.append(rec)
        return rec

    def check_convergence(self, change_rate: float, current_vf: float) -> bool:
        """Checks whether the optimisation has converged."""
        target_met = abs(current_vf - self.spec.target_volume_fraction) <= 0.01
        if not target_met:
            return False

        if change_rate <= self.spec.tol:
            return True

        if len(self.history) >= 5:
            comp_tail = [h.total_compliance for h in self.history[-5:]]
            c_diff = (max(comp_tail) - min(comp_tail)) / max(abs(comp_tail[-1]), 1e-12)
            if c_diff <= self.spec.tol:
                return True

        return False

    def is_converged(self) -> bool:
        """Returns True if the last iteration converged or max_iterations was reached."""
        if not self.history:
            return False
        if self.history[-1].converged:
            return True
        return self.iteration >= self.spec.max_iterations

    def get_solid_element_indices(self) -> np.ndarray:
        """Returns the indices of the solid elements (rho > 0.5)."""
        return np.nonzero(self.rho > 0.5)[0]

    def get_void_element_indices(self) -> np.ndarray:
        """Returns the indices of the void elements (rho <= 0.5)."""
        return np.nonzero(self.rho <= 0.5)[0]

    def create_solid_mesh(self) -> "Mesh":
        """Creates a new sub-mesh Mesh containing only the solid elements."""
        from ..mesh import Mesh

        solid_idx = self.get_solid_element_indices()
        if len(solid_idx) == 0:
            return Mesh(
                nodes=np.zeros((0, 3), dtype=float),
                tets=np.zeros((0, self.mesh.tets.shape[1]), dtype=self.mesh.tets.dtype),
                tet_part=np.zeros(0, dtype=self.mesh.tet_part.dtype),
                part_names=list(self.mesh.part_names),
            )

        solid_tets = self.mesh.tets[solid_idx]
        used_nodes, inverse = np.unique(solid_tets, return_inverse=True)
        new_nodes = self.mesh.nodes[used_nodes]
        new_tets = inverse.reshape(solid_tets.shape)
        new_tet_part = self.mesh.tet_part[solid_idx]

        return Mesh(
            nodes=new_nodes,
            tets=new_tets,
            tet_part=new_tet_part,
            part_names=list(self.mesh.part_names),
        )

    def check_study(self) -> list[str]:
        """Errors that prevent the optimisation from running (empty list = it can run)."""
        st = self.study
        if st is None:
            return ["The optimization needs a study: TopologyOptimizer(study, mesh, spec)"]
        errors: list[str] = []
        if not st.load_cases:
            errors.append("The study has no load case – without loads there are no sensitivities (add at least one)")
        if st.is_nonlinear():
            errors.append("Topology optimization is only available for linear statics (contacts / NLGEOM are not supported)")
        if self.mesh.tets.shape[1] != 10:
            errors.append("Topology optimization requires tet10 (C3D10) elements")
        for pname in self.mesh.part_names:
            try:
                mat = st.material_of(pname)
            except KeyError:
                errors.append(f"Part '{pname}' has no material")
                continue
            if getattr(mat, "is_print", False) or getattr(mat, "is_ortho", False) or getattr(mat, "is_plastic", False):
                errors.append(f"Part '{pname}': material '{mat.name}' is printed / orthotropic / plastic – the optimization is only for isotropic elastic material")
        if not self.spec.all_load_cases and not (0 <= int(self.spec.target_step) < len(st.load_cases)):
            errors.append(f"target_step {self.spec.target_step} does not exist (steps: {len(st.load_cases)})")
        return errors

    def evaluate(self, workdir: str | Path, cancel=None, on_log=None) -> np.ndarray:
        """One FE analysis with the current densities (ccx on the same mesh) → strain energies of the elements U0_e [N·mm] (sum over the
        load states, stiffness of the base material). Void elements have E_e = E_0 · max(rho^p, 1e-6).
        """
        import copy
        import dataclasses
        from ..ccx.inp import ElementScale
        from ..solve import run_study

        st = copy.copy(self.study)
        st.solver = dataclasses.replace(self.study.solver, backend="ccx")    # multiples of the stiffness can be written only by the .inp writer
        with ElementScale(self.stiffness_scale()) as es:
            res = run_study(st, Path(workdir), mesh=self.mesh, cancel=cancel, on_log=on_log)
        if es.used == 0:
            raise RuntimeError("The element stiffness multipliers did not reach the analysis (the solver did not write the ccx .inp)")
        n_lc = len(self.study.load_cases)
        if len(res.steps) < n_lc:
            raise RuntimeError(f"The analysis returned {len(res.steps)} steps, the study has {n_lc}")
        steps = res.steps[:n_lc]
        if not self.spec.all_load_cases:
            steps = [steps[int(self.spec.target_step)]]
        if self._C0 is None:
            from ..recovery import element_stiffness
            self._C0 = element_stiffness(self.study, self.mesh)
        return element_strain_energies(self.study, self.mesh, [s.U for s in steps], C=self._C0)

    def run(
        self,
        callback: Callable[[int, int, float, float], None] | None = None,
        workdir: str | Path | None = None,
        cancel=None,
    ) -> "TopologyOptimizer":
        """Runs the complete iteration loop: FE (ccx) with the stiffnesses from the current densities → strain energies of the elements → sensitivities →
        new densities; repeats until convergence or `max_iterations`.

        callback: optional function (iter, max_iter, volume_fraction, compliance)
        cancel: threading.Event for the cancellation (also stops the running ccx)
        After the loop the resulting design is calculated once more (`final_compliance`; `initial_compliance` = full material).
        """
        errors = self.check_study()
        if errors:
            raise ValueError("The topology optimization cannot be started:\n  " + "\n  ".join(errors))

        tmp_dir = None
        if workdir is None:
            tmp_dir = tempfile.TemporaryDirectory(prefix="fea_topo_")
            wdir = Path(tmp_dir.name)
        else:
            wdir = Path(workdir)
            wdir.mkdir(parents=True, exist_ok=True)

        try:
            for _ in range(self.spec.max_iterations):
                u0 = self.evaluate(wdir, cancel=cancel)
                rec = self.step(strain_energies=u0)
                if callback is not None:
                    callback(rec.iteration, self.spec.max_iterations, rec.volume_fraction, rec.total_compliance)
                if rec.converged:
                    break
            self.initial_compliance = self.history[0].total_compliance if self.history else None
            u0 = self.evaluate(wdir, cancel=cancel)
            self.final_compliance = self.current_compliance(u0)
        finally:
            if tmp_dir is not None:
                try:
                    tmp_dir.cleanup()
                except Exception:
                    pass

        return self

    def summary(self) -> dict:
        """Summary of the results of the optimisation."""
        return {
            "iterations": self.iteration,
            "target_volume_fraction": self.spec.target_volume_fraction,
            "final_volume_fraction": self.current_volume_fraction,
            "solid_elements": self.solid_elements_count,
            "void_elements": self.void_elements_count,
            "converged": self.is_converged(),
            "initial_compliance": self.initial_compliance,
            "final_compliance": self.final_compliance,
            "history": [it.to_dict() for it in self.history],
        }
