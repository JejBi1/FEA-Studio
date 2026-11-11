"""Parametric study and Design of Experiments (DOE / C4b).

Allows a systematic sweep of the parameters of a study:
  - change of the size of the load (multiples of the load states)
  - exchange of materials per part of the assembly
  - variants of shell thicknesses or the mesh size
  - evaluation of a table of Design Points (DP)
  - search for the Pareto front (mass vs. stress / deflection / safety).
"""
from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .assessment import part_nodes as _part_nodes
from .solve import run_study
from .study import Study, Material


@dataclass
class DesignPoint:
    index: int
    params: dict[str, Any]
    max_stress: float = 0.0          # max. von Mises [MPa]
    max_disp: float = 0.0            # max. posuv |U| [mm]
    mass: float = 0.0                # total mass [kg]
    min_safety: float = float("inf") # minimum safety factor against Re
    converged: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "params": self.params,
            "max_stress": self.max_stress,
            "max_disp": self.max_disp,
            "mass": self.mass,
            "min_safety": self.min_safety,
            "converged": self.converged,
            "error": self.error,
        }


@dataclass
class ParametricStudySpec:
    """Specification of the parameters for the DOE sweep."""
    load_factors: list[float] = field(default_factory=lambda: [1.0])
    materials: dict[str, list[str]] = field(default_factory=dict)  # part_name -> [mat_name1, mat_name2]
    mesh_sizes: list[float] = field(default_factory=list)          # mesh sizes [mm]
    max_points: int = 50                                          # safety ceiling of the number of points
    warnings: list[str] = field(default_factory=list)

    def combinations(self) -> list[dict[str, Any]]:
        keys = []
        val_lists = []
        if len(self.load_factors) > 1 or (len(self.load_factors) == 1 and self.load_factors[0] != 1.0):
            keys.append("load_factor")
            val_lists.append(self.load_factors)
        for part, mats in self.materials.items():
            if mats:
                keys.append(f"material:{part}")
                val_lists.append(mats)
        if len(self.mesh_sizes) >= 1:
            keys.append("mesh_size")
            val_lists.append(self.mesh_sizes)

        if not keys:
            # no variability, only the starting point
            return [{}]

        combos = []
        total_prod = 1
        for v in val_lists:
            total_prod *= len(v)

        for prod in itertools.product(*val_lists):
            combos.append(dict(zip(keys, prod)))
            if len(combos) >= self.max_points:
                break

        if total_prod > self.max_points:
            self.warnings = [
                f"The number of parameter combinations ({total_prod}) exceeded the limit max_points={self.max_points}. The point list was truncated to the first {self.max_points} points."
            ]
        else:
            self.warnings = []

        return combos


@dataclass
class ParametricStudyResult:
    points: list[DesignPoint]
    pareto_indices: list[int] = field(default_factory=list)
    baseline_index: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "points": [p.to_dict() for p in self.points],
            "pareto_indices": self.pareto_indices,
            "baseline_index": self.baseline_index,
            "total_points": len(self.points),
            "warnings": self.warnings,
        }

    def summary_table(self) -> str:
        lines = ["| DP | Parameters | Max σ [MPa] | Max u [mm] | Mass [kg] | Safety | Status |",
                 "|---|---|---|---|---|---|---|"]
        for p in self.points:
            p_str = ", ".join(f"{k}={v}" for k, v in p.params.items()) if p.params else "default"
            status = "OK" if p.converged else f"ERROR ({p.error})"
            star = " *" if p.index in self.pareto_indices else ""
            lines.append(f"| DP{p.index}{star} | {p_str} | {p.max_stress:.1f} | {p.max_disp:.3f} | {p.mass:.3f} | {p.min_safety:.2f} | {status} |")
        return "\n".join(lines)


def identify_pareto(points: list[DesignPoint]) -> list[int]:
    """Identifies the points of the Pareto front (minimisation of mass and stress while keeping the safety)."""
    valid = [p for p in points if p.converged]
    if not valid:
        return []

    pareto = []
    for p in valid:
        dominated = False
        for other in valid:
            if other.index == p.index:
                continue
            # other dominates p if it is at least as good in all objectives and sharply better in at least one
            if (other.mass <= p.mass and other.max_stress <= p.max_stress and
                (other.mass < p.mass or other.max_stress < p.max_stress)):
                dominated = True
                break
        if not dominated:
            pareto.append(p.index)
    return sorted(pareto)


def scale_study_loads(study: Study, factor: float) -> None:
    """Scales the loads of the study by the factor `factor` according to the kind of load (LOAD_KINDS).

    Scaling rules:
      - Force, BearingLoad, Gravity: the force / acceleration vector is scaled linearly (factor).
      - Pressure: the value and any linear gradient are scaled by the factor.
      - RemoteLoad: the force and the moment are scaled linearly (factor).
      - FrameNodeLoad: force and moment are scaled linearly (factor).
      - FrameMemberLoad: the distributed load q and q_end are scaled linearly (factor).
      - Rotation: the centrifugal force is proportional to omega^2, the angular acceleration alpha is proportional to the force linearly.
          To keep the proportion of the total force to the factor, omega is multiplied by sqrt(factor) (requires factor >= 0),
          while alpha is multiplied by the factor linearly.
      - Inertia: the translational acceleration accel and the angular acceleration alpha are multiplied by the factor;
          the angular velocity omega is multiplied by sqrt(factor); the matrix wmat by the factor.
      - ThermalLoad: NOT SCALED (temperature in °C is not a force load).
    """
    for lc in study.load_cases:
        for l in lc.loads:
            kind = getattr(l, "kind", "")
            if kind in ("force", "bearing", "gravity"):
                if hasattr(l, "vector") and l.vector:
                    l.vector = [float(v) * factor for v in l.vector]
            elif kind == "pressure":
                if hasattr(l, "value"):
                    l.value = float(l.value) * factor
                if hasattr(l, "gradient") and l.gradient:
                    l.gradient = [float(g) * factor for g in l.gradient]
            elif kind == "remote":
                if hasattr(l, "force") and l.force:
                    l.force = [float(v) * factor for v in l.force]
                if hasattr(l, "moment") and l.moment:
                    l.moment = [float(v) * factor for v in l.moment]
            elif kind == "rotation":
                if factor < 0 and getattr(l, "omega", 0.0) != 0.0:
                    raise ValueError(f"Negative load_factor={factor} cannot be used for rotation with a nonzero angular velocity omega")
                if hasattr(l, "omega"):
                    l.omega = float(l.omega) * math.sqrt(factor) if factor >= 0 else float(l.omega)
                if hasattr(l, "alpha"):
                    l.alpha = float(l.alpha) * factor
            elif kind == "inertia":
                if hasattr(l, "accel") and l.accel:
                    l.accel = [float(a) * factor for a in l.accel]
                if hasattr(l, "alpha") and l.alpha:
                    l.alpha = [float(a) * factor for a in l.alpha]
                if hasattr(l, "omega") and l.omega:
                    if factor < 0 and any(w != 0.0 for w in l.omega):
                        raise ValueError(f"Negative load_factor={factor} cannot be used for inertia with a nonzero omega")
                    l.omega = [float(w) * math.sqrt(factor) for w in l.omega] if factor >= 0 else l.omega
                if hasattr(l, "wmat") and l.wmat:
                    l.wmat = [[float(v) * factor for v in row] for row in l.wmat]
            elif kind == "frame_node":
                if hasattr(l, "force") and l.force:
                    l.force = [float(v) * factor for v in l.force]
                if hasattr(l, "moment") and l.moment:
                    l.moment = [float(v) * factor for v in l.moment]
            elif kind == "frame_member":
                if hasattr(l, "q") and l.q:
                    l.q = [float(v) * factor for v in l.q]
                if hasattr(l, "q_end") and l.q_end:
                    l.q_end = [float(v) * factor for v in l.q_end]
            elif kind == "frame_temperature":
                l.dT, l.grad_z, l.grad_y = (float(v) * factor for v in (l.dT, l.grad_z, l.grad_y))
            elif kind == "thermal":
                # The thermal load is not scaled
                pass
            else:
                if hasattr(l, "vector") and l.vector:
                    l.vector = [float(v) * factor for v in l.vector]
                elif hasattr(l, "value") and isinstance(l.value, (int, float)):
                    l.value = float(l.value) * factor


def run_parametric_study(study: Study, spec: ParametricStudySpec,
                         base_workdir: Path, on_progress: Callable[[int, int], None] | None = None) -> ParametricStudyResult:
    """Runs the parametric sweep of the study according to the given specification."""
    from .results import Results, von_mises
    from .signature import mesh_signature
    from .loads import element_volumes

    combos = spec.combinations()
    total = len(combos)
    points: list[DesignPoint] = []
    base_workdir = Path(base_workdir)
    base_workdir.mkdir(parents=True, exist_ok=True)
    warnings = list(getattr(spec, "warnings", []))

    # Shortcut for linear problems with a pure sweep of load_factor:
    # One calculation of the baseline mesh is enough and the displacements / stresses / reactions are rescaled linearly.
    is_pure_load_sweep = (
        not study.is_nonlinear()
        and len(combos) > 0
        and all(list(p.keys()) == ["load_factor"] or not p for p in combos)
    )

    if is_pure_load_sweep:
        base_dp_dir = base_workdir / "baseline"
        base_dp_dir.mkdir(parents=True, exist_ok=True)
        res_base = run_study(study, base_dp_dir)
        warnings.extend(res_base.warnings)

        # Mass (the same for all points)
        vols = element_volumes(res_base.mesh)
        tot_mass_t = 0.0
        for p_idx, p_name in enumerate(res_base.mesh.part_names):
            mat = study.material_of(p_name)
            rho = mat.rho if mat and mat.rho else 7.85e-9
            part_elem_mask = (res_base.mesh.tet_part == p_idx)
            tot_mass_t += float(np.sum(vols[part_elem_mask])) * rho
        mass_kg = tot_mass_t * 1000.0

        for idx, params in enumerate(combos):
            if on_progress:
                on_progress(idx, total)
            dp = DesignPoint(index=idx, params=params)
            factor = float(params.get("load_factor", 1.0))
            dp_dir = base_workdir / f"DP_{idx:03d}"
            dp_dir.mkdir(parents=True, exist_ok=True)

            try:
                s_vm_max = 0.0
                u_max = 0.0
                min_s = float("inf")
                scaled_steps = []

                for step in res_base.steps:
                    U_s = step.U * factor
                    S_s = step.S * factor
                    reac_s = {k: v * factor for k, v in step.reactions.items()}

                    vm = von_mises(S_s)
                    s_vm_max = max(s_vm_max, float(np.max(vm)))
                    unorm = np.linalg.norm(U_s, axis=1)
                    u_max = max(u_max, float(np.max(unorm)))

                    for p_idx, p_name in enumerate(res_base.mesh.part_names):
                        mat = study.material_of(p_name)
                        if mat and mat.Re and mat.Re > 0:
                            part_nodes = _part_nodes(res_base.mesh, p_idx)
                            if len(part_nodes) > 0:
                                vm_part = vm[part_nodes]
                                max_vm_part = float(np.max(vm_part))
                                if max_vm_part > 1e-3:
                                    min_s = min(min_s, mat.Re / max_vm_part)

                    new_step = copy.copy(step)
                    new_step.U = U_s
                    new_step.S = S_s
                    new_step.reactions = reac_s
                    new_step._mises = vm
                    new_step._umag = unorm
                    scaled_steps.append(new_step)

                scaled_res = Results(
                    mesh=res_base.mesh,
                    steps=scaled_steps,
                    warnings=list(res_base.warnings),
                    info=dict(res_base.info),
                )
                scaled_res.save(dp_dir / "results.npz")

                dp.max_stress = s_vm_max
                dp.max_disp = u_max
                dp.mass = mass_kg
                dp.min_safety = min_s if min_s != float("inf") else 0.0
                dp.converged = True
            except Exception as e:
                dp.converged = False
                dp.error = str(e)

            points.append(dp)

        pareto = identify_pareto(points)
        return ParametricStudyResult(points=points, pareto_indices=pareto, baseline_index=0, warnings=warnings)

    # General run with support for caching the mesh (re-meshing only when mesh_signature changes)
    cached_mesh = None
    cached_mesh_sig = None

    for idx, params in enumerate(combos):
        if on_progress:
            on_progress(idx, total)

        dp = DesignPoint(index=idx, params=params)
        dp_dir = base_workdir / f"DP_{idx:03d}"
        dp_dir.mkdir(parents=True, exist_ok=True)

        # Clone of the study for the given point
        st = copy.deepcopy(study)

        try:
            # 1. Application of the parameters
            if "load_factor" in params:
                factor = float(params["load_factor"])
                scale_study_loads(st, factor)

            for k, val in params.items():
                if k.startswith("material:"):
                    part_name = k.split(":", 1)[1]
                    st.part_materials[part_name] = str(val)

            if "mesh_size" in params:
                mesh_val = float(params["mesh_size"])
                if hasattr(st, "mesh"):
                    st.mesh.size = mesh_val
                elif hasattr(st, "mesh_settings"):
                    st.mesh_settings.size = mesh_val

            # 2. Calculation with reuse of the mesh
            cur_sig = mesh_signature(st)
            mesh_arg = cached_mesh if (cached_mesh is not None and cached_mesh_sig == cur_sig) else None

            res = run_study(st, dp_dir, mesh=mesh_arg)
            cached_mesh = res.mesh
            cached_mesh_sig = cur_sig
            warnings.extend(res.warnings)

            # 3. Evaluation
            s_vm_max = 0.0
            u_max = 0.0
            min_s = float("inf")
            for step in res.steps:
                vm = von_mises(step.S)
                s_vm_max = max(s_vm_max, float(np.max(vm)))
                unorm = np.linalg.norm(step.U, axis=1)
                u_max = max(u_max, float(np.max(unorm)))

                # safety against Re per material (correctly by the nodes of tets)
                for p_idx, p_name in enumerate(res.mesh.part_names):
                    mat = st.material_of(p_name)
                    if mat and mat.Re and mat.Re > 0:
                        part_nodes = _part_nodes(res.mesh, p_idx)
                        if len(part_nodes) > 0:
                            vm_part = vm[part_nodes]
                            max_vm_part = float(np.max(vm_part))
                            if max_vm_part > 1e-3:
                                min_s = min(min_s, mat.Re / max_vm_part)

            # Hmotnost
            vols = element_volumes(res.mesh)
            tot_mass_t = 0.0
            for p_idx, p_name in enumerate(res.mesh.part_names):
                mat = st.material_of(p_name)
                rho = mat.rho if mat and mat.rho else 7.85e-9
                part_elem_mask = (res.mesh.tet_part == p_idx)
                tot_mass_t += float(np.sum(vols[part_elem_mask])) * rho
            mass_kg = tot_mass_t * 1000.0  # t -> kg

            dp.max_stress = s_vm_max
            dp.max_disp = u_max
            dp.mass = mass_kg
            dp.min_safety = min_s if min_s != float("inf") else 0.0
            dp.converged = True

        except Exception as e:
            dp.converged = False
            dp.error = str(e)

        points.append(dp)

    pareto = identify_pareto(points)
    return ParametricStudyResult(points=points, pareto_indices=pareto, baseline_index=0, warnings=warnings)
