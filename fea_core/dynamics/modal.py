"""Modal basis of the dynamics from ccx *FREQUENCY (stage F8.1): shapes φ_j normalised by the mass, modal stresses σ_j, modal
reactions, participation factors Γ_j and effective masses.

Verified by a trial (scratch/f8_dyn/exp_norm.py, 15. 10. 2026, cantilever 10×10×100 C3D10):
  - the displacements in the .frd of the step *FREQUENCY are normalised by the mass: φᵀMφ = 1.000000 (M = consistent mass by own
    integration, dynamics.mass.mass_times); Γ_j from the .dat = φ_jᵀ·M·r to all digits;
  - *EL FILE S in the step *FREQUENCY writes the modal stresses (the same normalisation), *NODE FILE RF = K·φ_j in all nodes
    (Σ RF on the constraints = −ω_j²·Σ M·φ_j in the free nodes to 1e-5);
  - the nodes of the point mass (*MASS) are not written to the .frd → their motion from the kinematics of the face (mass.node_motion).
The sign of the shape is the sign of ccx (without turning as eigen._shapes) → Γ_j from the .dat and φ_j are consistent.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class DynamicsError(RuntimeError):
    pass


@dataclass
class ModalBasis:
    freq: np.ndarray                      # (K,) Hz
    shapes: np.ndarray                    # (K, N, 3) normalised by the mass (φᵀMφ = 1)
    stress: np.ndarray                    # (K, N, 6) float32 modal stresses
    gamma: np.ndarray                     # (K, 6) participation factors of ccx (x y z rx ry rz)
    eff_mass: np.ndarray                  # (K, 6) [t]
    total_mass: np.ndarray                # (6,) total mass of ccx [t] (without the mass of the constrained nodes)
    rf_sum: np.ndarray                    # (K, 3) Σ of the modal reactions on the constraints (K·φ_j)
    rf_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))   # constraint nodes
    rf: np.ndarray | None = None          # (K, n_c, 3) modal reactions in the nodes of the constraints
    mass_motion: list = field(default_factory=list)      # [(K, k, 3) motion of the points of the point masses]
    model_mass: float = 0.0               # mass of the model [t] (bodies + added mass)
    checks: dict = field(default_factory=dict)
    prestress: str | None = None
    prestress_nlgeom: bool = False
    rf_from_inertia: bool = False         # preload: the resultant of the reactions from the equilibrium of the inertia forces (ccx FORC without K_sigma·φ)

    @property
    def omega(self) -> np.ndarray:
        return 2.0 * np.pi * np.asarray(self.freq, float)

    @property
    def K(self) -> int:
        return len(self.freq)


def _read_blocks(frd_path: Path, n: int, last_step_only: bool):
    from ..ccx.frd import read_frd
    frd = read_frd(frd_path)
    steps = frd.steps()
    step = steps[-1] if last_step_only and steps else None
    rank = None
    if step is not None:
        # preload: the NUMSTP blocks of the step *FREQUENCY continue after the increments of the static step (modes 2, 3, 4, 5 instead of
        # 1…4) → without renumbering the shapes shifted by one (φ₁ = 0, normalisation φᵀMφ = [0, 1, 1, 1], Γ wrong);
        # found by the cross-check Y9 in Abaqus, verification/abaqus_dynamics.py) → the order of the mode in the step
        rank = {m: i + 1 for i, m in enumerate(sorted({b.mode for b in frd.blocks if b.step == step}))}
    out = {}
    for b in frd.blocks:
        if step is not None and b.step != step:
            continue
        if b.name not in ("DISP", "STRESS", "FORC"):
            continue
        msk = b.node_ids <= n
        nc = 3 if b.name != "STRESS" else 6
        a = np.zeros((n, nc))
        a[b.node_ids[msk] - 1] = b.values[msk, :nc]
        out[(b.name, rank[b.mode] if rank is not None else b.mode)] = a
    return out


def constrained_nodes(study, mesh) -> np.ndarray:
    """Nodes of the mesh held by the constraints (reactions RF): clamping, displacements, cylindrical and distributing remote constraint. The elastic support
    does not belong here – its nodes are free, the reaction = −k·φ (spring_reaction_sum).
    """
    from ..solve import constraint_masks
    msk = np.zeros(len(mesh.nodes), bool)
    for v in constraint_masks(study, mesh).values():
        msk |= v.any(axis=1)
    return np.nonzero(msk)[0]


def spring_reaction_sum(study, mesh, shapes: np.ndarray) -> np.ndarray:
    """(K, 3) sum of the reactions of the elastic support −k·φ_j over the modes."""
    from ..springs import supports, reactions
    out = np.zeros((len(shapes), 3))
    if not supports(study):
        return out
    for j, phi in enumerate(shapes):
        out[j] = sum(r.sum(axis=0) for r in reactions(study, mesh, phi).values())
    return out


def run_basis(study, mesh, workdir: Path, log, on_line=None, cancel=None, modes: int | None = None,
              f_max: float | None = None, prestress_case: str | None = None, storage: bool = False,
              probe_nodes=None, job: str = "dyn_modal") -> ModalBasis:
    """Job of ccx *FREQUENCY on the mesh of the study (constraints of the study, added mass) → ModalBasis. prestress_case = state,
    after which it is computed with the stress stiffness (*STEP, PERTURBATION).
    """
    from ..ccx.inp import write_inp
    from ..ccx.runner import run_ccx, CcxError
    from ..config import CCX_EXE
    from ..eigen import read_dat, _sub_study
    from ..study import LoadCase
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ds = study.dynamics
    k = int(modes or ds.modes)
    fm = f_max if f_max is not None else ds.f_max
    line = f"{k}, 0." + (f", {float(fm):.12g}" if fm else "")
    if prestress_case:
        lc = next((c for c in study.load_cases if c.name == prestress_case), None)
        if lc is None:
            raise DynamicsError(f"Prestress: load case '{prestress_case}' does not exist")
        st = _sub_study(study, [lc])
        log(f"Natural frequencies after prestress by case '{lc.name}' (CalculiX, {k} modes)")
    else:
        st = _sub_study(study, [LoadCase("Natural frequencies", [])])
        log(f"Modal basis of the dynamics (CalculiX, {k} modes)")
    temps = None
    if prestress_case and study.has_thermal_loads() and study.thermal_loads(lc):
        temps = None                                   # thermal preload not yet (the thermal results are not available)
        log("Prestress: the thermal load of the case is not used in the prestressed modal analysis")
        st.load_cases = [LoadCase(lc.name, [x for x in lc.loads if x.kind != "thermal"])]
    try:
        write_inp(st, mesh, workdir / f"{job}.inp", temperatures=temps, procedure="modal", eigen_line=line,
                  modal_output=True, storage=storage, prestress=bool(prestress_case), probe_nodes=probe_nodes)
    except ValueError as e:
        raise DynamicsError(str(e)) from None
    exe = Path(study.solver.exe) if study.solver.exe else CCX_EXE
    r = run_ccx(workdir, job, exe=exe, on_line=on_line, cancel=cancel)
    if not r.finished or r.errors:
        raise CcxError("CalculiX (dynamics natural frequencies) ended with an error:\n  "
                       + "\n  ".join(r.errors or [f"code {r.returncode}"]) + f"\nlog: {r.log_path}")
    d = read_dat(workdir / f"{job}.dat", "modal")
    if not d["freq"]:
        raise DynamicsError("Dynamics: ccx returned no mode (try a different frequency range)")
    n = len(mesh.nodes)
    blk = _read_blocks(workdir / f"{job}.frd", n, last_step_only=bool(prestress_case))
    kk = len(d["freq"])
    shapes = np.zeros((kk, n, 3))
    stress = np.zeros((kk, n, 6), np.float32)
    rfs = np.zeros((kk, n, 3))
    for j in range(kk):
        if ("DISP", j + 1) in blk:
            shapes[j] = blk[("DISP", j + 1)]
        if ("STRESS", j + 1) in blk:
            stress[j] = blk[("STRESS", j + 1)]
        if ("FORC", j + 1) in blk:
            rfs[j] = blk[("FORC", j + 1)]
    freq = np.asarray(d["freq"], float)
    ok = freq > 0
    if not ok.all():
        log(f"Dynamics: {int((~ok).sum())} modes with zero frequency (rigid-body motion of a free body) – skipped")
    cn = constrained_nodes(study, mesh)
    is_nlgeom = bool(getattr(study.dynamics.prestress, "nlgeom", False)) if prestress_case else False
    basis = ModalBasis(freq[ok], shapes[ok], stress[ok], np.asarray(d["participation"], float)[ok],
                       np.asarray(d["eff_mass"], float)[ok], np.asarray(d["total"] or [0.0] * 6, float),
                       rfs[ok][:, cn].sum(axis=1), cn, rfs[ok][:, cn].astype(np.float32), prestress=prestress_case,
                       prestress_nlgeom=is_nlgeom)
    from .mass import mass_points, node_motion, total_mass, mass_items
    if prestress_case:
        # the ccx .frd FORC in *STEP, PERTURBATION contains only K_0·phi without the geometric stiffness K_sigma·phi.
        # The RESULTANT of the reactions K_total·phi is given by the global equilibrium (exactly):
        # sum_{cn} (K_total·phi)_i = inert_i = -omega^2 sum_{free} (M·phi)_i → the reactions are rescaled to it.
        # The distribution among the nodes / supports remains according to K_0·phi (APPROXIMATE) and the check of the equilibrium of the modal reactions
        # loses its sense by it (it holds by construction) – in checks it is marked as not performed.
        Mphi = modal_mass_products(study, mesh, basis)
        free = np.ones(len(mesh.nodes), bool)
        free[cn] = False
        inert = -(basis.omega ** 2)[:, None] * Mphi[:, free].sum(axis=1)
        for mot, mu in point_mass_terms(study, mesh, basis):
            inert += -(basis.omega ** 2)[:, None] * np.einsum("kpi,p->ki", mot, mu)
        rfs_sum = basis.rf.sum(axis=1)
        rf_scaled = np.array(basis.rf, dtype=np.float32, copy=True)
        for k_idx in range(basis.K):
            for i in range(3):
                s_ccx = rfs_sum[k_idx, i]
                s_exact = inert[k_idx, i]
                if abs(s_ccx) > 1e-12:
                    rf_scaled[k_idx, :, i] = rf_scaled[k_idx, :, i] * (s_exact / s_ccx)
                elif abs(s_exact) > 1e-12 and len(cn) > 0:
                    rf_scaled[k_idx, :, i] = s_exact / len(cn)
        basis.rf = rf_scaled
        basis.rf_sum = basis.rf.sum(axis=1)
        basis.rf_from_inertia = True
    basis.rf_sum = basis.rf_sum + spring_reaction_sum(study, mesh, basis.shapes)   # elastic support = reaction −k·φ
    basis.mass_motion = [np.stack([node_motion(mesh, info, basis.shapes[j]) for j in range(basis.K)])
                         if basis.K else np.zeros((0, len(info["nodes"]), 3)) for info in mass_points(study, mesh)]
    added = 0.0
    for it in mass_items(study):
        if it.kind == "point_mass":
            added += float(it.mass)
        else:
            from .mass import surface_mass_nodes
            added += float(surface_mass_nodes(mesh, it)[1].sum())
    basis.model_mass = total_mass(study, mesh) + added
    basis.checks = check_basis(study, mesh, basis)
    c = basis.checks
    if c.get("gamma_error", 0.0) > 1e-3 and c.get("mass_norm_error", 1.0) <= 5e-3 and _has_rigid_body(study):
        # Γ from ccx is wrong for a *RIGID BODY with a free ROT NODE: ccx adds M·φ also in DOF 1–3 of the ROT node (rotation) as
        # displacements – a point mass off the face / with rotary inertia gave Σ Γ² = 1.03 up to 14× the mass of the model
        # (verified 15. 10. 2026, scratch/f8_examples/pm_gamma.py; own integration Σ Γ² → 0.993). The normalisation φᵀMφ
        # by own integration agrees → the mass is complete and Γ = φᵀMr by own integration is correct → it is used
        # (the normalisation of the high modes of a coarse mesh differs by up to ~4e-3 by the integration rule, not by a missing mass).
        gam = own_gamma(study, mesh, basis)
        basis.gamma = np.array(basis.gamma, float, copy=True)
        basis.eff_mass = np.array(basis.eff_mass, float, copy=True)
        basis.gamma[:, :3] = gam
        basis.eff_mass[:, :3] = gam ** 2                 # φᵀMφ = 1
        basis.checks = check_basis(study, mesh, basis)
        basis.checks.update(gamma_source="own", gamma_error_ccx=c["gamma_error"])
    return basis


def _has_rigid_body(study) -> bool:
    """The model has *RIGID BODY (point mass with a rigid constraint, rigid remote load / constraint)."""
    if any(getattr(m, "kind", "") == "point_mass" and m.coupling == "rigid" for m in study.masses):
        return True
    items = list(study.constraints) + [ld for lc in study.load_cases for ld in lc.loads]
    return any(getattr(x, "coupling", None) == "rigid" for x in items)


def own_gamma(study, mesh, basis: ModalBasis) -> np.ndarray:
    """Γ_j = φ_jᵀ·M·r (K, 3) by own integration (bodies, surface and point mass)."""
    gam = modal_mass_products(study, mesh, basis).sum(axis=1)
    for mot, mu in point_mass_terms(study, mesh, basis):
        gam += np.einsum("kpi,p->ki", mot, mu)
    return gam


def modal_mass_products(study, mesh, basis: ModalBasis) -> np.ndarray:
    """M·φ_j (K, N, 3) by own integration (consistent mass of the bodies + surface mass in the nodes)."""
    from .mass import mass_times, mass_items, surface_mass_nodes
    Mphi = mass_times(study, mesh, basis.shapes)
    for it in mass_items(study):
        if it.kind == "surface_mass":
            idx, mm = surface_mass_nodes(mesh, it)
            Mphi[:, idx] += mm[None, :, None] * basis.shapes[:, idx]
    return Mphi


def point_mass_terms(study, mesh, basis: ModalBasis) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(motion of the points (K, k, 3), masses of the points (k,))] of the point masses."""
    from .mass import mass_points
    return [(mot, np.array([mu for _, mu in info["nodes"]])) for mot, info in
            zip(basis.mass_motion, mass_points(study, mesh))]


def check_basis(study, mesh, basis: ModalBasis) -> dict:
    """Checks of the basis: normalisation φᵀMφ = 1 and Γ_j = φᵀMr by own integration against ccx (catches a wrong reading / the mass),
    the equilibrium of the mode Σ RF(constraints) + ω²·Σ M·φ(free) = 0, coverage of the effective mass by the directions.
    """
    if basis.K == 0:
        return {}
    Mphi = modal_mass_products(study, mesh, basis)
    gen = np.einsum("kni,kni->k", basis.shapes, Mphi)
    gam = Mphi.sum(axis=1)                              # (K, 3) φᵀ M r
    free = np.ones(len(mesh.nodes), bool)
    free[basis.rf_nodes] = False
    inert = -(basis.omega ** 2)[:, None] * Mphi[:, free].sum(axis=1)
    for mot, mu in point_mass_terms(study, mesh, basis):
        gen += np.einsum("kpi,kpi,p->k", mot, mot, mu)
        gam += np.einsum("kpi,p->ki", mot, mu)
        inert += -(basis.omega ** 2)[:, None] * np.einsum("kpi,p->ki", mot, mu)
    # only modes with a non-negligible effective mass (Γ² > 1e-4·m): for the others Γ is the noise of the 6 digits of the .frd
    m_tot = max(float(basis.model_mass), 1e-30)
    sig = (basis.gamma[:, :3] ** 2).max(axis=1) > 1e-4 * m_tot
    scale = np.maximum(np.abs(basis.gamma[:, :3]).max(), 1e-30)
    gam_err = float(np.abs(gam - basis.gamma[:, :3])[sig].max() / scale) if sig.any() else 0.0
    # scale = size of the individual reactions (the sum nearly cancels for torsional modes)
    rnorm = np.linalg.norm(basis.rf, axis=2).sum(axis=1) if basis.rf is not None and basis.rf.size else 0.0
    rscale = np.maximum(np.maximum(np.abs(inert).max(axis=1), rnorm), 1e-30)
    # Σ K·φ over all nodes = 0 (rigid displacements) → Σ RF(constraints) = −Σ K·φ(free) = −ω²·Σ M·φ(free) = inert
    has_supp = len(basis.rf_nodes) or np.abs(basis.rf_sum).max() > 0
    eq_err = float((np.abs(basis.rf_sum - inert).max(axis=1) / rscale)[sig].max()) if has_supp and sig.any() else 0.0
    tot = basis.total_mass[:3] if basis.total_mass is not None and len(basis.total_mass) else np.zeros(3)
    m_model = basis.model_mass
    cover = (basis.eff_mass[:, :3].sum(axis=0) / m_model).tolist() if m_model > 0 else [0.0] * 3
    if getattr(basis, "rf_from_inertia", False):      # the resultant of the reactions is taken from the equilibrium → the check would always hold
        eq_err = 0.0
    return {"mass_norm_error": float(np.abs(gen - 1.0).max()), "gamma_error": gam_err, "mode_equilibrium": eq_err,
            "mode_equilibrium_checked": not getattr(basis, "rf_from_inertia", False),
            "mass_fraction_xyz": [float(c) for c in cover], "model_mass_t": m_model,
            "ccx_total_mass_t": [float(v) for v in tot]}


def basis_warnings(basis: ModalBasis, dirs: set[int], f_top: float | None) -> list[str]:
    """Warnings on the modal basis: Σ m_eff < 90 % in the excited direction (EN 1998-1, 4.3.3.3.1), modes up to 1.5× the highest
    excitation frequency, checks of the normalisation.
    """
    out = []
    c = basis.checks
    if c.get("mass_norm_error", 0) > 1e-3 or c.get("gamma_error", 0) > 1e-3:
        out.append(f"Dynamics: the modal basis does not match the own mass integration (normalization {c['mass_norm_error']:.2g}, Γ {c['gamma_error']:.2g}) – verify the results")
    if c.get("mode_equilibrium", 0) > 1e-3:
        out.append(f"Dynamics: modal reaction equilibrium {c['mode_equilibrium']:.2g} > 1e-3")
    if getattr(basis, "rf_from_inertia", False):
        out.append("Prestressed dynamics: the resultant support reaction is exact (from equilibrium of inertia forces), its split between several supports is approximate (CalculiX does not output the stress-stiffness part)")
    if getattr(basis, "prestress_nlgeom", False):
        out.append("Dynamics: prestress with NLGEOM – CalculiX in *FREQUENCY computes stress stiffness K_sigma, but evaluates mode shapes on reference (undeformed) geometry (Abaqus with NLGEOM updates geometry, frequency difference ~0.26% in tension, ~0.53% in compression).")
    names = "xyz"
    for d in sorted(dirs):
        frac = c.get("mass_fraction_xyz", [1, 1, 1])[d]
        if frac < 0.9:
            out.append(f"Dynamics: effective modal mass in direction {names[d]} only {100 * frac:.1f} % (< 90 %) – add modes (the missing mass is added in the static correction / for shock)")
    if f_top and basis.K and basis.freq.max() < 1.5 * f_top:
        out.append(f"Dynamics: the highest basis mode {basis.freq.max():.4g} Hz < 1.5 × the highest excitation frequency {f_top:.4g} Hz – add modes")
    return out


# ------------------------------------------------------------------ damping

def zetas(spec, freq: np.ndarray) -> np.ndarray:
    """Relative damping of the modes ζ_j according to DampingSpec (structural → ζ = η/2 as a substitute in time)."""
    f = np.asarray(freq, float)
    if spec.kind == "rayleigh" and spec.f1 and spec.f2:
        w1, w2 = 2 * np.pi * spec.f1, 2 * np.pi * spec.f2
        a = 2 * spec.zeta * w1 * w2 / (w1 + w2)
        b = 2 * spec.zeta / (w1 + w2)
        w = 2 * np.pi * f
        return a / (2 * np.maximum(w, 1e-30)) + b * w / 2
    if spec.kind == "structural":
        return np.full_like(f, 0.5 * (spec.eta if spec.eta else 2 * spec.zeta))
    if spec.table:
        t = np.asarray(spec.table, float)
        return np.interp(f, t[:, 0], t[:, 1])
    return np.full_like(f, float(spec.zeta))


def static_solutions(study, mesh, cases: list, workdir: Path, log, on_line=None, cancel=None) -> list[dict]:
    """Static solution of the auxiliary states (the residual static correction: the state of the excitation by a force, the unit acceleration
    of the base) on the same mesh: [{"U", "S", "RF_sum"}]. One job of ccx (steps by states).
    """
    from ..ccx.inp import write_inp
    from ..ccx.runner import run_ccx, CcxError
    from ..ccx.frd import read_frd
    from ..config import CCX_EXE
    workdir = Path(workdir)
    st = copy.copy(study)
    st.load_cases = list(cases)
    st.joints = copy.copy(study.joints)
    st.joints.preload_case = False
    job = "dyn_static"
    write_inp(st, mesh, workdir / f"{job}.inp")
    exe = Path(study.solver.exe) if study.solver.exe else CCX_EXE
    r = run_ccx(workdir, job, exe=exe, on_line=on_line, cancel=cancel)
    if not r.finished or r.errors:
        raise CcxError("CalculiX (dynamics static correction) ended with an error:\n  "
                       + "\n  ".join(r.errors or [f"code {r.returncode}"]))
    frd = read_frd(workdir / f"{job}.frd")
    n = len(mesh.nodes)
    cn = constrained_nodes(study, mesh)
    out = []
    for s in frd.steps():
        res = {}
        for name, nc in (("DISP", 3), ("STRESS", 6), ("FORC", 3)):
            b = frd.get(name, s)
            a = np.zeros((n, nc))
            msk = b.node_ids <= n
            a[b.node_ids[msk] - 1] = b.values[msk, :nc]
            res[name] = a
        out.append({"U": res["DISP"], "S": res["STRESS"].astype(np.float32), "RF": res["FORC"][cn]})
    return out
