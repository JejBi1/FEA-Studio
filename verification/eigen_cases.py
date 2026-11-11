"""Verification of natural frequencies and linear buckling (fea_core.eigen): cantilever V1 200×20×10 mm, steel.

M1 clamped cantilever: f₁ (bending about the weak axis, h = 10) and f₂ (h = 20) against the Euler–Bernoulli beam
   f = (β₁L)²/(2πL²)·√(EI/ρA), β₁L = 1.875104; total mass = ρ·V; effective mass of the 1st mode = 0.6131·m
   (cantilever, analytically ∫φ dm)² / ∫φ² dm).
M2 free beam (without constraints): 6 zero rigid-body modes, f₇ = (4.730041)²/(2πL²)·√(EI/ρA).
B1 buckling of a cantilever compressed at the end: λ·F = π²EI/(4L²) about the weak axis, the second shape about the strong axis (4×).
Euler–Bernoulli neglects shear and rotary inertia (Timoshenko) – at L/h = 10–20 it lowers the frequency by ~0.2–1 %.
"""
from __future__ import annotations

import math
from pathlib import Path

from fea_core.study import Fixed, Force, LoadCase, MeshSettings
from fea_core.geometry import Part
from fea_core.selectors import find_faces
from verification.cases import Case, Check, E, RHO, _study, box

L, B, H = 200.0, 20.0, 10.0
A = B * H
I_WEAK = B * H ** 3 / 12          # bending in the direction z (h = 10)
I_STRONG = H * B ** 3 / 12        # bending in the direction y (h = 20)


def _f(beta_l: float, I: float) -> float:
    return beta_l ** 2 / (2 * math.pi * L ** 2) * math.sqrt(E * I / (RHO * A))


def _beam(wd: Path, name: str):
    return _study(name, wd / "model.step", [Part("Nosnik", box(0, 0, 0, L, B, H))], MeshSettings(size=2.5))


def m1_build(wd: Path):
    st, parts = _beam(wd, "M1 vlastni frekvence konzoly")
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.constraints.append(Fixed("vetknuti"))
    st.modal.enabled = True
    st.modal.modes = 6
    return st, parts


def m1_eval(res, st):
    m = res.modes[0]
    f = m.values
    mass = RHO * L * B * H        # note: the "total effective mass" of ccx does not include the mass of the constrained nodes
    frac_z = m.eff_mass[0][2] / mass
    # Effective-mass fraction of the 1st mode of a clamped beam (E–B): m_eff/m = (∫φ dξ)² / ∫φ² dξ for the shape
    # φ = cosh βξ − cos βξ − σ(sinh βξ − sin βξ), β = 1.875104, σ = (cosh β + cos β)/(sinh β + sin β)
    # (Blevins, Formulas for Natural Frequency and Mode Shape; Rao, Mechanical Vibrations) = 0.6131 (NOT 0.783 – that is
    # the value for the 2nd mode from another definition). Computed numerically in the test, independently of the program.
    import numpy as np
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    beta = 1.875104
    sig = (np.cosh(beta) + np.cos(beta)) / (np.sinh(beta) + np.sin(beta))
    xi = np.linspace(0.0, 1.0, 200001)
    phi = np.cosh(beta * xi) - np.cos(beta * xi) - sig * (np.sinh(beta * xi) - np.sin(beta * xi))
    frac_ref = float(_trapz(phi, xi) ** 2 / _trapz(phi ** 2, xi))
    return [
        Check("f₁ ohyb kolem slabé osy (E–B)", f[0], _f(1.875104, I_WEAK), 0.01, "Hz"),
        Check("f₂ ohyb kolem silné osy (E–B, L/h = 10)", f[1], _f(1.875104, I_STRONG), 0.02, "Hz"),
        Check("efektivní hmota 1. módu / m (E–B, numerická integrace tvaru, 0,6131)", frac_z, frac_ref, 0.01, ""),
    ]


def m2_build(wd: Path):
    st, parts = _beam(wd, "M2 volny nosnik")
    st.modal.enabled = True
    st.modal.modes = 8
    return st, parts


def m2_eval(res, st):
    m = res.modes[0]
    f = m.values
    f7 = _f(4.730041, I_WEAK)
    return [
        Check("celková hmota ρV (volné těleso – žádné vázané uzly)", m.total_mass[0], RHO * L * B * H, 1e-3, "t"),
        Check("6 módů tuhého pohybu: max f₁…₆ / f₇ (≈ 0) + 1", 1.0 + max(abs(v) for v in f[:6]) / f7, 1.0, 1e-2, ""),
        Check("f₇ volný nosník, ohyb kolem slabé osy (E–B, Timoshenko −0,9 %)", f[6], f7, 0.015, "Hz"),
    ]


F_TEST = 1000.0


def b1_build(wd: Path):
    st, parts = _beam(wd, "B1 vzper konzoly")
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=L))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Tlak 1 kN", [Force("konec", [-F_TEST, 0, 0])]))
    st.buckling.enabled = True
    st.buckling.modes = 3
    return st, parts


def b1_eval(res, st):
    lam = res.modes[0].values
    pcr = math.pi ** 2 * E * I_WEAK / (4 * L ** 2)
    return [
        Check("λ₁·F = π²EI/(4L²) (slabá osa)", lam[0] * F_TEST, pcr, 0.02, "N"),
        Check("λ₂/λ₁ = I_silná/I_slabá = 4", lam[1] / lam[0], I_STRONG / I_WEAK, 0.02, ""),
        Check("statický stav: tlak na konci = reakce", -float(res.step(0).reactions["vetknuti"][0]), -F_TEST, 1e-4, "N"),
    ]


EIGEN_CASES = {
    "M1": Case("M1", "Vlastní frekvence vetknuté konzoly (E–B, efektivní hmota)", m1_build, m1_eval),
    "M2": Case("M2", "Vlastní frekvence volného nosníku (6 tuhých pohybů)", m2_build, m2_eval),
    "B1": Case("B1", "Lineární vzpěr konzoly tlačené na konci (Euler)", b1_build, b1_eval),
}
