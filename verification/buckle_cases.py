"""Verification of the nonlinear buckling with imperfection (fea_core.imperfect, F9 C2): GNIA / GMNIA.

BK1 elastic cantilever (B1 geometry 200×20×10, steel) with an imperfection affine to the 1st buckling mode, e0 = L/1000: the
    additional tip deflection of the second-order theory δ = e0·r/(1 − r), r = N/N_cr (Timoshenko & Gere, Theory of Elastic
    Stability, 1.11 – exact for an imperfection in the shape of the buckling mode); r from the linear buckling factor of the same
    mesh, N_cr itself against Euler π²EI/(4L²).
BK2 pin-ended column L = 540 mm, rectangle 30×20 (half model: symmetry plane at mid-span, laterally held loaded end), S235,
    λ̄ = √(A·f_y/N_cr) ≈ 1.0. Half-sine bow with the Ayrton–Perry equivalent amplitude
    e0 = (1 − χ)(1 − χλ̄²)/χ · W_el/A, χ = EN 1993-1-1 (6.49), curve c (α = 0.49, solid sections, Table 6.2). GNIA (elastic) +
    first-yield criterion σ_max = f_y must give N = χ·A·f_y (the buckling curve is the first-yield solution of this imperfect
    column; EN 1993-1-1 5.3.2(11) uses the same equivalence).
BK3 the same column, GMNIA: bilinear material (f_y = 235 MPa, E_t = E/1000), half-sine bow e0 = L/1000 (EN 1090-2 tolerance),
    no residual stresses. Reference = column deflection curve (Chwalla / von Kármán method, own 1D solver below: fibre section
    with the same bilinear law, plane sections, equilibrium M = N·(w0 + v) at 60 points, displacement control of the mid-span
    deflection → the maximum of N exactly, past the limit point). Plausibility: χ_GMNIA ≥ χ of EN curve c (curve c covers the
    residual stresses of real hot-rolled solid sections, which this model does not have).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from fea_core.geometry import Part
from fea_core.imperfect.model import NonlinearBucklingSettings
from fea_core.selectors import find_faces
from fea_core.study import Displacement, Force, LoadCase, Material, MeshSettings
from verification.cases import Case, Check, E, NU, RHO, _study, box

# ------------------------------------------------------------------ BK1 elastic cantilever
L1, B1W, H1 = 200.0, 20.0, 10.0
I1 = B1W * H1 ** 3 / 12
F1 = 1000.0


def bk1_build(wd: Path):
    st, parts = _study("BK1 imperfektni konzola", wd / "model.step", [Part("Nosnik", box(0, 0, 0, L1, B1W, H1))],
                       MeshSettings(size=2.5))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=L1))
    from fea_core.study import Fixed
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Tlak 1 kN", [Force("konec", [-F1, 0, 0])]))
    pcr = math.pi ** 2 * E * I1 / (4 * L1 ** 2)
    st.nlbuckling = NonlinearBucklingSettings(enabled=True, imperfection="modes", modes=[1], amplitude=L1 / 1000,
                                              max_factor=0.92 * pcr / F1, increments=20, monitor_point=[L1, B1W / 2, H1 / 2],
                                              monitor_dir=[0.0, 0.0, 1.0])
    return st, parts


def _at(lam, u, target):
    return float(np.interp(target, lam, u))


def bk1_eval(res, st):
    info = res.info["nlbuckling"]
    lam = np.asarray(info["curve"]["lambda"])
    u = np.abs(np.asarray(info["curve"]["u_mm"]))
    lcr = info["linear_factor"]
    e0 = info["imperfection"]["e0_mm"]
    pcr = math.pi ** 2 * E * I1 / (4 * L1 ** 2)
    out = [Check("N_cr (linear buckling factor × F) = π²EI/(4L²)", lcr * F1, pcr, 0.02, "N"),
           Check("imperfection amplitude e0 = L/1000", e0, L1 / 1000, 1e-9, "mm")]
    for r in (0.5, 0.8, 0.9):
        out.append(Check(f"tip deflection at N/N_cr = {r}: e0·r/(1 − r)", _at(lam, u, r * lcr), e0 * r / (1 - r), 0.02, "mm"))
    out.append(Check("ramp completed (λ max reached, no limit point – elastic)", float(info["result"]["status"] == "max_factor"),
                     1.0, 0.0, ""))
    return out


# ------------------------------------------------------------------ BK2 / BK3 pin-ended column (half model)
LC, BC, HC = 540.0, 30.0, 20.0          # full length, width (y), thickness (z – weak axis)
LH = LC / 2
AC = BC * HC
IC = BC * HC ** 3 / 12
WC = BC * HC ** 2 / 6
FY = 235.0
FC = 1000.0                              # reference load of the case [N]
ALPHA_C = 0.49


def chi_en(lbar: float, alpha: float = ALPHA_C) -> float:
    """EN 1993-1-1 (6.49)."""
    phi = 0.5 * (1 + alpha * (lbar - 0.2) + lbar ** 2)
    return min(1.0, 1.0 / (phi + math.sqrt(phi ** 2 - lbar ** 2)))


def _column(wd: Path, name: str, mat: Material, e0: float, max_factor: float):
    st, parts = _study(name, wd / "model.step", [Part("Prut", box(0, 0, 0, LH, BC, HC))], MeshSettings(size=4.0))
    st.materials.clear()
    st.add_material(mat, default=True)
    st.add_selection("stred", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=LH))
    st.constraints.append(Displacement("stred", ux=0.0))           # symmetry plane at mid-span
    st.constraints.append(Displacement("konec", uy=0.0, uz=0.0))   # pin: laterally held, rotation and axial motion free
    st.load_cases.append(LoadCase("Tlak", [Force("konec", [-FC, 0, 0])]))
    st.nlbuckling = NonlinearBucklingSettings(
        enabled=True, imperfection="bow", bow_a=[-LH, BC / 2, HC / 2], bow_b=[LH, BC / 2, HC / 2],
        bow_direction=[0.0, 0.0, 1.0], amplitude=e0, max_factor=max_factor, increments=40,
        monitor_point=[0.0, BC / 2, HC / 2], monitor_dir=[0.0, 0.0, 1.0], min_increment=1e-4)
    return st, parts


def _ncr():
    return math.pi ** 2 * E * IC / LC ** 2


def _lbar():
    return math.sqrt(AC * FY / _ncr())


def e0_ayrton_perry() -> float:
    chi = chi_en(_lbar())
    return (1 - chi) * (1 - chi * _lbar() ** 2) / chi * WC / AC


def bk2_build(wd: Path):
    mat = Material("Ocel S235", E, NU, RHO, Re=FY, Rm=360.0)
    chi = chi_en(_lbar())
    return _column(wd, "BK2 GNIA prvni plasticita", mat, e0_ayrton_perry(), 1.15 * chi * AC * FY / FC)


def bk2_eval(res, st):
    info = res.info["nlbuckling"]
    chi = chi_en(_lbar())
    ly = info["result"].get("first_yield_factor")
    return [Check("λ̄ = √(A·f_y/N_cr) of the model (linear buckling factor)",
                  math.sqrt(AC * FY / (info["linear_factor"] * FC)), _lbar(), 0.02, ""),
            Check("GNIA first yield N_y = χ·A·f_y (EN 1993-1-1 curve c, Ayrton–Perry imperfection)",
                  (ly or 0.0) * FC, chi * AC * FY, 0.03, "N")]


# --- column deflection curve (independent 1D reference for BK3)
def _section(eps0, kappa, Et, nf=400):
    """N, M (compression positive) of the rectangle BC × HC for strain ε = ε0 + κ·z (arrays over points)."""
    z = ((np.arange(nf) + 0.5) / nf - 0.5) * HC
    eps = np.asarray(eps0)[:, None] + np.asarray(kappa)[:, None] * z[None, :]
    ey = FY / E
    sig = np.where(np.abs(eps) <= ey, E * eps, np.sign(eps) * (FY + Et * (np.abs(eps) - ey)))
    dA = BC * HC / nf
    return sig.sum(axis=1) * dA, (sig * z).sum(axis=1) * dA


def _moment(kappa, N, Et):
    """M(κ) at the axial force N: ε0 by bisection (N(ε0) is increasing)."""
    kappa = np.atleast_1d(np.asarray(kappa, float))
    lo = np.full_like(kappa, -0.2)
    hi = np.full_like(kappa, 0.2)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        n_mid, _ = _section(mid, kappa, Et)
        up = n_mid < N
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return _section(0.5 * (lo + hi), kappa, Et)[1]


def cdc_ultimate(e0: float, Et: float, m: int = 30) -> float:
    """Max N of the pin-ended imperfect column (half-sine bow e0): finite differences on the half length, displacement control of
    the mid-span deflection a (solution followed past the limit point), Newton with the tridiagonal + N column Jacobian."""
    hx = LH / m
    x = np.arange(m + 1) * hx                       # 0 = pinned end, m = mid-span
    w0 = e0 * np.sin(np.pi * x / LC)
    v = np.zeros(m + 1)
    N = 0.5 * _ncr()
    best = 0.0

    def resid(vin, Nv):
        vv = np.concatenate([vin, [vin[-2]]])        # symmetry v[m+1] = v[m−1]
        k = -(vv[2:] - 2 * vv[1:-1] + vv[:-2]) / hx ** 2       # nodes 1..m
        M = _moment(k, Nv, Et)
        return M - Nv * (w0[1:] + vin[1:]), k

    for a in np.linspace(0.05, 12.0, 240):
        v = v * (a / v[-1]) if v[-1] > 0 else a * np.sin(np.pi * x / LC)
        v[0], v[-1] = 0.0, a
        for _ in range(30):
            R, k = resid(v, N)
            # Jacobian: unknowns v_1..v_{m−1}, N
            dk = 1e-9 + 1e-6 * np.abs(k)
            dMdk = (_moment(k + dk, N, Et) - _moment(k - dk, N, Et)) / (2 * dk)
            dN = 1e-6 * max(N, 1.0)
            dMdN = (_moment(k, N + dN, Et) - _moment(k, N - dN, Et)) / (2 * dN)
            J = np.zeros((m, m))
            for i in range(m):                       # equation at node i+1
                node = i + 1
                for nb, c in ((node - 1, 1.0), (node, -2.0), (node + 1, 1.0)):
                    if nb == m + 1:
                        nb = m - 1
                    if 1 <= nb <= m - 1:
                        J[i, nb - 1] += dMdk[i] * (-c / hx ** 2)
                if 1 <= node <= m - 1:
                    J[i, node - 1] -= N
                J[i, m - 1] = dMdN[i] - (w0[node] + v[node])
            dx = np.linalg.solve(J, -R)
            v[1:m] += dx[:m - 1]
            N += dx[m - 1]
            if np.max(np.abs(dx[:m - 1])) < 1e-10 * max(a, 1.0) and abs(dx[m - 1]) < 1e-9 * N:
                break
        best = max(best, N)
        if N < 0.97 * best:
            break
    return best


def bk3_build(wd: Path):
    from fea_core.plasticity import PlasticitySpec
    mat = Material("Ocel S235 bilinearni", E, NU, RHO, Re=FY, Rm=360.0,
                   plastic=PlasticitySpec(model="bilinear", hardening="isotropic", Et=E / 1000).to_dict())
    return _column(wd, "BK3 GMNIA", mat, LC / 1000, 1.05 * AC * FY / FC)


def bk3_eval(res, st):
    info = res.info["nlbuckling"]
    lu = info["result"]["ultimate_factor"]
    Nu_ref = cdc_ultimate(LC / 1000, E / 1000)
    chi_g = lu * FC / (AC * FY)
    chi_c = chi_en(_lbar())
    return [Check("GMNIA ultimate N_u = column deflection curve (1D fibre model, e0 = L/1000)", lu * FC, Nu_ref, 0.03, "N"),
            Check("limit point detected (tangent stiffness ≈ 0 / stopped at the maximum)",
                  float(info["result"]["status"] == "limit_point"), 1.0, 0.0, ""),
            Check("χ_GMNIA ≥ χ EN 1993-1-1 curve c (no residual stresses in the model)", min(chi_g, chi_c) / chi_c, 1.0, 1e-9, "")]


BUCKLE_CASES = {
    "BK1": Case("BK1", "Nonlinear buckling: elastic cantilever with a mode-shaped imperfection (amplification 1/(1 − N/N_cr))",
                bk1_build, bk1_eval),
    "BK2": Case("BK2", "GNIA + first yield of a pin-ended column = EN 1993-1-1 buckling curve c (Ayrton–Perry)",
                bk2_build, bk2_eval),
    "BK3": Case("BK3", "GMNIA of a pin-ended steel column vs the column deflection curve (fibre model)", bk3_build, bk3_eval),
}
