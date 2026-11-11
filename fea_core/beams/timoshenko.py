"""Own solver of frame systems: a spatial Timoshenko beam with an EXACT stiffness matrix (float64, scipy.sparse).

Why not ccx U1: U1 has a shear stiffness 10–60 % higher than κGA, torsional frequencies with an error O(h), it does not know *BUCKLE or the temperature,
and the internal forces from the displacements with 6 valid digits are noisy (F8 |V| 102 instead of 100 N). Here the system is assembled and solved
in Python in double precision, the end forces of the bar are K_e·u_e − f_eq EXACTLY and the diagrams N, V, T, M and the displacements along the bar
are integrated from these forces analytically (polynomials up to degree 5) – the solution does not depend on the subdivision of the bar.

Element (local axes x = axis of the bar, y, z = main axes of the section, DOF u v w θx θy θz at both nodes):
- axial EA/L, torsion GJ/L (J = torsional constant of St. Venant of the section, NOT I_p);
- bending in the plane xy (v, θz; I_z, shear area A_sy = Av_y) and in the plane xz (w, θy; I_y, A_sz = Av_z) independently,
  φ = 12EI/(G·A_s·L²), K = EI/((1+φ)L³)·[[12, 6L, −12, 6L], …] (Timoshenko, exact matrix – 1 element is exact for nodal loads); in the plane xz w' = −θy + V_z/(G·A_s) and the matrix has the opposite sign of the coupling terms;
- shear areas: rectangle 5/6·A, circle 0.9·A, other sections EN 1993-1-1 6.2.6 (Av_y, Av_z); without shear deformation
  (FrameSettings.shear_deformation = False) φ = 0 (Euler–Bernoulli);
- shear centre = centroid (as so far; a section without restrained torsion and warping);
- equivalent nodal loads of a distributed q (trapezium in local and global axes) = ∫ Nᵀ q dx with the shape functions
  of the Timoshenko element (nodally exact), temperature: axial α·ΔT and curvature α·ΔT_g / h;
- hinge (released component of an end) = static condensation of the element (K* = K_rr − K_rc·K_cc⁻¹·K_cr); the same transformation holds
  for the mass and the geometric stiffness;
- support: a constraint = removal of the DOF, a spring (also rotational) = added to the diagonal; reaction = Σ end forces of the elements − nodal loads;
- consistent mass (translation + rotary inertia ρI, torsion ρ(I_y+I_z)) with the shape functions of the same element, natural frequencies
shift-invert eigsh; linear buckling K·φ = −λ·K_G·φ with the geometric stiffness from the axial forces of the reference state
(K_G = ∫N(x)·N_v'ᵀN_v' dx with the Timoshenko functions – Engesser's shear correction automatically).
Sign convention of the forces and moments the same as fem.recover: f_loc = K·u − f_eq is the force acting on the ends of the element, N = −F_ax − ∫q,
V = −F_a − ∫q, M = −M_a + x·(e_x × F_a) + e_x × ∫∫q (My > 0 = tension at the top, σ = My·z/Iy − Mz·y/Iz + N/A).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .fem import FrameError, segment_loads
from .model import FrameNodeLoad, FrameTemperature, DOF_NAMES

PIVOT_TOL = 1e-12                    # ratio of the LU pivot (system with unit diagonal) below this = mechanism


def _gauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1.0) / 2.0, w / 2.0


GAUSS3 = _gauss(3)                   # exact up to a polynomial of degree 5 (loads, K_G)
GAUSS4 = _gauss(4)                   # up to degree 7 (mass)
IDX_XY = np.array([1, 5, 7, 11])     # v, θz (start), v, θz (end)
IDX_XZ = np.array([2, 4, 8, 10])     # w, θy
SGN_XZ = np.array([1.0, -1.0, 1.0, -1.0])      # θy = −θ̂ (θ̂ = w' − γ)


# ------------------------------------------------------------------ section and elements

def shear_areas(sec) -> tuple[float, float]:
    """Shear areas A_sy (shear V_y, bending in the plane xy), A_sz (V_z, plane xz) [mm²]."""
    A = float(sec.A)
    sh = (sec.shape or "").upper()
    if sh == "RECT":
        return 5.0 / 6.0 * A, 5.0 / 6.0 * A
    if sh == "CIRC":
        return 0.9 * A, 0.9 * A
    ay = float(sec.Av_y) if sec.Av_y > 0 else 5.0 / 6.0 * A
    az = float(sec.Av_z) if sec.Av_z > 0 else 5.0 / 6.0 * A
    return ay, az


def section_depth(sec) -> tuple[float, float]:
    """Dimension of the section in the direction of the local axis y and z (from the fibres for the stresses in the main axes) – for the thermal gradient."""
    f = np.asarray(sec.fibers, float) if sec.fibers else np.zeros((0, 2))
    if len(f):
        return float(np.ptp(f[:, 0])), float(np.ptp(f[:, 1]))
    d = sec.dims or {}
    h = float(d.get("h", d.get("d", 0.0)))
    b = float(d.get("b", d.get("d", 0.0)))
    return b, h


@dataclass
class Elems:
    n: int
    a: np.ndarray
    b: np.ndarray
    L: np.ndarray
    R: np.ndarray
    E: np.ndarray
    G: np.ndarray
    rho: np.ndarray
    A: np.ndarray
    Iy: np.ndarray
    Iz: np.ndarray
    J: np.ndarray
    Asy: np.ndarray
    Asz: np.ndarray
    phi_y: np.ndarray                # φ for bending in the plane xy
    phi_z: np.ndarray                # φ for bending in the plane xz
    rel: np.ndarray                  # (n, 12) bool released components
    alpha: np.ndarray                # (n,) nan = material without α
    shear: bool


def elements(fe, shear: bool = True) -> Elems:
    segs = fe.segs
    n = len(segs)
    f = lambda g: np.array([g(s) for s in segs], float)              # noqa: E731
    sa = [shear_areas(s.sec) for s in segs]
    L, E, G = f(lambda s: s.L), f(lambda s: s.E), f(lambda s: s.G)
    Iy, Iz = f(lambda s: s.sec.Iy), f(lambda s: s.sec.Iz)
    Asy, Asz = np.array([x[0] for x in sa], float), np.array([x[1] for x in sa], float)
    if shear:
        phi_y, phi_z = 12 * E * Iz / (G * Asy * L * L), 12 * E * Iy / (G * Asz * L * L)
    else:
        phi_y = phi_z = np.zeros(n)
    rel = np.array([s.rel if s.rel is not None else np.zeros(12, bool) for s in segs], bool).reshape(n, 12)
    return Elems(n, np.array([s.a for s in segs], np.int64), np.array([s.b for s in segs], np.int64), L,
                 np.array([s.R for s in segs], float).reshape(n, 3, 3), E, G, f(lambda s: s.rho),
                 f(lambda s: s.sec.A), Iy, Iz, f(lambda s: 0.0 if s.truss else s.sec.J), Asy, Asz, phi_y, phi_z, rel,
                 np.array([np.nan if s.alpha is None else s.alpha for s in segs], float), shear)


def _bend_block(EI, phi, L, sign):
    c = EI / ((1 + phi) * L ** 3)
    s = sign * 6 * L
    m = np.zeros((len(L), 4, 4))
    m[:, 0, 0], m[:, 0, 1], m[:, 0, 2], m[:, 0, 3] = 12, s, -12, s
    m[:, 1, 1], m[:, 1, 2], m[:, 1, 3] = (4 + phi) * L * L, -s, (2 - phi) * L * L
    m[:, 2, 2], m[:, 2, 3] = 12, -s
    m[:, 3, 3] = (4 + phi) * L * L
    m = m + np.triu(m, 1).transpose(0, 2, 1)
    return m * c[:, None, None]


def local_k(el: Elems) -> np.ndarray:
    """Local stiffness matrix (n, 12, 12) – the exact Timoshenko element."""
    K = np.zeros((el.n, 12, 12))
    ea, gj = el.E * el.A / el.L, el.G * el.J / el.L
    for i, j, v in ((0, 0, ea), (0, 6, -ea), (6, 6, ea), (3, 3, gj), (3, 9, -gj), (9, 9, gj)):
        K[:, i, j] = v
        K[:, j, i] = v
    K[:, IDX_XY[:, None], IDX_XY[None, :]] += _bend_block(el.E * el.Iz, el.phi_y, el.L, +1)
    K[:, IDX_XZ[:, None], IDX_XZ[None, :]] += _bend_block(el.E * el.Iy, el.phi_z, el.L, -1)
    return K


def shapes(xi: np.ndarray, phi: np.ndarray, L: np.ndarray):
    """Shape functions of the Timoshenko element (n, 4, g) for (v_a, θ_a, v_b, θ_b): displacement N_v, its derivative dN_v/dx
    (total slope) and the rotation of the section N_θ (θ = v' − γ).
    """
    x = xi[None, :]
    p = phi[:, None]
    Lc = L[:, None]
    d = 1.0 + p
    Nv = np.stack([(2 * x ** 3 - 3 * x ** 2 - p * x + 1 + p) / d,
                   Lc * (x ** 3 - (2 + p / 2) * x ** 2 + (1 + p / 2) * x) / d,
                   (-2 * x ** 3 + 3 * x ** 2 + p * x) / d,
                   Lc * (x ** 3 - (1 - p / 2) * x ** 2 - (p / 2) * x) / d], axis=1)
    dNv = np.stack([(6 * x ** 2 - 6 * x - p) / d / Lc,
                    (3 * x ** 2 - (4 + p) * x + 1 + p / 2) / d,
                    (-6 * x ** 2 + 6 * x + p) / d / Lc,
                    (3 * x ** 2 - (2 - p) * x - p / 2) / d], axis=1)
    Nth = np.stack([6 * (x ** 2 - x) / (Lc * d),
                    (3 * x ** 2 - (4 + p) * x + 1 + p) / d,
                    -6 * (x ** 2 - x) / (Lc * d),
                    (3 * x ** 2 - (2 - p) * x) / d], axis=1)
    return Nv, dNv, Nth


def local_mass(el: Elems) -> np.ndarray:
    """Consistent mass matrix (n, 12, 12): translation ρA, rotary inertia ρI of the bending, torsion ρ(I_y + I_z)."""
    xi, w = GAUSS4
    M = np.zeros((el.n, 12, 12))
    am = el.rho * el.A * el.L / 6.0
    for i, j, v in ((0, 0, 2), (0, 6, 1), (6, 6, 2)):
        M[:, i, j] = am * v
        M[:, j, i] = am * v
    tm = el.rho * (el.Iy + el.Iz) * el.L / 6.0
    for i, j, v in ((3, 3, 2), (3, 9, 1), (9, 9, 2)):
        M[:, i, j] = tm * v
        M[:, j, i] = tm * v
    for idx, phi, I, sgn in ((IDX_XY, el.phi_y, el.Iz, np.ones(4)), (IDX_XZ, el.phi_z, el.Iy, SGN_XZ)):
        Nv, _, Nth = shapes(xi, phi, el.L)
        m = (el.rho * el.A * el.L)[:, None, None] * ((Nv * w) @ Nv.transpose(0, 2, 1)) \
            + (el.rho * I * el.L)[:, None, None] * ((Nth * w) @ Nth.transpose(0, 2, 1))
        M[:, idx[:, None], idx[None, :]] += m * (sgn[:, None] * sgn[None, :])
    return M


def local_kg(el: Elems, N_gauss: np.ndarray, torsion: bool) -> np.ndarray:
    """Geometric stiffness (n, 12, 12) from the axial force at the Gauss points GAUSS3 (n, 3) (tension +): K_G = ∫N(x)·N_v'ᵀN_v' dx
    in both planes (total slope with the functions of the Timoshenko element) and (optionally) the torsion term N̄·(I_y + I_z)/(A·L).
    """
    xi, w = GAUSS3
    K = np.zeros((el.n, 12, 12))
    for idx, phi, sgn in ((IDX_XY, el.phi_y, np.ones(4)), (IDX_XZ, el.phi_z, SGN_XZ)):
        _, dNv, _ = shapes(xi, phi, el.L)
        m = el.L[:, None, None] * ((dNv * (N_gauss * w)[:, None, :]) @ dNv.transpose(0, 2, 1))
        K[:, idx[:, None], idx[None, :]] += m * (sgn[:, None] * sgn[None, :])
    if torsion:
        nbar = N_gauss @ w
        kt = nbar * (el.Iy + el.Iz) / (el.A * el.L)
        for i, j, v in ((3, 3, 1), (3, 9, -1), (9, 3, -1), (9, 9, 1)):
            K[:, i, j] += kt * v
    return K


def eq_loads(el: Elems, qa: np.ndarray, qb: np.ndarray, eps_t: np.ndarray, kz: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Equivalent nodal loads of the element in local axes (n, 12): trapezoidal q (qa → qb, local vectors (n, 3)),
    axial strain by the temperature eps_t = α·ΔT and thermal curvatures kz (θy' in the direction of z), ky (θz').
    """
    xi, w = GAUSS3
    q = qa[:, None, :] + (qb - qa)[:, None, :] * xi[None, :, None]               # (n, g, 3)
    f = np.zeros((el.n, 12))
    L = el.L
    f[:, 0] = L * (q[:, :, 0] * ((1 - xi) * w)[None, :]).sum(1)
    f[:, 6] = L * (q[:, :, 0] * (xi * w)[None, :]).sum(1)
    for idx, phi, comp, sgn in ((IDX_XY, el.phi_y, 1, np.ones(4)), (IDX_XZ, el.phi_z, 2, SGN_XZ)):
        Nv, _, _ = shapes(xi, phi, L)
        f[:, idx] = L[:, None] * np.einsum("nig,g,ng->ni", Nv, w, q[:, :, comp]) * sgn[None, :]
    ea = el.E * el.A * eps_t
    f[:, 0] -= ea
    f[:, 6] += ea
    f[:, 4] -= el.E * el.Iy * kz
    f[:, 10] += el.E * el.Iy * kz
    f[:, 5] -= el.E * el.Iz * ky
    f[:, 11] += el.E * el.Iz * ky
    return f


def condensation(K: np.ndarray, rel: np.ndarray, who=None):
    """Static condensation of the released components: matrix P (n, 12, 12) with u_full = P·u_ret + u_p and the function u_p(f)."""
    n = len(K)
    P = np.tile(np.eye(12), (n, 1, 1))
    groups = []
    has = rel.any(axis=1)
    if has.any():
        pats, inv = np.unique(rel[has], axis=0, return_inverse=True)
        which = np.flatnonzero(has)
        for k, pat in enumerate(pats):
            idx = which[inv.ravel() == k]
            c = np.flatnonzero(pat)
            r = np.flatnonzero(~pat)
            Kcc = K[idx][:, c][:, :, c]
            Kcr = K[idx][:, c][:, :, r]
            try:
                X = np.linalg.solve(Kcc, Kcr)
                Kcc_inv_ok = True
            except np.linalg.LinAlgError:
                Kcc_inv_ok = False
            if not Kcc_inv_ok or not np.all(np.isfinite(X)) or \
                    np.linalg.cond(Kcc).max() > 1e14:
                name = who(idx[0]) if who else "?"
                raise FrameError(f"Member '{name}': the released components of the member ends exclude each other (the member would move freely – e.g. torsion released at both ends or axial force at both ends)")
            Pi = np.zeros((len(idx), 12, 12))
            Pi[:, r, r] = 1.0
            Pi[:, c[:, None], r[None, :]] = -X
            P[idx] = Pi
            groups.append((idx, c, Kcc))
    return P, groups


def _clean(Ke: np.ndarray) -> np.ndarray:
    """Rounding noise after the condensation (1e-16 · stiffness) → exact zero (inactive DOF)."""
    scale = np.abs(Ke).max(axis=(1, 2), keepdims=True)
    return np.where(np.abs(Ke) < 1e-12 * scale, 0.0, Ke)


# ------------------------------------------------------------------ transformation and assembly

def _blocks(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Global vectors (n, 12) → local (R rows = local axes)."""
    return np.einsum("nkp,nip->nik", R, v.reshape(-1, 4, 3)).reshape(-1, 12)


def _blocks_t(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Local vectors (n, 12) → global."""
    return np.einsum("nkp,nik->nip", R, v.reshape(-1, 4, 3)).reshape(-1, 12)


def to_global(R: np.ndarray, K: np.ndarray) -> np.ndarray:
    Kb = K.reshape(-1, 4, 3, 4, 3)
    tmp = np.einsum("nikjl,nlq->nikjq", Kb, R)
    return np.einsum("nkp,nikjq->nipjq", R, tmp).reshape(-1, 12, 12)


def element_dofs(el: Elems) -> np.ndarray:
    return np.concatenate([6 * el.a[:, None] + np.arange(6), 6 * el.b[:, None] + np.arange(6)], axis=1)


def assemble(dofs: np.ndarray, Kg: np.ndarray, ndof: int) -> sp.csr_matrix:
    n = len(dofs)
    rows = np.broadcast_to(dofs[:, :, None], (n, 12, 12)).ravel()
    cols = np.broadcast_to(dofs[:, None, :], (n, 12, 12)).ravel()
    return sp.coo_matrix((Kg.ravel(), (rows, cols)), shape=(ndof, ndof)).tocsr()


# ------------------------------------------------------------------ model

class TimModel:
    """Assembled system (stiffness matrix, constraints, active DOF, factorisation) – one per study and subdivision."""

    def __init__(self, study, fe, torsion_in_kg: bool | None = None, shear: bool | None = None):
        self.study, self.fe = study, fe
        fr = study.frame
        self.shear = bool(fr.settings.shear_deformation) if shear is None else bool(shear)
        self.torsion_kg = fr.settings.buckling_torsion if torsion_in_kg is None else torsion_in_kg
        if not fe.segs:
            raise FrameError("The beam structure has no members")
        self.el = el = elements(fe, self.shear)
        self.nf = nf = len(fe.xyz)
        self.ndof = 6 * nf
        self.K0 = local_k(el)
        who = lambda i: fe.segs[i].member                                  # noqa: E731
        self.P, self.cond = condensation(self.K0, el.rel, who)
        self.Ke = _clean(self.P.transpose(0, 2, 1) @ self.K0 @ self.P) if self.cond else self.K0
        self.dofs = element_dofs(el)
        self.K = assemble(self.dofs, to_global(el.R, self.Ke), self.ndof)
        ks = np.zeros(self.ndof)
        for n_, d, k in fe.springs:
            ks[6 * n_ + d - 1] += k
        self.ks = ks
        self.Kt = (self.K + sp.diags(ks)).tocsr()
        self._spin_penalty()
        # DOF activity: the node belongs to a bar; a rotation without any stiffness (all bars in the hinge) holds nothing
        connected = np.zeros(nf, bool)
        connected[el.a] = True
        connected[el.b] = True
        self.connected = connected
        diag = self.Kt.diagonal()
        rot = (np.arange(self.ndof) % 6) >= 3
        active = np.repeat(connected, 6) & ~(rot & (diag == 0.0))
        fixed = np.zeros(self.ndof, bool)
        for n_, d in fe.bc:
            fixed[6 * n_ + d - 1] = True
        self.fixed = fixed & active
        self.active = active
        self.free = np.flatnonzero(active & ~fixed)
        zero = self.free[diag[self.free] == 0.0]
        if len(zero):
            n_, d = divmod(int(zero[0]), 6)
            name = next((k for k, v in fe.joint.items() if v == n_), f"internal node of member no. {n_}")
            raise FrameError(f"Node '{name}': component {DOF_NAMES[d]} is not restrained at all (mechanism) – add a support or members")
        self._lu = None
        self.tension_scale = float((el.E * el.A).max())          # E·A of the largest bar [N]
        self.member_of = [s.member for s in fe.segs]

    def _spin_penalty(self) -> None:
        """Internal nodes of bars without torsional stiffness (trusses, torsion released; J = 0): the rotation about the axis of the bar in them
        has no resistance (a zero mode) and is neither loaded nor constrained → it is stiffened by a stiff spring only in the direction of the axis (e⊗e),
        the solution does not change; the natural frequencies of such auxiliary DOF lie far above the interesting range.
        """
        fe, el = self.fe, self.el
        nodes, axes = [], []
        for segs in fe.member_segs.values():
            if not fe.segs[segs[0]].truss:
                continue
            for j in range(len(segs) - 1):
                for si in (segs[j], segs[j + 1]):
                    nodes.append(fe.segs[segs[j]].b)
                    axes.append(el.R[si, 0])
        if not nodes:
            return
        diag = self.Kt.diagonal()
        rotd = diag.reshape(-1, 6)[:, 3:]
        eps = 1e3 * float(rotd.max()) if rotd.max() > 0 else 1.0
        nodes, axes = np.array(nodes), np.array(axes)
        blk = eps * axes[:, :, None] * axes[:, None, :]                       # (m, 3, 3)
        dofs = 6 * nodes[:, None] + 3 + np.arange(3)[None, :]
        rows = np.broadcast_to(dofs[:, :, None], blk.shape).ravel()
        cols = np.broadcast_to(dofs[:, None, :], blk.shape).ravel()
        self.Kt = (self.Kt + sp.coo_matrix((blk.ravel(), (rows, cols)), shape=self.Kt.shape)).tocsr()

    # --- solution
    def _factor(self):
        if self._lu is not None:
            return
        fr = self.free
        if len(fr) == 0:
            raise FrameError("The structure has no free degree of freedom")
        Kff = self.Kt[fr][:, fr].tocsc()
        d = Kff.diagonal()
        D = 1.0 / np.sqrt(d)
        Ks = (sp.diags(D) @ Kff @ sp.diags(D)).tocsc()
        try:
            lu = spla.splu(Ks)
        except RuntimeError:
            self._singular(Ks, None)
        piv = np.abs(lu.U.diagonal())
        if piv.min() < PIVOT_TOL:
            self._singular(Ks, lu)
        self._lu, self._D, self._Kff = lu, D, Kff

    def _singular(self, Ks, lu):
        msg = "The structure is unstable (mechanism) or insufficiently supported"
        dof = None
        try:                                           # a zero eigenvector of the system = motion without resistance
            w, v = spla.eigsh(Ks, k=1, sigma=-1e-4, which="LM")
            dof = int(self.free[int(np.argmax(np.abs(v[:, 0])))])
        except Exception:  # noqa: BLE001 – backup: zero LU pivot
            if lu is not None:
                dof = int(self.free[lu.perm_c[int(np.argmin(np.abs(lu.U.diagonal())))]])
        if dof is not None:
            n_, d = divmod(dof, 6)
            name = next((kk for kk, v_ in self.fe.joint.items() if v_ == n_), f"internal node of member no. {n_}")
            msg += f" – largest free motion at node '{name}', component {DOF_NAMES[d]}"
        raise FrameError(msg + ". Check the supports (at least 6 constrained components) and the joints (a node with truss members only needs three non-coplanar members or a support).")

    def solve_free(self, Ff: np.ndarray) -> np.ndarray:
        self._factor()
        D = self._D[:, None] if Ff.ndim == 2 else self._D
        u = D * self._lu.solve(D * Ff)
        r = Ff - self._Kff @ u                                              # one refinement step
        return u + D * self._lu.solve(D * r)

    # --- loads
    def case_loads(self, lc):
        """(nodal loads (nf, 6), local equivalent loads of the elements (n, 12), (qa, qb) locally, temperature (eps, kz, ky))."""
        fe, el, study = self.fe, self.el, self.study
        Fn = np.zeros((self.nf, 6))
        for l in lc.loads:
            if isinstance(l, FrameNodeLoad):
                if l.node not in fe.joint:
                    raise FrameError(f"Node load: node '{l.node}' does not exist")
                j = fe.joint[l.node]
                if not self.connected[j]:
                    raise FrameError(f"Load on node '{l.node}': the node does not belong to any member")
                Fn[j, :3] += np.asarray(l.force, float)
                Fn[j, 3:] += np.asarray(l.moment, float)
        qa = np.zeros((el.n, 3))
        qb = np.zeros((el.n, 3))
        for i, (a_, b_) in segment_loads(study, fe, lc).items():
            qa[i], qb[i] = a_, b_
        qa_l = np.einsum("nij,nj->ni", el.R, qa)
        qb_l = np.einsum("nij,nj->ni", el.R, qb)
        eps = np.zeros(el.n)
        kz = np.zeros(el.n)
        ky = np.zeros(el.n)
        members = fe.member_segs
        for l in lc.loads:
            if not isinstance(l, FrameTemperature):
                continue
            names = list(l.members) or list(members)
            for mn in names:
                if mn not in members:
                    raise FrameError(f"Temperature load: member '{mn}' does not exist")
                for i in members[mn]:
                    sg = fe.segs[i]
                    if not (l.dT or l.grad_z or l.grad_y):
                        continue
                    if sg.alpha is None or not sg.alpha > 0:
                        raise FrameError(f"Temperature load: the material of member '{mn}' has no coefficient of thermal expansion α")
                    eps[i] += sg.alpha * l.dT
                    if l.grad_z or l.grad_y:
                        hy, hz = section_depth(sg.sec)
                        if (l.grad_z and hz <= 0) or (l.grad_y and hy <= 0):
                            raise FrameError(f"Temperature gradient: member '{mn}' has no known section dimension")
                        if l.grad_z:
                            kz[i] += sg.alpha * l.grad_z / hz              # θy' = α·ΔT_z / h_z
                        if l.grad_y:
                            ky[i] += -sg.alpha * l.grad_y / hy             # θz' = −α·ΔT_y / h_y
        f_eq = eq_loads(el, qa_l, qb_l, eps, kz, ky)
        return Fn, f_eq, (qa_l, qb_l), (eps, kz, ky)

    def reduce_loads(self, f_eq: np.ndarray):
        """Loads reduced by the condensation (f* = Pᵀ(f − K·u_p)) and the partial solution u_p of the released components."""
        up = np.zeros_like(f_eq)
        for idx, c, Kcc in self.cond:
            up[idx[:, None], c[None, :]] = np.linalg.solve(Kcc, f_eq[idx][:, c][..., None])[..., 0]
        if not self.cond:
            return f_eq, up
        fred = np.einsum("nji,nj->ni", self.P, f_eq - np.einsum("nij,nj->ni", self.K0, up))
        return fred, up

    def scatter(self, v_loc: np.ndarray) -> np.ndarray:
        """Local vectors of the elements (n, 12) → global vector of the system (ndof,) – sum of the contributions of the nodes."""
        return np.bincount(self.dofs.ravel(), weights=_blocks_t(self.el.R, v_loc).ravel(), minlength=self.ndof)

    def solve_cases(self, cases) -> list[dict]:
        """Solution of the load states (one factorisation) → list of dictionaries with U (nf, 6), f_loc, u_full, the load."""
        el = self.el
        loads = []
        rhs = []
        for lc in cases:
            Fn, f_eq, ql, th = self.case_loads(lc)
            fred, up = self.reduce_loads(f_eq)
            F = self.scatter(fred) + Fn.ravel()
            bad = np.flatnonzero((F != 0.0) & ~self.active)
            if len(bad):
                n_, d = divmod(int(bad[0]), 6)
                raise FrameError("Moment applied to a node where all members are hinged (node rotation is not in the system)"
                                 if d >= 3 else f"Load on node #{n_}: component {DOF_NAMES[d]} does not belong to the structure")
            loads.append((Fn, f_eq, ql, th, up))
            rhs.append(F)
        Fm = np.stack(rhs, axis=1)
        Uf = self.solve_free(Fm[self.free]) if len(self.free) else np.zeros((0, Fm.shape[1]))
        out = []
        for k, (Fn, f_eq, ql, th, up) in enumerate(loads):
            u = np.zeros(self.ndof)
            u[self.free] = Uf[:, k]
            ue = _blocks(el.R, u[self.dofs])
            u_full = np.einsum("nij,nj->ni", self.P, ue) + up
            f_loc = np.einsum("nij,nj->ni", self.K0, u_full) - f_eq
            out.append({"u": u, "U": u.reshape(self.nf, 6), "u_full": u_full, "f_loc": f_loc, "f_eq": f_eq, "Fn": Fn,
                        "q": ql, "thermal": th, "F": Fm[:, k]})
        return out

    # --- reactions and equilibrium
    def reactions(self, res: dict):
        """R = Σ end forces of the elements − nodal loads (n_nodes, 6): the force that the constraint / spring gives to the node."""
        acc = np.bincount(self.dofs.ravel(), weights=_blocks_t(self.el.R, res["f_loc"]).ravel(), minlength=self.ndof)
        return acc.reshape(self.nf, 6) - res["Fn"]

    def resultants(self, res: dict, c0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(force of the loads, moment of the loads about c0, force of the reactions, moment of the reactions about c0) – the loads EXACTLY (the distributed q integrated
        analytically, not consistent nodal forces); vectorised (thousands of bars).
        """
        el, X = self.el, np.array(self.fe.xyz)
        c0 = np.asarray(c0, float)
        Fn = res["Fn"]
        Fa = Fn[:, :3].sum(axis=0)
        Ma = (Fn[:, 3:] + np.cross(X - c0, Fn[:, :3])).sum(axis=0)
        qa = np.einsum("nji,nj->ni", el.R, res["q"][0])                         # local → global
        qb = np.einsum("nji,nj->ni", el.R, res["q"][1])
        Fs = el.L[:, None] * (qa + qb) / 2
        Fa = Fa + Fs.sum(axis=0)
        Ma = Ma + (np.cross(X[el.a] - c0, Fs) + np.cross(el.R[:, 0], (el.L ** 2)[:, None] * (qa + 2 * qb) / 6)).sum(axis=0)
        held = sorted(self.held_nodes())
        R = self.reactions(res)[held]
        Rf = R[:, :3].sum(axis=0)
        Rm = (R[:, 3:] + np.cross(X[held] - c0, R[:, :3])).sum(axis=0)
        return Fa, Ma, Rf, Rm

    def held_nodes(self) -> set[int]:
        held = {n_ for n_, _ in self.fe.bc} | {n_ for n_, _, _ in self.fe.springs}
        return {n_ for n_ in held if self.connected[n_]}


# ------------------------------------------------------------------ shape functions and displacements along the member

def _roots_v(Fa: np.ndarray, q0: np.ndarray, q1: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Positions x∈(0, L) where V(x) = −Fa − q0·x − q1·x²/2 = 0 (extreme of the moment): (n, 2), nan outside the interval."""
    a, b, c = q1 / 2.0, q0, Fa
    out = np.full((len(L), 2), np.nan)
    quad = np.abs(a) * L > 1e-9 * np.abs(b) + 1e-300
    quad &= np.abs(a) > 0
    with np.errstate(all="ignore"):
        disc = b * b - 4 * a * c
        sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
        qq = -0.5 * (b + np.sign(b + (b == 0)) * sq)
        r1 = np.where(quad, qq / a, np.where(np.abs(b) > 0, -c / b, np.nan))
        r2 = np.where(quad, np.where(qq != 0, c / qq, np.nan), np.nan)
    for k, r in enumerate((r1, r2)):
        ok = np.isfinite(r) & (r > 1e-6 * L) & (r < (1 - 1e-6) * L)
        out[:, k] = np.where(ok, r, np.nan)
    return out


def fields_at(model: TimModel, res: dict, X: np.ndarray, idx: np.ndarray | None = None) -> dict:
    """Internal forces (local axes of the element) and displacements at the own points of the elements: X (m, K) = distance from the start of the element [mm],
    idx (m,) = numbers of the elements (None = all elements, m = n). All EXACTLY from the end forces of the element by integrating polynomials (q linear
    → N, V quadratic, M cubic, displacements up to degree 5):
    N = −F_x − ∫q_x, V = −F − ∫q, T = −M_x, M_y = −M_ay − x·F_z − ∫∫q_z, M_z = −M_az + x·F_y + ∫∫q_y,
    θ' = M/EI (+ thermal curvature), v' = θ + V/(G·A_s) (plane xz: w' = −θ + V_z/(G·A_s)); origin = displacements of the node a.
    """
    el = model.el
    g = (lambda a: a) if idx is None else (lambda a: a[idx])
    L = g(el.L)
    f, ue = g(res["f_loc"]), g(res["u_full"])
    qa_l, qb_l = g(res["q"][0]), g(res["q"][1])
    eps, kz, ky = (g(a) for a in res["thermal"])
    q0 = qa_l
    q1 = (qb_l - qa_l) / L[:, None]
    Fa, Ma = f[:, 0:3], f[:, 3:6]
    ax = lambda v: v[:, None]                                                   # noqa: E731
    N = -ax(Fa[:, 0]) - ax(q0[:, 0]) * X - ax(q1[:, 0]) * X ** 2 / 2
    Vy = -ax(Fa[:, 1]) - ax(q0[:, 1]) * X - ax(q1[:, 1]) * X ** 2 / 2
    Vz = -ax(Fa[:, 2]) - ax(q0[:, 2]) * X - ax(q1[:, 2]) * X ** 2 / 2
    T = -ax(Ma[:, 0]) + 0 * X
    My = -ax(Ma[:, 1]) - ax(Fa[:, 2]) * X - ax(q0[:, 2]) * X ** 2 / 2 - ax(q1[:, 2]) * X ** 3 / 6
    Mz = -ax(Ma[:, 2]) + ax(Fa[:, 1]) * X + ax(q0[:, 1]) * X ** 2 / 2 + ax(q1[:, 1]) * X ** 3 / 6
    iea = 1.0 / (g(el.E) * g(el.A))
    ieiz, ieiy = 1.0 / (g(el.E) * g(el.Iz)), 1.0 / (g(el.E) * g(el.Iy))
    igy = 1.0 / (g(el.G) * g(el.Asy)) if model.shear else np.zeros(len(L))
    igz = 1.0 / (g(el.G) * g(el.Asz)) if model.shear else np.zeros(len(L))
    u_ = ax(ue[:, 0]) + ax(iea) * (-ax(Fa[:, 0]) * X - ax(q0[:, 0]) * X ** 2 / 2 - ax(q1[:, 0]) * X ** 3 / 6) \
        + ax(eps) * X
    v_ = ax(ue[:, 1]) + ax(ue[:, 5]) * X \
        + ax(ieiz) * (-ax(Ma[:, 2]) * X ** 2 / 2 + ax(Fa[:, 1]) * X ** 3 / 6 + ax(q0[:, 1]) * X ** 4 / 24
                      + ax(q1[:, 1]) * X ** 5 / 120) \
        + ax(igy) * (-ax(Fa[:, 1]) * X - ax(q0[:, 1]) * X ** 2 / 2 - ax(q1[:, 1]) * X ** 3 / 6) + ax(ky) * X ** 2 / 2
    w_ = ax(ue[:, 2]) - ax(ue[:, 4]) * X \
        - ax(ieiy) * (-ax(Ma[:, 1]) * X ** 2 / 2 - ax(Fa[:, 2]) * X ** 3 / 6 - ax(q0[:, 2]) * X ** 4 / 24
                      - ax(q1[:, 2]) * X ** 5 / 120) \
        + ax(igz) * (-ax(Fa[:, 2]) * X - ax(q0[:, 2]) * X ** 2 / 2 - ax(q1[:, 2]) * X ** 3 / 6) - ax(kz) * X ** 2 / 2
    uloc = np.stack([u_, v_, w_], axis=2)                                       # (m, K, 3)
    ug = np.einsum("nkp,nik->nip", g(el.R), uloc)                               # globally
    return {"N": N, "Vy": Vy, "Vz": Vz, "T": T, "My": My, "Mz": Mz, "uloc": uloc, "ug": ug}


def member_results(model: TimModel, res: dict, n_base: int = 21) -> dict:
    """Diagrams N, V_y, V_z, T, M_y, M_z, displacements (global) and stresses along the bars – exactly from the end forces of the elements.
    Returns {bar: {s, N, Vy, Vz, T, My, Mz, points, u, sigma_max, tau_est, sigma_eq, section}}.
    """
    el, fe = model.el, model.fe
    n = el.n
    L = el.L
    f = res["f_loc"]
    qa_l, qb_l = res["q"]
    eps, kz, ky = res["thermal"]
    q0 = qa_l
    q1 = (qb_l - qa_l) / L[:, None]
    Fa = f[:, 0:3]
    # positions of the output points: uniformly + extremes of the moment (V = 0)
    nm = max(len(fe.member_segs), 1)
    per = n / nm
    if nm > 600:                                       # large systems: fewer points per bar (the diagrams are piecewise polynomials,
        n_base = min(n_base, 7)                        # the extremes of the moment are always at the points V = 0; saving memory and JSON)
    elif nm > 150:
        n_base = min(n_base, 11)
    K = n_base if per <= 1.5 else int(min(n_base, max(3, math.ceil((n_base - 1) / per) + 1)))
    base = np.tile(np.linspace(0.0, 1.0, K), (n, 1))
    ex = np.concatenate([_roots_v(Fa[:, 1], q0[:, 1], q1[:, 1], L), _roots_v(Fa[:, 2], q0[:, 2], q1[:, 2], L)], axis=1)
    tt = np.concatenate([base, np.where(np.isfinite(ex), ex / L[:, None], 2.0)], axis=1)
    tt.sort(axis=1)
    valid = tt <= 1.0
    t = np.where(valid, tt, 1.0)
    x = t * L[:, None]
    X = x
    fv = fields_at(model, res, X)
    N, Vy, Vz, T, My, Mz, ug = (fv[k] for k in ("N", "Vy", "Vz", "T", "My", "Mz", "ug"))
    Xn = np.array(fe.xyz)
    pts = Xn[el.a][:, None, :] + t[:, :, None] * (Xn[el.b] - Xn[el.a])[:, None, :]
    sel = valid.ravel()
    cnt = valid.sum(axis=1)
    off = np.concatenate([[0], np.cumsum(cnt)])
    flat = lambda a_: a_.reshape(n * a_.shape[1], *a_.shape[2:])[sel]          # noqa: E731
    s0 = np.array([sg.s0 for sg in fe.segs])
    S = flat(s0[:, None] + x)
    fN, fVy, fVz, fT, fMy, fMz = (flat(a_) for a_ in (N, Vy, Vz, T, My, Mz))
    fP, fU = flat(pts), flat(ug)
    # stresses by the section
    sec_id = {}
    sid = np.array([sec_id.setdefault(id(sg.sec), len(sec_id)) for sg in fe.segs])
    secs = {v: next(sg.sec for sg in fe.segs if id(sg.sec) == k) for k, v in sec_id.items()}
    sample_sec = np.repeat(sid, cnt)
    sig = np.zeros(len(S))
    tau = np.zeros(len(S))
    for k, sec in secs.items():
        m = sample_sec == k
        fib = sec.fibers or [(0.0, 0.0)]
        sg_ = np.zeros(int(m.sum()))
        for (fy, fz) in fib:
            sg_ = np.maximum(sg_, np.abs(fN[m] / sec.A + fMy[m] * fz / sec.Iy - fMz[m] * fy / sec.Iz))
        sig[m] = sg_
        ta = np.zeros(int(m.sum()))
        if sec.Av_y > 0:
            ta += np.abs(fVy[m]) / sec.Av_y
        if sec.Av_z > 0:
            ta = np.maximum(ta, np.abs(fVz[m]) / sec.Av_z)
        if sec.Wt > 0:
            ta += np.abs(fT[m]) / sec.Wt
        tau[m] = ta
    seq = np.sqrt(sig ** 2 + 3 * tau ** 2)
    L_ = {k: v.tolist() for k, v in (("s", S), ("N", fN), ("Vy", fVy), ("Vz", fVz), ("T", fT), ("My", fMy), ("Mz", fMz),
                                     ("points", fP), ("u", fU), ("sigma_max", sig), ("tau_est", tau), ("sigma_eq", seq))}
    members = {}
    for mname, segs in fe.member_segs.items():
        i0, i1 = off[segs[0]], off[segs[-1] + 1]
        d = {k: v[i0:i1] for k, v in L_.items()}
        d["section"] = fe.segs[segs[0]].sec.name
        members[mname] = d
    return members


def step_record(model: TimModel, res: dict, n_base: int = 21) -> dict:
    """Record `StepResult.frame`: bars (diagrams), reactions of the supports, displacements of the nodes, residual imbalance of the nodes."""
    fe = model.fe
    R = model.reactions(res)
    held = model.held_nodes()
    react = {name: R[j].tolist() for name, j in fe.joint.items() if j in held}
    U = res["U"]
    # residual of the equilibrium in the free DOF (the spring is counted in R as a constraint → add k·u)
    Rf = (R.ravel() + model.ks * res["u"])[model.free]
    trans = (model.free % 6) < 3
    resid = float(np.abs(Rf[trans]).max(initial=0.0))
    scale = max(float(np.abs(res["Fn"][:, :3]).sum()), float(sum(np.abs(np.array(v[:3])).sum() for v in react.values())),
                1e-30)
    members = member_results(model, res, n_base)
    return {"members": members, "reactions": react,
            "nodes": {name: U[j].tolist() for name, j in fe.joint.items()},
            "node_residual": resid / scale, "coupling_forces": {}}


# ------------------------------------------------------------------ natural frequencies and linear buckling

DENSE_MAX = 900                      # up to this number of free DOF the full (dense) eigenvalue problem


def _shapes_u(model: TimModel, phi: np.ndarray) -> np.ndarray:
    """Eigenvectors (ndof, k) → shapes of the displacements of the nodes (k, nf, 3) float32 normalised to max |u| = 1."""
    k = phi.shape[1]
    out = np.zeros((k, model.nf, 3), np.float32)
    for j in range(k):
        u = phi[:, j].reshape(model.nf, 3 if False else 6)[:, :3]
        m = float(np.linalg.norm(u, axis=1).max())
        out[j] = u / m if m > 0 else u
    return out


def modal(model: TimModel, nmodes: int, f_min: float = 0.0, f_max: float | None = None) -> dict:
    """Natural frequencies: K·φ = ω²·M·φ with the consistent mass. Returns {freq [Hz], shapes (k, nf, 3), participation (k, 6),
    eff_mass (k, 6) [t; rotations t·mm²], total_mass (6)}; rotations about the origin of the coordinates.
    """
    el = model.el
    if np.any(el.rho <= 0):
        raise FrameError("Natural frequencies: the member material has no density ρ")
    M0 = local_mass(el)
    Me = model.P.transpose(0, 2, 1) @ M0 @ model.P if model.cond else M0
    M = assemble(model.dofs, to_global(el.R, Me), model.ndof)
    fr = model.free
    Kff = model.Kt[fr][:, fr].tocsc()
    Mff = M[fr][:, fr].tocsc()
    n = len(fr)
    k = int(min(max(nmodes, 1), n - 1))
    f_lo = max(float(f_min or 0.0), 0.0)
    if n <= DENSE_MAX:
        import scipy.linalg as sla
        lam, V = sla.eigh(Kff.toarray(), Mff.toarray())
    else:
        extra = 6 if f_lo > 0 else 0
        scale = float(np.median(Kff.diagonal() / Mff.diagonal()))
        sigma = (2 * math.pi * f_lo) ** 2 if f_lo > 0 else -1e-6 * scale
        lam, V = spla.eigsh(Kff, k=min(k + extra, n - 2), M=Mff, sigma=sigma, which="LM")
    order = np.argsort(lam)
    lam, V = lam[order], V[:, order]
    f = np.sqrt(np.maximum(lam, 0.0)) / (2 * math.pi)
    keep = f >= f_lo - 1e-9
    if f_max:
        keep &= f <= float(f_max)
    idx = np.flatnonzero(keep)[:k]
    if not len(idx):
        raise FrameError("Natural frequencies: there is no mode in the given range")
    lam, f, V = lam[idx], f[idx], V[:, idx]
    phi = np.zeros((model.ndof, len(idx)))
    phi[fr] = V
    # rigid motions (translation 3 + rotation about the origin) for the participation and the effective mass
    X = np.array(model.fe.xyz)
    r = np.zeros((model.ndof, 6))
    for d in range(3):
        r[d::6, d] = 1.0
        r[3 + d::6, 3 + d] = 1.0
    e = np.eye(3)
    for d in range(3):
        rot = np.cross(e[d], X)
        r[0::6, 3 + d], r[1::6, 3 + d], r[2::6, 3 + d] = rot[:, 0], rot[:, 1], rot[:, 2]
    r[~model.active] = 0.0
    gam = phi.T @ (M @ r)
    total = np.einsum("ij,ij->j", r, M @ r)
    return {"freq": f.tolist(), "shapes": _shapes_u(model, phi), "participation": gam.tolist(),
            "eff_mass": (gam ** 2).tolist(), "total_mass": total.tolist()}


def buckling(model: TimModel, lc, nmodes: int) -> dict:
    """Linear buckling: (K + λ·K_G(N_ref))·φ = 0, N_ref = axial forces of the load state lc (compression negative). Returns
    {factors λ (ascending, only positive), shapes (k, nf, 3), case}.
    """
    el = model.el
    res = model.solve_cases([lc])[0]
    f, q0 = res["f_loc"], res["q"][0]
    q1 = (res["q"][1] - res["q"][0]) / el.L[:, None]
    xi, _ = GAUSS3
    x = xi[None, :] * el.L[:, None]
    Ng = -f[:, [0]] - q0[:, [0]] * x - q1[:, [0]] * x ** 2 / 2                   # axial force at the Gauss points (tension +)
    Kg = local_kg(el, Ng, model.torsion_kg)
    Kge = model.P.transpose(0, 2, 1) @ Kg @ model.P if model.cond else Kg
    Gm = assemble(model.dofs, to_global(el.R, -Kge), model.ndof)
    fr = model.free
    Kff = model.Kt[fr][:, fr].tocsc()
    Gff = Gm[fr][:, fr].tocsc()
    n = len(fr)
    k = int(min(max(nmodes, 1), n - 2))
    if n <= DENSE_MAX:
        import scipy.linalg as sla
        mu, V = sla.eigh(Gff.toarray(), Kff.toarray())
    else:
        mu, V = spla.eigsh(Gff, k=k + 2, M=Kff, which="LA")
    pos = np.flatnonzero(mu > 1e-14 * max(float(np.abs(mu).max()), 1e-300))
    if not len(pos):
        raise FrameError("Buckling: the reference state has no compression in the members (tensile axial force or bending only) – no critical load exists")
    pos = pos[np.argsort(-mu[pos])][:k]                # largest μ = smallest λ
    lam = 1.0 / mu[pos]
    phi = np.zeros((model.ndof, len(pos)))
    phi[fr] = V[:, pos]
    return {"factors": lam.tolist(), "shapes": _shapes_u(model, phi), "case": lc.name}
