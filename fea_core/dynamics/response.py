"""Response by the modal superposition (stage F8.1, decision 3 of the specification F8): u = Σ φ_j q_j + Σ r_k c_k.

- q_j = modal coordinates (harmonically q_j(f) = p_j(f)·H_j(f), in time the exact solution of the equations of the modes for a piecewise
  linear excitation – dynamics.modal_ode), r_k = residual static vector of the excitation k (static correction of the higher modes:
  force r = u_st − Σ φ_j p_j/ω_j², base r = −(u_g − Σ φ_j Γ_j/ω_j²) per unit a_b; u_g = static displacement from the unit acceleration in the direction
  of the excitation), c_k = course of the excitation (quasi-statically in phase).
- The excitation by the base is relative: p_j = −Γ_j·a_b (Γ_j from ccx = φᵀ·M·r); the absolute displacement = relative + displacement of the
  base, the absolute acceleration = relative + a_b.
- The harmonic response is a complex amplitude (Re + i·Im, phase e^{iωt}); the maximum over the phase φ: for the quantity
  a·cos φ + b·sin φ the max |·|² = (A + B)/2 + √(((A − B)/2)² + C²), A = q(a, a), B = q(b, b), C = q(a, b)
  (q = scalar product of the vector or the quadratic form of von Mises) – exact, without sampling the phases.
"""
from __future__ import annotations

import numpy as np

from .model import G_MM, Excitation

# quadratic form of von Mises for the components of ccx SXX SYY SZZ SXY SYZ SZX: σ_vM² = sᵀ·P·s
P_VM = np.array([[1.0, -0.5, -0.5, 0, 0, 0], [-0.5, 1.0, -0.5, 0, 0, 0], [-0.5, -0.5, 1.0, 0, 0, 0],
                 [0, 0, 0, 3.0, 0, 0], [0, 0, 0, 0, 3.0, 0], [0, 0, 0, 0, 0, 3.0]])


def direction(e) -> np.ndarray:
    if getattr(e, "direction", None) is not None:
        d = np.asarray(e.direction, float)
        return d / np.linalg.norm(d)
    d = np.zeros(3)
    d[int(e.dof or 3) - 1] = 1.0
    return d


def unit_scale(e: Excitation) -> float:
    """Scale of the unit of the excitation → mm/s² (acceleration), mm (displacement), 1 (multiple of the state)."""
    if e.kind == "base_acc":
        return G_MM * float(e.scale) if e.unit == "g" else float(e.scale)
    return float(e.scale)


def table_fn(table, default: float = 1.0):
    """Course by a table [[x, k], …] linearly (outside the range the extreme value by the frequency; 0 beyond the end in time is handled
    by the caller); an empty table = the constant default.
    """
    if not table:
        return lambda x: np.full(np.shape(x), default, float)
    t = np.asarray(table, float)
    return lambda x: np.interp(np.asarray(x, float), t[:, 0], t[:, 1])


def case_load(study, mesh, case: str):
    """Nodal forces of the load state for the dynamics: (F of the mesh (N, 3), [(point mass i, point k, force)]); remote
    loads as distributing (RBE3) forces to the face (a rigid load constraint is not modelled in the modal basis).
    """
    from ..loads import nodal_loads, distributing_forces, remote_vectors
    from ..study import RemoteLoad, LoadCase
    from .mass import static_forces
    lc = next(c for c in study.load_cases if c.name == case)
    plain = LoadCase(lc.name, [l for l in lc.loads if not isinstance(l, RemoteLoad)])
    keep_masses = study.masses
    study.masses = []                    # added mass separately (point forces)
    try:
        F, _, _ = nodal_loads(study, mesh, plain)
    finally:
        study.masses = keep_masses
    warn = []
    for l in lc.loads:
        if isinstance(l, RemoteLoad):
            f, m = remote_vectors(study, l)
            F = F + distributing_forces(mesh, l.selection, l.point, f, m)
            if l.coupling == "rigid":
                warn.append(f"Dynamics: remote load '{l.selection}' (rigid coupling) acts as distributing in dynamics")
        if getattr(l, "kind", "") == "thermal":
            warn.append(f"Dynamics: the thermal load of case '{lc.name}' is not used in force excitation")
    Fm, pts = static_forces(study, mesh, lc)
    return F + Fm, pts, warn


class Terms:
    """Terms of the excitation: modal forces p (R, K) per unit of the course, kind, direction of the base, scale of the unit."""

    def __init__(self):
        self.items: list[dict] = []

    @property
    def P(self) -> np.ndarray:
        return np.array([it["p"] for it in self.items]) if self.items else np.zeros((0, 0))


def build_terms(study, mesh, basis, excitations, Mphi_point=None) -> tuple[Terms, list[str]]:
    """Modal forces of the excitation (per unit of the course × scale) – one term per item of the excitation."""
    from .modal import point_mass_terms
    terms, warn = Terms(), []
    pm = point_mass_terms(study, mesh, basis)
    for e in excitations:
        it = {"exc": e, "kind": e.kind, "scale": unit_scale(e), "phase": float(e.phase or 0.0)}
        if e.kind == "force":
            F, pts, w = case_load(study, mesh, e.case)
            warn += w
            p = np.einsum("kni,ni->k", basis.shapes, F)
            for i, k, fv in pts:
                p += pm[i][0][:, k, :] @ fv
            it.update(p=p, F=F, pts=pts)
        else:
            d = direction(e)
            it.update(p=-(basis.gamma[:, :3] @ d), dir=d)
        terms.items.append(it)
    return terms, warn


def residual_cases(study, terms: Terms) -> list:
    """Auxiliary static states for the residual correction: the state of the excitation by a force / the unit acceleration in the direction of the base."""
    from ..study import LoadCase, Gravity
    out = []
    for i, it in enumerate(terms.items):
        if it["kind"] == "force":
            lc = next(c for c in study.load_cases if c.name == it["exc"].case)
            out.append(LoadCase(f"_dyn_res_{i}", list(lc.loads)))
        else:
            out.append(LoadCase(f"_dyn_res_{i}", [Gravity(vector=list(map(float, it["dir"])))]))
    return out


def residual_vectors(basis, terms: Terms, statics: list[dict]) -> dict:
    """Residual vectors (per unit of the course without the scale of the unit): U (R, N, 3), S (R, N, 6), RF (R, 3)."""
    w2 = basis.omega ** 2
    U, S, RF = [], [], []
    for it, st in zip(terms.items, statics):
        coef = it["p"] / w2                                 # static contribution of the modes per unit
        if it["kind"] == "force":
            u, s, r = st["U"], st["S"].astype(float), st["RF"].sum(axis=0)
        else:                                               # K u = M r·1 → the relative statics = −u_g per unit a_b
            u, s, r = -st["U"], -st["S"].astype(float), -st["RF"].sum(axis=0)
        U.append(u - np.einsum("k,kni->ni", coef, basis.shapes))
        S.append((s - np.einsum("k,kni->ni", coef, basis.stress.astype(float))).astype(np.float32))
        RF.append(r - coef @ basis.rf_sum)
    return {"U": np.array(U, np.float32), "S": np.array(S, np.float32), "RF": np.array(RF)}


# ------------------------------------------------------------------ harmonic response

def freq_grid(f_min: float, f_max: float, fn: np.ndarray, zeta: np.ndarray, points: int, bias: float) -> np.ndarray:
    """Frequencies of the calculation: natural frequencies in the range, half-power points f_n(1 ± ζ) and between the anchors
    `points` points densened to the edges (bias > 1 – the anchors are the resonances) as ccx *STEADY STATE DYNAMICS.
    """
    anchors = [f_min, f_max]
    for f, z in zip(fn, zeta):
        for x in (f, f * (1 - z), f * (1 + z)):
            if f_min < x < f_max:
                anchors.append(float(x))
    anchors = np.unique(anchors)
    out = [anchors[:1]]
    y = np.linspace(-1.0, 1.0, max(int(points), 2) + 1)[1:]
    b = max(float(bias), 1.0)
    frac = 0.5 + 0.5 * np.sign(y) * np.abs(y) ** b
    for a, c in zip(anchors[:-1], anchors[1:]):
        out.append(a + (c - a) * frac)
    return np.unique(np.concatenate(out))


def frf(freq: np.ndarray, basis_f: np.ndarray, zeta: np.ndarray, eta: float | None = None) -> np.ndarray:
    """H_j(f) = 1/(ω_j² − ω² + 2iζ_jω_jω), structural 1/(ω_j²(1 + iη) − ω²); (n_f, K) complex."""
    w = 2 * np.pi * np.asarray(freq, float)[:, None]
    wj = 2 * np.pi * np.asarray(basis_f, float)[None, :]
    if eta is not None:
        return 1.0 / (wj ** 2 * (1 + 1j * eta) - w ** 2)
    return 1.0 / (wj ** 2 - w ** 2 + 2j * np.asarray(zeta)[None, :] * wj * w)


def harmonic(basis, terms: Terms, freq: np.ndarray, zeta: np.ndarray, eta: float | None, residual: bool) -> dict:
    """Modal coordinates q (n_f, K), coefficients of the residual vectors c (n_f, R), the acceleration and the displacement of the base."""
    H = frf(freq, basis.freq, zeta, eta)
    w = 2 * np.pi * freq
    q = np.zeros((len(freq), basis.K), complex)
    c = np.zeros((len(freq), len(terms.items)), complex)
    ab = np.zeros((len(freq), 3), complex)                    # acceleration of the base (vector) [mm/s²]
    for r, it in enumerate(terms.items):
        e = it["exc"]
        amp = table_fn(e.table)(freq) * it["scale"] * np.exp(1j * np.deg2rad(it["phase"]))
        if e.kind == "base_disp":                             # a_b = −ω²·d_b
            amp = -(w ** 2) * amp
        q += amp[:, None] * it["p"][None, :] * H
        if residual:
            c[:, r] = amp
        if e.kind != "force":
            ab += amp[:, None] * it["dir"][None, :]
    ub = np.where(w[:, None] > 0, -ab / np.maximum(w, 1e-300)[:, None] ** 2, 0.0)
    return {"f": np.asarray(freq, float), "q": q, "c": c, "ab": ab, "ub": ub}


def max_over_phase(a: np.ndarray, b: np.ndarray, quad=None) -> np.ndarray:
    """max_φ |a·cos φ + b·sin φ| (vectors along the last dimension; quad = matrix of the quadratic form, None = |·|²)."""
    if quad is None:
        A, B, C = (a * a).sum(-1), (b * b).sum(-1), (a * b).sum(-1)
    else:                                     # symmetric form: a·P·b = Σ a·(b@P) – matrix multiplication, not einsum
        quad = np.asarray(quad, a.dtype if a.dtype.kind == "f" else float)
        qa, qb = a @ quad, b @ quad
        A, B, C = (a * qa).sum(-1), (b * qb).sum(-1), (a * qb).sum(-1)
    return np.sqrt(np.maximum(0.5 * (A + B) + np.sqrt(0.25 * (A - B) ** 2 + C ** 2), 0.0))


def chunk_size(N: int, comps: int = 6, budget: float = 8e6) -> int:
    """Number of frequencies / times in a batch of the envelope (~budget of the values of the field per batch)."""
    return int(max(8, min(512, budget // max(comps * N, 1))))


def stress_amplitudes(basis_stress: np.ndarray, Q: np.ndarray, C: np.ndarray | None, rs: np.ndarray | None,
                      complex_mode: bool) -> np.ndarray:
    """σ_vM of the batch (n, N) float32: complex = max over the phase, real = |σ|_vM. Q (n, K), C (n, R), rs (R, 6N)."""
    N = basis_stress.shape[1] // 6
    P = P_VM.astype(np.float32)
    parts = (Q.real, Q.imag) if complex_mode else (Q,)
    cp = ((C.real, C.imag) if complex_mode else (C,)) if C is not None and C.shape[1] else (None,) * len(parts)
    S = []
    for q, c in zip(parts, cp):
        s = q.astype(np.float32) @ basis_stress
        if c is not None:
            s = s + c.astype(np.float32) @ rs
        S.append(s.reshape(len(Q), N, 6))
    if complex_mode:
        return max_over_phase(S[0], -S[1], P)
    s = S[0]
    return np.sqrt(np.maximum((s * (s @ P)).sum(-1), 0.0))


def complex_fields(basis, res: dict, q: np.ndarray, c: np.ndarray, stress: bool = True):
    """Complex fields (N, 3), (N, 6) for one frequency / time from the modal coordinates."""
    U = np.tensordot(q, basis.shapes, axes=(0, 0))
    S = np.tensordot(q, basis.stress, axes=(0, 0)) if stress else None
    if res is not None and len(c):
        U = U + np.tensordot(c, res["U"], axes=(0, 0))
        if stress:
            S = S + np.tensordot(c, res["S"], axes=(0, 0))
    return U, S


def envelopes(basis, res: dict | None, Q: np.ndarray, C: np.ndarray, complex_mode: bool, chunk: int = 0) -> dict:
    """Envelope over the frequencies (complex – max over the phase) or the time (real): max |u| (N,), max σ_vM (N,) and the index of the
    frequency / time of the maximum (the first occurrence). Q (n, K), C (n, R).
    """
    from ..modules import available
    if available("accel") and not chunk:              # numba: without the arrays (n × N) in the memory
        from . import _env
        return _env.envelopes(basis.shapes, basis.stress, res["U"] if res is not None else None,
                              res["S"] if res is not None else None, Q, C, complex_mode)
    N = basis.shapes.shape[1]
    sh = basis.shapes.reshape(basis.K, -1).astype(np.float32)
    ss = basis.stress.reshape(basis.K, -1).astype(np.float32)
    ru = res["U"].reshape(len(res["U"]), -1).astype(np.float32) \
        if res is not None and len(C[0] if len(C) else []) else None
    rs = res["S"].reshape(len(res["S"]), -1).astype(np.float32) if ru is not None else None
    uenv = np.zeros(N, np.float32)
    senv = np.zeros(N, np.float32)
    uarg = np.zeros(N, np.int32)
    sarg = np.zeros(N, np.int32)
    chunk = chunk or chunk_size(N)
    cols = np.arange(N)
    for s0 in range(0, len(Q), chunk):
        qq = Q[s0:s0 + chunk]
        cc = C[s0:s0 + chunk] if ru is not None else None
        parts = (qq.real, qq.imag) if complex_mode else (qq,)
        cps = ((cc.real, cc.imag) if complex_mode else (cc,)) if cc is not None and cc.shape[1] else (None,) * len(parts)
        U = []
        for q, c in zip(parts, cps):
            u = q.astype(np.float32) @ sh
            if c is not None:
                u = u + c.astype(np.float32) @ ru
            U.append(u.reshape(len(qq), N, 3))
        um = max_over_phase(U[0], -U[1]) if complex_mode else np.sqrt((U[0] * U[0]).sum(-1))
        sm = stress_amplitudes(ss, qq, cc, rs, complex_mode)
        for m_, env, arg in ((um, uenv, uarg), (sm, senv, sarg)):
            k = m_.argmax(axis=0)                           # first maximum in the batch
            v = m_[k, cols]
            m = v > env
            env[m] = v[m]
            arg[m] = s0 + k[m]
    return {"U": uenv, "S": senv, "U_at": uarg, "S_at": sarg}


# ------------------------------------------------------------------ sondy

def probe_operators(study, mesh, probes) -> list[dict]:
    """Interpolation of the probes: point → element and shape functions; selection → mean of the nodes of the face. [{"name", "nodes", "w", "point"}]."""
    from ..results import Results, tet10_shape
    out = []
    tmp = Results(mesh, [])
    for p in probes:
        if p.selection:
            idx = mesh.surface_nodes(p.selection)
            out.append({"name": p.name, "nodes": idx, "w": np.full(len(idx), 1.0 / len(idx)),
                        "point": mesh.nodes[idx].mean(axis=0).tolist(), "quantity": p.quantity})
            continue
        e, xi = tmp.locate(np.asarray(p.point, float))
        conn = mesh.tets[e]
        N = tet10_shape(xi) if len(conn) == 10 else np.array([1 - xi.sum(), *xi])
        out.append({"name": p.name, "nodes": conn, "w": N, "point": list(map(float, p.point)),
                    "quantity": p.quantity})
    return out


def probe_series(basis, res, ops: list[dict], Q: np.ndarray, C: np.ndarray, Qdd: np.ndarray | None = None) -> dict:
    """Series in the probes: U (n, 3), S (n, 6), A (n, 3 – modal acceleration, only in time)."""
    out = {}
    for op in ops:
        phi = np.einsum("p,kpi->ki", op["w"], basis.shapes[:, op["nodes"]])
        sig = np.einsum("p,kpi->ki", op["w"], basis.stress[:, op["nodes"]].astype(float))
        U = Q @ phi
        S = Q @ sig
        if res is not None and C.shape[1]:
            U = U + C @ np.einsum("p,rpi->ri", op["w"], res["U"][:, op["nodes"]].astype(float))
            S = S + C @ np.einsum("p,rpi->ri", op["w"], res["S"][:, op["nodes"]].astype(float))
        d = {"U": U, "S": S}
        if Qdd is not None:
            d["A"] = Qdd @ phi
        out[op["name"]] = d
    return out


# ------------------------------------------------------------------ transient response

def base_disp_impulses(terms: Terms) -> list[tuple[float, np.ndarray]]:
    """Dirac impulses of the acceleration at the breaks of the piecewise linear displacement of the base (A3).

    Returns the list [(t_k, Δq̇_k)], where Δq̇_k = it['p'] * Δv_k (K,).
    """
    impulses_map: dict[float, np.ndarray] = {}
    for it in terms.items:
        if it["kind"] != "base_disp":
            continue
        e = it["exc"]
        if not e.table:
            continue
        tab = np.asarray(e.table, float)
        tt = tab[:, 0]
        dd = tab[:, 1] * it["scale"]
        dt_seg = np.diff(tt)
        if len(dt_seg) == 0 or np.any(dt_seg <= 0):
            continue
        v_seg = np.diff(dd) / dt_seg
        v_before = np.concatenate([[0.0], v_seg])
        v_after = np.concatenate([v_seg, [0.0]])
        delta_v = v_after - v_before
        p = it["p"]
        for t_k, dv_k in zip(tt, delta_v):
            if abs(dv_k) < 1e-15:
                continue
            t_key = round(float(t_k), 10)
            if t_key not in impulses_map:
                impulses_map[t_key] = np.zeros_like(p)
            impulses_map[t_key] = impulses_map[t_key] + p * dv_k
    return sorted(impulses_map.items(), key=lambda x: x[0])


def time_terms(terms: Terms, t: np.ndarray, ccx_convention: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Courses of the terms in time: P (T, K) modal forces, C (T, R) coefficients of the residual vectors, a_b (T, 3) acceleration of the
    base, u_b (T, 3) displacement of the base. Table [[t, k], …], beyond the end of the table 0 (a step = [[0, 1], [t_end, 1]]);
    displacement of the base: for the physically exact solution (ccx_convention=False, A3) the continuous acceleration between the breaks is zero
    and the exciting force is applied by Dirac impulses (base_disp_impulses); for ccx_convention=True the continuous acceleration is computed
    by the second central difference as in ccx *MODAL DYNAMIC.
    """
    T = len(t)
    K = len(terms.items[0]["p"]) if terms.items else 0
    P = np.zeros((T, K))
    C = np.zeros((T, len(terms.items)))
    ab = np.zeros((T, 3))
    ub = np.zeros((T, 3))
    dt = t[1] - t[0] if T > 1 else 1.0
    for r, it in enumerate(terms.items):
        e = it["exc"]
        if e.table:
            tab = np.asarray(e.table, float)
            s = np.interp(t, tab[:, 0], tab[:, 1], left=0.0)        # the displacement of the base holds the last value
            if e.kind != "base_disp":
                s[t > tab[-1, 0] + 1e-12 * max(abs(tab[-1, 0]), 1.0)] = 0.0
        else:
            s = np.ones(T)
        s = s * it["scale"]
        if e.kind == "base_disp":
            disp = s
            ub += disp[:, None] * it["dir"][None, :]
            dp = np.concatenate([disp[:1], disp, disp[-1:]])
            a = (dp[2:] - 2 * dp[1:-1] + dp[:-2]) / dt ** 2
            ab += a[:, None] * it["dir"][None, :]
            if ccx_convention:
                s = a
                P += s[:, None] * it["p"][None, :]
                C[:, r] = s
            else:
                s = np.zeros(T)
                C[:, r] = 0.0
        else:
            if e.kind != "force":
                ab += s[:, None] * it["dir"][None, :]
            P += s[:, None] * it["p"][None, :]
            C[:, r] = s
    # displacement of the base from the acceleration (base_acc): double integration by the trapezium from rest
    if any(it["kind"] == "base_acc" for it in terms.items):
        a_acc = np.zeros((T, 3))
        for r, it in enumerate(terms.items):
            if it["kind"] == "base_acc":
                a_acc += C[:, r][:, None] * it["dir"][None, :]
        v = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (a_acc[1:] + a_acc[:-1]) * dt, axis=0)])
        ub += np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (v[1:] + v[:-1]) * dt, axis=0)])
    return P, C, ab, ub


def settle_time(t: np.ndarray, u: np.ndarray, tol: float, ref: np.ndarray | None = None) -> float | None:
    """Settling time: the last time at which |u − u_end| > tol (u (T, 3)); None = it does not settle until the end (the deviation at the end
    > tol / 2 is not evaluated).
    """
    ref = u[-1] if ref is None else ref
    dev = np.linalg.norm(u - ref, axis=1)
    out = np.nonzero(dev > tol)[0]
    if not len(out):
        return 0.0
    if out[-1] >= len(t) - 2:
        return None
    return float(t[out[-1] + 1])
