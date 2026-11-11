"""Fatigue of printed parts (F4, chapter 8 of the specification: "S-N as calibration parameters").

Stresses in the axes of the material (print axes, for perimeters the axes of the line) separately in the shell and in the core from the same strain
at the Gauss points of the element (as printing.failure). Multiaxiality: Hill equivalent of fatigue
    σ_eq = SD_xy · √FI(σ; X_f, Y_f, Z_f, S_f),   X_f = X·k_xy, Y_f = Y·k_xy, Z_f = Z·k_z, S = S·k (xy / z),
k_xy = SD_xy/Rm (amplitude of the S-N at ND / static strength XY), k_z = SD_z/Rm_z. The fatigue strengths of the directions are
thus static strengths × the ratio of the S-N (for the core and the perimeters the same scaling as the statics). Uniaxial tension in the
direction i gives σ_eq = SD_xy·σ/SD_i → one S-N (XY) with a common slope k is enough. The sign = the sign of the component
with the largest |σ_i|/X_i. Then rainflow + Goodman/Gerber (with Rm XY – normalised stress) + Miner as for metals
(fatigue.kernel). Node = the worst of the points of the elements around it.

Limits: a common slope k for all directions, compressive amplitudes taken as tensile (conservative), no influence of the frequency
and of the heating (plastics heat up at a higher frequency), no notch sensitivity of the printing.
"""
from __future__ import annotations

import math

import numpy as np

from ..modules import need

_numba = need("numba", "accel")      # if the Numba module is missing, a clear error with the installation procedure
njit = _numba.njit
prange = _numba.prange

from ..fatigue import kernel
from .failure import element_strains, subtract_thermal
from .infill import strain_to_local


@njit(cache=True)
def _hill_eq(s, c, inv):
    """Hill equivalent with sign; c = (F, G, H, 1/S23², 1/S13², 1/S12²)·SD², inv = 1/X_f (3 components)."""
    fi = (c[0] * (s[1] - s[2]) ** 2 + c[1] * (s[2] - s[0]) ** 2 + c[2] * (s[0] - s[1]) ** 2
          + c[3] * s[5] ** 2 + c[4] * s[4] ** 2 + c[5] * s[3] ** 2)
    v = math.sqrt(max(fi, 0.0))
    best, sg = -1.0, 1.0
    for i in range(3):
        r = abs(s[i]) * inv[i]
        if r > best:
            best = r
            sg = 1.0 if s[i] >= 0 else -1.0
    return sg * v


@njit(parallel=True, cache=True)
def _points(F, S, C, INV, SD, ND, k, k2, Rm, mean_method, repeats, closed, out_D, out_sf, out_a):
    """F (T, K), S (K, P, 6) stresses of the points in the axes of the material for the unit states."""
    T, K = F.shape
    P = S.shape[1]
    for p in prange(P):
        x = np.empty(T)
        s = np.empty(6)
        for t in range(T):
            for j in range(6):
                acc = 0.0
                for kk in range(K):
                    acc += F[t, kk] * S[kk, p, j]
                s[j] = acc
            x[t] = _hill_eq(s, C[p], INV[p])
        if closed:
            x = kernel.close_block(x)
        L = x.shape[0] + 1
        rng = np.empty(L)
        mean = np.empty(L)
        cnt = np.empty(L)
        nc = kernel.rainflow_cycles(x, rng, mean, cnt)
        nc = kernel.merge_cycles(rng, mean, cnt, nc)
        out_D[p], out_sf[p], out_a[p] = kernel.node_result(rng, mean, cnt, nc, 1.0, 1.0, SD, ND, k, k2, Rm,
                                                           mean_method, repeats)


@njit(parallel=True, cache=True)
def _points_prop(f, Seff, C, INV, SD, ND, k, k2, Rm, mean_method, repeats, closed, out_D, out_sf, out_a):
    """Proportional loading σ(t) = f(t)·Seff → x(t) = f(t)·eq(Seff); rainflow only once."""
    x = kernel.close_block(f) if closed else f.copy()
    L = x.shape[0] + 1
    rng = np.empty(L)
    mean = np.empty(L)
    cnt = np.empty(L)
    nc = kernel.rainflow_cycles(x, rng, mean, cnt)
    nc = kernel.merge_cycles(rng, mean, cnt, nc)
    for p in prange(Seff.shape[0]):
        e = _hill_eq(Seff[p], C[p], INV[p])
        out_D[p], out_sf[p], out_a[p] = kernel.node_result(rng, mean, cnt, nc, e, e, SD, ND, k, k2, Rm,
                                                           mean_method, repeats)


def fatigue_strengths(m, k_xy: float, k_z: float, SD: float) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients of the Hill equivalent (normalised to SD) and 1/X_f for the material Ortho."""
    X, Y, Z = m.Xt * k_xy, m.Ytv * k_xy, m.Zt * k_z
    S12, S3 = m.Sxy * k_xy, m.Sz * k_z
    i1, i2, i3 = 1 / X ** 2, 1 / Y ** 2, 1 / Z ** 2
    c = np.array([0.5 * (i2 + i3 - i1), 0.5 * (i1 + i3 - i2), 0.5 * (i1 + i2 - i3),
                  1 / S3 ** 2, 1 / S3 ** 2, 1 / S12 ** 2]) * SD ** 2
    return c, np.array([1 / X, 1 / Y, 1 / Z])


def sn_for_print(study, mat, settings) -> tuple[dict, dict]:
    """S-N of the printed material (XY) + the ratios k_xy, k_z. A given S-N in the fatigue (FatigueSettings.sn) takes precedence."""
    from .props import props_of
    pr = props_of(study, mat)
    m = settings.k_load * settings.k_temp * settings.k_misc
    given = settings.sn.get(mat.name)
    if given:
        sn = {k: given[k] for k in ("SD", "ND", "k", "k2") if k in given}
        sn.setdefault("ND", 1e6); sn.setdefault("k", pr.sn_k); sn.setdefault("k2", pr.sn_k2)   # plastics have no fatigue limit
        src = given.get("source", "user-defined") + " (XY, print)"
    else:
        sn = {"SD": pr.sn_xy(mat.Rm), "ND": pr.sn_ND, "k": pr.sn_k, "k2": pr.sn_k2}
        src = ("3D tisk XY: " + (f"SD = {pr.sn_SD:g} MPa" if pr.sn_SD else f"SD = {pr.sn_SD_ratio:g}·Rm")
               + " – INDICATIVE (Ezeh & Susmel 2019), calibrate by testing")
    sn["SD"] *= m
    sn["source"] = src
    Rm = mat.Rm or sn["SD"] / pr.sn_SD_ratio
    k_xy = sn["SD"] / Rm
    if pr.sn_z_ratio is not None and mat.Rm_z:
        k_z = sn["SD"] * pr.sn_z_ratio / mat.Rm_z
    else:
        k_z = k_xy                                     # SD_z/SD_xy = Rm_z/Rm
    info = {"sn": sn, "marin": {"k": m, "ka": 1.0, "kb": 1.0, "kc": settings.k_load, "kd": settings.k_temp,
                                "ke": 1.0, "kf": settings.k_misc},
            "print": {"k_xy": k_xy, "k_z": k_z, "SD_z": k_z * (mat.Rm_z or Rm), "Rm": Rm,
                      "method": "Hill equivalent in print/line axes, shell and core"}}
    return sn, info


def run(study, results, F: np.ndarray, settings, mean_code: int) -> dict | None:
    """Fatigue of the printed parts: nodal fields D, safety, amplitude (NaN outside the printing) and the S-N used."""
    from . import shell_of, part_materials, element_orientation
    mesh = results.mesh
    sh = shell_of(mesh)
    if sh is None:
        return None
    n = len(mesh.nodes)
    D = np.full(n, np.nan)
    SF = np.full(n, np.nan)
    A = np.full(n, np.nan)
    used = {}
    from ..fatigue.analysis import proportional
    pf = proportional(F)
    for pi, pname in enumerate(mesh.part_names):
        el = np.nonzero((mesh.tet_part == pi) & (sh >= 0))[0]
        if len(el) == 0:
            continue
        mat = study.material_of(pname)
        sn, info = sn_for_print(study, mat, settings)
        used[mat.name] = info
        k_xy, k_z = info["print"]["k_xy"], info["print"]["k_z"]
        ms, ml, mc = part_materials(study, pname)
        kind, axes = element_orientation(study, mesh, pi, el)
        f = sh[el]
        # points: (element, Gauss point, shell/core)
        Sk, Cs, Is, owner = [], [], [], []
        conn = mesh.tets[el]
        eps_all = element_strains(mesh.nodes, conn, [st.U for st in results.steps])   # K × (E, 4, 6)
        eps_all = [subtract_thermal(e, conn, st.fields.get("T"), study.thermal.reference_temperature, mat.alpha)
                   for e, st in zip(eps_all, results.steps)]            # thermal states: mechanical strain
        groups = []
        for k_ in axes:                                 # shell by classes of the line direction
            sel = (kind == k_) & (f > 0)
            if sel.any():
                groups.append((sel, axes[k_], ms if k_ < 0 else ml))
        core_sel = f < 1
        if core_sel.any():
            groups.append((core_sel, axes[-1], mc))
        for sel, R, m in groups:
            Cm = m.stiffness()
            c, inv = fatigue_strengths(m, k_xy, k_z, sn["SD"])
            sig = np.stack([(strain_to_local(e[sel], R) @ Cm.T).reshape(-1, 6) for e in eps_all])   # (K, P, 6)
            Sk.append(sig)
            P = sig.shape[1]
            Cs.append(np.broadcast_to(c, (P, 6)))
            Is.append(np.broadcast_to(inv, (P, 3)))
            owner.append(np.repeat(np.nonzero(sel)[0], 4))
        S = np.ascontiguousarray(np.concatenate(Sk, axis=1))
        C = np.ascontiguousarray(np.concatenate(Cs))
        INV = np.ascontiguousarray(np.concatenate(Is))
        own = np.concatenate(owner)
        P = S.shape[1]
        oD, oS, oA = np.zeros(P), np.zeros(P), np.zeros(P)
        args = (C, INV, float(sn["SD"]), float(sn["ND"]), float(sn["k"]), float(sn["k2"]), float(info["print"]["Rm"]),
                int(mean_code), float(settings.repeats), bool(settings.closed_blocks), oD, oS, oA)
        if pf is not None:
            fv, cv = pf
            Seff = np.ascontiguousarray(np.tensordot(cv, S, axes=(0, 0)))
            _points_prop(np.ascontiguousarray(fv), Seff, *args)
        else:
            _points(np.ascontiguousarray(F), S, *args)
        # element = the worst point, node = the worst element
        eD = np.zeros(len(el)); eS = np.full(len(el), np.inf); eA = np.zeros(len(el))
        np.maximum.at(eD, own, oD); np.minimum.at(eS, own, oS); np.maximum.at(eA, own, oA)
        conn = mesh.tets[el]
        for arr, e_val in ((D, eD), (A, eA)):
            tmp = np.full(n, -np.inf)
            np.maximum.at(tmp, conn.ravel(), np.repeat(e_val, conn.shape[1]))
            hit = np.isfinite(tmp)
            arr[hit] = np.fmax(arr[hit], tmp[hit])
        tmp = np.full(n, np.inf)
        np.minimum.at(tmp, conn.ravel(), np.repeat(eS, conn.shape[1]))
        hit = np.unique(conn)
        SF[hit] = np.fmin(SF[hit], tmp[hit])
    if not used:
        return None
    return {"damage": D, "safety": SF, "amplitude": A, "sn_used": used,
            "warnings": ["3D printing: fatigue using the Hill equivalent in print axes; print S-N curves are INDICATIVE (calibration parameters) – verify by testing; Marin surface and size factors are not applied to printed parts."]}


def apply(study, results, F, settings, mean_code, idx, out_D, out_sf, out_a, sn_used, warnings) -> np.ndarray | None:
    """Overwrites the results of the metal core of the fatigue in the nodes of the printed parts (nodes idx). A node shared with an unprinted
    part takes the worse of both. Returns the mask (over idx) of the nodes where the printing result decides (None = no printing).
    """
    r = run(study, results, F, settings, mean_code)
    if r is None:
        return None
    mesh = results.mesh
    from . import shell_of
    sh = shell_of(mesh)
    other = np.zeros(len(mesh.nodes), bool)
    other[np.unique(mesh.tets[sh < 0])] = True
    D, SF, A = r["damage"][idx], r["safety"][idx], r["amplitude"][idx]
    pr = ~np.isnan(D)
    shared = pr & other[idx]
    only = pr & ~other[idx]
    used = only | (shared & (D > out_D))
    out_D[only], out_sf[only], out_a[only] = D[only], SF[only], A[only]
    out_D[shared] = np.maximum(out_D[shared], D[shared])
    out_sf[shared] = np.minimum(out_sf[shared], SF[shared])
    out_a[shared] = np.maximum(out_a[shared], A[shared])
    sn_used.update(r["sn_used"])
    warnings.extend(w for w in r["warnings"] if w not in warnings)
    warnings[:] = [w for w in warnings if not any(w.startswith(m + ":") for m in r["sn_used"])
                   or not any(k in w.lower() for k in ("calibration", "kalibrace"))]
    return used
