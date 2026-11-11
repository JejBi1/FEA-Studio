"""Fatigue from the dynamics (specification F8, 5a.6): S-N by materials of the nodes (as fatigue.analysis – given in
Study.fatigue or an estimate from Rm with Marin), fatigue from the sinusoidal sweep (D1) and from the spectrum (D3, fatigue.spectral).
Equivalent stress = von Mises (amplitude max over the phase, resp. Preumont & Piéfort for the PSD); printed parts
indicatively (the Hill equivalent in the axes of the print is not here – a warning).
"""
from __future__ import annotations

import numpy as np


def node_groups(study, mesh) -> list[tuple[str, np.ndarray, object]]:
    """[(material, nodes, SNCurve)] – a node shared by parts belongs to the worse S-N (lower SD), as fatigue.analysis."""
    from ..fatigue.analysis import sn_for_material
    from ..fatigue.settings import FatigueSettings
    st = study.fatigue or FatigueSettings()
    n = len(mesh.nodes)
    best = np.full(n, np.inf)
    owner = np.full(n, -1)
    sns = {}
    for pi, pname in enumerate(mesh.part_names):
        mat = study.material_of(pname)
        if mat.name not in sns:
            try:
                sns[mat.name] = sn_for_material(mat, st)[0]
            except ValueError:                       # without Rm and without a given S-N → the fatigue of the part is not computed
                sns[mat.name] = None
        sn = sns[mat.name]
        if sn is None:
            continue
        pn = np.unique(mesh.tets[mesh.tet_part == pi])
        w = sn.SD < best[pn]
        best[pn[w]] = sn.SD
        owner[pn[w]] = list(sns).index(mat.name)
    names = list(sns)
    return [(names[k], np.nonzero(owner == k)[0], sns[names[k]]) for k in range(len(names))
            if (owner == k).any()]


def missing_sn(study, mesh) -> list[str]:
    """Materials of the parts without S-N (neither Rm nor a given curve) – their fatigue from the dynamics is not computed."""
    st = study.fatigue
    out = []
    for p in mesh.part_names:
        mat = study.material_of(p)
        if mat.Rm is None and not (st and st.sn.get(mat.name)) and mat.name not in out:
            out.append(mat.name)
    return out


def print_warning(study, mesh) -> list[str]:
    from ..printing.region import print_parts
    pp = print_parts(study, mesh.part_names)
    return ["Dynamic fatigue of printed parts (" + ", ".join(pp[:3]) + ") uses von Mises – the Hill equivalent in the print axes is not used here, the result is indicative only"] if pp else []


def sweep_fatigue(study, mesh, out) -> dict:
    """Damage from the logarithmic sweep of the sine over the frequency grid of the harmonic response: amplitude σ_vM(f) max over the
    phase in each node, D = passes·60/(R·ln 2)·∫ df/N(σ_a(f)).
    """
    from ..fatigue.spectral import inverse_life, sweep_cycles_per_hz
    from ..dynamics.psd import trapz_weights
    from .response import stress_amplitudes, chunk_size
    h = out.harmonic
    spec = study.dynamics.harmonic
    f = np.asarray(h["f"], dtype=np.float64)
    if np.any(np.diff(f) <= 0):
        raise ValueError("Sweep frequencies must be increasing")
    N = out.shapes.shape[1]
    rs = None
    if out.residual is not None and len(h.get("res_index", [])) and h["c"].shape[1]:
        res = out.residual["S"][h["res_index"]]
        rs = res.reshape(len(res), -1).astype(np.float32)
    ss = out.stress.reshape(out.K, -1).astype(np.float32)
    wf = trapz_weights(f)
    groups = node_groups(study, mesh)
    from ..modules import available
    if available("accel"):                          # numba: S-N per node, σ_a(f) is not assembled
        from ..fatigue.spectral import _sn_terms
        from . import _env
        sn_n = np.zeros((4, N))
        for name, nodes, sn in groups:
            sn_n[:, nodes] = np.array(_sn_terms(sn), float)[:, None]
        D = _env.sweep_sum(ss.reshape(out.K, N, 6), rs.reshape(-1, N, 6) if rs is not None else None, h["q"],
                           h["c"] if rs is not None else None, wf, *sn_n)
    else:
        D = np.zeros(N)
        step = chunk_size(N)
        for s0 in range(0, len(f), step):           # amplitude of σ_vM(f) by batches – the whole field (n_f × N) is not kept
            sl = slice(s0, s0 + step)
            sig = stress_amplitudes(ss, h["q"][sl], h["c"][sl] if rs is not None else None, rs, True)
            for name, nodes, sn in groups:
                D[nodes] += wf[sl] @ inverse_life(sig[:, nodes], sn)
    D *= float(spec.sweeps) * sweep_cycles_per_hz(float(spec.sweep_rate))
    i = int(D.argmax())
    return {"D": D.astype(np.float32),
            "summary": {"max_damage": float(D.max()), "at_node": i + 1, "at": mesh.nodes[i].tolist(),
                        "rate_oct_min": float(spec.sweep_rate), "sweeps": float(spec.sweeps)}}
