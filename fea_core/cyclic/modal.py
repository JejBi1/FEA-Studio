"""Natural frequencies of a cyclic-symmetric sector by nodal diameters (ccx *CYCLIC SYMMETRY MODEL, see fea_core.cyclic).

One ccx job per contiguous range of nodal diameters (*SELECT CYCLIC SYMMETRY MODES, NMIN, NMAX); a model with nodes on the
axis has ND 0 and ND 1 in separate jobs (ccx manual node175: the axis nodes are treated differently for ND 0, 1 and > 1).
Every eigenvalue comes twice from ccx → modes are read in pairs (2·k eigenvalues requested for k modes per ND); for
0 < ND < N/2 the frequency belongs to a degenerate pair of the full structure (multiplicity 2), for ND 0 and N/2 to one mode.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import CyclicError, active, nodal_diameters, pairs

_NUM = r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?"
_ROW = re.compile(rf"^\s*(\d+)\s+(\d+)\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*$")


def groups(nds: list[int], has_axis: bool) -> list[tuple[int, int]]:
    """Contiguous ranges (NMIN, NMAX) of the requested nodal diameters; with axis nodes ND 0 and ND 1 alone."""
    out: list[list[int]] = []
    for k in sorted(set(nds)):
        alone = has_axis and k <= 1
        if out and not alone and out[-1][-1] == k - 1 and not (has_axis and out[-1][-1] <= 1):
            out[-1].append(k)
        else:
            out.append([k])
    return [(g[0], g[-1]) for g in out]


def read_dat(path: Path) -> list[tuple[int, int, float, float]]:
    """Rows of the cyclic eigenvalue tables: (nodal diameter, mode no. within the ND, eigenvalue, frequency [Hz])."""
    rows = []
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        mt = _ROW.match(line)
        if mt:
            v = [float(x.replace("D", "E").replace("d", "e")) for x in mt.groups()[2:]]
            rows.append((int(mt.group(1)), int(mt.group(2)), v[0], v[2]))
    return rows


def run_modal(study, mesh, workdir: Path, log, run, sub_study):
    """ModeSet of the cyclic sector: values = frequencies [Hz] ordered by nodal diameter, attributes nodal_diameters,
    multiplicity (modes of the full structure) and shapes_imag (imaginary part for the expansion)."""
    from ..ccx.inp import write_inp
    from ..ccx.frd import read_frd
    from ..eigen import ModeSet, EigenError
    from ..study import LoadCase
    spec = active(study)
    p = pairs(study, mesh)
    nds = nodal_diameters(spec)
    if not nds:
        raise EigenError("Cyclic symmetry: no valid nodal diameter selected")
    k = max(1, int(study.modal.modes))
    n = len(mesh.nodes)
    vals, nd_of, mult, re_s, im_s = [], [], [], [], []
    for gi, (a, b) in enumerate(groups(nds, len(p["axis"]) > 0)):
        job = f"modal_cyc{gi + 1}"
        log(f"Natural frequencies (CalculiX, cyclic symmetry N = {spec.sectors}, nodal diameters {a}–{b}, {k} modes each)")
        st = sub_study(study, [LoadCase("Natural frequencies", [])])
        line = f"{2 * k}\n*SELECT CYCLIC SYMMETRY MODES, NMIN={a}, NMAX={b}"
        try:
            write_inp(st, mesh, Path(workdir) / f"{job}.inp", procedure="modal", eigen_line=line)
        except ValueError as e:
            raise EigenError(str(e)) from None
        run(job)
        rows = read_dat(Path(workdir) / f"{job}.dat")
        if not rows:
            raise EigenError("Cyclic symmetry: ccx returned no eigenvalue")
        frd = read_frd(Path(workdir) / f"{job}.frd")
        disp = {bk.mode: bk for bk in frd.blocks if bk.name == "DISP"}
        dispi = {bk.mode: bk for bk in frd.blocks if bk.name == "DISPI"}
        by_nd: dict[int, list[tuple[int, int, float, float]]] = {}
        for g, (nd, mode, eig, f) in enumerate(rows, 1):
            by_nd.setdefault(nd, []).append((g, mode, eig, f))
        for nd in sorted(by_nd):
            if nd not in nds:
                continue
            lst = sorted(by_nd[nd], key=lambda r: r[1])
            for j in range(0, len(lst) - 1, 2):
                g, _, eig, f = lst[j]
                if g not in disp:
                    raise EigenError(f"Cyclic symmetry: the mode shape of nodal diameter {nd} (mode {g}) is missing in the .frd")
                u = np.zeros((n, 3))
                v = np.zeros((n, 3))
                bk = disp[g]
                msk = bk.node_ids <= n
                u[bk.node_ids[msk] - 1] = bk.values[msk, :3]
                if g in dispi:
                    bi = dispi[g]
                    mi = bi.node_ids <= n
                    v[bi.node_ids[mi] - 1] = bi.values[mi, :3]
                amp = np.sqrt((u ** 2).sum(axis=1) + (v ** 2).sum(axis=1)).max()
                if amp > 0:
                    sgn = 1.0 if u.ravel()[np.abs(u).argmax()] >= 0 else -1.0
                    u, v = sgn * u / amp, sgn * v / amp
                vals.append(float(f) if eig > 0 else 0.0)
                nd_of.append(int(nd))
                mult.append(1 if nd == 0 or 2 * nd == spec.sectors else 2)
                re_s.append(u.astype(np.float32))
                im_s.append(v.astype(np.float32))
    if not vals:
        raise CyclicError("Cyclic symmetry: no natural frequency was found")
    ms = ModeSet("modal", "Natural frequencies (cyclic symmetry)", vals, np.stack(re_s))
    ms.nodal_diameters = nd_of
    ms.multiplicity = mult
    ms.shapes_imag = np.stack(im_s)
    ms.sectors = int(spec.sectors)
    return ms


def table(ms) -> list[dict]:
    """Rows for the GUI, MCP and the report: nodal diameter, frequency, number of modes of the full structure."""
    nds = getattr(ms, "nodal_diameters", None) or []
    mult = getattr(ms, "multiplicity", None) or [1] * len(nds)
    return [{"mode": j + 1, "nodal_diameter": int(nd), "frequency_Hz": float(f), "multiplicity": int(m)}
            for j, (nd, f, m) in enumerate(zip(nds, ms.values, mult))]
