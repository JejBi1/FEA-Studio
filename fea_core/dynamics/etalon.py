"""Etalon of the modal superposition: the same model in ccx *STEADY STATE DYNAMICS and *MODAL DYNAMIC (from the .eig of the step
*FREQUENCY, STORAGE=YES) with the output only in the nodes of the probes (*NODE PRINT NSET=NPROBE, specification F8 5a.3, decision 3:
"ccx SSD and MD remain the etalon"). Compares the displacement in the node of the probe (the nearest node) Python × ccx.

Conventions (trials scratch/f8_dynamics): SSD writes into the .dat two blocks at the frequency (real and imaginary part),
the time in the heading = frequency [Hz]; *MODAL DYNAMIC time from 0 (the frequency step does not add the time), default limit of the increments 100
→ *STEP, INC=; with *BASE MOTION the displacements of ccx are ABSOLUTE.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np

from . import response as R


def _fmt(v: float) -> str:
    return f"{v:.12g}"


def _damping(zeta: np.ndarray) -> str:
    lines = "".join(f"{j + 1}, {j + 1}, {_fmt(z)}\n" for j, z in enumerate(zeta))
    return "*MODAL DAMPING\n" + lines


def _loads(terms, study, mesh, time_domain: bool, amp_names: list[str], ub: np.ndarray | None) -> str:
    out = []
    for r, it in enumerate(terms.items):
        e = it["exc"]
        a = f", AMPLITUDE={amp_names[r]}" if amp_names[r] else ""
        if e.kind == "force":
            F = it["F"] * it["scale"]
            nn, dd = np.nonzero(np.abs(F) > 0)
            if it.get("pts"):
                raise ValueError("reference: force excitation with an added point mass is not converted")
            out.append(f"*CLOAD{a}\n" + "".join(f"{n + 1}, {d + 1}, {_fmt(F[n, d])}\n" for n, d in zip(nn, dd)))
        else:
            k = int(np.argmax(np.abs(it["dir"])))
            if np.count_nonzero(np.abs(it["dir"]) > 1e-12) != 1:
                raise ValueError("reference: *BASE MOTION only along an axis")
            typ = "DISPLACEMENT" if time_domain else ("ACCELERATION" if e.kind == "base_acc" else "DISPLACEMENT")
            out.append(f"*BASE MOTION, DOF={k + 1}{a}, TYPE={typ}\n")
    return "".join(out)


def read_print(dat: Path, setname: str = "NPROBE") -> list[tuple[float, dict[int, np.ndarray]]]:
    """Blocks *NODE PRINT U for a set: [(time, {node: (3,)})] in the order of the output."""
    out, cur = [], None
    pat = re.compile(rf"displacements \(vx,vy,vz\) for set {setname} and time\s+(\S+)", re.I)
    for line in Path(dat).read_text(encoding="latin-1").splitlines():
        m = pat.search(line)
        if m:
            cur = {}
            out.append((float(m.group(1).replace("D", "E")), cur))
            continue
        if cur is not None:
            p = line.split()
            if not p:
                if cur:
                    cur = None
                continue
            try:
                cur[int(p[0])] = np.array([float(v) for v in p[1:4]])
            except ValueError:
                cur = None
    return out


def run_etalon(study, mesh, basis, zeta, out, workdir: Path, log, on_line=None, cancel=None) -> dict:
    """ccx SSD / MD in the same probes → {"harmonic": {"max_rel_error", …}, "transient": {…}}."""
    from ..ccx.inp import write_inp
    from ..ccx.runner import run_ccx
    from ..config import CCX_EXE
    from ..eigen import _sub_study
    from ..study import LoadCase
    ds = study.dynamics
    if ds.damping.kind == "structural":
        return {"skipped": "ccx reference: structural damping is not converted (*DAMPING, STRUCTURAL per material)"}
    ops = out.probes or []
    nodes = sorted({int(op["nodes"][int(np.argmax(op["w"]))]) for op in ops})
    peak = [rr.get("peak_node") for rr in (out.harmonic, out.transient) if rr is not None]
    nodes = sorted(set(nodes) | {int(n) for n in peak if n is not None})
    if not nodes:
        return {"skipped": "ccx reference: no probes"}
    exe = Path(study.solver.exe) if study.solver.exe else CCX_EXE
    rep = {}
    for kind in ("harmonic", "transient"):
        spec = getattr(ds, kind)
        rr = getattr(out, kind)
        if not spec.enabled or rr is None:
            continue
        terms, _ = R.build_terms(study, mesh, basis, spec.excitation)
        st = _sub_study(study, [LoadCase("Natural frequencies", [])])
        job = f"etalon_{kind}"
        try:
            write_inp(st, mesh, Path(workdir) / f"{job}.inp", procedure="modal", eigen_line=f"{basis.K}, 0.",
                      storage=True, probe_nodes=np.array(nodes))
            amps, names = [], []
            if kind == "harmonic":
                for r, it in enumerate(terms.items):
                    e = it["exc"]
                    if e.table or e.kind != "force":
                        tab = np.asarray(e.table, float) if e.table else np.array([[0.0, 1.0], [1e7, 1.0]])
                        sc = it["scale"] if e.kind != "force" else 1.0
                        amps.append(f"*AMPLITUDE, NAME=EA{r}\n" + "".join(f"{_fmt(x)}, {_fmt(sc * y)}\n" for x, y in tab))
                        names.append(f"EA{r}")
                    else:
                        names.append("")
                    if abs(it["phase"]) > 1e-9:
                        raise ValueError("reference: excitation phase shift is not converted")
                step = (f"*STEP\n*STEADY STATE DYNAMICS, SOLVER=PARDISO\n{_fmt(spec.f_min)}, {_fmt(spec.f_max)}, "
                        f"{int(spec.points)}, {_fmt(spec.bias)}\n" + _damping(zeta)
                        + _loads(terms, study, mesh, False, names, None) + "*NODE PRINT, NSET=NPROBE\nU\n*END STEP\n")
            else:
                t = rr["t"]
                for r, it in enumerate(terms.items):
                    e = it["exc"]
                    if e.kind == "force":
                        tab = np.asarray(e.table, float) if e.table else np.array([[0.0, 1.0], [t[-1], 1.0]])
                        amps.append(f"*AMPLITUDE, NAME=EA{r}\n" + "".join(f"{_fmt(x)}, {_fmt(y)}\n" for x, y in tab))
                    else:                                  # displacement of the base (u base_acc by double integration)
                        comp = rr["ub"][:, int(np.argmax(np.abs(it["dir"])))]
                        amps.append(f"*AMPLITUDE, NAME=EA{r}\n" + "".join(f"{_fmt(x)}, {_fmt(y)}\n" for x, y in zip(t, comp)))
                    names.append(f"EA{r}")
                if sum(it["kind"] != "force" for it in terms.items) > 1:
                    raise ValueError("reference: multiple base excitations in time are not converted")
                step = (f"*STEP, INC={len(t) + 10}\n*MODAL DYNAMIC, SOLVER=PARDISO\n{_fmt(t[1] - t[0])}, {_fmt(t[-1])}\n"
                        + _damping(zeta) + _loads(terms, study, mesh, True, names, None)
                        + "*NODE PRINT, NSET=NPROBE\nU\n*END STEP\n")
        except ValueError as e:
            rep[kind] = {"skipped": str(e)}
            continue
        p = Path(workdir) / f"{job}.inp"
        text = p.read_text(encoding="ascii", errors="replace")
        i = text.find("*STEP")
        text = text[:i] + "".join(amps) + text[i:] + step
        p.write_text(text, encoding="ascii", errors="replace")
        log(f"Dynamics: ccx reference ({('*STEADY STATE DYNAMICS' if kind == 'harmonic' else '*MODAL DYNAMIC')})")
        r_ = run_ccx(Path(workdir), job, exe=exe, on_line=on_line, cancel=cancel)
        if not r_.finished or r_.errors:
            rep[kind] = {"error": "; ".join(r_.errors or [f"code {r_.returncode}"])}
            continue
        blocks = read_print(Path(workdir) / f"{job}.dat")
        rep[kind] = _compare(kind, blocks, nodes, basis, out, rr, terms)
    return rep


def _compare(kind, blocks, nodes, basis, out, rr, terms) -> dict:
    """Python × ccx in the nodes of the probes: relative error to the maximum of the response of the node."""
    res = out.residual
    nidx = np.array(nodes)
    phi = basis.shapes[:, nidx, :]                         # (K, p, 3)
    if kind == "harmonic":
        # pair of blocks (Re, Im) at the frequency
        freqs = [blocks[i][0] for i in range(0, len(blocks) - 1, 2)]
        ccx = np.array([[blocks[i][1][n + 1] + 1j * blocks[i + 1][1][n + 1] for n in nodes]
                        for i in range(0, len(blocks) - 1, 2)])
        f = np.array(freqs)
        hr = R.harmonic(basis, terms, f, out.zeta, None, residual=False)
        q = hr["q"]
        if any(it["kind"] != "force" for it in terms.items):
            # ccx damps the ABSOLUTE velocity in *BASE MOTION (C·r·u̇_b ≠ 0 for the modal damping): p_j = −Γ_j·a_b·(1 −
            # 2iζ_jω_j/ω) – verified 15. 10. 2026 (scratch/f8_dyn/dbg_base.py, agreement 4.6e-6); Python damps the relative
            # motion (physically – the material damping does not damp the rigid motion) → for the comparison the convention of ccx
            H = R.frf(f, basis.freq, out.zeta)
            w = 2 * np.pi * f
            for it in terms.items:
                if it["kind"] == "force":
                    continue
                e = it["exc"]
                amp = R.table_fn(e.table)(f) * it["scale"]
                if e.kind == "base_disp":
                    amp = -(w ** 2) * amp
                q = q + amp[:, None] * it["p"][None, :] * H * (-2j * out.zeta[None, :] * basis.omega[None, :]
                                                                / w[:, None])
        py = np.einsum("fk,kpi->fpi", q, phi)
        # ccx SSD with the acceleration of the base writes the ABSOLUTE displacements (trial 2a) → + displacement of the base
        if any(it["kind"] != "force" for it in terms.items):
            py = py + hr["ub"][:, None, :]
        scale = np.abs(ccx).max(axis=(0, 2))
        err = np.abs(py - ccx).max(axis=(0, 2)) / np.maximum(scale, 1e-300)
        return {"frequencies": len(f), "nodes": [n + 1 for n in nodes], "max_rel_error": float(err.max()),
                "per_node": err.tolist(), "note": "without static correction (ccx SSD does not have it)"}
    t_ccx = np.array([b[0] for b in blocks])
    ccx = np.array([[b[1][n + 1] for n in nodes] for b in blocks])
    t = rr["t"]
    # ccx *MODAL DYNAMIC with *BASE MOTION damps the RELATIVE motion (unlike SSD) – agreement without changes
    q = rr["q"]
    note = "without static correction (ccx *MODAL DYNAMIC does not have it)"
    if any(it["kind"] == "base_disp" for it in terms.items):
        # ccx *MODAL DYNAMIC approximates the displacement of the base by the second central difference from the table.
        # For an exact comparison with the etalon of ccx we recompute q in the convention of ccx.
        from .modal_ode import integrate as ode_integrate
        P_ccx, _, _, _ = R.time_terms(terms, t, ccx_convention=True)
        q, _, _ = ode_integrate(t[1] - t[0], P_ccx, basis.omega, out.zeta)
        note += "; ccx convention used for base_disp (second central difference)"
    py = np.einsum("tk,kpi->tpi", q, phi)
    if any(it["kind"] != "force" for it in terms.items):
        py = py + rr["ub"][:, None, :]
    pyi = np.stack([np.stack([np.interp(t_ccx, t, py[:, p, c]) for c in range(3)], -1) for p in range(len(nodes))], 1)
    scale = np.abs(ccx).max(axis=(0, 2))
    err = np.abs(pyi - ccx).max(axis=(0, 2)) / np.maximum(scale, 1e-300)
    return {"times": len(t_ccx), "nodes": [n + 1 for n in nodes], "max_rel_error": float(err.max()),
            "per_node": err.tolist(), "note": note}
