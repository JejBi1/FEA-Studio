"""Frame system in MYSTRAN (CBAR + PBAR with A, I1, I2, J): linear buckling (SOL 105) and cross-check of the statics.

ccx U1 cannot buckle (the stress stiffness is missing, "no buckling") → the critical load factor of the system α_cr (EN 1993-1-1,
5.2.1) and the buckling shapes are computed by MYSTRAN (MIT, open source; verified 14. 10. 2026 scratch/beams_fem/mys_buckle.py:
Euler cantilever λ₁ = 183.9448 = π²EI/(4L²)/F exactly). CBAR: the orientation vector v = axis z of the section (our "up") → the plane
of 1 element = x–z of the section → I1 = Iy, I2 = Iz; releases of the ends PA/PB in the axes of the element (y_el = z, z_el = −y): N→1, Vz→2,
Vy→3, T→4, Mz→5, My→6. Distributed loads and weight = consistent nodal forces (fem.nodal_loads). Only the frame system alone
(connection to a body → error).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .fem import FrameError, build, nodal_loads
from ..mystran import _f, run_mystran, MystranError
from ..ccx import deck_text

PIN_MAP = {0: 1, 2: 2, 1: 3, 3: 4, 5: 5, 4: 6}        # our component (N Vy Vz T My Mz) → DOF of the CBAR element


def write_bdf(study, fe, path: Path, sol: str, lc=None, nd: int | None = None) -> dict:
    """BDF of the frame system (nodes without the auxiliary B'/E – CBAR has J and the releases of the ends directly)."""
    nd = nd or int(study.buckling.modes)                # number of buckling modes
    if study.frame.connections:
        raise FrameError("Buckling / MYSTRAN check: frame structure only (without connection to a body)")
    fr = study.frame
    used: dict[int, int] = {}                          # FE node index → GRID
    elems = []
    for mname, segs in fe.member_segs.items():
        m = fr.members[mname]
        for k, i in enumerate(segs):
            sg = fe.segs[i]
            a, b = sg.a, sg.b
            # ends of a bar with a release: in MYSTRAN a node of the system + pin flag (the E nodes do not exist)
            pa = pb = ""
            if k == 0 and any(m.releases_start) or (k == 0 and m.kind == "truss"):
                a = fe.joint[m.start]
                rel = m.releases_start if m.kind != "truss" else [False, False, False, False, True, True]
                pa = "".join(str(PIN_MAP[d]) for d in range(6) if rel[d])
            if k == len(segs) - 1 and (any(m.releases_end) or m.kind == "truss"):
                b = fe.joint[m.end]
                rel = m.releases_end if m.kind != "truss" else [False, False, False, True, True, True]
                pb = "".join(str(PIN_MAP[d]) for d in range(6) if rel[d])
            for n_ in (a, b):
                used.setdefault(n_, len(used) + 1)
            elems.append((sg, used[a], used[b], pa, pb))
    lines = [f"SOL {sol}", "CEND", f"TITLE = {deck_text(study.name)}", "ECHO = NONE",
             "SPC = 1", "DISPLACEMENT(PLOT) = ALL", "SPCFORCES(PLOT) = ALL"]
    cases = [lc] if lc is not None else list(study.load_cases)
    for k, c in enumerate(cases):
        lines += [f"SUBCASE {k + 1}", f"  LOAD = {k + 1}"]
    if sol == "105":
        lines += [f"SUBCASE {len(cases) + 1}", "  METHOD = 10", "  STATSUB = 1"]
    lines.append("BEGIN BULK")
    lines.append("PARAM,POST,-1")
    X = np.array(fe.xyz)
    for n_, g in used.items():
        lines.append(f"GRID,{g},,{_f(X[n_, 0])},{_f(X[n_, 1])},{_f(X[n_, 2])}")
    mats, props = {}, {}
    for e, (sg, ga, gb, pa, pb) in enumerate(elems, 1):
        mk = (sg.E, sg.G, sg.rho)
        if mk not in mats:
            mats[mk] = len(mats) + 1
            lines.append(f"MAT1,{mats[mk]},{_f(sg.E)},{_f(sg.G)},,{_f(sg.rho)}")
        p = sg.sec
        pk = (mats[mk], p.A, p.Iy, p.Iz, p.J)
        if pk not in props:
            props[pk] = len(props) + 1
            lines.append(f"PBAR,{props[pk]},{mats[mk]},{_f(p.A)},{_f(p.Iy)},{_f(p.Iz)},{_f(p.J)}")
        v = sg.R[2]
        card = f"CBAR,{e},{props[pk]},{ga},{gb},{_f(v[0])},{_f(v[1])},{_f(v[2])}"
        if pa or pb:
            card += f",+\n+,{pa},{pb}"
        lines.append(card)
    spc = {}
    for n_, d in fe.bc:
        if n_ in used:
            spc.setdefault(used[n_], "")
            spc[used[n_]] += str(d)
    for g, comp in spc.items():
        lines.append(f"SPC1,1,{''.join(sorted(set(comp)))},{g}")
    for k, (n_, d, kk) in enumerate(fe.springs, 1):
        if n_ in used:
            lines.append(f"CELAS2,{100000 + k},{_f(kk)},{used[n_]},{d}")
        else:
            raise FrameError("Rotational springs of the support are not yet converted for MYSTRAN")
    # unused rotations of the nodes with hinges of all bars (MYSTRAN would have a singularity) – AUTOSPC
    lines.append("PARAM,AUTOSPC,Y")
    from .timoshenko import TimModel                    # loads with condensation of the hinges (EB, as CBAR without shear)
    tm = TimModel(study, fe, shear=False)
    for k, c in enumerate(cases, 1):
        Fn, f_eq, _, _ = tm.case_loads(c)
        F = (tm.scatter(tm.reduce_loads(f_eq)[0]) + Fn.ravel()).reshape(-1, 6)
        for n_ in np.nonzero(np.abs(F).max(axis=1) > 0)[0]:
            if n_ not in used:
                continue
            f_, m_ = F[n_, :3], F[n_, 3:]
            if np.any(f_):
                lines.append(f"FORCE,{k},{used[n_]},0,1.,{_f(f_[0])},{_f(f_[1])},{_f(f_[2])}")
            if np.any(m_):
                lines.append(f"MOMENT,{k},{used[n_]},0,1.,{_f(m_[0])},{_f(m_[1])},{_f(m_[2])}")
    if sol == "105":
        # Lanczos (EIGRL); MGIV for a planar frame omitted the 1st mode with sway (verified portal_buckle.py)
        lines.append(f"EIGRL,10,,,{int(nd)}")
    lines.append("ENDDATA")
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii", errors="replace")
    return {"grid": used, "cases": [c.name for c in cases]}


BUCKLE_SEGMENTS = (4, 6)        # subdivision for buckling: 4 cubic elements per bar are enough (Euler, sway of the frame < 0.2 %)


def _buckle_once(study, wd: Path, lc, segments: int) -> tuple[list[float], np.ndarray, object, dict]:
    import copy
    st = copy.deepcopy(study)
    st.frame.settings.segments_straight = segments
    st.frame.settings.max_arc_angle = max(st.frame.settings.max_arc_angle, 15.0)
    fe = build(st, solver="timoshenko", segments=segments)         # nodes of the system and internal nodes of the bars (without the auxiliary nodes of U1)
    wd.mkdir(parents=True, exist_ok=True)
    mp = write_bdf(st, fe, wd / "frame_buckle.bdf", "105", lc)
    op2 = run_mystran(wd, "frame_buckle")
    text = (wd / "frame_buckle.F06").read_text(encoding="latin-1", errors="replace")
    i = text.find("buckling load factors")
    factors = []
    if i >= 0:
        for line in text[i:].splitlines()[1:]:
            parts = line.split()
            if len(parts) == 3 and parts[0].isdigit():
                factors.append(float(parts[2].rstrip("*")))     # MYSTRAN marks negative values with "*"
            elif factors and (not line.strip() or len(parts) != 3):
                break
    return factors, op2, fe, mp


def buckling(study, workdir, lc=None) -> dict:
    """Critical load factors of the frame system (MYSTRAN SOL 105, Lanczos) and shapes.

    Verified 14. 10. 2026: Euler (cantilever, pinned column) exactly, planar portal with sway −0.16 %. CAUTION: with a fine
    subdivision (10–20 elements per bar) the Lanczos of MYSTRAN for spatial frames got stuck at the shift −1 (cluster λ ≈ −1,
    zero subspace of the stress stiffness) and MGIV omitted the 1st mode → buckling on its own subdivision of 4 elements per bar,
    check with 6 elements; a difference > 3 % or λ ≈ −1 → warning "unreliable". CBAR has no lateral-torsional / spatial buckling with
    warping (Wagner terms) – those are assessed by EN 1993-1-1 (6.3) in the assessment of the bars.
    """
    if lc is None:
        names = [c.name for c in study.load_cases]
        k = names.index(study.buckling.case) if study.buckling.case in names else 0
        lc = study.load_cases[k]
    wd = Path(workdir)
    f1, op2, fe, mp = _buckle_once(study, wd / "d4", lc, BUCKLE_SEGMENTS[0])
    f2, *_ = _buckle_once(study, wd / "d6", lc, BUCKLE_SEGMENTS[1])
    if not f1:
        raise MystranError("MYSTRAN did not return buckling factors (F06)")
    warnings = []
    pos1 = [v for v in f1 if v > 0]
    pos2 = [v for v in f2 if v > 0]
    spurious = any(abs(v + 1.0) < 2e-3 for v in f1 + f2)
    if spurious or not pos1 or not pos2 or abs(pos1[0] - pos2[0]) > 0.03 * abs(pos1[0]):
        warnings.append(f"Frame buckling (MYSTRAN): the result is unreliable – the factors differ between 4 and 6 elements per member ({pos1[:1]} × {pos2[:1]}) or a non-physical mode λ ≈ −1 appeared. Assess stability per EN 1993-1-1 (member buckling, lateral-torsional buckling).")
    shapes = np.zeros((len(f1), len(fe.xyz), 3), np.float32)
    try:
        from pyNastran.op2.op2 import OP2
        o = OP2(debug=False, log=None)
        o.set_mode("nx")
        o.read_op2(str(op2), build_dataframe=False)
        ev = o.eigenvectors.get(2) or next(iter(o.eigenvectors.values()))
        inv = {g: n_ for n_, g in mp["grid"].items()}
        ids = ev.node_gridtype[:, 0]
        for j in range(min(len(f1), ev.data.shape[0])):
            u = np.zeros((len(fe.xyz), 3))
            for g, row in zip(ids, ev.data[j]):
                if g in inv:
                    u[inv[g]] = row[:3]
            mx = np.abs(u).max()
            shapes[j] = u / mx if mx > 0 else u
    except Exception as e:  # noqa: BLE001 – shapes only for display; the buckling factors stay valid
        warnings.append(f"Buckling mode shapes could not be read from the MYSTRAN output ({e}) – factors only, no shapes")
    return {"factors": f1, "factors_check": f2, "frame_shapes": shapes, "case": lc.name, "fe": fe,
            "warnings": warnings, "xyz": [list(map(float, p)) for p in fe.xyz]}


def static_check(study, results, workdir) -> list[dict]:
    """Cross-check of the statics of the frame system in MYSTRAN: max. difference of the displacements of the nodes of the system and of the reactions."""
    fe = build(study, solver="timoshenko")
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    mp = write_bdf(study, fe, wd / "frame_static.bdf", "1")
    op2 = run_mystran(wd, "frame_static")
    from pyNastran.op2.op2 import OP2
    o = OP2(debug=False, log=None)
    o.set_mode("nx")
    o.read_op2(str(op2), build_dataframe=False)
    out = []
    for k, st in enumerate(results.steps, 1):
        d = o.displacements.get(k)
        U = {int(g): row for g, row in zip(d.node_gridtype[:, 0], d.data[0])}
        du, umax = 0.0, 1e-30
        for name, j in fe.joint.items():
            if j in mp["grid"]:
                uc = np.asarray(st.frame["nodes"][name][:3])
                um = np.asarray(U[mp["grid"][j]][:3])
                du = max(du, float(np.abs(uc - um).max()))
                umax = max(umax, float(np.abs(uc).max()))
        out.append({"case": st.name, "du_rel": du / umax, "u_max_ccx": umax})
    return out
