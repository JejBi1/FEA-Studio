"""Cross-check of linear results in MYSTRAN (Nastran BDF, SOL 101).

The same mesh and the same nodal forces as for ccx (consistent loads from loads.nodal_loads),
the stiffness assembly, boundary conditions and the solver are independent. The node order of CTETRA(10)
in Nastran is the same as the C3D10 of ccx.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .loads import nodal_loads, global_vector, remote_vectors
from .mesh import Mesh
from .study import Study, Fixed, Gravity, RemoteLoad

MYSTRAN_EXE = config.PROJECT_ROOT / "tools" / "mystran" / ("mystran" + config.EXE)


class MystranError(RuntimeError):
    pass


def _f(v: float) -> str:
    """Real number in the BDF field: always with a decimal point (MYSTRAN rejects '10' as real), ≤ 16 characters."""
    if v == 0:
        return "0."
    s = f"{v:.9g}"
    if "." not in s:
        s = s.replace("e", ".e") if "e" in s else s + "."
    return s


def _card(name: str, fields: list) -> list[str]:
    """Free-format card with continuation: max. 8 data fields per line (MYSTRAN drops the rest)."""
    f = ["" if v is None else str(v) for v in fields]
    out = []
    for i in range(0, max(len(f), 1), 8):
        chunk = f[i:i + 8]
        head = name if i == 0 else "+"
        tail = ",+" if i + 8 < len(f) else ""
        out.append(",".join([head] + chunk) + tail)
    return out


@dataclass
class BdfMap:
    ref_nodes: dict[int, tuple[int, RemoteLoad]] = field(default_factory=dict)
    subcases: list[str] = field(default_factory=list)


def write_bdf(study: Study, mesh: Mesh, path: str | Path) -> BdfMap:
    """Writes the BDF in free format (commas). Node i = GRID i+1, element e = CTETRA e+1."""
    path = Path(path)
    bm = BdfMap()
    n = len(mesh.nodes)
    lines = ["SOL 101", "CEND", "TITLE = FEA Studio – MYSTRAN check", "ECHO = NONE",
             "DISPLACEMENT(PLOT) = ALL", "SPCFORCES(PLOT) = ALL", "STRESS(PLOT) = ALL", "SPC = 1"]
    for k, lc in enumerate(study.load_cases):
        lines += [f"SUBCASE {k + 1}", f"  LABEL = LC{k + 1}", f"  LOAD = {k + 1}"]
        bm.subcases.append(lc.name)
    lines += ["BEGIN BULK", "PARAM,POST,-1"]
    # what would be converted silently differently (isotropic / without bolts) → better an error than a wrong check
    if study.active_bolts() or study.active_welds():
        raise MystranError("Bolts and welds (U1 beams, split interface) are not converted for the MYSTRAN check")
    from .printing import shell_of
    if mesh.report.get("ties"):
        raise MystranError("Parts connected by *TIE are not yet converted for the MYSTRAN check (RBE3 / MPC)")
    if any(study.material_of(p).is_ortho for p in mesh.part_names):
        raise MystranError("Orthotropic material is not yet converted for the MYSTRAN check (MAT9 + CORD)")
    if shell_of(mesh) is not None or any(study.material_of(p).is_print for p in mesh.part_names):
        raise MystranError("3D printing (orthotropy, shell + infill) is not converted for the MYSTRAN check")
    if study.has_thermal_loads() or study.thermal.cases:
        raise MystranError("Heat (thermal cases, thermal loads) is not converted for the MYSTRAN check")
    for i, (x, y, z) in enumerate(mesh.nodes):
        lines.append(f"GRID,{i + 1},,{_f(x)},{_f(y)},{_f(z)}")
    mats = {}
    for pi, pname in enumerate(mesh.part_names):
        m = study.material_of(pname)
        mid = mats.setdefault(m.name, len(mats) + 1)
        lines.append(f"PSOLID,{pi + 1},{mid}")
    for name, mid in mats.items():
        m = study.materials[name]
        lines.append(f"MAT1,{mid},{_f(m.E)},,{_f(m.nu)},{_f(m.rho)}")
    for e, conn in enumerate(mesh.tets):
        c = [str(int(v) + 1) for v in conn]
        pid = int(mesh.tet_part[e]) + 1
        if len(c) == 10:
            lines.append(f"CTETRA,{e + 1},{pid},{','.join(c[:6])},+")
            lines.append(f"+,{','.join(c[6:])}")
        else:
            lines.append(f"CTETRA,{e + 1},{pid},{','.join(c)}")
    # constraints: zero displacements SPC1, non-zero SPC with a value D (MYSTRAN: SPC, SID, G, C, D – forced displacement)
    for c in study.constraints:
        nodes = mesh.surface_nodes(c.selection) + 1
        if getattr(c, "kind", "") == "cylindrical":
            raise MystranError("Cylindrical support is not yet converted for the MYSTRAN check (CORD2C/CD would also change other supports at shared nodes – MPC needed)")
        if getattr(c, "kind", "") == "remote_support":
            raise MystranError("Remote support is not yet converted for the MYSTRAN check (RBE2/RBE3 + SPC at the GRID point)")
        if getattr(c, "kind", "") == "elastic":
            raise MystranError("Elastic support is not yet converted for the MYSTRAN check (CELAS)")
        if isinstance(c, Fixed):
            comps = {"123": 0.0}
        else:
            comps = {str(d + 1): v for d, v in enumerate((c.ux, c.uy, c.uz)) if v is not None}
        for comp, val in comps.items():
            if val:
                for g in nodes:                    # SPC: 2 trojice (G, C, D) na kartu
                    lines.append(f"SPC,1,{int(g)},{comp},{_f(val)}")
            else:
                for i in range(0, len(nodes), 6):
                    lines.append(f"SPC1,1,{comp}," + ",".join(str(int(g)) for g in nodes[i:i + 6]))
    # remote points: a new GRID + RBE3 (distributing) or RBE2 (rigid)
    next_grid = n + 1
    next_elem = len(mesh.tets) + 1
    remote_by_case = {}
    for k, lc in enumerate(study.load_cases):
        for l in lc.loads:
            if isinstance(l, RemoteLoad):
                g = next_grid
                next_grid += 1
                x, y, z = l.point
                lines.append(f"GRID,{g},,{_f(x)},{_f(y)},{_f(z)}")
                surf = [int(s) for s in mesh.surface_nodes(l.selection) + 1]
                if l.coupling == "rigid":
                    lines += _card("RBE2", [next_elem, g, 123] + surf)
                else:
                    # weights as the ccx *DISTRIBUTING on a face: ∫N_i dA (tri6: corners 0 → leave out)
                    from .ccx.inp import consistent_nodal_forces
                    w = consistent_nodal_forces(mesh.nodes, mesh.surfaces[l.selection], np.array([1.0, 0, 0]))
                    groups = []
                    for node, fv in sorted(w.items()):
                        if fv[0] > 0:
                            groups += [_f(float(fv[0])), 123, node + 1]
                    lines += _card("RBE3", [next_elem, None, g, 123456] + groups)
                next_elem += 1
                bm.ref_nodes[g] = (k, l)
                remote_by_case.setdefault(k, []).append((g, l))
    # loads per state (consistent nodal forces, gravity GRAV)
    for k, lc in enumerate(study.load_cases):
        sid = k + 1
        n_before = len(lines)
        no_grav = type(lc)(lc.name, [l for l in lc.loads if not isinstance(l, (Gravity, RemoteLoad))])
        F, _, _ = nodal_loads(study, mesh, no_grav)
        for i in np.nonzero(np.abs(F).sum(axis=1) > 0)[0]:
            fx, fy, fz = F[i]
            mag = float(np.linalg.norm(F[i]))
            lines.append(f"FORCE,{sid},{i + 1},,{_f(mag)},{_f(fx / mag)},{_f(fy / mag)},{_f(fz / mag)}")
        for l in lc.loads:
            if isinstance(l, Gravity):
                g = global_vector(study, l)
                gn = float(np.linalg.norm(g))
                lines.append(f"GRAV,{sid},,{_f(gn)},{_f(g[0] / gn)},{_f(g[1] / gn)},{_f(g[2] / gn)}")
        for gid, l in remote_by_case.get(k, []):
            f, m = remote_vectors(study, l)
            if np.linalg.norm(f) > 0:
                lines.append(f"FORCE,{sid},{gid},,{_f(np.linalg.norm(f))},{','.join(_f(v) for v in f / np.linalg.norm(f))}")
            if np.linalg.norm(m) > 0:
                lines.append(f"MOMENT,{sid},{gid},,{_f(np.linalg.norm(m))},{','.join(_f(v) for v in m / np.linalg.norm(m))}")
        # empty state: zero force, so that LOAD refers to an existing set
        if len(lines) == n_before:
            lines.append(f"FORCE,{sid},1,,0.,1.,0.,0.")
    lines.append("ENDDATA")
    path.write_text("\n".join(lines) + "\n", encoding="ascii", errors="replace")
    return bm


@dataclass
class MystranResult:
    U: list[np.ndarray]            # posuvy po stavech (N, 3)
    spc_force: list[np.ndarray]    # reactions per state (N, 3) (only constrained nodes non-zero)
    log: Path


def run_mystran(workdir: str | Path, job: str = "mystran", timeout: float | None = 3600) -> Path:
    workdir = Path(workdir)
    if not MYSTRAN_EXE.exists():
        raise MystranError(f"MYSTRAN not found: {MYSTRAN_EXE}")
    from . import dllguard
    dllguard.check_workdir(workdir, "MYSTRAN")      # runs with cwd = workdir: no planted DLLs
    log = workdir / f"{job}.mystran.log"
    with open(log, "w", encoding="utf-8", errors="replace") as fl:
        p = subprocess.run([str(MYSTRAN_EXE), f"{job}.bdf"], cwd=workdir, stdout=fl, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, timeout=timeout, creationflags=config.NO_WINDOW)
    f06 = workdir / f"{job}.F06"
    text = f06.read_text(encoding="latin-1", errors="replace") if f06.exists() else ""
    if p.returncode != 0 or "FATAL" in text.upper():
        fatal = [ln.strip() for ln in text.splitlines() if "FATAL" in ln.upper()][:5]
        raise MystranError("MYSTRAN ended with an error:\n  " + "\n  ".join(fatal or [f"code {p.returncode}"])
                           + f"\nlog: {log}")
    return workdir / f"{job}.OP2"


def read_op2(op2: Path, n_nodes: int, n_cases: int) -> MystranResult:
    from pyNastran.op2.op2 import OP2
    m = OP2(debug=False, log=None)
    m.set_mode("nx")
    m.read_op2(str(op2), build_dataframe=False)
    U, R = [], []
    for k in range(1, n_cases + 1):
        u = np.zeros((n_nodes, 3))
        d = m.displacements.get(k)
        if d is None:
            raise MystranError(f"Displacements of case {k} are missing in OP2")
        ids = d.node_gridtype[:, 0]
        mask = ids <= n_nodes
        u[ids[mask] - 1] = d.data[0, mask, :3]
        U.append(u)
        r = np.zeros((n_nodes, 3))
        s = m.spc_forces.get(k)
        if s is not None:
            ids = s.node_gridtype[:, 0]
            mask = ids <= n_nodes
            r[ids[mask] - 1] = s.data[0, mask, :3]
        R.append(r)
    return MystranResult(U, R, op2.with_suffix(".F06"))


def prepare(study: Study, results, workdir: str | Path) -> Path:
    """Writes the BDF (for batch / parallel runs: prepare → run_mystran → compare)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    write_bdf(study, results.mesh, workdir / "mystran.bdf")
    return workdir


def compare(study: Study, results, workdir: str | Path) -> list[dict]:
    workdir = Path(workdir)
    mesh = results.mesh
    mr = read_op2(workdir / "mystran.OP2", len(mesh.nodes), len(study.load_cases))
    out = []
    x = mesh.nodes
    Lc = float(np.linalg.norm(x.max(axis=0) - x.min(axis=0)))
    for k, st in enumerate(results.steps):
        Uc, Um = st.U, mr.U[k]
        umax = float(np.abs(Uc).max())
        du = float(np.abs(Uc - Um).max()) / max(umax, 1e-300)
        c0 = st.moment_point if st.moment_point is not None else 0.5 * (x.min(axis=0) + x.max(axis=0))
        rc = np.asarray(st.equilibrium["reaction_total"])
        mc = np.asarray(st.equilibrium["reaction_moment_total"])
        R = mr.spc_force[k]                       # SPC forces = reactions acting on the body
        rm = R.sum(axis=0)
        mm = np.cross(x - c0, R).sum(axis=0)
        # force and moment brought to one scale (moment / characteristic dimension)
        scale = max(float(np.linalg.norm(rc)), float(np.linalg.norm(mc)) / Lc, 1e-12)
        dr = max(float(np.linalg.norm(rc - rm)), float(np.linalg.norm(mc - mm)) / Lc) / scale
        out.append({"case": st.name, "u_max_ccx": umax, "u_max_mystran": float(np.abs(Um).max()),
                    "du_rel": du, "reaction_ccx": rc.tolist(), "reaction_mystran": rm.tolist(),
                    "moment_ccx": mc.tolist(), "moment_mystran": mm.tolist(), "dr_rel": dr})
    return out
