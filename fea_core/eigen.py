"""Natural frequencies (ccx *FREQUENCY) and linear buckling (ccx *BUCKLE) on the same mesh as the statics.

- Natural frequencies: the constraints of the study (without them a free body – 6 zero rigid-body modes), mass from the
  densities of the materials (U1 bolts as steel). Output: frequencies [Hz], shapes normalised to max |u| = 1, participation
  factors and effective masses per direction (x, y, z, rx, ry, rz) from the ccx .dat, total mass.
- Buckling: the load of one load state (also thermal); factor λ = multiple of the load at loss of stability
  (linear – only stress stiffness, no imperfections and plasticity; the real capacity is usually lower). Negative λ = loss of
  stability under the reversed load.
Verified by trial (scratch/modal/exp1.py, 14. 10. 2026): cantilever V1 f₁ = 209.45 Hz (Euler–Bernoulli 208.9),
buckling by a compressive load at the end λ·1000 N = 21.65 kN (Euler β = 2: 21.59 kN). Shapes in the .frd: a block per mode, mode number
in NUMSTP of the 100C header (`ResultBlock.mode`); for *BUCKLE the first block is the static state (value 0).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ModalSettings:
    enabled: bool = False
    modes: int = 10                      # number of natural frequencies
    f_min: float = 0.0                   # lower bound of the search [Hz]
    f_max: float | None = None           # upper bound [Hz] (None = no bound)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict | None) -> "ModalSettings":
        from .study import _known
        return _known(cls, d)


@dataclass
class BucklingSettings:
    enabled: bool = False
    case: str | None = None              # load state (None = first)
    modes: int = 4

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict | None) -> "BucklingSettings":
        from .study import _known
        return _known(cls, d)


DOF_LABELS = ("x", "y", "z", "rx", "ry", "rz")


@dataclass
class ModeSet:
    """Eigenvalue result: kind "modal" (values = frequencies [Hz]) or "buckling" (values = factors λ)."""
    kind: str
    name: str
    values: list[float]
    shapes: np.ndarray                    # (k, N, 3) float32, max |u| = 1
    case: str | None = None               # buckling: load state
    participation: list[list[float]] = field(default_factory=list)   # (k, 6)
    eff_mass: list[list[float]] = field(default_factory=list)        # (k, 6) [t], rotation [t·mm²]
    total_mass: list[float] = field(default_factory=list)            # (6,) total effective mass [t]
    frame_shapes: np.ndarray | None = None                           # (k, n_FE_members, 3) shapes of the members (max = 1)

    def summary(self) -> dict:
        d = {"kind": self.kind, "name": self.name, "case": self.case}
        if self.kind == "modal":
            d["frequencies_Hz"] = [float(v) for v in self.values]
            if getattr(self, "nodal_diameters", None) is not None:      # cyclic symmetry
                d["nodal_diameters"] = list(self.nodal_diameters)
                d["multiplicity"] = list(getattr(self, "multiplicity", []))
            if self.eff_mass and self.total_mass:
                tot = np.asarray(self.total_mass, float)
                frac = np.asarray(self.eff_mass, float)[:, :3] / np.where(tot[:3] > 0, tot[:3], 1.0)
                d["effective_mass_fraction_xyz"] = np.round(frac, 4).tolist()
                d["cumulative_mass_fraction_xyz"] = np.round(frac.sum(axis=0), 4).tolist()
                d["total_mass_kg"] = float(tot[0] * 1000.0)
        elif self.kind == "nlbuckle":                # nonlinear buckling (fea_core.imperfect): λ of the stored increments
            d["frame_load_factors"] = [float(v) for v in self.values]
        else:
            d["buckling_factors"] = [float(v) for v in self.values]
        return d


class EigenError(RuntimeError):
    pass


def check(study, part_names) -> tuple[list[str], list[str]]:
    """Check of the settings of natural frequencies and buckling."""
    errors, warnings = [], []
    excl = study.excluded_parts()
    if study.modal.enabled:
        if int(study.modal.modes) < 1:
            errors.append("Natural frequencies: the number of modes must be ≥ 1")
        if study.modal.f_max is not None and study.modal.f_max <= study.modal.f_min:
            errors.append("Natural frequencies: the upper limit must be greater than the lower limit")
        for p in part_names:
            if p in excl:
                continue
            try:
                if not study.material_of(p).rho > 0:
                    errors.append(f"Natural frequencies: the material of part '{p}' has no density ρ")
            except KeyError:
                pass
        if not study.constraints and not (study.frame.active and study.frame.supports):
            warnings.append("Natural frequencies without supports: free body – the first 6 frequencies ≈ 0 Hz (rigid-body motions)")
    if study.buckling.enabled:
        if not study.load_cases:
            errors.append("Buckling: the study has no load case (buckling is computed from the load case loading)")
        elif study.buckling.case is not None and study.buckling.case not in [lc.name for lc in study.load_cases]:
            errors.append(f"Buckling: load case '{study.buckling.case}' does not exist")
        if int(study.buckling.modes) < 1:
            errors.append("Buckling: the number of modes must be ≥ 1")
        if not study.constraints and not (study.frame.active and study.frame.supports):
            errors.append("Buckling: the study has no supports")
        warnings.append("Buckling is linear (no imperfections or plasticity) – the actual resistance is often lower than λ·load; for design use a code procedure (e.g. EN 1993-1-1, 6.3) or a nonlinear analysis")
    return errors, warnings


def _sub_study(study, cases):
    """Shallow copy of the study with the given load states and without the bolt preload state."""
    st = copy.copy(study)
    st.load_cases = list(cases)
    st.joints = copy.copy(study.joints)
    st.joints.preload_case = False
    return st


_NUM = r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?"


def _table(text: str, title: str, ncol: int) -> tuple[dict[int, list[float]], list[float] | None]:
    """Rows of the .dat table after the heading: {mode number: ncol numbers} and the TOTAL row (| None). ccx leaves out modes with a
    negative eigenvalue (rigid motions of a free body numerically −1e-4) – therefore by mode number.
    """
    i = text.find(title)
    if i < 0:
        return {}, None
    rows, total = {}, None
    started = False
    for line in text[i + len(title):].splitlines():
        parts = line.split()
        if not parts:
            if started:
                break
            continue
        if parts[0].upper() == "TOTAL" and len(parts) >= ncol + 1:
            total = [float(v.replace("D", "E")) for v in parts[1:ncol + 1]]
            break
        if re.fullmatch(r"\d+", parts[0]) and len(parts) >= ncol + 1:
            try:
                rows[int(parts[0])] = [float(v.replace("D", "E")) for v in parts[1:ncol + 1]]
                started = True
            except ValueError:
                break
        elif started:
            break
    return rows, total


def read_dat(path: Path, kind: str) -> dict:
    """Eigenvalues from the .dat by modes 1…k (a missing mode = negative eigenvalue of a rigid motion → 0 Hz)."""
    text = Path(path).read_text(encoding="latin-1")
    if kind == "modal":
        eig, _ = _table(text, "E I G E N V A L U E   O U T P U T", 4)
        part, _ = _table(text, "P A R T I C I P A T I O N   F A C T O R S", 6)
        mass, _ = _table(text, "E F F E C T I V E   M O D A L   M A S S", 6)
        k = max([*eig, *part, *mass], default=0)
        i = text.find("T O T A L   E F F E C T I V E   M A S S")
        tot = []
        if i >= 0:
            for line in text[i:].splitlines()[1:]:
                nums = re.findall(_NUM, line)
                if len(nums) >= 6 and "COMPONENT" not in line:
                    tot = [float(v.replace("D", "E")) for v in nums[:6]]
                    break
        zero = [0.0] * 6
        return {"freq": [eig[j][2] if j in eig else 0.0 for j in range(1, k + 1)],
                "participation": [part.get(j, zero) for j in range(1, k + 1)],
                "eff_mass": [mass.get(j, zero) for j in range(1, k + 1)], "total": tot}
    fac, _ = _table(text, "B U C K L I N G   F A C T O R   O U T P U T", 1)
    return {"factors": [fac[j][0] for j in sorted(fac)]}


def _shapes(frd_path: Path, n_nodes: int, k: int, skip_first: bool) -> np.ndarray:
    """Mode shapes 1…k (k, N, 3) normalised to max |u| = 1; the mode is taken from NUMSTP of the block (ccx leaves out modes with a negative
    eigenvalue – their shape stays zero). *BUCKLE: block 1 = static state, mode j = block j + 1.
    """
    from .ccx.frd import read_frd
    frd = read_frd(frd_path)
    shift = 1 if skip_first else 0
    by_mode = {b.mode - shift: b for b in frd.blocks if b.name == "DISP"}
    out = np.zeros((k, n_nodes, 3), np.float32)
    for j in range(k):
        b = by_mode.get(j + 1)
        if b is None:
            continue
        msk = b.node_ids <= n_nodes
        u = np.zeros((n_nodes, 3))
        u[b.node_ids[msk] - 1] = b.values[msk, :3]
        m = np.abs(u).max()
        if m > 0:
            u /= m
            if u.ravel()[np.abs(u).argmax()] < 0:      # sign: the largest component positive
                u = -u
        out[j] = u
    return out


def _frd_max(frd_path: Path, n_nodes: int, k: int) -> list[float]:
    """Max |u| of the mode shapes 1…k in the .frd (not normalised – for the common scale with the members)."""
    from .ccx.frd import read_frd
    frd = read_frd(frd_path)
    by = {b.mode: b for b in frd.blocks if b.name == "DISP"}
    return [float(np.abs(by[j + 1].values[:, :3]).max()) if (j + 1) in by else 0.0 for j in range(k)]


def run_eigen(study, mesh, workdir: Path, log, on_line=None, cancel=None, thermal=None) -> list[ModeSet]:
    """Natural frequencies and buckling according to Study.modal / Study.buckling (separate ccx jobs)."""
    from .ccx.inp import write_inp
    from .ccx.runner import run_ccx, CcxError
    from .config import CCX_EXE
    from .study import LoadCase
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    exe = Path(study.solver.exe) if study.solver.exe else CCX_EXE
    n = len(mesh.nodes)

    def run(job: str, cancel_, on_line_) -> None:
        r = run_ccx(workdir, job, exe=exe, on_line=on_line_, cancel=cancel_)
        if not r.finished or r.errors:
            raise CcxError(f"CalculiX ({job}) ended with an error:\n  " + "\n  ".join(r.errors or [f"code {r.returncode}"])
                           + f"\nlog: {r.log_path}")

    def cyclic_modal(cancel_, on_line_) -> ModeSet:
        from .cyclic.modal import run_modal as run_cyclic_modal   # frequencies by nodal diameters (ccx cyclic symmetry)
        return run_cyclic_modal(study, mesh, workdir, log, lambda job: run(job, cancel_, on_line_), _sub_study)

    def modal(cancel_, on_line_) -> ModeSet:
        ms = study.modal
        log(f"Natural frequencies (CalculiX, {int(ms.modes)} modes)")
        st = _sub_study(study, [LoadCase("Natural frequencies", [])])
        line = f"{int(ms.modes)}, {float(ms.f_min or 0.0):.12g}" + (f", {float(ms.f_max):.12g}" if ms.f_max else "")
        try:
            imap = write_inp(st, mesh, workdir / "modal.inp", procedure="modal", eigen_line=line)
        except ValueError as e:
            raise EigenError(str(e)) from None
        run("modal", cancel_, on_line_)
        d = read_dat(workdir / "modal.dat", "modal")
        if not d["freq"]:
            raise EigenError("Natural frequencies: ccx returned no mode (try a different frequency range)")
        shapes = _shapes(workdir / "modal.frd", n, len(d["freq"]), skip_first=False)
        mset = ModeSet("modal", "Natural frequencies", d["freq"][:len(shapes)], shapes,
                       participation=d["participation"], eff_mass=d["eff_mass"], total_mass=d["total"])
        if imap.frame:                                # members: shapes of U1 nodes from the .dat (*NODE PRINT NSET=NFRAME)
            from .beams.fem import read_displacements
            nf = len(imap.frame["fe"].xyz)
            fsh = read_displacements(workdir / "modal.dat", imap.frame["base"], nf)
            fs = np.zeros((len(shapes), nf, 3), np.float32)
            for j, U6 in enumerate(fsh[:len(shapes)]):
                fs[j] = U6[:, :3]
            # common scale with the bodies: the shape of the bodies is normalised to max = 1 → the same factor
            frd_max = _frd_max(workdir / "modal.frd", n, len(shapes))
            for j in range(len(shapes)):
                m_ = max(frd_max[j], float(np.abs(fs[j]).max()))
                if m_ > 0:
                    fs[j] /= m_
                    shapes[j] *= frd_max[j] / m_         # the shape of the bodies was normalised by its maximum only
            mset.frame_shapes = fs
        return mset

    def buckle(cancel_, on_line_) -> ModeSet:
        bs = study.buckling
        names = [lc.name for lc in study.load_cases]
        k = names.index(bs.case) if bs.case in names else 0
        lc = study.load_cases[k]
        log(f"Buckling (CalculiX, case '{lc.name}', {int(bs.modes)} modes)")
        temps = None
        if study.has_thermal_loads() and study.thermal_loads(lc):
            from .thermal.field import node_temperatures
            temps = [node_temperatures(study, mesh, lc, {t.name: t for t in (thermal or [])})]
        st = _sub_study(study, [lc])
        try:
            write_inp(st, mesh, workdir / "buckle.inp", temperatures=temps, procedure="buckle",
                      eigen_line=f"{int(bs.modes)}")
        except ValueError as e:
            raise EigenError(str(e)) from None
        run("buckle", cancel_, on_line_)
        d = read_dat(workdir / "buckle.dat", "buckle")
        if not d["factors"]:
            raise EigenError("Buckling: ccx returned no factor")
        shapes = _shapes(workdir / "buckle.frd", n, len(d["factors"]), skip_first=True)
        return ModeSet("buckling", f"Buckling – {lc.name}", d["factors"][:len(shapes)], shapes, case=lc.name)

    # the two jobs are independent → side by side when the machine has room for both (ccx.concurrent)
    cyclic = getattr(study, "cyclic", None) is not None and study.cyclic.enabled
    tasks = (([cyclic_modal] if cyclic else [modal]) if study.modal.enabled else []) \
        + ([buckle] if study.buckling.enabled and study.load_cases else [])
    from .ccx.concurrent import run_tasks
    out = run_tasks(tasks, 3 * n, cancel, on_line)
    return out


# ----------------------------------------------------------------- storage (Results.save / load)

def save_arrays(modes: list[ModeSet], arrays: dict) -> list[dict]:
    meta = []
    for j, m in enumerate(modes):
        arrays[f"MODE_{j}"] = m.shapes
        if m.frame_shapes is not None:
            arrays[f"MODEF_{j}"] = m.frame_shapes
        own = getattr(m, "frame_shapes_own", None)      # buckling of members (MYSTRAN) on its own subdivision – display in the GUI
        if own is not None:
            arrays[f"MODEFO_{j}"] = np.asarray(own, np.float32)
        meta.append({"kind": m.kind, "name": m.name, "values": list(map(float, m.values)), "case": m.case,
                     "participation": m.participation, "eff_mass": m.eff_mass, "total_mass": m.total_mass,
                     **({"frame_xyz": getattr(m, "frame_xyz", None)} if own is not None else {})})
        if getattr(m, "nodal_diameters", None) is not None:      # cyclic symmetry (fea_core.cyclic.modal)
            arrays[f"MODEI_{j}"] = m.shapes_imag
            meta[-1].update(nodal_diameters=list(m.nodal_diameters), multiplicity=list(m.multiplicity),
                            sectors=int(m.sectors))
    return meta


def load_arrays(meta: list[dict], d: dict) -> list[ModeSet]:
    out = []
    for j, m in enumerate(meta or []):
        ms = ModeSet(m["kind"], m["name"], m["values"], d[f"MODE_{j}"], m.get("case"), m.get("participation") or [],
                     m.get("eff_mass") or [], m.get("total_mass") or [], d.get(f"MODEF_{j}"))
        if f"MODEFO_{j}" in d and m.get("frame_xyz") is not None:
            ms.frame_shapes_own = d[f"MODEFO_{j}"]
            ms.frame_xyz = m["frame_xyz"]
        if m.get("nodal_diameters") is not None and f"MODEI_{j}" in d:
            ms.nodal_diameters = [int(v) for v in m["nodal_diameters"]]
            ms.multiplicity = [int(v) for v in m.get("multiplicity") or [1] * len(ms.nodal_diameters)]
            ms.sectors = int(m.get("sectors") or 0)
            ms.shapes_imag = d[f"MODEI_{j}"]
        out.append(ms)
    return out
