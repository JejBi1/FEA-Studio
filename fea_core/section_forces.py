"""Section force (Free Body Cut) – computation of the resultants of the internal forces and moments in a cut through the body.

According to Abaqus Free Body Cut / ANSYS Force Reaction FREEBODY:
Plane of the cut (point P0, normal n pointing from side '-' to side '+').
Elements with the centroid on side '-' transfer their nodal forces to the nodes of the interface with side '+'.
The result is the components:
- normal force N
- shear forces Vy, Vz
- torsional moment Mx
- bending moments My, Mz
related to the centroid of the cut in the local coordinate system of the cut (axis x = normal of the cut n).
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .study import Study
from .results import StepResult
import io
import logging
import re
from pathlib import Path
from .recovery import element_nodal_forces
from .joints.forces import element_nodal_forces as joints_nodal_forces


ALL_ELEMENTS_MAX = 200_000      # up to this many elements the extra ccx run prints S of ALL elements (later cuts need no ccx)
_OUTPUT_CARDS = ("*NODE FILE", "*EL FILE", "*NODE PRINT", "*EL PRINT", "*CONTACT FILE", "*CONTACT PRINT",
                 "*NODE OUTPUT", "*ELEMENT OUTPUT", "*SECTION PRINT")
_DAT_SETS = ("ESECTION", "EALL", "ECUT", "EJOINT")
_HEAD_SET = re.compile(rb"stresses \(elem, integ\.pnt\.,sxx,syy,szz,sxy,sxz,syz\) for set (\S+)", re.IGNORECASE)


def _dat_prints(path: Path, sets=_DAT_SETS) -> dict[str, list[dict[int, np.ndarray]]]:
    """*EL PRINT S blocks of the sets present in one .dat (the file is read once): set -> [step -> {element: (nip, 6)}]."""
    from .joints.forces import _el_print_fast, _read_el_print_lines
    buf = path.read_bytes()
    present = {m.group(1).decode("latin-1").upper() for m in _HEAD_SET.finditer(buf)}
    out = {}
    for eset in sets:
        if eset not in present:
            continue
        try:
            r = _el_print_fast(buf, eset)
        except ValueError:
            r = None
        out[eset] = r if r is not None else _read_el_print_lines(path, eset)
    return out


def _step_only_job(text: str, step_idx: int, elset: str) -> str | None:
    """Text of a ccx job with the model and only the step `step_idx` (linear statics: the steps are independent, OP=NEW), all
    outputs of the step replaced by *EL PRINT S of `elset`. None = the step does not exist."""
    from .ccx.parallel import split_steps
    model, steps = split_steps(text)
    if not 0 <= step_idx < len(steps):
        return None
    body, skip = [], False
    for ln in steps[step_idx].splitlines(True):
        if ln.startswith("*") and not ln.startswith("**"):
            key = ln.split(",")[0].strip().upper()
            skip = key in _OUTPUT_CARDS
            if key == "*END STEP":
                body.append(f"*EL PRINT, ELSET={elset}\nS\n")
        if not skip:
            body.append(ln)
    return model + "".join(body)


def _fetch_section_stresses(step: StepResult, sel: np.ndarray, study: Study,
                            n_elements: int = 0) -> dict[int, np.ndarray] | None:
    """Obtains the exact stresses at the integration points for the elements of the cut `sel` (0-based):
    1. From the cache `step._sig_ip`
    2. From the `.dat` file of the existing run (one job or the parallel job of the step), if it contains *EL PRINT S for sel
    3. From a dedicated quick ccx run of this step only (model + one *STEP, no other outputs) with *EL PRINT S of all solid
       elements 1..n_elements (only the cut ones above ALL_ELEMENTS_MAX) – cached in `step._sig_ip`, so further cuts need no ccx
    4. None if ccx is not available (e.g. an Abaqus step) -> fallback to the displacements.
    """
    sig_cache = getattr(step, "_sig_ip", None)
    if sig_cache is not None:
        if all((e + 1) in sig_cache for e in sel):
            return sig_cache
    else:
        sig_cache = {}

    workdir = getattr(step, "_workdir", None)
    if workdir is None:
        return None
    workdir = Path(workdir)

    job_name = getattr(step, "_job", "job")
    inp_path = workdir / f"{job_name}.inp"
    step_idx = getattr(step, "_step_index", 0)

    for dat_path, k in ((workdir / f"{job_name}.dat", step_idx), (workdir / f"{job_name}_s{step_idx}.dat", 0)):
        if not dat_path.exists():
            continue
        try:
            prints = _dat_prints(dat_path)
        except Exception:  # noqa: BLE001 – an unreadable .dat: the next source
            continue
        for eset in _DAT_SETS:
            blocks = prints.get(eset)
            if blocks and k < len(blocks):
                cand = blocks[k]
                if all((e + 1) in cand for e in sel):
                    sig_cache.update(cand)
                    step._sig_ip = sig_cache
                    return sig_cache

    if not inp_path.exists():
        return None

    cut_dir = workdir / "section_cut"
    cut_dir.mkdir(parents=True, exist_ok=True)

    try:
        missing = [e for e in sel if (e + 1) not in sig_cache]
        if not missing:
            return sig_cache
        text_inp = inp_path.read_text(encoding="latin-1")
        elset = "ECUT"
        if 0 < n_elements <= ALL_ELEMENTS_MAX:          # solid elements are numbered 1..n (element e = index e − 1)
            extra = f"*ELSET, ELSET={elset}, GENERATE\n1, {int(n_elements)}, 1\n"
        else:
            from .ccx.inp import _write_ids
            buf = io.StringIO()
            buf.write(f"*ELSET, ELSET={elset}\n")
            _write_ids(buf, np.asarray(missing) + 1)
            extra = buf.getvalue()
        job = _step_only_job(text_inp, step_idx, elset)
        if job is None:
            return None
        pos = job.find("*STEP")
        job = job[:pos] + extra + job[pos:]
        (cut_dir / "job_cut.inp").write_text(job, encoding="latin-1")

        from .ccx.runner import run_ccx           # the same environment, without a console window, ends with the process
        exe = Path(study.solver.exe) if study.solver.exe else None
        run_ccx(cut_dir, "job_cut", exe=exe)

        dat_cut = cut_dir / "job_cut.dat"
        if not dat_cut.exists():
            return None

        prints = _dat_prints(dat_cut, (elset,)).get(elset) or []
        if prints:
            sig_cache.update(prints[0])
            step._sig_ip = sig_cache
            return sig_cache
    except Exception as e:  # noqa: BLE001 – exact stresses are not necessary: backup from the displacements (6 valid digits of .frd)
        logging.getLogger("fea_core.section_forces").warning(
            "Section force: integration-point stresses not available (%s) – computed from displacements (less accurate)", e)
        return None

    return None


def compute_element_forces(study: Study, mesh: Mesh, U: np.ndarray,
                           elem_indices: np.ndarray, dT: np.ndarray | None = None,
                           sig_ip: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Internal nodal forces fe (k, 10, 3) of the C3D10 elements:
    - from the stresses at the integration points of ccx (∫Bᵀσ dV, full precision),
    - or from the displacements U (`recovery.element_nodal_forces`, backup and solvers other than ccx).
    """
    if sig_ip is not None and len(sig_ip):
        elems = np.asarray(elem_indices, int)
        if all((e + 1) in sig_ip for e in elems):
            sig = np.stack([sig_ip[int(e) + 1] for e in elems])
            return joints_nodal_forces(mesh, elems, sig)
    return element_nodal_forces(study, mesh, U, np.asarray(elem_indices, int), dT)


def section_csys(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds an orthonormal basis (e_x, e_y, e_z) and the transformation matrix R (3, 3).
    e_x = normal n (axis of the cut).
    v_loc = R @ v_glob.
    """
    ex = np.asarray(normal, float)
    norm = np.linalg.norm(ex)
    if norm < 1e-12:
        raise ValueError("The section plane normal has zero length")
    ex = ex / norm

    ref = np.array([0.0, 1.0, 0.0]) if abs(ex[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    ez = np.cross(ex, ref)
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    R = np.stack([ex, ey, ez], axis=0)
    return ex, ey, ez, R


def section_forces(study: Study, mesh: Mesh, step: StepResult,
                   point, normal, parts: list[str] | None = None) -> dict:
    """Computes the resulting forces and moments on the cut through the body by the plane given by a point and a normal.
    The normal points from side '-' to side '+'.
    """
    P0 = np.asarray(point, float)
    ex, ey, ez, R = section_csys(np.asarray(normal, float))

    if parts:
        part_idx = [mesh.part_names.index(p) for p in parts if p in mesh.part_names]
        if not part_idx:
            raise ValueError(f"None of the selected parts {parts} was found in the mesh")
        elem_mask = np.isin(mesh.tet_part, part_idx)
        candidate_elems = np.nonzero(elem_mask)[0]
    else:
        candidate_elems = np.arange(len(mesh.tets))

    tets = mesh.tets[candidate_elems]
    centers = mesh.nodes[tets[:, :4]].mean(axis=1)
    dists = (centers - P0) @ ex

    neg_sub = np.nonzero(dists < 0)[0]
    pos_sub = np.nonzero(dists >= 0)[0]
    if len(neg_sub) == 0 or len(pos_sub) == 0:
        raise ValueError("The section does not cut the selected body (all elements lie on one side of the plane)")

    elems_neg = candidate_elems[neg_sub]
    elems_pos = candidate_elems[pos_sub]

    # nodes of the cut = nodes shared by the elements of both sides; only the elements of side '-' that touch the cut are computed
    cut_nodes = np.intersect1d(np.unique(mesh.tets[elems_neg]), np.unique(mesh.tets[elems_pos]))
    if len(cut_nodes) == 0:
        raise ValueError("The section does not pass through any shared nodes between the split parts")
    lut = np.full(len(mesh.nodes), -1, dtype=np.int64)
    lut[cut_nodes] = np.arange(len(cut_nodes))

    cut_coords = mesh.nodes[cut_nodes]
    centroid = cut_coords.mean(axis=0)

    touching = (lut[mesh.tets[elems_neg]] >= 0).any(axis=1)
    sel = elems_neg[touching]

    # thermal strain: ΔT = T − T_ref of the nodes (state with a thermal load)
    dT = None
    T = getattr(step, "fields", {}).get("T") if step is not None else None
    if T is not None and len(T) == len(mesh.nodes):
        t_ref = getattr(getattr(study, "thermal", None), "reference_temperature", 20.0)
        dT = np.asarray(T, float) - float(t_ref)

    # Computation of the internal forces of the elements touching side '-' and the contributions to the nodes of the cut
    sig_by_elem = _fetch_section_stresses(step, sel, study, len(mesh.tets))
    fe = compute_element_forces(study, mesh, step.U, sel, dT, sig_ip=sig_by_elem)
    idx = lut[mesh.tets[sel]]                                    # (k, 10) number of the cut node or -1
    on_cut = idx >= 0
    cut_node_forces = np.stack(
        [np.bincount(idx[on_cut], weights=fe[:, :, c][on_cut], minlength=len(cut_nodes)) for c in range(3)], axis=1)

    F_total = cut_node_forces.sum(axis=0)
    r = cut_coords - centroid
    M_total = np.cross(r, cut_node_forces).sum(axis=0)

    F_loc = R @ F_total
    M_loc = R @ M_total

    return {
        "centroid": centroid.tolist(),
        "normal": ex.tolist(),
        "plane_point": P0.tolist(),
        "axes": {"x": ex.tolist(), "y": ey.tolist(), "z": ez.tolist()},
        "F_global": F_total.tolist(),
        "M_global": M_total.tolist(),
        "F_local": F_loc.tolist(),
        "M_local": M_loc.tolist(),
        "N": float(F_loc[0]),
        "Vy": float(F_loc[1]),
        "Vz": float(F_loc[2]),
        "Mx": float(M_loc[0]),
        "My": float(M_loc[1]),
        "Mz": float(M_loc[2]),
        "cut_nodes_count": int(len(cut_nodes)),
        "elems_neg_count": int(len(elems_neg)),
        "elems_touching_cut_count": int(len(sel)),
        "elems_pos_count": int(len(elems_pos)),
    }
