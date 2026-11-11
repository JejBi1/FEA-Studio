"""Writing of the contact and the nonlinear steps into the .inp of ccx 2.23 (F8.5, specification 4.3–4.5; trials scratch/f8_contact).

- *SURFACE INTERACTION + *SURFACE BEHAVIOR (K ALWAYS written explicitly – the default of ccx takes the E of the first material
  in the input) + *FRICTION (μ > 0, λ explicitly; μ = 0 is refused by ccx) + *CONTACT PAIR (slave, master) [+ *CLEARANCE];
- no_separation = PRESSURE-OVERCLOSURE=TIED (only S2S penalty, needs *FRICTION with a slope) + μ 1e-4;
- compression_only without a counterpart = GAPUNI elements in the nodes of the face to fixed nodes (in the direction of the outward normal);
- MORTAR hard = without *SURFACE BEHAVIOR; one formulation per study (ccx does not allow mixing the types).
"""
from __future__ import annotations

import numpy as np

from .model import MU_NONE, MU_ROUGH
from ..ccx import deck_text

FORMULATION = {"surface_to_surface": "SURFACE TO SURFACE", "mortar": "MORTAR", "node_to_surface": "NODE TO SURFACE"}


def pair_params(study, mesh, c) -> dict:
    """K, λ, μ of the pair (default: K = 50·min E of the pair, λ = K/100, μ from the library by the purpose)."""
    from .detect import pair_parts
    pa, pb = pair_parts(study, c)
    Es = []
    for p in pa | pb:
        try:
            Es.append(study.material_of(p).E)
        except KeyError:
            pass
    E = min(Es) if Es else 210000.0
    K = float(c.stiffness) if c.stiffness else 50.0 * E
    lam = float(c.stick_slope) if c.stick_slope else K / 100.0
    mu = None
    if c.kind == "frictional":
        mu = float(c.mu) if c.mu is not None else default_mu(study, c)
    elif c.kind == "rough":
        mu = MU_ROUGH
    elif c.kind == "no_separation":
        mu = MU_NONE
    return {"K": K, "lambda": lam, "mu": mu}


def default_mu(study, c) -> float:
    """μ from the library of friction (fea_core.calc.friction_lib) by the pair / materials and purpose, otherwise 0.15 / 0.3."""
    try:
        from ..calc import friction_lib as fl
        if c.materials:
            return float(fl.mu(c.materials, c.purpose))
        from .detect import pair_parts
        pa, pb = pair_parts(study, c)
        ma = study.material_of(sorted(pa)[0]).name
        mb = study.material_of(sorted(pb)[0]).name
        s = fl.suggest(ma, mb)
        if s:
            name = s[0] if isinstance(s, (list, tuple)) else s
            name = name.get("name") if isinstance(name, dict) else name
            return float(fl.mu(name, c.purpose))
    except Exception:  # noqa: BLE001 – library unavailable / unknown pair
        pass
    return 0.15 if c.purpose == "slip" else 0.3


def needs_friction_card(study, c) -> bool:
    """The interaction needs *FRICTION from the 1st step: friction, without separation (TIED), or the friction changes in the history
    (*CHANGE FRICTION only for an interaction that has *FRICTION – otherwise a crash of ccx, verified scratch/f8_review/fric.py).
    """
    if c.kind in ("frictional", "rough", "no_separation"):
        return True
    return any(c.name in s.mu_change for s in study.analysis.sequence)


def write_pairs(f, study, mesh, m, surf_name, fmt, next_node: int, next_elem: int) -> tuple[int, int]:
    """Interactions and contact pairs (model data, before the first *STEP). m.contacts = [{pair, interaction, slave, master,
    gap_nodes, ground}].
    """
    rep = {r["pair"]: r for r in mesh.report.get("contacts", [])}
    m.contacts = []
    method = study.contact_settings.method
    ctype = FORMULATION.get(method, "SURFACE TO SURFACE")
    for i, c in enumerate(study.all_contacts(), 1):
        if c.name not in rep:
            continue
        r = rep[c.name]
        pp = pair_params(study, mesh, c)
        info = {"pair": c.name, "index": i, "interaction": f"SI{i}", "kind": c.kind, **pp}
        if c.kind == "compression_only" and r.get("master") is None:
            next_node, next_elem = _write_gapuni(f, study, mesh, c, r, info, pp, fmt, next_node, next_elem)
            m.contacts.append(info)
            continue
        slave, master = surf_name(r["slave"]), surf_name(r["master"])
        info.update(slave=slave, master=master)
        f.write(f"** contact {i}: {deck_text(c.name)} ({deck_text(c.kind)})\n"
                f"*SURFACE INTERACTION, NAME=SI{i}\n")
        if c.kind == "no_separation":
            if method != "surface_to_surface":
                raise ValueError(f"Contact '{c.name}': no separation (TIED) works only with the surface-to-surface formulation")
            f.write(f"*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=TIED\n{fmt(pp['K'])}\n")
        elif method == "mortar" and c.stiffness is None:
            pass                                     # hard contact mortar = without *SURFACE BEHAVIOR
        elif method == "node_to_surface":
            f.write(f"*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR\n{fmt(pp['K'])}, 3., 1.e-3\n")
        else:
            f.write(f"*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR\n{fmt(pp['K'])}\n")
        if needs_friction_card(study, c):
            mu = pp["mu"] if pp["mu"] is not None else MU_NONE
            f.write(f"*FRICTION\n{fmt(mu)}, {fmt(pp['lambda'])}\n")
            info["friction"] = True
        adj = ""
        if c.adjust and c.kind != "interference":
            adj = f", ADJUST={fmt(max(abs(r.get('gap_min_mm', 0.0)), 0.0) + 1e-6)}"
        f.write(f"*CONTACT PAIR, INTERACTION=SI{i}, TYPE={ctype}{adj}\n{slave}, {master}\n")
        if c.clearance is not None:
            if method == "node_to_surface":
                raise ValueError(f"Contact '{c.name}': *CLEARANCE (interference / clearance) works only with surface-to-surface / mortar")
            f.write(f"*CLEARANCE, MASTER={master}, SLAVE={slave}, VALUE={fmt(float(c.clearance))}\n")
        m.contacts.append(info)
    return next_node, next_elem


def _write_gapuni(f, study, mesh, c, r, info, pp, fmt, next_node, next_elem):
    """Support only in compression: GAPUNI from each node of the face to a fixed node in the direction of the outward normal of the node."""
    from .detect import _OPPOSITE_NORMALS
    nodes, normals = _OPPOSITE_NORMALS(mesh, mesh.surfaces[c.a])
    ground = []
    f.write(f"** compression-only support: {deck_text(c.name)}\n*NODE\n")
    for n_ in nodes:
        x = mesh.nodes[n_]
        f.write(f"{next_node}, {fmt(x[0])}, {fmt(x[1])}, {fmt(x[2])}\n")
        ground.append(next_node)
        next_node += 1
    # stiffness: K·(area of the node) – the weights of the area (consistent ∫N dA)
    from ..ccx.inp import consistent_nodal_forces
    w = consistent_nodal_forces(mesh.nodes, mesh.surfaces[c.a], np.array([1.0, 0.0, 0.0]))
    for k, (n_, g, nv) in enumerate(zip(nodes, ground, normals)):
        a = float(w.get(int(n_), np.zeros(3))[0])
        # ccx 2.23: GAPUNI has an effective stiffness k/2 (trial G1 15. 10. 2026: displacement 2·p/K, RF of the fixed node 2× the force)
        kk = 2.0 * pp["K"] * max(a, 1e-12)
        es = f"EGP{info['index']}_{k + 1}"
        f.write(f"*ELEMENT, TYPE=GAPUNI, ELSET={es}\n{next_elem}, {int(n_) + 1}, {g}\n"
                f"*GAP, ELSET={es}\n0., {fmt(nv[0])}, {fmt(nv[1])}, {fmt(nv[2])}, 0., {fmt(kk)}\n")
        next_elem += 1
    f.write(f"*NSET, NSET=NGP{info['index']}\n")
    from ..ccx.inp import _write_ids
    _write_ids(f, ground)
    info.update(ground=ground, ground_set=f"NGP{info['index']}", nodes=(nodes + 1).tolist())
    return next_node, next_elem


def step_header(study, step=None, nonlinear: bool = True) -> tuple[str, str]:
    """(*STEP line, line of the increments under *STATIC) of the nonlinear step."""
    cs = study.contact_settings
    dt0, dmin, dmax, inc = 0.1, 1e-5, 0.25, 200
    direct = False
    if step is not None:
        dt0, dmin, dmax, inc, direct = step.dt0, step.dt_min, step.dt_max, step.max_inc, step.direct
    nl = ", NLGEOM" if cs.nlgeom else ""
    head = f"*STEP, INC={int(inc)}{nl}"
    if step is not None and step.dynamic:              # implicit dynamics: times in seconds
        return head, "DYNAMIC", f"{dt0:.6g}, {float(step.time):.6g}, {dmin:.6g}, {dmax:.6g}"
    return head, (", DIRECT" if direct else ""), f"{dt0:.6g}, 1., {dmin:.6g}, {dmax:.6g}"


def outputs(study, m) -> str:
    """Outputs of the contact in the step: CDIS, CSTR (.frd), forces of the pairs *CONTACT PRINT CF only for the S2S penalty (mortar
    gives zeros, N2S crashes ccx).
    """
    if not getattr(m, "contacts", None):
        return ""
    fq = ", FREQUENCY=99999" if study.analysis.branches else ""
    out = f"*CONTACT FILE{fq}\nCDIS, CSTR\n"
    if study.contact_settings.method == "surface_to_surface":
        for c in m.contacts:                         # CF only for the chosen pair (parameters SLAVE, MASTER – node250)
            if c.get("slave"):
                out += f"*CONTACT PRINT, SLAVE={c['slave']}, MASTER={c['master']}\nCF\n"
    gsets = [c["ground_set"] for c in m.contacts if c.get("ground_set")]
    for g in gsets:
        out += f"*NODE PRINT, NSET={g}\nRF\n"
    return out


def change_cards(study, m, step) -> str:
    """*CHANGE FRICTION a *MODEL CHANGE kroku posloupnosti."""
    out = []
    by = {c["pair"]: c for c in getattr(m, "contacts", [])}
    for name, mu in (step.mu_change or {}).items():
        c = by.get(name)
        if c is None or not c.get("friction"):
            raise ValueError(f"Step '{step.name}': friction change on pair '{name}', which has no friction from the 1st step")
        if float(mu) <= 0:
            raise ValueError(f"Step '{step.name}': μ must be > 0 (no friction = 1e-4)")
        out.append(f"*CHANGE FRICTION, INTERACTION={c['interaction']}\n*FRICTION\n{float(mu):.6g}, {c['lambda']:.6g}\n")
    for name in step.contacts_off or []:
        c = by.get(name)
        if c is None or not c.get("slave"):
            continue
        out.append(f"*MODEL CHANGE, TYPE=CONTACT PAIR, REMOVE\n{c['slave']}, {c['master']}\n")
    return "".join(out)
