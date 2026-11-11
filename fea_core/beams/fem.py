"""FE of a frame system by the U1 element in ccx (Timoshenko, native DOF 1–6) – discretisation, writing the .inp, internal forces.

Verified by trials (scratch/beams_fem/exp_u1*.py, 14. 10. 2026):
- U1 takes the torsional stiffness from the polar moment Ip = I11 + I22 (ccx manual 6.2.46 – circular sections only); ν > 0.5 ccx
  rejects, so the replacement by G does not work → each element = TWO parallel U1: the main one (section × (1 − s)) with the torsion released
  at the end (copy of the node B' tied by *EQUATION in all components except the rotation about the axis) + the torsional one (section × s),
  s = J/Ip → axial, bending and shear stiffness exactly by the sum, torsion G·J. Verified also on an oblique axis (rotation 1e-6).
- Bending E–B and the rotation of U1 exact for any number of elements; the shear deformation (κ) converges with refinement: the share of shear
  −9 %/n² (n = 20 → −2.3 % of the shear share, ≪ 1 % of the whole) → a straight bar of 20 elements.
- Direction 1 (third line of *BEAM SECTION) = axis z of the section, I11 = Iy (bending in the direction z), I22 = Iz.
- U1: *FREQUENCY (mass) works, *BUCKLE does not ("no buckling"), thermal expansion wrong (do not use), SPRING1 on a rotational DOF does not act →
  rotational spring = auxiliary node g with SPRING1 in DOF 1 + *EQUATION u_g,1 = θ.
  The RF of ccx at the nodes of U1 is unusable (not K·u) → reactions and internal forces from the end forces of the elements (below).
- Distributed load = consistent nodal forces and moments (m_a = l²/12·(x̂ × q) …) – deflection and rotation exact.
Internal forces: end forces of the element f = K_U1·u − f_eq. K_U1 = the ACTUAL matrix of the U1 element from ccx (*FREQUENCY,
SOLVER=MATRIXSTORAGE → .sti/.dof; one run for all distinct segments, torsion replaced by G·J/L) – the exact Timoshenko matrix differs from U1 in shear
(U1 is ~10 % stiffer, verified exp_sti.py) and gave an imbalance of 9 %.
Recomputation from the result (as recovery.py), not a solver; the residual imbalance at the nodes is reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .model import FrameMember, FrameNodeLoad, FrameMemberLoad, DOF_NAMES, LOCAL_NAMES

G_RATIO_EPS = 1e-3
KAPPA_EB = 1e9


class FrameError(ValueError):
    pass


@dataclass
class Segment:
    member: str
    a: int                       # FE node index (the real end – not B')
    b: int
    R: np.ndarray                # rows = local axes x, y, z in global
    L: float
    sec: object                  # SectionProps
    E: float
    G: float
    rho: float
    kappa: float
    s_tor: float                 # share of the torsion element J/Ip (1 = the single element)
    bprime: int | None = None    # node B' of the main element (released torsion)
    s0: float = 0.0              # position of the start along the bar [mm]
    K: np.ndarray | None = None  # local stiffness matrix U1 (12×12) from ccx (u1_matrices), otherwise Timoshenko
    rel: np.ndarray | None = None  # own solver: released local components of the ends (12 × bool: N Vy Vz T My Mz of the start, of the end)
    alpha: float | None = None   # thermal expansion of the material [1/K] (own solver – thermal load)
    truss: bool = False          # own solver: a bar without torsional stiffness (truss / torsion released at the end): J = 0

    def key(self) -> tuple:
        p = self.sec
        return tuple(round(float(v), 9) for v in (p.A, p.Iy, p.Iz, p.J, self.kappa, self.E, self.G, self.L))


@dataclass
class FrameFE:
    xyz: list = field(default_factory=list)                  # coordinates of the FE nodes of the bars
    joint: dict = field(default_factory=dict)                # system node → index
    segs: list = field(default_factory=list)
    member_segs: dict = field(default_factory=dict)          # bar → [indices of the segments]
    eqs: list = field(default_factory=list)                  # [[(node, dof 1–6, coef)…]] (the first term is dependent)
    springs: list = field(default_factory=list)              # (node, dof 1–3, k) – SPRING1
    bc: list = field(default_factory=list)                   # (node, dof)
    couplings: list = field(default_factory=list)            # (node of the hinge, selection, aux node R)
    end_node: dict = field(default_factory=dict)             # (member, "start"|"end") → index of the member end node
    active: set = field(default_factory=set)                 # (node, dof) active in the system

    def add(self, p) -> int:
        self.xyz.append(np.asarray(p, float))
        return len(self.xyz) - 1


def shear_kappa(sec) -> float:
    """Shear coefficient κ (U1: one for both directions): solid rectangle 5/6, solid circle 0.9 (≈ 6(1+ν)/(7+6ν)),
    otherwise Av_z/A (EN 1993-1-1, 6.2.6 – shear area in the direction of the web), without it 5/6.
    """
    sh = (sec.shape or "").upper()
    if sh == "RECT":
        return 5.0 / 6.0
    if sh == "CIRC":
        return 0.9
    return sec.Av_z / sec.A if sec.Av_z > 0 and sec.A > 0 else 5.0 / 6.0


def _axes(x: np.ndarray, up, roll_deg: float, alpha: float, plane_n=None, center=None, mid=None) -> np.ndarray:
    """Local axes of the bar (rows x, y, z); pure Python instead of numpy (thousands of bars: 0.12 → 0.02 ms per bar)."""
    x0, x1, x2 = (float(v) for v in x)
    n = math.sqrt(x0 * x0 + x1 * x1 + x2 * x2)
    x0, x1, x2 = x0 / n, x1 / n, x2 / n
    if isinstance(up, str):
        if up == "normal" and plane_n is not None:
            u = tuple(float(v) for v in plane_n)
        elif up == "radial" and center is not None:
            u = tuple(float(a) - float(b) for a, b in zip(center, mid))
        else:
            raise FrameError(f"Orientation '{up}' applies only to an arc (normal | radial)")
    elif up is not None:
        u = tuple(float(v) for v in up)
    else:
        u = (0.0, 0.0, 1.0) if abs(x2) < 0.95 else (1.0, 0.0, 0.0)
    d = u[0] * x0 + u[1] * x1 + u[2] * x2
    z0, z1, z2 = u[0] - d * x0, u[1] - d * x1, u[2] - d * x2
    nz = math.sqrt(z0 * z0 + z1 * z1 + z2 * z2)
    if nz < 1e-9:
        raise FrameError("The cross-section orientation is parallel to the member axis – specify a different “up” vector")
    z0, z1, z2 = z0 / nz, z1 / nz, z2 / nz
    y0, y1, y2 = z1 * x2 - z2 * x1, z2 * x0 - z0 * x2, z0 * x1 - z1 * x0
    a = math.radians(roll_deg) + alpha                       # rotation + main axes of the section (L profiles)
    if a:
        c, sn = math.cos(a), math.sin(a)
        y0, y1, y2, z0, z1, z2 = (c * y0 + sn * z0, c * y1 + sn * z1, c * y2 + sn * z2,
                                  -sn * y0 + c * z0, -sn * y1 + c * z1, -sn * y2 + c * z2)
    return np.array([[x0, x1, x2], [y0, y1, y2], [z0, z1, z2]])


def _arc(p0, p1, p2):
    """Circle through the points p0, p1 (on the arc), p2: centre, radius, normal of the plane, angle of the arc p0 → p2 through p1."""
    a, b, c = (np.asarray(p, float) for p in (p0, p1, p2))
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = np.linalg.norm(n)
    if nn < 1e-12 * max(np.linalg.norm(ab), 1) ** 2:
        raise FrameError("Arc points lie on a straight line")
    n /= nn
    # centre: solution (x − a)·ab = |ab|²/2, (x − a)·ac = |ac|²/2, (x − a)·n = 0
    M = np.array([ab, ac, n])
    cen = a + np.linalg.solve(M, [ab @ ab / 2, ac @ ac / 2, 0.0])
    r = float(np.linalg.norm(a - cen))
    e1 = (a - cen) / r
    e2 = np.cross(n, e1)

    def ang(p):
        v = p - cen
        return math.atan2(v @ e2, v @ e1) % (2 * math.pi)
    th1, th2 = ang(b), ang(c)
    if not th1 < th2:                                        # passing through p1 → reverse the direction
        n, e2 = -n, -e2
        th1, th2 = ang(b), ang(c)
    return cen, r, n, e1, e2, th2


def solver_of(study, solver: str | None = None) -> str:
    """Bar solver: "timoshenko" (own exact element, the frame system alone) | "ccx_u1" (the U1 element in ccx – mixed
    models with bodies and the choice of FrameSettings.solver).
    """
    if solver:
        return solver
    if str(getattr(study, "geometry", "") or "").strip():
        return "ccx_u1"
    return "ccx_u1" if study.frame.settings.solver == "ccx_u1" else "timoshenko"


def build(study, solver: str | None = None, segments: int | None = None) -> FrameFE:
    """Discretisation of Study.frame into elements (nodes, segments, springs, constraints). For ccx_u1 in addition the equations of the hinges and torsion
    (U1 element: pair of elements, nodes E and B'); the own solver (timoshenko) solves the hinges statically at the element level, the nodes remain
    only the nodes of the system and the internal nodes of the bars. `segments` overrides the number of elements per straight bar (natural frequencies and buckling).
    """
    fr = study.frame
    st = fr.settings
    tim = solver_of(study, solver) == "timoshenko"
    fe = FrameFE()
    for name, nd in fr.nodes.items():
        fe.joint[name] = fe.add(nd.xyz)
    secs = {k: s.resolve() for k, s in fr.sections.items()}
    for mname, m in fr.members.items():
        for k in (m.start, m.end):
            if k not in fe.joint:
                raise FrameError(f"Member '{mname}': node '{k}' does not exist")
        if m.section not in secs:
            raise FrameError(f"Member '{mname}': cross-section '{m.section}' does not exist")
        sec = secs[m.section]
        mat = study.materials.get(m.material) if m.material else (
            study.materials.get(study.default_material) if study.default_material else None)
        if mat is None:
            raise FrameError(f"Member '{mname}' has no material")
        E, nu = float(mat.E), float(mat.nu)
        G = E / (2 * (1 + nu))
        alpha_t = None if getattr(mat, "alpha", None) is None else float(mat.alpha)
        kappa = shear_kappa(sec) if st.shear_deformation else KAPPA_EB
        Ip = sec.Iy + sec.Iz
        s_tor = 1.0 if sec.J >= (1 - G_RATIO_EPS) * Ip else max(sec.J / Ip, 1e-9)
        pa, pb = np.asarray(fr.nodes[m.start].xyz, float), np.asarray(fr.nodes[m.end].xyz, float)
        rel_a, rel_b = list(m.releases_start), list(m.releases_end)
        if m.kind == "truss":
            rel_a = [False, False, False, False, True, True]
            rel_b = [False, False, False, not tim, True, True]          # U1: torsion released at one end; own solver J = 0
        if rel_a[3] and rel_b[3]:
            raise FrameError(f"Member '{mname}': torsion released at both ends – the member would rotate freely")
        # own solver: a bar with released torsion at either end (and a truss) transmits no torsional moment (without
        # a distributed torsional load T is constant along the bar = 0) → torsional stiffness J = 0, the release of torsion is discarded
        # (a chain of elements would otherwise have a zero mode of spinning about the axis, natural frequency 0 Hz)
        no_torsion = bool(tim and (m.kind == "truss" or rel_a[3] or rel_b[3]))
        if no_torsion:
            rel_a[3] = rel_b[3] = False
        # points of the member axis
        if m.through is not None:
            cen, r, n, e1, e2, th = _arc(pa, m.through, pb)
            # own solver divides an arc into at most 2° per element (the elements are cheap; chord error O(h²)), U1 by max_arc_angle
            nseg = m.segments or max(8, math.ceil(math.degrees(th) / (min(st.max_arc_angle, 2.0) if tim else st.max_arc_angle)))
            pts = [cen + r * (math.cos(t) * e1 + math.sin(t) * e2) for t in np.linspace(0, th, nseg + 1)]
            arc = (cen, n)
        else:
            if np.linalg.norm(pb - pa) < 1e-9:
                raise FrameError(f"Member '{mname}' has zero length")
            if tim:
                nseg = m.segments or segments or max(int(st.segments_static), 1)
            else:
                nseg = m.segments or (1 if m.kind == "truss" else segments or st.segments_straight)
            d_ = pb - pa
            pts = [pa + d_ * (j / nseg) for j in range(nseg)] + [pb]
            arc = None
        ids = [fe.joint[m.start]] + [fe.add(p) for p in pts[1:-1]] + [fe.joint[m.end]]
        # ends of a bar with a release → own node E tied to the node of the system (U1 only; the own solver condenses the hinge)
        for end, rel, k in (("start", rel_a, 0), ("end", rel_b, len(ids) - 1)):
            if any(rel) and not tim:
                e = fe.add(pts[k])
                fe.end_node[(mname, end)] = e
                ids[k] = e
            else:
                fe.end_node[(mname, end)] = ids[k]
        segs = []
        s0 = 0.0
        for j in range(nseg):
            a, b = ids[j], ids[j + 1]
            xa, xb = pts[j], pts[j + 1]
            mid = 0.5 * (xa + xb)
            R = _axes(xb - xa, m.up if m.up is not None else ("normal" if arc is not None else None), m.roll,
                      sec.alpha, plane_n=None if arc is None else arc[1], center=None if arc is None else arc[0],
                      mid=mid)
            L = float(np.linalg.norm(xb - xa))
            sg = Segment(mname, a, b, R, L, sec, E, G, float(mat.rho or 0.0), kappa, s_tor, s0=s0, alpha=alpha_t,
                         truss=no_torsion)
            if tim:                                          # release of the ends of the bar: only the first / last element
                sg.rel = np.array((rel_a if j == 0 else [False] * 6) + (rel_b if j == nseg - 1 else [False] * 6), bool)
            elif s_tor < 1.0:                                # main element → node B' (torsion released)
                sg.bprime = fe.add(xb)
                _tie(fe, sg.bprime, b, R, keep=(True, True, True, False, True, True))
            fe.segs.append(sg)
            segs.append(len(fe.segs) - 1)
            s0 += L
        fe.member_segs[mname] = segs
        # equations of the hinges at the ends (local axes of the end segment)
        for end, rel, sgi in (("start", rel_a, segs[0]), ("end", rel_b, segs[-1])):
            if any(rel) and not tim:
                R = fe.segs[sgi].R
                j = fe.joint[m.start if end == "start" else m.end]
                _tie(fe, fe.end_node[(mname, end)], j, R, keep=tuple(not r for r in rel))
    if not tim:                                              # active DOF only for U1 (the own solver determines them from the matrix)
        for sg in fe.segs:
            for nd in (sg.a, sg.b) + ((sg.bprime,) if sg.bprime is not None else ()):
                fe.active |= {(nd, d) for d in range(1, 7)}
        for eq in fe.eqs:
            fe.active |= {(n_, d) for n_, d, _ in eq}
    # support
    for sup in fr.supports:
        if sup.node not in fe.joint:
            raise FrameError(f"Support: node '{sup.node}' does not exist")
        j = fe.joint[sup.node]
        for d in range(6):
            k = float(sup.stiffness[d]) if d < len(sup.stiffness) else 0.0
            if tim:                                          # spring and constraint directly in the global components of the node
                if k > 0:
                    fe.springs.append((j, d + 1, k))
                elif sup.dofs[d]:
                    fe.bc.append((j, d + 1))
                continue
            if (j, d + 1) not in fe.active:
                if sup.dofs[d] or k > 0:
                    if d >= 3:
                        continue                              # rotation of a node with the hinges of all bars – holds nothing
                    raise FrameError(f"Support of node '{sup.node}': component {DOF_NAMES[d]} is not active in the system")
                continue
            if k > 0:
                if d < 3:
                    fe.springs.append((j, d + 1, k))
                else:                                        # rotational spring: g,1 = θ (SPRING1 at g)
                    g = fe.add(fe.xyz[j])
                    fe.eqs.append([(g, 1, 1.0), (j, d + 1, -1.0)])
                    fe.springs.append((g, 1, k))
            elif sup.dofs[d]:
                fe.bc.append((j, d + 1))
    for c in fr.connections:
        if c.node not in fe.joint:
            raise FrameError(f"Connection to body: node '{c.node}' does not exist")
        if c.coupling != "rigid":
            raise FrameError("Member-to-body connection: only rigid coupling so far")
        if tim:
            raise FrameError("The connection of a member to a solid is computed by ccx only (U1 element) – a beam structure alone does not have it")
        j = fe.joint[c.node]
        fe.couplings.append((j, c.selection, fe.add(fe.xyz[j])))
    return fe


def _tie(fe: FrameFE, dep: int, ind: int, R: np.ndarray, keep) -> None:
    """Equations (u_dep − u_ind)·c = 0 for the retained local components (translation in 3 axes, rotation in 3 axes);
    by blocks with Gauss–Jordan so that every equation has its own dependent DOF of the node dep.
    """
    for blk in (0, 1):
        rows = [R[k] for k in range(3) if keep[3 * blk + k]]
        if not rows:
            continue
        A = np.array(rows, float)
        # choice of pivots (columns = DOF of the node dep in the block)
        A = A.copy()
        piv = []
        for i in range(len(A)):
            cand = [c for c in range(3) if c not in piv]
            c = max(cand, key=lambda q: abs(A[i, q]))
            A[i] /= A[i, c]
            for k in range(len(A)):
                if k != i:
                    A[k] -= A[k, c] * A[i]
            piv.append(c)
        for i, c in enumerate(piv):
            terms = [(dep, 3 * blk + c + 1, 1.0)]
            terms += [(dep, 3 * blk + q + 1, float(A[i, q])) for q in range(3) if q != c and abs(A[i, q]) > 1e-14]
            terms += [(ind, 3 * blk + q + 1, -float(A[i, q])) for q in range(3) if abs(A[i, q]) > 1e-14]
            fe.eqs.append(terms)


# ------------------------------------------------------------------ loads

def _consistent(q_a: np.ndarray, q_b: np.ndarray, L: float, R: np.ndarray) -> np.ndarray:
    """Consistent nodal forces (12,) globally for the linear distributed load q_a → q_b (global vectors N/mm)."""
    x = R[0]
    qa_ax, qb_ax = (q_a @ x) * x, (q_b @ x) * x
    qa_t, qb_t = q_a - qa_ax, q_b - qb_ax
    f = np.zeros(12)
    f[0:3] = L * (2 * qa_ax + qb_ax) / 6 + L * (7 * qa_t + 3 * qb_t) / 20
    f[6:9] = L * (qa_ax + 2 * qb_ax) / 6 + L * (3 * qa_t + 7 * qb_t) / 20
    f[3:6] = L * L / 60 * np.cross(x, 3 * qa_t + 2 * qb_t)
    f[9:12] = -L * L / 60 * np.cross(x, 2 * qa_t + 3 * qb_t)
    return f


def segment_loads(study, fe: FrameFE, lc) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Distributed load by segments: {segment: (q_a, q_b) globally [N/mm]} – loads of the bars + weight."""
    from ..study import Gravity
    out: dict[int, list] = {}
    total = {}
    for m in study.frame.members.values():
        total[m.name] = sum(fe.segs[i].L for i in fe.member_segs.get(m.name, []))
    for l in lc.loads:
        if isinstance(l, FrameMemberLoad):
            if l.member not in fe.member_segs:
                raise FrameError(f"Member load: member '{l.member}' does not exist")
            qa0 = np.asarray(l.q, float)
            qb0 = qa0 if l.q_end is None else np.asarray(l.q_end, float)
            Lm = total[l.member]
            for i in fe.member_segs[l.member]:
                sg = fe.segs[i]
                ta, tb = sg.s0 / Lm, (sg.s0 + sg.L) / Lm
                qa, qb = qa0 + (qb0 - qa0) * ta, qa0 + (qb0 - qa0) * tb
                if l.axes == "local":
                    qa, qb = sg.R.T @ qa, sg.R.T @ qb
                acc = out.setdefault(i, [np.zeros(3), np.zeros(3)])
                acc[0] = acc[0] + qa
                acc[1] = acc[1] + qb
        elif isinstance(l, Gravity):
            from ..loads import global_vector
            g = global_vector(study, l)
            for i, sg in enumerate(fe.segs):
                q = sg.rho * sg.sec.A * g
                acc = out.setdefault(i, [np.zeros(3), np.zeros(3)])
                acc[0] = acc[0] + q
                acc[1] = acc[1] + q
    return {k: (v[0], v[1]) for k, v in out.items()}


def nodal_loads(study, fe: FrameFE, lc) -> np.ndarray:
    """Nodal forces and moments (n_nodes, 6) of the load state (nodal loads + consistent distributed + weight)."""
    F = np.zeros((len(fe.xyz), 6))
    for l in lc.loads:
        if isinstance(l, FrameNodeLoad):
            if l.node not in fe.joint:
                raise FrameError(f"Node load: node '{l.node}' does not exist")
            F[fe.joint[l.node], :3] += np.asarray(l.force, float)
            F[fe.joint[l.node], 3:] += np.asarray(l.moment, float)
    for i, (qa, qb) in segment_loads(study, fe, lc).items():
        sg = fe.segs[i]
        f = _consistent(qa, qb, sg.L, sg.R)
        F[sg.a] += f[:6]
        F[sg.b] += f[6:]
    for n_, d in [(n_, d) for n_ in range(len(fe.xyz)) for d in range(6) if F[n_, d] != 0]:
        if (n_, d + 1) not in fe.active:
            raise FrameError("Moment applied to a node where all members are hinged (node rotation is not in the system)")
    return F


def load_resultant(study, fe: FrameFE, lc, c0) -> tuple[np.ndarray, np.ndarray]:
    """Resultant of the loads of the bars (force, moment about the point c0) – EXACT (without consistent moments)."""
    Fs, Ms = np.zeros(3), np.zeros(3)
    X = np.array(fe.xyz)
    c0 = np.asarray(c0, float)
    for l in lc.loads:
        if isinstance(l, FrameNodeLoad):
            p = X[fe.joint[l.node]]
            Fs += np.asarray(l.force, float)
            Ms += np.asarray(l.moment, float) + np.cross(p - c0, l.force)
    for i, (qa, qb) in segment_loads(study, fe, lc).items():
        sg = fe.segs[i]
        xa, xb = X[sg.a], X[sg.b]
        F = sg.L * (qa + qb) / 2
        # centroid of the trapezoidal load by components: ∫ s·q ds = L²(qa + 2qb)/6
        Fs += F
        Ms += np.cross(xa - c0, F) + np.cross(sg.R[0], sg.L ** 2 * (qa + 2 * qb) / 6)
    return Fs, Ms


# ------------------------------------------------------------------ writing the .inp

def write(f, study, fe: FrameFE, base_node: int, base_elem: int, fmt, surf_name=None,
          declare_u1: bool = True, mass: bool = True) -> tuple[int, int]:
    """Nodes (NSET NFRAME), U1 elements, materials and sections, equations, springs, connection to a body. The ccx node number
    = base_node + index; returns (next node, next element).
    """
    nid = lambda i: base_node + i
    f.write("** ---------------- prutova soustava (U1)\n*NODE, NSET=NFRAME\n")
    for i, p in enumerate(fe.xyz):
        f.write(f"{nid(i)}, {fmt(p[0])}, {fmt(p[1])}, {fmt(p[2])}\n")
    if declare_u1:
        f.write("*USER ELEMENT, TYPE=U1, INTEGRATION POINTS=2, MAXDOF=6, NODES=2\n")
    e = base_elem
    mats: dict[tuple, str] = {}
    for si, sg in enumerate(fe.segs):
        key = (sg.E, sg.G, sg.rho)
        if key not in mats:
            mats[key] = f"MF{len(mats) + 1}"
            nu = sg.E / (2 * sg.G) - 1
            f.write(f"*MATERIAL, NAME={mats[key]}\n*ELASTIC\n{fmt(sg.E)}, {fmt(nu)}\n"
                    + (f"*DENSITY\n{fmt(sg.rho)}\n" if mass and sg.rho > 0 else ""))
        parts = [(1.0 - sg.s_tor if sg.s_tor < 1 else 1.0, sg.bprime if sg.s_tor < 1 else sg.b, "M")]
        if sg.s_tor < 1:
            parts.append((sg.s_tor, sg.b, "T"))
        p = sg.sec
        for frac, nb, tag in parts:
            es = f"FR{si + 1}{tag}"
            f.write(f"*ELEMENT, TYPE=U1, ELSET={es}\n{e}, {nid(sg.a)}, {nid(nb)}\n")
            e += 1
            z = sg.R[2]
            f.write(f"*BEAM SECTION, SECTION=GENERAL, ELSET={es}, MATERIAL={mats[key]}\n"
                    f"{fmt(p.A * frac)}, {fmt(p.Iy * frac)}, 0., {fmt(p.Iz * frac)}, {fmt(sg.kappa)}\n"
                    f"{fmt(z[0])}, {fmt(z[1])}, {fmt(z[2])}\n")
    if fe.springs:
        def real(v):
            t = fmt(v)
            return t if "." in t else (t.replace("e", ".e") if "e" in t else t + ".")
        for k, (n_, d, kk) in enumerate(fe.springs):
            f.write(f"*ELEMENT, TYPE=SPRING1, ELSET=FSP{k + 1}\n{e}, {nid(n_)}\n*SPRING, ELSET=FSP{k + 1}\n{d}\n{real(kk)}\n")
            e += 1
    for j, sel, r in fe.couplings:
        f.write(f"*COUPLING, REF NODE={nid(r)}, SURFACE={surf_name(sel)}, CONSTRAINT NAME=CF{r}\n*KINEMATIC\n1, 3\n")
    # connection to a body: a node of the bar dependent on an auxiliary node of the kinematic constraint (as the bolts; the rotation of the REF node
    # *KINEMATIC ccx itself ties → R must not be dependent in another equation)
    eqs = list(fe.eqs) + [[(j, d, 1.0), (r, d, -1.0)] for j, _, r in fe.couplings for d in range(1, 7)]
    if eqs:
        f.write("*EQUATION\n")
        for terms in eqs:
            f.write(f"{len(terms)}\n")
            for q in range(0, len(terms), 4):
                f.write(", ".join(f"{nid(n_)}, {d}, {fmt(c)}" for n_, d, c in terms[q:q + 4]) + "\n")
    return base_node + len(fe.xyz), e


def boundary_lines(fe: FrameFE, base_node: int) -> list[str]:
    return [f"{base_node + n_}, {d}, {d}, 0." for n_, d in fe.bc]


def cload_lines(study, fe: FrameFE, lc, base_node: int, fmt) -> list[str]:
    F = nodal_loads(study, fe, lc)
    return [f"{base_node + n_}, {d + 1}, {fmt(F[n_, d])}" for n_ in range(len(fe.xyz)) for d in range(6)
            if F[n_, d] != 0.0]


# ------------------------------------------------------------------ results

def timoshenko_k(sg: Segment) -> np.ndarray:
    """Local stiffness matrix 12×12 (u v w θx θy θz)×2, Timoshenko (φ = 12EI/(κGA L²)); only for the recomputation of the forces."""
    p, L, E, G = sg.sec, sg.L, sg.E, sg.G
    K = np.zeros((12, 12))
    ea, gj = E * p.A / L, G * p.J / L
    for i, j, v in ((0, 0, ea), (0, 6, -ea), (6, 6, ea), (3, 3, gj), (3, 9, -gj), (9, 9, gj)):
        K[i, j] = K[j, i] = v
    GA = sg.kappa * G * p.A

    def bend(I, iv, ith, sign):
        phi = 12 * E * I / (GA * L * L)
        c = E * I / ((1 + phi) * L ** 3)
        m = c * np.array([[12, sign * 6 * L, -12, sign * 6 * L],
                          [sign * 6 * L, (4 + phi) * L * L, -sign * 6 * L, (2 - phi) * L * L],
                          [-12, -sign * 6 * L, 12, -sign * 6 * L],
                          [sign * 6 * L, (2 - phi) * L * L, -sign * 6 * L, (4 + phi) * L * L]])
        idx = [iv, ith, iv + 6, ith + 6]
        for a in range(4):
            for b in range(4):
                K[idx[a], idx[b]] += m[a, b]
    bend(p.Iz, 1, 5, +1)            # v, θz (bending in the plane xy)
    bend(p.Iy, 2, 4, -1)            # w, θy (bending in the plane xz)
    return K


def u1_matrices(fe: FrameFE, workdir, exe=None) -> None:
    """Local stiffness matrix U1 of all distinct segments by one ccx run (MATRIXSTORAGE) → Segment.K."""
    from pathlib import Path
    from ..ccx.runner import run_ccx, CcxError
    keys: dict[tuple, Segment] = {}
    for sg in fe.segs:
        keys.setdefault(sg.key(), sg)
    if not keys:
        return
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    fmt = lambda v: f"{v:.12g}"
    lines = ["*NODE, NSET=NALL"]
    for i, (k, sg) in enumerate(keys.items()):
        lines += [f"{2 * i + 1}, 0, 0, 0", f"{2 * i + 2}, {fmt(sg.L)}, 0, 0"]
    lines.append("*USER ELEMENT, TYPE=U1, INTEGRATION POINTS=2, MAXDOF=6, NODES=2")
    for i, (k, sg) in enumerate(keys.items()):
        p = sg.sec
        nu = sg.E / (2 * sg.G) - 1
        lines += [f"*ELEMENT, TYPE=U1, ELSET=E{i + 1}", f"{i + 1}, {2 * i + 1}, {2 * i + 2}",
                  f"*MATERIAL, NAME=M{i + 1}", "*ELASTIC", f"{fmt(sg.E)}, {fmt(nu)}", "*DENSITY", "1e-9",
                  f"*BEAM SECTION, SECTION=GENERAL, ELSET=E{i + 1}, MATERIAL=M{i + 1}",
                  f"{fmt(p.A)}, {fmt(p.Iy)}, 0., {fmt(p.Iz)}, {fmt(sg.kappa)}", "0, 0, 1"]
    lines += ["*STEP", "*FREQUENCY, SOLVER=MATRIXSTORAGE", "1", "*END STEP"]
    (wd / "u1k.inp").write_text("\n".join(lines) + "\n")
    r = run_ccx(wd, "u1k", exe=exe)
    if not r.finished or not (wd / "u1k.sti").exists():
        raise CcxError("Could not obtain the element matrices of U1 (MATRIXSTORAGE): " + "; ".join(r.errors[:3]))
    dof = (wd / "u1k.dof").read_text().split()
    pos = {}
    for i, d in enumerate(dof):
        nd, k = d.split(".")
        pos[(int(nd), int(k))] = i
    n = len(dof)
    K = np.zeros((n, n))
    for line in (wd / "u1k.sti").read_text().splitlines():
        a = line.split()
        if len(a) == 3:
            K[int(a[0]) - 1, int(a[1]) - 1] = float(a[2])
    K = K + K.T - np.diag(np.diag(K))
    mats = {}
    for i, (k, sg) in enumerate(keys.items()):
        idx = [pos[(2 * i + 1 + e, d)] for e in (0, 1) for d in range(1, 7)]
        Ke = K[np.ix_(idx, idx)].copy()
        gj = sg.G * sg.sec.J / sg.L                     # torsion: pair of elements → G·J/L (not G·Ip)
        for a_, b_, v in ((3, 3, gj), (3, 9, -gj), (9, 3, -gj), (9, 9, gj)):
            Ke[a_, b_] = v
        mats[k] = Ke
    for sg in fe.segs:
        sg.K = mats[sg.key()]


def recover(study, fe: FrameFE, lc, U6: np.ndarray, n_pts: int = 3) -> dict:
    """Internal forces by bars, reactions of the supports and the equilibrium check from the displacements and rotations of the nodes U6 (n, 6)."""
    seg_q = segment_loads(study, fe, lc)
    # only concentrated loads of the nodes – the consistent forces of the distributed load are already in the end forces of the elements (K·u − f_eq)
    F = np.zeros((len(fe.xyz), 6))
    for l in lc.loads:
        if isinstance(l, FrameNodeLoad):
            F[fe.joint[l.node], :3] += np.asarray(l.force, float)
            F[fe.joint[l.node], 3:] += np.asarray(l.moment, float)
    acc = np.zeros((len(fe.xyz), 6))                  # Σ of the end forces of the elements (forces of the elements on the nodes = −f)
    members = {}
    X = np.array(fe.xyz)
    for mname, segs in fe.member_segs.items():
        rows = {k: [] for k in ("s", *LOCAL_NAMES)}
        pos, disp = [], []
        for i in segs:
            sg = fe.segs[i]
            T = np.zeros((12, 12))
            for b in range(4):
                T[3 * b:3 * b + 3, 3 * b:3 * b + 3] = sg.R
            ue = np.concatenate([U6[sg.a], U6[sg.b]])
            qa, qb = seg_q.get(i, (np.zeros(3), np.zeros(3)))
            feq = _consistent(qa, qb, sg.L, sg.R)
            Kl = sg.K if sg.K is not None else timoshenko_k(sg)
            f_loc = Kl @ (T @ ue) - T @ feq
            acc[sg.a] += T[:6, :6].T @ f_loc[:6]
            acc[sg.b] += T[:6, :6].T @ f_loc[6:]
            Fa, Ma = f_loc[0:3], f_loc[3:6]
            qa_l, qb_l = sg.R @ qa, sg.R @ qb
            ex = np.array([1.0, 0, 0])
            for t in np.linspace(0.0, 1.0, n_pts):
                x = t * sg.L
                Iq = qa_l * x + (qb_l - qa_l) * x * x / (2 * sg.L)
                IIq = qa_l * x * x / 2 + (qb_l - qa_l) * x ** 3 / (6 * sg.L)
                S = -Fa - Iq
                M = -Ma + x * np.cross(ex, Fa) + np.cross(ex, IIq)
                rows["s"].append(sg.s0 + x)
                for k, v in zip(LOCAL_NAMES, (*S, *M)):
                    rows[k].append(float(v))
                pos.append((X[sg.a] + t * (X[sg.b] - X[sg.a])).tolist())
                disp.append(((1 - t) * U6[sg.a, :3] + t * U6[sg.b, :3]).tolist())
        sec = fe.segs[segs[0]].sec
        N, My, Mz = (np.asarray(rows[k]) for k in ("N", "My", "Mz"))
        Vy, Vz, Tq = (np.asarray(rows[k]) for k in ("Vy", "Vz", "T"))
        sig = np.zeros(len(N))
        for (fy, fz) in (sec.fibers or [(0.0, 0.0)]):
            sig = np.maximum(sig, np.abs(N / sec.A + My * fz / sec.Iy - Mz * fy / sec.Iz))
        tau = np.zeros(len(N))
        if sec.Av_y > 0:
            tau += np.abs(Vy) / sec.Av_y
        if sec.Av_z > 0:
            tau = np.maximum(tau, np.abs(Vz) / sec.Av_z)
        if sec.Wt > 0:
            tau += np.abs(Tq) / sec.Wt
        members[mname] = {**{k: v for k, v in rows.items()}, "points": pos, "u": disp,
                          "sigma_max": sig.tolist(), "tau_est": tau.tolist(),
                          "sigma_eq": np.sqrt(sig ** 2 + 3 * tau ** 2).tolist(), "section": sec.name}
    # equilibrium: R = Σ f_e − F at the node (f_e = K·u − f_eq end forces of the elements, F = concentrated loads of the node): the force that
    # the node needs "from outside" – constraint, spring, body; otherwise ≈ 0.
    # The nodes of hinges E, B' and auxiliary nodes (springs, connection) pass the forces through the equations to the node of the system → by groups.
    R = acc - F
    group = np.arange(len(fe.xyz))
    for eq in fe.eqs:
        dep = eq[0][0]
        others = sorted({n_ for n_, _, _ in eq[1:] if n_ != dep})
        if others:
            group[dep] = others[0]
    for _ in range(4):                                  # chains (B' → internal node → …) – followed back to the root
        group = group[group]
    gsum = np.zeros_like(R)
    np.add.at(gsum, group, R)
    held = set()                                        # groups with an external force (constraint, spring, connection to a body)
    for n_, _ in fe.bc:
        held.add(int(group[n_]))
    for n_, _, _ in fe.springs:
        held.add(int(group[n_]))
    for j, _, r in fe.couplings:
        held.add(int(group[j]))
    react = {name: gsum[group[j]].tolist() for name, j in fe.joint.items()
             if int(group[j]) in held and not any(c.node == name for c in study.frame.connections)}
    coupling = {c.node: gsum[group[fe.joint[c.node]]].tolist() for c in study.frame.connections}
    roots = np.unique(group)
    free_roots = [g for g in roots if int(g) not in held]
    scale = max(float(np.abs(F[:, :3]).sum()), float(sum(np.abs(v[:3]).sum() for v in react.values())), 1e-30)
    resid = float(np.abs(gsum[free_roots, :3]).max(initial=0.0)) / scale if free_roots else 0.0
    return {"members": members, "reactions": react, "nodes": {name: U6[j].tolist() for name, j in fe.joint.items()},
            "node_residual": resid, "coupling_forces": coupling}


def read_displacements(dat_path, base_node: int, n_nodes: int) -> list[np.ndarray]:
    """Displacements and rotations of the nodes NFRAME by steps from the .dat (*NODE PRINT U – 6 components at the nodes with rotation)."""
    from ..solve import _read_printed
    out = []
    for blk in _read_printed(dat_path, "NFRAME"):
        U = np.zeros((n_nodes, 6))
        for nid, v in blk.items():
            i = nid - base_node
            if 0 <= i < n_nodes:
                U[i, :len(v)] = v[:6]
        out.append(U)
    return out
