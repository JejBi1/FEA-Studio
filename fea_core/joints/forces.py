"""Nodal forces of the elements from the stresses at the integration points (post-processing, no stiffness matrix).

f_e,a = Σ_gp w·det J · σ_gp · ∇N_a  (internal nodal forces of the element at the nodes; the sum over all elements
of the node = RF of ccx). The force that the surroundings exert on a free body (weld, clamped part) at the node is the
sum of f_e over the elements of this body – exact equilibrium, unlike the integration of averaged nodal stresses
(*SECTION PRINT), which are inaccurate at stress jumps.
The stresses at the integration points are read from the .dat (*EL PRINT, S). Order of the components in the .dat: xx yy zz xy xz yz.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..ccx.frd import dat_pattern, header_lines, line_end, odd_line_breaks, parse_e_fields
from ..loads import GAUSS4, W_GAUSS4, _tet10_dN, inv3, scatter_add
from ..mesh import Mesh, TET_FACES, TET10_FACE_MIDS

_EL_HEAD = r"stresses \(elem, integ\.pnt\.,sxx,syy,szz,sxy,sxz,syz\) for set {} "
_STRAIN_HEAD = r"strains \(elem, integ\.pnt\.,exx,eyy,ezz,exy,exz,eyz\) for set {} "
_EL_WIDTH = 14 + 6 * 14          # line *EL PRINT S / E in the .dat: %10d%4d + 6 × (space + %13.6E)


def read_el_print(dat: str | Path, setname: str) -> list[dict[int, np.ndarray]]:
    """Stresses at the integration points from *EL PRINT for a set, by steps: element (1-based) -> (nip, 6)
    in the order xx yy zz xy yz zx (as the .frd and Results.S).

    Fast by blocks of fixed columns (numpy); anything unusual in the file → the original line-by-line reading
    (the same result, ~10× slower).
    """
    try:
        res = _el_print_fast(Path(dat).read_bytes(), setname)
    except ValueError:
        res = None
    return res if res is not None else _read_el_print_lines(dat, setname)


def _el_print_fast(buf: bytes, setname: str) -> list[dict[int, np.ndarray]] | None:
    """Blocks of data after the heading as a matrix of bytes (all lines of the same length, the separators in place), numbers
    through ccx.frd.parse_e_fields (bitwise as float()). None = something unusual → slow path.
    """
    n = len(buf)
    heads = header_lines(buf, dat_pattern(_EL_HEAD, setname.upper()))
    out = []
    for hi, h in enumerate(heads):
        limit = heads[hi + 1] if hi + 1 < len(heads) else n
        cur: dict[int, np.ndarray] = {}
        out.append(cur)
        pos = line_end(buf, h) + 1
        if odd_line_breaks(buf[h:pos]):
            return None
        while pos < limit:                              # empty lines before the data
            e = line_end(buf, pos)
            if odd_line_breaks(buf[pos:e + 1]):
                return None
            if buf[pos:e].decode("latin-1").split():
                break
            pos = e + 1
        if pos >= limit:
            continue
        e = buf.find(b"\n", pos)
        if e == -1:
            return None
        crlf = buf[e - 1] == 13
        stride = e + 1 - pos
        if stride - 1 - crlf != _EL_WIDTH:
            return None
        rows = (limit - pos) // stride
        a = np.frombuffer(buf, np.uint8, count=rows * stride, offset=pos).reshape(rows, stride)
        dig = lambda j: (a[:, j] - np.uint8(48)) <= 9     # noqa: E731
        sp = lambda j: a[:, j] == 32                      # noqa: E731
        # each line whole (the end of the line in the same place), number of the element and of the point right-aligned, spaces between the fields
        ok = (a[:, -1] == 10) & ((a[:, -2] == 13) if crlf else (a[:, -2] != 13))
        ok &= dig(9) & sp(10) & (sp(11) | (dig(11) & dig(12))) & (sp(12) | dig(12)) & dig(13)
        for k in range(6):
            ok &= sp(14 + 14 * k)
        bad = np.flatnonzero(~ok)
        L = int(bad[0]) if len(bad) else rows
        if L == 0:
            return None
        q = pos + L * stride                            # the line after the data: empty / another heading / end of the file
        if q < limit:
            seg = buf[q:line_end(buf, q) + 1]
            if odd_line_breaks(seg) or seg.decode("latin-1").split():
                return None
        a = a[:L]
        ids = a[:, :10].copy().view("S10").ravel().astype(np.int64)
        vals = parse_e_fields(a[:, 14:14 + 84].reshape(L, 6, 14)[:, :, 1:], 6).reshape(L, 6)
        cur.update(_group_rows(ids, vals[:, [0, 1, 2, 3, 5, 4]]))   # xx yy zz xy xz yz → xx yy zz xy yz zx
    return out


def read_el_print_strain(dat: str | Path, setname: str) -> list[dict[int, np.ndarray]]:
    """Strains at the integration points from *EL PRINT (E or ME with GLOBAL=YES), by steps:
    element (1-based) -> (nip, 6) in the order Voigt xx yy zz xy xz yz (engineering shear gamma = 2*eps).
    """
    try:
        res = _el_print_strain_fast(Path(dat).read_bytes(), setname)
    except Exception:
        res = None
    return res if res is not None else _read_el_print_strain_lines(dat, setname)


def _el_print_strain_fast(buf: bytes, setname: str) -> list[dict[int, np.ndarray]] | None:
    n = len(buf)
    heads = header_lines(buf, dat_pattern(_STRAIN_HEAD, setname.upper()))
    out = []
    for hi, h in enumerate(heads):
        limit = heads[hi + 1] if hi + 1 < len(heads) else n
        cur: dict[int, np.ndarray] = {}
        out.append(cur)
        pos = line_end(buf, h) + 1
        if odd_line_breaks(buf[h:pos]):
            return None
        while pos < limit:
            e = line_end(buf, pos)
            if odd_line_breaks(buf[pos:e + 1]):
                return None
            if buf[pos:e].decode("latin-1").split():
                break
            pos = e + 1
        if pos >= limit:
            continue
        e = buf.find(b"\n", pos)
        if e == -1:
            return None
        crlf = buf[e - 1] == 13
        stride = e + 1 - pos
        if stride - 1 - crlf != _EL_WIDTH:
            return None
        rows = (limit - pos) // stride
        a = np.frombuffer(buf, np.uint8, count=rows * stride, offset=pos).reshape(rows, stride)
        dig = lambda j: (a[:, j] - np.uint8(48)) <= 9     # noqa: E731
        sp = lambda j: a[:, j] == 32                      # noqa: E731
        ok = (a[:, -1] == 10) & ((a[:, -2] == 13) if crlf else (a[:, -2] != 13))
        ok &= dig(9) & sp(10) & (sp(11) | (dig(11) & dig(12))) & (sp(12) | dig(12)) & dig(13)
        for k in range(6):
            ok &= sp(14 + 14 * k)
        bad = np.flatnonzero(~ok)
        L = int(bad[0]) if len(bad) else rows
        if L == 0:
            return None
        q = pos + L * stride
        if q < limit:
            seg = buf[q:line_end(buf, q) + 1]
            if odd_line_breaks(seg) or seg.decode("latin-1").split():
                return None
        a = a[:L]
        ids = a[:, :10].copy().view("S10").ravel().astype(np.int64)
        vals = parse_e_fields(a[:, 14:14 + 84].reshape(L, 6, 14)[:, :, 1:], 6).reshape(L, 6)
        vals[:, 3:] *= 2.0                                # tensor shear → engineering shear (gamma = 2*eps)
        cur.update(_group_rows(ids, vals))
    return out


def _read_el_print_strain_lines(dat: str | Path, setname: str) -> list[dict[int, np.ndarray]]:
    out: list[dict[int, np.ndarray]] = []
    cur: dict[int, list] | None = None
    pat = re.compile(_STRAIN_HEAD.format(setname.upper()))
    with open(dat, "r", encoding="latin-1") as f:
        for line in f:
            if pat.search(line):
                cur = {}
                out.append(cur)
                continue
            if cur is None:
                continue
            parts = line.split()
            if not parts:
                if cur:
                    cur = None
                continue
            try:
                e = int(parts[0])
                vals = [float(v) for v in parts[2:8]]
                vals[3] *= 2.0
                vals[4] *= 2.0
                vals[5] *= 2.0
            except (ValueError, IndexError):
                cur = None
                continue
            cur.setdefault(e, []).append(vals)
    res = []
    for blk in out:
        d = {}
        for e, rows in blk.items():
            d[e] = np.asarray(rows, float)
        res.append(d)
    return res


def _group_rows(ids: np.ndarray, vals: np.ndarray) -> dict[int, np.ndarray]:
    """Rows by elements {element: (nip, 6)} in the order of the first occurrence, rows of the element in the order in the file."""
    if len(ids) == 0:
        return {}
    starts = np.concatenate([[0], np.flatnonzero(ids[1:] != ids[:-1]) + 1])
    gid = ids[starts]
    if len(np.unique(gid)) == len(gid):                 # usually: the rows of an element one after another
        counts = np.diff(np.append(starts, len(ids)))
        if np.all(counts == counts[0]):
            groups = list(vals.reshape(len(gid), int(counts[0]), vals.shape[1]))
        else:
            groups = np.split(vals, starts[1:])
        return dict(zip(gid.tolist(), groups))
    uniq, first, inv = np.unique(ids, return_index=True, return_inverse=True)
    parts = np.split(vals[np.argsort(inv, kind="stable")], np.cumsum(np.bincount(inv))[:-1])
    return {int(uniq[j]): parts[j] for j in np.argsort(first)}


def _read_el_print_lines(dat: str | Path, setname: str) -> list[dict[int, np.ndarray]]:
    """Original line-by-line reading (backup of the fast path and reference of the tests)."""
    out: list[dict[int, np.ndarray]] = []
    cur: dict[int, list] | None = None
    pat = re.compile(_EL_HEAD.format(setname.upper()))
    with open(dat, "r", encoding="latin-1") as f:
        for line in f:
            if pat.search(line):
                cur = {}
                out.append(cur)
                continue
            if cur is None:
                continue
            parts = line.split()
            if not parts:
                if cur:
                    cur = None
                continue
            try:
                e = int(parts[0])
                vals = [float(v) for v in parts[2:8]]
            except (ValueError, IndexError):
                cur = None
                continue
            cur.setdefault(e, []).append(vals)
    res = []
    for blk in out:
        d = {}
        for e, rows in blk.items():
            a = np.asarray(rows, float)
            d[e] = a[:, [0, 1, 2, 3, 5, 4]]              # xx yy zz xy xz yz → xx yy zz xy yz zx
        res.append(d)
    return res


def element_nodal_forces(mesh: Mesh, elems: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Internal nodal forces of tet10 elements: elems (k,), sig (k, 4, 6) → (k, 10, 3)."""
    X = mesh.nodes[mesh.tets[elems]]                     # (k, 10, 3)
    f = np.zeros((len(elems), 10, 3))
    for g, p in enumerate(GAUSS4):                       # integration points of C3D10 in ccx (gauss3d4)
        dN = _tet10_dN(*p)                                # (3, 10)
        J = np.einsum("ai,kij->kaj", dN, X)               # (k, 3, 3): J[a, j] = ∂x_j/∂ξ_a
        det = (J[:, 0, 0] * (J[:, 1, 1] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 1])
               - J[:, 0, 1] * (J[:, 1, 0] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 0])
               + J[:, 0, 2] * (J[:, 1, 0] * J[:, 2, 1] - J[:, 1, 1] * J[:, 2, 0]))
        dNdx = inv3(J) @ dN                               # (k, 3, 10)
        s = sig[:, g]
        S = np.empty((len(elems), 3, 3))
        S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
        S[:, 0, 1] = S[:, 1, 0] = s[:, 3]
        S[:, 1, 2] = S[:, 2, 1] = s[:, 4]
        S[:, 2, 0] = S[:, 0, 2] = s[:, 5]
        f += (W_GAUSS4 * det)[:, None, None] * (dNdx.swapaxes(1, 2) @ S)
    return f


def nodal_force_sum(mesh: Mesh, elems: np.ndarray, sig_by_elem: dict[int, np.ndarray]) -> np.ndarray:
    """Sum of the internal forces of the elements `elems` (0-based) at the nodes → (N, 3)."""
    sig = np.stack([sig_by_elem[int(e) + 1] for e in elems])
    fe = element_nodal_forces(mesh, elems, sig)
    out = np.zeros((len(mesh.nodes), 3))
    scatter_add(out, mesh.tets[elems].ravel()[:, None], fe.reshape(-1, 1, 3))   # = np.add.at by rows
    return out


def mesh_interfaces(mesh: Mesh) -> dict[tuple[int, int], dict]:
    """Common faces of elements of different parts: (part a, part b), a < b →
    {"tri": nodes of the face from the element of part a, "tri_b": from the element of part b, "elem_a", "elem_b"}.

    Faces are matched by the original node numbers (Mesh.canonical) – they are found even across a disconnected contact.
    """
    inv, cnt = mesh.face_ids()
    twice = np.nonzero(cnt[inv] == 2)[0]
    order = twice[np.argsort(inv[twice], kind="stable")]
    f1, f2 = order[0::2], order[1::2]                    # pairs of faces with the same corners
    p1, p2 = mesh.tet_part[f1 // 4], mesh.tet_part[f2 // 4]
    m = p1 != p2
    f1, f2, p1, p2 = f1[m], f2[m], p1[m], p2[m]
    swap = p1 > p2
    fa, fb = np.where(swap, f2, f1), np.where(swap, f1, f2)
    ea, eb = fa // 4, fb // 4
    pa, pb = mesh.tet_part[ea], mesh.tet_part[eb]
    def face_nodes(e, fn):
        corner = mesh.tets[e[:, None], TET_FACES[fn]]
        if mesh.tets.shape[1] == 10:
            return np.hstack([corner, mesh.tets[e[:, None], TET10_FACE_MIDS[fn]]])
        return corner
    tri = face_nodes(ea, fa % 4)
    tri_b = face_nodes(eb, fb % 4)
    out: dict[tuple[int, int], dict] = {}
    for key in set(zip(pa.tolist(), pb.tolist())):
        sel = (pa == key[0]) & (pb == key[1])
        out[key] = {"tri": tri[sel], "tri_b": tri_b[sel], "elem_a": ea[sel], "elem_b": eb[sel]}
    return out


def elements_touching(mesh: Mesh, part: int, nodes: np.ndarray) -> np.ndarray:
    """Elements of the part `part` that contain any of the nodes."""
    flag = np.zeros(len(mesh.nodes), bool)
    flag[nodes] = True
    sel = (mesh.tet_part == part) & flag[mesh.tets].any(axis=1)
    return np.nonzero(sel)[0]
