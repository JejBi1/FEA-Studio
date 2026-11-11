"""Connection of the parts in the conforming mesh (5.5 of the specification): which parts share the mesh (contact after `occ.fragment` = fixed
connection, no contact) and through which area – for the confirmation by the user (GUI step Joints, MCP).
"""
from __future__ import annotations

import numpy as np

from ..mesh import Mesh, TET_FACES


def bonded_pairs(mesh: Mesh) -> list[dict]:
    """Pairs of parts with common faces of the elements: [{"a", "b", "area" [mm²], "faces"}], sorted by the area.

    A face = a triple of corner nodes; a common face of two elements of different parts = a fixed connection. A split (disconnected) interface (welds,
    `node_alias`) is not counted – there are no common nodes. A touch only along an edge or a point is not reported (it transfers no
    area, but the mesh is continuous – see `edge_only`).
    """
    if len(mesh.tets) == 0 or len(mesh.part_names) < 2:
        return []
    faces = np.sort(mesh.tets[:, :4][:, TET_FACES].reshape(-1, 3), axis=1)
    elem = np.repeat(np.arange(len(mesh.tets)), 4)
    part = mesh.tet_part[elem]
    if faces.max() < (1 << 21):
        key = (faces[:, 0].astype(np.uint64) << 42) | (faces[:, 1].astype(np.uint64) << 21) | faces[:, 2].astype(np.uint64)
    else:
        key = np.ascontiguousarray(faces).view([("a", np.int64), ("b", np.int64), ("c", np.int64)]).ravel()
    order = np.argsort(key, kind="stable")
    ks = key[order]
    dup = np.nonzero(ks[1:] == ks[:-1])[0]                 # inner face = a pair in a row
    i, j = order[dup], order[dup + 1]
    pa, pb = part[i], part[j]
    diff = pa != pb
    i, pa, pb = i[diff], pa[diff], pb[diff]
    if len(i) == 0:
        return []
    x = mesh.nodes[faces[i]]
    area = 0.5 * np.linalg.norm(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), axis=1)
    lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
    pair = lo.astype(np.int64) * len(mesh.part_names) + hi
    out = []
    for pk in np.unique(pair):
        msk = pair == pk
        a, b = divmod(int(pk), len(mesh.part_names))
        out.append({"a": mesh.part_names[a], "b": mesh.part_names[b], "area": float(area[msk].sum()),
                    "faces": int(msk.sum())})
    return sorted(out, key=lambda d: -d["area"])


def edge_only(mesh: Mesh) -> list[tuple[str, str]]:
    """Pairs of parts with common nodes but without a common face (touch along an edge / point – connection through a few nodes,
    usually unwanted: a singularity, almost a mechanism).
    """
    n_p = len(mesh.part_names)
    if n_p < 2:
        return []
    owner_sets = []
    for pi in range(n_p):
        owner_sets.append(np.unique(mesh.tets[mesh.tet_part == pi]))
    face_pairs = {(d["a"], d["b"]) for d in bonded_pairs(mesh)}
    out = []
    for a in range(n_p):
        for b in range(a + 1, n_p):
            if len(np.intersect1d(owner_sets[a], owner_sets[b], assume_unique=True)):
                key = (mesh.part_names[a], mesh.part_names[b])
                if key not in face_pairs:
                    out.append(key)
    return out
