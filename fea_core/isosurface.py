"""Isosurfaces of a chosen value of a scalar field in a tetrahedral mesh (A5: display of results).

Algorithm Marching Tetrahedra:
  - Input: node coordinates nodes (N, 3), tetrahedra tets (M, 4|10), field values at the nodes values (N,),
    and the required isosurface value isovalue. Inside a tetrahedron the field is linear (corner nodes).
  - Output: triangle mesh of the isosurface: iso_vertices (V, 3), iso_faces (F, 3).

Orientation: triangle normals (right-hand rule, (p1 − p0) × (p2 − p0)) point uniformly AGAINST the gradient of the field,
i.e. outward from the region values ≥ isovalue (in topology optimisation outward from the solid material ρ > 0.5).

Vertices are welded exactly along the mesh edges: a point of the isosurface on the edge (a, b) is always interpolated from the node a with
values ≥ isovalue to the node b with values < isovalue, so neighbouring tetrahedra give a bitwise equal point (no rounding of
coordinates). A point exactly at a node (values = isovalue) has the key (a, a).

With `boundary_tris` (the surface of the mesh with outward normals) the part of the surface where values ≥ isovalue is added to the isosurface → a closed
(watertight) surface of the region values ≥ isovalue. Without them the isosurface is open at the boundary of the region.
"""
from __future__ import annotations

import numpy as np


# Edges of a tetrahedron (vertex indices 0..3): 6 edges
# 0: (0, 1), 1: (1, 2), 2: (2, 0), 3: (0, 3), 4: (1, 3), 5: (2, 3)
TET_EDGES = np.array([
    [0, 1], [1, 2], [2, 0],
    [0, 3], [1, 3], [2, 3]
], dtype=np.int32)

# Marching tetrahedra table: for each of the 16 cases (bit mask of the 4 vertices with v >= isovalue)
# contains the list of triangles formed by edge indices (0..5). At most 2 triangles per tetrahedron in total.
# The vertex order in the table is not uniformly oriented – extract_isosurface unifies the orientation by the gradient.
MT_TABLE = [
    [],                      # 0000: 0
    [[0, 2, 3]],             # 0001: node 0
    [[0, 1, 4]],             # 0010: node 1
    [[2, 3, 1], [1, 3, 4]],  # 0011: nodes 0, 1 (quadrilateral)
    [[1, 2, 5]],             # 0100: node 2
    [[0, 1, 5], [0, 5, 3]],  # 0101: nodes 0, 2
    [[0, 4, 2], [2, 4, 5]],  # 0110: nodes 1, 2
    [[3, 4, 5]],             # 0111: nodes 0, 1, 2 (node 3 inverted)
    [[3, 5, 4]],             # 1000: node 3
    [[0, 5, 2], [0, 4, 5]],  # 1001: nodes 0, 3
    [[0, 1, 3], [1, 5, 3]],  # 1010: nodes 1, 3
    [[1, 5, 2]],             # 1011: nodes 0, 1, 3 (node 2 inverted)
    [[1, 4, 2], [2, 4, 3]],  # 1100: nodes 2, 3
    [[0, 4, 1]],             # 1101: nodes 0, 2, 3 (node 1 inverted)
    [[0, 3, 2]],             # 1110: nodes 1, 2, 3 (node 0 inverted)
    [],                      # 1111: all
]


def _reference_flips() -> list[list[bool]]:
    """For each case and triangle of the table: True when its normal in the reference tetrahedron (0,0,0), (1,0,0),
    (0,1,0), (0,0,1) points along the gradient of the field (values = 1 at the nodes with the bit, otherwise 0) – such a triangle is reversed.
    """
    P = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]])
    out = []
    for case, tris in enumerate(MT_TABLE):
        v = np.array([(case >> i) & 1 for i in range(4)], dtype=float)
        g = np.linalg.solve(P[1:] - P[0], v[1:] - v[0])
        row = []
        for tri in tris:
            pts = []
            for e in tri:
                a, b = TET_EDGES[e]
                pts.append(0.5 * (P[a] + P[b]))
            n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            row.append(bool(n @ g > 0.0))
        out.append(row)
    return out


_REF_FLIP = _reference_flips()


def _empty() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)


def _edge_keys(n0: np.ndarray, n1: np.ndarray, values: np.ndarray, iso: float) -> np.ndarray:
    """Keys (a, b) of the isosurface points on the edges n0–n1 (K, 2): a = node with values ≥ iso, b = the other; a == b when values[a] == iso."""
    a_is_0 = values[n0] >= iso
    a = np.where(a_is_0, n0, n1)
    b = np.where(a_is_0, n1, n0)
    exact = values[a] == iso
    b = np.where(exact, a, b)
    return np.stack([a, b], axis=1).astype(np.int64)


def _cap_polygons(tris: np.ndarray, values: np.ndarray, iso: float) -> tuple[np.ndarray, np.ndarray]:
    """Part of the triangles of a surface (B, 3 nodes, normal outward) where values ≥ iso (Sutherland–Hodgman against a half-plane).

    Returns the vertex keys of the triangles (T, 3, 2) in the same orientation as the input and the sequence number of the source triangle.
    """
    B = len(tris)
    if B == 0:
        return np.zeros((0, 3, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    inside = values[tris] >= iso                                 # (B, 3)
    slots = np.zeros((B, 6, 2), dtype=np.int64)
    valid = np.zeros((B, 6), dtype=bool)
    for e in range(3):
        i, j = e, (e + 1) % 3
        ni, nj = tris[:, i], tris[:, j]
        slots[:, 2 * e, 0] = ni
        slots[:, 2 * e, 1] = ni
        valid[:, 2 * e] = inside[:, i]
        cross = inside[:, i] != inside[:, j]
        slots[:, 2 * e + 1] = _edge_keys(ni, nj, values, iso)
        valid[:, 2 * e + 1] = cross
    n = valid.sum(axis=1)
    order = np.argsort(~valid, axis=1, kind="stable")             # valid slots first, order kept
    comp = np.take_along_axis(slots, order[:, :, None], axis=1)
    t3 = np.nonzero(n >= 3)[0]
    t4 = np.nonzero(n == 4)[0]
    tri_a = comp[t3][:, [0, 1, 2]]
    tri_b = comp[t4][:, [0, 2, 3]]
    return np.concatenate([tri_a, tri_b]), np.concatenate([t3, t4])


def extract_isosurface(nodes: np.ndarray, tets: np.ndarray, values: np.ndarray,
                       isovalue: float, boundary_tris: np.ndarray | None = None,
                       orient: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Extracts the isosurface for the given value isovalue with the Marching Tetrahedra algorithm.

    Args:
        nodes: (N, 3) node coordinates
        tets: (M, 4) or (M, 10) finite elements (the corner nodes :4 are used)
        values: (N,) scalar values at the nodes (e.g. von Mises stress or temperature)
        isovalue: target value of the isosurface
        boundary_tris: (B, 3) optionally the surface of the mesh (corner nodes, normal outward) – its part where
            values ≥ isovalue is added, and the resulting surface is closed (boundary of the region values ≥ isovalue)
        orient: unify the orientation of the triangles by the gradient (normal against the gradient = outward from values ≥ isovalue)

    Returns:
        iso_vertices: (V, 3) vertices of the triangle mesh of the isosurface
        iso_faces: (F, 3) triangles of the isosurface (0-based indices into iso_vertices)
    """
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    iso = float(isovalue)
    corner_tets = np.asarray(tets)[:, :4]
    M = len(corner_tets)
    key_parts: list[np.ndarray] = []                              # (T, 3, 2) vertex keys by triangle groups

    if M:
        mask = values[corner_tets] >= iso
        case_indices = mask[:, 0] * 1 + mask[:, 1] * 2 + mask[:, 2] * 4 + mask[:, 3] * 8
        for case_id in range(1, 15):
            t_active = np.nonzero(case_indices == case_id)[0]
            if len(t_active) == 0:
                continue
            ct = corner_tets[t_active]
            # orientation: the table is corrected for the reference (positively oriented) tetrahedron; a left-handed one flips it
            left_handed = None
            if orient:
                P = nodes[ct]
                left_handed = np.einsum("ki,ki->k", P[:, 1] - P[:, 0], np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0])) < 0
            for tri, ref_flip in zip(MT_TABLE[case_id], _REF_FLIP[case_id]):
                keys = []
                for e_idx in tri:
                    v0, v1 = TET_EDGES[e_idx]
                    keys.append(_edge_keys(ct[:, v0], ct[:, v1], values, iso))
                keys = np.stack(keys, axis=1)                     # (K, 3, 2)
                if orient:
                    flip = left_handed != ref_flip                # XOR
                    keys[flip] = keys[flip][:, [0, 2, 1]]
                key_parts.append(keys)

    if boundary_tris is not None and len(boundary_tris):
        caps, _ = _cap_polygons(np.asarray(boundary_tris)[:, :3], values, iso)
        if len(caps):
            key_parts.append(caps)

    if not key_parts:
        return _empty()
    all_keys = np.concatenate(key_parts)                          # (F, 3, 2)
    flat = all_keys.reshape(-1, 2)
    # simplified welding: unique pairs (a, b)
    uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    # degenerate triangles (two equal vertices – the isosurface exactly at a node) are dropped
    good = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[good]
    if len(faces) == 0:
        return _empty()
    # unused vertices are discarded
    used, inv2 = np.unique(faces, return_inverse=True)
    faces = inv2.reshape(-1, 3).astype(np.int32)
    verts = _key_points(nodes, values, iso, uniq[used])
    return verts, faces


def _key_points(nodes: np.ndarray, values: np.ndarray, iso: float, keys: np.ndarray) -> np.ndarray:
    """Coordinates of the isosurface points from the keys (a, b): a + t (b − a), t = (v_a − iso) / (v_a − v_b), a == b → node a."""
    a, b = keys[:, 0], keys[:, 1]
    va, vb = values[a], values[b]
    denom = va - vb
    same = a == b
    t = np.where(same, 0.0, (va - iso) / np.where(same, 1.0, denom))
    t = np.clip(t, 0.0, 1.0)
    return nodes[a] + t[:, None] * (nodes[b] - nodes[a])
