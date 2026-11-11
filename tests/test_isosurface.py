"""Tests of the isosurface (Marching Tetrahedra): uniform orientation of the normals and watertightness of the closed surface."""
from collections import Counter

import numpy as np

from fea_core.isosurface import extract_isosurface
from fea_core.mesh import Mesh
from fea_core.topo.export import topo_surface, write_binary_stl


def _grid_mesh(nx=6, ny=5, nz=4, size=(10.0, 8.0, 6.0)) -> Mesh:
    """A block divided into tetrahedra (5 tets per cell with alternating orientation, so that the neighbouring cells have matching faces)."""
    xs = np.linspace(0, size[0], nx)
    ys = np.linspace(0, size[1], ny)
    zs = np.linspace(0, size[2], nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

    def nid(i, j, k):
        return (i * ny + j) * nz + k

    tets = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                c = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                if (i + j + k) % 2 == 0:
                    tets += [[c[0], c[1], c[3], c[4]], [c[1], c[2], c[3], c[6]], [c[1], c[4], c[5], c[6]],
                             [c[3], c[4], c[6], c[7]], [c[1], c[3], c[4], c[6]]]
                else:
                    tets += [[c[0], c[1], c[2], c[5]], [c[0], c[2], c[3], c[7]], [c[0], c[5], c[7], c[4]],
                             [c[2], c[5], c[6], c[7]], [c[0], c[2], c[5], c[7]]]
    tets = np.array(tets)
    return Mesh(nodes, tets, np.zeros(len(tets), dtype=int), ["blok"])


def _edge_report(faces: np.ndarray):
    """(the number of edges with a wrong number of triangles, the number of edges without an oppositely oriented pair)."""
    und = Counter()
    directed = Counter()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            und[frozenset((int(u), int(v)))] += 1
            directed[(int(u), int(v))] += 1
    bad_count = sum(1 for n in und.values() if n != 2)
    bad_orient = sum(1 for (u, v), n in directed.items() if n != 1 or directed.get((v, u), 0) != 1)
    return bad_count, bad_orient


def _signed_volume(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


def test_normals_point_against_gradient():
    """All triangles of the open isosurface have the normal against the gradient of the field (outwards from values >= iso)."""
    mesh = _grid_mesh()
    f = mesh.nodes[:, 0] + 0.3 * mesh.nodes[:, 1] + 0.1 * mesh.nodes[:, 2]      # linear field, gradient (1; 0.3; 0.1)
    verts, faces = extract_isosurface(mesh.nodes, mesh.tets, f, 4.2)
    assert len(faces) > 0
    n = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    area = np.linalg.norm(n, axis=1)
    n = n[area > 1e-12] / area[area > 1e-12, None]
    g = np.array([1.0, 0.3, 0.1])
    assert np.all(n @ g < 0.0)


def test_orient_false_keeps_table_orientation_mixed():
    """Without the unification of the orientation the normals are mixed (a check that orient=True changes something)."""
    mesh = _grid_mesh()
    f = mesh.nodes[:, 0] + 0.3 * mesh.nodes[:, 1]
    verts, faces = extract_isosurface(mesh.nodes, mesh.tets, f, 4.2, orient=False)
    n = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    s = n @ np.array([1.0, 0.3, 0.0])
    assert np.any(s > 0) and np.any(s < 0)


def test_closed_surface_is_watertight_and_oriented():
    mesh = _grid_mesh()
    rng = np.random.default_rng(1)
    f = mesh.nodes[:, 0] + 0.6 * np.sin(mesh.nodes[:, 1]) + 0.3 * rng.random(len(mesh.nodes))
    iso = 5.0
    boundary = mesh.boundary_faces()[0][:, :3]
    verts, faces = extract_isosurface(mesh.nodes, mesh.tets, f, iso, boundary_tris=boundary)
    assert len(faces) > 0
    bad_count, bad_orient = _edge_report(faces)
    assert bad_count == 0                       # each edge exactly in 2 triangles
    assert bad_orient == 0                      # and oriented oppositely to each other (a closed oriented manifold)
    assert _signed_volume(verts, faces) > 0      # normals outwards


def test_topo_surface_half_box_volume():
    """Half of the block solid (x <= 5): the closed surface has a positive volume close to half of the block and is watertight."""
    mesh = _grid_mesh(nx=11, ny=5, nz=4, size=(10.0, 8.0, 6.0))
    cent = mesh.nodes[mesh.tets[:, :4]].mean(axis=1)
    rho = np.where(cent[:, 0] < 5.0, 1.0, 0.0)
    verts, faces = topo_surface(mesh, rho, 0.5)
    assert _edge_report(faces) == (0, 0)
    vol = _signed_volume(verts, faces)
    assert 0.35 * 480.0 < vol < 0.65 * 480.0
    # ... and the saving into STL does not break
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = write_binary_stl(Path(d) / "t.stl", verts, faces)
        assert (p.stat().st_size - 84) // 50 == len(faces)


def test_all_solid_gives_boundary_only():
    """Field everywhere above the threshold: no boundary, with a boundary the whole surface of the mesh (closed)."""
    mesh = _grid_mesh(nx=3, ny=3, nz=3)
    f = np.ones(len(mesh.nodes))
    assert len(extract_isosurface(mesh.nodes, mesh.tets, f, 0.5)[1]) == 0
    boundary = mesh.boundary_faces()[0][:, :3]
    verts, faces = extract_isosurface(mesh.nodes, mesh.tets, f, 0.5, boundary_tris=boundary)
    assert len(faces) == len(boundary)
    assert _edge_report(faces) == (0, 0)
    assert abs(_signed_volume(verts, faces) - 10.0 * 8.0 * 6.0) < 1e-9
    assert _signed_volume(verts, faces) > 0


def test_iso_through_nodes_exact_value():
    """A field value exactly equal to the threshold at the nodes (v == iso) must not break the watertightness."""
    mesh = _grid_mesh(nx=5, ny=4, nz=3)
    f = np.where(mesh.nodes[:, 0] <= 5.0, 1.0, 0.0)
    f[mesh.nodes[:, 0] == 5.0] = 0.5
    boundary = mesh.boundary_faces()[0][:, :3]
    verts, faces = extract_isosurface(mesh.nodes, mesh.tets, f, 0.5, boundary_tris=boundary)
    assert _edge_report(faces) == (0, 0)
    assert _signed_volume(verts, faces) > 0
