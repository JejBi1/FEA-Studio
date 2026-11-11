"""Unit tests for the export of the topology optimisation to STL and the transfer of the densities to the nodes (fea_core.topo.export)."""
from pathlib import Path
import numpy as np
import pytest

from fea_core.mesh import Mesh
from fea_core.topo.export import densities_to_nodes, export_topo_stl, write_binary_stl


def _create_two_tet_mesh() -> Mesh:
    """Creates 2 tetrahedra sharing a face."""
    nodes = np.array([
        [0.0, 0.0, 0.0],   # 0
        [10.0, 0.0, 0.0],  # 1
        [0.0, 10.0, 0.0],  # 2
        [0.0, 0.0, 10.0],  # 3
        [10.0, 10.0, 10.0] # 4
    ], dtype=float)
    tets = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ], dtype=int)
    tet_part = np.array([0, 0], dtype=int)
    return Mesh(nodes, tets, tet_part, ["part1"])


def test_densities_to_nodes_uniform():
    mesh = _create_two_tet_mesh()
    elem_dens = np.array([0.7, 0.7])
    nodal_dens = densities_to_nodes(mesh, elem_dens)
    assert len(nodal_dens) == 5
    assert np.allclose(nodal_dens, 0.7)


def test_densities_to_nodes_gradient():
    mesh = _create_two_tet_mesh()
    elem_dens = np.array([1.0, 0.0])
    nodal_dens = densities_to_nodes(mesh, elem_dens)
    # Node 0 belongs only to element 0 -> density 1.0
    assert nodal_dens[0] == pytest.approx(1.0)
    # Node 4 belongs only to element 1 -> density 0.0
    assert nodal_dens[4] == pytest.approx(0.0)
    # Shared nodes 1, 2, 3 lie between the elements
    for n in (1, 2, 3):
        assert 0.0 < nodal_dens[n] < 1.0


def test_densities_to_nodes_with_filter():
    mesh = _create_two_tet_mesh()
    elem_dens = np.array([1.0, 0.2])
    nodal_dens = densities_to_nodes(mesh, elem_dens, filter_radius=15.0)
    assert len(nodal_dens) == 5
    assert np.all((nodal_dens >= 0.2) & (nodal_dens <= 1.0))


def test_write_binary_stl_empty(tmp_path: Path):
    stl_file = tmp_path / "empty.stl"
    verts = np.zeros((0, 3))
    faces = np.zeros((0, 3), dtype=int)
    res_path = write_binary_stl(stl_file, verts, faces)
    assert res_path.exists()
    assert res_path.stat().st_size == 84  # 80 bytes header + 4 bytes face count (0)


def test_export_topo_stl(tmp_path: Path):
    # A simple block divided into tetrahedra
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [10.0, 0.0, 10.0],
        [10.0, 10.0, 10.0],
        [0.0, 10.0, 10.0],
    ])
    tets = np.array([
        [0, 1, 2, 5],
        [0, 2, 3, 7],
        [0, 5, 7, 4],
        [2, 5, 6, 7],
        [0, 2, 5, 7],
    ])
    mesh = Mesh(nodes, tets, np.zeros(len(tets), dtype=int), ["box"])

    # Polovina solid, polovina void
    elem_dens = np.array([1.0, 1.0, 0.0, 0.0, 0.5])
    stl_path = tmp_path / "topo_result.stl"

    out = export_topo_stl(mesh, elem_dens, stl_path, isovalue=0.5)
    assert out.exists()
    size = out.stat().st_size
    # 84 bytes of the header + N * 50 bytes
    assert size >= 84
    num_faces = (size - 84) // 50
    assert (size - 84) % 50 == 0
    assert num_faces > 0
