"""Export of the topology optimisation to STL and conversion of the element densities to the nodal densities."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from fea_core.mesh import Mesh
from fea_core.loads import element_volumes
from fea_core.isosurface import extract_isosurface


def densities_to_nodes(
    mesh: Mesh,
    element_densities: np.ndarray,
    filter_radius: float | None = None,
) -> np.ndarray:
    """Converts the densities from the elements to the nodes (weighted mean over the volumes and the distances of the attached elements).

    Args:
        mesh: mesh of the model (contains nodes and tets)
        element_densities: (M,) array of the element densities (values in the range 0..1)
        filter_radius: optional radius of the smoothing filter [mm]

    Returns:
        node_densities: (N,) array of the densities at the nodes of the mesh
    """
    N = len(mesh.nodes)
    M = len(mesh.tets)
    if M == 0 or N == 0:
        return np.zeros(N, dtype=np.float64)

    elem_dens = np.asarray(element_densities, dtype=np.float64)
    if len(elem_dens) != M:
        raise ValueError(f"Number of element densities ({len(elem_dens)}) does not match the number of mesh elements ({M})")

    # centroids of the elements (from the corner nodes :4)
    corner_tets = mesh.tets[:, :4]
    centroids = mesh.nodes[corner_tets].mean(axis=1)  # (M, 3)

    # volumes of the elements
    vols = element_volumes(mesh)  # (M,)

    # mapping of the nodes to the attached elements
    nn_per_tet = mesh.tets.shape[1]
    elem_indices = np.repeat(np.arange(M), nn_per_tet)
    node_indices = mesh.tets.ravel()

    # distances from the nodes to the centroids of the elements
    node_coords = mesh.nodes[node_indices]
    elem_cent = centroids[elem_indices]
    dists = np.linalg.norm(node_coords - elem_cent, axis=1)

    elem_v = vols[elem_indices]

    if filter_radius is not None and filter_radius > 0:
        # weight by the distance and the filter radius: max(0, R - d) * V
        w = elem_v * np.maximum(0.0, float(filter_radius) - dists)
        w_sum = np.bincount(node_indices, weights=w, minlength=N)
        zero_w_mask = (w_sum == 0)
        if np.any(zero_w_mask):
            w_inv = elem_v / np.maximum(dists, 1e-12)
            active_fallback = zero_w_mask[node_indices]
            w = np.where(active_fallback, w_inv, w)
    else:
        # weighted mean over the volumes and the inverse distances
        w = elem_v / np.maximum(dists, 1e-12)

    num = np.bincount(node_indices, weights=w * elem_dens[elem_indices], minlength=N)
    den = np.bincount(node_indices, weights=w, minlength=N)

    node_densities = np.zeros(N, dtype=np.float64)
    valid = den > 0
    node_densities[valid] = num[valid] / den[valid]
    return node_densities


def write_binary_stl(
    path: Path | str,
    vertices: np.ndarray,
    faces: np.ndarray,
    header_text: str = "FEA Studio Topology Optimization STL",
) -> Path:
    """Writes a triangle mesh (vertices + faces) into a standard binary STL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    num_faces = len(faces)

    header = header_text.encode("ascii", errors="replace")[:80].ljust(80, b"\0")

    if num_faces == 0:
        with open(path, "wb") as f:
            f.write(header)
            f.write(np.uint32(0).tobytes())
        return path

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len = np.where(norm_len < 1e-12, 1.0, norm_len)
    normals = normals / norm_len

    STL_RECORD = np.dtype([
        ("n", "<f4", 3),
        ("v", "<f4", (3, 3)),
        ("a", "<u2"),
    ])

    rec = np.empty(num_faces, dtype=STL_RECORD)
    rec["n"] = normals.astype(np.float32)
    rec["v"][:, 0] = v0.astype(np.float32)
    rec["v"][:, 1] = v1.astype(np.float32)
    rec["v"][:, 2] = v2.astype(np.float32)
    rec["a"] = 0

    with open(path, "wb") as f:
        f.write(header)
        f.write(np.uint32(num_faces).tobytes())
        rec.tofile(f)

    return path


def topo_surface(
    mesh: Mesh,
    element_densities: np.ndarray,
    isovalue: float = 0.5,
    filter_radius: float | None = None,
    closed: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Surface of the solid region (nodal density ≥ isovalue): (vertices, triangles), normals outward from the material.

    closed = True adds to the isosurface the part of the surface of the mesh where the density is ≥ isovalue (the isosurface ρ = 0.5 is open at the
    boundary of the region) → a watertight surface suitable for STL / 3D printing.
    """
    node_densities = densities_to_nodes(mesh, element_densities, filter_radius=filter_radius)
    boundary = None
    if closed and len(mesh.tets):
        boundary = mesh.boundary_faces()[0][:, :3]
    return extract_isosurface(mesh.nodes, mesh.tets, node_densities, isovalue=float(isovalue), boundary_tris=boundary)


def export_topo_stl(
    mesh: Mesh,
    element_densities: np.ndarray,
    output_path: Path | str,
    isovalue: float = 0.5,
    filter_radius: float | None = None,
    closed: bool = True,
) -> Path:
    """Uses fea_core.isosurface.extract_isosurface on the nodal densities with the threshold isovalue and saves a binary STL.

    Args:
        mesh: mesh of the model (contains nodes and tets)
        element_densities: (M,) array of the element densities
        output_path: path to the output STL file
        isovalue: threshold value for the isosurface (default 0.5)
        filter_radius: optional filter radius for the conversion of the densities to the nodes
        closed: close the surface by the part of the surface of the mesh with density ≥ isovalue (watertight body, normals outward)

    Returns:
        Path of the created STL file
    """
    iso_verts, iso_faces = topo_surface(mesh, element_densities, isovalue, filter_radius, closed)
    output_path = Path(output_path)
    write_binary_stl(output_path, iso_verts, iso_faces)
    return output_path
