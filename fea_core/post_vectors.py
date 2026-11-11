"""Principal stress vectors (A5: display of results).

Computes the eigenvalues (sigma1 >= sigma2 >= sigma3) and the corresponding eigenvectors
of the symmetric stress tensor at each mesh node:
  - sigma1: largest tensile stress (or the smallest compressive one)
  - sigma2: intermediate principal stress
  - sigma3: largest compressive stress (or the smallest tensile one)

Generates segments / glyphs for 3D visualisation (PyVista / VTK / export) distinguishing
tension (positive, red/orange) and compression (negative, blue).
"""
from __future__ import annotations

import numpy as np
from .results import tensor


def principal_stresses_and_directions(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes the principal stresses sigma1 >= sigma2 >= sigma3 and the normalised direction vectors.

    Args:
        S: stress field of shape (N, 6) [sxx, syy, szz, sxy, syz, szx]

    Returns:
        values: (N, 3) - [sigma1, sigma2, sigma3] sorted descending
        vectors: (N, 3, 3) - vectors[:, :, j] is the unit vector corresponding to values[:, j]
    """
    T = tensor(S)  # (N, 3, 3)
    w, v = np.linalg.eigh(T)  # eigh returns in ascending order w[:, 0] <= w[:, 1] <= w[:, 2]
    
    # Reverse the order to descending: index 0 = sigma1, index 1 = sigma2, index 2 = sigma3
    values = w[..., ::-1]
    vectors = v[..., :, ::-1]
    return values, vectors


def principal_vector_segments(nodes: np.ndarray, S: np.ndarray, which: int = 1,
                              scale: float = 1.0, max_points: int = 2000,
                              min_stress_fraction: float = 0.05) -> dict[str, np.ndarray]:
    """Generates segments showing the direction and magnitude of the chosen principal stress.

    Args:
        nodes: (N, 3) node coordinates
        S: (N, 6) stress tensor at the nodes
        which: 1 for sigma1, 2 for sigma2, 3 for sigma3
        scale: length scale of the vector [mm / MPa]
        max_points: maximum number of points for a clear visualisation
        min_stress_fraction: filter out points with a stress smaller than this fraction of the maximum

    Returns:
        dict containing:
          'starts': (K, 3) starts of the segments
          'ends': (K, 3) ends of the segments
          'values': (K,) magnitude of the principal stress
          'is_tension': (K,) bool mask (True = tension, False = compression)
          'indices': (K,) mesh node indices from 0 (solver node number = index + 1)
          'directions': (K, 3) unit directions of the principal stress (the sign of the eigenvector is arbitrary)
    """
    if which not in (1, 2, 3):
        raise ValueError("which must be 1, 2 or 3")
    
    idx = which - 1
    max_points = max(int(max_points), 2)
    S = np.asarray(S, float)
    # selection by the eigenvalues alone (eigvalsh is cheap); eigenvectors (eigh) only for the selected nodes
    w = np.linalg.eigvalsh(tensor(S))[:, ::-1]            # descending: sigma1 >= sigma2 >= sigma3
    sigma_all = w[:, idx]
    mag_all = np.abs(sigma_all)
    max_mag = float(np.max(mag_all)) if len(mag_all) > 0 else 1.0

    # Selection of significant points
    threshold = min_stress_fraction * max_mag
    valid = np.nonzero(mag_all >= threshold)[0]

    if len(valid) > max_points:
        # Selection of the largest values and a uniform sample
        half = max(max_points // 2, 1)
        sorted_order = np.argsort(mag_all[valid])[::-1]
        top_half = sorted_order[:half]
        rest = sorted_order[half:]
        step = max(len(rest) // half, 1)
        sampled = np.concatenate([top_half, rest[::step]])[:max_points]
        chosen = valid[sampled]
    else:
        chosen = valid

    values_c, vectors_c = principal_stresses_and_directions(S[chosen])
    vals = values_c[:, idx]
    dirs = vectors_c[:, :, idx]
    mag = np.abs(vals)
    pts = nodes[chosen]
    lengths = mag * scale

    # Segment centred at the node: from (pt - 0.5 * len * dir) to (pt + 0.5 * len * dir)
    half_vec = 0.5 * lengths[:, None] * dirs
    starts = pts - half_vec
    ends = pts + half_vec
    
    return {
        "starts": starts,
        "ends": ends,
        "values": vals,
        "is_tension": vals >= 0.0,
        "indices": chosen,
        "directions": dirs,
    }
