"""Spatial filter of the sensitivities (strain energy of the elements).

Fast search of the neighbours within the distance filter_radius using scipy.spatial.cKDTree
on the centroids of the elements.
Computation of the weighted mean of the sensitivity:
    alpha_tilde_e = sum_i(w_ei * alpha_i) / sum_i(w_ei), where w_ei = max(0, filter_radius - dist(e, i)).
Averaging over the history to stabilise against oscillations:
    alpha_curr = 0.5 * (alpha_curr + alpha_prev).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


class SensitivityFilter:
    """Spatial filter of the sensitivities with a cache of the neighbours and a sparse weight matrix.

    Allows a very fast filtering of the sensitivities by the vector multiplication of the sparse matrix (W @ alpha).
    """

    def __init__(
        self,
        centroids: np.ndarray,
        filter_radius: float,
        volumes: np.ndarray | None = None,
    ) -> None:
        """centroids: (M, 3) coordinates of the centroids of the elements in mm
        filter_radius: radius of the filter r_min in mm
        volumes: (M,) optional vector of the volumes of the elements
        """
        self.centroids = np.ascontiguousarray(centroids, dtype=float)
        self.radius = float(filter_radius)
        self.num_elements = len(centroids)
        self.volumes = np.ascontiguousarray(volumes, dtype=float) if volumes is not None else None

        self.alpha_prev: np.ndarray | None = None

        if self.centroids.ndim != 2 or (self.num_elements > 0 and self.centroids.shape[1] != 3):
            raise ValueError(f"Centroids must have shape (M, 3), shape {self.centroids.shape} given")

        # Building of the tree and of the sparse weight matrix
        if self.num_elements > 0:
            self._tree = cKDTree(self.centroids)
            self._W = self._build_weight_matrix()
        else:
            self._tree = None
            self._W = sparse.csr_matrix((0, 0), dtype=float)

    def _build_weight_matrix(self) -> sparse.csr_matrix:
        """Builds the normalised sparse weight matrix W for alpha_tilde = W @ alpha.

        w_ei = max(0, r − d_ei) · V_i (the volume of the element i – a non-uniform mesh must not favour a densely meshed region),
        the rows normalised to the sum 1. The pairs are searched once vectorised (sparse_distance_matrix), the diagonal
        (d = 0, weight r·V_e > 0) is always present.
        """
        if self.radius <= 0.0 or self.num_elements == 0:
            return sparse.eye(self.num_elements, format="csr", dtype=float)

        D = self._tree.sparse_distance_matrix(self._tree, self.radius, output_type="ndarray")
        i, j, d = D["i"], D["j"], D["v"]
        w = np.maximum(0.0, self.radius - d)
        if self.volumes is not None:
            w = w * self.volumes[j]
        s = np.bincount(i, weights=w, minlength=self.num_elements)
        keep = w > 0.0
        i, j, w = i[keep], j[keep], w[keep]
        # an element without neighbours with a positive weight (zero volume …) = identity
        lonely = np.nonzero(s <= 0.0)[0]
        s_safe = np.where(s > 0.0, s, 1.0)
        rows = np.concatenate([i, lonely])
        cols = np.concatenate([j, lonely])
        vals = np.concatenate([w / s_safe[i], np.ones(len(lonely))])
        return sparse.csr_matrix((vals, (rows, cols)), shape=(self.num_elements, self.num_elements), dtype=float)

    def filter(self, alpha: np.ndarray, use_history: bool = True) -> np.ndarray:
        """Applies the spatial filter and optionally the averaging over the history.

        Parameters:
            alpha: (M,) vector of the sensitivities before the filtering
            use_history: whether to use the averaging over the history with the previous iteration
        Returns:
            alpha_curr: (M,) filtered vector of the sensitivities
        """
        alpha = np.ascontiguousarray(alpha, dtype=float)
        if len(alpha) != self.num_elements:
            raise ValueError(f"Length of the sensitivity vector ({len(alpha)}) does not match the number of elements ({self.num_elements})")

        if self.num_elements == 0:
            return np.zeros(0, dtype=float)

        # 1. Spatial filter
        alpha_tilde = self._W.dot(alpha)

        # 2. Averaging over the history: alpha_curr = 0.5 * (alpha_curr + alpha_prev)
        if use_history and self.alpha_prev is not None:
            alpha_curr = 0.5 * (alpha_tilde + self.alpha_prev)
        else:
            alpha_curr = alpha_tilde.copy()

        self.alpha_prev = alpha_curr.copy()
        return alpha_curr

    def reset(self) -> None:
        """Resets the history memory of the sensitivities."""
        self.alpha_prev = None


def filter_sensitivities(
    centroids: np.ndarray,
    alpha: np.ndarray,
    filter_radius: float,
    alpha_prev: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One-off function for the spatial filtering and the averaging over the history of the sensitivities.

    Parameters:
        centroids: (M, 3) coordinates of the centroids of the elements
        alpha: (M,) vector of the sensitivities
        filter_radius: radius of the filter
        alpha_prev: (M,) optional previous vector of the sensitivities from the last iteration
    Returns:
        (alpha_curr, alpha_updated_history)
    """
    filt = SensitivityFilter(centroids, filter_radius)
    if alpha_prev is not None:
        filt.alpha_prev = np.ascontiguousarray(alpha_prev, dtype=float)
    res = filt.filter(alpha, use_history=(alpha_prev is not None))
    return res, filt.alpha_prev
