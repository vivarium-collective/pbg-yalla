"""Scalar summary metrics for a population of agents.

Shared by both the real-yalla bridge (:class:`pbg_yalla.processes.YallaProcess`)
and the NumPy reproduction
(:class:`pbg_yalla.reproduction.YallaReproductionProcess`) so the two expose
*identical* output ports regardless of which engine produced the positions.

A summary is computed purely from ``positions`` (``[N, 3]``) and integer
``types`` (``[N]``) — nothing else — so it works on coordinates parsed back
from a yalla VTK file just as well as on a NumPy integration.
"""
from __future__ import annotations

import numpy as np

#: Output-port schema both processes advertise. Reused so the two engines can
#: never drift out of sync.
SUMMARY_PORTS = {
    'n_cells': 'overwrite[integer]',
    'gyration_radius': 'overwrite[float]',
    'mean_neighbor_distance': 'overwrite[float]',
    'sorting_score': 'overwrite[float]',
    'type_radial_spread': 'overwrite[float]',
    'center_x': 'overwrite[float]',
    'center_y': 'overwrite[float]',
    'center_z': 'overwrite[float]',
}

_EMITTABLE = {
    'n_cells': 'integer',
    'gyration_radius': 'float',
    'mean_neighbor_distance': 'float',
    'sorting_score': 'float',
    'type_radial_spread': 'float',
}


def summarize(positions: np.ndarray, types: np.ndarray, n_types: int = 1) -> dict:
    """Reduce an agent population to the scalar PBG output ports.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
    types     : ndarray, shape (N,), integer cell-type labels
    n_types   : number of distinct types (gates the sorting metrics)

    Returns
    -------
    dict with keys matching :data:`SUMMARY_PORTS`.
    """
    X = np.asarray(positions, dtype=float).reshape(-1, 3)
    t = np.asarray(types).reshape(-1)
    n = X.shape[0]

    if n == 0:
        return {
            'n_cells': 0, 'gyration_radius': 0.0,
            'mean_neighbor_distance': 0.0, 'sorting_score': 0.0,
            'type_radial_spread': 0.0,
            'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0,
        }

    centroid = X.mean(axis=0)
    radii = np.linalg.norm(X - centroid, axis=1)
    rg = float(np.sqrt((radii ** 2).mean()))

    mean_nn = 0.0
    sort_score = 1.0
    radial_spread = 0.0
    if n >= 2:
        disp = X[:, None, :] - X[None, :, :]
        d = np.linalg.norm(disp, axis=-1)
        np.fill_diagonal(d, np.inf)
        nearest = d.min(axis=1)
        mean_nn = float(nearest.mean())
        if n_types > 1 and len(np.unique(t)) > 1:
            # Local same-type neighbour fraction (k=3). Diagnostic only —
            # sensitive to geometry; use type_radial_spread for inner/outer
            # sorting.
            k = min(3, n - 1)
            nn_idx = np.argpartition(d, kth=k, axis=1)[:, :k]
            nn_types = t[nn_idx]
            same = (nn_types == t[:, None])
            sort_score = float(same.mean())
            # Radial spread: mean radial distance of type-1 minus type-0.
            # Positive when type-1 is pushed outward (the canonical
            # differential-adhesion outcome).
            mask0 = t == 0
            mask1 = t == 1
            r0 = float(radii[mask0].mean()) if mask0.any() else 0.0
            r1 = float(radii[mask1].mean()) if mask1.any() else 0.0
            radial_spread = r1 - r0

    return {
        'n_cells': int(n),
        'gyration_radius': rg,
        'mean_neighbor_distance': mean_nn,
        'sorting_score': sort_score,
        'type_radial_spread': radial_spread,
        'center_x': float(centroid[0]),
        'center_y': float(centroid[1]),
        'center_z': float(centroid[2]),
    }
