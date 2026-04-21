"""Initial condition generators — ports of yalla's inits.cuh.

Each function returns an (N, 3) float64 numpy array of agent positions.
"""

import numpy as np


def random_sphere(n_cells, dist_to_nb=0.5, seed=None):
    """Distribute agents uniformly in a sphere.

    Port of `random_sphere` from yalla's inits.cuh. The sphere radius is
    chosen so that the expected nearest-neighbour distance equals
    ``dist_to_nb`` assuming random close packing (packing fraction 0.64).
    """
    rng = np.random.default_rng(seed)
    r_max = ((n_cells / 0.64) ** (1.0 / 3.0)) * dist_to_nb / 2.0
    u = rng.random(n_cells)
    r = r_max * np.cbrt(u)
    theta = np.arccos(2.0 * rng.random(n_cells) - 1.0)
    phi = rng.random(n_cells) * 2.0 * np.pi
    positions = np.empty((n_cells, 3), dtype=np.float64)
    positions[:, 0] = r * np.sin(theta) * np.cos(phi)
    positions[:, 1] = r * np.sin(theta) * np.sin(phi)
    positions[:, 2] = r * np.cos(theta)
    return positions


def random_disk(n_cells, dist_to_nb=0.5, seed=None):
    """Distribute agents uniformly in a disk in the y-z plane."""
    rng = np.random.default_rng(seed)
    r_max = ((n_cells / 0.9069) ** 0.5) * dist_to_nb / 2.0
    r = r_max * np.sqrt(rng.random(n_cells))
    phi = rng.random(n_cells) * 2.0 * np.pi
    positions = np.zeros((n_cells, 3), dtype=np.float64)
    positions[:, 1] = r * np.sin(phi)
    positions[:, 2] = r * np.cos(phi)
    return positions


def relaxed_sphere(n_cells, mean_dist=0.75, seed=None, n_relax_steps=200):
    """Random sphere, then relaxed under a short-range repulsion.

    Port of yalla's ``relaxed_sphere`` — random placement followed by
    repeated repulsion to avoid overlaps. Uses a simple spring-like
    kernel to push neighbours apart until the mean nearest-neighbour
    distance roughly matches ``mean_dist``.
    """
    positions = random_sphere(n_cells, mean_dist, seed=seed)
    dt = 0.05
    L0 = mean_dist
    r_cut = mean_dist * 1.5
    for _ in range(n_relax_steps):
        disp = positions[:, None, :] - positions[None, :, :]
        dist = np.linalg.norm(disp, axis=-1)
        np.fill_diagonal(dist, np.inf)
        mask = (dist < r_cut) & np.isfinite(dist)
        safe = np.where(mask, dist, 1.0)
        inv = np.where(mask, (L0 - dist) / safe, 0.0)
        F = (disp * inv[..., None]).sum(axis=1)
        positions = positions + dt * F
    return positions
