"""Vectorized pair-wise force kernels — direct ports of yalla examples.

Each kernel has signature::

    force(positions, types, params) -> forces

where ``positions`` is an (N, 3) array, ``types`` an (N,) int array,
and ``forces`` the resulting (N, 3) array of net per-agent forces.

These follow yalla's CUDA pair-wise design (a ``__device__`` function of
``(Xi, r, dist, i, j)`` returning ``dF``), but computed in a single
vectorized pass over all pairs using NumPy.
"""

import numpy as np


def _pair_distances(positions):
    """Return (disp, dist, mask_self) where disp[i,j] = X[i] - X[j]."""
    disp = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(disp, axis=-1)
    np.fill_diagonal(dist, np.inf)  # exclude self, prevent div-by-zero
    return disp, dist


def spring(positions, types, params):
    """Harmonic spring between every pair of agents.

    Port of ``spring`` from examples/springs.cu:

        dF = r * (L_0 - dist) / dist

    Parameters
    ----------
    params : dict with ``L_0``, optional ``r_cut``.
    """
    L_0 = params.get('L_0', 0.5)
    r_cut = params.get('r_cut', np.inf)
    disp, dist = _pair_distances(positions)
    active = (dist < r_cut) & np.isfinite(dist)
    safe = np.where(active, dist, 1.0)
    coeff = np.where(active, (L_0 - dist) / safe, 0.0)
    return (disp * coeff[..., None]).sum(axis=1)


def differential_adhesion(positions, types, params):
    """Cell sorting via type-dependent force strengths.

    Port of ``differential_adhesion`` from examples/sorting.cu::

        if dist > r_max: dF = 0
        strength = (1 + 2*(j<half)) * (1 + 2*(i<half))
        F = 2*(r_min - dist)*(r_max - dist) + (r_max - dist)^2
        dF = strength * r * F / dist

    In yalla the strength factor is hard-coded by cell index; here we
    generalise it to ``type==0`` vs ``type==1``. Agents whose type is 0
    feel a 3x stronger pairwise force (yielding the canonical sorting
    behaviour).
    """
    r_min = params.get('r_min', 0.5)
    r_max = params.get('r_max', 1.0)
    disp, dist = _pair_distances(positions)
    active = (dist < r_max) & np.isfinite(dist)
    safe = np.where(active, dist, 1.0)

    type_boost_i = np.where(types == 0, 3.0, 1.0)
    type_boost_j = np.where(types == 0, 3.0, 1.0)
    strength = type_boost_i[:, None] * type_boost_j[None, :]

    F = 2.0 * (r_min - dist) * (r_max - dist) + (r_max - dist) ** 2
    coeff = np.where(active, strength * F / safe, 0.0)
    return (disp * coeff[..., None]).sum(axis=1)


def relu(positions, types, params):
    """ReLU-shaped adhesion/repulsion, the canonical yalla morphogenesis kernel.

    Port of ``relu_force`` used across passive_growth.cu, migration.cu,
    intercalation.cu::

        F = max(r_eq_in - dist, 0) * 2 - max(dist - r_eq_out, 0)
        dF = r * F / dist   for dist < r_max, else 0

    Two cells of the same type have equilibrium window
    [r_eq_same_lo, r_eq_same_hi]; different types have
    [r_eq_diff_lo, r_eq_diff_hi] — mirroring passive_growth.cu.
    """
    r_max = params.get('r_max', 1.0)
    r_eq_same_lo = params.get('r_eq_same_lo', 0.7)
    r_eq_same_hi = params.get('r_eq_same_hi', 0.8)
    r_eq_diff_lo = params.get('r_eq_diff_lo', 0.8)
    r_eq_diff_hi = params.get('r_eq_diff_hi', 0.9)

    disp, dist = _pair_distances(positions)
    active = (dist < r_max) & np.isfinite(dist)
    safe = np.where(active, dist, 1.0)

    same = types[:, None] == types[None, :]
    lo = np.where(same, r_eq_same_lo, r_eq_diff_lo)
    hi = np.where(same, r_eq_same_hi, r_eq_diff_hi)

    F = np.maximum(lo - dist, 0.0) * 2.0 - np.maximum(dist - hi, 0.0)
    coeff = np.where(active, F / safe, 0.0)
    return (disp * coeff[..., None]).sum(axis=1)


def lennard_jones_soft(positions, types, params):
    """Soft Lennard-Jones-style potential with a finite cut-off.

    Useful for stable relaxation of dense random-packed agents:

        u(d) = eps * ((r_min/d)^6 - 2*(r_min/d)^3)   for d < r_max
    """
    r_min = params.get('r_min', 0.7)
    r_max = params.get('r_max', 1.4)
    eps = params.get('eps', 1.0)
    disp, dist = _pair_distances(positions)
    active = (dist < r_max) & np.isfinite(dist)
    safe = np.where(active, dist, 1.0)
    x = r_min / safe
    F_mag = 12.0 * eps * (x ** 6 - x ** 3) / safe
    coeff = np.where(active, F_mag, 0.0)
    return (disp * coeff[..., None]).sum(axis=1)


FORCE_KERNELS = {
    'spring': spring,
    'differential_adhesion': differential_adhesion,
    'relu': relu,
    'lennard_jones_soft': lennard_jones_soft,
}
