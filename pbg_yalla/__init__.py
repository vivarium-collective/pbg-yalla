"""Process-bigraph wrapper for ya||a, a pair-wise ABM of morphogenesis.

Two process classes are provided:

* :class:`~pbg_yalla.processes.YallaProcess` — the headline bridge to the
  **actual** ya||a GPU solver (compiles a ``.cu`` with ``nvcc`` and drives the
  real binary). Requires CUDA + an NVIDIA GPU.
* :class:`~pbg_yalla.reproduction.YallaReproductionProcess` — a NumPy
  **reproduction** of ya||a's design (pair-wise force kernel + Euler stepping)
  that runs anywhere, used by the demo and default composites.

Both expose the same scalar summary ports via :mod:`pbg_yalla.summary`.
"""

from pbg_yalla.processes import YallaProcess
from pbg_yalla.reproduction import YallaReproductionProcess
from pbg_yalla.yalla_native import YallaCudaUnavailable
from pbg_yalla.composites import make_yalla_document
from pbg_yalla.force_kernels import FORCE_KERNELS
from pbg_yalla.inits import random_sphere, relaxed_sphere

__all__ = [
    'YallaProcess',
    'YallaReproductionProcess',
    'YallaCudaUnavailable',
    'make_yalla_document',
    'FORCE_KERNELS',
    'random_sphere',
    'relaxed_sphere',
]
