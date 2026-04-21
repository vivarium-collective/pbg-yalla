"""Process-bigraph wrapper for a yalla-style pair-wise ABM of morphogenesis.

Provides a NumPy re-implementation of yalla's core design — a pair-wise
force kernel plus Euler/Heun integration over N agents — exposed as a
process-bigraph Process. Kernels are direct ports of yalla example models
(springs.cu, sorting.cu, passive_growth.cu).
"""

from pbg_yalla.processes import YallaProcess
from pbg_yalla.composites import make_yalla_document
from pbg_yalla.force_kernels import FORCE_KERNELS
from pbg_yalla.inits import random_sphere, relaxed_sphere

__all__ = [
    'YallaProcess',
    'make_yalla_document',
    'FORCE_KERNELS',
    'random_sphere',
    'relaxed_sphere',
]
