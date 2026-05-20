"""YallaProcess — process-bigraph bridge to the *actual* ya||a simulator.

This is the headline wrapper: its ``update()`` drives the real ya||a GPU solver
by generating a ``.cu`` model, compiling it with ``nvcc`` against a ya||a
checkout, and round-tripping agent state through the compiled binary on every
step (see :mod:`pbg_yalla.yalla_native`).

ya||a is **GPU-only C++/CUDA** — it cannot run without an NVIDIA GPU and the
CUDA toolkit. Where those are absent (e.g. macOS / CPU hosts) this Process
raises :class:`~pbg_yalla.yalla_native.YallaCudaUnavailable` with guidance,
and **does not** silently fall back to anything. For a runs-anywhere NumPy
reproduction of the same model, use
:class:`pbg_yalla.reproduction.YallaReproductionProcess`.

Construction and ``initial_state()`` work everywhere (the initial population is
built in NumPy); only stepping requires CUDA. The bridge integrates a fixed
population under one pair-wise force kernel
(:data:`~pbg_yalla.yalla_native.SUPPORTED_KERNELS`); proliferation and the
wall/chemotaxis extensions live only in the reproduction.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Process

from pbg_yalla import yalla_native
from pbg_yalla.inits import random_sphere, relaxed_sphere
from pbg_yalla.reproduction import YallaReproductionProcess
from pbg_yalla.summary import SUMMARY_PORTS, summarize


class YallaProcess(Process):
    """Time-driven bridge to the real ya||a GPU solver.

    Shares :class:`YallaReproductionProcess`'s config surface (so a composite
    can target either class with the same parameters) plus ``cuda_arch``. Only
    the keys the native solver consumes — population, init, force kernel,
    ``L_0``/``r_min``/``r_max``, ``dt``, ``n_max`` — affect the result.
    """

    config_schema = {
        **YallaReproductionProcess.config_schema,
        # nvcc target architecture; override per GPU (e.g. sm_70, sm_80).
        'cuda_arch': {'_type': 'string', '_default': yalla_native.DEFAULT_CUDA_ARCH},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._positions = None
        self._types = None
        self._time = 0.0
        self._binary = None

    def inputs(self):
        return {}

    def outputs(self):
        return dict(SUMMARY_PORTS)

    # -- initial population (NumPy; no GPU needed) -------------------------

    def _build_initial(self):
        if self._positions is not None:
            return
        cfg = self.config
        if cfg['init'] == 'random_sphere':
            self._positions = random_sphere(
                cfg['n_cells'], cfg['init_dist'], seed=cfg['seed'])
        elif cfg['init'] == 'relaxed_sphere':
            self._positions = relaxed_sphere(
                cfg['n_cells'], cfg['init_dist'], seed=cfg['seed'],
                n_relax_steps=cfg['init_relax_steps'])
        else:
            raise ValueError(f'Unknown init: {cfg["init"]}')
        self._types = self._assign_types(cfg['n_cells'])

    def _assign_types(self, n):
        cfg = self.config
        if cfg['n_types'] <= 1:
            return np.zeros(n, dtype=np.int32)
        rng = np.random.default_rng(cfg['seed'])
        if cfg['type_mode'] == 'mixed':
            return rng.integers(0, cfg['n_types'], size=n).astype(np.int32)
        if cfg['type_mode'] == 'inner_outer':
            centroid = self._positions.mean(axis=0)
            r = np.linalg.norm(self._positions - centroid, axis=1)
            return (r > np.median(r)).astype(np.int32)
        if cfg['type_mode'] == 'hemispheres':
            return (self._positions[:, 0] > 0).astype(np.int32)
        raise ValueError(f'Unknown type_mode: {cfg["type_mode"]}')

    # -- native solver (requires CUDA) ------------------------------------

    def _ensure_binary(self):
        if self._binary is not None:
            return self._binary
        cfg = self.config
        source = yalla_native.generate_cu_source(
            force_kernel=cfg['force_kernel'],
            L_0=cfg['L_0'], r_min=cfg['r_min'], r_max=cfg['r_max'],
            n_max=max(cfg['n_max'], cfg['n_cells']))
        self._binary = yalla_native.compile_model(source, arch=cfg['cuda_arch'])
        return self._binary

    def _summary(self):
        return summarize(self._positions, self._types, self.config['n_types'])

    def initial_state(self):
        self._build_initial()
        return self._summary()

    def update(self, state, interval):
        self._build_initial()
        binary = self._ensure_binary()
        dt = self.config['dt']
        n_steps = max(1, int(round(interval / dt)))
        self._positions, self._types = yalla_native.run_steps(
            binary, self._positions, self._types, n_steps=n_steps, dt=dt)
        self._time += n_steps * dt
        return self._summary()

    def snapshot(self):
        """Full per-agent state for visualisation (mirrors the reproduction)."""
        self._build_initial()
        X = np.asarray(self._positions, dtype=float).reshape(-1, 3)
        return {
            'positions': X.tolist(),
            'types': np.asarray(self._types).reshape(-1).astype(int).tolist(),
            'n_cells': int(X.shape[0]),
            'time': float(self._time),
        }
