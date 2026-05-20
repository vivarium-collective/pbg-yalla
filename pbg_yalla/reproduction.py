"""YallaReproductionProcess — a NumPy *reproduction* of ya||a, not the tool.

This re-implements ya||a's core pair-wise agent-based design — a selectable
pair-wise force kernel plus Euler sub-stepping over N agents — in vectorized
NumPy so it runs on **any** platform (no GPU, no CUDA). It is a clean-room
reproduction of the published model, **not** the upstream binary.

* Upstream: ya||a, https://github.com/germannp/yalla (header-only C++/CUDA),
  Germann et al., *Cell Systems* 2019, doi:10.1016/j.cels.2019.02.007.
* The real-tool bridge lives in :mod:`pbg_yalla.processes`
  (:class:`~pbg_yalla.processes.YallaProcess`); prefer it when a CUDA GPU is
  available. This reproduction is the runs-anywhere fallback used by the demo
  and the default composites.

Known divergences from upstream: single-precision GPU arithmetic is replaced
by float64 NumPy; ya||a's `Tile_solver`/`Grid_solver` neighbour handling is
replaced by a dense O(N^2) pair sum; only a subset of force kernels is ported
(see :mod:`pbg_yalla.force_kernels`). The wall and chemotaxis terms are
extensions not present in the stock ya||a examples.
"""

import numpy as np
from process_bigraph import Process

from pbg_yalla.force_kernels import FORCE_KERNELS
from pbg_yalla.inits import random_sphere, relaxed_sphere
from pbg_yalla.summary import SUMMARY_PORTS, summarize


class YallaReproductionProcess(Process):
    """Time-driven pair-wise ABM process (NumPy reproduction of ya||a).

    Internally owns positions ``X[N, 3]`` and cell types ``type[N]``.
    On each ``update(interval)`` call, advances ``interval / dt`` sub-steps
    using the configured force kernel; optionally proliferates agents.

    Exposes scalar summary outputs through PBG ports; full agent state is
    available via ``snapshot()`` for demos and visualisation.
    """

    config_schema = {
        # Initial population
        'n_cells': {'_type': 'integer', '_default': 200},
        'init': {'_type': 'string', '_default': 'random_sphere'},
        'init_dist': {'_type': 'float', '_default': 0.5},
        'init_relax_steps': {'_type': 'integer', '_default': 50},
        'seed': {'_type': 'integer', '_default': 0},
        # Cell-type assignment
        'n_types': {'_type': 'integer', '_default': 1},
        'type_mode': {'_type': 'string', '_default': 'mixed'},
        # Force kernel
        'force_kernel': {'_type': 'string', '_default': 'spring'},
        'L_0': {'_type': 'float', '_default': 0.5},
        'r_cut': {'_type': 'float', '_default': 0.0},  # 0 = no cut-off (spring only)
        'r_min': {'_type': 'float', '_default': 0.5},
        'r_max': {'_type': 'float', '_default': 1.0},
        'r_eq_same_lo': {'_type': 'float', '_default': 0.7},
        'r_eq_same_hi': {'_type': 'float', '_default': 0.8},
        'r_eq_diff_lo': {'_type': 'float', '_default': 0.8},
        'r_eq_diff_hi': {'_type': 'float', '_default': 0.9},
        'eps': {'_type': 'float', '_default': 1.0},
        # Integrator
        'dt': {'_type': 'float', '_default': 0.05},
        'damping': {'_type': 'float', '_default': 1.0},
        # Proliferation
        'proliferation_rate': {'_type': 'float', '_default': 0.0},
        'n_max': {'_type': 'integer', '_default': 5000},
        'proliferation_mean_dist': {'_type': 'float', '_default': 0.25},
        'proliferation_type_bias': {'_type': 'float', '_default': 0.5},
        # Optional confining wall (soft spherical boundary)
        'wall_radius': {'_type': 'float', '_default': 0.0},
        'wall_strength': {'_type': 'float', '_default': 5.0},
        # Optional chemotactic field: cells climb the gradient of a
        # morphogen c(x) = exp(-|x - source| / decay_length) emanating
        # from a fixed point source. ``chemotaxis_responsive_types`` is
        # a comma-separated string of type indices that respond (e.g.
        # ``'0'`` for type 0 only, ``'0,2'`` for types 0 and 2). Empty
        # string means every type responds. A string is used because
        # process-bigraph config merging treats integer 0 as unset.
        'chemotaxis_strength': {'_type': 'float', '_default': 0.0},
        'chemotaxis_source_x': {'_type': 'float', '_default': 0.0},
        'chemotaxis_source_y': {'_type': 'float', '_default': 0.0},
        'chemotaxis_source_z': {'_type': 'float', '_default': 0.0},
        'chemotaxis_decay_length': {'_type': 'float', '_default': 2.0},
        'chemotaxis_responsive_types': {'_type': 'string', '_default': ''},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._positions = None
        self._types = None
        self._time = 0.0
        self._rng = None
        self._force_fn = None

    def inputs(self):
        return {}

    def outputs(self):
        return dict(SUMMARY_PORTS)

    def _build(self):
        if self._positions is not None:
            return
        cfg = self.config
        self._rng = np.random.default_rng(cfg['seed'])
        if cfg['init'] == 'random_sphere':
            self._positions = random_sphere(
                cfg['n_cells'], cfg['init_dist'], seed=cfg['seed'])
        elif cfg['init'] == 'relaxed_sphere':
            self._positions = relaxed_sphere(
                cfg['n_cells'], cfg['init_dist'],
                seed=cfg['seed'],
                n_relax_steps=cfg['init_relax_steps'])
        else:
            raise ValueError(f'Unknown init: {cfg["init"]}')

        self._types = self._assign_types(cfg['n_cells'])

        if cfg['force_kernel'] not in FORCE_KERNELS:
            raise ValueError(
                f'Unknown force_kernel: {cfg["force_kernel"]!r}. '
                f'Available: {sorted(FORCE_KERNELS)}')
        self._force_fn = FORCE_KERNELS[cfg['force_kernel']]

    def _assign_types(self, n):
        cfg = self.config
        if cfg['n_types'] <= 1:
            return np.zeros(n, dtype=np.int32)
        if cfg['type_mode'] == 'mixed':
            return self._rng.integers(0, cfg['n_types'], size=n).astype(np.int32)
        if cfg['type_mode'] == 'inner_outer':
            centroid = self._positions.mean(axis=0)
            r = np.linalg.norm(self._positions - centroid, axis=1)
            thresh = np.median(r)
            return (r > thresh).astype(np.int32)
        if cfg['type_mode'] == 'hemispheres':
            return (self._positions[:, 0] > 0).astype(np.int32)
        raise ValueError(f'Unknown type_mode: {cfg["type_mode"]}')

    def _force_params(self):
        cfg = self.config
        r_cut = cfg['r_cut'] if cfg['r_cut'] > 0 else np.inf
        return {
            'L_0': cfg['L_0'],
            'r_cut': r_cut,
            'r_min': cfg['r_min'],
            'r_max': cfg['r_max'],
            'r_eq_same_lo': cfg['r_eq_same_lo'],
            'r_eq_same_hi': cfg['r_eq_same_hi'],
            'r_eq_diff_lo': cfg['r_eq_diff_lo'],
            'r_eq_diff_hi': cfg['r_eq_diff_hi'],
            'eps': cfg['eps'],
        }

    def _wall_force(self, positions):
        cfg = self.config
        if cfg['wall_radius'] <= 0:
            return 0.0
        r = np.linalg.norm(positions, axis=1, keepdims=True)
        overflow = np.maximum(r - cfg['wall_radius'], 0.0)
        direction = -positions / np.maximum(r, 1e-9)
        return cfg['wall_strength'] * overflow * direction

    def _chemotaxis_force(self, positions, types):
        """Directed migration up an exponential morphogen gradient.

        Morphogen concentration ``c(x) = exp(-|x - source| / L)``. The
        analytical gradient ``grad c = -c * (x - source) / (L * |x - s|)``
        points from x toward the source. Force on cell i is
        ``F_chemo = -strength * grad c(x_i)`` — pulling up the gradient.
        """
        cfg = self.config
        if cfg['chemotaxis_strength'] <= 0:
            return 0.0
        source = np.array([
            cfg['chemotaxis_source_x'],
            cfg['chemotaxis_source_y'],
            cfg['chemotaxis_source_z'],
        ])
        L = max(cfg['chemotaxis_decay_length'], 1e-6)
        disp = positions - source  # (N, 3) pointing FROM source TO cell
        d = np.linalg.norm(disp, axis=1, keepdims=True)
        safe = np.maximum(d, 1e-9)
        c = np.exp(-d / L)  # concentration at each cell
        # Pull toward source (opposite of disp direction), scaled by c/L
        F = -cfg['chemotaxis_strength'] * c * disp / (L * safe)
        spec = cfg['chemotaxis_responsive_types'].strip()
        if spec:
            allowed = {int(x) for x in spec.split(',') if x.strip()}
            mask = np.isin(types, list(allowed)).astype(np.float64)[:, None]
            F = F * mask
        return F

    def _take_step(self, dt):
        """Compute pair-wise forces and advance positions by one Euler step.

        Mirrors yalla's ``cells.take_step<force>(dt)`` semantics. Adds
        optional wall + chemotaxis terms on top of the pair-wise kernel.
        """
        forces = self._force_fn(self._positions, self._types, self._force_params())
        forces = forces + self._wall_force(self._positions)
        forces = forces + self._chemotaxis_force(self._positions, self._types)
        self._positions = self._positions + dt * forces / self.config['damping']

    def _proliferate(self, dt):
        cfg = self.config
        if cfg['proliferation_rate'] <= 0:
            return
        n = self._positions.shape[0]
        if n >= cfg['n_max']:
            return
        p = 1.0 - np.exp(-cfg['proliferation_rate'] * dt)
        dividing = self._rng.random(n) < p
        n_new = int(min(dividing.sum(), cfg['n_max'] - n))
        if n_new == 0:
            return
        parent_idx = np.where(dividing)[0][:n_new]
        theta = np.arccos(2.0 * self._rng.random(n_new) - 1.0)
        phi = self._rng.random(n_new) * 2.0 * np.pi
        offset = np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ], axis=1) * cfg['proliferation_mean_dist']
        daughters = self._positions[parent_idx] + offset
        daughter_types = self._types[parent_idx].copy()
        if cfg['n_types'] > 1 and cfg['proliferation_type_bias'] < 1.0:
            flip = self._rng.random(n_new) > cfg['proliferation_type_bias']
            daughter_types = np.where(
                flip, 1 - daughter_types, daughter_types).astype(np.int32)
        self._positions = np.vstack([self._positions, daughters])
        self._types = np.concatenate([self._types, daughter_types])

    def _summary(self):
        return summarize(self._positions, self._types, self.config['n_types'])

    def initial_state(self):
        self._build()
        return self._summary()

    def update(self, state, interval):
        self._build()
        dt = self.config['dt']
        n_steps = max(1, int(round(interval / dt)))
        sub_dt = interval / n_steps
        for _ in range(n_steps):
            self._take_step(sub_dt)
            self._proliferate(sub_dt)
            self._time += sub_dt
        return self._summary()

    def snapshot(self):
        """Return the full per-agent state for visualisation.

        Returns a dict with ``positions`` (list of [x,y,z]), ``types``
        (list of ints), ``n_cells``, and ``time``.
        """
        self._build()
        return {
            'positions': self._positions.tolist(),
            'types': self._types.tolist(),
            'n_cells': int(self._positions.shape[0]),
            'time': float(self._time),
        }
