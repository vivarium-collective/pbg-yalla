# pbg-yalla

Process-bigraph wrapper for a NumPy port of [ya||a](https://github.com/germannp/yalla), a pair-wise agent-based model for morphogenesis.

**[View Interactive Demo Report](https://vivarium-collective.github.io/pbg-yalla/)** — spring relaxation, differential-adhesion cell sorting, and proliferation growth with Three.js 3D agent viewers, Plotly charts, and bigraph architecture diagrams.

## What it does

ya||a is a GPU-only C++/CUDA framework; `pbg-yalla` re-implements its
core pair-wise ABM design (Solution + force kernel + Euler stepping) in
vectorized NumPy so it runs on any platform, and exposes it as a
`process-bigraph` `Process` that can be wired into Composites.

The force kernels are direct ports of yalla's example models:

| Kernel | yalla example | Biology |
| --- | --- | --- |
| `spring` | `springs.cu` | N-body Hookean relaxation |
| `differential_adhesion` | `sorting.cu` | Steinberg cell sorting |
| `relu` | `passive_growth.cu`, `migration.cu` | ReLU-shaped adhesion used across growth models |
| `lennard_jones_soft` | — | Soft LJ potential for stable dense packings |

A proliferation (cell-division) kernel reproduces the dynamic-N behaviour
of `passive_growth.cu`.

## Installation

```bash
cd pbg-yalla
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Only dependencies are `process-bigraph`, `bigraph-schema`, `numpy`
(and `bigraph-viz` + `matplotlib` for the demo report).

## Quick Start

```python
from process_bigraph import allocate_core, Composite
from process_bigraph.emitter import RAMEmitter
from pbg_yalla import YallaProcess, make_yalla_document

core = allocate_core()
core.register_link('YallaProcess', YallaProcess)
core.register_link('ram-emitter', RAMEmitter)

doc = make_yalla_document(
    n_cells=200,
    force_kernel='differential_adhesion',
    n_types=2,
    dt=0.01,
    interval=0.5,
    r_min=0.5,
    r_max=1.0,
    wall_radius=2.5,
)

sim = Composite({'state': doc}, core=core)
sim.run(25.0)
print('final radial spread:', sim.state['stores']['type_radial_spread'])
```

To use the `Process` directly (useful when you want per-step snapshots of
positions for visualisation):

```python
proc = YallaProcess(config={
    'n_cells': 400, 'force_kernel': 'spring',
    'L_0': 0.5, 'dt': 0.01, 'seed': 1,
}, core=core)
proc.initial_state()
for _ in range(30):
    result = proc.update({}, interval=0.1)
    snap = proc.snapshot()  # positions, types, n_cells, time
```

## API Reference

### `YallaProcess`

A time-driven `Process` that owns positions `X[N, 3]` and cell types
`type[N]` internally. On every `update(interval)` call, advances
`interval / dt` Euler substeps of the configured pair-wise force kernel
plus (optionally) the proliferation kernel.

**Config**

| Key | Default | Description |
| --- | --- | --- |
| `n_cells` | 200 | Initial population |
| `init` | `'random_sphere'` | `'random_sphere'` or `'relaxed_sphere'` |
| `init_dist` | 0.5 | Target nearest-neighbour distance at init |
| `seed` | 0 | RNG seed |
| `n_types` | 1 | Number of agent types |
| `type_mode` | `'mixed'` | `'mixed'`, `'hemispheres'`, or `'inner_outer'` |
| `force_kernel` | `'spring'` | Key into `FORCE_KERNELS` |
| `L_0` | 0.5 | Spring equilibrium length (`spring`) |
| `r_min`, `r_max` | 0.5, 1.0 | Force range (`differential_adhesion`, LJ) |
| `r_eq_same_lo/hi`, `r_eq_diff_lo/hi` | 0.7/0.8/0.8/0.9 | ReLU equilibrium windows |
| `dt` | 0.05 | Euler sub-step |
| `damping` | 1.0 | Divides the force before integration |
| `proliferation_rate` | 0.0 | Per-cell per-unit-time division probability |
| `n_max` | 5000 | Cap on population |
| `wall_radius` | 0.0 | Soft spherical boundary (0 = off) |
| `wall_strength` | 5.0 | Wall force scale |

**Outputs (PBG ports)** — all scalar summaries:

| Port | Type | Meaning |
| --- | --- | --- |
| `n_cells` | integer | Current population |
| `gyration_radius` | float | sqrt of mean squared distance from centroid |
| `mean_neighbor_distance` | float | Average nearest-neighbour distance |
| `sorting_score` | float | Fraction of k=3 nearest neighbours sharing an agent's type |
| `type_radial_spread` | float | `mean(r\|type==1) - mean(r\|type==0)` — positive when type-1 is pushed outward |
| `center_x/y/z` | float | Centroid coordinates |

**Methods**

- `initial_state()` — builds the agent cloud, returns the summary dict
- `update(state, interval)` — advances, returns summary dict
- `snapshot()` — full agent state (`positions`, `types`, `n_cells`, `time`) for visualisation

### `make_yalla_document(**config) -> dict`

Returns a composite document wiring `YallaProcess` to a `RAMEmitter`.
All kwargs are forwarded to the process config. Accepts an `interval`
argument controlling how often the process updates and emits.

### `FORCE_KERNELS`

A dict mapping kernel name → function of signature
`force(positions, types, params) -> forces` where forces is the (N, 3)
net per-agent force. Each kernel is O(N²) per call (full pair-wise) but
vectorized through a single NumPy broadcast.

## Architecture

This wrapper follows v2ecoli's **bridge pattern**: a single `Process`
owns an internal NumPy state, pushes PBG inputs in at each `update()`,
runs the underlying integrator, and reads a summary back out through
PBG output ports. Full per-agent state is available via `snapshot()`
outside the PBG dataflow — convenient for visualisation without pushing
large arrays through the emitter.

```
     ┌──────────────────────────┐
     │      YallaProcess        │
     │  (pair-wise ABM core)    │
     │                          │
     │  positions X[N,3]        │
     │  types t[N]              │
     │  force_kernel(X,t,p) ───►┼── Euler substeps ────┐
     │                          │                      │
     └─┬────┬────┬────┬────┬────┘                     │
       │    │    │    │    │                           │
   n_cells rg mean_nn sort_score type_radial_spread    │
       │    │    │    │    │                           │
       ▼    ▼    ▼    ▼    ▼                           │
     ┌──────────────────────────┐                     │
     │       stores             │                     │
     └────────────┬─────────────┘                     │
                  │                                    │
                  ▼                                    │
            ┌───────────┐                              │
            │ RAMEmitter│                              │
            └───────────┘                              │
                                                       │
     snapshot() ◄──────────────────────────────────────┘
        │
        └─► demo/demo_report.py → Three.js viewers
```

## Demo

```bash
python demo/demo_report.py
```

Runs three configurations (spring relaxation, differential-adhesion
sorting, proliferation growth), generates
`demo/report.html` — a self-contained 2-3 MB interactive report with
Three.js agent viewers, Plotly time series, colored bigraph-viz
architecture diagrams, and a navigable JSON tree of each Composite
document — then opens it in Safari.

## Tests

```bash
pytest
```

Twenty tests cover process instantiation, force-kernel shapes, Newton's
3rd law, spring contraction, differential-adhesion radial segregation,
cell-division population growth with `n_max` enforcement, composite
assembly, and emitter round-trip.

## Relationship to real yalla

Real yalla is CUDA-only and optimised for GPU execution of millions of
agents. `pbg-yalla` is a faithful re-implementation of yalla's
*algorithmic* design — the same pair-wise force abstraction, the same
ReLU / differential-adhesion / spring kernels — in vectorized NumPy so
it can run on any platform and plug into `process-bigraph` composites.
For production-scale morphogenesis studies you still want real yalla on
a GPU; for integrative multi-process simulations where you want agent
dynamics as one component next to, say, a reaction-diffusion field or a
metabolic model, this wrapper is designed to drop in.

## License

MIT.
