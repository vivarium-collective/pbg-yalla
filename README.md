# pbg-yalla

Process-bigraph wrapper for [ya||a](https://github.com/germannp/yalla), the
pair-wise agent-based model for morphogenesis (Germann et al., *Cell Systems*
2019). It wraps the **actual** ya||a GPU solver by default, and ships a NumPy
**reproduction** of the same model as the runs-anywhere fallback.

**[View Interactive Demo Report](https://vivarium-collective.github.io/pbg-yalla/)** — spring relaxation, differential-adhesion cell sorting, and proliferation growth with Three.js 3D agent viewers, Plotly charts, and bigraph architecture diagrams. (The demo runs the reproduction so it works without a GPU.)

## Two processes

| Class | Engine | Runs on | Use when |
| --- | --- | --- | --- |
| **`YallaProcess`** | the real ya||a CUDA solver | NVIDIA GPU + `nvcc` | you have a CUDA host and want the genuine simulator |
| **`YallaReproductionProcess`** | vectorized NumPy reproduction | any platform | no GPU, demos, CI, integrative composites |

Both expose the **same scalar output ports** (via `pbg_yalla.summary`), so a
composite can swap one for the other without rewiring.

### `YallaProcess` — the real ya||a

ya||a is header-only C++/CUDA: there's no library to link against — each model
is a `.cu` file compiled against ya||a's headers into a standalone GPU binary.
`YallaProcess` makes that a stateful process-bigraph step:

1. **Resolve** a ya||a checkout — `$YALLA_HOME` → `~/code/yalla` → a shallow
   `git clone` of the upstream repo into `~/.cache/pbg-yalla/`.
2. **Generate** a parametrized `.cu` whose `main()` reads the initial
   population from a file, runs `round(interval/dt)` calls of
   `bodies.take_step<force>(dt)`, and writes the final state back.
3. **Compile** it once with `nvcc` (binary content-addressed and cached) and
   **run** it each step, round-tripping agent state through TSV files.

On a host without `nvcc`/GPU it raises `YallaCudaUnavailable` with guidance —
it never silently substitutes a reproduction. Construction and
`initial_state()` (the initial sphere is built in NumPy) work everywhere; only
stepping needs CUDA. Native kernels: `spring`, `differential_adhesion`, `relu`.

### `YallaReproductionProcess` — NumPy reproduction

A clean-room re-implementation of ya||a's design — the same pair-wise force
abstraction and Euler sub-stepping — in vectorized NumPy. Adds proliferation,
a confining wall, and a chemotaxis term not present in the stock ya||a
examples. Known divergences from upstream: float64 instead of GPU single
precision; a dense O(N²) pair sum instead of ya||a's tile/grid neighbour
solver; only a subset of kernels ported (see `pbg_yalla/force_kernels.py`).

## Getting a CUDA GPU for `YallaProcess`

ya||a needs an NVIDIA GPU, the CUDA toolkit (`nvcc`), and a C++14 compiler —
there is no CPU fallback and nothing for Apple Silicon. Easiest options:

- **Google Colab (free)** — GPU runtimes ship `nvcc` + a T4. Clone ya||a,
  compile an example, run.
- **Rental GPU hosts** — Lambda Labs, vast.ai, RunPod, Paperspace; or AWS
  `g4dn`/`p3`, GCP, Azure NC.
- **A local NVIDIA box or HPC cluster.**

Point the bridge at a checkout with `export YALLA_HOME=/path/to/yalla` and set
`cuda_arch` to match your GPU (e.g. `sm_70`, `sm_80`).

## Installation

```bash
cd pbg-yalla
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Dependencies: `process-bigraph`, `bigraph-schema`, `pbg-superpowers`, `numpy`
(plus `bigraph-viz` + `matplotlib` for the demo report). The CUDA toolkit is an
external requirement for `YallaProcess` only.

## Quick Start

Runs-anywhere (reproduction):

```python
from process_bigraph import Composite
from pbg_yalla.composites import register_yalla, make_yalla_document

core = register_yalla()          # registers both processes + emitter + viz
doc = make_yalla_document(
    n_cells=200, force_kernel='differential_adhesion',
    n_types=2, dt=0.01, interval=0.5, wall_radius=2.5,
)
sim = Composite({'state': doc}, core=core)
sim.run(25.0)
print('type radial spread:', sim.state['stores']['type_radial_spread'])
```

Real ya||a (on a CUDA host):

```python
from pbg_yalla.composites import register_yalla, build_composite

core = register_yalla()
sim = build_composite('yalla-real-springs',
                      overrides={'n_cells': 200, 'cuda_arch': 'sm_80'})
sim.run(0.5)   # compiles a .cu with nvcc, then drives the real solver
```

## Composites

Discoverable `*.composite.yaml` specs (dashboard Composites tab):

| Spec | Engine |
| --- | --- |
| `yalla-springs-demo` | reproduction (runs anywhere) |
| `yalla-springs-with-viz` | reproduction + wired Visualization step |
| `yalla-real-springs` | **real ya||a** (requires CUDA) |

## Config reference

`YallaReproductionProcess` and `YallaProcess` share a config surface. The
reproduction honours every key below; the bridge uses the subset the native
solver consumes (`n_cells`, `init*`, `seed`, `n_types`, `type_mode`,
`force_kernel`, `L_0`, `r_min`, `r_max`, `dt`, `n_max`, plus `cuda_arch`).

| Key | Default | Description |
| --- | --- | --- |
| `n_cells` | 200 | Initial population |
| `init` | `'random_sphere'` | `'random_sphere'` or `'relaxed_sphere'` |
| `init_dist` | 0.5 | Target nearest-neighbour distance at init |
| `seed` | 0 | RNG seed |
| `n_types` | 1 | Number of agent types |
| `type_mode` | `'mixed'` | `'mixed'`, `'hemispheres'`, or `'inner_outer'` |
| `force_kernel` | `'spring'` | Force kernel (bridge: `spring`/`differential_adhesion`/`relu`) |
| `L_0` | 0.5 | Spring equilibrium length |
| `r_min`, `r_max` | 0.5, 1.0 | Force range |
| `dt` | 0.05 | Integrator sub-step |
| `damping` | 1.0 | Divides force before integration *(reproduction only)* |
| `proliferation_rate` | 0.0 | Per-cell per-time division probability *(reproduction only)* |
| `n_max` | 5000 | Population cap |
| `wall_radius` | 0.0 | Soft spherical boundary, 0 = off *(reproduction only)* |
| `chemotaxis_strength` | 0.0 | Morphogen-gradient pull *(reproduction only)* |
| `cuda_arch` | `'sm_60'` | nvcc target architecture *(`YallaProcess` only)* |

**Output ports** (both processes) — all scalar summaries: `n_cells`,
`gyration_radius`, `mean_neighbor_distance`, `sorting_score`,
`type_radial_spread`, `center_x/y/z`. Full per-agent state is available off the
PBG dataflow via `snapshot()`.

## Architecture

```
   YallaProcess (real)                 YallaReproductionProcess (NumPy)
   ┌────────────────────┐              ┌────────────────────────┐
   │ generate .cu       │              │ positions X[N,3]        │
   │ nvcc compile+cache │              │ types t[N]              │
   │ run binary / step  │              │ force_kernel(X,t,p)     │
   │ parse state files  │              │ Euler substeps          │
   └─────────┬──────────┘              └───────────┬─────────────┘
             └──────────► same summary ports ◄─────┘
                  n_cells, gyration_radius, …
                          │
                          ▼
                       stores ──► RAMEmitter
```

## Demo & Tests

```bash
python demo/demo_report.py     # reproduction; writes demo/report.html
pytest                          # bridge GPU test auto-skips without nvcc
```

## License

MIT.
