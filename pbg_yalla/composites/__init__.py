"""yalla composite documents + composite-spec discovery.

Two flavors of composite construction live in this package:

1. **Hand-coded factory** — :func:`make_yalla_document` builds a PBG
   state-dict programmatically. This is the legacy entry point used by
   :mod:`demo.demo_report` and the existing test suite.

2. **Declarative ``*.composite.yaml``** — sibling files in this
   directory follow the pbg-superpowers composite-spec convention.
   :func:`build_composite` loads one by name and instantiates
   :class:`process_bigraph.Composite` with parameter substitution. The
   dashboard's composite explorer discovers these automatically once
   the package is installed in a workspace.

Both flavors are equivalent — pick the one that fits your use case.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import yaml
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_yalla.processes import YallaProcess


# ---------------------------------------------------------------------------
# Hand-coded composite factory (legacy / programmatic API)
# ---------------------------------------------------------------------------


def register_yalla(core=None):
    """Return a core with YallaProcess, the RAM emitter, and the yalla
    Visualization registered."""
    if core is None:
        core = allocate_core()
    core.register_link('YallaProcess', YallaProcess)
    # Two aliases — the legacy hand-coded factory wires
    # ``local:ram-emitter`` while the composite-spec wiring uses the
    # canonical ``local:RAMEmitter`` name.
    core.register_link('ram-emitter', RAMEmitter)
    core.register_link('RAMEmitter', RAMEmitter)
    # Register Visualization Step so composites can wire it by name.
    from pbg_yalla.visualizations import YallaSummaryPlots
    core.register_link('YallaSummaryPlots', YallaSummaryPlots)
    return core


def make_yalla_document(
    n_cells=200,
    force_kernel='spring',
    dt=0.05,
    interval=1.0,
    seed=0,
    n_types=1,
    type_mode='mixed',
    proliferation_rate=0.0,
    init='random_sphere',
    init_dist=0.5,
    L_0=0.5,
    r_min=0.5,
    r_max=1.0,
    wall_radius=0.0,
    wall_strength=5.0,
    **extra_config,
):
    """Build a composite document wiring a ``YallaProcess`` to an emitter.

    The process exposes scalar summary ports (n_cells, gyration_radius,
    mean_neighbor_distance, sorting_score, center_{x,y,z}) which are
    collected by a RAMEmitter.
    """
    cfg = {
        'n_cells': n_cells,
        'force_kernel': force_kernel,
        'dt': dt,
        'seed': seed,
        'n_types': n_types,
        'type_mode': type_mode,
        'proliferation_rate': proliferation_rate,
        'init': init,
        'init_dist': init_dist,
        'L_0': L_0,
        'r_min': r_min,
        'r_max': r_max,
        'wall_radius': wall_radius,
        'wall_strength': wall_strength,
    }
    cfg.update(extra_config)

    return {
        'yalla': {
            '_type': 'process',
            'address': 'local:YallaProcess',
            'config': cfg,
            'interval': interval,
            'inputs': {},
            'outputs': {
                'n_cells': ['stores', 'n_cells'],
                'gyration_radius': ['stores', 'gyration_radius'],
                'mean_neighbor_distance': ['stores', 'mean_neighbor_distance'],
                'sorting_score': ['stores', 'sorting_score'],
                'type_radial_spread': ['stores', 'type_radial_spread'],
                'center_x': ['stores', 'center_x'],
                'center_y': ['stores', 'center_y'],
                'center_z': ['stores', 'center_z'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'n_cells': 'integer',
                    'gyration_radius': 'float',
                    'mean_neighbor_distance': 'float',
                    'sorting_score': 'float',
                    'type_radial_spread': 'float',
                    'time': 'float',
                },
            },
            'inputs': {
                'n_cells': ['stores', 'n_cells'],
                'gyration_radius': ['stores', 'gyration_radius'],
                'mean_neighbor_distance': ['stores', 'mean_neighbor_distance'],
                'sorting_score': ['stores', 'sorting_score'],
                'type_radial_spread': ['stores', 'type_radial_spread'],
                'time': ['global_time'],
            },
        },
    }


# ---------------------------------------------------------------------------
# Declarative composite-spec loader (*.composite.yaml)
# ---------------------------------------------------------------------------

_COMPOSITES_DIR = Path(__file__).parent

_FULL_PLACEHOLDER = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_INLINE_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _cast(value: Any, declared_type: str | None) -> Any:
    if declared_type is None:
        return value
    if declared_type == "float":
        return float(value)
    if declared_type == "int":
        return int(value)
    if declared_type in ("string", "str"):
        return str(value)
    if declared_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _substitute(state: Any, params: dict, overrides: dict) -> Any:
    if isinstance(state, dict):
        return {k: _substitute(v, params, overrides) for k, v in state.items()}
    if isinstance(state, list):
        return [_substitute(v, params, overrides) for v in state]
    if isinstance(state, str):
        m = _FULL_PLACEHOLDER.match(state)
        if m:
            pname = m.group(1)
            pdef = params.get(pname, {})
            raw = overrides.get(pname, pdef.get("default"))
            return _cast(raw, pdef.get("type"))
        if _INLINE_PLACEHOLDER.search(state):
            return _INLINE_PLACEHOLDER.sub(
                lambda mm: str(overrides.get(mm.group(1), params.get(mm.group(1), {}).get("default", ""))),
                state,
            )
    return state


def list_composite_specs() -> list[str]:
    """Return short names of every `*.composite.yaml` shipped in this package."""
    out: list[str] = []
    for path in sorted(_COMPOSITES_DIR.glob("*.composite.yaml")):
        out.append(path.name[: -len(".composite.yaml")])
    return out


def load_composite_spec(name: str) -> dict:
    """Load and parse a named composite spec. `name` is the stem (no suffix)."""
    path = _COMPOSITES_DIR / f"{name}.composite.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"composite spec not found: {path}")
    return yaml.safe_load(path.read_text())


def build_composite(name: str, *, overrides: dict | None = None, core=None):
    """Load a *.composite.yaml by name and instantiate process_bigraph.Composite.

    overrides: parameter overrides (keys must match spec.parameters)
    core:      optional pre-built core; otherwise register_yalla() is used
    """
    from process_bigraph import Composite

    spec = load_composite_spec(name)
    if not isinstance(spec, dict) or "state" not in spec or "name" not in spec:
        raise ValueError(f"composite '{name}' missing required keys (name, state)")

    if core is None:
        core = register_yalla()

    params = spec.get("parameters") or {}
    state = _substitute(spec.get("state") or {}, params, overrides or {})
    return Composite({"state": state}, core=core)
