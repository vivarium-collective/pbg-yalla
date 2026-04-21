"""Composite document factories for yalla-style simulations."""


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
