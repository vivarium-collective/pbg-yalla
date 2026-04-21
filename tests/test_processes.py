"""Unit tests for YallaProcess and the force kernels."""

import numpy as np
import pytest
from process_bigraph import allocate_core

from pbg_yalla.processes import YallaProcess
from pbg_yalla.force_kernels import FORCE_KERNELS, spring, differential_adhesion, relu
from pbg_yalla.inits import random_sphere


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('YallaProcess', YallaProcess)
    return c


def test_instantiation(core):
    proc = YallaProcess(
        config={'n_cells': 20, 'force_kernel': 'spring', 'seed': 1},
        core=core)
    assert proc.config['n_cells'] == 20
    assert proc.config['force_kernel'] == 'spring'


def test_defaults(core):
    proc = YallaProcess(config={}, core=core)
    assert proc.config['n_cells'] == 200
    assert proc.config['force_kernel'] == 'spring'
    assert proc.config['dt'] == 0.05
    assert proc.config['n_types'] == 1


def test_initial_state(core):
    proc = YallaProcess(
        config={'n_cells': 50, 'force_kernel': 'spring', 'seed': 1},
        core=core)
    s = proc.initial_state()
    assert s['n_cells'] == 50
    assert s['gyration_radius'] > 0
    assert s['mean_neighbor_distance'] > 0
    assert s['sorting_score'] == pytest.approx(1.0)


def test_outputs_schema(core):
    proc = YallaProcess(config={}, core=core)
    outs = proc.outputs()
    expected = {
        'n_cells', 'gyration_radius', 'mean_neighbor_distance',
        'sorting_score', 'center_x', 'center_y', 'center_z',
    }
    assert expected.issubset(outs.keys())


def test_spring_contracts_cluster(core):
    proc = YallaProcess(
        config={'n_cells': 80, 'force_kernel': 'spring',
                'L_0': 0.4, 'dt': 0.02, 'seed': 1},
        core=core)
    s0 = proc.initial_state()
    r0 = s0['gyration_radius']
    s1 = proc.update({}, interval=2.0)
    r1 = s1['gyration_radius']
    # Spring force towards L_0 spacing pulls a sparse sphere inwards
    assert r1 < r0, f'Gyration radius should decrease: {r0} -> {r1}'


def test_sorting_produces_radial_segregation(core):
    # Differential adhesion should push the weakly-bound type to the outside
    # of the cluster. We measure the mean radial distance from the centroid
    # for each type and check the spread grows.
    import numpy as np
    proc = YallaProcess(
        config={
            'n_cells': 100,
            'force_kernel': 'differential_adhesion',
            'r_min': 0.5, 'r_max': 1.0,
            'dt': 0.01, 'n_types': 2, 'type_mode': 'mixed',
            'init': 'random_sphere', 'init_dist': 0.4,
            'wall_radius': 2.0, 'wall_strength': 15.0, 'damping': 2.0,
            'seed': 42,
        },
        core=core)
    proc.initial_state()
    snap0 = proc.snapshot()
    X0 = np.array(snap0['positions']); T0 = np.array(snap0['types'])
    r0 = np.linalg.norm(X0 - X0.mean(0), axis=1)
    spread0 = r0[T0 == 1].mean() - r0[T0 == 0].mean()

    proc.update({}, interval=20.0)
    snap1 = proc.snapshot()
    X1 = np.array(snap1['positions']); T1 = np.array(snap1['types'])
    r1 = np.linalg.norm(X1 - X1.mean(0), axis=1)
    spread1 = r1[T1 == 1].mean() - r1[T1 == 0].mean()

    # Spread grows: type-1 cells are pushed outward relative to type-0.
    assert spread1 > spread0 + 0.05, f'No sorting: {spread0} -> {spread1}'
    assert spread1 > 0.1, f'Radial spread too small: {spread1}'


def test_type_radial_spread_grows(core):
    # Differential adhesion should pull one type inward relative to the
    # other. We check the ``type_radial_spread`` output port directly.
    proc = YallaProcess(
        config={
            'n_cells': 100,
            'force_kernel': 'differential_adhesion',
            'r_min': 0.5, 'r_max': 1.0,
            'dt': 0.01, 'n_types': 2, 'type_mode': 'mixed',
            'init': 'random_sphere', 'init_dist': 0.4,
            'wall_radius': 2.0, 'wall_strength': 15.0, 'damping': 2.0,
            'seed': 42,
        },
        core=core)
    s0 = proc.initial_state()
    s1 = proc.update({}, interval=20.0)
    assert s1['type_radial_spread'] > s0['type_radial_spread'] + 0.05
    assert s1['type_radial_spread'] > 0.1


def test_proliferation_grows_population(core):
    proc = YallaProcess(
        config={
            'n_cells': 30, 'force_kernel': 'relu',
            'r_max': 1.0, 'dt': 0.1,
            'proliferation_rate': 0.1, 'n_max': 300,
            'seed': 3,
        },
        core=core)
    s0 = proc.initial_state()
    s1 = proc.update({}, interval=5.0)
    assert s1['n_cells'] > s0['n_cells']
    assert s1['n_cells'] <= 300


def test_n_max_respected(core):
    proc = YallaProcess(
        config={
            'n_cells': 40, 'force_kernel': 'relu',
            'r_max': 1.0, 'dt': 0.1,
            'proliferation_rate': 2.0, 'n_max': 60,
            'seed': 7,
        },
        core=core)
    proc.initial_state()
    s = proc.update({}, interval=10.0)
    assert s['n_cells'] <= 60


def test_unknown_kernel_raises(core):
    proc = YallaProcess(
        config={'n_cells': 10, 'force_kernel': 'does_not_exist'},
        core=core)
    with pytest.raises(ValueError, match='Unknown force_kernel'):
        proc.initial_state()


def test_snapshot(core):
    proc = YallaProcess(
        config={'n_cells': 30, 'force_kernel': 'spring', 'seed': 1},
        core=core)
    proc.initial_state()
    snap = proc.snapshot()
    assert snap['n_cells'] == 30
    assert len(snap['positions']) == 30
    assert len(snap['positions'][0]) == 3
    assert len(snap['types']) == 30


def test_wall_confines_cells(core):
    # Spring alone would expand a tight cluster; the wall should hold it in
    proc = YallaProcess(
        config={'n_cells': 60, 'force_kernel': 'spring',
                'L_0': 2.0, 'dt': 0.02, 'seed': 1,
                'wall_radius': 1.0, 'wall_strength': 20.0,
                'init_dist': 0.3},
        core=core)
    proc.initial_state()
    s = proc.update({}, interval=3.0)
    assert s['gyration_radius'] < 1.5  # confined by wall


def test_force_kernel_registry():
    assert 'spring' in FORCE_KERNELS
    assert 'differential_adhesion' in FORCE_KERNELS
    assert 'relu' in FORCE_KERNELS
    assert 'lennard_jones_soft' in FORCE_KERNELS


def test_force_kernel_shapes():
    X = random_sphere(12, 0.5, seed=1)
    t = np.zeros(12, dtype=np.int32)
    for name, fn in FORCE_KERNELS.items():
        F = fn(X, t, {'L_0': 0.5, 'r_min': 0.5, 'r_max': 1.0, 'eps': 1.0,
                      'r_eq_same_lo': 0.7, 'r_eq_same_hi': 0.8,
                      'r_eq_diff_lo': 0.8, 'r_eq_diff_hi': 0.9})
        assert F.shape == (12, 3), f'{name} wrong shape'
        assert np.isfinite(F).all(), f'{name} produced non-finite values'


def test_force_is_newton_third_law():
    # Newton's 3rd law: summing over all agents, net internal force ~ 0
    X = random_sphere(20, 0.5, seed=2)
    t = np.zeros(20, dtype=np.int32)
    F = spring(X, t, {'L_0': 0.5})
    assert np.allclose(F.sum(axis=0), 0.0, atol=1e-9)
