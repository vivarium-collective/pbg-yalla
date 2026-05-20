"""Integration tests for yalla composites."""

import pytest
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter

from pbg_yalla.reproduction import YallaReproductionProcess
from pbg_yalla.composites import make_yalla_document


@pytest.fixture
def core():
    # make_yalla_document wires the runs-anywhere reproduction.
    c = allocate_core()
    c.register_link('YallaReproductionProcess', YallaReproductionProcess)
    c.register_link('ram-emitter', RAMEmitter)
    return c


def test_composite_assembly(core):
    doc = make_yalla_document(n_cells=20, force_kernel='spring', interval=0.5)
    sim = Composite({'state': doc}, core=core)
    assert sim is not None


def test_composite_short_run(core):
    doc = make_yalla_document(
        n_cells=25, force_kernel='spring', dt=0.02, interval=0.5, seed=1)
    sim = Composite({'state': doc}, core=core)
    sim.run(2.0)
    stores = sim.state['stores']
    assert stores['n_cells'] == 25
    assert stores['gyration_radius'] > 0


def test_emitter_collects_timeseries(core):
    doc = make_yalla_document(
        n_cells=25, force_kernel='spring', dt=0.02, interval=0.5, seed=1)
    sim = Composite({'state': doc}, core=core)
    sim.run(3.0)
    raw = gather_emitter_results(sim)
    entries = raw[('emitter',)]
    assert len(entries) >= 4
    assert 'gyration_radius' in entries[0]
    assert 'time' in entries[0]
    last_time = entries[-1]['time']
    assert last_time == pytest.approx(3.0)


def test_sorting_composite(core):
    doc = make_yalla_document(
        n_cells=40, force_kernel='differential_adhesion',
        r_min=0.5, r_max=1.0, dt=0.02,
        interval=0.5, n_types=2, type_mode='mixed', seed=3)
    sim = Composite({'state': doc}, core=core)
    sim.run(5.0)
    raw = gather_emitter_results(sim)
    entries = raw[('emitter',)]
    # Sorting score should trend upward on average
    first = entries[1]['sorting_score']
    last = entries[-1]['sorting_score']
    assert last > first - 0.05  # at worst, stable


def test_growth_composite(core):
    doc = make_yalla_document(
        n_cells=20, force_kernel='relu', r_max=1.0, dt=0.1,
        interval=1.0, proliferation_rate=0.1, n_types=1, seed=5,
        n_max=200,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(5.0)
    raw = gather_emitter_results(sim)
    entries = raw[('emitter',)]
    assert entries[-1]['n_cells'] > entries[0]['n_cells']
