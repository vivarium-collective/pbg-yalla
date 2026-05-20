"""Tests for the real-ya||a bridge.

The GPU path (nvcc compile + run) can't execute on a CPU/CI host, so these
cover everything that doesn't need CUDA — source generation, the state-file
codec, the no-CUDA error contract, and summary parity with the reproduction —
and force the no-nvcc branch deterministically by monkeypatching
``find_nvcc``. A guarded smoke test actually compiles + runs only when nvcc is
present.
"""

import shutil

import numpy as np
import pytest
from process_bigraph import allocate_core

from pbg_yalla import yalla_native
from pbg_yalla.processes import YallaProcess
from pbg_yalla.reproduction import YallaReproductionProcess
from pbg_yalla.summary import SUMMARY_PORTS
from pbg_yalla.yalla_native import YallaCudaUnavailable

HAS_NVCC = shutil.which("nvcc") is not None


# --- source generation ----------------------------------------------------

def test_generate_cu_source_bakes_params():
    src = yalla_native.generate_cu_source(
        force_kernel="spring", L_0=0.5, r_min=0.4, r_max=1.2, n_max=128)
    assert "take_step<pbg_force>" in src
    assert 'Solution<float3, Tile_solver>' in src
    assert "0.5f" in src              # L_0 baked in
    assert "d_type[128]" in src       # n_max sizing
    assert "#include \"solvers.cuh\"" in src


def test_generate_cu_source_per_kernel():
    for kernel in yalla_native.SUPPORTED_KERNELS:
        src = yalla_native.generate_cu_source(
            force_kernel=kernel, L_0=0.5, r_min=0.4, r_max=1.0, n_max=64)
        assert "pbg_force" in src


def test_generate_cu_source_rejects_unknown_kernel():
    with pytest.raises(ValueError, match="no CUDA implementation"):
        yalla_native.generate_cu_source(
            force_kernel="lennard_jones_soft",
            L_0=0.5, r_min=0.4, r_max=1.0, n_max=64)


# --- state-file codec -----------------------------------------------------

def test_state_roundtrip(tmp_path):
    pos = np.array([[0.0, 1.0, 2.0], [-1.5, 0.25, 3.0]])
    typ = np.array([0, 1])
    path = tmp_path / "state.tsv"
    yalla_native.write_state(path, pos, typ)
    rpos, rtyp = yalla_native.read_state(path)
    assert np.allclose(rpos, pos)
    assert np.array_equal(rtyp, typ)
    assert rtyp.dtype == np.int32


def test_read_state_validates_count(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("3\n0 0 0 0\n1 1 1 1\n")  # claims 3, has 2
    with pytest.raises(ValueError, match="expected"):
        yalla_native.read_state(path)


# --- no-CUDA contract -----------------------------------------------------

def test_compile_raises_without_nvcc(monkeypatch):
    monkeypatch.setattr(yalla_native, "find_nvcc", lambda: None)
    src = yalla_native.generate_cu_source(
        force_kernel="spring", L_0=0.5, r_min=0.4, r_max=1.0, n_max=64)
    with pytest.raises(YallaCudaUnavailable, match="nvcc not found"):
        yalla_native.compile_model(src)


def test_resolve_yalla_home_no_clone(monkeypatch, tmp_path):
    # No env, no checkout, clone disallowed -> clear error.
    monkeypatch.delenv("YALLA_HOME", raising=False)
    monkeypatch.setattr(yalla_native.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("PBG_YALLA_CACHE", str(tmp_path / "cache"))
    with pytest.raises(YallaCudaUnavailable, match="no ya..a checkout"):
        yalla_native.resolve_yalla_home(allow_clone=False)


# --- YallaProcess (no GPU needed for construction / initial_state) --------

@pytest.fixture
def core():
    c = allocate_core()
    c.register_link("YallaProcess", YallaProcess)
    return c


def test_process_outputs_match_reproduction(core):
    proc = YallaProcess(config={}, core=core)
    assert proc.outputs() == dict(SUMMARY_PORTS)
    assert proc.outputs() == YallaReproductionProcess(config={}, core=core).outputs()


def test_initial_state_runs_without_gpu(core):
    proc = YallaProcess(
        config={"n_cells": 30, "force_kernel": "spring", "seed": 1}, core=core)
    s = proc.initial_state()
    assert s["n_cells"] == 30
    assert s["gyration_radius"] > 0


def test_initial_summary_parity_with_reproduction(core):
    # n_types=1: both build the same random_sphere(seed) and all-zero types,
    # so the initial summaries must be identical.
    cfg = {"n_cells": 40, "force_kernel": "spring", "seed": 7, "init_dist": 0.5}
    s_bridge = YallaProcess(config=cfg, core=core).initial_state()
    s_repro = YallaReproductionProcess(config=cfg, core=core).initial_state()
    for k in SUMMARY_PORTS:
        assert s_bridge[k] == pytest.approx(s_repro[k]), k


def test_update_raises_without_nvcc(core, monkeypatch):
    monkeypatch.setattr(yalla_native, "find_nvcc", lambda: None)
    proc = YallaProcess(
        config={"n_cells": 20, "force_kernel": "spring", "seed": 1}, core=core)
    proc.initial_state()
    with pytest.raises(YallaCudaUnavailable):
        proc.update({}, interval=0.01)


# --- guarded GPU smoke test ------------------------------------------------

@pytest.mark.skipif(not HAS_NVCC, reason="requires CUDA toolkit (nvcc) + GPU")
def test_real_yalla_step(core):
    proc = YallaProcess(
        config={"n_cells": 50, "force_kernel": "spring",
                "L_0": 0.4, "dt": 0.001, "seed": 1}, core=core)
    s0 = proc.initial_state()
    s1 = proc.update({}, interval=0.05)
    assert s1["n_cells"] == 50
    assert s1["gyration_radius"] != s0["gyration_radius"]
