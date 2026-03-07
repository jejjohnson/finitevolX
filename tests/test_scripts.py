from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")
pytest.importorskip("matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def load_script_module(module_name: str, script_name: str):
    """Load a script as an importable module for smoke testing.

    Parameters
    ----------
    module_name : str
        Synthetic module name used for the import.
    script_name : str
        File name of the script inside ``scripts/``.

    Returns
    -------
    module
        Imported Python module corresponding to the requested script.
    """
    script_path = SCRIPTS_DIR / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load {script_path}"
        raise RuntimeError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_linear_shallow_water_script_runs_stably(tmp_path: Path) -> None:
    """The linear shallow-water example should save finite, bounded output."""
    module = load_script_module("swm_linear_script", "swm_linear.py")
    config = module.LinearShallowWaterConfig(
        nx=24,
        ny=24,
        steps=240,
        snapshot_interval=40,
        zarr_path=tmp_path / "linear.zarr",
        figure_path=tmp_path / "linear.png",
    )

    dataset = module.run_simulation(config)
    reopened = xr.open_zarr(config.zarr_path, consolidated=False)

    eta = reopened["eta"].to_numpy()
    speed = reopened["speed"].to_numpy()
    assert config.zarr_path.exists()
    assert config.figure_path.exists()
    assert dataset.sizes["time"] >= 3
    assert np.isfinite(eta).all()
    assert np.isfinite(speed).all()
    assert np.max(np.abs(eta)) < 0.1
    assert np.max(speed) < 0.01


def test_nonlinear_shallow_water_script_runs_stably(tmp_path: Path) -> None:
    """The nonlinear shallow-water example should keep the fluid depth positive."""
    module = load_script_module("shallow_water_script", "shallow_water.py")
    config = module.ShallowWaterConfig(
        nx=24,
        ny=24,
        steps=240,
        snapshot_interval=40,
        zarr_path=tmp_path / "nonlinear.zarr",
        figure_path=tmp_path / "nonlinear.png",
    )

    dataset = module.run_simulation(config)
    reopened = xr.open_zarr(config.zarr_path, consolidated=False)

    eta = reopened["eta"].to_numpy()
    speed = reopened["speed"].to_numpy()
    minimum_depth = reopened["minimum_depth"].to_numpy()
    assert config.zarr_path.exists()
    assert config.figure_path.exists()
    assert dataset.sizes["time"] >= 3
    assert np.isfinite(eta).all()
    assert np.isfinite(speed).all()
    assert np.isfinite(minimum_depth).all()
    assert np.min(minimum_depth) > 499.9
    assert np.max(speed) < 0.01


def test_qg_script_runs_stably(tmp_path: Path) -> None:
    """The 1.5-layer QG example should keep the PV field finite and bounded."""
    module = load_script_module("qg_script", "qg_1p5_layer.py")
    config = module.QuasiGeostrophicConfig(
        nx=24,
        ny=24,
        steps=400,
        snapshot_interval=50,
        zarr_path=tmp_path / "qg.zarr",
        figure_path=tmp_path / "qg.png",
    )

    dataset = module.run_simulation(config)
    reopened = xr.open_zarr(config.zarr_path, consolidated=False)

    q = reopened["q"].to_numpy()
    psi = reopened["psi"].to_numpy()
    assert config.zarr_path.exists()
    assert config.figure_path.exists()
    assert dataset.sizes["time"] >= 3
    assert np.isfinite(q).all()
    assert np.isfinite(psi).all()
    assert np.max(np.abs(q)) < 1.0e-5
    assert np.max(np.abs(psi)) < 2.0e4
