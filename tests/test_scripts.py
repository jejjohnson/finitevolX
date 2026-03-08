from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import ArakawaCGrid2D, Difference2D, Interpolation2D

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


def checkerboard_metric(field: np.ndarray) -> float:
    """Return the normalized amplitude of the 2-D Nyquist checkerboard mode.

    Parameters
    ----------
    field : np.ndarray
        Two-dimensional sampled field to analyse.

    Returns
    -------
    float
        Normalized checkerboard amplitude. Values near 0 indicate little
        odd-even structure, while values closer to 1 indicate a strong
        checkerboard mode.
    """
    y_index, x_index = np.indices(field.shape)
    checker = (-1.0) ** (x_index + y_index)
    rms = np.sqrt(np.mean(field**2))
    return float(np.abs(np.mean(field * checker)) / (rms + 1.0e-30))


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
    relative_vorticity = reopened["relative_vorticity"].to_numpy()
    assert config.zarr_path.exists()
    assert config.figure_path.exists()
    assert dataset.sizes["time"] >= 3
    assert np.isfinite(eta).all()
    assert np.isfinite(speed).all()
    assert np.isfinite(relative_vorticity).all()
    assert np.max(np.abs(eta)) < 0.1
    assert np.max(speed) < 0.01
    assert checkerboard_metric(eta[-1]) < 1.0e-2
    assert checkerboard_metric(relative_vorticity[-1]) < 1.0e-2


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
    relative_vorticity = reopened["relative_vorticity"].to_numpy()
    assert config.zarr_path.exists()
    assert config.figure_path.exists()
    assert dataset.sizes["time"] >= 3
    assert np.isfinite(eta).all()
    assert np.isfinite(speed).all()
    assert np.isfinite(minimum_depth).all()
    assert np.isfinite(relative_vorticity).all()
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
    relative_vorticity = reopened["relative_vorticity"].to_numpy()
    assert config.zarr_path.exists()
    assert config.figure_path.exists()
    assert dataset.sizes["time"] >= 3
    assert np.isfinite(q).all()
    assert np.isfinite(psi).all()
    assert np.isfinite(relative_vorticity).all()
    assert np.max(np.abs(q)) < 2.0e-3
    assert np.max(np.abs(psi)) < 1.0e6
    assert checkerboard_metric(relative_vorticity[-1]) < 1.0e-2
    # The QG example uses a deterministic sinusoidal initial condition and
    # steady double-gyre forcing, so the relative-vorticity variability should
    # increase during the short smoke-test integration.
    assert np.std(relative_vorticity[-1]) > np.std(relative_vorticity[0])
    # The example must report closed-basin (not periodic) boundary conditions.
    notes = dataset.attrs.get("notes", "")
    assert "closed" in notes.lower(), (
        f"Expected 'closed' in dataset notes, got: {notes!r}"
    )
    assert "periodic" not in notes.lower(), (
        f"Found 'periodic' in dataset notes: {notes!r}"
    )


def test_qg_geostrophic_velocity_helper_is_nearly_nondivergent() -> None:
    """The QG streamfunction-to-velocity mapping should preserve incompressibility."""
    module = load_script_module("qg_module", "qg_1p5_layer.py")
    config = module.QuasiGeostrophicConfig(nx=16, ny=16)
    grid = ArakawaCGrid2D.from_interior(config.nx, config.ny, config.Lx, config.Ly)
    diff = Difference2D(grid=grid)
    interp = Interpolation2D(grid=grid)

    x = (np.arange(config.nx) + 0.5) * grid.dx
    y = (np.arange(config.ny) + 0.5) * grid.dy
    x2d, y2d = np.meshgrid(x, y)
    psi_interior = np.sin(2.0 * np.pi * x2d / config.Lx) * np.sin(
        2.0 * np.pi * y2d / config.Ly
    )
    # Use zero (Dirichlet wall) ghost cells, matching the closed-basin setup.
    psi_field = jnp.pad(jnp.asarray(psi_interior), pad_width=1, mode="constant")

    u_field, v_field = module.geostrophic_velocity_from_streamfunction(
        psi_field, diff, interp
    )
    divergence = diff.divergence(u_field, v_field)

    np.testing.assert_allclose(divergence[1:-1, 1:-1], 0.0, atol=1e-12)


def test_qg_wall_field_produces_zero_ghost_cells() -> None:
    """Verify that to_wall_field produces zero ghost cells (closed-basin Dirichlet BCs)."""
    module = load_script_module("qg_wall_module", "qg_1p5_layer.py")
    interior = np.ones((16, 16))
    da = xr.DataArray(interior, dims=("y", "x"))
    padded = np.asarray(module.to_wall_field(da))

    # All four ghost rows/columns must be exactly zero.
    np.testing.assert_array_equal(
        padded[0, :], 0.0, err_msg="south ghost row is not zero"
    )
    np.testing.assert_array_equal(
        padded[-1, :], 0.0, err_msg="north ghost row is not zero"
    )
    np.testing.assert_array_equal(
        padded[:, 0], 0.0, err_msg="west ghost col is not zero"
    )
    np.testing.assert_array_equal(
        padded[:, -1], 0.0, err_msg="east ghost col is not zero"
    )
    # Interior values must be preserved unchanged.
    np.testing.assert_array_equal(padded[1:-1, 1:-1], interior)
