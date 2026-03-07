from __future__ import annotations

"""Linear shallow-water double-gyre example.

This example uses the current :mod:`finitevolx` Arakawa C-grid API to integrate a
wind-driven linear shallow-water model on a periodic beta-plane. The script uses
``xarray`` for coordinate-aware preprocessing and postprocessing, writes sampled
fields to a Zarr store, and saves a static before/after comparison figure.

The prognostic variables are the free-surface anomaly ``eta`` at T-points and the
velocity components ``u`` and ``v`` on the C-grid faces. The linearised equations
are

- ∂η/∂t = -H ∇·u⃗ + nu ∇²eta
- ∂u/∂t = -g ∂η/∂x + f v - r u + nu ∇²u + Fₓ
- ∂v/∂t = -g ∂η/∂y - f u - r v + nu ∇²v

where the zonal forcing ``Fₓ`` follows the classic double-gyre wind pattern.

Examples
--------
Run the default configuration and write outputs into ``outputs/``::

    uv run python scripts/swm_linear.py

Use a shorter run and custom paths while iterating on the example::

    uv run python scripts/swm_linear.py --steps 200 --output-dir /tmp/linear-swe
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from finitevolx import (
    ArakawaCGrid2D,
    Difference2D,
    Interpolation2D,
    Vorticity2D,
    enforce_periodic,
)

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class LinearShallowWaterConfig:
    """Configuration for the linear shallow-water double-gyre example.

    Parameters
    ----------
    nx : int, optional
        Number of interior cells in x.
    ny : int, optional
        Number of interior cells in y.
    Lx : float, optional
        Domain length in x [m].
    Ly : float, optional
        Domain length in y [m].
    gravity : float, optional
        Gravitational acceleration [m s⁻²].
    mean_depth : float, optional
        Reference fluid depth ``H`` [m].
    f0 : float, optional
        Reference Coriolis parameter [s⁻¹].
    beta : float, optional
        Meridional Coriolis gradient [m⁻¹ s⁻¹].
    drag : float, optional
        Linear Rayleigh drag coefficient [s⁻¹].
    viscosity : float, optional
        Laplacian viscosity/diffusivity [m² s⁻¹].
    wind_acceleration : float, optional
        Peak zonal body-force acceleration [m s⁻²].
    dt : float, optional
        Explicit time step [s].
    steps : int, optional
        Number of time steps to integrate.
    snapshot_interval : int, optional
        Number of steps between saved snapshots.
    zarr_path : Path, optional
        Output path for the sampled Zarr dataset.
    figure_path : Path, optional
        Output path for the comparison figure.

    Examples
    --------
    The defaults provide a stable, medium-resolution run::

        config = LinearShallowWaterConfig()

    A smaller configuration is useful in tests::

        config = LinearShallowWaterConfig(nx=24, ny=24, steps=120)
    """

    nx: int = 64
    ny: int = 64
    Lx: float = 5.12e6
    Ly: float = 5.12e6
    gravity: float = 9.81
    mean_depth: float = 500.0
    f0: float = 9.375e-5
    beta: float = 1.754e-11
    drag: float = 1.0e-4
    viscosity: float = 2.0e5
    wind_acceleration: float = 2.0e-7
    dt: float = 30.0
    steps: int = 1200
    snapshot_interval: int = 150
    zarr_path: Path = Path("outputs/linear_shallow_water_double_gyre.zarr")
    figure_path: Path = Path("outputs/linear_shallow_water_double_gyre.png")


def make_preprocessing_dataset(
    config: LinearShallowWaterConfig, grid: ArakawaCGrid2D
) -> xr.Dataset:
    """Build the coordinate-aware forcing and initial-condition fields.

    Parameters
    ----------
    config : LinearShallowWaterConfig
        Example configuration.
    grid : ArakawaCGrid2D
        C-grid associated with the simulation.

    Returns
    -------
    xr.Dataset
        Interior-grid ``xarray`` dataset with the initial free-surface anomaly,
        Coriolis parameter, and double-gyre wind forcing.

    Examples
    --------
    Generate preprocessing fields for the default setup::

        grid = ArakawaCGrid2D.from_interior(config.nx, config.ny, config.Lx, config.Ly)
        forcing = make_preprocessing_dataset(config, grid)

    Inspect the forcing pattern as an ``xarray`` object::

        forcing["wind_u"]
    """
    x = xr.DataArray(
        (np.arange(config.nx) + 0.5) * grid.dx,
        dims=("x",),
        name="x",
        attrs={"long_name": "cell_center_x", "units": "m"},
    )
    y = xr.DataArray(
        (np.arange(config.ny) + 0.5) * grid.dy,
        dims=("y",),
        name="y",
        attrs={"long_name": "cell_center_y", "units": "m"},
    )
    x2d, y2d = xr.broadcast(x, y)
    x2d = x2d.transpose("y", "x")
    y2d = y2d.transpose("y", "x")

    eta0 = 0.01 * np.sin(np.pi * x2d / config.Lx) * np.sin(np.pi * y2d / config.Ly)
    coriolis = config.f0 + config.beta * (y2d - 0.5 * config.Ly)
    wind_u = -config.wind_acceleration * np.cos(2.0 * np.pi * y2d / config.Ly)

    return xr.Dataset(
        data_vars={
            "eta0": eta0.rename("eta0"),
            "coriolis": coriolis.rename("coriolis"),
            "wind_u": wind_u.rename("wind_u"),
        },
        coords={"x": x, "y": y},
        attrs={
            "model": "linear shallow water",
            "configuration": "double gyre",
            "boundary_conditions": "periodic in x and y",
        },
    )


def to_periodic_field(field: xr.DataArray) -> Float[Array, "Ny Nx"]:
    """Pad an interior ``xarray`` field with a periodic ghost-cell ring.

    Parameters
    ----------
    field : xr.DataArray
        Interior field stored on ``(y, x)`` coordinates.

    Returns
    -------
    Float[Array, "Ny Nx"]
        JAX array with one periodic ghost-cell ring.

    Examples
    --------
    Convert an interior forcing field to the finitevolx storage layout::

        wind_u = to_periodic_field(forcing["wind_u"])

    The returned array includes the ghost cells used by the operators::

        wind_u.shape
    """
    interior = jnp.asarray(field.to_numpy())
    return enforce_periodic(jnp.pad(interior, pad_width=1, mode="wrap"))


def save_comparison_plot(
    dataset: xr.Dataset,
    figure_path: Path,
    variable_name: str,
    title: str,
    cmap: str = "RdBu_r",
    scale_factor: float = 1.0,
    colorbar_label: str | None = None,
) -> None:
    """Save a static before/after plot for a sampled field.

    Parameters
    ----------
    dataset : xr.Dataset
        Sampled simulation output.
    figure_path : Path
        Output path for the saved figure.
    variable_name : str
        Name of the field to visualise.
    title : str
        Figure title.
    cmap : str, optional
        Matplotlib colour map.

    Examples
    --------
    Save a comparison of the initial and final free-surface anomaly::

        save_comparison_plot(dataset, Path("linear.png"), "eta", "Linear SWE")

    Save a streamfunction comparison using a sequential colour map::

        save_comparison_plot(dataset, Path("psi.png"), "psi", "QG", cmap="viridis")
    """
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    data = dataset[variable_name] * scale_factor
    initial = data.isel(time=0)
    final = data.isel(time=-1)
    vmax = float(np.nanmax(np.abs(np.stack([initial.to_numpy(), final.to_numpy()]))))
    if vmax == 0.0:
        vmax = 1.0
    vmin = -vmax

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, field, panel in zip(
        axes, [initial, final], ["Initial", "Final"], strict=True
    ):
        image = axis.pcolormesh(
            dataset["x"],
            dataset["y"],
            field,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(panel)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        colorbar = fig.colorbar(image, ax=axis, shrink=0.9)
        if colorbar_label is not None:
            colorbar.set_label(colorbar_label)

    fig.suptitle(title)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def run_simulation(config: LinearShallowWaterConfig | None = None) -> xr.Dataset:  # noqa: PLR0915
    """Run the linear shallow-water double-gyre example.

    Parameters
    ----------
    config : LinearShallowWaterConfig | None, optional
        Simulation configuration. When omitted, the default stable setup is used.

    Returns
    -------
    xr.Dataset
        Sampled model output written to ``config.zarr_path`` and returned in-memory.

    Examples
    --------
    Execute the default experiment and save its artifacts::

        dataset = run_simulation()

    Run a compact experiment for testing or CI smoke checks::

        dataset = run_simulation(LinearShallowWaterConfig(nx=24, ny=24, steps=120))
    """
    config = config or LinearShallowWaterConfig()
    grid = ArakawaCGrid2D.from_interior(config.nx, config.ny, config.Lx, config.Ly)
    diff = Difference2D(grid=grid)
    interp = Interpolation2D(grid=grid)
    vort = Vorticity2D(grid=grid)
    forcing = make_preprocessing_dataset(config, grid)

    eta = to_periodic_field(forcing["eta0"])
    u = enforce_periodic(jnp.zeros_like(eta))
    v = enforce_periodic(jnp.zeros_like(eta))
    coriolis = to_periodic_field(forcing["coriolis"])
    wind_u = to_periodic_field(forcing["wind_u"])

    viscosity = config.viscosity
    gravity = config.gravity
    mean_depth = config.mean_depth
    drag = config.drag
    dt = config.dt

    def tendency(
        eta_field: Float[Array, "Ny Nx"],
        u_field: Float[Array, "Ny Nx"],
        v_field: Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Compute the linear shallow-water tendencies on the C-grid."""
        eta_rhs = -mean_depth * diff.divergence(u_field, v_field)
        eta_rhs = eta_rhs + viscosity * diff.laplacian(eta_field)

        u_on_t = interp.U_to_T(u_field)
        v_on_t = interp.V_to_T(v_field)
        v_on_u = interp.T_to_U(v_on_t)
        u_on_v = interp.T_to_V(u_on_t)
        coriolis_on_u = interp.T_to_U(coriolis)
        coriolis_on_v = interp.T_to_V(coriolis)

        u_rhs = -gravity * diff.diff_x_T_to_U(eta_field)
        u_rhs = u_rhs + coriolis_on_u * v_on_u
        u_rhs = u_rhs + wind_u - drag * u_field + viscosity * diff.laplacian(u_field)

        v_rhs = -gravity * diff.diff_y_T_to_V(eta_field)
        v_rhs = v_rhs - coriolis_on_v * u_on_v
        v_rhs = v_rhs - drag * v_field + viscosity * diff.laplacian(v_field)
        return eta_rhs, u_rhs, v_rhs

    @jax.jit
    def step(
        eta_field: Float[Array, "Ny Nx"],
        u_field: Float[Array, "Ny Nx"],
        v_field: Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Advance one Heun step and refill the periodic ghost cells."""
        k1_eta, k1_u, k1_v = tendency(eta_field, u_field, v_field)
        eta_stage = enforce_periodic(eta_field + dt * k1_eta)
        u_stage = enforce_periodic(u_field + dt * k1_u)
        v_stage = enforce_periodic(v_field + dt * k1_v)
        k2_eta, k2_u, k2_v = tendency(eta_stage, u_stage, v_stage)
        eta_next = enforce_periodic(eta_field + 0.5 * dt * (k1_eta + k2_eta))
        u_next = enforce_periodic(u_field + 0.5 * dt * (k1_u + k2_u))
        v_next = enforce_periodic(v_field + 0.5 * dt * (k1_v + k2_v))
        return eta_next, u_next, v_next

    snapshot_times: list[float] = []
    eta_snapshots: list[np.ndarray] = []
    u_snapshots: list[np.ndarray] = []
    v_snapshots: list[np.ndarray] = []
    speed_snapshots: list[np.ndarray] = []
    relative_vorticity_snapshots: list[np.ndarray] = []
    kinetic_energy: list[float] = []
    mass_anomaly: list[float] = []

    def record_snapshot(
        step_index: int,
        eta_field: Float[Array, "Ny Nx"],
        u_field: Float[Array, "Ny Nx"],
        v_field: Float[Array, "Ny Nx"],
    ) -> None:
        """Convert a sampled state into ``xarray``-ready NumPy arrays."""
        eta_np = np.asarray(jax.device_get(eta_field[1:-1, 1:-1]))
        u_center = interp.U_to_T(u_field)
        v_center = interp.V_to_T(v_field)
        zeta_corner = vort.relative_vorticity(u_field, v_field)
        zeta_center = interp.X_to_T(zeta_corner)
        u_np = np.asarray(jax.device_get(u_center[1:-1, 1:-1]))
        v_np = np.asarray(jax.device_get(v_center[1:-1, 1:-1]))
        zeta_np = np.asarray(jax.device_get(zeta_center[1:-1, 1:-1]))
        speed_np = np.sqrt(u_np**2 + v_np**2)

        snapshot_times.append(step_index * dt)
        eta_snapshots.append(eta_np)
        u_snapshots.append(u_np)
        v_snapshots.append(v_np)
        speed_snapshots.append(speed_np)
        relative_vorticity_snapshots.append(zeta_np)
        kinetic_energy.append(float(0.5 * np.mean(speed_np**2)))
        mass_anomaly.append(float(np.mean(eta_np)))

    record_snapshot(step_index=0, eta_field=eta, u_field=u, v_field=v)

    for iteration in range(1, config.steps + 1):
        eta, u, v = step(eta, u, v)
        if iteration % config.snapshot_interval == 0 or iteration == config.steps:
            record_snapshot(step_index=iteration, eta_field=eta, u_field=u, v_field=v)

    dataset = xr.Dataset(
        data_vars={
            "eta": (
                ("time", "y", "x"),
                np.stack(eta_snapshots, axis=0),
                {"long_name": "free_surface_anomaly", "units": "m"},
            ),
            "u": (
                ("time", "y", "x"),
                np.stack(u_snapshots, axis=0),
                {"long_name": "zonal_velocity_at_t_points", "units": "m s-1"},
            ),
            "v": (
                ("time", "y", "x"),
                np.stack(v_snapshots, axis=0),
                {"long_name": "meridional_velocity_at_t_points", "units": "m s-1"},
            ),
            "speed": (
                ("time", "y", "x"),
                np.stack(speed_snapshots, axis=0),
                {"long_name": "speed_at_t_points", "units": "m s-1"},
            ),
            "relative_vorticity": (
                ("time", "y", "x"),
                np.stack(relative_vorticity_snapshots, axis=0),
                {"long_name": "relative_vorticity_at_t_points", "units": "s-1"},
            ),
            "wind_u": (
                ("y", "x"),
                forcing["wind_u"].to_numpy(),
                {"long_name": "double_gyre_zonal_forcing", "units": "m s-2"},
            ),
            "kinetic_energy": (
                ("time",),
                np.asarray(kinetic_energy),
                {"long_name": "domain_mean_kinetic_energy", "units": "m2 s-2"},
            ),
            "mass_anomaly": (
                ("time",),
                np.asarray(mass_anomaly),
                {"long_name": "domain_mean_eta", "units": "m"},
            ),
        },
        coords={
            "time": xr.DataArray(
                np.asarray(snapshot_times),
                dims=("time",),
                attrs={"long_name": "time", "units": "s"},
            ),
            "x": forcing["x"],
            "y": forcing["y"],
        },
        attrs={
            "model": "linear shallow water",
            "configuration": "double gyre",
            "time_step_seconds": config.dt,
            "num_steps": config.steps,
            "notes": "Finitevolx C-grid example with xarray preprocessing/postprocessing.",
        },
    )

    config.zarr_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(config.zarr_path, mode="w", consolidated=False)
    save_comparison_plot(
        dataset=dataset,
        figure_path=config.figure_path,
        variable_name="eta",
        title="Linear shallow-water double gyre: free-surface anomaly",
    )
    return dataset


def parse_args() -> LinearShallowWaterConfig:
    """Parse the command line for the linear shallow-water example."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = LinearShallowWaterConfig()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory that will receive the Zarr store and comparison plot.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=defaults.steps,
        help="Number of explicit time steps to integrate.",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=defaults.snapshot_interval,
        help="Number of steps between sampled outputs.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        zarr_path = defaults.zarr_path
        figure_path = defaults.figure_path
    else:
        zarr_path = args.output_dir / "linear_shallow_water_double_gyre.zarr"
        figure_path = args.output_dir / "linear_shallow_water_double_gyre.png"

    return LinearShallowWaterConfig(
        steps=args.steps,
        snapshot_interval=args.snapshot_interval,
        zarr_path=zarr_path,
        figure_path=figure_path,
    )


def main() -> None:
    """Run the linear shallow-water example from the command line."""
    config = parse_args()
    dataset = run_simulation(config)
    print(f"Saved linear shallow-water fields to {config.zarr_path}")
    print(f"Saved linear shallow-water comparison plot to {config.figure_path}")
    print(
        f"Final max |eta| = {float(np.abs(dataset['eta'].isel(time=-1)).max()):.3e} m"
    )


if __name__ == "__main__":
    main()
