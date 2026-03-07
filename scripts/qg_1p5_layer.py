from __future__ import annotations

"""Wind-driven 1.5-layer quasi-geostrophic double-gyre example.

This example shows how to combine the current :mod:`finitevolx` advection,
interpolation, and spectral elliptic solvers to build a compact 1.5-layer QG
model. The potential-vorticity field is advected on a periodic beta-plane, the
streamfunction is recovered from a Helmholtz inversion, and the output is saved
through ``xarray``/Zarr together with a static before/after figure.

The model evolves the QG potential-vorticity anomaly ``q`` according to

- ∂q/∂t = -u⃗·∇q + F - r q + nu ∇²q
- (∇² - 1 / Ld²) ψ = q - β (y - Ly / 2)
- u = -∂ψ/∂y,  v = ∂ψ/∂x

where ``F`` is a prescribed double-gyre wind-curl forcing.

Examples
--------
Run the default experiment and save the sampled fields::

    uv run python scripts/qg_1p5_layer.py

Run a shorter debugging case in a temporary directory::

    uv run python scripts/qg_1p5_layer.py --steps 400 --output-dir /tmp/qg-double-gyre
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
from jax import Array
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from finitevolx import (
    Advection2D,
    ArakawaCGrid2D,
    Difference2D,
    Interpolation2D,
    Vorticity2D,
    enforce_periodic,
    solve_helmholtz_fft,
)

jax.config.update("jax_enable_x64", True)


def to_periodic_field(field: xr.DataArray) -> Array:
    """Pad an interior ``xarray`` field with a periodic ghost-cell ring."""
    interior = jnp.asarray(field.to_numpy())
    return enforce_periodic(jnp.pad(interior, pad_width=1, mode="wrap"))


def geostrophic_velocity_from_streamfunction(
    psi_field: Array,
    diff: Difference2D,
    interp: Interpolation2D,
) -> tuple[Array, Array]:
    """Map a T-point streamfunction to face-centred geostrophic velocities.

    The streamfunction is first averaged to X-points so that the orthogonal
    derivatives land directly on the U and V faces:

    - u[j, i+1/2] = -(ψ[j+1/2, i+1/2] - ψ[j-1/2, i+1/2]) / dy
    - v[j+1/2, i] =  (ψ[j+1/2, i+1/2] - ψ[j+1/2, i-1/2]) / dx

    Parameters
    ----------
    psi_field : Array
        Streamfunction stored at T-points with one ghost-cell ring.
    diff : Difference2D
        Difference operators for the active grid.
    interp : Interpolation2D
        Interpolation operators for the active grid.

    Returns
    -------
    tuple[Array, Array]
        The zonal and meridional velocities on U and V points.
    """
    psi_on_x = interp.T_to_X(psi_field)
    u_field = -diff.diff_y_X_to_U(psi_on_x)
    v_field = diff.diff_x_X_to_V(psi_on_x)
    return u_field, v_field


def save_comparison_plot(
    dataset: xr.Dataset,
    figure_path: Path,
    variable_name: str,
    title: str,
    cmap: str = "RdBu_r",
    scale_factor: float = 1.0,
    colorbar_label: str | None = None,
) -> None:
    """Save a static before/after plot for a sampled field."""
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


@dataclass(frozen=True)
class QuasiGeostrophicConfig:
    """Configuration for the 1.5-layer QG double-gyre example.

    Parameters
    ----------
    nx, ny : int, optional
        Number of interior cells in x and y.
    Lx, Ly : float, optional
        Domain lengths [m].
    f0 : float, optional
        Reference Coriolis parameter [s⁻¹].
    beta : float, optional
        Meridional Coriolis gradient [m⁻¹ s⁻¹].
    rossby_radius : float, optional
        First baroclinic Rossby radius of deformation [m].
    drag : float, optional
        Linear PV damping coefficient [s⁻¹].
    viscosity : float, optional
        Laplacian viscosity [m² s⁻¹].
    wind_curl_forcing : float, optional
        Peak PV forcing amplitude [s⁻²].
    dt : float, optional
        Explicit time step [s].
    steps : int, optional
        Number of time steps.
    snapshot_interval : int, optional
        Steps between sampled outputs.
    zarr_path, figure_path : Path, optional
        Artifact paths written by the script.

    Examples
    --------
    Use the default stable setup::

        config = QuasiGeostrophicConfig()

    Use a smaller test case for CI smoke checks::

        config = QuasiGeostrophicConfig(nx=24, ny=24, steps=300)
    """

    nx: int = 64
    ny: int = 64
    Lx: float = 5.12e6
    Ly: float = 5.12e6
    f0: float = 9.375e-5
    beta: float = 1.754e-11
    rossby_radius: float = 4.0e4
    drag: float = 5.0e-8
    viscosity: float = 5.0e4
    wind_curl_forcing: float = 2.0e-10
    dt: float = 4000.0
    steps: int = 4000
    snapshot_interval: int = 500
    zarr_path: Path = Path("outputs/qg_1p5_layer_double_gyre.zarr")
    figure_path: Path = Path("outputs/qg_1p5_layer_double_gyre.png")


def make_preprocessing_dataset(
    config: QuasiGeostrophicConfig, grid: ArakawaCGrid2D
) -> xr.Dataset:
    """Build the coordinate-aware fields for the QG example.

    Parameters
    ----------
    config : QuasiGeostrophicConfig
        Example configuration.
    grid : ArakawaCGrid2D
        Underlying Arakawa C-grid.

    Returns
    -------
    xr.Dataset
        Interior-grid ``xarray`` dataset with the initial PV anomaly, beta-plane
        term, and double-gyre wind-curl forcing.

    Examples
    --------
    Prepare the forcing and inversion data for a QG run::

        forcing = make_preprocessing_dataset(config, grid)

    Inspect the wind-curl forcing with xarray-aware plotting::

        forcing["wind_curl"].plot()
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

    q0 = (
        5.0e-9 * np.sin(2.0 * np.pi * x2d / config.Lx) * np.sin(np.pi * y2d / config.Ly)
    )
    beta_term = config.beta * (y2d - 0.5 * config.Ly)
    # Match the sign convention used in MQGeometry's double_gyre.py:
    # curl_tau(y) = -tau0 * (2π / Ly) * sin(2π y / Ly)
    wind_curl = -config.wind_curl_forcing * np.sin(2.0 * np.pi * y2d / config.Ly)

    return xr.Dataset(
        data_vars={
            "q0": q0.rename("q0"),
            "beta_term": beta_term.rename("beta_term"),
            "wind_curl": wind_curl.rename("wind_curl"),
        },
        coords={"x": x, "y": y},
        attrs={
            "model": "1.5-layer quasi-geostrophic",
            "configuration": "double gyre",
            "boundary_conditions": "periodic in x and y",
        },
    )


def run_simulation(config: QuasiGeostrophicConfig | None = None) -> xr.Dataset:  # noqa: PLR0915
    """Run the 1.5-layer QG double-gyre example.

    Parameters
    ----------
    config : QuasiGeostrophicConfig | None, optional
        Simulation configuration. When omitted, the default stable setup is used.

    Returns
    -------
    xr.Dataset
        Sampled output dataset written to ``config.zarr_path`` and returned.

    Examples
    --------
    Execute the default QG experiment::

        dataset = run_simulation()

    Run a compact test-sized integration::

        dataset = run_simulation(QuasiGeostrophicConfig(nx=24, ny=24, steps=300))
    """
    config = config or QuasiGeostrophicConfig()
    grid = ArakawaCGrid2D.from_interior(config.nx, config.ny, config.Lx, config.Ly)
    diff = Difference2D(grid=grid)
    interp = Interpolation2D(grid=grid)
    adv = Advection2D(grid=grid)
    vort = Vorticity2D(grid=grid)
    forcing = make_preprocessing_dataset(config, grid)

    q = to_periodic_field(forcing["q0"])
    beta_term = jnp.asarray(forcing["beta_term"].to_numpy())
    wind_curl = to_periodic_field(forcing["wind_curl"])

    viscosity = config.viscosity
    drag = config.drag
    dt = config.dt
    deformation_wavenumber = 1.0 / config.rossby_radius**2

    def invert_streamfunction(q_field: Array) -> Array:
        """Recover the streamfunction from the PV anomaly using Helmholtz FFT."""
        rhs = q_field[1:-1, 1:-1] - beta_term
        psi_interior = solve_helmholtz_fft(
            rhs,
            grid.dx,
            grid.dy,
            lambda_=deformation_wavenumber,
        )
        psi_field = jnp.zeros_like(q_field)
        psi_field = psi_field.at[1:-1, 1:-1].set(psi_interior)
        return enforce_periodic(psi_field)

    def tendency(q_field: Array) -> tuple[Array, Array, Array, Array]:
        """Compute the QG PV tendency and the diagnosed balanced state."""
        psi_field = invert_streamfunction(q_field)
        u_field, v_field = geostrophic_velocity_from_streamfunction(
            psi_field, diff, interp
        )
        q_rhs = adv(q_field, u_field, v_field, method="upwind1")
        q_rhs = q_rhs + wind_curl - drag * q_field + viscosity * diff.laplacian(q_field)
        return q_rhs, psi_field, u_field, v_field

    @jax.jit
    def step(q_field: Array) -> tuple[Array, Array, Array, Array]:
        """Advance one Heun step and diagnose the balanced velocity field."""
        k1_q, _, _, _ = tendency(q_field)
        q_stage = enforce_periodic(q_field + dt * k1_q)
        k2_q, _, _, _ = tendency(q_stage)
        q_next = enforce_periodic(q_field + 0.5 * dt * (k1_q + k2_q))
        psi_next = invert_streamfunction(q_next)
        u_next, v_next = geostrophic_velocity_from_streamfunction(
            psi_next, diff, interp
        )
        return q_next, psi_next, u_next, v_next

    psi = invert_streamfunction(q)
    u, v = geostrophic_velocity_from_streamfunction(psi, diff, interp)

    snapshot_times: list[float] = []
    q_snapshots: list[np.ndarray] = []
    psi_snapshots: list[np.ndarray] = []
    u_snapshots: list[np.ndarray] = []
    v_snapshots: list[np.ndarray] = []
    speed_snapshots: list[np.ndarray] = []
    relative_vorticity_snapshots: list[np.ndarray] = []
    pv_enstrophy: list[float] = []

    def record_snapshot(
        step_index: int,
        q_field: Array,
        psi_field: Array,
        u_field: Array,
        v_field: Array,
    ) -> None:
        """Convert a sampled balanced state into ``xarray``-ready arrays."""
        q_np = np.asarray(jax.device_get(q_field[1:-1, 1:-1]))
        psi_np = np.asarray(jax.device_get(psi_field[1:-1, 1:-1]))
        u_center = interp.U_to_T(u_field)
        v_center = interp.V_to_T(v_field)
        zeta_corner = vort.relative_vorticity(u_field, v_field)
        zeta_center = interp.X_to_T(zeta_corner)
        u_np = np.asarray(jax.device_get(u_center[1:-1, 1:-1]))
        v_np = np.asarray(jax.device_get(v_center[1:-1, 1:-1]))
        zeta_np = np.asarray(jax.device_get(zeta_center[1:-1, 1:-1]))
        speed_np = np.sqrt(u_np**2 + v_np**2)

        snapshot_times.append(step_index * dt)
        q_snapshots.append(q_np)
        psi_snapshots.append(psi_np)
        u_snapshots.append(u_np)
        v_snapshots.append(v_np)
        speed_snapshots.append(speed_np)
        relative_vorticity_snapshots.append(zeta_np)
        pv_enstrophy.append(float(0.5 * np.mean(q_np**2)))

    record_snapshot(step_index=0, q_field=q, psi_field=psi, u_field=u, v_field=v)

    for iteration in range(1, config.steps + 1):
        q, psi, u, v = step(q)
        if iteration % config.snapshot_interval == 0 or iteration == config.steps:
            record_snapshot(
                step_index=iteration,
                q_field=q,
                psi_field=psi,
                u_field=u,
                v_field=v,
            )

    dataset = xr.Dataset(
        data_vars={
            "q": (
                ("time", "y", "x"),
                np.stack(q_snapshots, axis=0),
                {"long_name": "potential_vorticity_anomaly", "units": "s-1"},
            ),
            "psi": (
                ("time", "y", "x"),
                np.stack(psi_snapshots, axis=0),
                {"long_name": "streamfunction", "units": "m2 s-1"},
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
            "wind_curl": (
                ("y", "x"),
                forcing["wind_curl"].to_numpy(),
                {"long_name": "double_gyre_pv_forcing", "units": "s-2"},
            ),
            "pv_enstrophy": (
                ("time",),
                np.asarray(pv_enstrophy),
                {"long_name": "domain_mean_pv_enstrophy", "units": "s-2"},
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
            "model": "1.5-layer quasi-geostrophic",
            "configuration": "double gyre",
            "time_step_seconds": config.dt,
            "num_steps": config.steps,
            "notes": "Finitevolx advection plus FFT Helmholtz inversion with xarray output.",
        },
    )

    config.zarr_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(config.zarr_path, mode="w", consolidated=False)
    save_comparison_plot(
        dataset=dataset,
        figure_path=config.figure_path,
        variable_name="relative_vorticity",
        title="1.5-layer QG double gyre: relative vorticity",
        scale_factor=1.0e5,
        colorbar_label=r"[$10^{-5}\ \mathrm{s}^{-1}$]",
    )
    return dataset


def parse_args() -> QuasiGeostrophicConfig:
    """Parse the command line for the QG example."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = QuasiGeostrophicConfig()
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
        zarr_path = args.output_dir / "qg_1p5_layer_double_gyre.zarr"
        figure_path = args.output_dir / "qg_1p5_layer_double_gyre.png"

    return QuasiGeostrophicConfig(
        steps=args.steps,
        snapshot_interval=args.snapshot_interval,
        zarr_path=zarr_path,
        figure_path=figure_path,
    )


def main() -> None:
    """Run the 1.5-layer QG example from the command line."""
    config = parse_args()
    dataset = run_simulation(config)
    print(f"Saved QG fields to {config.zarr_path}")
    print(f"Saved QG comparison plot to {config.figure_path}")
    print(f"Final max |q| = {float(np.abs(dataset['q'].isel(time=-1)).max()):.3e} s^-1")


if __name__ == "__main__":
    main()
