"""
Shallow Water Equations (SWE) example using finitevolX.

Demonstrates the usage of finitevolX operators for a rotating shallow water
model on an Arakawa C-grid.

Equations (non-linear SWE)
--------------------------
  dh/dt = -div(h * u_vec)
  du/dt = -g * dh_dx + q * h * v   (PV flux form)
  dv/dt = -g * dh_dy - q * h * u   (PV flux form)

where q = (zeta + f) / h is the potential vorticity.

Usage
-----
  python scripts/shallow_water.py
"""

import jax
import jax.numpy as jnp

from finitevolx import (
    Advection2D,
    ArakawaCGrid2D,
    Difference2D,
    Interpolation2D,
    Vorticity2D,
    enforce_periodic,
)

# ---------------------------------------------------------------------------
# Grid setup
# ---------------------------------------------------------------------------

Lx = 1.0e6  # domain length in x [m]
Ly = 1.0e6  # domain length in y [m]
nx = 64  # interior cells in x
ny = 64  # interior cells in y

grid = ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)

diff = Difference2D(grid=grid)
interp = Interpolation2D(grid=grid)
adv = Advection2D(grid=grid)
vort = Vorticity2D(grid=grid)

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

g = 9.81  # gravitational acceleration [m/s^2]
f0 = 1.0e-4  # Coriolis parameter [1/s]
H0 = 100.0  # reference layer thickness [m]
dt = 100.0  # time step [s]
T = 1.0e5  # total simulation time [s]

# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def init_fields(grid: ArakawaCGrid2D):
    """Gaussian bump in height with zero velocity."""
    Ny, Nx = grid.Ny, grid.Nx
    # Cell-centre coordinates
    x = (jnp.arange(Nx) - 0.5) * grid.dx
    y = (jnp.arange(Ny) - 0.5) * grid.dy
    X, Y = jnp.meshgrid(x, y)

    xc = 0.5 * grid.Lx
    yc = 0.5 * grid.Ly
    sigma = 0.1 * grid.Lx

    # Height perturbation at T-points
    h = H0 + 0.1 * H0 * jnp.exp(-((X - xc) ** 2 + (Y - yc) ** 2) / (2.0 * sigma**2))
    u = jnp.zeros((Ny, Nx))  # U-points
    v = jnp.zeros((Ny, Nx))  # V-points
    return h, u, v


# ---------------------------------------------------------------------------
# Tendency functions
# ---------------------------------------------------------------------------


def swe_tendency(h, u, v, grid, g=g, f0=f0):
    """Compute SWE tendencies using finitevolX operators.

    Returns
    -------
    dh, du, dv : arrays of shape [Ny, Nx]
    """
    # Coriolis parameter at T-points (constant f-plane)
    f = f0 * jnp.ones_like(h)

    # --- height tendency: -div(h * u_vec) ---
    dh = adv(h, u, v, method="upwind1")

    # --- PV flux form momentum tendencies ---
    q = vort.potential_vorticity(u, v, h, f)  # q at X-points

    # Arakawa-Lamb PV flux
    qu, qv = vort.pv_flux_arakawa_lamb(q, u, v)  # qu at U, qv at V

    # Pressure gradient (geopotential phi = g*h)
    phi = g * h
    dphi_dx = diff.diff_x_T_to_U(phi)  # dphi/dx at U-points
    dphi_dy = diff.diff_y_T_to_V(phi)  # dphi/dy at V-points

    # h at faces for PV flux
    h_on_u = interp.T_to_U(h)
    h_on_v = interp.T_to_V(h)

    # du/dt = -dphi/dx + q * h * v_on_u
    du = jnp.zeros_like(u)
    du = du.at[1:-1, 1:-1].set(
        -dphi_dx[1:-1, 1:-1] + qu[1:-1, 1:-1] * h_on_u[1:-1, 1:-1]
    )

    # dv/dt = -dphi/dy - q * h * u_on_v
    dv = jnp.zeros_like(v)
    dv = dv.at[1:-1, 1:-1].set(
        -dphi_dy[1:-1, 1:-1] - qv[1:-1, 1:-1] * h_on_v[1:-1, 1:-1]
    )

    return dh, du, dv


# ---------------------------------------------------------------------------
# Time integration (simple forward Euler)
# ---------------------------------------------------------------------------


def step(h, u, v):
    """Single forward-Euler time step with periodic BCs."""
    dh, du, dv = swe_tendency(h, u, v, grid)
    h = enforce_periodic(h + dt * dh)
    u = enforce_periodic(u + dt * du)
    v = enforce_periodic(v + dt * dv)
    return h, u, v


step_jit = jax.jit(step)


def run():
    """Run the shallow water model."""
    h, u, v = init_fields(grid)
    h = enforce_periodic(h)

    nsteps = int(T / dt)
    print(f"Running SWE for {nsteps} steps on {nx}x{ny} grid ...")

    for i in range(nsteps):
        h, u, v = step_jit(h, u, v)
        if i % (nsteps // 10) == 0:
            ke = 0.5 * jnp.mean(
                h[1:-1, 1:-1] * (u[1:-1, 1:-1] ** 2 + v[1:-1, 1:-1] ** 2)
            )
            print(
                f"  step {i:5d}/{nsteps}  KE={float(ke):.4e}  h_max={float(h.max()):.4f}"
            )

    print("Done.")
    return h, u, v


if __name__ == "__main__":
    h, u, v = run()
