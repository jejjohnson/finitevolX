"""Tests for the ``wall="open"`` lateral-boundary mode (issue #234).

The advection / diffusion operators default to ``wall="closed"`` — no-flux
lateral walls, the correct convention for a closed / land-masked basin.
``wall="open"`` instead assembles the domain-wall face fluxes from the
caller-filled ghost ring, so Dirichlet / outflow / periodic lateral
boundaries drive horizontal transport.  These tests check that:

* the closed default is untouched (bit-for-bit),
* open mode writes the previously-frozen wall-adjacent ring,
* the open-mode flux field is single-valued so the tendency is exactly
  flux-conservative (total interior tendency == wall-flux imbalance), and
* periodic ghosts wrap and conserve mass.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.advection.advection import (
    Advection1D,
    Advection2D,
    Advection3D,
)
from finitevolx._src.diffusion.diffusion import Diffusion2D, Diffusion3D
from finitevolx._src.grid.cartesian import (
    CartesianGrid1D,
    CartesianGrid2D,
    CartesianGrid3D,
)


# ── fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def grid1d():
    return CartesianGrid1D.from_interior(8, 1.0)


@pytest.fixture
def grid2d():
    return CartesianGrid2D.from_interior(nx_interior=8, ny_interior=8, Lx=8.0, Ly=8.0)


@pytest.fixture
def grid3d():
    return CartesianGrid3D.from_interior(
        nx_interior=8, ny_interior=8, nz_interior=4, Lx=8.0, Ly=8.0, Lz=4.0
    )


def _periodic_x(f):
    return f.at[..., :, 0].set(f[..., :, -2]).at[..., :, -1].set(f[..., :, 1])


def _periodic_y(f):
    return f.at[..., 0, :].set(f[..., -2, :]).at[..., -1, :].set(f[..., 1, :])


# ── closed default is unchanged ─────────────────────────────────────────────
def test_advection_default_is_closed(grid3d):
    op = Advection3D(grid=grid3d)
    rng = np.random.default_rng(0)
    shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
    c = jnp.asarray(rng.normal(size=shape).astype(np.float32))
    u = jnp.asarray(rng.normal(size=shape).astype(np.float32))
    v = jnp.asarray(rng.normal(size=shape).astype(np.float32))
    default = op(c, u, v, method="weno5")
    closed = op(c, u, v, method="weno5", wall="closed")
    np.testing.assert_array_equal(np.asarray(default), np.asarray(closed))


def test_diffusion_default_is_closed(grid3d):
    op = Diffusion3D(grid=grid3d)
    rng = np.random.default_rng(1)
    shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
    c = jnp.asarray(rng.normal(size=shape).astype(np.float32))
    np.testing.assert_array_equal(
        np.asarray(op(c, 0.3)), np.asarray(op(c, 0.3, wall="closed"))
    )


def test_closed_freezes_wall_ring_open_updates_it(grid3d):
    """Closed leaves the wall-adjacent interior ring at zero; open fills it."""
    op = Advection3D(grid=grid3d)
    c = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx)).at[2, 4, 4].set(1.0)
    u = jnp.ones_like(c) * 0.7
    v = jnp.ones_like(c) * 0.5
    closed = op(c, u, v, method="weno5", wall="closed")
    opened = op(c, u, v, method="weno5", wall="open")
    # Wall-adjacent interior ring rows/cols (j=1 and i=1) are exactly 0 closed.
    assert float(jnp.max(jnp.abs(closed[1:-1, 1, 1:-1]))) == 0.0
    assert float(jnp.max(jnp.abs(closed[1:-1, 1:-1, 1]))) == 0.0
    # Deep interior is identical between modes (open only changes wall fluxes).
    np.testing.assert_allclose(
        np.asarray(closed[1:-1, 2:-2, 2:-2]),
        np.asarray(opened[1:-1, 2:-2, 2:-2]),
        rtol=1e-5,
        atol=1e-6,
    )


# ── exact flux-conservation (single-valued flux field) ──────────────────────
def _wall_imbalance_2d(h, u, v, dx, dy):
    """Independently reconstruct the first-order wall-face flux imbalance."""
    west = jnp.where(u[:, 0] >= 0.0, h[:, 0], h[:, 1]) * u[:, 0]
    east = jnp.where(u[:, -2] >= 0.0, h[:, -2], h[:, -1]) * u[:, -2]
    south = jnp.where(v[0, :] >= 0.0, h[0, :], h[1, :]) * v[0, :]
    north = jnp.where(v[-2, :] >= 0.0, h[-2, :], h[-1, :]) * v[-2, :]
    x_term = jnp.sum(east[1:-1] - west[1:-1]) / dx
    y_term = jnp.sum(north[1:-1] - south[1:-1]) / dy
    return -(x_term + y_term)


def test_advection_open_is_exactly_conservative_2d(grid2d):
    """Σ(interior tendency) equals the wall-flux imbalance for every scheme."""
    op = Advection2D(grid=grid2d)
    rng = np.random.default_rng(3)
    shape = (grid2d.Ny, grid2d.Nx)
    for method in ("upwind1", "weno3", "weno5", "van_leer"):
        c = jnp.asarray(rng.normal(size=shape).astype(np.float64))
        u = jnp.asarray(rng.normal(size=shape).astype(np.float64))
        v = jnp.asarray(rng.normal(size=shape).astype(np.float64))
        tend = op(c, u, v, method=method, wall="open")
        total = float(jnp.sum(tend[1:-1, 1:-1]))
        expected = float(_wall_imbalance_2d(c, u, v, grid2d.dx, grid2d.dy))
        np.testing.assert_allclose(total, expected, rtol=1e-6, atol=1e-6)


def test_advection_periodic_conserves_mass_2d(grid2d):
    """Fully-periodic uniform advection conserves interior mass over time."""
    op = Advection2D(grid=grid2d)
    rng = np.random.default_rng(4)
    c = jnp.asarray(rng.uniform(size=(grid2d.Ny, grid2d.Nx)).astype(np.float64))
    u = jnp.ones_like(c) * 0.9
    v = jnp.ones_like(c) * -0.6
    dt = 0.05
    mass0 = float(jnp.sum(c[1:-1, 1:-1]))
    for _ in range(150):
        c = _periodic_x(_periodic_y(c))
        c = c + dt * op(c, u, v, method="upwind1", wall="open")
    mass1 = float(jnp.sum(c[1:-1, 1:-1]))
    # Exact in real arithmetic; tolerance covers float32 accumulation only.
    np.testing.assert_allclose(mass1, mass0, rtol=1e-5)


def test_advection_periodic_y_wraps_3d(grid3d):
    """A blob advecting in +y past the north wall re-enters at the south."""
    op = Advection3D(grid=grid3d)
    c = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx)).at[2, -2, 4].set(1.0)
    u = jnp.zeros_like(c)
    v = jnp.ones_like(c) * 1.0
    dt = 0.2
    mass0 = float(jnp.sum(c[1:-1, 1:-1, 1:-1]))
    for _ in range(60):
        c = _periodic_y(c)
        c = c + dt * op(c, u, v, method="upwind1", wall="open")
    # Mass conserved (wrapped, not lost) and some mass has reached the south half.
    np.testing.assert_allclose(float(jnp.sum(c[1:-1, 1:-1, 1:-1])), mass0, rtol=1e-6)
    south_half = float(jnp.sum(c[2, 1 : grid3d.Ny // 2, 4]))
    assert south_half > 1e-3


def test_advection_outflow_loses_mass_only_through_open_boundary(grid2d):
    """Outflow east + zero-Dirichlet elsewhere: mass leaves only downwind."""
    op = Advection2D(grid=grid2d)
    # Blob near the east side, uniform +x wind.
    c = jnp.zeros((grid2d.Ny, grid2d.Nx)).at[4, -3].set(1.0)
    u = jnp.ones_like(c) * 1.0
    v = jnp.zeros_like(c)
    dt = 0.1
    masses = []
    for _ in range(40):
        # zero-gradient (outflow) east ghost; zero-Dirichlet elsewhere.
        c = c.at[:, -1].set(c[:, -2])  # east outflow
        c = c.at[:, 0].set(-c[:, 1])  # west dirichlet 0
        c = c.at[0, :].set(-c[1, :]).at[-1, :].set(-c[-2, :])  # y walls dirichlet 0
        c = c + dt * op(c, u, v, method="upwind1", wall="open")
        masses.append(float(jnp.sum(c[1:-1, 1:-1])))
    # Monotonically non-increasing, and strictly lost by the end (exited east).
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(masses))
    assert masses[-1] < 0.5


def test_advection_1d_open_conservative(grid1d):
    op = Advection1D(grid=grid1d)
    rng = np.random.default_rng(5)
    c = jnp.asarray(rng.normal(size=(grid1d.Nx,)).astype(np.float64))
    u = jnp.asarray(rng.normal(size=(grid1d.Nx,)).astype(np.float64))
    tend = op(c, u, method="weno5", wall="open")
    total = float(jnp.sum(tend[1:-1]))
    west = float(jnp.where(u[0] >= 0.0, c[0], c[1]) * u[0])
    east = float(jnp.where(u[-2] >= 0.0, c[-2], c[-1]) * u[-2])
    np.testing.assert_allclose(total, -(east - west) / grid1d.dx, rtol=1e-6, atol=1e-6)


# ── diffusion open mode ─────────────────────────────────────────────────────
def test_diffusion_open_is_exactly_conservative_2d(grid2d):
    op = Diffusion2D(grid=grid2d)
    rng = np.random.default_rng(6)
    h = jnp.asarray(rng.normal(size=(grid2d.Ny, grid2d.Nx)).astype(np.float64))
    dx, dy = grid2d.dx, grid2d.dy
    tend = op(h, 0.7, wall="open")
    fx, fy = op.fluxes(h, 0.7, wall="open")
    # Σ tendency telescopes to the wall-face flux imbalance.
    x_term = jnp.sum(fx[1:-1, -2] - fx[1:-1, 0]) / dx
    y_term = jnp.sum(fy[-2, 1:-1] - fy[0, 1:-1]) / dy
    np.testing.assert_allclose(
        float(jnp.sum(tend[1:-1, 1:-1])), float(x_term + y_term), rtol=1e-6, atol=1e-8
    )


def test_diffusion_open_field_kappa_ignores_coefficient_ghost(grid2d):
    """Field kappa with an *unfilled* (zero) ghost ring must not zero the
    west/south wall fluxes — those faces source kappa from the interior cell."""
    op = Diffusion2D(grid=grid2d)
    rng = np.random.default_rng(11)
    Ny, Nx = grid2d.Ny, grid2d.Nx
    # Coefficient populated on interior T-cells only; ghost ring left at 0.
    kappa = np.zeros((Ny, Nx))
    kappa[1:-1, 1:-1] = rng.uniform(0.3, 1.0, size=(Ny - 2, Nx - 2))
    kappa = jnp.asarray(kappa)
    h = jnp.asarray(rng.normal(size=(Ny, Nx)).astype(np.float64))
    fx, fy = op.fluxes(h, kappa, wall="open")
    # West wall face (col 0) and south wall face (row 0) carry a real flux.
    assert float(jnp.max(jnp.abs(fx[1:-1, 0]))) > 0.0
    assert float(jnp.max(jnp.abs(fy[0, 1:-1]))) > 0.0
    # Still exactly flux-conservative.
    tend = op(h, kappa, wall="open")
    x_term = jnp.sum(fx[1:-1, -2] - fx[1:-1, 0]) / grid2d.dx
    y_term = jnp.sum(fy[-2, 1:-1] - fy[0, 1:-1]) / grid2d.dy
    np.testing.assert_allclose(
        float(jnp.sum(tend[1:-1, 1:-1])), float(x_term + y_term), rtol=1e-6, atol=1e-8
    )


def test_diffusion_dirichlet_wall_flux(grid2d):
    """A Dirichlet ghost drives a nonzero wall diffusive flux (open only)."""
    op = Diffusion2D(grid=grid2d)
    kappa = 0.5
    h = jnp.ones((grid2d.Ny, grid2d.Nx))  # uniform interior => zero interior flux
    # West Dirichlet with boundary value 0 => ghost = 2*0 - interior = -1.
    h = h.at[:, 0].set(-h[:, 1])
    closed = op(h, kappa, wall="closed")
    opened = op(h, kappa, wall="open")
    # Closed: no wall flux anywhere -> zero tendency for the uniform interior.
    np.testing.assert_allclose(np.asarray(closed[1:-1, 1:-1]), 0.0, atol=1e-6)
    # Open: the west edge column (i=1) gets a nonzero inward diffusive flux.
    west_edge = opened[1:-1, 1]
    assert float(jnp.max(jnp.abs(west_edge))) > 1e-3
    # Cells away from any wall stay zero (uniform field, interior flux == 0).
    np.testing.assert_allclose(np.asarray(opened[2:-2, 2:-2]), 0.0, atol=1e-6)


def test_diffusion_periodic_conserves_mass_3d(grid3d):
    op = Diffusion3D(grid=grid3d)
    rng = np.random.default_rng(7)
    shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
    c = jnp.asarray(rng.normal(size=shape).astype(np.float64))
    dt = 0.02
    mass0 = float(jnp.sum(c[1:-1, 1:-1, 1:-1]))
    for _ in range(80):
        c = _periodic_x(_periodic_y(c))
        c = c + dt * op(c, 0.4, wall="open")
    # Exact in real arithmetic; tolerance covers float32 accumulation only.
    np.testing.assert_allclose(float(jnp.sum(c[1:-1, 1:-1, 1:-1])), mass0, rtol=1e-5)


# ── validation ──────────────────────────────────────────────────────────────
def test_invalid_wall_raises(grid2d):
    op = Advection2D(grid=grid2d)
    c = jnp.zeros((grid2d.Ny, grid2d.Nx))
    with pytest.raises(ValueError, match=r"wall must be"):
        op(c, c, c, method="upwind1", wall="bogus")


def test_open_with_mask_raises(grid2d):
    from finitevolx._src.mask import Mask2D

    mask = Mask2D.from_mask(jnp.ones((grid2d.Ny, grid2d.Nx)))
    op = Advection2D(grid=grid2d, mask=mask)
    c = jnp.zeros((grid2d.Ny, grid2d.Nx))
    with pytest.raises(NotImplementedError, match=r"open.*mask|mask"):
        op(c, c, c, method="weno5", wall="open")
