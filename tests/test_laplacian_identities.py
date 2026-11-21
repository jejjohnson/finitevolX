"""The Laplacians that several operators compute agree with one another (#455).

    ∇²h = ∇ · (∇h) = ∇ · (κ ∇h) with κ = 1,

and on the sphere, with R the radius and φ the latitude,

    ∇²h = 1/(R² cos²φ) ∂²h/∂λ² + 1/(R² cos φ) ∂/∂φ (cos φ ∂h/∂φ).

The diffusion operators close the domain walls (their wall-face fluxes are
zero) while the Laplacians read the ghost ring, so the two agree only on the
T points at least two cells from the edge of the array, ``[2:-2, 2:-2]``.
Each identity is checked on NumPy and on JAX under ``jax.jit``.
"""

from __future__ import annotations

import gridax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

AWAY_FROM_WALLS = (slice(2, -2), slice(2, -2))
INTERIOR = (slice(1, -1), slice(1, -1))
# float64 rounding of a few-term sum, relative to the field's largest Laplacian
RTOL = 1e-14


def _close(a, b, cells) -> None:
    a, b = np.asarray(a)[..., *cells], np.asarray(b)[..., *cells]
    scale = np.abs(a).max()
    assert np.abs(a - b).max() <= RTOL * scale


def _backends(fn):
    """The operator eagerly on NumPy, and jitted on JAX."""

    def jitted(*args):
        return jax.jit(fn)(*map(jnp.asarray, args))

    return [fn, jitted]


@pytest.fixture
def cartesian():
    grid = gridax.CartesianGrid2D.from_interior(30, 20, 60.0e3, 20.0e3)
    h = np.random.default_rng(1).standard_normal((grid.Ny, grid.Nx))
    return grid, h


@pytest.fixture
def spherical():
    grid = gridax.SphericalGrid2D.from_interior(30, 20, (0.0, 120.0), (-60.0, 60.0))
    h = np.random.default_rng(2).standard_normal((grid.Ny, grid.Nx))
    return grid, h


@pytest.mark.parametrize("backend", [0, 1])
def test_cartesian_laplacian_is_the_divergence_of_the_gradient(cartesian, backend):
    grid, h = cartesian
    diff = gridax.Difference2D(grid=grid)

    def div_grad(h):
        return diff.divergence(diff.diff_x_T_to_U(h), diff.diff_y_T_to_V(h))

    laplacian = _backends(diff.laplacian)[backend](h)
    _close(laplacian, _backends(div_grad)[backend](h), AWAY_FROM_WALLS)


@pytest.mark.parametrize("backend", [0, 1])
def test_cartesian_laplacian_is_diffusion_with_unit_diffusivity(cartesian, backend):
    grid, h = cartesian
    laplacian = _backends(gridax.Difference2D(grid=grid).laplacian)[backend](h)
    diffusion = gridax.Diffusion2D(grid=grid)

    def unit(h):
        return diffusion(h, kappa=1.0)

    _close(laplacian, _backends(unit)[backend](h), AWAY_FROM_WALLS)


def test_cartesian_laplacian_and_diffusion_differ_at_the_closed_walls(cartesian):
    # the wall cells read the ghost ring in one and a zero wall flux in the other
    grid, h = cartesian
    laplacian = np.asarray(gridax.Difference2D(grid=grid).laplacian(h))
    diffusion = np.asarray(gridax.Diffusion2D(grid=grid)(h, kappa=1.0))
    wall_row = laplacian[1, 2:-2] - diffusion[1, 2:-2]
    assert np.abs(wall_row).max() > 1e-3 * np.abs(laplacian).max()


@pytest.mark.parametrize("backend", [0, 1])
def test_spherical_laplacian_is_diffusion_with_unit_diffusivity(spherical, backend):
    grid, h = spherical
    laplacian = _backends(gridax.SphericalLaplacian2D(grid=grid))[backend](h)
    diffusion = gridax.SphericalDiffusion2D(grid=grid)

    def unit(h):
        return diffusion(h, kappa=1.0)

    _close(laplacian, _backends(unit)[backend](h), AWAY_FROM_WALLS)


def test_spherical_laplacian_is_its_two_terms_bit_for_bit(spherical):
    grid, h = spherical
    laplacian = np.asarray(gridax.SphericalLaplacian2D(grid=grid)(h))
    diff = gridax.SphericalDifference2D(grid=grid)
    terms = np.asarray(diff.diff2_lon(h)) + np.asarray(diff.laplacian_merid(h))
    np.testing.assert_array_equal(laplacian[INTERIOR], terms[INTERIOR])


def test_3d_laplacian_is_diffusion_with_unit_diffusivity_on_every_level():
    grid = gridax.CartesianGrid3D.from_interior(16, 12, 4, 32.0e3, 24.0e3, 400.0)
    h = np.random.default_rng(3).standard_normal((grid.Nz, grid.Ny, grid.Nx))
    laplacian = np.asarray(gridax.Difference3D(grid=grid).laplacian(h))
    diffusion = np.asarray(gridax.Diffusion3D(grid=grid)(h, kappa=1.0))
    for level in range(1, grid.Nz - 1):
        _close(laplacian[level], diffusion[level], AWAY_FROM_WALLS)
