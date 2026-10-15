"""The capacitance solver solves the CG problem exactly (#301, #306).

``build_capacitance_solver`` must return the solution of the masked
5-point problem ``masked_laplacian`` defines -- unknowns on every wet cell,
psi = 0 on the coast (the dry cells next to water) -- for every base and
lambda, including a singular base (``fft`` / ``dct`` at lambda = 0).

#301: with spectraldiffx < 0.1.1 a singular base left a uniform interior
residual equal to the wet mean of the right-hand side.  spectraldiffx 0.1.1
(gaussx's MaskedOperator borders the capacitance system with the base's
null vector) no longer shows it, so no base is refused; the test keeps the
singular combinations in the parametrisation to catch a regression.

#306: the solver used to constrain psi = 0 on the mask's inner *wet* ring,
one cell inside the coast where CG puts the wall.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import finitevolx as fvx

NY, NX = 34, 40
DX, DY = 1.0 / (NX - 2), 1.0 / (NY - 2)


def _island_basin() -> np.ndarray:
    """Dry ghost ring, wet interior, a 4x4 island in the middle."""
    m = np.zeros((NY, NX), dtype=bool)
    m[1:-1, 1:-1] = True
    cy, cx = NY // 2, NX // 2
    m[cy : cy + 4, cx : cx + 4] = False
    return m


def _rhs() -> jnp.ndarray:
    return jnp.asarray(np.random.default_rng(0).standard_normal((NY, NX)))


@pytest.mark.slow
@pytest.mark.parametrize("base_bc", ["dst", "fft", "dct"])
@pytest.mark.parametrize("lambda_", [0.0, 4.0])
def test_interior_residual(base_bc, lambda_):
    """5-point residual on every wet cell < 1e-10; psi = 0 on every dry cell."""
    mask = _island_basin()
    f = _rhs()
    solver = fvx.build_capacitance_solver(
        mask, DX, DY, lambda_=lambda_, base_bc=base_bc
    )
    psi = solver(f)
    mask_f = jnp.asarray(mask, dtype=float)
    res = fvx.masked_laplacian(psi, mask_f, DX, DY, lambda_=lambda_) - f * mask_f
    # A direct solve: round-off of |f| ~ 1 times the O(1/dx^2) stencil, ~1e-13.
    assert float(jnp.max(jnp.abs(res))) < 1e-10
    np.testing.assert_array_equal(np.asarray(psi)[~mask], 0.0)


def test_default_base_is_dst():
    solver = fvx.build_capacitance_solver(_island_basin(), DX, DY)
    assert solver.base_bc == "dst"


def test_unknown_base_raises():
    with pytest.raises(ValueError, match="base_bc must be"):
        fvx.build_capacitance_solver(_island_basin(), DX, DY, base_bc="dct1")


@pytest.mark.slow
def test_matches_cg_on_island_basin():
    """#306: capacitance and cg solve the same problem on the same Mask2D."""
    mask = fvx.Mask2D.from_mask(_island_basin())
    f = _rhs()
    lam = 4.0
    solver = fvx.build_capacitance_solver(mask, DX, DY, lambda_=lam)
    psi_cap = fvx.streamfunction_from_vorticity(
        f, DX, DY, lambda_=lam, method="capacitance", capacitance_solver=solver
    )
    psi_cg = fvx.streamfunction_from_vorticity(
        f, DX, DY, lambda_=lam, method="cg", mask=mask
    )
    # CG runs at rtol = atol = 1e-6 (float64 defaults); capacitance is exact.
    rel = float(jnp.max(jnp.abs(psi_cap - psi_cg)) / jnp.max(jnp.abs(psi_cg)))
    assert rel < 1e-5  # was 0.285 before the fix
    np.testing.assert_array_equal(
        np.asarray(psi_cap)[~np.asarray(mask.xy_corner_strict)], 0.0
    )


@pytest.mark.slow
def test_known_values_on_mask2d_faces():
    """#303 with capacitance: all four Dirichlet faces on the Mask2D wall, exact."""
    mask = fvx.Mask2D.from_mask(_island_basin())
    wet = jnp.asarray(mask.xy_corner_strict)
    lam = 4.0
    faces = {"south": -0.2, "north": 1.0, "west": 0.5, "east": 2.0}
    bc = fvx.BoundaryConditionSet(
        **{k: fvx.Dirichlet1D(k, value=v) for k, v in faces.items()}, mask=mask
    )
    f = _rhs()
    solver = fvx.build_capacitance_solver(mask, DX, DY, lambda_=lam)
    psi = fvx.streamfunction_from_vorticity(
        f, DX, DY, bc=bc, lambda_=lam, method="capacitance", capacitance_solver=solver
    )
    np.testing.assert_allclose(psi[0, 1:-2], faces["south"], atol=1e-12)
    np.testing.assert_allclose(psi[-2, 1:-2], faces["north"], atol=1e-12)
    np.testing.assert_allclose(psi[1:-2, 0], faces["west"], atol=1e-12)
    np.testing.assert_allclose(psi[1:-2, -2], faces["east"], atol=1e-12)
    # Residual on the solve domain (every wet node), stencil reading the wall
    # values; the island coast stays at zero.
    full = fvx.masked_laplacian(psi, jnp.ones_like(psi), DX, DY, lambda_=lam)
    res = jnp.where(wet, full - f, 0.0)
    assert float(jnp.max(jnp.abs(res))) < 1e-10
