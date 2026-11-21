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
one cell inside the coast where CG puts the wall.  The two tests of it
through ``streamfunction_from_vorticity`` moved to sesmax with the
inversions (#475, jejjohnson/sesmax#126).
"""

from __future__ import annotations

import gridax_jax as fvx
import jax.numpy as jnp
import numpy as np
import pytest

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


@pytest.mark.parametrize("base_bc", ["fft", "dct"])
def test_non_dst_base_rejects_wet_array_edge(base_bc):
    """A non-Dirichlet base reaches beyond the array edge, where CG has psi = 0."""
    mask = np.ones((NY, NX), dtype=bool)
    mask[NY // 2, :] = False  # dry somewhere, but wet on every array edge
    with pytest.raises(ValueError, match="wet cells on the edge"):
        fvx.build_capacitance_solver(mask, DX, DY, lambda_=4.0, base_bc=base_bc)


@pytest.mark.slow
def test_dst_base_matches_cg_with_wet_array_edge():
    """The Dirichlet base is masked_laplacian's zero pad, edge included."""
    mask = np.ones((NY, NX), dtype=bool)
    mask[NY // 2, 5:-5] = False
    f = _rhs()
    psi = fvx.build_capacitance_solver(mask, DX, DY, lambda_=4.0, base_bc="dst")(f)
    mask_f = jnp.asarray(mask, dtype=float)
    res = fvx.masked_laplacian(psi, mask_f, DX, DY, lambda_=4.0) - f * mask_f
    # direct solve: round-off of |f| ~ 1 times the O(1/dx^2) stencil
    assert float(jnp.max(jnp.abs(res))) < 1e-10
