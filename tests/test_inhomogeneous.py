"""Tests for boundary_ring, SolveDomain, and KnownValueLifting."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import finitevolx as fvx
from finitevolx._src.solvers.inhomogeneous import (
    KnownValueLifting,
    SolveDomain,
    boundary_ring,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _basin_mask(ny: int, nx: int) -> np.ndarray:
    """Simple rectangular basin mask: wet interior, dry border."""
    mask = np.zeros((ny, nx), dtype=np.float64)
    mask[1:-1, 1:-1] = 1.0
    return mask


def _island_mask(ny: int, nx: int) -> np.ndarray:
    """Basin with a 2x2 island in the middle."""
    mask = _basin_mask(ny, nx)
    cy, cx = ny // 2, nx // 2
    mask[cy : cy + 2, cx : cx + 2] = 0.0
    return mask


# ---------------------------------------------------------------------------
# boundary_ring
# ---------------------------------------------------------------------------


class TestBoundaryRing:
    def test_simple_basin(self):
        mask = jnp.array(_basin_mask(6, 8))
        ring = boundary_ring(mask)

        wet = mask > 0.5
        interior = jnp.zeros_like(wet)
        interior = interior.at[2:-2, 2:-2].set(True)

        assert ring.dtype == jnp.bool_
        assert ring.shape == mask.shape
        # Ring cells are wet
        assert jnp.all(ring <= wet)
        # Interior cells are not in the ring
        assert not jnp.any(ring & interior)
        # All wet non-interior cells are in the ring
        expected_ring = wet & ~interior
        np.testing.assert_array_equal(ring, expected_ring)

    def test_island_mask(self):
        mask = jnp.array(_island_mask(10, 10))
        ring = boundary_ring(mask)

        wet = mask > 0.5
        assert jnp.all(ring <= wet)

        # Cells adjacent to the island should also be in the ring
        cy, cx = 5, 5
        # Check cells around the island (north/south/east/west of island)
        island_neighbors = [
            (cy - 1, cx),
            (cy - 1, cx + 1),  # north of island
            (cy + 2, cx),
            (cy + 2, cx + 1),  # south of island
            (cy, cx - 1),
            (cy + 1, cx - 1),  # west of island
            (cy, cx + 2),
            (cy + 1, cx + 2),  # east of island
        ]
        for iy, ix in island_neighbors:
            assert ring[iy, ix], f"Cell ({iy},{ix}) should be in ring (island neighbor)"

    def test_all_wet_no_ring(self):
        mask = jnp.ones((6, 8))
        ring = boundary_ring(mask)
        # No dry cells → no ring
        assert not jnp.any(ring)

    def test_all_dry_no_ring(self):
        mask = jnp.zeros((6, 8))
        ring = boundary_ring(mask)
        assert not jnp.any(ring)

    def test_single_wet_cell(self):
        mask = jnp.zeros((5, 5))
        mask = mask.at[2, 2].set(1.0)
        ring = boundary_ring(mask)
        # The single wet cell has dry neighbors → it's in the ring
        assert ring[2, 2]
        assert int(jnp.sum(ring)) == 1

    def test_jit_compatible(self):
        mask = jnp.array(_basin_mask(6, 8))
        ring_eager = boundary_ring(mask)
        ring_jit = jax.jit(boundary_ring)(mask)
        np.testing.assert_array_equal(ring_eager, ring_jit)


# ---------------------------------------------------------------------------
# SolveDomain
# ---------------------------------------------------------------------------


class TestSolveDomain:
    def test_partition_is_complete(self):
        mask = jnp.array(_basin_mask(8, 8))
        domain = SolveDomain(mask)

        wet = mask > 0.5
        # all_known and effective_mask are disjoint
        assert not jnp.any(domain.all_known & domain.effective_mask)
        # their union is the full wet domain
        np.testing.assert_array_equal(domain.all_known | domain.effective_mask, wet)

    def test_no_known_mask(self):
        mask = jnp.array(_basin_mask(8, 8))
        domain = SolveDomain(mask)
        np.testing.assert_array_equal(domain.all_known, domain.boundary_ring)

    def test_with_known_mask(self):
        mask = jnp.array(_basin_mask(8, 8))
        known_mask = jnp.zeros((8, 8), dtype=bool)
        known_mask = known_mask.at[3, 3].set(True)

        domain = SolveDomain(mask, known_mask=known_mask)

        # Known mask cell is in all_known
        assert domain.all_known[3, 3]
        # But not in boundary_ring (it's interior)
        assert not domain.boundary_ring[3, 3]
        # And not in effective_mask (removed from solve domain)
        assert not domain.effective_mask[3, 3]

    def test_island_mask_finds_inner_rings(self):
        mask = jnp.array(_island_mask(10, 10))
        domain = SolveDomain(mask)

        # Cells adjacent to island are in all_known (via boundary_ring)
        cy, cx = 5, 5
        assert domain.all_known[cy - 1, cx]
        assert domain.all_known[cy + 2, cx]
        # And not in effective_mask
        assert not domain.effective_mask[cy - 1, cx]

    def test_dry_known_mask_harmless(self):
        mask = jnp.array(_basin_mask(8, 8))
        known_mask = jnp.zeros((8, 8), dtype=bool)
        known_mask = known_mask.at[0, 0].set(True)  # dry cell

        domain = SolveDomain(mask, known_mask=known_mask)

        # Dry cell in known_mask is filtered out (intersected with wet_mask)
        assert not domain.all_known[0, 0]
        assert not domain.effective_mask[0, 0]
        # Partition still holds
        wet = mask > 0.5
        np.testing.assert_array_equal(domain.all_known | domain.effective_mask, wet)


# ---------------------------------------------------------------------------
# KnownValueLifting
# ---------------------------------------------------------------------------


class TestKnownValueLifting:
    def test_homogeneous_is_identity(self):
        """known_values = 0 should produce rhs_corrected ≈ rhs on effective_mask."""
        mask = jnp.array(_basin_mask(8, 8))
        domain = SolveDomain(mask)
        lifter = KnownValueLifting(domain=domain, dx=1.0, dy=1.0, lambda_=0.0)

        rhs = jnp.ones((8, 8)) * mask
        known_values = jnp.zeros((8, 8))

        rhs_corrected, value_lift = lifter.preprocess(rhs, known_values)

        # Lift is zero everywhere
        np.testing.assert_allclose(value_lift, 0.0, atol=1e-15)
        # Corrected RHS equals original RHS on the effective domain
        eff = domain.effective_mask.astype(jnp.float64)
        np.testing.assert_allclose(rhs_corrected, rhs * eff, atol=1e-12)

    def test_postprocess_adds_lift(self):
        mask = jnp.array(_basin_mask(8, 8))
        domain = SolveDomain(mask)
        lifter = KnownValueLifting(domain=domain, dx=1.0, dy=1.0, lambda_=0.0)

        psi_hom = jnp.ones((8, 8)) * 2.0
        value_lift = jnp.ones((8, 8)) * 3.0

        psi = lifter.postprocess(psi_hom, value_lift)
        # postprocess masks psi_hom to effective_mask before adding lift
        eff = domain.effective_mask
        expected = value_lift + jnp.where(eff, 2.0, 0.0)
        np.testing.assert_allclose(psi, expected)

    def test_manufactured_solution_cg(self):
        """Verify the full lifting trick with a manufactured solution."""
        Ny, Nx = 34, 34
        dx = dy = 1.0 / (Ny - 2)
        lambda_ = 4.0

        mask = jnp.array(_basin_mask(Ny, Nx))
        domain = SolveDomain(mask)
        lifter = KnownValueLifting(domain=domain, dx=dx, dy=dy, lambda_=lambda_)

        # Manufactured solution: doesn't vanish at boundaries
        x = jnp.linspace(0, 1, Nx)
        y = jnp.linspace(0, 1, Ny)
        X, Y = jnp.meshgrid(x, y)
        psi_exact = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y) + 0.1 * jnp.sin(
            2 * jnp.pi * Y
        )

        # Exact RHS: (∇² - λ)ψ
        rhs_exact = fvx.masked_laplacian(psi_exact, mask, dx, dy, lambda_=lambda_)

        # Known values at inner ring: exact solution values
        known_values = jnp.where(domain.boundary_ring, psi_exact, 0.0)

        # Pre-process
        rhs_corrected, value_lift = lifter.preprocess(rhs_exact, known_values)

        # Solve with CG on effective_mask
        eff_mask_f = domain.effective_mask.astype(jnp.float64)

        def matvec(x):
            return fvx.masked_laplacian(x, eff_mask_f, dx, dy, lambda_=lambda_)

        psi_hom, _info = fvx.solve_cg(
            matvec, rhs_corrected, rtol=1e-10, atol=1e-10, max_steps=2000
        )
        psi_hom = psi_hom * eff_mask_f

        # Reconstruct
        psi_numerical = lifter.postprocess(psi_hom, value_lift)

        # Check error on effective_mask (interior solve cells)
        error = jnp.abs(psi_numerical - psi_exact) * eff_mask_f
        max_error = float(jnp.max(error))
        assert max_error < 1e-3, f"Max error {max_error:.2e} exceeds tolerance"

    def test_sparse_obs_pins_values(self):
        """Verify that known_mask pins interior cells correctly."""
        Ny, Nx = 16, 16
        dx = dy = 1.0

        mask = jnp.array(_basin_mask(Ny, Nx))

        obs_mask = jnp.zeros((Ny, Nx), dtype=bool)
        obs_mask = obs_mask.at[5, 5].set(True)
        obs_mask = obs_mask.at[8, 10].set(True)

        domain = SolveDomain(mask, known_mask=obs_mask)
        lifter = KnownValueLifting(domain=domain, dx=dx, dy=dy, lambda_=0.0)

        known_values = jnp.zeros((Ny, Nx))
        known_values = known_values.at[5, 5].set(3.0)
        known_values = known_values.at[8, 10].set(-2.0)

        rhs = jnp.zeros((Ny, Nx))
        rhs_corrected, value_lift = lifter.preprocess(rhs, known_values)

        # Obs cells should appear in the lift
        assert float(value_lift[5, 5]) == 3.0
        assert float(value_lift[8, 10]) == -2.0

        # Solve
        eff_mask_f = domain.effective_mask.astype(jnp.float64)

        def matvec(x):
            return fvx.masked_laplacian(x, eff_mask_f, dx, dy, lambda_=0.0)

        psi_hom, _ = fvx.solve_cg(matvec, rhs_corrected, rtol=1e-10, atol=1e-10)
        psi_hom = psi_hom * eff_mask_f

        psi = lifter.postprocess(psi_hom, value_lift)

        # Observation cells should be pinned to their values
        np.testing.assert_allclose(psi[5, 5], 3.0, atol=1e-10)
        np.testing.assert_allclose(psi[8, 10], -2.0, atol=1e-10)

    def test_jit_compatible(self):
        mask = jnp.array(_basin_mask(8, 8))
        domain = SolveDomain(mask)
        lifter = KnownValueLifting(domain=domain, dx=1.0, dy=1.0, lambda_=0.0)

        rhs = jnp.ones((8, 8)) * mask
        kv = jnp.zeros((8, 8))

        @jax.jit
        def run(r, k):
            return lifter.preprocess(r, k)

        rhs_c, vlift = run(rhs, kv)
        assert rhs_c.shape == (8, 8)
        assert vlift.shape == (8, 8)


# ---------------------------------------------------------------------------
# BoundaryConditionSet.mask and .closed()
# ---------------------------------------------------------------------------


class TestBoundaryConditionSetMask:
    def test_mask_field_default_none(self):
        bc = fvx.BoundaryConditionSet()
        assert bc.mask is None

    def test_mask_field_accepts_array(self):
        mask = jnp.ones((8, 8))
        bc = fvx.BoundaryConditionSet(mask=mask)
        assert bc.mask is not None

    def test_closed_factory(self):
        mask = jnp.ones((8, 8))
        bc = fvx.BoundaryConditionSet.closed(mask=mask)
        assert bc.mask is not None
        assert isinstance(bc.south, fvx.Dirichlet1D)
        assert isinstance(bc.north, fvx.Dirichlet1D)
        assert isinstance(bc.west, fvx.Dirichlet1D)
        assert isinstance(bc.east, fvx.Dirichlet1D)
        assert bc.south.value == 0.0

    def test_closed_factory_no_mask(self):
        bc = fvx.BoundaryConditionSet.closed()
        assert bc.mask is None
        assert isinstance(bc.south, fvx.Dirichlet1D)

    def test_existing_call_still_works(self):
        bc = fvx.BoundaryConditionSet(
            south=fvx.Periodic1D("south"),
            north=fvx.Periodic1D("north"),
            west=fvx.Periodic1D("west"),
            east=fvx.Periodic1D("east"),
        )
        field = jnp.ones((6, 8))
        result = bc(field, dx=1.0, dy=1.0)
        assert result.shape == (6, 8)
