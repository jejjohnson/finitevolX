"""Golden-output regression tests for Diffusion / BiharmonicDiffusion /
MomentumAdvection mask fields (Phase 7).

Phase 7 is the BREAKING commit that unifies the diffusion/momentum mask
API on ``Mask2D`` / ``Mask3D`` class fields:

* ``diffusion_2d`` free function loses its ``mask_h``/``mask_u``/``mask_v``
  raw-array kwargs and becomes mask-free (Layer-2 functional per #209).
* ``Diffusion2D`` / ``Diffusion3D`` grow a ``mask`` class field; the
  three-step intermediate flux-masking pattern is inlined via
  ``_diffusion_2d_impl`` so ``Diffusion3D`` can vmap it per-z-slice.
* ``BiharmonicDiffusion2D`` / ``BiharmonicDiffusion3D`` grow a ``mask``
  class field; since #292 they apply the masked no-flux harmonic operator
  twice (``-κ · Dm(Dm(h))``) after zeroing the dry inputs.
* ``MomentumAdvection2D`` / ``MomentumAdvection3D`` grow a ``mask``
  class field with pass-down into the internal Difference2D /
  Interpolation2D, plus a final post-multiply by ``mask.u`` / ``mask.v``.
"""

from __future__ import annotations

import equinox as eqx
import numpy as np
import pytest

from finitevolx._src.diffusion.diffusion import (
    BiharmonicDiffusion2D,
    BiharmonicDiffusion3D,
    Diffusion2D,
    Diffusion3D,
)
from finitevolx._src.diffusion.momentum import (
    MomentumAdvection2D,
    MomentumAdvection3D,
)
from tests.fixtures._helpers import assert_matches_golden
from tests.fixtures.inputs import (
    make_grid_2d,
    make_grid_3d,
    make_h_field_2d,
    make_h_field_3d,
    make_mask_2d,
    make_mask_2d_all_ocean,
    make_mask_3d,
    make_mask_3d_all_ocean,
    make_u_field_2d,
    make_u_field_3d,
    make_v_field_2d,
    make_v_field_3d,
)

# ---------------------------------------------------------------------------
# Diffusion2D
# ---------------------------------------------------------------------------


class TestDiffusion2DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return Diffusion2D(grid=make_grid_2d())

    @pytest.fixture
    def op_masked(self):
        return Diffusion2D(grid=make_grid_2d(), mask=make_mask_2d())

    @pytest.fixture
    def op_all_ocean(self):
        return Diffusion2D(grid=make_grid_2d(), mask=make_mask_2d_all_ocean())

    def test_call_unmasked_golden(self, op_unmasked):
        out = op_unmasked(make_h_field_2d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion2D", "__call__", "unmasked")

    def test_call_masked_golden(self, op_masked):
        out = op_masked(make_h_field_2d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion2D", "__call__", "masked")

    def test_call_all_ocean(self, op_unmasked, op_all_ocean):
        h = make_h_field_2d()
        np.testing.assert_array_equal(
            op_all_ocean(h, kappa=1.0),
            op_unmasked(h, kappa=1.0),
        )

    def test_fluxes_unmasked_golden(self, op_unmasked):
        out = op_unmasked.fluxes(make_h_field_2d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion2D", "fluxes", "unmasked")

    def test_fluxes_masked_golden(self, op_masked):
        out = op_masked.fluxes(make_h_field_2d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion2D", "fluxes", "masked")

    def test_fluxes_all_ocean(self, op_unmasked, op_all_ocean):
        h = make_h_field_2d()
        fx1, fy1 = op_unmasked.fluxes(h, kappa=1.0)
        fx2, fy2 = op_all_ocean.fluxes(h, kappa=1.0)
        np.testing.assert_array_equal(fx1, fx2)
        np.testing.assert_array_equal(fy1, fy2)


# ---------------------------------------------------------------------------
# Diffusion3D
# ---------------------------------------------------------------------------


class TestDiffusion3DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return Diffusion3D(grid=make_grid_3d())

    @pytest.fixture
    def op_masked(self):
        return Diffusion3D(grid=make_grid_3d(), mask=make_mask_3d())

    @pytest.fixture
    def op_all_ocean(self):
        return Diffusion3D(grid=make_grid_3d(), mask=make_mask_3d_all_ocean())

    def test_call_unmasked_golden(self, op_unmasked):
        out = op_unmasked(make_h_field_3d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion3D", "__call__", "unmasked")

    def test_call_masked_golden(self, op_masked):
        out = op_masked(make_h_field_3d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion3D", "__call__", "masked")

    def test_call_all_ocean(self, op_unmasked, op_all_ocean):
        h = make_h_field_3d()
        np.testing.assert_array_equal(
            op_all_ocean(h, kappa=1.0),
            op_unmasked(h, kappa=1.0),
        )

    def test_fluxes_unmasked_golden(self, op_unmasked):
        out = op_unmasked.fluxes(make_h_field_3d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion3D", "fluxes", "unmasked")

    def test_fluxes_masked_golden(self, op_masked):
        out = op_masked.fluxes(make_h_field_3d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion3D", "fluxes", "masked")

    def test_fluxes_all_ocean(self, op_unmasked, op_all_ocean):
        h = make_h_field_3d()
        fx1, fy1 = op_unmasked.fluxes(h, kappa=1.0)
        fx2, fy2 = op_all_ocean.fluxes(h, kappa=1.0)
        np.testing.assert_array_equal(fx1, fx2)
        np.testing.assert_array_equal(fy1, fy2)


# ---------------------------------------------------------------------------
# BiharmonicDiffusion2D / 3D
# ---------------------------------------------------------------------------


class TestBiharmonicDiffusionMasks:
    def test_2d_unmasked_golden(self):
        op = BiharmonicDiffusion2D(grid=make_grid_2d())
        out = op(make_h_field_2d(), kappa=1e-3)
        assert_matches_golden(out, "BiharmonicDiffusion2D", "__call__", "unmasked")

    def test_2d_masked_golden(self):
        op = BiharmonicDiffusion2D(grid=make_grid_2d(), mask=make_mask_2d())
        out = op(make_h_field_2d(), kappa=1e-3)
        assert_matches_golden(out, "BiharmonicDiffusion2D", "__call__", "masked")

    def test_2d_all_ocean(self):
        op_plain = BiharmonicDiffusion2D(grid=make_grid_2d())
        op_all = BiharmonicDiffusion2D(
            grid=make_grid_2d(), mask=make_mask_2d_all_ocean()
        )
        h = make_h_field_2d()
        np.testing.assert_array_equal(op_plain(h, kappa=1e-3), op_all(h, kappa=1e-3))

    def test_2d_inner_harm_is_masked(self):
        """#292: both Laplacian passes are the masked no-flux ``Dm``, so the
        inner ``_harm`` carries the outer mask (the second pass zeroes the
        wet→dry fluxes, so the dry zeros of ``Dm(h)`` are never read).
        """
        mask = make_mask_2d()
        op = BiharmonicDiffusion2D(grid=make_grid_2d(), mask=mask)
        assert op._harm.mask is mask

    def test_3d_unmasked_golden(self):
        op = BiharmonicDiffusion3D(grid=make_grid_3d())
        out = op(make_h_field_3d(), kappa=1e-3)
        assert_matches_golden(out, "BiharmonicDiffusion3D", "__call__", "unmasked")

    def test_3d_masked_golden(self):
        op = BiharmonicDiffusion3D(grid=make_grid_3d(), mask=make_mask_3d())
        out = op(make_h_field_3d(), kappa=1e-3)
        assert_matches_golden(out, "BiharmonicDiffusion3D", "__call__", "masked")

    def test_3d_all_ocean(self):
        op_plain = BiharmonicDiffusion3D(grid=make_grid_3d())
        op_all = BiharmonicDiffusion3D(
            grid=make_grid_3d(), mask=make_mask_3d_all_ocean()
        )
        h = make_h_field_3d()
        np.testing.assert_array_equal(op_plain(h, kappa=1e-3), op_all(h, kappa=1e-3))

    def test_3d_inner_harm_is_masked(self):
        mask = make_mask_3d()
        op = BiharmonicDiffusion3D(grid=make_grid_3d(), mask=mask)
        assert op._harm.mask is mask


# ---------------------------------------------------------------------------
# MomentumAdvection2D / 3D
# ---------------------------------------------------------------------------


class TestMomentumAdvectionMasks:
    def test_2d_unmasked_golden(self):
        op = MomentumAdvection2D(grid=make_grid_2d())
        out = op(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "MomentumAdvection2D", "__call__", "unmasked")

    def test_2d_masked_golden(self):
        op = MomentumAdvection2D(grid=make_grid_2d(), mask=make_mask_2d())
        out = op(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "MomentumAdvection2D", "__call__", "masked")

    def test_2d_all_ocean(self):
        op_plain = MomentumAdvection2D(grid=make_grid_2d())
        op_all = MomentumAdvection2D(grid=make_grid_2d(), mask=make_mask_2d_all_ocean())
        u = make_u_field_2d()
        v = make_v_field_2d()
        du1, dv1 = op_plain(u, v)
        du2, dv2 = op_all(u, v)
        np.testing.assert_array_equal(du1, du2)
        np.testing.assert_array_equal(dv1, dv2)

    def test_3d_unmasked_golden(self):
        op = MomentumAdvection3D(grid=make_grid_3d())
        out = op(make_u_field_3d(), make_v_field_3d())
        assert_matches_golden(out, "MomentumAdvection3D", "__call__", "unmasked")

    def test_3d_masked_golden(self):
        op = MomentumAdvection3D(grid=make_grid_3d(), mask=make_mask_3d())
        out = op(make_u_field_3d(), make_v_field_3d())
        assert_matches_golden(out, "MomentumAdvection3D", "__call__", "masked")

    def test_3d_all_ocean(self):
        op_plain = MomentumAdvection3D(grid=make_grid_3d())
        op_all = MomentumAdvection3D(grid=make_grid_3d(), mask=make_mask_3d_all_ocean())
        u = make_u_field_3d()
        v = make_v_field_3d()
        du1, dv1 = op_plain(u, v)
        du2, dv2 = op_all(u, v)
        np.testing.assert_array_equal(du1, du2)
        np.testing.assert_array_equal(dv1, dv2)


# ---------------------------------------------------------------------------
# Dry-cell-zero invariants
# ---------------------------------------------------------------------------


class TestDryCellsAreZero:
    """Every dry T-cell is exactly 0 in the diffusion tendency, and every
    dry U/V face is exactly 0 in the momentum tendencies.
    """

    def test_diffusion2d(self):
        mask = make_mask_2d()
        op = Diffusion2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(op(make_h_field_2d(), kappa=1.0))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_diffusion3d(self):
        mask = make_mask_3d()
        op = Diffusion3D(grid=make_grid_3d(), mask=mask)
        out = np.asarray(op(make_h_field_3d(), kappa=1.0))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_biharmonic2d(self):
        mask = make_mask_2d()
        op = BiharmonicDiffusion2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(op(make_h_field_2d(), kappa=1e-3))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_biharmonic3d(self):
        mask = make_mask_3d()
        op = BiharmonicDiffusion3D(grid=make_grid_3d(), mask=mask)
        out = np.asarray(op(make_h_field_3d(), kappa=1e-3))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_momentum2d(self):
        mask = make_mask_2d()
        op = MomentumAdvection2D(grid=make_grid_2d(), mask=mask)
        du, dv = op(make_u_field_2d(), make_v_field_2d())
        du = np.asarray(du)
        dv = np.asarray(dv)
        assert np.all(du[np.asarray(~mask.u)] == 0.0)
        assert np.all(dv[np.asarray(~mask.v)] == 0.0)

    def test_momentum3d(self):
        mask = make_mask_3d()
        op = MomentumAdvection3D(grid=make_grid_3d(), mask=mask)
        du, dv = op(make_u_field_3d(), make_v_field_3d())
        du = np.asarray(du)
        dv = np.asarray(dv)
        assert np.all(du[np.asarray(~mask.u)] == 0.0)
        assert np.all(dv[np.asarray(~mask.v)] == 0.0)


# ---------------------------------------------------------------------------
# #292: the masked biharmonic is the masked no-flux operator applied twice
# ---------------------------------------------------------------------------


def _basin_mask_2d(n: int = 12) -> np.ndarray:
    """Closed basin (dry ghost ring) with a 2x3 island off the walls."""
    m = np.ones((n, n), dtype=bool)
    m[[0, -1], :] = False
    m[:, [0, -1]] = False
    m[5:7, 4:7] = False
    return m


def _biharmonic_cases():
    """(name, op, area weight [Ny, Nx], wet [Ny, Nx]) for the 2-D operators."""
    from finitevolx import (
        CartesianGrid2D,
        Mask2D,
        SphericalBiharmonicDiffusion2D,
        SphericalGrid2D,
    )

    m = _basin_mask_2d()
    mask = Mask2D.from_mask(m)
    cart = CartesianGrid2D.from_interior(10, 10, 1.0, 1.0)
    # R = 1 keeps the spherical operator O(1) so absolute tolerances mean
    # the same thing as for the unit-square Cartesian grid
    sph = SphericalGrid2D.from_interior(
        10, 10, lon_range=(0.0, 60.0), lat_range=(-30.0, 30.0), R=1.0
    )
    return [
        ("cartesian", BiharmonicDiffusion2D(grid=cart, mask=mask), np.ones(m.shape), m),
        (
            "spherical",
            SphericalBiharmonicDiffusion2D(grid=sph, mask=mask),
            np.asarray(sph.cos_lat_T),  # cell area ∝ cos(lat_T)
            m,
        ),
    ]


_CASES = _biharmonic_cases()


@pytest.mark.parametrize(("name", "op", "w", "wet"), _CASES, ids=[c[0] for c in _CASES])
class TestMaskedBiharmonicNoFlux:
    """``−κ · Dm(Dm(h))`` with ``Dm`` the masked no-flux Laplacian (#292).

    Before #292 the inner Laplacian was unmasked: a constant wet field with
    land = 1e10 gave a wet tendency of 1.9e15 and Σ_wet(B q) was O(1e6).
    """

    def test_constant_wet_field_land_1e10(self, name, op, w, wet):
        import jax.numpy as jnp

        h = jnp.where(jnp.asarray(wet), 2.0, 1e10)
        t = np.asarray(eqx.filter_jit(op)(h, 1.0))
        # exactly zero in exact arithmetic: every wet flux is (2 - 2) / dx
        assert np.abs(t[wet]).max() < 1e-12

    def test_land_nan_is_finite(self, name, op, w, wet):
        import jax
        import jax.numpy as jnp

        rng = np.random.default_rng(0)
        q = jnp.where(
            jnp.asarray(wet), jnp.asarray(rng.normal(size=wet.shape)), jnp.nan
        )
        assert bool(jnp.all(jnp.isfinite(eqx.filter_jit(op)(q, 1.0))))
        g = jax.jit(jax.grad(lambda x: op(x, 1.0).sum()))(q)
        assert bool(jnp.all(jnp.isfinite(g)))

    def test_conservation(self, name, op, w, wet):
        import jax.numpy as jnp

        rng = np.random.default_rng(1)
        q = jnp.asarray(rng.normal(size=wet.shape))
        bq = np.asarray(eqx.filter_jit(op)(q, 1.0)) * w
        # roundoff of a sum of O(100) terms, each O(scale)
        assert abs(bq[wet].sum()) < 1e-10 * np.abs(bq).sum()

    def test_dense_operator(self, name, op, w, wet):
        """Symmetric (in the area-weighted inner product), negative
        semi-definite with the wet constant in its null space, and no
        land→wet coupling."""
        import jax
        import jax.numpy as jnp

        n = wet.size
        M = np.asarray(
            jax.jit(jax.jacfwd(lambda x: op(x.reshape(wet.shape), 1.0).ravel()))(
                jnp.zeros(n)
            )
        )
        wv = wet.ravel()
        # land values never reach a wet cell
        assert np.abs(M[np.ix_(wv, ~wv)]).max() == 0.0
        A = M[np.ix_(wv, wv)]
        W = w.ravel()[wv]
        S = W[:, None] * A  # area-weighted: <p, B q>_w = p^T diag(w) B q
        scale = np.abs(S).max()
        np.testing.assert_allclose(S, S.T, atol=1e-12 * scale)
        # column sums of diag(w) B are zero: conservation for every q
        np.testing.assert_allclose(S.sum(axis=0), 0.0, atol=1e-11 * scale)
        eig = np.linalg.eigvalsh(0.5 * (S + S.T))
        assert eig.max() < 1e-10 * scale


def test_masked_biharmonic_3d_constant_and_conservation():
    """The 3-D operators apply the same masked ``Dm(Dm(h))`` per level."""
    import jax.numpy as jnp

    from finitevolx import (
        CartesianGrid3D,
        Mask3D,
        SphericalBiharmonicDiffusion3D,
        SphericalGrid3D,
    )

    m2 = _basin_mask_2d()
    m3 = np.stack([np.zeros_like(m2), m2, m2, m2, np.zeros_like(m2)])  # [Nz, Ny, Nx]
    mask = Mask3D.from_mask(m3)
    cart = CartesianGrid3D.from_interior(10, 10, 3, 1.0, 1.0, 1.0)
    sph = SphericalGrid3D.from_interior(
        10, 10, 3, lon_range=(0.0, 60.0), lat_range=(-30.0, 30.0), Lz=1.0, R=1.0
    )
    rng = np.random.default_rng(2)
    q = jnp.asarray(rng.normal(size=m3.shape))
    for op, w in [
        (BiharmonicDiffusion3D(grid=cart, mask=mask), np.ones(m2.shape)),
        (
            SphericalBiharmonicDiffusion3D(grid=sph, mask=mask),
            np.asarray(sph.cos_lat_T),
        ),
    ]:
        f = eqx.filter_jit(op)
        h = jnp.where(jnp.asarray(m3), 2.0, 1e10)
        assert np.abs(np.asarray(f(h, 1.0))[m3]).max() < 1e-12
        bq = np.asarray(f(q, 1.0)) * w[None]
        for k in range(1, 4):
            assert abs(bq[k][m2].sum()) < 1e-10 * np.abs(bq[k]).sum()
