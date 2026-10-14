"""How the stencils lower: one slice per offset, fused tendencies, same values.

The operators below were rewritten so that reverse-mode AD stays a single
fused pass (see ``docs/performance.md``):

* each shifted view of an input is sliced once (``_neighbours``), so the VJP
  has one ``pad`` per *distinct* offset instead of one per slice expression;
* diffusion computes only the face fluxes the divergence reads and pads in
  the zero wall faces, instead of building full-size flux buffers;
* 3-D operators zero their z-ghost slabs with a pad of the interior slabs.

These tests pin both halves of that contract:

* **values** — each rewrite equals the original composed formula (bitwise in
  float64 for the forward pass, since the arithmetic is unchanged);
* **lowering** — the VJP jaxpr contains one ``pad`` per distinct offset.  This
  is a JAX-level property (no XLA, no timing), so it is deterministic on every
  backend and catches a regression back to slice-per-term code.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import finitevolx as fvx
from finitevolx._src.diffusion.diffusion import _diffusion_2d_fluxes_impl
from finitevolx._src.operators._ghost import interior, zero_z_ghosts
from finitevolx._src.operators.stencils import (
    _materialize,
    _neighbours,
    diff_x_bwd,
    diff_x_bwd_1d,
    diff_x_bwd_3d,
    diff_x_fwd,
    diff_x_fwd_1d,
    diff_x_fwd_3d,
    diff_y_bwd,
    diff_y_bwd_3d,
    diff_y_fwd,
    diff_y_fwd_3d,
)

NY, NX = 9, 12  # total shape incl. ghost ring; deliberately non-square


def _field(*shape, seed=0):
    return jax.random.normal(jax.random.key(seed), shape, dtype=jnp.float64)


def _primitives(jaxpr) -> set[str]:
    """Names of every primitive in ``jaxpr``, including nested sub-jaxprs."""
    names = set()
    for eqn in jaxpr.eqns:
        names.add(eqn.primitive.name)
        for param in eqn.params.values():
            sub = getattr(param, "jaxpr", None)
            if sub is not None:
                names |= _primitives(getattr(sub, "jaxpr", sub))
    return names


def _vjp_pads(fn, *xs):
    """Number of ``pad`` primitives in the VJP (transpose) jaxpr of ``fn``."""
    out, pullback = jax.vjp(fn, *xs)
    jaxpr = jax.make_jaxpr(pullback)(out)
    return sum(1 for eqn in jaxpr.jaxpr.eqns if eqn.primitive.name == "pad")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestNeighbours:
    @pytest.mark.slow
    @pytest.mark.parametrize("shape", [(NY, NX), (3, NY, NX), (2, 3, NY, NX)])
    def test_views_match_slices(self, shape):
        h = _field(*shape)
        nb = _neighbours(h)
        expected = {
            "c": h[..., 1:-1, 1:-1],
            "e": h[..., 1:-1, 2:],
            "w": h[..., 1:-1, :-2],
            "n": h[..., 2:, 1:-1],
            "s": h[..., :-2, 1:-1],
            "ne": h[..., 2:, 2:],
            "nw": h[..., 2:, :-2],
            "se": h[..., :-2, 2:],
            "sw": h[..., :-2, :-2],
        }
        for name, ref in expected.items():
            np.testing.assert_array_equal(getattr(nb, name), ref)

    def test_offsets_are_the_raw_stencil_offsets(self):
        h = _field(NY, NX)
        nb = _neighbours(h)
        np.testing.assert_array_equal(nb.e - nb.c, diff_x_fwd(h))
        np.testing.assert_array_equal(nb.c - nb.w, diff_x_bwd(h))
        np.testing.assert_array_equal(nb.n - nb.c, diff_y_fwd(h))
        np.testing.assert_array_equal(nb.c - nb.s, diff_y_bwd(h))


class TestMaterialize:
    def test_identity_values_and_gradient(self):
        h = _field(NY, NX)
        np.testing.assert_array_equal(_materialize(h), h)
        g = jax.grad(lambda x: jnp.sum(_materialize(x) ** 2))(h)
        np.testing.assert_allclose(g, 2 * h)

    def test_jit_and_vmap(self):
        h = _field(4, NY, NX)
        out = jax.jit(jax.vmap(_materialize))(h)
        np.testing.assert_array_equal(out, h)


class TestZeroZGhosts:
    @pytest.mark.parametrize("nz", [1, 2, 3, 6])
    def test_matches_scatter_reference(self, nz):
        arr = _field(nz, NY, NX)
        ref = arr.at[0].set(0.0).at[-1].set(0.0)
        out = zero_z_ghosts(arr)
        assert out.shape == arr.shape
        assert out.dtype == arr.dtype
        np.testing.assert_array_equal(out, ref)

    def test_float32_dtype_preserved(self):
        arr = jnp.ones((4, NY, NX), dtype=jnp.float32)
        assert zero_z_ghosts(arr).dtype == jnp.float32


# ---------------------------------------------------------------------------
# Values: each rewrite equals the original composed formula
# ---------------------------------------------------------------------------


class TestSameValues:
    def setup_method(self):
        self.g1 = fvx.CartesianGrid1D.from_interior(NX - 2, 3.0)
        self.g2 = fvx.CartesianGrid2D.from_interior(NX - 2, NY - 2, 3.0, 2.0)
        self.g3 = fvx.CartesianGrid3D.from_interior(NX - 2, NY - 2, 3, 3.0, 2.0, 1.0)

    def test_laplacian_1d(self):
        h = _field(NX)
        ref = interior((diff_x_fwd_1d(h) - diff_x_bwd_1d(h)) / self.g1.dx**2, h)
        np.testing.assert_array_equal(fvx.Difference1D(self.g1).laplacian(h), ref)

    def test_laplacian_2d(self):
        h = _field(NY, NX)
        d2x = (diff_x_fwd(h) - diff_x_bwd(h)) / self.g2.dx**2
        d2y = (diff_y_fwd(h) - diff_y_bwd(h)) / self.g2.dy**2
        ref = interior(d2x + d2y, h)
        np.testing.assert_array_equal(fvx.Difference2D(self.g2).laplacian(h), ref)

    @pytest.mark.slow
    def test_laplacian_3d(self):
        h = _field(5, NY, NX)
        d2x = (diff_x_fwd_3d(h) - diff_x_bwd_3d(h)) / self.g3.dx**2
        d2y = (diff_y_fwd_3d(h) - diff_y_bwd_3d(h)) / self.g3.dy**2
        ref = interior(d2x + d2y, h)
        np.testing.assert_array_equal(fvx.Difference3D(self.g3).laplacian(h), ref)

    def test_divergence_3d(self):
        u, v = _field(5, NY, NX, seed=1), _field(5, NY, NX, seed=2)
        op = fvx.Difference3D(self.g3)
        ref = op.diff_x_U_to_T(u) + op.diff_y_V_to_T(v)
        np.testing.assert_array_equal(op.divergence(u, v), ref)

    def test_arakawa_jacobian(self):
        f, g = _field(2, NY, NX, seed=3), _field(2, NY, NX, seed=4)
        # the original slice-per-term J++ / J+x / Jx+ forms
        jpp = (f[..., 1:-1, 2:] - f[..., 1:-1, :-2]) * (
            g[..., 2:, 1:-1] - g[..., :-2, 1:-1]
        ) - (f[..., 2:, 1:-1] - f[..., :-2, 1:-1]) * (
            g[..., 1:-1, 2:] - g[..., 1:-1, :-2]
        )
        jpx = (
            f[..., 1:-1, 2:] * (g[..., 2:, 2:] - g[..., :-2, 2:])
            - f[..., 1:-1, :-2] * (g[..., 2:, :-2] - g[..., :-2, :-2])
            - f[..., 2:, 1:-1] * (g[..., 2:, 2:] - g[..., 2:, :-2])
            + f[..., :-2, 1:-1] * (g[..., :-2, 2:] - g[..., :-2, :-2])
        )
        jxp = (
            g[..., 2:, 1:-1] * (f[..., 2:, 2:] - f[..., 2:, :-2])
            - g[..., :-2, 1:-1] * (f[..., :-2, 2:] - f[..., :-2, :-2])
            - g[..., 1:-1, 2:] * (f[..., 2:, 2:] - f[..., :-2, 2:])
            + g[..., 1:-1, :-2] * (f[..., 2:, :-2] - f[..., :-2, :-2])
        )
        ref = (jpp + jpx + jxp) / (12.0 * 0.5 * 0.25)
        np.testing.assert_array_equal(fvx.arakawa_jacobian(f, g, 0.5, 0.25), ref)

    @pytest.mark.parametrize("wall", ["closed", "open"])
    @pytest.mark.parametrize("kappa_field", [False, True])
    def test_diffusion_2d_matches_flux_divergence(self, wall, kappa_field):
        """The fused tendency equals the divergence of the public fluxes."""
        h = _field(NY, NX, seed=5)
        kappa = jnp.abs(_field(NY, NX, seed=6)) if kappa_field else 0.7
        dx, dy = self.g2.dx, self.g2.dy
        fx, fy = _diffusion_2d_fluxes_impl(h, kappa, dx, dy, None, None, wall=wall)
        du = (fx[1:-1, 1:-1] - fx[1:-1, :-2]) / dx
        dv = (fy[1:-1, 1:-1] - fy[:-2, 1:-1]) / dy
        ref = interior(du + dv, h)
        out = fvx.Diffusion2D(self.g2)(h, kappa, wall=wall)
        np.testing.assert_array_equal(out, ref)

    @pytest.mark.slow
    @pytest.mark.parametrize("wall", ["closed", "open"])
    @pytest.mark.parametrize("kappa_kind", ["field", "numpy_scalar"])
    def test_diffusion_2d_keeps_float32_with_float64_kappa(self, wall, kappa_kind):
        """The face fluxes stay h-typed, as the old ``zeros_like(h)`` buffers were."""
        h = _field(NY, NX, seed=5).astype(jnp.float32)
        if kappa_kind == "field":
            kappa = jnp.abs(_field(NY, NX, seed=6))  # float64
        else:
            kappa = np.float64(0.7)
        dx, dy = self.g2.dx, self.g2.dy
        # the old path wrote float64 fluxes into float32 buffers; JAX warns
        # about that implicit scatter cast, which is the behaviour we pin
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fx, fy = _diffusion_2d_fluxes_impl(h, kappa, dx, dy, None, None, wall=wall)
        du = (fx[1:-1, 1:-1] - fx[1:-1, :-2]) / dx
        dv = (fy[1:-1, 1:-1] - fy[:-2, 1:-1]) / dy
        ref = interior(du + dv, h)
        out = fvx.Diffusion2D(self.g2)(h, kappa, wall=wall)
        assert out.dtype == jnp.float32
        np.testing.assert_array_equal(out, ref)

    @pytest.mark.slow
    def test_spherical_diffusion_keeps_float32_with_float64_kappa(self):
        gs = fvx.SphericalGrid2D.from_interior(
            NX - 2, NY - 2, lon_range=(0.0, 30.0), lat_range=(-20.0, 20.0)
        )
        h = _field(NY, NX, seed=5).astype(jnp.float32)
        kappa = jnp.abs(_field(NY, NX, seed=6))  # float64 field
        assert fvx.SphericalDiffusion2D(gs)(h, kappa).dtype == jnp.float32

    def test_masked_diffusion_2d_matches_flux_divergence(self):
        h = _field(NY, NX, seed=7)
        wet = np.ones((NY, NX), dtype=bool)
        wet[3:5, 4:7] = False
        mask = fvx.Mask2D.from_mask(wet)
        dx, dy = self.g2.dx, self.g2.dy
        fx, fy = _diffusion_2d_fluxes_impl(h, 0.7, dx, dy, mask.u, mask.v)
        du = (fx[1:-1, 1:-1] - fx[1:-1, :-2]) / dx
        dv = (fy[1:-1, 1:-1] - fy[:-2, 1:-1]) / dy
        ref = interior(du + dv, h) * mask.h
        out = fvx.Diffusion2D(self.g2, mask=mask)(h, 0.7)
        np.testing.assert_array_equal(out, ref)

    def test_biharmonic_is_two_harmonic_passes(self):
        h = _field(NY, NX, seed=8)
        harm = fvx.Diffusion2D(self.g2)
        ref = -0.3 * harm(harm(h, 1.0), 1.0)
        out = fvx.BiharmonicDiffusion2D(self.g2)(h, 0.3)
        np.testing.assert_array_equal(out, ref)


# ---------------------------------------------------------------------------
# Lowering: one pad per distinct offset in the VJP
# ---------------------------------------------------------------------------


class TestOnePadPerOffset:
    def setup_method(self):
        self.g1 = fvx.CartesianGrid1D.from_interior(NX - 2, 3.0)
        self.g2 = fvx.CartesianGrid2D.from_interior(NX - 2, NY - 2, 3.0, 2.0)
        self.g3 = fvx.CartesianGrid3D.from_interior(NX - 2, NY - 2, 3, 3.0, 2.0, 1.0)
        self.gs = fvx.SphericalGrid2D.from_interior(
            NX - 2, NY - 2, lon_range=(0.0, 30.0), lat_range=(-20.0, 20.0)
        )

    def test_laplacian_1d(self):
        # h[i-1], h[i], h[i+1]
        op = fvx.Difference1D(self.g1)
        assert _vjp_pads(op.laplacian, _field(NX)) == 3

    def test_laplacian_2d(self):
        # c, e, w, n, s
        op = fvx.Difference2D(self.g2)
        assert _vjp_pads(op.laplacian, _field(NY, NX)) == 5

    def test_laplacian_3d(self):
        op = fvx.Difference3D(self.g3)
        # 5 horizontal offsets on the z-interior, plus the z-interior slice
        assert _vjp_pads(op.laplacian, _field(5, NY, NX)) == 6

    def test_spherical_laplacian(self):
        h = _field(NY, NX)
        assert _vjp_pads(fvx.SphericalLaplacian2D(self.gs), h) == 5

    def test_arakawa_jacobian(self):
        # eight off-centre neighbours of f and of g
        f, g = _field(NY, NX, seed=1), _field(NY, NX, seed=2)
        pads = _vjp_pads(lambda a, b: fvx.arakawa_jacobian(a, b, 1.0, 1.0), f, g)
        assert pads == 16

    def test_diffusion_2d(self):
        # h[i+1] - h[i] for x faces and h[j+1] - h[j] for y faces
        op = fvx.Diffusion2D(self.g2)
        assert _vjp_pads(lambda x: op(x, 1.0), _field(NY, NX)) == 4


# ---------------------------------------------------------------------------
# WENO-7 / WENO-9 smoothness indicators: unrolled quadratic form
# ---------------------------------------------------------------------------


def _einsum_smoothness_indicator(beta_matrix, stencil):
    """The original stacked-einsum form of ``q^T B q``."""
    values = jnp.stack(stencil, axis=0)
    beta = jnp.asarray(beta_matrix, dtype=values.dtype)
    return jnp.einsum("i...,ij,j...->...", values, beta, values)


class TestUnrolledSmoothnessIndicator:
    @pytest.mark.parametrize(
        "name, npts",
        [
            ("weno_7pts", 7),
            ("weno_7pts_right", 7),
            ("weno_9pts", 9),
            ("weno_9pts_right", 9),
        ],
    )
    def test_matches_einsum_form(self, name, npts, monkeypatch):
        from finitevolx._src.advection import weno

        qs = [_field(NY, NX, seed=k) * (1 + k) for k in range(npts)]
        fn = getattr(weno, name)
        out = fn(*qs)
        monkeypatch.setattr(weno, "_smoothness_indicator", _einsum_smoothness_indicator)
        ref = fn(*qs)
        # float64: only the summation order of q^T B q can differ
        np.testing.assert_allclose(out, ref, rtol=1e-13, atol=1e-13)

    def test_keeps_float32(self):
        from finitevolx._src.advection.weno import weno_7pts

        qs = [jnp.ones((NY, NX), dtype=jnp.float32) * k for k in range(7)]
        assert weno_7pts(*qs).dtype == jnp.float32

    def test_no_dot_general_in_weno7_advection(self):
        """The indicators lower to elementwise code, not a stack + dot."""
        g2 = fvx.CartesianGrid2D.from_interior(NX - 2, NY - 2, 3.0, 2.0)
        adv = fvx.Advection2D(g2)
        h, u, v = (_field(NY, NX, seed=k) for k in range(3))
        jaxpr = jax.make_jaxpr(lambda a, b, c: adv(a, b, c, method="weno7"))(h, u, v)
        assert "dot_general" not in _primitives(jaxpr.jaxpr)
