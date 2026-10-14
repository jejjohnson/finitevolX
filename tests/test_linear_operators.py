"""Tests for the Helmholtz operator as a gaussx linear operator.

``helmholtz_operator`` must be the operator whose inverse the spectral
solvers apply (same transform, same FD2 eigenvalues), and
``masked_helmholtz_operator`` the wet sub-block that ``masked_laplacian``
applies; gaussx's structured primitives then hold exactly.
"""

from __future__ import annotations

import gaussx
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from finitevolx import (
    Mask2D,
    dctn,
    helmholtz_operator,
    masked_helmholtz_operator,
    masked_laplacian,
    solve_helmholtz_dct,
    solve_helmholtz_dst,
    solve_helmholtz_fft,
)

NY, NX = 12, 16
DX, DY = 0.1, 0.25  # anisotropic on purpose
LAM = 2.0

_SOLVERS = {
    "fft": solve_helmholtz_fft,
    "dst": solve_helmholtz_dst,
    "dct": solve_helmholtz_dct,
}
# Padding that realises each rectangle BC for the 5-point stencil:
#   fft: periodic wrap; dst (DST-I): ψ = 0 one cell outside;
#   dct (DCT-II): ψ_ghost = ψ_edge (zero gradient at the half-cell face).
_PAD = {"fft": "wrap", "dst": "constant", "dct": "edge"}


def _field(seed: int, shape=(NY, NX)):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


def _stencil(psi, bc, dx=DX, dy=DY, lam=LAM):
    """Reference (∇² − λ)ψ with the rectangle BC applied by padding."""
    p = jnp.pad(psi, 1, mode=_PAD[bc])  # [Ny+2, Nx+2]
    # (ψ[j, i+1] - 2ψ[j, i] + ψ[j, i-1]) / dx² + (… along y) / dy² - λ ψ[j, i]
    lap = (p[1:-1, 2:] - 2 * p[1:-1, 1:-1] + p[1:-1, :-2]) / dx**2 + (
        p[2:, 1:-1] - 2 * p[1:-1, 1:-1] + p[:-2, 1:-1]
    ) / dy**2
    return lap - lam * psi


class TestHelmholtzOperator:
    @pytest.mark.slow
    @pytest.mark.parametrize("bc", ["fft", "dst", "dct"])
    def test_solve_matches_spectral_solver(self, bc):
        A = helmholtz_operator((NY, NX), DX, DY, lambda_=LAM, bc=bc)
        f = _field(0)
        psi = gaussx.solve(A, f.ravel()).reshape(NY, NX)
        ref = _SOLVERS[bc](f, DX, DY, LAM)
        np.testing.assert_allclose(np.asarray(psi), np.asarray(ref), atol=1e-12)

    @pytest.mark.parametrize("bc", ["fft", "dst", "dct"])
    def test_mv_is_the_five_point_stencil(self, bc):
        A = helmholtz_operator((NY, NX), DX, DY, lambda_=LAM, bc=bc)
        psi = _field(1)
        out = A.mv(psi.ravel()).reshape(NY, NX)
        np.testing.assert_allclose(
            np.asarray(out), np.asarray(_stencil(psi, bc)), rtol=1e-10, atol=1e-10
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("bc", ["fft", "dst", "dct"])
    def test_logdet_matches_dense(self, bc):
        A = helmholtz_operator((NY, NX), DX, DY, lambda_=LAM, bc=bc)
        neg_A = -A  # κ² - ∇² with κ² = λ: positive definite
        _sign, ref = jnp.linalg.slogdet(neg_A.as_matrix())
        np.testing.assert_allclose(float(gaussx.logdet(neg_A)), float(ref), rtol=1e-12)

    @pytest.mark.parametrize("bc", ["fft", "dct"])
    def test_poisson_null_mode_is_the_spectral_gauge(self, bc):
        """λ = 0 with periodic / Neumann walls is singular (Λ[0, 0] = 0):
        gaussx's solve is the pseudo-inverse, i.e. the zero-mean gauge of the
        spectral solvers, finite, and logdet is -inf (codex review on #341)."""
        A = helmholtz_operator((NY, NX), DX, DY, lambda_=0.0, bc=bc)
        assert int((A.eigenvalues == 0).sum()) == 1
        f = _field(6)
        psi = gaussx.solve(A, f.ravel()).reshape(NY, NX)
        ref = _SOLVERS[bc](f, DX, DY, 0.0)
        assert bool(jnp.all(jnp.isfinite(psi)))
        np.testing.assert_allclose(np.asarray(psi), np.asarray(ref), atol=1e-12)
        assert bool(jnp.all(jnp.isfinite(gaussx.inv(A).eigenvalues)))
        assert float(gaussx.logdet(-A)) == -np.inf

    def test_tags(self):
        A = helmholtz_operator((NY, NX), DX, DY, lambda_=LAM)
        assert lx.is_symmetric(A)
        assert lx.is_negative_semidefinite(A)

    def test_traced_lambda_is_not_tagged(self):
        A = helmholtz_operator((NY, NX), DX, DY, lambda_=jnp.asarray(LAM))
        assert not lx.is_negative_semidefinite(A)

    @pytest.mark.slow
    @pytest.mark.parametrize("bc", ["fft", "dst", "dct"])
    def test_float32_in_float32_out(self, bc):
        A = helmholtz_operator((NY, NX), DX, DY, lambda_=LAM, bc=bc)
        f = _field(5).astype(jnp.float32).ravel()
        assert A.mv(f).dtype == jnp.float32
        x = gaussx.solve(A, f)
        assert x.dtype == jnp.float32
        ref = _SOLVERS[bc](f.reshape(NY, NX), DX, DY, LAM)
        np.testing.assert_allclose(
            np.asarray(x.reshape(NY, NX)), np.asarray(ref), rtol=1e-4, atol=1e-5
        )

    def test_invalid_bc(self):
        with pytest.raises(ValueError, match="bc must be"):
            helmholtz_operator((NY, NX), DX, DY, bc="robin")

    @pytest.mark.slow
    def test_spde_precision_marginal_variance(self):
        """Q = (κ² − ∇²)² built from the operator's eigenvalues gives the
        exact prior marginal variance diag(Q⁻¹), and sqrt(Q)² = Q
        (DCT/Neumann). gaussx.diag_inv densifies a DiagonalizedOperator, so
        the variance is checked through the separable 1-D bases the SPDE
        notebook uses, against the dense inverse on a small grid."""
        kappa2 = 4.0
        K = -helmholtz_operator((NY, NX), DX, DY, lambda_=kappa2, bc="dct")
        Q = gaussx.DiagonalizedOperator(
            K.eigenvalues**2, K.forward, K.inverse, K.in_shape, normal=True
        )
        Q_dense = Q.as_matrix()
        # diag(Vᵀ Λ⁻¹ V)[j, i] = Σ_kl Vy[k, j]² Vx[l, i]² / Λ[k, l]
        vy = dctn(jnp.eye(NY), type=2, axes=[0], norm="ortho")  # [k, j]
        vx = dctn(jnp.eye(NX), type=2, axes=[0], norm="ortho")  # [l, i]
        var = (vy**2).T @ (1.0 / Q.eigenvalues) @ (vx**2)  # [Ny, Nx]
        np.testing.assert_allclose(
            np.asarray(var.ravel()),
            np.asarray(jnp.diag(jnp.linalg.inv(Q_dense))),
            rtol=1e-8,
        )
        S = gaussx.sqrt(Q).as_matrix()
        np.testing.assert_allclose(
            np.asarray(S @ S), np.asarray(Q_dense), rtol=1e-8, atol=1e-6
        )


def _basin(shape=(NY, NX)):
    m = np.zeros(shape, dtype=bool)
    m[2:-2, 1:-1] = True
    m[5:7, 6:10] = False  # an island
    return m


class TestMaskedHelmholtzOperator:
    @pytest.mark.slow
    def test_mv_matches_masked_laplacian(self):
        m = _basin()
        A_m = masked_helmholtz_operator(m, DX, DY, lambda_=LAM)
        wet = m.ravel()
        psi = _field(2) * m
        ref = masked_laplacian(psi, jnp.asarray(m, float), DX, DY, lambda_=LAM)
        np.testing.assert_allclose(
            np.asarray(A_m.mv(psi.ravel()[wet])),
            np.asarray(ref.ravel()[wet]),
            rtol=1e-12,
            atol=1e-10,
        )

    def test_capacitance_solve_inverts(self):
        m = _basin()
        A_m = masked_helmholtz_operator(m, DX, DY, lambda_=LAM)
        assert A_m.capacitance is not None  # the fast path, not the dense one
        x = _field(3).ravel()[m.ravel()]
        y = gaussx.solve(A_m, A_m.mv(x))
        np.testing.assert_allclose(np.asarray(y), np.asarray(x), atol=1e-11)

    @pytest.mark.slow
    @pytest.mark.parametrize("edge", [False, True])
    def test_poisson_capacitance_over_singular_base(self, edge):
        """λ = 0: the periodic FFT base is singular, but gaussx borders the
        capacitance system with its null vector, so the masked Poisson solve
        is exact — for a closed basin and for a mask that wraps the periodic
        edge (codex review on #341)."""
        if edge:
            m = np.ones((NY, NX), dtype=bool)
            m[4:7, 3:9] = False
        else:
            m = _basin()
        A_m = masked_helmholtz_operator(m, DX, DY, lambda_=0.0)
        x = _field(7).ravel()[m.ravel()]
        b = A_m.mv(x)
        y = gaussx.solve(A_m, b)
        np.testing.assert_allclose(np.asarray(y), np.asarray(x), atol=1e-11)
        ref = jnp.linalg.solve(A_m.as_matrix(), b)
        np.testing.assert_allclose(np.asarray(y), np.asarray(ref), atol=1e-11)

    def test_mask_touching_the_edge_wraps(self):
        """masked_laplacian wraps at the rectangle edge; so must the operator."""
        m = np.ones((NY, NX), dtype=bool)
        m[4:7, 3:9] = False
        A_m = masked_helmholtz_operator(m, DX, DY, lambda_=LAM)
        psi = _field(4) * m
        ref = masked_laplacian(psi, jnp.asarray(m, float), DX, DY, lambda_=LAM)
        wet = m.ravel()
        np.testing.assert_allclose(
            np.asarray(A_m.mv(psi.ravel()[wet])),
            np.asarray(ref.ravel()[wet]),
            rtol=1e-12,
            atol=1e-10,
        )
        x = psi.ravel()[wet]
        np.testing.assert_allclose(
            np.asarray(gaussx.solve(A_m, A_m.mv(x))), np.asarray(x), atol=1e-10
        )

    def test_mask2d_uses_psi_points(self):
        h = np.zeros((NY, NX), dtype=bool)
        h[1:-1, 1:-1] = True
        mask = Mask2D.from_mask(h)
        A_m = masked_helmholtz_operator(mask, DX, DY)
        assert A_m.in_size() == int(np.asarray(mask.xy_corner_strict).sum())

    def test_tags(self):
        A_m = masked_helmholtz_operator(_basin(), DX, DY, lambda_=LAM)
        assert lx.is_symmetric(A_m)
        assert lx.is_negative_semidefinite(A_m)
