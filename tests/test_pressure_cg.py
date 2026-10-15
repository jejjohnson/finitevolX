"""The masked pressure solve honours its boundary condition (#308) and the
wrappers accept a traced scalar Helmholtz parameter (#309).

``pressure_from_divergence(method="cg")`` used to build its operator from
``masked_laplacian`` (p = 0 on the coast) whatever ``bc`` said, so the
documented Neumann wall was never imposed, and ``bc`` was silently ignored
by the mask-based methods.  ``pv_inversion`` called ``float(lambda_)``,
which fails on a tracer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import finitevolx as fvx
from finitevolx._src.solvers.iterative import (
    _masked_face_dirichlet_laplacian,
    _masked_neumann_laplacian,
)

N = 34


def _grid():
    g = fvx.CartesianGrid2D.from_interior(N - 2, N - 2, Lx=1.0, Ly=1.3)
    return g.dx, g.dy


def _rect_mask():
    m = np.zeros((N, N), dtype=bool)
    m[1:-1, 1:-1] = True
    return fvx.Mask2D.from_mask(m)


def _island_basin():
    """Elliptic basin with a 4x4 island, dry ghost ring: one wet component."""
    yy, xx = np.mgrid[0:N, 0:N]
    m = (xx - N / 2) ** 2 / (0.42 * N) ** 2 + (yy - N / 2) ** 2 / (0.4 * N) ** 2 < 1
    m[14:18, 15:19] = False
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
    return fvx.Mask2D.from_mask(m)


def _rhs(seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((N, N)))


def _demeaned(field, wet):
    vals = np.asarray(field)[wet]
    return vals - vals.mean()


@pytest.mark.slow  # jit-compile bound (> 1 s each)
class TestNeumannOperator:
    def test_matches_dct2_on_the_rectangle(self):
        """On the rectangle's T mask the operator is the DCT-II one exactly."""
        dx, dy = _grid()
        mask = _rect_mask()
        wet = np.asarray(mask.h) > 0.5
        b = _rhs()
        p = fvx.pressure_from_divergence(b, dx, dy, bc="dct")  # spectral DCT-II
        lap = _masked_neumann_laplacian(p, mask.h.astype(b.dtype), dx, dy)
        b_proj = _demeaned(b, wet)  # the DCT solve drops the mean of b
        # roundoff of a 34x34 transform round trip
        np.testing.assert_allclose(np.asarray(lap)[wet], b_proj, atol=1e-12)

    def test_symmetric_negative_semidefinite_single_null_vector(self):
        dx, dy = _grid()
        mask = _island_basin()
        mf = mask.h.astype(jnp.float64)
        idx = np.flatnonzero(np.asarray(mask.h).ravel())
        cols = []
        for k in idx:
            e = jnp.zeros(N * N).at[k].set(1.0).reshape(N, N)
            cols.append(
                np.asarray(_masked_neumann_laplacian(e, mf, dx, dy)).ravel()[idx]
            )
        a = np.stack(cols, axis=1)
        np.testing.assert_array_equal(a, a.T)
        eig = np.linalg.eigvalsh(a)
        assert eig[-1] < 1e-10  # semidefinite (roundoff of a ~1e3-norm matrix)
        assert eig[-2] < -1.0  # one connected component: a single zero mode

    def test_matches_diffusion2d_closed_walls(self):
        """The third copy of the zero-flux operator: Diffusion2D with kappa = 1."""
        mask = _island_basin()
        g = fvx.CartesianGrid2D.from_interior(N - 2, N - 2, Lx=1.0, Ly=1.3)
        p = _rhs(9) * mask.h
        lap = _masked_neumann_laplacian(p, mask.h.astype(p.dtype), g.dx, g.dy)
        diff = fvx.Diffusion2D(g, mask=mask)(p, 1.0)
        wet = np.asarray(mask.h) > 0.5
        np.testing.assert_allclose(
            np.asarray(lap)[wet], np.asarray(diff)[wet], atol=1e-10
        )

    def test_face_dirichlet_matches_dst2_on_the_rectangle(self):
        """p = 0 on the faces: the T-point DST-II solve satisfies it exactly."""
        dx, dy = _grid()
        mask = _rect_mask()
        wet = np.asarray(mask.h) > 0.5
        b = _rhs(10)
        p = fvx.pressure_from_divergence(b, dx, dy, bc="dst")  # spectral DST-II
        lap = _masked_face_dirichlet_laplacian(p, mask.h.astype(b.dtype), dx, dy)
        # roundoff of a 34x34 transform round trip
        np.testing.assert_allclose(np.asarray(lap)[wet], np.asarray(b)[wet], atol=1e-11)


@pytest.mark.slow  # jit-compile bound (> 1 s each)
class TestPressureCG:
    def test_agrees_with_spectral_dct_on_rectangle(self):
        dx, dy = _grid()
        mask = _rect_mask()
        wet = np.asarray(mask.h) > 0.5
        b = _rhs()
        p_sp = fvx.pressure_from_divergence(b, dx, dy, bc="dct")
        p_cg = fvx.pressure_from_divergence(b, dx, dy, method="cg", mask=mask)
        # CG stops at rtol 1e-6 of |b|; |p| ~ 1e-2 here, so 1e-8 is that tolerance
        np.testing.assert_allclose(
            _demeaned(p_cg, wet), _demeaned(p_sp, wet), atol=1e-8
        )

    def test_neumann_on_masked_basin(self):
        """Zero flux through every coast face: the solution satisfies the
        Neumann operator with the (projected) rhs and has zero wet-mean."""
        dx, dy = _grid()
        mask = _island_basin()
        wet = np.asarray(mask.h) > 0.5
        b = _rhs(1)
        p = fvx.pressure_from_divergence(b, dx, dy, method="cg", mask=mask)
        assert abs(np.asarray(p)[wet].mean()) < 1e-14
        np.testing.assert_array_equal(np.asarray(p)[~wet], 0.0)
        lap = _masked_neumann_laplacian(p, mask.h.astype(b.dtype), dx, dy)
        # residual of the CG solve: rtol 1e-6 of |b| ~ 3
        np.testing.assert_allclose(np.asarray(lap)[wet], _demeaned(b, wet), atol=1e-5)

    def test_two_components_each_zero_mean(self):
        """Disconnected wet regions each fix p up to their own constant."""
        dx, dy = _grid()
        m = np.zeros((N, N), dtype=bool)
        m[1:-1, 1:-1] = True
        m[:, N // 2] = False  # a dry wall splits the basin in two
        mask = fvx.Mask2D.from_mask(m)
        wet = np.asarray(mask.h) > 0.5
        p = fvx.pressure_from_divergence(_rhs(11), dx, dy, method="cg", mask=mask)
        left, right = np.asarray(p)[:, : N // 2], np.asarray(p)[:, N // 2 + 1 :]
        assert abs(left[wet[:, : N // 2]].mean()) < 1e-14
        assert abs(right[wet[:, N // 2 + 1 :]].mean()) < 1e-14

    def test_dirichlet_agrees_with_spectral_dst_on_rectangle(self):
        """bc="dst": CG puts p = 0 on the faces, where DST-II does."""
        dx, dy = _grid()
        mask = _rect_mask()
        b = _rhs(12)
        p_sp = fvx.pressure_from_divergence(b, dx, dy, bc="dst")
        p_cg = fvx.pressure_from_divergence(b, dx, dy, bc="dst", method="cg", mask=mask)
        # CG rtol 1e-6 of |b|; |p| ~ 1e-2 here
        np.testing.assert_allclose(np.asarray(p_cg), np.asarray(p_sp), atol=1e-8)

    def test_dirichlet_is_second_order(self):
        """Manufactured p = sin(pi x) sin(pi y), zero on the faces x, y in {0, 1}."""
        errs = []
        for n in (16, 32):
            g = fvx.CartesianGrid2D.from_interior(n, n, Lx=1.0, Ly=1.0)
            x = (np.arange(n + 2) - 0.5) * g.dx  # T centres; faces at 0 and 1
            xx, yy = np.meshgrid(x, x)
            exact = np.sin(np.pi * xx) * np.sin(np.pi * yy)
            m = np.zeros((n + 2, n + 2), dtype=bool)
            m[1:-1, 1:-1] = True
            p = fvx.pressure_from_divergence(
                jnp.asarray(-2 * np.pi**2 * exact),
                g.dx,
                g.dx,
                bc="dst",
                method="cg",
                mask=fvx.Mask2D.from_mask(m),
            )
            errs.append(np.abs(np.asarray(p) - exact)[m].max())
        assert np.log2(errs[0] / errs[1]) > 1.9

    def test_differs_from_the_dirichlet_solve(self):
        """bc decides the operator: Neumann and Dirichlet give different p."""
        dx, dy = _grid()
        mask = _island_basin()
        b = _rhs(2)
        p_n = fvx.pressure_from_divergence(b, dx, dy, method="cg", mask=mask)
        p_d = fvx.pressure_from_divergence(b, dx, dy, bc="dst", method="cg", mask=mask)
        assert float(jnp.max(jnp.abs(p_n - p_d))) > 1e-3


class TestBcIsHonouredOrRejected:
    def test_cg_rejects_periodic(self):
        dx, dy = _grid()
        with pytest.raises(ValueError, match="periodic 'fft'"):
            fvx.pressure_from_divergence(
                _rhs(), dx, dy, bc="fft", method="cg", mask=_rect_mask()
            )

    def test_cg_streamfunction_rejects_neumann(self):
        dx, dy = _grid()
        with pytest.raises(ValueError, match="only defined for the T-point"):
            fvx.streamfunction_from_vorticity(
                _rhs(), dx, dy, bc="dct", method="cg", mask=_rect_mask()
            )

    def test_pressure_rejects_capacitance(self):
        """Its wall is the psi wall (dry cell centres), not the T faces."""
        dx, dy = _grid()
        with pytest.raises(ValueError, match="not available for the T-point"):
            fvx.pressure_from_divergence(_rhs(), dx, dy, bc="dst", method="capacitance")

    def test_streamfunction_capacitance_rejects_neumann(self):
        dx, dy = _grid()
        with pytest.raises(ValueError, match="needs bc='dst'"):
            fvx.streamfunction_from_vorticity(
                _rhs(), dx, dy, bc="dct", method="capacitance"
            )

    def test_pressure_rejects_known_values(self):
        dx, dy = _grid()
        with pytest.raises(ValueError, match="T-point pressure"):
            fvx.pressure_from_divergence(
                _rhs(),
                dx,
                dy,
                bc="dst",
                method="cg",
                mask=_rect_mask(),
                known_values=_rhs(3),
            )
        with pytest.raises(ValueError, match="known_mask needs known_values"):
            fvx.pressure_from_divergence(
                _rhs(),
                dx,
                dy,
                method="cg",
                mask=_rect_mask(),
                known_mask=np.zeros((N, N), dtype=bool),
            )

    def test_streamfunction_known_values_reject_neumann(self):
        dx, dy = _grid()
        with pytest.raises(ValueError, match="known_values prescribe Dirichlet"):
            fvx.streamfunction_from_vorticity(
                _rhs(),
                dx,
                dy,
                bc="dct",
                method="cg",
                mask=_rect_mask(),
                known_values=_rhs(3),
            )


@pytest.mark.slow  # jit-compile bound (> 1 s each)
class TestTracedScalarLambda:
    """#309: a traced scalar lambda_ works and changes nothing."""

    @pytest.mark.parametrize("method", ["spectral", "cg"])
    def test_jit(self, method):
        dx, dy = _grid()
        kw = {} if method == "spectral" else {"mask": _island_basin()}
        rhs = jnp.stack([_rhs(4), _rhs(5)])
        eager = fvx.pv_inversion(rhs, dx, dy, lambda_=4.0, method=method, **kw)
        jitted = jax.jit(
            lambda r, lam: fvx.pv_inversion(r, dx, dy, lambda_=lam, method=method, **kw)
        )(rhs, 4.0)
        np.testing.assert_array_equal(np.asarray(jitted), np.asarray(eager))

    def test_vmap(self):
        dx, dy = _grid()
        rhs = _rhs(6)
        lams = jnp.array([1.0, 4.0])
        out = jax.vmap(lambda lam: fvx.pv_inversion(rhs, dx, dy, lambda_=lam))(lams)
        for k, lam in enumerate([1.0, 4.0]):
            np.testing.assert_array_equal(
                np.asarray(out[k]),
                np.asarray(fvx.pv_inversion(rhs, dx, dy, lambda_=lam)),
            )

    def test_scalar_array_and_python_float_agree(self):
        dx, dy = _grid()
        rhs = jnp.stack([_rhs(7), _rhs(8)])
        a = fvx.pv_inversion(rhs, dx, dy, lambda_=4.0)
        b = fvx.pv_inversion(rhs, dx, dy, lambda_=jnp.asarray(4.0))
        c = fvx.pv_inversion(rhs, dx, dy, lambda_=jnp.array([4.0, 4.0]))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        np.testing.assert_allclose(np.asarray(a), np.asarray(c), rtol=1e-13, atol=1e-16)
