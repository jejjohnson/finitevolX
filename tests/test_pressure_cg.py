"""The masked T-point operators put the pressure wall on the cell faces (#308).

``masked_neumann_laplacian`` has zero flux through every coast face and
``masked_face_dirichlet_laplacian`` holds p = 0 there, the walls of the
DCT-II and DST-II solves on the rectangle.  The pressure solve that uses them,
``pressure_from_divergence``, moved to ``sesmax_core.gfd`` with its tests
(#475, jejjohnson/sesmax#126).
"""

from __future__ import annotations

import gridax_jax as fvx
import jax.numpy as jnp
import numpy as np
import pytest

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
        # spectral DCT-II on the T interior, zero ghost ring
        p = (
            jnp.zeros_like(b)
            .at[1:-1, 1:-1]
            .set(fvx.solve_helmholtz_dct2(b[1:-1, 1:-1], dx, dy, 0.0))
        )
        lap = fvx.masked_neumann_laplacian(p, mask.h.astype(b.dtype), dx, dy)
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
                np.asarray(fvx.masked_neumann_laplacian(e, mf, dx, dy)).ravel()[idx]
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
        lap = fvx.masked_neumann_laplacian(p, mask.h.astype(p.dtype), g.dx, g.dy)
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
        # spectral DST-II on the T interior, zero ghost ring
        p = (
            jnp.zeros_like(b)
            .at[1:-1, 1:-1]
            .set(fvx.solve_helmholtz_dst2(b[1:-1, 1:-1], dx, dy, 0.0))
        )
        lap = fvx.masked_face_dirichlet_laplacian(p, mask.h.astype(b.dtype), dx, dy)
        # roundoff of a 34x34 transform round trip
        np.testing.assert_allclose(np.asarray(lap)[wet], np.asarray(b)[wet], atol=1e-11)
