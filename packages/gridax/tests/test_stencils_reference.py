"""The 32 named C-grid stencils are the two-point staggered stencils of Layer 0 (#455).

Each named stencil ``<op>_<axis>_<side>_<d>d`` (and each corner average
``avg_x<side>_y<side>_<d>d``) is rebuilt from its name alone: the Layer 0
coefficients of the staggered two-point stencil, s = (-1/2, 1/2), with
c = (-1, 1) for ``diff`` and c = (1/2, 1/2) for ``avg``, applied along each
shifted axis, then the interior of every spatial axis. The one-axis stencils
must match bit for bit; the corner averages add four terms in another order
than the tensor product, so they agree within 4 eps max|f|.
"""

from __future__ import annotations

import re

import gridax.stencils as gs
import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # the gridax-numpy-only CI job
    jax = jnp = None

NAMED = sorted(
    name for name in gs.__all__ if re.fullmatch(r"(diff|avg)_[a-z_]+_[123]d", name)
)
CORNERS = [name for name in NAMED if re.search(r"x(fwd|bwd)_y(fwd|bwd)", name)]
AXES = {"x": -1, "y": -2, "z": -3}
SHAPES = {1: (13,), 2: (11, 13), 3: (9, 11, 13)}


def _parse(name: str) -> tuple[str, list[tuple[str, str]], int]:
    """``avg_xfwd_ybwd_2d`` → ("avg", [("x", "fwd"), ("y", "bwd")], 2)."""
    op, *middle, rank = name.split("_")
    shifts = re.findall(r"([xyz])_?(fwd|bwd)", "_".join(middle))
    return op, shifts, int(rank[0])


def _reference(name: str, field):
    """The named stencil rebuilt from the Layer 0 two-point staggered stencil."""
    op, shifts, ndim = _parse(name)
    derivative = 1 if op == "diff" else 0
    offsets = gs.difference_offsets(derivative=derivative, method="staggered")
    c_lo, c_hi = gs.difference_coeffs(offsets, derivative)
    out = field
    for axis_name, side in shifts:
        axis = AXES[axis_name]
        size = out.shape[axis]
        # forward: the face i + 1/2 reads i and i + 1; backward: i - 1/2 reads i - 1, i
        lo, hi = (1, 2) if side == "fwd" else (0, 1)
        lower = [slice(None)] * out.ndim
        upper = [slice(None)] * out.ndim
        lower[axis] = slice(lo, lo + size - 2)
        upper[axis] = slice(hi, hi + size - 2)
        out = c_lo * out[tuple(lower)] + c_hi * out[tuple(upper)]
    # the axes that were not shifted lose their ghost cells
    shifted = {AXES[axis_name] for axis_name, _ in shifts}
    crop = [slice(None)] * out.ndim
    for axis in range(-ndim, 0):
        if axis not in shifted:
            crop[axis] = slice(1, -1)
    return out[tuple(crop)]


def test_every_named_stencil_is_covered() -> None:
    # a new stencil without a reference fails here
    assert len(NAMED) == 32
    assert len(CORNERS) == 8


def _backends(dtype: str):
    yield np
    # float64 JAX arrays need x64; without it jnp.asarray would downcast
    if jnp is not None and (dtype == "float32" or jax.config.read("jax_enable_x64")):
        yield jnp


@pytest.mark.parametrize("name", NAMED)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("batch", [(2,), (2, 3)])
def test_named_stencils_equal_the_layer0_stencil(
    name: str, dtype: str, batch: tuple
) -> None:
    _, _, ndim = _parse(name)
    rng = np.random.default_rng(len(name) + len(batch))
    host = rng.standard_normal(batch + SHAPES[ndim]).astype(dtype)
    for xp in _backends(dtype):
        field = xp.asarray(host)
        got = np.asarray(getattr(gs, name)(field))
        want = np.asarray(_reference(name, field))
        assert got.shape == want.shape == batch + tuple(n - 2 for n in SHAPES[ndim])
        assert got.dtype == want.dtype == np.dtype(dtype)
        if name in CORNERS:
            tolerance = 4 * np.finfo(dtype).eps * np.abs(host).max()
            np.testing.assert_allclose(got, want, rtol=0, atol=tolerance)
        else:
            np.testing.assert_array_equal(got, want)
