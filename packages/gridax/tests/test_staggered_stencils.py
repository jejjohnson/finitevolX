"""Staggered differences and averages of any even width on the C-grid (#457).

``accuracy=1`` is the named two-point stencil, bit for bit. Wider stencils
are centred where they fit and shift inwards at the walls, so the observed
order is K away from the walls and K - n over every face (K points, n the
derivative order), and every face is exact on polynomials of degree below K.
"""

from __future__ import annotations

import itertools

import gridax.stencils as gs
import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # the gridax-numpy-only CI job
    jax = jnp = None

AXES = {"x": -1, "y": -2, "z": -3}
CASES = [
    (op, axis, side, ndim)
    for op, side, ndim in itertools.product(("diff", "avg"), ("fwd", "bwd"), (1, 2, 3))
    for axis in ("x", "y", "z")[:ndim]
]
FUNCTIONS = {"diff": gs.staggered_difference, "avg": gs.staggered_average}


@pytest.mark.parametrize(("op", "axis", "side", "ndim"), CASES)
def test_accuracy_one_is_the_named_stencil(op, axis, side, ndim) -> None:
    h = np.random.default_rng(ndim).standard_normal((2, *(9, 11, 13)[3 - ndim :]))
    got = FUNCTIONS[op](h, axis=axis, side=side, ndim=ndim)
    np.testing.assert_array_equal(got, getattr(gs, f"{op}_{axis}_{side}_{ndim}d")(h))


def _line(n_cells: int, axis: str, ndim: int, profile):
    """``profile`` sampled on cell centres x_i = (i - 1/2) dx of [0, 1] along ``axis``."""
    dx = 1.0 / (n_cells - 2)
    x = (np.arange(n_cells) - 0.5) * dx
    shape = [5] * ndim
    shape[AXES[axis]] = n_cells
    values = np.broadcast_to(
        profile(x).reshape([-1 if d == ndim + AXES[axis] else 1 for d in range(ndim)]),
        shape,
    )
    return np.array(values), x, dx


def _faces(x, side: str):
    # the faces the output writes: i + 1/2 (fwd) or i - 1/2 (bwd), i = 1 .. N-2
    dx = x[1] - x[0]
    return x[1:-1] + (0.5 if side == "fwd" else -0.5) * dx


def _errors(op, axis, side, ndim, accuracy, n_cells, frequency):
    """Max error away from the walls and over every face, on sin(w x + 0.7)."""

    def profile(x):
        return np.sin(frequency * x + 0.7)

    h, x, dx = _line(n_cells, axis, ndim, profile)
    out = FUNCTIONS[op](h, axis=axis, side=side, ndim=ndim, accuracy=accuracy)
    faces = _faces(x, side)
    if op == "diff":
        exact = frequency * np.cos(frequency * faces + 0.7)
    else:
        exact = np.sin(frequency * faces + 0.7)
    line = np.moveaxis(np.asarray(out), AXES[axis], -1)[(0,) * (ndim - 1)]
    if op == "diff":
        line = line / dx
    err = np.abs(line - exact)
    return err[6:-6].max(), err.max()  # faces 6 from a wall use the centred stencil


def _order(op, axis, side, ndim, accuracy, cells, frequency, which):
    coarse = _errors(op, axis, side, ndim, accuracy, cells[0], frequency)[which]
    fine = _errors(op, axis, side, ndim, accuracy, cells[1], frequency)[which]
    return np.log2(coarse / fine)


ORDER_CASES = [c for c in CASES if c[3] < 3 or c[1] == "z"]


@pytest.mark.parametrize("accuracy", [3, 5])
@pytest.mark.parametrize(("op", "axis", "side", "ndim"), ORDER_CASES)
def test_order_k_away_from_the_walls(op, axis, side, ndim, accuracy) -> None:
    # sin(6x + 0.7) on 40 and 80 cells: asymptotic, and far above roundoff
    n = 1 if op == "diff" else 0
    k = len(gs.difference_offsets(derivative=n, accuracy=accuracy, method="staggered"))
    order = _order(op, axis, side, ndim, accuracy, (42, 82), 6.0, which=0)
    assert order == pytest.approx(k, abs=0.1)


@pytest.mark.parametrize("accuracy", [3, 5])
@pytest.mark.parametrize(("op", "axis", "side", "ndim"), ORDER_CASES)
def test_order_k_minus_n_over_every_face(op, axis, side, ndim, accuracy) -> None:
    # the shifted wall stencil has order K - n and a large error constant, so
    # this is measured on sin(2x + 0.7), on 80/160 cells (K = 4) or 32/64 (K = 6)
    n = 1 if op == "diff" else 0
    k = len(gs.difference_offsets(derivative=n, accuracy=accuracy, method="staggered"))
    cells = (82, 162) if accuracy == 3 else (34, 66)
    order = _order(op, axis, side, ndim, accuracy, cells, 2.0, which=1)
    assert order == pytest.approx(k - n, abs=0.25)


@pytest.mark.parametrize("accuracy", [3, 5])
@pytest.mark.parametrize(
    ("op", "axis", "side", "ndim"), [c for c in CASES if c[3] == 2]
)
def test_polynomials_are_exact_on_every_face(op, axis, side, ndim, accuracy) -> None:
    n = 1 if op == "diff" else 0
    points = len(
        gs.difference_offsets(derivative=n, accuracy=accuracy, method="staggered")
    )
    for degree in range(points):
        h, x, dx = _line(14, axis, ndim, lambda x, d=degree: x**d)
        out = FUNCTIONS[op](h, axis=axis, side=side, ndim=ndim, accuracy=accuracy)
        faces = _faces(x, side)
        exact = (
            degree * faces ** max(degree - 1, 0) * dx if op == "diff" else faces**degree
        )
        line = np.moveaxis(np.asarray(out), AXES[axis], -1)[0]
        np.testing.assert_allclose(line, exact, rtol=1e-10, atol=1e-12)


def test_leading_batch_axes_pass_through() -> None:
    h = np.random.default_rng(5).standard_normal((3, 10, 12))
    whole = gs.staggered_average(h, axis="y", side="fwd", ndim=2, accuracy=3)
    each = np.stack(
        [gs.staggered_average(m, axis="y", side="fwd", ndim=2, accuracy=3) for m in h]
    )
    np.testing.assert_array_equal(whole, each)


def test_bad_arguments_raise() -> None:
    h = np.zeros((6, 6))
    with pytest.raises(ValueError, match="spatial axes"):
        gs.staggered_difference(h, axis="z", side="fwd", ndim=2, accuracy=3)
    with pytest.raises(ValueError, match="side"):
        gs.staggered_difference(h, axis="x", side="up", ndim=2, accuracy=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shorter"):
        gs.staggered_difference(np.zeros(5), axis="x", side="fwd", ndim=1, accuracy=5)


@pytest.mark.skipif(jax is None, reason="JAX is not installed")
def test_jit_vmap_and_grad() -> None:
    h = jnp.asarray(
        np.random.default_rng(6).standard_normal((4, 10, 12)), dtype=jnp.float32
    )

    def fourth(field):
        return gs.staggered_difference(field, axis="x", side="bwd", ndim=2, accuracy=3)

    eager = np.asarray(fourth(h))
    # XLA may fuse the multiply-adds under jit, so agree to float32 rounding
    np.testing.assert_allclose(
        np.asarray(jax.jit(fourth)(h)), eager, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(np.asarray(jax.vmap(fourth)(h)), eager, rtol=1e-6)
    # the operator is linear, so the gradient of a weighted sum is its adjoint
    weights = jnp.asarray(
        np.random.default_rng(7).standard_normal(eager.shape), dtype=h.dtype
    )
    grad = jax.grad(lambda f: jnp.sum(weights * fourth(f)))(h)
    probe = jnp.asarray(
        np.random.default_rng(8).standard_normal(h.shape), dtype=h.dtype
    )
    lhs = float(jnp.sum(grad * probe))
    rhs = float(jnp.sum(weights * fourth(probe)))
    assert lhs == pytest.approx(rhs, rel=1e-4)
