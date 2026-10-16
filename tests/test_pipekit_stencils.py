"""Stencil, interpolation, boundary and mask operators of `finitevolx.pipekit`."""

from __future__ import annotations

import pytest

pytest.importorskip("pipekit_jax")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pipekit import Graph, Input

import finitevolx as fvx
from finitevolx import pipekit as fp
from finitevolx.pipekit.testing import check_operator
from tests.fixtures.inputs import make_grid_2d, make_mask_2d

GRID = make_grid_2d()
SPEC = fp.as_grid_spec(GRID)
MASK = make_mask_2d()
D = fvx.Difference2D(grid=GRID)
I = fvx.Interpolation2D(grid=GRID)


def _field(seed: int) -> jax.Array:
    return jax.random.normal(jax.random.key(seed), (GRID.Ny, GRID.Nx))


def _f_beta(f0: float, beta: float) -> jax.Array:
    y = jnp.arange(GRID.Ny) * GRID.dy
    Y = jnp.broadcast_to(y[:, None], (GRID.Ny, GRID.Nx))
    return fvx.coriolis_fn(Y, f0=f0, beta=beta)


# (id, operator, Tier-A reference, number of input fields)
CASES = [
    *[
        (
            f"Difference-{a}-{s}{d}",
            fp.Difference(grid=SPEC, axis=a, src=s, dst=d),
            getattr(D, m),
            1,
        )
        for (a, s, d), m in sorted(fp.DIFFERENCES.items())
    ],
    *[
        (f"Interp-{s}{d}", fp.Interp(grid=SPEC, src=s, dst=d), getattr(I, m), 1)
        for (s, d), m in sorted(fp.INTERPOLATIONS.items())
    ],
    ("Divergence", fp.Divergence(grid=SPEC), D.divergence, 2),
    ("Curl", fp.Curl(grid=SPEC), D.curl, 2),
    ("Laplacian", fp.Laplacian(grid=SPEC), D.laplacian, 1),
    ("GradPerp", fp.GradPerp(grid=SPEC), D.grad_perp, 1),
    (
        "Coriolis",
        fp.Coriolis(grid=SPEC, f0=2.0, beta=0.3),
        lambda u, v: fvx.Coriolis2D(grid=GRID)(u, v, _f_beta(2.0, 0.3)),
        2,
    ),
    *[
        (f"FillGhosts-{k}", fp.FillGhosts(kind=k), f, 1)
        for k, f in sorted(fp.GHOST_FILLERS.items())
        if k != "wall"
    ],
    *[
        (
            f"FillGhosts-wall-{s}",
            fp.FillGhosts(kind="wall", staggering=s),
            lambda x, s=s: fvx.wall_boundaries(x, staggering=s),
            1,
        )
        for s in "huvq"
    ],
    *[
        (
            f"ApplyMask-{s}",
            fp.ApplyMask(mask=MASK, stagger=s),
            lambda x, a=a: jnp.where(getattr(MASK, a), x, 0.0),
            1,
        )
        for s, a in fp.MASK_ATTRIBUTES.items()
    ],
    (
        "LinComb",
        fp.LinComb(coeffs=(2.0, -0.5)),
        lambda x, y: 2.0 * x - 0.5 * y,
        2,
    ),
]
IDS = [c[0] for c in CASES]


@pytest.mark.parametrize(("name", "op", "ref", "n"), CASES, ids=IDS)
@pytest.mark.slow  # compiles the operator several times (~2 s)
def test_contract(name, op, ref, n):
    check_operator(op, *[_field(k) for k in range(n)])


@pytest.mark.parametrize(("name", "op", "ref", "n"), CASES, ids=IDS)
def test_equals_tier_a(name, op, ref, n):
    xs = [_field(10 + k) for k in range(n)]
    got, want = op(*xs), ref(*xs)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(a, b)


def test_tables_cover_every_tier_a_method():
    assert len(fp.DIFFERENCES) == 8
    assert len(fp.INTERPOLATIONS) == 12


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: fp.Difference(grid=SPEC, axis="x", src="T", dst="V"), "x/T/U"),
        (lambda: fp.Interp(grid=SPEC, src="T", dst="T"), "T/U"),
        (lambda: fp.FillGhosts(kind="reflect"), "periodic"),
        (lambda: fp.ApplyMask(mask=MASK, stagger="W"), "T"),
    ],
)
def test_invalid_choices_list_the_valid_set(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_lincomb_checks_its_arity():
    with pytest.raises(ValueError, match="2 coefficients"):
        fp.LinComb(coeffs=(1.0, 2.0))(_field(0))


def test_pick_selects_a_tuple_output():
    u, v = fp.GradPerp(grid=SPEC)(_field(0))
    np.testing.assert_array_equal(fp.Pick(index=1)((u, v)), v)


def test_coriolis_parameters_are_learnable():
    op = fp.Coriolis(grid=SPEC, f0=2.0, beta=0.3)
    u, v = _field(0), _field(1)

    def loss(o):
        cu, cv = o(u, v)
        return jnp.sum(cu**2 + cv**2)

    grads = eqx.filter_grad(loss)(op)
    eps = 1e-6
    fd = (
        loss(fp.Coriolis(grid=SPEC, f0=2.0 + eps, beta=0.3))
        - loss(fp.Coriolis(grid=SPEC, f0=2.0 - eps, beta=0.3))
    ) / (2 * eps)
    np.testing.assert_allclose(grads.f0, fd, rtol=1e-6)


# ---------------------------------------------------------------- composition
H, G, F0 = 1.0, 9.81, 2.0


def sw_tendency() -> Graph:
    """Linear rotating shallow water on a doubly periodic C-grid.

    ∂ₜh = −H ∇·(u, v),  ∂ₜu = +f v̄ − g δₓh,  ∂ₜv = −f ū − g δᵧh
    """
    h, u, v = Input("h"), Input("u"), Input("v")
    hp, up, vp = (fp.FillGhosts(kind="periodic")(x) for x in (h, u, v))
    dh = fp.LinComb(coeffs=(-H,))(fp.Divergence(grid=SPEC)(up, vp))
    cor = fp.Coriolis(grid=SPEC, f0=F0)(up, vp)
    dx_h = fp.Difference(grid=SPEC, axis="x", src="T", dst="U")(hp)
    dy_h = fp.Difference(grid=SPEC, axis="y", src="T", dst="V")(hp)
    du = fp.LinComb(coeffs=(1.0, -G))(fp.Pick(index=0)(cor), dx_h)
    dv = fp.LinComb(coeffs=(1.0, -G))(fp.Pick(index=1)(cor), dy_h)
    return Graph(inputs={"h": h, "u": u, "v": v}, outputs={"h": dh, "u": du, "v": dv})


def sw_tendency_tier_a(h, u, v):
    h, u, v = (fvx.enforce_periodic(x) for x in (h, u, v))
    cu, cv = fvx.Coriolis2D(grid=GRID)(u, v, _f_beta(F0, 0.0))
    return {
        "h": -H * D.divergence(u, v),
        "u": cu - G * D.diff_x_T_to_U(h),
        "v": cv - G * D.diff_y_T_to_V(h),
    }


def test_shallow_water_graph_equals_tier_a():
    state = {
        k: _field(i)[None].repeat(3, axis=0) * (1 + jnp.arange(3.0))[:, None, None]
        for i, k in enumerate("huv")
    }  # [3, Ny, Nx]: a batch of states
    graph = sw_tendency()
    want = sw_tendency_tier_a(**state)
    eager = graph(**state)
    jitted = eqx.filter_jit(lambda g, s: g(**s))(graph, state)
    for k in "huv":
        np.testing.assert_allclose(eager[k], want[k], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(jitted[k], want[k], rtol=1e-12, atol=1e-12)
