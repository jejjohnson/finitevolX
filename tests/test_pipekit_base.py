"""Scaffolding of the `finitevolx.pipekit` operator layer (Tier B)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("pipekit_jax")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pipekit import Operator

import finitevolx as fvx
from finitevolx.pipekit import (
    GridOperator,
    as_grid_spec,
    build_grid,
    over_leading,
)
from finitevolx.pipekit.testing import check_operator, compiled_hlo

SPEC = (8, 6, 2.0, 1.5)


class ScaledLaplacian(GridOperator):
    """h ↦ κ ∇²h at T-points (κ learnable): a stand-in Tier-B operator."""

    kappa: jax.Array = eqx.field(converter=jnp.asarray, default=1.0)

    def _apply(self, h):
        return self.kappa * fvx.Difference2D(grid=self.cartesian_grid).laplacian(h)


def _field(seed: int = 0) -> jax.Array:
    g = build_grid(SPEC)
    return jax.random.normal(jax.random.key(seed), (g.Ny, g.Nx))


class TestGridSpec:
    def test_normalises_lists_and_grids(self):
        assert as_grid_spec([8, 6, 2, 1.5]) == (8, 6, 2.0, 1.5)
        grid = fvx.CartesianGrid2D.from_interior(8, 6, 2.0, 1.5)
        assert as_grid_spec(grid) == SPEC

    @pytest.mark.parametrize("bad", [(8, 6, 2.0), (0, 6, 2.0, 1.5)])
    def test_rejects_bad_specs(self, bad):
        with pytest.raises(ValueError, match="grid spec"):
            as_grid_spec(bad)

    def test_build_grid_matches_from_interior_and_is_cached(self):
        g = build_grid(SPEC)
        ref = fvx.CartesianGrid2D.from_interior(*SPEC)
        assert (g.Nx, g.Ny, g.dx, g.dy) == (ref.Nx, ref.Ny, ref.dx, ref.dy)
        assert build_grid(SPEC) is g

    def test_grid_field_is_static_and_round_trips_json(self):
        op = ScaledLaplacian(grid=list(SPEC), kappa=0.5)
        assert op.grid == SPEC
        assert len(jax.tree.leaves(op)) == 1  # kappa only; the grid is static
        state = json.loads(json.dumps(op.state))
        rebuilt = Operator.from_state(state)
        assert rebuilt.grid == SPEC


class TestOverLeading:
    @pytest.mark.parametrize("lead", [(), (3,), (2, 3)])
    def test_lifts_a_2d_function(self, lead):
        def f(x):  # 2-D only: indexes axis 0 positionally
            return x[1:-1] - x[:-2]

        x = jax.random.normal(jax.random.key(1), (*lead, 5, 4))
        out = over_leading(f)(x)
        flat = x.reshape(-1, 5, 4)
        ref = jnp.stack([f(xi) for xi in flat]).reshape(*lead, 3, 4)
        np.testing.assert_array_equal(out, ref)

    def test_tuple_outputs_keep_the_leading_shape(self):
        def f(x, y):
            return x + y, (x * y).sum()

        x = jnp.ones((2, 3, 5, 4))
        a, s = over_leading(f)(x, 2 * x)
        assert a.shape == (2, 3, 5, 4)
        assert s.shape == (2, 3)

    def test_rejects_mismatched_leading_shapes(self):
        with pytest.raises(ValueError, match="leading shapes"):
            over_leading(lambda x, y: x + y)(jnp.ones((2, 5, 4)), jnp.ones((3, 5, 4)))


class TestCheckOperator:
    def test_passes_for_a_tier_b_operator(self):
        check_operator(ScaledLaplacian(grid=SPEC, kappa=0.5), _field())

    def test_matches_tier_a(self):
        h = _field(2)
        out = ScaledLaplacian(grid=SPEC, kappa=0.5)(h)
        ref = 0.5 * fvx.Difference2D(grid=build_grid(SPEC)).laplacian(h)
        np.testing.assert_array_equal(out, ref)

    def test_flags_a_gather(self):
        class Gathers(GridOperator):
            def _apply(self, h):
                # per-row lookup with a data-dependent index: a gather
                w = jnp.linspace(0.0, 1.0, h.shape[-1])
                return h * w[jnp.argmax(h, axis=-1)][..., None]

        with pytest.raises(AssertionError, match="gather"):
            check_operator(Gathers(grid=SPEC), _field())
        assert "gather(" in compiled_hlo(Gathers(grid=SPEC), _field())


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_import_finitevolx_does_not_import_pipekit():
    res = _run(
        "import sys, finitevolx; "
        "assert not any(m.split('.')[0] in ('pipekit', 'pipekit_jax') "
        "for m in sys.modules), 'pipekit imported'"
    )
    assert res.returncode == 0, res.stderr


def test_missing_extra_names_it():
    res = _run(
        "import sys; sys.modules['pipekit_jax'] = None; import finitevolx.pipekit"
    )
    assert res.returncode != 0
    assert "finitevolx[pipekit]" in res.stderr
