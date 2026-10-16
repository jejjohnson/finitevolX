"""Contract check for `finitevolx.pipekit` operators."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pipekit_jax.testing import check_jax_operator

#: HLO ops a C-grid stencil must not lower to (stencil-lowering rule:
#: static slices fuse into one pass; gathers, scatters and convolutions
#: do not).
FORBIDDEN_HLO = ("gather(", "scatter(", "convolution(")


def _member_inputs(fields: tuple[Any, ...], lead: tuple[int, ...]) -> list[Any]:
    """Distinct batch members: member ``m`` scales every field by ``1 + m/4``."""
    n = int(np.prod(lead))
    return [
        tuple(f * (1.0 + m / 4) if eqx.is_inexact_array(f) else f for f in fields)
        for m in range(n)
    ]


def compiled_hlo(op: Any, *fields: Any) -> str:
    """Optimised HLO text of ``op(*fields)`` under ``eqx.filter_jit``."""
    dynamic, static = eqx.partition((op, fields), eqx.is_array)

    def run(d: Any) -> Any:
        o, xs = eqx.combine(d, static)
        return o(*xs)

    return jax.jit(run).lower(dynamic).compile().as_text()


def check_operator(op: Any, *fields: Any, hlo: bool = True) -> None:
    """Assert that ``op`` is a well-behaved finitevolX pipekit operator.

    Runs `pipekit_jax.testing.check_jax_operator` (operator + module, eager
    ≡ ``filter_jit``, ``vmap``, visible leaves, JSON config round trip),
    then checks:

    1. **Leading axes**: on inputs ``[*lead, Ny, Nx]`` with ``lead`` = ``(2,)``
       and ``(2, 3)``, ``op`` equals a loop over the members.
    2. **Gradient**: ``∂/∂x₀ ∑ op(x)`` is finite.
    3. **Lowering** (when ``hlo``): the compiled HLO has no ``gather``,
       ``scatter`` or ``convolution``.

    Parameters
    ----------
    op : FVXOperator
        Operator under test.
    *fields : Array
        Example inputs ``[Ny, Nx]`` (non-array inputs are passed through).
    hlo : bool
        Also check the lowering.

    Raises
    ------
    AssertionError
        On the first violated property.
    """
    check_jax_operator(op, *fields)

    for lead in ((2,), (2, 3)):
        members = _member_inputs(fields, lead)
        per = [op(*m) for m in members]
        want = jax.tree.map(
            lambda *ys, lead=lead: jnp.stack(ys).reshape(*lead, *ys[0].shape), *per
        )
        batched = tuple(
            jnp.stack([m[i] for m in members]).reshape(*lead, *f.shape)
            if eqx.is_array(f)
            else f
            for i, f in enumerate(fields)
        )
        got = op(*batched)
        assert jax.tree.structure(got) == jax.tree.structure(want), (
            f"leading axes {lead}: output structure differs"
        )
        for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
            # Eager batched ≡ per-member up to rounding (a few ulp).
            tol = 64 * float(jnp.finfo(a.dtype).eps)
            np.testing.assert_allclose(
                a, b, rtol=tol, atol=tol, err_msg=f"leading axes {lead}"
            )

    first = next(i for i, f in enumerate(fields) if eqx.is_inexact_array(f))

    def loss(x: Any) -> Any:
        xs = list(fields)
        xs[first] = x
        return sum(jnp.sum(jnp.real(y)) for y in jax.tree.leaves(op(*xs)))

    grad = jax.grad(loss)(fields[first])
    assert bool(jnp.all(jnp.isfinite(grad))), "gradient w.r.t. the first input"

    if hlo:
        text = compiled_hlo(op, *fields)
        found = [name.rstrip("(") for name in FORBIDDEN_HLO if name in text]
        assert not found, f"compiled HLO contains {found}"
