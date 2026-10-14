"""Session-wide test configuration.

float64
    x64 is enabled here, once, before any test module is imported, so every
    test runs in float64 regardless of collection order or which xdist
    worker it lands on (previously 44 modules each flipped the flag at
    import, and modules that did not ran in whatever mode their worker had
    inherited; #327).  Tests about float32 behaviour build float32 inputs
    explicitly: dtypes follow the input even with x64 on (AGENTS.md,
    contract 7).

Tiers
    Every test is in exactly one tier, selected with ``-m``:

    * *fast* (unmarked): unit tests, well under a second each.  The default
      lane: ``make test`` / ``pytest -m "not slow and not integration"``.
    * ``slow``: correct but expensive — convergence sweeps, large grids,
      gradients of high-order schemes, solver construction.  Measured
      > ~1 s on a quiet 2-core machine.
    * ``integration``: end-to-end runs — the example models in
      ``scripts/``, the plugin's worked example, full time integrations.

    The markers are registered in ``pyproject.toml`` (``--strict-markers``
    rejects typos) and :func:`pytest_collection_modifyitems` below rejects a
    test marked both ``slow`` and ``integration``.
"""

from __future__ import annotations

import jax
import pytest

jax.config.update("jax_enable_x64", True)

_TIERS = ("slow", "integration")


@pytest.fixture(autouse=True, scope="module")
def _clear_jax_caches():
    """Drop compiled executables after each module.

    Every distinct shape / static argument compiles a new XLA executable and
    JAX keeps them all; over a few hundred modules a single xdist worker grows
    past 2 GB and the OS kills it on small runners.  Nothing is shared between
    modules, so clearing at module teardown costs only the recompiles that a
    fresh process would pay anyway.
    """
    yield
    jax.clear_caches()


def pytest_collection_modifyitems(config, items):
    """Every test is in at most one marked tier (unmarked = fast)."""
    both = [
        item.nodeid
        for item in items
        if all(item.get_closest_marker(t) is not None for t in _TIERS)
    ]
    if both:
        raise pytest.UsageError(
            "tests marked both slow and integration (pick one tier):\n  "
            + "\n  ".join(both)
        )
