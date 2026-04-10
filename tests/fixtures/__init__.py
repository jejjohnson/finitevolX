"""Shared test fixtures for finitevolX golden-output regression tests.

The :mod:`tests.fixtures.inputs` module exposes a single deterministic
test domain (16x16, ``ArakawaCGrid2D``) plus a single deterministic
:class:`~finitevolx.ArakawaCGridMask` with an island and a coastline.
Every operator under test consumes the same input fields and the same
mask, and asserts its output against a per-method ``.npz`` file under
``tests/fixtures/golden/``.

The goldens are produced by ``tests/fixtures/_gen_golden.py``.  Re-run
that script whenever the underlying operator math intentionally
changes; the script also serves as a single readable description of
"what each operator is expected to compute on the canonical input."

Importing this package always enables ``jax_enable_x64`` so that the
golden outputs are deterministic regardless of which other test file
ran first (several existing test modules enable ``x64`` and the JAX
config is process-global).
"""

import jax

jax.config.update("jax_enable_x64", True)
