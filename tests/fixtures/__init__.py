"""Shared test fixtures for finitevolX golden-output regression tests.

The :mod:`tests.fixtures.inputs` module exposes deterministic 1-D, 2-D,
and 3-D test domains (16 / 16×16 / 4×16×16 interior, plus ghost rings)
built on :class:`~finitevolx.CartesianGrid1D` / ``CartesianGrid2D`` /
``CartesianGrid3D``, plus matching :class:`~finitevolx.Mask1D` /
``Mask2D`` / ``Mask3D`` objects with island and coastline geometries.
Every operator under test consumes the same input fields and the same
mask of the relevant dimension, and asserts its output against a
per-method ``.npz`` file under ``tests/fixtures/golden/``.

The goldens are produced by ``tests/fixtures/_gen_golden.py``.  Re-run
that script whenever the underlying operator math intentionally
changes; the script also serves as a single readable description of
"what each operator is expected to compute on the canonical input."
"""
