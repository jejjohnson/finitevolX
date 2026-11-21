"""``import finitevolx`` is a deprecated shim over gridax_jax for one 0.X release.

It warns once, re-exports every public name as the same object (the grid and
mask classes come from gridax) and keeps ``finitevolx.pipekit`` working.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings

import gridax
import gridax_jax
import pytest


def test_import_warns_with_the_new_name():
    result = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", "import finitevolx"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "renamed gridax_jax" in result.stderr


@pytest.fixture(scope="module")
def finitevolx():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return importlib.import_module("finitevolx")


def test_every_name_is_the_gridax_jax_object(finitevolx):
    assert finitevolx.__all__ == gridax_jax.__all__
    for name in gridax_jax.__all__:
        if name in gridax_jax.__dict__:
            assert getattr(finitevolx, name) is getattr(gridax_jax, name), name


def test_grid_and_mask_classes_come_from_gridax(finitevolx):
    shared = set(gridax.__all__) & set(finitevolx.__all__)
    assert {"CartesianGrid2D", "SphericalGrid2D", "Mask2D"} <= shared
    for name in shared:
        assert getattr(finitevolx, name) is getattr(gridax, name), name


def test_pipekit_submodules(finitevolx):
    pytest.importorskip("pipekit_jax")
    shim = importlib.import_module("finitevolx.pipekit")
    new = importlib.import_module("gridax_jax.pipekit")
    for name in new.__all__:
        assert getattr(shim, name) is getattr(new, name), name
    testing = importlib.import_module("finitevolx.pipekit.testing")
    assert (
        testing.check_operator
        is importlib.import_module("gridax_jax.pipekit.testing").check_operator
    )
