"""The grid classes are frozen NumPy dataclasses and JAX pytrees.

Every grid is built with NumPy, so the grid modules import without JAX; when
JAX is importable each class is registered as a pytree whose sizes (``Nx``,
``Ny``, ``Nz``) are static metadata. That lets a grid go to plain
``jax.jit`` as an argument and still give concrete shapes.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import subprocess
import sys
import textwrap

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import (
    CartesianGrid1D,
    CartesianGrid2D,
    CartesianGrid3D,
    Difference1D,
    Difference2D,
    Difference3D,
    SphericalDifference2D,
    SphericalDifference3D,
    SphericalGrid2D,
    SphericalGrid3D,
)

ROOT = Path(__file__).resolve().parents[1]

GRIDS = {
    "CartesianGrid1D": lambda: CartesianGrid1D.from_interior(8, 1.0),
    "CartesianGrid2D": lambda: CartesianGrid2D.from_interior(8, 6, 1.0, 2.0),
    "CartesianGrid3D": lambda: CartesianGrid3D.from_interior(8, 6, 4, 1.0, 2.0, 3.0),
    "SphericalGrid2D": lambda: SphericalGrid2D.from_interior(
        8, 6, (0.0, 40.0), (-30.0, 30.0)
    ),
    "SphericalGrid3D": lambda: SphericalGrid3D.from_interior(
        8, 6, 4, (0.0, 40.0), (-30.0, 30.0), 100.0
    ),
}


@pytest.fixture(params=sorted(GRIDS))
def grid(request):
    return GRIDS[request.param]()


class TestDataclass:
    def test_is_frozen_dataclass(self, grid):
        assert dataclasses.is_dataclass(grid)
        assert not isinstance(grid, eqx.Module)
        with pytest.raises(dataclasses.FrozenInstanceError):
            grid.Nx = 3

    def test_arrays_are_numpy(self, grid):
        for field in dataclasses.fields(grid):
            value = getattr(grid, field.name)
            assert isinstance(value, int | float | np.ndarray), field.name

    def test_equal_when_built_the_same_way(self, grid):
        name = type(grid).__name__
        twin = GRIDS[name]()
        assert grid == twin
        assert hash(grid) == hash(twin)
        assert grid != dataclasses.replace(grid, dx=2.0 * grid.dx)


class TestPytree:
    def test_round_trip(self, grid):
        leaves, treedef = jax.tree_util.tree_flatten(grid)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert type(rebuilt) is type(grid)
        assert rebuilt == grid

    def test_sizes_are_static(self, grid):
        leaves = jax.tree_util.tree_leaves(grid)
        sizes = {grid.Nx, getattr(grid, "Ny", None), getattr(grid, "Nz", None)}
        assert not any(isinstance(leaf, int) and leaf in sizes for leaf in leaves)

    def test_key_paths_name_fields(self, grid):
        paths = [
            jax.tree_util.keystr(path)
            for path, _ in jax.tree_util.tree_flatten_with_path(grid)[0]
        ]
        assert ".dx" in paths

    def test_plain_jit_sees_concrete_shapes(self, grid):
        def zeros_like_grid(g):
            shape = tuple(getattr(g, n) for n in ("Nz", "Ny", "Nx") if hasattr(g, n))
            return jnp.zeros(shape)

        out = jax.jit(zeros_like_grid)(grid)
        assert out.shape[-1] == grid.Nx

    def test_tree_map_leaves_the_constructor_alone(self, grid):
        """Unflattening with non-array leaves (``None``) must not run checks."""
        emptied = jax.tree_util.tree_map(lambda _: None, grid)
        assert emptied.dx is None
        assert emptied.Nx == grid.Nx


def _jitted_operator_case(name):
    """One operator per grid class, jitted with the grid as an argument."""
    g = GRIDS[name]()
    if name == "CartesianGrid1D":
        h = jnp.sin(jnp.arange(g.Nx, dtype=float))
        return g, h, lambda grid, h: Difference1D(grid=grid).diff_x_T_to_U(h)
    if name == "CartesianGrid2D":
        h = jnp.sin(jnp.arange(g.Ny * g.Nx, dtype=float)).reshape(g.Ny, g.Nx)
        return g, h, lambda grid, h: Difference2D(grid=grid).laplacian(h)
    if name == "CartesianGrid3D":
        h = jnp.sin(jnp.arange(g.Nz * g.Ny * g.Nx, dtype=float)).reshape(
            g.Nz, g.Ny, g.Nx
        )
        return g, h, lambda grid, h: Difference3D(grid=grid).diff_x_T_to_U(h)
    if name == "SphericalGrid2D":
        h = jnp.sin(jnp.arange(g.Ny * g.Nx, dtype=float)).reshape(g.Ny, g.Nx)
        return g, h, lambda grid, h: SphericalDifference2D(grid=grid).diff_lon_T_to_U(h)
    h = jnp.sin(jnp.arange(g.Nz * g.Ny * g.Nx, dtype=float)).reshape(g.Nz, g.Ny, g.Nx)
    return g, h, lambda grid, h: SphericalDifference3D(grid=grid).diff_lon_T_to_U(h)


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_operator_jits_with_grid_as_argument(name):
    grid, h, fn = _jitted_operator_case(name)
    np.testing.assert_allclose(jax.jit(fn)(grid, h), fn(grid, h), rtol=1e-12)
    np.testing.assert_allclose(eqx.filter_jit(fn)(grid, h), fn(grid, h), rtol=1e-12)


def test_spherical_metric_properties_run_under_jit():
    grid = GRIDS["SphericalGrid2D"]()
    for prop in ("min_cell_width", "max_aspect"):
        eager = getattr(grid, prop)
        traced = jax.jit(lambda g, prop=prop: getattr(g, prop))(grid)
        np.testing.assert_allclose(traced, eager, rtol=1e-12)


def test_grid_modules_import_without_jax():
    """The grid modules build every grid with JAX and Equinox unimportable.

    ``finitevolx/__init__.py`` imports the operators, which need JAX, so the
    script stands in an empty ``finitevolx`` package and imports
    ``finitevolx._src.grid`` directly.
    """
    script = textwrap.dedent(
        """
        import importlib.abc, sys, types

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in {"jax", "jaxlib", "equinox"}:
                    raise ImportError(f"blocked: {name}")

        sys.meta_path.insert(0, Block())
        package = types.ModuleType("finitevolx")
        package.__path__ = [sys.argv[1]]
        sys.modules["finitevolx"] = package

        import finitevolx._src.grid as grid

        grid.CartesianGrid1D.from_interior(4, 1.0)
        grid.CartesianGrid3D.from_interior(4, 4, 4, 1.0, 1.0, 1.0).horizontal_grid()
        s = grid.SphericalGrid3D.from_interior(8, 6, 3, (0, 40), (-30, 30), 1.0)
        float(s.min_cell_width), float(s.max_aspect), s.horizontal_grid().dx_V
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in {"jax", "equinox"})
        assert not loaded, loaded
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT / "finitevolx")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
