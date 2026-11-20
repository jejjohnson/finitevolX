"""The mask classes are frozen NumPy dataclasses and JAX pytrees.

Masks are built and stored with NumPy, so the mask modules import without
JAX; when JAX is importable each class is a pytree. The fields that were
kept match the masks the Equinox classes built, value for value
(``tests/fixtures/mask_fields_pre_419.npz`` was written by the old classes
from the canonical fixture masks).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import equinox as eqx
from gridax_jax import (
    Advection1D,
    Advection2D,
    Advection3D,
    CartesianGrid1D,
    CartesianGrid2D,
    CartesianGrid3D,
    Mask1D,
    Mask2D,
)
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.fixtures import inputs

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "fixtures" / "mask_fields_pre_419.npz"

MASKS = {
    "1d": inputs.make_mask_1d,
    "2d": inputs.make_mask_2d,
    "3d": inputs.make_mask_3d,
}


@pytest.fixture(params=sorted(MASKS))
def mask(request):
    return MASKS[request.param]()


class TestDataclass:
    def test_is_frozen_numpy_dataclass(self, mask):
        assert dataclasses.is_dataclass(mask)
        assert not isinstance(mask, eqx.Module)
        with pytest.raises(dataclasses.FrozenInstanceError):
            mask.h = mask.h
        for leaf in jax.tree_util.tree_leaves(mask):
            assert isinstance(leaf, np.ndarray)

    def test_removed_fields_are_gone(self, mask):
        names = {f.name for f in dataclasses.fields(mask)}
        assert "sponge" not in names
        assert not any("irrbound" in n or n.startswith("not_") for n in names)

    def test_not_fields_are_inverses(self, mask):
        for name in ("h", "u"):
            np.testing.assert_array_equal(
                getattr(mask, f"not_{name}"), ~getattr(mask, name)
            )

    def test_narrow_integer_storage(self, mask):
        assert mask.classification.dtype == np.int8
        for field in dataclasses.fields(mask.stencil_capability):
            assert getattr(mask.stencil_capability, field.name).dtype == np.int8


def test_kept_fields_match_the_equinox_masks():
    ref = np.load(REFERENCE)
    for key in ref.files:
        dim, name = key.split("/")
        m = MASKS[dim]()
        if name.startswith("adaptive_x."):
            size = int(name.split(".")[1])
            kwargs = {} if dim == "1d" else {"direction": "x"}
            value = m.get_adaptive_masks(stencil_sizes=(2, 4, 6, 8, 10), **kwargs)[size]
        elif name.startswith("stencil_capability."):
            value = getattr(m.stencil_capability, name.split(".")[1])
        else:
            value = getattr(m, name)
        np.testing.assert_array_equal(np.asarray(value), ref[key], err_msg=key)


def test_stencil_counts_saturate_at_127():
    """Counts are int8; a 300-cell wet run reports 127, never wraps."""
    m = Mask1D.from_mask(np.ones(300, dtype=bool))
    counts = m.stencil_capability.x_pos
    assert counts.max() == 127 and counts.min() == 1
    tiers = m.get_adaptive_masks(stencil_sizes=(2, 4, 6, 8, 10))
    assert tiers[10][5:-5].all()


def test_mask2d_pytree_size_at_512():
    """#325: the 512² pytree was 9.96 MB with sponge, not_* and int32 counts."""
    m = Mask2D.from_dimensions(512, 512)
    nbytes = sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(m))
    assert nbytes <= 4_000_000, nbytes


class TestPytree:
    def test_round_trip(self, mask):
        leaves, treedef = jax.tree_util.tree_flatten(mask)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert type(rebuilt) is type(mask)
        assert rebuilt == mask

    def test_adaptive_masks_under_jit(self, mask):
        kwargs = {} if isinstance(mask, Mask1D) else {"direction": "x"}
        eager = mask.get_adaptive_masks(stencil_sizes=(2, 4, 6), **kwargs)
        traced = jax.jit(
            lambda m: m.get_adaptive_masks(stencil_sizes=(2, 4, 6), **kwargs)
        )(mask)
        for size in (2, 4, 6):
            np.testing.assert_array_equal(traced[size], eager[size])


def _advection_case(dim):
    if dim == "1d":
        grid = CartesianGrid1D.from_interior(inputs.NX - 2, 1.0)
        h = jnp.cos(jnp.linspace(0.0, 3.0, inputs.NX))
        return grid, (h, jnp.ones_like(h)), Advection1D
    if dim == "2d":
        grid = CartesianGrid2D.from_interior(inputs.NX - 2, inputs.NY - 2, 1.0, 1.0)
        h = inputs.make_h_field_2d()
        return grid, (h, jnp.ones_like(h), 0.5 * jnp.ones_like(h)), Advection2D
    grid = CartesianGrid3D.from_interior(
        inputs.NX - 2, inputs.NY - 2, inputs.NZ - 2, 1.0, 1.0, 1.0
    )
    h = jnp.broadcast_to(inputs.make_h_field_2d(), (inputs.NZ, inputs.NY, inputs.NX))
    return grid, (h, jnp.ones_like(h), 0.5 * jnp.ones_like(h)), Advection3D


@pytest.mark.parametrize("dim", sorted(MASKS))
def test_masked_advection_jits_with_mask_as_argument(dim):
    grid, fields, cls = _advection_case(dim)
    mask = MASKS[dim]()

    def tendency(mask, *fields):
        return cls(grid=grid, mask=mask)(*fields, method="weno5")

    np.testing.assert_allclose(
        eqx.filter_jit(tendency)(mask, *fields), tendency(mask, *fields), rtol=1e-12
    )


def test_from_bathymetry_equals_from_mask_of_the_threshold():
    rng = np.random.default_rng(0)
    depth = rng.uniform(-50.0, 200.0, size=(20, 24))
    depth[3, 4] = np.nan
    m = Mask2D.from_bathymetry(depth, min_depth=10.0)
    with np.errstate(invalid="ignore"):
        expected = Mask2D.from_mask(depth > 10.0)
    assert m == expected
    assert not m.h[3, 4]


def test_from_bathymetry_drops_small_wet_regions():
    depth = np.full((12, 12), -1.0)
    depth[1:9, 1:9] = 100.0  # basin of 64 cells
    depth[10, 10] = 100.0  # isolated one-cell pond
    assert Mask2D.from_bathymetry(depth).h[10, 10]
    m = Mask2D.from_bathymetry(depth, min_area=2)
    assert not m.h[10, 10]
    assert m.h[1:9, 1:9].all()
