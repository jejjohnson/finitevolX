"""Smoke test for the masks-everywhere golden-fixture infrastructure.

These checks fail loudly if the canonical inputs or the golden helpers
are misconfigured.  They do not exercise any operator under test — see
``test_difference.py`` etc. for the per-operator regression tests that
load the goldens themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures._helpers import (
    assert_matches_golden,
    golden_path,
    load_golden,
    save_golden,
)
from tests.fixtures.inputs import (
    NX,
    NX_INTERIOR,
    NY,
    NY_INTERIOR,
    NZ,
    NZ_INTERIOR,
    all_2d_fields,
    all_3d_fields,
    make_grid_2d,
    make_grid_3d,
    make_h_field_2d,
    make_h_mask_2d,
    make_mask_2d,
    make_mask_2d_all_ocean,
)


class TestDimensions:
    def test_total_equals_interior_plus_ghost(self):
        assert NY == NY_INTERIOR + 2
        assert NX == NX_INTERIOR + 2
        assert NZ == NZ_INTERIOR + 2

    def test_grid2d_matches_inputs(self):
        g = make_grid_2d()
        assert g.Ny == NY
        assert g.Nx == NX

    def test_grid3d_matches_inputs(self):
        g = make_grid_3d()
        assert g.Nz == NZ
        assert g.Ny == NY
        assert g.Nx == NX


class TestMask:
    def test_mask_shape(self):
        mask = make_mask_2d()
        assert mask.h.shape == (NY, NX)
        assert mask.u.shape == (NY, NX)
        assert mask.v.shape == (NY, NX)
        assert mask.psi.shape == (NY, NX)
        assert mask.w.shape == (NY, NX)

    def test_coastal_mask_has_dry_interior(self):
        """The cross-shaped island carves out interior dry cells."""
        mask = make_mask_2d()
        # Interior (excluding ghost ring) has dry cells from the island.
        interior_h = np.asarray(mask.h[1:-1, 1:-1])
        n_dry_interior = int((~interior_h).sum())
        assert n_dry_interior > 0, "expected island + coastal-ring dry cells"

    def test_all_ocean_mask_is_fully_wet(self):
        mask = make_mask_2d_all_ocean()
        assert int(np.asarray(mask.h).sum()) == NY * NX

    def test_h_mask_outer_ring_is_dry(self):
        h_mask = make_h_mask_2d()
        assert not h_mask[0, :].any()
        assert not h_mask[-1, :].any()
        assert not h_mask[:, 0].any()
        assert not h_mask[:, -1].any()


class TestFields:
    def test_field_shapes_2d(self):
        bundle = all_2d_fields()
        for name, arr in bundle.items():
            assert arr.shape == (NY, NX), (
                f"2-D field {name!r} has shape {arr.shape}, expected ({NY}, {NX})"
            )

    def test_field_shapes_3d(self):
        bundle = all_3d_fields()
        # h, u, v are 3-D; f is 2-D (depth-independent).
        for name in ("h", "u", "v"):
            assert bundle[name].shape == (NZ, NY, NX), (
                f"3-D field {name!r} has shape {bundle[name].shape}, "
                f"expected ({NZ}, {NY}, {NX})"
            )
        assert bundle["f"].shape == (NY, NX)

    def test_h_field_is_finite(self):
        h = np.asarray(make_h_field_2d())
        assert np.isfinite(h).all()


class TestHelpers:
    def test_save_and_load_single_array(self, tmp_path, monkeypatch):
        """save_golden + load_golden round-trip a single-output operator."""
        # Redirect golden dir to tmp.
        import tests.fixtures._helpers as helpers

        monkeypatch.setattr(helpers, "_GOLDEN_DIR", tmp_path)
        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        save_golden("FakeOp", "do_thing", "unmasked", arr)
        loaded = load_golden("FakeOp", "do_thing", "unmasked")
        assert "out" in loaded
        np.testing.assert_array_equal(loaded["out"], arr)

    def test_save_and_load_tuple(self, tmp_path, monkeypatch):
        """save_golden + load_golden round-trip a multi-output operator."""
        import tests.fixtures._helpers as helpers

        monkeypatch.setattr(helpers, "_GOLDEN_DIR", tmp_path)
        a = np.zeros((2, 2))
        b = np.ones((2, 2))
        save_golden("FakeOp", "two_out", "masked", (a, b))
        loaded = load_golden("FakeOp", "two_out", "masked")
        assert set(loaded) == {"out0", "out1"}
        np.testing.assert_array_equal(loaded["out0"], a)
        np.testing.assert_array_equal(loaded["out1"], b)

    def test_assert_matches_golden_passes_for_exact_match(self, tmp_path, monkeypatch):
        import tests.fixtures._helpers as helpers

        monkeypatch.setattr(helpers, "_GOLDEN_DIR", tmp_path)
        arr = np.linspace(0, 1, 16).reshape(4, 4)
        save_golden("FakeOp", "exact", "masked", arr)
        # Should not raise.
        assert_matches_golden(arr, "FakeOp", "exact", "masked")

    def test_assert_matches_golden_fails_for_drift(self, tmp_path, monkeypatch):
        import tests.fixtures._helpers as helpers

        monkeypatch.setattr(helpers, "_GOLDEN_DIR", tmp_path)
        save_golden("FakeOp", "drift", "masked", np.zeros((4, 4)))
        with pytest.raises(AssertionError):
            assert_matches_golden(np.ones((4, 4)), "FakeOp", "drift", "masked")

    def test_load_golden_raises_on_missing(self, tmp_path, monkeypatch):
        import tests.fixtures._helpers as helpers

        monkeypatch.setattr(helpers, "_GOLDEN_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="Re-run"):
            load_golden("Nope", "missing", "masked")

    def test_golden_path_naming(self):
        p = golden_path("Foo", "bar", "masked")
        assert p.name == "Foo__bar__masked.npz"
