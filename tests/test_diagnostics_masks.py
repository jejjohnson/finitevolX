"""Golden-output regression tests for the Energetics2D / Strain2D mask field.

The classes are the Layer-3 forms of the mask-free functional diagnostics
(issue #206).  Each method gets: unmasked golden, masked golden, all-ocean
invariant, a dry-cell-zero pin at its output stagger, and bit-identity to the
functional form (or documented composition) when ``mask=None``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import (
    Energetics2D,
    Interpolation2D,
    Strain2D,
    available_potential_energy,
    bernoulli_potential,
    kinetic_energy,
    okubo_weiss,
    relative_vorticity_cgrid,
    shear_strain,
    strain_magnitude_squared,
    tensor_strain,
)
from tests.fixtures._helpers import assert_matches_golden
from tests.fixtures.inputs import (
    make_grid_2d,
    make_h_field_2d,
    make_mask_2d,
    make_mask_2d_all_ocean,
    make_u_field_2d,
    make_v_field_2d,
)

G_PRIME = 0.02


def _args(method: str) -> tuple:
    """Positional arguments for each method, matching _gen_golden.py."""
    h = make_h_field_2d()
    u = make_u_field_2d()
    v = make_v_field_2d()
    if method == "bernoulli_potential":
        return (h, u, v)
    if method == "available_potential_energy":
        H = h.mean() + 0.0 * h
        return (h, H, G_PRIME)
    return (u, v)


ENERGETICS_METHODS = [
    "kinetic_energy",
    "bernoulli_potential",
    "available_potential_energy",
]
STRAIN_METHODS = ["shear", "tensor", "magnitude_squared", "okubo_weiss"]
CASES = [("Energetics2D", m) for m in ENERGETICS_METHODS] + [
    ("Strain2D", m) for m in STRAIN_METHODS
]
_CLASSES = {"Energetics2D": Energetics2D, "Strain2D": Strain2D}


def _op(name: str, mask=None):
    return _CLASSES[name](grid=make_grid_2d(), mask=mask)


def _dry(name: str, method: str, mask) -> np.ndarray:
    """Dry cells of the method's output stagger."""
    if name == "Strain2D" and method == "shear":
        return ~np.asarray(mask.xy_corner_strict)
    return ~np.asarray(mask.h)


@pytest.mark.parametrize(("name", "method"), CASES)
class TestDiagnosticMasks:
    def test_unmasked_golden(self, name, method):
        out = getattr(_op(name), method)(*_args(method))
        assert_matches_golden(out, name, method, "unmasked")

    def test_masked_golden(self, name, method):
        out = getattr(_op(name, make_mask_2d()), method)(*_args(method))
        assert_matches_golden(out, name, method, "masked")

    def test_all_ocean_matches_unmasked(self, name, method):
        out1 = getattr(_op(name), method)(*_args(method))
        out2 = getattr(_op(name, make_mask_2d_all_ocean()), method)(*_args(method))
        np.testing.assert_array_equal(out1, out2)

    def test_dry_cells_zero(self, name, method):
        mask = make_mask_2d()
        out = np.asarray(getattr(_op(name, mask), method)(*_args(method)))
        assert np.all(np.isfinite(out))
        dry = _dry(name, method, mask)
        assert dry.any()
        np.testing.assert_array_equal(out[dry], 0.0)


class TestMatchesFunctional:
    """With mask=None the classes are bit-identical to the functional forms."""

    def test_energetics(self):
        grid = make_grid_2d()
        op = Energetics2D(grid=grid)
        h, u, v = _args("bernoulli_potential")
        H = _args("available_potential_energy")[1]
        np.testing.assert_array_equal(op.kinetic_energy(u, v), kinetic_energy(u, v))
        np.testing.assert_array_equal(
            op.bernoulli_potential(h, u, v, 9.0), bernoulli_potential(h, u, v, 9.0)
        )
        np.testing.assert_array_equal(
            op.available_potential_energy(h, H, G_PRIME),
            available_potential_energy(h, H, G_PRIME),
        )

    def test_strain(self):
        grid = make_grid_2d()
        op = Strain2D(grid=grid)
        interp = Interpolation2D(grid=grid)
        u, v = _args("shear")
        ss = shear_strain(u, v, grid.dx, grid.dy)
        sn = tensor_strain(u, v, grid.dx, grid.dy)
        omega = relative_vorticity_cgrid(u, v, grid.dx, grid.dy)
        np.testing.assert_array_equal(op.shear(u, v), ss)
        np.testing.assert_array_equal(op.tensor(u, v), sn)
        np.testing.assert_array_equal(
            op.magnitude_squared(u, v),
            strain_magnitude_squared(sn, interp.X_to_T(ss)),
        )
        np.testing.assert_array_equal(
            op.okubo_weiss(u, v),
            okubo_weiss(sn, interp.X_to_T(ss), interp.X_to_T(omega)),
        )


class TestStrainPhysics:
    def test_solid_body_rotation_is_vorticity_dominated(self):
        """u = -y, v = x: no strain, uniform vorticity 2 -> OW = -4 inside."""
        grid = make_grid_2d()
        # Face coordinates: u lives at (y[j], x[i+1/2]), v at (y[j+1/2], x[i]).
        j, i = jnp.meshgrid(
            jnp.arange(grid.Ny, dtype=float),
            jnp.arange(grid.Nx, dtype=float),
            indexing="ij",
        )
        u = -(j * grid.dy)
        v = i * grid.dx
        op = Strain2D(grid=grid)
        # ow[j, i] for 2 <= j <= Ny-3, 2 <= i <= Nx-3 (all four corners interior)
        ow = op.okubo_weiss(u, v)[2:-2, 2:-2]
        np.testing.assert_allclose(ow, -4.0, rtol=1e-12)
        np.testing.assert_allclose(op.magnitude_squared(u, v)[2:-2, 2:-2], 0.0)
