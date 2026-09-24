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
    Strain2D,
    available_potential_energy,
    bernoulli_potential,
    kinetic_energy,
    shear_strain,
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
    """With mask=None the classes reproduce the functional forms."""

    def test_energetics(self):
        grid = make_grid_2d()
        op = Energetics2D(grid=grid)
        h, u, v = _args("bernoulli_potential")
        H = _args("available_potential_energy")[1]
        np.testing.assert_array_equal(op.kinetic_energy(u, v), kinetic_energy(u, v))
        np.testing.assert_array_equal(
            op.bernoulli_potential(h, u, v, 9.0), bernoulli_potential(h, u, v, 9.0)
        )
        # APE: the functional form is pointwise over the whole array; the
        # class keeps only the interior and zeroes the ghost ring.
        ape = op.available_potential_energy(h, H, G_PRIME)
        ref = available_potential_energy(h, H, G_PRIME)
        # ape[j, i] = ref[j, i] for 1 <= j <= Ny-2, 1 <= i <= Nx-2
        np.testing.assert_array_equal(ape[1:-1, 1:-1], ref[1:-1, 1:-1])
        ghost = np.ones(ape.shape, dtype=bool)
        ghost[1:-1, 1:-1] = False
        assert np.any(np.asarray(ref)[ghost] != 0.0)
        np.testing.assert_array_equal(np.asarray(ape)[ghost], 0.0)

    def test_strain_pointwise_parts(self):
        grid = make_grid_2d()
        op = Strain2D(grid=grid)
        u, v = _args("shear")
        np.testing.assert_array_equal(
            op.shear(u, v), shear_strain(u, v, grid.dx, grid.dy)
        )
        np.testing.assert_array_equal(
            op.tensor(u, v), tensor_strain(u, v, grid.dx, grid.dy)
        )


def _strain_reference(u, v, dx, dy):
    """Loop-based numpy reference for (sigma2, ow) at T-points.

    Every T-point average reads its four corners, including the south / west
    ghost corners computed from the ghost u / v.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    Ny, Nx = u.shape

    def corner(j, i, sign):
        # x[j+1/2, i+1/2] = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
        #                 + sign * (u[j+1, i+1/2] - u[j, i+1/2]) / dy
        return (v[j, i + 1] - v[j, i]) / dx + sign * (u[j + 1, i] - u[j, i]) / dy

    sigma2 = np.zeros((Ny, Nx))
    ow = np.zeros((Ny, Nx))
    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            # sn[j, i] = (u[j, i+1/2] - u[j, i-1/2]) / dx
            #          - (v[j+1/2, i] - v[j-1/2, i]) / dy
            sn = (u[j, i] - u[j, i - 1]) / dx - (v[j, i] - v[j - 1, i]) / dy
            corners = [(j, i), (j - 1, i), (j, i - 1), (j - 1, i - 1)]
            ss = 0.25 * sum(corner(a, b, 1.0) for a, b in corners)
            om = 0.25 * sum(corner(a, b, -1.0) for a, b in corners)
            sigma2[j, i] = sn**2 + ss**2
            ow[j, i] = sn**2 + ss**2 - om**2
    return sigma2, ow


class TestStrainGhostCorners:
    """The south / west ghost X-points are built from the ghost u / v."""

    def test_matches_loop_reference(self):
        grid = make_grid_2d()
        u, v = _args("shear")
        sigma2, ow = _strain_reference(u, v, grid.dx, grid.dy)
        op = Strain2D(grid=grid)
        np.testing.assert_allclose(op.magnitude_squared(u, v), sigma2, rtol=1e-12)
        np.testing.assert_allclose(op.okubo_weiss(u, v), ow, rtol=1e-12, atol=1e-12)

    def test_first_row_and_column_use_ghost_velocities(self):
        """Changing the south ghost u-row changes the first interior row."""
        grid = make_grid_2d()
        u, v = _args("shear")
        op = Strain2D(grid=grid)
        # u2[0, i] = u[0, i] + 1 (south ghost U-row only)
        u2 = u.at[0, :].add(1.0)
        d = np.asarray(op.magnitude_squared(u2, v) - op.magnitude_squared(u, v))
        assert np.any(d[1, 1:-1] != 0.0)
        np.testing.assert_array_equal(d[2:, :], 0.0)


class TestNaNOnLand:
    """Land values stored as NaN neither leak into wet cells nor survive."""

    @pytest.mark.parametrize(("name", "method"), CASES)
    def test_nan_land_inputs(self, name, method):
        mask = make_mask_2d()
        op = _op(name, mask)
        args = _args(method)
        wet_of = {
            "u": np.asarray(mask.u),
            "v": np.asarray(mask.v),
            "h": np.asarray(mask.h),
        }
        # Positional stagger of each array argument, per method.
        staggers = {
            "bernoulli_potential": ("h", "u", "v"),
            "available_potential_energy": ("h", "h"),
        }.get(method, ("u", "v"))
        # arg[j, i] = NaN on dry cells of its stagger
        nan_args = (
            tuple(
                jnp.where(wet_of[s], a, jnp.nan)
                for s, a in zip(staggers, args, strict=False)
            )
            + args[len(staggers) :]
        )
        out = np.asarray(getattr(op, method)(*args))
        out_nan = np.asarray(getattr(op, method)(*nan_args))
        assert np.all(np.isfinite(out_nan))
        np.testing.assert_array_equal(out_nan, out)


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
        # ow[j, i] for 1 <= j <= Ny-2, 1 <= i <= Nx-2: the south / west ghost
        # corners are built from the (linear) ghost velocities, so the first
        # interior row and column are exact too.
        ow = op.okubo_weiss(u, v)[1:-1, 1:-1]
        np.testing.assert_allclose(ow, -4.0, rtol=1e-12)
        np.testing.assert_allclose(
            op.magnitude_squared(u, v)[1:-1, 1:-1], 0.0, atol=1e-12
        )
