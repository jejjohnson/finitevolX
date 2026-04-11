"""Canonical input fixtures for finitevolX golden-output regression tests.

This module exposes deterministic 1-D, 2-D, and 3-D test domains plus
matching :class:`~finitevolx.Mask1D` / :class:`~finitevolx.Mask2D` /
:class:`~finitevolx.Mask3D` masks.  All "operator-attribute mask"
regression tests share these inputs so that:

* Every method under test sees the same field values, making it easy to
  cross-check operators that are expected to agree on smooth interiors.
* Each dimension's mask geometry is interesting enough to exercise both
  interior and boundary behavior — wet cells adjacent to land, dry
  interior cells, and a multi-cell interior island.
* The 18x18 total size (16x16 interior + ghost ring) keeps golden
  ``.npz`` files small enough to commit and diff comfortably.

The fields are smooth analytic functions (polynomials and trigs) — they
are deterministic, easy to reason about, and don't change between
``jax.random`` versions.

The 3-D fields are obtained by broadcasting the 2-D ones over a small
``Nz`` axis with a per-level multiplier so each z-slice is distinct.
The 1-D fields are obtained by slicing the 2-D fields along the middle
row ``j = NY // 2``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
import numpy as np

from finitevolx._src.grid.cartesian import (
    CartesianGrid1D,
    CartesianGrid2D,
    CartesianGrid3D,
)
from finitevolx._src.grid.spherical import (
    SphericalGrid2D,
    SphericalGrid3D,
)
from finitevolx._src.mask import Mask1D, Mask2D, Mask3D

# ----------------------------------------------------------------------
# Domain dimensions — total array shape (interior + 2 ghost cells per axis)
# ----------------------------------------------------------------------

NY_INTERIOR = 16
NX_INTERIOR = 16
NZ_INTERIOR = 4

NY = NY_INTERIOR + 2  # 18
NX = NX_INTERIOR + 2  # 18
NZ = NZ_INTERIOR + 2  # 6

LX = 1.0
LY = 1.0
LZ = 1.0


# ----------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------


def make_grid_1d() -> CartesianGrid1D:
    """Canonical 18 (16 interior) 1-D Cartesian Arakawa C-grid."""
    return CartesianGrid1D.from_interior(NX_INTERIOR, LX)


def make_grid_2d() -> CartesianGrid2D:
    """Canonical 18x18 (16x16 interior) 2-D Cartesian Arakawa C-grid."""
    return CartesianGrid2D.from_interior(NX_INTERIOR, NY_INTERIOR, LX, LY)


def make_grid_3d() -> CartesianGrid3D:
    """Canonical 6x18x18 (4x16x16 interior) 3-D Cartesian Arakawa C-grid."""
    return CartesianGrid3D.from_interior(
        NX_INTERIOR, NY_INTERIOR, NZ_INTERIOR, LX, LY, LZ
    )


# Spherical grids use a small (lon, lat) box centred on the equator so
# the canonical fixtures stay safely away from the poles — cos(lat) ≈ 1
# throughout, so the ``_safe_div_cos`` guard never fires on smooth
# analytic inputs.
_SPHERICAL_LON_RANGE = (-0.1, 0.1)  # radians (~11.5 degrees)
_SPHERICAL_LAT_RANGE = (-0.1, 0.1)


def make_spherical_grid_2d() -> SphericalGrid2D:
    """Canonical 18x18 2-D spherical Arakawa C-grid (near-equator)."""
    return SphericalGrid2D.from_interior(
        nx_interior=NX_INTERIOR,
        ny_interior=NY_INTERIOR,
        lon_range=_SPHERICAL_LON_RANGE,
        lat_range=_SPHERICAL_LAT_RANGE,
    )


def make_spherical_grid_3d() -> SphericalGrid3D:
    """Canonical 6x18x18 3-D spherical Arakawa C-grid (near-equator)."""
    return SphericalGrid3D.from_interior(
        nx_interior=NX_INTERIOR,
        ny_interior=NY_INTERIOR,
        nz_interior=NZ_INTERIOR,
        lon_range=_SPHERICAL_LON_RANGE,
        lat_range=_SPHERICAL_LAT_RANGE,
        Lz=LZ,
    )


# ----------------------------------------------------------------------
# h-masks (cell-centre wet/dry boolean arrays)
# ----------------------------------------------------------------------


def make_h_mask_1d() -> Bool[np.ndarray, "Nx"]:
    """1-D wet/dry mask: outer boundary land + interior dry triple.

    Layout (18 cells, 16 interior + ghost ring of land at each end)::

        L L O O O O L L L O O O O O O O L L
        0 1 2 3 4 5 6 7 8 9 ...

    The interior "L L L" block at indices 6-8 is the 1-D analogue of the
    cross-shaped island in 2-D — it ensures the mask has a non-trivial
    coast/near-coast/ocean classification and a non-trivial stencil
    capability.
    """
    h = np.zeros((NX,), dtype=bool)
    # Interior physical cells start as ocean.
    h[1:-1] = True
    # Outermost interior cell is land (same convention as 2-D).
    h[1] = False
    h[-2] = False
    # Interior dry block — 1-D analogue of the 2-D cross-island.
    h[6:9] = False
    return h


def make_h_mask_2d() -> Bool[np.ndarray, "Ny Nx"]:
    """2-D wet/dry mask: outer boundary land + cross-shaped interior island.

    The mask covers the full ``(NY, NX) = (18, 18)`` array including the
    ghost ring.

    Layout (interior 16x16, plus ghost ring of land all around)::

        L L L L L L L L L L L L L L L L L L
        L L L L L L L L L L L L L L L L L L
        L L O O O O O O O O O O O O O O L L
        L L O O O O O O O O O O O O O O L L
        L L O O O O O O O O O O O O O O L L
        L L O O O O O O O O O O O O O O L L
        L L O O O O O L L L O O O O O O L L
        L L O O O O L L L L L O O O O O L L
        L L O O O O L L L L L O O O O O L L
        L L O O O O L L L L L O O O O O L L
        L L O O O O O L L L O O O O O O L L
        L L O O O O O O O O O O O O O O L L
        L L O O O O O O O O O O O O O O L L
        L L O O O O O O O O O O O O O O L L
        L L O O O O O O O O O O O O O O L L
        L L O O O O O O O O O O O O O O L L
        L L L L L L L L L L L L L L L L L L
        L L L L L L L L L L L L L L L L L L

    The interior island is non-rectangular (cross-shaped) so the derived
    ``mask.xy_corner_strict`` (strict 4-of-4 corner mask) actually differs
    from ``mask.h`` near the corners — this exercises the staggered-mask
    propagation.
    """
    h = np.zeros((NY, NX), dtype=bool)
    # Interior physical cells start as ocean.
    h[1:-1, 1:-1] = True
    # Make the outermost interior ring land too (so dry-on-the-boundary
    # behaviour gets a real test, not just the ghost ring).
    h[1, :] = h[-2, :] = False
    h[:, 1] = h[:, -2] = False
    # Cross-shaped island roughly centred in the basin.
    # Coordinates are in the full (with-ghost) array system.
    h[6:9, 6:11] = False  # 3-row x 5-col horizontal arm
    h[5:10, 7:10] = False  # 5-row x 3-col vertical arm
    return h


def make_h_mask_3d() -> Bool[np.ndarray, "Nz Ny Nx"]:
    """3-D wet/dry mask: same horizontal 2-D mask at every z-level, plus
    land top/bottom caps on the outermost (ghost) z-levels.

    This gives each z-slice the same coastline geometry as the 2-D mask
    while still exercising the vertical-face (``w``) mask — the ghost
    ring at ``k=0`` and ``k=Nz-1`` is dry, so ``mask.w`` is non-trivial
    near the top/bottom.
    """
    h2 = make_h_mask_2d()
    h3 = np.broadcast_to(h2, (NZ, NY, NX)).copy()
    # Top/bottom ghost levels are entirely land.
    h3[0, :, :] = False
    h3[-1, :, :] = False
    return h3


# ----------------------------------------------------------------------
# Mask objects
# ----------------------------------------------------------------------


def make_mask_1d() -> Mask1D:
    """Canonical 1-D :class:`Mask1D` with interior island + coastline."""
    return Mask1D.from_mask(make_h_mask_1d())


def make_mask_2d() -> Mask2D:
    """Canonical 2-D :class:`Mask2D` with cross-shaped island + coastline."""
    return Mask2D.from_mask(make_h_mask_2d())


def make_mask_3d() -> Mask3D:
    """Canonical 3-D :class:`Mask3D` with a cross-shaped island at every
    wet z-level plus land top/bottom caps."""
    return Mask3D.from_mask(make_h_mask_3d())


def make_mask_1d_all_ocean() -> Mask1D:
    """Canonical 1-D all-ocean mask of the same total dimensions."""
    return Mask1D.from_dimensions(nx=NX)


def make_mask_2d_all_ocean() -> Mask2D:
    """Canonical 2-D all-ocean mask of the same total dimensions."""
    return Mask2D.from_dimensions(ny=NY, nx=NX)


def make_mask_3d_all_ocean() -> Mask3D:
    """Canonical 3-D all-ocean mask of the same total dimensions."""
    return Mask3D.from_dimensions(nz=NZ, ny=NY, nx=NX)


# ----------------------------------------------------------------------
# 2-D field generators (primary)
# ----------------------------------------------------------------------


def _xy_2d() -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Normalised (X, Y) coordinate meshes in [0, 1] over the full array."""
    x = jnp.linspace(0.0, 1.0, NX)
    y = jnp.linspace(0.0, 1.0, NY)
    Y, X = jnp.meshgrid(y, x, indexing="ij")
    return X, Y


def make_h_field_2d() -> Float[Array, "Ny Nx"]:
    """T-point scalar field h(x, y) = sin(2πx)·cos(πy) + 0.5(x + y).

    Smooth, distinct in both directions, non-trivial in both first and
    second derivatives.
    """
    X, Y = _xy_2d()
    return jnp.sin(2.0 * jnp.pi * X) * jnp.cos(jnp.pi * Y) + 0.5 * (X + Y)


def make_u_field_2d() -> Float[Array, "Ny Nx"]:
    """U-point velocity field u(x, y) = cos(πx)·sin(2πy)."""
    X, Y = _xy_2d()
    return jnp.cos(jnp.pi * X) * jnp.sin(2.0 * jnp.pi * Y)


def make_v_field_2d() -> Float[Array, "Ny Nx"]:
    """V-point velocity field v(x, y) = -sin(πx)·cos(2πy).

    Roughly divergence-free with the U field above.
    """
    X, Y = _xy_2d()
    return -jnp.sin(jnp.pi * X) * jnp.cos(2.0 * jnp.pi * Y)


def make_q_field_2d() -> Float[Array, "Ny Nx"]:
    """X-point (corner) scalar field q(x, y) = sin(πx)·sin(πy)."""
    X, Y = _xy_2d()
    return jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)


def make_f_field_2d() -> Float[Array, "Ny Nx"]:
    """T-point Coriolis parameter f(x, y) = 1.0 + 0.1·y (β-plane-ish)."""
    _, Y = _xy_2d()
    return 1.0 + 0.1 * Y


# ----------------------------------------------------------------------
# 1-D field generators (slices of the 2-D fields at j = NY // 2)
# ----------------------------------------------------------------------


def make_h_field_1d() -> Float[Array, "Nx"]:
    """T-point scalar field along the middle row of the 2-D h-field."""
    return make_h_field_2d()[NY // 2, :]


def make_u_field_1d() -> Float[Array, "Nx"]:
    """U-point velocity field along the middle row of the 2-D u-field."""
    return make_u_field_2d()[NY // 2, :]


# ----------------------------------------------------------------------
# 3-D field generators (broadcast 2-D fields with per-level multipliers)
# ----------------------------------------------------------------------


def _z_multipliers() -> Float[Array, "Nz"]:
    """Per-level multipliers so each z-slice has distinct values."""
    return jnp.linspace(1.0, 1.5, NZ)


def make_h_field_3d() -> Float[Array, "Nz Ny Nx"]:
    return _z_multipliers()[:, None, None] * make_h_field_2d()[None, :, :]


def make_u_field_3d() -> Float[Array, "Nz Ny Nx"]:
    return _z_multipliers()[:, None, None] * make_u_field_2d()[None, :, :]


def make_v_field_3d() -> Float[Array, "Nz Ny Nx"]:
    return _z_multipliers()[:, None, None] * make_v_field_2d()[None, :, :]


# ----------------------------------------------------------------------
# Convenience: a single dict bundle for tests that just want everything.
# ----------------------------------------------------------------------


def all_1d_fields() -> dict[str, Float[Array, "Nx"]]:
    """Bundle of every canonical 1-D field, keyed by stagger label."""
    return {
        "h": make_h_field_1d(),
        "u": make_u_field_1d(),
    }


def all_2d_fields() -> dict[str, Float[Array, "Ny Nx"]]:
    """Bundle of every canonical 2-D field, keyed by stagger label.

    Returns a dict with keys ``h``, ``u``, ``v``, ``q``, ``f`` whose
    values are JAX arrays of shape ``(NY, NX)``.
    """
    return {
        "h": make_h_field_2d(),
        "u": make_u_field_2d(),
        "v": make_v_field_2d(),
        "q": make_q_field_2d(),
        "f": make_f_field_2d(),
    }


def all_3d_fields() -> dict[str, Float[Array, "Nz Ny Nx"] | Float[Array, "Ny Nx"]]:
    """Bundle of every canonical 3-D field.

    The Coriolis parameter ``f`` stays 2-D — that is the depth-independent
    convention used by ``Coriolis3D`` and the rest of the 3-D operator
    suite.
    """
    return {
        "h": make_h_field_3d(),
        "u": make_u_field_3d(),
        "v": make_v_field_3d(),
        "f": make_f_field_2d(),
    }
