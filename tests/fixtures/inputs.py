"""Canonical input fixtures for finitevolX golden-output regression tests.

This module exposes a single deterministic 2-D test domain plus a single
deterministic :class:`~finitevolx.ArakawaCGridMask` with an island and a
coastline.  All "masks-everywhere" regression tests share these inputs so
that:

* Every method under test sees the same field values, making it easy to
  cross-check operators that are expected to agree on smooth interiors.
* The mask geometry is interesting enough to exercise both interior and
  boundary behavior — wet T-cells adjacent to land, dry interior cells,
  and a multi-cell island.
* The 18x18 total size (16x16 interior + ghost ring) keeps golden
  ``.npz`` files small enough to commit and diff comfortably.

The fields are smooth analytic functions (polynomials and trigs) — they
are deterministic, easy to reason about, and don't change between
``jax.random`` versions.

Both 2-D and 3-D versions are provided.  The 3-D fields are obtained by
broadcasting the 2-D ones over a small ``Nz`` axis with a per-level
multiplier so each z-slice is distinct.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
import numpy as np

from finitevolx._src.grid.cgrid_mask import ArakawaCGridMask
from finitevolx._src.grid.grid import ArakawaCGrid2D, ArakawaCGrid3D

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
# Grid and mask
# ----------------------------------------------------------------------


def make_grid_2d() -> ArakawaCGrid2D:
    """Canonical 18x18 (16x16 interior) 2-D Arakawa C-grid."""
    return ArakawaCGrid2D.from_interior(NX_INTERIOR, NY_INTERIOR, LX, LY)


def make_grid_3d() -> ArakawaCGrid3D:
    """Canonical 6x18x18 (4x16x16 interior) 3-D Arakawa C-grid."""
    return ArakawaCGrid3D.from_interior(
        NX_INTERIOR, NY_INTERIOR, NZ_INTERIOR, LX, LY, LZ
    )


def make_h_mask_2d() -> Bool[np.ndarray, "Ny Nx"]:
    """Cell-centre wet/dry mask: outer boundary land + cross-shaped island.

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

    The interior island is non-rectangular (cross-shaped) so the
    derived ``mask.psi`` (strict 4-of-4 corner mask) actually differs
    from ``mask.h`` near the corners — exercises the staggered-mask
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
    """Same horizontal mask broadcast over all z-levels."""
    h2 = make_h_mask_2d()
    return np.broadcast_to(h2, (NZ, NY, NX)).copy()


def make_mask_2d() -> ArakawaCGridMask:
    """Canonical 2-D :class:`ArakawaCGridMask` with island + coastline."""
    return ArakawaCGridMask.from_mask(make_h_mask_2d())


def make_mask_2d_all_ocean() -> ArakawaCGridMask:
    """Canonical 2-D all-ocean mask of the same total dimensions."""
    return ArakawaCGridMask.from_dimensions(ny=NY, nx=NX)


# ----------------------------------------------------------------------
# Field generators
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
# 3-D variants
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
