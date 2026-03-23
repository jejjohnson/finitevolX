"""
Face-value reconstruction operators for advective fluxes on Arakawa C-grids.

All operators follow the interior-point idiom (write only [1:-1, 1:-1]).

Notation
--------
  fe[j, i+1/2]  east-face flux
  fn[j+1/2, i]  north-face flux
"""

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.advection.flux import upwind_flux
from finitevolx._src.advection.limiters import mc, minmod, superbee, van_leer
from finitevolx._src.advection.weno import (
    weno_3pts as _weno3,
    weno_3pts_improved as _wenoz3,
    weno_3pts_improved_right as _wenoz3_right,
    weno_3pts_right as _weno3_right,
    weno_5pts as _weno5,
    weno_5pts_improved as _wenoz5,
    weno_5pts_improved_right as _wenoz5_right,
    weno_5pts_right as _weno5_right,
    weno_7pts as _weno7,
    weno_7pts_right as _weno7_right,
    weno_9pts as _weno9,
    weno_9pts_right as _weno9_right,
)
from finitevolx._src.grid.cgrid_mask import ArakawaCGridMask
from finitevolx._src.grid.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.operators._ghost import interior

# Small epsilon to avoid division by zero in TVD slope ratios.
_TVD_EPS: float = 1e-8

# Mapping from limiter name to callable for TVD reconstruction.
_LIMITERS = {
    "minmod": minmod,
    "van_leer": van_leer,
    "superbee": superbee,
    "mc": mc,
}


def _get_limiter(name: str):
    """Return the flux limiter callable for *name*.

    Parameters
    ----------
    name : str
        One of ``'minmod'``, ``'van_leer'``, ``'superbee'``, ``'mc'``.

    Raises
    ------
    ValueError
        If *name* is not a known limiter.
    """
    if name not in _LIMITERS:
        raise ValueError(
            f"Unknown flux limiter {name!r}. Choose one of {list(_LIMITERS)!r}."
        )
    return _LIMITERS[name]


def _weno7_positive_last_axis(h: Array) -> Array:
    h3_first = _weno3(h[..., 0:1], h[..., 1:2], h[..., 2:3])
    h5_second = _weno5(h[..., 0:1], h[..., 1:2], h[..., 2:3], h[..., 3:4], h[..., 4:5])
    h7_interior = _weno7(
        h[..., :-6],
        h[..., 1:-5],
        h[..., 2:-4],
        h[..., 3:-3],
        h[..., 4:-2],
        h[..., 5:-1],
        h[..., 6:],
    )
    h5_penultimate = _weno5(
        h[..., -5:-4],
        h[..., -4:-3],
        h[..., -3:-2],
        h[..., -2:-1],
        h[..., -1:],
    )
    h3_last = _weno3(h[..., -3:-2], h[..., -2:-1], h[..., -1:])
    return jnp.concatenate(
        [h3_first, h5_second, h7_interior, h5_penultimate, h3_last], axis=-1
    )


def _weno7_negative_last_axis(h: Array) -> Array:
    # Right-biased face reconstruction for negative flow.
    # Faces at positions 1.5, 2.5, ..., N-1.5 (N-2 total).
    # Boundary fallbacks are on the RIGHT side (the upwind boundary for u<0).
    h5_first = _weno5_right(
        h[..., 0:1], h[..., 1:2], h[..., 2:3], h[..., 3:4], h[..., 4:5]
    )
    h7_interior = _weno7_right(
        h[..., :-6],
        h[..., 1:-5],
        h[..., 2:-4],
        h[..., 3:-3],
        h[..., 4:-2],
        h[..., 5:-1],
        h[..., 6:],
    )
    h5_penultimate = _weno5_right(
        h[..., -5:-4],
        h[..., -4:-3],
        h[..., -3:-2],
        h[..., -2:-1],
        h[..., -1:],
    )
    h3_second_last = _weno3_right(h[..., -3:-2], h[..., -2:-1], h[..., -1:])
    h1_last = h[..., -1:]
    return jnp.concatenate(
        [h5_first, h7_interior, h5_penultimate, h3_second_last, h1_last],
        axis=-1,
    )


def _weno9_positive_last_axis(h: Array) -> Array:
    h3_first = _weno3(h[..., 0:1], h[..., 1:2], h[..., 2:3])
    h5_second = _weno5(h[..., 0:1], h[..., 1:2], h[..., 2:3], h[..., 3:4], h[..., 4:5])
    h7_third = _weno7(
        h[..., 0:1],
        h[..., 1:2],
        h[..., 2:3],
        h[..., 3:4],
        h[..., 4:5],
        h[..., 5:6],
        h[..., 6:7],
    )
    h9_interior = _weno9(
        h[..., :-8],
        h[..., 1:-7],
        h[..., 2:-6],
        h[..., 3:-5],
        h[..., 4:-4],
        h[..., 5:-3],
        h[..., 6:-2],
        h[..., 7:-1],
        h[..., 8:],
    )
    h7_third_last = _weno7(
        h[..., -7:-6],
        h[..., -6:-5],
        h[..., -5:-4],
        h[..., -4:-3],
        h[..., -3:-2],
        h[..., -2:-1],
        h[..., -1:],
    )
    h5_second_last = _weno5(
        h[..., -5:-4],
        h[..., -4:-3],
        h[..., -3:-2],
        h[..., -2:-1],
        h[..., -1:],
    )
    h3_last = _weno3(h[..., -3:-2], h[..., -2:-1], h[..., -1:])
    return jnp.concatenate(
        [
            h3_first,
            h5_second,
            h7_third,
            h9_interior,
            h7_third_last,
            h5_second_last,
            h3_last,
        ],
        axis=-1,
    )


def _weno9_negative_last_axis(h: Array) -> Array:
    # Right-biased face reconstruction for negative flow.
    # Faces at positions 1.5, 2.5, ..., N-1.5 (N-2 total).
    # Boundary fallbacks are on the RIGHT side (the upwind boundary for u<0).
    h5_first = _weno5_right(
        h[..., 0:1], h[..., 1:2], h[..., 2:3], h[..., 3:4], h[..., 4:5]
    )
    h7_second = _weno7_right(
        h[..., 0:1],
        h[..., 1:2],
        h[..., 2:3],
        h[..., 3:4],
        h[..., 4:5],
        h[..., 5:6],
        h[..., 6:7],
    )
    h9_interior = _weno9_right(
        h[..., :-8],
        h[..., 1:-7],
        h[..., 2:-6],
        h[..., 3:-5],
        h[..., 4:-4],
        h[..., 5:-3],
        h[..., 6:-2],
        h[..., 7:-1],
        h[..., 8:],
    )
    h7_third_last = _weno7_right(
        h[..., -7:-6],
        h[..., -6:-5],
        h[..., -5:-4],
        h[..., -4:-3],
        h[..., -3:-2],
        h[..., -2:-1],
        h[..., -1:],
    )
    h5_second_last = _weno5_right(
        h[..., -5:-4],
        h[..., -4:-3],
        h[..., -3:-2],
        h[..., -2:-1],
        h[..., -1:],
    )
    h3_third_last = _weno3_right(h[..., -3:-2], h[..., -2:-1], h[..., -1:])
    h1_last = h[..., -1:]
    return jnp.concatenate(
        [
            h5_first,
            h7_second,
            h9_interior,
            h7_third_last,
            h5_second_last,
            h3_third_last,
            h1_last,
        ],
        axis=-1,
    )


def _weno_last_axis_flux(h: Array, velocity: Array, order: int) -> Array:
    if order == 7:
        h_pos = _weno7_positive_last_axis(h)
        h_neg = _weno7_negative_last_axis(h)
    elif order == 9:
        h_pos = _weno9_positive_last_axis(h)
        h_neg = _weno9_negative_last_axis(h)
    else:
        raise ValueError(f"Unsupported WENO order: {order}")
    return jnp.where(velocity >= 0.0, h_pos, h_neg) * velocity


def _weno_flux_axis_1d(h: Array, velocity: Array, order: int) -> Array:
    out = interior(_weno_last_axis_flux(h, velocity[1:-1], order), h)
    return out


def _weno_flux_axis_2d_x(h: Array, velocity: Array, order: int) -> Array:
    out = interior(_weno_last_axis_flux(h[1:-1, :], velocity[1:-1, 1:-1], order), h)
    return out


def _weno_flux_axis_2d_y(h: Array, velocity: Array, order: int) -> Array:
    h_last = jnp.swapaxes(h[:, 1:-1], 0, 1)
    v_last = jnp.swapaxes(velocity[1:-1, 1:-1], 0, 1)
    flux = _weno_last_axis_flux(h_last, v_last, order)
    out = interior(jnp.swapaxes(flux, 0, 1), h)
    return out


def _weno_flux_axis_3d_x(h: Array, velocity: Array, order: int) -> Array:
    flux = _weno_last_axis_flux(h[1:-1, 1:-1, :], velocity[1:-1, 1:-1, 1:-1], order)
    out = interior(flux, h)
    return out


def _weno_flux_axis_3d_y(h: Array, velocity: Array, order: int) -> Array:
    h_last = jnp.moveaxis(h[1:-1, :, 1:-1], 1, -1)
    v_last = jnp.moveaxis(velocity[1:-1, 1:-1, 1:-1], 1, -1)
    flux = _weno_last_axis_flux(h_last, v_last, order)
    out = interior(jnp.moveaxis(flux, -1, 1), h)
    return out


class Reconstruction1D(eqx.Module):
    """1-D face-value reconstruction.

    Parameters
    ----------
    grid : ArakawaCGrid1D
    """

    grid: ArakawaCGrid1D

    def naive_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """Naive (centred) reconstruction at east face.

        fe[i+1/2] = 1/2 * (h[i] + h[i+1]) * u[i+1/2]
        """
        out = interior(0.5 * (h[1:-1] + h[2:]) * u[1:-1], h)
        return out

    def upwind1_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """1st-order upwind reconstruction at east face.

        fe[i+1/2] = h[i]   * u[i+1/2]   if u[i+1/2] >= 0
                  = h[i+1] * u[i+1/2]   otherwise
        """
        # fe[i+1/2] = upwind(h, u) * u[i+1/2]
        h_face = jnp.where(u[1:-1] >= 0.0, h[1:-1], h[2:])
        out = interior(h_face * u[1:-1], h)
        return out

    def upwind2_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """2nd-order upwind reconstruction at east face with boundary fallback.

        Positive flow:  h_face[i+1/2] = 3/2*h[i]   - 1/2*h[i-1]
        Negative flow:  h_face[i+1/2] = 3/2*h[i+1] - 1/2*h[i+2]
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        # h_face_pos[i+1/2] = 3/2*h[i] - 1/2*h[i-1]
        h_pos = 1.5 * h[1:-1] - 0.5 * h[:-2]
        # h_face_neg[i+1/2] = 3/2*h[i+1] - 1/2*h[i+2]
        # Use 2nd-order where i+2 is available (all except last interior face)
        h_neg_interior = 1.5 * h[2:-1] - 0.5 * h[3:]
        # Use 1st-order upwind on east boundary face (h_face = h[i+1])
        h_neg_boundary = h[-1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary])
        h_face = jnp.where(u[1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1], h)
        return out

    def upwind3_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """3rd-order upwind reconstruction at east face with boundary fallback.

        Positive flow:  h_face = -1/6*h[i-1] + 5/6*h[i]   + 1/3*h[i+1]
        Negative flow:  h_face =  1/3*h[i]   + 5/6*h[i+1] - 1/6*h[i+2]
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        # 3rd-order positive stencil (valid for all interior faces)
        h_pos = (
            -1.0 / 6.0 * h[:-2]  # h[i-1]
            + 5.0 / 6.0 * h[1:-1]  # h[i  ]
            + 1.0 / 3.0 * h[2:]  # h[i+1]
        )
        # 3rd-order negative stencil (valid only where h[i+2] exists)
        # For all interior faces except the last one
        h_neg_interior = (
            1.0 / 3.0 * h[1:-2]  # h[i  ]
            + 5.0 / 6.0 * h[2:-1]  # h[i+1]
            - 1.0 / 6.0 * h[3:]  # h[i+2]
        )
        # 1st-order upwind fallback at east boundary: h_face = h[i+1]
        h_neg_boundary = h[-1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary])
        h_face = jnp.where(u[1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1], h)
        return out

    def weno3_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """3-point WENO east-face flux with boundary fallback.

        Positive flow:  h_face[i+1/2] = WENO3(h[i-1], h[i],   h[i+1])
        Negative flow:  h_face[i+1/2] = WENO3_right(h[i], h[i+1], h[i+2])
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        # WENO-3 positive: left-biased stencil, valid for all interior faces
        h_pos = _weno3(h[:-2], h[1:-1], h[2:])
        # WENO-3 negative: right-biased stencil, valid for i+2 < Nx
        h_neg_interior = _weno3_right(h[1:-2], h[2:-1], h[3:])
        # 1st-order upwind fallback at east boundary
        h_neg_boundary = h[-1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary])
        h_face = jnp.where(u[1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1], h)
        return out

    def wenoz3_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """3-point WENO-Z east-face flux with boundary fallback.

        Positive flow:  h_face[i+1/2] = WENOZ3(h[i-1], h[i],   h[i+1])
        Negative flow:  h_face[i+1/2] = WENOZ3_right(h[i], h[i+1], h[i+2])
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        # WENO-Z-3 positive: left-biased stencil, valid for all interior faces
        h_pos = _wenoz3(h[:-2], h[1:-1], h[2:])
        # WENO-Z-3 negative: right-biased stencil, valid for i+2 < Nx
        h_neg_interior = _wenoz3_right(h[1:-2], h[2:-1], h[3:])
        # 1st-order upwind fallback at east boundary
        h_neg_boundary = h[-1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary])
        h_face = jnp.where(u[1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1], h)
        return out

    def weno5_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """5-point WENO east-face flux with sign-dependent boundary fallbacks.

        Positive flow:  h_face[i+1/2] = WENO5(h[i-2], h[i-1], h[i], h[i+1], h[i+2])
            for i = 2..Nx-3; WENO3 fallback at i = 1 and i = Nx-2.
        Negative flow:  h_face[i+1/2] = WENO5_right(h[i-1], h[i], h[i+1], h[i+2], h[i+3])
            for i = 1..Nx-4; WENO3_right fallback at i = Nx-3; 1st-order upwind at i = Nx-2.
        """
        # WENO-5 positive: valid for interior faces i=2..Nx-3 (needs h[i-2..i+2])
        h5_pos_interior = _weno5(h[:-4], h[1:-3], h[2:-2], h[3:-1], h[4:])
        # WENO-3 fallback at first interior face (i=1, h[i-2] = h[-1] wraps)
        h3_pos_first = _weno3(h[0:1], h[1:2], h[2:3])
        # WENO-3 fallback at last interior face (i=Nx-2, h[i+2] = h[Nx] out of bounds)
        h3_pos_last = _weno3(h[-3:-2], h[-2:-1], h[-1:])
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_interior, h3_pos_last])
        # WENO-5 right-biased: valid for faces i=1..Nx-4
        h5_neg_interior = _weno5_right(h[:-4], h[1:-3], h[2:-2], h[3:-1], h[4:])
        # WENO-3 right-biased fallback at face i=Nx-3
        h3_neg_penultimate = _weno3_right(h[-3:-2], h[-2:-1], h[-1:])
        # 1st-order upwind fallback at east boundary (i=Nx-2)
        h1_neg_last = h[-1:]
        h_neg = jnp.concatenate([h5_neg_interior, h3_neg_penultimate, h1_neg_last])
        h_face = jnp.where(u[1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1], h)
        return out

    def wenoz5_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """5-point WENO-Z east-face flux with sign-dependent boundary fallbacks.

        Positive flow:  h_face[i+1/2] = WENOZ5(h[i-2], h[i-1], h[i], h[i+1], h[i+2])
            for i = 2..Nx-3; WENO-Z-3 fallback at i = 1 and i = Nx-2.
        Negative flow:  h_face[i+1/2] = WENOZ5_right(h[i-1], h[i], h[i+1], h[i+2], h[i+3])
            for i = 1..Nx-4; WENO-Z-3_right fallback at i = Nx-3; 1st-order upwind at i = Nx-2.
        """
        # WENO-Z-5 positive: valid for interior faces i=2..Nx-3
        h5_pos_interior = _wenoz5(h[:-4], h[1:-3], h[2:-2], h[3:-1], h[4:])
        # WENO-Z-3 fallback at first interior face (i=1)
        h3_pos_first = _wenoz3(h[0:1], h[1:2], h[2:3])
        # WENO-Z-3 fallback at last interior face (i=Nx-2)
        h3_pos_last = _wenoz3(h[-3:-2], h[-2:-1], h[-1:])
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_interior, h3_pos_last])
        # WENO-Z-5 right-biased: valid for faces i=1..Nx-4
        h5_neg_interior = _wenoz5_right(h[:-4], h[1:-3], h[2:-2], h[3:-1], h[4:])
        # WENO-Z-3 right-biased fallback at face i=Nx-3
        h3_neg_penultimate = _wenoz3_right(h[-3:-2], h[-2:-1], h[-1:])
        # 1st-order upwind fallback at east boundary (i=Nx-2)
        h1_neg_last = h[-1:]
        h_neg = jnp.concatenate([h5_neg_interior, h3_neg_penultimate, h1_neg_last])
        h_face = jnp.where(u[1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1], h)
        return out

    def weno7_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """7-point WENO east-face flux with hierarchical boundary fallbacks.

        Positive flow:  h_face[i+1/2] = WENO7(h[i-3..i+3])
            for i = 3..Nx-4; WENO5 fallback at i = 2 and i = Nx-3;
            WENO3 fallback at i = 1 and i = Nx-2.
        Negative flow:  h_face[i+1/2] = WENO7(h[i+3..i-3])
            for i = 3..Nx-4; WENO3 fallback at i = 1; WENO5 fallback
            at i = 2 and i = Nx-3; 1st-order upwind at i = Nx-2.
        """
        return _weno_flux_axis_1d(h, u, order=7)

    def weno9_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """9-point WENO east-face flux with hierarchical boundary fallbacks.

        Positive flow:  h_face[i+1/2] = WENO9(h[i-4..i+4])
            for i = 4..Nx-5; WENO7 fallback at i = 3 and i = Nx-4;
            WENO5 fallback at i = 2 and i = Nx-3; WENO3 fallback at
            i = 1 and i = Nx-2.
        Negative flow:  h_face[i+1/2] = WENO9(h[i+4..i-4])
            for i = 4..Nx-5; WENO7 fallback at i = 3 and i = Nx-4;
            WENO5 fallback at i = 2 and i = Nx-3; WENO3 fallback at
            i = 1; 1st-order upwind at i = Nx-2.
        """
        return _weno_flux_axis_1d(h, u, order=9)

    def tvd_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
        limiter: str = "minmod",
    ) -> Float[Array, "Nx"]:
        """TVD east-face flux using a flux limiter.

        Blends 1st-order upwind with an anti-diffusive correction limited by
        φ(r) to preserve monotonicity:

        Positive flow:
            r     = (h[i] − h[i−1]) / (h[i+1] − h[i])
            h_L   = h[i]   + ½ φ(r) (h[i+1] − h[i])

        Negative flow:
            r     = (h[i+1] − h[i+2]) / (h[i] − h[i+1])
            h_R   = h[i+1] + ½ φ(r) (h[i]   − h[i+1])
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.

        Parameters
        ----------
        h : Float[Array, "Nx"]
            Scalar at T-points.
        u : Float[Array, "Nx"]
            Velocity at U-points.
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Nx"]
            East-face flux with zero ghost entries.
        """
        phi = _get_limiter(limiter)
        # Positive flow: h_face = h[i] + 0.5*phi(r)*(h[i+1]-h[i])
        # r = (h[i]-h[i-1]) / (h[i+1]-h[i])  for i=1..Nx-2
        diff_fwd = h[2:] - h[1:-1]  # h[i+1] - h[i]
        diff_bwd = h[1:-1] - h[:-2]  # h[i] - h[i-1]
        r_pos = diff_bwd / (diff_fwd + _TVD_EPS)
        h_pos = h[1:-1] + 0.5 * phi(r_pos) * diff_fwd
        # Negative flow: h_face = h[i+1] + 0.5*phi(r)*(h[i]-h[i+1])
        # r = (h[i+1]-h[i+2]) / (h[i]-h[i+1])  for i=1..Nx-3
        diff_neg = h[1:-2] - h[2:-1]  # h[i] - h[i+1]
        diff_neg2 = h[2:-1] - h[3:]  # h[i+1] - h[i+2]
        r_neg_int = diff_neg2 / (diff_neg + _TVD_EPS)
        h_neg_interior = h[2:-1] + 0.5 * phi(r_neg_int) * diff_neg
        # 1st-order upwind fallback at east boundary (i=Nx-2)
        h_neg_boundary = h[-1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary])
        h_face = jnp.where(u[1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1], h)
        return out


class Reconstruction2D(eqx.Module):
    """2-D face-value reconstruction for advective fluxes.

    Parameters
    ----------
    grid : ArakawaCGrid2D
    """

    grid: ArakawaCGrid2D

    def naive_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Naive (centred) east-face flux.

        fe[j, i+1/2] = 1/2 * (h[j, i] + h[j, i+1]) * u[j, i+1/2]
        """
        out = interior(0.5 * (h[1:-1, 1:-1] + h[1:-1, 2:]) * u[1:-1, 1:-1], h)
        return out

    def naive_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Naive (centred) north-face flux.

        fn[j+1/2, i] = 1/2 * (h[j, i] + h[j+1, i]) * v[j+1/2, i]
        """
        out = interior(0.5 * (h[1:-1, 1:-1] + h[2:, 1:-1]) * v[1:-1, 1:-1], h)
        return out

    def upwind1_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """1st-order upwind east-face flux.

        fe[j, i+1/2] = h[j, i]   * u  if u >= 0
                     = h[j, i+1] * u  otherwise
        """
        # fe[j, i+1/2] = upwind1(h, u) * u[j, i+1/2]
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h[1:-1, 1:-1], h[1:-1, 2:])
        out = interior(h_face * u[1:-1, 1:-1], h)
        return out

    def upwind1_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """1st-order upwind north-face flux.

        fn[j+1/2, i] = h[j, i]   * v  if v >= 0
                     = h[j+1, i] * v  otherwise
        """
        # fn[j+1/2, i] = upwind1(h, v) * v[j+1/2, i]
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h[1:-1, 1:-1], h[2:, 1:-1])
        out = interior(h_face * v[1:-1, 1:-1], h)
        return out

    def upwind2_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """2nd-order upwind east-face flux with boundary fallback.

        Positive: h_face = 3/2*h[j,i] - 1/2*h[j,i-1]
        Negative: h_face = 3/2*h[j,i+1] - 1/2*h[j,i+2]
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        # 2nd-order positive stencil (uses h[j,i-1])
        h_pos = 1.5 * h[1:-1, 1:-1] - 0.5 * h[1:-1, :-2]
        # 2nd-order negative stencil (uses h[j,i+2])
        # Use 2nd-order where i+2 is available (all except last interior column)
        h_neg_interior = 1.5 * h[1:-1, 2:-1] - 0.5 * h[1:-1, 3:]
        # Use 1st-order upwind on east boundary column (h_face = h[j,i+1])
        h_neg_boundary = h[1:-1, -1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=1)
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1, 1:-1], h)
        return out

    def upwind2_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """2nd-order upwind north-face flux with boundary fallback.

        Positive: h_face = 3/2*h[j,i] - 1/2*h[j-1,i]
        Negative: h_face = 3/2*h[j+1,i] - 1/2*h[j+2,i]
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        # 2nd-order positive stencil (uses h[j-1,i])
        h_pos = 1.5 * h[1:-1, 1:-1] - 0.5 * h[:-2, 1:-1]
        # 2nd-order negative stencil (uses h[j+2,i])
        # Use 2nd-order where j+2 is available (all except last interior row)
        h_neg_interior = 1.5 * h[2:-1, 1:-1] - 0.5 * h[3:, 1:-1]
        # Use 1st-order upwind on north boundary row (h_face = h[j+1,i])
        h_neg_boundary = h[-1:, 1:-1]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=0)
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * v[1:-1, 1:-1], h)
        return out

    def upwind3_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3rd-order upwind east-face flux with boundary fallback.

        Positive: h_face = -1/6*h[j,i-1] + 5/6*h[j,i]   + 1/3*h[j,i+1]
        Negative: h_face =  1/3*h[j,i]   + 5/6*h[j,i+1] - 1/6*h[j,i+2]
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        # 3rd-order positive stencil (valid for all interior faces)
        h_pos = (
            -1.0 / 6.0 * h[1:-1, :-2]  # h[j, i-1]
            + 5.0 / 6.0 * h[1:-1, 1:-1]  # h[j, i  ]
            + 1.0 / 3.0 * h[1:-1, 2:]  # h[j, i+1]
        )
        # 3rd-order negative stencil (valid only where h[j,i+2] exists)
        # For all interior columns except the last one
        h_neg_interior = (
            1.0 / 3.0 * h[1:-1, 1:-2]  # h[j, i  ]
            + 5.0 / 6.0 * h[1:-1, 2:-1]  # h[j, i+1]
            - 1.0 / 6.0 * h[1:-1, 3:]  # h[j, i+2]
        )
        # 1st-order upwind fallback at east boundary: h_face = h[j,i+1]
        h_neg_boundary = h[1:-1, -1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=1)
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1, 1:-1], h)
        return out

    def upwind3_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3rd-order upwind north-face flux with boundary fallback.

        Positive: h_face = -1/6*h[j-1,i] + 5/6*h[j,i]   + 1/3*h[j+1,i]
        Negative: h_face =  1/3*h[j,i]   + 5/6*h[j+1,i] - 1/6*h[j+2,i]
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        # 3rd-order positive stencil (valid for all interior faces)
        h_pos = (
            -1.0 / 6.0 * h[:-2, 1:-1]  # h[j-1, i]
            + 5.0 / 6.0 * h[1:-1, 1:-1]  # h[j,   i]
            + 1.0 / 3.0 * h[2:, 1:-1]  # h[j+1, i]
        )
        # 3rd-order negative stencil (valid only where h[j+2,i] exists)
        # For all interior rows except the last one
        h_neg_interior = (
            1.0 / 3.0 * h[1:-2, 1:-1]  # h[j,   i]
            + 5.0 / 6.0 * h[2:-1, 1:-1]  # h[j+1, i]
            - 1.0 / 6.0 * h[3:, 1:-1]  # h[j+2, i]
        )
        # 1st-order upwind fallback at north boundary: h_face = h[j+1,i]
        h_neg_boundary = h[-1:, 1:-1]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=0)
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * v[1:-1, 1:-1], h)
        return out

    def weno3_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3-point WENO east-face flux with boundary fallback.

        Positive flow:  fe[j,i+1/2] = WENO3(h[j,i-1], h[j,i],   h[j,i+1]) * u
        Negative flow:  fe[j,i+1/2] = WENO3_right(h[j,i], h[j,i+1], h[j,i+2]) * u
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        # WENO-3 positive: left-biased, valid for all interior faces
        h_pos = _weno3(h[1:-1, :-2], h[1:-1, 1:-1], h[1:-1, 2:])
        # WENO-3 negative: right-biased, valid for i+2 < Nx
        h_neg_interior = _weno3_right(h[1:-1, 1:-2], h[1:-1, 2:-1], h[1:-1, 3:])
        # 1st-order upwind fallback at east boundary column
        h_neg_boundary = h[1:-1, -1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=1)
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1, 1:-1], h)
        return out

    def weno3_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3-point WENO north-face flux with boundary fallback.

        Positive flow:  fn[j+1/2,i] = WENO3(h[j-1,i], h[j,i],   h[j+1,i]) * v
        Negative flow:  fn[j+1/2,i] = WENO3_right(h[j,i], h[j+1,i], h[j+2,i]) * v
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        # WENO-3 positive: left-biased, valid for all interior faces
        h_pos = _weno3(h[:-2, 1:-1], h[1:-1, 1:-1], h[2:, 1:-1])
        # WENO-3 negative: right-biased, valid for j+2 < Ny
        h_neg_interior = _weno3_right(h[1:-2, 1:-1], h[2:-1, 1:-1], h[3:, 1:-1])
        # 1st-order upwind fallback at north boundary row
        h_neg_boundary = h[-1:, 1:-1]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=0)
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * v[1:-1, 1:-1], h)
        return out

    def wenoz3_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3-point WENO-Z east-face flux with boundary fallback.

        Positive flow:  fe[j,i+1/2] = WENOZ3(h[j,i-1], h[j,i],   h[j,i+1]) * u
        Negative flow:  fe[j,i+1/2] = WENOZ3_right(h[j,i], h[j,i+1], h[j,i+2]) * u
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        # WENO-Z-3 positive: left-biased, valid for all interior faces
        h_pos = _wenoz3(h[1:-1, :-2], h[1:-1, 1:-1], h[1:-1, 2:])
        # WENO-Z-3 negative: right-biased, valid for i+2 < Nx
        h_neg_interior = _wenoz3_right(h[1:-1, 1:-2], h[1:-1, 2:-1], h[1:-1, 3:])
        # 1st-order upwind fallback at east boundary column
        h_neg_boundary = h[1:-1, -1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=1)
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1, 1:-1], h)
        return out

    def wenoz3_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3-point WENO-Z north-face flux with boundary fallback.

        Positive flow:  fn[j+1/2,i] = WENOZ3(h[j-1,i], h[j,i],   h[j+1,i]) * v
        Negative flow:  fn[j+1/2,i] = WENOZ3_right(h[j,i], h[j+1,i], h[j+2,i]) * v
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        # WENO-Z-3 positive: left-biased, valid for all interior faces
        h_pos = _wenoz3(h[:-2, 1:-1], h[1:-1, 1:-1], h[2:, 1:-1])
        # WENO-Z-3 negative: right-biased, valid for j+2 < Ny
        h_neg_interior = _wenoz3_right(h[1:-2, 1:-1], h[2:-1, 1:-1], h[3:, 1:-1])
        # 1st-order upwind fallback at north boundary row
        h_neg_boundary = h[-1:, 1:-1]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=0)
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * v[1:-1, 1:-1], h)
        return out

    def weno5_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO east-face flux with sign-dependent boundary fallbacks.

        Positive flow:  fe[j,i+1/2] = WENO5(h[j,i-2..i+2]) * u
            for i = 2..Nx-3; WENO3 fallback at i = 1 and i = Nx-2.
        Negative flow:  fe[j,i+1/2] = WENO5_right(h[j,i-1], h[j,i], h[j,i+1], h[j,i+2], h[j,i+3]) * u
            for i = 1..Nx-4; WENO3_right fallback at i = Nx-3; 1st-order upwind at i = Nx-2.
        """
        # WENO-5 positive: valid for i=2..Nx-3
        h5_pos_int = _weno5(
            h[1:-1, :-4], h[1:-1, 1:-3], h[1:-1, 2:-2], h[1:-1, 3:-1], h[1:-1, 4:]
        )
        # WENO-3 fallback at first interior column (i=1)
        h3_pos_first = _weno3(h[1:-1, 0:1], h[1:-1, 1:2], h[1:-1, 2:3])
        # WENO-3 fallback at last interior column (i=Nx-2)
        h3_pos_last = _weno3(h[1:-1, -3:-2], h[1:-1, -2:-1], h[1:-1, -1:])
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=1)
        # WENO-5 right-biased: valid for i=1..Nx-4
        h5_neg_int = _weno5_right(
            h[1:-1, :-4], h[1:-1, 1:-3], h[1:-1, 2:-2], h[1:-1, 3:-1], h[1:-1, 4:]
        )
        # WENO-3 right-biased fallback at column i=Nx-3
        h3_neg_penultimate = _weno3_right(h[1:-1, -3:-2], h[1:-1, -2:-1], h[1:-1, -1:])
        # 1st-order upwind fallback at east boundary column (i=Nx-2)
        h1_neg_last = h[1:-1, -1:]
        h_neg = jnp.concatenate([h5_neg_int, h3_neg_penultimate, h1_neg_last], axis=1)
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1, 1:-1], h)
        return out

    def weno5_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO north-face flux with sign-dependent boundary fallbacks.

        Positive flow:  fn[j+1/2,i] = WENO5(h[j-2..j+2,i]) * v
            for j = 2..Ny-3; WENO3 fallback at j = 1 and j = Ny-2.
        Negative flow:  fn[j+1/2,i] = WENO5_right(h[j-1,i], h[j,i], h[j+1,i], h[j+2,i], h[j+3,i]) * v
            for j = 1..Ny-4; WENO3_right fallback at j = Ny-3; 1st-order upwind at j = Ny-2.
        """
        # WENO-5 positive: valid for j=2..Ny-3
        h5_pos_int = _weno5(
            h[:-4, 1:-1], h[1:-3, 1:-1], h[2:-2, 1:-1], h[3:-1, 1:-1], h[4:, 1:-1]
        )
        # WENO-3 fallback at first interior row (j=1)
        h3_pos_first = _weno3(h[0:1, 1:-1], h[1:2, 1:-1], h[2:3, 1:-1])
        # WENO-3 fallback at last interior row (j=Ny-2)
        h3_pos_last = _weno3(h[-3:-2, 1:-1], h[-2:-1, 1:-1], h[-1:, 1:-1])
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=0)
        # WENO-5 right-biased: valid for j=1..Ny-4
        h5_neg_int = _weno5_right(
            h[:-4, 1:-1], h[1:-3, 1:-1], h[2:-2, 1:-1], h[3:-1, 1:-1], h[4:, 1:-1]
        )
        # WENO-3 right-biased fallback at row j=Ny-3
        h3_neg_penultimate = _weno3_right(h[-3:-2, 1:-1], h[-2:-1, 1:-1], h[-1:, 1:-1])
        # 1st-order upwind fallback at north boundary row (j=Ny-2)
        h1_neg_last = h[-1:, 1:-1]
        h_neg = jnp.concatenate([h5_neg_int, h3_neg_penultimate, h1_neg_last], axis=0)
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * v[1:-1, 1:-1], h)
        return out

    def wenoz5_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO-Z east-face flux with sign-dependent boundary fallbacks.

        Positive flow:  fe[j,i+1/2] = WENOZ5(h[j,i-2..i+2]) * u
            for i = 2..Nx-3; WENO-Z-3 fallback at i = 1 and i = Nx-2.
        Negative flow:  fe[j,i+1/2] = WENOZ5_right(h[j,i-1], h[j,i], h[j,i+1], h[j,i+2], h[j,i+3]) * u
            for i = 1..Nx-4; WENO-Z-3_right fallback at i = Nx-3; 1st-order upwind at i = Nx-2.
        """
        # WENO-Z-5 positive: valid for i=2..Nx-3
        h5_pos_int = _wenoz5(
            h[1:-1, :-4], h[1:-1, 1:-3], h[1:-1, 2:-2], h[1:-1, 3:-1], h[1:-1, 4:]
        )
        # WENO-Z-3 fallback at first interior column (i=1)
        h3_pos_first = _wenoz3(h[1:-1, 0:1], h[1:-1, 1:2], h[1:-1, 2:3])
        # WENO-Z-3 fallback at last interior column (i=Nx-2)
        h3_pos_last = _wenoz3(h[1:-1, -3:-2], h[1:-1, -2:-1], h[1:-1, -1:])
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=1)
        # WENO-Z-5 right-biased: valid for i=1..Nx-4
        h5_neg_int = _wenoz5_right(
            h[1:-1, :-4], h[1:-1, 1:-3], h[1:-1, 2:-2], h[1:-1, 3:-1], h[1:-1, 4:]
        )
        # WENO-Z-3 right-biased fallback at column i=Nx-3
        h3_neg_penultimate = _wenoz3_right(h[1:-1, -3:-2], h[1:-1, -2:-1], h[1:-1, -1:])
        # 1st-order upwind fallback at east boundary column (i=Nx-2)
        h1_neg_last = h[1:-1, -1:]
        h_neg = jnp.concatenate([h5_neg_int, h3_neg_penultimate, h1_neg_last], axis=1)
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1, 1:-1], h)
        return out

    def wenoz5_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO-Z north-face flux with sign-dependent boundary fallbacks.

        Positive flow:  fn[j+1/2,i] = WENOZ5(h[j-2..j+2,i]) * v
            for j = 2..Ny-3; WENO-Z-3 fallback at j = 1 and j = Ny-2.
        Negative flow:  fn[j+1/2,i] = WENOZ5_right(h[j-1,i], h[j,i], h[j+1,i], h[j+2,i], h[j+3,i]) * v
            for j = 1..Ny-4; WENO-Z-3_right fallback at j = Ny-3; 1st-order upwind at j = Ny-2.
        """
        # WENO-Z-5 positive: valid for j=2..Ny-3
        h5_pos_int = _wenoz5(
            h[:-4, 1:-1], h[1:-3, 1:-1], h[2:-2, 1:-1], h[3:-1, 1:-1], h[4:, 1:-1]
        )
        # WENO-Z-3 fallback at first interior row (j=1)
        h3_pos_first = _wenoz3(h[0:1, 1:-1], h[1:2, 1:-1], h[2:3, 1:-1])
        # WENO-Z-3 fallback at last interior row (j=Ny-2)
        h3_pos_last = _wenoz3(h[-3:-2, 1:-1], h[-2:-1, 1:-1], h[-1:, 1:-1])
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=0)
        # WENO-Z-5 right-biased: valid for j=1..Ny-4
        h5_neg_int = _wenoz5_right(
            h[:-4, 1:-1], h[1:-3, 1:-1], h[2:-2, 1:-1], h[3:-1, 1:-1], h[4:, 1:-1]
        )
        # WENO-Z-3 right-biased fallback at row j=Ny-3
        h3_neg_penultimate = _wenoz3_right(h[-3:-2, 1:-1], h[-2:-1, 1:-1], h[-1:, 1:-1])
        # 1st-order upwind fallback at north boundary row (j=Ny-2)
        h1_neg_last = h[-1:, 1:-1]
        h_neg = jnp.concatenate([h5_neg_int, h3_neg_penultimate, h1_neg_last], axis=0)
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * v[1:-1, 1:-1], h)
        return out

    def weno7_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """7-point WENO east-face flux with hierarchical boundary fallbacks."""
        return _weno_flux_axis_2d_x(h, u, order=7)

    def weno7_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """7-point WENO north-face flux with hierarchical boundary fallbacks."""
        return _weno_flux_axis_2d_y(h, v, order=7)

    def weno9_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """9-point WENO east-face flux with hierarchical boundary fallbacks."""
        return _weno_flux_axis_2d_x(h, u, order=9)

    def weno9_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """9-point WENO north-face flux with hierarchical boundary fallbacks."""
        return _weno_flux_axis_2d_y(h, v, order=9)

    def tvd_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        limiter: str = "minmod",
    ) -> Float[Array, "Ny Nx"]:
        """TVD east-face flux using a flux limiter.

        Positive flow:
            r         = (h[j,i] − h[j,i−1]) / (h[j,i+1] − h[j,i])
            h_L[j,i+½] = h[j,i]   + ½ φ(r) (h[j,i+1] − h[j,i])

        Negative flow:
            r         = (h[j,i+1] − h[j,i+2]) / (h[j,i] − h[j,i+1])
            h_R[j,i+½] = h[j,i+1] + ½ φ(r) (h[j,i]   − h[j,i+1])
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            East-face flux with zero ghost ring.
        """
        phi = _get_limiter(limiter)
        # Positive flow: h_face = h[j,i] + 0.5*phi(r)*(h[j,i+1]-h[j,i])
        # r = (h[j,i]-h[j,i-1]) / (h[j,i+1]-h[j,i])
        diff_fwd = h[1:-1, 2:] - h[1:-1, 1:-1]  # h[j,i+1] - h[j,i]
        diff_bwd = h[1:-1, 1:-1] - h[1:-1, :-2]  # h[j,i] - h[j,i-1]
        r_pos = diff_bwd / (diff_fwd + _TVD_EPS)
        h_pos = h[1:-1, 1:-1] + 0.5 * phi(r_pos) * diff_fwd
        # Negative flow: h_face = h[j,i+1] + 0.5*phi(r)*(h[j,i]-h[j,i+1])
        # r = (h[j,i+1]-h[j,i+2]) / (h[j,i]-h[j,i+1])  for i=1..Nx-3
        diff_neg = h[1:-1, 1:-2] - h[1:-1, 2:-1]  # h[j,i] - h[j,i+1]
        diff_neg2 = h[1:-1, 2:-1] - h[1:-1, 3:]  # h[j,i+1] - h[j,i+2]
        r_neg_int = diff_neg2 / (diff_neg + _TVD_EPS)
        h_neg_interior = h[1:-1, 2:-1] + 0.5 * phi(r_neg_int) * diff_neg
        # 1st-order upwind fallback at east boundary column (i=Nx-2)
        h_neg_boundary = h[1:-1, -1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=1)
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1, 1:-1], h)
        return out

    def tvd_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        limiter: str = "minmod",
    ) -> Float[Array, "Ny Nx"]:
        """TVD north-face flux using a flux limiter.

        Positive flow:
            r         = (h[j,i] − h[j−1,i]) / (h[j+1,i] − h[j,i])
            h_L[j+½,i] = h[j,i]   + ½ φ(r) (h[j+1,i] − h[j,i])

        Negative flow:
            r         = (h[j+1,i] − h[j+2,i]) / (h[j,i] − h[j+1,i])
            h_R[j+½,i] = h[j+1,i] + ½ φ(r) (h[j,i]   − h[j+1,i])
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            North-face flux with zero ghost ring.
        """
        phi = _get_limiter(limiter)
        # Positive flow: h_face = h[j,i] + 0.5*phi(r)*(h[j+1,i]-h[j,i])
        # r = (h[j,i]-h[j-1,i]) / (h[j+1,i]-h[j,i])
        diff_fwd = h[2:, 1:-1] - h[1:-1, 1:-1]  # h[j+1,i] - h[j,i]
        diff_bwd = h[1:-1, 1:-1] - h[:-2, 1:-1]  # h[j,i] - h[j-1,i]
        r_pos = diff_bwd / (diff_fwd + _TVD_EPS)
        h_pos = h[1:-1, 1:-1] + 0.5 * phi(r_pos) * diff_fwd
        # Negative flow: h_face = h[j+1,i] + 0.5*phi(r)*(h[j,i]-h[j+1,i])
        # r = (h[j+1,i]-h[j+2,i]) / (h[j,i]-h[j+1,i])  for j=1..Ny-3
        diff_neg = h[1:-2, 1:-1] - h[2:-1, 1:-1]  # h[j,i] - h[j+1,i]
        diff_neg2 = h[2:-1, 1:-1] - h[3:, 1:-1]  # h[j+1,i] - h[j+2,i]
        r_neg_int = diff_neg2 / (diff_neg + _TVD_EPS)
        h_neg_interior = h[2:-1, 1:-1] + 0.5 * phi(r_neg_int) * diff_neg
        # 1st-order upwind fallback at north boundary row (j=Ny-2)
        h_neg_boundary = h[-1:, 1:-1]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=0)
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * v[1:-1, 1:-1], h)
        return out

    def tvd_x_masked(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask,
        limiter: str = "minmod",
    ) -> Float[Array, "Ny Nx"]:
        """TVD east-face flux with mask-aware stencil selection.

        Delegates to :func:`~finitevolx._src.advection.flux.upwind_flux` with a
        two-tier hierarchy (upwind1 / TVD).

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Cell-centre tracer field.
        u : Float[Array, "Ny Nx"]
            East-face velocity.
        mask : ArakawaCGridMask
            Arakawa C-grid mask providing stencil-capability information.
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            East-face flux with zero ghost ring.
        """
        amasks = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4))
        rec_funcs: dict[int, Callable[..., Float[Array, "Ny Nx"]]] = {
            2: self.upwind1_x,
            4: lambda q, vel: self.tvd_x(q, vel, limiter=limiter),
        }
        return upwind_flux(h, u, dim=1, rec_funcs=rec_funcs, mask_hierarchy=amasks)

    def tvd_y_masked(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask,
        limiter: str = "minmod",
    ) -> Float[Array, "Ny Nx"]:
        """TVD north-face flux with mask-aware stencil selection.

        Delegates to :func:`~finitevolx._src.advection.flux.upwind_flux` with a
        two-tier hierarchy (upwind1 / TVD).

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Cell-centre tracer field.
        v : Float[Array, "Ny Nx"]
            North-face velocity.
        mask : ArakawaCGridMask
            Arakawa C-grid mask providing stencil-capability information.
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            North-face flux with zero ghost ring.
        """
        amasks = mask.get_adaptive_masks(direction="y", stencil_sizes=(2, 4))
        rec_funcs: dict[int, Callable[..., Float[Array, "Ny Nx"]]] = {
            2: self.upwind1_y,
            4: lambda q, vel: self.tvd_y(q, vel, limiter=limiter),
        }
        return upwind_flux(h, v, dim=0, rec_funcs=rec_funcs, mask_hierarchy=amasks)

    def weno5_x_masked(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask,
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO east-face flux with mask-aware adaptive stencil selection.

        Delegates to :func:`~finitevolx._src.advection.flux.upwind_flux` with a
        three-tier hierarchy (upwind1 / WENO3 / WENO5).

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Cell-centre tracer field.
        u : Float[Array, "Ny Nx"]
            East-face velocity.
        mask : ArakawaCGridMask
            Arakawa C-grid mask providing stencil-capability information.

        Returns
        -------
        Float[Array, "Ny Nx"]
            East-face flux with zero ghost ring.
        """
        amasks = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        rec_funcs: dict[int, Callable[..., Float[Array, "Ny Nx"]]] = {
            2: self.upwind1_x,
            4: self.weno3_x,
            6: self.weno5_x,
        }
        return upwind_flux(h, u, dim=1, rec_funcs=rec_funcs, mask_hierarchy=amasks)

    def weno5_y_masked(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask,
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO north-face flux with mask-aware adaptive stencil selection.

        Delegates to :func:`~finitevolx._src.advection.flux.upwind_flux` with a
        three-tier hierarchy (upwind1 / WENO3 / WENO5).

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Cell-centre tracer field.
        v : Float[Array, "Ny Nx"]
            North-face velocity.
        mask : ArakawaCGridMask
            Arakawa C-grid mask providing stencil-capability information.

        Returns
        -------
        Float[Array, "Ny Nx"]
            North-face flux with zero ghost ring.
        """
        amasks = mask.get_adaptive_masks(direction="y", stencil_sizes=(2, 4, 6))
        rec_funcs: dict[int, Callable[..., Float[Array, "Ny Nx"]]] = {
            2: self.upwind1_y,
            4: self.weno3_y,
            6: self.weno5_y,
        }
        return upwind_flux(h, v, dim=0, rec_funcs=rec_funcs, mask_hierarchy=amasks)

    def wenoz5_x_masked(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask,
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO-Z east-face flux with mask-aware adaptive stencil selection.

        Delegates to :func:`~finitevolx._src.advection.flux.upwind_flux` with a
        three-tier hierarchy (upwind1 / WENO-Z-3 / WENO-Z-5).

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Cell-centre tracer field.
        u : Float[Array, "Ny Nx"]
            East-face velocity.
        mask : ArakawaCGridMask
            Arakawa C-grid mask providing stencil-capability information.

        Returns
        -------
        Float[Array, "Ny Nx"]
            East-face flux with zero ghost ring.
        """
        amasks = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        rec_funcs: dict[int, Callable[..., Float[Array, "Ny Nx"]]] = {
            2: self.upwind1_x,
            4: self.wenoz3_x,
            6: self.wenoz5_x,
        }
        return upwind_flux(h, u, dim=1, rec_funcs=rec_funcs, mask_hierarchy=amasks)

    def wenoz5_y_masked(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask,
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO-Z north-face flux with mask-aware adaptive stencil selection.

        Delegates to :func:`~finitevolx._src.advection.flux.upwind_flux` with a
        three-tier hierarchy (upwind1 / WENO-Z-3 / WENO-Z-5).

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Cell-centre tracer field.
        v : Float[Array, "Ny Nx"]
            North-face velocity.
        mask : ArakawaCGridMask
            Arakawa C-grid mask providing stencil-capability information.

        Returns
        -------
        Float[Array, "Ny Nx"]
            North-face flux with zero ghost ring.
        """
        amasks = mask.get_adaptive_masks(direction="y", stencil_sizes=(2, 4, 6))
        rec_funcs: dict[int, Callable[..., Float[Array, "Ny Nx"]]] = {
            2: self.upwind1_y,
            4: self.wenoz3_y,
            6: self.wenoz5_y,
        }
        return upwind_flux(h, v, dim=0, rec_funcs=rec_funcs, mask_hierarchy=amasks)


class Reconstruction3D(eqx.Module):
    """3-D face-value reconstruction.

    Parameters
    ----------
    grid : ArakawaCGrid3D
    """

    grid: ArakawaCGrid3D

    def naive_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """Naive east-face flux over all z-levels.

        fe[k, j, i+1/2] = 1/2 * (h[k,j,i] + h[k,j,i+1]) * u[k,j,i+1/2]
        """
        out = interior(
            0.5 * (h[1:-1, 1:-1, 1:-1] + h[1:-1, 1:-1, 2:]) * u[1:-1, 1:-1, 1:-1],
            h,
        )
        return out

    def naive_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """Naive north-face flux over all z-levels.

        fn[k, j+1/2, i] = 1/2 * (h[k,j,i] + h[k,j+1,i]) * v[k,j+1/2,i]
        """
        out = interior(
            0.5 * (h[1:-1, 1:-1, 1:-1] + h[1:-1, 2:, 1:-1]) * v[1:-1, 1:-1, 1:-1],
            h,
        )
        return out

    def upwind1_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """1st-order upwind east-face flux over all z-levels."""
        h_face = jnp.where(
            u[1:-1, 1:-1, 1:-1] >= 0.0,
            h[1:-1, 1:-1, 1:-1],
            h[1:-1, 1:-1, 2:],
        )
        out = interior(h_face * u[1:-1, 1:-1, 1:-1], h)
        return out

    def upwind1_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """1st-order upwind north-face flux over all z-levels."""
        h_face = jnp.where(
            v[1:-1, 1:-1, 1:-1] >= 0.0,
            h[1:-1, 1:-1, 1:-1],
            h[1:-1, 2:, 1:-1],
        )
        out = interior(h_face * v[1:-1, 1:-1, 1:-1], h)
        return out

    def weno3_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO east-face flux over all z-levels with boundary fallback.

        Positive flow:  fe[k,j,i+1/2] = WENO3(h[k,j,i-1], h[k,j,i],   h[k,j,i+1]) * u
        Negative flow:  fe[k,j,i+1/2] = WENO3_right(h[k,j,i], h[k,j,i+1], h[k,j,i+2]) * u
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        # WENO-3 positive: left-biased, valid for all interior faces
        h_pos = _weno3(h[1:-1, 1:-1, :-2], h[1:-1, 1:-1, 1:-1], h[1:-1, 1:-1, 2:])
        # WENO-3 negative: right-biased, valid for i+2 < Nx
        h_neg_interior = _weno3_right(
            h[1:-1, 1:-1, 1:-2], h[1:-1, 1:-1, 2:-1], h[1:-1, 1:-1, 3:]
        )
        # 1st-order upwind fallback at east boundary
        h_neg_boundary = h[1:-1, 1:-1, -1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=2)
        h_face = jnp.where(u[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1, 1:-1, 1:-1], h)
        return out

    def weno3_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO north-face flux over all z-levels with boundary fallback.

        Positive flow:  fn[k,j+1/2,i] = WENO3(h[k,j-1,i], h[k,j,i],   h[k,j+1,i]) * v
        Negative flow:  fn[k,j+1/2,i] = WENO3_right(h[k,j,i], h[k,j+1,i], h[k,j+2,i]) * v
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        # WENO-3 positive: left-biased, valid for all interior faces
        h_pos = _weno3(h[1:-1, :-2, 1:-1], h[1:-1, 1:-1, 1:-1], h[1:-1, 2:, 1:-1])
        # WENO-3 negative: right-biased, valid for j+2 < Ny
        h_neg_interior = _weno3_right(
            h[1:-1, 1:-2, 1:-1], h[1:-1, 2:-1, 1:-1], h[1:-1, 3:, 1:-1]
        )
        # 1st-order upwind fallback at north boundary
        h_neg_boundary = h[1:-1, -1:, 1:-1]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=1)
        h_face = jnp.where(v[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * v[1:-1, 1:-1, 1:-1], h)
        return out

    def wenoz3_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO-Z east-face flux over all z-levels with boundary fallback.

        Positive flow:  fe[k,j,i+1/2] = WENOZ3(h[k,j,i-1], h[k,j,i],   h[k,j,i+1]) * u
        Negative flow:  fe[k,j,i+1/2] = WENOZ3_right(h[k,j,i], h[k,j,i+1], h[k,j,i+2]) * u
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        # WENO-Z-3 positive: left-biased, valid for all interior faces
        h_pos = _wenoz3(h[1:-1, 1:-1, :-2], h[1:-1, 1:-1, 1:-1], h[1:-1, 1:-1, 2:])
        # WENO-Z-3 negative: right-biased, valid for i+2 < Nx
        h_neg_interior = _wenoz3_right(
            h[1:-1, 1:-1, 1:-2], h[1:-1, 1:-1, 2:-1], h[1:-1, 1:-1, 3:]
        )
        # 1st-order upwind fallback at east boundary
        h_neg_boundary = h[1:-1, 1:-1, -1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=2)
        h_face = jnp.where(u[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1, 1:-1, 1:-1], h)
        return out

    def wenoz3_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO-Z north-face flux over all z-levels with boundary fallback.

        Positive flow:  fn[k,j+1/2,i] = WENOZ3(h[k,j-1,i], h[k,j,i],   h[k,j+1,i]) * v
        Negative flow:  fn[k,j+1/2,i] = WENOZ3_right(h[k,j,i], h[k,j+1,i], h[k,j+2,i]) * v
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        # WENO-Z-3 positive: left-biased, valid for all interior faces
        h_pos = _wenoz3(h[1:-1, :-2, 1:-1], h[1:-1, 1:-1, 1:-1], h[1:-1, 2:, 1:-1])
        # WENO-Z-3 negative: right-biased, valid for j+2 < Ny
        h_neg_interior = _wenoz3_right(
            h[1:-1, 1:-2, 1:-1], h[1:-1, 2:-1, 1:-1], h[1:-1, 3:, 1:-1]
        )
        # 1st-order upwind fallback at north boundary
        h_neg_boundary = h[1:-1, -1:, 1:-1]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=1)
        h_face = jnp.where(v[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * v[1:-1, 1:-1, 1:-1], h)
        return out

    def weno5_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO east-face flux over all z-levels with sign-dependent fallbacks.

        Positive flow:  fe[k,j,i+1/2] = WENO5(h[k,j,i-2..i+2]) * u
            for i = 2..Nx-3; WENO3 fallback at i = 1 and i = Nx-2.
        Negative flow:  fe[k,j,i+1/2] = WENO5_right(h[k,j,i-1], h[k,j,i], h[k,j,i+1], h[k,j,i+2], h[k,j,i+3]) * u
            for i = 1..Nx-4; WENO3_right fallback at i = Nx-3; 1st-order upwind at i = Nx-2.
        """
        h5_pos_int = _weno5(
            h[1:-1, 1:-1, :-4],
            h[1:-1, 1:-1, 1:-3],
            h[1:-1, 1:-1, 2:-2],
            h[1:-1, 1:-1, 3:-1],
            h[1:-1, 1:-1, 4:],
        )
        h3_pos_first = _weno3(
            h[1:-1, 1:-1, 0:1], h[1:-1, 1:-1, 1:2], h[1:-1, 1:-1, 2:3]
        )
        h3_pos_last = _weno3(
            h[1:-1, 1:-1, -3:-2], h[1:-1, 1:-1, -2:-1], h[1:-1, 1:-1, -1:]
        )
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=2)
        h5_neg_int = _weno5_right(
            h[1:-1, 1:-1, :-4],
            h[1:-1, 1:-1, 1:-3],
            h[1:-1, 1:-1, 2:-2],
            h[1:-1, 1:-1, 3:-1],
            h[1:-1, 1:-1, 4:],
        )
        h3_neg_penultimate = _weno3_right(
            h[1:-1, 1:-1, -3:-2], h[1:-1, 1:-1, -2:-1], h[1:-1, 1:-1, -1:]
        )
        h_neg = jnp.concatenate(
            [h5_neg_int, h3_neg_penultimate, h[1:-1, 1:-1, -1:]], axis=2
        )
        h_face = jnp.where(u[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1, 1:-1, 1:-1], h)
        return out

    def weno5_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO north-face flux over all z-levels with sign-dependent fallbacks.

        Positive flow:  fn[k,j+1/2,i] = WENO5(h[k,j-2..j+2,i]) * v
            for j = 2..Ny-3; WENO3 fallback at j = 1 and j = Ny-2.
        Negative flow:  fn[k,j+1/2,i] = WENO5_right(h[k,j-1,i], h[k,j,i], h[k,j+1,i], h[k,j+2,i], h[k,j+3,i]) * v
            for j = 1..Ny-4; WENO3_right fallback at j = Ny-3; 1st-order upwind at j = Ny-2.
        """
        h5_pos_int = _weno5(
            h[1:-1, :-4, 1:-1],
            h[1:-1, 1:-3, 1:-1],
            h[1:-1, 2:-2, 1:-1],
            h[1:-1, 3:-1, 1:-1],
            h[1:-1, 4:, 1:-1],
        )
        h3_pos_first = _weno3(
            h[1:-1, 0:1, 1:-1], h[1:-1, 1:2, 1:-1], h[1:-1, 2:3, 1:-1]
        )
        h3_pos_last = _weno3(
            h[1:-1, -3:-2, 1:-1], h[1:-1, -2:-1, 1:-1], h[1:-1, -1:, 1:-1]
        )
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=1)
        h5_neg_int = _weno5_right(
            h[1:-1, :-4, 1:-1],
            h[1:-1, 1:-3, 1:-1],
            h[1:-1, 2:-2, 1:-1],
            h[1:-1, 3:-1, 1:-1],
            h[1:-1, 4:, 1:-1],
        )
        h3_neg_penultimate = _weno3_right(
            h[1:-1, -3:-2, 1:-1], h[1:-1, -2:-1, 1:-1], h[1:-1, -1:, 1:-1]
        )
        h_neg = jnp.concatenate(
            [h5_neg_int, h3_neg_penultimate, h[1:-1, -1:, 1:-1]], axis=1
        )
        h_face = jnp.where(v[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * v[1:-1, 1:-1, 1:-1], h)
        return out

    def wenoz5_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO-Z east-face flux over all z-levels with sign-dependent fallbacks.

        Positive flow:  fe[k,j,i+1/2] = WENOZ5(h[k,j,i-2..i+2]) * u
            for i = 2..Nx-3; WENO-Z-3 fallback at i = 1 and i = Nx-2.
        Negative flow:  fe[k,j,i+1/2] = WENOZ5_right(h[k,j,i-1], h[k,j,i], h[k,j,i+1], h[k,j,i+2], h[k,j,i+3]) * u
            for i = 1..Nx-4; WENO-Z-3_right fallback at i = Nx-3; 1st-order upwind at i = Nx-2.
        """
        h5_pos_int = _wenoz5(
            h[1:-1, 1:-1, :-4],
            h[1:-1, 1:-1, 1:-3],
            h[1:-1, 1:-1, 2:-2],
            h[1:-1, 1:-1, 3:-1],
            h[1:-1, 1:-1, 4:],
        )
        h3_pos_first = _wenoz3(
            h[1:-1, 1:-1, 0:1], h[1:-1, 1:-1, 1:2], h[1:-1, 1:-1, 2:3]
        )
        h3_pos_last = _wenoz3(
            h[1:-1, 1:-1, -3:-2], h[1:-1, 1:-1, -2:-1], h[1:-1, 1:-1, -1:]
        )
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=2)
        h5_neg_int = _wenoz5_right(
            h[1:-1, 1:-1, :-4],
            h[1:-1, 1:-1, 1:-3],
            h[1:-1, 1:-1, 2:-2],
            h[1:-1, 1:-1, 3:-1],
            h[1:-1, 1:-1, 4:],
        )
        h3_neg_penultimate = _wenoz3_right(
            h[1:-1, 1:-1, -3:-2], h[1:-1, 1:-1, -2:-1], h[1:-1, 1:-1, -1:]
        )
        h_neg = jnp.concatenate(
            [h5_neg_int, h3_neg_penultimate, h[1:-1, 1:-1, -1:]], axis=2
        )
        h_face = jnp.where(u[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1, 1:-1, 1:-1], h)
        return out

    def wenoz5_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO-Z north-face flux over all z-levels with sign-dependent fallbacks.

        Positive flow:  fn[k,j+1/2,i] = WENOZ5(h[k,j-2..j+2,i]) * v
            for j = 2..Ny-3; WENO-Z-3 fallback at j = 1 and j = Ny-2.
        Negative flow:  fn[k,j+1/2,i] = WENOZ5_right(h[k,j-1,i], h[k,j,i], h[k,j+1,i], h[k,j+2,i], h[k,j+3,i]) * v
            for j = 1..Ny-4; WENO-Z-3_right fallback at j = Ny-3; 1st-order upwind at j = Ny-2.
        """
        h5_pos_int = _wenoz5(
            h[1:-1, :-4, 1:-1],
            h[1:-1, 1:-3, 1:-1],
            h[1:-1, 2:-2, 1:-1],
            h[1:-1, 3:-1, 1:-1],
            h[1:-1, 4:, 1:-1],
        )
        h3_pos_first = _wenoz3(
            h[1:-1, 0:1, 1:-1], h[1:-1, 1:2, 1:-1], h[1:-1, 2:3, 1:-1]
        )
        h3_pos_last = _wenoz3(
            h[1:-1, -3:-2, 1:-1], h[1:-1, -2:-1, 1:-1], h[1:-1, -1:, 1:-1]
        )
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=1)
        h5_neg_int = _wenoz5_right(
            h[1:-1, :-4, 1:-1],
            h[1:-1, 1:-3, 1:-1],
            h[1:-1, 2:-2, 1:-1],
            h[1:-1, 3:-1, 1:-1],
            h[1:-1, 4:, 1:-1],
        )
        h3_neg_penultimate = _wenoz3_right(
            h[1:-1, -3:-2, 1:-1], h[1:-1, -2:-1, 1:-1], h[1:-1, -1:, 1:-1]
        )
        h_neg = jnp.concatenate(
            [h5_neg_int, h3_neg_penultimate, h[1:-1, -1:, 1:-1]], axis=1
        )
        h_face = jnp.where(v[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * v[1:-1, 1:-1, 1:-1], h)
        return out

    def weno7_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """7-point WENO east-face flux over all z-levels with hierarchical fallbacks."""
        return _weno_flux_axis_3d_x(h, u, order=7)

    def weno7_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """7-point WENO north-face flux over all z-levels with hierarchical fallbacks."""
        return _weno_flux_axis_3d_y(h, v, order=7)

    def weno9_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """9-point WENO east-face flux over all z-levels with hierarchical fallbacks."""
        return _weno_flux_axis_3d_x(h, u, order=9)

    def weno9_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """9-point WENO north-face flux over all z-levels with hierarchical fallbacks."""
        return _weno_flux_axis_3d_y(h, v, order=9)

    def tvd_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
        limiter: str = "minmod",
    ) -> Float[Array, "Nz Ny Nx"]:
        """TVD east-face flux over all z-levels using a flux limiter.

        Positive flow:
            r           = (h[k,j,i] − h[k,j,i−1]) / (h[k,j,i+1] − h[k,j,i])
            h_L[k,j,i+½] = h[k,j,i]   + ½ φ(r) (h[k,j,i+1] − h[k,j,i])

        Negative flow:
            r           = (h[k,j,i+1] − h[k,j,i+2]) / (h[k,j,i] − h[k,j,i+1])
            h_R[k,j,i+½] = h[k,j,i+1] + ½ φ(r) (h[k,j,i]   − h[k,j,i+1])
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar at T-points.
        u : Float[Array, "Nz Ny Nx"]
            x-velocity at U-points.
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            East-face flux with zero ghost ring.
        """
        phi = _get_limiter(limiter)
        # Positive flow: h_face = h[k,j,i] + 0.5*phi(r)*(h[k,j,i+1]-h[k,j,i])
        diff_fwd = h[1:-1, 1:-1, 2:] - h[1:-1, 1:-1, 1:-1]
        diff_bwd = h[1:-1, 1:-1, 1:-1] - h[1:-1, 1:-1, :-2]
        r_pos = diff_bwd / (diff_fwd + _TVD_EPS)
        h_pos = h[1:-1, 1:-1, 1:-1] + 0.5 * phi(r_pos) * diff_fwd
        # Negative flow: h_face = h[k,j,i+1] + 0.5*phi(r)*(h[k,j,i]-h[k,j,i+1])
        diff_neg = h[1:-1, 1:-1, 1:-2] - h[1:-1, 1:-1, 2:-1]
        diff_neg2 = h[1:-1, 1:-1, 2:-1] - h[1:-1, 1:-1, 3:]
        r_neg_int = diff_neg2 / (diff_neg + _TVD_EPS)
        h_neg_interior = h[1:-1, 1:-1, 2:-1] + 0.5 * phi(r_neg_int) * diff_neg
        # 1st-order upwind fallback at east boundary column (i=Nx-2)
        h_neg_boundary = h[1:-1, 1:-1, -1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=2)
        h_face = jnp.where(u[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * u[1:-1, 1:-1, 1:-1], h)
        return out

    def tvd_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        limiter: str = "minmod",
    ) -> Float[Array, "Nz Ny Nx"]:
        """TVD north-face flux over all z-levels using a flux limiter.

        Positive flow:
            r           = (h[k,j,i] − h[k,j−1,i]) / (h[k,j+1,i] − h[k,j,i])
            h_L[k,j+½,i] = h[k,j,i]   + ½ φ(r) (h[k,j+1,i] − h[k,j,i])

        Negative flow:
            r           = (h[k,j+1,i] − h[k,j+2,i]) / (h[k,j,i] − h[k,j+1,i])
            h_R[k,j+½,i] = h[k,j+1,i] + ½ φ(r) (h[k,j,i]   − h[k,j+1,i])
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar at T-points.
        v : Float[Array, "Nz Ny Nx"]
            y-velocity at V-points.
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            North-face flux with zero ghost ring.
        """
        phi = _get_limiter(limiter)
        # Positive flow: h_face = h[k,j,i] + 0.5*phi(r)*(h[k,j+1,i]-h[k,j,i])
        diff_fwd = h[1:-1, 2:, 1:-1] - h[1:-1, 1:-1, 1:-1]
        diff_bwd = h[1:-1, 1:-1, 1:-1] - h[1:-1, :-2, 1:-1]
        r_pos = diff_bwd / (diff_fwd + _TVD_EPS)
        h_pos = h[1:-1, 1:-1, 1:-1] + 0.5 * phi(r_pos) * diff_fwd
        # Negative flow: h_face = h[k,j+1,i] + 0.5*phi(r)*(h[k,j,i]-h[k,j+1,i])
        diff_neg = h[1:-1, 1:-2, 1:-1] - h[1:-1, 2:-1, 1:-1]
        diff_neg2 = h[1:-1, 2:-1, 1:-1] - h[1:-1, 3:, 1:-1]
        r_neg_int = diff_neg2 / (diff_neg + _TVD_EPS)
        h_neg_interior = h[1:-1, 2:-1, 1:-1] + 0.5 * phi(r_neg_int) * diff_neg
        # 1st-order upwind fallback at north boundary row (j=Ny-2)
        h_neg_boundary = h[1:-1, -1:, 1:-1]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=1)
        h_face = jnp.where(v[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = interior(h_face * v[1:-1, 1:-1, 1:-1], h)
        return out

    def tvd_x_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask,
        limiter: str = "minmod",
    ) -> Float[Array, "Nz Ny Nx"]:
        """TVD east-face flux over all z-levels with mask-aware stencil selection.

        The 2-D ``mask`` is broadcast over the z-dimension.  Falls back to
        1st-order upwind near coastlines where the TVD stencil crosses land.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        u : Float[Array, "Nz Ny Nx"]
            East-face velocity.
        mask : ArakawaCGridMask
            2-D Arakawa C-grid mask (broadcast over z).
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            East-face flux with zero ghost ring.
        """
        amasks = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4))
        m_tvd = amasks[4]
        fe_tvd = self.tvd_x(h, u, limiter=limiter)
        fe_u1 = self.upwind1_x(h, u)
        pos_flow = u[1:-1, 1:-1, 1:-1] >= 0.0
        use_tvd = jnp.where(pos_flow, m_tvd[None, 1:-1, 1:-1], m_tvd[None, 1:-1, 2:])
        selected = jnp.where(use_tvd, fe_tvd[1:-1, 1:-1, 1:-1], fe_u1[1:-1, 1:-1, 1:-1])
        out = interior(selected, h)
        return out

    def tvd_y_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask,
        limiter: str = "minmod",
    ) -> Float[Array, "Nz Ny Nx"]:
        """TVD north-face flux over all z-levels with mask-aware stencil selection.

        The 2-D ``mask`` is broadcast over the z-dimension.  Falls back to
        1st-order upwind near coastlines where the TVD stencil crosses land.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        v : Float[Array, "Nz Ny Nx"]
            North-face velocity.
        mask : ArakawaCGridMask
            2-D Arakawa C-grid mask (broadcast over z).
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            North-face flux with zero ghost ring.
        """
        amasks = mask.get_adaptive_masks(direction="y", stencil_sizes=(2, 4))
        m_tvd = amasks[4]
        fn_tvd = self.tvd_y(h, v, limiter=limiter)
        fn_u1 = self.upwind1_y(h, v)
        pos_flow = v[1:-1, 1:-1, 1:-1] >= 0.0
        use_tvd = jnp.where(pos_flow, m_tvd[None, 1:-1, 1:-1], m_tvd[None, 2:, 1:-1])
        selected = jnp.where(use_tvd, fn_tvd[1:-1, 1:-1, 1:-1], fn_u1[1:-1, 1:-1, 1:-1])
        out = interior(selected, h)
        return out

    def weno3_x_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask,
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO east-face flux over all z-levels with mask-aware stencil selection.

        The 2-D ``mask`` is broadcast over the z-dimension.  Falls back to
        1st-order upwind near coastlines where the WENO3 stencil crosses land.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        u : Float[Array, "Nz Ny Nx"]
            East-face velocity.
        mask : ArakawaCGridMask
            2-D Arakawa C-grid mask (broadcast over z).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            East-face flux with zero ghost ring.
        """
        amasks = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4))
        m_w3 = amasks[4]
        fe_w3 = self.weno3_x(h, u)
        fe_u1 = self.upwind1_x(h, u)
        pos_flow = u[1:-1, 1:-1, 1:-1] >= 0.0
        use_w3 = jnp.where(pos_flow, m_w3[None, 1:-1, 1:-1], m_w3[None, 1:-1, 2:])
        selected = jnp.where(use_w3, fe_w3[1:-1, 1:-1, 1:-1], fe_u1[1:-1, 1:-1, 1:-1])
        out = interior(selected, h)
        return out

    def weno3_y_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask,
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO north-face flux over all z-levels with mask-aware stencil selection.

        The 2-D ``mask`` is broadcast over the z-dimension.  Falls back to
        1st-order upwind near coastlines where the WENO3 stencil crosses land.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        v : Float[Array, "Nz Ny Nx"]
            North-face velocity.
        mask : ArakawaCGridMask
            2-D Arakawa C-grid mask (broadcast over z).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            North-face flux with zero ghost ring.
        """
        amasks = mask.get_adaptive_masks(direction="y", stencil_sizes=(2, 4))
        m_w3 = amasks[4]
        fn_w3 = self.weno3_y(h, v)
        fn_u1 = self.upwind1_y(h, v)
        pos_flow = v[1:-1, 1:-1, 1:-1] >= 0.0
        use_w3 = jnp.where(pos_flow, m_w3[None, 1:-1, 1:-1], m_w3[None, 2:, 1:-1])
        selected = jnp.where(use_w3, fn_w3[1:-1, 1:-1, 1:-1], fn_u1[1:-1, 1:-1, 1:-1])
        out = interior(selected, h)
        return out

    def weno5_x_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask,
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO east-face flux over all z-levels with mask-aware adaptive stencil.

        The 2-D ``mask`` is broadcast over the z-dimension so that the same
        horizontal stencil-capability map is applied at every depth level.
        Adaptively falls back to WENO3 or 1st-order upwind near coastlines.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        u : Float[Array, "Nz Ny Nx"]
            East-face velocity.
        mask : ArakawaCGridMask
            2-D Arakawa C-grid mask (broadcast over z).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            East-face flux with zero ghost ring.
        """
        amasks = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        m5 = amasks[6]  # (Ny, Nx)
        m3 = amasks[4]
        fe_w5 = self.weno5_x(h, u)
        fe_w3 = self.weno3_x(h, u)
        fe_u1 = self.upwind1_x(h, u)
        # Select tier based on upwind-cell stencil capability, broadcast mask over z
        # Positive flow: upwind cell is (j, i)   → mask at [1:-1, 1:-1]
        # Negative flow: upwind cell is (j, i+1) → mask at [1:-1, 2:]
        pos_flow = u[1:-1, 1:-1, 1:-1] >= 0.0
        use_w5 = jnp.where(pos_flow, m5[None, 1:-1, 1:-1], m5[None, 1:-1, 2:])
        use_w3 = jnp.where(pos_flow, m3[None, 1:-1, 1:-1], m3[None, 1:-1, 2:])
        selected = jnp.where(
            use_w5,
            fe_w5[1:-1, 1:-1, 1:-1],
            jnp.where(use_w3, fe_w3[1:-1, 1:-1, 1:-1], fe_u1[1:-1, 1:-1, 1:-1]),
        )
        out = interior(selected, h)
        return out

    def weno5_y_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask,
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO north-face flux over all z-levels with mask-aware adaptive stencil.

        The 2-D ``mask`` is broadcast over the z-dimension so that the same
        horizontal stencil-capability map is applied at every depth level.
        Adaptively falls back to WENO3 or 1st-order upwind near coastlines.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        v : Float[Array, "Nz Ny Nx"]
            North-face velocity.
        mask : ArakawaCGridMask
            2-D Arakawa C-grid mask (broadcast over z).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            North-face flux with zero ghost ring.
        """
        amasks = mask.get_adaptive_masks(direction="y", stencil_sizes=(2, 4, 6))
        m5 = amasks[6]
        m3 = amasks[4]
        fn_w5 = self.weno5_y(h, v)
        fn_w3 = self.weno3_y(h, v)
        fn_u1 = self.upwind1_y(h, v)
        # Select tier based on upwind-cell stencil capability, broadcast mask over z
        # Positive flow: upwind cell is (j, i)   → mask at [1:-1, 1:-1]
        # Negative flow: upwind cell is (j+1, i) → mask at [2:, 1:-1]
        pos_flow = v[1:-1, 1:-1, 1:-1] >= 0.0
        use_w5 = jnp.where(pos_flow, m5[None, 1:-1, 1:-1], m5[None, 2:, 1:-1])
        use_w3 = jnp.where(pos_flow, m3[None, 1:-1, 1:-1], m3[None, 2:, 1:-1])
        selected = jnp.where(
            use_w5,
            fn_w5[1:-1, 1:-1, 1:-1],
            jnp.where(use_w3, fn_w3[1:-1, 1:-1, 1:-1], fn_u1[1:-1, 1:-1, 1:-1]),
        )
        out = interior(selected, h)
        return out

    def wenoz5_x_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask,
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO-Z east-face flux over all z-levels with mask-aware adaptive stencil.

        The 2-D ``mask`` is broadcast over the z-dimension so that the same
        horizontal stencil-capability map is applied at every depth level.
        Adaptively falls back to WENO-Z-3 or 1st-order upwind near coastlines.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        u : Float[Array, "Nz Ny Nx"]
            East-face velocity.
        mask : ArakawaCGridMask
            2-D Arakawa C-grid mask (broadcast over z).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            East-face flux with zero ghost ring.
        """
        amasks = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        m5 = amasks[6]
        m3 = amasks[4]
        fe_w5 = self.wenoz5_x(h, u)
        fe_w3 = self.wenoz3_x(h, u)
        fe_u1 = self.upwind1_x(h, u)
        pos_flow = u[1:-1, 1:-1, 1:-1] >= 0.0
        use_w5 = jnp.where(pos_flow, m5[None, 1:-1, 1:-1], m5[None, 1:-1, 2:])
        use_w3 = jnp.where(pos_flow, m3[None, 1:-1, 1:-1], m3[None, 1:-1, 2:])
        selected = jnp.where(
            use_w5,
            fe_w5[1:-1, 1:-1, 1:-1],
            jnp.where(use_w3, fe_w3[1:-1, 1:-1, 1:-1], fe_u1[1:-1, 1:-1, 1:-1]),
        )
        out = interior(selected, h)
        return out

    def wenoz5_y_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask,
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO-Z north-face flux over all z-levels with mask-aware adaptive stencil.

        The 2-D ``mask`` is broadcast over the z-dimension so that the same
        horizontal stencil-capability map is applied at every depth level.
        Adaptively falls back to WENO-Z-3 or 1st-order upwind near coastlines.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        v : Float[Array, "Nz Ny Nx"]
            North-face velocity.
        mask : ArakawaCGridMask
            2-D Arakawa C-grid mask (broadcast over z).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            North-face flux with zero ghost ring.
        """
        amasks = mask.get_adaptive_masks(direction="y", stencil_sizes=(2, 4, 6))
        m5 = amasks[6]
        m3 = amasks[4]
        fn_w5 = self.wenoz5_y(h, v)
        fn_w3 = self.wenoz3_y(h, v)
        fn_u1 = self.upwind1_y(h, v)
        pos_flow = v[1:-1, 1:-1, 1:-1] >= 0.0
        use_w5 = jnp.where(pos_flow, m5[None, 1:-1, 1:-1], m5[None, 2:, 1:-1])
        use_w3 = jnp.where(pos_flow, m3[None, 1:-1, 1:-1], m3[None, 2:, 1:-1])
        selected = jnp.where(
            use_w5,
            fn_w5[1:-1, 1:-1, 1:-1],
            jnp.where(use_w3, fn_w3[1:-1, 1:-1, 1:-1], fn_u1[1:-1, 1:-1, 1:-1]),
        )
        out = interior(selected, h)
        return out
