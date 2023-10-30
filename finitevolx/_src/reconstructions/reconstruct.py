import functools as ft
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from finitevolx._src.masks.masks import FaceMask
from finitevolx._src.reconstructions.upwind import (
    plusminus,
    upwind_1pt,
    upwind_2pt_bnds,
    upwind_3pt,
    upwind_3pt_bnds,
    upwind_5pt,
)


def reconstruct(
    q: Array,
    u: Array,
    dim: int,
    u_mask: Optional[FaceMask] = None,
    method: str = "wenoz",
    num_pts: int = 5,
):
    if num_pts == 1:
        return reconstruct_1pt(q=q, u=u, dim=dim, u_mask=u_mask)
    elif num_pts == 3:
        return reconstruct_3pt(q=q, u=u, dim=dim, u_mask=u_mask, method=method)
    elif num_pts == 5:
        return reconstruct_5pt(q=q, u=u, dim=dim, u_mask=u_mask, method=method)
    else:
        msg = "Unrecognized num_pts."
        msg += "\nShould be 1, 3, or 5"
        msg += f"\nGiven: {num_pts}"
        raise ValueError(msg)


def reconstruct_1pt(
    q: Array, u: Array, dim: int, u_mask: Optional[FaceMask] = None
) -> Array:
    qi_left_1pt, qi_right_1pt = upwind_1pt(q=q, dim=dim)
    u_pos, u_neg = plusminus(u)
    flux = u_pos * qi_left_1pt + u_neg * qi_right_1pt
    if u_mask:
        flux *= u_mask.distbound1
    return flux


def reconstruct_3pt(
    q: Array,
    u: Array,
    dim: int,
    u_mask: Optional[FaceMask] = None,
    method: str = "weno",
) -> Array:
    if u_mask:
        return _reconstruct_3pt_mask(q=q, u=u, dim=dim, u_mask=u_mask, method=method)
    else:
        return _reconstruct_3pt_nomask(q=q, u=u, dim=dim, method=method)


def _reconstruct_3pt_nomask(
    q: Array, u: Array, dim: int, method: str = "linear"
) -> Array:
    # get number of points
    num_pts = q.shape[dim]

    # define slicers
    dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

    qi_left_interior, qi_right_interior = upwind_3pt(q=q, dim=dim, method=method)

    qi_left_interior = dyn_slicer(qi_left_interior, 0, num_pts - 3)
    qi_right_interior = dyn_slicer(qi_right_interior, 1, num_pts - 3)

    # left-right boundary slices (linear only!)
    qi_left_bd, qi_right_bd = upwind_2pt_bnds(q=q, dim=dim, method="linear")

    # concatenate
    qi_left = jnp.concatenate([qi_left_bd, qi_left_interior, qi_right_bd], axis=dim)
    qi_right = jnp.concatenate([qi_left_bd, qi_right_interior, qi_right_bd], axis=dim)

    # calculate +ve and -ve points
    u_pos, u_neg = plusminus(u)

    # calculate upwind flux
    flux = u_pos * qi_left + u_neg * qi_right

    return flux


def _reconstruct_3pt_mask(
    q: Array,
    u: Array,
    dim: int,
    u_mask: FaceMask,
    method: str = "linear",
):
    num_dims = q.ndim

    q = jnp.swapaxes(q, dim, -1)
    u = jnp.swapaxes(u, dim, -1)

    # get padding
    extra_dims_pad = ((0, 0),) * (num_dims - 1)
    pad_left_3pt = extra_dims_pad + ((1, 0),)
    pad_right_3pt = extra_dims_pad + ((0, 1),)

    # # get padding
    # if dim == 0:
    #     pad_left = ((1, 0), (0, 0))
    #     pad_right = ((0, 1), (0, 0))
    # elif dim == 1:
    #     pad_left = ((0, 0), (1, 0))
    #     pad_right = ((0, 0), (0, 1))
    # else:
    #     msg = f"Dims should be between 0 and 1!"
    #     msg += f"\nDims: {dim}"
    #     raise ValueError(msg)

    # 1 point flux
    qi_left_i_1pt, qi_right_i_1pt = upwind_1pt(q=q, dim=-1)

    # 3 point flux
    qi_left_i_3pt, qi_right_i_3pt = upwind_3pt(q=q, dim=-1, method=method)

    # add padding
    qi_left_i_3pt = jnp.pad(qi_left_i_3pt, pad_width=pad_left_3pt)
    qi_right_i_3pt = jnp.pad(qi_right_i_3pt, pad_width=pad_right_3pt)

    # calculate +ve and -ve points
    u_pos, u_neg = plusminus(u)

    # calculate upwind flux
    flux_1pt = u_pos * qi_left_i_1pt + u_neg * qi_right_i_1pt
    flux_3pt = u_pos * qi_left_i_3pt + u_neg * qi_right_i_3pt

    # unswap axis
    flux_1pt = jnp.swapaxes(flux_1pt, -1, dim)
    flux_3pt = jnp.swapaxes(flux_3pt, -1, dim)

    # calculate total flux
    flux = flux_1pt * u_mask.distbound1 + flux_3pt * u_mask.distbound2plus

    return flux


def reconstruct_5pt(
    q: Array,
    u: Array,
    dim: int,
    u_mask: Optional[FaceMask] = None,
    method: str = "wenoz",
) -> Array:
    if u_mask is not None:
        return _reconstruct_5pt_mask(q=q, u=u, dim=dim, u_mask=u_mask, method=method)
    else:
        return _reconstruct_5pt_nomask(q=q, u=u, dim=dim, method=method)


def _reconstruct_5pt_nomask(
    q: Array, u: Array, dim: int, method: str = "linear"
) -> Array:

    # define slicers
    dyn_slicer = ft.partial(jax.lax.dynamic_slice_in_dim, axis=dim)

    # 5-pts inside domain
    qi_left_interior, qi_right_interior = upwind_5pt(q=q, dim=dim, method="linear")

    # 3pts-near boundary
    qi_left_b, qi_right_b = upwind_3pt_bnds(q, dim=dim, method=method)

    qi_left_b0 = dyn_slicer(qi_left_b, 0, 1)
    qi_left_m = dyn_slicer(qi_left_b, -1, 1)

    qi_right_0 = dyn_slicer(qi_right_b, 0, 1)
    qi_right_bm = dyn_slicer(qi_right_b, -1, 1)

    # 1pts at end-points
    qi_left_0, qi_right_m = upwind_2pt_bnds(q=q, dim=dim)

    # concatenate
    qi_left = jnp.concatenate(
        [qi_left_0, qi_left_b0, qi_left_interior, qi_left_m], axis=dim
    )
    qi_right = jnp.concatenate(
        [qi_right_0, qi_right_interior, qi_right_bm, qi_right_m], axis=dim
    )

    # calculate +ve and -ve points
    u_pos, u_neg = plusminus(u)

    # calculate upwind flux
    flux = u_pos * qi_left + u_neg * qi_right

    return flux


def _reconstruct_5pt_mask(
    q: Array,
    u: Array,
    u_mask: FaceMask,
    dim: int,
    method: str = "linear",
):
    """Tasks - ++"""
    # transpose all dimensions
    num_dims = q.ndim

    q = jnp.swapaxes(q, dim, -1)
    u = jnp.swapaxes(u, dim, -1)

    # get padding
    extra_dims_pad = ((0, 0),) * (num_dims - 1)
    pad_left_3pt = extra_dims_pad + ((1, 0),)
    pad_right_3pt = extra_dims_pad + ((0, 1),)
    pad_left_5pt = extra_dims_pad + ((2, 1),)
    pad_right_5pt = extra_dims_pad + ((1, 2),)

    # 1 point flux
    qi_left_i_1pt, qi_right_i_1pt = upwind_1pt(q=q, dim=-1)

    # 3 point flux
    qi_left_i_3pt, qi_right_i_3pt = upwind_3pt(q=q, dim=-1, method="linear")

    # add padding
    qi_left_i_3pt = jnp.pad(qi_left_i_3pt, pad_width=pad_left_3pt)
    qi_right_i_3pt = jnp.pad(qi_right_i_3pt, pad_width=pad_right_3pt)

    # 5 point flux
    qi_left_i_5pt, qi_right_i_5pt = upwind_5pt(q=q, dim=-1, method=method)

    # add padding
    qi_left_i_5pt = jnp.pad(qi_left_i_5pt, pad_width=pad_left_5pt)
    qi_right_i_5pt = jnp.pad(qi_right_i_5pt, pad_width=pad_right_5pt)

    # calculate +ve and -ve points
    u_pos, u_neg = plusminus(u)

    # calculate upwind flux
    # NOTE: This was in the code originally but it was unstable...
    # flux_1pt = u * linear_2pts(qi_left_i_1pt, qi_right_i_1pt)
    flux_1pt = u_pos * qi_left_i_1pt + u_neg * qi_right_i_1pt
    flux_3pt = u_pos * qi_left_i_3pt + u_neg * qi_right_i_3pt
    flux_5pt = u_pos * qi_left_i_5pt + u_neg * qi_right_i_5pt

    # unswap axis
    flux_1pt = jnp.swapaxes(flux_1pt, -1, dim)
    flux_3pt = jnp.swapaxes(flux_3pt, -1, dim)
    flux_5pt = jnp.swapaxes(flux_5pt, -1, dim)

    # calculate total flux
    flux = (
        flux_1pt * u_mask.distbound1
        + flux_3pt * u_mask.distbound2
        + flux_5pt * u_mask.distbound3plus
    )

    return flux
