"""
Face-value reconstruction operators for advective fluxes on Arakawa C-grids.

All operators follow the interior-point idiom (write only [1:-1, 1:-1]).

Notation
--------
  fe[j, i+1/2]  east-face flux
  fn[j+1/2, i]  north-face flux
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.masks.cgrid_mask import ArakawaCGridMask
from finitevolx._src.reconstructions.weno import (
    weno_3pts as _weno3,
    weno_3pts_improved as _wenoz3,
    weno_5pts as _weno5,
    weno_5pts_improved as _wenoz5,
)


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
        out = jnp.zeros_like(h)
        # fe[i+1/2] = 1/2 * (h[i] + h[i+1]) * u[i+1/2]
        out = out.at[1:-1].set(0.5 * (h[1:-1] + h[2:]) * u[1:-1])
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
        out = jnp.zeros_like(h)
        # fe[i+1/2] = upwind(h, u) * u[i+1/2]
        h_face = jnp.where(u[1:-1] >= 0.0, h[1:-1], h[2:])
        out = out.at[1:-1].set(h_face * u[1:-1])
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
        out = jnp.zeros_like(h)
        # h_face_pos[i+1/2] = 3/2*h[i] - 1/2*h[i-1]
        h_pos = 1.5 * h[1:-1] - 0.5 * h[:-2]
        # h_face_neg[i+1/2] = 3/2*h[i+1] - 1/2*h[i+2]
        # Use 2nd-order where i+2 is available (all except last interior face)
        h_neg_interior = 1.5 * h[2:-1] - 0.5 * h[3:]
        # Use 1st-order upwind on east boundary face (h_face = h[i+1])
        h_neg_boundary = h[-1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary])
        h_face = jnp.where(u[1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1].set(h_face * u[1:-1])
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
        out = jnp.zeros_like(h)
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
        out = out.at[1:-1].set(h_face * u[1:-1])
        return out

    def weno3_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """3-point WENO east-face flux with boundary fallback.

        Positive flow:  h_face[i+1/2] = WENO3(h[i-1], h[i],   h[i+1])
        Negative flow:  h_face[i+1/2] = WENO3(h[i+2], h[i+1], h[i])
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        out = jnp.zeros_like(h)
        # WENO-3 positive: left-biased stencil, valid for all interior faces
        h_pos = _weno3(h[:-2], h[1:-1], h[2:])
        # WENO-3 negative: right-biased stencil, valid for i+2 < Nx
        h_neg_interior = _weno3(h[3:], h[2:-1], h[1:-2])
        # 1st-order upwind fallback at east boundary
        h_neg_boundary = h[-1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary])
        h_face = jnp.where(u[1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1].set(h_face * u[1:-1])
        return out

    def wenoz3_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """3-point WENO-Z east-face flux with boundary fallback.

        Positive flow:  h_face[i+1/2] = WENOZ3(h[i-1], h[i],   h[i+1])
        Negative flow:  h_face[i+1/2] = WENOZ3(h[i+2], h[i+1], h[i])
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        out = jnp.zeros_like(h)
        # WENO-Z-3 positive: left-biased stencil, valid for all interior faces
        h_pos = _wenoz3(h[:-2], h[1:-1], h[2:])
        # WENO-Z-3 negative: right-biased stencil, valid for i+2 < Nx
        h_neg_interior = _wenoz3(h[3:], h[2:-1], h[1:-2])
        # 1st-order upwind fallback at east boundary
        h_neg_boundary = h[-1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary])
        h_face = jnp.where(u[1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1].set(h_face * u[1:-1])
        return out

    def weno5_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """5-point WENO east-face flux with sign-dependent boundary fallbacks.

        Positive flow:  h_face[i+1/2] = WENO5(h[i-2], h[i-1], h[i], h[i+1], h[i+2])
            for i = 2..Nx-3; WENO3 fallback at i = 1 and i = Nx-2.
        Negative flow:  h_face[i+1/2] = WENO5(h[i+3], h[i+2], h[i+1], h[i], h[i-1])
            for i = 2..Nx-3; WENO3 fallback at i = 1; 1st-order upwind at i = Nx-2.
        """
        out = jnp.zeros_like(h)
        # WENO-5 positive: valid for interior faces i=2..Nx-3 (needs h[i-2..i+2])
        h5_pos_interior = _weno5(h[:-4], h[1:-3], h[2:-2], h[3:-1], h[4:])
        # WENO-3 fallback at first interior face (i=1, h[i-2] = h[-1] wraps)
        h3_pos_first = _weno3(h[0:1], h[1:2], h[2:3])
        # WENO-3 fallback at last interior face (i=Nx-2, h[i+2] = h[Nx] out of bounds)
        h3_pos_last = _weno3(h[-3:-2], h[-2:-1], h[-1:])
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_interior, h3_pos_last])
        # WENO-5 negative: valid for interior faces i=2..Nx-3
        h5_neg_interior = _weno5(h[4:], h[3:-1], h[2:-2], h[1:-3], h[:-4])
        # WENO-3 fallback at first interior face (i=1)
        h3_neg_first = _weno3(h[3:4], h[2:3], h[1:2])
        # 1st-order upwind fallback at east boundary (i=Nx-2)
        h1_neg_last = h[-1:]
        h_neg = jnp.concatenate([h3_neg_first, h5_neg_interior, h1_neg_last])
        h_face = jnp.where(u[1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1].set(h_face * u[1:-1])
        return out

    def wenoz5_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """5-point WENO-Z east-face flux with sign-dependent boundary fallbacks.

        Positive flow:  h_face[i+1/2] = WENOZ5(h[i-2], h[i-1], h[i], h[i+1], h[i+2])
            for i = 2..Nx-3; WENO-Z-3 fallback at i = 1 and i = Nx-2.
        Negative flow:  h_face[i+1/2] = WENOZ5(h[i+3], h[i+2], h[i+1], h[i], h[i-1])
            for i = 2..Nx-3; WENO-Z-3 fallback at i = 1; 1st-order upwind at i = Nx-2.
        """
        out = jnp.zeros_like(h)
        # WENO-Z-5 positive: valid for interior faces i=2..Nx-3
        h5_pos_interior = _wenoz5(h[:-4], h[1:-3], h[2:-2], h[3:-1], h[4:])
        # WENO-Z-3 fallback at first interior face (i=1)
        h3_pos_first = _wenoz3(h[0:1], h[1:2], h[2:3])
        # WENO-Z-3 fallback at last interior face (i=Nx-2)
        h3_pos_last = _wenoz3(h[-3:-2], h[-2:-1], h[-1:])
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_interior, h3_pos_last])
        # WENO-Z-5 negative: valid for interior faces i=2..Nx-3
        h5_neg_interior = _wenoz5(h[4:], h[3:-1], h[2:-2], h[1:-3], h[:-4])
        # WENO-Z-3 fallback at first interior face (i=1)
        h3_neg_first = _wenoz3(h[3:4], h[2:3], h[1:2])
        # 1st-order upwind fallback at east boundary (i=Nx-2)
        h1_neg_last = h[-1:]
        h_neg = jnp.concatenate([h3_neg_first, h5_neg_interior, h1_neg_last])
        h_face = jnp.where(u[1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1].set(h_face * u[1:-1])
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
        out = jnp.zeros_like(h)
        # fe[j, i+1/2] = 1/2 * (h[j, i] + h[j, i+1]) * u[j, i+1/2]
        out = out.at[1:-1, 1:-1].set(
            0.5 * (h[1:-1, 1:-1] + h[1:-1, 2:]) * u[1:-1, 1:-1]
        )
        return out

    def naive_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Naive (centred) north-face flux.

        fn[j+1/2, i] = 1/2 * (h[j, i] + h[j+1, i]) * v[j+1/2, i]
        """
        out = jnp.zeros_like(h)
        # fn[j+1/2, i] = 1/2 * (h[j, i] + h[j+1, i]) * v[j+1/2, i]
        out = out.at[1:-1, 1:-1].set(
            0.5 * (h[1:-1, 1:-1] + h[2:, 1:-1]) * v[1:-1, 1:-1]
        )
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
        out = jnp.zeros_like(h)
        # fe[j, i+1/2] = upwind1(h, u) * u[j, i+1/2]
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h[1:-1, 1:-1], h[1:-1, 2:])
        out = out.at[1:-1, 1:-1].set(h_face * u[1:-1, 1:-1])
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
        out = jnp.zeros_like(h)
        # fn[j+1/2, i] = upwind1(h, v) * v[j+1/2, i]
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h[1:-1, 1:-1], h[2:, 1:-1])
        out = out.at[1:-1, 1:-1].set(h_face * v[1:-1, 1:-1])
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
        out = jnp.zeros_like(h)
        # 2nd-order positive stencil (uses h[j,i-1])
        h_pos = 1.5 * h[1:-1, 1:-1] - 0.5 * h[1:-1, :-2]
        # 2nd-order negative stencil (uses h[j,i+2])
        # Use 2nd-order where i+2 is available (all except last interior column)
        h_neg_interior = 1.5 * h[1:-1, 2:-1] - 0.5 * h[1:-1, 3:]
        # Use 1st-order upwind on east boundary column (h_face = h[j,i+1])
        h_neg_boundary = h[1:-1, -1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=1)
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * u[1:-1, 1:-1])
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
        out = jnp.zeros_like(h)
        # 2nd-order positive stencil (uses h[j-1,i])
        h_pos = 1.5 * h[1:-1, 1:-1] - 0.5 * h[:-2, 1:-1]
        # 2nd-order negative stencil (uses h[j+2,i])
        # Use 2nd-order where j+2 is available (all except last interior row)
        h_neg_interior = 1.5 * h[2:-1, 1:-1] - 0.5 * h[3:, 1:-1]
        # Use 1st-order upwind on north boundary row (h_face = h[j+1,i])
        h_neg_boundary = h[-1:, 1:-1]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=0)
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * v[1:-1, 1:-1])
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
        out = jnp.zeros_like(h)
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
        out = out.at[1:-1, 1:-1].set(h_face * u[1:-1, 1:-1])
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
        out = jnp.zeros_like(h)
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
        out = out.at[1:-1, 1:-1].set(h_face * v[1:-1, 1:-1])
        return out

    def weno3_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3-point WENO east-face flux with boundary fallback.

        Positive flow:  fe[j,i+1/2] = WENO3(h[j,i-1], h[j,i],   h[j,i+1]) * u
        Negative flow:  fe[j,i+1/2] = WENO3(h[j,i+2], h[j,i+1], h[j,i])   * u
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        out = jnp.zeros_like(h)
        # WENO-3 positive: left-biased, valid for all interior faces
        h_pos = _weno3(h[1:-1, :-2], h[1:-1, 1:-1], h[1:-1, 2:])
        # WENO-3 negative: right-biased, valid for i+2 < Nx
        h_neg_interior = _weno3(h[1:-1, 3:], h[1:-1, 2:-1], h[1:-1, 1:-2])
        # 1st-order upwind fallback at east boundary column
        h_neg_boundary = h[1:-1, -1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=1)
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * u[1:-1, 1:-1])
        return out

    def weno3_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3-point WENO north-face flux with boundary fallback.

        Positive flow:  fn[j+1/2,i] = WENO3(h[j-1,i], h[j,i],   h[j+1,i]) * v
        Negative flow:  fn[j+1/2,i] = WENO3(h[j+2,i], h[j+1,i], h[j,i])   * v
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        out = jnp.zeros_like(h)
        # WENO-3 positive: left-biased, valid for all interior faces
        h_pos = _weno3(h[:-2, 1:-1], h[1:-1, 1:-1], h[2:, 1:-1])
        # WENO-3 negative: right-biased, valid for j+2 < Ny
        h_neg_interior = _weno3(h[3:, 1:-1], h[2:-1, 1:-1], h[1:-2, 1:-1])
        # 1st-order upwind fallback at north boundary row
        h_neg_boundary = h[-1:, 1:-1]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=0)
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * v[1:-1, 1:-1])
        return out

    def wenoz3_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3-point WENO-Z east-face flux with boundary fallback.

        Positive flow:  fe[j,i+1/2] = WENOZ3(h[j,i-1], h[j,i],   h[j,i+1]) * u
        Negative flow:  fe[j,i+1/2] = WENOZ3(h[j,i+2], h[j,i+1], h[j,i])   * u
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        out = jnp.zeros_like(h)
        # WENO-Z-3 positive: left-biased, valid for all interior faces
        h_pos = _wenoz3(h[1:-1, :-2], h[1:-1, 1:-1], h[1:-1, 2:])
        # WENO-Z-3 negative: right-biased, valid for i+2 < Nx
        h_neg_interior = _wenoz3(h[1:-1, 3:], h[1:-1, 2:-1], h[1:-1, 1:-2])
        # 1st-order upwind fallback at east boundary column
        h_neg_boundary = h[1:-1, -1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=1)
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * u[1:-1, 1:-1])
        return out

    def wenoz3_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3-point WENO-Z north-face flux with boundary fallback.

        Positive flow:  fn[j+1/2,i] = WENOZ3(h[j-1,i], h[j,i],   h[j+1,i]) * v
        Negative flow:  fn[j+1/2,i] = WENOZ3(h[j+2,i], h[j+1,i], h[j,i])   * v
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        out = jnp.zeros_like(h)
        # WENO-Z-3 positive: left-biased, valid for all interior faces
        h_pos = _wenoz3(h[:-2, 1:-1], h[1:-1, 1:-1], h[2:, 1:-1])
        # WENO-Z-3 negative: right-biased, valid for j+2 < Ny
        h_neg_interior = _wenoz3(h[3:, 1:-1], h[2:-1, 1:-1], h[1:-2, 1:-1])
        # 1st-order upwind fallback at north boundary row
        h_neg_boundary = h[-1:, 1:-1]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=0)
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * v[1:-1, 1:-1])
        return out

    def weno5_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO east-face flux with sign-dependent boundary fallbacks.

        Positive flow:  fe[j,i+1/2] = WENO5(h[j,i-2..i+2]) * u
            for i = 2..Nx-3; WENO3 fallback at i = 1 and i = Nx-2.
        Negative flow:  fe[j,i+1/2] = WENO5(h[j,i+3..i-1]) * u
            for i = 2..Nx-3; WENO3 fallback at i = 1; 1st-order upwind at i = Nx-2.
        """
        out = jnp.zeros_like(h)
        # WENO-5 positive: valid for i=2..Nx-3
        h5_pos_int = _weno5(
            h[1:-1, :-4], h[1:-1, 1:-3], h[1:-1, 2:-2], h[1:-1, 3:-1], h[1:-1, 4:]
        )
        # WENO-3 fallback at first interior column (i=1)
        h3_pos_first = _weno3(h[1:-1, 0:1], h[1:-1, 1:2], h[1:-1, 2:3])
        # WENO-3 fallback at last interior column (i=Nx-2)
        h3_pos_last = _weno3(h[1:-1, -3:-2], h[1:-1, -2:-1], h[1:-1, -1:])
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=1)
        # WENO-5 negative: valid for i=2..Nx-3
        h5_neg_int = _weno5(
            h[1:-1, 4:], h[1:-1, 3:-1], h[1:-1, 2:-2], h[1:-1, 1:-3], h[1:-1, :-4]
        )
        # WENO-3 fallback at first interior column (i=1)
        h3_neg_first = _weno3(h[1:-1, 3:4], h[1:-1, 2:3], h[1:-1, 1:2])
        # 1st-order upwind fallback at east boundary column (i=Nx-2)
        h1_neg_last = h[1:-1, -1:]
        h_neg = jnp.concatenate([h3_neg_first, h5_neg_int, h1_neg_last], axis=1)
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * u[1:-1, 1:-1])
        return out

    def weno5_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO north-face flux with sign-dependent boundary fallbacks.

        Positive flow:  fn[j+1/2,i] = WENO5(h[j-2..j+2,i]) * v
            for j = 2..Ny-3; WENO3 fallback at j = 1 and j = Ny-2.
        Negative flow:  fn[j+1/2,i] = WENO5(h[j+3..j-1,i]) * v
            for j = 2..Ny-3; WENO3 fallback at j = 1; 1st-order upwind at j = Ny-2.
        """
        out = jnp.zeros_like(h)
        # WENO-5 positive: valid for j=2..Ny-3
        h5_pos_int = _weno5(
            h[:-4, 1:-1], h[1:-3, 1:-1], h[2:-2, 1:-1], h[3:-1, 1:-1], h[4:, 1:-1]
        )
        # WENO-3 fallback at first interior row (j=1)
        h3_pos_first = _weno3(h[0:1, 1:-1], h[1:2, 1:-1], h[2:3, 1:-1])
        # WENO-3 fallback at last interior row (j=Ny-2)
        h3_pos_last = _weno3(h[-3:-2, 1:-1], h[-2:-1, 1:-1], h[-1:, 1:-1])
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=0)
        # WENO-5 negative: valid for j=2..Ny-3
        h5_neg_int = _weno5(
            h[4:, 1:-1], h[3:-1, 1:-1], h[2:-2, 1:-1], h[1:-3, 1:-1], h[:-4, 1:-1]
        )
        # WENO-3 fallback at first interior row (j=1)
        h3_neg_first = _weno3(h[3:4, 1:-1], h[2:3, 1:-1], h[1:2, 1:-1])
        # 1st-order upwind fallback at north boundary row (j=Ny-2)
        h1_neg_last = h[-1:, 1:-1]
        h_neg = jnp.concatenate([h3_neg_first, h5_neg_int, h1_neg_last], axis=0)
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * v[1:-1, 1:-1])
        return out

    def wenoz5_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO-Z east-face flux with sign-dependent boundary fallbacks.

        Positive flow:  fe[j,i+1/2] = WENOZ5(h[j,i-2..i+2]) * u
            for i = 2..Nx-3; WENO-Z-3 fallback at i = 1 and i = Nx-2.
        Negative flow:  fe[j,i+1/2] = WENOZ5(h[j,i+3..i-1]) * u
            for i = 2..Nx-3; WENO-Z-3 fallback at i = 1; 1st-order upwind at i = Nx-2.
        """
        out = jnp.zeros_like(h)
        # WENO-Z-5 positive: valid for i=2..Nx-3
        h5_pos_int = _wenoz5(
            h[1:-1, :-4], h[1:-1, 1:-3], h[1:-1, 2:-2], h[1:-1, 3:-1], h[1:-1, 4:]
        )
        # WENO-Z-3 fallback at first interior column (i=1)
        h3_pos_first = _wenoz3(h[1:-1, 0:1], h[1:-1, 1:2], h[1:-1, 2:3])
        # WENO-Z-3 fallback at last interior column (i=Nx-2)
        h3_pos_last = _wenoz3(h[1:-1, -3:-2], h[1:-1, -2:-1], h[1:-1, -1:])
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=1)
        # WENO-Z-5 negative: valid for i=2..Nx-3
        h5_neg_int = _wenoz5(
            h[1:-1, 4:], h[1:-1, 3:-1], h[1:-1, 2:-2], h[1:-1, 1:-3], h[1:-1, :-4]
        )
        # WENO-Z-3 fallback at first interior column (i=1)
        h3_neg_first = _wenoz3(h[1:-1, 3:4], h[1:-1, 2:3], h[1:-1, 1:2])
        # 1st-order upwind fallback at east boundary column (i=Nx-2)
        h1_neg_last = h[1:-1, -1:]
        h_neg = jnp.concatenate([h3_neg_first, h5_neg_int, h1_neg_last], axis=1)
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * u[1:-1, 1:-1])
        return out

    def wenoz5_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO-Z north-face flux with sign-dependent boundary fallbacks.

        Positive flow:  fn[j+1/2,i] = WENOZ5(h[j-2..j+2,i]) * v
            for j = 2..Ny-3; WENO-Z-3 fallback at j = 1 and j = Ny-2.
        Negative flow:  fn[j+1/2,i] = WENOZ5(h[j+3..j-1,i]) * v
            for j = 2..Ny-3; WENO-Z-3 fallback at j = 1; 1st-order upwind at j = Ny-2.
        """
        out = jnp.zeros_like(h)
        # WENO-Z-5 positive: valid for j=2..Ny-3
        h5_pos_int = _wenoz5(
            h[:-4, 1:-1], h[1:-3, 1:-1], h[2:-2, 1:-1], h[3:-1, 1:-1], h[4:, 1:-1]
        )
        # WENO-Z-3 fallback at first interior row (j=1)
        h3_pos_first = _wenoz3(h[0:1, 1:-1], h[1:2, 1:-1], h[2:3, 1:-1])
        # WENO-Z-3 fallback at last interior row (j=Ny-2)
        h3_pos_last = _wenoz3(h[-3:-2, 1:-1], h[-2:-1, 1:-1], h[-1:, 1:-1])
        h_pos = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=0)
        # WENO-Z-5 negative: valid for j=2..Ny-3
        h5_neg_int = _wenoz5(
            h[4:, 1:-1], h[3:-1, 1:-1], h[2:-2, 1:-1], h[1:-3, 1:-1], h[:-4, 1:-1]
        )
        # WENO-Z-3 fallback at first interior row (j=1)
        h3_neg_first = _wenoz3(h[3:4, 1:-1], h[2:3, 1:-1], h[1:2, 1:-1])
        # 1st-order upwind fallback at north boundary row (j=Ny-2)
        h1_neg_last = h[-1:, 1:-1]
        h_neg = jnp.concatenate([h3_neg_first, h5_neg_int, h1_neg_last], axis=0)
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * v[1:-1, 1:-1])
        return out

    def weno5_x_masked(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask,
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO east-face flux with mask-aware adaptive stencil selection.

        Uses :meth:`ArakawaCGridMask.get_adaptive_masks` to adaptively choose
        the highest-order WENO stencil available at each grid point, falling
        back to WENO3 or 1st-order upwind near coastlines or irregular
        boundaries.

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
        out = jnp.zeros_like(h)
        amasks = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        m5 = amasks[6]  # WENO5 stencil available
        m3 = amasks[4]  # WENO3 stencil available (but not WENO5)
        # --- WENO5 face values ---
        h5_pos_int = _weno5(
            h[1:-1, :-4], h[1:-1, 1:-3], h[1:-1, 2:-2], h[1:-1, 3:-1], h[1:-1, 4:]
        )
        h3_pos_first = _weno3(h[1:-1, 0:1], h[1:-1, 1:2], h[1:-1, 2:3])
        h3_pos_last = _weno3(h[1:-1, -3:-2], h[1:-1, -2:-1], h[1:-1, -1:])
        h_pos_w5 = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=1)
        h5_neg_int = _weno5(
            h[1:-1, 4:], h[1:-1, 3:-1], h[1:-1, 2:-2], h[1:-1, 1:-3], h[1:-1, :-4]
        )
        h3_neg_first = _weno3(h[1:-1, 3:4], h[1:-1, 2:3], h[1:-1, 1:2])
        h_neg_w5 = jnp.concatenate([h3_neg_first, h5_neg_int, h[1:-1, -1:]], axis=1)
        # --- WENO3 face values ---
        h_pos_w3 = _weno3(h[1:-1, :-2], h[1:-1, 1:-1], h[1:-1, 2:])
        h_neg_w3_int = _weno3(h[1:-1, 3:], h[1:-1, 2:-1], h[1:-1, 1:-2])
        h_neg_w3 = jnp.concatenate([h_neg_w3_int, h[1:-1, -1:]], axis=1)
        # --- 1st-order upwind face values ---
        h_pos_u1 = h[1:-1, 1:-1]
        h_neg_u1 = h[1:-1, 2:]
        # --- Mask-aware selection ---
        # Positive flow: upwind cell is (j, i) → mask at [1:-1, 1:-1]
        h_pos = jnp.where(
            m5[1:-1, 1:-1],
            h_pos_w5,
            jnp.where(m3[1:-1, 1:-1], h_pos_w3, h_pos_u1),
        )
        # Negative flow: upwind cell is (j, i+1) → mask at [1:-1, 2:]
        h_neg = jnp.where(
            m5[1:-1, 2:],
            h_neg_w5,
            jnp.where(m3[1:-1, 2:], h_neg_w3, h_neg_u1),
        )
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * u[1:-1, 1:-1])
        return out

    def weno5_y_masked(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask,
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO north-face flux with mask-aware adaptive stencil selection.

        Uses :meth:`ArakawaCGridMask.get_adaptive_masks` to adaptively choose
        the highest-order WENO stencil available at each grid point, falling
        back to WENO3 or 1st-order upwind near coastlines or irregular
        boundaries.

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
        out = jnp.zeros_like(h)
        amasks = mask.get_adaptive_masks(direction="y", stencil_sizes=(2, 4, 6))
        m5 = amasks[6]
        m3 = amasks[4]
        # --- WENO5 face values ---
        h5_pos_int = _weno5(
            h[:-4, 1:-1], h[1:-3, 1:-1], h[2:-2, 1:-1], h[3:-1, 1:-1], h[4:, 1:-1]
        )
        h3_pos_first = _weno3(h[0:1, 1:-1], h[1:2, 1:-1], h[2:3, 1:-1])
        h3_pos_last = _weno3(h[-3:-2, 1:-1], h[-2:-1, 1:-1], h[-1:, 1:-1])
        h_pos_w5 = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=0)
        h5_neg_int = _weno5(
            h[4:, 1:-1], h[3:-1, 1:-1], h[2:-2, 1:-1], h[1:-3, 1:-1], h[:-4, 1:-1]
        )
        h3_neg_first = _weno3(h[3:4, 1:-1], h[2:3, 1:-1], h[1:2, 1:-1])
        h_neg_w5 = jnp.concatenate([h3_neg_first, h5_neg_int, h[-1:, 1:-1]], axis=0)
        # --- WENO3 face values ---
        h_pos_w3 = _weno3(h[:-2, 1:-1], h[1:-1, 1:-1], h[2:, 1:-1])
        h_neg_w3_int = _weno3(h[3:, 1:-1], h[2:-1, 1:-1], h[1:-2, 1:-1])
        h_neg_w3 = jnp.concatenate([h_neg_w3_int, h[-1:, 1:-1]], axis=0)
        # --- 1st-order upwind face values ---
        h_pos_u1 = h[1:-1, 1:-1]
        h_neg_u1 = h[2:, 1:-1]
        # --- Mask-aware selection ---
        # Positive flow: upwind cell is (j, i) → mask at [1:-1, 1:-1]
        h_pos = jnp.where(
            m5[1:-1, 1:-1],
            h_pos_w5,
            jnp.where(m3[1:-1, 1:-1], h_pos_w3, h_pos_u1),
        )
        # Negative flow: upwind cell is (j+1, i) → mask at [2:, 1:-1]
        h_neg = jnp.where(
            m5[2:, 1:-1],
            h_neg_w5,
            jnp.where(m3[2:, 1:-1], h_neg_w3, h_neg_u1),
        )
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * v[1:-1, 1:-1])
        return out

    def wenoz5_x_masked(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask,
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO-Z east-face flux with mask-aware adaptive stencil selection.

        Uses :meth:`ArakawaCGridMask.get_adaptive_masks` to adaptively choose
        the highest-order WENO-Z stencil available at each grid point, falling
        back to WENO-Z-3 or 1st-order upwind near coastlines or irregular
        boundaries.

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
        out = jnp.zeros_like(h)
        amasks = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        m5 = amasks[6]
        m3 = amasks[4]
        # --- WENO-Z-5 face values ---
        h5_pos_int = _wenoz5(
            h[1:-1, :-4], h[1:-1, 1:-3], h[1:-1, 2:-2], h[1:-1, 3:-1], h[1:-1, 4:]
        )
        h3_pos_first = _wenoz3(h[1:-1, 0:1], h[1:-1, 1:2], h[1:-1, 2:3])
        h3_pos_last = _wenoz3(h[1:-1, -3:-2], h[1:-1, -2:-1], h[1:-1, -1:])
        h_pos_w5 = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=1)
        h5_neg_int = _wenoz5(
            h[1:-1, 4:], h[1:-1, 3:-1], h[1:-1, 2:-2], h[1:-1, 1:-3], h[1:-1, :-4]
        )
        h3_neg_first = _wenoz3(h[1:-1, 3:4], h[1:-1, 2:3], h[1:-1, 1:2])
        h_neg_w5 = jnp.concatenate([h3_neg_first, h5_neg_int, h[1:-1, -1:]], axis=1)
        # --- WENO-Z-3 face values ---
        h_pos_w3 = _wenoz3(h[1:-1, :-2], h[1:-1, 1:-1], h[1:-1, 2:])
        h_neg_w3_int = _wenoz3(h[1:-1, 3:], h[1:-1, 2:-1], h[1:-1, 1:-2])
        h_neg_w3 = jnp.concatenate([h_neg_w3_int, h[1:-1, -1:]], axis=1)
        # --- 1st-order upwind face values ---
        h_pos_u1 = h[1:-1, 1:-1]
        h_neg_u1 = h[1:-1, 2:]
        # --- Mask-aware selection ---
        h_pos = jnp.where(
            m5[1:-1, 1:-1],
            h_pos_w5,
            jnp.where(m3[1:-1, 1:-1], h_pos_w3, h_pos_u1),
        )
        h_neg = jnp.where(
            m5[1:-1, 2:],
            h_neg_w5,
            jnp.where(m3[1:-1, 2:], h_neg_w3, h_neg_u1),
        )
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * u[1:-1, 1:-1])
        return out

    def wenoz5_y_masked(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask,
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO-Z north-face flux with mask-aware adaptive stencil selection.

        Uses :meth:`ArakawaCGridMask.get_adaptive_masks` to adaptively choose
        the highest-order WENO-Z stencil available at each grid point, falling
        back to WENO-Z-3 or 1st-order upwind near coastlines or irregular
        boundaries.

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
        out = jnp.zeros_like(h)
        amasks = mask.get_adaptive_masks(direction="y", stencil_sizes=(2, 4, 6))
        m5 = amasks[6]
        m3 = amasks[4]
        # --- WENO-Z-5 face values ---
        h5_pos_int = _wenoz5(
            h[:-4, 1:-1], h[1:-3, 1:-1], h[2:-2, 1:-1], h[3:-1, 1:-1], h[4:, 1:-1]
        )
        h3_pos_first = _wenoz3(h[0:1, 1:-1], h[1:2, 1:-1], h[2:3, 1:-1])
        h3_pos_last = _wenoz3(h[-3:-2, 1:-1], h[-2:-1, 1:-1], h[-1:, 1:-1])
        h_pos_w5 = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=0)
        h5_neg_int = _wenoz5(
            h[4:, 1:-1], h[3:-1, 1:-1], h[2:-2, 1:-1], h[1:-3, 1:-1], h[:-4, 1:-1]
        )
        h3_neg_first = _wenoz3(h[3:4, 1:-1], h[2:3, 1:-1], h[1:2, 1:-1])
        h_neg_w5 = jnp.concatenate([h3_neg_first, h5_neg_int, h[-1:, 1:-1]], axis=0)
        # --- WENO-Z-3 face values ---
        h_pos_w3 = _wenoz3(h[:-2, 1:-1], h[1:-1, 1:-1], h[2:, 1:-1])
        h_neg_w3_int = _wenoz3(h[3:, 1:-1], h[2:-1, 1:-1], h[1:-2, 1:-1])
        h_neg_w3 = jnp.concatenate([h_neg_w3_int, h[-1:, 1:-1]], axis=0)
        # --- 1st-order upwind face values ---
        h_pos_u1 = h[1:-1, 1:-1]
        h_neg_u1 = h[2:, 1:-1]
        # --- Mask-aware selection ---
        h_pos = jnp.where(
            m5[1:-1, 1:-1],
            h_pos_w5,
            jnp.where(m3[1:-1, 1:-1], h_pos_w3, h_pos_u1),
        )
        h_neg = jnp.where(
            m5[2:, 1:-1],
            h_neg_w5,
            jnp.where(m3[2:, 1:-1], h_neg_w3, h_neg_u1),
        )
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * v[1:-1, 1:-1])
        return out


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
        out = jnp.zeros_like(h)
        out = out.at[1:-1, 1:-1, 1:-1].set(
            0.5 * (h[1:-1, 1:-1, 1:-1] + h[1:-1, 1:-1, 2:]) * u[1:-1, 1:-1, 1:-1]
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
        out = jnp.zeros_like(h)
        out = out.at[1:-1, 1:-1, 1:-1].set(
            0.5 * (h[1:-1, 1:-1, 1:-1] + h[1:-1, 2:, 1:-1]) * v[1:-1, 1:-1, 1:-1]
        )
        return out

    def upwind1_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """1st-order upwind east-face flux over all z-levels."""
        out = jnp.zeros_like(h)
        h_face = jnp.where(
            u[1:-1, 1:-1, 1:-1] >= 0.0,
            h[1:-1, 1:-1, 1:-1],
            h[1:-1, 1:-1, 2:],
        )
        out = out.at[1:-1, 1:-1, 1:-1].set(h_face * u[1:-1, 1:-1, 1:-1])
        return out

    def upwind1_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """1st-order upwind north-face flux over all z-levels."""
        out = jnp.zeros_like(h)
        h_face = jnp.where(
            v[1:-1, 1:-1, 1:-1] >= 0.0,
            h[1:-1, 1:-1, 1:-1],
            h[1:-1, 2:, 1:-1],
        )
        out = out.at[1:-1, 1:-1, 1:-1].set(h_face * v[1:-1, 1:-1, 1:-1])
        return out

    def weno3_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO east-face flux over all z-levels with boundary fallback.

        Positive flow:  fe[k,j,i+1/2] = WENO3(h[k,j,i-1], h[k,j,i],   h[k,j,i+1]) * u
        Negative flow:  fe[k,j,i+1/2] = WENO3(h[k,j,i+2], h[k,j,i+1], h[k,j,i])   * u
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        out = jnp.zeros_like(h)
        # WENO-3 positive: left-biased, valid for all interior faces
        h_pos = _weno3(h[1:-1, 1:-1, :-2], h[1:-1, 1:-1, 1:-1], h[1:-1, 1:-1, 2:])
        # WENO-3 negative: right-biased, valid for i+2 < Nx
        h_neg_interior = _weno3(
            h[1:-1, 1:-1, 3:], h[1:-1, 1:-1, 2:-1], h[1:-1, 1:-1, 1:-2]
        )
        # 1st-order upwind fallback at east boundary
        h_neg_boundary = h[1:-1, 1:-1, -1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=2)
        h_face = jnp.where(u[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1, 1:-1].set(h_face * u[1:-1, 1:-1, 1:-1])
        return out

    def weno3_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO north-face flux over all z-levels with boundary fallback.

        Positive flow:  fn[k,j+1/2,i] = WENO3(h[k,j-1,i], h[k,j,i],   h[k,j+1,i]) * v
        Negative flow:  fn[k,j+1/2,i] = WENO3(h[k,j+2,i], h[k,j+1,i], h[k,j,i])   * v
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        out = jnp.zeros_like(h)
        # WENO-3 positive: left-biased, valid for all interior faces
        h_pos = _weno3(h[1:-1, :-2, 1:-1], h[1:-1, 1:-1, 1:-1], h[1:-1, 2:, 1:-1])
        # WENO-3 negative: right-biased, valid for j+2 < Ny
        h_neg_interior = _weno3(
            h[1:-1, 3:, 1:-1], h[1:-1, 2:-1, 1:-1], h[1:-1, 1:-2, 1:-1]
        )
        # 1st-order upwind fallback at north boundary
        h_neg_boundary = h[1:-1, -1:, 1:-1]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=1)
        h_face = jnp.where(v[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1, 1:-1].set(h_face * v[1:-1, 1:-1, 1:-1])
        return out

    def wenoz3_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO-Z east-face flux over all z-levels with boundary fallback.

        Positive flow:  fe[k,j,i+1/2] = WENOZ3(h[k,j,i-1], h[k,j,i],   h[k,j,i+1]) * u
        Negative flow:  fe[k,j,i+1/2] = WENOZ3(h[k,j,i+2], h[k,j,i+1], h[k,j,i])   * u
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        out = jnp.zeros_like(h)
        # WENO-Z-3 positive: left-biased, valid for all interior faces
        h_pos = _wenoz3(h[1:-1, 1:-1, :-2], h[1:-1, 1:-1, 1:-1], h[1:-1, 1:-1, 2:])
        # WENO-Z-3 negative: right-biased, valid for i+2 < Nx
        h_neg_interior = _wenoz3(
            h[1:-1, 1:-1, 3:], h[1:-1, 1:-1, 2:-1], h[1:-1, 1:-1, 1:-2]
        )
        # 1st-order upwind fallback at east boundary
        h_neg_boundary = h[1:-1, 1:-1, -1:]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=2)
        h_face = jnp.where(u[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1, 1:-1].set(h_face * u[1:-1, 1:-1, 1:-1])
        return out

    def wenoz3_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO-Z north-face flux over all z-levels with boundary fallback.

        Positive flow:  fn[k,j+1/2,i] = WENOZ3(h[k,j-1,i], h[k,j,i],   h[k,j+1,i]) * v
        Negative flow:  fn[k,j+1/2,i] = WENOZ3(h[k,j+2,i], h[k,j+1,i], h[k,j,i])   * v
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        out = jnp.zeros_like(h)
        # WENO-Z-3 positive: left-biased, valid for all interior faces
        h_pos = _wenoz3(h[1:-1, :-2, 1:-1], h[1:-1, 1:-1, 1:-1], h[1:-1, 2:, 1:-1])
        # WENO-Z-3 negative: right-biased, valid for j+2 < Ny
        h_neg_interior = _wenoz3(
            h[1:-1, 3:, 1:-1], h[1:-1, 2:-1, 1:-1], h[1:-1, 1:-2, 1:-1]
        )
        # 1st-order upwind fallback at north boundary
        h_neg_boundary = h[1:-1, -1:, 1:-1]
        h_neg = jnp.concatenate([h_neg_interior, h_neg_boundary], axis=1)
        h_face = jnp.where(v[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1, 1:-1].set(h_face * v[1:-1, 1:-1, 1:-1])
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
        out = jnp.zeros_like(h)
        amasks = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        m5 = amasks[6]  # (Ny, Nx)
        m3 = amasks[4]
        # --- WENO5 face values ---
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
        h_pos_w5 = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=2)
        h5_neg_int = _weno5(
            h[1:-1, 1:-1, 4:],
            h[1:-1, 1:-1, 3:-1],
            h[1:-1, 1:-1, 2:-2],
            h[1:-1, 1:-1, 1:-3],
            h[1:-1, 1:-1, :-4],
        )
        h3_neg_first = _weno3(
            h[1:-1, 1:-1, 3:4], h[1:-1, 1:-1, 2:3], h[1:-1, 1:-1, 1:2]
        )
        h_neg_w5 = jnp.concatenate(
            [h3_neg_first, h5_neg_int, h[1:-1, 1:-1, -1:]], axis=2
        )
        # --- WENO3 face values ---
        h_pos_w3 = _weno3(
            h[1:-1, 1:-1, :-2], h[1:-1, 1:-1, 1:-1], h[1:-1, 1:-1, 2:]
        )
        h_neg_w3_int = _weno3(
            h[1:-1, 1:-1, 3:], h[1:-1, 1:-1, 2:-1], h[1:-1, 1:-1, 1:-2]
        )
        h_neg_w3 = jnp.concatenate([h_neg_w3_int, h[1:-1, 1:-1, -1:]], axis=2)
        # --- 1st-order upwind face values ---
        h_pos_u1 = h[1:-1, 1:-1, 1:-1]
        h_neg_u1 = h[1:-1, 1:-1, 2:]
        # --- Mask-aware selection (broadcast 2D mask over z) ---
        m5_pos = m5[None, 1:-1, 1:-1]  # (1, Ny-2, Nx-2)
        m3_pos = m3[None, 1:-1, 1:-1]
        h_pos = jnp.where(m5_pos, h_pos_w5, jnp.where(m3_pos, h_pos_w3, h_pos_u1))
        m5_neg = m5[None, 1:-1, 2:]
        m3_neg = m3[None, 1:-1, 2:]
        h_neg = jnp.where(m5_neg, h_neg_w5, jnp.where(m3_neg, h_neg_w3, h_neg_u1))
        h_face = jnp.where(u[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1, 1:-1].set(h_face * u[1:-1, 1:-1, 1:-1])
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
        out = jnp.zeros_like(h)
        amasks = mask.get_adaptive_masks(direction="y", stencil_sizes=(2, 4, 6))
        m5 = amasks[6]
        m3 = amasks[4]
        # --- WENO5 face values ---
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
        h_pos_w5 = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=1)
        h5_neg_int = _weno5(
            h[1:-1, 4:, 1:-1],
            h[1:-1, 3:-1, 1:-1],
            h[1:-1, 2:-2, 1:-1],
            h[1:-1, 1:-3, 1:-1],
            h[1:-1, :-4, 1:-1],
        )
        h3_neg_first = _weno3(
            h[1:-1, 3:4, 1:-1], h[1:-1, 2:3, 1:-1], h[1:-1, 1:2, 1:-1]
        )
        h_neg_w5 = jnp.concatenate(
            [h3_neg_first, h5_neg_int, h[1:-1, -1:, 1:-1]], axis=1
        )
        # --- WENO3 face values ---
        h_pos_w3 = _weno3(
            h[1:-1, :-2, 1:-1], h[1:-1, 1:-1, 1:-1], h[1:-1, 2:, 1:-1]
        )
        h_neg_w3_int = _weno3(
            h[1:-1, 3:, 1:-1], h[1:-1, 2:-1, 1:-1], h[1:-1, 1:-2, 1:-1]
        )
        h_neg_w3 = jnp.concatenate([h_neg_w3_int, h[1:-1, -1:, 1:-1]], axis=1)
        # --- 1st-order upwind face values ---
        h_pos_u1 = h[1:-1, 1:-1, 1:-1]
        h_neg_u1 = h[1:-1, 2:, 1:-1]
        # --- Mask-aware selection (broadcast 2D mask over z) ---
        m5_pos = m5[None, 1:-1, 1:-1]
        m3_pos = m3[None, 1:-1, 1:-1]
        h_pos = jnp.where(m5_pos, h_pos_w5, jnp.where(m3_pos, h_pos_w3, h_pos_u1))
        m5_neg = m5[None, 2:, 1:-1]
        m3_neg = m3[None, 2:, 1:-1]
        h_neg = jnp.where(m5_neg, h_neg_w5, jnp.where(m3_neg, h_neg_w3, h_neg_u1))
        h_face = jnp.where(v[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1, 1:-1].set(h_face * v[1:-1, 1:-1, 1:-1])
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
        out = jnp.zeros_like(h)
        amasks = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        m5 = amasks[6]
        m3 = amasks[4]
        # --- WENO-Z-5 face values ---
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
        h_pos_w5 = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=2)
        h5_neg_int = _wenoz5(
            h[1:-1, 1:-1, 4:],
            h[1:-1, 1:-1, 3:-1],
            h[1:-1, 1:-1, 2:-2],
            h[1:-1, 1:-1, 1:-3],
            h[1:-1, 1:-1, :-4],
        )
        h3_neg_first = _wenoz3(
            h[1:-1, 1:-1, 3:4], h[1:-1, 1:-1, 2:3], h[1:-1, 1:-1, 1:2]
        )
        h_neg_w5 = jnp.concatenate(
            [h3_neg_first, h5_neg_int, h[1:-1, 1:-1, -1:]], axis=2
        )
        # --- WENO-Z-3 face values ---
        h_pos_w3 = _wenoz3(
            h[1:-1, 1:-1, :-2], h[1:-1, 1:-1, 1:-1], h[1:-1, 1:-1, 2:]
        )
        h_neg_w3_int = _wenoz3(
            h[1:-1, 1:-1, 3:], h[1:-1, 1:-1, 2:-1], h[1:-1, 1:-1, 1:-2]
        )
        h_neg_w3 = jnp.concatenate([h_neg_w3_int, h[1:-1, 1:-1, -1:]], axis=2)
        # --- 1st-order upwind face values ---
        h_pos_u1 = h[1:-1, 1:-1, 1:-1]
        h_neg_u1 = h[1:-1, 1:-1, 2:]
        # --- Mask-aware selection (broadcast 2D mask over z) ---
        m5_pos = m5[None, 1:-1, 1:-1]
        m3_pos = m3[None, 1:-1, 1:-1]
        h_pos = jnp.where(m5_pos, h_pos_w5, jnp.where(m3_pos, h_pos_w3, h_pos_u1))
        m5_neg = m5[None, 1:-1, 2:]
        m3_neg = m3[None, 1:-1, 2:]
        h_neg = jnp.where(m5_neg, h_neg_w5, jnp.where(m3_neg, h_neg_w3, h_neg_u1))
        h_face = jnp.where(u[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1, 1:-1].set(h_face * u[1:-1, 1:-1, 1:-1])
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
        out = jnp.zeros_like(h)
        amasks = mask.get_adaptive_masks(direction="y", stencil_sizes=(2, 4, 6))
        m5 = amasks[6]
        m3 = amasks[4]
        # --- WENO-Z-5 face values ---
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
        h_pos_w5 = jnp.concatenate([h3_pos_first, h5_pos_int, h3_pos_last], axis=1)
        h5_neg_int = _wenoz5(
            h[1:-1, 4:, 1:-1],
            h[1:-1, 3:-1, 1:-1],
            h[1:-1, 2:-2, 1:-1],
            h[1:-1, 1:-3, 1:-1],
            h[1:-1, :-4, 1:-1],
        )
        h3_neg_first = _wenoz3(
            h[1:-1, 3:4, 1:-1], h[1:-1, 2:3, 1:-1], h[1:-1, 1:2, 1:-1]
        )
        h_neg_w5 = jnp.concatenate(
            [h3_neg_first, h5_neg_int, h[1:-1, -1:, 1:-1]], axis=1
        )
        # --- WENO-Z-3 face values ---
        h_pos_w3 = _wenoz3(
            h[1:-1, :-2, 1:-1], h[1:-1, 1:-1, 1:-1], h[1:-1, 2:, 1:-1]
        )
        h_neg_w3_int = _wenoz3(
            h[1:-1, 3:, 1:-1], h[1:-1, 2:-1, 1:-1], h[1:-1, 1:-2, 1:-1]
        )
        h_neg_w3 = jnp.concatenate([h_neg_w3_int, h[1:-1, -1:, 1:-1]], axis=1)
        # --- 1st-order upwind face values ---
        h_pos_u1 = h[1:-1, 1:-1, 1:-1]
        h_neg_u1 = h[1:-1, 2:, 1:-1]
        # --- Mask-aware selection (broadcast 2D mask over z) ---
        m5_pos = m5[None, 1:-1, 1:-1]
        m3_pos = m3[None, 1:-1, 1:-1]
        h_pos = jnp.where(m5_pos, h_pos_w5, jnp.where(m3_pos, h_pos_w3, h_pos_u1))
        m5_neg = m5[None, 2:, 1:-1]
        m3_neg = m3[None, 2:, 1:-1]
        h_neg = jnp.where(m5_neg, h_neg_w5, jnp.where(m3_neg, h_neg_w3, h_neg_u1))
        h_face = jnp.where(v[1:-1, 1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1, 1:-1].set(h_face * v[1:-1, 1:-1, 1:-1])
        return out
