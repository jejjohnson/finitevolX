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
        """2nd-order upwind reconstruction at east face.

        Positive flow:  h_face[i+1/2] = 3/2*h[i]   - 1/2*h[i-1]
        Negative flow:  h_face[i+1/2] = 3/2*h[i+1] - 1/2*h[i+2]
        Boundary cells (i=1) fall back to 1st-order upwind.
        """
        out = jnp.zeros_like(h)
        # h_face_pos[i+1/2] = 3/2*h[i] - 1/2*h[i-1]
        h_pos = 1.5 * h[1:-1] - 0.5 * h[:-2]
        # h_face_neg[i+1/2] = 3/2*h[i+1] - 1/2*h[i+2]
        h_neg = 1.5 * h[2:] - 0.5 * jnp.concatenate([h[3:], h[-1:]])
        h_face = jnp.where(u[1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1].set(h_face * u[1:-1])
        return out

    def upwind3_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """3rd-order upwind reconstruction at east face.

        Positive flow:  h_face = -1/6*h[i-1] + 5/6*h[i]   + 1/3*h[i+1]
        Negative flow:  h_face =  1/3*h[i]   + 5/6*h[i+1] - 1/6*h[i+2]
        Falls back to upwind1 at boundaries.
        """
        out = jnp.zeros_like(h)
        # 3rd-order positive stencil
        h_pos = (
            -1.0 / 6.0 * h[:-2]  # h[i-1]
            + 5.0 / 6.0 * h[1:-1]  # h[i  ]
            + 1.0 / 3.0 * h[2:]  # h[i+1]
        )
        # 3rd-order negative stencil
        h_neg = (
            1.0 / 3.0 * h[1:-1]  # h[i  ]
            + 5.0 / 6.0 * h[2:]  # h[i+1]
            - 1.0 / 6.0 * jnp.concatenate([h[3:], h[-1:]])  # h[i+2]
        )
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
        """
        out = jnp.zeros_like(h)
        # 2nd-order positive stencil (uses h[j,i-1])
        h_pos = 1.5 * h[1:-1, 1:-1] - 0.5 * h[1:-1, :-2]
        # 2nd-order negative stencil (uses h[j,i+2])
        h_neg = 1.5 * h[1:-1, 2:] - 0.5 * jnp.concatenate(
            [h[1:-1, 3:], h[1:-1, -1:]], axis=1
        )
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
        """
        out = jnp.zeros_like(h)
        # 2nd-order positive stencil (uses h[j-1,i])
        h_pos = 1.5 * h[1:-1, 1:-1] - 0.5 * h[:-2, 1:-1]
        # 2nd-order negative stencil (uses h[j+2,i])
        h_neg = 1.5 * h[2:, 1:-1] - 0.5 * jnp.concatenate(
            [h[3:, 1:-1], h[-1:, 1:-1]], axis=0
        )
        h_face = jnp.where(v[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * v[1:-1, 1:-1])
        return out

    def upwind3_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3rd-order upwind east-face flux.

        Positive: h_face = -1/6*h[j,i-1] + 5/6*h[j,i]   + 1/3*h[j,i+1]
        Negative: h_face =  1/3*h[j,i]   + 5/6*h[j,i+1] - 1/6*h[j,i+2]
        """
        out = jnp.zeros_like(h)
        # 3rd-order positive stencil
        h_pos = (
            -1.0 / 6.0 * h[1:-1, :-2]  # h[j, i-1]
            + 5.0 / 6.0 * h[1:-1, 1:-1]  # h[j, i  ]
            + 1.0 / 3.0 * h[1:-1, 2:]  # h[j, i+1]
        )
        # 3rd-order negative stencil
        h_neg = (
            1.0 / 3.0 * h[1:-1, 1:-1]  # h[j, i  ]
            + 5.0 / 6.0 * h[1:-1, 2:]  # h[j, i+1]
            - 1.0
            / 6.0
            * jnp.concatenate([h[1:-1, 3:], h[1:-1, -1:]], axis=1)  # h[j, i+2]
        )
        h_face = jnp.where(u[1:-1, 1:-1] >= 0.0, h_pos, h_neg)
        out = out.at[1:-1, 1:-1].set(h_face * u[1:-1, 1:-1])
        return out

    def upwind3_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3rd-order upwind north-face flux.

        Positive: h_face = -1/6*h[j-1,i] + 5/6*h[j,i]   + 1/3*h[j+1,i]
        Negative: h_face =  1/3*h[j,i]   + 5/6*h[j+1,i] - 1/6*h[j+2,i]
        """
        out = jnp.zeros_like(h)
        # 3rd-order positive stencil
        h_pos = (
            -1.0 / 6.0 * h[:-2, 1:-1]  # h[j-1, i]
            + 5.0 / 6.0 * h[1:-1, 1:-1]  # h[j,   i]
            + 1.0 / 3.0 * h[2:, 1:-1]  # h[j+1, i]
        )
        # 3rd-order negative stencil
        h_neg = (
            1.0 / 3.0 * h[1:-1, 1:-1]  # h[j,   i]
            + 5.0 / 6.0 * h[2:, 1:-1]  # h[j+1, i]
            - 1.0
            / 6.0
            * jnp.concatenate([h[3:, 1:-1], h[-1:, 1:-1]], axis=0)  # h[j+2, i]
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
