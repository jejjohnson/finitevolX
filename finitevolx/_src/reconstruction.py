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


# ---------------------------------------------------------------------------
# Private WENO stencil helpers (module-level, no class overhead)
# ---------------------------------------------------------------------------


def _weno3(qm: Array, q0: Array, qp: Array) -> Array:
    """3-point WENO left-biased stencil.

    qi[i+1/2] ~ WENO(q[i-1], q[i], q[i+1])

    Jiang & Shu, J. Comput. Phys. 126, 202-228 (1996).
    """
    eps = 1e-8
    # sub-stencil candidates
    qi1 = -0.5 * qm + 1.5 * q0
    qi2 = 0.5 * (q0 + qp)
    # smoothness indicators
    b1 = (q0 - qm) ** 2
    b2 = (qp - q0) ** 2
    # ideal weights
    g1, g2 = 1.0 / 3.0, 2.0 / 3.0
    w1 = g1 / (b1 + eps) ** 2
    w2 = g2 / (b2 + eps) ** 2
    return (w1 * qi1 + w2 * qi2) / (w1 + w2)


def _wenoz3(qm: Array, q0: Array, qp: Array) -> Array:
    """3-point WENO-Z left-biased stencil.

    qi[i+1/2] ~ WENO-Z(q[i-1], q[i], q[i+1])

    Borges et al., J. Comput. Phys. 227, (2008).
    """
    eps = 1e-14
    # sub-stencil candidates
    qi1 = -0.5 * qm + 1.5 * q0
    qi2 = 0.5 * (q0 + qp)
    # smoothness indicators
    b1 = (q0 - qm) ** 2
    b2 = (qp - q0) ** 2
    tau = jnp.abs(b2 - b1)
    # ideal weights
    g1, g2 = 1.0 / 3.0, 2.0 / 3.0
    w1 = g1 * (1.0 + tau / (b1 + eps))
    w2 = g2 * (1.0 + tau / (b2 + eps))
    return (w1 * qi1 + w2 * qi2) / (w1 + w2)


def _weno5(qmm: Array, qm: Array, q0: Array, qp: Array, qpp: Array) -> Array:
    """5-point WENO left-biased stencil.

    qi[i+1/2] ~ WENO(q[i-2], q[i-1], q[i], q[i+1], q[i+2])

    Jiang & Shu, J. Comput. Phys. 126, 202-228 (1996).
    """
    eps = 1e-8
    # sub-stencil candidates
    qi1 = 1.0 / 3.0 * qmm - 7.0 / 6.0 * qm + 11.0 / 6.0 * q0
    qi2 = -1.0 / 6.0 * qm + 5.0 / 6.0 * q0 + 1.0 / 3.0 * qp
    qi3 = 1.0 / 3.0 * q0 + 5.0 / 6.0 * qp - 1.0 / 6.0 * qpp
    # smoothness indicators
    k1, k2 = 13.0 / 12.0, 0.25
    b1 = k1 * (qmm - 2.0 * qm + q0) ** 2 + k2 * (qmm - 4.0 * qm + 3.0 * q0) ** 2
    b2 = k1 * (qm - 2.0 * q0 + qp) ** 2 + k2 * (qm - qp) ** 2
    b3 = k1 * (q0 - 2.0 * qp + qpp) ** 2 + k2 * (3.0 * q0 - 4.0 * qp + qpp) ** 2
    # ideal weights
    g1, g2, g3 = 0.1, 0.6, 0.3
    w1 = g1 / (b1 + eps) ** 2
    w2 = g2 / (b2 + eps) ** 2
    w3 = g3 / (b3 + eps) ** 2
    return (w1 * qi1 + w2 * qi2 + w3 * qi3) / (w1 + w2 + w3)


def _wenoz5(qmm: Array, qm: Array, q0: Array, qp: Array, qpp: Array) -> Array:
    """5-point WENO-Z left-biased stencil.

    qi[i+1/2] ~ WENO-Z(q[i-2], q[i-1], q[i], q[i+1], q[i+2])

    Borges et al., J. Comput. Phys. 227, (2008).
    """
    eps = 1e-16
    # sub-stencil candidates
    qi1 = 1.0 / 3.0 * qmm - 7.0 / 6.0 * qm + 11.0 / 6.0 * q0
    qi2 = -1.0 / 6.0 * qm + 5.0 / 6.0 * q0 + 1.0 / 3.0 * qp
    qi3 = 1.0 / 3.0 * q0 + 5.0 / 6.0 * qp - 1.0 / 6.0 * qpp
    # smoothness indicators
    k1, k2 = 13.0 / 12.0, 0.25
    b1 = k1 * (qmm - 2.0 * qm + q0) ** 2 + k2 * (qmm - 4.0 * qm + 3.0 * q0) ** 2
    b2 = k1 * (qm - 2.0 * q0 + qp) ** 2 + k2 * (qm - qp) ** 2
    b3 = k1 * (q0 - 2.0 * qp + qpp) ** 2 + k2 * (3.0 * q0 - 4.0 * qp + qpp) ** 2
    tau5 = jnp.abs(b1 - b3)
    # ideal weights
    g1, g2, g3 = 0.1, 0.6, 0.3
    w1 = g1 * (1.0 + tau5 / (b1 + eps))
    w2 = g2 * (1.0 + tau5 / (b2 + eps))
    w3 = g3 * (1.0 + tau5 / (b3 + eps))
    return (w1 * qi1 + w2 * qi2 + w3 * qi3) / (w1 + w2 + w3)


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
        """5-point WENO east-face flux with boundary fallback.

        Positive flow:  h_face[i+1/2] = WENO5(h[i-2], h[i-1], h[i], h[i+1], h[i+2])
        Negative flow:  h_face[i+1/2] = WENO5(h[i+3], h[i+2], h[i+1], h[i], h[i-1])
        Falls back to WENO3 on first interior face and 1st-order on east boundary.
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
        """5-point WENO-Z east-face flux with boundary fallback.

        Positive flow:  h_face[i+1/2] = WENOZ5(h[i-2], h[i-1], h[i], h[i+1], h[i+2])
        Negative flow:  h_face[i+1/2] = WENOZ5(h[i+3], h[i+2], h[i+1], h[i], h[i-1])
        Falls back to WENO-Z-3 on first interior face and 1st-order on east boundary.
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
        """5-point WENO east-face flux with boundary fallback.

        Positive flow:  fe[j,i+1/2] = WENO5(h[j,i-2..i+2]) * u
        Negative flow:  fe[j,i+1/2] = WENO5(h[j,i+3..i-1]) * u
        Falls back to WENO3 on first interior column and 1st-order on east boundary.
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
        """5-point WENO north-face flux with boundary fallback.

        Positive flow:  fn[j+1/2,i] = WENO5(h[j-2..j+2,i]) * v
        Negative flow:  fn[j+1/2,i] = WENO5(h[j+3..j-1,i]) * v
        Falls back to WENO3 on first interior row and 1st-order on north boundary.
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
        """5-point WENO-Z east-face flux with boundary fallback.

        Positive flow:  fe[j,i+1/2] = WENOZ5(h[j,i-2..i+2]) * u
        Negative flow:  fe[j,i+1/2] = WENOZ5(h[j,i+3..i-1]) * u
        Falls back to WENO-Z-3 on first interior column and 1st-order on east boundary.
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
        """5-point WENO-Z north-face flux with boundary fallback.

        Positive flow:  fn[j+1/2,i] = WENOZ5(h[j-2..j+2,i]) * v
        Negative flow:  fn[j+1/2,i] = WENOZ5(h[j+3..j-1,i]) * v
        Falls back to WENO-Z-3 on first interior row and 1st-order on north boundary.
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
        h_pos = _weno3(
            h[1:-1, 1:-1, :-2], h[1:-1, 1:-1, 1:-1], h[1:-1, 1:-1, 2:]
        )
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
        h_pos = _weno3(
            h[1:-1, :-2, 1:-1], h[1:-1, 1:-1, 1:-1], h[1:-1, 2:, 1:-1]
        )
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
        h_pos = _wenoz3(
            h[1:-1, 1:-1, :-2], h[1:-1, 1:-1, 1:-1], h[1:-1, 1:-1, 2:]
        )
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
        h_pos = _wenoz3(
            h[1:-1, :-2, 1:-1], h[1:-1, 1:-1, 1:-1], h[1:-1, 2:, 1:-1]
        )
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
