"""
Arakawa (1966) Jacobian operator for energy- and enstrophy-conserving advection.

The Arakawa discretization of J(f, g) = ∂f/∂x·∂g/∂y − ∂f/∂y·∂g/∂x is the
standard advection operator for quasi-geostrophic models.  Unlike simple
centred-difference advection, the three-term average (J⁺⁺ + J⁺× + J×⁺)/3
conserves energy, enstrophy, and satisfies J(f,f) = 0 and ∫J(f,g)dA = 0
exactly at the discrete level.

Reference
---------
Arakawa, A. (1966). Computational design for long-term numerical integration
of the equations of fluid motion: Two-dimensional incompressible flow.
Part I. *Journal of Computational Physics*, 1(1), 119–143.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid.cartesian import CartesianGrid2D
from finitevolx._src.mask import Mask2D
from finitevolx._src.operators.diagnostic_operators import _sanitize


def arakawa_jacobian(
    f: Float[Array, "... Ny Nx"],
    g: Float[Array, "... Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "... Ny_i Nx_i"]:
    """Arakawa (1966) discretization of J(f, g).

    Computes the Jacobian J(f, g) = ∂f/∂x·∂g/∂y − ∂f/∂y·∂g/∂x using the
    energy- and enstrophy-conserving three-term Arakawa scheme on a collocated
    grid.  The inputs must include a one-point boundary halo on each side so
    that the returned interior array has shape ``(..., Ny-2, Nx-2)``
    (i.e. ``Ny_i = Ny - 2``, ``Nx_i = Nx - 2``).

    Parameters
    ----------
    f : Float[Array, "... Ny Nx"]
        First scalar field (including one halo cell on each side).
    g : Float[Array, "... Ny Nx"]
        Second scalar field (same shape as *f*).
    dx : float
        Grid spacing in the x-direction (last array axis).
    dy : float
        Grid spacing in the y-direction (second-to-last array axis).

    Returns
    -------
    Float[Array, "... Ny_i Nx_i"]
        Jacobian evaluated on the interior grid points, where
        ``Ny_i = Ny - 2`` and ``Nx_i = Nx - 2``.
        Boundary points are consumed by the stencil and are not included
        in the output.

    Notes
    -----
    The Arakawa scheme averages three discrete forms:

    * J⁺⁺ (standard centred form):

      .. code-block:: text

         Jpp[j,i] = ( (f[j,i+1] - f[j,i-1]) * (g[j+1,i] - g[j-1,i])
                    - (f[j+1,i] - f[j-1,i]) * (g[j,i+1] - g[j,i-1]) ) / (4 dx dy)

    * J⁺× (advective form):

      .. code-block:: text

         Jpx[j,i] = ( f[j,i+1] * (g[j+1,i+1] - g[j-1,i+1])
                    - f[j,i-1] * (g[j+1,i-1] - g[j-1,i-1])
                    - f[j+1,i] * (g[j+1,i+1] - g[j+1,i-1])
                    + f[j-1,i] * (g[j-1,i+1] - g[j-1,i-1]) ) / (4 dx dy)

    * J×⁺ (divergence form):

      .. code-block:: text

         Jxp[j,i] = ( g[j+1,i] * (f[j+1,i+1] - f[j+1,i-1])
                    - g[j-1,i] * (f[j-1,i+1] - f[j-1,i-1])
                    - g[j,i+1] * (f[j+1,i+1] - f[j-1,i+1])
                    + g[j,i-1] * (f[j+1,i-1] - f[j-1,i-1]) ) / (4 dx dy)

    Together: ``J = (Jpp + Jpx + Jxp) / 3``

    This triple average conserves energy (∫∫ f·J dA = 0), enstrophy
    (∫∫ g·J dA = 0), satisfies J(f, f) = 0, and ∫∫ J(f, g) dA = 0 at the
    discrete level.

    The function is JAX-compatible and ``jit``-able.  Batch dimensions
    (``...``) are supported via standard broadcasting.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import arakawa_jacobian
    >>> Ny, Nx = 12, 10
    >>> x = jnp.linspace(0, 1, Nx)
    >>> y = jnp.linspace(0, 1, Ny)
    >>> dx, dy = x[1] - x[0], y[1] - y[0]
    >>> X, Y = jnp.meshgrid(x, y)
    >>> J = arakawa_jacobian(X, Y, float(dx), float(dy))
    >>> J.shape
    (10, 8)
    """
    # J++ (standard centred form)
    # Jpp[j,i] = (df/dx * dg/dy - df/dy * dg/dx)
    # where df/dx ~ (f[j, i+1] - f[j, i-1]) / (2dx), etc.
    Jpp = (f[..., 1:-1, 2:] - f[..., 1:-1, :-2]) * (
        g[..., 2:, 1:-1] - g[..., :-2, 1:-1]
    ) - (f[..., 2:, 1:-1] - f[..., :-2, 1:-1]) * (g[..., 1:-1, 2:] - g[..., 1:-1, :-2])

    # J+x (advective form)
    # f evaluated at off-centre x-neighbours, g differenced in y at those neighbours
    Jpx = (
        f[..., 1:-1, 2:] * (g[..., 2:, 2:] - g[..., :-2, 2:])
        - f[..., 1:-1, :-2] * (g[..., 2:, :-2] - g[..., :-2, :-2])
        - f[..., 2:, 1:-1] * (g[..., 2:, 2:] - g[..., 2:, :-2])
        + f[..., :-2, 1:-1] * (g[..., :-2, 2:] - g[..., :-2, :-2])
    )

    # Jx+ (divergence form)
    # g evaluated at off-centre y-neighbours, f differenced in x at those neighbours
    Jxp = (
        g[..., 2:, 1:-1] * (f[..., 2:, 2:] - f[..., 2:, :-2])
        - g[..., :-2, 1:-1] * (f[..., :-2, 2:] - f[..., :-2, :-2])
        - g[..., 1:-1, 2:] * (f[..., 2:, 2:] - f[..., :-2, 2:])
        + g[..., 1:-1, :-2] * (f[..., 2:, :-2] - f[..., :-2, :-2])
    )

    return (Jpp + Jpx + Jxp) / (12.0 * dx * dy)


class ArakawaJacobian2D(eqx.Module):
    """Arakawa (1966) Jacobian J(f, g) at T-points, on the full grid.

    Class form of :func:`arakawa_jacobian`.  Unlike the function, which
    returns only the interior ``(..., Ny-2, Nx-2)``, the class returns the
    full ``(..., Ny, Nx)`` array with a zero ghost ring, like every other
    Layer-3 operator:

    J_full[..., j, i] = J[..., j-1, i-1]   for 1 <= j <= Ny-2, 1 <= i <= Nx-2
    J_full[..., j, i] = 0                  on the ghost ring

    When ``mask`` is set, ``f`` and ``g`` are first zeroed on dry *interior*
    T-cells (the stencil reads the eight neighbours of each T-cell, so a
    coastal cell would otherwise read land values, ``NaN`` or not; a
    streamfunction from a masked elliptic solve is zero there anyway; the
    ghost ring is BC-owned and passed through), and dry output cells
    are zeroed last via ``jnp.where(mask.h, ...)`` (Pattern 1 in
    ``docs/masks.md``), broadcast over any leading batch axes.

    Masking zeroes the Jacobian at dry cells, so the discrete identities
    ``sum(J) = 0`` and ``sum(g * J) = 0`` are **not** guaranteed for sums
    over a masked domain -- they hold for the unmasked operator with
    suitable boundary conditions.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid (supplies ``dx`` and ``dy``).
    mask : Mask2D or None, optional
        Optional land/ocean mask.  ``None`` (default) returns the functional
        form's output padded with a zero ghost ring.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import ArakawaJacobian2D, CartesianGrid2D
    >>> grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> psi = jnp.zeros((grid.Ny, grid.Nx))
    >>> ArakawaJacobian2D(grid=grid)(psi, psi).shape
    (10, 10)
    """

    grid: CartesianGrid2D
    mask: Mask2D | None = None

    def __call__(
        self,
        f: Float[Array, "... Ny Nx"],
        g: Float[Array, "... Ny Nx"],
    ) -> Float[Array, "... Ny Nx"]:
        """Evaluate J(f, g) at T-points.

        Parameters
        ----------
        f : Float[Array, "... Ny Nx"]
            First scalar field at T-points (e.g. the streamfunction).
        g : Float[Array, "... Ny Nx"]
            Second scalar field at T-points (e.g. the potential vorticity).

        Returns
        -------
        Float[Array, "... Ny Nx"]
            J(f, g) at T-points, zero in the ghost ring and, when
            ``self.mask`` is set, at dry T-cells.
        """
        if self.mask is not None:
            # f[..., j, i] = 0 on dry interior T-cells (same for g)
            f = _sanitize(self.mask.h, f)
            g = _sanitize(self.mask.h, g)
        J = arakawa_jacobian(f, g, self.grid.dx, self.grid.dy)
        shape = jnp.broadcast_shapes(f.shape, g.shape)
        # J_full[..., j, i] = J[..., j-1, i-1]  for 1 <= j <= Ny-2, 1 <= i <= Nx-2
        out = jnp.zeros_like(f, dtype=J.dtype, shape=shape)
        out = out.at[..., 1:-1, 1:-1].set(J)
        if self.mask is not None:
            out = jnp.where(self.mask.h, out, 0.0)
        return out
