"""
Standalone divergence operator on 2-D Arakawa C-grids.

Computes ∇·(u, v) at T-points from staggered face velocities.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from finitevolx._src.grid.cartesian import CartesianGrid2D
from finitevolx._src.operators.difference import Difference2D, _divergence_2d


def divergence_2d(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Compute ∇·(u, v) at T-points on the 2-D Arakawa C-grid.

    delta[j, i] = (u[j, i+1/2] - u[j, i-1/2]) / dx
                + (v[j+1/2, i] - v[j-1/2, i]) / dy

    Only interior cells ``[1:-1, 1:-1]`` are written; the ghost ring is
    left as zero.  The caller is responsible for setting ghost-cell
    boundary conditions before calling this function.

    This is a standalone functional form that shares the same underlying
    implementation as :meth:`Difference2D.divergence`.

    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        x-velocity at U-points (east faces).
    v : Float[Array, "Ny Nx"]
        y-velocity at V-points (north faces).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Divergence at T-points, same shape as the inputs.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> u = jnp.zeros((10, 10))
    >>> v = jnp.zeros((10, 10))
    >>> div = divergence_2d(u, v, dx=0.1, dy=0.1)
    >>> div.shape
    (10, 10)
    """
    return _divergence_2d(u, v, dx, dy)


class Divergence2D(eqx.Module):
    """Divergence operator on a 2-D Arakawa C-grid.

    Computes ∇·(u, v) at T-points from staggered face velocities via
    backward finite differences.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid2D, Divergence2D
    >>> grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> div_op = Divergence2D(grid=grid)
    >>> u = jnp.zeros((grid.Ny, grid.Nx))
    >>> v = jnp.zeros((grid.Ny, grid.Nx))
    >>> delta = div_op(u, v)  # standard divergence
    >>> delta_bc = div_op.noflux(u, v)  # no-flux BC variant
    """

    grid: CartesianGrid2D
    diff: Difference2D

    def __init__(self, grid: CartesianGrid2D) -> None:
        self.grid = grid
        self.diff = Difference2D(grid=grid)

    def __call__(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Divergence of (u, v) at T-points.

        delta[j, i] = (u[j, i+1/2] - u[j, i-1/2]) / dx
                    + (v[j+1/2, i] - v[j-1/2, i]) / dy

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Divergence at T-points.
        """
        return self.diff.divergence(u, v)

    def noflux(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Divergence with closed-basin no-flux boundary conditions.

        Zeros all four normal-flow boundary faces before computing the
        backward-difference divergence:

        - ``u_bc[:, 0]   = 0``  west wall U-face   (ghost, read by backward diff at i=1)
        - ``u_bc[:, -2]  = 0``  east wall U-face   (last interior U-face, read at i=Nx-2)
        - ``v_bc[0, :]   = 0``  south wall V-face  (ghost, read by backward diff at j=1)
        - ``v_bc[-2, :]  = 0``  north wall V-face  (last interior V-face, read at j=Ny-2)

        This enforces zero normal velocity on all four sides of a closed basin,
        consistent with the no-flux reference implementations
        (louity/qgsw-pytorch ``div_nofluxbc``).

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Divergence at T-points with closed-basin no-flux BCs applied.
        """
        u_bc = u.at[:, 0].set(0.0).at[:, -2].set(0.0)  # zero west & east wall U-faces
        v_bc = v.at[0, :].set(0.0).at[-2, :].set(0.0)  # zero south & north wall V-faces
        return self.diff.divergence(u_bc, v_bc)
