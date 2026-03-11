"""
Standalone divergence operator on 2-D Arakawa C-grids.

Computes ∇·(u, v) at T-points from staggered face velocities.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.difference import Difference2D
from finitevolx._src.grid import ArakawaCGrid2D


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
    out = jnp.zeros_like(u)
    # delta[j, i] = du/dx + dv/dy  (backward differences U→T, V→T)
    # du_dx[j, i] = (u[j, i+1/2] - u[j, i-1/2]) / dx  →  (u[1:-1,1:-1] - u[1:-1,:-2]) / dx
    du_dx = (u[1:-1, 1:-1] - u[1:-1, :-2]) / dx
    # dv_dy[j, i] = (v[j+1/2, i] - v[j-1/2, i]) / dy  →  (v[1:-1,1:-1] - v[:-2,1:-1]) / dy
    dv_dy = (v[1:-1, 1:-1] - v[:-2, 1:-1]) / dy
    return out.at[1:-1, 1:-1].set(du_dx + dv_dy)


class Divergence2D(eqx.Module):
    """Divergence operator on a 2-D Arakawa C-grid.

    Computes ∇·(u, v) at T-points from staggered face velocities via
    backward finite differences.

    Parameters
    ----------
    grid : ArakawaCGrid2D
        The underlying 2-D grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import ArakawaCGrid2D, Divergence2D
    >>> grid = ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> div_op = Divergence2D(grid=grid)
    >>> u = jnp.zeros((grid.Ny, grid.Nx))
    >>> v = jnp.zeros((grid.Ny, grid.Nx))
    >>> delta = div_op(u, v)  # standard divergence
    >>> delta_bc = div_op.noflux(u, v)  # no-flux BC variant
    """

    grid: ArakawaCGrid2D
    diff: Difference2D

    def __init__(self, grid: ArakawaCGrid2D) -> None:
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
        """Divergence with no-flux boundary conditions.

        Zeros the west ghost of u (``u[:, 0]``) and the south ghost of v
        (``v[0, :]``) before computing the backward-difference divergence.
        This enforces zero normal velocity at the west and south walls,
        consistent with a closed-basin no-flux boundary condition.

        delta_noflux[j, i] = (u_bc[j, i+1/2] - u_bc[j, i-1/2]) / dx
                           + (v_bc[j+1/2, i] - v_bc[j-1/2, i]) / dy

        where ``u_bc[:, 0] = 0`` and ``v_bc[0, :] = 0``.

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Divergence at T-points with no-flux BCs applied.
        """
        u_bc = u.at[:, 0].set(0.0)  # zero west ghost U-face
        v_bc = v.at[0, :].set(0.0)  # zero south ghost V-face
        return self.diff.divergence(u_bc, v_bc)
