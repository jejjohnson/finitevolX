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

from jaxtyping import Array, Float


def arakawa_jacobian(
    f: Float[Array, "... Ny Nx"],
    g: Float[Array, "... Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "... Ny-2 Nx-2"]:
    """Arakawa (1966) discretization of J(f, g).

    Computes the Jacobian J(f, g) = ∂f/∂x·∂g/∂y − ∂f/∂y·∂g/∂x using the
    energy- and enstrophy-conserving three-term Arakawa scheme on a collocated
    grid.  The inputs must include a one-point boundary halo on each side so
    that the returned interior array has shape ``(..., Ny-2, Nx-2)``.

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
    Float[Array, "... Ny-2 Nx-2"]
        Jacobian evaluated on the interior grid points.  Boundary points are
        consumed by the stencil and are not included in the output.

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
