import jax
from jaxtyping import (
    Array,
    Float,
)

from finitevolx._src.constants import GRAVITY


def difference(
    u: Array, axis: int = 0, step_size: float | Array = 1.0, derivative: int = 1
) -> Array:
    import finitediffx as fdx

    if derivative == 1:
        du = fdx.difference(
            u,
            step_size=step_size,
            axis=axis,
            accuracy=1,
            derivative=derivative,
            method="backward",
        )
        du = jax.lax.slice_in_dim(du, axis=axis, start_index=1, limit_index=None)
    elif derivative == 2:
        du = fdx.difference(
            u,
            step_size=step_size,
            axis=axis,
            accuracy=1,
            derivative=derivative,
            method="central",
        )
        du = jax.lax.slice_in_dim(du, axis=axis, start_index=1, limit_index=-1)
    else:
        msg = "Derivative must be 1 or 2"
        raise ValueError(msg)

    return du


def laplacian(u: Array, step_size: float | tuple[float, ...] | Array = 1) -> Array:
    import finitediffx as fdx

    msg = "Laplacian must be 2D or 3D"
    assert u.ndim in [2, 3], msg
    # calculate laplacian
    lap_u = fdx.laplacian(array=u, accuracy=1, step_size=step_size)

    # remove external dimensions
    lap_u = lap_u[1:-1, 1:-1]

    if u.ndim == 3:
        lap_u = lap_u[..., 1:-1]

    return lap_u


def geostrophic_gradient(
    p: Float[Array, "Nx Ny"],
    dx: float | Array,
    dy: float | Array,
) -> tuple[Float[Array, "Nx Ny-1"], Float[Array, "Nx-1 Ny"]]:
    """Calculates the geostrophic gradient for a staggered grid

    Equation:
        u = -∂yΨ
        v = ∂xΨ

    Args:
        p (Array): the input variable
            Size = [Nx,Ny]
        dx (float | Array): the stepsize for the x-direction
        dy (float | Array): the stepsize for the y-direction

    Returns:
        dp_dy (Array): the geostrophic velocity in the y-direction
            Size = [Nx,Ny-1]
        dp_dx (Array): the geostrophic velocity in the x-direction
            Size = [Nx-1,Ny]

    Note:
        for the geostrophic velocity, we need to multiply the
        derivative in the x-direction by negative 1.
    """

    dp_dy = difference(u=p, axis=1, step_size=dy, derivative=1)
    dp_dx = difference(u=p, axis=0, step_size=dx, derivative=1)
    return -dp_dy, dp_dx


def divergence(u: Array, v: Array, dx: float, dy: float) -> Array:
    """Calculates the divergence for a staggered grid

    Equation:
        ∇⋅u̅  = ∂x(u) + ∂y(v)

    Args:
        u (Array): the input array for the u direction
            Size = [Nx, Ny-1]
        v (Array): the input array for the v direction
            Size = [Nx-1, Ny]
        dx (float | Array): the stepsize for the x-direction
        dy (float | Array): the stepsize for the y-direction

    Returns:
        div (Array): the divergence
            Size = [Nx-1,Ny-1]

    """
    # ∂xu
    dudx = difference(u=u, axis=0, step_size=dx, derivative=1)
    # ∂yv
    dvdx = difference(u=v, axis=1, step_size=dy, derivative=1)

    return dudx + dvdx


def relative_vorticity(
    u: Float[Array, "Nx Ny-1"],
    v: Float[Array, "Nx-1 Ny"],
    dx: float | Array,
    dy: float | Array,
) -> Float[Array, "Nx-1 Ny-1"]:
    """Calculates the relative vorticity by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        ζ = ∂v/∂x - ∂u/∂y

    Args:
        u (Array): the input array for the u direction
            Size = [Nx, Ny-1]
        v (Array): the input array for the v direction
            Size = [Nx-1, Ny]
        dx (float | Array): the stepsize for the x-direction
        dy (float | Array): the stepsize for the y-direction

    Returns:
        zeta (Array): the relative vorticity
            Size = [Nx-1,Ny-1]
    """
    # ∂xv
    dv_dx: Float[Array, "Nx-1 Ny-1"] = difference(
        u=v, axis=0, step_size=dx, derivative=1
    )
    # ∂yu
    du_dy: Float[Array, "Nx-1 Ny-1"] = difference(
        u=u, axis=1, step_size=dy, derivative=1
    )

    return dv_dx - du_dy


def kinetic_energy(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
) -> Float[Array, "Ny Nx"]:
    """Kinetic energy at T-points (cell centers) on an Arakawa C-grid.

    Eq:
        ke[j, i] = 0.5 * (u²_on_T[j, i] + v²_on_T[j, i])

    where u² and v² are averaged from face-points to T-points:
        u²_on_T[j, i] = 0.5 * (u[j, i+1/2]² + u[j, i-1/2]²)
                       = 0.5 * (u[j, i]² + u[j, i-1]²)
        v²_on_T[j, i] = 0.5 * (v[j+1/2, i]² + v[j-1/2, i]²)
                       = 0.5 * (v[j, i]² + v[j-1, i]²)

    Args:
        u (Array): x-velocity at U-points (east faces), shape [Ny, Nx].
        v (Array): y-velocity at V-points (north faces), shape [Ny, Nx].

    Returns:
        ke (Array): kinetic energy at T-points, shape [Ny, Nx].
            Ghost ring is zero; interior is [1:-1, 1:-1].
    """
    u2 = u**2
    v2 = v**2
    out = jnp.zeros_like(u)
    # u²_on_T[j, i] = 0.5 * (u²[j, i] + u²[j, i-1])  (east + west U-faces)
    # v²_on_T[j, i] = 0.5 * (v²[j, i] + v²[j-1, i])  (north + south V-faces)
    u2_on_T = 0.5 * (u2[1:-1, 1:-1] + u2[1:-1, :-2])
    v2_on_T = 0.5 * (v2[1:-1, 1:-1] + v2[:-2, 1:-1])
    out = out.at[1:-1, 1:-1].set(0.5 * (u2_on_T + v2_on_T))
    return out


def absolute_vorticity(
    u: Float[Array, "Nx Ny-1"],
    v: Float[Array, "Nx-1 Ny"],
    dx: float | Array,
    dy: float | Array,
) -> Float[Array, "Nx-1 Ny-1"]:
    """Calculates the relative vorticity by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        ζ = ∂v/∂x + ∂u/∂y

    Args:
        u (Array): the input array for the u direction
            Size = [Nx, Ny-1]
        v (Array): the input array for the v direction
            Size = [Nx-1, Ny]
        dx (float | Array): the stepsize for the x-direction
        dy (float | Array): the stepsize for the y-direction

    Returns:
        zeta (Array): the absolute vorticity
            Size = [Nx-1,Ny-1]
    """
    # ∂xv
    dv_dx: Float[Array, "Nx-1 Ny-1"] = difference(
        u=v, axis=0, step_size=dx, derivative=1
    )
    # ∂yu
    du_dy: Float[Array, "Nx-1 Ny-1"] = difference(
        u=u, axis=1, step_size=dy, derivative=1
    )

    return dv_dx + du_dy


def bernoulli_potential(
    h: Float[Array, "Ny Nx"],
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    gravity: float = GRAVITY,
) -> Float[Array, "Ny Nx"]:
    """Bernoulli potential at T-points on an Arakawa C-grid.

    Eq:
        p[j, i] = ke[j, i] + g * h[j, i]

    where ke is the kinetic energy at T-points.

    Args:
        h (Array): layer thickness at T-points, shape [Ny, Nx].
        u (Array): x-velocity at U-points (east faces), shape [Ny, Nx].
        v (Array): y-velocity at V-points (north faces), shape [Ny, Nx].
        gravity (float): gravitational acceleration. Default = 9.81.

    Returns:
        p (Array): Bernoulli potential at T-points, shape [Ny, Nx].
            Ghost ring is zero; interior is [1:-1, 1:-1].

    Example:
        >>> u, v, h = ...
        >>> p = bernoulli_potential(h=h, u=u, v=v)
    """
    ke = kinetic_energy(u=u, v=v)
    out = jnp.zeros_like(h)
    out = out.at[1:-1, 1:-1].set(ke[1:-1, 1:-1] + gravity * h[1:-1, 1:-1])
    return out
