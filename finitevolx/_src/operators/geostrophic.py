from jaxtyping import Array, Float
from finitevolx._src.operators.operators import difference
from finitevolx._src.constants import  GRAVITY

def geostrophic_gradient(u: Array, dx: float | Array, dy: float | Array) -> tuple[Array, Array]:
    """Calculates the geostrophic gradient for a staggered grid

    Equation:
        u = -∂yΨ
        v = ∂xΨ

    Args:
        u (Array): the input variable
            Size = [Nx,Ny]
        dx (float | Array): the stepsize for the x-direction
        dy (float | Array): the stepsize for the y-direction

    Returns:
        du_dy (Array): the geostrophic velocity in the y-direction
            Size = [Nx,Ny-1]
        du_dx (Array): the geostrophic velocity in the x-direction
            Size = [Nx-1,Ny]

    Note:
        for the geostrophic velocity, we need to multiply the
        derivative in the x-direction by negative 1.
    """

    du_dy = difference(u=u, axis=1, step_size=dy, derivative=1)
    dv_dx = difference(u=u, axis=0, step_size=dx, derivative=1)
    return - du_dy, dv_dx


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


def relative_vorticity(u: Float[Array, "Nx Ny-1"], v: Float[Array, "Nx-1 Ny"], dx: float | Array, dy: float | Array) -> Array:
    """Calculates the relative vorticity by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        ζ = ∂v/∂x - ∂u/∂y

    Args:
        u (Array): the input array for the u direction
            Size = [Nx+1, Ny]
        v (Array): the input array for the v direction
            Size = [Nx, Ny+1]
        dx (float | Array): the stepsize for the x-direction
        dy (float | Array): the stepsize for the y-direction

    Returns:
        zeta (Array): the relative vorticity
            Size = [Nx-1,Ny-1]
    """
    # ∂xv
    dv_dx: Float[Array, "Nx-1 Ny-1"] = difference(u=v, axis=0, step_size=dx, derivative=1)
    # ∂yu
    du_dy: Float[Array, "Nx-1 Ny-1"] = difference(u=u, axis=1, step_size=dy, derivative=1)

    return dv_dx - du_dy


def ssh_to_streamfn(ssh: Array, f0: float = 1e-5, g: float = GRAVITY) -> Array:
    """Calculates the ssh to stream function

    Eq:
        η = (g/f₀) Ψ

    Args:
        ssh (Array): the sea surface height [m]
        f0 (Array|float): the coriolis parameter
        g (float): the acceleration due to gravity

    Returns:
        psi (Array): the stream function
    """
    return (g / f0) * ssh


def streamfn_to_ssh(psi: Array, f0: float = 1e-5, g: float = GRAVITY) -> Array:
    """Calculates the stream function to ssh

    Eq:
        Ψ = (f₀/g) η

    Args:
        psi (Array): the stream function
        f0 (Array|float): the coriolis parameter
        g (float): the acceleration due to gravity

    Returns:
        ssh (Array): the sea surface height [m]
    """
    return (f0 / g) * psi