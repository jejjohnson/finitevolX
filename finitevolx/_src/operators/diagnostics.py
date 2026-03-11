from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)

from finitevolx._src.grid.constants import GRAVITY


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
    dtype = jnp.result_type(u, v, 0.0)
    u_float = jnp.asarray(u, dtype=dtype)
    v_float = jnp.asarray(v, dtype=dtype)
    u2 = u_float**2
    v2 = v_float**2
    out = jnp.zeros_like(u_float)
    # u²_on_T[j, i] = 0.5 * (u²[j, i] + u²[j, i-1])  (east + west U-faces)
    # v²_on_T[j, i] = 0.5 * (v²[j, i] + v²[j-1, i])  (north + south V-faces)
    u2_on_T = 0.5 * (u2[1:-1, 1:-1] + u2[1:-1, :-2])
    v2_on_T = 0.5 * (v2[1:-1, 1:-1] + v2[:-2, 1:-1])
    out = out.at[1:-1, 1:-1].set(0.5 * (u2_on_T + v2_on_T))
    return out


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
    dtype = jnp.result_type(h, u, v, 0.0)
    h_float = jnp.asarray(h, dtype=dtype)
    ke = kinetic_energy(u=u, v=v)
    out = jnp.zeros_like(h_float)
    out = out.at[1:-1, 1:-1].set(ke[1:-1, 1:-1] + gravity * h_float[1:-1, 1:-1])
    return out
