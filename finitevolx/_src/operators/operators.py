from jaxtyping import Array
import jax.numpy as jnp
import jax
import finitediffx as fdx


def difference(u: Array, axis: int = 0, step_size: float = 1.0, derivative: int=1) -> Array:

    msg = "Derivative must be 1 or 2"
    assert derivative in [1, 2], msg
    du = fdx.difference(
        u, step_size=step_size, axis=axis, accuracy=1, derivative=derivative, method="backward"
    )

    if derivative == 1:
        du = jax.lax.slice_in_dim(du, axis=axis, start_index=1, limit_index=None)
    else:
        du = jax.lax.slice_in_dim(du, axis=axis, start_index=2, limit_index=None)

    return du

def laplacian(u: Array, step_size: float | tuple[float, ...] | Array  = 1) -> Array:

    # calculate laplacian
    lap_u = fdx.laplacian(array=u, accuracy=1, step_size=step_size)

    # remove external dimensions
    lap_u = lap_u[..., 1:-1, 1:-1]
    return lap_u