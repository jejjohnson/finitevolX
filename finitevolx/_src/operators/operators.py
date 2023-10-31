import finitediffx as fdx
import jax
from jaxtyping import Array


def difference(u: Array, axis: int = 0, step_size: float = 1.0, derivative: int = 1) -> Array:
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
    msg = "Laplacian must be 2D or 3D"
    assert u.ndim in [2, 3], msg
    # calculate laplacian
    lap_u = fdx.laplacian(array=u, accuracy=1, step_size=step_size)

    # remove external dimensions
    lap_u = lap_u[1:-1, 1:-1]

    if u.ndim == 3:
        lap_u = lap_u[..., 1:-1]

    return lap_u
