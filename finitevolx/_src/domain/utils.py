import typing as tp
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array
import einops
import math


def make_coords(xmin, xmax, nx):
    return jnp.linspace(start=xmin, stop=xmax, num=nx, endpoint=True)


def make_grid_from_coords(coords: tp.Iterable) -> tp.List[Array]:
    if isinstance(coords, tp.Iterable):
        return jnp.meshgrid(*coords, indexing="ij")
    elif isinstance(coords, (jnp.ndarray, np.ndarray)):
        return jnp.meshgrid(coords, indexing="ij")
    else:
        raise ValueError("Unrecognized dtype for inputs")


def make_grid_coords(coords: tp.Iterable) -> Array:
    grid = make_grid_from_coords(coords)

    grid = jnp.stack(grid, axis=0)

    grid = einops.rearrange(grid, "N ... -> (...) N")

    return grid


def create_meshgrid_coordinates(shape):
    meshgrid = jnp.meshgrid(*[jnp.arange(size) for size in shape], indexing="ij")
    # create indices
    indices = jnp.concatenate([jnp.expand_dims(x, axis=-1) for x in meshgrid], axis=-1)

    return indices


def bounds_and_step_to_points(xmin: float, xmax: float, dx: float) -> int:
    return 1 + int(math.floor(((float(xmax) - float(xmin)) / float(dx))))


def bounds_to_length(xmin: float, xmax: float) -> float:
    """Calculates the Lx from the minmax
    Eq:
        Lx = xmax - xmin"""
    return abs(float(xmax) - float(xmin))


def bounds_and_points_to_step(xmin: float, xmax: float, Nx: float) -> float:
    return (float(xmax) - float(xmin)) / (float(Nx) - 1.0)


def length_and_points_to_step(Lx: float, Nx: float) -> float:
    return float(Lx) / (float(Nx) - 1.0)


def length_and_step_to_points(Lx: float, dx: float) -> int:
    return math.floor(1.0 + float(Lx) / float(dx))


def check_stagger(dx: tp.Tuple, stagger: tp.Tuple[str] = None):
    """Creates stagger values based on semantic names.
    Useful for C-Grid operations

    Args:
    -----
        dx (Iterable): the step sizes
        stagger (Iterable): the stagger direction

    Returns:
    --------
        stagger (Iterable): the stagger values (as a fraction
            of dx).
    """
    if stagger is None:
        stagger = (None,) * len(dx)

    msg = "Length of stagger and dx is off"
    msg += f"\ndx: {len(dx)}"
    msg += f"\nstagger: {len(stagger)}"
    assert len(dx) == len(stagger), msg

    stagger_values = list()
    for istagger in stagger:
        if istagger is None:
            stagger_values.append(0.0)
        elif istagger == "right":
            stagger_values.append(0.5)
        elif istagger == "left":
            stagger_values.append(-0.5)
        else:
            raise ValueError("Unrecognized command")

    return stagger_values


def check_tuple_inputs(x) -> tp.Tuple:
    if isinstance(x, tuple):
        return x
    elif isinstance(x, float) or isinstance(x, int):
        return tuple(x)
    elif isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    elif x is None:
        return None
    else:
        raise ValueError(f"Unrecognized type: {x} | {type(x)}")