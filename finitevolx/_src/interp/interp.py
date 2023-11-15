from typing import (
    Callable,
    Optional,
)

import jax.numpy as jnp
from jaxtyping import Array, Float
import kernex as kex
from interpax import Interpolator2D
from jaxinterp2d import CartesianGrid
from finitevolx._src.domain.domain import Domain


def avg_pool(
    u: Array,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: Optional = None,
    mean_fn: str = "arithmetic",
    **kwargs,
) -> Array:
    # get mean function
    mean_fn = get_mean_function(mean_fn=mean_fn)

    # create mean kernel
    @kex.kmap(kernel_size=kernel_size, strides=stride, padding=padding, **kwargs)
    def kernel_fn(x):
        return mean_fn(x)

    # apply kernel function
    return kernel_fn(u)


def x_avg_1D(u: Array, mean_fn: str = "arithmetic") -> Array:
    assert u.ndim == 1
    return avg_pool(u, kernel_size=(2,), stride=(1,), padding="VALID", mean_fn=mean_fn)


def x_avg_2D(u: Array, mean_fn: str = "arithmetic") -> Array:
    assert u.ndim == 2
    return avg_pool(
        u, kernel_size=(2, 1), stride=(1, 1), padding="VALID", mean_fn=mean_fn
    )


def y_avg_2D(u: Array, mean_fn: str = "arithmetic") -> Array:
    assert u.ndim == 2
    return avg_pool(
        u, kernel_size=(1, 2), stride=(1, 1), padding="VALID", mean_fn=mean_fn
    )


def center_avg_2D(u: Array, mean_fn: str = "arithmetic") -> Array:
    assert u.ndim == 2
    return avg_pool(
        u, kernel_size=(2, 2), stride=(1, 1), padding="VALID", mean_fn=mean_fn
    )


def get_mean_function(mean_fn: str = "arithmetic") -> Callable:
    if mean_fn.lower() == "arithmetic":
        fn = lambda x: jnp.mean(x)
        return fn
    elif mean_fn.lower() == "geometric":
        fn = lambda x: jnp.exp(jnp.mean(jnp.log(x)))
        return fn
    elif mean_fn.lower() == "harmonic":
        fn = lambda x: jnp.reciprocal(jnp.mean(jnp.reciprocal(x)))
        return fn
    elif mean_fn.lower() == "quadratic":
        fn = lambda x: jnp.sqrt(jnp.mean(jnp.square(x)))
        return fn
    else:
        msg = "Unrecognized function"
        msg += f"\n{mean_fn}"
        raise ValueError(msg)


def avg_arithmetic(x, y):
    return 0.5 * (x + y)


def avg_harmonic(x, y):
    x_ = jnp.reciprocal(x)
    y_ = jnp.reciprocal(y)
    return jnp.reciprocal(avg_arithmetic(x_, y_))


def avg_geometric(x, y):
    x_ = jnp.log(x)
    y_ = jnp.log(y)
    return jnp.exp(avg_arithmetic(x_, y_))


def avg_quadratic(x, y):
    x_ = jnp.square(x)
    y_ = jnp.square(y)
    return jnp.sqrt(avg_arithmetic(x_, y_))


def domain_interpolation_2D(
        u: Float[Array, "Nx Ny"],
        source_domain: Domain,
        target_domain: Domain,
        method: str = "linear",
        extrap: bool = True
) -> Array:
    """This function will interpolate the values
    from one domain to a target domain

    Args:
        u (Array): the input array
            Size = [Nx, Ny]
        source_domain (Domain): the domain of the input array
        target_domain (Domain): the target domain

    Returns:
        u_ (Array): the input array for the target domain
    """

    assert len(source_domain.Nx) == len(target_domain.Nx) == 2
    assert source_domain.Nx == u.shape

    # initialize interpolator
    interpolator = Interpolator2D(
        x=source_domain.coords_axis[0],
        y=source_domain.coords_axis[1],
        f=u,
        method=method,
        extrap=extrap
    )

    # get coordinates of target grid
    X, Y = target_domain.grid_axis

    # interpolate
    u_on_target = interpolator(xq=X.ravel(), yq=Y.ravel())

    # reshape
    u_on_target = u_on_target.reshape(target_domain.Nx)

    return u_on_target


def cartesian_interpolator_2D(
        u: Float[Array, "Nx Ny"],
        source_domain: Domain,
        target_domain: Domain,
        mode: str = "constant",
        cval: float = 0.0
) -> Array:
    """This function will interpolate the values
    from one domain to a target domain assuming a
    Cartesian grid, i.e., a constant dx,dy,....
    This method is very fast

    Args:
        u (Array): the input array
            Size = [Nx, Ny]
        source_domain (Domain): the domain of the input array
        target_domain (Domain): the target domain

    Returns:
        u_ (Array): the input array for the target domain
    """

    assert len(source_domain.Nx) == len(target_domain.Nx) == 2
    assert source_domain.Nx == u.shape

    # get limits for domain
    xlims = (source_domain.xmin[0], source_domain.xmax[0])
    ylims = (source_domain.xmin[1], source_domain.xmax[1])

    # initialize interpolator
    interpolator = CartesianGrid(
        limits=(xlims, ylims),
        values=u,
        mode=mode, cval=cval
    )

    # get coordinates of target grid
    X, Y = target_domain.grid_axis

    # interpolate
    u_on_target = interpolator(X.ravel(), Y.ravel())

    # reshape
    u_on_target = u_on_target.reshape(target_domain.Nx)

    return u_on_target
