"""Raw finite-difference and averaging stencils for Arakawa C-grids.

One rule names and shapes every function here. ``<op>_<axis>_<side>_<d>d``
reads the trailing ``d`` axes of a ``[*batch, *spatial]`` array (any leading
axes are batch axes: time, ensemble member, layer), returns the interior of
all ``d`` spatial axes (each one cell shorter at both ends), and shifts by
one cell along ``<axis>``:

- ``<op>``: ``diff`` (``h[+] - h[-]``) or ``avg`` (``(h[+] + h[-]) / 2``);
- ``<axis>``: ``x`` (last axis), ``y`` (second to last) or ``z`` (third);
- ``<side>``: ``fwd`` (the destination is half a cell ahead of the source,
  e.g. T → U) or ``bwd`` (half a cell behind, e.g. U → T);
- ``<d>``: the spatial rank, 1, 2 or 3.

The four-point corner averages ``avg_x<side>_y<side>_<d>d`` shift in both x
and y. No function applies a metric (no division by ``dx`` or
``R cos φ dλ``) or pads the result: the caller scales it and pads it back
with :func:`interior`. They are plain slicing and arithmetic, so they run on
any array backend (NumPy, JAX, CuPy, ...); on JAX they are compatible with
``jax.jit``, ``jax.vmap`` and ``jax.grad``.

Example
-------
>>> from gridax import diff_x_fwd_2d, interior
>>> raw = diff_x_fwd_2d(h)  # pure index arithmetic, [*batch, Ny-2, Nx-2]
>>> scaled = raw / my_custom_metric  # the caller applies the metric
>>> result = interior(scaled, h)  # pad back to [*batch, Ny, Nx]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from jaxtyping import Float
import numpy as np

from gridax._src.backend import optimization_barrier

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

# Names of the previous API (the 2-D stencils had no rank suffix, the corner
# averages mixed two schemes) -> the names of this module. gridax-jax serves
# the old names as deprecated aliases for one release.
RENAMED = {
    "diff_x_fwd": "diff_x_fwd_2d",
    "diff_x_bwd": "diff_x_bwd_2d",
    "diff_y_fwd": "diff_y_fwd_2d",
    "diff_y_bwd": "diff_y_bwd_2d",
    "avg_x_fwd": "avg_x_fwd_2d",
    "avg_x_bwd": "avg_x_bwd_2d",
    "avg_y_fwd": "avg_y_fwd_2d",
    "avg_y_bwd": "avg_y_bwd_2d",
    "avg_xy_fwd": "avg_xfwd_yfwd_2d",
    "avg_xy_bwd": "avg_xbwd_ybwd_2d",
    "avg_xbwd_yfwd": "avg_xbwd_yfwd_2d",
    "avg_xfwd_ybwd": "avg_xfwd_ybwd_2d",
}


def diff_x_fwd_1d(h: Float[Array, "*batch Nx"]) -> Float[Array, "*batch Nx-2"]:
    """Forward difference in x, 1-D: T (or V) → U (or X) points.

    Δx h[i+½] = h[i+1] − h[i]

    Parameters
    ----------
    h : Float[Array, "*batch Nx"]
        Field with its ghost ring on the trailing 1 axis; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nx-2"]
        The raw difference on the interior of all 1 spatial axis; no metric applied.
    """
    return h[..., 2:] - h[..., 1:-1]


def diff_x_bwd_1d(h: Float[Array, "*batch Nx"]) -> Float[Array, "*batch Nx-2"]:
    """Backward difference in x, 1-D: U (or X) → T (or V) points.

    Δx h[i] = h[i+½] − h[i−½]

    Parameters
    ----------
    h : Float[Array, "*batch Nx"]
        Field with its ghost ring on the trailing 1 axis; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nx-2"]
        The raw difference on the interior of all 1 spatial axis; no metric applied.
    """
    return h[..., 1:-1] - h[..., :-2]


def avg_x_fwd_1d(h: Float[Array, "*batch Nx"]) -> Float[Array, "*batch Nx-2"]:
    """Forward average in x, 1-D: T (or V) → U (or X) points.

    h̄[i+½] = ½ (h[i] + h[i+1])

    Parameters
    ----------
    h : Float[Array, "*batch Nx"]
        Field with its ghost ring on the trailing 1 axis; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nx-2"]
        The raw average on the interior of all 1 spatial axis; no metric applied.
    """
    return 0.5 * (h[..., 1:-1] + h[..., 2:])


def avg_x_bwd_1d(h: Float[Array, "*batch Nx"]) -> Float[Array, "*batch Nx-2"]:
    """Backward average in x, 1-D: U (or X) → T (or V) points.

    h̄[i] = ½ (h[i+½] + h[i−½])

    Parameters
    ----------
    h : Float[Array, "*batch Nx"]
        Field with its ghost ring on the trailing 1 axis; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nx-2"]
        The raw average on the interior of all 1 spatial axis; no metric applied.
    """
    return 0.5 * (h[..., 1:-1] + h[..., :-2])


def diff_x_fwd_2d(h: Float[Array, "*batch Ny Nx"]) -> Float[Array, "*batch Ny-2 Nx-2"]:
    """Forward difference in x, 2-D: T (or V) → U (or X) points.

    Δx h[j, i+½] = h[j, i+1] − h[j, i]

    Parameters
    ----------
    h : Float[Array, "*batch Ny Nx"]
        Field with its ghost ring on the trailing 2 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Ny-2 Nx-2"]
        The raw difference on the interior of all 2 spatial axes; no metric applied.
    """
    return h[..., 1:-1, 2:] - h[..., 1:-1, 1:-1]


def diff_x_bwd_2d(h: Float[Array, "*batch Ny Nx"]) -> Float[Array, "*batch Ny-2 Nx-2"]:
    """Backward difference in x, 2-D: U (or X) → T (or V) points.

    Δx h[j, i] = h[j, i+½] − h[j, i−½]

    Parameters
    ----------
    h : Float[Array, "*batch Ny Nx"]
        Field with its ghost ring on the trailing 2 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Ny-2 Nx-2"]
        The raw difference on the interior of all 2 spatial axes; no metric applied.
    """
    return h[..., 1:-1, 1:-1] - h[..., 1:-1, :-2]


def diff_y_fwd_2d(h: Float[Array, "*batch Ny Nx"]) -> Float[Array, "*batch Ny-2 Nx-2"]:
    """Forward difference in y, 2-D: T (or U) → V (or X) points.

    Δy h[j+½, i] = h[j+1, i] − h[j, i]

    Parameters
    ----------
    h : Float[Array, "*batch Ny Nx"]
        Field with its ghost ring on the trailing 2 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Ny-2 Nx-2"]
        The raw difference on the interior of all 2 spatial axes; no metric applied.
    """
    return h[..., 2:, 1:-1] - h[..., 1:-1, 1:-1]


def diff_y_bwd_2d(h: Float[Array, "*batch Ny Nx"]) -> Float[Array, "*batch Ny-2 Nx-2"]:
    """Backward difference in y, 2-D: V (or X) → T (or U) points.

    Δy h[j, i] = h[j+½, i] − h[j−½, i]

    Parameters
    ----------
    h : Float[Array, "*batch Ny Nx"]
        Field with its ghost ring on the trailing 2 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Ny-2 Nx-2"]
        The raw difference on the interior of all 2 spatial axes; no metric applied.
    """
    return h[..., 1:-1, 1:-1] - h[..., :-2, 1:-1]


def avg_x_fwd_2d(h: Float[Array, "*batch Ny Nx"]) -> Float[Array, "*batch Ny-2 Nx-2"]:
    """Forward average in x, 2-D: T (or V) → U (or X) points.

    h̄[j, i+½] = ½ (h[j, i] + h[j, i+1])

    Parameters
    ----------
    h : Float[Array, "*batch Ny Nx"]
        Field with its ghost ring on the trailing 2 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Ny-2 Nx-2"]
        The raw average on the interior of all 2 spatial axes; no metric applied.
    """
    return 0.5 * (h[..., 1:-1, 1:-1] + h[..., 1:-1, 2:])


def avg_x_bwd_2d(h: Float[Array, "*batch Ny Nx"]) -> Float[Array, "*batch Ny-2 Nx-2"]:
    """Backward average in x, 2-D: U (or X) → T (or V) points.

    h̄[j, i] = ½ (h[j, i+½] + h[j, i−½])

    Parameters
    ----------
    h : Float[Array, "*batch Ny Nx"]
        Field with its ghost ring on the trailing 2 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Ny-2 Nx-2"]
        The raw average on the interior of all 2 spatial axes; no metric applied.
    """
    return 0.5 * (h[..., 1:-1, 1:-1] + h[..., 1:-1, :-2])


def avg_y_fwd_2d(h: Float[Array, "*batch Ny Nx"]) -> Float[Array, "*batch Ny-2 Nx-2"]:
    """Forward average in y, 2-D: T (or U) → V (or X) points.

    h̄[j+½, i] = ½ (h[j, i] + h[j+1, i])

    Parameters
    ----------
    h : Float[Array, "*batch Ny Nx"]
        Field with its ghost ring on the trailing 2 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Ny-2 Nx-2"]
        The raw average on the interior of all 2 spatial axes; no metric applied.
    """
    return 0.5 * (h[..., 1:-1, 1:-1] + h[..., 2:, 1:-1])


def avg_y_bwd_2d(h: Float[Array, "*batch Ny Nx"]) -> Float[Array, "*batch Ny-2 Nx-2"]:
    """Backward average in y, 2-D: V (or X) → T (or U) points.

    h̄[j, i] = ½ (h[j+½, i] + h[j−½, i])

    Parameters
    ----------
    h : Float[Array, "*batch Ny Nx"]
        Field with its ghost ring on the trailing 2 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Ny-2 Nx-2"]
        The raw average on the interior of all 2 spatial axes; no metric applied.
    """
    return 0.5 * (h[..., 1:-1, 1:-1] + h[..., :-2, 1:-1])


def diff_x_fwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Forward difference in x, 3-D: T (or V) → U (or X) points.

    Δx h[k, j, i+½] = h[k, j, i+1] − h[k, j, i]

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw difference on the interior of all 3 spatial axes; no metric applied.
    """
    return h[..., 1:-1, 1:-1, 2:] - h[..., 1:-1, 1:-1, 1:-1]


def diff_x_bwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Backward difference in x, 3-D: U (or X) → T (or V) points.

    Δx h[k, j, i] = h[k, j, i+½] − h[k, j, i−½]

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw difference on the interior of all 3 spatial axes; no metric applied.
    """
    return h[..., 1:-1, 1:-1, 1:-1] - h[..., 1:-1, 1:-1, :-2]


def diff_y_fwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Forward difference in y, 3-D: T (or U) → V (or X) points.

    Δy h[k, j+½, i] = h[k, j+1, i] − h[k, j, i]

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw difference on the interior of all 3 spatial axes; no metric applied.
    """
    return h[..., 1:-1, 2:, 1:-1] - h[..., 1:-1, 1:-1, 1:-1]


def diff_y_bwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Backward difference in y, 3-D: V (or X) → T (or U) points.

    Δy h[k, j, i] = h[k, j+½, i] − h[k, j−½, i]

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw difference on the interior of all 3 spatial axes; no metric applied.
    """
    return h[..., 1:-1, 1:-1, 1:-1] - h[..., 1:-1, :-2, 1:-1]


def diff_z_fwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Forward difference in z, 3-D: T → W points.

    Δz h[k+½, j, i] = h[k+1, j, i] − h[k, j, i]

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw difference on the interior of all 3 spatial axes; no metric applied.
    """
    return h[..., 2:, 1:-1, 1:-1] - h[..., 1:-1, 1:-1, 1:-1]


def diff_z_bwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Backward difference in z, 3-D: W → T points.

    Δz h[k, j, i] = h[k+½, j, i] − h[k−½, j, i]

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw difference on the interior of all 3 spatial axes; no metric applied.
    """
    return h[..., 1:-1, 1:-1, 1:-1] - h[..., :-2, 1:-1, 1:-1]


def avg_x_fwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Forward average in x, 3-D: T (or V) → U (or X) points.

    h̄[k, j, i+½] = ½ (h[k, j, i] + h[k, j, i+1])

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw average on the interior of all 3 spatial axes; no metric applied.
    """
    return 0.5 * (h[..., 1:-1, 1:-1, 1:-1] + h[..., 1:-1, 1:-1, 2:])


def avg_x_bwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Backward average in x, 3-D: U (or X) → T (or V) points.

    h̄[k, j, i] = ½ (h[k, j, i+½] + h[k, j, i−½])

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw average on the interior of all 3 spatial axes; no metric applied.
    """
    return 0.5 * (h[..., 1:-1, 1:-1, 1:-1] + h[..., 1:-1, 1:-1, :-2])


def avg_y_fwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Forward average in y, 3-D: T (or U) → V (or X) points.

    h̄[k, j+½, i] = ½ (h[k, j, i] + h[k, j+1, i])

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw average on the interior of all 3 spatial axes; no metric applied.
    """
    return 0.5 * (h[..., 1:-1, 1:-1, 1:-1] + h[..., 1:-1, 2:, 1:-1])


def avg_y_bwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Backward average in y, 3-D: V (or X) → T (or U) points.

    h̄[k, j, i] = ½ (h[k, j+½, i] + h[k, j−½, i])

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw average on the interior of all 3 spatial axes; no metric applied.
    """
    return 0.5 * (h[..., 1:-1, 1:-1, 1:-1] + h[..., 1:-1, :-2, 1:-1])


def avg_z_fwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Forward average in z, 3-D: T → W points.

    h̄[k+½, j, i] = ½ (h[k, j, i] + h[k+1, j, i])

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw average on the interior of all 3 spatial axes; no metric applied.
    """
    return 0.5 * (h[..., 1:-1, 1:-1, 1:-1] + h[..., 2:, 1:-1, 1:-1])


def avg_z_bwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Backward average in z, 3-D: W → T points.

    h̄[k, j, i] = ½ (h[k+½, j, i] + h[k−½, j, i])

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw average on the interior of all 3 spatial axes; no metric applied.
    """
    return 0.5 * (h[..., 1:-1, 1:-1, 1:-1] + h[..., :-2, 1:-1, 1:-1])


def avg_xfwd_yfwd_2d(
    h: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny-2 Nx-2"]:
    """Four-point average, fwd in x and fwd in y, 2-D: T → X points.

    h̄[j+½, i+½] = ¼ (h[j, i] + h[j, i+1] + h[j+1, i] + h[j+1, i+1])

    Parameters
    ----------
    h : Float[Array, "*batch Ny Nx"]
        Field with its ghost ring on the trailing 2 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Ny-2 Nx-2"]
        The raw average on the interior of all 2 spatial axes.
    """
    return 0.25 * (
        h[..., 1:-1, 1:-1] + h[..., 1:-1, 2:] + h[..., 2:, 1:-1] + h[..., 2:, 2:]
    )


def avg_xbwd_ybwd_2d(
    h: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny-2 Nx-2"]:
    """Four-point average, bwd in x and bwd in y, 2-D: X → T points.

    h̄[j, i] = ¼ (h[j+½, i+½] + h[j−½, i+½] + h[j+½, i−½] + h[j−½, i−½])

    Parameters
    ----------
    h : Float[Array, "*batch Ny Nx"]
        Field with its ghost ring on the trailing 2 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Ny-2 Nx-2"]
        The raw average on the interior of all 2 spatial axes.
    """
    return 0.25 * (
        h[..., 1:-1, 1:-1] + h[..., :-2, 1:-1] + h[..., 1:-1, :-2] + h[..., :-2, :-2]
    )


def avg_xbwd_yfwd_2d(
    h: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny-2 Nx-2"]:
    """Four-point average, bwd in x and fwd in y, 2-D: U → V points.

    h̄[j+½, i] = ¼ (h[j, i+½] + h[j+1, i+½] + h[j, i−½] + h[j+1, i−½])

    Parameters
    ----------
    h : Float[Array, "*batch Ny Nx"]
        Field with its ghost ring on the trailing 2 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Ny-2 Nx-2"]
        The raw average on the interior of all 2 spatial axes.
    """
    return 0.25 * (
        h[..., 1:-1, 1:-1] + h[..., 2:, 1:-1] + h[..., 1:-1, :-2] + h[..., 2:, :-2]
    )


def avg_xfwd_ybwd_2d(
    h: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny-2 Nx-2"]:
    """Four-point average, fwd in x and bwd in y, 2-D: V → U points.

    h̄[j, i+½] = ¼ (h[j+½, i] + h[j−½, i] + h[j+½, i+1] + h[j−½, i+1])

    Parameters
    ----------
    h : Float[Array, "*batch Ny Nx"]
        Field with its ghost ring on the trailing 2 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Ny-2 Nx-2"]
        The raw average on the interior of all 2 spatial axes.
    """
    return 0.25 * (
        h[..., 1:-1, 1:-1] + h[..., :-2, 1:-1] + h[..., 1:-1, 2:] + h[..., :-2, 2:]
    )


def avg_xfwd_yfwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Four-point average, fwd in x and fwd in y, 3-D: T → X points on every z-level.

    h̄[j+½, i+½] = ¼ (h[j, i] + h[j, i+1] + h[j+1, i] + h[j+1, i+1])

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw average on the interior of all 3 spatial axes.
    """
    return 0.25 * (
        h[..., 1:-1, 1:-1, 1:-1]
        + h[..., 1:-1, 1:-1, 2:]
        + h[..., 1:-1, 2:, 1:-1]
        + h[..., 1:-1, 2:, 2:]
    )


def avg_xbwd_ybwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Four-point average, bwd in x and bwd in y, 3-D: X → T points on every z-level.

    h̄[j, i] = ¼ (h[j+½, i+½] + h[j−½, i+½] + h[j+½, i−½] + h[j−½, i−½])

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw average on the interior of all 3 spatial axes.
    """
    return 0.25 * (
        h[..., 1:-1, 1:-1, 1:-1]
        + h[..., 1:-1, :-2, 1:-1]
        + h[..., 1:-1, 1:-1, :-2]
        + h[..., 1:-1, :-2, :-2]
    )


def avg_xbwd_yfwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Four-point average, bwd in x and fwd in y, 3-D: U → V points on every z-level.

    h̄[j+½, i] = ¼ (h[j, i+½] + h[j+1, i+½] + h[j, i−½] + h[j+1, i−½])

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw average on the interior of all 3 spatial axes.
    """
    return 0.25 * (
        h[..., 1:-1, 1:-1, 1:-1]
        + h[..., 1:-1, 2:, 1:-1]
        + h[..., 1:-1, 1:-1, :-2]
        + h[..., 1:-1, 2:, :-2]
    )


def avg_xfwd_ybwd_3d(
    h: Float[Array, "*batch Nz Ny Nx"],
) -> Float[Array, "*batch Nz-2 Ny-2 Nx-2"]:
    """Four-point average, fwd in x and bwd in y, 3-D: V → U points on every z-level.

    h̄[j, i+½] = ¼ (h[j+½, i] + h[j−½, i] + h[j+½, i+1] + h[j−½, i+1])

    Parameters
    ----------
    h : Float[Array, "*batch Nz Ny Nx"]
        Field with its ghost ring on the trailing 3 axes; leading axes are batch axes.

    Returns
    -------
    Float[Array, "*batch Nz-2 Ny-2 Nx-2"]
        The raw average on the interior of all 3 spatial axes.
    """
    return 0.25 * (
        h[..., 1:-1, 1:-1, 1:-1]
        + h[..., 1:-1, :-2, 1:-1]
        + h[..., 1:-1, 1:-1, 2:]
        + h[..., 1:-1, :-2, 2:]
    )


# =====================================================================
# Named neighbour views (one static slice per offset)
# =====================================================================


class _Neighbours2D(NamedTuple):
    """The 3×3 neighbourhood of every interior point, one slice per offset.

    Each field is ``h[..., 1+dj : Ny-1+dj, 1+di : Nx-1+di]`` — the value at
    offset ``(dj, di)`` from interior point ``(j, i)`` — with shape
    ``(..., Ny-2, Nx-2)``, exactly the slices the raw stencils above use:

    ========  ================  ========================
    field     offset (dj, di)   slice
    ========  ================  ========================
    ``c``     ( 0,  0)          ``h[..., 1:-1, 1:-1]``
    ``e``     ( 0, +1)          ``h[..., 1:-1, 2:]``
    ``w``     ( 0, -1)          ``h[..., 1:-1, :-2]``
    ``n``     (+1,  0)          ``h[..., 2:, 1:-1]``
    ``s``     (-1,  0)          ``h[..., :-2, 1:-1]``
    ``ne``    (+1, +1)          ``h[..., 2:, 2:]``
    ``nw``    (+1, -1)          ``h[..., 2:, :-2]``
    ``se``    (-1, +1)          ``h[..., :-2, 2:]``
    ``sw``    (-1, -1)          ``h[..., :-2, :-2]``
    ========  ================  ========================

    Why: a stencil that slices the same offset twice (e.g. ``h[1:-1, 1:-1]``
    in both ``diff_x_fwd_2d(h)`` and ``diff_x_bwd_2d(h)``) computes the same forward
    values, but reverse-mode AD transposes *each* slice into its own ``pad``
    of the cotangent.  Slicing each offset once lets JAX sum the cotangents
    of that view first, so the VJP is one fused pass with one ``pad`` per
    offset.  Unused fields cost nothing (XLA drops dead slices).
    """

    c: Array
    e: Array
    w: Array
    n: Array
    s: Array
    ne: Array
    nw: Array
    se: Array
    sw: Array


def _neighbours(h: Float[Array, "... Ny Nx"]) -> _Neighbours2D:
    """Slice the nine 3×3-neighbourhood views of ``h`` once each.

    See :class:`_Neighbours2D` for the offset of each field.  All views have
    shape ``(..., Ny-2, Nx-2)``.
    """
    return _Neighbours2D(
        c=h[..., 1:-1, 1:-1],
        e=h[..., 1:-1, 2:],
        w=h[..., 1:-1, :-2],
        n=h[..., 2:, 1:-1],
        s=h[..., :-2, 1:-1],
        ne=h[..., 2:, 2:],
        nw=h[..., 2:, :-2],
        se=h[..., :-2, 2:],
        sw=h[..., :-2, :-2],
    )


def _materialize(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Keep ``x`` as its own buffer instead of fusing it into its consumers.

    Identity on values and derivatives (``jax.lax.optimization_barrier`` on
    JAX inputs, a plain identity on other backends).
    Use it on an intermediate field that the *next* stencil reads at several
    offsets, e.g. the first Laplacian of a biharmonic operator: without it
    XLA may fuse the producer into each shifted read and recompute it once
    per offset (measured ~1.5x slower forward on CPU for the biharmonic).
    """
    return optimization_barrier(x)
