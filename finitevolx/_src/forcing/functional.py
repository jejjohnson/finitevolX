"""Raw functional forcing primitives (Layer 0).

Pure, stateless functions that compute the core math of surface wind
stress, bottom drag, and Rayleigh damping.  They mirror the
:mod:`~finitevolx._src.operators.stencils` pattern: no grid object, no
interpolation, no ghost ring, and no masking.  The caller supplies every
field **already at the correct stagger** and is responsible for applying
:func:`~finitevolx.interior` and any land/ocean mask afterwards.

Each momentum primitive acts on a single velocity component, so it is
called once for the U-face and once for the V-face.

Because masking happens afterwards, the inputs must already be finite
everywhere: a zero thickness on dry cells gives ``0 / 0 = NaN`` in
:func:`wind_stress_tendency` / :func:`quadratic_drag_tendency`, and
``NaN * 0`` stays ``NaN``.  Select a safe denominator first (e.g.
``jnp.where(mask, dz, 1.0)``) and zero dry points with ``jnp.where`` rather
than by multiplication.

All functions are compatible with ``jax.jit``, ``jax.vmap``, and
``jax.grad``, and broadcast scalar coefficients against array fields.

Example
-------
>>> import jax.numpy as jnp
>>> from finitevolx import interior, wind_stress_tendency
>>> u = jnp.zeros((6, 6))
>>> tau_x_on_u = 0.1 * jnp.ones((6, 6))  # already interpolated to U-points
>>> du_raw = wind_stress_tendency(tau_x_on_u[1:-1, 1:-1], rho0=1025.0, dz_top=50.0)
>>> du_wind = interior(du_raw, u)  # pad back to full grid shape [Ny, Nx]

References
----------
.. [1] Veros ocean model, ``veros/core/momentum.py`` (``tend_windstress``)
       and ``veros/core/friction.py`` (``linear_bottom_friction``,
       ``quadratic_bottom_friction``, ``rayleigh_friction``).
"""

from __future__ import annotations

from jaxtyping import Array, Float


def wind_stress_tendency(
    tau_on_face: Float[Array, "..."],
    rho0: float | Float[Array, "..."],
    dz_top: float | Float[Array, "..."],
) -> Float[Array, "..."]:
    """Momentum tendency from a surface wind stress.

    Converts a kinematic stress into an acceleration of the top layer:

        du_wind = tau / (rho0 * dz_top)

    Parameters
    ----------
    tau_on_face : Float[Array, "..."]
        Wind-stress component **already interpolated** to the target face
        points (U-points for ``tau_x``, V-points for ``tau_y``) [N/m^2].
    rho0 : float or Float[Array, "..."]
        Reference density [kg/m^3].
    dz_top : float or Float[Array, "..."]
        Top-layer thickness at the same face points [m].  Scalar or a
        field broadcastable to ``tau_on_face``.  Must be non-zero
        everywhere — replace dry-cell zeros before calling (e.g.
        ``jnp.where(mask, dz_top, 1.0)``).

    Returns
    -------
    Float[Array, "..."]
        Wind-stress tendency [m/s^2], same shape as ``tau_on_face``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import wind_stress_tendency
    >>> du = wind_stress_tendency(jnp.array([1025.0]), rho0=1025.0, dz_top=2.0)
    >>> float(du[0])
    0.5
    """
    # du_wind = tau / (rho0 * dz_top)
    return tau_on_face / (rho0 * dz_top)


def linear_drag_tendency(
    vel: Float[Array, "..."],
    r: float | Float[Array, "..."],
) -> Float[Array, "..."]:
    """Linear (Rayleigh-type) drag tendency on one velocity component.

        dvel = -r * vel

    Parameters
    ----------
    vel : Float[Array, "..."]
        Velocity component at its native face points (U or V).
    r : float or Float[Array, "..."]
        Drag coefficient [1/s].  Scalar or a field at the same face points.

    Returns
    -------
    Float[Array, "..."]
        Drag tendency [m/s^2], same shape as ``vel``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import linear_drag_tendency
    >>> linear_drag_tendency(jnp.array([2.0, -4.0]), r=0.5).tolist()
    [-1.0, 2.0]
    """
    # dvel = -r * vel
    return -r * vel


def quadratic_drag_tendency(
    vel: Float[Array, "..."],
    speed: Float[Array, "..."],
    cd: float | Float[Array, "..."],
    h_bot: float | Float[Array, "..."],
) -> Float[Array, "..."]:
    """Quadratic bottom-drag tendency on one velocity component.

        dvel = -Cd * |u| * vel / h_bot

    Parameters
    ----------
    vel : Float[Array, "..."]
        Velocity component at its native face points (U or V).
    speed : Float[Array, "..."]
        Flow speed ``|u|`` at the same face points [m/s].
    cd : float or Float[Array, "..."]
        Dimensionless drag coefficient (typically ``~1e-3``).
    h_bot : float or Float[Array, "..."]
        Bottom-layer thickness at the same face points [m].  Must be
        non-zero everywhere — replace dry-cell zeros before calling (e.g.
        ``jnp.where(mask, h_bot, 1.0)``).

    Returns
    -------
    Float[Array, "..."]
        Drag tendency [m/s^2], same shape as ``vel``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import quadratic_drag_tendency
    >>> vel = jnp.array([2.0])
    >>> float(quadratic_drag_tendency(vel, jnp.abs(vel), cd=0.5, h_bot=4.0)[0])
    -0.5
    """
    # dvel = -Cd * |u| * vel / h_bot
    return -cd * speed * vel / h_bot


def rayleigh_tendency(
    q: Float[Array, "..."],
    r: float | Float[Array, "..."],
    q_ref: float | Float[Array, "..."] | None = None,
) -> Float[Array, "..."]:
    """Rayleigh damping tendency toward a reference state.

        dq = -r * (q - q_ref)

    Parameters
    ----------
    q : Float[Array, "..."]
        Field value at its native grid points.
    r : float or Float[Array, "..."]
        Damping rate [1/s].  Scalar or a field at the same points.
    q_ref : float, Float[Array, "..."] or None, optional
        Reference state broadcastable to ``q``.  ``None`` (default) means
        pure linear damping toward zero.

    Returns
    -------
    Float[Array, "..."]
        Damping tendency, same shape as ``q``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import rayleigh_tendency
    >>> q = jnp.array([3.0])
    >>> float(rayleigh_tendency(q, r=0.5)[0])
    -1.5
    >>> float(rayleigh_tendency(q, r=0.5, q_ref=1.0)[0])
    -1.0
    """
    if q_ref is None:
        # dq = -r * q
        return -r * q
    # dq = -r * (q - q_ref)
    return -r * (q - q_ref)
