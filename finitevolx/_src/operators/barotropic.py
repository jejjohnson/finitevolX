"""
Implicit barotropic-mode (external gravity wave) filter for multi-layer
shallow water.

For a stacked multi-layer model the external (barotropic) gravity wave sets a
punishing CFL.  This operator implements the implicit filter of
Dukowicz & Smith (2000), as used in MASSH ``sw.py::filter_barotropic_waves``:
it forms the depth-integrated (barotropic) provisional transport, solves a
single variable-coefficient Helmholtz problem for the fast surface mode, and
returns the momentum-tendency correction that removes the divergent fast part
implicitly — letting the caller take baroclinic-scale time steps.

The hard part — the variable-coefficient Helmholtz solve ``div(h grad w)`` —
is **not** re-implemented here: it is injected as ``helm_solve``, which the
caller builds from the existing multigrid solver
(``build_multigrid_solver(mask, dx, dy, coeff=h)``, which is itself callable).
This module is the thin operator that wires the transport, the solve, and the
gradient correction together.

References
----------
Dukowicz, J. K. & Smith, R. D. (1994/2000). Implicit free-surface method for
the Bryan-Cox-Semtner ocean model. doi:10.1029/2000JC900089.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.operators._ghost import interior
from finitevolx._src.operators.divergence import divergence_2d
from finitevolx._src.operators.stencils import diff_x_fwd, diff_y_fwd


def barotropic_filter(
    u_star: Float[Array, "Nz Ny Nx"],
    v_star: Float[Array, "Nz Ny Nx"],
    h_u: Float[Array, "Nz Ny Nx"],
    h_v: Float[Array, "Nz Ny Nx"],
    *,
    dx: float,
    dy: float,
    g: float,
    tau: float,
    dt: float,
    helm_solve: Callable[[Float[Array, "Ny Nx"]], Float[Array, "Ny Nx"]],
    layer_axis: int = -3,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    r"""Implicit external-gravity-wave (barotropic) mode filter.

    Given the provisional velocities ``u_star`` / ``v_star`` (after the
    explicit RHS) and the per-layer thicknesses on the velocity faces, this:

    1. forms the barotropic (depth-summed) provisional mass transport
       ``U_bt = sum_k h_u u*`` on U-faces and ``V_bt = sum_k h_v v*`` on
       V-faces;
    2. builds the Helmholtz right-hand side
       :math:`\text{rhs} = \frac{1}{g\,\tau\,\Delta t}\, \nabla\!\cdot(U_{bt}, V_{bt})`;
    3. solves the variable-coefficient Helmholtz problem
       :math:`\nabla\!\cdot(h\,\nabla w) = \text{rhs}` via the injected
       ``helm_solve``;
    4. returns the implicit fast-mode momentum correction
       :math:`\text{filt}_u = -g\,\tau\,\partial_x w`,
       :math:`\text{filt}_v = -g\,\tau\,\partial_y w`.

    The returned corrections are single-layer (barotropic) fields the caller
    adds back to every layer's momentum tendency, e.g. ``dt_u += filt_u``.

    ``helm_solve`` is invoked as a plain callable (``helm_solve(rhs)``); pass
    the :class:`~finitevolx.MultigridSolver` object itself (it is callable via
    ``__call__``).

    .. note::
        :func:`~finitevolx.build_multigrid_solver` **freezes** the coefficient
        (interpolated to the staggered ``cx``/``cy`` face fields) into the level
        hierarchy at construction time.  A prebuilt ``helm_solve`` therefore
        solves with a *fixed* coefficient; it is correct only when the
        barotropic coefficient is time-invariant.  If ``h_u``/``h_v`` evolve,
        either accept the fixed-coefficient (frozen-``h``) barotropic
        approximation or rebuild the solver each step — do not reuse a stale
        solver across changing thicknesses.

    Parameters
    ----------
    u_star, v_star : Float[Array, "Nz Ny Nx"]
        Provisional velocities at U-/V-points, stacked over layers.
    h_u, h_v : Float[Array, "Nz Ny Nx"]
        Layer thicknesses interpolated to U-/V-points.
    dx, dy : float
        Grid spacing.
    g : float
        Gravitational acceleration (or reduced gravity for the barotropic
        mode).
    tau : float
        Implicitness / relaxation parameter of the filter.
    dt : float
        Time step.
    helm_solve : Callable[[Array], Array]
        Solver for the variable-coefficient Helmholtz problem
        ``div(h grad w) = rhs`` at T-points, e.g. a
        :class:`~finitevolx.MultigridSolver` built with ``coeff=h`` (the
        solver object is itself callable).
    layer_axis : int, optional
        Axis to sum the layers over.  Default ``-3``.

    Returns
    -------
    tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
        ``(filt_u, filt_v)`` — barotropic momentum-tendency corrections at
        U-/V-points, ghost ring zero.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import barotropic_filter
    >>> u = jnp.zeros((2, 8, 8))
    >>> h = jnp.ones((2, 8, 8))
    >>> # quiescent flow -> zero correction (helm_solve never sees divergence)
    >>> fu, fv = barotropic_filter(
    ...     u,
    ...     u,
    ...     h,
    ...     h,
    ...     dx=1.0,
    ...     dy=1.0,
    ...     g=9.81,
    ...     tau=1.0,
    ...     dt=1.0,
    ...     helm_solve=lambda rhs: rhs,
    ... )
    >>> bool((fu == 0).all())
    True
    """
    # Barotropic (depth-summed) provisional mass transport on the faces.
    u_bt = jnp.sum(h_u * u_star, axis=layer_axis)  # (Ny, Nx) on U-faces
    v_bt = jnp.sum(h_v * v_star, axis=layer_axis)  # (Ny, Nx) on V-faces

    # Divergence of the barotropic transport, scaled into the Helmholtz RHS.
    rhs = divergence_2d(u_bt, v_bt, dx, dy) / (g * tau * dt)

    # Solve the variable-coefficient Helmholtz problem for the surface mode.
    w = helm_solve(rhs)

    # Implicit fast-mode correction: minus the (scaled) gradient of w at faces.
    filt_u = interior(-g * tau * diff_x_fwd(w) / dx, w)
    filt_v = interior(-g * tau * diff_y_fwd(w) / dy, w)
    return filt_u, filt_v
