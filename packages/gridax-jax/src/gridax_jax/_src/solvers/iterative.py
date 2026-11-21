"""Iterative solvers for 2-D elliptic PDEs.

Masked Laplacian
----------------
:func:`masked_laplacian` applies the discrete Helmholtz operator
``(∇² − λ)`` on a masked (irregular) domain, enforcing homogeneous
Dirichlet conditions at the mask boundary and at the array edge.
"""

from __future__ import annotations

from gridax import Mask2D
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# Masked Laplacian operator
# ---------------------------------------------------------------------------


def masked_laplacian(
    psi: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"] | Mask2D,
    dx: float,
    dy: float,
    lambda_: float | Array = 0.0,
) -> Float[Array, "Ny Nx"]:
    """Apply the masked discrete Helmholtz operator (∇² − λ)·ψ.

    Enforces homogeneous Dirichlet conditions at the mask boundary by zeroing
    *psi* outside the mask before applying the 5-point stencil.  The output
    is also zeroed outside the mask.

    Neighbours beyond the array edge are zero (dry), so the array edge is a
    homogeneous Dirichlet boundary one node outside the array, wherever the
    mask is wet there; nothing wraps around.  On an all-ones mask this is
    the dense Dirichlet 5-point operator that the DST-I solvers invert, so
    ``make_spectral_preconditioner(..., bc="dst")`` matches it exactly on a
    rectangle; the FFT preconditioner (periodic) is an approximation.

    Parameters
    ----------
    psi : Float[Array, "Ny Nx"]
        Field to which the operator is applied.
    mask : Float[Array, "Ny Nx"] or Mask2D
        Binary mask: 1 inside the physical domain, 0 outside (land/exterior).
        When an :class:`Mask2D` is passed, the ``psi`` staggering
        mask is used (all four surrounding h-cells must be wet).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter.  Default: 0.0 (pure Laplacian).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Result of (∇² − λ)·(ψ·mask), zeroed outside the mask.
    """
    if isinstance(mask, Mask2D):
        mask_arr = mask.xy_corner_strict.astype(psi.dtype)
    else:
        mask_arr = mask
    psi_m = psi * mask_arr  # enforce zero outside domain  [Ny, Nx]
    # Zero ring outside the array (Dirichlet, not periodic); pad once and
    # bind each neighbour view once, so the stencil lowers to one fused pass.
    p = jnp.pad(psi_m, 1)  # [Ny+2, Nx+2]
    east = p[1:-1, 2:]  # ψ[j, i+1]  [Ny, Nx]
    west = p[1:-1, :-2]  # ψ[j, i-1]  [Ny, Nx]
    north = p[2:, 1:-1]  # ψ[j+1, i]  [Ny, Nx]
    south = p[:-2, 1:-1]  # ψ[j-1, i]  [Ny, Nx]
    # ∇²ψ[j, i] = (ψ[j, i+1] + ψ[j, i-1] - 2ψ[j, i]) / dx²
    #           + (ψ[j+1, i] + ψ[j-1, i] - 2ψ[j, i]) / dy²
    lap = (east + west - 2.0 * psi_m) / dx**2 + (north + south - 2.0 * psi_m) / dy**2
    return (lap - lambda_ * psi_m) * mask_arr  # [Ny, Nx]


def masked_neumann_laplacian(
    p: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Array = 0.0,
) -> Float[Array, "Ny Nx"]:
    """Masked Helmholtz operator (∇² − λ)·p with zero flux through every coast face.

    The Neumann counterpart of :func:`masked_laplacian`, for T-point fields
    such as pressure: a face carries flux only when both of its cells are
    wet, so ∂p/∂n = 0 on every wet–dry face and on the array edge.  On the
    T mask ``[1:-1, 1:-1]`` of a ``from_interior`` rectangle it is exactly the
    DCT-II (Neumann-on-the-faces) operator.  Symmetric and negative
    semidefinite; at λ = 0 its null space is the constants on each connected
    wet component.  The output is zeroed outside the mask.

    Examples
    --------
    A constant has no flux through any face, so the operator annihilates it:

    >>> import jax.numpy as jnp
    >>> mask = jnp.zeros((6, 6)).at[1:-1, 1:-1].set(1.0)
    >>> out = masked_neumann_laplacian(jnp.ones((6, 6)), mask, 1.0, 1.0)
    >>> float(jnp.max(jnp.abs(out)))
    0.0
    """
    return _masked_t_laplacian(p, mask, dx, dy, lambda_, face_dirichlet=False)


def masked_face_dirichlet_laplacian(
    p: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Array = 0.0,
) -> Float[Array, "Ny Nx"]:
    """Masked Helmholtz operator (∇² − λ)·p with p = 0 on every coast *face*.

    For T-point fields: the wall sits on the wet–dry face, half a cell from
    the wet centre, so the face flux is ``(0 − p) / (h/2)``.  This is where
    the T-point spectral Dirichlet solve (DST-II) puts the wall, and it is
    exactly that operator on the T mask of a ``from_interior`` rectangle.
    (:func:`masked_laplacian` instead holds p = 0 at the dry cell *centre*,
    a full cell away: right for ψ at X points, first order for T points.)
    Symmetric and negative definite.  The output is zeroed outside the mask.

    Examples
    --------
    A single wet cell has four coast faces, each with flux ``(0 - p) / (h/2)``:

    >>> import jax.numpy as jnp
    >>> mask = jnp.zeros((3, 3)).at[1, 1].set(1.0)
    >>> out = masked_face_dirichlet_laplacian(jnp.ones((3, 3)), mask, 1.0, 1.0)
    >>> float(out[1, 1])
    -8.0
    """
    return _masked_t_laplacian(p, mask, dx, dy, lambda_, face_dirichlet=True)


def _masked_t_laplacian(
    p: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Array,
    face_dirichlet: bool,
) -> Float[Array, "Ny Nx"]:
    """Five-point (∇² − λ)·p on a masked T field with the wall on the faces.

    Open faces (both cells wet) carry ``p_nb − p``; a coast face (wet–dry,
    or the array edge) carries 0 (zero flux) or, with *face_dirichlet*,
    ``−2 p`` (p = 0 on the face, half a cell away).
    """
    c = 2.0 if face_dirichlet else 0.0
    p_m = p * mask  # [Ny, Nx]
    # Pad once with a dry ring and bind each neighbour view once (one fused pass).
    pp = jnp.pad(p_m, 1)  # [Ny+2, Nx+2]
    mp = jnp.pad(mask, 1)  # [Ny+2, Nx+2]
    # Face weights: mu_e[j, i] = m[j, i] m[j, i+1] (the face at i+1/2 is open), etc.
    mu_e = mask * mp[1:-1, 2:]
    mu_w = mask * mp[1:-1, :-2]
    mu_n = mask * mp[2:, 1:-1]
    mu_s = mask * mp[:-2, 1:-1]
    # ∇²p[j, i] = (mu_e (p[j, i+1] - p) - mu_w (p - p[j, i-1])
    #              - c (2 - mu_e - mu_w) p) / dx²            (x faces)
    #           + (mu_n (p[j+1, i] - p) - mu_s (p - p[j-1, i])
    #              - c (2 - mu_n - mu_s) p) / dy²            (y faces)
    # where (1 - mu) = 1 on a coast face of a wet cell.
    lap_x = mu_e * (pp[1:-1, 2:] - p_m) - mu_w * (p_m - pp[1:-1, :-2])
    lap_y = mu_n * (pp[2:, 1:-1] - p_m) - mu_s * (p_m - pp[:-2, 1:-1])
    if face_dirichlet:
        lap_x = lap_x - c * (2.0 - mu_e - mu_w) * p_m
        lap_y = lap_y - c * (2.0 - mu_n - mu_s) * p_m
    lap = lap_x / dx**2 + lap_y / dy**2
    return (lap - lambda_ * p_m) * mask  # [Ny, Nx]


def wet_components(mask: np.ndarray) -> tuple[Array, Array]:
    """Label the 4-connected wet components of a concrete mask (construction time).

    Returns ``(labels, counts)`` for :func:`project_zero_mean`: ``labels``
    is 0 on dry cells and 1..n on the wet components, ``counts[k]`` the wet
    cells of component ``k`` (``counts[0]`` set to 1, never used).

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.array([[1, 1, 0, 1]])
    >>> labels, counts = wet_components(mask)
    >>> labels.tolist(), counts.tolist()
    ([[1, 1, 0, 2]], [1.0, 2.0, 1.0])
    """
    labels, n_comp = ndimage.label(np.asarray(mask) > 0.5)  # 4-connectivity
    counts = np.bincount(labels.ravel(), minlength=n_comp + 1).astype(np.float64)
    counts[0] = 1.0  # dry label: never used as a divisor of wet values
    return jnp.asarray(labels), jnp.asarray(counts)


def project_zero_mean(
    u: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"],
    labels: Int[Array, "Ny Nx"],
    counts: Float[Array, " n"],
) -> Float[Array, "Ny Nx"]:
    """Remove the wet mean of *u* on each connected wet component.

    The Neumann operator at ``lambda = 0`` annihilates a constant on every
    connected wet component, so ``A u = rhs`` is solvable only for a
    zero-mean ``rhs`` and fixes ``u`` up to those constants.  This
    orthogonal projection (symmetric, idempotent) imposes the zero-mean
    gauge::

        u[j, i] <- u[j, i] - mean(u over the component of (j, i))

    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        Field to project.
    mask : Float[Array, "Ny Nx"]
        Wet mask.
    labels : Int[Array, "Ny Nx"]
        Component label of each cell (0 = dry, 1..n wet components), from
        :func:`wet_components`.
    counts : Float[Array, " n"]
        Wet-cell count per label (``counts[0]`` is unused, set to 1).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Projected field, zero on dry cells.

    Examples
    --------
    Each of the two wet components loses its own mean:

    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> mask = np.array([[1.0, 1.0, 0.0, 1.0]])
    >>> labels, counts = wet_components(mask)
    >>> u = jnp.array([[1.0, 3.0, 5.0, 7.0]])
    >>> project_zero_mean(u, jnp.asarray(mask), labels, counts).tolist()
    [[-1.0, 1.0, 0.0, 0.0]]
    """
    mask = mask.astype(u.dtype)
    u_m = u * mask
    # per-component sum, [n_components + 1]
    sums = jax.ops.segment_sum(u_m.ravel(), labels.ravel(), counts.shape[0])
    mean = (sums / counts.astype(u.dtype))[labels]  # [Ny, Nx]
    return (u_m - mean) * mask
