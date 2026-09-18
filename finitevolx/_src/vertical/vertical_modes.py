from __future__ import annotations

"""Vertical coupling matrix and layer/mode transforms for multi-layer ocean models.

The **A matrix** encodes the coupling between layers through buoyancy (reduced
gravities) and layer thicknesses.  Its eigendecomposition yields:

- Rossby deformation radii for each baroclinic mode.
- The layer-to-mode transform ``Cl2m`` (projects layer-space fields onto modes).
- The mode-to-layer transform ``Cm2l`` (reconstructs layer-space fields from modes).

These building blocks decouple the multi-layer Helmholtz PV-inversion problem
into independent 2-D modal solves and are required for any multi-layer QG or
layered shallow-water model.

Physical background
-------------------
For an n-layer model the A matrix is tridiagonal with entries::

    A[i, i]   =  1/(H[i]*g'[i])  +  1/(H[i]*g'[i+1])    (interior rows)
    A[i, i+1] = -1/(H[i]*g'[i+1])
    A[i, i-1] = -1/(H[i]*g'[i])

where ``H[i]`` is the resting thickness of layer i and ``g'[i]`` is the
reduced gravity at the interface above layer i.  The boundary rows are:

    A[0, 0]  = 1/(H[0]*g'[0]) + 1/(H[0]*g'[1])   (top row, if nl > 1)
    A[-1,-1] = 1/(H[-1]*g'[-1])                    (bottom row)

For a single layer::

    A[0, 0] = 1 / (H[0] * g'[0])

The Rossby deformation radius for mode k is then::

    Rd_k = 1 / (|f0| * sqrt(λ_k))

where λ_k is the k-th eigenvalue of A.

References
----------
- louity/MQGeometry: https://github.com/louity/MQGeometry/blob/main/qgm.py
- louity/qgsw-pytorch: https://github.com/louity/qgsw-pytorch/blob/main/src/qg.py
"""

import jax.numpy as jnp
from jaxtyping import Array, Float


def build_coupling_matrix(
    H: Float[Array, "nl"],
    g_prime: Float[Array, "nl"],
) -> Float[Array, "nl nl"]:
    """Build the tridiagonal vertical coupling (stratification) matrix A.

    A encodes the buoyancy coupling between layers via layer thicknesses and
    reduced gravities.  For positive ``H`` and ``g_prime``, A is tridiagonal
    and diagonally similar to a symmetric positive-definite matrix; it
    becomes exactly symmetric when all layer thicknesses are equal.

    Parameters
    ----------
    H : Float[Array, "nl"]
        Layer resting thicknesses [m].  All values must be positive.
    g_prime : Float[Array, "nl"]
        Reduced gravities [m s⁻²].  ``g_prime[i]`` is the reduced gravity at
        the interface *above* layer i (i.e., between layers i-1 and i).
        The bottom entry ``g_prime[nl-1]`` is the rigid-lid reduced gravity
        and equals ``g_prime[nl-2]`` for a standard ocean configuration.
        All values must be positive.

    Returns
    -------
    Float[Array, "nl nl"]
        Tridiagonal coupling matrix A of shape ``(nl, nl)``.

    Examples
    --------
    Single-layer case:

    >>> import jax.numpy as jnp
    >>> H = jnp.array([1000.0])
    >>> g_prime = jnp.array([9.81])
    >>> A = build_coupling_matrix(H, g_prime)
    >>> A.shape
    (1, 1)

    Two-layer case with equal thicknesses:

    >>> H = jnp.array([500.0, 500.0])
    >>> g_prime = jnp.array([0.02, 0.02])
    >>> A = build_coupling_matrix(H, g_prime)
    >>> A.shape
    (2, 2)
    """
    H = jnp.asarray(H, dtype=float)
    g_prime = jnp.asarray(g_prime, dtype=float)
    nl = H.shape[0]

    # Build A using JAX primitives so the function is JIT-compatible.
    A = jnp.zeros((nl, nl), dtype=H.dtype)

    if nl == 1:
        # Single-layer: A[0,0] = 1 / (H[0] * g'[0])
        A = A.at[0, 0].set(1.0 / (H[0] * g_prime[0]))
    else:
        # Top row (layer 0)
        A = A.at[0, 0].set(1.0 / (H[0] * g_prime[0]) + 1.0 / (H[0] * g_prime[1]))
        A = A.at[0, 1].set(-1.0 / (H[0] * g_prime[1]))

        # Interior rows
        for i in range(1, nl - 1):
            A = A.at[i, i - 1].set(-1.0 / (H[i] * g_prime[i]))
            A = A.at[i, i].set((1.0 / H[i]) * (1.0 / g_prime[i + 1] + 1.0 / g_prime[i]))
            A = A.at[i, i + 1].set(-1.0 / (H[i] * g_prime[i + 1]))

        # Bottom row (layer nl-1)
        A = A.at[-1, -1].set(1.0 / (H[nl - 1] * g_prime[nl - 1]))
        A = A.at[-1, -2].set(-1.0 / (H[nl - 1] * g_prime[nl - 1]))

    return A


def decompose_vertical_modes(
    A: Float[Array, "nl nl"],
    f0: float,
) -> tuple[Float[Array, "nl"], Float[Array, "nl nl"], Float[Array, "nl nl"]]:
    """Eigendecompose the coupling matrix A to get Rossby radii and transforms.

    A is **not** symmetric unless all layer thicknesses are equal: it is
    ``diag(1/H) @ B`` for a symmetric B, so ``A[i, i+1] / A[i+1, i] =
    H[i+1] / H[i]``.  Handing it straight to a symmetric eigensolver would
    silently decompose a different matrix — the one built from a single
    triangle — whose eigenvectors do not diagonalize A, leaving the modes
    coupled and the deformation radii wrong.

    So A is diagonalized through its symmetric similarity
    ``S = diag(d) @ A @ diag(1/d)``, where ``d[i] ∝ sqrt(H[i])`` is recovered
    from A's own off-diagonals.  S shares A's real, non-negative eigenvalues,
    and A's eigenvectors are ``r = v / d`` for S's orthonormal ``v``.  This
    keeps ``jnp.linalg.eigh`` — stable, JIT-compatible, and sorted real
    output — rather than reaching for a general eigensolver, and reduces to
    the plain symmetric case (``d`` constant) for equal thicknesses::

        Cl2m = vᵀ diag(d)
        Cm2l = diag(1/d) v
        Cl2m @ Cm2l = vᵀ v = I

    The mode basis is scaled by ``d[0] = 1`` rather than to unit column norm.
    A vertical mode is only defined up to a scale, the pair inverts and
    diagonalizes the same either way, and the choice cancels in a
    layer → mode → layer round trip.

    Parameters
    ----------
    A : Float[Array, "nl nl"]
        Vertical coupling matrix (from :func:`build_coupling_matrix`).
        Assumed tridiagonal with same-sign off-diagonal pairs, which
        :func:`build_coupling_matrix` gives for positive ``H`` and
        ``g_prime``.
    f0 : float
        Reference Coriolis parameter [s⁻¹].

    Returns
    -------
    rossby_radii : Float[Array, "nl"]
        Rossby deformation radii [m] for each vertical mode.  A rigid lid
        makes the gravest eigenvalue zero and its radius infinite; with a
        free surface (a finite ``g_prime[0]``) that mode is instead the
        external one, at a large but finite ``sqrt(g H_total) / f0``.
    Cl2m : Float[Array, "nl nl"]
        Layer-to-mode transform matrix.  ``mode = Cl2m @ layer``.
    Cm2l : Float[Array, "nl nl"]
        Mode-to-layer transform matrix.  ``layer = Cm2l @ mode``.

    Examples
    --------
    Two-layer decomposition:

    >>> import jax.numpy as jnp
    >>> H = jnp.array([500.0, 500.0])
    >>> g_prime = jnp.array([0.02, 0.02])
    >>> A = build_coupling_matrix(H, g_prime)
    >>> radii, Cl2m, Cm2l = decompose_vertical_modes(A, f0=1e-4)
    >>> radii.shape
    (2,)
    >>> Cl2m.shape
    (2, 2)
    """
    A = jnp.asarray(A, dtype=float)
    f0_arr = jnp.asarray(f0, dtype=float)

    # Symmetrizing scale. S = diag(d) A diag(1/d) is symmetric when
    # d[i]**2 A[i, i+1] == d[i+1]**2 A[i+1, i], i.e. when successive ratios
    # are sqrt(A[i, i+1] / A[i+1, i]) — which is sqrt(H[i+1] / H[i]) for a
    # coupling matrix, so d ∝ sqrt(H) without needing H itself. Both
    # off-diagonals are negative, so the ratio is positive.  For a single
    # layer the diagonals are empty and d is [1.0].
    ratio = jnp.diagonal(A, 1) / jnp.diagonal(A, -1)
    d = jnp.concatenate([jnp.ones((1,), dtype=A.dtype), jnp.cumprod(jnp.sqrt(ratio))])

    S = (d[:, None] * A) / d[None, :]
    S = 0.5 * (S + S.T)  # drop the residual asymmetry left by round-off
    lambd, v = jnp.linalg.eigh(S)  # ascending, real; v orthonormal

    # Map S's orthonormal eigenvectors back to A's: r = v / d. The inverse is
    # exact and needs no solve, since Cm2l = diag(1/d) v inverts to vᵀ diag(d).
    Cm2l = v / d[:, None]
    Cl2m = v.T * d[None, :]

    # Rossby deformation radii: Rd = 1 / (|f0| * sqrt(lambda))
    # Only positive eigenvalues yield a finite radius.  Eigenvalues that are
    # zero (barotropic mode, rigid lid) or negative (numerical noise in a
    # near-singular A) are mapped to inf; A is positive-semi-definite so
    # negative eigenvalues indicate floating-point rounding, not physics.
    # Use jnp.where with a safe fallback to avoid NaN under JIT.
    positive = lambd > 0
    safe_lambd = jnp.where(positive, lambd, 1.0)
    finite_r = 1.0 / (jnp.abs(f0_arr) * jnp.sqrt(safe_lambd))
    rossby_radii = jnp.where(positive, finite_r, jnp.inf)

    return rossby_radii, Cl2m, Cm2l


def layer_to_mode(
    field: Float[Array, "nl *rest"],
    Cl2m: Float[Array, "nl nl"],
) -> Float[Array, "nl *rest"]:
    """Transform a field from layer space to mode space.

    Applies the layer-to-mode transform matrix along the leading (layer) axis.

    Parameters
    ----------
    field : Float[Array, "nl *rest"]
        Field in layer space.  The first dimension is the layer index; any
        trailing dimensions (e.g. ``Ny``, ``Nx``) are preserved.
    Cl2m : Float[Array, "nl nl"]
        Layer-to-mode transform matrix (from :func:`decompose_vertical_modes`).

    Returns
    -------
    Float[Array, "nl *rest"]
        Field in mode space, same shape as input.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> field = jnp.ones((3,))  # 3-layer 0-D
    >>> Cl2m = jnp.eye(3)
    >>> layer_to_mode(field, Cl2m)
    Array([1., 1., 1.], dtype=float32)
    """
    # field: [nl, *rest]  →  flatten rest, matmul, reshape back
    nl = field.shape[0]
    rest = field.shape[1:]
    flat = field.reshape(nl, -1)  # [nl, prod(rest)]
    out_flat = Cl2m @ flat  # [nl, prod(rest)]
    return out_flat.reshape((nl, *rest))


def mode_to_layer(
    field: Float[Array, "nl *rest"],
    Cm2l: Float[Array, "nl nl"],
) -> Float[Array, "nl *rest"]:
    """Transform a field from mode space to layer space.

    Applies the mode-to-layer transform matrix along the leading (mode) axis.

    Parameters
    ----------
    field : Float[Array, "nl *rest"]
        Field in mode space.  The first dimension is the mode index; any
        trailing dimensions are preserved.
    Cm2l : Float[Array, "nl nl"]
        Mode-to-layer transform matrix (from :func:`decompose_vertical_modes`).

    Returns
    -------
    Float[Array, "nl *rest"]
        Field in layer space, same shape as input.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> field = jnp.ones((3,))  # 3-mode 0-D
    >>> Cm2l = jnp.eye(3)
    >>> mode_to_layer(field, Cm2l)
    Array([1., 1., 1.], dtype=float32)
    """
    nl = field.shape[0]
    rest = field.shape[1:]
    flat = field.reshape(nl, -1)  # [nl, prod(rest)]
    out_flat = Cm2l @ flat  # [nl, prod(rest)]
    return out_flat.reshape((nl, *rest))
