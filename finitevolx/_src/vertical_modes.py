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
import numpy as np


def build_coupling_matrix(
    H: Float[Array, " nl"],
    g_prime: Float[Array, " nl"],
) -> Float[Array, "nl nl"]:
    """Build the tridiagonal vertical coupling (stratification) matrix A.

    A encodes the buoyancy coupling between layers via layer thicknesses and
    reduced gravities.  It is symmetric and positive-semi-definite.

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

    A = np.zeros((nl, nl), dtype=float)

    if nl == 1:
        # Single-layer: A[0,0] = 1 / (H[0] * g'[0])
        A[0, 0] = 1.0 / (float(H[0]) * float(g_prime[0]))
    else:
        # Top row (layer 0)
        A[0, 0] = 1.0 / (float(H[0]) * float(g_prime[0])) + 1.0 / (
            float(H[0]) * float(g_prime[1])
        )
        A[0, 1] = -1.0 / (float(H[0]) * float(g_prime[1]))

        # Interior rows
        for i in range(1, nl - 1):
            A[i, i - 1] = -1.0 / (float(H[i]) * float(g_prime[i]))
            A[i, i] = (
                1.0
                / float(H[i])
                * (1.0 / float(g_prime[i + 1]) + 1.0 / float(g_prime[i]))
            )
            A[i, i + 1] = -1.0 / (float(H[i]) * float(g_prime[i + 1]))

        # Bottom row (layer nl-1)
        A[-1, -1] = 1.0 / (float(H[nl - 1]) * float(g_prime[nl - 1]))
        A[-1, -2] = -1.0 / (float(H[nl - 1]) * float(g_prime[nl - 1]))

    return jnp.array(A)


def decompose_vertical_modes(
    A: Float[Array, "nl nl"],
    f0: float,
) -> tuple[Float[Array, " nl"], Float[Array, "nl nl"], Float[Array, "nl nl"]]:
    """Eigendecompose the coupling matrix A to get Rossby radii and transforms.

    Uses the biorthogonal normalisation from louity/qgsw-pytorch for numerical
    stability::

        Cl2m = diag(1 / diag(Lᵀ R))  ·  Lᵀ
        Cm2l = R

    where R (right eigenvectors) and L (left eigenvectors, i.e. right
    eigenvectors of Aᵀ) are real parts of the complex eigenvector matrices.

    Parameters
    ----------
    A : Float[Array, "nl nl"]
        Vertical coupling matrix (from :func:`build_coupling_matrix`).
    f0 : float
        Reference Coriolis parameter [s⁻¹].

    Returns
    -------
    rossby_radii : Float[Array, "nl"]
        Rossby deformation radii [m] for each vertical mode.
        The barotropic mode (zero eigenvalue) has an infinite radius.
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
    A_np = np.array(A)
    f0 = float(f0)

    # Right eigenvectors: A R = R diag(lambda)
    lambd_r, R = np.linalg.eig(A_np)
    # Left eigenvectors: A^T L = L diag(lambda_l)
    _lambd_l, L = np.linalg.eig(A_np.T)

    # Use real parts (A is real-symmetric in practice)
    lambd = lambd_r.real
    R = R.real
    L = L.real

    # Biorthogonal normalisation: Cl2m = diag(1/diag(L^T R)) @ L^T
    # This ensures Cl2m @ Cm2l == I
    diag_LtR = np.diag(L.T @ R)
    Cl2m_np = np.diag(1.0 / diag_LtR) @ L.T
    Cm2l_np = R

    # Rossby deformation radii: Rd = 1 / (|f0| * sqrt(lambda))
    # Only positive eigenvalues yield a finite radius.  Eigenvalues that are
    # zero (barotropic mode, rigid lid) or negative (numerical noise in a
    # near-singular A) are mapped to inf; A is positive-semi-definite so
    # negative eigenvalues indicate floating-point rounding, not physics.
    with np.errstate(divide="ignore", invalid="ignore"):
        rossby_radii_np = np.where(
            lambd > 0, 1.0 / (np.abs(f0) * np.sqrt(lambd)), np.inf
        )

    return jnp.array(rossby_radii_np), jnp.array(Cl2m_np), jnp.array(Cm2l_np)


def layer_to_mode(
    field: Float[Array, " nl *rest"],
    Cl2m: Float[Array, "nl nl"],
) -> Float[Array, " nl *rest"]:
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
    field: Float[Array, " nl *rest"],
    Cm2l: Float[Array, "nl nl"],
) -> Float[Array, " nl *rest"]:
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
