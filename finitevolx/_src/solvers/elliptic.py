"""Capacitance matrix solver and convenience wrappers for 2-D elliptic PDEs.

This module re-exports the spectral, iterative, and preconditioner APIs from
their dedicated sub-modules and adds:

Capacitance matrix method (irregular/masked domains)
-----------------------------------------------------
Extends the fast spectral solver to domains that are subsets of a
rectangle (e.g. ocean basins with land masks) using the classic
Sherman-Morrison correction via boundary Green's functions.

``build_capacitance_solver`` performs a one-time offline precomputation
(N_b rectangular solves, where N_b = number of irregular-boundary points).
The returned ``CapacitanceSolver`` callable is then cheap to evaluate for
any right-hand side.

Convenience wrappers
--------------------
* :func:`streamfunction_from_vorticity` — ∇²ψ − λψ = ζ
* :func:`pressure_from_divergence` — ∇²p = ∇·u
* :func:`pv_inversion` — (∇² − λ)ψ = q  (multi-layer / batched)
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np

from finitevolx._src.grid.cgrid_mask import ArakawaCGridMask

# Re-export from iterative module
from finitevolx._src.solvers.iterative import (  # noqa: F401
    CGInfo,
    masked_laplacian,
    solve_cg,
)

# Re-export from preconditioners module
from finitevolx._src.solvers.preconditioners import (  # noqa: F401
    make_multigrid_preconditioner,
    make_nystrom_preconditioner,
    make_preconditioner,
    make_spectral_preconditioner,
)

# Re-export from spectral module
from finitevolx._src.solvers.spectral import (  # noqa: F401
    _HELMHOLTZ_DISPATCH,
    _spectral_solve,
    dct2_eigenvalues,
    dst1_eigenvalues,
    fft_eigenvalues,
    solve_helmholtz_dct,
    solve_helmholtz_dst,
    solve_helmholtz_fft,
    solve_poisson_dct,
    solve_poisson_dst,
    solve_poisson_fft,
)

# ---------------------------------------------------------------------------
# Capacitance matrix solver (irregular / masked domains)
# ---------------------------------------------------------------------------


class CapacitanceSolver(eqx.Module):
    """Spectral Poisson/Helmholtz solver for masked irregular domains.

    Uses the **capacitance matrix method** (Buzbee, Golub & Nielson 1970) to
    extend a fast rectangular spectral solver to a domain defined by a binary
    mask.

    The algorithm relies on two observations:

    1. The irregular-domain solution ``ψ`` equals the rectangular spectral
       solution ``u`` minus a correction ``Σ_k α_k g_k``, where ``g_k`` are
       Green's functions (response to unit sources at each inner-boundary
       point ``b_k``).

    2. Requiring ``ψ(b_k) = 0`` at every inner-boundary point yields the
       linear system ``C α = u[B]``, where ``C[k,l] = g_l(b_k)`` is the
       **capacitance matrix**.

    Construct with :func:`build_capacitance_solver` (offline, runs *N_b*
    spectral solves where *N_b* = number of inner-boundary points).

    Attributes
    ----------
    _C_inv : Float[Array, "Nb Nb"]
        Pre-inverted capacitance matrix.
    _green_flat : Float[Array, "Nb NyNx"]
        Green's functions (one row per boundary point), stored flat.
    _j_b : Int[Array, "Nb"]
        Row indices of inner-boundary points.
    _i_b : Int[Array, "Nb"]
        Column indices of inner-boundary points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter.
    base_bc : str
        Spectral solver used as the rectangular base (``"fft"``, ``"dst"``,
        or ``"dct"``).
    """

    _C_inv: Float[Array, "Nb Nb"]
    _green_flat: Float[Array, "Nb NyNx"]
    _j_b: Array
    _i_b: Array
    dx: float
    dy: float
    lambda_: float = eqx.field(static=True)
    base_bc: str = eqx.field(static=True)

    def __call__(
        self,
        rhs: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Solve (∇² − λ)ψ = rhs on the masked domain.

        Parameters
        ----------
        rhs : Float[Array, "Ny Nx"]
            Right-hand side, defined on the full rectangular grid.
            Values outside the mask are ignored.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Solution ψ, satisfying ψ = 0 at all inner-boundary points
            and (approximately) (∇² − λ)ψ = rhs at interior points.
        """
        Ny, Nx = rhs.shape
        # Step 1: rectangular spectral solve
        u = _spectral_solve(rhs, self.dx, self.dy, self.lambda_, self.base_bc)
        # Step 2: values of u at inner-boundary points
        u_b = u[self._j_b, self._i_b]  # [Nb]
        # Step 3: correction coefficients  alpha = C^{-1} u_b
        alpha = self._C_inv @ u_b  # [Nb]
        # Step 4: correction field  sum_k alpha_k g_k
        correction = (self._green_flat.T @ alpha).reshape(Ny, Nx)  # [Ny, Nx]
        return u - correction


def build_capacitance_solver(
    mask: np.ndarray | ArakawaCGridMask,
    dx: float,
    dy: float,
    lambda_: float = 0.0,
    base_bc: str = "fft",
) -> CapacitanceSolver:
    """Pre-compute the capacitance matrix and return a ready-to-use solver.

    This is an **offline** function that performs *N_b* rectangular spectral
    solves (``N_b`` = number of inner-boundary points).  The result is a
    :class:`CapacitanceSolver` whose ``__call__`` method is JIT-compilable.

    Algorithm (Buzbee, Golub & Nielson 1970):

    1. Find inner-boundary points ``B`` = mask points adjacent to exterior.
    2. For each ``b_k ∈ B``, solve ``L_rect g_k = e_{b_k}`` (Green's function).
    3. Build ``C[k, l] = g_l(b_k)``  and invert to ``C⁻¹``.

    Parameters
    ----------
    mask : np.ndarray of bool shape (Ny, Nx), or ArakawaCGridMask
        Physical domain mask.  ``True`` = interior (ocean/fluid),
        ``False`` = exterior (land/walls).

        When a plain array is passed, inner-boundary points are computed
        as wet (``True``) cells that are 4-connected to at least one dry
        (``False``) cell.

        When an :class:`ArakawaCGridMask` is passed, the ``psi``
        staggering mask is used and the precomputed
        ``psi_irrbound_xids`` / ``psi_irrbound_yids`` supply the
        inner-boundary indices directly.  These are **wet** points on
        the psi grid that border at least one dry cell — the same
        convention as the plain-array path.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter λ.  Use ``0.0`` for pure Poisson.
    base_bc : {"fft", "dst", "dct"}
        Rectangular spectral solver used as the base.  ``"fft"`` (periodic)
        is a good default; ``"dst"`` (Dirichlet) handles rectangle-boundary
        conditions directly.

    Returns
    -------
    CapacitanceSolver
        A callable equinox Module with all precomputed arrays baked in.

    Notes
    -----
    Memory cost: ``O(N_b × Ny × Nx)`` for the Green's function matrix.
    Time cost (offline): ``O(N_b × Ny × Nx × log(Ny × Nx))``.
    Time cost (online): ``O(N_b² + Ny × Nx × log(Ny × Nx))``.

    Raises
    ------
    ValueError
        If the mask has no inner-boundary points (e.g. all-ones mask).
    """
    # Extract mask array and boundary indices from ArakawaCGridMask
    if isinstance(mask, ArakawaCGridMask):
        mask_bool = np.asarray(mask.psi, dtype=bool)
        # psi_irrbound_yids stores row (j) indices,
        # psi_irrbound_xids stores column (i) indices.
        j_b = np.asarray(mask.psi_irrbound_yids, dtype=np.intp)
        i_b = np.asarray(mask.psi_irrbound_xids, dtype=np.intp)
        Ny, Nx = mask_bool.shape
        N_b = len(j_b)
    else:
        from scipy.ndimage import binary_dilation  # local import (offline only)

        mask_bool = np.asarray(mask, dtype=bool)
        Ny, Nx = mask_bool.shape

        # Inner-boundary: mask-interior cells adjacent to at least one exterior cell
        exterior = ~mask_bool
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        dilated = binary_dilation(exterior, structure=struct)
        inner_boundary = mask_bool & dilated  # [Ny, Nx] bool

        j_b, i_b = np.where(inner_boundary)  # row/col indices of boundary points
        N_b = len(j_b)
    if N_b == 0:
        raise ValueError(
            "No inner-boundary points found.  Check that the mask has a "
            "non-trivial interior/exterior structure."
        )

    # Helper: one rectangular spectral solve (numpy interface)
    def _base_solve_np(f_2d: np.ndarray) -> np.ndarray:
        f_jax = jnp.array(f_2d, dtype=float)
        result = _spectral_solve(f_jax, dx, dy, lambda_, base_bc)
        return np.array(result)

    # Green's functions: G[k] = solution to L_rect g_k = e_{b_k}
    # Shape: [N_b, Ny, Nx]
    green = np.zeros((N_b, Ny, Nx), dtype=float)
    for k in range(N_b):
        e_k = np.zeros((Ny, Nx), dtype=float)
        e_k[j_b[k], i_b[k]] = 1.0
        green[k] = _base_solve_np(e_k)

    # Capacitance matrix C[k, l] = green[l] evaluated at boundary point b_k
    # green[:, j_b, i_b] has shape [N_b, N_b] with element [l, k] = green[l, b_k]
    # We need C[k, l], so transpose.
    C = green[:, j_b, i_b].T  # [N_b, N_b]
    C_inv = np.linalg.inv(C)

    return CapacitanceSolver(
        _C_inv=jnp.array(C_inv),
        _green_flat=jnp.array(green.reshape(N_b, Ny * Nx)),
        _j_b=jnp.array(j_b),
        _i_b=jnp.array(i_b),
        dx=float(dx),
        dy=float(dy),
        lambda_=float(lambda_),
        base_bc=base_bc,
    )


# ---------------------------------------------------------------------------
# Convenience wrappers: streamfunction, pressure, PV inversion
# ---------------------------------------------------------------------------

# Type alias for the mask parameter accepted by the convenience wrappers.
_MaskLike = Float[Array, "Ny Nx"] | ArakawaCGridMask
_PrecondLike = Callable[[Float[Array, "Ny Nx"]], Float[Array, "Ny Nx"]]


def _resolve_mask_arr(
    mask: _MaskLike | None,
) -> Float[Array, "Ny Nx"] | None:
    """Extract a float mask array from *mask*, or return None."""
    if mask is None:
        return None
    if isinstance(mask, ArakawaCGridMask):
        return jnp.asarray(mask.psi, dtype=jnp.float32)
    return mask


def _solve_spectral(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
    bc: str,
) -> Float[Array, "Ny Nx"]:
    """Solve using a rectangular spectral solver (DST/DCT/FFT)."""
    return _spectral_solve(rhs, dx, dy, lambda_, bc)


def _solve_cg_method(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
    mask: _MaskLike | None,
    preconditioner: _PrecondLike | None,
) -> Float[Array, "Ny Nx"]:
    """Solve using preconditioned Conjugate Gradient on a masked domain."""
    mask_arr = _resolve_mask_arr(mask)
    if mask_arr is None:
        raise ValueError("method='cg' requires a mask (array or ArakawaCGridMask)")

    def _matvec(x: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        return masked_laplacian(x, mask_arr, dx, dy, lambda_=lambda_)

    if preconditioner is None:
        preconditioner = make_spectral_preconditioner(dx, dy, lambda_=lambda_, bc="fft")

    x, _info = solve_cg(_matvec, rhs * mask_arr, preconditioner=preconditioner)
    return x * mask_arr


def _solve_capacitance_method(
    rhs: Float[Array, "Ny Nx"],
    capacitance_solver: CapacitanceSolver | None,
) -> Float[Array, "Ny Nx"]:
    """Solve using a pre-built capacitance matrix solver."""
    if capacitance_solver is None:
        raise ValueError(
            "method='capacitance' requires a pre-built CapacitanceSolver "
            "(see build_capacitance_solver)"
        )
    return capacitance_solver(rhs)


def _solve_dispatch(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
    bc: str,
    method: str,
    mask: _MaskLike | None,
    capacitance_solver: CapacitanceSolver | None,
    preconditioner: _PrecondLike | None,
) -> Float[Array, "Ny Nx"]:
    """Dispatch an elliptic solve to the selected solver method."""
    if method == "spectral":
        return _solve_spectral(rhs, dx, dy, lambda_, bc)
    if method == "cg":
        return _solve_cg_method(rhs, dx, dy, lambda_, mask, preconditioner)
    if method == "capacitance":
        return _solve_capacitance_method(rhs, capacitance_solver)
    raise ValueError(
        f"method must be 'spectral', 'cg', or 'capacitance'; got {method!r}"
    )


def streamfunction_from_vorticity(
    zeta: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    bc: str = "dst",
    lambda_: float = 0.0,
    method: str = "spectral",
    mask: _MaskLike | None = None,
    capacitance_solver: CapacitanceSolver | None = None,
    preconditioner: _PrecondLike | None = None,
) -> Float[Array, "Ny Nx"]:
    r"""Invert the vorticity–streamfunction relation ∇²ψ − λψ = ζ.

    Solves the Poisson (λ = 0) or Helmholtz (λ ≠ 0) equation to recover the
    streamfunction from relative vorticity.

    Three solver methods are available:

    * ``"spectral"`` — Direct spectral solver (DST/DCT/FFT) for rectangular
      domains.  Selected by *bc*.  Default.
    * ``"cg"`` — Preconditioned Conjugate Gradient for masked / irregular
      domains.  Requires *mask*.  Uses a spectral preconditioner by default,
      or a custom one via *preconditioner*.
    * ``"capacitance"`` — Capacitance matrix method for masked domains.
      Requires a pre-built :class:`CapacitanceSolver` via
      *capacitance_solver*.

    Parameters
    ----------
    zeta : Float[Array, "Ny Nx"]
        Relative vorticity (right-hand side).
    dx, dy : float
        Grid spacings.
    bc : {"dst", "dct", "fft"}
        Boundary-condition type for the spectral solver (used by
        ``method="spectral"``).
        ``"dst"`` (Dirichlet, ψ = 0 on boundary) is the most common choice
        for streamfunction inversion.
    lambda_ : float
        Helmholtz parameter.  Use 0.0 for the pure Poisson problem
        (streamfunction from vorticity).  Non-zero values arise in QG PV
        inversion: (∇² − λ)ψ = q.
    method : {"spectral", "cg", "capacitance"}
        Solver method.  Default: ``"spectral"``.
    mask : Float[Array, "Ny Nx"] or ArakawaCGridMask or None
        Domain mask.  Required for ``method="cg"``.  When an
        :class:`ArakawaCGridMask` is passed the ``psi`` staggering mask is
        extracted automatically.
    capacitance_solver : CapacitanceSolver or None
        Pre-built capacitance solver.  Required for
        ``method="capacitance"``.
    preconditioner : callable or None
        Custom preconditioner for ``method="cg"``.  Signature:
        ``preconditioner(r: Array) -> Array``.  When ``None``, a spectral
        preconditioner (FFT-based) is used automatically.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Streamfunction ψ.
    """
    return _solve_dispatch(
        zeta, dx, dy, lambda_, bc, method, mask, capacitance_solver, preconditioner
    )


def pressure_from_divergence(
    div_u: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    bc: str = "dct",
    method: str = "spectral",
    mask: _MaskLike | None = None,
    capacitance_solver: CapacitanceSolver | None = None,
    preconditioner: _PrecondLike | None = None,
) -> Float[Array, "Ny Nx"]:
    r"""Solve ∇²p = ∇·u for the pressure correction.

    Used in pressure-projection methods (Chorin splitting) where the
    divergence of the provisional velocity field must be removed.

    Solver selection follows the same three-method dispatch as
    :func:`streamfunction_from_vorticity`.

    Parameters
    ----------
    div_u : Float[Array, "Ny Nx"]
        Divergence of the velocity field (right-hand side).
    dx, dy : float
        Grid spacings.
    bc : {"dct", "dst", "fft"}
        Boundary-condition type for the spectral solver.
        ``"dct"`` (Neumann, ∂p/∂n = 0) is the standard choice for
        pressure with solid walls.
    method : {"spectral", "cg", "capacitance"}
        Solver method.  Default: ``"spectral"``.
    mask : Float[Array, "Ny Nx"] or ArakawaCGridMask or None
        Domain mask.  Required for ``method="cg"``.
    capacitance_solver : CapacitanceSolver or None
        Pre-built capacitance solver.  Required for
        ``method="capacitance"``.
    preconditioner : callable or None
        Custom preconditioner for ``method="cg"``.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Pressure field p.
    """
    return _solve_dispatch(
        div_u, dx, dy, 0.0, bc, method, mask, capacitance_solver, preconditioner
    )


def pv_inversion(
    pv: Float[Array, "... Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Float[Array, " nl"],
    bc: str = "dst",
    method: str = "spectral",
    mask: _MaskLike | None = None,
    capacitance_solver: CapacitanceSolver | None = None,
    preconditioner: _PrecondLike | None = None,
) -> Float[Array, "... Ny Nx"]:
    r"""QG potential-vorticity inversion: solve (∇² − λ)ψ = q.

    Supports batched / multi-layer PV fields.  When *lambda_* is a 1-D
    array of shape ``(nl,)``, each layer is solved with its own Helmholtz
    parameter (e.g. 1/Rd² per vertical mode from
    :func:`~finitevolx.decompose_vertical_modes`).

    Solver selection follows the same three-method dispatch as
    :func:`streamfunction_from_vorticity`.

    Parameters
    ----------
    pv : Float[Array, "... Ny Nx"]
        Potential-vorticity field.  Leading dimensions are batched.
    dx, dy : float
        Grid spacings.
    lambda_ : float or Float[Array, " nl"]
        Helmholtz parameter(s).  Scalar for a single layer; array of
        shape ``(nl,)`` for multi-layer inversion.
    bc : {"dst", "dct", "fft"}
        Boundary-condition type (for ``method="spectral"``).
    method : {"spectral", "cg", "capacitance"}
        Solver method.  Default: ``"spectral"``.
    mask : Float[Array, "Ny Nx"] or ArakawaCGridMask or None
        Domain mask.  Required for ``method="cg"``.
    capacitance_solver : CapacitanceSolver or None
        Pre-built capacitance solver.  Required for
        ``method="capacitance"``.
    preconditioner : callable or None
        Custom preconditioner for ``method="cg"``.

    Returns
    -------
    Float[Array, "... Ny Nx"]
        Streamfunction ψ, same shape as *pv*.
    """
    lam = jnp.asarray(lambda_)

    if lam.ndim == 0:
        # Scalar lambda: vmap over all leading dims if present
        if pv.ndim == 2:
            return _solve_dispatch(
                pv,
                dx,
                dy,
                float(lam),
                bc,
                method,
                mask,
                capacitance_solver,
                preconditioner,
            )
        # Flatten leading dims, solve each, reshape
        shape = pv.shape
        flat = pv.reshape(-1, shape[-2], shape[-1])

        def _solve_one(rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
            return _solve_dispatch(
                rhs,
                dx,
                dy,
                float(lam),
                bc,
                method,
                mask,
                capacitance_solver,
                preconditioner,
            )

        out = eqx.filter_vmap(_solve_one)(flat)
        return out.reshape(shape)

    # Array lambda: leading dim must match lam.shape[0]
    if pv.ndim < 3:
        raise ValueError(
            f"pv must have at least 3 dims when lambda_ is an array, "
            f"got shape {pv.shape}"
        )
    nl = lam.shape[0]
    if pv.shape[-3] != nl:
        raise ValueError(
            f"pv.shape[-3]={pv.shape[-3]} does not match lambda_ length {nl}"
        )

    # Solve each layer with its own lambda.
    # We call the Helmholtz solver directly (not _solve_dispatch) because
    # lam_i is a JAX tracer inside vmap and Python-level ``if lam == 0``
    # branches in _spectral_solve would fail.
    if method == "capacitance":
        raise ValueError(
            "method='capacitance' does not support array-valued lambda_; "
            "solve each layer separately or use method='spectral' or 'cg' "
            "for multi-layer problems."
        )

    elif method == "cg":
        mask_arr = _resolve_mask_arr(mask)
        if mask_arr is None:
            raise ValueError("method='cg' requires a mask")

        _precond = preconditioner

        def _solve_layer(
            rhs: Float[Array, "Ny Nx"], lam_i: float
        ) -> Float[Array, "Ny Nx"]:
            def _matvec(x: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
                return masked_laplacian(x, mask_arr, dx, dy, lambda_=lam_i)

            pc = (
                _precond
                if _precond is not None
                else make_spectral_preconditioner(dx, dy, lambda_=lam_i, bc="fft")
            )
            x, _info = solve_cg(_matvec, rhs * mask_arr, preconditioner=pc)
            return x * mask_arr

    elif method == "spectral":
        _helmholtz = _HELMHOLTZ_DISPATCH.get(bc)
        if _helmholtz is None:
            raise ValueError(f"bc must be 'fft', 'dst', or 'dct'; got {bc!r}")

        def _solve_layer(
            rhs: Float[Array, "Ny Nx"], lam_i: float
        ) -> Float[Array, "Ny Nx"]:
            return _helmholtz(rhs, dx, dy, lam_i)

    else:
        raise ValueError(
            f"method must be 'spectral', 'cg', or 'capacitance'; got {method!r}"
        )

    # Flatten any leading batch dims: (..., nl, Ny, Nx) -> (batch, nl, Ny, Nx)
    shape = pv.shape
    ny, nx = shape[-2], shape[-1]
    pv_4d = pv.reshape(-1, nl, ny, nx)

    # vmap over layer axis (pairing each layer with its lambda)
    _solve_layers = eqx.filter_vmap(_solve_layer, in_axes=(0, 0))

    # vmap over the (flattened) batch axis
    out_4d = eqx.filter_vmap(lambda batch: _solve_layers(batch, lam))(pv_4d)
    return out_4d.reshape(shape)
