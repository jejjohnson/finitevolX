"""Geometric multigrid solver for variable-coefficient Helmholtz equations.

Solves the generalised Helmholtz equation on masked Arakawa C-grids::

    nabla . (c(x,y) nabla u) - lambda u = rhs

where ``c(x,y)`` may vary in space (e.g. layer thickness, spatially varying
diffusivity, or stratification-dependent coefficient).

Multigrid Overview
------------------
Simple iterative smoothers like Jacobi efficiently damp *high-frequency*
error but leave *low-frequency* error nearly untouched.  Multigrid
accelerates convergence by transferring the problem to a hierarchy of
progressively coarser grids, where those low-frequency modes become
high-frequency and can be cheaply damped.  The cost per V-cycle is
O(N) thanks to the geometric sum of grid sizes.

Public API
----------
``build_multigrid_solver``
    One-time offline precomputation of the multigrid level hierarchy
    (masks, face coefficients, operator diagonals).  Returns a
    ``MultigridSolver``.
``MultigridSolver``
    Stateless ``equinox.Module`` with three solve modes
    (implicit / one-step / unrolled differentiation).
See also :func:`~finitevolx._src.solvers.preconditioners.make_multigrid_preconditioner`
which wraps a single V-cycle as a preconditioner closure compatible with
:func:`~finitevolx._src.solvers.iterative.solve_cg`.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np

from finitevolx._src.mask.cgrid_mask import ArakawaCGridMask

# ---------------------------------------------------------------------------
# Internal helpers — grid hierarchy construction (numpy, offline)
# ---------------------------------------------------------------------------


def _compute_n_levels(ny: int, nx: int, min_coarse: int = 8) -> int:
    """Auto-detect the number of multigrid levels by repeated halving.

    Starting from the fine grid ``(ny, nx)``, halve both dimensions until
    either would drop below *min_coarse*.  For example, a 64x64 grid with
    ``min_coarse=8`` yields 4 levels: 64 -> 32 -> 16 -> 8.

    Parameters
    ----------
    ny, nx : int
        Fine-grid dimensions.
    min_coarse : int
        Minimum allowed coarse-grid dimension.  Default: 8.

    Returns
    -------
    int
        Number of levels (>= 1).  Level 0 is finest, level L-1 is coarsest.
    """
    n = 1
    while (
        ny % 2 == 0 and nx % 2 == 0 and ny // 2 >= min_coarse and nx // 2 >= min_coarse
    ):
        ny //= 2
        nx //= 2
        n += 1
    return n


def _restrict_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Coarsen a cell-centre mask by 2x via 4-point averaging.

    Each coarse cell is the average of its four constituent fine cells.
    If the average >= *threshold* (default 0.5, i.e. at least 2 of 4 fine
    cells are wet), the coarse cell is classified as wet (1.0)::

        mask_coarse[J, I] = 1  if  mean(mask_fine[2J:2J+2, 2I:2I+2]) >= 0.5

    Parameters
    ----------
    mask : ndarray, shape (Ny, Nx)
        Fine-grid mask (1 = fluid, 0 = land).  Both Ny and Nx must be even.
    threshold : float
        Fraction of wet fine cells required to keep the coarse cell wet.

    Returns
    -------
    ndarray, shape (Ny//2, Nx//2)
        Coarsened mask.
    """
    avg = (mask[::2, ::2] + mask[1::2, ::2] + mask[::2, 1::2] + mask[1::2, 1::2]) / 4.0
    return (avg >= threshold).astype(np.float64)


def _compute_face_masks(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive x-face and y-face masks from a cell-centre mask.

    On a cell-centred grid, *face* values live between adjacent cell centres.
    A face is active (mask = 1) only when **both** cells it connects are wet.

    Face layout (x-direction)::

        cell (j, i)  --face_cx[j, i]-->  cell (j, i+1)

    Face layout (y-direction)::

        cell (j, i)  --face_cy[j, i]-->  cell (j+1, i)

    Parameters
    ----------
    mask : ndarray, shape (Ny, Nx)
        Cell-centre mask (1 = fluid, 0 = land).

    Returns
    -------
    mask_cx : ndarray, shape (Ny, Nx)
        X-direction face mask.  ``mask_cx[j, i] = mask[j, i] * mask[j, i+1]``.
        The last column ``mask_cx[:, -1]`` is always 0 (no right neighbour).
    mask_cy : ndarray, shape (Ny, Nx)
        Y-direction face mask.  ``mask_cy[j, i] = mask[j, i] * mask[j+1, i]``.
        The last row ``mask_cy[-1, :]`` is always 0 (no lower neighbour).
    """
    # x-face: active between (j, i) and (j, i+1)
    mask_cx = np.zeros_like(mask)
    mask_cx[:, :-1] = mask[:, :-1] * mask[:, 1:]
    # y-face: active between (j, i) and (j+1, i)
    mask_cy = np.zeros_like(mask)
    mask_cy[:-1, :] = mask[:-1, :] * mask[1:, :]
    return mask_cx, mask_cy


def _interpolate_coeff_to_faces(
    coeff: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a cell-centre coefficient ``c(x,y)`` to face coefficients.

    The face coefficient is the arithmetic mean of the two adjacent cell
    values, multiplied by the face mask (so faces bordering land are zero)::

        cx[j, i] = 0.5 * (coeff[j, i] + coeff[j, i + 1]) * mask_cx[j, i]
        cy[j, i] = 0.5 * (coeff[j, i] + coeff[j + 1, i]) * mask_cy[j, i]

    These face coefficients appear directly in the variable-coefficient
    Helmholtz stencil (see :func:`_apply_operator`).

    Parameters
    ----------
    coeff : ndarray, shape (Ny, Nx)
        Cell-centre coefficient field.
    mask : ndarray, shape (Ny, Nx)
        Cell-centre mask (used to derive face masks).

    Returns
    -------
    cx : ndarray, shape (Ny, Nx)
        X-direction face coefficient (zero at domain boundaries).
    cy : ndarray, shape (Ny, Nx)
        Y-direction face coefficient (zero at domain boundaries).
    """
    mask_cx, mask_cy = _compute_face_masks(mask)
    cx = np.zeros_like(coeff)
    cx[:, :-1] = 0.5 * (coeff[:, :-1] + coeff[:, 1:])
    cx *= mask_cx
    cy = np.zeros_like(coeff)
    cy[:-1, :] = 0.5 * (coeff[:-1, :] + coeff[1:, :])
    cy *= mask_cy
    return cx, cy


def _restrict_coeff(
    coeff: np.ndarray,
    mask_fine: np.ndarray,
    mask_coarse: np.ndarray,
) -> np.ndarray:
    """Coarsen a cell-centre coefficient field by 2x (mask-weighted average).

    Each coarse cell averages the coefficient over its four fine sub-cells,
    weighted by the fine mask to avoid contamination from land cells::

        c_coarse[J, I] = sum(c_fine * m_fine over 2x2 block)
                         / max(sum(m_fine over 2x2 block), 1)

    Parameters
    ----------
    coeff : ndarray, shape (Ny, Nx)
        Fine-grid coefficient.
    mask_fine : ndarray, shape (Ny, Nx)
        Fine-grid mask.
    mask_coarse : ndarray, shape (Ny//2, Nx//2)
        Coarse-grid mask (from :func:`_restrict_mask`).

    Returns
    -------
    ndarray, shape (Ny//2, Nx//2)
        Coarsened coefficient, zeroed outside the coarse mask.
    """
    c_m = coeff * mask_fine
    sum4 = c_m[::2, ::2] + c_m[1::2, ::2] + c_m[::2, 1::2] + c_m[1::2, 1::2]
    count = (
        mask_fine[::2, ::2]
        + mask_fine[1::2, ::2]
        + mask_fine[::2, 1::2]
        + mask_fine[1::2, 1::2]
    )
    divisor = np.maximum(count, 1.0)
    return (sum4 / divisor) * mask_coarse


def _compute_inv_diagonal(
    cx: np.ndarray,
    cy: np.ndarray,
    mask: np.ndarray,
    dx: float,
    dy: float,
    lambda_: float,
) -> np.ndarray:
    """Precompute ``1 / diag(A)`` for the weighted Jacobi smoother.

    The diagonal entry of the variable-coefficient Helmholtz operator at
    cell ``(j, i)`` collects the coefficients of ``u[j, i]`` in the stencil::

        diag[j, i] = -(cx[j, i] + cx[j, i-1]) / dx**2
                     -(cy[j, i] + cy[j-1, i]) / dy**2
                     - lambda

    where ``cx[j, i]`` is the x-face coefficient between ``(j,i)`` and
    ``(j,i+1)`` and ``cx[j, i-1]`` is the face between ``(j,i-1)`` and
    ``(j,i)``.  Outside the mask, ``diag = 0`` and the inverse is set to 0.

    Parameters
    ----------
    cx, cy : ndarray, shape (Ny, Nx)
        Face coefficients (from :func:`_interpolate_coeff_to_faces`).
    mask : ndarray, shape (Ny, Nx)
        Cell-centre mask.
    dx, dy : float
        Grid spacings.
    lambda_ : float
        Helmholtz parameter.

    Returns
    -------
    ndarray, shape (Ny, Nx)
        Inverse diagonal, zeroed outside the mask.  Used by
        :func:`_weighted_jacobi` as ``D^{-1}`` in the update formula.
    """
    # Shift face coefficients to get the "west" and "south" faces at (j, i).
    # cx_west[j, i] = cx[j, i-1] = face between cells (j, i-1) and (j, i)
    cx_west = np.zeros_like(cx)
    cx_west[:, 1:] = cx[:, :-1]
    # cy_south[j, i] = cy[j-1, i] = face between cells (j-1, i) and (j, i)
    cy_south = np.zeros_like(cy)
    cy_south[1:, :] = cy[:-1, :]

    # Diagonal = sum of (negative) face contributions minus lambda
    diag = -(cx + cx_west) / dx**2 - (cy + cy_south) / dy**2 - lambda_
    diag *= mask

    # Safe inversion: avoid division by zero outside domain / on land cells.
    # Two-step where() avoids evaluating 1.0/0.0 on all elements.
    safe_diag = np.where(np.abs(diag) > 1e-30, diag, 1.0)
    inv_diag = np.where(np.abs(diag) > 1e-30, 1.0 / safe_diag, 0.0)
    return inv_diag * mask


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class MultigridLevel(eqx.Module):
    """Precomputed data for a single multigrid level.

    Each level stores the arrays needed to apply the Helmholtz operator,
    smooth, restrict, and prolongate at that grid resolution.  Levels are
    constructed by :func:`build_multigrid_solver` (offline) and stored
    as frozen JAX arrays in the ``MultigridSolver``.

    Attributes
    ----------
    mask : Float[Array, "Ny Nx"]
        Cell-centre domain mask (1 = fluid, 0 = land) at this level's
        resolution.
    cx : Float[Array, "Ny Nx"]
        X-direction face coefficient, including face mask.
        ``cx[j, i]`` is the coefficient on the face between cells
        ``(j, i)`` and ``(j, i+1)``.  Zero if either cell is land.
    cy : Float[Array, "Ny Nx"]
        Y-direction face coefficient, including face mask.
        ``cy[j, i]`` is the coefficient on the face between cells
        ``(j, i)`` and ``(j+1, i)``.  Zero if either cell is land.
    dx, dy : float
        Grid spacing at this level (doubles at each coarser level).
    lambda_ : float
        Helmholtz parameter (same on all levels).
    inv_diagonal : Float[Array, "Ny Nx"]
        Precomputed ``1 / diag(A)`` for the weighted Jacobi smoother.
        Zero outside the mask.
    """

    mask: Float[Array, "Ny Nx"]
    cx: Float[Array, "Ny Nx"]
    cy: Float[Array, "Ny Nx"]
    dx: float
    dy: float
    lambda_: float = eqx.field(static=True)
    inv_diagonal: Float[Array, "Ny Nx"]


# ---------------------------------------------------------------------------
# Core JAX operators (JIT-compiled, run on accelerator)
# ---------------------------------------------------------------------------


def _apply_operator(
    u: Float[Array, "Ny Nx"],
    level: MultigridLevel,
) -> Float[Array, "Ny Nx"]:
    r"""Apply the variable-coefficient Helmholtz operator: A u.

    Discrete stencil (5-point, conservative finite-volume form)::

        (Au)[j,i] =
          (cx[j,i] * (u[j,i+1] - u[j,i]) - cx[j,i-1] * (u[j,i] - u[j,i-1])) / dx^2
        + (cy[j,i] * (u[j+1,i] - u[j,i]) - cy[j-1,i] * (u[j,i] - u[j-1,i])) / dy^2
        - lambda * u[j,i]

    This is a discrete approximation of ``div(c grad u) - lambda u``.  The
    ``cx`` / ``cy`` face coefficients already include the face mask, so
    fluxes across land boundaries are automatically zero.

    Boundary treatment
    ~~~~~~~~~~~~~~~~~~
    Zero normal flux at the domain edges is enforced primarily by the
    face coefficients ``cx``/``cy``: boundary faces have zero coefficient,
    so no flux crosses the domain boundary regardless of ghost values.
    Out-of-bounds neighbours are zero-padded as an implementation
    convenience (avoids periodic wrapping from ``jnp.roll``).  Unlike
    ``jnp.roll``, this does not introduce spurious fluxes across
    opposite edges — the natural choice for bounded ocean basins.

    Implementation note
    ~~~~~~~~~~~~~~~~~~~
    Shifted arrays are constructed via ``jnp.zeros_like().at[].set()``
    rather than ``jnp.roll``.  The zero padding is a convenience that
    avoids periodic wrapping; the actual BC is enforced by the zeroed
    face coefficients at boundary faces::

        u_east[j, i] = u[j, i+1]   for i < Nx-1
                      = 0           for i = Nx-1   (ghost = 0)

    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        Input field (solution iterate).
    level : MultigridLevel
        Precomputed operator data for this grid level.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Result of ``A u``, zeroed outside the mask.
    """
    u_m = u * level.mask  # zero u outside domain before computing fluxes
    cx, cy = level.cx, level.cy
    dx2 = level.dx**2
    dy2 = level.dy**2

    # --- Shifted neighbours (zero-padded; actual BC via zeroed face coeffs) ---
    # u_east[j, i] = u[j, i+1];  u_east[j, Nx-1] = 0
    u_east = jnp.zeros_like(u_m).at[:, :-1].set(u_m[:, 1:])
    # u_west[j, i] = u[j, i-1];  u_west[j, 0] = 0
    u_west = jnp.zeros_like(u_m).at[:, 1:].set(u_m[:, :-1])
    # u_north[j, i] = u[j+1, i]; u_north[Ny-1, i] = 0
    u_north = jnp.zeros_like(u_m).at[:-1, :].set(u_m[1:, :])
    # u_south[j, i] = u[j-1, i]; u_south[0, i] = 0
    u_south = jnp.zeros_like(u_m).at[1:, :].set(u_m[:-1, :])

    # --- Shifted face coefficients ---
    # cx_west[j, i] = cx[j, i-1] (face between (j,i-1) and (j,i))
    cx_west = jnp.zeros_like(cx).at[:, 1:].set(cx[:, :-1])
    # cy_south[j, i] = cy[j-1, i] (face between (j-1,i) and (j,i))
    cy_south = jnp.zeros_like(cy).at[1:, :].set(cy[:-1, :])

    # --- Assemble the 5-point stencil ---
    # x-flux divergence: [cx_{j,i} * (u_{j,i+1} - u_{j,i})
    #                    - cx_{j,i-1} * (u_{j,i} - u_{j,i-1})] / dx^2
    # y-flux divergence: analogous in the y-direction
    lap = (cx * (u_east - u_m) - cx_west * (u_m - u_west)) / dx2 + (
        cy * (u_north - u_m) - cy_south * (u_m - u_south)
    ) / dy2

    return (lap - level.lambda_ * u_m) * level.mask


def _weighted_jacobi(
    u: Float[Array, "Ny Nx"],
    rhs: Float[Array, "Ny Nx"],
    level: MultigridLevel,
    omega: float,
    n_iters: int,
) -> Float[Array, "Ny Nx"]:
    r"""Weighted Jacobi smoother for the multigrid V-cycle.

    Each iteration updates every cell simultaneously::

        u^{k+1} = u^k + omega * D^{-1} * (rhs - A u^k)

    where ``D = diag(A)`` (precomputed in ``level.inv_diagonal``) and
    ``omega`` is the relaxation weight.  The update is equivalent to
    solving ``D u^{k+1} = D u^k + omega * (rhs - A u^k)``.

    Why weighted Jacobi works as a multigrid smoother
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Jacobi efficiently damps *high-frequency* error modes (oscillatory
    components with wavelength ~ 2*dx) but barely touches *low-frequency*
    modes.  This is exactly what multigrid needs: after a few Jacobi
    sweeps the residual is smooth enough to be accurately represented on
    the coarse grid, where those low-frequency modes become high-frequency
    and can be damped cheaply.

    The weight ``omega < 1`` (typically 0.8-0.95) under-relaxes the update
    to prevent amplification of certain error modes, improving smoothing
    for the 5-point stencil.

    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        Current solution iterate.
    rhs : Float[Array, "Ny Nx"]
        Right-hand side of ``A u = rhs``.
    level : MultigridLevel
        Operator data (includes ``inv_diagonal`` = ``D^{-1}``).
    omega : float
        Relaxation weight, typically 0.8-0.95.
    n_iters : int
        Number of Jacobi sweeps.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Smoothed solution, zeroed outside the mask.
    """

    def _step(_, u_cur: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        r = rhs - _apply_operator(u_cur, level)  # residual: r = f - A u
        return (u_cur + omega * level.inv_diagonal * r) * level.mask

    return jax.lax.fori_loop(0, n_iters, _step, u)


def _restrict(
    v: Float[Array, "Ny Nx"],
    mask_fine: Float[Array, "Ny Nx"],
    mask_coarse: Float[Array, "Ny_c Nx_c"],
) -> Float[Array, "Ny_c Nx_c"]:
    """Cell-centred full-weighting restriction (fine -> coarse).

    Transfers a fine-grid field to the coarse grid by averaging each 2x2
    block of fine cells, weighted by the mask to handle irregular domains::

        v_coarse[J, I] = sum(v_fine[j,i] * mask_fine[j,i]  for (j,i) in block)
                       / max(sum(mask_fine[j,i]             for (j,i) in block), 1)

    where the block is ``{(2J, 2I), (2J+1, 2I), (2J, 2I+1), (2J+1, 2I+1)}``.

    The mask-weighted divisor ensures that coarse cells adjacent to land
    boundaries are averaged over wet fine cells only, preventing land
    contamination.  If all four fine cells are dry, the divisor clips to 1
    to avoid division by zero (the result is 0 regardless).

    Parameters
    ----------
    v : Float[Array, "Ny Nx"]
        Fine-grid field to restrict.
    mask_fine : Float[Array, "Ny Nx"]
        Fine-grid mask.
    mask_coarse : Float[Array, "Ny_c Nx_c"]
        Coarse-grid mask (from :func:`_restrict_mask`).

    Returns
    -------
    Float[Array, "Ny_c Nx_c"]
        Coarsened field, zeroed outside the coarse mask.
    """
    v_m = v * mask_fine
    sum4 = v_m[::2, ::2] + v_m[1::2, ::2] + v_m[::2, 1::2] + v_m[1::2, 1::2]
    count = (
        mask_fine[::2, ::2]
        + mask_fine[1::2, ::2]
        + mask_fine[::2, 1::2]
        + mask_fine[1::2, 1::2]
    )
    divisor = jnp.maximum(count, 1.0)
    return (sum4 / divisor) * mask_coarse


def _prolongate(
    v_coarse: Float[Array, "Ny_c Nx_c"],
    mask_coarse: Float[Array, "Ny_c Nx_c"],
    mask_fine: Float[Array, "Ny Nx"],
) -> Float[Array, "Ny Nx"]:
    r"""Cell-centred bilinear prolongation (coarse -> fine).

    Each coarse cell ``(J, I)`` maps to four fine sub-cells at positions
    ``(2J, 2I)``, ``(2J, 2I+1)``, ``(2J+1, 2I)``, ``(2J+1, 2I+1)``.
    The value at each sub-cell is a bilinear interpolation using the
    coarse cell and its three nearest neighbours, weighted 9/3/3/1::

        v_fine[2J, 2I] = (9*v[J,I] + 3*v[J,I-1] + 3*v[J-1,I] + v[J-1,I-1])
                        / (9*m[J,I] + 3*m[J,I-1] + 3*m[J-1,I] + m[J-1,I-1])

    The other three sub-cells use the same weights but with shifted
    neighbour directions (e.g. ``(J, I+1)`` instead of ``(J, I-1)``).

    The 9:3:3:1 ratio comes from bilinear interpolation distances:
    the sub-cell at ``(2J, 2I)`` is closest to coarse cell ``(J, I)``
    (distance 0.25*dx in each direction), and farthest from the
    diagonal neighbour ``(J-1, I-1)`` (distance 0.75*dx in each direction).
    The weight is ``(1-d_x)(1-d_y)`` where ``d`` is the normalised distance.

    The mask-weighted divisor handles irregular boundaries: if a coarse
    neighbour is land (mask=0), its contribution is excluded and the
    remaining weights are renormalised.

    Parameters
    ----------
    v_coarse : Float[Array, "Ny_c Nx_c"]
        Coarse-grid field to prolongate.
    mask_coarse : Float[Array, "Ny_c Nx_c"]
        Coarse-grid mask.
    mask_fine : Float[Array, "Ny Nx"]
        Fine-grid mask.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Interpolated fine-grid field, zeroed outside the fine mask.
    """
    ny_c, nx_c = v_coarse.shape
    ny_f, nx_f = 2 * ny_c, 2 * nx_c
    v_m = v_coarse * mask_coarse
    m = mask_coarse

    # Pad with zeros for neighbour access at boundaries (ghost ring).
    # After padding: padded[1:-1, 1:-1] == original array.
    vp = jnp.pad(v_m, ((1, 1), (1, 1)), mode="constant")
    mp = jnp.pad(m, ((1, 1), (1, 1)), mode="constant")

    # Centre slice of padded array (the original coarse grid)
    vc = vp[1:-1, 1:-1]
    mc = mp[1:-1, 1:-1]

    def _interp(dj: int, di: int) -> Float[Array, "Ny_c Nx_c"]:
        """Bilinear interpolation for sub-cell at offset ``(dj, di)`` in {0,1}^2.

        ``dj=0, di=0`` -> upper-left sub-cell (nearest to ``(J-1, I-1)``)
        ``dj=0, di=1`` -> upper-right sub-cell (nearest to ``(J-1, I+1)``)
        ``dj=1, di=0`` -> lower-left sub-cell (nearest to ``(J+1, I-1)``)
        ``dj=1, di=1`` -> lower-right sub-cell (nearest to ``(J+1, I+1)``)
        """
        # Map {0,1} offset to {-1,+1} neighbour direction
        sj = 2 * dj - 1  # -1 for dj=0 (look at J-1), +1 for dj=1 (look at J+1)
        si = 2 * di - 1  # -1 for di=0 (look at I-1), +1 for di=1 (look at I+1)

        # Extract neighbour values and masks from padded array via static slicing
        v_si = vp[1:-1, 1 + si : nx_c + 1 + si]  # horizontal neighbour
        m_si = mp[1:-1, 1 + si : nx_c + 1 + si]
        v_sj = vp[1 + sj : ny_c + 1 + sj, 1:-1]  # vertical neighbour
        m_sj = mp[1 + sj : ny_c + 1 + sj, 1:-1]
        v_diag = vp[1 + sj : ny_c + 1 + sj, 1 + si : nx_c + 1 + si]  # diagonal
        m_diag = mp[1 + sj : ny_c + 1 + sj, 1 + si : nx_c + 1 + si]

        # 9/3/3/1 weighted sum (bilinear weights)
        w_sum = 9 * vc + 3 * v_si + 3 * v_sj + 1 * v_diag
        d_sum = 9 * mc + 3 * m_si + 3 * m_sj + 1 * m_diag
        return w_sum / jnp.maximum(d_sum, 1.0)

    # Scatter the four sub-cell values into the fine grid
    v_fine = jnp.zeros((ny_f, nx_f))
    v_fine = v_fine.at[::2, ::2].set(_interp(0, 0))  # sub-cell (2J,   2I)
    v_fine = v_fine.at[::2, 1::2].set(_interp(0, 1))  # sub-cell (2J,   2I+1)
    v_fine = v_fine.at[1::2, ::2].set(_interp(1, 0))  # sub-cell (2J+1, 2I)
    v_fine = v_fine.at[1::2, 1::2].set(_interp(1, 1))  # sub-cell (2J+1, 2I+1)

    return v_fine * mask_fine


# ---------------------------------------------------------------------------
# Multigrid solver
# ---------------------------------------------------------------------------


class MultigridSolver(eqx.Module):
    r"""Geometric multigrid V-cycle solver for the variable-coefficient
    Helmholtz equation::

        div(c(x,y) grad u) - lambda u = rhs

    Supports spatially varying coefficients ``c(x, y)`` on staggered faces
    and masked (irregular) domains.

    V-Cycle Algorithm
    -----------------
    The V-cycle is a recursive algorithm that visits coarser grids to
    correct low-frequency error that the smoother cannot resolve::

        Level 0 (fine)    *---smooth---*-----------*---smooth---*
                               | restrict    prolong |
        Level 1           .....*---smooth---*---smooth---*.....
                                    | restrict  prolong |
        Level 2 (coarse)  ..........*--bottom solve--*..........

    At each level:

    1. **Pre-smooth** (nu_1 weighted Jacobi iterations): damp
       high-frequency error.
    2. **Compute residual**: ``r = rhs - A u``.
    3. **Restrict** residual to the coarse grid (2x2 averaging).
    4. **Recurse**: solve for the error on the coarse grid.
    5. **Prolongate** the coarse correction back to the fine grid
       (bilinear interpolation).
    6. **Post-smooth** (nu_2 weighted Jacobi iterations): damp any
       high-frequency error introduced by the prolongation.

    The recursion is **statically unrolled** at JAX trace time because
    each level has a different array shape.  All integer parameters
    (``n_levels``, ``n_pre``, etc.) are ``eqx.field(static=True)``,
    so the unrolled structure is visible to the XLA compiler.

    Differentiation Modes
    ---------------------
    Three solve modes trade off backward-pass cost vs gradient accuracy:

    * ``__call__`` — **Implicit differentiation** via
      ``jax.lax.custom_linear_solve(symmetric=True)``.  The backward pass
      solves the adjoint system ``A^T v = dL/du`` with one multigrid call.
      Since ``A`` is symmetric, this costs the same as the forward pass.
      O(1) memory, exact gradients for the linear system being solved
      (gradient accuracy is limited by how well the V-cycles approximate
      ``A^{-1}``, i.e. depends on ``n_cycles`` and smoother settings).

    * ``solve_onestep`` — **One-step differentiation** (Bolte, Pauwels &
      Vaiter, NeurIPS 2023).  Runs K V-cycles, applies ``stop_gradient``
      after K-1, then autodiffs through the last cycle only.  O(1 V-cycle)
      memory, approximate gradients with error O(rho).

    * ``solve_unrolled`` — **Unrolled differentiation** via
      ``jax.lax.fori_loop``.  Backward replays all K iterations.
      O(K) memory, exact through-iteration gradients (reproduces the
      forward computation exactly, so gradient accuracy matches the
      forward solve accuracy).

    Parameters
    ----------
    levels : tuple of MultigridLevel
        Precomputed level data, finest (index 0) to coarsest (index L-1).
    n_levels : int
        Number of multigrid levels.
    n_pre, n_post : int
        Pre- and post-smoothing iterations (weighted Jacobi).
    n_coarse : int
        Jacobi iterations on the coarsest grid (bottom solver).
    omega : float
        Jacobi relaxation weight (typically 0.8-0.95).
    n_cycles : int
        Number of V-cycles per solve.
    """

    levels: tuple[MultigridLevel, ...]
    n_levels: int = eqx.field(static=True)
    n_pre: int = eqx.field(static=True, default=6)
    n_post: int = eqx.field(static=True, default=6)
    n_coarse: int = eqx.field(static=True, default=50)
    omega: float = eqx.field(static=True, default=0.95)
    n_cycles: int = eqx.field(static=True, default=5)

    # -- Public API ----------------------------------------------------------

    def __call__(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        r"""Solve ``A u = rhs`` with implicit differentiation.

        The forward pass runs K V-cycles (identical to :meth:`solve_unrolled`).

        The backward pass uses ``jax.lax.custom_linear_solve`` with
        ``symmetric=True`` to compute gradients via the implicit function
        theorem (IFT) rather than unrolling through V-cycle iterations.

        For a scalar loss ``L(u)``, the gradient w.r.t. the RHS is::

            dL/d(rhs) = A^{-T} dL/du = A^{-1} dL/du   (since A = A^T)

        This adjoint solve is just another multigrid call — so the
        backward pass costs the same as the forward pass, with O(1) extra
        memory (no iteration history stored).

        Parameters
        ----------
        rhs : Float[Array, "Ny Nx"]
            Right-hand side of the linear system.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Approximate solution ``u``.
        """
        level = self.levels[0]

        # _matvec defines the linear operator A for custom_linear_solve
        def _matvec(u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
            return _apply_operator(u, level)

        # _solve provides the forward solve; custom_linear_solve will also
        # call it for the backward (adjoint) pass since symmetric=True
        def _solve(
            _matvec_fn: Callable,
            b: Float[Array, "Ny Nx"],
        ) -> Float[Array, "Ny Nx"]:
            return self._run_vcycles(b)

        return jax.lax.custom_linear_solve(_matvec, rhs, solve=_solve, symmetric=True)

    def solve_onestep(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        r"""Solve with one-step differentiation (Bolte et al., NeurIPS 2023).

        Runs K V-cycles to convergence, but autodiff only sees the **last**
        cycle.  The first K-1 cycles are wrapped in ``jax.lax.stop_gradient``
        so they contribute no backward-pass cost.

        The forward result is identical to :meth:`solve_unrolled`.  The
        gradient approximation error is O(rho) where rho is the per-cycle
        convergence rate (typically 0.1-0.3 for multigrid).

        Gradient structure::

            u_0 = 0
            u_1 = V(u_0, rhs)
            ...
            u_{K-1} = V(u_{K-2}, rhs)      <-- stop_gradient here
            u_K     = V(u_{K-1}, rhs)       <-- autodiff traces only this

        Parameters
        ----------
        rhs : Float[Array, "Ny Nx"]
            Right-hand side.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Approximate solution.

        References
        ----------
        Bolte, Pauwels & Vaiter (NeurIPS 2023). "One-step differentiation
        of iterative algorithms." https://arxiv.org/abs/2305.13768
        """
        u = jnp.zeros_like(rhs)

        # Run K-1 cycles with stop_gradient: the forward pass converges
        # normally, but no gradient graph is built for these iterations
        if self.n_cycles > 1:

            def _body(_: int, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
                return self.v_cycle(u, rhs)

            u = jax.lax.fori_loop(0, self.n_cycles - 1, _body, u)
            u = jax.lax.stop_gradient(u)

        # Final V-cycle: JAX autodiff traces through this one only.
        # The gradient cost is O(1 V-cycle) regardless of n_cycles.
        return self.v_cycle(u, rhs)

    def solve_unrolled(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Solve by unrolling all V-cycles through ``lax.fori_loop``.

        The backward pass differentiates through every iteration, storing
        intermediate states for replay.  This costs O(n_cycles) memory.

        Use this mode when you specifically need gradients through the
        iteration dynamics itself.  For most applications, prefer
        ``__call__`` (implicit differentiation) which gives exact gradients
        at O(1) memory cost.

        Parameters
        ----------
        rhs : Float[Array, "Ny Nx"]
            Right-hand side.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Approximate solution.
        """
        return self._run_vcycles(rhs)

    def v_cycle(
        self,
        u: Float[Array, "Ny Nx"],
        rhs: Float[Array, "Ny Nx"],
        level_idx: int = 0,
    ) -> Float[Array, "Ny Nx"]:
        """Execute a single multigrid V-cycle starting at *level_idx*.

        Algorithm::

            if coarsest level:
                return jacobi(u, rhs, n_coarse)   # bottom solve

            u = jacobi(u, rhs, n_pre)             # 1. pre-smooth
            r = rhs - A(u)                        # 2. compute residual
            r_c = restrict(r)                     # 3. restrict to coarse grid
            e_c = v_cycle(0, r_c, level+1)        # 4. recurse (solve A_c e_c = r_c)
            u = u + prolongate(e_c)               # 5. correct with coarse error
            u = jacobi(u, rhs, n_post)            # 6. post-smooth

        The recursion unrolls statically at JAX trace time because
        ``level_idx`` and ``n_levels`` are Python ints (static fields).

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            Initial guess (typically zeros for the error equation on
            coarse grids, or the current iterate on the fine grid).
        rhs : Float[Array, "Ny Nx"]
            Right-hand side (original RHS on the fine grid, or the
            restricted residual on coarser grids).
        level_idx : int
            Current level (0 = finest, n_levels-1 = coarsest).

        Returns
        -------
        Float[Array, "Ny Nx"]
            Improved solution after one V-cycle.
        """
        level = self.levels[level_idx]

        # --- Coarsest level: iterated Jacobi as the bottom solver ---
        # The grid is small (typically 8x8 to 16x16), so many Jacobi
        # iterations are cheap and sufficient for convergence.
        if level_idx == self.n_levels - 1:
            return _weighted_jacobi(u, rhs, level, self.omega, self.n_coarse)

        # 1. Pre-smooth: damp high-frequency error on the current grid
        u = _weighted_jacobi(u, rhs, level, self.omega, self.n_pre)

        # 2. Compute residual: r = f - A(u)
        #    After smoothing, r contains mostly low-frequency error that
        #    the smoother cannot resolve at this grid resolution.
        r = (rhs - _apply_operator(u, level)) * level.mask

        # 3. Restrict residual to the coarse grid (2x coarser in each dim)
        coarse_level = self.levels[level_idx + 1]
        r_coarse = _restrict(r, level.mask, coarse_level.mask)

        # 4. Recurse: solve A_c * e_c = r_c on the coarse grid.
        #    Start from zero because we're solving for the *error*, not
        #    the solution itself.  On the coarse grid, the low-frequency
        #    residual becomes high-frequency and can be efficiently damped.
        e_coarse = self.v_cycle(jnp.zeros_like(r_coarse), r_coarse, level_idx + 1)

        # 5. Prolongate (interpolate) the coarse correction back to the
        #    fine grid and add to the current solution: u <- u + e_fine
        e_fine = _prolongate(e_coarse, coarse_level.mask, level.mask)
        u = (u + e_fine) * level.mask

        # 6. Post-smooth: clean up high-frequency error introduced by
        #    the coarse-to-fine interpolation
        u = _weighted_jacobi(u, rhs, level, self.omega, self.n_post)
        return u

    # -- Internal ------------------------------------------------------------

    def _run_vcycles(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Run *n_cycles* V-cycles from a zero initial guess.

        Uses ``jax.lax.fori_loop`` so that the iteration count does not
        increase the traced program size (unlike a Python for-loop, which
        would unroll each cycle into separate XLA operations).
        """

        def _body(_: int, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
            return self.v_cycle(u, rhs)

        u0 = jnp.zeros_like(rhs)
        return jax.lax.fori_loop(0, self.n_cycles, _body, u0)


# ---------------------------------------------------------------------------
# Factory — offline precomputation (runs on CPU with NumPy)
# ---------------------------------------------------------------------------


def build_multigrid_solver(
    mask: np.ndarray | Float[Array, "Ny Nx"] | ArakawaCGridMask,
    dx: float,
    dy: float,
    lambda_: float = 0.0,
    coeff: np.ndarray | Float[Array, "Ny Nx"] | None = None,
    n_levels: int | None = None,
    n_pre: int = 6,
    n_post: int = 6,
    n_coarse: int = 50,
    omega: float = 0.95,
    n_cycles: int = 5,
) -> MultigridSolver:
    r"""Build a multigrid solver with precomputed level hierarchies.

    This is an **offline** function (runs once on CPU with NumPy) that
    constructs the entire multigrid hierarchy:

    1. **Mask coarsening**: at each level, the cell mask is coarsened by
       2x via 4-point averaging (threshold >= 0.5).
    2. **Coefficient interpolation**: the cell-centre coefficient
       ``c(x, y)`` is averaged to staggered face coefficients ``cx``,
       ``cy`` at each level, then coarsened for the next level.
    3. **Diagonal precomputation**: the inverse diagonal ``D^{-1}`` of
       the Helmholtz operator is computed at each level for the Jacobi
       smoother.
    4. **Grid spacing doubling**: ``dx`` and ``dy`` double at each
       coarser level.

    Grid hierarchy example (64x64, auto levels)::

        Level 0:  64 x 64   (dx,    dy)     <- finest (solve here)
        Level 1:  32 x 32   (2*dx,  2*dy)
        Level 2:  16 x 16   (4*dx,  4*dy)
        Level 3:   8 x  8   (8*dx,  8*dy)   <- coarsest (bottom solve)

    The returned ``MultigridSolver`` is an immutable ``equinox.Module``
    with frozen JAX arrays.  All subsequent calls (forward solves,
    gradients, JIT compilation) use the precomputed hierarchy.

    Parameters
    ----------
    mask : array, shape (Ny, Nx), or ArakawaCGridMask
        Domain mask (1 = fluid, 0 = land).  ``None`` is *not* accepted;
        pass ``np.ones((Ny, Nx))`` for a rectangular domain.
        When an :class:`ArakawaCGridMask` is passed, the ``psi``
        staggering mask is extracted automatically.
    dx, dy : float
        Fine-grid spacings (metres or non-dimensional).
    lambda_ : float
        Helmholtz parameter (>= 0).  Use 0.0 for pure Poisson (Laplacian
        only).  For QG PV inversion, ``lambda_ = 1 / Rd**2``.
    coeff : array, shape (Ny, Nx), or None
        Spatially varying coefficient ``c(x, y)`` at cell centres.
        ``None`` -> constant coefficient = 1 everywhere (reduces to the
        standard constant-coefficient Helmholtz operator).
    n_levels : int or None
        Number of multigrid levels.  ``None`` -> auto-detect by halving
        until either dimension drops below 8.  Both dimensions must be
        divisible by ``2**(n_levels - 1)``; a ``ValueError`` is raised
        otherwise.
    n_pre, n_post : int
        Number of pre- and post-smoothing Jacobi iterations per V-cycle.
        More smoothing improves the convergence rate but increases cost
        per cycle.  Default: 6 each.
    n_coarse : int
        Number of Jacobi iterations on the coarsest grid (bottom solver).
        The coarsest grid is small (typically 8x8), so this is cheap.
        Default: 50.
    omega : float
        Jacobi relaxation weight (0 < omega < 1).  Under-relaxation
        improves smoothing stability.  Default: 0.95.
    n_cycles : int
        Number of V-cycles applied per solve.  5 cycles typically reduce
        the residual by 3-5 orders of magnitude.  Default: 5.

    Returns
    -------
    MultigridSolver
        Ready-to-use solver (JIT-compilable ``equinox.Module``).

    Raises
    ------
    ValueError
        If grid dimensions are not divisible by ``2**(n_levels - 1)``.
    """
    # --- Extract mask as a NumPy float64 array ---
    if isinstance(mask, ArakawaCGridMask):
        mask_np = np.asarray(mask.psi, dtype=np.float64)
    else:
        mask_np = np.asarray(mask, dtype=np.float64)

    ny, nx = mask_np.shape

    # --- Auto-detect or validate number of levels ---
    if n_levels is None:
        n_levels = _compute_n_levels(ny, nx)
    factor = 2 ** (n_levels - 1)
    if ny % factor != 0 or nx % factor != 0:
        raise ValueError(
            f"Grid shape ({ny}, {nx}) is not divisible by "
            f"2^(n_levels-1) = {factor}.  Choose a different n_levels or "
            f"pad the grid."
        )

    # --- Default coefficient: c(x,y) = 1 everywhere ---
    if coeff is None:
        coeff_np = np.ones_like(mask_np)
    else:
        coeff_np = np.asarray(coeff, dtype=np.float64)
    coeff_np = coeff_np * mask_np  # zero outside domain

    # --- Build level hierarchy (finest to coarsest) ---
    levels: list[MultigridLevel] = []
    cur_mask = mask_np
    cur_coeff = coeff_np
    cur_dx, cur_dy = float(dx), float(dy)

    for lev in range(n_levels):
        # Interpolate cell-centre coefficient to staggered face coefficients.
        # cx[j,i] = face coeff between (j,i) and (j,i+1), zero if either is land.
        # cy[j,i] = face coeff between (j,i) and (j+1,i), zero if either is land.
        cx, cy = _interpolate_coeff_to_faces(cur_coeff, cur_mask)

        # Precompute 1/diag(A) for the Jacobi smoother at this level
        inv_diag = _compute_inv_diagonal(cx, cy, cur_mask, cur_dx, cur_dy, lambda_)

        # Store as frozen JAX arrays
        levels.append(
            MultigridLevel(
                mask=jnp.array(cur_mask),
                cx=jnp.array(cx),
                cy=jnp.array(cy),
                dx=cur_dx,
                dy=cur_dy,
                lambda_=float(lambda_),
                inv_diagonal=jnp.array(inv_diag),
            )
        )

        # Coarsen mask and coefficient for the next (coarser) level
        if lev < n_levels - 1:
            next_mask = _restrict_mask(cur_mask)
            next_coeff = _restrict_coeff(cur_coeff, cur_mask, next_mask)
            cur_mask = next_mask
            cur_coeff = next_coeff
            cur_dx *= 2.0  # grid spacing doubles at each coarser level
            cur_dy *= 2.0

    return MultigridSolver(
        levels=tuple(levels),
        n_levels=n_levels,
        n_pre=n_pre,
        n_post=n_post,
        n_coarse=n_coarse,
        omega=omega,
        n_cycles=n_cycles,
    )
