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

Hierarchy
---------
Every grid shape coarsens, odd or even: a dimension of ``n`` points becomes
``(n + 1) // 2``, coarse point ``J`` sitting on fine point ``2J`` (so a
standard ``N_interior + 2`` grid gets the same depth as ``N_interior``).
Prolongation is bilinear interpolation between those nested points,
restriction is its transpose (full weighting), and every coarse operator is
the Galerkin product ``A_c = R A P``; the coarsest level is solved exactly
with an inverse precomputed at build time.  With equal pre- and
post-smoothing the V-cycle is a symmetric linear operator.

Public API
----------
``build_multigrid_solver``
    One-time offline precomputation of the multigrid level hierarchy
    (masks, operator stencils, transfer weights, the bottom inverse).
    Returns a ``MultigridSolver``.
``MultigridSolver``
    Stateless ``equinox.Module`` with three solve modes
    (implicit / one-step / unrolled differentiation).
See also :func:`~finitevolx._src.solvers.preconditioners.make_multigrid_preconditioner`
which wraps a single V-cycle as a preconditioner closure compatible with
:func:`~finitevolx._src.solvers.iterative.solve_cg`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import numpy as np
from scipy import linalg, sparse
from scipy.sparse import csgraph

from finitevolx._src.mask import Mask2D
from finitevolx._src.solvers.iterative import _project_zero_mean, _wet_components

# Stencil offsets (dj, di): the finest operator is 5-point, the Galerkin
# coarse operators are 9-point.
_OFFSETS_5 = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))
_OFFSETS_9 = (
    (0, 0),
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)

# Largest coarsest level (wet points) the dense bottom inverse is built for.
_MAX_BOTTOM = 4096

# ---------------------------------------------------------------------------
# Internal helpers — grid hierarchy construction (numpy, offline)
# ---------------------------------------------------------------------------


def _coarse_size(n: int) -> int:
    """Number of coarse points over ``n`` fine points: ``(n + 1) // 2``.

    Coarse point ``J`` sits on fine point ``2J``; an odd fine point ``2J+1``
    lies between coarse points ``J`` and ``J+1``.
    """
    return (n + 1) // 2


def _compute_n_levels(ny: int, nx: int, min_coarse: int = 4) -> int:
    """Auto-detect the number of multigrid levels by repeated coarsening.

    Starting from the fine grid ``(ny, nx)``, coarsen both dimensions with
    ``n -> (n + 1) // 2`` while neither would drop below *min_coarse*.  Odd
    sizes coarsen too, so a standard ``N_interior + 2`` grid gets the same
    depth as ``N_interior``: 34 -> 17 -> 9 -> 5 (4 levels), 130 -> ... -> 5
    (6 levels), 64 -> 32 -> 16 -> 8 -> 4 (5 levels).

    Parameters
    ----------
    ny, nx : int
        Fine-grid dimensions.
    min_coarse : int
        Minimum allowed coarse-grid dimension.  Default: 4.

    Returns
    -------
    int
        Number of levels (>= 1).  Level 0 is finest, level L-1 is coarsest.
    """
    n = 1
    while _coarse_size(ny) >= min_coarse and _coarse_size(nx) >= min_coarse:
        ny, nx = _coarse_size(ny), _coarse_size(nx)
        n += 1
    return n


def _prolongation_1d(n_fine: int) -> sparse.csr_matrix:
    """1-D linear interpolation between nested points, ``[n_fine, n_coarse]``.

    ::

        v_f[2J]   = v_c[J]
        v_f[2J+1] = (v_c[J] + v_c[J+1]) / 2      (v_c[n_coarse] = 0)
    """
    n_coarse = _coarse_size(n_fine)
    f = np.arange(n_fine)
    rows = [f, f[1::2]]
    cols = [f // 2, f[1::2] // 2 + 1]
    vals = [np.where(f % 2 == 0, 1.0, 0.5), np.full(f[1::2].size, 0.5)]
    rows, cols, vals = (np.concatenate(a) for a in (rows, cols, vals))
    keep = cols < n_coarse  # the last odd point of an even n has no right partner
    return sparse.csr_matrix(
        (vals[keep], (rows[keep], cols[keep])), shape=(n_fine, n_coarse)
    )


def _coarse_mask(
    mask_f: np.ndarray, p0: sparse.csr_matrix, shape_c: tuple[int, int]
) -> np.ndarray:
    """Coarse wet mask such that the masked prolongation has full column rank.

    Mask-coarsening rule:

    1. Coarse point ``(J, I)`` is wet if its coincident fine point
       ``(2J, 2I)`` is wet.  These columns of ``P0`` are independent: fine
       point ``(2J, 2I)`` is interpolated from ``(J, I)`` alone.
    2. A coarse point whose coincident fine point is dry but whose
       interpolation support (fine points ``2J-1 .. 2J+1`` by
       ``2I-1 .. 2I+1``) touches a wet fine point is a candidate; it is kept
       only if its column (on the wet fine points) is linearly independent of
       the other kept candidates.  Coarse points next to a coast thus extend
       the coarse space, while a one-cell lake or channel at an odd index,
       whose up to four candidates have identical columns, keeps one of them.

    No candidate column can depend on a rule-1 column (those own a fine
    point no other column touches), so the independence test runs on the
    candidates alone.  A candidate that is the only remaining candidate on
    some wet fine point is independent of all the others (its coefficient
    in any linear dependency is zero); those are kept and peeled off,
    repeatedly — on a plain coast every candidate goes this way.  Only the
    candidates left after peeling (one-cell lakes and channels at odd
    indices) get a column-pivoted QR, per cluster of candidates that share
    fine points.  With full column rank, ``A_c = R A P`` is non-singular for
    Dirichlet and has only the constants as null space for Neumann at
    ``lambda = 0``.

    Parameters
    ----------
    mask_f : ndarray, shape (Ny, Nx)
        Fine mask (1 = wet).
    p0 : scipy.sparse.csr_matrix, shape (Ny*Nx, Ny_c*Nx_c)
        Unmasked bilinear prolongation.
    shape_c : (int, int)
        Coarse shape.

    Returns
    -------
    ndarray, shape_c
        Coarse mask (1 = wet).
    """
    wet_f = mask_f.ravel() > 0
    support = p0.T @ wet_f.astype(np.float64)  # [Nc], wet weight in support
    keep = (mask_f[::2, ::2] > 0).ravel()  # rule 1: coincident fine point wet
    cand = np.flatnonzero((support > 0) & ~keep)
    if cand.size:
        sub = p0[wet_f][:, cand].tocsc()  # [n_wet_f, n_cand]
        touch = (sub != 0).astype(np.int64).tocsr()  # [n_wet_f, n_cand]
        # peel: a candidate alone on some wet fine point is independent
        active = np.ones(cand.size, dtype=bool)
        while active.any():
            count = touch @ active.astype(np.int64)  # active candidates per row
            rows = touch[count == 1].tocoo()  # rows with a single active one
            owners = np.zeros(cand.size, dtype=bool)
            owners[rows.col[active[rows.col]]] = True
            if not owners.any():
                break
            keep[cand[owners]] = True
            active &= ~owners
        core = np.flatnonzero(active)
        if core.size:
            sub = sub[:, core]
            # clusters of the remaining candidates that share a wet fine point
            n_clusters, label = csgraph.connected_components(
                (sub.T @ sub) != 0, directed=False
            )
            for k in range(n_clusters):
                cols = np.flatnonzero(label == k)
                block = sub[:, cols]
                dense = block[np.unique(block.indices)].toarray()  # [rows, cols]
                r, piv = linalg.qr(dense, mode="r", pivoting=True)
                d = np.abs(np.diag(r))
                # exact dependencies give |r_kk| ~ 1e-16; the weights are O(1)
                rank = int(np.sum(d > 1e-10 * d[0]))
                keep[cand[core[cols[piv[:rank]]]]] = True
    return keep.reshape(shape_c).astype(np.float64)


def _transfer_operators(
    mask_f: np.ndarray, bc: str
) -> tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    """Coarse mask, prolongation row scale and the sparse prolongation.

    ``P0 = P0_y (x) P0_x`` is bilinear interpolation between nested points
    (:func:`_prolongation_1d`).  The coarse mask follows
    :func:`_coarse_mask` (every wet fine point is interpolated from at least
    one wet coarse point, and ``P`` has full column rank); the Galerkin
    coarse operator sees the true fine geometry through ``P``.  The
    prolongation is::

        P = diag(s) P0 diag(m_c),   s = m_f                 (bc="dirichlet")
                                    s = m_f / (P0 m_c)      (bc="neumann")

    Dirichlet interpolates with zero at dry coarse points (the wall value);
    Neumann renormalises over the wet coarse points, so constants are
    interpolated exactly (no-flux wall, and the null space of the
    lambda = 0 operator is kept on every level).

    Parameters
    ----------
    mask_f : ndarray, shape (Ny, Nx)
        Fine mask (1 = wet).
    bc : {"dirichlet", "neumann"}
        Wall condition.

    Returns
    -------
    mask_c : ndarray, shape ((Ny+1)//2, (Nx+1)//2)
        Coarse mask.
    scale : ndarray, shape (Ny, Nx)
        Row scale ``s`` of ``P``.
    P : scipy.sparse.csr_matrix, shape (Ny*Nx, Ny_c*Nx_c)
        The prolongation, row-major flattening.
    """
    ny, nx = mask_f.shape
    p0 = sparse.kron(_prolongation_1d(ny), _prolongation_1d(nx), format="csr")
    shape_c = (_coarse_size(ny), _coarse_size(nx))
    mask_c = _coarse_mask(mask_f, p0, shape_c)
    if bc == "neumann":
        weight = (p0 @ mask_c.ravel()).reshape(ny, nx)  # sum of wet weights
        scale = np.where(weight > 0, mask_f / np.where(weight > 0, weight, 1.0), 0.0)
    else:
        scale = mask_f.copy()
    p = sparse.diags(scale.ravel()) @ p0 @ sparse.diags(mask_c.ravel())
    return mask_c, scale, p.tocsr()


def _fine_stencil(
    coeff: np.ndarray,
    mask: np.ndarray,
    dx: float,
    dy: float,
    lambda_: float,
    bc: str,
) -> np.ndarray:
    """5-point stencil of ``div(c grad u) - lambda u`` on the finest grid.

    Face coefficients are the mean of the two cell values on a wet–wet face.
    A wet–dry face (land, or outside the array) keeps the wet cell's ``c``
    and sees ``u = 0`` beyond it for ``bc="dirichlet"`` (only the diagonal
    changes), and carries no flux for ``bc="neumann"``::

        (Au)[j,i] = sum_f c_f (u_nb - u[j,i]) / h_f**2 - lambda u[j,i]

    With ``c = 1`` and Dirichlet this is :func:`masked_laplacian` with dry
    neighbours zero (no periodic wrap).

    Parameters
    ----------
    coeff : ndarray, shape (Ny, Nx)
        Cell coefficient ``c``, zero on dry cells.
    mask : ndarray, shape (Ny, Nx)
        Cell mask (1 = wet).
    dx, dy : float
        Grid spacings.
    lambda_ : float
        Helmholtz parameter.
    bc : {"dirichlet", "neumann"}
        Wall condition.

    Returns
    -------
    ndarray, shape (5, Ny, Nx)
        Coefficients for :data:`_OFFSETS_5` (centre, west, east, south,
        north), zero on dry rows and towards dry neighbours.
    """
    mp = np.pad(mask, 1)  # [Ny+2, Nx+2], zero ring = dry
    cp = np.pad(coeff * mask, 1)
    c = cp[1:-1, 1:-1]
    stencil = np.zeros((5, *mask.shape))
    centre = -lambda_ * mask
    for k, (dj, di) in enumerate(_OFFSETS_5[1:], start=1):
        h2 = dx**2 if dj == 0 else dy**2
        m_nb = mp[1 + dj : mp.shape[0] - 1 + dj, 1 + di : mp.shape[1] - 1 + di]
        c_nb = cp[1 + dj : cp.shape[0] - 1 + dj, 1 + di : cp.shape[1] - 1 + di]
        # wet-wet face: c_f = (c[j,i] + c_nb) / 2, couples to the neighbour
        c_face = 0.5 * (c + c_nb) * mask * m_nb
        stencil[k] = c_face / h2
        centre = centre - c_face / h2
        if bc == "dirichlet":
            # wet-dry face: c_f = c[j,i], neighbour value 0 -> diagonal only
            centre = centre - c * mask * (1.0 - m_nb) / h2
    stencil[0] = centre
    return stencil


def _stencil_to_sparse(
    stencil: np.ndarray, offsets: tuple[tuple[int, int], ...]
) -> sparse.csr_matrix:
    """Sparse matrix of a stencil operator (zero-padded beyond the array)."""
    _, ny, nx = stencil.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, vals = [], [], []
    for k, (dj, di) in enumerate(offsets):
        j0, j1 = max(0, -dj), min(ny, ny - dj)
        i0, i1 = max(0, -di), min(nx, nx - di)
        rows.append(idx[j0:j1, i0:i1].ravel())
        cols.append(idx[j0 + dj : j1 + dj, i0 + di : i1 + di].ravel())
        vals.append(stencil[k, j0:j1, i0:i1].ravel())
    rows, cols, vals = (np.concatenate(a) for a in (rows, cols, vals))
    nz = vals != 0
    return sparse.csr_matrix((vals[nz], (rows[nz], cols[nz])), shape=(ny * nx, ny * nx))


def _sparse_to_stencil(a: sparse.csr_matrix, shape: tuple[int, int]) -> np.ndarray:
    """9-point stencil of a sparse operator on ``shape`` (:data:`_OFFSETS_9`)."""
    ny, nx = shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    a = a.tocsr()
    stencil = np.zeros((9, ny, nx))
    for k, (dj, di) in enumerate(_OFFSETS_9):
        j0, j1 = max(0, -dj), min(ny, ny - dj)
        i0, i1 = max(0, -di), min(nx, nx - di)
        rows = idx[j0:j1, i0:i1].ravel()
        cols = idx[j0 + dj : j1 + dj, i0 + di : i1 + di].ravel()
        stencil[k, j0:j1, i0:i1] = np.asarray(a[rows, cols]).reshape(j1 - j0, i1 - i0)
    # the Galerkin product of a 9-point (or 5-point) operator with nested
    # bilinear transfers is 9-point: nothing may fall outside the stencil
    total = float(abs(a).sum())
    assert abs(float(np.abs(stencil).sum()) - total) <= 1e-12 * max(total, 1.0)
    return stencil


def _symmetric_inverse(a: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric matrix on its range (eigendecomposition).

    Eigenvalues below ``1e-10 * max|eig|`` are dropped, so the result is the
    exact inverse of a non-singular operator and the pseudo-inverse of the
    singular Neumann ``lambda = 0`` operator (its round-off-sized null
    eigenvalues would otherwise give entries ~1e13).  The cut-off is
    relative, so it does not depend on the units of the operator.
    """
    if a.size == 0:
        return a.copy()
    w, v = np.linalg.eigh(a)
    ok = np.abs(w) > 1e-10 * np.abs(w).max()
    return (v[:, ok] / w[ok]) @ v[:, ok].T


def _inverse_diagonal(diag: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``1 / diag(A)`` on wet points for the Jacobi smoother, else 0.

    A zero diagonal (a Neumann lake entirely inside one coarse point at
    ``lambda = 0``) gets no Jacobi update; the threshold is relative, so it
    does not depend on the units of the operator.
    """
    big = float(np.abs(diag).max()) if diag.size else 0.0
    ok = (np.abs(diag) > 1e-12 * big) & (mask > 0)
    return np.where(ok, 1.0 / np.where(ok, diag, 1.0), 0.0)


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
        Domain mask (1 = fluid, 0 = land) at this level's resolution.
    stencil : Float[Array, "K Ny Nx"]
        Operator coefficients for each offset in ``offsets``::

            (A u)[j, i] = sum_k stencil[k, j, i] * u[j + dj_k, i + di_k]

        (``u`` zero beyond the array and on dry points).  The finest level
        is the 5-point discretisation, coarser levels the 9-point Galerkin
        products ``R A P``.
    offsets : tuple of (int, int)
        The ``(dj, di)`` of each stencil entry (static).
    inv_diagonal : Float[Array, "Ny Nx"]
        Precomputed ``1 / diag(A)`` for the weighted Jacobi smoother.
        Zero outside the mask.
    prolong_scale : Float[Array, "Ny Nx"] or None
        Row scale ``s`` of the prolongation from the next coarser level,
        ``P = diag(s) P0 diag(m_c)`` (see :func:`_prolongate`); ``None`` on
        the coarsest level.
    dx, dy : float
        Grid spacing at this level (doubles at each coarser level).
    lambda_ : float
        Helmholtz parameter (same on all levels).
    """

    mask: Float[Array, "Ny Nx"]
    stencil: Float[Array, "K Ny Nx"]
    inv_diagonal: Float[Array, "Ny Nx"]
    prolong_scale: Float[Array, "Ny Nx"] | None
    dx: float
    dy: float
    lambda_: float = eqx.field(static=True)
    offsets: tuple[tuple[int, int], ...] = eqx.field(static=True, default=_OFFSETS_5)


# ---------------------------------------------------------------------------
# Core JAX operators (JIT-compiled, run on accelerator)
# ---------------------------------------------------------------------------


def _apply_operator(
    u: Float[Array, "Ny Nx"],
    level: MultigridLevel,
) -> Float[Array, "Ny Nx"]:
    r"""Apply the level's Helmholtz operator: A u.

    Stencil form, with ``u`` zeroed on dry points and zero-padded beyond
    the array (nothing wraps across opposite edges)::

        (A u)[j, i] = sum_k stencil[k, j, i] * u[j + dj_k, i + di_k]

    On the finest level this is the 5-point finite-volume stencil::

        (Au)[j,i] = sum over the 4 faces f of c_f (u_nb - u[j,i]) / h_f^2
                    - lambda u[j,i]

    with ``u_nb = 0`` beyond a wet–dry face for ``bc="dirichlet"`` and no
    flux through it for ``bc="neumann"`` (see :func:`_fine_stencil`).  With
    ``c = 1`` and Dirichlet it equals :func:`masked_laplacian` with dry
    neighbours zero.

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
    ny, nx = u.shape
    # the hierarchy is float64; compute in the dtype of u (contract 7)
    mask = level.mask.astype(u.dtype)
    stencil = level.stencil.astype(u.dtype)  # [K, Ny, Nx]
    up = jnp.pad(u * mask, 1)  # [Ny+2, Nx+2], zero beyond the array
    out = jnp.zeros_like(up[1:-1, 1:-1])
    for k, (dj, di) in enumerate(level.offsets):
        # term k: stencil[k, j, i] * u[j + dj, i + di]
        out = out + stencil[k] * up[1 + dj : 1 + dj + ny, 1 + di : 1 + di + nx]
    return out * mask


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

    The weight ``omega < 1`` under-relaxes the update so the
    highest-frequency (checkerboard) modes are damped too; ``omega = 0.8``
    (the default) damps the whole upper half of the 5-point spectrum by at
    least 3/5 per sweep.

    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        Current solution iterate.
    rhs : Float[Array, "Ny Nx"]
        Right-hand side of ``A u = rhs``.
    level : MultigridLevel
        Operator data (includes ``inv_diagonal`` = ``D^{-1}``).
    omega : float
        Relaxation weight (default 0.8).
    n_iters : int
        Number of Jacobi sweeps.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Smoothed solution, zeroed outside the mask.
    """

    inv_diag = level.inv_diagonal.astype(u.dtype)  # hierarchy is float64
    mask = level.mask.astype(u.dtype)

    def _step(_, u_cur: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        r = rhs - _apply_operator(u_cur, level)  # residual: r = f - A u
        return (u_cur + omega * inv_diag * r) * mask

    return jax.lax.fori_loop(0, n_iters, _step, u)


def _interp_axis(v: Array, n_fine: int, axis: int) -> Array:
    """Linear interpolation between nested points along *axis* (``P0_1d v``).

    ::

        out[2J]   = v[J]
        out[2J+1] = (v[J] + v[J+1]) / 2      (v[n_coarse] = 0)

    then cropped to *n_fine* points along *axis*.
    """
    v = jnp.moveaxis(v, axis, 0)  # [n_c, ...]
    vp = jnp.concatenate([v, jnp.zeros_like(v[:1])], axis=0)  # [n_c+1, ...]
    odd = 0.5 * (vp[:-1] + vp[1:])  # out[2J+1], [n_c, ...]
    out = jnp.stack([v, odd], axis=1).reshape(2 * v.shape[0], *v.shape[1:])
    return jnp.moveaxis(out[:n_fine], 0, axis)


def _interp_axis_t(w: Array, n_coarse: int, axis: int) -> Array:
    """Transpose of :func:`_interp_axis` along *axis* (``P0_1dᵀ w``).

    ::

        out[J] = w[2J] + (w[2J+1] + w[2J-1]) / 2      (w beyond the array = 0)
    """
    w = jnp.moveaxis(w, axis, 0)  # [n_f, ...]
    pad = 2 * n_coarse - w.shape[0]  # 0 or 1 trailing zero
    w = jnp.concatenate([w, jnp.zeros((pad, *w.shape[1:]), w.dtype)], axis=0)
    even, odd = w[0::2], w[1::2]  # w[2J], w[2J+1]; [n_c, ...] each
    odd_prev = jnp.concatenate([jnp.zeros_like(odd[:1]), odd[:-1]], axis=0)
    return jnp.moveaxis(even + 0.5 * (odd + odd_prev), 0, axis)


def _prolongate(
    v_coarse: Float[Array, "Ny_c Nx_c"],
    mask_coarse: Float[Array, "Ny_c Nx_c"],
    scale: Float[Array, "Ny Nx"],
) -> Float[Array, "Ny Nx"]:
    r"""Bilinear prolongation between nested points (coarse -> fine).

    Coarse point ``(J, I)`` sits on fine point ``(2J, 2I)``; the fine points
    in between are interpolated linearly in each direction::

        v_f[2J,   2I  ] = v_c[J, I]
        v_f[2J,   2I+1] = (v_c[J, I] + v_c[J, I+1]) / 2
        v_f[2J+1, 2I  ] = (v_c[J, I] + v_c[J+1, I]) / 2
        v_f[2J+1, 2I+1] = (v_c[J, I] + v_c[J, I+1] + v_c[J+1, I] + v_c[J+1, I+1]) / 4

    with dry coarse values taken as zero, then scaled row-wise:
    ``P = diag(scale) P0 diag(m_c)`` (``scale = m_f`` for Dirichlet,
    ``m_f / (P0 m_c)`` for Neumann, see :func:`_transfer_operators`).

    Parameters
    ----------
    v_coarse : Float[Array, "Ny_c Nx_c"]
        Coarse-grid field, ``Ny_c = (Ny + 1) // 2``.
    mask_coarse : Float[Array, "Ny_c Nx_c"]
        Coarse-grid mask.
    scale : Float[Array, "Ny Nx"]
        Row scale of ``P`` (``MultigridLevel.prolong_scale`` of the fine
        level); its shape sets the fine shape.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Interpolated fine-grid field, zero outside the fine mask.
    """
    ny, nx = scale.shape
    dtype = v_coarse.dtype
    v = v_coarse * mask_coarse.astype(dtype)  # [Ny_c, Nx_c]
    v = _interp_axis(v, ny, axis=0)  # [Ny, Nx_c]
    v = _interp_axis(v, nx, axis=1)  # [Ny, Nx]
    return v * scale.astype(dtype)


def _restrict(
    v: Float[Array, "Ny Nx"],
    scale: Float[Array, "Ny Nx"],
    mask_coarse: Float[Array, "Ny_c Nx_c"],
) -> Float[Array, "Ny_c Nx_c"]:
    """Full-weighting restriction, the scaled transpose of :func:`_prolongate`.

    ``R = Pᵀ / 4``, so ``<R v, w>_c = <v, P w>_f / 4`` exactly.  Away from
    walls (``scale = 1``) it is the 9-point full-weighting average::

        v_c[J, I] = (4 v[2J, 2I] + 2 (v[2J, 2I±1] + v[2J±1, 2I])
                     + v[2J±1, 2I±1]) / 16

    The transpose (rather than an independent average) makes the V-cycle
    symmetric and the Galerkin coarse operator ``R A P`` consistent.

    Parameters
    ----------
    v : Float[Array, "Ny Nx"]
        Fine-grid field to restrict (e.g. the residual).
    scale : Float[Array, "Ny Nx"]
        Row scale of ``P`` (``MultigridLevel.prolong_scale``).
    mask_coarse : Float[Array, "Ny_c Nx_c"]
        Coarse-grid mask.

    Returns
    -------
    Float[Array, "Ny_c Nx_c"]
        Coarse field, zero outside the coarse mask.
    """
    ny_c, nx_c = mask_coarse.shape
    w = v * scale.astype(v.dtype)  # [Ny, Nx]
    w = _interp_axis_t(w, nx_c, axis=1)  # [Ny, Nx_c]
    w = _interp_axis_t(w, ny_c, axis=0)  # [Ny_c, Nx_c]
    return 0.25 * w * mask_coarse.astype(v.dtype)


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
        Level 2 (coarse)  ..........*--exact solve--*..........

    At each level:

    1. **Pre-smooth** (nu_1 weighted Jacobi iterations): damp
       high-frequency error.
    2. **Compute residual**: ``r = rhs - A u``.
    3. **Restrict** the residual to the coarse grid (full weighting,
       ``R = Pᵀ / 4``).
    4. **Recurse**: solve for the error on the coarse grid, whose operator
       is the Galerkin product ``A_c = R A P``.
    5. **Prolongate** the coarse correction back to the fine grid
       (bilinear interpolation ``P``).
    6. **Post-smooth** (nu_2 weighted Jacobi iterations): damp any
       high-frequency error introduced by the prolongation.

    The coarsest level is solved exactly (an inverse precomputed at build
    time).  With ``n_pre == n_post`` one V-cycle from a zero guess,
    ``r -> v_cycle(0, r)``, is a symmetric linear operator, so it is a valid
    CG preconditioner.

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

    Attributes
    ----------
    levels : tuple of MultigridLevel
        Precomputed level data, finest (index 0) to coarsest (index L-1).
    n_levels : int
        Number of multigrid levels.
    bottom_inverse : Float[Array, "n_wet n_wet"]
        Inverse of the coarsest operator on its wet points (the
        pseudo-inverse, with a relative eigenvalue cut-off, for the singular
        Neumann ``lambda = 0`` problem).
    bottom_index : Int[Array, " n_wet"]
        Flat (row-major) indices of the coarsest level's wet points.
    n_pre, n_post : int
        Pre- and post-smoothing iterations (weighted Jacobi).  Keep them
        equal for a symmetric V-cycle.
    omega : float
        Jacobi relaxation weight.
    n_cycles : int
        Number of V-cycles per solve.
    bc : {"dirichlet", "neumann"}
        Wall condition on the wet–dry faces the hierarchy was built with
        (see :func:`build_multigrid_solver`).
    null_labels, null_counts : Array or None
        Connected-component labels and wet counts of the finest mask, set
        only for ``bc="neumann"`` with ``lambda_ = 0``, where the operator
        is singular; the solve then returns the zero-mean solution on each
        component (and projects ``rhs`` onto the solvable, zero-mean part).
    """

    levels: tuple[MultigridLevel, ...]
    n_levels: int = eqx.field(static=True)
    bottom_inverse: Float[Array, "n_wet n_wet"]
    bottom_index: Int[Array, " n_wet"]
    n_pre: int = eqx.field(static=True, default=3)
    n_post: int = eqx.field(static=True, default=3)
    omega: float = eqx.field(static=True, default=0.8)
    n_cycles: int = eqx.field(static=True, default=5)
    bc: str = eqx.field(static=True, default="dirichlet")
    null_labels: Array | None = None
    null_counts: Array | None = None

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
        convergence rate (about 0.07 per cycle on the default settings,
        measured on 34² to 258² basins; the one-step gradient is then within
        5 % of the implicit one).

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
                return A_c^{-1} rhs                # exact bottom solve

            u = jacobi(u, rhs, n_pre)             # 1. pre-smooth
            r = rhs - A(u)                        # 2. compute residual
            r_c = restrict(r)                     # 3. R = P^T / 4
            e_c = v_cycle(0, r_c, level+1)        # 4. recurse on A_c = R A P
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
        # Singular Neumann problem (lambda = 0): solve in the zero-mean gauge.
        # Projecting the input and output of the finest cycle keeps the
        # V-cycle linear and symmetric (P V P with P an orthogonal projection).
        if level_idx == 0 and self.null_labels is not None:
            rhs = self._project(rhs)
            return self._project(self._v_cycle_from(u, rhs, level_idx))
        return self._v_cycle_from(u, rhs, level_idx)

    def _project(self, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Zero-mean projection on each wet component (Neumann, lambda = 0)."""
        assert self.null_labels is not None and self.null_counts is not None
        return _project_zero_mean(
            u, self.levels[0].mask, self.null_labels, self.null_counts
        )

    def _bottom_solve(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Exact solve on the coarsest level: ``u = A_c^{-1} rhs`` on wet points."""
        flat = rhs.ravel()  # [Ny_c*Nx_c]
        idx = self.bottom_index  # [n_wet], distinct flat indices
        u_wet = self.bottom_inverse.astype(rhs.dtype) @ flat[idx]  # [n_wet]
        # unique_indices: the scatter is then transposable (jax.linear_transpose)
        u = jnp.zeros_like(flat).at[idx].set(u_wet, unique_indices=True)
        return u.reshape(rhs.shape)

    def _v_cycle_from(
        self,
        u: Float[Array, "Ny Nx"],
        rhs: Float[Array, "Ny Nx"],
        level_idx: int,
    ) -> Float[Array, "Ny Nx"]:
        """One V-cycle at *level_idx* without the zero-mean projection."""
        level = self.levels[level_idx]

        # --- Coarsest level: exact solve (the guess u is not needed) ---
        if level_idx == self.n_levels - 1:
            return self._bottom_solve(rhs)

        # 1. Pre-smooth: damp high-frequency error on the current grid
        u = _weighted_jacobi(u, rhs, level, self.omega, self.n_pre)

        # 2. Compute residual: r = f - A(u)
        #    After smoothing, r contains mostly low-frequency error that
        #    the smoother cannot resolve at this grid resolution.
        mask = level.mask.astype(rhs.dtype)  # the hierarchy is float64
        r = (rhs - _apply_operator(u, level)) * mask

        # 3. Restrict the residual to the coarse grid: r_c = P^T r / 4
        coarse_level = self.levels[level_idx + 1]
        assert level.prolong_scale is not None
        r_coarse = _restrict(r, level.prolong_scale, coarse_level.mask)

        # 4. Recurse: solve A_c e_c = r_c (A_c = R A P) for the error,
        #    starting from zero.
        e_coarse = self.v_cycle(jnp.zeros_like(r_coarse), r_coarse, level_idx + 1)

        # 5. Prolongate the coarse correction and add it: u <- u + P e_c
        e_fine = _prolongate(e_coarse, coarse_level.mask, level.prolong_scale)
        u = (u + e_fine) * mask

        # 6. Post-smooth: clean up high-frequency error introduced by
        #    the coarse-to-fine interpolation
        return _weighted_jacobi(u, rhs, level, self.omega, self.n_post)

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
    mask: np.ndarray | Float[Array, "Ny Nx"] | Mask2D,
    dx: float,
    dy: float,
    lambda_: float = 0.0,
    coeff: np.ndarray | Float[Array, "Ny Nx"] | None = None,
    n_levels: int | None = None,
    n_pre: int = 3,
    n_post: int = 3,
    omega: float = 0.8,
    n_cycles: int = 5,
    bc: Literal["dirichlet", "neumann"] = "dirichlet",
) -> MultigridSolver:
    r"""Build a multigrid solver with precomputed level hierarchies.

    This is an **offline** function (runs once on CPU with NumPy / SciPy)
    that constructs the entire multigrid hierarchy:

    1. **Finest operator**: the 5-point stencil of ``div(c grad u) -
       lambda u``, face coefficients the mean of the two cell values, wet–dry
       faces set by ``bc`` (below).
    2. **Coarsening**: each dimension of ``n`` points becomes
       ``(n + 1) // 2``, coarse point ``J`` on fine point ``2J`` — odd and
       even sizes alike, so a standard ``N + 2`` grid is as deep as ``N``.
       A coarse point is wet if its coincident fine point is wet; a point
       whose interpolation support (fine points ``2J-1 .. 2J+1`` in each
       direction) only touches wet fine points is added when its
       prolongation column is linearly independent of the others, so
       narrow channels keep a coarse unknown and ``P`` keeps full column
       rank (:func:`_coarse_mask`).
    3. **Transfers**: bilinear prolongation ``P`` between the nested points
       (zero at dry coarse points for Dirichlet; renormalised over the wet
       ones for Neumann) and full-weighting restriction ``R = Pᵀ / 4``.
    4. **Galerkin coarse operators** ``A_c = R A P`` (9-point stencils),
       which follow the fine geometry and coefficient exactly.
    5. **Exact bottom solve**: the inverse of the coarsest operator on its
       wet points (at most a few dozen unknowns for square grids).

    Grid hierarchy example (34x34, a 32-cell interior plus ghost ring)::

        Level 0:  34 x 34   (dx,    dy)     <- finest (solve here)
        Level 1:  17 x 17   (2*dx,  2*dy)
        Level 2:   9 x  9   (4*dx,  4*dy)
        Level 3:   5 x  5   (8*dx,  8*dy)   <- coarsest (exact solve)

    The returned ``MultigridSolver`` is an immutable ``equinox.Module``
    with frozen JAX arrays.  All subsequent calls (forward solves,
    gradients, JIT compilation) use the precomputed hierarchy.

    Boundary condition
    ~~~~~~~~~~~~~~~~~~
    ``bc`` sets what happens on every face between a wet cell and a dry
    one (land, or outside the array):

    * ``"dirichlet"`` (default): ``u = 0`` on the dry neighbour, the face
      keeps its coefficient.  With ``coeff=None`` the finest-level operator
      is exactly :func:`~finitevolx.masked_laplacian` with dry neighbours
      zero — the streamfunction problem, ``psi = 0`` on the wall / coast
      X nodes of ``Mask2D.xy_corner_strict``.  Non-singular for every
      ``lambda_ >= 0``.
    * ``"neumann"``: zero flux through wet–dry faces — the pressure /
      free-surface problem at T points (pass ``mask.h``).  At
      ``lambda_ = 0`` the operator is singular (a constant per connected
      wet component); the solver then projects ``rhs`` and the solution
      onto zero mean on each component.

    Parameters
    ----------
    mask : array, shape (Ny, Nx), or Mask2D
        Domain mask (1 = fluid, 0 = land).  ``None`` is *not* accepted;
        pass ``np.ones((Ny, Nx))`` for a rectangular domain.
        When a :class:`Mask2D` is passed, the ψ mask
        ``xy_corner_strict`` is extracted (pair it with the default
        ``bc="dirichlet"``); for a T-point problem pass ``mask.h``.
    dx, dy : float
        Fine-grid spacings (metres or non-dimensional).
    lambda_ : float
        Helmholtz parameter (>= 0).  Use 0.0 for pure Poisson (Laplacian
        only).  For QG PV inversion (``bc="dirichlet"``),
        ``lambda_ = 1 / Rd**2``.
    coeff : array, shape (Ny, Nx), or None
        Spatially varying coefficient ``c(x, y)`` at cell centres.
        ``None`` -> constant coefficient = 1 everywhere (reduces to the
        standard constant-coefficient Helmholtz operator).
    n_levels : int or None
        Number of multigrid levels.  ``None`` -> auto-detect by coarsening
        while both dimensions stay >= 4 (see :func:`_compute_n_levels`).
        Any shape works with any ``n_levels >= 1``; ``1`` is a direct
        solve.
    n_pre, n_post : int
        Number of pre- and post-smoothing Jacobi iterations per V-cycle.
        Keep them equal: the V-cycle is then a symmetric operator (needed
        when it preconditions CG).  Default: 3 each.
    omega : float
        Jacobi relaxation weight (0 < omega < 1).  Default 0.8, the
        weight that damps the high-frequency half of the 5-point spectrum
        best (by at least 3/5 per sweep).
    n_cycles : int
        Number of V-cycles applied per solve.  Each cycle reduces the
        residual by 0.06-0.07 on the default settings (measured on 34² to
        258² circular basins, independent of the grid size), so 5 cycles
        reduce it by about 3e-7.  An island is slower: 0.14 / 0.17 per
        cycle for ``bc="dirichlet"`` in a circular basin with an island at
        130² / 258² (5 cycles: 7e-5 / 4e-4).  One-cell channels and islands
        are slower still for ``bc="neumann"`` (0.1-0.5 per cycle, measured on
        34² masks); use the V-cycle as a CG preconditioner there.
        Default: 5.
    bc : {"dirichlet", "neumann"}
        Condition on wet–dry faces (see above).  Default: ``"dirichlet"``.

    Returns
    -------
    MultigridSolver
        Ready-to-use solver (JIT-compilable ``equinox.Module``).

    Raises
    ------
    ValueError
        If ``n_levels < 1`` or ``bc`` is not ``"dirichlet"`` or
        ``"neumann"``.
    """
    if bc not in ("dirichlet", "neumann"):
        raise ValueError(f"bc must be 'dirichlet' or 'neumann'; got {bc!r}")

    # --- Extract mask as a NumPy float64 array ---
    if isinstance(mask, Mask2D):
        mask_np = np.asarray(mask.xy_corner_strict, dtype=np.float64)
    else:
        mask_np = np.asarray(mask, dtype=np.float64)

    ny, nx = mask_np.shape

    # --- Auto-detect or validate number of levels ---
    if n_levels is None:
        n_levels = _compute_n_levels(ny, nx)
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1; got {n_levels}")

    # --- Default coefficient: c(x,y) = 1 everywhere ---
    if coeff is None:
        coeff_np = np.ones_like(mask_np)
    else:
        coeff_np = np.asarray(coeff, dtype=np.float64)
    coeff_np = coeff_np * mask_np  # zero outside domain

    # --- Finest operator, then Galerkin products down the hierarchy ---
    cur_mask = mask_np
    cur_dx, cur_dy = float(dx), float(dy)
    stencil = _fine_stencil(coeff_np, mask_np, cur_dx, cur_dy, float(lambda_), bc)
    offsets = _OFFSETS_5
    a = _stencil_to_sparse(stencil, offsets)  # [Ny*Nx, Ny*Nx]

    levels: list[MultigridLevel] = []
    for lev in range(n_levels):
        scale = None
        next_a = next_mask = None
        if lev < n_levels - 1:
            # P = diag(s) P0 diag(m_c);  A_c = R A P with R = P^T / 4
            next_mask, scale, p = _transfer_operators(cur_mask, bc)
            next_a = (0.25 * (p.T @ a @ p)).tocsr()

        levels.append(
            MultigridLevel(
                mask=jnp.asarray(cur_mask),
                stencil=jnp.asarray(stencil),
                inv_diagonal=jnp.asarray(_inverse_diagonal(stencil[0], cur_mask)),
                prolong_scale=None if scale is None else jnp.asarray(scale),
                dx=cur_dx,
                dy=cur_dy,
                lambda_=float(lambda_),
                offsets=offsets,
            )
        )

        if next_a is not None and next_mask is not None:
            cur_mask, a = next_mask, next_a
            offsets = _OFFSETS_9
            stencil = _sparse_to_stencil(a, cur_mask.shape)
            cur_dx *= 2.0  # grid spacing doubles at each coarser level
            cur_dy *= 2.0

    # --- Exact bottom solve: dense inverse on the coarsest wet points ---
    wet = np.flatnonzero(cur_mask.ravel() > 0)
    if wet.size > _MAX_BOTTOM:
        raise ValueError(
            f"the coarsest level ({cur_mask.shape[0]} x {cur_mask.shape[1]}) has "
            f"{wet.size} wet points, more than the {_MAX_BOTTOM} the dense bottom "
            "solve allows; use more levels (n_levels=None picks them)"
        )
    a_wet = a.toarray()[np.ix_(wet, wet)]
    bottom_inverse = _symmetric_inverse(a_wet)

    # --- Neumann at lambda = 0: constants per wet component are the null space
    null_labels = null_counts = None
    if bc == "neumann" and lambda_ == 0.0:
        null_labels, null_counts = _wet_components(mask_np)

    return MultigridSolver(
        levels=tuple(levels),
        n_levels=n_levels,
        bottom_inverse=jnp.asarray(bottom_inverse),
        bottom_index=jnp.asarray(wet),
        n_pre=n_pre,
        n_post=n_post,
        omega=omega,
        n_cycles=n_cycles,
        bc=bc,
        null_labels=null_labels,
        null_counts=null_counts,
    )
