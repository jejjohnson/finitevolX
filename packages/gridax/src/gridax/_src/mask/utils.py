"""Mask construction primitives.

These helpers are dimension-agnostic numpy/scipy routines that the
``Mask1D`` / ``Mask2D`` / ``Mask3D`` factory methods compose to derive
staggered masks, classify cells, build sponge layers, and infer h-grid
masks from non-centre staggerings.  They are exposed as public API so
that users can build custom mask classes or one-off mask manipulations
without re-implementing the same primitives.

All routines operate on ``np.ndarray`` inputs and return ``np.ndarray``
outputs (no JAX); conversion to JAX arrays is the caller's
responsibility.

Public functions
----------------
pool_bool
    n-D average pool of a binary mask with leading-side zero padding.
    The forward stencil for deriving u/v/w/corner masks from h.
h_from_pooled
    Inverse of :func:`pool_bool` for binary inputs.  Used by the
    ``Mask*.from_<face>`` constructors to derive an h-mask from a
    staggered mask.  Non-unique → ``mode={'permissive','conservative'}``.
dilate_mask
    Binary dilation by 1 cell with an n-D *cross* structuring element
    (no diagonals).  The building block for the 4-level land/coast
    classification.
count_contiguous
    Per-cell count of consecutive wet cells along an axis in one of the
    two directions.  The basis for stencil capability.
make_sponge
    n-D linear sponge ramp: 0 at every wall, 1 in the interior.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.ndimage import binary_dilation


def pool_bool(
    arr: np.ndarray,
    kernel: tuple[int, ...],
    threshold: float,
    direction: str = "leading",
) -> np.ndarray:
    """n-D average-pool of a binary mask with one-sided zero padding.

    Computes a windowed average of ``arr`` over the given ``kernel``
    footprint and compares against ``threshold``.  One side of each
    axis is zero-padded by ``kernel[a] - 1`` cells so the output shape
    equals the input shape.  This is the forward stencil that derives
    staggered (u, v, w, xy_corner) masks from an h-grid mask on an
    Arakawa C-grid.

    The ``direction`` argument selects which side of the same-index rule
    the pooled cell sits on:

    * ``'leading'`` — pad the **start** of each axis; ``pool[j]`` then
      depends on ``arr[j - (k-1) : j + 1]``.  Encodes the *negative*
      half-step convention (south / west / SW endpoint).  Boundary
      cells at index ``0`` see ``k-1`` zero-pad neighbours.
    * ``'trailing'`` — pad the **end** of each axis; ``pool[j]`` then
      depends on ``arr[j : j + k]``.  Encodes the *positive* half-step
      convention (north / east / NE endpoint), matching the grid
      module's same-index rule (``U[j, i]`` at ``i + 1/2``,
      ``V[j, i]`` at ``j + 1/2``, ``X[j, i]`` at NE corner).  Boundary
      cells at index ``N - 1`` see ``k - 1`` zero-pad neighbours.

    Algorithm
    ---------
    1. Cast ``arr`` to float so the mean is well-defined.
    2. Pre-pad ``k - 1`` zeros on the chosen side of each axis so the
       windowed sum lines up at the appropriate boundary index.
    3. Sum every shifted view ``arr_padded[off : off + N, ...]`` over
       all kernel offsets ``off ∈ [0, k)``.
    4. Divide by ``prod(kernel)`` to get the local mean and compare
       against ``threshold``.

    Parameters
    ----------
    arr : np.ndarray
        Input mask, shape ``(..., N_a, ...)``.  Treated as float
        ``{0.0, 1.0}`` internally.
    kernel : tuple of int
        Per-axis kernel sizes; ``len(kernel)`` must equal ``arr.ndim``.
        Each ``k_a`` controls how many cells along axis *a* contribute
        to each pooled output cell.
    threshold : float
        Wet/dry threshold applied to the local mean.  For a binary
        input the mean takes values in ``{i / prod(kernel) : i = 0..K}``
        where ``K = prod(kernel)``, so common thresholds are:

        * ``> (K-1)/K`` — *all* cells must be wet (strict).
        * ``> 1/K``     — *at least one* cell must be wet (lenient).
    direction : {'leading', 'trailing'}
        Which side of the same-index rule the pooled cell sits on.
        Default ``'leading'`` (south/west/SW convention).

    Returns
    -------
    np.ndarray
        Boolean array, same shape as ``arr``.

    Examples
    --------
    Strict south-face mask under the leading (negative half-step)
    convention (kernel ``(2,)``, threshold 3/4 → "both adjacent h-cells
    wet"):

    >>> import numpy as np
    >>> h = np.array([True, True, False, True, True])
    >>> pool_bool(h, kernel=(2,), threshold=3.0 / 4.0)
    array([False,  True, False, False,  True])

    The same mask under the trailing (positive half-step) convention.
    The wet/dry pattern is the same set of physical faces, but indexed
    one cell to the left because ``pool[j]`` is now ``h[j] AND h[j+1]``
    instead of ``h[j-1] AND h[j]``:

    >>> pool_bool(h, kernel=(2,), threshold=3.0 / 4.0, direction="trailing")
    array([ True, False, False,  True, False])

    Lenient 2x2 corner mask (threshold 1/8 → "at least one of the 4
    surrounding cells wet"):

    >>> h2 = np.array([[True, True], [False, True]])
    >>> pool_bool(h2, kernel=(2, 2), threshold=1.0 / 8.0)
    array([[ True,  True],
           [ True,  True]])
    """
    if len(kernel) != arr.ndim:
        raise ValueError(
            f"kernel length {len(kernel)} does not match arr.ndim {arr.ndim}"
        )
    if direction not in ("leading", "trailing"):
        raise ValueError(
            f"direction must be 'leading' or 'trailing', got {direction!r}"
        )

    # Cast to float so the windowed sum is well-defined.  Shape: arr.shape
    arr_f = arr.astype(float)

    # One-sided zero pad: pad k-1 cells on the leading or trailing side of each
    # axis so that the windowed sum lines up at the appropriate boundary
    # output index (index 0 for leading, index N-1 for trailing).
    if direction == "leading":
        pad_widths = tuple((k - 1, 0) for k in kernel)
    else:
        pad_widths = tuple((0, k - 1) for k in kernel)
    arr_padded = np.pad(arr_f, pad_widths)

    # Sum the prod(kernel) shifted views.  For each multi-axis offset
    # `(o_0, o_1, ...)` with `0 <= o_a < k_a`, take the slice
    # `arr_padded[o_0 : o_0 + N_0, o_1 : o_1 + N_1, ...]` and add it to
    # `total`.  After all offsets are accumulated, dividing by
    # prod(kernel) gives the local mean.  Shape: arr.shape
    total = np.zeros_like(arr_f)
    for offsets in itertools.product(*(range(k) for k in kernel)):
        slices = tuple(slice(o, o + s) for o, s in zip(offsets, arr.shape, strict=True))
        total += arr_padded[slices]

    return total / float(np.prod(kernel)) > threshold


def h_from_pooled(
    pooled_mask: np.ndarray,
    kernel: tuple[int, ...],
    mode: str = "permissive",
    direction: str = "leading",
) -> np.ndarray:
    """Inverse of :func:`pool_bool` for binary inputs.

    Given a same-shape staggered mask that was forward-pooled with the
    given ``kernel`` and ``direction``, infer an underlying h-grid
    (cell-centre) mask.  The inverse mapping is non-unique, so the
    result depends on ``mode``:

    * ``'permissive'``  — h is wet iff *any* contributing pooled cell
      is wet (logical OR over the kernel footprint).  Yields the
      largest h-mask compatible with the staggered cells that are
      wet.
    * ``'conservative'`` — h is wet iff *all* contributing pooled cells
      are wet (logical AND over the kernel footprint).  Yields the
      smallest h-mask whose every wet h-cell is fully surrounded by
      wet staggered cells.

    Algorithm
    ---------
    For each axis *a* where ``kernel[a] = k``:

    * ``direction='leading'`` — ``pool[..., j, ...]`` was built from
      ``arr[..., j-(k-1) : j+1, ...]``, so ``h[..., j, ...]`` is
      constrained by the pooled cells at indices ``j, j+1, ..., j+k-1``
      (the function shifts pooled *right* and combines).
    * ``direction='trailing'`` — ``pool[..., j, ...]`` was built from
      ``arr[..., j : j+k, ...]``, so ``h[..., j, ...]`` is constrained
      by the pooled cells at indices ``j-(k-1), ..., j-1, j`` (the
      function shifts pooled *left* and combines).

    The function takes shifted views of ``pooled_mask`` along each axis
    (offsets ``0..k-1``) and reduces them with OR (permissive) or AND
    (conservative) elementwise.

    Indices outside ``[0, N - 1]`` are padded with the *identity element*
    of the chosen mode — ``False`` for ``permissive`` (OR's identity)
    and ``True`` for ``conservative`` (AND's identity) — so out-of-bounds
    entries do not artificially wipe or fill the boundary.

    Note that the round-trip ``pool_bool → h_from_pooled`` is **lossy**
    in the permissive direction at isolated wet cells (an h-cell with
    no axis-aligned wet neighbour will have both adjacent faces dry,
    and the permissive inverse marks it as dry).  The recovered h-mask
    is always a *subset* of the true h-mask.

    Parameters
    ----------
    pooled_mask : np.ndarray
        Boolean staggered mask (e.g. u, v, w, xy_corner).  Same shape
        as the desired h-mask.
    kernel : tuple of int
        Per-axis forward-pool kernel (the same tuple that was passed
        to :func:`pool_bool` to derive ``pooled_mask`` from h).
    mode : {'permissive', 'conservative'}
        Inversion strategy when the inverse is ambiguous.
    direction : {'leading', 'trailing'}
        Pad direction of the forward :func:`pool_bool` call.  Must
        match the direction the input was produced with.  Default
        ``'leading'``.

    Returns
    -------
    np.ndarray
        Inferred h-mask, same shape and dtype (bool) as ``pooled_mask``.

    Examples
    --------
    Recover an all-ocean h-mask from its forward y-face pool:

    >>> import numpy as np
    >>> h = np.ones((4,), dtype=bool)
    >>> u = pool_bool(h.astype(float), kernel=(2,), threshold=3.0 / 4.0)
    >>> h_from_pooled(u, kernel=(2,), mode="permissive")
    array([ True,  True,  True,  True])

    Permissive vs conservative on the same input:

    >>> u = np.array([False, True, True, False])
    >>> h_from_pooled(u, kernel=(2,), mode="permissive")
    array([ True,  True,  True, False])
    >>> h_from_pooled(u, kernel=(2,), mode="conservative")
    array([False,  True, False, False])
    """
    if mode not in ("permissive", "conservative"):
        raise ValueError(f"mode must be 'permissive' or 'conservative', got {mode!r}")
    if direction not in ("leading", "trailing"):
        raise ValueError(
            f"direction must be 'leading' or 'trailing', got {direction!r}"
        )
    if len(kernel) != pooled_mask.ndim:
        raise ValueError(
            f"kernel length {len(kernel)} does not match pooled_mask.ndim "
            f"{pooled_mask.ndim}"
        )

    pooled_b = np.asarray(pooled_mask, dtype=bool)
    ndim = pooled_b.ndim

    # Pick the OR identity (False) for permissive and the AND identity (True)
    # for conservative so that out-of-bounds shifted entries are neutral.
    if mode == "permissive":
        result = np.zeros(pooled_b.shape, dtype=bool)
        identity = False  # x OR False == x
    else:
        result = np.ones(pooled_b.shape, dtype=bool)
        identity = True  # x AND True == x

    # For each multi-axis offset (o_0, ..., o_{ndim-1}) with 0 <= o_a < k_a,
    # build the shifted view pooled[..., j + o_a, ...] (leading) or
    # pooled[..., j - o_a, ...] (trailing) and combine it into `result`.
    # Shape: pooled_mask.shape (every shifted view is padded back up to full
    # shape with `identity`).
    for offsets in itertools.product(*(range(k) for k in kernel)):
        shifted = pooled_b
        for axis, off in enumerate(offsets):
            if off == 0:
                continue
            sl = [slice(None)] * ndim
            pad_width = [(0, 0)] * ndim
            if direction == "leading":
                # Shift pooled RIGHT by `off` (take from `off` to end, pad
                # trailing).  This aligns pooled[..., j + off, ...] with the
                # h[..., j, ...] slot.
                sl[axis] = slice(off, None)
                pad_width[axis] = (0, off)
            else:
                # Shift pooled LEFT by `off` (take from start to `-off`, pad
                # leading).  This aligns pooled[..., j - off, ...] with the
                # h[..., j, ...] slot.
                sl[axis] = slice(0, -off)
                pad_width[axis] = (off, 0)
            front = shifted[tuple(sl)]
            shifted = np.pad(front, pad_width, constant_values=identity)

        if mode == "permissive":
            result = result | shifted
        else:
            result = result & shifted

    return result


def dilate_mask(mask: np.ndarray) -> np.ndarray:
    """Binary dilation by 1 cell with an n-D *cross* structuring element.

    Wraps :func:`scipy.ndimage.binary_dilation` with a cross-shaped
    structuring element of size ``(3,) * ndim``: the centre and the
    six (or four, in 2-D) immediate axis-aligned neighbours are
    ``True``; diagonals are ``False``.  Boundary cells are treated as
    zero (no wraparound).

    This is the building block for the 4-level land / coast / near-coast
    / open-ocean classification used in the mask factories: starting
    from the land mask ``~h``, two successive dilations identify the
    coast (1 ring out from land) and near-coast (2 rings out) tiers.

    Algorithm
    ---------
    1. Build a structuring element ``struct`` of shape ``(3,)*ndim``,
       all-zero except the centre ``(1, 1, ..., 1)`` and each of the
       ``2 * ndim`` axis-aligned face neighbours.
    2. Call ``scipy.ndimage.binary_dilation(mask, structure=struct,
       border_value=0)``.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask of any number of dimensions.

    Returns
    -------
    np.ndarray
        Dilated boolean mask, same shape.

    Examples
    --------
    A single wet cell dilates to a 5-cell cross (no diagonals):

    >>> import numpy as np
    >>> m = np.zeros((5, 5), dtype=bool)
    >>> m[2, 2] = True
    >>> dilate_mask(m).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])

    The cross shape means corner-touching cells do *not* propagate
    through dilation:

    >>> m = np.zeros((4, 4), dtype=bool)
    >>> m[0, 0] = True
    >>> dilate_mask(m).astype(int)
    array([[1, 1, 0, 0],
           [1, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    """
    ndim = mask.ndim
    # struct shape: (3,) * ndim — only the centre and the 2*ndim face
    # neighbours are True (no diagonals).
    struct = np.zeros((3,) * ndim, dtype=bool)
    centre = (1,) * ndim
    struct[centre] = True
    for axis in range(ndim):
        for offset in (0, 2):
            idx = list(centre)
            idx[axis] = offset
            struct[tuple(idx)] = True
    return binary_dilation(mask.astype(bool), structure=struct, border_value=0)


def count_contiguous(
    arr: np.ndarray,
    axis: int,
    forward: bool,
) -> np.ndarray:
    """Count contiguous wet cells from each point along one axis (n-D).

    For each cell, returns the number of consecutive wet cells starting
    *at* that cell and moving in the chosen direction along ``axis``.
    A wet cell at the start counts as 1; a dry cell returns 0.  Used to
    derive directional stencil capability for adaptive stencil masks.

    Algorithm
    ---------
    Linear single-pass recurrence along the chosen axis:

    * forward direction (scanning from high indices down)::

        count[i] = arr[i] * (1 + count[i + 1])

    * backward direction (scanning from low indices up)::

        count[i] = arr[i] * (1 + count[i - 1])

    The scan axis is moved to position 0 with :func:`np.moveaxis` so the
    Python loop iterates over scalars and the perpendicular axes
    vectorise.  At ``O(N_axis)`` per perpendicular column, the total
    cost is ``O(arr.size)``.

    Parameters
    ----------
    arr : np.ndarray
        Boolean / 0-1 wet mask of any number of dimensions.
    axis : int
        Axis along which to scan.
    forward : bool
        ``True``  → scan in the positive-axis direction (count of wet
        cells starting *at* this index and going to higher indices).
        ``False`` → scan in the negative-axis direction (count of wet
        cells starting *at* this index and going to lower indices).

    Returns
    -------
    np.ndarray
        ``int32`` array of the same shape as ``arr``.

    Examples
    --------
    Forward count along axis 0 of a 1-D mask with one dry cell:

    >>> import numpy as np
    >>> h = np.array([True, True, False, True, True, True])
    >>> count_contiguous(h, axis=0, forward=True)
    array([2, 1, 0, 3, 2, 1], dtype=int32)

    Backward count of the same mask:

    >>> count_contiguous(h, axis=0, forward=False)
    array([1, 2, 0, 1, 2, 3], dtype=int32)
    """
    # Cast to int so the recurrence multiplies cleanly (a wet cell contributes
    # 1 + count_neighbour, a dry cell contributes 0).  Shape: arr.shape
    arr_int = np.asarray(arr, dtype=np.int32)

    # Move the scan axis to position 0 so the Python loop iterates over
    # scalars while the perpendicular axes vectorise.  Shape:
    # (N_axis, ..._perpendicular)
    arr_moved = np.moveaxis(arr_int, axis, 0)
    n = arr_moved.shape[0]
    count = np.zeros_like(arr_moved)

    if forward:
        # Initialise the boundary cell from arr alone (no further neighbour),
        # then sweep down the axis applying the recurrence.
        count[-1] = arr_moved[-1]
        for i in range(n - 2, -1, -1):
            count[i] = arr_moved[i] * (1 + count[i + 1])
    else:
        count[0] = arr_moved[0]
        for i in range(1, n):
            count[i] = arr_moved[i] * (1 + count[i - 1])

    # Restore the original axis ordering.  Shape: arr.shape
    return np.moveaxis(count, 0, axis)


def make_sponge(shape: tuple[int, ...], width: int) -> np.ndarray:
    """n-D linear sponge ramp: 0 at every wall, 1 in the interior.

    Produces a damping-weight array used to relax model fields toward
    a reference state near the domain boundaries.  Per axis, builds a
    linear ramp that rises from 0 at the wall to 1 at distance
    ``width`` cells inside.  The full sponge is the elementwise product
    of all per-axis ramps via broadcasting, so the corner regions damp
    in *both* directions simultaneously.

    A width of ``0`` (or any non-positive value, or ``None``) returns
    an all-ones array — i.e. no damping anywhere.

    Algorithm
    ---------
    For each axis *a* with extent ``n_a``::

        idx_a = arange(n_a)
        ramp_a = clip(min(idx_a, n_a - 1 - idx_a) / width, 0, 1)
        out *= ramp_a.reshape((1, ..., n_a, ..., 1))

    The ``min(idx, n-1-idx)`` term gives the distance to the *nearest*
    wall along that axis; clipping to ``[0, 1]`` flattens the ramp at
    the interior plateau.

    Parameters
    ----------
    shape : tuple of int
        Output array shape.
    width : int
        Number of cells over which the ramp rises from 0 to 1.  A
        non-positive value returns an all-ones array.

    Returns
    -------
    np.ndarray
        ``float32`` sponge array with the requested shape.

    Examples
    --------
    1-D sponge of width 2 in a 6-cell domain (note the corners damp to
    zero and the interior plateaus at 1):

    >>> make_sponge((6,), width=2)
    array([0. , 0.5, 1. , 1. , 0.5, 0. ], dtype=float32)

    2-D sponge of width 1 in a 5×5 domain (the corners damp in both
    directions, so they end up at 0):

    >>> make_sponge((5, 5), width=1)
    array([[0., 0., 0., 0., 0.],
           [0., 1., 1., 1., 0.],
           [0., 1., 1., 1., 0.],
           [0., 1., 1., 1., 0.],
           [0., 0., 0., 0., 0.]], dtype=float32)

    A non-positive width disables damping entirely:

    >>> make_sponge((4,), width=0)
    array([1., 1., 1., 1.], dtype=float32)
    """
    if width is None or width <= 0:
        return np.ones(shape, dtype=np.float32)

    width_f = float(width)
    out = np.ones(shape, dtype=np.float32)
    # Build a per-axis ramp and multiply it onto `out` via broadcasting.
    # Each per-axis ramp has shape (1, ..., n_a, ..., 1) so axis-a damping
    # only modulates that axis.  Shape: shape (preserved through the loop)
    for axis, n in enumerate(shape):
        idx = np.arange(n, dtype=np.float32)
        ramp = np.clip(np.minimum(idx, (n - 1) - idx) / width_f, 0.0, 1.0)
        broadcast_shape = [1] * len(shape)
        broadcast_shape[axis] = n
        out = out * ramp.reshape(broadcast_shape)
    return out
