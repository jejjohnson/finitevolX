"""
Harmonic and biharmonic diffusion operators (flux form) on Arakawa C-grids.

Harmonic diffusion computes dh/dt = div(kappa * grad h) at T-points from
staggered face fluxes via forward-then-backward finite differences.

Biharmonic diffusion applies the harmonic operator twice to give
dh/dt = -kappa * nabla^4 h = -kappa * nabla^2(nabla^2 h), providing
scale-selective dissipation that damps short-wave modes much more
strongly than long-wave modes.

Algorithm (2-D uniform grid with spacing dx, dy)
-------------------------------------------------
Step 1 - East-face flux at U-points (forward diff T -> U):

    flux_x[j, i+1/2] = kappa * (h[j, i+1] - h[j, i]) / dx

Step 2 - North-face flux at V-points (forward diff T -> V):

    flux_y[j+1/2, i] = kappa * (h[j+1, i] - h[j, i]) / dy

Step 3 - Tendency at T-points (backward diff of fluxes, U -> T and V -> T):

    dh[j, i] = (flux_x[j, i+1/2] - flux_x[j, i-1/2]) / dx
             + (flux_y[j+1/2, i] - flux_y[j-1/2, i]) / dy

Boundary conditions
-------------------
Face fluxes at domain walls are zero by construction:

* West boundary face (U-point col 0) is never written - stays zero.
* East boundary face (U-point col Nx-2) is not computed - stays zero.
* South boundary face (V-point row 0) is never written - stays zero.
* North boundary face (V-point row Ny-2) is not computed - stays zero.

This gives no-flux (closed-wall) BCs at all four domain walls by default.
Custom boundary conditions must be imposed via the tracer field ``h`` or
the diffusivity ``kappa``.

Masking
-------
Diffusion is one of the two operator families where mask application
must happen *between* the flux step and the divergence step (the other
is advection): a wet T-cell adjacent to a dry T-cell would otherwise
read polluted values for the dry-side flux.  See
``docs/concepts/masking.md`` for the rationale and the comparison with
the simpler post-compute-zero convention used by everything else.

Per the design decision that masks live at the *class-operator* layer,
the public functional form ``diffusion_2d`` is mask-free; the
``Diffusion2D`` / ``Diffusion3D`` class wrappers take an optional
``mask: ArakawaCGridMask`` and dispatch to the masked code path
internally.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid.cgrid_mask import ArakawaCGridMask
from finitevolx._src.grid.grid import ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.operators._ghost import interior, zero_z_ghosts

# ======================================================================
# Functional form (mask-free, public)
# ======================================================================


def diffusion_2d(
    h: Float[Array, "Ny Nx"],
    kappa: float | Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Horizontal tracer diffusion tendency at T-points (flux form).

    Computes dh/dt = div(kappa * grad h) = d/dx(kappa * dh/dx) +
    d/dy(kappa * dh/dy) at interior T-points using forward-then-backward
    finite differences.

    Only interior cells ``[1:-1, 1:-1]`` are written; the ghost ring is left
    as zero.  East and north boundary faces are not computed, giving no-flux
    (closed-wall) BCs at all four domain walls by default.

    This is the pure functional form. It does not accept a mask
    parameter — for masked diffusion, use the class wrapper
    :class:`Diffusion2D` which inlines the intermediate-flux masking
    correctly.  See ``docs/concepts/masking.md`` for the design rationale.

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Tracer field at T-points.
    kappa : float or Float[Array, "Ny Nx"]
        Diffusion coefficient.  May be a scalar or an array of the same shape
        as ``h`` (spatially varying diffusivity at T-points).  When ``kappa``
        is an array, the value at the source T-point is used for each face
        flux (i.e., the western/southern cell value for east/north faces).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Diffusion tendency dh/dt at T-points, same shape as ``h``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> h = jnp.zeros((10, 10))
    >>> tendency = diffusion_2d(h, kappa=1.0, dx=0.1, dy=0.1)
    >>> tendency.shape
    (10, 10)
    """
    return _diffusion_2d_impl(h, kappa, dx, dy, mask=None)


def _diffusion_2d_impl(
    h: Float[Array, "Ny Nx"],
    kappa: float | Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    *,
    mask: ArakawaCGridMask | None,
) -> Float[Array, "Ny Nx"]:
    """Shared implementation of 2-D diffusion, with optional mask.

    Private helper used by both the public mask-free functional form
    :func:`diffusion_2d` and the class wrapper :class:`Diffusion2D`.

    When ``mask`` is provided, masking is applied at the *intermediate
    flux* step (not just on the final output), because the simple
    "multiply the output by mask.h" pattern would still leave wet T-cells
    adjacent to land contaminated by polluted dry-side flux values.
    Specifically:

      flux_x *= mask.u   # no flux through dry east faces
      flux_y *= mask.v   # no flux through dry north faces
      out    *= mask.h   # zero tendency at dry T-cells
    """
    # Prepare kappa slices for each face direction.
    # When kappa is a full [Ny, Nx] array, use the western/southern source
    # T-cell value for each face:
    #   flux_x at (j, i+1/2) uses kappa[j, i] -> slice kappa_arr[1:-1, 1:-2]
    #   flux_y at (j+1/2, i) uses kappa[j, i] -> slice kappa_arr[1:-2, 1:-1]
    kappa_arr = jnp.asarray(kappa)
    if kappa_arr.ndim >= 2:
        kappa_x = kappa_arr[1:-1, 1:-2]  # (Ny-2, Nx-3) — source T-cell for east faces
        kappa_y = kappa_arr[1:-2, 1:-1]  # (Ny-3, Nx-2) — source T-cell for north faces
    else:
        kappa_x = kappa_arr
        kappa_y = kappa_arr

    # Step 1: East-face flux at U-points
    # flux_x[j, i+1/2] = kappa * (h[j, i+1] - h[j, i]) / dx
    flux_x = jnp.zeros_like(h)
    flux_x = flux_x.at[1:-1, 1:-2].set(kappa_x * (h[1:-1, 2:-1] - h[1:-1, 1:-2]) / dx)
    if mask is not None:
        flux_x = flux_x * mask.u

    # Step 2: North-face flux at V-points
    # flux_y[j+1/2, i] = kappa * (h[j+1, i] - h[j, i]) / dy
    flux_y = jnp.zeros_like(h)
    flux_y = flux_y.at[1:-2, 1:-1].set(kappa_y * (h[2:-1, 1:-1] - h[1:-2, 1:-1]) / dy)
    if mask is not None:
        flux_y = flux_y * mask.v

    # Step 3: Tendency at T-points (divergence of flux)
    du = (flux_x[1:-1, 1:-1] - flux_x[1:-1, :-2]) / dx
    dv = (flux_y[1:-1, 1:-1] - flux_y[:-2, 1:-1]) / dy
    out = interior(du + dv, h)

    if mask is not None:
        out = out * mask.h

    return out


def _diffusion_2d_fluxes_impl(
    h: Float[Array, "Ny Nx"],
    kappa: float | Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    *,
    mask: ArakawaCGridMask | None,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Shared implementation of the diagnostic flux pair, with optional mask."""
    kappa_arr = jnp.asarray(kappa)
    if kappa_arr.ndim >= 2:
        kappa_x = kappa_arr[1:-1, 1:-2]
        kappa_y = kappa_arr[1:-2, 1:-1]
    else:
        kappa_x = kappa_arr
        kappa_y = kappa_arr

    flux_x = jnp.zeros_like(h)
    flux_x = flux_x.at[1:-1, 1:-2].set(kappa_x * (h[1:-1, 2:-1] - h[1:-1, 1:-2]) / dx)
    if mask is not None:
        flux_x = flux_x * mask.u

    flux_y = jnp.zeros_like(h)
    flux_y = flux_y.at[1:-2, 1:-1].set(kappa_y * (h[2:-1, 1:-1] - h[1:-2, 1:-1]) / dy)
    if mask is not None:
        flux_y = flux_y * mask.v

    return flux_x, flux_y


# ======================================================================
# Class wrappers
# ======================================================================


class Diffusion2D(eqx.Module):
    """Horizontal diffusion operator (flux form) on a 2-D Arakawa C-grid.

    Computes dh/dt = div(kappa * grad h) at T-points from staggered face
    fluxes via forward-then-backward finite differences.

    Parameters
    ----------
    grid : ArakawaCGrid2D
        The underlying 2-D grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import ArakawaCGrid2D, Diffusion2D
    >>> grid = ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> diff_op = Diffusion2D(grid=grid)
    >>> h = jnp.ones((grid.Ny, grid.Nx))
    >>> tendency = diff_op(h, kappa=1e-3)  # zero for constant tracer
    >>> flux_x, flux_y = diff_op.fluxes(h, kappa=1e-3)
    """

    grid: ArakawaCGrid2D

    def __call__(
        self,
        h: Float[Array, "Ny Nx"],
        kappa: float | Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """Diffusion tendency dh/dt = div(kappa * grad h) at T-points.

        dh[j, i] = (flux_x[j, i+1/2] - flux_x[j, i-1/2]) / dx
                 + (flux_y[j+1/2, i] - flux_y[j-1/2, i]) / dy

        where:
            flux_x[j, i+1/2] = kappa * (h[j, i+1] - h[j, i]) / dx
            flux_y[j+1/2, i] = kappa * (h[j+1, i] - h[j, i]) / dy

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Tracer field at T-points.
        kappa : float or Float[Array, "Ny Nx"]
            Diffusion coefficient (scalar or T-point field).
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the U-face fluxes are
            multiplied by ``mask.u``, the V-face fluxes by ``mask.v``,
            and the T-point tendency by ``mask.h``. This is the
            intermediate-masking pattern (see ``docs/concepts/masking.md``).

        Returns
        -------
        Float[Array, "Ny Nx"]
            Diffusion tendency at T-points.
        """
        return _diffusion_2d_impl(h, kappa, self.grid.dx, self.grid.dy, mask=mask)

    def fluxes(
        self,
        h: Float[Array, "Ny Nx"],
        kappa: float | Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Diagnostic diffusive face fluxes at U- and V-points.

        Returns the east-face and north-face diffusive fluxes before the
        divergence step, useful for flux-conservative time-stepping and
        diagnostics.

            flux_x[j, i+1/2] = kappa * (h[j, i+1] - h[j, i]) / dx  (U-points)
            flux_y[j+1/2, i] = kappa * (h[j+1, i] - h[j, i]) / dy  (V-points)

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Tracer field at T-points.
        kappa : float or Float[Array, "Ny Nx"]
            Diffusion coefficient (scalar or T-point field).
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, ``flux_x`` is
            multiplied by ``mask.u`` and ``flux_y`` by ``mask.v``.

        Returns
        -------
        tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
            ``(flux_x, flux_y)`` — east-face fluxes at U-points and
            north-face fluxes at V-points.
        """
        return _diffusion_2d_fluxes_impl(
            h, kappa, self.grid.dx, self.grid.dy, mask=mask
        )


class Diffusion3D(eqx.Module):
    """Horizontal diffusion operator (flux form) on a 3-D Arakawa C-grid.

    Applies dh/dt = div(kappa * grad h) independently at each z-level.
    The 3-D array shape is [Nz, Ny, Nx].

    Parameters
    ----------
    grid : ArakawaCGrid3D
        The underlying 3-D grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import ArakawaCGrid3D, Diffusion3D
    >>> grid = ArakawaCGrid3D.from_interior(4, 8, 8, 1.0, 1.0, 1.0)
    >>> diff_op = Diffusion3D(grid=grid)
    >>> h = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
    >>> tendency = diff_op(h, kappa=1e-3)  # zero for constant tracer
    """

    grid: ArakawaCGrid3D

    def __call__(
        self,
        h: Float[Array, "Nz Ny Nx"],
        kappa: float | Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Diffusion tendency dh/dt = div(kappa * grad h) at T-points
        over all z-levels.

        Applies the horizontal diffusion stencil independently at each
        z-level.  Only interior cells ``[1:-1, 1:-1, 1:-1]`` are written;
        the ghost ring is left as zero.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Tracer field at T-points.
        kappa : float or Float[Array, "Nz Ny Nx"]
            Diffusion coefficient (scalar or T-point field).
        mask : ArakawaCGridMask or None
            Optional 2-D land/ocean mask, broadcast over all z-levels.
            If provided, the same intermediate-masking pattern as
            :class:`Diffusion2D` is applied at each z-level.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Diffusion tendency at T-points.
        """
        dx, dy = self.grid.dx, self.grid.dy
        kappa_arr = jnp.asarray(kappa)
        kappa_ax = 0 if kappa_arr.ndim >= 3 else None

        def _apply(h_k, kap_k):
            return _diffusion_2d_impl(h_k, kap_k, dx, dy, mask=mask)

        out = eqx.filter_vmap(_apply, in_axes=(0, kappa_ax))(h, kappa_arr)
        # Zero z-ghost slices.
        return zero_z_ghosts(out)

    def fluxes(
        self,
        h: Float[Array, "Nz Ny Nx"],
        kappa: float | Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]:
        """Diagnostic diffusive face fluxes at U- and V-points, all z-levels.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Tracer field at T-points.
        kappa : float or Float[Array, "Nz Ny Nx"]
            Diffusion coefficient (scalar or T-point field).
        mask : ArakawaCGridMask or None
            Optional 2-D land/ocean mask, broadcast over all z-levels.

        Returns
        -------
        tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]
            ``(flux_x, flux_y)`` — east-face fluxes at U-points and
            north-face fluxes at V-points.
        """
        dx, dy = self.grid.dx, self.grid.dy
        kappa_arr = jnp.asarray(kappa)
        kappa_ax = 0 if kappa_arr.ndim >= 3 else None

        def _apply(h_k, kap_k):
            return _diffusion_2d_fluxes_impl(h_k, kap_k, dx, dy, mask=mask)

        fx, fy = eqx.filter_vmap(_apply, in_axes=(0, kappa_ax))(h, kappa_arr)
        # Zero z-ghost slices.
        fx = zero_z_ghosts(fx)
        fy = zero_z_ghosts(fy)
        return fx, fy


class BiharmonicDiffusion2D(eqx.Module):
    """Biharmonic (nabla^4) diffusion operator on a 2-D Arakawa C-grid.

    Computes the local biharmonic diffusion tendency:

        dh/dt|_diff = -kappa * nabla^4 h

    where nabla^4 h = nabla^2(nabla^2 h) is implemented as two
    successive flux-form Laplacians via :class:`Diffusion2D`.  The
    negative sign ensures that a positive kappa provides dissipation
    (the operator damps high-wavenumber modes).

    Scale-selective property: for a Fourier mode with wavenumber **k**, the
    harmonic tendency scales as ``-kappa_h * k^2`` while the biharmonic
    tendency scales as ``-kappa_bi * k^4``.  Biharmonic diffusion therefore
    damps small scales much more strongly than large scales.

    Only interior cells ``[1:-1, 1:-1]`` are written; the ghost ring is
    zero.  The caller is responsible for boundary conditions.

    Notes
    -----
    The ghost ring of the intermediate Laplacian nabla^2 h is zero
    (Dirichlet-0), not a zero-normal-gradient (Neumann) BC.  This means
    the second Laplacian pass reads a zero halo for the intermediate
    field, which contaminates the outermost interior row/column of the
    final tendency even if the input ``h`` had correctly set ghost
    cells.  Only results in the deep interior ``[2:-2, 2:-2]`` are
    fully BC-consistent.  For periodic domains, call ``enforce_periodic``
    on ``h`` before invoking this operator.

    When a mask is provided, it is applied to the *final* output only
    (not the intermediate Laplacian).  This avoids zeroing the
    intermediate field — which would corrupt the second-pass stencil
    near the coastline — while still guaranteeing the canonical
    "tendency is exactly zero in dry cells" invariant.

    Parameters
    ----------
    grid : ArakawaCGrid2D
        The underlying 2-D grid.

    References
    ----------
    .. [1] MITgcm Biharmonic Mixing:
           https://mitgcm.readthedocs.io/en/latest/optionals/packages/mixing.html#biharmonic-mixing
    .. [2] Leith, C. E. (1968). Diffusion approximation for two-dimensional
           turbulence. *Physics of Fluids*, 11(3), 671-673.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import ArakawaCGrid2D, BiharmonicDiffusion2D
    >>> grid = ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> op = BiharmonicDiffusion2D(grid=grid)
    >>> h = jnp.ones((grid.Ny, grid.Nx))
    >>> tend = op(h, kappa=1e-6)  # zero for constant field
    >>> tend.shape
    (10, 10)
    """

    grid: ArakawaCGrid2D
    _harm: Diffusion2D

    def __init__(self, grid: ArakawaCGrid2D) -> None:
        self.grid = grid
        self._harm = Diffusion2D(grid=grid)

    def __call__(
        self,
        h: Float[Array, "Ny Nx"],
        kappa: float,
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """Apply biharmonic diffusion and return -kappa * nabla^4 h.

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar tracer field at T-points, shape ``[Ny, Nx]``.
        kappa : float
            Biharmonic diffusion coefficient (kappa >= 0 gives dissipation).
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the *final* T-point
            output is multiplied by ``mask.h``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Tendency -kappa * nabla^4 h at T-points, same shape as ``h``.
            Ghost cells are zero.
        """
        # First Laplacian pass: kappa=1.0 gives pure nabla^2 h.
        # Intentionally unmasked — see class docstring.
        lap1 = self._harm(h, kappa=1.0)
        # Second Laplacian pass: nabla^2(nabla^2 h) = nabla^4 h
        lap2 = self._harm(lap1, kappa=1.0)
        out = -kappa * lap2
        if mask is not None:
            out = out * mask.h
        return out


class BiharmonicDiffusion3D(eqx.Module):
    """Biharmonic (nabla^4) diffusion operator on a 3-D Arakawa C-grid.

    Applies the horizontal biharmonic Laplacian independently at each
    z-level:

        dh/dt|_diff = -kappa * nabla^4_h h

    where nabla^4_h = nabla^2_h(nabla^2_h) denotes the horizontal biharmonic
    operator, implemented as two successive :class:`Diffusion3D` passes.

    Only interior cells ``[1:-1, 1:-1, 1:-1]`` are written; the ghost ring
    is zero.  The caller is responsible for boundary conditions.

    Notes
    -----
    Same intermediate-Dirichlet-0 caveats and same final-output-only
    masking convention as :class:`BiharmonicDiffusion2D`.

    Parameters
    ----------
    grid : ArakawaCGrid3D
        The underlying 3-D grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import ArakawaCGrid3D, BiharmonicDiffusion3D
    >>> grid = ArakawaCGrid3D.from_interior(4, 8, 8, 1.0, 1.0, 1.0)
    >>> op = BiharmonicDiffusion3D(grid=grid)
    >>> h = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
    >>> tend = op(h, kappa=1e-6)
    >>> tend.shape
    (6, 10, 10)
    """

    grid: ArakawaCGrid3D
    _harm: Diffusion3D

    def __init__(self, grid: ArakawaCGrid3D) -> None:
        self.grid = grid
        self._harm = Diffusion3D(grid=grid)

    def __call__(
        self,
        h: Float[Array, "Nz Ny Nx"],
        kappa: float,
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Apply horizontal biharmonic diffusion and return
        -kappa * nabla^4_h h.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar tracer field at T-points, shape ``[Nz, Ny, Nx]``.
        kappa : float
            Biharmonic diffusion coefficient (kappa >= 0 gives dissipation).
        mask : ArakawaCGridMask or None
            Optional 2-D land/ocean mask, broadcast over all z-levels.
            Applied to the *final* output only.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Tendency -kappa * nabla^4_h h at T-points, same shape as ``h``.
            Ghost cells are zero.
        """
        # First Laplacian pass: kappa=1.0 gives pure nabla^2_h h
        lap1 = self._harm(h, kappa=1.0)
        # Second Laplacian pass: nabla^2_h(nabla^2_h h) = nabla^4_h h
        lap2 = self._harm(lap1, kappa=1.0)
        out = -kappa * lap2
        if mask is not None:
            out = out * mask.h
        return out
