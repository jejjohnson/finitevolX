"""
Harmonic and biharmonic diffusion operators (flux form) on Arakawa C-grids.

Harmonic diffusion computes ∂h/∂t = ∇·(κ ∇h) at T-points from staggered
face fluxes via forward-then-backward finite differences.

Biharmonic diffusion applies the harmonic operator twice to give
∂h/∂t = -κ ∇⁴h = -κ ∇²(∇²h), providing scale-selective dissipation that
damps short-wave modes much more strongly than long-wave modes.

Algorithm (2-D uniform grid with spacing dx, dy)
-------------------------------------------------
Step 1 – East-face flux at U-points (forward diff T → U):

    flux_x[j, i+1/2] = κ * (h[j, i+1] - h[j, i]) / dx

Step 2 – North-face flux at V-points (forward diff T → V):

    flux_y[j+1/2, i] = κ * (h[j+1, i] - h[j, i]) / dy

Step 3 – Tendency at T-points (backward diff of fluxes, U → T and V → T):

    dh[j, i] = (flux_x[j, i+1/2] - flux_x[j, i-1/2]) / dx
             + (flux_y[j+1/2, i] - flux_y[j-1/2, i]) / dy

Boundary conditions
-------------------
Face fluxes at domain walls are zero by construction:

* West boundary face (U-point col 0) is never written — stays zero.
* East boundary face (U-point col Nx-2) is not computed — stays zero.
* South boundary face (V-point row 0) is never written — stays zero.
* North boundary face (V-point row Ny-2) is not computed — stays zero.

This gives no-flux (closed-wall) BCs at all four domain walls by default.
Custom boundary conditions must be imposed via the tracer field ``h``, the
diffusivity ``kappa``, or by providing a ``Mask2D`` / ``Mask3D`` to the
class operator.

Masking
-------
The class operators ``Diffusion2D`` / ``Diffusion3D`` take an optional
``mask`` class attribute.  Unlike the simpler post-compute pattern used
by most other operators in this package, diffusion needs *intermediate*
flux masking — multiplying the already-computed tendency by ``mask.h``
would leave wet T-cells adjacent to land contaminated by the polluted
dry-side flux contributions.  The class wrappers therefore apply the
mask via the three-step pattern:

* ``flux_x *= mask.u`` at the U-face stage,
* ``flux_y *= mask.v`` at the V-face stage,
* ``tendency *= mask.h`` on the final output.

The ``diffusion_2d`` free function stays **mask-free** by design, per the
layering rule that functional helpers don't know about masks (#209).
Users who want masked diffusion should use ``Diffusion2D`` / ``Diffusion3D``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators._ghost import interior, zero_z_ghosts


def diffusion_2d(
    h: Float[Array, "Ny Nx"],
    kappa: float | Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Horizontal tracer diffusion tendency at T-points (flux form).

    Computes ∂h/∂t = ∇·(κ ∇h) = ∂/∂x(κ ∂h/∂x) + ∂/∂y(κ ∂h/∂y)
    at interior T-points using forward-then-backward finite differences.

    Only interior cells ``[1:-1, 1:-1]`` are written; the ghost ring is left
    as zero.  East and north boundary faces are not computed, giving no-flux
    (closed-wall) BCs at all four domain walls by default.

    This is the mask-free functional form.  For masked diffusion, use
    :class:`Diffusion2D` with a ``mask=`` class attribute; it applies the
    mask via the intermediate-flux pattern described in the module
    docstring.

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
        Diffusion tendency ∂h/∂t at T-points, same shape as ``h``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> h = jnp.zeros((10, 10))
    >>> tendency = diffusion_2d(h, kappa=1.0, dx=0.1, dy=0.1)
    >>> tendency.shape
    (10, 10)
    """
    return _diffusion_2d_impl(h, kappa, dx, dy, mh=None, mu=None, mv=None)


def _diffusion_2d_impl(
    h: Float[Array, "Ny Nx"],
    kappa: float | Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    mh: Bool[Array, "Ny Nx"] | Float[Array, "Ny Nx"] | None,
    mu: Bool[Array, "Ny Nx"] | Float[Array, "Ny Nx"] | None,
    mv: Bool[Array, "Ny Nx"] | Float[Array, "Ny Nx"] | None,
) -> Float[Array, "Ny Nx"]:
    """Shared 2-D diffusion kernel with explicit raw-array masks.

    Internal helper used by both :func:`diffusion_2d` (which passes
    all-None) and :class:`Diffusion2D`/:class:`Diffusion3D` (which pass
    raw bool / float arrays).  The three-step intermediate-flux masking
    pattern (see module docstring) is inlined here so vmap can also
    use it per-z-slice from ``Diffusion3D``.
    """
    # Prepare kappa slices for each face direction.
    # When kappa is a full [Ny, Nx] array, use the western/southern source
    # T-cell value for each face:
    #   flux_x at (j, i+½) uses κ[j, i] → slice kappa_arr[1:-1, 1:-2]
    #   flux_y at (j+½, i) uses κ[j, i] → slice kappa_arr[1:-2, 1:-1]
    kappa_arr = jnp.asarray(kappa)
    if kappa_arr.ndim >= 2:
        kappa_x = kappa_arr[1:-1, 1:-2]  # (Ny-2, Nx-3) — source T-cell for east faces
        kappa_y = kappa_arr[1:-2, 1:-1]  # (Ny-3, Nx-2) — source T-cell for north faces
    else:
        kappa_x = kappa_arr
        kappa_y = kappa_arr

    # Step 1: East-face flux at U-points
    # flux_x[j, i+1/2] = κ * (h[j, i+1] - h[j, i]) / dx
    # Written for i = 1 ... Nx-3 only; east boundary face (i=Nx-2) stays 0.
    flux_x = jnp.zeros_like(h)
    flux_x = flux_x.at[1:-1, 1:-2].set(kappa_x * (h[1:-1, 2:-1] - h[1:-1, 1:-2]) / dx)
    if mu is not None:
        flux_x = flux_x * mu

    # Step 2: North-face flux at V-points
    # flux_y[j+1/2, i] = κ * (h[j+1, i] - h[j, i]) / dy
    # Written for j = 1 ... Ny-3 only; north boundary face (j=Ny-2) stays 0.
    flux_y = jnp.zeros_like(h)
    flux_y = flux_y.at[1:-2, 1:-1].set(kappa_y * (h[2:-1, 1:-1] - h[1:-2, 1:-1]) / dy)
    if mv is not None:
        flux_y = flux_y * mv

    # Step 3: Tendency at T-points (divergence of flux)
    # dh[j, i] = (flux_x[j, i+1/2] - flux_x[j, i-1/2]) / dx
    #           + (flux_y[j+1/2, i] - flux_y[j-1/2, i]) / dy
    du = (flux_x[1:-1, 1:-1] - flux_x[1:-1, :-2]) / dx
    dv = (flux_y[1:-1, 1:-1] - flux_y[:-2, 1:-1]) / dy
    out = interior(du + dv, h)

    if mh is not None:
        out = out * mh

    return out


def _diffusion_2d_fluxes_impl(
    h: Float[Array, "Ny Nx"],
    kappa: float | Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    mu: Bool[Array, "Ny Nx"] | Float[Array, "Ny Nx"] | None,
    mv: Bool[Array, "Ny Nx"] | Float[Array, "Ny Nx"] | None,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Shared 2-D diagnostic-flux kernel with explicit raw-array masks."""
    kappa_arr = jnp.asarray(kappa)
    if kappa_arr.ndim >= 2:
        kappa_x = kappa_arr[1:-1, 1:-2]
        kappa_y = kappa_arr[1:-2, 1:-1]
    else:
        kappa_x = kappa_arr
        kappa_y = kappa_arr

    flux_x = jnp.zeros_like(h)
    flux_x = flux_x.at[1:-1, 1:-2].set(kappa_x * (h[1:-1, 2:-1] - h[1:-1, 1:-2]) / dx)
    if mu is not None:
        flux_x = flux_x * mu

    flux_y = jnp.zeros_like(h)
    flux_y = flux_y.at[1:-2, 1:-1].set(kappa_y * (h[2:-1, 1:-1] - h[1:-2, 1:-1]) / dy)
    if mv is not None:
        flux_y = flux_y * mv

    return flux_x, flux_y


class Diffusion2D(eqx.Module):
    """Horizontal diffusion operator (flux form) on a 2-D Arakawa C-grid.

    Computes ∂h/∂t = ∇·(κ ∇h) at T-points from staggered face fluxes via
    forward-then-backward finite differences.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided, the three-step
        intermediate-flux masking pattern is applied inside both
        :meth:`__call__` and :meth:`fluxes`:

        * ``flux_x *= mask.u`` at the U-face stage,
        * ``flux_y *= mask.v`` at the V-face stage,
        * tendency ``*= mask.h`` on the final output (``__call__`` only).

        Unlike most other operators in this package, diffusion cannot
        use the simpler post-compute pattern because the divergence at
        wet T-cells adjacent to land would otherwise be contaminated
        by polluted dry-side face fluxes.  ``None`` (default) matches
        the pre-existing unmasked behaviour bit for bit.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid2D, Diffusion2D
    >>> grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> diff_op = Diffusion2D(grid=grid)
    >>> h = jnp.ones((grid.Ny, grid.Nx))
    >>> tendency = diff_op(h, kappa=1e-3)  # zero for constant tracer
    >>> flux_x, flux_y = diff_op.fluxes(h, kappa=1e-3)
    """

    grid: CartesianGrid2D
    mask: Mask2D | None = None

    def __call__(
        self,
        h: Float[Array, "Ny Nx"],
        kappa: float | Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Diffusion tendency ∂h/∂t = ∇·(κ ∇h) at T-points.

        dh[j, i] = (flux_x[j, i+1/2] - flux_x[j, i-1/2]) / dx
                 + (flux_y[j+1/2, i] - flux_y[j-1/2, i]) / dy

        where:
            flux_x[j, i+1/2] = κ * (h[j, i+1] - h[j, i]) / dx
            flux_y[j+1/2, i] = κ * (h[j+1, i] - h[j, i]) / dy

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Tracer field at T-points.
        kappa : float or Float[Array, "Ny Nx"]
            Diffusion coefficient (scalar or T-point field).

        Returns
        -------
        Float[Array, "Ny Nx"]
            Diffusion tendency at T-points.  When ``self.mask`` is set,
            the intermediate-flux masking pattern described in the
            class docstring is applied.
        """
        if self.mask is None:
            return _diffusion_2d_impl(
                h, kappa, self.grid.dx, self.grid.dy, mh=None, mu=None, mv=None
            )
        return _diffusion_2d_impl(
            h,
            kappa,
            self.grid.dx,
            self.grid.dy,
            mh=self.mask.h,
            mu=self.mask.u,
            mv=self.mask.v,
        )

    def fluxes(
        self,
        h: Float[Array, "Ny Nx"],
        kappa: float | Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Diagnostic diffusive face fluxes at U- and V-points.

        Returns the east-face and north-face diffusive fluxes before the
        divergence step, useful for flux-conservative time-stepping and
        diagnostics.

            flux_x[j, i+1/2] = κ * (h[j, i+1] - h[j, i]) / dx  (U-points)
            flux_y[j+1/2, i] = κ * (h[j+1, i] - h[j, i]) / dy  (V-points)

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Tracer field at T-points.
        kappa : float or Float[Array, "Ny Nx"]
            Diffusion coefficient (scalar or T-point field).

        Returns
        -------
        tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
            ``(flux_x, flux_y)`` — east-face fluxes at U-points and
            north-face fluxes at V-points.  When ``self.mask`` is set,
            ``flux_x`` is multiplied by ``mask.u`` and ``flux_y`` by
            ``mask.v``.
        """
        if self.mask is None:
            return _diffusion_2d_fluxes_impl(
                h, kappa, self.grid.dx, self.grid.dy, mu=None, mv=None
            )
        return _diffusion_2d_fluxes_impl(
            h,
            kappa,
            self.grid.dx,
            self.grid.dy,
            mu=self.mask.u,
            mv=self.mask.v,
        )


class Diffusion3D(eqx.Module):
    """Horizontal diffusion operator (flux form) on a 3-D Arakawa C-grid.

    Applies ∂h/∂t = ∇·(κ ∇h) independently at each z-level.
    The 3-D array shape is [Nz, Ny, Nx].

    Parameters
    ----------
    grid : CartesianGrid3D
        The underlying 3-D grid.
    mask : Mask3D or None, optional
        Optional 3-D land/ocean mask.  When provided, the intermediate
        flux masking pattern from :class:`Diffusion2D` is applied at
        every z-level via vmap with per-z slices of ``mask.h``,
        ``mask.u``, ``mask.v`` (Pattern B per issue #209 — the only
        way to get correct divergence at wet T-cells adjacent to
        coastlines).  ``None`` (default) matches the pre-existing
        unmasked behaviour bit for bit.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid3D, Diffusion3D
    >>> grid = CartesianGrid3D.from_interior(4, 8, 8, 1.0, 1.0, 1.0)
    >>> diff_op = Diffusion3D(grid=grid)
    >>> h = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
    >>> tendency = diff_op(h, kappa=1e-3)  # zero for constant tracer
    """

    grid: CartesianGrid3D
    mask: Mask3D | None = None

    def __call__(
        self,
        h: Float[Array, "Nz Ny Nx"],
        kappa: float | Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """Diffusion tendency ∂h/∂t = ∇·(κ ∇h) at T-points over all z-levels.

        Applies the horizontal diffusion stencil independently at each
        z-level.  Only interior cells ``[1:-1, 1:-1, 1:-1]`` are written;
        the ghost ring is left as zero.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Tracer field at T-points.
        kappa : float or Float[Array, "Nz Ny Nx"]
            Diffusion coefficient (scalar or T-point field).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Diffusion tendency at T-points.  When ``self.mask`` is set,
            the intermediate-flux masking pattern is applied per-z-slice.
        """
        dx, dy = self.grid.dx, self.grid.dy
        kappa_arr = jnp.asarray(kappa)
        kappa_ax = 0 if kappa_arr.ndim >= 3 else None

        if self.mask is None:
            # Unmasked path: vmap the free function over z-levels.
            def _apply_unmasked(h_k, kap_k):
                return _diffusion_2d_impl(h_k, kap_k, dx, dy, mh=None, mu=None, mv=None)

            out = eqx.filter_vmap(_apply_unmasked, in_axes=(0, kappa_ax))(h, kappa_arr)
            return zero_z_ghosts(out)

        # Masked path: vmap with per-z slices of mask.h / mask.u / mask.v.
        mh = self.mask.h  # (Nz, Ny, Nx)
        mu = self.mask.u
        mv = self.mask.v

        def _apply_masked(h_k, kap_k, mh_k, mu_k, mv_k):
            return _diffusion_2d_impl(h_k, kap_k, dx, dy, mh=mh_k, mu=mu_k, mv=mv_k)

        out = eqx.filter_vmap(_apply_masked, in_axes=(0, kappa_ax, 0, 0, 0))(
            h, kappa_arr, mh, mu, mv
        )
        return zero_z_ghosts(out)

    def fluxes(
        self,
        h: Float[Array, "Nz Ny Nx"],
        kappa: float | Float[Array, "Nz Ny Nx"],
    ) -> tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]:
        """Diagnostic diffusive face fluxes at U- and V-points, all z-levels.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Tracer field at T-points.
        kappa : float or Float[Array, "Nz Ny Nx"]
            Diffusion coefficient (scalar or T-point field).

        Returns
        -------
        tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]
            ``(flux_x, flux_y)`` — east-face fluxes at U-points and
            north-face fluxes at V-points.  When ``self.mask`` is set,
            ``flux_x`` is multiplied by ``mask.u`` and ``flux_y`` by
            ``mask.v`` at each z-level.
        """
        dx, dy = self.grid.dx, self.grid.dy
        kappa_arr = jnp.asarray(kappa)
        kappa_ax = 0 if kappa_arr.ndim >= 3 else None

        if self.mask is None:

            def _apply_unmasked(h_k, kap_k):
                return _diffusion_2d_fluxes_impl(h_k, kap_k, dx, dy, mu=None, mv=None)

            fx, fy = eqx.filter_vmap(_apply_unmasked, in_axes=(0, kappa_ax))(
                h, kappa_arr
            )
            return zero_z_ghosts(fx), zero_z_ghosts(fy)

        mu = self.mask.u
        mv = self.mask.v

        def _apply_masked(h_k, kap_k, mu_k, mv_k):
            return _diffusion_2d_fluxes_impl(h_k, kap_k, dx, dy, mu=mu_k, mv=mv_k)

        fx, fy = eqx.filter_vmap(_apply_masked, in_axes=(0, kappa_ax, 0, 0))(
            h, kappa_arr, mu, mv
        )
        return zero_z_ghosts(fx), zero_z_ghosts(fy)


class BiharmonicDiffusion2D(eqx.Module):
    """Biharmonic (∇⁴) diffusion operator on a 2-D Arakawa C-grid.

    Computes the local biharmonic diffusion tendency:

        ∂h/∂t|_diff = −κ · ∇⁴h

    where ∇⁴h = ∇²(∇²h) is implemented as two successive flux-form
    Laplacians via :class:`Diffusion2D`.  The negative sign ensures that a
    positive κ provides dissipation (the operator damps high-wavenumber
    modes).

    Scale-selective property: for a Fourier mode with wavenumber **k**, the
    harmonic tendency scales as ``−κ_h · k²`` while the biharmonic tendency
    scales as ``−κ_bi · k⁴``.  Biharmonic diffusion therefore damps small
    scales much more strongly than large scales.

    Only interior cells ``[1:-1, 1:-1]`` are written; the ghost ring is
    zero.  The caller is responsible for boundary conditions.

    Notes
    -----
    The ghost ring of the intermediate Laplacian ∇²h is zero (Dirichlet-0),
    not a zero-normal-gradient (Neumann) BC.  This means the second Laplacian
    pass reads a zero halo for the intermediate field, which contaminates the
    outermost interior row/column of the final tendency even if the input ``h``
    had correctly set ghost cells.  Only results in the deep interior
    ``[2:-2, 2:-2]`` are fully BC-consistent.  For periodic domains, call
    ``enforce_periodic`` on ``h`` before invoking this operator; this sets the
    input ghost ring correctly and reduces (but does not eliminate) the
    Dirichlet-0 contamination of the intermediate field.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided, the **inner harmonic
        Diffusion2D is deliberately built with** ``mask=None`` — masking
        the intermediate Laplacian would corrupt the second harmonic
        pass because ``lap1 == 0`` at dry T-cells becomes a forced
        zero-Dirichlet boundary for the second pass, which changes the
        ∇⁴ stencil at wet cells adjacent to land.  Instead, the mask is
        applied via a post-compute ``* mask.h`` on the **final**
        ``-κ ∇⁴h`` tendency only.  This is the design exception called
        out in issue #209 §4.

    References
    ----------
    .. [1] MITgcm Biharmonic Mixing:
           https://mitgcm.readthedocs.io/en/latest/optionals/packages/mixing.html#biharmonic-mixing
    .. [2] Leith, C. E. (1968). Diffusion approximation for two-dimensional
           turbulence. *Physics of Fluids*, 11(3), 671-673.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid2D, BiharmonicDiffusion2D
    >>> grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> op = BiharmonicDiffusion2D(grid=grid)
    >>> h = jnp.ones((grid.Ny, grid.Nx))
    >>> tend = op(h, kappa=1e-6)  # zero for constant field
    >>> tend.shape
    (10, 10)
    """

    grid: CartesianGrid2D
    mask: Mask2D | None
    _harm: Diffusion2D

    def __init__(
        self,
        grid: CartesianGrid2D,
        mask: Mask2D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        # Critical: inner harmonic operator is ALWAYS mask=None, even
        # when BiharmonicDiffusion2D has a mask.  See class docstring.
        self._harm = Diffusion2D(grid=grid)

    def __call__(
        self,
        h: Float[Array, "Ny Nx"],
        kappa: float,
    ) -> Float[Array, "Ny Nx"]:
        """Apply biharmonic diffusion and return the tendency -kappa * nabla^4 h.

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar tracer field at T-points, shape ``[Ny, Nx]``.
        kappa : float
            Biharmonic diffusion coefficient (kappa >= 0 gives dissipation).

        Returns
        -------
        Float[Array, "Ny Nx"]
            Tendency -kappa * nabla^4 h at T-points, same shape as ``h``.
            Ghost cells are zero.  When ``self.mask`` is set, the final
            output is post-multiplied by ``mask.h``.
        """
        # First Laplacian pass: kappa=1.0 gives pure nabla^2 h.
        # Ghost ring of lap1 is zero (Dirichlet-0 BC on intermediate field).
        # Inner _harm is mask=None so the intermediate Laplacian is *not*
        # zeroed at dry cells — see class docstring for why.
        lap1 = self._harm(h, kappa=1.0)
        # Second Laplacian pass: nabla^2(nabla^2 h) = nabla^4 h
        lap2 = self._harm(lap1, kappa=1.0)
        out = -kappa * lap2
        if self.mask is not None:
            out = out * self.mask.h
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
    The ghost ring of the intermediate Laplacian is zero (Dirichlet-0).
    See :class:`BiharmonicDiffusion2D` notes for details.

    The inner harmonic :class:`Diffusion3D` is deliberately built with
    ``mask=None``; the outer mask is applied as a post-compute ``* mask.h``
    on the final tendency only — same exception as
    :class:`BiharmonicDiffusion2D`.

    Parameters
    ----------
    grid : CartesianGrid3D
        The underlying 3-D grid.
    mask : Mask3D or None, optional
        Optional 3-D land/ocean mask.  Applied final-only (see class
        docstring).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid3D, BiharmonicDiffusion3D
    >>> grid = CartesianGrid3D.from_interior(4, 8, 8, 1.0, 1.0, 1.0)
    >>> op = BiharmonicDiffusion3D(grid=grid)
    >>> h = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
    >>> tend = op(h, kappa=1e-6)
    >>> tend.shape
    (6, 10, 10)
    """

    grid: CartesianGrid3D
    mask: Mask3D | None
    _harm: Diffusion3D

    def __init__(
        self,
        grid: CartesianGrid3D,
        mask: Mask3D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        # Critical: inner harmonic operator is ALWAYS mask=None.
        self._harm = Diffusion3D(grid=grid)

    def __call__(
        self,
        h: Float[Array, "Nz Ny Nx"],
        kappa: float,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Apply horizontal biharmonic diffusion: -kappa * nabla^4_h h.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar tracer field at T-points, shape ``[Nz, Ny, Nx]``.
        kappa : float
            Biharmonic diffusion coefficient (kappa >= 0 gives dissipation).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Tendency -kappa * nabla^4_h h at T-points, same shape as ``h``.
            When ``self.mask`` is set, the final output is post-multiplied
            by ``mask.h``.
        """
        lap1 = self._harm(h, kappa=1.0)
        lap2 = self._harm(lap1, kappa=1.0)
        out = -kappa * lap2
        if self.mask is not None:
            out = out * self.mask.h
        return out
