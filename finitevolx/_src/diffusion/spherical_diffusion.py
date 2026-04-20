"""
Spherical-coordinate harmonic diffusion operators (flux form).

Computes ``∂h/∂t = ∇·(κ ∇h)`` on an Arakawa C-grid in spherical
coordinates::

    ∂h/∂t = 1/(R·cosφ) · [∂F_λ/∂λ + ∂(cosφ · F_φ)/∂φ]

with staggered face fluxes

    F_λ = κ · (1/(R·cosφ)) · ∂h/∂λ       at U-points
    F_φ = κ · (1/R)        · ∂h/∂φ       at V-points

Algorithm (2-D, spacings dlon, dlat in radians, planet radius R)
----------------------------------------------------------------
Step 1 — East-face flux at U-points (forward diff T → U)::

    F_λ[j, i+½] = κ · (h[j, i+1] − h[j, i]) / (R · cos(lat_U) · dlon)

Step 2 — North-face flux at V-points (forward diff T → V)::

    F_φ[j+½, i] = κ · (h[j+1, i] − h[j, i]) / (R · dlat)

Step 3 — Tendency at T-points (backward diff of fluxes with cosφ weight on
the meridional term)::

    dh[j, i] = 1/(R · cos(lat_T)) · [
        (F_λ[j, i+½] − F_λ[j, i−½]) / dlon
      + (cos(lat_V_N) · F_φ[j+½, i] − cos(lat_V_S) · F_φ[j−½, i]) / dlat
    ]

where ``cos(lat_V_N) = cos_lat_V[j, i]`` is the cos(lat) at the north
face of cell ``[j, i]`` and ``cos(lat_V_S) = cos_lat_V[j−1, i]`` the
south face.

At the equator (``cos(lat) = 1``) this reduces to the Cartesian
flux-form Laplacian ``∇·(κ∇h)`` with ``dx = R·dlon`` and
``dy = R·dlat``.

Boundary conditions
-------------------
Face fluxes at domain walls are zero by construction (west, east,
south, north) — see :mod:`finitevolx._src.diffusion.diffusion` for the
ghost-ring rationale.  This gives no-flux walls by default.

Pole handling
-------------
Any step that divides by ``cos(lat)`` uses :func:`_safe_div_cos` and
returns NaN where ``|cos(lat)| < 1e-12`` instead of ±inf.

Masking
-------
Like the Cartesian :class:`Diffusion2D`, the spherical operator applies
the three-step intermediate-flux masking pattern when a ``Mask2D`` /
``Mask3D`` is provided (`#209 <https://github.com/jejjohnson/finitevolX/issues/209>`_):

* ``F_λ *= mask.u``
* ``F_φ *= mask.v``
* tendency ``*= mask.h``

Per issue #209 Q2, spherical 2-D operators take a Cartesian ``Mask2D``
pending a dedicated ``SphericalMask2D`` follow-up.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from finitevolx._src.grid.spherical import SphericalGrid2D, SphericalGrid3D
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators._ghost import interior, zero_z_ghosts
from finitevolx._src.operators._utils import _safe_div_cos


def _spherical_diffusion_2d_impl(
    h: Float[Array, "Ny Nx"],
    kappa: float | Float[Array, "Ny Nx"],
    dlon: float,
    dlat: float,
    R: float,
    cos_lat_T: Float[Array, "Ny Nx"],
    cos_lat_U: Float[Array, "Ny Nx"],
    cos_lat_V: Float[Array, "Ny Nx"],
    mh: Bool[Array, "Ny Nx"] | Float[Array, "Ny Nx"] | None,
    mu: Bool[Array, "Ny Nx"] | Float[Array, "Ny Nx"] | None,
    mv: Bool[Array, "Ny Nx"] | Float[Array, "Ny Nx"] | None,
) -> Float[Array, "Ny Nx"]:
    """Shared 2-D spherical diffusion kernel with explicit raw-array masks."""
    kappa_arr = jnp.asarray(kappa)
    if kappa_arr.ndim >= 2:
        kappa_x = kappa_arr[1:-1, 1:-2]
        kappa_y = kappa_arr[1:-2, 1:-1]
    else:
        kappa_x = kappa_arr
        kappa_y = kappa_arr

    # Step 1 — east-face flux at U-points.
    # F_lam at U[j, i+1/2] = kappa * (h[j, i+1] - h[j, i]) / (R * cos(lat_U) * dlon)
    dh_dlon_raw = kappa_x * (h[1:-1, 2:-1] - h[1:-1, 1:-2])
    cos_U_face = cos_lat_U[1:-1, 1:-2]
    flux_x_interior = _safe_div_cos(dh_dlon_raw, cos_U_face, R * dlon)
    flux_x = jnp.zeros_like(h).at[1:-1, 1:-2].set(flux_x_interior)
    if mu is not None:
        flux_x = flux_x * mu

    # Step 2 — north-face flux at V-points.
    # F_phi at V[j+1/2, i] = kappa * (h[j+1, i] - h[j, i]) / (R * dlat)
    flux_y_interior = kappa_y * (h[2:-1, 1:-1] - h[1:-2, 1:-1]) / (R * dlat)
    flux_y = jnp.zeros_like(h).at[1:-2, 1:-1].set(flux_y_interior)
    if mv is not None:
        flux_y = flux_y * mv

    # Step 3 — tendency at T-points (spherical flux-divergence).
    cos_V_N = cos_lat_V[1:-1, 1:-1]  # north face of cell [j, i]
    cos_V_S = cos_lat_V[:-2, 1:-1]  # south face of cell [j, i]
    cos_T_c = cos_lat_T[1:-1, 1:-1]

    du = (flux_x[1:-1, 1:-1] - flux_x[1:-1, :-2]) / dlon
    dv = (cos_V_N * flux_y[1:-1, 1:-1] - cos_V_S * flux_y[:-2, 1:-1]) / dlat
    out = interior(_safe_div_cos(du + dv, cos_T_c, R), h)

    if mh is not None:
        out = out * mh

    return out


def _spherical_diffusion_2d_fluxes_impl(
    h: Float[Array, "Ny Nx"],
    kappa: float | Float[Array, "Ny Nx"],
    dlon: float,
    dlat: float,
    R: float,
    cos_lat_U: Float[Array, "Ny Nx"],
    mu: Bool[Array, "Ny Nx"] | Float[Array, "Ny Nx"] | None,
    mv: Bool[Array, "Ny Nx"] | Float[Array, "Ny Nx"] | None,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Shared diagnostic-flux kernel for :class:`SphericalDiffusion2D`."""
    kappa_arr = jnp.asarray(kappa)
    if kappa_arr.ndim >= 2:
        kappa_x = kappa_arr[1:-1, 1:-2]
        kappa_y = kappa_arr[1:-2, 1:-1]
    else:
        kappa_x = kappa_arr
        kappa_y = kappa_arr

    dh_dlon_raw = kappa_x * (h[1:-1, 2:-1] - h[1:-1, 1:-2])
    cos_U_face = cos_lat_U[1:-1, 1:-2]
    flux_x_interior = _safe_div_cos(dh_dlon_raw, cos_U_face, R * dlon)
    flux_x = jnp.zeros_like(h).at[1:-1, 1:-2].set(flux_x_interior)
    if mu is not None:
        flux_x = flux_x * mu

    flux_y_interior = kappa_y * (h[2:-1, 1:-1] - h[1:-2, 1:-1]) / (R * dlat)
    flux_y = jnp.zeros_like(h).at[1:-2, 1:-1].set(flux_y_interior)
    if mv is not None:
        flux_y = flux_y * mv

    return flux_x, flux_y


class SphericalDiffusion2D(eqx.Module):
    """Horizontal tracer diffusion (flux form) on a 2-D spherical C-grid.

    Computes ``∂h/∂t = ∇·(κ ∇h)`` at T-points from staggered face
    fluxes, applying the cos(lat) metric terms inside the zonal flux
    and the meridional flux-divergence (see module docstring for the
    full stencil).

    Parameters
    ----------
    grid : SphericalGrid2D
        The underlying 2-D spherical Arakawa C-grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided, the three-step
        intermediate-flux masking pattern is applied inside both
        :meth:`__call__` and :meth:`fluxes`:

        * ``flux_x *= mask.u`` at the U-face stage,
        * ``flux_y *= mask.v`` at the V-face stage,
        * tendency ``*= mask.h`` on the final output (``__call__`` only).

        Per #209 Q2, this is a Cartesian ``Mask2D`` pending a dedicated
        ``SphericalMask2D`` follow-up.  ``None`` (default) gives the
        unmasked behaviour.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import SphericalGrid2D, SphericalDiffusion2D
    >>> grid = SphericalGrid2D.from_interior(
    ...     nx_interior=16,
    ...     ny_interior=8,
    ...     lon_range=(0.0, 360.0),
    ...     lat_range=(-40.0, 40.0),
    ...     R=6.371e6,
    ... )
    >>> op = SphericalDiffusion2D(grid=grid)
    >>> h = jnp.ones((grid.Ny, grid.Nx))
    >>> tend = op(h, kappa=1e3)
    >>> tend.shape
    (10, 18)
    """

    grid: SphericalGrid2D
    mask: Mask2D | None = None

    def __call__(
        self,
        h: Float[Array, "Ny Nx"],
        kappa: float | Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Harmonic diffusion tendency at T-points.

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Tracer field at T-points.
        kappa : float or Float[Array, "Ny Nx"]
            Diffusion coefficient (scalar or T-point field).  When an
            array, the value at the source (western/southern) T-cell
            is used for each face flux.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Diffusion tendency at T-points.  Ghost ring is zero.
        """
        mh = None if self.mask is None else self.mask.h
        mu = None if self.mask is None else self.mask.u
        mv = None if self.mask is None else self.mask.v
        return _spherical_diffusion_2d_impl(
            h,
            kappa,
            self.grid.dlon,
            self.grid.dlat,
            self.grid.R,
            self.grid.cos_lat_T,
            self.grid.cos_lat_U,
            self.grid.cos_lat_V,
            mh=mh,
            mu=mu,
            mv=mv,
        )

    def fluxes(
        self,
        h: Float[Array, "Ny Nx"],
        kappa: float | Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Diagnostic diffusive face fluxes at U- and V-points.

        Returns the metric-scaled east-face and north-face fluxes before
        the spherical divergence step::

            F_λ[j, i+½] = κ · (h[j, i+1] − h[j, i]) / (R · cos(lat_U) · dlon)
            F_φ[j+½, i] = κ · (h[j+1, i] − h[j, i]) / (R · dlat)

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Tracer field at T-points.
        kappa : float or Float[Array, "Ny Nx"]
            Diffusion coefficient.

        Returns
        -------
        tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
            ``(flux_x, flux_y)``.  Masked by ``mask.u`` / ``mask.v``
            when ``self.mask`` is set.
        """
        mu = None if self.mask is None else self.mask.u
        mv = None if self.mask is None else self.mask.v
        return _spherical_diffusion_2d_fluxes_impl(
            h,
            kappa,
            self.grid.dlon,
            self.grid.dlat,
            self.grid.R,
            self.grid.cos_lat_U,
            mu=mu,
            mv=mv,
        )


class SphericalDiffusion3D(eqx.Module):
    """Horizontal tracer diffusion on a 3-D spherical Arakawa C-grid.

    Applies :class:`SphericalDiffusion2D` independently at each z-level
    via ``eqx.filter_vmap``.  The 3-D field shape is ``[Nz, Ny, Nx]``.

    Parameters
    ----------
    grid : SphericalGrid3D
    mask : Mask3D or None, optional
        Optional 3-D land/ocean mask.  When provided, the intermediate
        flux masking pattern from :class:`SphericalDiffusion2D` is
        applied at every z-level with per-z slices of ``mask.h``,
        ``mask.u``, ``mask.v`` — same approach as :class:`Diffusion3D`.
    """

    grid: SphericalGrid3D
    mask: Mask3D | None = None

    def __call__(
        self,
        h: Float[Array, "Nz Ny Nx"],
        kappa: float | Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """Diffusion tendency at T-points over all z-levels."""
        dlon, dlat, R = self.grid.dlon, self.grid.dlat, self.grid.R
        cos_T = self.grid.cos_lat_T
        cos_U = self.grid.cos_lat_U
        cos_V = self.grid.cos_lat_V

        kappa_arr = jnp.asarray(kappa)
        kappa_ax = 0 if kappa_arr.ndim >= 3 else None

        if self.mask is None:

            def _apply(h_k, kap_k):
                return _spherical_diffusion_2d_impl(
                    h_k,
                    kap_k,
                    dlon,
                    dlat,
                    R,
                    cos_T,
                    cos_U,
                    cos_V,
                    mh=None,
                    mu=None,
                    mv=None,
                )

            out = eqx.filter_vmap(_apply, in_axes=(0, kappa_ax))(h, kappa_arr)
            return zero_z_ghosts(out)

        mh = self.mask.h
        mu = self.mask.u
        mv = self.mask.v

        def _apply_masked(h_k, kap_k, mh_k, mu_k, mv_k):
            return _spherical_diffusion_2d_impl(
                h_k,
                kap_k,
                dlon,
                dlat,
                R,
                cos_T,
                cos_U,
                cos_V,
                mh=mh_k,
                mu=mu_k,
                mv=mv_k,
            )

        out = eqx.filter_vmap(_apply_masked, in_axes=(0, kappa_ax, 0, 0, 0))(
            h, kappa_arr, mh, mu, mv
        )
        return zero_z_ghosts(out)

    def fluxes(
        self,
        h: Float[Array, "Nz Ny Nx"],
        kappa: float | Float[Array, "Nz Ny Nx"],
    ) -> tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]:
        """Diagnostic diffusive face fluxes at U- and V-points, all z."""
        dlon, dlat, R = self.grid.dlon, self.grid.dlat, self.grid.R
        cos_U = self.grid.cos_lat_U

        kappa_arr = jnp.asarray(kappa)
        kappa_ax = 0 if kappa_arr.ndim >= 3 else None

        if self.mask is None:

            def _apply(h_k, kap_k):
                return _spherical_diffusion_2d_fluxes_impl(
                    h_k, kap_k, dlon, dlat, R, cos_U, mu=None, mv=None
                )

            fx, fy = eqx.filter_vmap(_apply, in_axes=(0, kappa_ax))(h, kappa_arr)
            return zero_z_ghosts(fx), zero_z_ghosts(fy)

        mu = self.mask.u
        mv = self.mask.v

        def _apply_masked(h_k, kap_k, mu_k, mv_k):
            return _spherical_diffusion_2d_fluxes_impl(
                h_k, kap_k, dlon, dlat, R, cos_U, mu=mu_k, mv=mv_k
            )

        fx, fy = eqx.filter_vmap(_apply_masked, in_axes=(0, kappa_ax, 0, 0))(
            h, kappa_arr, mu, mv
        )
        return zero_z_ghosts(fx), zero_z_ghosts(fy)
