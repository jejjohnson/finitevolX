"""Energy-conserving momentum advection operators on spherical C-grids.

Spherical counterpart of :mod:`finitevolx._src.diffusion.momentum`.

Uses the vector-invariant (vortex-force) form of the horizontal
momentum equations:

    du/dt|adv[j, i+½] = +(ζ·v)_u − ∂K/∂λ_spherical
    dv/dt|adv[j+½, i] = −(ζ·u)_v − ∂K/∂φ_spherical

where ``ζ`` is the spherical relative vorticity at X-points (NE
corners), ``K = ½ (u_T² + v_T²)`` is the kinetic energy at T-points,
and the vorticity-flux products (ζ·v) / (ζ·u) are computed with one of
three discrete schemes (Sadourny E/Z, Arakawa–Lamb 1/3–2/3 blend).

The three flux schemes are **coordinate-independent** — they involve
only interpolation and multiplication.  What changes on a sphere is:

* the curl is computed with :class:`SphericalVorticity2D`, which
  applies the ``1/(R·cosφ)`` metric to ``∂v/∂λ − ∂(u·cosφ)/∂φ``;
* the kinetic-energy gradients at U-/V-points use
  :class:`SphericalDifference2D` so ``∂K/∂λ`` picks up
  ``1/(R·cos(lat_U))`` and ``∂K/∂φ`` picks up ``1/R``.

At the equator (``cos(lat) = 1``) the spherical operator reduces to
the Cartesian :class:`~finitevolx.MomentumAdvection2D` with
``dx = R·dlon`` and ``dy = R·dlat``.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from finitevolx._src.grid.spherical import SphericalGrid2D, SphericalGrid3D
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators._ghost import interior, zero_z_ghosts
from finitevolx._src.operators.interpolation import Interpolation2D
from finitevolx._src.operators.spherical_compound import SphericalVorticity2D
from finitevolx._src.operators.spherical_difference import SphericalDifference2D


class SphericalMomentumAdvection2D(eqx.Module):
    """Energy-conserving momentum advection on a 2-D spherical C-grid.

    Computes the vortex-force form on a sphere:

        du/dt|adv[j, i+½] = +(ζ·v)_u − ∂K/∂λ_spherical
        dv/dt|adv[j+½, i] = −(ζ·u)_v − ∂K/∂φ_spherical

    where ``ζ = 1/(R·cosφ)·[∂v/∂λ − ∂(u·cosφ)/∂φ]`` at X-points and
    ``K = ½(u_T² + v_T²)`` at T-points.

    Three vorticity-flux schemes are available via the ``scheme`` argument:

    * ``'energy'`` — Sadourny (1975) **E-scheme**: interpolate ζ to faces
      first, then multiply by the cross-face velocity.
    * ``'enstrophy'`` — Sadourny (1975) **Z-scheme**: interpolate the
      velocity to corners, multiply by ζ at corners, then interpolate the
      product to faces.
    * ``'al'`` — **Arakawa-Lamb (1981)** blend: ⅓ energy + ⅔ enstrophy.

    Parameters
    ----------
    grid : SphericalGrid2D
        The underlying 2-D spherical grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  Pass-down pattern — the internal
        :class:`SphericalDifference2D` (for ``∂K/∂λ``, ``∂K/∂φ``),
        :class:`Interpolation2D` (for the vorticity-flux averages and
        ``u_T``/``v_T``), and :class:`SphericalVorticity2D` (for ``ζ``)
        are constructed with the same mask, so every intermediate
        staggered field carries the correct post-compute zero.  The
        final tendencies are additionally post-multiplied by
        ``mask.u`` / ``mask.v``.  Per #209 Q2 a Cartesian ``Mask2D`` is
        used pending a dedicated ``SphericalMask2D`` follow-up.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import SphericalGrid2D, SphericalMomentumAdvection2D
    >>> grid = SphericalGrid2D.from_interior(
    ...     nx_interior=16,
    ...     ny_interior=8,
    ...     lon_range=(0.0, 360.0),
    ...     lat_range=(-40.0, 40.0),
    ... )
    >>> madv = SphericalMomentumAdvection2D(grid=grid)
    >>> u = jnp.zeros((grid.Ny, grid.Nx))
    >>> v = jnp.zeros((grid.Ny, grid.Nx))
    >>> du, dv = madv(u, v)
    """

    grid: SphericalGrid2D
    mask: Mask2D | None
    diff: SphericalDifference2D
    interp: Interpolation2D
    vort: SphericalVorticity2D

    def __init__(
        self,
        grid: SphericalGrid2D,
        mask: Mask2D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        self.diff = SphericalDifference2D(grid=grid, mask=mask)
        self.interp = Interpolation2D(grid=grid, mask=mask)
        self.vort = SphericalVorticity2D(grid=grid, mask=mask)

    def _kinetic_energy_gradients(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Spherical KE gradients (∂K/∂λ at U-points, ∂K/∂φ at V-points).

        K[j, i] = ½ (u_T² + v_T²)   at T-points
        ∂K/∂λ[j, i+½] = (K[j, i+1] − K[j, i]) / (R · cos(lat_U) · dlon)
        ∂K/∂φ[j+½, i] = (K[j+1, i] − K[j, i]) / (R · dlat)
        """
        u_on_T = self.interp.U_to_T(u)
        v_on_T = self.interp.V_to_T(v)
        K = interior(0.5 * (u_on_T[1:-1, 1:-1] ** 2 + v_on_T[1:-1, 1:-1] ** 2), u)
        return self.diff.diff_lon_T_to_U(K), self.diff.diff_lat_T_to_V(K)

    def _vorticity_flux_energy(
        self,
        zeta: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Sadourny E-scheme vorticity flux at (U-points, V-points)."""
        zeta_on_u = self.interp.X_to_U(zeta)
        zeta_on_v = self.interp.X_to_V(zeta)
        v_on_u = self.interp.V_to_U(v)
        u_on_v = self.interp.U_to_V(u)
        zv_u = interior(zeta_on_u[1:-1, 1:-1] * v_on_u[1:-1, 1:-1], u)
        zu_v = interior(zeta_on_v[1:-1, 1:-1] * u_on_v[1:-1, 1:-1], v)
        return zv_u, zu_v

    def _vorticity_flux_enstrophy(
        self,
        zeta: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Sadourny Z-scheme vorticity flux at (U-points, V-points)."""
        v_on_q = self.interp.V_to_X(v)
        u_on_q = self.interp.U_to_X(u)
        zv_at_q = interior(zeta[1:-1, 1:-1] * v_on_q[1:-1, 1:-1], u)
        zu_at_q = interior(zeta[1:-1, 1:-1] * u_on_q[1:-1, 1:-1], v)
        return self.interp.X_to_U(zv_at_q), self.interp.X_to_V(zu_at_q)

    def __call__(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        scheme: str = "energy",
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Momentum advection tendencies on a sphere.

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            Zonal velocity at U-points.
        v : Float[Array, "Ny Nx"]
            Meridional velocity at V-points.
        scheme : str
            Vorticity-flux scheme: ``'energy'`` (default), ``'enstrophy'``,
            or ``'al'`` (Arakawa-Lamb blend).

        Returns
        -------
        tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
            ``(du_adv, dv_adv)`` — tendencies at U- and V-points, zero
            in the ghost ring.

        Raises
        ------
        ValueError
            If ``scheme`` is not one of ``'energy'``, ``'enstrophy'``,
            or ``'al'``.
        """
        zeta = self.vort.relative_vorticity(u, v)
        dK_dlon, dK_dlat = self._kinetic_energy_gradients(u, v)

        if scheme == "energy":
            zv_u, zu_v = self._vorticity_flux_energy(zeta, u, v)
        elif scheme == "enstrophy":
            zv_u, zu_v = self._vorticity_flux_enstrophy(zeta, u, v)
        elif scheme == "al":
            alpha = 1.0 / 3.0
            zv_u_e, zu_v_e = self._vorticity_flux_energy(zeta, u, v)
            zv_u_s, zu_v_s = self._vorticity_flux_enstrophy(zeta, u, v)
            zv_u = interior(
                alpha * zv_u_e[1:-1, 1:-1] + (1.0 - alpha) * zv_u_s[1:-1, 1:-1], u
            )
            zu_v = interior(
                alpha * zu_v_e[1:-1, 1:-1] + (1.0 - alpha) * zu_v_s[1:-1, 1:-1], v
            )
        else:
            raise ValueError(
                f"Unknown scheme: {scheme!r}.  Choose 'energy', 'enstrophy', or 'al'."
            )

        # du_adv at U[j, i+1/2] = +(zeta * v)_u - dK/dlon
        du_adv = interior(zv_u[2:-2, 2:-2] - dK_dlon[2:-2, 2:-2], u, ghost=2)
        # dv_adv at V[j+1/2, i] = -(zeta * u)_v - dK/dlat
        dv_adv = interior(-zu_v[2:-2, 2:-2] - dK_dlat[2:-2, 2:-2], v, ghost=2)

        if self.mask is not None:
            du_adv = du_adv * self.mask.u
            dv_adv = dv_adv * self.mask.v
        return du_adv, dv_adv


class SphericalMomentumAdvection3D(eqx.Module):
    """Energy-conserving momentum advection on a 3-D spherical C-grid.

    Applies :class:`SphericalMomentumAdvection2D`-equivalent stencils
    independently at each z-level of a ``[Nz, Ny, Nx]`` array.  The
    output write region is ``[1:-1, 2:-2, 2:-2]`` (all interior z-levels,
    strict horizontal interior), matching the 3-D advection convention.

    Parameters
    ----------
    grid : SphericalGrid3D
    mask : Mask3D or None, optional
        Optional 3-D land/ocean mask.  Pattern A (post-compute) — the
        inner :class:`SphericalMomentumAdvection2D` is always built
        ``mask=None`` and the vmap'd 3-D result is post-multiplied by
        ``mask.u`` / ``mask.v``.
    """

    grid: SphericalGrid3D
    mask: Mask3D | None
    _madv2d: SphericalMomentumAdvection2D

    def __init__(
        self,
        grid: SphericalGrid3D,
        mask: Mask3D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        self._madv2d = SphericalMomentumAdvection2D(grid=grid.horizontal_grid())

    def __call__(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        scheme: str = "energy",
    ) -> tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]:
        """Spherical momentum advection tendencies over all z-levels.

        Parameters
        ----------
        u : Float[Array, "Nz Ny Nx"]
            Zonal velocity at U-points.
        v : Float[Array, "Nz Ny Nx"]
            Meridional velocity at V-points.
        scheme : str
            Vorticity-flux scheme: ``'energy'`` (default), ``'enstrophy'``,
            or ``'al'``.

        Returns
        -------
        tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]
            ``(du_adv, dv_adv)`` at U- and V-points; zero at z-ghost
            slices and at outermost horizontal ring.
        """
        du_adv, dv_adv = eqx.filter_vmap(
            lambda u_k, v_k: self._madv2d(u_k, v_k, scheme=scheme)
        )(u, v)
        du_adv = zero_z_ghosts(du_adv)
        dv_adv = zero_z_ghosts(dv_adv)
        if self.mask is not None:
            du_adv = du_adv * self.mask.u
            dv_adv = dv_adv * self.mask.v
        return du_adv, dv_adv
