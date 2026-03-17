"""Energy-conserving momentum advection operators for Arakawa C-grids.

Uses the vector-invariant (vortex-force) form of the horizontal momentum
equations:

    du/dt|adv[j, i+1/2] = +(ζ·v)[j, i+1/2] − ∂K/∂x[j, i+1/2]
    dv/dt|adv[j+1/2, i] = −(ζ·u)[j+1/2, i] − ∂K/∂y[j+1/2, i]

where ζ = ∂v/∂x − ∂u/∂y is the relative vorticity at X-points (NE corners),
K = ½(ū² + v̄²) is the kinetic energy at T-points, and the vorticity-flux
products (ζ·v) and (ζ·u) are computed using one of three discrete schemes
that each conserve either energy, enstrophy, or a combination of both.

References
----------
.. [1] Sadourny (1975) "The dynamics of finite-difference models of the
       shallow-water equations", J. Atmos. Sci., 32, 680–689.
.. [2] Arakawa and Lamb (1981) "A potential enstrophy and energy conserving
       scheme for the shallow water equations", Mon. Wea. Rev., 109, 18–36.
.. [3] Veros ocean model, ``veros/core/momentum.py``.
"""

from __future__ import annotations

import jax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid.grid import ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.operators.difference import Difference2D
from finitevolx._src.operators.interpolation import Interpolation2D


class MomentumAdvection2D(eqx.Module):
    """Energy-conserving momentum advection on a 2-D Arakawa C-grid.

    Computes the vortex-force form of the horizontal momentum advection:

        du/dt|adv[j, i+1/2] = +(ζ·v)[j, i+1/2] − ∂K/∂x[j, i+1/2]
        dv/dt|adv[j+1/2, i] = −(ζ·u)[j+1/2, i] − ∂K/∂y[j+1/2, i]

    where ζ = ∂v/∂x − ∂u/∂y is the relative vorticity at X-points and
    K = ½(ū² + v̄²) is the kinetic energy at T-points.

    Three vorticity-flux schemes are available via the ``scheme`` argument:

    * ``'energy'`` — Sadourny (1975) **E-scheme**: interpolate ζ to faces
      first, then multiply by the cross-face velocity.  Conserves total
      kinetic energy discretely for non-divergent flow.
    * ``'enstrophy'`` — Sadourny (1975) **Z-scheme**: interpolate the
      velocity to corners, multiply by ζ at corners, then interpolate the
      product to faces.  Conserves potential enstrophy.
    * ``'al'`` — **Arakawa-Lamb (1981)** blend: ⅓ energy + ⅔ enstrophy.
      Conserves both energy and enstrophy simultaneously.

    Parameters
    ----------
    grid : ArakawaCGrid2D
        The underlying 2-D grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import ArakawaCGrid2D, MomentumAdvection2D
    >>> grid = ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> madv = MomentumAdvection2D(grid=grid)
    >>> u = jnp.zeros((grid.Ny, grid.Nx))
    >>> v = jnp.zeros((grid.Ny, grid.Nx))
    >>> du, dv = madv(u, v)
    """

    grid: ArakawaCGrid2D
    diff: Difference2D
    interp: Interpolation2D

    def __init__(self, grid: ArakawaCGrid2D) -> None:
        self.grid = grid
        self.diff = Difference2D(grid=grid)
        self.interp = Interpolation2D(grid=grid)

    def _kinetic_energy_gradients(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Kinetic energy gradients (dK/dx at U-points, dK/dy at V-points).

        K[j, i] = ½ (u_T[j,i]² + v_T[j,i]²) at T-points
        dK/dx[j, i+1/2] = (K[j, i+1] - K[j, i]) / dx
        dK/dy[j+1/2, i] = (K[j+1, i] - K[j, i]) / dy
        """
        u_on_T = self.interp.U_to_T(u)
        v_on_T = self.interp.V_to_T(v)
        K = jnp.zeros_like(u)
        K = K.at[1:-1, 1:-1].set(
            0.5 * (u_on_T[1:-1, 1:-1] ** 2 + v_on_T[1:-1, 1:-1] ** 2)
        )
        return self.diff.diff_x_T_to_U(K), self.diff.diff_y_T_to_V(K)

    def _vorticity_flux_energy(
        self,
        zeta: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Sadourny E-scheme vorticity flux at (U-points, V-points).

        zv_u[j, i+1/2] = zeta_on_u * v_on_u
        zu_v[j+1/2, i] = zeta_on_v * u_on_v
        """
        zeta_on_u = self.interp.X_to_U(zeta)
        zeta_on_v = self.interp.X_to_V(zeta)
        v_on_u = self.interp.V_to_U(v)
        u_on_v = self.interp.U_to_V(u)
        zv_u = jnp.zeros_like(u)
        zu_v = jnp.zeros_like(v)
        zv_u = zv_u.at[1:-1, 1:-1].set(zeta_on_u[1:-1, 1:-1] * v_on_u[1:-1, 1:-1])
        zu_v = zu_v.at[1:-1, 1:-1].set(zeta_on_v[1:-1, 1:-1] * u_on_v[1:-1, 1:-1])
        return zv_u, zu_v

    def _vorticity_flux_enstrophy(
        self,
        zeta: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Sadourny Z-scheme vorticity flux at (U-points, V-points).

        zv_at_q = zeta * v_on_q at X-points, then avg to U-points
        zu_at_q = zeta * u_on_q at X-points, then avg to V-points
        """
        v_on_q = self.interp.V_to_X(v)
        u_on_q = self.interp.U_to_X(u)
        zv_at_q = jnp.zeros_like(u)
        zu_at_q = jnp.zeros_like(v)
        zv_at_q = zv_at_q.at[1:-1, 1:-1].set(zeta[1:-1, 1:-1] * v_on_q[1:-1, 1:-1])
        zu_at_q = zu_at_q.at[1:-1, 1:-1].set(zeta[1:-1, 1:-1] * u_on_q[1:-1, 1:-1])
        return self.interp.X_to_U(zv_at_q), self.interp.X_to_V(zu_at_q)

    def __call__(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        scheme: str = "energy",
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Momentum advection tendencies (du_adv, dv_adv).

        du_adv[j, i+1/2] = +(ζ·v)_u − ∂K/∂x
        dv_adv[j+1/2, i] = −(ζ·u)_v − ∂K/∂y

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.
        scheme : str
            Vorticity-flux scheme: ``'energy'`` (default), ``'enstrophy'``,
            or ``'al'`` (Arakawa-Lamb blend).

        Returns
        -------
        tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
            ``(du_adv, dv_adv)`` — tendencies at U-points and V-points,
            both zero in the ghost ring.

        Raises
        ------
        ValueError
            If ``scheme`` is not one of ``'energy'``, ``'enstrophy'``,
            or ``'al'``.
        """
        # zeta[j+1/2, i+1/2] = dv/dx - du/dy  at X-points
        zeta = self.diff.curl(u, v)
        # dK/dx at U-points, dK/dy at V-points
        dK_dx, dK_dy = self._kinetic_energy_gradients(u, v)

        if scheme == "energy":
            zv_u, zu_v = self._vorticity_flux_energy(zeta, u, v)
        elif scheme == "enstrophy":
            zv_u, zu_v = self._vorticity_flux_enstrophy(zeta, u, v)
        elif scheme == "al":
            # Arakawa-Lamb: 1/3 energy + 2/3 enstrophy
            alpha = 1.0 / 3.0
            zv_u_e, zu_v_e = self._vorticity_flux_energy(zeta, u, v)
            zv_u_s, zu_v_s = self._vorticity_flux_enstrophy(zeta, u, v)
            zv_u = jnp.zeros_like(u)
            zu_v = jnp.zeros_like(v)
            zv_u = zv_u.at[1:-1, 1:-1].set(
                alpha * zv_u_e[1:-1, 1:-1] + (1.0 - alpha) * zv_u_s[1:-1, 1:-1]
            )
            zu_v = zu_v.at[1:-1, 1:-1].set(
                alpha * zu_v_e[1:-1, 1:-1] + (1.0 - alpha) * zu_v_s[1:-1, 1:-1]
            )
        else:
            raise ValueError(
                f"Unknown scheme: {scheme!r}.  Choose 'energy', 'enstrophy', or 'al'."
            )

        du_adv = jnp.zeros_like(u)
        dv_adv = jnp.zeros_like(v)
        # du_adv[j, i+1/2] = +(zeta*v)_u - dK/dx
        du_adv = du_adv.at[2:-2, 2:-2].set(zv_u[2:-2, 2:-2] - dK_dx[2:-2, 2:-2])
        # dv_adv[j+1/2, i] = -(zeta*u)_v - dK/dy
        dv_adv = dv_adv.at[2:-2, 2:-2].set(-zu_v[2:-2, 2:-2] - dK_dy[2:-2, 2:-2])
        return du_adv, dv_adv


class MomentumAdvection3D(eqx.Module):
    """Energy-conserving momentum advection on a 3-D Arakawa C-grid.

    Applies ``MomentumAdvection2D``-equivalent stencils independently at
    each z-level of a ``[Nz, Ny, Nx]`` array.  The output write region is
    ``[1:-1, 2:-2, 2:-2]`` (all interior z-levels, strict horizontal interior),
    matching the ``Advection3D`` convention.  All other cells are zero.

    Three vorticity-flux schemes are available via the ``scheme`` argument:

    * ``'energy'`` — Sadourny (1975) E-scheme (default).
    * ``'enstrophy'`` — Sadourny (1975) Z-scheme.
    * ``'al'`` — Arakawa-Lamb (1981): ⅓ energy + ⅔ enstrophy.

    Parameters
    ----------
    grid : ArakawaCGrid3D
        The underlying 3-D grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import ArakawaCGrid3D, MomentumAdvection3D
    >>> grid = ArakawaCGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)
    >>> madv = MomentumAdvection3D(grid=grid)
    >>> u = jnp.zeros((grid.Nz, grid.Ny, grid.Nx))
    >>> v = jnp.zeros((grid.Nz, grid.Ny, grid.Nx))
    >>> du, dv = madv(u, v)
    """

    grid: ArakawaCGrid3D
    _madv2d: MomentumAdvection2D

    def __init__(self, grid: ArakawaCGrid3D) -> None:
        self.grid = grid
        self._madv2d = MomentumAdvection2D(grid=grid.horizontal_grid())

    def __call__(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        scheme: str = "energy",
    ) -> tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]:
        """Momentum advection tendencies over all z-levels.

        du_adv[k, j, i+1/2] = +(zeta*v)_u - dK/dx
        dv_adv[k, j+1/2, i] = -(zeta*u)_v - dK/dy

        Parameters
        ----------
        u : Float[Array, "Nz Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Nz Ny Nx"]
            y-velocity at V-points.
        scheme : str
            Vorticity-flux scheme: ``'energy'`` (default), ``'enstrophy'``,
            or ``'al'`` (Arakawa-Lamb blend).

        Returns
        -------
        tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]
            ``(du_adv, dv_adv)`` — tendencies at U-points and V-points,
            both zero in the ghost ring.

        Raises
        ------
        ValueError
            If ``scheme`` is not one of ``'energy'``, ``'enstrophy'``,
            or ``'al'``.
        """
        du_adv, dv_adv = jax.vmap(
            lambda u_k, v_k: self._madv2d(u_k, v_k, scheme=scheme)
        )(u, v)
        # Zero z-ghost slices.
        du_adv = du_adv.at[0].set(0.0).at[-1].set(0.0)
        dv_adv = dv_adv.at[0].set(0.0).at[-1].set(0.0)
        return du_adv, dv_adv
