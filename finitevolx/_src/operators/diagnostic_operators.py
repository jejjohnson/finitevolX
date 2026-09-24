"""Class forms of the diagnostic operators, with operator-attribute masks.

The functional diagnostics in :mod:`finitevolx._src.operators.diagnostics`
are mask-free Layer-2 helpers.  The classes here are their Layer-3
counterparts: the mask is set once at construction and every method zeroes
the dry cells of its output stagger, following the rule in ``docs/masks.md``
("one output -> one mask field, chosen by the output stagger"):

==================================  ======  =========================
Method                              Output  Mask field
==================================  ======  =========================
``Energetics2D.*``                  T       ``mask.h``
``Strain2D.shear``                  X       ``mask.xy_corner_strict``
``Strain2D.tensor``                 T       ``mask.h``
``Strain2D.magnitude_squared``      T       ``mask.h``
``Strain2D.okubo_weiss``            T       ``mask.h``
``QGPotentialVorticity2D.*``        T       ``mask.h``
==================================  ======  =========================

With ``mask=None`` every method returns exactly what the functional form
(or the documented composition of functional forms) returns.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from finitevolx._src.grid.cartesian import CartesianGrid2D
from finitevolx._src.mask import Mask2D
from finitevolx._src.operators.diagnostics import (
    available_potential_energy,
    bernoulli_potential,
    kinetic_energy,
    okubo_weiss,
    potential_vorticity_multilayer,
    qg_potential_vorticity,
    relative_vorticity_cgrid,
    shear_strain,
    strain_magnitude_squared,
    stretching_term,
    tensor_strain,
)
from finitevolx._src.operators.interpolation import Interpolation2D
from finitevolx._src.utils.constants import GRAVITY


class Energetics2D(eqx.Module):
    """Energy diagnostics at T-points on a 2-D Arakawa C-grid.

    Class form of :func:`~finitevolx.kinetic_energy`,
    :func:`~finitevolx.bernoulli_potential` and
    :func:`~finitevolx.available_potential_energy`.  Every method returns a
    T-point field; when ``mask`` is set, dry T-cells are zeroed via
    ``* mask.h`` (post-compute, Pattern 1 in ``docs/masks.md``).

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  ``None`` (default) returns exactly the
        functional forms' output.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid2D, Energetics2D
    >>> grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> u = jnp.ones((grid.Ny, grid.Nx))
    >>> ke = Energetics2D(grid=grid).kinetic_energy(u, 0.0 * u)
    """

    grid: CartesianGrid2D
    mask: Mask2D | None = None

    def _mask_h(self, out: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Zero dry T-cells: out[j, i] = out[j, i] * mask.h[j, i]."""
        if self.mask is None:
            return out
        return out * self.mask.h

    def kinetic_energy(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Kinetic energy at T-points.

        ke[j, i] = 1/2 * (u2_on_T[j, i] + v2_on_T[j, i])

        u2_on_T[j, i] = 1/2 * (u[j, i+1/2]^2 + u[j, i-1/2]^2)
        v2_on_T[j, i] = 1/2 * (v[j+1/2, i]^2 + v[j-1/2, i]^2)

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Kinetic energy at T-points, zero in the ghost ring and, when
            ``self.mask`` is set, at dry T-cells.
        """
        return self._mask_h(kinetic_energy(u, v))

    def bernoulli_potential(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        gravity: float = GRAVITY,
    ) -> Float[Array, "Ny Nx"]:
        """Bernoulli potential at T-points.

        p[j, i] = ke[j, i] + gravity * h[j, i]

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Layer thickness at T-points.
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.
        gravity : float, optional
            Gravitational acceleration.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Bernoulli potential at T-points, zero in the ghost ring and,
            when ``self.mask`` is set, at dry T-cells.
        """
        return self._mask_h(bernoulli_potential(h, u, v, gravity))

    def available_potential_energy(
        self,
        h: Float[Array, "Ny Nx"],
        H: Float[Array, "Ny Nx"],
        g_prime: float,
    ) -> Float[Array, "Ny Nx"]:
        """Available potential energy at T-points.

        ape[j, i] = 1/2 * g_prime * (h[j, i] - H[j, i])^2

        Pointwise, so -- like the functional form -- the ghost ring holds
        the pointwise value unless a mask zeroes it.

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Layer thickness at T-points.
        H : Float[Array, "Ny Nx"]
            Reference layer thickness at T-points.
        g_prime : float
            Reduced gravity.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Available potential energy at T-points; zero at dry T-cells when
            ``self.mask`` is set.
        """
        return self._mask_h(available_potential_energy(h, H, g_prime))


class Strain2D(eqx.Module):
    """Strain diagnostics on a 2-D Arakawa C-grid.

    Class form of :func:`~finitevolx.shear_strain`,
    :func:`~finitevolx.tensor_strain`,
    :func:`~finitevolx.strain_magnitude_squared` and
    :func:`~finitevolx.okubo_weiss`.

    The shear strain lives at X-points and the tensor strain at T-points.
    :meth:`magnitude_squared` and :meth:`okubo_weiss` combine them at
    **T-points**: the X-point shear (and vorticity) are averaged to T-points
    with :meth:`Interpolation2D.X_to_T` first.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  Each method zeroes the dry cells of its
        output stagger (``mask.xy_corner_strict`` for :meth:`shear`,
        ``mask.h`` otherwise).  For the T-point combinations the X-point
        intermediates are masked first (pass-down, Pattern 2 in
        ``docs/masks.md``), so a coastal T-cell averages in zero from its
        dry corners.  ``None`` (default) returns exactly the functional
        forms' output.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid2D, Strain2D
    >>> grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> u = jnp.ones((grid.Ny, grid.Nx))
    >>> ow = Strain2D(grid=grid).okubo_weiss(u, 0.0 * u)
    """

    grid: CartesianGrid2D
    mask: Mask2D | None
    interp: Interpolation2D

    def __init__(
        self,
        grid: CartesianGrid2D,
        mask: Mask2D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        self.interp = Interpolation2D(grid=grid, mask=mask)

    def shear(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Shear strain at X-points.

        ss[j+1/2, i+1/2] = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
                         + (u[j+1, i+1/2] - u[j, i+1/2]) / dy

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Shear strain at X-points, zero in the ghost ring and, when
            ``self.mask`` is set, at dry X-corners.
        """
        out = shear_strain(u, v, self.grid.dx, self.grid.dy)
        if self.mask is not None:
            out = out * self.mask.xy_corner_strict
        return out

    def tensor(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Tensor (normal) strain at T-points.

        sn[j, i] = (u[j, i+1/2] - u[j, i-1/2]) / dx
                 - (v[j+1/2, i] - v[j-1/2, i]) / dy

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Tensor strain at T-points, zero in the ghost ring and, when
            ``self.mask`` is set, at dry T-cells.
        """
        out = tensor_strain(u, v, self.grid.dx, self.grid.dy)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def magnitude_squared(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Squared strain magnitude at T-points.

        sigma2[j, i] = sn[j, i]^2 + ss_on_T[j, i]^2

        ss_on_T[j, i] = 1/4 * (ss[j+1/2, i+1/2] + ss[j-1/2, i+1/2]
                             + ss[j+1/2, i-1/2] + ss[j-1/2, i-1/2])

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Squared strain magnitude at T-points, zero in the ghost ring
            and, when ``self.mask`` is set, at dry T-cells.
        """
        sn = self.tensor(u, v)
        ss_on_T = self.interp.X_to_T(self.shear(u, v))
        out = strain_magnitude_squared(sn, ss_on_T)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def okubo_weiss(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Okubo-Weiss parameter at T-points.

        ow[j, i] = sn[j, i]^2 + ss_on_T[j, i]^2 - omega_on_T[j, i]^2

        where the X-point shear strain and relative vorticity

        omega[j+1/2, i+1/2] = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
                            - (u[j+1, i+1/2] - u[j, i+1/2]) / dy

        are averaged to T-points over their four surrounding corners, as in
        :meth:`magnitude_squared`.  Positive = strain-dominated, negative =
        vorticity-dominated.

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Okubo-Weiss parameter at T-points, zero in the ghost ring and,
            when ``self.mask`` is set, at dry T-cells.
        """
        sn = self.tensor(u, v)
        ss_on_T = self.interp.X_to_T(self.shear(u, v))
        omega = relative_vorticity_cgrid(u, v, self.grid.dx, self.grid.dy)
        if self.mask is not None:
            omega = omega * self.mask.xy_corner_strict
        omega_on_T = self.interp.X_to_T(omega)
        out = okubo_weiss(sn, ss_on_T, omega_on_T)
        if self.mask is not None:
            out = out * self.mask.h
        return out


class QGPotentialVorticity2D(eqx.Module):
    """Quasi-geostrophic potential vorticity at T-points.

    Class form of :func:`~finitevolx.qg_potential_vorticity`,
    :func:`~finitevolx.stretching_term` and
    :func:`~finitevolx.potential_vorticity_multilayer`.  Every method
    returns a T-point field (``[Ny, Nx]`` for one layer, ``[nl, Ny, Nx]``
    for the multilayer methods); when ``mask`` is set, dry T-cells are
    zeroed via ``* mask.h`` (post-compute, Pattern 1 in ``docs/masks.md``),
    broadcast over the layer axis.

    The Laplacian stencil reads ``psi`` at the four neighbours of each
    T-cell, so at a coastal cell it sees whatever ``psi`` holds on land.
    Streamfunctions from a masked elliptic solve are zero on land, which is
    the usual no-normal-flow condition.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  ``None`` (default) returns exactly the
        functional forms' output.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid2D, QGPotentialVorticity2D
    >>> grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> psi = jnp.zeros((grid.Ny, grid.Nx))
    >>> y = jnp.zeros((grid.Ny, grid.Nx))
    >>> q = QGPotentialVorticity2D(grid=grid)(psi, 1e-4, 0.0, y, 0.0)
    """

    grid: CartesianGrid2D
    mask: Mask2D | None = None

    def _mask_h(self, out: Float[Array, "... Ny Nx"]) -> Float[Array, "... Ny Nx"]:
        """Zero dry T-cells: out[..., j, i] = out[..., j, i] * mask.h[j, i]."""
        if self.mask is None:
            return out
        return out * self.mask.h

    def __call__(
        self,
        psi: Float[Array, "Ny Nx"],
        f0: float,
        beta: float,
        y: Float[Array, "Ny Nx"],
        y0: float,
    ) -> Float[Array, "Ny Nx"]:
        """Single-layer QG potential vorticity.

        q[j, i] = lap_psi[j, i] / f0 + beta * (y[j, i] - y0) / f0

        lap_psi[j, i] = (psi[j, i+1] - 2 * psi[j, i] + psi[j, i-1]) / dx^2
                      + (psi[j+1, i] - 2 * psi[j, i] + psi[j-1, i]) / dy^2

        Parameters
        ----------
        psi : Float[Array, "Ny Nx"]
            Streamfunction at T-points.
        f0 : float
            Reference Coriolis parameter.
        beta : float
            Meridional gradient of the Coriolis parameter.
        y : Float[Array, "Ny Nx"]
            Meridional coordinate at T-points.
        y0 : float
            Reference latitude.

        Returns
        -------
        Float[Array, "Ny Nx"]
            QG potential vorticity at T-points, zero in the ghost ring and,
            when ``self.mask`` is set, at dry T-cells.
        """
        return self._mask_h(
            qg_potential_vorticity(psi, f0, beta, self.grid.dx, self.grid.dy, y, y0)
        )

    def stretching(
        self,
        A: Float[Array, "nl nl"],
        psi: Float[Array, "nl Ny Nx"],
    ) -> Float[Array, "nl Ny Nx"]:
        """Cross-layer stretching term.

        s[k, j, i] = sum_m A[k, m] * psi[m, j, i]

        Parameters
        ----------
        A : Float[Array, "nl nl"]
            Coupling (stretching) matrix.
        psi : Float[Array, "nl Ny Nx"]
            Streamfunction at T-points for all layers.

        Returns
        -------
        Float[Array, "nl Ny Nx"]
            Stretching contribution at T-points, zero in the ghost ring and,
            when ``self.mask`` is set, at dry T-cells of every layer.
        """
        return self._mask_h(stretching_term(A, psi))

    def multilayer(
        self,
        psi: Float[Array, "nl Ny Nx"],
        A: Float[Array, "nl nl"],
        f0: float,
        beta: float,
        y: Float[Array, "Ny Nx"],
        y0: float,
    ) -> Float[Array, "nl Ny Nx"]:
        """Multi-layer QG potential vorticity.

        q[k, j, i] = lap_psi[k, j, i] / f0 + beta * (y[j, i] - y0) / f0
                   - sum_m A[k, m] * psi[m, j, i]

        Parameters
        ----------
        psi : Float[Array, "nl Ny Nx"]
            Streamfunction at T-points for all layers.
        A : Float[Array, "nl nl"]
            Coupling (stretching) matrix.
        f0 : float
            Reference Coriolis parameter.
        beta : float
            Meridional gradient of the Coriolis parameter.
        y : Float[Array, "Ny Nx"]
            Meridional coordinate at T-points.
        y0 : float
            Reference latitude.

        Returns
        -------
        Float[Array, "nl Ny Nx"]
            QG potential vorticity at T-points, zero in the ghost ring and,
            when ``self.mask`` is set, at dry T-cells of every layer.
        """
        return self._mask_h(
            potential_vorticity_multilayer(
                psi, A, f0, beta, self.grid.dx, self.grid.dy, y, y0
            )
        )
