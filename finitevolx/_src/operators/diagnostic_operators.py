"""Class forms of the diagnostic operators, with operator-attribute masks.

The functional diagnostics in :mod:`finitevolx._src.operators.diagnostics`
are mask-free Layer-2 helpers.  The classes here are their Layer-3
counterparts: the mask is set once at construction and every method zeroes
the dry cells of its output stagger, following the rule in ``docs/masks.md``
("one output -> one mask field, chosen by the output stagger"):

======================  ===============  ==========================
Method                  Output stagger   Mask field
======================  ===============  ==========================
``Energetics2D.*``      T                ``mask.h``
``Strain2D.shear``      X                ``mask.xy_corner_strict``
``Strain2D.tensor``     T                ``mask.h``
``Strain2D.magnitude``  T                ``mask.h``
``Strain2D.okubo_weiss``T                ``mask.h``
======================  ===============  ==========================

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
    relative_vorticity_cgrid,
    shear_strain,
    strain_magnitude_squared,
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
