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

Under a mask, inputs are zeroed on dry cells of their stagger before any
stencil reads them (``u`` by ``mask.u``, ``v`` by ``mask.v`` -- the
no-normal-flow condition; the QG streamfunction by ``mask.h``), and
outputs are zeroed with ``jnp.where`` rather than a multiply, so land values stored as ``NaN`` (as :meth:`Mask2D.from_center`
supports) can neither leak into wet cells nor survive at dry ones.

With ``mask=None`` every method returns exactly what the functional form
returns, except where a method's docstring says otherwise.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
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
    shear_strain,
    strain_magnitude_squared,
    stretching_term,
    tensor_strain,
)
from finitevolx._src.operators.interpolation import Interpolation2D
from finitevolx._src.utils.constants import GRAVITY


def _where(wet: Array | None, field: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
    """Zero ``field`` where ``wet`` is False; identity when ``wet`` is None.

    field[j, i] = field[j, i] if wet[j, i] else 0
    """
    if wet is None:
        return field
    return jnp.where(wet, field, 0.0)


class Energetics2D(eqx.Module):
    """Energy diagnostics at T-points on a 2-D Arakawa C-grid.

    Class form of :func:`~finitevolx.kinetic_energy`,
    :func:`~finitevolx.bernoulli_potential` and
    :func:`~finitevolx.available_potential_energy`.  Every method returns a
    T-point field with a zero ghost ring; when ``mask`` is set, velocities
    are zeroed on dry faces first and dry T-cells are zeroed last (via
    ``jnp.where(mask.h, ...)``).

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  ``None`` (default) returns exactly the
        functional forms' output (for :meth:`available_potential_energy`,
        on the interior; see there).

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

    def _uv(
        self, u: Float[Array, "Ny Nx"], v: Float[Array, "Ny Nx"]
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Zero velocities on dry faces (no-normal-flow)."""
        if self.mask is None:
            return u, v
        return _where(self.mask.u, u), _where(self.mask.v, v)

    def _out(self, out: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Zero dry T-cells: out[j, i] = out[j, i] if mask.h[j, i] else 0."""
        return _where(None if self.mask is None else self.mask.h, out)

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
        u, v = self._uv(u, v)
        return self._out(kinetic_energy(u, v))

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
        u, v = self._uv(u, v)
        return self._out(bernoulli_potential(h, u, v, gravity))

    def available_potential_energy(
        self,
        h: Float[Array, "Ny Nx"],
        H: Float[Array, "Ny Nx"],
        g_prime: float,
    ) -> Float[Array, "Ny Nx"]:
        """Available potential energy at T-points.

        ape[j, i] = 1/2 * g_prime * (h[j, i] - H[j, i])^2

        The functional form is pointwise over the whole array; the class
        writes only the interior ``[1:-1, 1:-1]`` and leaves the ghost ring
        zero, like every other operator.

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
            Available potential energy at T-points, zero in the ghost ring
            and, when ``self.mask`` is set, at dry T-cells.
        """
        ape = available_potential_energy(h, H, g_prime)
        # out[j, i] = ape[j, i]  for 1 <= j <= Ny-2, 1 <= i <= Nx-2
        out = jnp.zeros_like(ape).at[1:-1, 1:-1].set(ape[1:-1, 1:-1])
        return self._out(out)


class Strain2D(eqx.Module):
    """Strain diagnostics on a 2-D Arakawa C-grid.

    Class form of :func:`~finitevolx.shear_strain`,
    :func:`~finitevolx.tensor_strain`,
    :func:`~finitevolx.strain_magnitude_squared` and
    :func:`~finitevolx.okubo_weiss`.

    The shear strain lives at X-points and the tensor strain at T-points.
    :meth:`magnitude_squared` and :meth:`okubo_weiss` combine them at
    **T-points**: the X-point shear (and vorticity) are averaged to T-points
    with :meth:`Interpolation2D.X_to_T`, which reads the south ghost X-row
    and west ghost X-column.  Those X ghosts are BC-owned but hidden inside
    the method, so they are computed here from the caller's ghost ``u`` and
    ``v`` (see :meth:`magnitude_squared`) -- apply boundary conditions to
    ``u`` and ``v`` before calling.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  Velocities are zeroed on dry faces
        first; each method then zeroes the dry cells of its output stagger
        (``mask.xy_corner_strict`` for :meth:`shear`, ``mask.h``
        otherwise).  For the T-point combinations the X-point intermediates
        are masked before averaging (pass-down, Pattern 2 in
        ``docs/masks.md``), so a coastal T-cell averages in zero from its
        dry corners.  ``None`` (default) returns exactly the functional
        forms' output for :meth:`shear` and :meth:`tensor`.

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

    def _uv(
        self, u: Float[Array, "Ny Nx"], v: Float[Array, "Ny Nx"]
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Zero velocities on dry faces (no-normal-flow)."""
        if self.mask is None:
            return u, v
        return _where(self.mask.u, u), _where(self.mask.v, v)

    def _x_with_ghosts(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        sign: float,
    ) -> Float[Array, "Ny Nx"]:
        """dv/dx + sign * du/dy at X-points, including the BC-owned ghosts.

        sign = +1 gives the shear strain, sign = -1 the relative vorticity:

        x[j+1/2, i+1/2] = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
                        + sign * (u[j+1, i+1/2] - u[j, i+1/2]) / dy

        written for 0 <= j <= Ny-2, 0 <= i <= Nx-2, i.e. the interior plus
        the south ghost X-row (j = 0) and west ghost X-column (i = 0), which
        read the caller's ghost u / v.  The north / east X-ghosts are
        outside the domain and stay zero.  At interior X-points this is the
        same arithmetic as :func:`~finitevolx.shear_strain` /
        :func:`~finitevolx.relative_vorticity_cgrid`.
        """
        # dv_dx[j+1/2, i+1/2] = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
        dv_dx = (v[:-1, 1:] - v[:-1, :-1]) / self.grid.dx
        # du_dy[j+1/2, i+1/2] = (u[j+1, i+1/2] - u[j, i+1/2]) / dy
        du_dy = (u[1:, :-1] - u[:-1, :-1]) / self.grid.dy
        # x[j, i] set for 0 <= j <= Ny-2, 0 <= i <= Nx-2
        out = jnp.zeros_like(u).at[:-1, :-1].set(dv_dx + sign * du_dy)
        return _where(None if self.mask is None else self.mask.xy_corner_strict, out)

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
        u, v = self._uv(u, v)
        out = shear_strain(u, v, self.grid.dx, self.grid.dy)
        return _where(None if self.mask is None else self.mask.xy_corner_strict, out)

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
        u, v = self._uv(u, v)
        out = tensor_strain(u, v, self.grid.dx, self.grid.dy)
        return _where(None if self.mask is None else self.mask.h, out)

    def magnitude_squared(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Squared strain magnitude at T-points.

        sigma2[j, i] = sn[j, i]^2 + ss_on_T[j, i]^2

        ss_on_T[j, i] = 1/4 * (ss[j+1/2, i+1/2] + ss[j-1/2, i+1/2]
                             + ss[j+1/2, i-1/2] + ss[j-1/2, i-1/2])

        The first interior row and column read the south / west ghost
        X-points ``ss[1/2, *]`` and ``ss[*, 1/2]``; those are computed from
        the caller's ghost ``u`` / ``v``, so they carry whatever boundary
        condition was applied to the velocities.

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
        u, v = self._uv(u, v)
        sn = tensor_strain(u, v, self.grid.dx, self.grid.dy)
        ss_on_T = self.interp.X_to_T(self._x_with_ghosts(u, v, 1.0))
        out = strain_magnitude_squared(sn, ss_on_T)
        return _where(None if self.mask is None else self.mask.h, out)

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

        are averaged to T-points over their four surrounding corners
        (including the south / west ghost corners), as in
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
        u, v = self._uv(u, v)
        sn = tensor_strain(u, v, self.grid.dx, self.grid.dy)
        ss_on_T = self.interp.X_to_T(self._x_with_ghosts(u, v, 1.0))
        omega_on_T = self.interp.X_to_T(self._x_with_ghosts(u, v, -1.0))
        out = okubo_weiss(sn, ss_on_T, omega_on_T)
        return _where(None if self.mask is None else self.mask.h, out)


class QGPotentialVorticity2D(eqx.Module):
    """Quasi-geostrophic potential vorticity at T-points.

    Class form of :func:`~finitevolx.qg_potential_vorticity`,
    :func:`~finitevolx.stretching_term` and
    :func:`~finitevolx.potential_vorticity_multilayer`.  Every method
    returns a T-point field (``[Ny, Nx]`` for one layer, ``[nl, Ny, Nx]``
    for the multilayer methods); when ``mask`` is set, dry T-cells are
    zeroed via ``jnp.where(mask.h, ...)`` (post-compute, Pattern 1 in
    ``docs/masks.md``), broadcast over the layer axis.

    The Laplacian stencil reads ``psi`` at the four neighbours of each
    T-cell.  Under a mask, ``psi`` is first set to zero on dry T-cells --
    the usual no-normal-flow condition, and what a masked elliptic solve
    returns -- so a coastal cell never reads land values (``NaN`` or
    otherwise).

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
        """Zero dry T-cells: out[..., j, i] = out[..., j, i] if mask.h[j, i] else 0."""
        return _where(None if self.mask is None else self.mask.h, out)

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
        psi = self._mask_h(psi)
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
        return self._mask_h(stretching_term(A, self._mask_h(psi)))

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
        psi = self._mask_h(psi)
        return self._mask_h(
            potential_vorticity_multilayer(
                psi, A, f0, beta, self.grid.dx, self.grid.dy, y, y0
            )
        )
