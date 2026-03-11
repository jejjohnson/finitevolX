"""
Harmonic and biharmonic diffusion operators on Arakawa C-grids.

Both operators follow the interior-point idiom used throughout finitevolX:
  * The output array has the same shape as the input.
  * Only interior cells [1:-1, 1:-1] (2-D) or [1:-1, 1:-1, 1:-1] (3-D) are
    written; the ghost ring is left as zero.
  * The caller is responsible for boundary conditions.

Conventions
-----------
Harmonic diffusion tendency:

    ∂h/∂t|_diff = κ · ∇²h

Biharmonic diffusion tendency (note the negative sign, standard in ocean/
atmosphere models — see MITgcm documentation and Leith 1968):

    ∂h/∂t|_diff = −κ · ∇⁴h

With a positive κ both operators provide dissipation.

The biharmonic operator is implemented as two successive Laplacians in
flux form.  The ghost ring of the intermediate Laplacian is zero (Dirichlet-0
at the halo), which is equivalent to a zero-normal-gradient BC on the
intermediate field.  Use ``enforce_periodic`` or set ghost cells explicitly
before calling if a different intermediate BC is required.

References
----------
.. [1] MITgcm Biharmonic Mixing:
       https://mitgcm.readthedocs.io/en/latest/optionals/packages/mixing.html#biharmonic-mixing
.. [2] Leith, C. E. (1968). Diffusion approximation for two-dimensional
       turbulence. *Physics of Fluids*, 11(3), 671–673.
.. [3] Veros biharmonic diffusion:
       https://github.com/team-ocean/veros/blob/main/veros/core/diffusion.py
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from finitevolx._src.difference import Difference2D, Difference3D
from finitevolx._src.grid import ArakawaCGrid2D, ArakawaCGrid3D


class HarmonicDiffusion2D(eqx.Module):
    """Harmonic (∇²) diffusion operator on a 2-D Arakawa C-grid.

    Computes the local diffusion tendency:

        ∂h/∂t|_diff = κ · ∇²h

    Only interior cells ``[1:-1, 1:-1]`` are written; the ghost ring is
    zero.  The caller is responsible for boundary conditions.

    Parameters
    ----------
    grid : ArakawaCGrid2D
        The underlying 2-D grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import ArakawaCGrid2D, HarmonicDiffusion2D
    >>> grid = ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> op = HarmonicDiffusion2D(grid=grid)
    >>> h = jnp.ones((grid.Ny, grid.Nx))
    >>> tend = op(h, kappa=1e-3)  # zero for constant field
    >>> tend.shape
    (10, 10)
    """

    grid: ArakawaCGrid2D
    _diff: Difference2D

    def __init__(self, grid: ArakawaCGrid2D) -> None:
        self.grid = grid
        self._diff = Difference2D(grid=grid)

    def __call__(
        self,
        h: Float[Array, "Ny Nx"],
        kappa: float,
    ) -> Float[Array, "Ny Nx"]:
        """Apply harmonic diffusion and return the tendency κ · ∇²h.

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar tracer field at T-points, shape ``[Ny, Nx]``.
        kappa : float
            Harmonic diffusion coefficient (κ ≥ 0 gives dissipation).

        Returns
        -------
        Float[Array, "Ny Nx"]
            Tendency κ · ∇²h at T-points, same shape as ``h``.
            Ghost cells are zero.
        """
        return kappa * self._diff.laplacian(h)


class BiharmonicDiffusion2D(eqx.Module):
    """Biharmonic (∇⁴) diffusion operator on a 2-D Arakawa C-grid.

    Computes the local biharmonic diffusion tendency:

        ∂h/∂t|_diff = −κ · ∇⁴h

    where ∇⁴h = ∇²(∇²h) is implemented as two successive flux-form
    Laplacians.  The negative sign ensures that a positive κ provides
    dissipation (the operator damps high-wavenumber modes).

    Scale-selective property: for a Fourier mode with wavenumber **k**, the
    harmonic tendency scales as ``−κ_h · k²`` while the biharmonic tendency
    scales as ``−κ_bi · k⁴``.  Biharmonic diffusion therefore damps small
    scales much more strongly than large scales.

    Only interior cells ``[1:-1, 1:-1]`` are written; the ghost ring is
    zero.  The caller is responsible for boundary conditions.

    Notes
    -----
    The ghost ring of the intermediate Laplacian ∇²h is zero (Dirichlet-0).
    For problems with periodic boundaries, call ``enforce_periodic`` on ``h``
    before invoking this operator so that the ghost cells of the input are
    correctly set; the intermediate Laplacian then also inherits sensible
    ghost values from the periodic input.

    Parameters
    ----------
    grid : ArakawaCGrid2D
        The underlying 2-D grid.

    References
    ----------
    .. [1] MITgcm Biharmonic Mixing:
           https://mitgcm.readthedocs.io/en/latest/optionals/packages/mixing.html#biharmonic-mixing
    .. [2] Leith, C. E. (1968). Diffusion approximation for two-dimensional
           turbulence. *Physics of Fluids*, 11(3), 671–673.

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
    _diff: Difference2D

    def __init__(self, grid: ArakawaCGrid2D) -> None:
        self.grid = grid
        self._diff = Difference2D(grid=grid)

    def __call__(
        self,
        h: Float[Array, "Ny Nx"],
        kappa: float,
    ) -> Float[Array, "Ny Nx"]:
        """Apply biharmonic diffusion and return the tendency −κ · ∇⁴h.

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar tracer field at T-points, shape ``[Ny, Nx]``.
        kappa : float
            Biharmonic diffusion coefficient (κ ≥ 0 gives dissipation).

        Returns
        -------
        Float[Array, "Ny Nx"]
            Tendency −κ · ∇⁴h at T-points, same shape as ``h``.
            Ghost cells are zero.
        """
        # ∇²h at T-points: writes to [1:-1, 1:-1], ghost ring = 0
        lap1 = self._diff.laplacian(h)
        # ∇²(∇²h) at T-points: reads zero ghost ring of lap1 (Dirichlet-0 BC)
        lap2 = self._diff.laplacian(lap1)
        return -kappa * lap2


class HarmonicDiffusion3D(eqx.Module):
    """Harmonic (∇²) diffusion operator on a 3-D Arakawa C-grid.

    Applies the horizontal Laplacian independently at each z-level:

        ∂h/∂t|_diff = κ · ∇²_h h

    where ∇²_h denotes the horizontal (x, y) Laplacian.

    Only interior cells ``[1:-1, 1:-1, 1:-1]`` are written; the ghost ring
    is zero.  The caller is responsible for boundary conditions.

    Parameters
    ----------
    grid : ArakawaCGrid3D
        The underlying 3-D grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import ArakawaCGrid3D, HarmonicDiffusion3D
    >>> grid = ArakawaCGrid3D.from_interior(4, 8, 8, 1.0, 1.0, 1.0)
    >>> op = HarmonicDiffusion3D(grid=grid)
    >>> h = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
    >>> tend = op(h, kappa=1e-3)
    >>> tend.shape
    (6, 10, 10)
    """

    grid: ArakawaCGrid3D
    _diff: Difference3D

    def __init__(self, grid: ArakawaCGrid3D) -> None:
        self.grid = grid
        self._diff = Difference3D(grid=grid)

    def __call__(
        self,
        h: Float[Array, "Nz Ny Nx"],
        kappa: float,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Apply horizontal harmonic diffusion and return the tendency κ · ∇²_h h.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar tracer field at T-points, shape ``[Nz, Ny, Nx]``.
        kappa : float
            Harmonic diffusion coefficient (κ ≥ 0 gives dissipation).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Tendency κ · ∇²_h h at T-points, same shape as ``h``.
            Ghost cells are zero.
        """
        return kappa * self._diff.laplacian(h)


class BiharmonicDiffusion3D(eqx.Module):
    """Biharmonic (∇⁴) diffusion operator on a 3-D Arakawa C-grid.

    Applies the horizontal biharmonic Laplacian independently at each
    z-level:

        ∂h/∂t|_diff = −κ · ∇⁴_h h

    where ∇⁴_h = ∇²_h(∇²_h) denotes the horizontal biharmonic operator.

    Only interior cells ``[1:-1, 1:-1, 1:-1]`` are written; the ghost ring
    is zero.  The caller is responsible for boundary conditions.

    Notes
    -----
    The ghost ring of the intermediate Laplacian ∇²_h h is zero
    (Dirichlet-0).  See :class:`BiharmonicDiffusion2D` notes for details.

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
    _diff: Difference3D

    def __init__(self, grid: ArakawaCGrid3D) -> None:
        self.grid = grid
        self._diff = Difference3D(grid=grid)

    def __call__(
        self,
        h: Float[Array, "Nz Ny Nx"],
        kappa: float,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Apply horizontal biharmonic diffusion and return the tendency −κ · ∇⁴_h h.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar tracer field at T-points, shape ``[Nz, Ny, Nx]``.
        kappa : float
            Biharmonic diffusion coefficient (κ ≥ 0 gives dissipation).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Tendency −κ · ∇⁴_h h at T-points, same shape as ``h``.
            Ghost cells are zero.
        """
        # ∇²_h h at T-points: writes to [1:-1, 1:-1, 1:-1], ghost ring = 0
        lap1 = self._diff.laplacian(h)
        # ∇²_h(∇²_h h): reads zero ghost ring of lap1 (Dirichlet-0 BC)
        lap2 = self._diff.laplacian(lap1)
        return -kappa * lap2
