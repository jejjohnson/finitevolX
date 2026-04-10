"""Spherical Arakawa C-grid containers (2-D, 3-D).

These grids are concrete subclasses of :class:`CurvilinearGrid2D` /
:class:`CurvilinearGrid3D` that add spherical metric information:
angular spacings (``dlon``, ``dlat`` in radians), the planet radius
``R``, precomputed ``cos(lat)`` at all four staggering locations, and
coordinate arrays for diagnostics.

The Cartesian-equivalent metric lengths inherited from
``CurvilinearGrid*`` are populated as ``dx = R * dlon``,
``dy = R * dlat``, etc., so spherical grids are drop-in compatible
with operators that consume ``grid.dx`` / ``grid.dy``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid.base import CurvilinearGrid2D, CurvilinearGrid3D
from finitevolx._src.utils.constants import R_EARTH


class SphericalGrid2D(CurvilinearGrid2D):
    """2-D Arakawa C-grid on a sphere.

    Inherits ``Nx``, ``Ny``, ``Lx``, ``Ly``, ``dx``, ``dy`` from
    :class:`CurvilinearGrid2D` and adds spherical metric fields.  All
    field arrays have total shape ``[Ny, Nx]``; the physical interior
    is ``(Ny-2) x (Nx-2)`` (slice ``[1:-1, 1:-1]``); one ghost-cell
    ring on each side is reserved for boundary conditions.
    Cartesian-equivalent metric lengths are stored as
    ``dx = R * dlon``, ``dy = R * dlat``,
    ``Lx = R * (lon_max - lon_min)``, ``Ly = R * (lat_max - lat_min)``.

    Parameters
    ----------
    dlon : float
        Longitudinal grid spacing in radians.
    dlat : float
        Latitudinal grid spacing in radians.
    R : float
        Radius of the sphere (metres).
    cos_lat_T : Float[Array, "Ny Nx"]
        cos(latitude) at T-points (cell centres).
    cos_lat_U : Float[Array, "Ny Nx"]
        cos(latitude) at U-points (east faces).  Equal to ``cos_lat_T``
        because U-points sit at the same latitude as T-points.
    cos_lat_V : Float[Array, "Ny Nx"]
        cos(latitude) at V-points (north faces), half-cell north of T.
    cos_lat_X : Float[Array, "Ny Nx"]
        cos(latitude) at X-points (NE corners).  Equal to ``cos_lat_V``.
    lat_T : Float[Array, "Ny Nx"]
        Latitude (radians) at T-points.
    lon_T : Float[Array, "Ny Nx"]
        Longitude (radians) at T-points.

    Notes
    -----
    Colocation convention (same-index rule, in spherical coordinates)::

        T[j, i]  cell centre  at  ( lon_i        ,  lat_j         )
        U[j, i]  east face    at  ( lon_{i+1/2}  ,  lat_j         )
        V[j, i]  north face   at  ( lon_i        ,  lat_{j+1/2}   )
        X[j, i]  NE corner    at  ( lon_{i+1/2}  ,  lat_{j+1/2}   )

    where ``lon_i = lon_min + (i-1) * dlon`` and
    ``lat_j = lat_min + (j-1) * dlat`` (the ``-1`` shift accounts for
    the south/west ghost cells at ``j=0``/``i=0``).  Because U-points
    share the latitude of T-points, ``cos_lat_U == cos_lat_T``;
    similarly ``cos_lat_X == cos_lat_V``.

    As in the Cartesian case, array index ``[j, i]`` encodes the
    **south-west** corner of the stencil neighbourhood.
    """

    dlon: float
    dlat: float
    R: float
    cos_lat_T: Float[Array, "Ny Nx"]
    cos_lat_U: Float[Array, "Ny Nx"]
    cos_lat_V: Float[Array, "Ny Nx"]
    cos_lat_X: Float[Array, "Ny Nx"]
    lat_T: Float[Array, "Ny Nx"]
    lon_T: Float[Array, "Ny Nx"]

    @classmethod
    def from_interior(  # intentionally different signature from Curvilinear
        cls,
        nx_interior: int,
        ny_interior: int,
        lon_range: tuple[float, float],
        lat_range: tuple[float, float],
        R: float = R_EARTH,
    ) -> SphericalGrid2D:
        """Construct a spherical grid from interior cell counts and coordinate ranges.

        Parameters
        ----------
        nx_interior : int
            Number of interior (physical) cells in longitude.
        ny_interior : int
            Number of interior (physical) cells in latitude.
        lon_range : tuple[float, float]
            (lon_min, lon_max) in **degrees**.
        lat_range : tuple[float, float]
            (lat_min, lat_max) in **degrees**.
        R : float
            Planet radius in metres (default :data:`R_EARTH`).

        Returns
        -------
        SphericalGrid2D
        """
        lon_min_rad = jnp.deg2rad(lon_range[0])
        lon_max_rad = jnp.deg2rad(lon_range[1])
        lat_min_rad = jnp.deg2rad(lat_range[0])
        lat_max_rad = jnp.deg2rad(lat_range[1])

        dlon = float((lon_max_rad - lon_min_rad) / nx_interior)
        dlat = float((lat_max_rad - lat_min_rad) / ny_interior)

        Nx = nx_interior + 2
        Ny = ny_interior + 2

        # Cartesian-equivalent metric lengths
        dx = R * dlon
        dy = R * dlat
        Lx = R * float(lon_max_rad - lon_min_rad)
        Ly = R * float(lat_max_rad - lat_min_rad)

        # 1-D coordinate arrays with ghost cells
        # Ghost cell at j=0 is one cell south of the physical domain;
        # j=1 is the first interior cell at lat_min.
        lat_1d = lat_min_rad + (jnp.arange(Ny) - 1) * dlat
        lon_1d = lon_min_rad + (jnp.arange(Nx) - 1) * dlon

        # 2-D meshgrid (j, i) ordering
        lon_2d, lat_2d = jnp.meshgrid(lon_1d, lat_1d)

        cos_lat_T = jnp.cos(lat_2d)
        cos_lat_U = cos_lat_T  # U sits at the same latitude as T

        lat_V = lat_2d + 0.5 * dlat
        cos_lat_V = jnp.cos(lat_V)
        cos_lat_X = cos_lat_V  # X sits at the same latitude as V

        return cls(
            Nx=Nx,
            Ny=Ny,
            Lx=Lx,
            Ly=Ly,
            dx=dx,
            dy=dy,
            dlon=dlon,
            dlat=dlat,
            R=R,
            cos_lat_T=cos_lat_T,
            cos_lat_U=cos_lat_U,
            cos_lat_V=cos_lat_V,
            cos_lat_X=cos_lat_X,
            lat_T=lat_2d,
            lon_T=lon_2d,
        )


class SphericalGrid3D(CurvilinearGrid3D):
    """3-D Arakawa C-grid on a sphere.

    Inherits ``Nx``, ``Ny``, ``Nz``, ``Lx``, ``Ly``, ``Lz``, ``dx``,
    ``dy``, ``dz`` from :class:`CurvilinearGrid3D` and adds spherical
    metric fields.  Scalar field arrays have total shape
    ``[Nz, Ny, Nx]`` with one ghost-cell ring on every axis; the
    physical interior is ``(Nz-2) x (Ny-2) x (Nx-2)`` (slice
    ``[1:-1, 1:-1, 1:-1]``).  The ``cos_lat_*`` and coordinate arrays
    are 2-D ``[Ny, Nx]`` — they are shared across all z-levels because
    the horizontal staggering is independent of depth.

    As with the Cartesian :class:`CartesianGrid3D`, only the horizontal
    axes are staggered (T, U, V, X share the same ``k * dz``); W-point
    fields (vertical velocity at z-interfaces) live in a separate
    ``[Nz+1, Ny, Nx]`` array — see
    ``finitevolx._src.operators.diagnostics.vertical_velocity``.

    Parameters
    ----------
    dlon : float
        Longitudinal grid spacing in radians.
    dlat : float
        Latitudinal grid spacing in radians.
    R : float
        Radius of the sphere (metres).
    cos_lat_T : Float[Array, "Ny Nx"]
        cos(latitude) at T-points.
    cos_lat_U : Float[Array, "Ny Nx"]
        cos(latitude) at U-points.  Equal to ``cos_lat_T``.
    cos_lat_V : Float[Array, "Ny Nx"]
        cos(latitude) at V-points (half-cell north of T).
    cos_lat_X : Float[Array, "Ny Nx"]
        cos(latitude) at X-points.  Equal to ``cos_lat_V``.
    lat_T : Float[Array, "Ny Nx"]
        Latitude (radians) at T-points.
    lon_T : Float[Array, "Ny Nx"]
        Longitude (radians) at T-points.

    Notes
    -----
    Colocation convention (same-index rule, horizontal staggering only)::

        T[k, j, i]  cell centre  at  ( lon_i       ,  lat_j        ,  k * dz )
        U[k, j, i]  east face    at  ( lon_{i+1/2} ,  lat_j        ,  k * dz )
        V[k, j, i]  north face   at  ( lon_i       ,  lat_{j+1/2}  ,  k * dz )
        X[k, j, i]  NE corner    at  ( lon_{i+1/2} ,  lat_{j+1/2}  ,  k * dz )

    All four point types share the same z-coordinate ``k * dz``, so the
    ``cos_lat_*`` arrays are 2-D and broadcast over the leading ``Nz``
    axis when used in operators.
    """

    dlon: float
    dlat: float
    R: float
    cos_lat_T: Float[Array, "Ny Nx"]
    cos_lat_U: Float[Array, "Ny Nx"]
    cos_lat_V: Float[Array, "Ny Nx"]
    cos_lat_X: Float[Array, "Ny Nx"]
    lat_T: Float[Array, "Ny Nx"]
    lon_T: Float[Array, "Ny Nx"]

    @classmethod
    def from_interior(  # intentionally different signature from Curvilinear
        cls,
        nx_interior: int,
        ny_interior: int,
        nz_interior: int,
        lon_range: tuple[float, float],
        lat_range: tuple[float, float],
        Lz: float,
        R: float = R_EARTH,
    ) -> SphericalGrid3D:
        """Construct a 3-D spherical grid from interior cell counts.

        Parameters
        ----------
        nx_interior : int
            Number of interior cells in longitude.
        ny_interior : int
            Number of interior cells in latitude.
        nz_interior : int
            Number of interior cells in z.
        lon_range : tuple[float, float]
            (lon_min, lon_max) in **degrees**.
        lat_range : tuple[float, float]
            (lat_min, lat_max) in **degrees**.
        Lz : float
            Physical domain depth/height.
        R : float
            Planet radius in metres (default :data:`R_EARTH`).

        Returns
        -------
        SphericalGrid3D
        """
        # Build the horizontal grid first
        h_grid = SphericalGrid2D.from_interior(
            nx_interior, ny_interior, lon_range, lat_range, R
        )

        Nz = nz_interior + 2
        dz = Lz / nz_interior

        return cls(
            Nx=h_grid.Nx,
            Ny=h_grid.Ny,
            Nz=Nz,
            Lx=h_grid.Lx,
            Ly=h_grid.Ly,
            Lz=Lz,
            dx=h_grid.dx,
            dy=h_grid.dy,
            dz=dz,
            dlon=h_grid.dlon,
            dlat=h_grid.dlat,
            R=R,
            cos_lat_T=h_grid.cos_lat_T,
            cos_lat_U=h_grid.cos_lat_U,
            cos_lat_V=h_grid.cos_lat_V,
            cos_lat_X=h_grid.cos_lat_X,
            lat_T=h_grid.lat_T,
            lon_T=h_grid.lon_T,
        )

    def horizontal_grid(self) -> SphericalGrid2D:
        """Extract the horizontal 2-D spherical grid.

        Returns
        -------
        SphericalGrid2D
            A 2-D grid with the same horizontal fields.
        """
        return SphericalGrid2D(
            Nx=self.Nx,
            Ny=self.Ny,
            Lx=self.Lx,
            Ly=self.Ly,
            dx=self.dx,
            dy=self.dy,
            dlon=self.dlon,
            dlat=self.dlat,
            R=self.R,
            cos_lat_T=self.cos_lat_T,
            cos_lat_U=self.cos_lat_U,
            cos_lat_V=self.cos_lat_V,
            cos_lat_X=self.cos_lat_X,
            lat_T=self.lat_T,
            lon_T=self.lon_T,
        )
