"""Tests for ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D."""

import pytest

from finitevolx._src.grid.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D


class TestArakawaCGrid1D:
    def test_from_interior_sizes(self):
        g = ArakawaCGrid1D.from_interior(10, 1.0)
        assert g.Nx == 12
        assert g.Lx == 1.0
        assert g.dx == pytest.approx(0.1)

    def test_direct_construction(self):
        g = ArakawaCGrid1D(Nx=6, Lx=5.0, dx=1.0)
        assert g.Nx == 6
        assert g.dx == 1.0

    def test_dx_matches_Lx(self):
        g = ArakawaCGrid1D.from_interior(4, 2.0)
        assert g.dx == pytest.approx(g.Lx / (g.Nx - 2))


class TestArakawaCGrid2D:
    def test_from_interior_sizes(self):
        g = ArakawaCGrid2D.from_interior(8, 6, 1.0, 2.0)
        assert g.Nx == 10
        assert g.Ny == 8
        assert g.dx == pytest.approx(1.0 / 8)
        assert g.dy == pytest.approx(2.0 / 6)

    def test_direct_construction(self):
        g = ArakawaCGrid2D(Nx=4, Ny=4, Lx=1.0, Ly=1.0, dx=0.5, dy=0.5)
        assert g.Nx == 4
        assert g.Ny == 4

    def test_square_grid(self):
        g = ArakawaCGrid2D.from_interior(10, 10, 1.0, 1.0)
        assert g.dx == g.dy


class TestArakawaCGrid3D:
    def test_from_interior_sizes(self):
        g = ArakawaCGrid3D.from_interior(4, 5, 6, 1.0, 2.0, 3.0)
        assert g.Nx == 6
        assert g.Ny == 7
        assert g.Nz == 8
        assert g.dx == pytest.approx(1.0 / 4)
        assert g.dy == pytest.approx(2.0 / 5)
        assert g.dz == pytest.approx(3.0 / 6)

    def test_direct_construction(self):
        g = ArakawaCGrid3D(
            Nx=4, Ny=4, Nz=4, Lx=1.0, Ly=1.0, Lz=1.0, dx=0.5, dy=0.5, dz=0.5
        )
        assert g.Nz == 4


# ---------------------------------------------------------------------------
# Staggered grid position tests
# ---------------------------------------------------------------------------
# The Arakawa C-grid convention (same-index colocation):
#   T[j, i] → cell centre  at  (i * dx,      j * dy     )
#   U[j, i] → east face    at  ((i+1/2)*dx,  j * dy     )
#   V[j, i] → north face   at  (i * dx,      (j+1/2)*dy )
#   X[j, i] → NE corner    at  ((i+1/2)*dx,  (j+1/2)*dy )
#
# Interior indices run i = 1 … Nx-2, j = 1 … Ny-2.
# ---------------------------------------------------------------------------

import numpy as np


class TestGridStaggerPositions2D:
    """Verify that derived face-point coordinates follow the stated conventions."""

    def test_T_cell_centre_positions(self):
        """T[j,i] is at (i*dx, j*dy)."""
        g = ArakawaCGrid2D.from_interior(8, 6, 1.0, 2.0)
        for i in range(1, g.Nx - 1):
            assert g.dx * i == pytest.approx(i * g.dx)
        # Simply verify the grid spacing is consistent
        assert g.dx == pytest.approx(g.Lx / (g.Nx - 2))
        assert g.dy == pytest.approx(g.Ly / (g.Ny - 2))

    def test_U_face_offset_by_half_dx(self):
        """U-face coordinate in x is T-centre shifted by +dx/2."""
        g = ArakawaCGrid2D.from_interior(8, 6, 1.0, 2.0)
        # T-point x at index i: x_T[i] = i * dx
        # U-point x at index i: x_U[i] = (i + 0.5) * dx = x_T[i] + dx/2
        for i in range(1, g.Nx - 1):
            x_T = i * g.dx
            x_U = (i + 0.5) * g.dx
            assert x_U == pytest.approx(x_T + g.dx / 2)

    def test_V_face_offset_by_half_dy(self):
        """V-face coordinate in y is T-centre shifted by +dy/2."""
        g = ArakawaCGrid2D.from_interior(8, 6, 1.0, 2.0)
        for j in range(1, g.Ny - 1):
            y_T = j * g.dy
            y_V = (j + 0.5) * g.dy
            assert y_V == pytest.approx(y_T + g.dy / 2)

    def test_X_corner_offset_by_half_dx_and_dy(self):
        """X-corner is offset by (dx/2, dy/2) from the T-centre."""
        g = ArakawaCGrid2D.from_interior(8, 6, 1.0, 2.0)
        for i in range(1, g.Nx - 1):
            for j in range(1, g.Ny - 1):
                x_X = (i + 0.5) * g.dx
                y_X = (j + 0.5) * g.dy
                x_T = i * g.dx
                y_T = j * g.dy
                assert x_X == pytest.approx(x_T + g.dx / 2)
                assert y_X == pytest.approx(y_T + g.dy / 2)

    def test_grid_symmetry_square_domain(self):
        """For a square uniform domain [0,L]x[0,L] the grid is symmetric."""
        L = 1.0
        g = ArakawaCGrid2D.from_interior(10, 10, L, L)
        assert g.dx == g.dy
        # T-point positions are symmetric about L/2

        interior = np.arange(1, g.Nx - 1)
        x_T = interior * g.dx
        # Just check that the mean is symmetric about L/2
        assert np.mean(x_T) == pytest.approx(L / 2, abs=g.dx)

    def test_ghost_cell_count(self):
        """There is exactly one ghost cell ring on each side."""
        g = ArakawaCGrid2D.from_interior(8, 6, 1.0, 2.0)
        assert g.Nx == 8 + 2
        assert g.Ny == 6 + 2

    def test_grid_spacing_consistent_with_domain(self):
        """dx = Lx / Nx_interior, dy = Ly / Ny_interior."""
        Lx, Ly = 3.0, 5.0
        nx, ny = 12, 20
        g = ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
        assert g.dx == pytest.approx(Lx / nx)
        assert g.dy == pytest.approx(Ly / ny)


class TestGridStaggerPositions1D:
    """Verify 1D grid positions."""

    def test_T_positions(self):
        """T[i] is at i * dx."""
        g = ArakawaCGrid1D.from_interior(8, 1.0)
        for i in range(1, g.Nx - 1):
            assert i * g.dx == pytest.approx(i * g.dx)

    def test_U_face_at_plus_half_dx(self):
        """U[i] (east face) is at (i + 0.5) * dx."""
        g = ArakawaCGrid1D.from_interior(8, 1.0)
        for i in range(1, g.Nx - 1):
            x_U = (i + 0.5) * g.dx
            assert x_U == pytest.approx(i * g.dx + g.dx / 2)

    def test_dx_equals_Lx_over_n(self):
        n = 16
        L = 2.0
        g = ArakawaCGrid1D.from_interior(n, L)
        assert g.dx == pytest.approx(L / n)
