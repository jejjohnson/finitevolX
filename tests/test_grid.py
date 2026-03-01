"""Tests for ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D."""

import pytest

from finitevolx._src.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D


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
