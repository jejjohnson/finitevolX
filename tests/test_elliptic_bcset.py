"""Tests for passing a BoundaryConditionSet as ``bc=`` to the elliptic wrappers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import finitevolx as fvx
from finitevolx import (
    BoundaryConditionSet,
    Dirichlet1D,
    Neumann1D,
    Periodic1D,
)

jax.config.update("jax_enable_x64", True)

NY, NX = 20, 24
DX, DY = 0.05, 0.04
CG_TOL = 1e-6


def _rhs() -> jnp.ndarray:
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.standard_normal((NY, NX)))


def _basin_mask() -> jnp.ndarray:
    return jnp.zeros((NY, NX)).at[1:-1, 1:-1].set(1.0)


def _dirichlet(south=0.0, north=0.0, west=0.0, east=0.0, mask=None):
    return BoundaryConditionSet(
        south=Dirichlet1D("south", value=south),
        north=Dirichlet1D("north", value=north),
        west=Dirichlet1D("west", value=west),
        east=Dirichlet1D("east", value=east),
        mask=mask,
    )


def _neumann(value=0.0):
    return BoundaryConditionSet(
        south=Neumann1D("south", value=value),
        north=Neumann1D("north", value=value),
        west=Neumann1D("west", value=value),
        east=Neumann1D("east", value=value),
    )


class TestTransformDispatch:
    def test_closed_is_dst(self):
        rhs = _rhs()
        psi = fvx.streamfunction_from_vorticity(
            rhs, DX, DY, bc=BoundaryConditionSet.closed(), lambda_=2.0
        )
        ref = fvx.streamfunction_from_vorticity(rhs, DX, DY, bc="dst", lambda_=2.0)
        np.testing.assert_array_equal(psi, ref)

    def test_zero_neumann_is_dct(self):
        rhs = _rhs() - jnp.mean(_rhs())
        p = fvx.pressure_from_divergence(rhs, DX, DY, bc=_neumann())
        ref = fvx.pressure_from_divergence(rhs, DX, DY, bc="dct")
        np.testing.assert_array_equal(p, ref)

    def test_periodic_is_fft(self):
        rhs = _rhs()
        psi = fvx.streamfunction_from_vorticity(
            rhs, DX, DY, bc=BoundaryConditionSet.periodic(), lambda_=3.0
        )
        ref = fvx.streamfunction_from_vorticity(rhs, DX, DY, bc="fft", lambda_=3.0)
        np.testing.assert_array_equal(psi, ref)

    def test_pv_inversion_array_lambda(self):
        pv = jnp.stack([_rhs(), 2.0 * _rhs()])
        lambdas = jnp.array([1.0, 4.0])
        psi = fvx.pv_inversion(pv, DX, DY, lambdas, bc=BoundaryConditionSet.closed())
        ref = fvx.pv_inversion(pv, DX, DY, lambdas, bc="dst")
        np.testing.assert_array_equal(psi, ref)


class TestMaskExtraction:
    def test_cg_uses_bc_mask(self):
        mask = _basin_mask()
        rhs = _rhs() * mask
        psi = fvx.streamfunction_from_vorticity(
            rhs, DX, DY, bc=BoundaryConditionSet.closed(mask=mask), method="cg"
        )
        ref = fvx.streamfunction_from_vorticity(rhs, DX, DY, method="cg", mask=mask)
        np.testing.assert_allclose(psi, ref, atol=1e-12)

    def test_mask_twice_raises(self):
        mask = _basin_mask()
        with pytest.raises(ValueError, match="not both"):
            fvx.streamfunction_from_vorticity(
                _rhs(),
                DX,
                DY,
                bc=BoundaryConditionSet.closed(mask=mask),
                method="cg",
                mask=mask,
            )

    def test_spectral_ignores_bc_mask(self):
        """The spectral path is rectangular; bc.mask only feeds mask methods."""
        rhs = _rhs()
        psi = fvx.streamfunction_from_vorticity(
            rhs, DX, DY, bc=BoundaryConditionSet.closed(mask=_basin_mask())
        )
        ref = fvx.streamfunction_from_vorticity(rhs, DX, DY, bc="dst")
        np.testing.assert_array_equal(psi, ref)


class TestInhomogeneousDirichletFaces:
    def test_face_values_pin_wall_rows(self):
        bc = _dirichlet(south=-0.2, north=0.1, west=0.05, east=0.3)
        psi = fvx.streamfunction_from_vorticity(_rhs(), DX, DY, bc=bc, lambda_=4.0)
        # Wall-adjacent wet cells (inner ring), corners owned by west/east.
        np.testing.assert_allclose(psi[1, 2:-2], -0.2, atol=1e-12)
        np.testing.assert_allclose(psi[-2, 2:-2], 0.1, atol=1e-12)
        np.testing.assert_allclose(psi[1:-1, 1], 0.05, atol=1e-12)
        np.testing.assert_allclose(psi[1:-1, -2], 0.3, atol=1e-12)

    def test_matches_explicit_known_values(self):
        bc = _dirichlet(north=0.1)
        # West/east (value 0) own the corner cells of the north wall row.
        g = jnp.zeros((NY, NX)).at[-2, 2:-2].set(0.1)
        rhs = _rhs()
        psi = fvx.streamfunction_from_vorticity(rhs, DX, DY, bc=bc)
        ref = fvx.streamfunction_from_vorticity(rhs, DX, DY, known_values=g)
        np.testing.assert_array_equal(psi, ref)

    def test_cg_with_bc_mask_matches_spectral(self):
        mask = _basin_mask()
        rhs = _rhs() * mask
        psi_cg = fvx.streamfunction_from_vorticity(
            rhs, DX, DY, bc=_dirichlet(north=0.1, mask=mask), method="cg"
        )
        psi_sp = fvx.streamfunction_from_vorticity(
            rhs, DX, DY, bc=_dirichlet(north=0.1)
        )
        np.testing.assert_allclose(psi_cg, psi_sp, atol=CG_TOL)

    def test_known_values_take_precedence(self):
        rhs = _rhs()
        g = jnp.full((NY, NX), 0.7)
        psi = fvx.streamfunction_from_vorticity(
            rhs, DX, DY, bc=_dirichlet(north=0.1), known_values=g
        )
        ref = fvx.streamfunction_from_vorticity(rhs, DX, DY, known_values=g)
        np.testing.assert_array_equal(psi, ref)

    def test_pv_inversion_face_values(self):
        pv = jnp.stack([_rhs(), _rhs()])
        psi = fvx.pv_inversion(
            pv, DX, DY, jnp.array([1.0, 9.0]), bc=_dirichlet(east=0.25)
        )
        np.testing.assert_allclose(psi[:, 1:-1, -2], 0.25, atol=1e-12)


class TestErrors:
    def test_mixed_face_types(self):
        bc = BoundaryConditionSet(
            south=Dirichlet1D("south", value=0.0),
            north=Dirichlet1D("north", value=0.0),
            west=Periodic1D("west"),
            east=Periodic1D("east"),
        )
        with pytest.raises(ValueError, match="no mixing"):
            fvx.streamfunction_from_vorticity(_rhs(), DX, DY, bc=bc)

    def test_missing_face(self):
        bc = BoundaryConditionSet(south=Dirichlet1D("south", value=0.0))
        with pytest.raises(ValueError, match="all four faces"):
            fvx.streamfunction_from_vorticity(_rhs(), DX, DY, bc=bc)

    def test_unsupported_face_type(self):
        with pytest.raises(ValueError, match="Outflow1D"):
            fvx.streamfunction_from_vorticity(
                _rhs(), DX, DY, bc=BoundaryConditionSet.open()
            )

    def test_inhomogeneous_neumann(self):
        with pytest.raises(ValueError, match="Inhomogeneous Neumann"):
            fvx.pressure_from_divergence(_rhs(), DX, DY, bc=_neumann(0.5))
