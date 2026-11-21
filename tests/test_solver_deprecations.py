"""The solver, stepper and constant names that moved to other repos are gone.

CG, its report and the multigrid and Nyström preconditioners are gaussx's
(#474); the forcing operators, the vertical modes, ``multilayer`` and the
barotropic filter are sesmax-core's (#422); the time steppers are integrix.march's; the constants, the Coriolis helpers and the linear equation of state are geophysix's (#446). The deprecation window is over:
``gridax.<name>`` and ``gridax.solvers.<name>`` raise ``AttributeError``.
"""

from __future__ import annotations

import importlib
import warnings

import gaussx
import gridax
import gridax.solvers as gs
import jax.numpy as jnp
import numpy as np
import pytest


def test_gridax_solvers_imports_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        importlib.reload(gs)


def _assert_gone(name: str) -> None:
    for module in (gridax, gs):
        assert name not in module.__all__
        with pytest.raises(AttributeError):
            getattr(module, name)


def test_as_preconditioner_is_a_gaussx_multigrid_preconditioner() -> None:
    n = 16
    mask = np.zeros((n + 2, n + 2))
    mask[1:-1, 1:-1] = 1.0
    solver = gs.build_multigrid_solver(mask, 1.0 / n, 1.0 / n, lambda_=4.0)
    pre = solver.as_preconditioner()
    assert isinstance(pre, gaussx.MultigridPreconditioner)
    rhs = jnp.asarray(np.random.default_rng(0).standard_normal(mask.shape) * mask)
    # one gaussx V-cycle from zero is the solver's own V-cycle
    np.testing.assert_array_equal(
        pre.v_cycle(rhs.ravel()).reshape(rhs.shape),
        solver.v_cycle(jnp.zeros_like(rhs), rhs),
    )


# --- the T-point pieces the pressure solve is built from (#475) --------------


@pytest.mark.parametrize(
    "name",
    [
        "masked_neumann_laplacian",
        "masked_face_dirichlet_laplacian",
        "wet_components",
        "project_zero_mean",
    ],
)
def test_t_point_pieces_are_public(name: str) -> None:
    assert name in gs.__all__
    assert callable(getattr(gs, name))


@pytest.mark.parametrize(
    "name",
    [
        "streamfunction_from_vorticity",
        "pressure_from_divergence",
        "pv_inversion",
        "solve_ocean_pde",
    ],
)
def test_moved_to_sesmax_core_and_gone(name: str) -> None:
    # sesmax_core.gfd and sesmax_core.solve_pde own them; gridax keeps no alias
    _assert_gone(name)


REMOVED_NAMES = [
    # gaussx (#474)
    "solve_cg",
    "CGInfo",
    "make_preconditioner",
    "make_nystrom_preconditioner",
    "make_multigrid_preconditioner",
    "make_gaussx_multigrid_preconditioner",
    # sesmax_core.forcing (#422)
    "AbstractForcing",
    "ForcingSum",
    "ForcingProduct",
    "TimeVaryingForcing",
    "LinearDrag2D",
    "LinearDrag3D",
    "QuadraticDrag2D",
    "QuadraticDrag3D",
    "RayleighDamping2D",
    "RayleighDamping3D",
    "WindStress2D",
    "WindStress3D",
    "WindCurl2D",
    "linear_drag_tendency",
    "quadratic_drag_tendency",
    "rayleigh_tendency",
    "wind_stress_tendency",
    "linear_drag",
    "rayleigh_relaxation",
    # sesmax_core.vertical (#422)
    "build_coupling_matrix",
    "decompose_vertical_modes",
    "layer_to_mode",
    "mode_to_layer",
    "multilayer",
    "barotropic_filter",
    # integrix.march (#476)
    "semi_lagrangian_step",
    "SemiLagrangianSolver",
    "SemiLagrangianTerm",
    # integrix.march (the time steppers)
    "ForwardEulerDfx",
    "RK2Heun",
    "SSP_RK2",
    "RK3SSP",
    "RK4Classic",
    "SSP_RK104",
    "IMEX_SSP2",
    "AB2Solver",
    "AB3Solver",
    "LeapfrogRAFSolver",
    "SplitExplicitRKSolver",
    "euler_step",
    "heun_step",
    "rk3_ssp_step",
    "rk4_step",
    "imex_ssp2_step",
    "ab2_step",
    "ab3_step",
    "leapfrog_raf_step",
    "split_explicit_step",
    # geophysix (#446): the linear equation of state and the Coriolis helpers
    "linear_density",
    "linear_density_anomaly",
    "linear_drho_dT",
    "linear_drho_dS",
    "buoyancy",
    "reduced_gravity",
    "coriolis_param",
    "beta_param",
    "coriolis_fn",
]


@pytest.mark.parametrize(
    "name",
    [
        "R_EARTH",
        "GRAVITY",
        "OMEGA",
        "RHO",
        "DEG2M",
        "coriolis_param",
        "beta_param",
        "coriolis_fn",
    ],
)
def test_names_moved_to_geophysix_are_gone_from_gridax(name: str) -> None:
    assert name not in gridax.__all__
    with pytest.raises(AttributeError):
        getattr(gridax, name)


@pytest.mark.parametrize("name", REMOVED_NAMES)
def test_removed_names_raise_attribute_error(name: str) -> None:
    _assert_gone(name)
