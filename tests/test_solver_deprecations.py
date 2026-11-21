"""The generic CG solve and preconditioner factories are deprecated (#474).

CG, its report and the multigrid and Nyström preconditioners are gaussx's
(jejjohnson/geostack#291, rule 1). For one release ``gridax_jax.<name>``
still works, warns, and names the gaussx call; the object is unchanged.
"""

from __future__ import annotations

import importlib
import warnings

import gaussx
import gridax_jax
from gridax_jax import _GAUSSX_SOLVES
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("name", sorted(_GAUSSX_SOLVES))
def test_name_warns_and_names_gaussx(name: str) -> None:
    module, _ = _GAUSSX_SOLVES[name]
    with pytest.warns(DeprecationWarning, match=r"gaussx\.") as record:
        obj = getattr(gridax_jax, name)
    assert f"gridax_jax.{name} is deprecated" in str(record[0].message)
    assert obj is getattr(importlib.import_module(module), name)


def test_gridax_jax_imports_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        importlib.reload(gridax_jax)


def test_as_preconditioner_is_a_gaussx_multigrid_preconditioner() -> None:
    n = 16
    mask = np.zeros((n + 2, n + 2))
    mask[1:-1, 1:-1] = 1.0
    solver = gridax_jax.build_multigrid_solver(mask, 1.0 / n, 1.0 / n, lambda_=4.0)
    pre = solver.as_preconditioner()
    assert isinstance(pre, gaussx.MultigridPreconditioner)
    rhs = jnp.asarray(np.random.default_rng(0).standard_normal(mask.shape) * mask)
    # one gaussx V-cycle from zero is the solver's own V-cycle
    np.testing.assert_array_equal(
        pre.v_cycle(rhs.ravel()).reshape(rhs.shape),
        solver.v_cycle(jnp.zeros_like(rhs), rhs),
    )


# --- the inversions and the PDE driver moved to sesmax-core (#475) ----------

_INVERSIONS = (
    "pressure_from_divergence",
    "pv_inversion",
    "streamfunction_from_vorticity",
)


@pytest.mark.parametrize("name", _INVERSIONS)
def test_inversion_warns_and_names_sesmax_core_gfd(name: str) -> None:
    from gridax_jax._src.solvers import elliptic

    with pytest.warns(DeprecationWarning, match=rf"sesmax_core\.gfd\.{name}"):
        obj = getattr(gridax_jax, name)
    assert obj is getattr(elliptic, name)
    assert name not in gridax_jax.__all__


def test_solve_ocean_pde_warns_and_names_solve_pde() -> None:
    from gridax_jax._src.timestepping import _solve

    with pytest.warns(DeprecationWarning, match=r"sesmax_core\.solve_pde") as record:
        obj = gridax_jax.solve_ocean_pde
    assert "tendency_bc=" in str(record[0].message)
    assert obj is _solve.solve_ocean_pde
    assert "solve_ocean_pde" not in gridax_jax.__all__


@pytest.mark.parametrize(
    "name",
    [
        "masked_neumann_laplacian",
        "masked_face_dirichlet_laplacian",
        "wet_components",
        "project_zero_mean",
    ],
)
def test_t_point_pieces_are_public_and_keep_their_private_names(name: str) -> None:
    from gridax_jax._src.solvers import iterative

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obj = getattr(gridax_jax, name)
    assert name in gridax_jax.__all__
    # sesmax-core 0.1 imports the private names
    assert obj is getattr(iterative, f"_{name}")
