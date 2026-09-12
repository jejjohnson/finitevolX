"""Contract tests: every spatial operator must be unit-agnostic.

Rescaling the grid by a factor ``L`` and the input fields by their own
amplitudes must rescale an operator's output by the dimensionally
correct power of those factors, and by nothing else. An operator that
bakes in an SI constant, or that guards a ratio with an absolute
epsilon, breaks this and is no longer safe to run in nondimensional
units.

Downstream non-dimensionalisation (jejjohnson/somax#170) is a
factory-only change — models are rebuilt with ``Lx=1``, ``f0=1`` and the
same operators — so it rests entirely on this property. Pinning it here
means a regression fails in finitevolx rather than silently producing a
wrong nondimensional run downstream.

For an operator ``F`` of differential order ``p`` acting on fields with
amplitudes ``a_1 ... a_k``::

    F(x_1/a_1, ..., x_k/a_k;  grid/L)  ==  L**p / (a_1 * ... * a_k) * F(x_1, ..., x_k;  grid)

where ``p`` counts inverse-length factors: 0 for interpolation, 1 for a
gradient or flux divergence, 2 for a Laplacian, 4 for a biharmonic, and
-2 for a Poisson solve (which multiplies by length squared).
"""

import ast
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from finitevolx._src.advection.advection import (
    Advection1D,
    Advection2D,
    Advection3D,
)
from finitevolx._src.advection.spherical_advection import (
    SphericalAdvection2D,
    SphericalAdvection3D,
)
from finitevolx._src.diffusion.diffusion import (
    BiharmonicDiffusion2D,
    BiharmonicDiffusion3D,
    Diffusion2D,
    Diffusion3D,
)
from finitevolx._src.diffusion.spherical_diffusion import (
    SphericalBiharmonicDiffusion2D,
    SphericalBiharmonicDiffusion3D,
    SphericalDiffusion2D,
    SphericalDiffusion3D,
)
from finitevolx._src.grid.cartesian import (
    CartesianGrid1D,
    CartesianGrid2D,
    CartesianGrid3D,
)
from finitevolx._src.grid.spherical import SphericalGrid2D, SphericalGrid3D
from finitevolx._src.mask.cartesian import Mask1D, Mask2D, Mask3D
from finitevolx._src.operators.difference import (
    Difference1D,
    Difference2D,
    Difference3D,
)
from finitevolx._src.operators.divergence import Divergence2D
from finitevolx._src.operators.interpolation import (
    Interpolation1D,
    Interpolation2D,
    Interpolation3D,
)
from finitevolx._src.operators.jacobian import arakawa_jacobian
from finitevolx._src.operators.spherical_compound import (
    SphericalDivergence2D,
    SphericalDivergence3D,
    SphericalLaplacian2D,
    SphericalLaplacian3D,
    SphericalVorticity2D,
    SphericalVorticity3D,
)
from finitevolx._src.operators.spherical_difference import (
    SphericalDifference2D,
    SphericalDifference3D,
)
from finitevolx._src.operators.vorticity import Vorticity2D, Vorticity3D
from finitevolx._src.solvers.spectral import (
    solve_poisson_dst,
    solve_poisson_fft,
)

NX = NY = 16
NZ = 4
LX = LY = 2.0
LZ = 0.5

#: Array shapes the whole matrix is run at, as ``(nz, ny, nx)``.
#: The parameterizations vary physical *lengths*, not array sizes, so
#: an indexing or spectral-transform bug confined to odd extents, to a
#: non-square grid, or to a different number of vertical levels would
#: be invisible at a single shape. The second entry is odd in both
#: horizontal directions, non-square, and one level deeper.
DEFAULT_SHAPE = (NZ, NY, NX)
ALTERNATE_SHAPE = (5, 13, 17)
SHAPES = (DEFAULT_SHAPE, ALTERNATE_SHAPE)

# Grid rescalings and field amplitudes spanning the physical -> nondimensional
# range. A purely relative bug survives L = U = 1; an absolute-tolerance bug
# only shows up at the extremes.
LENGTH_FACTORS = [1e-3, 1.0, 1e6]
AMPLITUDE_FACTORS = [1e-2, 1.0, 1e3]

#: Per-label multipliers applied on top of the amplitude factor, in
#: order of first appearance. Unequal and not powers of one another, so
#: a mix-up between two of a bilinear operator's inputs cannot cancel.
LABEL_SPREAD = (1.0, 4.0, 0.125, 32.0, 0.0625)


def smooth_field(seed, shape=DEFAULT_SHAPE):
    """A smooth O(1) field — smooth so that limiters behave consistently."""
    ny, nx = shape[-2:]
    rng = np.random.RandomState(seed)
    y = np.linspace(0.0, 2.0 * np.pi, ny)[:, None]
    x = np.linspace(0.0, 2.0 * np.pi, nx)[None, :]
    out = np.zeros((ny, nx))
    for k in (1, 2, 3):
        out += rng.randn() * np.sin(k * x + rng.randn()) * np.cos(k * y + rng.randn())
    return jnp.asarray(out / np.abs(out).max())


def smooth_field_1d(seed, shape=DEFAULT_SHAPE):
    """The 1-D counterpart of :func:`smooth_field`."""
    nx = shape[-1]
    rng = np.random.RandomState(seed)
    x = np.linspace(0.0, 2.0 * np.pi, nx)
    out = np.zeros(nx)
    for k in (1, 2, 3):
        out += rng.randn() * np.sin(k * x + rng.randn())
    return jnp.asarray(out / np.abs(out).max())


def smooth_field_3d(seed, shape=DEFAULT_SHAPE):
    """A stack of decorrelated 2-D fields, one per z-level."""
    nz, ny, nx = shape
    return jnp.stack([smooth_field(100 * seed + k, (ny, nx)) for k in range(nz)])


def cartesian_grid_1d(length_factor, shape=DEFAULT_SHAPE):
    return CartesianGrid1D.from_interior(shape[-1] - 2, LX / length_factor)


def cartesian_grid(length_factor, shape=DEFAULT_SHAPE):
    return CartesianGrid2D.from_interior(
        shape[-1] - 2, shape[-2] - 2, LX / length_factor, LY / length_factor
    )


def cartesian_grid_3d(length_factor, shape=DEFAULT_SHAPE):
    return CartesianGrid3D.from_interior(
        shape[-1] - 2,
        shape[-2] - 2,
        shape[-3] - 2,
        LX / length_factor,
        LY / length_factor,
        LZ / length_factor,
    )


def spherical_grid(length_factor, shape=DEFAULT_SHAPE):
    """Angles are invariant under rescaling; only the radius carries length."""
    return SphericalGrid2D.from_interior(
        shape[-1] - 2,
        shape[-2] - 2,
        lon_range=(0.0, 40.0),
        lat_range=(10.0, 50.0),
        R=6.0e6 / length_factor,
    )


def spherical_grid_3d(length_factor, shape=DEFAULT_SHAPE):
    """As :func:`spherical_grid`; the vertical extent is a length too."""
    return SphericalGrid3D.from_interior(
        shape[-1] - 2,
        shape[-2] - 2,
        shape[-3] - 2,
        lon_range=(0.0, 40.0),
        lat_range=(10.0, 50.0),
        Lz=LZ / length_factor,
        R=6.0e6 / length_factor,
    )


def _dry_block(extent):
    """An interior slice to mark as land, scaled to the extent."""
    start = extent // 3
    return slice(start, min(start + 3, extent - 1))


def land_mask_1d(shape=DEFAULT_SHAPE):
    wet = np.ones(shape[-1], dtype=bool)
    wet[_dry_block(shape[-1])] = False
    return Mask1D.from_mask(jnp.asarray(wet))


def land_mask(shape=DEFAULT_SHAPE):
    wet = np.ones(shape[-2:], dtype=bool)
    wet[_dry_block(shape[-2]), _dry_block(shape[-1])] = False
    return Mask2D.from_mask(jnp.asarray(wet))


def land_mask_3d(shape=DEFAULT_SHAPE):
    wet = np.ones(shape, dtype=bool)
    wet[:, _dry_block(shape[-2]), _dry_block(shape[-1])] = False
    return Mask3D.from_mask(jnp.asarray(wet))


#: Grid factory and field/mask makers, keyed by ``(dim, geometry)``.
GRID_FACTORIES = {
    (1, "cartesian"): cartesian_grid_1d,
    (2, "cartesian"): cartesian_grid,
    (2, "spherical"): spherical_grid,
    (3, "cartesian"): cartesian_grid_3d,
    (3, "spherical"): spherical_grid_3d,
}
FIELD_MAKERS = {1: smooth_field_1d, 2: smooth_field, 3: smooth_field_3d}
MASK_MAKERS = {1: land_mask_1d, 2: land_mask, 3: land_mask_3d}


# ----------------------------------------------------------------------
# Operator cases
# ----------------------------------------------------------------------
#
# Each case declares:
#   run      (grid, mask, fields) -> output array
#   labels   one amplitude label per field; the expected factor divides
#            by the product of the *distinct* labels, so a bilinear
#            operator such as advection (h * u, h * v) divides by
#            A * B once, not A * B * B
#   order    power of the length factor L in the expected output scaling
#   grid     "cartesian" or "spherical"
#   dim      1, 2 or 3 — picks the grid, field and mask factories
#
# Distinct labels get *distinct* amplitudes (see LABEL_SPREAD): scaling
# every input by the same number would only test total homogeneity, so
# an operator that accidentally squared one argument and ignored
# another would still come out covariant.


class Case:
    def __init__(self, name, run, labels, order, grid="cartesian", masked=True, dim=2):
        self.name = name
        self.run = run
        self.labels = labels
        self.order = order
        self.grid = grid
        self.masked = masked
        self.dim = dim

    def __repr__(self):
        return self.name


def _cases():
    return [
        # --- order 0: interpolation carries no length at all ---------
        Case(
            "interp_T_to_U",
            lambda g, m, f: Interpolation2D(grid=g).T_to_U(f[0]),
            ("A",),
            0,
            masked=False,
        ),
        Case(
            "interp_T_to_X",
            lambda g, m, f: Interpolation2D(grid=g).T_to_X(f[0]),
            ("A",),
            0,
            masked=False,
        ),
        # --- order 1: gradients, divergence, curl --------------------
        Case(
            "difference_x_T_to_U",
            lambda g, m, f: Difference2D(grid=g, mask=m).diff_x_T_to_U(f[0]),
            ("A",),
            1,
        ),
        Case(
            "difference_y_T_to_V",
            lambda g, m, f: Difference2D(grid=g, mask=m).diff_y_T_to_V(f[0]),
            ("A",),
            1,
        ),
        Case(
            "difference_divergence",
            lambda g, m, f: Difference2D(grid=g, mask=m).divergence(f[0], f[1]),
            ("B", "B"),
            1,
        ),
        Case(
            "difference_curl",
            lambda g, m, f: Difference2D(grid=g, mask=m).curl(f[0], f[1]),
            ("B", "B"),
            1,
        ),
        Case(
            "divergence_operator",
            lambda g, m, f: Divergence2D(grid=g, mask=m)(f[0], f[1]),
            ("B", "B"),
            1,
        ),
        Case(
            "relative_vorticity",
            lambda g, m, f: Vorticity2D(grid=g, mask=m).relative_vorticity(f[0], f[1]),
            ("B", "B"),
            1,
        ),
        # --- order 2: laplacian, jacobian, harmonic diffusion --------
        Case(
            "laplacian",
            lambda g, m, f: Difference2D(grid=g, mask=m).laplacian(f[0]),
            ("A",),
            2,
        ),
        Case(
            "arakawa_jacobian",
            lambda g, m, f: arakawa_jacobian(f[0], f[1], g.dx, g.dy),
            ("A", "B"),
            2,
            masked=False,
        ),
        Case(
            "harmonic_diffusion",
            lambda g, m, f: Diffusion2D(grid=g, mask=m)(f[0], f[1]),
            ("A", "K"),
            2,
        ),
        # --- order 4: biharmonic -------------------------------------
        Case(
            "biharmonic_diffusion",
            lambda g, m, f: BiharmonicDiffusion2D(grid=g, mask=m)(f[0], 3.0),
            ("A",),
            4,
        ),
        # --- order -2: elliptic solves multiply by length squared ----
        Case(
            "poisson_dst",
            lambda g, m, f: solve_poisson_dst(f[0], g.dx, g.dy),
            ("A",),
            -2,
            masked=False,
        ),
        Case(
            "poisson_fft",
            lambda g, m, f: solve_poisson_fft(f[0], g.dx, g.dy),
            ("A",),
            -2,
            masked=False,
        ),
        # --- spherical counterparts ----------------------------------
        Case(
            "spherical_difference_lon",
            lambda g, m, f: SphericalDifference2D(grid=g, mask=m).diff_lon_T_to_U(f[0]),
            ("A",),
            1,
            grid="spherical",
        ),
        Case(
            "spherical_divergence",
            lambda g, m, f: SphericalDivergence2D(grid=g, mask=m)(f[0], f[1]),
            ("B", "B"),
            1,
            grid="spherical",
        ),
        Case(
            "spherical_vorticity",
            lambda g, m, f: SphericalVorticity2D(grid=g, mask=m).relative_vorticity(
                f[0], f[1]
            ),
            ("B", "B"),
            1,
            grid="spherical",
        ),
        Case(
            "spherical_laplacian",
            lambda g, m, f: SphericalLaplacian2D(grid=g, mask=m)(f[0]),
            ("A",),
            2,
            grid="spherical",
        ),
        Case(
            "spherical_diffusion",
            lambda g, m, f: SphericalDiffusion2D(grid=g, mask=m)(f[0], f[1]),
            ("A", "K"),
            2,
            grid="spherical",
        ),
        Case(
            "spherical_biharmonic",
            lambda g, m, f: SphericalBiharmonicDiffusion2D(grid=g, mask=m)(f[0], 3.0),
            ("A",),
            4,
            grid="spherical",
        ),
        # --- 1-D operators -------------------------------------------
        Case(
            "interp_1d_T_to_U",
            lambda g, m, f: Interpolation1D(grid=g).T_to_U(f[0]),
            ("A",),
            0,
            masked=False,
            dim=1,
        ),
        Case(
            "difference_1d_x_T_to_U",
            lambda g, m, f: Difference1D(grid=g, mask=m).diff_x_T_to_U(f[0]),
            ("A",),
            1,
            dim=1,
        ),
        Case(
            "difference_1d_laplacian",
            lambda g, m, f: Difference1D(grid=g, mask=m).laplacian(f[0]),
            ("A",),
            2,
            dim=1,
        ),
        Case(
            "advection_1d_upwind1",
            lambda g, m, f: Advection1D(grid=g, mask=m)(f[0], f[1], method="upwind1"),
            ("A", "B"),
            1,
            dim=1,
        ),
        # --- 3-D operators -------------------------------------------
        Case(
            "interp_3d_T_to_U",
            lambda g, m, f: Interpolation3D(grid=g).T_to_U(f[0]),
            ("A",),
            0,
            masked=False,
            dim=3,
        ),
        Case(
            "difference_3d_x_T_to_U",
            lambda g, m, f: Difference3D(grid=g, mask=m).diff_x_T_to_U(f[0]),
            ("A",),
            1,
            dim=3,
        ),
        Case(
            "difference_3d_divergence",
            lambda g, m, f: Difference3D(grid=g, mask=m).divergence(f[0], f[1]),
            ("B", "B"),
            1,
            dim=3,
        ),
        Case(
            "difference_3d_laplacian",
            lambda g, m, f: Difference3D(grid=g, mask=m).laplacian(f[0]),
            ("A",),
            2,
            dim=3,
        ),
        Case(
            "relative_vorticity_3d",
            lambda g, m, f: Vorticity3D(grid=g, mask=m).relative_vorticity(f[0], f[1]),
            ("B", "B"),
            1,
            dim=3,
        ),
        Case(
            "harmonic_diffusion_3d",
            lambda g, m, f: Diffusion3D(grid=g, mask=m)(f[0], f[1]),
            ("A", "K"),
            2,
            dim=3,
        ),
        Case(
            "biharmonic_diffusion_3d",
            lambda g, m, f: BiharmonicDiffusion3D(grid=g, mask=m)(f[0], 3.0),
            ("A",),
            4,
            dim=3,
        ),
        Case(
            "advection_3d_upwind1",
            lambda g, m, f: Advection3D(grid=g, mask=m)(
                f[0], f[1], f[2], method="upwind1"
            ),
            ("A", "B", "B"),
            1,
            dim=3,
        ),
        Case(
            "spherical_difference_3d_lon",
            lambda g, m, f: SphericalDifference3D(grid=g, mask=m).diff_lon_T_to_U(f[0]),
            ("A",),
            1,
            grid="spherical",
            dim=3,
        ),
        Case(
            "spherical_divergence_3d",
            lambda g, m, f: SphericalDivergence3D(grid=g, mask=m)(f[0], f[1]),
            ("B", "B"),
            1,
            grid="spherical",
            dim=3,
        ),
        Case(
            "spherical_vorticity_3d",
            lambda g, m, f: SphericalVorticity3D(grid=g, mask=m).relative_vorticity(
                f[0], f[1]
            ),
            ("B", "B"),
            1,
            grid="spherical",
            dim=3,
        ),
        Case(
            "spherical_laplacian_3d",
            lambda g, m, f: SphericalLaplacian3D(grid=g, mask=m)(f[0]),
            ("A",),
            2,
            grid="spherical",
            dim=3,
        ),
        Case(
            "spherical_diffusion_3d",
            lambda g, m, f: SphericalDiffusion3D(grid=g, mask=m)(f[0], f[1]),
            ("A", "K"),
            2,
            grid="spherical",
            dim=3,
        ),
        Case(
            "spherical_biharmonic_3d",
            lambda g, m, f: SphericalBiharmonicDiffusion3D(grid=g, mask=m)(f[0], 3.0),
            ("A",),
            4,
            grid="spherical",
            dim=3,
        ),
    ]


#: Linear and upwind stencils: exactly covariant in both length and amplitude.
LINEAR_METHODS = ("naive", "upwind1", "upwind2", "upwind3")

#: WENO stencils: covariant in length, but *not* in amplitude, because the
#: smoothness-indicator epsilon is absolute while beta carries units of
#: amplitude squared. Tracked as #243.
WENO_METHODS = ("weno3", "weno5", "wenoz5", "weno7", "weno9")

#: TVD limiters. They guard a slope *ratio* with the absolute
#: ``_TVD_EPS``, so they have the same shape of amplitude dependence
#: #243 describes for WENO — a ratio is dimensionless, but the epsilon
#: added to its denominator is not.
TVD_METHODS = ("minmod", "van_leer", "superbee", "mc")

#: Advection is implemented separately per dimension — different
#: dispatch, different reconstruction paths — so a scale regression in
#: one of them is invisible from the others.
ADVECTION_DIMS = (1, 2, 3)


#: Advection operator and amplitude labels per ``(geometry, dim)``.
#: The spherical operators have their own metric and divergence paths,
#: so a scale regression in one is invisible from the Cartesian ones.
ADVECTION_OPERATORS = {
    ("cartesian", 1): (Advection1D, ("A", "B")),
    ("cartesian", 2): (Advection2D, ("A", "B", "B")),
    ("cartesian", 3): (Advection3D, ("A", "B", "B")),
    ("spherical", 2): (SphericalAdvection2D, ("A", "B", "B")),
    ("spherical", 3): (SphericalAdvection3D, ("A", "B", "B")),
}


def _advection_case(method, dim, geometry="cartesian"):
    """One advection case, for whichever operator the pair selects."""
    operator, labels = ADVECTION_OPERATORS[(geometry, dim)]
    prefix = "advection" if geometry == "cartesian" else "spherical_advection"

    def run(g, m, f, method=method, operator=operator):
        return operator(grid=g, mask=m)(*f, method=method)

    return Case(f"{prefix}_{dim}d_{method}", run, labels, 1, grid=geometry, dim=dim)


#: Not every dimension implements every scheme — ``Advection3D`` has no
#: ``upwind2``/``upwind3``, and only the 2-D operator has ``wenoz5``.
#: Probed rather than hard-coded so the matrix cannot drift out of date,
#: and asserted non-empty below so a probe that silently matches nothing
#: cannot quietly empty the suite.
def _supported(operator, grid, fields, method):
    # Only the deliberate "this operator has no such scheme" error
    # means unsupported. Any other ``ValueError`` — a construction or
    # execution failure in a scheme that *is* advertised — would
    # otherwise drop that case from the matrix and leave the suite
    # green while the operator is broken, so it is re-raised.
    try:
        operator(grid=grid)(*fields, method=method)
    except ValueError as error:
        if "unknown method" in str(error).lower():
            return False
        raise
    return True


def _advection_cases(methods):
    cases = []
    for geometry, dim in ADVECTION_OPERATORS:
        if dim not in ADVECTION_DIMS:
            continue
        grid = GRID_FACTORIES[(dim, geometry)](1.0)
        make_field = FIELD_MAKERS[dim]
        operator, labels = ADVECTION_OPERATORS[(geometry, dim)]
        probe = [make_field(seed) for seed in range(len(labels))]
        for method in methods:
            if _supported(operator, grid, probe, method):
                cases.append(_advection_case(method, dim, geometry))
    return cases


CASES = _cases()
LINEAR_ADVECTION_CASES = _advection_cases(LINEAR_METHODS)
WENO_ADVECTION_CASES = _advection_cases(WENO_METHODS)
TVD_ADVECTION_CASES = _advection_cases(TVD_METHODS)
ADVECTION_CASES = LINEAR_ADVECTION_CASES + WENO_ADVECTION_CASES + TVD_ADVECTION_CASES


def run_case(case, length_factor, amplitude, use_mask, shape=DEFAULT_SHAPE):
    """Run a case on the base grid and on the rescaled grid.

    Returns ``(base_output, scaled_output, expected_factor)``.
    """
    n_fields = len(case.labels)
    make_field = FIELD_MAKERS[case.dim]
    fields = [make_field(seed, shape) for seed in range(n_fields)]

    amplitudes = {
        label: amplitude * spread
        for label, spread in zip(dict.fromkeys(case.labels), LABEL_SPREAD, strict=False)
    }
    scaled_fields = [
        f / amplitudes[label] for f, label in zip(fields, case.labels, strict=True)
    ]

    make_grid = GRID_FACTORIES[(case.dim, case.grid)]
    mask = MASK_MAKERS[case.dim](shape) if (use_mask and case.masked) else None

    base = case.run(make_grid(1.0, shape), mask, fields)
    scaled = case.run(make_grid(length_factor, shape), mask, scaled_fields)

    amplitude_product = np.prod(list(amplitudes.values()))
    factor = length_factor**case.order / amplitude_product
    return np.asarray(base), np.asarray(scaled), factor


def assert_covariant(
    case, length_factor, amplitude, use_mask, rtol=1e-10, shape=DEFAULT_SHAPE
):
    base, scaled, factor = run_case(case, length_factor, amplitude, use_mask, shape)
    expected = base * factor
    # Compare relative to the expected magnitude, so a large ``factor``
    # does not make the check vacuous.
    scale = max(np.abs(expected).max(), 1e-300)
    err = np.abs(scaled - expected).max() / scale
    assert err < rtol, (
        f"{case.name}: not scale-covariant "
        f"(L={length_factor:g}, amp={amplitude:g}, mask={use_mask}); "
        f"relative error {err:.3e}"
    )


class TestScaleCovariance:
    @pytest.mark.parametrize("case", CASES, ids=repr)
    @pytest.mark.parametrize("length_factor", LENGTH_FACTORS)
    def test_length_rescaling(self, case, length_factor):
        assert_covariant(case, length_factor, 1.0, use_mask=False)

    @pytest.mark.parametrize("case", CASES, ids=repr)
    @pytest.mark.parametrize("amplitude", AMPLITUDE_FACTORS)
    def test_amplitude_rescaling(self, case, amplitude):
        assert_covariant(case, 1.0, amplitude, use_mask=False)

    @pytest.mark.parametrize("case", CASES, ids=repr)
    def test_combined_rescaling(self, case):
        assert_covariant(case, 1e6, 1e-2, use_mask=False)

    @pytest.mark.parametrize("case", [c for c in CASES if c.masked], ids=repr)
    def test_masked_is_covariant_too(self, case):
        assert_covariant(case, 1e-3, 1e3, use_mask=True)

    @pytest.mark.parametrize(
        "case", [c for c in CASES + ADVECTION_CASES if c.masked], ids=repr
    )
    def test_masked_cells_stay_zero_on_both_sides(self, case):
        """Zeroed by the mask, and still zeroed after rescaling.

        The cells to check are found by differencing against an
        *unmasked* run rather than by reading the mask: an operator's
        output stagger is not declared on the case, so which of
        ``mask.h`` / ``.u`` / ``.v`` applies is not known here. Cells
        that the masked run zeroes and the unmasked run does not are
        exactly the ones masking is responsible for — which also makes
        the check fail if an operator stops masking at all, instead of
        quietly falling back to the structural ghost-ring zeros.

        The generated advection cases are included here rather than
        left to their own class: covariance is all that class checks,
        and an operator that stopped masking altogether would still
        satisfy it.
        """
        masked, masked_scaled, _ = run_case(case, 1e6, 1e-2, use_mask=True)
        unmasked, _, _ = run_case(case, 1e6, 1e-2, use_mask=False)

        zeroed_by_the_mask = (masked == 0.0) & (unmasked != 0.0)
        assert zeroed_by_the_mask.any(), (
            f"{case.name}: masking zeroed no cell that was nonzero without "
            f"it — the mask is not reaching the operator"
        )
        np.testing.assert_array_equal(
            masked_scaled[zeroed_by_the_mask],
            np.zeros(int(zeroed_by_the_mask.sum())),
        )


class TestAdvectionScaleCovariance:
    """Reconstruction stencils are ratio-based and must be exactly covariant.

    An amplitude failure here means a limiter guards a ratio with an
    epsilon in absolute units (a bare constant added to a squared
    gradient), which quietly changes the scheme when the field is
    rescaled — see :class:`TestWenoAmplitudeCovariance`.
    """

    @pytest.mark.parametrize("case", ADVECTION_CASES, ids=repr)
    @pytest.mark.parametrize("length_factor", LENGTH_FACTORS)
    def test_length_rescaling(self, case, length_factor):
        """Every scheme, WENO included, is exact under grid rescaling."""
        assert_covariant(case, length_factor, 1.0, use_mask=False)

    @pytest.mark.parametrize("case", LINEAR_ADVECTION_CASES, ids=repr)
    @pytest.mark.parametrize("amplitude", AMPLITUDE_FACTORS)
    def test_amplitude_rescaling(self, case, amplitude):
        assert_covariant(case, 1.0, amplitude, use_mask=False)

    @pytest.mark.parametrize("case", LINEAR_ADVECTION_CASES, ids=repr)
    def test_masked_advection_is_covariant(self, case):
        assert_covariant(case, 1e-3, 1e3, use_mask=True)


class TestWenoAmplitudeCovariance:
    """WENO weights currently depend on the amplitude of the field.

    ``beta`` (the smoothness indicator) has units of amplitude squared,
    but the epsilon added to it in ``weno.py`` is a bare absolute
    constant. Rescaling the field therefore shifts the nonlinear
    weights: a 3% change in the tendency at amplitude 1e3. Tracked as
    #243; these xfails flip to passes once the epsilon is made relative.
    """

    @pytest.mark.xfail(
        reason="#243: absolute epsilon in the WENO smoothness indicators",
        strict=True,
        # Only the numerical assertion is expected to fail. Without
        # this, a WENO operator that started raising during
        # construction or execution would still report XFAIL and keep
        # the suite green.
        raises=AssertionError,
    )
    @pytest.mark.parametrize("case", WENO_ADVECTION_CASES, ids=repr)
    def test_amplitude_rescaling(self, case):
        assert_covariant(case, 1.0, 1e3, use_mask=False)

    @pytest.mark.xfail(
        reason="#243: absolute epsilon in the WENO smoothness indicators",
        strict=True,
        # Only the numerical assertion is expected to fail. Without
        # this, a WENO operator that started raising during
        # construction or execution would still report XFAIL and keep
        # the suite green.
        raises=AssertionError,
    )
    @pytest.mark.parametrize("case", WENO_ADVECTION_CASES, ids=repr)
    def test_masked_advection_is_covariant(self, case):
        assert_covariant(case, 1e-3, 1e3, use_mask=True)

    @pytest.mark.parametrize("case", WENO_ADVECTION_CASES, ids=repr)
    def test_departure_is_bounded_until_243_lands(self, case):
        """Pin the size of the known breakage so it cannot silently grow."""
        base, scaled, factor = run_case(case, 1.0, 1e3, use_mask=False)
        expected = base * factor
        err = np.abs(scaled - expected).max() / np.abs(expected).max()
        assert err < 5e-2


# ----------------------------------------------------------------------
# Constants audit
# ----------------------------------------------------------------------

PHYSICAL_CONSTANTS = {"R_EARTH", "GRAVITY", "OMEGA", "RHO", "DEG2M"}

AUDITED_PACKAGES = ("operators", "advection", "diffusion", "solvers")


def _audited_modules():
    root = pathlib.Path(__file__).resolve().parents[1] / "finitevolx" / "_src"
    for package in AUDITED_PACKAGES:
        yield from sorted((root / package).rglob("*.py"))


#: Dotted path of the module the constants live in.
CONSTANTS_MODULE = "finitevolx._src.utils.constants"


def _constant_aliases(tree):
    """Local names that resolve to a physical constant, and to its module.

    ``from ...constants import GRAVITY as g`` binds the constant to
    ``g``, so matching the original name alone would miss every later
    use of it. Returns ``(constant_aliases, module_aliases)``: the
    first maps a local name to the constant it came from (including
    the identity entries for unaliased imports), the second is the set
    of local names bound to the constants *module* itself.
    """
    constants = {}
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for imported in node.names:
                if imported.name in PHYSICAL_CONSTANTS:
                    constants[imported.asname or imported.name] = imported.name
                elif f"{module}.{imported.name}" == CONSTANTS_MODULE:
                    modules.add(imported.asname or imported.name)
        elif isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == CONSTANTS_MODULE:
                    # ``import a.b.constants`` binds ``a``; only the
                    # aliased form binds a name we can follow.
                    if imported.asname:
                        modules.add(imported.asname)
    return constants, modules


def _referenced_constant(node, aliases, modules):
    """The physical constant an expression node refers to, if any.

    Covers the three spellings a module can reach one by: the bare
    imported name, a local alias of it, and an attribute access such
    as ``constants.GRAVITY`` on the imported module.

    The attribute form is only accepted when its base name is bound to
    the constants module. Matching on the attribute name alone would
    fail the audit on innocent code such as ``state.RHO`` or
    ``config.GRAVITY``, which have nothing to do with
    :mod:`finitevolx._src.utils.constants`.
    """
    if isinstance(node, ast.Name):
        return aliases.get(node.id) or (
            node.id if node.id in PHYSICAL_CONSTANTS else None
        )
    if (
        isinstance(node, ast.Attribute)
        and node.attr in PHYSICAL_CONSTANTS
        and isinstance(node.value, ast.Name)
        and node.value.id in modules
    ):
        return node.attr
    return None


def _constants_outside_defaults(path):
    """Names of physical constants used anywhere but as a parameter default.

    A constant reached through an explicit, overridable parameter
    default (``gravity: float = GRAVITY``) keeps the operator
    unit-agnostic — the caller can pass 1. A constant referenced inside
    an expression bakes SI units into the numerics.

    Qualified (``constants.GRAVITY``) and aliased
    (``GRAVITY as g``) references count: otherwise a routine change of
    import style would quietly switch the contract off.
    """
    tree = ast.parse(path.read_text())
    aliases, modules = _constant_aliases(tree)

    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None
            ]
            for default in defaults:
                for sub in ast.walk(default):
                    allowed.add(id(sub))

    offenders = []
    for node in ast.walk(tree):
        # An ``import ... as`` clause binds an alias; ``ast.alias`` is
        # neither a Name nor an Attribute, so it is not a use.
        constant = _referenced_constant(node, aliases, modules)
        if constant is not None and id(node) not in allowed:
            offenders.append(f"{constant} (line {node.lineno})")
    return offenders


class TestConstantsAudit:
    def test_no_physical_constants_inside_operator_expressions(self):
        """Operators may default to SI constants, never hard-code them."""
        violations = {}
        for path in _audited_modules():
            offenders = _constants_outside_defaults(path)
            if offenders:
                violations[str(path.name)] = offenders
        assert not violations, (
            "Physical constants used outside a parameter default — these bake "
            f"SI units into the numerics: {violations}"
        )

    def test_audit_actually_scans_the_operator_modules(self):
        """Guard against the audit silently scanning nothing."""
        modules = list(_audited_modules())
        assert len(modules) > 20
        names = {p.name for p in modules}
        assert "difference.py" in names
        assert "spherical_diffusion.py" in names

    def test_audit_detects_an_aliased_constant(self, tmp_path):
        """``GRAVITY as g`` is the same constant under another name."""
        offending = tmp_path / "aliased.py"
        offending.write_text(
            "from finitevolx._src.utils.constants import GRAVITY as g\n"
            "def f(h):\n"
            "    return g * h\n"
        )
        assert _constants_outside_defaults(offending) == ["GRAVITY (line 3)"]

    def test_audit_detects_a_qualified_constant(self, tmp_path):
        """``constants.GRAVITY`` bakes in SI just as surely."""
        offending = tmp_path / "qualified.py"
        offending.write_text(
            "from finitevolx._src.utils import constants\n"
            "def f(h):\n"
            "    return constants.GRAVITY * h\n"
        )
        assert _constants_outside_defaults(offending) == ["GRAVITY (line 3)"]

    def test_audit_allows_an_aliased_default(self, tmp_path):
        """The alias is still overridable when it is only a default."""
        fine = tmp_path / "aliased_default.py"
        fine.write_text(
            "from finitevolx._src.utils.constants import GRAVITY as g\n"
            "def f(h, gravity=g):\n"
            "    return gravity * h\n"
        )
        assert not _constants_outside_defaults(fine)

    def test_audit_allows_a_qualified_default(self, tmp_path):
        fine = tmp_path / "qualified_default.py"
        fine.write_text(
            "from finitevolx._src.utils import constants\n"
            "def f(h, gravity=constants.GRAVITY):\n"
            "    return gravity * h\n"
        )
        assert not _constants_outside_defaults(fine)

    def test_audit_ignores_an_unrelated_attribute_of_the_same_name(self, tmp_path):
        """``state.RHO`` is not the constants module's ``RHO``.

        Matching on the attribute name alone would fail the audit on
        any object that happens to carry one of these names.
        """
        fine = tmp_path / "unrelated_attribute.py"
        fine.write_text("def f(state, h):\n    return state.RHO * h\n")
        assert not _constants_outside_defaults(fine)

    def test_audit_still_catches_it_through_a_module_alias(self, tmp_path):
        """Renaming the module on import must not switch the audit off."""
        offending = tmp_path / "module_alias.py"
        offending.write_text(
            "from finitevolx._src.utils import constants as c\n"
            "def f(h):\n"
            "    return c.RHO * h\n"
        )
        assert _constants_outside_defaults(offending) == ["RHO (line 3)"]

    def test_audit_detects_a_hard_coded_constant(self, tmp_path):
        """The audit must fail on the pattern it exists to catch."""
        offending = tmp_path / "bad_operator.py"
        offending.write_text(
            "from finitevolx._src.utils.constants import GRAVITY\n"
            "def tendency(h, dx):\n"
            "    return GRAVITY * h / dx\n"
        )
        assert _constants_outside_defaults(offending)

    def test_audit_allows_an_overridable_default(self, tmp_path):
        fine = tmp_path / "good_operator.py"
        fine.write_text(
            "from finitevolx._src.utils.constants import GRAVITY\n"
            "def tendency(h, dx, g: float = GRAVITY):\n"
            "    return g * h / dx\n"
        )
        assert not _constants_outside_defaults(fine)


class TestTvdAmplitudeCovariance:
    """TVD limiters share WENO's defect, three orders of magnitude smaller.

    Each guards a slope *ratio* with the absolute ``_TVD_EPS = 1e-8``.
    A ratio is dimensionless but that epsilon is not, so rescaling the
    field shifts the limiter slightly — the same mechanism as #243.

    The size is what differs, and it is why these are not ``xfail``
    like the WENO cases: at amplitude 1e3 a WENO tendency moves by 3%,
    a TVD one by about 1e-5. Too large to call exact, far too small to
    call broken. The bound below states that, so a regression toward
    WENO-scale breakage fails here.
    """

    #: Measured departure is ~1.5e-5; an order of magnitude of headroom.
    TOLERANCE = 1e-4

    @pytest.mark.parametrize("case", TVD_ADVECTION_CASES, ids=repr)
    @pytest.mark.parametrize("length_factor", LENGTH_FACTORS)
    def test_length_rescaling_is_exact(self, case, length_factor):
        """The grid carries no amplitude, so this half is unaffected."""
        assert_covariant(case, length_factor, 1.0, use_mask=False)

    @pytest.mark.parametrize("case", TVD_ADVECTION_CASES, ids=repr)
    def test_amplitude_departure_is_bounded(self, case):
        base, scaled, factor = run_case(case, 1.0, 1e3, use_mask=False)
        expected = base * factor
        err = np.abs(scaled - expected).max() / np.abs(expected).max()
        assert err < self.TOLERANCE, f"{case.name}: amplitude error {err:.3e}"

    @pytest.mark.parametrize("case", TVD_ADVECTION_CASES, ids=repr)
    def test_the_departure_is_real_and_not_rounding(self, case):
        """Otherwise the bound above could be met by an exact operator.

        Pinning this means that if #243's fix also makes the limiters
        exact, this test fails and the suite is updated deliberately
        rather than keeping a stale allowance.
        """
        base, scaled, factor = run_case(case, 1.0, 1e3, use_mask=False)
        expected = base * factor
        err = np.abs(scaled - expected).max() / np.abs(expected).max()
        assert err > 1e-8

    @pytest.mark.parametrize("case", TVD_ADVECTION_CASES, ids=repr)
    def test_masked_departure_is_bounded_too(self, case):
        base, scaled, factor = run_case(case, 1e-3, 1e3, use_mask=True)
        expected = base * factor
        err = np.abs(scaled - expected).max() / np.abs(expected).max()
        assert err < self.TOLERANCE


class TestTheContractsHoldOnAnotherGrid:
    """Same contracts, a different array shape.

    Every other parameterization varies physical lengths, not array
    sizes, so an indexing or spectral-transform regression confined to
    odd extents, to a non-square grid, or to a different number of
    vertical levels would pass the whole suite. This re-runs the
    matrix at :data:`ALTERNATE_SHAPE`, which is all three.

    The WENO and TVD cases keep their own tolerances — the grid shape
    does not change #243's amplitude dependence.
    """

    @pytest.mark.parametrize("case", CASES + LINEAR_ADVECTION_CASES, ids=repr)
    def test_combined_rescaling(self, case):
        assert_covariant(case, 1e6, 1e-2, use_mask=False, shape=ALTERNATE_SHAPE)

    @pytest.mark.parametrize(
        "case",
        [c for c in CASES + LINEAR_ADVECTION_CASES if c.masked],
        ids=repr,
    )
    def test_masked_is_covariant_too(self, case):
        assert_covariant(case, 1e-3, 1e3, use_mask=True, shape=ALTERNATE_SHAPE)

    @pytest.mark.parametrize(
        "case", WENO_ADVECTION_CASES + TVD_ADVECTION_CASES, ids=repr
    )
    def test_length_rescaling_is_still_exact(self, case):
        """The length half is exact for every scheme, #243 or not."""
        assert_covariant(case, 1e6, 1.0, use_mask=False, shape=ALTERNATE_SHAPE)


class TestTheCaseMatrixIsPopulated:
    """A probe that matched nothing would silently empty the suite."""

    @pytest.mark.parametrize("dim", ADVECTION_DIMS)
    def test_every_dimension_contributes_advection_cases(self, dim):
        for group in (
            LINEAR_ADVECTION_CASES,
            WENO_ADVECTION_CASES,
            TVD_ADVECTION_CASES,
        ):
            assert any(case.dim == dim for case in group)

    def test_every_dimension_contributes_operator_cases(self):
        for dim in (1, 2, 3):
            assert any(case.dim == dim for case in CASES)

    def test_both_geometries_are_covered(self):
        for geometry in ("cartesian", "spherical"):
            assert any(case.grid == geometry for case in CASES)
