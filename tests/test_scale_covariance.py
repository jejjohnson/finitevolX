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

from finitevolx._src.advection.advection import Advection2D
from finitevolx._src.diffusion.diffusion import (
    BiharmonicDiffusion2D,
    Diffusion2D,
)
from finitevolx._src.diffusion.spherical_diffusion import (
    SphericalBiharmonicDiffusion2D,
    SphericalDiffusion2D,
)
from finitevolx._src.grid.cartesian import CartesianGrid2D
from finitevolx._src.grid.spherical import SphericalGrid2D
from finitevolx._src.mask.cartesian import Mask2D
from finitevolx._src.operators.difference import Difference2D
from finitevolx._src.operators.divergence import Divergence2D
from finitevolx._src.operators.interpolation import Interpolation2D
from finitevolx._src.operators.jacobian import arakawa_jacobian
from finitevolx._src.operators.spherical_compound import (
    SphericalDivergence2D,
    SphericalLaplacian2D,
    SphericalVorticity2D,
)
from finitevolx._src.operators.spherical_difference import (
    SphericalDifference2D,
)
from finitevolx._src.operators.vorticity import Vorticity2D
from finitevolx._src.solvers.spectral import (
    solve_poisson_dst,
    solve_poisson_fft,
)

NX = NY = 16
LX = LY = 2.0

# Grid rescalings and field amplitudes spanning the physical -> nondimensional
# range. A purely relative bug survives L = U = 1; an absolute-tolerance bug
# only shows up at the extremes.
LENGTH_FACTORS = [1e-3, 1.0, 1e6]
AMPLITUDE_FACTORS = [1e-2, 1.0, 1e3]


def smooth_field(seed, ny=NY, nx=NX):
    """A smooth O(1) field — smooth so that limiters behave consistently."""
    rng = np.random.RandomState(seed)
    y = np.linspace(0.0, 2.0 * np.pi, ny)[:, None]
    x = np.linspace(0.0, 2.0 * np.pi, nx)[None, :]
    out = np.zeros((ny, nx))
    for k in (1, 2, 3):
        out += rng.randn() * np.sin(k * x + rng.randn()) * np.cos(k * y + rng.randn())
    return jnp.asarray(out / np.abs(out).max())


def cartesian_grid(length_factor):
    return CartesianGrid2D.from_interior(
        NX - 2, NY - 2, LX / length_factor, LY / length_factor
    )


def spherical_grid(length_factor):
    """Angles are invariant under rescaling; only the radius carries length."""
    return SphericalGrid2D.from_interior(
        NX - 2,
        NY - 2,
        lon_range=(0.0, 40.0),
        lat_range=(10.0, 50.0),
        R=6.0e6 / length_factor,
    )


def land_mask():
    wet = np.ones((NY, NX), dtype=bool)
    wet[4:7, 5:8] = False
    return Mask2D.from_mask(jnp.asarray(wet))


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


class Case:
    def __init__(self, name, run, labels, order, grid="cartesian", masked=True):
        self.name = name
        self.run = run
        self.labels = labels
        self.order = order
        self.grid = grid
        self.masked = masked

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
    ]


#: Linear and upwind stencils: exactly covariant in both length and amplitude.
LINEAR_METHODS = ("naive", "upwind1", "upwind2", "upwind3")

#: WENO stencils: covariant in length, but *not* in amplitude, because the
#: smoothness-indicator epsilon is absolute while beta carries units of
#: amplitude squared. Tracked as #243.
WENO_METHODS = ("weno3", "weno5", "wenoz5", "weno7", "weno9")


def _advection_case(method):
    return Case(
        f"advection_{method}",
        lambda g, m, f, method=method: Advection2D(grid=g, mask=m)(
            f[0], f[1], f[2], method=method
        ),
        ("A", "B", "B"),
        1,
    )


CASES = _cases()
LINEAR_ADVECTION_CASES = [_advection_case(m) for m in LINEAR_METHODS]
WENO_ADVECTION_CASES = [_advection_case(m) for m in WENO_METHODS]
ADVECTION_CASES = LINEAR_ADVECTION_CASES + WENO_ADVECTION_CASES


def run_case(case, length_factor, amplitude, use_mask):
    """Run a case on the base grid and on the rescaled grid.

    Returns ``(base_output, scaled_output, expected_factor)``.
    """
    n_fields = len(case.labels)
    fields = [smooth_field(seed) for seed in range(n_fields)]

    amplitudes = {label: amplitude for label in dict.fromkeys(case.labels)}
    scaled_fields = [
        f / amplitudes[label] for f, label in zip(fields, case.labels, strict=True)
    ]

    make_grid = spherical_grid if case.grid == "spherical" else cartesian_grid
    mask = land_mask() if (use_mask and case.masked) else None

    base = case.run(make_grid(1.0), mask, fields)
    scaled = case.run(make_grid(length_factor), mask, scaled_fields)

    amplitude_product = np.prod(list(amplitudes.values()))
    factor = length_factor**case.order / amplitude_product
    return np.asarray(base), np.asarray(scaled), factor


def assert_covariant(case, length_factor, amplitude, use_mask, rtol=1e-10):
    base, scaled, factor = run_case(case, length_factor, amplitude, use_mask)
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

    @pytest.mark.parametrize("case", [c for c in CASES if c.masked], ids=repr)
    def test_masked_cells_stay_zero_on_both_sides(self, case):
        base, scaled, _ = run_case(case, 1e6, 1e-2, use_mask=True)
        zero_in_base = base == 0.0
        np.testing.assert_array_equal(
            scaled[zero_in_base] == 0.0, np.ones(zero_in_base.sum(), dtype=bool)
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
    )
    @pytest.mark.parametrize("case", WENO_ADVECTION_CASES, ids=repr)
    def test_amplitude_rescaling(self, case):
        assert_covariant(case, 1.0, 1e3, use_mask=False)

    @pytest.mark.xfail(
        reason="#243: absolute epsilon in the WENO smoothness indicators",
        strict=True,
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


def _constants_outside_defaults(path):
    """Names of physical constants used anywhere but as a parameter default.

    A constant reached through an explicit, overridable parameter
    default (``gravity: float = GRAVITY``) keeps the operator
    unit-agnostic — the caller can pass 1. A constant referenced inside
    an expression bakes SI units into the numerics.
    """
    tree = ast.parse(path.read_text())

    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None
            ]
            for default in defaults:
                for sub in ast.walk(default):
                    if isinstance(sub, ast.Name):
                        allowed.add(id(sub))

    offenders = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id in PHYSICAL_CONSTANTS
            and id(node) not in allowed
        ):
            offenders.append(f"{node.id} (line {node.lineno})")
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
