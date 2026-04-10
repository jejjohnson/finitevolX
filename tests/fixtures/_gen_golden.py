"""Generate golden ``.npz`` files for the masks-everywhere regression suite.

Run this script (NOT pytest) to (re)generate every golden file under
``tests/fixtures/golden/``::

    uv run python tests/fixtures/_gen_golden.py

The generator is intentionally split per operator family.  Adding a new
operator (or a new method on an existing operator) means:

1. Add the corresponding entry in :func:`_register_all`.
2. Re-run this script.
3. ``git add tests/fixtures/golden/<operator>__<method>__*.npz``
4. Update the matching ``test_<operator>.py`` to call
   ``assert_matches_golden(...)``.

Each registered entry is a dictionary describing one operator method on
one input + one variant ("unmasked", "masked", or "all_ocean") and the
zero-arg callable that produces the output.  This makes the registration
list itself a single readable description of "what each operator is
expected to compute on the canonical input."

The generator never imports from any of the operator modules at import
time; instead :func:`_register_all` does the imports inside the
function body so a partial check-in (e.g. an operator added but its
mask-aware variant not yet wired up) does not break the script.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys

# Make this script runnable directly via `python tests/fixtures/_gen_golden.py`
# (and via `uv run python ...`).  Python prepends the script's *own* directory
# (``tests/fixtures``) to ``sys.path`` on script invocation, which would
# shadow the top-level ``tests`` package, and ``uv`` may install an unrelated
# package called ``tests`` into the environment's site-packages.  Force the
# repo root to the very front of ``sys.path`` so that ``tests.fixtures``
# resolves to *this* repo's tests directory and not anything else.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)
while _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from tests.fixtures._helpers import save_golden
from tests.fixtures.inputs import (
    make_grid_2d,
    make_grid_3d,
    make_h_field_2d,
    make_h_field_3d,
    make_mask_2d,
    make_q_field_2d,
    make_u_field_2d,
    make_u_field_3d,
    make_v_field_2d,
    make_v_field_3d,
)

# A registered entry: (operator_name, method_name, variant, callable).
Entry = tuple[str, str, str, Callable[[], object]]


def _register_all() -> list[Entry]:
    """Return every (operator, method, variant, fn) tuple to materialize.

    The list is grouped by operator family.  Within each family the
    "unmasked" entry is registered first, then "masked", so that
    ``ls tests/fixtures/golden/`` reads top-to-bottom in roughly the
    same order as the operator source files.

    Adding a new method:
        Insert a new tuple inside the appropriate group.  Both the
        ``unmasked`` and ``masked`` variants are recommended for any
        operator with mask support, so that the test suite can verify
        both code paths and confirm that ``mask=None`` does not perturb
        the unmasked output.
    """
    entries: list[Entry] = []
    grid2d = make_grid_2d()
    grid3d = make_grid_3d()
    mask2d = make_mask_2d()
    h2d = make_h_field_2d()
    u2d = make_u_field_2d()
    v2d = make_v_field_2d()
    q2d = make_q_field_2d()
    h3d = make_h_field_3d()
    u3d = make_u_field_3d()
    v3d = make_v_field_3d()

    # ------------------------------------------------------------------
    # Difference2D / Difference3D
    # ------------------------------------------------------------------
    from finitevolx._src.operators.difference import Difference2D, Difference3D

    diff2d = Difference2D(grid=grid2d)
    diff3d = Difference3D(grid=grid3d)

    # 2D — single-input methods, exhaustive coverage of every public method.
    diff2d_methods: list[
        tuple[str, Callable[[], object]]  # (method_name, masked-output factory)
    ] = [
        ("diff_x_T_to_U", lambda: diff2d.diff_x_T_to_U(h2d, mask=mask2d)),
        ("diff_y_T_to_V", lambda: diff2d.diff_y_T_to_V(h2d, mask=mask2d)),
        ("diff_y_U_to_X", lambda: diff2d.diff_y_U_to_X(u2d, mask=mask2d)),
        ("diff_x_V_to_X", lambda: diff2d.diff_x_V_to_X(v2d, mask=mask2d)),
        ("diff_y_X_to_U", lambda: diff2d.diff_y_X_to_U(q2d, mask=mask2d)),
        ("diff_x_X_to_V", lambda: diff2d.diff_x_X_to_V(q2d, mask=mask2d)),
        ("diff_x_U_to_T", lambda: diff2d.diff_x_U_to_T(u2d, mask=mask2d)),
        ("diff_y_V_to_T", lambda: diff2d.diff_y_V_to_T(v2d, mask=mask2d)),
        ("divergence", lambda: diff2d.divergence(u2d, v2d, mask=mask2d)),
        ("curl", lambda: diff2d.curl(u2d, v2d, mask=mask2d)),
        ("laplacian", lambda: diff2d.laplacian(h2d, mask=mask2d)),
        ("grad_perp", lambda: diff2d.grad_perp(h2d, mask=mask2d)),
    ]
    diff2d_unmasked: list[tuple[str, Callable[[], object]]] = [
        ("diff_x_T_to_U", lambda: diff2d.diff_x_T_to_U(h2d)),
        ("diff_y_T_to_V", lambda: diff2d.diff_y_T_to_V(h2d)),
        ("diff_y_U_to_X", lambda: diff2d.diff_y_U_to_X(u2d)),
        ("diff_x_V_to_X", lambda: diff2d.diff_x_V_to_X(v2d)),
        ("diff_y_X_to_U", lambda: diff2d.diff_y_X_to_U(q2d)),
        ("diff_x_X_to_V", lambda: diff2d.diff_x_X_to_V(q2d)),
        ("diff_x_U_to_T", lambda: diff2d.diff_x_U_to_T(u2d)),
        ("diff_y_V_to_T", lambda: diff2d.diff_y_V_to_T(v2d)),
        ("divergence", lambda: diff2d.divergence(u2d, v2d)),
        ("curl", lambda: diff2d.curl(u2d, v2d)),
        ("laplacian", lambda: diff2d.laplacian(h2d)),
        ("grad_perp", lambda: diff2d.grad_perp(h2d)),
    ]
    for method, fn in diff2d_unmasked:
        entries.append(("Difference2D", method, "unmasked", fn))
    for method, fn in diff2d_methods:
        entries.append(("Difference2D", method, "masked", fn))

    # 3D — subset of methods (exhaustive 3D coverage isn't load-bearing
    # because each method is a vmap over the 2D version).
    diff3d_unmasked: list[tuple[str, Callable[[], object]]] = [
        ("diff_x_T_to_U", lambda: diff3d.diff_x_T_to_U(h3d)),
        ("diff_y_T_to_V", lambda: diff3d.diff_y_T_to_V(h3d)),
        ("diff_x_U_to_T", lambda: diff3d.diff_x_U_to_T(u3d)),
        ("diff_y_V_to_T", lambda: diff3d.diff_y_V_to_T(v3d)),
        ("divergence", lambda: diff3d.divergence(u3d, v3d)),
        ("laplacian", lambda: diff3d.laplacian(h3d)),
    ]
    diff3d_masked: list[tuple[str, Callable[[], object]]] = [
        ("diff_x_T_to_U", lambda: diff3d.diff_x_T_to_U(h3d, mask=mask2d)),
        ("diff_y_T_to_V", lambda: diff3d.diff_y_T_to_V(h3d, mask=mask2d)),
        ("diff_x_U_to_T", lambda: diff3d.diff_x_U_to_T(u3d, mask=mask2d)),
        ("diff_y_V_to_T", lambda: diff3d.diff_y_V_to_T(v3d, mask=mask2d)),
        ("divergence", lambda: diff3d.divergence(u3d, v3d, mask=mask2d)),
        ("laplacian", lambda: diff3d.laplacian(h3d, mask=mask2d)),
    ]
    for method, fn in diff3d_unmasked:
        entries.append(("Difference3D", method, "unmasked", fn))
    for method, fn in diff3d_masked:
        entries.append(("Difference3D", method, "masked", fn))

    # ------------------------------------------------------------------
    # Divergence2D
    # ------------------------------------------------------------------
    from finitevolx._src.operators.divergence import Divergence2D

    div2d = Divergence2D(grid=grid2d)
    entries += [
        ("Divergence2D", "__call__", "unmasked", lambda: div2d(u2d, v2d)),
        ("Divergence2D", "__call__", "masked", lambda: div2d(u2d, v2d, mask=mask2d)),
        ("Divergence2D", "noflux", "unmasked", lambda: div2d.noflux(u2d, v2d)),
        (
            "Divergence2D",
            "noflux",
            "masked",
            lambda: div2d.noflux(u2d, v2d, mask=mask2d),
        ),
    ]

    # ------------------------------------------------------------------
    # Interpolation2D / Interpolation3D
    # ------------------------------------------------------------------
    from finitevolx._src.operators.interpolation import (
        Interpolation2D,
        Interpolation3D,
    )

    interp2d = Interpolation2D(grid=grid2d)
    interp3d = Interpolation3D(grid=grid3d)

    interp2d_calls: list[tuple[str, Callable[[], object], Callable[[], object]]] = [
        # (method_name, unmasked_fn, masked_fn)
        (
            "T_to_U",
            lambda: interp2d.T_to_U(h2d),
            lambda: interp2d.T_to_U(h2d, mask=mask2d),
        ),
        (
            "T_to_V",
            lambda: interp2d.T_to_V(h2d),
            lambda: interp2d.T_to_V(h2d, mask=mask2d),
        ),
        (
            "T_to_X",
            lambda: interp2d.T_to_X(h2d),
            lambda: interp2d.T_to_X(h2d, mask=mask2d),
        ),
        (
            "X_to_U",
            lambda: interp2d.X_to_U(q2d),
            lambda: interp2d.X_to_U(q2d, mask=mask2d),
        ),
        (
            "X_to_V",
            lambda: interp2d.X_to_V(q2d),
            lambda: interp2d.X_to_V(q2d, mask=mask2d),
        ),
        (
            "U_to_T",
            lambda: interp2d.U_to_T(u2d),
            lambda: interp2d.U_to_T(u2d, mask=mask2d),
        ),
        (
            "V_to_T",
            lambda: interp2d.V_to_T(v2d),
            lambda: interp2d.V_to_T(v2d, mask=mask2d),
        ),
        (
            "X_to_T",
            lambda: interp2d.X_to_T(q2d),
            lambda: interp2d.X_to_T(q2d, mask=mask2d),
        ),
        (
            "U_to_X",
            lambda: interp2d.U_to_X(u2d),
            lambda: interp2d.U_to_X(u2d, mask=mask2d),
        ),
        (
            "V_to_X",
            lambda: interp2d.V_to_X(v2d),
            lambda: interp2d.V_to_X(v2d, mask=mask2d),
        ),
        (
            "U_to_V",
            lambda: interp2d.U_to_V(u2d),
            lambda: interp2d.U_to_V(u2d, mask=mask2d),
        ),
        (
            "V_to_U",
            lambda: interp2d.V_to_U(v2d),
            lambda: interp2d.V_to_U(v2d, mask=mask2d),
        ),
    ]
    for method, unmasked_fn, masked_fn in interp2d_calls:
        entries.append(("Interpolation2D", method, "unmasked", unmasked_fn))
        entries.append(("Interpolation2D", method, "masked", masked_fn))

    interp3d_calls: list[tuple[str, Callable[[], object], Callable[[], object]]] = [
        (
            "T_to_U",
            lambda: interp3d.T_to_U(h3d),
            lambda: interp3d.T_to_U(h3d, mask=mask2d),
        ),
        (
            "T_to_V",
            lambda: interp3d.T_to_V(h3d),
            lambda: interp3d.T_to_V(h3d, mask=mask2d),
        ),
        (
            "U_to_T",
            lambda: interp3d.U_to_T(u3d),
            lambda: interp3d.U_to_T(u3d, mask=mask2d),
        ),
        (
            "V_to_T",
            lambda: interp3d.V_to_T(v3d),
            lambda: interp3d.V_to_T(v3d, mask=mask2d),
        ),
    ]
    for method, unmasked_fn, masked_fn in interp3d_calls:
        entries.append(("Interpolation3D", method, "unmasked", unmasked_fn))
        entries.append(("Interpolation3D", method, "masked", masked_fn))

    # ------------------------------------------------------------------
    # Vorticity2D / Vorticity3D
    # ------------------------------------------------------------------
    from finitevolx._src.operators.vorticity import Vorticity2D, Vorticity3D

    vort2d = Vorticity2D(grid=grid2d)
    vort3d = Vorticity3D(grid=grid3d)
    f2d = make_h_field_2d() * 0.0 + 1.0  # placeholder constant Coriolis
    # Build a usable PV-like q2d from relative_vorticity for the flux tests.
    q_vort = vort2d.relative_vorticity(u2d, v2d)

    vort2d_calls: list[tuple[str, Callable[[], object], Callable[[], object]]] = [
        (
            "relative_vorticity",
            lambda: vort2d.relative_vorticity(u2d, v2d),
            lambda: vort2d.relative_vorticity(u2d, v2d, mask=mask2d),
        ),
        (
            "potential_vorticity",
            lambda: vort2d.potential_vorticity(u2d, v2d, h2d, f2d),
            lambda: vort2d.potential_vorticity(u2d, v2d, h2d, f2d, mask=mask2d),
        ),
        (
            "pv_flux_energy_conserving",
            lambda: vort2d.pv_flux_energy_conserving(q_vort, u2d, v2d),
            lambda: vort2d.pv_flux_energy_conserving(q_vort, u2d, v2d, mask=mask2d),
        ),
        (
            "pv_flux_enstrophy_conserving",
            lambda: vort2d.pv_flux_enstrophy_conserving(q_vort, u2d, v2d),
            lambda: vort2d.pv_flux_enstrophy_conserving(q_vort, u2d, v2d, mask=mask2d),
        ),
        (
            "pv_flux_arakawa_lamb",
            lambda: vort2d.pv_flux_arakawa_lamb(q_vort, u2d, v2d),
            lambda: vort2d.pv_flux_arakawa_lamb(q_vort, u2d, v2d, mask=mask2d),
        ),
    ]
    for method, unmasked_fn, masked_fn in vort2d_calls:
        entries.append(("Vorticity2D", method, "unmasked", unmasked_fn))
        entries.append(("Vorticity2D", method, "masked", masked_fn))

    entries += [
        (
            "Vorticity3D",
            "relative_vorticity",
            "unmasked",
            lambda: vort3d.relative_vorticity(u3d, v3d),
        ),
        (
            "Vorticity3D",
            "relative_vorticity",
            "masked",
            lambda: vort3d.relative_vorticity(u3d, v3d, mask=mask2d),
        ),
    ]

    # ------------------------------------------------------------------
    # Spherical operators (SphericalDifference{2D,3D},
    # SphericalDivergence{2D,3D}, SphericalLaplacian{2D,3D},
    # SphericalVorticity{2D,3D})
    # ------------------------------------------------------------------
    from finitevolx._src.grid.spherical_grid import (
        SphericalArakawaCGrid2D,
        SphericalArakawaCGrid3D,
    )
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
    from tests.fixtures.inputs import (
        NX_INTERIOR,
        NY_INTERIOR,
        NZ_INTERIOR,
    )

    sgrid2 = SphericalArakawaCGrid2D.from_interior(
        NX_INTERIOR,
        NY_INTERIOR,
        lon_range=(0.0, 10.0),
        lat_range=(10.0, 20.0),
    )
    sgrid3 = SphericalArakawaCGrid3D.from_interior(
        NX_INTERIOR,
        NY_INTERIOR,
        NZ_INTERIOR,
        lon_range=(0.0, 10.0),
        lat_range=(10.0, 20.0),
        Lz=1.0,
    )
    sdiff2 = SphericalDifference2D(grid=sgrid2)
    sdiff3 = SphericalDifference3D(grid=sgrid3)
    sdiv2 = SphericalDivergence2D(grid=sgrid2)
    sdiv3 = SphericalDivergence3D(grid=sgrid3)
    slap2 = SphericalLaplacian2D(grid=sgrid2)
    slap3 = SphericalLaplacian3D(grid=sgrid3)
    svort2 = SphericalVorticity2D(grid=sgrid2)
    svort3 = SphericalVorticity3D(grid=sgrid3)

    sdiff2_calls: list[tuple[str, Callable[[], object], Callable[[], object]]] = [
        (
            "diff_lon_T_to_U",
            lambda: sdiff2.diff_lon_T_to_U(h2d),
            lambda: sdiff2.diff_lon_T_to_U(h2d, mask=mask2d),
        ),
        (
            "diff_lat_T_to_V",
            lambda: sdiff2.diff_lat_T_to_V(h2d),
            lambda: sdiff2.diff_lat_T_to_V(h2d, mask=mask2d),
        ),
        (
            "diff_lon_V_to_X",
            lambda: sdiff2.diff_lon_V_to_X(v2d),
            lambda: sdiff2.diff_lon_V_to_X(v2d, mask=mask2d),
        ),
        (
            "diff_lat_U_to_X",
            lambda: sdiff2.diff_lat_U_to_X(u2d),
            lambda: sdiff2.diff_lat_U_to_X(u2d, mask=mask2d),
        ),
        (
            "diff_lon_U_to_T",
            lambda: sdiff2.diff_lon_U_to_T(u2d),
            lambda: sdiff2.diff_lon_U_to_T(u2d, mask=mask2d),
        ),
        (
            "diff_lat_V_to_T",
            lambda: sdiff2.diff_lat_V_to_T(v2d),
            lambda: sdiff2.diff_lat_V_to_T(v2d, mask=mask2d),
        ),
        (
            "diff2_lon",
            lambda: sdiff2.diff2_lon(h2d),
            lambda: sdiff2.diff2_lon(h2d, mask=mask2d),
        ),
        (
            "laplacian_merid",
            lambda: sdiff2.laplacian_merid(h2d),
            lambda: sdiff2.laplacian_merid(h2d, mask=mask2d),
        ),
    ]
    for method, unmasked_fn, masked_fn in sdiff2_calls:
        entries.append(("SphericalDifference2D", method, "unmasked", unmasked_fn))
        entries.append(("SphericalDifference2D", method, "masked", masked_fn))

    sdiff3_calls: list[tuple[str, Callable[[], object], Callable[[], object]]] = [
        (
            "diff_lon_T_to_U",
            lambda: sdiff3.diff_lon_T_to_U(h3d),
            lambda: sdiff3.diff_lon_T_to_U(h3d, mask=mask2d),
        ),
        (
            "diff_lat_T_to_V",
            lambda: sdiff3.diff_lat_T_to_V(h3d),
            lambda: sdiff3.diff_lat_T_to_V(h3d, mask=mask2d),
        ),
        (
            "diff_lon_U_to_T",
            lambda: sdiff3.diff_lon_U_to_T(u3d),
            lambda: sdiff3.diff_lon_U_to_T(u3d, mask=mask2d),
        ),
        (
            "diff_lat_V_to_T",
            lambda: sdiff3.diff_lat_V_to_T(v3d),
            lambda: sdiff3.diff_lat_V_to_T(v3d, mask=mask2d),
        ),
        (
            "laplacian_merid",
            lambda: sdiff3.laplacian_merid(h3d),
            lambda: sdiff3.laplacian_merid(h3d, mask=mask2d),
        ),
    ]
    for method, unmasked_fn, masked_fn in sdiff3_calls:
        entries.append(("SphericalDifference3D", method, "unmasked", unmasked_fn))
        entries.append(("SphericalDifference3D", method, "masked", masked_fn))

    # ------------------------------------------------------------------
    # Diffusion2D / Diffusion3D / BiharmonicDiffusion2D / BiharmonicDiffusion3D
    # ------------------------------------------------------------------
    from finitevolx._src.diffusion.diffusion import (
        BiharmonicDiffusion2D,
        BiharmonicDiffusion3D,
        Diffusion2D,
        Diffusion3D,
    )

    diffop2d = Diffusion2D(grid=grid2d)
    diffop3d = Diffusion3D(grid=grid3d)
    biharm2d = BiharmonicDiffusion2D(grid=grid2d)
    biharm3d = BiharmonicDiffusion3D(grid=grid3d)
    kappa = 1e-3

    entries += [
        ("Diffusion2D", "__call__", "unmasked", lambda: diffop2d(h2d, kappa=kappa)),
        (
            "Diffusion2D",
            "__call__",
            "masked",
            lambda: diffop2d(h2d, kappa=kappa, mask=mask2d),
        ),
        (
            "Diffusion2D",
            "fluxes",
            "unmasked",
            lambda: diffop2d.fluxes(h2d, kappa=kappa),
        ),
        (
            "Diffusion2D",
            "fluxes",
            "masked",
            lambda: diffop2d.fluxes(h2d, kappa=kappa, mask=mask2d),
        ),
        ("Diffusion3D", "__call__", "unmasked", lambda: diffop3d(h3d, kappa=kappa)),
        (
            "Diffusion3D",
            "__call__",
            "masked",
            lambda: diffop3d(h3d, kappa=kappa, mask=mask2d),
        ),
        (
            "Diffusion3D",
            "fluxes",
            "unmasked",
            lambda: diffop3d.fluxes(h3d, kappa=kappa),
        ),
        (
            "Diffusion3D",
            "fluxes",
            "masked",
            lambda: diffop3d.fluxes(h3d, kappa=kappa, mask=mask2d),
        ),
        (
            "BiharmonicDiffusion2D",
            "__call__",
            "unmasked",
            lambda: biharm2d(h2d, kappa=kappa),
        ),
        (
            "BiharmonicDiffusion2D",
            "__call__",
            "masked",
            lambda: biharm2d(h2d, kappa=kappa, mask=mask2d),
        ),
        (
            "BiharmonicDiffusion3D",
            "__call__",
            "unmasked",
            lambda: biharm3d(h3d, kappa=kappa),
        ),
        (
            "BiharmonicDiffusion3D",
            "__call__",
            "masked",
            lambda: biharm3d(h3d, kappa=kappa, mask=mask2d),
        ),
    ]

    entries += [
        ("SphericalDivergence2D", "__call__", "unmasked", lambda: sdiv2(u2d, v2d)),
        (
            "SphericalDivergence2D",
            "__call__",
            "masked",
            lambda: sdiv2(u2d, v2d, mask=mask2d),
        ),
        ("SphericalDivergence3D", "__call__", "unmasked", lambda: sdiv3(u3d, v3d)),
        (
            "SphericalDivergence3D",
            "__call__",
            "masked",
            lambda: sdiv3(u3d, v3d, mask=mask2d),
        ),
        ("SphericalLaplacian2D", "__call__", "unmasked", lambda: slap2(h2d)),
        ("SphericalLaplacian2D", "__call__", "masked", lambda: slap2(h2d, mask=mask2d)),
        ("SphericalLaplacian3D", "__call__", "unmasked", lambda: slap3(h3d)),
        ("SphericalLaplacian3D", "__call__", "masked", lambda: slap3(h3d, mask=mask2d)),
        (
            "SphericalVorticity2D",
            "relative_vorticity",
            "unmasked",
            lambda: svort2.relative_vorticity(u2d, v2d),
        ),
        (
            "SphericalVorticity2D",
            "relative_vorticity",
            "masked",
            lambda: svort2.relative_vorticity(u2d, v2d, mask=mask2d),
        ),
        (
            "SphericalVorticity3D",
            "relative_vorticity",
            "unmasked",
            lambda: svort3.relative_vorticity(u3d, v3d),
        ),
        (
            "SphericalVorticity3D",
            "relative_vorticity",
            "masked",
            lambda: svort3.relative_vorticity(u3d, v3d, mask=mask2d),
        ),
    ]

    return entries


def main() -> int:
    """Materialize every registered entry to ``tests/fixtures/golden/``."""
    entries = _register_all()
    if not entries:
        print(
            "No goldens registered.  Edit tests/fixtures/_gen_golden.py "
            "and add entries to _register_all().",
            file=sys.stderr,
        )
        return 0

    print(f"Generating {len(entries)} golden files…")
    for operator, method, variant, fn in entries:
        value = fn()
        path = save_golden(operator, method, variant, value)
        rel = path.relative_to(_REPO_ROOT)
        print(f"  wrote {rel}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
