"""Generate golden ``.npz`` files for the operator-attribute mask regression suite.

Run this script (NOT pytest) to (re)generate every golden file under
``tests/fixtures/golden/``::

    uv run python tests/fixtures/_gen_golden.py

The generator is intentionally split per operator family.  Adding a new
operator (or a new method on an existing operator) means:

1. Add the corresponding entry in :func:`_register_all`.
2. Re-run this script.
3. ``git add tests/fixtures/golden/<operator>__<method>__*.npz``
4. Update the matching ``test_<operator>_masks.py`` to call
   ``assert_matches_golden(...)``.

Each registered entry is a tuple describing one operator method on one
input + one variant ("unmasked", "masked", or "all_ocean") and the
zero-arg callable that produces the output.  The registration list is
the single source of truth for "what each operator is expected to
compute on the canonical input."

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

# Enable float64 *before* any finitevolx/JAX import.  The goldens must
# be saved in the same dtype regime that the test suite runs under; the
# test-side sibling ``tests/fixtures/__init__.py`` does the same thing.
import jax

jax.config.update("jax_enable_x64", True)

from tests.fixtures._helpers import save_golden
from tests.fixtures.inputs import (
    make_grid_1d,
    make_grid_2d,
    make_grid_3d,
    make_h_field_1d,
    make_h_field_2d,
    make_h_field_3d,
    make_mask_1d,
    make_mask_2d,
    make_mask_3d,
    make_q_field_2d,
    make_u_field_1d,
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
    grid1d = make_grid_1d()
    grid2d = make_grid_2d()
    grid3d = make_grid_3d()
    mask1d = make_mask_1d()
    mask2d = make_mask_2d()
    mask3d = make_mask_3d()
    h1d = make_h_field_1d()
    u1d = make_u_field_1d()
    h2d = make_h_field_2d()
    u2d = make_u_field_2d()
    v2d = make_v_field_2d()
    q2d = make_q_field_2d()
    h3d = make_h_field_3d()
    u3d = make_u_field_3d()
    v3d = make_v_field_3d()

    # Touch q2d so the linter doesn't complain about unused locals until
    # later commits wire in operators that consume X-point fields.
    _ = q2d

    # ------------------------------------------------------------------
    # Difference1D / 2D / 3D
    # ------------------------------------------------------------------
    entries.extend(
        _difference_entries(
            grid1d,
            grid2d,
            grid3d,
            mask1d,
            mask2d,
            mask3d,
            h1d,
            u1d,
            h2d,
            u2d,
            v2d,
            h3d,
            u3d,
            v3d,
        )
    )

    return entries


def _difference_entries(
    grid1d,
    grid2d,
    grid3d,
    mask1d,
    mask2d,
    mask3d,
    h1d,
    u1d,
    h2d,
    u2d,
    v2d,
    h3d,
    u3d,
    v3d,
) -> list[Entry]:
    """Register goldens for Difference1D / Difference2D / Difference3D."""
    from finitevolx._src.operators.difference import (
        Difference1D,
        Difference2D,
        Difference3D,
    )

    entries: list[Entry] = []

    # --- Difference1D -------------------------------------------------
    d1 = Difference1D(grid=grid1d)
    d1m = Difference1D(grid=grid1d, mask=mask1d)
    for method, arg in (
        ("diff_x_T_to_U", h1d),
        ("diff_x_U_to_T", u1d),
        ("laplacian", h1d),
    ):
        entries.append(
            (
                "Difference1D",
                method,
                "unmasked",
                (lambda m=method, a=arg: getattr(d1, m)(a)),
            )
        )
        entries.append(
            (
                "Difference1D",
                method,
                "masked",
                (lambda m=method, a=arg: getattr(d1m, m)(a)),
            )
        )

    # --- Difference2D -------------------------------------------------
    d2 = Difference2D(grid=grid2d)
    d2m = Difference2D(grid=grid2d, mask=mask2d)
    # (method_name, *args_to_pass)
    d2_unary_specs = (
        ("diff_x_T_to_U", h2d),
        ("diff_y_T_to_V", h2d),
        ("diff_y_U_to_X", u2d),
        ("diff_x_V_to_X", v2d),
        ("diff_y_X_to_U", h2d),  # pretend h2d lives at X — smooth analytic field
        ("diff_x_X_to_V", h2d),
        ("diff_x_U_to_T", u2d),
        ("diff_y_V_to_T", v2d),
        ("laplacian", h2d),
    )
    for method, arg in d2_unary_specs:
        entries.append(
            (
                "Difference2D",
                method,
                "unmasked",
                (lambda m=method, a=arg: getattr(d2, m)(a)),
            )
        )
        entries.append(
            (
                "Difference2D",
                method,
                "masked",
                (lambda m=method, a=arg: getattr(d2m, m)(a)),
            )
        )
    # Compound: divergence, curl, grad_perp (grad_perp takes psi only).
    entries.append(
        ("Difference2D", "divergence", "unmasked", lambda: d2.divergence(u2d, v2d))
    )
    entries.append(
        ("Difference2D", "divergence", "masked", lambda: d2m.divergence(u2d, v2d))
    )
    entries.append(("Difference2D", "curl", "unmasked", lambda: d2.curl(u2d, v2d)))
    entries.append(("Difference2D", "curl", "masked", lambda: d2m.curl(u2d, v2d)))
    entries.append(("Difference2D", "grad_perp", "unmasked", lambda: d2.grad_perp(h2d)))
    entries.append(("Difference2D", "grad_perp", "masked", lambda: d2m.grad_perp(h2d)))

    # --- Difference3D -------------------------------------------------
    d3 = Difference3D(grid=grid3d)
    d3m = Difference3D(grid=grid3d, mask=mask3d)
    d3_unary_specs = (
        ("diff_x_T_to_U", h3d),
        ("diff_y_T_to_V", h3d),
        ("diff_x_U_to_T", u3d),
        ("diff_y_V_to_T", v3d),
        ("laplacian", h3d),
    )
    for method, arg in d3_unary_specs:
        entries.append(
            (
                "Difference3D",
                method,
                "unmasked",
                (lambda m=method, a=arg: getattr(d3, m)(a)),
            )
        )
        entries.append(
            (
                "Difference3D",
                method,
                "masked",
                (lambda m=method, a=arg: getattr(d3m, m)(a)),
            )
        )
    entries.append(
        ("Difference3D", "divergence", "unmasked", lambda: d3.divergence(u3d, v3d))
    )
    entries.append(
        ("Difference3D", "divergence", "masked", lambda: d3m.divergence(u3d, v3d))
    )

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
