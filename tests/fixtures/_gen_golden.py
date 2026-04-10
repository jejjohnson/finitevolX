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
