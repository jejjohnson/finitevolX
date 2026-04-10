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
    # Operator families are wired up in subsequent commits.  This first
    # commit only ships the infrastructure; entries are added alongside
    # the operator changes themselves so each commit is self-contained.
    # ------------------------------------------------------------------

    # Touch the bindings to keep linters from complaining about unused
    # locals while the registration tables are still being written
    # commit-by-commit.  These references are deliberately cheap.
    _ = (grid2d, grid3d, mask2d, h2d, u2d, v2d, q2d, h3d, u3d, v3d)

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
