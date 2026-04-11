"""Test helpers for loading and asserting against golden ``.npz`` files.

These helpers are imported by individual ``test_<operator>_masks.py``
files in the operator-attribute mask regression suite.  See
:mod:`tests.fixtures.inputs` for the canonical input fields and
:mod:`tests.fixtures._gen_golden` for the script that produces the
goldens.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_GOLDEN_DIR = Path(__file__).parent / "golden"


def golden_path(operator: str, method: str, variant: str) -> Path:
    """Return the on-disk path for a given golden file.

    Args:
        operator: Operator class or function name (e.g. ``"Difference2D"``).
        method: Method name (or ``"__call__"`` for a function).
        variant: One of ``"unmasked"``, ``"masked"``, ``"all_ocean"``.
    """
    return _GOLDEN_DIR / f"{operator}__{method}__{variant}.npz"


def load_golden(
    operator: str, method: str, variant: str = "masked"
) -> dict[str, np.ndarray]:
    """Load a golden ``.npz`` file as a plain dict of numpy arrays.

    Multi-output operators (e.g. ``Coriolis2D`` returning ``(du, dv)``)
    use keys ``"out0"``, ``"out1"``, ``...``.  Single-output operators
    use the key ``"out"``.
    """
    path = golden_path(operator, method, variant)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing golden file {path}.  Re-run "
            f"`uv run python tests/fixtures/_gen_golden.py` to (re)generate."
        )
    with np.load(path) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def assert_matches_golden(
    actual,
    operator: str,
    method: str,
    variant: str = "masked",
    *,
    rtol: float = 1e-12,
    atol: float = 0.0,
) -> None:
    """Assert that an operator output matches its committed golden file.

    Single-output (``actual`` is an array) and multi-output (``actual``
    is a tuple of arrays) operators are both supported.  The default
    tolerances are exact: golden tests should be deterministic up to
    floating-point reproducibility on a single machine.  Pass ``rtol``
    or ``atol`` only if you have a documented reason for the loosening.
    """
    golden = load_golden(operator, method, variant)
    if isinstance(actual, tuple):
        for i, a in enumerate(actual):
            key = f"out{i}"
            assert key in golden, (
                f"golden for {operator}.{method} ({variant}) missing key {key}; "
                f"available keys: {list(golden)}"
            )
            np.testing.assert_allclose(
                np.asarray(a),
                golden[key],
                rtol=rtol,
                atol=atol,
                err_msg=f"{operator}.{method} ({variant}) output {i} mismatch",
            )
    else:
        key = "out"
        assert key in golden, (
            f"golden for {operator}.{method} ({variant}) missing key {key}; "
            f"available keys: {list(golden)}"
        )
        np.testing.assert_allclose(
            np.asarray(actual),
            golden[key],
            rtol=rtol,
            atol=atol,
            err_msg=f"{operator}.{method} ({variant}) output mismatch",
        )


def save_golden(operator: str, method: str, variant: str, value) -> Path:
    """Persist an operator output to its golden ``.npz`` file.

    Used by ``_gen_golden.py``; not called from tests.
    """
    path = golden_path(operator, method, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, tuple):
        kwargs = {f"out{i}": np.asarray(v) for i, v in enumerate(value)}
    else:
        kwargs = {"out": np.asarray(value)}
    np.savez(path, **kwargs)
    return path
