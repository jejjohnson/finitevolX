"""The capability index (docs/capabilities.md) is current.

Agents and contributors search the index before writing a helper ("Reuse
before you write" in AGENTS.md), so a stale index sends them to
re-implement something that exists. ``scripts/capabilities.py`` regenerates
it; this runs its ``--check``. The upstream sections (spectraldiffx, gaussx)
are compared only at the versions the index records.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "capabilities.py"
if not SCRIPT.is_file() or not (ROOT / "docs" / "capabilities.md").is_file():
    # A built distribution ships tests/ without scripts/ or the docs.
    pytest.skip("scripts/capabilities.py is not present", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("capabilities", SCRIPT)
assert _spec is not None and _spec.loader is not None
capabilities = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capabilities)


def test_capability_index_is_current() -> None:
    current = capabilities.INDEX.read_text(encoding="utf-8")
    assert not capabilities.stale(current, capabilities.render()), (
        "docs/capabilities.md is stale: run `make capabilities` and commit it."
    )


def test_no_name_is_bound_to_two_objects() -> None:
    _, clashes = capabilities.collect()
    assert not clashes, (
        "finitevolx exports a name that spectraldiffx or gaussx exports for a "
        f"different object: {clashes}. Rename it, or add it to "
        "ALLOWED_SHARED_NAMES in scripts/capabilities.py with a reason."
    )
