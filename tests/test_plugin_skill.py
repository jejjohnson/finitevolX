"""Keep the Claude Code plugin's guidance runnable and current.

``plugins/finitevolx/skills/finite-volume-models-with-finitevolx/SKILL.md``
is what agents in downstream projects read before writing finite-volume
code on finitevolX, so a stale name or a broken example there teaches them
the wrong API. These tests run its worked example, check the claims its
prose makes about the result, and check that every ``fvx.X`` /
``finitevolx.X`` it and the plugin's reviewer name is a current public name.

The repository has no test-tier markers, so the worked example runs with
the rest of the suite (a few seconds on a 64 × 64 grid).
"""

from __future__ import annotations

import importlib
from pathlib import Path
import re

import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "finitevolx"
SKILL = PLUGIN / "skills" / "finite-volume-models-with-finitevolx" / "SKILL.md"
AGENT = PLUGIN / "agents" / "finitevolx-reuse-reviewer.md"
if not SKILL.is_file():
    # A built distribution ships tests/ without the repository root.
    pytest.skip("the plugin is not present", allow_module_level=True)

_ALIASES = {"fvx": "finitevolx"}
# ``fvx.Name`` / ``finitevolx.Name``, but not inside a URL or a longer path.
_NAME = re.compile(r"(?<![\w./])(fvx|finitevolx)\.([A-Za-z_]\w*)")
_BLOCKS = re.compile(r"```python\n(.*?)```", re.S)


def test_worked_example_runs_and_its_claims_hold():
    (block,) = [b for b in _BLOCKS.findall(SKILL.read_text()) if "grad_perp" in b]
    ns: dict = {}
    exec(block, ns)
    # grad_perp is discretely non-divergent: the divergence is round-off,
    # |u| / dx ≈ 2π · 64 ≈ 400 times float64 epsilon times a few terms
    # (≈ 1e-13; the prose says "of order 1e-13"); 1e-10 still catches a
    # stagger error, which gives O(1) divergence.
    assert float(jnp.max(jnp.abs(ns["div"]))) < 1e-10
    # The FFT solver's eigenvalues are those of the five-point Laplacian,
    # so the inversion is exact up to round-off (the prose says ~1e-15).
    assert float(ns["inv_err"]) < 1e-12
    # Periodic flux form with filled ghosts: one flux per face, so the
    # area-weighted total is conserved to round-off (the prose says ~1e-14).
    assert abs(float(ns["mass_drift"])) < 1e-12
    # A masked operator is exactly zero on land, not merely small.
    assert float(ns["land_max"]) == 0.0


@pytest.mark.parametrize("path", [SKILL, AGENT], ids=lambda p: p.name)
def test_named_api_is_current(path: Path):
    stale = []
    for prefix, attr in set(_NAME.findall(path.read_text())):
        if attr.startswith("__"):  # ``finitevolx.__all__`` and friends
            continue
        module = importlib.import_module(_ALIASES.get(prefix, prefix))
        if attr not in module.__all__:
            stale.append(f"{prefix}.{attr}")
    assert not stale, f"{path.name} names objects that do not exist: {sorted(stale)}"
