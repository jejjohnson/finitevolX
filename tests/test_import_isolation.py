"""`import gridax` stays light; `gridax.solvers` and `gridax.pipekit` are opt-in.

The solvers need the `jax` extra and the pipekit operators the `pipekit`
extra, so `import gridax` must not import either subpackage or what they
build on (#477). JAX itself is the one exception in this environment: when it
is installed, gridax registers its grids and masks as pytrees, which imports
`jax` (and never Equinox); `packages/gridax/tests/test_numpy_only.py` checks
the install without JAX.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_gridax_imports_neither_extra():
    res = _run(
        """
        import sys

        import gridax

        heavy = {
            "gridax.solvers", "gridax.pipekit", "gridax._src.solvers",
            "gridax._src.pipekit", "equinox", "lineax", "gaussx",
            "spectraldiffx", "integrix", "pipekit_ops", "pipekit_jax",
        }
        loaded = sorted(m for m in sys.modules if m in heavy or m.split(".")[0] in heavy)
        assert not loaded, loaded
        """
    )
    assert res.returncode == 0, res.stderr


def test_import_gridax_without_jax_imports_no_jax():
    # Block JAX and Equinox at import: gridax imports, and gridax.solvers
    # names the extra it needs.
    res = _run(
        """
        import importlib.abc, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in {"jax", "jaxlib", "equinox"}:
                    raise ImportError(f"blocked: {name}")

        sys.meta_path.insert(0, Block())

        import gridax

        assert not any(m.split(".")[0] in {"jax", "equinox"} for m in sys.modules)
        try:
            import gridax.solvers
        except ImportError as err:
            assert 'gridax[jax]' in str(err), err
        else:
            raise AssertionError("gridax.solvers imported without JAX")
        """
    )
    assert res.returncode == 0, res.stderr


def test_gridax_solvers_names_resolve():
    import gridax.solvers as gs

    assert gs.__all__ == sorted(gs.__all__)
    missing = [name for name in gs.__all__ if not hasattr(gs, name)]
    assert not missing, missing


def test_gridax_pipekit_names_resolve():
    import gridax.pipekit as gp

    missing = [name for name in gp.__all__ if not hasattr(gp, name)]
    assert not missing, missing
