"""The dtype rule: float64 host data with x64 off raises; everything else passes.

``tests/conftest.py`` turns x64 on for the session, so the x64-off cases run
in a subprocess with the flag left at JAX's default.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from gridax.solvers import check_dtype
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(3),
        np.zeros(3, dtype=np.complex128),
        np.float64(1.0),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.int64),
        1.0,
        jnp.zeros(3),
    ],
    ids=["f64", "c128", "f64-scalar", "f32", "i64", "py-float", "jax"],
)
def test_passes_with_x64_on(value):
    check_dtype(value)


def test_x64_off():
    """float64 and complex128 raise; float32, ints, Python floats and JAX arrays pass."""
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np

        from gridax.solvers import check_dtype

        assert not jax.config.read("jax_enable_x64")
        for bad, narrow in (
            (np.zeros(3), "float32"),
            (np.float64(2.0), "float32"),
            (np.zeros(3, dtype=np.complex128), "complex64"),
        ):
            try:
                check_dtype(bad, name="sst")
            except TypeError as err:
                assert "sst is" in str(err) and narrow in str(err), err
            else:
                raise AssertionError(f"{bad.dtype} passed with x64 off")

        for good in (np.zeros(3, dtype=np.float32), np.arange(3), 1.0, jnp.zeros(3)):
            check_dtype(good)

        jax.config.update("jax_enable_x64", True)
        check_dtype(np.zeros(3))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
