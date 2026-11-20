"""gridax builds every grid and mask with JAX and Equinox absent.

CI runs this file in an environment with gridax and pytest only, where JAX
is not installed. In the development environment JAX is installed, so the
same checks run in a subprocess that blocks ``jax``, ``jaxlib`` and
``equinox`` at import.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import subprocess
import sys
import textwrap

CHECKS = textwrap.dedent(
    """
    import numpy as np

    import gridax

    gridax.CartesianGrid1D.from_interior(4, 1.0)
    gridax.CartesianGrid3D.from_interior(4, 4, 4, 1.0, 1.0, 1.0).horizontal_grid()
    s = gridax.SphericalGrid3D.from_interior(8, 6, 3, (0, 40), (-30, 30), 1.0)
    float(s.min_cell_width), float(s.max_aspect), s.horizontal_grid().dx_V

    h = np.ones((8, 8), dtype=bool)
    h[3:5, 3:5] = False
    m2 = gridax.Mask2D.from_mask(h)
    m2.get_adaptive_masks(direction="y"), m2.ind_boundary, m2.not_xy_corner
    gridax.Mask2D.from_bathymetry(np.where(h, 50.0, -1.0), min_depth=1.0)
    gridax.Mask1D.from_dimensions(8).get_adaptive_masks()
    gridax.Mask3D.from_mask(np.broadcast_to(h, (4, 8, 8))).get_adaptive_masks("z")

    h2 = np.arange(30.0).reshape(5, 6)
    gridax.interior(gridax.diff_x_fwd_2d(h2) + gridax.avg_y_bwd_2d(h2), h2)
    gridax.smooth_clamp(h2, 3.0)

    g2 = gridax.CartesianGrid2D.from_interior(4, 3, 1.0, 1.0)
    gridax.Difference2D(grid=g2, mask=gridax.Mask2D.from_mask(np.ones((5, 6), bool))).laplacian(h2)
    g3 = gridax.CartesianGrid3D.from_interior(4, 3, 2, 1.0, 1.0, 1.0)
    gridax.Vorticity3D(grid=g3).relative_vorticity(np.stack([h2] * 4), np.stack([h2] * 4))

    from gridax import analysis

    y, x = np.mgrid[0:6, 0:8].astype(float)
    analysis.adaptive_laplacian_2d(x**2 + y**2, mask=h[:6])
    analysis.remap_axis(x, axis=-1, source_coords=np.arange(8.0), target_coords=np.array([1.5]))

    import sys

    loaded = sorted(m for m in sys.modules if m.split(".")[0] in {"jax", "equinox"})
    assert not loaded, loaded
    """
)

BLOCKER = textwrap.dedent(
    """
    import importlib.abc, sys

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in {"jax", "jaxlib", "equinox"}:
                raise ImportError(f"blocked: {name}")

    sys.meta_path.insert(0, Block())
    """
)


def test_builds_without_jax():
    if importlib.util.find_spec("jax") is None:
        exec(CHECKS, {})  # the CI job: JAX is not installed at all
        return
    result = subprocess.run(
        [sys.executable, "-c", BLOCKER + CHECKS],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_declares_no_jax_dependency():
    requires = importlib.metadata.requires("gridax") or []
    names = {re.split(r"[\s\[<>=!~;@(]", r, maxsplit=1)[0].lower() for r in requires}
    assert not names & {"jax", "jaxlib", "equinox"}, names
