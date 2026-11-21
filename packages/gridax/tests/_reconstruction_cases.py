"""Every reconstruction call of gridax, on fixed random data (#464).

``freeze()`` records what each call returns; ``test_reconstruct.py`` checks the
axis-generic rewrite against the record, bit for bit. Rerun this module as a
script only to re-freeze on purpose:

    uv run python packages/gridax/tests/_reconstruction_cases.py
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import gridax
import numpy as np

FROZEN = Path(__file__).parent / "data" / "reconstruction_frozen.npz"
LIMITERS = ("minmod", "van_leer", "superbee", "mc")


def _fields(shape: tuple[int, ...], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape), rng.standard_normal(shape)


def _mask_2d(ny: int, nx: int) -> gridax.Mask2D:
    wet = np.ones((ny, nx), dtype=bool)
    wet[0, :] = wet[-1, :] = wet[:, 0] = wet[:, -1] = False
    wet[5:8, 6:10] = False  # an island
    wet[1:3, nx - 5 : nx - 2] = False  # a coastal notch
    return gridax.Mask2D.from_mask(wet)


def cases() -> Iterator[tuple[str, Callable[[], np.ndarray]]]:
    """``(name, call)`` for every method of Reconstruction1D/2D/3D and the old reconstruct."""
    setups = {
        1: (
            gridax.Reconstruction1D(grid=gridax.CartesianGrid1D.from_interior(18, 1.0)),
            (20,),
        ),
        2: (
            gridax.Reconstruction2D(
                grid=gridax.CartesianGrid2D.from_interior(16, 12, 1.0, 1.0)
            ),
            (14, 18),
        ),
        3: (
            gridax.Reconstruction3D(
                grid=gridax.CartesianGrid3D.from_interior(12, 10, 4, 1.0, 1.0, 1.0)
            ),
            (6, 12, 14),
        ),
    }
    masks = {2: _mask_2d(14, 18), 3: _mask_2d(12, 14)}
    for ndim, (recon, shape) in setups.items():
        h, u = _fields(shape, ndim)
        mask = masks.get(ndim)
        for name in sorted(
            n for n in dir(recon) if n[:1] != "_" and callable(getattr(recon, n))
        ):
            method = getattr(recon, name)
            if name.startswith("tvd"):
                for limiter in LIMITERS:
                    if name.endswith("_masked"):
                        yield (
                            f"{ndim}d.{name}.{limiter}",
                            lambda m=method, lim=limiter, h=h, u=u, k=mask: m(
                                h, u, k, limiter=lim
                            ),
                        )
                    else:
                        yield (
                            f"{ndim}d.{name}.{limiter}",
                            lambda m=method, lim=limiter, h=h, u=u: m(
                                h, u, limiter=lim
                            ),
                        )
            elif name.endswith("_masked"):
                yield f"{ndim}d.{name}", lambda m=method, h=h, u=u, k=mask: m(h, u, k)
            else:
                yield f"{ndim}d.{name}", lambda m=method, h=h, u=u: m(h, u)
    h, u = _fields((14, 18), 9)
    for dim in (0, 1):
        for method in ("weno", "wenoz", "linear"):
            for num_pts in (1, 3, 5):
                yield (
                    f"reconstruct.dim{dim}.{method}.{num_pts}",
                    lambda d=dim, m=method, n=num_pts, h=h, u=u: gridax.reconstruct(
                        h, u, d, m, n
                    ),
                )


def freeze() -> None:
    np.savez(FROZEN, **{name: np.asarray(call()) for name, call in cases()})


if __name__ == "__main__":
    freeze()
    print(f"froze {len(np.load(FROZEN).files)} calls into {FROZEN}")
