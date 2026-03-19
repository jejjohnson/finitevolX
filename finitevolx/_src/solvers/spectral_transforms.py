"""Spectral transforms — re-exported from spectraldiffx.

All transforms follow the unnormalized scipy convention (norm=None).
See spectraldiffx documentation for details on each transform type.
"""

from spectraldiffx import (
    dct,
    dctn,
    dst,
    dstn,
    idct,
    idctn,
    idst,
    idstn,
)

__all__ = ["dct", "dctn", "dst", "dstn", "idct", "idctn", "idst", "idstn"]
