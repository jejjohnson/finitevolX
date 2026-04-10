"""Spherical Arakawa C-grid masks (TODO).

Placeholder module for masks on a spherical-coordinate grid.

TODO: Implement ``SphericalMask2D`` and ``SphericalMask3D`` as concrete
mask classes for ``SphericalGrid2D`` / ``SphericalGrid3D``.  Most of
the boolean topology logic from the Cartesian masks should carry over
unchanged — the only spherical-specific behaviour expected is:

* Optional longitudinal wraparound for global domains (no west/east
  ghost ring; instead `enforce_periodic` is applied to mask construction).
* Pole handling: cells at the poles need their classification overridden
  because the metric area shrinks to zero (boundary-by-construction).
* Sponge layer construction may want to use lat/lon coordinates rather
  than cell counts.
"""
