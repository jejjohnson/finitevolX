"""Cylindrical Arakawa C-grid masks (TODO).

Placeholder module for masks on a cylindrical-coordinate grid.

TODO: Implement ``CylindricalMask2D`` and ``CylindricalMask3D`` as
concrete mask classes for ``CylindricalGrid2D`` / ``CylindricalGrid3D``.
The construction logic from the Cartesian masks should carry over
unchanged — the only cylindrical-specific behaviour expected is:

* Optional azimuthal wraparound (theta-direction is periodic by default).
* Special handling at the cylinder axis (r = 0): the mask classification
  needs to treat the axis as a soft boundary, not a wall.
"""
