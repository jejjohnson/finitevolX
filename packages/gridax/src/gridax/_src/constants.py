"""Deprecated physical constants; geophysix owns them (#446).

The values are geophysix's: R_EARTH is the IUGG mean radius of WGS84
(``geophysix.earth.SPHERE``), GRAVITY standard gravity
(``geophysix.constants.STANDARD``) and OMEGA WGS84's rotation rate. gridax's
own defaults read the geophysix records directly; these names stay for one
release, behind a ``DeprecationWarning`` in ``gridax`` and ``gridax_jax``.
"""

import math

from geophysix.constants import STANDARD
from geophysix.earth import SPHERE, WGS84

R_EARTH = SPHERE.semi_major  # IUGG mean radius of WGS84 (m)
GRAVITY = STANDARD.standard_gravity  # standard gravity (m/s^2)
OMEGA = WGS84.rotation_rate  # WGS84 rotation rate (rad/s)
DEG2M = math.pi * R_EARTH / 180.0  # metres per degree on that sphere
RHO = 1.0e3  # density of water (kg/m^3)
