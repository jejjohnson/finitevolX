"""Equation of state operators for ocean density from temperature and salinity.

Provides functional operators for computing density, density anomaly,
partial derivatives, buoyancy, and reduced gravity from tracer fields.

Phase 1 implements the **linear equation of state**::

    ρ(T, S) = ρ₀ · (1 − α·(T − T_ref) + β·(S − S_ref))

where α is the thermal expansion coefficient and β is the haline
contraction coefficient.
"""

from __future__ import annotations

from jaxtyping import Array, Float

from finitevolx._src.utils.constants import GRAVITY


def linear_density(
    T: Float[Array, "... Ny Nx"],
    S: Float[Array, "... Ny Nx"],
    rho_0: float = 1025.0,
    alpha: float = 2e-4,
    beta: float = 7e-4,
    T_ref: float = 10.0,
    S_ref: float = 35.0,
) -> Float[Array, "... Ny Nx"]:
    r"""Compute density from temperature and salinity using a linear EOS.

    .. math::
        \rho = \rho_0 \bigl(1 - \alpha\,(T - T_{\mathrm{ref}}) + \beta\,(S - S_{\mathrm{ref}})\bigr)

    Parameters
    ----------
    T : Float[Array, "... Ny Nx"]
        Temperature (°C).
    S : Float[Array, "... Ny Nx"]
        Salinity (PSU).
    rho_0 : float
        Reference density (kg/m³).
    alpha : float
        Thermal expansion coefficient (1/K).
    beta : float
        Haline contraction coefficient (1/PSU).
    T_ref : float
        Reference temperature (°C).
    S_ref : float
        Reference salinity (PSU).

    Returns
    -------
    Float[Array, "... Ny Nx"]
        Density ρ (kg/m³).
    """
    return rho_0 * (1.0 - alpha * (T - T_ref) + beta * (S - S_ref))


def linear_density_anomaly(
    T: Float[Array, "... Ny Nx"],
    S: Float[Array, "... Ny Nx"],
    rho_0: float = 1025.0,
    alpha: float = 2e-4,
    beta: float = 7e-4,
    T_ref: float = 10.0,
    S_ref: float = 35.0,
) -> Float[Array, "... Ny Nx"]:
    r"""Compute density anomaly ρ' = ρ − ρ₀ using a linear EOS.

    .. math::
        \rho' = \rho_0 \bigl(-\alpha\,(T - T_{\mathrm{ref}}) + \beta\,(S - S_{\mathrm{ref}})\bigr)

    Parameters
    ----------
    T : Float[Array, "... Ny Nx"]
        Temperature (°C).
    S : Float[Array, "... Ny Nx"]
        Salinity (PSU).
    rho_0 : float
        Reference density (kg/m³).
    alpha : float
        Thermal expansion coefficient (1/K).
    beta : float
        Haline contraction coefficient (1/PSU).
    T_ref : float
        Reference temperature (°C).
    S_ref : float
        Reference salinity (PSU).

    Returns
    -------
    Float[Array, "... Ny Nx"]
        Density anomaly ρ' = ρ − ρ₀ (kg/m³).
    """
    return rho_0 * (-alpha * (T - T_ref) + beta * (S - S_ref))


def linear_drho_dT(
    rho_0: float = 1025.0,
    alpha: float = 2e-4,
) -> float:
    r"""Partial derivative ∂ρ/∂T for the linear EOS (constant).

    .. math::
        \frac{\partial\rho}{\partial T} = -\rho_0 \, \alpha

    Parameters
    ----------
    rho_0 : float
        Reference density (kg/m³).
    alpha : float
        Thermal expansion coefficient (1/K).

    Returns
    -------
    float
        ∂ρ/∂T (kg/(m³·K)).  Negative: warmer water is lighter.
    """
    return -rho_0 * alpha


def linear_drho_dS(
    rho_0: float = 1025.0,
    beta: float = 7e-4,
) -> float:
    r"""Partial derivative ∂ρ/∂S for the linear EOS (constant).

    .. math::
        \frac{\partial\rho}{\partial S} = \rho_0 \, \beta

    Parameters
    ----------
    rho_0 : float
        Reference density (kg/m³).
    beta : float
        Haline contraction coefficient (1/PSU).

    Returns
    -------
    float
        ∂ρ/∂S (kg/(m³·PSU)).  Positive: saltier water is heavier.
    """
    return rho_0 * beta


def buoyancy(
    rho: Float[Array, "... Ny Nx"],
    rho_0: float = 1025.0,
    g: float = GRAVITY,
) -> Float[Array, "... Ny Nx"]:
    r"""Compute buoyancy from density.

    .. math::
        b = -\frac{g}{\rho_0}\,(\rho - \rho_0)

    Parameters
    ----------
    rho : Float[Array, "... Ny Nx"]
        Density (kg/m³).
    rho_0 : float
        Reference density (kg/m³).
    g : float
        Gravitational acceleration (m/s²).

    Returns
    -------
    Float[Array, "... Ny Nx"]
        Buoyancy (m/s²).  Positive when ρ < ρ₀ (lighter than reference).
    """
    return -g * (rho - rho_0) / rho_0


def reduced_gravity(
    rho_upper: Float[Array, "... Ny Nx"],
    rho_lower: Float[Array, "... Ny Nx"],
    rho_0: float = 1025.0,
    g: float = GRAVITY,
) -> Float[Array, "... Ny Nx"]:
    r"""Compute reduced gravity between two layers.

    .. math::
        g' = g \, \frac{\rho_{\text{lower}} - \rho_{\text{upper}}}{\rho_0}

    Computes the interface reduced gravity between two adjacent layers.
    To build the ``g_prime`` vector expected by
    :func:`~finitevolx.build_coupling_matrix`, compute this for each
    interface and stack into a 1-D array of shape ``(nl,)``.

    Parameters
    ----------
    rho_upper : Float[Array, "... Ny Nx"]
        Density of the upper layer (kg/m³).
    rho_lower : Float[Array, "... Ny Nx"]
        Density of the lower layer (kg/m³).
    rho_0 : float
        Reference density (kg/m³).
    g : float
        Gravitational acceleration (m/s²).

    Returns
    -------
    Float[Array, "... Ny Nx"]
        Reduced gravity g' (m/s²).  Positive when the lower layer is
        denser (stable stratification).
    """
    return g * (rho_lower - rho_upper) / rho_0
