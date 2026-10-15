from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np

#: Dimensionless epsilon of the WENO nonlinear weights (#243).  The
#: smoothness indicators ``beta_k`` carry units of (field amplitude)², so an
#: absolute epsilon made the weights, and hence the scheme, depend on the
#: field's amplitude.  The weights here use ``eps_eff = _WENO_EPS_REL · Σβ``
#: instead, so they depend on the stencil only through ratios and the
#: reconstruction is exactly amplitude-covariant.  A constant stencil (all
#: ``beta_k = 0``) gives exactly the linear weights.  ``1e-12`` keeps the
#: epsilon out of the weights wherever a sub-stencil is resolved at all: at
#: amplitude 1 the tendencies of WENO-Z and of every scheme at a step are
#: unchanged to < 2e-11 relative (WENO-JS on smooth fields moves by <= 4e-5,
#: the influence the old absolute ``1e-8`` had there).
_WENO_EPS_REL: float = 1e-12


def _scaled_stencil(*q: Array) -> tuple[Array | float, ...]:
    """Stencil differences rescaled to O(1), for the smoothness indicators.

    ``d_j = (q_j - q_0) / s`` with ``s = max_j |q_j - q_0|`` (``1`` on a
    constant stencil), so ``max_j |d_j| = 1``.  The indicators are quadratic
    forms that vanish on constants, so ``beta_k(d) = beta_k(q) / s²``, and
    the weights are invariant to that common factor; computing them on ``d``
    keeps every intermediate O(1), so neither the indicators nor their
    gradients under- or overflow at small or large amplitude (float32
    included).  ``s`` is held out of the gradient (``stop_gradient``): exact,
    because the weights do not depend on it.
    """
    diffs = [q_j - q[0] for q_j in q[1:]]
    scale = jnp.abs(diffs[0])
    for d in diffs[1:]:
        scale = jnp.maximum(scale, jnp.abs(d))
    scale = jax.lax.stop_gradient(scale)
    inv_scale = 1.0 / jnp.where(scale > 0.0, scale, 1.0)
    return (0.0, *(d * inv_scale for d in diffs))


def _js_alphas(
    linear_weights: tuple[float, ...], betas: tuple[Array | float, ...]
) -> list[Array]:
    """Unnormalised WENO-JS weights, in ``(0, g_k]``.

    ``alpha_k = g_k · ((beta_min + eps_eff) / (beta_k + eps_eff))²`` with
    ``eps_eff = _WENO_EPS_REL · Σβ``: the normalised weights of
    ``g_k / (beta_k + eps_eff)²``, scaled so the smoothest sub-stencil has
    ``alpha = g_k``; ``Σα`` is then bounded below by ``min g_k``, so neither
    the weights nor the gradient of their normalisation overflow.  A
    constant stencil (``Σβ = 0``) gives ``alpha_k = g_k``, through a double
    ``where`` so its gradient stays finite.
    """
    total = sum(betas)
    pos = total > 0.0
    eps_eff = _WENO_EPS_REL * total
    beta_min = betas[0]
    for b in betas[1:]:
        beta_min = jnp.minimum(beta_min, b)
    num = beta_min + eps_eff
    return [
        g * jnp.where(pos, num / jnp.where(pos, b + eps_eff, 1.0), 1.0) ** 2
        for g, b in zip(linear_weights, betas, strict=True)
    ]


def _z_alphas(
    linear_weights: tuple[float, ...], betas: tuple[Array | float, ...], tau: Array
) -> list[Array]:
    """Unnormalised WENO-Z weights ``g_k (1 + tau / (beta_k + eps_eff))``.

    ``eps_eff = _WENO_EPS_REL · Σβ``; in ``[g_k, g_k (1 + 1/eps_rel)]``.  A
    constant stencil gives ``g_k`` (double ``where``, finite gradient).
    """
    total = sum(betas)
    pos = total > 0.0
    eps_eff = _WENO_EPS_REL * total
    return [
        g * (1.0 + jnp.where(pos, tau / jnp.where(pos, b + eps_eff, 1.0), 0.0))
        for g, b in zip(linear_weights, betas, strict=True)
    ]


def _weighted(alphas: list[Array], candidates: list[Array]) -> Array:
    """``Σ alpha_k q_k / Σ alpha_k``."""
    return sum(a * c for a, c in zip(alphas, candidates, strict=True)) / sum(alphas)


_WENO7_COEFFS = (
    (-1.0 / 4.0, 13.0 / 12.0, -23.0 / 12.0, 25.0 / 12.0),
    (1.0 / 12.0, -5.0 / 12.0, 13.0 / 12.0, 1.0 / 4.0),
    (-1.0 / 12.0, 7.0 / 12.0, 7.0 / 12.0, -1.0 / 12.0),
    (1.0 / 4.0, 13.0 / 12.0, -5.0 / 12.0, 1.0 / 12.0),
)
_WENO7_WEIGHTS = (1.0 / 35.0, 12.0 / 35.0, 18.0 / 35.0, 4.0 / 35.0)
_WENO7_BETA_MATRICES = (
    np.array(
        [
            [547.0 / 240.0, -647.0 / 80.0, 2321.0 / 240.0, -309.0 / 80.0],
            [-647.0 / 80.0, 7043.0 / 240.0, -8623.0 / 240.0, 3521.0 / 240.0],
            [2321.0 / 240.0, -8623.0 / 240.0, 11003.0 / 240.0, -1567.0 / 80.0],
            [-309.0 / 80.0, 3521.0 / 240.0, -1567.0 / 80.0, 2107.0 / 240.0],
        ]
    ),
    np.array(
        [
            [89.0 / 80.0, -821.0 / 240.0, 267.0 / 80.0, -247.0 / 240.0],
            [-821.0 / 240.0, 2843.0 / 240.0, -2983.0 / 240.0, 961.0 / 240.0],
            [267.0 / 80.0, -2983.0 / 240.0, 3443.0 / 240.0, -1261.0 / 240.0],
            [-247.0 / 240.0, 961.0 / 240.0, -1261.0 / 240.0, 547.0 / 240.0],
        ]
    ),
    np.array(
        [
            [547.0 / 240.0, -1261.0 / 240.0, 961.0 / 240.0, -247.0 / 240.0],
            [-1261.0 / 240.0, 3443.0 / 240.0, -2983.0 / 240.0, 267.0 / 80.0],
            [961.0 / 240.0, -2983.0 / 240.0, 2843.0 / 240.0, -821.0 / 240.0],
            [-247.0 / 240.0, 267.0 / 80.0, -821.0 / 240.0, 89.0 / 80.0],
        ]
    ),
    np.array(
        [
            [2107.0 / 240.0, -1567.0 / 80.0, 3521.0 / 240.0, -309.0 / 80.0],
            [-1567.0 / 80.0, 11003.0 / 240.0, -8623.0 / 240.0, 2321.0 / 240.0],
            [3521.0 / 240.0, -8623.0 / 240.0, 7043.0 / 240.0, -647.0 / 80.0],
            [-309.0 / 80.0, 2321.0 / 240.0, -647.0 / 80.0, 547.0 / 240.0],
        ]
    ),
)

_WENO9_COEFFS = (
    (1.0 / 5.0, -21.0 / 20.0, 137.0 / 60.0, -163.0 / 60.0, 137.0 / 60.0),
    (-1.0 / 20.0, 17.0 / 60.0, -43.0 / 60.0, 77.0 / 60.0, 1.0 / 5.0),
    (1.0 / 30.0, -13.0 / 60.0, 47.0 / 60.0, 9.0 / 20.0, -1.0 / 20.0),
    (-1.0 / 20.0, 9.0 / 20.0, 47.0 / 60.0, -13.0 / 60.0, 1.0 / 30.0),
    (1.0 / 5.0, 77.0 / 60.0, -43.0 / 60.0, 17.0 / 60.0, -1.0 / 20.0),
)
_WENO9_WEIGHTS = (1.0 / 126.0, 10.0 / 63.0, 10.0 / 21.0, 20.0 / 63.0, 5.0 / 126.0)
_WENO9_BETA_MATRICES = (
    np.array(
        [
            [
                11329.0 / 2520.0,
                -208501.0 / 10080.0,
                121621.0 / 3360.0,
                -288007.0 / 10080.0,
                86329.0 / 10080.0,
            ],
            [
                -208501.0 / 10080.0,
                482963.0 / 5040.0,
                -142033.0 / 840.0,
                679229.0 / 5040.0,
                -411487.0 / 10080.0,
            ],
            [
                121621.0 / 3360.0,
                -142033.0 / 840.0,
                507131.0 / 1680.0,
                -68391.0 / 280.0,
                252941.0 / 3360.0,
            ],
            [
                -288007.0 / 10080.0,
                679229.0 / 5040.0,
                -68391.0 / 280.0,
                1020563.0 / 5040.0,
                -649501.0 / 10080.0,
            ],
            [
                86329.0 / 10080.0,
                -411487.0 / 10080.0,
                252941.0 / 3360.0,
                -649501.0 / 10080.0,
                53959.0 / 2520.0,
            ],
        ]
    ),
    np.array(
        [
            [
                1727.0 / 1260.0,
                -60871.0 / 10080.0,
                33071.0 / 3360.0,
                -70237.0 / 10080.0,
                18079.0 / 10080.0,
            ],
            [
                -60871.0 / 10080.0,
                138563.0 / 5040.0,
                -3229.0 / 70.0,
                168509.0 / 5040.0,
                -88297.0 / 10080.0,
            ],
            [
                33071.0 / 3360.0,
                -3229.0 / 70.0,
                135431.0 / 1680.0,
                -25499.0 / 420.0,
                55051.0 / 3360.0,
            ],
            [
                -70237.0 / 10080.0,
                168509.0 / 5040.0,
                -25499.0 / 420.0,
                242723.0 / 5040.0,
                -140251.0 / 10080.0,
            ],
            [
                18079.0 / 10080.0,
                -88297.0 / 10080.0,
                55051.0 / 3360.0,
                -140251.0 / 10080.0,
                11329.0 / 2520.0,
            ],
        ]
    ),
    np.array(
        [
            [
                1727.0 / 1260.0,
                -51001.0 / 10080.0,
                7547.0 / 1120.0,
                -38947.0 / 10080.0,
                8209.0 / 10080.0,
            ],
            [
                -51001.0 / 10080.0,
                104963.0 / 5040.0,
                -24923.0 / 840.0,
                89549.0 / 5040.0,
                -38947.0 / 10080.0,
            ],
            [
                7547.0 / 1120.0,
                -24923.0 / 840.0,
                77051.0 / 1680.0,
                -24923.0 / 840.0,
                7547.0 / 1120.0,
            ],
            [
                -38947.0 / 10080.0,
                89549.0 / 5040.0,
                -24923.0 / 840.0,
                104963.0 / 5040.0,
                -51001.0 / 10080.0,
            ],
            [
                8209.0 / 10080.0,
                -38947.0 / 10080.0,
                7547.0 / 1120.0,
                -51001.0 / 10080.0,
                1727.0 / 1260.0,
            ],
        ]
    ),
    np.array(
        [
            [
                11329.0 / 2520.0,
                -140251.0 / 10080.0,
                55051.0 / 3360.0,
                -88297.0 / 10080.0,
                18079.0 / 10080.0,
            ],
            [
                -140251.0 / 10080.0,
                242723.0 / 5040.0,
                -25499.0 / 420.0,
                168509.0 / 5040.0,
                -70237.0 / 10080.0,
            ],
            [
                55051.0 / 3360.0,
                -25499.0 / 420.0,
                135431.0 / 1680.0,
                -3229.0 / 70.0,
                33071.0 / 3360.0,
            ],
            [
                -88297.0 / 10080.0,
                168509.0 / 5040.0,
                -3229.0 / 70.0,
                138563.0 / 5040.0,
                -60871.0 / 10080.0,
            ],
            [
                18079.0 / 10080.0,
                -70237.0 / 10080.0,
                33071.0 / 3360.0,
                -60871.0 / 10080.0,
                1727.0 / 1260.0,
            ],
        ]
    ),
    np.array(
        [
            [
                53959.0 / 2520.0,
                -649501.0 / 10080.0,
                252941.0 / 3360.0,
                -411487.0 / 10080.0,
                86329.0 / 10080.0,
            ],
            [
                -649501.0 / 10080.0,
                1020563.0 / 5040.0,
                -68391.0 / 280.0,
                679229.0 / 5040.0,
                -288007.0 / 10080.0,
            ],
            [
                252941.0 / 3360.0,
                -68391.0 / 280.0,
                507131.0 / 1680.0,
                -142033.0 / 840.0,
                121621.0 / 3360.0,
            ],
            [
                -411487.0 / 10080.0,
                679229.0 / 5040.0,
                -142033.0 / 840.0,
                482963.0 / 5040.0,
                -208501.0 / 10080.0,
            ],
            [
                86329.0 / 10080.0,
                -288007.0 / 10080.0,
                121621.0 / 3360.0,
                -208501.0 / 10080.0,
                11329.0 / 2520.0,
            ],
        ]
    ),
)


def _linear_combo(coeffs: tuple[float, ...], stencil: tuple[Array, ...]) -> Array:
    return sum(coeff * value for coeff, value in zip(coeffs, stencil, strict=True))


def _smoothness_indicator(
    beta_matrix: np.ndarray, stencil: tuple[Array | float, ...]
) -> Array:
    # beta = q^T B q = sum_i q_i * (sum_j B[i, j] * q_j)
    #
    # Same quadratic form as
    #   jnp.einsum("i...,ij,j...->...", jnp.stack(stencil), B, jnp.stack(stencil))
    # but unrolled over the (at most 5x5) constant matrix, so it lowers to one
    # fused elementwise pass instead of a stack, a dot and transposes of the
    # stacked field (WENO-7 advection forward ~11x, WENO-9 ~14x on CPU).  The
    # coefficients are Python floats (weakly typed), so the result keeps the
    # input dtype under float32 and float64.
    return sum(
        q_i * sum(float(beta_matrix[i, j]) * q_j for j, q_j in enumerate(stencil))
        for i, q_i in enumerate(stencil)
    )


def _weno_reconstruct(
    cells: tuple[Array, ...],
    coeffs: tuple[tuple[float, ...], ...],
    beta_matrices: tuple[np.ndarray, ...],
    linear_weights: tuple[float, ...],
) -> Array:
    """WENO-JS from ``K`` overlapping sub-stencils of ``r`` cells each.

    Sub-stencil ``k`` is ``cells[k : k + r]``; candidates use the raw cells,
    the indicators the rescaled stencil (:func:`_scaled_stencil`).
    """
    width = len(coeffs[0])
    windows = range(len(coeffs))
    scaled = _scaled_stencil(*cells)
    candidates = [_linear_combo(coeffs[k], cells[k : k + width]) for k in windows]
    betas = tuple(
        _smoothness_indicator(beta_matrices[k], scaled[k : k + width]) for k in windows
    )
    return _weighted(_js_alphas(linear_weights, betas), candidates)


def weno_3pts(qm: Array, q0: Array, qp: Array) -> Array:
    """
    3-points non-linear left-biased stencil reconstruction:

    qm-----q0--x--qp

    Efficient Implementation of Weighted ENO Schemes, Jiang and Shu,
    Journal of Computation Physics 126, 202–228 (1996)
    """

    qi1 = -1.0 / 2.0 * qm + 3.0 / 2.0 * q0
    qi2 = 1.0 / 2.0 * (q0 + qp)

    # indicators on the rescaled stencil (dm, d0, dp) = (q - qm) / s
    dm, d0, dp = _scaled_stencil(qm, q0, qp)
    beta1 = (d0 - dm) ** 2
    beta2 = (dp - d0) ** 2

    # w_k ∝ g_k / (beta_k + eps_rel Σβ)²,  (g1, g2) = (1/3, 2/3)
    alphas = _js_alphas((1.0 / 3.0, 2.0 / 3.0), (beta1, beta2))
    return _weighted(alphas, [qi1, qi2])


def weno_3pts_improved(qm: Array, q0: Array, qp: Array) -> Array:
    """
    3-points non-linear left-biased stencil reconstruction:

    qm-----q0--x--qp

    An improved weighted essentially non-oscillatory scheme for hyperbolic
    conservation laws, Borges et al, Journal of Computational Physics 227 (2008).
    """

    qi1 = -1.0 / 2.0 * qm + 3.0 / 2.0 * q0
    qi2 = 1.0 / 2.0 * (q0 + qp)

    dm, d0, dp = _scaled_stencil(qm, q0, qp)
    beta1 = (d0 - dm) ** 2
    beta2 = (dp - d0) ** 2
    tau = jnp.abs(beta2 - beta1)

    # w_k ∝ g_k (1 + tau / (beta_k + eps_rel Σβ))
    alphas = _z_alphas((1.0 / 3.0, 2.0 / 3.0), (beta1, beta2), tau)
    return _weighted(alphas, [qi1, qi2])


def weno_5pts(qmm: Array, qm: Array, q0: Array, qp: Array, qpp: Array) -> Array:
    """
    5-points non-linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    Efficient Implementation of Weighted ENO Schemes, Jiang and Shu,
    Journal of Computation Physics 126, 202–228 (1996)
    """
    qi1 = 1.0 / 3.0 * qmm - 7.0 / 6.0 * qm + 11.0 / 6.0 * q0
    qi2 = -1.0 / 6.0 * qm + 5.0 / 6.0 * q0 + 1.0 / 3.0 * qp
    qi3 = 1.0 / 3.0 * q0 + 5.0 / 6.0 * qp - 1.0 / 6.0 * qpp

    k1, k2 = 13.0 / 12.0, 0.25
    # indicators on the rescaled stencil d = (q - qmm) / s
    dmm, dm, d0, dp, dpp = _scaled_stencil(qmm, qm, q0, qp, qpp)
    beta1 = k1 * (dmm - 2 * dm + d0) ** 2 + k2 * (dmm - 4 * dm + 3 * d0) ** 2
    beta2 = k1 * (dm - 2 * d0 + dp) ** 2 + k2 * (dm - dp) ** 2
    beta3 = k1 * (d0 - 2 * dp + dpp) ** 2 + k2 * (3 * d0 - 4 * dp + dpp) ** 2

    # w_k ∝ g_k / (beta_k + eps_rel Σβ)²,  (g1, g2, g3) = (0.1, 0.6, 0.3)
    alphas = _js_alphas((0.1, 0.6, 0.3), (beta1, beta2, beta3))

    return _weighted(alphas, [qi1, qi2, qi3])


def weno_5pts_improved(
    qmm: Array, qm: Array, q0: Array, qp: Array, qpp: Array
) -> Array:
    """
    5-points non-linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    An improved weighted essentially non-oscillatory scheme for hyperbolic
    conservation laws, Borges et al, Journal of Computational Physics 227 (2008)
    """

    qi1 = 1.0 / 3.0 * qmm - 7.0 / 6.0 * qm + 11.0 / 6.0 * q0
    qi2 = -1.0 / 6.0 * qm + 5.0 / 6.0 * q0 + 1.0 / 3.0 * qp
    qi3 = 1.0 / 3.0 * q0 + 5.0 / 6.0 * qp - 1.0 / 6.0 * qpp

    k1, k2 = 13.0 / 12.0, 0.25
    # indicators on the rescaled stencil d = (q - qmm) / s
    dmm, dm, d0, dp, dpp = _scaled_stencil(qmm, qm, q0, qp, qpp)
    beta1 = k1 * (dmm - 2 * dm + d0) ** 2 + k2 * (dmm - 4 * dm + 3 * d0) ** 2
    beta2 = k1 * (dm - 2 * d0 + dp) ** 2 + k2 * (dm - dp) ** 2
    beta3 = k1 * (d0 - 2 * dp + dpp) ** 2 + k2 * (3 * d0 - 4 * dp + dpp) ** 2

    tau5 = jnp.abs(beta1 - beta3)

    # w_k ∝ g_k (1 + tau5 / (beta_k + eps_rel Σβ))
    alphas = _z_alphas((0.1, 0.6, 0.3), (beta1, beta2, beta3), tau5)

    return _weighted(alphas, [qi1, qi2, qi3])


def weno_3pts_right(q0: Array, qp: Array, qpp: Array) -> Array:
    """
    3-points non-linear right-biased stencil reconstruction at face q0+1/2:

    q0--x--qp-----qpp

    Uses sub-stencils {q0, qp} and {qp, qpp} with reversed optimal weights.

    Jiang and Shu, J. Comput. Phys. 126, 202–228 (1996), Section 2.3.
    """

    qi1 = 0.5 * (q0 + qp)
    qi2 = 1.5 * qp - 0.5 * qpp

    d0, dp, dpp = _scaled_stencil(q0, qp, qpp)
    beta1 = (dp - d0) ** 2
    beta2 = (dpp - dp) ** 2

    alphas = _js_alphas((2.0 / 3.0, 1.0 / 3.0), (beta1, beta2))
    return _weighted(alphas, [qi1, qi2])


def weno_3pts_improved_right(q0: Array, qp: Array, qpp: Array) -> Array:
    """
    3-points non-linear right-biased WENO-Z stencil reconstruction at face q0+1/2:

    q0--x--qp-----qpp

    Borges et al, J. Comput. Phys. 227 (2008).
    """

    qi1 = 0.5 * (q0 + qp)
    qi2 = 1.5 * qp - 0.5 * qpp

    d0, dp, dpp = _scaled_stencil(q0, qp, qpp)
    beta1 = (dp - d0) ** 2
    beta2 = (dpp - dp) ** 2
    tau = jnp.abs(beta2 - beta1)

    alphas = _z_alphas((2.0 / 3.0, 1.0 / 3.0), (beta1, beta2), tau)
    return _weighted(alphas, [qi1, qi2])


def weno_5pts_right(qm: Array, q0: Array, qp: Array, qpp: Array, qppp: Array) -> Array:
    """
    5-points non-linear right-biased stencil reconstruction at face q0+1/2:

    qm-----q0--x--qp----qpp---qppp

    Uses cells {qm, q0, qp, qpp, qppp} to reconstruct the face value from
    the right (upwind for negative velocity).

    Jiang and Shu, J. Comput. Phys. 126, 202–228 (1996), Section 2.3.
    """

    # Candidate reconstructions at face q0+1/2
    qi1 = 1.0 / 3.0 * qppp - 7.0 / 6.0 * qpp + 11.0 / 6.0 * qp
    qi2 = -1.0 / 6.0 * qpp + 5.0 / 6.0 * qp + 1.0 / 3.0 * q0
    qi3 = 1.0 / 3.0 * qp + 5.0 / 6.0 * q0 - 1.0 / 6.0 * qm

    # Smoothness indicators
    k1, k2 = 13.0 / 12.0, 0.25
    # indicators on the rescaled stencil d = (q - qm) / s
    dm, d0, dp, dpp, dppp = _scaled_stencil(qm, q0, qp, qpp, qppp)
    beta1 = k1 * (dp - 2 * dpp + dppp) ** 2 + k2 * (3 * dp - 4 * dpp + dppp) ** 2
    beta2 = k1 * (d0 - 2 * dp + dpp) ** 2 + k2 * (d0 - dpp) ** 2
    beta3 = k1 * (dm - 2 * d0 + dp) ** 2 + k2 * (dm - 4 * d0 + 3 * dp) ** 2

    # w_k ∝ g_k / (beta_k + eps_rel Σβ)²,  (g1, g2, g3) = (0.1, 0.6, 0.3)
    alphas = _js_alphas((0.1, 0.6, 0.3), (beta1, beta2, beta3))

    return _weighted(alphas, [qi1, qi2, qi3])


def weno_5pts_improved_right(
    qm: Array, q0: Array, qp: Array, qpp: Array, qppp: Array
) -> Array:
    """
    5-points non-linear right-biased WENO-Z stencil reconstruction at face q0+1/2:

    qm-----q0--x--qp----qpp---qppp

    Borges et al, J. Comput. Phys. 227 (2008).
    """

    qi1 = 1.0 / 3.0 * qppp - 7.0 / 6.0 * qpp + 11.0 / 6.0 * qp
    qi2 = -1.0 / 6.0 * qpp + 5.0 / 6.0 * qp + 1.0 / 3.0 * q0
    qi3 = 1.0 / 3.0 * qp + 5.0 / 6.0 * q0 - 1.0 / 6.0 * qm

    k1, k2 = 13.0 / 12.0, 0.25
    # indicators on the rescaled stencil d = (q - qm) / s
    dm, d0, dp, dpp, dppp = _scaled_stencil(qm, q0, qp, qpp, qppp)
    beta1 = k1 * (dp - 2 * dpp + dppp) ** 2 + k2 * (3 * dp - 4 * dpp + dppp) ** 2
    beta2 = k1 * (d0 - 2 * dp + dpp) ** 2 + k2 * (d0 - dpp) ** 2
    beta3 = k1 * (dm - 2 * d0 + dp) ** 2 + k2 * (dm - 4 * d0 + 3 * dp) ** 2

    tau5 = jnp.abs(beta1 - beta3)

    # w_k ∝ g_k (1 + tau5 / (beta_k + eps_rel Σβ))
    alphas = _z_alphas((0.1, 0.6, 0.3), (beta1, beta2, beta3), tau5)

    return _weighted(alphas, [qi1, qi2, qi3])


_WENO7_COEFFS_RIGHT = tuple(tuple(reversed(c)) for c in reversed(_WENO7_COEFFS))
_WENO7_WEIGHTS_RIGHT = tuple(reversed(_WENO7_WEIGHTS))
_WENO7_BETA_MATRICES_RIGHT = tuple(
    m[::-1, ::-1] for m in reversed(_WENO7_BETA_MATRICES)
)

_WENO9_COEFFS_RIGHT = tuple(tuple(reversed(c)) for c in reversed(_WENO9_COEFFS))
_WENO9_WEIGHTS_RIGHT = tuple(reversed(_WENO9_WEIGHTS))
_WENO9_BETA_MATRICES_RIGHT = tuple(
    m[::-1, ::-1] for m in reversed(_WENO9_BETA_MATRICES)
)


def weno_7pts(
    qmmm: Array,
    qmm: Array,
    qm: Array,
    q0: Array,
    qp: Array,
    qpp: Array,
    qppp: Array,
) -> Array:
    """
    7-points non-linear left-biased stencil reconstruction

    qmmm---qmm----qm-----q0--x--qp----qpp---qppp

    Efficient finite-volume WENO-7 reconstruction using Jiang-Shu
    smoothness indicators.
    """
    return _weno_reconstruct(
        (qmmm, qmm, qm, q0, qp, qpp, qppp),
        _WENO7_COEFFS,
        _WENO7_BETA_MATRICES,
        _WENO7_WEIGHTS,
    )


def weno_9pts(
    qmmmm: Array,
    qmmm: Array,
    qmm: Array,
    qm: Array,
    q0: Array,
    qp: Array,
    qpp: Array,
    qppp: Array,
    qpppp: Array,
) -> Array:
    """
    9-points non-linear left-biased stencil reconstruction

    qmmmm--qmmm---qmm----qm-----q0--x--qp----qpp---qppp--qpppp

    Efficient finite-volume WENO-9 reconstruction using Jiang-Shu
    smoothness indicators.
    """
    return _weno_reconstruct(
        (qmmmm, qmmm, qmm, qm, q0, qp, qpp, qppp, qpppp),
        _WENO9_COEFFS,
        _WENO9_BETA_MATRICES,
        _WENO9_WEIGHTS,
    )


def weno_7pts_right(
    qmm: Array,
    qm: Array,
    q0: Array,
    qp: Array,
    qpp: Array,
    qppp: Array,
    qpppp: Array,
) -> Array:
    """
    7-points non-linear right-biased stencil reconstruction at face q0+1/2:

    qmm----qm-----q0--x--qp----qpp---qppp--qpppp

    Uses cells {q_{i-2}..q_{i+4}} for right-biased (negative-flow) WENO-7
    reconstruction.  Sub-stencil coefficients, optimal weights, and smoothness
    indicators are the mirror of the left-biased formulas.

    Jiang and Shu, J. Comput. Phys. 126, 202–228 (1996), Section 2.3.
    """
    return _weno_reconstruct(
        (qmm, qm, q0, qp, qpp, qppp, qpppp),
        _WENO7_COEFFS_RIGHT,
        _WENO7_BETA_MATRICES_RIGHT,
        _WENO7_WEIGHTS_RIGHT,
    )


def weno_9pts_right(
    qmmm: Array,
    qmm: Array,
    qm: Array,
    q0: Array,
    qp: Array,
    qpp: Array,
    qppp: Array,
    qpppp: Array,
    qppppp: Array,
) -> Array:
    """
    9-points non-linear right-biased stencil reconstruction at face q0+1/2:

    qmmm---qmm----qm-----q0--x--qp----qpp---qppp--qpppp-qppppp

    Uses cells {q_{i-3}..q_{i+5}} for right-biased (negative-flow) WENO-9
    reconstruction.  Sub-stencil coefficients, optimal weights, and smoothness
    indicators are the mirror of the left-biased formulas.

    Jiang and Shu, J. Comput. Phys. 126, 202–228 (1996), Section 2.3.
    """
    return _weno_reconstruct(
        (qmmm, qmm, qm, q0, qp, qpp, qppp, qpppp, qppppp),
        _WENO9_COEFFS_RIGHT,
        _WENO9_BETA_MATRICES_RIGHT,
        _WENO9_WEIGHTS_RIGHT,
    )
