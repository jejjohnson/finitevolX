import jax.numpy as jnp
from jaxtyping import Array


_WENO7_COEFFS = (
    (-1.0 / 4.0, 13.0 / 12.0, -23.0 / 12.0, 25.0 / 12.0),
    (1.0 / 12.0, -5.0 / 12.0, 13.0 / 12.0, 1.0 / 4.0),
    (-1.0 / 12.0, 7.0 / 12.0, 7.0 / 12.0, -1.0 / 12.0),
    (1.0 / 4.0, 13.0 / 12.0, -5.0 / 12.0, 1.0 / 12.0),
)
_WENO7_WEIGHTS = (1.0 / 35.0, 12.0 / 35.0, 18.0 / 35.0, 4.0 / 35.0)
_WENO7_BETA_MATRICES = (
    jnp.array(
        [
            [547.0 / 240.0, -647.0 / 80.0, 2321.0 / 240.0, -309.0 / 80.0],
            [-647.0 / 80.0, 7043.0 / 240.0, -8623.0 / 240.0, 3521.0 / 240.0],
            [2321.0 / 240.0, -8623.0 / 240.0, 11003.0 / 240.0, -1567.0 / 80.0],
            [-309.0 / 80.0, 3521.0 / 240.0, -1567.0 / 80.0, 2107.0 / 240.0],
        ]
    ),
    jnp.array(
        [
            [89.0 / 80.0, -821.0 / 240.0, 267.0 / 80.0, -247.0 / 240.0],
            [-821.0 / 240.0, 2843.0 / 240.0, -2983.0 / 240.0, 961.0 / 240.0],
            [267.0 / 80.0, -2983.0 / 240.0, 3443.0 / 240.0, -1261.0 / 240.0],
            [-247.0 / 240.0, 961.0 / 240.0, -1261.0 / 240.0, 547.0 / 240.0],
        ]
    ),
    jnp.array(
        [
            [547.0 / 240.0, -1261.0 / 240.0, 961.0 / 240.0, -247.0 / 240.0],
            [-1261.0 / 240.0, 3443.0 / 240.0, -2983.0 / 240.0, 267.0 / 80.0],
            [961.0 / 240.0, -2983.0 / 240.0, 2843.0 / 240.0, -821.0 / 240.0],
            [-247.0 / 240.0, 267.0 / 80.0, -821.0 / 240.0, 89.0 / 80.0],
        ]
    ),
    jnp.array(
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
    jnp.array(
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
    jnp.array(
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
    jnp.array(
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
    jnp.array(
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
    jnp.array(
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


def _smoothness_indicator(beta_matrix: Array, stencil: tuple[Array, ...]) -> Array:
    values = jnp.stack(stencil, axis=0)
    return jnp.einsum("i...,ij,j...->...", values, beta_matrix, values)


def _weno_reconstruct(
    stencils: tuple[tuple[Array, ...], ...],
    coeffs: tuple[tuple[float, ...], ...],
    beta_matrices: tuple[Array, ...],
    linear_weights: tuple[float, ...],
    eps: float = 1e-8,
) -> Array:
    candidates = [
        _linear_combo(stencil_coeffs, stencil)
        for stencil_coeffs, stencil in zip(coeffs, stencils, strict=True)
    ]
    betas = [
        _smoothness_indicator(beta_matrix, stencil)
        for beta_matrix, stencil in zip(beta_matrices, stencils, strict=True)
    ]
    alphas = [
        weight / (beta + eps) ** 2
        for weight, beta in zip(linear_weights, betas, strict=True)
    ]
    alpha_sum = sum(alphas)
    return (
        sum(alpha * candidate for alpha, candidate in zip(alphas, candidates, strict=True))
        / alpha_sum
    )


def weno_3pts(qm: Array, q0: Array, qp: Array) -> Array:
    """
    3-points non-linear left-biased stencil reconstruction:

    qm-----q0--x--qp

    Efficient Implementation of Weighted ENO Schemes, Jiang and Shu,
    Journal of Computation Physics 126, 202–228 (1996)
    """
    eps = 1e-8

    qi1 = -1.0 / 2.0 * qm + 3.0 / 2.0 * q0
    qi2 = 1.0 / 2.0 * (q0 + qp)

    beta1 = (q0 - qm) ** 2
    beta2 = (qp - q0) ** 2

    g1, g2 = 1.0 / 3.0, 2.0 / 3.0
    w1 = g1 / (beta1 + eps) ** 2
    w2 = g2 / (beta2 + eps) ** 2

    qi_weno3 = (w1 * qi1 + w2 * qi2) / (w1 + w2)

    return qi_weno3


def weno_3pts_improved(qm: Array, q0: Array, qp: Array) -> Array:
    """
    3-points non-linear left-biased stencil reconstruction:

    qm-----q0--x--qp

    An improved weighted essentially non-oscillatory scheme for hyperbolic
    conservation laws, Borges et al, Journal of Computational Physics 227 (2008).
    """
    eps = 1e-14

    qi1 = -1.0 / 2.0 * qm + 3.0 / 2.0 * q0
    qi2 = 1.0 / 2.0 * (q0 + qp)

    beta1 = (q0 - qm) ** 2
    beta2 = (qp - q0) ** 2
    tau = jnp.abs(beta2 - beta1)

    g1, g2 = 1.0 / 3.0, 2.0 / 3.0
    w1 = g1 * (1.0 + tau / (beta1 + eps))
    w2 = g2 * (1.0 + tau / (beta2 + eps))

    qi_weno3 = (w1 * qi1 + w2 * qi2) / (w1 + w2)

    return qi_weno3


def weno_5pts(qmm: Array, qm: Array, q0: Array, qp: Array, qpp: Array) -> Array:
    """
    5-points non-linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    Efficient Implementation of Weighted ENO Schemes, Jiang and Shu,
    Journal of Computation Physics 126, 202–228 (1996)
    """
    eps = 1e-8
    qi1 = 1.0 / 3.0 * qmm - 7.0 / 6.0 * qm + 11.0 / 6.0 * q0
    qi2 = -1.0 / 6.0 * qm + 5.0 / 6.0 * q0 + 1.0 / 3.0 * qp
    qi3 = 1.0 / 3.0 * q0 + 5.0 / 6.0 * qp - 1.0 / 6.0 * qpp

    k1, k2 = 13.0 / 12.0, 0.25
    beta1 = k1 * (qmm - 2 * qm + q0) ** 2 + k2 * (qmm - 4 * qm + 3 * q0) ** 2
    beta2 = k1 * (qm - 2 * q0 + qp) ** 2 + k2 * (qm - qp) ** 2
    beta3 = k1 * (q0 - 2 * qp + qpp) ** 2 + k2 * (3 * q0 - 4 * qp + qpp) ** 2

    g1, g2, g3 = 0.1, 0.6, 0.3
    w1 = g1 / (beta1 + eps) ** 2
    w2 = g2 / (beta2 + eps) ** 2
    w3 = g3 / (beta3 + eps) ** 2

    qi_weno5 = (w1 * qi1 + w2 * qi2 + w3 * qi3) / (w1 + w2 + w3)

    return qi_weno5


def weno_5pts_improved(
    qmm: Array, qm: Array, q0: Array, qp: Array, qpp: Array
) -> Array:
    """
    5-points non-linear left-biased stencil reconstruction

    qmm----qm-----q0--x--qp----qpp

    An improved weighted essentially non-oscillatory scheme for hyperbolic
    conservation laws, Borges et al, Journal of Computational Physics 227 (2008)
    """
    eps = 1e-16

    qi1 = 1.0 / 3.0 * qmm - 7.0 / 6.0 * qm + 11.0 / 6.0 * q0
    qi2 = -1.0 / 6.0 * qm + 5.0 / 6.0 * q0 + 1.0 / 3.0 * qp
    qi3 = 1.0 / 3.0 * q0 + 5.0 / 6.0 * qp - 1.0 / 6.0 * qpp

    k1, k2 = 13.0 / 12.0, 0.25
    beta1 = k1 * (qmm - 2 * qm + q0) ** 2 + k2 * (qmm - 4 * qm + 3 * q0) ** 2
    beta2 = k1 * (qm - 2 * q0 + qp) ** 2 + k2 * (qm - qp) ** 2
    beta3 = k1 * (q0 - 2 * qp + qpp) ** 2 + k2 * (3 * q0 - 4 * qp + qpp) ** 2

    tau5 = jnp.abs(beta1 - beta3)

    g1, g2, g3 = 0.1, 0.6, 0.3
    w1 = g1 * (1 + tau5 / (beta1 + eps))
    w2 = g2 * (1 + tau5 / (beta2 + eps))
    w3 = g3 * (1 + tau5 / (beta3 + eps))

    qi_weno5 = (w1 * qi1 + w2 * qi2 + w3 * qi3) / (w1 + w2 + w3)

    return qi_weno5


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
    stencils = (
        (qmmm, qmm, qm, q0),
        (qmm, qm, q0, qp),
        (qm, q0, qp, qpp),
        (q0, qp, qpp, qppp),
    )
    return _weno_reconstruct(stencils, _WENO7_COEFFS, _WENO7_BETA_MATRICES, _WENO7_WEIGHTS)


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
    stencils = (
        (qmmmm, qmmm, qmm, qm, q0),
        (qmmm, qmm, qm, q0, qp),
        (qmm, qm, q0, qp, qpp),
        (qm, q0, qp, qpp, qppp),
        (q0, qp, qpp, qppp, qpppp),
    )
    return _weno_reconstruct(stencils, _WENO9_COEFFS, _WENO9_BETA_MATRICES, _WENO9_WEIGHTS)
