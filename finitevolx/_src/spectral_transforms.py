"""JAX-native spectral transforms: DCT and DST types I–IV.

All transforms follow the unnormalized scipy convention (norm=None):

* DCT-I:   Y[k] = x[0] + (-1)^k x[N-1] + 2 Σ_{n=1}^{N-2} x[n] cos(πnk/(N-1))
* DCT-II:  Y[k] = 2 Σ_{n=0}^{N-1} x[n] cos(πk(2n+1)/(2N))
* DCT-III: Y[k] = x[0] + 2 Σ_{n=1}^{N-1} x[n] cos(πn(2k+1)/(2N))
* DCT-IV:  Y[k] = 2 Σ_{n=0}^{N-1} x[n] cos(π(2n+1)(2k+1)/(4N))

* DST-I:   Y[k] = 2 Σ_{n=0}^{N-1} x[n] sin(π(n+1)(k+1)/(N+1))
* DST-II:  Y[k] = 2 Σ_{n=0}^{N-1} x[n] sin(π(2n+1)(k+1)/(2N))
* DST-III: Y[k] = (-1)^k x[N-1] + 2 Σ_{n=0}^{N-2} x[n] sin(π(n+1)(2k+1)/(2N))
* DST-IV:  Y[k] = 2 Σ_{n=0}^{N-1} x[n] sin(π(2n+1)(2k+1)/(4N))

Inverse transforms satisfy:
  idct(dct(x, t), t) == x  for all types t ∈ {1,2,3,4}
  idst(dst(x, t), t) == x  for all types t ∈ {1,2,3,4}

Multi-axis helpers ``dctn`` / ``idctn`` / ``dstn`` / ``idstn`` apply the
corresponding 1-D transform along each requested axis in sequence.

All functions accept JAX arrays and are compatible with ``jax.jit``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sl(x: Array, start: int, stop: int, axis: int) -> Array:
    """Take a contiguous slice [start:stop] along *axis*."""
    idx = [slice(None)] * x.ndim
    idx[axis] = slice(start, stop)
    return x[tuple(idx)]


def _phase_shape(ndim: int, axis: int, size: int) -> tuple[int, ...]:
    """Return a shape broadcastable with an array of *ndim* dims along *axis*."""
    s = [1] * ndim
    s[axis] = size
    return tuple(s)


def _norm_axis(axis: int, ndim: int) -> int:
    """Normalise a (possibly negative) axis index."""
    if axis < 0:
        axis = axis + ndim
    return axis


# ---------------------------------------------------------------------------
# DCT types I–IV  (unnormalized, scipy-compatible)
# ---------------------------------------------------------------------------


def _dct1(x: Array, axis: int) -> Array:
    """Unnormalized DCT Type I via rfft (even-extension trick).

    Even-extension of length 2*(N-1):
        v = [x[0], x[1], ..., x[N-1], x[N-2], ..., x[1]]
    DCT-I[k] = Re( rfft(v)[k] )
    """
    N = x.shape[axis]
    # v = [x, x[1:-1 reversed]]  shape: 2*(N-1) along axis
    interior = _sl(x, 1, N - 1, axis)  # x[1:-1]
    v = jnp.concatenate([x, jnp.flip(interior, axis=axis)], axis=axis)
    V = jnp.fft.rfft(v, axis=axis)
    return jnp.real(_sl(V, 0, N, axis))


def _dct2(x: Array, axis: int) -> Array:
    """Unnormalized DCT Type II via Makhoul algorithm (N-point FFT).

    Reorder: v = [x[0::2], x[1::2][::-1]]
    DCT-II[k] = Re( 2·exp(−iπk/(2N)) · FFT(v)[k] )
    """
    N = x.shape[axis]
    # even-indexed then reversed odd-indexed
    even = _sl(x, 0, None, axis)[..., 0::1] if axis == x.ndim - 1 else None
    # Use raw slicing via index objects for arbitrary axis
    idx_even = [slice(None)] * x.ndim
    idx_even[axis] = slice(0, None, 2)
    idx_odd = [slice(None)] * x.ndim
    idx_odd[axis] = slice(1, None, 2)
    even = x[tuple(idx_even)]
    odd_rev = jnp.flip(x[tuple(idx_odd)], axis=axis)
    v = jnp.concatenate([even, odd_rev], axis=axis)
    V = jnp.fft.fft(v, axis=axis)
    k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    phase = 2.0 * jnp.exp(-1j * jnp.pi * k / (2 * N))
    return jnp.real(V * phase)


def _dct3(x: Array, axis: int) -> Array:
    """Unnormalized DCT Type III via irfft.

    Form B[k] = x[k]·exp(iπk/(2N)), B[N]=0 (Hermitian one-sided spectrum).
    DCT-III[n] = 2N · irfft([B[0], ..., B[N-1], 0], n=2N)[n]

    Note: DCT-III is the unnormalized inverse of DCT-II.
    """
    N = x.shape[axis]
    k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    phase = jnp.exp(1j * jnp.pi * k / (2 * N))
    b = x.astype(jnp.result_type(x.dtype, jnp.complex64)) * phase
    # Pad with one zero to form the N+1-length one-sided spectrum
    zero_shape = list(b.shape)
    zero_shape[axis] = 1
    zeros = jnp.zeros(zero_shape, dtype=b.dtype)
    B = jnp.concatenate([b, zeros], axis=axis)  # length N+1
    out_full = jnp.fft.irfft(B, n=2 * N, axis=axis)  # length 2N
    return 2.0 * N * _sl(out_full, 0, N, axis)


def _dct4(x: Array, axis: int) -> Array:
    """Unnormalized DCT Type IV via zero-padded IFFT.

    w[n] = x[n]·exp(iπ(2n+1)/(4N))
    W_pad = [w[0], ..., w[N-1], 0, ..., 0]  (length 2N)
    A[k]  = 2N · IFFT(W_pad)[k] · exp(iπk/(2N))
    DCT-IV[k] = 2·Re(A[k])
    """
    N = x.shape[axis]
    n = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    w = x.astype(jnp.result_type(x.dtype, jnp.complex64)) * jnp.exp(
        1j * jnp.pi * (2 * n + 1) / (4 * N)
    )
    zero_shape = list(w.shape)
    zero_shape[axis] = N
    zeros = jnp.zeros(zero_shape, dtype=w.dtype)
    W_pad = jnp.concatenate([w, zeros], axis=axis)  # length 2N
    # jnp.fft.ifft divides by 2N, so ifft(W_pad) * 2N reverses that
    A = 2.0 * N * jnp.fft.ifft(W_pad, axis=axis)
    A = _sl(A, 0, N, axis) * jnp.exp(1j * jnp.pi * k / (2 * N))
    return 2.0 * jnp.real(A)


# ---------------------------------------------------------------------------
# DST types I–IV  (unnormalized, scipy-compatible)
# ---------------------------------------------------------------------------


def _dst1(x: Array, axis: int) -> Array:
    """Unnormalized DST Type I via rfft (odd-extension trick).

    Odd-antisymmetric extension of length 2*(N+1):
        v = [0, x[0], ..., x[N-1], 0, −x[N-1], ..., −x[0]]
    DST-I[k] = −Im( rfft(v)[k+1] )
    """
    N = x.shape[axis]
    zero_shape = list(x.shape)
    zero_shape[axis] = 1
    zeros = jnp.zeros(zero_shape, dtype=x.dtype)
    # v = [0, x, 0, -x_reversed]
    v = jnp.concatenate([zeros, x, zeros, -jnp.flip(x, axis=axis)], axis=axis)
    V = jnp.fft.rfft(v, axis=axis)  # length N+2
    return -jnp.imag(_sl(V, 1, N + 1, axis))


def _dst2(x: Array, axis: int) -> Array:
    """Unnormalized DST Type II via rfft.

    F = rfft(x, n=2N)
    DST-II[k] = 2·Im( exp(iπ(k+1)/(2N)) · conj(F[k+1]) )
    """
    N = x.shape[axis]
    F = jnp.fft.rfft(x, n=2 * N, axis=axis)  # length N+1
    k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    phase = jnp.exp(1j * jnp.pi * (k + 1) / (2 * N))
    F_slice = _sl(F, 1, N + 1, axis)  # F[1], ..., F[N]
    return 2.0 * jnp.imag(phase * jnp.conj(F_slice))


def _dst3(x: Array, axis: int) -> Array:
    """Unnormalized DST Type III via the DCT-III / reversal identity.

    Let z[n] = (−1)^n · x[N−1−n], then
        DST-III(x)[k] = (−1)^k · DCT-III(z)[N−1−k]

    Note: DST-III is the unnormalized inverse of DST-II.
    """
    N = x.shape[axis]
    n = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    # z[n] = (-1)^n * x[N-1-n]  (reversal + alternating signs)
    z = (-1.0) ** n * jnp.flip(x, axis=axis)
    dct3_z = _dct3(z, axis)
    return (-1.0) ** k * jnp.flip(dct3_z, axis=axis)


def _dst4(x: Array, axis: int) -> Array:
    """Unnormalized DST Type IV via zero-padded IFFT.

    Uses the same intermediate array A as DCT-IV:
        DST-IV[k] = 2·Im(A[k])   (A defined in ``_dct4``)
    """
    N = x.shape[axis]
    n = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
    w = x.astype(jnp.result_type(x.dtype, jnp.complex64)) * jnp.exp(
        1j * jnp.pi * (2 * n + 1) / (4 * N)
    )
    zero_shape = list(w.shape)
    zero_shape[axis] = N
    zeros = jnp.zeros(zero_shape, dtype=w.dtype)
    W_pad = jnp.concatenate([w, zeros], axis=axis)
    A = 2.0 * N * jnp.fft.ifft(W_pad, axis=axis)
    A = _sl(A, 0, N, axis) * jnp.exp(1j * jnp.pi * k / (2 * N))
    return 2.0 * jnp.imag(A)


# ---------------------------------------------------------------------------
# Public single-axis transforms
# ---------------------------------------------------------------------------

_DCT_IMPLS = {1: _dct1, 2: _dct2, 3: _dct3, 4: _dct4}
_DST_IMPLS = {1: _dst1, 2: _dst2, 3: _dst3, 4: _dst4}


def dct(
    x: Float[Array, "..."],
    type: Literal[1, 2, 3, 4] = 2,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Unnormalized Discrete Cosine Transform (scipy norm=None convention).

    Parameters
    ----------
    x : Float[Array, "..."]
        Input array. The transform is applied along *axis*.
    type : {1, 2, 3, 4}
        DCT variant.  Default: 2.
    axis : int
        Axis along which the transform is computed.  Default: −1.

    Returns
    -------
    Float[Array, "..."]
        DCT of *x*, same shape as input.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx._src.spectral_transforms import dct, idct
    >>> x = jnp.array([1.0, 2.0, 3.0, 4.0])
    >>> y = dct(x, type=2)
    >>> jnp.allclose(idct(y, type=2), x, atol=1e-5)
    Array(True, dtype=bool)
    """
    axis = _norm_axis(axis, x.ndim)
    if type not in _DCT_IMPLS:
        raise ValueError(f"DCT type must be 1, 2, 3, or 4; got {type}")
    return _DCT_IMPLS[type](x, axis)


def idct(
    x: Float[Array, "..."],
    type: Literal[1, 2, 3, 4] = 2,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Unnormalized Inverse Discrete Cosine Transform.

    Satisfies ``idct(dct(x, t), t) == x`` for all types t ∈ {1, 2, 3, 4}.

    Inverse relationships (unnormalized forward transforms):

    * IDCT-I  = DCT-I  / (2*(N−1))
    * IDCT-II = DCT-III / (2*N)      [DCT-III is transpose of DCT-II]
    * IDCT-III = DCT-II / (2*N)
    * IDCT-IV = DCT-IV / (2*N)       [DCT-IV is its own inverse]

    Parameters
    ----------
    x : Float[Array, "..."]
        DCT-transformed array.
    type : {1, 2, 3, 4}
        DCT variant of the *forward* transform to invert.
    axis : int
        Axis along which the inverse is computed.

    Returns
    -------
    Float[Array, "..."]
        Reconstructed signal, same shape as *x*.
    """
    axis = _norm_axis(axis, x.ndim)
    N = x.shape[axis]
    if type == 1:
        # DCT-I is its own inverse up to 2*(N-1) scaling
        return _dct1(x, axis) / (2 * (N - 1))
    if type == 2:
        # IDCT-II = DCT-III / (2N), but _dct3 already includes the 2N factor,
        # so we use the irfft form directly (= _dct3(x) / (2N))
        k = jnp.arange(N).reshape(_phase_shape(x.ndim, axis, N))
        phase = jnp.exp(1j * jnp.pi * k / (2 * N))
        b = x.astype(jnp.result_type(x.dtype, jnp.complex64)) * phase
        zero_shape = list(b.shape)
        zero_shape[axis] = 1
        zeros = jnp.zeros(zero_shape, dtype=b.dtype)
        B = jnp.concatenate([b, zeros], axis=axis)
        out_full = jnp.fft.irfft(B, n=2 * N, axis=axis)
        return _sl(out_full, 0, N, axis)
    if type == 3:
        # IDCT-III = DCT-II / (2N)
        return _dct2(x, axis) / (2 * N)
    if type == 4:
        # DCT-IV is its own inverse up to 2N scaling
        return _dct4(x, axis) / (2 * N)
    raise ValueError(f"DCT type must be 1, 2, 3, or 4; got {type}")


def dst(
    x: Float[Array, "..."],
    type: Literal[1, 2, 3, 4] = 1,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Unnormalized Discrete Sine Transform (scipy norm=None convention).

    Parameters
    ----------
    x : Float[Array, "..."]
        Input array.
    type : {1, 2, 3, 4}
        DST variant.  Default: 1.
    axis : int
        Axis along which the transform is computed.

    Returns
    -------
    Float[Array, "..."]
        DST of *x*, same shape as input.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx._src.spectral_transforms import dst, idst
    >>> x = jnp.array([1.0, 2.0, 3.0, 4.0])
    >>> y = dst(x, type=1)
    >>> jnp.allclose(idst(y, type=1), x, atol=1e-5)
    Array(True, dtype=bool)
    """
    axis = _norm_axis(axis, x.ndim)
    if type not in _DST_IMPLS:
        raise ValueError(f"DST type must be 1, 2, 3, or 4; got {type}")
    return _DST_IMPLS[type](x, axis)


def idst(
    x: Float[Array, "..."],
    type: Literal[1, 2, 3, 4] = 1,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Unnormalized Inverse Discrete Sine Transform.

    Satisfies ``idst(dst(x, t), t) == x`` for all types t ∈ {1, 2, 3, 4}.

    Inverse relationships (unnormalized forward transforms):

    * IDST-I  = DST-I  / (2*(N+1))
    * IDST-II = DST-III / (2*N)      [DST-III is transpose of DST-II]
    * IDST-III = DST-II / (2*N)
    * IDST-IV = DST-IV / (2*N)       [DST-IV is its own inverse]

    Parameters
    ----------
    x : Float[Array, "..."]
        DST-transformed array.
    type : {1, 2, 3, 4}
        DST variant of the *forward* transform to invert.
    axis : int
        Axis along which the inverse is computed.

    Returns
    -------
    Float[Array, "..."]
        Reconstructed signal, same shape as *x*.
    """
    axis = _norm_axis(axis, x.ndim)
    N = x.shape[axis]
    if type == 1:
        # DST-I is its own inverse up to 2*(N+1) scaling
        return _dst1(x, axis) / (2 * (N + 1))
    if type == 2:
        # IDST-II = DST-III / (2N)
        return _dst3(x, axis) / (2 * N)
    if type == 3:
        # IDST-III = DST-II / (2N)
        return _dst2(x, axis) / (2 * N)
    if type == 4:
        # DST-IV is its own inverse up to 2N scaling
        return _dst4(x, axis) / (2 * N)
    raise ValueError(f"DST type must be 1, 2, 3, or 4; got {type}")


# ---------------------------------------------------------------------------
# Multi-axis helpers
# ---------------------------------------------------------------------------


def dctn(
    x: Float[Array, "..."],
    type: Literal[1, 2, 3, 4] = 2,
    axes: Sequence[int] | None = None,
) -> Float[Array, "..."]:
    """N-dimensional DCT: apply ``dct`` sequentially along each axis.

    Parameters
    ----------
    x : Float[Array, "..."]
        Input array.
    type : {1, 2, 3, 4}
        DCT variant.  Default: 2.
    axes : sequence of int or None
        Axes to transform.  ``None`` transforms all axes.

    Returns
    -------
    Float[Array, "..."]
        N-D DCT of *x*, same shape as input.
    """
    if axes is None:
        axes = list(range(x.ndim))
    y = x
    for ax in axes:
        y = dct(y, type=type, axis=ax)
    return y


def idctn(
    x: Float[Array, "..."],
    type: Literal[1, 2, 3, 4] = 2,
    axes: Sequence[int] | None = None,
) -> Float[Array, "..."]:
    """N-dimensional inverse DCT: apply ``idct`` sequentially along each axis.

    Parameters
    ----------
    x : Float[Array, "..."]
        DCT-transformed array.
    type : {1, 2, 3, 4}
        DCT variant of the *forward* transform to invert.
    axes : sequence of int or None
        Axes to inverse-transform.  ``None`` transforms all axes.

    Returns
    -------
    Float[Array, "..."]
        Reconstructed N-D array, same shape as input.
    """
    if axes is None:
        axes = list(range(x.ndim))
    y = x
    for ax in axes:
        y = idct(y, type=type, axis=ax)
    return y


def dstn(
    x: Float[Array, "..."],
    type: Literal[1, 2, 3, 4] = 1,
    axes: Sequence[int] | None = None,
) -> Float[Array, "..."]:
    """N-dimensional DST: apply ``dst`` sequentially along each axis.

    Parameters
    ----------
    x : Float[Array, "..."]
        Input array.
    type : {1, 2, 3, 4}
        DST variant.  Default: 1.
    axes : sequence of int or None
        Axes to transform.  ``None`` transforms all axes.

    Returns
    -------
    Float[Array, "..."]
        N-D DST of *x*, same shape as input.
    """
    if axes is None:
        axes = list(range(x.ndim))
    y = x
    for ax in axes:
        y = dst(y, type=type, axis=ax)
    return y


def idstn(
    x: Float[Array, "..."],
    type: Literal[1, 2, 3, 4] = 1,
    axes: Sequence[int] | None = None,
) -> Float[Array, "..."]:
    """N-dimensional inverse DST: apply ``idst`` sequentially along each axis.

    Parameters
    ----------
    x : Float[Array, "..."]
        DST-transformed array.
    type : {1, 2, 3, 4}
        DST variant of the *forward* transform to invert.
    axes : sequence of int or None
        Axes to inverse-transform.  ``None`` transforms all axes.

    Returns
    -------
    Float[Array, "..."]
        Reconstructed N-D array, same shape as input.
    """
    if axes is None:
        axes = list(range(x.ndim))
    y = x
    for ax in axes:
        y = idst(y, type=type, axis=ax)
    return y
