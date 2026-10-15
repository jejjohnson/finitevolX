"""Accuracy benchmark of the finitevolX advection schemes.

Regenerates the measured tables in ``docs/advection.md``:

* ``exactness`` -- the highest polynomial degree ``d`` each 1-D face
  reconstruction reproduces exactly, fed **cell averages** of ``x^d`` (the
  finite-volume reading) and, for contrast, **point values** ``x_i^d``;
* ``convergence`` -- observed L∞ / L1 orders of the 1-D flux-form tendency of
  ``A sin(2πx)`` (u = 1, periodic) for every method on cell averages and on
  point values, and of weno5 / van_leer at amplitudes ``A = 1e-9, 1, 1e3``;
* ``order2d`` -- observed orders of the unsplit 2-D tendency (1-D face-point
  reconstructions) with a constant and with a spatially varying velocity;
* ``benchmark`` -- solid-body rotation of a Gaussian, one revolution, SSP-RK3,
  CFL 0.25, relative L2 error, min / max and wall time per step for every
  method.

Usage::

    uv run python scripts/benchmark_advection.py            # every part
    uv run python scripts/benchmark_advection.py --part benchmark --n 64

It is a developer tool: nothing in the test suite imports it.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.special import erf

import finitevolx as fvx

METHODS = (
    "naive",
    "upwind1",
    "upwind2",
    "upwind3",
    "weno3",
    "weno5",
    "wenoz5",
    "weno7",
    "weno9",
    "minmod",
    "van_leer",
    "superbee",
    "mc",
)
TVD = ("minmod", "van_leer", "superbee", "mc")

# Periodic halo (cells on each side, ghost ring included) wide enough for the
# widest stencil (weno9: 5) plus the closed-wall write offset.
HALO = 6

# Gauss-Legendre nodes / weights on [-1/2, 1/2] for exact cell / face averages
# of the smooth test functions (8 points: exact to degree 15).
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_GL_X, _GL_W = 0.5 * _GL_X, 0.5 * _GL_W


def _face_value_1d(recon: fvx.Reconstruction1D, method: str, q, sign: float):
    """Reconstructed east-face value q̂[i+1/2] (flux / u with u = ±1)."""
    u = jnp.full_like(q, sign)
    if method in TVD:
        flux = recon.tvd_x(q, u, limiter=method)
    else:
        flux = getattr(recon, f"{method}_x")(q, u)
    return flux / u  # [Nx]


# --------------------------------------------------------------------------
# 1. Exactness degree
# --------------------------------------------------------------------------


def exactness(max_degree: int = 8) -> dict[str, tuple[int, int]]:
    """Largest d with q̂ = x^d exact at every probed face, for both flow signs.

    Returns ``{method: (d_cell_averages, d_point_values)}``.
    """
    n = 40
    dx = 1.0 / 8.0  # O(1) polynomial values, so 1e-9 is a clean threshold
    grid = fvx.CartesianGrid1D.from_interior(n, n * dx)
    recon = fvx.Reconstruction1D(grid=grid)
    k = np.arange(n + 2)
    xl = (k - 1 - n / 2) * dx  # west face of cell k, centred on 0
    xc = xl + 0.5 * dx
    xf = xl + dx  # east face of cell k (the U-point of index k)
    probe = slice(HALO, -HALO)

    out = {}
    for m in METHODS:
        degs = []
        for kind in ("avg", "point"):
            d_ok = -1
            for d in range(max_degree + 1):
                if kind == "avg":
                    q = ((xl + dx) ** (d + 1) - xl ** (d + 1)) / ((d + 1) * dx)
                else:
                    q = xc**d
                q = jnp.asarray(q)
                err = 0.0
                for sign in (1.0, -1.0):
                    face = np.asarray(_face_value_1d(recon, m, q, sign))
                    err = max(err, np.max(np.abs(face[probe] - xf[probe] ** d)))
                scale = max(1.0, np.max(np.abs(xf[probe] ** d)))
                if err > 1e-9 * scale:
                    break
                d_ok = d
            degs.append(d_ok)
        out[m] = (degs[0], degs[1])
    return out


# --------------------------------------------------------------------------
# 2. 1-D convergence
# --------------------------------------------------------------------------


def _tendency_errors_1d(
    method: str, n: int, amp: float, kind: str, profile: str, variable: bool = False
):
    """L∞, L1 error of the 1-D tendency, u = 1, periodic on [0, 1].

    ``profile="sin"``: q = A sin(2πx) (one Fourier mode: every linear scheme
    sees cell averages and point values alike, up to a constant factor).
    ``profile="expsin"``: q = exp(sin(2πx)) (all modes), the case that
    separates the cell-average and point-value readings.
    ``kind="avg"``: input q̄_i = (1/dx) ∫ q dx, compared with the exact
    d q̄_i/dt = -(q(x_{i+½}) - q(x_{i-½})) / dx.  ``kind="point"``: input
    q(x_i), compared with -q'(x_i).  ``variable``: u(x) = 1 + ½ sin(2πx)
    sampled at the faces (the U-points) instead of u = 1.
    """
    ntot = n + 2 * HALO
    dx = 1.0 / n
    grid = fvx.CartesianGrid1D.from_interior(ntot - 2, (ntot - 2) * dx)
    adv = fvx.Advection1D(grid=grid)
    k = np.arange(ntot)
    xl = (k - HALO) * dx  # west face of cell k
    xc = xl + 0.5 * dx
    w = 2.0 * np.pi
    if profile == "sin":

        def f(x):
            return amp * np.sin(w * x)

        def df(x):
            return amp * w * np.cos(w * x)
    else:

        def f(x):
            return amp * np.exp(np.sin(w * x))

        def df(x):
            return amp * w * np.cos(w * x) * np.exp(np.sin(w * x))

    def uf(x):
        return 1.0 + 0.5 * np.sin(w * x) if variable else np.ones_like(x)

    def duf(x):
        return 0.5 * w * np.cos(w * x) if variable else np.zeros_like(x)

    if kind == "avg":
        q = f(xl[:, None] + dx * (0.5 + _GL_X[None, :])) @ _GL_W  # [ntot]
        # d q̄_i/dt = -((u q)(x_{i+½}) - (u q)(x_{i-½})) / dx
        exact = -(uf(xl + dx) * f(xl + dx) - uf(xl) * f(xl)) / dx
    else:
        q = f(xc)
        # dq/dt(x_i) = -(u q)'(x_i)
        exact = -(duf(xc) * f(xc) + uf(xc) * df(xc))
    u = jnp.asarray(uf(xl + dx))  # U-point k sits on the east face of cell k
    t = np.asarray(adv(jnp.asarray(q), u, method=method))
    e = np.abs(t - exact)[HALO:-HALO] / (amp * w)
    return e.max(), e.mean()


def _rates(errs: list[float], ns: list[int]) -> list[float]:
    return [
        math.log(errs[i] / errs[i + 1]) / math.log(ns[i + 1] / ns[i])
        for i in range(len(ns) - 1)
    ]


def convergence(ns=(16, 32, 64, 128, 256)) -> None:
    ns = list(ns)
    print("\n## 1-D tendency of exp(sin(2πx)), periodic: observed orders")
    print(f"N = {ns}; L∞ / L1 rate of the last pair")
    cols = [(k, v) for v in (False, True) for k in ("avg", "point")]
    hdr = " ".join(f"{('u(x) ' if v else 'u=1 ') + k:>14s}" for k, v in cols)
    print(f"{'method':10s} {hdr}   L∞ rates, u=1 avg, per refinement")
    for m in METHODS:
        row, full = [], ""
        for kind, variable in cols:
            ei = [_tendency_errors_1d(m, n, 1.0, kind, "expsin", variable) for n in ns]
            ri = _rates([e[0] for e in ei], ns)
            r1 = _rates([e[1] for e in ei], ns)
            row.append(f"{ri[-1]:5.2f} / {r1[-1]:5.2f}")
            if kind == "avg" and not variable:
                full = " ".join(f"{r:5.2f}" for r in ri)
        print(f"{m:10s} " + " ".join(f"{c:>14s}" for c in row) + f"   {full}")

    print("\n## amplitude dependence: A sin(2πx) cell averages, N = 32 ... 512")
    ns2 = [32, 64, 128, 256, 512]
    for m in ("weno5", "van_leer"):
        for amp in (1e-9, 1.0, 1e3):
            ei = [_tendency_errors_1d(m, n, amp, "avg", "sin") for n in ns2]
            ri = " ".join(f"{r:5.2f}" for r in _rates([e[0] for e in ei], ns2))
            r1 = " ".join(f"{r:5.2f}" for r in _rates([e[1] for e in ei], ns2))
            print(f"{m:9s} A={amp:7.0e}  L∞ {ri}   L1 {r1}")


# --------------------------------------------------------------------------
# 3. 2-D order (unsplit, 1-D face-point reconstructions)
# --------------------------------------------------------------------------


def _q2d(x, y):
    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + 0.5 * np.sin(
        2 * np.pi * (x + y)
    )


def _tendency_error_2d(method: str, n: int, variable: bool) -> float:
    """Max error of the 2-D cell-average tendency against the exact one."""
    ntot = n + 2 * HALO
    d = 1.0 / n
    grid = fvx.CartesianGrid2D.from_interior(
        ntot - 2, ntot - 2, (ntot - 2) * d, (ntot - 2) * d
    )
    adv = fvx.Advection2D(grid=grid)
    k = np.arange(ntot)
    lo = (k - HALO) * d  # west / south edge of cell k
    if variable:
        # u = u(y), v = v(x): divergence-free, varying along each face
        def uf(y):
            return 1.0 + 0.5 * np.sin(2 * np.pi * y)

        def vf(x):
            return 0.5 + 0.25 * np.cos(2 * np.pi * x)
    else:

        def uf(y):
            return np.ones_like(y)

        def vf(x):
            return 0.5 * np.ones_like(x)

    # Cell averages q̄[j, i] (8x8 Gauss-Legendre) and face-averaged fluxes
    #   Fx[j, i+½] = (1/dy) ∫ u(y) q(x_{i+½}, y) dy,  Fy[j+½, i] likewise.
    xg = lo[:, None] + d * (0.5 + _GL_X[None, :])  # [N, G]
    q = np.einsum(
        "a,b,jbia->ji",
        _GL_W,
        _GL_W,
        _q2d(xg[None, None, :, :], xg[:, :, None, None]),
    )
    xe = lo + d  # east face x / north face y of cell k
    fx = np.einsum(
        "b,jb,jbi->ji", _GL_W, uf(xg), _q2d(xe[None, None, :], xg[:, :, None])
    )
    fy = np.einsum(
        "a,ia,jia->ji", _GL_W, vf(xg), _q2d(xg[None, :, :], xe[:, None, None])
    )
    # d q̄[j, i]/dt = -(Fx[j, i+½] - Fx[j, i-½]) / dx - (Fy[j+½, i] - Fy[j-½, i]) / dy
    exact = np.zeros_like(q)  # [ntot, ntot]
    exact[1:, 1:] = -((fx[1:, 1:] - fx[1:, :-1]) + (fy[1:, 1:] - fy[:-1, 1:])) / d
    # The C-grid velocity is the face-midpoint value.
    yc = lo + 0.5 * d
    u = np.broadcast_to(uf(yc)[:, None], (ntot, ntot))
    v = np.broadcast_to(vf(yc)[None, :], (ntot, ntot))
    t = np.asarray(adv(jnp.asarray(q), jnp.asarray(u), jnp.asarray(v), method=method))
    s = slice(HALO, -HALO)
    return float(np.max(np.abs(t[s, s] - exact[s, s])))


def order2d(ns=(16, 32, 64, 128)) -> None:
    ns = list(ns)
    print("\n## 2-D cell-average tendency: observed L∞ orders, N =", ns)
    for m in ("upwind3", "weno5", "weno9"):
        for variable in (False, True):
            errs = [_tendency_error_2d(m, n, variable) for n in ns]
            r = " ".join(f"{x:5.2f}" for x in _rates(errs, ns))
            lab = "variable u(y), v(x)" if variable else "constant u, v     "
            print(f"{m:8s} {lab}  rates {r}   (err N={ns[-1]}: {errs[-1]:.1e})")


# --------------------------------------------------------------------------
# 4. Solid-body rotation of a Gaussian
# --------------------------------------------------------------------------


def _gauss_avg(lo, d, c, sig):
    """(1/d) ∫_lo^{lo+d} exp(-(x-c)²/(2σ²)) dx."""
    s = np.sqrt(2.0) * sig
    return 0.5 * np.sqrt(np.pi) * s * (erf((lo + d - c) / s) - erf((lo - c) / s)) / d


def benchmark(n: int = 64, cfl: float = 0.25, sigma: float = 0.05) -> None:
    """One revolution of q0 = exp(-r²/2σ²) about (½, ½), centred at (½, ¾)."""
    grid = fvx.CartesianGrid2D.from_interior(n, n, 1.0, 1.0)
    d = 1.0 / n
    k = np.arange(n + 2)
    lo = (k - 1) * d  # west / south edge of cell k (cell 1 starts at 0)
    q0 = (
        _gauss_avg(lo, d, 0.75, sigma)[:, None] * _gauss_avg(lo, d, 0.5, sigma)[None, :]
    )
    # u[j, i] at (x = i d, y = (j - ½) d), v[j, i] at (x = (i - ½) d, y = j d)
    yc = lo + 0.5 * d
    u = np.broadcast_to((-2 * np.pi * (yc - 0.5))[:, None], (n + 2, n + 2))
    v = np.broadcast_to((2 * np.pi * (yc - 0.5))[None, :], (n + 2, n + 2))
    umax = np.pi  # max |u|, |v| on the domain; period 1
    nsteps = math.ceil(1.0 / (cfl * d / umax))
    dt = 1.0 / nsteps
    q0j, uj, vj = jnp.asarray(q0), jnp.asarray(u), jnp.asarray(v)
    adv = fvx.Advection2D(grid=grid)

    print(
        f"\n## solid-body Gaussian, N = {n}, sigma = {sigma}, CFL {cfl}, "
        f"{nsteps} SSP-RK3 steps (one revolution)"
    )
    print(f"{'method':10s} {'rel-L2':>8s} {'min':>9s} {'max':>7s} {'ms 256²':>8s}")
    cost = _forward_cost()
    for m in METHODS:

        def rhs(q, _m=m):
            return adv(q, uj, vj, method=_m)

        @jax.jit
        def run(q):
            return jax.lax.fori_loop(
                0, nsteps, lambda _, s: fvx.rk3_ssp_step(s, rhs, dt), q
            )

        q1 = np.asarray(run(q0j))
        e = q1[1:-1, 1:-1] - q0[1:-1, 1:-1]
        rel = np.sqrt(np.sum(e**2) / np.sum(q0[1:-1, 1:-1] ** 2))
        print(
            f"{m:10s} {rel:8.1e} {q1[1:-1, 1:-1].min():9.1e} "
            f"{q1[1:-1, 1:-1].max():7.3f} {cost[m]:8.2f}"
        )
    print(
        f"(min / max over the interior; initial {q0.min():.1e} / {q0.max():.3f}. "
        "ms 256²: fastest of 50 jitted forward tendencies on a 256² grid)"
    )


def _forward_cost(n: int = 256, repeats: int = 50) -> dict[str, float]:
    """Fastest wall time (ms) of the jitted 2-D tendency."""
    grid = fvx.CartesianGrid2D.from_interior(n, n, 1.0, 1.0)
    adv = fvx.Advection2D(grid=grid)
    q = jax.random.normal(jax.random.PRNGKey(0), (n + 2, n + 2))
    u = jax.random.normal(jax.random.PRNGKey(1), (n + 2, n + 2))
    v = jax.random.normal(jax.random.PRNGKey(2), (n + 2, n + 2))
    times = {}
    for m in METHODS:
        f = jax.jit(lambda q, u, v, _m=m: adv(q, u, v, method=_m))
        f(q, u, v).block_until_ready()
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            f(q, u, v).block_until_ready()
            ts.append(time.perf_counter() - t0)
        times[m] = min(ts)  # the least-disturbed run (shared machines)
    return {m: 1e3 * t for m, t in times.items()}


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return f"{line.split(':', 1)[1].strip()} x {os.cpu_count()}"
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--part",
        choices=("all", "exactness", "convergence", "order2d", "benchmark"),
        default="all",
    )
    p.add_argument("--n", type=int, default=64, help="benchmark resolution")
    args = p.parse_args()
    print(
        f"# {_cpu_name()}, jax {jax.__version__} ({jax.devices()[0].platform}), float64"
    )
    if args.part in ("all", "exactness"):
        print("\n## exactness degree of the face value (x^d), both flow signs")
        print(f"{'method':10s} {'cell averages':>14s} {'point values':>13s}")
        for m, (da, dp) in exactness().items():
            print(f"{m:10s} {da:14d} {dp:13d}")
    if args.part in ("all", "convergence"):
        convergence()
    if args.part in ("all", "order2d"):
        order2d()
    if args.part in ("all", "benchmark"):
        benchmark(n=args.n)


if __name__ == "__main__":
    main()
