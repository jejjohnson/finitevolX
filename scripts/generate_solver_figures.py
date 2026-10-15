"""Benchmark the elliptic solvers and generate the solver-comparison figures.

Solves the streamfunction Helmholtz problem

    (∇² − λ) ψ = f   on the wet X points of Mask2D.xy_corner_strict,
    ψ = 0             on the dry wall / coast X nodes,

(the problem every finitevolX method solves, see docs/elliptic_solvers.md)
on two basins, a rectangle and a circular basin with an island, at
128² and 256² interior cells and λ ∈ {4, 0}, with:

* spectral DST (``streamfunction_from_vorticity(bc="dst")``, rectangle only),
* capacitance (``build_capacitance_solver``, DST base),
* CG (``solve_cg`` on ``masked_laplacian``) with a DST, FFT or multigrid
  preconditioner or none, to ``max|f − Aψ| <= 2e-8 max|f|``,
* standalone multigrid (5 V-cycles, the default, and 8 V-cycles),
* SciPy sparse LU (SuperLU) as the reference and as the "sparse direct"
  data point (CPU only, not JAX: no ``jit`` / ``grad``).

For each it reports build time (min of 2 builds), jitted solve time (minimum over repeats),
the time of ``jax.grad`` of ``½ Σ ψ²`` with respect to ``f``, iterations
or V-cycles, the bytes of precomputed arrays the solver holds, and the max
error relative to the sparse-LU solution.  Iteration counts and errors are
deterministic; timings are not, hence the minimum over several repeats.

Outputs: a Markdown table on stdout (copied into docs/solver_comparison.md)
and PNG figures in docs/images/solver_comparison/.

Usage:
    uv run python scripts/generate_solver_figures.py            # 128², 256²
    uv run python scripts/generate_solver_figures.py --sizes 32 # smoke run
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import finitevolx as fvx

jax.config.update("jax_enable_x64", True)

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "solver_comparison"

# CG stopping rule |r_i| <= ATOL + RTOL |f_i| with ATOL = RTOL max|f|, so
# max|f - A psi| <= 2e-8 max|f|.  (atol = 0 does not work with a masked rhs:
# lineax divides r_i by rtol |f_i| = 0 on dry points, gets NaN and stops.)
RTOL = 1e-8
MAX_STEPS = 20_000


# ── Geometry ────────────────────────────────────────────────────────────
def make_case(geometry: str, n: int) -> dict:
    """Grid spacing, Mask2D, psi mask [Ny, Nx] and rhs [Ny, Nx] on the unit square."""
    grid = fvx.CartesianGrid2D.from_interior(n, n, 1.0, 1.0)
    dx, dy = float(grid.dx), float(grid.dy)
    ny, nx = n + 2, n + 2
    # T-point centres: x[i] = (i - 1/2) dx, i = 0 .. Nx-1  (ghost ring at 0, N-1)
    x = (np.arange(nx) - 0.5) * dx
    y = (np.arange(ny) - 0.5) * dy
    X, Y = np.meshgrid(x, y)  # [Ny, Nx]
    h = np.zeros((ny, nx), dtype=bool)
    h[1:-1, 1:-1] = True
    if geometry == "island":
        # circular basin of radius 0.45 with a disc island of radius 0.1
        r_basin = np.hypot(X - 0.5, Y - 0.5)
        r_island = np.hypot(X - 0.62, Y - 0.55)
        h &= (r_basin < 0.45) & (r_island > 0.1)
    mask = fvx.Mask2D.from_mask(h)
    wet = np.asarray(mask.xy_corner_strict, dtype=bool)  # psi (X-point) mask
    # X-point coordinates: node i sits at i + 1/2 in T-index units -> x = i dx
    Xq, Yq = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy)  # [Ny, Nx]
    f = np.sin(3 * np.pi * Xq) * np.cos(2 * np.pi * Yq) + np.exp(
        -((Xq - 0.3) ** 2 + (Yq - 0.7) ** 2) / 0.01
    )
    f = np.where(wet, f, 0.0)
    return {"dx": dx, "dy": dy, "mask": mask, "wet": wet, "f": jnp.asarray(f)}


def sparse_operator(wet: np.ndarray, dx: float, dy: float, lam: float):
    """(∇² − λ) on the wet points as a CSR matrix: dry neighbours are zero.

    (A ψ)[j, i] = (ψ[j, i+1] − 2ψ[j, i] + ψ[j, i−1]) / dx²
                + (ψ[j+1, i] − 2ψ[j, i] + ψ[j−1, i]) / dy² − λ ψ[j, i],
    the matrix of masked_laplacian restricted to the wet points.
    """
    ny, nx = wet.shape
    idx = -np.ones((ny, nx), dtype=np.int64)
    jw, iw = np.nonzero(wet)
    idx[jw, iw] = np.arange(jw.size)
    rows = [idx[jw, iw]]
    cols = [idx[jw, iw]]
    vals = [np.full(jw.size, -2.0 / dx**2 - 2.0 / dy**2 - lam)]
    for dj, di, c in ((0, 1, dx**-2), (0, -1, dx**-2), (1, 0, dy**-2), (-1, 0, dy**-2)):
        jn, in_ = jw + dj, iw + di
        ok = (jn >= 0) & (jn < ny) & (in_ >= 0) & (in_ < nx)
        ok[ok] &= wet[jn[ok], in_[ok]]
        rows.append(idx[jw[ok], iw[ok]])
        cols.append(idx[jn[ok], in_[ok]])
        vals.append(np.full(ok.sum(), c))
    n_wet = jw.size
    a = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_wet, n_wet),
    )
    return a, (jw, iw)


# ── Timing helpers ──────────────────────────────────────────────────────
def min_time(fn, *args, repeats: int, budget_s: float = 20.0) -> float:
    """Minimum wall time (s) of ``fn(*args)`` over repeats, after one warm-up call."""
    jax.block_until_ready(fn(*args))  # compile + warm up
    best = np.inf
    t_start = time.perf_counter()
    for k in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
        if k >= 2 and time.perf_counter() - t_start > budget_s:
            break
    return best


def min_build(build, repeats: int = 2):
    """Run an (eager, NumPy-heavy) builder ``repeats`` times; return (result, min s)."""
    best, out = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = jax.block_until_ready(build())
        best = min(best, time.perf_counter() - t0)
    return out, best


def array_bytes(tree) -> int:
    """Bytes of the arrays a (built) solver holds."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
    return int(sum(leaf.nbytes for leaf in leaves))


def grad_fn(solve):
    """jit(grad) of the loss L(f) = ½ Σ ψ(f)² with respect to f (argument 1)."""
    return eqx.filter_jit(
        jax.grad(lambda p, f: 0.5 * jnp.sum(solve(p, f) ** 2), argnums=1)
    )


# ── One case ────────────────────────────────────────────────────────────
def run_case(geometry: str, n: int, lam: float, repeats: int) -> tuple[list, dict]:
    case = make_case(geometry, n)
    dx, dy, mask, wet, f = case["dx"], case["dy"], case["mask"], case["wet"], case["f"]
    wet_j = jnp.asarray(wet, dtype=f.dtype)  # [Ny, Nx]
    rows: list[dict] = []
    fields: dict = {}

    # Reference: SciPy sparse LU on the wet points.
    (a, (jw, iw)), t_op = min_build(lambda: sparse_operator(wet, dx, dy, lam))
    lu, t_lu = min_build(lambda: spla.splu(a.tocsc()))
    t_build = t_op + t_lu
    f_np = np.asarray(f)
    t_solve = np.inf
    for _ in range(max(3, repeats // 2)):
        t0 = time.perf_counter()
        sol = lu.solve(f_np[jw, iw])
        t_solve = min(t_solve, time.perf_counter() - t0)
    psi_ref = np.zeros_like(f_np)
    psi_ref[jw, iw] = sol
    # The reference must solve the masked_laplacian problem (residual at round-off).
    res = f - fvx.masked_laplacian(jnp.asarray(psi_ref), wet_j, dx, dy, lambda_=lam)
    assert float(jnp.max(jnp.abs(res * wet_j))) < 1e-9 * float(jnp.max(jnp.abs(f)))
    scale = np.max(np.abs(psi_ref))
    lu_bytes = lu.L.data.nbytes + lu.U.data.nbytes
    lu_bytes += lu.L.indices.nbytes + lu.U.indices.nbytes
    rows.append(
        dict(
            method="sparse LU (SciPy)",
            build=t_build,
            solve=t_solve,
            grad=np.nan,
            iters="direct",
            bytes=lu_bytes,
            err=0.0,
        )
    )
    fields["reference"] = psi_ref

    def record(method, build, solve, iters, nbytes, params=None):
        """Time ``solve(params, f)``; ``params`` (the built solver) is a jit argument.

        Closing over a built solver instead would embed its arrays in the
        program as constants, which XLA then constant-folds: compiling the
        capacitance solve of the 256² island basin that way took 8 minutes.
        """
        p = (params, wet_j)
        solve_j = eqx.filter_jit(solve)
        psi = np.asarray(solve_j(p, f)) * wet
        err = float(np.max(np.abs(psi - psi_ref)) / scale)
        t = min_time(solve_j, p, f, repeats=repeats)
        tg = min_time(grad_fn(solve), p, f, repeats=max(3, repeats // 2))
        rows.append(
            dict(
                method=method,
                build=build,
                solve=t,
                grad=tg,
                iters=iters,
                bytes=nbytes,
                err=err,
            )
        )
        fields[method] = psi
        print(
            f"    {method:<20} build {build * 1e3:9.1f} ms  solve {t * 1e3:9.2f} ms  "
            f"grad {tg * 1e3:9.2f} ms  err {err:.1e}  {iters}",
            flush=True,
        )

    # Spectral DST (rectangle only): the wrapper solves the wet X nodes [1:-2, 1:-2].
    if geometry == "rectangle":
        record(
            "spectral DST",
            0.0,
            lambda p, r: fvx.streamfunction_from_vorticity(
                r, dx, dy, bc="dst", lambda_=lam
            ),
            "direct",
            0,
        )

    # Capacitance (DST base, the default).
    cap, t_cap = min_build(
        lambda: fvx.build_capacitance_solver(mask, dx, dy, lambda_=lam)
    )
    record(
        "capacitance",
        t_cap,
        lambda p, r: p[0](r * p[1]),
        "direct",
        array_bytes(cap),
        params=cap,
    )

    # Multigrid hierarchy (Dirichlet, the default), shared by MG and CG + MG.
    mg, t_mg = min_build(lambda: fvx.build_multigrid_solver(mask, dx, dy, lambda_=lam))
    mg_bytes = array_bytes(mg)

    atol = RTOL * float(jnp.max(jnp.abs(f)))

    def cg(p, r, kind):
        """CG on masked_laplacian (psi = 0 on the dry neighbours), [Ny, Nx]."""
        solver, w = p

        def matvec(x):
            return fvx.masked_laplacian(x, w, dx, dy, lambda_=lam)

        if kind == "multigrid":
            pc = fvx.make_multigrid_preconditioner(solver)
        elif kind in ("dst", "fft"):
            pc = fvx.make_spectral_preconditioner(dx, dy, lam, bc=kind)
        else:
            pc = None
        return fvx.solve_cg(
            matvec, r * w, preconditioner=pc, rtol=RTOL, atol=atol, max_steps=MAX_STEPS
        )

    variants = {
        "CG + DST": ("dst", None, 0.0, 0),
        "CG + FFT": ("fft", None, 0.0, 0),
        "CG + multigrid": ("multigrid", mg, t_mg, mg_bytes),
        "CG (none)": (None, None, 0.0, 0),
    }
    for name, (kind, params, t_build, nbytes) in variants.items():
        _, info = eqx.filter_jit(cg)((params, wet_j), f, kind)
        iters = f"{int(info.iterations)} it"
        if not bool(info.converged):
            iters += " (not converged)"
        record(
            name,
            t_build,
            lambda p, r, k=kind: cg(p, r, k)[0] * p[1],
            iters,
            nbytes,
            params=params,
        )

    # Standalone multigrid: 5 V-cycles (the default) and 8.
    for cycles in (5, 8):
        record(
            f"multigrid ({cycles} V)",
            t_mg,
            lambda p, r: p[0](r * p[1]) * p[1],
            f"{cycles} V",
            mg_bytes,
            params=dataclasses.replace(mg, n_cycles=cycles),
        )

    return rows, {**case, "fields": fields}


# ── Figures ─────────────────────────────────────────────────────────────
COLORS = {
    "spectral DST": "#1f77b4",
    "capacitance": "#2ca02c",
    "CG + DST": "#ff7f0e",
    "CG + FFT": "#ffbb78",
    "CG + multigrid": "#9467bd",
    "CG (none)": "#7f7f7f",
    "multigrid (5 V)": "#d62728",
    "multigrid (8 V)": "#ff9896",
    "sparse LU (SciPy)": "#17becf",
}


def plot_geometries(cases: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 4))
    for ax, (name, case) in zip(np.atleast_1d(axes), cases.items(), strict=True):
        wet = case["wet"]
        ring = np.asarray(fvx.coast_ring(jnp.asarray(wet, dtype=float)))
        img = np.where(wet, 1.0, np.where(ring, 0.5, 0.0))
        ax.imshow(
            img, origin="lower", cmap="Blues", vmin=0, vmax=1, interpolation="nearest"
        )
        n = wet.shape[0] - 2
        ax.set_title(
            f"{name} {n}²: {int(wet.sum())} wet psi nodes,\n"
            f"{int(ring.sum())} coast nodes"
        )
        ax.axis("off")
    fig.suptitle("psi (X-point) domains: wet (dark), coast ring (light)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_errors(case: dict, path: Path) -> None:
    wet = case["wet"]
    ref = case["fields"]["reference"]
    methods = ["capacitance", "CG + FFT", "CG + multigrid", "multigrid (5 V)"]
    fig, axes = plt.subplots(
        1, 1 + len(methods), figsize=(3.6 * (1 + len(methods)), 3.4)
    )
    im = axes[0].imshow(np.where(wet, ref, np.nan), origin="lower", cmap="RdBu_r")
    axes[0].set_title("psi (sparse LU)")
    fig.colorbar(im, ax=axes[0], shrink=0.75)
    scale = np.max(np.abs(ref))
    for ax, m in zip(axes[1:], methods, strict=True):
        err = np.where(wet, (case["fields"][m] - ref) / scale, np.nan)
        vmax = max(float(np.nanmax(np.abs(err))), 1e-300)
        im = ax.imshow(err, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{m}\nerror / max|psi|")
        fig.colorbar(im, ax=ax, shrink=0.75, format="%.0e")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_summary(results: dict, sizes: list[int], lam: float, path: Path) -> None:
    """Solve time and error per method, grouped by geometry and size, at one λ."""
    groups = [(g, n) for g in ("rectangle", "island") for n in sizes]
    methods = list(COLORS)
    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(15, 4.8))
    width = 0.8 / len(methods)
    for gi, key in enumerate(groups):
        rows = {r["method"]: r for r in results[(*key, lam)]}
        for mi, m in enumerate(methods):
            if m not in rows:
                continue
            x = gi + (mi - (len(methods) - 1) / 2) * width
            ax_t.bar(x, rows[m]["solve"] * 1e3, width, color=COLORS[m])
            ax_e.bar(x, max(rows[m]["err"], 1e-16), width, color=COLORS[m])
    labels = [f"{g}\n{n}²" for g, n in groups]
    for ax in (ax_t, ax_e):
        ax.set_xticks(range(len(groups)), labels)
        ax.set_yscale("log")
    ax_t.set_ylabel("solve time (ms, jitted, min of repeats)")
    ax_e.set_ylabel("max |psi - psi_LU| / max |psi_LU|")
    ax_e.set_ylim(bottom=1e-16)
    ax_t.set_title(f"Solve time, lambda = {lam:g}")
    ax_e.set_title(f"Error against sparse LU, lambda = {lam:g} (LU = 1e-16 floor)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in methods]
    fig.legend(handles, methods, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────
def fmt_ms(t: float) -> str:
    if not np.isfinite(t):
        return "—"
    if t == 0.0:
        return "—"
    ms = t * 1e3
    return f"{ms:.3g}" if ms < 1000 else f"{ms:,.0f}"


def fmt_bytes(b: int) -> str:
    return "—" if b == 0 else f"{b / 2**20:.2g} MiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[4.0, 0.0])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    results: dict = {}
    keep: dict = {}
    for lam in args.lambdas:
        for geometry in ("rectangle", "island"):
            for n in args.sizes:
                print(f"{geometry} {n}², lambda = {lam:g}", flush=True)
                rows, case = run_case(geometry, n, lam, args.repeats)
                results[(geometry, n, lam)] = rows
                if n == args.sizes[0] and lam == args.lambdas[0]:
                    keep[geometry] = case

    print(f"\njax {jax.__version__}, backend {jax.default_backend()}, float64")
    for lam in args.lambdas:
        print(f"\n### lambda = {lam:g}\n")
        print(
            "| domain | N | method | build (ms) | solve (ms) | grad (ms) "
            "| iterations | precomputed | max error |"
        )
        print("|---|---|---|--:|--:|--:|--:|--:|--:|")
        for geometry in ("rectangle", "island"):
            for n in args.sizes:
                for r in results[(geometry, n, lam)]:
                    print(
                        f"| {geometry} | {n}² | {r['method']} | {fmt_ms(r['build'])} "
                        f"| {fmt_ms(r['solve'])} | {fmt_ms(r['grad'])} | {r['iters']} "
                        f"| {fmt_bytes(r['bytes'])} | {r['err']:.1e} |"
                    )

    if args.no_figures:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    plot_geometries(keep, OUT / "geometries.png")
    plot_errors(keep["island"], OUT / "island_errors.png")
    for lam in args.lambdas:
        plot_summary(results, args.sizes, lam, OUT / f"summary_lambda{lam:g}.png")
    print(f"\nfigures written to {OUT}")


if __name__ == "__main__":
    main()
