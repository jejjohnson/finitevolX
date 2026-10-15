"""Forward / reverse-mode benchmark of the finitevolX operator surface.

For every operator it records, at one grid size:

* median wall time of the jitted forward, of the jitted VJP
  (``jax.vjp(op, *inputs)[1](cotangent)`` with a fixed random cotangent:
  one adjoint / reverse-mode application, forward included) and of the
  jitted JVP (``jax.jvp(op, inputs, tangents)``: one tangent-linear
  application, as in 4DVar's TLM),
* the number of XLA fusions and the temp-buffer bytes of each compiled
  program (``compiled.memory_analysis()``),
* the number of ``slice`` primitives in the forward jaxpr (one per stencil
  read; each becomes a ``pad`` in the transpose),
* an efficiency figure: the time an elementwise kernel moving the same
  number of bytes takes, divided by the operator's time (1.0 = at the
  memory roofline of this machine).

Usage::

    uv run python scripts/benchmark_operators.py --n 512 --out before.json
    uv run python scripts/benchmark_operators.py --n 512 --out after.json
    uv run python scripts/benchmark_operators.py --compare before.json after.json
    uv run python scripts/benchmark_operators.py --profile "Advection2D.weno5"

``--profile`` traces the VJP of the matching cases with ``jax.profiler`` and
prints the most expensive XLA kernels (fusions), which is how the hotspots in
``docs/design/research/stencil_performance_deep_dive.md`` were found.

It is a developer tool: nothing in the test suite imports it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
import re
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import finitevolx as fvx


@dataclass
class Case:
    name: str
    fn: Callable
    n_in: int  # number of differentiable array inputs
    shape: tuple[int, ...]
    n_out: int = 1


def _op(obj, attr: str | None = None, *extra, **kwargs) -> Callable:
    """``f(*arrays) = obj.attr(*arrays, *extra, **kwargs)`` as a plain function.

    jax.jit hashes the callable it is given; an Equinox bound method of an
    operator whose grid holds arrays (the spherical grids) is unhashable, so
    every case goes through this closure instead.
    """
    target = obj if attr is None else getattr(obj, attr)

    def f(*arrays):
        return target(*arrays, *extra, **kwargs)

    return f


def _okubo_weiss_fn(u, v):
    ss = fvx.shear_strain(u, v, 1.0, 1.0)
    sn = fvx.tensor_strain(u, v, 1.0, 1.0)
    om = fvx.relative_vorticity_cgrid(u, v, 1.0, 1.0)
    return fvx.okubo_weiss(sn, ss, om)


def _cases(n: int, nz: int) -> list[Case]:
    g1 = fvx.CartesianGrid1D.from_interior(n * n - 2, 1.0)
    g2 = fvx.CartesianGrid2D.from_interior(n - 2, n - 2, 1.0, 1.0)
    g3 = fvx.CartesianGrid3D.from_interior(n - 2, n - 2, nz - 2, 1.0, 1.0, 1.0)
    gs = fvx.SphericalGrid2D.from_interior(
        n - 2, n - 2, lon_range=(0.0, 60.0), lat_range=(-30.0, 30.0)
    )
    s1, s2, s3 = (n * n,), (n, n), (nz, n, n)
    f2 = jnp.ones(s2)

    d1, d2, d3 = fvx.Difference1D(g1), fvx.Difference2D(g2), fvx.Difference3D(g3)
    i2, vo = fvx.Interpolation2D(g2), fvx.Vorticity2D(g2)
    ad2, ad3 = fvx.Advection2D(g2), fvx.Advection3D(g3)
    ma = fvx.MomentumAdvection2D(g2)

    C = [
        Case("Difference1D.diff_x_T_to_U", _op(d1, "diff_x_T_to_U"), 1, s1),
        Case("Difference1D.laplacian", _op(d1, "laplacian"), 1, s1),
    ]
    for m in ["diff_x_T_to_U", "diff_y_T_to_V", "diff_x_U_to_T", "diff_y_V_to_T"]:
        C.append(Case(f"Difference2D.{m}", _op(d2, m), 1, s2))
    C += [
        Case("Difference2D.divergence", _op(d2, "divergence"), 2, s2),
        Case("Difference2D.curl", _op(d2, "curl"), 2, s2),
        Case("Difference2D.laplacian", _op(d2, "laplacian"), 1, s2),
        Case("Difference2D.grad_perp", _op(d2, "grad_perp"), 1, s2, 2),
        Case("Difference3D.divergence", _op(d3, "divergence"), 2, s3),
        Case("Difference3D.laplacian", _op(d3, "laplacian"), 1, s3),
    ]
    for m in ["T_to_U", "T_to_X", "X_to_T", "U_to_V", "V_to_U", "U_to_T"]:
        C.append(Case(f"Interpolation2D.{m}", _op(i2, m), 1, s2))
    C += [
        Case("Divergence2D", _op(fvx.Divergence2D(g2)), 2, s2),
        Case("Coriolis2D", _op(fvx.Coriolis2D(g2)), 3, s2, 2),
        Case("Coriolis3D", _op(fvx.Coriolis3D(g3), None, f2), 2, s3, 2),
        Case("Vorticity2D.relative_vorticity", _op(vo, "relative_vorticity"), 2, s2),
    ]
    for m in [
        "pv_flux_energy_conserving",
        "pv_flux_enstrophy_conserving",
        "pv_flux_blend",
    ]:
        C.append(Case(f"Vorticity2D.{m}", _op(vo, m), 3, s2, 2))
    C += [
        Case(
            "Energetics2D.kinetic_energy",
            _op(fvx.Energetics2D(g2), "kinetic_energy"),
            2,
            s2,
        ),
        Case("Strain2D.okubo_weiss", _op(fvx.Strain2D(g2), "okubo_weiss"), 2, s2),
        Case("ArakawaJacobian2D", _op(fvx.ArakawaJacobian2D(g2)), 2, s2),
        Case("Diffusion2D", _op(fvx.Diffusion2D(g2), None, 1.0), 1, s2),
        Case("Diffusion2D[kappa field]", _op(fvx.Diffusion2D(g2)), 2, s2),
        Case(
            "BiharmonicDiffusion2D",
            _op(fvx.BiharmonicDiffusion2D(g2), None, 1.0),
            1,
            s2,
        ),
        Case("Diffusion3D", _op(fvx.Diffusion3D(g3), None, 1.0), 1, s3),
        Case("MomentumAdvection2D[energy]", _op(ma, None, "energy"), 2, s2, 2),
        Case("MomentumAdvection2D[enstrophy]", _op(ma, None, "enstrophy"), 2, s2, 2),
    ]
    for m in ["upwind1", "upwind3", "weno3", "weno5", "wenoz5", "weno7", "van_leer"]:
        C.append(Case(f"Advection2D[{m}]", _op(ad2, None, method=m), 3, s2))
    C.append(
        Case(
            "Advection2D[weno5, open]",
            _op(ad2, None, method="weno5", wall="open"),
            3,
            s2,
        )
    )
    # a basin with an island: exercises the adaptive-stencil (upwind_flux) path
    wet = np.ones(s2, dtype=bool)
    wet[n // 3 : n // 2, n // 3 : n // 2] = False
    ad2m = fvx.Advection2D(g2, mask=fvx.Mask2D.from_mask(wet))
    C.append(Case("Advection2D[weno5, masked]", _op(ad2m, None, method="weno5"), 3, s2))
    C.append(
        Case(
            "Advection1D[weno5]",
            _op(fvx.Advection1D(g1), None, method="weno5"),
            2,
            s1,
        )
    )
    C.append(Case("Advection3D[weno5]", _op(ad3, None, method="weno5"), 3, s3))

    sd = fvx.SphericalDifference2D(gs)
    C += [
        Case(
            "SphericalDifference2D.diff_lon_T_to_U",
            _op(sd, "diff_lon_T_to_U"),
            1,
            s2,
        ),
        Case(
            "SphericalDifference2D.laplacian_merid",
            _op(sd, "laplacian_merid"),
            1,
            s2,
        ),
        Case("SphericalDivergence2D", _op(fvx.SphericalDivergence2D(gs)), 2, s2),
        Case("SphericalLaplacian2D", _op(fvx.SphericalLaplacian2D(gs)), 1, s2),
        Case(
            "SphericalVorticity2D.relative_vorticity",
            _op(fvx.SphericalVorticity2D(gs), "relative_vorticity"),
            2,
            s2,
        ),
        Case(
            "SphericalDiffusion2D",
            _op(fvx.SphericalDiffusion2D(gs), None, 1.0),
            1,
            s2,
        ),
        Case(
            "SphericalAdvection2D[weno5]",
            _op(fvx.SphericalAdvection2D(gs), None, method="weno5"),
            3,
            s2,
        ),
        Case(
            "SphericalMomentumAdvection2D",
            _op(fvx.SphericalMomentumAdvection2D(gs)),
            2,
            s2,
            2,
        ),
        Case("arakawa_jacobian", _op(fvx.arakawa_jacobian, None, 1.0, 1.0), 2, s2),
        Case("kinetic_energy", fvx.kinetic_energy, 2, s2),
        Case(
            "relative_vorticity_cgrid",
            _op(fvx.relative_vorticity_cgrid, None, 1.0, 1.0),
            2,
            s2,
        ),
        Case("okubo_weiss[fn]", _okubo_weiss_fn, 2, s2),
        Case(
            "masked_laplacian",
            _op(fvx.masked_laplacian, None, f2, 1.0, 1.0),
            1,
            s2,
        ),
    ]
    return C


# -----------------------------------------------------------------------------


def _timeit(f, args, min_time):
    out = f(*args)
    jax.block_until_ready(out)
    jax.block_until_ready(f(*args))
    ts, t0 = [], time.perf_counter()
    while len(ts) < 5 or (time.perf_counter() - t0 < min_time and len(ts) < 500):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def _hlo(f, args):
    c = f.lower(*args).compile()
    txt = c.as_text()
    try:
        temp = int(c.memory_analysis().temp_size_in_bytes)
    except Exception:  # pragma: no cover
        temp = -1
    return len(re.findall(r"=\s*\S+\s+fusion\(", txt)), temp


def _n_slices(fn, args):
    jaxpr = jax.make_jaxpr(fn)(*args)

    def walk(j):
        n = 0
        for e in j.eqns:
            if e.primitive.name == "slice":
                n += 1
            for p in e.params.values():
                if hasattr(p, "jaxpr"):
                    n += walk(p.jaxpr)
        return n

    return walk(jaxpr.jaxpr)


def run(n: int, nz: int, min_time: float, only: str | None, repeat: int = 1):
    key, key2 = jax.random.key(0), jax.random.key(7)
    # roofline reference: an elementwise kernel reading 1 and writing 1 array
    ref_x = jax.random.normal(key, (n, n))
    t_ref = _timeit(jax.jit(lambda x: 2.0 * x + 1.0), (ref_x,), min_time)
    bw = 2 * ref_x.size * 4 / t_ref  # bytes / s
    rows = []
    for c in _cases(n, nz):
        if only and not re.search(only, c.name):
            continue
        ks = jax.random.split(key, c.n_in)
        args = tuple(jax.random.normal(k, c.shape) for k in ks)
        fwd = jax.jit(c.fn)
        vjp = jax.jit(lambda ct, *a, _f=c.fn: jax.vjp(_f, *a)[1](ct))
        jvp = jax.jit(lambda *at, _f=c.fn, _n=c.n_in: jax.jvp(_f, at[:_n], at[_n:])[1])
        tans = tuple(
            jax.random.normal(k, c.shape) for k in jax.random.split(key2, c.n_in)
        )
        try:
            out_shape = jax.eval_shape(c.fn, *args)
            ct = jax.tree_util.tree_map(
                lambda s: jax.random.normal(jax.random.key(99), s.shape, s.dtype),
                out_shape,
            )
            tf = min(_timeit(fwd, args, min_time) for _ in range(repeat))
            tb = min(_timeit(vjp, (ct, *args), min_time) for _ in range(repeat))
            tj = min(_timeit(jvp, (*args, *tans), min_time) for _ in range(repeat))
            ff, mf = _hlo(fwd, args)
            fb, mb = _hlo(vjp, (ct, *args))
            fj, _ = _hlo(jvp, (*args, *tans))
            ns = _n_slices(c.fn, args)
        except Exception as e:  # pragma: no cover
            print(f"!! {c.name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        size = int(np.prod(c.shape)) * 4
        ideal_f = (c.n_in + c.n_out) * size / bw
        ideal_b = (2 * c.n_in + c.n_out) * size / bw
        row = dict(
            name=c.name,
            shape=list(c.shape),
            fwd_ms=tf * 1e3,
            vjp_ms=tb * 1e3,
            jvp_ms=tj * 1e3,
            fwd_fusions=ff,
            vjp_fusions=fb,
            jvp_fusions=fj,
            fwd_temp_mb=mf / 2**20,
            vjp_temp_mb=mb / 2**20,
            slices=ns,
            fwd_eff=ideal_f / tf,
            vjp_eff=ideal_b / tb,
        )
        rows.append(row)
        print(
            f"{c.name:44s} fwd {tf * 1e3:8.3f} ms ({row['fwd_eff']:4.0%}) {ff:2d}f {mf / 2**20:6.1f}MB | "
            f"vjp {tb * 1e3:8.3f} ms ({row['vjp_eff']:4.0%}) {fb:2d}f {mb / 2**20:6.1f}MB | "
            f"jvp {tj * 1e3:8.3f} ms {fj:2d}f | {ns:3d} slices",
            file=sys.stderr,
        )
    meta = dict(
        n=n, nz=nz, jax=jax.__version__, device=str(jax.devices()[0]), bw_gbps=bw / 1e9
    )
    return dict(meta=meta, rows=rows)


def profile(n: int, nz: int, pattern: str, reps: int = 10, top: int = 8):
    """Print the most expensive XLA kernels of the jitted VJP of each match."""
    import collections
    import glob
    import gzip
    import tempfile

    key = jax.random.key(0)
    for c in _cases(n, nz):
        if not re.search(pattern, c.name):
            continue
        args = tuple(
            jax.random.normal(k, c.shape) for k in jax.random.split(key, c.n_in)
        )
        ct = jax.tree_util.tree_map(
            lambda s: jax.random.normal(jax.random.key(99), s.shape, s.dtype),
            jax.eval_shape(c.fn, *args),
        )
        vjp = jax.jit(lambda ct, *a, _f=c.fn: jax.vjp(_f, *a)[1](ct))
        jax.block_until_ready(vjp(ct, *args))
        with tempfile.TemporaryDirectory() as tmp:
            with jax.profiler.trace(tmp):
                for _ in range(reps):
                    jax.block_until_ready(vjp(ct, *args))
            path = glob.glob(f"{tmp}/**/*.trace.json.gz", recursive=True)[0]
            with gzip.open(path) as fh:
                events = json.load(fh)["traceEvents"]
        kernels = collections.Counter()
        for e in events:
            name = e.get("name", "")
            is_kernel = "fusion" in name or name.split(".")[0] in (
                "concatenate",
                "pad",
                "slice",
                "copy",
                "transpose",
            )
            if e.get("ph") == "X" and is_kernel and not name.startswith("end:"):
                kernels[name] += e.get("dur", 0)
        total = sum(kernels.values()) / reps / 1e3
        print(f"{c.name}: VJP kernels {total:.3f} ms")
        for name, dur in kernels.most_common(top):
            print(f"  {dur / reps / 1e3:8.3f} ms  {name}")


def compare(a_path: str, b_path: str):
    A = {r["name"]: r for r in json.load(open(a_path))["rows"]}
    B = {r["name"]: r for r in json.load(open(b_path))["rows"]}
    print(
        "| operator | fwd before → after (ms) | speedup "
        "| vjp before → after (ms) | speedup "
        "| jvp before → after (ms) | speedup | vjp fusions | slices |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in A:
        if k not in B:
            continue
        a, b = A[k], B[k]
        print(
            f"| `{k}` | {a['fwd_ms']:.3f} → {b['fwd_ms']:.3f} | {a['fwd_ms'] / b['fwd_ms']:.2f} "
            f"| {a['vjp_ms']:.3f} → {b['vjp_ms']:.3f} | {a['vjp_ms'] / b['vjp_ms']:.2f} "
            f"| {a['jvp_ms']:.3f} → {b['jvp_ms']:.3f} | {a['jvp_ms'] / b['jvp_ms']:.2f} "
            f"| {a['vjp_fusions']} → {b['vjp_fusions']} | {a['slices']} → {b['slices']} |"
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, default=512, help="2-D grid size incl. ghost ring")
    p.add_argument("--nz", type=int, default=10, help="3-D levels incl. ghost slabs")
    p.add_argument("--min-time", type=float, default=0.3)
    p.add_argument("--only", default=None, help="regex filter on operator names")
    p.add_argument(
        "--repeat", type=int, default=1, help="time each case k times, keep the min"
    )
    p.add_argument("--out", default=None)
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    p.add_argument(
        "--profile", metavar="REGEX", help="print the top XLA kernels of the VJP"
    )
    a = p.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    if a.profile:
        profile(a.n, a.nz, a.profile)
        return
    res = run(a.n, a.nz, a.min_time, a.only, a.repeat)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
