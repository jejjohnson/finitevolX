# Writing fast stencils

finitevolX operators are plain `jnp` slicing (see [Raw Stencils](stencils.md)).
How a stencil is *written* still decides what XLA compiles, especially for
reverse-mode AD (`jax.grad`, `jax.vjp`), which data assimilation and
training call far more often than the forward pass. These rules come from
profiling every operator; the measurements, the negative results and the
roadmap are in the
[stencil performance deep dive](design/research/stencil_performance_deep_dive.md).

!!! note "Where the numbers come from"
    All timings here are XLA:CPU (2-vCPU VM, JAX 0.10, float32). The
    *structural* effects (number of `pad`s in the VJP, number of fusions,
    temporary buffers) are backend-independent; the speedups are not.
    Re-measure on GPU / TPU with `scripts/benchmark_operators.py` before
    relying on a ratio there.

## The cost model in one paragraph

A stencil is memory-bound: the ideal compiled program reads each input once
and writes each output once, inside **one fused loop**. Every static slice
`h[a:b, c:d]` is free in the forward pass, but its transpose in the VJP is a
`pad` of the cotangent, one per slice *expression*. XLA then has to sum those
pads, and when there are many (or when a full-size intermediate sits between
two stencils) it splits the program into several fusions with temporary
buffers. Fewer distinct reads and no full-size intermediates keep it as one
pass.

## Rules

1. **Slice each offset once.** Bind shifted views to names and reuse them,
   rather than re-slicing inside each sub-expression. For the 3×3
   neighbourhood use `_neighbours(h)` (`operators/stencils.py`):

   ```python
   nb = _neighbours(h)  # nb.c, nb.e, nb.w, nb.n, nb.s, nb.ne, ...  [..., Ny-2, Nx-2]
   # same arithmetic as (diff_x_fwd(h) - diff_x_bwd(h)) / dx**2
   d2x = ((nb.e - nb.c) - (nb.c - nb.w)) / dx**2
   ```

   Keep the arithmetic of the composed form so the forward pass is
   bitwise-identical, and say so in a comment that names the composed form.
   Laplacian VJP: 1.5–2.3× faster; spherical Laplacian 2.0–2.3×; Arakawa
   Jacobian VJP: 4.0–4.4× faster.

   This rule is for **linear and bilinear** stencils. For heavy nonlinear
   stencils (WENO) it did not help and sometimes hurt: their VJP cost is
   dominated by something else (see the deep dive, §3.1).

2. **Don't build a full-size intermediate just to slice it again.**
   `zeros_like(h).at[1:-1, 1:-2].set(flux)` followed by
   `flux[1:-1, 1:-1] - flux[1:-1, :-2]` materialises the flux array. Compute
   only the faces the next stencil reads and add zero wall faces as a `pad`
   of the *result* (see `diffusion._diffusion_2d_impl`). 2-D diffusion:
   forward and VJP ~2× faster, one fusion, no temporaries; 3-D diffusion:
   forward 2–3×, VJP 1.5–1.9×.

3. **Fuse sums of operators.** `op_a(u) + op_b(v)` with each returning a
   padded full-size array is two passes plus an add; one `interior(a + b)`
   is one pass (`Difference3D.divergence`: 1.5–2.5× faster forward).

4. **Materialise an intermediate that the next stencil reads at several
   offsets.** XLA may otherwise fuse a cheap producer into every shifted
   read and recompute it per offset. `_materialize(x)` (an
   `optimization_barrier`, identity for values and derivatives) keeps it as
   one buffer: the biharmonic's first Laplacian is the example (VJP fusions
   11 → 5; forward 1.2–1.5× and VJP 1.4–1.9× faster together with rule 2).

5. **Zero ghost slabs with a pad, not scatters.** `zero_z_ghosts` pads the
   interior slabs, which also lets XLA skip computing the discarded ones.

6. **Unroll small constant contractions.** `jnp.einsum` over a `jnp.stack`
   of shifted views (the WENO-7/9 smoothness indicators `qᵀ B q`) lowers to
   a stack, a dot and transposes of the stacked field. Writing the
   quadratic form as nested Python sums over the constant coefficients keeps
   it one elementwise fusion: WENO-7 advection forward ~11× faster (WENO-9
   ~14×), JVP 4–6× faster.

7. **Measure the VJP with an external cotangent.** Benchmark
   `jax.vjp(op, *x)[1](ct)` with a fixed random `ct`. A loss such as
   `0.5 * sum(op(x)**2)` makes the cotangent a function of the inputs, XLA
   fuses it back in, and rankings can invert.

## Known open problem: the advection VJP

The reverse pass of the WENO / TVD advection operators costs 10–35× their
forward pass on XLA:CPU (WENO-5 2-D: ~1 ms forward, ~30 ms VJP), which makes
advection the dominant cost of any adjoint or training loop. The rewrites
above do not fix it; the deep dive (§3.1, §5) has the profile, the
experiments that did not work and the plan.

## Tools

- `scripts/benchmark_operators.py` times the forward pass, the VJP and the
  JVP of every operator and reports XLA fusion counts, temporary bytes and
  an efficiency figure against an elementwise roofline; `--profile REGEX`
  prints the most expensive XLA kernels of a VJP:

  ```bash
  uv run python scripts/benchmark_operators.py --n 514 --repeat 3 --out before.json
  # … change code …
  uv run python scripts/benchmark_operators.py --n 514 --repeat 3 --out after.json
  uv run python scripts/benchmark_operators.py --compare before.json after.json
  uv run python scripts/benchmark_operators.py --profile "Advection2D.weno5"
  ```

  On a shared machine, sequential runs are noisy (±30 % below 0.3 ms, and
  occasional 2× outliers on larger cases): use `--repeat`, run the
  before / after pair more than once, and trust a speedup only when the
  fusion count or the `slices` column moved with it.

- `tests/test_stencil_lowering.py` pins the lowering contract without any
  timing: the VJP jaxpr of each rewritten operator contains exactly one
  `pad` per distinct offset, each rewrite equals its composed form, and the
  WENO-7 jaxpr contains no `dot_general`. Add a case there when you rewrite
  an operator for speed.
