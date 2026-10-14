# Stencil performance deep dive

**Scope:** every operator in `scripts/benchmark_operators.py` (58 cases: difference, interpolation,
divergence, vorticity / PV fluxes, Coriolis, Arakawa Jacobian, diffusion, momentum advection,
advection and the spherical operators), measured forward, reverse-mode (VJP) and forward-mode
(JVP). **Date:** 2026-10-14. **Baseline:** `main` at `32ffb45`.

The question: given that finitevolX stencils are plain `jnp` slicing (no `kernex`, no
convolutions, no custom kernels), how fast can each operator be made, and what is left once
the slicing is written well? Rules distilled from this note live in
[Writing fast stencils](../../performance.md).

!!! warning "One backend measured"
    Every number below is **XLA:CPU** (a 2-vCPU cloud VM, JAX 0.10.1, float32). The structural
    quantities (jaxpr `slice` / `pad` counts, XLA fusion counts, temporary buffers) explain
    *why* a rewrite is faster and transfer across backends; the ratios themselves must be
    re-measured on GPU / TPU (§5, phase 3).

-----

## 0. Summary

| # | Finding | Cost before | Status |
|---|---|---|---|
| 1 | **Advection VJP is the dominant cost of the library.** WENO-5 2-D: 0.93 ms forward, 34 ms VJP (36×). 83 % of the VJP is a single XLA fusion that sums the transposed `pad`s. | 10–36× the forward for WENO / TVD / upwind-3 | Root-caused; **not fixed** — seven JAX-level fixes tried and rejected (§4). Plan in §5. |
| 2 | WENO-7 / WENO-9 smoothness indicators used `jnp.stack` + `jnp.einsum` over a 4×4 / 5×5 constant matrix → stack, dot and transposes. | WENO-7 forward 33.5 ms vs WENO-5 0.93 ms | **Fixed**: forward 11×, JVP 6.3×, VJP 1.65× (WENO-9 forward 14×). |
| 3 | Composed linear stencils slice the same offset several times (`diff_x_fwd(h) - diff_x_bwd(h)` reads `h[1:-1, 1:-1]` twice); each slice expression transposes to its own `pad`. | Laplacian VJP 2 fusions | **Fixed** (slice each offset once): Laplacian VJP 1.5–2.3×, spherical Laplacian 2.0–2.3×, Arakawa Jacobian VJP 4.0–4.4×. |
| 4 | Diffusion built full-size zero flux buffers (`zeros().at[...].set(...)`) only to re-slice them. | Diffusion 4–6 fusions | **Fixed**: 2-D forward / VJP / JVP ~2×, one fusion; 3-D forward 2–3×. |
| 5 | Sums of padded operators (`Difference3D.divergence = diff_x_U_to_T(u) + diff_y_V_to_T(v)`) | two passes + add | **Fixed**: forward 1.5–2.5×. |
| 6 | 3-D z-ghost zeroing by two scatters (`.at[0].set(0).at[-1].set(0)`) | | **Fixed** (pad of the interior slabs): `Coriolis3D` forward 1.6×, VJP fusions 6 → 4. |
| 7 | Biharmonic: XLA fused the first Laplacian into each of the four shifted reads of the second. | 11 VJP fusions | **Fixed** (`_materialize`): 5 fusions, forward 1.2–1.5×, VJP 1.4–1.9×. |

What did *not* need changing: single-offset differences and averages
(`diff_*`, `avg_*`, `Interpolation*`, `Divergence2D`, `curl`), `interior()` (XLA turns
`zeros.at[1:-1, 1:-1].set(v)` into a fused `pad`; it is free) and `masked_laplacian`
(`jnp.roll` lowers to the same slices + concatenate a hand-written wrap-pad does; §4).
These already run at 70–130 % of a `2x + 1` kernel moving the same number of bytes at 514²
(cache-resident, where the reference kernel is itself not at peak); after this PR the
Laplacian runs at 91 % forward and 79 % VJP of that roofline at 1026².

## 1. Method

- **Harness:** `scripts/benchmark_operators.py` (extended in this PR with JVP timing, `--repeat`
  and `--profile`). Each case is jitted; forward, `jax.vjp(op, *x)[1](ct)` with a fixed random
  cotangent, and `jax.jvp(op, x, t)` are timed as the median of ≥0.3 s of calls, `--repeat 3`
  keeps the minimum of three medians.
- **Noise control:** `main` and this branch were each run three times, alternating, and the
  tables report the minimum per side. Operators whose code did not change are listed as
  controls; they land at 0.95–1.08×, which is the noise floor. A speedup is only claimed when
  the fusion count or the slice count moved with it.
- **Grids:** 514² (512² interior; 1 MB per float32 field, cache-resident) for everything, 1026²
  (4 MB, memory-bound) for the changed operators; 3-D cases use 10 levels.
- **Profiles:** `--profile REGEX` runs the jitted VJP under `jax.profiler.trace` and ranks XLA
  kernels by time.

## 2. Results

### 2.1 Changed operators, 514² (min of 3 runs per side)

| operator | forward ms | × | VJP ms | × | JVP ms | × | VJP fusions |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Difference1D.laplacian` | 0.077 → 0.079 | 0.97 | 0.236 → 0.101 | **2.34** | 0.082 → 0.078 | 1.05 | 2 → 1 |
| `Difference2D.laplacian` | 0.095 → 0.090 | 1.06 | 0.230 → 0.130 | **1.76** | 0.087 → 0.091 | 0.96 | 2 → 1 |
| `Difference3D.divergence` | 1.39 → 0.547 | **2.53** | 1.79 → 1.34 | 1.34 | 1.36 → 0.575 | **2.37** | 3 → 3 |
| `Difference3D.laplacian` | 0.523 → 0.551 | 0.95 | 0.936 → 0.755 | 1.24 | 0.527 → 0.539 | 0.98 | 2 → 1 |
| `Coriolis3D` | 1.38 → 0.848 | **1.62** | 1.96 → 1.82 | 1.08 | 1.38 → 0.899 | **1.53** | 6 → 4 |
| `ArakawaJacobian2D` | 0.179 → 0.183 | 0.97 | 2.76 → 0.638 | **4.32** | 0.355 → 0.362 | 0.98 | 3 → 2 |
| `arakawa_jacobian` | 0.132 → 0.126 | 1.05 | 2.56 → 0.641 | **3.99** | 0.225 → 0.223 | 1.01 | 3 → 2 |
| `Diffusion2D` | 0.256 → 0.129 | **1.98** | 0.399 → 0.187 | **2.13** | 0.268 → 0.135 | **1.98** | 4 → 1 |
| `Diffusion2D[kappa field]` | 0.300 → 0.158 | **1.89** | 0.508 → 0.399 | 1.27 | 0.361 → 0.252 | 1.43 | 5 → 2 |
| `BiharmonicDiffusion2D` | 0.410 → 0.334 | 1.23 | 0.842 → 0.602 | 1.40 | 0.411 → 0.294 | 1.40 | 11 → 5 |
| `Diffusion3D` | 1.61 → 0.807 | **2.00** | 2.37 → 1.55 | 1.53 | 1.70 → 0.814 | **2.09** | 6 → 2 |
| `SphericalLaplacian2D` | 0.152 → 0.160 | 0.95 | 0.464 → 0.233 | **1.99** | 0.148 → 0.170 | 0.87 | 2 → 1 |
| `SphericalDifference2D.laplacian_merid` | 0.143 → 0.131 | 1.09 | 0.254 → 0.214 | 1.19 | 0.130 → 0.130 | 1.00 | 1 → 1 |
| `SphericalDiffusion2D` | 0.341 → 0.362 | 0.94 | 0.539 → 0.277 | **1.95** | 0.360 → 0.368 | 0.98 | 5 → 1 |
| `Advection2D[weno7]` | 33.5 → 3.0 | **11.2** | 183 → 111 | **1.65** | 81.1 → 12.9 | **6.27** | 198 → 133 |
| `Advection2D[weno9]` (forward only) | 57.4 → 4.2 | **13.8** | | | | | |
| *controls (code unchanged)* | | | | | | | |
| `Interpolation2D.T_to_U` | 0.074 → 0.069 | 1.07 | 0.078 → 0.075 | 1.04 | | | 1 → 1 |
| `Divergence2D` | 0.093 → 0.095 | 0.98 | 0.205 → 0.200 | 1.03 | | | 3 → 3 |
| `Coriolis2D` | 0.218 → 0.224 | 0.97 | 0.635 → 0.621 | 1.02 | | | 3 → 3 |
| `Advection2D[weno5]` | 0.927 → 0.943 | 0.98 | 33.9 → 31.4 | 1.08 | 2.58 → 2.57 | 1.00 | 48 → 48 |

### 2.2 Changed operators, 1026² (memory-bound)

| operator | forward ms | × | VJP ms | × | JVP ms | × |
|---|---:|---:|---:|---:|---:|---:|
| `Difference2D.laplacian` | 0.255 → 0.278 | 0.92 | 0.753 → 0.486 | **1.55** | 0.287 → 0.277 | 1.04 |
| `ArakawaJacobian2D` | 0.593 → 0.583 | 1.02 | 10.6 → 2.41 | **4.40** | 1.22 → 1.17 | 1.05 |
| `Diffusion2D` | 0.782 → 0.396 | **1.97** | 1.20 → 0.601 | **2.00** | 0.772 → 0.395 | **1.95** |
| `BiharmonicDiffusion2D` | 1.32 → 0.856 | **1.54** | 3.54 → 1.86 | **1.90** | 1.71 → 0.901 | **1.90** |
| `Diffusion3D` | 39.2 → 13.1 | **2.99** | 48.9 → 25.6 | **1.91** | 37.3 → 13.8 | **2.71** |
| `SphericalLaplacian2D` | 0.520 → 0.507 | 1.02 | 1.61 → 0.716 | **2.25** | 0.519 → 0.489 | 1.06 |
| `SphericalDifference2D.laplacian_merid` | 0.422 → 0.436 | 0.97 | 0.806 → 0.618 | 1.30 | | |
| `Difference3D.divergence` | 18.7 → 12.6 | **1.48** | 25.1 → 24.9 | 1.01 | 17.6 → 12.6 | **1.39** |
| `Advection2D[weno7]` | 164 → 15.4 | **10.6** | 961 → 577 | **1.67** | 390 → 89.5 | **4.36** |

Single runs; the 3-D cases at this size (40 MB per field) are the noisiest rows.

## 3. Root causes

### 3.1 The advection VJP (open)

Reverse-mode AD turns every static slice `h[a:b, c:d]` into a `pad` of that slice's cotangent,
and the input's cotangent is the sum of those pads. For a linear stencil the pad operands are
cheap (a coefficient times the output cotangent), and XLA:CPU emits one fused loop over the
input — the Laplacian VJP runs at 79 % of the elementwise roofline at 1026² after this PR.

For a nonlinear stencil the operand of each pad is the *transposed* stencil: for WENO-5 the
derivative of `q̂ = Σ ω_r q_r / Σ ω_r` with `ω_r = γ_r / (β_r + ε)²` with respect to one of its
five inputs, evaluated on both flow signs. XLA:CPU still fuses every pad and its producer into
one loop over the input (`pad_add_fusion`), so for each input point it evaluates the transposed
WENO once **per shifted read** (5 offsets × 2 flow signs), behind per-element bounds checks that
block vectorisation. Profile of `Advection2D(method="weno5")` VJP at 514²:

| kernel | ms | share |
|---|---:|---:|
| `pad_add_fusion` | 35.1 | 83 % |
| next 7 kernels | 3.3 | 8 % |
| total | 42.2 | |

The boundary fallbacks make it worse. Each reconstruction stitches WENO-3 / first-order columns
onto the WENO-5 bulk with `jnp.concatenate`; isolating that on the x-face flux:

| x-face WENO-5 flux, 514² | forward | VJP |
|---|---:|---:|
| bulk faces only (no fallbacks) | 0.28 ms, 1 fusion | 4.0 ms |
| + fallbacks stitched with `concatenate` | 0.78 ms, 8 fusions | 18.2 ms |
| + `interior()` (= the library's `weno5_x`) | 0.66 ms, 8 fusions | 22.4 ms |

So the bulk WENO VJP is ~14× its forward, and the stitching multiplies it by another ~4.5×.
The same structure explains `weno3` (12×), `wenoz5` (46×), `van_leer` (8.5×), `upwind3` (5×)
and the masked path (`upwind_flux` blends three full-size tiers).

### 3.2 WENO-7 / WENO-9 smoothness indicators (fixed)

`_smoothness_indicator` evaluated `βᵣ = qᵣᵀ Bᵣ qᵣ` as
`jnp.einsum("i...,ij,j...->...", jnp.stack(q), B, jnp.stack(q))`. On a 4×4 or 5×5 constant
matrix that is a stack (materialised), a `dot_general` (a library dot call on XLA:CPU,
`ynn_fusion`) and transposes of the stacked field; the y-direction added `swapaxes` copies on top.
Profile before: `concatenate_bitcast_fusion` (the stacks), `ynn_fusion` (the dots) and
`transpose_copy_fusion` were the top kernels. After unrolling to
`Σᵢ qᵢ (Σⱼ Bᵢⱼ qⱼ)` with Python-float coefficients the indicators are elementwise code inside
one fusion; float64 results are bitwise identical, float32 within ~3e-6 relative.

### 3.3 One `pad` per slice *expression* (fixed)

`(diff_x_fwd(h) - diff_x_bwd(h))` slices `h[1:-1, 1:-1]` twice. Forward, XLA CSEs the two
reads. Reverse, JAX transposes each `slice` equation separately, so the VJP sums two pads of
the same offset; with five offsets the Laplacian VJP had 8 pads and split into 2 fusions with
an intermediate. Binding each offset to one view (`_neighbours(h)`) makes JAX accumulate the
cotangent of that view *before* transposing (`add_any`), so the VJP has exactly one pad per
distinct offset (5 for the Laplacian, 16 for the Arakawa Jacobian, down from 32) and is one
fusion. The forward arithmetic is unchanged, so outputs are bitwise identical.
`tests/test_stencil_lowering.py` pins the pad counts.

### 3.4 Full-size intermediates (fixed)

`Diffusion2D` built `flux_x = zeros_like(h).at[1:-1, 1:-2].set(...)`, multiplied by the mask and
re-sliced at two offsets for the divergence; the spherical diffusion did the same. Computing
only the faces the divergence reads, and entering the zero closed-wall faces as a `pad` of the
*face* array, removes the buffers: one fusion, forward and VJP. `Difference3D.divergence`
summed two full-size padded arrays; writing `interior(a + b)` fuses them.
`zero_z_ghosts` replaced two scatters into the full 3-D array by a pad of the interior slabs,
which also lets XLA skip computing the two discarded slabs.

### 3.5 Recomputation of a reused intermediate (fixed)

In the biharmonic, the second Laplacian reads the first at four offsets. With the rewritten
diffusion but without a barrier, XLA splits the forward into 5 fusions around the shifted
reads; with `_materialize` (`lax.optimization_barrier`) the first Laplacian is one buffer and
the forward is 2 fusions (VJP 11 → 5 fusions for rules 2 and 4 together). Verified by
comparing the compiled HLO with and without the barrier: they differ, so the barrier is not
stripped before fusion on CPU (it does not, however, survive on a *cotangent*; §4).

## 4. Negative results (tried on the advection VJP, not adopted)

All bitwise-checked against the library where they claim equality; timings at 514².

| Idea | Result |
|---|---|
| Slice each offset once in WENO-5 (rule 3.3) | bulk-faces VJP 3.9 → 6.5 ms (worse); library structure 1.2–1.5× faster in two runs, within the noise of that size |
| Mirror trick: select the 5 upwind inputs by flow sign, evaluate one WENO-5 | forward ≈ same, VJP 3.9 → 12.4 ms; not bitwise (β summation order) |
| `optimization_barrier` on each view's *cotangent* (identity with a barriered tangent) | fusion count unchanged (6 → 6), 3.3 → 3.2 ms |
| `jax.checkpoint` (default, `nothing_saveable`, `everything_saveable`) | 15.0–18.9 ms vs 15.5 ms |
| `custom_jvp` with the five partials precomputed (via `jvp` or `vjp`) and barriered | 14.8–22.7 ms vs 4.0 ms (XLA re-fuses the partials into the pads) |
| `custom_vjp` with a gather-form adjoint | 5.1 vs 5.7 ms; loses forward mode (`jax.jvp`, 4DVar TLM) → rejected |
| Fallback columns by `pad` + add, or `.at[:, k].set`, instead of `concatenate` | bitwise equal; forward 2.4 / 0.97 vs 0.69 ms, VJP unchanged |
| Fallbacks by `jnp.where` on a column index (one shared set of views, no concat) | bitwise equal; full `Advection2D` forward 1.1 → 3.0 ms, VJP 46 → 55 ms |
| `masked_laplacian`: `jnp.roll` → wrap-pad + slices, or explicit concatenate | bitwise equal; identical compiled cost (roll already lowers to slices) |

The common thread: the expensive part is an XLA:CPU fusion decision (inline the transposed
nonlinear stencil into every pad), and nothing expressible in `jnp` slicing changed it. The
only lever that changes the transposed program by construction, `custom_vjp`, costs forward
mode and still did not win on CPU.

## 5. Plan of attack

### Phase 1 — this PR

Rules 1–7 of [Writing fast stencils](../../performance.md) applied to the operators in §2,
each rewrite commented with the composed form it equals; lowering tests (`pad` per offset,
equality with the composed form, no `dot_general` in WENO-7); benchmark harness with JVP,
repeats and kernel profiles.

### Phase 2 — the advection adjoint (largest remaining cost)

Gate everything on numbers from the target backends first: if XLA:GPU's fusion cost model
already refuses to duplicate the transposed WENO (it treats `divide`-heavy producers as
expensive), the CPU result may not carry over.

| PR | Change | Gate to merge | Depends on |
|---|---|---|---|
| 2.1 | Run `benchmark_operators.py --only Advection --repeat 3` on one GPU and one TPU; add the results here | — | Phase 1 |
| 2.2 | Restructure the boundary fallbacks so the bulk reconstruction is never stitched with `concatenate` on the differentiated path (e.g. compute the closed-wall tendency from bulk faces and add the boundary-cell corrections as separate thin tendencies) | bitwise-equal goldens; VJP ≥ 2× on at least one backend; no forward regression | 2.1 |
| 2.3 | Closed-form WENO partials `∂q̂/∂q_k = Σ_r ω_r ∂q_r/∂q_k + Σ_r ω_r (q_r − q̂) ∂ln α_r/∂q_k` from at most nine shared intermediates, as a `custom_jvp` (keeps JVP and VJP, higher-order AD still works) | VJP ≥ 2× without forward regression; JVP unchanged; second derivatives match autodiff | 2.1 |
| 2.4 | If 2.2–2.3 do not reach ~4× the forward on accelerators: a fused flux kernel (Pallas) behind the same API, with `custom_vjp` + a separate JVP path | maintainer decision: AGENTS.md currently says pure `jnp` slicing | 2.3 |

### Phase 3 — validate the rules off CPU

Re-run the whole suite on GPU / TPU; in particular check that `_materialize` (rule 4) and the
one-view-per-offset rule (rule 1) do not regress there, and record the ratios in
`docs/performance.md`.

### Phase 4 — medium-cost operators

Each is 1–2 ms VJP at 514² with 7–15 fusions; apply rules 1–2:

- `Strain2D.okubo_weiss` (14 VJP fusions): build the X-point `∂v/∂x` and `∂u/∂y` once and share
  them between the shear strain and the vorticity instead of two `_x_with_ghosts` passes.
- `MomentumAdvection2D` (14–15): curl → vorticity flux → `[2:-2]` slice chains through
  full-size `interior()` buffers.
- `Vorticity2D.pv_flux_*` (7): the same `interior()` chains (`qu_at_q` → `avg_y_bwd`).
- `Advection2D[weno5, masked]`: `upwind_flux` evaluates three full-size tiers and blends them;
  restrict each tier to the faces whose mask selects it, or blend inside one fusion.

### Phase 5 — guard rails

`tests/test_stencil_lowering.py` already fails on a regression to slice-per-term code; add a
fusion-count check for the Phase 1 operators behind an opt-in marker, and an optional
benchmark job on a GPU runner that posts `--compare` against `main`.
