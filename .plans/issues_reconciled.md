# Reconciled Plan: GitHub Issues × Snippet Porting Opportunities

> Updated 2026-03-11 | All 25 open issues have been updated with comments.
> 5 new issues created: #105, #106, #107, #108, #109.
> This document now focuses on **bundled items and spectral redirects**.

---

## Status of All Issues (Updated)

Every open GitHub issue has been reviewed and updated with a comment detailing:
- Relevant `jej_vc_snippets` source files (where applicable)
- Implementation details, formulas, and code sketches
- Dependencies on other issues
- Current finitevolX coverage

| Group | Issues | Summary |
|-------|--------|---------|
| FULL MATCH (snippets directly address) | #70, #73, #2, #25, #7, #68 | Comments added with snippet sources, module structure, implementation plans |
| PARTIAL MATCH (snippets cover some) | #75, #81, #79, #87, #82, #86 | Comments added clarifying what exists, what's missing, consolidation opportunities |
| NO SNIPPET (build from scratch) | #71, #84, #85, #83, #13, #3, #8 | Comments added with dependency mapping, JAX implementation notes |
| HOUSEKEEPING | #21, #22, #23, #24 | Comments added confirming status |
| LIKELY STALE (already implemented) | #10, #72 | Comments recommending closure |

---

## PREVIOUSLY UNTRACKED: Now Tracked as Issues #105–#109

The following code from `jej_vc_snippets` previously had no GitHub issue. Issues have now been created.

---

### 1. Initial Conditions & Test Case Library → **#105**

**Source:** `jej_vc_snippets/shallow_water_model/initial_conditions.py`

**What it provides:**

| Category | Items |
|----------|-------|
| **Height field ICs** | Uniform Westerly (geostrophically balanced), Zonal Jet (tanh profile, barotropic instability), Sinusoidal (standing waves), Equatorial Easterly, Gaussian Blob (isolated anomaly), Cyclone in Westerly, Step Function (shock tests), Sharp Shear (Kelvin-Helmholtz) |
| **Topography** | Flat, Linear Slope (continental shelf), Gaussian Mountain (lee waves, Taylor column), Seamount (axisymmetric, bottom friction) |
| **Balancing utilities** | `add_geostrophic_balance(h, f, g)` → computes balanced (u, v); `add_random_noise(field, amplitude)` → trigger instabilities |

**Why it matters:**
- finitevolX has 3 example scripts but no **reusable** IC library
- Every test case and demo currently needs to manually construct initial conditions
- Geostrophic balancing utility is needed for any initialized shallow water simulation

**Suggested target:** `finitevolx/_src/test_cases/shallow_water.py`

**Priority:** MEDIUM — high usability impact, low effort (~1-2 days)

---

### 2. Semi-Lagrangian Solver → **#109**

**Source:** `jej_vc_snippets/steps_to_navier_stokes/time_steppers.py`

**What it provides:**
- Characteristic-based advection solver: traces departure points backward, interpolates
- **Unconditionally stable** — CFL can exceed 1
- Configurable interpolation: linear (order 1, diffusive but monotonic) or cubic (order 3, accurate but oscillatory)
- Periodic and edge boundary handling

**Algorithm:**
1. Compute velocity field from `rhs_fn`
2. Backtrack departure points: `x_dep = x_i − u·Δt`
3. Interpolate old field at departure points

**Why it matters:**
- Enables large time steps for advection-dominated problems
- Useful for tracer transport in coarse-resolution models
- Depends on generalized interpolation (relates to Issue #3 / interpax)

**Suggested target:** `finitevolx/_src/timestepping/semi_lagrangian.py`

**Priority:** LOW — niche use case, but ready to port

---

### 3. Robin Boundary Condition → **#106**

**Source:** `jej_vc_snippets/derivatives/boundary_conditions.py`

**What it provides:**
- Linear combination BC: `α·u + β·∂u/∂n = γ`
- Generalizes Dirichlet (β=0) and Neumann (α=0) into a single framework
- Precomputed ghost cell coefficients for efficient repeated application

**Why it matters:**
- **Radiation/absorbing boundary conditions** for open ocean boundaries use Robin BCs
- finitevolX has Dirichlet, Neumann, Periodic, Reflective, Slip, Outflow, Sponge — but no Robin
- Adding Robin completes the standard BC taxonomy

**Suggested target:** Add `Robin1D` class to `finitevolx/_src/bc_1d.py`

**Priority:** LOW-MEDIUM — important for open-boundary ocean modeling

---

### 4. High-Order Extrapolation Boundary Condition → **#107**

**Source:** `jej_vc_snippets/derivatives/boundary_conditions.py`

**What it provides:**
- Polynomial extrapolation (orders 1 through 5) for ghost cell filling
- Precomputed index/coefficient arrays (cache-friendly, JIT-compatible)
- Higher-order extrapolation reduces boundary truncation error

**Why it matters:**
- finitevolX's `Outflow1D` is zero-gradient (order 0 extrapolation)
- Orders 2-5 give more accurate outflow/radiation boundaries
- Essential for high-resolution simulations where boundary errors propagate inward

**Suggested target:** Add `Extrapolation1D(face, order)` class to `finitevolx/_src/bc_1d.py`

**Priority:** LOW — refinement of existing outflow BC

---

### 5. IMEX Time Stepping (expanded scope beyond #70) → **#108**

> **Note:** Pressure projection for incompressible flow (previously listed here) has been moved to `spectraldiffx/.plans/port_from_snippets.md` as it relies on FFT-based spectral Poisson solvers. The orchestration layer (divergence → solve → velocity correction) could live in finitevolX once spectraldiffx provides the solver.

**Source:** `jej_vc_snippets/steps_to_navier_stokes/time_steppers.py`

**What it provides:**
- **IMEX-SSP2** — 2nd-order implicit-explicit splitting (γ = 1 − 1/√2)
  - Explicit part: SSP with C=1 (advection, nonlinear terms)
  - Implicit part: A-stable, L-stable (vertical diffusion, Coriolis)
- **SplitExplicitRK** — Barotropic/baroclinic mode splitting
  - RK3 for 3D baroclinic with slow Δt
  - N substeps for 2D barotropic with fast δt = Δt/N
  - Time-averaging of barotropic solution before coupling back

**Why it matters:**
- Issue #70 only asks for Euler, RK2, SSP-RK3, AB3
- IMEX and SplitExplicit are **critical for realistic ocean models** (vertical mixing is stiff, barotropic gravity waves are fast)
- Both are in the snippet and ready to port

**Note:** These were mentioned in the update comment on #70 as "extended scope". Consider either expanding #70's acceptance criteria or creating a separate issue for advanced time steppers.

---

### 6. Leapfrog with Robert-Asselin(-Williams) Filter

**Source:** `jej_vc_snippets/steps_to_navier_stokes/time_steppers.py`

**What it provides:**
- Three-level leapfrog: `y^{n+1} = y^{n-1} + 2·Δt·f(t^n, y^n)`
- Robert-Asselin filter: `y^n_filtered = y^n + α·(y^{n-1} − 2·y^n + y^{n+1})` with α=0.01-0.1
- Damps the spurious computational mode inherent in leapfrog

**Why it matters:**
- Legacy scheme used in NEMO and POM — important for reproducing/comparing with existing model output
- Niche but still used in operational oceanography

**Priority:** LOW — skip unless NEMO compatibility is needed

---

### 7. Smagorinsky SGS Model (standalone)

**Source:** `jej_vc_snippets/methane_retrieval/les_fvm_jax.py`

**What it provides:**
- Eddy viscosity: `ν_t = (C_s·Δ)²·|S̄|`
- Strain rate magnitude computed from C-grid velocity fields
- Blending of molecular + SGS viscosity

**Why it matters:**
- Issue #86 covers full TKE closure but Smagorinsky is a simpler, standalone diagnostic closure
- Only depends on strain rate operators (from #73 diagnostics) and existing `Diffusion2D`
- ~20 lines to implement once diagnostics module exists

**Note:** Partially covered by #86 comment update. Could be a quick sub-deliverable of #86 or standalone.

---

## Summary: New Issues Created

| Issue | Feature | Source File | Priority | Effort | Dependencies |
|-------|---------|------------|----------|--------|-------------|
| **#105** | Initial conditions & test case library | `shallow_water_model/initial_conditions.py` | MEDIUM | 1-2 days | None |
| **#106** | Robin boundary condition | `derivatives/boundary_conditions.py` | LOW-MEDIUM | 0.5 days | None |
| **#107** | High-order extrapolation BC | `derivatives/boundary_conditions.py` | LOW | 0.5 days | None |
| **#108** | IMEX + SplitExplicit time steppers | `steps_to_navier_stokes/time_steppers.py` | HIGH | 3-5 days | #70 (basic steppers), #85 (TDMA for implicit part) |
| **#109** | Semi-Lagrangian solver | `steps_to_navier_stokes/time_steppers.py` | LOW | 2-3 days | #3 (interpax, optional) |

Items that can be **bundled into existing issues** instead of creating new ones:
- Rossby number, deformation rate → bundle with **#73** (already noted in comment)
- Smagorinsky SGS → bundle with **#86** (already noted in comment)
- Leapfrog + RAF → bundle with **#70** if desired (noted in comment)

Items **redirected to `spectraldiffx`** (see `spectraldiffx/.plans/port_from_snippets.md`):
- Pressure projection (FFT-based Poisson solver) — from `methane_retrieval/les_fvm_jax.py`
- Spectral FFT derivatives (Jacobian, biharmonic, hyperviscosity) — from `derivatives/spectral_fft_jax.py`
- Chebyshev spectral methods (3D, integration, quadrature) — from `derivatives/spectral_chebychev.py`
- DST/DCT solvers (Dirichlet/Neumann BCs) — from `linear_algebra/solver_spectral_jax.py`
- Pure JAX DCT/DST transforms — from `linear_algebra/spectral_transforms_jax.py`
- Capacitance matrix solver — from `linear_algebra/solver_capacitance.py`

---

## CROSS-REFERENCE: Issue → Snippet File → Target Module

| Issue | Snippet File(s) | Target Module in finitevolX |
|-------|-----------------|----------------------------|
| #70 | `time_steppers.py` | `_src/timestepping/` (new) |
| #73 | `kinematics_kernex.py`, `spatial_ops_kernex.py`, `ocean.py` | `_src/diagnostics.py` (new) |
| #2 | `kinematics_kernex.py`, `spatial_ops_kernex.py` | `_src/diagnostics.py` (new) |
| #25 | `spatial_ops_kernex.py`, `ocean.py` | `_src/diagnostics.py` (replaces `operators.py`) |
| #7 | `finite_volume_arakawac_geo.py`, `spatial_discrerization_geographical.py` | `_src/grid.py` (extend) + `_src/geographic.py` (new) |
| #68 | `masks_dataclasses.py` | `_src/reconstruction.py` (extend) |
| #75, #79, #81 | `ocean.py` (partial) | `_src/forcing.py` (new) |
| #82 | `atmosphere.py` (partial) | `_src/diagnostics.py` or `_src/difference.py` (extend) |
| #86 | `les_fvm_jax.py` (Smagorinsky only) | `_src/sgs.py` (new) |
| #87 | (mostly already done) | `_src/elliptic.py` (extend with wrappers) — **spectral solvers may migrate to spectraldiffx** |
| #105 | `shallow_water_model/initial_conditions.py` | `_src/test_cases/` (new) |
| #106 | `derivatives/boundary_conditions.py` | `_src/bc_1d.py` (extend) |
| #107 | `derivatives/boundary_conditions.py` | `_src/bc_1d.py` (extend) |
| #108 | `steps_to_navier_stokes/time_steppers.py` | `_src/timestepping/` (new) |
| #109 | `steps_to_navier_stokes/time_steppers.py` | `_src/timestepping/` (new) |
| **SPECTRAL** | `spectral_fft_jax.py`, `spectral_chebychev.py`, `solver_spectral_jax.py`, `spectral_transforms_jax.py`, `solver_capacitance.py` | **→ spectraldiffx** (see `spectraldiffx/.plans/port_from_snippets.md`) |

---

## EPIC ISSUES

| Epic | Issue | Sub-Issues |
|------|-------|------------|
| Operators & Diagnostics | **#110** | #2, #7, #73, #82 |
| Time Integration | **#111** | #70, #108, #109 |
| Boundary Conditions | **#112** | #106, #107 |
| Forcing & Parameterizations | **#113** | #75 (→ #79, #81), #86 |
| Diffusion & Mixing | **#114** | #83, #84 |
| Solvers & Linear Algebra | **#115** | #85, #87, #71, #10, #72 |
| Advection & Reconstruction | **#116** | #68, #13, #3 |
| Infrastructure & Testing | **#117** | #21 (→ #22, #23, #24, #25), #105, #8 |

---

## DEPENDENCY GRAPH

### Tier 0 — Foundational (no internal dependencies)

| Issue | Title | Blocks |
|-------|-------|--------|
| #2 | Mega List of Operators | #25, #73 |
| #3 | Integrate Generalized Interpolator | (optional: #109) |
| #7 | Spherical Coordinate Derivatives | — |
| #8 | CPU vs GPU Examples | — |
| #85 | TDMA Solver | #84, #86, #108 |
| #83 | Equation of State | #84 |

### Tier 1 — Housekeeping

| Issue | Title | Parent | Blocks |
|-------|-------|--------|--------|
| #21 | Reconcile Legacy/Refactored API | — | parent of #22, #23, #24, #25 |
| #22 | Move cgrid_mask.py | #21 | — |
| #23 | Move test files | #21 | — |
| #24 | Promote weno.py | #21 | — |
| #25 | Port legacy operators | #21 | #73 |

### Tier 2 — Core Modules

| Issue | Title | Depends On | Blocks |
|-------|-------|-----------|--------|
| #70 | Explicit Time Integrators | — | #108 |
| #87 | Streamfunction/Pressure Solvers | — | #108 |
| #68 | Upwind Flux Dispatch | — | — |
| #105 | Test Case Library | — | — |
| #106 | Robin BC | — | — |
| #107 | High-Order Extrapolation BC | — | — |

### Tier 3 — Physics Modules

| Issue | Title | Depends On | Blocks |
|-------|-------|-----------|--------|
| #73 | Energy/Enstrophy Diagnostics | #2, #25 | #86 |
| #75 | Forcing & Dissipation (parent) | — | parent of #79, #81 |
| #79 | Wind Stress | #75 | — |
| #81 | Friction Operators | #75 | — |
| #82 | Vertical Velocity Diagnostic | — | — |

### Tier 4 — Advanced Physics

| Issue | Title | Depends On |
|-------|-------|-----------|
| #84 | Isoneutral/GM-Redi Diffusion | #83, #85 |
| #86 | TKE Turbulence Closure | #73, #85 |
| #108 | IMEX + Split-Explicit | #70, #85, #87 |
| #109 | Semi-Lagrangian Advection | (#3 optional) |

### Solver Cluster (related, potentially stale)

| Issue | Title | Notes |
|-------|-------|-------|
| #10 | Elliptical Solvers | Likely stale — may already be covered |
| #71 | Multigrid Helmholtz | Related to #72, #87 |
| #72 | Capacitance Matrix | Likely stale — already in `_src/elliptic.py` |

### Parent-Child Groups

- **#21** → #22, #23, #24, #25
- **#75** → #79, #81

### Critical Path Chains

```
#85 (TDMA) ──→ #84 (GM/Redi)
           ──→ #86 (TKE)
           ──→ #108 (IMEX)

#70 (explicit integrators) ──→ #108 (IMEX/split-explicit)
#87 (pressure solvers)     ──→ #108 (IMEX/split-explicit)

#83 (EOS) ──→ #84 (GM/Redi)

#2 (operator list) ──→ #25 (port legacy) ──→ #73 (diagnostics) ──→ #86 (TKE)
```
