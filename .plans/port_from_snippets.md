# Gap Analysis: What to Port from `jej_vc_snippets` to `finitevolX`

> Generated 2026-03-11 | Comparing `finitevolX` (current) vs `jej_vc_snippets` (source)

---

## Executive Summary

`finitevolX` already has excellent coverage of core spatial operators (difference, interpolation, reconstruction, advection, diffusion, vorticity, momentum, Coriolis, elliptic solvers, boundary conditions) on 1D/2D/3D Arakawa C-grids in JAX.

The major gaps fall into **6 categories**:

| # | Category | Priority | Effort |
|---|----------|----------|--------|
| 1 | **Time Integration Schemes** | HIGH | Medium |
| 2 | **Diagnostic Operators** | HIGH | Low-Medium |
| 3 | **Geographic/Spherical Coordinates** | MEDIUM | High |
| 4 | **Initial Condition & Test Case Library** | MEDIUM | Low |
| 5 | **Boundary-Aware Robust Operators** | LOW-MEDIUM | Medium |
| 6 | **Subgrid-Scale (SGS) Turbulence Models** | LOW | Medium |

---

## 1. TIME INTEGRATION SCHEMES (HIGH PRIORITY)

**Status in finitevolX:** COMPLETELY ABSENT. No time-stepping module exists.

**What to port from `jej_vc_snippets`:**

Source: `steps_to_navier_stokes/time_steppers.py`

### 1.1 Explicit Runge-Kutta Family
| Scheme | Order | Stages | SSP | Notes |
|--------|-------|--------|-----|-------|
| Forward Euler | 1 | 1 | Yes | Baseline |
| RK2 (Heun) | 2 | 2 | Yes | Simple, good for testing |
| **SSP-RK3** | **3** | **3** | **Yes (C=1)** | **Standard for hyperbolic PDEs, TVD** |
| RK4 (Classic) | 4 | 4 | No | Best for smooth problems |
| **SSP-RK(10,4)** | **4** | **10** | **Yes (C=6)** | **High-order + SSP, niche but powerful** |

### 1.2 Multi-Step Methods
| Scheme | Order | Notes |
|--------|-------|-------|
| **Adams-Bashforth 2 (AB2)** | 2 | 1 RHS eval/step, standard in ocean models (ROMS uses AB3) |
| **Leapfrog + Robert-Asselin Filter** | 2 | Legacy (NEMO, POM), symmetric in time, good phase accuracy |

### 1.3 IMEX (Implicit-Explicit) Splitting
| Scheme | Order | Notes |
|--------|-------|-------|
| **IMEX-SSP2** | 2 | Explicit advection + implicit vertical diffusion; A-stable implicit part |

### 1.4 Split-Explicit for Barotropic/Baroclinic
| Scheme | Notes |
|--------|-------|
| **SplitExplicitRK** | Separates fast barotropic (Δt~3s) from slow baroclinic (Δt~300s); critical for ocean models |

### 1.5 Semi-Lagrangian
| Scheme | Notes |
|--------|-------|
| **SemiLagrangianSolver** | Unconditionally stable advection, CFL > 1 allowed; linear or cubic interpolation |

### Recommendation
Port at minimum: **Forward Euler, SSP-RK3, RK4, AB2, IMEX-SSP2**. These cover 90% of use cases. Add SplitExplicitRK for ocean modeling. The semi-Lagrangian solver is a nice-to-have.

### Suggested Module Structure
```
finitevolx/_src/
  timestepping/
    __init__.py
    explicit_rk.py      # Euler, RK2, SSP-RK3, RK4, SSP-RK(10,4)
    multistep.py         # AB2, Leapfrog-RAF
    imex.py              # IMEX-SSP2
    split_explicit.py    # SplitExplicitRK
    semi_lagrangian.py   # SemiLagrangianSolver
```

### Source Files
- `jej_vc_snippets/steps_to_navier_stokes/time_steppers.py` (primary, ~1200 lines)

---

## 2. DIAGNOSTIC OPERATORS (HIGH PRIORITY)

**Status in finitevolX:** Has vorticity, divergence, kinetic energy (inside momentum). Missing strain, enstrophy, Okubo-Weiss, PV diagnostics, Rossby number, Bernoulli, geostrophic velocity.

**What to port from `jej_vc_snippets`:**

### 2.1 Strain Rate Tensor
Source: `pdes/kernex/kinematics_kernex.py`, `pdes/operators/spatial_ops_kernex.py`

- **Normal strain**: E_n = du/dx - dv/dy (stretching deformation)
- **Shear strain**: E_s = dv/dx + du/dy (shearing deformation)
- **Total strain rate magnitude**: |S| = sqrt(E_n^2 + E_s^2)
- All computed on proper C-grid staggering

### 2.2 Enstrophy
Source: `pdes/kernex/kinematics_kernex.py`, `pdes/operators/spatial_ops_kernex.py`

- **Enstrophy**: Z = zeta^2 / 2 (vorticity variance)
- Key conservation diagnostic for 2D turbulence

### 2.3 Okubo-Weiss Parameter
Source: `kinematics/ocean.py`, `pdes/operators/spatial_ops_kernex.py`

- **OW**: W = E_n^2 + E_s^2 - zeta^2
- W > 0: strain-dominated (filaments), W < 0: vorticity-dominated (eddies)
- Essential for coherent structure identification

### 2.4 Potential Vorticity (Shallow Water & QG)
Source: `pdes/operators/spatial_ops_kernex.py`

- **Shallow water PV**: q = (zeta + f) / h
- **QG barotropic PV**: q = nabla^2(psi) + f + beta*y
- finitevolX has `potential_vorticity()` and PV flux but not standalone SW/QG PV diagnostics

### 2.5 Rossby Number
Source: `pdes/operators/spatial_ops_kernex.py`

- **Ro**: |zeta| / |f| — measures nonlinearity/ageostrophy

### 2.6 Bernoulli Potential
Source: `pdes/operators/spatial_ops_kernex.py`

- **B**: (u^2 + v^2)/2 + g*h — total energy per unit mass

### 2.7 Geostrophic Velocity from Height/Pressure Field
Source: `kinematics/ocean.py`, `pdes/operators/spatial_ops_kernex.py`

- **u_g** = -(g/f) * dh/dy
- **v_g** = (g/f) * dh/dx
- Useful for initialization and diagnostics

### Recommendation
Create a `diagnostics.py` module. These are all straightforward to implement using existing `Difference` and `Interpolation` operators.

### Suggested Module Structure
```
finitevolx/_src/
  diagnostics.py   # strain, enstrophy, okubo_weiss, rossby_number, bernoulli, geostrophic_velocity
```

### Source Files
- `jej_vc_snippets/pdes/kernex/kinematics_kernex.py`
- `jej_vc_snippets/pdes/operators/spatial_ops_kernex.py`
- `jej_vc_snippets/kinematics/ocean.py`

---

## 3. GEOGRAPHIC / SPHERICAL COORDINATE SUPPORT (MEDIUM PRIORITY)

**Status in finitevolX:** COMPLETELY ABSENT. All operators assume Cartesian (uniform dx, dy).

**What to port from `jej_vc_snippets`:**

Source: `derivatives/finite_volume_mesh_geo.py`, `derivatives/finite_volume_arakawac_geo.py`, `steps_to_navier_stokes/spatial_discrerization_geographical.py`

### 3.1 Spherical Metric Terms
- Zonal: dx_physical = R * cos(lat) * dlon
- Meridional: dy_physical = R * dlat
- Area: dA = R^2 * cos(lat) * dlon * dlat
- Volume: dV = R^2 * cos(lat) * dlon * dlat * dz

### 3.2 Metric-Aware Derivatives
- d/dx = (1 / (R cos(lat))) * d/dlon
- d/dy = (1/R) * d/dlat

### 3.3 Spherical Divergence
- div(u,v) = (1 / (R cos lat)) * [du/dlon + d(v cos lat)/dlat]
- Extra -(v/R) tan(lat) correction term

### 3.4 Spherical Vorticity
- curl = (1 / (R cos lat)) * [d(v cos lat)/dlat - du/dlon] + (u/R) tan(lat) correction

### 3.5 Spherical Laplacian
- nabla^2 f = (1/(R^2 cos^2 lat)) d^2f/dlon^2 + (1/(R^2 cos lat)) d(cos lat * df/dlat)/dlat

### 3.6 Physical Constants
- Coriolis: f = 2*Omega*sin(lat)
- Beta: beta = (2*Omega/R)*cos(lat)
- Earth radius, rotation rate

### Recommendation
This is the largest effort item. Two approaches:
1. **Metric tensor approach** (cleaner): Add optional metric factors to existing operators. Grid stores metric arrays (dx_u, dy_v, area_t, etc.) and operators multiply/divide by them.
2. **Separate geographic classes** (simpler initially): Create `ArakawaCGrid2DGeo` with geographic grid generation and spherical operator variants.

Option 1 is preferred long-term (unifies Cartesian and geographic under one interface). The grid already stores dx/dy — extend to spatially-varying arrays.

### Suggested Module Structure
```
finitevolx/_src/
  geographic.py        # Spherical metrics, Coriolis field, beta-plane
  # OR extend existing grid.py with metric arrays
```

### Source Files
- `jej_vc_snippets/derivatives/finite_volume_arakawac_geo.py` (primary — JAX C-grid operators with metrics)
- `jej_vc_snippets/steps_to_navier_stokes/spatial_discrerization_geographical.py` (JAX equinox implementation)

---

## 4. INITIAL CONDITIONS & TEST CASE LIBRARY (MEDIUM PRIORITY)

**Status in finitevolX:** Has 3 example scripts but no reusable IC library.

**What to port from `jej_vc_snippets`:**

Source: `shallow_water_model/initial_conditions.py`

### 4.1 Height Field Generators
| IC | Description | Use Case |
|----|-------------|----------|
| Uniform Westerly | Geostrophically balanced zonal flow | Basic test |
| Zonal Jet | Sharp tanh profile | Barotropic instability |
| Sinusoidal | Standing wave patterns | Linear wave tests |
| Equatorial Easterly | Equatorially symmetric | Equatorial dynamics |
| Gaussian Blob | Isolated pressure anomaly | Adjustment problems |
| Cyclone in Westerly | Blob + shear flow | Realistic weather |
| Step Function | Sharp discontinuity | Shock/front tests |
| Sharp Shear | Piecewise velocity | Kelvin-Helmholtz |

### 4.2 Topography/Orography Generators
| Topography | Description | Use Case |
|------------|-------------|----------|
| Flat | Zero bottom | Baseline |
| Slope | Linear shelf | Continental shelf |
| Gaussian Mountain | Isolated peak | Lee wave, Taylor column |
| Seamount | Axisymmetric underwater | Bottom friction tests |

### 4.3 Balancing Utilities
- **Geostrophic balance**: Given h, compute balanced (u, v)
- **Random noise**: Small-amplitude perturbation to trigger instabilities

### Recommendation
Useful for testing and examples. Port as a `test_cases` or `initial_conditions` submodule.

### Suggested Module Structure
```
finitevolx/_src/
  test_cases/
    __init__.py
    shallow_water.py    # IC generators, topography, geostrophic balancing
```

### Source Files
- `jej_vc_snippets/shallow_water_model/initial_conditions.py`

---

## 5. BOUNDARY-AWARE ROBUST OPERATORS (LOW-MEDIUM PRIORITY)

**Status in finitevolX:** Has `ArakawaCGridMask` with stencil capability for WENO degradation. Has basic BCs (Dirichlet, Neumann, Periodic, etc.). Missing some advanced BC types and robust operators.

**What to port from `jej_vc_snippets`:**

### 5.1 Robin Boundary Condition
Source: `derivatives/boundary_conditions.py`

- Linear combination: alpha * u + beta * du/dn = gamma
- Generalizes Dirichlet (beta=0) and Neumann (alpha=0)
- Useful for radiation BCs, absorbing boundaries

### 5.2 High-Order Extrapolation BC
Source: `derivatives/boundary_conditions.py`

- Polynomial extrapolation (orders 1-5) for ghost cells
- Cache-friendly precomputed index/coefficient arrays
- Useful for outflow boundaries

### 5.3 Land Mask Boundary Condition
Source: `derivatives/boundary_conditions.py`

- No-normal-flow at arbitrary land boundaries (not just domain edges)
- Works with stencil capability metadata

### 5.4 Robust Finite Differences Near Boundaries
Source: `pdes/masks_dataclasses.py`

- **Central difference with fallback**: Uses central where neighbors exist, degrades to forward/backward near walls
- **Multi-scale forward/backward**: 2nd-order where space permits, 1st-order near land
- **Upwind flux with adaptive stencil**: 2nd-order (Beam-Warming) in open water, 1st-order near coast, zero at land

### 5.5 Advanced Sponge Layers
Source: `derivatives/boundary_conditions.py`, `pdes/masks_dataclasses.py`

- Multiple damping profiles: linear, quadratic, exponential
- Precomputed damping coefficient arrays
- Configurable sponge width

### Recommendation
finitevolX already has `Sponge1D` and `ArakawaCGridMask` with stencil capability. The main additions would be:
- Robin BC
- Extrapolation BC (orders 1-5)
- Robust multi-scale difference operators that auto-degrade near land

### Source Files
- `jej_vc_snippets/derivatives/boundary_conditions.py`
- `jej_vc_snippets/pdes/masks_dataclasses.py`

---

## 6. SUBGRID-SCALE TURBULENCE MODELS (LOW PRIORITY)

**Status in finitevolX:** Has diffusion and biharmonic diffusion. No SGS models.

**What to port from `jej_vc_snippets`:**

Source: `methane_retrieval/les_fvm_jax.py`

### 6.1 Smagorinsky Model
- Eddy viscosity: nu_t = (C_s * Delta)^2 * |S|
- |S| computed from strain rate tensor on C-grid
- C_s typically 0.1-0.2
- Filter width Delta = sqrt(dx * dy)

### Recommendation
Low priority — more model-level than operator-level. But Smagorinsky is simple and widely used. Could be a thin wrapper around existing strain rate + diffusion operators.

> **Note:** Pressure projection (FFT-based Poisson solver for incompressible flow) was previously listed here but has been redirected to `spectraldiffx/.plans/port_from_snippets.md`.

### Source Files
- `jej_vc_snippets/methane_retrieval/les_fvm_jax.py`

---

## ITEMS EXPLICITLY NOT RECOMMENDED FOR PORTING

These exist in `jej_vc_snippets` but are out of scope for a finite volume operator library:

| Item | Reason |
|------|--------|
| 4D-Var data assimilation | Model-level, not operator |
| Ensemble Kalman filter | Model-level |
| DINEOF interpolation | Data analysis tool |
| xarray-based operators | finitevolX is pure JAX arrays |
| kernex-based stencil operators | kernex is deprecated; finitevolX already has native JAX implementations |
| QG/SWM model classes | Model-level (scripts/ already has examples) |
| Ice dynamics kinematics | Too specialized |
| Atmospheric thermodynamics | Too specialized |
| Coordinate/xarray infrastructure | Different paradigm |

## SPECTRAL ITEMS → REDIRECTED TO `spectraldiffx`

The following spectral-related snippet code belongs in [`spectraldiffx`](https://github.com/jejjohnson/spectraldiffx) rather than finitevolX. A detailed porting plan is at **`spectraldiffx/.plans/port_from_snippets.md`**.

| Snippet File | What It Provides | spectraldiffx Gap |
|-------------|-----------------|-------------------|
| `derivatives/spectral_fft_jax.py` | Fourier Jacobian, inverse Laplacian, biharmonic, hyperviscosity, dealiasing | Physics-oriented Fourier operators |
| `derivatives/spectral_chebychev.py` | 3D Chebyshev, integration matrix, Clenshaw-Curtis quadrature | Extended Chebyshev support |
| `linear_algebra/solver_spectral_jax.py` | DST/DCT Poisson/Helmholtz solvers (Dirichlet/Neumann BCs) | Non-periodic BC solvers |
| `linear_algebra/spectral_transforms_jax.py` | Pure JAX DCT/DST types I-IV (no scipy dependency) | JAX-native transforms |
| `linear_algebra/solver_capacitance.py` | Capacitance matrix method for irregular domains | Extends spectral solvers to masked domains |

**Note on finitevolX deduplication:** finitevolX currently bundles its own spectral transforms (`_src/spectral_transforms.py`) and spectral elliptic solvers (`_src/elliptic.py`). Once spectraldiffx has feature parity, finitevolX should import from spectraldiffx instead of maintaining copies.

---

## IMPLEMENTATION PRIORITY ROADMAP

### Phase 1 (Quick Wins)
1. **Diagnostics module** — strain, enstrophy, Okubo-Weiss, Rossby number, Bernoulli
   - Effort: ~1-2 days
   - Builds on existing Difference/Interpolation operators
   - High value for users analyzing simulation output

2. **Basic time steppers** — Forward Euler, SSP-RK3, RK4
   - Effort: ~1-2 days
   - Clean Butcher tableau implementation
   - Makes finitevolX usable for actual simulations without external ODE solvers

### Phase 2 (Core Additions)
3. **Advanced time steppers** — AB2, IMEX-SSP2, SplitExplicitRK
   - Effort: ~3-5 days
   - Essential for ocean modeling use cases

4. **Initial conditions library**
   - Effort: ~1-2 days
   - Makes testing and demos much easier

### Phase 3 (Major Feature)
5. **Geographic/spherical coordinate support**
   - Effort: ~1-2 weeks
   - Extends finitevolX from regional Cartesian to global models
   - Requires careful design decisions about metric tensor integration

### Phase 4 (Polish)
6. **Advanced BCs** — Robin, high-order extrapolation, robust multi-scale operators
   - Effort: ~3-5 days
   - Improves handling of complex coastlines and open boundaries

7. **Smagorinsky SGS model**
   - Effort: ~1 day
   - Thin wrapper around strain + diffusion

---

## CROSS-REFERENCE: Source File → Target Module

| Source File (jej_vc_snippets) | Target Module (finitevolx) | Gap Category |
|-------------------------------|---------------------------|--------------|
| `steps_to_navier_stokes/time_steppers.py` | `_src/timestepping/` | 1. Time Integration |
| `pdes/kernex/kinematics_kernex.py` | `_src/diagnostics.py` | 2. Diagnostics |
| `pdes/operators/spatial_ops_kernex.py` | `_src/diagnostics.py` | 2. Diagnostics |
| `kinematics/ocean.py` | `_src/diagnostics.py` | 2. Diagnostics |
| `derivatives/finite_volume_arakawac_geo.py` | `_src/geographic.py` | 3. Geographic |
| `steps_to_navier_stokes/spatial_discrerization_geographical.py` | `_src/geographic.py` | 3. Geographic |
| `derivatives/finite_volume_mesh_geo.py` | `_src/grid.py` (extend) | 3. Geographic |
| `shallow_water_model/initial_conditions.py` | `_src/test_cases/` | 4. Test Cases |
| `derivatives/boundary_conditions.py` | `_src/bc_1d.py` (extend) | 5. Boundary |
| `pdes/masks_dataclasses.py` | `_src/masks/` (extend) | 5. Boundary |
| `methane_retrieval/les_fvm_jax.py` | `_src/sgs.py` | 6. SGS |
