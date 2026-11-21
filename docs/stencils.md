# Raw Stencil Primitives

This page explains the theory behind the raw stencil functions in `gridax`,
walking from the continuous equations through the Gauss divergence theorem to
the discrete stencils that the library implements.

For details on ghost-cell layout, same-size array conventions, and boundary
interactions, see the [C-Grid Discretization](cgrid_discretization.md) page.

---

## 1. The Continuous Equations

Geophysical fluid dynamics models solve **conservation laws** — partial
differential equations that express the conservation of mass, momentum,
and tracers:

$$
\frac{\partial q}{\partial t} + \nabla \cdot \mathbf{F}(q) = 0
$$

where $q$ is a conserved density (layer thickness $h$, momentum
$hu$, tracer concentration $c$, etc.) and $\mathbf{F}(q)$ is the
corresponding flux vector.  In two dimensions this expands to:

$$
\frac{\partial q}{\partial t}
  + \frac{\partial F^x}{\partial x}
  + \frac{\partial F^y}{\partial y} = 0
$$

For example, in the shallow-water mass equation $q = h$ and
$\mathbf{F} = (hu,\, hv)$, giving the familiar continuity equation:

$$
\frac{\partial h}{\partial t}
  + \frac{\partial (hu)}{\partial x}
  + \frac{\partial (hv)}{\partial y} = 0
$$

These are continuous equations defined at every point in space.  To solve
them numerically, we need to represent them on a finite grid.

---

## 2. The Finite-Volume Approximation

### Integrating over a control volume

Rather than approximating derivatives at a point (as finite-difference
methods do), the **finite-volume method** works with the *integral* form
of the conservation law.  We divide the domain into non-overlapping cells
$\Omega_{j,i}$ and integrate the conservation law over each cell:

$$
\int_{\Omega_{j,i}} \frac{\partial q}{\partial t}\, dA
  + \int_{\Omega_{j,i}} \nabla \cdot \mathbf{F}\, dA = 0
$$

### The Gauss divergence theorem

The key step is applying the **Gauss divergence theorem** (also known as
the divergence theorem or Green's theorem in 2D) to convert the volume
integral of the divergence into a surface integral over the cell boundary:

$$
\int_{\Omega_{j,i}} \nabla \cdot \mathbf{F}\, dA
  = \oint_{\partial \Omega_{j,i}} \mathbf{F} \cdot \hat{\mathbf{n}}\, ds
$$

where $\hat{\mathbf{n}}$ is the outward unit normal to the cell boundary
and $ds$ is the line element along the boundary.  This gives us:

$$
\frac{d}{dt} \int_{\Omega_{j,i}} q\, dA
  = -\oint_{\partial \Omega_{j,i}} \mathbf{F} \cdot \hat{\mathbf{n}}\, ds
$$

The physical meaning is clear: **the rate of change of $q$ inside a cell
equals the net flux through its faces**.  This is exact — no approximation
has been made yet.

### Discretizing: cell averages and face fluxes

Now we introduce approximations.  Define the **cell average**:

$$
\bar{q}_{j,i} = \frac{1}{|\Omega_{j,i}|} \int_{\Omega_{j,i}} q\, dA
$$

For a rectangular cell with sides $\Delta x$ and $\Delta y$, the boundary
integral decomposes into four face contributions — east, west, north,
south.  Approximating the flux as constant along each face:

$$
\oint_{\partial \Omega} \mathbf{F} \cdot \hat{\mathbf{n}}\, ds
  \approx
    \underbrace{F^x_{j,\, i+\frac{1}{2}}}_{\text{east}} \Delta y
  - \underbrace{F^x_{j,\, i-\frac{1}{2}}}_{\text{west}} \Delta y
  + \underbrace{F^y_{j+\frac{1}{2},\, i}}_{\text{north}} \Delta x
  - \underbrace{F^y_{j-\frac{1}{2},\, i}}_{\text{south}} \Delta x
$$

Dividing by the cell area $\Delta x \Delta y$:

$$
\frac{d\bar{q}_{j,i}}{dt}
  = -\frac{F^x_{j,\, i+\frac{1}{2}} - F^x_{j,\, i-\frac{1}{2}}}{\Delta x}
    -\frac{F^y_{j+\frac{1}{2},\, i} - F^y_{j-\frac{1}{2},\, i}}{\Delta y}
$$

This is the **semi-discrete finite-volume equation**: continuous in time,
discrete in space.  Two things are now apparent:

1. **Cell averages** ($\bar{q}_{j,i}$) naturally live at cell centres.
2. **Face fluxes** ($F^x_{j,i+1/2}$, $F^y_{j+1/2,i}$) naturally live at cell faces.

This is *exactly* the Arakawa C-grid staggering — it is not an arbitrary
choice but a direct consequence of the Gauss divergence theorem.

---

## 3. The Arakawa C-Grid

### Why stagger the grid?

The finite-volume derivation tells us that scalar quantities and fluxes
live at geometrically different locations: centres vs. faces.  Storing
them at the same location (a co-located or "A-grid") introduces
computational modes and requires explicit filtering.  The **Arakawa C-grid**
respects the natural staggering by assigning:

- **Scalars** (thickness, pressure, tracers) to cell **centres** (T-points)
- **x-fluxes** and x-velocity to east **faces** (U-points)
- **y-fluxes** and y-velocity to north **faces** (V-points)
- **Vorticity** and related quantities to cell **corners** (X-points)

```
   V[j,i] ──── X[j,i] ──── V[j,i+1]
     │                        │
     │        T[j,i]          │
     │                        │
   V[j-1,i] ── X[j-1,i] ── V[j-1,i+1]

   U[j,i-1]    U[j,i]      U[j,i+1]
```

| Point | Position | Physical role |
|-------|----------|---------------|
| **T** | $(j,\, i)$ — cell centre | Scalars: thickness $h$, pressure $p$, tracers |
| **U** | $(j,\, i+\tfrac{1}{2})$ — east face | x-velocity $u$, east-face flux $F^x$ |
| **V** | $(j+\tfrac{1}{2},\, i)$ — north face | y-velocity $v$, north-face flux $F^y$ |
| **X** | $(j+\tfrac{1}{2},\, i+\tfrac{1}{2})$ — NE corner | Vorticity $\zeta$, potential vorticity $q$ |

### The discrete divergence theorem on a C-grid

With this layout, the semi-discrete equation from Section 2 maps directly
to array operations.  The discrete divergence of the velocity field at
T-point $(j, i)$ reads the four surrounding face values:

$$
(\nabla \cdot \mathbf{u})_{j,i}
  = \frac{u_{j,\, i+\frac{1}{2}} - u_{j,\, i-\frac{1}{2}}}{\Delta x}
  + \frac{v_{j+\frac{1}{2},\, i} - v_{j-\frac{1}{2},\, i}}{\Delta y}
$$

Each term is a **backward difference** of a face quantity returning to
the cell centre — this is `diff_x_bwd_2d(u) / dx + diff_y_bwd_2d(v) / dy`.

Similarly, the **curl** (relative vorticity) at X-point $(j+1/2, i+1/2)$
reads the four surrounding face values:

$$
\zeta_{j+\frac{1}{2},\, i+\frac{1}{2}}
  = \frac{v_{j+\frac{1}{2},\, i+1} - v_{j+\frac{1}{2},\, i}}{\Delta x}
  - \frac{u_{j+1,\, i+\frac{1}{2}} - u_{j,\, i+\frac{1}{2}}}{\Delta y}
$$

Each term is a **forward difference** of a face quantity advancing to
the corner — this is `diff_x_fwd_2d(v) / dx - diff_y_fwd_2d(u) / dy`.

:::{note} Discrete conservation
Because the east-face flux of cell $(j,i)$ is identically the west-face
flux of cell $(j, i+1)$, the finite-volume scheme conserves the
total integral of $q$ over the domain to machine precision (up to
boundary fluxes).  This **telescoping property** is built into the
C-grid layout and is preserved by the raw stencils.
:::

### Ghost cells and the `interior()` helper

In `gridax`, every array has the same total shape `[Ny, Nx]` with a
one-cell ghost ring on each side.  The physical interior occupies
`[1:-1, 1:-1]`.  Raw stencils return arrays of shape `(Ny-2, Nx-2)` —
the interior only.  The `interior(values, like)` helper pads the result
back to full grid shape with zeros in the ghost ring.

For full details on ghost-cell layout and boundary interactions, see the
[C-Grid Discretization](cgrid_discretization.md) page.

---

(derivative-stack)=
## 4. The derivative stack

gridax has one derivative stack for collocated and staggered fields, in four layers. The numbering is the one `docs/masks.md` and `AGENTS.md` use, with a Layer 0 below the raw stencils.

| Layer | Holds | Collocated | Staggered (C-grid) |
|---|---|---|---|
| **0 — coefficients** | offsets and exact coefficients | `gridax.stencils.difference_offsets`, `difference_coeffs`, `nonuniform_coeffs` | the same with `method="staggered"`, `bias=` and `data="cell"`; `weno_stencils`, `optimal_weights`, `smoothness_matrix` |
| **1 — raw stencils** | index arithmetic, no metric, no mask | applied inside Layer 2 | the 32 named stencils on this page; `staggered_difference`, `staggered_average`; the `linear_*` and `weno_*` reconstructions |
| **2 — functional helpers** | metric-scaled, mask-free | `gridax.analysis`: `difference`, `gradient_2d`, `divergence_2d`, `curl_2d`, `laplacian_2d`, … | `diffusion_2d`, `arakawa_jacobian`, `kinetic_energy`, `shear_strain`, …; `gridax.advection.reconstruct` (with an optional mask) |
| **3 — class operators** | carry a grid and a mask | none | `Difference2D`, `Divergence2D`, `Vorticity2D`, `Reconstruction2D`, `Advection2D`, `SphericalDifference2D`, … |

A C-grid helper exists at Layer 2 only when it is more than one line of Layer 1; the divergence and the curl are `Difference2D.divergence` and `.curl` (the free functions `divergence_2d` and `relative_vorticity_cgrid` are deprecated).

### Layer 0: one solver for every stencil

Write $f$ for a smooth function, $h$ for the grid step, $s_1, \dots, s_K$ for a stencil's offsets in units of $h$, $c_1, \dots, c_K$ for its coefficients and $n$ for the derivative order ($n = 0$ for a value). The stencil approximates $f^{(n)}(0) \approx h^{-n} \sum_k c_k\, \sigma_k$, where $\sigma_k$ is sample $k$, and its coefficients solve the $K$ conditions

$$\sum_{k=1}^{K} c_k\, \mu_m(s_k) = n!\, \delta_{mn}, \qquad m = 0, 1, \dots, K - 1,$$ (stencil-moments)

with $\delta_{mn}$ equal to 1 when $m = n$ and 0 otherwise. The moment $\mu_m(s)$ is what a sample at $s$ sees of $x^m$: $\mu_m(s) = s^m$ when the samples are point values, and the average over the unit cell centred on $s$ when they are cell averages,

$$\mu_m(s) = \frac{(s + 1/2)^{m+1} - (s - 1/2)^{m+1}}{m + 1}.$$ (cell-moment)

`difference_coeffs` solves {eq}`stencil-moments` in exact rational arithmetic and rounds each coefficient once to float64. Integer offsets give collocated stencils, as `gridax.analysis` uses them; half-integer offsets give staggered ones, whose result sits half a cell from every input. The C-grid stencils on this page are the two-point case, $s = (-1/2, 1/2)$:

| Offsets | Samples | $n$ | Coefficients | Where |
|---|---|---|---|---|
| $\pm 1/2$ | point or cell | 1 | $(-1, 1)$ | `diff_*` |
| $\pm 1/2$ | point or cell | 0 | $(1/2, 1/2)$ | `avg_*`, `linear_2pts` |
| $\pm 1/2, \pm 3/2$ | point | 1 | $(1, -27, 27, -1)/24$ | `staggered_difference(..., accuracy=3)` |
| $\pm 1/2, \pm 3/2$ | point | 0 | $(-1, 9, 9, -1)/16$ | `staggered_average(..., accuracy=3)` |
| $\pm 1/2, \pm 3/2$ | cell | 0 | $(-1, 7, 7, -1)/12$ | `linear_4pts` |
| $-5/2, \dots, 3/2$ | cell | 0 | $(2, -13, 47, 27, -3)/60$ | `linear_5pts_left`, the WENO-5 limit |

Point values and cell averages give the same two-point stencils, which is why the named stencils serve both, and different ones from three samples on. Use the point weights for C-grid fields that are point values (velocities, pressure, Coriolis terms) and the cell weights for a finite-volume tracer; the wrong ones drop fourth order to second (`packages/gridax/tests/test_stencil_cell_averages.py`). The WENO sub-stencil coefficients, optimal weights and Jiang–Shu smoothness matrices of `weno.py` are generated the same way, exactly (`weno_stencils`, `optimal_weights`, `smoothness_matrix`).

### Two wall rules

A stencil wider than two points reads past the one-cell ghost ring on the faces next to a wall, and the two families treat those faces differently.

- **Centred operators shift inwards.** `staggered_difference` and `staggered_average` keep their width and move the stencil into the domain, so a fourth-order stencil has order 3 on that face instead of dropping to 2.
- **Upwind reconstructions narrow.** `gridax.advection.reconstruct` gives each face the widest tier of its scheme that fits (WENO-5, then WENO-3, then first-order upwind), so it never reads a cell downwind of the face. With a mask, the coast ends the tiers in the same way.

### Example

```python
import gridax
import gridax.advection as gadv
import gridax.analysis as ga
import gridax.stencils as gs

grid = gridax.CartesianGrid2D.from_interior(60, 40, 120.0e3, 40.0e3)  # Ny, Nx = 42, 62

# Layer 0: the two-point staggered stencil → (-1.0, 1.0)
coeffs = gs.difference_coeffs(gs.difference_offsets(method="staggered"), 1)
# Layer 1: h[j, i+1] - h[j, i], no metric → (40, 60)
raw = gs.diff_x_fwd_2d(h)
# Layer 1, fourth order: (h[i-1] - 27 h[i] + 27 h[i+1] - h[i+2]) / 24 → (40, 60)
raw4 = gs.staggered_difference(h, axis="x", side="fwd", ndim=2, accuracy=3)
# Layer 2, collocated: dh/dx on the same points → (42, 62)
dh_dx_collocated = ga.difference(h, axis=-1, step=grid.dx)
# Layer 2, advection: the WENO-5 flux on the east faces → (42, 62)
flux = gadv.reconstruct(h, u, axis=-1, scheme=gadv.WENO(5))
# Layer 3: dh/dx at U points, zeroed at dry faces → (42, 62)
dh_dx = gridax.Difference2D(grid=grid, mask=mask).diff_x_T_to_U(h)
```

The rest of this page documents **Layer 1**, the raw stencils. They perform no metric scaling (no division by $\Delta x$, $R\cos\varphi\,\Delta\lambda$, etc.) and no ghost-ring padding, and return interior-sized arrays that the caller can scale by any coordinate metric, so they are reused across Cartesian, spherical and curvilinear coordinates.

---

## 5. Difference Stencils

### Forward differences

A forward difference shifts the output one half-step in the positive
direction.  These map centre-points to face/corner-points.

**Forward x-difference (T → U or V → X):**

$$
\Delta_x\, h_{j,\, i+\frac{1}{2}} = h_{j,\, i+1} - h_{j,\, i}
$$

```python
def diff_x_fwd_2d(h):
    return h[1:-1, 2:] - h[1:-1, 1:-1]
```

**Forward y-difference (T → V or U → X):**

$$
\Delta_y\, h_{j+\frac{1}{2},\, i} = h_{j+1,\, i} - h_{j,\, i}
$$

```python
def diff_y_fwd_2d(h):
    return h[2:, 1:-1] - h[1:-1, 1:-1]
```

### Backward differences

A backward difference shifts the output one half-step in the negative
direction.  These map face/corner-points back to centre-points.

**Backward x-difference (U → T or X → V):**

$$
\Delta_x\, h_{j,\, i} = h_{j,\, i+\frac{1}{2}} - h_{j,\, i-\frac{1}{2}}
$$

```python
def diff_x_bwd_2d(h):
    return h[1:-1, 1:-1] - h[1:-1, :-2]
```

**Backward y-difference (V → T or X → U):**

$$
\Delta_y\, h_{j,\, i} = h_{j+\frac{1}{2},\, i} - h_{j-\frac{1}{2},\, i}
$$

```python
def diff_y_bwd_2d(h):
    return h[1:-1, 1:-1] - h[:-2, 1:-1]
```

### Composing second derivatives

The Laplacian is a forward-then-backward composition:

$$
\frac{\partial^2 h}{\partial x^2} \approx
  \frac{\Delta_x^{\text{fwd}} h - \Delta_x^{\text{bwd}} h}{\Delta x^2}
$$

```python
d2h_dx2 = (diff_x_fwd_2d(h) - diff_x_bwd_2d(h)) / dx**2
d2h_dy2 = (diff_y_fwd_2d(h) - diff_y_bwd_2d(h)) / dy**2
laplacian = d2h_dx2 + d2h_dy2
```

This is exactly how `Difference2D.laplacian` is implemented internally.

---

## 6. Averaging Stencils

Interpolation between grid locations uses arithmetic averages of
nearest neighbours.  Like the difference stencils, these return
interior-sized arrays with no ghost-ring padding.

### 2-point averages

**Forward x-average (T → U or V → X):**

$$
\bar{h}_{j,\, i+\frac{1}{2}} = \tfrac{1}{2}(h_{j,\, i} + h_{j,\, i+1})
$$

```python
def avg_x_fwd_2d(h):
    return 0.5 * (h[1:-1, 1:-1] + h[1:-1, 2:])
```

**Backward x-average (U → T or X → V):**

$$
\bar{h}_{j,\, i} = \tfrac{1}{2}(h_{j,\, i-\frac{1}{2}} + h_{j,\, i+\frac{1}{2}})
$$

```python
def avg_x_bwd_2d(h):
    return 0.5 * (h[1:-1, 1:-1] + h[1:-1, :-2])
```

The y-direction analogues `avg_y_fwd_2d` and `avg_y_bwd_2d` follow the same
pattern along the j-axis.

### 4-point bilinear averages

For diagonal transfers (T ↔ X, U ↔ V), a **4-point bilinear average**
is used:

**T → X (NE corner average):**

$$
\bar{h}_{j+\frac{1}{2},\, i+\frac{1}{2}} = \tfrac{1}{4}
  (h_{j,i} + h_{j,i+1} + h_{j+1,i} + h_{j+1,i+1})
$$

```python
def avg_xfwd_yfwd_2d(h):
    return 0.25 * (h[1:-1, 1:-1] + h[1:-1, 2:]
                   + h[2:, 1:-1] + h[2:, 2:])
```

**X → T (SW corner average):**

$$
\bar{q}_{j,i} = \tfrac{1}{4}
  (q_{j+\frac{1}{2},i+\frac{1}{2}} + q_{j-\frac{1}{2},i+\frac{1}{2}}
 + q_{j+\frac{1}{2},i-\frac{1}{2}} + q_{j-\frac{1}{2},i-\frac{1}{2}})
$$

**U → V** (`avg_xbwd_yfwd_2d`) and **V → U** (`avg_xfwd_ybwd_2d`) use the same
4-point pattern but with mixed forward/backward shifts in each direction.

---

## 7. Complete Reference

### Difference stencils

| Function | Formula | Maps |
|----------|---------|------|
| `diff_x_fwd_2d` | $h_{j,i+1} - h_{j,i}$ | T→U, V→X |
| `diff_y_fwd_2d` | $h_{j+1,i} - h_{j,i}$ | T→V, U→X |
| `diff_x_bwd_2d` | $h_{j,i+1/2} - h_{j,i-1/2}$ | U→T, X→V |
| `diff_y_bwd_2d` | $h_{j+1/2,i} - h_{j-1/2,i}$ | V→T, X→U |
| `diff_x_fwd_1d` | $h_{i+1} - h_i$ | T→U (1D) |
| `diff_x_bwd_1d` | $h_{i+1/2} - h_{i-1/2}$ | U→T (1D) |
| `diff_x_fwd_3d` | $h_{k,j,i+1} - h_{k,j,i}$ | T→U (3D) |
| `diff_y_fwd_3d` | $h_{k,j+1,i} - h_{k,j,i}$ | T→V (3D) |
| `diff_x_bwd_3d` | $h_{k,j,i+1/2} - h_{k,j,i-1/2}$ | U→T (3D) |
| `diff_y_bwd_3d` | $h_{k,j+1/2,i} - h_{k,j-1/2,i}$ | V→T (3D) |

### Averaging stencils

| Function | Formula | Maps |
|----------|---------|------|
| `avg_x_fwd_2d` | $\frac{1}{2}(h_{j,i} + h_{j,i+1})$ | T→U, V→X |
| `avg_y_fwd_2d` | $\frac{1}{2}(h_{j,i} + h_{j+1,i})$ | T→V, U→X |
| `avg_x_bwd_2d` | $\frac{1}{2}(h_{j,i-1/2} + h_{j,i+1/2})$ | U→T, X→V |
| `avg_y_bwd_2d` | $\frac{1}{2}(h_{j-1/2,i} + h_{j+1/2,i})$ | V→T, X→U |
| `avg_xfwd_yfwd_2d` | $\frac{1}{4}\sum$ (NE quad) | T→X |
| `avg_xbwd_ybwd_2d` | $\frac{1}{4}\sum$ (SW quad) | X→T |
| `avg_xbwd_yfwd_2d` | $\frac{1}{4}\sum$ (bwd-x, fwd-y) | U→V |
| `avg_xfwd_ybwd_2d` | $\frac{1}{4}\sum$ (fwd-x, bwd-y) | V→U |
| `avg_x_fwd_1d` | $\frac{1}{2}(h_i + h_{i+1})$ | T→U (1D) |
| `avg_x_bwd_1d` | $\frac{1}{2}(h_{i-1/2} + h_{i+1/2})$ | U→T (1D) |
| `avg_x_fwd_3d` | $\frac{1}{2}(h_{k,j,i} + h_{k,j,i+1})$ | T→U (3D) |
| `avg_y_fwd_3d` | $\frac{1}{2}(h_{k,j,i} + h_{k,j+1,i})$ | T→V (3D) |
| `avg_x_bwd_3d` | $\frac{1}{2}(h_{k,j,i-1/2} + h_{k,j,i+1/2})$ | U→T (3D) |
| `avg_y_bwd_3d` | $\frac{1}{2}(h_{k,j-1/2,i} + h_{k,j+1/2,i})$ | V→T (3D) |

---

## 8. Usage: Building Custom Operators

The raw stencils are designed as building blocks for custom operators,
especially when working with non-Cartesian coordinate systems where the
standard `Difference2D` / `Interpolation2D` classes (which divide by
scalar `dx`, `dy`) are not appropriate.

### Example: Custom spherical gradient

On a latitude-longitude grid, the x-metric depends on latitude:

$$
\frac{\partial h}{\partial x}\bigg|_{j,i+\frac{1}{2}}
  = \frac{h_{j,i+1} - h_{j,i}}{R\cos\varphi_j\,\Delta\lambda}
$$

```python
from gridax import diff_x_fwd_2d, interior

# Pure index arithmetic — same for any coordinate system
raw = diff_x_fwd_2d(h)

# Apply spherical metric (array, not scalar!)
scaled = raw / (R * cos_lat[1:-1, 1:-1] * dlon)

# Pad back to full grid shape with zero ghost ring
result = interior(scaled, h)
```

### Example: Strain from raw stencils

```python
from gridax import diff_x_fwd_2d, diff_y_fwd_2d, diff_x_bwd_2d, diff_y_bwd_2d, interior

# Shear strain at X-points: dv/dx + du/dy
shear = interior(
    diff_x_fwd_2d(v) / dx + diff_y_fwd_2d(u) / dy,
    u,
)

# Tensor strain at T-points: du/dx - dv/dy
tensor = interior(
    diff_x_bwd_2d(u) / dx - diff_y_bwd_2d(v) / dy,
    u,
)
```

---

## Design Principles

- **Pure functions**: Every stencil is a stateless function of a single
  array argument.  No grid objects, no side effects.
- **JAX-compatible**: All stencils work with `jax.jit`, `jax.vmap`, and
  `jax.grad` out of the box.
- **No metric scaling**: The caller chooses whether to divide by `dx`,
  `R·cos(φ)·dλ`, or any other metric.  This keeps the stencils
  coordinate-system-agnostic.
- **Interior-only output**: Results have shape `(Ny-2, Nx-2)` — the
  interior without the ghost ring.  Use `interior(result, like)` to
  pad back to full grid shape.

---

## References

- Arakawa, A. & Lamb, V. R. (1977). Computational design of the basic
  dynamical processes of the UCLA general circulation model. *Methods
  in Computational Physics*, 17, 173-265.
- Sadourny, R. (1975). The dynamics of finite-difference models of the
  shallow-water equations. *J. Atmos. Sci.*, 32, 680-689.
- LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*.
  Cambridge University Press.
