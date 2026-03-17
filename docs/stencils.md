# Raw Stencil Primitives

This page explains the theory behind the raw stencil functions in `finitevolx`,
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
2. **Face fluxes** ($F^x_{j,i+½}$, $F^y_{j+½,i}$) naturally live at cell faces.

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
the cell centre — this is `diff_x_bwd(u) / dx + diff_y_bwd(v) / dy`.

Similarly, the **curl** (relative vorticity) at X-point $(j+½, i+½)$
reads the four surrounding face values:

$$
\zeta_{j+\frac{1}{2},\, i+\frac{1}{2}}
  = \frac{v_{j+\frac{1}{2},\, i+1} - v_{j+\frac{1}{2},\, i}}{\Delta x}
  - \frac{u_{j+1,\, i+\frac{1}{2}} - u_{j,\, i+\frac{1}{2}}}{\Delta y}
$$

Each term is a **forward difference** of a face quantity advancing to
the corner — this is `diff_x_fwd(v) / dx - diff_y_fwd(u) / dy`.

!!! note "Discrete conservation"
    Because the east-face flux of cell $(j,i)$ is identically the west-face
    flux of cell $(j, i+1)$, the finite-volume scheme conserves the
    total integral of $q$ over the domain to machine precision (up to
    boundary fluxes).  This **telescoping property** is built into the
    C-grid layout and is preserved by the raw stencils.

### Ghost cells and the `interior()` helper

In `finitevolx`, every array has the same total shape `[Ny, Nx]` with a
one-cell ghost ring on each side.  The physical interior occupies
`[1:-1, 1:-1]`.  Raw stencils return arrays of shape `(Ny-2, Nx-2)` —
the interior only.  The `interior(values, like)` helper pads the result
back to full grid shape with zeros in the ghost ring.

For full details on ghost-cell layout and boundary interactions, see the
[C-Grid Discretization](cgrid_discretization.md) page.

---

## 4. The 3-Layer Architecture

`finitevolx` organises its spatial operators into three layers:

| Layer | What | Example |
|-------|------|---------|
| **1 — Raw stencils** | Pure index arithmetic, no scaling | `diff_x_fwd(h)` returns `h[j, i+1] − h[j, i]` |
| **2 — Scaled primitives** | Layer 1 + metric scaling | `Difference2D.diff_x_T_to_U(h)` returns `diff_x_fwd(h) / dx` |
| **3 — Compound operators** | Compose Layer 2 | `Difference2D.divergence(u, v)` = `diff_x_U_to_T(u) + diff_y_V_to_T(v)` |

**Layer 1** — the raw stencils documented here — is the foundation.
These functions perform no metric scaling (no division by $\Delta x$,
$R\cos\varphi\,\Delta\lambda$, etc.) and no ghost-ring padding.  They
return interior-sized arrays that the caller can scale by any coordinate
metric, making them reusable across Cartesian, spherical, cylindrical,
and curvilinear coordinate systems.

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
def diff_x_fwd(h):
    return h[1:-1, 2:] - h[1:-1, 1:-1]
```

**Forward y-difference (T → V or U → X):**

$$
\Delta_y\, h_{j+\frac{1}{2},\, i} = h_{j+1,\, i} - h_{j,\, i}
$$

```python
def diff_y_fwd(h):
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
def diff_x_bwd(h):
    return h[1:-1, 1:-1] - h[1:-1, :-2]
```

**Backward y-difference (V → T or X → U):**

$$
\Delta_y\, h_{j,\, i} = h_{j+\frac{1}{2},\, i} - h_{j-\frac{1}{2},\, i}
$$

```python
def diff_y_bwd(h):
    return h[1:-1, 1:-1] - h[:-2, 1:-1]
```

### Composing second derivatives

The Laplacian is a forward-then-backward composition:

$$
\frac{\partial^2 h}{\partial x^2} \approx
  \frac{\Delta_x^{\text{fwd}} h - \Delta_x^{\text{bwd}} h}{\Delta x^2}
$$

```python
d2h_dx2 = (diff_x_fwd(h) - diff_x_bwd(h)) / dx**2
d2h_dy2 = (diff_y_fwd(h) - diff_y_bwd(h)) / dy**2
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
def avg_x_fwd(h):
    return 0.5 * (h[1:-1, 1:-1] + h[1:-1, 2:])
```

**Backward x-average (U → T or X → V):**

$$
\bar{h}_{j,\, i} = \tfrac{1}{2}(h_{j,\, i-\frac{1}{2}} + h_{j,\, i+\frac{1}{2}})
$$

```python
def avg_x_bwd(h):
    return 0.5 * (h[1:-1, 1:-1] + h[1:-1, :-2])
```

The y-direction analogues `avg_y_fwd` and `avg_y_bwd` follow the same
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
def avg_xy_fwd(h):
    return 0.25 * (h[1:-1, 1:-1] + h[1:-1, 2:]
                   + h[2:, 1:-1] + h[2:, 2:])
```

**X → T (SW corner average):**

$$
\bar{q}_{j,i} = \tfrac{1}{4}
  (q_{j+\frac{1}{2},i+\frac{1}{2}} + q_{j-\frac{1}{2},i+\frac{1}{2}}
 + q_{j+\frac{1}{2},i-\frac{1}{2}} + q_{j-\frac{1}{2},i-\frac{1}{2}})
$$

**U → V** (`avg_xbwd_yfwd`) and **V → U** (`avg_xfwd_ybwd`) use the same
4-point pattern but with mixed forward/backward shifts in each direction.

---

## 7. Complete Reference

### Difference stencils

| Function | Formula | Maps |
|----------|---------|------|
| `diff_x_fwd` | $h_{j,i+1} - h_{j,i}$ | T→U, V→X |
| `diff_y_fwd` | $h_{j+1,i} - h_{j,i}$ | T→V, U→X |
| `diff_x_bwd` | $h_{j,i+½} - h_{j,i-½}$ | U→T, X→V |
| `diff_y_bwd` | $h_{j+½,i} - h_{j-½,i}$ | V→T, X→U |
| `diff_x_fwd_1d` | $h_{i+1} - h_i$ | T→U (1D) |
| `diff_x_bwd_1d` | $h_{i+½} - h_{i-½}$ | U→T (1D) |
| `diff_x_fwd_3d` | $h_{k,j,i+1} - h_{k,j,i}$ | T→U (3D) |
| `diff_y_fwd_3d` | $h_{k,j+1,i} - h_{k,j,i}$ | T→V (3D) |
| `diff_x_bwd_3d` | $h_{k,j,i+½} - h_{k,j,i-½}$ | U→T (3D) |
| `diff_y_bwd_3d` | $h_{k,j+½,i} - h_{k,j-½,i}$ | V→T (3D) |

### Averaging stencils

| Function | Formula | Maps |
|----------|---------|------|
| `avg_x_fwd` | $\frac{1}{2}(h_{j,i} + h_{j,i+1})$ | T→U, V→X |
| `avg_y_fwd` | $\frac{1}{2}(h_{j,i} + h_{j+1,i})$ | T→V, U→X |
| `avg_x_bwd` | $\frac{1}{2}(h_{j,i-½} + h_{j,i+½})$ | U→T, X→V |
| `avg_y_bwd` | $\frac{1}{2}(h_{j-½,i} + h_{j+½,i})$ | V→T, X→U |
| `avg_xy_fwd` | $\frac{1}{4}\sum$ (NE quad) | T→X |
| `avg_xy_bwd` | $\frac{1}{4}\sum$ (SW quad) | X→T |
| `avg_xbwd_yfwd` | $\frac{1}{4}\sum$ (bwd-x, fwd-y) | U→V |
| `avg_xfwd_ybwd` | $\frac{1}{4}\sum$ (fwd-x, bwd-y) | V→U |
| `avg_x_fwd_1d` | $\frac{1}{2}(h_i + h_{i+1})$ | T→U (1D) |
| `avg_x_bwd_1d` | $\frac{1}{2}(h_{i-½} + h_{i+½})$ | U→T (1D) |
| `avg_x_fwd_3d` | $\frac{1}{2}(h_{k,j,i} + h_{k,j,i+1})$ | T→U (3D) |
| `avg_y_fwd_3d` | $\frac{1}{2}(h_{k,j,i} + h_{k,j+1,i})$ | T→V (3D) |
| `avg_x_bwd_3d` | $\frac{1}{2}(h_{k,j,i-½} + h_{k,j,i+½})$ | U→T (3D) |
| `avg_y_bwd_3d` | $\frac{1}{2}(h_{k,j-½,i} + h_{k,j+½,i})$ | V→T (3D) |

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
from finitevolx import diff_x_fwd, interior

# Pure index arithmetic — same for any coordinate system
raw = diff_x_fwd(h)

# Apply spherical metric (array, not scalar!)
scaled = raw / (R * cos_lat[1:-1, 1:-1] * dlon)

# Pad back to full grid shape with zero ghost ring
result = interior(scaled, h)
```

### Example: Strain from raw stencils

```python
from finitevolx import diff_x_fwd, diff_y_fwd, diff_x_bwd, diff_y_bwd, interior

# Shear strain at X-points: dv/dx + du/dy
shear = interior(
    diff_x_fwd(v) / dx + diff_y_fwd(u) / dy,
    u,
)

# Tensor strain at T-points: du/dx - dv/dy
tensor = interior(
    diff_x_bwd(u) / dx - diff_y_bwd(v) / dy,
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
