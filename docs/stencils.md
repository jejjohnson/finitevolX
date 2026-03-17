# Raw Stencil Primitives

This page explains the theory behind the raw stencil functions in `finitevolx`,
connecting the continuous finite-volume formulation to the discrete index
arithmetic that the library implements.

---

## From Continuous to Discrete

### The finite-volume idea

Geophysical fluid dynamics models solve conservation laws of the form

$$
\frac{\partial q}{\partial t} + \nabla \cdot \mathbf{F}(q) = 0
$$

where $q$ is a conserved quantity (mass, momentum, tracer) and $\mathbf{F}$
is its flux.  The **finite-volume method** integrates this equation over each
grid cell $\Omega_{j,i}$ and applies the **Gauss divergence theorem**:

$$
\frac{d}{dt}\int_{\Omega_{j,i}} q\, dA
  = -\oint_{\partial \Omega_{j,i}} \mathbf{F} \cdot \hat{n}\, ds
$$

The volume integral becomes a cell average; the surface integral becomes a
sum of **face fluxes**.  This is why Arakawa C-grids store scalars at cell
centres (T-points) and fluxes at cell faces (U- and V-points) — the grid
layout directly mirrors the mathematical structure of finite-volume
discretization.

### Arakawa C-grid staggering

On an Arakawa C-grid, the four variable locations are:

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
| **U** | $(j,\, i+\tfrac{1}{2})$ — east face | x-velocity $u$, east-face flux |
| **V** | $(j+\tfrac{1}{2},\, i)$ — north face | y-velocity $v$, north-face flux |
| **X** | $(j+\tfrac{1}{2},\, i+\tfrac{1}{2})$ — NE corner | Vorticity $\zeta$, potential vorticity $q$ |

The discrete approximation to the divergence theorem for cell $(j, i)$ is:

$$
\frac{\partial \bar{q}_{j,i}}{\partial t}
  = -\frac{F^x_{j,\, i+\frac{1}{2}} - F^x_{j,\, i-\frac{1}{2}}}{\Delta x}
    -\frac{F^y_{j+\frac{1}{2},\, i} - F^y_{j-\frac{1}{2},\, i}}{\Delta y}
$$

where $F^x$ and $F^y$ are the east-face and north-face fluxes (at U- and
V-points respectively).

---

## The 3-Layer Architecture

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

## Difference Stencils

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

## Averaging Stencils

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

## Complete Reference

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

## Usage: Building Custom Operators

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
