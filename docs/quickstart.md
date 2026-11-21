---
title: Quickstart
short_title: Quickstart
description: Install gridax, carry a tracer round an island on a masked C-grid with NumPy alone, and run the same tendency under jax.jit and jax.grad.
---

This page installs gridax, runs the example from the README one step at a time, says where its closed walls need care, and then runs the same tendency under `jax.jit` and `jax.grad`. It uses the notation of the README and of the [derivative stack](stencils.md#derivative-stack).

## Install

gridax is not on PyPI yet ([#426](https://github.com/jejjohnson/gridax/issues/426)), and neither is geophysix, which it imports for its physical constants. uv reads `[tool.uv.sources]` only from the project it installs, so a project that uses gridax names both sources in its own `pyproject.toml`:

```toml
[project]
dependencies = ["gridax"]

[tool.uv.sources]
gridax = { git = "https://github.com/jejjohnson/gridax", tag = "gridax-v0.2.0", subdirectory = "packages/gridax" }
geophysix = { git = "https://github.com/jejjohnson/geophysix", tag = "v0.0.3" }
```

That install brings in NumPy, SciPy, jaxtyping and geophysix, and nothing else. gridax has no extras: with JAX installed beside it, every grid, mask and operator is also a JAX pytree, and the same calls take JAX arrays.

## A tracer round an island

We solve the semi-discrete finite-volume equation for a tracer $\bar{q}$ held at the T points,

$$\frac{\mathrm{d}\bar{q}_{j,i}}{\mathrm{d}t} = -\frac{F^x_{j,i+1/2} - F^x_{j,i-1/2}}{\Delta x} - \frac{F^y_{j+1/2,i} - F^y_{j-1/2,i}}{\Delta y}$$

where $F^x = u \hat{q}$ and $F^y = v \hat{q}$ are the fluxes through the east and north faces, $\hat{q}$ is the face value that WENO-5 reconstructs from the cells upwind of the face, and $\Delta x$ and $\Delta y$ are the grid spacings. Every array has the shape $(N_y, N_x)$ of the grid with its ghost ring.

### The grid and the mask

`CartesianGrid2D.from_interior` takes the interior size and the domain lengths and adds one ghost ring, so a 32 × 32 interior gives arrays of shape $(34, 34)$. We put the origin on the south-west corner of the walls, so T point $(j, i)$ sits at $x_i = (i - 1/2) \Delta x$, $y_j = (j - 1/2) \Delta y$, and the walls at $x = 0, 1$ and $y = 0, 1$ (`gridax.t_point_coordinates` measures from T point $(0, 0)$ instead, half a cell further out). The mask starts from a boolean array of wet T points, and `Mask2D.from_mask` derives the wet U, V and X points from it, with the coast classification and the stencil capability that `Advection2D` reads near land.

```python
import gridax
import numpy as np

# Shapes: N_y = N_x = 34, a 32 × 32 interior and one ghost ring

# today (gridax)
grid = gridax.CartesianGrid2D.from_interior(32, 32, Lx=1.0, Ly=1.0)
# (x_i, y_j), the T-point coordinates
x, y = np.meshgrid(
    (np.arange(grid.Nx) - 0.5) * grid.dx, (np.arange(grid.Ny) - 0.5) * grid.dy
)  # → (N_y, N_x), (N_y, N_x)
# a closed basin with a round island in the middle
wet = (x - 0.5) ** 2 + (y - 0.5) ** 2 > 0.15**2  # → (N_y, N_x)
wet[:2, :] = wet[-2:, :] = wet[:, :2] = wet[:, -2:] = False
# today (gridax)
mask = gridax.Mask2D.from_mask(wet)
```

The line that sets `wet[:2, :]` and its three partners makes the ghost ring and the first interior ring dry, which closes the basin with land. Why the first interior ring has to be land is the subject of the section on the walls below.

### The flow and the tracer

The flow is one gyre with streamfunction $\psi = \sin(\pi x) \sin(\pi y) / \pi$ at the T points. `Difference2D.grad_perp` turns it into $u = -\partial \psi / \partial y$ at the U points and $v = \partial \psi / \partial x$ at the V points. Without a mask that velocity is discretely non-divergent; with one, `grad_perp` zeroes it on the dry faces, so it no longer is. The flux form does not need it to be: a dry face carries no flux whatever the velocity next to it.

```python
# ψ = sin(πx) sin(πy) / π, one gyre
psi = np.sin(np.pi * x) * np.sin(np.pi * y) / np.pi  # → (N_y, N_x)
# today (gridax)
difference = gridax.Difference2D(grid=grid, mask=mask)
# today (gridax)
# u = −∂ψ/∂y at U points, v = ∂ψ/∂x at V points
u, v = difference.grad_perp(psi)  # → (N_y, N_x), (N_y, N_x)
# a Gaussian blob of tracer west of the island
q = np.exp(-((x - 0.25) ** 2 + (y - 0.5) ** 2) / 0.01) * wet  # → (N_y, N_x)
```

### Stepping and the budget

`Advection2D` is a Layer 3 operator: it holds the grid and the mask, and its call returns $\mathrm{d}\bar{q}/\mathrm{d}t$ on the T points. We step it with Heun's method and $\Delta t = 0.2 \Delta x$, a Courant number of at most 0.2 since $|u| \le 1$, and compare the tracer total $\sum_{j,i} A_{j,i} \bar{q}_{j,i}$ over the wet cells before and after, where $A_{j,i}$ are the cell areas.

```python
# today (gridax)
advection = gridax.Advection2D(grid=grid, mask=mask)
# today (gridax)
# ∑ A_ji q̄_ji over the wet cells
total = gridax.area_sum(q, grid, mask)  # → ()
dt = 0.2 * grid.dx
for _ in range(200):
    # Heun's method on dq̄/dt, with WENO-5 face values
    q1 = q + dt * advection(q, u, v, method="weno5")  # (N_y, N_x) → (N_y, N_x)
    q = 0.5 * (q + q1 + dt * advection(q1, u, v, method="weno5"))
# today (gridax)
change = gridax.area_sum(q, grid, mask) / total - 1.0  # → ()
print(f"{float(change):.1e} {float(q.max()):.3f} {float(q.min()):.1e}")
```

This prints `0.0e+00 0.807 -4.8e-14`. The tracer total after 200 steps equals the initial one to the last bit: each flux leaves one cell and enters the next, so the sum of the tendencies telescopes to the fluxes through the walls, and those are zero. The peak falls from 0.952 to 0.807, which is the numerical diffusion of WENO-5 on a grid with about six cells across the blob. The smallest value is $-4.8 \times 10^{-14}$: WENO is not positivity-preserving, but here its undershoot is at the level of rounding.

The time loop is written out to keep the example in one library. In a model, the steppers of `integrix.march` (`heun_step`, `rk3_ssp_step` and the others) take a right-hand side built on `advection` instead.

## Where the walls need care

`Advection2D` treats the four domain walls as closed by default (`wall="closed"`). It writes the tendency only on the strict interior `[2:-2, 2:-2]`, two cells in from each edge, so that no stencil reads a ghost flux. The cells of the first interior ring are therefore not updated, and if the flow crosses the faces between them and the strict interior, the tracer total drifts. In this example, with the dry ring removed and no mask, the sum of one WENO-5 tendency is $-3.9 \times 10^{-6}$ instead of zero.

There are two ways to close a basin that keep the budget:

- **Make the first interior ring dry**, as above. The mask then zeroes every flux through the faces next to it, and those cells hold no tracer.
- **Open the walls** with `wall="open"`, after filling the ghost ring with boundary conditions (a `BoundaryConditionSet`). The wall faces are then computed with first-order upwind from the ghost cells, the tendency is written on the full interior, and the flux stays single-valued, so mass is conserved exactly. `wall="open"` does not combine with a mask.

The [advection page](advection.md) has the stencil cascade near the coast and the full list of methods, and the [boundary conditions page](boundary_conditions.md) has the atoms that fill the ghost ring.

## The same tendency under `jit` and `grad`

Nothing in the example imports JAX. With JAX installed, the same operator takes JAX arrays, traces under `jax.jit`, and differentiates under `jax.grad`. Here we take the gradient of $\mathcal{J}(\bar{q}) = \sum_{j,i} A_{j,i} (\mathrm{d}\bar{q}_{j,i}/\mathrm{d}t)^2$ with respect to the tracer after the 200 steps above.

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Shapes: N_y = N_x = 34 (grid, mask, advection, psi, q, u and v from the block above)

# the same fields as JAX arrays
q_j, u_j, v_j = jnp.asarray(q), jnp.asarray(u), jnp.asarray(v)  # → (N_y, N_x) × 3


# today (gridax)
@jax.jit
def tendency(q, u, v):
    # dq̄/dt, with WENO-5 face values
    return advection(q, u, v, method="weno5")  # (N_y, N_x) → (N_y, N_x)


# today (gridax)
# J(q̄) = ∑ A (dq̄/dt)², a scalar to differentiate
def cost(q):
    return gridax.area_sum(tendency(q, u_j, v_j) ** 2, grid, mask)  # → ()


# ∂J/∂q̄, by reverse-mode autodiff through the WENO weights
grad_q = jax.grad(cost)(q_j)  # (N_y, N_x) → (N_y, N_x)
gap = jnp.abs(tendency(q_j, u_j, v_j) - advection(q, u, v, method="weno5")).max()
print(f"{float(gap):.1e} {grad_q.shape} {float(jnp.abs(grad_q).max()):.3f}")
```

This prints `5.8e-15 (34, 34) 0.162`. The jitted tendency on JAX arrays differs from the NumPy one by $5.8 \times 10^{-15}$, which is rounding (XLA fuses and reorders the arithmetic), and the gradient has the shape of the tracer, with a largest entry of 0.162.

## Next

- [C-grid discretisation](cgrid_discretization.md) has the staggering and the ghost-ring contract that every operator follows.
- [Masks](masks.md) has how the staggered masks and the stencil capability are derived.
- [The derivative stack](stencils.md#derivative-stack) has the four layers that this example used the top of.
