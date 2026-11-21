# Time Integration: Usage Guide

This page covers practical usage of time integration with finitevolX,
from quick-start examples to building custom diffrax solvers.

The generic time steppers (the pure step functions, the diffrax solvers and the multistep / split-explicit steppers) live in [integrix](https://github.com/jejjohnson/integrix) and are imported from `integrix.march`. The semi-Lagrangian scheme moved to `integrix.march` as well (#476); `solve_ocean_pde` moved to `sesmax_core.solve_pde` (#475).

---

## Quick Start

The fastest way to time-step a PDE: use a pure functional step function
inside a loop.

```python
import jax
import jax.numpy as jnp
import gridax_jax as fvx
from integrix.march import rk3_ssp_step

jax.config.update("jax_enable_x64", True)

# 1. Define your spatial RHS (any JAX pytree in, same pytree out)
grid = fvx.CartesianGrid2D.from_interior(64, 64, 1e4, 1e4)

def rhs(state):
    """Right-hand side: returns tendency with same pytree structure."""
    h, u, v = state
    # ... your spatial operators here ...
    return dh_dt, du_dt, dv_dt

# 2. Set up initial conditions
h0 = jnp.ones((grid.Ny, grid.Nx))
u0 = jnp.zeros((grid.Ny, grid.Nx))
v0 = jnp.zeros((grid.Ny, grid.Nx))
state = (h0, u0, v0)

# 3. Time-step with SSP-RK3
dt = 10.0  # seconds
for n in range(1000):
    state = rk3_ssp_step(state, rhs, dt)
```

All pure functional integrators accept **any JAX pytree** as state — tuples,
dicts, nested structures, or single arrays all work transparently.

---

## Pure Functional Integrators

These are standalone functions with no external dependencies beyond JAX.
They show exactly how each scheme works and are ideal for learning,
prototyping, and simple models.

### Explicit Runge-Kutta

All share the same signature: `step_fn(state, rhs_fn, dt) -> new_state`.

::::{tab-set}
:::{tab-item} SSP-RK3 (recommended)

```python
from integrix.march import rk3_ssp_step

state = rk3_ssp_step(state, rhs, dt)
```
:::
:::{tab-item} Forward Euler

```python
from integrix.march import euler_step

state = euler_step(state, rhs, dt)
```
:::
:::{tab-item} Heun (RK2)

```python
from integrix.march import heun_step

state = heun_step(state, rhs, dt)
```
:::
:::{tab-item} Classic RK4

```python
from integrix.march import rk4_step

state = rk4_step(state, rhs, dt)
```
:::
::::

### Adams-Bashforth (Multistep)

Multistep methods require history from previous steps.  The caller must
thread the history through the loop.

```python
from integrix.march import ab2_step, euler_step

# Bootstrap: one Euler step to get the first RHS history
rhs_nm1 = rhs(state)
state = euler_step(state, rhs, dt)

# Main loop: AB2 makes only 1 RHS evaluation per step
for n in range(n_steps):
    state, rhs_n, rhs_nm1 = ab2_step(state, rhs, dt, rhs_nm1)
    rhs_nm1 = rhs_n  # shift history
```

`ab3_step` is similar but requires two history levels (`rhs_nm1`, `rhs_nm2`).
Bootstrap them with at least second-order steps (e.g. two `rk3_ssp_step`s):
the crude start `rhs_nm2 = rhs_nm1 = f(y0)` lowers AB3 to second order.
`AB3Solver` does the RK3 bootstrap for you inside `diffrax.diffeqsolve`
(fixed, uniform steps):

```python
import diffrax as dfx
from integrix.march import AB3Solver

sol = dfx.diffeqsolve(dfx.ODETerm(rhs), AB3Solver(), t0=0.0, t1=1.0, dt0=0.01, y0=y0)
```

### Leapfrog with Robert-Asselin Filter

A three-level scheme that stores the previous state.  The filter damps the
computational mode.

```python
from integrix.march import leapfrog_raf_step, euler_step

# Bootstrap
state_nm1 = state
state = euler_step(state, rhs, dt)

for n in range(n_steps):
    state_new, state_filtered = leapfrog_raf_step(
        state, state_nm1, rhs, dt, alpha=0.05
    )
    state_nm1 = state_filtered  # filtered middle level
    state = state_new
```

### IMEX-SSP2

For problems with stiff + non-stiff splitting (e.g., advection + vertical
diffusion).

```python
from integrix.march import imex_ssp2_step

def rhs_explicit(state):
    """Non-stiff: advection, Coriolis, pressure gradient."""
    return tendency_explicit

def rhs_implicit(state):
    """Stiff: vertical diffusion."""
    return tendency_implicit

def implicit_solve(rhs, gamma_dt):
    """Solve: Y - gamma_dt * F_I(Y) = rhs for Y.

    For vertical diffusion, this is a tridiagonal (TDMA) solve
    along each water column.
    """
    return solved_state

for n in range(n_steps):
    state = imex_ssp2_step(
        state, rhs_explicit, rhs_implicit, implicit_solve, dt
    )
```

### Split-Explicit

For barotropic/baroclinic mode splitting.

```python
from integrix.march import split_explicit_step

def rhs_3d(state_3d, state_2d_avg):
    """Slow (baroclinic) tendency."""
    return tendency_3d

def rhs_2d(t_sub, state_2d, state_3d):
    """Fast (barotropic) tendency."""
    return tendency_2d

def couple(state_3d, state_2d_avg):
    """Ensure 3D/2D consistency after the slow step (the caller's job)."""
    return corrected_3d

def slow_forcing(state_3d):
    """Baroclinic forcing of the barotropic mode, held fixed over the cycle."""
    return forcing_2d   # same structure as state_2d

state_3d, state_2d = split_explicit_step(
    state_3d, state_2d,
    rhs_3d, rhs_2d, couple,
    dt_slow=600.0,    # baroclinic timestep
    n_substeps=30,    # 30 barotropic RK4 sub-steps
    slow_forcing=slow_forcing,   # optional
)
```

The fast mode is subcycled with RK4 (stable for oscillatory gravity waves),
`rhs_3d` and `couple` receive the trapezoidal time mean of the fast state
over the cycle, and the returned `state_2d` is the end-of-cycle state.

### Semi-Lagrangian Advection

Advects a 2D field by backtracking along characteristic curves. The scheme is `integrix.march.semi_lagrangian_step`; gridax supplies the T-point coordinates, and the read-out is a functax plan on the same points.

```python
import functax.geometry as fg
import functax.interp as fi
import gridax
import integrix.march as im
import jax.numpy as jnp

# field, u_T, v_T: [Ny, Nx] T-point fields, velocity in m/s
nodes = jnp.asarray(gridax.t_point_coordinates(grid))  # [Ny * Nx, 2], (y, x)
y = jnp.arange(grid.Ny) * grid.dy
x = jnp.arange(grid.Nx) * grid.dx
points = fg.RectilinearGrid((fg.Axis(y, period=grid.Ny * grid.dy), fg.Axis(x, period=grid.Nx * grid.dx)))


def readout(x_d):
    return fi.BSpline(3).plan(points, fg.Points(x_d))  # or fi.Linear(), fi.Pchip()


def velocity(t, p):
    plan = readout(p)
    return jnp.column_stack([plan(v_T.ravel()), plan(u_T.ravel())])


new_field = im.semi_lagrangian_step(field.ravel(), nodes, velocity, readout, t, dt).reshape(field.shape)
```

---

## Diffrax Integration (Advanced)

[Diffrax](https://docs.kidger.site/diffrax/) is a JAX-native ODE/SDE
library that provides adaptive stepping, checkpointing, dense output
(`SaveAt`), and more.  integrix provides Butcher-tableau solvers that plug
directly into `diffrax.diffeqsolve`.

### Basic Usage

```python
import diffrax as dfx
from integrix.march import RK3SSP

def rhs(t, y, args):
    """diffrax convention: (t, y, args) -> dy/dt."""
    return -y  # your PDE RHS here

solver = RK3SSP()
sol = dfx.diffeqsolve(
    dfx.ODETerm(rhs),
    solver,
    t0=0.0,
    t1=100.0,
    dt0=0.1,
    y0=initial_state,
    saveat=dfx.SaveAt(ts=jnp.linspace(0, 100, 101)),
)

# sol.ys has shape [101, ...] — the state at each saved time
```

### Available Diffrax Solvers

| Class | Order | Type | Adaptive steps | Usage |
|---|---|---|---|---|
| `ForwardEulerDfx` | 1 | ERK | no (raises) | Debugging |
| `RK2Heun` | 2 | ERK | yes (embedded Euler) | General |
| `SSP_RK2` | 2 | ERK | yes (embedded Euler) | Same as Heun, SSP form |
| `RK3SSP` | 3 | ERK | yes (embedded Heun) | **Recommended** |
| `RK4Classic` | 4 | ERK | no (raises) | High accuracy |
| `SSP_RK104` | 4 | ERK (10 stages) | no (raises) | Large CFL + 4th order |
| `IMEX_SSP2` | 2 | IMEX (MultiTerm) | yes (embedded 1st order) | Stiff/non-stiff split |
| `AB3Solver` | 3 | multistep (RK3 start) | no (raises) | One RHS per step, uniform `dt` |
| `SemiLagrangianSolver` (integrix) | 2 | semi-Lagrangian (`SemiLagrangianTerm`) | no (raises) | Advection at CFL > 1 |

"Adaptive steps" means `stepsize_controller=dfx.PIDController(...)` works;
the solvers marked "no" have no embedded error estimate and raise under an
adaptive controller rather than step blindly.  For adaptive 4th / 5th order
use diffrax's own `Tsit5` or `Dopri5`.

```python
# adaptive RK3-SSP
sol = dfx.diffeqsolve(
    dfx.ODETerm(rhs), RK3SSP(), t0=0.0, t1=10.0, dt0=0.1, y0=y0,
    stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-8),
)

# IMEX: explicit + implicit terms; with fixed steps the implicit root finder
# needs its own tolerances
terms = dfx.MultiTerm(dfx.ODETerm(rhs_explicit), dfx.ODETerm(rhs_implicit))
solver = IMEX_SSP2(root_finder=dfx.VeryChord(rtol=1e-8, atol=1e-8))
sol = dfx.diffeqsolve(terms, solver, t0=0.0, t1=1.0, dt0=0.01, y0=y0)

# semi-Lagrangian: the term is the velocity at N points, the solver holds the
# nodes and the read-out (see "Semi-Lagrangian Advection" above)
term = SemiLagrangianTerm(velocity)
sol = dfx.diffeqsolve(
    term, SemiLagrangianSolver(nodes, readout), t0=0.0, t1=10.0, dt0=0.5,
    y0=q0.ravel(), saveat=dfx.SaveAt(ts=jnp.linspace(0.0, 10.0, 11)),
)
```

### The `solve_ocean_pde` Convenience Wrapper

`solve_ocean_pde`, the thin `diffeqsolve` wrapper that applies boundary conditions to the tendency at each stage evaluation, moved to `sesmax_core.solve_pde`, where `bc_fn=` is `tendency_bc=`.
gridax no longer has it.
See the [`sesmax_core.solve_pde` API](https://jejjohnson.github.io/sesmax/reference/core/).

### Saving Trajectories

Diffrax's `SaveAt` gives fine-grained control over what to save:

```python
# Save at specific times
saveat = dfx.SaveAt(ts=jnp.array([0.0, 10.0, 50.0, 100.0]))

# Save only the final state (default)
saveat = dfx.SaveAt(t1=True)

# Save every step (careful with memory)
saveat = dfx.SaveAt(steps=True)

sol = dfx.diffeqsolve(..., saveat=saveat)
```

---

## Steppers with History (Advanced)

For schemes that don't fit the standard Runge-Kutta framework (multistep,
split-explicit), integrix provides `integrix.march.AbstractStepper`s run by
`integrix.march.run(stepper, rhs, t0, y0, dt, n_steps)`. They take the diffrax
vector field `rhs(t, y, args)` and carry their history in the loop.

### AB2Solver

```python
import jax.numpy as jnp
from integrix.march import AB2Solver, run

def rhs(t, y, args):
    return -y

y = run(AB2Solver(), rhs, 0.0, jnp.array(1.0), 0.01, 100)
```

### LeapfrogRAFSolver

```python
from integrix.march import LeapfrogRAFSolver, run

y = run(LeapfrogRAFSolver(alpha=0.05), rhs, 0.0, jnp.array(1.0), 0.01, 100)
```

### SplitExplicitRKSolver

The state is the pair `(y_3d, y_2d)` and the right-hand side is a
`SplitExplicitTerms`:

```python
from integrix.march import SplitExplicitRKSolver, SplitExplicitTerms, run

terms = SplitExplicitTerms(
    slow=rhs_slow,                 # (t, y_3d, y_2d_avg, args) -> tendency_3d
    fast=rhs_fast,                 # (t_sub, y_2d, y_3d, args) -> tendency_2d
    slow_forcing=slow_forcing,     # optional, (t, y_3d, args) -> forcing_2d
    couple=couple,                 # optional, (new_y_3d, y_2d_avg, args) -> y_3d
)
y_3d, y_2d = run(
    SplitExplicitRKSolver(n_substeps=30),
    terms,
    0.0,
    (state_3d, state_2d),
    600.0,  # slow timestep
    n_steps,
)
```

---

## Building a Custom Diffrax Solver

You can define your own Butcher-tableau solver and immediately use it with
`diffeqsolve`.

### Example: A Custom 2-Stage Method

```python
import diffrax as dfx
import jax.numpy as jnp
from typing import Any, ClassVar

class MyRK2(dfx.AbstractERK):
    """Midpoint method (RK2).

    Butcher tableau:
        0   |
        1/2 | 1/2
        ----+--------
            | 0    1
    """

    tableau: ClassVar[dfx.ButcherTableau] = dfx.ButcherTableau(
        # c: abscissae for stages 1..s-1 (excludes first stage c=0)
        c=jnp.array([0.5]),
        # b_sol: weights for combining stages
        b_sol=jnp.array([0.0, 1.0]),
        # b_error: error estimate (zeros = no embedded method)
        b_error=jnp.zeros(2),
        # a_lower: strictly lower-triangular part, row by row
        a_lower=(jnp.array([0.5]),),
    )
    interpolation_cls: ClassVar[Any] = (
        dfx.ThirdOrderHermitePolynomialInterpolation.from_k
    )

# Use it exactly like the built-in solvers
sol = dfx.diffeqsolve(
    dfx.ODETerm(lambda t, y, args: -y),
    MyRK2(),
    t0=0.0, t1=1.0, dt0=0.01, y0=1.0,
)
```

### Key Rules for Butcher Tableaux in Diffrax

1. **`c` has length `num_stages - 1`** — the first stage always has $c_0 = 0$
   (or set `c1=` for non-zero).
2. **`a_lower` is a tuple of arrays** — `a_lower[i]` has length `i + 1` and
   represents row `i + 1` of the lower-triangular A matrix.
3. **`b_error` must be an array** (not `None`) — use `jnp.zeros(num_stages)`
   if you don't have an embedded error estimate.
4. **`b_sol` must sum to 1** for consistency.
5. **`c[i]` must equal `sum(a_lower[i])`** — the row-sum condition.

### Example: Custom `AbstractSolver` (Non-RK)

For methods that don't fit the Runge-Kutta framework:

```python
import diffrax as dfx
import jax
from diffrax import RESULTS
from typing import Any, ClassVar

class MyCustomSolver(dfx.AbstractSolver):
    """A solver with a completely custom step method."""

    term_structure: ClassVar[Any] = dfx.AbstractTerm
    interpolation_cls: ClassVar[Any] = dfx.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None  # no solver state needed

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        dt = t1 - t0
        f0 = terms.vf(t0, y0, args)
        y1 = jax.tree.map(lambda y, f: y + dt * f, y0, f0)

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
```

---

## JIT Compilation

All integrators are fully `jax.jit`-compatible.  For best performance,
JIT the entire time-stepping loop:

```python
@jax.jit
def integrate(state, n_steps):
    def body(carry, _):
        return rk3_ssp_step(carry, rhs, dt), None

    final_state, _ = jax.lax.scan(body, state, None, length=n_steps)
    return final_state

result = integrate(initial_state, 10000)
```

Using `jax.lax.scan` instead of a Python loop avoids retracing and enables
XLA to optimise the entire integration as a single compiled kernel.
