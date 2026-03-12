# Time Integration: Usage Guide

This page covers practical usage of finitevolX's time integration module,
from quick-start examples to building custom diffrax solvers.

---

## Quick Start

The fastest way to time-step a PDE: use a pure functional step function
inside a loop.

```python
import jax
import jax.numpy as jnp
import finitevolx as fvx

jax.config.update("jax_enable_x64", True)

# 1. Define your spatial RHS (any JAX pytree in, same pytree out)
grid = fvx.ArakawaCGrid2D.from_interior(64, 64, 1e4, 1e4)

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
    state = fvx.rk3_ssp_step(state, rhs, dt)
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

=== "SSP-RK3 (recommended)"

    ```python
    from finitevolx import rk3_ssp_step

    state = rk3_ssp_step(state, rhs, dt)
    ```

=== "Forward Euler"

    ```python
    from finitevolx import euler_step

    state = euler_step(state, rhs, dt)
    ```

=== "Heun (RK2)"

    ```python
    from finitevolx import heun_step

    state = heun_step(state, rhs, dt)
    ```

=== "Classic RK4"

    ```python
    from finitevolx import rk4_step

    state = rk4_step(state, rhs, dt)
    ```

### Adams-Bashforth (Multistep)

Multistep methods require history from previous steps.  The caller must
thread the history through the loop.

```python
from finitevolx import ab2_step, euler_step

# Bootstrap: one Euler step to get the first RHS history
rhs_nm1 = rhs(state)
state = euler_step(state, rhs, dt)

# Main loop: AB2 makes only 1 RHS evaluation per step
for n in range(n_steps):
    state, rhs_n, rhs_nm1 = ab2_step(state, rhs, dt, rhs_nm1)
    rhs_nm1 = rhs_n  # shift history
```

`ab3_step` is similar but requires two history levels (`rhs_nm1`, `rhs_nm2`).

### Leapfrog with Robert-Asselin Filter

A three-level scheme that stores the previous state.  The filter damps the
computational mode.

```python
from finitevolx import leapfrog_raf_step, euler_step

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
from finitevolx import imex_ssp2_step

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
from finitevolx import split_explicit_step

def rhs_3d(state_3d, state_2d_avg):
    """Slow (baroclinic) tendency."""
    return tendency_3d

def rhs_2d(t_sub, state_2d, state_3d):
    """Fast (barotropic) tendency."""
    return tendency_2d

def couple(state_3d, state_2d_avg):
    """Ensure 3D/2D consistency after the slow step."""
    return corrected_3d

state_3d, state_2d = split_explicit_step(
    state_3d, state_2d,
    rhs_3d, rhs_2d, couple,
    dt_slow=600.0,    # baroclinic timestep
    n_substeps=30,    # 30 barotropic sub-steps
)
```

### Semi-Lagrangian Advection

Advects a 2D field by backtracking along characteristic curves.

```python
from finitevolx import semi_lagrangian_step

# field: [Ny, Nx], u/v: velocity in m/s, dx/dy: grid spacing in m
new_field = semi_lagrangian_step(
    field, u, v, dx, dy, dt,
    interp_order=1,  # 1 = linear (monotone), 0 = nearest
    bc="periodic",   # or "edge" for clamped boundaries
)
```

---

## Diffrax Integration (Advanced)

[Diffrax](https://docs.kidger.site/diffrax/) is a JAX-native ODE/SDE
library that provides adaptive stepping, checkpointing, dense output
(`SaveAt`), and more.  finitevolX provides Butcher-tableau solvers that plug
directly into `diffrax.diffeqsolve`.

### Basic Usage

```python
import diffrax as dfx
from finitevolx import RK3SSP

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

| Class | Order | Type | Usage |
|---|---|---|---|
| `ForwardEulerDfx` | 1 | ERK | Debugging |
| `RK2Heun` | 2 | ERK | General |
| `SSP_RK2` | 2 | ERK | Same as Heun, SSP form |
| `RK3SSP` | 3 | ERK | **Recommended** |
| `RK4Classic` | 4 | ERK | High accuracy |
| `SSP_RK104` | 4 | ERK (10 stages) | Large CFL + 4th order |
| `IMEX_SSP2` | 2 | IMEX (MultiTerm) | Stiff/non-stiff split |

### The `solve_ocean_pde` Convenience Wrapper

A thin wrapper around `diffeqsolve` that optionally applies boundary
conditions to the tendency at each stage evaluation.

```python
from finitevolx import solve_ocean_pde, RK3SSP

def rhs(t, y, args):
    return compute_tendency(y)

def apply_bc(dydt):
    """Zero out tendency in ghost cells."""
    return dydt.at[0, :].set(0).at[-1, :].set(0)

sol = solve_ocean_pde(
    rhs,
    RK3SSP(),
    y0=initial_state,
    t0=0.0,
    t1=1000.0,
    dt0=1.0,
    bc_fn=apply_bc,
    saveat=dfx.SaveAt(t1=True),
)
```

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

## Manual Solver Interfaces (Advanced)

For schemes that don't fit the standard Runge-Kutta framework (multistep,
split-explicit), finitevolX provides `equinox.Module`-based solvers with
explicit `init`/`step` interfaces.

### AB2Solver

```python
from finitevolx import AB2Solver

def rhs(t, y):
    """Note: (t, y) signature, not (t, y, args)."""
    return -y

solver = AB2Solver()
solver, y = solver.init(rhs, t0=0.0, y0=jnp.array(1.0), dt=0.01)

for n in range(1, n_steps):
    y, solver = solver.step(rhs, t0 + n * dt, y, dt)
```

The solver is an immutable equinox Module — `solver.step()` returns a
*new* solver object with updated history.  This makes it fully compatible
with `jax.jit` and functional JAX patterns.

### LeapfrogRAFSolver

```python
from finitevolx import LeapfrogRAFSolver

solver = LeapfrogRAFSolver(alpha=0.05)
solver, y = solver.init(rhs, t0=0.0, y0=jnp.array(1.0), dt=0.01)

for n in range(1, n_steps):
    y, solver = solver.step(rhs, t0 + n * dt, y, dt)
```

### SplitExplicitRKSolver

```python
from finitevolx import SplitExplicitRKSolver

solver = SplitExplicitRKSolver(n_substeps=30)

y_3d_new, y_2d_new = solver.step(
    rhs_slow,   # (t, y_3d, y_2d_avg) -> tendency_3d
    rhs_fast,   # (t_sub, y_2d, y_3d) -> tendency_2d
    t=0.0,
    y_3d=state_3d,
    y_2d=state_2d,
    dt_slow=600.0,
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
        return fvx.rk3_ssp_step(carry, rhs, dt), None

    final_state, _ = jax.lax.scan(body, state, None, length=n_steps)
    return final_state

result = integrate(initial_state, 10000)
```

Using `jax.lax.scan` instead of a Python loop avoids retracing and enables
XLA to optimise the entire integration as a single compiled kernel.
