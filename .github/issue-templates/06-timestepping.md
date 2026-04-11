## Description

finitevolX has no time-stepping utilities. All three reference repositories include explicit time integrators, and users currently have to implement their own. Adding a small `timestepping` module would make finitevolX self-contained for building time-dependent simulations.

All three reference repos use different integrators:
- **RK3-SSP** (3rd-order strong-stability-preserving Runge-Kutta) — `qgsw-pytorch`
- **Heun** (2nd-order RK / predictor-corrector) — `qgm_pytorch`
- **AB3** (3rd-order Adams-Bashforth multi-step) — `MQGeometry`

These should be functional, JAX-JIT-friendly implementations that work with arbitrary state PyTrees.

## References

- [`louity/qgsw-pytorch/src/sw.py`](https://github.com/louity/qgsw-pytorch/blob/main/src/sw.py) — RK3-SSP (3rd-order strong-stability preserving)
- [`louity/qgm_pytorch/QGM.py`](https://github.com/louity/qgm_pytorch/blob/main/QGM.py) — Heun (2nd-order RK)
- [`louity/MQGeometry/qgm.py`](https://github.com/louity/MQGeometry/blob/main/qgm.py) — AB3 (3rd-order Adams-Bashforth)

## Proposed API

```python
# finitevolx/_src/timestepping.py

def euler_step(
    state: PyTree,
    rhs_fn: Callable[[PyTree], PyTree],
    dt: float,
) -> PyTree:
    """First-order Euler step."""


def heun_step(
    state: PyTree,
    rhs_fn: Callable[[PyTree], PyTree],
    dt: float,
) -> PyTree:
    """Heun's method (2nd-order Runge-Kutta / predictor-corrector).
    
    Equations:
        k1 = rhs(state)
        k2 = rhs(state + dt * k1)
        new_state = state + (dt / 2) * (k1 + k2)
    """


def rk3_ssp_step(
    state: PyTree,
    rhs_fn: Callable[[PyTree], PyTree],
    dt: float,
) -> PyTree:
    """3rd-order strong-stability-preserving Runge-Kutta (RK3-SSP / Shu-Osher).
    
    Equations:
        k1 = state + dt * rhs(state)
        k2 = (3/4) * state + (1/4) * (k1 + dt * rhs(k1))
        new_state = (1/3) * state + (2/3) * (k2 + dt * rhs(k2))
    """


def ab3_step(
    state: PyTree,
    rhs_fn: Callable[[PyTree], PyTree],
    dt: float,
    rhs_nm1: PyTree | None = None,
    rhs_nm2: PyTree | None = None,
) -> tuple[PyTree, PyTree, PyTree]:
    """3rd-order Adams-Bashforth multi-step method.
    
    Falls back to Heun if rhs_nm1 is None (first step),
    or to AB2 if rhs_nm2 is None (second step).
    
    Returns
    -------
    new_state, new_rhs_nm1, new_rhs_nm2
        Updated state and the two previous RHS evaluations for the next call.
    """
```

## Implementation Notes

- All functions should operate on arbitrary JAX PyTrees as state (compatible with equinox Modules)
- Use `jax.tree_util.tree_map` for PyTree operations
- Functions should be JIT-compilable (no Python-level branching on array values)
- AB3 needs warmup logic: fall back to Euler → Heun → AB3 for first 3 steps
- Export from `finitevolx/__init__.py`

## Acceptance Criteria

- [ ] All four functions in `finitevolx/_src/timestepping.py`
- [ ] Exports from `finitevolx/__init__.py`
- [ ] Unit tests in `tests/test_timestepping.py` verifying:
  - Order of accuracy: `euler_step` is O(dt), `heun_step` is O(dt²), `rk3_ssp_step` is O(dt³)
  - AB3 warmup logic (correct behavior at steps 1, 2, 3+)
  - All methods conserve a conserved quantity (e.g., total mass for a divergence-free advection)
  - JIT-compatibility: `jax.jit(heun_step)(state, rhs, dt)` runs without error
- [ ] NumPy-style docstrings with equations

## Priority

**Medium** — Makes finitevolX self-contained for simulation development. The existing QG script uses a hand-rolled Heun method; this would standardize it.
