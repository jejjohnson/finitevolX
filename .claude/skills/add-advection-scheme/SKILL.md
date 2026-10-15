---
name: add-advection-scheme
description: Add an advection scheme to finitevolX — a face-value reconstruction (upwind, linear, WENO / WENO-Z variant), a TVD flux limiter, or a face-flux function (Rusanov-style) — wired through Reconstruction1D/2D/3D, the Advection1D/2D/3D and SphericalAdvection method dispatch, the coastal stencil fallback, the scale-covariance method lists and the golden fixtures. Use when asked to add, port or implement a reconstruction, WENO stencil, limiter or numerical flux in finitevolx/_src/advection.
---

# Add an advection scheme

Read contract 2 (the advection write region and `wall` modes), contract 3
(masks) and contract 6 (units) in `AGENTS.md` first, then
`docs/advection.md`.

## 1. Make sure it does not exist yet

Search `docs/capabilities.md` and `Reconstruction2D`'s methods: `naive`,
`upwind1/2/3`, `weno3/5/7/9`, `wenoz3/5`, `tvd` with `minmod` / `van_leer`
/ `superbee` / `mc`, the functional `reconstruct` / `upwind_1pt` /
`upwind_3pt` / `upwind_5pt`, the `weno_*pts*` and `linear_*pts*` stencil
functions, and the face fluxes `uv_center_flux`, `uv_node_flux`,
`rusanov_flux`. A different limiter for TVD is a new limiter, not a new
scheme.

## 2. Where each piece goes (`finitevolx/_src/advection/`)

| Piece | File | Wire it into |
|---|---|---|
| Point stencil (`qm, q0, qp, … → q at the face`) | `weno.py` (nonlinear, left and `_right` variants), `linear.py` (polynomial) | `__init__.py` exports |
| Limiter `φ(r)` | `limiters.py` | `_LIMITERS` in `reconstruction.py` and `_TVD_LIMITERS` in `advection.py` (then every advection operator accepts it as `method=`) |
| Face-value method on a grid | `reconstruction.py`: `<name>_x` on `Reconstruction1D`, `<name>_x` / `<name>_y` on `Reconstruction2D/3D` (there is no `_z`: z-levels are independent) | the `elif method == ...` chain in `Advection1D/2D/3D.__call__` (`advection.py`) and `_reconstruct_faces_2d` / `_reconstruct_faces_3d` (`spherical_advection.py`); the error for an unknown method must keep the words "Unknown method" (the scale-covariance probe relies on them) |
| Coastal fallback (wide stencils) | `_rec_funcs_for_method_1d/2d/3d` in `advection.py`: `{stencil_size: fn}` down to `upwind1` at size 2 | `_MASK_DISPATCHABLE` and the ladder in `_tier_funcs`; sizes must be a subset of `_HIERARCHY_SIZES = (2, 4, 6, 8, 10)` |
| Functional dispatcher | `reconstruct_fn.py` (`reconstruct(q, u, dim, method, num_pts)`) | only if the scheme fits its `method` × `num_pts` grid |

Rules:

- The face value is upwinded on the sign of the face velocity
  (`jnp.where(u >= 0, left_biased, right_biased)`), and the flux is
  `h_face * u`, single-valued at each face so the divergence conserves mass.
- Closed walls: the Cartesian operators write the tendency on `[2:-2]` /
  `[2:-2, 2:-2]`; a stencil wider than the ghost ring must fall back near
  the walls (as `weno5` does) rather than read outside the array.
- No absolute epsilon on a dimensional quantity: floor the WENO weights
  through `weno._normalised` + `_WENO_EPS_REL` and TVD ratios through
  `reconstruction._slope_ratio`; add a nonlinear scheme to `WENO_METHODS` /
  `TVD_METHODS` in `tests/test_scale_covariance.py`
  (`TestReconstructionAmplitudeCovariance` holds them to rounding) and to
  `tests/test_relative_eps.py` (finite gradients on flat stencils, float32).
- Comment the stencil with its formula and cite the scheme in the
  docstring.

## 3. Export and document

`finitevolx/__init__.py` (import + `__all__` group, e.g. `# WENO stencil
functions`, `# Flux limiters`), the `docs/api/advection.md` /
`docs/api/reconstruction.md` page, the method list in the `Advection*`
docstrings and in `docs/advection.md`, then `make capabilities`.

## 4. Tests

- Point stencils / limiters: `tests/test_weno.py` (constant fields, polynomial
  exactness, monotonicity near a step, symmetry), `tests/test_limiters.py`
  (TVD region, φ(1) = 1).
- Grid methods: `tests/test_reconstruction.py` (`TestReconstruction1D/2D/3D`
  and the `*Masked` classes); the method in the `test_all_methods_run`
  lists of `tests/test_advection.py`; conservation
  (`test_1d_conservation`).
- Units: add the method to `LINEAR_METHODS`, `WENO_METHODS` or `TVD_METHODS`
  in `tests/test_scale_covariance.py` (the probe adds a case per operator
  that supports it); if it carries an absolute epsilon, mark it as the WENO
  cases are and open an issue.
- Masked dispatch: golden entries in `_advection_entries`
  (`tests/fixtures/_gen_golden.py`) and `tests/test_advection_masks.py`;
  regenerate with `uv run python tests/fixtures/_gen_golden.py` and check
  that `git diff --stat tests/fixtures/golden/` lists only new files.
- Spherical: `tests/test_spherical_advection.py`.

## 5. Verify

```bash
uv run pytest -n 2 tests/test_weno.py tests/test_limiters.py tests/test_reconstruction.py \
  tests/test_reconstruction_pipeline.py tests/test_advection.py tests/test_advection_masks.py \
  tests/test_flux.py tests/test_spherical_advection.py tests/test_open_walls.py \
  tests/test_scale_covariance.py tests/test_capabilities.py
```

Then the `pre-pr-check` skill.
