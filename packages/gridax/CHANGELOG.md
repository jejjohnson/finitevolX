# Changelog

## 0.1.0 (2026-11-20)


### Features

* first release: grids, masks, the raw stencils, the C-grid operators, advection, diffusion, boundary conditions, reductions and gridax.analysis, on NumPy, JAX or any array-API backend ([#435](https://github.com/jejjohnson/finitevolX/issues/435))
* **gridax:** analysis kernels on collocated grids — finite differences and axis remapping on any array backend ([#434](https://github.com/jejjohnson/finitevolX/issues/434)) ([95798a1](https://github.com/jejjohnson/finitevolX/commit/95798a1a7d7d4052bd4d6bea634f1e0322f39f64))


### Bug Fixes

* **grid:** spherical T rows are cell centres, so whole-sphere area weights sum to 4πR² ([#433](https://github.com/jejjohnson/finitevolX/issues/433)) ([067d292](https://github.com/jejjohnson/finitevolX/commit/067d292cac70ff7eade615aaadc7d86be1039b5b)), closes [#322](https://github.com/jejjohnson/finitevolX/issues/322)


### Code Refactoring

* rename gridax-ops to gridax-jax (import gridax_jax) ([#437](https://github.com/jejjohnson/finitevolX/issues/437)) ([aff7146](https://github.com/jejjohnson/finitevolX/commit/aff714687ebebc14fd1ad06a8d11f520124a6146)), closes [#435](https://github.com/jejjohnson/finitevolX/issues/435)
