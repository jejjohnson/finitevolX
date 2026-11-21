# Changelog

## [0.2.0](https://github.com/jejjohnson/gridax/compare/gridax-v0.1.0...gridax-v0.2.0) (2026-11-21)


### ⚠ BREAKING CHANGES

* **gridax:** spherical results move by about 3e-5 relative with the new default radius; pass R=6_371_200.0 to keep the old numbers.

### Features

* **gridax:** fourth- and sixth-order staggered differences and averages, staggered_difference and staggered_average ([#469](https://github.com/jejjohnson/gridax/issues/469)) ([bf5ac89](https://github.com/jejjohnson/gridax/commit/bf5ac8991a8350e487867b884b7c23a76c7d0f83)), closes [#457](https://github.com/jejjohnson/gridax/issues/457)
* **gridax:** gridax.stencils, one stencil module for collocated and staggered points, with exact coefficients ([#465](https://github.com/jejjohnson/gridax/issues/465)) ([87dce56](https://github.com/jejjohnson/gridax/commit/87dce5655fbb43b48a6745c3dd36e40322168792)), closes [#454](https://github.com/jejjohnson/gridax/issues/454)
* **gridax:** the stencil module solves for cell averages and WENO, with data="cell", biased offsets, optimal weights and smoothness matrices ([#468](https://github.com/jejjohnson/gridax/issues/468)) ([76bc36f](https://github.com/jejjohnson/gridax/commit/76bc36f068b6f67834623b3bce6682fc3a7a1fc8)), closes [#462](https://github.com/jejjohnson/gridax/issues/462)


### Code Refactoring

* **gridax:** the constants, the Coriolis helpers and the linear equation of state come from geophysix ([#459](https://github.com/jejjohnson/gridax/issues/459)) ([4d2253d](https://github.com/jejjohnson/gridax/commit/4d2253de3fbe32fc93ec272f6344a17312ba603e)), closes [#446](https://github.com/jejjohnson/gridax/issues/446)

## 0.1.0 (2026-11-20)


### Features

* first release: grids, masks, the raw stencils, the C-grid operators, advection, diffusion, boundary conditions, reductions and gridax.analysis, on NumPy, JAX or any array-API backend ([#435](https://github.com/jejjohnson/finitevolX/issues/435))
* **gridax:** analysis kernels on collocated grids — finite differences and axis remapping on any array backend ([#434](https://github.com/jejjohnson/finitevolX/issues/434)) ([95798a1](https://github.com/jejjohnson/finitevolX/commit/95798a1a7d7d4052bd4d6bea634f1e0322f39f64))


### Bug Fixes

* **grid:** spherical T rows are cell centres, so whole-sphere area weights sum to 4πR² ([#433](https://github.com/jejjohnson/finitevolX/issues/433)) ([067d292](https://github.com/jejjohnson/finitevolX/commit/067d292cac70ff7eade615aaadc7d86be1039b5b)), closes [#322](https://github.com/jejjohnson/finitevolX/issues/322)


### Code Refactoring

* rename gridax-ops to gridax-jax (import gridax_jax) ([#437](https://github.com/jejjohnson/finitevolX/issues/437)) ([aff7146](https://github.com/jejjohnson/finitevolX/commit/aff714687ebebc14fd1ad06a8d11f520124a6146)), closes [#435](https://github.com/jejjohnson/finitevolX/issues/435)
