# Changelog

## [0.3.0](https://github.com/jejjohnson/gridax/compare/gridax-jax-v0.2.0...gridax-jax-v0.3.0) (2026-11-21)


### ⚠ BREAKING CHANGES

* **gridax-jax:** spherical results move by about 3e-5 relative with the new default radius; pass R=6_371_200.0 to keep the old numbers.
* **gridax:** spherical results move by about 3e-5 relative with the new default radius; pass R=6_371_200.0 to keep the old numbers.

### Code Refactoring

* **gridax-jax:** deprecate the forcing, vertical modes, relaxation and the barotropic filter, now in sesmax-core ([#460](https://github.com/jejjohnson/gridax/issues/460)) ([dd5adcd](https://github.com/jejjohnson/gridax/commit/dd5adcd5ac10dbad576734b932a03722d818328e)), closes [#422](https://github.com/jejjohnson/gridax/issues/422)
* **gridax:** the constants, the Coriolis helpers and the linear equation of state come from geophysix ([#459](https://github.com/jejjohnson/gridax/issues/459)) ([4d2253d](https://github.com/jejjohnson/gridax/commit/4d2253de3fbe32fc93ec272f6344a17312ba603e)), closes [#446](https://github.com/jejjohnson/gridax/issues/446)

## [0.2.0](https://github.com/jejjohnson/finitevolX/compare/gridax-jax-v0.1.0...gridax-jax-v0.2.0) (2026-11-20)


### ⚠ BREAKING CHANGES

* the package is gridax-jax and the import gridax_jax.

### Code Refactoring

* rename gridax-ops to gridax-jax (import gridax_jax) ([#437](https://github.com/jejjohnson/finitevolX/issues/437)) ([aff7146](https://github.com/jejjohnson/finitevolX/commit/aff714687ebebc14fd1ad06a8d11f520124a6146)), closes [#435](https://github.com/jejjohnson/finitevolX/issues/435)
