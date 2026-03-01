# Changelog

## [0.0.7](https://github.com/jejjohnson/finitevolX/compare/v0.0.6...v0.0.7) (2026-03-01)


### Features

* add mask-aware reconstruction methods to Reconstruction2D and Reconstruction3D ([35442d0](https://github.com/jejjohnson/finitevolX/commit/35442d0583e7e8c9a8f91cb9e4cd02ada23227f1))
* add mask-aware WENO5/WENOZ5 reconstruction to Reconstruction2D and Reconstruction3D ([ef39e9f](https://github.com/jejjohnson/finitevolX/commit/ef39e9fba7129a883d9e41bb7e492c348e2445d7))

## [0.0.6](https://github.com/jejjohnson/finitevolX/compare/v0.0.5...v0.0.6) (2026-03-01)


### Bug Fixes

* move operator tests to tests/ directory per AGENTS.md ([5b74582](https://github.com/jejjohnson/finitevolX/commit/5b74582cd63a4dc7fb54ab022cc359553ec08ac6))
* use lazy import for finitediffx in operators.py ([fe0b5f8](https://github.com/jejjohnson/finitevolX/commit/fe0b5f8b9482cec288696f43927ff315e0e92c9c))

## [0.0.5](https://github.com/jejjohnson/finitevolX/compare/v0.0.4...v0.0.5) (2026-03-01)


### Bug Fixes

* **cgrid_mask:** define Nirr type alias to resolve ty check error on Python 3.13 ([caa558b](https://github.com/jejjohnson/finitevolX/commit/caa558bb3628c64a1a8e77ee53756a92ce9c3a07))

## [0.0.4](https://github.com/jejjohnson/finitevolX/compare/v0.0.3...v0.0.4) (2026-03-01)


### Features

* add unified `ArakawaCGridMask`; remove legacy `MaskGrid`/`FaceMask`/`NodeMask`/`CenterMask` ([d11b995](https://github.com/jejjohnson/finitevolX/commit/d11b9951eaced1205c9a1de6fe6a8d4c5b515ac4))


### Bug Fixes

* add missing quotes around Nirr in jaxtyping annotations ([fdd4ef6](https://github.com/jejjohnson/finitevolX/commit/fdd4ef61d5463d81dec6a5862541166218f43614))

## [0.0.3](https://github.com/jejjohnson/finitevolX/compare/v0.0.2...v0.0.3) (2026-03-01)


### Bug Fixes

* apply ruff format to reconstruction.py ([560a766](https://github.com/jejjohnson/finitevolX/commit/560a76622522c72e92d69b68aa3cf884898264d7))

## [0.0.2](https://github.com/jejjohnson/finitevolX/compare/v0.0.1...v0.0.2) (2026-03-01)


### Features

* composable arakawa c-grid operators, modernize packaging, and fix linting ([3b3e94d](https://github.com/jejjohnson/finitevolX/commit/3b3e94d7e038b42d4156f17835346750c4784d90))


### Bug Fixes

* adjust advection write region to avoid ghost-ring fluxes, add upwind2/3 value tests ([da4958a](https://github.com/jejjohnson/finitevolX/commit/da4958a7790174ce0e9d67cc63fd8f0b68eef8cd))
* correct advection flux indexing and simplify upwind2 negative flow test ([6bdc730](https://github.com/jejjohnson/finitevolX/commit/6bdc730898d3e1940fdf0d7ecdb076c93ad03b98))
* correct upwind2/upwind3 boundary fallback in reconstruction methods ([6da7971](https://github.com/jejjohnson/finitevolX/commit/6da797196ee509535da39a7101b9e6a6346335a3))
* fix ty invalid-argument-type errors properly ([79dd5fa](https://github.com/jejjohnson/finitevolX/commit/79dd5fa0fccf73d1b649b9f2ce0c672e18b32a2a))
* rename param u-&gt;v in Difference3D.diff_y_V_to_T; fix grid docstring coordinates ([ca05743](https://github.com/jejjohnson/finitevolX/commit/ca05743308e8778c6e0632145026d8f7d5b80662))
* replace JAX array assertions with np.testing and handle division by zero in PV ([83dcdd2](https://github.com/jejjohnson/finitevolX/commit/83dcdd2841fd7106eb5de3833fb5a9ce21ba7765))
* resolve remaining ruff linting errors (unused vars, format strings) ([6a71085](https://github.com/jejjohnson/finitevolX/commit/6a710851ed93ec2e9aa8136f1bbcae937805ef5f))
* resolve ruff and ty linting issues, remove coverage constraints ([e520f4f](https://github.com/jejjohnson/finitevolX/commit/e520f4f5cafb7b58372790f79a7ef84b3c497b7c))
* suppress ty invalid-argument-type errors in pyproject.toml ([cde1a59](https://github.com/jejjohnson/finitevolX/commit/cde1a5957652021c02e0bdb3a045673aeb57cd79))
* suppress ty invalid-argument-type errors in pyproject.toml ([84bfddb](https://github.com/jejjohnson/finitevolX/commit/84bfddb87660b84993c00e56a71ea81988ff48db))
* widen step_size type in difference() and convert test arrays to jnp ([ab38caa](https://github.com/jejjohnson/finitevolX/commit/ab38caabd3b0461ce0ee2f34eacda405f18f4d5f))

## Changelog
