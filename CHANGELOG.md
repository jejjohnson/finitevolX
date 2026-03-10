# Changelog

## [0.0.14](https://github.com/jejjohnson/finitevolX/compare/v0.0.13...v0.0.14) (2026-03-10)


### Bug Fixes

* correct physical formulation in all three double-gyre example scripts ([3bafe48](https://github.com/jejjohnson/finitevolX/commit/3bafe48e8f3ca2b92c425c012a3cc631498ef2b4))
* correct QG 1.5-layer PV formulation, drag, beta term, and forcing amplitude ([758c377](https://github.com/jejjohnson/finitevolX/commit/758c37773d3964311edfae8ed83d3116c03b8bb5))
* correct wind sign, nonlinear SWM Bernoulli double-counting, and Coriolis staggering ([4adb0eb](https://github.com/jejjohnson/finitevolX/commit/4adb0eb5babf7cdb8fd0ab1749a95e3a97ef130d))

## [0.0.13](https://github.com/jejjohnson/finitevolX/compare/v0.0.12...v0.0.13) (2026-03-08)


### Bug Fixes

* convert QG example from periodic beta-plane to closed-basin double-gyre ([b132e8e](https://github.com/jejjohnson/finitevolX/commit/b132e8e0001ab6a111e8d86d6b735f26eb8e3a20))
* convert QG example from periodic beta-plane to closed-basin double-gyre ([10a253d](https://github.com/jejjohnson/finitevolX/commit/10a253df437afbad5f9178e8bbbce82366301549))

## [0.0.12](https://github.com/jejjohnson/finitevolX/compare/v0.0.11...v0.0.12) (2026-03-07)


### Bug Fixes

* address operator dtype and fixture review feedback ([e2d6afc](https://github.com/jejjohnson/finitevolX/commit/e2d6afc4c243ace9ec08584aca8b93c6344f9a4e))
* Advection3D ghost flux bug; add U/V/X ghost cell tests; update instructions and figures ([a4ca415](https://github.com/jejjohnson/finitevolX/commit/a4ca4156e9a1ec42f8ed4e22a321b29747b927a7))
* correct Arakawa C-grid operators, ghost-cell tests for all staggered types, and agent instructions ([d0c3c8c](https://github.com/jejjohnson/finitevolX/commit/d0c3c8c03f3595fe1d9e5e224f43451d25416b2d))
* correct operators, add non-constant tests, document C-grid discretization ([5a1e831](https://github.com/jejjohnson/finitevolX/commit/5a1e8316ec6d73621e6e68be2f1173d01e7f658f))
* rewrite kinetic_energy and bernoulli_potential for Arakawa C-grid same-size convention ([06d8c9a](https://github.com/jejjohnson/finitevolX/commit/06d8c9ae06a762d64801067f20abeced4a9faf46))

## [0.0.11](https://github.com/jejjohnson/finitevolX/compare/v0.0.10...v0.0.11) (2026-03-07)


### Features

* modernize double gyre example scripts ([1189ede](https://github.com/jejjohnson/finitevolX/commit/1189edefad59acd5319ce0aad7c796a4e9276240))


### Bug Fixes

* address example script review feedback ([c09953d](https://github.com/jejjohnson/finitevolX/commit/c09953d518c4a4dc162008d8da2d8c9ba1a00352))
* tune double gyre example diagnostics ([65c0fe6](https://github.com/jejjohnson/finitevolX/commit/65c0fe6f1097c3ca4e314dd3c908367a5865487d))
* use corner-based qg geostrophic mapping ([e6f995d](https://github.com/jejjohnson/finitevolX/commit/e6f995df79e3ac448ffe5ee4790129cff942583f))

## [0.0.10](https://github.com/jejjohnson/finitevolX/compare/v0.0.9...v0.0.10) (2026-03-07)


### Features

* add per-side boundary condition API for 2D ghost-ring fields ([928e0b5](https://github.com/jejjohnson/finitevolX/commit/928e0b525dcaeab5e5279677005f1f9922749381))


### Bug Fixes

* correct Neumann face signs and exports ([b08cdfa](https://github.com/jejjohnson/finitevolX/commit/b08cdfaaf72d2237e950a2e79f66d1decc5176f7))

## [0.0.9](https://github.com/jejjohnson/finitevolX/compare/v0.0.8...v0.0.9) (2026-03-07)


### Features

* add capacitance matrix solver and preconditioned CG ([3b96918](https://github.com/jejjohnson/finitevolX/commit/3b96918d72e597dbf9a180ec169fd3701f13450e))
* add spectral transforms and core elliptic solvers (DST/DCT/FFT) ([3674204](https://github.com/jejjohnson/finitevolX/commit/3674204cf9e4c8bf80de0f068551e1904974d7e0))
* integrate JAX spectral transforms, elliptic solvers, capacitance matrix method, and lineax-based PCG into finitevolX ([e8f0191](https://github.com/jejjohnson/finitevolX/commit/e8f0191a00b5725bf2234f518fb5bf2652c691b2))


### Bug Fixes

* handle lambda_=0 in solve_helmholtz_dct and solve_helmholtz_fft ([72ec96b](https://github.com/jejjohnson/finitevolX/commit/72ec96be8bd6defba5ff6dcbedb7a20e1f15000d))
* resolve ruff lint errors in test files (RUF059, RUF003) ([cc9400a](https://github.com/jejjohnson/finitevolX/commit/cc9400a45b0700efd24bae7feb456883984ab5fc))

## [0.0.8](https://github.com/jejjohnson/finitevolX/compare/v0.0.7...v0.0.8) (2026-03-03)


### Features

* adopt pypackage_template standards and best practices ([71afd3e](https://github.com/jejjohnson/finitevolX/commit/71afd3e3f40f2c06b9a23dce4863944f3e446b76))


### Bug Fixes

* correct API reference documentation to use actual exported functions ([4dc1804](https://github.com/jejjohnson/finitevolX/commit/4dc1804668ff687e6b9457b0fe7c5d3503c37302))

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
