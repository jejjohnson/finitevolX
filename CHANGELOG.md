# Changelog

## [0.0.25](https://github.com/jejjohnson/finitevolX/compare/v0.0.24...v0.0.25) (2026-03-12)


### Features

* add diagnostic and geographic operators ([fe71938](https://github.com/jejjohnson/finitevolX/commit/fe719388864e0ecd573832d49b2db440edafdd7d))
* **diagnostics:** add strain, enstrophy, QG PV, and conservation operators ([040079d](https://github.com/jejjohnson/finitevolX/commit/040079d02829fbbec6b73d657484ffdc16727f60))
* export diagnostic and geographic operators from package ([2b4f7af](https://github.com/jejjohnson/finitevolX/commit/2b4f7af1d6d36641be8d2b557ed8e163d743dee3))
* **operators:** add spherical geographic operators ([83f6274](https://github.com/jejjohnson/finitevolX/commit/83f6274a53ea148beaa540fefef44c5573a16e3c))


### Bug Fixes

* **geographic:** add pole guard and address PR review comments ([77fdb10](https://github.com/jejjohnson/finitevolX/commit/77fdb10689db184efb2236f661bace4d44ca81f1))

## [0.0.24](https://github.com/jejjohnson/finitevolX/compare/v0.0.23...v0.0.24) (2026-03-11)


### Bug Fixes

* **masks:** correct psi_irrbound index offset and x/y swap ([add4d36](https://github.com/jejjohnson/finitevolX/commit/add4d36cfd22404a7abb453c08cca9c458cb7727))
* **masks:** correct vorticity boundary adjacency at w-points ([648099f](https://github.com/jejjohnson/finitevolX/commit/648099f06e331cbfbbc6d6bb788c7d51ba42495c))
* **notebooks:** use correct from_interior API (Lx/Ly not dx/dy) ([a158663](https://github.com/jejjohnson/finitevolX/commit/a158663d9851edcb93ef000fa6aa1bbe4cfb37f1))
* **tests:** replace ambiguous multiplication sign with ASCII x ([84332db](https://github.com/jejjohnson/finitevolX/commit/84332db6dff25ce9283c9d6326dfdadb305b489a))

## [0.0.23](https://github.com/jejjohnson/finitevolX/compare/v0.0.22...v0.0.23) (2026-03-11)


### Features

* **bc:** add Robin and Extrapolation boundary conditions ([9c1fec3](https://github.com/jejjohnson/finitevolX/commit/9c1fec30650a200948c70a0169ccaffde01f24d6))
* **bc:** add Robin and Extrapolation boundary conditions ([36ea563](https://github.com/jejjohnson/finitevolX/commit/36ea56353d12cd208bc2934cd50eadb74c664927))

## [0.0.22](https://github.com/jejjohnson/finitevolX/compare/v0.0.21...v0.0.22) (2026-03-11)


### Features

* add mask parameter to Advection2D and Advection3D ([4545060](https://github.com/jejjohnson/finitevolX/commit/454506071816df1a79fc32793a9531a1daff4fdd))


### Bug Fixes

* add input validation and NaN-safe blending to upwind_flux ([87357c4](https://github.com/jejjohnson/finitevolX/commit/87357c422a6fe52ca739997240aecaf6c9718159))

## [0.0.21](https://github.com/jejjohnson/finitevolX/compare/v0.0.20...v0.0.21) (2026-03-11)


### Features

* add `multilayer()` vmap helper to lift 2D operators over layer/mode batch axis ([0848cfd](https://github.com/jejjohnson/finitevolX/commit/0848cfddbea7627ecbc878a4d248657d647b846a))
* add multilayer() vmap helper, tests, and docs page ([e1f8c8b](https://github.com/jejjohnson/finitevolX/commit/e1f8c8bddbcc62d39116dffd8d3ea8226087cce2))


### Bug Fixes

* address review comments on multilayer helper ([20bd2ba](https://github.com/jejjohnson/finitevolX/commit/20bd2ba9f1ff1c5cb149be543bec1bcf75f620a0))

## [0.0.20](https://github.com/jejjohnson/finitevolX/compare/v0.0.19...v0.0.20) (2026-03-11)


### Features

* add biharmonic (∇⁴) diffusion operator for C-grid staggered models ([80c22ad](https://github.com/jejjohnson/finitevolX/commit/80c22ad306e0a86ff4f21f32fcfaa1752b5410ee))
* add Coriolis force operator for staggered C-grid models ([9ce3f8d](https://github.com/jejjohnson/finitevolX/commit/9ce3f8d69ba5cafe20f2cbf71b9f5175f3c60f08))
* add Coriolis force operator for staggered C-grid models ([b0d70b6](https://github.com/jejjohnson/finitevolX/commit/b0d70b6a0406faa61160325fed220fdcbcac918a))
* add harmonic and biharmonic (∇⁴) diffusion operators for C-grid staggered models ([6a31f5d](https://github.com/jejjohnson/finitevolX/commit/6a31f5da02d3d41d58e76f147590d50c39aa6074))
* add horizontal diffusion operator (flux-form) for C-grid staggered models ([319ce69](https://github.com/jejjohnson/finitevolX/commit/319ce69865ed97ae14e310aeac7cca7fc858b3a7))
* add horizontal diffusion operator (flux-form) for C-grid staggered models ([44c0638](https://github.com/jejjohnson/finitevolX/commit/44c063852372880e07f28804d4d47e5270771702))
* merge main (PR [#97](https://github.com/jejjohnson/finitevolX/issues/97)) — reconcile with Diffusion2D/3D/diffusion_2d ([1a3408d](https://github.com/jejjohnson/finitevolX/commit/1a3408d225223a13a2d2caa6a9b6174efb0dea3f))


### Bug Fixes

* address PR review comments on diffusion module ([d8d9390](https://github.com/jejjohnson/finitevolX/commit/d8d9390425416c6faf6fd8291793d89e7864ccc8))
* address PR review comments on diffusion operator ([8442ead](https://github.com/jejjohnson/finitevolX/commit/8442eadeabed36e7c0a0881b9af0a94e866090bb))

## [0.0.19](https://github.com/jejjohnson/finitevolX/compare/v0.0.18...v0.0.19) (2026-03-11)


### Features

* add Arakawa (1966) Jacobian operator for energy- and enstrophy-conserving advection ([beb6e9a](https://github.com/jejjohnson/finitevolX/commit/beb6e9a0dadfc7dee7ef96ffacb0010b50ee3e98))
* add Arakawa (1966) Jacobian operator for energy- and enstrophy-conserving advection ([c7610f6](https://github.com/jejjohnson/finitevolX/commit/c7610f62a8562b2cd91e3ccfdf8d9c94906d071b))
* add energy-conserving momentum advection operator for C-grid models ([8ea2a21](https://github.com/jejjohnson/finitevolX/commit/8ea2a21f639665025b78f91709689112103496c5))
* add energy-conserving momentum advection operator for C-grid models ([db36273](https://github.com/jejjohnson/finitevolX/commit/db36273a49a36e931cf9e2c8bec5b1fb2ec6146f))
* add multi-layer vertical coupling matrix and layer↔mode transforms ([8cdafe9](https://github.com/jejjohnson/finitevolX/commit/8cdafe9e05824a82265d4ab774217ea3b9f5e9b0))
* add vertical coupling matrix (A) and layer↔mode transforms ([aa45bcc](https://github.com/jejjohnson/finitevolX/commit/aa45bcce97e107aed545e171517efe365d82caec))


### Bug Fixes

* address PR review - JAX-jittable vertical modes, fix annotations and docstring ([8fbf23a](https://github.com/jejjohnson/finitevolX/commit/8fbf23a4f389b6a9344985856d2f951a02395f2f))
* address PR review comments - fix jaxtyping shape annotation and test import path ([3ed8077](https://github.com/jejjohnson/finitevolX/commit/3ed80778639f8793bb6824f6fb46d39e04791ee3))
* resolve I001 import-sort error in tests and run ruff check on full repo ([9316961](https://github.com/jejjohnson/finitevolX/commit/93169614b7061983c41bc38fa0bb49e0923405ea))
* resolve ruff lint errors in test_momentum.py ([6faa3a2](https://github.com/jejjohnson/finitevolX/commit/6faa3a2f99976d203a96e18094703c6d3464800d))
* restrict momentum advection write region and enable x64 in tests ([3686fae](https://github.com/jejjohnson/finitevolX/commit/3686faee4debfddf08f7ec4fa1bf1890996d53fc))

## [0.0.18](https://github.com/jejjohnson/finitevolX/compare/v0.0.17...v0.0.18) (2026-03-11)


### Features

* add `grad_perp` operator to `Difference2D` for geostrophic velocity ([09ecb96](https://github.com/jejjohnson/finitevolX/commit/09ecb964cba85ccb17bd5d38fa1b4be9778d1e91))
* add grad_perp operator to Difference2D for geostrophic velocity ([fad6966](https://github.com/jejjohnson/finitevolX/commit/fad6966aad5272ecc85763443f54552be522a0cd))
* add Slip1D — free/no/partial-slip boundary condition for tangential velocity ([591a551](https://github.com/jejjohnson/finitevolX/commit/591a5511b2300caffb36dbb2038ee912bf051950))
* add SlipBC1D boundary condition for tangential velocity at solid walls ([8e7a9fa](https://github.com/jejjohnson/finitevolX/commit/8e7a9fad52563a60f4cb960795e4c93b0d3e3e9b))
* add standalone `Divergence2D` / `divergence_2d` operator ([56e4f0f](https://github.com/jejjohnson/finitevolX/commit/56e4f0f67e73029a50bd42a41a1a4b511105a825))
* add standalone Divergence2D / divergence_2d operator ([522c98b](https://github.com/jejjohnson/finitevolX/commit/522c98b798802b425a9f7b7068897a7edddf1a29))


### Bug Fixes

* address PR review - noflux 4-wall BC, dtype promotion, transform coverage ([940035b](https://github.com/jejjohnson/finitevolX/commit/940035b64ff262de1eeb073ecb9bf8c0edaceeb7))
* use direct stencil in grad_perp to avoid ghost-cell contamination ([1d80ccc](https://github.com/jejjohnson/finitevolX/commit/1d80ccc6cd21803e3ac5e297058eb55eac3076cd))

## [0.0.17](https://github.com/jejjohnson/finitevolX/compare/v0.0.16...v0.0.17) (2026-03-10)


### Features

* add WENO7 and WENO9 advection schemes ([6fd9ebc](https://github.com/jejjohnson/finitevolX/commit/6fd9ebc81ad46c720ffde40f85756f087d583d0b))

## [0.0.16](https://github.com/jejjohnson/finitevolX/compare/v0.0.15...v0.0.16) (2026-03-10)


### Features

* replace static PNG comparison plots with animated GIFs ([ebf4b18](https://github.com/jejjohnson/finitevolX/commit/ebf4b188f6c0fb4fede0e6a92042ab704cd43f99))
* replace static PNG comparison plots with animated GIFs ([9d813d1](https://github.com/jejjohnson/finitevolX/commit/9d813d1ccf640269762bf4cf5cd78927be917945))


### Bug Fixes

* address reviewer feedback on GIF animation changes ([94ebc36](https://github.com/jejjohnson/finitevolX/commit/94ebc3657ddd1580f3f804ca9e671499de99e162))

## [0.0.15](https://github.com/jejjohnson/finitevolX/compare/v0.0.14...v0.0.15) (2026-03-10)


### Features

* add TVD flux limiters, masked TVD schemes, and advection method support ([863b3ac](https://github.com/jejjohnson/finitevolX/commit/863b3ace4622a010a6e6fc3f84716c1cae745470))
* flux limiters and masked advection schemes ([7fc7654](https://github.com/jejjohnson/finitevolX/commit/7fc7654a56437464c2efa4e865372e05d2de79ec))


### Bug Fixes

* correct inaccurate comment on TVD stencil size in tvd_x_masked ([39db0c2](https://github.com/jejjohnson/finitevolX/commit/39db0c22dcab86696878f4006deee344c8975d99))
* keep jaxtyping shape strings compatible with ruff and ty ([263f491](https://github.com/jejjohnson/finitevolX/commit/263f49127b9ec57ed182519ac081f18c29a704f3))

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
