# Changelog

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
