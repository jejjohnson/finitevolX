# Changelog

## [0.1.0](https://github.com/jejjohnson/finitevolX/compare/v0.0.48...v0.1.0) (2026-10-17)


### ⚠ BREAKING CHANGES

* **coriolis:** Coriolis2D / Coriolis3D outputs change for spatially varying f (by O(dy^2 * grad f) in the interior); uniform-f results are unchanged up to round-off. With a mask, velocities on dry faces are ignored (Coriolis3D included). The Coriolis2D.interp attribute is removed.
* **diagnostics:** vertical_velocity returns [Nz, Ny, Nx] (was [Nz+1, Ny, Nx]) with w[k] the top face of T[k]; old w[k+1] on the physical levels is now w[k]. The z-ghost slabs of u and v are no longer integrated. Passing a raw float mask is deprecated in favour of Mask3D.
* **boundary:** wall_boundaries(field, "u") / ("v") now zero the east / north wall face (index -2) and write -field[-3] to the outside slot -1 instead of zeroing index -1.
* **advection:** WENO and TVD outputs change slightly (at most 4e-7 relative on the golden fields, more on fields far from O(1) amplitude, which were previously computed with an amplitude-dependent scheme).
* **advection:** masked Advection1D/2D/3D and SphericalAdvection2D/3D tendencies change at coastal cells: no flux crosses a dry face, and naive / upwind1/2/3 / weno7 / weno9 no longer read land values (they fall back to narrower stencils near the coast).  Unmasked outputs are unchanged.
* **diffusion:** Diffusion2D/3D, diffusion_2d, their fluxes and SphericalDiffusion2D/3D return different values for a spatially varying kappa: each face now uses the mean of its two cells instead of the western / southern cell. With wall="open" both wall faces average the edge and ghost kappa (before: the west / south wall face used the ghost kappa and the east / north one the edge kappa). Scalar and constant kappa are unchanged.
* **diffusion:** masked BiharmonicDiffusion2D/3D and SphericalBiharmonicDiffusion2D/3D return different values at wet cells within two cells of land: the coast is now no-flux for h and lap(h) and land values no longer enter the tendency. Unmasked results are unchanged.
* **multigrid:** `n_coarse` is removed from `build_multigrid_solver` and `MultigridSolver` (the bottom solve is exact). The defaults change to n_pre = n_post = 3 and omega = 0.8 (were 6, 6, 0.95), `MultigridLevel` stores a stencil instead of `cx` / `cy` / `wall`, and `make_multigrid_preconditioner` raises for n_pre != n_post.
* **multigrid:** `build_multigrid_solver` now defaults to bc="dirichlet". Callers solving a T-point pressure / free-surface problem with zero normal flux (e.g. the `helm_solve` of `barotropic_filter`) must pass bc="neumann" to keep the previous operator.
* **solvers:** pressure_from_divergence(method="cg") now solves the Neumann problem by default, and with bc="dst" puts p = 0 on the cell faces (the old cell-centre Dirichlet result is gone). pressure_from_divergence no longer accepts method="capacitance", known_values or known_mask. method="capacitance" and known values require bc="dst"; method="cg" with bc="fft" (and bc="dct" for the streamfunction) raises.
* **solvers:** method="spectral" results change for every streamfunction_from_vorticity / pv_inversion / pressure_from_divergence call: the wrappers take the full [Ny, Nx] array and return zeros on the ghost ring (and on the psi wall nodes); do not slice the interior by hand. Known values (known_values= and non-zero Dirichlet1D faces) now sit on the coast ring (the dry wall nodes) instead of the inner wet ring, every wet cell is solved for, and SolveDomain.all_known / effective_mask change accordingly.  bc.mask with method="spectral", and non-zero Dirichlet faces or spectral known values on pressure_from_divergence, now raise; so do non-zero Dirichlet faces on a Mask2D whose outer coast is not on the four walls.

### Features

* **forcing:** add WindCurl2D, the wind-stress curl forcing of vorticity / qg pv, and use it in the qg notebook ([#377](https://github.com/jejjohnson/finitevolX/issues/377)) ([986b69e](https://github.com/jejjohnson/finitevolX/commit/986b69e94c08aa6403cd32b650936acc6157d4c8)), closes [#314](https://github.com/jejjohnson/finitevolX/issues/314)
* **operators:** leading batch axes on the 2-D Cartesian operators ([#392](https://github.com/jejjohnson/finitevolX/issues/392)) ([796c144](https://github.com/jejjohnson/finitevolX/commit/796c14424c94e484f380b4bab1b866d82dbfdd68)), closes [#386](https://github.com/jejjohnson/finitevolX/issues/386)
* **pipekit:** diagnostic, reduction and time-step operators ([#395](https://github.com/jejjohnson/finitevolX/issues/395)) ([c3bd924](https://github.com/jejjohnson/finitevolX/commit/c3bd924c069d760cce93c6fd28a5d5e4a34e1301))
* **pipekit:** finitevolx.pipekit scaffolding — FVXOperator, GridSpec, check_operator ([#393](https://github.com/jejjohnson/finitevolX/issues/393)) ([b97a875](https://github.com/jejjohnson/finitevolX/commit/b97a87594160b33223578591425ff4f0f65cb2d0))
* **pipekit:** stencil, interpolation, boundary and mask operators ([#394](https://github.com/jejjohnson/finitevolX/issues/394)) ([90803f0](https://github.com/jejjohnson/finitevolX/commit/90803f0d2383191f8f88fd48acd2555899fbecaa)), closes [#388](https://github.com/jejjohnson/finitevolX/issues/388)
* **timestepping:** add AB3Solver with an rk3 start for diffeqsolve and document the ab3 start-up and leapfrog-raf orders ([#376](https://github.com/jejjohnson/finitevolX/issues/376)) ([eb92bf3](https://github.com/jejjohnson/finitevolX/commit/eb92bf3063419e7ef8b7a8b9a22efbbd757a8704))
* **vorticity:** rename the energy/enstrophy pv-flux blend to pv_flux_blend and deprecate the arakawa-lamb names ([#380](https://github.com/jejjohnson/finitevolX/issues/380)) ([be2ad27](https://github.com/jejjohnson/finitevolX/commit/be2ad279fb42c72706b4ba3f59afb648b628feef)), closes [#283](https://github.com/jejjohnson/finitevolX/issues/283)


### Bug Fixes

* **advection:** make the weno and tvd epsilons relative so reconstruction is amplitude-covariant ([#372](https://github.com/jejjohnson/finitevolX/issues/372)) ([02c8144](https://github.com/jejjohnson/finitevolX/commit/02c81444777047bff3e658cb1135eab5b8740c5d)), closes [#243](https://github.com/jejjohnson/finitevolX/issues/243) [#294](https://github.com/jejjohnson/finitevolX/issues/294)
* **advection:** zero the dry-face fluxes and make every method land-safe under a mask ([#368](https://github.com/jejjohnson/finitevolX/issues/368)) ([bc19747](https://github.com/jejjohnson/finitevolX/commit/bc19747b5f7fa21672aeaeec19af9bdc9fab2063)), closes [#296](https://github.com/jejjohnson/finitevolX/issues/296)
* **boundary:** make the bc atoms stagger-aware so a closed wall stops u and v on the wall faces ([#378](https://github.com/jejjohnson/finitevolX/issues/378)) ([440cc96](https://github.com/jejjohnson/finitevolX/commit/440cc9645b78f8710288b541f0396f0f35af6df4)), closes [#286](https://github.com/jejjohnson/finitevolX/issues/286)
* **coriolis:** use the energy-conserving corner-product form so the coriolis term conserves energy on a beta-plane ([#381](https://github.com/jejjohnson/finitevolX/issues/381)) ([fcc1c7f](https://github.com/jejjohnson/finitevolX/commit/fcc1c7f3d00ea146121f2b981e3614266773a338)), closes [#289](https://github.com/jejjohnson/finitevolX/issues/289)
* **diagnostics:** vertical_velocity integrates the physical levels only and returns [Nz, Ny, Nx] like Mask3D.w ([#379](https://github.com/jejjohnson/finitevolX/issues/379)) ([eed41df](https://github.com/jejjohnson/finitevolX/commit/eed41df7e81820c58979b9f90f62f24ebfb523d7)), closes [#290](https://github.com/jejjohnson/finitevolX/issues/290) [#210](https://github.com/jejjohnson/finitevolX/issues/210)
* **diffusion:** apply the masked no-flux operator twice in masked biharmonic diffusion ([#366](https://github.com/jejjohnson/finitevolX/issues/366)) ([eaa0b05](https://github.com/jejjohnson/finitevolX/commit/eaa0b057f6d87daa1de6f9180190f8dc23792628))
* **diffusion:** average a variable kappa to the faces in harmonic diffusion ([#367](https://github.com/jejjohnson/finitevolX/issues/367)) ([008b282](https://github.com/jejjohnson/finitevolX/commit/008b28254d8f5902b38fec1c5cd7670ca479eb2d))
* **multigrid:** add a dirichlet wall (the default) matching masked_laplacian ([#361](https://github.com/jejjohnson/finitevolX/issues/361)) ([8df8903](https://github.com/jejjohnson/finitevolX/commit/8df8903b2939f9645792252b335b1a7abc6748e8)), closes [#304](https://github.com/jejjohnson/finitevolX/issues/304) [#277](https://github.com/jejjohnson/finitevolX/issues/277)
* **multigrid:** full-depth coarsening, exact bottom solve and a symmetric v-cycle ([#362](https://github.com/jejjohnson/finitevolX/issues/362)) ([f2de21b](https://github.com/jejjohnson/finitevolX/commit/f2de21b66dd1733c5788fc71a7d9568a335e3179)), closes [#305](https://github.com/jejjohnson/finitevolX/issues/305) [#344](https://github.com/jejjohnson/finitevolX/issues/344) [#277](https://github.com/jejjohnson/finitevolX/issues/277)
* **solvers:** honour bc in the masked pressure solve and accept a traced scalar lambda ([#360](https://github.com/jejjohnson/finitevolX/issues/360)) ([86dc542](https://github.com/jejjohnson/finitevolX/commit/86dc542d080e8ced8dc9e85afc80c4114ed0c36f)), closes [#308](https://github.com/jejjohnson/finitevolX/issues/308) [#309](https://github.com/jejjohnson/finitevolX/issues/309)
* **solvers:** keep the rhs dtype when a Mask2D is used ([#355](https://github.com/jejjohnson/finitevolX/issues/355)) ([ca2aafd](https://github.com/jejjohnson/finitevolX/commit/ca2aafdbebba1d608717f539c6fd3fb4bef60f49)), closes [#302](https://github.com/jejjohnson/finitevolX/issues/302)
* **solvers:** make masked_laplacian zero-padded instead of periodic at the array edge ([#359](https://github.com/jejjohnson/finitevolX/issues/359)) ([a713f2b](https://github.com/jejjohnson/finitevolX/commit/a713f2b7ee805e612e9ac6c5aa29ab17fe3ab7f5)), closes [#310](https://github.com/jejjohnson/finitevolX/issues/310)
* **solvers:** put the psi wall on the c-grid wall nodes for every method ([#356](https://github.com/jejjohnson/finitevolX/issues/356)) ([a26417e](https://github.com/jejjohnson/finitevolX/commit/a26417e271dac583084e18c548b986fb83a18161)), closes [#300](https://github.com/jejjohnson/finitevolX/issues/300) [#303](https://github.com/jejjohnson/finitevolX/issues/303)
* **solvers:** report cg non-convergence instead of raising and make the preconditioned solve differentiable ([#358](https://github.com/jejjohnson/finitevolX/issues/358)) ([be058d1](https://github.com/jejjohnson/finitevolX/commit/be058d10ac1f7eee961077e44b0f039da40122d1)), closes [#307](https://github.com/jejjohnson/finitevolX/issues/307)
* **solvers:** solve the capacitance problem on the cg domain and default to a dst base ([#357](https://github.com/jejjohnson/finitevolX/issues/357)) ([a058edf](https://github.com/jejjohnson/finitevolX/commit/a058edfd12c0a6709c07739b6c8618844f5487a5)), closes [#306](https://github.com/jejjohnson/finitevolX/issues/306) [#301](https://github.com/jejjohnson/finitevolX/issues/301)
* **spherical:** add wall="open" to SphericalAdvection2D/3D so the flux divergence telescopes ([#369](https://github.com/jejjohnson/finitevolX/issues/369)) ([13a4758](https://github.com/jejjohnson/finitevolX/commit/13a475814ba1bcd917332935334f894d06358be8)), closes [#297](https://github.com/jejjohnson/finitevolX/issues/297) [#247](https://github.com/jejjohnson/finitevolX/issues/247)
* **timestepping:** make every diffrax wrapper run inside diffeqsolve and give the adaptive ones real error estimates ([#375](https://github.com/jejjohnson/finitevolX/issues/375)) ([78e3376](https://github.com/jejjohnson/finitevolX/commit/78e3376f86013677dd12bd841deabb1d92b9b713)), closes [#313](https://github.com/jejjohnson/finitevolX/issues/313)

## [0.0.48](https://github.com/jejjohnson/finitevolX/compare/v0.0.47...v0.0.48) (2026-10-14)


### Features

* **solvers:** expose the Helmholtz operator as a structured gaussx operator ([#341](https://github.com/jejjohnson/finitevolX/issues/341)) ([4a61df8](https://github.com/jejjohnson/finitevolX/commit/4a61df8f0a00553ee160a531fdd8d29b5185aa60))
* **solvers:** expose the multigrid V-cycle as a gaussx preconditioner ([#340](https://github.com/jejjohnson/finitevolX/issues/340)) ([62921dc](https://github.com/jejjohnson/finitevolX/commit/62921dc7df0c9d365b6d82118b35483a75b85f8c))


### Bug Fixes

* **advection:** align negative-flow face in functional reconstruct ([#348](https://github.com/jejjohnson/finitevolX/issues/348)) ([7558278](https://github.com/jejjohnson/finitevolX/commit/7558278ae89f5fa9eae97ddfa80c3a9676383172)), closes [#293](https://github.com/jejjohnson/finitevolX/issues/293)
* **boundary:** use the outward normal on every face for neumann and robin ([#345](https://github.com/jejjohnson/finitevolX/issues/345)) ([449abf5](https://github.com/jejjohnson/finitevolX/commit/449abf52f8671b545ebd2f2de64fd2c80f3527e0)), closes [#285](https://github.com/jejjohnson/finitevolX/issues/285)
* **diagnostics:** apply f0 to the multilayer qg stretching term ([#346](https://github.com/jejjohnson/finitevolX/issues/346)) ([567a598](https://github.com/jejjohnson/finitevolX/commit/567a5988f987af70ee513a00132636d5e86c2dbd)), closes [#282](https://github.com/jejjohnson/finitevolX/issues/282)
* **operators:** correct small doc, comment and test mismatches ([#349](https://github.com/jejjohnson/finitevolX/issues/349)) ([2e2eaa1](https://github.com/jejjohnson/finitevolX/commit/2e2eaa161dde1dc4acab3b13f209031408f57991)), closes [#291](https://github.com/jejjohnson/finitevolX/issues/291)
* **solvers:** make solve_cg's default atol reachable in float32 ([#339](https://github.com/jejjohnson/finitevolX/issues/339)) ([08044e9](https://github.com/jejjohnson/finitevolX/commit/08044e91cfccc9247a38edf9a48044e79636dec2))
* **spherical:** read grid.cos_lat_V for V-face cosine and guard masked spherical pv ([#350](https://github.com/jejjohnson/finitevolX/issues/350)) ([ccf5540](https://github.com/jejjohnson/finitevolX/commit/ccf55404870d569d6ef9c1e1df737e29bc85dc7a)), closes [#287](https://github.com/jejjohnson/finitevolX/issues/287) [#288](https://github.com/jejjohnson/finitevolX/issues/288)
* **timestepping:** make imex_ssp2_step second order in the implicit part ([#347](https://github.com/jejjohnson/finitevolX/issues/347)) ([0af059c](https://github.com/jejjohnson/finitevolX/commit/0af059c5b05b16e997f1e674f891c65a28ff9399)), closes [#311](https://github.com/jejjohnson/finitevolX/issues/311)
* **vorticity:** swap the energy/enstrophy PV-flux forms and fill the ghost-corner row ([#351](https://github.com/jejjohnson/finitevolX/issues/351)) ([446ac6e](https://github.com/jejjohnson/finitevolX/commit/446ac6e9d36cb801cc59ba03b71e671487325235)), closes [#284](https://github.com/jejjohnson/finitevolX/issues/284). **Behaviour change:** `pv_flux_energy_conserving` / `pv_flux_enstrophy_conserving` and the `MomentumAdvection*` `'energy'` / `'enstrophy'` schemes now compute what their names say, so their outputs trade places relative to 0.0.47.


### Performance Improvements

* **operators:** lower stencils to one fused pass and unroll WENO-7/9 indicators ([#337](https://github.com/jejjohnson/finitevolX/issues/337)) ([7debd0a](https://github.com/jejjohnson/finitevolX/commit/7debd0a53159cdaca359ca4401b2f3829fa94c6c))

## [0.0.47](https://github.com/jejjohnson/finitevolX/compare/v0.0.46...v0.0.47) (2026-09-24)


### Features

* **diagnostics:** add Energetics2D and Strain2D mask-aware classes ([#206](https://github.com/jejjohnson/finitevolX/issues/206), 1/4) ([#267](https://github.com/jejjohnson/finitevolX/issues/267)) ([f60986f](https://github.com/jejjohnson/finitevolX/commit/f60986f97dc84928c35c92a6857c0545d91ab2ca))
* **diagnostics:** add Vorticity2D enstrophy methods + QGPotentialVorticity2D ([#206](https://github.com/jejjohnson/finitevolX/issues/206), 2/4) ([#268](https://github.com/jejjohnson/finitevolX/issues/268)) ([e5d2529](https://github.com/jejjohnson/finitevolX/commit/e5d252973dba2b049f666691c4b845552f9c740e))
* **forcing:** add 2D wind-stress, drag and Rayleigh damping operators ([#258](https://github.com/jejjohnson/finitevolX/issues/258)) ([eeae02e](https://github.com/jejjohnson/finitevolX/commit/eeae02e72523ad5dfe645f6f3f242571d616f6d2))
* **forcing:** add 3D wind-stress, drag and Rayleigh damping operators ([#259](https://github.com/jejjohnson/finitevolX/issues/259)) ([9a8a886](https://github.com/jejjohnson/finitevolX/commit/9a8a8864abb514e56ab342a01fe4fc0e07cf4045))
* **forcing:** add AbstractForcing base class and composition wrappers ([#257](https://github.com/jejjohnson/finitevolX/issues/257)) ([851225b](https://github.com/jejjohnson/finitevolX/commit/851225bc0a7ca7431d7ca93c4e4e89636457b8d0)), closes [#173](https://github.com/jejjohnson/finitevolX/issues/173)
* **forcing:** add Layer 0 functional forcing primitives ([#256](https://github.com/jejjohnson/finitevolX/issues/256)) ([19f388f](https://github.com/jejjohnson/finitevolX/commit/19f388f9b3f2a62d95a1299e25c0484c0de83ad2))
* **operators:** add ArakawaJacobian2D + SphericalDifference2D.geostrophic_velocity ([#206](https://github.com/jejjohnson/finitevolX/issues/206), 3/4) ([#269](https://github.com/jejjohnson/finitevolX/issues/269)) ([f757637](https://github.com/jejjohnson/finitevolX/commit/f757637e14c37f9d6804cbd570571939704318e6))
* **solvers:** accept a BoundaryConditionSet as bc= in elliptic wrappers ([#263](https://github.com/jejjohnson/finitevolX/issues/263)) ([0350a90](https://github.com/jejjohnson/finitevolX/commit/0350a908623a7ebec7bec8bc92b223b2e4b07608))
* **solvers:** accept known_values/known_mask in elliptic wrappers ([#262](https://github.com/jejjohnson/finitevolX/issues/262)) ([2fe5f71](https://github.com/jejjohnson/finitevolX/commit/2fe5f71cb60a58a0f99bf3ba65bdacc430315402))

## [0.0.46](https://github.com/jejjohnson/finitevolX/compare/v0.0.45...v0.0.46) (2026-09-24)


### Bug Fixes

* **vertical:** diagonalize the non-symmetric coupling matrix correctly ([#252](https://github.com/jejjohnson/finitevolX/issues/252)) ([276a288](https://github.com/jejjohnson/finitevolX/commit/276a28827389753d7157a308ecab7bce16393666))

## [0.0.45](https://github.com/jejjohnson/finitevolX/compare/v0.0.44...v0.0.45) (2026-09-12)


### Features

* **grid:** physical cell-width helpers on spherical grids ([6fae0d6](https://github.com/jejjohnson/finitevolX/commit/6fae0d6c43239ee50d5cfa3bab6cf2422647b20a))
* **grid:** physical cell-width helpers on spherical grids ([409404b](https://github.com/jejjohnson/finitevolX/commit/409404b7423617fe4769ae29bdbd8f9ac547433c))
* **reductions:** mask-aware sample statistics over a leading axis ([adeabf0](https://github.com/jejjohnson/finitevolX/commit/adeabf0dce8cced7b179fa306388bfcd13764ee3))
* **reductions:** mask-aware sample statistics over a leading axis ([9525c29](https://github.com/jejjohnson/finitevolX/commit/9525c29a745f62b1802f5fb4fe257bcab8d2aa70))


### Bug Fixes

* **grid:** keep rejecting genuinely negative zonal widths ([06b3986](https://github.com/jejjohnson/finitevolX/commit/06b398654240160a0f2fb76b1fae178545c65400))
* **grid:** recognise positive pole roundoff as a degenerate width ([d8f8bb1](https://github.com/jejjohnson/finitevolX/commit/d8f8bb15f822b0b698441cd9e60661ba575252ed))
* **reductions:** accumulate masked statistics in at least float32 ([1f1c78c](https://github.com/jejjohnson/finitevolX/commit/1f1c78c25f54b09b281ff6f0b71e98d895b8e60d))
* **reductions:** honour any eps, and keep dry cells at identity for ddof &lt; 0 ([7cb775a](https://github.com/jejjohnson/finitevolX/commit/7cb775a764e9f099fe53f4335efbda7bdfa86fa2))
* **reductions:** honour the requested floor, and stop sizing the count ([ddd5153](https://github.com/jejjohnson/finitevolX/commit/ddd5153a821a58b602dd946b76dc9b1feb2e2cf4))

## [0.0.44](https://github.com/jejjohnson/finitevolX/compare/v0.0.43...v0.0.44) (2026-07-17)


### Features

* **operators:** opt-in "open" lateral-wall mode for advection & diffusion ([#235](https://github.com/jejjohnson/finitevolX/issues/235)) ([89e43bb](https://github.com/jejjohnson/finitevolX/commit/89e43bb54f358715bafd7b17bb7a2999ac5b0648))

## [0.0.43](https://github.com/jejjohnson/finitevolX/compare/v0.0.42...v0.0.43) (2026-06-07)


### Features

* **operators:** add massh gap-analysis note and the operators it proposes ([#230](https://github.com/jejjohnson/finitevolX/issues/230)) ([a72f722](https://github.com/jejjohnson/finitevolX/commit/a72f722e603795c07d286ed96bf8c2ced4da1505))

## [0.0.42](https://github.com/jejjohnson/finitevolX/compare/v0.0.41...v0.0.42) (2026-06-02)


### Features

* add linear equation of state module ([#226](https://github.com/jejjohnson/finitevolX/issues/226)) ([3e80896](https://github.com/jejjohnson/finitevolX/commit/3e80896023c921b5c02af3c134914f11faa9e00e))
* boundary_ring, SolveDomain, KnownValueLifting for inhomogeneous BCs ([#224](https://github.com/jejjohnson/finitevolX/issues/224)) ([31c7cb2](https://github.com/jejjohnson/finitevolX/commit/31c7cb20a0420ac21fcaffe1fe9a62d6259a773f))

## [0.0.41](https://github.com/jejjohnson/finitevolX/compare/v0.0.40...v0.0.41) (2026-04-20)


### Features

* spherical diffusion and advection operators ([#165](https://github.com/jejjohnson/finitevolX/issues/165)) ([#216](https://github.com/jejjohnson/finitevolX/issues/216)) ([33fb9e7](https://github.com/jejjohnson/finitevolX/commit/33fb9e778c339e2fe2809479b97f5f38b18fe820))
* spherical follow-ups — biharmonic, momentum advection, reductions ([#221](https://github.com/jejjohnson/finitevolX/issues/221)) ([f36ba97](https://github.com/jejjohnson/finitevolX/commit/f36ba97fc2d8246fee7ec2ce6e4b0f313ce54d64))

## [0.0.40](https://github.com/jejjohnson/finitevolX/compare/v0.0.39...v0.0.40) (2026-04-11)


### Features

* operator-attribute mask API (Mask1D/2D/3D class fields, [#209](https://github.com/jejjohnson/finitevolX/issues/209)) ([#211](https://github.com/jejjohnson/finitevolX/issues/211)) ([9f67b18](https://github.com/jejjohnson/finitevolX/commit/9f67b18ea75488aedb634fe510eb12d881385668))
* **operators:** operator-attribute mask API ([#209](https://github.com/jejjohnson/finitevolX/issues/209)) ([9f67b18](https://github.com/jejjohnson/finitevolX/commit/9f67b18ea75488aedb634fe510eb12d881385668))

### Breaking changes — grid and mask submodule restructure ([#207](https://github.com/jejjohnson/finitevolX/issues/207), [#208](https://github.com/jejjohnson/finitevolX/issues/208))

The internal `_src/grid/` and `_src/mask/` packages were reorganised
to support a layered abstraction over coordinate systems and grid
dimensions.  Public-API renames are listed below; users who imported
through the top-level `finitevolx` namespace need to update class
names but no import-path changes.

**Grid module** (`refactor: restructure into base / cartesian / spherical / cylindrical layout`)

- `ArakawaCGrid1D / 2D / 3D` (the old concrete classes) → renamed to
  `CartesianGrid1D / 2D / 3D`.
- `SphericalArakawaCGrid2D / 3D` → renamed to `SphericalGrid2D / 3D`.
  These are now siblings of `CartesianGrid*` under a common
  `CurvilinearGrid*` base, *not* subclasses.
- The names `ArakawaCGrid1D / 2D / 3D` are reused as a new abstract
  topology layer (just `Nx`, `Ny`, `Nz`).
- New abstract layer `CurvilinearGrid1D / 2D / 3D` adds the uniform
  metric (`Lx`, `Ly`, `Lz`, `dx`, `dy`, `dz`).
- New `CylindricalGrid` stub (TODO).
- Path moves: `_src/grid/cgrid_mask.py` → `_src/mask/cgrid_mask.py`,
  `_src/grid/constants.py` → `_src/utils/constants.py`.

**Mask module** (`refactor: split into base / cartesian / spherical / cylindrical`)

- `ArakawaCGridMask` → renamed to `Mask2D`.
- New `Mask1D` and `Mask3D` concrete classes.  `Mask3D` includes a
  full vertical face mask (`w`) — see below.
- `StencilCapability` → split into `StencilCapability1D / 2D / 3D`.
- `_src/mask/cgrid_mask.py` → split into `_src/mask/{base,cartesian}.py`
  with TODO stubs at `_src/mask/{spherical,cylindrical}.py`.
- New `_src/mask/utils.py` with public n-D-aware mask construction
  primitives: `pool_bool`, `h_from_pooled`, `dilate_mask`,
  `count_contiguous`, `make_sponge`.

**Mask2D field renames** (frees up `w` for the 3-D vertical face mask):

| Old name              | New name                            |
|-----------------------|-------------------------------------|
| `w`                   | `xy_corner` (lenient)               |
| `psi`                 | `xy_corner_strict` (strict)         |
| `not_w`, `not_psi`    | `not_xy_corner`, `not_xy_corner_strict` |
| `w_valid`             | `xy_corner_valid`                   |
| `w_vertical_bound`    | `xy_corner_y_wall`                  |
| `w_horizontal_bound`  | `xy_corner_x_wall`                  |
| `w_cornerout_bound`   | `xy_corner_convex`                  |
| `psi_irrbound_xids`   | `xy_corner_strict_irrbound_cols`    |
| `psi_irrbound_yids`   | `xy_corner_strict_irrbound_rows`    |

The last pair also fixes a long-standing naming bug: the old `_xids`
field stored column indices and `_yids` stored row indices.

**Removed:**

- `Mask2D.distbound1 / distbound2 / distbound3plus` properties — these
  were exact aliases for `ind_coast / ind_near_coast / ind_ocean`.
- `visualize_masks()` function — plotting code does not belong in the
  library.  Equivalent inline code is available in
  `docs/notebooks/demo_masks.py`.

## [0.0.39](https://github.com/jejjohnson/finitevolX/compare/v0.0.38...v0.0.39) (2026-03-24)


### Features

* expose functional boundary helpers for wall, padding, and corners ([19312bf](https://github.com/jejjohnson/finitevolX/commit/19312bf3579cf50fbe1cced8ea70181490c7c796))
* expose functional boundary helpers for wall, padding, and corners ([#152](https://github.com/jejjohnson/finitevolX/issues/152)) ([0d7277a](https://github.com/jejjohnson/finitevolX/commit/0d7277ab2486a733bd486837c7bc8becb71d318d))
* expose raw face fluxes as public functional API ([6298323](https://github.com/jejjohnson/finitevolX/commit/6298323ee00ae440d420571d7340dae8ee3ed16c))
* expose raw face fluxes as public functional API ([#143](https://github.com/jejjohnson/finitevolX/issues/143)) ([291be3b](https://github.com/jejjohnson/finitevolX/commit/291be3b53d68c3523a76adbf80731b86a70b2566))
* expose reconstruction pipeline as public functional API ([47748d4](https://github.com/jejjohnson/finitevolX/commit/47748d49ed41e08e351b72854bf7248a4c3fc44b))
* expose reconstruction pipeline as public functional API ([#151](https://github.com/jejjohnson/finitevolX/issues/151)) ([945f15d](https://github.com/jejjohnson/finitevolX/commit/945f15d3f118f4c67f78214e4a76020f4e0793f7))


### Bug Fixes

* address PR [#197](https://github.com/jejjohnson/finitevolX/issues/197) review comments ([46890be](https://github.com/jejjohnson/finitevolX/commit/46890beb5a964857152a52a9bc759d6e86e7158d))
* address PR [#198](https://github.com/jejjohnson/finitevolX/issues/198) review comments ([902c55d](https://github.com/jejjohnson/finitevolX/commit/902c55dba4831e43145407febeee561124b1e9f3))
* address PR [#199](https://github.com/jejjohnson/finitevolX/issues/199) review comments ([8a632e6](https://github.com/jejjohnson/finitevolX/commit/8a632e69c573543c89ca87ec1f503c9577ba2b16))

## [0.0.38](https://github.com/jejjohnson/finitevolX/compare/v0.0.37...v0.0.38) (2026-03-24)


### Features

* upgrade spectraldiffx dependency to &gt;=0.0.10 ([af8c9a8](https://github.com/jejjohnson/finitevolX/commit/af8c9a87764dcb2a1ef6d74793b328614034cbd1))
* upgrade spectraldiffx to &gt;=0.0.10 ([336e7a2](https://github.com/jejjohnson/finitevolX/commit/336e7a2de1824c7ff32bd3c9ea1159c93f8aca6f))


### Bug Fixes

* address PR review comments ([3871618](https://github.com/jejjohnson/finitevolX/commit/3871618625c562de099ed802903f143ecc81c28d))
* address PR review comments on advection tests ([8d873f3](https://github.com/jejjohnson/finitevolX/commit/8d873f3abe57d09ba0bff00edcaa6dfe3949092e))
* nyström preconditioner false convergence with low rank ([#187](https://github.com/jejjohnson/finitevolX/issues/187)) ([91080d2](https://github.com/jejjohnson/finitevolX/commit/91080d2f51ee9db985d4d9ccd806be4e3346b912))
* Nyström preconditioner full-rank correction for low-rank usage ([#187](https://github.com/jejjohnson/finitevolX/issues/187)) ([a0517ea](https://github.com/jejjohnson/finitevolX/commit/a0517eaaaef5340757ef7bf49b6ecb6be3d8f753))
* use non-trivial mask in Nystrom masked-domain test ([61b749f](https://github.com/jejjohnson/finitevolX/commit/61b749f9720c2bf60df4abd6ef57de09486409b4))
* weno7/9 right-biased reconstruction for negative flow ([c906949](https://github.com/jejjohnson/finitevolX/commit/c90694904b0b22363017e333cc5cc1297bdb15cb))
* WENO7/9 right-biased reconstruction for negative flow ([ad07fa6](https://github.com/jejjohnson/finitevolX/commit/ad07fa6de967c3df76e053392038d9dcd6cdd0f5)), closes [#191](https://github.com/jejjohnson/finitevolX/issues/191)

## [0.0.37](https://github.com/jejjohnson/finitevolX/compare/v0.0.36...v0.0.37) (2026-03-20)


### Features

* **docs:** add three advection tutorial notebooks ([1e12210](https://github.com/jejjohnson/finitevolX/commit/1e122105361f1f3fd9fef39d62f73dd2e67bdfce))


### Bug Fixes

* address PR [#184](https://github.com/jejjohnson/finitevolX/issues/184) review comments ([afe7428](https://github.com/jejjohnson/finitevolX/commit/afe7428a03c6dda23ee79723703baf5fa9a2d5b5))
* address PR [#184](https://github.com/jejjohnson/finitevolX/issues/184) review comments (round 2) ([8eaa904](https://github.com/jejjohnson/finitevolX/commit/8eaa9045f7375022c08dfb3c108aa3341b2bbe2c))
* **advection:** correct right-biased WENO reconstruction for negative flow ([db5b079](https://github.com/jejjohnson/finitevolX/commit/db5b0792d97fc19ee379ebadc6b8f45ebbfb86eb))
* **docs:** add notebook symlinks so mkdocs-jupyter can find them ([5d9fecf](https://github.com/jejjohnson/finitevolX/commit/5d9fecf3cfce2690a5451894a70372348bef671c))
* **docs:** correct image paths — use ../../images/ for mkdocs-jupyter ([ef0f18b](https://github.com/jejjohnson/finitevolX/commit/ef0f18ba5cbe79c271c3fb6d2eab48d1f40ac590))
* **docs:** embed all figures as markdown images for non-executed rendering ([98c776b](https://github.com/jejjohnson/finitevolX/commit/98c776b7535358bd6b7ff03594d5b558531687cb))
* **docs:** enable MathJax rendering in mkdocs-jupyter notebooks ([42cf22d](https://github.com/jejjohnson/finitevolX/commit/42cf22d0a145bb8524bee0b9d81d868dabd57864))
* **docs:** move notebooks to docs/ and embed figures for mkdocs rendering ([81be84d](https://github.com/jejjohnson/finitevolX/commit/81be84db436d91e0cb789cbcf208820a270baa72))
* **docs:** remove restrictive MathJax class filter breaking notebook math ([1549298](https://github.com/jejjohnson/finitevolX/commit/15492984edab8af18c990ab32379fb521b8f7fba))
* **docs:** use constrained_layout for masked advection plots ([b91a05b](https://github.com/jejjohnson/finitevolX/commit/b91a05b549ece54a3442cee76cd4fec502656cfb))

## [0.0.36](https://github.com/jejjohnson/finitevolX/compare/v0.0.35...v0.0.36) (2026-03-19)


### Bug Fixes

* address PR review feedback ([a7484b0](https://github.com/jejjohnson/finitevolX/commit/a7484b07178bf8e3347615ce88e0f6af67ad614b))
* resolve docs build failure and reorganize API reference ([0a01584](https://github.com/jejjohnson/finitevolX/commit/0a015849855590c07f976ac51a4ec805cd3ffae5))
* resolve docs build failure from spectraldiffx re-exports ([30969d4](https://github.com/jejjohnson/finitevolX/commit/30969d4f1c858ef26f51c9e0c8dc3af7f2d235f4))

## [0.0.35](https://github.com/jejjohnson/finitevolX/compare/v0.0.34...v0.0.35) (2026-03-18)


### Features

* add spherical compound operators (divergence, vorticity, laplacian) ([d96505f](https://github.com/jejjohnson/finitevolX/commit/d96505f7b2d0a3d7b2984a4ec2d8168c96cdb5bc))
* add SphericalArakawaCGrid2D and SphericalArakawaCGrid3D ([d66186c](https://github.com/jejjohnson/finitevolX/commit/d66186ceac4f65345db3e40d43cec80eda6a7ec7))
* add SphericalDifference2D/3D and extract _safe_div_cos utility ([0cb35a6](https://github.com/jejjohnson/finitevolX/commit/0cb35a69bc66d5bd0227e7d58b2e530c176204e4))
* export spherical operators and fix CI issues ([820f3e0](https://github.com/jejjohnson/finitevolX/commit/820f3e013923d37f5ef5c22938f95a1b880c5388))
* introduce spherical Arakawa C-grid operators ([d277ead](https://github.com/jejjohnson/finitevolX/commit/d277ead0f163eaf84a2b1fd72d8ca4a22b2939f4))


### Bug Fixes

* zero z-ghost slices in 3D spherical operators ([bff01e9](https://github.com/jejjohnson/finitevolX/commit/bff01e96a3f37b5cadb1f979040905a62345d844))

## [0.0.34](https://github.com/jejjohnson/finitevolX/compare/v0.0.33...v0.0.34) (2026-03-17)


### Bug Fixes

* export interior() and fix 1D jaxtyping shape strings ([06bdede](https://github.com/jejjohnson/finitevolX/commit/06bdede936346e63eaff93a0a8f9382b65d5036f))

## [0.0.33](https://github.com/jejjohnson/finitevolX/compare/v0.0.32...v0.0.33) (2026-03-17)


### Bug Fixes

* correct _interp_T_to_X stencil to match Interpolation2D.T_to_X ([4852194](https://github.com/jejjohnson/finitevolX/commit/4852194d18e6e6be9874f4a06d10978383270b26))
* remove unrelated files accidentally committed ([116a640](https://github.com/jejjohnson/finitevolX/commit/116a640eb00fda1ae7949164fbbfe5b37d82042d))
* zero z-ghost slices in Vorticity3D.relative_vorticity ([f5a3d44](https://github.com/jejjohnson/finitevolX/commit/f5a3d44515f975adbb6efcf72888e496d9d79183))

## [0.0.32](https://github.com/jejjohnson/finitevolX/compare/v0.0.31...v0.0.32) (2026-03-17)


### Features

* add QG/SWM convenience utilities ([d7cf5bc](https://github.com/jejjohnson/finitevolX/commit/d7cf5bc87f654b25fdc93d584aa5fb1c6499277c))
* add QG/SWM convenience utilities (closes [#153](https://github.com/jejjohnson/finitevolX/issues/153)) ([4c749ca](https://github.com/jejjohnson/finitevolX/commit/4c749ca822930af012230ccda84ffb1ca3240058))
* add shallow-water PV from prognostic variables ([db73d3e](https://github.com/jejjohnson/finitevolX/commit/db73d3ea7f2be1342c54611966ed4a7288b52aa7))


### Bug Fixes

* address PR review comments on docstrings ([4e619b5](https://github.com/jejjohnson/finitevolX/commit/4e619b51189448abfb6878df6071a08d594b42ca))

## [0.0.31](https://github.com/jejjohnson/finitevolX/compare/v0.0.30...v0.0.31) (2026-03-13)


### Bug Fixes

* resolve ty 0.0.23 invalid-argument-type errors in upwind_flux callers ([f94d008](https://github.com/jejjohnson/finitevolX/commit/f94d0086e3ff8dde16816d0c4a8ed4303ad091b0))

## [0.0.30](https://github.com/jejjohnson/finitevolX/compare/v0.0.29...v0.0.30) (2026-03-13)


### Features

* **solvers:** add geometric multigrid Helmholtz solver ([f33c687](https://github.com/jejjohnson/finitevolX/commit/f33c687e74f3392ce99f9374d34c67a0ec8f8d39)), closes [#71](https://github.com/jejjohnson/finitevolX/issues/71)


### Bug Fixes

* **docs:** configure MathJax equation rendering with $ delimiters ([51313c9](https://github.com/jejjohnson/finitevolX/commit/51313c9dd4a844f62857dd46a159bf1b3f403e80))
* **solvers:** add QR orthogonalization to Nyström preconditioner ([981312d](https://github.com/jejjohnson/finitevolX/commit/981312dbc7854d33b4cd18136eaab5676035a897))
* **solvers:** address PR review comments on multigrid ([c37166e](https://github.com/jejjohnson/finitevolX/commit/c37166e34657b0aecac23c948c8e06890e3c213a))

## [0.0.29](https://github.com/jejjohnson/finitevolX/compare/v0.0.28...v0.0.29) (2026-03-13)


### Features

* **solvers:** add convenience wrappers and refactor elliptic module ([8b9d233](https://github.com/jejjohnson/finitevolX/commit/8b9d233003e3c2b87918916e0805bb994a274fc8)), closes [#87](https://github.com/jejjohnson/finitevolX/issues/87)
* **solvers:** add elliptic solver wrappers and modularize solver package ([78b0d4d](https://github.com/jejjohnson/finitevolX/commit/78b0d4d4fe1ec7ad830d771bc9aea5a2563336f7))
* **solvers:** add preconditioners module with spectral and Nyström methods ([b1e60fb](https://github.com/jejjohnson/finitevolX/commit/b1e60fbd1ad6b5e4613fdc092a827d963ce4b6e7))


### Bug Fixes

* **solvers:** address PR review comments on elliptic module ([014bf95](https://github.com/jejjohnson/finitevolX/commit/014bf95455bae9949d2a66cf55e917652c4ba35e))
* **solvers:** make _spectral_solve vmap-safe by using Helmholtz dispatch unconditionally ([a19f5ca](https://github.com/jejjohnson/finitevolX/commit/a19f5ca00d68d98e3edfeb497ea643309d975ec7))
* **tests:** make Nyström CG test robust to convergence edge cases ([62f3f34](https://github.com/jejjohnson/finitevolX/commit/62f3f342f5f6e8e31de0d988f765d3ff81933889))

## [0.0.28](https://github.com/jejjohnson/finitevolX/compare/v0.0.27...v0.0.28) (2026-03-12)


### Features

* **elliptic:** make spectral Helmholtz solvers vmap-compatible ([9b6d51d](https://github.com/jejjohnson/finitevolX/commit/9b6d51db5193915123b23a131eb7c2cc7098b2db)), closes [#86](https://github.com/jejjohnson/finitevolX/issues/86)
* **elliptic:** make spectral solvers vmap-compatible ([248ac5d](https://github.com/jejjohnson/finitevolX/commit/248ac5da988515b716c0411dae068f3e91ed9984))
* **solvers:** add tridiagonal (TDMA) solver ([6d9e485](https://github.com/jejjohnson/finitevolX/commit/6d9e485e309761f67f76916e39f515132d9c121b))
* **solvers:** add tridiagonal (TDMA) solver wrapping lineax ([1802ea6](https://github.com/jejjohnson/finitevolX/commit/1802ea6d1fab7f8828ed7c720b0bd9519caf48eb)), closes [#85](https://github.com/jejjohnson/finitevolX/issues/85)


### Bug Fixes

* **elliptic:** guard only (0,0) null mode, not full spectrum ([3c9f24e](https://github.com/jejjohnson/finitevolX/commit/3c9f24eec64769f329ca68dd0c174744d90e8901))
* **tridiagonal:** document lineax throw=True error handling for singular systems ([60a0814](https://github.com/jejjohnson/finitevolX/commit/60a081439be07ae179be464d47a2ba5105bcb0cc))

## [0.0.27](https://github.com/jejjohnson/finitevolX/compare/v0.0.26...v0.0.27) (2026-03-12)


### Features

* add time integration module (Epic [#111](https://github.com/jejjohnson/finitevolX/issues/111)) ([443d571](https://github.com/jejjohnson/finitevolX/commit/443d57119294be8dac0ecb047a3cbdc1aa7e656a))
* **timestepping:** add diffrax Butcher tableau solvers and solve_ocean_pde ([95657c6](https://github.com/jejjohnson/finitevolX/commit/95657c66c542cea2815ad2a36bcbd58cd60907c2))
* **timestepping:** add pure functional time integrators ([c9c7c03](https://github.com/jejjohnson/finitevolX/commit/c9c7c037a7b2edc2f20bf64e642a46b102533c78))
* **timestepping:** export all time integration symbols from finitevolx ([1ae8f0c](https://github.com/jejjohnson/finitevolX/commit/1ae8f0c45d52b7cccf4a3d95fc510973c137e735))


### Bug Fixes

* address PR review comments for time integration module ([735a3b8](https://github.com/jejjohnson/finitevolX/commit/735a3b82c0aa05e694ee3e3516e34c1df28b6de3))
* **timestepping:** resolve ty type checker errors in diffrax solvers ([d9c0c2d](https://github.com/jejjohnson/finitevolX/commit/d9c0c2d23e7c7f1a05b736fbfacb143f1685986c))

## [0.0.26](https://github.com/jejjohnson/finitevolX/compare/v0.0.25...v0.0.26) (2026-03-12)


### Features

* **scripts:** add eta output and spinup phase to QG model ([ada93ea](https://github.com/jejjohnson/finitevolX/commit/ada93ea3b6bf18b8ba7b28c687a598305b63ef7a))


### Bug Fixes

* correct double-gyre script physics and BCs ([0432d2a](https://github.com/jejjohnson/finitevolX/commit/0432d2a312f04cdc9c80a78a68a9bf8b5d1e7ce5))
* **scripts:** address PR review comments ([2495633](https://github.com/jejjohnson/finitevolX/commit/24956332da6a0ca00f4d0aad73f98e8206d27f0e))
* **scripts:** correct wind sign and switch SWM to closed-basin BCs ([d526fc2](https://github.com/jejjohnson/finitevolX/commit/d526fc2316a25c6908f2ade14ff91ba85697a945))

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

* add unified `Mask2D`; remove legacy `MaskGrid`/`FaceMask`/`NodeMask`/`CenterMask` ([d11b995](https://github.com/jejjohnson/finitevolX/commit/d11b9951eaced1205c9a1de6fe6a8d4c5b515ac4))


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
