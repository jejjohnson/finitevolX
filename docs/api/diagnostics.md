# Diagnostic Quantities

Pointwise and domain-integrated diagnostic quantities on Arakawa C-grids.

Each diagnostic in the table below comes in two forms:

- **Class operators** (Layer 3) take the land/ocean mask at construction and
  zero the dry cells of their output stagger — use these in masked domains.
- **Functional helpers** (Layer 2) are mask-free; apply the mask yourself at
  the call site if you use them directly.

The domain-integrated `total_energy` / `total_enstrophy` and
`vertical_velocity` have **no** class form: pass the reductions fields that
are already masked (e.g. from the class operators), and give
`vertical_velocity` a `Mask3D` as its per-call `mask=` argument.

See [Masks → Operator API](../masks.md#operator-api-how-operators-consume-masks)
for the stagger → mask mapping.

| Diagnostic | Class method | Functional helper | Output |
|---|---|---|---|
| Kinetic energy | `Energetics2D.kinetic_energy` | `kinetic_energy` | T |
| Bernoulli potential | `Energetics2D.bernoulli_potential` | `bernoulli_potential` | T |
| Available potential energy | `Energetics2D.available_potential_energy` | `available_potential_energy` | T |
| Relative vorticity | `Vorticity2D.relative_vorticity` | `relative_vorticity_cgrid` | X |
| Shallow-water PV | `Vorticity2D.potential_vorticity` | `sw_potential_vorticity`¹ | X |
| Enstrophy | `Vorticity2D.enstrophy` | `enstrophy` | X |
| Potential enstrophy | `Vorticity2D.potential_enstrophy` | `potential_enstrophy` | X |
| Shear strain | `Strain2D.shear` | `shear_strain` | X |
| Tensor strain | `Strain2D.tensor` | `tensor_strain` | T |
| Strain magnitude² | `Strain2D.magnitude_squared` | `strain_magnitude_squared` | T |
| Okubo–Weiss | `Strain2D.okubo_weiss` | `okubo_weiss` | T |
| QG PV (one layer) | `QGPotentialVorticity2D.__call__` | `qg_potential_vorticity` | T |
| Stretching term | `QGPotentialVorticity2D.stretching` | `stretching_term` | T |
| QG PV (multilayer) | `QGPotentialVorticity2D.multilayer` | `potential_vorticity_multilayer` | T |

¹ Not identical where the thickness at a corner is zero: `sw_potential_vorticity`
returns `0` there, while `Vorticity2D.potential_vorticity` returns a `NaN`
sentinel at wet corners (dry corners are exactly `0` under a mask), so a
degenerate layer is flagged rather than hidden.

The functional `strain_magnitude_squared` / `okubo_weiss` / `enstrophy` /
`potential_enstrophy` are pointwise: their inputs must already share a grid
point.  The class methods do the staggering for you — `Strain2D` averages the
X-point shear and vorticity to T-points, `Vorticity2D.potential_enstrophy`
averages `h` to X-points.

## Class operators

::: gridax_ops.Energetics2D

::: gridax_ops.Strain2D

::: gridax_ops.QGPotentialVorticity2D

Enstrophy and potential enstrophy are methods of
[`Vorticity2D`](vorticity.md) (`enstrophy`, `potential_enstrophy`), next to
the vorticity they are built from.

## Functional helpers

### Kinetic Energy

::: gridax_ops.kinetic_energy

### Bernoulli Potential

::: gridax_ops.bernoulli_potential

### Relative Vorticity

::: gridax_ops.relative_vorticity_cgrid

### Potential Vorticity

`sw_potential_vorticity` is the functional counterpart of
`Vorticity2D.potential_vorticity` (C-grid vorticity and interpolation
included); `potential_vorticity` is the pointwise `(omega + f) / h` for
inputs that already share a grid point.

::: gridax_ops.sw_potential_vorticity

::: gridax_ops.potential_vorticity

### QG Potential Vorticity

::: gridax_ops.qg_potential_vorticity

### Stretching Term (Multi-Layer QG)

::: gridax_ops.stretching_term

### Strain

::: gridax_ops.shear_strain

::: gridax_ops.tensor_strain

::: gridax_ops.strain_magnitude_squared

### Okubo-Weiss Parameter

::: gridax_ops.okubo_weiss

### Enstrophy

::: gridax_ops.enstrophy

::: gridax_ops.potential_enstrophy

### Available Potential Energy

::: gridax_ops.available_potential_energy

### Domain-Integrated Quantities

::: gridax_ops.total_energy

::: gridax_ops.total_enstrophy

### Vertical Velocity

::: gridax_ops.vertical_velocity
