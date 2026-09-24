# Diagnostic Quantities

Pointwise and domain-integrated diagnostic quantities on Arakawa C-grids.

Each diagnostic comes in two forms:

- **Class operators** (Layer 3) take the land/ocean mask at construction and
  zero the dry cells of their output stagger — use these in masked domains.
- **Functional helpers** (Layer 2) are mask-free; apply the mask yourself at
  the call site if you use them directly.

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

::: finitevolx.Energetics2D

::: finitevolx.Strain2D

::: finitevolx.QGPotentialVorticity2D

Enstrophy and potential enstrophy are methods of
[`Vorticity2D`](vorticity.md) (`enstrophy`, `potential_enstrophy`), next to
the vorticity they are built from.

## Functional helpers

### Kinetic Energy

::: finitevolx.kinetic_energy

### Bernoulli Potential

::: finitevolx.bernoulli_potential

### Relative Vorticity

::: finitevolx.relative_vorticity_cgrid

### Potential Vorticity

::: finitevolx.potential_vorticity

### QG Potential Vorticity

::: finitevolx.qg_potential_vorticity

### Stretching Term (Multi-Layer QG)

::: finitevolx.stretching_term

### Strain

::: finitevolx.shear_strain

::: finitevolx.tensor_strain

::: finitevolx.strain_magnitude_squared

### Okubo-Weiss Parameter

::: finitevolx.okubo_weiss

### Enstrophy

::: finitevolx.enstrophy

::: finitevolx.potential_enstrophy

### Available Potential Energy

::: finitevolx.available_potential_energy

### Domain-Integrated Quantities

::: finitevolx.total_energy

::: finitevolx.total_enstrophy

### Vertical Velocity

::: finitevolx.vertical_velocity
