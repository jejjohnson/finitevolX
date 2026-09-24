# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Known Boundary Values: Theory, Practice, and the finitevolX API
#
# This tutorial builds inhomogeneous Dirichlet boundary conditions for
# elliptic solves **from scratch** — continuous mathematics, the discrete
# linear system, pseudocode, a hand-written NumPy implementation — and then
# shows how the same algorithm is packaged in finitevolX.  Every claim made
# along the way is checked numerically in the cell that follows it.
#
# **The problem.**  Given a right-hand side $f$ and prescribed boundary data
# $g$, find $\psi$ with
#
# $$
# (\nabla^2 - \lambda)\,\psi = f \quad \text{in } \Omega,
# \qquad
# \psi = g \quad \text{on } \partial\Omega,
# \qquad \lambda \ge 0 .
# $$
#
# This one equation covers streamfunction inversion ($\lambda = 0$,
# $f = \zeta$), pressure recovery, and quasi-geostrophic PV inversion
# ($\lambda = 1/L_d^2$, $f = q$).  "Inhomogeneous" means $g \ne 0$: the
# boundary carries data — SSH from a parent model, a tide gauge, a
# reanalysis — rather than a wall at rest.
#
# **Roadmap.**
#
# | Part | Question it answers |
# |------|---------------------|
# | 1. Continuous theory | Why can a non-zero boundary be traded for a modified right-hand side? |
# | 2. Discrete setting | Which grid cells are unknowns, which are data, and what is the matrix? |
# | 3. From scratch | The whole algorithm in ~30 lines of NumPy, checked against a dense solve. |
# | 4. Why the *inner* ring | The one convention that makes the stencil and the mask agree. |
# | 5. Solvers | CG, spectral, capacitance, multigrid — what each needs. |
# | 6. The finitevolX API | `known_values`, `known_mask`, `SolveDomain`, `KnownValueLifting`, `bc=`. |
# | 7. Verification | Second-order convergence on a manufactured solution. |
# | 8. In practice | Observations, multi-layer PV, JIT loops, gradients w.r.t. the data. |
# | 9. Pitfalls | A checklist of the ways this goes wrong. |

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

import finitevolx as fvx

IMG_DIR = (
    Path(__file__).resolve().parent.parent / "images" / "inhomogeneous_bcs_tutorial"
)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ---
#
# ## Part 1 — Continuous theory: the lifting trick
#
# Write the operator as $\mathcal{L}_\lambda = \nabla^2 - \lambda$.  It is
# **linear**, so a solution can be split into pieces that each carry part of
# the data.  Pick *any* function $\psi_{\text{lift}}$ that matches the
# boundary data,
#
# $$
# \psi_{\text{lift}} = g \quad \text{on } \partial\Omega ,
# $$
#
# and write $\psi = \psi_{\text{lift}} + \psi_{\text{hom}}$.  Substituting,
#
# $$
# \mathcal{L}_\lambda \psi_{\text{hom}}
#   = f - \mathcal{L}_\lambda \psi_{\text{lift}}
#   \quad \text{in } \Omega,
# \qquad
# \psi_{\text{hom}} = g - g = 0 \quad \text{on } \partial\Omega .
# $$
#
# So $\psi_{\text{hom}}$ solves the **homogeneous** problem — zero boundary
# values — with a **corrected** right-hand side
# $f' = f - \mathcal{L}_\lambda \psi_{\text{lift}}$.  Any solver that
# handles $\psi = 0$ on the boundary can now handle $\psi = g$.
#
# Three facts make this safe:
#
# 1. **Uniqueness.**  For $\lambda \ge 0$ the homogeneous problem has only
#    the trivial solution (multiply $\mathcal{L}_\lambda u = 0$ by $u$,
#    integrate by parts: $-\int |\nabla u|^2 - \lambda \int u^2 = 0$, so
#    $u = 0$).  Hence $\psi$ does not depend on *which* lift you pick — only
#    on its boundary values.
# 2. **Linearity in the data.**  $\psi$ depends linearly on $(f, g)$, so
#    derivatives with respect to boundary data exist and are cheap (Part 8).
# 3. **The lift can be crude.**  In the continuous setting the lift must be
#    smooth enough to differentiate.  On a grid we will use the crudest
#    possible lift — $g$ on the boundary cells, **zero everywhere else** —
#    and the discrete algebra (Part 2) shows that this is still exact.

# %% [markdown]
# ---
#
# ## Part 2 — The discrete setting
#
# ### 2.1 Grid and stencil
#
# Fields live at cell centres of an $N_y \times N_x$ array, indexed
# `psi[j, i]` (row $j$ is $y$, column $i$ is $x$).  The discrete operator is
# the 5-point stencil
#
# ```
# (A psi)[j, i] = (psi[j, i+1] - 2 psi[j, i] + psi[j, i-1]) / dx^2
#               + (psi[j+1, i] - 2 psi[j, i] + psi[j-1, i]) / dy^2
#               - lambda * psi[j, i]
# ```
#
# which is second-order accurate for $\nabla^2 - \lambda$.  In finitevolX it
# is `fvx.masked_laplacian(psi, mask, dx, dy, lambda_)`, which first zeroes
# `psi` outside `mask` and then applies the stencil.
#
# ### 2.2 Index sets
#
# A binary `mask` (1 = wet, 0 = dry) splits the grid.  Write $\mathcal{N}(c)$
# for the four stencil neighbours of cell $c$.
#
# | Symbol | Name | Definition | finitevolX |
# |--------|------|------------|------------|
# | $W$ | wet cells | `mask == 1` | `domain.wet_mask` |
# | $D$ | dry cells | `mask == 0` | `~domain.wet_mask` |
# | $R$ | inner boundary ring | $\{c \in W : \mathcal{N}(c) \cap D \ne \emptyset\}$ | `fvx.boundary_ring(mask)` |
# | $O$ | observed cells | user-supplied, restricted to $W$ | `known_mask & wet` |
# | $K$ | known cells | $R \cup O$ | `domain.all_known` |
# | $E$ | effective (solve) cells | $W \setminus K$ | `domain.effective_mask` |
#
# The data $g$ is prescribed on $K$; the unknowns are $\psi$ on $E$.  The
# property that makes everything work:
#
# > **Every stencil neighbour of an effective cell is wet:**
# > $c \in E \Rightarrow \mathcal{N}(c) \subset W = E \cup K$.
#
# (If a neighbour were dry, $c$ would be in the ring $R$, hence in $K$,
# not $E$.)  So the equations at $E$ only ever touch unknowns in $E$ and data
# in $K$ — never a dry cell whose value would be meaningless.
#
# Here is a small basin with an island and two observation points.

# %%
Ny, Nx = 14, 18
mask_small = np.zeros((Ny, Nx))
# mask[j, i] = 1 for 1 <= j <= Ny-2, 1 <= i <= Nx-2   (dry ghost ring at 0 / N-1)
mask_small[1:-1, 1:-1] = 1.0
# mask[j, i] = 0 for 6 <= j <= 7, 7 <= i <= 9           (island)
mask_small[6:8, 7:10] = 0.0
obs_small = np.zeros((Ny, Nx), dtype=bool)
obs_small[4, 4] = obs_small[10, 13] = True  # two "tide gauges"

domain_small = fvx.SolveDomain(jnp.array(mask_small), known_mask=jnp.array(obs_small))
W = np.asarray(domain_small.wet_mask)
R = np.asarray(domain_small.boundary_ring)
K = np.asarray(domain_small.all_known)
E = np.asarray(domain_small.effective_mask)
O = K & ~R

# Partition check: D, R, O, E are disjoint and cover the grid.
labels = np.select([~W, R, O, E], [0, 1, 2, 3])
assert np.all((~W).astype(int) + R + O + E == 1)
print(f"|W| = {W.sum()},  |R| = {R.sum()},  |O| = {O.sum()},  |E| = {E.sum()}")

# The key property: every neighbour of an E cell is wet.
# Pad with "dry" so that padded_dry[j+1, i+1] = dry[j, i]; then the four
# shifted slices below have the interior's shape and read one neighbour each:
#   nbr_dry[j, i] = dry[j+1, i] | dry[j-1, i] | dry[j, i+1] | dry[j, i-1]
padded_dry = np.pad(~W, 1, constant_values=True)
nbr_dry = (
    padded_dry[2:, 1:-1]  # dry[j+1, i]  (north)
    | padded_dry[:-2, 1:-1]  # dry[j-1, i]  (south)
    | padded_dry[1:-1, 2:]  # dry[j, i+1]  (east)
    | padded_dry[1:-1, :-2]  # dry[j, i-1]  (west)
)
assert not np.any(nbr_dry & E)
print("Every stencil neighbour of an effective cell is wet: OK")

# %%
fig, ax = plt.subplots(figsize=(7, 5.5))
cmap = ListedColormap(["#6b6b6b", "#d95f02", "#7570b3", "#a6cee3"])
ax.imshow(labels, origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5)
for j in range(Ny):
    for i in range(Nx):
        ax.text(i, j, "DROE"[labels[j, i]], ha="center", va="center", fontsize=7)
ax.set_xticks(range(Nx))
ax.set_yticks(range(Ny))
ax.set_xlabel("i (x)")
ax.set_ylabel("j (y)")
ax.set_title("Index sets: D dry, R inner ring, O observed, E effective")
plt.tight_layout()
fig.savefig(IMG_DIR / "index_sets.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Index sets on a basin with an island](../../images/inhomogeneous_bcs_tutorial/index_sets.png)
#
# The ring $R$ wraps **both** the outer walls and the island coast; the
# observations $O$ sit inside the wet domain.  Only the light-blue cells $E$
# are solved for.

# %% [markdown]
# ### 2.3 The linear system and its block structure
#
# Order the wet cells as $[E, K]$ and write the stencil restricted to wet
# cells as a matrix $A$.  The equations at the effective cells read
#
# $$
# \begin{pmatrix} A_{EE} & A_{EK} \end{pmatrix}
# \begin{pmatrix} \psi_E \\ \psi_K \end{pmatrix} = f_E .
# $$
#
# At the known cells we do **not** impose the PDE; we impose the data,
# $\psi_K = g_K$.  Moving the known part to the right-hand side gives the
# system that is actually solved:
#
# $$
# \boxed{\;A_{EE}\,\psi_E = f_E - A_{EK}\, g_K\;}
# $$
#
# This is block elimination of known unknowns — nothing is approximated.
# The correction $A_{EK} g_K$ is non-zero only at effective cells that
# touch a known cell.  For example, at an effective cell whose **south**
# neighbour is on the ring:
#
# ```
#            psi[j+1, i]                    (E, unknown)
#                 |
# psi[j, i-1] -- psi[j, i] -- psi[j, i+1]   (E, unknowns)
#                 |
#            g[j-1, i]                      (K, data -> moved to RHS)
#
#   rhs_corrected[j, i] = f[j, i] - g[j-1, i] / dy^2
# ```
#
# Two structural facts matter for choosing a solver:
#
# * $A$ is **symmetric** ($a_{cd} = a_{dc}$: both are $1/h^2$ for
#   neighbours).
# * $A_{EE}$ is **negative definite** for $\lambda \ge 0$: it is a
#   discrete Laplacian with Dirichlet points on its boundary, hence
#   irreducibly diagonally dominant with negative diagonal.  So
#   $-A_{EE}$ is symmetric positive definite — exactly what conjugate
#   gradients needs — and it is invertible even for $\lambda = 0$.
#
# Let us build $A$ densely on the small grid and look at it.

# %%
dx_s, dy_s, lam_s = 0.3, 0.25, 2.0


def dense_operator(mask, dx, dy, lam):
    """Dense 5-point (A - lambda) restricted to wet cells, with an index map.

    A[k, k]    = -2/dx^2 - 2/dy^2 - lambda
    A[k, nbr]  = 1/dx^2 (east/west),  1/dy^2 (north/south)  for WET neighbours
    """
    wet = np.argwhere(mask > 0.5)
    index = {tuple(c): k for k, c in enumerate(wet)}
    A = np.zeros((len(wet), len(wet)))
    for k, (j, i) in enumerate(wet):
        # A[(j,i), (j,i)] = -2/dx^2 - 2/dy^2 - lambda
        A[k, k] = -2.0 / dx**2 - 2.0 / dy**2 - lam
        # Off-diagonal entries, one per WET neighbour:
        #   A[(j,i), (j,i+1)] = 1/dx^2   (east)
        #   A[(j,i), (j,i-1)] = 1/dx^2   (west)
        #   A[(j,i), (j+1,i)] = 1/dy^2   (north)
        #   A[(j,i), (j-1,i)] = 1/dy^2   (south)
        for dj, di, h in ((0, 1, dx), (0, -1, dx), (1, 0, dy), (-1, 0, dy)):
            nbr = (j + dj, i + di)
            if nbr in index:
                A[k, index[nbr]] += 1.0 / h**2
    return A, wet


A, wet_cells = dense_operator(mask_small, dx_s, dy_s, lam_s)
is_known = np.array([K[j, i] for j, i in wet_cells])
order = np.concatenate([np.where(~is_known)[0], np.where(is_known)[0]])
A_ord = A[np.ix_(order, order)]
nE = int((~is_known).sum())

A_EE = A[np.ix_(~is_known, ~is_known)]
print(f"A symmetric:            {np.allclose(A, A.T)}")
print(f"A_EE eigenvalues < 0:   {np.all(np.linalg.eigvalsh(A_EE) < 0)}")
print(f"cond(A_EE) = {np.linalg.cond(A_EE):.1f}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.spy(A_ord, markersize=1.5)
ax.axhline(nE - 0.5, color="C3")
ax.axvline(nE - 0.5, color="C3")
nK = len(order) - nE
for label, (row, col) in {
    "$A_{EE}$": (nE / 2, nE / 2),
    "$A_{EK}$": (nE / 2, nE + nK / 2),
    "$A_{KE}$": (nE + nK / 2, nE / 2),
    "$A_{KK}$": (nE + nK / 2, nE + nK / 2),
}.items():
    ax.text(col, row, label, color="C3", fontsize=14, ha="center", va="center",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
ax.set_title("Sparsity of A, wet cells ordered [E | K]", pad=12)
plt.tight_layout()
fig.savefig(IMG_DIR / "block_structure.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Block structure of the discrete operator](../../images/inhomogeneous_bcs_tutorial/block_structure.png)
#
# The top-left block is $A_{EE}$ (what the solver inverts); the top-right
# block $A_{EK}$ is the coupling that becomes the right-hand-side
# correction.  The bottom rows are never used: at known cells the
# equation is replaced by $\psi = g$.

# %% [markdown]
# ---
#
# ## Part 3 — From scratch
#
# ### 3.1 Pseudocode
#
# The algorithm has an **offline** half (depends only on the mask; done
# once) and an **online** half (depends on $f$ and $g$; done every solve,
# e.g. every time step).
#
# ```
# OFFLINE — once per mask
#   W[j,i] <- mask[j,i] > 0.5                                   # wet cells
#   D[j,i] <- not W[j,i]                                        # dry cells
#   R[j,i] <- W[j,i] and (D[j+1,i] or D[j-1,i]
#                          or D[j,i+1] or D[j,i-1])             # inner ring
#   K[j,i] <- R[j,i] or (known_mask[j,i] and W[j,i])            # known cells
#   E[j,i] <- W[j,i] and not K[j,i]                             # unknowns
#   S      <- solver for  A_EE x = b                            # CG / spectral / ...
#
# ONLINE — every solve, given f and g
#   lift[j,i] <- g[j,i] if K[j,i] else 0                        # the crude lift
#   l[j,i]    <- lift[j,i] * W[j,i]                             # WET mask
#   (A_W l)[j,i] = (l[j,i+1] - 2 l[j,i] + l[j,i-1]) / dx^2
#                + (l[j+1,i] - 2 l[j,i] + l[j-1,i]) / dy^2 - lambda l[j,i]
#   b[j,i]    <- (f[j,i] - (A_W l)[j,i]) if E[j,i] else 0      # corrected RHS
#   x         <- S(b)                                           # x[j,i] = 0 off E
#   psi[j,i]  <- lift[j,i] + x[j,i]
# ```
#
# The one subtle line is `A_W lift`: the stencil must be applied with the
# **wet** mask (so it reads the lift on $K$), whereas the solver works on
# $E$.  Applying the stencil to the zero-off-$K$ lift and keeping only
# rows in $E$ produces exactly $A_{EK}\, g_K$, because the lift vanishes on
# $E$.  Part 4 shows what happens if you get the mask wrong.
#
# ### 3.2 Implementation in plain NumPy
#
# Nothing below uses finitevolX's lifting code — only a stencil and a
# dense solve.

# %%
def stencil(psi, mask, dx, dy, lam):
    """(A - lambda) applied to psi*mask, output zeroed outside mask (5-point)."""
    # Zero-pad by one cell:  p[j+1, i+1] = (psi * mask)[j, i]
    p = np.pad(psi * mask, 1)
    c = p[1:-1, 1:-1]  # psi[j, i]
    e = p[1:-1, 2:]  # psi[j, i+1]
    w = p[1:-1, :-2]  # psi[j, i-1]
    n = p[2:, 1:-1]  # psi[j+1, i]
    s = p[:-2, 1:-1]  # psi[j-1, i]
    # (A psi)[j, i] = (psi[j, i+1] - 2 psi[j, i] + psi[j, i-1]) / dx^2
    #               + (psi[j+1, i] - 2 psi[j, i] + psi[j-1, i]) / dy^2
    #               - lambda * psi[j, i]
    lap = (e - 2 * c + w) / dx**2 + (n - 2 * c + s) / dy**2
    return (lap - lam * c) * mask


def solve_known_values_numpy(f, g, mask, known_mask, dx, dy, lam):
    """Inhomogeneous Dirichlet solve by lifting — the pseudocode, verbatim."""
    # OFFLINE
    wet = mask > 0.5
    # pd[j+1, i+1] = dry[j, i]  (dry padding outside the array)
    pd = np.pad(~wet, 1, constant_values=True)
    # ring[j, i] = wet[j, i] & (dry[j+1, i] | dry[j-1, i] | dry[j, i+1] | dry[j, i-1])
    ring = wet & (pd[2:, 1:-1] | pd[:-2, 1:-1] | pd[1:-1, 2:] | pd[1:-1, :-2])
    known = ring | (known_mask & wet)
    eff = wet & ~known
    A_EE, eff_cells = dense_operator(eff.astype(float), dx, dy, lam)
    # ONLINE
    lift = np.where(known, g, 0.0)
    b = (f - stencil(lift, wet.astype(float), dx, dy, lam)) * eff
    x = np.zeros_like(f)
    x[tuple(eff_cells.T)] = np.linalg.solve(A_EE, b[tuple(eff_cells.T)])
    return lift + x


rng = np.random.default_rng(0)
f_small = rng.standard_normal((Ny, Nx)) * mask_small
g_small = rng.standard_normal((Ny, Nx))

psi_scratch = solve_known_values_numpy(
    f_small, g_small, mask_small, obs_small, dx_s, dy_s, lam_s
)

# Reference 1: block elimination on the full dense matrix (Part 2.3).
f_w = np.array([f_small[j, i] for j, i in wet_cells])
g_w = np.array([g_small[j, i] for j, i in wet_cells])
psi_E = np.linalg.solve(
    A[np.ix_(~is_known, ~is_known)],
    f_w[~is_known] - A[np.ix_(~is_known, is_known)] @ g_w[is_known],
)
psi_block = np.zeros((Ny, Nx))
psi_block[tuple(wet_cells[is_known].T)] = g_w[is_known]
psi_block[tuple(wet_cells[~is_known].T)] = psi_E
print(f"scratch lifting vs block elimination: {np.max(np.abs(psi_scratch - psi_block)):.2e}")

# The defining properties of the answer:
res = f_small - stencil(psi_scratch, mask_small, dx_s, dy_s, lam_s)
print(f"max |psi - g| on K:         {np.max(np.abs((psi_scratch - g_small)[K])):.2e}")
print(f"max |PDE residual| on E:    {np.max(np.abs(res[E])):.2e}")
print(f"max |PDE residual| on K:    {np.max(np.abs(res[K])):.2e}   (not zero, by design)")

# %% [markdown]
# Read those three numbers carefully:
#
# * $\psi = g$ **exactly** on the known cells;
# * the PDE holds to round-off on the effective cells;
# * the PDE does **not** hold on the known cells — there the equation was
#   *replaced* by the data.  This is the defining property of a Dirichlet
#   point, not an error.

# %% [markdown]
# ---
#
# ## Part 4 — Why the *inner* ring?
#
# A natural first attempt (and the one many codes use) places the boundary
# values on the **dry** cells just outside the ocean — the "outer ring",
# like ghost cells.  With a masked stencil this silently fails:
# `masked_laplacian` multiplies its input by the mask *before* applying the
# stencil, so values stored on dry cells are erased and the correction is
# identically zero.  The solve then returns the **homogeneous** answer.
#
# The inner-ring convention avoids this by keeping the data on **wet**
# cells, which is why two masks appear in the algorithm:
#
# | Step | Mask | Why |
# |------|------|-----|
# | correction `A_W lift` | wet $W$ | the stencil must *see* the lift on $K \subset W$ |
# | solve `S(b)` | effective $E$ | known cells are data, not unknowns |
#
# Using $E$ for the correction would erase the lift (zero correction);
# using $W$ for the solve would treat known cells as unknowns and overwrite
# the data.  Both mistakes are demonstrated below.

# %%
f_j, g_j, m_j = jnp.array(f_small), jnp.array(g_small), jnp.array(mask_small)
# pm[j+1, i+1] = mask[j, i]  (zero padding)
pm = jnp.pad(m_j, 1)
# outer ring: dry cells with a wet neighbour,
#   dry_ring[j, i] = dry[j, i] & (wet[j+1, i] | wet[j-1, i] | wet[j, i+1] | wet[j, i-1])
dry_ring = np.asarray(
    pm[2:, 1:-1]  # mask[j+1, i]
    + pm[:-2, 1:-1]  # mask[j-1, i]
    + pm[1:-1, 2:]  # mask[j, i+1]
    + pm[1:-1, :-2]  # mask[j, i-1]
    > 0
) & ~W
outer_lift = jnp.where(dry_ring, g_j, 0.0)
print(f"outer-ring lift: max |correction| = "
      f"{float(jnp.max(jnp.abs(fvx.masked_laplacian(outer_lift, m_j, dx_s, dy_s, lam_s)))):.1e}"
      "   <- the data never reaches the RHS")

eff_j = domain_small.effective_mask.astype(float)
wrong_corr = fvx.masked_laplacian(jnp.where(K, g_j, 0.0), eff_j, dx_s, dy_s, lam_s)
print(f"correction with E-mask:  max |correction| = {float(jnp.max(jnp.abs(wrong_corr))):.1e}"
      "   <- lift erased again")
right_corr = fvx.masked_laplacian(jnp.where(K, g_j, 0.0), m_j, dx_s, dy_s, lam_s) * eff_j
print(f"correction with W-mask:  max |correction| = {float(jnp.max(jnp.abs(right_corr))):.1e}"
      "   <- correct")

# %% [markdown]
# ---
#
# ## Part 5 — Solvers for $A_{EE}\,x = b$
#
# The lifting is solver-agnostic: it only needs *some* way to apply
# $A_{EE}^{-1}$.  What each solver needs to be told:
#
# | Solver | Works on | Build it with | Notes |
# |--------|----------|---------------|-------|
# | Conjugate gradient | any mask | operator `masked_laplacian(., E)` | $-A_{EE}$ is SPD; iterations $\sim\sqrt{\kappa}$ |
# | CG + multigrid preconditioner | any mask | `build_multigrid_solver(E)` as preconditioner | near grid-independent iterations |
# | Spectral (DST) | rectangle only | nothing | $O(N \log N)$, direct |
# | Capacitance matrix | any mask, fixed | `build_capacitance_solver(W)` — **wet** mask | direct after an $O(N_b)$ setup |
#
# **Spectral.**  If $W$ is the rectangle `[1:-1, 1:-1]`
# ($1 \le j \le N_y-2$, $1 \le i \le N_x-2$), its ring is the frame of that
# rectangle (rows/columns $1$ and $N-2$) and $E$ = `[2:-2, 2:-2]`
# ($2 \le j \le N_y-3$, $2 \le i \le N_x-3$) is again a rectangle.
# $A_{EE}$ is then the Dirichlet Laplacian on a rectangle, diagonalised by
# the type-I discrete sine transform $S$:
#
# $$
# A_{EE} = S^{-1} \Lambda S,
# \qquad
# \Lambda_{kl} = -\frac{4}{\Delta x^2}\sin^2\!\frac{\pi k}{2(n_x+1)}
#                -\frac{4}{\Delta y^2}\sin^2\!\frac{\pi l}{2(n_y+1)} - \lambda ,
# $$
#
# so $x = S^{-1}\Lambda^{-1} S\, b$ — exactly, in two FFT-like transforms.
#
# **Capacitance.**  A capacitance solver built on a mask $M$ enforces
# $\psi = 0$ on $M$'s own inner ring.  Built on $W$, its unknowns are
# $W \setminus R = E$ — precisely the lifted system (when there are no
# observations).  Built on $E$ it would zero $E$'s ring too, removing a
# second layer of cells.  With observations there is no mask $M$ whose
# "interior minus inner ring" equals $E$, so capacitance cannot take a
# `known_mask`.
#
# **Multigrid.**  finitevolX's `MultigridSolver` discretises the operator
# with face coefficients (the Dirichlet condition sits on cell faces), which
# is not the same matrix as `masked_laplacian`.  It is an excellent
# *preconditioner* for $A_{EE}$ but a standalone multigrid solve does not
# reproduce the lifted problem.
#
# The cell below solves one problem four ways on a rectangular basin, where
# all four apply.

# %%
Nb = 34
dxb = dyb = 1.0 / (Nb - 3)
lam_b = 4.0
# mask[j, i] = 1 for 1 <= j, i <= Nb-2   (rectangular basin, dry ghost ring)
mask_b = jnp.zeros((Nb, Nb)).at[1:-1, 1:-1].set(1.0)
f_b = jnp.asarray(np.random.default_rng(1).standard_normal((Nb, Nb))) * mask_b
g_b = jnp.asarray(np.random.default_rng(2).standard_normal((Nb, Nb)))
dom_b = fvx.SolveDomain(mask_b)

psi_cg = fvx.streamfunction_from_vorticity(
    f_b, dxb, dyb, lambda_=lam_b, method="cg", mask=mask_b, known_values=g_b
)
psi_spec = fvx.streamfunction_from_vorticity(
    f_b, dxb, dyb, bc="dst", lambda_=lam_b, known_values=g_b
)
cap = fvx.build_capacitance_solver(
    np.asarray(dom_b.wet_mask), dxb, dyb, lambda_=lam_b, base_bc="dst"
)
psi_cap = fvx.streamfunction_from_vorticity(
    f_b, dxb, dyb, lambda_=lam_b, method="capacitance", mask=mask_b,
    capacitance_solver=cap, known_values=g_b,
)
mg = fvx.build_multigrid_solver(
    np.asarray(dom_b.effective_mask, dtype=float), dxb, dyb, lambda_=lam_b
)
psi_mgcg = fvx.streamfunction_from_vorticity(
    f_b, dxb, dyb, lambda_=lam_b, method="cg", mask=mask_b, known_values=g_b,
    preconditioner=fvx.make_multigrid_preconditioner(mg),
)
for name, psi in [("spectral", psi_spec), ("capacitance", psi_cap), ("CG + MG", psi_mgcg)]:
    print(f"{name:12s} vs CG: {float(jnp.max(jnp.abs(psi - psi_cg))):.1e}")

# %% [markdown]
# ---
#
# ## Part 6 — The finitevolX API
#
# ### 6.1 One call: `known_values`
#
# All three convenience wrappers — `streamfunction_from_vorticity`,
# `pressure_from_divergence`, `pv_inversion` — accept `known_values`, a full
# `(Ny, Nx)` field of which only the known cells are read.  The wrapper
# builds $W, R, K, E$, the lift and the correction, calls the chosen
# solver on $E$, and adds the lift back.
#
# ```python
# psi = fvx.streamfunction_from_vorticity(
#     zeta, dx, dy, lambda_=lam,
#     method="cg", mask=mask,
#     known_values=g,            # read on the inner ring (+ known_mask cells)
#     known_mask=obs,            # optional interior observations (CG only)
# )
# ```
#
# ### 6.2 The building blocks: `SolveDomain` + `KnownValueLifting`
#
# The wrapper is a thin composition of two equinox modules that you can use
# directly — for a custom solver, or to build the domain once outside a
# time loop.  They map one-to-one onto the pseudocode:
#
# | Pseudocode | finitevolX |
# |------------|------------|
# | `W, R, K, E` | `domain = fvx.SolveDomain(mask, known_mask)` |
# | `lift`, `b` | `b, lift = fvx.KnownValueLifting(domain, dx, dy, lam).preprocess(f, g)` |
# | `x = S(b)` | *your solver on* `domain.effective_mask` |
# | `psi = lift + x` | `psi = lifter.postprocess(x, lift)` |

# %%
lifter = fvx.KnownValueLifting(domain_small, dx_s, dy_s, lam_s)
b, lift = lifter.preprocess(f_j, g_j)

eff_small = domain_small.effective_mask.astype(float)
A_eff = lambda x: fvx.masked_laplacian(x, eff_small, dx_s, dy_s, lambda_=lam_s)
x, info = fvx.solve_cg(A_eff, b, rtol=1e-12, atol=1e-12)
psi_blocks = lifter.postprocess(x * eff_small, lift)

psi_wrapper = fvx.streamfunction_from_vorticity(
    f_j, dx_s, dy_s, lambda_=lam_s, method="cg", mask=m_j,
    known_values=g_j, known_mask=jnp.array(obs_small),
)
print(f"CG iterations: {info.iterations}")
print(f"building blocks vs NumPy scratch: {float(jnp.max(jnp.abs(psi_blocks - psi_scratch))):.1e}")
print(f"wrapper         vs NumPy scratch: {float(jnp.max(jnp.abs(psi_wrapper - psi_scratch))):.1e}")

# %% [markdown]
# ### 6.3 Boundary conditions as a `BoundaryConditionSet`
#
# When the data is "one value per wall", describe it with the same
# `BoundaryConditionSet` used for time stepping and pass it as `bc=`.
# Non-zero `Dirichlet1D` values are written onto the wall-adjacent wet
# cells (rows/columns 1 and −2; west/east own the corners), the set's
# `mask` is used by the mask-based solvers, and an all-zero set is the
# ordinary homogeneous solve.

# %%
bcset = fvx.BoundaryConditionSet(
    mask=mask_b,
    south=fvx.Dirichlet1D("south", value=0.0),
    north=fvx.Dirichlet1D("north", value=0.2),
    west=fvx.Dirichlet1D("west", value=0.0),
    east=fvx.Dirichlet1D("east", value=-0.1),
)
psi_bc = fvx.streamfunction_from_vorticity(f_b, dxb, dyb, lambda_=lam_b, bc=bcset, method="cg")
print(f"north wall row: {float(psi_bc[-2, 5]):+.3f}   east wall column: {float(psi_bc[5, -2]):+.3f}")

# %% [markdown]
# ### 6.4 What the lift and the correction look like
#
# On a larger basin with an island and a smooth, spatially varying $g$:

# %%
Nd = 64
dxd = dyd = 1.0 / Nd
# mask[j, i] = 1 for 1 <= j, i <= Nd-2, then 0 on the island 26 <= j <= 37, 28 <= i <= 39
mask_d = jnp.zeros((Nd, Nd)).at[1:-1, 1:-1].set(1.0).at[26:38, 28:40].set(0.0)
yy, xx = jnp.meshgrid(jnp.arange(Nd) / Nd, jnp.arange(Nd) / Nd, indexing="ij")
g_d = 0.2 * jnp.sin(2 * jnp.pi * yy) + 0.1 * jnp.cos(3 * jnp.pi * xx)
f_d = -30.0 * jnp.exp(-((xx - 0.3) ** 2 + (yy - 0.7) ** 2) / 0.01) * mask_d

dom_d = fvx.SolveDomain(mask_d)
lifter_d = fvx.KnownValueLifting(dom_d, dxd, dyd, 1.0)
b_d, lift_d = lifter_d.preprocess(f_d, g_d)
psi_d = fvx.streamfunction_from_vorticity(
    f_d, dxd, dyd, lambda_=1.0, method="cg", mask=mask_d, known_values=g_d
)
hom_d = (psi_d - lift_d) * dom_d.effective_mask

wet_d = np.asarray(dom_d.wet_mask)
panels = [
    (np.where(np.asarray(dom_d.all_known), lift_d, np.nan), "lift: g on the ring $R$"),
    (np.where(np.asarray(dom_d.effective_mask), b_d - f_d, np.nan), "correction $-A_{EK}\\,g_K$"),
    (np.where(np.asarray(dom_d.effective_mask), hom_d, np.nan), "$\\psi_{hom}$ (0 on the ring)"),
    (np.where(wet_d, psi_d, np.nan), "$\\psi = $ lift $+ \\psi_{hom}$"),
]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, (field, title) in zip(axes, panels, strict=True):
    vmax = float(np.nanmax(np.abs(field)))  # symmetric: white = 0
    im = ax.imshow(field, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
fig.savefig(IMG_DIR / "lift_decomposition.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Lift, correction, homogeneous part and full solution](../../images/inhomogeneous_bcs_tutorial/lift_decomposition.png)
#
# The correction (second panel) is non-zero only on the single layer of
# effective cells touching the ring — the rows of $A_{EK}$ are that sparse.

# %% [markdown]
# ---
#
# ## Part 7 — Verification: a manufactured solution
#
# To check the *discretisation*, not just the algebra, take a continuous
# solution that does **not** vanish on the boundary,
#
# $$
# \psi^\star(x, y) = \sin(\pi x)\sin(\pi y) + 0.1\,\sin(2\pi y),
# \qquad (x, y) \in [0, 1]^2,
# $$
#
# compute $f = (\nabla^2 - \lambda)\psi^\star$ analytically,
#
# $$
# f = -2\pi^2 \sin(\pi x)\sin(\pi y) - 0.4\pi^2 \sin(2\pi y) - \lambda\,\psi^\star ,
# $$
#
# and place the grid so the ring cell centres lie **on** $\partial\Omega$:
# with $n$ intervals of width $h = 1/n$, cell $k$ of the array sits at
# $x = (k - 1)h$, so the ring (index $1$ and $n+1$) is at $x = 0$ and
# $x = 1$, and the array has $n + 3$ cells including the dry ghost ring.
# Feeding $g = \psi^\star$ on the ring, the error should fall like $h^2$.

# %%
def exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y) + 0.1 * np.sin(2 * np.pi * y)


def exact_rhs(x, y, lam):
    lap = -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) - 0.4 * np.pi**2 * np.sin(
        2 * np.pi * y
    )
    return lap - lam * exact(x, y)


lam_m = 4.0
hs, errs = [], []
for n in (8, 16, 32, 64, 128):
    h = 1.0 / n
    coord = (np.arange(n + 3) - 1) * h  # ring at 0 and 1
    X, Y = np.meshgrid(coord, coord)
    psi_star = exact(X, Y)
    psi_num = fvx.streamfunction_from_vorticity(
        jnp.array(exact_rhs(X, Y, lam_m)), h, h, bc="dst", lambda_=lam_m,
        known_values=jnp.array(psi_star),
    )
    wet = np.zeros_like(psi_star, dtype=bool)
    # wet[j, i] = True for 1 <= j, i <= n+1   (ring at x, y = 0 and 1 included)
    wet[1:-1, 1:-1] = True
    hs.append(h)
    errs.append(np.max(np.abs((np.asarray(psi_num) - psi_star)[wet])))

# order[k] = log2( err[k] / err[k+1] )   (h halves between k and k+1)
rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
for n, e in zip((8, 16, 32, 64, 128), errs, strict=True):
    print(f"n = {n:4d}   max error = {e:.3e}")
print("observed orders:", np.round(rates, 3))

fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.loglog(hs, errs, "o-", label="max error")
ax.loglog(hs, errs[0] * (np.array(hs) / hs[0]) ** 2, "k--", label="$O(h^2)$")
ax.set_xlabel("h")
ax.set_ylabel("max |psi - psi*|")
ax.set_title("Manufactured solution: second-order convergence")
ax.legend()
plt.tight_layout()
fig.savefig(IMG_DIR / "convergence.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Second-order convergence](../../images/inhomogeneous_bcs_tutorial/convergence.png)
#
# The observed order is 2: the known-value machinery adds no error of its
# own — the only error is the 5-point stencil's truncation error.

# %% [markdown]
# ---
#
# ## Part 8 — In practice
#
# ### 8.1 Interior observations (`known_mask`)
#
# Nothing in Part 2 required $K$ to lie on the boundary.  Marking interior
# cells as known turns observations into internal Dirichlet points: the
# solution passes through them exactly and the PDE holds everywhere else.
# (This is a *hard* constraint; a weighted, noisy-observation version would
# instead add a penalty term.)

# %%
obs_d = jnp.zeros((Nd, Nd), dtype=bool)
gauges = [(12, 12), (50, 20), (45, 50), (15, 48)]
values_d = g_d
for j, i in gauges:
    obs_d = obs_d.at[j, i].set(True)
    values_d = values_d.at[j, i].set(0.4)

psi_obs = fvx.streamfunction_from_vorticity(
    f_d, dxd, dyd, lambda_=1.0, method="cg", mask=mask_d,
    known_values=values_d, known_mask=obs_d,
)
print("psi at gauges:", [round(float(psi_obs[j, i]), 6) for j, i in gauges])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
diff = np.where(wet_d, psi_obs - psi_d, np.nan)
for ax, field, title in [
    (axes[0], np.where(wet_d, psi_d, np.nan), "boundary data only"),
    (axes[1], np.where(wet_d, psi_obs, np.nan), "+ 4 gauges pinned to 0.4"),
    (axes[2], diff, "difference"),
]:
    im = ax.imshow(field, origin="lower", cmap="RdBu_r")
    ax.plot([i for _, i in gauges], [j for j, _ in gauges], "k^", ms=7)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
fig.savefig(IMG_DIR / "observations.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Pinning interior observations](../../images/inhomogeneous_bcs_tutorial/observations.png)
#
# ### 8.2 Multi-layer PV inversion
#
# In QG models each vertical mode $k$ has its own $\lambda_k = 1/L_{d,k}^2$.
# A subtle point: $\lambda$ enters $A$ only on the **diagonal**, and the lift
# is **zero on $E$**, so the correction $A_{EK}\, g_K$ is the same for every
# $\lambda$.  What changes from layer to layer is the matrix being inverted,
# $A_{EE}(\lambda_k)$.  finitevolX runs the per-layer lift-and-solve inside
# a `vmap`; `known_values` broadcasts against `pv` — a `(Ny, Nx)` field is
# shared by all layers, `(nl, Ny, Nx)` gives each layer its own data.

# %%
corr = [
    fvx.KnownValueLifting(dom_d, dxd, dyd, lam).preprocess(f_d, g_d)[0] - f_d * dom_d.effective_mask
    for lam in (0.0, 100.0)
]
print(f"correction(lambda=0) - correction(lambda=100): {float(jnp.max(jnp.abs(corr[0] - corr[1]))):.1e}")

lambdas = jnp.array([0.0, 10.0, 100.0, 1000.0])
# pv = 0 isolates the response to the boundary data alone.
pv = jnp.zeros((len(lambdas), Nd, Nd))
psi_layers = fvx.pv_inversion(
    pv, dxd, dyd, lambda_=lambdas, method="cg", mask=mask_d, known_values=g_d
)
ring_d = dom_d.boundary_ring
for k, lam_k in enumerate(lambdas):
    print(
        f"layer {k}: lambda = {float(lam_k):7.1f}   ring exact: "
        f"{bool(jnp.allclose(psi_layers[k][ring_d], g_d[ring_d]))}   "
        f"psi at (16, 16) = {float(psi_layers[k][16, 16]):+.4f}"
    )

# %% [markdown]
# With no interior forcing, $\psi$ is driven by the boundary data alone.
# Larger $\lambda$ **screens** that data: its influence decays over a
# distance $\sim \lambda^{-1/2}$ from the walls, so the interior value at
# $(16, 16)$ — about 15 cells from the nearest wall — shrinks towards zero.
#
# ### 8.3 Time-varying boundary data inside `jax.jit`
#
# In a nested or data-driven model $g$ changes every step.  Everything in
# the online half is plain `jax.numpy`, so the whole solve compiles once
# and is re-run with new data.

# %%
@jax.jit
def solve_step(f, g):
    return fvx.streamfunction_from_vorticity(
        f, dxd, dyd, lambda_=1.0, method="cg", mask=mask_d, known_values=g
    )


for t in range(3):
    g_t = g_d * jnp.cos(0.5 * t)
    psi_t = solve_step(f_d, g_t)
    print(f"t = {t}:  ring matches g(t): {bool(jnp.allclose(psi_t[ring_d], g_t[ring_d]))}")

# %% [markdown]
# ### 8.4 Gradients with respect to the boundary data
#
# The map $g \mapsto \psi$ is linear, and JAX differentiates straight
# through the lifting and the solver.  Take the value at one interior point,
# $J(g) = \psi_{17,17}$ on a 32-interval square with $f = 0$ and
# $\lambda = 0$.  Then $\partial J / \partial g$ is the **discrete harmonic
# measure** of that point: non-negative weights on the ring saying how much
# each boundary value contributes.  For $\lambda = 0$ they must sum to 1
# (a constant $g \equiv c$ gives $\psi \equiv c$), and for $\lambda > 0$
# they sum to less than 1 — screening again.

# %%
Nh = 35
hh = 1.0 / 32
# basin mask: mask[j, i] = 1 for 1 <= j, i <= Nh-2
ring_h = fvx.boundary_ring(jnp.zeros((Nh, Nh)).at[1:-1, 1:-1].set(1.0))


def point_value(g, lam):
    psi = fvx.streamfunction_from_vorticity(
        jnp.zeros((Nh, Nh)), hh, hh, bc="dst", lambda_=lam, known_values=g
    )
    return psi[17, 10]


weights = {}
for lam in (0.0, 50.0):
    wgt = jax.grad(point_value)(jnp.zeros((Nh, Nh)), lam)
    weights[lam] = wgt
    print(
        f"lambda = {lam:5.1f}:  sum of weights on ring = {float(wgt[ring_h].sum()):.6f}, "
        f"min = {float(wgt[ring_h].min()):.2e}, off-ring max |w| = {float(jnp.abs(wgt[~ring_h]).max()):.1e}"
    )

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, lam in zip(axes, (0.0, 50.0), strict=True):
    wgt = np.where(np.asarray(ring_h), np.asarray(weights[lam]), np.nan)
    im = ax.imshow(wgt, origin="lower", cmap="viridis")
    ax.plot(10, 17, "r*", ms=12)
    ax.set_title(f"dpsi[17,10] / dg on the ring, lambda = {lam:g}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
fig.savefig(IMG_DIR / "sensitivity.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Sensitivity of an interior value to the boundary data](../../images/inhomogeneous_bcs_tutorial/sensitivity.png)
#
# Two details worth noticing:
#
# * The weights are **non-negative** (a discrete maximum principle) and
#   concentrate on the walls nearest the point.
# * The four **corner** ring cells get weight exactly zero: their stencil
#   neighbours are ring cells, so no effective cell ever reads them and
#   their data cannot influence the solution.
#
# This is the same object a data-assimilation system needs to push an
# interior misfit back onto the boundary data — obtained here for free
# from `jax.grad`.

# %% [markdown]
# ---
#
# ## Part 9 — Pitfalls checklist
#
# | Symptom | Cause | Fix |
# |---------|-------|-----|
# | Solution ignores the boundary data | Data stored on **dry** cells; `masked_laplacian` erases it | Put data on the inner ring (wet cells); `known_values` does this |
# | Known cells overwritten | Solver built on the wet mask instead of $E$ | CG / multigrid operators on `domain.effective_mask` |
# | Capacitance answer off by $O(1)$ | Capacitance built on `effective_mask` (strips two rings) | Build it on the **wet** mask |
# | Capacitance wrong at $\lambda = 0$ | Default `base_bc="fft"` is singular for Poisson | Use `base_bc="dst"` |
# | Capacitance + observations raises | No mask gives $E$ as "interior minus ring" | Use `method="cg"` |
# | Spectral solve raises with a mask | The spectral path is the rectangular basin only | Use `method="cg"` / `"capacitance"` for other masks |
# | Spectral result differs from plain `bc="dst"` | With `known_values`, unknowns are `[2:-2, 2:-2]` (ring convention) | Expected; plain `bc="dst"` treats the whole array as unknown |
# | Standalone multigrid disagrees | Different (face-based) discretisation | Use multigrid as the CG preconditioner |
# | PDE residual non-zero on the ring | Known cells carry data, not the equation | Expected — check the residual on $E$ only |
# | Sign surprises | The operator is $(\nabla^2 - \lambda)$, $\lambda \ge 0$ | Use positive $\lambda = 1/L_d^2$ for screening |
#
# ## Summary
#
# * A non-zero Dirichlet boundary is the same problem as a zero one with a
#   corrected right-hand side: $A_{EE}\,\psi_E = f_E - A_{EK}\, g_K$.
# * On a grid the lift can be "data on the known cells, zero elsewhere";
#   the elimination is exact, and accuracy is set by the stencil alone
#   (second order).
# * Keep the data on **wet** cells (the inner ring), apply the correction
#   with the **wet** mask, solve on the **effective** mask.
# * finitevolX packages this as `known_values` / `known_mask` on every
#   elliptic wrapper, `bc=BoundaryConditionSet` for per-wall values, and
#   `SolveDomain` + `KnownValueLifting` for your own solvers — all
#   JIT-compatible and differentiable.
