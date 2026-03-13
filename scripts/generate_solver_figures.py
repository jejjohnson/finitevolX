"""Generate solver comparison figures for the documentation.

Produces PNG figures in docs/images/solvers_* showing:
  1. Domain geometries used in the comparison
  2. Solutions from each solver on each geometry
  3. Relative residual norms (how well each solver satisfies its equation)
  4. Timing bar chart

Usage:
    uv run python scripts/generate_solver_figures.py
"""

from __future__ import annotations

from pathlib import Path
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import finitevolx as fvx
from finitevolx._src.solvers.multigrid import _apply_operator

jax.config.update("jax_enable_x64", True)

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# ── Grid setup ──────────────────────────────────────────────────────────
Ny, Nx = 64, 64
dx, dy = 1.0 / Nx, 1.0 / Ny
lambda_ = 4.0

j_idx = jnp.arange(Ny)[:, None]
i_idx = jnp.arange(Nx)[None, :]
rhs = jnp.sin(jnp.pi * (j_idx + 1) / (Ny + 1)) * jnp.sin(
    jnp.pi * (i_idx + 1) / (Nx + 1)
)

Y, X = np.mgrid[:Ny, :Nx]


# ── Helpers ─────────────────────────────────────────────────────────────
def _time_fn(fn, warmup=2, repeats=5):
    """JIT-compile, warm up, then time *repeats* calls."""
    jitted = jax.jit(fn)
    for _ in range(warmup):
        jitted().block_until_ready()
    t0 = time.perf_counter()
    for _ in range(repeats):
        jitted().block_until_ready()
    return (time.perf_counter() - t0) / repeats


def _rel_residual(sol, rhs_loc, matvec, interior_mask=None):
    """||rhs - A(sol)|| / ||rhs||, the relative residual norm.

    If *interior_mask* is given, restrict the norm to interior cells.
    This avoids spurious boundary mismatch when comparing solvers that
    use different boundary treatments.
    """
    residual = rhs_loc - matvec(sol)
    if interior_mask is not None:
        residual = residual * interior_mask
        rhs_loc = rhs_loc * interior_mask
    rhs_norm = float(jnp.linalg.norm(rhs_loc))
    if rhs_norm == 0.0:
        return 0.0
    return float(jnp.linalg.norm(residual)) / rhs_norm


# ======================================================================
# 1. Domain geometries
# ======================================================================
mask_rect = np.ones((Ny, Nx))

mask_basin = np.ones((Ny, Nx))
mask_basin[:4, :] = mask_basin[-4:, :] = 0.0
mask_basin[:, :4] = mask_basin[:, -4:] = 0.0

mask_circle = ((X - Nx / 2) ** 2 + (Y - Ny / 2) ** 2 < (0.4 * Ny) ** 2).astype(float)

mask_notch = np.ones((Ny, Nx))
mask_notch[:8, :] = mask_notch[-8:, :] = 0.0
mask_notch[:, :8] = mask_notch[:, -8:] = 0.0
mask_notch[20:44, :20] = 0.0

geometries = {
    "Rectangle": mask_rect,
    "Basin": mask_basin,
    "Circle": mask_circle,
    "Notch\n(variable coeff)": mask_notch,
}

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, (name, mask) in zip(axes, geometries.items(), strict=True):
    ax.imshow(
        mask, origin="lower", cmap="Blues", interpolation="nearest", vmin=0, vmax=1
    )
    wet = int(mask.sum())
    ax.set_title(f"{name}\n({wet} wet cells)", fontsize=11)
    ax.axis("off")
fig.suptitle("Domain Geometries", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(OUT / "solvers_geometries.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved solvers_geometries.png")


# ======================================================================
# 2. Solve each geometry — collect solution, residual, timing
# ======================================================================
# Each entry: {solver_name: {sol, time_ms, rel_residual, label}}

results: dict[str, dict] = {}

# ── Rectangle ───────────────────────────────────────────────────────────
# Spectral (exact for this geometry)
sol_sp_rect = fvx.solve_helmholtz_dst(rhs, dx, dy, lambda_)
t_sp_rect = _time_fn(lambda: fvx.solve_helmholtz_dst(rhs, dx, dy, lambda_))
# Interior mask: exclude boundary cells where different BC treatments diverge
interior_rect = jnp.zeros((Ny, Nx)).at[1:-1, 1:-1].set(1.0)

# Spectral solves exactly via eigenvalue division — no iterative residual
rr_sp_rect = 0.0  # exact by construction

# CG + spectral preconditioner
mask_rect_jnp = jnp.array(mask_rect)
A_rect = lambda x: fvx.masked_laplacian(x, mask_rect_jnp, dx, dy, lambda_=lambda_)
pc_rect = fvx.make_spectral_preconditioner(dx, dy, lambda_=lambda_, bc="dst")
sol_cg_rect, info_cg_rect = fvx.solve_cg(
    A_rect, rhs, preconditioner=pc_rect, rtol=1e-10, atol=1e-10
)
t_cg_rect = _time_fn(
    lambda: fvx.solve_cg(A_rect, rhs, preconditioner=pc_rect, rtol=1e-10, atol=1e-10)[0]
)
rr_cg_rect = _rel_residual(sol_cg_rect, rhs, A_rect, interior_mask=interior_rect)

# Multigrid standalone
mg_rect = fvx.build_multigrid_solver(mask_rect, dx, dy, lambda_=lambda_, n_cycles=8)
sol_mg_rect = mg_rect(rhs)
t_mg_rect = _time_fn(lambda: mg_rect(rhs))
rr_mg_rect = _rel_residual(
    sol_mg_rect,
    rhs,
    lambda u: _apply_operator(u, mg_rect.levels[0]),
    interior_mask=interior_rect,
)

results["Rectangle"] = {
    "Spectral": {
        "sol": sol_sp_rect,
        "time_ms": t_sp_rect * 1000,
        "rel_residual": rr_sp_rect,
        "label": "exact",
    },
    "CG": {
        "sol": sol_cg_rect,
        "time_ms": t_cg_rect * 1000,
        "rel_residual": rr_cg_rect,
        "label": f"{info_cg_rect.iterations} iters",
    },
    "Multigrid": {
        "sol": sol_mg_rect,
        "time_ms": t_mg_rect * 1000,
        "rel_residual": rr_mg_rect,
        "label": "8 V-cyc",
    },
}

# ── Basin ───────────────────────────────────────────────────────────────
mask_basin_jnp = jnp.array(mask_basin)
rhs_basin = rhs * mask_basin_jnp
A_basin = lambda x: fvx.masked_laplacian(x, mask_basin_jnp, dx, dy, lambda_=lambda_)

# Capacitance
cap = fvx.build_capacitance_solver(
    mask_basin.astype(bool), dx, dy, lambda_=lambda_, base_bc="dst"
)
sol_cap = cap(rhs_basin)
t_cap = _time_fn(lambda: cap(rhs_basin))
# Capacitance is a direct solver — exact by construction
rr_cap = 0.0

# CG
pc_basin = fvx.make_spectral_preconditioner(dx, dy, lambda_=lambda_, bc="dst")
sol_cg_basin, info_cg_basin = fvx.solve_cg(
    A_basin, rhs_basin, preconditioner=pc_basin, rtol=1e-10, atol=1e-10
)
sol_cg_basin = sol_cg_basin * mask_basin_jnp
t_cg_basin = _time_fn(
    lambda: (
        fvx.solve_cg(
            A_basin, rhs_basin, preconditioner=pc_basin, rtol=1e-10, atol=1e-10
        )[0]
        * mask_basin_jnp
    )
)
rr_cg_basin = _rel_residual(
    sol_cg_basin, rhs_basin, A_basin, interior_mask=mask_basin_jnp
)

# Multigrid
mg_basin = fvx.build_multigrid_solver(mask_basin, dx, dy, lambda_=lambda_, n_cycles=8)
sol_mg_basin = mg_basin(rhs_basin)
t_mg_basin = _time_fn(lambda: mg_basin(rhs_basin))
rr_mg_basin = _rel_residual(
    sol_mg_basin,
    rhs_basin,
    lambda u: _apply_operator(u, mg_basin.levels[0]),
    interior_mask=mask_basin_jnp,
)

results["Basin"] = {
    "Capacitance": {
        "sol": sol_cap,
        "time_ms": t_cap * 1000,
        "rel_residual": rr_cap,
        "label": "direct",
    },
    "CG": {
        "sol": sol_cg_basin,
        "time_ms": t_cg_basin * 1000,
        "rel_residual": rr_cg_basin,
        "label": f"{info_cg_basin.iterations} iters",
    },
    "Multigrid": {
        "sol": sol_mg_basin,
        "time_ms": t_mg_basin * 1000,
        "rel_residual": rr_mg_basin,
        "label": "8 V-cyc",
    },
}

# ── Circle ──────────────────────────────────────────────────────────────
mask_circle_jnp = jnp.array(mask_circle)
rhs_circle = rhs * mask_circle_jnp
A_circle = lambda x: fvx.masked_laplacian(x, mask_circle_jnp, dx, dy, lambda_=lambda_)

# CG
pc_circle = fvx.make_spectral_preconditioner(dx, dy, lambda_=lambda_, bc="dst")
sol_cg_circle, info_cg_circle = fvx.solve_cg(
    A_circle, rhs_circle, preconditioner=pc_circle, rtol=1e-10, atol=1e-10
)
sol_cg_circle = sol_cg_circle * mask_circle_jnp
t_cg_circle = _time_fn(
    lambda: (
        fvx.solve_cg(
            A_circle, rhs_circle, preconditioner=pc_circle, rtol=1e-10, atol=1e-10
        )[0]
        * mask_circle_jnp
    )
)
rr_cg_circle = _rel_residual(
    sol_cg_circle, rhs_circle, A_circle, interior_mask=mask_circle_jnp
)

# Multigrid
mg_circle = fvx.build_multigrid_solver(mask_circle, dx, dy, lambda_=lambda_, n_cycles=8)
sol_mg_circle = mg_circle(rhs_circle)
t_mg_circle = _time_fn(lambda: mg_circle(rhs_circle))
rr_mg_circle = _rel_residual(
    sol_mg_circle,
    rhs_circle,
    lambda u: _apply_operator(u, mg_circle.levels[0]),
    interior_mask=mask_circle_jnp,
)

results["Circle"] = {
    "CG": {
        "sol": sol_cg_circle,
        "time_ms": t_cg_circle * 1000,
        "rel_residual": rr_cg_circle,
        "label": f"{info_cg_circle.iterations} iters",
    },
    "Multigrid": {
        "sol": sol_mg_circle,
        "time_ms": t_mg_circle * 1000,
        "rel_residual": rr_mg_circle,
        "label": "8 V-cyc",
    },
}

# ── Notch (variable coefficient) ────────────────────────────────────────
coeff = 1.0 + 0.8 * np.sin(2 * np.pi * X / Nx)
mask_notch_jnp = jnp.array(mask_notch)
rhs_notch = rhs * mask_notch_jnp

mg_notch = fvx.build_multigrid_solver(
    mask_notch, dx, dy, lambda_=lambda_, coeff=coeff, n_cycles=10
)
A_notch = lambda x: _apply_operator(x, mg_notch.levels[0])

# Multigrid standalone
sol_mg_notch = mg_notch(rhs_notch)
t_mg_notch = _time_fn(lambda: mg_notch(rhs_notch))
rr_mg_notch = _rel_residual(
    sol_mg_notch, rhs_notch, A_notch, interior_mask=mask_notch_jnp
)

# MG-preconditioned CG
pc_mg_notch = fvx.make_multigrid_preconditioner(mg_notch)
sol_mgcg_notch, info_mgcg_notch = fvx.solve_cg(
    A_notch, rhs_notch, preconditioner=pc_mg_notch, rtol=1e-12, atol=1e-12
)
sol_mgcg_notch = sol_mgcg_notch * mask_notch_jnp
t_mgcg_notch = _time_fn(
    lambda: (
        fvx.solve_cg(
            A_notch, rhs_notch, preconditioner=pc_mg_notch, rtol=1e-12, atol=1e-12
        )[0]
        * mask_notch_jnp
    )
)
rr_mgcg_notch = _rel_residual(
    sol_mgcg_notch, rhs_notch, A_notch, interior_mask=mask_notch_jnp
)

results["Notch\n(variable coeff)"] = {
    "Multigrid": {
        "sol": sol_mg_notch,
        "time_ms": t_mg_notch * 1000,
        "rel_residual": rr_mg_notch,
        "label": "10 V-cyc",
    },
    "MG+CG": {
        "sol": sol_mgcg_notch,
        "time_ms": t_mgcg_notch * 1000,
        "rel_residual": rr_mgcg_notch,
        "label": f"{info_mgcg_notch.iterations} iters",
    },
}


# ======================================================================
# 3. Solutions figure
# ======================================================================
geom_order = list(results.keys())
max_solvers = max(len(v) for v in results.values())

fig, axes = plt.subplots(4, max_solvers, figsize=(5 * max_solvers, 16))

for row, geom in enumerate(geom_order):
    solvers = list(results[geom].keys())
    mask = geometries[geom]
    for col in range(max_solvers):
        ax = axes[row, col]
        if col < len(solvers):
            sname = solvers[col]
            sol = np.asarray(results[geom][sname]["sol"])
            sol_masked = np.where(mask > 0.5, sol, np.nan)
            im = ax.imshow(
                sol_masked,
                origin="lower",
                cmap="RdBu_r",
                interpolation="nearest",
            )
            label = results[geom][sname]["label"]
            ax.set_title(f"{sname} ({label})", fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.7)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(geom, fontsize=11, fontweight="bold")

fig.suptitle("Solutions by Solver and Geometry", fontsize=15, y=1.01)
plt.tight_layout()
fig.savefig(OUT / "solvers_solutions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved solvers_solutions.png")


# ======================================================================
# 4. Residual + timing combined figure
# ======================================================================
fig, (ax_res, ax_time) = plt.subplots(1, 2, figsize=(16, 5))

colors = {
    "Spectral": "#2196F3",
    "Capacitance": "#4CAF50",
    "CG": "#FF9800",
    "Multigrid": "#9C27B0",
    "MG+CG": "#E91E63",
}

# Collect data for grouped bar chart
all_entries = []  # (geom_short, solver, time_ms, rel_residual)
for geom in geom_order:
    geom_short = geom.split("\n")[0]  # strip "(variable coeff)" line
    for sname, data in results[geom].items():
        all_entries.append((geom_short, sname, data["time_ms"], data["rel_residual"]))

# Group by geometry for bar positions
geom_shorts = [g.split("\n")[0] for g in geom_order]
x_positions = []
x_labels = []
bar_colors = []
times_list = []
residuals_list = []
solver_names = []
group_width = 0.8
idx = 0
for gi, geom in enumerate(geom_order):
    solvers = list(results[geom].keys())
    n_s = len(solvers)
    bar_w = group_width / n_s
    for si, sname in enumerate(solvers):
        x_pos = gi + (si - (n_s - 1) / 2) * bar_w
        x_positions.append(x_pos)
        x_labels.append(sname)
        bar_colors.append(colors.get(sname, "#888888"))
        times_list.append(results[geom][sname]["time_ms"])
        residuals_list.append(results[geom][sname]["rel_residual"])
        solver_names.append(sname)
        idx += 1

# Timing bars
bars = ax_time.bar(
    x_positions,
    times_list,
    width=group_width / 3.5,
    color=bar_colors,
    edgecolor="black",
    linewidth=0.5,
)
for bar, t in zip(bars, times_list, strict=True):
    ax_time.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"{t:.1f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
ax_time.set_xticks(range(len(geom_shorts)))
ax_time.set_xticklabels(geom_shorts, fontsize=10)
ax_time.set_ylabel("Time (ms)", fontsize=11)
ax_time.set_title("Solve Time (JIT-compiled, 64x64)", fontsize=12)

# Residual bars
# Direct solvers have residual=0; floor at 1e-15 for log scale display
residuals_plot = [max(r, 1e-15) for r in residuals_list]
bars_r = ax_res.bar(
    x_positions,
    residuals_plot,
    width=group_width / 3.5,
    color=bar_colors,
    edgecolor="black",
    linewidth=0.5,
)
ax_res.set_yscale("log")
ax_res.set_xticks(range(len(geom_shorts)))
ax_res.set_xticklabels(geom_shorts, fontsize=10)
ax_res.set_ylabel("Relative Residual ||r|| / ||b||", fontsize=11)
ax_res.set_title("Accuracy (Relative Residual Norm)", fontsize=12)
ax_res.set_ylim(bottom=1e-16)

# Legend (unique solver names)
from matplotlib.patches import Patch

legend_entries = []
seen = set()
for sname in solver_names:
    if sname not in seen:
        seen.add(sname)
        legend_entries.append(Patch(facecolor=colors.get(sname, "#888"), label=sname))
ax_res.legend(handles=legend_entries, loc="upper left", fontsize=9)
ax_time.legend(handles=legend_entries, loc="upper left", fontsize=9)

plt.tight_layout()
fig.savefig(OUT / "solvers_accuracy_timing.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved solvers_accuracy_timing.png")


# ======================================================================
# 5. Print summary table
# ======================================================================
print("\n" + "=" * 80)
print(
    f"{'Geometry':<22} {'Solver':<14} {'Time (ms)':>10} {'Rel. Residual':>14} {'Detail'}"
)
print("=" * 80)
for geom in geom_order:
    geom_short = geom.split("\n")[0]
    for sname, data in results[geom].items():
        print(
            f"{geom_short:<22} {sname:<14} {data['time_ms']:>10.2f} "
            f"{data['rel_residual']:>14.1e} {data['label']}"
        )
    print("-" * 80)
