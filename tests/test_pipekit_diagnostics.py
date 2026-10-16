"""Diagnostic, reduction and time-step operators of `finitevolx.pipekit`."""

from __future__ import annotations

import pytest

pytest.importorskip("pipekit_jax")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pipekit import Graph, Input
from pipekit_jax import Rollout

import finitevolx as fvx
from finitevolx import pipekit as fp
from finitevolx.pipekit.testing import check_operator
from tests.fixtures.inputs import make_grid_2d

GRID = make_grid_2d()
SPEC = fp.as_grid_spec(GRID)
V2 = fvx.Vorticity2D(grid=GRID)
E2 = fvx.Energetics2D(grid=GRID)
S2 = fvx.Strain2D(grid=GRID)


def _field(seed: int, lead: tuple[int, ...] = ()) -> jax.Array:
    return jax.random.normal(jax.random.key(seed), (*lead, GRID.Ny, GRID.Nx))


def _y() -> jax.Array:
    y = jnp.arange(GRID.Ny) * GRID.dy
    return jnp.broadcast_to(y[:, None], (GRID.Ny, GRID.Nx))


def _positive(seed: int, lead: tuple[int, ...] = ()) -> jax.Array:
    return 2.0 + 0.1 * _field(seed, lead)


# (id, operator, Tier-A reference, input makers)
CASES = [
    ("RelativeVorticity", fp.RelativeVorticity(grid=SPEC), V2.relative_vorticity, 2),
    ("Enstrophy", fp.Enstrophy(grid=SPEC), V2.enstrophy, 2),
    (
        "PotentialVorticity",
        fp.PotentialVorticity(grid=SPEC),
        V2.potential_vorticity,
        (_field, _field, _positive, _field),
    ),
    (
        "QGPotentialVorticity",
        fp.QGPotentialVorticity(grid=SPEC, f0=1.5, beta=0.2),
        lambda psi: fvx.QGPotentialVorticity2D(grid=GRID)(
            psi, 1.5, 0.2, _y(), (GRID.Ny - 1) * GRID.dy / 2
        ),
        1,
    ),
    ("KineticEnergy", fp.KineticEnergy(grid=SPEC), E2.kinetic_energy, 2),
    (
        "AvailablePE",
        fp.AvailablePE(grid=SPEC, g_prime=0.7, H=1.2),
        lambda h: E2.available_potential_energy(h, jnp.full_like(h, 1.2), 0.7),
        1,
    ),
    *[
        (f"Strain-{c}", fp.Strain(grid=SPEC, component=c), getattr(S2, c), 2)
        for c in sorted(fp.STRAIN_COMPONENTS)
    ],
    ("OkuboWeiss", fp.OkuboWeiss(grid=SPEC), S2.okubo_weiss, 2),
    (
        "TotalEnergy",
        fp.TotalEnergy(grid=SPEC),
        lambda ke, ape: fvx.total_energy(ke, ape, GRID.dx, GRID.dy),
        2,
    ),
    (
        "DomainIntegral",
        fp.DomainIntegral(grid=SPEC),
        lambda x: fvx.area_sum(x, GRID),
        1,
    ),
    ("DomainMean", fp.DomainMean(grid=SPEC), lambda x: fvx.area_mean(x, GRID), 1),
]
IDS = [c[0] for c in CASES]


def _inputs(makers, seed0: int, lead: tuple[int, ...] = ()) -> list[jax.Array]:
    if isinstance(makers, int):
        makers = (_field,) * makers
    return [make(seed0 + k, lead) for k, make in enumerate(makers)]


@pytest.mark.slow  # compiles the operator several times (~2 s)
@pytest.mark.parametrize(("name", "op", "ref", "makers"), CASES, ids=IDS)
def test_contract(name, op, ref, makers):
    check_operator(op, *_inputs(makers, 0))


@pytest.mark.parametrize(("name", "op", "ref", "makers"), CASES, ids=IDS)
def test_equals_tier_a(name, op, ref, makers):
    xs = _inputs(makers, 10)
    np.testing.assert_allclose(op(*xs), ref(*xs), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(("name", "op", "ref", "makers"), CASES, ids=IDS)
def test_keeps_leading_axes(name, op, ref, makers):
    lead = (2, 3)
    xs = _inputs(makers, 20, lead)
    got = op(*xs)
    flat = [x.reshape(-1, GRID.Ny, GRID.Nx) for x in xs]
    want = jnp.stack([ref(*[f[m] for f in flat]) for m in range(6)])
    np.testing.assert_allclose(got, want.reshape(*lead, *want.shape[1:]), rtol=1e-13)


def test_invalid_strain_component_lists_the_valid_set():
    with pytest.raises(ValueError, match="magnitude_squared"):
        fp.Strain(grid=SPEC, component="normal")


def test_explicit_step_rejects_unknown_schemes():
    with pytest.raises(ValueError, match="rk4"):
        fp.ExplicitStep(rhs=fp.Laplacian(grid=SPEC), dt=0.1, scheme="leapfrog")


# ------------------------------------------------- energy of linear shallow water
N, H, G, F0 = 16, 1.0, 1.0, 5.0
SW_SPEC = (N, N, 1.0, 1.0)
DT = 0.2 / N  # CFL c Δt / Δx = 0.2 with c = √(gH) = 1


def sw_tendency() -> Graph:
    """∂ₜh = −H ∇·(u, v),  ∂ₜu = f v̄ − g δₓh,  ∂ₜv = −f ū − g δᵧh (periodic)."""
    h, u, v = Input("h"), Input("u"), Input("v")
    hp, up, vp = (fp.FillGhosts(kind="periodic")(x) for x in (h, u, v))
    dh = fp.LinComb(coeffs=(-H,))(fp.Divergence(grid=SW_SPEC)(up, vp))
    cor = fp.Coriolis(grid=SW_SPEC, f0=F0)(up, vp)
    gx = fp.Difference(grid=SW_SPEC, axis="x", src="T", dst="U")(hp)
    gy = fp.Difference(grid=SW_SPEC, axis="y", src="T", dst="V")(hp)
    du = fp.LinComb(coeffs=(1.0, -G))(fp.Pick(index=0)(cor), gx)
    dv = fp.LinComb(coeffs=(1.0, -G))(fp.Pick(index=1)(cor), gy)
    return Graph(inputs={"h": h, "u": u, "v": v}, outputs={"h": dh, "u": du, "v": dv})


def energy_graph() -> Graph:
    """E = ∑ (½ H (ū² + v̄²) + ½ g h²) ΔA over a [T, Ny, Nx] trajectory → [T]."""
    h, u, v = Input("h"), Input("u"), Input("v")
    hp, up, vp = (fp.FillGhosts(kind="periodic")(x) for x in (h, u, v))
    ke = fp.LinComb(coeffs=(H,))(fp.KineticEnergy(grid=SW_SPEC)(up, vp))
    ape = fp.AvailablePE(grid=SW_SPEC, g_prime=G)(hp)
    return Graph(
        inputs={"h": h, "u": u, "v": v},
        outputs={"E": fp.TotalEnergy(grid=SW_SPEC)(ke, ape)},
    )


def _initial_state() -> dict[str, jax.Array]:
    g = fp.build_grid(SW_SPEC)
    y = (jnp.arange(g.Ny) - 0.5) / N
    x = (jnp.arange(g.Nx) - 0.5) / N
    Y, X = jnp.meshgrid(y, x, indexing="ij")
    h = 0.1 * jnp.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * 0.1**2))
    z = jnp.zeros_like(h)
    return {"h": h, "u": z, "v": z}


def test_energy_graph_equals_tier_a_on_a_trajectory():
    traj = {k: _field(i, (4,)) for i, k in enumerate("huv")}
    got = energy_graph()(**traj)["E"]
    g = fp.build_grid(SW_SPEC)
    filled = {k: fvx.enforce_periodic(traj[k]) for k in "huv"}
    ke = fvx.Energetics2D(grid=g).kinetic_energy(filled["u"], filled["v"])
    ape = 0.5 * G * filled["h"] ** 2
    want = fvx.total_energy(H * ke, ape, g.dx, g.dy)
    assert got.shape == (4,)
    np.testing.assert_allclose(got, want, rtol=1e-13)


def test_explicit_step_equals_rk4_step():
    s0 = _initial_state()
    rhs = sw_tendency()
    got = fp.ExplicitStep(rhs=rhs, dt=DT)(s0)
    want = fvx.rk4_step(s0, lambda s: rhs(**s), DT)
    for k in "huv":
        np.testing.assert_array_equal(got[k], want[k])


def _energy_drift(refine: int) -> tuple[float, jax.Array]:
    """max |E(t)/E(0) − 1| over t ∈ [0, 200 Δt], with Δt / refine."""
    step = fp.ExplicitStep(rhs=sw_tendency(), dt=DT / refine)
    rollout = Rollout(step=step, n_steps=200 * refine, save_every=20 * refine)
    traj = eqx.filter_jit(lambda r, s: r(s))(rollout, _initial_state())
    E = energy_graph()(**traj)["E"]  # [11]
    assert E.shape == (11,)
    return float(jnp.max(jnp.abs(E / E[0] - 1))), traj["h"]


@pytest.mark.integration
def test_energy_is_conserved_over_a_rollout():
    # The spatial scheme conserves E exactly, so all drift is RK4 time
    # error, which shrinks like Δt⁴ over a fixed interval (measured: 5e-5,
    # 1.7e-6, 5.4e-8 for refine = 1, 2, 4).
    drift2, h = _energy_drift(2)
    drift4, _ = _energy_drift(4)
    assert drift2 < 1e-5
    assert drift2 / drift4 > 16
    assert not np.allclose(h[-1], h[0])  # the wave did move


def test_explicit_step_config_and_gradients():
    import json

    step = fp.ExplicitStep(rhs=sw_tendency(), dt=DT, scheme="rk3_ssp")
    cfg = json.loads(json.dumps(step.state))["config"]
    assert cfg["scheme"] == "rk3_ssp" and cfg["rhs"]["class"] == "Graph"

    def loss(st):
        return jnp.sum(st(_initial_state())["u"] ** 2)

    grads = eqx.filter_grad(loss)(step)
    f0_grads = [
        g.f0
        for g in jax.tree.leaves(grads, is_leaf=lambda x: isinstance(x, fp.Coriolis))
        if isinstance(g, fp.Coriolis)
    ]
    assert f0_grads and all(jnp.isfinite(g) for g in f0_grads)
