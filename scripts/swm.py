""" swm.py

2D shallow water model with:

- varying Coriolis force
- nonlinear terms
- lateral friction
- periodic boundary conditions

Script Taken from:
    https://github.com/dionhaefner/shallow-water/blob/master/shallow_water_nonlinear.py

        # ########
        # H --> V :=> HV (Flux)
        # HV --> Q :=> QHV (Flux)
        # QHV --> U :=> QHV_on_U
        # ########
        # H --> V :=> HV (Flux)
        # HV --> U :=> HV_on_U (Flux)
        # Q --> U,Recons :==> QHV_on_U
"""

import numpy as np
import matplotlib.pyplot as plt
import autoroot
from jaxtyping import Float, Array
import jax.numpy as jnp
from finitevolx import x_avg_2D, y_avg_2D,center_avg_2D, MaskGrid, difference, divergence, relative_vorticity, reconstruct
import jax

jax.config.update("jax_enable_x64", True)

# grid setup
n_x = 200
dx = 5e3
l_x = n_x * dx

n_y = 104
dy = 5e3
l_y = n_y * dy

x, y = (
    np.arange(n_x) * dx,
    np.arange(n_y) * dy
)
X, Y = np.meshgrid(x, y, indexing='ij')

# physical parameters
gravity = 9.81
depth = 100.
coriolis_f = 2e-4
coriolis_beta = 2e-11
coriolis_param: Float[Array, "Nx Ny"] = coriolis_f + Y * coriolis_beta
lateral_viscosity = 1e-3 * coriolis_f * dx ** 2

# other parameters
periodic_boundary_x = False
linear_momentum_equation = False

adams_bashforth_a = 1.5 + 0.1
adams_bashforth_b = -(0.5 + 0.1)

dt = 0.125 * min(dx, dy) / np.sqrt(gravity * depth)

phase_speed = np.sqrt(gravity * depth)
rossby_radius = np.sqrt(gravity * depth) / coriolis_param.mean()

# plot parameters
plot_range = 10
plot_every = 10
max_quivers = 41



# mask
mask = jnp.ones_like(X)
mask = mask.at[-1].set(0.0)
mask = mask.at[:, 0].set(0.0)
mask = mask.at[:, -1].set(0.0)
mask = mask.at[0].set(0.0)
masks = MaskGrid.init_mask(mask, "center")

# initial conditions
x_, y_ = (
    np.arange(n_x+1) * dx,
    np.arange(n_y) * dy
)
X_, Y_ = np.meshgrid(x_, y_, indexing='ij')
u0 = 10 * np.exp(-(Y_ - y_[n_y // 2])**2 / (0.02 * l_x)**2)

v0 = np.zeros_like(masks.face_v.values)

# approximate balance h_y = -(f/g)u
h_geostrophy = np.cumsum(-dy * x_avg_2D(u0) * coriolis_param / gravity, axis=0)
h0 = (
    depth
    + h_geostrophy
    # make sure h0 is centered around depth
    - h_geostrophy.mean()
    # small perturbation
    + 0.2 * np.sin(X / l_x * 10 * np.pi) * np.cos(Y / l_y * 8 * np.pi)
)




def prepare_plot():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    u0_on_h = x_avg_2D(u0)
    v0_on_h = y_avg_2D(v0)

    cs = update_plot(0, h0, u0_on_h, v0_on_h, ax)
    plt.colorbar(cs, label="$\\eta$ (m)")
    return fig, ax


def update_plot(t, h, u, v, ax):
    eta = h - depth

    quiver_stride = (slice(1, -1, n_x // max_quivers), slice(1, -1, n_y // max_quivers))

    ax.clear()
    cs = ax.pcolormesh(
        x[1:-1] / 1e3,
        y[1:-1] / 1e3,
        eta[1:-1, 1:-1].T,
        vmin=-plot_range,
        vmax=plot_range,
        cmap="RdBu_r",
    )

    if np.any((u[quiver_stride] != 0) | (v[quiver_stride] != 0)):
        ax.quiver(
            x[quiver_stride[0]] / 1e3,
            y[quiver_stride[1]] / 1e3,
            u[quiver_stride].T,
            v[quiver_stride].T,
            clip_on=False,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (km)")
    ax.set_ylabel("$y$ (km)")
    ax.set_xlim(x[1] / 1e3, x[-2] / 1e3)
    ax.set_ylim(y[1] / 1e3, y[-2] / 1e3)
    ax.set_title(
        "t=%5.2f days, R=%5.1f km, c=%5.1f m/s " % (t / 86400, rossby_radius / 1e3, phase_speed)
    )
    plt.pause(0.1)
    return cs


def enforce_boundaries(u, grid: str="h"):
    assert grid in ('h', 'u', 'v')
    if periodic_boundary_x:
        u = u.at[0].set(u[-2])
        u = u.at[-1].set(u[1])
    elif grid == 'u':
        u = u.at[-1, :].set(0.0)
    if grid == 'v':
        u = u.at[:,-1].set(0.0)
    return u


def iterate_shallow_water():
    # allocate arrays
    u, v, h = jnp.empty_like(u0), jnp.empty_like(v0), jnp.empty_like(h0)

    # initial conditions
    h: Float[Array, "Nx Ny"] = h.at[:].set(h0)
    u: Float[Array, "Nx+1 Ny"] = u.at[:].set(u0)
    v: Float[Array, "Nx Ny+1"] = v.at[:].set(v0)

    # # apply masks
    # h *= masks.center.values
    # u *= masks.face_u.values
    # v *= masks.face_v.values


    first_step = True

    # time step equations
    while True:
        # print(f"h: {h.shape}")
        h_pad: Float[Array, "Nx+2 Ny+2"] = jnp.pad(h, 1, "edge")
        u = enforce_boundaries(u, "u")
        v = enforce_boundaries(v, "v")
        u_pad: Float[Array, "Nx+3 Ny+2"] = jnp.pad(u, 1, "constant")
        v_pad: Float[Array, "Nx+2 Ny+3"] = jnp.pad(v, 1, "constant")



        # ===================================
        # flux - north/south | east/west
        # ===================================
        # uh_flux: Float[Array, "Nx-1 Ny"] = reconstruct(q=hc, u=u[1:-1,:], u_mask=masks.face_u[1:-1,:], dim=0, num_pts=3, method="linear")
        # vh_flux: Float[Array, "Nx Ny-1"] = reconstruct(q=hc, u=v[:, 1:-1], u_mask=masks.face_v[:,1:-1], dim=1, num_pts=3, method="linear")
        uh_flux: Float[Array, "Nx+1 Ny+2"] = x_avg_2D(h_pad) * u_pad[1:-1,:]
        vh_flux: Float[Array, "Nx+2 Ny+1"] = y_avg_2D(h_pad) * v_pad[:,1:-1]

        # uh_flux = enforce_boundaries(uh_flux, "u")
        # vh_flux = enforce_boundaries(vh_flux, "v")


        # =========================
        # RHS HEIGHT
        # =========================
        dhu_flux: Float[Array, "Nx Ny"] = difference(uh_flux[:, 1:-1], axis=0, step_size=dx, derivative=1)
        dhv_flux: Float[Array, "Nx Ny"] = difference(vh_flux[1:-1,:], axis=1, step_size=dy, derivative=1)
        # print(f"h: {h.shape} | dh_dx: {dfe_dx.shape} | dh_dy: {dfn_dy.shape}")
        h_rhs: Float[Array, "Nx Ny"] = - (dhu_flux + dhv_flux)

        # h_rhs *= masks.center.values

        # ================================
        # update zonal velocity, u
        # ================================
        v_avg: Float[Array, "Nx+1 Ny+2"] = center_avg_2D(v_pad)
        dh_dx: Float[Array, "Nx+1 Ny+2"] = difference(h_pad, step_size=dx, axis=0, derivative=1)

        f0_on_u = x_avg_2D(coriolis_param)
        u_rhs: Float[Array, "Nx-1 Ny"] = f0_on_u * v_avg[1:-1, 1:-1] - gravity * dh_dx[1:-1, 1:-1]

        # apply masks
        # u_rhs *= masks.face_u.values
        
        u_rhs: Float[Array, "Nx+1 Ny"] = jnp.pad(u_rhs, pad_width=((1,1),(0, 0),))



        # =================================
        # update meridional velocity, v
        # =================================
        u_avg: Float[Array, "Nx+2 Ny+1"] = center_avg_2D(u_pad)
        dh_dy: Float[Array, "Nx+2 Ny+1"] = difference(h_pad, step_size=dy, axis=1, derivative=1)

        f0_on_v = y_avg_2D(coriolis_param)
        v_rhs: Float[Array, "Nx Ny-1"] = - f0_on_v * u_avg[1:-1,1:-1] - gravity * dh_dy[1:-1,1:-1]

        # apply masks
        # v_rhs *= masks.face_v.values[:, 1:-1]

        v_rhs: Float[Array, "Nx Ny+1"] = jnp.pad(v_rhs, pad_width=((0, 0),(1, 1)))

        # # ===================================
        # # kinetic energy
        # # ===================================
        # u2_on_h: Float[Array, "Nx Ny"] = x_avg_2D(u**2)
        # v2_on_h: Float[Array, "Nx Ny"] = y_avg_2D(v**2)
        # ke_on_h: Float[Array, "Nx Ny"] = 0.5 * (u2_on_h + v2_on_h)

        # # ==============================
        # # Potential Vorticity
        # # ==============================
        # # planetary vorticity
        # f0_on_q: Float[Array, "Nx-1 Ny-1"] = center_avg_2D(coriolis_param)
        # # relative vorticity
        # u_i: Float[Array, "Nx-1 Ny"] = u[1:-1, :]
        # v_i: Float[Array, "Nx Ny-1"] = v[:, 1:-1]
        # vort_r: Float[Array, "Nx-1 Ny-1"] = relative_vorticity(u=u_i, v=v_i, dx=dx, dy=dy)
        # # potential vorticity, q=(zeta + f)/h
        # hc_on_q: Float[Array, "Nx-1 Ny-1"] = center_avg_2D(hc)
        # q: Float[Array, "Nx-1 Ny-1"] = (f0_on_q + vort_r) / hc_on_q


        # # ========================
        # # RHS ZONAL VELOCITY, U
        # # ========================
        # # nonlinear momentum equation
        #
        # # flux term
        # vh_flux_on_u: Float[Array, "Nx-1 Ny-2"] = center_avg_2D(vh_flux)
        #
        # qhv_on_u: Float[Array, "Nx-1 Ny-2"] = reconstruct(q=q, u=vh_flux_on_u, dim=1, u_mask=masks.face_u[1:-1,1:-1])
        #
        # # calculate work
        # dh_dx: Float[Array, "Nx-1 Ny"] = difference(h, step_size=dx, derivative=1, axis=0)
        # work = gravity * dh_dx
        #
        # dke_on_u: Float[Array, "Nx-1 Ny"] = difference(ke_on_h, axis=0, step_size=dx, derivative=1)
        #
        # u_rhs: Float[Array, "Nx-1 Ny-2"] = - work[:, 1:-1] + qhv_on_u - dke_on_u[:, 1:-1]
        #
        # u_rhs: Float[Array, "Nx+1 Ny"] = jnp.pad(u_rhs, pad_width=1, mode="constant", constant_values=0.0)
        #
        # u_rhs *= masks.face_u.values
        #
        # # ========================
        # # RHS ZONAL VELOCITY, U
        # # ========================
        # # nonlinear momentum equation
        # # # QHV on U
        # # fe_on_q: Float[Array, "Nx-1 Ny-1"] = y_avg_2D(uh_flux)
        # # qhu: Float[Array, "Nx-1 Ny-1"] = q[1:-1, 1:-1] * fe_on_q
        # # qhu_on_v: Float[Array, "Nx-2 Ny-1"] = x_avg_2D(qhu)
        #
        # # flux term
        # uh_flux_on_v: Float[Array, "Nx-2 Ny-1"] = center_avg_2D(uh_flux)
        # # Q_i: [Nx-1,Ny-1], VH_flux: [Nx Ny-1]
        # qhu_on_v: Float[Array, "Nx-2 Ny-1"] = reconstruct(q=q, u=uh_flux_on_v, dim=0, u_mask=masks.face_v[1:-1,1:-1])
        #
        #
        # dke_on_v: Float[Array, "Nx Ny-1"] = difference(ke_on_h, axis=1, step_size=dy, derivative=1)
        #
        # # calculate work
        # dh_dy: Float[Array, "Nx Ny-1"] = difference(h, step_size=dy, derivative=1, axis=1)
        # work = gravity * dh_dy
        #
        #
        # v_rhs: Float[Array, "Nx-2 Ny-1"] = - work[1:-1,:] - qhu_on_v - dke_on_v[1:-1,:]
        #
        # v_rhs: Float[Array, "Nx+1 Ny"] = jnp.pad(v_rhs, pad_width=1, mode="constant", constant_values=0.0)
        #
        # v_rhs *= masks.face_v.values
        
        print(h.shape, u.shape, v.shape)
        print(h_rhs.shape, u_rhs.shape, v_rhs.shape)

        if first_step:

            u += dt * u_rhs
            v += dt * v_rhs
            h += dt * h_rhs
            first_step = False
        else:
            u += dt * (
                adams_bashforth_a * u_rhs
                + adams_bashforth_b * u_rhs_old
            )
            v += dt * (
                adams_bashforth_a * v_rhs
                + adams_bashforth_b * v_rhs_old
            )
            h += dt * (
                adams_bashforth_a * h_rhs_old
                + adams_bashforth_b * h_rhs
            )
        #
        h = enforce_boundaries(h, 'h')
        u = enforce_boundaries(u, 'u')
        v = enforce_boundaries(v, 'v')
        #
        # if lateral_viscosity > 0:
        #     # lateral friction
        #     fe[1:-1, 1:-1] = lateral_viscosity * (u[1:-1, 2:] - u[1:-1, 1:-1]) / dx
        #     fn[1:-1, 1:-1] = lateral_viscosity * (u[2:, 1:-1] - u[1:-1, 1:-1]) / dy
        #     fe = enforce_boundaries(fe, 'u')
        #     fn = enforce_boundaries(fn, 'v')
        #
        #     u[1:-1, 1:-1] += dt * (
        #         (fe[1:-1, 1:-1] - fe[1:-1, :-2]) / dx
        #         + (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / dy
        #     )
        #
        #     fe[1:-1, 1:-1] = lateral_viscosity * (v[1:-1, 2:] - u[1:-1, 1:-1]) / dx
        #     fn[1:-1, 1:-1] = lateral_viscosity * (v[2:, 1:-1] - u[1:-1, 1:-1]) / dy
        #     fe = enforce_boundaries(fe, 'u')
        #     fn = enforce_boundaries(fn, 'v')
        #
        #     v[1:-1, 1:-1] += dt * (
        #         (fe[1:-1, 1:-1] - fe[1:-1, :-2]) / dx
        #         + (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / dy
        #     )
        #
        # rotate quantities
        h_rhs_old = h_rhs
        v_rhs_old = v_rhs
        u_rhs_old = u_rhs

        yield h, u, v


if __name__ == "__main__":
    fig, ax = prepare_plot()

    # create model generator
    model = iterate_shallow_water()

    # iterate through steps
    for iteration, (h, u, v) in enumerate(model):
        if iteration % plot_every == 0:
            t = iteration * dt

            # move face variables to center
            # u,v --> h
            u_on_h = center_avg_2D(u)
            v_on_h = center_avg_2D(v)

            # update plot
            print("here!")
            print(h.shape, u_on_h.shape, v_on_h.shape)
            update_plot(t, h, u_on_h, v_on_h, ax)

        # stop if user closes plot window
        if not plt.fignum_exists(fig.number):
            break