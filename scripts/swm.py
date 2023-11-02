""" swm.py

2D shallow water model with:

- varying Coriolis force
- nonlinear terms
- no lateral friction
- periodic boundary conditions
- reconstructions, 5pt, improved weno
    * mass term: h --> u,v
    * momentum term: q --> uh,vh
"""
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
import autoroot
from jaxtyping import Float, Array
import jax.numpy as jnp
from fieldx._src.domain.domain import Domain
from finitevolx import x_avg_2D, y_avg_2D,center_avg_2D, MaskGrid, difference, divergence, relative_vorticity, reconstruct
import jax

jax.config.update("jax_enable_x64", True)



# ==============================
# DOMAIN
# ==============================

# grid setup, height
Nx, Ny = 200, 104
dx, dy = 5e3, 5e3
Lx, Ly = Nx * dx, Ny * dy

# define domains - Arakawa C-Grid Configuration
h_domain = Domain(xmin=(0.0, 0.0), xmax=(Lx, Ly), Lx=(Lx,Ly), Nx=(Nx,Ny), dx=(dx,dy))
u_domain = Domain(xmin=(-0.5, 0.0), xmax=(Lx+0.5*dx, Ly), Lx=(Lx+dx,Ly), Nx=(Nx+1,Ny), dx=(dx,dy))
v_domain = Domain(xmin=(0.0, -0.5), xmax=(Lx, Ly+0.5*dy), Lx=(Lx,Ly+0.5*dy), Nx=(Nx,Ny+1), dx=(dx,dy))
q_domain = Domain(xmin=(-0.5,-0.5), xmax=(Lx+0.5*dx, Ly+0.5*dy), Lx=(Lx+dx,Ly+dy), Nx=(Nx+1,Ny+1), dx=(dx,dy))

# ==============================
# PARAMETERS
# ==============================
# physical parameters
gravity = 9.81
depth = 100.
coriolis_f = 2e-4
coriolis_beta = 2e-11
coriolis_param: Float[Array, "Nx Ny"] = coriolis_f + h_domain.grid_axis[1] * coriolis_beta
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


# model params
num_pts = 5
method = "linear"



# ==============================
# MASK
# ==============================
mask = jnp.ones(h_domain.Nx)
masks = MaskGrid.init_mask(mask, "center")

# ==============================
# INITIAL CONDITIONS
# ==============================
def init_u0(domain):
    Y = domain.grid_axis[1]
    y = domain.coords_axis[1]
    Ny = domain.Nx[1]
    Lx = domain.Lx[0]
    u0 = 10 * jnp.exp( - (Y - y[Ny//2])**2 / (0.02 * Lx)**2)
    return u0
def init_h0_jet(domain, u0):
    
    dy = domain.dx[1]
    Lx, Ly = domain.Lx
    X, Y = domain.grid_axis
    
    h_geostrophy = jnp.cumsum(
        - dy 
        * x_avg_2D(u0)
        * coriolis_param / gravity, 
        axis=1
    )
    
    
    
    h0 = (
        depth
        + h_geostrophy
        # make sure h0 is centered around depth
        - h_geostrophy.mean()
        # small perturbation
        + 0.2 * jnp.sin(X / Lx * 10. * jnp.pi) *
        jnp.cos(Y / Ly * 8. * jnp.pi)
    )
    
    return h0

u0 = init_u0(u_domain)
h0 = init_h0_jet(h_domain, u0)
v0 = jnp.zeros(v_domain.Nx)


# ==============================
# STATE
# ==============================
class State(NamedTuple):
    h: Array
    u: Array
    v: Array

# ==============================
# BOUNDARY CONDITIONS
# ==============================

def enforce_boundaries(u, grid: str="h"):
    assert grid in ('h', 'u', 'v')
    if periodic_boundary_x:
        u = u.at[0].set(u[-2])
        u = u.at[-1].set(u[1])
    elif grid == "u":
        u = u.at[-1,:].set(0.0)
    if grid == "v":
        u = u.at[:,-1].set(0.0)
    return u




def prepare_plot():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # move velocities (faces) to height (centers)
    u0_on_h = x_avg_2D(u0)
    v0_on_h = y_avg_2D(v0)

    cs = update_plot(0, h0, u0_on_h, v0_on_h, ax)
    plt.colorbar(cs, label="$\\eta$ (m)")
    return fig, ax


def update_plot(t, h, u, v, ax):
    eta = h - depth
    
    Nx, Ny = h_domain.Nx
    x, y = h_domain.coords_axis
    quiver_stride = (slice(1, -1, Nx // max_quivers), slice(1, -1, Ny // max_quivers))

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
    plt.pause(0.001)
    return cs

# ####################################
# EQUATION OF MOTION
# ####################################

linear_mass = False
linear_momentum = False


# ====================================
# Nonlinear Terms
# ====================================
def calculate_uvh_flux(h, u, v):
    """
    Eq: 
        (uh), (vh)
    """
    
    h_pad: Float[Array, "Nx+2 Ny+2"] = jnp.pad(h, pad_width=((1,1),(1,1)), mode="edge")
    
    # calculate h fluxes
    uh_flux: Float[Array, "Nx+1 Ny"] = reconstruct(q=h_pad[:,1:-1], u=u, u_mask=masks.face_u, dim=0, num_pts=num_pts, method=method)
    vh_flux: Float[Array, "Nx Ny+1"] = reconstruct(q=h_pad[1:-1,:], u=v, u_mask=masks.face_v, dim=1, num_pts=num_pts, method=method)
    
    uh_flux *= masks.face_u.values
    vh_flux *= masks.face_v.values
    
    return uh_flux, vh_flux


def kinetic_energy(u, v):    
    """
    Eq: 
        ke = 0.5 (u² + v²)
    """
    # calculate squared components
    u2_on_h: Float[Array, "Nx Ny"] = x_avg_2D(u**2)
    v2_on_h: Float[Array, "Nx Ny"] = y_avg_2D(v**2)
    
    # calculate kinetic energy
    ke_on_h: Float[Array, "Nx Ny"] = 0.5 * (u2_on_h + v2_on_h)
    
    # apply mask
    ke_on_h *= masks.center.values
    
    return ke_on_h

def potential_vorticity(h, u, v):
    """
    Eq: 
        ζ = ∂v/∂x - ∂u/∂y
        q = (ζ + f) / h
    """
    # pad arrays
    h_pad: Float[Array, "Nx+2 Ny+2"] = jnp.pad(h, pad_width=1, mode="edge")
    u_pad: Float[Array, "Nx+1 Ny+2"] = jnp.pad(u, pad_width=((0,0),(1,1)), mode="constant")
    v_pad: Float[Array, "Nx+2 Ny+1"] = jnp.pad(v, pad_width=((1,1),(0,0)), mode="constant")
    
    # planetary vorticity, f
    f_on_q: Float[Array, "Nx+1 Ny+1"] = coriolis_f + q_domain.grid_axis[1] * coriolis_beta
    
    # relative vorticity, ζ = dv/dx - du/dy
    vort_r: Float[Array, "Nx+1 Ny+1"] = relative_vorticity(u=u_pad, v=v_pad, dx=v_domain.dx[0], dy=u_domain.dx[1])
    
    # potential vorticity, q = (ζ + f) / h
    h_on_q: Float[Array, "Nx+1 Ny+1"] = center_avg_2D(h_pad)
    q: Float[Array, "Nx+1 Ny+1"] = (vort_r + f_on_q) / h_on_q
    
    # apply masks
    q *= masks.node.values
    
    return q

# ====================================
# HEIGHT
# ====================================
def h_linear_rhs(u, v):
    """
    Eq:
       ∂h/∂t = - H (∂u/∂x + ∂v/∂y)
    """

    # calculate RHS terms
    du_dx: Float[Array, "Nx Ny"] = difference(u, step_size=u_domain.dx[0], axis=0, derivative=1)
    dv_dy: Float[Array, "Nx Ny"] = difference(v, step_size=v_domain.dx[1], axis=1, derivative=1)

    # calculate RHS
    h_rhs: Float[Array, "Nx Ny"] = - depth * (du_dx + dv_dy)

    # apply masks
    h_rhs *= masks.center.values

    return h_rhs


def h_nonlinear_rhs(uh_flux, vh_flux):
    """
    Eq:
        ∂h/∂t + ∂/∂x((H+h)u) + ∂/∂y((H+h)v) = 0
    """
    
    # calculate RHS terms
    dhu_dx: Float[Array, "Nx Ny"] = difference(uh_flux, step_size=u_domain.dx[0], axis=0, derivative=1)
    dhv_dy: Float[Array, "Nx Ny"] = difference(vh_flux, step_size=v_domain.dx[1], axis=1, derivative=1)

    # calculate RHS
    h_rhs: Float[Array, "Nx Ny"] = - (dhu_dx + dhv_dy)

    # apply masks
    h_rhs *= masks.center.values
    
    return h_rhs


# ================================
# ZONAL VELOCITY, u
# ================================
def u_linear_rhs(h, v):
    """
    Eq:
        ∂u/∂t = fv - g ∂h/∂x
    """
    # pad arrays
    h_pad: Float[Array, "Nx+2 Ny"] = jnp.pad(h, pad_width=((1,1),(0,0)), mode="edge")
    v_pad: Float[Array, "Nx+2 Ny+1"] = jnp.pad(v, pad_width=((1,1),(0,0)), mode="constant")
    
    # calculate RHS terms
    v_avg: Float[Array, "Nx+1 Ny"] = center_avg_2D(v_pad)
    dh_dx: Float[Array, "Nx+1 Ny"] = difference(h_pad, step_size=h_domain.dx[0], axis=0, derivative=1)


    # calculate RHS
    u_rhs: Float[Array, "Nx+1 Ny"] = coriolis_f * v_avg - gravity * dh_dx

    # apply masks
    u_rhs *= masks.face_u.values

    return u_rhs

def u_nonlinear_rhs(h, q, vh_flux, ke):
    """
    Eq:
        work = g ∂h/∂x
        ke = 0.5 (u² + v²)
        ∂u/∂t = qhv - work - ke
        
    Notes:
        - uses reconstruction (5pt, improved weno) of q on vh flux
    """
    
    # pad arrays
    h_pad: Float[Array, "Nx+2 Ny"] = jnp.pad(h, pad_width=((1,1),(0,0)), mode="edge")
    ke_pad: Float[Array, "Nx+2 Ny"] = jnp.pad(ke, pad_width=((1,1),(0,0)), mode="edge")
    
    vh_flux_on_u: Float[Array, "Nx-1 Ny"] = center_avg_2D(vh_flux)
    
    
    qhv_flux_on_u: Float[Array, "Nx-1 Ny-2"] = reconstruct(
        q=q[1:-1,1:-1], u=vh_flux_on_u[:, 1:-1], u_mask=masks.face_u[1:-1,1:-1],
        dim=1, method=method, num_pts=num_pts
    )
    
    qhv_flux_on_u: Float[Array, "Nx+1 Ny"] = jnp.pad(qhv_flux_on_u, pad_width=((1,1),(1,1)), mode="constant")
    
    # apply mask
    qhv_flux_on_u *= masks.face_u.values
    
    # calculate work
    dh_dx: Float[Array, "Nx+1 Ny"] = difference(h_pad, step_size=h_domain.dx[0], axis=0, derivative=1)
    work = gravity * dh_dx
    
    # calculate kinetic energy
    dke_on_u: Float[Array, "Nx+1 Ny"] = difference(ke_pad, step_size=h_domain.dx[0], axis=0, derivative=1)
    
    # calculate u RHS
    u_rhs: Float[Array, "Nx+1 Ny"] = - work + qhv_flux_on_u - dke_on_u
    
    # apply mask
    u_rhs *= masks.face_u.values

    return u_rhs


# ================================
# MERIDIONAL VELOCITY, v
# ================================

def v_linear_rhs(h, u):
    """
    Eq:
        ∂v/∂t = - fu - g ∂h/∂y
    """
    # pad arrays
    h_pad: Float[Array, "Nx Ny+2"] = jnp.pad(h, pad_width=((0,0),(1,1)), mode="edge")
    u_pad: Float[Array, "Nx+1 Ny+2"] = jnp.pad(u, pad_width=((0,0),(1,1)), mode="constant")
    
    # calculate RHS terms
    u_avg: Float[Array, "Nx Ny+1"] = center_avg_2D(u_pad)
    dh_dy: Float[Array, "Nx Ny+1"] = difference(h_pad, step_size=h_domain.dx[1], axis=1, derivative=1)

    # calculate RHS
    v_rhs: Float[Array, "Nx Ny+1"] = - coriolis_f * u_avg - gravity * dh_dy

    # apply masks
    v_rhs *= masks.face_v.values

    return v_rhs

def v_nonlinear_rhs(h, q, uh_flux, ke):
    """
    Eq:
        work = g ∂h/∂y
        ke = 0.5 (u² + v²)
        ∂v/∂t = - qhu - work - ke
        
    Notes:
        - uses reconstruction (5pt, improved weno) of q on uh flux
    """
    h_pad: Float[Array, "Nx Ny+2"] = jnp.pad(h, pad_width=((0,0),(1,1),), mode="edge")
    ke_pad: Float[Array, "Nx Ny+2"] = jnp.pad(ke, pad_width=((0,0),(1,1),), mode="edge")
    
    uh_flux_on_v: Float[Array, "Nx Ny-1"] = center_avg_2D(uh_flux)
    
    
    qhu_flux_on_v: Float[Array, "Nx-2 Ny-1"] = reconstruct(
        q=q[1:-1,1:-1], u=uh_flux_on_v[1:-1], u_mask=masks.face_v[1:-1,1:-1],
        dim=0, method=method, num_pts=num_pts
    )
    
    qhu_flux_on_v: Float[Array, "Nx Ny+1"] = jnp.pad(qhu_flux_on_v, pad_width=((1,1),(1,1)), mode="constant")
    
    qhu_flux_on_v *= masks.face_v.values
    
    # calculate work
    dh_dy: Float[Array, "Nx Ny+1"] = difference(h_pad, step_size=h_domain.dx[1], axis=1, derivative=1)
    work = gravity * dh_dy
    
    # calculate kinetic energy
    dke_on_v: Float[Array, "Nx Ny+1"] = difference(ke_pad, step_size=h_domain.dx[1], axis=1, derivative=1)
    
    # calculate u RHS
    v_rhs: Float[Array, "Nx Ny+1"] = - work - qhu_flux_on_v - dke_on_v
    
    # apply masks
    v_rhs *= masks.face_v.values

    return v_rhs

# vector field
def equation_of_motion(h, u, v):
    if not linear_mass or not linear_momentum:
        uh_flux, vh_flux = calculate_uvh_flux(h=h, u=u, v=v)
        uh_flux = enforce_boundaries(uh_flux, "u")
        vh_flux = enforce_boundaries(vh_flux, "v")


    # mass equation

    if linear_mass:
        h_rhs = h_linear_rhs(u=u, v=v)
    else:
        h_rhs = h_nonlinear_rhs(uh_flux=uh_flux, vh_flux=vh_flux)


    # momentum equations

    if linear_momentum:
        u_rhs = u_linear_rhs(h=h, v=v)
        v_rhs = v_linear_rhs(h=h, u=u)
    else:
        ke = kinetic_energy(u=u, v=v)
        q = potential_vorticity(h=h, u=u, v=v)
        u_rhs = u_nonlinear_rhs(h=h, q=q, vh_flux=vh_flux, ke=ke)
        v_rhs = v_nonlinear_rhs(h=h, q=q, uh_flux=uh_flux, ke=ke)
        
    return h_rhs, u_rhs, v_rhs


equation_of_motion_jitted = jax.jit(equation_of_motion)

def iterate_shallow_water():
    # allocate arrays
    u, v, h = jnp.empty_like(u0), jnp.empty_like(v0), jnp.empty_like(h0)

    # initial conditions
    h: Float[Array, "Nx Ny"] = h.at[:].set(h0)
    u: Float[Array, "Nx+1 Ny"] = u.at[:].set(u0)
    v: Float[Array, "Nx Ny+1"] = v.at[:].set(v0)

    # apply masks
    h = enforce_boundaries(h, "h")
    u = enforce_boundaries(u, "u")
    v = enforce_boundaries(v, "v")


    first_step = True

    # time step equations
    while True:        
        # ==================
        # SPATIAL OPERATIONS
        # ==================
        h_rhs, u_rhs, v_rhs = equation_of_motion_jitted(h, u, v)
        
        


        # ==================
        # TIME STEPPING
        # ==================
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
                adams_bashforth_a * h_rhs
                + adams_bashforth_b * h_rhs_old
            )
        #
        h = enforce_boundaries(h, 'h')
        u = enforce_boundaries(u, 'u')
        v = enforce_boundaries(v, 'v')

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
            update_plot(t, h, u_on_h, v_on_h, ax)

        # stop if user closes plot window
        if not plt.fignum_exists(fig.number):
            break