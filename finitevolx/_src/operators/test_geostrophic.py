from finitevolx._src.operators.operators import difference
from finitevolx._src.operators.geostrophic import  geostrophic_gradient, divergence, relative_vorticity
import numpy as np

rng = np.random.RandomState(123)

def test_geostrophic_gradient():
    psi = rng.randn(50,25)
    dx, dy = 0.1, 0.2

    # gradient froms scratch
    u_ = -difference(psi, axis=1, step_size=dy, derivative=1)
    v_ = difference(psi, axis=0, step_size=dx, derivative=1)

    # convenience function
    u, v = geostrophic_gradient(psi, dx=dx, dy=dy)

    np.testing.assert_array_almost_equal(u_, u)
    np.testing.assert_array_almost_equal(v_, v)

def test_divergence():
    u = rng.randn(50,25)
    v = rng.randn(49,26)

    dx, dy = 0.1, 0.2

    # gradient froms scratch
    du_dx_ = difference(u, axis=0, step_size=dx, derivative=1)
    dv_dy_ = difference(v, axis=1, step_size=dy, derivative=1)
    div_ = du_dx_ + dv_dy_
    # convenience function
    div = divergence(u, v, dx=dx, dy=dy)

    np.testing.assert_array_almost_equal(div_, div)


def test_relative_vorticity():
    u = rng.randn(50,26)
    v = rng.randn(51,25)

    dx, dy = 0.1, 0.2

    # gradient froms scratch
    du_dy_ = difference(u, axis=1, step_size=dy, derivative=1)
    dv_dx_ = difference(v, axis=0, step_size=dx, derivative=1)
    vort_r_ = dv_dx_ - du_dy_
    # convenience function
    vort_r = relative_vorticity(u, v, dx=dx, dy=dy)

    np.testing.assert_array_almost_equal(vort_r_, vort_r)

