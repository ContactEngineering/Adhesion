from SurfaceTopography import make_sphere
from SurfaceTopography.Generation import fourier_synthesis
import Adhesion.Interactions as Inter
from Adhesion.System import SmoothContactSystem
import ContactMechanics as Solid
import numpy as np





nx = 32

sx = sy = 2
R = 10.
Es = 50.

interaction = Inter.RepulsiveExponential(0, 0.5, 0, 1.)

substrate = Solid.PeriodicFFTElasticHalfSpace((nx,), young=Es,
                                              physical_sizes=(sx,))

topography = make_sphere(R, (nx,), (sx,), kind="paraboloid")

system = SmoothContactSystem(substrate=substrate, surface=topography,
                             interaction=interaction)


def check_fun_grad_consistency(fun,
        x0 , dx=None,
        hs=np.array([1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5,1e-6, 1e-7])):
    """
    Tests the consistency between the function and its gradient values.

    Parameters
    _________

    fun:  callable function returning fun and gradient at x.
    x:  value to test at.
    """

    obj = fun

    x = x0
    if dx is None: 
        dx = 1 * (0.5 + np.random.random(size=(x0.shape)))
    dx *= np.linalg.norm(dx) # make it a unit vector 

    en, grad = obj(x)  

    taylor = []

    for h in hs:
        _en, _grad = obj(x + h * dx)
        _taylor = _en - en - np.sum(grad * h * dx)
        _taylor = _taylor/h**2
        if not taylor :
            _max_taylor = _taylor
            lower_bnd = _max_taylor/10
            upper_bnd = _max_taylor*10
        taylor.append(_taylor)
        np.testing.assert_array_less(lower_bnd,_taylor,err_msg='lower bound not met.')
        np.testing.assert_array_less(_taylor,upper_bnd,err_msg='upper bound not met.')

    if True :
        # Visualize the quadratic convergence of the taylor expansion
        # What to expect:
        # Taylor expansion: g(x + h ∆x) - g(x) = Hessian * h * ∆x + O(h^2)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(hs, taylor, "+-")
        ax.set_xscale("log")
        ax.set_xlabel('hs')
        ax.set_yscale("log")
        ax.set_ylabel('taylor diff')
        ax.grid(True)
        plt.show()


obj_float = system.objective_k_float(0, True, True)
obj_real = system.objective(0, True, True)

x = np.random.uniform(size=nx) 
# dx = np.zeros(nx)
# dx[8]=1

check_fun_grad_consistency(obj_real, x)
check_fun_grad_consistency(obj_float, x)
