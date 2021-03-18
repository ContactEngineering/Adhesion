from SurfaceTopography import make_sphere
import ContactMechanics as Solid
from Adhesion.Interactions import Exponential
from Adhesion.System import BoundedSmoothContactSystem
from NuMPI.Optimization import bugnicourt_cg, polonsky_keer
import numpy as np
import scipy.optimize as optim
import pytest


@pytest.mark.parametrize("offset", [0, 1.])
def test_primal_obj(offset):
    nx, ny = 256, 256
    # FIXED by the nondimensionalisation
    # maugis_K = 1.
    Es = 3 / 4  # maugis K = 1.
    w = 1 / np.pi
    R = 1.

    dx = 0.02
    rho = 0.2

    interaction = Exponential(w, rho)

    sx = sy = dx * nx

    gtol = 1e-5 * abs(interaction.max_tensile) * dx ** 2

    surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
    substrate = Solid.PeriodicFFTElasticHalfSpace((nx, ny), young=Es,
                                                  physical_sizes=(sx, sy))

    system = BoundedSmoothContactSystem(substrate, interaction, surface)

    lbounds = np.zeros((nx, ny))
    bnds = system._reshape_bounds(lbounds, )

    init_disp = np.zeros((nx, ny))
    init_gap = init_disp - surface.heights() - offset

    # ####################POLONSKY-KEER##############################
    res = polonsky_keer.constrained_conjugate_gradients(
        system.primal_objective(offset, gradient=True),
        system.primal_hessian_product, x0=init_gap, gtol=gtol)

    assert res.success
    polonsky_gap = res.x.reshape((nx, ny))

    # ####################BUGNICOURT###################################
    res = bugnicourt_cg.constrained_conjugate_gradients(
        system.primal_objective(offset, gradient=True),
        system.primal_hessian_product, x0=init_gap, mean_val=None, gtol=gtol)
    assert res.success

    bugnicourt_gap = res.x.reshape((nx, ny))

    # #####################LBFGSB#####################################
    res = optim.minimize(system.primal_objective(offset, gradient=True),
                         init_gap,
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=gtol, ftol=1e-20))

    assert res.success
    lbfgsb_gap = res.x.reshape((nx, ny))

    np.testing.assert_allclose(polonsky_gap, bugnicourt_gap, atol=1e-3)
    np.testing.assert_allclose(polonsky_gap, lbfgsb_gap, atol=1e-3)
    np.testing.assert_allclose(lbfgsb_gap, bugnicourt_gap, atol=1e-3)

    # ##########TEST MEAN VALUES#######################################
    mean_val = np.mean(lbfgsb_gap)
    # ####################POLONSKY-KEER##############################
    res = polonsky_keer.constrained_conjugate_gradients(
        system.primal_objective(offset, gradient=True),
        system.primal_hessian_product, init_gap, gtol=gtol,
        mean_value=mean_val)

    assert res.success
    polonsky_gap_mean_cons = res.x.reshape((nx, ny))

    # ####################BUGNICOURT###################################
    bugnicourt_cg.constrained_conjugate_gradients(system.primal_objective
                                                  (offset, gradient=True),
                                                  system.
                                                  primal_hessian_product,
                                                  x0=init_gap,
                                                  mean_val=mean_val,
                                                  gtol=gtol
                                                  )
    assert res.success

    bugnicourt_gap_mean_cons = res.x.reshape((nx, ny))

    np.testing.assert_allclose(polonsky_gap_mean_cons, lbfgsb_gap, atol=1e-3)
    np.testing.assert_allclose(bugnicourt_gap_mean_cons, lbfgsb_gap, atol=1e-3)
    np.testing.assert_allclose(lbfgsb_gap, bugnicourt_gap, atol=1e-3)
    np.testing.assert_allclose(lbfgsb_gap, bugnicourt_gap_mean_cons, atol=1e-3)
