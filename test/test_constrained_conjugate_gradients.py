from SurfaceTopography import make_sphere
import ContactMechanics as Solid
from Adhesion.Interactions import Exponential
from Adhesion.System import BoundedSmoothContactSystem
from NuMPI.Optimization import generic_cg_polonsky, bugnicourt_cg
import numpy as np
import scipy.optimize as optim
import pytest


@pytest.mark.parametrize("offset", [0, 1.])
@pytest.mark.parametrize('_solver', [  # 'generic_cg_polonsky',
    'bugnicourt_cg'])
def test_primal_obj(_solver, offset):
    nx, ny = 256, 256
    ## FIXED by the nondimensionalisation
    maugis_K = 1.
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

    # #####################LBFGSB#####################################
    res = optim.minimize(system.primal_objective(offset, gradient=True),
                         init_gap,
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=gtol, ftol=0))

    assert res.success
    _lbfgsb = res.x.reshape((nx, ny))

    if _solver == 'generic_cg_polonsky':
        # ####################POLONSKY-KEER##############################
        res = generic_cg_polonsky.min_cg(
            system.primal_objective(offset, gradient=True),
            system.primal_hessian_product,
            init_gap, polonskykeer=True, gtol=gtol)

        assert res.success
        polonsky = res.x.reshape((nx, ny))

        # ####################BUGNICOURT###################################
        res = generic_cg_polonsky.min_cg(
            system.primal_objective(offset, gradient=True),
            system.primal_hessian_product,
            init_gap, bugnicourt=True, gtol=gtol)
        assert res.success

        bugnicourt = res.x.reshape((nx, ny))

        np.testing.assert_allclose(polonsky, bugnicourt, atol=1e-3)
        np.testing.assert_allclose(polonsky, _lbfgsb, atol=1e-3)
        np.testing.assert_allclose(_lbfgsb, bugnicourt, atol=1e-3)

        # ##########TEST MEAN VALUES#######################################
        mean_val = np.mean(_lbfgsb)
        # disp = _lbfgsb
        # ####################POLONSKY-KEER##############################
        res = generic_cg_polonsky.min_cg(
            system.primal_objective(offset, gradient=True),
            system.primal_hessian_product,
            init_gap,
            polonskykeer=True,
            mean_value=mean_val,
            gtol=gtol,
            maxiter=500
            )

        assert res.success
        polonsky_mean = res.x.reshape((nx, ny))

        # ####################BUGNICOURT###################################
        res = generic_cg_polonsky.min_cg(
            system.primal_objective(offset, gradient=True),
            system.primal_hessian_product,
            init_gap, bugnicourt=True, mean_value=mean_val, gtol=gtol)
        assert res.success

        bugnicourt_mean = res.x.reshape((nx, ny))

        np.testing.assert_allclose(polonsky_mean, _lbfgsb, atol=1e-3)
        np.testing.assert_allclose(bugnicourt_mean, _lbfgsb, atol=1e-3)

    elif _solver == 'bugnicourt_cg':
        bugnicourt_cg.constrained_conjugate_gradients(system.primal_objective
                                                      (offset, gradient=True),
                                                      system.
                                                      primal_hessian_product,
                                                      disp0=init_gap,
                                                      mean_val=None, gtol=gtol)

        bugnicourt = res.x.reshape((nx, ny))

        np.testing.assert_allclose(_lbfgsb, bugnicourt, atol=1e-3)

        mean_val = np.mean(_lbfgsb)

        bugnicourt_cg.constrained_conjugate_gradients(system.primal_objective
                                                      (offset, gradient=True),
                                                      system.primal_hessian_product,
                                                      disp0=init_gap,
                                                      mean_val=mean_val,
                                                      gtol=gtol
                                                      )

        bugnicourt_mean = res.x.reshape((nx, ny))

        np.testing.assert_allclose(_lbfgsb, bugnicourt_mean, atol=1e-3)
    else:
        assert False
