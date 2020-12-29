from SurfaceTopography import make_sphere
import ContactMechanics as Solid
from SurfaceTopography.Generation import fourier_synthesis

from Adhesion.Interactions import Exponential
from Adhesion.System import BoundedSmoothContactSystem
from NuMPI.Optimization import generic_cg_polonsky, bugnicourt_cg
import numpy as np
import scipy.optimize as optim
import pytest
from NuMPI import MPI

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="tests only serial funcionalities,"
                                       " please execute with pytest")


@pytest.mark.parametrize("offset", [0, 1.])
@pytest.mark.parametrize('_solver', [  # 'generic_cg_polonsky',
    'bugnicourt_cg'])
def test_primal_obj(_solver, offset):
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
                                                      x0=init_gap,
                                                      mean_val=None, gtol=gtol)

        bugnicourt = res.x.reshape((nx, ny))

        np.testing.assert_allclose(_lbfgsb, bugnicourt, atol=1e-3)

        mean_val = np.mean(_lbfgsb)

        bugnicourt_cg.constrained_conjugate_gradients(
            system.primal_objective(offset, gradient=True),
            system.primal_hessian_product,
            x0=init_gap,
            mean_val=mean_val,
            gtol=gtol
            )

        bugnicourt_mean = res.x.reshape((nx, ny))

        np.testing.assert_allclose(_lbfgsb, bugnicourt_mean, atol=1e-3)
    else:
        assert False


def test_force_computation_mean_gap_constrained():
    pnp = np
    nx, ny = 32, 32
    # FIXED by the nondimensionalisation
    # maugis_K = 1.
    Es = 3 / 4  # maugis K = 1.
    w = 1 / np.pi * .001
    R = 1.

    dx = 0.15
    rho = 2.

    interaction = Exponential(w, rho)

    sx = sy = dx * nx

    gtol = 1e-11

    topography = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
    substrate = Solid.PeriodicFFTElasticHalfSpace((nx, ny), young=Es,
                                                  physical_sizes=(sx, sy))

    system = BoundedSmoothContactSystem(substrate, interaction, topography)

    penetration = 0.2

    sol = system.minimize_proxy(
        offset=penetration,
        options=dict(gtol=gtol, ftol=0),
        lbounds="auto"
        )

    assert sol.success
    print("{}".format(sol.message))

    print("lbfgs nit: {}".format(sol.nit))
    disp = sol.x

    gap_lbfgs = system.compute_gap(disp, penetration)

    mean_gap = np.sum(gap_lbfgs) / np.prod(substrate.nb_domain_grid_pts)

    forces_lbfgs = - substrate.evaluate_force(disp)

    print("max abs force {}".format(np.max(abs(forces_lbfgs))))

    #  Check that at least the not mean constrained way works
    init_disp = np.zeros((nx, ny))
    init_gap = init_disp - topography.heights() - penetration
    init_gap[init_gap < 0] = 0

    res = bugnicourt_cg.constrained_conjugate_gradients(
        system.primal_objective(penetration, gradient=True),
        system.primal_hessian_product,
        x0=init_gap,
        gtol=gtol,
        maxiter=1000,
        )

    assert res.success
    print("bugnicourt nit: {}".format(res.nit))
    gap = res.x.reshape(substrate.nb_subdomain_grid_pts)

    grad = system.primal_objective(penetration, gradient=True)(
        gap)[1].reshape(substrate.nb_subdomain_grid_pts)

    assert np.max(abs(grad * (gap > 0))) < gtol

    np.testing.assert_allclose(gap, gap_lbfgs,
                               atol=1e-6 * abs(topography.min()))

    ##############################
    # typical initial guess

    init_disp = np.zeros(substrate.nb_subdomain_grid_pts)
    _penetration = np.sum(
        init_disp - topography.heights() - mean_gap) / np.prod(
        substrate.nb_domain_grid_pts)

    print(_penetration)

    init_gap = init_disp - topography.heights() - _penetration
    init_gap[init_gap < 0] = 0

    res = bugnicourt_cg.constrained_conjugate_gradients(
        system.primal_objective(_penetration, gradient=True),
        system.primal_hessian_product,
        x0=init_gap, mean_val=mean_gap,
        gtol=gtol,
        maxiter=1000,
        )

    assert res.success
    print("bugnicourt nit: {}".format(res.nit))
    gap = res.x.reshape(substrate.nb_subdomain_grid_pts)

    np.testing.assert_allclose(gap, gap_lbfgs,
                               atol=1e-6 * abs(topography.min()))

    grad = system.primal_objective(0, gradient=True)(
        gap)[1].reshape(substrate.nb_subdomain_grid_pts)

    nc_points = gap > 0
    nb_nc_points = pnp.sum(np.count_nonzero(nc_points))
    lagrange_mean_gap = pnp.sum(grad[nc_points]) / nb_nc_points

    forces = - substrate.evaluate_force(gap + topography.heights()) \
             - lagrange_mean_gap

    np.testing.assert_allclose(forces, forces_lbfgs, atol=10 * gtol)


def test_mean_value_mode_is_penetration_indepentent():
    nx, ny = 128, 128
    # FIXED by the nondimensionalisation
    # maugis_K = 1.
    Es = 1  # maugis K = 1.
    w = .0001

    dx = 1
    rho = 1

    rms_slope = 0.1

    interaction = Exponential(w, rho)

    sx = sy = dx * nx

    gtol = 1e-11
    np.random.seed(0)
    topography = fourier_synthesis(
        (nx, ny), (sx, sy),
        0.8,
        rms_slope=rms_slope,
        long_cutoff=sx / 2,
        short_cutoff=4 * dx,
        )
    topography ._heights = topography .heights() - np.max(
        topography .heights())
    substrate = Solid.PeriodicFFTElasticHalfSpace((nx, ny), young=Es,
                                                  physical_sizes=(sx, sy))

    system = BoundedSmoothContactSystem(substrate, interaction, topography)

    mean_gap = - np.mean(topography.heights()) - 0.7

    ##############################
    # typical initial guess

    init_disp = np.zeros(substrate.nb_subdomain_grid_pts)
    _penetration = np.sum(
        init_disp - topography.heights() - mean_gap) / np.prod(
        substrate.nb_domain_grid_pts)

    print(_penetration)

    init_gap = init_disp - topography.heights() - _penetration
    init_gap[init_gap < 0] = 0

    ca = []
    arbitrary_penetration = 0
    res = bugnicourt_cg.constrained_conjugate_gradients(
        system.primal_objective(arbitrary_penetration, gradient=True),
        system.primal_hessian_product,
        x0=init_gap, mean_val=mean_gap,
        gtol=gtol,
        maxiter=1000,
        callback=lambda x: ca.append(np.count_nonzero(x == 0))
        )
    first_ca = ca[0]
    assert res.success

    ca = []
    arbitrary_penetration = 100
    res = bugnicourt_cg.constrained_conjugate_gradients(
        system.primal_objective(arbitrary_penetration, gradient=True),
        system.primal_hessian_product,
        x0=init_gap, mean_val=mean_gap,
        gtol=gtol,
        maxiter=1000,
        callback=lambda x: ca.append(np.count_nonzero(x == 0))
        )
    assert ca[0] == first_ca
    assert res.success
    print("bugnicourt nit: {}".format(res.nit))

    ca = []
    arbitrary_penetration = -100
    res = bugnicourt_cg.constrained_conjugate_gradients(
        system.primal_objective(arbitrary_penetration, gradient=True),
        system.primal_hessian_product,
        x0=init_gap, mean_val=mean_gap,
        gtol=gtol,
        maxiter=1000,
        callback=lambda x: ca.append(np.count_nonzero(x == 0))
        )
    assert ca[0] == first_ca
    assert res.success
    print("bugnicourt nit: {}".format(res.nit))

def test_bugnicourt_free_system():
    pnp = np
    # TODO: there is an old bug in the nonsmoothcontactsystem objective
    nx, ny = 8, 8
    sx = sy = 4.
    R = 1.
    Es = 0.75

    w = 1 / np.pi * 1e-2
    rho = 6.

    surface = make_sphere(R, (nx, ny), (sx, sy),
        centre  = (sx / 2, sy / 2),
        kind="paraboloid")
    padded_surface = make_sphere(R, (2 * nx, 2* ny), (2* sx, 2 * sy),
        centre = (sx / 2, sy / 2),
        kind="paraboloid")

    interaction = Exponential(w, rho)


    substrate = Solid.FreeFFTElasticHalfSpace((nx, ny), young=Es,
                                               physical_sizes=(sx, sy))

    system = BoundedSmoothContactSystem(substrate, interaction, surface)

    penetration = 0.005
    lbounds = system._lbounds_from_heights(penetration)

    bnds = system._reshape_bounds(lbounds, )
    init_disp = np.zeros(substrate.nb_subdomain_grid_pts) # .flatten()

    bounded =  init_disp<lbounds
    init_disp[bounded] == lbounds[bounded]

    res = optim.minimize(system.objective(penetration, gradient=True),
                         init_disp,
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=1e-13, ftol=1e-20))

    assert res.success
    _lbfgsb = res.x.reshape((2 * nx, 2 * ny))
    _lbfgs_jac = res.jac.reshape((2 * nx, 2 * ny))

    res = bugnicourt_cg.constrained_conjugate_gradients(
        system.objective(penetration, gradient=True),
        system.hessian_product_function(penetration),
        init_disp.reshape(-1),
        gtol=1e-13,
        bounds=lbounds.filled().reshape(-1),
        maxiter=500
    )
    assert res.success

    print(res.nit)
    _bug = res.x.reshape((2 * nx, 2 * ny))
    _bug_jac = res.jac.reshape((2 * nx, 2 * ny))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(_bug[:, ny//2], label="bug")
    ax.plot(_lbfgsb[:, ny//2], label="lbfgsb")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(system.objective(penetration, gradient=True)(_bug)[1].reshape((2 * nx, 2 * ny))[:, ny//2], label="bug")
    ax.plot(system.objective(penetration, gradient=True)(_lbfgsb)[1].reshape((2 * nx, 2 * ny))[:, ny//2], label="lbfgsb")
    ax.legend()
    plt.show()

    assert pnp.max(abs(_bug-_lbfgsb)) < 1e-5


