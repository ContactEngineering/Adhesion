# This aims to make a strong test for CGs using the simulation in figure 3 of
# the Bugnicourt paper for approx 20% contact area.
# Note that they use a nonadhesive simulation result to initialize the fields
# which is not done in this test.
# Bugnicourt, R., Sainsot, P., Dureisseix, D. et al. FFT-Based Methods for
# Solving a Rough Adhesive Contact Tribol Lett 66, 29 (2018).
# https://doi.org/10.1007/s11249-017-0980-z
import pytest
from SurfaceTopography.Generation import fourier_synthesis
from matplotlib import pyplot as plt

from Adhesion.Interactions import Exponential
from Adhesion.System import make_system, BoundedSmoothContactSystem
import numpy as np
from NuMPI.Optimization import ccg_without_restart
from NuMPI import MPI
import scipy.optimize as optim

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="tests only serial funcionalities, "
                                       "please execute with pytest")


def test_bug_bench():
    # Remark:
    #
    # This test fails right now.
    # But my guess is that the contact is multistable and LBFGS-B and Bugnicourt sample different metastable states.
    #
    # For lower adhesion the two algorithms are in good agreement.

    np.random.seed(0)
    # nondim we use
    Es = 1.
    # rms_curvature = 1.
    w = 1.
    n = 2048

    # Rc ** 2/3 (w / Es)**(1/3)
    lateral_unit = (60e-9 ** (2 / 3) * (5e-3 / 25e6) ** (1 / 3))

    s = 4e-6 / lateral_unit

    # dx = s / n

    long_cutoff = s * 1 / 4
    short_cutoff = s * 0.1 / 4

    topo = fourier_synthesis(nb_grid_pts=(n, n),
                             hurst=0.8,
                             physical_sizes=(s, s),
                             rms_slope=1,
                             long_cutoff=long_cutoff,
                             short_cutoff=short_cutoff,
                             )

    topo = topo.scale(1 / topo.rms_curvature_from_area()).squeeze()

    # Adhesion parameters:

    tabor = 3.0
    interaction_length = 1 / tabor  # original was 1/tabor.

    interaction = Exponential(w, interaction_length)
    # process_zone_size = Es * w / np.pi / abs(interaction.max_tensile) ** 2

    system = make_system(interaction=interaction,
                         surface=topo,
                         substrate='periodic',
                         young=Es,
                         system_class=BoundedSmoothContactSystem)

    print("max height {}".format(np.max(system.surface.heights())))

    gtol = 1e-4

    disp0 = np.zeros(topo.nb_grid_pts)

    penetration = 30.0

    init_gap = disp0 - system.surface.heights() - penetration

    lbounds = np.zeros((n, n))
    bnds_gap = system._reshape_bounds(lbounds, )

    # lbounds = system._lbounds_from_heights(penetration)
    # bnds_disp = system._reshape_bounds(lbounds, )

    sol = optim.minimize(
        system.primal_objective(penetration, gradient=True),
        system.shape_minimisation_input(init_gap),
        method='L-BFGS-B', jac=True, bounds=bnds_gap,
        options=dict(gtol=gtol * max(Es * topo.rms_gradient(), abs(interaction.max_tensile)) * topo.area_per_pt,
                     ftol=0,
                     maxcor=3),
        callback=None)

    print(sol.message, sol.nfev)
    assert sol.success, sol.message
    # sol_lbfgs = sol
    mask = sol.x == 0
    area_lbfgs = mask.sum()
    print('lbfgsb frac ar. {}'.format(area_lbfgs / n ** 2))
    sol_lbfgs = sol.x
    # iter_lbfgsb = sol.nfev
    # mean_val_lbfgs = np.mean(sol.x)

    res = ccg_without_restart.constrained_conjugate_gradients(
        system.primal_objective(penetration, gradient=True),
        system.primal_hessian_product,
        x0=init_gap, mean_val=None,
        gtol=gtol * max(Es * topo.rms_gradient(), abs(interaction.max_tensile)) * topo.area_per_pt,
        maxiter=1000)

    print(res.message, res.nit)
    assert res.success
    sol_bug = res.x
    mask = sol_bug == 0
    area_cg = mask.sum()
    print('cg frac ar. {}'.format(area_cg / n ** 2))

    print('min {} and max {} lbfgs'.format(np.min(sol_lbfgs), np.max(sol_lbfgs)))
    print('min {} and max {} bug'.format(np.min(sol_bug), np.max(sol_bug)))

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    levels = np.linspace(0, np.max(sol_lbfgs), 5)
    plt.colorbar(ax.contour(sol_lbfgs.reshape((n, n)), linewidths=0.1, levels=levels))
    ax.contour(sol_bug.reshape((n, n)), linestyles="dashed", linewidths=0.1, levels=levels)

    fig.savefig("gaps.pdf")

    assert area_cg == area_lbfgs

    np.testing.assert_allclose(sol_lbfgs, sol_bug, atol=1e-2)
