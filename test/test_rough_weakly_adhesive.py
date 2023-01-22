from SurfaceTopography import Topography
from SurfaceTopography.Generation import fourier_synthesis
from Adhesion.Interactions import Exponential
from ContactMechanics.Systems import NonSmoothContactSystem
from ContactMechanics import PeriodicFFTElasticHalfSpace
from Adhesion.System import make_system, BoundedSmoothContactSystem
from NuMPI.Optimization.ccg_without_restart import constrained_conjugate_gradients
# from NuMPI import MPI
from NuMPI.Tools import Reduction
import pytest
from NuMPI import MPI

import numpy as np

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="tests only serial funcionalities, "
                                       "please execute with pytest")


def test_ccg_without_restart_weakly_adhesive(comm, verbose=False):
    Es = 1.
    gtol = 1e-8
    n = 512
    dx = 1.
    s = n

    hprms = 0.1

    np.random.seed(0)
    full_topography = fourier_synthesis((n, n), (s, s), 0.8, rms_slope=hprms,
                                        long_cutoff=s / 2, short_cutoff=32, )

    full_topography._heights = full_topography.heights() - np.max(full_topography.heights())

    pnp = Reduction(comm)

    # interaction parameters
    elastocapillary_length = 0.005 * dx
    w = Es * elastocapillary_length
    # Interaction range as in P&R PNAS
    # with some modifications
    rho = dx / 4
    # P&R keep the interaction parameters and vary h'rms,
    # We keep h'rms constant and hence need to vary the interaction parameters
    # accordingly

    interaction = Exponential(w, rho)

    substrate = PeriodicFFTElasticHalfSpace(
        nb_grid_pts=(n, n),
        physical_sizes=(n, n),
        young=Es,
        fft="mpi",
        communicator=comm)

    topography = Topography(
        full_topography.heights(),
        physical_sizes=(n, n),
        decomposition="domain",
        nb_subdomain_grid_pts=substrate.nb_subdomain_grid_pts,
        subdomain_locations=substrate.subdomain_locations,
        communicator=comm,)

    system = make_system(interaction=interaction, surface=topography,
                         substrate=substrate,
                         system_class=BoundedSmoothContactSystem,
                         communicator=comm)
    substrate = system.substrate

    if verbose:
        print('Some informations on the system')

        process_zone_size = Es * w / np.pi / abs(interaction.max_tensile) ** 2

        print("nb pixels in process zone: {}".format(process_zone_size / dx))
        # print("highcut wavelength / process_zone size: "
        #       "{}".format(short_cutoff / process_zone_size))
        print("rms_height / interaction_length: "
              "{}".format(topography.rms_height() / rho))

        print("Persson and Tosatti adhesion criterion: ")
        print("elastic deformation energy for full contact / work of "
              "adhesion: {}".format(substrate.evaluate(
                topography.detrend().heights())[0] / w / np.prod(
                topography.physical_sizes)))  # ^
        #            sets the mean to zero, so         |
        #            that the energy of the q = 0   --
        #            mode is not taken into
        #            account

        # P&R equation 10, with kappa_rep = 2, d_rep=4 h'rms / h''rms
        # (Only valid in the DMT regime, i.e. until the onset of stickiness)
        # If this value is smaller then 1 the surfaces are sticky and
        # normal CG's might not work anymore
        print("P&R stickiness criterion: {}".format(topography.rms_gradient() *
                                                    Es * rho / 2 / w *
              (topography.rms_gradient() ** 2 / topography.rms_curvature() / rho)
                                                    ** (2 / 3)))
        R = 1 / topography.rms_curvature()
        print("Generalized Tabor parameter à la Martin Müser")
        print("{}".format(R**(1/3) * elastocapillary_length ** (2 / 3) / rho))

    print(""" ########## PENETRATION CONTROLLED ################## """)

    penetration = 0.1

    init_disp = np.zeros(substrate.nb_subdomain_grid_pts)
    init_gap = init_disp - topography.heights() - penetration
    init_gap[init_gap < 0] = 0

    res = constrained_conjugate_gradients(
        system.primal_objective(penetration, gradient=True),
        system.primal_hessian_product,
        x0=init_gap, mean_val=None,
        gtol=gtol * max(Es * hprms, abs(
                interaction.max_tensile)) * topography.area_per_pt,
        maxiter=1000)

    print(res.nit)

    assert res.success, res.message
    gap = res.x

    print(""" ########## mean_gap controlled ################# """)

    mean_gap = pnp.sum(gap) / np.prod(substrate.nb_domain_grid_pts)

    # typical initial guess
    init_disp = np.zeros(substrate.nb_subdomain_grid_pts)
    penetration = pnp.sum(init_disp - topography.heights() - mean_gap) / \
        np.prod(substrate.nb_domain_grid_pts)

    print(penetration)

    init_gap = init_disp - topography.heights() - penetration
    init_gap[init_gap < 0] = 0

    res = constrained_conjugate_gradients(
        system.primal_objective(penetration, gradient=True),
        system.primal_hessian_product,
        x0=init_gap, mean_val=mean_gap,
        gtol=gtol * max(Es * hprms, abs(
                interaction.max_tensile)) * topography.area_per_pt,
        maxiter=1000)

    assert res.success, res.message
    gap = res.x
    print(res.nit)

    print(""" #########   NONADHESIVE, same mean gap ############# """)
    nonadh_system = NonSmoothContactSystem(substrate=substrate,
                                           surface=topography)

    res = constrained_conjugate_gradients(
        nonadh_system.primal_objective(penetration, gradient=True),
        nonadh_system.primal_hessian_product,
        x0=init_gap, mean_val=mean_gap,
        gtol=gtol * max(Es * hprms, abs(
                interaction.max_tensile)) * topography.area_per_pt,
        maxiter=1000)

    assert res.success, res.message
    gap_nonadh = res.x
    print(res.nit)

    if verbose and comm.Get_size() == 1.:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        fig, ax = plt.subplots()

        contactcmap = LinearSegmentedColormap.from_list('contactcmap',
                                                        ((1, 1, 1, 0.),
                                                         (1, 0, 0, 1)),
                                                        N=256)
        ax.imshow(gap.reshape(substrate.nb_grid_pts) == 0, cmap=contactcmap)

        contactcmap = LinearSegmentedColormap.from_list('contactcmap',
                                                        ((1, 1, 1, 0.),
                                                         (0, 1, 0, 1)),
                                                        N=256)
        ax.imshow(gap_nonadh.reshape(substrate.nb_grid_pts) == 0,
                  cmap=contactcmap)
