"""

Runs the  primal (gap as DOFS) ccg_without_restart algorithm with fixed penetration
and with constrained mean gap. Compares the gaps with the old serial lbfgsb
code with displacements as variables

"""

from NuMPI.IO import load_npy
from SurfaceTopography.IO.NPY import NPYReader
from SurfaceTopography.Generation import fourier_synthesis
from Adhesion.Interactions import Exponential
from ContactMechanics.Systems import NonSmoothContactSystem
from ContactMechanics import PeriodicFFTElasticHalfSpace
from Adhesion.System import make_system, BoundedSmoothContactSystem
from NuMPI.Optimization.ccg_without_restart import constrained_conjugate_gradients
from NuMPI.Tools import Reduction

import os
import numpy as np
from NuMPI import MPI

_comm = MPI.COMM_WORLD
Es = 1.

gtol = 1e-8

n = 512
dx = 1.
s = n

rms_slope = 0.1

penetration = 0.1

# interaction parameters
elastocapillary_length = 0.005 * dx
w = Es * elastocapillary_length
# Interaction range as in P&R PNAS
# with some modifications
rho = dx / 4
# P&R keep the interaction parameters and vary h'rms,
# We keep h'rms constant and hence need to vary the interaction
# parameters accordingly

interaction = Exponential(w, rho)


def get_topography_file(_comm):
    """
    generates the topography for the tests
    """

    print("topography fixture")
    path = "topography.npy"
    if _comm.rank == 0:
        if not os.path.exists(path):
            np.random.seed(0)
            full_topography = fourier_synthesis((n, n), (s, s),
                                                0.8,
                                                rms_slope=rms_slope,
                                                long_cutoff=s / 2,
                                                short_cutoff=32,
                                                )
            full_topography._heights = full_topography.heights() - np.max(
                full_topography.heights())

            np.save(path, full_topography.heights())

    _comm.barrier()
    return path


def get_reference_data_file(_comm):
    """
    computes a reference solution using lbfgsb. Is serial, so cannot be done
    inside the parallel test
    """
    print("reference fixture")
    path = "reference_gap.npy"
    if _comm.rank == 0:
        if not os.path.exists(path):
            system = make_system(interaction=interaction,
                                 surface="topography.npy",
                                 substrate="periodic",
                                 physical_sizes=(n, n),
                                 young=Es, communicator=MPI.COMM_SELF,
                                 system_class=BoundedSmoothContactSystem
                                 )
            print("computing reference")
            sol = system.minimize_proxy(
                offset=penetration,
                lbounds="auto",
                options=dict(
                    ftol=0,
                    gtol=1e-5 * max(Es * rms_slope, abs(
                        interaction.max_tensile)) * system.surface.area_per_pt,
                    maxcor=3,
                    maxiter=1000,
                    )

                )

            assert sol.success
            gap = system.compute_gap(
                disp=sol.x,
                offset=penetration)
            np.save(path, gap)
            print("finished computing reference")
    _comm.barrier()

    return path


def test_ccg_without_restart_weakly_adhesive(comm, verbose=False):
    topography_file = get_topography_file(comm)
    reference_data_file = get_reference_data_file(comm)
    _penetration = penetration
    pnp = Reduction(comm)

    substrate = PeriodicFFTElasticHalfSpace(
        nb_grid_pts=(n, n),
        physical_sizes=(n, n),
        young=Es,
        fft="mpi",
        communicator=comm)

    # topography = Topography(
    #     full_topography.heights(),
    #     physical_sizes=(n, n),
    #     decomposition="domain",
    #     nb_subdomain_grid_pts=substrate.nb_subdomain_grid_pts,
    #     subdomain_locations=substrate.subdomain_locations,
    #     communicator=comm,
    #     )

    print("reading topography")
    topography = NPYReader(topography_file, communicator=comm).topography(
        physical_sizes=(n, n),
        nb_subdomain_grid_pts=substrate.nb_subdomain_grid_pts,
        subdomain_locations=substrate.subdomain_locations)

    tol = 1e-5 * topography.rms_height_from_area()
    forcetol = 10 * gtol * max(Es * rms_slope, abs(
        interaction.max_tensile)) * topography.area_per_pt
    system = make_system(interaction=interaction,
                         surface=topography,
                         substrate=substrate,
                         system_class=BoundedSmoothContactSystem,
                         communicator=comm)
    substrate = system.substrate

    if verbose:
        print("""
        Some informations on the system
        """
              )

        process_zone_size = Es * w / np.pi / abs(interaction.max_tensile) ** 2

        print("nb pixels in process zone: {}".format(process_zone_size / dx))
        # print("highcut wavelength / process_zone size: "
        #       "{}".format(short_cutoff / process_zone_size))
        print("rms_height / interaction_length: "
              "{}".format(topography.rms_height() / rho))

        print("Persson and Tosatti adhesion criterion: ")
        print(
            "elastic deformation energy for full contact / work of adhesion: "
            "{}".format(substrate.evaluate(topography.detrend().heights())[0]
                        / w / np.prod(topography.physical_sizes)))  # ^
        #            sets the mean to zero, so         |
        #            that the energy of the q = 0   --
        #            mode is not taken into
        #            account

        # P&R equation 10, with kappa_rep = 2, d_rep=4 h'rms / h''rms
        # (Only valid in the DMT regime, i.e. until the onset of stickiness)
        # If this value is smaller then 1 the surfaces are sticky and
        # normal CG's might not work anymore
        print("P&R stickiness criterion: "
              "{}".format(rms_slope * Es * rho / 2 / w *
                          (rms_slope ** 2 / topography.rms_curvature() / rho)
                          ** (2 / 3)))

        R = 1 / topography.rms_curvature()
        print("Generalized Tabor parameter à la Martin Müser")
        print("{}".format(
            R ** (1 / 3) * elastocapillary_length ** (2 / 3) / rho))

    reference_gap = load_npy(
        reference_data_file,
        subdomain_locations=substrate.subdomain_locations,
        nb_subdomain_grid_pts=substrate.nb_subdomain_grid_pts,
        comm=comm
        )

    print("""
    ########## PENETRATION CONTROLLED ##################
    """)

    init_disp = np.zeros(substrate.nb_subdomain_grid_pts)
    init_gap = init_disp - topography.heights() - _penetration
    init_gap[init_gap < 0] = 0

    res = constrained_conjugate_gradients(
        system.primal_objective(_penetration, gradient=True),
        system.primal_hessian_product,
        x0=init_gap, mean_val=None,
        gtol=gtol * max(Es * rms_slope, abs(
            interaction.max_tensile)) * topography.area_per_pt,
        maxiter=1000,
        communicator=comm,
        )

    print(res.nit)

    assert res.success, res.message
    gap = res.x.reshape(substrate.nb_subdomain_grid_pts)

    assert pnp.max(abs(gap - reference_gap)) < tol

    forces_ref = substrate.evaluate_force(
        gap + topography.heights() + _penetration)

    print("""
    ########## mean_gap controlled #################
    """)

    mean_gap = pnp.sum(gap) / np.prod(substrate.nb_domain_grid_pts)

    # typical initial guess

    init_disp = np.zeros(substrate.nb_subdomain_grid_pts)
    _penetration = pnp.sum(
        init_disp - topography.heights() - mean_gap) / np.prod(
        substrate.nb_domain_grid_pts)

    print(_penetration)

    init_gap = init_disp - topography.heights() - _penetration
    init_gap[init_gap < 0] = 0

    res = constrained_conjugate_gradients(
        system.primal_objective(_penetration, gradient=True),
        system.primal_hessian_product,
        x0=init_gap, mean_val=mean_gap,
        gtol=gtol * max(Es * rms_slope, abs(
            interaction.max_tensile)) * topography.area_per_pt,
        maxiter=1000,
        communicator=comm,
        )

    assert res.success, res.message
    gap = res.x.reshape(substrate.nb_subdomain_grid_pts)
    print(res.nit)

    max_abs_error = pnp.max(abs(gap - reference_gap))
    assert max_abs_error < tol, "{} >= tol = {}".format(max_abs_error, tol)

    # just check that I am still able to compute the correct pressures
    grad = system.primal_objective(0, gradient=True)(
        gap)[1].reshape(substrate.nb_subdomain_grid_pts)

    nc_points = gap > 0
    nb_nc_points = pnp.sum(np.count_nonzero(nc_points))
    lagrange_mean_gap = pnp.sum(grad[nc_points]) / nb_nc_points

    forces = substrate.evaluate_force(gap + topography.heights()) + lagrange_mean_gap

    max_abs_error = pnp.max(abs(forces - forces_ref))
    assert max_abs_error < forcetol, \
        "{} >= tol = {}".format(max_abs_error, forcetol)

    print("""
    #########   NONADHESIVE, same mean gap #############
    """)
    nonadh_system = NonSmoothContactSystem(substrate=substrate,
                                           surface=topography)

    res = constrained_conjugate_gradients(
        nonadh_system.primal_objective(_penetration, gradient=True),
        nonadh_system.primal_hessian_product,
        x0=init_gap, mean_val=mean_gap,
        gtol=gtol * max(Es * rms_slope, abs(
            interaction.max_tensile)) * topography.area_per_pt,
        maxiter=1000,
        communicator=comm,
        )

    assert res.success, res.message
    gap_nonadh = res.x
    print(res.nit)

    if verbose and comm.Get_size() == 1.:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        fig, ax = plt.subplots()

        contactcmap = LinearSegmentedColormap.from_list('contactcmap', (
            (1, 1, 1, 0.), (1, 0, 0, 1)), N=256)
        ax.imshow(gap.reshape(substrate.nb_grid_pts) == 0, cmap=contactcmap)

        contactcmap = LinearSegmentedColormap.from_list('contactcmap', (
            (1, 1, 1, 0.), (0, 1, 0, 1)), N=256)
        ax.imshow(gap_nonadh.reshape(substrate.nb_grid_pts) == 0,
                  cmap=contactcmap)
