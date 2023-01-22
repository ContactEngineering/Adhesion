import numpy as np
import pytest
from muFFT import FFT
from ContactMechanics import PeriodicFFTElasticHalfSpace
from SurfaceTopography.Generation import fourier_synthesis
from Adhesion.Interactions import RepulsiveExponential
from Adhesion.System import make_system, SmoothContactSystem
from NuMPI import MPI

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="tests only serial funcionalities, "
                                       "please execute with pytest")


@pytest.mark.parametrize("nx, ny", [(15, 15),
                                    (8, 8),
                                    (9, 9),
                                    (113, 113)])
@pytest.mark.parametrize("k", [(1, 0),
                               (0, 1),
                               (1, 2),
                               (4, 0),
                               (1, 4),
                               (0, 2),
                               (4, 4),
                               (0, 4)])
def test_sinewave_(k, nx, ny):
    """
    for given sinusoidal displacements, compares the energies
    to the analytical solutions

    Special cases at the edges of the fourier domain are done

    Parameters
    ----------
    k

    Returns
    -------

    """

    sx = 2.45  # 30.0
    sy = 1.0

    # equivalent Young's modulus
    E_s = 1.0

    qx = k[0] * np.pi * 2 / sx
    qy = k[1] * np.pi * 2 / sy
    q = np.sqrt(qx ** 2 + qy ** 2)

    ###########################################################
    # This is only for the purpose of using the mass-weighted objective
    # accessible only through the Adhesive system. Interaction and Topography
    # do not affect the computation of mw_energy.

    topo = fourier_synthesis(nb_grid_pts=(nx, ny),
                             hurst=0.8,  # Fig. 4
                             physical_sizes=(sx, sy),
                             rms_slope=1.,
                             long_cutoff=sx,
                             short_cutoff=0.01 * sx,
                             )

    inter = RepulsiveExponential(0, 0.5, 0, 1.0)

    system = make_system(interaction=inter,
                         surface=topo,
                         substrate="periodic",
                         young=1.0,
                         system_class=SmoothContactSystem)
    ################################################################

    Y, X = np.meshgrid(np.linspace(0, sy, ny + 1)[:-1],
                       np.linspace(0, sx, nx + 1)[:-1])
    disp = np.cos(qx * X + qy * Y) + np.sin(qx * X + qy * Y)

    substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                            fft='fftw')
    engine = FFT((nx, ny), fft='fftw')

    real_buffer = engine.register_halfcomplex_field(
        "real-space", 1)
    fourier_buffer = engine.register_halfcomplex_field(
        "fourier-space", 1)
    real_buffer.array()[...] = disp.copy()
    engine.hcfft(real_buffer, fourier_buffer)
    k_float_disp = fourier_buffer.array()[...].copy()
    k_float_disp_mw = k_float_disp * np.sqrt(system.stiffness_k)

    engine.create_plan(1)

    expected_k_disp = np.zeros((nx, ny), dtype=complex)
    expected_k_disp[k[0], k[1]] += (.5 - .5j) * (nx * ny)

    mw_energy = \
        system.preconditioned_objective(0, gradient=True)(k_float_disp_mw)[0]

    fourier_energy = system.objective_k_float(0., gradient=True)(k_float_disp)[0]

    refenergy = E_s / 8 * 2 * q * sx * sy

    np.testing.assert_allclose(
        mw_energy, refenergy, rtol=1e-10,
        err_msg=f"wavevektor {k} for "
                f"nb_domain_grid_pts {substrate.nb_domain_grid_pts},"
                f" subdomain nb_grid_pts {substrate.nb_subdomain_grid_pts}, "
                f"nb_fourier_grid_pts {substrate.nb_fourier_grid_pts}")

    np.testing.assert_allclose(
        fourier_energy, refenergy, rtol=1e-10,
        err_msg=f"wavevektor {k} for "
                f"nb_domain_grid_pts {substrate.nb_domain_grid_pts},"
                f" subdomain nb_grid_pts {substrate.nb_subdomain_grid_pts}, "
                f"nb_fourier_grid_pts {substrate.nb_fourier_grid_pts}")
