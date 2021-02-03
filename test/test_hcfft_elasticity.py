import numpy as np
import pytest
from muFFT import FFT
from ContactMechanics import PeriodicFFTElasticHalfSpace
from SurfaceTopography.Generation import fourier_synthesis
from Adhesion.Interactions import RepulsiveExponential
from Adhesion.System import make_system, SmoothContactSystem

# @pytest.mark.parametrize("nx, ny", [(15, 15),
#                                     (8, 8),
#                                     (9, 9),
#                                     (113, 113)])

@pytest.mark.parametrize("k", [  (1, 0),
        (0, 1),
    (1, 2),
    (4, 0),
    (1, 4),
    (0, 2),
    (4, 4),
    (0, 4)])
def test_sineWave_(k):
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

    nx = ny = 8

    # equivalent Young's modulus
    E_s = 1.0

    qx = k[0] * np.pi * 2 / sx
    qy = k[1] * np.pi * 2 / sy
    q = np.sqrt(qx ** 2 + qy ** 2)

     ###########################################################
    # This is only for the purpose of using the mass-weighted objective
    # accessible only through the Adhesive system. Interaction and Topography
    # do not affect the computation of energy.


    topo = fourier_synthesis(nb_grid_pts=(nx,ny),
                             hurst=0.8,  # Fig. 4
                             physical_sizes=(sx,sy),
                             rms_slope=1.,
                             long_cutoff=sx,
                             short_cutoff=0.01*sx ,
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

    real_buffer = engine.register_hc_space_field(
        "real-space", 1)
    fourier_buffer = engine.register_hc_space_field(
        "fourier-space", 1)
    real_buffer.array()[...] = disp.copy()
    engine.hcfft(real_buffer, fourier_buffer)
    k_float_disp = fourier_buffer.array()[...].copy()
    k_float_disp_mw = k_float_disp * np.sqrt(system.stiffness_k)

    refpressure = - disp * E_s / 2 * q

    engine.create_plan(1)

    kpressure = substrate.evaluate_k_force(
        disp[substrate.subdomain_slices]) / substrate.area_per_pt

    expected_k_disp = np.zeros((nx, ny), dtype=complex)
    expected_k_disp[k[0], k[1]] += (.5 - .5j) * (nx * ny)

    # add the symetrics
    if k[0] == 0:
        expected_k_disp[0, -k[1]] += (.5 + .5j) * (nx * ny)
    if k[0] == nx // 2 and nx % 2 == 0:
        expected_k_disp[k[0], -k[1]] += (.5 + .5j) * (nx * ny)

    expected_k_pressure = - E_s / 2 * q * expected_k_disp

    energy = system.objective_k_float_mw(0, gradient=True)(k_float_disp_mw)[0]

    computedenergy_kspace = energy

    refenergy = E_s / 8 * 2 * q * sx * sy

    np.testing.assert_allclose(
        computedenergy_kspace, refenergy,
        rtol=1e-10,
        err_msg="wavevektor {} for nb_domain_grid_pts {}, "
                "subdomain nb_grid_pts {}, nb_fourier_grid_pts {}"
            .format(k, substrate.nb_domain_grid_pts,
                    substrate.nb_subdomain_grid_pts,
                    substrate.nb_fourier_grid_pts))

