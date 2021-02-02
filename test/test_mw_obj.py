import numpy as np
import pytest
from muFFT import FFT
from ContactMechanics import PeriodicFFTElasticHalfSpace


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
    for given sinusoidal displacements, compares the pressures and the energies
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

    # for k_in:
    # print("testing wavevector ({}* np.pi * 2 / sx,
    # {}* np.pi * 2 / sy) ".format(*k))
    qx = k[0] * np.pi * 2 / sx
    qy = k[1] * np.pi * 2 / sy
    q = np.sqrt(qx ** 2 + qy ** 2)

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
    # k_float_disp_mw = k_float_disp * np.sqrt(system.stiffness_k)

    refpressure = - disp * E_s / 2 * q

    engine.create_plan(1)

    kpressure = substrate.evaluate_k_force(
        disp[substrate.subdomain_slices]) / substrate.area_per_pt

    expected_k_disp = np.zeros((nx, ny), dtype=complex)
    expected_k_disp[k[0], k[1]] += (.5 - .5j) * (nx * ny)
    # expected_k_disp[0,ny-1] += (-.5) * (nx * ny)

    # add the symetrics
    if k[0] == 0:
        expected_k_disp[0, -k[1]] += (.5 + .5j) * (nx * ny)
    if k[0] == nx // 2 and nx % 2 == 0:
        expected_k_disp[k[0], -k[1]] += (.5 + .5j) * (nx * ny)

    print(expected_k_disp)
    print(k_float_disp)

    expected_k_pressure = - E_s / 2 * q * expected_k_disp

    n = substrate.nb_grid_pts[0]
    nb_dims = len(substrate.nb_grid_pts)

    disp_k = k_float_disp

    if nb_dims == 2:
        grad_k = np.zeros(substrate.nb_grid_pts)
        n0 = substrate.nb_grid_pts[0]
        n1 = substrate.nb_grid_pts[1]

        if (n % 2 == 0):  # For even number of grid points
            grad_k[0, 0] = disp_k[0, 0] / (n0 * n1)
            grad_k[0, 1:n // 2] = 2 * disp_k[0, 1:n // 2] / (n0 * n1)
            grad_k[0, n // 2 + 1:] = 2 * disp_k[0,
                                         n // 2 + 1:] / (n0 * n1)
            grad_k[1:n // 2, 0] = 2 * disp_k[1:n // 2, 0] / (n0 * n1)
            grad_k[n // 2 + 1:, 0] = 2 * disp_k[n // 2 + 1:,
                                         0] / (n0 * n1)
            grad_k[1:n // 2, n // 2] = 2 * disp_k[1:n // 2,
                                           n // 2] / (n0 * n1)
            grad_k[n // 2 + 1:, n // 2] = 2 * disp_k[n // 2 + 1:,
                                              n // 2] / (n0 * n1)
            grad_k[n // 2, 1:n // 2] = 2 * disp_k[n // 2,
                                           1:n // 2] / (n0 * n1)
            grad_k[n // 2, n // 2 + 1:] = 2 * disp_k[n // 2,
                                              n // 2 + 1:] / (n0 * n1)
            grad_k[1:n // 2, 1:n // 2] = 4 * disp_k[1:n // 2,
                                             1:n // 2] / (n0 * n1)
            grad_k[n // 2 + 1:, 1:n // 2] = 4 * disp_k[n // 2 + 1:,
                                                1:n // 2] / (n0 * n1)
            grad_k[1:n // 2, n // 2 + 1:] = 4 * disp_k[1:n // 2,
                                                n // 2 + 1:] / (n0 * n1)
            grad_k[n // 2 + 1:, n // 2 + 1:] = 4 * disp_k[n // 2 + 1:,
                                                   n // 2 + 1:] / (n0 * n1)
            grad_k[n // 2, n // 2] = disp_k[n // 2, n // 2] / (n0 * n1)
            grad_k[n // 2, 0] = disp_k[n // 2, 0] / (n0 * n1)
            grad_k[0, n // 2] = disp_k[0, n // 2] / (n0 * n1)

        else:  # For odd number of grid points
            grad_k[0, 0] = disp_k[0, 0] / (n0 * n1)
            grad_k[0, 1:] = 2 * disp_k[0, 1:] / (n0 * n1)
            grad_k[1:, 0] = 2 * disp_k[1:, 0] / (n0 * n1)
            grad_k[1:, 1:] = 4 * disp_k[1:, 1:] / (n0 * n1)

    elif nb_dims == 1:
        if (n % 2 == 0):  # For even number of grid points 1-D.
            grad_k = np.zeros(substrate.nb_grid_pts)
            grad_k[0] = disp_k[0] / n
            grad_k[1:n // 2] = 2 * disp_k[1:n // 2] / n
            grad_k[n // 2 + 1:] = 2 * disp_k[n // 2 + 1:] / n
            grad_k[n // 2] = disp_k[n // 2] / n
        else:  # For odd number of grid points
            grad_k = np.zeros(substrate.nb_grid_pts)
            grad_k[0] = disp_k[0] / n
            grad_k[1:] = 2 * disp_k[1:] / n

    grad_k *= 0.5 * q * substrate.area_per_pt

    # ENERGY FROM SUBSTRATE

    energy = 0.5 * (
        np.sum(grad_k * disp_k))

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
