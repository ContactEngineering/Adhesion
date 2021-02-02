import numpy as np
from ContactMechanics import PeriodicFFTElasticHalfSpace
from SurfaceTopography import Topography
from Adhesion.Interactions import VDW82, RepulsiveExponential
from Adhesion.System import SmoothContactSystem, make_system
import muFFT
from SurfaceTopography.Generation import fourier_synthesis
import scipy.optimize as optim
import matplotlib.pyplot as plt


def test_2d():
    n = 32
    surf_res = (n, n)

    z0 = 1
    Es = 1
    w = 0.01 * z0 * Es
    surf_size = (n, n)

    inter = RepulsiveExponential(w, 0.5, 0, 1.)

    substrate = PeriodicFFTElasticHalfSpace(surf_res, young=Es,
                                            physical_sizes=surf_size)

    surface = Topography(
        np.cos(np.arange(0, n) * np.pi * 2. / n) * np.ones((n, 1)),
        physical_sizes=surf_size)

    system = SmoothContactSystem(substrate, inter, surface)

    penetrations = np.linspace(-np.max(surface.heights()), 0, 10)

    # disp0 = np.zeros(surface.nb_grid_pts)
    # init_gap = disp0 - system.surface.heights() - penetrations[5]
    # init_gap[init_gap < 0] = 0
    # disp0 = init_gap + system.surface.heights() + penetrations[5]

    disp0 = np.random.uniform(0,1,size=surface.nb_grid_pts)
    # disp0 = np.ones(surface.nb_grid_pts)
    # disp0 = 0.5*disp0

    engine = muFFT.FFT(substrate.nb_grid_pts, fft='fftw',
                       allow_temporary_buffer=False,
                       allow_destroy_input=True)

    real_buffer = engine.register_hc_space_field(
        "real-space", 1)
    fourier_buffer = engine.register_hc_space_field(
        "fourier-space", 1)

    real_buffer.array()[...] = disp0.copy()
    engine.hcfft(real_buffer, fourier_buffer)
    k_float_disp = fourier_buffer.array()[...].copy()
    k_float_disp_mw = k_float_disp * np.sqrt(
        system.stiffness_k)

    en_mw = system.objective_k_float_mw(penetrations[5], gradient=True)(
        k_float_disp_mw)[0]
    print('mw el {}'.format(system.substrate.energy))
    en_real = system.objective(penetrations[5], gradient=True)(disp0)[0]
    print('real el {} '.format(system.substrate.energy))
    en_fourier = \
        system.objective_k_float(penetrations[5], gradient=True)(k_float_disp)[
            0]
    print('fourier el {} '.format(system.substrate.energy))

    print(en_fourier, en_real, en_mw)
    np.testing.assert_allclose(en_mw, en_real, atol=1e-3)
    np.testing.assert_allclose(en_fourier, en_real, atol=1e-3)
    np.testing.assert_allclose(en_fourier, en_mw, atol=1e-3)


def test_1d():
    Es = 1.
    rms_slope = 1.
    dx = 1
    n = 1024
    s = n * dx

    short_cutoff = 0.01 * s

    topo = fourier_synthesis(nb_grid_pts=(n,),
                             hurst=0.8,  # Fig. 4
                             physical_sizes=(s,),
                             rms_slope=1.,
                             long_cutoff=s,  # Fig. 4
                             short_cutoff=short_cutoff,
                             )

    Rc = 1 / topo.rms_curvature()
    interaction_length = 2.56e-2 * Rc

    gam_att = 2.05 * Es * Rc
    gam_rep = 2.10e3 * Es * Rc

    interaction = RepulsiveExponential(gam_rep, interaction_length / 2,
                                       gam_att,
                                       interaction_length).linearize_core(
        hardness=1000 * Es * topo.rms_slope())

    w = abs(interaction.v_min)

    system = make_system(interaction=interaction,
                         surface=topo,
                         substrate="periodic",
                         young=Es,
                         system_class=SmoothContactSystem)
    substrate = system.substrate

    engine = muFFT.FFT(substrate.nb_grid_pts, fft='fftw',
                       allow_temporary_buffer=False,
                       allow_destroy_input=True)

    real_buffer = engine.register_hc_space_field(
        "real-space", 1)
    fourier_buffer = engine.register_hc_space_field(
        "fourier-space", 1)

    penetrations = np.linspace(-np.max(topo.heights()), 0, 10)

    disp0 = np.zeros(topo.nb_grid_pts)
    init_gap = disp0 - system.surface.heights() - penetrations[5]
    init_gap[init_gap < 0] = 0
    disp0 = init_gap + system.surface.heights() + penetrations[5]

    real_buffer.array()[...] = disp0.copy()
    engine.hcfft(real_buffer, fourier_buffer)
    k_float_disp = fourier_buffer.array()[...].copy()
    k_float_disp_mw = k_float_disp * np.sqrt(
        system.stiffness_k)

    en_mw = system.objective_k_float_mw(penetrations[5], gradient=True)(
        k_float_disp_mw)[0]
    en_real = system.objective(penetrations[5], gradient=True)(disp0)[0]
    en_fourier = \
    system.objective_k_float(penetrations[5], gradient=True)(k_float_disp)[0]

    print(en_fourier, en_real, en_mw)
    np.testing.assert_allclose(en_mw, en_real, atol=1e-3)
    np.testing.assert_allclose(en_fourier, en_real, atol=1e-3)
    np.testing.assert_allclose(en_fourier, en_mw, atol=1e-3)
