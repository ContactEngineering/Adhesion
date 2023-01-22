import numpy as np
from ContactMechanics import PeriodicFFTElasticHalfSpace
from SurfaceTopography import Topography
from Adhesion.Interactions import RepulsiveExponential
from Adhesion.System import SmoothContactSystem, make_system
import muFFT
from SurfaceTopography.Generation import fourier_synthesis
import pytest
from NuMPI import MPI

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="tests only serial funcionalities, "
                                       "please execute with pytest")


def test_2d():
    n = 32
    surf_res = (n, n)

    z0 = 1
    Es = 1
    w = 0.01 * z0 * Es
    surf_size = (n, n)

    inter = RepulsiveExponential(2 * w, 0.5, w, 1.)

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

    disp0 = np.random.uniform(0, 1, size=surface.nb_grid_pts)

    engine = muFFT.FFT(substrate.nb_grid_pts, fft='fftw',
                       allow_temporary_buffer=False,
                       allow_destroy_input=True)

    real_buffer = engine.register_halfcomplex_field(
        "real-space", 1)
    fourier_buffer = engine.register_halfcomplex_field(
        "fourier-space", 1)

    real_buffer.array()[...] = disp0.copy()
    engine.hcfft(real_buffer, fourier_buffer)
    k_float_disp = fourier_buffer.array()[...].copy()
    k_float_disp_mw = k_float_disp * np.sqrt(
        system.stiffness_k)

    en_mw = system.preconditioned_objective(penetrations[5], gradient=True)(
        k_float_disp_mw)[0]
    # print('mw el {}'.format(system.substrate.energy))
    en_mw_el = system.substrate.energy
    en_mw_adh = system.interaction_energy
    en_real = system.objective(penetrations[5], gradient=True)(disp0)[0]
    en_real_el = system.substrate.energy
    en_real_adh = system.interaction_energy
    # print('real el {} '.format(system.substrate.energy))
    en_fourier = \
        system.objective_k_float(penetrations[5], gradient=True)(k_float_disp)[
            0]
    en_fourier_el = system.substrate.energy
    en_fourier_adh = system.interaction_energy
    # print('fourier el {} '.format(system.substrate.energy))

    # print(en_fourier, en_real, en_mw)

    np.testing.assert_allclose(en_mw_el, en_real_el, atol=1e-8)
    np.testing.assert_allclose(en_fourier_el, en_real_el, atol=1e-8)
    np.testing.assert_allclose(en_fourier_el, en_mw_el, atol=1e-8)
    np.testing.assert_allclose(en_mw_adh, en_real_adh, atol=1e-8)
    np.testing.assert_allclose(en_fourier_adh, en_real_adh, atol=1e-8)
    np.testing.assert_allclose(en_fourier_adh, en_mw_adh, atol=1e-8)
    np.testing.assert_allclose(en_mw, en_real, atol=1e-8)
    np.testing.assert_allclose(en_fourier, en_real, atol=1e-8)
    np.testing.assert_allclose(en_fourier, en_mw, atol=1e-8)


def test_1d():
    Es = 1.
    # rms_slope = 1.
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

    Rc = 1 / topo.rms_curvature_from_profile()
    interaction_length = 2.56e-2 * Rc

    gam_att = 2.05 * Es * Rc
    gam_rep = 2.10e3 * Es * Rc

    interaction = RepulsiveExponential(gam_rep, interaction_length / 2,
                                       gam_att,
                                       interaction_length).linearize_core(
        hardness=1000 * Es * topo.rms_slope_from_profile())

    # w = abs(interaction.v_min)

    system = make_system(interaction=interaction,
                         surface=topo,
                         substrate="periodic",
                         young=Es,
                         system_class=SmoothContactSystem)
    substrate = system.substrate

    engine = muFFT.FFT(substrate.nb_grid_pts, fft='fftw',
                       allow_temporary_buffer=False,
                       allow_destroy_input=True)

    real_buffer = engine.register_halfcomplex_field(
        "real-space", 1)
    fourier_buffer = engine.register_halfcomplex_field(
        "fourier-space", 1)

    penetrations = np.linspace(-np.max(topo.heights()), 0, 10)

    disp0 = np.zeros(topo.nb_grid_pts)
    init_gap = disp0 - system.surface.heights() - penetrations[5]
    disp0 = init_gap
    # init_gap[init_gap < 0] = 0
    # disp0 = init_gap + system.surface.heights() + penetrations[5]

    real_buffer.array()[...] = disp0.copy()
    engine.hcfft(real_buffer, fourier_buffer)
    k_float_disp = fourier_buffer.array()[...].copy()
    k_float_disp_mw = k_float_disp * np.sqrt(
        system.stiffness_k)

    en_mw = system.preconditioned_objective(penetrations[5], gradient=True)(
        k_float_disp_mw)[0]
    en_mw_el = system.substrate.energy
    en_mw_adh = system.interaction_energy
    en_real = system.objective(penetrations[5], gradient=True)(disp0)[0]
    en_real_el = system.substrate.energy
    en_real_adh = system.interaction_energy
    en_fourier = \
        system.objective_k_float(penetrations[5], gradient=True)(k_float_disp)[
            0]
    en_fourier_el = system.substrate.energy
    en_fourier_adh = system.interaction_energy

    # print(en_fourier, en_real, en_mw, en_real_el, en_fourier_el, en_mw_el)

    np.testing.assert_allclose(en_mw, en_real, atol=1e-8)
    np.testing.assert_allclose(en_fourier, en_real, atol=1e-8)
    np.testing.assert_allclose(en_fourier, en_mw, atol=1e-8)
    np.testing.assert_allclose(en_mw_el, en_real_el, atol=1e-8)
    np.testing.assert_allclose(en_fourier_el, en_real_el, atol=1e-8)
    np.testing.assert_allclose(en_fourier_el, en_mw_el, atol=1e-8)
    np.testing.assert_allclose(en_mw_adh, en_real_adh, atol=1e-8)
    np.testing.assert_allclose(en_fourier_adh, en_real_adh, atol=1e-8)
    np.testing.assert_allclose(en_fourier_adh, en_mw_adh, atol=1e-8)


@pytest.mark.skip("preconditioned hessian (product) not implemented yet. See issue #49")
def test_preconditioned_hessian():
    n = 32
    surf_res = (n, n)

    z0 = 1
    Es = 1
    w = 0.01 * z0 * Es
    surf_size = (n, n)

    inter = RepulsiveExponential(2 * w, 0.5, w, 1.)

    substrate = PeriodicFFTElasticHalfSpace(surf_res, young=Es,
                                            physical_sizes=surf_size)

    surface = Topography(
        np.cos(np.arange(0, n) * np.pi * 2. / n) * np.ones((n, 1)),
        physical_sizes=surf_size)

    system = SmoothContactSystem(substrate, inter, surface)

    # penetrations = np.linspace(-np.max(surface.heights()), 0, 10)
    offset = 0  # penetrations[5]

    engine = muFFT.FFT(substrate.nb_grid_pts, fft='fftw',
                       allow_temporary_buffer=False,
                       allow_destroy_input=True)

    real_buffer = engine.register_halfcomplex_field(
        "real-space", 1)
    fourier_buffer = engine.register_halfcomplex_field(
        "fourier-space", 1)

    obj = system.preconditioned_objective(offset, True)

    gaps = np.random.uniform(0, 1, size=surface.nb_grid_pts)
    # np.random.random(size=(n, n))
    real_buffer.array()[...] = gaps.copy()
    engine.hcfft(real_buffer, fourier_buffer)
    k_float_disp = fourier_buffer.array()[...].copy()

    dgaps = np.random.normal(size=(n, n)) * np.mean(gaps) / 10
    real_buffer.array()[...] = dgaps.copy()
    engine.hcfft(real_buffer, fourier_buffer)
    d_k_float_disp = fourier_buffer.array()[...].copy()

    _, grad = obj(k_float_disp)

    if True:
        hs = np.array([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5,
                       1e-6, 1e-7])
        rms_errors = []
        for h in hs:
            _, grad_d = obj(k_float_disp + h * d_k_float_disp)
            dgrad = grad_d - grad
            # dgrad_from_hess = cf.hessian_product(h * da, a, penetration)
            dgrad_from_hess = system.hessian_product_preconditioned(offset)(
                k_float_disp, h * d_k_float_disp)

            rms_errors.append(np.sqrt(np.mean((dgrad_from_hess - dgrad) ** 2)))

        # Visualize the quadratic convergence of the taylor expansion
        # What to expect:
        # Taylor expansion: g(x + h ∆x) - g(x) = Hessian * h * ∆x + O(h^2)
        # We should see quadratic convergence as long as h^2 > g epsmach,
        # the precision with which we are able to determine ∆g.
        # What is the precision with which the hessian product is made ?
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(hs, rms_errors / hs ** 2, "+-")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)
        plt.show()

        hs = np.array([1e-2, 1e-3, 1e-4])
    rms_errors = []
    for h in hs:
        grad_d = obj(k_float_disp + h * d_k_float_disp)[1]
        dgrad = grad_d - grad
        dgrad_from_hess = system.hessian_product_preconditioned(offset)(
            k_float_disp, h * d_k_float_disp)
        rms_errors.append(np.sqrt(np.mean((dgrad_from_hess - dgrad) ** 2)))
        rms_errors.append(
            np.sqrt(
                np.mean(
                    (dgrad_from_hess.reshape(-1) - dgrad.reshape(-1)) ** 2)))

    rms_errors = np.array(rms_errors)
    assert rms_errors[-1] / rms_errors[0] < 1.5 * (hs[-1] / hs[0]) ** 2
