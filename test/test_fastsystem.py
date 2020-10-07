
import time
import unittest

import numpy as np
import pytest
from scipy.optimize import minimize

from NuMPI import MPI

import ContactMechanics as ContactMechanics
import Adhesion.Interactions as Contact
import ContactMechanics.Tools as Tools
from Adhesion.System import SmoothContactSystem
from ContactMechanics.Systems import NonSmoothContactSystem
from Adhesion.System import FastSmoothContactSystem
from Adhesion.System import make_system
from SurfaceTopography import make_sphere

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="tests only serial funcionalities, please execute with pytest")

pytestmark = pytest.mark.skip(reason="no support for FastSystem for the moment")

@pytest.fixture(params=range(50))
def self(request):
    np.random.seed(request.param)
    self.physical_sizes = (15, 15)  # (7.5+5*rand(), 7.5+5*rand())
    self.radius = 4
    base_res = 64  # TODO: put this back on 32, see issue #139
    self.res = (base_res, base_res)
    self.young = 3  # +2*random()

    self.substrate = ContactMechanics.FreeFFTElasticHalfSpace(
        self.res, self.young, self.physical_sizes, fft="serial")

    self.eps = 1  # +np.random.rand()
    self.sig = 2  # +np.random.rand()
    self.gam = 5  # +np.random.rand()
    self.rcut = 2.5 * self.sig  # +np.random.rand()
    self.interaction = Contact.LJ93smooth(self.eps, self.sig, self.gam)
    # self.min_pot = Contact.LJ93smooth(self.eps, self.sig, self.gam)
    self.min_pot = Contact.LJ93smoothMin(self.eps, self.sig, self.gam)
    # self.interaction = Contact.LJ93(self.eps, self.sig)
    # self.min_pot = Contact.LJ93SimpleSmooth(self.eps, self.sig, self.rcut)

    # self.interaction =Contact.Exponential(self.gam, 0.05, self.rcut)
    # self.min_pot = Contact.Exponential(self.gam, 0.05, self.rcut)

    self.surface = make_sphere(self.radius, self.res,
                                                 self.physical_sizes,
                                                 standoff=float('inf'))

    if False:
        import matplotlib.pyplot as plt
        fig, (axE, axF, axC) = plt.subplots(3, 1)
        z = np.linspace(0.8, 2, 100)
        V, dV, ddV = self.min_pot.evaluate(z, True, True, True)
        axE.plot(z, V)
        axF.plot(z, dV)
        axC.plot(z, ddV)

        V, dV, ddV = self.interaction.evaluate(z, True, True, True)
        axE.plot(z, V)
        axF.plot(z, dV)
        axC.plot(z, ddV)

        plt.show(block=True)

def test_FastSmoothContactSystem(self):
    S = FastSmoothContactSystem(self.substrate,
                                self.interaction,
                                self.surface)
    fun = S.objective(.95 * self.interaction.cutoff_radius)
    print(fun(np.zeros(S.babushka.substrate.nb_domain_grid_pts)))

def test_SystemFactory(self):
    S = make_system(self.substrate,
                    self.interaction,
                    self.surface, system_class=FastSmoothContactSystem)
    print("Mofo is periodic ?: ", self.substrate.is_periodic())
    print("substrate: ", self.substrate)
    self.assertIsInstance(S, FastSmoothContactSystem)
    self.assertIsInstance(S, SmoothContactSystem)

def test_babushka_translations(self):
    S = FastSmoothContactSystem(self.substrate,
                                self.interaction,
                                self.surface)
    fun = S.objective(.95 * self.interaction.cutoff_radius)

def test_equivalence(self):
    tol = 1e-6
    # here, i deliberately avoid using the make_system, because I want to
    # explicitly test the dumb (yet safer) way of computing problems with a
    # free, non-periodic  boundary. A user who invokes a system constructor
    # directliy like this is almost certainly mistaken
    systems = (SmoothContactSystem, FastSmoothContactSystem)

    def eval(system):
        print("running for system {}".format(system.__name__))
        S = system(self.substrate,
                   self.min_pot,
                   self.surface)
        offset = .8 * S.interaction.cutoff_radius
        fun = S.objective(offset, gradient=True)

        options = dict(ftol=1e-18, gtol=1e-10)
        disp = S.shape_minimisation_input(
            np.zeros(self.substrate.nb_domain_grid_pts))
        bla = fun(disp)
        result = minimize(fun, disp, jac=True,
                          method='L-BFGS-B', options=options)
        assert result.success, "{}".format(result)
        if system.is_proxy():
            dummy, force, disp = S.deproxified()

        else:
            disp = S.shape_minimisation_output(result.x)
        gap = S.compute_gap(disp, offset)
        gap[np.isinf(gap)] = self.min_pot.cutoff_radius

        print('r_min = {}'.format(self.min_pot.r_min))
        return S.interaction_force, disp, gap, S.compute_normal_force()

    def timer(fun, *args):
        start = time.perf_counter()
        res = fun(*args)
        delay = time.perf_counter() - start
        return res, delay

    (((force_slow, disp_slow, gap_slow, N_slow), slow_time),
     ((force_fast, disp_fast, gap_fast, N_fast), fast_time)) = tuple(
        (timer(eval, system) for system in systems))
    error = Tools.mean_err(disp_slow, disp_fast)

    print("Normal forces: fast: {}, slow: {}, error: {}".format(
        N_fast, N_slow, abs(N_slow - N_fast)))

    print("timings: fast: {}, slow: {}, gain: {:2f}%".format(
        fast_time, slow_time, 100 * (1 - fast_time / slow_time)))
    self.assertTrue(error < tol,
                    "error = {} > tol = {}".format(
                        error, tol))

def test_minimize_proxy(self):
    tol = 1e-6
    # here, i deliberately avoid using the make_system, because I want to
    # explicitly test the dumb (yet safer) way of computing problems with a
    # free, non-periodic  boundary. A user who invokes a system constructor
    # directliy like this is almost certainly mistaken
    systems = (SmoothContactSystem, FastSmoothContactSystem)

    def eval(system):
        print("running for system {}".format(system.__name__))
        S = system(self.substrate,
                   self.min_pot,
                   self.surface)
        offset = .8 * S.interaction.cutoff_radius
        options = dict(ftol=1e-18, gtol=1e-10)
        result = S.minimize_proxy(offset, options=options)

        gap = S.compute_gap(S.disp, offset)
        gap[np.isinf(gap)] = self.min_pot.cutoff_radius

        return S.interaction_force, S.disp, gap, S.compute_normal_force()

    def timer(fun, *args):
        start = time.perf_counter()
        res = fun(*args)
        delay = time.perf_counter() - start
        return res, delay

    (((force_slow, disp_slow, gap_slow, N_slow), slow_time),
     ((force_fast, disp_fast, gap_fast, N_fast), fast_time)) = tuple(
        (timer(eval, system) for system in systems))
    error = Tools.mean_err(disp_slow, disp_fast)

    print("Normal forces: fast: {}, slow: {}, error: {}".format(
        N_fast, N_slow, abs(N_slow - N_fast)))

    print("timings: fast: {}, slow: {}, gain: {:2f}%".format(
        fast_time, slow_time, 100 * (1 - fast_time / slow_time)))
    self.assertTrue(error < tol,
                    "error = {} > tol = {}".format(
                        error, tol))

def test_babuschka_eval(self):
    tol = 1e-6
    # here, i deliberately avoid using the make_system, because I want to
    # explicitly test the dumb (yet safer) way of computing problems with a
    # free, non-periodic  boundary. A user who invokes a system constructor
    # directliy like this is almost certainly mistaken
    S = FastSmoothContactSystem(self.substrate,
                                self.min_pot,
                                self.surface)
    offset = .8 * S.interaction.cutoff_radius
    S.create_babushka(offset)
    S.babushka.evaluate(
        np.zeros(S.babushka.substrate.nb_domain_grid_pts), offset,
        forces=True)
    F_n = S.babushka.compute_normal_force()
    babushka = S.babushka
    S = SmoothContactSystem(self.substrate,
                            self.min_pot,
                            self.surface)
    S.evaluate(np.zeros(S.substrate.nb_domain_grid_pts), offset, forces=True)
    F_n2 = S.compute_normal_force()

    error = abs(1 - F_n / F_n2)
    tol = 1e-14
    self.assertTrue(error < tol,
                    ("F_n = {}, F_n2 = {}, should be equal. type(S) = {}. "
                     "type(S.babushka) = {}, err = {}").format(
                        F_n, F_n2, type(S), type(babushka), error))

def test_unit_neutrality(self):
    tol = 2e-7
    # runs the same problem in two unit sets and checks whether results are
    # changed

    # Conversion factors
    length_c = 1. + 9  # np.random.rand()
    force_c = 2. + 1  # np.random.rand()
    pressure_c = force_c / length_c ** 2
    energy_per_area_c = force_c / length_c
    energy_c = force_c * length_c

    young = (self.young, pressure_c * self.young)
    size = (self.physical_sizes, tuple((length_c * s for s in self.physical_sizes)))
    print("SIZES!!!!! = ", size)
    radius = (self.radius, length_c * self.radius)
    res = self.res
    eps = (self.eps, energy_per_area_c * self.eps)
    sig = (self.sig, length_c * self.sig)
    gam = (self.gam, energy_per_area_c * self.gam)

    systems = list()
    offsets = list()
    length_rc = (1., 1. / length_c)
    force_rc = (1., 1. / force_c)
    energy_per_area_rc = (1., 1. / energy_per_area_c)
    energy_rc = (1., 1. / energy_c)

    for i in range(2):
        substrate = ContactMechanics.PeriodicFFTElasticHalfSpace(res, young[i],
                                                      size[i])
        interaction = Contact.LJ93smoothMin(
            eps[i], sig[i], gam[i])
        surface = make_sphere(radius[i], res, size[i], standoff=float(sig[i] * 1000))
        systems.append(make_system(substrate, interaction, surface, system_class=SmoothContactSystem))
        offsets.append(.8 * systems[i].interaction.cutoff_radius)

    gaps = list()
    for i in range(2):
        gap = systems[i].compute_gap(np.zeros(systems[i].nb_grid_pts), offsets[i])
        gaps.append(gap * length_rc[i])

    error = Tools.mean_err(gaps[0], gaps[1])
    self.assertTrue(error < tol,
                    "error = {} ≥ tol = {}".format(error, tol))

    forces = list()
    for i in range(2):
        energy, force = systems[i].evaluate(np.zeros(res), offsets[i], forces=True)
        forces.append(force * force_rc[i])

    error = Tools.mean_err(forces[0], forces[1])
    self.assertTrue(error < tol,
                    "error = {} ≥ tol = {}".format(error, tol))

    energies, forces = list(), list()
    substrate_energies = list()
    interaction_energies = list()
    disp = np.random.random(res)
    disp -= disp.mean()
    disp = (disp, disp * length_c)
    gaps = list()

    for i in range(2):
        energy, force = systems[i].evaluate(disp[i], offsets[i], forces=True)
        gap = systems[i].compute_gap(disp[i], offsets[i])
        gaps.append(gap * length_rc[i])
        energies.append(energy * energy_rc[i])
        substrate_energies.append(systems[i].substrate.energy * energy_rc[i])
        interaction_energies.append(systems[i].interaction.energy * energy_rc[i])
        forces.append(force * force_rc[i])

    error = Tools.mean_err(gaps[0], gaps[1])
    self.assertTrue(error < tol,
                    "error = {} ≥ tol = {}".format(error, tol))

    error = Tools.mean_err(forces[0], forces[1])

    self.assertTrue(error < tol,
                    "error = {} ≥ tol = {}".format(error, tol))

    error = abs(interaction_energies[0] - interaction_energies[1])
    self.assertTrue(error < tol,
                    "error = {} ≥ tol = {}".format(error, tol))

    error = abs(substrate_energies[0] - substrate_energies[1])
    self.assertTrue(error < tol,
                    "error = {} ≥ tol = {}, (c = {})".format(error, tol, energy_c))

    error = abs(energies[0] - energies[1])
    self.assertTrue(error < tol,
                    "error = {} ≥ tol = {}".format(error, tol))

    disps = list()
    for i in range(2):
        options = dict(ftol=1e-32, gtol=1e-20)
        result = systems[i].minimize_proxy(offsets[i], options=options)
        disps.append(systems[i].shape_minimisation_output(result.x) * length_rc[i])

    error = Tools.mean_err(disps[0], disps[1])
    self.assertTrue(error < tol,
                    "error = {} ≥ tol = {}, (c = {})".format(error, tol, length_c))

def test_BabushkaBoundaryError(self):
    """
    makes a Simulation in JKR-Like condition so that the contact area jump (snap-in) will lead to a too small
    babushka-subdomain area
    """
    with self.assertRaises(FastSmoothContactSystem.BabushkaBoundaryError):
        s = 128
        n = 64
        dx = 2
        size = (s, s)
        res = (n, n)
        radius = 100
        young = 1
        gam = 0.05

        surface = make_sphere(radius, res, size)
        ext_surface = make_sphere(radius, (2 * n, 2 * n), (2 * s, 2 * s), centre=(s / 2, s / 2))

        interaction = Contact.LJ93smoothMin(young / 18 * np.sqrt(2 / 5), 2.5 ** (1 / 6), gamma=gam)

        substrate = ContactMechanics.FreeFFTElasticHalfSpace(surface.nb_grid_pts, young, surface.physical_sizes)
        system = FastSmoothContactSystem(substrate, interaction, surface, margin=4)

        start_disp = - interaction.cutoff_radius + 1e-10
        load_history = np.concatenate((
            np.array((start_disp,)),
            np.arange(-1.63, -1.6, 2e-3),
            np.arange(-1.6, 0.6, 2e-1)[1:]))

        u = None
        for offset in load_history:
            opt = system.minimize_proxy(offset,
                                        u,
                                        method='L-BFGS-B',
                                        options=dict(ftol=1e-18, gtol=1e-10),
                                        lbounds=ext_surface.heights() + offset)

            u = system.disp

        import matplotlib.pyplot as plt
        X, Y = np.meshgrid((np.arange(0, int(n / 2))) * dx, (np.arange(0, int(n / 2))) * dx)
        fig, ax = plt.subplots()
        plt.colorbar(ax.pcolormesh(X, Y, substrate.interact_forces[-1, int(n / 2):, int(n / 2):]))

def test_FreeBoundaryError(self):
    """
    Returns
    -------
    """
    radius = 100
    young = 1

    s = 128.
    n = 64
    dx = s / n
    res = (n, n)
    size = (s, s)

    centre = (0.75 * s, 0.5 * s)

    topography = make_sphere(radius, res, size, centre=centre)
    ext_topography = make_sphere(radius, (2 * n, 2 * n), (2 * s, 2 * s), centre=centre)

    substrate = ContactMechanics.FreeFFTElasticHalfSpace(topography.nb_grid_pts, young,
                                              topography.physical_sizes,
                                              check_boundaries=True)

    for system in [NonSmoothContactSystem(substrate, topography), # TODO: this actually belongs to ContactMechanics
                   SmoothContactSystem(substrate, Contact.LJ93SimpleSmooth(0.01, 0.01, 10), topography)]:
        with self.subTest(system=system):
            offset = 15
            with self.assertRaises(ContactMechanics.FreeFFTElasticHalfSpace.FreeBoundaryError):
                opt = system.minimize_proxy(offset=offset)
            if False:
                import matplotlib.pyplot as plt
                X, Y = np.meshgrid((np.arange(0, n)) * dx,
                                   (np.arange(0, n)) * dx)
                fig, ax = plt.subplots()
                plt.colorbar(
                    ax.pcolormesh(X, Y, substrate.force)
                )
                plt.show(block=True)