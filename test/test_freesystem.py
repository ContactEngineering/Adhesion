#
# Copyright 2018, 2020 Antoine Sanner
#           2016, 2018, 2020 Lars Pastewka
#           2015-2016 Till Junge
#           2015 junge@cmsjunge
#
# ## MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
tests that the Fast optimization for free (non-periodic) systems is
consistent with computing the full system
"""

import time

import numpy as np
import pytest
from scipy.optimize import minimize

from NuMPI import MPI

import ContactMechanics as ContactMechanics
import Adhesion.Interactions as Contact
import ContactMechanics.Tools as Tools
from Adhesion.System import SmoothContactSystem
from SurfaceTopography import make_sphere
import os
from netCDF4 import Dataset

BASEDIR = os.path.dirname(os.path.realpath(__file__))

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="tests only serial funcionalities, "
                                       "please execute with pytest")


@pytest.mark.parametrize("r_c", [2., 6.])
@pytest.mark.parametrize("young", [3., 10, 100.])  # ,10.,100.
def test_minimization_parabolic_cutoff_linear_core(young, r_c):
    eps = 1.
    sig = 1.  # 2

    radius = 4.

    base_res = 32
    res = (base_res, base_res)

    size = (15., 15.)
    surface = make_sphere(radius, res, size, standoff=float('inf'))
    # ext_surface = make_sphere(radius,
    #                          [2 * r for r in res],
    #                          [2 * s for s in size],
    #                          centre=[s / 2 for s in size],
    #                          standoff=1000)

    substrate = ContactMechanics.FreeFFTElasticHalfSpace(
        res, young, size)

    pot = Contact.LJ93(eps, sig
                       ).parabolic_cutoff(r_c).linearize_core(0.5)

    if False:
        import matplotlib.pyplot as plt
        fig, (axp, axf) = plt.subplots(1, 2)

        z = np.linspace(0.5, 10, 50)

        v, dv, ddv = pot.evaluate(z, True, True)
        axp.plot(z, v)
        axf.plot(z, dv)

    S = SmoothContactSystem(substrate,
                            pot,
                            surface)

    # print("testing: {}, rc = {}, offset= {}".format(
    #   pot.__class__, S.interaction.cutoff_radius, S.interaction.offset))
    offset = .8 * S.interaction.parent_potential.cutoff_radius
    fun = S.objective(offset, gradient=True)

    options = dict(
        ftol=0,
        gtol=1e-5 *
        max(
            abs(pot.max_tensile),
            2 * young / np.pi * np.sqrt(
                (pot.r_min + offset) / radius)
            # max pressure in the hertzian contact
            ) * S.area_per_pt,
        maxcor=3)
    disp = S.shape_minimisation_input(
        np.zeros(substrate.nb_domain_grid_pts))

    # lbounds = S.shape_minimisation_input(ext_surface.heights() + offset)
    # bnds = tuple(zip(lbounds.tolist(), [None for i in range(len(lbounds))]))
    result = minimize(fun, disp, jac=True,
                      method='L-BFGS-B', options=options)
    # , bounds=bnds)
    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        # ax.pcolormesh(result.x.reshape(substrate.computational_nb_grid_pts))
        plt.colorbar(ax.pcolormesh(S.interaction_force))
        # np.savetxt("{}_forces.txt".format(pot_class.__name__),
        # S.interaction_force)
        ax.set_xlabel("x")
        ax.set_ylabel("")
        ax.grid(True)
        ax.legend()
        ax.set_title("Es={}, cutoff_radius={}".format(young, r_c))

        fig.tight_layout()
        # plt.show(block=True)
    assert result.success, "{}".format(result)

    if hasattr(result.message, "decode"):
        decoded_message = result.message.decode()
    else:
        decoded_message = result.message

    assert decoded_message == \
        'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'


"""
# TODO
@pytest.mark.skip(reason="Strange problem see issue #139 work on this in the mufft branch")
@pytest.mark.parametrize("base_res", [pytest.param(128, marks=pytest.mark.xfail),
                                      256])
@pytest.mark.parametrize("young", [3., 100.])  # mit young = 100 geht auch LJ93smoothMin durch
@pytest.mark.parametrize("pot_class", [pytest.param(Contact.LJ93smooth, marks=pytest.mark.xfail),
                                       Contact.LJ93smoothMin])
def test_minimization(pot_class, young, base_res):
    eps = 1
    sig = 2
    gam = 5

    radius = 4.

    res = (base_res, base_res)

    size = (15., 15.)
    surface = make_sphere(radius, res, size,
                                            standoff=float("inf"))
    ext_surface = make_sphere(radius, [2 * r for r in res], [2 * s for s in size],
                                                centre=[s / 2 for s in size], standoff=float("inf"))

    substrate = ContactMechanics.FreeFFTElasticHalfSpace(res, young, size, fft="numpy")

    if pot_class == Contact.VDW82smoothMin:
        pot = pot_class(gam * eps ** 8 / 3, 16 * np.pi * gam * eps ** 2, gamma=gam)
    elif pot_class == Contact.LJ93SimpleSmoothMin:
        pot = pot_class(eps, sig, cutoff_radius=10., r_ti=0.5)
    else:
        pot = pot_class(eps, sig, gam, )
    if hasattr(pot, "r_ti"):
        assert pot.r_ti < pot.r_t

    if False:
        import matplotlib.pyplot as plt
        fig, (axp, axf) = plt.subplots(1, 2)

        z = np.linspace(0.5, 10, 50)

        v, dv, ddv = pot.evaluate(z, True, True)
        axp.plot(z, v)
        axf.plot(z, dv)

    S = SmoothContactSystem(substrate,
                            pot,
                            surface)

    print("testing: {}, rc = {}, offset= {}".format(pot.__class__, S.interaction.cutoff_radius, S.interaction.offset))
    offset = .8 * S.interaction.cutoff_radius
    fun = S.objective(offset, gradient=True)

    options = dict(ftol=1e-18, gtol=1e-10)
    disp = S.shape_minimisation_input(
        np.zeros(substrate.nb_domain_grid_pts))

    lbounds = S.shape_minimisation_input(ext_surface.heights() + offset)
    bnds = tuple(zip(lbounds.tolist(), [None for i in range(len(lbounds))]))
    result = minimize(fun, disp, jac=True,
                      method='L-BFGS-B', options=options)  # , bounds=bnds)
    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        # ax.pcolormesh(result.x.reshape(substrate.computational_nb_grid_pts))
        ax.pcolormesh(S.interaction_force)
        np.savetxt("{}_forces.txt".format(pot_class.__name__), S.interaction_force)
        ax.set_xlabel("x")
        ax.set_ylabel("")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        plt.show(block=True)
    assert result.success, "{}".format(result)
"""  # noqa: E501


@pytest.fixture(params=range(50))
def self(request):
    np.random.seed(request.param)

    self.physical_sizes = (
        7.5 + 5 * np.random.rand(), 7.5 + 5 * np.random.rand())
    self.radius = 100
    base_res = 16
    self.res = (base_res, base_res)
    self.young = 3 + 2 * np.random.rand()

    self.substrate = ContactMechanics.FreeFFTElasticHalfSpace(
        self.res, self.young, self.physical_sizes)

    self.eps = 1 + np.random.rand()
    self.sig = 3 + np.random.rand()
    self.gam = (5 + np.random.rand())
    self.rcut = 2.5 * self.sig + np.random.rand()
    self.smooth = Contact.LJ93(self.eps, self.sig).spline_cutoff(self.gam)

    self.sphere = make_sphere(self.radius, self.res, self.physical_sizes)
    return self


def test_unconfirmed_minimization(self):
    # this merely makes sure that the code doesn't throw exceptions
    # the plausibility of the result is not verified
    res = self.res[0]
    size = self.physical_sizes[0]
    substrate = ContactMechanics.PeriodicFFTElasticHalfSpace(
        res,
        25 * self.young,
        self.physical_sizes[0])
    sphere = make_sphere(self.radius, res, size)
    # here, i deliberately avoid using the make_system, because I want to
    # explicitly test the dumb (yet safer) way of computing problems with a
    # free, non-periodic  boundary. A user who invokes a system constructor
    # directliy like this is almost certainly mistaken
    S = SmoothContactSystem(substrate, self.smooth, sphere)
    offset = -self.sig
    disp = np.zeros(substrate.nb_domain_grid_pts)

    fun_jac = S.objective(offset, gradient=True)
    fun = S.objective(offset, gradient=False)

    info = []
    start = time.perf_counter()
    result_grad = minimize(fun_jac, disp.reshape(-1), jac=True)
    duration_g = time.perf_counter() - start
    info.append("using gradient:")
    info.append("solved in {} seconds using {} fevals".format(
        duration_g, result_grad.nfev))

    start = time.perf_counter()
    result_simple = minimize(fun, disp, )
    duration_w = time.perf_counter() - start
    info.append("without gradient:")
    info.append("solved in {} seconds using {} fevals".format(
        duration_w, result_simple.nfev))

    info.append("speedup (timewise) was {}".format(duration_w / duration_g))

    print('\n'.join(info))

    message = ("Success with gradient: {0.success}, message was '{0.message"
               "}',\nSuccess without: {1.success}, message was '{1.message}"
               "'").format(result_grad, result_simple)
    assert result_grad.success and result_simple.success, message


def test_comparison_pycontact(self):
    tol = 1e-9
    ref_fpath = os.path.join(BASEDIR, 'ref_smooth_sphere.nc')
    out_fpath = os.path.join(BASEDIR, 'ref_smooth_sphere.out')
    ref_data = Dataset(ref_fpath, mode='r', format='NETCDF4')
    with open(out_fpath) as fh:
        fh.__next__()
        fh.__next__()
        ref_N = float(fh.__next__().split()[0])
    # 1. replicate potential
    sig = ref_data.lj_sigma
    eps = ref_data.lj_epsilon
    # pycontact doesn't store gamma (work of adhesion) in the nc file, but
    # the computed rc1, which I can test for consistency
    gamma = 0.001
    potential = Contact.LJ93(eps, sig).spline_cutoff(gamma)
    error = abs(potential.cutoff_radius - ref_data.lj_rc2)
    assert error < tol, \
        ("computation of lj93smooth cut-off radius differs from pycontact "
         "reference: PyCo: {}, pycontact: {}, error: {}, tol: "
         "{}").format(potential.cutoff_radius, ref_data.lj_rc2, error, tol)
    # 2. replicate substrate
    res = (ref_data.size // 2, ref_data.size // 2)

    size = tuple((float(r) for r in res))
    young = 2.  # pycontact convention (hardcoded)
    substrate = ContactMechanics.FreeFFTElasticHalfSpace(res, young, size)

    # 3. replicate surface
    radius = ref_data.Hertz
    centre = (15.5, 15.5)
    surface = make_sphere(radius, res, size, centre=centre)
    # ref_h = -np.array(ref_data.variables['h'])
    # ref_h -= ref_h.max()
    # surface = SurfaceTopography.NumpySurface(ref_h)
    # 4. Set up system:
    S = SmoothContactSystem(substrate, potential, surface)

    ref_profile = np.array(
        ref_data.variables['h'] + ref_data.variables['avgh'][0])[:32, :32]
    offset = -.8 * potential.cutoff_radius
    gap = S.compute_gap(np.zeros(substrate.nb_domain_grid_pts), offset)
    # diff = ref_profile - gap
    # pycontact centres spheres at (n + 0.5, m + 0.5). need to correct for test
    correction = radius - np.sqrt(radius ** 2 - .5)
    error = Tools.mean_err(ref_profile + correction, gap)
    assert error < tol, \
        ("initial profiles differ (mean error ē = {} > tol = {}, mean gap = {}"
         "mean ref_profile = {})").format(
            error, tol, gap.mean(), (ref_profile + correction).mean())
    # pycontact does not save the offset in the nc, so this one has to be
    # taken on faith
    fun = S.objective(offset + correction, gradient=True)
    fun_hard = S.objective(offset + correction, gradient=False)  # noqa: F841

    # initial guess (cheating) is the solution of pycontact
    disp = np.zeros(S.substrate.nb_domain_grid_pts)
    disp[:ref_data.size, :ref_data.size] = -ref_data.variables['u'][0]
    gap = S.compute_gap(disp, offset)
    print("gap:     min, max = {}, offset = {}".format((gap.min(), gap.max()),
                                                       offset))
    print("profile: min, max = {}".format(
        (S.surface.heights().min(), S.surface.heights().max())))
    options = dict(ftol=1e-15, gtol=1e-12)
    result = minimize(fun, disp, jac=True,
                      callback=S.callback(force=True), method='L-BFGS-B',
                      options=options)

    # options = dict(ftol = 1e-12, gtol = 1e-10, maxiter=100000)
    # result = minimize(fun_hard, disp, jac=False,
    #   callback=S.callback(force=False), method = 'L-BFGS-B', options=options)

    e, force = fun(result.x)

    error = abs(ref_N - S.compute_normal_force())
    # here the answers differ slightly, relaxing the tol for this one
    ftol = 1e-7

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # CS = plt.contourf(ref_data.variables['f'][0])
    # plt.colorbar(CS)
    # plt.title("ref")
    # fig = plt.figure()
    # CS = plt.contourf(S.substrate.force[:32, :32])
    # plt.colorbar(CS)
    # plt.title("substrate")
    # fig = plt.figure()
    # CS = plt.contourf(S.interaction_force[:32, :32])
    # plt.colorbar(CS)
    # plt.title("interaction")
    # plt.show()

    # fig = plt.figure()
    # CS = plt.contourf(-ref_data.variables['u'][0][:32, :32])
    # plt.colorbar(CS)
    # plt.title("ref_u")
    # fig = plt.figure()
    # CS = plt.contourf(result.x.reshape([64, 64])[:32, :32])
    # plt.colorbar(CS)
    # plt.title("my_u")
    # fig = plt.figure()
    # CS = plt.contourf(result.x.reshape([64, 64])[:32, :32]
    #   + ref_data.variables['u'][0][:32, :32])
    # plt.colorbar(CS)
    # plt.title("my_u - ref_u")
    # plt.show()

    assert \
        error < ftol, \
        ("resulting normal forces differ: error = {} > tol = {}, "
         "ref_force_n = {}, my_force_n = {}\n"
         "OptimResult was {}\n"
         "elast energy = {}\n"
         "interaction_force = {}\n"
         "#substrate_force = {}\n"
         " System type = '{}'").format(
            error, ftol, ref_N, S.compute_normal_force(), result,
            S.substrate.energy, S.interaction_force.sum(),
            S.substrate.force.sum(), type(S))
    error = Tools.mean_err(
        disp, result.x.reshape(S.substrate.nb_domain_grid_pts))
    assert error < ftol, \
        "resulting displacements differ: error = {} > tol = {}".format(
            error, ftol)


def test_size_insensitivity(self):
    tol = 1e-6
    ref_fpath = os.path.join(BASEDIR, 'ref_smooth_sphere.nc')
    out_fpath = os.path.join(BASEDIR, 'ref_smooth_sphere.out')
    ref_data = Dataset(ref_fpath, mode='r', format='NETCDF4')
    with open(out_fpath) as fh:
        fh.__next__()
        fh.__next__()
        # ref_N = float(fh.__next__().split()[0])
    # 1. replicate potential
    sig = ref_data.lj_sigma
    eps = ref_data.lj_epsilon
    # pycontact doesn't store gamma (work of adhesion) in the nc file, but
    # the computed rc1, which I can test for consistency
    gamma = 0.001
    potential = Contact.LJ93(eps, sig).spline_cutoff(gamma)
    error = abs(potential.cutoff_radius - ref_data.lj_rc2)
    assert error < tol, \
        ("computation of lj93smooth cut-off radius differs from pycontact "
         "reference: PyCo: {}, pycontact: {}, error: {}, tol: "
         "{}").format(potential.cutoff_radius, ref_data.lj_rc2, error, tol)
    nb_compars = 3
    normalforce = np.zeros(nb_compars)
    options = dict(ftol=1e-12, gtol=1e-10)

    for i, nb_grid_pts in ((i, ref_data.size // 4 * 2 ** i) for i in
                           range(nb_compars)):
        res = (nb_grid_pts, nb_grid_pts)

        size = tuple((float(r) for r in res))
        young = 2.  # pycontact convention (hardcoded)
        substrate = ContactMechanics.FreeFFTElasticHalfSpace(res, young, size)

        # 3. replicate surface
        radius = ref_data.Hertz
        surface = make_sphere(radius, res, size, kind="paraboloid")

        # 4. Set up system:
        S = SmoothContactSystem(substrate, potential, surface)
        # pycontact does not save the offset in the nc, so this one has to be
        # taken on faith
        offset = -.8 * potential.cutoff_radius
        fun = S.objective(offset, gradient=True)
        disp = np.zeros(np.prod(res) * 4)
        minimize(fun, disp, jac=True,
                 method='L-BFGS-B', options=options)
        normalforce[i] = S.interaction_force.sum()
        error = abs(normalforce - normalforce.mean()).mean()
    assert error < tol, "error = {:.15g} > tol = {}, N = {}".format(
        error, tol, normalforce)
