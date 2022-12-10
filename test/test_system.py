#
# Copyright 2018, 2020 Antoine Sanner
#           2016, 2020 Lars Pastewka
#           2015-2016 Till Junge
#           2015 junge@cmsjunge
#
# ### MIT license
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
Tests the creation of tribosystems
"""

from numpy.random import rand, random
import numpy as np

from scipy.optimize import minimize
import time

import os

from Adhesion.System import make_system
from ContactMechanics.Systems import IncompatibleResolutionError
from Adhesion.System import SmoothContactSystem
import ContactMechanics as Solid
import Adhesion.Interactions as Inter

import ContactMechanics.Tools as Tools
from SurfaceTopography import make_sphere, Topography

import pytest
from NuMPI import MPI

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="tests only serial functionalities, "
                                       "please execute with pytest")

BASEDIR = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(params=range(50))
def self(request):
    np.random.seed(request.param)
    self.physical_sizes = (7.5 + 5 * rand(), 7.5 + 5 * rand())
    self.radius = 100
    base_res = 16
    self.res = (base_res, base_res)
    self.young = 3 + 2 * random()

    self.substrate = Solid.PeriodicFFTElasticHalfSpace(self.res, self.young,
                                                       self.physical_sizes)

    self.eps = 1 + np.random.rand()
    self.sig = 3 + np.random.rand()
    self.gam = (5 + np.random.rand())
    self.rcut = 2.5 * self.sig + np.random.rand()
    self.smooth = Inter.LJ93(self.eps, self.sig
                             ).spline_cutoff(self.gam
                                             ).linearize_core()

    self.sphere = make_sphere(self.radius, self.res, self.physical_sizes)
    return self


def test_DecoratedTopography(self):
    top = self.sphere.detrend()
    make_system(substrate="periodic",
                interaction="hardwall",
                young=1.,
                surface=top
                )


def test_RejectInconsistentSizes(self):
    incompat_res = tuple((2 * r for r in self.res))
    incompat_sphere = make_sphere(self.radius, incompat_res,
                                  self.physical_sizes)
    with pytest.raises(IncompatibleResolutionError):
        make_system(self.substrate, self.smooth, incompat_sphere,
                    system_class=SmoothContactSystem)


def test_SmoothContact(self):
    S = SmoothContactSystem(self.substrate, self.smooth, self.sphere)
    offset = self.sig
    disp = np.zeros(self.res)
    pot, forces = S.evaluate(disp, offset, forces=True)


def test_SystemGradient(self):
    res = self.res
    size = [r * 1.28 for r in self.res]
    substrate = Solid.PeriodicFFTElasticHalfSpace(res, 25 * self.young,
                                                  size)
    sphere = make_sphere(self.radius, res, size)
    S = SmoothContactSystem(substrate, self.smooth, sphere)
    disp = random(res) * self.sig / 10
    disp -= disp.mean()
    offset = -self.sig
    gap = S.compute_gap(disp, offset)

    # check subgradient of potential
    V, dV, ddV = S.interaction.evaluate(gap, potential=True, gradient=True)
    f = V.sum()
    g = dV

    def fun(x):
        return S.interaction.evaluate(x)[0].sum()

    approx_g = Tools.evaluate_gradient(
        fun, gap, self.sig / 1e5)

    tol = 1e-8
    error = Tools.mean_err(g, approx_g)
    msg = ["interaction: "]
    msg.append("f = {}".format(f))
    msg.append("g = {}".format(g))
    msg.append('approx = {}'.format(approx_g))
    msg.append("g/approx = {}".format(g / approx_g))
    msg.append("error = {}".format(error))
    msg.append("tol = {}".format(tol))
    assert error < tol, ", ".join(msg)
    interaction = dict({
        "e": f * S.area_per_pt,
        "g": g * S.area_per_pt,
        "a": approx_g * S.area_per_pt
        })
    # check subgradient of substrate
    V, dV = S.substrate.evaluate(disp, pot=True, forces=True)
    f = V.sum()
    g = -dV

    def fun(x):
        return S.substrate.evaluate(x)[0].sum()

    approx_g = Tools.evaluate_gradient(
        fun, disp, self.sig / 1e5)

    tol = 1e-8
    error = Tools.mean_err(g, approx_g)
    msg = ["substrate: "]
    msg.append("f = {}".format(f))
    msg.append("g = {}".format(g))
    msg.append('approx = {}'.format(approx_g))
    msg.append("error = {}".format(error))
    msg.append("tol = {}".format(tol))
    assert error < tol, ", ".join(msg)
    substrate = dict({
        "e": f,
        "g": g,
        "a": approx_g
        })

    V, dV = S.evaluate(disp, offset, forces=True)
    f = V
    g = -dV
    approx_g = Tools.evaluate_gradient(S.objective(offset), disp, 1e-5)
    approx_g2 = Tools.evaluate_gradient(
        lambda x: S.objective(offset, gradient=True)(x)[0], disp, 1e-5)
    tol = 1e-6

    assert Tools.mean_err(approx_g2, approx_g) < tol, \
        "approx_g  = {}\napprox_g2 = {}\nerror = {}, tol = {}".format(
            approx_g, approx_g2, Tools.mean_err(approx_g2, approx_g),
            tol)

    i, s = interaction, substrate
    f_combo = i['e'] + s['e']
    error = abs(f_combo - V)

    assert error < tol, \
        "f_combo = {}, f = {}, error = {}, tol = {}".format(
            f_combo, V, error, tol)

    g_combo = i['g'] + s['g']
    error = Tools.mean_err(g_combo, g)
    assert error < tol, \
        "g_combo = {}, g = {}, error = {}, tol = {}, g/g_combo = {}".format(
            g_combo, g, error, tol, g / g_combo)

    approx_g_combo = i['a'] + s['a']
    error = Tools.mean_err(approx_g_combo, approx_g)
    assert error < tol, \
        "approx_g_combo = {}, approx_g = {}, error = {}, tol = {}".format(
            approx_g_combo, approx_g, error, tol)

    error = Tools.mean_err(g, approx_g)
    msg = []
    msg.append("f = {}".format(f))
    msg.append("g = {}".format(g))
    msg.append('approx = {}'.format(approx_g))
    msg.append("error = {}".format(error))
    msg.append("tol = {}".format(tol))
    assert error < tol, ", ".join(msg)


def test_unconfirmed_minimization(self):
    # this merely makes sure that the code doesn't throw exceptions
    # the plausibility of the result is not verified
    res = self.res[0]
    size = self.physical_sizes[0]
    substrate = Solid.PeriodicFFTElasticHalfSpace(res, 25 * self.young,
                                                  self.physical_sizes[0])
    sphere = make_sphere(self.radius, res, size)
    S = SmoothContactSystem(substrate, self.smooth, sphere)
    offset = self.sig
    disp = np.zeros(res)

    fun_jac = S.objective(offset, gradient=True)
    fun = S.objective(offset, gradient=False)

    info = []
    start = time.perf_counter()
    result_grad = minimize(fun_jac, disp.reshape(-1),
                           jac=True, method="L-BFGS-B")
    duration_g = time.perf_counter() - start
    info.append("using gradient:")
    info.append("solved in {} seconds using {} fevals".format(
        duration_g, result_grad.nfev))

    start = time.perf_counter()
    result_simple = minimize(fun, disp, method="L-BFGS-B")
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


@pytest.mark.skip("uses deprecated disp scale. TODO: is this test useful at all ?")
def test_minimize_proxy(self):
    res = self.res
    size = self.physical_sizes
    substrate = Solid.PeriodicFFTElasticHalfSpace(res, 25 * self.young,
                                                  self.physical_sizes[0])
    sphere = make_sphere(self.radius, res, size)
    S = SmoothContactSystem(substrate, self.smooth, sphere)
    offset = self.sig
    nb_scales = 5
    n_iter = np.zeros(nb_scales, dtype=int)
    n_force = np.zeros(nb_scales, dtype=float)
    for i in range(nb_scales):
        scale = 10 ** (i - 2)
        res = S.minimize_proxy(offset, disp_scale=scale, tol=1e-40,
                               gradient=True, callback=True)
        print(res.message)
        n_iter[i] = res.nit
        n_force[i] = S.compute_normal_force()
    print("N_iter = ", n_iter)
    print("N_force = ", n_force)


def test_minimize_proxy_tol(self):
    res = self.res
    size = self.physical_sizes
    substrate = Solid.PeriodicFFTElasticHalfSpace(res, 25 * self.young,
                                                  self.physical_sizes[0])
    sphere = make_sphere(self.radius, res, size)
    S = SmoothContactSystem(substrate, self.smooth, sphere)
    offset = self.sig

    res = S.minimize_proxy(offset, tol=1e-20,
                           gradient=True, callback=True)
    print(res.message)

    rep_force = np.where(
        S.interaction_force > 0, S.interaction_force, 0
        )
    alt_rep_force = -np.where(
        S.substrate.force < 0, S.substrate.force, 0
        )

    error = Tools.mean_err(rep_force, alt_rep_force)

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # CS = plt.contourf(S.interaction_force)
    # plt.colorbar(CS)
    # plt.title("interaction")
    # fig = plt.figure()
    # CS = plt.contourf(S.substrate.force)
    # plt.colorbar(CS)
    # plt.title("substrate")
    # plt.show()

    assert error < 1e-5, "error = {}".format(error)

    error = rep_force.sum() - S.compute_repulsive_force()
    assert error < 1e-5, "error = {}".format(error)

    error = (rep_force.sum() + S.compute_attractive_force() -
             S.compute_normal_force())
    assert error < 1e-5, "error = {}".format(error)


def test_LBFGSB_Hertz():
    """
    goal is that this test run the hertzian contact unsing L-BFGS-B

    For some reason it is difficult to reach the gradient tolerance

    """
    nx, ny = 64, 64
    sx, sy = 20., 20.
    R = 11.

    surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
    Es = 50.
    substrate = Solid.FreeFFTElasticHalfSpace((nx, ny), young=Es,
                                              physical_sizes=(sx, sy),
                                              fft="serial",
                                              communicator=MPI.COMM_SELF)

    interaction = Inter.Exponential(0., 0.0001)
    system = SmoothContactSystem(substrate, interaction, surface)

    gtol = 1e-6
    offset = 1.
    res = system.minimize_proxy(offset=offset, lbounds="auto",
                                options=dict(gtol=gtol, ftol=0))

    assert res.success, res.message
    print(res.message)
    print(np.max(abs(res.jac)))  # This far beyond the tolerance because
    # at the points where the
    # constraint act the gradient is allowed to not be zero

    padding_mask = np.full(substrate.nb_subdomain_grid_pts, True)
    padding_mask[substrate.topography_subdomain_slices] = False

    print(np.max(abs(res.jac[padding_mask])))
    # no force in padding area
    np.testing.assert_allclose(
        system.substrate.force[padding_mask], 0, rtol=0, atol=gtol)

    contacting_points = np.where(system.compute_gap(res.x, offset) == 0, 1.,
                                 0.)
    comp_contact_area = np.sum(contacting_points) * system.area_per_pt

    contacting_points_forces = np.where(abs(system.force) > gtol, 1., 0.)

    assert (contacting_points_forces[
                system.substrate.local_topography_subdomain_slices]
            == contacting_points).all()

    comp_normal_force = np.sum(-substrate.evaluate_force(res.x))
    from ContactMechanics.ReferenceSolutions import Hertz as Hz
    a, p0 = Hz.radius_and_pressure(Hz.normal_load(offset, R, Es), R, Es)

    np.testing.assert_allclose(
        comp_normal_force, Hz.normal_load(offset, R, Es),
        rtol=1e-3,
        err_msg="computed normal force doesn't match with hertz "
                "theory for imposed Penetration {}".format(
            offset))

    np.testing.assert_allclose(
        comp_contact_area, np.pi * a ** 2,
        rtol=5e-2,
        err_msg="Computed area doesn't match Hertz Theory")

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.colorbar(ax.pcolormesh(- system.substrate.force), label="pressure")
        plt.show(block=True)


def test_undefined_data():
    t = Topography(
        np.ma.masked_array(
            data=[[1, 2, 3],
                  [4, 5, 6]],
            mask=[[False, True, False],
                  [False, False, False]]
            ), (2, 3))

    interaction = Inter.Lj82(1., 1.)
    substrate = Solid.PeriodicFFTElasticHalfSpace((2, 3), 1, (2, 3))
    with pytest.raises(ValueError):
        SmoothContactSystem(interaction=interaction,
                            substrate=substrate, surface=t)


@pytest.mark.parametrize("s", (1., 2.))
def test_primal_hessian_product(s):
    nx = 64
    ny = 32

    sx = sy = s
    R = 10.
    Es = 50.

    interaction = Inter.RepulsiveExponential(1000, 0.5, 1., 1.)

    substrate = Solid.PeriodicFFTElasticHalfSpace((nx, ny), young=Es,
                                                  physical_sizes=(sx, sy))

    topography = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")

    system = SmoothContactSystem(substrate=substrate, surface=topography,
                                 interaction=interaction)

    obj = system.primal_objective(0, True, True)

    gaps = interaction.r_min * (0.8 + np.random.random(size=(nx, ny)))
    dgaps = interaction.r_min * (0.5 + np.random.random(size=(nx, ny)))

    _, grad = obj(gaps)

    hs = np.array([1e-2, 1e-3, 1e-4])
    rms_errors = []
    for h in hs:
        _, grad_d = obj(gaps + h * dgaps)
        dgrad = grad_d - grad
        dgrad_from_hess = system.primal_hessian_product(gaps, h * dgaps)
        rms_errors.append(
            np.sqrt(
                np.mean(
                    (dgrad_from_hess.reshape(-1) - dgrad.reshape(-1)) ** 2)))

    rms_errors = np.array(rms_errors)
    assert rms_errors[-1] / rms_errors[0] < 1.5 * (hs[-1] / hs[0]) ** 2

    if False:
        hs = np.array([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5,
                       1e-6, 1e-7])
        rms_errors = []
        for h in hs:
            _, grad_d = obj(gaps + h * dgaps)
            dgrad = grad_d - grad
            dgrad_from_hess = system.primal_hessian_product(gaps, h * dgaps)
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

    # np.testing.assert_allclose(dgrad_from_hess, dgrad)
