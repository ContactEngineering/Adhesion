#
# Copyright 2018, 2020 Antoine Sanner
#           2016, 2019-2020 Lars Pastewka
#           2019 Lintao Fang
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
Tests the potential classes
"""

import unittest
import numpy as np

from Adhesion.Interactions import LJ93
from Adhesion.Interactions import VDW82
from Adhesion.Interactions.cutoffs import LinearCorePotential

from Adhesion.Interactions import Exponential, Morse
from Adhesion.Interactions import RepulsiveExponential
from Adhesion.Interactions import PowerLaw
from Adhesion.Interactions import SmoothPotential

import ContactMechanics.Tools as Tools

from test.Interactions.lj93_ref_potential import (
    V as LJ_ref_V, dV as LJ_ref_dV,
    d2V as LJ_ref_ddV
    )
from test.Interactions.lj93smooth_ref_potential import (
    V as LJs_ref_V, dV as LJs_ref_dV,
    d2V as LJs_ref_ddV
    )

import pytest
from NuMPI import MPI

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="tests only serial funcionalities,"
                                " please execute with pytest")


class PotentialTest(unittest.TestCase):
    tol = 1e-14

    def setUp(self):
        self.eps = 1 + np.random.rand()
        self.sig = 3 + np.random.rand()
        self.gam = (5 + np.random.rand())
        self.rcut = 2.5 * self.sig + np.random.rand()
        self.r = np.arange(.65, 3, .01) * self.sig

    def test_LJReference(self):
        """ compare lj93 to reference implementation
        """
        V, dV, ddV = LJ93(
            self.eps, self.sig).cutoff(self.rcut).evaluate(
            self.r, potential=True, gradient=True, curvature=True)
        V_ref = LJ_ref_V(self.r, self.eps, self.sig, self.rcut)
        dV_ref = - LJ_ref_dV(self.r, self.eps, self.sig, self.rcut)
        ddV_ref = LJ_ref_ddV(self.r, self.eps, self.sig, self.rcut)

        err_V = ((V - V_ref) ** 2).sum()
        err_dV = ((dV + dV_ref) ** 2).sum()
        err_ddV = ((ddV - ddV_ref) ** 2).sum()
        error = err_V + err_dV + err_ddV

        self.assertTrue(error < self.tol)

    def test_LJsmoothReference(self):
        """ compare lj93smooth to reference implementation
        """
        smooth_pot = LJ93(self.eps, self.sig).spline_cutoff(self.gam)
        rc1 = smooth_pot.r_t
        rc2 = smooth_pot.cutoff_radius
        V, dV, ddV = smooth_pot.evaluate(
            self.r, potential=True, gradient=True, curvature=True)
        V_ref = LJs_ref_V(self.r, self.eps, self.sig, rc1, rc2)
        dV_ref = LJs_ref_dV(self.r, self.eps, self.sig, rc1, rc2)
        ddV_ref = LJs_ref_ddV(self.r, self.eps, self.sig, rc1, rc2)

        err_V = ((V - V_ref) ** 2).sum()
        err_dV = ((dV - dV_ref) ** 2).sum()
        err_ddV = ((ddV - ddV_ref) ** 2).sum()
        error = err_V + err_dV + err_ddV
        self.assertTrue(
            error < self.tol,
            ("Error = {}, (tol = {})\n"
             "   err_V = {}, err_dV = {}, err_ddV = {}").format(
                error, self.tol, err_V, err_dV, err_ddV))

    def test_LJsmoothMinReference(self):
        """
        compare lj93smoothmin to reference implementation (where it applies).
        """
        smooth_pot = LJ93(self.eps, self.sig
                          ).spline_cutoff(self.gam
                                          ).linearize_core()
        # cutoff radii of the splice cutoff
        rc1 = smooth_pot.parent_potential.r_t
        rc2 = smooth_pot.parent_potential.cutoff_radius
        V, dV, ddV = smooth_pot.evaluate(
            self.r, potential=True, gradient=True, curvature=True)
        V_ref = LJs_ref_V(self.r, self.eps, self.sig, rc1, rc2)
        dV_ref = LJs_ref_dV(self.r, self.eps, self.sig, rc1, rc2)
        ddV_ref = LJs_ref_ddV(self.r, self.eps, self.sig, rc1, rc2)

        err_V = ((V - V_ref) ** 2).sum()
        err_dV = ((dV - dV_ref) ** 2).sum()
        err_ddV = ((ddV - ddV_ref) ** 2).sum()
        error = err_V + err_dV + err_ddV
        self.assertTrue(
            error < self.tol,
            ("Error = {}, (tol = {})\n"
             "   err_V = {}, err_dV = {}, err_ddV = {}").format(
                error, self.tol, err_V, err_dV, err_ddV))

    #     def test_triplePlot(self):
    #         lj_pot = LJ93(self.eps, self.sig, self.rcut)
    #         gam = float(-lj_pot.evaluate(lj_pot.r_min)[0])
    #         smooth_pot = LJ93smooth(self.eps, self.sig, gam)
    #         min_pot = LJ93smoothMin(self.eps, self.sig, gam, lj_pot.r_min)
    #         plots = (("LJ", lj_pot),
    #                  ("smooth", smooth_pot),
    #                  ("min", min_pot))
    #         import matplotlib.pyplot as plt
    #         plt.figure()
    #         r = self.r
    #         for name, pot in plots:
    #             V, dV, ddV = pot.evaluate(r)
    #             plt.plot(r, V, label=name)
    #         plt.legend(loc='best')
    #         plt.grid(True)
    #         plt.show()

    def test_spline_cutoff_sanity(self):
        """ make sure spline_cutoff rejects common bad input
        """
        self.assertRaises(SmoothPotential.PotentialError,
                          LJ93(self.eps, self.sig).spline_cutoff,
                          -self.gam)

    def test_LJ_gradient(self):
        pot = LJ93(self.eps, self.sig).cutoff(self.rcut)
        x = np.random.random(3) - .5 + self.sig
        V, g, ddV = pot.evaluate(x, gradient=True)

        delta = self.sig / 1e5
        approx_g = Tools.evaluate_gradient(
            lambda x: pot.evaluate(x)[0].sum(),
            x, delta)
        tol = 1e-8
        error = Tools.mean_err(g, approx_g)
        msg = []
        msg.append("g = {}".format(g))
        msg.append('approx = {}'.format(approx_g))
        msg.append("error = {}".format(error))
        msg.append("tol = {}".format(tol))
        self.assertTrue(error < tol, ", ".join(msg))

    def test_LJsmooth_gradient(self):
        pot = LJ93(self.eps, self.sig).spline_cutoff(self.gam)
        x = np.random.random(3) - .5 + self.sig
        V, dV, ddV = pot.evaluate(x, gradient=True)
        f = V.sum()
        g = dV

        delta = self.sig / 1e5
        approx_g = Tools.evaluate_gradient(
            lambda x: pot.evaluate(x)[0].sum(),
            x, delta)
        tol = 1e-8
        error = Tools.mean_err(g, approx_g)
        msg = []
        msg.append("f = {}".format(f))
        msg.append("g = {}".format(g))
        msg.append('approx = {}'.format(approx_g))
        msg.append("error = {}".format(error))
        msg.append("tol = {}".format(tol))
        self.assertTrue(error < tol, ", ".join(msg))

    def test_single_point_eval(self):
        pot = LJ93(self.eps, self.sig, self.gam)
        r_m = pot.r_min
        curvature = pot.evaluate(r_m, potential=False,  # noqa: F841
                                 gradient=False,
                                 curvature=True)[2]  # noqa: F841

    def test_ad_hoc(self):
        # 'Potential 'lj9-3smooth', ε = 1.7294663266397667,
        # σ = 3.253732668164946, γ = 5.845648523044794, r_t = r_min' failed.
        # Please check whether the inputs make sense
        eps = 1.7294663266397667
        sig = 3.253732668164946
        gam = 5.845648523044794
        smooth_pot = LJ93(eps, sig).spline_cutoff(gam)

        rc1 = smooth_pot.r_t
        rc2 = smooth_pot.cutoff_radius
        V, dV, ddV = smooth_pot.evaluate(
            self.r, potential=True, gradient=True, curvature=True)
        V_ref = LJs_ref_V(self.r, eps, sig, rc1, rc2)
        dV_ref = LJs_ref_dV(self.r, eps, sig, rc1, rc2)
        ddV_ref = LJs_ref_ddV(self.r, eps, sig, rc1, rc2)

        err_V = ((V - V_ref) ** 2).sum()
        err_dV = ((dV - dV_ref) ** 2).sum()
        err_ddV = ((ddV - ddV_ref) ** 2).sum()
        error = err_V + err_dV + err_ddV
        self.assertTrue(
            error < self.tol,
            ("Error = {}, (tol = {})\n"
             "   err_V = {}, err_dV = {}, err_ddV = {}").format(
                error, self.tol, err_V, err_dV, err_ddV))

    def test_vanDerWaalsSimple(self):
        # reproduces the graph in
        # http://dx.doi.org/10.1103/PhysRevLett.111.035502
        # originally visally compared then just regression checking
        c_sr = 2  # 2.1e-78
        hamaker = 4  # 68.1e-78
        vdw = VDW82(c_sr, hamaker)
        r_min = vdw.r_min
        V_min, dV_min, ddV_min = vdw.evaluate(r_min, True, True, True)
        vdws = VDW82(c_sr, hamaker).spline_cutoff(r_t=2.5)  # noqa: F841
        vdwm = VDW82(c_sr, hamaker  # noqa: F841
                     ).spline_cutoff().linearize_core(r_ti=1.95)
        # r = np.arange(0.5, 2.01, .005) * r_min
        # import matplotlib.pyplot as plt
        # plt.figure()
        # pot = vdw.evaluate(r)[0]
        # print()
        # print("  V_min = {}".format(  V_min))
        # print(" dV_min = {}".format( dV_min))
        # print("ddV_min = {}".format(ddV_min))
        # print("transition = {}".format(vdws.r_t))
        # bot = 1.1*V_min
        # plt.plot(r, pot,label='simple')
        # plt.plot(r, vdws.evaluate(r)[0],label='smooth')
        # plt.plot(r, vdwm.evaluate(r)[0],label='minim')
        # plt.scatter(vdw.r_min, V_min, marker='+')
        # plt.ylim(bottom=bot, top=0)
        # plt.legend(loc='best')
        # plt.show()

    def test_vanDerWaals(self):
        # reproduces the graph in
        # http://dx.doi.org/10.1103/PhysRevLett.111.035502
        # originally visally compared then just regression checking

        c_sr = 2.1e-78
        hamaker = 68.1e-21
        pots = (("vdW", VDW82(c_sr, hamaker)),  # noqa: F841
                ("smooth", VDW82(c_sr, hamaker).spline_cutoff()),
                ("min", VDW82(c_sr, hamaker).spline_cutoff().linearize_core()))
        # r = np.arange(0.25, 2.01, .01) * 1e-9  #
        # import matplotlib.pyplot as plt
        # plt.figure()
        # bot = None
        # for name, t_h_phile in pots:
        #     pot = t_h_phile.evaluate(r)[0]
        #     if bot is None:
        #         bot = 1.1*t_h_phile.evaluate(t_h_phile.r_min)[0]
        #     plt.plot(r, pot,label=name)
        # plt.ylim(bottom=bot, top=0)
        # plt.legend(loc='best')
        # plt.show()

    def test_SimpleSmoothLJ(self):
        eps = 1.7294663266397667
        sig = 3.253732668164946
        pot = LJ93(eps, sig).parabolic_cutoff(3 * sig)  # noqa: F841

        # import matplotlib.pyplot as plt
        # plt.figure()
        # r = np.linspace(pot.r_min*.7, pot.cutoff_radius*1.1, 100)
        # p = pot.evaluate(r)[0]
        # plt.plot(r, p)
        # pois = [pot.cutoff_radius, pot.r_min]
        # plt.scatter(pois, pot.evaluate(pois)[0])
        # plt.ylim(bottom=1.1*p.min(), top=-.3*p.min())
        # plt.grid(True)
        # plt.legend(loc='best')
        # plt.show()

    def test_SimpleSmoothVDW(self):
        hamaker = 68.1e-21
        c_sr = 2.1e-78 * 1e-6
        r_c = 10e-10  # noqa: F841
        pot = VDW82(c_sr, hamaker).parabolic_cutoff(10e-10)  # noqa: F841

        # import matplotlib.pyplot as plt
        # r = np.linspace(pot.r_min*.7, pot.cutoff_radius*1.1, 1000)
        # ps = pot.evaluate(r, pot=True, forces=True)
        #
        # for i, name in enumerate(('potential', 'force')):
        #     plt.figure()
        #     p = ps[i]
        #     plt.plot(r, p, label=name)
        #
        #     pois = [pot.cutoff_radius, pot.r_min]
        #     plt.scatter(pois, pot.evaluate(pois, pot=True, forces=True)[i])
        #     plt.ylim(bottom=1.1*p.min(), top=-.3*p.min())
        #     plt.grid(True)
        #     plt.legend(loc='best')
        # plt.show()

    #    @unittest.expectedFailure
    #    def test_VDW82SimpleSmoothMin(self):
    #        hamaker = 68.1e-21
    #        c_sr = 2.1e-78 * 1e-6
    #        cutoff_radius = 10e-10
    #        potref = VDW82SimpleSmooth(c_sr, hamaker, 10e-10)
    #        pot = VDW82SimpleSmoothMin(c_sr, hamaker,
    #           r_ti = potref.r_min / 2, cutoff_radius= 10e-10)

    # import matplotlib.pyplot as plt
    # r = np.linspace(pot.r_min*.7, pot.cutoff_radius*1.1, 1000)
    # ps = pot.evaluate(r, pot=True, forces=True)
    #
    # for i, name in enumerate(('potential', 'force')):
    #     plt.figure()
    #     p = ps[i]
    #     plt.plot(r, p, label=name)
    #
    #     pois = [pot.cutoff_radius, pot.r_min]
    #     plt.scatter(pois, pot.evaluate(pois, pot=True, forces=True)[i])
    #     plt.ylim(bottom=1.1*p.min(), top=-.3*p.min())
    #     plt.grid(True)
    #     plt.legend(loc='best')
    # plt.show()

    def test_LinearCorePotential(self):
        w = 3
        z0 = 0.5
        r_ti = 0.4 * z0

        refpot = VDW82(w * z0 ** 8 / 3, 16 * np.pi * w * z0 ** 2)

        pot = LinearCorePotential(refpot, r_ti=r_ti)
        z = [0.8 * r_ti, r_ti, 1.5 * r_ti]
        for zi in z:
            zi = np.array(zi)
            np.testing.assert_allclose(
                np.array(pot.pipeline()[0].evaluate(zi, True, True, True)),
                np.array(refpot.evaluate(zi, True, True, True, )))
            if zi >= r_ti:
                np.testing.assert_allclose(
                    pot.evaluate(zi, True, True, True, ),
                    refpot.evaluate(zi, True, True, True, ))

        if False:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3)

            z = np.linspace(0.8 * r_ti, 2 * z0)
            for poti in [refpot, pot]:
                p, f, c = poti.evaluate(z, True, True, True)
                ax[0].plot(z, p)
                ax[1].plot(z, f)
                ax[2].plot(z, c)
            fig.savefig("test_LinearCorePotential.png")

    def test_LinearCoreSimpleSmoothPotential(self):
        w = 3
        z0 = 0.5
        r_ti = 0.4 * z0
        r_c = 10 * z0

        refpot = VDW82(w * z0 ** 8 / 3, 16 * np.pi * w * z0 ** 2)

        smoothpot = refpot.parabolic_cutoff(cutoff_radius=r_c)

        pot = LinearCorePotential(smoothpot, r_ti=r_ti)

        # assert pot.cutoff_radius == smoothpot.cutoff_radius,
        #   "{0.cutoff_radius}{1.cutoff_radius}"
        assert pot.r_infl == smoothpot.r_infl
        assert pot.r_min == smoothpot.r_min

        z = [0.8 * r_ti, r_ti, 1.5 * r_ti, r_c, 1.1 * r_c]
        for zi in z:
            if zi >= r_ti:
                np.testing.assert_allclose(
                    np.array(pot.evaluate(zi, True, True, True, )).reshape(-1),
                    np.array(
                        smoothpot.evaluate(zi, True, True, True, )).reshape(
                        -1))
            if zi >= r_c:
                assert pot.evaluate(zi, True, True, True, ) == (
                    0, 0, 0), "Potential nonzero outside of cutoff"

        if False:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3)

            z = np.linspace(0.8 * r_ti, 2 * z0)
            for poti in [refpot, pot]:
                p, f, c = poti.evaluate(z, True, True, True)
                ax[0].plot(z, p)
                ax[1].plot(z, f)
                ax[2].plot(z, c)
            fig.savefig("test_LinearCoreSimpleSmoothPotential.png")

    def test_Exponential(self):
        r = np.linspace(-10, 10, 1001)
        pot = Exponential(1.0, 1.0)
        V, dV, ddV = pot.evaluate(r)
        self.assertTrue((V < 0.0).all())
        self.assertTrue((dV > 0.0).all())
        self.assertTrue((ddV < 0.0).all())
        dV_num = np.diff(V) / np.diff(r)
        ddV_num = np.diff(dV_num) / (r[1] - r[0])
        self.assertLess(abs((dV[:-1] + dV[1:]) / 2 - dV_num).max(), 1e-4)
        self.assertLess(abs(ddV[1:-1] - ddV_num).max(), 1e-2)

    def test_RepulsiveExponential(self):
        ns = np.array([10, 100, 1000, 10000])
        errordV = np.zeros_like(ns, dtype=float)
        errorddV = np.zeros_like(ns, dtype=float)

        for i, n in enumerate(ns):
            r = np.linspace(-.1, 10, n + 1)
            pot = RepulsiveExponential(0.5, 0.5, 1.3, 1.)
            V, dV, ddV = pot.evaluate(r, potential=True,
                                      gradient=True,
                                      curvature=True)
            dV_num = np.diff(V) / np.diff(r)
            ddV_num = np.diff(dV_num) / (r[1] - r[0])
            errordV[i] = abs((dV[:-1] + dV[1:]) / 2 - dV_num).max()
            errorddV[i] = abs(ddV[1:-1] - ddV_num).max()

        if False:
            import matplotlib.pyplot as plt
            plt.loglog(ns, errordV, label="dV")
            plt.loglog(ns, errorddV, label="ddV")
            plt.plot((ns[0], ns[-1]), np.array((ns[0], ns[-1])) ** (-2) * 10,
                     c="gray", label="$10 / n^2$")
            plt.plot((ns[0], ns[-1]), np.array((ns[0], ns[-1])) ** (-2) * 100,
                     c="gray", label="$100 / n^2$")
            plt.xlabel("# points")
            plt.ylabel(r"error = $|analytical - numerical|_{\infty}$")
            plt.legend()
            plt.show(block=True)

        errordV = errordV / errordV[0] * ns ** 2 / ns[0] ** 2
        errorddV = errorddV / errorddV[0] * ns ** 2 / ns[0] ** 2

        if False:
            plt.loglog(ns, errordV - 1, label="dV")
            plt.loglog(ns, errorddV - 1, label="ddV")
            plt.legend()
            plt.xlabel("# points n")
            plt.ylabel("(error * n**2) / (error * n**2)(0)")
            plt.show(block=True)
        assert (errordV < 10).all()
        assert (errorddV < 10).all()


# TODO: is there a smart way to do regression tests ?
# def test_lj93smoothmin_regression(self):
#
#     import copy
#
#     print("########################################################################")
#     z = np.random.random((200)) *10 -1
#
#
#     mask = np.zeros_like(z)
#     mask[-2:] = True
#     mask[0]=True
#     z = np.ma.masked_array(z, mask=mask)
#
#
#     print(np.max(z))
#     print(np.min(z))
#
#     for params in [dict(epsilon=1.7294663266397667,
#       sigma=3.253732668164946, gamma = 2, r_ti=0.5, r_t_ls = 3),
#                    dict(epsilon=1, sigma=2, gamma=5)]:
#         new = LJ93smoothMin(**params)
#         old = LJ93smoothMin_old(**params)
#
#         z[1] = params["sigma"]
#         z[2] = 0
#         z[3] = 0.5
#         z[4] = 3.
#         z[5] = float("inf")
#         z[10] = LJ93(params["epsilon"], params["sigma"]).r_min
#         z[11] = old.r_infl
#         z[12] = old.r_t
#         z[13] = old.r_ti
#         z[14] = old.cutoff_radius
#
#         Vnew, dVnew, ddVnew = \
#           new.evaluate(z, True, True, True, area_scale=1.5)
#         Vold, dVold, ddVold = \
#           old.evaluate(z, True, True, True, area_scale=1.5)
#
#         np.testing.assert_allclose(Vnew, Vold, rtol=1e-17)
#         np.testing.assert_allclose(dVnew, dVold, rtol=1e-17)
#         np.testing.assert_allclose(ddVnew, ddVold, rtol=1e-17)
#
#         new.compute(z, True, True, True, area_scale=1.5)
#         old.compute(z, True, True, True, area_scale=1.5)
#
#         copied_new = copy.deepcopy(new)
#
#         print(old.lin_part)
#         print(new.lin_part)
#
#         for key in old.__dict__.keys():
#             print(key)
#
#             def allarraycomp(a, b, msg):
#                 if a.dtype == object:
#                     for i, j in zip(a, b):
#                         allarraycomp(i, j, msg=msg)
#                 else:
#                     np.testing.assert_allclose(a, b, err_msg=msg)
#
#             def asserteverything(a, b, msg):
#                 if hasattr(a, "dtype"):
#                     allarraycomp(a,b,msg)
#                 else:
#                     self.assertEqual(a,b, msg = msg)
#
#             asserteverything(getattr(new, key), getattr(old, key),
#             msg="problem for {} : newval: {}, "
#             "oldval: {}".format(key, getattr(new, key), getattr(old, key)))
#             asserteverything(getattr(copied_new, key),
#             getattr(old, key),
#             msg="problem for {} : newval: {},
#             oldval: {}".format(key, getattr(new, key), getattr(old, key)))

@pytest.mark.parametrize(
    "pot_creation",
    [
        "LJ93(eps, sig)",
        # 'LJ93(eps, sig).parabolic_cutoff(3*sig)', # TODO: issue #5
        'LJ93(eps, sig).spline_cutoff()',
        'LJ93(eps,  sig).spline_cutoff(r_t="inflection")',
        'LJ93(eps, sig).spline_cutoff(r_t=LJ93(eps, sig).r_infl*1.05)',
        'VDW82(c_sr,  hamaker).spline_cutoff()',
        'VDW82(c_sr,  hamaker).spline_cutoff(r_t="inflection")',
        'VDW82(c_sr,  hamaker)'
        '.spline_cutoff(r_t=VDW82(c_sr, hamaker).r_infl * 1.05)',
        # 'VDW82(c_sr, hamaker)'
        # '.parabolic_cutoff(cutoff_radius=VDW82(c_sr, hamaker).r_infl * 2)',
        # # TODO: issue #5
        # 'VDW82(c_sr, hamaker)'
        # '.parabolic_cutoff(VDW82(c_sr, hamaker).r_infl * 2)'
        # '.linearize_core(VDW82(c_sr, hamaker).r_min * 0.8)',# TODO: issue #5
        "VDW82(c_sr, hamaker)",
        ]
    )
def test_linearize_core_isinstance(pot_creation):
    eps = 1.7294663266397667  # noqa: F841
    sig = 3.253732668164946  # noqa: F841

    c_sr = 2.1e-78  # noqa: F841
    hamaker = 68.1e-21  # noqa: F841

    basepot = eval(pot_creation)
    linpot = basepot.linearize_core()
    assert isinstance(linpot, LinearCorePotential)


def test_LinearCorePotential_hardness():
    w = 1
    z0 = 1
    hardness = 5.

    refpot = VDW82(w * z0 ** 8 / 3, 16 * np.pi * w * z0 ** 2)
    pot = refpot.linearize_core(hardness=hardness)

    r_ti = pot.r_ti

    assert abs(- pot.evaluate(r_ti, True, True)[1] - hardness) < 1e-10
    #           The gradient is minus the force
    "{}".format(LinearCorePotential)
    if False:
        import matplotlib.pyplot as plt
        import subprocess
        fig, ax = plt.subplots(3)

        z = np.linspace(0.9 * r_ti, 2 * z0)
        for poti in [refpot, pot]:
            p, f, c = poti.evaluate(z, True, True, True)
            ax[0].plot(z, p)
            ax[1].plot(z, f)
            ax[2].plot(z, c)
        for a in ax:
            a.axvline(r_ti)
            a.axvline(r_ti)
            a.grid()

        fig.savefig("test_LinearCorePotential_hardness.png")
        subprocess.call("open test_LinearCorePotential_hardness.png",
                        shell=True)


def test_linearize_core_hardness_infinite():
    """
    the cutoff radius matching the hardness value is found using an iterative
    rootfinding algorithm

    for very steep potentials the potential evaluation can produce infinite
    values during this process.

    here we test the mechanism that can handle that
    """
    pot = RepulsiveExponential(2, 0.75, 1000, 1.5)
    pot.linearize_core(hardness=1000000000000000)


@pytest.mark.parametrize(
    "pot_creation",
    [
        "LJ93(eps, sig)",
        'LJ93(eps, sig)',
        # 'LJ93(eps, sig).parabolic_cutoff(3*sig)', # TODO: issue #5
        'LJ93(eps, sig).spline_cutoff()',
        'LJ93(eps, sig).spline_cutoff().linearize_core()',
        'LJ93(eps,  sig).spline_cutoff(r_t="inflection")',
        'LJ93(eps, sig).spline_cutoff(r_t="inflection").linearize_core()',
        'LJ93(eps, sig).spline_cutoff(r_t=LJ93(eps, sig).r_infl*1.05)',
        'LJ93(eps, sig)'
        '.spline_cutoff(r_t=LJ93(eps, sig).r_infl*1.05).linearize_core()',
        'LJ93(eps, sig)'
        '.spline_cutoff(LJ93(eps, sig).r_infl * 2)'
        '.linearize_core(LJ93(eps, sig).r_min * 0.8)',
        'VDW82(c_sr, hamaker)',
        'VDW82(c_sr,  hamaker).spline_cutoff()',
        'VDW82(c_sr,  hamaker).spline_cutoff().linearize_core()',
        'VDW82(c_sr,  hamaker).spline_cutoff(r_t="inflection")',
        'VDW82(c_sr,  hamaker)'
        '.spline_cutoff(r_t="inflection")'
        '.linearize_core()',
        'VDW82(c_sr,  hamaker)'
        '.spline_cutoff(r_t=VDW82(c_sr, hamaker).r_infl * 1.05)',
        # 'VDW82(c_sr, hamaker)
        # .parabolic_cutoff(cutoff_radius=VDW82(c_sr, hamaker).r_infl * 2)',
        # TODO: issue #5
        # 'VDW82(c_sr, hamaker)
        # .parabolic_cutoff(VDW82(c_sr, hamaker).r_infl * 2)
        # .linearize_core(VDW82(c_sr, hamaker).r_min * 0.8)',# TODO: issue #5
        'RepulsiveExponential(1., 0.5, 1., 1.)',
        "VDW82(c_sr, hamaker)",
        ]
    )
def test_rinfl(pot_creation):
    """
    Test if the inflection point calculated analyticaly is really a
    signum change of the second dericative
    """

    eps = 1.7294663266397667  # noqa: F841
    sig = 3.253732668164946  # noqa: F841

    c_sr = 2.1e-78  # noqa: F841
    hamaker = 68.1e-21  # noqa: F841
    pot = eval(pot_creation)
    # tests wether the
    assert (pot.evaluate(pot.r_infl * (1 - 1e-4), True, True, True)[2]
            * pot.evaluate(pot.r_infl * (1 + 1e-4), True, True, True)[2] < 0)


@pytest.mark.parametrize(
    "pot_creation", [
        'LJ93(eps, sig)',
        'LJ93(eps, sig).parabolic_cutoff(3*sig)',
        'LJ93(eps, sig).spline_cutoff()',
        'LJ93(eps, sig).spline_cutoff().linearize_core()',
        'LJ93(eps,  sig).spline_cutoff(r_t="inflection")',
        'LJ93(eps, sig).spline_cutoff(r_t="inflection").linearize_core()',
        'LJ93(eps, sig).spline_cutoff(r_t=LJ93(eps, sig).r_infl*1.05)',
        'LJ93(eps, sig)'
        '.spline_cutoff(r_t=LJ93(eps, sig).r_infl*1.05)'
        '.linearize_core()',
        'LJ93(eps, sig)'
        '.spline_cutoff(LJ93(eps, sig).r_infl * 2)'
        '.linearize_core(LJ93(eps, sig).r_min * 0.8)',
        'VDW82(c_sr, hamaker)',
        'VDW82(c_sr, hamaker).spline_cutoff()',
        'VDW82(c_sr, hamaker).spline_cutoff().linearize_core()',
        'VDW82(c_sr, hamaker).spline_cutoff(r_t="inflection")',
        'VDW82(c_sr, hamaker).spline_cutoff(r_t="inflection")'
        '.linearize_core()',
        'VDW82(c_sr, hamaker)'
        '.spline_cutoff(r_t=VDW82(c_sr, hamaker).r_infl * 1.05)',
        'VDW82(c_sr, hamaker)'
        '.parabolic_cutoff(cutoff_radius=VDW82(c_sr, hamaker).r_infl * 2)',
        'VDW82(c_sr, hamaker)'
        '.parabolic_cutoff(VDW82(c_sr, hamaker).r_infl * 2)'
        '.linearize_core(VDW82(c_sr, hamaker).r_min * 0.8)',
        'RepulsiveExponential(1., 0.5, 1., 1.)',
        'Exponential(sig, eps)',
        'PowerLaw(sig, eps, 3)',
        'RepulsiveExponential(1., 0.5, 1., 1.)',
        ])
def test_deepcopy(pot_creation):
    w = 3  # noqa: F841
    z0 = 0.5
    r_ti = 0.4 * z0  # noqa: F841
    r_c = 10 * z0  # noqa: F841
    import copy

    eps = 1.7294663266397667  # noqa: F841
    sig = 3.253732668164946  # noqa: F841

    c_sr = 2.1e-78  # noqa: F841
    hamaker = 68.1e-21  # noqa: F841

    pot = eval(pot_creation)
    copied_potential = copy.deepcopy(pot)

    if pot.r_min is not None:
        z = np.array([
            pot.r_min * 1e-4,
            pot.r_min * (1 - 1e-4),
            pot.r_min * (1 + 1e-4),
            pot.r_infl * (1 - 1e-4),
            pot.r_infl * (1 + 1e-4),
            pot.r_infl * 1e3,
            pot.r_infl * 1e3,
            pot.r_infl * 1e3
            ])
    else:
        z = np.linspace(0, 3)
    mask = np.zeros_like(z)
    mask[-2:] = True
    z = np.ma.masked_array(z, mask=mask)

    print("testing {}".format(pot))
    pot.evaluate(z, True, True, True)
    copied_potential.evaluate(z, True, True, True)

    np.testing.assert_allclose(
        copied_potential.evaluate(z, True, True, True),
        pot.evaluate(z, True, True, True))

    if hasattr(pot, "parent_potential"):
        # assert parent potential
        # has also been copied
        assert pot.parent_potential is not copied_potential.parent_potential


@pytest.mark.parametrize("pot_creation", [
    'LJ93(eps, sig)',
    'LJ93(eps, sig).parabolic_cutoff(3*sig)',
    'LJ93(eps, sig).spline_cutoff()',
    'LJ93(eps, sig).spline_cutoff().linearize_core()',
    'LJ93(eps,  sig).spline_cutoff(r_t="inflection")',
    'LJ93(eps, sig).spline_cutoff(r_t="inflection").linearize_core()',
    'LJ93(eps, sig).spline_cutoff(r_t=LJ93(eps, sig).r_infl*1.05)',
    'LJ93(eps, sig)'
    '.spline_cutoff(r_t=LJ93(eps, sig).r_infl*1.05).linearize_core()',
    'LJ93(eps, sig)'
    '.spline_cutoff(LJ93(eps, sig).r_infl * 2)'
    '.linearize_core(LJ93(eps, sig).r_min * 0.8)',
    'VDW82(c_sr, hamaker)',
    'VDW82(c_sr,  hamaker).spline_cutoff()',
    'VDW82(c_sr,  hamaker).spline_cutoff().linearize_core()',
    'VDW82(c_sr,  hamaker).spline_cutoff(r_t="inflection")',
    'VDW82(c_sr,  hamaker).spline_cutoff(r_t="inflection").linearize_core()',
    'VDW82(c_sr,  hamaker)'
    '.spline_cutoff(r_t=VDW82(c_sr, hamaker).r_infl * 1.05)',
    'VDW82(c_sr, hamaker)'
    '.parabolic_cutoff(cutoff_radius=VDW82(c_sr, hamaker).r_infl * 2)',
    'VDW82(c_sr, hamaker)'
    '.parabolic_cutoff(VDW82(c_sr, hamaker).r_infl * 2)'
    '.linearize_core(VDW82(c_sr, hamaker).r_min * 0.8)',
    'RepulsiveExponential(1., 0.5, 1., 1.)',
    'Exponential(sig, eps)',
    'PowerLaw(sig, eps, 3)'
    ])
def test_max_tensile(pot_creation):
    eps = 1.7294663266397667  # noqa: F841
    sig = 3.253732668164946  # noqa: F841

    c_sr = 2.1e-78  # noqa: F841
    hamaker = 68.1e-21  # noqa: F841

    pot = eval(pot_creation)
    en1 = np.isscalar(pot.max_tensile)

    # print("{}".format(en1))

    assert en1


@pytest.mark.parametrize("pot_creation", [
    # Not supported on the Lennard Jones potentials, see issue 6
    # 'LJ93(eps, sig)',
    # 'LJ93SimpleSmooth(eps, sig, 3*sig)',
    # 'LJ93smooth(eps, sig)',
    # 'LJ93smoothMin(eps, sig)',
    # 'LJ93smooth(eps,  sig, r_t="inflection")',
    # 'LJ93smoothMin(eps, sig, r_t_ls="inflection")',
    # 'LJ93smooth(eps,  sig, r_t=LJ93(eps, sig).r_infl*1.05)',
    # 'LJ93smoothMin(eps,  sig, r_t_ls=LJ93(eps, sig).r_infl*1.05)',
    # 'LJ93SimpleSmoothMin(eps, sig, LJ93(eps, sig).r_infl * 2,'
    # 'LJ93(eps, sig).r_min * 0.8)',
    # 'Lj82(w, z0)',
    # 'VDW82(c_sr, hamaker)',
    # 'VDW82smooth(c_sr,  hamaker)',
    # 'VDW82smoothMin(c_sr,  hamaker)',
    # 'VDW82smooth(c_sr,  hamaker, r_t="inflection")',
    # 'VDW82smoothMin(c_sr,  hamaker, r_t_ls="inflection")',
    # 'VDW82smooth(c_sr,  hamaker, r_t=VDW82(c_sr, hamaker).r_infl * 1.05)',
    # 'VDW82smoothMin(c_sr,  hamaker, '
    # 'r_t_ls=VDW82(c_sr, hamaker).r_infl*1.05)',
    # 'VDW82SimpleSmooth(c_sr, hamaker, '
    #  'cutoff_radius=VDW82(c_sr, hamaker).r_infl * 2)',
    'RepulsiveExponential(0.5 * sig, 0.5* eps , sig, eps)',
    'Exponential(sig, eps)',
    'PowerLaw(sig, eps, 3)'
    ])
def test_max_tensile_array(pot_creation):
    eps = 1.7
    sig = np.array([[3., 3.2], [2.6, 4.]])
    z0 = eps  # noqa: F841
    w = sig  # noqa: F841

    # c_sr = w * z0 ** 8 / 3
    # hamaker =  16 * np.pi * w * z0 ** 2
    pot = eval(pot_creation)

    assert pot.max_tensile.shape == sig.shape


@pytest.mark.parametrize("pot_creation", [
    'Exponential(w, rho,)',
    'PowerLaw(w, 3*rho, 3)'
    ])
def test_work_range_array(pot_creation):
    """
    tests that it is possible to change interaction
    range and work of adhesion at the same time
    """

    x = np.linspace(0, 1)
    dw = 0.5
    work_fluctuation = dw * np.cos(np.pi * x)
    work_factor = 1 + work_fluctuation

    # base values around which the work will fluctuate
    w0 = 1.
    rho0 = 0.1
    sigma = w0 / rho0

    w = w0 * work_factor  # noqa: F841
    rho = rho0 * np.sqrt(work_factor)  # noqa: F841
    interaction = eval(pot_creation)

    np.testing.assert_allclose(-interaction.max_tensile,
                               sigma * np.sqrt(work_factor))

    # test evaluation

    interaction.evaluate(np.random.uniform(0, 2, size=x.shape), True, True,
                         True)


@pytest.mark.parametrize("cutoff_procedure", [
    "pot",
    "pot.parabolic_cutoff(cutoff_radius)",
    "pot.parabolic_cutoff(cutoff_radius).linearize_core(r_t_lin)",
    "pot.linearize_core(r_t_lin).parabolic_cutoff(cutoff_radius)",
    "pot.linearize_core(r_t_lin)",
    ])
def test_cutoff_derivatives(cutoff_procedure):
    eps = 1.
    sig = 1.

    r_t_lin = 0.5  # noqa: F841
    cutoff_radius = 1.5

    pot = LJ93(eps, sig)  # noqa: F841
    cutoffpot = eval(cutoff_procedure)

    def approx_fprime(x0, fun, eps):
        return (fun(x0 + eps) - fun(x0)) / eps

    for x0 in [1., 1.1, 0.9 * cutoff_radius, 0.999 * cutoff_radius,
               cutoff_radius, 1.1 * cutoff_radius]:
        eps = 1e-8
        # forward finite differece wrt x0,
        # or central fintite differences wrt x0+eps / 2
        numder = approx_fprime(x0, lambda x: cutoffpot.evaluate(x)[0], eps)
        analder = cutoffpot.evaluate(x0 + eps / 2, True, True)[1]

        assert abs(numder - analder) < 1e-7

        numder = approx_fprime(x0,
                               lambda x: cutoffpot.evaluate(x, True, True)[1],
                               eps)
        analder = cutoffpot.evaluate(x0 + eps / 2, True, True, True)[2]
        assert abs(numder - analder) < 1e-7


@pytest.mark.parametrize("r_t", [None, "inflection"])
def test_spline_cutoff_derivatives(r_t):
    eps = 1.
    sig = 1.

    r_t_lin = 0.5  # noqa: F841
    r_c = 1.5  # noqa: F841

    pot = LJ93(eps, sig)
    cutoffpot = pot.spline_cutoff(r_t=r_t)

    def approx_fprime(x0, fun, eps):
        return (fun(x0 + eps) - fun(x0)) / eps

    for x0 in [1., 1.1,
               0.9 * cutoffpot.r_min,
               0.999 * cutoffpot.r_min,
               cutoffpot.r_min,
               1.1 * cutoffpot.r_min,
               0.9 * cutoffpot.r_infl,
               0.999 * cutoffpot.r_infl,
               cutoffpot.r_infl,
               1.1 * cutoffpot.r_infl,
               0.9 * cutoffpot.cutoff_radius,
               0.999 * cutoffpot.cutoff_radius,
               cutoffpot.cutoff_radius,
               1.1 * cutoffpot.cutoff_radius,
               ]:
        eps = 1e-8
        # forward finite differece wrt x0,
        # or central fintite differences wrt x0+eps / 2
        numder = approx_fprime(x0, lambda x: cutoffpot.evaluate(x)[0], eps)
        analder = cutoffpot.evaluate(x0 + eps / 2, True, True)[1]

        assert abs(numder - analder) < 1e-5

        numder = approx_fprime(x0,
                               lambda x: cutoffpot.evaluate(x, True, True)[1],
                               eps)
        analder = cutoffpot.evaluate(x0 + eps / 2, True, True, True)[2]
        assert abs(numder - analder) < 1e-5


def test_parabolic_cutoff():
    _plot = False
    eps = 1.
    sig = 1.

    r_t_lin = 0.5  # noqa: F841
    r_c = 1.5

    pot = LJ93(eps, sig)
    cutoff_pot = pot.parabolic_cutoff(r_c)
    z = np.linspace(0.9 * r_c, 1.01 * r_c, 100)
    ratio = cutoff_pot.evaluate(z)[0] / pot.evaluate(z)[0]
    assert ((ratio[1:] - ratio[:-1]) <= 0).all()
    if _plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(ratio)
        plt.show()

    z = np.linspace(0.9 * r_c, 1.01 * r_c, 100)

    ratio = abs(
        cutoff_pot.evaluate(z, True, True)[1] / pot.evaluate(z, True, True)[1])
    assert ((ratio[1:] - ratio[:-1]) <= 0).all()

    if _plot:
        fig, ax = plt.subplots()
        ax.plot(ratio, )
        plt.show()

        z = np.linspace(0.9 * r_c, 1.01 * r_c, 100)

    ratio = abs(cutoff_pot.evaluate(z, True, True, True)[2] /
                pot.evaluate(z, True, True, True)[2])
    assert ((ratio[1:] - ratio[:-1]) <= 0).all()

    if _plot:
        fig, ax = plt.subplots()
        ax.plot(ratio, )
        plt.show()


@pytest.mark.skip("just plotting")
def test_smooth_potential_energy():
    eps = 1.703
    sigma = 3.633
    gam = 5.606

    pot = LJ93(eps, sigma).spline_cutoff(gam)
    z = np.linspace(2., 8, 5000)
    zm = (z[1:] + z[:-1]) / 2
    num_der = (pot.evaluate(z[1:])[0]
               - pot.evaluate(z[:-1])[0]) / (z[1:] - z[:-1])
    exact_der = pot.evaluate(zm, True, True)[1]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.axvline(pot.r_min, c="k")
    ax.axvline(pot.cutoff_radius, c="k")

    ax.plot(zm, abs(num_der - exact_der))
    ax.set_yscale("log")

    pot = LJ93(eps, sigma)
    z = np.linspace(2., 8, 5000)
    zm = (z[1:] + z[:-1]) / 2
    num_der = (pot.evaluate(z[1:])[0] - pot.evaluate(z[:-1])[0]) / (
            z[1:] - z[:-1])
    exact_der = pot.evaluate(zm, True, True)[1]
    import matplotlib.pyplot as plt

    ax.plot(zm, abs(num_der - exact_der))
    ax.set_yscale("log")
    plt.show()

    fig, ax = plt.subplots()
    pot = LJ93(eps, sigma).spline_cutoff(gam)
    z = np.linspace(2., 8, 5000)

    num_curb = (pot.evaluate(z[2:])[0] - 2 * pot.evaluate(z[1:-1])[0] +
                pot.evaluate(z[:-2])[0]) / ((z[2:] - z[:-2]) / 2) ** 2
    # exact_der = pot.evaluate(zm, True, True)[1]
    ax.plot(z[1:-1], abs(num_curb))
    ax.set_yscale("log")
    plt.show()


def test_pipeline():
    pot1 = LJ93(1., 1.)
    pot2 = pot1.linearize_core()
    p = pot2.pipeline()
    assert isinstance(p[0], LJ93)
    assert isinstance(p[1], LinearCorePotential)


@pytest.mark.parametrize("pot",
                         [
                             PowerLaw(1., 1.),  # has intrinsically a cutoff
                             LJ93(1., 1., ).parabolic_cutoff(2.),
                             LJ93(1., 1., ).spline_cutoff(2.)
                             ])
def test_cutoffs(pot):
    assert (np.array(
        pot.evaluate(2 * pot.cutoff_radius, True, True, True)) == 0).all()


def test_morse():
    w = 1.456
    range = 7.56
    pot = Morse(w, range)
    assert pot.evaluate(0)[0] == - w
