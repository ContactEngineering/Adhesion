#
# Copyright 2018, 2020 Antoine Sanner
#           2018, 2020 Lars Pastewka
#           2015-2016 Till Junge
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
Tests for higher system methods, such as computing pulloff-force
"""

import unittest
import numpy as np
from scipy.optimize import minimize
import time
import math

from ContactMechanics import make_system
from SurfaceTopography import make_sphere
from Adhesion.Interactions import LJ93
from ContactMechanics import FreeFFTElasticHalfSpace as Substrate


class PulloffTest(unittest.TestCase):
    def setUp(self):
        base_res = 64
        res = (base_res, base_res)
        base_size = 100.
        size = (base_size, base_size)
        young = 10

        self.substrate = Substrate(res, young, size)

        radius = size[0] / 10
        self.surface = make_sphere(radius, res, size, standoff=float('inf'))

        sigma = radius / 10
        epsilon = sigma * young / 100
        self.pot = LJ93(epsilon, sigma).spline_cutoff().linearize_core()

    def tst_FirstContactThenOffset(self):
        system = make_system(self.substrate, self.pot, self.surface)
        offset0 = .5 * self.pot.r_min + .5 * self.pot.cutoff_radius
        disp0 = np.zeros(self.substrate.nb_domain_grid_pts)

        def obj_fun(offset):
            nonlocal disp0
            inner = None
            print("Running with offset = {}".format(offset))

            def callback(xk):
                nonlocal inner
                if inner is None:
                    inner = system.callback(True)
                inner(xk)

            options = {'gtol': 1e-20, 'ftol': 1e-20}
            result = system.minimize_proxy(offset, disp0,  # callback=callback,
                                           options=options)
            if result.success:
                disp0 = system.disp
                return system.compute_normal_force()
            diagnostic = "offset = {}, " \
                         "cutoff_radius = {}, " \
                         "mean(f_pot) = {}" \
                         "".format(offset, self.pot.cutoff_radius,
                                   system.interaction_force.mean())
            gap = system.compute_gap(system.disp, offset)
            diagnostic += ", gap: (min, max) = ({}, {})".format(gap.min(),
                                                                gap.max())
            diagnostic += ", disp: (min, max) = ({}, {})".format(
                system.disp.min(), system.disp.max())

            raise Exception(
                "couldn't minimize: {}: {}".format(result.message, diagnostic))

        # constrain the force to be negative by convention
        def fun(x):
            retval = -obj_fun(x)
            print("WILL BE RETURNING: {}".format(retval))
            return retval

        constraints = ({'type': 'ineq', 'fun': fun},)
        bounds = ((0.7, 1),)
        start = time.perf_counter()
        result = minimize(obj_fun, x0=offset0, constraints=constraints,
                          bounds=bounds, method='slsqp',
                          options={'ftol': 1e-8})
        duration = time.perf_counter() - start
        msg = str(result)
        msg += "\ntook {} seconds".format(duration)
        raise Exception(msg)

    def tst_FirstOffsetThenContact(self):
        system = make_system(self.substrate, self.pot, self.surface)
        offset0 = .5 * self.pot.r_min + .5 * self.pot.cutoff_radius
        disp0 = np.zeros(self.substrate.nb_domain_grid_pts)

        def minimize_force(offset0, const_disp):
            system.create_babushka(offset0, const_disp)
            min_gap = system.compute_gap(const_disp, 0).min()
            # bounds insure non-penetration and non-separation
            bounds = ((-min_gap, .99 * self.pot.cutoff_radius - min_gap),)

            def obj(offset):
                babushka_disp = system._get_babushka_array(const_disp)
                system.babushka.evaluate(babushka_disp, offset,
                                         pot=False, forces=True)
                return system.babushka.compute_normal_force()

            result = minimize(obj, x0=offset0, bounds=bounds)

            if result.success:
                return result.x, result.fun
            raise Exception(str(result))

        result = minimize_force(offset0, disp0)

        r_range = self.pot.cutoff_radius - self.pot.r_min
        tol = r_range * 1e-8
        delta_offset = tol + 1.
        max_iter = 100
        it = 0
        disp = disp0
        offset = offset0
        while delta_offset > tol and it < max_iter:
            it += 1
            new_offset, pulloff_force = minimize_force(offset, disp)
            signed_delta = offset - new_offset
            delta_offset = abs(signed_delta)
            offset -= min(delta_offset, r_range * .01) \
                * math.copysign(1., signed_delta)
            result = system.minimize_proxy(offset, disp)
            disp = system.disp
            message = "iteration {:>3}: ".format(it)
            if result.success:
                message += "success"
            else:
                message += "FAIL!!"
            message += ", delta_offset = {}, force = {}".format(delta_offset,
                                                                pulloff_force)
            print(message)

        raise Exception(
            "Done! force = {}, offset = {} (offset_max = {}),"
            "\nresult = {}".format(
                pulloff_force, offset, self.pot.cutoff_radius, result))
