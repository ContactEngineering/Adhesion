#
# Copyright 2020 Antoine Sanner
#           2020 Lars Pastewka
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
Implements a van der Waals-type interaction as described in
http://dx.doi.org/10.1103/PhysRevLett.111.035502
"""

import numpy as np
from NuMPI import MPI

from Adhesion.Interactions import Potential


class VDW82(Potential):
    """
    Van der Waals-type attractive potential with a fantasy repulsive model
    (like Lennard-Jones). The potential uses the formulation of Lessel et al.
    2013 (http://dx.doi.org/10.1103/PhysRevLett.111.035502). However, the oxide
    layer is supposed do be thick


                           A      C_sr
            vdW(r)  = - ─────── + ────
                            2       8
                        12⋅r ⋅π    r

                        A      8⋅C_sr
            vdW'(r) = ────── - ──────
                         3        9
                      6⋅r ⋅π     r

                          A      72⋅C_sr
            vdW"(r) = - ────── + ───────
                           4        10
                        2⋅r ⋅π     r


    """
    name = 'v-d-Waals82'

    def __init__(self, c_sr, hamaker, communicator=MPI.COMM_WORLD):
        """
        Parameters:
        -----------
        c_sr: float
            coefficient for repulsive part
        hamaker: float
            Hamaker constant for substrate
        """
        self.c_sr = c_sr
        self.hamaker = hamaker
        Potential.__init__(self, communicator=communicator)

    def __getstate__(self):
        state = super().__getstate__(), self.c_sr, self.hamaker
        return state

    def __setstate__(self, state):
        superstate, self.c_sr, self.hamaker = state
        super().__setstate__(superstate)

    def __repr__(self, ):
        return ("Potential '{0.name}': C_SR = {0.c_sr}, "
                "A_l = {0.hamaker}").format(
                    self, )  # nopep8

    def evaluate(self, gap, potential=True, gradient=False, curvature=False,
                 mask=(slice(None), slice(None))):
        r = np.asarray(gap)
        V = dV = ddV = None
        r_2 = r ** -2
        c_sr_r6 = self.c_sr * r_2 ** 3 if np.isscalar(self.c_sr) \
            else self.c_sr[mask] * r_2 ** 3
        a_pi = self.hamaker / np.pi if np.isscalar(self.hamaker) \
            else self.hamaker[mask] / np.pi

        if potential:
            V = r_2 * (-a_pi / 12 + c_sr_r6)
        if gradient or curvature:
            r_3 = r_2 / r
        if gradient:
            # Forces are the negative gradient
            dV = - r_3 * (-a_pi / 6 + 8 * c_sr_r6)
        if curvature:
            ddV = r_3/r*(-a_pi/2 + 72*c_sr_r6)
        return (V, dV, ddV)

    @property
    def r_min(self):
        """convenience function returning the location of the enery minimum
                               ________
                 2/3 6 ___    ╱ C_sr⋅π
        r_min = 2   ⋅╲╱ 3 ⋅6 ╱  ──────
                           ╲╱     A
        """
        return 2**(2./3)*3**(1./6)*(self.c_sr*np.pi/self.hamaker)**(1./6)

    @property
    def r_infl(self):
        """convenience function returning the location of the potential's
        inflection point (if applicable)

                            ________
                 3 ____    ╱ C_sr⋅π
        r_infl = ╲╱ 12 ⋅6 ╱  ──────
                        ╲╱     A
        """
        return (144 * np.pi * self.c_sr / self.hamaker) ** (1. / 6.)


def Lj82(w, z0, **kwargs):
    return VDW82(w * z0 ** 8 / 3, 16 * np.pi * w * z0 ** 2, **kwargs)
