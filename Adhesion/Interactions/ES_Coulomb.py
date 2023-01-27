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
Electrostatic potential for Coulombic Interactions

Persson, B.N.J., 2018. The dependency of adhesion and friction on electrostatic attraction. The Journal of chemical physics, 148(14), p.144701.
"""

from Adhesion.Interactions import Potential

import numpy as np
from NuMPI import MPI


class ES_C(Potential):
    """ Electrostatic potential for Coulombic interaction when a voltage V is applied across an air gap between two perfect insulators of thicknesses d1,d2 and permittivities eps1,eps2

    """

    name = "es-c"

    def __init__(self, eps1, eps2, d1, d2, V, communicator=MPI.COMM_WORLD):
        """
        Parameters:
        -----------
        eps1: float
            Dielectric permittivity of bottom surface
        eps2: float
            Dielectric permittivity of top surface
        d1: float
            Thickness of bottom surface
        d2: float
            Thickness of top surface
        V: float
            Applied voltage
        communicator: not used
        """
        self.eps1 = float(eps1)
        self.eps2 = float(eps2)
        self.d1 = float(d1)
        self.d2 = float(d2)
        self.V = float(V)
        
        Potential.__init__(self, communicator=communicator)

    def __getstate__(self):
        state = super().__getstate__(), self.eps1, self.eps2, self.d1, self.d2, self.V
        return state

    def __setstate__(self, state):
        superstate, self.eps1, self.eps2, self.d1, self.d2, self.V = state
        super().__setstate__(superstate)

    def __repr__(self, ):
        return ("Potential '{0.name}': eps1 = {0.eps1}, eps2 = {0.eps2}, d1 = {0.d1}, d2 = {0.d2}, V = {0.V},).format(self)

    def evaluate(self, gap, potential=True, gradient=False, curvature=False,
                 mask=None):
        r = np.asarray(gap)

        V = dV = ddV = None
        
        eps0 = 8.85e-12
	h0 = (self.d1 / self.eps1) + (self.d2 / self.eps2)
	
        if potential:
            V = 0.5 * self.eps0 * (self.V**2) * (1 ./ (r + h0))
        if gradient:
            dV = - 0.5 * self.eps0 * (self.V**2) * (1 ./ (r + h0))**2
        if curvature:
            ddV = 2 * 0.5 * self.eps0 * (self.V**2) * (1 ./ (r + h0))**3

        return (V, dV, ddV)
