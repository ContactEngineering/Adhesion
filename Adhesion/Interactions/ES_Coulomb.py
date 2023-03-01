#
# Copyright 2023 Sitangshu Chatterjee
#           2020 Antoine Sanner
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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

Heß, M. and Forsbach, F., 2020. Macroscopic modeling of fingerpad friction under electroadhesion: Possibilities and limitations. Frontiers in Mechanical Engineering, 6, p.567386.
"""

from Adhesion.Interactions import Potential

import numpy as np
from NuMPI import MPI


class ES_C(Potential):
    """ Electrostatic potential for Coulombic interaction when a voltage is applied across a gap between two perfect insulators
@doi https://doi.org/10.3389/fmech.2020.567386
    """

    name = "es-c"

    def __init__(self, eps0, eps1, epsg = 1, eps2, d1, d2, voltage, communicator=MPI.COMM_WORLD):
        """
        Parameters:
        -----------
        eps0: float
            Permittivity of free space        
        eps1: float
            Relative dielectric permittivity of bottom surface
        epsg: float
            Relative dielectric permittivity of gap (Default = 1 for air)         
        eps2: float
            Relative dielectric permittivity of top surface
        d1: float
            Thickness of bottom surface
        d2: float
            Thickness of top surface
        voltage: float
            Applied voltage
        communicator: not used
        """
        self.eps0 = float(eps0)
        self.eps1 = float(eps1)
        self.epsg = float(epsg)        
        self.eps2 = float(eps2)
        self.d1 = float(d1)
        self.d2 = float(d2)
        self.voltage = float(voltage)
        
        Potential.__init__(self, communicator=communicator)

    def __getstate__(self):
        state = super().__getstate__(), self.eps0, self.eps1, self.epsg, self.eps2, self.d1, self.d2, self.voltage
        return state

    def __setstate__(self, state):
        superstate, self.eps1, self.epsg, self.eps0, self.eps2, self.d1, self.d2, self.voltage = state
        super().__setstate__(superstate)

    def __repr__(self, ):
        return ("Potential '{0.name}': eps0 = {0.eps0}, eps1 = {0.eps1}, epsg = {0.epsg}, eps2 = {0.eps2}, d1 = {0.d1}, d2 = {0.d2}, voltage = {0.voltage},).format(self)

    def evaluate(self, gap, potential=True, gradient=False, curvature=False,
                 mask=None):
                 
        r = np.asarray(gap)

        V = dV = ddV = None
        
	h0 = (self.d1 / self.eps1) + (self.d2 / self.eps2)
	
        if potential:
            V = 0.5 * self.eps0 * (self.voltage**2) * (1 / (r/self.epsg + h0))
        if gradient:
            dV = - 0.5 * self.eps0 * (self.voltage**2) * (1 / (r/self.epsg + h0))**2
        if curvature:
            ddV = 2 * 0.5 * self.eps0 * (self.voltage**2) * (1 / (r/self.epsg + h0))**3

        return (V, dV, ddV)
