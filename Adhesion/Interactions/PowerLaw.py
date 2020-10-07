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
import numpy as np
from NuMPI import MPI

from Adhesion.Interactions import Potential, SoftWall


class PowerLaw(Potential):
    r""" Polynomial interaction wiches value, first and second derivatives are
    0 at the cutoff radius :math:`r_c`

    .. math ::

         (r < r_c) \ (1 - r / r_c)^p

    With the exponent :math:`p >= 3`
    """

    name = "PowerLaw"

    def __init__(self, work_of_adhesion, cutoff_radius, exponent=3,
                 communicator=MPI.COMM_WORLD):
        """
        Parameters:
        -----------
        work_of_adhesion: float or ndarray
            surface energy at perfect contact
        cutoff_radius: float or ndarray
            distance :math:`r_c` at which the potential has decayed to 0
        """
        self.cutoff_radius = self.rho = cutoff_radius
        self.work_of_adhesion = work_of_adhesion
        self.exponent = exponent
        SoftWall.__init__(self, communicator=communicator)

    def __repr__(self, ):
        return (
                "Potential '{0.name}': "
                "work_of_adhesion = {0.work_of_adhesion},"
                "cutoff_radius = {0.cutoff_radius}, exponent = {0.exponent}"
                ).format(self)

    def __getstate__(self):
        state = super().__getstate__(), \
            self.exponent, self.rho, self.work_of_adhesion
        return state

    def __setstate__(self, state):
        superstate, self.exponent, self.rho, self.work_of_adhesion = state
        super().__setstate__(superstate)

    @property
    def has_cutoff(self):
        return True

    @property
    def r_min(self):
        return None

    @property
    def r_infl(self):
        return None

    @property
    def max_tensile(self):
        return - self.work_of_adhesion / self.rho * self.exponent

    def evaluate(self, gap, potential=True, gradient=False, curvature=False,
                 mask=None):
        r = np.asarray(gap)
        if mask is None:
            mask = (slice(None), ) * len(r.shape)
        w = self.work_of_adhesion if np.isscalar(self.work_of_adhesion) \
            else self.work_of_adhesion[mask]
        rc = self.rho if np.isscalar(self.rho) else self.rho[mask]
        p = self.exponent

        g = (1 - r / rc)
        V = dV = ddV = None

        gpm2 = g ** (p - 2)
        gpm1 = gpm2 * g

        if potential:
            V = np.where(g > 0, - w * gpm1 * g, 0)
        if gradient:
            dV = np.where(g > 0, p * w / rc * gpm1, 0)
        if curvature:
            ddV = np.where(g > 0, - p * (p - 1) * w / rc ** 2 * gpm2, 0)

        return V, dV, ddV
