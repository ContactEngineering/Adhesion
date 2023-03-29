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
Defines the base class for contact description
"""

import numpy as np
import copy
from NuMPI import MPI
from NuMPI.Tools import Reduction


class Interaction(object):
    """base class for all interactions, e.g. interatomic potentials"""
    # pylint: disable=too-few-public-methods
    pass


class HardWall(Interaction):
    """base class for non-smooth contact mechanics"""

    # pylint: disable=too-few-public-methods
    def __init__(self):
        self.penetration = None

    def compute(self, gap, tol=0.):
        """
        Parameters:
        -----------
        gap: array_like
            array containing the point-wise gap values
        tol: float, optional
            tolerance for determining whether the gap is closed, default 0.
        """
        self.penetration = np.where(gap < tol, -gap, 0)


class Dugdale(HardWall):
    """Potential class representing a Dugdale cohesive zone model"""

    def __init__(self, stress, length):
        super().__init__()
        self._stress = stress
        self._length = length

    @property
    def stress(self):
        return self._stress

    @property
    def length(self):
        return self._length

    def evaluate(self, gap, tol=0.):
        return np.where(gap < self._length,
                        self._stress*np.ones_like(gap),
                        np.zeros_like(gap))


class SoftWall(Interaction):
    """base class for smooth contact mechanics"""
    def __init__(self, communicator=MPI.COMM_WORLD):
        self.communicator = communicator
        self.reduction = Reduction(communicator)

    def __deepcopy__(self, memo):
        """
        makes a deepcopy of all the attributes except self.reduction,
        where it stores the same reference

        Parameters
        ----------
        memo

        Returns
        -------
        result SoftWall instance
        """

        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result

        keys = set(self.__dict__.keys())

        # exceptions
        # pnp is a module or a class impolenting computation methods,
        # it is not copied
        result.reduction = self.reduction
        keys.remove('reduction')
        # same for communicator
        result.communicator = self.communicator
        keys.remove('communicator')

        for k in keys:
            setattr(result, k, copy.deepcopy(getattr(self, k), memo))
        return result

    def __getstate__(self):
        return self.energy, self.gradient

    def __setstate__(self, state):
        self.energy, self.gradient = state

    def evaluate(self, gap, potential=True, gradient=False):
        """
        computes and returns the interaction energy and/or forces based on the
        as fuction of the gap
        Parameters:
        -----------
        gap: array like
            the point-wise gap values
        potential: bool
            (default True) whether the energy should be evaluated
        gradient: bool
            (default False) whether the gradient should be evaluated
        """
        raise NotImplementedError()

    def plot(self):
        import matplotlib.pyplot as plt
        fig, (axpot, axf, axcurv) = plt.subplots(3, 1)
        r = np.linspace(0.7 * self.r_min, 4 * self.r_min, 200)

        v, dv, ddv = self.evaluate(r, True, True, True)

        axpot.plot(r, v, )
        axf.plot(r, dv, )
        axcurv.plot(r, ddv, )

        axpot.set_ylabel("Potential")
        axf.set_ylabel("interaction stress")
        axcurv.set_ylabel("curvature")

        for a in (axpot, axf, axcurv):
            a.grid()
            a.axvline(self.r_min, label="r_min")
            a.axvline(self.r_infl, label="r_infl")

        axpot.axhline(self.v_min, c="k", label="v_min")
        axf.axhline(self.max_tensile, c="k", label="maxtensile")

        axcurv.set_xlabel("gap")
        axpot.legend()

        return fig, (axpot, axf, axcurv)
