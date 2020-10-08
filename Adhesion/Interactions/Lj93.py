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
9-3 Lennard-Jones potential for wall interactions
"""

from Adhesion.Interactions import Potential

import numpy as np
from NuMPI import MPI


class LJ93(Potential):
    """ 9-3 Lennard-Jones potential with optional cutoff radius.

        9-3 Lennard-Jones potential:
        V_l (r) = ε[ 2/15 (σ/r)**9 - (σ/r)**3]


                         ⎛   3       9⎞
                         ⎜  σ     2⋅σ ⎟
            V_l(r) =   ε⋅⎜- ── + ─────⎟
                         ⎜   3       9⎟
                         ⎝  r    15⋅r ⎠

                         ⎛   3       9⎞
                         ⎜3⋅σ     6⋅σ ⎟
            V_l'(r) =  ε⋅⎜──── - ─────⎟
                         ⎜  4       10⎟
                         ⎝ r     5⋅r  ⎠

                         ⎛      3       9⎞
                         ⎜  12⋅σ    12⋅σ ⎟
            V_l''(r) = ε⋅⎜- ───── + ─────⎟
                         ⎜     5      11 ⎟
                         ⎝    r      r   ⎠

    """

    name = "lj-93"

    def __init__(self, epsilon, sigma, communicator=MPI.COMM_WORLD):
        """
        Parameters:
        -----------
        epsilon: float
            Lennard-Jones potential well ε
        sigma: float
            Lennard-Jones distance parameter σ
        communicator: not used
        """
        self.eps = float(epsilon)
        self.sig = float(sigma)
        Potential.__init__(self, communicator=communicator)

    def __getstate__(self):
        state = super().__getstate__(), self.eps, self.sig
        return state

    def __setstate__(self, state):
        superstate, self.eps, self.sig = state
        super().__setstate__(superstate)

    def __repr__(self, ):
        return ("Potential '{0.name}': ε = {0.eps}, σ = {0.sig}").format(self)

    @property
    def r_min(self):
        """convenience function returning the location of the energy minimum

                6 ___  5/6
        r_min = ╲╱ 2 ⋅5   ⋅σ
                ────────────
                     5
        """
        return self.sig*(2*5**5)**(1./6)/5.

    @property
    def r_infl(self):
        """convenience function returning the location of the potential's
        inflection point (if applicable)

        r_infl = σ
        """
        return self.sig

    def evaluate(self, gap, potential=True, gradient=False, curvature=False,
                 mask=None):
        r = np.asarray(gap)

        V = dV = ddV = None
        sig_r3 = (self.sig / r) ** 3
        sig_r9 = sig_r3 ** 3

        if potential:
            V = self.eps * (2. / 15 * sig_r9 - sig_r3)
        if gradient or curvature:
            eps_r = self.eps / r
        if gradient:
            dV = - eps_r * (6. / 5 * sig_r9 - 3 * sig_r3)
        if curvature:
            ddV = 12 * eps_r / r * (sig_r9 - sig_r3)

        return (V, dV, ddV)
