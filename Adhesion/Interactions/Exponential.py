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
Exponential attraction.
"""

from Adhesion.Interactions import Potential
import numpy as np
from NuMPI import MPI


class Exponential(Potential):
    """ V(g) = -gamma0*e^(-g(r)/rho)

            V(g) = -gamma0*e^(-g(r)/rho)
            V'(g) = (gamma0/rho)*e^(-g(r)/rho)
            V''(g) = -(gamma0/rho^2)*e^(-g(r)/rho)
    """

    name = "exp"

    def __init__(self, gamma0, rho, communicator=MPI.COMM_WORLD):
        r"""
        Parameters:
        -----------
        gamma0: float or ndarray
            surface energy at perfect contact
        rho: float or ndarray
            attenuation length
        """
        self.rho = rho
        self.gam = gamma0
        Potential.__init__(self, communicator=communicator)

    def __repr__(self, ):
        return ("Potential '{0.name}': gam = {0.gam}, rho = {0.rho},").format(
            self)

    def __getstate__(self):
        state = super().__getstate__(), self.rho, self.gam
        return state

    def __setstate__(self, state):
        superstate, self.rho, self.gam = state
        super().__setstate__(superstate)

    @property
    def r_min(self):
        return None

    @property
    def r_infl(self):
        return None

    @property
    def max_tensile(self):
        return - self.gam / self.rho

    def evaluate(self,
                 gap,
                 potential=True,
                 gradient=False,
                 curvature=False,
                 mask=None):

        r = np.asarray(gap)
        if mask is None:
            mask = (slice(None),) * len(r.shape)
        rho = self.rho if np.isscalar(self.rho) else self.rho[mask]
        g = -r / rho

        # Use exponential only for r > 0
        m = g < 0.0
        if np.isscalar(r):
            if m:
                V = -self.gam * np.exp(g)
                dV = - V / self.rho
                ddV = V / self.rho ** 2
            else:
                V = -self.gam * (1 + g + 0.5 * g ** 2)
                dV = self.gam / self.rho * (1 + g)
                ddV = -self.gam / self.rho ** 2
        else:
            V = np.zeros_like(g)
            dV = np.zeros_like(g)
            ddV = np.zeros_like(g)

            gam = self.gam if np.isscalar(self.gam) else self.gam[mask][m]
            rho = self.rho if np.isscalar(self.rho) else self.rho[mask][m]

            V[m] = - gam * np.exp(g[m])
            dV[m] = - V[m] / rho
            ddV[m] = V[m] / rho ** 2

            # Quadratic function for r < 0.
            # This avoids numerical overflow at small r.
            m = np.logical_not(m)

            gam = self.gam if np.isscalar(self.gam) else self.gam[mask][m]
            rho = self.rho if np.isscalar(self.rho) else self.rho[mask][m]

            V[m] = -gam * (1 + g[m] + 0.5 * g[m] ** 2)
            dV[m] = gam / rho * (1 + g[m])
            ddV[m] = -gam / rho ** 2

        return V, dV, ddV


class RepulsiveExponential(Potential):
    """ V(g) = -gamma_{rep}*e^(-r/rho_{rep}) - gamma_{att}*e^(-r/rho_{att})
    """

    name = "repulsive_exp"

    def __init__(self, gamma_rep, rho_rep, gamma_att, rho_att,
                 communicator=MPI.COMM_WORLD):
        """
        Parameters:
        -----------
        gamma_rep: array_like
            prefactor of the repulsive exponential
        rho_rep: float or ndarray
            attenuation length of the repulsive exponential
        gamma_att: array_like
            prefactor of the attractive exponential
        rho_att: float or ndarray
            attenuation length of the attractive exponential
        communicator: optional
        """

        self.rho_att = rho_att
        self.gam_att = gamma_att
        self.rho_rep = rho_rep
        self.gam_rep = gamma_rep
        Potential.__init__(self, communicator=communicator)

    def __repr__(self, ):
        return ("Potential '{0.name}': gamma_rep = {0.gam_rep}, "
                "rho_rep = {0.rho_rep}, gamma_att = {0.gam_att}, "
                "rho_att = {0.rho_att},"
                "cutoff_radius = {1}").format(
            self, self.r_c if self.has_cutoff else 'None')

    def __getstate__(self):
        state = super().__getstate__(), self.rho_rep, self.gam_rep, \
                self.rho_att, self.gam_att
        return state

    def __setstate__(self, state):
        superstate, self.rho_rep, self.gam_rep, \
            self.rho_att, self.gam_att = state
        super().__setstate__(superstate)

    @property
    def r_min(self):
        return np.log(
            self.gam_rep / self.gam_att * self.rho_att / self.rho_rep) \
               / (1 / self.rho_rep - 1 / self.rho_att)

    @property
    def r_infl(self):
        return np.log(
            self.gam_rep / self.gam_att
            * self.rho_att ** 2 / self.rho_rep ** 2) \
               / (1 / self.rho_rep - 1 / self.rho_att)

    def evaluate(self, gap, potential=True, gradient=False, curvature=False,
                 mask=(slice(None), slice(None))):

        if np.isscalar(self.rho_att):
            rho_att = self.rho_att
        else:
            rho_att = self.rho_att[mask]
        if np.isscalar(self.rho_rep):
            rho_rep = self.rho_rep
        else:
            rho_rep = self.rho_rep[mask]
        if np.isscalar(self.gam_att):
            gam_att = self.gam_att
        else:
            gam_att = self.gam_att[mask]
        if np.isscalar(self.gam_rep):
            gam_rep = self.gam_rep
        else:
            gam_rep = self.gam_rep[mask]
        g_att = -gap / self.rho_att
        g_rep = -gap / self.rho_rep

        # if np.isscalar(r):

        V_att = -gam_att * np.exp(g_att)
        dV_att = - V_att / rho_att  # = derivatibe of V_att
        ddV_att = V_att / rho_att ** 2

        V_rep = gam_rep * np.exp(g_rep)
        dV_rep = - V_rep / rho_rep
        ddV_rep = V_rep / rho_rep ** 2

        V = V_att + V_rep
        dV = dV_att + dV_rep
        ddV = ddV_att + ddV_rep

        return V, dV, ddV


class Morse(RepulsiveExponential):
    name = "morse"

    def __init__(self, work_of_adhesion, interaction_range,
                 communicator=MPI.COMM_WORLD):
        """
        Morse potential as in Wang, Zhou, Müser, modelling adhesion hysteresis

        This potential has it's mimum at  gap = 0

        Parameters
        ----------
        work_of_adhesion: float or ndarray
            surface energy at perfect contact: minimum of the potential
        interaction_range: float or ndarray
            lengthscale of the decay of the interaction
        communicator: optional
            not used here

        """
        super().__init__(1. * work_of_adhesion, 0.5 * interaction_range,
                         2. * work_of_adhesion, 1. * interaction_range,
                         communicator=communicator)
