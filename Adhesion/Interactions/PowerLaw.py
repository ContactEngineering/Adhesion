import numpy as np
from NuMPI import MPI

from Adhesion.Interactions import Potential, SoftWall


class PowerLaw(Potential):
    r""" Polynomial interaction wiches value, first and second derivatives are
    0 at the cutoff radius

    .. math ::

         (r < cutoff_radius) \ (1 - r / cutoff_radius)^p

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
            distance at which the potential has decayed to 0
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
        state = super().__getstate__(), self.exponent, self.rho, self.work_of_adhesion
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
