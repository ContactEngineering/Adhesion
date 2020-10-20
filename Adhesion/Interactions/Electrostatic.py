from Adhesion.Interactions import Potential
from NuMPI import MPI


def ChargePatternsInteraction(Potential):
    """
    Potential for the interaction of charges

    Please cite:
    Persson, B. N. J. et al. EPL 103, 36003 (2013)
    """

    def __init__(self,
                 charge_distribution,
                 physical_sizes,
                 dielectric_constant_material=1.,
                 dielectric_constant_gap=1.,
                 communicator=MPI.COMM_WORLD):
        """

        Parameters
        ----------
        charge_distribution: float ndarray
            spatial distribution
        physical_sizes: tuple

        dielectric_constant_material: float
        dielectric_constant_gap: float

        Returns
        -------

        """
        Potential.__init__(self, communicator=communicator)
        self.nb_grid_points = charge_distribution.shape

    def evaluate(gap, ):
        """

        Parameters
        ----------
        gap: array_like

        Returns
        -------

        potential: float

        gradient: ndarray
            first derivative of the potential wrt. gap  (= - forces by pixel)
        curvature: ndarray or linear operator (callable)
            second derivative of the potential
            # TODO: is that easy/possible/computationally tractable ?
        """
        pass