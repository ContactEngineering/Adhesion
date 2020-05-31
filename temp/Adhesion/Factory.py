
from NuMPI.Tools import Reduction
from NuMPI import MPI
from PyCo.ContactMechanics.Factory import _make_system_args
import PyCo.ContactMechanics.Factory

def make_system(substrate, interaction, surface, communicator=MPI.COMM_WORLD,
                physical_sizes=None, system_class=None,
                **kwargs):
    """
    Factory function for contact systems. The returned object is always of a subtype
    of SystemBase.
    Parameters:
    -----------
    substrate   -- An instance of HalfSpace. Defines the solid mechanics in
                   the substrate
    interaction -- An instance of Interaction. Defines the contact formulation
    surface     -- An instance of SurfaceTopography, defines the profile.
    Returns
    -------
    """

    if interaction=="hardwall":
        return PyCo.ContactMechanics.Factory.make_system(substrate, surface, communicator=MPI.COMM_WORLD,
                physical_sizes=physical_sizes,
                **kwargs)
    else:
        substrate, surface = _make_system_args(substrate, surface, communicator=MPI.COMM_WORLD,
                physical_sizes=physical_sizes,
                **kwargs)
        # make shure the interaction has the correcrt communicator
        interaction.pnp = Reduction(communicator)
        interaction.communicator = communicator

        return system_class(substrate, interaction, surface)