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

from NuMPI.Tools import Reduction
from NuMPI import MPI
from ContactMechanics.Factory import _make_system_args
import ContactMechanics.Factory


def make_system(substrate, interaction, surface, communicator=MPI.COMM_WORLD,
                physical_sizes=None, system_class=None,
                **kwargs):
    """
    Factory function for contact systems. The returned object is always
    of a subtype of SystemBase.

    Parameters:
    -----------
    substrate : ContactMechanics.Substrate
        Defines the solid mechanics in the substrate
    interaction : Adhesion.Interaction
        Defines the contact formulation
    surface : SurfaceTopography.Topography
        Defines the profile.

    Returns
    -------
    system: child class of SystemBase

    """

    if interaction == "hardwall":
        return ContactMechanics.Factory.make_system(
            substrate, surface,
            communicator=communicator,
            physical_sizes=physical_sizes,
            **kwargs)
    else:
        substrate, surface = _make_system_args(substrate, surface,
                                               communicator=communicator,
                                               physical_sizes=physical_sizes,
                                               **kwargs)
        # make shure the interaction has the correcrt communicator
        interaction.reduction = Reduction(communicator)
        interaction.communicator = communicator

        return system_class(substrate, interaction, surface)
