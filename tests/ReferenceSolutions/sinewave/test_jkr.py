from ContactMechanics.ReferenceSolutions import Westergaard
from Adhesion.ReferenceSolutions.sinewave import JKR

import pytest
from NuMPI import MPI

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="tests only serial funcionalities, "
                                       "please execute with pytest")


def test_elastic_energy_vs_westergaard():
    a = 0.2
    mean_pressure = JKR.mean_pressure(a, 0)
    jkr_energy = JKR.elastic_energy(a, mean_pressure)

    assert Westergaard.elastic_energy(mean_pressure) == jkr_energy


# TODO: tests of the derivatives of JKRE