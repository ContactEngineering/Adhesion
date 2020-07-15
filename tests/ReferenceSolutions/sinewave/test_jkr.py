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


def test_against_symmetric():
    a = 0.2
    P = 0.1
    assert JKR.stress_intensity_factor_asymmetric(a, a, P=P) \
           == JKR.stress_intensity_factor_symmetric(a, P, der="0")
    assert JKR.stress_intensity_factor_asymmetric(a, a, P=P, der="1_a_o") \
           + JKR.stress_intensity_factor_asymmetric(a, a, P=P, der="1_a_s") \
           == JKR.stress_intensity_factor_symmetric(a, P, der="1_a")
