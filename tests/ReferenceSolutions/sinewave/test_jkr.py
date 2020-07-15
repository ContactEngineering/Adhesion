from ContactMechanics.ReferenceSolutions import Westergaard
from Adhesion.ReferenceSolutions.sinewave import JKR

import numpy as np
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


def test_SIF_asymmetric_against_symmetric():
    a = 0.2
    P = 0.1
    assert JKR.stress_intensity_factor_asymmetric(a, a, P=P) \
           == JKR.stress_intensity_factor_symmetric(a, P, der="0")
    assert JKR.stress_intensity_factor_asymmetric(a, a, P=P, der="1_a_o") \
           + JKR.stress_intensity_factor_asymmetric(a, a, P=P, der="1_a_s") \
           == JKR.stress_intensity_factor_symmetric(a, P, der="1_a")


def test_stress_intensity_factor_derivative_1_a_s():
    a = np.linspace(0.1, 0.3, 50)
    a_o = 0.2

    P = JKR.mean_pressure(a_o, 0.3)

    am = (a[1:] + a[:-1]) / 2
    da = a[1:] - a[:-1]
    dK_da_num = (JKR.stress_intensity_factor_asymmetric(a[1:], a_o, P=P)
                 - JKR.stress_intensity_factor_asymmetric(a[:-1], a_o, P=P, )
                 ) / da
    dK_da_analytical = JKR.stress_intensity_factor_asymmetric(
        a_s=am,
        a_o=a_o,
        P=P, der="1_a_s")
    if True:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(am, dK_da_num, "+", label="numerical")
        ax.plot(am, dK_da_analytical, "o", mfc="none",
                label="analytical")
        plt.show()

    np.testing.assert_allclose(dK_da_analytical, dK_da_num,
                               atol=1e-14, rtol=1e-4)


def test_stress_intensity_factor_derivative_1_a_o():
    a = np.linspace(0.1, 0.3, 2000)
    a_s = 0.2

    P = JKR.mean_pressure(a_s, 0.3)

    am = (a[1:] + a[:-1]) / 2
    da = a[1:] - a[:-1]
    dK_da_num = (JKR.stress_intensity_factor_asymmetric(a_s, a[1:], P=P)
                 - JKR.stress_intensity_factor_asymmetric(a_s, a[:-1],
                                                          P=P, )) / da
    dK_da_analytical = JKR.stress_intensity_factor_asymmetric(
        a_s=a_s,
        a_o=am,
        P=P, der="1_a_o")
    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(am, dK_da_num, "+", label="numerical")
        ax.plot(am, dK_da_analytical, "o", mfc="none",
                label="analytical")
        plt.show()

    np.testing.assert_allclose(dK_da_analytical, dK_da_num,
                               atol=1e-12, rtol=1e-4)
