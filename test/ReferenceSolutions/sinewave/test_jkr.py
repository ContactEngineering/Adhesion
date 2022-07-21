#
# Copyright 2020 Antoine Sanner
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
from ContactMechanics.ReferenceSolutions import Westergaard
from Adhesion.ReferenceSolutions.sinewave import JKR

import numpy as np
import pytest
from NuMPI import MPI

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="tests only serial funcionalities, "
                                       "please execute with pytest")


def test_find_min_max_a():
    a_min, a_inflexion, a_max = JKR._find_min_max_a(0.1)


def test_contact_radius_mean_pressure_inverse():
    assert abs(JKR.contact_radius(JKR.mean_pressure(0.1, 0.2), 0.2) - 0.1) \
           < 1e-5


def test_elastic_energy_vs_westergaard():
    a = 0.2
    mean_pressure = JKR.mean_pressure(a, 0)
    jkr_energy = JKR.elastic_energy(a, mean_pressure)

    assert Westergaard.elastic_energy(mean_pressure) == jkr_energy


@pytest.mark.parametrize("mean_pressure", [.1, .8])
def test_sif_westergaard(mean_pressure):
    "assert the SIF is 0 when the contact radius is the westergaard solution"
    west_rad = 1 / np.pi * np.arcsin(np.sqrt(mean_pressure))
    sif = JKR.stress_intensity_factor_asymmetric(west_rad, west_rad,
                                                 mean_pressure)
    assert abs(sif) < 1e-12


@pytest.mark.parametrize("contact_radius, mean_pressure",
                         [(.1, -0.2), (.4, 0.6)])
def test_sif_contact_radius_roundtrip(contact_radius, mean_pressure):
    sif = JKR.stress_intensity_factor_asymmetric(contact_radius,
                                                 contact_radius, mean_pressure)
    alpha = sif

    assert abs(JKR.contact_radius(mean_pressure, alpha) - contact_radius
               ) < 1e-12


def test_SIF_asymmetric_against_symmetric():
    a = 0.2
    P = 0.1
    np.testing.assert_allclose(JKR.stress_intensity_factor_asymmetric(a, a, P=P),
                               JKR.stress_intensity_factor_symmetric(a, P, der="0"))
    np.testing.assert_allclose(JKR.stress_intensity_factor_asymmetric(a, a, P=P, der="1_a_o")
                               + JKR.stress_intensity_factor_asymmetric(a, a, P=P, der="1_a_s"),
                               JKR.stress_intensity_factor_symmetric(a, P, der="1_a"))


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
    if False:
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
