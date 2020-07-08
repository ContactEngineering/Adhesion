import pytest

from Adhesion.ReferenceSolutions import JKR

import numpy as np


def test_penetration_radius_inverse():
    params = dict(radius=1., contact_modulus=1., work_of_adhesion=1.)
    assert abs(JKR.contact_radius(
        penetration=JKR.penetration(contact_radius=1., **params), **params) - 1
               ) < 1e-10


def test_stress_intensity_factor_energy_release_rate():
    # tests that the implementation of the strss intensity factor and the
    # implementation of the energy release rate are equivalent

    e = 3 / 4  # contact modulus
    a = np.linspace(0.5, 2)  # contact radius
    penetration = - 0.5
    en_release_rate = JKR.nonequilibrium_elastic_energy_release_rate(
        penetration=penetration,
        contact_radius=a)
    stress_intensity_factor = JKR.stress_intensity_factor(
        penetration=penetration,
        contact_radius=a
        )

    if True:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(a, stress_intensity_factor, "+",
                label="sif")
        ax.plot(a, np.sqrt(en_release_rate * 2 * e), "x",
                label=r"$\sqrt{2 E^*G }$")
        ax.legend()
        plt.show()

    np.testing.assert_allclose(stress_intensity_factor,
                               np.sqrt(en_release_rate * (2 * e)),
                               atol=1e-14, rtol=1e-10)


@pytest.mark.parametrize("radius", (0.1, 1.))
@pytest.mark.parametrize("contact_modulus", (3 / 4, 1.))
@pytest.mark.parametrize("penetration", (0.1, 1.))
def test_stress_intensity_factor_hertzian(radius, contact_modulus,
                                          penetration):
    hertzian_radius = np.sqrt(radius * penetration)
    assert abs(
        JKR.stress_intensity_factor(contact_radius=hertzian_radius,
                                    penetration=penetration,
                                    radius=radius,
                                    contact_modulus=contact_modulus)) < 1e-15


def test_stress_intensity_factor_derivative():
    a = np.linspace(0.5, 1, 50)

    pen = 0.5
    am = (a[1:] + a[:-1]) / 2
    da = a[1:] - a[:-1]
    dK_da_num = (JKR.stress_intensity_factor(a[1:], pen,)
                 - JKR.stress_intensity_factor(a[:-1], pen,)) / da
    dK_da_analytical = JKR.stress_intensity_factor(
            contact_radius=am,
            penetration=pen, der="1_a")
    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(am, dK_da_num, "+", label="numerical")
        ax.plot(am, dK_da_analytical, "o", mfc="none",
                label="analytical")
        plt.show()

    np.testing.assert_allclose(dK_da_analytical, dK_da_num,
                               atol=1e-14, rtol=1e-4)


def test_stress_intensity_factor_second_derivative():
    a = np.linspace(0.5, 1, 1000)

    pen = 0.5
    am = a[1:-1]
    da = a[1] - a[0]
    dK_da2_num = (JKR.stress_intensity_factor(a[2:], pen,)
                  - 2 * JKR.stress_intensity_factor(a[1:-1], pen,)
                  + JKR.stress_intensity_factor(a[:-2], pen,)) / da ** 2
    dK_da2_analytical = JKR.stress_intensity_factor(
        contact_radius=am,
        penetration=pen, der="2_a")
    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(am, dK_da2_num, "+", label="numerical")
        ax.plot(am, dK_da2_analytical, "o", mfc="none",
                label="analytical")
        plt.show()

    np.testing.assert_allclose(dK_da2_analytical, dK_da2_num,
                               atol=1e-6, rtol=1e-4)
