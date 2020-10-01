import pytest

from Adhesion.ReferenceSolutions import JKR

import numpy as np


def test_penetration_radius_inverse():
    params = dict(radius=1., contact_modulus=1., work_of_adhesion=1.)
    assert abs(JKR.contact_radius(
        penetration=JKR.penetration(contact_radius=1., **params), **params) - 1
               ) < 1e-10


def test_penetration_force():
    JKR.penetration(force=0)  # TODO : accuracy
    

def test_force_penetration_vs_radius():
    a = 2. # has to be on the stable branch
    force_from_a = JKR.force(contact_radius=a)
    force_from_pen = JKR.force(penetration=JKR.penetration(contact_radius=a))

    assert abs(force_from_a - force_from_pen) < 1e-10


def test_force_radius_inverse():
     assert abs(JKR.contact_radius(
        force=JKR.force(contact_radius=2.)) - 2.
               ) < 1e-10


@pytest.mark.parametrize("w", [1 / np.pi, 2.])
def test_force_consistency_pen_w(w):
    contact_radius = 3.

    force_from_w = JKR.force(contact_radius=contact_radius, work_of_adhesion=w)
    pen = JKR.penetration(contact_radius=contact_radius, work_of_adhesion=w)
    force_r_pen = JKR.force(contact_radius=contact_radius, penetration=pen)

    assert force_from_w == force_r_pen


def test_stress_intensity_factor_energy_release_rate():
    # tests that the implementation of the stress intensity factor and the
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

    if False:
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
    """
    asserts the stress intensity factor is 0 for the hertzian contact radius
    """
    hertzian_radius = np.sqrt(radius * penetration)
    assert abs(
        JKR.stress_intensity_factor(contact_radius=hertzian_radius,
                                    penetration=penetration,
                                    radius=radius,
                                    contact_modulus=contact_modulus)) < 1e-15


@pytest.mark.parametrize("work_of_adhesion", (1 / np.pi, 10.))
@pytest.mark.parametrize("radius", (0.1, 1.))
@pytest.mark.parametrize("contact_modulus", (3 / 4, 1.))
@pytest.mark.parametrize("penetration", (0.1, 1.))
def test_stress_intensity_factor_JKR_radius(radius, contact_modulus,
                                            penetration, work_of_adhesion):
    """
    asserts the stress intensity factor and the jkr radius calculation are
    consistent
    """
    contact_radius = JKR.contact_radius(penetration=penetration,
                                        radius=radius,
                                        contact_modulus=contact_modulus,
                                        work_of_adhesion=work_of_adhesion,
                                        )

    sif = JKR.stress_intensity_factor(contact_radius=contact_radius,
                                      penetration=penetration,
                                      radius=radius,
                                      contact_modulus=contact_modulus)
    sif_ref = np.sqrt(work_of_adhesion * 2 * contact_modulus)
    assert abs(sif - sif_ref) / sif_ref < 1e-10


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


def test_equilibrium_elastic_energy_vs_nonequilibrium():
    a = 0.5
    Eel = JKR.equilibrium_elastic_energy(a)

    np.testing.assert_allclose(Eel,JKR.nonequilibrium_elastic_energy(JKR.penetration(a), a))