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
import matplotlib.pyplot as plt
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
    a = 2.  # has to be on the stable branch
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


@pytest.mark.parametrize("contact_modulus", [0.75, 1.9])
@pytest.mark.parametrize("radius", [1., 10.])
def test_stress_intensity_factor_energy_release_rate(contact_modulus, radius):
    # tests that the implementation of the stress intensity factor and the
    # implementation of the energy release rate are equivalent

    a = np.linspace(0.5, 2)  # contact radius
    penetration = - 0.5
    en_release_rate = JKR.elastic_energy_release_rate(contact_radius=a, penetration=penetration,
                                                      contact_modulus=contact_modulus, radius=radius)
    stress_intensity_factor = JKR.stress_intensity_factor(
        penetration=penetration,
        contact_radius=a,
        contact_modulus=contact_modulus,
        radius=radius,
        )

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(a, stress_intensity_factor, "+",
                label="sif")
        ax.plot(a, np.sqrt(en_release_rate * 2 * contact_modulus), "x",
                label=r"$\sqrt{2 E^*G }$")
        ax.legend()
        plt.show()

    np.testing.assert_allclose(stress_intensity_factor,
                               np.sqrt(en_release_rate * (2 * contact_modulus)),
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
    dK_da_num = (JKR.stress_intensity_factor(a[1:], pen, )
                 - JKR.stress_intensity_factor(a[:-1], pen, )) / da
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
    dK_da2_num = (JKR.stress_intensity_factor(a[2:], pen, )
                  - 2 * JKR.stress_intensity_factor(a[1:-1], pen, )
                  + JKR.stress_intensity_factor(a[:-2], pen, )) / da ** 2
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


@pytest.mark.parametrize("contact_modulus", [0.75, 1.9])
@pytest.mark.parametrize("radius", [1., 10.])
def test_energy_release_rate_derivatives_against_sif_derivatives(radius, contact_modulus):
    contact_radius = np.random.uniform(0.1, 3)
    penetration = np.random.uniform(0.1, 3)

    sif = JKR.stress_intensity_factor(
        penetration=penetration,
        contact_radius=contact_radius,
        radius=radius, contact_modulus=contact_modulus)

    np.testing.assert_allclose(
        JKR.elastic_energy_release_rate(contact_radius=contact_radius, penetration=penetration,
                                        contact_modulus=contact_modulus, der="1_a", radius=radius),
        # Irwin:
        sif / contact_modulus
        * JKR.stress_intensity_factor(contact_radius=contact_radius, penetration=penetration, der="1_a",
                                      radius=radius, contact_modulus=contact_modulus),
        atol=1e-14, rtol=0
        )

    np.testing.assert_allclose(
        JKR.elastic_energy_release_rate(contact_radius=contact_radius, penetration=penetration,
                                        contact_modulus=contact_modulus, der="1_d", radius=radius),
        # Irwin:
        sif / contact_modulus
        * JKR.stress_intensity_factor(contact_radius=contact_radius, penetration=penetration, der="1_d",
                                      radius=radius, contact_modulus=contact_modulus),
        atol=1e-13, rtol=0
        )

    np.testing.assert_allclose(
        JKR.elastic_energy_release_rate(contact_radius=contact_radius, penetration=penetration,
                                        contact_modulus=contact_modulus, der="2_da", radius=radius),
        # Irwin:
        1 / contact_modulus * (
                sif * JKR.stress_intensity_factor(contact_radius=contact_radius, penetration=penetration, der="2_da",
                                                  radius=radius, contact_modulus=contact_modulus)
                + JKR.stress_intensity_factor(contact_radius=contact_radius, penetration=penetration, der="1_d",
                                              radius=radius, contact_modulus=contact_modulus) *
                JKR.stress_intensity_factor(contact_radius=contact_radius, penetration=penetration, der="1_a",
                                            radius=radius, contact_modulus=contact_modulus)
        ),
        atol=1e-13, rtol=0
        )
    np.testing.assert_allclose(
        JKR.elastic_energy_release_rate(contact_radius=contact_radius, penetration=penetration,
                                        contact_modulus=contact_modulus, der="2_a", radius=radius),
        # Irwin:
        1 / contact_modulus * (
                sif * JKR.stress_intensity_factor(contact_radius=contact_radius, penetration=penetration, der="2_a",
                                                  radius=radius, contact_modulus=contact_modulus)
                + JKR.stress_intensity_factor(contact_radius=contact_radius, penetration=penetration, der="1_a",
                                              radius=radius, contact_modulus=contact_modulus) ** 2
        ),
        atol=1e-13, rtol=0
    )


@pytest.mark.parametrize("contact_modulus", [0.75, 1.9])
@pytest.mark.parametrize("radius", [1., 10.])
@pytest.mark.parametrize("force", [0., 1.])
def test_energy_release_rate_from_force(radius, contact_modulus, force):
    work_of_adhesion = 2.
    contact_radius = JKR.contact_radius(force=force,
                                        contact_modulus=contact_modulus,
                                        radius=radius,
                                        work_of_adhesion=work_of_adhesion)

    np.testing.assert_allclose(JKR.elastic_energy_release_rate(
        contact_radius=contact_radius,
        force=force,
        contact_modulus=contact_modulus, radius=radius),
        work_of_adhesion)


def test_equilibrium_elastic_energy_vs_nonequilibrium():
    a = 0.5
    Eel = JKR.equilibrium_elastic_energy(a)

    np.testing.assert_allclose(Eel,
                               JKR.elastic_energy(
                                   JKR.penetration(a), a))


def test_deformed_profile():
    Es = 3 / 4  # maugis K = 1.
    w = 1 / np.pi
    R = 1.
    penetration = 0.5

    contact_radius = JKR.contact_radius(penetration=penetration)
    rho = np.linspace(0.00001, 0.001)

    g = JKR.deformed_profile(contact_radius + rho, contact_radius=contact_radius, radius=R, contact_modulus=Es,
                             work_of_adhesion=w)

    sif = np.sqrt(2 * Es * w)

    if False:
        fig, ax = plt.subplots()
        ax.plot(rho, g, "-", c="gray")
        ax.plot(rho, np.sqrt(rho / (2 * np.pi)) * 4 * sif / Es, "--", c="k")
        plt.show()

    np.testing.assert_allclose(g, np.sqrt(rho / (2 * np.pi)) * 4 * sif / Es, atol=1e-4)


# In JKR units
@pytest.mark.parametrize("contact_radius", [
    0.05, 0.5, 1., 1.5, 1.8,  # negative forces
    1.9, 2., 20.  # positive forces
    ])
def test_contact_radius_from_penetration_force(contact_radius):
    # ISSUE: seems to be broken for positive forces.

    penetration = JKR.penetration(contact_radius=contact_radius)
    force = JKR.force(contact_radius=contact_radius)
    contact_radius_computed = JKR._contact_radius_from_penetration_force(
        penetration=penetration,
        force=force
        )
    assert abs(contact_radius_computed - contact_radius) < 1e-10, f"force={force}"
