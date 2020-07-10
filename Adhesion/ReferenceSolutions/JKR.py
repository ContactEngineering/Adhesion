#
# Copyright 2016 Lars Pastewka
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""

JKR solutions for the indentation of an elastic halfspace by a paraboloid
indenter.

The parameters of the system are:

radius : float, optional
    Sphere (actually paraboloid) radius.
contact_modulus : float, optional
    Contact modulus: :math:`E^* = E/(1-\nu**2)`
    with Young's modulus E and Poisson number :math:`\nu`.
    The default value is so that Maugis's contact Modulus is one
    (:math:`K = 4 / 3 E^*`)
work_of_adhesion : float, optional
    Work of adhesion.

This module provides implementations of the formulas relating the rigid body
penetration (`penetration`), the indentation force (`force`) and the radius
of the contact disk (`contact_radius`).

If the parameters are not provided, the relations are nondimensional.

The nondimensionalisation follows
Maugis's book (p.290):

    # TODO: put the Latex formulas of the nondimensionalisation here

"""

import numpy as np
from scipy.optimize import newton


###

def radius_unit(radius, contact_modulus, work_of_adhesion):
    k = 4 / 3 * contact_modulus # maugis contact modulus
    return (np.pi * work_of_adhesion * radius ** 2 / k) ** (1 / 3)


def height_unit(radius, contact_modulus, work_of_adhesion):
    k = 4 / 3 * contact_modulus
    return (np.pi ** 2 * work_of_adhesion ** 2 * radius / k ** 2) ** (1 / 3)


def load_unit(radius, work_of_adhesion):
    return np.pi * work_of_adhesion * radius


def contact_radius(force=None,
                   penetration=None,
                   radius=1.,
                   contact_modulus=3. / 4,
                   work_of_adhesion=1 / np.pi):
    r"""
    Given normal load or rigid body penetration, sphere radius and contact
    modulus compute contact radius.

    if only force or penetration is provided, it is assumed that the
    nondimensionalisation
    from Maugis's book is used.

    Parameters
    ----------
    force : float or array of floats, optional
        Normal force.
    penetration : float, optional
        rigid body penetration
    radius : float, optional
        Sphere (actually paraboloid) radius.
    contact_modulus : float, optional
        Contact modulus: :math:`E^* = E/(1-\nu**2)`
        with Young's modulus E and Poisson number :math:`\nu`.
        The default value is so that Maugis's contact Modulus is one
        (:math:`K = 4 / 3 E^*`)
    work_of_adhesion : float, optional
        Work of adhesion.
    """
    if force is None and penetration is None:
        raise ValueError("either force or penetration "
                         "should be provided")
    elif force is not None and penetration is not None:
        raise ValueError("only one of force or penetration "
                         "should be provided")

    if force is not None:
        A = force + 3 * work_of_adhesion * np.pi * radius
        B = np.sqrt(6 * work_of_adhesion * np.pi * radius * force + (
                    3 * work_of_adhesion * np.pi * radius) ** 2)

        fac = 3. * radius / (4. * contact_modulus)
        A *= fac
        B *= fac

        return (A + B) ** (1. / 3)

    elif penetration is not None:
        # TODO: is this solvable symbolically ?
        K = 4/3 * contact_modulus
        w = work_of_adhesion
        R = radius
        radius_pulloff = (np.pi * w * R**2 / 6 * K)**(1/3)
        penetration_pulloff = - 3 * radius_pulloff**2 / R
        if penetration >= penetration_pulloff:
            root = newton(lambda a:
                          a ** 2 / R
                          - np.sqrt(2 * np.pi * a * w / contact_modulus)
                          - penetration, radius_pulloff)
            return root
        else:
            return 0
_contact_radius_fun = contact_radius


def peak_pressure(force=None,
                  penetration=None,
                  radius=1.,
                  contact_modulus=3. / 4,
                  work_of_adhesion=1 / np.pi):
    r"""
    Given normal load or rigid body penetration, sphere radius and contact
    modulus compute contact radius.

    if only force or penetration is provided, it is assumed that the
    nondimensionalisation
    from Maugis's book is used.

    Parameters
    ----------
    force : float or array of floats, optional
        Normal force.
    penetration : float, optional
        rigid body penetration
    radius : float, optional
        Sphere (actually paraboloid) radius.
    contact_modulus : float, optional
        Contact modulus: :math:`E^* = E/(1-\nu**2)`
        with Young's modulus E and Poisson number :math:`\nu`.
        The default value is so that Maugis's contact Modulus is one
        (:math:`K = 4 / 3 E^*`)
    work_of_adhesion : float, optional
        Work of adhesion.
    """
    if force is None and penetration is None:
        raise ValueError("either force or penetration "
                         "should be provided")
    elif force is not None and penetration is not None:
        raise ValueError("only one of force or penetration "
                         "should be provided")

    if force is not None:
        A = force + 3 * work_of_adhesion * np.pi * radius
        B = np.sqrt(6 * work_of_adhesion * np.pi * radius * force + (
                    3 * work_of_adhesion * np.pi * radius) ** 2)

        fac = 3. * radius / (4. * contact_modulus)
        A *= fac
        B *= fac

        return (A - B) ** (1. / 3)

    raise NotImplementedError


def force(contact_radius=None,
          penetration=None,
          radius=1.,
          contact_modulus=3. / 4,
          work_of_adhesion=1 / np.pi):
    """
    Parameters
    ----------
    contact_radius : float or array of floats, optional
        Normal force.
    penetration : float, optional
        rigid body penetration
    radius : float, optional
        Sphere (actually paraboloid) radius.
    contact_modulus : float, optional
        Contact modulus: :math:`E^* = E/(1-\nu**2)`
        with Young's modulus E and Poisson number :math:`\nu`.
        The default value is so that Maugis's contact Modulus is one
        (:math:`K = 4 / 3 E^*`)
    work_of_adhesion : float, optional
        Work of adhesion.
    """

    if contact_radius is None and penetration is None:
        raise ValueError("either contact_radius or penetration "
                         "should be provided")
    elif contact_radius is not None and penetration is not None:
        raise ValueError("only one of contact_radius or penetration "
                         "should be provided")

    if contact_radius is None:
        contact_radius = _contact_radius_fun(penetration=penetration,
                                             radius=radius,
                                             contact_modulus=contact_modulus,
                                             work_of_adhesion=work_of_adhesion)

    maugis_modulus = 4 * contact_modulus / 3
    hertzian_force = contact_radius ** 3 * maugis_modulus / radius
    # Force in the hertzian contact at the same radius

    return hertzian_force \
        - np.sqrt(6 * np.pi * work_of_adhesion * radius * hertzian_force)


def penetration(contact_radius=None,
                force=None,
                radius=1.,
                contact_modulus=3. / 4,
                work_of_adhesion=1 / np.pi):
    """
        Parameters
    ----------
    contact_radius : float or array of floats, optional
        Normal force.
    penetration : float, optional
        rigid body penetration
    radius : float, optional
        Sphere (actually paraboloid) radius.
    contact_modulus : float, optional
        Contact modulus: :math:`E^* = E/(1-\nu**2)`
        with Young's modulus E and Poisson number :math:`\nu`.
        The default value is so that Maugis's contact Modulus is one
        (:math:`K = 4 / 3 E^*`)
    work_of_adhesion : float, optional
        Work of adhesion.
    """
    if contact_radius is None and force is None:
        raise ValueError("either contact_radius or force "
                         "should be provided")
    elif contact_radius is not None and force is not None:
        raise ValueError("only one of contact_radius or force "
                         "should be provided")

    if contact_radius is None:
        contact_radius = _contact_radius_fun(force=force, radius=radius,
                                             contact_modulus=contact_modulus,
                                             work_of_adhesion=work_of_adhesion)

    return contact_radius ** 2 / radius - np.sqrt(
        2 * np.pi * contact_radius * work_of_adhesion / contact_modulus)




def equilibrium_elastic_energy(contact_radius):
    A = contact_radius
    return 1/3 * A * (A**2 - np.sqrt(6 * A))**2 + A**5 / 15

# dimensional version
#def elastic_energy__a(a, R, Es, w):
#    return (a**5 * 4 * Es) / (9 * R**2) * (1/5 + (1 - R * np.sqrt(9 * np.pi * w / (2*a**3 * Es)))**2)


def nonequilibrium_elastic_energy(penetration, contact_radius):
    r"""

    Returns
    ..math:: \frac{U_el}{\pi w R (\pi^2 w^2 R / K^2)^{1/3}} = \frac{3}{4} A\left(\Delta - \frac{A^2}{3}\right)^2 + \frac{A^5}{15}

    For the units, see maugis p.290

    Parameters
    ----------
    penetration: :math:`\Delta` in maugis
    contact_radius: :math:`A` in maugis

    Returns
    -------

    elastic energy in units of :math:`\pi w R (\pi^2 w^2 R / K^2)^{1/3}`

    """
    A = contact_radius
    d = penetration

    return 3/4 * A * (d - A**2 / 3 )**2 + A**5 / 15


def nonequilibrium_elastic_energy_release_rate(penetration, contact_radius):
    r"""

    Returns the nondimensional energy release rate (with respect to the nondimensional area)
    ..math \frac{\partial U_{el}}{\partial pi A^2} = \frac{3}{8 \pi} w_{ref} \frac{1}{A} (\Delta - A^2)^2

    be careful, this is

    Here :math:`\tilde U_{el} = \frac{U_el}{\pi w R (\pi^2 w^2 R / K^2)^{1/3}}` is the nondimensionalized elastic energy

    For the units, see maugis p.290

    Parameters
    ----------
    penetration: :math:`\Delta` in maugis
    contact_radius: :math:`A` in maugis

    Returns
    -------

    """
    return 3 / 8 / np.pi * (penetration - contact_radius**2)**2 / contact_radius


def stress_intensity_factor(contact_radius, penetration, der="0",
                            radius=1, contact_modulus=3./4):
    r"""

    if R is not given, the length and the penetration
    are epressed in units of R

    Parameters:
    -----------
    a: contact radius
    d: penetration
    der: {"0", "1_a", "2_a"}
    R: default 1, optional
    radius of the sphere
    Es: default 3/4, optional
    johnson's contact modulus

    Returns:
    --------
    stress intensity factor K or it's first derivative according to the
    contact_radius

    if R and Es are not given it is in units of 4 / 3 Es \sqrt{R} / R^{der}
    """
    a = contact_radius
    R = radius
    d = penetration
    Es = contact_modulus

    dh = a ** 2 / R # hertzian displacement
    dc = d - dh # displacement imposed to the external crack

    if der == "0":
        return - (Es * dc / np.sqrt(np.pi * a) )
    elif der == "1_a":
        return 1 / 2 * a ** (-3/2) * Es * dc / np.sqrt(np.pi) + Es / np.sqrt(np.pi * a) * 2 * a / R
    elif der == "2_a":
        return - 3 / 4 * a**(-5/2) * Es * dc / np.sqrt(np.pi)


def dispField(r, contact_radius, radius, contact_modulus, work_of_adhesion):
    """
    Parameters
    ----------
    contact_radius : contact radius
    radius : float
        Sphere radius.
    contact_modulus : float
        Contact modulus: Es = E/(1-nu**2) with Young's modulus E and Poisson
        number nu.
    work_of_adhesion : float
        Work of adhesion.

    return a function of the distance from the contact center giving the displacement
    """
    R = radius
    a = contact_radius
    Es = contact_modulus

    P1 = a ** 3 * 4 * Es / 3 / R
    P1mP = np.sqrt(6 * np.pi * w * R * P1)
    assert np.any(r > 0)
    r_a = r / a
    sl_inner = r_a < 1
    sl_outer = np.logical_not(sl_inner)

    u = np.zeros_like(r_a)
    u[sl_outer] = (- P1mP / (Es * np.pi * a) * np.arcsin(1 / r_a[sl_outer])
                   + a ** 2 / (np.pi * R) * (
                               np.sqrt(r_a[sl_outer] ** 2 - 1) + (
                                   2 - r_a[sl_outer] ** 2) * np.arcsin(
                           1 / r_a[sl_outer])))
    u[sl_inner] = + penetration(contact_radius=contact_radius,
                                radius=radius,
                                contact_modulus=contact_modulus,
                                work_of_adhesion=work_of_adhesion) \
                  - 1 / (2 * R) * r[sl_inner] ** 2
    return u


def deformed_profile(r, contact_radius, radius, contact_modulus,
                     work_of_adhesion):
    return 1 / (2 * radius) * r ** 2 \
           - penetration(radius, contact_modulus, work_of_adhesion) \
           + dispField(r,
                       contact_radius, radius, contact_modulus,
                       work_of_adhesion)


def stress_distribution(r, contact_radius, radius, contact_modulus,
                        work_of_adhesion):
    R = radius
    a = contact_radius
    Es = contact_modulus
    w = work_of_adhesion

    P1 = a ** 3 * 4 * Es / 3 / R
    P1mP = np.sqrt(6 * np.pi * w * R * P1)

    assert np.any(r > 0)
    r_a = r / a
    sl_inner = r_a < 1

    sig = np.zeros_like(r_a)
    sig[sl_inner] = P1mP / (2 * np.pi * a ** 2) / np.sqrt(
        1 - r_a[sl_inner] ** 2) - 1.5 * P1 / (np.pi * a ** 2) * \
        np.sqrt(1 - r_a[sl_inner] ** 2)

    return sig
