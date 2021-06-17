#
# Copyright 2020 Antoine Sanner
#           2015-2016, 2020 Lars Pastewka
#           2015-2016 Till Junge
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

r"""

JKR solutions for the indentation of an elastic halfspace by a paraboloid
indenter.

Physical quantities:
--------------------

The parameters of the system are:

- radius : float, optional
      Sphere (actually paraboloid) radius.
- contact_modulus : float, optional
      Contact modulus: :math:`E^* = E/(1-\nu^2)`
      with Young's modulus E and Poisson number :math:`\nu`.
      The default value is so that Maugis's contact Modulus is one
      (:math:`K = 4 / 3 E^*`)
- work_of_adhesion : float, optional
      Work of adhesion.

This module provides implementations of the formulas relating the rigid body
penetration (:code:`penetration`), the indentation force (:code:`force`)
and the radius of the contact disk (:code:`contact_radius`).

If the parameters are not provided, the relations are nondimensional.

Nondimensional units
--------------------

The nondimensionalisation follows
Maugis's book (p.290):

- lengths in the vertical direction
   (penetration, heights, displacements, gaps),
   are in units of

.. math ::

    (\pi^2 w^2 R / K^2)^{1/3}

- lengths in the lateral direction (contact radius) are in units of

.. math ::

    (\pi w R^2 / K)^{1/3}

- forces are in unit of

.. math ::

    \pi w R


Function reference:
===================

"""

import numpy as np
from scipy.optimize import newton


###

def radius_unit(radius, contact_modulus, work_of_adhesion):
    k = 4 / 3 * contact_modulus  # maugis contact modulus
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
    modulus compute contact radius on the stable branch.

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
        Contact modulus: :math:`E^* = E/(1-\nu^2)`
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
          work_of_adhesion=None):
    r"""
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

    Examples
    --------
    >>> JKR.force(contact_radius=2.)
    >>> JKR.force(contact_radius=2., radius=1., contact_modulus=3./4,
    ...           work_of_adhesion=1/np.pi)
    >>> JKR.force(penetration=1.)
    >>> JKR.force(penetration=1.,radius=1., contact_modulus=3./4,
    ...           work_of_adhesion=1/np.pi)
    >>> JKR.force(contact_radius=2., penetration=1.)
    >>> JKR.force(contact_radius=2., penetration=1., radius=1.,
    ...           contact_modulus=3./4,)

    Note that in the last usage, both contact radius and penetration are given
    instead of the work of adhesion

    """
    maugis_modulus = 4 * contact_modulus / 3
    if contact_radius is None and penetration is None:
        raise ValueError("either contact_radius or penetration "
                         "should be provided")
    elif contact_radius is not None and penetration is not None:
        if work_of_adhesion is not None:
            raise ValueError("if both contact_radius and penetration are "
                             "given, the work of adhesion will depend on them"
                             "and can't be prescribed")
        hertzian_force = contact_radius ** 3 * maugis_modulus / radius
        hertzian_pen = contact_radius ** 2 / radius

        flat_punch_pen = penetration - hertzian_pen
        flat_punch_force = \
            2 * contact_radius * flat_punch_pen * contact_modulus

        return hertzian_force + flat_punch_force

    else:
        if work_of_adhesion is None:
            work_of_adhesion = 1 / np.pi
        if contact_radius is None:
            contact_radius = _contact_radius_fun(
                penetration=penetration,
                radius=radius,
                contact_modulus=contact_modulus,
                work_of_adhesion=work_of_adhesion)

        hertzian_force = contact_radius ** 3 * maugis_modulus / radius
        # Force in the hertzian contact at the same radius

        return hertzian_force - np.sqrt(
            6 * np.pi * work_of_adhesion * radius * hertzian_force)


def penetration(contact_radius=None,
                force=None,
                radius=1.,
                contact_modulus=3. / 4,
                work_of_adhesion=1 / np.pi):
    r"""
    Parameters
    ----------
    contact_radius : float or array of floats, optional
        Normal force.
    penetration : float, optional
        rigid body penetration
    radius : float, optional
        Sphere (actually paraboloid) radius.
    contact_modulus : float, optional
        Contact modulus: :math:`E^* = E/(1-\nu^2)`
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
    return 1 / 3 * A * (A ** 2 - np.sqrt(6 * A)) ** 2 + A ** 5 / 15


# dimensional version
# def elastic_energy__a(a, R, Es, w):
#    return (a**5 * 4 * Es) / (9 * R**2)
#       * (1/5 + (1 - R * np.sqrt(9 * np.pi * w / (2*a**3 * Es)))**2)


def nonequilibrium_elastic_energy(penetration, contact_radius):
    r"""

    .. math::

        \frac{U_{el}}{\pi w R (\pi^2 w^2 R / K^2)^{1/3}}
        = \frac{3}{4} A \left(\Delta - \frac{A^2}{3}\right)^2 + \frac{A^5}{15}

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

    return 3 / 4 * A * (d - A ** 2 / 3) ** 2 + A ** 5 / 15


def nonequilibrium_elastic_energy_release_rate(penetration, contact_radius, radius=1, contact_modulus=3./4, der="0"):
    r"""

    Returns the nondimensional energy release rate
    (with respect to the nondimensional area)

    .. math ::
          \frac{\partial U_{el}}{\partial \pi A^2}
          = \frac{3}{8 \pi}  \frac{1}{A} (\Delta - A^2)^2


    :math:`A` is the contact radius, :math:`\Delta` is the penetration.

    With the default values of `radius` and `contact_modulus`, this function returns the energy release rate in units
    of

     - :math:`w` if :math:`\Delta` and :math:`A` are in maugis - JKR units.

     - :math:`E_M R` if :math:`\Delta` and :math:`A` are in units of :math:`R`

    Note that I think perfect consistent use of the maugis-JKR units would require to express the energy release rate
    in units of :math:`\pi w` instead of just :math:`w`. I might need to change this at some point.

    Parameters
    ----------
    penetration: float or np.array
        :math:`\Delta` in maugis
    contact_radius: float or np.array
        in units of :math:`R`
    radius: float
        default 1, optional
        radius of the sphere
    contact_modulus: float
        default 3/4, optional
        johnsons contact modulus
    der: {"0", "1_a", "1_d", "2_a", "2_da"}, optional
        order of the derivative
    Returns
    -------

    """
    prefactor = radius * contact_modulus / (2 * np.pi)
    if der == "0":
        return prefactor * (penetration - contact_radius ** 2) ** 2 / contact_radius
    elif der == "1_a":
        return prefactor * (4 * (contact_radius ** 2 - penetration)
                            - (penetration - contact_radius ** 2) ** 2 / contact_radius ** 2)
    elif der == "1_d":
        return prefactor * 2 * (penetration / contact_radius - contact_radius)
    elif der == "2_da" or der == "2_ad":
        return - prefactor * 2 * (1 + penetration / contact_radius ** 2)
    elif der == "2_a":
        return prefactor * (8 * contact_radius
                            + 4 * (penetration - contact_radius ** 2) / contact_radius
                            + 2 * (penetration - contact_radius ** 2) ** 2 / contact_radius ** 3)
    else:
        raise ValueError(der)


def stress_intensity_factor(contact_radius, penetration, der="0",
                            radius=1, contact_modulus=3. / 4):
    r"""

    if R is not given, the length and the penetration
    are epressed in units of R

    Parameters
    ----------
    contact_radius: float or ndarray of floats
        radius of the contact area
    penetration: float or ndarray of floats
        rigid body penetration
    der: {"0", "1_a", "2_a", "1_d", "2_ad"}
    R: float
        default 1, optional
        radius of the sphere
    Es: float
        default 3/4, optional
        johnson's contact modulus

    Returns
    -------
    stress intensity factor K or it's first derivative according to the
    contact_radius

    if R and Es are not given it is in units of 4 / 3 Es \sqrt{R} / R^{der}
    """
    a = contact_radius
    R = radius
    d = penetration
    Es = contact_modulus

    dh = a ** 2 / R  # hertzian displacement
    dc = d - dh  # displacement imposed to the external crack

    if der == "0":
        return - (Es * dc / np.sqrt(np.pi * a))
    elif der == "1_a":
        return 1 / 2 * a ** (-3 / 2) * Es * dc / np.sqrt(np.pi) \
               + Es / np.sqrt(np.pi * a) * 2 * a / R
    elif der == "2_a":
        return - 3 / 4 * a ** (-5 / 2) * Es * dc / np.sqrt(np.pi)
    elif der == "1_d":
        return - Es / np.sqrt(np.pi * a)
    elif der == "2_ad" or der == "2_da":
        return Es / (2 * np.sqrt(np.pi)) * a ** (-3 / 2)
    else:
        raise ValueError()


def displacement_field(r, contact_radius,
                       radius, contact_modulus, work_of_adhesion):
    r"""
    a function of the distance from the contact center giving the displacement

    Parameters
    ----------
    contact_radius : contact radius
    radius : float
        Sphere radius.
    contact_modulus : float
        Contact modulus: :math:`E^* = E/(1-\nu^2)` with Young's
        modulus E and Poisson number :math:`\nu`.
    work_of_adhesion : float
        Work of adhesion.
    Returns
    -------
    ndarray
        displacements
    """
    R = radius
    a = contact_radius
    Es = contact_modulus

    P1 = a ** 3 * 4 * Es / 3 / R
    P1mP = np.sqrt(6 * np.pi * work_of_adhesion * R * P1)
    assert np.any(r > 0)
    r_a = r / a
    sl_inner = r_a < 1
    sl_outer = np.logical_not(sl_inner)

    u = np.zeros_like(r_a)
    u[sl_outer] = (- P1mP / (Es * np.pi * a) * np.arcsin(1 / r_a[sl_outer])
                   + a ** 2 / (np.pi * R) * (
                           np.sqrt(r_a[sl_outer] ** 2 - 1)
                           + (2 - r_a[sl_outer] ** 2)
                           * np.arcsin(1 / r_a[sl_outer])
                   )
                   )
    u[sl_inner] = + penetration(contact_radius=contact_radius,
                                radius=radius,
                                contact_modulus=contact_modulus,
                                work_of_adhesion=work_of_adhesion) \
        - 1 / (2 * R) * r[sl_inner] ** 2
    return u


def deformed_profile(r, contact_radius, radius=1., contact_modulus=3./4,
                     work_of_adhesion=1/np.pi):
    r"""
    Computes the gap in the JKR contact at radius r
    Parameters
    ----------
    r: float or array of floats
        radius at which to compute the gap
    contact_radius : float or array of floats
        Normal force.
    radius : float, optional
        Sphere (actually paraboloid) radius.
        Default 1.
    contact_modulus : float, optional
        Contact modulus: :math:`E^* = E/(1-\nu^2)`
        with Young's modulus E and Poisson number :math:`\nu`.
        The default value is so that Maugis's contact Modulus is one
        (:math:`K = 4 / 3 E^*`)
    work_of_adhesion : float, optional
        Work of adhesion.
        Default :math:`1 / \pi`
    Returns
    -------
    ndarray
        gaps at radius r
    """
    return 1 / (2 * radius) * r ** 2 \
        - penetration(contact_radius=contact_radius,
                      radius=radius, contact_modulus=contact_modulus, work_of_adhesion=work_of_adhesion) \
        + displacement_field(r,
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
    sig[sl_inner] = P1mP / (2 * np.pi * a ** 2) \
        / np.sqrt(1 - r_a[sl_inner] ** 2) \
        - 1.5 * P1 / (np.pi * a ** 2) \
        * np.sqrt(1 - r_a[sl_inner] ** 2)

    return sig
