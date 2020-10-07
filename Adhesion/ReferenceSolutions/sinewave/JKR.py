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
r"""

References:
-----------

Original paper

Johnson, K. L. The adhesion of two elastic bodies with slightly wavy surfaces.
International Journal of Solids and Structures 32, 423–430 (1995)
DOI: 10.1016/0020-7683(94)00111-9

Asymmetric case and stress intensity factor:

Carbone, G. et al.
Journal of the Mechanics and Physics of Solids 52, 1267–1287 (2004)
DOI: 10.1016/j.jmps.2003.12.001


Physical quantities:
--------------------

The geometry of the sinusoidal indenter is

.. math ::

   2\ h\ sin^2(\pi\ a\ / \lambda )

- :math:`\lambda`: wavelength of the sinusoidal indenter
- :math:`h`: half peak to tale distance

- :math:`E^*`: contact modulus


Nondimensional units
--------------------

- lateral lengths are in unit of :math:`\lambda`
- vertical lengths and displacements are in unit of :math:`h` (or :math:`2 h `)
- Pressures are in units of :math:`p_{wfc} = \pi E^* h/\lambda` the amplitude
of the sinusoidal pressures in the full contact.

- Johnson Parameter :math:`\alpha`.

.. math ::

    \alpha = \frac{\sqrt{2}}{\pi} \frac{\lambda}{h} \sqrt{\frac{w}{E^* \lambda}}


:math:`\alpha^2` can be interpreted as a ratio of surface energy
and the elastic energy necessary for full contact.

.. math ::

    \alpha^2  = \frac{w}{h p_{wfc}} \frac{2}{\pi}

:math:`\alpha` is also the stress intensity factor in nondimensional form

.. math ::

    \alpha = \frac{K}{p_{wfc} \sqrt{\lambda}} = \frac{K}{\pi E^* h / \sqrt{\lambda}}

"""  # noqa E501

import numpy as np
from numpy import sqrt, cos, tan, sin, pi, log
import scipy.optimize

from ContactMechanics.ReferenceSolutions import Westergaard


def flatpunch_pressure(x, a):
    r"""
    
    solution by koiter
    
    Flat punch solution (uniform deformation on periodic strides)
    
    Parameters
    ----------
    
    x: float, np.array
        in units of lambda
    a: float
        half width of the flat punch
    
    Returns
    -------
    
    Pressure distribution with mean -1
    
    References
    ----------
    
    Johnson, K. L. The adhesion of two elastic bodies with slightly wavy surfaces. International Journal of Solids and Structures 32, 423–430 (1995).
    
    Koiter, W. T. An infinite row of collinear cracks in an infinite elastic sheet. Ing. arch 28, 168–172 (1959).
    
    Zilberman, S. & Persson, B. N. J. Adhesion between elastic bodies with rough surfaces. Solid State Communications 123, 173–177 (2002).
    
    """  # noqa: E501, W293
    res = np.zeros_like(x)

    sl = abs((x + 1 / 2) % 1 - 1 / 2) < a
    x = x[sl]
    res[sl] = - (1 - (np.cos(pi * a) / np.cos(pi * x)) ** 2) ** (-1. / 2)
    return res


def mean_pressure(a, alpha, der="0"):
    r"""
    mean pressure in units of :math:`\pi E^* h/ \lambda`

    and it's derivatives

    Parameters
    ----------
    a: half contact width in units of lambda
    alpha: float or array
        johnson parameter
    der: {"0", "1", "2"}
        order of the derivative with respect to  (a/\lambda)
    Returns
    -------
    der="0"
        mean pressure in units of pi Es h/lambda
    der="1"
        d (mean pressure) / d (a) in units of  pi Es h / lambda^2
    der="2"
        d (mean pressure)^2 / d^2 (a) in units of  pi Es h / lambda^3

    """
    pia = pi * a
    if der == "0":
        return sin(pia) ** 2 - alpha * sqrt(tan(pia))
    elif der == "1":
        return pi * (sin(2 * pia) - alpha / cos(pia) ** 2 / 2 / sqrt(tan(pia)))
    elif der == "2":
        pia = pi * a
        return pi ** 2 * (2 * cos(2 * pia)
                          + alpha * sqrt(tan(pia)) *
                          (1 / sin(2 * pia) ** 2 - 1 / cos(pia) ** 2))
    else:
        raise ValueError('derivative flag should be "0", "1", "2"')


_mean_pressure = mean_pressure


def _find_min_max_a(alpha):
    """

    Parameters
    ----------
    alpha

    Returns
    -------
    """
    a_upper = 0.3
    a_lower = 0.2

    def obj(a):
        return _mean_pressure(a, alpha, der="2")

    while obj(a_upper) > 0:
        a_upper += (0.5 - a_upper) * 0.2

    while obj(a_lower) < 0:
        a_lower *= 0.5

    a_inflexion = scipy.optimize.brentq(obj, a_lower, a_upper)

    def obj(a):
        return mean_pressure(a, alpha, der="1")

    while obj(a_upper) > 0:
        a_upper += (0.5 - a_upper) * 0.2
    a_max = scipy.optimize.brentq(obj, a_inflexion, a_upper)

    while obj(a_lower) > 0:
        a_lower *= 0.5

    a_min = scipy.optimize.brentq(obj, a_lower, a_inflexion)

    return a_min, a_inflexion, a_max


def contact_radius(mean_pressure, alpha):
    r"""

    Parameters
    ----------
    mean_pressure:
        mean pressure in units of :math:`\pi Es h/\lambda`
    alpha: float
        johnson parameter

    Returns
    -------
    half the contact width

    """
    a_min, a_inflexion, a_max = _find_min_max_a(alpha)

    if mean_pressure > _mean_pressure(a_max, alpha):
        raise ValueError("Given load is bigger then the max possible value")
    elif mean_pressure < _mean_pressure(a_min, alpha):
        raise ValueError("Given load is smaller then the max possible value")

    return scipy.optimize.brentq(
        lambda a: _mean_pressure(a, alpha) - mean_pressure, a_min, a_max)


def mean_gap(a, alpha):
    r"""
    from carbon mangialardi equation (39)
    
    Parameters
    ----------
    a: float
      half contact width in units of lambda
    alpha: float 
     Johnson Parameter

    Returns
    -------
    mean gap in units of :math:`h`

    h is half the peak to valley distance

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> a = np.linspace(0, 0.5 )
    >>> ls = [ax.plot(a, mean_gap(a, alpha), label="{}".format(alpha) ) for alpha in (0., 0.1,0.2,0.5)]
    >>> leg = ax.legend()
    >>> _ = ax.set_xlabel(r"contact radius $a$ ($\lambda$)");
    >>> _ = ax.set_ylabel(r"mean gap ($h$)");
    >>> plt.show()
    """  # noqa:  E501
    return cos(np.pi * a) ** 2 \
        + 2 * mean_pressure(a, alpha) * np.log(sin(np.pi * a))


def pressure(x, a, mean_pressure):
    r"""
    Parameters
    ----------
    x:
        position in units of the period :math:`\lambda`
    a:
        half contact width in units of the period :math:`\lambda`
    mean_pressure:
        externally applied mean pressure in units of
        the westergaard full contact pressure pi E^* h / lambda

    Returns
    -------

    """
    flatpunch_load = sin(pi * a) ** 2 - mean_pressure
    return Westergaard._pressure(x, a) \
        - flatpunch_load * flatpunch_pressure(x, a)


def elastic_energy(a, mean_pressure):
    r"""
     :math:`\frac{U - U_{flat}}{h p_{wfc} A}`


    Parameters
    ----------
    mean_pressure: in units of :math:`p_{wfc}`
    a: in units of the period :math:`\lambda`

    Returns
    -------
    energy per unit area in units of :math:`h p_{wfc}`


    """
    return - log(sin(pi * a)) * mean_pressure ** 2 + (sin(pi * a) ** 4) / 4


def stress_intensity_factor_asymmetric(a_s, a_o, P, der="0"):
    r"""

    Stress intensity factor at the crack front at `a_s`

    partial derivatives are taken at constant mean pressure P

    units for a_s and a_o: wavelength of the sinewave
    units of pressure: westergaard full contact pressure :math:`\pi E^* h / \lambda`

    returns the stress intensity factor in units of :math:`\pi E^* h / \sqrt{\lambda}`
    or it's partial derivative.

    Notation for the derivative flag:
    for example "1_a_s" returns the partial derivative according `a_s`, holding `a_o` and `P` constant

    Parameters
    ----------
    a_s: float between 0 and 0.5
        position of the (positive x) crack front at which the SIF is computed
    a_o: float between 0 and 0.5
        position of the (negative x) crack front opposite to where the SIF is computed
    P:
        mean pressure
    der: string {"0", "1_a_s", "1_a_o"}
        choose the partial derivative of K to be returned

    References
    ----------
    Carbone, G. et al.
    Journal of the Mechanics and Physics of Solids 52, 1267–1287 (2004)
    DOI: 10.1016/j.jmps.2003.12.001
    """  # noqa:  E501

    a = (a_s + a_o) / 2  # mean contact width
    e = (a_s - a_o) / 2  # excenctricity

    if der == "0":
        return 1 / 2 * np.sqrt(1 / np.tan(np.pi * a)) * (
                np.cos(2 * np.pi * e) - np.cos(2 * np.pi * a_s) - 2 * P)
    elif der == "1_a_s":
        B = (np.cos(2 * np.pi * e) - np.cos(2 * np.pi * a_s) - 2 * P)
        dB = 2 * np.pi * (
                np.sin(2 * np.pi * a_s) - 1 / 2 * np.sin(2 * np.pi * e))
        dsqcotan = - np.pi / 4 * np.tan(np.pi * a) ** (-3 / 2) \
            / np.cos(np.pi * a) ** 2
        return (dsqcotan * B + np.tan(np.pi * a) ** (-1 / 2) * dB) * 1 / 2
    elif der == "1_a_o":
        B = (np.cos(2 * np.pi * e) - np.cos(2 * np.pi * a_s) - 2 * P)
        dB = np.pi * np.sin(2 * np.pi * e)
        dsqcotan = - np.pi / 4 * np.tan(np.pi * a) ** (-3 / 2) \
            / np.cos(np.pi * a) ** 2
        return (dsqcotan * B + np.tan(np.pi * a) ** (-1 / 2) * dB) * 1 / 2


def stress_intensity_factor_symmetric(a, P, der="0"):
    if der == "0":
        return np.sqrt(np.cos(np.pi * a) / np.sin(np.pi * a)) * (
                1 - np.cos(2 * np.pi * a) - 2 * P) * 1 / 2
    elif der == "1_a":
        return np.pi / 2 * (
                -1 / 2 / (np.tan(np.pi * a)) ** (3 / 2)
                / np.cos(np.pi * a) ** 2 * (1 - np.cos(2 * np.pi * a) - 2 * P)
                + 2 * np.sin(2 * np.pi * a)
                / np.sqrt(np.tan(np.pi * a)))
