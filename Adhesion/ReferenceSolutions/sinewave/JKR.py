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


Nommenclature:
--------------

TODO:


"""

import numpy as np
from numpy import sqrt, cos, tan, sin, pi, log
from ContactMechanics.ReferenceSolutions import Westergaard


def flatpunch_pressure(x, a):
    """
    solution by koiter

    "Flat punch" solution (uniform deformation on periodic strides)

    x: float, np.array
        in units of lambda
    a: float
        half width of the flat punch

    Returns:
    --------

    Pressure distribution with mean -1

    References:
    -----------

    1.Johnson, K. L. The adhesion of two elastic bodies with slightly wavy surfaces. International Journal of Solids and Structures 32, 423–430 (1995).
    2.Koiter, W. T. An infinite row of collinear cracks in an infinite elastic sheet. Ing. arch 28, 168–172 (1959).
    3.Zilberman, S. & Persson, B. N. J. Adhesion between elastic bodies with rough surfaces. Solid State Communications 123, 173–177 (2002).


    """
    res = np.zeros_like(x)

    sl = abs((x + 1 / 2) % 1 - 1 / 2) < a
    x = x[sl]
    res[sl] = - (1 - (np.cos(pi * a) / np.cos(pi * x)) ** 2) ** (-1. / 2)
    return res


def mean_pressure(a, alpha, der="0"):
    r"""
    mean pressure in units of pi Es h/lambda

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
        return pi * (sin(2*pia) - alpha / cos(pia)**2 / 2 / sqrt(tan(pia)))
    elif der == "2":
        pia = pi * a
        return pi ** 2 * (2 * cos(2 * pia)
                          + alpha * sqrt(tan(pia)) *
                          (1 / sin(2 * pia) ** 2 - 1 / cos(pia) ** 2))
    else:
        raise ValueError('derivative flag should be "0", "1", "2"')


def mean_gap(a, alpha):
    """
    from carbon mangialardi equation (39)
    Parameters
    ----------
    a: half contact width in units of lambda
    alpha:

    Returns
    -------
    mean gap in units of h  # TODO: 2h or h !!!!!!!!

    h is half the peak to valley distance

    Demo
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> a = np.linspace(0, 0.5 )
    >>> ls = [ax.plot(a, mean_gap(a, alpha), alpha ) for alpha in (0.1,0.2,0.5)]
    >>> plt.show()
    """
    return cos(np.pi * a) ** 2 \
        + 2 * mean_pressure(a, alpha) * np.log(sin(np.pi * a))


def pressure(x, a, mean_pressure):
    r"""
    Parameters
    ----------
    x:
        position in units of the period \lambda
    a:
        half contact width in units of the period \lambda
    mean_pressure:
        externally applied mean pressure in units of
        the westergaard full contact pressure pi E^* h / lambda

    Returns
    -------

    """
    flatpunch_load = sin(pi * a) ** 2 - mean_pressure
    return Westergaard._pressure(x, a) \
        - flatpunch_load * flatpunch_pressure(x, a)

# TODO: K's: from the crack front repo
