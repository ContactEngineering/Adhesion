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

import math

import numpy as np


###

def contact_radius(force,
                   radius=1.,
                   contact_modulus=3. / 4,
                   work_of_adhesion=1 / np.pi):
    r"""
    Given normal load, sphere radius and contact modulus compute contact radius
    and peak pressure.

    if only force is provided, it is assumed that the nondimensionalisation
    from Maugis's book is used.

    Parameters
    ----------
    force : float
        Normal force.
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
    A = force + 3 * work_of_adhesion * math.pi * radius
    B = np.sqrt(6 * work_of_adhesion * math.pi * radius * force + (
                3 * work_of_adhesion * math.pi * radius) ** 2)

    fac = 3. * radius / (4. * contact_modulus)
    A *= fac
    B *= fac

    return (A + B) ** (1. / 3), (A - B) ** (1. / 3)



