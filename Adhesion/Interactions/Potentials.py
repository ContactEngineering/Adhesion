#
# Copyright 2020 Antoine Sanner
#           2020 Lars Pastewka
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

"""
Generic potential class, all potentials inherit it
"""

import abc

import numpy as np


from NuMPI import MPI

from .Interactions import SoftWall


class Potential(SoftWall, metaclass=abc.ABCMeta):
    """ Describes the minimum interface to interaction potentials for
        Adhesion. These potentials are purely 1D, which allows for a few
        simplifications. For instance, the potential energy and forces can be
        computed at any point in the problem from just the one-dimensional gap
        (h(x,y)-z(x,y)) at that point
    """
    _functions = {}
    name = "generic_potential"

    class PotentialError(Exception):
        "umbrella exception for potential-related issues"
        pass

    class SliceableNone(object):
        """small helper class to remedy numpy's lack of views on
        index-sliced array views. This construction avoid the computation
        of all interactions as with np.where, and copies"""
        # pylint: disable=too-few-public-methods
        __slots__ = ()

        def __setitem__(self, index, val):
            pass

        def __getitem__(self, index):
            pass

    @classmethod
    def register_function(cls, name, function):
        cls._functions.update({name: function})

    def __getattr__(self, name):
        if name in self._functions:
            def func(*args, **kwargs):
                return self._functions[name](self, *args, **kwargs)

            func.__doc__ = self._functions[name].__doc__
            return func
        else:
            raise AttributeError(
                "Unkown attribute '{}' and no analysis or pipeline function "
                "of this name registered (class {}). Available functions: {}"
                .format(name, self.__class__.__name__,
                        ', '.join(self._functions.keys())))

    def __dir__(self):
        return sorted(super().__dir__() + [*self._functions])

    @abc.abstractmethod
    def __init__(self, communicator=MPI.COMM_WORLD):
        super().__init__(communicator)
        self.curvature = None

    @abc.abstractmethod
    def __repr__(self):
        return ("Potential '{0.name}'").format(self)

    @property
    def has_cutoff(self):
        return False

    def evaluate(self, gap, potential=True, gradient=False, curvature=False,
                 mask=None):
        """Evaluates the potential and its derivatives

        Parameters:
        -----------
        gap: array_like of float
            array of distances between the two surfaces
        potential: bool, optional
            if true, returns potential energy (default True)
        gradient: bool, optional
            if true, returns gradient (default False)
        curvature: bool, optional
            if true, returns second derivative (default False)
        mask: ndarray of bool, optional
            potential is only evaluated on gap[mask]
            this property is used by the child potential
        """
        raise NotImplementedError

    @abc.abstractproperty
    def r_min(self):
        """
        convenience function returning the location of the energy minimum
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def r_infl(self):
        """
        convenience function returning the location of the potential's
        inflection point (if applicable)
        """
        raise NotImplementedError()

    @property
    def max_tensile(self):
        """
        convenience function returning the value of the maximum stress
        (i.e. the value of the gradient at the inflexion point pot.r_infl)
        """
        max_tensile = self.evaluate(self.r_infl, gradient=True)[1]
        return max_tensile.item() if np.prod(max_tensile.shape) == 1 \
            else max_tensile

    @property
    def v_min(self):
        """ convenience function returning the value of the energy minimum
        """
        return float(self.evaluate(self.r_min)[0])

    def pipeline(self):
        return [self]


class DecoratedPotential(Potential):
    def __init__(self, parent_potential):
        self.parent_potential = parent_potential
        self.reduction = parent_potential.reduction
        self.communicator = parent_potential.communicator

    def __getstate__(self):
        state = super().__getstate__(), self.parent_potential
        return state

    def __setstate__(self, state):
        superstate, self.parent_potential = state
        super().__setstate__(superstate)

    def pipeline(self):
        return self.parent_potential.pipeline() + [self]
