#
# Copyright 2018-2019 Antoine Sanner
#           2019 Lintao Fang
#           2016, 2019 Lars Pastewka
#           2016 Till Junge
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
Generic potential class, all potentials inherit it
"""

import math
import abc

import numpy as np


from NuMPI import MPI

from .Interactions import SoftWall


# TODO: This should probably be moved together with Interactions

class Potential(SoftWall, metaclass=abc.ABCMeta):
    """ Describes the minimum interface to interaction potentials for
        Adhesion. These potentials are purely 1D, which allows for a few
        simplifications. For instance, the potential energy and forces can be
        computed at any point in the problem from just the one-dimensional gap
        (h(x,y)-z(x,y)) at that point
    """
    _functions={}
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

    def compute(self, gap, potential=True, gradient=False, curvature=False, area_scale=1.):
        # pylint: disable=arguments-differ
        energy, self.gradient, self.curvature = self.evaluate(gap, potential, gradient,
                                                         curvature, area_scale=area_scale)
        self.energy = self.pnp.sum(energy) if potential else None

    @abc.abstractmethod
    def naive_pot(self, r, potential=True, gradient=False, curvature=False):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets.
            Keyword Arguments:
            r      -- array of distances
            potential    -- (default True) if true, returns potential energy
            gradient -- (default False) if true, returns gradient
            curvature   -- (default False) if true, returns second derivative

        """
        raise NotImplementedError()

    def evaluate(self, gap, potential=True, gradient=False, curvature=False,
                 area_scale=1.):
        """Evaluates the potential and its derivatives
        Parameters:
        -----------
        r:
            array of distances between the two surfaces
        potential: bool (default True)
            if true, returns potential energy
        gradient: bool, (default False)
            if true, returns gradient
        curvature: bool, (default False)
            if true, returns second derivative
        area_scale: float (default 1.)
            scale by this.
            (Interaction quantities are supposed to be expressed per unit
            area, so systems need to be able to scale their response for their
            nb_grid_pts)
        """
        if np.isscalar(gap):
            gap = np.asarray(gap)
        if gap.shape == ():
            gap.shape = (1,)

        V, dV, ddV = self._evaluate(gap, potential, gradient, curvature)

        return (area_scale * V if potential else None,
                area_scale * dV if gradient else None,
                area_scale * ddV if curvature else None)

    def _evaluate(self, r, potential=True, gradient=False, curvature=False,
                  mask=None):
        """Evaluates the potential and its derivatives
        Keyword Arguments:
        r          -- array of distances
        pot        -- (default True) if true, returns potential energy
        gradient     -- (default False) if true, returns gradient
        curvature       -- (default False) if true, returns second derivative
        """
        # pylint: disable=bad-whitespace
        # pylint: disable=arguments-differ
        if mask is None:
            mask = (slice(None),) * len(r.shape)
        return self.naive_pot(r, potential, gradient, curvature, mask=mask)

    @abc.abstractproperty
    def r_min(self):
        """
        convenience function returning the location of the enery minimum
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
        convenience function returning the value of the maximum stress (at r_infl)
        """
        max_tensile= self.evaluate(self.r_infl, gradient=True)[1]
        return max_tensile.item() if np.prod(max_tensile.shape) == 1 else max_tensile

    @property
    def v_min(self):
        """ convenience function returning the energy minimum
        """
        return float(self.evaluate(self.r_min)[0])

    @property
    def naive_min(self):
        """ convenience function returning the energy minimum of the bare
           potential

        """
        return self.naive_pot(self.r_min)[0]

class ChildPotential(Potential):
    def __init__(self, parent_potential):
        self.parent_potential = parent_potential
        self.pnp = parent_potential.pnp
        self.communicator = parent_potential.communicator

    def __getattr__(self, item):
        #print("looking up item {} in {}".format(item, self.parent_potential))
        #print(self.parent_potential)
        if item[:2]=="__" and item[-2:]=="__":
            #print("not allow to lookup")
            raise AttributeError
        else:
            return getattr(self.parent_potential, item)

    def __getstate__(self):
        state = super().__getstate__(), self.parent_potential
        return state

    def __setstate__(self, state):
        superstate, self.parent_potential = state
        super().__setstate__(superstate)

    def naive_pot(self, r, potential=True, gradient=False, curvature=False):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets.
            Keyword Arguments:
            r      -- array of distances
            potential    -- (default True) if true, returns potential energy
            gradient -- (default False) if true, returns gradient
            curvature   -- (default False) if true, returns second derivative

        """
        return self.parent_potential.naive_pot(r, potential, gradient, curvature)



