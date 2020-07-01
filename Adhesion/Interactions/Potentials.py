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
import scipy.optimize

from NuMPI import MPI

from .Interactions import SoftWall




class Potential(SoftWall, metaclass=abc.ABCMeta):
    """ Describes the minimum interface to interaction potentials for
        Adhesion. These potentials are purely 1D, which allows for a few
        simplifications. For instance, the potential energy and forces can be
        computed at any point in the problem from just the one-dimensional gap
        (h(x,y)-z(x,y)) at that point
    """
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

    @abc.abstractmethod
    def __init__(self, r_cut, communicator=MPI.COMM_WORLD):
        super().__init__(communicator)
        self.r_c = r_cut
        if r_cut is not None:
            self.has_cutoff = not math.isinf(self.r_c)
        else:
            self.has_cutoff = False
        if self.has_cutoff:
            self.offset = self.naive_pot(self.r_c)[0]
        else:
            self.offset = 0

        self.curb = None


    @abc.abstractmethod
    def __repr__(self):
        return ("Potential '{0.name}', cut-off radius r_cut = " +
                "{0.r_c}").format(self)

    def __getstate__(self): #TODO: should the energy be serialized ?, I think not
        state = super().__getstate__(), self.has_cutoff, self.offset,  self.r_c, self.curb
        return state

    def __setstate__(self, state):
        superstate, self.has_cutoff, self.offset,  self.r_c, self.curb = state
        super().__setstate__(superstate)

    def compute(self, gap, pot=True, forces=False, curb=False, area_scale=1.):
        # pylint: disable=arguments-differ
        energy, self.gradient, self.curb = self.evaluate(
            gap, pot, forces, curb, area_scale)
        self.energy = self.pnp.sum(energy) if pot else None

    @abc.abstractmethod
    def naive_pot(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets.
            Keyword Arguments:
            r      -- array of distances
            potential    -- (default True) if true, returns potential energy
            gradient -- (default False) if true, returns gradient
            curvature   -- (default False) if true, returns second derivative

        """
        raise NotImplementedError()

    def evaluate(self, r, pot=True, forces=False, curb=False, area_scale=1.):
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
        if np.isscalar(r):
            r = np.asarray(r)
        if r.shape == ():
            r.shape = (1, )

        # we use subok = False to ensure V will not be a masked array
        V = np.zeros_like(r, subok=False) if pot else self.SliceableNone()
        dV = np.zeros_like(r,
                           subok=False) if forces else self.SliceableNone()
        ddV = np.zeros_like(r,
                            subok=False) if curb else self.SliceableNone()

        if np.ma.getmask(r) is not np.ma.nomask:
            sl = np.logical_not(r.mask)
            V[sl], dV[sl], ddV[sl] = self._evaluate( r[sl], pot, forces, curb)
        else:
            V, dV, ddV = self._evaluate( r, pot, forces,curb)

        # note, if in future we want to return masked arrays we should
        # set the fill_value to zero afterward

        return (area_scale * V if pot else None,
                area_scale * dV if forces else None,
                area_scale * ddV if curb else None)


    def _evaluate(self, r, pot=True, forces=False, curb=False):
        """Evaluates the potential and its derivatives
        Keyword Arguments:
        r          -- array of distances
        pot        -- (default True) if true, returns potential energy
        forces     -- (default False) if true, returns forces
        curb       -- (default False) if true, returns second derivative
        """
        # pylint: disable=bad-whitespace
        # pylint: disable=arguments-differ

        inside_slice = np.ma.filled(r < self.r_c, fill_value=False)
        V = np.zeros_like(r) if pot else self.SliceableNone()
        dV = np.zeros_like(r) if forces else self.SliceableNone()
        ddV = np.zeros_like(r) if curb else self.SliceableNone()

        V[inside_slice], dV[inside_slice], ddV[inside_slice] = self.naive_pot(
            r[inside_slice], pot, forces, curb, mask=inside_slice)
        if V[inside_slice] is not None:
            V[inside_slice] -= self.offset
        return (V if pot else None,
                dV if forces else None,
                ddV if curb else None)

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
        max_tensile=self.evaluate(self.r_infl, forces=True)[1]
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

    def naive_pot(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets.
            Keyword Arguments:
            r      -- array of distances
            potential    -- (default True) if true, returns potential energy
            gradient -- (default False) if true, returns gradient
            curvature   -- (default False) if true, returns second derivative

        """
        return self.parent_potential.naive_pot(r, pot, forces, curb)


class LinearCorePotential(ChildPotential):
    """
    Replaces the singular repulsive part of potentials by a linear part. This
    makes potentials maximally robust for the use with very bad initial
    parameters. Consider using this instead of loosing time guessing initial
    states for optimization.

    The repulsive part of the potential is linear, meaning that the pressure is
    constant. This thus corresponds to an ideal plastic model.
    """
    def __init__(self, parent_potential, r_ti=None, hardness=None):
        """
        Parameters:
        -----------
        r_ti: (default r_min/2) transition point between linear function
                   and lj, defaults to r_min
        hardness:
        maximum repulsive stress.
        r_ti is choosen so that the maximum repulsive stress is hardness
        """
        # pylint: disable=super-init-not-called
        # not calling the superclass's __init__ because this is used in diamond
        # inheritance and I do not want to have to worry about python's method
        # nb_grid_pts order
        super().__init__(parent_potential)

        if hardness is not None:
            def f(r):
                pot, grad, curb =  self.parent_potential.evaluate(r,
                forces=True, curb=True)
                return (- grad - hardness)

            def fprime(r):
               pot, pres, curb =  self.parent_potential.evaluate(r, forces=True, curb=True)
               return -curb

            self.r_ti = scipy.optimize.newton(f, parent_potential.r_min, fprime)
        else:
            self.r_ti = r_ti if r_ti is not None else parent_potential.r_min/2
        self.lin_part = self.compute_linear_part()

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self.r_ti, self.lin_part
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.r_ti, self.lin_part = state
        super().__setstate__(superstate)


    def compute_linear_part(self):
        " evaluates the two coefficients of the linear part of the potential"
        f_val, f_prime, dummy = self.parent_potential.evaluate(self.r_ti, True, True)
        return np.poly1d((float(f_prime), f_val - f_prime*self.r_ti))

    def __repr__(self):
        return "{0} -> LinearCorePotential: r_ti = {1.r_ti}".format(self.parent_potential.__repr__(),self)

    def _evaluate(self, r, pot=True, forces=False, curb=False, mask=None):
        """Evaluates the potential and its derivatives
        Keyword Arguments:
        r          -- array of distances
        pot        -- (default True) if true, returns potential energy
        forces     -- (default False) if true, returns forces
        curb       -- (default False) if true, returns second derivative
        """
        # pylint: disable=bad-whitespace
        # pylint: disable=invalid-name

        nb_dim = len(r.shape)
        if nb_dim == 0:
            r.shape = (1,)
        # we use subok = False to ensure V will not be a masked array
        V = np.zeros_like(r, subok=False) if pot else self.SliceableNone()
        dV = np.zeros_like(r, subok=False) if forces else self.SliceableNone()
        ddV = np.zeros_like(r, subok=False) if curb else self.SliceableNone()


        sl_core = np.ma.filled(r < self.r_ti, fill_value=False)
        sl_rest = np.logical_not(sl_core)

        # little hack to work around numpy bug
        if np.array_equal(sl_core, np.array([True])):
            V, dV, ddV = self.lin_pot(r, pot, forces, curb)
            #raise AssertionError(" I thought this code is never executed")
        else:
            V[sl_core], dV[sl_core], ddV[sl_core] = \
                self.lin_pot(r[sl_core], pot, forces, curb)
            V[sl_rest], dV[sl_rest], ddV[sl_rest] = \
                self.parent_potential._evaluate(r[sl_rest], pot, forces, curb)

        return (V if pot else None,
                dV if forces else None,
                ddV if curb else None)


    def lin_pot(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the linear part and its derivatives of the potential.
        Keyword Arguments:
        r      -- array of distances
        pot    -- (default True) if true, returns potential energy
        forces -- (default False) if true, returns forces
        curb   -- (default False) if true, returns second derivative
        """
        V = None if pot is False else self.lin_part(r)
        dV = None if forces is False else self.lin_part[1]
        ddV = None if curb is False else 0.
        return V, dV, ddV

    @property
    def r_min(self):
        """
        convenience function returning the location of the enery minimum
        """
        return self.parent_potential.r_min
    @property
    def r_infl(self):
        """
        convenience function returning the location of the potential's
        inflection point (if applicable)
        """

        return self.parent_potential.r_infl

class ParabolicCutoffPotential(ChildPotential):
    """
        Implements a very simple smoothing of a potential, by complementing the
        functional form of the potential with a parabola that brings to zero the
        potential's zeroth, first and second derivative at an imposed (and freely
        chosen) cut_off radius r_c
    """

    def __init__(self, parent_potential, r_c):
        """
        Keyword Arguments:
        r_c -- user-defined cut-off radius
        """
        super().__init__(parent_potential)
        self.r_c = r_c

        self.poly = None
        self.compute_poly()
        self._r_min = self.precompute_min()
        self._r_infl = self.precompute_infl()

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self.r_c ,self.poly,self.dpoly, self.ddpoly, self._r_min, self._r_infl
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.r_c ,self.poly,self.dpoly, self.ddpoly, self._r_min, self._r_infl= state
        super().__setstate__(superstate)

    def __repr__(self):
        return "{0} -> ParabolaCutoffPotential: r_c = {1.r_c}".format(self.parent_potential.__repr__(),self)

    def precompute_min(self):
        """
        computes r_min
        """
        result = scipy.optimize.fminbound(func=lambda r: self.evaluate(r)[0],
                                          x1=0.01*self.r_c,
                                          x2=self.r_c,
                                          disp=1,
                                          xtol=1e-5*self.r_c,
                                          full_output=True)
        error = result[2]
        if error:
            raise self.PotentialError(
                ("Couldn't find minimum of potential, something went wrong. "
                 "This was the full minimisation result: {}").format(result))
        return float(result[0])

    def precompute_infl(self):
        """
        computes r_infl
        """
        result = scipy.optimize.fminbound(
            func=lambda r: self.evaluate(r, False, True, False)[1],
            x1=0.01*self.r_c,
            x2=self.r_c,
            disp=1,
            xtol=1e-5*self.r_c,
            full_output=True)
        error = result[2]
        if error:
            raise self.PotentialError(
                ("Couldn't find minimumm of derivative, something went wrong. "
                 "This was the full minimisation result: {}").format(result))
        return float(result[0])


    @property
    def r_min(self):
        """
        convenience function returning the location of the energy minimum
        """
        return self._r_min

    @property
    def r_infl(self):
        """
        convenience function returning the location of the inflection point
        """
        return self._r_infl

    def compute_poly(self):
        """
        computes the coefficients of the corrective parabola
        """
        ΔV, forces, ΔddV = [-float(dummy)
                            for dummy
                            in self.naive_pot(
                                self.r_c, pot=True,
                                forces=True,
                                curb=True)]
        ΔdV = -forces - ΔddV*self.r_c
        ΔV -= ΔddV/2*self.r_c**2 + ΔdV*self.r_c

        self.poly = np.poly1d([ΔddV/2, ΔdV, ΔV])
        self.dpoly = np.polyder(self.poly)
        self.ddpoly = np.polyder(self.dpoly)

    def _evaluate(self, r, pot=True, forces=False, curb=False):
        """
        Evaluates the potential and its derivatives
        Keyword Arguments:
        r          -- array of distances
        pot        -- (default True) if true, returns potential energy
        forces     -- (default False) if true, returns forces
        curb       -- (default False) if true, returns second derivative
        """

        V = np.zeros_like(r) if pot else self.SliceableNone()
        dV = np.zeros_like(r) if forces else self.SliceableNone()
        ddV = np.zeros_like(r) if curb else self.SliceableNone()

        sl_in_range = np.ma.filled(r < self.r_c, fill_value=False)

        def adjust_pot(r):
            " shifts potentials, if an offset has been set"
            V, dV, ddV = self.naive_pot(r, pot, forces, curb)
            for val, cond, fun, sgn in zip(  # pylint: disable=W0612
                    (V, dV, ddV),
                    (pot, -forces, curb),
                    (self.poly, self.dpoly, self.ddpoly),
                    (1., -1., 1.)):
                if cond:
                    val += sgn*fun(r)
            return V, dV, ddV

        if np.array_equal(sl_in_range, np.array([True])):
            V, dV, ddV = adjust_pot(r)
        else:
            V[sl_in_range], dV[sl_in_range], ddV[sl_in_range] = adjust_pot(
                r[sl_in_range])

        return (V if pot else None,
                dV if forces else None,
                ddV if curb else None)

