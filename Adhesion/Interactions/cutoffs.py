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
import scipy.optimize
import numpy as np
from Adhesion.Interactions.Potentials import DecoratedPotential, Potential


class LinearCorePotential(DecoratedPotential):
    """
    Replaces the singular repulsive part of potentials by a linear part. This
    makes potentials maximally robust for the use with very bad initial
    parameters. Consider using this instead of loosing time guessing initial
    states for optimization.

    The repulsive part of the potential is linear, meaning that the pressure is
    constant. This thus corresponds to an ideal plastic model.
    """

    def __init__(self, parent_potential, r_ti=None, hardness=None,
                 htol=1e-5):
        """

        Either the cutoff radius or the hardness (- the gradient at the cutoff)
        should be provided

        Parameters:
        -----------
        r_ti: float, optional
            transition point between linear function
            and lj, defaults to r_min / 2
        hardness: float, optional
            maximum repulsive stress.
            r_ti is choosen so that the maximum repulsive stress is hardness
        htol: float, optional
            relevant only if hardness is provided
            relative tolerance for the hardness value (since the matching
            cutoff value is determined iteratively)
            |(f(r_ti) - hardness) / hardness| < hardnesstol
            default 1e-5
        """
        super().__init__(parent_potential)

        if hardness is not None:
            htol = 1e-5  #

            class FinfinityError(Exception):
                def __init__(self, x):
                    self.x = x

            def f(r):
                pot, grad, curvature = self.parent_potential.evaluate(
                    r,
                    gradient=True,
                    curvature=True)
                f = (- grad - hardness)
                if not np.isfinite(f):
                    raise FinfinityError(r)
                return f

            def fprime(r):
                pot, grad, curvature = self.parent_potential.evaluate(
                    r,
                    gradient=True, curvature=True)
                return - curvature

            self.r_ti = self.parent_potential.r_min

            try:
                sol = scipy.optimize.root_scalar(
                    f, x0=self.r_ti,
                    fprime=fprime,
                    options=dict(maxiter=4000,
                                 tol=htol / abs(fprime(self.r_ti)))
                    # this is the tolerance in |x - x0|
                    )
                assert sol.converged, sol.flag
                self.r_ti = sol.root
            except FinfinityError as err:
                # print("encountered infinity, make use of tweaky method")
                left = err.x

                # make a tweaked f that never gets infinity
                def ftweaked(r):
                    pot, grad, curvature = self.parent_potential.evaluate(
                        r,
                        gradient=True,
                        curvature=True)
                    f = (- grad - hardness)
                    if not np.isfinite(f):
                        return 1000 * hardness
                    return f

                # use brentq so we do not have to tweak fprime as well, and
                # since now have a bracketing interval
                self.r_ti, rr = scipy.optimize.brentq(
                    ftweaked,
                    left, self.r_ti,
                    xtol=1e-10 / abs(fprime(
                        self.r_ti)) * hardness,
                    full_output=True,
                    maxiter=1000)
                # conversion from htol to xtol using the curvature

            # since the curvature was not necessarily close to
            # the curvature at the root we need to check if we meet the
            # tolerence and eventually iterate again
            while abs(f(self.r_ti) / hardness) > htol:
                sol = scipy.optimize.root_scalar(
                    f, x0=self.r_ti,
                    fprime=fprime,
                    options=dict(maxiter=1000,
                                 tol=1e-1 * htol / abs(
                                     fprime(
                                         self.r_ti)) * hardness)
                    # this is the tolerance in |x - x0|
                                                 )
                assert sol.converged
                self.r_ti = sol.root

        else:
            self.r_ti = r_ti if r_ti is not None else parent_potential.r_min/2
        self.lin_part = self._compute_linear_part()

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self.r_ti, self.lin_part
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Parameters:
        -----------
        state:
            result of __getstate__
        """
        superstate, self.r_ti, self.lin_part = state
        super().__setstate__(superstate)

    def _compute_linear_part(self):
        " evaluates the two coefficients of the linear part of the potential"
        f_val, f_prime, dummy = self.parent_potential.evaluate(
            self.r_ti,
            True,
            True)
        return np.poly1d((float(f_prime), f_val - f_prime*self.r_ti))

    def __repr__(self):
        return "{0} -> LinearCorePotential: r_ti = {1.r_ti}".format(
            self.parent_potential.__repr__(), self)

    def evaluate(self, gap, potential=True, gradient=False,
                 curvature=False, mask=None):

        r = np.asarray(gap)
        nb_dim = len(r.shape)
        if nb_dim == 0:
            r.shape = (1,)
        # we use subok = False to ensure V will not be a masked array
        V = np.zeros_like(r, subok=False) \
            if potential else self.SliceableNone()
        dV = np.zeros_like(r, subok=False) \
            if gradient else self.SliceableNone()
        ddV = np.zeros_like(r, subok=False) \
            if curvature else self.SliceableNone()

        sl_core = np.ma.filled(r < self.r_ti, fill_value=False)
        sl_rest = np.logical_not(sl_core)

        # little hack to work around numpy bug
        if np.array_equal(sl_core, np.array([True])):
            V, dV, ddV = self._lin_pot(r, potential, gradient, curvature)
            # raise AssertionError(" I thought this code is never executed")
        else:
            V[sl_core], dV[sl_core], ddV[sl_core] = \
                self._lin_pot(r[sl_core], potential, gradient, curvature)
            V[sl_rest], dV[sl_rest], ddV[sl_rest] = \
                self.parent_potential.evaluate(r[sl_rest],
                                               potential, gradient, curvature)

        return (V if potential else None,
                dV if gradient else None,
                ddV if curvature else None)

    def _lin_pot(self, gap, potential=True, gradient=False, curvature=False):
        """ Evaluates the linear part and its derivatives of the potential.
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
        """
        V = None if potential is False else self.lin_part(gap)
        dV = None if gradient is False else self.lin_part[1]
        ddV = None if curvature is False else 0.
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


class CutoffPotential(DecoratedPotential):
    """
        sets the potential to 0 above the cutoff radius and shifts it up to
        enforce continuity of the potential. This potential hence has a
        discontinuity in the force
    """

    def __init__(self, parent_potential, cutoff_radius):
        """
        Parameters
        ----------
        parent_potential: `Adhesion.Interactions.Potential`
            potential on which to apply the cutoff
        cutoff_radius: float
            distance above which the potential is set to 0
        """
        super().__init__(parent_potential)
        self.cutoff_radius = cutoff_radius
        self.potential_offset = \
            self.parent_potential.evaluate(self.cutoff_radius)[0]

    def __repr__(self):
        return ("{0} -> CutoffPotential: cut-off radius = {1.cutoff_radius} "
                "potential offset: {1.potential_offset}").format(
            self.parent_potential.__repr__(), self)

    def __getstate__(self):
        state = (super().__getstate__(),
                 self.potential_offset,
                 self.cutoff_radius)
        return state

    def __setstate__(self, state):
        superstate, self.potential_offset, self.cutoff_radius = state
        super().__setstate__(superstate)

    @property
    def has_cutoff(self):
        return True

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

    def evaluate(self, gap, potential=True, gradient=False, curvature=False,
                 mask=None):

        inside_mask = np.ma.filled(gap < self.cutoff_radius, fill_value=False)
        if mask is not None:
            inside_mask = np.logical_and(inside_mask, mask)
        V = np.zeros_like(gap) if potential else self.SliceableNone()
        dV = np.zeros_like(gap) if gradient else self.SliceableNone()
        ddV = np.zeros_like(gap) if curvature else self.SliceableNone()

        V[inside_mask], dV[inside_mask], ddV[inside_mask] = \
            self.parent_potential.evaluate(gap[inside_mask],
                                           potential, gradient, curvature,
                                           mask=inside_mask)
        if V[inside_mask] is not None:
            V[inside_mask] -= self.potential_offset
        return (V if potential else None,
                dV if gradient else None,
                ddV if curvature else None)


class ParabolicCutoffPotential(DecoratedPotential):
    """
    Implements a very simple smoothing of a potential, by complementing the
    functional form of the potential with a parabola that brings to zero the
    potential's zeroth, first and second derivative at an imposed (and freely
    chosen) cut_off radius r_c
    """

    def __init__(self, parent_potential, cutoff_radius):
        """
        Parameters:
        -----------
        cutoff_radius: float
            cut-off radius :math:`r_c`
        """
        super().__init__(parent_potential)
        self.cutoff_radius = cutoff_radius

        self.poly = None
        self._compute_poly()
        self._r_min = self._precompute_min()
        self._r_infl = self._precompute_infl()

    def __getstate__(self):
        """ is called and the returned object is pickled as the contents for
            the instance
        """
        state = super().__getstate__(), self.cutoff_radius, \
            self.poly, self.dpoly, self.ddpoly, self._r_min, self._r_infl
        return state

    def __setstate__(self, state):
        """ Upon unpickling, it is called with the unpickled state
        Keyword Arguments:
        state -- result of __getstate__
        """
        superstate, self.cutoff_radius, self.poly, self.dpoly, self.ddpoly, \
            self._r_min, self._r_infl = state
        super().__setstate__(superstate)

    def __repr__(self):
        return "{0} -> ParabolicCutoffPotential: " \
               "cutoff_radius = {1.cutoff_radius}".format(
                self.parent_potential.__repr__(), self)

    def _precompute_min(self):
        """
        computes r_min
        """
        result = scipy.optimize.fminbound(func=lambda r: self.evaluate(r)[0],
                                          x1=0.01 * self.cutoff_radius,
                                          x2=self.cutoff_radius,
                                          disp=1,
                                          xtol=1e-5 * self.cutoff_radius,
                                          full_output=True)
        error = result[2]
        if error:
            raise self.PotentialError(
                ("Couldn't find minimum of potential, something went wrong. "
                 "This was the full minimisation result: {}").format(result))
        return float(result[0])

    def _precompute_infl(self):
        """
        computes r_infl
        """
        result = scipy.optimize.fminbound(
            func=lambda r: self.evaluate(r, False, True, False)[1],
            x1=0.01*self.cutoff_radius,
            x2=self.cutoff_radius,
            disp=1,
            xtol=1e-5*self.cutoff_radius,
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

    def _compute_poly(self):
        """
        computes the coefficients of the corrective parabola
        """
        ΔV, ΔdV, ΔddV = [-float(dummy)
                         for dummy
                         in self.parent_potential.evaluate(
                self.cutoff_radius, potential=True,
                gradient=True,
                curvature=True)]
        ΔdV = ΔdV - ΔddV * self.cutoff_radius
        ΔV -= ΔddV / 2 * self.cutoff_radius ** 2 + ΔdV * self.cutoff_radius

        self.poly = np.poly1d([ΔddV / 2, ΔdV, ΔV])
        self.dpoly = np.polyder(self.poly)
        self.ddpoly = np.polyder(self.dpoly)

    def evaluate(self, gap, potential=True, gradient=False, curvature=False):

        r = np.asarray(gap)
        V = np.zeros_like(r) if potential else self.SliceableNone()
        dV = np.zeros_like(r) if gradient else self.SliceableNone()
        ddV = np.zeros_like(r) if curvature else self.SliceableNone()

        sl_in_range = np.ma.filled(r < self.cutoff_radius, fill_value=False)

        def adjust_pot(r):
            " shifts potentials, if an offset has been set"
            V, dV, ddV = self.parent_potential.evaluate(r, potential, gradient,
                                                        curvature)
            for val, cond, fun in zip(  # pylint: disable=W0612
                    (V, dV, ddV),
                    (potential, gradient, curvature),
                    (self.poly, self.dpoly, self.ddpoly)):
                if cond:
                    val += fun(r)
            return V, dV, ddV

        if np.array_equal(sl_in_range, np.array([True])):
            V, dV, ddV = adjust_pot(r)
        else:
            V[sl_in_range], dV[sl_in_range], ddV[sl_in_range] = adjust_pot(
                r[sl_in_range])

        return (V if potential else None,
                dV if gradient else None,
                ddV if curvature else None)

    @property
    def has_cutoff(self):
        return True


Potential.register_function("linearize_core", LinearCorePotential)
Potential.register_function("cutoff", CutoffPotential)
Potential.register_function("parabolic_cutoff", ParabolicCutoffPotential)
