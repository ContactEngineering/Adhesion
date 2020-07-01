import scipy.optimize
import numpy as np
from Adhesion.Interactions.Potentials import ChildPotential

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
