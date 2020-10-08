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
from Adhesion.Interactions.Potentials import DecoratedPotential, Potential
import numpy as np
import scipy.optimize


class SmoothPotential(DecoratedPotential):
    """
    implements the splining of the potential tail using a fourth order
    polynomial than reaches zero within a finite cutoff radius

    decoupled potential:
    V_lγ (r) = V_l(r) - V_l(r_t) + γ

    spline: (Δr = r-r_t)
                                2        3        4
                           C₂⋅Δr    C₃⋅Δr    C₄⋅Δr
    V_s(Δr) = C₀ - C₁⋅Δr - ────── - ────── - ──────
                             2        3        4

    The formulation allows to choose the contact stiffness (the original
    repulsive zone) independently from the work of adhesion (the energy
    well). By default, the work of adhesion equals -min(V(r)) and the
    transition point r_t between the naive potential and the  spline is at the
    minimumm, however they can be chosen freely.

    The spline is chosen to guarantee continuity of the second derivative
    of the potential, leading to the following conditions:
    (1): V_s' (Δr_t)    =  V_lγ' (r_t)
    (2): V_s''(Δr_t)    =  V_lγ''(r_t)
    (3): V_s  (Δr_c) =  0, where Δr_c = r_c-r_o
    (4): V_s' (Δr_c) =  0
    (5): V_s''(Δr_c) =  0
    (6): V_s  (Δr_m) = -γ, where Δr_m = r_min-r_o
    The unknowns are C_i (i in {0,..4}) and r_c
    r_o is the origin for the spline, which can be chosen freely. The original
    choice was r_o = r_t, but it turned out to be a bad choice, leading to a
    system of nonlinear equation in which only two of the six unknowns can be
    eliminated.

    With r_o = r_c, all coefficients C_i can be eliminated and a good initial
    guess for the remaining scalar nonlinear equation can be computed. see
    _eval_poly_and_cutoff
    """

    def __init__(self, parent_potential, gamma=None, r_t=None):
        """
        Parameters:
        -----------
        parent_potential: Potential instance
            Potential which the spline cutoff should be applied
        gamma: (default -V_min)
            Work of adhesion, defaults to the norm of
            minimum of the potential. Note the sign. γ is assumed to be
            non-negative value
        r_t: (default r_min)
            transition point, defaults to r_min (argmin
            of pontential) can also be 'inflection' to transition at the
            inflection point
        """
        super().__init__(parent_potential)
        self.gamma = gamma if gamma is not None \
            else -self.parent_potential.v_min
        # Warning: this assumes that the minimum of the potential is a negative
        # value. This will fail curiously if you use this class to implement a
        # potential with a positive minimum
        if self.gamma < 0:
            raise self.PotentialError(
                ("γ should be a positive value."
                 "you specified and γ = {}.").format(
                    self.gamma))  # nopep8
        if isinstance(r_t, str):
            if r_t == 'inflection':
                r_t = self.r_infl
            else:
                raise ValueError("only 'inflection' "
                                 "is allowed as string input")
        self.r_t = r_t if r_t is not None else self.r_min
        # coefficients of the spline
        self.coeffs = np.zeros(5)
        self.poly = None
        self.dpoly = None
        self.ddpoly = None
        self._eval_poly_and_cutoff()

    def __getstate__(self):
        state = super().__getstate__(), self.gamma, self.r_t, self.coeffs, \
                self.poly, self.dpoly, self.ddpoly, \
                self.cutoff_radius, self.offset
        return state

    def __setstate__(self, state):
        superstate, self.gamma, self.r_t, self.coeffs, self.poly, \
            self.dpoly, self.ddpoly, self.cutoff_radius, self.offset = state
        super().__setstate__(superstate)

    def __repr__(self):
        return "{0} -> SmoothPotential: gamma = {1.gamma}, r_t = {1.r_t}" \
            .format(self.parent_potential.__repr__(), self)

    def get_r_infl_spline(self):
        """
        compute the inflection Point of the spline. The inflexion Point of the
        smoothed potential could be below r_t and so staying the original one
        """
        if hasattr(self, "poly") and self.poly is not None:
            C4 = self.poly.coeffs[0]
            C3 = self.poly.coeffs[1]
            return self.cutoff_radius - 0.5 * C3 / C4
        else:
            return None

    @property
    def r_infl(self):
        r"""
        convenience function returning the location of the potential's
        inflection point
        Depending on where the transition between the parent_potential and the
        spline has been made this returns the inflection point of the spline or
        of the parent_potential
        """
        r_infl_poly = SmoothPotential.get_r_infl_spline(self)
        if r_infl_poly is not None:
            if r_infl_poly < self.r_t:
                return self.parent_potential.r_infl
                # This is the property implementation in the parent_potential
            else:
                return r_infl_poly
        else:
            # The Spline wasn't determined already
            return self.parent_potential.r_infl
            # This is the old property implementation

    @property
    def r_min(self):
        """
        convenience function returning the location of the enery minimum
        of the parent potential. (The spline begins at radii above r_min)
        """
        return self.parent_potential.r_min

    def evaluate(self, gap, potential=True, gradient=False, curvature=False):
        # if np.isscalar(r):
        #     r = np.asarray(r)
        # nb_dim = len(r.shape)
        # if nb_dim == 0:
        #     r.shape = (1,)
        r = np.asarray(gap)
        V = np.zeros_like(r) if potential else self.SliceableNone()
        dV = np.zeros_like(r) if gradient else self.SliceableNone()
        ddV = np.zeros_like(r) if curvature else self.SliceableNone()

        sl_inner = np.ma.filled(r < self.r_t, fill_value=False)
        sl_rest = np.logical_not(sl_inner)
        # little hack to work around numpy bug
        if np.array_equal(sl_inner, np.array([True])):
            #            raise AssertionError(" I thought this code is never
            #            executed")
            V, dV, ddV = self.parent_potential.evaluate(r,
                                                        potential, gradient,
                                                        curvature)
            V -= self.offset
            return (V if potential else None,
                    dV if gradient else None,
                    ddV if curvature else None)
        else:
            V[sl_inner], dV[sl_inner], ddV[sl_inner] = \
                self.parent_potential.evaluate(r[sl_inner],
                                               potential, gradient, curvature)
        V[sl_inner] -= self.offset

        sl_outer = np.logical_and(
            np.ma.filled(r < self.cutoff_radius, fill_value=False),
            sl_rest)
        # little hack to work around numpy bug
        if np.array_equal(sl_outer, np.array([True])):
            V, dV, ddV = self._spline_pot(r, potential, gradient, curvature)
        else:
            V[sl_outer], dV[sl_outer], ddV[sl_outer] = self._spline_pot(
                r[sl_outer], potential, gradient, curvature)

        return (V if potential else None,
                dV if gradient else None,
                ddV if curvature else None)

    def _spline_pot(self, r, potential=True, gradient=False, curvature=False):
        """ Evaluates the spline part and its derivatives of the potential.
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

        V = dV = ddV = None
        dr = r - self.cutoff_radius
        if potential:
            V = self.poly(dr)
        if gradient:
            dV = self.dpoly(dr)
        if curvature:
            ddV = self.ddpoly(dr)
        return (V, dV, ddV)

    def _eval_poly_and_cutoff(self, xtol=1e-10):
        """
        Computes the coefficients C_i of the spline and the cutoff radius r_c
        based on the work of adhesion γ and the slope V'(r_t) and curvature
        V"(r_t) at the transition point. The calculations leading to the
        formulation I use here are done symbolically in SplineHelper.py. Check
        there for details. Short version:


                                     2        3        4
                                C₂⋅Δr    C₃⋅Δr    C₄⋅Δr
        V_s(Δr) = C₀ - C₁⋅Δr - ────── - ────── - ──────
                                  2        3        4

                                      2        3
        V_s'(Δr) = -C₁ - C₂⋅Δr - C₃⋅Δr  - C₄⋅Δr


                                          2
        V_s"(Δr) = -C₂ - 2⋅C₃⋅Δr - 3⋅C₄⋅Δr

        By applying the following boundary conditions
        (1): V_s'(Δr_t)    =  V_lγ'(r_t)
        (2): V_s"(Δr_t)    =  V_lγ"(r_t)
        (3): V_s (Δr_c) =  0, where Δr_c = r_c-r_o
        (4): V_s'(Δr_c) =  0
        (5): V_s"(Δr_c) =  0
        (6): V_s (Δr_m) = -γ, where Δr_m = r_min-r_o,           if r_t < r_m
             V_s (Δr_t) = -γ + Δγ, where Δγ = V_lγ(r_t) - V_lγ(r_m) else

        The unknowns are C_i (i in {0,..4}) and r_c
        and choosing the origin of the spline r_o at the cut-off r_c, the
        coefficients C_i can be determined explicitly (see SplineHelper.py):

        C₀: 0, C₁: 0, C₂: 0,

            -3⋅V'(r_t) + V"(r_t)⋅Δrt      2⋅V'(r_t) - V"(r_t)⋅Δrt
        C₃: ────────────────────────, C₄: ───────────────────────
                            2                            3
                         Δrt                          Δrt
        The remaining unknown is Δr_t (our proxy for r_c) but that equation has
        no analytical solution (I think), except for r_t = r_m. The equation
        is:
                                               4
                  Δrt⋅(3⋅V'(r_t) - V"(r_t)⋅Δrt)
        f(Δrt) = ────────────────────────────── + γ = 0
                                              3
                  12⋅(2⋅V'(r_t) - V"(r_t)⋅Δrt)

        When r_t = r_m (the default); the solution is
                      _________
                     ╱   3γ
        Δr_t = -2⋅  ╱  ───────
                  ╲╱   V"(r_m)

        this can be used as initial guess for solving f(Δrt). If r_t <= r_min.
        The derivative f'(Δrt) is to bulky to pretty-print here, so look at
        SplineHelper.py if
        you wish to see it.

        If r_t > r_min, the solution is explicit (see SplineHelper.py):

                            ____________________________
                           ╱                          2
                3⋅dV_t - ╲╱  -12⋅(Δγ-γ)⋅ddV_t + 9⋅dV_t
        Δr_t = ────────────────────────────────────────, for r_t > r_min
                                 ddV_t

        The expressions for C₃ and C₄ remain the same.  This expression may
        have numerical stability issues around the inflection point, where
        ddV_t is close to zero and the fraction is essentially 0/0. At the
        exact location of the the inflection point, the solution is

               2⋅(Δγ-γ)
        Δr_t = ────────, for r_t = r_infl.
                 dV_t

        In order to express the solution close to the inflection point, this
        method uses the first order Taylor expansion. The derivative is

        ∂(Δr_t)/∂(ddV_t) at zero :

                           2
                   2⋅(Δγ-γ)
        Δr_t'(0) = ─────────
                          3
                    3⋅dV_t

        In conclusion, with the inputs of r_t, γ, V'(r_t) and V"(r_t), we can
        conpute the objective function f(Δrt) and its derivative f'(Δrt), which
        should be roughly constant around the solution (inspect output of
        SplineHelper.py). If r_m = r_t, we can directly compute Δrt. Else, with
        the additional input of V"(r_m), we can compute a good initial guess to
        solve the problem numerically.


        Parameters:
        -----------
        xtol: float, optional
             tolerance for numerical solution. Is scaled
             by γ internally. (default 1e-10)
        """
        dummy, gradient_t, ddV_t = self.parent_potential.evaluate(
            self.r_t, potential=False, gradient=True, curvature=True)
        dV_t = gradient_t

        def spline(Δrt):
            " from SplineHelper.py"
            C_i = [0, 0, 0, (-3 * dV_t + ddV_t * Δrt) / Δrt ** 2,
                   (2 * dV_t - ddV_t * Δrt) / Δrt ** 3]
            # in numpy polynomial form
            polycoeffs = [-C_i[4] / 4,
                          -C_i[3] / 3,
                          0, 0, 0]
            return np.poly1d(polycoeffs)

        if self.r_t <= self.r_min:
            dummy, dummy, ddV_m = self.parent_potential.evaluate(
                self.r_min, potential=False, gradient=False, curvature=True)

            def inner_obj_fun(Δrt, gam_star):
                " from SplineHelper.py"
                return (Δrt * (3 * dV_t - ddV_t * Δrt) ** 4 /
                        (12 * (2 * dV_t - ddV_t * Δrt) ** 3) +
                        gam_star)

            def inner_obj_derivative(Δrt, dummy):
                " from SplineHelper.py"
                return ((3 * dV_t - ddV_t * Δrt) ** 3 *
                        (3 * ddV_t * Δrt * (3 * dV_t - ddV_t * Δrt) +
                         (2 * dV_t - ddV_t * Δrt) *
                         (3 * dV_t - 5 * ddV_t * Δrt)) /
                        (12 * (2 * dV_t - ddV_t * Δrt) ** 4))

            # start with initial guess:
            Δrt0 = -2 * np.sqrt(3 * self.gamma / ddV_m)
            gam_star0 = self.gamma

            options = dict(xtol=self.gamma * xtol)
            Δrt = None

            def outer_obj_fun(gam):
                """
                Outer loop of minimization, finds the work of adhesion γ.
                Convergence of Δr_t (the proxy for the cutoff) is assured by
                the inner loop defined by inner_obj_fun."""
                sol = scipy.optimize.root(
                    inner_obj_fun, Δrt0, args=(gam,),
                    jac=inner_obj_derivative, options=options)
                nonlocal Δrt
                if sol.success:
                    Δrt = sol.x.item()
                    offset = self.parent_potential.evaluate(self.r_t)[0] - \
                        np.array(spline(Δrt)(Δrt))
                    error = self.parent_potential.evaluate(self.r_min)[0] \
                        - offset + self.gamma

                    return error
                else:
                    err_str = ("Evaluation of spline for potential '{}' "
                               "failed. Please check whether the inputs make "
                               "sense").format(self)
                    raise self.PotentialError(err_str)

            def outer_obj_derivative(dummy):
                " Jacobian of outer loop, see SplineHelper.py for details"
                return np.array([1.])

            sol = scipy.optimize.root(
                outer_obj_fun, gam_star0,
                jac=outer_obj_derivative, options=options)

            if not sol.success:
                err_str = ("Evaluation of spline for potential '{}' failed. "
                           "Please check whether the inputs make "
                           "sense").format(self)
                raise self.PotentialError(err_str)
        else:
            V_m_t = self.parent_potential.evaluate(
                np.array([self.r_t, self.r_min]))[0]
            Δgamma = V_m_t[0] - V_m_t[1]
            dgam = (Δgamma - self.gamma)
            if abs(ddV_t) > 1e-10:
                # general case
                Δrt = ((3 * dV_t - np.sqrt(
                    9 * dV_t ** 2 - 12 * dgam * ddV_t)) /
                       ddV_t)
            else:
                # close to numerically difficult inflection point: Taylor
                val = 2 * dgam / dV_t
                derivative = 2 * dgam ** 2 / (3 * dV_t ** 3)
                Δrt = val + derivative * ddV_t

        self.cutoff_radius = self.r_t - Δrt
        self.poly = spline(Δrt)
        self.dpoly = np.polyder(self.poly)
        self.ddpoly = np.polyder(self.dpoly)
        self.offset = -(self._spline_pot(self.r_t)[0] -
                        self.parent_potential.evaluate(self.r_t)[0])

    def __eval_poly_and_cutoff_legacy(self, xtol=1e-10):
        """ Seems to be a bad method, do not use in general, will likely
            disappear.

            Computes the coefficients of the spline and the cutoff based on σ
            and ε. Since this is a non-linear system of equations, this
            requires some numerics.
            The equations derive from the continuity conditions in the class's
            docstring. A few simplifications can be made to conditions (1),
            (2), and (6):

            (1): C_1 = -V_l' (r_t)

            (2): C_2 = -V_l''(r_t)

            (6): C_0 = + γ + C_1*Δr_m + C_2/2*Δr_m^2
                       + C_3/3*Δr_m^3 + C_4/4*Δr_m^4
            Also, we have some info about the shape of the spline:
            since Δr_c is both an inflection point and an extremum and it has
            to be on the right of the minimum, the minimum has to be the
            leftmost extremum of the spline and therefore the global minimum.
            It follows:

            1) C_4 < 0
            2) we know sgn(C_1) and sgn(C_2)
            3) any reasonable choice of r_t leads to a Δr_c > 0

            from (5), we get the 2 possible solutions for Δr_c(C_2, C_3, C_4):
                   ⎡         ________________   ⎛        ________________⎞ ⎤
                   ⎢        ╱              2    ⎜       ╱              2 ⎟ ⎥
            Δr_c = ⎢-C₃ + ╲╱  -3⋅C₂⋅C₄ + C₃    -⎝C₃ + ╲╱  -3⋅C₂⋅C₄ + C₃  ⎠ ⎥
                   ⎢─────────────────────────, ────────────────────────────⎥
                   ⎣           3⋅C₄                        3⋅C₄            ⎦
            This however seems to lead to intractable polynomial equations
            (minutes of sympy without finding a solution, ran out of patience)

            Numerical approach: equations (1) and (2) are evauated immediately,
            Then the system of equations

                   ⎡                                   2         ⎤
                   ⎢          -C₂ - 2⋅C₃⋅Δrc - 3⋅C₄⋅Δrc          ⎥
                   ⎢                                             ⎥
                   ⎢                            2         3      ⎥
                   ⎢       -C₁ - C₂⋅Δrc - C₃⋅Δrc  - C₄⋅Δrc       ⎥
                   ⎢                                             ⎥
                   ⎢                       2         3         4 ⎥
            F(x) = ⎢                 C₂⋅Δrc    C₃⋅Δrc    C₄⋅Δrc  ⎥ = 0
                   ⎢   C₀ - C₁⋅Δrc - ─────── - ─────── - ─────── ⎥
                   ⎢                    2         3         4    ⎥
                   ⎢                                             ⎥
                   ⎢                    2         3         4    ⎥
                   ⎢              C₂⋅Δrm    C₃⋅Δrm    C₄⋅Δrm     ⎥
                   ⎢C₀ - C₁⋅Δrm - ─────── - ─────── - ─────── - γ⎥
                   ⎣                 2         3         4       ⎦
            is solved. Note that a positive work of adhesion (sticky surface)
            has a negative value for γ. The jacobian is:
                   ⎡                 2                                  ⎤
                   ⎢0  -2⋅Δrc  -3⋅Δrc           -2⋅C₃ - 6⋅C₄⋅Δrc        ⎥
                   ⎢                                                    ⎥
                   ⎢       2        3                               2   ⎥
                   ⎢0  -Δrc     -Δrc       -C₂ - 2⋅C₃⋅Δrc - 3⋅C₄⋅Δrc    ⎥
                   ⎢                                                    ⎥
                   ⎢       3       4                                    ⎥
            G(x) = ⎢   -Δrc    -Δrc                          2         3⎥
                   ⎢1  ──────  ──────   -C₁ - C₂⋅Δrc - C₃⋅Δrc  - C₄⋅Δrc ⎥
                   ⎢     3       4                                      ⎥
                   ⎢                                                    ⎥
                   ⎢       3       4                                    ⎥
                   ⎢   -Δrm    -Δrm                                     ⎥
                   ⎢1  ──────  ──────                  0                ⎥
                   ⎣     3       4                                      ⎦

            with x = [C₀  C₃  C₄  Δrc]

        Parameters:
        -----------
        xtol: float, optional
            tolerance for numerical solution. Is multiplied by ε
            internally.

        """
        # pylint: disable=bad-whitespace
        # pylint: disable=bad-continuation
        # pylint: disable=invalid-name
        # known coeffs
        dummy, dV, ddV = self.parent_potential.evaluate(
            self.r_t, potential=False, gradient=True, curvature=True)
        C1 = self.coeffs[1] = -dV
        C2 = self.coeffs[2] = -ddV
        gam = -self.gamma
        r_t = self.r_t
        dr_m = self.r_min - r_t

        def obj_fun(x):
            "the root of this function defined the spline coefficients"
            C0, C3, C4, dr_c = x
            return np.array(
                [- C2 - 2 * C3 * dr_c - 3 * C4 * dr_c ** 2,
                 - C1 - C2 * dr_c - C3 * dr_c ** 2 - C4 * dr_c ** 3,
                 C0 - C1 * dr_c - C2 / 2 * dr_c ** 2 - C3 / 3 * dr_c ** 3 -
                 C4 / 4 * dr_c ** 4,
                 C0 - C1 * dr_m - C2 / 2 * dr_m ** 2 - C3 / 3 * dr_m ** 3 -
                 C4 / 4 * dr_m ** 4 - gam])

        def jacobian(x):
            "evaluate the gradient of the objetive function"
            dummy, C3, C4, dr_c = x
            return np.array(
                [[0, -2 * dr_c, -3 * dr_c ** 2,
                  -2 * C3 - 6 * C4 * dr_c],
                 [0, -dr_c ** 2, -dr_c ** 3,
                  -C2 - 2 * C3 * dr_c - 3 * C4 * dr_c ** 2],
                 [1, -dr_c ** 3 / 3, -dr_c ** 4 / 4,
                  -C1 - C2 * dr_c - C3 * dr_c ** 2 - C4 * dr_c ** 3],
                 [1, -dr_m ** 3 / 3, -dr_m ** 4 / 4,
                  0]])

        C3guess = -C2

        x0 = np.array([-gam, C3guess, 0, 2 * self.r_min])
        options = dict(xtol=-gam * xtol)
        sol = scipy.optimize.root(obj_fun, x0, jac=jacobian, options=options)
        if sol.success:
            self.coeffs[0], self.coeffs[3], self.coeffs[
                4], self.cutoff_radius = sol.x
            self.cutoff_radius += self.r_t
            # !!WARNING!! poly is 'backwards': poly = [C4, C3, C2, C1, C0] and
            # all coeffs except C0 have the wrong sign, furthermore they are
            # divided by their order
            polycoeffs = [-self.coeffs[4] / 4,
                          -self.coeffs[3] / 3,
                          -self.coeffs[2] / 2,
                          -self.coeffs[1],
                          +self.coeffs[0]]
            self.poly = np.poly1d(polycoeffs)
            self.dpoly = np.polyder(self.poly)
            self.ddpoly = np.polyder(self.dpoly)
            self.offset = -(self._spline_pot(self.r_t)[0] -
                            self.parent_potential.evaluate(self.r_t)[0])
        else:
            err_str = ("Evaluation of spline for potential '{}' failed. Please"
                       " check whether the inputs make sense").format(self)
            raise self.PotentialError(err_str)


Potential.register_function("spline_cutoff", SmoothPotential)
