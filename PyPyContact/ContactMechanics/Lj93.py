#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Lj93.py

@author Till Junge <till.junge@kit.edu>

@date   22 Jan 2015

@brief  9-3 Lennard-Jones potential for wall interactions

@section LICENCE

 Copyright (C) 2015 Till Junge

PyPyContact is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyPyContact is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

from . import Potential


class LJ93(Potential):
    """ 9-3 Lennard-Jones potential with optional cutoff radius.

        9-3 Lennard-Jones potential:
        V_l (r) = ε[ 2/15 (σ/r)**9 - (σ/r)**3]

        When used with a cutoff radius, the potential is shifted in orden to
        guarantee continuity of forces

        V_lc (r) = V_l (r) - V_l (r_c)
    """

    name = "lj9-3"

    def __init__(self, epsilon, sigma, r_cut=float('inf')):
        """
        Keyword Arguments:
        epsilon -- Lennard-Jones potential well ε
        sigma   -- Lennard-Jones distance parameter σ
        r_cut   -- (default +∞) optional cutoff radius
        """
        self.eps = epsilon
        self.sig = sigma
        super().__init__(r_cut)

    def __repr__(self, ):
        return ("Potential '{0.name}': ε = {0.eps}, σ = {0.sig}, "
                "r_c = {1}").format(
                    self, self.r_c if self.has_cutoff else '∞')

    @property
    def r_min(self):
        """convenience function returning the location of the enery minimum

                6 ___  5/6
        r_min = ╲╱ 2 ⋅5   ⋅σ
                ────────────
                     5
        """
        return self.sig*(2*5**5)**(1./6)/5.

    def naive_pot(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets. These have been collected in a single method to reuse the
            computated LJ terms for efficiency
                         ⎛   3       9⎞
                         ⎜  σ     2⋅σ ⎟
            V_l(r) =   ε⋅⎜- ── + ─────⎟
                         ⎜   3       9⎟
                         ⎝  r    15⋅r ⎠

                         ⎛   3       9⎞
                         ⎜3⋅σ     6⋅σ ⎟
            V_l'(r) =  ε⋅⎜──── - ─────⎟
                         ⎜  4       10⎟
                         ⎝ r     5⋅r  ⎠

                         ⎛      3       9⎞
                         ⎜  12⋅σ    12⋅σ ⎟
            V_l''(r) = ε⋅⎜- ───── + ─────⎟
                         ⎜     5      11 ⎟
                         ⎝    r      r   ⎠

            Keyword Arguments:
            r      -- array of distances
            pot    -- (default True) if true, returns potential energy
            forces -- (default False) if true, returns forces
            curb   -- (default False) if true, returns second derivative
        """
        # pylint: disable=bad-whitespace
        # pylint: disable=invalid-name
        V = dV = ddV = None
        sig_r3 = (self.sig/r)**3
        sig_r9 = sig_r3**3
        if pot:
            V = self.eps*(2./15*sig_r9 - sig_r3)
        if forces or curb:
            eps_r = self.eps/r
        if forces:
            # Forces are the negative gradient
            dV = eps_r*(6./5*sig_r9 - 3*sig_r3)
        if curb:
            ddV = 12*eps_r/r*(sig_r9 - sig_r3)
        return (V, dV, ddV)
