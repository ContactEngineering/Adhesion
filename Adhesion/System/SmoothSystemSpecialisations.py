#
# Copyright 2018, 2020 Antoine Sanner
#           2016, 2018, 2020 Lars Pastewka
#           2015-2016 Till Junge
#           2015 junge@Festwemse-3
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
implements the periodic and non-periodic smooth contact systems
"""

from collections import namedtuple
import copy

import numpy as np
import scipy

from Adhesion.System import SmoothContactSystem
from Adhesion.System import IncompatibleResolutionError
from SurfaceTopography import Topography
import Adhesion
import ContactMechanics


# convenient container for storing correspondences betwees small and large
# system
BndSet = namedtuple('BndSet', ('large', 'small'))


class FastSmoothContactSystem(SmoothContactSystem):
    """
    This proxy class tries to take advantage of the system-physical_sizes
    independence of non-periodic FFT-solved systems by determining the required
    minimum system physical_sizes, encapsulating a SmoothContactSystem of
    that physical_sizes and run it almost transparently instead of the full
    system.
    It's almost transparent, because by its nature, the small system does not
    compute displacements everywhere the large system exists. Therefore, for
    full-domain output, an extra evaluation step must be taken. But since using
    the small system significantly increases optimisation speed, this tradeoff
    seems to be well worth the trouble.
    """
    # declare that this class is only a proxy
    _proxyclass = True

    class FreeBoundaryError(Exception):
        """
        called when the supplied system cannot be computed with the current
        constraints
        """
        def __init__(self, message, disp):
            super().__init__(message)
            self.disp = disp

    class BabushkaBoundaryError(Exception):
        """
        Called when the choosen physical_sizes of the Babushka System
        had an Influence on the Solution. In the Future, one may catch
        this error, increase the margin and restart minimization
        """
        pass

    def __init__(self, substrate, interaction, surface, margin=4):
        """ Represents a contact problem
        Parameters:
        -----------
        substrate: An instance of HalfSpace.
            Defines the solid mechanics in the substrate
        interaction: An instance of Interaction.
            Defines the contact
            formulation. If this computes interaction energies,
            forces etc, these are supposed to be expressed per unit
            area in whatever units you use. The conversion is
            performed by the system
        surface:     An instance of SurfaceTopography, defines the profile.
        margin:      (default 4) safety margin (in pixels) around the initial
                     contact area bounding box
        """
        super().__init__(substrate, interaction, surface)
        # create empty encapsulated system
        self.babushka = None
        self.margin = margin
        self.bounds = None
        # coordinates of the bottom-right corner of the small scale domain in
        # the large scale domain
        self.__babushka_offset = None
        # This is the vertical offset between surface and substrate. Determined
        # indentation depth. This class needs to keep track of the offsets for
        # which its babuška has been evaluated, in order to have the ability
        # to interpolate the babuška-results onto the domain
        self.offset = None
        self.energy = None
        self.force = None
        if self.dim == 1:
            raise Exception(
                ("Class '{}' is not fully implemented for 1D. Please help!"
                 "").format(type(self).__name__))

    def shape_minimisation_input(self, in_array):
        """
        For minimisation of smart systems, the initial guess array (e.g.
        displacement) may have a non-intuitive shape and physical_sizes
        (The problem physical_sizes
        may be decreased, as for free, non-periodic systems, or increased as
        with augmented-lagrangian-type issues). Use the output of this function
        as argument x0 for scipy minimisation functions. Also, if your initial
        guess has a shape that makes no sense, this will tell you before you
        get caught in debugging scipy-code

        Arguments:
        in_array -- array with the initial guess. has the intuitive shape you
                    think it has
        """
        if np.prod(self.substrate.nb_domain_grid_pts) == in_array.size:
            return self._get_babushka_array(in_array).reshape(-1)
        elif (np.prod(self.babushka.substrate.nb_domain_grid_pts) ==
              in_array.size):
            return in_array.reshape(-1)
        raise IncompatibleResolutionError()

    def check_margins(self):
        """
        Checks whether the safety margin is sufficient (i.e. the outer ring of
        the force array equals zero
        """
        is_ok = True
        if self.dim == 2:
            is_ok &= (self.interaction_force[:, 0] == 0.).all()
            is_ok &= (self.interaction_force[:, -1] == 0.).all()
            is_ok &= (self.interaction_force[0, :] == 0.).all()
            is_ok &= (self.interaction_force[-1, :] == 0.).all()
        if not is_ok:
            self.deproxified()
            raise self.FreeBoundaryError(
                ("Small system probably too small, increase the margins and "
                 "reevaluate self.objective(...)."), self.disp)

    def check_margins_babushka(self):
        """
        Inspired from check_margins but acts on the babushka System:
        When the pressures are non-zero at the boundary of the babushka-area,
        boundaries have an effect on the Solution
        this raises an error because the area has been chosen too small and the

        """
        is_ok = True
        if self.dim == 2:
            # Sometimes, the Topograsphy.__h is a masked array,
            # where the masked values represent infinite heights
            # At this points there is no interaction

            is_ok &= np.ma.filled(self.babushka.interaction_force[:, 0] == 0.,
                                  True).all()
            is_ok &= np.ma.filled(self.babushka.interaction_force[:, -1] == 0.,
                                  True).all()
            is_ok &= np.ma.filled(self.babushka.interaction_force[0, :] == 0.,
                                  True).all()
            is_ok &= np.ma.filled(self.babushka.interaction_force[-1, :] == 0.,
                                  True).all()

        if not is_ok:
            self.deproxified()
            raise self.BabushkaBoundaryError(
                ("Babushka Area has been chosen too small: "
                 "Pressure is non-zero at the bounds of the Babushka System"))

    @staticmethod
    def handles(substrate_type, interaction_type, surface_type,
                is_domain_decomposed):
        is_ok = True
        # any periodic type of substrate formulation should do
        is_ok &= issubclass(substrate_type,
                            ContactMechanics.Substrate)
        if is_ok:
            is_ok &= ~substrate_type.is_periodic()
        # only soft interactions allowed
        is_ok &= issubclass(interaction_type,
                            Adhesion.SoftWall)

        # any surface should do
        is_ok &= issubclass(surface_type,
                            Topography)

        is_ok &= not is_domain_decomposed
        return is_ok

    def objective(self, offset, disp0=None, gradient=False, disp_scale=1.):
        """
        See super().objective for general description this method's purpose.
        Difference for this class wrt 'dumb' systems:
        Needs an initial guess for the displacement field in order to estimate
        the contact area. returns both the objective and the adapted ininial
        guess as a tuple
        Parameters:
        -----------
        offset: float
            determines indentation depth
        gradient: bool, optional
            whether the gradient is supposed to be used (default False)
        disp0: ndarray of float, optional
            initial guess for displacement field. If
            not chosen appropriately, results may be unreliable. (default zero)
        disp_scale : float, optional
            (default 1.) allows to specify a scaling of the
            dislacement before evaluation. This can be necessary when
            using dumb minimizers with hardcoded convergence criteria
            such as scipy's L-BFGS-B.
        """
        self.create_babushka(offset, disp0, disp_scale)
        return self.babushka.objective(offset, gradient, disp_scale)

    def create_babushka(self, offset, disp0=None, disp_scale=1.):
        """
        Create a (much smaller) system with just the contacting patch plus the
        margin
        """
        # this class needs to remember its offset since the evaluate method
        # does not accept it as argument anymore
        self.offset = offset
        if disp0 is None:
            disp0 = np.zeros(self.substrate.nb_domain_grid_pts)
        gap = self.compute_gap(disp_scale*disp0, offset)
        contact = np.argwhere(gap < self.interaction.cutoff_radius)
        if contact.size == 0:
            contact = np.array(
                np.unravel_index(
                    np.argmin(gap), gap.shape)).reshape((-1, 2))
        # Lower bounds by dimension of the indices of contacting cells
        bnd_lo = np.min(contact, 0)
        # Upper bounds by dimension of the indices of contacting cells
        bnd_up = np.max(contact, 0)

        self.__babushka_offset = tuple(bnd - self.margin for bnd in bnd_lo)
        sm_res = tuple((hi-lo + 2*self.margin for (hi, lo) in
                        zip(bnd_up, bnd_lo)))
        if any(bnd < 0 for bnd in self.__babushka_offset):
            raise self.FreeBoundaryError(
                ("With the current margin of {}, the system overlaps the lower"
                 " bounds by {}. Total nb_grid_pts is {}").format(
                     self.margin, self.__babushka_offset, self.nb_grid_pts),
                disp0)  # nopep8
        if any(res + self.__babushka_offset[i] > self.nb_grid_pts[i] for i, res
               in enumerate(sm_res)):
            raise self.FreeBoundaryError(
                ("With the current margin of {}, the system overlaps the upper"
                 " bounds by {}. Total nb_grid_pts is {}").format(
                    self.margin,
                    tuple(self.__babushka_offset[i]
                          + res - self.nb_grid_pts[i]
                          for i, res in enumerate(sm_res)),
                    self.nb_grid_pts), disp0)  # nopep8

        self.compute_babushka_bounds(sm_res)
        sm_surf = self._get_babushka_array(self.surface.heights(),
                                           np.zeros(sm_res))

        sm_substrate = self.substrate.spawn_child(sm_res)
        sm_surface = Topography(sm_surf, sm_surf.shape)
        # It is necessary to copy the interaction, or else deproxifying an
        # instance of this class changes the babushka!
        self.babushka = SmoothContactSystem(
            sm_substrate, copy.deepcopy(self.interaction), sm_surface)

    def compute_normal_force(self):
        return self.babushka.interaction_force.sum()

    def callback(self, force=False):
        return self.babushka.callback(force=force)

    def evaluate(self, disp, offset, pot=True, forces=False):
        raise Exception(
            "This proxy-class cannot be evaluated. If you do not understand "
            "this, use the base-class instead")

    def deproxified(self):
        """
        Extrapolates the state of the babushka system onto the proxied sytem
        """
        self.substrate.force = self._get_full_array(
            self.babushka.substrate.force)
        self.interaction_force = self._get_full_array(
            self.babushka.interaction_force)
        self.energy = self.babushka.energy
        self.interaction.energy = self.babushka.interaction.energy
        self.substrate.energy = self.babushka.substrate.energy

        self.force = self.substrate.force.copy()
        if self.dim == 1:
            self.force[:self.nb_grid_pts[0]] -= self.interaction_force
        else:
            self.force[:self.nb_grid_pts[0], :self.nb_grid_pts[1]] -= \
              self.interaction_force   # nopep8
        self.disp = self.substrate.evaluate_disp(self.substrate.force)
        return self.energy, self.force, self.disp

    def compute_babushka_bounds(self, babushka_nb_grid_pts):
        """
        returns a list of tuples that contain the equivalent slices in the
        small and the large array. It differentiates between nb_grid_pts and
        nb_domain_grid_pts.
        Parameters:
        babushka_nb_grid_pts -- nb_grid_pts of smaller scale
        """
        def boundary_generator():
            """
            computes slices for the boundaries. helps translating between large
            and small arrays using copy-less ndarray views
            """
            sm_res = babushka_nb_grid_pts
            lg_res = self.nb_grid_pts
            for i in (0, 1):
                for j in (0, 1):
                    sm_slice = tuple((slice(i*sm_res[0], (i+1)*sm_res[0]),
                                      slice(j*sm_res[1], (j+1)*sm_res[1])))
                    lg_slice = tuple((
                        slice(i*lg_res[0]+self.__babushka_offset[0],
                              i*lg_res[0]+sm_res[0]+self.__babushka_offset[0]),
                        slice(j * lg_res[1] + self.__babushka_offset[1],
                              j * lg_res[1] + sm_res[1] +
                              self.__babushka_offset[1])))
                    yield BndSet(large=lg_slice, small=sm_slice)

        self.bounds = tuple((bnd for bnd in boundary_generator()))

    def _get_babushka_array(self, full_array, babushka_array=None):
        """
        returns the equivalent small-scale array representation of a given
        large-scale array. In the case of nb_domain_grid_pts arrays, this
        is a copy. Else a view.
        Parameters:
        full_array     -- large-scale input array
        babushka_array -- optional small-scale output array to overwrite
        """

        # pylint: disable=unused-argument
        def nb_domain_grid_pts():
            "used when arrays correspond to the substrate"
            nonlocal babushka_array
            if babushka_array is None:
                babushka_array = np.zeros(
                    self.babushka.substrate.nb_domain_grid_pts)
            for bnd in self.bounds:
                babushka_array[bnd.small] = full_array[bnd.large]
            return babushka_array

        def normal_nb_grid_pts():
            "used when arrays correspond to the interaction or the surface"
            nonlocal babushka_array
            if babushka_array is None:
                babushka_array = np.zeros(self.babushka.nb_grid_pts)
            bnd = self.bounds[0]
            babushka_array[bnd.small] = full_array[bnd.large]
            return babushka_array
        if full_array.shape == self.nb_grid_pts:
            return normal_nb_grid_pts()
        else:
            return nb_domain_grid_pts()

    def _get_full_array(self, babushka_array, full_array=None):
        """
        returns the equivalent large-scale array representation of a given
        Small-scale array.
        Parameters:
        full_array     -- optional large-scale output array to overwrite
        babushka_array -- small-scale input array
        """

        # pylint: disable=unused-argument
        def nb_domain_grid_pts():
            "used when arrays correspond to the substrate"
            nonlocal full_array
            if full_array is None:
                full_array = np.zeros(
                    self.substrate.nb_domain_grid_pts)
            for bnd in self.bounds:
                full_array[bnd.large] = babushka_array[bnd.small]
            return full_array

        def normal_nb_grid_pts():
            "used when arrays correspond to the interaction or the surface"
            nonlocal full_array
            if full_array is None:
                full_array = np.zeros(self.nb_grid_pts)
            bnd = self.bounds[0]
            full_array[bnd.large] = babushka_array[bnd.small]
            return full_array

        if babushka_array.shape == self.babushka.nb_grid_pts:
            return normal_nb_grid_pts()
        else:
            return nb_domain_grid_pts()

    def minimize_proxy(self, offset, disp0=None, method='L-BFGS-B',
                       options=None, gradient=True, lbounds=None, ubounds=None,
                       tol=None, callback=None, disp_scale=1.,
                       deproxify_everytime=True):
        """
        Convenience function. Eliminates boilerplate code for most minimisation
        problems by encapsulating the use of scipy.minimize for common default
        options. In the case of smart proxy systems, this may also encapsulate
        things like dynamics computation of safety margins, extrapolation of
        results onto the proxied system, etc.

        Parameters:
        offset     -- determines indentation depth
        disp0      -- (default zero) initial guess for displacement field. If
                      not chosen appropriately, results may be unreliable.
        method     -- (defaults to L-BFGS-B, see scipy documentation). Be sure
                      to choose method that can handle high-dimensional
                      parameter spaces.
        options    -- (default None) options to be passed to the minimizer
                      method
        gradient   -- (default True) whether to use the gradient or not
        tol        -- (default None) tolerance for termination. For detailed
                      control, use solver-specific options.
        callback   -- (default None) callback function to be at each iteration
                      as callback(disp_k) where disp_k is the current
                      displacement vector. Instead of a callable, it can be set
                      to 'True', in which case the system's default callback
                      function is called.
        disp_scale -- (default 1.) allows to specify a scaling of the
                      dislacement before evaluation.
        """
        # pylint: disable=arguments-differ
        fun = self.objective(offset, disp0, gradient=gradient,
                             disp_scale=disp_scale)
        if disp0 is None:
            disp0 = np.zeros(self.substrate.nb_domain_grid_pts)
        disp0 = self.shape_minimisation_input(disp0)
        if callback is True:
            use_callback = self.callback(force=gradient)
        elif callback is None:
            def use_callback(disp_k):
                # pylint: disable=missing-docstring
                # pylint: disable=unused-argument
                pass
        else:
            use_callback = callback

        def compound_callback(disp_k):
            """
            The callback first check whether the new state of the system
            violates the physical_sizes restrictions of the babuška
            system before calling the user-provided callback function
            Parameter:
            disp_k -- flattened displacement vector at the current optimization
                      step
            """
            if deproxify_everytime:
                self.deproxified()
            self.check_margins()
            self.check_margins_babushka()
            return use_callback(disp_k)

        bnds = None
        if lbounds is not None and ubounds is not None:
            ubounds = disp_scale * self.shape_minimisation_input(ubounds)
            lbounds = disp_scale * self.shape_minimisation_input(lbounds)
            bnds = tuple(zip(lbounds.tolist(), ubounds.tolist()))
        elif lbounds is not None:
            lbounds = disp_scale * self.shape_minimisation_input(lbounds)
            bnds = tuple(
                zip(lbounds.tolist(), [None for i in range(len(lbounds))]))
        elif ubounds is not None:
            ubounds = disp_scale * self.shape_minimisation_input(ubounds)
            bnds = tuple(
                zip([None for i in range(len(ubounds))], ubounds.tolist()))
        # Scipy minimizers that accept bounds
        bounded_minimizers = {'L-BFGS-B', 'TNC', 'SLSQP'}

        try:
            if method in bounded_minimizers:
                result = scipy.optimize.minimize(
                    fun, x0=disp0, method=method, jac=gradient, tol=tol,
                    bounds=bnds, callback=compound_callback,
                    options=options)
            else:
                result = scipy.optimize.minimize(
                    fun, x0=disp0, method=method, jac=gradient, tol=tol,
                    callback=compound_callback,
                    options=options)
                # if deproxify_everytime:
            self.deproxified()
        except self.FreeBoundaryError as err:
            print("Caught FreeBoundaryError. Reevaluating margins")
            self.check_margins()
            return self.minimize_proxy(offset, err.disp, method, options,
                                       gradient, tol, callback)
        return result
