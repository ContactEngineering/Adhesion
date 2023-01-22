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
Defines the interface for Adhesion systems
"""

import numpy as np

import Adhesion
import ContactMechanics
import SurfaceTopography
from ContactMechanics.Tools import compare_containers
from ContactMechanics.Systems import IncompatibleResolutionError, SystemBase


class SmoothContactSystem(SystemBase):
    """
    For smooth contact mechanics (i.e. the ones for which optimization is only
    kinda-hell
    """

    def __init__(self, substrate, interaction, surface):
        """ Represents a contact problem
        Parameters
        ----------
        substrate: An instance of HalfSpace.
            Defines the solid mechanics in the substrate
        interaction: Adhesion.Interactions.SoftWall
            Defines the contact formulation.
            If this computes interaction energies, forces etc,
            these are supposed to be expressed per unit area in whatever units
             you use. The conversion is performed by the system
        surface: SurfaceTopography.Topography
            Defines the profile.
        """
        if surface.has_undefined_data:
            raise ValueError("The topography you provided contains undefined "
                             "data")
        super().__init__(substrate=substrate, surface=surface)
        self.interaction = interaction
        if not compare_containers(surface.nb_grid_pts, substrate.nb_grid_pts):
            raise IncompatibleResolutionError(
                ("the substrate ({}) and the surface ({}) have incompatible "
                 "nb_grid_ptss.").format(
                    substrate.nb_grid_pts, surface.nb_grid_pts))  # nopep8
        self.dim = len(self.substrate.nb_grid_pts)
        self.energy = None
        self.force = None
        self.force_h = None
        self.interaction_energy = None
        self.interaction_force = None
        self.heights_k = None
        self.engine = self.substrate.fftengine

        if hasattr(substrate.fftengine, "register_halfcomplex_field") and self.engine.communicator.size == 1:
            # avoids the initialization to fail if we use an fftengine without these hcffts implemnented
            # preconditionning is not parallelized yet, this we have no parallelized wrapper for hcfft in muFFT yet
            self.real_buffer = self.engine.fetch_or_register_halfcomplex_field("hc-real-space", 1)
            self.fourier_buffer = self.engine.fetch_or_register_halfcomplex_field("hc-fourier-space", 1)

            self.stiffness_k = self._compute_stiffness_k()

    @property
    def nb_grid_pts(self):
        return self.surface.nb_grid_pts

    @staticmethod
    def handles(substrate_type, interaction_type, surface_type, comm):
        is_ok = True
        # any periodic type of substrate formulation should do
        is_ok &= issubclass(substrate_type,
                            ContactMechanics.Substrate)

        # only soft interactions allowed
        is_ok &= issubclass(interaction_type,
                            Adhesion.SoftWall)

        # any surface should do
        is_ok &= issubclass(surface_type,
                            SurfaceTopography.UniformTopographyInterface)
        return is_ok

    def compute_repulsive_force(self):
        "computes and returns the sum of all repulsive forces"
        return self.reduction.sum(np.where(
            self.interaction_force > 0, self.interaction_force, 0
        ))

    def compute_attractive_force(self):
        "computes and returns the sum of all attractive forces"
        return self.reduction.sum(np.where(
            self.interaction_force < 0, self.interaction_force, 0
        ))

    def _compute_stiffness_k(self):
        """
        computes and returns the wavevectors q that exist for the surfaces
        physical_sizes and nb_grid_pts as one vector of components per
        dimension
        """

        vectors = []
        q = []
        nb_dims = len(self.substrate.nb_grid_pts)
        # if nb_dims == 1:
        #     nb_grid_pts = [self.substrate.nb_grid_pts]
        #     physical_sizes = [self.substrate.physical_sizes]
        for dim in range(nb_dims):
            vectors.append(2 * np.pi * np.fft.fftfreq(
                self.substrate.nb_grid_pts[dim],
                self.substrate.physical_sizes[dim] /
                self.substrate.nb_grid_pts[dim]))
        if nb_dims == 1:
            q = vectors[0]
            q[0] = q[1]
        elif nb_dims == 2:
            qx = vectors[0]
            qy = vectors[1]
            q = np.sqrt(
                (qx * qx).reshape((-1, 1)) + (qy * qy).reshape((1, -1)))
            q[0, 0] = (q[0, 1] + q[1, 0]) / 2

        return 0.5 * self.substrate.contact_modulus * abs(q)

    def compute_normal_force(self):
        "computes and returns the sum of all forces"
        return self.reduction.sum(self.interaction_force)

    def compute_repulsive_contact_area(self):
        "computes and returns the area where contact pressure is repulsive"
        return self.compute_nb_repulsive_pts() * self.area_per_pt

    def compute_attractive_contact_area(self):
        "computes and returns the are where contact pressure is attractive"
        return self.compute_nb_attractive_pts() * self.area_per_pt

    def compute_nb_contact_pts(self):
        """
        compute and return the number of contact points. Note that this is of
        no physical interest, as it is a purely numerical artefact
        """
        return self.reduction.sum(
            np.where(self.interaction_force != 0., 1., 0.))

    def compute_nb_repulsive_pts(self):
        """
        compute and return the number of contact points under repulsive
        pressure. Note that this is of no physical interest, as it is a
        purely numerical artefact
        """
        return self.reduction.sum(
            np.where(self.interaction_force > 0., 1., 0.))

    def compute_nb_attractive_pts(self):
        """
        compute and return the number of contact points under attractive
        pressure. Note that this is of no physical interest, as it is a
        purely numerical artefact
        """
        return self.reduction.sum(
            np.where(self.interaction_force < 0., 1., 0.))

    def compute_repulsive_coordinates(self):
        """
        returns an array of all coordinates, where contact pressure is
        repulsive. Useful for evaluating the number of contact islands etc.
        """
        return np.argwhere(self.interaction_force > 0.)

    def compute_attractive_coordinates(self):
        """
        returns an array of all coordinates, where contact pressure is
        attractive. Useful for evaluating the number of contact islands etc.
        """
        return np.argwhere(self.interaction_force < 0.)

    def compute_mean_gap(self):
        """
        mean of the gap in the the physical domain (means excluding padding
        region for the FreeFFTElasticHalfspace)
        """
        return self.reduction.sum(self.gap) / np.prod(self.nb_grid_pts)

    def logger_input(self):
        """

        Returns
        -------
        headers: list of strings
        values: list
        """
        tot_nb_grid_pts = np.prod(self.nb_grid_pts)
        rel_rep_area = self.compute_nb_repulsive_pts() / tot_nb_grid_pts
        rel_att_area = self.compute_nb_attractive_pts() / tot_nb_grid_pts
        # TODO: eventually put a flag to turn
        #  reductions off since this is an additional communication.
        return (['energy',
                 'max. abs. grad.',
                 'mean gap',
                 'frac. rep. area',
                 'frac. att. area',
                 'frac. int. area', 'substrate force', 'interaction force'],
                [self.energy,
                 self.reduction.max(np.abs(self.force)),
                 self.compute_mean_gap(),
                 rel_rep_area,
                 rel_att_area,
                 rel_rep_area + rel_att_area,
                 -self.reduction.sum(self.substrate.force),
                 self.reduction.sum(self.interaction_force)])

    def evaluate(self, disp, offset, pot=True, forces=False, logger=None):
        """
        Compute the energies and forces in the system for a given displacement
        field

        Parameters:
        -----------
        disp: ndarray
            displacement field, in the shape of
            system.substrate.nb_subdomain_grid_pts
        offset: float
            determines indentation depth,
            constant value added to the heights (system.topography)
        pot: bool, optional
            Wether to evaluate the energy, default True
        forces: bool, optional
            Wether to evaluate the forces, default False
        logger: ContactMechanics.Tools.Logger
            informations of current state of the system will be passed to
            logger at every evaluation
        """
        # attention: the substrate may have a higher nb_grid_pts than the gap
        # and the interaction (e.g. FreeElasticHalfSpace)
        self.gap = self.compute_gap(disp, offset)
        interaction_energies, self.interaction_force, _ = \
            self.interaction.evaluate(self.gap,
                                      potential=pot,
                                      gradient=forces,
                                      curvature=False)

        self.interaction_energy = \
            self.reduction.sum(interaction_energies) * self.area_per_pt

        self.substrate.compute(disp, pot, forces)
        self.energy = (self.interaction_energy +
                       self.substrate.energy
                       if pot else None)
        if forces:
            self.interaction_force *= -self.area_per_pt
            #                       ^ gradient to force per pixel
            self.force = self.substrate.force.copy()
            self.force[self.comp_slice] += \
                self.interaction_force
        else:
            self.force = None

        if logger is not None:
            logger.st(*self.logger_input())
        return (self.energy, self.force)

    def objective(self, offset, disp0=None, gradient=False, logger=None):
        r"""
        This helper method exposes a scipy.optimize-friendly interface to the
        evaluate() method. Use this for optimization purposes, it makes sure
        that the shape of disp is maintained and lets you set the offset and
        'forces' flag without using scipy's cumbersome argument passing
        interface. Returns a function of only disp

        Parameters:
        -----------
        disp0: ndarray
            unused variable, present only for interface compatibility
            with inheriting classes
        offset: float
            determines indentation depth,
            constant value added to the heights (system.topography)
        gradient: bool, optional
            Wether to evaluate the gradient, default False
        logger: ContactMechanics.Tools.Logger
            informations of current state of the system will be passed to
            logger at every evaluation

        Returns:
            function(disp)
                Parameters:
                disp: an ndarray of shape
                      `system.substrate.nb_subdomain_grid_pts`
                      displacements
                Returns:
                    energy or energy, gradient
        """
        res = self.substrate.nb_subdomain_grid_pts
        if gradient:
            def fun(disp):
                # pylint: disable=missing-docstring
                try:
                    self.evaluate(
                        disp.reshape(res), offset, forces=True,
                        logger=logger)
                except ValueError as err:
                    raise ValueError(
                        "{}: disp.shape: {}, res: {}".format(
                            err, disp.shape, res))
                return (self.energy, -self.force.reshape(-1))
        else:
            def fun(disp):
                # pylint: disable=missing-docstring
                return self.evaluate(
                    disp.reshape(res), offset, forces=False,
                    logger=logger)[0]

        return fun

    def primal_evaluate(self, disp, gap, pot=True, forces=False, logger=None):
        """
        Compute the energies and forces in the system for a given
        displacement and gap..

        Parameters:
        -----------
        disp: ndarray
            displacement field, in the shape of
            system.substrate.nb_subdomain_grid_pts
        gap: ndarray
            gap , in the shape of
            system.substrate.nb_subdomain_grid_pts
        pot: bool, optional
            Whether to evaluate the energy, default True
        forces: bool, optional
            Whether to evaluate the forces, default False
        logger: ContactMechanics.Tools.Logger
            information of current state of the system will be
            passed to
            logger at every evaluation
        """
        # attention: the substrate may have a higher nb_grid_pts than the gap
        # and the interaction (e.g. FreeElasticHalfSpace)

        self.gap = gap
        interaction_energies, self.interaction_force, _ = \
            self.interaction.evaluate(self.gap,
                                      potential=pot,
                                      gradient=forces,
                                      curvature=False)

        self.interaction_energy = \
            self.reduction.sum(interaction_energies) * self.area_per_pt

        self.substrate.compute(disp, pot, forces)
        self.energy = (self.interaction_energy +
                       self.substrate.energy
                       if pot else None)
        if forces:
            self.interaction_force *= -self.area_per_pt
            #                       ^ gradient to force per pixel
            self.force = self.substrate.force.copy()

            self.force[self.comp_slice] += self.interaction_force
        else:
            self.force = None

        if logger is not None:
            logger.st(*self.logger_input())

        return (self.energy, self.force)

    def primal_objective(self, offset, disp0=None, gradient=False,
                         logger=None):
        r"""To solve the primal objective using gap as the variable.
        Can be fed directly to standard solvers ex: scipy solvers etc
        and returns the elastic energy and it's gradient (negative of
        the forces) as a function of the gap.

        Parameters
        __________

        gap : float
              gap between the contact surfaces.
        offset : float
                constant value to add to the surface heights
        pot : (default False)

        gradient : (default True)

        Returns
        _______
        energy : float
                value of energy(scalar value).
        force : float,array
                value of force(array).

        Notes
        _____

        Objective:

        .. math ::

            \min_u f = \frac{1}{2} u_i K_{ij} u_j + \phi (u_{ij})\\
            \\
            \nabla f = K_{ij} u_j + \phi^{\prime} \text{    which is the force} \\

        """

        res = self.substrate.nb_subdomain_grid_pts
        if gradient:
            def fun(gap):
                disp = gap.reshape(res) + self.surface.heights() + offset
                try:
                    self.primal_evaluate(
                        disp, gap.reshape(res), forces=True, logger=logger)
                except ValueError as err:
                    raise ValueError(
                        "{}: gap.shape: {}, res: {}".format(
                            err, gap.shape, res))
                return (self.energy, -self.force.reshape(-1))
        else:
            def fun(gap):
                disp = gap.reshape(res) + self.surface.heights() + offset
                return self.primal_evaluate(
                    disp, gap.reshape(res), forces=False, logger=logger)[0]

        return fun

    def primal_hessian_product(self, gap, des_dir):
        """Returns the hessian product of the primal_objective function.
        """
        _, _, adh_curv = self.interaction.evaluate(gap, curvature=True)

        hessp_val = - self.substrate.evaluate_force(
            des_dir.reshape(self.substrate.nb_subdomain_grid_pts)
        ).reshape(np.shape(des_dir)) \
            + adh_curv * des_dir * self.substrate.area_per_pt

        return hessp_val.reshape(des_dir.shape)

    def hessian_product_function(self, offset):
        def hessp(disp, des_dir):
            gap = disp.reshape(self.substrate.nb_subdomain_grid_pts
                               )[self.comp_slice] \
                  - (self.surface.heights() + offset)
            _, _, adh_curv = self.interaction.evaluate(gap, curvature=True)
            hessp_val = - self.substrate.evaluate_force(
                des_dir.reshape(self.substrate.nb_subdomain_grid_pts)
            )

            hessp_val[self.comp_slice] += adh_curv \
                * des_dir.reshape(
                self.substrate.nb_subdomain_grid_pts)[self.comp_slice] * \
                self.substrate.area_per_pt
            return hessp_val.reshape(des_dir.shape)

        return hessp

    def _fourier_coefficients(self):
        """
        Returns the coefficients for elasticity matrix when working in fourier
        space for both 1D and 2D system.
        """

        nx = self.substrate.nb_grid_pts[0]
        nb_dims = len(self.substrate.nb_grid_pts)

        if nb_dims == 2:
            ny = self.substrate.nb_grid_pts[1]
            coeffs = np.zeros(self.substrate.nb_grid_pts)
            if np.logical_and((nx % 2 == 0), (ny % 2 == 0)):
                coeffs[0, 0] = 1 / (nx * ny)
                coeffs[0, 1:ny // 2] = 2 / (nx * ny)
                coeffs[0, ny // 2 + 1:] = 2 / (nx * ny)
                coeffs[1:nx // 2, 0] = 2 / (nx * ny)
                coeffs[nx // 2 + 1:, 0] = 2 / (nx * ny)
                coeffs[:nx // 2, ny // 2] = 2 / (nx * ny)
                coeffs[nx // 2 + 1:, ny // 2] = 2 / (nx * ny)
                coeffs[nx // 2, :ny // 2] = 2 / (nx * ny)
                coeffs[nx // 2, ny // 2 + 1:] = 2 / (nx * ny)
                coeffs[1:nx // 2, 1:ny // 2] = 4 / (nx * ny)
                coeffs[nx // 2 + 1:, 1:ny // 2] = 4 / (nx * ny)
                coeffs[1:nx // 2, ny // 2 + 1:] = 4 / (nx * ny)
                coeffs[nx // 2 + 1:, ny // 2 + 1:] = 4 / (nx * ny)
                coeffs[nx // 2, ny // 2] = 1 / (nx * ny)
                coeffs[nx // 2, 0] = 1 / (nx * ny)
                coeffs[0, ny // 2] = 1 / (nx * ny)
            else:
                coeffs[0, 0] = 1 / (nx * ny)
                coeffs[0, 1:] = 2 / (nx * ny)
                coeffs[1:, 0] = 2 / (nx * ny)
                coeffs[1:, 1:] = 4 / (nx * ny)
        elif nb_dims == 1:
            coeffs = np.zeros(self.substrate.nb_grid_pts)
            if (nx % 2 == 0):
                coeffs[0] = 1 / nx
                coeffs[1:nx // 2] = 2 / nx
                coeffs[nx // 2 + 1:] = 2 / nx
                coeffs[nx // 2] = 1 / nx
            else:
                coeffs[0] = 1 / nx
                coeffs[1:] = 2 / nx

        return coeffs

    def evaluate_k(self, disp_k, gap, offset, mw=False, pot=True, forces=False,
                   logger=None):

        """
        Compute the energies and forces in the system for a given displacement
        field in fourier space.

        Parameters
        -----------

        disp_k: ndarray
            displacement field in fourier space.
        gap:  ndarray
            displacement field in real space, in the shape of
            system.substrate.nb_subdomain_grid_pts
        offset: float
            determines indentation depth,
            constant value added to the heights (system.topography)
        mw: bool, optional
            when mass weighting is required then set this to TRUE.
        pot: bool, optional
            Wether to evaluate the energy, default True
        forces: bool, optional
            Wether to evaluate the forces, default False
        logger: ContactMechanics.Tools.Logger
            informations of current state of the system will be passed to
            logger at every evaluation.
        """

        # self.gap = self.compute_gap(disp, offset)
        self.gap = gap
        interaction_energies, self.interaction_force, _ = \
            self.interaction.evaluate(self.gap,
                                      potential=pot,
                                      gradient=forces,
                                      curvature=False)
        self.interaction_energy = \
            self.reduction.sum(interaction_energies) * self.area_per_pt

        self.grad_k = np.zeros(self.substrate.nb_grid_pts)

        coeff = self._fourier_coefficients()

        if mw:
            self.grad_k = disp_k * coeff
        else:
            self.grad_k = disp_k * coeff * self.stiffness_k

        self.grad_k *= self.area_per_pt

        # ENERGY FROM SUBSTRATE
        self.energy = 0.5 * (np.sum(self.grad_k * disp_k))

        self.substrate.energy = self.energy

        self.force_h = -self.grad_k

        # TOTAL ENERGY
        self.energy += self.interaction_energy

        if forces:
            self.interaction_force *= -self.area_per_pt
            #                     ^ gradient to force per pixel

            self.real_buffer.array()[...] = self.interaction_force
            self.engine.hcfft(self.real_buffer, self.fourier_buffer)
            interaction_force_float_k = self.fourier_buffer.array()[...].copy()

            adh_coeffs = self._fourier_coefficients()
            interaction_force_float_k *= adh_coeffs

            if mw:
                k = np.sqrt(self.stiffness_k.copy() * self.area_per_pt)
                interaction_force_float_k = interaction_force_float_k * (1 / k)

            self.force_h += interaction_force_float_k
        else:
            self.force_h = None

        if logger is not None:
            disp_real = self.gap + self.surface.heights().copy() + offset
            force_real = self.substrate.evaluate_force(disp_real)
            force_real = force_real + self.interaction_force
            logger.st(*(['energy',
                         'max. abs. grad.',
                         'max. abs. grad. real'],
                        [self.energy,
                         self.reduction.max(np.abs(self.force_h)),
                         self.reduction.max(np.abs(force_real))
                         ]))

        return (self.energy, self.force_h)

    # def hessian_product_k(self, dispk, des_dir_k):
    #     """Returns the hessian product of the fourier space
    #     objective_k function.
    #     """
    #     self.substrate.fourier_buffer.array()[...] = dispk.copy()
    #     self.substrate.fftengine.ifft(self.substrate.fourier_buffer,
    #                                   self.substrate.real_buffer)
    #     disp = self.substrate.real_buffer.array()[...].copy() \
    #         * self.substrate.fftengine.normalisation
    #
    #     gap = self.compute_gap(disp)
    #     _, _, adh_curv = self.interaction.evaluate(gap, curvature=True)
    #
    #     self.substrate.real_buffer.array()[...] = adh_curv.reshape(
    #         self.substrate.nb_grid_pts).copy()
    #     self.substrate.fftengine.fft(self.substrate.real_buffer,
    #                                  self.substrate.fourier_buffer)
    #     adh_curv_k = self.substrate.fourier_buffer.array()[...].copy()
    #
    #     hessp_val_k = -self.substrate.evaluate_k_force_k(des_dir_k) + \
    #         adh_curv_k * des_dir_k * self.substrate.area_per_pt
    #
    #     return hessp_val_k

    # def hessian_product_preconditioned(self, offset):

    #     def hessp_precond(disp_h, des_dir):
    #
    #         self.real_buffer.array()[...] = offset
    #         self.engine.hcfft(self.real_buffer, self.fourier_buffer)
    #         offset_k = self.fourier_buffer.array()[...].copy()
    #
    #         self.real_buffer.array()[...] = self.surface.heights().copy()
    #         self.engine.hcfft(self.real_buffer, self.fourier_buffer)
    #         self.heights_k_float = self.fourier_buffer.array()[...].copy()
    #
    #         disp_float_k = disp_h.copy()
    #         disp_float_k = disp_float_k.reshape(self.substrate.nb_grid_pts)
    #         gap_float_k = (disp_float_k / np.sqrt(self.stiffness_k *
    #                                               self.area_per_pt)) - \
    #                       self.heights_k_float - offset_k
    #
    #         self.fourier_buffer.array()[...] = gap_float_k.copy()
    #         self.engine.ihcfft(self.fourier_buffer, self.real_buffer)
    #         gap = self.real_buffer.array()[...].copy() * \
    #               self.engine.normalisation
    #
    #         _, _, adh_curv = \
    #             self.interaction.evaluate(gap,
    #                                       curvature=True)
    #
    #         coeff = self._fourier_coefficients()
    #
    #         el_hess_k = coeff @ coeff
    #         el_hess_k *= self.area_per_pt
    #
    #         orig_shape = np.shape(des_dir)
    #
    #         el_hess_k *= des_dir #.reshape(el_hess_k)
    #
    #         adh_curv *= self.area_per_pt
    #
    #         self.real_buffer.array()[...] = adh_curv
    #         self.engine.hcfft(self.real_buffer, self.fourier_buffer)
    #         adh_curv_float_k = self.fourier_buffer.array()[...].copy()
    #
    #         adh_curv_float_k *= coeff @ coeff
    #
    #         k = np.sqrt(self.stiffness_k.copy() * self.area_per_pt)
    #
    #         adh_hess_k = adh_curv_float_k * des_dir * (1 / k) * (1 / k)
    #
    #         hess_k = el_hess_k.reshape(orig_shape) + \
    #                  adh_hess_k.reshape(orig_shape)
    #
    #         return hess_k
    #
    #     return hessp_precond

    def preconditioned_objective(self, offset, gradient=False, logger=None):
        r"""
        This helper method interface to the evaluate_k() method with
        preconditioning active. That is, it tries to solve a simpler problem
        formulated using,

        original problem:

        .. math ::

            \frac{1}{2(n_x n_y)} \tilde{u}\tilde{K} \bar{\tilde{u}} +
            \phi(F^{-1}(\tilde{u} - \tilde{h}))

        preconditioned problem:

        .. math ::
             \tilde{v} = \tilde{k}^{\frac{1}{2}} \tilde{u} \\

             \frac{1}{2(n_x n_y)} \tilde{v}\bar{\tilde{v}} +
            \phi(F^{-1}(\frac{\tilde{v}}{\tilde{k}^{\frac{1}{2}}} - \tilde{h}))

        we solve for variable :math:`\tilde{v}`.

        Parameters:
        -----------

        offset: float
            determines indentation depth,
            constant value added to the heights (system.topography)
        gradient: bool, optional
            Whether to evaluate the gradient, default False
        logger: ContactMechanics.Tools.Logger
            information of current state of the system will be passed to
            logger at every evaluation

        Returns
        _______

            function(disp_k)

                Parameters
                __________

                disp_k: an ndarray in fourier halfcomplex space

                Returns
                _______

                energy: scalar
                        energy of the system

                force_h: an halfcomplex array of shape(disp_k)
                        force of the system
        """

        # TODO: fourier transforming the offset is useless
        self.real_buffer.array()[...] = offset
        self.engine.hcfft(self.real_buffer, self.fourier_buffer)
        offset_k = self.fourier_buffer.array()[...].copy()

        self.real_buffer.array()[...] = self.surface.heights().copy()
        self.engine.hcfft(self.real_buffer, self.fourier_buffer)
        self.heights_k_float = self.fourier_buffer.array()[...].copy()

        if gradient:
            def fun(disp_):
                disp_float_k = disp_.copy()
                orig_shape = np.shape(disp_float_k)
                disp_float_k = disp_float_k.reshape(self.substrate.nb_grid_pts)
                gap_float_k = (disp_float_k / np.sqrt(self.stiffness_k *
                                                      self.area_per_pt)) - \
                    self.heights_k_float - offset_k

                self.fourier_buffer.array()[...] = gap_float_k.copy()
                self.engine.ihcfft(self.fourier_buffer, self.real_buffer)
                gap = self.real_buffer.array()[...].copy() * self.engine.normalisation

                self.energy, self.force_h = self.evaluate_k(disp_float_k,
                                                            gap, offset,
                                                            mw=True,
                                                            forces=True,
                                                            logger=logger
                                                            )

                return (self.energy, -self.force_h.reshape(orig_shape))
        else:
            raise NotImplementedError

        return fun

    def objective_k_float(self, offset, gradient=False, logger=None):
        r"""

        Returns callable objective as needed by scipy minimizers.

        The optimisation varialbe is the halfcomplex fourier transform of the gap.

        This helper method interface to the evaluate_k() method without
        preconditioning active.

        .. math ::

            \frac{1}{2(n_x n_y)} \tilde{u}\tilde{K} \bar{\tilde{u}} +
            \phi(F^{-1}(\tilde{u} - \tilde{h}))  \\

        preconditioned problem:

        .. math ::

             \tilde{v} = \tilde{k}^{\frac{1}{2}} \tilde{u} \\

             \frac{1}{2(n_x n_y)} \tilde{v} \bar{\tilde{v}} +
            \phi(F^{-1}(\frac{\tilde{v}}{\tilde{k}^{\frac{1}{2}}} - \tilde{h}))
        \\
        we solve for variable :math:`\tilde{v}`.

        Parameters:
        -----------

        offset: float
            determines indentation depth,
            constant value added to the heights (system.topography)
        gradient: bool, optional
            Whether to evaluate the gradient, default False
        logger: ContactMechanics.Tools.Logger
            information of current state of the system will be passed to
            logger at every evaluation

        Returns
        _______

            function(disp_k)

                Parameters
                __________

                disp_k: an ndarray in fourier space

                Returns
                _______

                    energy, gradient_k_float
        """

        self.real_buffer.array()[...] = offset
        self.engine.hcfft(self.real_buffer, self.fourier_buffer)
        offset_k = self.fourier_buffer.array()[...].copy()

        self.real_buffer.array()[...] = self.surface.heights().copy()
        self.engine.hcfft(self.real_buffer, self.fourier_buffer)
        self.heights_k_float = self.fourier_buffer.array()[...].copy()

        if gradient:
            def fun(disp_k):
                disp_float_k = disp_k.copy()
                orig_shape = np.shape(disp_float_k)
                disp_float_k = disp_float_k.reshape(self.substrate.nb_grid_pts)
                gap_float_k = disp_float_k - self.heights_k_float - offset_k
                self.fourier_buffer.array()[...] = gap_float_k.copy()
                self.engine.ihcfft(self.fourier_buffer, self.real_buffer)
                gap = self.real_buffer.array()[...].copy() \
                    * self.engine.normalisation

                self.energy, self.force_h = self.evaluate_k(disp_float_k,
                                                            gap, offset,
                                                            forces=True,
                                                            logger=logger
                                                            )
                return (self.energy, -self.force_h.reshape(orig_shape))
        else:
            def fun(disp_k):
                # pylint: disable=missing-docstring
                disp_float_k = disp_k.copy()
                disp_float_k = disp_float_k.reshape(self.substrate.nb_grid_pts)
                gap_float_k = disp_float_k - self.heights_k_float - offset_k
                self.fourier_buffer.array()[...] = gap_float_k.copy()
                self.engine.ihcfft(self.fourier_buffer, self.real_buffer)
                gap = self.real_buffer.array()[...].copy() \
                    * self.engine.normalisation

                return self.evaluate_k(disp_float_k, gap, offset, forces=True,
                                       logger=logger)[0]

        return fun

    def callback(self, force=False):
        """
        Simple callback function that can be handed over to scipy's minimize to
        get updates during minimisation
        Parameters:
        ----------
        force: bool, optional
            whether to include the norm of the force
            vector in the update message
            (default False)
        """
        counter = 0
        if force:
            def fun(dummy):
                "includes the force norm in its output"
                nonlocal counter
                counter += 1
                print("at it {}, e = {}, |f| = {}".format(
                    counter, self.energy,
                    np.linalg.norm(np.ravel(self.force))))
        else:
            def fun(dummy):
                "prints messages without force information"
                nonlocal counter
                counter += 1
                print("at it {}, e = {}".format(
                    counter, self.energy))
        return fun


class BoundedSmoothContactSystem(SmoothContactSystem):
    @staticmethod
    def handles(*args, **kwargs):  # FIXME work around, see issue #208
        return False

    def compute_nb_contact_pts(self):
        """
        compute and return the number of contact points.
        """
        return self.reduction.sum(np.where(self.gap == 0., 1., 0.))

    def logger_input(self):
        """

        Returns
        -------
        headers: list of strings
        values: list
        """
        tot_nb_grid_pts = np.prod(self.nb_grid_pts)
        rel_rep_area = self.compute_nb_repulsive_pts() / tot_nb_grid_pts
        rel_att_area = self.compute_nb_attractive_pts() / tot_nb_grid_pts
        # TODO: eventually put a flag to turn
        #  reductions off since this is an additional communication.

        contacting_points = self.gap == 0.
        mask = np.ones(self.substrate.nb_subdomain_grid_pts)
        mask[self.substrate.local_topography_subdomain_slices][
            contacting_points] = 0
        max_proj_grad = self.reduction.max(abs(mask * self.force))

        return (['energy',
                 'max. proj. grad.',
                 'mean gap',
                 'frac. cont. area',
                 'frac. rep. area',
                 'frac. att. area',
                 'frac. int. area',
                 'substrate force',
                 'interaction force'],
                [self.energy,
                 max_proj_grad,
                 self.compute_mean_gap(),
                 self.compute_nb_contact_pts() / np.prod(self.nb_grid_pts),
                 rel_rep_area,
                 rel_att_area,
                 rel_rep_area + rel_att_area,
                 -self.reduction.sum(self.substrate.force),
                 self.reduction.sum(self.interaction_force)])

    def compute_normal_force(self):
        "computes and returns the sum of all forces"

        # sum of the jacobian in the contact area (Lagrange multiplier)
        # and the ineraction forces.
        # can also be computed easily from the substrate forces,
        # what we do here
        return self.reduction.sum(
            - self.substrate.force[self.substrate.local_topography_subdomain_slices])

    def compute_repulsive_force(self):
        """computes and returns the sum of all repulsive forces

        Assumptions: there
        """
        return self.reduction.sum(
            np.where(
                - self.substrate.force[
                    self.substrate.local_topography_subdomain_slices] > 0,
                - self.substrate.force[
                    self.substrate.local_topography_subdomain_slices], 0.))

    def compute_attractive_force(self):
        "computes and returns the sum of all attractive forces"
        return self.reduction.sum(
            np.where(
                - self.substrate.force[
                    self.substrate.local_topography_subdomain_slices] < 0,
                - self.substrate.force[
                    self.substrate.local_topography_subdomain_slices],
                0.))

    def compute_nb_repulsive_pts(self):
        """
        compute and return the number of contact points under repulsive
        pressure.

        """
        return self.reduction.sum(
            np.where(
                np.logical_and(
                    self.gap == 0.,
                    - self.substrate.force[
                        self.substrate.local_topography_subdomain_slices] > 0),
                1., 0.))

    def compute_nb_attractive_pts(self):
        """
        compute and return the number of contact points under attractive
        pressure.
        """

        # Compute points where substrate force is negative
        # or there is no contact
        pts = np.logical_or(- self.substrate.force[
            self.substrate.local_topography_subdomain_slices] < 0,
                            self.gap > 0.)

        # exclude points where there is no contact
        # and the interaction force is 0.
        pts[np.logical_and(self.gap > 0.,
                           self.interaction_force == 0.)] = 0.

        return self.reduction.sum(pts)

    def compute_repulsive_coordinates(self):
        """
        returns an array of all coordinates, where contact pressure is
        repulsive. Useful for evaluating the number of contact islands etc.
        """
        return np.argwhere(
            np.logical_and(
                self.gap == 0.,
                - self.substrate.force[
                    self.substrate.local_topography_subdomain_slices]
                > 0))

    def compute_attractive_coordinates(self):
        """
        returns an array of all coordinates, where contact pressure is
        attractive. Useful for evaluating the number of contact islands etc.
        """

        # Compute points where substrate force is negative
        # or there is no contact
        pts = np.logical_or(
            - self.substrate.force[self.substrate.local_topography_subdomain_slices]
            < 0,
            self.gap > 0.)

        # exclude points where there is no contact
        # and the interaction force is 0.
        pts[np.logical_and(self.gap > 0.,
                           self.interaction_force == 0.)] = 0.

        return np.argwhere(pts)
