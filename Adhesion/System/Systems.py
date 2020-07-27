#
# Copyright 2018, 2020 Antoine Sanner
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
Defines the interface for Adhesion systems
"""

import numpy as np

import Adhesion, ContactMechanics, SurfaceTopography
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
        self.interaction_energy = None
        self.interaction_force = None

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
        return self.pnp.sum(np.where(
            self.interaction_force > 0, self.interaction_force, 0
            ))

    def compute_attractive_force(self):
        "computes and returns the sum of all attractive forces"
        return self.pnp.sum(np.where(
            self.interaction_force < 0, self.interaction_force, 0
            ))

    def compute_normal_force(self):
        "computes and returns the sum of all forces"
        return self.pnp.sum(self.interaction_force)

    def compute_repulsive_contact_area(self):
        "computes and returns the area where contact pressure is repulsive"
        return self.compute_nb_repulsive_pts()*self.area_per_pt

    def compute_attractive_contact_area(self):
        "computes and returns the are where contact pressure is attractive"
        return self.compute_nb_attractive_pts()*self.area_per_pt

    def compute_nb_contact_pts(self):
        """
        compute and return the number of contact points. Note that this is of
        no physical interest, as it is a purely numerical artefact
        """
        return self.pnp.sum(np.where(self.interaction_force != 0., 1., 0.))

    def compute_nb_repulsive_pts(self):
        """
        compute and return the number of contact points under repulsive
        pressure. Note that this is of no physical interest, as it is a
        purely numerical artefact
        """
        return self.pnp.sum(np.where(self.interaction_force > 0., 1., 0.))

    def compute_nb_attractive_pts(self):
        """
        compute and return the number of contact points under attractive
        pressure. Note that this is of no physical interest, as it is a
        purely numerical artefact
        """
        return self.pnp.sum(np.where(self.interaction_force < 0., 1., 0.))

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
        return self.pnp.sum(self.gap) / np.prod(self.nb_grid_pts)

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

        return (['energy', 'mean gap', 'frac. rep. area',
                       'frac. att. area',
                       'frac. int. area', 'substrate force', 'interaction force'],
              [self.energy,
               self.compute_mean_gap(),
               rel_rep_area,
               rel_att_area,
               rel_rep_area + rel_att_area,
               -self.pnp.sum(self.substrate.force),
               self.pnp.sum(self.interaction_force)])

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
        interaction_energies, self.interaction_force, _ =  \
            self.interaction.evaluate(self.gap,
                                      potential=pot,
                                      gradient=forces,
                                      curvature=False)


        self.interaction_energy = \
            self.pnp.sum(interaction_energies) * self.area_per_pt

        self.substrate.compute(disp, pot, forces)
        self.energy = (self.interaction_energy+
                       self.substrate.energy
                       if pot else None)
        if forces:
            self.interaction_force *= -self.area_per_pt
            #                       ^ gradient to force per pixel
            self.force = self.substrate.force.copy()
            if self.dim == 1: # TODO: remove this if
                self.force[self.comp_slice] += \
                  self.interaction_force
            else:
                self.force[self.comp_slice] += \
                  self.interaction_force
        else:
            self.force = None
        if logger is not None:
            logger.st(*self.logger_input())
        return (self.energy, self.force)

    def objective(self, offset, disp0=None, gradient=False, disp_scale=1.,
                  logger=None):
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
        disp_scale: float
            (default 1.) allows to specify a scaling of the
            dislacement before evaluation.
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
        dummy = disp0
        res = self.substrate.nb_subdomain_grid_pts
        if gradient:
            def fun(disp):
                # pylint: disable=missing-docstring
                try:
                    self.evaluate(
                        disp_scale * disp.reshape(res), offset, forces=True,
                        logger=logger)
                except ValueError as err:
                    raise ValueError(
                        "{}: disp.shape: {}, res: {}".format(
                            err, disp.shape, res))
                return (self.energy, -self.force.reshape(-1)*disp_scale)
        else:
            def fun(disp):
                # pylint: disable=missing-docstring
                return self.evaluate(
                    disp_scale * disp.reshape(res), offset, forces=False,
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
    def handles(*args, **kwargs): # FIXME work around, see issue #208
        return False

    def compute_nb_contact_pts(self):
        """
        compute and return the number of contact points.
        """
        return self.pnp.sum(np.where(self.gap == 0., 1., 0.))

    def logger_input(self):
        headers, vals = super().logger_input()
        headers.append("frac. cont. area")
        vals.append(self.compute_nb_contact_pts() / np.prod(self.nb_grid_pts))
        return headers, vals

    def compute_normal_force(self):
        "computes and returns the sum of all forces"

        # sum of the jacobian in the contact area (Lagrange multiplier)
        # and the ineraction forces.
        # can also be computed easily from the substrate forces, what we do here
        return self.pnp.sum( - self.substrate.force[self.substrate.topography_subdomain_slices])

    def compute_repulsive_force(self):
        """computes and returns the sum of all repulsive forces

        Assumptions: there
        """
        return self.pnp.sum(np.where( - self.substrate.force[self.substrate.topography_subdomain_slices] > 0,
                 - self.substrate.force[self.substrate.topography_subdomain_slices], 0.))

    def compute_attractive_force(self):
        "computes and returns the sum of all attractive forces"
        return self.pnp.sum(np.where( - self.substrate.force[self.substrate.topography_subdomain_slices] < 0,
                - self.substrate.force[self.substrate.topography_subdomain_slices], 0.))

    def compute_nb_repulsive_pts(self):
        """
        compute and return the number of contact points under repulsive
        pressure.

        """
        return self.pnp.sum(np.where(np.logical_and(self.gap == 0., - self.substrate.force[self.substrate.topography_subdomain_slices] > 0), 1., 0.))

    def compute_nb_attractive_pts(self):
        """
        compute and return the number of contact points under attractive
        pressure.
        """

        # Compute points where substrate force is negative or there is no contact
        pts = np.logical_or( - self.substrate.force[self.substrate.topography_subdomain_slices] < 0,
                                            self.gap > 0.)

        # exclude points where there is no contact and the interaction force is 0.
        pts[np.logical_and(self.gap > 0.,
            self.interaction_force == 0.)] = 0.

        return self.pnp.sum(pts)

    def compute_repulsive_coordinates(self):
        """
        returns an array of all coordinates, where contact pressure is
        repulsive. Useful for evaluating the number of contact islands etc.
        """
        return np.argwhere(np.logical_and(self.gap == 0., - self.substrate.force[self.substrate.topography_subdomain_slices] > 0))

    def compute_attractive_coordinates(self):
        """
        returns an array of all coordinates, where contact pressure is
        attractive. Useful for evaluating the number of contact islands etc.
        """

        # Compute points where substrate force is negative or there is no contact
        pts = np.logical_or( - self.substrate.force[self.substrate.topography_subdomain_slices] < 0,
                                            self.gap > 0.)

        # exclude points where there is no contact and the interaction force is 0.
        pts[np.logical_and(self.gap > 0.,
            self.interaction_force == 0.)] = 0.

        return np.argwhere(pts)
