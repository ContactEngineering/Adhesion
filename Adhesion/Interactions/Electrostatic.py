from Adhesion.Interactions import Potential
from NuMPI import MPI
import numpy as np

class ChargePatternsInteraction(Potential):
    """
    Potential for the interaction of charges

    Please cite:
    Persson, B. N. J. et al. EPL 103, 36003 (2013)
    """

    def __init__(self,
                 charge_distribution,
                 physical_sizes,
                 dielectric_constant_material=1.,
                 dielectric_constant_gap=1.,
                 pixel_sizes=(1e-9,1e-9),
                 communicator=MPI.COMM_WORLD):
        """

        Parameters
        ----------
        charge_distribution: float ndarray
            spatial distribution
        physical_sizes: tuple
            length and width of the plate
        dielectric_constant_material: float
        dielectric_constant_gap: float
        pixel_sizes: tuple float

        Returns
        -------

        """
        assert np.ndim(charge_distribution) == 2
        self.charge_distribution = charge_distribution
        self.physical_sizes = physical_sizes
        epsilon0 = 8.854e-12  # [C/m^2], permittivity in vacuum
        self.epsilon_material = dielectric_constant_material * epsilon0
        self.epsilon_gap = dielectric_constant_gap * epsilon0
        self.pixel_area = np.prod(pixel_sizes)
        Potential.__init__(self, communicator=communicator)
        self.num_grid_points = charge_distribution.shape
    
    @property
    def r_min(self):
        return None
    
    @property
    def r_infl(self):
        return None
    
    def __repr__(self, ):
        return ("Potential '{0.name}': ε = {0.eps}, σ = {0.sig}").format(self)

    def evaluate(self,
                 gap, 
                 potential=True,
                 gradient=False,
                 curvature=False):
        """

        Parameters
        ----------
        gap: array_like

        Returns
        -------

        potential: float ndarray

        gradient: ndarray
            first derivative of the potential wrt. gap  (= - forces by pixel)
        curvature: ndarray or linear operator (callable)
            second derivative of the potential
            # TODO: is that easy/possible/computationally tractable ?
        """
        assert np.ndim(gap) == 1  # unit[m]
        
        # fast Fourier transform, σ(x1, x2) -> σ(q1, q2)
        fft_charge_distribution = np.fft.fft2(self.charge_distribution, 
            s=self.num_grid_points)
        fft_magnitude = np.abs(fft_charge_distribution) * self.pixel_area
            # According to sampling theory, the magnitude after discretization 
            # is not exactly the same as in the continuous cases. One must
            # multiply the result by a sampling resolution to keep equivalent.
        
        # q1 as first axis, q2 as second axis, z as third axis
        # reshape to become three independent axes for easier handling
        size1, size2 = self.num_grid_points
        q1_axis = np.fft.fftfreq(size1, 
            d=self.physical_sizes[0]/(2*np.pi*size1)).reshape(-1,1,1)
        q2_axis = np.fft.fftfreq(size2, 
            d=self.physical_sizes[1]/(2*np.pi*size2)).reshape(1,-1,1)
        gap_axis = np.reshape(gap, (1,1,-1))
        
        # dq1, dq2 are assumed to be discretized as constants
        #          2 π
        # d_qi = ───────
        #          L_i
        d_q1 = 2 * np.pi / self.physical_sizes[0]
        d_q2 = 2 * np.pi / self.physical_sizes[1]
                        
        #                          ϵ_m
        # coefficient A = ─────────────────────
        #                  2 π^2 (ϵ_m + ϵ_g)^2
        A = self.epsilon_material
        A /= 2 * np.pi**2 * (self.epsilon_material + self.epsilon_gap)**2
        
        #                 A
        # integral = - ─────── ∫∫ dq1 dq2  σ(q1, q2)^2  K(q1, q2, z)
        #               L1 L2
        # K is the kernel function, will be different in different cases
        integral = lambda K: -A / np.prod(self.physical_sizes) * np.einsum(
            "...,...,pq,pqz->z", d_q1, d_q2, fft_magnitude**2, K)
            # ... is used to handle zero dimension arrays
        
        # q_norm = |q| = √(q1^2 + q2^2)
        q_norm = np.sqrt(q1_axis**2 + q2_axis**2)
        
        # decay = exp(-|q|z)
        decay = np.exp(-q_norm * gap_axis)
                
        #                  ϵ_m - ϵ_g
        # coefficient B = ───────────
        #                  ϵ_m + ϵ_g
        B = self.epsilon_material - self.epsilon_gap
        B /= self.epsilon_material + self.epsilon_gap
        
        if potential:  
            #                     -exp(-|q|z)
            # K(q1, q2, z) = ──────────────────────
            #                |q| [1 - B exp(-|q|z)]
            kernel = -decay / q_norm / (1 - B*decay)
            potential = integral(kernel)
        else:
            potential = None
        
        if gradient:
            #                     exp(-|q|z)
            # K(q1, q2, z) = ─────────────────────
            #                [1 - B exp(-|q|z)]^2
            kernel = decay / (1 - B*decay)**2
            gradient = integral(kernel)
        else:
            gradient = None
        
        if curvature:
            #                 -|q| exp(-|q|z) [1 + B exp(-|q|z)]
            # K(q1, q2, z) = ────────────────────────────────────
            #                        [1 - B exp(-|q|z)]^3
            kernel = -q_norm * decay * (1 + B*decay) / (1 - B*decay)**3
            curvature = integral(kernel)
        else:
            curvature =  None
        
        return (potential, gradient, curvature)

 
        
        