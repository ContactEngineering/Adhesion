# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example of rough simulation with adhesion on the contact challenge 512x512 topography
#
# The adhesion is longer ranged in this example than in the original contact challenge problem definitition because we use a much coarser grid here. 

# %%
from SurfaceTopography import read_published_container
import Adhesion # required to register the "make_system" function
from Adhesion.Interactions import Exponential
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.colors import LinearSegmentedColormap

# %%
c, = read_published_container("https://doi.org/10.57703/ce-rs7jq")

# %%
t = c._topographies[1].to_unit("m")

# %% tags=[]
t.is_periodic

# %%
t.rms_gradient()

# %%
t.rms_height_from_area()

# %%
1/t.rms_curvature_from_area()

# %%
Es = 25e6 # N / m2
challenge_wint = 0.05 # mJ / m2
challenge_rho = 2.071e-9 # interaction_range in the contact challenge
# External load 250 kPa
challenge_external_pressure = 250e3 # Pa

# %% [markdown]
# ## Defining meaningful numerical tolerances

# %% [markdown]
# In adhesive simulations, the optimisation variable is the displacement. 
# The algorithm seeks the minimum of the total energy $\Pi_\mathrm{tot}$, where the residual $\partial \Pi_\mathrm{tot}/\partial u_{ij} = 0$

# %% [markdown]
# A typical repulsive contact pressure is $E^\prime h^\prime_{rms}$, leading to the typical repulsive pixel force $E^\prime h^\prime_{rms} \Delta x^2$. 
# We use a small fraction of this typical force as the numerical tolerance for the residual: 

# %%
pixel_force_tol = 1e-5 * Es * t.rms_gradient() * t.area_per_pt

# %% [markdown]
# Another valid choice of "typical pressure" would be to use the maximum interaction force. 

# %% [markdown]
# In dual minimizers (used in the nonadhesive contact system), the residual is the gap inside the contact area. Let's use a typical change of heights for the tolerance

# %%
penetration_tol = 1e-3 * t.rms_gradient() * t.pixel_size[0]

# %% [markdown]
# ### Nonadhesive contact system

# %%
nonadhesive_system = t.make_contact_system(interaction="hardwall", young=Es)

# %%
sol = nonadhesive_system.minimize_proxy(external_force=np.prod(t.physical_sizes) 
                                        * challenge_external_pressure, pentol=penetration_tol)

# %%
nonadhesive_offset = sol.offset

# %%
fig,ax=plt.subplots()
cmap = LinearSegmentedColormap.from_list('contactcmap', ((1,1,1,0.), "black"), N=256)
ax.matshow(nonadhesive_system.contact_zone.T, cmap = cmap)
ax.grid(False)

# %% [markdown]
# ### contact system with hardwall repulsion

# %%
from Adhesion.Interactions import Exponential
from Adhesion.System import BoundedSmoothContactSystem

# %%
interaction=Exponential(gamma0=challenge_wint ,rho=challenge_rho*30)

# %%
w = -interaction.evaluate(0)[0]

# %%
dx = t.pixel_size[0]

# %% [markdown]
# ### Properties of the adhesive rough contact

# %% [markdown]
# ##### Energy for full conformal contact

# %%
print("Persson and Tosatti adhesion criterion: ")
print(
    "elastic deformation energy for full contact / work of adhesion: "
    "{}".format(nonadhesive_system.substrate.evaluate(t.detrend(detrend_mode="center").heights())[0]
                / w / np.prod(t.physical_sizes)))  # ^
#                           sets the mean to zero, so         |
#                           that the energy of the q = 0   --
#                           mode is not taken into
#                           account

# %% [markdown]
# ##### Pastewka and Robbins criterion
# P&R equation 10, with $\kappa_{rep} = 2$, $d_{rep}=4 h^{\prime}_{rms} / h^{\prime\prime}_{rms}$
# (Only valid in the DMT regime, i.e. until the onset of stickiness)
# If this value is smaller then 1 the surfaces are sticky and
# normal CG's might not work anymore
#
# A requirement for the Pastewka Robbins theory to be valid is that the interaction is short ranged enough. 
# See below. 

# %%
rho = interaction.rho


print("P&R stickiness criterion: "
      "{}".format(t.rms_gradient() * Es * rho / 2 / w *
                  (t.rms_gradient() ** 2 / t.rms_curvature_from_area() / rho)
                  ** (2 / 3)))

R = 1 / t.rms_curvature_from_area()
elastocapillary_length = w / Es
print("Generalized Tabor parameter following Martin Müser")
print("{}".format(
    R ** (1 / 3) * elastocapillary_length ** (2 / 3) / rho))

# %% [markdown]
# Is the Pastewka Robbins assumption of short-ranged adhesion valid ? 

# %%
print("rms_height / interaction_length: "
      "{}".format(t.rms_height_from_area() / interaction.rho))

# %% [markdown]
# [Monti Sanner and Pastewka 2021](https://doi.org/10.1007/s11249-021-01454-6)

# %%
g0 = 16 /3 * t.rms_gradient()**2 / t.rms_curvature_from_area()

# %%
g0

# %%
print("rho / g0: "
      "{}".format(g0 / interaction.rho))

# %% [markdown]
# This value should be smaller than 0.1 for the assumptions behind the Pastewka and Robbins theory to be valid. We hence cannot apply the theory to the system considered here.

# %% [markdown]
# #### Estimate whether the discretisation is alright

# %% [markdown]
# ##### Process zone in a crack

# %%
process_zone_size = Es * w / np.pi / abs(interaction.max_tensile) ** 2

print("nb pixels in process zone: {}".format(process_zone_size / dx))
# print("highcut wavelength / process_zone size: "
#       "{}".format(short_cutoff / process_zone_size))


# %% [markdown]
# ##### Empirical criterion on contact of spheres (see [adhesive_simulation_sphere.ipynb]())
# $$
# dx \leq 2 \rho^2 \frac{4}{3\pi} \frac{1}{\ell_a}
# $$

# %% [markdown]
# The value below should be greater than 1: 

# %%
interaction.rho ** 2 * 8 / 3 / np.pi * Es / w / dx

# %% [markdown]
# ##### [Wang, Zhou and Müser 2021](https://doi.org/10.3390/lubricants9020017) Equation (9) and Fig.8:

# %% [markdown]
# $\mu_{rho}$

# %%
np.sqrt(- interaction.evaluate(0, True, True, True)[-1] # min curvature for the exponential interaction
                 / Es * dx
                )

# %% [markdown]
# should be smaller than 0.5

# %% [markdown]
# ### Adhesive simulation

# %%
system = t.make_contact_system(interaction=interaction, young=Es, system_class=BoundedSmoothContactSystem)

# %%
sol=system.minimize_proxy(
               offset=nonadhesive_offset, 
               # tol=pixel_force_tol, 
               lbounds="auto",
                options=dict(
                    ftol=0,
                    gtol=pixel_force_tol,
                    maxcor=3,
                    maxiter=1000,
                    )
                         )
assert sol.success

# %%
fig,ax=plt.subplots(figsize=(10, 10))
color = "red"
cmap = LinearSegmentedColormap.from_list('contactcmap', ((1,1,1,0.), color), N=256)
ax.matshow(system.contact_zone.T, cmap = cmap)
cmap = LinearSegmentedColormap.from_list('contactcmap', ((1,1,1,0.), "black"), N=256)
ax.matshow(nonadhesive_system.contact_zone.T, cmap = cmap)

ax.grid(False)

# %%
ax.set_xlim(0, 128)
ax.set_ylim(0, 128)
fig

# %% [markdown]
# Gaps

# %%
fig,ax=plt.subplots()
plt.colorbar(ax.matshow(system.gap[:128,:128].T / interaction.rho), label=r"$g/\rho$")
ax.set_xlim(0, 128)
ax.set_ylim(0, 128)

ax.grid(False)

# %% [markdown]
# ### How adhesion affected the mean pressure (in MPa)

# %%
system.compute_normal_force() / np.prod(t.physical_sizes) / 1000

# %%
nonadhesive_system.compute_normal_force() / np.prod(t.physical_sizes) / 1000

# %% [markdown]
# ## Further examples

# %% [markdown]
#
# - [../test/test_bugnicourt_primal_rough_weakly_adhesive.py]()

# %%
