# ---
# jupyter:
#   jupytext:
#     formats: py:percent

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
# # Adhesive Plastic Simulation

# %%
import numpy as np
import matplotlib.pyplot as plt

from SurfaceTopography import Topography, PlasticTopography
from ContactMechanics import FreeFFTElasticHalfSpace
from Adhesion.Interactions import Lj82
from Adhesion.System import make_system
from Adhesion.System import PlasticSmoothContactSystem
from ContactMechanics.Tools.Logger import screen, Logger

# %% [markdown]
# ## Prepare Geometry

# %%
nx, ny = 128, 128

sx = 250
sy = 250

x = np.arange(0, nx).reshape(-1, 1) * sx / nx - sx / 2
y = np.arange(0, ny).reshape(1, -1) * sy / ny - sy / 2

topography = Topography(- np.sqrt(x ** 2 + y ** 2) * 0.05, physical_sizes=(sx, sy))

# %%
fig, ax = plt.subplots()
plt.colorbar(ax.pcolormesh(x * np.ones_like(y), y * np.ones_like(x), topography.heights()), label="heights")
ax.set_aspect(1)
ax.set_xlabel("x ($z_0$)")
ax.set_ylabel("y ($z_0$)")

# %%
fig, ax = plt.subplots()
ax.plot(x, topography.heights()[:, ny // 2])
ax.set_aspect(1)
ax.set_xlabel("x ($z_0$)")
ax.set_ylabel("heights ($z_0$)")

# %% [markdown]
# ## Material Properties

# %%
Es = 50.
hardness = 2.
z0 = 1.
w = 1.

# %% [markdown]
# ## setup system
#
# the `make_system` automatically knows it has to do a plastic simulation if we pass a `PlasticTopogaraphy`

# %%
system = make_system(
    interaction=Lj82(w, z0),
    substrate=FreeFFTElasticHalfSpace(
        nb_grid_pts=topography.nb_grid_pts, young=Es,
        physical_sizes=topography.physical_sizes,
        check_boundaries=False),
    surface=PlasticTopography(topography=topography, hardness=hardness),
    system_class=PlasticSmoothContactSystem
)

# %%
fig, ax = plt.subplots(3, sharex=True, figsize=(4, 8))

z = np.linspace(0.85 * z0, 2 * z0)
for poti in [Lj82(w, z0), system.interaction]:
    p, f, c = poti.evaluate(z, True, True, True)
    ax[0].plot(z, p)
    ax[1].plot(z, f)
    ax[2].plot(z, c)
    ax[0].set_ylabel("potential")
    ax[1].set_ylabel("force")
    ax[2].set_ylabel("curvature")
for a in ax:
    a.grid()

ax[2].set_xlabel("separation")
fig.subplots_adjust(hspace=0.05)

# %% [markdown]
# #### loading history

# %%
penetrations = np.linspace(-1, 2, 5)
penetrations = np.concatenate((penetrations, penetrations[-2::-1]))

# %% [markdown]
# ### Simulate
#
# Choice of the gradient tolerance: The gradient is the error in the balance 
# of the elastic-force and the adhesive force on each pixel. 
#
# This error should be small compared to this force. 
#
# In plastic and adhesive simulations the pressures are bopund by the hardness or 
# the maximum interaction stress. This value multiplied by the pixel size gives us 
# the order of magnitude of the pixel forces. These forces increase with pixel size, 
# and so do the achievable gradient tolerance. 
#  

# %%
hardness * topography.area_per_pt

# %% pycharm={"name": "#%%\n"}
gtol = 1e-3

# %%
# prepare empty arrays to contain results
offsets = []
plastic_areas = []
normal_forces = []
repulsive_areas = []
forces = np.zeros(
    (len(penetrations), *topography.nb_grid_pts))  # forces[timestep,...]: array of forces for each gridpoint
elastic_displacements = np.zeros((len(penetrations), *topography.nb_grid_pts))
plastified_topographies = []
disp0 = None
i = 0
for penetration in penetrations:
    print(f"penetration = {penetration}")
    sol = system.minimize_proxy(
        initial_displacements=disp0,
        options=dict(gtol=gtol,  # max absolute value of the gradient of the objective for convergence
                     ftol=0,  # stop only if the gradient criterion is fullfilled
                     maxcor=3  # number of gradients stored for hessian approximation
                     ),
        logger=Logger("laststep.log"),
        offset=penetration,
        callback=None,
    )
    assert sol.success, sol.message
    disp0 = u = system.disp  # TODO: the disp0 should only include the elastic displacements. u contains ela
    normal_forces.append(system.compute_normal_force())
    plastic_areas.append(system.surface.plastic_area)
    repulsive_areas.append(system.compute_repulsive_contact_area())

    plastified_topographies.append(system.surface.squeeze())
    # system.surface=PlasticTopography(topography=topography, hardness=hardness) # reset surface
    forces[i, ...] = - system.substrate.evaluate_force(u)[system.substrate.topography_subdomain_slices]
    # this may be wronmg as well because it includes the plastic displacements ? 
    elastic_displacements[i, ...] = system.disp[system.surface.subdomain_slices]
    # doesn't this include the plastic displacements as well now ? 
    # no the plastic displacement is in the extra compliance of the linear part of the potential.

    i += 1
    # print(np.max(system.surface.plastic_displ))
    # print(np.min(system.surface.plastic_displ))

# %% [markdown]
# Bug small steps in the not yet in contact region leads to problems, why ? 

# %%
penetration

# %%
# !open laststep.log

# %% [markdown]
# ## plot pressure distributions and deformed profiles

# %%
for i in range(len(penetrations)):

    fig, (axf, axfcut, axtopcut) = plt.subplots(1, 3, figsize=(14, 3))

    axf.set_xlabel("x (mm)")
    axf.set_ylabel("y (mm)")

    axfcut.plot(system.surface.positions()[0][:, 0], forces[i, :, ny // 2] / system.area_per_pt)
    axfcut.set_xlabel("x")
    axfcut.set_ylabel("pressures MPa")

    for a in (axfcut, axtopcut):
        a.set_xlabel("x (mm)")
    axtopcut.set_ylabel("height (mm)")

    plt.colorbar(axf.pcolormesh(*system.surface.positions(), forces[i, ...] / system.area_per_pt),
                 label="pressures MPa", ax=axf)
    axf.set_aspect(1)

    axtopcut.plot(system.surface.positions()[0][:, 0], topography.heights()[:, ny // 2],
                  color="k", label="original")
    axtopcut.plot(system.surface.positions()[0][:, 0], plastified_topographies[i].heights()[:, ny // 2],
                  color="r", label="plast.")
    axtopcut.plot(system.surface.positions()[0][:, 0],
                  plastified_topographies[i].heights()[:, ny // 2] - elastic_displacements[i, :, ny // 2],
                  c="b", label="plast. el.")
    axtopcut.legend()

    fig.tight_layout()

# %% [markdown]
# ## scalar quantities during loading

# %%
fig, ax = plt.subplots()

ax.plot(penetrations, normal_forces, "+-")
ax.set_xlabel("rigid body penetration [mm]")
ax.set_ylabel("Force [N]")

# %%
fig, ax = plt.subplots()

ax.plot(normal_forces, plastic_areas, "-+")
ax.set_xlabel("Force (N)")
ax.set_ylabel("plastic area (mm^2)")
fig.tight_layout()

# %% pycharm={"name": "#%%\n"}
fig, ax = plt.subplots()

ax.plot(normal_forces, repulsive_areas, "+-")
ax.set_xlabel("indentation force (N)")
ax.set_ylabel("contact area (mm^2)")
fig.tight_layout()

# %% pycharm={"name": "#%%\n"}
