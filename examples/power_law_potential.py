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

# %% pycharm={"name": "#%%\n"}
import matplotlib.pyplot as plt
import numpy as np

from Adhesion.Interactions import PowerLaw, Exponential

# %% pycharm={"name": "#%%\n"}
pot = PowerLaw(0.5, 3 * 0.2, 3)
exp = Exponential(0.5, 0.2) 

fig, (axpot, axf, axcurv) = plt.subplots(3,1)
r = np.linspace(-0.001, 2)

v, dv, ddv = pot.evaluate(r, True, True, True)

axpot.plot(r, v, label="PowerLaw")
axf.plot(r, dv, label="PowerLaw")
axcurv.plot(r, ddv, label="PowerLaw")

v, dv, ddv = exp.evaluate(r, True, True, True)

axpot.plot(r, v, label="exponential")
axf.plot(r, dv, label="exponential")
axcurv.plot(r, ddv, label="exponential")

axpot.set_ylabel("Potential")
axf.set_ylabel("interaction stress")
axcurv.set_ylabel("curvature")

for a in (axpot, axf, axcurv):
    a.grid()

axcurv.set_xlabel("gap")
axpot.legend() 
fig.savefig("PowerLawPotential.png")

# %% pycharm={"name": "#%%\n"}


