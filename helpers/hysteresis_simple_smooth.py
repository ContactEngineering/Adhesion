#
# Copyright 2018, 2020 Antoine Sanner
#           2018, 2020 Lars Pastewka
#           2015-2016 Till Junge
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
simpre sphere-on-flat contact simulation to measure pull-off hysteresis
"""

import numpy as np
import matplotlib.pyplot as plt

from Adhesion.System import make_system
from SurfaceTopography import make_sphere
from Adhesion.Interactions import VDW82SimpleSmooth as VdwPot
from ContactMechanics import FreeFFTElasticHalfSpace as Substrate


plt.ion()

E_silicon = 166e9
nu_silicon = .17
E_diamond = 1220e9
nu_diamond = .2
young = 1./((1-nu_silicon**2)/E_silicon+(1-nu_diamond**2)/E_diamond)

radius = 18e-9
base_size = 2*radius
size = (base_size, base_size)

c_sr = 2.1e-78*1e-6
hamaker = 68.1e-21
r_cut = 5e-10
pot = VdwPot(c_sr, hamaker).apply_cutoff(r_cut)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

for base_res in (32, 64, 128, 256, 512):
    res = (base_res, base_res)

    substrate = Substrate(res, young, size)

    surface = make_sphere(radius, res, size, standoff=float('inf'))
    system = make_system(substrate, pot, surface)
    offset = pot.cutoff_radius * .4
    step = pot.r_min*.01
    pullof_forces = list()
    offsets = list()
    contact_area = list()
    disp = np.zeros(substrate.nb_domain_grid_pts)
    force = -1.

    def iterator(initial_offset):
        loc_offset = float(initial_offset)
        yield loc_offset
        while force <= 0.:
            loc_offset -= step
            yield loc_offset
        for i in range(3):
            loc_offset += step
            yield loc_offset
        #while force < 0.:
        while offset < pot.cutoff_radius*.4:
            loc_offset += step
            yield loc_offset

    ax1.set_ylabel("normal force")
    ax2.set_ylabel("contact area", color='r')
    line_force, = ax1.plot(offsets, pullof_forces,
                           label="nb_grid_pts = {}".format(res))
    line_area, = ax2.plot(offsets, contact_area, color=line_force.get_color(),
                          linestyle='--')
    ax1.legend(loc='center right')
    marker, = ax1.plot((), (), marker='+', color='b', markersize=20)

    for offset in iterator(offset):
        result = system.minimize_proxy(offset, disp, deproxify_everytime=False)
        disp = system.disp
        force = system.babushka.compute_normal_force()
        contact_area.append(system.babushka.compute_contact_area())
        pullof_forces.append(force)
        offsets.append(offset)
        line_force.set_xdata(offsets)
        line_force.set_ydata(pullof_forces)
        marker.set_ydata((force,))
        marker.set_xdata((offset,))
        line_area.set_xdata(offsets)
        line_area.set_ydata(contact_area)
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        fig.canvas.draw()

    marker.set_ydata(())
    marker.set_xdata(())
    fig.savefig("fig_{:0>5}.png".format(res[0]))



plt.ioff()
plt.show()
