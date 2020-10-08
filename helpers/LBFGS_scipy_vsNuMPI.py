#
# Copyright 2019-2020 Antoine Sanner
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


import time

import scipy.optimize


starttime=time.time()
import numpy as np
from ContactMechanics import FreeFFTElasticHalfSpace
from SurfaceTopography import make_sphere

from FFTEngine import PFFTEngine
from NuMPI.Optimization import LBFGS
from NuMPI.Tools.Reduction import Reduction
from Adhesion import VDW82smoothMin
from System import SmoothContactSystem

from NuMPI import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
pnp = Reduction(comm=comm)

class iter_inspector():
    def __init__(self):
        self.neval = 0
        self.energies = []
        self.maxgradients =[]


    def __call__(self, system):

        self.neval += 1
        self.energies.append(system.energy)
        self.maxgradients.append(pnp.max(abs(system.force)))


class decorated_objective:
    def __init__(self, system, objective):
        self.system = system
        self.objective = objective
        self.neval = 0
        self.energies = []
        self.maxgradients =[]

    def __call__(self, *args, **kwargs):
        val = self.objective(*args, **kwargs)
        self.neval += 1
        self.energies.append(system.energy)
        self.maxgradients.append(pnp.max(abs(system.force)))
        return val

import matplotlib.pyplot as plt

fig, (axt, axit) = plt.subplots(2, 1, sharex=True)
ns = [128,256, 512]
nrepetition = 2
for method, name in zip([LBFGS,"L-BFGS-B"],
                        ["NuMPI", "Scipy"]):
    times = np.zeros((len(ns), nrepetition))
    nits = np.zeros((len(ns), nrepetition))
    nevals =np.zeros((len(ns), nrepetition))
    for i, n in enumerate(ns):

        # sphere radius:
        r_s = 10.0
        # contact radius
        r_c = .2
        # peak pressure
        p_0 = 2.5
        # equivalent Young's modulus
        E_s = 102.#102.
        # work of adhesion
        w = 1.0
        # tolerance for optimizer
        tol = 1e-12
        # tolerance for contact area
        gap_tol = 1e-6

        nx, ny = n, n
        sx = 21.0

        z0 = 0.05 # needed to get small tolerance, but very very slow

        fftengine = PFFTEngine((2*nx, 2*ny), comm=comm)


        # the "Min" part of the potential (linear for small z) is needed for the LBFGS without bounds
        inter = VDW82smoothMin(w * z0 ** 8 / 3, 16 * np.pi * w * z0 ** 2, gamma=w, pnp = pnp)

        # Parallel SurfaceTopography Patch

        substrate = FreeFFTElasticHalfSpace((nx,ny), young=E_s, physical_sizes=(sx, sx), fft=fftengine, pnp=pnp)
        #print(substrate._comp_nb_grid_pts)
        #print(fftengine.nb_domain_grid_pts)


        surface = make_sphere(radius=r_s, nb_grid_pts=(nx, ny), physical_sizes=(sx, sx),
                              subdomain_locations=substrate.topography_subdomain_locations,
                              nb_subdomain_grid_pts=substrate.topography_nb_subdomain_grid_pts,
                              pnp=pnp,
                              standoff=float('inf'))
        ext_surface = make_sphere(r_s, (2 * nx, 2 * ny), (2 * sx, 2 * sx),
                                  centre=(sx / 2, sx / 2),
                                  subdomain_locations=substrate.subdomain_locations,
                                  nb_subdomain_grid_pts=substrate.nb_subdomain_grid_pts,
                                  pnp=pnp,
                                  standoff=float('inf'))
        system = SmoothContactSystem(substrate, inter, surface)

        penetration = 0

        disp0 = ext_surface.heights() + penetration
        disp0 = np.where(disp0 > 0, disp0, 0)
        #disp0 = system.shape_minimisation_input(disp0)

        maxcor = 10
        for j in range(nrepetition):
            starttime =time.time()
            counter = iter_inspector()

            objective_monitor = decorated_objective(system, system.objective(penetration, gradient=True))

            result = scipy.optimize.minimize(objective_monitor,
                                    disp0, method=method, jac=True,
                                    options=dict(gtol=1e-6 * abs(w/z0),
                                                          ftol=1e-25,
                                                          maxcor=maxcor))

            nevals[i,j]= objective_monitor.neval
            times[i,j] = time.time() - starttime
            nits[i,j]  = result.nit

            print(method)
            print(result.message)
            print("nevals: {}".format(objective_monitor.neval))
            print(result.nit)
            print(times[i,j])

            converged = result.success
            assert converged

    axt.plot(ns, np.mean(times, axis=1), "o",label="{}".format(name))
    l, =axit.plot(ns, np.mean(nits, axis=1), "o",label="{}, nits".format(name))
    axit.plot(ns, np.mean(nevals, axis=1), "+",c = l.get_color(), label="{}, nfeval".format(name))




axit.set_xlabel("lateral nb_grid_pts (-)")
axt.set_ylabel("execution time (s)")
axit.set_ylabel("# of iterations")
axit.legend(fancybox=True, framealpha=0.5)
axt.legend(fancybox=True, framealpha=0.5)
fig.savefig("LBFGS_scipy_vs_NuMPI.png")


