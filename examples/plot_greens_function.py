#
# Copyright 2019 Lars Pastewka
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

import numpy as np

import matplotlib.pyplot as plt

from PyCo.SolidMechanics.GreensFunctions import AnisotropicGreensFunction, IsotropicGreensFunction

###

C11 = 1
C44 = 0.3
C12 = C11 - 2 * C44 + 0.01  # 0.3
aniso = AnisotropicGreensFunction(C11, C12, C44)

nu = 0.3
E = 2 * C44 * (1 + nu)
iso = IsotropicGreensFunction(C44, nu)

special_points = [('$\Gamma$', (0., 0.)), ('$M$', (0.5, 0.5)), ('$X$', (0.5, 0.)), ('$\Gamma$', (0., 0.))]
colors = ['r', 'g', 'b', 'k']
labels = ['isotropic', 'anisotropic']

npts = 100
z = 0.0
for (sym1, (qx1, qy1)), (sym2, (qx2, qy2)) in zip(special_points[:-1], special_points[1:]):
    print(sym1, sym2)
    qx = 2 * np.pi * np.linspace(qx1, qx2, npts)
    qy = 2 * np.pi * np.linspace(qy1, qy2, npts)
    for color, label, gf in zip(colors, labels, [iso, aniso]):
        phi = gf.stiffness(qx, qy)
        phizz = np.real(phi[:, 2, 2])
        plt.plot(np.linspace(z, z + 1.0, npts), phizz, color + '-', label=label)
    z += 1.0
    labels = [None] * len(colors)

plt.xticks(np.arange(len(special_points)), [name for name, pts in special_points])
plt.gca().xaxis.grid(True)
plt.ylabel('Stiffness, $a_0 \Phi_{zz}/E$')
plt.xlabel('Wavevector, $a_0 q$')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('sc100.pdf')
plt.show()
