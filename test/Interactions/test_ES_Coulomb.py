#
# Copyright 2023 sitangshugk95@meen-flkmpr3.engr.tamu.edu
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
Created on Wed Feb 22 22:52:48 2023

@author: sitangshugk95
"""
import numpy as np
import matplotlib.pyplot as plt


eps0 = 8.85e-12
eps1 = 3.9
epsg = 1
eps2 = 3000
d1 = 1e-6
d2 = 250e-6
voltage = 100

h0 = (d1 / eps1) + (d2 / eps2)

r = np.linspace(0*h0,5*h0,1001)

dV = 0.5 * eps0 * (voltage**2) * (1 / (r/epsg + h0))**2
sig0 = 0.5 * eps0 * (voltage**2) * (1/h0**2)

x = r/h0
y = dV/sig0

plt.plot(x,y)

indices = [0,200, 400, 600, 800, 1000]; 
y_test = [] ##values calculated based on given input

for i in indices:
    y_test.append(y[i])
    
y_true = [1.0, 0.25, 0.12, 0.625, 0.04, 0.028]; ##true values from paper

error = (np.square(np.array(y_true) - np.array(y_test))).mean(axis=0)


assert error < 0.5, "error should be lower!"