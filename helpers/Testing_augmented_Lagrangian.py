#
# Copyright 2020 Antoine Sanner
#           2020 Lars Pastewka
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

# coding: utf-8

## Testing the Augmented Lagrangian of Adhesion

# The implementation of the augmented Lagrangian in Tools follows closely the description of the `LANCELOT` algorithm described in Bierlaire (2006)

# The function `augmented_lagrangian` has the form of custom minimizer for [scipy.optimize.minimize](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html)

# In[4]:

import sys
import os
import numpy as np

import scipy.optimize
sys.path.append(os.path.join(os.getcwd(), "../PyCo/Tools/"))
from AugmentedLagrangian import augmented_lagrangian


### Book example

# Example 20.5: Minimise the fuction $f(x)$
# $$\min_{x\in\mathbb{R}^2} 2(x_1^2+x_2^2 -1)-x_1$$
# under the constraint
# $$ x_1^2 + x_2^2 = 1$$

# ugly workaround to get a fresh AugmentedLagrangian without module loads

# In[9]:

# fname = "../PyCo/Tools/AugmentedLagrangian.py"
# with open(fname) as filehandle:
#     content = ''.join((line for line in filehandle))
# exec(content)


# In[11]:

def fun(x):
    return (x[0]**2 + x[1]**2 - 1) - x[0]
def constraint(x):
    return x[0]**2 + x[1]**2 - 1
tol = 1.e-2
result = scipy.optimize.minimize(fun, x0=np.array((-1, .1)),
       	                         constraints={'type':'eq','fun':constraint},
	                         method=augmented_lagrangian, tol=tol,
	                         options={'multiplier0': np.array((0.)),
                                          'disp': True,
                                          'store_iterates': 'iterate'})

print(result)

