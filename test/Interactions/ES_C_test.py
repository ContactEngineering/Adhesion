#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 22:52:48 2023

@author: sitangshugk95
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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

error = mean_squared_error(y_true, y_test)

assert error < 0.5, "error should be lower!"