#
# Copyright 2018, 2020 Antoine Sanner
#           2016-2017, 2020 Lars Pastewka
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
Tests for PyCo ReferenceSolutions
"""

import numpy as np

import Adhesion.ReferenceSolutions.MaugisDugdale as MD


def test_md_dmt_limit():
    A = np.linspace(0.001, 10, 11)
    N, d = MD._load_and_displacement(A, 1e-12)
    np.testing.assert_allclose(N, A ** 3 - 2)
    np.testing.assert_allclose(d, A ** 2, atol=1e-5)


def test_md_jkr_limit():
    A = np.linspace(0.001, 10, 11)
    N, d = MD._load_and_displacement(A, 1e3)
    np.testing.assert_allclose(N, A ** 3 - A * np.sqrt(6 * A), atol=1e-4)
    np.testing.assert_allclose(d, A ** 2 - 2 / 3 * np.sqrt(6 * A), atol=1e-4)
