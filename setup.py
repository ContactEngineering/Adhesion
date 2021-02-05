#
# Copyright 2018, 2020 Antoine Sanner
#           2016, 2020 Lars Pastewka
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

from setuptools import setup, find_packages

scripts = [
   'commandline/soft_wall.py',
   ]

setup(
    name="Adhesion",
    scripts=scripts,
    packages=find_packages(),
    package_data={'': ['ChangeLog.md']},
    include_package_data=True,
    # metadata for upload to PyPI
    author="Lars Pastewka",
    author_email="lars.pastewka@imtek.uni-freiburg.de",
    description="Efficient contact mechanics with Python",
    license="MIT",
    test_suite='test',
    # dependencies
    python_requires='>=3.5.0',
    use_scm_version=True,
    zip_safe=False,
    setup_requires=[
        'setuptools_scm>=3.5.0'
    ],
    install_requires=[
        'numpy>=1.11.0',
        'NuMPI',
        'muFFT>=0.14.0',
        'SurfaceTopography',
        'ContactMechanics',
    ]
)
