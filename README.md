Adhesion
====

*Contact mechanics with adhesion with Python.* This code implements computation of contact geometry and pressure of a rigid object on a flat elastic half-space. All calculations assume small deformations; in that limit, the contact of any two objects of arbitrary geometry and elastic moduli can be mapped on that of a rigid on an elastic flat.

The methods that are implemented in this code are described in various papers:

- Fast-Fourier transform (FFT) for the computation of elastic deformation of periodic substrates.
    - [Stanley, Kato, J. Tribol. 119, 481 (1997)](https://doi.org/10.1115/1.2833523)
    - [Campana, Müser, Phys. Rev. B 74, 075420 (2006)](https://doi.org/10.1103/PhysRevB.74.075420)
    - [Pastewka, Sharp, Robbins, Phys. Rev. B 86, 075459 (2012)](https://doi.org/10.1103/PhysRevB.86.075459)
- Decoupling of images for non-periodic calculation with the FFT.
    - Hockney, Methods Comput. Phys. 9, 135 (1970)
    - [Pastewka, Robbins, Appl. Phys. Lett. 108, 221601 (2016)](https://doi.org/10.1063/1.4950802)
- Fast solution of nonadhesive, hard-wall interactions.
    - [Polonsky, Keer, Wear 231, 206 (1999)](https://doi.org/10.1016/S0043-1648(99)00113-1)
- Adhesive interactions.
    - [Pastewka, Robbins, PNAS 111, 3298 (2014)](https://doi.org/10.1073/pnas.1320846111)
- Contact plasticity.
    - [Weber, Suhina, Junge, Pastewka, Brouwer, Bonn, Nature Comm. 9, 888 (2018)](https://doi.org/10.1038/s41467-018-02981-y)

*Analyzing surface topography data with Python.* This code also contains a rich set of import filters for experimental surface topography data. Surface topographies can be easily analyzed using standard (rms height, power spectrum, ...) and some special purpose (autocorrelation function, variable bandwidth analysis, ...) statistical techniques.

Build status
------------

The following badge should say _build passing_. This means that all automated tests completed successfully for the master branch.

[![Build Status](https://travis-ci.com/ComputationalMechanics/Adhesion.svg?token=qEEcjJuzdytivTUsSm9h&branch=master)](https://travis-ci.com/ComputationalMechanics/Adhesion)

Installation
------------

You need Python 3 and [FFTW3](http://www.fftw.org/) to run Adhesion. All Python dependencies can be installed automatically by invoking

#### Installation directly with pip

```bash
# dependencies not installable with requirements.txt
pip install [--user] numpy
pip install [--user] pylint
pip install [--user] cython
pip install [--user] mpi4py #optional

# install pyco
pip  install [--user]  git+https://github.com/pastewka/Adhesion.git#egg=Adhesion
```

The last command will install other dependencies including 
[muFFT](https://gitlab.com/muspectre/muspectre.git), 
[NuMPI](https://github.com/IMTEK-Simulation/NuMPI.git) and [a fork of runtests](https://github.com/AntoineSIMTEK/runtests.git)

Note: sometimes [muFFT](https://gitlab.com/muspectre/muspectre.git) will not find the FFTW3 installation you expect.
You can specify the directory where you installed [FFTW3](http://www.fftw.org/)  
by setting the environment variable `FFTWDIR` (e.g. to `$USER/.local`) 

#### Installation from source directory 

If you cloned the repository. You can install the dependencies with

```
pip install [--user] numpy
pip install [--user] pylint
pip install [--user] cython
pip install [--user] mpi4py #optional
pip3 install [--user] -r requirements.txt
```

in the source directory. Adhesion can be installed by invoking

```pip3 install [--user] .```

in the source directoy. The command line parameter --user is optional and leads to a local installation in the current user's `$HOME/.local` directory.

#### Installation problems with lapack and openblas

`bicubic.cpp` is linked with `lapack`, that is already available as a dependency of `numpy`. 

If during build, `setup.py` fails to link to one of the lapack implementations 
provided by numpy, as was experienced for mac, try providing following environment variables: 

```bash
export LDFLAGS="-L/usr/local/opt/openblas/lib $LDFLAGS"
export CPPFLAGS="-I/usr/local/opt/openblas/include $CPPFLAGS"
export PKG_CONFIG_PATH="/usr/local/opt/openblas/lib/pkgconfig:$PKG_CONFIG_PATH"

export LDFLAGS="-L/usr/local/opt/lapack/lib $LDFLAGS"
export CPPFLAGS="-I/usr/local/opt/lapack/include $CPPFLAGS"
export PKG_CONFIG_PATH="/usr/local/opt/lapack/lib/pkgconfig:$PKG_CONFIG_PATH"
```    
where the paths have probably to be adapted to your particular installation method
(here it was an extra homebrew installation).

Updating Adhesion
------------- 

If you update Adhesion (whether with pip or `git pull` if you cloned the repository), 
you may need to uninstall `NuMPI`, `muSpectre` and or `runtests`, so that the 
newest version of them will be installed.

Testing
-------

To run the automated tests, go to the main source directory and execute the following:

```
pytest
```

Tests that are parallelizable have to run with [runtests](https://github.com/AntoineSIMTEK/runtests)
```
python run-tests.py 
``` 

You can choose the number of processors with the option `--mpirun="mpirun -np 4"`. For development purposes you can go beyond the number of processors of your computer using `--mpirun="mpirun -np 10 --oversubscribe"`

Other usefull flags:
- `--xterm`: one window per processor
- `--xterm --pdb`: debugging

Development
-----------

To use the code without installing it, e.g. for development purposes, use the `env.sh` script to set the environment:

```source /path/to/Adhesion/env.sh [python3]```

Note that the parameter to `env.sh` specifies the Python interpreter for which the environment is set up. Adhesion contains portions that need to be compiled, make sure to run

```python setup.py build```

whenever any of the Cython (.pyx) sources are modified.

Please read [CONTRIBUTING](CONTRIBUTING.md) if you plan to contribute to this code.

Usage
-----

The code is documented via Python's documentation strings that can be accesses via the `help` command or by appending a questions mark `?` in ipython/jupyter. There are two command line tools available that may be a good starting point. They are in the `commandline` subdirectory:

- `hard_wall.py`: Command line front end for calculations with hard, impenetrable walls between rigid and elastic flat. This front end exclusively uses Polonsky & Keer's constrained conjugate gradient solver to find the deformation of the substrate under the additional contact constraints. Run `hard_wall.py --help` to get a list of command line options.
- `soft_wall.py`: Command line front end for calculations with soft (possibly adhesive) interactions between rigid and elastic flat. This is a stub rather than a fully featured command line tool that can be used as a starting point for modified script. The present implementation is set up for a solution of Martin Müser's contact mechanics challenge.

Compiling the documentation
---------------------------

- After changes to the Adhesion source, you have to build again: ```python setup.py build```
- Navigate into the docs folder: ```cd docs/``` 
- Automatically generate reStructuredText files from the source: ```sphinx-apidoc -o source/ ../Adhesion``` 
Do just once, or if you have added/removed classes or methods. In case of the latter, be sure to remove the previous source before: ```rm -rf source/```
- Build html files: ```make html```
- The resulting html files can be found in the ```Adhesion/docs/_build/html/``` folder. Root is ```Adhesion/docs/_build/html/index.html```.
