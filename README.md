Adhesion
========

*Adhesion.* This code implements adhesive interaction for [ContactMechanics](https://https://github.com/ComputationalMechanics/ContactMechanics).

The methods that are implemented in this code are described in the following papers:

- [Pastewka, Robbins, PNAS 111, 3298 (2014)](https://doi.org/10.1073/pnas.1320846111)

Build status
------------

The following badge should say _build passing_. This means that all automated tests completed successfully for the master branch.

Tests: [![Weekly tests](https://github.com/ContactEngineering/Adhesion/actions/workflows/tests.yml/badge.svg)](https://github.com/ContactEngineering/Adhesion/actions/workflows/tests.yml)

Building documentation: [![CI](https://github.com/ContactEngineering/Adhesion/actions/workflows/publish.yml/badge.svg)](https://github.com/ContactEngineering/Adhesion/actions/workflows/publish.yml)

Installation
------------

You need Python 3 and [FFTW3](http://www.fftw.org/) to run Adhesion. All Python dependencies can be installed automatically by invoking

#### Installation directly with pip

```bash
# install Adhesion
pip  install Adhesion
```

The last command will install other dependencies including 
[muFFT](https://gitlab.com/muspectre/muspectre.git), 
[NuMPI](https://github.com/IMTEK-Simulation/NuMPI.git) and [runtests](https://github.com/bccp/runtests.git)

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

- `soft_wall.py`: Command line front end for calculations with soft (possibly adhesive) interactions between rigid and elastic flat. This is a stub rather than a fully featured command line tool that can be used as a starting point for modified script. The present implementation is set up for a solution of Martin Müser's contact mechanics challenge.

Compiling the documentation
---------------------------

- After changes to the Adhesion source, you have to build again: ```python setup.py build```
- Navigate into the docs folder: ```cd docs/``` 
- Automatically generate reStructuredText files from the source: ```sphinx-apidoc -o source/ ../Adhesion``` 
Do just once, or if you have added/removed classes or methods. In case of the latter, be sure to remove the previous source before: ```rm -rf source/```
- Build html files: ```make html```
- The resulting html files can be found in the ```Adhesion/docs/_build/html/``` folder. Root is ```Adhesion/docs/_build/html/index.html```.

For convenience, all these steps are implemented in `compile_doc.sh`.

Funding
-------

Development of this project is funded by the [European Research Council](https://erc.europa.eu) within [Starting Grant 757343](https://cordis.europa.eu/project/id/757343) and by the [Deutsche Forschungsgemeinschaft](https://www.dfg.de/en) within projects [PA 2023/2](https://gepris.dfg.de/gepris/projekt/258153560) and [EXC 2193](https://gepris.dfg.de/gepris/projekt/390951807).
