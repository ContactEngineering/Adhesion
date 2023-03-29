#
# Copyright 2018, 2020 Antoine Sanner
#           2019-2020 Lars Pastewka
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

represents the UseCase of creating System with MPI parallelization

"""

import pytest

from NuMPI import MPI
from ContactMechanics import FreeFFTElasticHalfSpace
from Adhesion.System import make_system
from Adhesion.Interactions import VDW82, Exponential
from SurfaceTopography import make_sphere
from Adhesion.System import SmoothContactSystem, BoundedSmoothContactSystem
from ContactMechanics.Tools.Logger import Logger
from Adhesion.Interactions import HardWall

import numpy as np

import os

DATADIR = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def examplefile(comm):
    fn = DATADIR + "/worflowtest.npy"
    res = (128, 64)
    np.random.seed(1)
    data = np.random.random(res)
    data -= np.mean(data)
    if comm.rank == 0:
        np.save(fn, data)

    comm.barrier()
    return (fn, res, data)


# DATAFILE = DATADIR + "/worflowtest.npy"
# @pytest.fixture
# def data(comm):
#    res = (256,256)#(128, 64)
#    np.random.seed(1)
#    data = np.random.random(res)
#    data -= np.mean(data)
#    if comm.Get_rank() == 0:
#        np.save(DATAFILE, data)
#    comm.barrier() # all processors wait on the file to be created
#    return data

@pytest.mark.skip("not supported for now")
def test_make_system_from_file(examplefile, comm):
    """
    longtermgoal for confortable and secure use
    Returns
    -------

    """
    # TODO: test this on npy and nc file
    # Maybe it will be another Function or class
    fn, res, data = examplefile

    interaction = HardWall()

    system = make_system(substrate="periodic",
                         interaction=interaction,
                         surface=fn,
                         communicator=comm,
                         physical_sizes=(20., 30.),
                         young=1)

    print(system.__class__)


def test_make_system_from_file_serial(comm_self):
    """
    same as test_make_system_from_file but with the reader being not MPI
    compatible
    Returns
    -------

    """
    pass


# def test_automake_substrate(comm):
#    surface = make_sphere(2, (4,4), (1., 1.), )

def test_smoothcontactsystem_no_minimize_proxy(examplefile, comm):
    """
    asserts the smoothcontactsystem doesn't allow the use of minimize_proxy for
    distributed computations
    """
    fn, res, data = examplefile

    interaction = VDW82(1., 1., communicator=comm)

    if comm.size == 1:
        system = make_system(substrate="periodic",
                             interaction=interaction,
                             surface=fn,
                             communicator=comm,
                             physical_sizes=(20., 30.),
                             young=1, system_class=SmoothContactSystem)
    else:

        with pytest.raises(ValueError):
            system = make_system(substrate="periodic",
                                 interaction=interaction,
                                 surface=fn,
                                 communicator=comm,
                                 physical_sizes=(20., 30.),
                                 young=1, system_class=SmoothContactSystem)
            print(system.surface.is_domain_decomposed)
            system.minimize_proxy()


@pytest.mark.skip("automatic choice of systemclass not supported for now")
def test_make_free_system(examplefile, comm):
    """
    For number of processors > 1 it SmartSmoothContactSystem
    doesn't work.
    """
    fn, res, data = examplefile

    interaction = VDW82(1., 1., communicator=comm)
    system = make_system(substrate="free",
                         interaction=interaction,
                         surface=fn,
                         communicator=comm,
                         physical_sizes=(20., 30.),
                         young=1)

    if comm.size == 1:
        assert system.__class__.__name__ == "FastSmoothContactSystem"
    else:
        assert system.__class__.__name__ == "SmoothContactSystem"


def test_choose_smoothcontactsystem(comm, examplefile):
    """
    even on one processor, one should be able to force the usage of the
    smooth contact system.
    The occurence of jump instabilities make the babushka
    system difficult to use.

    """
    fn, res, data = examplefile

    interaction = VDW82(1., 1., communicator=comm)
    system = make_system(substrate="free",
                         interaction=interaction,
                         surface=fn,
                         communicator=comm,
                         physical_sizes=(20., 30.),
                         young=1,
                         system_class=SmoothContactSystem)

    assert system.__class__.__name__ == "SmoothContactSystem"


@pytest.mark.skip("automatic choice of systemclass not supported for now")
def test_incompatible_system_prescribed(comm_self, examplefile):
    fn, res, data = examplefile
    from ContactMechanics.Systems import IncompatibleFormulationError
    with pytest.raises(IncompatibleFormulationError):
        system = make_system(substrate="free",  # noqa: F841
                             interaction="hardwall",
                             surface=fn,
                             communicator=comm_self,
                             physical_sizes=(20., 30.),
                             young=1,
                             system_class=SmoothContactSystem)


def test_hardwall_as_string(comm, examplefile):
    fn, res, data = examplefile
    make_system(substrate="periodic",
                interaction="hardwall",
                surface=fn,
                physical_sizes=(1., 1.),
                young=1,
                communicator=comm)


def test_bounded_logger():
    """
    This should test the following:
    - that that the logger works in an MPI application
       (on the example of Softwall) # TODO: this not tested actually
    - Test that the reductions are well done). To do that we compare the
    quantities computed at the step with the

    The simplest way to check the independence on number of processors is
    to store a reference computation somewhere. Looking at the
    """
    nx, ny = 16, 16
    sx, sy = 20., 20.
    R = 11.

    surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
    Es = 50.
    substrate = FreeFFTElasticHalfSpace((nx, ny), young=Es,
                                        physical_sizes=(sx, sy),
                                        fft="serial",
                                        communicator=MPI.COMM_SELF)

    interaction = Exponential(0., 0.0001)
    system = BoundedSmoothContactSystem(substrate, interaction, surface)

    gtol = 1e-5
    offset = 1.
    res = system.minimize_proxy(offset=offset, lbounds="auto",  # noqa: F841
                                options=dict(gtol=gtol, ftol=0),
                                logger=Logger("test_logger.log"))

    MPI.COMM_WORLD.barrier()  # Synchronize before reading log file

    log = np.loadtxt("test_logger.log")

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.colorbar(ax.pcolormesh(- system.substrate.force), label="pressure")

        fig, ax = plt.subplots()
        ax.plot(log[:, 1])
        ax.set_xlabel("# objective eval")
        ax.set_ylabel("max abs proj grad")

        ax.set_yscale("log")

        plt.show(block=True)

    assert log[-1, 1] < gtol


def test_smooth_logger():
    """
    This should test the following:
    - that that the logger works in an MPI application
       (on the example of Softwall) # TODO: this not tested actually
    - Test that the reductions are well done). To do that we compare the
    quantities computed at the step with the

    The simplest way to check the independence on number of processors is
    to store a reference computation somewhere. Looking at the
    """
    nx, ny = 16, 16
    sx, sy = 20., 20.
    R = 11.

    surface = make_sphere(R, (nx, ny), (sx, sy), kind="paraboloid")
    Es = 50.
    substrate = FreeFFTElasticHalfSpace((nx, ny), young=Es,
                                        physical_sizes=(sx, sy),
                                        fft="serial",
                                        communicator=MPI.COMM_SELF)

    interaction = Exponential(-100, 0.1)
    system = SmoothContactSystem(substrate, interaction, surface)

    gtol = 1e-5
    offset = 1.
    res = system.minimize_proxy(offset=offset,  # noqa: F841
                                options=dict(gtol=gtol, ftol=0),
                                logger=Logger("test_logger.log"))

    MPI.COMM_WORLD.barrier()  # Synchronize before reading log file

    log = np.loadtxt("test_logger.log")

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.colorbar(ax.pcolormesh(- system.substrate.force), label="pressure")

        fig, ax = plt.subplots()
        ax.plot(log[:, 1])
        ax.set_xlabel("# objective eval")
        ax.set_ylabel("max abs proj grad")

        ax.set_yscale("log")

        plt.show(block=True)

    assert log[-1, 1] < gtol
