# Singularity images with Adhesion

The process is split into two steps. 

Build an image containing the dependencies

```bash
sudo singularity build dep_serial.sif dep_serial.def
```
From this image you should also be able to run Adhesion without installing it (but don't forget to run `python3 setup.py build` from inside the container) 


Based on this image, you can create an image with Adhesion "pip installed":
```bash
sudo singularity build adhesion_serial.sif adhesion_serial.def
```

Similarly, you can build the Adhesion image with mpi support. 

```bash
sudo singularity build dep_mpi.sif dep_mpi.def
sudo singularity build adhesion_mpi.sif adhesion_mpi.def

```

## Running test 

In the Adhesion main directory, create a file `testjob.sh` with the following content:

```bash
source env.sh
pytest
# only for mpi 
python3 run-tests.py --mpirun="mpirun -np 4 --oversubscribe" --verbose $@
```

run it:
```
singularity exec dep_mpi.sif bash testjob.sh
```



