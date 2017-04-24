#!/bin/bash
#PBS -l nodes=32:ppn=8,walltime=1:40:00
#PBS -N fold_2048_secs
#PBS -m abe

## Note that the total number of mpi processes in a runjob (i.e., the --np argument) should be the ranks-per-node times the number of nodes (set by bg_size in the loadleveler script). So for the same number of nodes, if you change ranks-per-node by a factor of two, you should also multiply the total number of mpi processes by two.
## One would therefore ideally use 64 / $OMP = 16 ranks per node
NP=256
OMP=4 # number of threads per node (4 available)
RPN=4 # does not exist in intel mpirunRPN = 8 does not give memory errors for LOFAR, but 16 seems problematic
# load modules (must match modules used for compilation)
#module unload mpich2/xl
#module load   python/2.7.3         binutils/2.23      bgqgcc/4.8.1       mpich2/gcc-4.8.1 fftw/3.3.3-gcc4.8.1 
#module load xlf/14.1 essl/5.1
#module load vacpp
#module load hdf5/189-v18-mpich2-xlc
#module load binutils/2.23 bgqgcc/4.8.1 mpich2/gcc-4.8.1 hdf5/1814-v18-mpich2-gcc python/2.7.3 
#module load fftw/3.3.4-gcc4.8.1 
module purge
module load gcc/4.8.1 binutils 
module load intel/16.0.3 python/2.7.8
module load  intelmpi/5.0.3.048 hdf5/187-v18-intelmpi-intel
module load fftw/3.3.4-intel-impi

# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
cd $PBS_O_WORKDIR

# got a funny error regarding python eggs, maybe cannot write to $HOME from compute nodes?
export PYTHON_EGG_CACHE=/scratch2/p/pen/franzk/

# PIN THE MPI DOMAINS ACCORDING TO OMP
export I_MPI_PIN_DOMAIN=${omp}

# EXECUTION COMMAND; -np = nodes*ppn
echo "----------------------"
echo "STARTING in directory $PWD"
date
echo "np ${NP}, rpn ${RPN}, omp ${OMP}"
# EXECUTION COMMAND; -np = nodes*processes_per_nodes; --byhost forces a round robin of nodes.
export OMP_NUM_THREADS=${OMP}
export PYTHONPATH=/home/p/pen/franzk/git/scintellometry:/home/p/pen/franzk/git/pulsar:/home/p/pen/franzk/git/baseband:${PYTHONPATH}
time mpirun -np ${NP} /scinet/gpc/tools/Python/Python278-shared-intel/bin/python /home/p/pen/franzk/git/scintellometry/scintellometry/trials/aro_1133_14june2014/reduce_data.py --telescope arochime -d 2014-06-14T17:20:41 --duration 2048 --ntbin 128 -v -v -v -v --dedisperse incoherent --conf /home/p/pen/franzk/git/scintellometry/scintellometry/trials/aro_1133_14june2014/observations.conf

echo "ENDED"
date

