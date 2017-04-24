#!/bin/bash
#PBS -l nodes=32:ppn=8,walltime=1:40:00
#PBS -N fold_2048_secs
#PBS -m abe

NP=256
OMP=4 # number of threads per node (4 available)

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
export I_MPI_PIN_DOMAIN=omp

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

