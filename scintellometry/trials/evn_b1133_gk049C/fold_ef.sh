#!/bin/bash
#PBS -l nodes=32:ppn=8,walltime=0:45:00
#PBS -N gk049c_Ar_fold_all_scans_chans
#PBS -m abe

nodes=32
PPN=8
let NP=$nodes*$PPN
OMP=1 # number of threads per node (4 available)

module purge
module load gcc/4.8.1 binutils 
module load intel/16.0.3 python/2.7.8
module load  intelmpi/5.0.3.048 hdf5/187-v18-intelmpi-intel
module load fftw/3.3.4-intel-impi

# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
# here we assume we submitted from /scratch2/p/pen/franzk/data/gk049/c-1133
# (this is where the fold* scripts used to live..., now they're in 
# scintellometry/trials/evn_b1133_gk049C on branch 1133-trials )
cd $PBS_O_WORKDIR

# directory that contains reduce_data.py and observations.conf (actually use observations.conf in $PBS_O_WORKDIR)
export run_from=/home/p/pen/franzk/git/scintellometry/scintellometry/trials/evn_b1133_gk049C/

# got a funny error regarding python eggs, maybe because we cannot write to $HOME from compute nodes?
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

#to fold EF -- doesn't work. I get
#File "/home/p/pen/franzk/git/scintellometry/scintellometry/io/vlbi_helpers.py", line 21, i
#n <lambda>
#return lambda x: (x[word_index] >> bit_index) & mask
#IndexError: tuple index out of range
mpirun -np ${NP} /scinet/gpc/tools/Python/Python278-shared-intel/bin/python ${run_from}/reduce_data.py --telescope ef -d 2017-03-04T04:02:10 --duration 64 --ntbin 8 -v -v -v -v --dedisperse incoherent --conf ${run_from}/observations.conf --nchan 128

echo "ENDED"
date

