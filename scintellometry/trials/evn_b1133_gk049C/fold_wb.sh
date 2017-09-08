#!/bin/bash
#PBS -l nodes=256:ppn=8,walltime=8:45:00
#PBS -N gk049c_Ar_fold_all_scans_chans_Tres5s
#PBS -m abe

nodes=256
PPN=1 # be careful about memory usage
source /home/p/pen/franzk/software/scripts/prep_rank_machine_file.sh $PPN intel
#let NP=$nodes*$PPN
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
mkdir -p $PBS_O_WORKDIR/joboutput/
rm -rf $PBS_O_WORKDIR/joboutput/output.tee

# directory that contains reduce_data.py and observations.conf
export run_from=/home/p/pen/franzk/git/scintellometry/scintellometry/trials/evn_b1133_gk049C/

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

#to fold AR -- works
# times according to scan starttimes in gk049c.sum
# scan 2, ~19min: 2017-03-04T04:02:10
# scan 8,9 ~11min: 2017-03-04T04:30:39 
# scan12,13, ~23min: 2017-03-04T04:44:32
# scan 19,20, ~25min: 2017-03-04T05:12:47
# scan 25,26, ~21min: 2017-03-04T05:41:00
# scan 31,32, ~18min: 2017-03-04T06:11:32


#to fold WB -- doesn't work. Get startime wrong (should be 2017, reports 2014)
# the folded spectra also zero all over, probably because no polycos can be found for time
time mpirun -np ${NP} /scinet/gpc/tools/Python/Python278-shared-intel/bin/python ${run_from}/reduce_data.py --telescope wb -d 2017-03-04T04:02:10 --duration 64 --ntbin 8 -v -v -v -v --dedisperse incoherent --conf ${run_from}/observations.conf --nchan 4

echo "ENDED"
date

