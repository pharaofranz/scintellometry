#!/bin/sh
# @ job_name           = FOLD_AROCHIME
# @ job_type           = bluegene
# @ comment            = "by-channel AROCHIME"
# @ error              = $(job_name).$(Host).$(jobid).err
# @ output             = $(job_name).$(Host).$(jobid).out
# @ bg_size            = 64
# @ wall_clock_limit   = 02:00:00
# @ bg_connectivity    = Torus
# @ queue
# Launch all BGQ jobs using runjob   
#PBS -l nodes=10:ppn=8,walltime=0:40:00
#PBS -N fold_32_secs
#PBS -m abe

## Note that the total number of mpi processes in a runjob (i.e., the --np argument) should be the ranks-per-node times the number of nodes (set by bg_size in the loadleveler script). So for the same number of nodes, if you change ranks-per-node by a factor of two, you should also multiply the total number of mpi processes by two.
## One would therefore ideally use 64 / $OMP = 16 ranks per node
NP=256
OMP=4 # number of threads per node (4 available)
RPN=4 # RPN = 8 does not give memory errors for LOFAR, but 16 seems problematic
# load modules (must match modules used for compilation)
module purge
module unload mpich2/xl
#module load   python/2.7.3         binutils/2.23      bgqgcc/4.8.1       mpich2/gcc-4.8.1 fftw/3.3.3-gcc4.8.1 
module load xlf/14.1 essl/5.1
module load vacpp
#module load hdf5/189-v18-mpich2-xlc
module load binutils/2.23 bgqgcc/4.8.1 mpich2/gcc-4.8.1 hdf5/1814-v18-mpich2-gcc python/2.7.3 
module load fftw/3.3.4-gcc4.8.1 
 
# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
cd /scratch/p/pen/franzk/data/aro-1133-14june2014 #$PBS_O_WORKDIR

# got a funny error regarding python eggs, maybe cannot write to $HOME from compute nodes?
export PYTHON_EGG_CACHE=/scratch/p/pen/franzk/

# PIN THE MPI DOMAINS ACCORDING TO OMP
export I_MPI_PIN_DOMAIN=omp

# EXECUTION COMMAND; -np = nodes*ppn
echo "----------------------"
echo "STARTING in directory $PWD"
date
echo "np ${NP}, rpn ${RPN}, omp ${OMP}"
# EXECUTION COMMAND; -np = nodes*processes_per_nodes; --byhost forces a round robin of nodes.
# time runjob --np ${NP} --ranks-per-node=${RPN} --envs OMP_NUM_THREADS=${OMP} HOME=$HOME LD_LIBRARY_PATH=/scinet/bgq/Libraries/HDF5-1.8.12/mpich2-gcc4.8.1//lib:/scinet/bgq/Libraries/fftw-3.3.3-gcc4.8.1/lib:/scinet/bgq/tools/Python/python2.7.3-20131205/lib:$LD_LIBRARY_PATH PYTHONPATH=/scinet/bgq/tools/Python/python2.7.3-20131205/lib/python2.7/site-packages/:/home/m/mhvk/ramain/packages/scintellometry:/home/m/mhvk/mhvk/packages/pulsar : /scinet/bgq/tools/Python/python2.7.3-20131205/bin/python2.7 /home/m/mhvk/ramain/packages/scintellometry/scintellometry/trials/vlbi_j1012/reduce_data.py --telescope lofar -d 2014-01-20T22:46:00 --dedisperse by-channel --ngate 128 --nchan 6945 --ntbin 128 -v
# test for B0809+74

#time runjob --np ${NP} --ranks-per-node=${RPN} --envs OMP_NUM_THREADS=${OMP} HOME=$HOME LD_LIBRARY_PATH=/scinet/bgq/Libraries/HDF5-1.8.12/mpich2-gcc4.8.1//lib:/scinet/bgq/Libraries/fftw-3.3.3-gcc4.8.1/lib:/scinet/bgq/tools/Python/python2.7.3-20131205/lib:$LD_LIBRARY_PATH PYTHONPATH=/scinet/bgq/tools/Python/python2.7.3-20131205/lib/python2.7/site-packages/:/home/m/mhvk/ramain/packages/scintellometry:/home/m/mhvk/mhvk/packages/pulsar : /scinet/bgq/tools/Python/python2.7.3-20131205/bin/python2.7 /home/m/mhvk/ramain/packages/scintellometry/scintellometry/trials/vlbi_b0655/reduce_data.py --telescope arochime -d 2014-06-15T10:57:19 --dedisperse incoherent --ngate 128 --ntbin 5 -v

time runjob --np ${NP} --ranks-per-node=${RPN} --envs OMP_NUM_THREADS=${OMP} HOME=$HOME LD_LIBRARY_PATH=/scinet/bgq/Libraries/HDF5-1.8.12/mpich2-gcc4.8.1//lib:/scinet/bgq/Libraries/fftw-3.3.3-gcc4.8.1/lib:/scinet/bgq/tools/Python/python2.7.3-20131205/lib:$LD_LIBRARY_PATH PYTHONPATH=/scinet/bgq/tools/Python/python2.7.3-20131205/lib/python2.7/site-packages/:/home/p/pen/franzk/git/scintellometry:/home/m/mhvk/mhvk/packages/pulsar:/home/p/pen/franzk/git/baseband : /scinet/bgq/tools/Python/python2.7.3-20131205/bin/python2.7 /home/p/pen/franzk/git/scintellometry/scintellometry/trials/aro_1133_14june2014/reduce_data.py --telescope arochime -d 2014-06-14T17:20:41 --duration 32 --ntbin 4 -v -v -v -v --dedisperse incoherent

echo "ENDED"
date

