#!/bin/bash
#PBS -l nodes=8:ppn=8,walltime=0:20:00
#PBS -N gk049c_YY_fold_100s_allchans
#PBS -m abe

NP=64
OMP=4 # number of threads per node (4 available)

module purge
module load gcc/4.8.1 binutils 
module load intel/16.0.3 python/2.7.8
module load  intelmpi/5.0.3.048 hdf5/187-v18-intelmpi-intel
module load fftw/3.3.4-intel-impi

# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
cd $PBS_O_WORKDIR

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

times=(2017-03-04T04:02:10)
duration=(100)
ntimebins=(10)
channels=(0,1 2,3 4,5 6,7)
dirs=(chan0-1 chan2-3 chan4-5 chan6-7)
conf_file=${PBS_O_WORKDIR}/observations.conf

#loop through all channels and times.
ccount=0
previous=0
for channel in ${channels[@]};do
    if [[ ${ccount} -gt 0 ]];then 
	let previous=${ccount}-1
    fi
    old_channel=${channels[${previous}]}
    echo "changing channels from ${old_channel} to ${channel} in ${conf_file}"
    sed -i -e "s;channels = ${old_channel};channels = ${channel};g" ${conf_file}
    echo "mkdir -p ${PBS_O_WORKDIR}/ar/${dirs[${ccount}]}"
    mkdir -p ${PBS_O_WORKDIR}/yy/${dirs[${ccount}]}
    cd ${PBS_O_WORKDIR}/yy/${dirs[${ccount}]}
    echo "Working in `pwd`"
    let ccount=${ccount}+1
    tcount=0
    for t in ${times[@]};do
	dur=${duration[${tcount}]}
	nt=${ntimebins[${tcount}]}
	let tcount=${tcount}+1
	mpirun -np ${NP} /scinet/gpc/tools/Python/Python278-shared-intel/bin/python ${run_from}/reduce_data.py --telescope yy -d ${t} --duration ${dur} --ntbin ${nt} --dedisperse incoherent --nchan 64 --conf ${conf_file}
    done
done
# put channel names back to 0,1 in conf file
last_chan=6,7
first_chan=0,1
sed -i -e "s;channels = ${last_chan};channels = ${first_chan};g" ${conf_file}

#to fold WB -- doesn't work. Get startime wrong (should be 2017, reports 2014)
# the folded spectra also zero all over, probably because no polycos can be found for time
#time mpirun -np ${NP} /scinet/gpc/tools/Python/Python278-shared-intel/bin/python ${run_from}/reduce_data.py --telescope wb -d 2017-03-04T04:02:10 --duration 64 --ntbin 8 -v -v -v -v --dedisperse incoherent --conf ${run_from}/observations.conf --nchan 4

#to fold EF -- doesn't work. I get
#File "/home/p/pen/franzk/git/scintellometry/scintellometry/io/vlbi_helpers.py", line 21, i
#n <lambda>
#return lambda x: (x[word_index] >> bit_index) & mask
#IndexError: tuple index out of range
#mpirun -np ${NP} /scinet/gpc/tools/Python/Python278-shared-intel/bin/python ${run_from}/reduce_data.py --telescope ef -d 2017-03-04T04:02:10 --duration 64 --ntbin 8 -v -v -v -v --dedisperse incoherent --conf ${run_from}/observations.conf --nchan 128

echo "ENDED"
date

