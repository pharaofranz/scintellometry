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

#times=(2017-03-04T04:02:10)
times=(2017-03-04T04:02:10 2017-03-04T04:30:39 2017-03-04T04:44:32 2017-03-04T04:54:47 2017-03-04T05:12:47 2017-03-04T05:23:02 2017-03-04T05:41:00 2017-03-04T05:51:15 2017-03-04T06:11:32 2017-03-04T06:21:47)
#times=(scan2 scan8 scan12 scan13 scan19 scan20 scan25 scan26 scan31 scan32)
duration=(1100 560 560 760 400 890 560 650 540 480)
ntimebins=(220 112 112 152 80 178 112 130 108 96) #gives 5 second resolution
channels=(0,1 2,3 4,5 6,7)
#channels=(0,1)
dirs=(chan0-1 chan2-3 chan4-5 chan6-7)
conf_file=${run_from}/observations.conf
nchan=8000 # powers of 2 don't work!

#loop through all channels and times.
tcount=0
for t in ${times[@]};do
    dur=${duration[${tcount}]}
    nt=${ntimebins[${tcount}]}
    let tcount=${tcount}+1
    ccount=0
    previous=0
    for channel in ${channels[@]};do
	if [[ ${ccount} -gt 0 ]];then 
	    let previous=${ccount}-1
	fi
	old_channel=${channels[${previous}]}
	echo "changing channels from ${old_channel} to ${channel} in ${conf_file}"
	sed -i -e "s;channels = ${old_channel};channels = ${channel};g" ${conf_file}
	if [[ ${channel} == '0,1' ]] || [[ ${channel} == '4,5' ]];then
	    echo sed -i -e "s;fedge_at_top = False;fedge_at_top = True;g" ${conf_file}
	    sed -i -e "s;fedge_at_top = False;fedge_at_top = True;g" ${conf_file}
	else
	    echo sed -i -e "s;fedge_at_top = True;fedge_at_top = False;g" ${conf_file}
	    sed -i -e "s;fedge_at_top = True;fedge_at_top = False;g" ${conf_file}
	fi
	if [[ ${channel} == '0,1' ]] || [[ ${channel} == '2,3' ]];then
	    echo sed -i -e "s;fedge = 336.25;fedge = 320.25;g" ${conf_file}
	    sed -i -e "s;fedge = 336.25;fedge = 320.25;g" ${conf_file}
	else
	    echo sed -i -e "s;fedge = 320.25;fedge = 336.25;g" ${conf_file}
	    sed -i -e "s;fedge = 320.25;fedge = 336.25;g" ${conf_file}
	fi
	echo "mkdir -p ${PBS_O_WORKDIR}/ar/${dirs[${ccount}]}"
	mkdir -p ${PBS_O_WORKDIR}/ar/${dirs[${ccount}]}
	cd ${PBS_O_WORKDIR}/ar/${dirs[${ccount}]}
	echo "Working in `pwd`"
	let ccount=${ccount}+1
	mpirun -np ${NP} -hostfile ${MACHINES_FILE} -machinefile ${RANKS_FILE}  /scinet/gpc/tools/Python/Python278-shared-intel/bin/python ${run_from}/reduce_data.py --telescope ao -d ${t} --duration ${dur} --ntbin ${nt} --dedisperse incoherent --nchan ${nchan} --conf ${conf_file} 2>&1 | tee --append $PBS_O_WORKDIR/joboutput/output.tee
    done
    # put channel names back to 0,1 in conf file
    last_chan=6,7
    first_chan=0,1
    sed -i -e "s;channels = ${last_chan};channels = ${first_chan};g" ${conf_file}
done
#-ppn ${PPN}

echo "ENDED"
date

