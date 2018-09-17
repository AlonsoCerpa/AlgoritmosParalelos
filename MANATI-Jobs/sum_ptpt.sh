#!/bin/bash
#PBS -N sum_ptpt
#PBS -l nodes=2:ppn=3
NPROC=wc -l < $PBS_NODEFILE
source /opt/shared/openmpi-2.0.1/environment.sh
cd $PBS_O_WORKDIR
mpiexec -np $NPROC sum_ptpt.out