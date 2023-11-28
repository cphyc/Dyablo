#!/bin/bash

# This is meant to be used on AdAstra

#SBATCH -J dyablo_cpu
#SBATCH -N <nb_nodes> -n <nb_mpi> -c <nb_threads>
#SBATCH --hint=multithread
#SBATCH --account=<account>
#SBATCH --constraint=GENOA
#SBATCH -t 1:00:00
#SBATCH --exclusive
#SBATCH -o log.out

module purge
module load cpe/23.05
module load craype-x86-milan
module load PrgEnv-cray
module load cray-hdf5-parallel

export OMP_NUM_THREADS=<nb_threads>
export OMP_PROC_BIND=close
export OMP_PLACES=threads

srun -n <nb_mpi> -c <nb_threads> ./dyablo_genoa restart.ini
